import { CharacterSupportedState, isGizmoInteracting, pickAsync } from "babylon-lite";
import type { Mesh, PickOptions } from "babylon-lite";
import { CROUCH_CAPSULE_HEIGHT, CROUCH_CAPSULE_RADIUS } from "../constants.js";
import type { FluidElectricityRegistration, FluidParticleAabb } from "../fluid-runtime.js";
import type { AquanovaGameContext, JumpApertureAssist } from "./game-context.js";
import type { ManagedSound } from "./sound-manager.js";
import type { Behavior, PlayerBehaviorConfig } from "./types.js";

const LOOK_SENSITIVITY = 1 / 600;
const WALK_SPEED = 4;
const FLY_SPEED = 8;
const MOVE_ACCELERATION = 12;
const LOOK_ACCELERATION = 30;
const JUMP_SPEED = 6;
const JUMP_GRAVITY = 16;
const CROUCH_SPEED_FACTOR = 0.75;
const CROUCH_TRANSITION_SECONDS = 0.2;
const JUMP_BUFFER_SECONDS = 0.3;
const APERTURE_EDGE_ADVANCE = 0.2;
const APERTURE_EDGE_ADVANCE_SPEED = 2;
const APERTURE_LATERAL_SPEED = 1;
const APERTURE_ASSIST_SECONDS = 1.5;
const GROUND_ADHESION_SPEED = 2;
const DEFAULT_CHARACTER_STRENGTH = 10_000;
const DEFAULT_MAX_GRAB_DISTANCE = 8;
const DEFAULT_MAX_HELD_OBJECT_DISTANCE = 8;
const DEFAULT_SUBMERGED_PARTICLE_COUNT = 100;
const DEFAULT_ELECTRIFIED_PARTICLE_COUNT = 8;
const SUBMERGED_EXIT_THRESHOLD_FACTOR = 0.5;
const SUBMERGED_QUERY_MAX_Y = 1_000_000;
const WALK_STEP_DISTANCE = 1.8;
const RUN_STEP_DISTANCE = 2.4;
const FOOTSTEP_VOLUME = 2.5;
const FOOTSTEP_SOUND_URL = "/aquanova/sounds/stepMetallic.mp3";
const DOWN = { x: 0, y: -1, z: 0 };

export function playerSupportedMovementVelocity(
    horizontalX: number,
    horizontalZ: number,
    surfaceNormal: Readonly<{ x: number; y: number; z: number }>
): { x: number; y: number; z: number } {
    const length = Math.hypot(surfaceNormal.x, surfaceNormal.y, surfaceNormal.z);
    const normal = length > 1e-6 ? { x: surfaceNormal.x / length, y: surfaceNormal.y / length, z: surfaceNormal.z / length } : { x: 0, y: 1, z: 0 };
    const normalVelocity = horizontalX * normal.x + horizontalZ * normal.z + GROUND_ADHESION_SPEED;
    return {
        x: horizontalX - normal.x * normalVelocity,
        y: -normal.y * normalVelocity,
        z: horizontalZ - normal.z * normalVelocity,
    };
}

export function selectClosestClearApertureOffset(pathClear: (lateralOffset: number) => boolean): number | null {
    for (const lateralOffset of [0, -0.05, 0.05, -0.1, 0.1]) {
        if (pathClear(lateralOffset)) {
            return lateralOffset;
        }
    }
    return null;
}

export function remainingForwardApertureAssist(distance: number, resolvedForwardProgress: number): number {
    return Math.max(0, distance - Math.max(0, resolvedForwardProgress));
}

export function remainingLateralApertureAssist(offset: number, resolvedLateralProgress: number): number {
    if (Math.sign(resolvedLateralProgress) !== Math.sign(offset)) {
        return offset;
    }
    return offset - Math.sign(offset) * Math.min(Math.abs(offset), Math.abs(resolvedLateralProgress));
}

export function shouldUseJumpApertureAssist(jumpActive: boolean, forwardInput: number, crouched: boolean, apertureAssist: JumpApertureAssist | null): boolean {
    return jumpActive && forwardInput > 0 && !crouched && apertureAssist !== null;
}

export function playerWeaponSwayMultiplier(keys: ReadonlySet<string>, frozen = false): 1 | 2 | 4 {
    if (frozen) return 1;
    const moving =
        keys.has("KeyW") ||
        keys.has("KeyS") ||
        keys.has("KeyA") ||
        keys.has("KeyD") ||
        keys.has("ArrowUp") ||
        keys.has("ArrowDown") ||
        keys.has("ArrowLeft") ||
        keys.has("ArrowRight");
    if (!moving) return 1;
    return keys.has("ShiftLeft") || keys.has("ShiftRight") ? 4 : 2;
}

export function weaponWheelDirection(deltaY: number): -1 | 1 | null {
    if (!Number.isFinite(deltaY) || deltaY === 0) {
        return null;
    }
    return deltaY < 0 ? -1 : 1;
}

export function playerCapsuleSpawnPosition(markerMin: readonly [number, number, number], markerMax: readonly [number, number, number]): { x: number; y: number; z: number } {
    return {
        x: (markerMin[0] + markerMax[0]) / 2,
        y: (markerMin[1] + markerMax[1]) / 2,
        z: (markerMin[2] + markerMax[2]) / 2,
    };
}

export function playerSubmersionAabb(
    position: Readonly<{ x: number; y: number; z: number }>,
    currentCapsuleHeight: number,
    standingCapsuleHeight: number,
    standingEyeHeight: number,
    radius: number
): FluidParticleAabb {
    const eyeY = position.y + standingEyeHeight * (currentCapsuleHeight / standingCapsuleHeight);
    return {
        min: [position.x - radius, eyeY, position.z - radius],
        max: [position.x + radius, SUBMERGED_QUERY_MAX_Y, position.z + radius],
    };
}

export function playerSubmergedState(submerged: boolean, particleCount: number, enterThreshold: number): boolean {
    return submerged ? particleCount >= enterThreshold * SUBMERGED_EXIT_THRESHOLD_FACTOR : particleCount > enterThreshold;
}

export function playerCapsuleAabb(position: Readonly<{ x: number; y: number; z: number }>, capsuleHeight: number, capsuleRadius: number): FluidParticleAabb {
    const centerY = position.y - capsuleRadius * 0.5;
    return {
        min: [position.x - capsuleRadius, centerY - capsuleHeight * 0.5, position.z - capsuleRadius],
        max: [position.x + capsuleRadius, centerY + capsuleHeight * 0.5, position.z + capsuleRadius],
    };
}

export function playerElectricalContactState(contact: boolean, particleCount: number, enterThreshold: number): boolean {
    return contact ? particleCount >= enterThreshold * SUBMERGED_EXIT_THRESHOLD_FACTOR : particleCount >= enterThreshold;
}

export class PlayerBehavior implements Behavior<"player"> {
    public readonly name = "player";
    public readonly mesh: Mesh;
    private readonly entityName: string;
    private readonly context: AquanovaGameContext;
    private readonly weaponPickOptions: PickOptions;
    private readonly keys = new Set<string>();
    private readonly freePosition = { x: 0, y: 0, z: 0 };
    private readonly walkVelocity = { x: 0, z: 0 };
    private readonly flyVelocity = { x: 0, y: 0, z: 0 };
    private readonly disposers: Array<() => void> = [];
    private yaw: number;
    private pitch = 0;
    private yawTarget: number;
    private pitchTarget = 0;
    private targetDistance = 1;
    private verticalVelocity = 0;
    private jumpBufferSeconds = 0;
    private jumpActive = false;
    private crouchToggleQueued = false;
    private crouchTarget = false;
    private apertureCrouchActive = false;
    private apertureAdvanceRemaining = 0;
    private apertureLateralRemaining = 0;
    private apertureAssistSeconds = 0;
    private apertureForwardX = 0;
    private apertureForwardZ = 0;
    private apertureRightX = 0;
    private apertureRightZ = 0;
    private noclip = false;
    private frozen = false;
    private weaponTriggerHeld = false;
    private weaponTriggerSequence = 0;
    private weaponAimPickPending = false;
    private lastWeaponWheelTime = Number.NEGATIVE_INFINITY;
    private footstepSound: ManagedSound | null = null;
    private footstepDistance = 0;
    private footstepActive = false;
    private crosshair: HTMLDivElement | null = null;
    private contactOverlay: HTMLDivElement | null = null;
    private electricityRegistration: FluidElectricityRegistration | null = null;
    private submerged = false;
    private electricalContact = false;
    private electricalParticleCount = 0;
    private readonly characterStrength: number;
    public readonly maxGrabDistance: number;
    public readonly maxHeldObjectDistance: number;
    public readonly submergedParticleCount: number;
    public readonly electrifiedParticleCount: number;

    public constructor(entityName: string, meshes: readonly Mesh[], config: PlayerBehaviorConfig, context: AquanovaGameContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] player requires at least one mesh");
        }
        this.mesh = mesh;
        this.entityName = entityName;
        this.context = context;
        this.weaponPickOptions = { filter: context.isCollisionActive };
        const characterStrength = config.characterStrength ?? DEFAULT_CHARACTER_STRENGTH;
        if (!Number.isFinite(characterStrength) || characterStrength < 0) {
            throw new Error(`[aquanova] player.characterStrength must be a finite non-negative number, received ${String(characterStrength)}`);
        }
        this.characterStrength = characterStrength;
        const maxGrabDistance = config.maxGrabDistance ?? DEFAULT_MAX_GRAB_DISTANCE;
        if (!Number.isFinite(maxGrabDistance) || maxGrabDistance <= 0) {
            throw new Error(`[aquanova] player.maxGrabDistance must be a finite positive number, received ${String(maxGrabDistance)}`);
        }
        this.maxGrabDistance = maxGrabDistance;
        const maxHeldObjectDistance = config.maxHeldObjectDistance ?? DEFAULT_MAX_HELD_OBJECT_DISTANCE;
        if (!Number.isFinite(maxHeldObjectDistance) || maxHeldObjectDistance <= 0) {
            throw new Error(`[aquanova] player.maxHeldObjectDistance must be a finite positive number, received ${String(maxHeldObjectDistance)}`);
        }
        this.maxHeldObjectDistance = maxHeldObjectDistance;
        const submergedParticleCount = config.submergedParticleCount ?? DEFAULT_SUBMERGED_PARTICLE_COUNT;
        if (!Number.isInteger(submergedParticleCount) || submergedParticleCount <= 0) {
            throw new Error(`[aquanova] player.submergedParticleCount must be a positive integer, received ${String(submergedParticleCount)}`);
        }
        this.submergedParticleCount = submergedParticleCount;
        const electrifiedParticleCount = config.electrifiedParticleCount ?? DEFAULT_ELECTRIFIED_PARTICLE_COUNT;
        if (!Number.isInteger(electrifiedParticleCount) || electrifiedParticleCount <= 0) {
            throw new Error(`[aquanova] player.electrifiedParticleCount must be a positive integer, received ${String(electrifiedParticleCount)}`);
        }
        this.electrifiedParticleCount = electrifiedParticleCount;
        const direction = config.direction;
        this.yaw = direction && (direction[0] || direction[2]) ? Math.atan2(-direction[0]!, direction[2]!) : -Math.PI / 2;
        this.yawTarget = this.yaw;
    }

    public async init(): Promise<void> {
        try {
            this.footstepSound = await this.context.sounds.load("player:stepMetallic", FOOTSTEP_SOUND_URL, { preloadCount: 1 });
        } catch (error) {
            throw new Error(`[aquanova] failed to preload player footstep sound from "${FOOTSTEP_SOUND_URL}"`, { cause: error });
        }
    }

    public start(): void {
        this.context.events.emit("entityEvent", { name: this.entityName, event: "disableCollision" });
        this.context.character.characterStrength = this.characterStrength;
        this.context.canvas.dataset.characterStrength = String(this.characterStrength);
        this.context.canvas.dataset.maxGrabDistance = String(this.maxGrabDistance);
        this.context.canvas.dataset.maxHeldObjectDistance = String(this.maxHeldObjectDistance);
        this.context.canvas.dataset.submergedParticleThreshold = String(this.submergedParticleCount);
        this.context.canvas.dataset.electrifiedParticleThreshold = String(this.electrifiedParticleCount);
        this.updateCrouchDataset();
        this.createCrosshair();
        this.createContactOverlay();
        this.electricityRegistration = this.context.fluidSimulations.registerElectricityReceiver({
            entityName: this.entityName,
            particleThreshold: this.electrifiedParticleCount,
            aabb: () => (this.noclip ? null : playerCapsuleAabb(this.context.character.getPosition(), this.capsuleHeight(), this.capsuleRadius())),
            includeParticleRadius: true,
            onCount: (particleCount) => {
                this.updateElectricalContact(particleCount);
            },
        });
        this.disposers.push(
            this.context.events.on("physicsStep", ({ deltaSeconds }) => {
                this.update(deltaSeconds);
                this.refreshHeldWeaponAim();
            })
        );
        this.listen(this.context.canvas, "click", this.onClick);
        this.listen(this.context.canvas, "contextmenu", this.onContextMenu);
        this.listen(this.context.canvas, "pointerdown", this.onPointerDown);
        this.listen(this.context.canvas, "pointercancel", this.onPointerCancel);
        this.listen(this.context.canvas, "pointermove", this.onPointerMove);
        this.listen(this.context.canvas, "wheel", this.onWheel);
        this.listen(window, "pointerup", this.onPointerUp);
        this.listen(window, "blur", this.onWindowBlur);
        this.listen(document, "pointerlockchange", this.onPointerLockChange);
        this.listen(window, "keydown", this.onKeyDown);
        this.listen(window, "keyup", this.onKeyUp);
    }

    public dispose(): void {
        for (const dispose of this.disposers.splice(0)) dispose();
        this.crosshair?.remove();
        this.crosshair = null;
        this.contactOverlay?.remove();
        this.contactOverlay = null;
        this.electricityRegistration?.dispose();
        this.electricityRegistration = null;
    }

    public get isNoclip(): boolean {
        return this.noclip;
    }

    public get isCrouched(): boolean {
        return this.crouchTarget || !this.isFullyStanding();
    }

    public get currentElectrifiedParticleCount(): number {
        return this.electricalParticleCount;
    }

    public get weaponSwayMultiplier(): 1 | 2 | 4 {
        return playerWeaponSwayMultiplier(this.keys, this.frozen);
    }

    public get isWeaponTriggerHeld(): boolean {
        return this.weaponTriggerHeld;
    }

    public getPosition(): { x: number; y: number; z: number } {
        return this.context.character.getPosition();
    }

    public setYaw(yaw: number): void {
        this.yaw = this.yawTarget = yaw;
    }

    public setCameraPosition(position: { x: number; y: number; z: number }): void {
        if (this.noclip) {
            this.freePosition.x = position.x;
            this.freePosition.y = position.y;
            this.freePosition.z = position.z;
        } else {
            this.context.character.setPosition({
                x: position.x,
                y: position.y - this.currentEyeHeight(),
                z: position.z,
            });
        }
        this.context.camera.position.set(position.x, position.y, position.z);
    }

    public setCameraTarget(target: { x: number; y: number; z: number }): void {
        const position = this.context.camera.position;
        const dx = target.x - position.x;
        const dy = target.y - position.y;
        const dz = target.z - position.z;
        const horizontal = Math.hypot(dx, dz);
        if (horizontal > 1e-6 || Math.abs(dy) > 1e-6) {
            this.yaw = this.yawTarget = Math.atan2(dx, dz);
            this.pitch = this.pitchTarget = Math.max(-1.45, Math.min(1.45, Math.atan2(dy, horizontal)));
            this.targetDistance = Math.hypot(horizontal, dy);
        }
        this.context.camera.target.set(target.x, target.y, target.z);
    }

    public look(dx: number, dy: number): void {
        this.yaw = this.yawTarget = this.yawTarget + dx * LOOK_SENSITIVITY;
        this.pitch = this.pitchTarget = Math.max(-1.45, Math.min(1.45, this.pitchTarget - dy * LOOK_SENSITIVITY));
    }

    public press(code: string): void {
        if (this.keys.has(code)) return;
        this.keys.add(code);
        if (code === "KeyC" && !this.noclip) {
            this.crouchToggleQueued = true;
        } else if (code === "Space" && !this.noclip) {
            this.jumpBufferSeconds = JUMP_BUFFER_SECONDS;
        }
    }

    public release(code: string): void {
        this.keys.delete(code);
    }

    public toggleNoclip(): void {
        this.noclip = !this.noclip;
        if (this.noclip) {
            this.crouchToggleQueued = false;
            this.freePosition.x = this.context.camera.position.x;
            this.freePosition.y = this.context.camera.position.y;
            this.freePosition.z = this.context.camera.position.z;
            this.flyVelocity.x = this.flyVelocity.y = this.flyVelocity.z = 0;
        } else {
            const eyeHeight = this.currentEyeHeight();
            const capsuleHeight = this.capsuleHeight();
            this.context.character.setPosition({
                x: this.freePosition.x,
                y: Math.max(this.freePosition.y - eyeHeight, capsuleHeight / 2),
                z: this.freePosition.z,
            });
            this.walkVelocity.x = this.walkVelocity.z = 0;
        }
        this.context.canvas.dataset.noclip = String(this.noclip);
    }

    public warp(x: number, y: number, z: number): void {
        this.context.character.setPosition({ x, y, z });
    }

    public freeze(): void {
        this.frozen = true;
    }

    public async fire(requireHeldTrigger = false, triggerSequence = ++this.weaponTriggerSequence): Promise<void> {
        if (!requireHeldTrigger) this.context.events.emit("weaponTriggerPressed", { held: false });
        const [pickX, pickY] = this.crosshairPickCoordinates();
        const info = await pickAsync(this.context.getPicker(), pickX, pickY, this.weaponPickOptions);
        const mesh = info.hit ? (info.pickedMesh as Mesh | null) : null;
        const activeTrigger = triggerSequence === this.weaponTriggerSequence && (!requireHeldTrigger || this.weaponTriggerHeld);
        const justReleasedTrigger = requireHeldTrigger && !this.weaponTriggerHeld && this.weaponTriggerSequence === triggerSequence + 1;
        if (!activeTrigger && !justReleasedTrigger) return;
        this.updateTargetDataset(mesh);
        this.context.events.emit("weaponAimUpdated", {
            mesh,
            point: info.hit ? info.pickedPoint : null,
            distance: info.hit ? info.distance : null,
        });
    }

    private crosshairPickCoordinates(): readonly [number, number] {
        const canvas = this.context.canvas;
        const canvasRect = canvas.getBoundingClientRect();
        const crosshairRect = this.crosshair?.getBoundingClientRect();
        if (!crosshairRect) return [canvas.clientWidth * 0.5, canvas.clientHeight * 0.5];
        return [
            ((crosshairRect.left + crosshairRect.width * 0.5 - canvasRect.left) / Math.max(1, canvasRect.width)) * canvas.clientWidth,
            ((crosshairRect.top + crosshairRect.height * 0.5 - canvasRect.top) / Math.max(1, canvasRect.height)) * canvas.clientHeight,
        ];
    }

    private updateTargetDataset(mesh: Mesh | null): void {
        this.context.canvas.dataset.target = mesh ? `${this.context.nodeNameOf(mesh)}|${this.context.isLiquefiable(mesh) ? "liq" : "no"}` : "none";
    }

    private refreshHeldWeaponAim(): void {
        if (!this.weaponTriggerHeld || this.weaponAimPickPending) return;
        const triggerSequence = this.weaponTriggerSequence;
        const [pickX, pickY] = this.crosshairPickCoordinates();
        this.weaponAimPickPending = true;
        void pickAsync(this.context.getPicker(), pickX, pickY, this.weaponPickOptions)
            .then((info) => {
                if (!this.weaponTriggerHeld || triggerSequence !== this.weaponTriggerSequence) return;
                const mesh = info.hit ? (info.pickedMesh as Mesh | null) : null;
                this.updateTargetDataset(mesh);
                this.context.events.emit("weaponAimUpdated", {
                    mesh,
                    point: info.hit ? info.pickedPoint : null,
                    distance: info.hit ? info.distance : null,
                });
            })
            .catch((err: unknown) => {
                console.warn("[aquanova] held weapon aim pick failed", err);
            })
            .finally(() => {
                this.weaponAimPickPending = false;
            });
    }

    private update(deltaSeconds: number): void {
        const lookFactor = 1 - Math.exp(-deltaSeconds * LOOK_ACCELERATION);
        this.yaw += (this.yawTarget - this.yaw) * lookFactor;
        this.pitch += (this.pitchTarget - this.pitch) * lookFactor;
        const inputZ = (this.keys.has("KeyW") || this.keys.has("ArrowUp") ? 1 : 0) - (this.keys.has("KeyS") || this.keys.has("ArrowDown") ? 1 : 0);
        const inputX = (this.keys.has("KeyD") || this.keys.has("ArrowRight") ? 1 : 0) - (this.keys.has("KeyA") || this.keys.has("ArrowLeft") ? 1 : 0);
        const cos = Math.cos(this.yaw);
        const sin = Math.sin(this.yaw);
        const cosPitch = Math.cos(this.pitch);
        const sinPitch = Math.sin(this.pitch);
        const runRequested = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight");
        const moveFactor = 1 - Math.exp(-deltaSeconds * MOVE_ACCELERATION);

        if (this.noclip) {
            this.resetFootsteps();
            const up = (this.keys.has("Space") ? 1 : 0) - (this.keys.has("ControlLeft") || this.keys.has("KeyC") ? 1 : 0);
            const speed = FLY_SPEED * (runRequested ? 2 : 1);
            this.flyVelocity.x += ((sin * cosPitch * inputZ + cos * inputX) * speed - this.flyVelocity.x) * moveFactor;
            this.flyVelocity.y += ((sinPitch * inputZ + up) * speed - this.flyVelocity.y) * moveFactor;
            this.flyVelocity.z += ((cos * cosPitch * inputZ - sin * inputX) * speed - this.flyVelocity.z) * moveFactor;
            this.freePosition.x += this.flyVelocity.x * deltaSeconds;
            this.freePosition.y += this.flyVelocity.y * deltaSeconds;
            this.freePosition.z += this.flyVelocity.z * deltaSeconds;
            this.context.camera.position.set(this.freePosition.x, this.freePosition.y, this.freePosition.z);
            this.context.camera.target.set(
                this.freePosition.x + sin * cosPitch * this.targetDistance,
                this.freePosition.y + sinPitch * this.targetDistance,
                this.freePosition.z + cos * cosPitch * this.targetDistance
            );
            this.updatePositionDataset(this.freePosition);
            this.updateSubmersion({
                min: [this.freePosition.x - this.context.capsuleRadius, this.freePosition.y, this.freePosition.z - this.context.capsuleRadius],
                max: [this.freePosition.x + this.context.capsuleRadius, SUBMERGED_QUERY_MAX_Y, this.freePosition.z + this.context.capsuleRadius],
            });
            return;
        }

        const support = this.context.character.checkSupport(deltaSeconds, DOWN);
        const grounded = support.supportedState === CharacterSupportedState.SUPPORTED;
        if (grounded && this.verticalVelocity <= 0) {
            this.jumpActive = false;
        }
        if (this.crouchToggleQueued) {
            this.crouchTarget = !this.crouchTarget;
            this.crouchToggleQueued = false;
        }
        if (!this.apertureCrouchActive && (runRequested || this.jumpBufferSeconds > 0) && (this.crouchTarget || !this.isFullyStanding())) {
            this.crouchTarget = false;
        }
        const apertureAssist = this.jumpActive && inputZ > 0 && !this.crouchTarget ? this.context.jumpApertureAssist(sin, cos) : null;
        if (apertureAssist && shouldUseJumpApertureAssist(this.jumpActive, inputZ, this.crouchTarget, apertureAssist)) {
            this.crouchTarget = true;
            this.apertureCrouchActive = true;
            this.apertureAdvanceRemaining = APERTURE_EDGE_ADVANCE;
            this.apertureLateralRemaining = apertureAssist.lateralOffset;
            this.apertureAssistSeconds = APERTURE_ASSIST_SECONDS;
            this.apertureForwardX = sin;
            this.apertureForwardZ = cos;
            this.apertureRightX = cos;
            this.apertureRightZ = -sin;
        }
        this.updateCrouch(deltaSeconds);

        const fullyStanding = this.isFullyStanding();
        const run = runRequested && fullyStanding ? 2 : 1;
        const speed = WALK_SPEED * run * (1 - (1 - CROUCH_SPEED_FACTOR) * this.crouchAmount());
        this.walkVelocity.x += ((inputX * cos + inputZ * sin) * speed - this.walkVelocity.x) * moveFactor;
        this.walkVelocity.z += ((-inputX * sin + inputZ * cos) * speed - this.walkVelocity.z) * moveFactor;
        if (grounded && this.verticalVelocity <= 0) {
            if (this.jumpBufferSeconds > 0 && fullyStanding) {
                this.verticalVelocity = JUMP_SPEED;
                this.jumpActive = true;
                this.jumpBufferSeconds = 0;
            } else {
                this.verticalVelocity = 0;
            }
        } else {
            this.verticalVelocity -= JUMP_GRAVITY * deltaSeconds;
        }
        this.jumpBufferSeconds = Math.max(0, this.jumpBufferSeconds - deltaSeconds);
        const supportedVelocity =
            grounded && this.verticalVelocity <= 0
                ? playerSupportedMovementVelocity(this.walkVelocity.x, this.walkVelocity.z, support.averageSurfaceNormal)
                : { x: this.walkVelocity.x, y: this.verticalVelocity, z: this.walkVelocity.z };
        let moveX = supportedVelocity.x * deltaSeconds;
        let moveZ = supportedVelocity.z * deltaSeconds;
        const apertureMoving = this.apertureCrouchActive && inputZ > 0 && this.crouchAmount() >= 0.95;
        if (apertureMoving && this.apertureAdvanceRemaining > 0) {
            const advance = Math.min(this.apertureAdvanceRemaining, APERTURE_EDGE_ADVANCE_SPEED * deltaSeconds);
            moveX += this.apertureForwardX * advance;
            moveZ += this.apertureForwardZ * advance;
        }
        if (apertureMoving && Math.abs(this.apertureLateralRemaining) > 1e-4) {
            const correction = Math.sign(this.apertureLateralRemaining) * Math.min(Math.abs(this.apertureLateralRemaining), APERTURE_LATERAL_SPEED * deltaSeconds);
            moveX += this.apertureRightX * correction;
            moveZ += this.apertureRightZ * correction;
        }
        const previousPosition = this.context.character.getPosition();
        const previousX = previousPosition.x;
        const previousZ = previousPosition.z;
        this.context.character.moveWithCollisions({
            x: moveX,
            y: supportedVelocity.y * deltaSeconds,
            z: moveZ,
        });
        const position = this.context.character.getPosition();
        if (this.apertureCrouchActive) {
            const resolvedX = position.x - previousX;
            const resolvedZ = position.z - previousZ;
            const forwardProgress = resolvedX * this.apertureForwardX + resolvedZ * this.apertureForwardZ;
            const lateralProgress = resolvedX * this.apertureRightX + resolvedZ * this.apertureRightZ;
            this.apertureAdvanceRemaining = remainingForwardApertureAssist(this.apertureAdvanceRemaining, forwardProgress);
            this.apertureLateralRemaining = remainingLateralApertureAssist(this.apertureLateralRemaining, lateralProgress);
            this.apertureAssistSeconds -= deltaSeconds;
            if (
                inputZ <= 0 ||
                this.apertureAssistSeconds <= 0 ||
                (this.crouchAmount() >= 0.95 && this.apertureAdvanceRemaining <= 1e-3 && Math.abs(this.apertureLateralRemaining) <= 1e-3)
            ) {
                this.apertureCrouchActive = false;
                this.apertureAdvanceRemaining = 0;
                this.apertureLateralRemaining = 0;
            }
        }
        this.updatePositionDataset(position);
        this.updateSubmersion(playerSubmersionAabb(position, this.capsuleHeight(), this.context.capsuleHeight, this.context.eyeHeight, this.capsuleRadius()));
        this.updateFootsteps(!this.frozen && grounded && this.verticalVelocity <= 0 && (inputX !== 0 || inputZ !== 0), run > 1, position.x - previousX, position.z - previousZ);
        if (this.frozen) return;
        const eyeHeight = this.currentEyeHeight();
        this.context.camera.position.set(position.x, position.y + eyeHeight, position.z);
        this.context.camera.target.set(
            position.x + sin * cosPitch * this.targetDistance,
            position.y + eyeHeight + sinPitch * this.targetDistance,
            position.z + cos * cosPitch * this.targetDistance
        );
    }

    private updateFootsteps(moving: boolean, running: boolean, resolvedX: number, resolvedZ: number): void {
        if (!moving || !this.footstepSound) {
            this.resetFootsteps();
            return;
        }
        const distance = Math.hypot(resolvedX, resolvedZ);
        if (distance <= 1e-5) {
            return;
        }
        if (!this.footstepActive) {
            this.footstepActive = true;
            this.footstepDistance = 0;
            this.context.sounds.play(this.footstepSound, { volume: FOOTSTEP_VOLUME });
            return;
        }
        this.footstepDistance += distance;
        const stepDistance = running ? RUN_STEP_DISTANCE : WALK_STEP_DISTANCE;
        if (this.footstepDistance >= stepDistance) {
            this.footstepDistance %= stepDistance;
            this.context.sounds.play(this.footstepSound, { volume: FOOTSTEP_VOLUME });
        }
    }

    private resetFootsteps(): void {
        this.footstepActive = false;
        this.footstepDistance = 0;
    }

    private updateCrouch(deltaSeconds: number): void {
        const currentHeight = this.capsuleHeight();
        const targetHeight = this.crouchTarget ? CROUCH_CAPSULE_HEIGHT : this.context.capsuleHeight;
        const transitionSpeed = (this.context.capsuleHeight - CROUCH_CAPSULE_HEIGHT) / CROUCH_TRANSITION_SECONDS;
        const heightDelta = Math.min(Math.abs(targetHeight - currentHeight), transitionSpeed * deltaSeconds);
        if (heightDelta === 0) return;
        const nextHeight = currentHeight + Math.sign(targetHeight - currentHeight) * heightDelta;
        if (nextHeight > currentHeight && !this.context.canStand()) return;
        const amount = Math.max(0, Math.min(1, (this.context.capsuleHeight - nextHeight) / (this.context.capsuleHeight - CROUCH_CAPSULE_HEIGHT)));
        const radius = this.context.capsuleRadius + (CROUCH_CAPSULE_RADIUS - this.context.capsuleRadius) * amount;
        this.context.character.setShapeOptions({ capsuleHeight: nextHeight, capsuleRadius: Math.min(radius, nextHeight * 0.5) }, !this.apertureCrouchActive);
        this.updateCrouchDataset();
    }

    private isFullyStanding(): boolean {
        return this.capsuleHeight() >= this.context.capsuleHeight - 1e-4;
    }

    private crouchAmount(): number {
        const heightRange = this.context.capsuleHeight - CROUCH_CAPSULE_HEIGHT;
        return Math.max(0, Math.min(1, (this.context.capsuleHeight - this.capsuleHeight()) / heightRange));
    }

    private currentEyeHeight(): number {
        return this.context.eyeHeight * (this.capsuleHeight() / this.context.capsuleHeight);
    }

    private capsuleHeight(): number {
        return this.context.character.shapeOptions.capsuleHeight ?? this.context.capsuleHeight;
    }

    private capsuleRadius(): number {
        return this.context.character.shapeOptions.capsuleRadius ?? this.context.capsuleRadius;
    }

    private updateCrouchDataset(): void {
        this.context.canvas.dataset.crouched = String(this.isCrouched);
        this.context.canvas.dataset.capsuleHeight = this.capsuleHeight().toFixed(2);
        this.context.canvas.dataset.crouchAmount = this.crouchAmount().toFixed(3);
    }

    private updatePositionDataset(position: { x: number; y: number; z: number }): void {
        this.context.canvas.dataset.pos = `${position.x.toFixed(2)},${position.y.toFixed(2)},${position.z.toFixed(2)}`;
    }

    private createCrosshair(): void {
        const crosshair = document.createElement("div");
        crosshair.id = "aq-crosshair";
        crosshair.style.cssText = "position:fixed;left:50%;top:50%;width:24px;height:24px;transform:translate(-50%,-50%);z-index:15;pointer-events:none;";
        const bar = (style: string): string => `<div style="position:absolute;background:rgba(120,232,255,.92);${style}"></div>`;
        crosshair.innerHTML =
            bar("left:50%;top:0;width:2px;height:8px;margin-left:-1px;") +
            bar("left:50%;bottom:0;width:2px;height:8px;margin-left:-1px;") +
            bar("top:50%;left:0;height:2px;width:8px;margin-top:-1px;") +
            bar("top:50%;right:0;height:2px;width:8px;margin-top:-1px;");
        if (this.context.canvas.dataset.weaponEnabled === "false") {
            crosshair.style.display = "none";
        }
        document.body.appendChild(crosshair);
        this.crosshair = crosshair;
    }

    private createContactOverlay(): void {
        const overlay = document.createElement("div");
        overlay.id = "aq-player-contact";
        overlay.style.cssText =
            "position:fixed;inset:0;z-index:15;pointer-events:none;border:3px solid transparent;opacity:0;transition:opacity .08s linear,border-color .08s linear,box-shadow .08s linear,background .08s linear;";
        document.body.appendChild(overlay);
        this.contactOverlay = overlay;
        this.context.canvas.dataset.submerged = "false";
        this.context.canvas.dataset.submergedParticleCount = "0";
        this.context.canvas.dataset.electricalContact = "false";
        this.context.canvas.dataset.electrifiedParticleCount = "0";
    }

    private updateSubmersion(aabb: FluidParticleAabb): void {
        const particleCount = this.context.fluidSimulations.countParticlesInAabb(aabb);
        this.context.canvas.dataset.submergedParticleCount = String(particleCount);
        const submerged = playerSubmergedState(this.submerged, particleCount, this.submergedParticleCount);
        if (submerged === this.submerged) {
            return;
        }
        this.submerged = submerged;
        this.context.canvas.dataset.submerged = String(submerged);
        this.syncContactOverlay();
    }

    private updateElectricalContact(particleCount: number): void {
        this.electricalParticleCount = particleCount;
        this.context.canvas.dataset.electrifiedParticleCount = String(particleCount);
        const contact = playerElectricalContactState(this.electricalContact, particleCount, this.electrifiedParticleCount);
        if (contact === this.electricalContact) {
            return;
        }
        this.electricalContact = contact;
        this.context.canvas.dataset.electricalContact = String(contact);
        this.syncContactOverlay();
        this.context.events.emit("entityEvent", {
            name: this.entityName,
            event: contact ? "electricalContact" : "electricalContactEnded",
        });
    }

    private syncContactOverlay(): void {
        const overlay = this.contactOverlay;
        if (!overlay) return;
        if (this.submerged && this.electricalContact) {
            overlay.style.borderColor = "rgba(220,160,255,.98)";
            overlay.style.boxShadow = "inset 32px 0 70px 12px rgba(255,55,70,.9),inset -32px 0 70px 12px rgba(45,155,255,.9),inset 0 0 150px 46px rgba(170,55,255,.58)";
            overlay.style.background = "rgba(175,75,210,.16)";
        } else if (this.submerged) {
            overlay.style.borderColor = "rgba(255,145,145,.95)";
            overlay.style.boxShadow = "inset 0 0 48px 18px rgba(255,70,70,.95),inset 0 0 150px 46px rgba(255,25,25,.55)";
            overlay.style.background = "rgba(255,110,110,.14)";
        } else if (this.electricalContact) {
            overlay.style.borderColor = "rgba(145,235,255,.95)";
            overlay.style.boxShadow = "inset 0 0 48px 18px rgba(70,205,255,.95),inset 0 0 150px 46px rgba(25,105,255,.55)";
            overlay.style.background = "rgba(110,215,255,.14)";
        }
        overlay.style.opacity = this.submerged || this.electricalContact ? "1" : "0";
    }

    private readonly onClick = (): void => {
        if (this.context.isInspecting() || isGizmoInteracting(this.context.canvas)) return;
        if (document.pointerLockElement !== this.context.canvas) {
            void Promise.resolve(this.context.canvas.requestPointerLock()).catch(() => {});
        }
    };

    private readonly onPointerDown = (event: PointerEvent): void => {
        if (event.button !== 0 && event.button !== 2) return;
        if (document.pointerLockElement !== this.context.canvas || this.context.isInspecting() || isGizmoInteracting(this.context.canvas) || this.noclip) return;
        if (event.button === 2) {
            event.preventDefault();
            this.context.events.emit("weaponSecondaryPressed", {});
            return;
        }
        if (this.weaponTriggerHeld) return;
        this.weaponTriggerHeld = true;
        const sequence = ++this.weaponTriggerSequence;
        this.context.events.emit("weaponTriggerPressed", { held: true });
        void this.fire(true, sequence);
    };

    private readonly onContextMenu = (event: MouseEvent): void => {
        if (document.pointerLockElement === this.context.canvas) {
            event.preventDefault();
        }
    };

    private readonly onPointerUp = (event: PointerEvent): void => {
        if (event.button === 0) this.releaseWeaponTrigger();
    };

    private readonly onPointerCancel = (): void => {
        this.releaseWeaponTrigger();
    };

    private readonly onWindowBlur = (): void => {
        this.releaseWeaponTrigger();
    };

    private readonly onPointerLockChange = (): void => {
        if (document.pointerLockElement !== this.context.canvas) this.releaseWeaponTrigger();
    };

    private readonly onPointerMove = (event: PointerEvent): void => {
        if (document.pointerLockElement === this.context.canvas) {
            this.yawTarget += event.movementX * LOOK_SENSITIVITY;
            this.pitchTarget = Math.max(-1.45, Math.min(1.45, this.pitchTarget - event.movementY * LOOK_SENSITIVITY));
            return;
        }
        this.context.inspectAt(event.offsetX, event.offsetY);
    };

    private readonly onWheel = (event: WheelEvent): void => {
        const direction = weaponWheelDirection(event.deltaY);
        if (direction === null || document.pointerLockElement !== this.context.canvas || this.context.isInspecting()) {
            return;
        }
        event.preventDefault();
        const now = performance.now();
        if (now - this.lastWeaponWheelTime < 120) {
            return;
        }
        this.lastWeaponWheelTime = now;
        this.context.events.emit("weaponCycleRequested", { direction });
    };

    private readonly onKeyDown = (event: KeyboardEvent): void => {
        this.press(event.code);
    };

    private readonly onKeyUp = (event: KeyboardEvent): void => {
        this.keys.delete(event.code);
    };

    private releaseWeaponTrigger(): void {
        if (!this.weaponTriggerHeld) return;
        this.weaponTriggerHeld = false;
        this.weaponTriggerSequence++;
        this.context.events.emit("weaponTriggerReleased", {});
    }

    private listen<K extends keyof HTMLElementEventMap>(target: HTMLElement, name: K, listener: (this: HTMLElement, event: HTMLElementEventMap[K]) => unknown): void;
    private listen<K extends keyof WindowEventMap>(target: Window, name: K, listener: (this: Window, event: WindowEventMap[K]) => unknown): void;
    private listen<K extends keyof DocumentEventMap>(target: Document, name: K, listener: (this: Document, event: DocumentEventMap[K]) => unknown): void;
    private listen(target: HTMLElement | Window | Document, name: string, listener: EventListener): void {
        target.addEventListener(name, listener);
        this.disposers.push(() => target.removeEventListener(name, listener));
    }
}
