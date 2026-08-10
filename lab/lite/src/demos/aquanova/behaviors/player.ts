import { CharacterSupportedState, pickAsync } from "babylon-lite";
import type { Mesh } from "babylon-lite";
import type { Behavior, BehaviorContext, PlayerBehaviorConfig } from "./types.js";

const LOOK_SENSITIVITY = 1 / 600;
const WALK_SPEED = 4;
const FLY_SPEED = 8;
const MOVE_ACCELERATION = 12;
const LOOK_ACCELERATION = 30;
const JUMP_SPEED = 6;
const JUMP_GRAVITY = 16;
const CROUCH_CAPSULE_HEIGHT = 0.8;
const CROUCH_SPEED_FACTOR = 0.75;
const CROUCH_TRANSITION_SECONDS = 0.2;
const JUMP_BUFFER_SECONDS = 0.3;
const DEFAULT_CHARACTER_STRENGTH = 100;
const DOWN = { x: 0, y: -1, z: 0 };

export class PlayerBehavior implements Behavior<"player"> {
    public readonly name = "player";
    public readonly mesh: Mesh;
    private readonly context: BehaviorContext;
    private readonly keys = new Set<string>();
    private readonly freePosition = { x: 0, y: 0, z: 0 };
    private readonly walkVelocity = { x: 0, z: 0 };
    private readonly flyVelocity = { x: 0, y: 0, z: 0 };
    private readonly disposers: Array<() => void> = [];
    private yaw: number;
    private pitch = 0;
    private yawTarget: number;
    private pitchTarget = 0;
    private verticalVelocity = 0;
    private jumpBufferSeconds = 0;
    private crouchToggleQueued = false;
    private crouchTarget = false;
    private noclip = false;
    private frozen = false;
    private fusionTriggerHeld = false;
    private fusionTriggerSequence = 0;
    private crosshair: HTMLDivElement | null = null;
    private readonly characterStrength: number;

    public constructor(mesh: Mesh, config: PlayerBehaviorConfig, context: BehaviorContext) {
        this.mesh = mesh;
        this.context = context;
        const characterStrength = config.characterStrength ?? DEFAULT_CHARACTER_STRENGTH;
        if (!Number.isFinite(characterStrength) || characterStrength < 0) {
            throw new Error(`[aquanova] player.characterStrength must be a finite non-negative number, received ${String(characterStrength)}`);
        }
        this.characterStrength = characterStrength;
        const direction = config.direction;
        this.yaw = direction && (direction[0] || direction[2]) ? Math.atan2(-direction[0]!, direction[2]!) : -Math.PI / 2;
        this.yawTarget = this.yaw;
    }

    public start(): void {
        this.context.character.characterStrength = this.characterStrength;
        this.context.canvas.dataset.characterStrength = String(this.characterStrength);
        this.updateCrouchDataset();
        this.createCrosshair();
        this.disposers.push(this.context.events.on("physicsStep", ({ deltaSeconds }) => this.update(deltaSeconds)));
        this.listen(this.context.canvas, "click", this.onClick);
        this.listen(this.context.canvas, "pointerdown", this.onPointerDown);
        this.listen(this.context.canvas, "pointercancel", this.onPointerCancel);
        this.listen(this.context.canvas, "pointermove", this.onPointerMove);
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
    }

    public get isNoclip(): boolean {
        return this.noclip;
    }

    public get isCrouched(): boolean {
        return this.crouchTarget || !this.isFullyStanding();
    }

    public getPosition(): { x: number; y: number; z: number } {
        return this.context.character.getPosition();
    }

    public setYaw(yaw: number): void {
        this.yaw = this.yawTarget = yaw;
    }

    public look(dx: number, dy: number): void {
        this.yaw = this.yawTarget = this.yawTarget + dx * LOOK_SENSITIVITY;
        this.pitch = this.pitchTarget = Math.max(-1.45, Math.min(1.45, this.pitchTarget - dy * LOOK_SENSITIVITY));
    }

    public press(code: string): void {
        if (this.keys.has(code)) return;
        this.keys.add(code);
        if (code === "KeyV") {
            this.toggleNoclip();
        } else if (code === "KeyC" && !this.noclip) {
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

    public async fire(requireHeldTrigger = false, triggerSequence = this.fusionTriggerSequence, resumeToken?: number): Promise<void> {
        const canvas = this.context.canvas;
        const info = await pickAsync(this.context.getPicker(), canvas.clientWidth / 2, canvas.clientHeight / 2);
        const mesh = info.hit ? (info.pickedMesh as Mesh | null) : null;
        if (requireHeldTrigger && (!this.fusionTriggerHeld || triggerSequence !== this.fusionTriggerSequence)) {
            if (resumeToken !== undefined) this.context.resolveFusionResume(resumeToken, null);
            return;
        }
        canvas.dataset.target = mesh ? `${this.context.nodeNameOf(mesh)}|${this.context.isLiquefiable(mesh) ? "liq" : "no"}` : "none";
        if (resumeToken !== undefined) {
            const result = this.context.resolveFusionResume(resumeToken, mesh);
            if (result === "start-new" && mesh) {
                this.context.events.emit("hitWithWeapon", {
                    mesh,
                    point: info.pickedPoint,
                });
            }
            return;
        }
        if (mesh) {
            this.context.events.emit("hitWithWeapon", {
                mesh,
                point: info.pickedPoint,
            });
        }
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
            const up = (this.keys.has("Space") ? 1 : 0) - (this.keys.has("ControlLeft") || this.keys.has("KeyC") ? 1 : 0);
            const speed = FLY_SPEED * (runRequested ? 2 : 1);
            this.flyVelocity.x += ((sin * cosPitch * inputZ + cos * inputX) * speed - this.flyVelocity.x) * moveFactor;
            this.flyVelocity.y += ((sinPitch * inputZ + up) * speed - this.flyVelocity.y) * moveFactor;
            this.flyVelocity.z += ((cos * cosPitch * inputZ - sin * inputX) * speed - this.flyVelocity.z) * moveFactor;
            this.freePosition.x += this.flyVelocity.x * deltaSeconds;
            this.freePosition.y += this.flyVelocity.y * deltaSeconds;
            this.freePosition.z += this.flyVelocity.z * deltaSeconds;
            this.context.camera.position.set(this.freePosition.x, this.freePosition.y, this.freePosition.z);
            this.context.camera.target.set(this.freePosition.x + sin * cosPitch, this.freePosition.y + sinPitch, this.freePosition.z + cos * cosPitch);
            this.updatePositionDataset(this.freePosition);
            return;
        }

        const grounded = this.context.character.checkSupport(deltaSeconds, DOWN).supportedState === CharacterSupportedState.SUPPORTED;
        if (this.crouchToggleQueued) {
            this.crouchTarget = !this.crouchTarget;
            this.crouchToggleQueued = false;
        }
        if ((runRequested || this.jumpBufferSeconds > 0) && (this.crouchTarget || !this.isFullyStanding())) {
            this.crouchTarget = false;
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
                this.jumpBufferSeconds = 0;
            } else {
                this.verticalVelocity = -2;
            }
        } else {
            this.verticalVelocity -= JUMP_GRAVITY * deltaSeconds;
        }
        this.jumpBufferSeconds = Math.max(0, this.jumpBufferSeconds - deltaSeconds);
        this.context.character.moveWithCollisions({
            x: this.walkVelocity.x * deltaSeconds,
            y: this.verticalVelocity * deltaSeconds,
            z: this.walkVelocity.z * deltaSeconds,
        });
        const position = this.context.character.getPosition();
        this.updatePositionDataset(position);
        if (this.frozen) return;
        const eyeHeight = this.currentEyeHeight();
        this.context.camera.position.set(position.x, position.y + eyeHeight, position.z);
        this.context.camera.target.set(position.x + sin * cosPitch, position.y + eyeHeight + sinPitch, position.z + cos * cosPitch);
    }

    private updateCrouch(deltaSeconds: number): void {
        const currentHeight = this.capsuleHeight();
        const targetHeight = this.crouchTarget ? CROUCH_CAPSULE_HEIGHT : this.context.capsuleHeight;
        const transitionSpeed = (this.context.capsuleHeight - CROUCH_CAPSULE_HEIGHT) / CROUCH_TRANSITION_SECONDS;
        const heightDelta = Math.min(Math.abs(targetHeight - currentHeight), transitionSpeed * deltaSeconds);
        if (heightDelta === 0) return;
        const nextHeight = currentHeight + Math.sign(targetHeight - currentHeight) * heightDelta;
        if (nextHeight > currentHeight && !this.context.canStand()) return;
        this.context.character.setShapeOptions({ ...this.context.character.shapeOptions, capsuleHeight: nextHeight });
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
        document.body.appendChild(crosshair);
        this.crosshair = crosshair;
    }

    private readonly onClick = (): void => {
        if (this.context.isInspecting()) return;
        if (document.pointerLockElement !== this.context.canvas) {
            void Promise.resolve(this.context.canvas.requestPointerLock()).catch(() => {});
        }
    };

    private readonly onPointerDown = (event: PointerEvent): void => {
        if (event.button !== 0) return;
        if (document.pointerLockElement !== this.context.canvas || this.context.isInspecting() || this.noclip) return;
        this.fusionTriggerHeld = true;
        const sequence = ++this.fusionTriggerSequence;
        const resumeToken = this.context.requestFusionResume();
        if (resumeToken !== null) {
            if (resumeToken > 0) void this.fire(true, sequence, resumeToken);
            return;
        }
        void this.fire(true, sequence);
    };

    private readonly onPointerUp = (event: PointerEvent): void => {
        if (event.button === 0) this.releaseFusionTrigger();
    };

    private readonly onPointerCancel = (): void => {
        this.releaseFusionTrigger();
    };

    private readonly onWindowBlur = (): void => {
        this.releaseFusionTrigger();
    };

    private readonly onPointerLockChange = (): void => {
        if (document.pointerLockElement !== this.context.canvas) this.releaseFusionTrigger();
    };

    private readonly onPointerMove = (event: PointerEvent): void => {
        if (document.pointerLockElement === this.context.canvas) {
            this.yawTarget += event.movementX * LOOK_SENSITIVITY;
            this.pitchTarget = Math.max(-1.45, Math.min(1.45, this.pitchTarget - event.movementY * LOOK_SENSITIVITY));
            return;
        }
        this.context.inspectAt(event.offsetX, event.offsetY);
    };

    private readonly onKeyDown = (event: KeyboardEvent): void => {
        this.press(event.code);
    };

    private readonly onKeyUp = (event: KeyboardEvent): void => {
        this.keys.delete(event.code);
    };

    private releaseFusionTrigger(): void {
        if (!this.fusionTriggerHeld) return;
        this.fusionTriggerHeld = false;
        this.fusionTriggerSequence++;
        this.context.reverseFusion();
    }

    private listen<K extends keyof HTMLElementEventMap>(target: HTMLElement, name: K, listener: (this: HTMLElement, event: HTMLElementEventMap[K]) => unknown): void;
    private listen<K extends keyof WindowEventMap>(target: Window, name: K, listener: (this: Window, event: WindowEventMap[K]) => unknown): void;
    private listen<K extends keyof DocumentEventMap>(target: Document, name: K, listener: (this: Document, event: DocumentEventMap[K]) => unknown): void;
    private listen(target: HTMLElement | Window | Document, name: string, listener: EventListener): void {
        target.addEventListener(name, listener);
        this.disposers.push(() => target.removeEventListener(name, listener));
    }
}
