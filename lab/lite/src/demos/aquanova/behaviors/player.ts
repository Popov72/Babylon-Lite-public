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
    private jumpQueued = false;
    private noclip = false;
    private frozen = false;
    private firing = false;
    private crosshair: HTMLDivElement | null = null;

    public constructor(mesh: Mesh, config: PlayerBehaviorConfig, context: BehaviorContext) {
        this.mesh = mesh;
        this.context = context;
        const direction = config.direction;
        this.yaw = direction && (direction[0] || direction[2]) ? Math.atan2(-direction[0]!, direction[2]!) : -Math.PI / 2;
        this.yawTarget = this.yaw;
    }

    public start(): void {
        this.createCrosshair();
        this.disposers.push(this.context.events.on("physicsStep", ({ deltaSeconds }) => this.update(deltaSeconds)));
        this.listen(this.context.canvas, "click", this.onClick);
        this.listen(this.context.canvas, "pointerdown", this.onPointerDown);
        this.listen(this.context.canvas, "pointermove", this.onPointerMove);
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
        this.keys.add(code);
    }

    public release(code: string): void {
        this.keys.delete(code);
    }

    public toggleNoclip(): void {
        this.noclip = !this.noclip;
        if (this.noclip) {
            const position = this.context.character.getPosition();
            this.freePosition.x = position.x;
            this.freePosition.y = position.y + this.context.eyeHeight;
            this.freePosition.z = position.z;
            this.flyVelocity.x = this.flyVelocity.y = this.flyVelocity.z = 0;
        } else {
            this.context.character.setPosition({
                x: this.freePosition.x,
                y: Math.max(this.freePosition.y - this.context.eyeHeight, this.context.capsuleHeight / 2),
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

    public async fire(): Promise<void> {
        if (this.firing) return;
        this.firing = true;
        try {
            const canvas = this.context.canvas;
            const info = await pickAsync(this.context.getPicker(), canvas.clientWidth / 2, canvas.clientHeight / 2);
            const mesh = info.hit ? (info.pickedMesh as Mesh | null) : null;
            canvas.dataset.target = mesh ? `${this.context.nodeNameOf(mesh)}|${this.context.isLiquefiable(mesh) ? "liq" : "no"}` : "none";
            if (mesh) {
                this.context.events.emit("hitWithWeapon", {
                    mesh,
                    point: info.pickedPoint,
                });
            }
        } finally {
            this.firing = false;
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
        const run = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? 2 : 1;
        const moveFactor = 1 - Math.exp(-deltaSeconds * MOVE_ACCELERATION);

        if (this.noclip) {
            const up = (this.keys.has("Space") ? 1 : 0) - (this.keys.has("ControlLeft") || this.keys.has("KeyC") ? 1 : 0);
            const speed = FLY_SPEED * run;
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

        const speed = WALK_SPEED * run;
        this.walkVelocity.x += ((inputX * cos + inputZ * sin) * speed - this.walkVelocity.x) * moveFactor;
        this.walkVelocity.z += ((-inputX * sin + inputZ * cos) * speed - this.walkVelocity.z) * moveFactor;
        const grounded = this.context.character.checkSupport(deltaSeconds, DOWN).supportedState === CharacterSupportedState.SUPPORTED;
        if (grounded && this.verticalVelocity <= 0) {
            this.verticalVelocity = this.jumpQueued ? JUMP_SPEED : -2;
        } else {
            this.verticalVelocity -= JUMP_GRAVITY * deltaSeconds;
        }
        this.jumpQueued = false;
        this.context.character.moveWithCollisions({
            x: this.walkVelocity.x * deltaSeconds,
            y: this.verticalVelocity * deltaSeconds,
            z: this.walkVelocity.z * deltaSeconds,
        });
        const position = this.context.character.getPosition();
        this.updatePositionDataset(position);
        if (this.frozen) return;
        this.context.camera.position.set(position.x, position.y + this.context.eyeHeight, position.z);
        this.context.camera.target.set(position.x + sin * cosPitch, position.y + this.context.eyeHeight + sinPitch, position.z + cos * cosPitch);
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
        void this.fire();
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
        if (event.code === "KeyV" && !event.repeat) this.toggleNoclip();
        if (event.code === "Space" && !event.repeat && !this.noclip) this.jumpQueued = true;
        this.keys.add(event.code);
    };

    private readonly onKeyUp = (event: KeyboardEvent): void => {
        this.keys.delete(event.code);
    };

    private listen<K extends keyof HTMLElementEventMap>(target: HTMLElement, name: K, listener: (this: HTMLElement, event: HTMLElementEventMap[K]) => unknown): void;
    private listen<K extends keyof WindowEventMap>(target: Window, name: K, listener: (this: Window, event: WindowEventMap[K]) => unknown): void;
    private listen(target: HTMLElement | Window, name: string, listener: EventListener): void {
        target.addEventListener(name, listener);
        this.disposers.push(() => target.removeEventListener(name, listener));
    }
}
