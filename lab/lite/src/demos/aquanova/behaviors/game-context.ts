import type { AnimationGroup, FreeCamera, GpuPicker, Mesh, PhysicsCharacterController } from "babylon-lite";
import type { AquanovaEventManager } from "./aquanova-event-manager.js";
import type { AquanovaFluidRuntime } from "../fluid-runtime.js";
import type { SoundManager } from "./sound-manager.js";
import type { FluidSimShape, LiquefiableBehaviorConfig } from "./types.js";
import type { PistolImpact } from "../pistol-projectiles.js";

export interface WeaponLiquefactorRuntime {
    setEnabled(enabled: boolean, animated?: boolean): void;
    setTargetDistance(distance: number | null, restart?: boolean): void;
    isReady(): boolean;
    stop(): void;
    update(deltaMs: number): boolean;
}

export interface WeaponAntiGravityGunRuntime {
    setEnabled(enabled: boolean, animated?: boolean): void;
    isReady(): boolean;
    update(deltaMs: number): void;
    grab(mesh: Mesh): boolean;
    updateGrab(deltaMs: number): boolean;
    releaseGrab(throwSpeed: number): void;
}

export interface WeaponPistolRuntime {
    setEnabled(enabled: boolean, animated?: boolean): void;
    isReady(): boolean;
    fire(mesh: Mesh | null, point: readonly [number, number, number] | null, distance: number | null, range: number, speed: number): void;
    update(deltaMs: number): readonly PistolImpact[];
    clear(): void;
}

export interface WeaponInventoryRuntime {
    acquire(slot: number): void;
    isOwned(slot: number): boolean;
    isEquipped(slot: number): boolean;
}

export interface JumpApertureAssist {
    lateralOffset: number;
}

export interface IntersectionTriggerRegistration {
    setEnabled(enabled: boolean): void;
    dispose(): void;
}

export interface IntersectionTriggerCallbacks {
    onEntered(): void;
    onExited(): void;
}

export interface FluidSimShapeRegistration {
    readonly meshes: readonly Mesh[];
    readonly shape: FluidSimShape;
}

export interface AquanovaGameContext {
    readonly canvas: HTMLCanvasElement;
    readonly camera: FreeCamera;
    readonly character: PhysicsCharacterController;
    readonly events: AquanovaEventManager;
    readonly fluidSimulations: AquanovaFluidRuntime;
    readonly sounds: SoundManager;
    readonly animationGroups: readonly AnimationGroup[];
    readonly capsuleHeight: number;
    readonly capsuleRadius: number;
    readonly eyeHeight: number;
    readonly canStand: () => boolean;
    readonly jumpApertureAssist: (forwardX: number, forwardZ: number) => JumpApertureAssist | null;
    readonly getPicker: () => GpuPicker;
    readonly nodeNameOf: (mesh: Mesh) => string;
    readonly isLiquefiable: (mesh: Mesh) => boolean;
    readonly getLiquefiableConfig: (mesh: Mesh) => LiquefiableBehaviorConfig | undefined;
    readonly isInspecting: () => boolean;
    readonly inspectAt: (x: number, y: number) => void;
    readonly weaponInventory: WeaponInventoryRuntime;
    readonly weaponLiquefactor: WeaponLiquefactorRuntime;
    readonly weaponAntiGravityGun: WeaponAntiGravityGunRuntime;
    readonly weaponPistol: WeaponPistolRuntime;
    readonly playerMaxGrabDistance: () => number;
    readonly dynamicMassOf: (mesh: Mesh) => number | null;
    readonly setCollisionShape: (entityName: string, type: "aabb" | "mesh", fluidSimShape?: FluidSimShapeRegistration) => void;
    readonly registerIntersectionTrigger: (entityName: string, playerOnly: boolean, callbacks: IntersectionTriggerCallbacks) => IntersectionTriggerRegistration;
    readonly requestFusionResume: () => number | null;
    readonly resolveFusionResume: (token: number, mesh: Mesh | null) => "resumed" | "start-new" | "await-target" | "continue";
    readonly resolveFusionTarget: (mesh: Mesh | null, point: readonly [number, number, number] | null) => Mesh | null;
    readonly fusionTargetLost: (mesh: Mesh | null) => boolean;
    readonly reverseFusion: () => void;
    readonly liquefy: (mesh: Mesh, point: readonly [number, number, number] | null, config: LiquefiableBehaviorConfig) => void;
}
