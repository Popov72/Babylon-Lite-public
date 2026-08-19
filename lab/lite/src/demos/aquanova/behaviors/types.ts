import type { AnimationGroup, FreeCamera, GpuPicker, Mesh, PhysicsCharacterController } from "babylon-lite";
import type { EventManager } from "./event-manager.js";

export interface DynamicBehaviorConfig {
    dynamic?: boolean;
    /** Rigid-body mass in kilograms. Defaults to 10. */
    mass?: number;
}

export interface LiquefiableBehaviorConfig {
    liquefiable: true;
    fluidSim?: string[];
    linked?: string[];
    /** Splash sound category played when this mesh enters the fluid phase. */
    sound?: string;
}

export interface PlayerBehaviorConfig {
    direction?: number[];
    /** Maximum force the character controller applies while pushing dynamic bodies. */
    characterStrength?: number;
}

export interface EntityEventConfig {
    name: string;
    event: string;
}

export interface EntityToggleBehaviorConfig {
    /** Event addressed to the entity owning this behavior that triggers the forwarding. */
    onEvent: string;
    /** Entity or door that receives the forwarded `enable` or `disable` event. */
    entity: string;
}

export interface PickEntityBehaviorConfig {
    /** Per-axis scale applied to the pickup's world-space bounding box. Defaults to `[1, 1, 1]`. */
    boundingBoxScale?: number[];
    raiseEvent?: EntityEventConfig;
    /** MP3 file name without extension under `/aquanova/sounds/`. Defaults to `pickItem`. */
    sound?: string;
    /** Multiplier for the default one-revolution-per-3-seconds Y rotation. Defaults to `1`. */
    speed?: number;
}

export interface PlayAnimationBehaviorConfig {
    /** Exact glTF animation name. Defaults to the first animation in file order. */
    animation?: string;
    /** Whether playback loops. Defaults to true. */
    loop?: boolean;
}

export interface WeaponLiquefactorBehaviorConfig {
    direction?: number[];
    /** Maximum beam range when the crosshair does not hit geometry. */
    range?: number;
    /** Sound category names mapped to MP3 file names without extensions. */
    sounds?: Record<string, string[]>;
}

export interface WeaponAntiGravityGunBehaviorConfig {
    /** Maximum centre-screen hit distance that may be grabbed. Defaults to 6 metres. */
    maxGrabDistance?: number;
    /** Maximum dynamic-body mass that may be grabbed. Defaults to 100 kilograms. */
    maxMass?: number;
}

/** All parameters that a manifest behavior definition or entity override may provide. */
export interface BehaviorConfig {
    dynamic?: boolean;
    mass?: number;
    liquefiable?: boolean;
    fluidSim?: string[];
    linked?: string[];
    direction?: number[];
    characterStrength?: number;
    range?: number;
    sound?: string;
    sounds?: Record<string, string[]>;
    speed?: number;
    boundingBoxScale?: number[];
    entity?: string;
    onEvent?: string;
    raiseEvent?: EntityEventConfig;
    reflectionProbe?: "exclude";
    animation?: string;
    loop?: boolean;
    maxGrabDistance?: number;
    maxMass?: number;
}

export function isLiquefiableBehaviorConfig<Config extends BehaviorConfig>(config: Config): config is Config & LiquefiableBehaviorConfig {
    return config.liquefiable === true;
}

export function isPickEntityBehaviorConfig(config: BehaviorReference): config is BehaviorReference & PickEntityBehaviorConfig & { name: "pickEntity" } {
    return config.name === "pickEntity";
}

export function isEntityToggleBehaviorConfig(config: BehaviorReference): config is BehaviorReference & EntityToggleBehaviorConfig & { name: "disableEntity" | "enableEntity" } {
    return config.name === "disableEntity" || config.name === "enableEntity";
}

export interface BehaviorReference extends BehaviorConfig {
    name: string;
}

export type BehaviorAssignment = BehaviorReference;
export type BehaviorLibrary = Record<string, BehaviorConfig>;
export type Entities = Record<string, { behaviors?: BehaviorReference[] }>;

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

export interface WeaponInventoryRuntime {
    acquire(slot: number): void;
    isOwned(slot: number): boolean;
    isEquipped(slot: number): boolean;
}

export interface JumpApertureAssist {
    /** Signed correction along the player's right axis, in metres. */
    lateralOffset: number;
}

export interface BehaviorContext {
    readonly canvas: HTMLCanvasElement;
    readonly camera: FreeCamera;
    readonly character: PhysicsCharacterController;
    readonly events: EventManager;
    readonly animationGroups: readonly AnimationGroup[];
    readonly capsuleHeight: number;
    readonly capsuleRadius: number;
    readonly eyeHeight: number;
    /** Whether the standing capsule can expand upward without intersecting the ship. */
    readonly canStand: () => boolean;
    /** Entry correction when the forward path is clear only for the crouched capsule. */
    readonly jumpApertureAssist: (forwardX: number, forwardZ: number) => JumpApertureAssist | null;
    readonly getPicker: () => GpuPicker;
    readonly nodeNameOf: (mesh: Mesh) => string;
    readonly isLiquefiable: (mesh: Mesh) => boolean;
    readonly isInspecting: () => boolean;
    readonly inspectAt: (x: number, y: number) => void;
    readonly weaponInventory: WeaponInventoryRuntime;
    readonly weaponLiquefactor: WeaponLiquefactorRuntime;
    readonly weaponAntiGravityGun: WeaponAntiGravityGunRuntime;
    readonly dynamicMassOf: (mesh: Mesh) => number | null;
    /** Begin validating a re-press against reversing liquefaction. Null means there is no liquefaction to resume. */
    readonly requestFusionResume: () => number | null;
    readonly resolveFusionResume: (token: number, mesh: Mesh | null) => "resumed" | "start-new" | "await-target" | "continue";
    /** Resolve clipped active-liquefaction geometry before the visible picker target behind it. */
    readonly resolveFusionTarget: (mesh: Mesh | null, point: readonly [number, number, number] | null) => Mesh | null;
    /** Whether forward liquefaction should reverse because the held beam left its active mesh group. */
    readonly fusionTargetLost: (mesh: Mesh | null) => boolean;
    readonly reverseFusion: () => void;
    readonly liquefy: (mesh: Mesh, point: readonly [number, number, number] | null, config: LiquefiableBehaviorConfig) => void;
}

export interface Behavior<Name extends string = string> {
    readonly name: Name;
    readonly mesh: Mesh;
    start(): void;
    dispose(): void;
}
