import type { FreeCamera, GpuPicker, Mesh, PhysicsCharacterController } from "babylon-lite";
import type { EventManager } from "./event-manager.js";

export interface DynamicBehaviorConfig {
    dynamic?: boolean;
}

export interface LiquefiableBehaviorConfig {
    liquefiable: true;
    fluidSim?: string[];
    linked?: string[];
}

export interface PlayerBehaviorConfig {
    direction?: number[];
    /** Maximum force the character controller applies while pushing dynamic bodies. */
    characterStrength?: number;
}

export interface WeaponBehaviorConfig {
    direction?: number[];
}

/** All parameters that a manifest behavior definition or entity override may provide. */
export interface BehaviorConfig {
    dynamic?: boolean;
    liquefiable?: boolean;
    fluidSim?: string[];
    linked?: string[];
    direction?: number[];
    characterStrength?: number;
}

export function isLiquefiableBehaviorConfig<Config extends BehaviorConfig>(config: Config): config is Config & LiquefiableBehaviorConfig {
    return config.liquefiable === true;
}

export interface BehaviorReference extends BehaviorConfig {
    name: string;
}

export type BehaviorAssignment = BehaviorReference;
export type BehaviorLibrary = Record<string, BehaviorConfig>;
export type Entities = Record<string, { behaviors?: BehaviorReference[] }>;

export interface BehaviorContext {
    readonly canvas: HTMLCanvasElement;
    readonly camera: FreeCamera;
    readonly character: PhysicsCharacterController;
    readonly events: EventManager;
    readonly capsuleHeight: number;
    readonly eyeHeight: number;
    readonly getPicker: () => GpuPicker;
    readonly nodeNameOf: (mesh: Mesh) => string;
    readonly isLiquefiable: (mesh: Mesh) => boolean;
    readonly isInspecting: () => boolean;
    readonly inspectAt: (x: number, y: number) => void;
    /** Begin validating a re-press against the reversing fusion. Null means there is no fusion to resume. */
    readonly requestFusionResume: () => number | null;
    readonly resolveFusionResume: (token: number, mesh: Mesh | null) => "resumed" | "start-new" | "continue";
    readonly reverseFusion: () => void;
    readonly liquefy: (mesh: Mesh, point: readonly [number, number, number] | null, config: LiquefiableBehaviorConfig) => void;
}

export interface Behavior<Name extends string = string> {
    readonly name: Name;
    readonly mesh: Mesh;
    start(): void;
    dispose(): void;
}
