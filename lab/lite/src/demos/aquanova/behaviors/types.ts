import type { FreeCamera, GpuPicker, Mesh, PhysicsCharacterController } from "babylon-lite";
import type { EventManager } from "./event-manager.js";

export interface DynamicBehaviorConfig {
    dynamic?: boolean;
}

export interface LiquefiableBehaviorConfig {
    liquefiable: true;
    fluidSim?: string[];
    linked?: string[];
    excludeSDF?: string[];
}

export interface PlayerBehaviorConfig {
    direction?: number[];
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
    excludeSDF?: string[];
    direction?: number[];
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
    readonly liquefy: (mesh: Mesh, point: readonly [number, number, number] | null, config: LiquefiableBehaviorConfig) => void;
}

export interface Behavior<Name extends string = string> {
    readonly name: Name;
    readonly mesh: Mesh;
    start(): void;
    dispose(): void;
}
