import type { Mesh } from "babylon-lite";
import type { Behavior, DynamicBehaviorConfig } from "./types.js";

export const DEFAULT_DYNAMIC_MASS = 10;

export function resolveDynamicMass(config: DynamicBehaviorConfig): number {
    const mass = config.mass ?? DEFAULT_DYNAMIC_MASS;
    if (!Number.isFinite(mass) || mass <= 0) {
        throw new Error(`[aquanova] dynamic.mass must be a finite positive number, received ${String(mass)}`);
    }
    return mass;
}

export class DynamicBehavior implements Behavior<"dynamic"> {
    public readonly name = "dynamic";
    public readonly mesh: Mesh;
    public readonly config: DynamicBehaviorConfig;

    public constructor(mesh: Mesh, config: DynamicBehaviorConfig) {
        resolveDynamicMass(config);
        this.mesh = mesh;
        this.config = config;
    }

    public start(): void {}

    public dispose(): void {}
}
