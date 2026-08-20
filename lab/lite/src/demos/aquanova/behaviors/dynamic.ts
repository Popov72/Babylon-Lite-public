import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
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

    public constructor(_entityName: string, meshes: readonly Mesh[], config: DynamicBehaviorConfig, _context: AquanovaGameContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] dynamic requires at least one mesh");
        }
        resolveDynamicMass(config);
        this.mesh = mesh;
        this.config = config;
    }

    public init(): void {}

    public start(): void {}

    public dispose(): void {}
}
