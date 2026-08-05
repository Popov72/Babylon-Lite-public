import type { Mesh } from "babylon-lite";
import type { Behavior, DynamicBehaviorConfig } from "./types.js";

export class DynamicBehavior implements Behavior<"dynamic"> {
    public readonly name = "dynamic";
    public readonly mesh: Mesh;
    public readonly config: DynamicBehaviorConfig;

    public constructor(mesh: Mesh, config: DynamicBehaviorConfig) {
        this.mesh = mesh;
        this.config = config;
    }

    public start(): void {}

    public dispose(): void {}
}
