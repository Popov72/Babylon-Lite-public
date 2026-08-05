import type { Mesh } from "babylon-lite";
import type { Behavior, WeaponBehaviorConfig } from "./types.js";

export class WeaponBehavior implements Behavior<"weapon"> {
    public readonly name = "weapon";
    public readonly mesh: Mesh;
    public readonly config: WeaponBehaviorConfig;

    public constructor(mesh: Mesh, config: WeaponBehaviorConfig) {
        this.mesh = mesh;
        this.config = config;
    }

    public start(): void {}

    public dispose(): void {}
}
