import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, LiquefiableBehaviorConfig } from "./types.js";

/** Data-driven weapon-hit handling for every resolved `liquefaction` behavior. */
export class LiquefiableBehavior implements Behavior {
    public readonly name: string;
    public readonly mesh: Mesh;
    protected readonly config: LiquefiableBehaviorConfig;
    protected readonly context: AquanovaGameContext;
    private stopHit: (() => void) | null = null;

    public constructor(_entityName: string, meshes: readonly Mesh[], config: LiquefiableBehaviorConfig & { name: string }, context: AquanovaGameContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] liquefiable behavior requires at least one mesh");
        }
        if (config.electrifiable !== undefined && typeof config.electrifiable !== "boolean") {
            throw new Error("[aquanova] liquefaction.electrifiable must be true or false");
        }
        this.name = config.name;
        this.mesh = mesh;
        this.config = config;
        this.context = context;
    }

    public init(): void {}

    public start(): void {
        this.stopHit = this.context.events.on("hitWithWeapon", ({ mesh, point }) => {
            if (this.context.isLiquefiable(mesh) && this.context.getLiquefiableConfig(mesh) === this.config) {
                this.context.liquefy(mesh, point, this.config);
            }
        });
    }

    public dispose(): void {
        this.stopHit?.();
        this.stopHit = null;
    }
}
