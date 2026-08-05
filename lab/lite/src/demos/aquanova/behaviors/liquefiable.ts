import type { Mesh } from "babylon-lite";
import type { Behavior, BehaviorContext, LiquefiableBehaviorConfig } from "./types.js";

/** Data-driven weapon-hit handling for every manifest behavior marked `liquefiable: true`. */
export class LiquefiableBehavior implements Behavior {
    public readonly name: string;
    public readonly mesh: Mesh;
    protected readonly config: LiquefiableBehaviorConfig;
    protected readonly context: BehaviorContext;
    private stopHit: (() => void) | null = null;

    public constructor(name: string, mesh: Mesh, config: LiquefiableBehaviorConfig, context: BehaviorContext) {
        this.name = name;
        this.mesh = mesh;
        this.config = config;
        this.context = context;
    }

    public start(): void {
        this.stopHit = this.context.events.on("hitWithWeapon", ({ mesh, point }) => {
            if (mesh === this.mesh && this.context.isLiquefiable(mesh)) {
                this.context.liquefy(this.mesh, point, this.config);
            }
        });
    }

    public dispose(): void {
        this.stopHit?.();
        this.stopHit = null;
    }
}
