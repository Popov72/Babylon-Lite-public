import type { Mesh } from "babylon-lite";
import { meshGroupAabbProvider } from "../mesh-bounds.js";
import type { FluidElectricityRegistration } from "../fluid-runtime.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, ElectricalDetonatorBehaviorConfig } from "./types.js";

const DEFAULT_PARTICLE_THRESHOLD = 4;

type ElectricalDetonatorContext = Pick<AquanovaGameContext, "events" | "fluidSimulations">;

export class ElectricalDetonatorBehavior implements Behavior<"electricalDetonator"> {
    public readonly name = "electricalDetonator";
    public readonly mesh: Mesh;
    public readonly particleThreshold: number;
    private readonly entityName: string;
    private readonly context: ElectricalDetonatorContext;
    private readonly aabb: ReturnType<typeof meshGroupAabbProvider>;
    private registration: FluidElectricityRegistration | null = null;
    private detonated = false;

    public constructor(entityName: string, meshes: readonly Mesh[], config: ElectricalDetonatorBehaviorConfig, context: ElectricalDetonatorContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] electricalDetonator requires at least one mesh");
        }
        assertBehaviorConfigKeys(config, "electricalDetonator", ["particleThreshold"]);
        const particleThreshold = config.particleThreshold ?? DEFAULT_PARTICLE_THRESHOLD;
        if (!Number.isInteger(particleThreshold) || particleThreshold <= 0) {
            throw new Error("[aquanova] electricalDetonator.particleThreshold must be a positive integer");
        }
        this.entityName = entityName;
        this.mesh = mesh;
        this.particleThreshold = particleThreshold;
        this.context = context;
        this.aabb = meshGroupAabbProvider(meshes);
    }

    public init(): void {}

    public start(): void {
        this.registration = this.context.fluidSimulations.registerElectricityReceiver({
            entityName: this.entityName,
            particleThreshold: this.particleThreshold,
            aabb: this.aabb,
            onCount: (particleCount) => {
                if (this.detonated || particleCount < this.particleThreshold) {
                    return;
                }
                this.detonated = true;
                this.context.events.emit("entityEvent", { name: this.entityName, event: "explode" });
                this.registration?.dispose();
                this.registration = null;
            },
        });
    }

    public dispose(): void {
        this.registration?.dispose();
        this.registration = null;
    }
}
