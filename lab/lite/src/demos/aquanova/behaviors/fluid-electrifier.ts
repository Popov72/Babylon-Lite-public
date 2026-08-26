import type { Mesh } from "babylon-lite";
import { meshGroupAabbProvider } from "../mesh-bounds.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, FluidElectrifierBehaviorConfig } from "./types.js";
import type { FluidElectricityRegistration } from "../fluid-runtime.js";

const DEFAULT_PARTICLE_THRESHOLD = 24;
const DEFAULT_PROPAGATION_SPEED = 8;

type FluidElectrifierContext = Pick<AquanovaGameContext, "events" | "fluidSimulations">;

export class FluidElectrifierBehavior implements Behavior<"fluidElectrifier"> {
    public readonly name = "fluidElectrifier";
    public readonly mesh: Mesh;
    public readonly particleThreshold: number;
    public readonly propagationSpeed: number;
    private readonly entityName: string;
    private readonly context: FluidElectrifierContext;
    private readonly aabb: ReturnType<typeof meshGroupAabbProvider>;
    private registration: FluidElectricityRegistration | null = null;

    public constructor(entityName: string, meshes: readonly Mesh[], config: FluidElectrifierBehaviorConfig, context: FluidElectrifierContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] fluidElectrifier requires at least one mesh");
        }
        assertBehaviorConfigKeys(config, "fluidElectrifier", ["particleThreshold", "propagationSpeed"]);
        this.particleThreshold = positiveInteger(config.particleThreshold ?? DEFAULT_PARTICLE_THRESHOLD, "particleThreshold");
        this.propagationSpeed = positiveFinite(config.propagationSpeed ?? DEFAULT_PROPAGATION_SPEED, "propagationSpeed");
        this.entityName = entityName;
        this.mesh = mesh;
        this.context = context;
        this.aabb = meshGroupAabbProvider(meshes);
    }

    public init(): void {}

    public start(): void {
        this.registration = this.context.fluidSimulations.registerElectrifier({
            entityName: this.entityName,
            particleThreshold: this.particleThreshold,
            propagationSpeed: this.propagationSpeed,
            aabb: this.aabb,
            onElectrified: () => {
                this.context.events.emit("entityEvent", { name: this.entityName, event: "fluidElectrified" });
            },
        });
    }

    public dispose(): void {
        this.registration?.dispose();
        this.registration = null;
    }
}

function positiveInteger(value: number, name: string): number {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`[aquanova] fluidElectrifier.${name} must be a positive integer`);
    }
    return value;
}

function positiveFinite(value: number, name: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`[aquanova] fluidElectrifier.${name} must be finite and positive`);
    }
    return value;
}
