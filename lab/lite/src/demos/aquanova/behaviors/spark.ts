import type { Mesh } from "babylon-lite";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import type { AquanovaGameContext, SparkOptions, SparkRegistration } from "./game-context.js";
import type { Behavior, SparkBehaviorConfig } from "./types.js";

const DEFAULT_RATE = 30;
const DEFAULT_SPEED = 2.5;
const DEFAULT_LIFETIME = 0.45;
const DEFAULT_SIZE = 0.035;
const DEFAULT_SPREAD = 0.08;
const DEFAULT_GRAVITY = 9.81;

type SparkContext = Pick<AquanovaGameContext, "sparks">;

function finitePositive(value: number, path: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`[aquanova] ${path} must be a finite positive number`);
    }
    return value;
}

function finiteNonNegative(value: number, path: string): number {
    if (!Number.isFinite(value) || value < 0) {
        throw new Error(`[aquanova] ${path} must be a finite non-negative number`);
    }
    return value;
}

export class SparkBehavior implements Behavior<"spark"> {
    public readonly name = "spark";
    public readonly mesh: Mesh;
    public readonly config: SparkBehaviorConfig;
    public readonly options: SparkOptions;
    private readonly context: SparkContext;
    private registration: SparkRegistration | null = null;

    public constructor(_entityName: string, meshes: readonly Mesh[], config: SparkBehaviorConfig, context: SparkContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] spark requires at least one mesh");
        }
        assertBehaviorConfigKeys(config, "spark", ["rate", "speed", "lifetime", "size", "spread", "gravity"]);

        this.mesh = mesh;
        this.config = config;
        this.context = context;
        this.options = {
            rate: finitePositive(config.rate ?? DEFAULT_RATE, "spark.rate"),
            speed: finitePositive(config.speed ?? DEFAULT_SPEED, "spark.speed"),
            lifetime: finitePositive(config.lifetime ?? DEFAULT_LIFETIME, "spark.lifetime"),
            size: finitePositive(config.size ?? DEFAULT_SIZE, "spark.size"),
            spread: finiteNonNegative(config.spread ?? DEFAULT_SPREAD, "spark.spread"),
            gravity: finiteNonNegative(config.gravity ?? DEFAULT_GRAVITY, "spark.gravity"),
        };
    }

    public init(): void {}

    public start(): void {
        this.registration = this.context.sparks.register(this.mesh, this.options);
    }

    public dispose(): void {
        this.registration?.dispose();
        this.registration = null;
    }
}
