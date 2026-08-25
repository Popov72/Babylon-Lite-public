import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, FluidSimShape, SetCollisionShapeBehaviorConfig } from "./types.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";

type SetCollisionShapeContext = Pick<AquanovaGameContext, "setCollisionShape">;

export function normalizeFluidSimShape(shape: unknown): FluidSimShape | undefined {
    if (shape === undefined) return undefined;
    if (!shape || typeof shape !== "object" || Array.isArray(shape)) {
        throw new Error("[aquanova] setCollisionShape.fluidSimShape must be an object");
    }
    const value = shape as Record<string, unknown>;
    assertBehaviorConfigKeys(value, "setCollisionShape.fluidSimShape", ["type", "start", "height", "innerRadius", "outerRadius"]);
    if (value.type !== "hollowCylinder") {
        throw new Error('[aquanova] setCollisionShape.fluidSimShape.type must be "hollowCylinder"');
    }
    const start = value.start;
    if (!Array.isArray(start) || start.length !== 3 || !start.every((component) => typeof component === "number" && Number.isFinite(component))) {
        throw new Error("[aquanova] setCollisionShape.fluidSimShape.start must contain three finite numbers");
    }
    const height = value.height;
    if (typeof height !== "number" || !Number.isFinite(height) || height <= 0) {
        throw new Error("[aquanova] setCollisionShape.fluidSimShape.height must be positive");
    }
    const innerRadius = value.innerRadius;
    if (typeof innerRadius !== "number" || !Number.isFinite(innerRadius) || innerRadius <= 0) {
        throw new Error("[aquanova] setCollisionShape.fluidSimShape.innerRadius must be positive");
    }
    const outerRadius = value.outerRadius;
    if (typeof outerRadius !== "number" || !Number.isFinite(outerRadius) || outerRadius <= innerRadius) {
        throw new Error("[aquanova] setCollisionShape.fluidSimShape.outerRadius must be greater than innerRadius");
    }
    return {
        type: "hollowCylinder",
        start: [start[0], start[1], start[2]],
        height,
        innerRadius,
        outerRadius,
    };
}

export class SetCollisionShapeBehavior implements Behavior<"setCollisionShape"> {
    public readonly name = "setCollisionShape";
    public readonly mesh: Mesh;
    public readonly config: SetCollisionShapeBehaviorConfig;
    private readonly entityName: string;
    private readonly meshes: readonly Mesh[];
    private readonly type: "aabb" | "mesh";
    private readonly fluidSimShape: FluidSimShape | undefined;
    private readonly context: SetCollisionShapeContext;

    public constructor(entityName: string, meshes: readonly Mesh[], config: SetCollisionShapeBehaviorConfig, context: SetCollisionShapeContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] setCollisionShape requires at least one mesh");
        }
        assertBehaviorConfigKeys(config, "setCollisionShape", ["type", "fluidSimShape"]);
        const type = config.type;
        if (type !== undefined && type !== "mesh") {
            throw new Error('[aquanova] setCollisionShape.type must be "mesh" when provided');
        }
        this.entityName = entityName;
        this.meshes = meshes;
        this.type = type ?? "aabb";
        this.fluidSimShape = normalizeFluidSimShape(config.fluidSimShape);
        this.mesh = mesh;
        this.config = config;
        this.context = context;
    }

    public init(): void {}

    public start(): void {
        if (this.fluidSimShape) {
            this.context.setCollisionShape(this.entityName, this.type, { meshes: this.meshes, shape: this.fluidSimShape });
        } else {
            this.context.setCollisionShape(this.entityName, this.type);
        }
    }

    public dispose(): void {}
}
