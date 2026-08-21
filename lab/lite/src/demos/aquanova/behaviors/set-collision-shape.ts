import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, SetCollisionShapeBehaviorConfig } from "./types.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";

type SetCollisionShapeContext = Pick<AquanovaGameContext, "setCollisionShape">;

export class SetCollisionShapeBehavior implements Behavior<"setCollisionShape"> {
    public readonly name = "setCollisionShape";
    public readonly mesh: Mesh;
    public readonly config: SetCollisionShapeBehaviorConfig;
    private readonly entityName: string;
    private readonly type: "aabb" | "mesh";
    private readonly context: SetCollisionShapeContext;

    public constructor(entityName: string, meshes: readonly Mesh[], config: SetCollisionShapeBehaviorConfig, context: SetCollisionShapeContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] setCollisionShape requires at least one mesh");
        }
        assertBehaviorConfigKeys(config, "setCollisionShape", ["type"]);
        const type = config.type;
        if (type !== undefined && type !== "mesh") {
            throw new Error('[aquanova] setCollisionShape.type must be "mesh" when provided');
        }
        this.entityName = entityName;
        this.type = type ?? "aabb";
        this.mesh = mesh;
        this.config = config;
        this.context = context;
    }

    public init(): void {}

    public start(): void {
        this.context.setCollisionShape(this.entityName, this.type);
    }

    public dispose(): void {}
}
