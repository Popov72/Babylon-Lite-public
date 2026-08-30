import type { Mesh, SceneNode } from "babylon-lite";
import { meshGroupAabbProvider, type MeshGroupAabb, type MeshGroupBounds } from "../mesh-bounds.js";
import type { AquanovaGameContext } from "./game-context.js";
import type { ManagedSound } from "./sound-manager.js";
import type { Behavior, PickEntityBehaviorConfig } from "./types.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import { aquanovaSoundUrl, validateAquanovaSoundName } from "./sound-asset.js";

const DEFAULT_SOUND = "pickItem";
const DEFAULT_BOUNDING_BOX_SCALE = [1, 1, 1] as const;
const DEFAULT_ROTATION_SPEED = (Math.PI * 2) / 3;

type PickEntityContext = Pick<AquanovaGameContext, "character" | "events" | "retireEntity" | "sounds">;
type BoundingBoxScale = readonly [number, number, number];
type RotationAxis = NonNullable<PickEntityBehaviorConfig["rotationAxis"]>;

export class PickEntityBehavior implements Behavior<"pickEntity"> {
    public readonly name = "pickEntity";
    public readonly mesh: Mesh;
    public readonly config: PickEntityBehaviorConfig;
    private readonly meshes: readonly Mesh[];
    private readonly rotationNodes: readonly SceneNode[];
    private readonly context: PickEntityContext;
    private readonly entityName: string;
    private readonly boundingBoxScale: BoundingBoxScale;
    private readonly rotationAxis: RotationAxis;
    private readonly rotationSpeed: number;
    private boundsProvider: (() => MeshGroupAabb | null) | null = null;
    private sound: ManagedSound | null = null;
    private stopPhysicsStep: (() => void) | null = null;
    private picked = false;

    public constructor(entityName: string, meshes: readonly Mesh[], config: PickEntityBehaviorConfig, context: PickEntityContext) {
        if (!meshes.length) {
            throw new Error("[aquanova] pickEntity requires at least one mesh");
        }
        assertBehaviorConfigKeys(config, "pickEntity", ["boundingBoxScale", "raiseEvent", "rotationAxis", "sound", "speed"]);
        validateEvent(config.raiseEvent);
        validateAquanovaSoundName("pickEntity sound", config.sound ?? DEFAULT_SOUND);
        this.mesh = meshes[0]!;
        this.meshes = meshes;
        this.rotationNodes = resolveRotationNodes(meshes, entityName);
        this.config = config;
        this.context = context;
        this.entityName = entityName;
        this.boundingBoxScale = resolveBoundingBoxScale(config.boundingBoxScale);
        this.rotationAxis = resolveRotationAxis(config.rotationAxis);
        this.rotationSpeed = DEFAULT_ROTATION_SPEED * resolveSpeed(config.speed);
    }

    public async init(): Promise<void> {
        const soundName = this.config.sound ?? DEFAULT_SOUND;
        const url = aquanovaSoundUrl(soundName);
        try {
            this.sound = await this.context.sounds.load(`pickEntity:${soundName}`, url, { preloadCount: 1 });
        } catch (error) {
            throw new Error(`[aquanova] failed to preload pickEntity sound "${soundName}" from "${url}"`, { cause: error });
        }
    }

    public start(): void {
        const boundsProvider = meshGroupAabbProvider(this.meshes);
        if (!boundsProvider()) {
            throw new Error("[aquanova] pickEntity target has no readable mesh geometry");
        }
        this.boundsProvider = boundsProvider;
        this.stopPhysicsStep = this.context.events.on("physicsStep", ({ deltaSeconds }) => this.update(deltaSeconds));
    }

    public dispose(): void {
        this.stopPhysicsStep?.();
        this.stopPhysicsStep = null;
    }

    private update(deltaSeconds: number): void {
        if (this.picked) {
            return;
        }
        const aabb = this.boundsProvider?.();
        if (!aabb) {
            return;
        }
        if (!playerIntersectsBounds(this.context, scaleBounds(aabb, this.boundingBoxScale))) {
            for (const node of this.rotationNodes) {
                node.rotation[this.rotationAxis] += this.rotationSpeed * deltaSeconds;
            }
            return;
        }
        this.picked = true;
        this.stopPhysicsStep?.();
        this.stopPhysicsStep = null;
        if (!this.sound) {
            throw new Error("[aquanova] pickEntity sound was not initialized");
        }
        this.context.sounds.play(this.sound);
        if (this.config.raiseEvent) {
            this.context.events.emit("entityEvent", {
                name: this.config.raiseEvent.target ?? this.entityName,
                event: this.config.raiseEvent.event,
            });
        }
        this.context.retireEntity(this.entityName, this.meshes);
    }
}

function validateEvent(event: PickEntityBehaviorConfig["raiseEvent"]): void {
    if (!event) {
        return;
    }
    assertBehaviorConfigKeys(event, "pickEntity.raiseEvent", ["target", "event"]);
    if (!event.event) {
        throw new Error("[aquanova] pickEntity.raiseEvent.event must be a non-empty event name");
    }
    if (event.target !== undefined && !event.target) {
        throw new Error("[aquanova] pickEntity.raiseEvent.target must be a non-empty entity or door name when provided");
    }
}

function resolveBoundingBoxScale(scale: PickEntityBehaviorConfig["boundingBoxScale"]): BoundingBoxScale {
    if (scale === undefined) {
        return DEFAULT_BOUNDING_BOX_SCALE;
    }
    if (scale.length !== 3 || scale.some((value) => !Number.isFinite(value) || value < 0)) {
        throw new Error(`[aquanova] pickEntity.boundingBoxScale must contain three finite non-negative values, received ${JSON.stringify(scale)}`);
    }
    return [scale[0]!, scale[1]!, scale[2]!];
}

function resolveRotationAxis(axis: PickEntityBehaviorConfig["rotationAxis"]): RotationAxis {
    if (axis === undefined) {
        return "y";
    }
    if (axis !== "x" && axis !== "y" && axis !== "z") {
        throw new Error(`[aquanova] pickEntity.rotationAxis must be "x", "y", or "z", received ${String(axis)}`);
    }
    return axis;
}

function resolveSpeed(speed: PickEntityBehaviorConfig["speed"]): number {
    if (speed === undefined) {
        return 1;
    }
    if (!Number.isFinite(speed) || speed <= 0) {
        throw new Error(`[aquanova] pickEntity.speed must be a finite positive value, received ${String(speed)}`);
    }
    return speed;
}

function resolveRotationNodes(meshes: readonly Mesh[], entityName: string): SceneNode[] {
    const nodes = new Set<SceneNode>();
    for (const mesh of meshes) {
        let node: SceneNode | null = mesh;
        let entityNode: SceneNode | null = null;
        while (node) {
            if (node.name === entityName) {
                entityNode = node;
                break;
            }
            node = isSceneNode(node.parent) ? node.parent : null;
        }
        nodes.add(entityNode ?? mesh);
    }
    return [...nodes];
}

function isSceneNode(value: Mesh["parent"]): value is SceneNode {
    return value !== null && "name" in value && "rotation" in value;
}

function scaleBounds(bounds: MeshGroupAabb, scale: BoundingBoxScale): MeshGroupBounds {
    const centre: [number, number, number] = [(bounds.min[0] + bounds.max[0]) * 0.5, (bounds.min[1] + bounds.max[1]) * 0.5, (bounds.min[2] + bounds.max[2]) * 0.5];
    return {
        centre,
        half: [(bounds.max[0] - centre[0]) * scale[0], (bounds.max[1] - centre[1]) * scale[1], (bounds.max[2] - centre[2]) * scale[2]],
    };
}

function playerIntersectsBounds(context: PickEntityContext, bounds: MeshGroupBounds): boolean {
    const position = context.character.getPosition();
    const shape = context.character.shapeOptions;
    const radius = shape.capsuleRadius;
    const halfHeight = shape.capsuleHeight * 0.5;
    const playerMin = [position.x - radius, position.y - halfHeight, position.z - radius] as const;
    const playerMax = [position.x + radius, position.y + halfHeight, position.z + radius] as const;
    for (let axis = 0; axis < 3; axis++) {
        const itemMin = bounds.centre[axis]! - bounds.half[axis]!;
        const itemMax = bounds.centre[axis]! + bounds.half[axis]!;
        if (playerMax[axis]! < itemMin || playerMin[axis]! > itemMax) {
            return false;
        }
    }
    return true;
}
