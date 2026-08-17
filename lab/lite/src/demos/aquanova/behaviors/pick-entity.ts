import { createAudioEngineAsync, createStreamingSoundAsync, disposeAudioEngine, playStreamingSound, preloadStreamingInstanceAsync, setMeshVisible } from "babylon-lite";
import type { AudioEngine, Mesh, SceneNode, StreamingSound } from "babylon-lite";
import { meshGroupBounds, type MeshGroupBounds } from "../mesh-bounds.js";
import type { Behavior, BehaviorContext, PickEntityBehaviorConfig } from "./types.js";

const SOUND_ROOT = "/aquanova/sounds";
const SOUND_ASSET_VERSION = "20260813-1";
const DEFAULT_SOUND = "pickItem";
const DEFAULT_BOUNDING_BOX_SCALE = [1, 1, 1] as const;
const DEFAULT_ROTATION_SPEED = (Math.PI * 2) / 3;

type PickEntityContext = Pick<BehaviorContext, "character" | "events">;
type BoundingBoxScale = readonly [number, number, number];

export class PickEntityBehavior implements Behavior<"pickEntity"> {
    private static initialization: Promise<void> | null = null;
    private static audioEngine: AudioEngine | null = null;
    private static sounds: Map<string, StreamingSound> | null = null;
    private static soundEnabled = true;
    public readonly name = "pickEntity";
    public readonly mesh: Mesh;
    public readonly config: PickEntityBehaviorConfig;
    private readonly meshes: readonly Mesh[];
    private readonly rotationNodes: readonly SceneNode[];
    private readonly context: PickEntityContext;
    private readonly boundingBoxScale: BoundingBoxScale;
    private readonly rotationSpeed: number;
    private bounds: MeshGroupBounds | null = null;
    private stopPhysicsStep: (() => void) | null = null;
    private picked = false;

    public constructor(meshes: readonly Mesh[], config: PickEntityBehaviorConfig, context: PickEntityContext, entityName = meshes[0]?.name ?? "") {
        if (!meshes.length) {
            throw new Error("[aquanova] pickEntity requires at least one mesh");
        }
        validateEvent(config.raiseEvent);
        this.mesh = meshes[0]!;
        this.meshes = meshes;
        this.rotationNodes = resolveRotationNodes(meshes, entityName);
        this.config = config;
        this.context = context;
        this.boundingBoxScale = resolveBoundingBoxScale(config.boundingBoxScale);
        this.rotationSpeed = DEFAULT_ROTATION_SPEED * resolveSpeed(config.speed);
    }

    public static init(configs: readonly PickEntityBehaviorConfig[]): Promise<void> {
        if (!this.initialization) {
            this.initialization = this.initialize(configs).catch((error: unknown) => {
                this.initialization = null;
                throw error;
            });
        }
        return this.initialization;
    }

    public static dispose(): void {
        if (this.audioEngine) {
            disposeAudioEngine(this.audioEngine);
        }
        this.audioEngine = null;
        this.sounds = null;
        this.soundEnabled = true;
        this.initialization = null;
    }

    public static setSoundEnabled(enabled: boolean): void {
        this.soundEnabled = enabled;
    }

    private static async initialize(configs: readonly PickEntityBehaviorConfig[]): Promise<void> {
        const soundNames = new Set<string>();
        if (!configs.length) {
            this.sounds = new Map();
            return;
        }
        for (const config of configs) {
            const soundName = config.sound ?? DEFAULT_SOUND;
            validateSoundName(soundName);
            soundNames.add(soundName);
        }

        const engine = await createAudioEngineAsync();
        try {
            const sounds = new Map<string, StreamingSound>();
            for (const soundName of soundNames) {
                const url = soundUrl(soundName);
                try {
                    sounds.set(soundName, await createStreamingSoundAsync(engine, url, { preloadCount: 1 }));
                } catch (error) {
                    throw new Error(`[aquanova] failed to preload pickEntity sound "${soundName}" from "${url}"`, { cause: error });
                }
            }
            this.sounds = sounds;
            this.audioEngine = engine;
        } catch (error) {
            disposeAudioEngine(engine);
            throw error;
        }
    }

    public start(): void {
        const bounds = meshGroupBounds(this.meshes);
        if (!bounds) {
            throw new Error("[aquanova] pickEntity target has no readable mesh geometry");
        }
        this.bounds = scaleBounds(bounds, this.boundingBoxScale);
        this.stopPhysicsStep = this.context.events.on("physicsStep", ({ deltaSeconds }) => this.update(deltaSeconds));
    }

    public dispose(): void {
        this.stopPhysicsStep?.();
        this.stopPhysicsStep = null;
    }

    private update(deltaSeconds: number): void {
        if (this.picked || !this.bounds) {
            return;
        }
        if (!playerIntersectsBounds(this.context, this.bounds)) {
            for (const node of this.rotationNodes) {
                node.rotation.y += this.rotationSpeed * deltaSeconds;
            }
            return;
        }
        this.picked = true;
        this.stopPhysicsStep?.();
        this.stopPhysicsStep = null;
        for (const mesh of this.meshes) {
            setMeshVisible(mesh, false);
        }
        PickEntityBehavior.playSound(this.config.sound ?? DEFAULT_SOUND);
        if (this.config.raiseEvent) {
            this.context.events.emit("entityEvent", this.config.raiseEvent);
        }
    }

    private static playSound(soundName: string): void {
        if (!this.soundEnabled) {
            return;
        }
        const sound = this.sounds?.get(soundName);
        if (!sound) {
            throw new Error(`[aquanova] pickEntity sound "${soundName}" was not preloaded`);
        }
        playStreamingSound(sound);
        void preloadStreamingInstanceAsync(sound).catch((error: unknown) => {
            console.warn(`[aquanova] failed to replenish preloaded pickEntity sound "${soundName}"`, error);
        });
    }
}

function validateEvent(event: PickEntityBehaviorConfig["raiseEvent"]): void {
    if (!event) {
        return;
    }
    if (!event.name || !event.event) {
        throw new Error("[aquanova] pickEntity.raiseEvent requires non-empty name and event values");
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

function scaleBounds(bounds: MeshGroupBounds, scale: BoundingBoxScale): MeshGroupBounds {
    return {
        centre: bounds.centre,
        half: [bounds.half[0] * scale[0], bounds.half[1] * scale[1], bounds.half[2] * scale[2]],
    };
}

function validateSoundName(soundName: string): void {
    if (!soundName || soundName.endsWith(".mp3") || soundName.includes("/") || soundName.includes("\\")) {
        throw new Error(`[aquanova] pickEntity sound "${soundName}" must be an MP3 file name without its extension`);
    }
}

function soundUrl(soundName: string): string {
    return `${SOUND_ROOT}/${encodeURIComponent(soundName)}.mp3?v=${SOUND_ASSET_VERSION}`;
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
