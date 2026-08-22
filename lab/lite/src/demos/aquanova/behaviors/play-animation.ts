import { pauseAnimation, playAnimation, stopAnimation, type AnimationGroup, type Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, PlayAnimationBehaviorConfig } from "./types.js";

type PlayAnimationContext = Pick<AquanovaGameContext, "animationGroups">;

export class PlayAnimationBehavior implements Behavior<"playAnimation"> {
    public readonly name = "playAnimation";
    public readonly mesh: Mesh;
    public readonly config: PlayAnimationBehaviorConfig;
    private readonly entityName: string;
    private readonly animationGroups: readonly AnimationGroup[];
    private selected: AnimationGroup | null = null;

    public constructor(entityName: string, meshes: readonly Mesh[], config: PlayAnimationBehaviorConfig, context: PlayAnimationContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] playAnimation requires at least one mesh");
        }
        if (config.animation !== undefined && (typeof config.animation !== "string" || config.animation.length === 0)) {
            throw new Error("[aquanova] playAnimation.animation must be a non-empty animation name");
        }
        if (config.loop !== undefined && typeof config.loop !== "boolean") {
            throw new Error("[aquanova] playAnimation.loop must be a boolean");
        }
        this.entityName = entityName;
        this.mesh = mesh;
        this.config = config;
        this.animationGroups = context.animationGroups;
    }

    public init(): void {}

    public start(): void {
        const requested = this.config.animation;
        const entityAnimations = this.animationGroups.filter((group) => animationTargetsEntity(group, this.entityName));
        const selected = requested === undefined ? entityAnimations[0] : entityAnimations.find((group) => group.name === requested);
        if (!selected) {
            if (requested === undefined) {
                return;
            }
            throw new Error(`[aquanova] playAnimation animation "${requested}" was not found`);
        }
        stopAnimation(selected);
        selected.loopAnimation = this.config.loop ?? true;
        playAnimation(selected);
        this.selected = selected;
    }

    public dispose(): void {
        if (this.selected) {
            stopAnimation(this.selected);
            this.selected = null;
        }
    }
}

export function pauseAnimationsTargetingEntities(animationGroups: readonly AnimationGroup[], entityNames: ReadonlySet<string>): AnimationGroup[] {
    const paused: AnimationGroup[] = [];
    for (const group of animationGroups) {
        if (!group.isPlaying) {
            continue;
        }
        for (const entityName of entityNames) {
            if (animationTargetsEntity(group, entityName)) {
                pauseAnimation(group);
                paused.push(group);
                break;
            }
        }
    }
    return paused;
}

export function resumeAnimations(animationGroups: readonly AnimationGroup[]): void {
    for (const group of animationGroups) {
        playAnimation(group);
    }
}

export function animationTargetsEntity(group: AnimationGroup, entityName: string): boolean {
    const childPrefix = `${entityName}_`;
    return group.targetedAnimations.some(({ targetName, target }) => {
        if (targetName === entityName || targetName?.startsWith(childPrefix)) {
            return true;
        }
        let node = target as { name?: unknown; parent?: unknown } | undefined;
        while (node) {
            if (node.name === entityName || (typeof node.name === "string" && node.name.startsWith(childPrefix))) {
                return true;
            }
            node = typeof node.parent === "object" && node.parent !== null ? (node.parent as { name?: unknown; parent?: unknown }) : undefined;
        }
        return false;
    });
}
