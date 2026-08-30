import type { Mesh } from "babylon-lite";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import { eventSubscriptionMatches, validateEventSubscriptions } from "./event-subscription.js";
import type { AquanovaGameContext, ExplosionOptions } from "./game-context.js";
import { aquanovaSoundUrl, validateAquanovaSoundName } from "./sound-asset.js";
import type { ManagedSound } from "./sound-manager.js";
import type { Behavior, BehaviorEventSubscription, ExplodeBehaviorConfig } from "./types.js";

const DEFAULT_RADIUS = 10;
const DEFAULT_FRAGMENT_COUNT = 8;
const DEFAULT_STRENGTH = 12;
const DEFAULT_DEBRIS_LIFETIME = 15;
const DEFAULT_FADE_DURATION = 2;
const DEFAULT_SOUND = "bigExplosion";

type ExplodeContext = Pick<AquanovaGameContext, "events" | "explosions" | "sounds">;

function finitePositive(value: number, path: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`[aquanova] ${path} must be a finite positive number`);
    }
    return value;
}

export class ExplodeBehavior implements Behavior<"explode"> {
    public readonly name = "explode";
    public readonly mesh: Mesh;
    public readonly config: ExplodeBehaviorConfig;
    public readonly fragmentCount: number;
    private readonly entityName: string;
    private readonly meshes: readonly Mesh[];
    private readonly context: ExplodeContext;
    private readonly subscriptions: readonly BehaviorEventSubscription[];
    private readonly options: ExplosionOptions;
    private sound: ManagedSound | null = null;
    private stopEntityEvent: (() => void) | null = null;
    private exploded = false;

    public constructor(entityName: string, meshes: readonly Mesh[], config: ExplodeBehaviorConfig, context: ExplodeContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] explode requires at least one mesh");
        }
        assertBehaviorConfigKeys(config, "explode", ["events", "radius", "fragmentCount", "strength", "debrisLifetime", "fadeDuration", "sound"]);
        validateEventSubscriptions("explode", config.events);
        validateAquanovaSoundName("explode sound", config.sound ?? DEFAULT_SOUND);
        const fragmentCount = config.fragmentCount ?? DEFAULT_FRAGMENT_COUNT;
        if (!Number.isInteger(fragmentCount) || fragmentCount < 2 || fragmentCount > 32) {
            throw new Error("[aquanova] explode.fragmentCount must be an integer from 2 through 32");
        }
        const debrisLifetime = config.debrisLifetime ?? DEFAULT_DEBRIS_LIFETIME;
        if (!Number.isFinite(debrisLifetime) || debrisLifetime < 0) {
            throw new Error("[aquanova] explode.debrisLifetime must be a finite non-negative number");
        }

        this.entityName = entityName;
        this.mesh = mesh;
        this.meshes = meshes;
        this.config = config;
        this.fragmentCount = fragmentCount;
        this.context = context;
        this.subscriptions = config.events ?? [{ name: "explode", source: entityName }];
        this.options = {
            radius: finitePositive(config.radius ?? DEFAULT_RADIUS, "explode.radius"),
            fragmentCount: this.fragmentCount,
            strength: finitePositive(config.strength ?? DEFAULT_STRENGTH, "explode.strength"),
            debrisLifetime,
            fadeDuration: finitePositive(config.fadeDuration ?? DEFAULT_FADE_DURATION, "explode.fadeDuration"),
        };
    }

    public async init(): Promise<void> {
        const soundName = this.config.sound ?? DEFAULT_SOUND;
        const url = aquanovaSoundUrl(soundName);
        try {
            this.sound = await this.context.sounds.load(`explode:${soundName}`, url, { preloadCount: 1 });
        } catch (error) {
            throw new Error(`[aquanova] failed to preload explode sound "${soundName}" from "${url}"`, { cause: error });
        }
    }

    public start(): void {
        this.stopEntityEvent = this.context.events.on("entityEvent", ({ name, event }) => {
            if (this.exploded || !this.subscriptions.some((subscription) => eventSubscriptionMatches(subscription, name, event))) {
                return;
            }
            this.exploded = true;
            if (!this.sound) {
                throw new Error("[aquanova] explode sound was not initialized");
            }
            this.context.sounds.play(this.sound);
            this.context.explosions.explode(this.entityName, this.meshes, this.options);
        });
    }

    public dispose(): void {
        this.stopEntityEvent?.();
        this.stopEntityEvent = null;
    }
}
