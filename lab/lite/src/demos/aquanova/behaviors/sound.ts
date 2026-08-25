import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { ManagedSound } from "./sound-manager.js";
import type { Behavior, SoundBehaviorConfig, SoundCueConfig } from "./types.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import { eventSubscriptionMatches, validateEventSubscriptions } from "./event-subscription.js";

const SOUND_ROOT = "/aquanova/sounds";
const SOUND_ASSET_VERSION = "20260813-1";

type SoundContext = Pick<AquanovaGameContext, "events" | "sounds">;

interface SoundCue {
    readonly config: SoundCueConfig;
    readonly soundName: string;
    readonly delay: number;
    readonly fade: number;
    sound: ManagedSound | null;
}

export class SoundBehavior implements Behavior<"sound"> {
    public readonly name = "sound";
    public readonly mesh: Mesh | null;
    public readonly config: SoundBehaviorConfig;
    private readonly context: SoundContext;
    private readonly cues: SoundCue[];
    private stopEntityEvent: (() => void) | null = null;
    private readonly timers = new Set<ReturnType<typeof setTimeout>>();
    private disposed = false;

    public constructor(_entityName: string, meshes: readonly Mesh[], config: SoundBehaviorConfig, context: SoundContext) {
        assertBehaviorConfigKeys(config, "sound", ["cues"]);
        if (!Array.isArray(config.cues) || config.cues.length === 0) {
            throw new Error("[aquanova] sound.cues must contain at least one cue");
        }
        this.mesh = meshes[0] ?? null;
        this.config = config;
        this.context = context;
        this.cues = config.cues.map((cue) => {
            if (!cue || typeof cue !== "object" || Array.isArray(cue)) {
                throw new Error("[aquanova] sound.cues[] must be an object");
            }
            assertBehaviorConfigKeys(cue, "sound.cues[]", ["action", "delay", "events", "fade", "sound"]);
            validateSoundName(cue.sound);
            validateEventSubscriptions("sound.cues[]", cue.events);
            if (cue.action !== "play" && cue.action !== "stop") {
                throw new Error('[aquanova] sound.cues[].action must be "play" or "stop"');
            }
            return {
                config: cue,
                soundName: cue.sound,
                delay: nonNegativeSeconds("delay", cue.delay),
                fade: nonNegativeSeconds("fade", cue.fade),
                sound: null,
            };
        });
    }

    public async init(): Promise<void> {
        const sounds = new Map<string, ManagedSound>();
        await Promise.all(
            [...new Set(this.cues.map((cue) => cue.soundName))].map(async (soundName) => {
                const url = `${SOUND_ROOT}/${encodeURIComponent(soundName)}.mp3?v=${SOUND_ASSET_VERSION}`;
                try {
                    sounds.set(soundName, await this.context.sounds.load(`sound:${soundName}`, url, { preloadCount: 1 }));
                } catch (error) {
                    throw new Error(`[aquanova] failed to preload sound behavior sound "${soundName}" from "${url}"`, { cause: error });
                }
            })
        );
        for (const cue of this.cues) {
            cue.sound = sounds.get(cue.soundName) ?? null;
        }
    }

    public start(): void {
        if (this.cues.some((cue) => !cue.sound)) {
            throw new Error("[aquanova] sound behavior was not initialized");
        }
        for (const cue of this.cues) {
            if (!cue.config.events) {
                this.scheduleAction(cue);
            }
        }
        if (!this.cues.some((cue) => cue.config.events)) {
            return;
        }
        this.stopEntityEvent = this.context.events.on("entityEvent", ({ name, event }) => {
            for (const cue of this.cues) {
                if (cue.config.events?.some((subscription) => eventSubscriptionMatches(subscription, name, event))) {
                    this.scheduleAction(cue);
                }
            }
        });
    }

    public dispose(): void {
        this.disposed = true;
        this.stopEntityEvent?.();
        this.stopEntityEvent = null;
        for (const timer of this.timers) {
            clearTimeout(timer);
        }
        this.timers.clear();
        const sounds = new Set(this.cues.flatMap((cue) => (cue.sound ? [cue.sound] : [])));
        for (const sound of sounds) {
            this.context.sounds.stop(sound);
        }
    }

    private scheduleAction(cue: SoundCue): void {
        if (cue.delay === 0) {
            this.runAction(cue);
            return;
        }
        const timer = setTimeout(() => {
            this.timers.delete(timer);
            if (!this.disposed) {
                this.runAction(cue);
            }
        }, cue.delay * 1000);
        this.timers.add(timer);
    }

    private runAction(cue: SoundCue): void {
        const sound = cue.sound;
        if (!sound) {
            throw new Error("[aquanova] sound behavior was not initialized");
        }
        if (cue.config.action === "play") {
            this.context.sounds.play(sound, { fade: cue.fade });
        } else {
            this.context.sounds.stop(sound, cue.fade);
        }
    }
}

function validateSoundName(soundName: string): void {
    if (!soundName || soundName.endsWith(".mp3") || soundName.includes("/") || soundName.includes("\\")) {
        throw new Error(`[aquanova] sound.sound "${soundName}" must be an MP3 file name without its extension`);
    }
}

function nonNegativeSeconds(property: "delay" | "fade", value: number | undefined): number {
    const seconds = value ?? 0;
    if (!Number.isFinite(seconds) || seconds < 0) {
        throw new Error(`[aquanova] sound.cues[].${property} must be finite and non-negative`);
    }
    return seconds;
}
