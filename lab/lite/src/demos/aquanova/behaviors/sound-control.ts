import type { Mesh } from "babylon-lite";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import type { AquanovaGameContext } from "./game-context.js";
import type { ManagedSound } from "./sound-manager.js";
import type { Behavior, PlaySoundBehaviorConfig, SoundEventTriggerConfig, StopSoundBehaviorConfig } from "./types.js";

const SOUND_ROOT = "/aquanova/sounds";
const SOUND_ASSET_VERSION = "20260813-1";

type SoundControlContext = Pick<AquanovaGameContext, "events" | "sounds">;

class SoundControl {
    private readonly behaviorName: "playSound" | "stopSound";
    private readonly trigger: SoundEventTriggerConfig;
    private readonly context: SoundControlContext;
    private readonly resolveSound: () => Promise<ManagedSound>;
    private readonly action: (sound: ManagedSound) => void;
    private sound: ManagedSound | null = null;
    private stopEntityEvent: (() => void) | null = null;

    public constructor(
        behaviorName: "playSound" | "stopSound",
        trigger: SoundEventTriggerConfig,
        context: SoundControlContext,
        resolveSound: () => Promise<ManagedSound>,
        action: (sound: ManagedSound) => void
    ) {
        validateTrigger(behaviorName, trigger);
        this.behaviorName = behaviorName;
        this.trigger = trigger;
        this.context = context;
        this.resolveSound = resolveSound;
        this.action = action;
    }

    public async init(): Promise<void> {
        try {
            this.sound = await this.resolveSound();
        } catch (error) {
            throw new Error(`[aquanova] failed to initialize ${this.behaviorName}`, { cause: error });
        }
    }

    public start(): void {
        if (!this.sound) {
            throw new Error(`[aquanova] ${this.behaviorName} behavior was not initialized`);
        }
        if (this.trigger.source === undefined) {
            this.action(this.sound);
            return;
        }
        this.stopEntityEvent = this.context.events.on("entityEvent", ({ name, event }) => {
            if (event === this.trigger.event && sourceMatches(this.trigger.source!, name)) {
                this.action(this.sound!);
            }
        });
    }

    public dispose(stopPlayback: boolean): void {
        this.stopEntityEvent?.();
        this.stopEntityEvent = null;
        if (stopPlayback && this.sound) {
            this.context.sounds.stop(this.sound);
        }
        this.sound = null;
    }
}

export class PlaySoundBehavior implements Behavior<"playSound"> {
    public readonly name = "playSound";
    public readonly mesh: Mesh | null;
    public readonly config: PlaySoundBehaviorConfig;
    private readonly control: SoundControl;

    public constructor(_entityName: string, meshes: readonly Mesh[], config: PlaySoundBehaviorConfig, context: SoundControlContext) {
        assertBehaviorConfigKeys(config, "playSound", ["event", "fadeInDelay", "id", "loop", "sound", "source", "volume"]);
        validatePlaybackId("playSound.id", config.id);
        validateSoundName(config.sound);
        const fadeInDelay = nonNegativeSeconds("playSound.fadeInDelay", config.fadeInDelay);
        const volume = soundVolume(config.volume);
        if (config.loop !== undefined && typeof config.loop !== "boolean") {
            throw new Error("[aquanova] playSound.loop must be true or false");
        }
        this.mesh = meshes[0] ?? null;
        this.config = config;
        const url = `${SOUND_ROOT}/${encodeURIComponent(config.sound)}.mp3?v=${SOUND_ASSET_VERSION}`;
        context.sounds.registerPlayback(config.id, url, { preloadCount: 1 });
        this.control = new SoundControl(
            "playSound",
            config,
            context,
            () => context.sounds.resolvePlayback(config.id),
            (sound) => {
                context.sounds.play(sound, { fade: fadeInDelay, loop: config.loop ?? false, volume });
            }
        );
    }

    public init(): Promise<void> {
        return this.control.init();
    }

    public start(): void {
        this.control.start();
    }

    public dispose(): void {
        this.control.dispose(true);
    }
}

export class StopSoundBehavior implements Behavior<"stopSound"> {
    public readonly name = "stopSound";
    public readonly mesh: Mesh | null;
    public readonly config: StopSoundBehaviorConfig;
    private readonly control: SoundControl;

    public constructor(_entityName: string, meshes: readonly Mesh[], config: StopSoundBehaviorConfig, context: SoundControlContext) {
        assertBehaviorConfigKeys(config, "stopSound", ["event", "fadeOutDelay", "soundId", "source"]);
        validatePlaybackId("stopSound.soundId", config.soundId);
        const fadeOutDelay = nonNegativeSeconds("stopSound.fadeOutDelay", config.fadeOutDelay);
        this.mesh = meshes[0] ?? null;
        this.config = config;
        this.control = new SoundControl(
            "stopSound",
            config,
            context,
            () => context.sounds.resolvePlayback(config.soundId),
            (sound) => {
                context.sounds.stop(sound, fadeOutDelay);
            }
        );
    }

    public init(): Promise<void> {
        return this.control.init();
    }

    public start(): void {
        this.control.start();
    }

    public dispose(): void {
        this.control.dispose(false);
    }
}

function validateSoundName(soundName: string): void {
    if (typeof soundName !== "string" || !soundName || soundName.endsWith(".mp3") || soundName.includes("/") || soundName.includes("\\")) {
        throw new Error(`[aquanova] playSound.sound "${soundName}" must be an MP3 file name without its extension`);
    }
}

function validatePlaybackId(property: string, id: string): void {
    if (typeof id !== "string" || !id.trim()) {
        throw new Error(`[aquanova] ${property} must be a non-empty sound playback ID`);
    }
}

function validateTrigger(behaviorName: "playSound" | "stopSound", trigger: SoundEventTriggerConfig): void {
    if ((trigger.source === undefined) !== (trigger.event === undefined)) {
        throw new Error(`[aquanova] ${behaviorName}.source and ${behaviorName}.event must be provided together`);
    }
    if (trigger.source === undefined) {
        return;
    }
    if (typeof trigger.source === "string") {
        if (!trigger.source.trim()) {
            throw new Error(`[aquanova] ${behaviorName}.source must be a non-empty entity or door name`);
        }
    } else if (!Array.isArray(trigger.source) || trigger.source.length === 0 || trigger.source.some((source) => typeof source !== "string" || !source.trim())) {
        throw new Error(`[aquanova] ${behaviorName}.source must contain at least one non-empty entity or door name`);
    }
    if (typeof trigger.event !== "string" || !trigger.event.trim()) {
        throw new Error(`[aquanova] ${behaviorName}.event must be a non-empty event name`);
    }
}

function nonNegativeSeconds(property: string, value: number | undefined): number {
    const seconds = value ?? 0;
    if (!Number.isFinite(seconds) || seconds < 0) {
        throw new Error(`[aquanova] ${property} must be finite and non-negative`);
    }
    return seconds;
}

function soundVolume(value: number | undefined): number {
    const volume = value ?? 1;
    if (!Number.isFinite(volume) || volume < 0 || volume > 1) {
        throw new Error("[aquanova] playSound.volume must be finite and between 0 and 1");
    }
    return volume;
}

function sourceMatches(sources: string | string[], source: string): boolean {
    return typeof sources === "string" ? sources === source : sources.includes(source);
}
