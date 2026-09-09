import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { ManagedSound } from "./sound-manager.js";
import type { Behavior, SoundBehaviorConfig, SoundPlayCueConfig, SoundStopCueConfig } from "./types.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import { eventSubscriptionMatches, validateEventSubscriptions } from "./event-subscription.js";
import { aquanovaSoundUrl, validateAquanovaSoundName } from "./sound-asset.js";

type SoundContext = Pick<AquanovaGameContext, "camera" | "events" | "sounds">;

interface SoundCueBase {
    readonly delay: number;
    readonly fade: number;
    sound: ManagedSound | null;
}

interface SoundPlayCue extends SoundCueBase {
    readonly config: SoundPlayCueConfig;
    readonly volume: number;
    readonly loop: boolean;
}

interface SoundStopCue extends SoundCueBase {
    readonly config: SoundStopCueConfig;
}

type SoundCue = SoundPlayCue | SoundStopCue;

interface DistanceGatedPlayback {
    readonly cue: SoundPlayCue;
    audible: boolean;
}

export class SoundBehavior implements Behavior<"sound"> {
    public readonly name = "sound";
    public readonly mesh: Mesh | null;
    public readonly config: SoundBehaviorConfig;
    private readonly context: SoundContext;
    private readonly cues: SoundCue[];
    private readonly radius: number;
    private readonly distanceGatedPlaybacks = new Map<ManagedSound, DistanceGatedPlayback>();
    private stopEntityEvent: (() => void) | null = null;
    private stopFrameStart: (() => void) | null = null;
    private readonly timers = new Set<ReturnType<typeof setTimeout>>();
    private disposed = false;

    public constructor(_entityName: string, meshes: readonly Mesh[], config: SoundBehaviorConfig, context: SoundContext) {
        assertBehaviorConfigKeys(config, "sound", ["cues", "radius"]);
        if (!Array.isArray(config.cues) || config.cues.length === 0) {
            throw new Error("[aquanova] sound.cues must contain at least one cue");
        }
        const radius = config.radius ?? 0;
        if (!Number.isFinite(radius) || radius < 0) {
            throw new Error("[aquanova] sound.radius must be a finite non-negative number");
        }
        this.mesh = meshes[0] ?? null;
        if (radius > 0 && !this.mesh) {
            throw new Error("[aquanova] sound.radius requires an owner mesh");
        }
        this.config = config;
        this.context = context;
        this.radius = radius;
        this.cues = config.cues.map((cue) => {
            if (!cue || typeof cue !== "object" || Array.isArray(cue)) {
                throw new Error("[aquanova] sound.cues[] must be an object");
            }
            if (cue.action !== "play" && cue.action !== "stop") {
                throw new Error('[aquanova] sound.cues[].action must be "play" or "stop"');
            }
            validateEventSubscriptions("sound.cues[]", cue.events);
            const base = {
                delay: nonNegativeSeconds("delay", cue.delay),
                fade: nonNegativeSeconds("fade", cue.fade),
                sound: null,
            };
            if (cue.action === "stop") {
                assertBehaviorConfigKeys(cue, "sound.cues[]", ["action", "delay", "events", "fade", "soundId"]);
                validatePlaybackId("sound.cues[].soundId", cue.soundId);
                return { ...base, config: cue };
            }
            assertBehaviorConfigKeys(cue, "sound.cues[]", ["action", "delay", "events", "fade", "id", "loop", "sound", "volume"]);
            validatePlaybackId("sound.cues[].id", cue.id);
            validateAquanovaSoundName("sound.cues[].sound", cue.sound);
            if (cue.loop !== undefined && typeof cue.loop !== "boolean") {
                throw new Error("[aquanova] sound.cues[].loop must be true or false");
            }
            return { ...base, config: cue, volume: soundVolume(cue.volume), loop: cue.loop ?? false };
        });
        for (const cue of this.cues) {
            if (!isPlayCue(cue)) {
                continue;
            }
            const url = aquanovaSoundUrl(cue.config.sound);
            context.sounds.registerPlayback(cue.config.id, url, { preloadCount: 1 });
        }
    }

    public async init(): Promise<void> {
        const sounds = new Map<string, ManagedSound>();
        await Promise.all(
            [...new Set(this.cues.map((cue) => playbackId(cue)))].map(async (id) => {
                try {
                    sounds.set(id, await this.context.sounds.resolvePlayback(id));
                } catch (error) {
                    throw new Error(`[aquanova] failed to initialize sound behavior playback ID "${id}"`, { cause: error });
                }
            })
        );
        for (const cue of this.cues) {
            cue.sound = sounds.get(playbackId(cue)) ?? null;
        }
        if (this.radius > 0) {
            for (const cue of this.cues) {
                if (isPlayCue(cue) && cue.sound) {
                    this.context.sounds.enableDistanceAttenuation(cue.sound, this.mesh!, this.context.camera, this.radius);
                }
            }
        }
    }

    public start(): void {
        if (this.cues.some((cue) => !cue.sound)) {
            throw new Error("[aquanova] sound behavior was not initialized");
        }
        if (this.radius > 0) {
            this.stopFrameStart = this.context.events.on("frameStart", () => this.updateDistanceGating());
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
        this.stopFrameStart?.();
        this.stopFrameStart = null;
        for (const timer of this.timers) {
            clearTimeout(timer);
        }
        this.timers.clear();
        const sounds = new Set(this.cues.flatMap((cue) => (isPlayCue(cue) && cue.sound ? [cue.sound] : [])));
        for (const sound of sounds) {
            this.context.sounds.stop(sound);
            if (this.radius > 0) {
                this.context.sounds.disableDistanceAttenuation(sound);
            }
        }
        this.distanceGatedPlaybacks.clear();
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
        if (isPlayCue(cue)) {
            if (this.radius === 0) {
                this.play(cue, sound);
                return;
            }
            const audible = this.isWithinHearingRadius();
            if (cue.loop) {
                this.distanceGatedPlaybacks.set(sound, { cue, audible });
            } else if (audible) {
                this.distanceGatedPlaybacks.set(sound, { cue, audible: true });
            }
            if (audible) {
                this.play(cue, sound);
            }
        } else {
            this.distanceGatedPlaybacks.delete(sound);
            this.context.sounds.stop(sound, cue.fade);
        }
    }

    private play(cue: SoundPlayCue, sound: ManagedSound): void {
        this.context.sounds.play(sound, { fade: cue.fade, loop: cue.loop, volume: cue.volume });
    }

    private updateDistanceGating(): void {
        const withinRadius = this.isWithinHearingRadius();
        for (const [sound, playback] of this.distanceGatedPlaybacks) {
            if (withinRadius === playback.audible) {
                continue;
            }
            if (!withinRadius) {
                this.context.sounds.stop(sound);
                playback.audible = false;
                if (!playback.cue.loop) {
                    this.distanceGatedPlaybacks.delete(sound);
                }
                continue;
            }
            if (playback.cue.loop) {
                this.play(playback.cue, sound);
                playback.audible = true;
            }
        }
    }

    private isWithinHearingRadius(): boolean {
        const world = this.mesh!.worldMatrix;
        const listener = this.context.camera.position;
        const dx = world[12]! - listener.x;
        const dy = world[13]! - listener.y;
        const dz = world[14]! - listener.z;
        return dx * dx + dy * dy + dz * dz < this.radius * this.radius;
    }
}

function validatePlaybackId(property: string, id: string): void {
    if (typeof id !== "string" || !id.trim()) {
        throw new Error(`[aquanova] ${property} must be a non-empty sound playback ID`);
    }
}

function nonNegativeSeconds(property: "delay" | "fade", value: number | undefined): number {
    const seconds = value ?? 0;
    if (!Number.isFinite(seconds) || seconds < 0) {
        throw new Error(`[aquanova] sound.cues[].${property} must be finite and non-negative`);
    }
    return seconds;
}

function soundVolume(value: number | undefined): number {
    const volume = value ?? 1;
    if (!Number.isFinite(volume) || volume < 0 || volume > 1) {
        throw new Error("[aquanova] sound.cues[].volume must be finite and between 0 and 1");
    }
    return volume;
}

function playbackId(cue: SoundCue): string {
    return cue.config.action === "play" ? cue.config.id : cue.config.soundId;
}

function isPlayCue(cue: SoundCue): cue is SoundPlayCue {
    return cue.config.action === "play";
}
