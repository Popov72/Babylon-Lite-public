import {
    createAudioEngineAsync,
    createStreamingSoundAsync,
    disposeAudioEngine,
    playStreamingSound,
    preloadStreamingInstanceAsync,
    setMasterVolume,
    setStreamingSoundVolume,
    stopStreamingSound,
} from "babylon-lite";
import type { AudioEngine, StreamingSound, StreamingSoundOptions, StreamingSoundPlayOptions } from "babylon-lite";
import { normalizeSoundVolume } from "./sound-volume.js";

export interface ManagedSound {
    readonly label: string;
    readonly source: string;
    /** @internal */
    readonly sound: StreamingSound;
}

export interface ManagedSoundPlayOptions extends StreamingSoundPlayOptions {
    /** Fade-in duration in seconds. Defaults to 0. */
    fade?: number;
}

interface RegisteredPlayback {
    readonly source: string;
    readonly options: StreamingSoundOptions;
    load: Promise<ManagedSound> | null;
}

export class SoundManager {
    private engineInitialization: Promise<AudioEngine> | null = null;
    private engine: AudioEngine | null = null;
    private readonly loads = new Map<string, Promise<ManagedSound>>();
    private readonly registeredPlaybacks = new Map<string, RegisteredPlayback>();
    private readonly activeLoops = new Set<ManagedSound>();
    private readonly pendingStops = new Map<ManagedSound, ReturnType<typeof setTimeout>>();
    private enabled = true;
    private volume = 1;

    public async load(label: string, source: string, options: StreamingSoundOptions = {}): Promise<ManagedSound> {
        let load = this.loads.get(source);
        if (!load) {
            load = this.createSound(label, source, options).catch((error: unknown) => {
                this.loads.delete(source);
                throw error;
            });
            this.loads.set(source, load);
        }
        return load;
    }

    public registerPlayback(id: string, source: string, options: StreamingSoundOptions = {}): void {
        if (typeof id !== "string" || !id.trim()) {
            throw new Error("[aquanova] sound playback ID must be non-empty");
        }
        if (this.registeredPlaybacks.has(id)) {
            throw new Error(`[aquanova] sound playback ID "${id}" is defined more than once`);
        }
        this.registeredPlaybacks.set(id, { source, options: { ...options }, load: null });
    }

    public resolvePlayback(id: string): Promise<ManagedSound> {
        const playback = this.registeredPlaybacks.get(id);
        if (!playback) {
            throw new Error(`[aquanova] sound playback ID "${id}" is not defined by a sound play cue`);
        }
        playback.load ??= this.createSound(`sound:${id}`, playback.source, playback.options).catch((error: unknown) => {
            playback.load = null;
            throw error;
        });
        return playback.load;
    }

    public play(sound: ManagedSound, options: ManagedSoundPlayOptions = {}): void {
        if (!this.enabled) {
            return;
        }
        const fade = validateFade(options.fade);
        const interruptedFade = this.cancelPendingStop(sound);
        const playOptions: StreamingSoundPlayOptions = {
            ...(options.loop !== undefined ? { loop: options.loop } : {}),
            ...(options.startOffset !== undefined ? { startOffset: options.startOffset } : {}),
            ...(options.volume !== undefined ? { volume: options.volume } : {}),
        };
        if (fade > 0) {
            setStreamingSoundVolume(sound.sound, 0, { shape: "none" });
        } else if (interruptedFade) {
            setStreamingSoundVolume(sound.sound, 1, { shape: "none" });
        }
        if (Object.keys(playOptions).length > 0) {
            playStreamingSound(sound.sound, playOptions);
        } else {
            playStreamingSound(sound.sound);
        }
        if (fade > 0) {
            setStreamingSoundVolume(sound.sound, 1, { duration: fade, shape: "linear" });
        }
        if (options.loop) {
            this.activeLoops.add(sound);
        }
        void preloadStreamingInstanceAsync(sound.sound).catch((error: unknown) => {
            console.warn(`[aquanova] failed to replenish preloaded sound "${sound.label}"`, error);
        });
    }

    public stop(sound: ManagedSound, fade = 0): void {
        const duration = validateFade(fade);
        this.cancelPendingStop(sound);
        if (duration === 0) {
            this.stopImmediately(sound);
            return;
        }
        setStreamingSoundVolume(sound.sound, 0, { duration, shape: "linear" });
        const timer = setTimeout(() => {
            this.pendingStops.delete(sound);
            this.stopImmediately(sound);
        }, duration * 1000);
        this.pendingStops.set(sound, timer);
    }

    public setEnabled(enabled: boolean): void {
        this.enabled = enabled;
        if (!enabled) {
            this.stopActiveLoops();
        }
    }

    public setVolume(volume: number): void {
        this.volume = normalizeSoundVolume(volume);
        if (this.engine) {
            setMasterVolume(this.engine, this.volume);
        }
    }

    public dispose(): void {
        this.stopActiveLoops();
        for (const timer of this.pendingStops.values()) {
            clearTimeout(timer);
        }
        this.pendingStops.clear();
        if (this.engine) {
            disposeAudioEngine(this.engine);
        }
        this.engine = null;
        this.engineInitialization = null;
        this.loads.clear();
        this.registeredPlaybacks.clear();
    }

    private async createSound(label: string, source: string, options: StreamingSoundOptions): Promise<ManagedSound> {
        const engine = await this.initializeEngine();
        return {
            label,
            source,
            sound: await createStreamingSoundAsync(engine, source, options),
        };
    }

    private initializeEngine(): Promise<AudioEngine> {
        if (!this.engineInitialization) {
            this.engineInitialization = createAudioEngineAsync()
                .then((engine) => {
                    this.engine = engine;
                    setMasterVolume(engine, this.volume);
                    return engine;
                })
                .catch((error: unknown) => {
                    this.engineInitialization = null;
                    throw error;
                });
        }
        return this.engineInitialization;
    }

    private stopActiveLoops(): void {
        for (const sound of this.activeLoops) {
            this.cancelPendingStop(sound);
            this.stopImmediately(sound);
        }
        this.activeLoops.clear();
    }

    private cancelPendingStop(sound: ManagedSound): boolean {
        const timer = this.pendingStops.get(sound);
        if (timer === undefined) {
            return false;
        }
        clearTimeout(timer);
        this.pendingStops.delete(sound);
        return true;
    }

    private stopImmediately(sound: ManagedSound): void {
        this.activeLoops.delete(sound);
        stopStreamingSound(sound.sound);
        setStreamingSoundVolume(sound.sound, 1, { shape: "none" });
    }
}

function validateFade(fade: number | undefined): number {
    const duration = fade ?? 0;
    if (!Number.isFinite(duration) || duration < 0) {
        throw new Error("[aquanova] sound fade must be finite and non-negative");
    }
    return duration;
}
