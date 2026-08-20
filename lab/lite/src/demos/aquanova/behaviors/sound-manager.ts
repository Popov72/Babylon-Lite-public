import {
    createAudioEngineAsync,
    createStreamingSoundAsync,
    disposeAudioEngine,
    playStreamingSound,
    preloadStreamingInstanceAsync,
    setMasterVolume,
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

export class SoundManager {
    private engineInitialization: Promise<AudioEngine> | null = null;
    private engine: AudioEngine | null = null;
    private readonly loads = new Map<string, Promise<ManagedSound>>();
    private readonly activeLoops = new Set<ManagedSound>();
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

    public play(sound: ManagedSound, options: StreamingSoundPlayOptions = {}): void {
        if (!this.enabled) {
            return;
        }
        if (Object.keys(options).length > 0) {
            playStreamingSound(sound.sound, options);
        } else {
            playStreamingSound(sound.sound);
        }
        if (options.loop) {
            this.activeLoops.add(sound);
        }
        void preloadStreamingInstanceAsync(sound.sound).catch((error: unknown) => {
            console.warn(`[aquanova] failed to replenish preloaded sound "${sound.label}"`, error);
        });
    }

    public stop(sound: ManagedSound): void {
        this.activeLoops.delete(sound);
        stopStreamingSound(sound.sound);
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
        if (this.engine) {
            disposeAudioEngine(this.engine);
        }
        this.engine = null;
        this.engineInitialization = null;
        this.loads.clear();
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
            stopStreamingSound(sound.sound);
        }
        this.activeLoops.clear();
    }
}
