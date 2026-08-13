import { createAudioEngineAsync, createStreamingSoundAsync, disposeAudioEngine, playStreamingSound, preloadStreamingInstanceAsync, stopStreamingSound } from "babylon-lite";
import type { AudioEngine, Mesh, StreamingSound } from "babylon-lite";
import type { Behavior, BehaviorContext, WeaponLiquefactorBehaviorConfig } from "./types.js";

const DEFAULT_RANGE = 100;
const SOUND_ROOT = "/aquanova/sounds";
const SOUND_ASSET_VERSION = "20260813-1";
const START_SHOT_SOUND = "liquefactorStartShot";
const LIQUEFY_SOUND = "liquefactorLiquefy";

type WeaponLiquefactorContext = Pick<
    BehaviorContext,
    "events" | "weaponLiquefactor" | "requestFusionResume" | "resolveFusionResume" | "resolveFusionTarget" | "fusionTargetLost" | "reverseFusion"
>;

interface PendingHit {
    readonly mesh: Mesh;
    readonly point: readonly [number, number, number] | null;
    readonly distance: number;
}

export class WeaponLiquefactorBehavior implements Behavior<"weaponLiquefactor"> {
    private static initialization: Promise<void> | null = null;
    private static audioEngine: AudioEngine | null = null;
    private static startShotSound: StreamingSound | null = null;
    private static liquefySound: StreamingSound | null = null;
    private static soundCategories: Map<string, readonly StreamingSound[]> | null = null;
    public readonly name = "weaponLiquefactor";
    public readonly mesh: Mesh;
    public readonly config: WeaponLiquefactorBehaviorConfig;
    private readonly context: WeaponLiquefactorContext;
    private readonly disposers: Array<() => void> = [];
    private pendingHit: PendingHit | null = null;
    private resumeToken: number | null = null;
    private triggerActive = false;
    private triggerHeld = false;
    private hitDelivered = false;
    private currentAimMesh: Mesh | null = null;
    private laserStarted = false;
    private beamDistance: number;

    public constructor(mesh: Mesh, config: WeaponLiquefactorBehaviorConfig, context: WeaponLiquefactorContext) {
        this.mesh = mesh;
        this.config = config;
        this.context = context;
        this.beamDistance = config.range ?? DEFAULT_RANGE;
        if (!Number.isFinite(this.beamDistance) || this.beamDistance <= 0) {
            throw new Error(`[aquanova] weaponLiquefactor.range must be a finite positive number, received ${String(this.beamDistance)}`);
        }
    }

    public static init(config: WeaponLiquefactorBehaviorConfig): Promise<void> {
        if (!this.initialization) {
            this.initialization = this.initialize(config).catch((error: unknown) => {
                this.initialization = null;
                throw error;
            });
        }
        return this.initialization;
    }

    public static dispose(): void {
        if (this.audioEngine) disposeAudioEngine(this.audioEngine);
        this.audioEngine = null;
        this.startShotSound = null;
        this.liquefySound = null;
        this.soundCategories = null;
        this.initialization = null;
    }

    private static async initialize(config: WeaponLiquefactorBehaviorConfig): Promise<void> {
        const categories = config.sounds;
        const entries = Object.entries(categories ?? {});
        const names = new Set<string>([START_SHOT_SOUND, LIQUEFY_SOUND]);
        for (const [category, soundNames] of entries) {
            if (!category || !Array.isArray(soundNames) || soundNames.length === 0) {
                throw new Error(`[aquanova] weaponLiquefactor sound category "${category}" must contain at least one sound`);
            }
            for (const soundName of soundNames) {
                if (!soundName || soundName.endsWith(".mp3") || soundName.includes("/") || soundName.includes("\\")) {
                    throw new Error(`[aquanova] weaponLiquefactor sound "${String(soundName)}" must be an MP3 file name without its extension`);
                }
                names.add(soundName);
            }
        }
        if (categories && !categories.quickSplash?.length) {
            throw new Error('[aquanova] weaponLiquefactor sounds must define the default "quickSplash" category');
        }

        const engine = await createAudioEngineAsync();
        try {
            const soundsByName = new Map<string, StreamingSound>();
            for (const soundName of names) {
                const url = `${SOUND_ROOT}/${encodeURIComponent(soundName)}.mp3?v=${SOUND_ASSET_VERSION}`;
                try {
                    const sound = await createStreamingSoundAsync(engine, url, {
                        preloadCount: 1,
                        ...(soundName === START_SHOT_SOUND || soundName === LIQUEFY_SOUND ? { maxInstances: 1 } : {}),
                    });
                    soundsByName.set(soundName, sound);
                } catch (error) {
                    throw new Error(`[aquanova] failed to preload Liquefactor sound "${soundName}" from "${url}"`, { cause: error });
                }
            }
            const soundCategories = new Map<string, readonly StreamingSound[]>();
            for (const [category, soundNames] of entries) {
                soundCategories.set(
                    category,
                    soundNames.map((soundName) => soundsByName.get(soundName)!)
                );
            }
            this.startShotSound = soundsByName.get(START_SHOT_SOUND)!;
            this.liquefySound = soundsByName.get(LIQUEFY_SOUND)!;
            this.soundCategories = soundCategories;
            this.audioEngine = engine;
        } catch (error) {
            disposeAudioEngine(engine);
            throw error;
        }
    }

    public start(): void {
        this.disposers.push(
            this.context.events.on("weaponTriggerPressed", ({ held }) => this.pressTrigger(held)),
            this.context.events.on("weaponAimUpdated", (aim) => this.updateAim(aim.mesh, aim.point, aim.distance)),
            this.context.events.on("weaponTriggerReleased", () => this.releaseTrigger()),
            this.context.events.on("liquefactionStarted", () => WeaponLiquefactorBehavior.playLiquefySound()),
            this.context.events.on("liquefactionReversed", () => WeaponLiquefactorBehavior.stopLiquefySound()),
            this.context.events.on("liquefactionCompleted", ({ sound }) => this.completeLiquefaction(sound)),
            this.context.events.on("frameEnd", ({ deltaMs }) => this.update(deltaMs))
        );
    }

    public dispose(): void {
        for (const dispose of this.disposers.splice(0)) dispose();
        WeaponLiquefactorBehavior.stopActionSounds();
        this.reset();
    }

    private pressTrigger(held: boolean): void {
        this.triggerActive = true;
        this.triggerHeld = held;
        this.hitDelivered = false;
        this.currentAimMesh = null;
        this.laserStarted = false;
        this.beamDistance = this.config.range ?? DEFAULT_RANGE;
        this.pendingHit = null;
        this.context.weaponLiquefactor.stop();
        this.resumeToken = this.context.requestFusionResume();
        if (this.resumeToken === 0) {
            this.triggerActive = false;
            this.resumeToken = null;
            return;
        }
        WeaponLiquefactorBehavior.playStartShotSound();
    }

    private updateAim(mesh: Mesh | null, point: readonly [number, number, number] | null, distance: number | null): void {
        if (!this.triggerActive) return;
        const targetMesh = this.context.resolveFusionTarget(mesh, point);
        if (this.context.fusionTargetLost(targetMesh)) {
            this.context.reverseFusion();
            this.resumeToken = this.context.requestFusionResume();
        }
        if (targetMesh !== this.currentAimMesh) {
            this.currentAimMesh = targetMesh;
            this.hitDelivered = false;
        }
        if (!this.hitDelivered) this.pendingHit = targetMesh && distance !== null ? { mesh: targetMesh, point, distance } : null;
        if (distance !== null) this.beamDistance = distance;
        this.context.weaponLiquefactor.setTargetDistance(this.beamDistance, !this.laserStarted);
        this.laserStarted = true;
        if (!this.triggerHeld && this.pendingHit === null) this.releaseTrigger();
    }

    private releaseTrigger(): void {
        WeaponLiquefactorBehavior.stopActionSounds();
        this.context.reverseFusion();
        this.reset();
    }

    private completeLiquefaction(soundCategory: string): void {
        WeaponLiquefactorBehavior.stopActionSounds();
        WeaponLiquefactorBehavior.playSplashSound(soundCategory);
        if (!this.triggerActive || this.triggerHeld) return;
        this.reset();
    }

    private static playSplashSound(category: string): void {
        const categories = this.soundCategories;
        if (!categories?.size) return;
        const sounds = categories.get(category);
        if (!sounds?.length) {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] Liquefactor sound category "${category}" was requested but not preloaded`);
            return;
        }
        const sound = sounds[Math.floor(Math.random() * sounds.length)]!;
        this.playSound(sound);
    }

    private static playStartShotSound(): void {
        this.stopActionSounds();
        if (this.startShotSound) this.playSound(this.startShotSound);
    }

    private static playLiquefySound(): void {
        this.stopActionSounds();
        if (this.liquefySound) this.playSound(this.liquefySound, true);
    }

    private static stopLiquefySound(): void {
        if (this.liquefySound) stopStreamingSound(this.liquefySound);
    }

    private static stopActionSounds(): void {
        if (this.startShotSound) stopStreamingSound(this.startShotSound);
        this.stopLiquefySound();
    }

    private static playSound(sound: StreamingSound, loop = false): void {
        if (loop) playStreamingSound(sound, { loop: true });
        else playStreamingSound(sound);
        void preloadStreamingInstanceAsync(sound).catch((error: unknown) => {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] failed to replenish a preloaded Liquefactor sound instance`, error);
        });
    }

    private update(deltaMs: number): void {
        const reachedTarget = this.context.weaponLiquefactor.update(deltaMs);
        if (!reachedTarget || !this.triggerActive || this.hitDelivered || !this.pendingHit) return;
        const hit = this.pendingHit;
        this.pendingHit = null;
        this.hitDelivered = true;
        let shouldEmitHit = true;
        if (this.resumeToken !== null) {
            const result = this.context.resolveFusionResume(this.resumeToken, hit.mesh);
            if (result !== "await-target") {
                shouldEmitHit = result !== "resumed";
                this.resumeToken = null;
            }
        }
        if (shouldEmitHit) {
            this.context.events.emit("hitWithWeapon", {
                mesh: hit.mesh,
                point: hit.point,
                distance: hit.distance,
            });
        }
        if (!this.triggerHeld) this.reset();
    }

    private reset(): void {
        this.triggerActive = false;
        this.triggerHeld = false;
        this.hitDelivered = false;
        this.currentAimMesh = null;
        this.laserStarted = false;
        this.beamDistance = this.config.range ?? DEFAULT_RANGE;
        this.pendingHit = null;
        this.resumeToken = null;
        this.context.weaponLiquefactor.stop();
    }
}
