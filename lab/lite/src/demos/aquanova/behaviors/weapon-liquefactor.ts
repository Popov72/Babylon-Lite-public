import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { ManagedSound } from "./sound-manager.js";
import type { Behavior, WeaponLiquefactorBehaviorConfig } from "./types.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import { aquanovaSoundUrl, validateAquanovaSoundName } from "./sound-asset.js";

const DEFAULT_RANGE = 100;
const START_SHOT_SOUND = "liquefactorStartShot";
const LIQUEFY_SOUND = "liquefactorLiquefy";

type WeaponLiquefactorContext = Pick<
    AquanovaGameContext,
    | "events"
    | "sounds"
    | "nodeNameOf"
    | "isCollisionActive"
    | "weaponInventory"
    | "weaponLiquefactor"
    | "requestFusionResume"
    | "resolveFusionResume"
    | "resolveFusionTarget"
    | "fusionTargetLost"
    | "reverseFusion"
>;

const WEAPON_SLOT = 1;
interface PendingHit {
    readonly mesh: Mesh;
    readonly point: readonly [number, number, number] | null;
    readonly distance: number;
}

export class WeaponLiquefactorBehavior implements Behavior<"weaponLiquefactor"> {
    public readonly name = "weaponLiquefactor";
    public readonly retainOnEntityRetire = true;
    public readonly mesh: Mesh;
    public readonly config: WeaponLiquefactorBehaviorConfig;
    private readonly context: WeaponLiquefactorContext;
    private readonly entityName: string;
    private readonly disposers: Array<() => void> = [];
    private startShotSound: ManagedSound | null = null;
    private liquefySound: ManagedSound | null = null;
    private soundCategories: Map<string, readonly ManagedSound[]> | null = null;
    private pendingHit: PendingHit | null = null;
    private resumeToken: number | null = null;
    private triggerActive = false;
    private triggerHeld = false;
    private hitDelivered = false;
    private currentAimMesh: Mesh | null = null;
    private laserStarted = false;
    private beamDistance: number;
    private owned = false;
    private equipped = false;

    public constructor(entityName: string, meshes: readonly Mesh[], config: WeaponLiquefactorBehaviorConfig, context: WeaponLiquefactorContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] weaponLiquefactor requires at least one mesh");
        }
        assertBehaviorConfigKeys(config, "weaponLiquefactor", ["range", "sounds"]);
        this.entityName = entityName;
        this.mesh = mesh;
        this.config = config;
        this.context = context;
        this.beamDistance = config.range ?? DEFAULT_RANGE;
        if (!Number.isFinite(this.beamDistance) || this.beamDistance <= 0) {
            throw new Error(`[aquanova] weaponLiquefactor.range must be a finite positive number, received ${String(this.beamDistance)}`);
        }
    }

    public async init(): Promise<void> {
        const categories = this.config.sounds;
        const entries = Object.entries(categories ?? {});
        const names = new Set<string>([START_SHOT_SOUND, LIQUEFY_SOUND]);
        for (const [category, soundNames] of entries) {
            if (!category || !Array.isArray(soundNames) || soundNames.length === 0) {
                throw new Error(`[aquanova] weaponLiquefactor sound category "${category}" must contain at least one sound`);
            }
            for (const soundName of soundNames) {
                validateAquanovaSoundName("weaponLiquefactor sound", soundName);
                names.add(soundName);
            }
        }
        if (categories && !categories.quickSplash?.length) {
            throw new Error('[aquanova] weaponLiquefactor sounds must define the default "quickSplash" category');
        }

        const sounds = await Promise.all(
            [...names].map(async (soundName) => {
                const url = aquanovaSoundUrl(soundName);
                try {
                    return await this.context.sounds.load(`weaponLiquefactor:${soundName}`, url, {
                        preloadCount: 1,
                        ...(soundName === START_SHOT_SOUND || soundName === LIQUEFY_SOUND ? { maxInstances: 1 } : {}),
                    });
                } catch (error) {
                    throw new Error(`[aquanova] failed to preload Liquefactor sound "${soundName}" from "${url}"`, { cause: error });
                }
            })
        );
        const soundsByName = new Map([...names].map((soundName, index) => [soundName, sounds[index]!]));
        const soundCategories = new Map<string, readonly ManagedSound[]>();
        for (const [category, soundNames] of entries) {
            soundCategories.set(
                category,
                soundNames.map((soundName) => soundsByName.get(soundName)!)
            );
        }
        this.startShotSound = soundsByName.get(START_SHOT_SOUND)!;
        this.liquefySound = soundsByName.get(LIQUEFY_SOUND)!;
        this.soundCategories = soundCategories;
    }

    public start(): void {
        this.context.weaponLiquefactor.setEnabled(false, false);
        this.disposers.push(
            this.context.events.on("entityEvent", ({ name, event }) => {
                if (name === this.entityName && event === "enable") {
                    this.acquire();
                }
            }),
            this.context.events.on("weaponEquippedChanged", ({ slot }) => this.setEquipped(slot === WEAPON_SLOT)),
            this.context.events.on("weaponTriggerPressed", ({ held }) => this.pressTrigger(held)),
            this.context.events.on("weaponAimUpdated", (aim) => this.updateAim(aim.mesh, aim.point, aim.distance)),
            this.context.events.on("weaponTriggerReleased", () => this.releaseTrigger()),
            this.context.events.on("liquefactionStarted", ({ meshes }) => {
                if (this.equipped) {
                    this.playLiquefySound();
                }
                this.emitEntityEventForTargets(meshes, "startLiquefaction");
            }),
            this.context.events.on("liquefactionReversed", () => {
                if (this.equipped) {
                    this.stopLiquefySound();
                }
            }),
            this.context.events.on("liquefactionCancelled", ({ meshes }) => {
                this.emitEntityEventForTargets(meshes, "cancelLiquefaction");
            }),
            this.context.events.on("liquefactionCompleted", ({ meshes, sound }) => {
                this.emitEntityEventForTargets(meshes, "endLiquefaction");
                this.completeLiquefaction(sound);
            }),
            this.context.events.on("frameEnd", ({ deltaMs }) => {
                if (this.owned) {
                    this.update(deltaMs);
                }
            })
        );
    }

    public dispose(): void {
        for (const dispose of this.disposers.splice(0)) dispose();
        this.stopActionSounds();
        this.reset();
        this.owned = false;
        this.equipped = false;
        this.context.weaponLiquefactor.setEnabled(false, false);
    }

    private acquire(): void {
        if (this.owned) {
            return;
        }
        this.owned = true;
        this.context.weaponInventory.acquire(WEAPON_SLOT);
    }

    private setEquipped(equipped: boolean): void {
        if (this.equipped === equipped) {
            return;
        }
        if (!equipped) {
            this.stopActionSounds();
            if (this.triggerActive) {
                this.context.reverseFusion();
            }
            this.reset();
        }
        this.equipped = equipped;
        this.context.weaponLiquefactor.setEnabled(equipped, true);
    }

    private emitEntityEventForTargets(meshes: readonly Mesh[], event: string): void {
        const entityNames = new Set(meshes.map((mesh) => this.context.nodeNameOf(mesh)));
        for (const name of entityNames) {
            this.context.events.emit("entityEvent", { name, event });
        }
    }

    private pressTrigger(held: boolean): void {
        if (!this.equipped || !this.context.weaponLiquefactor.isReady()) {
            return;
        }
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
            this.reset();
            return;
        }
        this.playStartShotSound();
    }

    private updateAim(mesh: Mesh | null, point: readonly [number, number, number] | null, distance: number | null): void {
        if (!this.equipped || !this.triggerActive) {
            return;
        }
        const activeMesh = mesh && this.context.isCollisionActive(mesh) ? mesh : null;
        const targetMesh = this.context.resolveFusionTarget(activeMesh, activeMesh ? point : null);
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
        if (!this.equipped) {
            return;
        }
        this.stopActionSounds();
        this.context.reverseFusion();
        this.reset();
    }

    private completeLiquefaction(soundCategory: string): void {
        if (!this.equipped) {
            return;
        }
        this.stopActionSounds();
        this.playSplashSound(soundCategory);
        if (!this.triggerActive) return;
        if (this.triggerHeld) {
            this.playStartShotSound();
            return;
        }
        this.reset();
    }

    private playSplashSound(category: string): void {
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

    private playStartShotSound(): void {
        this.stopActionSounds();
        if (this.startShotSound) this.playSound(this.startShotSound, true);
    }

    private playLiquefySound(): void {
        this.stopActionSounds();
        if (this.liquefySound) this.playSound(this.liquefySound, true);
    }

    private stopLiquefySound(): void {
        if (this.liquefySound) this.context.sounds.stop(this.liquefySound);
    }

    private stopActionSounds(): void {
        if (this.startShotSound) this.context.sounds.stop(this.startShotSound);
        this.stopLiquefySound();
    }

    private playSound(sound: ManagedSound, loop = false): void {
        this.context.sounds.play(sound, loop ? { loop: true } : {});
    }

    private update(deltaMs: number): void {
        const reachedTarget = this.context.weaponLiquefactor.update(deltaMs);
        if (!this.equipped || !reachedTarget || !this.triggerActive || this.hitDelivered || !this.pendingHit) return;
        const hit = this.pendingHit;
        this.pendingHit = null;
        if (!this.context.isCollisionActive(hit.mesh)) {
            this.hitDelivered = false;
            this.currentAimMesh = null;
            if (!this.triggerHeld) this.releaseTrigger();
            return;
        }
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
