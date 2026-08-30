import type { Mesh } from "babylon-lite";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import type { AquanovaGameContext } from "./game-context.js";
import { aquanovaSoundUrl, validateAquanovaSoundName } from "./sound-asset.js";
import type { ManagedSound } from "./sound-manager.js";
import type { Behavior, WeaponPistolBehaviorConfig } from "./types.js";

const WEAPON_SLOT = 3;
const DEFAULT_RANGE = 100;
const DEFAULT_BULLET_SPEED = 80;
const DEFAULT_IMPACT_IMPULSE = 10;
const DEFAULT_BULLET_HOLE_SIZE = 1;
const DEFAULT_SOUND = "pistolShot";

type WeaponPistolContext = Pick<AquanovaGameContext, "events" | "sounds" | "weaponInventory" | "weaponPistol">;

export class WeaponPistolBehavior implements Behavior<"weaponPistol"> {
    public readonly name = "weaponPistol";
    public readonly retainOnEntityRetire = true;
    public readonly mesh: Mesh | null;
    public readonly config: WeaponPistolBehaviorConfig;
    private readonly context: WeaponPistolContext;
    private readonly entityName: string;
    private readonly disposers: Array<() => void> = [];
    private readonly range: number;
    private readonly bulletSpeed: number;
    private readonly impactImpulse: number;
    private readonly bulletHoleSize: number;
    private shotSound: ManagedSound | null = null;
    private owned = false;
    private equipped = false;
    private awaitingAim = false;

    public constructor(entityName: string, meshes: readonly Mesh[], config: WeaponPistolBehaviorConfig, context: WeaponPistolContext) {
        assertBehaviorConfigKeys(config, "weaponPistol", ["range", "bulletSpeed", "impactImpulse", "bulletHoleSize", "sound"]);
        validateAquanovaSoundName("weaponPistol sound", config.sound ?? DEFAULT_SOUND);
        this.entityName = entityName;
        this.mesh = meshes[0] ?? null;
        this.config = config;
        this.context = context;
        this.range = positive(config.range ?? DEFAULT_RANGE, "range");
        this.bulletSpeed = positive(config.bulletSpeed ?? DEFAULT_BULLET_SPEED, "bulletSpeed");
        this.impactImpulse = nonNegative(config.impactImpulse ?? DEFAULT_IMPACT_IMPULSE, "impactImpulse");
        this.bulletHoleSize = positive(config.bulletHoleSize ?? DEFAULT_BULLET_HOLE_SIZE, "bulletHoleSize");
    }

    public async init(): Promise<void> {
        const soundName = this.config.sound ?? DEFAULT_SOUND;
        const url = aquanovaSoundUrl(soundName);
        try {
            this.shotSound = await this.context.sounds.load(`weaponPistol:${soundName}`, url, { preloadCount: 1 });
        } catch (error) {
            throw new Error(`[aquanova] failed to preload weaponPistol sound "${soundName}" from "${url}"`, { cause: error });
        }
    }

    public start(): void {
        this.context.weaponPistol.setEnabled(false, false);
        this.disposers.push(
            this.context.events.on("entityEvent", ({ name, event }) => {
                if (name === this.entityName && event === "enable") {
                    this.acquire();
                }
            }),
            this.context.events.on("weaponEquippedChanged", ({ slot }) => this.setEquipped(slot === WEAPON_SLOT)),
            this.context.events.on("weaponTriggerPressed", () => this.pressTrigger()),
            this.context.events.on("weaponAimUpdated", ({ mesh, point, distance }) => this.updateAim(mesh, point, distance)),
            this.context.events.on("frameEnd", ({ deltaMs }) => this.update(deltaMs))
        );
    }

    public dispose(): void {
        for (const dispose of this.disposers.splice(0)) {
            dispose();
        }
        this.awaitingAim = false;
        this.owned = false;
        this.equipped = false;
        this.context.weaponPistol.clear();
        this.context.weaponPistol.setEnabled(false, false);
    }

    private acquire(): void {
        if (this.owned) return;
        this.owned = true;
        this.context.weaponInventory.acquire(WEAPON_SLOT);
    }

    private setEquipped(equipped: boolean): void {
        if (!this.owned || this.equipped === equipped) return;
        this.equipped = equipped;
        if (!equipped) {
            this.awaitingAim = false;
        }
        this.context.weaponPistol.setEnabled(equipped, true);
    }

    private pressTrigger(): void {
        if (!this.equipped || !this.context.weaponPistol.isReady()) return;
        this.awaitingAim = true;
    }

    private updateAim(mesh: Mesh | null, point: readonly [number, number, number] | null, distance: number | null): void {
        if (!this.equipped || !this.awaitingAim) return;
        this.awaitingAim = false;
        if (!this.shotSound) {
            throw new Error("[aquanova] weaponPistol sound was not initialized");
        }
        this.context.weaponPistol.fire(mesh, point, distance, this.range, this.bulletSpeed);
        this.context.sounds.play(this.shotSound);
    }

    private update(deltaMs: number): void {
        if (!this.owned) return;
        for (const impact of this.context.weaponPistol.update(deltaMs)) {
            this.context.events.emit("hitWithPistol", { ...impact, impulse: this.impactImpulse, bulletHoleSize: this.bulletHoleSize });
        }
    }
}

function positive(value: number, name: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`[aquanova] weaponPistol.${name} must be a finite positive number, received ${String(value)}`);
    }
    return value;
}

function nonNegative(value: number, name: string): number {
    if (!Number.isFinite(value) || value < 0) {
        throw new Error(`[aquanova] weaponPistol.${name} must be a finite non-negative number, received ${String(value)}`);
    }
    return value;
}
