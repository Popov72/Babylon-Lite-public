import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, WeaponAntiGravityGunBehaviorConfig } from "./types.js";

const WEAPON_SLOT = 2;
const DEFAULT_MAX_GRAB_DISTANCE = 6;
const DEFAULT_MAX_MASS = 100;
const QUICK_DROP_MS = 150;
const MAX_CHARGE_MS = 2000;
const MAX_THROW_SPEED = 15;

type WeaponAntiGravityGunContext = Pick<AquanovaGameContext, "events" | "weaponInventory" | "weaponAntiGravityGun" | "dynamicMassOf">;

export function antiGravityThrowSpeed(chargeMs: number): number {
    const normalized = Math.max(0, Math.min(MAX_CHARGE_MS, chargeMs) - QUICK_DROP_MS) / (MAX_CHARGE_MS - QUICK_DROP_MS);
    return normalized * MAX_THROW_SPEED;
}

export class WeaponAntiGravityGunBehavior implements Behavior<"weaponAntiGravityGun"> {
    public readonly name = "weaponAntiGravityGun";
    public readonly mesh: Mesh;
    public readonly config: WeaponAntiGravityGunBehaviorConfig;
    private readonly entityName: string;
    private readonly context: WeaponAntiGravityGunContext;
    private readonly disposers: Array<() => void> = [];
    private readonly maxGrabDistance: number;
    private readonly maxMass: number;
    private owned = false;
    private equipped = false;
    private awaitingGrab = false;
    private grabbed = false;
    private charging = false;
    private chargeMs = 0;

    public constructor(entityName: string, meshes: readonly Mesh[], config: WeaponAntiGravityGunBehaviorConfig, context: WeaponAntiGravityGunContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] weaponAntiGravityGun requires at least one mesh");
        }
        this.entityName = entityName;
        this.mesh = mesh;
        this.config = config;
        this.context = context;
        this.maxGrabDistance = positive(config.maxGrabDistance ?? DEFAULT_MAX_GRAB_DISTANCE, "maxGrabDistance");
        this.maxMass = positive(config.maxMass ?? DEFAULT_MAX_MASS, "maxMass");
    }

    public init(): void {}

    public start(): void {
        this.context.weaponAntiGravityGun.setEnabled(false, false);
        this.disposers.push(
            this.context.events.on("entityEvent", ({ name, event }) => {
                if (name === this.entityName && event === "enable") {
                    this.acquire();
                }
            }),
            this.context.events.on("weaponEquippedChanged", ({ slot }) => this.setEquipped(slot === WEAPON_SLOT)),
            this.context.events.on("weaponTriggerPressed", () => this.pressTrigger()),
            this.context.events.on("weaponAimUpdated", ({ mesh, distance }) => this.updateAim(mesh, distance)),
            this.context.events.on("weaponTriggerReleased", () => this.releaseTrigger()),
            this.context.events.on("frameEnd", ({ deltaMs }) => this.update(deltaMs))
        );
    }

    public dispose(): void {
        for (const dispose of this.disposers.splice(0)) {
            dispose();
        }
        this.drop();
        this.owned = false;
        this.equipped = false;
        this.context.weaponAntiGravityGun.setEnabled(false, false);
    }

    private acquire(): void {
        if (this.owned) {
            return;
        }
        this.owned = true;
        this.context.weaponInventory.acquire(WEAPON_SLOT);
    }

    private setEquipped(equipped: boolean): void {
        if (!this.owned || this.equipped === equipped) {
            return;
        }
        if (!equipped) {
            this.drop();
        }
        this.equipped = equipped;
        this.context.weaponAntiGravityGun.setEnabled(equipped, true);
    }

    private pressTrigger(): void {
        if (!this.equipped || !this.context.weaponAntiGravityGun.isReady()) {
            return;
        }
        if (this.grabbed) {
            this.charging = true;
            this.chargeMs = 0;
            return;
        }
        this.awaitingGrab = true;
    }

    private updateAim(mesh: Mesh | null, distance: number | null): void {
        if (!this.equipped || !this.awaitingGrab || this.grabbed) {
            return;
        }
        this.awaitingGrab = false;
        if (!mesh || distance === null || distance > this.maxGrabDistance) {
            return;
        }
        const mass = this.context.dynamicMassOf(mesh);
        if (mass === null || mass > this.maxMass) {
            return;
        }
        this.grabbed = this.context.weaponAntiGravityGun.grab(mesh);
    }

    private releaseTrigger(): void {
        if (!this.equipped) {
            this.awaitingGrab = false;
            return;
        }
        if (!this.grabbed || !this.charging) {
            return;
        }
        const throwSpeed = antiGravityThrowSpeed(this.chargeMs);
        this.context.weaponAntiGravityGun.releaseGrab(throwSpeed);
        this.grabbed = false;
        this.charging = false;
        this.chargeMs = 0;
    }

    private update(deltaMs: number): void {
        if (!this.owned) {
            return;
        }
        this.context.weaponAntiGravityGun.update(deltaMs);
        if (!this.grabbed) {
            return;
        }
        if (!this.context.weaponAntiGravityGun.updateGrab(deltaMs)) {
            this.grabbed = false;
            this.charging = false;
            this.chargeMs = 0;
            return;
        }
        if (this.charging) {
            this.chargeMs = Math.min(MAX_CHARGE_MS, this.chargeMs + Math.max(0, deltaMs));
        }
    }

    private drop(): void {
        if (this.grabbed) {
            this.context.weaponAntiGravityGun.releaseGrab(0);
        }
        this.awaitingGrab = false;
        this.grabbed = false;
        this.charging = false;
        this.chargeMs = 0;
    }
}

function positive(value: number, name: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`[aquanova] weaponAntiGravityGun.${name} must be a finite positive number, received ${String(value)}`);
    }
    return value;
}
