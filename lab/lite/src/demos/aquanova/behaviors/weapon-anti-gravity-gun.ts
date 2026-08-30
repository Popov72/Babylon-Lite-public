import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, WeaponAntiGravityGunBehaviorConfig } from "./types.js";

const WEAPON_SLOT = 2;
const DEFAULT_MAX_MASS = 100;
const MAX_THROW_SPEED = 15;
const COLLISION_SLIDE_ITERATIONS = 3;

type CollisionHit = {
    readonly hasHit: boolean;
    readonly fraction: number;
    readonly hitNormal: { readonly x: number; readonly y: number; readonly z: number };
};
type Displacement = readonly [number, number, number];

type WeaponAntiGravityGunContext = Pick<AquanovaGameContext, "events" | "weaponInventory" | "weaponAntiGravityGun" | "playerMaxGrabDistance" | "dynamicMassOf">;

export function antiGravityCollisionMoveFraction(displacement: Displacement, hit: CollisionHit, skin = 0.01): number {
    if (!hit.hasHit) {
        return 1;
    }
    const approach = displacement[0] * hit.hitNormal.x + displacement[1] * hit.hitNormal.y + displacement[2] * hit.hitNormal.z;
    if (approach >= -1e-8) {
        return 1;
    }
    return Math.max(0, Math.min(1, hit.fraction - Math.max(0, skin) / -approach));
}

export function antiGravityCollisionSlideDisplacement(
    displacement: Displacement,
    cast: (offset: Displacement, displacement: Displacement) => CollisionHit,
    skin = 0.01
): { readonly movement: Displacement; readonly blocked: boolean } {
    const movement: [number, number, number] = [0, 0, 0];
    let remaining: [number, number, number] = [...displacement];
    let blocked = false;

    for (let iteration = 0; iteration < COLLISION_SLIDE_ITERATIONS; iteration++) {
        if (Math.hypot(...remaining) <= 1e-8) {
            break;
        }
        const hit = cast(movement, remaining);
        const fraction = antiGravityCollisionMoveFraction(remaining, hit, skin);
        movement[0] += remaining[0] * fraction;
        movement[1] += remaining[1] * fraction;
        movement[2] += remaining[2] * fraction;
        if (fraction >= 1) {
            break;
        }
        blocked = true;
        const normalLength = Math.hypot(hit.hitNormal.x, hit.hitNormal.y, hit.hitNormal.z);
        if (normalLength <= 1e-8) {
            break;
        }
        const nx = hit.hitNormal.x / normalLength;
        const ny = hit.hitNormal.y / normalLength;
        const nz = hit.hitNormal.z / normalLength;
        const residualScale = 1 - fraction;
        remaining = [remaining[0] * residualScale, remaining[1] * residualScale, remaining[2] * residualScale];
        const inward = remaining[0] * nx + remaining[1] * ny + remaining[2] * nz;
        if (inward >= -1e-8) {
            movement[0] += remaining[0];
            movement[1] += remaining[1];
            movement[2] += remaining[2];
            break;
        }
        remaining[0] -= nx * inward;
        remaining[1] -= ny * inward;
        remaining[2] -= nz * inward;
    }

    return { movement, blocked };
}

export class WeaponAntiGravityGunBehavior implements Behavior<"weaponAntiGravityGun"> {
    public readonly name = "weaponAntiGravityGun";
    public readonly retainOnEntityRetire = true;
    public readonly mesh: Mesh;
    public readonly config: WeaponAntiGravityGunBehaviorConfig;
    private readonly entityName: string;
    private readonly context: WeaponAntiGravityGunContext;
    private readonly disposers: Array<() => void> = [];
    private readonly maxGrabDistanceOverride: number | null;
    private readonly maxMass: number;
    private owned = false;
    private equipped = false;
    private awaitingGrab = false;
    private grabbed = false;

    public constructor(entityName: string, meshes: readonly Mesh[], config: WeaponAntiGravityGunBehaviorConfig, context: WeaponAntiGravityGunContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] weaponAntiGravityGun requires at least one mesh");
        }
        this.entityName = entityName;
        this.mesh = mesh;
        this.config = config;
        this.context = context;
        this.maxGrabDistanceOverride = config.maxGrabDistance === undefined ? null : positive(config.maxGrabDistance, "maxGrabDistance");
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
            this.context.events.on("weaponSecondaryPressed", () => this.drop()),
            this.context.events.on("weaponAimUpdated", ({ mesh, distance }) => this.updateAim(mesh, distance)),
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
            this.context.weaponAntiGravityGun.releaseGrab(MAX_THROW_SPEED);
            this.grabbed = false;
            return;
        }
        this.awaitingGrab = true;
    }

    private updateAim(mesh: Mesh | null, distance: number | null): void {
        if (!this.equipped || !this.awaitingGrab || this.grabbed) {
            return;
        }
        this.awaitingGrab = false;
        if (!mesh || distance === null || distance > (this.maxGrabDistanceOverride ?? this.context.playerMaxGrabDistance())) {
            return;
        }
        const mass = this.context.dynamicMassOf(mesh);
        if (mass === null || mass > this.maxMass) {
            return;
        }
        this.grabbed = this.context.weaponAntiGravityGun.grab(mesh);
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
        }
    }

    private drop(): void {
        if (this.grabbed) {
            this.context.weaponAntiGravityGun.releaseGrab(0);
        }
        this.awaitingGrab = false;
        this.grabbed = false;
    }
}

function positive(value: number, name: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`[aquanova] weaponAntiGravityGun.${name} must be a finite positive number, received ${String(value)}`);
    }
    return value;
}
