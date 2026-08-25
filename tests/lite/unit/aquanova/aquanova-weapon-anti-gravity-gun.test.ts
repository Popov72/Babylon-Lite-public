import { describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../../packages/babylon-lite/src";
import { createSceneNode } from "../../../../packages/babylon-lite/src/scene/scene-node";
import { AquanovaEventManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import type { AquanovaGameContext, WeaponAntiGravityGunRuntime } from "../../../../lab/lite/src/demos/aquanova/behaviors/game-context";
import { WeaponAntiGravityGunBehavior, antiGravityCollisionMoveFraction } from "../../../../lab/lite/src/demos/aquanova/behaviors/weapon-anti-gravity-gun";
import { WeaponInventory } from "../../../../lab/lite/src/demos/aquanova/behaviors/weapon-inventory";

type WeaponContext = Pick<AquanovaGameContext, "events" | "weaponInventory" | "weaponAntiGravityGun" | "playerMaxGrabDistance" | "dynamicMassOf">;

function mesh(name: string): Mesh {
    return createSceneNode(name) as Mesh;
}

function createHarness(options: { maxGrabDistance?: number; legacyWeaponMaxGrabDistance?: number; maxMass?: number; mass?: number | null; enabled?: boolean } = {}) {
    const events = new AquanovaEventManager();
    const weaponInventory = new WeaponInventory();
    weaponInventory.start(events);
    let ready = true;
    let grabActive = false;
    const runtime: WeaponAntiGravityGunRuntime = {
        setEnabled: vi.fn(),
        isReady: vi.fn(() => ready),
        update: vi.fn(),
        grab: vi.fn(() => {
            grabActive = true;
            return true;
        }),
        updateGrab: vi.fn(() => grabActive),
        releaseGrab: vi.fn(() => {
            grabActive = false;
        }),
    };
    const context: WeaponContext = {
        events,
        weaponInventory,
        weaponAntiGravityGun: runtime,
        playerMaxGrabDistance: () => options.maxGrabDistance ?? 8,
        dynamicMassOf: vi.fn(() => (options.mass === undefined ? 10 : options.mass)),
    };
    const behavior = new WeaponAntiGravityGunBehavior(
        "itemAntiGravityGun",
        [mesh("weapon")],
        { maxGrabDistance: options.legacyWeaponMaxGrabDistance, maxMass: options.maxMass },
        context
    );
    behavior.start();
    if (options.enabled !== false) {
        events.emit("entityEvent", { name: "itemAntiGravityGun", event: "enable" });
    }
    return {
        behavior,
        events,
        runtime,
        setReady(value: boolean): void {
            ready = value;
        },
    };
}

describe("Aquanova anti-gravity gun behavior", () => {
    it("starts hidden, acquires slot 2, and ignores the trigger until presented", () => {
        const harness = createHarness({ enabled: false });

        expect(harness.runtime.setEnabled).toHaveBeenCalledWith(false, false);
        harness.events.emit("weaponTriggerPressed", { held: true });
        expect(harness.runtime.grab).not.toHaveBeenCalled();

        harness.events.emit("entityEvent", { name: "itemAntiGravityGun", event: "enable" });
        expect(harness.runtime.setEnabled).toHaveBeenLastCalledWith(true, true);
        harness.setReady(false);
        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: mesh("crate"), point: null, distance: 2 });
        expect(harness.runtime.grab).not.toHaveBeenCalled();
    });

    it("grabs only dynamic targets within the configured distance and mass limits", () => {
        const target = mesh("crate");
        const harness = createHarness({ maxGrabDistance: 4, maxMass: 20, mass: 10 });

        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 4 });
        expect(harness.runtime.grab).toHaveBeenCalledWith(target);

        const distant = createHarness({ maxGrabDistance: 4, mass: 10 });
        distant.events.emit("weaponTriggerPressed", { held: true });
        distant.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 4.01 });
        expect(distant.runtime.grab).not.toHaveBeenCalled();

        const legacyDistance = createHarness({ maxGrabDistance: 8, legacyWeaponMaxGrabDistance: 4, mass: 10 });
        legacyDistance.events.emit("weaponTriggerPressed", { held: true });
        legacyDistance.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 4.01 });
        expect(legacyDistance.runtime.grab).not.toHaveBeenCalled();

        const heavy = createHarness({ maxMass: 20, mass: 20.01 });
        heavy.events.emit("weaponTriggerPressed", { held: true });
        heavy.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 2 });
        expect(heavy.runtime.grab).not.toHaveBeenCalled();

        const staticTarget = createHarness({ mass: null });
        staticTarget.events.emit("weaponTriggerPressed", { held: true });
        staticTarget.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 2 });
        expect(staticTarget.runtime.grab).not.toHaveBeenCalled();
    });

    it("keeps the first click grabbed, then throws immediately on the next left press", () => {
        const target = mesh("crate");
        const harness = createHarness();

        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponTriggerReleased", {});
        harness.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 2 });
        expect(harness.runtime.releaseGrab).not.toHaveBeenCalled();

        harness.events.emit("weaponTriggerPressed", { held: true });
        expect(harness.runtime.releaseGrab).toHaveBeenLastCalledWith(15);
        harness.events.emit("frameEnd", { deltaMs: 2000 });
        harness.events.emit("weaponTriggerReleased", {});
        expect(harness.runtime.releaseGrab).toHaveBeenCalledTimes(1);
    });

    it("drops a grabbed object on the right button", () => {
        const target = mesh("crate");
        const harness = createHarness();
        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 2 });
        harness.events.emit("weaponTriggerReleased", {});
        harness.events.emit("weaponSecondaryPressed", {});
        expect(harness.runtime.releaseGrab).toHaveBeenLastCalledWith(0);
    });

    it("drops a held body when holstered", () => {
        const harness = createHarness();
        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: mesh("crate"), point: null, distance: 2 });

        harness.events.emit("weaponSlotSelected", { slot: 2 });

        expect(harness.runtime.releaseGrab).toHaveBeenLastCalledWith(0);
        expect(harness.runtime.setEnabled).toHaveBeenLastCalledWith(false, true);
    });

    it("validates parameters", () => {
        expect(() => createHarness({ legacyWeaponMaxGrabDistance: 0 })).toThrow("maxGrabDistance");
        expect(() => createHarness({ maxMass: Number.NaN })).toThrow("maxMass");
    });

    it("stops short of incoming collisions but permits movement away from a contact", () => {
        expect(antiGravityCollisionMoveFraction([1, 0, 0], { hasHit: false, fraction: 0, hitNormal: { x: 0, y: 0, z: 0 } })).toBe(1);
        expect(antiGravityCollisionMoveFraction([1, 0, 0], { hasHit: true, fraction: 0.5, hitNormal: { x: -1, y: 0, z: 0 } })).toBeCloseTo(0.49);
        expect(antiGravityCollisionMoveFraction([0, 1, 0], { hasHit: true, fraction: 0, hitNormal: { x: 0, y: 1, z: 0 } })).toBe(1);
    });
});
