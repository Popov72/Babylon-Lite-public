import { describe, expect, it, vi } from "vitest";
import { EventManager } from "../../../lab/lite/src/demos/aquanova/behaviors/event-manager";
import { WeaponInventory } from "../../../lab/lite/src/demos/aquanova/behaviors/weapon-inventory";

describe("Aquanova weapon inventory", () => {
    it("selects acquired slots, toggles the selected slot, and holsters for an unowned slot", () => {
        const events = new EventManager();
        const inventory = new WeaponInventory();
        const changed = vi.fn();
        events.on("weaponEquippedChanged", changed);
        inventory.start(events);

        inventory.acquire(1);
        expect(inventory.isEquipped(1)).toBe(true);

        events.emit("weaponSlotSelected", { slot: 1 });
        expect(inventory.isEquipped(1)).toBe(false);

        events.emit("weaponSlotSelected", { slot: 1 });
        expect(inventory.isEquipped(1)).toBe(true);

        events.emit("weaponSlotSelected", { slot: 2 });
        expect(inventory.isEquipped(1)).toBe(false);
        expect(changed.mock.calls.map(([event]) => event.slot)).toEqual([1, null, 1, null]);
    });

    it("cycles both directions through owned weapons and the hidden state", () => {
        const events = new EventManager();
        const inventory = new WeaponInventory();
        inventory.start(events);
        inventory.acquire(1);
        inventory.acquire(2);

        events.emit("weaponCycleRequested", { direction: 1 });
        expect(inventory.isEquipped(1)).toBe(false);
        expect(inventory.isEquipped(2)).toBe(false);

        events.emit("weaponCycleRequested", { direction: 1 });
        expect(inventory.isEquipped(1)).toBe(true);

        events.emit("weaponCycleRequested", { direction: -1 });
        expect(inventory.isEquipped(1)).toBe(false);
        expect(inventory.isEquipped(2)).toBe(false);

        events.emit("weaponCycleRequested", { direction: -1 });
        expect(inventory.isEquipped(2)).toBe(true);
    });
});
