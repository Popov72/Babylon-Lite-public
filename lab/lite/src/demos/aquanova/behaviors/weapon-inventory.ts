import type { EventManager } from "./event-manager.js";
import type { WeaponInventoryRuntime } from "./types.js";

export class WeaponInventory implements WeaponInventoryRuntime {
    private readonly owned = new Set<number>();
    private readonly disposers: Array<() => void> = [];
    private events: EventManager | null = null;
    private equipped: number | null = null;

    public start(events: EventManager): void {
        if (this.events) {
            throw new Error("[aquanova] weapon inventory is already started");
        }
        this.events = events;
        this.disposers.push(
            events.on("weaponSlotSelected", ({ slot }) => this.select(slot)),
            events.on("weaponCycleRequested", ({ direction }) => this.cycle(direction))
        );
    }

    public acquire(slot: number): void {
        validateSlot(slot);
        this.owned.add(slot);
        this.setEquipped(slot);
    }

    public isOwned(slot: number): boolean {
        return this.owned.has(slot);
    }

    public isEquipped(slot: number): boolean {
        return this.equipped === slot;
    }

    public dispose(): void {
        for (const dispose of this.disposers.splice(0)) {
            dispose();
        }
        this.owned.clear();
        this.events = null;
        this.equipped = null;
    }

    private select(slot: number): void {
        validateSlot(slot);
        this.setEquipped(this.owned.has(slot) && this.equipped !== slot ? slot : null);
    }

    private cycle(direction: -1 | 1): void {
        if (this.owned.size === 0) {
            return;
        }
        const slots: Array<number | null> = [...this.owned].sort((left, right) => left - right);
        slots.push(null);
        const current = slots.indexOf(this.equipped);
        const next = (current + direction + slots.length) % slots.length;
        this.setEquipped(slots[next] ?? null);
    }

    private setEquipped(slot: number | null): void {
        if (this.equipped === slot) {
            return;
        }
        this.equipped = slot;
        this.events?.emit("weaponEquippedChanged", { slot });
    }
}

function validateSlot(slot: number): void {
    if (!Number.isInteger(slot) || slot <= 0) {
        throw new Error(`[aquanova] weapon slot must be a positive integer, received ${String(slot)}`);
    }
}
