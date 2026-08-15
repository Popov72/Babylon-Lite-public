import type { EventManager } from "./behaviors/event-manager.js";
import type { ShipDoor } from "./manifest.js";

export function registerDoorEntityEventHandlers(events: EventManager, doors: readonly ShipDoor[], setDoorEnabled: (door: string, enabled: boolean) => void): () => void {
    const disposers: Array<() => void> = [];
    for (const door of doors) {
        const apply = (enabled: boolean): void => {
            door.enabled = enabled;
            setDoorEnabled(door.id, enabled);
        };
        disposers.push(
            events.on("entityEvent", ({ name, event }) => {
                if (name === door.id && event === "enable") apply(true);
            }),
            events.on("entityEvent", ({ name, event }) => {
                if (name === door.id && event === "disable") apply(false);
            })
        );
    }
    return () => {
        for (const dispose of disposers.splice(0)) dispose();
    };
}
