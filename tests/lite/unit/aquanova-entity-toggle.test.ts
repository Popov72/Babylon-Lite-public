import { describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../packages/babylon-lite/src";
import { createSceneNode } from "../../../packages/babylon-lite/src/scene/scene-node";
import { DisableEntityBehavior, EnableEntityBehavior } from "../../../lab/lite/src/demos/aquanova/behaviors/entity-toggle";
import { EventManager } from "../../../lab/lite/src/demos/aquanova/behaviors/event-manager";

function mesh(name: string): Mesh {
    return createSceneNode(name) as Mesh;
}

describe("Aquanova entity toggle behaviors", () => {
    it("forwards configured source events as enable and disable events", () => {
        const events = new EventManager();
        const enable = new EnableEntityBehavior("storageDoorLF", mesh("door-leaf"), { onEvent: "startLiquefaction", entity: "Door_D00" }, { events });
        const disable = new DisableEntityBehavior("storageDoorLF", mesh("door-leaf"), { onEvent: "cancelLiquefaction", entity: "Door_D00" }, { events });
        const forwarded = vi.fn();
        events.on("entityEvent", forwarded);
        enable.start();
        disable.start();

        events.emit("entityEvent", { name: "otherEntity", event: "startLiquefaction" });
        events.emit("entityEvent", { name: "storageDoorLF", event: "otherEvent" });
        events.emit("entityEvent", { name: "storageDoorLF", event: "startLiquefaction" });
        events.emit("entityEvent", { name: "storageDoorLF", event: "cancelLiquefaction" });

        expect(forwarded.mock.calls.map(([event]) => event)).toEqual([
            { name: "otherEntity", event: "startLiquefaction" },
            { name: "storageDoorLF", event: "otherEvent" },
            { name: "storageDoorLF", event: "startLiquefaction" },
            { name: "Door_D00", event: "enable" },
            { name: "storageDoorLF", event: "cancelLiquefaction" },
            { name: "Door_D00", event: "disable" },
        ]);

        enable.dispose();
        disable.dispose();
    });

    it("rejects missing source events and targets", () => {
        expect(() => new EnableEntityBehavior("source", mesh("source"), { onEvent: "", entity: "target" }, { events: new EventManager() })).toThrow(
            "enableEntity.onEvent must be a non-empty event name"
        );
        expect(() => new DisableEntityBehavior("source", mesh("source"), { onEvent: "cancel", entity: "" }, { events: new EventManager() })).toThrow(
            "disableEntity.entity must be a non-empty entity or door name"
        );
    });
});
