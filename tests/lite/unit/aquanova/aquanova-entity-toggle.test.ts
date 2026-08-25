import { describe, expect, it, vi } from "vitest";
import type { Mesh, SceneContext } from "../../../../packages/babylon-lite/src";
import { createSceneNode } from "../../../../packages/babylon-lite/src/scene/scene-node";
import {
    DisableCollisionBehavior,
    DisableEntityBehavior,
    EnableCollisionBehavior,
    EnableEntityBehavior,
    HideEntityBehavior,
    RemoveEntityBehavior,
    ShowEntityBehavior,
} from "../../../../lab/lite/src/demos/aquanova/behaviors/entity-toggle";
import { AquanovaBehaviorManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-behavior-manager";
import { AquanovaEventManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import { registerEntityCollisionEventHandlers } from "../../../../lab/lite/src/demos/aquanova/entity-collision-events";
import { createPersistentMeshEntityEventOperations, registerMeshEntityEventHandlers } from "../../../../lab/lite/src/demos/aquanova/entity-events";
import { registerDoorEntityEventHandlers } from "../../../../lab/lite/src/demos/aquanova/door-events";

function mesh(name: string): Mesh {
    return createSceneNode(name) as Mesh;
}

describe("Aquanova entity toggle behaviors", () => {
    it("applies matching source events to the behavior owner", () => {
        const events = new AquanovaEventManager();
        const disableCollision = new DisableCollisionBehavior("Prop_A", [mesh("prop")], { events: [{ name: "startLiquefaction", source: "capsule" }] }, { events });
        const enable = new EnableEntityBehavior("Door_D00", [], { events: [{ name: "startLiquefaction", source: "capsule" }] }, { events });
        const disable = new DisableEntityBehavior("Door_D00", [], { events: [{ name: "cancelLiquefaction", source: "capsule" }] }, { events });
        const enableCollision = new EnableCollisionBehavior("Prop_A", [mesh("prop")], { events: [{ name: "cancelLiquefaction", source: "capsule" }] }, { events });
        const hide = new HideEntityBehavior("Prop_A", [mesh("prop")], { events: [{ name: "hideProp", source: "controlPanel" }] }, { events });
        const remove = new RemoveEntityBehavior("Prop_A", [mesh("prop")], { events: [{ name: "removeProp", source: "controlPanel" }] }, { events });
        const show = new ShowEntityBehavior(
            "Prop_A",
            [mesh("prop")],
            {
                events: [
                    { name: "showProp", source: "controlPanel" },
                    { name: "activated", source: ["trapTrigger", "backupTrapTrigger"] },
                    { name: "opened", source: ["Door_D01", "Door_D02"] },
                ],
            },
            { events }
        );
        const forwarded = vi.fn();
        events.on("entityEvent", forwarded);
        disableCollision.start();
        enable.start();
        disable.start();
        enableCollision.start();
        hide.start();
        remove.start();
        show.start();

        events.emit("entityEvent", { name: "otherEntity", event: "startLiquefaction" });
        events.emit("entityEvent", { name: "capsule", event: "otherEvent" });
        events.emit("entityEvent", { name: "capsule", event: "startLiquefaction" });
        events.emit("entityEvent", { name: "capsule", event: "cancelLiquefaction" });
        events.emit("entityEvent", { name: "controlPanel", event: "hideProp" });
        events.emit("entityEvent", { name: "controlPanel", event: "removeProp" });
        events.emit("entityEvent", { name: "trapTrigger", event: "activated" });
        events.emit("entityEvent", { name: "Door_D01", event: "opened" });
        events.emit("entityEvent", { name: "backupTrapTrigger", event: "activated" });
        events.emit("entityEvent", { name: "Door_D02", event: "opened" });

        expect(forwarded.mock.calls.map(([event]) => event)).toEqual([
            { name: "otherEntity", event: "startLiquefaction" },
            { name: "capsule", event: "otherEvent" },
            { name: "capsule", event: "startLiquefaction" },
            { name: "Prop_A", event: "disableCollision" },
            { name: "Door_D00", event: "enable" },
            { name: "capsule", event: "cancelLiquefaction" },
            { name: "Door_D00", event: "disable" },
            { name: "Prop_A", event: "enableCollision" },
            { name: "controlPanel", event: "hideProp" },
            { name: "Prop_A", event: "hide" },
            { name: "controlPanel", event: "removeProp" },
            { name: "Prop_A", event: "remove" },
            { name: "trapTrigger", event: "activated" },
            { name: "Prop_A", event: "show" },
            { name: "Door_D01", event: "opened" },
            { name: "Prop_A", event: "show" },
            { name: "backupTrapTrigger", event: "activated" },
            { name: "Prop_A", event: "show" },
            { name: "Door_D02", event: "opened" },
            { name: "Prop_A", event: "show" },
        ]);

        disableCollision.dispose();
        enable.dispose();
        disable.dispose();
        enableCollision.dispose();
        hide.dispose();
        remove.dispose();
        show.dispose();
    });

    it("validates event subscriptions", () => {
        expect(() => new EnableEntityBehavior("target", [], { events: [] }, { events: new AquanovaEventManager() })).toThrow("enableEntity.events must contain at least one event");
        expect(() => new DisableEntityBehavior("target", [], { events: [{ name: "cancel", source: "" }] }, { events: new AquanovaEventManager() })).toThrow(
            "disableEntity.events[].source must be a non-empty entity or door name"
        );
        expect(() => new DisableEntityBehavior("target", [], { events: [{ name: "cancel", source: [] }] }, { events: new AquanovaEventManager() })).toThrow(
            "disableEntity.events[].source must contain at least one entity or door name"
        );
        expect(() => new DisableEntityBehavior("target", [], { events: [{ name: "cancel", source: ["source", ""] }] }, { events: new AquanovaEventManager() })).toThrow(
            "disableEntity.events[].source entries must be non-empty entity or door names"
        );
        expect(() => new RemoveEntityBehavior("target", [mesh("target")], { events: [{ name: "", source: "source" }] }, { events: new AquanovaEventManager() })).toThrow(
            "removeEntity.events[].name must be a non-empty event name"
        );
        expect(() => new ShowEntityBehavior("target", [mesh("target")], { unexpected: true } as never, { events: new AquanovaEventManager() })).toThrow(
            "showEntity.unexpected is not supported"
        );
    });

    it("instantiates behaviors owned by meshless doors", async () => {
        const door = {
            id: "Door_D06",
            enabled: false,
            behaviors: [{ name: "enableEntity", events: [{ name: "startLiquefaction", source: "capsule" }] }],
        };
        const manager = new AquanovaBehaviorManager({
            entities: undefined,
            doors: [door],
            meshesByEntityName: new Map(),
            entityNameOf: (value) => value.name,
        });
        const forwarded = vi.fn();
        const setDoorEnabled = vi.fn();
        manager.events.on("entityEvent", forwarded);
        const stopDoorEvents = registerDoorEntityEventHandlers(manager.events, [door], setDoorEnabled);

        await manager.start({} as never);
        manager.events.emit("entityEvent", { name: "capsule", event: "startLiquefaction" });

        expect(manager.instances).toHaveLength(1);
        expect(manager.instances[0]?.mesh).toBeNull();
        expect(manager.describeInstances()).toEqual([{ name: "enableEntity", mesh: "Door_D06" }]);
        expect(forwarded).toHaveBeenLastCalledWith({ name: "Door_D06", event: "enable" });
        expect(door.enabled).toBe(true);
        expect(setDoorEnabled).toHaveBeenCalledWith("Door_D06", true);

        stopDoorEvents();
        manager.dispose();
    });

    it("removes, hides, and shows every mesh belonging to the target entity", () => {
        const events = new AquanovaEventManager();
        const first = mesh("first");
        const second = mesh("second");
        const scene = {} as SceneContext;
        const hidden = new Set<Mesh>();
        const remove = vi.fn();
        const setVisible = vi.fn();
        registerMeshEntityEventHandlers(events, scene, new Map([["Prop_A", [first, second]]]), createPersistentMeshEntityEventOperations(hidden, { remove, setVisible }));

        events.emit("entityEvent", { name: "Other", event: "hide" });
        events.emit("entityEvent", { name: "Prop_A", event: "disable" });
        events.emit("entityEvent", { name: "Prop_A", event: "hide" });
        expect(hidden).toEqual(new Set([first, second]));
        events.emit("entityEvent", { name: "Prop_A", event: "show" });
        expect(hidden).toEqual(new Set());
        events.emit("entityEvent", { name: "Prop_A", event: "remove" });
        expect(hidden).toEqual(new Set([first, second]));

        expect(setVisible.mock.calls).toEqual([
            [first, false],
            [second, false],
            [first, true],
            [second, true],
        ]);
        expect(remove.mock.calls).toEqual([
            [scene, first],
            [scene, second],
        ]);
    });

    it("disables and enables collision for the target entity", () => {
        const events = new AquanovaEventManager();
        const setCollisionActive = vi.fn();
        registerEntityCollisionEventHandlers(events, setCollisionActive);

        events.emit("entityEvent", { name: "Prop_A", event: "hide" });
        events.emit("entityEvent", { name: "Prop_A", event: "disableCollision" });
        events.emit("entityEvent", { name: "Prop_A", event: "enableCollision" });
        events.emit("entityEvent", { name: "Prop_A", event: "remove" });

        expect(setCollisionActive.mock.calls).toEqual([
            ["Prop_A", false],
            ["Prop_A", true],
            ["Prop_A", false],
        ]);
    });

    it("applies collision marker behaviors immediately to their owning entity", () => {
        const events = new AquanovaEventManager();
        const setCollisionActive = vi.fn();
        registerEntityCollisionEventHandlers(events, setCollisionActive);
        const disable = new DisableCollisionBehavior("weaponAntiGravityGunHolder", [mesh("holder")], {}, { events });
        const enable = new EnableCollisionBehavior("weaponAntiGravityGunHolder", [mesh("holder")], {}, { events });

        disable.start();
        enable.start();

        expect(setCollisionActive.mock.calls).toEqual([
            ["weaponAntiGravityGunHolder", false],
            ["weaponAntiGravityGunHolder", true],
        ]);
    });
});
