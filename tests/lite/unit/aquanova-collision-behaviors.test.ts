import { describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../packages/babylon-lite/src";
import { createSceneNode } from "../../../packages/babylon-lite/src/scene/scene-node";
import { AquanovaEventManager } from "../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import { SetCollisionShapeBehavior } from "../../../lab/lite/src/demos/aquanova/behaviors/set-collision-shape";
import { TriggerBehavior } from "../../../lab/lite/src/demos/aquanova/behaviors/trigger";

function mesh(name: string): Mesh {
    return createSceneNode(name) as Mesh;
}

describe("Aquanova collision behaviors", () => {
    it("sets an entity AABB collision shape by default", () => {
        const setCollisionShape = vi.fn();
        const behavior = new SetCollisionShapeBehavior("pipe", [mesh("pipe")], {}, { setCollisionShape });

        behavior.start();

        expect(setCollisionShape).toHaveBeenCalledWith("pipe", "aabb");
    });

    it("supports explicit mesh shapes and rejects unsupported types", () => {
        const setCollisionShape = vi.fn();
        const behavior = new SetCollisionShapeBehavior("pipe", [mesh("pipe")], { type: "mesh" }, { setCollisionShape });
        behavior.start();
        expect(setCollisionShape).toHaveBeenCalledWith("pipe", "mesh");
        expect(() => new SetCollisionShapeBehavior("pipe", [mesh("pipe")], { type: "box" } as never, { setCollisionShape: vi.fn() })).toThrow(
            'setCollisionShape.type must be "mesh" when provided'
        );
    });

    it("raises entry and exit events on the trigger owner and forwards player filtering", () => {
        const events = new AquanovaEventManager();
        const setEnabled = vi.fn();
        const dispose = vi.fn();
        let enter = (): void => {};
        let exit = (): void => {};
        const registerIntersectionTrigger = vi.fn((_entityName: string, _playerOnly: boolean, callbacks: { onEntered(): void; onExited(): void }) => {
            enter = callbacks.onEntered;
            exit = callbacks.onExited;
            return { setEnabled, dispose };
        });
        const raised = vi.fn();
        events.on("entityEvent", raised);
        const behavior = new TriggerBehavior(
            "trapTrigger",
            [mesh("trigger")],
            { onIntersection: { enterEvent: "activated", exitEvent: "deactivated", playerOnly: true } },
            { events, registerIntersectionTrigger }
        );

        behavior.start();
        enter();
        exit();

        expect(registerIntersectionTrigger).toHaveBeenCalledWith("trapTrigger", true, {
            onEntered: expect.any(Function),
            onExited: expect.any(Function),
        });
        expect(raised.mock.calls.map(([event]) => event)).toEqual([
            { name: "trapTrigger", event: "activated" },
            { name: "trapTrigger", event: "deactivated" },
        ]);
    });

    it("remains silent while disabled", () => {
        const events = new AquanovaEventManager();
        const setEnabled = vi.fn();
        let enter = (): void => {};
        let exit = (): void => {};
        const behavior = new TriggerBehavior(
            "trapTrigger",
            [mesh("trigger")],
            { onIntersection: { enterEvent: "activated", exitEvent: "deactivated" } },
            {
                events,
                registerIntersectionTrigger: (_entityName, _playerOnly, callbacks) => {
                    enter = callbacks.onEntered;
                    exit = callbacks.onExited;
                    return { setEnabled, dispose: vi.fn() };
                },
            }
        );
        const raised = vi.fn();
        events.on("entityEvent", raised);
        behavior.start();

        events.emit("entityEvent", { name: "trapTrigger", event: "disable" });
        enter();
        exit();
        expect(raised).not.toHaveBeenCalledWith({ name: "trapTrigger", event: "activated" });
        expect(raised).not.toHaveBeenCalledWith({ name: "trapTrigger", event: "deactivated" });
        expect(setEnabled).toHaveBeenLastCalledWith(false);

        events.emit("entityEvent", { name: "trapTrigger", event: "enable" });
        enter();
        expect(raised).toHaveBeenLastCalledWith({ name: "trapTrigger", event: "activated" });
        expect(setEnabled).toHaveBeenLastCalledWith(true);
    });

    it("keeps legacy trigger routing working during migration", () => {
        const events = new AquanovaEventManager();
        let enter = (): void => {};
        const behavior = new TriggerBehavior(
            "trapTrigger",
            [mesh("trigger")],
            { onIntersection: { raiseEvent: "activated", entity: "trap" } },
            {
                events,
                registerIntersectionTrigger: (_entityName, _playerOnly, callbacks) => {
                    enter = callbacks.onEntered;
                    return { setEnabled: vi.fn(), dispose: vi.fn() };
                },
            }
        );
        const raised = vi.fn();
        events.on("entityEvent", raised);
        behavior.start();

        enter();

        expect(raised).toHaveBeenLastCalledWith({ name: "trap", event: "activated" });
    });

    it("validates trigger event configuration", () => {
        const context = { events: new AquanovaEventManager(), registerIntersectionTrigger: vi.fn() };
        expect(() => new TriggerBehavior("trigger", [mesh("trigger")], { onIntersection: {} }, context)).not.toThrow();
        expect(() => new TriggerBehavior("trigger", [mesh("trigger")], { onIntersection: { enterEvent: "" } }, context)).toThrow(
            "trigger.onIntersection.enterEvent must be a non-empty event name when provided"
        );
        expect(() => new TriggerBehavior("trigger", [mesh("trigger")], { onIntersection: { raiseEvent: "open", entity: "" } }, context)).toThrow(
            "trigger.onIntersection.entity must be a non-empty entity name when provided"
        );
        expect(() => new TriggerBehavior("trigger", [mesh("trigger")], { onIntersection: { enterEvent: "open", raiseEvent: "legacy" } }, context)).toThrow(
            "trigger.onIntersection cannot combine enterEvent/exitEvent with legacy raiseEvent"
        );
        expect(() => new TriggerBehavior("trigger", [mesh("trigger")], { onIntersection: { enterEvent: "open", entity: "target" } }, context)).toThrow(
            "trigger.onIntersection.entity is only supported with legacy raiseEvent"
        );
    });

    it("allows a trigger with neither entry nor exit event", () => {
        const events = new AquanovaEventManager();
        let enter = (): void => {};
        let exit = (): void => {};
        const raised = vi.fn();
        events.on("entityEvent", raised);
        const behavior = new TriggerBehavior(
            "trigger",
            [mesh("trigger")],
            { onIntersection: {} },
            {
                events,
                registerIntersectionTrigger: (_entityName, _playerOnly, callbacks) => {
                    enter = callbacks.onEntered;
                    exit = callbacks.onExited;
                    return { setEnabled: vi.fn(), dispose: vi.fn() };
                },
            }
        );

        behavior.start();
        enter();
        exit();

        expect(raised).not.toHaveBeenCalled();
    });
});
