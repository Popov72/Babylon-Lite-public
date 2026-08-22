import { describe, expect, it, vi } from "vitest";
import { registerDoorEntityEventHandlers } from "../../../../lab/lite/src/demos/aquanova/door-events";
import { AquanovaEventManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";

describe("Aquanova door entity events", () => {
    it("updates each addressed door and its portal state", () => {
        const events = new AquanovaEventManager();
        const doors = [
            { id: "Door_D00", enabled: false },
            { id: "Door_D01", enabled: true },
        ];
        const setDoorEnabled = vi.fn();
        const dispose = registerDoorEntityEventHandlers(events, doors, setDoorEnabled);

        events.emit("entityEvent", { name: "Door_D00", event: "enable" });
        events.emit("entityEvent", { name: "Door_D01", event: "disable" });
        events.emit("entityEvent", { name: "Door_D00", event: "other" });

        expect(doors).toEqual([
            { id: "Door_D00", enabled: true },
            { id: "Door_D01", enabled: false },
        ]);
        expect(setDoorEnabled.mock.calls).toEqual([
            ["Door_D00", true],
            ["Door_D01", false],
        ]);

        dispose();
        events.emit("entityEvent", { name: "Door_D00", event: "disable" });
        expect(doors[0]!.enabled).toBe(true);
    });
});
