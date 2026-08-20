import { describe, expect, it, vi } from "vitest";
import type { PhysicsBody, PhysicsShape, PhysicsWorld } from "../../../packages/babylon-lite/src";
import { createIntersectionTriggerRegistry } from "../../../lab/lite/src/demos/aquanova/intersection-triggers";

function body(id: bigint, shapeId: number): PhysicsBody {
    return { _hkBody: [id], _shape: { _hkShape: shapeId } as PhysicsShape } as PhysicsBody;
}

describe("Aquanova intersection trigger registry", () => {
    it("filters player-only overlaps and disables trigger collision", () => {
        const memory = new ArrayBuffer(128);
        const address = 16;
        const event = new Int32Array(memory, address);
        event[0] = 8;
        const triggerBody = body(1n, 11);
        const playerBody = body(2n, 22);
        const propBody = body(3n, 33);
        const hknp = {
            HEAPU8: new Uint8Array(memory),
            HP_World_GetTriggerEvents: vi.fn(() => [0, address]),
            HP_World_GetNextTriggerEvent: vi.fn(() => 0),
            HP_Shape_SetTrigger: vi.fn(),
            HP_Shape_GetFilterInfo: vi.fn(() => [0, [0xffffffff, 0xffffffff]]),
            HP_Shape_SetFilterInfo: vi.fn(),
        };
        const world = { _hknp: hknp, _hkWorld: {}, _bodies: [triggerBody, playerBody, propBody], _afterStep: [] } as unknown as PhysicsWorld;
        const entered = vi.fn();
        const exited = vi.fn();
        const registry = createIntersectionTriggerRegistry(world, playerBody, () => [triggerBody]);
        const registration = registry.register("trigger", true, { onEntered: entered, onExited: exited });

        event[2] = 1;
        event[6] = 3;
        world._afterStep![0]!(1 / 60);
        expect(entered).not.toHaveBeenCalled();

        event[6] = 2;
        world._afterStep![0]!(1 / 60);
        expect(entered).toHaveBeenCalledTimes(1);

        event[0] = 16;
        world._afterStep![0]!(1 / 60);
        expect(exited).toHaveBeenCalledTimes(1);

        registration.setEnabled(false);
        event[0] = 8;
        world._afterStep![0]!(1 / 60);
        expect(entered).toHaveBeenCalledTimes(1);
        expect(hknp.HP_Shape_SetFilterInfo).toHaveBeenLastCalledWith(11, [0xffffffff, 0]);
    });

    it("raises one entry for the first overlap and one exit after the last overlap", () => {
        const memory = new ArrayBuffer(128);
        const address = 16;
        const event = new Int32Array(memory, address);
        const triggerBody = body(1n, 11);
        const firstBody = body(2n, 22);
        const secondBody = body(3n, 33);
        const hknp = {
            HEAPU8: new Uint8Array(memory),
            HP_World_GetTriggerEvents: vi.fn(() => [0, address]),
            HP_World_GetNextTriggerEvent: vi.fn(() => 0),
            HP_Shape_SetTrigger: vi.fn(),
            HP_Shape_GetFilterInfo: vi.fn(() => [0, [0xffffffff, 0xffffffff]]),
            HP_Shape_SetFilterInfo: vi.fn(),
        };
        const world = { _hknp: hknp, _hkWorld: {}, _bodies: [triggerBody, firstBody, secondBody], _afterStep: [] } as unknown as PhysicsWorld;
        const entered = vi.fn();
        const exited = vi.fn();
        const registry = createIntersectionTriggerRegistry(world, firstBody, () => [triggerBody]);
        registry.register("trigger", false, { onEntered: entered, onExited: exited });

        event[0] = 8;
        event[2] = 1;
        event[6] = 2;
        world._afterStep![0]!(1 / 60);
        event[6] = 3;
        world._afterStep![0]!(1 / 60);
        expect(entered).toHaveBeenCalledTimes(1);

        event[0] = 16;
        event[6] = 2;
        world._afterStep![0]!(1 / 60);
        expect(exited).not.toHaveBeenCalled();
        event[6] = 3;
        world._afterStep![0]!(1 / 60);
        expect(exited).toHaveBeenCalledTimes(1);
    });
});
