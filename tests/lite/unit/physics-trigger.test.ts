import { describe, expect, it, vi } from "vitest";
import type { PhysicsBody, PhysicsWorld } from "../../../packages/babylon-lite/src/physics/havok";
import { onPhysicsTriggerBodies } from "../../../packages/babylon-lite/src/physics/havok-trigger";

describe("Havok trigger events", () => {
    it("resolves both native body ids and supports disposal", () => {
        const memory = new ArrayBuffer(128);
        const eventAddress = 16;
        const event = new Int32Array(memory, eventAddress);
        event[0] = 8;
        event[2] = 101;
        event[6] = 202;
        const bodyA = { _hkBody: [101] } as PhysicsBody;
        const bodyB = { _hkBody: [202] } as PhysicsBody;
        const world = {
            _hknp: {
                HEAPU8: new Uint8Array(memory),
                HP_World_GetTriggerEvents: vi.fn(() => [0, eventAddress]),
                HP_World_GetNextTriggerEvent: vi.fn(() => 0),
            },
            _hkWorld: {},
            _bodies: [bodyA, bodyB],
            _afterStep: [],
        } as unknown as PhysicsWorld;
        const received = vi.fn();

        const dispose = onPhysicsTriggerBodies(world, received);
        world._afterStep![0]!(1 / 60);

        expect(received).toHaveBeenCalledWith({ type: "ENTERED", bodyA, bodyB });
        dispose();
        expect(world._afterStep).toHaveLength(0);
    });
});
