import { describe, expect, it, vi } from "vitest";
import type { PhysicsBody, PhysicsShape, PhysicsWorld } from "../../../packages/babylon-lite/src";
import { shapeCast } from "../../../packages/babylon-lite/src/physics/havok-queries";

describe("Havok shape queries", () => {
    it("passes ignored body ids to shape casts", () => {
        const cast = vi.fn();
        const hknp = {
            HP_QueryCollector_Create: vi.fn(() => [0, "collector"]),
            HP_World_ShapeCastWithCollector: cast,
            HP_QueryCollector_GetNumHits: vi.fn(() => [0, 0]),
        };
        const ignoredBody = { _hkBody: [42n] } as PhysicsBody;
        const world = { _hknp: hknp, _hkWorld: "world" } as unknown as PhysicsWorld;
        const shape = { _hkShape: "shape" } as PhysicsShape;

        shapeCast(world, {
            shape,
            rotation: { x: 0, y: 0, z: 0, w: 1 },
            startPosition: { x: 1, y: 2, z: 3 },
            endPosition: { x: 4, y: 5, z: 6 },
            ignoreBodies: [ignoredBody],
        });

        expect(cast).toHaveBeenCalledWith("world", "collector", ["shape", [0, 0, 0, 1], [1, 2, 3], [4, 5, 6], false, [42n]]);
    });
});
