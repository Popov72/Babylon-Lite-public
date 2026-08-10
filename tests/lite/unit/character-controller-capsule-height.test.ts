import { describe, expect, it, vi } from "vitest";

import type { Vec3 } from "../../../packages/babylon-lite/src/math/types";
import { PhysicsCharacterController } from "../../../packages/babylon-lite/src/physics/character-controller";
import type { PhysicsBody, PhysicsShape, PhysicsWorld } from "../../../packages/babylon-lite/src/physics/havok";

interface MutableController {
    _world: PhysicsWorld;
    _shape: PhysicsShape;
    _node: { position: { set: (x: number, y: number, z: number) => void } };
    _body: PhysicsBody;
    _startCollector: unknown;
    _capsuleRadius: number;
    _capsuleHeight: number;
    _position: Vec3;
    _orientation: { x: number; y: number; z: number; w: number };
    _manifold: unknown[];
}

function makeController(height: number, y: number, proximityDistances: number[] = []) {
    const oldShape = { _hkShape: "old-shape", _type: 2 } as unknown as PhysicsShape;
    const body = { _hkBody: ["body"], _shape: oldShape } as unknown as PhysicsBody;
    const setPosition = vi.fn();
    const hknp = {
        HP_Shape_CreateCapsule: vi.fn(() => [0, "next-shape"]),
        HP_World_ShapeProximityWithCollector: vi.fn(),
        HP_QueryCollector_GetNumHits: vi.fn(() => [0, proximityDistances.length]),
        HP_QueryCollector_GetShapeProximityResult: vi.fn((_collector: unknown, index: number) => [0, [proximityDistances[index]]]),
        HP_Body_SetShape: vi.fn(),
        HP_Shape_Release: vi.fn(),
    };
    const world = { _hknp: hknp, _hkWorld: "world" } as unknown as PhysicsWorld;
    const raw = Object.create(PhysicsCharacterController.prototype) as MutableController;
    raw._world = world;
    raw._shape = oldShape;
    raw._node = { position: { set: setPosition } };
    raw._body = body;
    raw._startCollector = "collector";
    raw._capsuleRadius = 0.4;
    raw._capsuleHeight = height;
    raw._position = { x: 2, y, z: 3 };
    raw._orientation = { x: 0, y: 0, z: 0, w: 1 };
    raw._manifold = [{ contact: true }];
    return { cc: raw as unknown as PhysicsCharacterController, hknp, body, setPosition };
}

describe("character controller capsule height", () => {
    it("shrinks to a sphere while preserving the foot position", () => {
        const { cc, hknp, body, setPosition } = makeController(1.8, 0.9);

        expect(cc.trySetCapsuleHeight(0.8)).toBe(true);

        expect(cc.getCapsuleHeight()).toBe(0.8);
        expect(cc.getPosition()).toEqual({ x: 2, y: 0.4, z: 3 });
        expect(hknp.HP_Shape_CreateCapsule).toHaveBeenCalledWith([0, 0, 0], [0, 0, 0], 0.4);
        expect(hknp.HP_World_ShapeProximityWithCollector).not.toHaveBeenCalled();
        expect(hknp.HP_Body_SetShape).toHaveBeenCalledWith(body._hkBody, "next-shape");
        expect(hknp.HP_Shape_Release).toHaveBeenCalledWith("old-shape");
        expect(setPosition).toHaveBeenCalledWith(2, 0.4, 3);
    });

    it("rejects standing expansion when the taller capsule overlaps geometry", () => {
        const { cc, hknp, setPosition } = makeController(0.8, 0.4, [-0.01]);

        expect(cc.trySetCapsuleHeight(1.8)).toBe(false);

        expect(cc.getCapsuleHeight()).toBe(0.8);
        expect(cc.getPosition()).toEqual({ x: 2, y: 0.4, z: 3 });
        expect(hknp.HP_Body_SetShape).not.toHaveBeenCalled();
        expect(hknp.HP_Shape_Release).toHaveBeenCalledWith("next-shape");
        expect(setPosition).not.toHaveBeenCalled();
    });

    it("allows standing expansion when geometry only touches the proposed capsule", () => {
        const { cc, hknp, setPosition } = makeController(0.8, 0.4, [0]);

        expect(cc.trySetCapsuleHeight(1.8)).toBe(true);

        expect(cc.getCapsuleHeight()).toBe(1.8);
        expect(cc.getPosition()).toEqual({ x: 2, y: 0.9, z: 3 });
        expect(hknp.HP_Body_SetShape).toHaveBeenCalledTimes(1);
        expect(hknp.HP_Shape_Release).toHaveBeenCalledWith("old-shape");
        expect(setPosition).toHaveBeenCalledWith(2, 0.9, 3);
    });
});
