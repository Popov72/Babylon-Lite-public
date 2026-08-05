// The analytic collision field the Aquanova fluid collides against (collision-field.ts).
//
// These test the CPU mirror of the WGSL, which exists precisely so this maths can be checked without
// a GPU. The stakes are asymmetric: a SIGN error reads as "water falls through the floor" and a
// magnitude error as "water floats a few centimetres above it", and both are easy to mistake for a
// solver problem. So every case pins an EXACT expected distance rather than just a sign.

import { describe, expect, it } from "vitest";
import {
    localizePrimitive,
    packPrimitives,
    primBufferBytes,
    primitiveSdf,
    primitivesSdf,
    PRIM_HEADER,
    PRIM_STRIDE,
    PRIM_BOX,
    PRIM_SPHERE,
    PRIM_CAPSULE,
    PRIM_CYLINDER,
    type FluidPrimitive,
} from "../../../lab/lite/src/demos/aquanova/collision-field";

/** Quaternion for a rotation of `deg` about `axis`. */
function quat(axis: [number, number, number], deg: number): [number, number, number, number] {
    const r = (deg * Math.PI) / 360; // half angle
    const n = Math.hypot(...axis) || 1;
    const s = Math.sin(r);
    return [(axis[0] / n) * s, (axis[1] / n) * s, (axis[2] / n) * s, Math.cos(r)];
}

describe("collision-field: sphere", () => {
    const s: FluidPrimitive = { kind: "sphere", a: [1, 2, 3], radius: 0.5 };
    it("is negative inside, zero on the surface, positive outside", () => {
        expect(primitiveSdf(s, [1, 2, 3])).toBeCloseTo(-0.5, 6); // centre
        expect(primitiveSdf(s, [1.5, 2, 3])).toBeCloseTo(0, 6); // on the surface
        expect(primitiveSdf(s, [3, 2, 3])).toBeCloseTo(1.5, 6); // 2 m out, minus the radius
    });
});

describe("collision-field: axis-aligned box", () => {
    const b: FluidPrimitive = { kind: "box", a: [0, 0, 0], b: [1, 2, 3] };
    it("returns the true distance outside, on each axis", () => {
        expect(primitiveSdf(b, [3, 0, 0])).toBeCloseTo(2, 6);
        expect(primitiveSdf(b, [0, 5, 0])).toBeCloseTo(3, 6);
        expect(primitiveSdf(b, [0, 0, 4])).toBeCloseTo(1, 6);
    });
    it("returns the distance to the NEAREST face inside (negative)", () => {
        expect(primitiveSdf(b, [0, 0, 0])).toBeCloseTo(-1, 6); // nearest face is x = ±1
        expect(primitiveSdf(b, [0.5, 0, 0])).toBeCloseTo(-0.5, 6);
    });
    it("returns the corner distance diagonally outside", () => {
        // (2,3,4) is (1,1,1) beyond the +x/+y/+z corner
        expect(primitiveSdf(b, [2, 3, 4])).toBeCloseTo(Math.sqrt(3), 6);
    });
});

describe("collision-field: oriented box", () => {
    it("rotating the box 90° about Y swaps its x and z extents", () => {
        const flat: FluidPrimitive = { kind: "box", a: [0, 0, 0], b: [1, 1, 4], rotation: quat([0, 1, 0], 90) };
        // The long axis (4) now runs along world X, the short one (1) along world Z.
        expect(primitiveSdf(flat, [3.5, 0, 0])).toBeCloseTo(-0.5, 5); // still inside along the long axis
        expect(primitiveSdf(flat, [0, 0, 3.5])).toBeCloseTo(2.5, 5); // well outside along the short one
    });
    it("a 45° turn puts the corner where the flat face used to be", () => {
        const d: FluidPrimitive = { kind: "box", a: [0, 0, 0], b: [1, 1, 1], rotation: quat([0, 1, 0], 45) };
        // Along +X the surface is now the corner, at sqrt(2) from the centre.
        expect(primitiveSdf(d, [Math.SQRT2, 0, 0])).toBeCloseTo(0, 5);
        expect(primitiveSdf(d, [0, 0, Math.SQRT2])).toBeCloseTo(0, 5);
    });
    it("an unrotated box and an identity-quaternion box agree", () => {
        const plain: FluidPrimitive = { kind: "box", a: [1, 2, 3], b: [1, 2, 3] };
        const ident: FluidPrimitive = { ...plain, rotation: [0, 0, 0, 1] };
        for (const p of [
            [0, 0, 0],
            [5, 5, 5],
            [1, 2, 3],
            [1, 6, 3],
        ] as Array<[number, number, number]>) {
            expect(primitiveSdf(ident, p)).toBeCloseTo(primitiveSdf(plain, p), 6);
        }
    });
    // The exporter writes identity as [0,0,0,-1]; -q is the same rotation and must decode the same.
    it("treats -q as the same rotation as q", () => {
        const q = quat([0.3, 1, 0.2], 37);
        const a: FluidPrimitive = { kind: "box", a: [0, 1, 0], b: [0.5, 1.5, 2], rotation: q };
        const b: FluidPrimitive = { ...a, rotation: [-q[0], -q[1], -q[2], -q[3]] };
        for (const p of [
            [0, 0, 0],
            [2, 2, 2],
            [-1, 1, 0.5],
        ] as Array<[number, number, number]>) {
            expect(primitiveSdf(b, p)).toBeCloseTo(primitiveSdf(a, p), 6);
        }
    });
});

describe("collision-field: capsule", () => {
    // Vertical segment from y=0 to y=2, radius 0.5 — a barrel.
    const c: FluidPrimitive = { kind: "capsule", a: [0, 0, 0], b: [0, 2, 0], radius: 0.5 };
    it("measures from the segment, not the end points", () => {
        expect(primitiveSdf(c, [0, 1, 0])).toBeCloseTo(-0.5, 6); // on the axis, mid-segment
        expect(primitiveSdf(c, [2, 1, 0])).toBeCloseTo(1.5, 6); // 2 m sideways
    });
    it("has ROUND caps: the end is a hemisphere", () => {
        expect(primitiveSdf(c, [0, 3, 0])).toBeCloseTo(0.5, 6); // 1 m above the top point
        expect(primitiveSdf(c, [0, -1, 0])).toBeCloseTo(0.5, 6);
    });
});

describe("collision-field: cylinder", () => {
    // Same segment as the capsule above, so the difference is only in the caps.
    const c: FluidPrimitive = { kind: "cylinder", a: [0, 0, 0], b: [0, 2, 0], radius: 0.5 };
    it("measures radially from the axis", () => {
        expect(primitiveSdf(c, [0, 1, 0])).toBeCloseTo(-0.5, 5);
        expect(primitiveSdf(c, [2, 1, 0])).toBeCloseTo(1.5, 5);
    });
    it("has FLAT caps, unlike a capsule", () => {
        // 1 m above the top face, on the axis: a flat cap is 1 m away, a round one would be 0.5.
        expect(primitiveSdf(c, [0, 3, 0])).toBeCloseTo(1, 5);
        const cap: FluidPrimitive = { ...c, kind: "capsule" };
        expect(primitiveSdf(cap, [0, 3, 0])).toBeCloseTo(0.5, 5);
    });
});

describe("collision-field: union", () => {
    it("takes the nearest solid (min)", () => {
        const prims: FluidPrimitive[] = [
            { kind: "sphere", a: [0, 0, 0], radius: 1 },
            { kind: "sphere", a: [10, 0, 0], radius: 1 },
        ];
        expect(primitivesSdf(prims, [5, 0, 0])).toBeCloseTo(4, 6); // 5 from each centre, minus r
        expect(primitivesSdf(prims, [10, 0, 0])).toBeCloseTo(-1, 6); // inside the second
    });
    it("is far-positive when empty, so an empty set adds no solid", () => {
        expect(primitivesSdf([], [0, 0, 0])).toBeGreaterThan(1e8);
    });
    // A floor slab is the case that matters most: water must rest ON it, not sink through.
    it("a floor slab reads solid below its top face and free above", () => {
        // Centre y = -0.25, half 0.25 → the slab spans y ∈ [-0.5, 0]: its TOP face is the walking
        // plane at y = 0, with the body hanging below (the editor's one-sided hull convention).
        const floor: FluidPrimitive = { kind: "box", a: [0, -0.25, 0], b: [10, 0.25, 10] };
        expect(primitiveSdf(floor, [0, 0.1, 0])).toBeCloseTo(0.1, 6); // just above: free
        expect(primitiveSdf(floor, [0, -0.1, 0])).toBeCloseTo(-0.1, 6); // just below: inside, and the
        // nearest face is the top one 0.1 away — not the bottom (0.4) or the distant sides (10).
        expect(primitiveSdf(floor, [0, 0, 0])).toBeCloseTo(0, 6); // exactly on the surface
        expect(primitiveSdf(floor, [0, -0.45, 0])).toBeCloseTo(-0.05, 6); // near the underside now
    });
});

describe("collision-field: velocity", () => {
    it("advances the primitive along its velocity, so -d(sdf)/dt is the boundary speed", () => {
        const moving: FluidPrimitive = { kind: "sphere", a: [0, 0, 0], radius: 1, velocity: [2, 0, 0] };
        expect(primitiveSdf(moving, [5, 0, 0], 0)).toBeCloseTo(4, 6);
        expect(primitiveSdf(moving, [5, 0, 0], 0.5)).toBeCloseTo(3, 6); // sphere moved 1 m closer
        const dt = 1e-3;
        const speed = -(primitiveSdf(moving, [5, 0, 0], dt) - primitiveSdf(moving, [5, 0, 0], 0)) / dt;
        expect(speed).toBeCloseTo(2, 4);
    });
    it("moves a capsule's whole segment, not just one end", () => {
        const m: FluidPrimitive = { kind: "capsule", a: [0, 0, 0], b: [0, 2, 0], radius: 0.5, velocity: [0, 1, 0] };
        // After 1 s the segment spans y = 1..3, so the point at y = 4 is 1 m above the top cap.
        expect(primitiveSdf(m, [0, 4, 0], 1)).toBeCloseTo(0.5, 6);
    });
});

describe("collision-field: packing", () => {
    it("round-trips every field into the buffer layout the WGSL reads", () => {
        const prims: FluidPrimitive[] = [
            { kind: "box", a: [1, 2, 3], b: [4, 5, 6], rotation: [0.1, 0.2, 0.3, 0.927], velocity: [7, 8, 9] },
            { kind: "sphere", a: [-1, -2, -3], radius: 2.5 },
            { kind: "capsule", a: [0, 0, 0], b: [0, 1, 0], radius: 0.25 },
            { kind: "cylinder", a: [1, 1, 1], b: [1, 3, 1], radius: 0.75 },
        ];
        const buf = new Float32Array(PRIM_HEADER + prims.length * PRIM_STRIDE);
        packPrimitives(buf, prims);
        expect(buf[0]).toBe(4); // header count — the WGSL loop bound

        const at = (i: number, k: number): number => buf[PRIM_HEADER + i * PRIM_STRIDE + k]!;
        expect(at(0, 0)).toBe(PRIM_BOX);
        expect([at(0, 1), at(0, 2), at(0, 3)]).toEqual([1, 2, 3]);
        expect([at(0, 4), at(0, 5), at(0, 6)]).toEqual([4, 5, 6]);
        expect([at(0, 8), at(0, 9), at(0, 10)]).toEqual([0.1, 0.2, 0.3].map((v) => Math.fround(v)));
        expect([at(0, 12), at(0, 13), at(0, 14)]).toEqual([7, 8, 9]);
        expect(at(1, 0)).toBe(PRIM_SPHERE);
        expect(at(1, 7)).toBe(2.5);
        expect(at(2, 0)).toBe(PRIM_CAPSULE);
        expect(at(3, 0)).toBe(PRIM_CYLINDER);
    });
    it("defaults the optional fields rather than leaving them undefined", () => {
        const buf = new Float32Array(PRIM_HEADER + PRIM_STRIDE);
        packPrimitives(buf, [{ kind: "sphere", a: [0, 0, 0], radius: 1 }]);
        // identity quaternion and zero velocity, so an unrotated static primitive behaves
        expect(buf[PRIM_HEADER + 11]).toBe(1); // q.w
        expect(buf.slice(PRIM_HEADER + 12, PRIM_HEADER + 15)).toEqual(new Float32Array([0, 0, 0]));
    });
    it("sizes the buffer for the header plus the primitives", () => {
        expect(primBufferBytes(0)).toBe(PRIM_HEADER * 4);
        expect(primBufferBytes(10)).toBe((PRIM_HEADER + 10 * PRIM_STRIDE) * 4);
    });
});

describe("localizePrimitive", () => {
    // Regression: this shipped re-basing `b` for EVERY kind. For a box `b` is half extents, not a
    // point, so a crate at x=11 with 0.56 half extents became (0.56-11, 0, 0.56-3) = a zero-height,
    // negative-sized box. That box has no interior, so the fluid poured straight through the crate
    // and the debug overlay drew nothing — 1185 particles measured inside the crate before the fix,
    // 0 after.
    const centre: [number, number, number] = [11, 0.56, 3];

    it("leaves a box's half extents alone and re-bases only its centre", () => {
        const box: FluidPrimitive = { kind: "box", a: [11, 0.56, 3], b: [0.56, 0.56, 0.56], rotation: [0, 0, 0, 1] };
        const local = localizePrimitive(box, centre);
        expect(local.a).toEqual([0, 0, 0]);
        expect(local.b).toEqual([0.56, 0.56, 0.56]); // NOT [-10.44, 0, -2.44]
        expect(local.rotation).toEqual([0, 0, 0, 1]);
    });

    it("keeps a localized box solid at its own centre", () => {
        const box: FluidPrimitive = { kind: "box", a: [11, 0.56, 3], b: [0.5, 0.5, 0.5] };
        const local = localizePrimitive(box, centre);
        // Re-posed back to world by adding the rest centre, the box must still contain its centre
        // and exclude a point just outside a face. A degenerate box fails the first assertion.
        const world: FluidPrimitive = { ...local, a: [local.a[0] + centre[0], local.a[1] + centre[1], local.a[2] + centre[2]] };
        expect(primitiveSdf(world, [11, 0.56, 3])).toBeLessThan(0);
        expect(primitiveSdf(world, [11, 0.56 + 0.75, 3])).toBeGreaterThan(0);
    });

    it("re-bases BOTH end points of a capsule, because there `b` is a point", () => {
        const cap: FluidPrimitive = { kind: "capsule", a: [11, 0.2, 3], b: [11, 1.2, 3], radius: 0.3 };
        const local = localizePrimitive(cap, centre);
        expect(local.a).toEqual([0, -0.36000000000000004, 0]);
        expect(local.b).toEqual([0, 0.6399999999999999, 0]);
        expect(local.radius).toBe(0.3);
    });

    it("handles a sphere, which has no b at all", () => {
        const s: FluidPrimitive = { kind: "sphere", a: [11, 0.56, 3], radius: 0.4 };
        const local = localizePrimitive(s, centre);
        expect(local.a).toEqual([0, 0, 0]);
        expect(local.b).toBeUndefined();
    });
});
