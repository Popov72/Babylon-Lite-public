import { describe, expect, it } from "vitest";
import { mat4Decompose } from "../../../packages/babylon-lite/src/math/mat4-decompose";
import type { Mat4 } from "../../../packages/babylon-lite/src/math/types";

/** Compose a column-major TRS matrix (test-only mirror of the decompose contract). */
function compose(t: [number, number, number], q: [number, number, number, number], s: [number, number, number]): Mat4 {
    const [x, y, z, w] = q;
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;
    const xx = x * x2;
    const xy = x * y2;
    const xz = x * z2;
    const yy = y * y2;
    const yz = y * z2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    // prettier-ignore
    return new Float32Array([
        (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
        (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
        (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
        t[0], t[1], t[2], 1,
    ]) as unknown as Mat4;
}

function maxAbsDiff(a: Mat4, b: Mat4): number {
    let d = 0;
    for (let i = 0; i < 16; i++) {
        d = Math.max(d, Math.abs((a as unknown as Float32Array)[i]! - (b as unknown as Float32Array)[i]!));
    }
    return d;
}

/** Decompose then recompose — the result must reproduce the input matrix. */
function roundTrip(m: Mat4): number {
    const d = mat4Decompose(m);
    const back = compose([d.translation.x, d.translation.y, d.translation.z], [d.rotation.x, d.rotation.y, d.rotation.z, d.rotation.w], [d.scale.x, d.scale.y, d.scale.z]);
    return maxAbsDiff(m, back);
}

const ROT_Y_90: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];

describe("mat4Decompose", () => {
    it("round-trips a plain rotation + positive scale", () => {
        expect(roundTrip(compose([1, 2, 3], ROT_Y_90, [0.65, 0.85, 1]))).toBeLessThan(1e-6);
    });

    it("round-trips a MIRRORED matrix (negative determinant)", () => {
        // A negative scale makes the basis a reflection. Extracting a quaternion from a reflection
        // silently yields a wrong orientation unless the mirror is folded into the scale — this is
        // the regression that left mirrored glTF props (a Blender-mirrored door leaf reparented
        // under a mirrored display root) visibly mis-rotated.
        expect(roundTrip(compose([-4, 0.073, 2], ROT_Y_90, [-0.65, 0.85, 1]))).toBeLessThan(1e-6);
        expect(roundTrip(compose([0, 0, 0], [0.5, -0.5, 0.5, 0.5], [1, 1, -1]))).toBeLessThan(1e-6);
    });

    it("reports the mirror as a negative X scale, leaving a proper rotation", () => {
        const d = mat4Decompose(compose([0, 0, 0], ROT_Y_90, [-0.65, 0.85, 1]));
        expect(d.scale.x).toBeLessThan(0);
        expect(Math.abs(d.scale.x)).toBeCloseTo(0.65, 5);
        expect(d.scale.y).toBeCloseTo(0.85, 5);
        expect(d.scale.z).toBeCloseTo(1, 5);
        // Unit quaternion => the basis it came from is a rotation, not a reflection.
        expect(Math.hypot(d.rotation.x, d.rotation.y, d.rotation.z, d.rotation.w)).toBeCloseTo(1, 5);
    });

    it("leaves non-mirrored input with positive scales", () => {
        const d = mat4Decompose(compose([0, 0, 0], ROT_Y_90, [2, 3, 4]));
        expect(d.scale.x).toBeCloseTo(2, 5);
        expect(d.scale.y).toBeCloseTo(3, 5);
        expect(d.scale.z).toBeCloseTo(4, 5);
    });
});
