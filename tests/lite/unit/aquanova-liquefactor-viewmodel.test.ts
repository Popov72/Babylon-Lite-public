import { describe, expect, it } from "vitest";
import {
    computeLiquefactorPose,
    LIQUEFACTOR_ADJUSTMENT_MUZZLE,
    LIQUEFACTOR_BARREL_DIRECTION,
    LIQUEFACTOR_MODEL_SCALE,
} from "../../../lab/lite/src/demos/aquanova/liquefactor-viewmodel";

function rotate(q: readonly [number, number, number, number], v: readonly [number, number, number]): [number, number, number] {
    const [x, y, z, w] = q;
    const tx = 2 * (y * v[2] - z * v[1]);
    const ty = 2 * (z * v[0] - x * v[2]);
    const tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

describe("Aquanova Liquefactor viewmodel", () => {
    const cameraWorld = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

    it("keeps its screen anchor stable across aspect-ratio changes", () => {
        const narrowProjection = new Float32Array([2.4, 0, 0, 0, 0, 2.4, 0, 0, 0, 0, 1, 1, 0, 0, -0.1, 0]);
        const wideProjection = new Float32Array([1.2, 0, 0, 0, 0, 2.4, 0, 0, 0, 0, 1, 1, 0, 0, -0.1, 0]);
        const narrow = computeLiquefactorPose(cameraWorld, narrowProjection);
        const wide = computeLiquefactorPose(cameraWorld, wideProjection);
        const narrowOffset = rotate(narrow.rotation, [LIQUEFACTOR_ADJUSTMENT_MUZZLE.x, LIQUEFACTOR_ADJUSTMENT_MUZZLE.y, LIQUEFACTOR_ADJUSTMENT_MUZZLE.z]);
        const wideOffset = rotate(wide.rotation, [LIQUEFACTOR_ADJUSTMENT_MUZZLE.x, LIQUEFACTOR_ADJUSTMENT_MUZZLE.y, LIQUEFACTOR_ADJUSTMENT_MUZZLE.z]);
        const narrowMuzzle = narrow.position.map((value, index) => value + narrowOffset[index]! * LIQUEFACTOR_MODEL_SCALE);
        const wideMuzzle = wide.position.map((value, index) => value + wideOffset[index]! * LIQUEFACTOR_MODEL_SCALE);

        expect((narrowMuzzle[0]! * narrowProjection[0]!) / narrowMuzzle[2]!).toBeCloseTo((wideMuzzle[0]! * wideProjection[0]!) / wideMuzzle[2]!, 6);
        expect((narrowMuzzle[1]! * narrowProjection[5]!) / narrowMuzzle[2]!).toBeCloseTo((wideMuzzle[1]! * wideProjection[5]!) / wideMuzzle[2]!, 6);
    });

    it("aims the authored barrel ray from the muzzle at screen centre", () => {
        const projection = new Float32Array([1.5, 0, 0, 0, 0, 2.4, 0, 0, 0, 0, 1, 1, 0, 0, -0.1, 0]);
        const pose = computeLiquefactorPose(cameraWorld, projection);
        const barrel = rotate(pose.rotation, LIQUEFACTOR_BARREL_DIRECTION);
        const muzzleOffset = rotate(pose.rotation, [LIQUEFACTOR_ADJUSTMENT_MUZZLE.x, LIQUEFACTOR_ADJUSTMENT_MUZZLE.y, LIQUEFACTOR_ADJUSTMENT_MUZZLE.z]);
        const tx = -pose.position[0] - muzzleOffset[0] * LIQUEFACTOR_MODEL_SCALE;
        const ty = -pose.position[1] - muzzleOffset[1] * LIQUEFACTOR_MODEL_SCALE;
        const tz = 2.5 - pose.position[2] - muzzleOffset[2] * LIQUEFACTOR_MODEL_SCALE;
        const inv = 1 / Math.hypot(tx, ty, tz);

        expect(barrel[0]).toBeCloseTo(tx * inv, 5);
        expect(barrel[1]).toBeCloseTo(ty * inv, 5);
        expect(barrel[2]).toBeCloseTo(tz * inv, 5);
    });
});
