import { describe, expect, it } from "vitest";
import {
    advanceLiquefactorPresentation,
    advanceLiquefactorSwayBlend,
    advanceLiquefactorSwayScale,
    ANTI_GRAVITY_GUN_TRANSFORM,
    computeLiquefactorPose,
    liquefactorPresentationPose,
    liquefactorSwayPose,
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

    it("keeps the calibrated anti-gravity gun transform", () => {
        expect(ANTI_GRAVITY_GUN_TRANSFORM).toEqual({
            position: [-0.2063, -0.7618, -0.0354],
            rotationDegrees: [-0.05, 19.92, 6.41],
            scale: [2.74, 2.74, 2.74],
            localGuidePosition: [0, 0.1707, 0.2889],
            localGuideRotationDegrees: [0, 4.3, 0],
        });
    });

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

    it("animates reversibly between a lowered hidden pose and the horizontal firing pose", () => {
        const raised = liquefactorPresentationPose(1);
        const lowered = liquefactorPresentationPose(0);
        const halfway = advanceLiquefactorPresentation(0, 1, 210);

        expect(lowered.rotationX).toBeGreaterThan(1);
        expect(lowered.positionY).toBeLessThan(-0.5);
        expect(raised.positionY).toBeCloseTo(0);
        expect(raised.rotationX).toBeCloseTo(0);
        expect(halfway).toBeCloseTo(0.5);
        expect(advanceLiquefactorPresentation(halfway, 0, 210)).toBe(0);
    });

    it("adds subtle reversible cosmetic balancing without shifting the aim guide", () => {
        const pose = liquefactorSwayPose(1.25);
        const disabled = liquefactorSwayPose(1.25, 0);

        expect(Math.abs(pose.position[0])).toBeLessThan(0.02);
        expect(Math.abs(pose.position[1])).toBeLessThan(0.011);
        expect(Math.abs(pose.rotation[1])).toBeLessThan(0.019);
        for (const value of [...disabled.position, ...disabled.rotation]) {
            expect(value).toBeCloseTo(0);
        }
        expect(advanceLiquefactorSwayBlend(1, false, 130)).toBeCloseTo(0.5);
        expect(advanceLiquefactorSwayBlend(0, true, 260)).toBe(1);
        expect(advanceLiquefactorSwayBlend(1, true, 0, true)).toBe(0);
        expect(advanceLiquefactorSwayBlend(0, true, 130)).toBeCloseTo(0.5);
    });

    it("scales both sway amplitude and pace for walking and running", () => {
        const idle = liquefactorSwayPose(0.75, 1);
        const walking = liquefactorSwayPose(0.75, 2);
        const running = liquefactorSwayPose(0.75, 4);

        expect(walking.position[0]).toBeCloseTo(idle.position[0] * 2);
        expect(running.rotation[2]).toBeCloseTo(idle.rotation[2] * 4);
        expect(advanceLiquefactorSwayScale(1, 2, 1000)).toBeCloseTo(2, 4);
        expect(advanceLiquefactorSwayScale(2, 4, 1000)).toBeCloseTo(4, 4);
    });
});
