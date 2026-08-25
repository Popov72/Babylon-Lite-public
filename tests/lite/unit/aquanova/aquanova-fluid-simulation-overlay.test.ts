import { describe, expect, it } from "vitest";
import { fluidGridWireframeLines, fluidShapeWireframeLines } from "../../../../lab/lite/src/demos/aquanova/debug/fluid-simulation-overlay";

describe("Aquanova fluid simulation debug overlay", () => {
    it("draws the twelve edges of the exact grid extents", () => {
        const lines = fluidGridWireframeLines([10, 4, 6]);
        const points = lines.flat();

        expect(lines).toHaveLength(12);
        expect(Math.min(...points.map(({ x }) => x))).toBe(-5);
        expect(Math.max(...points.map(({ x }) => x))).toBe(5);
        expect(Math.min(...points.map(({ y }) => y))).toBe(-2);
        expect(Math.max(...points.map(({ y }) => y))).toBe(2);
        expect(Math.min(...points.map(({ z }) => z))).toBe(-3);
        expect(Math.max(...points.map(({ z }) => z))).toBe(3);
    });

    it("draws flow-object boxes and spheres in local shape space", () => {
        const box = fluidShapeWireframeLines({ type: "box", size: [2, 4, 6] });
        const sphere = fluidShapeWireframeLines({ type: "sphere", radius: 2 });

        expect(box).toHaveLength(12);
        expect(sphere).toHaveLength(72);
        expect(Math.max(...sphere.flat().map(({ x }) => Math.abs(x)))).toBeCloseTo(2);
    });

    it("draws finite capsule coordinates with the requested total height", () => {
        const capsule = fluidShapeWireframeLines({ type: "capsule", radius: 1, height: 6 });
        const points = capsule.flat();

        expect(capsule).toHaveLength(114);
        expect(points.every(({ x, y, z }) => Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z))).toBe(true);
        expect(Math.min(...points.map(({ y }) => y))).toBeCloseTo(-3);
        expect(Math.max(...points.map(({ y }) => y))).toBeCloseTo(3);
    });
});
