import { describe, expect, it } from "vitest";

import { boxProbeNdf, selectContainingBoxProbe, selectPoiProbeBlend, type BoxProbeInfluence, type BoxProbeRegion } from "../../../lab/lite/src/demos/aquanova/probe-blending";

const probes: BoxProbeInfluence[] = [
    { id: "A", centre: [0, 0, 0], innerHalfSize: [2, 2, 2], outerHalfSize: [4, 4, 4] },
    { id: "B", centre: [6, 0, 0], innerHalfSize: [2, 2, 2], outerHalfSize: [4, 4, 4] },
];

describe("Aquanova POI probe blending", () => {
    it("selects the containing parallax box independently of influence weights", () => {
        const regions: BoxProbeRegion[] = [
            { id: "CH00", projectionCentre: [0, 0, 0], projectionHalfSize: [1, 1, 1] },
            { id: "CH01", projectionCentre: [3, 0, 0], projectionHalfSize: [1, 1, 1] },
        ];

        expect(selectContainingBoxProbe(regions, [3.5, 0, 0])?.id).toBe("CH01");
        expect(selectContainingBoxProbe(regions, [1.5, 0, 0])).toBeUndefined();
    });

    it("uses the tighter box when authored parallax boxes overlap", () => {
        const regions: BoxProbeRegion[] = [
            { id: "large", projectionCentre: [0, 0, 0], projectionHalfSize: [4, 4, 4] },
            { id: "small", projectionCentre: [0, 0, 0], projectionHalfSize: [2, 2, 2] },
        ];

        expect(selectContainingBoxProbe(regions, [1, 0, 0])?.id).toBe("small");
    });

    it("computes the box normalized distance field", () => {
        expect(boxProbeNdf(probes[0]!, [0, 0, 0])).toBe(-1);
        expect(boxProbeNdf(probes[0]!, [2, 0, 0])).toBe(0);
        expect(boxProbeNdf(probes[0]!, [3, 0, 0])).toBe(0.5);
        expect(boxProbeNdf(probes[0]!, [4, 0, 0])).toBe(1);
    });

    it("gives full weight to a probe inside its inner range", () => {
        expect(selectPoiProbeBlend(probes, [1, 0, 0])).toEqual([{ id: "A", weight: 1, ndf: -0.5 }]);
    });

    it("smoothly blends overlapping outer ranges", () => {
        const blend = selectPoiProbeBlend(probes, [3, 0, 0]);
        expect(blend.map((entry) => entry.id)).toEqual(["A", "B"]);
        expect(blend[0]!.weight).toBeCloseTo(0.5);
        expect(blend[1]!.weight).toBeCloseTo(0.5);
    });

    it("keeps texture order stable when the dominant probe changes", () => {
        const left = selectPoiProbeBlend(probes, [2.5, 0, 0]);
        const right = selectPoiProbeBlend(probes, [3.5, 0, 0]);
        expect(left.map((entry) => entry.id)).toEqual(["A", "B"]);
        expect(right.map((entry) => entry.id)).toEqual(["A", "B"]);
        expect(left[0]!.weight).toBeGreaterThan(left[1]!.weight);
        expect(right[1]!.weight).toBeGreaterThan(right[0]!.weight);
    });

    it("falls back to the closest influence when the POI is outside every outer box", () => {
        expect(selectPoiProbeBlend(probes, [20, 0, 0])).toEqual([{ id: "B", weight: 1, ndf: 6 }]);
    });
});
