import { describe, expect, it } from "vitest";

import {
    boxProbeNdf,
    intersectsProbeProjection,
    intersectsProbeProjectionBox,
    selectContainingBoxProbe,
    selectContainingProbe,
    selectPoiProbeBlend,
    selectPoiProbeCandidates,
    selectStaticProbe,
    selectStaticBoxProbe,
    sphereProbeNdf,
    type BoxProbeInfluence,
    type BoxProbeRegion,
    type SphereProbeInfluence,
    type SphereProbeRegion,
} from "../../../../lab/lite/src/demos/aquanova/probe-blending";

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

    it("intersects mesh world bounds with yaw-oriented projection boxes", () => {
        const region: BoxProbeRegion = {
            id: "rotated",
            projectionCentre: [0, 0, 0],
            projectionHalfSize: [2, 1, 0.5],
            angleRadians: Math.PI / 4,
        };

        expect(intersectsProbeProjectionBox(region, { centre: [1.7, 0, -1.7], halfSize: [0.3, 0.3, 0.3] })).toBe(true);
        expect(intersectsProbeProjectionBox(region, { centre: [3, 0, 3], halfSize: [0.2, 0.2, 0.2] })).toBe(false);
    });

    it("selects one static probe from mesh bounds with a deterministic nearest fallback", () => {
        const regions: BoxProbeRegion[] = [
            { id: "A", projectionCentre: [0, 0, 0], projectionHalfSize: [1, 1, 1] },
            { id: "B", projectionCentre: [4, 0, 0], projectionHalfSize: [1, 1, 1] },
        ];

        expect(selectStaticBoxProbe(regions, { centre: [0.5, 0, 0], halfSize: [0.25, 0.25, 0.25] })?.id).toBe("A");
        expect(selectStaticBoxProbe(regions, { centre: [2.5, 0, 0], halfSize: [0.6, 0.25, 0.25] })?.id).toBe("B");
        expect(selectStaticBoxProbe(regions, { centre: [20, 0, 0], halfSize: [0.25, 0.25, 0.25] })?.id).toBe("B");
    });

    it("computes the box normalized distance field", () => {
        expect(boxProbeNdf(probes[0]!, [0, 0, 0])).toBe(-1);
        expect(boxProbeNdf(probes[0]!, [2, 0, 0])).toBe(0);
        expect(boxProbeNdf(probes[0]!, [3, 0, 0])).toBe(0.5);
        expect(boxProbeNdf(probes[0]!, [4, 0, 0])).toBe(1);
    });

    it("computes spherical influence and blends it with box probes", () => {
        const sphere: SphereProbeInfluence = {
            id: "sphere",
            shape: "sphere",
            centre: [6, 0, 0],
            innerRadius: 2,
            outerRadius: 4,
        };

        expect(sphereProbeNdf(sphere, [6, 0, 0])).toBe(-1);
        expect(sphereProbeNdf(sphere, [8, 0, 0])).toBe(0);
        expect(sphereProbeNdf(sphere, [9, 0, 0])).toBe(0.5);
        expect(sphereProbeNdf(sphere, [10, 0, 0])).toBe(1);
        const blend = selectPoiProbeBlend([probes[0]!, sphere], [3, 0, 0]);
        expect(blend.map((entry) => entry.id)).toEqual(["A", "sphere"]);
        expect(blend[0]!.weight).toBeCloseTo(0.5);
        expect(blend[1]!.weight).toBeCloseTo(0.5);
    });

    it("selects and intersects spherical projection volumes", () => {
        const sphere: SphereProbeRegion = {
            id: "sphere",
            shape: "sphere",
            projectionCentre: [3, 0, 0],
            projectionRadius: 2,
        };
        const box: BoxProbeRegion = {
            id: "box",
            projectionCentre: [0, 0, 0],
            projectionHalfSize: [1, 1, 1],
        };

        expect(selectContainingProbe([box, sphere], [3.5, 0, 0])?.id).toBe("sphere");
        expect(intersectsProbeProjection(sphere, { centre: [5.25, 0, 0], halfSize: [0.5, 0.5, 0.5] })).toBe(true);
        expect(intersectsProbeProjection(sphere, { centre: [6, 0, 0], halfSize: [0.25, 0.25, 0.25] })).toBe(false);
        expect(selectStaticProbe([box, sphere], { centre: [5.25, 0, 0], halfSize: [0.5, 0.5, 0.5] })?.id).toBe("sphere");
    });

    it("rotates influence and projection boxes by their authored yaw", () => {
        const influence: BoxProbeInfluence = {
            id: "rotated",
            centre: [0, 0, 0],
            innerHalfSize: [0.5, 1, 2],
            outerHalfSize: [1, 2, 3],
            angleRadians: Math.PI / 2,
        };
        const region: BoxProbeRegion = {
            id: "rotated",
            projectionCentre: [0, 0, 0],
            projectionHalfSize: [1, 2, 3],
            angleRadians: Math.PI / 2,
        };

        expect(boxProbeNdf(influence, [2, 0, 0])).toBeLessThan(1);
        expect(selectContainingBoxProbe([region], [2, 0, 0])?.id).toBe("rotated");
        expect(selectContainingBoxProbe([{ ...region, angleRadians: 0 }], [2, 0, 0])).toBeUndefined();
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

    it("ranks a bounded candidate list without calculating shader weights", () => {
        expect(selectPoiProbeCandidates(probes, [3.5, 0, 0], 1)).toEqual([1]);
        expect(selectPoiProbeCandidates(probes, [3.5, 0, 0], 2)).toEqual([1, 0]);
        expect(selectPoiProbeCandidates(probes, [3.5, 0, 0], 0)).toEqual([]);
    });
});
