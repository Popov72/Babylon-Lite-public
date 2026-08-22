import { describe, expect, it } from "vitest";

import { carryMethodIndependentState } from "../../../../lab/lite/src/demos/fluid/method-independent-state";
import type { PairState } from "../../../../lab/lite/src/demos/fluid/demo";

const state = (schema: Record<string, number>): PairState => ({
    schema,
    demoParams: {},
    emitters: [],
    sinks: [],
    color: "#16a3c3",
    half: false,
    thicknessDownscale: 1,
    absorption: 1,
    size: 1,
    physScale: 1,
    count: 15_000,
});

describe("method-independent fluid authoring state", () => {
    it("carries shared authoring values while retaining target-method controls", () => {
        const target: PairState = {
            ...state({ gravity: 9.8, stiffness: 80, viscosity: 0.01 }),
            material: 2,
            activeBlocks: true,
            pagedGrid: true,
            pagedGridMaxPages: 2_000,
            fusedBlockDiscovery: true,
        };
        const shared: PairState = {
            ...state({ gravity: 3.25, relaxation: 70, viscosity: 0.2 }),
            emitters: [
                {
                    id: "source",
                    name: "Source",
                    enabled: true,
                    behavior: "inflow",
                    transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                    shape: { type: "sphere", radius: 1 },
                    sampling: "volume",
                    velocity: [0, 1, 0],
                    velocitySpace: "world",
                    spread: 0,
                },
            ],
            physScale: 0.73,
            grid: { position: [2, 4, -3], size: [10, 8, 12] },
            showGridBounds: true,
            count: 7_500,
        };

        const merged = carryMethodIndependentState(target, shared);

        expect(merged.schema).toEqual({ gravity: 3.25, stiffness: 80, viscosity: 0.01 });
        expect(merged.emitters).toEqual(shared.emitters);
        expect(merged.emitters).not.toBe(shared.emitters);
        expect(merged.physScale).toBe(0.73);
        expect(merged.grid).toEqual(shared.grid);
        expect(merged.showGridBounds).toBe(true);
        expect(merged.count).toBe(7_500);
        expect(merged.material).toBe(2);
        expect(merged.activeBlocks).toBe(true);
        expect(merged.pagedGrid).toBe(true);
        expect(merged.pagedGridMaxPages).toBe(2_000);
        expect(merged.fusedBlockDiscovery).toBe(true);
    });

    it("retains material-specific presentation when requested", () => {
        const target: PairState = { ...state({ gravity: 9.8 }), color: "#c2b280", renderMode: "spheres", material: 2 };
        const shared: PairState = { ...state({ gravity: 3.25 }), color: "#16a3c3", renderMode: "surface" };

        const merged = carryMethodIndependentState(target, shared, { retainTargetPresentation: true });

        expect(merged.color).toBe("#c2b280");
        expect(merged.renderMode).toBe("spheres");
        expect(merged.schema.gravity).toBe(3.25);
    });

    it("retains FLIP pressure and transfer controls while carrying shared gravity", () => {
        const target = {
            ...state({ gravity: 9.8, flipRatio: 0.95, pressureIterations: 40, pressureRelaxation: 0.8 }),
            gridResolution: 192,
            markersPerCell: 8,
        };
        const shared = state({ gravity: 2.5, relaxation: 70 });

        const merged = carryMethodIndependentState(target, shared);

        expect(merged.schema).toEqual({ gravity: 2.5, flipRatio: 0.95, pressureIterations: 40, pressureRelaxation: 0.8 });
        expect(merged.gridResolution).toBe(192);
        expect(merged.markersPerCell).toBe(8);
    });
});
