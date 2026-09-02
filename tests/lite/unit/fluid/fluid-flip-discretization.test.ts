import { describe, expect, it } from "vitest";

import { resolveFlipDiscretization } from "../../../../packages/babylon-lite/src/fluid/solvers/flip-sim.js";

describe("FLIP discretization", () => {
    it("derives the cell width, dimensions, marker volume, and marker radius from one resolution", () => {
        const resolved = resolveFlipDiscretization({
            boundsMin: [-3, -2, -7],
            boundsMax: [3, 2, 7],
            gridResolution: 178,
            markersPerCell: 8,
        });

        expect(resolved.dx).toBeCloseTo(14 / 178, 12);
        expect(resolved.gridDim).toEqual([77, 51, 178]);
        expect(resolved.gridResolution).toBe(178);
        expect(resolved.markerVolume).toBeCloseTo((14 / 178) ** 3 / 8, 12);
        expect(resolved.particleRadius).toBeCloseTo((14 / 178) * (0.09 / 0.25), 12);
    });

    it("rejects conflicting resolution inputs", () => {
        expect(() => resolveFlipDiscretization({ gridResolution: 160, dx: 0.25 })).toThrow(/mutually exclusive/);
    });
});
