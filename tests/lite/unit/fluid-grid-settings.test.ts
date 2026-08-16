import { describe, expect, it } from "vitest";

import {
    cellSizeForGridResolution,
    cellSizeForPhysicsScale,
    gridBounds,
    gridCellsForBounds,
    gridCellsForSize,
    gridPositionForBounds,
    gridResolutionForScale,
    gridSizeForBounds,
    gridWorldSize,
} from "../../../lab/lite/src/demos/fluid/grid-settings";

describe("fluid grid settings", () => {
    it("derives cubic cell size from the explicit physics scale", () => {
        expect(cellSizeForPhysicsScale("PBF", 1)).toBeCloseTo(0.4);
        expect(cellSizeForPhysicsScale("MLS-MPM", 2)).toBeCloseTo(0.44);
        expect(cellSizeForPhysicsScale("PB-MPM", 0.5)).toBeCloseTo(0.11);
    });

    it("derives centered bounds from exact world-space grid size", () => {
        expect(gridWorldSize([4, 6, 8], 0.5)).toEqual([2, 3, 4]);
        expect(gridBounds([10, 20, 30], [2, 3, 4])).toEqual({
            min: [9, 18.5, 28],
            max: [11, 21.5, 32],
        });
    });

    it("migrates historical MPM bounds to a center and per-axis cells", () => {
        const bounds = { min: [-20, -1, -20] as [number, number, number], max: [20, 20, 20] as [number, number, number] };
        expect(gridPositionForBounds(bounds)).toEqual([0, 9.5, 0]);
        expect(gridSizeForBounds(bounds)).toEqual([40, 21, 40]);
        expect(gridCellsForBounds(bounds, 0.22)).toEqual([182, 96, 182]);
        expect(gridCellsForSize([40, 21, 40], 0.22)).toEqual([182, 96, 182]);
    });

    it("retains legacy longest-axis conversion for old preset migration", () => {
        expect(gridResolutionForScale("PBF", 1)).toBe(100);
        expect(cellSizeForGridResolution(200, 120)).toBeCloseTo(0.6);
    });

    it("enforces the solver minimum of four cells per axis during migration", () => {
        expect(gridCellsForBounds({ min: [0, 0, 0], max: [0.1, 0.2, 0.3] }, 1)).toEqual([4, 4, 4]);
    });

    it("does not add a cell when migrated bounds contain floating-point noise", () => {
        const cellSize = 0.22 * 0.6;
        expect(gridCellsForBounds({ min: [-20.13, -1.06, -20.064], max: [20.13, 20.06, 20.064] }, cellSize)).toEqual([305, 160, 304]);
    });
});
