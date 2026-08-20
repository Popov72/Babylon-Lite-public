import { describe, expect, it } from "vitest";

import {
    cellSizeForGridResolution,
    cellSizeForPhysicsScale,
    FLIP_DEFAULT_MARKERS_PER_CELL,
    FLIP_HIGH_MARKERS_PER_CELL,
    flipMarkersPerAuthoredCell,
    flipParticleCountForVolume,
    GRID_RESOLUTION_MAX,
    gridBounds,
    gridCellsForBounds,
    gridCellsForSize,
    gridPositionForBounds,
    gridResolutionForScale,
    gridSizeForBounds,
    gridWorldSize,
    highestFittingGridResolution,
} from "../../../lab/lite/src/demos/fluid/grid-settings";

describe("fluid grid settings", () => {
    it("derives cubic cell size from the explicit physics scale", () => {
        expect(cellSizeForPhysicsScale("PBF", 1)).toBeCloseTo(0.4);
        expect(cellSizeForPhysicsScale("FLIP", 1)).toBeCloseTo(0.25);
        expect(cellSizeForPhysicsScale("MLS-MPM", 2)).toBeCloseTo(0.44);
        expect(cellSizeForPhysicsScale("PB-MPM", 0.5)).toBeCloseTo(0.11);
    });

    it("finds the highest resolution accepted by a monotonic device limit", () => {
        expect(highestFittingGridResolution(160, 16, (resolution) => resolution <= 93)).toBe(93);
        expect(highestFittingGridResolution(160, 16, (resolution) => resolution <= 15)).toBeUndefined();
    });

    it("fits FLIP resolution to a particle-buffer capacity", () => {
        const fluidVolume = 6_912;
        const longestSide = 40;
        const capacity = 700_000;
        const fitted = highestFittingGridResolution(200, 16, (resolution) => {
            const cellSize = longestSide / resolution;
            return flipParticleCountForVolume(fluidVolume, cellSize, 8) <= capacity;
        });

        expect(fitted).toBe(93);
        expect(flipParticleCountForVolume(fluidVolume, longestSide / fitted!, 8)).toBe(694_965);
        expect(flipParticleCountForVolume(fluidVolume, longestSide / (fitted! + 1), 8)).toBe(717_625);
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
        expect(GRID_RESOLUTION_MAX).toBe(400);
        expect(gridResolutionForScale("PBF", 1)).toBe(100);
        expect(gridResolutionForScale("FLIP", 1)).toBe(160);
        expect(cellSizeForGridResolution(200, 120)).toBeCloseTo(0.6);
    });

    it("estimates high FLIP marker density per authored MAC cell", () => {
        expect(flipMarkersPerAuthoredCell(80_000, 0.25, 96)).toBeCloseTo(13.0208, 3);
        expect(flipMarkersPerAuthoredCell(120_000, 0.25, 1.5)).toBeGreaterThan(FLIP_HIGH_MARKERS_PER_CELL);
        expect(flipMarkersPerAuthoredCell(0, 0.25, 96)).toBe(0);
    });

    it("derives FLIP marker count from authored volume and cell resolution", () => {
        expect(FLIP_DEFAULT_MARKERS_PER_CELL).toBe(8);
        expect(flipParticleCountForVolume(96, 0.25)).toBe(49_152);
        expect(flipParticleCountForVolume(96, 0.5, 16)).toBe(12_288);
        expect(flipParticleCountForVolume(0, 0.25)).toBe(0);
    });

    it("enforces the solver minimum of four cells per axis during migration", () => {
        expect(gridCellsForBounds({ min: [0, 0, 0], max: [0.1, 0.2, 0.3] }, 1)).toEqual([4, 4, 4]);
    });

    it("does not add a cell when migrated bounds contain floating-point noise", () => {
        const cellSize = 0.22 * 0.6;
        expect(gridCellsForBounds({ min: [-20.13, -1.06, -20.064], max: [20.13, 20.06, 20.064] }, cellSize)).toEqual([305, 160, 304]);
    });
});
