import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { exportJsonFromPairState, presetFromExportJson, type FluidExportJson } from "../../../lab/lite/src/demos/fluid/preset-io";
import type { PairState } from "../../../lab/lite/src/demos/fluid/demo";
import { cellSizeForPhysicsScale, gridCellsForSize } from "../../../lab/lite/src/demos/fluid/grid-settings";

describe("fluid preset grid migration", () => {
    it("keeps Waterfall's old 3x world scale explicit without increasing grid allocation", () => {
        const expectedCells = new Map<string, [number, number, number]>([
            ["waterfall.sph.low.json", [125, 63, 125]],
            ["waterfall.sph.middle.json", [167, 84, 167]],
            ["waterfall.sph.high.json", [250, 125, 250]],
            ["waterfall.mlsmpm.low.json", [228, 114, 228]],
            ["waterfall.mlsmpm.middle.json", [455, 228, 455]],
            ["waterfall.mlsmpm.high.json", [607, 304, 607]],
            ["waterfall.pbmpm.liquid.low.json", [228, 114, 228]],
            ["waterfall.pbmpm.liquid.middle.json", [364, 182, 364]],
            ["waterfall.pbmpm.liquid.high.json", [455, 228, 455]],
        ]);

        for (const [file, cells] of expectedCells) {
            const path = resolve(process.cwd(), "lab/public/fluid-presets", file);
            const json = JSON.parse(readFileSync(path, "utf8")) as FluidExportJson;
            const preset = presetFromExportJson(json);

            expect(json.formatVersion, file).toBe(5);
            expect(json.emitters, file).toBeUndefined();
            expect(json.sinks, file).toBeUndefined();
            expect(preset.grid?.position, file).toEqual([0, 30, 0]);
            expect(preset.grid?.size, file).toEqual([120, 60, 120]);
            expect(gridCellsForSize(preset.grid!.size, cellSizeForPhysicsScale(json.meta.method, preset.physScale!)), file).toEqual(cells);
        }
    });

    it("round-trips format-5 world size and grid-local positions", () => {
        const state = {
            schema: {},
            demoParams: {},
            simulationDuration: 12,
            alphaDecay: 2.5,
            emitters: [
                {
                    id: "source",
                    name: "Source",
                    enabled: true,
                    behavior: "inflow",
                    transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                    shape: { type: "box", size: [1, 1, 1] },
                    sampling: "volume",
                    velocity: [0, 0, 0],
                    velocitySpace: "world",
                    spread: 0,
                },
            ],
            sinks: [],
            color: "#000000",
            half: false,
            thicknessDownscale: 1,
            absorption: 1,
            size: 1,
            physScale: 1,
            grid: { position: [10, 20, 30], size: [8.8, 11, 13.2] },
            count: 1,
            showContainer: true,
        } as PairState;

        const exported = exportJsonFromPairState("box", "MLS-MPM", state);
        const imported = presetFromExportJson(exported);

        expect(exported.formatVersion).toBe(5);
        expect(imported.simulationDuration).toBe(12);
        expect(imported.alphaDecay).toBe(2.5);
        expect(imported.grid).toEqual(state.grid);
        expect(imported.emitters?.[0]?.transform.position).toEqual([1, 2, 3]);
    });

    it("leaves gridless legacy presets on the demo-authored default domain", () => {
        const path = resolve(process.cwd(), "lab/public/fluid-presets/box.sph.low.json");
        const json = JSON.parse(readFileSync(path, "utf8")) as FluidExportJson;
        delete json.gridPosition;
        delete json.gridSize;
        delete json.gridCells;
        delete json.domain;
        delete json.gridResolution;

        expect(presetFromExportJson(json).grid).toBeUndefined();
    });

    it("migrates format-4 cell counts to world-space size without translating flow", () => {
        const state = {
            schema: {},
            demoParams: {},
            emitters: [
                {
                    id: "source",
                    name: "Source",
                    enabled: true,
                    behavior: "inflow",
                    transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                    shape: { type: "box", size: [1, 1, 1] },
                    sampling: "volume",
                    velocity: [0, 0, 0],
                    velocitySpace: "world",
                    spread: 0,
                },
            ],
            sinks: [],
            color: "#000000",
            half: false,
            thicknessDownscale: 1,
            absorption: 1,
            size: 1,
            physScale: 1,
            grid: { position: [10, 20, 30], size: [8.8, 11, 13.2] },
            count: 1,
            showContainer: true,
        } as PairState;
        const format4 = exportJsonFromPairState("box", "MLS-MPM", state);
        format4.formatVersion = 4;
        format4.gridCells = [40, 50, 60];
        delete format4.gridSize;
        delete format4.simulationDuration;
        delete format4.alphaDecay;

        const imported = presetFromExportJson(format4);

        expect(imported.grid).toEqual(state.grid);
        expect(imported.simulationDuration).toBe(0);
        expect(imported.alphaDecay).toBe(2);
        expect(imported.emitters?.[0]?.transform.position).toEqual([1, 2, 3]);
    });
});
