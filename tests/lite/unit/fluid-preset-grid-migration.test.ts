import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import type { PairState } from "../../../lab/lite/src/demos/fluid/demo";
import { exportJsonFromPairState, presetFromExportJson, type FluidExportJson } from "../../../lab/lite/src/demos/fluid/preset-io";

const pairState = (): PairState => ({
    schema: {},
    demoParams: {},
    color: "#000000",
    half: false,
    thicknessDownscale: 1,
    absorption: 1,
    size: 1,
    physScale: 1,
    grid: { position: [10, 20, 30], size: [8.8, 11, 13.2] },
    count: 1,
    showContainer: true,
});

describe("fluid preset grid migration", () => {
    it("keeps the known-good Waterfall on its gridless hidden domain scale and 0.4 preset", () => {
        const path = resolve(process.cwd(), "lab/public/fluid-presets/waterfall.pbmpm.liquid.high.json");
        const json = JSON.parse(readFileSync(path, "utf8")) as FluidExportJson;
        const preset = presetFromExportJson(json);

        expect(json.physicsParticleSize).toBe(0.4);
        expect(preset.physScale).toBe(0.4);
        expect(preset.grid).toBeUndefined();
    });

    it("round-trips format-5 world position and size", () => {
        const state = pairState();
        const exported = exportJsonFromPairState("box", "MLS-MPM", state);
        const imported = presetFromExportJson(exported);

        expect(exported.formatVersion).toBe(5);
        expect(exported.gridPosition).toEqual(state.grid?.position);
        expect(exported.gridSize).toEqual(state.grid?.size);
        expect(imported.grid).toEqual(state.grid);
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

    it("migrates format-4 cell counts to world-space size", () => {
        const format4 = exportJsonFromPairState("box", "MLS-MPM", pairState());
        format4.formatVersion = 4;
        format4.gridCells = [40, 50, 60];
        delete format4.gridSize;

        expect(presetFromExportJson(format4).grid).toEqual({
            position: [10, 20, 30],
            size: [8.8, 11, 13.2],
        });
    });

    it("migrates legacy domain and gridResolution markers to an explicit world-space grid", () => {
        const legacy = exportJsonFromPairState("box", "PBF", pairState());
        legacy.formatVersion = 2;
        legacy.domain = { min: [-4, 1, -6], max: [8, 9, 10] };
        legacy.gridResolution = 120;
        delete legacy.gridPosition;
        delete legacy.gridSize;

        expect(presetFromExportJson(legacy).grid).toEqual({
            position: [2, 5, 2],
            size: [12, 8, 16],
        });
    });

    it("makes the historical Marble Tower domain multiplier explicit when a legacy grid is present", () => {
        const legacy = exportJsonFromPairState("marbleTower", "MLS-MPM", pairState());
        legacy.formatVersion = 2;
        legacy.physicsParticleSize = 0.5;
        legacy.demoParams.meshScale = 4;
        legacy.domain = { min: [-80, -4, -80], max: [80, 80, 80] };
        delete legacy.gridPosition;
        delete legacy.gridSize;

        const imported = presetFromExportJson(legacy);
        expect(imported.physScale).toBe(2);
        expect(imported.grid).toEqual({ position: [0, 38, 0], size: [160, 84, 160] });
    });
});
