import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { gridCellsForSize } from "../../../../packages/babylon-lite/src/fluid/authoring/grid-settings";
import { presetFromExportJson, type FluidExportJson } from "../../../../packages/babylon-lite/src/fluid/authoring/preset-io";
import { fluidSimulationCellSize, fluidSimulationParticleCapacity, fluidSimulationParticleRadius } from "../../../../packages/babylon-lite/src/fluid/core/simulation-config";

describe("Aquanova fluid preset discretization", () => {
    it("reads capsule-tank2 discretization from its JSON", () => {
        const path = resolve(process.cwd(), "lab/public/aquanova/fluidSim/capsule-tank2.json");
        const json = JSON.parse(readFileSync(path, "utf8")) as FluidExportJson;
        const preset = presetFromExportJson(json);
        const settings = {
            physicsParticleSize: preset.physScale!,
            samplingType: preset.demoState?.simulationType === "fluid" ? ("fluid" as const) : ("mesh" as const),
            particleRadius: preset.demoParams?.particleRadius,
        };
        const cellSize = fluidSimulationCellSize("PB-MPM", settings);
        const cells = gridCellsForSize(preset.grid!.size, cellSize);

        expect(json.gridSize).toEqual([6, 4, 14]);
        expect("cellSize" in json).toBe(false);
        expect(cellSize).toBe(0.18);
        expect(fluidSimulationParticleRadius(settings)).toBeCloseTo(0.008);
        expect(fluidSimulationParticleCapacity("PB-MPM", preset.count!, { emitters: preset.emitters!, sinks: preset.sinks! }, (0.008 * 2) ** 3)).toBe(10_000);
        expect(cells).toEqual([34, 23, 78]);
        expect(cells[0] * cells[1] * cells[2] * 16).toBe(975_936);
    });

    it("uses the authored radius when the preset samples a mesh", () => {
        expect(fluidSimulationCellSize("PB-MPM", { physicsParticleSize: 0.1, samplingType: "mesh", particleRadius: 0.12 })).toBeCloseTo(0.288);
    });
});
