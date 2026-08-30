import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import type { PairState } from "../../../../lab/lite/src/demos/fluid/demo";
import { exportJsonFromPairState, presetFromExportJson, type FluidExportJson } from "../../../../lab/lite/src/demos/fluid/preset-io";

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
    it("makes the Waterfall domain explicit without dropping hidden flow parameters", () => {
        const expected = {
            "waterfall.mlsmpm.high.json": [0.9, 0, 6, 0, 8.5],
            "waterfall.mlsmpm.low.json": [2.4, 0, 6, 0, 8.5],
            "waterfall.mlsmpm.middle.json": [1.2, 0, 6, 0, 8.5],
            "waterfall.pbmpm.liquid.high.json": [1.2, 0, 0.05, 0.5, 16],
            "waterfall.pbmpm.liquid.low.json": [2.4, 0, 6, 0.5, 16],
            "waterfall.pbmpm.liquid.middle.json": [1.5, 0, 6, 0.5, 16],
            "waterfall.sph.high.json": [1.2, 0.6, 3.25, 0.5, 4.5],
            "waterfall.sph.low.json": [2.4, 0.6, 3.25, 0.5, 4.5],
            "waterfall.sph.middle.json": [1.8, 0.6, 6, 0.5, 4.5],
        } as const;

        for (const [filename, [particleSize, sourceSpeed, emitRate, spread, frontBias]] of Object.entries(expected)) {
            const path = resolve(process.cwd(), "lab/public/fluid-presets", filename);
            const json = JSON.parse(readFileSync(path, "utf8")) as FluidExportJson;
            const preset = presetFromExportJson(json);
            const currentFormat = filename.endsWith(".high.json");

            expect(json.formatVersion, filename).toBe(currentFormat ? 13 : 5);
            expect(json.physicsParticleSize, filename).toBe(particleSize);
            expect(json.gridPosition, filename).toEqual([0, 30, 0]);
            expect(json.gridSize, filename).toEqual([120, 60, 120]);
            if (currentFormat) {
                expect(json.emitters, filename).toHaveLength(4);
                expect(json.sinks, filename).toHaveLength(1);
                expect(preset.legacyFlow, filename).toBe(false);
            } else {
                expect(json.demoParams, filename).toMatchObject({ sourceSpeed, emitRate, spread, frontBias });
            }
            expect(preset.physScale, filename).toBe(particleSize);
            expect(preset.grid, filename).toEqual({ position: [0, 30, 0], size: [120, 60, 120] });
        }
    });

    it("round-trips current-format world position and size", () => {
        const state = pairState();
        state.simulationDuration = 12;
        state.alphaDecay = 2.5;
        state.independentRendering = true;
        state.polygonShader = "ocean";
        state.showGridBounds = true;
        state.showGridBoundsSolid = true;
        state.camera = { alpha: 0.25, beta: 1.1, radius: 18, target: [2, 3, 4] };
        state.freeCamera = { position: [5, 6, 7], target: [8, 9, 10] };
        const exported = exportJsonFromPairState("box", "MLS-MPM", state);
        const imported = presetFromExportJson(exported);

        expect(exported.formatVersion).toBe(13);
        expect(exported.gridPosition).toEqual(state.grid?.position);
        expect(exported.gridSize).toEqual(state.grid?.size);
        expect(exported.simulationDuration).toBe(12);
        expect(exported.alphaDecay).toBe(2.5);
        expect(imported.grid).toEqual(state.grid);
        expect(imported.simulationDuration).toBe(12);
        expect(imported.alphaDecay).toBe(2.5);
        expect(exported.render.independentRendering).toBe(true);
        expect(imported.independentRendering).toBe(true);
        expect(exported.showGridBounds).toBe(true);
        expect(exported.showGridBoundsSolid).toBe(true);
        expect(imported.showGridBounds).toBe(true);
        expect(imported.showGridBoundsSolid).toBe(true);
        expect(exported.render.polygonShader).toBe("ocean");
        expect(imported.polygonShader).toBe("ocean");
        expect(exported.camera).toEqual(state.camera);
        expect(imported.camera).toEqual(state.camera);
        expect(exported.freeCamera).toEqual(state.freeCamera);
        expect(imported.freeCamera).toEqual(state.freeCamera);
    });

    it("round-trips FLIP resolution divisions and marker density", () => {
        const state = pairState();
        state.physScale = 0.5;
        state.gridResolution = 80;
        state.markersPerCell = 8;
        const exported = exportJsonFromPairState("box", "FLIP", state);
        const imported = presetFromExportJson(exported);

        expect(exported.physicsParticleSize).toBeUndefined();
        expect(exported.gridResolution).toBe(80);
        expect(exported.markersPerCell).toBe(8);
        expect(imported.gridResolution).toBe(80);
        expect(imported.markersPerCell).toBe(8);
        expect(imported.physScale).toBe(1);
    });

    it("round-trips current FLIP timestep and material controls", () => {
        const state = pairState();
        state.schema = {
            velocityDamping: 0.1,
            kinematicViscosity: 0.25,
            viscosityIterations: 16,
            surfaceTension: 0.4,
            minSubsteps: 2,
            maxSubsteps: 12,
            cflNumber: 1.5,
            maxSubDtMs: 6,
        };

        expect(presetFromExportJson(exportJsonFromPairState("box", "FLIP", state)).schema).toEqual(state.schema);
    });

    it("round-trips advanced FLIP whitewater controls", () => {
        const state = pairState();
        state.foam = {
            enabled: true,
            activeParticles: true,
            generateSpray: false,
            generateFoam: true,
            generateBubbles: false,
            surfaceFiltering: true,
            kTa: 30,
            kWc: 40,
            kTurb: 25,
            energySpeedMin: 0.8,
            energySpeedMax: 7,
            curvatureMin: 0.1,
            curvatureMax: 2,
            turbulenceMin: 0.2,
            turbulenceMax: 3,
            foamLayerDepth: 1.5,
            sprayDrag: 0.6,
            kb: 0.8,
            kd: 0.5,
            tMin: 0.3,
            tMax: 2,
            poolScale: 3,
            blurRadius: 4,
            lightIntensity: 0.9,
            ambient: 0.5,
            aoStrength: 0.5,
            normalStrength: 6,
            debugTexture: "off",
        };

        const imported = presetFromExportJson(exportJsonFromPairState("box", "FLIP", state));
        expect(imported.foam).toMatchObject({
            generateSpray: false,
            generateFoam: true,
            generateBubbles: false,
            surfaceFiltering: true,
            kTurb: 25,
            energySpeedMin: 0.8,
            energySpeedMax: 7,
            curvatureMin: 0.1,
            curvatureMax: 2,
            turbulenceMin: 0.2,
            turbulenceMax: 3,
            foamLayerDepth: 1.5,
            sprayDrag: 0.6,
        });
    });

    it("defaults legacy strict foam filtering by solver", () => {
        const flip = exportJsonFromPairState("box", "FLIP", pairState());
        const pbf = exportJsonFromPairState("box", "PBF", pairState());
        delete flip.foam.surfaceFiltering;
        delete pbf.foam.surfaceFiltering;

        expect(presetFromExportJson(flip).foam?.surfaceFiltering).toBe(true);
        expect(presetFromExportJson(pbf).foam?.surfaceFiltering).toBe(false);
    });

    it("defaults legacy lifecycle settings to indefinite with a two-second decay", () => {
        const legacy = exportJsonFromPairState("box", "PBF", pairState());
        delete legacy.simulationDuration;
        delete legacy.alphaDecay;

        const imported = presetFromExportJson(legacy);
        expect(imported.simulationDuration).toBe(0);
        expect(imported.alphaDecay).toBe(2);
    });

    it("truncates imported gravity and artificial pressure to three decimal places", () => {
        const json = exportJsonFromPairState("box", "PBF", pairState());
        json.physics = {
            gravity: 9.810999,
            scorr: 0.123999,
            viscosity: 0.087654,
        };

        expect(presetFromExportJson(json).schema).toEqual({
            gravity: 9.81,
            scorr: 0.123,
            viscosity: 0.087654,
        });
    });

    it("round-trips config-level initial allocation and structured sink recycle semantics", () => {
        const state = pairState();
        state.initialEmittersFillCapacity = true;
        state.emitters = [
            {
                id: "source",
                name: "source",
                enabled: true,
                behavior: "inflow",
                transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: { type: "box", size: [2, 2, 2] },
                sampling: "volume",
                velocity: [0, -1, 0],
                velocitySpace: "world",
                spread: 0.5,
            },
        ];
        state.sinks = [
            {
                id: "sink",
                name: "sink",
                enabled: true,
                mode: "recycle",
                transform: { position: [4, 5, 6], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: { type: "box", size: [8, 1, 8] },
                targets: ["source"],
                perParticleRecycleRate: 0.7,
            },
        ];

        const exported = exportJsonFromPairState("waterfall", "PB-MPM", state);
        const imported = presetFromExportJson(exported);

        expect(exported.initialEmittersFillCapacity).toBe(true);
        expect(imported.initialEmittersFillCapacity).toBe(true);
        expect(imported.emitters).toEqual(state.emitters);
        expect(imported.sinks).toEqual(state.sinks);
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
