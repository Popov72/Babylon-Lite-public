import { describe, expect, it } from "vitest";

import {
    CURRENT_FLUID_SIMULATION_SEMANTICS,
    fluidCompatibilityProfile,
    pbfScaleAdjustedParam,
    resolveFluidSimulationConfig,
    resolveFluidSimulationSemantics,
} from "../../../../packages/babylon-lite/src/fluid/core/simulation-config";
import { cellSizeForPhysicsScale } from "../../../../packages/babylon-lite/src/fluid/authoring/grid-settings";
import { resolveFlipDiscretization } from "../../../../packages/babylon-lite/src/fluid/solvers/flip-sim";

describe("fluid compatibility profiles", () => {
    it("exposes the divergent host policy constants as named profiles", () => {
        const fluid = fluidCompatibilityProfile("fluid");
        const aquanova = fluidCompatibilityProfile("aquanova");

        expect(fluid.hasGridlessFallback).toBe(true);
        expect(fluid.gridlessFallbackRadiusPerScale).toBeCloseTo(0.09);
        expect(fluid.gridlessFallbackAppliesDomainScale).toBe(true);
        expect(fluid.nonFlipParticleVolumeFloor).toBeCloseTo(1e-6);

        expect(aquanova.hasGridlessFallback).toBe(false);
        expect(aquanova.gridlessFallbackRadiusPerScale).toBeCloseTo(0.08);
        expect(aquanova.gridlessFallbackAppliesDomainScale).toBe(false);
        expect(aquanova.nonFlipParticleVolumeFloor).toBe(0);
    });

    it("reproduces the fluid demo's gridless legacy fallback for non-FLIP methods", () => {
        const config = resolveFluidSimulationConfig("fluid", { method: "MLS-MPM", physicsScale: 2, domainScale: 3, samplingType: "fluid" });
        // radius = 0.09 * scale * domainScale; cellSize = (0.22 * scale) * domainScale.
        expect(config.particleRadius).toBeCloseTo(0.09 * 2 * 3);
        expect(config.cellSize).toBeCloseTo(0.22 * 2 * 3);
        expect(config.particleVolume).toBeCloseTo(Math.max((0.09 * 2 * 3 * 2) ** 3, 1e-6));
        expect(config.gridDim).toBeUndefined();
    });

    it("reproduces Aquanova's shared-discretization path for non-FLIP methods", () => {
        const config = resolveFluidSimulationConfig("aquanova", { method: "PBF", physicsScale: 1, samplingType: "fluid" });
        // radius = 0.08 * scale (FLUID_BASE_PARTICLE_RADIUS); cellSize = max(radius * 4, 0.3).
        expect(config.particleRadius).toBeCloseTo(0.08);
        expect(config.cellSize).toBeCloseTo(0.32);
        // Aquanova applies no volume floor (particle diameter cubed); capacity math floors instead.
        expect(config.particleVolume).toBeCloseTo((0.08 * 2) ** 3);
    });

    it("honors an authored mesh radius identically under either profile", () => {
        for (const profile of ["fluid", "aquanova"] as const) {
            const config = resolveFluidSimulationConfig(profile, { method: "PBF", physicsScale: 1, samplingType: "mesh", particleRadius: 0.15 });
            expect(config.particleRadius).toBeCloseTo(0.15);
            expect(config.cellSize).toBeCloseTo(0.6); // max(0.15 * 4, 0.3)
        }
    });

    it("derives non-FLIP grid dimensions from an explicit world size", () => {
        const config = resolveFluidSimulationConfig("aquanova", { method: "MLS-MPM", physicsScale: 1, explicitGrid: true, gridSize: [8, 4, 8] });
        // dx = fluidSimulationCellSize("MLS-MPM", 0.08 radius) = max(0.08 * 2.4, 0.18) = 0.192.
        expect(config.cellSize).toBeCloseTo(0.192);
        expect(config.gridDim).toEqual([42, 21, 42]);
    });

    it("delegates FLIP to the authoritative discretization resolver", () => {
        const config = resolveFluidSimulationConfig("fluid", {
            method: "FLIP",
            physicsScale: 1,
            bounds: { min: [-20, 0, -20], max: [20, 20, 20] },
            gridResolution: 160,
            markersPerCell: 8,
        });
        expect(config.cellSize).toBeCloseTo(0.25);
        expect(config.particleRadius).toBeCloseTo(0.09);
        expect(config.particleVolume).toBeCloseTo(0.25 ** 3 / 8);
        expect(config.gridDim).toEqual([160, 80, 160]);
        expect(config.flip?.markerVolume).toBeCloseTo(0.25 ** 3 / 8);
    });

    it("requires bounds for FLIP", () => {
        expect(() => resolveFluidSimulationConfig("fluid", { method: "FLIP", physicsScale: 1 })).toThrow(/FLIP requires world-space bounds/);
    });

    it("falls back to the profile physics-scale cell size when FLIP omits gridResolution", () => {
        const bounds = { min: [-5, 0, -5] as [number, number, number], max: [5, 4, 5] as [number, number, number] };
        const scale = 1.5;
        // Aquanova: dx = cellSizeForPhysicsScale("FLIP", scale) (no gridless domain scale). This is
        // exactly what Aquanova production passed as dx when a preset omitted gridResolution.
        const aq = resolveFluidSimulationConfig("aquanova", { method: "FLIP", physicsScale: scale, bounds, markersPerCell: 8 });
        const expectedAq = resolveFlipDiscretization({ boundsMin: bounds.min, boundsMax: bounds.max, dx: cellSizeForPhysicsScale("FLIP", scale), markersPerCell: 8 });
        expect(aq.cellSize).toBeCloseTo(expectedAq.dx);
        expect(aq.particleRadius).toBeCloseTo(expectedAq.particleRadius);
        expect(aq.particleVolume).toBeCloseTo(expectedAq.markerVolume);
        expect(aq.gridDim).toEqual([...expectedAq.gridDim]);

        // Fluid: the gridless domain scale multiplies the fallback cell size.
        const fl = resolveFluidSimulationConfig("fluid", { method: "FLIP", physicsScale: scale, domainScale: 2, bounds, markersPerCell: 8 });
        const expectedFl = resolveFlipDiscretization({ boundsMin: bounds.min, boundsMax: bounds.max, dx: cellSizeForPhysicsScale("FLIP", scale) * 2, markersPerCell: 8 });
        expect(fl.cellSize).toBeCloseTo(expectedFl.dx);

        // An explicit gridResolution still wins over the fallback.
        const withRes = resolveFluidSimulationConfig("aquanova", { method: "FLIP", physicsScale: scale, bounds, gridResolution: 40, markersPerCell: 8 });
        expect(withRes.cellSize).toBeCloseTo(10 / 40); // max extent 10 / gridResolution 40
    });

    it("centralizes PBF particle-scale coupling for rest density and relaxation", () => {
        expect(pbfScaleAdjustedParam("restDensity", 341, 2)).toBeCloseTo(341 / 8);
        expect(pbfScaleAdjustedParam("relaxation", 50, 2)).toBeCloseTo(50 / 4);
        expect(pbfScaleAdjustedParam("restDensity", 341, 1)).toBe(341);
        expect(pbfScaleAdjustedParam("gravity", 9.8, 3)).toBe(9.8);
    });

    it("resolves identical solver-ready PBF options in Fluid, Aquanova authoring/production, and external consumers", () => {
        const preset = {
            physicsScale: 2,
            semantics: CURRENT_FLUID_SIMULATION_SEMANTICS,
            physics: { gravity: 9.81, viscosity: 0.35, restDensity: 341, relaxation: 50 },
        };
        const resolveFor = (profile: "fluid" | "aquanova") =>
            resolveFluidSimulationConfig(profile, {
                method: "PBF",
                physicsScale: preset.physicsScale,
                semantics: preset.semantics,
                physics: preset.physics,
            }).physics;

        const fluid = resolveFor("fluid");
        const aquanovaAuthoring = resolveFor("aquanova");
        const aquanovaProduction = resolveFor("aquanova");
        const minimalExternalConsumer = resolveFor("fluid");

        expect(fluid).toEqual(aquanovaAuthoring);
        expect(fluid).toEqual(aquanovaProduction);
        expect(fluid).toEqual(minimalExternalConsumer);
        expect(fluid).toMatchObject({ gravity: 9.81, viscosity: 0.35, restDensity: 341 / 8, relaxation: 50 / 4 });
    });

    it("infers legacy host PBF semantics and requires explicit semantics for new presets", () => {
        expect(resolveFluidSimulationSemantics({ formatVersion: 13, demo: "aquanova" })).toMatchObject({
            profile: "legacy-aquanova",
            pbfPhysics: "literal",
        });
        expect(resolveFluidSimulationSemantics({ formatVersion: 13, demo: "fluid" })).toMatchObject({
            profile: "legacy-fluid",
            pbfPhysics: "scale-adjusted",
        });
        expect(() => resolveFluidSimulationSemantics({ formatVersion: 14, demo: "fluid" })).toThrow(/require explicit/);
    });
});
