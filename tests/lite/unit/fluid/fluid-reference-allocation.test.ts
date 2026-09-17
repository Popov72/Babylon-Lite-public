import { describe, expect, it } from "vitest";
import { resolveFluidSimulationConfig } from "../../../../packages/babylon-lite/src/fluid/core/simulation-config";
import { planFlipReferenceFluidAllocation } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/facade-adapter";
import type { FluidSimulationOptions } from "../../../../packages/babylon-lite/src/fluid/core/fluid-facade";
import { FLIP_REFERENCE_MAX_FRAME_SUBSTEPS, flipReferenceStorageSizes } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/allocation";
import { planFlipReferenceWhitewater } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/whitewater";

const options: FluidSimulationOptions = {
    method: "FLIP",
    particleCount: 128,
    physicsScale: 1,
    gridResolution: 20,
    explicitGrid: true,
    bounds: { min: [0, 0, 0], max: [2, 2, 2] },
};

describe("Reference UI allocation projection", () => {
    it("includes the optional whitewater working, publication and telemetry resources exactly once", () => {
        const configured = { ...options, initialPositions: new Float32Array(64 * 3), foam: { poolScale: 3 } };
        const config = resolveFluidSimulationConfig("fluid", configured);
        const without = planFlipReferenceFluidAllocation({ ...configured, foam: null }, config);
        const withFoam = planFlipReferenceFluidAllocation(configured, config);
        const whitewater = planFlipReferenceWhitewater(options.particleCount, without.dimensions.cellCount!, configured.foam);
        expect(withFoam.foamCapacity).toBe(whitewater.capacity);
        expect(withFoam.foamCapacity).toBeGreaterThan(0);
        expect(withFoam.steadyBytes - without.steadyBytes).toBe(whitewater.resources.reduce((sum, resource) => sum + resource.bytes, 0));
        expect(withFoam.errors).toEqual([]);
        expect(without.foamCapacity).toBe(0);
    });

    it("reserves the full cleanup histogram so live substep-budget changes do not reallocate", () => {
        const withoutHistogram = flipReferenceStorageSizes(8000, 25200, 9261, 64, true, 1);
        const withHistogram = flipReferenceStorageSizes(8000, 25200, 9261, 64, true, FLIP_REFERENCE_MAX_FRAME_SUBSTEPS);
        expect(withHistogram[9]! - withoutHistogram[9]!).toBe((FLIP_REFERENCE_MAX_FRAME_SUBSTEPS - 1) * 4);
        expect(withHistogram.slice(0, 9)).toEqual(withoutHistogram.slice(0, 9));
    });

    it("matches the actual exact-seed working and publication allocations", () => {
        const config = resolveFluidSimulationConfig("fluid", options);
        const plan = planFlipReferenceFluidAllocation({ ...options, initialPositions: new Float32Array(64 * 3) }, config);
        expect(plan.steadyBytes).toBe(3725948);
        expect(plan.resources.reduce((sum, resource) => sum + resource.bytes, 0)).toBe(plan.steadyBytes);
        expect(plan.foamCapacity).toBe(0);
        expect(plan.polygonTriangleCapacity).toBe(0);
    });

    it("uses a conservative capacity bound without exact seeds and reports per-buffer limits", () => {
        const config = resolveFluidSimulationConfig("fluid", options);
        const plan = planFlipReferenceFluidAllocation(options, config, { maxBufferSize: 1024, maxStorageBufferBindingSize: 1024 });
        expect(plan.steadyBytes).toBeGreaterThan(3725948);
        expect(plan.errors.length).toBeGreaterThan(0);
    });
});
