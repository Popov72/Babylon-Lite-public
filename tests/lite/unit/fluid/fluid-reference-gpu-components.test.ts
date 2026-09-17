import { describe, expect, it } from "vitest";
import {
    FLIP_REFERENCE_GPU_COMPONENT_BINDINGS,
    FLIP_REFERENCE_GPU_COMPONENT_WGSL,
    flipReferenceGpuComponentBytes,
} from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-components";

describe("FLIP Reference GPU components", () => {
    it("allocates two parents, two counts, two flags, and two float accumulators per cell", () => {
        expect(flipReferenceGpuComponentBytes(0)).toBe(0);
        expect(flipReferenceGpuComponentBytes(17)).toBe(17 * 32);
    });

    it("exposes only the component stage bindings", () => {
        expect(FLIP_REFERENCE_GPU_COMPONENT_BINDINGS).toEqual({
            initializeGpuComponents: [0, 5, 6, 11, 12],
            linkGpuComponents: [0, 5, 6, 11, 12],
            aggregateGpuComponents: [5, 6, 11, 12],
            markGpuClosedPockets: [5, 6, 11, 12],
            sumGpuSealedFlux: [5, 6, 11, 12],
            fixGpuPressureGauges: [0, 5, 6, 11, 12],
        });
    });

    it("keeps conditioning and gauge connectivity distinct and accumulates only sealed flux", () => {
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("coefficient >= 1.0e-6");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("coefficient > 0.0");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("count > 1u");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("cells[index].geometry.y");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("cells[index].negative.w");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("if (atomicLoad(&gpuComponents[root].gaugeAir) != 0u) { return; }");
    });

    it("uses monotone weak-CAS unions, finite float-CAS sums, and explicit incompatibility failure", () => {
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("let higher = max(aRoot, bRoot)");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("let lower = min(aRoot, bRoot)");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL.match(/atomicCompareExchangeWeak/g)).toHaveLength(3);
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("runtimeFailure(16u)");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("runtimeFailure(32u)");
        expect(FLIP_REFERENCE_GPU_COMPONENT_WGSL).toContain("cells[index].positive.w = 1.0");
    });
});
