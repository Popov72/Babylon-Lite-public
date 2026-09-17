import { describe, expect, it } from "vitest";
import {
    FLIP_REFERENCE_GPU_PRESSURE_BINDINGS,
    FLIP_REFERENCE_GPU_PRESSURE_WGSL,
    flipReferenceGpuPressureBytes,
} from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-pressure";

describe("cached Reference pressure", () => {
    it("uses the same dense 64-byte row allocation for previews and execution", () => {
        expect(flipReferenceGpuPressureBytes(1)).toBe(64);
        expect(flipReferenceGpuPressureBytes(5400)).toBe(345600);
        expect(flipReferenceGpuPressureBytes(40800)).toBe(2611200);
    });

    it("declares the preparation, residual initialization and solve bindings", () => {
        expect(FLIP_REFERENCE_GPU_PRESSURE_BINDINGS).toEqual({
            prepareGpuPressure: [0, 5, 8, 11, 19],
            initializeGpuPressure: [0, 8, 11, 19],
            solveGpuPressure: [0, 5, 6, 8, 11, 19],
        });
    });

    it("retains RHS-relative acceptance through the original independent operator", () => {
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("max(params.tolerances.y, params.tolerances.x * rhsSquared)");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("rowRhs(index) - matrixValue(index, false)");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("residualSquared > rhsSquared");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("pcg[gid.x].y = residual");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("pcg[gid.x].z = preconditioned");
    });
});
