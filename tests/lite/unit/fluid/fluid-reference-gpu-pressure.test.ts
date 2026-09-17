import { describe, expect, it } from "vitest";
import { flipReferenceWgsl } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/shaders";
import {
    FLIP_REFERENCE_GPU_PRESSURE_BINDINGS,
    FLIP_REFERENCE_GPU_PRESSURE_WGSL,
    flipReferenceGpuPressureBytes,
    flipReferenceGpuPressureWorkgroupSize,
} from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-pressure";

describe("cached Reference pressure", () => {
    it("enables paired pressure only for the pressure runtime, not the oracle or whitewater helper shader", () => {
        expect(flipReferenceWgsl(false)).not.toContain("@binding(21)");
        expect(flipReferenceWgsl(true)).not.toContain("@binding(21)");
        expect(flipReferenceWgsl(true, true)).toContain("@binding(21)");
        expect(flipReferenceWgsl(true, true)).toContain("let difference = pressurePairAdd(pr, -pl)");
    });

    it("uses the same dense 64-byte row allocation for previews and execution", () => {
        expect(flipReferenceGpuPressureBytes(1)).toBe(64);
        expect(flipReferenceGpuPressureBytes(5400)).toBe(345600);
        expect(flipReferenceGpuPressureBytes(40800)).toBe(2611200);
    });

    it("declares the preparation, residual initialization and solve bindings", () => {
        expect(FLIP_REFERENCE_GPU_PRESSURE_BINDINGS).toEqual({
            prepareGpuPressure: [0, 5, 8, 11, 19, 21],
            initializeGpuPressure: [0, 8, 11, 19, 21],
            solveGpuPressure: [0, 5, 6, 8, 9, 11, 19, 21],
            initializeParallelGpuPressure: [0, 5, 6, 8, 9, 11, 19, 21],
            resumeGpuPressure: [0, 5, 6, 8, 9, 11, 19, 21],
            project: [0, 4, 5, 8, 21],
            trueResidual: [0, 5, 8, 21],
        });
    });

    it("keeps wider reductions within invocation, X-size and shared-memory limits", () => {
        const limits = { maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupSizeX: 1024, maxComputeWorkgroupStorageSize: 32768 };
        expect(flipReferenceGpuPressureWorkgroupSize(limits)).toBe(1024);
        expect(flipReferenceGpuPressureWorkgroupSize({ ...limits, maxComputeWorkgroupStorageSize: 16384 })).toBe(512);
        expect(flipReferenceGpuPressureWorkgroupSize({ ...limits, maxComputeWorkgroupSizeX: 256 })).toBe(256);
        expect(flipReferenceGpuPressureWorkgroupSize({ ...limits, maxComputeInvocationsPerWorkgroup: 256 })).toBe(256);
        expect(flipReferenceGpuPressureWorkgroupSize({ ...limits, maxComputeWorkgroupStorageSize: 8192 + 8 })).toBe(256);
        expect(flipReferenceGpuPressureWorkgroupSize({ ...limits, maxComputeWorkgroupStorageSize: 8192 + 12 })).toBe(256);
        expect(flipReferenceGpuPressureWorkgroupSize({ ...limits, maxComputeWorkgroupStorageSize: 8192 + 44 })).toBe(256);
        expect(flipReferenceGpuPressureWorkgroupSize({ ...limits, maxComputeWorkgroupStorageSize: 8192 + 48 })).toBe(512);
    });

    it("retains RHS-relative acceptance through the original independent operator", () => {
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("max(params.tolerances.y, params.tolerances.x * rhsSquared)");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("pressureResidualValue(index)");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("residualSquared > rhsSquared");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("pcg[gid.x].y = residual");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("pcg[gid.x].z = preconditioned");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("activeCount <= arrayLength(&particleScratch)");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("workgroupUniformLoad(&pressureState[2])");
        expect(FLIP_REFERENCE_GPU_PRESSURE_WGSL).toContain("phase == 2u && invalidInitial");
    });
});
