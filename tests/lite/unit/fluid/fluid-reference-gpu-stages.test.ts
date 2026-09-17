import { describe, expect, it } from "vitest";
import { FLIP_REFERENCE_GPU_STAGES_BINDINGS, FLIP_REFERENCE_GPU_STAGES_WGSL } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-stages";

describe("FLIP Reference fused GPU stages", () => {
    it("publishes exact reachable bindings", () => {
        expect(FLIP_REFERENCE_GPU_STAGES_BINDINGS).toEqual({
            clearGpuParticleStage: [0, 6, 10],
            conditionGpuMatrix: [0, 4, 5, 11],
        });
    });

    it("does not retain the register-sensitive transfer fusion", () => {
        expect(FLIP_REFERENCE_GPU_STAGES_WGSL).not.toContain("transferGpuParticles");
    });

    it("recomputes only cached-volume RHS values with virtual conditioned solid velocity", () => {
        expect(FLIP_REFERENCE_GPU_STAGES_WGSL).toContain("atomicLoad(&runtime[9]) == 0u");
        expect(FLIP_REFERENCE_GPU_STAGES_WGSL).toContain("let volume = cells[gid.x].geometry.x");
        expect(FLIP_REFERENCE_GPU_STAGES_WGSL).toContain("let area = faces[index].area");
        expect(FLIP_REFERENCE_GPU_STAGES_WGSL).toContain("let velocity = faces[index].velocity");
        expect(FLIP_REFERENCE_GPU_STAGES_WGSL).toContain("faces[index].solid");
        expect(FLIP_REFERENCE_GPU_STAGES_WGSL).toContain("cells[gid.x].state.z = rhs");
        expect(FLIP_REFERENCE_GPU_STAGES_WGSL).not.toContain("let face = faces[index]");
    });
});
