import { describe, expect, it } from "vitest";
import type { EngineContext } from "../../../../packages/babylon-lite/src/engine/engine";
import { createFlipReferenceSimulation } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/solver";
import { flipReferenceWgsl } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/shaders";
import { FLIP_REFERENCE_ADVECTION_WGSL } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/advection";

describe("Reference advection budget", () => {
    it("rejects budgets whose endpoint counter cannot be represented exactly", () => {
        expect(() =>
            createFlipReferenceSimulation({} as EngineContext, {
                gridOrigin: [0, 0, 0],
                gridDimensions: [2, 2, 2],
                cellSize: 1,
                initialPositions: new Float32Array(),
                initialVelocities: new Float32Array(),
                maxAdvectionSubsteps: 16777216,
            })
        ).toThrow("maxAdvectionSubsteps must not exceed 16777215");
    });

    it("shares the same bounded advection implementation between both execution modes", () => {
        expect(flipReferenceWgsl(false)).toContain(FLIP_REFERENCE_ADVECTION_WGSL);
        expect(flipReferenceWgsl(true)).toContain(FLIP_REFERENCE_ADVECTION_WGSL);
        expect(FLIP_REFERENCE_ADVECTION_WGSL).toContain("samplesLeft -= sweep.samples");
        expect(FLIP_REFERENCE_ADVECTION_WGSL).toContain("attempt < params.switches.z");
        expect(FLIP_REFERENCE_ADVECTION_WGSL).not.toContain("atomicStore(&lists[statusIndex(6u)]");
    });
});
