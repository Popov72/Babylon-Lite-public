import { expect, test } from "@playwright/test";
import { flipReferenceWgsl, FLIP_REFERENCE_BINDINGS } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/shaders";
import { FLIP_REFERENCE_GPU_CONTROL_WGSL } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-control";
import { FLIP_REFERENCE_GPU_PRESSURE_BINDINGS, FLIP_REFERENCE_GPU_PRESSURE_WGSL } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-pressure";
import {
    FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_BINDINGS,
    FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_WGSL,
} from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-pressure-parallel";
import { runReferencePressureProbe } from "./fluid-reference-pressure-fixture";
import { runReferencePressureProjectionProbe } from "./fluid-reference-pressure-projection-fixture";

test("Reference projection preserves small pressure differences on large common offsets", async ({ page }) => {
    await page.goto("/");
    const gradient = await page.evaluate(runReferencePressureProjectionProbe, flipReferenceWgsl(true, true));
    expect(gradient).toBeCloseTo(-0.0001, 8);
});

for (const fused of [false, true]) {
    test(`Reference ${fused ? "fused" : "distributed"} pressure preserves accuracy, gauges, warm starts and failure budgets`, async ({ page }) => {
        await page.goto("/");
        const results = await page.evaluate(runReferencePressureProbe, {
            code: flipReferenceWgsl(true, true) + FLIP_REFERENCE_GPU_CONTROL_WGSL + FLIP_REFERENCE_GPU_PRESSURE_WGSL + FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_WGSL,
            bindings: { ...FLIP_REFERENCE_BINDINGS, ...FLIP_REFERENCE_GPU_PRESSURE_BINDINGS, ...FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_BINDINGS },
            fused,
            workgroupSize: 256,
        });
        expect(results).toHaveLength(11);
        expect(results.find((result) => result.scenario === "gauges")!.iterations).toBeGreaterThan(128);
        expect(results.find((result) => result.scenario === "short-budget")).toMatchObject({ iterations: 3, failure: 4 });
        expect(results.find((result) => result.scenario === "bad-diagonal")!.failure).toBe(2);
    });
}
