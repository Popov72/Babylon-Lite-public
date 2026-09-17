import { expect, test } from "@playwright/test";
import { FLIP_REFERENCE_ADVECTION_WGSL } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/advection";
import { runReferenceAdvectionProbe } from "./fluid-reference-advection-fixture";

test("Reference swept advection resolves early collisions and refines oversized trials within bounded work", async ({ page }) => {
    await page.goto("/");
    const result = await page.evaluate(runReferenceAdvectionProbe, FLIP_REFERENCE_ADVECTION_WGSL);
    expect(result).toHaveLength(16);
    expect(result.find((entry) => entry.name === "ordinary-nonlinear-rk3")!.velocityQueries).toBe(3);
    expect(result.find((entry) => entry.name === "nonlinear-refinement")!.velocityQueries).toBeGreaterThan(3);
    expect(result.find((entry) => entry.name === "discarded-unsupported-trial")!.errors[6]).toBe(0);
    expect(result.find((entry) => entry.name === "free-flight-budget")!.errors[5]).toBe(1);
});
