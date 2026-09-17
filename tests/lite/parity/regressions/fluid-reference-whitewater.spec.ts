import { expect, test } from "@playwright/test";
import { FLIP_REFERENCE_WHITEWATER_BINDINGS, flipReferenceWhitewaterWgsl } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/whitewater-shaders";
import { runReferenceWhitewaterProbe } from "./fluid-reference-whitewater-fixture";

test("Reference whitewater batches preserve births, partial workgroups, saturation and publication tails", async ({ page }) => {
    await page.goto("/");
    const results = await page.evaluate(runReferenceWhitewaterProbe, { code: flipReferenceWhitewaterWgsl(), bindings: FLIP_REFERENCE_WHITEWATER_BINDINGS });
    expect(results).toEqual([
        { capacity: 4096, emitted: 1096, rejected: 0, cleared: true },
        { capacity: 1024, emitted: 1024, rejected: 72, cleared: true },
    ]);
});
