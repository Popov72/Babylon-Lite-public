import { expect, test } from "../parity-fixtures";
import { attachCompareArtifacts, compareImages, waitForCanvasReady } from "../compare-utils";
import * as path from "path";

const REFERENCE_DIR = path.resolve(__dirname, "../../../../reference/lite/demo-fluid-waterfall-pbmpm-high");
const GOLDEN_PATH = path.join(REFERENCE_DIR, "known-good.png");
const ACTUAL_PATH = path.join(REFERENCE_DIR, "test-actual.png");
// Identical fixed-step runs measured about 2.6 MAD locally; retain device/GPU margin
// while still catching domain-scale or emitter-direction changes that move large regions.
const MAX_MAD = 8;

test("Waterfall PB-MPM High remains visually stable after 10 simulated seconds", async ({ page }, testInfo) => {
    test.setTimeout(180_000);

    await page.goto("/demo-fluid.html?demo=waterfall&method=PB-MPM&quality=high&fixedDt=0.016666666666666666&captureSeconds=10");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Waterfall fluid demo" });
    await waitForCanvasReady(page, { timeout: 150_000, label: "Waterfall at 10 simulated seconds", flag: "captureReady", pollMs: 100 });

    const state = await page.locator("canvas").evaluate((canvas) => ({
        error: (canvas as HTMLCanvasElement).dataset.error,
        method: (canvas as HTMLCanvasElement).dataset.method,
        captureTime: (canvas as HTMLCanvasElement).dataset.captureTime,
    }));
    expect(state.error).toBeUndefined();
    expect(state.method).toBe("PB-MPM");
    expect(Number(state.captureTime)).toBeCloseTo(10, 8);

    await page.locator("canvas").screenshot({ path: ACTUAL_PATH });
    const comparison = compareImages(ACTUAL_PATH, GOLDEN_PATH);
    await attachCompareArtifacts(testInfo, ACTUAL_PATH, GOLDEN_PATH, REFERENCE_DIR);

    expect(comparison.mad, `Waterfall MAD should be <= ${MAX_MAD}`).toBeLessThanOrEqual(MAX_MAD);
});
