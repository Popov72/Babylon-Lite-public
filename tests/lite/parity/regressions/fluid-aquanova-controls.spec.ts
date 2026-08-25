import type { Page } from "@playwright/test";

import { expect, test } from "../parity-fixtures";

async function setRangeByInfo(page: Page, titlePrefix: string, value: number): Promise<void> {
    await page.evaluate(
        ({ titlePrefix, value }) => {
            const info = [...document.querySelectorAll<HTMLSpanElement>("span[title]")].find((element) => element.title.startsWith(titlePrefix));
            let host = info?.parentElement;
            while (host && !host.querySelector('input[type="range"]')) {
                host = host.parentElement;
            }
            const input = host?.querySelector<HTMLInputElement>('input[type="range"]');
            if (!input) {
                throw new Error(`Range control not found for ${titlePrefix}`);
            }
            input.value = String(value);
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
        },
        { titlePrefix, value }
    );
}

test("Aquanova applies the shared FLIP polygon-surface control", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/lite/demo-aquanova-fluid-sim.html");

    await expect.poll(() => page.evaluate(() => Boolean((window as unknown as { __aquanovaFluidSim?: unknown }).__aquanovaFluidSim))).toBe(true);

    const method = page
        .locator("select")
        .filter({ has: page.locator('option[value="FLIP"]') })
        .first();
    await method.selectOption("FLIP");
    await setRangeByInfo(page, "FLIP grid voxels along the longest side", 24);

    const simulationType = page
        .locator("select")
        .filter({ has: page.locator('option[value="fluid"]') })
        .first();
    await simulationType.selectOption("fluid");

    const polygonSurface = page.locator('[data-fluid-physics-param="polygonSurface"] input[type="checkbox"]');
    await expect(polygonSurface).toBeEnabled();
    await polygonSurface.check();
    const polygonReconstruction = page.locator('[data-fluid-physics-param="polygonReconstructionMultiplier"]');
    await expect(polygonReconstruction).toBeVisible();
    const polygonReconstructionInput = polygonReconstruction.locator('input[type="range"]');
    await expect(polygonReconstructionInput).toHaveValue("1");

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-render", "polygon");
    await page.evaluate(() => {
        const qa = (
            window as unknown as {
                __aquanovaFluidSim: { manualRunning(): boolean; setPaused(paused: boolean): void; startManual(): void };
            }
        ).__aquanovaFluidSim;
        qa.setPaused(false);
        if (!qa.manualRunning()) {
            qa.startManual();
        }
    });
    await expect.poll(() => page.evaluate(() => (window as unknown as { __aquanovaFluidSim: { manualRunning(): boolean } }).__aquanovaFluidSim.manualRunning())).toBe(true);
    await expect
        .poll(() => page.evaluate(() => (window as unknown as { __aquanovaFluidSim: { manualStepCount(): number } }).__aquanovaFluidSim.manualStepCount()))
        .toBeGreaterThan(0);
    await expect.poll(() => page.evaluate(() => (window as unknown as { __aquanovaFluidSim: { polygonSurfaceCount(): number } }).__aquanovaFluidSim.polygonSurfaceCount())).toBe(1);
    await expect.poll(() => page.evaluate(() => (window as unknown as { __aquanovaFluidSim: { renderMode(): string } }).__aquanovaFluidSim.renderMode())).toBe("polygon");
    await expect(canvas).toHaveAttribute("data-polygon-reconstruction-multiplier", "1");
    await expect.poll(async () => Number(await canvas.getAttribute("data-polygon-triangle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
    await page.evaluate(() => {
        (window as unknown as { __aquanovaFluidSim: { setPaused(paused: boolean): void } }).__aquanovaFluidSim.setPaused(true);
    });
    await expect(canvas).toHaveAttribute("data-paused", "true");
    await polygonReconstructionInput.evaluate((input: HTMLInputElement) => {
        input.value = "2";
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(canvas).toHaveAttribute("data-polygon-reconstruction-multiplier", "2");
    await expect.poll(async () => Number(await canvas.getAttribute("data-polygon-triangle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
    await expect(page.locator('[data-fluid-polygon-triangle-count="true"]')).toHaveText(/^Triangles:\u00a0[\d,]+$/);
    const debug = page.locator('select[data-fluid-debug="true"]');
    await debug.selectOption("polygonWireframe");
    await expect(debug).toHaveValue("polygonWireframe");
    await debug.selectOption("none");
});
