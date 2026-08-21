import type { Page } from "@playwright/test";

import { expect, test } from "../parity-fixtures";
import { waitForCanvasReady } from "../compare-utils";

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

async function setGridVector(page: Page, titlePrefix: string, values: [number, number, number]): Promise<void> {
    await page.evaluate(
        ({ titlePrefix, values }) => {
            const info = [...document.querySelectorAll<HTMLSpanElement>("span[title]")].find((element) => element.title.startsWith(titlePrefix));
            const row = info?.parentElement?.parentElement;
            const inputs = [...(row?.querySelectorAll<HTMLInputElement>('input[type="number"]') ?? [])];
            if (inputs.length !== 3) {
                throw new Error(`Grid vector control not found for ${titlePrefix}`);
            }
            for (let index = 0; index < 3; index++) {
                inputs[index]!.value = String(values[index]);
            }
            inputs[2]!.dispatchEvent(new Event("change", { bubbles: true }));
        },
        { titlePrefix, values }
    );
}

async function setCheckboxByInfo(page: Page, titlePrefix: string, checked: boolean): Promise<void> {
    await page.evaluate(
        ({ titlePrefix, checked }) => {
            const info = [...document.querySelectorAll<HTMLSpanElement>("span[title]")].find((element) => element.title.startsWith(titlePrefix));
            const input = info?.parentElement?.parentElement?.querySelector<HTMLInputElement>('input[type="checkbox"]');
            if (!input) {
                throw new Error(`Checkbox not found for ${titlePrefix}`);
            }
            input.checked = checked;
            input.dispatchEvent(new Event("change", { bubbles: true }));
        },
        { titlePrefix, checked }
    );
}

test("Whiteboard preserves authored state across fluid methods", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });

    const canvas = page.locator("canvas");
    const demoSelect = page.locator('select:has(option[value="whiteboard"])');
    const methodSelect = page.locator('select:has(option[value="PBF"]):has(option[value="FLIP"]):has(option[value="MLS-MPM"]):has(option[value="PB-MPM"])');
    const qualitySelect = page.locator('select:has(option[value="low"]):has(option[value="middle"]):has(option[value="high"])');
    const controlsPanel = page.locator('div[style*="resize: both"]').first();

    await demoSelect.selectOption("whiteboard");
    await expect(controlsPanel).toBeVisible();
    expect(await controlsPanel.evaluate((element) => element.style.resize)).toBe("both");
    await expect(canvas).toHaveAttribute("data-demo", "whiteboard");
    await expect(canvas).toHaveAttribute("data-emitter-count", "0");
    await expect(canvas).toHaveAttribute("data-sink-count", "0");
    await expect(canvas).toHaveAttribute("data-quality-presets", "false");
    await expect(qualitySelect).toBeHidden();
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 15_000 }).toBe(0);
    await expect(page.getByText("Active foam particles", { exact: true })).toHaveCount(0);
    const foamCounts = controlsPanel.locator('[data-fluid-foam-counts="true"]');
    await expect(foamCounts).toHaveText("Disabled");
    await page.getByLabel("Enable foam").check();
    await expect.poll(async () => foamCounts.locator("div").count(), { timeout: 15_000 }).toBe(4);
    const foamCountLines = await foamCounts.locator("div").allTextContents();
    expect(foamCountLines[0]).toContain("\u00a0/\u00a0");
    expect(foamCountLines[0]).toMatch(/particles$/);
    expect(foamCountLines.slice(1).map((line) => line.split(":")[0])).toEqual(["Foam", "Spray", "Bubbles"]);
    await page.getByLabel("Generate spray").uncheck();
    await expect
        .poll(async () => (await foamCounts.locator("div").allTextContents()).map((line) => line.split(":")[0]))
        .toEqual([foamCountLines[0]!.split(":")[0]!, "Foam", "Bubbles"]);
    await page.getByLabel("Generate spray").check();
    await page.getByLabel("Enable foam").uncheck();
    await methodSelect.selectOption("PBF");
    await expect(canvas).toHaveAttribute("data-method", "PBF");
    await expect(page.getByText(/^Relaxation ε/)).toBeVisible();
    await expect(page.getByText("FLIP advanced whitewater", { exact: true })).toBeHidden();
    await expect(page.getByText(/^Turbulence rate/)).toBeHidden();

    await page.getByRole("button", { name: "+ Emitter", exact: true }).click();
    await page.getByRole("button", { name: "+ Sink", exact: true }).click();
    const spreadInput = page
        .locator("label")
        .filter({ hasText: /^Spread/ })
        .locator("input")
        .first();
    await expect(spreadInput).toHaveAttribute("type", "text");
    await spreadInput.click();
    await spreadInput.press(process.platform === "darwin" ? "Meta+A" : "Control+A");
    await spreadInput.pressSequentially("0.35");
    await spreadInput.blur();
    await expect(spreadInput).toHaveValue("0.35");
    const delayBeforeStart = page
        .locator("label")
        .filter({ hasText: /^Delay before start/ })
        .locator("input");
    await delayBeforeStart.fill("2.5");
    await delayBeforeStart.blur();
    const sourceAndNormalField = page.locator("label").filter({ hasText: /^Source \+ normal/ });
    const sourceAndNormal = sourceAndNormalField.locator('input[type="checkbox"]');
    await sourceAndNormal.check();
    await expect(page.locator("label").filter({ hasText: /^Source velocity/ })).toHaveCount(0);
    await expect.poll(() => sourceAndNormalField.evaluate((label) => label.nextElementSibling?.textContent?.startsWith("Source factor") === true)).toBe(true);
    const sourceFactor = page
        .locator("label")
        .filter({ hasText: /^Source factor/ })
        .locator("input");
    await sourceFactor.fill("0.75");
    await sourceFactor.blur();
    const normalVelocity = page
        .locator("label")
        .filter({ hasText: /^Normal velocity/ })
        .locator("input");
    await normalVelocity.fill("-1.25");
    await normalVelocity.blur();
    await setGridVector(page, "World-space center of the simulation grid.", [2, 4, -3]);
    await setGridVector(page, "Exact world-space X/Y/Z extents", [10, 8, 12]);
    await setRangeByInfo(page, "Scales the SIMULATION particle radius", 0.73);
    await setRangeByInfo(page, "Downward acceleration applied to every particle", 3.2);
    await setCheckboxByInfo(page, "Displays the active solver's simulation-domain bounding box.", true);
    await setCheckboxByInfo(page, "Shows position and scale gizmos together.", true);

    await expect(canvas).toHaveAttribute("data-emitter-count", "1");
    await expect(canvas).toHaveAttribute("data-sink-count", "1");
    await expect(canvas).toHaveAttribute("data-grid-position", "2,4,-3");
    await expect(canvas).toHaveAttribute("data-grid-size", "10,8,12");
    await expect(canvas).toHaveAttribute("data-physics-particle-size", "0.73");
    await expect(canvas).toHaveAttribute("data-gravity", "3.2");
    await expect(canvas).toHaveAttribute("data-show-grid-bounds", "true");
    await expect(canvas).toHaveAttribute("data-grid-gizmo", "true");

    await methodSelect.selectOption("MLS-MPM");

    await expect(canvas).toHaveAttribute("data-method", "MLS-MPM");
    await expect(canvas).toHaveAttribute("data-emitter-count", "1");
    await expect(canvas).toHaveAttribute("data-sink-count", "1");
    await expect(canvas).toHaveAttribute("data-grid-position", "2,4,-3");
    await expect(canvas).toHaveAttribute("data-grid-size", "10,8,12");
    await expect(canvas).toHaveAttribute("data-physics-particle-size", "0.73");
    await expect(canvas).toHaveAttribute("data-gravity", "3.2");
    await expect(canvas).toHaveAttribute("data-show-grid-bounds", "true");
    await expect(canvas).toHaveAttribute("data-grid-gizmo", "true");
    await expect(page.getByText(/^Stiffness \(EOS\)/)).toBeVisible();
    await expect(page.getByText(/^Relaxation ε/)).toHaveCount(0);
    await expect(canvas).toHaveAttribute("data-mls-container-lo", /.+/);
    await expect(canvas).toHaveAttribute("data-mls-container-hi", /.+/);
    await expect(sourceAndNormal).toBeChecked();
    await expect(delayBeforeStart).toHaveValue("2.5");
    await expect(sourceFactor).toHaveValue("0.75");
    await expect(normalVelocity).toHaveValue("-1.25");

    await methodSelect.selectOption("FLIP");
    await expect(canvas).toHaveAttribute("data-method", "FLIP");
    await expect(canvas).toHaveAttribute("data-emitter-count", "1");
    await expect(canvas).toHaveAttribute("data-sink-count", "1");
    await expect(canvas).toHaveAttribute("data-grid-position", "2,4,-3");
    await expect(canvas).toHaveAttribute("data-grid-size", "10,8,12");
    await expect(canvas).toHaveAttribute("data-physics-particle-size", "0.48");
    await expect(canvas).toHaveAttribute("data-gravity", "3.2");
    await expect(canvas).toHaveAttribute("data-show-grid-bounds", "true");
    await expect(canvas).toHaveAttribute("data-grid-gizmo", "true");
    await expect(page.getByText(/^FLIP ratio/)).toBeVisible();
    await expect(page.getByText("FLIP advanced whitewater", { exact: true })).toBeVisible();
    await expect(page.getByText(/^Turbulence rate/)).toBeVisible();
    const advancedFlipFoam = page.locator("[data-fluid-flip-foam-advanced]");
    await expect(advancedFlipFoam).toBeVisible();
    await expect(page.getByText(/^Relaxation ε/)).toHaveCount(0);
    const limitVolumeRate = page.locator('[data-flow-field-label="Limit volume rate"] input[type="checkbox"]');
    await expect(limitVolumeRate).toBeVisible();
    await expect(limitVolumeRate).not.toBeChecked();
    await page
        .locator('[data-flow-field-label="Behavior"]')
        .filter({ has: page.locator('option[value="initial"]') })
        .locator("select")
        .selectOption("initial");
    const initialEmitterParticleCount = page.locator("[data-fluid-initial-emitter-particle-count]");
    await expect(initialEmitterParticleCount).toBeVisible();
    await setRangeByInfo(page, "FLIP grid voxels along the longest side", 100);
    await expect(initialEmitterParticleCount).toHaveText("4,630");
    await expect(canvas).toHaveAttribute("data-selected-initial-emitter-particle-count", "4630");
    await setRangeByInfo(page, "FLIP grid voxels along the longest side", 50);
    await expect(initialEmitterParticleCount).toHaveText("579");
    await expect(canvas).toHaveAttribute("data-selected-initial-emitter-particle-count", "579");

    await methodSelect.selectOption("PB-MPM");
    await expect(canvas).toHaveAttribute("data-method", "PB-MPM");
    await expect(advancedFlipFoam).toBeHidden();
    const materialSelect = page.getByText("PB-MPM material", { exact: true }).locator("..").locator("select");
    await expect(materialSelect).toBeVisible();
    await expect(materialSelect).toHaveValue("0");
    await materialSelect.selectOption("2");
    await expect(materialSelect).toHaveValue("2");
    await expect(page.getByText(/^Sand friction angle/)).toBeVisible();
    const waterColor = page.locator("label").filter({ hasText: "Water color" }).locator('input[type="color"]');
    const renderAsSpheres = page.locator("label").filter({ hasText: "Render as spheres" }).locator('input[type="checkbox"]');
    await expect(waterColor).toHaveValue("#c2b280");
    await expect(renderAsSpheres).toBeChecked();
    await methodSelect.selectOption("PBF");
    await methodSelect.selectOption("PB-MPM");
    await expect(materialSelect).toHaveValue("2");
    await expect(waterColor).toHaveValue("#c2b280");
    await expect(renderAsSpheres).toBeChecked();
});
