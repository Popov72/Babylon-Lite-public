import type { Page } from "@playwright/test";
import { PNG } from "pngjs";

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

test("screen-space surface switches to Ocean PBR", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });

    const canvas = page.locator("canvas");
    await page.locator('select:has(option[value="PBF"]):has(option[value="FLIP"]):has(option[value="MLS-MPM"]):has(option[value="PB-MPM"])').selectOption("PBF");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
    const renderAsSpheres = page.locator("label").filter({ hasText: "Render as spheres" }).locator('input[type="checkbox"]');
    await renderAsSpheres.uncheck();
    await expect(canvas).toHaveAttribute("data-render", "surface");
    await page.waitForTimeout(1_000);
    await page.keyboard.press("p");
    await expect(canvas).toHaveAttribute("data-paused", "true");
    const shader = page.locator('[data-fluid-surface-shader="true"]');
    await expect(shader).toHaveValue("physical");
    const physical = PNG.sync.read(await canvas.screenshot());

    await shader.selectOption("ocean");
    await expect(canvas).toHaveAttribute("data-surface-shader", "ocean");
    await page.waitForTimeout(500);
    const ocean = PNG.sync.read(await canvas.screenshot());

    let changedPixels = 0;
    for (let offset = 0; offset < physical.data.length; offset += 4) {
        const difference =
            Math.abs(physical.data[offset]! - ocean.data[offset]!) +
            Math.abs(physical.data[offset + 1]! - ocean.data[offset + 1]!) +
            Math.abs(physical.data[offset + 2]! - ocean.data[offset + 2]!);
        if (difference > 18) {
            changedPixels++;
        }
    }
    expect(changedPixels).toBeGreaterThan(1_000);
});

test("grid bounds can render as transparent solid faces", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });

    const canvas = page.locator("canvas");
    await page.keyboard.press("p");
    await expect(canvas).toHaveAttribute("data-paused", "true");
    const solidFaces = page.getByLabel("Solid faces");
    await expect(solidFaces).toBeVisible();
    await expect(solidFaces).toBeDisabled();
    await setCheckboxByInfo(page, "Displays the active solver's simulation-domain bounding box.", true);
    await expect(solidFaces).toBeEnabled();
    await setCheckboxByInfo(page, "Displays the active solver's simulation-domain bounding box.", false);
    await expect(solidFaces).toBeDisabled();
    await page.keyboard.press("F8");
    await page.waitForTimeout(250);
    const hidden = PNG.sync.read(await canvas.screenshot());

    await setCheckboxByInfo(page, "Displays the active solver's simulation-domain bounding box.", true);
    await page.waitForTimeout(250);
    const wireframe = PNG.sync.read(await canvas.screenshot());

    await setCheckboxByInfo(page, "Shows transparent, depth-tested faces instead of only the simulation-domain wireframe.", true);
    await expect(canvas).toHaveAttribute("data-show-grid-bounds-solid", "true");
    await page.waitForTimeout(250);
    const solid = PNG.sync.read(await canvas.screenshot());

    let filledPixels = 0;
    for (let offset = 0; offset < hidden.data.length; offset += 4) {
        const wireDifference =
            Math.abs(hidden.data[offset]! - wireframe.data[offset]!) +
            Math.abs(hidden.data[offset + 1]! - wireframe.data[offset + 1]!) +
            Math.abs(hidden.data[offset + 2]! - wireframe.data[offset + 2]!);
        const solidDifference =
            Math.abs(hidden.data[offset]! - solid.data[offset]!) +
            Math.abs(hidden.data[offset + 1]! - solid.data[offset + 1]!) +
            Math.abs(hidden.data[offset + 2]! - solid.data[offset + 2]!);
        if (wireDifference <= 6 && solidDifference > 24) {
            filledPixels++;
        }
    }
    expect(filledPixels).toBeGreaterThan(50_000);

    const colorAt = (x: number, y: number): [number, number, number] => {
        const offset = (Math.floor(y * solid.height) * solid.width + Math.floor(x * solid.width)) * 4;
        return [solid.data[offset]!, solid.data[offset + 1]!, solid.data[offset + 2]!];
    };
    const leftFace = colorAt(0.1, 0.45);
    const backFace = colorAt(0.5, 0.2);
    const floorFace = colorAt(0.5, 0.85);
    expect(leftFace[0] - leftFace[1]).toBeGreaterThan(30);
    expect(backFace[2] - backFace[0]).toBeGreaterThan(30);
    expect(floorFace[1] - floorFace[0]).toBeGreaterThan(30);
});

test("FLIP stage timing survives three substeps", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });

    await page.locator('select:has(option[value="whiteboard"])').selectOption("whiteboard");
    await page.getByRole("button", { name: "+ Emitter", exact: true }).click();
    await page.locator('select:has(option[value="PBF"]):has(option[value="FLIP"]):has(option[value="MLS-MPM"]):has(option[value="PB-MPM"])').selectOption("FLIP");
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-active-particle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
    test.skip((await page.getByText("GPU timing unavailable", { exact: false }).count()) > 0, "WebGPU timestamp queries are unavailable");

    await page.locator('[data-fluid-physics-param="polygonSurface"] input[type="checkbox"]').check();
    await page.getByLabel("Enable foam").check();
    await page.locator('[data-fluid-physics-param="minSubsteps"] input[type="range"]').evaluate((input: HTMLInputElement) => {
        input.value = "3";
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });

    for (const stage of ["Simulation", "Foam gen", "Surface render", "Foam render"]) {
        const row = page.locator(`[data-fluid-gpu-stage="${stage}"]`);
        await expect
            .poll(
                async () => {
                    const match = (await row.textContent())?.match(/([0-9.]+)\s*ms/);
                    return match ? Number(match[1]) : 0;
                },
                { timeout: 45_000, message: `${stage} should retain a timestamp pair` }
            )
            .toBeGreaterThan(0);
    }
});

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
    const strictSurfaceFiltering = page.getByLabel("Strict surface filtering");
    await expect(strictSurfaceFiltering).toBeDisabled();
    await expect(strictSurfaceFiltering).not.toBeChecked();
    await page.getByLabel("Enable foam").check();
    await expect(strictSurfaceFiltering).toBeEnabled();
    await strictSurfaceFiltering.check();
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
    await expect(strictSurfaceFiltering).toBeDisabled();
    await expect(strictSurfaceFiltering).not.toBeChecked();
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
    const solidGridBounds = page.getByLabel("Solid faces");
    await expect(solidGridBounds).toBeVisible();
    await expect(solidGridBounds).toBeDisabled();
    await setCheckboxByInfo(page, "Displays the active solver's simulation-domain bounding box.", true);
    await expect(solidGridBounds).toBeEnabled();
    await solidGridBounds.check();
    await setCheckboxByInfo(page, "Shows position and scale gizmos together.", true);

    await expect(canvas).toHaveAttribute("data-emitter-count", "1");
    await expect(canvas).toHaveAttribute("data-sink-count", "1");
    await expect(canvas).toHaveAttribute("data-grid-position", "2,4,-3");
    await expect(canvas).toHaveAttribute("data-grid-size", "10,8,12");
    await expect(canvas).toHaveAttribute("data-physics-particle-size", "0.73");
    await expect(canvas).toHaveAttribute("data-gravity", "3.2");
    await expect(canvas).toHaveAttribute("data-show-grid-bounds", "true");
    await expect(canvas).toHaveAttribute("data-show-grid-bounds-solid", "true");
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
    await expect(canvas).toHaveAttribute("data-show-grid-bounds-solid", "true");
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
    await expect(canvas).toHaveAttribute("data-show-grid-bounds-solid", "true");
    await expect(canvas).toHaveAttribute("data-grid-gizmo", "true");
    await expect(page.getByText(/^FLIP ratio/)).toBeVisible();
    const advancedNumerical = page.locator('[data-fluid-physics-group="advanced"]');
    const advancedNumericalSummary = advancedNumerical.locator("summary");
    await expect(page.getByText("Advanced numerical", { exact: true })).toHaveCount(1);
    await advancedNumericalSummary.click();
    await expect(advancedNumerical).not.toHaveAttribute("open", "");
    await advancedNumericalSummary.click();
    await expect(advancedNumerical).toHaveAttribute("open", "");
    const pressureSolver = page.locator('[data-fluid-physics-param="pressureSolver"] select');
    const pressureIterations = page.locator('[data-fluid-physics-param="pressureIterations"]');
    const pressureRelaxation = page.locator('[data-fluid-physics-param="pressureRelaxation"]');
    const multigridCycles = page.locator('[data-fluid-physics-param="multigridCycles"]');
    const pressureTolerance = page.locator('[data-fluid-physics-param="pressureTolerance"]');
    const pressureDiagnostics = page.locator('[data-fluid-physics-param="pressureDiagnostics"] input[type="checkbox"]');
    await expect(pressureSolver).toHaveValue("0");
    await expect(pressureIterations).toBeVisible();
    await expect(pressureRelaxation).toBeVisible();
    await expect(multigridCycles).toBeHidden();
    await pressureSolver.selectOption("1");
    await expect(canvas).toHaveAttribute("data-pressure-solver", "multigrid");
    await expect(pressureIterations).toBeHidden();
    await expect(pressureRelaxation).toBeHidden();
    await expect(multigridCycles).toBeVisible();
    await expect(pressureTolerance).toBeVisible();
    await expect(pressureDiagnostics).not.toBeChecked();
    await pressureDiagnostics.check();
    await expect(canvas).toHaveAttribute("data-pressure-diagnostics", "true");
    await expect(page.locator('[data-fluid-pressure-diagnostics="true"]')).toContainText("Pressure residual:");
    const liquidSdf = page.locator('[data-fluid-physics-param="liquidSdf"] input[type="checkbox"]');
    const ghostFluid = page.locator('[data-fluid-physics-param="ghostFluid"] input[type="checkbox"]');
    const fractionalSolids = page.locator('[data-fluid-physics-param="fractionalSolids"] input[type="checkbox"]');
    const movingSolidBoundaries = page.locator('[data-fluid-physics-param="movingSolidBoundaries"] input[type="checkbox"]');
    await expect(liquidSdf).not.toBeChecked();
    await expect(ghostFluid).toBeVisible();
    await expect(ghostFluid).toBeDisabled();
    await liquidSdf.check();
    await expect(canvas).toHaveAttribute("data-liquid-sdf", "true");
    await expect(ghostFluid).toBeEnabled();
    await ghostFluid.check();
    await expect(canvas).toHaveAttribute("data-ghost-fluid", "true");
    await liquidSdf.uncheck();
    await expect(canvas).toHaveAttribute("data-liquid-sdf", "false");
    await expect(ghostFluid).not.toBeChecked();
    await expect(ghostFluid).toBeDisabled();
    await expect(canvas).toHaveAttribute("data-ghost-fluid", "false");
    await liquidSdf.check();
    await expect(ghostFluid).toBeEnabled();
    await expect(ghostFluid).not.toBeChecked();
    await expect(fractionalSolids).not.toBeChecked();
    await expect(movingSolidBoundaries).toBeVisible();
    await expect(movingSolidBoundaries).toBeDisabled();
    await fractionalSolids.check();
    await expect(canvas).toHaveAttribute("data-fractional-solids", "true");
    await expect(movingSolidBoundaries).toBeEnabled();
    await movingSolidBoundaries.check();
    await expect(canvas).toHaveAttribute("data-moving-solid-boundaries", "true");
    await fractionalSolids.uncheck();
    await expect(canvas).toHaveAttribute("data-fractional-solids", "false");
    await expect(movingSolidBoundaries).not.toBeChecked();
    await expect(movingSolidBoundaries).toBeDisabled();
    await expect(canvas).toHaveAttribute("data-moving-solid-boundaries", "false");
    await fractionalSolids.check();
    await expect(movingSolidBoundaries).toBeEnabled();
    await expect(movingSolidBoundaries).not.toBeChecked();
    const reseedParticles = page.locator('[data-fluid-physics-param="reseedParticles"] input[type="checkbox"]');
    const reseedMinimum = page.locator('[data-fluid-physics-param="reseedMinParticles"]');
    await expect(reseedParticles).not.toBeChecked();
    await expect(reseedMinimum).toBeHidden();
    await reseedParticles.check();
    await expect(canvas).toHaveAttribute("data-reseed-particles", "true");
    await expect(reseedMinimum).toBeVisible();
    const particleSheeting = page.locator('[data-fluid-physics-param="particleSheeting"] input[type="checkbox"]');
    const sheetingStrength = page.locator('[data-fluid-physics-param="sheetingStrength"]');
    const polygonSurface = page.locator('[data-fluid-physics-param="polygonSurface"] input[type="checkbox"]');
    const polygonReconstruction = page.locator('[data-fluid-physics-param="polygonReconstructionMultiplier"]');
    await expect(sheetingStrength).toBeHidden();
    await expect(polygonReconstruction).toBeHidden();
    await particleSheeting.check();
    await expect(canvas).toHaveAttribute("data-particle-sheeting", "true");
    await expect(sheetingStrength).toBeVisible();
    await polygonSurface.check();
    await expect(canvas).toHaveAttribute("data-polygon-surface", "true");
    await expect(canvas).toHaveAttribute("data-render", "polygon");
    const polygonShader = page.locator('[data-fluid-polygon-shader="true"]');
    await expect(polygonShader).toHaveValue("physical");
    await polygonShader.selectOption("ocean");
    await expect(canvas).toHaveAttribute("data-polygon-shader", "ocean");
    await expect(polygonReconstruction).toBeVisible();
    const polygonReconstructionInput = polygonReconstruction.locator('input[type="range"]');
    await expect(polygonReconstructionInput).toHaveValue("1");
    await expect.poll(async () => Number(await canvas.getAttribute("data-polygon-triangle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
    await page.keyboard.press("p");
    await expect(canvas).toHaveAttribute("data-paused", "true");
    await polygonReconstructionInput.evaluate((input: HTMLInputElement) => {
        input.value = "2";
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(canvas).toHaveAttribute("data-polygon-reconstruction-multiplier", "2");
    const triangleCount = page.locator('[data-fluid-polygon-triangle-count="true"]');
    await expect(triangleCount).toBeVisible();
    await expect.poll(async () => Number(await canvas.getAttribute("data-polygon-triangle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
    await expect(triangleCount).toHaveText(/^Triangles:\u00a0[\d,]+$/);
    await expect(canvas).toHaveAttribute("data-paused", "true");
    await page.keyboard.press("p");
    await expect(canvas).toHaveAttribute("data-paused", "false");
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
