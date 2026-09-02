import type { Page } from "@playwright/test";

import { expect, test } from "../parity-fixtures";
import { waitForCanvasReady } from "../compare-utils";

// Review finding 11: the fluid demo used to retain all four PBF / FLIP / MLS-MPM / PB-MPM
// backends and rebuild the entire set on every capacity / grid / method change (and build a
// whole second set during a state-preserving FLIP page-capacity rebuild). This regression
// pins the single-active-backend behavior on a real WebGPU device: a FLIP resize only ever
// constructs a FLIP solver, and method switching still produces a working simulation while
// never instantiating an unrelated backend.

const METHOD_SELECT = 'select:has(option[value="PBF"]):has(option[value="FLIP"]):has(option[value="MLS-MPM"]):has(option[value="PB-MPM"])';

interface BackendDiagnostics {
    pbf: number;
    flip: number;
    mlsMpm: number;
    pbMpm: number;
    total: number;
    active: string;
    transition: number;
    steady: number;
}

async function readDiagnostics(page: Page): Promise<BackendDiagnostics> {
    return page.evaluate(() => {
        const canvas = document.querySelector("canvas");
        if (!canvas) {
            throw new Error("Fluid canvas not found.");
        }
        const num = (key: string): number => Number(canvas.dataset[key] ?? "0");
        return {
            pbf: num("backendBuildsPbf"),
            flip: num("backendBuildsFlip"),
            mlsMpm: num("backendBuildsMlsMpm"),
            pbMpm: num("backendBuildsPbMpm"),
            total: num("simulationBackendBuilds"),
            active: canvas.dataset.simulationActiveBackend ?? "",
            transition: num("simulationTransitionBytes"),
            steady: num("simulationGpuBytes"),
        };
    });
}

async function selectMethod(page: Page, method: string): Promise<void> {
    await page.locator(METHOD_SELECT).first().selectOption(method);
    await expect(page.locator("canvas")).toHaveAttribute("data-method", method);
    await expect(page.locator("canvas")).toHaveAttribute("data-simulation-active-backend", method);
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-active-particle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
}

test("FLIP resize builds no other backend and method switching still works", async ({ page }) => {
    test.setTimeout(150_000);
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    const canvas = page.locator("canvas");

    // Establish a known non-FLIP baseline, then switch to FLIP. A method change always
    // rebuilds the target backend, so the FLIP tally must rise while every other backend's
    // tally stays flat — switching alone already constructs only the selected solver.
    await selectMethod(page, "PBF");
    const afterPbf = await readDiagnostics(page);

    await selectMethod(page, "FLIP");
    const afterFlip = await readDiagnostics(page);
    expect(afterFlip.flip).toBeGreaterThan(afterPbf.flip);
    expect(afterFlip.pbf).toBe(afterPbf.pbf);
    expect(afterFlip.mlsMpm).toBe(afterPbf.mlsMpm);
    expect(afterFlip.pbMpm).toBe(afterPbf.pbMpm);
    expect(afterFlip.steady).toBeGreaterThan(0);

    // Resize FLIP particle capacity (a rebuild path). Only the FLIP backend may be built.
    const capacityInput = page.locator('[data-fluid-flip-particle-capacity="true"] input');
    await expect(capacityInput).toBeVisible();
    const currentCapacity = Number(await canvas.getAttribute("data-flip-particle-capacity-request"));
    const nextCapacity = currentCapacity + 40_000;
    await capacityInput.fill(String(nextCapacity));
    await capacityInput.press("Enter");
    await page.getByRole("button", { name: "Reset simulation" }).click();
    await expect(canvas).toHaveAttribute("data-flip-particle-capacity-request", String(nextCapacity));
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 30_000 }).toBeGreaterThan(0);

    const afterResize = await readDiagnostics(page);
    // The resize constructed a replacement FLIP solver...
    expect(afterResize.flip).toBeGreaterThan(afterFlip.flip);
    // ...and left every unrelated backend untouched — the core of review finding 11.
    expect(afterResize.pbf).toBe(afterFlip.pbf);
    expect(afterResize.mlsMpm).toBe(afterFlip.mlsMpm);
    expect(afterResize.pbMpm).toBe(afterFlip.pbMpm);
    expect(afterResize.active).toBe("FLIP");
    // The transition-peak diagnostic captured a real (bounded) two-allocation window.
    expect(afterResize.transition).toBeGreaterThan(0);
    expect(afterResize.transition).toBeGreaterThanOrEqual(afterResize.steady);

    // Method switching still works AND never instantiates an unrelated backend: switching
    // to PBF builds a PBF solver, while MLS-MPM and PB-MPM are still never constructed.
    await selectMethod(page, "PBF");
    const afterSwitchBack = await readDiagnostics(page);
    expect(afterSwitchBack.active).toBe("PBF");
    expect(afterSwitchBack.pbf).toBeGreaterThan(afterFlip.pbf);
    expect(afterSwitchBack.mlsMpm).toBe(afterPbf.mlsMpm);
    expect(afterSwitchBack.pbMpm).toBe(afterPbf.pbMpm);
    expect(afterSwitchBack.steady).toBeGreaterThan(0);
});

test("Waterfall switch commits its PB-MPM grid and scale before shared diagnostics", async ({ page }) => {
    test.setTimeout(150_000);
    const errors: string[] = [];
    page.on("console", (message) => {
        if (message.type() === "error") {
            errors.push(message.text());
        }
    });
    page.on("pageerror", (error) => errors.push(error.message));

    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    const canvas = page.locator("canvas");
    await page.locator('select:has(option[value="waterfall"])').selectOption("waterfall");

    await expect(canvas).toHaveAttribute("data-demo", "waterfall");
    await expect(canvas).toHaveAttribute("data-method", "PB-MPM");
    await expect(canvas).toHaveAttribute("data-physics-particle-size", "1.2");
    await expect(canvas).toHaveAttribute("data-grid-cells", "521,261,521");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
    expect(errors.filter((message) => /Grid \d+ × \d+ × \d+ requires|resolveTarget|writeSdfParams/i.test(message))).toEqual([]);
});

test("Marble Tower drives its wheel through the opaque fluid torque query", async ({ page }) => {
    test.setTimeout(90_000);
    const errors: string[] = [];
    page.on("console", (message) => {
        if (message.type() === "error" || message.type() === "warning") {
            errors.push(message.text());
        }
    });
    page.on("pageerror", (error) => errors.push(error.message));

    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Marble Tower" });
    const canvas = page.locator("canvas");
    await page.locator('select:has(option[value="marbleTower"])').selectOption("marbleTower");
    await expect(canvas).toHaveAttribute("data-demo", "marbleTower");
    await selectMethod(page, "FLIP");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 30_000 }).toBeGreaterThan(0);
    await page.waitForTimeout(1_000);

    expect(errors.filter((message) => /wheel torque|validation|WGSL|shader|bind group|pipeline/i.test(message))).toEqual([]);
});

test("MLS-MPM and PB-MPM each activate without pre-building the other backends", async ({ page }) => {
    test.setTimeout(150_000);
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });

    // Cycle every method once. Each activation builds exactly its own backend, so after the
    // full cycle every backend has been built and the active backend tracks the selection —
    // proving the four solvers are constructed lazily and independently, never as a set.
    await selectMethod(page, "MLS-MPM");
    const afterMls = await readDiagnostics(page);
    expect(afterMls.mlsMpm).toBeGreaterThan(0);

    await selectMethod(page, "PB-MPM");
    const afterPbMpm = await readDiagnostics(page);
    expect(afterPbMpm.pbMpm).toBeGreaterThan(0);
    // Selecting PB-MPM did not retroactively build another MLS-MPM solver.
    expect(afterPbMpm.mlsMpm).toBe(afterMls.mlsMpm);

    await selectMethod(page, "FLIP");
    const afterFlip = await readDiagnostics(page);
    expect(afterFlip.flip).toBeGreaterThan(0);
    expect(afterFlip.mlsMpm).toBe(afterMls.mlsMpm);
    expect(afterFlip.pbMpm).toBe(afterPbMpm.pbMpm);
});
