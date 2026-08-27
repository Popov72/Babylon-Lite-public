import type { Page } from "@playwright/test";
import { PNG } from "pngjs";

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

test("Aquanova reports the authored placement position for dynamic entities", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/lite/demo-aquanova-fluid-sim.html");

    await expect.poll(() => page.evaluate(() => Boolean((window as unknown as { __aquanovaFluidSim?: unknown }).__aquanovaFluidSim))).toBe(true);
    await expect
        .poll(() =>
            page.evaluate(() => {
                const qa = (
                    window as unknown as {
                        __aquanovaFluidSim: { selectMeshGizmo(entityName: string): boolean };
                    }
                ).__aquanovaFluidSim;
                return qa.selectMeshGizmo("capsule");
            })
        )
        .toBe(true);
    await expect
        .poll(() =>
            page.evaluate(() =>
                (
                    window as unknown as {
                        __aquanovaFluidSim: { meshGizmoPosition(): [number, number, number] | null };
                    }
                ).__aquanovaFluidSim.meshGizmoPosition()
            )
        )
        .toEqual([-10, 0, -16.007999420166016]);
    await expect
        .poll(() =>
            page.evaluate(() =>
                (
                    window as unknown as {
                        __aquanovaFluidSim: { meshGizmoPositionText(): string | null };
                    }
                ).__aquanovaFluidSim.meshGizmoPositionText()
            )
        )
        .toContain("-10.00, 0.00, -16.01");
});

test("Aquanova highlights authoring action buttons when clicked", async ({ page }) => {
    await page.goto("/lite/demo-aquanova-fluid-sim.html");

    await expect.poll(() => page.evaluate(() => Boolean((window as unknown as { __aquanovaFluidSim?: unknown }).__aquanovaFluidSim))).toBe(true);
    const feedbackButtons = page.locator('button[data-click-feedback="true"]');
    await expect(feedbackButtons).toHaveCount(6);
    expect(await feedbackButtons.allTextContents()).toEqual(["Create", "Update", "Delete", "Reset liquefaction", "Export", "Import"]);

    const animationCount = await page.getByRole("button", { name: "Reset liquefaction" }).evaluate((button: HTMLButtonElement) => {
        button.click();
        return button.getAnimations().length;
    });
    expect(animationCount).toBeGreaterThan(0);
});

test("Aquanova shows shared grid controls and their dependencies by default", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/lite/demo-aquanova-fluid-sim.html");

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-ready", "true", { timeout: 90_000 });
    await page.evaluate(() => {
        (window as unknown as { __aquanovaFluidSim: { setPaused(paused: boolean): void } }).__aquanovaFluidSim.setPaused(true);
    });
    const hidden = PNG.sync.read(await canvas.screenshot());
    const gridBounds = page.getByLabel("Show grid bounds");
    const solidFaces = page.getByLabel("Solid faces");
    await expect(gridBounds).toBeVisible();
    await expect(solidFaces).toBeVisible();
    await expect(solidFaces).toBeDisabled();
    await gridBounds.check();
    await expect(solidFaces).toBeEnabled();
    await solidFaces.check();
    await expect(canvas).toHaveAttribute("data-show-grid-bounds-solid", "true");
    await page.waitForTimeout(250);
    const solid = PNG.sync.read(await canvas.screenshot());

    let filledPixels = 0;
    const testedWidth = Math.floor(solid.width * 0.7);
    for (let y = 0; y < solid.height; y++) {
        for (let x = 0; x < testedWidth; x++) {
            const offset = (y * solid.width + x) * 4;
            const difference =
                Math.abs(hidden.data[offset]! - solid.data[offset]!) +
                Math.abs(hidden.data[offset + 1]! - solid.data[offset + 1]!) +
                Math.abs(hidden.data[offset + 2]! - solid.data[offset + 2]!);
            if (difference > 24) {
                filledPixels++;
            }
        }
    }
    expect(filledPixels).toBeGreaterThan(10_000);
});

test("Aquanova shows FLIP Initial-emitter counts and Fluid physics particle size", async ({ page }) => {
    await page.goto("/lite/demo-aquanova-fluid-sim.html");

    await expect.poll(() => page.evaluate(() => Boolean((window as unknown as { __aquanovaFluidSim?: unknown }).__aquanovaFluidSim))).toBe(true);
    await page.evaluate(() => {
        const qa = (
            window as unknown as {
                __aquanovaFluidSim: {
                    setSimulationType(type: "mesh" | "fluid"): void;
                    setFlow(flow: {
                        emitters: Array<{
                            id: string;
                            name: string;
                            enabled: boolean;
                            behavior: "initial";
                            transform: { position: [number, number, number]; rotation: [number, number, number, number]; scale: [number, number, number] };
                            shape: { type: "box"; size: [number, number, number] };
                            sampling: "volume";
                            velocity: [number, number, number];
                        }>;
                        sinks: [];
                    }): void;
                };
            }
        ).__aquanovaFluidSim;
        qa.setSimulationType("fluid");
        qa.setFlow({
            emitters: [
                {
                    id: "initial-test",
                    name: "Initial test",
                    enabled: true,
                    behavior: "initial",
                    transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                    shape: { type: "box", size: [2, 2, 2] },
                    sampling: "volume",
                    velocity: [0, 0, 0],
                },
            ],
            sinks: [],
        });
    });

    const method = page
        .locator("select")
        .filter({ has: page.locator('option[value="FLIP"]') })
        .first();
    const physicsParticleSize = page.locator('span[title^="Scales the SIMULATION particle radius"]');
    for (const value of ["PBF", "MLS-MPM", "PB-MPM"]) {
        await method.selectOption(value);
        await expect(physicsParticleSize).toBeVisible();
    }
    await method.selectOption("PBF");
    const physicsParticleSizeInput = page.locator('[data-fluid-physics-particle-size="true"] input[type="range"]');
    await expect(physicsParticleSizeInput).toHaveAttribute("min", "0.1");
    await physicsParticleSizeInput.evaluate((input: HTMLInputElement) => {
        input.value = "2";
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect
        .poll(() =>
            page.evaluate(
                () =>
                    (
                        window as unknown as {
                            __aquanovaFluidSim: { exportPreset(): { physicsParticleSize: number } };
                        }
                    ).__aquanovaFluidSim.exportPreset().physicsParticleSize
            )
        )
        .toBe(2);
    await method.selectOption("FLIP");
    await expect(physicsParticleSize).toBeHidden();
    await expect(page.locator('[data-flow-field-label="FLIP particles"]')).toBeVisible();
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-selected-initial-emitter-particle-count"))).toBeGreaterThan(0);

    await page.evaluate(() =>
        (
            window as unknown as {
                __aquanovaFluidSim: { setSimulationType(type: "mesh" | "fluid"): void };
            }
        ).__aquanovaFluidSim.setSimulationType("mesh")
    );
    await method.selectOption("PBF");
    await expect(physicsParticleSize).toBeHidden();
});

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
