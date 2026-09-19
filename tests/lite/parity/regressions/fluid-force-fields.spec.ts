import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { readFile } from "node:fs/promises";
import type { FluidExportJson } from "../../../../packages/babylon-lite/src/index.js";

const PROBE_ENTRY = `/@fs/${resolve(__dirname, "../force-field-probe.ts").replace(/\\/g, "/")}`;

test("configured fields match their equations and compose on all production solvers", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas width="16" height="16"></canvas>
<script type="module">
import { runFluidForceFieldProbe } from "${PROBE_ENTRY}";
const canvas = document.querySelector("canvas");
try {
    canvas.dataset.result = JSON.stringify(await runFluidForceFieldProbe(canvas));
} catch (error) {
    canvas.dataset.error = error.stack || String(error);
}
</script>`);
    await page.waitForFunction(() => {
        const data = document.querySelector("canvas")?.dataset;
        return data?.result || data?.error;
    });
    const data = await page.locator("canvas").evaluate((canvas) => ({ ...canvas.dataset }));
    expect(data.error).toBeUndefined();
    expect(JSON.parse(data.result!).map((entry: { method: string }) => entry.method)).toEqual(["PBF", "FLIP", "MLS-MPM", "PB-MPM"]);
});

test("the force panel preserves authored fields through reset, import and method selection", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/demo-fluid.html");
    await page.waitForFunction(() => document.querySelector("canvas")?.dataset.ready === "true");
    const canvas = page.locator("canvas");
    const editor = page.locator("[data-fluid-force-fields]");
    async function exported(): Promise<FluidExportJson> {
        const download = page.waitForEvent("download");
        await page.getByRole("button", { name: "Export parameters", exact: true }).click();
        const file = await (await download).path();
        return JSON.parse(await readFile(file!, "utf8")) as FluidExportJson;
    }
    async function imported(preset: FluidExportJson): Promise<void> {
        await canvas.evaluate((canvas) => {
            delete canvas.dataset.presetImportStatus;
        });
        await page.locator('input[type="file"][multiple]').setInputFiles({ name: "forces.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(preset)) });
        await expect(canvas).toHaveAttribute("data-preset-import-status", "ready");
    }
    async function edit(label: string, value: string): Promise<void> {
        const input = editor.locator(`[data-force-field-label="${label}"] input`);
        await input.fill(value);
        await input.dispatchEvent("change");
    }
    await page.locator('select:has(option[value="whiteboard"])').selectOption("whiteboard");
    const small = await exported();
    small.meta.method = "FLIP";
    small.gridResolution = 24;
    small.gridSize = [4, 4, 4];
    small.gridPosition = [0, 0, 0];
    small.particleCount = 2048;
    small.markersPerCell = 2;
    small.pagedGrid = false;
    small.activeBlocks = false;
    small.fusedBlockDiscovery = false;
    small.foam.enableFoam = false;
    small.physics.kinematicViscosity = 0;
    small.physics.surfaceTension = 0;
    small.physics.polygonSurface = 0;
    small.emitters = [
        {
            id: "initial",
            name: "Initial",
            enabled: true,
            behavior: "initial",
            transform: { position: [0.5, 0.4, 0.5], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [0.25, 0.25, 0.25] },
            sampling: "volume",
            velocity: [0, 0, 0],
            velocitySpace: "world",
            spread: 0,
        },
    ];
    small.sinks = [];
    small.initialEmittersFillCapacity = false;
    await imported(small);
    const builds = await canvas.getAttribute("data-simulation-backend-builds");
    await editor.getByRole("button", { name: "Add point", exact: true }).click();
    await edit("Name", "Attractor");
    await edit("Strength", "-12");
    await editor.getByRole("spinbutton", { name: "Position X", exact: true }).fill("-1.25");
    await editor.getByRole("spinbutton", { name: "Position X", exact: true }).dispatchEvent("change");
    await edit("Max distance", "0.1");
    await expect(editor.getByRole("alert")).toContainText("minDistance");
    await expect(editor.locator('[data-force-field-label="Max distance"] input')).toHaveValue("5");
    await edit("Max distance", "6");
    await editor.getByRole("button", { name: "Add guide", exact: true }).click();
    await edit("Flow strength", "7");
    await edit("Spin strength", "-3");
    await editor.locator('[data-force-field-label="End caps"] input').uncheck();
    const authored = await exported();
    expect(authored.forceFields).toHaveLength(2);
    expect(authored.forceFields![0]).toMatchObject({ name: "Attractor", strength: -12, position: [-1.25, expect.any(Number), expect.any(Number)] });
    expect(authored.forceFields![1]).toMatchObject({ flowStrength: 7, spinStrength: -3, endCaps: false });
    await expect(canvas).toHaveAttribute("data-simulation-backend-builds", builds!);
    await page.locator("body").click({ position: { x: 10, y: 10 } });
    await page.keyboard.press("r");
    expect((await exported()).forceFields).toEqual(authored.forceFields);
    await editor.getByRole("button", { name: "Remove force", exact: true }).click();
    await expect(canvas).toHaveAttribute("data-force-field-count", "1");
    await imported(authored);
    expect((await exported()).forceFields).toEqual(authored.forceFields);
    for (const method of ["PBF", "MLS-MPM", "PB-MPM", "FLIP"]) {
        await page.locator('[data-fluid-method="true"]').selectOption(method);
        await expect(canvas).toHaveAttribute("data-method", method);
        expect((await exported()).forceFields).toEqual(authored.forceFields);
    }
    const implementation = page.locator('[data-fluid-backend="true"]');
    await implementation.selectOption("flip-reference");
    await expect(canvas).toHaveAttribute("data-reference-error", /disable configured force fields/);
    await expect(canvas).toHaveAttribute("data-fluid-backend", "production");
    for (const field of authored.forceFields!) {
        await editor.getByRole("combobox", { name: "Force field", exact: true }).selectOption(field.id);
        await editor.locator('[data-force-field-label="Enabled"] input').uncheck();
    }
    await implementation.selectOption("flip-reference");
    await expect(canvas).toHaveAttribute("data-fluid-backend", "flip-reference");
    await expect(editor.getByRole("button", { name: "Add point", exact: true })).toBeDisabled();
    await implementation.selectOption("production");
    await expect(canvas).toHaveAttribute("data-fluid-backend", "production");
    await expect(editor.getByRole("button", { name: "Add point", exact: true })).toBeEnabled();
    expect((await exported()).forceFields).toEqual(authored.forceFields!.map((field) => ({ ...field, enabled: false })));
    delete authored.forceFields;
    await imported(authored);
    await expect(canvas).toHaveAttribute("data-force-field-count", "0");
    expect((await exported()).forceFields).toBeUndefined();
});
