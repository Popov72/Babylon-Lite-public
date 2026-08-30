import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const FLIP_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/flip-sim.ts").replace(/\\/g, "/")}`;

test("paged FLIP runs beyond the dense binding limit", async ({ page }) => {
    test.setTimeout(240_000);
    page.on("console", (message) => console.log(message.type(), message.text()));
    page.on("pageerror", (error) => console.log("pageerror", error.message));
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";
const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", event => canvas.dataset.error = event.message);
window.addEventListener("unhandledrejection", event => canvas.dataset.error = event.reason?.message ?? String(event.reason));
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const sim = createFlipSim(engine, {
        count: 40000,
        initialPositions: (() => {
            const positions = new Float32Array(40000 * 3);
            for (let i = 0; i < 40000; i++) {
                positions[i * 3 + 1] = 1;
            }
            return positions;
        })(),
        boundsMin: [-30, 0, -30],
        boundsMax: [30, 30, 30],
        gridDim: [600, 300, 600],
        dx: 0.1,
        particleRadius: 0.025,
        pagedGrid: true,
        pagedGridMaxPages: 64,
        onPagedGridOverflow: (required, capacity) => canvas.dataset.overflow = required + "/" + capacity,
        pressureIterations: 4,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1 / 60,
    });
    sim.setFlow({
        emitters: [{
            id: "initial",
            name: "Initial",
            enabled: true,
            behavior: "initial",
            transform: { position: [0, 2, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [4, 2.5, 4] },
            sampling: "volume",
            velocity: [0, 0, 0],
        }],
        sinks: [],
        initialEmittersFillCapacity: true,
    });
    const sdfParams = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const sdfGrid = device.createBuffer({ size: 512 * 512 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sdfParams, 0, new Float32Array([
        -0.5, -0.5, 511, 511,
        512, 512, 0, 1,
        1, 0, 0, 0,
        0, 0, 0, 0,
    ]));
    sim.setSceneSdf({
        struct: "struct SceneSdfParams { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, };",
        sdf: \`fn wfGrid(i: i32, j: i32, nx: i32, nz: i32) -> f32 {
            let c = clamp(vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(nx - 1, nz - 1));
            return sceneSdfGrid[c.x + nx * c.y];
        }
        fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
            return pt.y - wfGrid(i32(pt.x), i32(pt.z), i32(sceneSdfParams.p1.x), i32(sceneSdfParams.p1.y));
        }\`,
        buffer: sdfParams,
        sdfGrid,
        gridConfine: false,
    });
    sim.reset();

    sim.setFoam({ activeParticles: true, kTa: 40, kWc: 40, poolScale: 2, poolCapMax: 64 });
    for (let frame = 0; frame < 2; frame++) {
        const encoder = device.createCommandEncoder();
        sim.step(encoder, 1 / 60);
        device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    const readback = device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, 8 * 16);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    canvas.dataset.result = JSON.stringify({ finite: values.every(Number.isFinite), y: values[1], bytes: sim.gpuBytes });
    readback.unmap();
    readback.destroy();
    sim.dispose();
    sdfParams.destroy();
    sdfGrid.destroy();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 180_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as { finite: boolean; y: number; bytes: number };
    expect(result.finite).toBe(true);
    expect(result.y).not.toBeCloseTo(1, 4);
    expect(result.bytes).toBeLessThan(64 * 1024 * 1024);
});

test("paged FLIP keeps multigrid across an asynchronous scene-SDF replacement", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";
const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", event => canvas.dataset.error = event.message);
window.addEventListener("unhandledrejection", event => canvas.dataset.error = event.reason?.message ?? String(event.reason));
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const count = 512;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        positions[i * 3] = 1.5 + (i % 8) * 0.08;
        positions[i * 3 + 1] = 2 + (Math.floor(i / 64) % 8) * 0.08;
        positions[i * 3 + 2] = 1.5 + (Math.floor(i / 8) % 8) * 0.08;
    }
    const sim = createFlipSim(engine, {
        count,
        initialPositions: positions,
        boundsMin: [0, 0, 0],
        boundsMax: [4, 4, 4],
        gridDim: [32, 32, 32],
        dx: 0.125,
        particleRadius: 0.04,
        pagedGrid: true,
        pagedGridMaxPages: 64,
        pressureSolver: "multigrid",
        multigridCycles: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1 / 120,
    });
    const first = device.createCommandEncoder();
    sim.step(first, 1 / 120);
    device.queue.submit([first.finish()]);

    const sdfParams = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const sdfGrid = device.createBuffer({ size: 8 * 8 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sdfGrid, 0, new Float32Array(8 * 8).fill(0.25));
    sim.setSceneSdf({
        struct: "struct SceneSdfParams { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, };",
        sdf: \`fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
            let sample = sceneSdfGrid[clamp(i32(pt.x), 0, 7) + 8 * clamp(i32(pt.z), 0, 7)];
            return pt.y - sample + sceneSdfParams.p0.x;
        }\`,
        buffer: sdfParams,
        sdfGrid,
        gridConfine: false,
    });
    for (let frame = 0; frame < 4; frame++) {
        const encoder = device.createCommandEncoder();
        sim.step(encoder, 1 / 120);
        device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    const readback = device.createBuffer({ size: 32 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, readback.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    canvas.dataset.result = JSON.stringify({
        finite: values.every(Number.isFinite),
        moved: values.some((value, index) => index % 4 < 3 && value !== positions[Math.floor(index / 4) * 3 + index % 4]),
    });
    readback.unmap();
    readback.destroy();
    sim.dispose();
    sdfParams.destroy();
    sdfGrid.destroy();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect.poll(async () => (await canvas.getAttribute("data-result")) ?? (await canvas.getAttribute("data-error")), { timeout: 90_000 }).toBeTruthy();
    expect(await canvas.getAttribute("data-error")).toBeNull();
    expect(JSON.parse((await canvas.getAttribute("data-result"))!)).toEqual({ finite: true, moved: true });
});

test("preserves live FLIP particles while switching to paged storage", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim, transferFlipSimState } from "${FLIP_ENTRY}";
const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", event => canvas.dataset.error = event.message);
window.addEventListener("unhandledrejection", event => canvas.dataset.error = event.reason?.message ?? String(event.reason));
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const positions = count => {
        const values = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            values[i * 3] = (i % 4) * 0.1;
            values[i * 3 + 1] = 2 + Math.floor(i / 16) * 0.1;
            values[i * 3 + 2] = (Math.floor(i / 4) % 4) * 0.1;
        }
        return values;
    };
    const common = {
        count: 64,
        boundsMin: [-2, 0, -2],
        boundsMax: [2, 4, 2],
        gridDim: [16, 16, 16],
        dx: 0.25,
        particleRadius: 0.05,
        pressureIterations: 4,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1 / 60,
    };
    const dense = createFlipSim(engine, { ...common, initialPositions: positions(32) });
    const denseStep = device.createCommandEncoder();
    dense.step(denseStep, 1 / 60);
    device.queue.submit([denseStep.finish()]);
    const paged = createFlipSim(engine, {
        ...common,
        initialPositions: positions(8),
        pagedGrid: true,
        pagedGridMaxPages: 64,
    });
    const before = device.createBuffer({ size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const after = device.createBuffer({ size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const transfer = device.createCommandEncoder();
    transfer.copyBufferToBuffer(dense.positionBuffer, 0, before, 0, 64);
    transfer.copyBufferToBuffer(dense.velocityBuffer, 0, before, 64, 64);
    const transferred = transferFlipSimState(transfer, dense, paged);
    transfer.copyBufferToBuffer(paged.positionBuffer, 0, after, 0, 64);
    transfer.copyBufferToBuffer(paged.velocityBuffer, 0, after, 64, 64);
    device.queue.submit([transfer.finish()]);
    await Promise.all([before.mapAsync(GPUMapMode.READ), after.mapAsync(GPUMapMode.READ)]);
    const sourceValues = Array.from(new Float32Array(before.getMappedRange().slice(0)));
    const targetValues = Array.from(new Float32Array(after.getMappedRange().slice(0)));
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({
        transferred,
        identical: sourceValues.every((value, index) => value === targetValues[index]),
        activeCount: paged.activeCount,
        renderCount: paged.renderCount,
    });
    before.unmap();
    after.unmap();
    before.destroy();
    after.destroy();
    dense.dispose();
    paged.dispose();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 60_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    expect(JSON.parse((await canvas.getAttribute("data-result"))!)).toEqual({
        transferred: true,
        identical: true,
        activeCount: 32,
        renderCount: 32,
    });
});

test("enabling paging preserves the live multigrid pressure solver", async ({ page }) => {
    test.setTimeout(120_000);
    const preset = JSON.parse(readFileSync(resolve(__dirname, "../../../../lab/public/fluid-presets/waterfall.sph.low.json"), "utf8"));
    preset.formatVersion = 13;
    preset.meta.method = "FLIP";
    preset.pagedGrid = false;
    preset.pagedGridMaxPages = 64;
    preset.gridPosition = [0, 2, 0];
    preset.gridSize = [4, 4, 4];
    preset.gridResolution = 32;
    preset.markersPerCell = 8;
    preset.particleCount = 8;
    preset.physics.pressureSolver = 1;
    await page.goto("/lite/demo-fluid.html");
    const canvas = page.locator("canvas");
    await expect(canvas).toHaveAttribute("data-ready", "true", { timeout: 60_000 });
    await page.locator('select:has(option[value="FLIP"])').selectOption("FLIP");
    await expect(canvas).toHaveAttribute("data-method", "FLIP", { timeout: 60_000 });
    await page.keyboard.press("p");
    await expect(canvas).toHaveAttribute("data-paused", "true");
    await page.locator('input[type="file"][accept*=".json"]').setInputFiles({
        name: "fluid-paged-multigrid-FLIP.json",
        mimeType: "application/json",
        buffer: Buffer.from(JSON.stringify(preset)),
    });
    await expect(canvas).toHaveAttribute("data-pressure-solver", "multigrid");
    const capacityRow = page.locator('[data-fluid-paged-grid-capacity="true"]');
    const capacitySlider = capacityRow.locator('input[type="range"]');
    await expect(capacityRow).toHaveCSS("display", "block");
    expect(
        await capacityRow.evaluate((row) => {
            const [, value, slider] = Array.from(row.children);
            return value!.getBoundingClientRect().bottom <= slider!.getBoundingClientRect().top;
        })
    ).toBe(true);
    await expect(capacitySlider).toHaveAttribute("max", "1");
    await expect(capacityRow).toContainText("64 pages");
    await capacitySlider.dispatchEvent("input");
    await capacitySlider.dispatchEvent("change");
    await expect(canvas).toHaveAttribute("data-paged-grid-max-pages", "64");
    await page.locator('label:has-text("Paged grid") input[type="checkbox"]').check();
    await expect(canvas).toHaveAttribute("data-paged-grid", "true", { timeout: 60_000 });
    await expect(canvas).toHaveAttribute("data-pressure-solver", "multigrid");
    await expect(canvas).toHaveAttribute("data-paged-grid-overflow", "false");
});

test("imports paging and an above-dense-limit FLIP grid atomically", async ({ page }) => {
    test.setTimeout(240_000);
    const preset = JSON.parse(readFileSync(resolve(__dirname, "../../../../lab/public/fluid-presets/waterfall.sph.low.json"), "utf8"));
    preset.formatVersion = 13;
    preset.meta.method = "FLIP";
    preset.pagedGrid = true;
    preset.pagedGridMaxPages = 64;
    preset.gridPosition = [0, 12, 0];
    preset.gridSize = [50, 24, 50];
    preset.gridResolution = 600;
    preset.markersPerCell = 8;
    preset.particleCount = 8;
    await page.goto("/lite/demo-fluid.html");
    const canvas = page.locator("canvas");
    await expect(canvas).toHaveAttribute("data-ready", "true", { timeout: 60_000 });
    await page.locator('select:has(option[value="waterfall"])').selectOption("waterfall");
    await expect(canvas).toHaveAttribute("data-demo", "waterfall", { timeout: 60_000 });
    await page.locator('select:has(option[value="FLIP"])').selectOption("FLIP");
    await expect(canvas).toHaveAttribute("data-method", "FLIP", { timeout: 60_000 });
    await page.keyboard.press("p");
    await expect(canvas).toHaveAttribute("data-paused", "true");
    await page.locator('input[type="file"][accept*=".json"]').setInputFiles({
        name: "fluid-paged-FLIP.json",
        mimeType: "application/json",
        buffer: Buffer.from(JSON.stringify(preset)),
    });
    await expect(canvas).toHaveAttribute("data-paged-grid", "true", { timeout: 10_000 });
    await expect(canvas).toHaveAttribute("data-grid-cells", "600,288,600", { timeout: 480_000 });
    await expect(canvas).toHaveAttribute("data-paged-grid-max-pages", "64");
    await expect(canvas).toHaveAttribute("data-paged-grid-overflow", "false");
});
