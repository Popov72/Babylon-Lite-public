import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const FLIP_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/solvers/flip-sim.ts").replace(/\\/g, "/")}`;

test("DOM control commits FLIP state with an explicitly submitted encoder", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<button id="reconfigure">Reconfigure</button>
<script type="module">
import {
    commitFluidReconfiguration,
    createEngine,
    createFluidSimulation,
    prepareFluidReconfigurationUpdate,
    readFluidSimulationPositions,
    writeFluidSimulationPositions,
} from "${LITE_ENTRY}";
const canvas = document.getElementById("renderCanvas");
const button = document.getElementById("reconfigure");
window.addEventListener("error", event => canvas.dataset.error = event.message);
window.addEventListener("unhandledrejection", event => canvas.dataset.error = event.reason?.message ?? String(event.reason));
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const initialPositions = new Float32Array(32 * 3);
    const sim = createFluidSimulation(engine, {
        method: "FLIP",
        particleCount: 32,
        bounds: { min: [-1, 0, -1], max: [1, 2, 1] },
        physicsScale: 1,
        gridResolution: 8,
        markersPerCell: 4,
        initialPositions,
        physics: { minSubsteps: 1, maxSubsteps: 1, pressureIterations: 2 },
    });
    const live = new Float32Array([
        0.125, 0.75, -0.25,
        -0.5, 1.25, 0.375,
        0.625, 1.5, -0.75,
        -0.875, 0.5, 0.875,
    ]);
    writeFluidSimulationPositions(sim, live);
    await device.queue.onSubmittedWorkDone();
    const before = Array.from(await readFluidSimulationPositions(sim, { particleCount: 4 }));
    button.onclick = async () => {
        device.pushErrorScope("validation");
        const prepared = prepareFluidReconfigurationUpdate(sim, { particleCount: 64 }, true);
        commitFluidReconfiguration(prepared);
        await device.queue.onSubmittedWorkDone();
        const validation = await device.popErrorScope();
        if (validation) {
            throw validation;
        }
        const after = Array.from(await readFluidSimulationPositions(sim, { particleCount: 4 }));
        canvas.dataset.result = JSON.stringify({ before, after, count: sim.count });
    };
    button.click();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 90_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as { before: number[]; after: number[]; count: number };
    expect(result.count).toBe(64);
    expect(result.after).toEqual(result.before);
});

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
        pagedGridMaxPages: 512,
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
    page.on("console", (message) => console.log(message.type(), message.text()));
    page.on("pageerror", (error) => console.log("pageerror", error.message));
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
    await expect(canvas).toHaveAttribute("data-grid-cells", "600,288,600", { timeout: 480_000 });
    await expect(canvas).toHaveAttribute("data-paged-grid", "true");
    await expect(canvas).toHaveAttribute("data-paged-grid-max-pages", "64");
    await expect(canvas).toHaveAttribute("data-paged-grid-overflow", "false");
});

test("paged FLIP reports the effective clamped page capacity", async ({ page }) => {
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
    const sim = createFlipSim(engine, {
        count: 8,
        initialPositions: new Float32Array([0.25, 0.25, 0.25]),
        boundsMin: [0, 0, 0],
        boundsMax: [1, 1, 1],
        gridDim: [8, 8, 8],
        dx: 0.125,
        particleRadius: 0.04,
        pagedGrid: true,
        pagedGridMaxPages: 64,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
        onPagedGridPages: (required, capacity) => {
            canvas.dataset.result = JSON.stringify({ required, capacity });
        },
    });
    for (let frame = 0; frame < 3 && !canvas.dataset.result; frame++) {
        const encoder = device.createCommandEncoder();
        sim.step(encoder, 1 / 60);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await Promise.resolve();
    }
    for (let attempt = 0; attempt < 60 && !canvas.dataset.result; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 10));
    }
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    if (!canvas.dataset.result) throw new Error("Paged-grid status callback did not report.");
    sim.dispose();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect.poll(async () => (await canvas.getAttribute("data-result")) ?? (await canvas.getAttribute("data-error")), { timeout: 90_000 }).toBeTruthy();
    expect(await canvas.getAttribute("data-error")).toBeNull();
    expect(JSON.parse((await canvas.getAttribute("data-result"))!)).toEqual({ required: 1, capacity: 1 });
});

test("paged FLIP inflow grows again after moving beyond the discovered page halo", async ({ page }) => {
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
function emitter(x) {
    return {
        id: "source",
        name: "Source",
        enabled: true,
        behavior: "inflow",
        transform: { position: [x, 0.8, 0.8], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
        shape: { type: "box", size: [0.4, 0.4, 0.4] },
        sampling: "volume",
        velocity: [0, 0, 0],
        velocitySpace: "world",
        spread: 0,
    };
}
async function sample(sim, device) {
    const readback = device.createBuffer({ size: sim.count * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = device.createCommandEncoder();
    copy.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, sim.count * 16);
    device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const positions = new Float32Array(readback.getMappedRange().slice(0));
    let total = 0;
    let moved = 0;
    for (let index = 0; index < sim.count; index++) {
        const x = positions[index * 4];
        const y = positions[index * 4 + 1];
        if (y <= -1000) {
            continue;
        }
        total++;
        if (x > 2.5) {
            moved++;
        }
    }
    readback.unmap();
    readback.destroy();
    return { total, moved };
}
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const sim = createFlipSim(engine, {
        count: 256,
        initialPositions: new Float32Array(0),
        boundsMin: [0, 0, 0],
        boundsMax: [4.8, 1.6, 1.6],
        gridDim: [48, 16, 16],
        dx: 0.1,
        markersPerCell: 1,
        particleRadius: 0.04,
        gravity: 0,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
        pagedGrid: true,
        pagedGridMaxPages: 64,
    });
    sim.setFlow({ emitters: [emitter(0.6)], sinks: [] });
    sim.reset();
    for (let frame = 0; frame < 2; frame++) {
        const encoder = device.createCommandEncoder();
        sim.step(encoder, 1 / 60);
        device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const beforeMove = await sample(sim, device);
    sim.setFlow({ emitters: [emitter(3.4)], sinks: [] });
    const movedEncoder = device.createCommandEncoder();
    sim.step(movedEncoder, 1 / 60);
    device.queue.submit([movedEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const afterMove = await sample(sim, device);
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({ beforeMove, afterMove, markersPerCell: 1 });
    sim.dispose();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 90_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        beforeMove: { total: number; moved: number };
        afterMove: { total: number; moved: number };
        markersPerCell: number;
    };
    expect(result.beforeMove.total).toBeGreaterThan(result.markersPerCell);
    expect(result.afterMove.total).toBeGreaterThan(result.beforeMove.total + result.markersPerCell);
    expect(result.afterMove.moved).toBeGreaterThan(result.markersPerCell);
});

test("paged legacy recycle preflights relaunch destinations before teleporting", async ({ page }) => {
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
async function readPosition(device, buffer) {
    const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = device.createCommandEncoder();
    copy.copyBufferToBuffer(buffer, 0, readback, 0, 16);
    device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const position = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    readback.unmap();
    readback.destroy();
    return position;
}
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    let resolveOverflow;
    const overflowed = new Promise(resolve => resolveOverflow = resolve);
    let demand = 0;
    const sim = createFlipSim(engine, {
        count: 1,
        initialPositions: new Float32Array([0.4, 0.8, 0.8]),
        boundsMin: [0, 0, 0],
        boundsMax: [4.8, 1.6, 1.6],
        gridDim: [48, 16, 16],
        dx: 0.1,
        particleRadius: 0.04,
        gravity: 0,
        pagedGrid: true,
        pagedGridMaxPages: 12,
        pressureIterations: 1,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
        onPagedGridPages: required => demand = required,
        onPagedGridOverflow: () => resolveOverflow(),
    });
    sim.setEmitters({
        emitters: [{ pos: [3.6, 0.8, 0.8], dir: [1, 0, 0], speed: 0, radius: 0.1, halfExtents: [0.1, 0.1, 0.1] }],
        intakeMin: [0.2, 0.6, 0.6],
        intakeMax: [0.6, 1.0, 1.0],
        rate: 1000,
        spread: 0,
    });
    sim.reset();
    const before = await readPosition(device, sim.positionBuffer);
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    sim.step(device.createCommandEncoder(), 0);
    await overflowed;
    const after = await readPosition(device, sim.positionBuffer);
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({
        demand,
        identical: before.every((value, index) => value === after[index]),
    });
    sim.dispose();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect.poll(async () => (await canvas.getAttribute("data-result")) ?? (await canvas.getAttribute("data-error")), { timeout: 90_000 }).toBeTruthy();
    expect(await canvas.getAttribute("data-error")).toBeNull();
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as { demand: number; identical: boolean };
    expect(result.demand).toBeGreaterThan(12);
    expect(result.identical).toBe(true);
});

test("refreshPolygonSurface rebuilds paged lookup state after reset while paused", async ({ page }) => {
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
function initialPositions() {
    const positions = new Float32Array(64 * 3);
    for (let i = 0; i < 64; i++) {
        positions[i * 3] = 0.6 + (i % 4) * 0.09;
        positions[i * 3 + 1] = 0.6 + (Math.floor(i / 4) % 4) * 0.09;
        positions[i * 3 + 2] = 0.6 + (Math.floor(i / 16) % 4) * 0.09;
    }
    return positions;
}
async function readIndexCount(surface, device) {
    const readback = device.createBuffer({ size: 20, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = device.createCommandEncoder();
    copy.copyBufferToBuffer(surface.drawIndirect, 0, readback, 0, 20);
    device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const count = new Uint32Array(readback.getMappedRange().slice(0))[0];
    readback.unmap();
    readback.destroy();
    return count;
}
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const sim = createFlipSim(engine, {
        count: 64,
        initialPositions: initialPositions(),
        boundsMin: [0, 0, 0],
        boundsMax: [4, 4, 4],
        gridDim: [32, 32, 32],
        dx: 0.125,
        particleRadius: 0.04,
        gravity: 0,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
        pagedGrid: true,
        pagedGridMaxPages: 64,
        polygonSurface: true,
    });
    const first = device.createCommandEncoder();
    sim.step(first, 1 / 60);
    device.queue.submit([first.finish()]);
    await device.queue.onSubmittedWorkDone();
    const surface = sim.polygonSurface;
    if (!surface) throw new Error("Polygon surface was not created.");
    const beforeReset = await readIndexCount(surface, device);
    sim.reset();
    const refresh = device.createCommandEncoder();
    sim.refreshPolygonSurface(refresh);
    device.queue.submit([refresh.finish()]);
    await device.queue.onSubmittedWorkDone();
    const afterReset = await readIndexCount(surface, device);
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({ beforeReset, afterReset });
    sim.dispose();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 90_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as { beforeReset: number; afterReset: number };
    expect(result.beforeReset).toBeGreaterThan(0);
    expect(result.afterReset).toBeGreaterThan(0);
});

test("paged overflow leaves particles, foam, polygon buffers, and recoverable history byte-identical", async ({ page }) => {
    test.setTimeout(120_000);
    page.on("console", (message) => console.log(message.type(), message.text()));
    page.on("pageerror", (error) => console.log("pageerror", error.message));
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim, transferFlipSimState } from "${FLIP_ENTRY}";
const canvas = document.getElementById("renderCanvas");
async function bytes(device, ranges) {
    const size = ranges.reduce((sum, range) => sum + range.size, 0);
    const readback = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    let offset = 0;
    for (const range of ranges) {
        encoder.copyBufferToBuffer(range.buffer, range.offset ?? 0, readback, offset, range.size);
        offset += range.size;
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    return result;
}
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const captured = new Map();
    const captureLabels = new Set(["fluid-particle-lifecycle", "flip-foam-active-dispatch", "flip-polygon-surface-reconstructed-sdf"]);
    const simEngine = {
        _device: new Proxy(device, {
            get(target, prop) {
                if (prop === "createBuffer") {
                    return descriptor => {
                        const capture = captureLabels.has(descriptor.label);
                        const buffer = target.createBuffer(capture ? { ...descriptor, usage: descriptor.usage | GPUBufferUsage.COPY_SRC } : descriptor);
                        if (capture) captured.set(descriptor.label, buffer);
                        return buffer;
                    };
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        }),
    };
    device.pushErrorScope("validation");
    let resolvePages;
    let resolveOverflow;
    const pagesReported = new Promise(resolve => resolvePages = resolve);
    const overflowed = new Promise(resolve => resolveOverflow = resolve);
    const sim = createFlipSim(simEngine, {
        count: 4,
        initialPositions: new Float32Array([
            0.2, 0.2, 0.2,
            0.3, 0.2, 0.2,
            0.4, 0.2, 0.2,
            0.5, 0.2, 0.2,
        ]),
        boundsMin: [0, 0, 0],
        boundsMax: [3.2, 3.2, 3.2],
        gridDim: [32, 32, 32],
        dx: 0.1,
        particleRadius: 0.04,
        pagedGrid: true,
        pagedGridMaxPages: 32,
        polygonSurface: true,
        warmupFrames: 4,
        onPagedGridPages: () => resolvePages(),
        onPagedGridOverflow: () => resolveOverflow(),
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
    });
    const forceBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(forceBuffer, 0, new Float32Array([100, 50, -25, 0]));
    sim.setForceField({
        struct: "struct ForceFieldParams { value: vec4<f32>, };",
        wgsl: "fn externalForce(position: vec3<f32>, velocity: vec3<f32>, dt: f32) -> vec3<f32> { return forceFieldParams.value.xyz * dt; }",
        buffer: forceBuffer,
    });
    sim.setFoam({ activeParticles: true, poolScale: 1, poolCapMax: 1024, kTa: 10000, kWc: 10000 });
    const diffuse = sim.diffuse;
    if (!diffuse?.activeIndices || !diffuse.drawIndirect) throw new Error("Active diffuse pool was not created.");
    device.queue.writeBuffer(diffuse.buffer, 0, new Float32Array([
        0.2, 0.2, 0.2, 3, 1, 2, 3, 1,
        0.4, 0.4, 0.4, 2, 4, 5, 6, 0,
    ]));
    const activeHeader = new Uint32Array(66);
    activeHeader.set([2, 2, 0, 0]);
    activeHeader[64] = 0;
    activeHeader[65] = 1;
    device.queue.writeBuffer(diffuse.headBuffer, 0, activeHeader);
    device.queue.writeBuffer(diffuse.drawIndirect, 0, new Uint32Array([6, 2, 0, 0]));
    const lifecycle = captured.get("fluid-particle-lifecycle");
    const foamDispatch = captured.get("flip-foam-active-dispatch");
    if (!lifecycle || !foamDispatch) throw new Error("Atomic state buffers were not captured.");
    const primaryRanges = [
        { buffer: sim.positionBuffer, size: sim.positionBuffer.size },
        { buffer: sim.velocityBuffer, size: sim.velocityBuffer.size },
        { buffer: sim.debugBuffer, size: sim.debugBuffer.size },
        { buffer: lifecycle, size: lifecycle.size },
    ];
    const diffuseRanges = [
        { buffer: diffuse.buffer, size: diffuse.buffer.size },
        { buffer: diffuse.headBuffer, size: diffuse.headBuffer.size },
        { buffer: diffuse.drawIndirect, size: diffuse.drawIndirect.size },
    ];
    const successful = device.createCommandEncoder();
    sim.step(successful, 1 / 60);
    device.queue.submit([successful.finish()]);
    await device.queue.onSubmittedWorkDone();
    sim.step(device.createCommandEncoder(), 0);
    await pagesReported;
    await device.queue.onSubmittedWorkDone();
    const surface = sim.polygonSurface;
    const reconstructedSdf = captured.get("flip-polygon-surface-reconstructed-sdf");
    if (!surface?.wireframeIndexBuffer || !surface.wireframeDrawIndirect || !reconstructedSdf) {
        throw new Error("Paged polygon buffers were not created.");
    }
    for (let attempt = 0; attempt < 8 && surface.triangleCount === undefined; attempt++) {
        const poll = device.createCommandEncoder();
        sim.step(poll, 0);
        device.queue.submit([poll.finish()]);
        await device.queue.onSubmittedWorkDone();
        await Promise.resolve();
    }
    const polygonRanges = [
        { buffer: reconstructedSdf, size: reconstructedSdf.size },
        { buffer: surface.liquidSdfBuffer, size: surface.liquidSdfBuffer.size },
        { buffer: surface.vertexBuffer, size: surface.vertexBuffer.size },
        { buffer: surface.indexBuffer, size: surface.indexBuffer.size },
        { buffer: surface.drawIndirect, size: surface.drawIndirect.size },
        { buffer: surface.wireframeIndexBuffer, size: surface.wireframeIndexBuffer.size },
        { buffer: surface.wireframeDrawIndirect, size: surface.wireframeDrawIndirect.size },
    ];
    sim.setFlow({
        emitters: [{
            id: "overflow-inflow",
            name: "Overflow inflow",
            enabled: true,
            behavior: "inflow",
            transform: { position: [1.6, 1.6, 1.6], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [3, 3, 3] },
            sampling: "volume",
            velocity: [3, 0, 0],
            velocitySpace: "world",
            spread: 0,
            volumeRate: 1,
        }],
        sinks: [],
    });
    const beforePrimary = await bytes(device, primaryRanges);
    const beforeDiffuse = await bytes(device, diffuseRanges);
    const beforePolygon = await bytes(device, polygonRanges);
    const beforeHistory = await bytes(device, [{ buffer: surface.liquidSdfBuffer, size: surface.liquidSdfBuffer.size }]);
    const beforeTriangleCount = surface.triangleCount;
    const beforeMetadata = {
        activeCount: sim.activeCount,
        renderCount: sim.renderCount,
        activeIndicesOffset: sim.diffuse.activeIndicesOffset,
    };
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    sim.step(device.createCommandEncoder(), 0);
    await overflowed;
    await device.queue.onSubmittedWorkDone();
    const afterPrimary = await bytes(device, primaryRanges);
    const afterDiffuse = await bytes(device, diffuseRanges);
    const afterPolygon = await bytes(device, polygonRanges);
    const afterFoamDispatch = await bytes(device, [{ buffer: foamDispatch, size: foamDispatch.size }]);
    const afterMetadata = {
        activeCount: sim.activeCount,
        renderCount: sim.renderCount,
        activeIndicesOffset: sim.diffuse.activeIndicesOffset,
    };
    const recovery = createFlipSim(simEngine, {
        count: 4,
        initialPositions: new Float32Array(0),
        boundsMin: [0, 0, 0],
        boundsMax: [3.2, 3.2, 3.2],
        gridDim: [32, 32, 32],
        dx: 0.1,
        particleRadius: 0.04,
        pagedGrid: true,
        pagedGridMaxPages: 64,
        polygonSurface: true,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
    });
    const transfer = device.createCommandEncoder();
    const transferred = transferFlipSimState(transfer, sim, recovery);
    device.queue.submit([transfer.finish()]);
    await device.queue.onSubmittedWorkDone();
    const recoverySurface = recovery.polygonSurface;
    if (!recoverySurface) throw new Error("Recovery polygon surface was not created.");
    const recoveredHistory = await bytes(device, [{ buffer: recoverySurface.liquidSdfBuffer, size: recoverySurface.liquidSdfBuffer.size }]);
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({
        primaryIdentical: beforePrimary.length === afterPrimary.length && beforePrimary.every((value, index) => value === afterPrimary[index]),
        diffuseIdentical: beforeDiffuse.length === afterDiffuse.length && beforeDiffuse.every((value, index) => value === afterDiffuse[index]),
        polygonIdentical: beforePolygon.length === afterPolygon.length && beforePolygon.every((value, index) => value === afterPolygon[index]),
        triangleCountAvailable: beforeTriangleCount !== undefined,
        triangleCountIdentical: beforeTriangleCount === surface.triangleCount,
        metadataIdentical: JSON.stringify(beforeMetadata) === JSON.stringify(afterMetadata),
        activeDispatchCleared: new Uint32Array(afterFoamDispatch.buffer)[0] === 0,
        recoveryHistoryIdentical:
            transferred && beforeHistory.length === recoveredHistory.length && beforeHistory.every((value, index) => value === recoveredHistory[index]),
    });
    forceBuffer.destroy();
    recovery.dispose();
    sim.dispose();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect.poll(async () => (await canvas.getAttribute("data-result")) ?? (await canvas.getAttribute("data-error")), { timeout: 90_000 }).toBeTruthy();
    expect(await canvas.getAttribute("data-error")).toBeNull();
    expect(JSON.parse((await canvas.getAttribute("data-result"))!)).toEqual({
        primaryIdentical: true,
        diffuseIdentical: true,
        polygonIdentical: true,
        triangleCountAvailable: true,
        triangleCountIdentical: true,
        metadataIdentical: true,
        activeDispatchCleared: true,
        recoveryHistoryIdentical: true,
    });
});

test("whole-frame preflight keeps a later-substep page crossing fail-atomic", async ({ page }) => {
    test.setTimeout(120_000);
    page.on("console", (message) => console.log(message.type(), message.text()));
    page.on("pageerror", (error) => console.log("pageerror", error.message));
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";
const canvas = document.getElementById("renderCanvas");
async function bytes(device, ranges) {
    const size = ranges.reduce((sum, range) => sum + range.size, 0);
    const readback = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    let offset = 0;
    for (const range of ranges) {
        encoder.copyBufferToBuffer(range.buffer, 0, readback, offset, range.size);
        offset += range.size;
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    return result;
}
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    let lifecycle;
    const simEngine = {
        _device: new Proxy(device, {
            get(target, prop) {
                if (prop === "createBuffer") {
                    return descriptor => {
                        const capture = descriptor.label === "fluid-particle-lifecycle";
                        const buffer = target.createBuffer(capture ? { ...descriptor, usage: descriptor.usage | GPUBufferUsage.COPY_SRC } : descriptor);
                        if (capture) lifecycle = buffer;
                        return buffer;
                    };
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        }),
    };
    device.pushErrorScope("validation");
    let resolveOverflow;
    const overflowed = new Promise(resolve => resolveOverflow = resolve);
    let requiredPages = 0;
    const sim = createFlipSim(simEngine, {
        count: 1,
        initialPositions: new Float32Array([1.55, 0.4, 0.4]),
        boundsMin: [0, 0, 0],
        boundsMax: [4, 0.8, 0.8],
        gridDim: [40, 8, 8],
        dx: 0.1,
        particleRadius: 0.04,
        gravity: 0,
        pressureIterations: 1,
        minSubsteps: 2,
        maxSubsteps: 2,
        maxSubDt: 0.05,
        pagedGrid: true,
        pagedGridMaxPages: 3,
        onPagedGridPages: required => requiredPages = required,
        onPagedGridOverflow: () => resolveOverflow(),
    });
    if (!lifecycle) throw new Error("Lifecycle buffer was not captured.");
    device.queue.writeBuffer(sim.velocityBuffer, 0, new Float32Array([5, 0, 0, 0]));
    await device.queue.onSubmittedWorkDone();
    const ranges = [
        { buffer: sim.positionBuffer, size: sim.positionBuffer.size },
        { buffer: sim.velocityBuffer, size: sim.velocityBuffer.size },
        { buffer: sim.debugBuffer, size: sim.debugBuffer.size },
        { buffer: lifecycle, size: lifecycle.size },
    ];
    const before = await bytes(device, ranges);
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 0.1);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    sim.step(device.createCommandEncoder(), 0);
    await overflowed;
    const after = await bytes(device, ranges);
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({
        requiredPages,
        identical: before.length === after.length && before.every((value, index) => value === after[index]),
    });
    sim.dispose();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect.poll(async () => (await canvas.getAttribute("data-result")) ?? (await canvas.getAttribute("data-error")), { timeout: 90_000 }).toBeTruthy();
    expect(await canvas.getAttribute("data-error")).toBeNull();
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as { requiredPages: number; identical: boolean };
    expect(result.requiredPages).toBeGreaterThan(3);
    expect(result.identical).toBe(true);
});

test("paged status mapping does not skip consecutive simulation frames", async ({ page }) => {
    test.setTimeout(120_000);
    page.on("console", (message) => console.log(message.type(), message.text()));
    page.on("pageerror", (error) => console.log("pageerror", error.message));
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";
const canvas = document.getElementById("renderCanvas");
async function readVelocityX(device, buffer) {
    const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, 16);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const value = new Float32Array(readback.getMappedRange().slice(0))[0];
    readback.unmap();
    readback.destroy();
    return value;
}
async function run(engine, paged) {
    const device = engine._device;
    let paramsBuffer;
    const subDts = [];
    const queueProxy = new Proxy(device.queue, {
        get(target, prop) {
            if (prop === "writeBuffer") {
                return (...args) => {
                    const [buffer, bufferOffset, data] = args;
                    if (buffer === paramsBuffer && bufferOffset === 0) {
                        subDts.push(new Float32Array(data)[16]);
                    }
                    return target.writeBuffer(...args);
                };
            }
            const value = Reflect.get(target, prop, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    const simEngine = {
        _device: new Proxy(device, {
            get(target, prop) {
                if (prop === "queue") return queueProxy;
                if (prop === "createBuffer") {
                    return descriptor => {
                        const buffer = target.createBuffer(descriptor);
                        if (descriptor.label === "flip-params") paramsBuffer = buffer;
                        return buffer;
                    };
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        }),
    };
    const sim = createFlipSim(simEngine, {
        count: 1,
        initialPositions: new Float32Array([2, 2, 2]),
        boundsMin: [0, 0, 0],
        boundsMax: [4, 4, 4],
        gridDim: [16, 16, 16],
        dx: 0.25,
        particleRadius: 0.08,
        gravity: 0,
        flipRatio: 1,
        pagedGrid: paged,
        pagedGridMaxPages: 8,
        pressureIterations: 1,
        minSubsteps: 1,
        maxSubsteps: 2,
        maxSubDt: 0.01,
    });
    subDts.length = 0;
    const forceBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(forceBuffer, 0, new Float32Array([1, 0, 0, 0]));
    sim.setForceField({
        struct: "struct ForceFieldParams { value: vec4<f32>, };",
        wgsl: "fn externalForce(position: vec3<f32>, velocity: vec3<f32>, dt: f32) -> vec3<f32> { return forceFieldParams.value.xyz * dt; }",
        buffer: forceBuffer,
    });
    const renderDts = new Array(8).fill(0.01);
    const velocitySamples = [];
    for (let call = 0; call < renderDts.length; call++) {
        const encoder = device.createCommandEncoder();
        sim.step(encoder, renderDts[call]);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        velocitySamples.push(await readVelocityX(device, sim.velocityBuffer));
    }
    const velocityX = await readVelocityX(device, sim.velocityBuffer);
    sim.dispose();
    forceBuffer.destroy();
    return { velocityX, elapsed: renderDts.reduce((sum, value) => sum + value, 0), subDts, velocitySamples };
}
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const dense = await run(engine, false);
    const paged = await run(engine, true);
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({ dense, paged });
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect.poll(async () => (await canvas.getAttribute("data-result")) ?? (await canvas.getAttribute("data-error")), { timeout: 90_000 }).toBeTruthy();
    expect(await canvas.getAttribute("data-error")).toBeNull();
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        dense: { velocityX: number; elapsed: number; subDts: number[]; velocitySamples: number[] };
        paged: { velocityX: number; elapsed: number; subDts: number[]; velocitySamples: number[] };
    };
    expect(result.paged.elapsed).toBeCloseTo(result.dense.elapsed, 8);
    expect(Math.max(...result.paged.subDts)).toBeLessThanOrEqual(0.010001);
    expect(result.paged.subDts).toHaveLength(8);
    for (let index = 1; index < result.paged.velocitySamples.length; index++) {
        expect(result.paged.velocitySamples[index]!).toBeGreaterThan(result.paged.velocitySamples[index - 1]! + 0.001);
    }
    expect(result.dense.velocityX).toBeGreaterThan(0.03);
    expect(result.paged.velocityX / result.dense.velocityX).toBeGreaterThan(0.9);
    expect(result.paged.velocityX / result.dense.velocityX).toBeLessThan(1.1);
});

test("paged emitter discovery covers squat capsules without allocating a rotated cylinder AABB", async ({ page }) => {
    test.setTimeout(120_000);
    page.on("console", (message) => console.log(message.type(), message.text()));
    page.on("pageerror", (error) => console.log("pageerror", error.message));
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";
const canvas = document.getElementById("renderCanvas");
async function discover(engine, options, emitter) {
    let status;
    let overflow = false;
    const sim = createFlipSim(engine, {
        ...options,
        count: 1,
        initialPositions: new Float32Array(0),
        particleRadius: 0.04,
        pagedGrid: true,
        pressureIterations: 1,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
        onPagedGridPages: (required, capacity) => status = { required, capacity },
        onPagedGridOverflow: () => overflow = true,
    });
    sim.setFlow({ emitters: [{ ...emitter, volumeRate: 1 }], sinks: [] });
    sim.reset();
    for (let frame = 0; frame < 4 && !status; frame++) {
        const encoder = engine._device.createCommandEncoder();
        sim.step(encoder, 1 / 60);
        engine._device.queue.submit([encoder.finish()]);
        await engine._device.queue.onSubmittedWorkDone();
        await Promise.resolve();
    }
    if (!status) throw new Error("Page discovery did not report.");
    sim.dispose();
    return { ...status, overflow };
}
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const identity = [0, 0, 0, 1];
    const commonEmitter = {
        id: "source", name: "Source", enabled: true, behavior: "inflow", sampling: "volume",
        velocity: [0, 0, 0], velocitySpace: "world", spread: 0,
    };
    const squat = await discover(engine, {
        boundsMin: [0, 0, 0], boundsMax: [3.2, 3.2, 3.2], gridDim: [32, 32, 32], dx: 0.1, pagedGridMaxPages: 64,
    }, {
        ...commonEmitter,
        transform: { position: [1.2, 1.2, 1.2], rotation: identity, scale: [1, 1, 1] },
        shape: { type: "capsule", radius: 0.7, height: 0.2 },
    });
    const angle = Math.PI / 8;
    const rotated = await discover(engine, {
        boundsMin: [0, 0, 0], boundsMax: [12.8, 12.8, 12.8], gridDim: [128, 128, 128], dx: 0.1, pagedGridMaxPages: 450,
    }, {
        ...commonEmitter,
        transform: { position: [6.4, 6.4, 6.4], rotation: [0, 0, Math.sin(angle), Math.cos(angle)], scale: [1, 1, 1] },
        shape: { type: "cylinder", radius: 0.1, height: 12 },
    });
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({ squat, rotated });
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect.poll(async () => (await canvas.getAttribute("data-result")) ?? (await canvas.getAttribute("data-error")), { timeout: 90_000 }).toBeTruthy();
    expect(await canvas.getAttribute("data-error")).toBeNull();
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        squat: { required: number; overflow: boolean };
        rotated: { required: number; capacity: number; overflow: boolean };
    };
    expect(result.squat.required).toBeGreaterThan(9);
    expect(result.squat.overflow).toBe(false);
    expect(result.rotated.required).toBeLessThanOrEqual(result.rotated.capacity);
    expect(result.rotated.overflow).toBe(false);
});

test("FLIP transfer preserves diffuse slots, active heads, and draw state across capacity growth", async ({ page }) => {
    test.setTimeout(120_000);
    page.on("console", (message) => console.log(message.type(), message.text()));
    page.on("pageerror", (error) => console.log("pageerror", error.message));
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim, transferFlipSimState } from "${FLIP_ENTRY}";
const canvas = document.getElementById("renderCanvas");
const stride = capacity => Math.ceil(capacity / 64) * 64;
const listOffset = (capacity, side) => (64 + side * stride(capacity)) * 4;
const flagsOffset = capacity => (64 + 2 * stride(capacity)) * 4;
async function read(device, buffer, offset, size) {
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offset, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return result;
}
async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const options = count => ({
        count, initialPositions: new Float32Array(0),
        boundsMin: [0, 0, 0], boundsMax: [1, 1, 1], gridDim: [8, 8, 8], dx: 0.125,
        particleRadius: 0.04, minSubsteps: 1, maxSubsteps: 1, maxSubDt: 1,
    });
    const source = createFlipSim(engine, options(1024));
    const target = createFlipSim(engine, options(2048));
    source.setFoam({ activeParticles: true, poolScale: 1, poolCapMax: 4096 });
    target.setFoam({ activeParticles: true, poolScale: 1, poolCapMax: 4096 });
    const from = source.diffuse;
    const to = target.diffuse;
    if (!from?.activeIndices || !from.drawIndirect || !to?.activeIndices || !to.drawIndirect) throw new Error("Active pools were not created.");
    const slots = new Float32Array(from.capacity * 8);
    slots.set([1, 2, 3, 4, 5, 6, 7, 1], 3 * 8);
    slots.set([8, 9, 10, 11, 12, 13, 14, 2], 7 * 8);
    device.queue.writeBuffer(from.buffer, 0, slots);
    device.queue.writeBuffer(from.headBuffer, 0, new Uint32Array([17, 2, 1, 0]));
    device.queue.writeBuffer(from.activeIndices, listOffset(from.capacity, 0), new Uint32Array([3, 7]));
    device.queue.writeBuffer(from.activeIndices, listOffset(from.capacity, 1), new Uint32Array([7]));
    device.queue.writeBuffer(from.activeIndices, flagsOffset(from.capacity) + 3 * 4, new Uint32Array([1, 0, 0, 0, 1]));
    device.queue.writeBuffer(from.drawIndirect, 0, new Uint32Array([6, 2, 0, 0]));
    const encoder = device.createCommandEncoder();
    const transferred = transferFlipSimState(encoder, source, target);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const [sourceSlots, targetSlots, header, list0, list1, flags, draw] = await Promise.all([
        read(device, from.buffer, 0, from.buffer.size),
        read(device, to.buffer, 0, from.buffer.size),
        read(device, to.headBuffer, 0, 16),
        read(device, to.activeIndices, listOffset(to.capacity, 0), 8),
        read(device, to.activeIndices, listOffset(to.capacity, 1), 4),
        read(device, to.activeIndices, flagsOffset(to.capacity) + 3 * 4, 20),
        read(device, to.drawIndirect, 0, 16),
    ]);
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;
    canvas.dataset.result = JSON.stringify({
        transferred,
        slotsIdentical: sourceSlots.every((value, index) => value === targetSlots[index]),
        header: Array.from(new Uint32Array(header.buffer)),
        list0: Array.from(new Uint32Array(list0.buffer)),
        list1: Array.from(new Uint32Array(list1.buffer)),
        flags: Array.from(new Uint32Array(flags.buffer)),
        draw: Array.from(new Uint32Array(draw.buffer)),
        activeOffset: to.activeIndicesOffset,
    });
    source.dispose();
    target.dispose();
}
main().catch(error => canvas.dataset.error = error?.message ?? String(error));
</script>`);
    const canvas = page.locator("#renderCanvas");
    await expect.poll(async () => (await canvas.getAttribute("data-result")) ?? (await canvas.getAttribute("data-error")), { timeout: 90_000 }).toBeTruthy();
    expect(await canvas.getAttribute("data-error")).toBeNull();
    const result = JSON.parse((await canvas.getAttribute("data-result"))!);
    expect(result).toMatchObject({
        transferred: true,
        slotsIdentical: true,
        header: [17, 2, 1, 0],
        list0: [3, 7],
        list1: [7],
        flags: [1, 0, 0, 0, 1],
        draw: [6, 2, 0, 0],
        activeOffset: 256,
    });
});
