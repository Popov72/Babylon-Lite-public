import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const FLIP_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/flip-sim.ts").replace(/\\/g, "/")}`;

test("FLIP inflows refill empty source cells with zero velocity and respect a volume cap", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

const emitter = {
    id: "source",
    name: "Source",
    enabled: true,
    behavior: "inflow",
    transform: { position: [0, 1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    shape: { type: "box", size: [1, 1, 1] },
    sampling: "volume",
    velocity: [0, 0, 0],
    velocitySpace: "world",
    spread: 0,
};
const inactiveSink = {
    id: "sink",
    name: "Sink",
    enabled: true,
    mode: "delete",
    transform: { position: [0.8, 0.2, 0.8], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    shape: { type: "box", size: [0.1, 0.1, 0.1] },
    targets: [],
};

function createSim(engine, count = 512, initialPositions = new Float32Array(0)) {
    return createFlipSim(engine, {
        count,
        initialPositions,
        boundsMin: [-1, 0, -1],
        boundsMax: [1, 2, 1],
        gridDim: [4, 4, 4],
        dx: 0.5,
        markersPerCell: 8,
        particleRadius: 0.1,
        gravity: 0,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
    });
}

async function stepAndCount(engine, sim, dt) {
    const encoder = engine._device.createCommandEncoder();
    sim.step(encoder, dt);
    engine._device.queue.submit([encoder.finish()]);
    await engine._device.queue.onSubmittedWorkDone();
    const bytes = sim.count * 16;
    const readback = engine._device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = engine._device.createCommandEncoder();
    copy.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, bytes);
    engine._device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const positions = new Float32Array(readback.getMappedRange());
    let active = 0;
    for (let index = 0; index < sim.count; index++) {
        if (positions[index * 4 + 1] > -1000) {
            active++;
        }
    }
    readback.unmap();
    readback.destroy();
    return active;
}

async function simulationPassCount(engine, withEmitter) {
    const positions = new Float32Array(64 * 3);
    for (let index = 0; index < 64; index++) {
        positions[index * 3] = -0.45 + (index % 4) * 0.3;
        positions[index * 3 + 1] = 0.55 + (Math.floor(index / 4) % 4) * 0.3;
        positions[index * 3 + 2] = -0.45 + Math.floor(index / 16) * 0.3;
    }
    const sim = createSim(engine, 64, positions);
    sim.setFlow({ emitters: withEmitter ? [emitter] : [], sinks: [] });
    sim.reset();
    let passes = 0;
    sim.setProfiler({ pass: () => { passes++; return undefined; } });
    const encoder = engine._device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    engine._device.queue.submit([encoder.finish()]);
    await engine._device.queue.onSubmittedWorkDone();
    sim.dispose();
    return passes;
}

async function main() {
    const engine = await createEngine(canvas);
    engine._device.pushErrorScope("validation");

    const unlimited = createSim(engine);
    unlimited.setFlow({ emitters: [emitter], sinks: [] });
    unlimited.reset();
    const firstOccupancyCount = await stepAndCount(engine, unlimited, 1);
    let settledOccupancyCount = firstOccupancyCount;
    for (let frame = 0; frame < 4; frame++) {
        settledOccupancyCount = await stepAndCount(engine, unlimited, 1 / 60);
    }
    unlimited.dispose();

    const capped = createSim(engine);
    capped.setFlow({ emitters: [{ ...emitter, volumeRate: 0.15625 }], sinks: [] });
    capped.reset();
    const cappedCount = await stepAndCount(engine, capped, 1);
    capped.dispose();

    const lifecycleScan = createSim(engine);
    lifecycleScan.setFlow({ emitters: [emitter], sinks: [inactiveSink] });
    lifecycleScan.reset();
    const lifecycleScanCount = await stepAndCount(engine, lifecycleScan, 1);
    lifecycleScan.dispose();

    const saturatedEnabledPasses = await simulationPassCount(engine, true);
    const saturatedDisabledPasses = await simulationPassCount(engine, false);

    const validationError = await engine._device.popErrorScope();
    if (validationError) {
        throw new Error(validationError.message);
    }
    canvas.dataset.result = JSON.stringify({
        firstOccupancyCount,
        settledOccupancyCount,
        cappedCount,
        lifecycleScanCount,
        saturatedEnabledPasses,
        saturatedDisabledPasses,
    });
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 90_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        firstOccupancyCount: number;
        settledOccupancyCount: number;
        cappedCount: number;
        lifecycleScanCount: number;
        saturatedEnabledPasses: number;
        saturatedDisabledPasses: number;
    };
    expect(result.firstOccupancyCount).toBeGreaterThan(0);
    expect(result.firstOccupancyCount).toBeLessThanOrEqual(64);
    expect(result.settledOccupancyCount).toBeLessThanOrEqual(64);
    expect(result.cappedCount).toBeGreaterThan(0);
    expect(result.cappedCount).toBeLessThanOrEqual(10);
    expect(result.lifecycleScanCount).toBeGreaterThan(0);
    expect(result.lifecycleScanCount).toBeLessThanOrEqual(64);
    expect(result.saturatedEnabledPasses).toBe(result.saturatedDisabledPasses);
});
