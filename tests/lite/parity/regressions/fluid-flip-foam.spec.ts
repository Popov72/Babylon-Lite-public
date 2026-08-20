import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const FLIP_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/flip-sim.ts").replace(/\\/g, "/")}`;

test("FLIP whitewater is opt-in, generates diffuse particles, and clears on reset", async ({ page }) => {
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

async function readDiffuse(device, pool) {
    const bytes = pool.capacity * 32;
    const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(pool.buffer, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange());
    let active = 0;
    let lowActive = 0;
    const kinds = [0, 0, 0];
    for (let index = 0; index < pool.capacity; index++) {
        if (values[index * 8 + 3] > 0) {
            active++;
            if (values[index * 8 + 1] < 3) {
                lowActive++;
            }
            const kind = Math.round(values[index * 8 + 7]);
            if (kind >= 0 && kind <= 2) {
                kinds[kind]++;
            }
        }
    }
    const slotKinds = [Math.round(values[7]), Math.round(values[15]), Math.round(values[23]), Math.round(values[31])];
    const slotLives = [values[3], values[11], values[19], values[27]];
    readback.unmap();
    readback.destroy();
    return { active, lowActive, kinds, slotKinds, slotLives };
}

async function step(engine, sim) {
    const encoder = engine._device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    engine._device.queue.submit([encoder.finish()]);
    await engine._device.queue.onSubmittedWorkDone();
}

async function main() {
    const engine = await createEngine(canvas);
    engine._device.pushErrorScope("validation");
    const positions = new Float32Array([
        0.25, 0.75, 0.25, 0.75, 0.75, 0.25,
        0.25, 1.25, 0.25, 0.75, 1.25, 0.25,
        0.25, 0.75, 0.75, 0.75, 0.75, 0.75,
        0.25, 1.25, 0.75, 0.75, 1.25, 0.75,
    ]);
    const sim = createFlipSim(engine, {
        count: 8,
        initialPositions: positions,
        boundsMin: [-1, 0, -1],
        boundsMax: [2, 3, 2],
        gridDim: [6, 6, 6],
        dx: 0.5,
        markersPerCell: 1,
        particleRadius: 0.08,
        gravity: 0,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
    });
    const disabledDiffuse = sim.diffuse;
    let disabledFoamPasses = 0;
    sim.setProfiler({ pass: (stage) => { if (stage === "Foam gen") disabledFoamPasses++; return undefined; } });
    await step(engine, sim);
    const initialDisabledFoamPasses = disabledFoamPasses;

    sim.setFoam({ kTa: 10000, kWc: 10000, poolScale: 128, poolCapMax: 1024 });
    await step(engine, sim);
    const calm = await readDiffuse(engine._device, sim.diffuse);
    const velocities = new Float32Array(sim.count * 4);
    for (let index = 0; index < sim.count; index++) {
        velocities[index * 4 + 1] = -10;
    }
    engine._device.queue.writeBuffer(sim.velocityBuffer, 0, velocities);
    await step(engine, sim);
    const generated = await readDiffuse(engine._device, sim.diffuse);

    sim.reset();
    await engine._device.queue.onSubmittedWorkDone();
    const afterReset = await readDiffuse(engine._device, sim.diffuse);
    sim.setFoam(null);
    const foamPassesBeforeDisabledStep = disabledFoamPasses;
    await step(engine, sim);
    const afterDisable = await readDiffuse(engine._device, sim.diffuse);
    const disabledStepFoamPasses = disabledFoamPasses - foamPassesBeforeDisabledStep;

    const activeSim = createFlipSim(engine, {
        count: 8,
        initialPositions: positions,
        boundsMin: [-1, 0, -1],
        boundsMax: [2, 3, 2],
        gridDim: [6, 6, 6],
        dx: 0.5,
        markersPerCell: 1,
        particleRadius: 0.08,
        gravity: 0,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
    });
    engine._device.queue.writeBuffer(activeSim.velocityBuffer, 0, velocities);
    activeSim.setFoam({ activeParticles: true, kTa: 10000, kWc: 10000, poolScale: 128, poolCapMax: 1024 });
    await step(engine, activeSim);
    const activeGenerated = await readDiffuse(engine._device, activeSim.diffuse);
    const hasCompactPool =
        activeSim.diffuse.activeIndices !== undefined &&
        activeSim.diffuse.activeIndicesOffset !== undefined &&
        activeSim.diffuse.drawIndirect !== undefined;

    const filledPositions = new Float32Array(27 * 3);
    let filledIndex = 0;
    for (let z = 1; z <= 3; z++) {
        for (let y = 1; y <= 3; y++) {
            for (let x = 1; x <= 3; x++) {
                filledPositions[filledIndex++] = x + 0.5;
                filledPositions[filledIndex++] = y + 0.5;
                filledPositions[filledIndex++] = z + 0.5;
            }
        }
    }
    const classificationSim = createFlipSim(engine, {
        count: 27,
        initialPositions: filledPositions,
        boundsMin: [0, 0, 0],
        boundsMax: [5, 5, 5],
        gridDim: [5, 5, 5],
        dx: 1,
        markersPerCell: 1,
        particleRadius: 0.08,
        gravity: 0,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
    });
    classificationSim.setFoam({ kTa: 0, kWc: 0, poolScale: 38, poolCapMax: 1024 });
    const injectedDiffuse = new Float32Array(32);
    injectedDiffuse.set([2.5, 2.5, 2.5, 1, 0, 0, 0, 1], 0);
    injectedDiffuse.set([2.5, 3.5, 2.5, 1, 0, 0, 0, 1], 8);
    injectedDiffuse.set([3.5, 2.5, 2.5, 1, 0, 0, 0, 1], 16);
    injectedDiffuse.set([2.5, 4.1, 2.5, 1, 0, 0, 0, 1], 24);
    engine._device.queue.writeBuffer(classificationSim.diffuse.buffer, 0, injectedDiffuse);
    await step(engine, classificationSim);
    const classified = await readDiffuse(engine._device, classificationSim.diffuse);

    const sideEmissionSim = createFlipSim(engine, {
        count: 27,
        initialPositions: filledPositions,
        boundsMin: [0, 0, 0],
        boundsMax: [5, 5, 5],
        gridDim: [5, 5, 5],
        dx: 1,
        markersPerCell: 1,
        particleRadius: 0.08,
        gravity: 0,
        pressureIterations: 2,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
    });
    const sideVelocities = new Float32Array(27 * 4);
    for (let index = 0; index < 27; index++) {
        sideVelocities[index * 4] = 10;
    }
    engine._device.queue.writeBuffer(sideEmissionSim.velocityBuffer, 0, sideVelocities);
    sideEmissionSim.setFoam({ kTa: 10000, kWc: 10000, poolScale: 38, poolCapMax: 1024 });
    await step(engine, sideEmissionSim);
    const sideGenerated = await readDiffuse(engine._device, sideEmissionSim.diffuse);

    const validationError = await engine._device.popErrorScope();
    sim.dispose();
    activeSim.dispose();
    classificationSim.dispose();
    sideEmissionSim.dispose();
    if (validationError) {
        throw new Error(validationError.message);
    }
    canvas.dataset.result = JSON.stringify({
        disabledDiffuse: disabledDiffuse === undefined,
        initialDisabledFoamPasses,
        calm,
        generated,
        afterReset,
        afterDisable,
        disabledStepFoamPasses,
        activeGenerated,
        hasCompactPool,
        classified,
        sideGenerated,
    });
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await page.waitForFunction(() => {
        const element = document.querySelector("#renderCanvas");
        return element?.hasAttribute("data-result") || element?.hasAttribute("data-error");
    });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 90_000 });
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        disabledDiffuse: boolean;
        initialDisabledFoamPasses: number;
        calm: { active: number };
        generated: { active: number; kinds: number[] };
        afterReset: { active: number };
        afterDisable: { active: number };
        disabledStepFoamPasses: number;
        activeGenerated: { active: number };
        hasCompactPool: boolean;
        classified: { slotKinds: number[]; slotLives: number[] };
        sideGenerated: { lowActive: number };
    };
    expect(result.disabledDiffuse).toBe(true);
    expect(result.initialDisabledFoamPasses).toBe(0);
    expect(result.calm.active).toBe(0);
    expect(result.generated.active).toBeGreaterThan(0);
    expect(result.generated.kinds.reduce((sum, count) => sum + count, 0)).toBe(result.generated.active);
    expect(result.afterReset.active).toBe(0);
    expect(result.afterDisable.active).toBe(0);
    expect(result.disabledStepFoamPasses).toBe(0);
    expect(result.activeGenerated.active).toBeGreaterThan(0);
    expect(result.hasCompactPool).toBe(true);
    expect(result.classified.slotLives[0]).toBeGreaterThan(0);
    expect(result.classified.slotLives[1]).toBeGreaterThan(0);
    expect(result.classified.slotLives[2]).toBeGreaterThan(0);
    expect(result.classified.slotLives[3]).toBeGreaterThan(0);
    expect(result.classified.slotKinds[0]).toBe(2);
    expect(result.classified.slotKinds[1]).toBe(1);
    expect(result.classified.slotKinds[2]).toBe(2);
    expect(result.classified.slotKinds[3]).toBe(1);
    expect(result.sideGenerated.lowActive).toBe(0);
});
