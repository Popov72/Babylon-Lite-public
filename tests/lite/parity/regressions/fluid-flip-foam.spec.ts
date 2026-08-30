import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const FLIP_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/flip-sim.ts").replace(/\\/g, "/")}`;
const COMMON_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/sim-common.ts").replace(/\\/g, "/")}`;

test("FLIP whitewater is opt-in, generates diffuse particles, and clears on reset", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";
import { createDiffuseCountTracker, foamActiveStateBytes } from "${COMMON_ENTRY}";

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
    const slotVelocities = [
        [values[4], values[5], values[6]],
        [values[12], values[13], values[14]],
        [values[20], values[21], values[22]],
        [values[28], values[29], values[30]],
    ];
    readback.unmap();
    readback.destroy();
    return { active, lowActive, kinds, slotKinds, slotLives, slotVelocities };
}

async function readBufferWords(device, buffer) {
    const readback = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    return words;
}

async function readBufferFloats(device, buffer) {
    const readback = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    return values;
}

async function step(engine, sim) {
    const encoder = engine._device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    engine._device.queue.submit([encoder.finish()]);
    await engine._device.queue.onSubmittedWorkDone();
}

async function sampleCounts(device, tracker, activeDispatch) {
    const encoder = device.createCommandEncoder();
    for (let frame = 0; frame < 30; frame++) {
        tracker.encode(encoder, 1, activeDispatch);
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const pollEncoder = device.createCommandEncoder();
    tracker.encode(pollEncoder, 1, activeDispatch);
    device.queue.submit([pollEncoder.finish()]);
    for (let attempt = 0; attempt < 100 && !tracker.counts; attempt++) {
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    if (!tracker.counts) {
        throw new Error("Diffuse count readback did not complete");
    }
    return tracker.counts;
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
        liquidSdf: true,
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
        liquidSdf: true,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1,
    });
    classificationSim.setFoam({ activeParticles: false, kTa: 0, kWc: 0, poolScale: 38, poolCapMax: 1024 });
    const injectedDiffuse = new Float32Array(32);
    injectedDiffuse.set([2.5, 2.5, 2.5, 1, 0, 0, 0, 1], 0);
    injectedDiffuse.set([2.5, 3.5, 2.5, 1, 0, 0, 0, 1], 8);
    injectedDiffuse.set([3.5, 2.5, 2.5, 1, 0, 0, 0, 1], 16);
    injectedDiffuse.set([2.5, 4.1, 2.5, 1, 0, 0, 0, 1], 24);
    engine._device.queue.writeBuffer(classificationSim.diffuse.buffer, 0, injectedDiffuse);
    await step(engine, classificationSim);
    const classified = await readDiffuse(engine._device, classificationSim.diffuse);
    classificationSim.setFoam({
        activeParticles: false,
        generateSpray: false,
        generateFoam: true,
        generateBubbles: false,
        kTa: 0,
        kWc: 0,
        poolScale: 38,
        poolCapMax: 1024,
    });
    engine._device.queue.writeBuffer(classificationSim.diffuse.buffer, 0, injectedDiffuse);
    await step(engine, classificationSim);
    const foamOnly = await readDiffuse(engine._device, classificationSim.diffuse);
    classificationSim.setFoam({
        activeParticles: false,
        generateSpray: false,
        generateFoam: false,
        generateBubbles: false,
        kTa: 0,
        kWc: 0,
        poolScale: 38,
        poolCapMax: 1024,
    });
    await step(engine, classificationSim);
    const allKindsDisabled = await readDiffuse(engine._device, classificationSim.diffuse);

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

    const turbulenceSim = createFlipSim(engine, {
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
    const swirlVelocities = new Float32Array(27 * 4);
    for (let index = 0; index < 27; index++) {
        const x = filledPositions[index * 3];
        const z = filledPositions[index * 3 + 2];
        swirlVelocities[index * 4] = -(z - 2.5) * 8;
        swirlVelocities[index * 4 + 2] = (x - 2.5) * 8;
    }
    engine._device.queue.writeBuffer(turbulenceSim.velocityBuffer, 0, swirlVelocities);
    turbulenceSim.setFoam({
        kTa: 0,
        kWc: 0,
        kTurb: 10000,
        energySpeedMin: 0,
        energySpeedMax: 0.1,
        turbulenceMin: 0,
        turbulenceMax: 0.1,
        poolScale: 38,
        poolCapMax: 1024,
    });
    await step(engine, turbulenceSim);
    const turbulenceGenerated = await readDiffuse(engine._device, turbulenceSim.diffuse);

    const advancedUpdateSim = createFlipSim(engine, {
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
    advancedUpdateSim.setFoam({ activeParticles: false, kTa: 0, kWc: 0, foamLayerDepth: 2, sprayDrag: 6, poolScale: 38, poolCapMax: 1024 });
    const advancedDiffuse = new Float32Array(16);
    advancedDiffuse.set([2.5, 2.5, 2.5, 1, 0, 0, 0, 2], 0);
    advancedDiffuse.set([2.5, 4.5, 2.5, 1, 10, 0, 0, 0], 8);
    engine._device.queue.writeBuffer(advancedUpdateSim.diffuse.buffer, 0, advancedDiffuse);
    await step(engine, advancedUpdateSim);
    const advancedUpdated = await readDiffuse(engine._device, advancedUpdateSim.diffuse);

    const invariantOptions = {
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
        surfaceMaxTriangles: 4096,
    };
    const invariantBaselineSim = createFlipSim(engine, invariantOptions);
    const invariantPolygonSim = createFlipSim(engine, invariantOptions);
    engine._device.queue.writeBuffer(invariantBaselineSim.velocityBuffer, 0, sideVelocities);
    engine._device.queue.writeBuffer(invariantPolygonSim.velocityBuffer, 0, sideVelocities);
    const invariantFoam = { activeParticles: false, kTa: 10000, kWc: 10000, poolScale: 38, poolCapMax: 1024 };
    invariantBaselineSim.setFoam(invariantFoam);
    invariantPolygonSim.setFoam(invariantFoam);
    invariantPolygonSim.setParam("polygonSurface", 1);
    invariantPolygonSim.setParam("polygonReconstructionMultiplier", 2);
    const baselineProfileStages = [];
    const polygonProfileStages = [];
    invariantBaselineSim.setProfiler({ pass: (stage) => { baselineProfileStages.push(stage); return undefined; } });
    invariantPolygonSim.setProfiler({ pass: (stage) => { polygonProfileStages.push(stage); return undefined; } });
    await step(engine, invariantBaselineSim);
    await step(engine, invariantPolygonSim);
    await step(engine, invariantBaselineSim);
    await step(engine, invariantPolygonSim);
    const baselineDiffuseWords = await readBufferWords(engine._device, invariantBaselineSim.diffuse.buffer);
    const polygonDiffuseWords = await readBufferWords(engine._device, invariantPolygonSim.diffuse.buffer);
    const polygonWhitewaterInvariant =
        baselineDiffuseWords.length === polygonDiffuseWords.length &&
        baselineDiffuseWords.every((word, index) => word === polygonDiffuseWords[index]);
    const baselineSimulationPasses = baselineProfileStages.filter((stage) => stage === "Simulation").length;
    const polygonSimulationPasses = polygonProfileStages.filter((stage) => stage === "Simulation").length;
    const polygonSurfacePasses = polygonProfileStages.filter((stage) => stage === "Surface").length;
    const polygonGridDimensions = invariantPolygonSim.polygonSurface?.gridDimensions;
    const polygonGridSpacing = invariantPolygonSim.polygonSurface?.gridSpacing;
    const polygonReconstructionMultiplier = invariantPolygonSim.polygonSurface?.reconstructionMultiplier;
    for (let attempt = 0; attempt < 100 && invariantPolygonSim.polygonSurface?.triangleCount === undefined; attempt++) {
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const polygonIndexCount = (invariantPolygonSim.polygonSurface?.triangleCount ?? 0) * 3;

    const coarsePolygonSim = createFlipSim(engine, invariantOptions);
    const finePolygonSim = createFlipSim(engine, invariantOptions);
    const pagedPolygonSim = createFlipSim(engine, {
        ...invariantOptions,
        boundsMax: [12, 8, 8],
        gridDim: [12, 8, 8],
        pagedGrid: true,
        pagedGridMaxPages: 8,
    });
    coarsePolygonSim.setParam("polygonSurface", 1);
    finePolygonSim.setParam("polygonSurface", 1);
    pagedPolygonSim.setParam("polygonSurface", 1);
    finePolygonSim.setParam("polygonReconstructionMultiplier", 2);
    await step(engine, coarsePolygonSim);
    await step(engine, finePolygonSim);
    await step(engine, pagedPolygonSim);
    await step(engine, pagedPolygonSim);
    const coarseSdf = await readBufferFloats(engine._device, coarsePolygonSim.polygonSurface.liquidSdfBuffer);
    const fineSdf = await readBufferFloats(engine._device, finePolygonSim.polygonSurface.liquidSdfBuffer);
    const coarseDim = coarsePolygonSim.polygonSurface.gridDimensions;
    const fineDim = finePolygonSim.polygonSurface.gridDimensions;
    const pagedPolygonSdf = await readBufferFloats(engine._device, pagedPolygonSim.polygonSurface.liquidSdfBuffer);
    for (let attempt = 0; attempt < 100 && pagedPolygonSim.polygonSurface?.triangleCount === undefined; attempt++) {
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const pagedPolygonIndexCount = (pagedPolygonSim.polygonSurface?.triangleCount ?? 0) * 3;
    const pagedPolygonSdfFinite = pagedPolygonSdf.every(Number.isFinite);
    const coarseAt = (x, y, z) => {
        const qx = Math.min(coarseDim[0] - 1, Math.max(0, x));
        const qy = Math.min(coarseDim[1] - 1, Math.max(0, y));
        const qz = Math.min(coarseDim[2] - 1, Math.max(0, z));
        return coarseSdf[qx + coarseDim[0] * (qy + coarseDim[1] * qz)];
    };
    let polygonSdfUpsampleMaxError = 0;
    for (let z = 0; z < fineDim[2]; z++) {
        for (let y = 0; y < fineDim[1]; y++) {
            for (let x = 0; x < fineDim[0]; x++) {
                const sx = (x + 0.5) / 2 - 0.5;
                const sy = (y + 0.5) / 2 - 0.5;
                const sz = (z + 0.5) / 2 - 0.5;
                const bx = Math.floor(sx);
                const by = Math.floor(sy);
                const bz = Math.floor(sz);
                const wx = sx - bx;
                const wy = sy - by;
                const wz = sz - bz;
                const mix = (a, b, weight) => a + (b - a) * weight;
                const z0 = mix(
                    mix(coarseAt(bx, by, bz), coarseAt(bx + 1, by, bz), wx),
                    mix(coarseAt(bx, by + 1, bz), coarseAt(bx + 1, by + 1, bz), wx),
                    wy,
                );
                const z1 = mix(
                    mix(coarseAt(bx, by, bz + 1), coarseAt(bx + 1, by, bz + 1), wx),
                    mix(coarseAt(bx, by + 1, bz + 1), coarseAt(bx + 1, by + 1, bz + 1), wx),
                    wy,
                );
                const expected = mix(z0, z1, wz);
                const actual = fineSdf[x + fineDim[0] * (y + fineDim[1] * z)];
                polygonSdfUpsampleMaxError = Math.max(polygonSdfUpsampleMaxError, Math.abs(actual - expected));
            }
        }
    }

    const countDiffuse = engine._device.createBuffer({
        size: 8 * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const countData = new Float32Array(8 * 8);
    countData.set([0, 0, 0, 10, 0, 0, 0, 0], 0);
    countData.set([0, 0, 0, 10, 0, 0, 0, 1], 16);
    countData.set([0, 0, 0, 10, 0, 0, 0, 1], 24);
    countData.set([0, 0, 0, 10, 0, 0, 0, 2], 56);
    engine._device.queue.writeBuffer(countDiffuse, 0, countData);
    const denseCountTracker = createDiffuseCountTracker(engine._device, "test-dense");
    denseCountTracker.configure(countDiffuse, 8);
    const denseCounts = await sampleCounts(engine._device, denseCountTracker);

    const activeState = engine._device.createBuffer({
        size: foamActiveStateBytes(8),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const activeData = new Uint32Array(64 + 64 * 2 + 8);
    activeData[1] = 4;
    activeData[3] = 0;
    activeData.set([0, 2, 3, 7], 64);
    engine._device.queue.writeBuffer(activeState, 0, activeData);
    const activeDispatch = engine._device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    engine._device.queue.writeBuffer(activeDispatch, 0, new Uint32Array([1, 1, 1]));
    const activeCountTracker = createDiffuseCountTracker(engine._device, "test-active");
    activeCountTracker.configure(countDiffuse, 8, activeState);
    const activeCounts = await sampleCounts(engine._device, activeCountTracker, activeDispatch);

    const validationError = await engine._device.popErrorScope();
    sim.dispose();
    activeSim.dispose();
    classificationSim.dispose();
    sideEmissionSim.dispose();
    turbulenceSim.dispose();
    advancedUpdateSim.dispose();
    invariantBaselineSim.dispose();
    invariantPolygonSim.dispose();
    coarsePolygonSim.dispose();
    finePolygonSim.dispose();
    pagedPolygonSim.dispose();
    denseCountTracker.dispose();
    activeCountTracker.dispose();
    countDiffuse.destroy();
    activeState.destroy();
    activeDispatch.destroy();
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
        foamOnly,
        allKindsDisabled,
        sideGenerated,
        turbulenceGenerated,
        advancedUpdated,
        polygonWhitewaterInvariant,
        baselineSimulationPasses,
        polygonSimulationPasses,
        polygonSurfacePasses,
        polygonGridDimensions,
        polygonGridSpacing,
        polygonReconstructionMultiplier,
        polygonIndexCount,
        polygonSdfUpsampleMaxError,
        pagedPolygonIndexCount,
        pagedPolygonSdfFinite,
        denseCounts,
        activeCounts,
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
        foamOnly: { active: number; kinds: number[]; slotLives: number[] };
        allKindsDisabled: { active: number };
        sideGenerated: { lowActive: number };
        turbulenceGenerated: { active: number };
        advancedUpdated: { slotKinds: number[]; slotLives: number[]; slotVelocities: number[][] };
        polygonWhitewaterInvariant: boolean;
        baselineSimulationPasses: number;
        polygonSimulationPasses: number;
        polygonSurfacePasses: number;
        polygonGridDimensions: number[];
        polygonGridSpacing: number;
        polygonReconstructionMultiplier: number;
        polygonIndexCount: number;
        polygonSdfUpsampleMaxError: number;
        pagedPolygonIndexCount: number;
        pagedPolygonSdfFinite: boolean;
        denseCounts: { total: number; spray: number; foam: number; bubble: number; capacity: number };
        activeCounts: { total: number; spray: number; foam: number; bubble: number; capacity: number };
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
    expect(result.foamOnly.active).toBe(2);
    expect(result.foamOnly.kinds).toEqual([0, 2, 0]);
    expect(result.foamOnly.slotLives[0]).toBe(0);
    expect(result.foamOnly.slotLives[1]).toBeGreaterThan(0);
    expect(result.foamOnly.slotLives[2]).toBe(0);
    expect(result.foamOnly.slotLives[3]).toBeGreaterThan(0);
    expect(result.allKindsDisabled.active).toBe(0);
    expect(result.sideGenerated.lowActive).toBe(0);
    expect(result.turbulenceGenerated.active).toBeGreaterThan(0);
    expect(result.advancedUpdated.slotLives[0]).toBeGreaterThan(0);
    expect(result.advancedUpdated.slotKinds[0]).toBe(1);
    expect(result.advancedUpdated.slotKinds[1]).toBe(0);
    expect(result.advancedUpdated.slotVelocities[1]![0]).toBeGreaterThan(0);
    expect(result.advancedUpdated.slotVelocities[1]![0]).toBeLessThan(10);
    expect(result.polygonWhitewaterInvariant).toBe(true);
    expect(result.polygonSimulationPasses).toBe(result.baselineSimulationPasses);
    expect(result.polygonSurfacePasses).toBeGreaterThan(0);
    expect(result.polygonGridDimensions).toEqual([10, 10, 10]);
    expect(result.polygonGridSpacing).toBe(0.5);
    expect(result.polygonReconstructionMultiplier).toBe(2);
    expect(result.polygonIndexCount).toBeGreaterThan(0);
    expect(result.polygonIndexCount % 3).toBe(0);
    expect(result.polygonSdfUpsampleMaxError).toBeLessThan(1e-5);
    expect(result.pagedPolygonIndexCount).toBeGreaterThan(0);
    expect(result.pagedPolygonIndexCount % 3).toBe(0);
    expect(result.pagedPolygonSdfFinite).toBe(true);
    expect(result.denseCounts).toEqual({ total: 4, spray: 1, foam: 2, bubble: 1, capacity: 8 });
    expect(result.activeCounts).toEqual(result.denseCounts);
});
