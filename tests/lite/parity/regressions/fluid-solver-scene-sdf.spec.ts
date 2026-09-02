import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const PBF_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/solvers/pbf-sim.ts").replace(/\\/g, "/")}`;
const MLS_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts").replace(/\\/g, "/")}`;
const PBMPM_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/solvers/pbmpm-sim.ts").replace(/\\/g, "/")}`;

test("PBF, MLS-MPM, and PB-MPM keep null-scene stages active and reuse equivalent scene variants", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createPbfSim } from "${PBF_ENTRY}";
import { createMlsMpmSim } from "${MLS_ENTRY}";
import { createPbMpmSim } from "${PBMPM_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

function createTrackedEngine(device) {
    const counts = new Map();
    const trackedDevice = new Proxy(device, {
        get(target, prop, receiver) {
            if (prop === "createComputePipeline") {
                return (descriptor) => {
                    const label = descriptor?.label ?? "unnamed";
                    counts.set(label, (counts.get(label) ?? 0) + 1);
                    return target.createComputePipeline(descriptor);
                };
            }
            const value = Reflect.get(target, prop, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    return {
        engine: { _device: trackedDevice },
        getCount(label) {
            return counts.get(label) ?? 0;
        },
    };
}

function createTracingEncoder(device, labels) {
    const encoder = device.createCommandEncoder();
    return new Proxy(encoder, {
        get(target, prop, receiver) {
            if (prop === "beginComputePass") {
                return (descriptor) => {
                    labels.push(descriptor?.label ?? "");
                    return target.beginComputePass(descriptor);
                };
            }
            const value = Reflect.get(target, prop, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
}

async function step(device, sim, dt = 1 / 60) {
    const labels = [];
    const encoder = createTracingEncoder(device, labels);
    sim.step(encoder, dt);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return labels;
}

function createScene(device, gridConfine = true) {
    const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, new Float32Array([0.5, 0.5, 0.5, 0.45]));
    return {
        struct: "struct SceneSdfParams { sphere: vec4<f32>, };",
        sdf: "fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return length(pt - sceneSdfParams.sphere.xyz) - sceneSdfParams.sphere.w; }",
        buffer,
        gridConfine,
    };
}

async function main() {
    const engine = await createEngine(canvas);
    const positions = new Float32Array([
        0.25, 0.75, 0.25, 0.75, 0.75, 0.25,
        0.25, 1.25, 0.25, 0.75, 1.25, 0.25,
        0.25, 0.75, 0.75, 0.75, 0.75, 0.75,
        0.25, 1.25, 0.75, 0.75, 1.25, 0.75,
    ]);
    const boundsMin = [-1, 0, -1];
    const boundsMax = [2, 3, 2];
    const gridDim = [6, 6, 6];
    const dx = 0.5;
    const buffers = [];

    const pbfTracked = createTrackedEngine(engine._device);
    const pbf = createPbfSim(pbfTracked.engine, {
        count: 8,
        initialPositions: positions,
        boundsMin,
        boundsMax,
        gridDim,
        particleRadius: 0.08,
        gravity: 0,
        iterations: 2,
    });
    pbf.setSceneSdf(null);
    const pbfNullLabels = await step(engine._device, pbf);
    const pbfNullPipes = pbfTracked.getCount("fluid-apply");
    const pbfSceneA = createScene(engine._device);
    buffers.push(pbfSceneA.buffer);
    pbf.setSceneSdf(pbfSceneA);
    const pbfScenePipes = pbfTracked.getCount("fluid-apply");
    const pbfSceneB = createScene(engine._device);
    buffers.push(pbfSceneB.buffer);
    pbf.setSceneSdf(pbfSceneB);
    const pbfEquivalentPipes = pbfTracked.getCount("fluid-apply");
    pbf.setSceneSdf(null);
    const pbfNullAgainPipes = pbfTracked.getCount("fluid-apply");

    const mlsTracked = createTrackedEngine(engine._device);
    const mls = createMlsMpmSim(mlsTracked.engine, {
        count: 8,
        initialPositions: positions,
        boundsMin,
        boundsMax,
        gridDim,
        dx,
        particleRadius: 0.08,
        gravity: 0,
        restDensity: 3,
        stiffness: 80,
        viscosity: 0.1,
        substeps: 1,
        maxSubDt: 1,
    });
    mls.setSceneSdf(null);
    const mlsNullLabels = await step(engine._device, mls);
    const mlsNullUpdatePipes = mlsTracked.getCount("mpm-update");
    const mlsNullG2pPipes = mlsTracked.getCount("mpm-g2p");
    const mlsSceneA = createScene(engine._device);
    buffers.push(mlsSceneA.buffer);
    mls.setSceneSdf(mlsSceneA);
    const mlsSceneUpdatePipes = mlsTracked.getCount("mpm-update");
    const mlsSceneG2pPipes = mlsTracked.getCount("mpm-g2p");
    const mlsSceneB = createScene(engine._device);
    buffers.push(mlsSceneB.buffer);
    mls.setSceneSdf(mlsSceneB);
    const mlsEquivalentUpdatePipes = mlsTracked.getCount("mpm-update");
    const mlsEquivalentG2pPipes = mlsTracked.getCount("mpm-g2p");
    mls.setSceneSdf(null);
    const mlsNullAgainUpdatePipes = mlsTracked.getCount("mpm-update");
    const mlsNullAgainG2pPipes = mlsTracked.getCount("mpm-g2p");

    const pbTracked = createTrackedEngine(engine._device);
    const pb = createPbMpmSim(pbTracked.engine, {
        count: 8,
        initialPositions: positions,
        boundsMin,
        boundsMax,
        gridDim,
        dx,
        particleRadius: 0.08,
        gravity: 0,
        iterations: 1,
        substeps: 1,
        maxSubDt: 1,
    });
    pb.setSceneSdf(null);
    const pbNullLabels = await step(engine._device, pb);
    const pbNullGridPipes = pbTracked.getCount("pbmpm-grid-update");
    const pbNullIntegratePipes = pbTracked.getCount("pbmpm-integrate");
    const pbSceneA = createScene(engine._device);
    buffers.push(pbSceneA.buffer);
    pb.setSceneSdf(pbSceneA);
    const pbSceneGridPipes = pbTracked.getCount("pbmpm-grid-update");
    const pbSceneIntegratePipes = pbTracked.getCount("pbmpm-integrate");
    const pbSceneB = createScene(engine._device);
    buffers.push(pbSceneB.buffer);
    pb.setSceneSdf(pbSceneB);
    const pbEquivalentGridPipes = pbTracked.getCount("pbmpm-grid-update");
    const pbEquivalentIntegratePipes = pbTracked.getCount("pbmpm-integrate");
    pb.setSceneSdf(null);
    const pbNullAgainGridPipes = pbTracked.getCount("pbmpm-grid-update");
    const pbNullAgainIntegratePipes = pbTracked.getCount("pbmpm-integrate");

    pbf.dispose();
    mls.dispose();
    pb.dispose();
    for (const buffer of buffers) {
        buffer.destroy();
    }

    return {
        pbfNullLabels,
        pbfNullPipes,
        pbfScenePipes,
        pbfEquivalentPipes,
        pbfNullAgainPipes,
        mlsNullLabels,
        mlsNullUpdatePipes,
        mlsNullG2pPipes,
        mlsSceneUpdatePipes,
        mlsSceneG2pPipes,
        mlsEquivalentUpdatePipes,
        mlsEquivalentG2pPipes,
        mlsNullAgainUpdatePipes,
        mlsNullAgainG2pPipes,
        pbNullLabels,
        pbNullGridPipes,
        pbNullIntegratePipes,
        pbSceneGridPipes,
        pbSceneIntegratePipes,
        pbEquivalentGridPipes,
        pbEquivalentIntegratePipes,
        pbNullAgainGridPipes,
        pbNullAgainIntegratePipes,
    };
}

main()
    .then((result) => {
        canvas.dataset.result = JSON.stringify(result);
    })
    .catch((error) => {
        canvas.dataset.error = error?.message ?? String(error);
    });
</script>`);

    await page.waitForFunction(() => {
        const canvas = document.querySelector("canvas");
        return Boolean(canvas?.dataset.result || canvas?.dataset.error);
    });

    const state = await page.locator("canvas").evaluate((node) => ({
        error: node.dataset.error,
        result: node.dataset.result,
    }));

    expect(state.error).toBeUndefined();
    const result = JSON.parse(state.result!);
    expect(result.pbfNullLabels).toContain("fluid-apply");
    expect(result.pbfScenePipes).toBe(result.pbfNullPipes + 1);
    expect(result.pbfEquivalentPipes).toBe(result.pbfScenePipes);
    expect(result.pbfNullAgainPipes).toBe(result.pbfScenePipes);

    expect(result.mlsNullLabels).toContain("mpm-update");
    expect(result.mlsNullLabels).toContain("mpm-g2p");
    expect(result.mlsSceneUpdatePipes).toBe(result.mlsNullUpdatePipes + 1);
    expect(result.mlsSceneG2pPipes).toBe(result.mlsNullG2pPipes + 1);
    expect(result.mlsEquivalentUpdatePipes).toBe(result.mlsSceneUpdatePipes);
    expect(result.mlsEquivalentG2pPipes).toBe(result.mlsSceneG2pPipes);
    expect(result.mlsNullAgainUpdatePipes).toBe(result.mlsSceneUpdatePipes);
    expect(result.mlsNullAgainG2pPipes).toBe(result.mlsSceneG2pPipes);

    expect(result.pbNullLabels).toContain("pbmpm-grid-update");
    expect(result.pbNullLabels).toContain("pbmpm-integrate");
    expect(result.pbSceneGridPipes).toBe(result.pbNullGridPipes + 1);
    expect(result.pbSceneIntegratePipes).toBe(result.pbNullIntegratePipes + 1);
    expect(result.pbEquivalentGridPipes).toBe(result.pbSceneGridPipes);
    expect(result.pbEquivalentIntegratePipes).toBe(result.pbSceneIntegratePipes);
    expect(result.pbNullAgainGridPipes).toBe(result.pbSceneGridPipes);
    expect(result.pbNullAgainIntegratePipes).toBe(result.pbSceneIntegratePipes);
});

test("PB-MPM restitution reflects downward collision impulses", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createPbMpmSim } from "${PBMPM_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

async function step(device, sim, dt = 1 / 60) {
    const encoder = device.createCommandEncoder();
    sim.step(encoder, dt);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
}

async function readVec4(device, buffer) {
    const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, 16);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = Array.from(new Float32Array(readback.getMappedRange().slice(0, 16)));
    readback.unmap();
    readback.destroy();
    return values;
}

async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const forceBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(forceBuffer, 0, new Float32Array([0, -30, 0, 0]));
    const forceField = {
        struct: "struct ForceFieldParams { delta: vec4<f32>, };",
        wgsl: "fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> { return forceFieldParams.delta.xyz; }",
        buffer: forceBuffer,
    };
    const options = {
        count: 1,
        initialPositions: new Float32Array([0, 0.05, 0]),
        boundsMin: [-1, 0, -1],
        boundsMax: [1, 1, 1],
        gridDim: [4, 4, 4],
        dx: 0.25,
        particleRadius: 0.08,
        gravity: 0,
        iterations: 1,
        substeps: 1,
        maxSubDt: 1,
    };

    const stick = createPbMpmSim({ _device: device }, { ...options, restitution: 0 });
    const bounce = createPbMpmSim({ _device: device }, { ...options, restitution: 1 });
    stick.setForceField(forceField);
    bounce.setForceField(forceField);
    for (let i = 0; i < 4; i += 1) {
        await step(device, stick);
        await step(device, bounce);
    }
    stick.setForceField(null);
    bounce.setForceField(null);
    await step(device, stick);
    await step(device, bounce);
    await step(device, stick);
    await step(device, bounce);

    const stickPosition = await readVec4(device, stick.positionBuffer);
    const bouncePosition = await readVec4(device, bounce.positionBuffer);

    stick.dispose();
    bounce.dispose();
    forceBuffer.destroy();

    return {
        stickY: stickPosition[1],
        bounceY: bouncePosition[1],
    };
}

main()
    .then((result) => {
        canvas.dataset.result = JSON.stringify(result);
    })
    .catch((error) => {
        canvas.dataset.error = error?.message ?? String(error);
    });
</script>`);

    await page.waitForFunction(() => {
        const canvas = document.querySelector("canvas");
        return Boolean(canvas?.dataset.result || canvas?.dataset.error);
    });

    const state = await page.locator("canvas").evaluate((node) => ({
        error: node.dataset.error,
        result: node.dataset.result,
    }));

    expect(state.error).toBeUndefined();
    const result = JSON.parse(state.result!);
    expect(result.stickY).toBeGreaterThanOrEqual(0);
    expect(result.bounceY).toBeGreaterThanOrEqual(0);
    expect(result.bounceY).toBeGreaterThan(result.stickY);
});
