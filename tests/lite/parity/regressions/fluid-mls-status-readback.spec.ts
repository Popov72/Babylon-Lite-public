import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

import { MLS_MIN_CELL_PARTICLE_CONTRIBUTORS } from "../../../../packages/babylon-lite/src/fluid/solvers/wgsl-shared";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const MLS_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts").replace(/\\/g, "/")}`;

test("MLS-MPM throws rejected page and accumulation status maps on the next step", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createMlsMpmSim } from "${MLS_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

function engineRejectingStatusMap(device, labelFragment, rejectionMessage) {
    const trackedDevice = new Proxy(device, {
        get(target, prop) {
            if (prop === "createBuffer") {
                return (descriptor) => {
                    const buffer = target.createBuffer(descriptor);
                    if (descriptor.label?.includes(labelFragment)) {
                        Object.defineProperty(buffer, "mapAsync", {
                            configurable: true,
                            value: () => Promise.reject(new Error(rejectionMessage)),
                        });
                    }
                    return buffer;
                };
            }
            const value = Reflect.get(target, prop, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    return { _device: trackedDevice };
}

async function submitStep(device, sim) {
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
}

async function rejectedStatus(device, kind) {
    const label = kind === "page" ? "mpm-page-status-readback" : "mpm-accumulation-status-readback";
    const rejectionMessage = "forced-" + kind + "-map-rejection";
    const sim = createMlsMpmSim(engineRejectingStatusMap(device, label, rejectionMessage), {
        count: 8,
        initialPositions: new Float32Array([
            0.25, 0.75, 0.25, 0.75, 0.75, 0.25,
            0.25, 1.25, 0.25, 0.75, 1.25, 0.25,
            0.25, 0.75, 0.75, 0.75, 0.75, 0.75,
            0.25, 1.25, 0.75, 0.75, 1.25, 0.75,
        ]),
        boundsMin: [-1, 0, -1],
        boundsMax: [2, 3, 2],
        gridDim: [6, 6, 6],
        dx: 0.5,
        gravity: 0,
        restDensity: 3,
        stiffness: 80,
        viscosity: 0.1,
        substeps: 1,
        maxSubDt: 1,
        ...(kind === "page" ? { pagedGrid: true, pagedGridMaxPages: 8 } : {}),
    });

    await submitStep(device, sim);
    await submitStep(device, sim);
    await new Promise((resolve) => setTimeout(resolve, 0));

    let caught;
    try {
        sim.step(device.createCommandEncoder(), 1 / 60);
    } catch (error) {
        caught = {
            message: error?.message ?? String(error),
            cause: error?.cause?.message ?? "",
        };
    }
    sim.dispose();
    return caught;
}

async function main() {
    const engine = await createEngine(canvas);
    return {
        accumulation: await rejectedStatus(engine._device, "accumulation"),
        page: await rejectedStatus(engine._device, "page"),
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
    expect(result.accumulation.message).toContain("fixed-point accumulation status");
    expect(result.accumulation.cause).toBe("forced-accumulation-map-rejection");
    expect(result.page.message).toContain("paged-grid overflow status");
    expect(result.page.cause).toBe("forced-page-map-rejection");
});

test("MLS-MPM reset retires copied status and ignores stale mapped rejections", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createMlsMpmSim } from "${MLS_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

function createTrackedEngine(device, labelFragment, mode, tracker) {
    return {
        _device: new Proxy(device, {
            get(target, prop) {
                if (prop === "createBuffer") {
                    return (descriptor) => {
                        const buffer = target.createBuffer(descriptor);
                        if (descriptor.label?.includes(labelFragment)) {
                            Object.defineProperty(buffer, "mapAsync", {
                                configurable: true,
                                value: () => {
                                    tracker.mapCalls++;
                                    if (mode === "reject") {
                                        return Promise.reject(new Error("stale-copied-map"));
                                    }
                                    return new Promise((resolve, reject) => {
                                        tracker.rejectors.push(reject);
                                    });
                                },
                            });
                        }
                        return buffer;
                    };
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        }),
    };
}

function createSim(engine, kind) {
    return createMlsMpmSim(engine, {
        count: 8,
        initialPositions: new Float32Array([
            0.25, 0.75, 0.25, 0.75, 0.75, 0.25,
            0.25, 1.25, 0.25, 0.75, 1.25, 0.25,
            0.25, 0.75, 0.75, 0.75, 0.75, 0.75,
            0.25, 1.25, 0.75, 0.75, 1.25, 0.75,
        ]),
        boundsMin: [-1, 0, -1],
        boundsMax: [2, 3, 2],
        gridDim: [6, 6, 6],
        dx: 0.5,
        gravity: 0,
        restDensity: 3,
        stiffness: 80,
        viscosity: 0.1,
        substeps: 1,
        maxSubDt: 1,
        ...(kind === "page" ? { pagedGrid: true, pagedGridMaxPages: 8 } : {}),
    });
}

async function submitStep(device, sim) {
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
}

async function copiedReset(device, kind) {
    const tracker = { mapCalls: 0, rejectors: [] };
    const label = kind === "page" ? "mpm-page-status-readback" : "mpm-accumulation-status-readback";
    const sim = createSim(createTrackedEngine(device, label, "reject", tracker), kind);
    await submitStep(device, sim);
    sim.reset();
    await submitStep(device, sim);
    await new Promise((resolve) => setTimeout(resolve, 0));
    sim.dispose();
    return tracker.mapCalls;
}

async function mappingReset(device, kind) {
    const tracker = { mapCalls: 0, rejectors: [] };
    const label = kind === "page" ? "mpm-page-status-readback" : "mpm-accumulation-status-readback";
    const sim = createSim(createTrackedEngine(device, label, "defer", tracker), kind);
    await submitStep(device, sim);
    await submitStep(device, sim);
    sim.reset();
    for (const reject of tracker.rejectors) {
        reject(new Error("stale-mapping-rejection"));
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
    let thrown = "";
    try {
        sim.step(device.createCommandEncoder(), 1 / 60);
    } catch (error) {
        thrown = error?.message ?? String(error);
    }
    sim.dispose();
    return { mapCalls: tracker.mapCalls, thrown };
}

async function main() {
    const engine = await createEngine(canvas);
    return {
        accumulationCopied: await copiedReset(engine._device, "accumulation"),
        pageCopied: await copiedReset(engine._device, "page"),
        accumulationMapping: await mappingReset(engine._device, "accumulation"),
        pageMapping: await mappingReset(engine._device, "page"),
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
    const state = await page.locator("canvas").evaluate((node) => ({ error: node.dataset.error, result: node.dataset.result }));
    expect(state.error).toBeUndefined();
    const result = JSON.parse(state.result!);
    expect(result.accumulationCopied).toBe(0);
    expect(result.pageCopied).toBe(0);
    expect(result.accumulationMapping.mapCalls).toBeGreaterThan(0);
    expect(result.accumulationMapping.thrown).toBe("");
    expect(result.pageMapping.mapCalls).toBeGreaterThan(0);
    expect(result.pageMapping.thrown).toBe("");
});

test("MLS-MPM pauses saturated status rings and retains the next impossible codec", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createMlsMpmSim } from "${MLS_ENTRY}";
import { MLS_MIN_CELL_PARTICLE_CONTRIBUTORS } from "${MLS_ENTRY.replace("mls-mpm-sim.ts", "wgsl-shared.ts")}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

function gatedMapEngine(device, labelFragment, tracker) {
    return {
        _device: new Proxy(device, {
            get(target, prop) {
                if (prop === "createBuffer") {
                    return (descriptor) => {
                        const buffer = target.createBuffer(descriptor);
                        if (descriptor.label === "mpm-particles") {
                            tracker.particles = buffer;
                        }
                        if (descriptor.label?.includes(labelFragment)) {
                            const nativeMap = buffer.mapAsync.bind(buffer);
                            Object.defineProperty(buffer, "mapAsync", {
                                configurable: true,
                                value: (...args) => new Promise((resolve, reject) => {
                                    tracker.gates.push(() => nativeMap(...args).then(resolve, reject));
                                }),
                            });
                        }
                        return buffer;
                    };
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        }),
    };
}

function particleBytes(positions) {
    const data = new Float32Array((positions.length / 3) * 20);
    for (let i = 0; i < positions.length / 3; i++) {
        data[i * 20] = positions[i * 3];
        data[i * 20 + 1] = positions[i * 3 + 1];
        data[i * 20 + 2] = positions[i * 3 + 2];
    }
    return data;
}

function tracingEncoder(device, operations) {
    const encoder = device.createCommandEncoder();
    return new Proxy(encoder, {
        get(target, prop) {
            if (["copyBufferToBuffer", "clearBuffer", "beginComputePass", "pushDebugGroup"].includes(prop)) {
                return (...args) => {
                    operations.push(prop);
                    return target[prop](...args);
                };
            }
            const value = Reflect.get(target, prop, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
}

async function submitStep(device, sim) {
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
}

async function releaseGate(tracker, index) {
    await tracker.gates[index]();
    await new Promise((resolve) => setTimeout(resolve, 0));
}

async function saturatedOverflow(device, kind) {
    const accumulation = kind === "accumulation";
    const count = accumulation ? MLS_MIN_CELL_PARTICLE_CONTRIBUTORS + 1 : 16;
    const safe = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        safe[i * 3] = accumulation ? 1.25 + (i % 4) * 0.5 : 0.75;
        safe[i * 3 + 1] = accumulation ? 2.25 + (Math.floor(i / 4) % 4) * 0.5 : 0.75;
        safe[i * 3 + 2] = accumulation ? 1.25 + (Math.floor(i / 16) % 4) * 0.5 : 0.75;
    }
    const tracker = { gates: [], particles: null };
    let pageOverflow = 0;
    const label = accumulation ? "mpm-accumulation-status-readback" : "mpm-page-status-readback";
    const sim = createMlsMpmSim(gatedMapEngine(device, label, tracker), {
        count,
        initialPositions: safe,
        boundsMin: [0, 0, 0],
        boundsMax: [8, 8, 8],
        gridDim: [16, 16, 16],
        dx: 0.5,
        gravity: 0,
        restDensity: 3,
        stiffness: accumulation ? 9_500_000 : 0,
        viscosity: 0,
        substeps: 1,
        maxSubDt: 1,
        ...(accumulation ? {} : { pagedGrid: true, pagedGridMaxPages: 10, onPagedGridOverflow: () => { pageOverflow++; } }),
    });
    sim.setFoam({ activeParticles: true });
    await submitStep(device, sim);
    await submitStep(device, sim);

    const overflow = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const far = !accumulation && i >= count / 2;
        overflow[i * 3] = far ? 6.75 : 0.75;
        overflow[i * 3 + 1] = far ? 6.75 : 0.75;
        overflow[i * 3 + 2] = far ? 6.75 : 0.75;
    }
    device.queue.writeBuffer(tracker.particles, 0, particleBytes(overflow));
    const sideBeforePause = sim.diffuse.activeIndicesOffset;
    const firstPauseOperations = [];
    sim.step(tracingEncoder(device, firstPauseOperations), 1 / 60);
    const sideAfterPause = sim.diffuse.activeIndicesOffset;

    await releaseGate(tracker, 0);
    await submitStep(device, sim);
    const optimisticOverflowSide = sim.diffuse.activeIndicesOffset;
    const secondPauseOperations = [];
    sim.step(tracingEncoder(device, secondPauseOperations), 1 / 60);
    await releaseGate(tracker, 2);

    let thrown = "";
    const finalOperations = [];
    try {
        sim.step(tracingEncoder(device, finalOperations), 1 / 60);
    } catch (error) {
        thrown = error?.message ?? String(error);
    }
    const restoredSide = sim.diffuse.activeIndicesOffset;
    sim.dispose();
    return {
        firstPauseOperations,
        secondPauseOperations,
        finalOperations,
        sideBeforePause,
        sideAfterPause,
        optimisticOverflowSide,
        restoredSide,
        pageOverflow,
        thrown,
        gateCount: tracker.gates.length,
    };
}

async function main() {
    const engine = await createEngine(canvas);
    return {
        accumulation: await saturatedOverflow(engine._device, "accumulation"),
        page: await saturatedOverflow(engine._device, "page"),
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
    const state = await page.locator("canvas").evaluate((node) => ({ error: node.dataset.error, result: node.dataset.result }));
    expect(state.error).toBeUndefined();
    const result = JSON.parse(state.result!);
    for (const entry of [result.accumulation, result.page]) {
        expect(entry.gateCount).toBe(3);
        expect(entry.firstPauseOperations).toEqual([]);
        expect(entry.secondPauseOperations).toEqual([]);
        expect(entry.sideAfterPause).toBe(entry.sideBeforePause);
        expect(entry.optimisticOverflowSide).not.toBe(entry.sideBeforePause);
        expect(entry.restoredSide).toBe(entry.sideBeforePause);
    }
    expect(result.accumulation.thrown).toContain("no overflow-safe 32-bit codec");
    expect(result.accumulation.finalOperations).toEqual([]);
    expect(result.page.pageOverflow).toBe(1);
    expect(result.page.thrown).toBe("");
    expect(result.page.finalOperations).toEqual([]);
});

test("MLS-MPM first impossible codec leaves primary and diffuse buffers byte-identical", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createMlsMpmSim } from "${MLS_ENTRY}";
import { MLS_MIN_CELL_PARTICLE_CONTRIBUTORS } from "${MLS_ENTRY.replace("mls-mpm-sim.ts", "wgsl-shared.ts")}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

const capturedLabels = new Set([
    "mpm-particles",
    "fluid-particle-lifecycle",
    "mpm-render-pos",
    "mpm-render-vel",
    "mpm-debug",
    "mpm-foam-pool",
    "mpm-foam-head",
    "mpm-foam-active-state",
    "mpm-foam-active-dispatch",
    "mpm-foam-draw-indirect",
    "mpm-accumulation-state",
    "mpm-particles-working",
]);

function capturingEngine(device, captured) {
    return {
        _device: new Proxy(device, {
            get(target, prop) {
                if (prop === "createBuffer") {
                    return (descriptor) => {
                        const capture = capturedLabels.has(descriptor.label);
                        const buffer = target.createBuffer(capture ? { ...descriptor, usage: descriptor.usage | GPUBufferUsage.COPY_SRC } : descriptor);
                        if (capture) {
                            captured.set(descriptor.label, buffer);
                        }
                        return buffer;
                    };
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        }),
    };
}

async function snapshot(device, captured) {
    const entries = [...captured.entries()]
        .filter(([label]) => label !== "mpm-accumulation-state" && label !== "mpm-particles-working")
        .sort(([a], [b]) => a.localeCompare(b));
    const staging = entries.map(([label, buffer]) => [
        label,
        device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        buffer,
    ]);
    const encoder = device.createCommandEncoder();
    for (const [, readback, buffer] of staging) {
        encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
    }
    device.queue.submit([encoder.finish()]);
    await Promise.all(staging.map(([, readback]) => readback.mapAsync(GPUMapMode.READ)));
    const result = new Map();
    for (const [label, readback] of staging) {
        result.set(label, new Uint8Array(readback.getMappedRange()).slice());
        readback.unmap();
        readback.destroy();
    }
    return result;
}

async function readBuffer(device, buffer) {
    const readback = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Uint8Array(readback.getMappedRange()).slice();
    readback.unmap();
    readback.destroy();
    return result;
}

function firstDifference(a, b) {
    if (a.length !== b.length) { return 0; }
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) { return i; }
    }
    return -1;
}

function compareSnapshots(before, after) {
    const mismatches = [];
    for (const [label, expected] of before) {
        const actual = after.get(label);
        if (!actual || actual.length !== expected.length) {
            mismatches.push(label + ":size");
            continue;
        }
        for (let i = 0; i < expected.length; i++) {
            if (actual[i] !== expected[i]) {
                mismatches.push(label + ":" + i);
                break;
            }
        }
    }
    return mismatches;
}

async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const captured = new Map();
    const count = MLS_MIN_CELL_PARTICLE_CONTRIBUTORS + 1;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        positions[i * 3] = 0.25;
        positions[i * 3 + 1] = 0.75;
        positions[i * 3 + 2] = 0.25;
    }
    const sim = createMlsMpmSim(capturingEngine(device, captured), {
        count,
        initialPositions: positions,
        boundsMin: [-1, 0, -1],
        boundsMax: [2, 3, 2],
        gridDim: [6, 6, 6],
        dx: 0.5,
        gravity: 0,
        restDensity: 3,
        stiffness: 19_000_000,
        viscosity: 0,
        substeps: 2,
        maxSubDt: 1,
    });
    sim.setFlow({
        emitters: [{
            id: "source",
            name: "Source",
            enabled: true,
            behavior: "inflow",
            transform: { position: [0.25, 0.75, 0.25], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [0.5, 0.5, 0.5] },
            sampling: "volume",
            velocity: [4, 0, 0],
            velocitySpace: "world",
            spread: 0,
        }],
        sinks: [{
            id: "sink",
            name: "Sink",
            enabled: true,
            mode: "delete",
            transform: { position: [0.25, 0.75, 0.25], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [1, 1, 1] },
            targets: [],
        }],
    });
    const forceBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    sim.setForceField({
        struct: "struct ForceFieldParams { value: vec4<f32>, };",
        wgsl: "fn externalForce(position: vec3<f32>, velocity: vec3<f32>, dt: f32) -> vec3<f32> { return forceFieldParams.value.xyz * dt; }",
        buffer: forceBuffer,
    });
    device.queue.writeBuffer(forceBuffer, 0, new Float32Array([100, 50, -25, 0]));
    sim.setFoam({ activeParticles: true });
    device.queue.writeBuffer(captured.get("mpm-foam-pool"), 0, new Float32Array([0.25, 0.75, 0.25, 1, 1, 0, 0, 1]));

    const beforeOffset = sim.diffuse.activeIndicesOffset;
    const before = await snapshot(device, captured);
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const after = await snapshot(device, captured);
    const workingParticles = await readBuffer(device, captured.get("mpm-particles-working"));
    const optimisticOffset = sim.diffuse.activeIndicesOffset;

    const abortingEncoder = new Proxy(device.createCommandEncoder(), {
        get(target, prop) {
            if (prop === "copyBufferToBuffer") {
                return () => { throw new Error("stop-after-status-poll"); };
            }
            const value = Reflect.get(target, prop, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    try {
        sim.step(abortingEncoder, 1 / 60);
    } catch (error) {
        if (error?.message !== "stop-after-status-poll") { throw error; }
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
    const activeStateBytes = await readBuffer(device, captured.get("mpm-foam-active-state"));
    const drawBytes = await readBuffer(device, captured.get("mpm-foam-draw-indirect"));
    const activeHeader = new Uint32Array(activeStateBytes.buffer, activeStateBytes.byteOffset, 4);
    const drawArgs = new Uint32Array(drawBytes.buffer, drawBytes.byteOffset, 4);
    const gpuSide = activeHeader[3];
    const activeStride = Math.ceil(sim.diffuse.capacity / 64) * 64;
    const expectedOffset = (64 + gpuSide * activeStride) * 4;

    const statusReadback = device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const statusCopy = device.createCommandEncoder();
    const accumulationState = captured.get("mpm-accumulation-state");
    if (accumulationState) {
        statusCopy.copyBufferToBuffer(accumulationState, 0, statusReadback, 0, 8);
    }
    device.queue.submit([statusCopy.finish()]);
    let status = [];
    if (accumulationState) {
        await statusReadback.mapAsync(GPUMapMode.READ);
        status = [...new Uint32Array(statusReadback.getMappedRange())];
        statusReadback.unmap();
    }
    statusReadback.destroy();
    sim.dispose();
    forceBuffer.destroy();
    return {
        mismatches: compareSnapshots(before, after),
        workingParticleDifference: firstDifference(after.get("mpm-particles"), workingParticles),
        beforeOffset,
        optimisticOffset,
        restoredOffset: sim.diffuse.activeIndicesOffset,
        expectedOffset,
        gpuSide,
        headMatchesActiveIndices: sim.diffuse.headBuffer === sim.diffuse.activeIndices,
        listCount: activeHeader[1 + gpuSide],
        drawCount: drawArgs[1],
        status,
        captured: [...before.keys()],
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
    const state = await page.locator("canvas").evaluate((node) => ({ error: node.dataset.error, result: node.dataset.result }));
    expect(state.error).toBeUndefined();
    const result = JSON.parse(state.result!);
    expect(result.captured).toEqual([
        "fluid-particle-lifecycle",
        "mpm-debug",
        "mpm-foam-active-dispatch",
        "mpm-foam-active-state",
        "mpm-foam-draw-indirect",
        "mpm-foam-head",
        "mpm-foam-pool",
        "mpm-particles",
        "mpm-render-pos",
        "mpm-render-vel",
    ]);
    expect(result.status[0]).toBe(1);
    expect(result.status[1]).toBe(MLS_MIN_CELL_PARTICLE_CONTRIBUTORS + 1);
    expect(result.workingParticleDifference).toBeGreaterThanOrEqual(0);
    expect(result.optimisticOffset).not.toBe(result.beforeOffset);
    expect(result.restoredOffset).toBe(result.beforeOffset);
    expect(result.restoredOffset).toBe(result.expectedOffset);
    expect(result.gpuSide).toBe(0);
    expect(result.headMatchesActiveIndices).toBe(true);
    expect(result.listCount).toBe(result.drawCount);
    expect(result.mismatches).toEqual([]);
});
