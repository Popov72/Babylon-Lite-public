import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const COMMON_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/core/sim-common.ts").replace(/\\/g, "/")}`;

test("shared flow recycle claims stay transactional when routes cannot relaunch", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import {
    createFluidFlowState,
    disposeFluidFlowState,
    FLUID_FLOW_RUNTIME_WGSL,
    FLUID_FLOW_STRUCT_WGSL,
    MAX_FLUID_EMITTERS,
    prepareFluidFlowFrame,
    setFluidFlowConfig,
} from "${COMMON_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

const PARTICLES = 64;
const RESULT_EMITTER_CLAIMS = PARTICLES;
const RESULT_SINK_CLAIMS = PARTICLES + 1;
const RESULT_WORDS = PARTICLES + 2;
const EMITTER_COUNTER_WORDS = MAX_FLUID_EMITTERS * 2;

function transform(position) {
    return { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1] };
}

function createEmitter(extra = {}) {
    return {
        id: "source",
        name: "Source",
        enabled: true,
        behavior: "inflow",
        transform: transform([4, 0, 0]),
        shape: { type: "box", size: [1, 1, 1] },
        sampling: "volume",
        velocity: [0, 0, 0],
        velocitySpace: "world",
        spread: 0,
        ...extra,
    };
}

function createSink(volumeRate) {
    return {
        id: "sink",
        name: "Sink",
        enabled: true,
        mode: "recycle",
        transform: transform([0, 0, 0]),
        shape: { type: "box", size: [2, 2, 2] },
        targets: ["source"],
        volumeRate,
    };
}

async function readWords(device, buffer) {
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

const shader = [
    FLUID_FLOW_STRUCT_WGSL,
    "@group(0) @binding(0) var<uniform> flow: FluidFlowData;",
    "@group(0) @binding(1) var<storage, read_write> flowCounters: array<atomic<u32>>;",
    "@group(0) @binding(2) var<storage, read_write> results: array<u32>;",
    FLUID_FLOW_RUNTIME_WGSL,
    "@compute @workgroup_size(" + PARTICLES + ")",
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) localIndex: u32) {",
    "    let index = gid.x;",
    "    let launch = fluidTryRelaunch(vec3<f32>(0.0), index);",
    "    results[index] = launch.launched;",
    "    storageBarrier();",
    "    if (localIndex == 0u) {",
    "        results[" + RESULT_EMITTER_CLAIMS + "] = atomicLoad(&flowCounters[0u]);",
    "        results[" + RESULT_SINK_CLAIMS + "] = atomicLoad(&flowCounters[" + EMITTER_COUNTER_WORDS + "u]);",
    "    }",
    "}",
].join("\\n");

async function runCase(device, pipeline, options) {
    const state = createFluidFlowState(device, PARTICLES, 0.1, 1);
    setFluidFlowConfig(state, {
        emitters: [
            createEmitter({
                ...(options.delayBeforeStart === undefined ? {} : { delayBeforeStart: options.delayBeforeStart }),
                ...(options.emitterBudgetParticles === undefined ? {} : { volumeRate: options.emitterBudgetParticles }),
            }),
        ],
        sinks: [createSink(options.sinkBudgetParticles)],
    });
    prepareFluidFlowFrame(state, 1);

    const resultBuffer = device.createBuffer({
        label: "flow-recycle-results",
        size: RESULT_WORDS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: state.uniformBuffer } },
            { binding: 1, resource: { buffer: state.counterBuffer } },
            { binding: 2, resource: { buffer: resultBuffer } },
        ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const words = await readWords(device, resultBuffer);
    let launched = 0;
    for (let index = 0; index < PARTICLES; index++) {
        launched += words[index] ?? 0;
    }
    const summary = {
        launched,
        emitterClaims: words[RESULT_EMITTER_CLAIMS] ?? 0,
        sinkClaims: words[RESULT_SINK_CLAIMS] ?? 0,
    };

    resultBuffer.destroy();
    disposeFluidFlowState(state);
    return summary;
}

async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({
        label: "flow-recycle-transactional",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "flow-recycle-transactional", code: shader }),
            entryPoint: "main",
        },
    });

    const delayed = await runCase(device, pipeline, { delayBeforeStart: 10, sinkBudgetParticles: 7 });
    const exhausted = await runCase(device, pipeline, { emitterBudgetParticles: 0, sinkBudgetParticles: 7 });
    const successful = await runCase(device, pipeline, { emitterBudgetParticles: 7, sinkBudgetParticles: 7 });

    const validationError = await device.popErrorScope();
    if (validationError) {
        throw new Error(validationError.message);
    }
    canvas.dataset.result = JSON.stringify({ delayed, exhausted, successful });
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 90_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);

    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        delayed: { launched: number; emitterClaims: number; sinkClaims: number };
        exhausted: { launched: number; emitterClaims: number; sinkClaims: number };
        successful: { launched: number; emitterClaims: number; sinkClaims: number };
    };
    expect(result.delayed).toEqual({ launched: 0, emitterClaims: 0, sinkClaims: 0 });
    expect(result.exhausted).toEqual({ launched: 0, emitterClaims: 0, sinkClaims: 0 });
    expect(result.successful).toEqual({ launched: 7, emitterClaims: 7, sinkClaims: 7 });
});
