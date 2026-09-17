import { BU } from "../../../engine/gpu-flags.js";
import type { FluidAllocationResource } from "../../core/allocation-plan.js";
import type { FluidProfiler } from "../../core/sim-common.js";
import type { FluidTimestepDiagnostics } from "../../core/timestep-scheduler.js";
import { FLIP_REFERENCE_BINDINGS, FLIP_REFERENCE_RUNTIME_ENTRIES, flipReferenceWgsl } from "./shaders.js";
import { FLIP_REFERENCE_GPU_CONTROL_BINDINGS, FLIP_REFERENCE_GPU_CONTROL_WGSL } from "./gpu-control.js";
import { FLIP_REFERENCE_GPU_COMPONENT_BINDINGS, FLIP_REFERENCE_GPU_COMPONENT_WGSL, flipReferenceGpuComponentBytes } from "./gpu-components.js";
import { FLIP_REFERENCE_GPU_PRESSURE_BINDINGS, FLIP_REFERENCE_GPU_PRESSURE_WGSL, flipReferenceGpuPressureBytes, flipReferenceGpuPressureWorkgroupSize } from "./gpu-pressure.js";
import {
    FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_BINDINGS,
    FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_WGSL,
    FLIP_REFERENCE_PARALLEL_PRESSURE_DISTRIBUTED_PARTICLES,
    FLIP_REFERENCE_PARALLEL_PRESSURE_ITERATIONS,
    FLIP_REFERENCE_PARALLEL_PRESSURE_MIN_PARTICLES,
} from "./gpu-pressure-parallel.js";
import { FLIP_REFERENCE_GPU_STAGES_BINDINGS, FLIP_REFERENCE_GPU_STAGES_WGSL } from "./gpu-stages.js";
import type { FlipReferenceSimulation } from "./types.js";

const STATE_BYTES = 128;
const ARGUMENT_BYTES = 72;
const TELEMETRY_BYTES = 208;
const TELEMETRY_SLOTS = 3;

interface TelemetrySlot {
    readonly buffer: GPUBuffer;
    pending: boolean;
    mapping: Promise<void> | null;
    sequence: number;
}

/** @internal All per-frame numerical control stays in stateBuffer, never in telemetry. */
export interface FlipReferenceGpuRuntime {
    readonly core: FlipReferenceSimulation;
    readonly stateBuffer: GPUBuffer;
    readonly argumentBuffer: GPUBuffer;
    readonly frameBuffer: GPUBuffer;
    readonly frameData: Float32Array;
    readonly publishedDraw: GPUBuffer;
    readonly buffers: GPUBuffer[];
    readonly pipelines: Record<string, GPUComputePipeline>;
    readonly bindGroups: Record<string, GPUBindGroup>;
    readonly slots: TelemetrySlot[];
    readonly bytes: number;
    readonly pressureWorkgroupSize: number;
    ready: Promise<void>;
    error: Error | null;
    telemetryError: Error | null;
    timestepDiagnostics: FluidTimestepDiagnostics;
    disposed: boolean;
    publishedCount: number;
    sequence: number;
    publishedSequence: number;
    lastEncoder: GPUCommandEncoder | null;
}

/** @internal Canonical extra allocations, shared with the facade's memory projection. */
export function flipReferenceGpuResources(cells: number): FluidAllocationResource[] {
    return [
        { name: "Reference GPU control", kind: "buffer", binding: "storage", bytes: STATE_BYTES },
        { name: "Reference dispatch arguments", kind: "buffer", binding: "storage", bytes: ARGUMENT_BYTES },
        { name: "Reference GPU components", kind: "buffer", binding: "storage", bytes: flipReferenceGpuComponentBytes(cells) },
        { name: "Reference frame parameters", kind: "buffer", binding: "uniform", bytes: 16 },
        { name: "Reference pressure rows", kind: "buffer", binding: "storage", bytes: flipReferenceGpuPressureBytes(cells) },
        { name: "Reference pressure low components", kind: "buffer", binding: "storage", bytes: cells * 4 },
        ...Array.from({ length: TELEMETRY_SLOTS }, (_, index): FluidAllocationResource => ({
            name: `Reference telemetry ${index}`,
            kind: "buffer",
            binding: "none",
            bytes: TELEMETRY_BYTES,
        })),
    ];
}

/** @internal Create once per seed; source and publication buffer identities remain stable. */
export function createFlipReferenceGpuRuntime(
    core: FlipReferenceSimulation,
    publishedPositions: GPUBuffer,
    publishedVelocities: GPUBuffer,
    publishedSpeeds: GPUBuffer,
    publishedDraw: GPUBuffer
): FlipReferenceGpuRuntime {
    const device = core._device;
    if (device.limits.maxStorageBuffersPerShaderStage < 8 || device.limits.maxComputeInvocationsPerWorkgroup < 256 || device.limits.maxComputeWorkgroupSizeX < 256) {
        throw new Error("[FLIP Reference] GPU-resident execution requires eight storage bindings and 256-invocation compute workgroups.");
    }
    const resources = flipReferenceGpuResources(core._cells);
    const pressureWorkgroupSize = flipReferenceGpuPressureWorkgroupSize(device.limits);
    for (const resource of resources) {
        if (resource.bytes > device.limits.maxBufferSize || (resource.binding === "storage" && resource.bytes > device.limits.maxStorageBufferBindingSize)) {
            throw new RangeError(`[FLIP Reference] ${resource.name} exceeds this device's buffer limits.`);
        }
    }
    const buffers: GPUBuffer[] = [];
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    let scopesOpen = true;
    try {
        const storage = BU.STORAGE | BU.COPY_SRC | BU.COPY_DST;
        for (let index = 0; index < resources.length; index++) {
            const resource = resources[index]!;
            const usage =
                index === 1
                    ? storage | BU.INDIRECT
                    : resource.binding === "storage"
                      ? storage
                      : resource.binding === "uniform"
                        ? BU.UNIFORM | BU.COPY_DST
                        : BU.MAP_READ | BU.COPY_DST;
            buffers.push(device.createBuffer({ label: resource.name, size: resource.bytes, usage }));
        }
        const stateBuffer = buffers[0]!;
        const argumentBuffer = buffers[1]!;
        const frameBuffer = buffers[3]!;
        const bindingBuffers = [
            core._uniformBuffer,
            core._positionBuffer,
            core._velocityBuffer,
            core._debugBuffer,
            core._faceBuffer,
            core._cellBuffer,
            core._listBuffer,
            core._solidBuffer,
            core._pressureBuffer,
            core._particleScratchBuffer,
            core._particleStateBuffer,
            stateBuffer,
            buffers[2]!,
            frameBuffer,
            argumentBuffer,
            publishedPositions,
            publishedVelocities,
            publishedSpeeds,
            publishedDraw,
            buffers[4]!,
        ];
        bindingBuffers[21] = buffers[5]!;
        const module = device.createShaderModule({
            label: "flip-reference:gpu-resident",
            code:
                flipReferenceWgsl(true, true) +
                FLIP_REFERENCE_GPU_CONTROL_WGSL +
                FLIP_REFERENCE_GPU_COMPONENT_WGSL +
                FLIP_REFERENCE_GPU_PRESSURE_WGSL +
                FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_WGSL +
                FLIP_REFERENCE_GPU_STAGES_WGSL,
        });
        const pipelines: Record<string, GPUComputePipeline> = {};
        const bindGroups: Record<string, GPUBindGroup> = {};
        const bindings = {
            ...FLIP_REFERENCE_BINDINGS,
            ...FLIP_REFERENCE_GPU_CONTROL_BINDINGS,
            ...FLIP_REFERENCE_GPU_COMPONENT_BINDINGS,
            ...FLIP_REFERENCE_GPU_PRESSURE_BINDINGS,
            ...FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_BINDINGS,
            ...FLIP_REFERENCE_GPU_STAGES_BINDINGS,
        };
        for (const [entryPoint, original] of Object.entries(bindings)) {
            const readsRuntime = FLIP_REFERENCE_RUNTIME_ENTRIES.includes(entryPoint);
            if (Object.hasOwn(FLIP_REFERENCE_BINDINGS, entryPoint) && !readsRuntime && !Object.hasOwn(FLIP_REFERENCE_GPU_PRESSURE_BINDINGS, entryPoint)) {
                pipelines[entryPoint] = core._pipelines[entryPoint]!;
                bindGroups[entryPoint] = core._bindGroups[entryPoint]!;
                continue;
            }
            const indices = readsRuntime ? [...original, 11] : original;
            const pipeline = device.createComputePipeline({
                label: `flip-reference:${entryPoint}`,
                layout: "auto",
                compute: {
                    module,
                    entryPoint,
                    ...(entryPoint === "solveGpuPressure" ||
                    entryPoint === "initializeParallelGpuPressure" ||
                    entryPoint === "resumeGpuPressure" ||
                    entryPoint === "finishParallelPressureIteration"
                        ? { constants: { pressureWorkgroupSize } }
                        : {}),
                },
            });
            pipelines[entryPoint] = pipeline;
            bindGroups[entryPoint] = device.createBindGroup({
                label: `flip-reference:${entryPoint}`,
                layout: pipeline.getBindGroupLayout(0),
                entries: indices.map((binding) => ({ binding, resource: { buffer: bindingBuffers[binding]! } })),
            });
        }
        const gpu: FlipReferenceGpuRuntime = {
            core,
            stateBuffer,
            argumentBuffer,
            frameBuffer,
            frameData: new Float32Array(4),
            publishedDraw,
            buffers,
            pipelines,
            bindGroups,
            slots: buffers.slice(6).map((buffer) => ({ buffer, pending: false, mapping: null, sequence: 0 })),
            bytes: resources.reduce((sum, resource) => sum + resource.bytes, 0),
            pressureWorkgroupSize,
            ready: Promise.resolve(),
            error: null,
            telemetryError: null,
            timestepDiagnostics: { deferredSeconds: 0, droppedSeconds: 0, saturated: false },
            disposed: false,
            publishedCount: core.count,
            sequence: 0,
            publishedSequence: 0,
            lastEncoder: null,
        };
        device.queue.writeBuffer(stateBuffer, 0, new Uint32Array([core.count]));
        core._params[43] = core._maxPressureIterations;
        const initialEncoder = device.createCommandEncoder({ label: "flip-reference:initial-speed" });
        const initialPass = initialEncoder.beginComputePass();
        initialPass.setPipeline(pipelines.initializeGpuMaximumSpeed!);
        initialPass.setBindGroup(0, bindGroups.initializeGpuMaximumSpeed!);
        initialPass.dispatchWorkgroups(Math.max(1, Math.ceil(core.capacity / 128)));
        initialPass.end();
        device.queue.submit([initialEncoder.finish()]);
        const allocation = device.popErrorScope();
        const validation = device.popErrorScope();
        scopesOpen = false;
        gpu.ready = Promise.all([core._ready, module.getCompilationInfo(), allocation, validation]).then(([, info, outOfMemory, invalid]) => {
            const shaderErrors = info.messages.filter((message) => message.type === "error");
            if (core._error || outOfMemory || invalid || shaderErrors.length) {
                throw (
                    core._error ??
                    new Error(
                        `[FLIP Reference] GPU initialization failed: ${outOfMemory?.message ?? invalid?.message ?? shaderErrors.map((message) => message.message).join("\n")}`
                    )
                );
            }
        });
        void gpu.ready.catch((error: unknown) => {
            gpu.error = error instanceof Error ? error : new Error(String(error));
        });
        return gpu;
    } catch (error) {
        for (const buffer of buffers) {
            buffer.destroy();
        }
        if (scopesOpen) {
            void device.popErrorScope();
            void device.popErrorScope();
        }
        throw error;
    }
}

function direct(gpu: FlipReferenceGpuRuntime, pass: GPUComputePassEncoder, entry: string): void {
    pass.setPipeline(gpu.pipelines[entry]!);
    pass.setBindGroup(0, gpu.bindGroups[entry]!);
    pass.dispatchWorkgroups(1);
}

function indirect(gpu: FlipReferenceGpuRuntime, pass: GPUComputePassEncoder, entry: string, kind: number): void {
    pass.setPipeline(gpu.pipelines[entry]!);
    pass.setBindGroup(0, gpu.bindGroups[entry]!);
    pass.dispatchWorkgroupsIndirect(gpu.argumentBuffer, kind * 12);
}

function extend(gpu: FlipReferenceGpuRuntime, pass: GPUComputePassEncoder, beforeForces: boolean): void {
    for (let layer = 0; layer < gpu.core._extrapolationLayers; layer++) {
        const entry = beforeForces ? (layer % 2 === 0 ? "extendBeforeToB" : "extendBeforeToA") : layer % 2 === 0 ? "extendToB" : "extendToA";
        indirect(gpu, pass, entry, 1);
    }
    if (gpu.core._extrapolationLayers % 2 !== 0) {
        indirect(gpu, pass, "copyToA", 1);
    }
}

/** @internal Records bounded work for one requested frame. The caller submits this encoder; no wait is introduced. */
export function recordFlipReferenceFrame(
    gpu: FlipReferenceGpuRuntime,
    encoder: GPUCommandEncoder,
    dt: number,
    minSubsteps: number,
    maxSubsteps: number,
    maxSubDt: number,
    cfl: number,
    sampler: { readonly pipeline: GPUComputePipeline; readonly bindGroup: GPUBindGroup } | null,
    profiler: FluidProfiler | null = null,
    outsideFrame = false,
    force: { readonly pipeline: GPUComputePipeline; readonly bindGroup: GPUBindGroup } | null = null
): void {
    if (gpu.disposed) {
        throw new Error("[FLIP Reference] GPU runtime is disposed.");
    }
    if (gpu.error) {
        throw gpu.error;
    }
    if (gpu.lastEncoder === encoder) {
        throw new Error("[FLIP Reference] record only one simulation frame per command encoder.");
    }
    // Previous recordings have been submitted before this frame begins. Mapping only updates diagnostics.
    void collectFlipReferenceGpuStatus(gpu);
    gpu.lastEncoder = encoder;
    const core = gpu.core;
    const parallelSubsteps = core._removalEnabled && core.capacity >= FLIP_REFERENCE_PARALLEL_PRESSURE_MIN_PARTICLES ? minSubsteps : 0;
    const parallelIterations = Math.min(FLIP_REFERENCE_PARALLEL_PRESSURE_ITERATIONS, core._maxPressureIterations);
    const fusedParallelUpdates = core.capacity < FLIP_REFERENCE_PARALLEL_PRESSURE_DISTRIBUTED_PARTICLES && gpu.pressureWorkgroupSize >= 512;
    core._device.queue.writeBuffer(core._uniformBuffer, 0, core._params);
    gpu.frameData.set([dt, maxSubDt, minSubsteps, cfl]);
    core._device.queue.writeBuffer(gpu.frameBuffer, 0, gpu.frameData);
    const commandEnd = outsideFrame ? profiler?.commandSpan?.(encoder, "Simulation") : undefined;
    const span = outsideFrame ? undefined : profiler?.stageSpan?.("Simulation");
    const pass = encoder.beginComputePass({ label: "flip-reference:gpu-frame", timestampWrites: span?.begin });
    direct(gpu, pass, "beginGpuFrame");
    for (let substep = 0; substep < maxSubsteps; substep++) {
        direct(gpu, pass, "beginGpuSubstep");
        direct(gpu, pass, "updateGpuDispatch");
        if (sampler) {
            pass.setPipeline(sampler.pipeline);
            pass.setBindGroup(0, sampler.bindGroup);
            pass.dispatchWorkgroupsIndirect(gpu.argumentBuffer, 24);
        }
        indirect(gpu, pass, "clearGpuParticleStage", 5);
        if (force) {
            pass.setPipeline(force.pipeline);
            pass.setBindGroup(0, force.bindGroup);
            pass.dispatchWorkgroupsIndirect(gpu.argumentBuffer, 36);
            direct(gpu, pass, "updateGpuDispatch");
        }
        indirect(gpu, pass, "linkParticles", 3);
        indirect(gpu, pass, "liquidLevelSet", 0);
        indirect(gpu, pass, "particleToGrid", 1);
        extend(gpu, pass, true);
        indirect(gpu, pass, "snapshotAndForce", 1);
        indirect(gpu, pass, "buildMatrix", 0);
        indirect(gpu, pass, "initializeGpuComponents", 0);
        indirect(gpu, pass, "linkGpuComponents", 0);
        indirect(gpu, pass, "aggregateGpuComponents", 0);
        if (core.referenceNumerics) {
            indirect(gpu, pass, "markGpuClosedPockets", 0);
            indirect(gpu, pass, "conditionGpuMatrix", 1);
        }
        indirect(gpu, pass, "sumGpuSealedFlux", 0);
        indirect(gpu, pass, "fixGpuPressureGauges", 0);
        indirect(gpu, pass, "prepareGpuPressure", 0);
        indirect(gpu, pass, "initializeGpuPressure", 0);
        if (substep < parallelSubsteps) {
            indirect(gpu, pass, "initializeParallelGpuPressure", 4);
            direct(gpu, pass, "beginParallelPressure");
            for (let iteration = 0; iteration < parallelIterations; iteration++) {
                indirect(gpu, pass, "applyParallelPressure", 0);
                if (fusedParallelUpdates) {
                    direct(gpu, pass, "finishParallelPressureIteration");
                } else {
                    direct(gpu, pass, "alphaParallelPressure");
                    indirect(gpu, pass, "updateParallelPressure", 0);
                    direct(gpu, pass, "betaParallelPressure");
                }
            }
            direct(gpu, pass, "finishParallelPressure");
            indirect(gpu, pass, "continueParallelPressure", 0);
            indirect(gpu, pass, "resumeGpuPressure", 4);
        } else {
            indirect(gpu, pass, "solveGpuPressure", 4);
        }
        direct(gpu, pass, "updateGpuDispatch");
        indirect(gpu, pass, "project", 1);
        indirect(gpu, pass, "measureDivergence", 0);
        extend(gpu, pass, core.referenceNumerics);
        if (core.referenceNumerics || core._constrainSnapshot) {
            indirect(gpu, pass, "constrainSnapshots", 1);
        }
        indirect(gpu, pass, "gridToParticles", 3);
        if (core._removalEnabled) {
            if (core._extremeRemoval) {
                indirect(gpu, pass, "countExtremeGroups", 3);
                indirect(gpu, pass, "chooseExtremeThreshold", 4);
            }
            indirect(gpu, pass, "scanSurvivors", 3);
            indirect(gpu, pass, "scanSurvivorGroups", 4);
            indirect(gpu, pass, "packSurvivors", 3);
            indirect(gpu, pass, "commitSurvivors", 3);
        }
        direct(gpu, pass, "finishGpuSubstep");
    }
    direct(gpu, pass, "finishGpuFrame");
    pass.setPipeline(gpu.pipelines.publishGpuParticles!);
    pass.setBindGroup(0, gpu.bindGroups.publishGpuParticles!);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(core.capacity / 128)));
    pass.end();
    if (span) {
        encoder.beginComputePass({ label: "flip-reference:gpu-frame-end", timestampWrites: span.end }).end();
    }
    commandEnd?.();
    const slot = gpu.slots.find((entry) => !entry.pending && !entry.mapping);
    if (slot) {
        encoder.copyBufferToBuffer(gpu.stateBuffer, 0, slot.buffer, 0, STATE_BYTES);
        encoder.copyBufferToBuffer(core._pressureBuffer, (core._cells + core._reductionGroups) * 16, slot.buffer, 128, 32);
        encoder.copyBufferToBuffer(core._listBuffer, (core._cells + core.capacity) * 4, slot.buffer, 160, 32);
        encoder.copyBufferToBuffer(gpu.publishedDraw, 0, slot.buffer, 192, 16);
        slot.pending = true;
        slot.sequence = ++gpu.sequence;
    }
}

function statusError(bits: number, control: Float32Array): Error {
    const reasons: string[] = [];
    if (bits & 1) {
        reasons.push("non-finite/out-of-domain particles or invalid compaction");
    }
    if (bits & 2) {
        reasons.push("unresolved solid penetration");
    }
    if (bits & 4) {
        reasons.push("advection subdivision budget exhausted");
    }
    if (bits & 8) {
        reasons.push("advection left the extended velocity band");
    }
    if (bits & 16) {
        reasons.push(`pressure solve failed after ${control[6]} iterations (true relative L2 ${Math.sqrt(control[1]! / Math.max(control[2]!, 1e-30))}, code ${control[7]})`);
    }
    if (bits & 32) {
        reasons.push("closed liquid component has incompatible moving-solid flux");
    }
    if (bits & 64) {
        reasons.push("timestep subdivision cannot make finite progress");
    }
    return new Error(`[FLIP Reference] ${reasons.join("; ")}. Last completed water state retained; reset to resume.`);
}

/** @internal Only call after the copied encoders are submitted. Interactive callers never await this. */
export async function collectFlipReferenceGpuStatus(gpu: FlipReferenceGpuRuntime): Promise<void> {
    if (gpu.disposed) {
        return;
    }
    for (const slot of gpu.slots) {
        if (!slot.pending) {
            continue;
        }
        slot.pending = false;
        slot.mapping = slot.buffer
            .mapAsync(GPUMapMode.READ)
            .then(() => {
                if (gpu.disposed || slot.sequence < gpu.publishedSequence) {
                    return;
                }
                const mapped = slot.buffer.getMappedRange();
                const words = new Uint32Array(mapped);
                const floats = new Float32Array(mapped);
                gpu.telemetryError = null;
                gpu.publishedSequence = slot.sequence;
                gpu.publishedCount = words[49]!;
                gpu.core._count = words[0]!;
                gpu.core.elapsedSeconds = floats[16]!;
                gpu.timestepDiagnostics = { deferredSeconds: 0, droppedSeconds: floats[17]!, saturated: floats[17]! > 0 };
                gpu.core.totalRemovedInsideSolids = words[14]!;
                gpu.core.totalRemovedExtremeVelocities = words[15]!;
                gpu.core._controlReadback.set(floats.subarray(32, 40));
                gpu.core._statusReadback.set(words.subarray(40, 48));
                if (words[1] !== 0) {
                    gpu.error = statusError(words[1]!, floats.subarray(32, 40));
                }
            })
            .catch((error: unknown) => {
                if (!gpu.disposed && slot.sequence >= gpu.publishedSequence) {
                    gpu.publishedSequence = slot.sequence;
                    if (!gpu.telemetryError) {
                        console.warn("[FLIP Reference] diagnostic readback failed; simulation remains GPU-controlled.", error);
                    }
                    gpu.telemetryError = new Error("[FLIP Reference] diagnostic readback failed.", { cause: error });
                }
            })
            .finally(() => {
                if (slot.buffer.mapState === "mapped") {
                    slot.buffer.unmap();
                }
                slot.mapping = null;
            });
    }
    for (const slot of gpu.slots) {
        if (slot.mapping) {
            await slot.mapping;
        }
    }
}

/** @internal No callbacks can submit new physics, so already-submitted work needs no host fence here. */
export function disposeFlipReferenceGpuRuntime(gpu: FlipReferenceGpuRuntime): void {
    if (gpu.disposed) {
        return;
    }
    gpu.disposed = true;
    for (const buffer of gpu.buffers) {
        buffer.destroy();
    }
}
