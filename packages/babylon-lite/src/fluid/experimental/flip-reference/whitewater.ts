import { BU } from "../../../engine/gpu-flags.js";
import type { FluidAllocationResource, FluidDeviceLimitsSnapshot } from "../../core/allocation-plan.js";
import { FOAM_BYTES, foamActiveStateBytes, type DiffuseParticleCounts, type DiffusePool, type FluidProfiler, type FoamConfig } from "../../core/sim-common.js";
import type { FlipReferenceGpuRuntime } from "./gpu-runtime.js";
import type { FlipReferenceSimulation } from "./types.js";
import { FLIP_REFERENCE_WHITEWATER_BINDINGS, flipReferenceWhitewaterWgsl } from "./whitewater-shaders.js";

const FIELD_BYTES_PER_CELL = 32;
const WORKING_DISPATCH_BYTES = 36;
const PUBLISHED_DISPATCH_BYTES = 12;
const DRAW_BYTES = 16;
const TELEMETRY_BYTES = 32;
const TELEMETRY_SLOTS = 2;
const WORKGROUP_SIZE = 64;
const DEFAULT_MAX_WORKGROUPS = 65535;

interface WhitewaterLimits extends FluidDeviceLimitsSnapshot {
    readonly maxComputeWorkgroupsPerDimension?: number;
}

interface ResolvedFoamConfig {
    readonly activeParticles: boolean;
    readonly generateSpray: boolean;
    readonly generateFoam: boolean;
    readonly generateBubbles: boolean;
    readonly kTa: number;
    readonly kWc: number;
    readonly kTurb: number;
    readonly energySpeedMin: number;
    readonly energySpeedMax: number;
    readonly curvatureMin: number;
    readonly curvatureMax: number;
    readonly turbulenceMin: number;
    readonly turbulenceMax: number;
    readonly foamLayerDepth: number;
    readonly sprayDrag: number;
    readonly kb: number;
    readonly kd: number;
    readonly rv?: number;
    readonly tMin: number;
    readonly tMax: number;
    readonly poolScale: number;
    readonly poolCapMax?: number;
}

interface WhitewaterTelemetrySlot {
    readonly buffer: GPUBuffer;
    pending: boolean;
    mapping: Promise<void> | null;
    sequence: number;
}

/** @internal Pure state for the optional Reference whitewater module. */
export interface FlipReferenceWhitewater {
    readonly pool: DiffusePool;
    readonly bytes: number;
    readonly ready: Promise<void>;
    error: Error | null;
    lastEncoder: GPUCommandEncoder | null;
    /** @internal */
    readonly _core: FlipReferenceSimulation;
    /** @internal */
    readonly _gpu: FlipReferenceGpuRuntime;
    /** @internal */
    readonly _particleCapacity: number;
    /** @internal */
    readonly _cells: number;
    /** @internal */
    readonly _paramsBuffer: GPUBuffer;
    /** @internal */
    readonly _workingState: GPUBuffer;
    /** @internal */
    readonly _publishedState: GPUBuffer;
    /** @internal */
    readonly _buffers: readonly GPUBuffer[];
    /** @internal */
    readonly _pipelines: Readonly<Record<string, GPUComputePipeline>>;
    /** @internal */
    readonly _bindGroups: Readonly<Record<string, GPUBindGroup>>;
    /** @internal */
    readonly _telemetrySlots: readonly WhitewaterTelemetrySlot[];
    /** @internal */
    _config: ResolvedFoamConfig;
    /** @internal */
    _counts: DiffuseParticleCounts | undefined;
    /** @internal */
    _sequence: number;
    /** @internal */
    _publishedSequence: number;
    /** @internal */
    _disposed: boolean;
}

export interface FlipReferenceWhitewaterPlan {
    readonly capacity: number;
    readonly resources: FluidAllocationResource[];
    readonly errors: string[];
}

function finite(value: number | undefined, fallback: number, name: string, errors: string[], minimum = -Infinity, maximum = Infinity): number {
    const resolved = value ?? fallback;
    const f32 = Math.fround(resolved);
    if (!Number.isFinite(resolved) || !Number.isFinite(f32) || (resolved !== 0 && f32 === 0) || resolved < minimum || resolved > maximum) {
        errors.push(`${name} must be finite and in [${minimum}, ${maximum}].`);
        return fallback;
    }
    return resolved;
}

function booleanValue(value: boolean | undefined, fallback: boolean, name: string, errors: string[]): boolean {
    if (value !== undefined && typeof value !== "boolean") {
        errors.push(`${name} must be boolean.`);
        return fallback;
    }
    return value ?? fallback;
}

function resolveConfig(config: FoamConfig, errors: string[]): ResolvedFoamConfig {
    const energySpeedMin = finite(config.energySpeedMin, Math.sqrt(0.5), "foam.energySpeedMin", errors, 0);
    const energySpeedMax = finite(config.energySpeedMax, Math.sqrt(40), "foam.energySpeedMax", errors, 0);
    const curvatureMin = finite(config.curvatureMin, 0.05, "foam.curvatureMin", errors, 0);
    const curvatureMax = finite(config.curvatureMax, 1.5, "foam.curvatureMax", errors, 0);
    const turbulenceMin = finite(config.turbulenceMin, 0.1, "foam.turbulenceMin", errors, 0);
    const turbulenceMax = finite(config.turbulenceMax, 2.5, "foam.turbulenceMax", errors, 0);
    const tMin = finite(config.tMin, 0.3, "foam.tMin", errors, 0);
    const tMax = finite(config.tMax, 2, "foam.tMax", errors, 0);
    if (!(Math.fround(energySpeedMax) > Math.fround(energySpeedMin))) {
        errors.push("foam.energySpeedMax must be greater than foam.energySpeedMin.");
    }
    if (!(Math.fround(curvatureMax) > Math.fround(curvatureMin))) {
        errors.push("foam.curvatureMax must be greater than foam.curvatureMin.");
    }
    if (!(Math.fround(turbulenceMax) > Math.fround(turbulenceMin))) {
        errors.push("foam.turbulenceMax must be greater than foam.turbulenceMin.");
    }
    if (!(tMax >= tMin) || !(tMax > 0)) {
        errors.push("foam.tMax must be positive and at least foam.tMin.");
    }
    let poolCapMax: number | undefined;
    if (config.poolCapMax !== undefined) {
        if (!Number.isSafeInteger(config.poolCapMax) || config.poolCapMax < 1024) {
            errors.push("foam.poolCapMax must be a safe integer of at least 1024.");
        } else {
            poolCapMax = config.poolCapMax;
        }
    }
    let rv: number | undefined;
    if (config.rv !== undefined) {
        rv = finite(config.rv, 1, "foam.rv", errors, Number.MIN_VALUE);
    }
    return {
        activeParticles: booleanValue(config.activeParticles, true, "foam.activeParticles", errors),
        generateSpray: booleanValue(config.generateSpray, true, "foam.generateSpray", errors),
        generateFoam: booleanValue(config.generateFoam, true, "foam.generateFoam", errors),
        generateBubbles: booleanValue(config.generateBubbles, true, "foam.generateBubbles", errors),
        kTa: finite(config.kTa, 40, "foam.kTa", errors, 0),
        kWc: finite(config.kWc, 40, "foam.kWc", errors, 0),
        kTurb: finite(config.kTurb, 0, "foam.kTurb", errors, 0),
        energySpeedMin,
        energySpeedMax,
        curvatureMin,
        curvatureMax,
        turbulenceMin,
        turbulenceMax,
        foamLayerDepth: finite(config.foamLayerDepth, 0, "foam.foamLayerDepth", errors, 0, 4),
        sprayDrag: finite(config.sprayDrag, 0, "foam.sprayDrag", errors, 0),
        kb: finite(config.kb, 0.8, "foam.kb", errors, 0),
        kd: finite(config.kd, 0.5, "foam.kd", errors, 0, 1),
        rv,
        tMin,
        tMax,
        poolScale: finite(config.poolScale, 3, "foam.poolScale", errors, Number.MIN_VALUE),
        poolCapMax,
    };
}

function validLimit(value: number | undefined, fallback: number, name: string, errors: string[]): number {
    if (value === undefined) {
        return fallback;
    }
    if (!Number.isSafeInteger(value) || value <= 0) {
        errors.push(`${name} must be a positive safe integer.`);
        return fallback;
    }
    return value;
}

function maximumStateCapacity(byteLimit: number): number {
    if (!Number.isFinite(byteLimit)) {
        return Number.MAX_SAFE_INTEGER;
    }
    let low = 0;
    let high = Math.floor(byteLimit / 4);
    while (low < high) {
        const middle = Math.ceil((low + high) / 2);
        if (foamActiveStateBytes(middle) <= byteLimit) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

/** @internal Plans the exact optional allocations used by {@link createFlipReferenceWhitewater}. */
export function planFlipReferenceWhitewater(particleCapacity: number, cells: number, config: FoamConfig, limits: FluidDeviceLimitsSnapshot = {}): FlipReferenceWhitewaterPlan {
    const errors: string[] = [];
    if (!Number.isSafeInteger(particleCapacity) || particleCapacity < 0) {
        errors.push("particleCapacity must be a non-negative safe integer.");
    }
    if (!Number.isSafeInteger(cells) || cells <= 0) {
        errors.push("cells must be a positive safe integer.");
    }
    const resolved = resolveConfig(config, errors);
    const storageLimit = validLimit(limits.maxStorageBufferBindingSize, Number.MAX_SAFE_INTEGER, "maxStorageBufferBindingSize", errors);
    const bufferLimit = validLimit(limits.maxBufferSize, Number.MAX_SAFE_INTEGER, "maxBufferSize", errors);
    const dispatchLimit = validLimit((limits as WhitewaterLimits).maxComputeWorkgroupsPerDimension, DEFAULT_MAX_WORKGROUPS, "maxComputeWorkgroupsPerDimension", errors);
    const allocationLimit = Math.min(storageLimit, bufferLimit);
    const requested = Math.max(1024, Math.round(Math.max(0, particleCapacity) * resolved.poolScale));
    const configured = Math.min(requested, resolved.poolCapMax ?? Number.MAX_SAFE_INTEGER);
    const maximumPoolCapacity = Math.floor(allocationLimit / 32);
    const maximumListCapacity = maximumStateCapacity(allocationLimit);
    const maximumDispatchCapacity = dispatchLimit * dispatchLimit * WORKGROUP_SIZE;
    const maximumCapacity = Math.min(maximumPoolCapacity, maximumListCapacity, maximumDispatchCapacity);
    const capacity = Math.max(1024, Math.min(configured, maximumCapacity));
    if (maximumCapacity < 1024) {
        errors.push("Reference whitewater requires device limits large enough for the minimum 1024-slot pool.");
    }
    const resources: FluidAllocationResource[] = [
        { name: "Reference whitewater parameters", kind: "buffer", binding: "uniform", bytes: FOAM_BYTES },
        { name: "Reference whitewater fields", kind: "buffer", binding: "storage", bytes: Math.max(0, cells) * FIELD_BYTES_PER_CELL },
        { name: "Reference whitewater working pool", kind: "buffer", binding: "storage", bytes: capacity * 32 },
        { name: "Reference whitewater working state", kind: "buffer", binding: "storage", bytes: foamActiveStateBytes(capacity) },
        { name: "Reference whitewater working dispatch", kind: "buffer", binding: "storage", bytes: WORKING_DISPATCH_BYTES },
        { name: "Reference whitewater published pool", kind: "buffer", binding: "storage", bytes: capacity * 32 },
        { name: "Reference whitewater published state", kind: "buffer", binding: "storage", bytes: foamActiveStateBytes(capacity) },
        { name: "Reference whitewater published draw", kind: "buffer", binding: "storage", bytes: DRAW_BYTES },
        { name: "Reference whitewater published dispatch", kind: "buffer", binding: "storage", bytes: PUBLISHED_DISPATCH_BYTES },
        ...Array.from({ length: TELEMETRY_SLOTS }, (_, index): FluidAllocationResource => ({
            name: `Reference whitewater telemetry ${index}`,
            kind: "buffer",
            binding: "none",
            bytes: TELEMETRY_BYTES,
        })),
    ];
    for (const resource of resources) {
        if (!Number.isSafeInteger(resource.bytes) || resource.bytes <= 0 || resource.bytes > bufferLimit) {
            errors.push(`${resource.name} requires ${resource.bytes} bytes, exceeding the device buffer limit.`);
        } else if (resource.binding === "storage" && resource.bytes > storageLimit) {
            errors.push(`${resource.name} requires ${resource.bytes} bytes, exceeding the device storage-binding limit.`);
        }
    }
    const maximumInvocationCount = dispatchLimit * dispatchLimit * WORKGROUP_SIZE;
    if (particleCapacity > maximumInvocationCount) {
        errors.push("Reference whitewater liquid-particle dispatch exceeds the device workgroup limit.");
    }
    if (cells > maximumInvocationCount) {
        errors.push("Reference whitewater field dispatch exceeds the device workgroup limit.");
    }
    return { capacity, resources, errors };
}

function packConfig(config: ResolvedFoamConfig, particleRadius: number): ArrayBuffer {
    const data = new ArrayBuffer(FOAM_BYTES);
    const f32 = new Float32Array(data);
    const u32 = new Uint32Array(data);
    f32.set([0.5, 4, 0.05, 1.5, 0.25, 20, config.kTa, config.kWc, config.kb, config.kd, config.rv ?? particleRadius, 0, config.tMin, config.tMax]);
    f32.set([config.kTurb, config.energySpeedMin, config.energySpeedMax, config.sprayDrag], 16);
    f32.set([config.turbulenceMin, config.turbulenceMax, config.curvatureMin, config.curvatureMax], 20);
    f32[24] = config.foamLayerDepth;
    u32[28] = config.generateSpray ? 1 : 0;
    u32[29] = config.generateFoam ? 1 : 0;
    u32[30] = config.generateBubbles ? 1 : 0;
    return data;
}

function errorValue(error: unknown): Error {
    return error instanceof Error ? error : new Error(String(error));
}

function dispatchDimensions(count: number, limit: number): [number, number] {
    const groups = Math.ceil(count / WORKGROUP_SIZE);
    const x = Math.min(groups, limit);
    return [x, Math.max(1, Math.ceil(groups / Math.max(1, x)))];
}

/** @internal Allocates, initializes, and compiles an isolated Reference whitewater module. */
export function createFlipReferenceWhitewater(core: FlipReferenceSimulation, gpu: FlipReferenceGpuRuntime, particleCapacity: number, config: FoamConfig): FlipReferenceWhitewater {
    if (core._disposed || gpu.disposed || gpu.core !== core) {
        throw new Error("[FLIP Reference] cannot create whitewater for an unavailable GPU runtime.");
    }
    const device = core._device;
    if (
        device.limits.maxStorageBuffersPerShaderStage < 8 ||
        device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE ||
        device.limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE
    ) {
        throw new Error("[FLIP Reference] whitewater requires eight storage bindings and 64-invocation compute workgroups.");
    }
    const plan = planFlipReferenceWhitewater(particleCapacity, core._cells, config, device.limits);
    if (plan.errors.length) {
        throw new RangeError("[FLIP Reference] " + plan.errors.join(" "));
    }
    const configErrors: string[] = [];
    const resolved = resolveConfig(config, configErrors);
    if (configErrors.length) {
        throw new RangeError("[FLIP Reference] " + configErrors.join(" "));
    }
    const buffers: GPUBuffer[] = [];
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    let scopesOpen = true;
    try {
        const storage = BU.STORAGE | BU.COPY_DST;
        for (let index = 0; index < plan.resources.length; index++) {
            const resource = plan.resources[index]!;
            let usage: GPUBufferUsageFlags;
            if (resource.binding === "uniform") {
                usage = BU.UNIFORM | BU.COPY_DST;
            } else if (resource.binding === "none") {
                usage = BU.MAP_READ | BU.COPY_DST;
            } else {
                usage = storage;
                if (index === 4 || index === 7 || index === 8) {
                    usage |= BU.INDIRECT;
                }
                if (index === 3 || index === 5 || index === 6 || index === 7) {
                    usage |= BU.COPY_SRC;
                }
            }
            buffers.push(device.createBuffer({ label: resource.name, size: resource.bytes, usage }));
        }
        const [paramsBuffer, fieldBuffer, workingPool, workingState, workingDispatch, publishedPool, publishedState, publishedDraw, publishedDispatch] = buffers;
        const bindingBuffers = new Map<number, GPUBuffer>([
            [0, core._uniformBuffer],
            [1, core._positionBuffer],
            [2, core._velocityBuffer],
            [4, core._faceBuffer],
            [5, core._cellBuffer],
            [7, core._solidBuffer],
            [11, gpu.stateBuffer],
            [12, paramsBuffer!],
            [13, fieldBuffer!],
            [14, workingPool!],
            [15, workingState!],
            [16, workingDispatch!],
            [17, publishedPool!],
            [18, publishedState!],
            [19, publishedDraw!],
            [20, publishedDispatch!],
        ]);
        const module = device.createShaderModule({ label: "flip-reference:whitewater", code: flipReferenceWhitewaterWgsl() });
        const pipelines: Record<string, GPUComputePipeline> = {};
        const bindGroups: Record<string, GPUBindGroup> = {};
        for (const [entryPoint, bindings] of Object.entries(FLIP_REFERENCE_WHITEWATER_BINDINGS)) {
            const pipeline = device.createComputePipeline({ label: `flip-reference:whitewater:${entryPoint}`, layout: "auto", compute: { module, entryPoint } });
            pipelines[entryPoint] = pipeline;
            bindGroups[entryPoint] = device.createBindGroup({
                label: `flip-reference:whitewater:${entryPoint}`,
                layout: pipeline.getBindGroupLayout(0),
                entries: bindings.map((binding) => ({ binding, resource: { buffer: bindingBuffers.get(binding)! } })),
            });
        }
        const pool: DiffusePool = {
            buffer: publishedPool!,
            headBuffer: publishedState!,
            capacity: plan.capacity,
            activeIndices: publishedState!,
            activeIndicesOffset: 256,
            drawIndirect: publishedDraw!,
            get counts() {
                return ww._counts;
            },
        };
        const allocation = device.popErrorScope();
        const validation = device.popErrorScope();
        scopesOpen = false;
        const telemetrySlots = buffers.slice(9).map((buffer) => ({ buffer, pending: false, mapping: null, sequence: 0 }));
        const ready = Promise.all([core._ready, gpu.ready, module.getCompilationInfo(), allocation, validation]).then(([, , info, outOfMemory, invalid]) => {
            const shaderErrors = info.messages.filter((message) => message.type === "error");
            if (core._error || gpu.error || outOfMemory || invalid || shaderErrors.length) {
                throw (
                    core._error ??
                    gpu.error ??
                    new Error(
                        `[FLIP Reference] whitewater initialization failed: ${outOfMemory?.message ?? invalid?.message ?? shaderErrors.map((message) => message.message).join("\n")}`
                    )
                );
            }
        });
        const ww: FlipReferenceWhitewater = {
            pool,
            bytes: plan.resources.reduce((sum, resource) => sum + resource.bytes, 0),
            ready,
            error: null,
            lastEncoder: null,
            _core: core,
            _gpu: gpu,
            _particleCapacity: particleCapacity,
            _cells: core._cells,
            _paramsBuffer: paramsBuffer!,
            _workingState: workingState!,
            _publishedState: publishedState!,
            _buffers: buffers,
            _pipelines: pipelines,
            _bindGroups: bindGroups,
            _telemetrySlots: telemetrySlots,
            _config: resolved,
            _counts: undefined,
            _sequence: 0,
            _publishedSequence: 0,
            _disposed: false,
        };
        device.queue.writeBuffer(paramsBuffer!, 0, packConfig(resolved, core.particleRadius));
        const encoder = device.createCommandEncoder({ label: "flip-reference:whitewater-init" });
        for (const buffer of buffers.slice(1, 9)) {
            encoder.clearBuffer(buffer);
        }
        const pass = encoder.beginComputePass({ label: "flip-reference:whitewater-init" });
        pass.setPipeline(pipelines.initializeFlipReferenceWhitewater!);
        pass.setBindGroup(0, bindGroups.initializeFlipReferenceWhitewater!);
        const [x, y] = dispatchDimensions(plan.capacity, device.limits.maxComputeWorkgroupsPerDimension);
        pass.dispatchWorkgroups(x, y);
        pass.end();
        device.queue.submit([encoder.finish()]);
        device.queue.writeBuffer(workingState!, 12 * 4, new Uint32Array([resolved.activeParticles ? 1 : 0, device.limits.maxComputeWorkgroupsPerDimension]));
        void ready.catch((error: unknown) => {
            ww.error = errorValue(error);
        });
        return ww;
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

/** @internal Applies live numerical settings. Capacity changes require dispose/create and clear the pool. */
export function configureFlipReferenceWhitewater(ww: FlipReferenceWhitewater, config: FoamConfig): void {
    if (ww._disposed) {
        throw new Error("[FLIP Reference] cannot configure disposed whitewater.");
    }
    if (ww.error) {
        throw ww.error;
    }
    const plan = planFlipReferenceWhitewater(ww._particleCapacity, ww._cells, config, ww._core._device.limits);
    if (plan.errors.length) {
        throw new RangeError("[FLIP Reference] " + plan.errors.join(" "));
    }
    if (plan.capacity !== ww.pool.capacity) {
        throw new RangeError("[FLIP Reference] whitewater pool-size changes require recreation.");
    }
    const errors: string[] = [];
    const resolved = resolveConfig(config, errors);
    if (errors.length) {
        throw new RangeError("[FLIP Reference] " + errors.join(" "));
    }
    ww._config = resolved;
    ww._core._device.queue.writeBuffer(ww._paramsBuffer, 0, packConfig(resolved, ww._core.particleRadius));
    ww._core._device.queue.writeBuffer(ww._workingState, 12 * 4, new Uint32Array([resolved.activeParticles ? 1 : 0]));
}

function direct(ww: FlipReferenceWhitewater, pass: GPUComputePassEncoder, entryPoint: string): void {
    pass.setPipeline(ww._pipelines[entryPoint]!);
    pass.setBindGroup(0, ww._bindGroups[entryPoint]!);
    pass.dispatchWorkgroups(1);
}

function indirect(ww: FlipReferenceWhitewater, pass: GPUComputePassEncoder, entryPoint: string, buffer: GPUBuffer, offset: number): void {
    pass.setPipeline(ww._pipelines[entryPoint]!);
    pass.setBindGroup(0, ww._bindGroups[entryPoint]!);
    pass.dispatchWorkgroupsIndirect(buffer, offset);
}

/** @internal Records once after the liquid frame into the same encoder; it never submits or waits. */
export function recordFlipReferenceWhitewater(ww: FlipReferenceWhitewater, encoder: GPUCommandEncoder, profiler: FluidProfiler | null = null): void {
    if (ww._disposed) {
        throw new Error("[FLIP Reference] cannot record disposed whitewater.");
    }
    if (ww.error) {
        throw ww.error;
    }
    if (ww._core._disposed || ww._gpu.disposed || ww._gpu.core !== ww._core) {
        throw new Error("[FLIP Reference] whitewater is bound to an unavailable liquid runtime.");
    }
    if (ww.lastEncoder === encoder) {
        throw new Error("[FLIP Reference] record whitewater only once per command encoder.");
    }
    void collectFlipReferenceWhitewaterStatus(ww);
    ww.lastEncoder = encoder;
    const workingDispatch = ww._buffers[4]!;
    const publishedDispatch = ww._buffers[8]!;
    const pass = encoder.beginComputePass({ label: "flip-reference:whitewater", timestampWrites: profiler?.pass("Foam gen") });
    direct(ww, pass, "prepareFlipReferenceWhitewater");
    indirect(ww, pass, "prepareFlipReferenceWhitewaterField", workingDispatch, 0);
    indirect(ww, pass, "updateFlipReferenceWhitewater", workingDispatch, 12);
    indirect(ww, pass, "emitFlipReferenceWhitewater", workingDispatch, 24);
    direct(ww, pass, "finishFlipReferenceWhitewater");
    indirect(ww, pass, "publishFlipReferenceWhitewater", publishedDispatch, 0);
    pass.end();
    const slot = ww._telemetrySlots.find((entry) => !entry.pending && !entry.mapping);
    if (slot) {
        encoder.copyBufferToBuffer(ww._publishedState, 8 * 4, slot.buffer, 0, 16);
        encoder.copyBufferToBuffer(ww._workingState, 6 * 4, slot.buffer, 16, 8);
        encoder.copyBufferToBuffer(ww._gpu.stateBuffer, 1 * 4, slot.buffer, 24, 4);
        encoder.copyBufferToBuffer(ww._gpu.stateBuffer, 19 * 4, slot.buffer, 28, 4);
        slot.pending = true;
        slot.sequence = ++ww._sequence;
    }
}

/** @internal Call only after the encoder containing the telemetry copy has been submitted. */
export async function collectFlipReferenceWhitewaterStatus(ww: FlipReferenceWhitewater): Promise<void> {
    if (ww._disposed) {
        return;
    }
    for (const slot of ww._telemetrySlots) {
        if (!slot.pending) {
            continue;
        }
        slot.pending = false;
        slot.mapping = slot.buffer
            .mapAsync(GPUMapMode.READ)
            .then(() => {
                if (ww._disposed || slot.sequence < ww._publishedSequence) {
                    return;
                }
                const mapped = slot.buffer.getMappedRange();
                const words = new Uint32Array(mapped);
                const floats = new Float32Array(mapped);
                const total = words[0]!;
                const spray = words[1]!;
                const foam = words[2]!;
                const bubble = words[3]!;
                ww._publishedSequence = slot.sequence;
                if (words[6] === 0 && floats[7]! > 0) {
                    if (total > ww.pool.capacity || total !== spray + foam + bubble) {
                        throw new Error("[FLIP Reference] whitewater counters violate their capacity/type accounting.");
                    }
                    ww._counts = { total, spray, foam, bubble, capacity: ww.pool.capacity };
                }
            })
            .catch((error: unknown) => {
                if (!ww._disposed && slot.sequence >= ww._publishedSequence) {
                    ww._publishedSequence = slot.sequence;
                    ww._counts = undefined;
                    console.warn("[FLIP Reference] whitewater telemetry readback failed; physics remains GPU-controlled.", error);
                }
            })
            .finally(() => {
                if (slot.buffer.mapState === "mapped") {
                    slot.buffer.unmap();
                }
                slot.mapping = null;
            });
    }
    for (const slot of ww._telemetrySlots) {
        if (slot.mapping) {
            await slot.mapping;
        }
    }
}

/** @internal Releases only resources owned by this optional module. */
export function disposeFlipReferenceWhitewater(ww: FlipReferenceWhitewater): void {
    if (ww._disposed) {
        return;
    }
    ww._disposed = true;
    ww.lastEncoder = null;
    ww._counts = undefined;
    for (const buffer of ww._buffers) {
        buffer.destroy();
    }
}
