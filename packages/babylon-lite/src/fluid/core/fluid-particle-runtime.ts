import type { EngineContext } from "../../engine/engine.js";
import type { Task } from "../../frame-graph/task.js";
import { GpuReadbackPool } from "./gpu-readback.js";
import type { FluidSim } from "./sim-common.js";

export interface FluidParticleChannelRuntime {
    readonly buffer: GPUBuffer;
    readonly components: 1 | 4;
    readonly capacity: number;
    write(data: Float32Array, particleOffset: number): void;
    fill(value: number | readonly [number, number, number, number], start: number, count: number): void;
    read(particleOffset: number, particleCount: number): Promise<Float32Array>;
    dispose(): void;
}

export interface FluidAggregateSourceRuntime {
    readonly sim: FluidSim;
    readonly count?: number;
    readonly opacity: number;
    readonly alpha?: FluidParticleChannelRuntime;
    readonly color?: FluidParticleChannelRuntime;
}

export interface FluidParticleStreamRuntime {
    readonly positionBuffer: GPUBuffer;
    readonly alphaBuffer?: GPUBuffer;
    readonly count: number;
    readonly capacity: number;
}

export interface FluidAggregateRuntime {
    readonly sim: FluidSim;
    readonly task: Task;
    readonly stream: FluidParticleStreamRuntime;
    readonly colorBuffer: GPUBuffer;
    readonly usesParticleColor: boolean;
    readonly overflowed: boolean;
    setOnUpdate(callback: (() => void) | null): void;
    dispose(): void;
}

export interface FluidSpatialQueryRuntimeRequest {
    readonly key: string | number;
    readonly offset: number;
    readonly count: number;
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
    readonly sphere?: {
        readonly origin: readonly [number, number, number];
        readonly radius: number;
    };
}

export interface FluidSpatialQueryRuntimeResult {
    readonly key: string | number;
    readonly count: number;
}

export interface FluidSpatialQueryRuntime {
    readonly results: readonly FluidSpatialQueryRuntimeResult[];
    readonly status: "idle" | "pending" | "ready" | "failed";
    poll(): void;
    record(stream: FluidParticleStreamRuntime, requests: readonly FluidSpatialQueryRuntimeRequest[]): void;
    dispose(): void;
}

export interface FluidWheelTorqueQueryRuntimeOptions {
    readonly center: readonly [number, number, number];
    readonly axis: readonly [number, number, number];
    readonly verticalAxis: readonly [number, number, number];
    readonly driveAxis: readonly [number, number, number];
    readonly driveSide: -1 | 1;
    readonly axialHalfExtent: number;
    readonly radialMin: number;
    readonly radialMax: number;
    readonly speedThreshold: number;
    readonly maximumParticles: number;
}

export interface FluidWheelTorqueQueryRuntime {
    readonly torque: number;
    readonly status: "idle" | "pending" | "ready" | "failed";
    poll(): void;
    record(sim: FluidSim): void;
    dispose(): void;
}

export function createFluidParticleChannelRuntime(
    engine: EngineContext,
    options: {
        readonly capacity: number;
        readonly components: 1 | 4;
        readonly label?: string;
        readonly initialData?: Float32Array;
    }
): FluidParticleChannelRuntime {
    const capacity = Math.max(1, Math.floor(options.capacity));
    const components = options.components;
    const buffer = engine._device.createBuffer({
        label: options.label ?? "fluid-particle-channel",
        size: capacity * components * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (options.initialData) {
        if (options.initialData.length > capacity * components) {
            buffer.destroy();
            throw new RangeError("[fluid] initial particle-channel data exceeds its capacity.");
        }
        engine._device.queue.writeBuffer(buffer, 0, options.initialData);
    }
    return {
        buffer,
        components,
        capacity,
        write(data, particleOffset): void {
            const offset = Math.max(0, Math.floor(particleOffset));
            if (data.length % components !== 0 || offset * components + data.length > capacity * components) {
                throw new RangeError("[fluid] particle-channel write exceeds its capacity.");
            }
            engine._device.queue.writeBuffer(buffer, offset * components * 4, data);
        },
        fill(value, start, count): void {
            const first = Math.max(0, Math.floor(start));
            const length = Math.max(0, Math.floor(count));
            if (first + length > capacity) {
                throw new RangeError("[fluid] particle-channel fill exceeds its capacity.");
            }
            const data = new Float32Array(length * components);
            if (components === 1) {
                data.fill(typeof value === "number" ? value : value[0]);
            } else {
                const rgba = typeof value === "number" ? ([value, value, value, value] as const) : value;
                for (let index = 0; index < length; index++) {
                    data.set(rgba, index * 4);
                }
            }
            engine._device.queue.writeBuffer(buffer, first * components * 4, data);
        },
        async read(particleOffset, particleCount): Promise<Float32Array> {
            const offset = Math.max(0, Math.floor(particleOffset));
            const count = Math.min(capacity - offset, Math.max(0, Math.floor(particleCount)));
            if (count <= 0) {
                return new Float32Array(0);
            }
            const byteLength = count * components * 4;
            const readback = engine._device.createBuffer({
                label: `${options.label ?? "fluid-particle-channel"}-readback`,
                size: byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            try {
                const encoder = engine._device.createCommandEncoder({ label: "fluid-particle-channel-readback" });
                encoder.copyBufferToBuffer(buffer, offset * components * 4, readback, 0, byteLength);
                engine._device.queue.submit([encoder.finish()]);
                await engine._device.queue.onSubmittedWorkDone();
                await readback.mapAsync(GPUMapMode.READ);
                return new Float32Array(readback.getMappedRange()).slice();
            } finally {
                if (readback.mapState === "mapped") {
                    readback.unmap();
                }
                readback.destroy();
            }
        },
        dispose(): void {
            buffer.destroy();
        },
    };
}

export function adoptFluidParticleChannelRuntime(
    engine: EngineContext,
    buffer: GPUBuffer,
    options: { readonly capacity: number; readonly components: 1 | 4; readonly ownsBuffer?: boolean; readonly label?: string }
): FluidParticleChannelRuntime {
    const capacity = Math.max(1, Math.floor(options.capacity));
    const components = options.components;
    if (buffer.size < capacity * components * 4) {
        throw new RangeError("[fluid] adopted particle-channel buffer is smaller than its declared capacity.");
    }
    return {
        buffer,
        components,
        capacity,
        write(data, particleOffset): void {
            const offset = Math.max(0, Math.floor(particleOffset));
            if (data.length % components !== 0 || offset * components + data.length > capacity * components) {
                throw new RangeError("[fluid] particle-channel write exceeds its capacity.");
            }
            engine._device.queue.writeBuffer(buffer, offset * components * 4, data);
        },
        fill(value, start, count): void {
            const first = Math.max(0, Math.floor(start));
            const length = Math.max(0, Math.floor(count));
            if (first + length > capacity) {
                throw new RangeError("[fluid] particle-channel fill exceeds its capacity.");
            }
            const data = new Float32Array(length * components);
            if (components === 1) {
                data.fill(typeof value === "number" ? value : value[0]);
            } else {
                const rgba = typeof value === "number" ? ([value, value, value, value] as const) : value;
                for (let index = 0; index < length; index++) {
                    data.set(rgba, index * 4);
                }
            }
            engine._device.queue.writeBuffer(buffer, first * components * 4, data);
        },
        async read(particleOffset, particleCount): Promise<Float32Array> {
            const offset = Math.max(0, Math.floor(particleOffset));
            const count = Math.min(capacity - offset, Math.max(0, Math.floor(particleCount)));
            if (count <= 0) {
                return new Float32Array(0);
            }
            const byteLength = count * components * 4;
            const readback = engine._device.createBuffer({
                label: `${options.label ?? "adopted-fluid-particle-channel"}-readback`,
                size: byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            try {
                const encoder = engine._device.createCommandEncoder({ label: "adopted-fluid-particle-channel-readback" });
                encoder.copyBufferToBuffer(buffer, offset * components * 4, readback, 0, byteLength);
                engine._device.queue.submit([encoder.finish()]);
                await engine._device.queue.onSubmittedWorkDone();
                await readback.mapAsync(GPUMapMode.READ);
                return new Float32Array(readback.getMappedRange()).slice();
            } finally {
                if (readback.mapState === "mapped") {
                    readback.unmap();
                }
                readback.destroy();
            }
        },
        dispose(): void {
            if (options.ownsBuffer) {
                buffer.destroy();
            }
        },
    };
}

export function createFluidAggregateRuntime(engine: EngineContext, capacityInput: number, readSources: () => readonly FluidAggregateSourceRuntime[]): FluidAggregateRuntime {
    const capacity = Math.max(1, Math.floor(capacityInput));
    const positionBuffer = engine._device.createBuffer({
        label: "fluid-aggregate-position",
        size: capacity * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const debugBuffer = engine._device.createBuffer({
        label: "fluid-aggregate-debug",
        size: capacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const alphaBuffer = engine._device.createBuffer({
        label: "fluid-aggregate-alpha",
        size: capacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const colorBuffer = engine._device.createBuffer({
        label: "fluid-aggregate-color",
        size: capacity * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const alphaScratch = new Float32Array(capacity);
    const colorScratch = new Float32Array(capacity * 4);
    let count = 0;
    let particleRadius = 0.08;
    let surfaceSizeScale = 1;
    let surfaceThicknessScale = 1;
    let rejectSparse = false;
    let usesParticleColor = false;
    let overflowed = false;
    let onUpdate: (() => void) | null = null;
    const virtualSim: FluidSim = {
        get count(): number {
            return count;
        },
        get activeCount(): number {
            return count;
        },
        get renderCount(): number {
            return count;
        },
        get particleRadius(): number {
            return particleRadius;
        },
        get surfaceSizeScale(): number {
            return surfaceSizeScale;
        },
        get surfaceThicknessScale(): number {
            return surfaceThicknessScale;
        },
        get surfaceRejectSparseMarkers(): boolean {
            return rejectSparse;
        },
        positionBuffer,
        velocityBuffer: positionBuffer,
        debugBuffer,
        debugNorm: 1,
        gpuBytes: positionBuffer.size + debugBuffer.size + alphaBuffer.size + colorBuffer.size,
        step: () => undefined,
        reset: () => undefined,
        setParam: () => undefined,
        setSceneSdf: () => undefined,
        setEmitters: () => undefined,
        setFlow: () => undefined,
        updateFlowEmitter: () => undefined,
        setSpawn: () => undefined,
        setForceField: () => undefined,
        dispose: () => undefined,
    };
    const stream: FluidParticleStreamRuntime = {
        positionBuffer,
        alphaBuffer,
        get count(): number {
            return count;
        },
        capacity,
    };
    const task: Task = {
        name: "fluid-aggregate-stream",
        engine,
        _passes: [],
        record: () => undefined,
        execute(): number {
            const sources = readSources();
            const encoder = engine._currentEncoder;
            let offset = 0;
            usesParticleColor = false;
            overflowed = false;
            const first = sources[0];
            if (first) {
                particleRadius = first.sim.particleRadius;
                surfaceSizeScale = first.sim.surfaceSizeScale ?? 1;
                surfaceThicknessScale = first.sim.surfaceThicknessScale ?? 1;
                rejectSparse = first.sim.surfaceRejectSparseMarkers ?? false;
            }
            for (const source of sources) {
                const available = source.sim.renderCount ?? source.sim.activeCount ?? source.sim.count;
                const requested = source.count === undefined ? available : Math.min(available, Math.max(0, Math.floor(source.count)));
                if (offset + requested > capacity) {
                    overflowed = true;
                }
                const sourceCount = Math.min(requested, capacity - offset);
                if (sourceCount <= 0) {
                    continue;
                }
                encoder.copyBufferToBuffer(source.sim.positionBuffer, 0, positionBuffer, offset * 16, sourceCount * 16);
                encoder.copyBufferToBuffer(source.sim.debugBuffer, 0, debugBuffer, offset * 4, sourceCount * 4);
                alphaScratch.fill(source.opacity, offset, offset + sourceCount);
                if (source.alpha) {
                    encoder.copyBufferToBuffer(source.alpha.buffer, 0, alphaBuffer, offset * 4, sourceCount * 4);
                }
                if (source.color) {
                    if (source.color.components !== 4) {
                        throw new TypeError("[fluid] render color channels must contain four components per particle.");
                    }
                    encoder.copyBufferToBuffer(source.color.buffer, 0, colorBuffer, offset * 16, sourceCount * 16);
                    usesParticleColor = true;
                } else {
                    for (let index = offset; index < offset + sourceCount; index++) {
                        colorScratch.set([1, 1, 1, 1], index * 4);
                    }
                }
                offset += sourceCount;
            }
            count = offset;
            if (count > 0) {
                engine._device.queue.writeBuffer(alphaBuffer, 0, alphaScratch, 0, count);
                engine._device.queue.writeBuffer(colorBuffer, 0, colorScratch, 0, count * 4);
            }
            onUpdate?.();
            return 0;
        },
        dispose: () => undefined,
    };
    return {
        sim: virtualSim,
        task,
        stream,
        colorBuffer,
        get usesParticleColor(): boolean {
            return usesParticleColor;
        },
        get overflowed(): boolean {
            return overflowed;
        },
        setOnUpdate(callback): void {
            onUpdate = callback;
        },
        dispose(): void {
            positionBuffer.destroy();
            debugBuffer.destroy();
            alphaBuffer.destroy();
            colorBuffer.destroy();
        },
    };
}

const SPATIAL_QUERY_BYTES = 64;
const SPATIAL_QUERY_WORKGROUP_SIZE = 256;
const SPATIAL_QUERY_SHADER = /* wgsl */ `
struct Query {
    aabbMin: vec4<f32>,
    aabbMax: vec4<f32>,
    originRadius: vec4<f32>,
    offset: u32,
    count: u32,
    output: u32,
    spherical: u32,
};
struct Params {
    queryCount: u32,
    useAlpha: u32,
    _padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> matches: array<u32, ${SPATIAL_QUERY_WORKGROUP_SIZE}>;
@compute @workgroup_size(${SPATIAL_QUERY_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>, @builtin(local_invocation_index) localIndex: u32) {
    let queryIndex = invocation.y;
    var matched = 0u;
    if (queryIndex < params.queryCount) {
        let query = queries[queryIndex];
        let localParticle = invocation.x;
        if (localParticle < query.count) {
            let particleIndex = query.offset + localParticle;
            if (params.useAlpha == 0u || alpha[particleIndex] > 0.001) {
                let position = positions[particleIndex].xyz;
                let inside = all(position >= query.aabbMin.xyz) && all(position <= query.aabbMax.xyz);
                let reached = query.spherical == 0u || distance(position, query.originRadius.xyz) <= query.originRadius.w;
                matched = select(0u, 1u, inside && reached);
            }
        }
    }
    matches[localIndex] = matched;
    workgroupBarrier();
    var stride = ${SPATIAL_QUERY_WORKGROUP_SIZE / 2}u;
    loop {
        if (localIndex < stride) {
            matches[localIndex] += matches[localIndex + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride /= 2u;
    }
    if (localIndex == 0u && queryIndex < params.queryCount && matches[0] > 0u) {
        atomicAdd(&results[queries[queryIndex].output], matches[0]);
    }
}`;

export function createFluidSpatialQueryRuntime(engine: EngineContext, maximumQueriesInput: number): FluidSpatialQueryRuntime {
    const maximumQueries = Math.max(1, Math.floor(maximumQueriesInput));
    const device = engine._device;
    const queryBuffer = device.createBuffer({
        label: "fluid-spatial-queries",
        size: maximumQueries * SPATIAL_QUERY_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const resultBuffer = device.createBuffer({
        label: "fluid-spatial-results",
        size: maximumQueries * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const paramsBuffer = device.createBuffer({
        label: "fluid-spatial-query-params",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const pipeline = device.createComputePipeline({
        label: "fluid-spatial-query",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "fluid-spatial-query", code: SPATIAL_QUERY_SHADER }),
            entryPoint: "main",
        },
    });
    let bindGroup: GPUBindGroup | null = null;
    let boundPosition: GPUBuffer | null = null;
    let boundAlpha: GPUBuffer | null = null;
    let latest: readonly FluidSpatialQueryRuntimeResult[] = [];
    let status: FluidSpatialQueryRuntime["status"] = "idle";
    let submittedGeneration = 0;
    let completedGeneration = -1;
    const readback = new GpuReadbackPool<readonly (string | number)[]>({
        device,
        label: "fluid-spatial-query-readback",
        slotCount: 3,
        byteLength: maximumQueries * 4,
        defaultPayload: [],
        onComplete: (view, slot) => {
            if (slot.generation < completedGeneration) {
                return;
            }
            const counts = new Uint32Array(view);
            latest = slot.payload.map((key, index) => ({ key, count: counts[index] ?? 0 }));
            completedGeneration = slot.generation;
            status = "ready";
        },
        onError: (error, slot) => {
            if (slot.generation >= completedGeneration) {
                latest = [];
                completedGeneration = slot.generation;
                status = "failed";
            }
            console.error("[fluid] spatial query readback failed", error);
        },
    });
    const runtime: FluidSpatialQueryRuntime = {
        get results(): readonly FluidSpatialQueryRuntimeResult[] {
            return latest;
        },
        get status(): FluidSpatialQueryRuntime["status"] {
            return status;
        },
        poll(): void {
            readback.pump();
        },
        record(stream, requests): void {
            readback.pump();
            if (requests.length === 0) {
                latest = [];
                completedGeneration = submittedGeneration++;
                status = "ready";
                return;
            }
            if (requests.length > maximumQueries) {
                throw new RangeError(`[fluid] ${requests.length} spatial queries exceed the ${maximumQueries} query capacity.`);
            }
            const slot = readback.acquire();
            if (!slot) {
                return;
            }
            const alpha = stream.alphaBuffer ?? stream.positionBuffer;
            if (!bindGroup || boundPosition !== stream.positionBuffer || boundAlpha !== alpha) {
                boundPosition = stream.positionBuffer;
                boundAlpha = alpha;
                bindGroup = device.createBindGroup({
                    label: "fluid-spatial-query",
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: stream.positionBuffer } },
                        { binding: 1, resource: { buffer: alpha } },
                        { binding: 2, resource: { buffer: queryBuffer } },
                        { binding: 3, resource: { buffer: resultBuffer } },
                        { binding: 4, resource: { buffer: paramsBuffer } },
                    ],
                });
            }
            const data = new ArrayBuffer(requests.length * SPATIAL_QUERY_BYTES);
            const floats = new Float32Array(data);
            const uints = new Uint32Array(data);
            let maximumCount = 0;
            requests.forEach((request, index) => {
                const word = (index * SPATIAL_QUERY_BYTES) / 4;
                floats.set(request.min, word);
                floats.set(request.max, word + 4);
                if (request.sphere) {
                    floats.set(request.sphere.origin, word + 8);
                    floats[word + 11] = request.sphere.radius;
                }
                const offset = Math.min(stream.count, Math.max(0, Math.floor(request.offset)));
                const count = Math.min(stream.count - offset, Math.max(0, Math.floor(request.count)));
                uints[word + 12] = offset;
                uints[word + 13] = count;
                uints[word + 14] = index;
                uints[word + 15] = request.sphere ? 1 : 0;
                maximumCount = Math.max(maximumCount, count);
            });
            device.queue.writeBuffer(queryBuffer, 0, data);
            device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([requests.length, stream.alphaBuffer ? 1 : 0, 0, 0]));
            const byteLength = requests.length * 4;
            const encoder = engine._currentEncoder;
            encoder.clearBuffer(resultBuffer, 0, byteLength);
            if (maximumCount > 0) {
                const pass = encoder.beginComputePass({ label: "fluid-spatial-query" });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(maximumCount / SPATIAL_QUERY_WORKGROUP_SIZE), requests.length);
                pass.end();
            }
            encoder.copyBufferToBuffer(resultBuffer, 0, slot.buffer, 0, byteLength);
            readback.submit(
                slot,
                submittedGeneration++,
                requests.map((request) => request.key),
                byteLength
            );
            status = "pending";
        },
        dispose(): void {
            readback.dispose();
            queryBuffer.destroy();
            resultBuffer.destroy();
            paramsBuffer.destroy();
        },
    };
    return runtime;
}

const WHEEL_TORQUE_WORKGROUP_SIZE = 256;
const WHEEL_TORQUE_FIXED_POINT = 256;
const WHEEL_TORQUE_SHADER = /* wgsl */ `
struct Params {
    centerMax: vec4<f32>,
    axisAxial: vec4<f32>,
    verticalMin: vec4<f32>,
    driveSpeed: vec4<f32>,
    limits: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> result: atomic<i32>;
@group(0) @binding(2) var<storage, read> speeds: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WHEEL_TORQUE_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let particleCount = u32(params.limits.x);
    if (i >= particleCount || i >= u32(params.limits.y)) { return; }
    let offset = positions[i].xyz - params.centerMax.xyz;
    let axial = dot(offset, params.axisAxial.xyz);
    if (abs(axial) > params.axisAxial.w) { return; }
    let perpendicular = offset - axial * params.axisAxial.xyz;
    let radius = length(perpendicular);
    if (radius < params.verticalMin.w || radius > params.centerMax.w) { return; }
    if (dot(perpendicular, params.verticalMin.xyz) <= 0.0) { return; }
    let driveDistance = dot(perpendicular, params.driveSpeed.xyz);
    if (params.limits.w * driveDistance <= 0.0) { return; }
    if (speeds[i] * params.limits.z <= params.driveSpeed.w) { return; }
    atomicAdd(&result, i32(driveDistance * ${WHEEL_TORQUE_FIXED_POINT}.0));
}`;

export function createFluidWheelTorqueQueryRuntime(engine: EngineContext, options: FluidWheelTorqueQueryRuntimeOptions): FluidWheelTorqueQueryRuntime {
    const device = engine._device;
    const resultBuffer = device.createBuffer({
        label: "fluid-wheel-torque-result",
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const paramsBuffer = device.createBuffer({
        label: "fluid-wheel-torque-params",
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const pipeline = device.createComputePipeline({
        label: "fluid-wheel-torque",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "fluid-wheel-torque", code: WHEEL_TORQUE_SHADER }),
            entryPoint: "main",
        },
    });
    let bindGroup: GPUBindGroup | null = null;
    let boundPosition: GPUBuffer | null = null;
    let boundSpeed: GPUBuffer | null = null;
    let latestTorque = 0;
    let status: FluidWheelTorqueQueryRuntime["status"] = "idle";
    let submittedGeneration = 0;
    let completedGeneration = -1;
    const readback = new GpuReadbackPool<void>({
        device,
        label: "fluid-wheel-torque-readback",
        slotCount: 3,
        byteLength: 4,
        defaultPayload: undefined,
        onComplete: (view, slot) => {
            if (slot.generation < completedGeneration) {
                return;
            }
            latestTorque = (new Int32Array(view)[0] ?? 0) / WHEEL_TORQUE_FIXED_POINT;
            completedGeneration = slot.generation;
            status = "ready";
        },
        onError: (error, slot) => {
            if (slot.generation >= completedGeneration) {
                completedGeneration = slot.generation;
                status = "failed";
            }
            console.error("[fluid] wheel torque readback failed", error);
        },
    });
    const baseParams = new Float32Array(20);
    baseParams.set(options.center, 0);
    baseParams[3] = options.radialMax;
    baseParams.set(options.axis, 4);
    baseParams[7] = options.axialHalfExtent;
    baseParams.set(options.verticalAxis, 8);
    baseParams[11] = options.radialMin;
    baseParams.set(options.driveAxis, 12);
    baseParams[15] = options.speedThreshold;
    baseParams[17] = options.maximumParticles;
    baseParams[19] = options.driveSide;

    return {
        get torque(): number {
            return latestTorque;
        },
        get status(): FluidWheelTorqueQueryRuntime["status"] {
            return status;
        },
        poll(): void {
            readback.pump();
        },
        record(sim): void {
            readback.pump();
            const slot = readback.acquire();
            if (!slot) {
                return;
            }
            if (!bindGroup || boundPosition !== sim.positionBuffer || boundSpeed !== sim.debugBuffer) {
                boundPosition = sim.positionBuffer;
                boundSpeed = sim.debugBuffer;
                bindGroup = device.createBindGroup({
                    label: "fluid-wheel-torque",
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: sim.positionBuffer } },
                        { binding: 1, resource: { buffer: resultBuffer } },
                        { binding: 2, resource: { buffer: sim.debugBuffer } },
                        { binding: 3, resource: { buffer: paramsBuffer } },
                    ],
                });
            }
            const particleCount = sim.renderCount ?? sim.activeCount ?? sim.count;
            if (particleCount <= 0) {
                latestTorque = 0;
                completedGeneration = submittedGeneration++;
                status = "ready";
                return;
            }
            const params = baseParams.slice();
            params[16] = particleCount;
            params[18] = sim.debugNorm;
            device.queue.writeBuffer(paramsBuffer, 0, params);
            const encoder = engine._currentEncoder;
            encoder.clearBuffer(resultBuffer, 0, 4);
            const pass = encoder.beginComputePass({ label: "fluid-wheel-torque" });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(Math.min(particleCount, options.maximumParticles) / WHEEL_TORQUE_WORKGROUP_SIZE));
            pass.end();
            encoder.copyBufferToBuffer(resultBuffer, 0, slot.buffer, 0, 4);
            readback.submit(slot, submittedGeneration++, undefined);
            status = "pending";
        },
        dispose(): void {
            readback.dispose();
            resultBuffer.destroy();
            paramsBuffer.destroy();
        },
    };
}

export function fluidSimParticleStream(sim: FluidSim): FluidParticleStreamRuntime {
    return {
        get positionBuffer(): GPUBuffer {
            return sim.positionBuffer;
        },
        get count(): number {
            return sim.renderCount ?? sim.activeCount ?? sim.count;
        },
        get capacity(): number {
            return sim.count;
        },
    };
}

export function writeFluidSimPositions(engine: EngineContext, sim: FluidSim, positions: Float32Array, particleOffset: number): void {
    const offset = Math.max(0, Math.floor(particleOffset));
    if (positions.length % 4 !== 0 || offset + positions.length / 4 > sim.count) {
        throw new RangeError("[fluid] packed position writes require XYZW values within simulation capacity.");
    }
    engine._device.queue.writeBuffer(sim.positionBuffer, offset * 16, positions);
}

export async function readFluidSimPositions(engine: EngineContext, sim: FluidSim, particleOffsetInput: number, particleCountInput: number): Promise<Float32Array> {
    const particleOffset = Math.max(0, Math.floor(particleOffsetInput));
    const particleCount = Math.min(sim.count - particleOffset, Math.max(0, Math.floor(particleCountInput)));
    if (particleCount <= 0) {
        return new Float32Array(0);
    }
    const byteLength = particleCount * 16;
    const buffer = engine._device.createBuffer({
        label: "fluid-position-readback",
        size: byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        const encoder = engine._device.createCommandEncoder({ label: "fluid-position-readback" });
        encoder.copyBufferToBuffer(sim.positionBuffer, particleOffset * 16, buffer, 0, byteLength);
        engine._device.queue.submit([encoder.finish()]);
        await engine._device.queue.onSubmittedWorkDone();
        await buffer.mapAsync(GPUMapMode.READ);
        return new Float32Array(buffer.getMappedRange()).slice();
    } finally {
        if (buffer.mapState === "mapped") {
            buffer.unmap();
        }
        buffer.destroy();
    }
}
