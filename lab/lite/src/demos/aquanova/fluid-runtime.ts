import { FluidSimulationRuntime } from "./behaviors/fluid-simulation-runtime.js";

export interface FluidParticleAabb {
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
}

export interface FluidParticleCounterSource {
    readonly device: GPUDevice;
    readonly positionBuffer: GPUBuffer;
    readonly alphaBuffer: GPUBuffer;
}

export interface FluidElectricityState {
    readonly origin: readonly [number, number, number];
    readonly startedAtSeconds: number;
    readonly propagationSpeed: number;
}

export function electricityPropagationRadius(state: FluidElectricityState, elapsedSeconds: number): number {
    return Math.max(0, elapsedSeconds - state.startedAtSeconds) * state.propagationSpeed;
}

export interface FluidElectricityDomain {
    readonly id: number;
    readonly label: string;
    readonly electricity: FluidElectricityState | null;
}

export interface FluidElectricityFrameDomain {
    readonly domain: FluidElectricityDomain;
    readonly offset: number;
    readonly count: number;
    readonly particleRadius: number;
    readonly gridAabb: FluidParticleAabb;
    readonly elapsedSeconds: number;
}

export interface FluidElectrifierRegistration {
    readonly entityName: string;
    readonly particleThreshold: number;
    readonly propagationSpeed: number;
    readonly aabb: () => FluidParticleAabb | null;
    readonly onElectrified?: (domain: FluidElectricityDomain) => void;
}

export interface ElectrifiedFluidReceiverRegistration {
    readonly entityName: string;
    readonly particleThreshold: number;
    readonly aabb: () => FluidParticleAabb | null;
    readonly includeParticleRadius?: boolean;
    readonly onCount: (particleCount: number) => void;
}

export interface FluidElectricityRegistration {
    dispose(): void;
}

type ReadbackState = "idle" | "copied" | "mapping";

interface ReadbackSlot {
    readonly buffer: GPUBuffer;
    state: ReadbackState;
    version: number;
}

type ElectricityResult =
    | {
          readonly kind: "electrifier";
          readonly registration: FluidElectrifierRegistration;
          readonly domain: FluidElectricityDomainInternal;
          readonly origin: readonly [number, number, number];
          readonly elapsedSeconds: number;
      }
    | {
          readonly kind: "receiver";
          readonly registration: ElectrifiedFluidReceiverRegistration;
          readonly domain: FluidElectricityDomainInternal;
      };

interface ElectricityReadbackSlot {
    readonly buffer: GPUBuffer;
    state: ReadbackState;
    byteLength: number;
    results: readonly ElectricityResult[];
    receivers: readonly ElectrifiedFluidReceiverRegistration[];
}

interface FluidElectricityDomainInternal extends FluidElectricityDomain {
    electricity: FluidElectricityState | null;
}

const PARTICLE_COUNT_WORKGROUP_SIZE = 256;
export const ELECTRICITY_PARTICLE_MASK_RADIUS_SCALE = 1.35;
const PARTICLE_COUNT_PARAMS_BYTES = 64;
const PARTICLE_COUNT_SAMPLE_INTERVAL_FRAMES = 6;
const ELECTRICITY_WORKGROUP_SIZE = 256;
const ELECTRICITY_QUERY_BYTES = 64;
const ELECTRICITY_MAX_QUERY_PAIRS = 1024;
const ELECTRICITY_MAX_RESULTS = 1024;
const ELECTRICITY_PARAMS_BYTES = 32;
const PARTICLE_COUNT_SHADER = `
struct Params {
    aabbMin: vec4<f32>,
    aabbMax: vec4<f32>,
    particleCount: u32,
    _padding: vec3<u32>,
}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> particleAlpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${PARTICLE_COUNT_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= params.particleCount || particleAlpha[index] <= 0.001) {
        return;
    }
    let position = positions[index].xyz;
    if (all(position >= params.aabbMin.xyz) && all(position <= params.aabbMax.xyz)) {
        atomicAdd(&result, 1u);
    }
}`;

const ELECTRICITY_QUERY_SHADER = `
struct Query {
    aabbMin: vec4<f32>,
    aabbMax: vec4<f32>,
    originRadius: vec4<f32>,
    offset: u32,
    count: u32,
    output: u32,
    propagated: u32,
}
struct Params {
    queryCount: u32,
    _padding: vec3<u32>,
}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> particleAlpha: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> matches: array<u32, ${ELECTRICITY_WORKGROUP_SIZE}>;

@compute @workgroup_size(${ELECTRICITY_WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) invocation: vec3<u32>,
    @builtin(local_invocation_index) localIndex: u32
) {
    let queryIndex = invocation.y;
    var matched = 0u;
    if (queryIndex < params.queryCount) {
        let query = queries[queryIndex];
        let localParticle = invocation.x;
        if (localParticle < query.count) {
            let particleIndex = query.offset + localParticle;
            if (particleAlpha[particleIndex] > 0.001) {
                let position = positions[particleIndex].xyz;
                let inside = all(position >= query.aabbMin.xyz) && all(position <= query.aabbMax.xyz);
                let reached = query.propagated == 0u || distance(position, query.originRadius.xyz) <= query.originRadius.w;
                matched = select(0u, 1u, inside && reached);
            }
        }
    }
    matches[localIndex] = matched;
    workgroupBarrier();
    var stride = ${ELECTRICITY_WORKGROUP_SIZE / 2}u;
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

/**
 * Aquanova's shared fluid service.
 *
 * Behavior-owned simulations use the inherited registration lifecycle. Liquefaction and
 * behavior-owned water are aggregated into one visible particle stream by the demo, and this
 * runtime samples that stream asynchronously without copying particle positions to the CPU.
 */
export class AquanovaFluidRuntime extends FluidSimulationRuntime {
    private device: GPUDevice | null = null;
    private positionBuffer: GPUBuffer | null = null;
    private alphaBuffer: GPUBuffer | null = null;
    private counterBuffer: GPUBuffer | null = null;
    private paramsBuffer: GPUBuffer | null = null;
    private pipeline: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private readonly paramsData = new ArrayBuffer(PARTICLE_COUNT_PARAMS_BYTES);
    private readonly paramsFloats = new Float32Array(this.paramsData);
    private readonly paramsUints = new Uint32Array(this.paramsData);
    private readonly readbacks: ReadbackSlot[] = [];
    private requestedAabb: FluidParticleAabb | null = null;
    private requestedVersion = 0;
    private completedVersion = 0;
    private completedCount = 0;
    private sampleCooldown = 0;
    private electricityPipeline: GPUComputePipeline | null = null;
    private electricityQueryBuffer: GPUBuffer | null = null;
    private electricityResultBuffer: GPUBuffer | null = null;
    private electricityParamsBuffer: GPUBuffer | null = null;
    private electricityBindGroup: GPUBindGroup | null = null;
    private readonly electricityReadbacks: ElectricityReadbackSlot[] = [];
    private electricitySampleCooldown = 0;
    private nextElectricityDomainId = 1;
    private readonly electricityDomains = new Set<FluidElectricityDomainInternal>();
    private readonly electrifiers = new Set<FluidElectrifierRegistration>();
    private readonly electricityReceivers = new Set<ElectrifiedFluidReceiverRegistration>();
    private electricityPairCount = 0;
    private disposed = false;

    public installParticleCounter(source: FluidParticleCounterSource): void {
        if (this.device) {
            throw new Error("[aquanova] fluid particle counter is already installed");
        }
        if (this.disposed) {
            throw new Error("[aquanova] cannot install a fluid particle counter after disposal");
        }
        this.device = source.device;
        this.positionBuffer = source.positionBuffer;
        this.alphaBuffer = source.alphaBuffer;
        this.counterBuffer = source.device.createBuffer({
            label: "aq-fluid-aabb-count",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.paramsBuffer = source.device.createBuffer({
            label: "aq-fluid-aabb-count-params",
            size: PARTICLE_COUNT_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.pipeline = source.device.createComputePipeline({
            label: "aq-fluid-aabb-count",
            layout: "auto",
            compute: {
                module: source.device.createShaderModule({
                    label: "aq-fluid-aabb-count",
                    code: PARTICLE_COUNT_SHADER,
                }),
                entryPoint: "main",
            },
        });
        this.bindGroup = source.device.createBindGroup({
            label: "aq-fluid-aabb-count",
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: source.positionBuffer } },
                { binding: 1, resource: { buffer: source.alphaBuffer } },
                { binding: 2, resource: { buffer: this.counterBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
            ],
        });
        for (let index = 0; index < 3; index++) {
            this.readbacks.push({
                buffer: source.device.createBuffer({
                    label: `aq-fluid-aabb-count-readback-${index}`,
                    size: 4,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                }),
                state: "idle",
                version: 0,
            });
        }
    }

    public createElectricityDomain(label: string): FluidElectricityDomain {
        if (this.disposed) {
            throw new Error("[aquanova] cannot create a fluid electricity domain after disposal");
        }
        const domain: FluidElectricityDomainInternal = {
            id: this.nextElectricityDomainId++,
            label,
            electricity: null,
        };
        this.electricityDomains.add(domain);
        return domain;
    }

    public disposeElectricityDomain(domain: FluidElectricityDomain): void {
        this.electricityDomains.delete(domain as FluidElectricityDomainInternal);
    }

    public registerElectrifier(registration: FluidElectrifierRegistration): FluidElectricityRegistration {
        validatePositiveInteger(registration.particleThreshold, `${registration.entityName}.particleThreshold`);
        validatePositiveFinite(registration.propagationSpeed, `${registration.entityName}.propagationSpeed`);
        this.electrifiers.add(registration);
        return {
            dispose: () => {
                this.electrifiers.delete(registration);
            },
        };
    }

    public registerElectricityReceiver(registration: ElectrifiedFluidReceiverRegistration): FluidElectricityRegistration {
        validatePositiveInteger(registration.particleThreshold, `${registration.entityName}.particleThreshold`);
        this.electricityReceivers.add(registration);
        return {
            dispose: () => {
                this.electricityReceivers.delete(registration);
            },
        };
    }

    public recordElectricity(encoder: GPUCommandEncoder, frameDomains: readonly FluidElectricityFrameDomain[]): void {
        this.startElectricityReadbacks();
        if (this.electricitySampleCooldown > 0) {
            this.electricitySampleCooldown--;
            return;
        }
        const domains = frameDomains.filter(
            (frame): frame is FluidElectricityFrameDomain & { domain: FluidElectricityDomainInternal } =>
                frame.count > 0 && this.electricityDomains.has(frame.domain as FluidElectricityDomainInternal)
        );
        const pairs: Array<{
            readonly aabb: FluidParticleAabb;
            readonly frame: FluidElectricityFrameDomain & { domain: FluidElectricityDomainInternal };
            readonly output: number;
            readonly propagation: FluidElectricityState | null;
        }> = [];
        const results: ElectricityResult[] = [];

        for (const registration of this.electrifiers) {
            const aabb = registration.aabb();
            if (!aabb) continue;
            validateAabb(aabb);
            for (const frame of domains) {
                if (frame.domain.electricity || !aabbIntersects(aabb, frame.gridAabb)) continue;
                const output = results.length;
                results.push({
                    kind: "electrifier",
                    registration,
                    domain: frame.domain,
                    origin: aabbCenter(aabb),
                    elapsedSeconds: frame.elapsedSeconds,
                });
                pairs.push({ aabb, frame, output, propagation: null });
            }
        }

        const receivers = [...this.electricityReceivers];
        for (const registration of receivers) {
            const aabb = registration.aabb();
            if (!aabb) continue;
            validateAabb(aabb);
            for (const frame of domains) {
                const propagation = frame.domain.electricity;
                const queryAabb = registration.includeParticleRadius ? expandAabb(aabb, frame.particleRadius * ELECTRICITY_PARTICLE_MASK_RADIUS_SCALE) : aabb;
                if (!propagation || !aabbIntersects(queryAabb, frame.gridAabb)) continue;
                const output = results.length;
                results.push({ kind: "receiver", registration, domain: frame.domain });
                pairs.push({ aabb: queryAabb, frame, output, propagation });
            }
        }

        this.electricityPairCount = pairs.length;
        if (pairs.length === 0) {
            for (const registration of receivers) {
                if (this.electricityReceivers.has(registration)) registration.onCount(0);
            }
            this.electricitySampleCooldown = PARTICLE_COUNT_SAMPLE_INTERVAL_FRAMES - 1;
            return;
        }
        if (pairs.length > ELECTRICITY_MAX_QUERY_PAIRS) {
            throw new RangeError(`[aquanova] ${pairs.length} fluid electricity query pairs exceed the ${ELECTRICITY_MAX_QUERY_PAIRS} pair capacity`);
        }
        if (results.length > ELECTRICITY_MAX_RESULTS) {
            throw new RangeError(`[aquanova] ${results.length} fluid electricity results exceed the ${ELECTRICITY_MAX_RESULTS} result capacity`);
        }
        this.ensureElectricityGpu();
        const readback = this.electricityReadbacks.find((slot) => slot.state === "idle");
        if (!readback) {
            return;
        }
        const device = this.device!;
        const queryBuffer = this.electricityQueryBuffer!;
        const resultBuffer = this.electricityResultBuffer!;
        const paramsBuffer = this.electricityParamsBuffer!;
        const pipeline = this.electricityPipeline!;
        const bindGroup = this.electricityBindGroup!;
        const queryData = new ArrayBuffer(pairs.length * ELECTRICITY_QUERY_BYTES);
        const queryFloats = new Float32Array(queryData);
        const queryUints = new Uint32Array(queryData);
        let maxCount = 0;
        pairs.forEach(({ aabb, frame, output, propagation }, index) => {
            const word = (index * ELECTRICITY_QUERY_BYTES) / 4;
            queryFloats.set(aabb.min, word);
            queryFloats.set(aabb.max, word + 4);
            if (propagation) {
                queryFloats.set(propagation.origin, word + 8);
                queryFloats[word + 11] = electricityPropagationRadius(propagation, frame.elapsedSeconds);
            }
            queryUints[word + 12] = Math.max(0, Math.floor(frame.offset));
            queryUints[word + 13] = Math.max(0, Math.floor(frame.count));
            queryUints[word + 14] = output;
            queryUints[word + 15] = propagation ? 1 : 0;
            maxCount = Math.max(maxCount, frame.count);
        });
        device.queue.writeBuffer(queryBuffer, 0, queryData);
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([pairs.length, 0, 0, 0]));
        const byteLength = results.length * 4;
        encoder.clearBuffer(resultBuffer, 0, byteLength);
        const pass = encoder.beginComputePass({ label: "aq-fluid-electricity-query" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(maxCount / ELECTRICITY_WORKGROUP_SIZE), pairs.length);
        pass.end();
        encoder.copyBufferToBuffer(resultBuffer, 0, readback.buffer, 0, byteLength);
        readback.byteLength = byteLength;
        readback.results = results;
        readback.receivers = receivers;
        readback.state = "copied";
        this.electricitySampleCooldown = PARTICLE_COUNT_SAMPLE_INTERVAL_FRAMES - 1;
    }

    public electricityStats(): { domains: number; electrified: number; electrifiers: number; receivers: number; queryPairs: number } {
        return {
            domains: this.electricityDomains.size,
            electrified: [...this.electricityDomains].filter((domain) => domain.electricity !== null).length,
            electrifiers: this.electrifiers.size,
            receivers: this.electricityReceivers.size,
            queryPairs: this.electricityPairCount,
        };
    }

    /**
     * Return the latest asynchronously completed count and request a new sample for this AABB.
     *
     * The returned count can correspond to an earlier frame. Callers must own temporal filtering
     * such as hysteresis; this service reports only the raw spatial measurement.
     */
    public countParticlesInAabb(aabb: FluidParticleAabb): number {
        validateAabb(aabb);
        this.requestedAabb = {
            min: [...aabb.min],
            max: [...aabb.max],
        };
        this.requestedVersion++;
        return this.completedCount;
    }

    /** Encode the pending query after the frame's visible-particle aggregation is complete. */
    public recordParticleCount(encoder: GPUCommandEncoder, particleCount: number): void {
        this.startReadbacks();
        const device = this.device;
        const counterBuffer = this.counterBuffer;
        const paramsBuffer = this.paramsBuffer;
        const pipeline = this.pipeline;
        const bindGroup = this.bindGroup;
        const aabb = this.requestedAabb;
        if (!device || !counterBuffer || !paramsBuffer || !pipeline || !bindGroup || !aabb) {
            return;
        }
        const count = Math.max(0, Math.floor(particleCount));
        if (count === 0) {
            this.completedCount = 0;
            this.completedVersion = this.requestedVersion;
            this.sampleCooldown = 0;
            return;
        }
        if (this.sampleCooldown > 0) {
            this.sampleCooldown--;
            return;
        }
        const readback = this.readbacks.find((slot) => slot.state === "idle");
        if (!readback) {
            return;
        }

        this.paramsFloats.fill(0);
        this.paramsFloats.set(aabb.min, 0);
        this.paramsFloats.set(aabb.max, 4);
        this.paramsUints[8] = count;
        device.queue.writeBuffer(paramsBuffer, 0, this.paramsData);

        encoder.clearBuffer(counterBuffer);
        const pass = encoder.beginComputePass({ label: "aq-fluid-aabb-count" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(count / PARTICLE_COUNT_WORKGROUP_SIZE));
        pass.end();
        encoder.copyBufferToBuffer(counterBuffer, 0, readback.buffer, 0, 4);
        readback.version = this.requestedVersion;
        readback.state = "copied";
        this.sampleCooldown = PARTICLE_COUNT_SAMPLE_INTERVAL_FRAMES - 1;
    }

    public override dispose(): void {
        if (this.disposed) {
            return;
        }
        this.disposed = true;
        super.dispose();
        this.counterBuffer?.destroy();
        this.paramsBuffer?.destroy();
        for (const readback of this.readbacks) {
            readback.buffer.destroy();
        }
        this.readbacks.length = 0;
        this.electricityQueryBuffer?.destroy();
        this.electricityResultBuffer?.destroy();
        this.electricityParamsBuffer?.destroy();
        for (const readback of this.electricityReadbacks) {
            readback.buffer.destroy();
        }
        this.electricityReadbacks.length = 0;
        this.electricityDomains.clear();
        this.electrifiers.clear();
        this.electricityReceivers.clear();
        this.device = null;
        this.positionBuffer = null;
        this.alphaBuffer = null;
        this.counterBuffer = null;
        this.paramsBuffer = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.electricityPipeline = null;
        this.electricityQueryBuffer = null;
        this.electricityResultBuffer = null;
        this.electricityParamsBuffer = null;
        this.electricityBindGroup = null;
    }

    private startReadbacks(): void {
        for (const readback of this.readbacks) {
            if (readback.state !== "copied") {
                continue;
            }
            readback.state = "mapping";
            void readback.buffer
                .mapAsync(GPUMapMode.READ)
                .then(() => {
                    if (!this.disposed && readback.version >= this.completedVersion) {
                        this.completedCount = new Uint32Array(readback.buffer.getMappedRange())[0] ?? 0;
                        this.completedVersion = readback.version;
                    }
                    if (!this.disposed) {
                        readback.buffer.unmap();
                        readback.state = "idle";
                    }
                })
                .catch(() => {
                    if (!this.disposed) {
                        readback.state = "idle";
                    }
                });
        }
    }

    private ensureElectricityGpu(): void {
        if (this.electricityPipeline) {
            return;
        }
        const device = this.device;
        const positionBuffer = this.positionBuffer;
        const alphaBuffer = this.alphaBuffer;
        if (!device || !positionBuffer || !alphaBuffer) {
            throw new Error("[aquanova] fluid particle counter must be installed before electricity queries");
        }
        this.electricityQueryBuffer = device.createBuffer({
            label: "aq-fluid-electricity-queries",
            size: ELECTRICITY_MAX_QUERY_PAIRS * ELECTRICITY_QUERY_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.electricityResultBuffer = device.createBuffer({
            label: "aq-fluid-electricity-results",
            size: ELECTRICITY_MAX_RESULTS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.electricityParamsBuffer = device.createBuffer({
            label: "aq-fluid-electricity-params",
            size: ELECTRICITY_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.electricityPipeline = device.createComputePipeline({
            label: "aq-fluid-electricity-query",
            layout: "auto",
            compute: {
                module: device.createShaderModule({
                    label: "aq-fluid-electricity-query",
                    code: ELECTRICITY_QUERY_SHADER,
                }),
                entryPoint: "main",
            },
        });
        this.electricityBindGroup = device.createBindGroup({
            label: "aq-fluid-electricity-query",
            layout: this.electricityPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: positionBuffer } },
                { binding: 1, resource: { buffer: alphaBuffer } },
                { binding: 2, resource: { buffer: this.electricityQueryBuffer } },
                { binding: 3, resource: { buffer: this.electricityResultBuffer } },
                { binding: 4, resource: { buffer: this.electricityParamsBuffer } },
            ],
        });
        for (let index = 0; index < 3; index++) {
            this.electricityReadbacks.push({
                buffer: device.createBuffer({
                    label: `aq-fluid-electricity-readback-${index}`,
                    size: ELECTRICITY_MAX_RESULTS * 4,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                }),
                state: "idle",
                byteLength: 0,
                results: [],
                receivers: [],
            });
        }
    }

    private startElectricityReadbacks(): void {
        for (const readback of this.electricityReadbacks) {
            if (readback.state !== "copied") {
                continue;
            }
            readback.state = "mapping";
            void readback.buffer
                .mapAsync(GPUMapMode.READ, 0, readback.byteLength)
                .then(() => {
                    if (this.disposed) {
                        return;
                    }
                    const counts = new Uint32Array(readback.buffer.getMappedRange(0, readback.byteLength));
                    const receiverCounts = new Map(readback.receivers.map((registration) => [registration, 0]));
                    readback.results.forEach((result, index) => {
                        const count = counts[index] ?? 0;
                        if (result.kind === "receiver") {
                            if (this.electricityReceivers.has(result.registration) && this.electricityDomains.has(result.domain)) {
                                receiverCounts.set(result.registration, (receiverCounts.get(result.registration) ?? 0) + count);
                            }
                            return;
                        }
                        if (
                            count >= result.registration.particleThreshold &&
                            this.electrifiers.has(result.registration) &&
                            this.electricityDomains.has(result.domain) &&
                            !result.domain.electricity
                        ) {
                            result.domain.electricity = {
                                origin: [...result.origin],
                                startedAtSeconds: result.elapsedSeconds,
                                propagationSpeed: result.registration.propagationSpeed,
                            };
                            result.registration.onElectrified?.(result.domain);
                        }
                    });
                    for (const [registration, count] of receiverCounts) {
                        if (this.electricityReceivers.has(registration)) registration.onCount(count);
                    }
                })
                .finally(() => {
                    if (readback.buffer.mapState === "mapped") {
                        readback.buffer.unmap();
                    }
                    if (!this.disposed) {
                        readback.state = "idle";
                        readback.results = [];
                        readback.receivers = [];
                    }
                });
        }
    }
}

function validateAabb(aabb: FluidParticleAabb): void {
    for (let axis = 0; axis < 3; axis++) {
        const min = aabb.min[axis]!;
        const max = aabb.max[axis]!;
        if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) {
            throw new Error(`[aquanova] fluid particle AABB axis ${axis} must have finite min <= max`);
        }
    }
}

function validatePositiveInteger(value: number, label: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`[aquanova] ${label} must be a positive integer`);
    }
}

function validatePositiveFinite(value: number, label: string): void {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`[aquanova] ${label} must be finite and positive`);
    }
}

function aabbIntersects(a: FluidParticleAabb, b: FluidParticleAabb): boolean {
    return a.min[0] <= b.max[0] && a.max[0] >= b.min[0] && a.min[1] <= b.max[1] && a.max[1] >= b.min[1] && a.min[2] <= b.max[2] && a.max[2] >= b.min[2];
}

function expandAabb(aabb: FluidParticleAabb, padding: number): FluidParticleAabb {
    return {
        min: [aabb.min[0] - padding, aabb.min[1] - padding, aabb.min[2] - padding],
        max: [aabb.max[0] + padding, aabb.max[1] + padding, aabb.max[2] + padding],
    };
}

function aabbCenter(aabb: FluidParticleAabb): [number, number, number] {
    return [(aabb.min[0] + aabb.max[0]) * 0.5, (aabb.min[1] + aabb.max[1]) * 0.5, (aabb.min[2] + aabb.max[2]) * 0.5];
}
