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

type ReadbackState = "idle" | "copied" | "mapping";

interface ReadbackSlot {
    readonly buffer: GPUBuffer;
    state: ReadbackState;
    version: number;
}

const PARTICLE_COUNT_WORKGROUP_SIZE = 256;
const PARTICLE_COUNT_PARAMS_BYTES = 64;
const PARTICLE_COUNT_SAMPLE_INTERVAL_FRAMES = 6;
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

/**
 * Aquanova's shared fluid service.
 *
 * Behavior-owned simulations use the inherited registration lifecycle. Liquefaction and
 * behavior-owned water are aggregated into one visible particle stream by the demo, and this
 * runtime samples that stream asynchronously without copying particle positions to the CPU.
 */
export class AquanovaFluidRuntime extends FluidSimulationRuntime {
    private device: GPUDevice | null = null;
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
    private disposed = false;

    public installParticleCounter(source: FluidParticleCounterSource): void {
        if (this.device) {
            throw new Error("[aquanova] fluid particle counter is already installed");
        }
        if (this.disposed) {
            throw new Error("[aquanova] cannot install a fluid particle counter after disposal");
        }
        this.device = source.device;
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
        this.device = null;
        this.counterBuffer = null;
        this.paramsBuffer = null;
        this.pipeline = null;
        this.bindGroup = null;
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
