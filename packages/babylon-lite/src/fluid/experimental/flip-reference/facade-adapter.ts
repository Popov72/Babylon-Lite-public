import type { EngineContext } from "../../../engine/engine.js";
import { BU, SS } from "../../../engine/gpu-flags.js";
import type { FluidSimulationOptions } from "../../core/fluid-facade.js";
import type { ResolvedFluidSimulationConfig } from "../../core/simulation-config.js";
import type { FluidAllocationPlan, FluidAllocationResource, FluidDeviceLimitsSnapshot } from "../../core/allocation-plan.js";
import { FLIP_REFERENCE_MAX_FRAME_SUBSTEPS, FLIP_REFERENCE_PARAMETER_FLOATS, flipReferenceReadbackBytes, flipReferenceStorageSizes } from "./allocation.js";
import {
    createFluidInitialParticles,
    legacyEmitterConfigToFluidFlow,
    sceneSdfGridBindingWgsl,
    type EmitterConfig,
    type DiffusePool,
    type FluidEmitter,
    type FluidFlowConfig,
    type FluidInitialParticles,
    type FluidSim,
    type FluidProfiler,
    type ForceFieldSpec,
    type FoamConfig,
    type SceneSdfSpec,
} from "../../core/sim-common.js";
import { createFlipReferenceSimulation, disposeFlipReferenceSimulation } from "./solver.js";
import {
    collectFlipReferenceGpuStatus,
    createFlipReferenceGpuRuntime,
    disposeFlipReferenceGpuRuntime,
    flipReferenceGpuResources,
    recordFlipReferenceFrame,
    type FlipReferenceGpuRuntime,
} from "./gpu-runtime.js";
import type { FlipReferenceSimulation } from "./types.js";
import { flipReferenceForceWgsl } from "./force.js";
import {
    collectFlipReferenceWhitewaterStatus,
    configureFlipReferenceWhitewater,
    createFlipReferenceWhitewater,
    disposeFlipReferenceWhitewater,
    planFlipReferenceWhitewater,
    recordFlipReferenceWhitewater,
    type FlipReferenceWhitewater,
} from "./whitewater.js";

const DEFAULT_PRESSURE_TOLERANCE = 1e-5;
const DEFAULT_PRESSURE_ABSOLUTE_TOLERANCE = 1e-8;
const DEFAULT_MAX_PRESSURE_ITERATIONS = 1024;
const DEFAULT_MAX_SUBSTEP_SECONDS = 1 / 120;
const REFERENCE_DOMAIN_INSET_EPSILON = 5e-5;

/** @internal Conservative without explicit seed data; exact for imported initial particle state. */
export function planFlipReferenceFluidAllocation(options: FluidSimulationOptions, config: ResolvedFluidSimulationConfig, limits?: FluidDeviceLimitsSnapshot): FluidAllocationPlan {
    if (!config.flip) {
        throw new Error("[FLIP Reference] memory projection requires resolved FLIP discretization.");
    }
    const [x, y, z] = config.flip.gridDim;
    const cells = x * y * z;
    const faces = (x + 1) * y * z + x * (y + 1) * z + x * y * (z + 1);
    const vertices = (x + 1) * (y + 1) * (z + 1);
    const particles = options.initialPositions ? options.initialPositions.length / 3 : options.particleCount;
    const sizes = flipReferenceStorageSizes(cells, faces, vertices, particles, true, FLIP_REFERENCE_MAX_FRAME_SUBSTEPS);
    const names = ["positions", "velocities", "speeds", "MAC faces", "cells", "particle lists", "solid nodes", "PCG", "compaction scratch", "particle state"];
    const resources: FluidAllocationResource[] = sizes.map((bytes, index) => ({ name: "Reference " + names[index]!, kind: "buffer", binding: "storage", bytes }));
    resources.push(
        { name: "Reference parameters", kind: "buffer", binding: "uniform", bytes: FLIP_REFERENCE_PARAMETER_FLOATS * 4 },
        { name: "Reference readback", kind: "buffer", binding: "none", bytes: flipReferenceReadbackBytes(cells, particles) },
        { name: "Published positions", kind: "buffer", binding: "storage", bytes: options.particleCount * 16 },
        { name: "Published velocities", kind: "buffer", binding: "storage", bytes: options.particleCount * 16 },
        { name: "Published speeds", kind: "buffer", binding: "storage", bytes: options.particleCount * 4 },
        { name: "Published draw arguments", kind: "buffer", binding: "storage", bytes: 16 },
        { name: "Collision sampler parameters", kind: "buffer", binding: "uniform", bytes: 48 }
    );
    resources.push(...flipReferenceGpuResources(cells));
    const whitewaterPlan = options.foam ? planFlipReferenceWhitewater(options.particleCount, cells, options.foam, limits) : null;
    if (whitewaterPlan) {
        resources.push(...whitewaterPlan.resources);
    }
    const errors: string[] = whitewaterPlan ? [...whitewaterPlan.errors] : [];
    for (const resource of resources) {
        if (resource.bytes > (limits?.maxBufferSize ?? Infinity) || (resource.binding === "storage" && resource.bytes > (limits?.maxStorageBufferBindingSize ?? Infinity))) {
            errors.push(resource.name + " requires " + resource.bytes + " bytes, exceeding the device buffer limit.");
        }
    }
    if (particles > options.particleCount) {
        errors.push("Exact initial particles exceed the requested published capacity.");
    }
    const steadyBytes = resources.reduce((sum, resource) => sum + resource.bytes, 0);
    return {
        method: "FLIP",
        dimensions: { gridDim: [x, y, z], cellCount: cells, faceCount: faces, particleCount: options.particleCount },
        foamCapacity: whitewaterPlan?.capacity ?? 0,
        polygonTriangleCapacity: 0,
        steadyBytes,
        rebuildPeakBytes: steadyBytes,
        resources,
        errors,
    };
}

interface ReferenceSeed {
    readonly positions: Float32Array;
    readonly velocities: Float32Array;
    readonly activeCount: number;
    readonly emitterCounts: ReadonlyMap<string, number>;
}

interface CollisionSampler {
    readonly scene: SceneSdfSpec | null;
    readonly core: FlipReferenceSimulation;
    readonly pipeline: GPUComputePipeline;
    readonly bindGroup: GPUBindGroup;
}

function finite(name: string, value: number): number {
    if (!Number.isFinite(value)) {
        throw new RangeError(`[FLIP Reference] ${name} must be finite.`);
    }
    return value;
}

function positive(name: string, value: number): number {
    if (!(finite(name, value) > 0)) {
        throw new RangeError(`[FLIP Reference] ${name} must be positive.`);
    }
    return value;
}

function positiveF32(name: string, value: number): number {
    positive(name, value);
    if (!Number.isFinite(Math.fround(value)) || !(Math.fround(value) >= 2 ** -126)) {
        throw new RangeError(`[FLIP Reference] ${name} must be representable as a positive normal f32 value.`);
    }
    return value;
}

function positiveSquaredF32(name: string, value: number): number {
    positiveF32(name, value);
    positiveF32(`squared ${name}`, value * value);
    return value;
}

function positiveInteger(name: string, value: number): number {
    positive(name, value);
    const integer = Math.round(value);
    if (!Number.isSafeInteger(integer) || integer < 1) {
        throw new RangeError(`[FLIP Reference] ${name} must be a positive integer.`);
    }
    return integer;
}

function pressureIterationLimit(value: number): number {
    const count = positiveInteger("physics.maxPressureIterations", value);
    if (count > 16_777_216) {
        throw new RangeError("[FLIP Reference] maxPressureIterations exceeds the exact GPU iteration-counter range.");
    }
    return count;
}

function substepLimit(name: string, value: number): number {
    const count = positiveInteger(name, value);
    if (count > FLIP_REFERENCE_MAX_FRAME_SUBSTEPS) {
        throw new RangeError(`[FLIP Reference] ${name} exceeds the ${FLIP_REFERENCE_MAX_FRAME_SUBSTEPS}-substep GPU recording limit.`);
    }
    return count;
}

function cflValue(value: number): number {
    return value > 0 ? positiveF32("physics.cflNumber", value) : Math.max(0, finite("physics.cflNumber", value));
}

function abortError(reason: string): Error {
    const error = new Error(`[FLIP Reference] asynchronous step cancelled by ${reason}.`);
    error.name = "AbortError";
    return error;
}

function validateFlow(config: FluidFlowConfig | null): void {
    if (!config) {
        return;
    }
    const inflow = config.emitters.find((emitter) => emitter.enabled && emitter.behavior === "inflow");
    if (inflow) {
        throw new Error(`[FLIP Reference] active inflow emitter '${inflow.name || inflow.id}' is unsupported; only reset-time Initial emitters are available.`);
    }
    const sink = config.sinks.find((entry) => entry.enabled);
    if (sink) {
        throw new Error(`[FLIP Reference] active sink '${sink.name || sink.id}' is unsupported.`);
    }
}

function exactSeed(options: FluidSimulationOptions, capacity: number): ReferenceSeed | null {
    const positions = options.initialPositions;
    const velocities = options.initialVelocities;
    if (!positions) {
        if (velocities) {
            throw new RangeError("[FLIP Reference] initialVelocities requires initialPositions.");
        }
        return null;
    }
    if (positions.length % 3 !== 0) {
        throw new RangeError("[FLIP Reference] initialPositions must be tightly packed XYZ values.");
    }
    const count = positions.length / 3;
    if (count > capacity) {
        throw new RangeError(`[FLIP Reference] exact initial state has ${count} particles but the published capacity is ${capacity}.`);
    }
    if (velocities && velocities.length !== positions.length) {
        throw new RangeError("[FLIP Reference] initialVelocities must match initialPositions.");
    }
    const initial = options.flow?.emitters.filter((emitter) => emitter.enabled && emitter.behavior === "initial") ?? [];
    return {
        positions: positions.slice(),
        velocities: velocities?.slice() ?? new Float32Array(positions.length),
        activeCount: count,
        emitterCounts: new Map(initial.length === 1 ? [[initial[0]!.id, count]] : []),
    };
}

function clippedSpawnBounds(
    authoredMin: readonly [number, number, number],
    authoredMax: readonly [number, number, number],
    gridMax: readonly [number, number, number],
    margin: number,
    requestedMin: readonly [number, number, number],
    requestedMax: readonly [number, number, number]
): { min: [number, number, number]; max: [number, number, number] } {
    const domainMin = authoredMin.map((value) => value + margin) as [number, number, number];
    const domainMax = authoredMax.map((value, axis) => Math.min(value - margin, gridMax[axis]! - margin)) as [number, number, number];
    let min = requestedMin.map((value, axis) => Math.max(value, domainMin[axis]!)) as [number, number, number];
    let max = requestedMax.map((value, axis) => Math.min(value, domainMax[axis]!)) as [number, number, number];
    if (min.some((value, axis) => !(max[axis]! > value))) {
        const extent = domainMax.map((value, axis) => value - domainMin[axis]!) as [number, number, number];
        if (extent.some((value) => !(value > 0))) {
            throw new RangeError("[FLIP Reference] the snapped grid and reference domain inset leave no valid initial-particle volume.");
        }
        min = extent.map((value, axis) => domainMin[axis]! + value * (axis === 1 ? 0.1 : 0.25)) as [number, number, number];
        max = extent.map((value, axis) => domainMin[axis]! + value * (axis === 1 ? 0.6 : 0.75)) as [number, number, number];
    }
    return { min, max };
}

function fallbackFlow(min: readonly [number, number, number], max: readonly [number, number, number]): FluidFlowConfig {
    return {
        initialEmittersFillCapacity: true,
        emitters: [
            {
                id: "flip-reference-spawn",
                name: "FLIP Reference spawn",
                enabled: true,
                behavior: "initial",
                transform: {
                    position: [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5],
                    rotation: [0, 0, 0, 1],
                    scale: [1, 1, 1],
                },
                shape: { type: "box", size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]] },
                sampling: "volume",
                velocity: [0, 0, 0],
                velocitySpace: "world",
                spread: 0,
            },
        ],
        sinks: [],
    };
}

function filterSpawnAcceptance(particles: FluidInitialParticles, accept: ((x: number, y: number, z: number) => boolean) | null): FluidInitialParticles {
    if (!accept) {
        return particles;
    }
    const positions = new Float32Array(particles.positions.length);
    const velocities = new Float32Array(particles.velocities.length);
    let cursor = 0;
    for (let index = 0; index < particles.activeCount; index++) {
        const offset = index * 3;
        const x = particles.positions[offset]!;
        const y = particles.positions[offset + 1]!;
        const z = particles.positions[offset + 2]!;
        if (!accept(x, y, z)) {
            continue;
        }
        positions[cursor] = x;
        velocities[cursor++] = particles.velocities[offset]!;
        positions[cursor] = y;
        velocities[cursor++] = particles.velocities[offset + 1]!;
        positions[cursor] = z;
        velocities[cursor++] = particles.velocities[offset + 2]!;
    }
    return {
        positions: positions.slice(0, cursor),
        velocities: velocities.slice(0, cursor),
        activeCount: cursor / 3,
        emitterCounts: new Map([["flip-reference-spawn", cursor / 3]]),
    };
}

function sceneSamplerWgsl(scene: SceneSdfSpec, outputBinding: number): string {
    return /* wgsl */ `
${scene.struct}
@group(0) @binding(0) var<uniform> sceneSdfParams: SceneSdfParams;
struct FlipReferenceSampleParams {
    originDx: vec4<f32>,
    dimensions: vec4<u32>,
    temporal: vec4<f32>,
}
@group(0) @binding(1) var<uniform> sampleParams: FlipReferenceSampleParams;
${sceneSdfGridBindingWgsl(scene, 2)}
@group(0) @binding(${outputBinding}) var<storage, read_write> sampledSolids: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> stepState: array<u32>;
${scene.sdf}
@compute @workgroup_size(128)
fn sampleFlipReferenceSolids(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = sampleParams.dimensions.xyz + vec3<u32>(1u);
    let count = dimensions.x * dimensions.y * dimensions.z;
    if (id.x >= count) { return; }
    let coordinate = vec3<u32>(id.x % dimensions.x, (id.x / dimensions.x) % dimensions.y, id.x / (dimensions.x * dimensions.y));
    let point = sampleParams.originDx.xyz + vec3<f32>(coordinate) * sampleParams.originDx.w;
    let substepDt = bitcast<f32>(stepState[4]);
    let timeOffset = substepDt - bitcast<f32>(stepState[6]);
    let epsilon = max(sampleParams.originDx.w * 0.25, 1.0e-4);
    let ex = vec3<f32>(epsilon, 0.0, 0.0);
    let ey = vec3<f32>(0.0, epsilon, 0.0);
    let ez = vec3<f32>(0.0, 0.0, epsilon);
    let gradient = vec3<f32>(
        sceneSdf(point + ex, timeOffset) - sceneSdf(point - ex, timeOffset),
        sceneSdf(point + ey, timeOffset) - sceneSdf(point - ey, timeOffset),
        sceneSdf(point + ez, timeOffset) - sceneSdf(point - ez, timeOffset));
    let gradientSquared = dot(gradient, gradient);
    var velocity = vec3<f32>(0.0);
    if (gradientSquared > 1.0e-20) {
        let temporalEpsilon = max(1.0e-10, min(substepDt * 0.5, 1.0 / 120.0));
        let derivative = (sceneSdf(point, timeOffset + temporalEpsilon) - sceneSdf(point, timeOffset - temporalEpsilon)) / (2.0 * temporalEpsilon);
        velocity = -derivative * gradient * inverseSqrt(gradientSquared);
    }
    sampledSolids[id.x] = vec4<f32>(sceneSdf(point, timeOffset), velocity);
}`;
}

function clearSamplerWgsl(): string {
    return /* wgsl */ `
struct FlipReferenceSampleParams {
    originDx: vec4<f32>,
    dimensions: vec4<u32>,
    temporal: vec4<f32>,
}
@group(0) @binding(1) var<uniform> sampleParams: FlipReferenceSampleParams;
@group(0) @binding(2) var<storage, read_write> sampledSolids: array<vec4<f32>>;
@compute @workgroup_size(128)
fn clearFlipReferenceSolids(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = sampleParams.dimensions.xyz + vec3<u32>(1u);
    let count = dimensions.x * dimensions.y * dimensions.z;
    if (id.x < count) { sampledSolids[id.x] = vec4<f32>(1.0e30, 0.0, 0.0, 0.0); }
}`;
}

function checkedPipeline(target: FlipReferenceSimulation, label: string, code: string, entryPoint: string, layout: GPUPipelineLayout | "auto" = "auto"): GPUComputePipeline {
    const device = target._device;
    device.pushErrorScope("validation");
    try {
        const module = device.createShaderModule({ label, code });
        return device.createComputePipeline({ label, layout, compute: { module, entryPoint } });
    } finally {
        const previousReady = target._ready;
        target._ready = Promise.all([previousReady, device.popErrorScope()]).then(([, validation]) => {
            if (validation) {
                target._error = new Error(`[FLIP Reference] ${label} failed: ${validation.message}`);
            }
        });
    }
}

/** @internal Accuracy-oriented FLIP implementation adapted to the stable FluidSim renderer contract. */
export function createFlipReferenceFluidSim(engine: EngineContext, options: FluidSimulationOptions, config: ResolvedFluidSimulationConfig): FluidSim {
    if (options.method !== "FLIP" || config.method !== "FLIP" || !config.flip) {
        throw new TypeError("[FLIP Reference] the adapter requires resolved FLIP discretization.");
    }
    if (options.pagedGrid) {
        throw new Error("[FLIP Reference] paged grids are unsupported.");
    }
    const device = engine._device;
    const capacity = positiveInteger("particleCount", options.particleCount);
    const publishedPositionBytes = capacity * 16;
    const publishedDebugBytes = capacity * 4;
    if (
        !Number.isSafeInteger(publishedPositionBytes) ||
        publishedPositionBytes > device.limits.maxBufferSize ||
        publishedPositionBytes > device.limits.maxStorageBufferBindingSize ||
        publishedDebugBytes > device.limits.maxBufferSize ||
        publishedDebugBytes > device.limits.maxStorageBufferBindingSize
    ) {
        throw new RangeError("[FLIP Reference] published particle capacity exceeds this device's storage-buffer limits.");
    }
    const dx = config.flip.dx;
    const gridDimensions = [...config.flip.gridDim] as [number, number, number];
    const gridOrigin = [...options.bounds.min] as [number, number, number];
    const gridMax = gridOrigin.map((value, axis) => value + gridDimensions[axis]! * dx) as [number, number, number];
    const domainInset = 1.5 * dx + REFERENCE_DOMAIN_INSET_EPSILON;
    const seedMargin = domainInset + 0.1 * dx + 1e-6;
    const seedBounds = clippedSpawnBounds(options.bounds.min, options.bounds.max, gridMax, seedMargin, options.bounds.min, options.bounds.max);
    const defaultSpawn = clippedSpawnBounds(options.bounds.min, options.bounds.max, gridMax, seedMargin, [-2, 4, -2], [2, 12, 2]);
    let spawnMin = defaultSpawn.min;
    let spawnMax = defaultSpawn.max;
    let spawnAccept: ((x: number, y: number, z: number) => boolean) | null = null;
    let flow = options.flow ? structuredClone(options.flow) : null;
    validateFlow(flow);
    const importedSeed = exactSeed(options, capacity);
    let generation = 0;
    let publishedActiveCount = 0;
    let initialEmitterParticleCounts: ReadonlyMap<string, number> = new Map();
    let sceneSdf: SceneSdfSpec | null = null;
    let sampler: CollisionSampler | null = null;
    let forceSpec: ForceFieldSpec | null = null;
    let forcePipeline: GPUComputePipeline | null = null;
    let forceLayout: GPUBindGroupLayout | null = null;
    let forceStruct = "";
    let forceWgsl = "";
    let forceBinding: { core: FlipReferenceSimulation; buffer: GPUBuffer; pipeline: GPUComputePipeline; bindGroup: GPUBindGroup } | null = null;
    let pendingOperation: Promise<void> | null = null;
    let pendingReset: ReferenceSeed | null = null;
    let core: FlipReferenceSimulation | null = null;
    let coreParamsU32: Uint32Array | null = null;
    let gpu: FlipReferenceGpuRuntime | null = null;
    let foamConfig: FoamConfig | null = null;
    let whitewater: FlipReferenceWhitewater | null = null;
    let workingInvalid = false;
    let disposed = false;
    let profiler: FluidProfiler | null = null;

    let gravity = finite("physics.gravity", config.physics.gravity ?? 9.8);
    let flipRatio = Math.min(1, Math.max(0, finite("physics.flipRatio", config.physics.flipRatio ?? 0.95)));
    let minSubsteps = substepLimit("physics.minSubsteps", config.physics.minSubsteps ?? 1);
    let maxSubsteps = Math.max(minSubsteps, substepLimit("physics.maxSubsteps", config.physics.maxSubsteps ?? 8));
    let maxSubstepSeconds = positiveF32("maximum substep seconds", (config.physics.maxSubDtMs ?? DEFAULT_MAX_SUBSTEP_SECONDS * 1000) / 1000);
    let cflNumber = cflValue(config.physics.cflNumber ?? 2);
    let pressureTolerance = positiveSquaredF32("physics.pressureTolerance", options.physics?.pressureTolerance ?? DEFAULT_PRESSURE_TOLERANCE);
    let pressureAbsoluteTolerance = positiveSquaredF32("physics.pressureAbsoluteTolerance", config.physics.pressureAbsoluteTolerance ?? DEFAULT_PRESSURE_ABSOLUTE_TOLERANCE);
    let maxPressureIterations = pressureIterationLimit(config.physics.maxPressureIterations ?? DEFAULT_MAX_PRESSURE_ITERATIONS);

    const publicationUsage = BU.STORAGE | BU.COPY_DST | BU.COPY_SRC;
    const positionBuffer = device.createBuffer({ label: "flip-reference:published-positions", size: publishedPositionBytes, usage: publicationUsage });
    const velocityBuffer = device.createBuffer({ label: "flip-reference:published-velocities", size: publishedPositionBytes, usage: publicationUsage });
    const debugBuffer = device.createBuffer({ label: "flip-reference:published-speeds", size: publishedDebugBytes, usage: publicationUsage });
    const renderIndirectBuffer = device.createBuffer({ label: "flip-reference:published-draw", size: 16, usage: publicationUsage | BU.INDIRECT });
    const samplerParamsBuffer = device.createBuffer({ label: "flip-reference:scene-sampler-params", size: 48, usage: BU.UNIFORM | BU.COPY_DST });
    const samplerParams = new Float32Array(12);
    const samplerParamsU32 = new Uint32Array(samplerParams.buffer);
    samplerParams.set([...gridOrigin, dx], 0);
    samplerParamsU32.set([...gridDimensions, 0], 4);
    device.queue.writeBuffer(samplerParamsBuffer, 0, samplerParams);

    function createSeed(): ReferenceSeed {
        if (importedSeed) {
            return importedSeed;
        }
        validateFlow(flow);
        const seedFlow = flow ?? fallbackFlow(spawnMin, spawnMax);
        const bounds = flow ? seedBounds : { min: spawnMin, max: spawnMax };
        const particles = createFluidInitialParticles(capacity, seedFlow, config.particleVolume, bounds, true);
        if (!particles) {
            throw new Error("[FLIP Reference] failed to create reset-time initial particles.");
        }
        const accepted = flow ? particles : filterSpawnAcceptance(particles, spawnAccept);
        return {
            positions: accepted.positions,
            velocities: accepted.velocities,
            activeCount: accepted.activeCount,
            emitterCounts: accepted.emitterCounts,
        };
    }

    function publishSeed(seed: ReferenceSeed): void {
        const positions = new Float32Array(capacity * 4);
        const velocities = new Float32Array(capacity * 4);
        const speeds = new Float32Array(capacity);
        for (let index = 0; index < seed.activeCount; index++) {
            const source = index * 3;
            const target = index * 4;
            const vx = seed.velocities[source]!;
            const vy = seed.velocities[source + 1]!;
            const vz = seed.velocities[source + 2]!;
            positions[target] = seed.positions[source]!;
            positions[target + 1] = seed.positions[source + 1]!;
            positions[target + 2] = seed.positions[source + 2]!;
            positions[target + 3] = 1;
            velocities[target] = vx;
            velocities[target + 1] = vy;
            velocities[target + 2] = vz;
            speeds[index] = Math.hypot(vx, vy, vz);
        }
        device.queue.writeBuffer(positionBuffer, 0, positions);
        device.queue.writeBuffer(velocityBuffer, 0, velocities);
        device.queue.writeBuffer(debugBuffer, 0, speeds);
        device.queue.writeBuffer(renderIndirectBuffer, 0, new Uint32Array([6, seed.activeCount, 0, 0]));
        publishedActiveCount = seed.activeCount;
        initialEmitterParticleCounts = seed.emitterCounts;
    }

    function createCore(seed: ReferenceSeed): FlipReferenceSimulation {
        const simulation = createFlipReferenceSimulation(engine, {
            referenceNumerics: true,
            gridOrigin,
            gridDimensions,
            cellSize: dx,
            initialPositions: seed.positions,
            initialVelocities: seed.velocities,
            gravity: [0, -gravity, 0],
            picFraction: 1 - flipRatio,
            pressureTolerance,
            pressureAbsoluteTolerance,
            maxPressureIterations,
            particleRadius: config.particleRadius,
            domainInset,
            removeInsideSolids: true,
            extremeVelocityRemoval: {
                frameDtSeconds: 1 / 60,
                cfl: 1,
                maxFrameSubsteps: FLIP_REFERENCE_MAX_FRAME_SUBSTEPS,
            },
            solidsIncludeDomain: false,
        });
        simulation._profiler = profiler;
        return simulation;
    }

    function replaceCore(seed: ReferenceSeed): void {
        if (whitewater) {
            disposeFlipReferenceWhitewater(whitewater);
            whitewater = null;
        }
        if (core?._error) {
            forcePipeline = null;
        }
        if (gpu) {
            disposeFlipReferenceGpuRuntime(gpu);
            gpu = null;
        }
        if (core) {
            disposeFlipReferenceSimulation(core);
            core = null;
        }
        core = createCore(seed);
        coreParamsU32 = new Uint32Array(core._params.buffer);
        gpu = createFlipReferenceGpuRuntime(core, positionBuffer, velocityBuffer, debugBuffer, renderIndirectBuffer);
        if (foamConfig) {
            whitewater = createFlipReferenceWhitewater(core, gpu, capacity, foamConfig);
        }
        sampler = null;
        forceBinding = null;
        workingInvalid = false;
    }

    function applyLiveParameters(target: FlipReferenceSimulation, frameDtSeconds: number): void {
        target._params[8] = 0;
        target._params[9] = -gravity;
        target._params[10] = 0;
        target._params[12] = 1 - flipRatio;
        target._params[24] = pressureTolerance ** 2;
        target._params[25] = pressureAbsoluteTolerance ** 2;
        target._pressureTolerance = pressureTolerance;
        target._pressureAbsoluteTolerance = pressureAbsoluteTolerance;
        target._maxPressureIterations = maxPressureIterations;
        target._params[43] = maxPressureIterations;
        target._extremeRemoval = cflNumber > 0;
        target._removalHistogramBins = maxSubsteps;
        coreParamsU32![38] = maxSubsteps;
        coreParamsU32![39] = target._extremeRemoval ? 3 : 1;
        const interval = target._extremeRemoval ? positiveF32("extreme-speed histogram interval", (cflNumber * dx) / frameDtSeconds) : 0;
        if (target._extremeRemoval) {
            positiveSquaredF32("extreme-speed histogram threshold", (maxSubsteps + 3) * interval);
        }
        target._params[40] = interval;
    }

    function buildCollisionSampler(target: FlipReferenceSimulation, scene: SceneSdfSpec | null): CollisionSampler {
        if (!scene) {
            const pipeline = checkedPipeline(target, "flip-reference:clear-scene-sdf", clearSamplerWgsl(), "clearFlipReferenceSolids");
            return {
                scene,
                core: target,
                pipeline,
                bindGroup: device.createBindGroup({
                    label: "flip-reference:clear-scene-sdf",
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 1, resource: { buffer: samplerParamsBuffer } },
                        { binding: 2, resource: { buffer: target._solidBuffer } },
                    ],
                }),
            };
        }
        const outputBinding = scene.sdfGrid ? 3 : 2;
        const pipeline = checkedPipeline(target, "flip-reference:sample-scene-sdf", sceneSamplerWgsl(scene, outputBinding), "sampleFlipReferenceSolids");
        if (!gpu) {
            throw new Error("[FLIP Reference] GPU runtime is unavailable while preparing scene collisions.");
        }
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: scene.buffer } },
            { binding: 1, resource: { buffer: samplerParamsBuffer } },
            { binding: outputBinding, resource: { buffer: target._solidBuffer } },
            { binding: 4, resource: { buffer: gpu.stateBuffer } },
        ];
        if (scene.sdfGrid) {
            entries.push({ binding: 2, resource: { buffer: scene.sdfGrid } });
        }
        return {
            scene,
            core: target,
            pipeline,
            bindGroup: device.createBindGroup({
                label: "flip-reference:sample-scene-sdf",
                layout: pipeline.getBindGroupLayout(0),
                entries,
            }),
        };
    }

    function prepareCollisionSampler(target: FlipReferenceSimulation): CollisionSampler | null {
        if (!sceneSdf && !target._hasSolidData) {
            return null;
        }
        if (sampler?.scene === sceneSdf && sampler.core === target) {
            return sampler;
        }
        sampler = buildCollisionSampler(target, sceneSdf);
        target._hasSolidData = sceneSdf !== null;
        return sampler;
    }

    function recordStep(encoder: GPUCommandEncoder, dtSeconds: number): void {
        positiveF32("dtSeconds", dtSeconds);
        const target = core;
        if (disposed || !target || !gpu || workingInvalid) {
            throw new Error("[FLIP Reference] the working simulation is unavailable; reset before stepping again.");
        }
        if (target._error) {
            throw target._error;
        }
        if (whitewater?.error) {
            throw whitewater.error;
        }
        applyLiveParameters(target, dtSeconds);
        recordFlipReferenceFrame(
            gpu,
            encoder,
            dtSeconds,
            minSubsteps,
            maxSubsteps,
            maxSubstepSeconds,
            cflNumber,
            prepareCollisionSampler(target),
            profiler,
            engine._currentEncoder !== encoder,
            prepareForce(target)
        );
        if (whitewater) {
            const outsideFrame = engine._currentEncoder !== encoder;
            const endProfile = outsideFrame ? profiler?.commandSpan?.(encoder, "Foam gen") : undefined;
            recordFlipReferenceWhitewater(whitewater, encoder, outsideFrame ? null : profiler);
            endProfile?.();
        }
    }

    function prepareForce(target: FlipReferenceSimulation): { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup } | null {
        if (!forceSpec) {
            return null;
        }
        if (!gpu) {
            throw new Error("[FLIP Reference] GPU runtime is unavailable while preparing a force.");
        }
        if (!forcePipeline || forceStruct !== forceSpec.struct || forceWgsl !== forceSpec.wgsl) {
            forceLayout ??= device.createBindGroupLayout({
                label: "flip-reference:force",
                entries: [
                    { binding: 0, visibility: SS.COMPUTE, buffer: { type: "uniform" } },
                    { binding: 1, visibility: SS.COMPUTE, buffer: { type: "read-only-storage" } },
                    { binding: 2, visibility: SS.COMPUTE, buffer: { type: "storage" } },
                    { binding: 3, visibility: SS.COMPUTE, buffer: { type: "storage" } },
                ],
            });
            forcePipeline = checkedPipeline(
                target,
                "flip-reference:force",
                flipReferenceForceWgsl(forceSpec),
                "applyFlipReferenceForce",
                device.createPipelineLayout({ bindGroupLayouts: [forceLayout] })
            );
            forceStruct = forceSpec.struct;
            forceWgsl = forceSpec.wgsl;
            forceBinding = null;
        }
        if (!forceBinding || forceBinding.core !== target || forceBinding.buffer !== forceSpec.buffer) {
            forceBinding = {
                core: target,
                buffer: forceSpec.buffer,
                pipeline: forcePipeline,
                bindGroup: device.createBindGroup({
                    label: "flip-reference:force",
                    layout: forceLayout!,
                    entries: [
                        { binding: 0, resource: { buffer: forceSpec.buffer } },
                        { binding: 1, resource: { buffer: target._positionBuffer } },
                        { binding: 2, resource: { buffer: target._velocityBuffer } },
                        { binding: 3, resource: { buffer: gpu.stateBuffer } },
                    ],
                }),
            };
        }
        return forceBinding;
    }

    function destroyOwnedResources(): void {
        if (whitewater) {
            disposeFlipReferenceWhitewater(whitewater);
            whitewater = null;
        }
        if (gpu) {
            disposeFlipReferenceGpuRuntime(gpu);
            gpu = null;
        }
        if (core) {
            disposeFlipReferenceSimulation(core);
            core = null;
        }
        coreParamsU32 = null;
        positionBuffer.destroy();
        velocityBuffer.destroy();
        debugBuffer.destroy();
        renderIndirectBuffer.destroy();
        samplerParamsBuffer.destroy();
        sampler = null;
        forceBinding = null;
        forcePipeline = null;
        forceLayout = null;
        forceSpec = null;
        pendingReset = null;
    }

    function finishOperation(): void {
        if (disposed) {
            destroyOwnedResources();
            return;
        }
        if (pendingReset) {
            const seed = pendingReset;
            pendingReset = null;
            replaceCore(seed);
        }
    }

    function activeCount(): number {
        if (gpu && gpu.lastEncoder !== engine._currentEncoder) {
            void collectFlipReferenceGpuStatus(gpu);
        }
        return pendingReset ? publishedActiveCount : (gpu?.publishedCount ?? publishedActiveCount);
    }

    try {
        const initialSeed = createSeed();
        publishSeed(initialSeed);
        replaceCore(initialSeed);
    } catch (error) {
        destroyOwnedResources();
        throw error;
    }

    return {
        count: capacity,
        get activeCount(): number {
            return activeCount();
        },
        get renderCount(): number {
            return activeCount();
        },
        get diffuse(): DiffusePool | undefined {
            if (whitewater && whitewater.lastEncoder !== engine._currentEncoder) {
                void collectFlipReferenceWhitewaterStatus(whitewater);
            }
            return whitewater?.pool;
        },
        get timestepDiagnostics() {
            return pendingReset ? undefined : gpu?.timestepDiagnostics;
        },
        get initialEmitterParticleCounts(): ReadonlyMap<string, number> {
            return initialEmitterParticleCounts;
        },
        particleRadius: config.particleRadius,
        surfaceSizeScale: 1,
        surfaceThicknessScale: 8 / config.flip.markersPerCell,
        surfaceRejectSparseMarkers: true,
        positionBuffer,
        velocityBuffer,
        debugBuffer,
        renderIndirectBuffer,
        debugNorm: 1 / 8,
        get gpuBytes(): number {
            const coreBytes = core?._buffers.reduce((sum, buffer) => sum + buffer.size, 0) ?? 0;
            return (
                coreBytes +
                (gpu?.bytes ?? 0) +
                (whitewater?.bytes ?? 0) +
                positionBuffer.size +
                velocityBuffer.size +
                debugBuffer.size +
                renderIndirectBuffer.size +
                samplerParamsBuffer.size
            );
        },
        prepare(): void {
            if (core) {
                prepareCollisionSampler(core);
                prepareForce(core);
            }
        },
        step(encoder: GPUCommandEncoder, dtSeconds: number): void {
            if (pendingOperation) {
                throw new Error("[FLIP Reference] cannot record a frame during an offline step.");
            }
            recordStep(encoder, dtSeconds);
        },
        async submitStep(dtSeconds: number, beforeSubstep?: (deltaSeconds: number) => void): Promise<void> {
            positiveF32("dtSeconds", dtSeconds);
            if (disposed) {
                throw new Error("[FLIP Reference] cannot step a disposed simulation.");
            }
            if (pendingOperation) {
                throw new Error("[FLIP Reference] an asynchronous step is already in progress.");
            }
            const operationGeneration = generation;
            const target = gpu;
            if (!target || !core) {
                throw new Error("[FLIP Reference] the working simulation is unavailable; reset before stepping again.");
            }
            const operation = (async () => {
                await target.ready;
                await target.core._ready;
                if (disposed || operationGeneration !== generation) {
                    throw abortError(disposed ? "disposal" : "reset");
                }
                beforeSubstep?.(dtSeconds);
                if (disposed || operationGeneration !== generation) {
                    throw abortError(disposed ? "disposal" : "reset");
                }
                const encoder = device.createCommandEncoder({ label: "flip-reference:offline-frame" });
                recordStep(encoder, dtSeconds);
                const submittedWhitewater = whitewater;
                device.queue.submit([encoder.finish()]);
                await collectFlipReferenceGpuStatus(target);
                if (submittedWhitewater) {
                    await collectFlipReferenceWhitewaterStatus(submittedWhitewater);
                    await submittedWhitewater.ready;
                }
                await target.core._ready;
                if (disposed || operationGeneration !== generation) {
                    throw abortError(disposed ? "disposal" : "reset");
                }
                if (target.error) {
                    throw target.error;
                }
                if (target.core._error) {
                    throw target.core._error;
                }
                if (target.telemetryError) {
                    throw target.telemetryError;
                }
                if (submittedWhitewater?.error) {
                    throw submittedWhitewater.error;
                }
            })();
            pendingOperation = operation;
            let failure: unknown;
            try {
                await operation;
            } catch (error) {
                if (!disposed && operationGeneration === generation && error instanceof Error && error.name !== "AbortError") {
                    workingInvalid = true;
                }
                failure = error;
            }
            pendingOperation = null;
            try {
                finishOperation();
            } catch (error) {
                failure = failure === undefined ? error : new AggregateError([failure, error], "[FLIP Reference] step and deferred lifecycle update both failed.");
            }
            if (failure !== undefined) {
                throw failure;
            }
        },
        reset(): void {
            if (disposed) {
                throw new Error("[FLIP Reference] cannot reset a disposed simulation.");
            }
            const seed = createSeed();
            generation++;
            if (whitewater) {
                disposeFlipReferenceWhitewater(whitewater);
                whitewater = null;
            }
            publishSeed(seed);
            if (pendingOperation) {
                pendingReset = seed;
            } else {
                replaceCore(seed);
            }
        },
        setParam(key: string, value: number): void {
            finite(`physics.${key}`, value);
            switch (key) {
                case "gravity":
                    gravity = value;
                    return;
                case "flipRatio":
                    flipRatio = Math.min(1, Math.max(0, value));
                    return;
                case "minSubsteps":
                    minSubsteps = substepLimit("physics.minSubsteps", value);
                    maxSubsteps = Math.max(maxSubsteps, minSubsteps);
                    return;
                case "maxSubsteps":
                    maxSubsteps = Math.max(minSubsteps, substepLimit("physics.maxSubsteps", value));
                    return;
                case "maxSubDtMs":
                    maxSubstepSeconds = positiveF32("maximum substep seconds", value / 1000);
                    return;
                case "cflNumber":
                    cflNumber = cflValue(value);
                    return;
                case "pressureTolerance":
                    pressureTolerance = positiveSquaredF32("physics.pressureTolerance", value);
                    return;
                case "pressureAbsoluteTolerance":
                    pressureAbsoluteTolerance = positiveSquaredF32("physics.pressureAbsoluteTolerance", value);
                    return;
                case "maxPressureIterations":
                    maxPressureIterations = pressureIterationLimit(value);
                    return;
                case "velocityDamping":
                case "kinematicViscosity":
                case "surfaceTension":
                case "polygonSurface":
                case "reseedParticles":
                case "particleSheeting":
                    if (value === 0) {
                        return;
                    }
                    break;
            }
            throw new Error(`[FLIP Reference] physics parameter '${key}' is unsupported by this backend.`);
        },
        setSceneSdf(spec: SceneSdfSpec | null): void {
            if (sceneSdf !== spec) {
                sceneSdf = spec;
                sampler = null;
            }
        },
        setEmitters(config: EmitterConfig | null): void {
            if (config) {
                const adapted = legacyEmitterConfigToFluidFlow(config, capacity);
                validateFlow(adapted);
            }
            flow = null;
        },
        setFlow(config: FluidFlowConfig | null): void {
            validateFlow(config);
            flow = config ? structuredClone(config) : null;
        },
        updateFlowEmitter(emitter: FluidEmitter): void {
            if (!flow) {
                throw new Error(`[FLIP Reference] cannot update unknown emitter '${emitter.id}'.`);
            }
            const index = flow.emitters.findIndex((entry) => entry.id === emitter.id);
            if (index < 0) {
                throw new Error(`[FLIP Reference] cannot update unknown emitter '${emitter.id}'.`);
            }
            if (emitter.enabled && emitter.behavior === "inflow") {
                throw new Error(`[FLIP Reference] active inflow emitter '${emitter.name || emitter.id}' is unsupported.`);
            }
            flow.emitters[index] = structuredClone(emitter);
        },
        setSpawn(min: [number, number, number], max: [number, number, number], accept?: ((x: number, y: number, z: number) => boolean) | null): void {
            const clipped = clippedSpawnBounds(options.bounds.min, options.bounds.max, gridMax, seedMargin, min, max);
            spawnMin = clipped.min;
            spawnMax = clipped.max;
            spawnAccept = accept ?? null;
        },
        setForceField(spec: ForceFieldSpec | null): void {
            forceSpec = spec;
            if (spec && core) {
                prepareForce(core);
            }
        },
        setFoam(config: FoamConfig | null): void {
            if (disposed || !core || !gpu) {
                throw new Error("[FLIP Reference] cannot configure whitewater on an unavailable simulation.");
            }
            if (!config) {
                if (whitewater) {
                    disposeFlipReferenceWhitewater(whitewater);
                    whitewater = null;
                }
                foamConfig = null;
                return;
            }
            const next = { ...config };
            const plan = planFlipReferenceWhitewater(capacity, core._cells, next, device.limits);
            if (plan.errors.length) {
                throw new RangeError("[FLIP Reference] " + plan.errors.join(" "));
            }
            if (whitewater && whitewater.pool.capacity === plan.capacity) {
                configureFlipReferenceWhitewater(whitewater, next);
            } else {
                const replacement = createFlipReferenceWhitewater(core, gpu, capacity, next);
                const previous = whitewater;
                whitewater = replacement;
                if (previous) {
                    disposeFlipReferenceWhitewater(previous);
                }
            }
            foamConfig = next;
        },
        setProfiler(value: FluidProfiler | null): void {
            if (value && !value.stageSpan) {
                throw new Error("[FLIP Reference] GPU timing requires a stage-span profiler.");
            }
            profiler = value;
            if (core) {
                core._profiler = value;
            }
        },
        dispose(): void {
            if (disposed) {
                return;
            }
            disposed = true;
            generation++;
            if (!pendingOperation) {
                destroyOwnedResources();
            }
        },
    };
}
