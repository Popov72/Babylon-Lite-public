import type { EngineContext } from "../../../engine/engine.js";
import { BU } from "../../../engine/gpu-flags.js";
import { FLIP_REFERENCE_BINDINGS, flipReferenceWgsl } from "./shaders.js";
import { markFlipReferenceClosedPockets } from "./closed-pockets.js";
import { FLIP_REFERENCE_PARAMETER_FLOATS, flipReferenceReadbackBytes, flipReferenceStorageSizes } from "./allocation.js";
import type { FlipReferenceDiagnostics, FlipReferenceOptions, FlipReferenceSimulation } from "./types.js";

export type { FlipReferenceDiagnostics, FlipReferenceExtremeRemovalOptions, FlipReferenceOptions, FlipReferenceSimulation } from "./types.js";

const CELL_FLOATS = 16;
const CELL_BYTES = CELL_FLOATS * 4;

function finiteFloat(name: string, value: number): number {
    if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
        throw new RangeError(`FLIP reference: ${name} must be a finite f32 value.`);
    }
    return value;
}

function positiveFloat(name: string, value: number): number {
    finiteFloat(name, value);
    if (!(Math.fround(value) >= 2 ** -126)) {
        throw new RangeError(`FLIP reference: ${name} must be a positive normal f32 value.`);
    }
    return value;
}

function integer(name: string, value: number, minimum: number): number {
    if (!Number.isSafeInteger(value) || value < minimum || value > 0x7fffffff) {
        throw new RangeError(`FLIP reference: ${name} must be an integer in [${minimum}, 2147483647].`);
    }
    return value;
}

function unitInterval(name: string, value: number, positive = false): number {
    finiteFloat(name, value);
    if (value < 0 || value > 1 || (positive && !(Math.fround(value) > 0))) {
        throw new RangeError(`FLIP reference: ${name} must be in ${positive ? "(0" : "[0"}, 1].`);
    }
    return value;
}

function errorValue(error: unknown): Error {
    return error instanceof Error ? error : new Error(String(error));
}

function assertUsable(sim: FlipReferenceSimulation): void {
    if (sim._error) {
        throw sim._error;
    }
    if (sim._disposed) {
        throw new Error("FLIP reference: simulation is disposed.");
    }
    if (sim._busy) {
        throw new Error("FLIP reference: a step or particle readback is already in progress.");
    }
}

function validateSolidData(sim: FlipReferenceSimulation, distances: Float32Array, velocities?: Float32Array): void {
    if (distances.length !== sim._vertices || (velocities !== undefined && velocities.length !== sim._vertices * 3)) {
        throw new RangeError(`FLIP reference: expected ${sim._vertices} nodal distances and ${sim._vertices * 3} nodal velocity components.`);
    }
    for (let i = 0; i < distances.length; i++) {
        if (!Number.isFinite(distances[i]!)) {
            throw new RangeError(`FLIP reference: solidDistances[${i}] must be finite.`);
        }
    }
    if (velocities) {
        for (let i = 0; i < velocities.length; i++) {
            if (!Number.isFinite(velocities[i]!)) {
                throw new RangeError(`FLIP reference: solidVelocities[${i}] must be finite.`);
            }
        }
    }
}

function uploadSolids(sim: FlipReferenceSimulation, distances?: Float32Array, velocities?: Float32Array): void {
    const data = sim._solidUpload;
    const outside = 2 * sim.cellSize * Math.hypot(...sim.gridDimensions);
    for (let i = 0; i < sim._vertices; i++) {
        data[i * 4] = distances ? distances[i]! : outside;
        data[i * 4 + 1] = velocities ? velocities[i * 3]! : 0;
        data[i * 4 + 2] = velocities ? velocities[i * 3 + 1]! : 0;
        data[i * 4 + 3] = velocities ? velocities[i * 3 + 2]! : 0;
    }
    sim._device.queue.writeBuffer(sim._solidBuffer, 0, data);
    sim._hasSolidData = distances !== undefined;
}

/** Creates an isolated, closed-domain dense MAC solver. Shader errors surface at the first async operation. */
export function createFlipReferenceSimulation(engine: EngineContext, options: FlipReferenceOptions): FlipReferenceSimulation {
    const device = engine._device;
    const dimensions: [number, number, number] = [
        integer("gridDimensions[0]", options.gridDimensions[0], 1),
        integer("gridDimensions[1]", options.gridDimensions[1], 1),
        integer("gridDimensions[2]", options.gridDimensions[2], 1),
    ];
    const origin: [number, number, number] = [
        finiteFloat("gridOrigin[0]", options.gridOrigin[0]),
        finiteFloat("gridOrigin[1]", options.gridOrigin[1]),
        finiteFloat("gridOrigin[2]", options.gridOrigin[2]),
    ];
    const dx = positiveFloat("cellSize", options.cellSize);
    const referenceNumerics = options.referenceNumerics ?? false;
    const solidsIncludeDomain = options.solidsIncludeDomain ?? false;
    const removeInsideSolids = options.removeInsideSolids ?? false;
    const extreme = options.extremeVelocityRemoval;
    const extremeRemoval = extreme !== undefined;
    const removalEnabled = removeInsideSolids || extremeRemoval;
    const kernel = options.kernel ?? (referenceNumerics ? "wyvill" : "trilinear");
    const advection = options.advection ?? (referenceNumerics ? "rk3" : "rk2");
    const collisionMode = options.collisionMode ?? (referenceNumerics ? "sweep" : "project");
    const constrainSnapshot = options.constrainSnapshot ?? referenceNumerics;
    const solidVolumeCorrection = options.solidVolumeCorrection ?? true;
    const geometryFractions = options.geometryFractions ?? (referenceNumerics ? "reference" : "center");
    if (
        typeof referenceNumerics !== "boolean" ||
        typeof constrainSnapshot !== "boolean" ||
        typeof solidVolumeCorrection !== "boolean" ||
        typeof solidsIncludeDomain !== "boolean" ||
        typeof removeInsideSolids !== "boolean"
    ) {
        throw new TypeError("FLIP reference: numerical, domain, and removal switches must be booleans.");
    }
    let histogramBins = 1;
    let extremeBinWidth = 0;
    if (extremeRemoval) {
        if (extreme === null || typeof extreme !== "object") {
            throw new TypeError("FLIP reference: extremeVelocityRemoval must specify frameDtSeconds, cfl, and maxFrameSubsteps.");
        }
        const frameDt = positiveFloat("extremeVelocityRemoval.frameDtSeconds", extreme.frameDtSeconds);
        const cfl = positiveFloat("extremeVelocityRemoval.cfl", extreme.cfl);
        histogramBins = integer("extremeVelocityRemoval.maxFrameSubsteps", extreme.maxFrameSubsteps, 1);
        if (histogramBins > 65535) {
            throw new RangeError("FLIP reference: maxFrameSubsteps must not exceed 65535 histogram bins.");
        }
        extremeBinWidth = positiveFloat("extreme removal bin width", (cfl * dx) / frameDt);
        positiveFloat("squared extreme removal threshold", ((histogramBins + 3) * extremeBinWidth) ** 2);
    }
    const gravity = options.gravity ?? [0, -9.81, 0];
    for (let axis = 0; axis < 3; axis++) {
        finiteFloat(`gravity[${axis}]`, gravity[axis]!);
        finiteFloat(`domain upper bound[${axis}]`, origin[axis]! + dimensions[axis]! * dx);
        if (Math.fround(origin[axis]! + dx) <= Math.fround(origin[axis]!) || Math.fround(origin[axis]! + dimensions[axis]! * dx) <= Math.fround(origin[axis]!)) {
            throw new RangeError("FLIP reference: grid spacing is not representable at the requested f32 origin.");
        }
    }
    positiveFloat("domain diagonal", 2 * dx * Math.hypot(...dimensions));
    if (options.initialPositions.length % 3 !== 0 || options.initialVelocities.length !== options.initialPositions.length) {
        throw new RangeError("FLIP reference: initial positions and velocities must have equal, tightly packed XYZ lengths.");
    }
    const count = integer("particle count", options.initialPositions.length / 3, 0);
    const cellCount = integer("cell count", dimensions[0] * dimensions[1] * dimensions[2], 1);
    const uCount = (dimensions[0] + 1) * dimensions[1] * dimensions[2];
    const vCount = dimensions[0] * (dimensions[1] + 1) * dimensions[2];
    const wCount = dimensions[0] * dimensions[1] * (dimensions[2] + 1);
    const faceCount = integer("face count", uCount + vCount + wCount, 1);
    const vertexCount = integer("vertex count", (dimensions[0] + 1) * (dimensions[1] + 1) * (dimensions[2] + 1), 1);
    const picFraction = unitInterval("picFraction", options.picFraction ?? 0.05);
    const relativeTolerance = positiveFloat("pressureTolerance", options.pressureTolerance ?? 1e-6);
    const absoluteTolerance = positiveFloat("pressureAbsoluteTolerance", options.pressureAbsoluteTolerance ?? 1e-8);
    positiveFloat("squared pressureTolerance", relativeTolerance ** 2);
    positiveFloat("squared pressureAbsoluteTolerance", absoluteTolerance ** 2);
    const maxIterations = integer("maxPressureIterations", options.maxPressureIterations ?? 400, 1);
    if (maxIterations > 16_777_216) {
        throw new RangeError("FLIP reference: the GPU iteration counter supports at most 16777216 iterations.");
    }
    const thetaMin = unitInterval("ghostFluidThetaMin", options.ghostFluidThetaMin ?? (referenceNumerics ? 1 / 26 : 0.01), true);
    const liquidRadius = positiveFloat("liquidRadius", options.liquidRadius ?? (Math.sqrt(3) / 2) * dx);
    const transferRadius = positiveFloat("transferRadius", options.transferRadius ?? (kernel === "wyvill" ? (Math.sqrt(3) / 2) * dx : dx));
    const particleRadius = positiveFloat("particleRadius", options.particleRadius ?? 0.1 * dx);
    const collisionRadius = positiveFloat("collisionRadius", options.collisionRadius ?? (referenceNumerics ? 0.2 : 0.01) * dx);
    const domainInset = finiteFloat("domainInset", options.domainInset ?? (referenceNumerics ? 1.5 * dx + 5e-5 : 0));
    const domainCollisionRadius = positiveFloat("domainCollisionRadius", options.domainCollisionRadius ?? (referenceNumerics ? 0.1 * dx : collisionRadius));
    if (domainInset < 0 || 2 * (domainInset + domainCollisionRadius) >= dx * Math.min(...dimensions)) {
        throw new RangeError("FLIP reference: domainInset and domainCollisionRadius must leave a positive closed-domain interior.");
    }
    if (collisionRadius * 2 >= dx * Math.min(...dimensions)) {
        throw new RangeError("FLIP reference: collisionRadius leaves no interior in the requested closed domain.");
    }
    if (Math.max(liquidRadius, transferRadius) / dx > Math.max(...dimensions)) {
        throw new RangeError("FLIP reference: liquid and transfer radii must not exceed the longest domain extent.");
    }
    if (kernel === "trilinear" && options.transferRadius !== undefined && transferRadius !== dx) {
        throw new RangeError("FLIP reference: trilinear transfer support is fixed at cellSize.");
    }
    const layers = integer("extrapolationLayers", options.extrapolationLayers ?? (referenceNumerics ? 12 : 8), 0);
    const advectionSubsteps = integer("maxAdvectionSubsteps", options.maxAdvectionSubsteps ?? 256, 1);
    if (kernel !== "trilinear" && kernel !== "radial" && kernel !== "wyvill") {
        throw new RangeError("FLIP reference: kernel must be 'trilinear', 'radial', or 'wyvill'.");
    }
    if (advection !== "rk2" && advection !== "rk3") {
        throw new RangeError("FLIP reference: advection must be 'rk2' or 'rk3'.");
    }
    if (collisionMode !== "project" && collisionMode !== "sweep") {
        throw new RangeError("FLIP reference: collisionMode must be 'project' or 'sweep'.");
    }
    if (geometryFractions !== "center" && geometryFractions !== "reference") {
        throw new RangeError("FLIP reference: geometryFractions must be 'center' or 'reference'.");
    }
    if (options.solidVelocities && !options.solidDistances) {
        throw new RangeError("FLIP reference: solidVelocities requires solidDistances.");
    }
    const reductionGroups = Math.ceil(cellCount / 128);
    const particleGroups = Math.max(1, Math.ceil(count / 128));
    const removalControlWord = count * 2 + particleGroups;
    if (Math.ceil(Math.max(cellCount, faceCount, count, histogramBins, 8) / 128) > device.limits.maxComputeWorkgroupsPerDimension) {
        throw new RangeError("FLIP reference: requested grid/particle dispatch exceeds the device workgroup limit.");
    }
    if (device.limits.maxComputeInvocationsPerWorkgroup < 128 || device.limits.maxComputeWorkgroupSizeX < 128 || device.limits.maxStorageBuffersPerShaderStage < 7) {
        throw new RangeError("FLIP reference: device requires 128-invocation workgroups and seven compute storage buffers.");
    }
    const storageSizes = flipReferenceStorageSizes(cellCount, faceCount, vertexCount, count, removalEnabled, histogramBins);
    for (const size of storageSizes) {
        if (!Number.isSafeInteger(size) || size > device.limits.maxStorageBufferBindingSize || size > device.limits.maxBufferSize) {
            throw new RangeError(`FLIP reference: requested ${size}-byte storage allocation exceeds device limits.`);
        }
    }
    const readbackBytes = flipReferenceReadbackBytes(cellCount, count);
    if (readbackBytes > device.limits.maxBufferSize) {
        throw new RangeError("FLIP reference: readback allocation exceeds device limits.");
    }
    const packedPositions = new Float32Array(Math.max(4, count * 4));
    const packedVelocities = new Float32Array(Math.max(4, count * 4));
    const initialSpeeds = new Float32Array(Math.max(1, count));
    for (let i = 0; i < count; i++) {
        for (let axis = 0; axis < 3; axis++) {
            const p = finiteFloat(`initialPositions[${i * 3 + axis}]`, options.initialPositions[i * 3 + axis]!);
            const v = finiteFloat(`initialVelocities[${i * 3 + axis}]`, options.initialVelocities[i * 3 + axis]!);
            const q = Math.fround(Math.fround(p - Math.fround(origin[axis]!)) / Math.fround(dx));
            if (q < 0 || q >= dimensions[axis]!) {
                throw new RangeError(`FLIP reference: initial marker ${i} is outside the half-open grid domain on axis ${axis}.`);
            }
            packedPositions[i * 4 + axis] = p;
            packedVelocities[i * 4 + axis] = v;
        }
        packedPositions[i * 4 + 3] = 1;
        initialSpeeds[i] = finiteFloat("initial marker speed", Math.hypot(packedVelocities[i * 4]!, packedVelocities[i * 4 + 1]!, packedVelocities[i * 4 + 2]!));
    }
    const params = new Float32Array(FLIP_REFERENCE_PARAMETER_FLOATS);
    const uints = new Uint32Array(params.buffer);
    uints.set([...dimensions, count], 0);
    params.set([...origin, dx], 4);
    params.set([...gravity, 0], 8);
    params.set([picFraction, thetaMin, liquidRadius / dx, collisionRadius], 12);
    uints.set([cellCount, faceCount, uCount, vCount], 16);
    uints.set([kernel === "wyvill" ? 2 : kernel === "radial" ? 1 : 0, advection === "rk3" ? 1 : 0, advectionSubsteps, reductionGroups], 20);
    params.set([relativeTolerance ** 2, absoluteTolerance ** 2, 0, 0], 24);
    params.set([transferRadius / dx, domainInset, domainCollisionRadius, referenceNumerics || kernel === "wyvill" ? 1e-6 : 0], 28);
    uints.set(
        [
            Number(referenceNumerics),
            (geometryFractions === "reference" ? 1 : 0) | (constrainSnapshot ? 2 : 0) | (solidsIncludeDomain ? 4 : 0),
            Number(solidVolumeCorrection),
            collisionMode === "sweep" ? 1 : 0,
        ],
        32
    );
    uints.set([count, particleGroups, histogramBins, (removeInsideSolids ? 1 : 0) | (extremeRemoval ? 2 : 0)], 36);
    params.set([extremeBinWidth, 0, 0, 0], 40);
    const buffers: GPUBuffer[] = [];
    let scopesOpen = true;
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    try {
        const allocate = (label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer => {
            const buffer = device.createBuffer({ label: `flip-reference:${label}`, size, usage });
            buffers.push(buffer);
            return buffer;
        };
        const storageUsage = BU.STORAGE | BU.COPY_DST | BU.COPY_SRC;
        const positionBuffer = allocate("positions", storageSizes[0]!, storageUsage);
        const velocityBuffer = allocate("velocities", storageSizes[1]!, storageUsage);
        const debugBuffer = allocate("speeds", storageSizes[2]!, storageUsage);
        const faceBuffer = allocate("faces", storageSizes[3]!, storageUsage);
        const cellBuffer = allocate("cells", storageSizes[4]!, storageUsage);
        const listBuffer = allocate("lists", storageSizes[5]!, storageUsage);
        const solidBuffer = allocate("solids", storageSizes[6]!, storageUsage);
        const pressureBuffer = allocate("pcg", storageSizes[7]!, storageUsage);
        const particleScratchBuffer = allocate("particle-scratch", storageSizes[8]!, storageUsage);
        const particleStateBuffer = allocate("particle-state", storageSizes[9]!, storageUsage);
        const uniformBuffer = allocate("params", params.byteLength, BU.UNIFORM | BU.COPY_DST);
        const readbackBuffer = allocate("readback", readbackBytes, BU.MAP_READ | BU.COPY_DST);
        const bindingBuffers = [
            uniformBuffer,
            positionBuffer,
            velocityBuffer,
            debugBuffer,
            faceBuffer,
            cellBuffer,
            listBuffer,
            solidBuffer,
            pressureBuffer,
            particleScratchBuffer,
            particleStateBuffer,
        ];
        const module = device.createShaderModule({ label: "flip-reference", code: flipReferenceWgsl() });
        const pipelines: Record<string, GPUComputePipeline> = {};
        const bindGroups: Record<string, GPUBindGroup> = {};
        for (const [entryPoint, bindings] of Object.entries(FLIP_REFERENCE_BINDINGS)) {
            const pipeline = device.createComputePipeline({ label: `flip-reference:${entryPoint}`, layout: "auto", compute: { module, entryPoint } });
            pipelines[entryPoint] = pipeline;
            bindGroups[entryPoint] = device.createBindGroup({
                label: `flip-reference:${entryPoint}`,
                layout: pipeline.getBindGroupLayout(0),
                entries: bindings.map((binding) => ({ binding, resource: { buffer: bindingBuffers[binding]! } })),
            });
        }
        const sim: FlipReferenceSimulation = {
            gridOrigin: origin,
            gridDimensions: dimensions,
            cellSize: dx,
            get count(): number {
                return sim._count;
            },
            _count: count,
            capacity: count,
            totalRemovedInsideSolids: 0,
            totalRemovedExtremeVelocities: 0,
            particleRadius,
            referenceNumerics,
            solidsIncludeDomain,
            elapsedSeconds: 0,
            diagnostics: null,
            _positionBuffer: positionBuffer,
            _velocityBuffer: velocityBuffer,
            _debugBuffer: debugBuffer,
            _device: device,
            _buffers: buffers,
            _uniformBuffer: uniformBuffer,
            _faceBuffer: faceBuffer,
            _cellBuffer: cellBuffer,
            _listBuffer: listBuffer,
            _solidBuffer: solidBuffer,
            _pressureBuffer: pressureBuffer,
            _particleScratchBuffer: particleScratchBuffer,
            _particleStateBuffer: particleStateBuffer,
            _removalControlOffset: removalControlWord * 4,
            _removalReadback: new Uint32Array(8),
            _removalEnabled: removalEnabled,
            _extremeRemoval: extremeRemoval,
            _removalHistogramBins: histogramBins,
            _readbackBuffer: readbackBuffer,
            _pipelines: pipelines,
            _bindGroups: bindGroups,
            _params: params,
            _cells: cellCount,
            _faces: faceCount,
            _vertices: vertexCount,
            _reductionGroups: reductionGroups,
            _cellReadback: new Float32Array(cellCount * CELL_FLOATS + 8),
            _controlReadback: new Float32Array(8),
            _statusReadback: new Uint32Array(8),
            _componentVisited: new Uint8Array(cellCount),
            _componentQueue: new Int32Array(cellCount),
            _solidUpload: new Float32Array(vertexCount * 4),
            _hasSolidData: false,
            _pressureTolerance: relativeTolerance,
            _pressureAbsoluteTolerance: absoluteTolerance,
            _maxPressureIterations: maxIterations,
            _extrapolationLayers: layers,
            _constrainSnapshot: constrainSnapshot,
            _solidVolumeCorrection: solidVolumeCorrection,
            _ready: Promise.resolve(),
            _error: null,
            _busy: false,
            _disposed: false,
        };
        if (options.solidDistances) {
            validateSolidData(sim, options.solidDistances, options.solidVelocities);
        }
        device.queue.writeBuffer(positionBuffer, 0, packedPositions);
        device.queue.writeBuffer(velocityBuffer, 0, packedVelocities);
        device.queue.writeBuffer(debugBuffer, 0, initialSpeeds);
        device.queue.writeBuffer(uniformBuffer, 0, params);
        uploadSolids(sim, options.solidDistances, options.solidVelocities);
        const allocationError = device.popErrorScope();
        const validationError = device.popErrorScope();
        scopesOpen = false;
        sim._ready = Promise.all([module.getCompilationInfo(), allocationError, validationError])
            .then(([info, allocation, validation]) => {
                const messages = info.messages.filter((message) => message.type === "error");
                if (allocation || validation || messages.length) {
                    throw new Error(`FLIP reference initialization failed: ${allocation?.message ?? validation?.message ?? messages.map((message) => message.message).join("\n")}`);
                }
            })
            .catch((error: unknown) => {
                sim._error = errorValue(error);
                disposeFlipReferenceSimulation(sim);
            });
        return sim;
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

/** Replaces the complete nodal collision field before a physical step; no geometry is inferred from rendering. */
export function updateFlipReferenceSolids(sim: FlipReferenceSimulation, distances: Float32Array, velocities: Float32Array): void {
    assertUsable(sim);
    validateSolidData(sim, distances, velocities);
    uploadSolids(sim, distances, velocities);
}

function dispatch(sim: FlipReferenceSimulation, encoder: GPUCommandEncoder, entry: string, count: number): void {
    if (count === 0) {
        return;
    }
    const pass = encoder.beginComputePass({ label: `flip-reference:${entry}` });
    pass.setPipeline(sim._pipelines[entry]!);
    pass.setBindGroup(0, sim._bindGroups[entry]!);
    pass.dispatchWorkgroups(Math.ceil(count / 128));
    pass.end();
}

function commandEncoder(sim: FlipReferenceSimulation, label: string): GPUCommandEncoder {
    const encoder = sim._device.createCommandEncoder({ label });
    sim._profileEnd = sim._profiler?.commandSpan?.(encoder, "Simulation");
    return encoder;
}

function extend(sim: FlipReferenceSimulation, encoder: GPUCommandEncoder, beforeForces = false): void {
    for (let layer = 0; layer < sim._extrapolationLayers; layer++) {
        const entry = beforeForces ? (layer % 2 === 0 ? "extendBeforeToB" : "extendBeforeToA") : layer % 2 === 0 ? "extendToB" : "extendToA";
        dispatch(sim, encoder, entry, sim._faces);
    }
    if (sim._extrapolationLayers % 2 !== 0) {
        dispatch(sim, encoder, "copyToA", sim._faces);
    }
}

async function readback(sim: FlipReferenceSimulation, encoder: GPUCommandEncoder, target: Float32Array | Uint32Array): Promise<void> {
    sim._profileEnd?.();
    sim._profileEnd = undefined;
    sim._device.queue.submit([encoder.finish()]);
    await sim._readbackBuffer.mapAsync(GPUMapMode.READ, 0, target.byteLength);
    try {
        const mapped = sim._readbackBuffer.getMappedRange(0, target.byteLength);
        new Uint8Array(target.buffer, target.byteOffset, target.byteLength).set(new Uint8Array(mapped));
    } finally {
        sim._readbackBuffer.unmap();
    }
    if (sim._disposed) {
        throw new Error("FLIP reference: simulation was disposed during a GPU operation.");
    }
}

function reduce(sim: FlipReferenceSimulation, encoder: GPUCommandEncoder, finish: string): void {
    dispatch(sim, encoder, "reduceProducts", sim._cells);
    dispatch(sim, encoder, finish, 1);
}

async function readControl(sim: FlipReferenceSimulation, encoder: GPUCommandEncoder): Promise<void> {
    encoder.copyBufferToBuffer(sim._pressureBuffer, (sim._cells + sim._reductionGroups) * 16, sim._readbackBuffer, 0, 32);
    await readback(sim, encoder, sim._controlReadback);
}

async function conditionClosedPockets(sim: FlipReferenceSimulation): Promise<{ components: number; cells: number }> {
    if (!sim.referenceNumerics) {
        return { components: 0, cells: 0 };
    }
    const rows = sim._cellReadback;
    const result = markFlipReferenceClosedPockets(rows, sim.gridDimensions, sim._componentVisited, sim._componentQueue);
    if (result.components > 0) {
        sim._device.queue.writeBuffer(sim._cellBuffer, 0, rows.buffer, rows.byteOffset, sim._cells * CELL_BYTES);
        const encoder = commandEncoder(sim, "flip-reference:condition-closed-pockets");
        dispatch(sim, encoder, "conditionSolidVelocities", sim._faces);
        dispatch(sim, encoder, "buildMatrix", sim._cells);
        encoder.copyBufferToBuffer(sim._cellBuffer, 0, sim._readbackBuffer, 0, sim._cells * CELL_BYTES);
        encoder.copyBufferToBuffer(sim._listBuffer, (sim._cells + sim.capacity) * 4, sim._readbackBuffer, sim._cells * CELL_BYTES, 32);
        await readback(sim, encoder, rows);
    }
    return result;
}

function fixPressureGauges(sim: FlipReferenceSimulation): { fluidCells: number; sealedComponents: number; maxFlux: number } {
    const rows = sim._cellReadback;
    const visited = sim._componentVisited;
    const queue = sim._componentQueue;
    const strides = [1, sim.gridDimensions[0], sim.gridDimensions[0] * sim.gridDimensions[1]];
    const gauge = new Float32Array([1]);
    visited.fill(0);
    let fluidCells = 0;
    let sealedComponents = 0;
    let maxFlux = 0;
    for (let root = 0; root < sim._cells; root++) {
        if (rows[root * CELL_FLOATS + 3] === 0 || visited[root]) {
            continue;
        }
        let head = 0;
        let tail = 1;
        let hasAir = false;
        let flux = 0;
        let normSquared = 0;
        queue[0] = root;
        visited[root] = 1;
        while (head < tail) {
            const index = queue[head++]!;
            const offset = index * CELL_FLOATS;
            const rhs = rows[offset + 2]!;
            if (!Number.isFinite(rhs) || !Number.isFinite(rows[offset + 1]!)) {
                throw new Error("FLIP reference: non-finite pressure matrix or boundary flux.");
            }
            hasAir ||= rows[offset + 11] !== 0;
            flux += rhs;
            normSquared += rhs * rhs;
            for (let axis = 0; axis < 3; axis++) {
                for (let sign = -1; sign <= 1; sign += 2) {
                    const coefficient = rows[offset + (sign > 0 ? 4 : 8) + axis]!;
                    if (coefficient > 0) {
                        const neighbor = index + sign * strides[axis]!;
                        if (!visited[neighbor]) {
                            visited[neighbor] = 1;
                            queue[tail++] = neighbor;
                        }
                    }
                }
            }
        }
        fluidCells += tail;
        if (!hasAir) {
            sealedComponents++;
            maxFlux = Math.max(maxFlux, Math.abs(flux));
            const compatibleTolerance = 4 * Math.sqrt(tail) * Math.max(sim._pressureAbsoluteTolerance, sim._pressureTolerance * Math.sqrt(normSquared));
            if (Math.abs(flux) > compatibleTolerance) {
                throw new Error(`FLIP reference: closed liquid component (${tail} cells) has incompatible moving-solid flux ${flux}; incompressible projection has no solution.`);
            }
            // Removing the anchor's off-diagonals symmetrically fixes the Neumann gauge without a diagonal leak.
            sim._device.queue.writeBuffer(sim._cellBuffer, root * CELL_BYTES + 28, gauge);
        }
    }
    return { fluidCells, sealedComponents, maxFlux };
}

async function solvePressure(sim: FlipReferenceSimulation): Promise<void> {
    let encoder = commandEncoder(sim, "flip-reference:initialize-pressure");
    dispatch(sim, encoder, "initializeCg", sim._cells);
    reduce(sim, encoder, "finishInitialize");
    await readControl(sim, encoder);
    const control = sim._controlReadback;
    let previousRestartIteration = -1;
    while (true) {
        while (control[3] !== 0 && control[6]! < sim._maxPressureIterations) {
            encoder = commandEncoder(sim, "flip-reference:pcg-batch");
            const batch = Math.min(8, sim._maxPressureIterations - control[6]!);
            for (let iteration = 0; iteration < batch; iteration++) {
                dispatch(sim, encoder, "applyDirection", sim._cells);
                reduce(sim, encoder, "finishDot");
                dispatch(sim, encoder, "updateResidual", sim._cells);
                reduce(sim, encoder, "finishResidual");
                dispatch(sim, encoder, "updateDirection", sim._cells);
            }
            await readControl(sim, encoder);
            if (control[7] !== 0) {
                throw new Error(`FLIP reference: PCG breakdown (${control[7]}) after ${control[6]} iterations.`);
            }
        }
        encoder = commandEncoder(sim, "flip-reference:true-residual");
        dispatch(sim, encoder, "trueResidual", sim._cells);
        reduce(sim, encoder, "finishRestart");
        await readControl(sim, encoder);
        if (control[7] !== 0) {
            throw new Error("FLIP reference: pressure solve produced a non-finite true residual.");
        }
        if (control[3] === 0 || control[6]! >= sim._maxPressureIterations) {
            return;
        }
        if (control[6] === previousRestartIteration) {
            throw new Error("FLIP reference: pressure residual replacement made no progress.");
        }
        previousRestartIteration = control[6]!;
    }
}

function checkParticleStatus(status: Uint32Array): void {
    if (status[0] !== 0) {
        throw new Error(`FLIP reference: ${status[0]} markers are non-finite or outside the grid; none were silently removed.`);
    }
    if (status[1] !== 0) {
        throw new Error("FLIP reference: solid penetration correction failed (flat SDF gradient or unresolved contact).");
    }
    if (status[5] !== 0) {
        throw new Error("FLIP reference: maxAdvectionSubsteps was exhausted; use a smaller explicit physical step.");
    }
    if (status[6] !== 0) {
        throw new Error("FLIP reference: marker advection left the extended velocity band; increase extrapolationLayers or reduce the explicit timestep.");
    }
}

async function compactParticles(sim: FlipReferenceSimulation): Promise<void> {
    const before = sim.count;
    const encoder = commandEncoder(sim, "flip-reference:particle-removal");
    if (sim._extremeRemoval) {
        dispatch(sim, encoder, "countExtremeGroups", before);
        dispatch(sim, encoder, "chooseExtremeThreshold", 1);
    }
    dispatch(sim, encoder, "scanSurvivors", before);
    dispatch(sim, encoder, "scanSurvivorGroups", 1);
    dispatch(sim, encoder, "packSurvivors", before);
    dispatch(sim, encoder, "commitSurvivors", before);
    encoder.copyBufferToBuffer(sim._particleStateBuffer, sim._removalControlOffset, sim._readbackBuffer, 0, 32);
    await readback(sim, encoder, sim._removalReadback);
    const remaining = sim._removalReadback[0]!;
    const inside = sim._removalReadback[1]!;
    const extreme = sim._removalReadback[2]!;
    const floats = new Float32Array(sim._removalReadback.buffer);
    if (remaining + inside + extreme !== before || remaining > sim.capacity || !Number.isFinite(floats[6]!)) {
        throw new Error("FLIP reference: particle compaction violated its count or finite-speed contract.");
    }
    sim._count = remaining;
    sim.totalRemovedInsideSolids += inside;
    sim.totalRemovedExtremeVelocities += extreme;
    new Uint32Array(sim._params.buffer)[3] = remaining;
    sim._device.queue.writeBuffer(sim._uniformBuffer, 12, sim._params.buffer, 12, 4);
}

/** Consumes exactly one declared physical timestep. No frame scheduler, dropped time, or hidden reseeding. */
export async function stepFlipReferenceSimulation(sim: FlipReferenceSimulation, dtSeconds: number): Promise<FlipReferenceDiagnostics> {
    assertUsable(sim);
    positiveFloat("dtSeconds", dtSeconds);
    if (sim.solidsIncludeDomain && !sim._hasSolidData) {
        throw new Error("FLIP reference: supply the nodal domain field with updateFlipReferenceSolids before stepping.");
    }
    for (let axis = 0; axis < 3; axis++) {
        finiteFloat("gravity impulse", sim._params[8 + axis]! * dtSeconds);
    }
    if (!Number.isFinite(sim.elapsedSeconds + dtSeconds) || sim.elapsedSeconds + dtSeconds === sim.elapsedSeconds) {
        throw new RangeError("FLIP reference: timestep cannot advance the simulation clock.");
    }
    sim._busy = true;
    const particleCountBeforeRemoval = sim.count;
    try {
        await sim._ready;
        if (sim._error) {
            throw sim._error;
        }
        if (sim._disposed) {
            throw new Error("FLIP reference: simulation is disposed.");
        }
        sim._params[11] = dtSeconds;
        sim._device.queue.writeBuffer(sim._uniformBuffer, 0, sim._params);
        let encoder = commandEncoder(sim, "flip-reference:transfer-and-system");
        dispatch(sim, encoder, "clearLists", Math.max(8, sim._cells));
        dispatch(sim, encoder, "linkParticles", sim.count);
        dispatch(sim, encoder, "liquidLevelSet", sim._cells);
        dispatch(sim, encoder, "particleToGrid", sim._faces);
        extend(sim, encoder, true);
        dispatch(sim, encoder, "snapshotAndForce", sim._faces);
        dispatch(sim, encoder, "buildMatrix", sim._cells);
        encoder.copyBufferToBuffer(sim._cellBuffer, 0, sim._readbackBuffer, 0, sim._cells * CELL_BYTES);
        encoder.copyBufferToBuffer(sim._listBuffer, (sim._cells + sim.capacity) * 4, sim._readbackBuffer, sim._cells * CELL_BYTES, 32);
        await readback(sim, encoder, sim._cellReadback);
        checkParticleStatus(new Uint32Array(sim._cellReadback.buffer, sim._cells * CELL_BYTES, 8));
        const conditioning = await conditionClosedPockets(sim);
        const components = fixPressureGauges(sim);
        await solvePressure(sim);
        encoder = commandEncoder(sim, "flip-reference:projection-and-advection");
        dispatch(sim, encoder, "project", sim._faces);
        dispatch(sim, encoder, "measureDivergence", sim._cells);
        extend(sim, encoder, sim.referenceNumerics);
        if (sim.referenceNumerics || sim._constrainSnapshot) {
            dispatch(sim, encoder, "constrainSnapshots", sim._faces);
        }
        if (sim._removalEnabled) {
            dispatch(sim, encoder, "clearRemoval", Math.max(8, sim._removalHistogramBins));
        }
        dispatch(sim, encoder, "gridToParticles", sim.count);
        encoder.copyBufferToBuffer(sim._listBuffer, (sim._cells + sim.capacity) * 4, sim._readbackBuffer, 0, 32);
        await readback(sim, encoder, sim._statusReadback);
        checkParticleStatus(sim._statusReadback);
        const floats = new Float32Array(sim._statusReadback.buffer);
        if (!Number.isFinite(floats[3]!) || !Number.isFinite(floats[4]!)) {
            throw new Error("FLIP reference: non-finite projected divergence or particle speed.");
        }
        let maxSpeed = floats[4]!;
        let removedInsideSolids = 0;
        let removedExtremeVelocities = 0;
        let extremeSpeedThreshold: number | null = null;
        if (sim._removalEnabled) {
            await compactParticles(sim);
            const removalFloats = new Float32Array(sim._removalReadback.buffer);
            maxSpeed = removalFloats[6]!;
            removedInsideSolids = sim._removalReadback[1]!;
            removedExtremeVelocities = sim._removalReadback[2]!;
            if (sim._extremeRemoval) {
                extremeSpeedThreshold = removalFloats[3]!;
            }
        }
        sim.elapsedSeconds += dtSeconds;
        const absoluteResidual = Math.sqrt(sim._controlReadback[1]!);
        const rhsNorm = Math.sqrt(sim._controlReadback[2]!);
        const converged = sim._controlReadback[3] === 0;
        const diagnostics: FlipReferenceDiagnostics = {
            consumedDtSeconds: dtSeconds,
            elapsedSeconds: sim.elapsedSeconds,
            particleCount: sim.count,
            particleCapacity: sim.capacity,
            particleCountBeforeRemoval,
            removedParticleCount: particleCountBeforeRemoval - sim.count,
            removedInsideSolids,
            removedExtremeVelocities,
            totalRemovedInsideSolids: sim.totalRemovedInsideSolids,
            totalRemovedExtremeVelocities: sim.totalRemovedExtremeVelocities,
            fluidCellCount: components.fluidCells,
            pressureIterations: sim._controlReadback[6]!,
            converged,
            pressureConverged: converged,
            pressureResidualNorm: "l2",
            pressureResidualUnits: "velocity",
            pressureAbsoluteResidual: absoluteResidual,
            pressureRelativeResidual: rhsNorm > 0 ? absoluteResidual / rhsNorm : 0,
            pressureRhsNorm: rhsNorm,
            maxDivergence: floats[3]!,
            sealedComponentCount: components.sealedComponents,
            conditionedComponentCount: conditioning.components,
            conditionedCellCount: conditioning.cells,
            maxSealedComponentFlux: components.maxFlux,
            collisionCount: sim._statusReadback[2]!,
            maxSpeed,
            maxSpeedBeforeRemoval: floats[4]!,
            extremeSpeedThreshold,
            collisionFallbackCount: sim._statusReadback[7]!,
            referenceNumerics: sim.referenceNumerics,
            solidsIncludeDomain: sim.solidsIncludeDomain,
            solidVolumeCorrection: sim._solidVolumeCorrection,
        };
        sim.diagnostics = diagnostics;
        return diagnostics;
    } catch (error) {
        sim._error = errorValue(error);
        throw error;
    } finally {
        sim._busy = false;
    }
}

/** Downloads the live prefix as tightly packed XYZ arrays. Compaction preserves the relative order of survivors. */
export async function readFlipReferenceParticles(sim: FlipReferenceSimulation): Promise<{ positions: Float32Array; velocities: Float32Array }> {
    assertUsable(sim);
    sim._busy = true;
    try {
        await sim._ready;
        if (sim._error) {
            throw sim._error;
        }
        if (sim._disposed) {
            throw new Error("FLIP reference: simulation is disposed.");
        }
        const positions = new Float32Array(sim.count * 3);
        const velocities = new Float32Array(sim.count * 3);
        if (sim.count > 0) {
            const packed = new Float32Array(sim.count * 8);
            const bytes = sim.count * 16;
            const encoder = sim._device.createCommandEncoder({ label: "flip-reference:read-particles" });
            encoder.copyBufferToBuffer(sim._positionBuffer, 0, sim._readbackBuffer, 0, bytes);
            encoder.copyBufferToBuffer(sim._velocityBuffer, 0, sim._readbackBuffer, bytes, bytes);
            await readback(sim, encoder, packed);
            for (let i = 0; i < sim.count; i++) {
                for (let axis = 0; axis < 3; axis++) {
                    positions[i * 3 + axis] = packed[i * 4 + axis]!;
                    velocities[i * 3 + axis] = packed[sim.count * 4 + i * 4 + axis]!;
                }
            }
        }
        return { positions, velocities };
    } finally {
        sim._busy = false;
    }
}

/** Releases all owned buffers. Safe to call repeatedly, including after a failed initialization or step. */
export function disposeFlipReferenceSimulation(sim: FlipReferenceSimulation): void {
    if (sim._disposed) {
        return;
    }
    sim._disposed = true;
    for (const buffer of sim._buffers) {
        buffer.destroy();
    }
    sim._buffers.length = 0;
    sim._pipelines = {};
    sim._bindGroups = {};
}
