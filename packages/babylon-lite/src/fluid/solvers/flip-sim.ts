// GPU FLIP liquid backend using marker particles and a staggered MAC grid.
//
// References:
//   Robert Bridson, Fluid Simulation for Computer Graphics
//   https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf
//   Zhu & Bridson 2005, Animating Sand as a Fluid
//   https://www.cs.ubc.ca/~rbridson/docs/zhu-siggraph05-sandfluid.pdf
//   GridFluidSim3D
//   https://github.com/rlguy/GridFluidSim3D

import type { EngineContext } from "../../engine/engine.js";
import { createFluidForcePasses, updateFluidForcePasses, type FluidForcePasses } from "../core/force-field-passes.js";
import { FLIP_PAGE_DISPATCH_BYTES, FLIP_PAGE_DISPATCH_CONFIG_BYTES, FLIP_PAGE_STATUS_BYTES, FLIP_PAGE_STATUS_READBACK_BYTES } from "./flip-layout.js";
import type {
    DiffuseParticleCounts,
    DiffusePool,
    EmitterConfig,
    FluidEmitter,
    FluidFlowCarry,
    FluidFlowConfig,
    FluidPolygonSurface,
    FluidProfiler,
    FluidPressureDiagnostics,
    FluidSim,
    FluidSimBaseOptions,
    FoamConfig,
    ForceFieldSpec,
    SceneSdfSpec,
} from "../core/sim-common.js";
import {
    FLUID_FLOW_RUNTIME_WGSL,
    FLUID_FLOW_SHAPE_WGSL,
    FLUID_FLOW_STRUCT_WGSL,
    FLUID_LIFECYCLE_RUNTIME_WGSL,
    FLUID_LIFECYCLE_STRUCT_WGSL,
    FOAM_ACTIVE_FINISH_WGSL,
    FOAM_ACTIVE_PREPARE_WGSL,
    FOAM_BYTES,
    FOAM_COMMON_WGSL,
    SCENE_NORMAL_WGSL,
    sceneSdfGridBindingWgsl,
    SPAWN_ACCEPT_TRIES,
    createDiffuseCountTracker,
    createGpuBuffersAtomically,
    captureFluidFlowCarries,
    createFluidFlowState,
    createFluidInitialParticles,
    disposeFluidFlowState,
    encodeFluidActiveCountReadback,
    enableFluidActiveCountReadback,
    estimateFluidFlowGpuBytes,
    foamActiveListStride,
    foamActiveListOffset,
    foamActiveStateBytes,
    fluidFlowGpuBytes,
    fluidShapeVolume,
    legacyEmitterConfigToFluidFlow,
    pollFluidActiveCount,
    prepareFluidFlowFrame,
    resetFluidFlowState,
    resetFluidParticleLifecycle,
    restoreFluidFlowCarries,
    setFluidFlowConfig,
    updateFluidFlowEmitter,
} from "../core/sim-common.js";
import { createFluidTimestepScheduler, deferFluidTimestep, getFluidTimestepDiagnostics, resetFluidTimestepScheduler, scheduleFluidTimestep } from "../core/timestep-scheduler.js";
import { FLIP_FACE_APERTURE_WGSL } from "./flip-face-aperture.js";
import { FLIP_DENSITY_CORRECTION_WGSL } from "./flip-density-correction.js";

const WORKGROUP_SIZE = 64;
const MAX_WORKGROUPS = 65535;
const FIXED_POINT = 10000;
export const FLIP_PAGE_SIZE = 8;
export const FLIP_PAGE_CELLS = FLIP_PAGE_SIZE ** 3;
export const FLIP_DEFAULT_PAGE_CAPACITY = 8_000;
const PARAMS_BYTES = 10 * 16;
const MULTIGRID_PARAMS_BYTES = 3 * 16;
const MULTIGRID_PRE_SMOOTH = 2;
const MULTIGRID_POST_SMOOTH = 4;
const MULTIGRID_COARSE_SMOOTH = 16;
const MULTIGRID_RELAXATION = 0.8;
const MULTIGRID_CORRECTION_DAMPING = 0.5;
const MAX_MULTIGRID_LEVELS = 8;
const EXTRAPOLATION_LAYERS = 2;
const LIQUID_SDF_LAYERS = 3;
const PRESSURE_DIAGNOSTIC_BYTES = 6 * 4;
const RESEED_HEADER_WORDS = 8;
const RESEED_HEADER_BYTES = RESEED_HEADER_WORDS * 4;
const RESEED_MIN_WORK_BUDGET = 256;
const RESEED_WORK_BUDGET_DIVISOR = 1000;
const SHEETING_HEADER_WORDS = 4;
const FLIP_PIPELINE_CACHE_LIMIT = 256;

let flipComputePipelineCaches: WeakMap<GPUDevice, Map<string, GPUComputePipeline>> | null = null;

function cachedFlipComputePipeline(device: GPUDevice, label: string, code: string): GPUComputePipeline {
    flipComputePipelineCaches ??= new WeakMap();
    let cache = flipComputePipelineCaches.get(device);
    if (!cache) {
        cache = new Map();
        flipComputePipelineCaches.set(device, cache);
    }
    let pipeline = cache.get(code);
    if (pipeline) {
        cache.delete(code);
        cache.set(code, pipeline);
        return pipeline;
    }
    pipeline = device.createComputePipeline({
        label,
        layout: "auto",
        compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
    });
    cache.set(code, pipeline);
    if (cache.size > FLIP_PIPELINE_CACHE_LIMIT) {
        cache.delete(cache.keys().next().value!);
    }
    return pipeline;
}
const SHEETING_HEADER_BYTES = SHEETING_HEADER_WORDS * 4;
const SHEETING_MIN_WORK_BUDGET = 64;
const SHEETING_WORK_BUDGET_DIVISOR = 1000;
const DEFAULT_SURFACE_TRIANGLE_CAPACITY = 1_000_000;
const SURFACE_VERTEX_STRIDE = 32;
const SURFACE_INDIRECT_BYTES = 5 * 4;
const SURFACE_COUNT_READBACK_BYTES = 2 * 4;
const PAGE_DISPATCH_CELLS_OFFSET = 0;
const PAGE_DISPATCH_FACES_OFFSET = 12;
const PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET = 24;
const PAGE_DISPATCH_ALL_PARTICLES_OFFSET = 36;
const PAGE_DISPATCH_SINGLE_OFFSET = 48;
const PAGE_DISPATCH_FOAM_OFFSET = 60;
const PAGE_DISPATCH_POLYGON_CELLS_OFFSET = 72;
const PAGE_DISPATCH_POLYGON_CUBES_OFFSET = 84;
const PAGE_DISPATCH_BYTES = FLIP_PAGE_DISPATCH_BYTES;
const MAX_PAGED_SPECULATIVE_FRAMES = 8;

interface FlipSimTransferMetadata {
    readonly liveCount: number;
    readonly initialTargetCount: number;
    readonly warmupStep: number;
    readonly activePrefixValid: boolean;
    readonly initialEmitterParticleCounts: ReadonlyMap<string, number>;
    readonly reseedSubstep: number;
    readonly sheetingSubstep: number;
    readonly flowActiveCount: number;
    readonly flowFrameSeed: number;
    readonly flowEmitterCursorId: string | undefined;
    readonly flowEmitterCarries: readonly FluidFlowCarry[];
    readonly flowElapsedSeconds: number;
    readonly flowSinkCarries: readonly FluidFlowCarry[];
    readonly foamActiveSide: number;
    readonly foamCounts: DiffuseParticleCounts | undefined;
    readonly foamSeed: number;
}

interface FlipDiffuseTransferState {
    readonly capacity: number;
    readonly active: boolean;
    readonly buffer: GPUBuffer;
    readonly denseHead: GPUBuffer;
    readonly activeState: GPUBuffer | null;
    readonly activeDispatch: GPUBuffer | null;
    readonly drawIndirect: GPUBuffer | null;
}

interface FlipPolygonTransferState {
    readonly stabilizedSdf: GPUBuffer;
    readonly historyValid: boolean;
}

interface PagedStepMetadataSnapshot {
    readonly liveCount: number;
    readonly activePrefixValid: boolean;
    readonly flowOccupancyValid: boolean;
    readonly reseedSubstep: number;
    readonly sheetingSubstep: number;
    readonly pressureCurrentIsA: boolean;
    readonly flowActiveCount: number;
    readonly flowFrameSeed: number;
    readonly flowEmitterCursor: number;
    readonly flowEmitterCarries: Float64Array;
    readonly flowElapsedSeconds: number;
    readonly flowSinkCarries: Float64Array;
    readonly flowData: Uint8Array;
    readonly flowCounterData: Uint32Array;
    readonly foamActiveSide: number;
    readonly foamSeed: number;
    readonly polygonSurfaceRefreshPending: boolean;
    readonly polygonHistoryValid: boolean | undefined;
}

interface FlipSimTransferEndpoint {
    readonly layoutKey: string;
    readonly capacity: number;
    readonly positionBuffer: GPUBuffer;
    readonly velocityBuffer: GPUBuffer;
    readonly debugBuffer: GPUBuffer;
    readonly lifecycleBuffer: GPUBuffer;
    diffuse(): FlipDiffuseTransferState | null;
    polygon(): FlipPolygonTransferState | null;
    capture(): FlipSimTransferMetadata;
    restore(metadata: FlipSimTransferMetadata, diffuseTransferred: boolean, polygonHistoryTransferred: boolean): void;
}

let flipSimTransferEndpoints: WeakMap<FluidSim, FlipSimTransferEndpoint> | undefined;

function getFlipSimTransferEndpoints(): WeakMap<FluidSim, FlipSimTransferEndpoint> {
    return (flipSimTransferEndpoints ??= new WeakMap());
}

/** Copies live marker/lifecycle state between equivalent FLIP allocations. Grid fields are
 * intentionally rebuilt by the destination backend on its next step. */
/** @internal */
export function encodeFlipSimStateTransfer(encoder: GPUCommandEncoder, source: FluidSim, target: FluidSim): (() => void) | null {
    const from = flipSimTransferEndpoints?.get(source);
    const to = flipSimTransferEndpoints?.get(target);
    if (!from || !to || from.layoutKey !== to.layoutKey) {
        return null;
    }
    const metadata = from.capture();
    if (metadata.liveCount > to.capacity || metadata.initialTargetCount > to.capacity || metadata.flowActiveCount > to.capacity) {
        return null;
    }
    const copyShared = (sourceBuffer: GPUBuffer, targetBuffer: GPUBuffer): void => {
        const bytes = Math.min(sourceBuffer.size, targetBuffer.size);
        if (bytes > 0) {
            encoder.copyBufferToBuffer(sourceBuffer, 0, targetBuffer, 0, bytes);
        }
    };
    copyShared(from.positionBuffer, to.positionBuffer);
    copyShared(from.velocityBuffer, to.velocityBuffer);
    copyShared(from.debugBuffer, to.debugBuffer);
    encoder.copyBufferToBuffer(from.lifecycleBuffer, 0, to.lifecycleBuffer, 0, 4);
    encoder.copyBufferToBuffer(from.lifecycleBuffer, 8, to.lifecycleBuffer, 8, 8);
    const sharedLifecycleSlots = Math.min(from.capacity, to.capacity);
    encoder.copyBufferToBuffer(from.lifecycleBuffer, 16, to.lifecycleBuffer, 16, sharedLifecycleSlots * 4);
    if (to.capacity > sharedLifecycleSlots) {
        encoder.clearBuffer(to.lifecycleBuffer, 16 + sharedLifecycleSlots * 4, (to.capacity - sharedLifecycleSlots) * 4);
    }

    const sourceDiffuse = from.diffuse();
    const targetDiffuse = to.diffuse();
    let diffuseTransferred = false;
    if (sourceDiffuse && targetDiffuse && sourceDiffuse.active === targetDiffuse.active && targetDiffuse.capacity >= sourceDiffuse.capacity) {
        encoder.clearBuffer(targetDiffuse.buffer);
        encoder.copyBufferToBuffer(sourceDiffuse.buffer, 0, targetDiffuse.buffer, 0, sourceDiffuse.buffer.size);
        if (sourceDiffuse.active) {
            encoder.clearBuffer(targetDiffuse.activeState!);
            encoder.copyBufferToBuffer(sourceDiffuse.activeState!, 0, targetDiffuse.activeState!, 0, 16);
            for (let side = 0; side < 2; side++) {
                encoder.copyBufferToBuffer(
                    sourceDiffuse.activeState!,
                    foamActiveListOffset(sourceDiffuse.capacity, side),
                    targetDiffuse.activeState!,
                    foamActiveListOffset(targetDiffuse.capacity, side),
                    sourceDiffuse.capacity * 4
                );
            }
            const sourceFlagsOffset = foamActiveListOffset(sourceDiffuse.capacity, 0) + 2 * foamActiveListStride(sourceDiffuse.capacity) * 4;
            const targetFlagsOffset = foamActiveListOffset(targetDiffuse.capacity, 0) + 2 * foamActiveListStride(targetDiffuse.capacity) * 4;
            encoder.copyBufferToBuffer(sourceDiffuse.activeState!, sourceFlagsOffset, targetDiffuse.activeState!, targetFlagsOffset, sourceDiffuse.capacity * 4);
            copyShared(sourceDiffuse.activeDispatch!, targetDiffuse.activeDispatch!);
            copyShared(sourceDiffuse.drawIndirect!, targetDiffuse.drawIndirect!);
        } else {
            copyShared(sourceDiffuse.denseHead, targetDiffuse.denseHead);
        }
        diffuseTransferred = true;
    }

    const sourcePolygon = from.polygon();
    const targetPolygon = to.polygon();
    const polygonHistoryTransferred = sourcePolygon?.historyValid === true && targetPolygon !== null && sourcePolygon.stabilizedSdf.size === targetPolygon.stabilizedSdf.size;
    if (polygonHistoryTransferred) {
        encoder.copyBufferToBuffer(sourcePolygon!.stabilizedSdf, 0, targetPolygon!.stabilizedSdf, 0, sourcePolygon!.stabilizedSdf.size);
    }
    return () => to.restore(metadata, diffuseTransferred, polygonHistoryTransferred);
}

/** Copies and immediately applies compatible FLIP transfer metadata. Callers that own
 * submission separately should use {@link encodeFlipSimStateTransfer}. */
/** @internal */
export function transferFlipSimState(encoder: GPUCommandEncoder, source: FluidSim, target: FluidSim): boolean {
    const commitMetadata = encodeFlipSimStateTransfer(encoder, source, target);
    commitMetadata?.();
    return commitMetadata !== null;
}

export type FlipPressureSolver = "jacobi" | "multigrid";

/** Optional high-quality FLIP geometry paths. Every feature defaults off so the
 * legacy fast path allocates no extra buffers and records no extra passes. */
export interface FlipQualityFeatures {
    /** Use the sparse 8³-page storage backend. */
    pagedGrid?: boolean;
    /** Active page-pool capacity for the sparse backend. */
    pagedGridMaxPages?: number;
    /** Allocate asynchronous pressure residual/divergence diagnostics. */
    pressureDiagnostics?: boolean;
    /** Build a marker-sphere narrow-band particle level set. */
    liquidSdf?: boolean;
    /** Use liquid-SDF interface fractions in the free-surface pressure solve. Requires liquidSdf. */
    ghostFluid?: boolean;
    /** Sample the scene SDF into fractional open-area weights on MAC faces. */
    fractionalSolids?: boolean;
    /** Include scene-boundary velocity in weighted face fluxes. Requires fractionalSolids. */
    movingSolidBoundaries?: boolean;
    /** Allocate production marker-reseeding work lists. */
    reseedParticles?: boolean;
    /** Allocate free-surface particle-sheeting work lists. */
    particleSheeting?: boolean;
    /** Allocate a GPU polygon-surface vertex/index pool. */
    polygonSurface?: boolean;
    /** Render-only polygon reconstruction samples per solver-cell axis. Default 1. */
    polygonReconstructionMultiplier?: number;
}

function multigridDimensions(gridDim: readonly [number, number, number]): Array<[number, number, number]> {
    const dimensions: Array<[number, number, number]> = [[...gridDim]];
    while (dimensions.length < MAX_MULTIGRID_LEVELS) {
        const current = dimensions[dimensions.length - 1]!;
        if (Math.min(...current) <= 4 || current[0] * current[1] * current[2] <= 64) {
            break;
        }
        dimensions.push(current.map((value) => Math.max(1, Math.ceil(value / 2))) as [number, number, number]);
    }
    return dimensions;
}

function estimateMultigridGpuBytes(gridDim: readonly [number, number, number], fineStorageCells?: number): number {
    return multigridDimensions(gridDim).reduce((bytes, dim, index) => {
        const count = index === 0 && fineStorageCells !== undefined ? fineStorageCells : dim[0] * dim[1] * dim[2];
        return bytes + count * (index === 0 ? 12 : 24) + MULTIGRID_PARAMS_BYTES;
    }, 0);
}

function storageBufferByteLimit(limits: Pick<GPUSupportedLimits, "maxStorageBufferBindingSize" | "maxBufferSize">): number {
    return Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
}

function assertDenseMultigridLevelFits(levelIndex: number, dim: readonly [number, number, number], bytes: number, maxStorageBytes: number): void {
    if (bytes <= maxStorageBytes) {
        return;
    }
    throw new RangeError(
        `[FLIP] Multigrid dense level ${levelIndex} (${dim.join(" x ")}) needs ${(bytes / (1024 * 1024)).toFixed(1)} MiB per coarse-grid buffer; device limit is ${(maxStorageBytes / (1024 * 1024)).toFixed(1)} MiB.`
    );
}

export function flipMacFaceBufferBytes(gridDim: readonly [number, number, number]): number {
    const [nx, ny, nz] = gridDim;
    const totalFaces = (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1);
    return totalFaces * 8;
}

export function pagedFlipStorageCounts(pageCapacity: number): { cells: number; faces: number } {
    const pages = Math.max(1, Math.floor(pageCapacity));
    return {
        cells: pages * FLIP_PAGE_CELLS + 1,
        faces: pages * FLIP_PAGE_CELLS * 3 + 1,
    };
}

export interface FlipPageLayout {
    readonly blockDim: [number, number, number];
    readonly numBlocks: number;
    readonly requestedMaxPages: number;
    readonly maxPages: number;
    readonly storageCells: number;
    readonly storageFaces: number;
    readonly allocatedCells: number;
    readonly allocatedFaces: number;
    readonly lookupWidth: number;
    readonly lookupHeight: number;
    readonly lookupWords: number;
}

export function resolveFlipPageLayout(gridDim: readonly [number, number, number], requestedMaxPages = FLIP_DEFAULT_PAGE_CAPACITY, maxTextureDimension2D = 8192): FlipPageLayout {
    const dim = gridDim.map((value) => Math.max(4, Math.round(value))) as [number, number, number];
    const blockDim = dim.map((value) => Math.ceil(value / FLIP_PAGE_SIZE)) as [number, number, number];
    const numBlocks = blockDim[0] * blockDim[1] * blockDim[2];
    const normalizedRequestedPages = Number.isFinite(requestedMaxPages) ? Math.max(1, Math.floor(requestedMaxPages)) : FLIP_DEFAULT_PAGE_CAPACITY;
    const maxPages = Math.min(numBlocks, normalizedRequestedPages);
    const allocated = pagedFlipStorageCounts(maxPages);
    const lookupWidth = Math.min(8192, Number.isFinite(maxTextureDimension2D) ? Math.max(1, Math.floor(maxTextureDimension2D)) : 8192);
    const lookupWords = 2 + numBlocks + maxPages;
    return {
        blockDim,
        numBlocks,
        requestedMaxPages: normalizedRequestedPages,
        maxPages,
        storageCells: allocated.cells - 1,
        storageFaces: allocated.faces - 1,
        allocatedCells: allocated.cells,
        allocatedFaces: allocated.faces,
        lookupWidth,
        lookupHeight: Math.ceil(lookupWords / lookupWidth),
        lookupWords,
    };
}

export function estimateFlipGpuBytes(
    particleCount: number,
    gridDim: readonly [number, number, number],
    pressureSolver: FlipPressureSolver = "jacobi",
    quality: FlipQualityFeatures = {},
    warmupSeedBuffersAllocated = false,
    maxTextureDimension2D = 8192
): number {
    const count = Math.max(1, Math.floor(particleCount));
    const dim = gridDim.map((value) => Math.max(4, Math.round(value))) as [number, number, number];
    const numCells = dim[0] * dim[1] * dim[2];
    const totalFaces = (dim[0] + 1) * dim[1] * dim[2] + dim[0] * (dim[1] + 1) * dim[2] + dim[0] * dim[1] * (dim[2] + 1);
    const paged = quality.pagedGrid === true;
    const pageLayout = paged ? resolveFlipPageLayout(dim, quality.pagedGridMaxPages ?? FLIP_DEFAULT_PAGE_CAPACITY, maxTextureDimension2D) : null;
    const storedCells = pageLayout?.allocatedCells ?? numCells;
    const storedFaces = pageLayout?.allocatedFaces ?? totalFaces;
    const particleBytes = count * (16 + 16 + 4);
    const warmupSeedBytes = warmupSeedBuffersAllocated ? count * (16 + 16) : 0;
    const faceBytes = storedFaces * (8 + 4 + 8 + 8 + 8 + 8 + 4);
    const cellBytes = storedCells * (4 + 4 + 4 + 4 + 4 + 16 + 4);
    const fixedBytes = PARAMS_BYTES + 32 + 4 + 2 * 4;
    const pressureDiagnosticBytes = quality.pressureDiagnostics ? storedCells * 4 + PRESSURE_DIAGNOSTIC_BYTES * 3 : 0;
    const multigridBytes = pressureSolver === "multigrid" ? estimateMultigridGpuBytes(dim, paged ? storedCells : undefined) : 0;
    const polygonMultiplier = Math.round(Math.min(2, Math.max(1, quality.polygonReconstructionMultiplier ?? 1)) * 4) / 4;
    const liquidSdfBytes = quality.liquidSdf || quality.particleSheeting || quality.polygonSurface ? storedCells * 8 : 0;
    const solidFaceBytes = quality.fractionalSolids ? storedFaces * 8 : 0;
    const reseedBytes = quality.reseedParticles ? RESEED_HEADER_BYTES + count * 8 : 0;
    const sheetingBytes = quality.particleSheeting ? SHEETING_HEADER_BYTES + count * 4 : 0;
    const polygonDim = dim.map((value) => Math.max(4, Math.ceil(value * polygonMultiplier))) as [number, number, number];
    const polygonCellCount = polygonDim[0] * polygonDim[1] * polygonDim[2];
    const cubeCount = Math.max(1, (polygonDim[0] - 1) * (polygonDim[1] - 1) * (polygonDim[2] - 1));
    const surfaceTriangleCapacity = Math.min(DEFAULT_SURFACE_TRIANGLE_CAPACITY, cubeCount * 6);
    const polygonSurfaceBytes = quality.polygonSurface
        ? cubeCount * SURFACE_VERTEX_STRIDE +
          surfaceTriangleCapacity * (12 + 16) +
          SURFACE_INDIRECT_BYTES * 2 +
          SURFACE_COUNT_READBACK_BYTES +
          PARAMS_BYTES +
          polygonCellCount * (polygonMultiplier > 1 || paged ? 8 : 4)
        : 0;
    const pagedGridBytes = pageLayout
        ? pageLayout.lookupWords * 4 +
          pageLayout.lookupWidth * pageLayout.lookupHeight * 4 +
          FLIP_PAGE_DISPATCH_BYTES +
          FLIP_PAGE_DISPATCH_CONFIG_BYTES +
          FLIP_PAGE_STATUS_BYTES +
          2 * FLIP_PAGE_STATUS_READBACK_BYTES
        : 0;
    return (
        particleBytes +
        warmupSeedBytes +
        faceBytes +
        cellBytes +
        fixedBytes +
        estimateFluidFlowGpuBytes(count) +
        multigridBytes +
        pressureDiagnosticBytes +
        liquidSdfBytes +
        solidFaceBytes +
        reseedBytes +
        sheetingBytes +
        polygonSurfaceBytes +
        pagedGridBytes
    );
}

export const FLIP_DEFAULT_CELL_SIZE = 0.25;
export const FLIP_DEFAULT_PARTICLE_RADIUS = 0.09;
const FLIP_PARTICLE_RADIUS_TO_CELL_SIZE = FLIP_DEFAULT_PARTICLE_RADIUS / FLIP_DEFAULT_CELL_SIZE;

export interface FlipDiscretizationOptions {
    /** Simulation-domain minimum corner. Default [-20, 0, -20]. */
    boundsMin?: readonly [number, number, number];
    /** Simulation-domain maximum corner. Default [20, 20, 20]. */
    boundsMax?: readonly [number, number, number];
    /** Pressure-cell divisions along the longest domain axis. */
    gridResolution?: number;
    /** Legacy explicit MAC-grid cell width. Mutually exclusive with gridResolution. */
    dx?: number;
    /** Legacy exact pressure-cell count. Mutually exclusive with gridResolution. */
    gridDim?: readonly [number, number, number];
    /** Marker sampling density represented by one full MAC cell. Default 8. */
    markersPerCell?: number;
    /** Explicit marker radius override. Omit to derive it from the cell width. */
    particleRadius?: number;
}

export interface FlipDiscretization {
    readonly dx: number;
    readonly gridDim: [number, number, number];
    readonly gridResolution: number;
    readonly markersPerCell: number;
    readonly markerVolume: number;
    readonly particleRadius: number;
}

function cellsForExtent(extent: number, dx: number): number {
    const exact = extent / dx;
    const nearest = Math.round(exact);
    const tolerance = Number.EPSILON * 16 * Math.max(1, Math.abs(exact));
    return Math.max(4, Math.abs(exact - nearest) <= tolerance ? nearest : Math.ceil(exact));
}

/** Resolve every FLIP discretization value from one authoritative domain/resolution contract. */
export function resolveFlipDiscretization(options: FlipDiscretizationOptions = {}): FlipDiscretization {
    const boundsMin: [number, number, number] = options.boundsMin ? [...options.boundsMin] : [-20, 0, -20];
    const boundsMax: [number, number, number] = options.boundsMax ? [...options.boundsMax] : [20, 20, 20];
    const extents = boundsMax.map((value, axis) => value - boundsMin[axis]!) as [number, number, number];
    if (extents.some((extent) => !(extent > 0) || !Number.isFinite(extent))) {
        throw new RangeError("[FLIP] bounds must contain finite positive extents.");
    }
    if (options.gridResolution !== undefined && (options.dx !== undefined || options.gridDim !== undefined)) {
        throw new RangeError("[FLIP] gridResolution is mutually exclusive with dx and gridDim.");
    }
    let gridResolution: number;
    let dx: number;
    if (options.gridResolution !== undefined) {
        if (!(options.gridResolution > 0) || !Number.isFinite(options.gridResolution)) {
            throw new RangeError("[FLIP] gridResolution must be a positive finite number.");
        }
        gridResolution = Math.max(1, Math.round(options.gridResolution));
        dx = Math.max(...extents) / gridResolution;
    } else {
        dx = options.dx ?? FLIP_DEFAULT_CELL_SIZE;
        if (!(dx > 0) || !Number.isFinite(dx)) {
            throw new RangeError("[FLIP] dx must be a positive finite number.");
        }
        gridResolution = cellsForExtent(Math.max(...extents), dx);
    }
    const requestedMarkersPerCell = options.markersPerCell ?? 8;
    if (!(requestedMarkersPerCell > 0) || !Number.isFinite(requestedMarkersPerCell)) {
        throw new RangeError("[FLIP] markersPerCell must be a positive finite number.");
    }
    const markersPerCell = Math.max(1, Math.round(requestedMarkersPerCell));
    const particleRadius = options.particleRadius ?? dx * FLIP_PARTICLE_RADIUS_TO_CELL_SIZE;
    if (!(particleRadius > 0) || !Number.isFinite(particleRadius)) {
        throw new RangeError("[FLIP] particleRadius must be a positive finite number.");
    }
    if (options.gridDim?.some((value) => !(value > 0) || !Number.isFinite(value))) {
        throw new RangeError("[FLIP] gridDim must contain positive finite values.");
    }
    const gridDim: [number, number, number] = options.gridDim
        ? (options.gridDim.map((value) => Math.max(4, Math.round(value))) as [number, number, number])
        : (extents.map((extent) => cellsForExtent(extent, dx)) as [number, number, number]);
    return {
        dx,
        gridDim,
        gridResolution,
        markersPerCell,
        markerVolume: dx ** 3 / markersPerCell,
        particleRadius,
    };
}

/** @internal */
export interface FlipOptions extends FluidSimBaseOptions {
    /** Simulation-domain minimum corner. Default [-20, 0, -20]. */
    boundsMin?: [number, number, number];
    /** Simulation-domain maximum corner. Default [20, 20, 20]. */
    boundsMax?: [number, number, number];
    /** Exact pressure-cell count along X/Y/Z. Derived from bounds when omitted. */
    gridDim?: [number, number, number];
    /** Store cells and staggered faces in an active 8³-page pool. Default false. */
    pagedGrid?: boolean;
    /** Maximum number of active grid pages when pagedGrid is enabled. Default 8,000. */
    pagedGridMaxPages?: number;
    /** Called asynchronously when active fluid pages exceed the configured pool. */
    onPagedGridOverflow?: (requiredPages: number, capacity: number) => void;
    /** Called asynchronously with the latest active-page demand. */
    onPagedGridPages?: (requiredPages: number, capacity: number) => void;
    /** Safety floor height. Default boundsMin.y. */
    groundY?: number;
    /** Pressure-cell divisions along the longest domain axis. Preferred over supplying dx. */
    gridResolution?: number;
    /** Legacy explicit MAC-grid cell width in world units. Default 0.25. */
    dx?: number;
    /** Marker sampling density represented by one full MAC cell. Default 8. */
    markersPerCell?: number;
    /** Minimum substeps per frame. Default 1. */
    minSubsteps?: number;
    /** Maximum adaptive substeps per frame. Default 8. */
    maxSubsteps?: number;
    /** Maximum cells travelled per substep. 0 disables adaptive CFL. Default 2. */
    cflNumber?: number;
    /** Maximum duration represented by one substep. Default 1/120. */
    maxSubDt?: number;
    /** Weighted-Jacobi pressure iterations per substep. Default 40. */
    pressureIterations?: number;
    /** Weighted-Jacobi relaxation factor. Default 0.8. */
    pressureRelaxation?: number;
    /** Pressure projection solver. Default "jacobi". */
    pressureSolver?: FlipPressureSolver;
    /** Geometric multigrid V-cycles per substep. Default 2. */
    multigridCycles?: number;
    /** Relative pressure residual target. 0 retains fixed-cycle behavior. Default 0. */
    pressureTolerance?: number;
    /** Asynchronously sample pressure residual and post-projection divergence. Default false. */
    pressureDiagnostics?: boolean;
    /** FLIP share in the FLIP/PIC blend. Default 0.95. */
    flipRatio?: number;
    /** Exponential non-physical particle-velocity damping per second. Default 0. */
    velocityDamping?: number;
    /** Kinematic viscosity in world-units squared per second. Default 0. */
    kinematicViscosity?: number;
    /** Jacobi iterations used by the implicit viscosity solve. Default 12. */
    viscosityIterations?: number;
    /** Liquid-air surface tension coefficient. Default 0. */
    surfaceTension?: number;
    /** Build the optional liquid signed-distance field. Default false. */
    liquidSdf?: boolean;
    /** Use ghost-fluid free-surface pressure fractions. Requires liquidSdf. Default false. */
    ghostFluid?: boolean;
    /** Use fractional scene-solid MAC face weights. Default false. */
    fractionalSolids?: boolean;
    /** Include moving-solid face velocity in fractional fluxes. Requires fractionalSolids. Default false. */
    movingSolidBoundaries?: boolean;
    /** Enable min/target/max marker reseeding. Default false. */
    reseedParticles?: boolean;
    /** Under-populated cell threshold. Default max(1, markersPerCell / 2). */
    reseedMinParticles?: number;
    /** Marker count restored in under-populated cells. Default markersPerCell. */
    reseedTargetParticles?: number;
    /** Markers above this count are recycled. Default ceil(markersPerCell * 1.5). */
    reseedMaxParticles?: number;
    /** Number of substeps between reseeding passes. Default 5. */
    reseedInterval?: number;
    /** Add particles from unused capacity to under-sampled thin free-surface sheets. Default false. */
    particleSheeting?: boolean;
    /** Fraction of the marker density restored in detected sheets. Default 0.5. */
    sheetingStrength?: number;
    /** Number of substeps between particle-sheeting passes. Default 5. */
    sheetingInterval?: number;
    /** Build an indexed polygon surface from the liquid SDF entirely on the GPU. Default false. */
    polygonSurface?: boolean;
    /** Render-only polygon reconstruction samples per solver-cell axis. Default 1. */
    polygonReconstructionMultiplier?: number;
    /** Maximum number of generated surface triangles. Default 1,000,000. */
    surfaceMaxTriangles?: number;
    /** Particle collision restitution. Default 0. */
    restitution?: number;
    /** Explicit world-space xyz seed positions. */
    initialPositions?: Float32Array;
    /** Explicit world-space xyz seed velocities aligned with initialPositions. */
    initialVelocities?: Float32Array;
}

interface MultigridLevel {
    readonly dim: [number, number, number];
    readonly count: number;
    readonly groups: number;
    readonly cellTypes: GPUBuffer;
    readonly rhs: GPUBuffer;
    readonly residual: GPUBuffer;
    readonly fluidFraction: GPUBuffer;
    readonly pressureA: GPUBuffer;
    readonly pressureB: GPUBuffer;
    readonly params: GPUBuffer;
    readonly smoothAB: GPUBindGroup;
    readonly smoothBA: GPUBindGroup;
    readonly residualA: GPUBindGroup;
    readonly residualB: GPUBindGroup;
    restrict?: GPUBindGroup;
    prolongAA?: GPUBindGroup;
    prolongAB?: GPUBindGroup;
    prolongBA?: GPUBindGroup;
    prolongBB?: GPUBindGroup;
}

interface MultigridResources {
    readonly levels: MultigridLevel[];
    readonly buildRhs: GPUBindGroup;
    fineResidualA?: GPUBindGroup;
    fineResidualB?: GPUBindGroup;
    readonly gpuBytes: number;
}

interface LiquidSdfResources {
    readonly sdfA: GPUBuffer;
    readonly sdfB: GPUBuffer;
    readonly clearPipeline: GPUComputePipeline;
    readonly scatterPipeline: GPUComputePipeline;
    readonly finalizePipeline: GPUComputePipeline;
    readonly relaxPipeline: GPUComputePipeline;
    readonly clear: GPUBindGroup;
    readonly scatter: GPUBindGroup;
    readonly finalize: GPUBindGroup;
    readonly relaxBA: GPUBindGroup;
    readonly relaxAB: GPUBindGroup;
    readonly gpuBytes: number;
}

interface SolidFaceResources {
    readonly geometry: GPUBuffer;
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    readonly gpuBytes: number;
}

interface PressureDiagnosticResources {
    readonly residual: GPUBuffer;
    readonly output: GPUBuffer;
    readonly readbacks: GPUBuffer[];
    readonly readbackStates: Array<"idle" | "copied" | "mapping">;
    readonly readbackGenerations: number[];
    readonly readbackSampleIds: number[];
    readonly readbackIterations: number[];
    readbackNext: number;
    readbackError: { readonly generation: number; readonly sampleId: number; readonly error: unknown } | null;
    disposed: boolean;
    residualPipeline: GPUComputePipeline;
    readonly reducePipeline: GPUComputePipeline;
    readonly postDivergencePipeline: GPUComputePipeline;
    residualA: GPUBindGroup;
    residualB: GPUBindGroup;
    readonly reduce: GPUBindGroup;
    readonly postDivergence: GPUBindGroup;
    readonly gpuBytes: number;
}

interface ReseedResources {
    readonly state: GPUBuffer;
    readonly deleteOverfullPipeline: GPUComputePipeline;
    readonly deleteSurplusPipeline: GPUComputePipeline;
    readonly buildPipeline: GPUComputePipeline;
    emitPipeline: GPUComputePipeline;
    readonly deleteOverfullBindGroup: GPUBindGroup;
    readonly deleteSurplusBindGroup: GPUBindGroup;
    readonly buildBindGroup: GPUBindGroup;
    emitBindGroup: GPUBindGroup;
    readonly gpuBytes: number;
}

interface SheetingResources {
    readonly state: GPUBuffer;
    readonly buildPipeline: GPUComputePipeline;
    emitPipeline: GPUComputePipeline;
    readonly buildBindGroup: GPUBindGroup;
    emitBindGroup: GPUBindGroup;
    readonly gpuBytes: number;
}

interface PolygonSurfaceResources {
    readonly surface: FluidPolygonSurface;
    readonly dimensions: [number, number, number];
    readonly cellCount: number;
    readonly spacing: number;
    readonly paramsBuffer: GPUBuffer;
    readonly reconstructedSdf: GPUBuffer | null;
    readonly stabilizedSdf: GPUBuffer;
    readonly sdfUpsamplePipeline: GPUComputePipeline | null;
    readonly sdfUpsampleBindGroup: GPUBindGroup | null;
    readonly copyPipeline: GPUComputePipeline;
    readonly stabilizePipeline: GPUComputePipeline;
    readonly resetPipeline: GPUComputePipeline;
    readonly vertexPipeline: GPUComputePipeline;
    readonly indexPipeline: GPUComputePipeline;
    readonly finalizePipeline: GPUComputePipeline;
    readonly copyBindGroup: GPUBindGroup;
    readonly stabilizeBindGroup: GPUBindGroup;
    readonly resetBindGroup: GPUBindGroup;
    readonly vertexBindGroup: GPUBindGroup;
    readonly indexBindGroup: GPUBindGroup;
    readonly finalizeBindGroup: GPUBindGroup;
    readonly cubeCount: number;
    readonly gpuBytes: number;
    readonly triangleCountReadback: {
        readonly buffers: GPUBuffer[];
        readonly states: Array<"idle" | "copied" | "mapping">;
        readonly generations: number[];
        readonly copyEncoders: Array<GPUCommandEncoder | null>;
        generation: number;
        frame: number;
        next: number;
        latest: number | undefined;
        error: unknown;
    };
    historyValid: boolean;
}

const COMMON_WGSL = /* wgsl */ `
const FIXED_POINT: f32 = ${FIXED_POINT}.0;
const FIXED_POINT_INV: f32 = ${1 / FIXED_POINT};
const FACE_U: u32 = 0u;
const FACE_V: u32 = 1u;
const FACE_W: u32 = 2u;
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

struct Params {
    originDx: vec4<f32>,
    dimGround: vec4<f32>,
    boundsMin: vec4<f32>,
    boundsMax: vec4<f32>,
    sim: vec4<f32>,
    solve: vec4<f32>,
    material: vec4<f32>,
    counts: vec4<u32>,
    reseed: vec4<u32>,
    sheeting: vec4<f32>,
};

struct FaceAccum {
    momentum: atomic<i32>,
    weight: atomic<i32>,
};

fn gridDim(p: Params) -> vec3<i32> {
    return vec3<i32>(p.dimGround.xyz);
}

fn cellIndex(c: vec3<i32>, p: Params) -> u32 {
    let d = gridDim(p);
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn inCellGrid(c: vec3<i32>, p: Params) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < gridDim(p));
}

fn storageCellExists(c: vec3<i32>, p: Params) -> bool {
    return inCellGrid(c, p);
}

fn uDim(p: Params) -> vec3<i32> {
    let d = gridDim(p);
    return vec3<i32>(d.x + 1, d.y, d.z);
}

fn vDim(p: Params) -> vec3<i32> {
    let d = gridDim(p);
    return vec3<i32>(d.x, d.y + 1, d.z);
}

fn wDim(p: Params) -> vec3<i32> {
    let d = gridDim(p);
    return vec3<i32>(d.x, d.y, d.z + 1);
}

fn faceCount(d: vec3<i32>) -> u32 {
    return u32(d.x * d.y * d.z);
}

fn uCount(p: Params) -> u32 {
    return faceCount(uDim(p));
}

fn vCount(p: Params) -> u32 {
    return faceCount(vDim(p));
}

fn totalFaceCount(p: Params) -> u32 {
    return uCount(p) + vCount(p) + faceCount(wDim(p));
}

fn localFaceIndex(c: vec3<i32>, d: vec3<i32>) -> u32 {
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn globalFaceIndex(kind: u32, c: vec3<i32>, p: Params) -> u32 {
    if (kind == FACE_U) {
        return localFaceIndex(c, uDim(p));
    }
    if (kind == FACE_V) {
        return uCount(p) + localFaceIndex(c, vDim(p));
    }
    return uCount(p) + vCount(p) + localFaceIndex(c, wDim(p));
}

fn faceCoord(localIndex: u32, d: vec3<i32>) -> vec3<i32> {
    let x = i32(localIndex % u32(d.x));
    let yz = i32(localIndex / u32(d.x));
    let y = yz % d.y;
    return vec3<i32>(x, y, yz / d.y);
}

fn faceKind(globalIndex: u32, p: Params) -> u32 {
    if (globalIndex < uCount(p)) {
        return FACE_U;
    }
    if (globalIndex < uCount(p) + vCount(p)) {
        return FACE_V;
    }
    return FACE_W;
}

fn faceLocalIndex(globalIndex: u32, kind: u32, p: Params) -> u32 {
    if (kind == FACE_U) {
        return globalIndex;
    }
    if (kind == FACE_V) {
        return globalIndex - uCount(p);
    }
    return globalIndex - uCount(p) - vCount(p);
}

fn faceGridDim(kind: u32, p: Params) -> vec3<i32> {
    if (kind == FACE_U) {
        return uDim(p);
    }
    if (kind == FACE_V) {
        return vDim(p);
    }
    return wDim(p);
}

fn faceOffset(kind: u32) -> vec3<f32> {
    if (kind == FACE_U) {
        return vec3<f32>(0.0, 0.5, 0.5);
    }
    if (kind == FACE_V) {
        return vec3<f32>(0.5, 0.0, 0.5);
    }
    return vec3<f32>(0.5, 0.5, 0.0);
}

fn adjacentCells(kind: u32, c: vec3<i32>) -> array<vec3<i32>, 2> {
    if (kind == FACE_U) {
        return array<vec3<i32>, 2>(c - vec3<i32>(1, 0, 0), c);
    }
    if (kind == FACE_V) {
        return array<vec3<i32>, 2>(c - vec3<i32>(0, 1, 0), c);
    }
    return array<vec3<i32>, 2>(c - vec3<i32>(0, 0, 1), c);
}

fn encodeFixed(value: f32) -> i32 {
    return i32(clamp(value, -1000.0, 1000.0) * FIXED_POINT);
}

fn decodeFixed(value: i32) -> f32 {
    return f32(value) * FIXED_POINT_INV;
}

fn safeNormal(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    return select(fallback, v / len, len > 1.0e-6);
}

fn clampLength(v: vec3<f32>, maxLength: f32) -> vec3<f32> {
    let len = length(v);
    return select(v, v * (maxLength / len), len > maxLength && len > 1.0e-6);
}
`;

const P2G_WGSL = /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> faces: array<FaceAccum>;
@group(0) @binding(3) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p: Params;
@group(0) @binding(5) var<storage, read_write> lifecycle: FluidLifecycle;

fn scatterFace(kind: u32, world: vec3<f32>, component: f32) {
    let d = faceGridDim(kind, p);
    let grid = (world - p.originDx.xyz) / p.originDx.w - faceOffset(kind);
    let base = vec3<i32>(floor(grid));
    let f = grid - floor(grid);
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (any(c < vec3<i32>(0)) || any(c >= d)) {
                    continue;
                }
                let q = vec3<f32>(f32(x), f32(y), f32(z));
                let w3 = select(vec3<f32>(1.0) - f, f, q > vec3<f32>(0.5));
                let weight = w3.x * w3.y * w3.z;
                let index = globalFaceIndex(kind, c, p);
                atomicAdd(&faces[index].momentum, encodeFixed(weight * component));
                atomicAdd(&faces[index].weight, encodeFixed(weight));
            }
        }
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsActive(i)) {
        return;
    }
    let world = positions[i].xyz;
    let velocity = clampLength(velocities[i].xyz, p.solve.w);
    scatterFace(FACE_U, world, velocity.x);
    scatterFace(FACE_V, world, velocity.y);
    scatterFace(FACE_W, world, velocity.z);
    let cell = clamp(vec3<i32>(floor((world - p.originDx.xyz) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
    atomicAdd(&cellMarks[cellIndex(cell, p)], 1u);
}`;

function buildClassifyWgsl(scene: SceneSdfSpec | null, fractionalSolids = false): string {
    const sceneDecl = scene
        ? `${scene.struct}
@group(0) @binding(3) var<uniform> sceneSdfParams: SceneSdfParams;
${sceneSdfGridBindingWgsl(scene, 4)}
${scene.sdf}`
        : "fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return 1.0e30; }";
    return /* wgsl */ `
${COMMON_WGSL}
${sceneDecl}
@group(0) @binding(0) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cellTypes: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&cellTypes)) {
        return;
    }
    let d = gridDim(p);
    let c = faceCoord(i, d);
    let center = p.originDx.xyz + (vec3<f32>(c) + 0.5) * p.originDx.w;
    // Fractional mode keeps every cut cell active; the half-diagonal threshold
    // guarantees that a solid cell is fully behind a proper signed-distance surface.
    if (sceneSdf(center, 0.0) <= ${fractionalSolids ? "-0.8660254" : "-0.5"} * p.originDx.w) {
        cellTypes[i] = CELL_SOLID;
    } else {
        cellTypes[i] = select(CELL_AIR, CELL_FLUID, atomicLoad(&cellMarks[i]) > 0u);
    }
}`;
}

const LIQUID_SDF_CLEAR_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> orderedSdf: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> p: Params;

fn floatToOrdered(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    return select(~bits, bits ^ 0x80000000u, (bits & 0x80000000u) == 0u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i < arrayLength(&orderedSdf)) {
        atomicStore(&orderedSdf[i], floatToOrdered(f32(${LIQUID_SDF_LAYERS}) * p.originDx.w));
    }
}
`;

const LIQUID_SDF_SCATTER_WGSL = /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> orderedSdf: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read_write> lifecycle: FluidLifecycle;

fn floatToOrdered(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    return select(~bits, bits ^ 0x80000000u, (bits & 0x80000000u) == 0u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsActive(i)) {
        return;
    }
    let position = positions[i].xyz;
    let particleCell = vec3<i32>(floor((position - p.originDx.xyz) / p.originDx.w));
    let reconstructionRadius = max(p.solve.z, 0.75 * p.originDx.w);
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let c = particleCell + vec3<i32>(x, y, z);
                if (!storageCellExists(c, p)) {
                    continue;
                }
                let center = p.originDx.xyz + (vec3<f32>(c) + 0.5) * p.originDx.w;
                let distance = length(center - position) - reconstructionRadius;
                atomicMin(&orderedSdf[cellIndex(c, p)], floatToOrdered(distance));
            }
        }
    }
}
`;

const LIQUID_SDF_FINALIZE_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> orderedSdf: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> liquidSdf: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn orderedToFloat(value: u32) -> f32 {
    let bits = select(~value, value ^ 0x80000000u, (value & 0x80000000u) != 0u);
    return bitcast<f32>(bits);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&liquidSdf)) {
        return;
    }
    let band = f32(${LIQUID_SDF_LAYERS}) * p.originDx.w;
    let nearest = clamp(orderedToFloat(atomicLoad(&orderedSdf[i])), -band, band);
    let kind = cellTypes[i];
    if (kind == CELL_SOLID) {
        liquidSdf[i] = band;
    } else if (kind == CELL_FLUID) {
        liquidSdf[i] = min(nearest, -0.05 * p.originDx.w);
    } else {
        liquidSdf[i] = max(nearest, 0.05 * p.originDx.w);
    }
}`;

const LIQUID_SDF_RELAX_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sdfIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> sdfOut: array<f32>;
@group(0) @binding(2) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(3) var<uniform> p: Params;

fn distanceAt(c: vec3<i32>, fallback: f32) -> f32 {
    if (!inCellGrid(c, p) || cellTypes[cellIndex(c, p)] == CELL_SOLID) {
        return fallback;
    }
    return abs(sdfIn[cellIndex(c, p)]);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&sdfOut)) {
        return;
    }
    let kind = cellTypes[i];
    if (kind == CELL_SOLID) {
        sdfOut[i] = f32(${LIQUID_SDF_LAYERS}) * p.originDx.w;
        return;
    }
    let c = faceCoord(i, gridDim(p));
    var distance = abs(sdfIn[i]);
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                if (x == 0 && y == 0 && z == 0) {
                    continue;
                }
                let offset = vec3<i32>(x, y, z);
                let stepDistance = length(vec3<f32>(offset)) * p.originDx.w;
                distance = min(distance, distanceAt(c + offset, distance) + stepDistance);
            }
        }
    }
    sdfOut[i] = select(distance, -distance, kind == CELL_FLUID);
}`;

function buildPolygonSdfUpsampleWgsl(layout: FlipPageLayout | null): string {
    const pagedAddressing = layout
        ? /* wgsl */ `
@group(1) @binding(0) var polygonPageLookup: texture_2d<u32>;
const POLYGON_PAGE_SIZE: u32 = ${FLIP_PAGE_SIZE}u;
const POLYGON_PAGE_CELLS: u32 = ${FLIP_PAGE_CELLS}u;
const POLYGON_PAGE_BLOCK_DIM: vec3<u32> = vec3<u32>(${layout.blockDim[0]}u, ${layout.blockDim[1]}u, ${layout.blockDim[2]}u);
const POLYGON_PAGE_MAP_OFFSET: u32 = 2u;
const POLYGON_PAGE_LOOKUP_WIDTH: u32 = ${layout.lookupWidth}u;

fn polygonPageWord(index: u32) -> u32 {
    return textureLoad(polygonPageLookup, vec2<i32>(i32(index % POLYGON_PAGE_LOOKUP_WIDTH), i32(index / POLYGON_PAGE_LOOKUP_WIDTH)), 0).x;
}

fn coarseAt(c: vec3<i32>) -> f32 {
    let d = coarseDim();
    let q = clamp(c, vec3<i32>(0), d - vec3<i32>(1));
    let cell = vec3<u32>(q);
    let block = cell / vec3<u32>(POLYGON_PAGE_SIZE);
    let blockIndex = block.x + POLYGON_PAGE_BLOCK_DIM.x * (block.y + POLYGON_PAGE_BLOCK_DIM.y * block.z);
    let page = polygonPageWord(POLYGON_PAGE_MAP_OFFSET + blockIndex);
    let maxPages = (arrayLength(&coarseSdf) - 1u) / POLYGON_PAGE_CELLS;
    if (page == 0u || page > maxPages) {
        return f32(${LIQUID_SDF_LAYERS}) * p.originDx.w * p.solve.w;
    }
    let local = cell % vec3<u32>(POLYGON_PAGE_SIZE);
    let localIndex = local.x + POLYGON_PAGE_SIZE * (local.y + POLYGON_PAGE_SIZE * local.z);
    return coarseSdf[(page - 1u) * POLYGON_PAGE_CELLS + localIndex];
}`
        : /* wgsl */ `
fn coarseAt(c: vec3<i32>) -> f32 {
    let d = coarseDim();
    let q = clamp(c, vec3<i32>(0), d - vec3<i32>(1));
    return coarseSdf[u32(q.x + d.x * (q.y + d.y * q.z))];
}`;
    return /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> coarseSdf: array<f32>;
@group(0) @binding(1) var<storage, read_write> fineSdf: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

fn coarseDim() -> vec3<i32> {
    return vec3<i32>(p.solve.xyz);
}

${pagedAddressing}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    ${layout ? "if (polygonPageWord(1u) != 0u) { return; }" : ""}
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&fineSdf)) {
        return;
    }
    let fineCoord = faceCoord(i, gridDim(p));
    let position = p.originDx.xyz + (vec3<f32>(fineCoord) + 0.5) * p.originDx.w;
    let coarseDx = p.originDx.w * p.solve.w;
    let sampleCoord = (position - p.originDx.xyz) / coarseDx - 0.5;
    let base = vec3<i32>(floor(sampleCoord));
    let weight = sampleCoord - vec3<f32>(base);
    let z0 = mix(
        mix(coarseAt(base), coarseAt(base + vec3<i32>(1, 0, 0)), weight.x),
        mix(coarseAt(base + vec3<i32>(0, 1, 0)), coarseAt(base + vec3<i32>(1, 1, 0)), weight.x),
        weight.y);
    let z1 = mix(
        mix(coarseAt(base + vec3<i32>(0, 0, 1)), coarseAt(base + vec3<i32>(1, 0, 1)), weight.x),
        mix(coarseAt(base + vec3<i32>(0, 1, 1)), coarseAt(base + vec3<i32>(1, 1, 1)), weight.x),
        weight.y);
    fineSdf[i] = mix(z0, z1, weight.z);
}
`;
}

function buildSolidFaceGeometryWgsl(scene: SceneSdfSpec | null): string {
    const sceneDecl = scene
        ? `${scene.struct}
@group(0) @binding(2) var<uniform> sceneSdfParams: SceneSdfParams;
${sceneSdfGridBindingWgsl(scene, 3)}
${scene.sdf}
${SCENE_NORMAL_WGSL}`
        : `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return 1.0e30; }
fn sceneNormal(pt: vec3<f32>, dt: f32) -> vec3<f32> { return vec3<f32>(0.0, 1.0, 0.0); }`;
    return /* wgsl */ `
${COMMON_WGSL}
${sceneDecl}
@group(0) @binding(0) var<storage, read_write> faceGeometry: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> p: Params;
${FLIP_FACE_APERTURE_WGSL}

fn faceCenter(kind: u32, c: vec3<i32>) -> vec3<f32> {
    var offset = vec3<f32>(0.5);
    if (kind == FACE_U) { offset.x = 0.0; }
    if (kind == FACE_V) { offset.y = 0.0; }
    if (kind == FACE_W) { offset.z = 0.0; }
    return p.originDx.xyz + (vec3<f32>(c) + offset) * p.originDx.w;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= totalFaceCount(p)) {
        return;
    }
    let kind = faceKind(i, p);
    let d = faceGridDim(kind, p);
    let c = faceCoord(faceLocalIndex(i, kind, p), d);
    if ((kind == FACE_U && (c.x == 0 || c.x == d.x - 1))
        || (kind == FACE_V && (c.y == 0 || c.y == d.y - 1))
        || (kind == FACE_W && (c.z == 0 || c.z == d.z - 1))) {
        faceGeometry[i] = vec2<f32>(0.0);
        return;
    }
    let center = faceCenter(kind, c);
    var tangentA = vec3<f32>(0.0);
    var tangentB = vec3<f32>(0.0);
    if (kind == FACE_U) {
        tangentA.y = 0.5 * p.originDx.w;
        tangentB.z = 0.5 * p.originDx.w;
    } else if (kind == FACE_V) {
        tangentA.x = 0.5 * p.originDx.w;
        tangentB.z = 0.5 * p.originDx.w;
    } else {
        tangentA.x = 0.5 * p.originDx.w;
        tangentB.y = 0.5 * p.originDx.w;
    }
    let open = flipFaceAperture(vec4<f32>(
        sceneSdf(center - tangentA - tangentB, 0.0),
        sceneSdf(center + tangentA - tangentB, 0.0),
        sceneSdf(center - tangentA + tangentB, 0.0),
        sceneSdf(center + tangentA + tangentB, 0.0)), p.originDx.w);
    var solidVelocity = 0.0;
    if (p.counts.z != 0u && open < 1.0) {
        let epsilon = max(1.0e-4, min(p.sim.x, 0.002));
        let normal = sceneNormal(center, 0.0);
        let normalSpeed = -(sceneSdf(center, epsilon) - sceneSdf(center, 0.0)) / epsilon;
        var component = normal.z;
        if (kind == FACE_U) { component = normal.x; }
        if (kind == FACE_V) { component = normal.y; }
        solidVelocity = normalSpeed * component;
    }
    faceGeometry[i] = vec2<f32>(open, solidVelocity);
}`;
}

const NORMALIZE_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> accum: array<FaceAccum>;
@group(0) @binding(1) var<storage, read_write> oldVelocity: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(4) var<uniform> p: Params;

fn cellTypeAt(c: vec3<i32>) -> u32 {
    if (!inCellGrid(c, p)) {
        return CELL_SOLID;
    }
    return cellTypes[cellIndex(c, p)];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= totalFaceCount(p)) {
        return;
    }
    let kind = faceKind(i, p);
    let d = faceGridDim(kind, p);
    let c = faceCoord(faceLocalIndex(i, kind, p), d);
    let adjacent = adjacentCells(kind, c);
    let solid = cellTypeAt(adjacent[0]) == CELL_SOLID || cellTypeAt(adjacent[1]) == CELL_SOLID;
    let weight = decodeFixed(atomicLoad(&accum[i].weight));
    if (solid || weight <= 1.0e-6) {
        oldVelocity[i] = 0.0;
        velocity[i] = vec2<f32>(0.0);
        return;
    }
    let base = decodeFixed(atomicLoad(&accum[i].momentum)) / weight;
    oldVelocity[i] = base;
    let forced = base - select(0.0, p.sim.y * p.sim.x, kind == FACE_V);
    velocity[i] = vec2<f32>(forced, 1.0);
}`;

const VISCOSITY_RHS_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> velocity: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= totalFaceCount(p)) {
        return;
    }
    rhs[i] = velocity[i].x;
}`;

const VISCOSITY_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> velocityIn: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocityOut: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(4) var<uniform> p: Params;

fn cellTypeAt(c: vec3<i32>) -> u32 {
    if (!inCellGrid(c, p)) {
        return CELL_SOLID;
    }
    return cellTypes[cellIndex(c, p)];
}

fn faceIsSolid(kind: u32, c: vec3<i32>) -> bool {
    let adjacent = adjacentCells(kind, c);
    return cellTypeAt(adjacent[0]) == CELL_SOLID || cellTypeAt(adjacent[1]) == CELL_SOLID;
}

fn addNeighbour(kind: u32, c: vec3<i32>, d: vec3<i32>, sum: ptr<function, f32>, count: ptr<function, f32>) {
    if (any(c < vec3<i32>(0)) || any(c >= d) || faceIsSolid(kind, c)) {
        return;
    }
    let sample = velocityIn[globalFaceIndex(kind, c, p)];
    if (sample.y > 0.5) {
        *sum += sample.x;
        *count += 1.0;
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= totalFaceCount(p)) {
        return;
    }
    let kind = faceKind(i, p);
    let d = faceGridDim(kind, p);
    let c = faceCoord(faceLocalIndex(i, kind, p), d);
    if (faceIsSolid(kind, c) || velocityIn[i].y <= 0.5) {
        velocityOut[i] = vec2<f32>(0.0);
        return;
    }
    var sum = 0.0;
    var count = 0.0;
    addNeighbour(kind, c + vec3<i32>(-1, 0, 0), d, &sum, &count);
    addNeighbour(kind, c + vec3<i32>(1, 0, 0), d, &sum, &count);
    addNeighbour(kind, c + vec3<i32>(0, -1, 0), d, &sum, &count);
    addNeighbour(kind, c + vec3<i32>(0, 1, 0), d, &sum, &count);
    addNeighbour(kind, c + vec3<i32>(0, 0, -1), d, &sum, &count);
    addNeighbour(kind, c + vec3<i32>(0, 0, 1), d, &sum, &count);
    let alpha = max(0.0, p.material.x) * p.sim.x / (p.originDx.w * p.originDx.w);
    velocityOut[i] = vec2<f32>((rhs[i] + alpha * sum) / (1.0 + alpha * count), 1.0);
}`;

const SURFACE_NORMAL_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;

fn occupancy(c: vec3<i32>, fallback: f32) -> f32 {
    if (!inCellGrid(c, p) || cellTypes[cellIndex(c, p)] == CELL_SOLID) {
        return fallback;
    }
    return min(f32(atomicLoad(&cellMarks[cellIndex(c, p)])) / max(1.0, f32(p.counts.y)), 1.0);
}

fn cellTypeAt(c: vec3<i32>) -> u32 {
    if (!inCellGrid(c, p)) {
        return CELL_SOLID;
    }
    return cellTypes[cellIndex(c, p)];
}

fn touchesAir(c: vec3<i32>) -> bool {
    return cellTypeAt(c + vec3<i32>(-1, 0, 0)) == CELL_AIR
        || cellTypeAt(c + vec3<i32>(1, 0, 0)) == CELL_AIR
        || cellTypeAt(c + vec3<i32>(0, -1, 0)) == CELL_AIR
        || cellTypeAt(c + vec3<i32>(0, 1, 0)) == CELL_AIR
        || cellTypeAt(c + vec3<i32>(0, 0, -1)) == CELL_AIR
        || cellTypeAt(c + vec3<i32>(0, 0, 1)) == CELL_AIR;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&normals)) {
        return;
    }
    let c = faceCoord(i, gridDim(p));
    if (cellTypes[i] != CELL_FLUID || !touchesAir(c)) {
        normals[i] = vec4<f32>(0.0);
        return;
    }
    let center = occupancy(c, 0.0);
    let gradient = vec3<f32>(
        occupancy(c + vec3<i32>(1, 0, 0), center) - occupancy(c - vec3<i32>(1, 0, 0), center),
        occupancy(c + vec3<i32>(0, 1, 0), center) - occupancy(c - vec3<i32>(0, 1, 0), center),
        occupancy(c + vec3<i32>(0, 0, 1), center) - occupancy(c - vec3<i32>(0, 0, 1), center)
    ) / (2.0 * p.originDx.w);
    let magnitude = length(gradient);
    normals[i] = vec4<f32>(select(vec3<f32>(0.0), gradient / magnitude, magnitude > 1.0e-6), magnitude);
}`;

const SURFACE_NORMAL_SDF_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> liquidSdf: array<f32>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;

fn phiAt(c: vec3<i32>, fallback: f32) -> f32 {
    if (!inCellGrid(c, p) || cellTypes[cellIndex(c, p)] == CELL_SOLID) {
        return fallback;
    }
    return liquidSdf[cellIndex(c, p)];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&normals)) {
        return;
    }
    let centerPhi = liquidSdf[i];
    if (cellTypes[i] == CELL_SOLID || abs(centerPhi) > 1.5 * p.originDx.w) {
        normals[i] = vec4<f32>(0.0);
        return;
    }
    let c = faceCoord(i, gridDim(p));
    let gradient = vec3<f32>(
        phiAt(c + vec3<i32>(1, 0, 0), centerPhi) - phiAt(c - vec3<i32>(1, 0, 0), centerPhi),
        phiAt(c + vec3<i32>(0, 1, 0), centerPhi) - phiAt(c - vec3<i32>(0, 1, 0), centerPhi),
        phiAt(c + vec3<i32>(0, 0, 1), centerPhi) - phiAt(c - vec3<i32>(0, 0, 1), centerPhi)
    ) / (2.0 * p.originDx.w);
    let magnitude = length(gradient);
    // Match the inward-facing legacy normal and its resolution-independent
    // interface-strength encoding.
    normals[i] = vec4<f32>(select(vec3<f32>(0.0), -gradient / magnitude, magnitude > 1.0e-6), magnitude / p.originDx.w);
}`;

const SURFACE_CURVATURE_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> curvature: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn normalAt(c: vec3<i32>, fallback: vec3<f32>) -> vec3<f32> {
    if (!inCellGrid(c, p) || cellTypes[cellIndex(c, p)] == CELL_SOLID) {
        return fallback;
    }
    return normals[cellIndex(c, p)].xyz;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&curvature)) {
        return;
    }
    if (cellTypes[i] == CELL_SOLID || normals[i].w <= 1.0e-6) {
        curvature[i] = 0.0;
        return;
    }
    let c = faceCoord(i, gridDim(p));
    let center = normals[i].xyz;
    let divergence =
        normalAt(c + vec3<i32>(1, 0, 0), center).x - normalAt(c - vec3<i32>(1, 0, 0), center).x
        + normalAt(c + vec3<i32>(0, 1, 0), center).y - normalAt(c - vec3<i32>(0, 1, 0), center).y
        + normalAt(c + vec3<i32>(0, 0, 1), center).z - normalAt(c - vec3<i32>(0, 0, 1), center).z;
    curvature[i] = -divergence / (2.0 * p.originDx.w);
}`;

const SURFACE_FORCE_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> velocity: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(3) var<storage, read> curvature: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

fn cellTypeAt(c: vec3<i32>) -> u32 {
    if (!inCellGrid(c, p)) {
        return CELL_SOLID;
    }
    return cellTypes[cellIndex(c, p)];
}

fn occupancy(c: vec3<i32>) -> f32 {
    if (!inCellGrid(c, p) || cellTypeAt(c) == CELL_SOLID) {
        return 0.0;
    }
    return min(f32(atomicLoad(&cellMarks[cellIndex(c, p)])) / max(1.0, f32(p.counts.y)), 1.0);
}

fn curvatureAt(c: vec3<i32>) -> f32 {
    if (!inCellGrid(c, p) || cellTypeAt(c) == CELL_SOLID) {
        return 0.0;
    }
    return curvature[cellIndex(c, p)];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= totalFaceCount(p)) {
        return;
    }
    let kind = faceKind(i, p);
    let d = faceGridDim(kind, p);
    let c = faceCoord(faceLocalIndex(i, kind, p), d);
    let adjacent = adjacentCells(kind, c);
    let leftType = cellTypeAt(adjacent[0]);
    let rightType = cellTypeAt(adjacent[1]);
    if (leftType == CELL_SOLID || rightType == CELL_SOLID || (leftType != CELL_FLUID && rightType != CELL_FLUID)) {
        return;
    }
    let indicatorGradient = (occupancy(adjacent[1]) - occupancy(adjacent[0])) / p.originDx.w;
    let faceCurvature = 0.5 * (curvatureAt(adjacent[0]) + curvatureAt(adjacent[1]));
    let force = p.material.y * faceCurvature * indicatorGradient;
    velocity[i] = vec2<f32>(velocity[i].x + force * p.sim.x, 1.0);
}`;

const SURFACE_FORCE_SDF_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> velocity: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> liquidSdf: array<f32>;
@group(0) @binding(2) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(3) var<storage, read> curvature: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

fn cellTypeAt(c: vec3<i32>) -> u32 {
    if (!inCellGrid(c, p)) {
        return CELL_SOLID;
    }
    return cellTypes[cellIndex(c, p)];
}

fn indicator(c: vec3<i32>) -> f32 {
    if (!inCellGrid(c, p) || cellTypeAt(c) == CELL_SOLID) {
        return 0.0;
    }
    return clamp(0.5 - liquidSdf[cellIndex(c, p)] / p.originDx.w, 0.0, 1.0);
}

fn curvatureAt(c: vec3<i32>) -> f32 {
    if (!inCellGrid(c, p) || cellTypeAt(c) == CELL_SOLID) {
        return 0.0;
    }
    return curvature[cellIndex(c, p)];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= totalFaceCount(p)) {
        return;
    }
    let kind = faceKind(i, p);
    let d = faceGridDim(kind, p);
    let c = faceCoord(faceLocalIndex(i, kind, p), d);
    let adjacent = adjacentCells(kind, c);
    let leftType = cellTypeAt(adjacent[0]);
    let rightType = cellTypeAt(adjacent[1]);
    if (leftType == CELL_SOLID || rightType == CELL_SOLID || (leftType != CELL_FLUID && rightType != CELL_FLUID)) {
        return;
    }
    let indicatorGradient = (indicator(adjacent[1]) - indicator(adjacent[0])) / p.originDx.w;
    let faceCurvature = 0.5 * (curvatureAt(adjacent[0]) + curvatureAt(adjacent[1]));
    let force = p.material.y * faceCurvature * indicatorGradient;
    velocity[i] = vec2<f32>(velocity[i].x + force * p.sim.x, 1.0);
}`;

function buildDivergenceWgsl(scene: SceneSdfSpec | null, fractionalSolids = false): string {
    const sceneDecl = scene
        ? `${scene.struct}
@group(0) @binding(5) var<uniform> sceneSdfParams: SceneSdfParams;
${sceneSdfGridBindingWgsl(scene, 6)}
${scene.sdf}`
        : "fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return 1.0e30; }";
    return /* wgsl */ `
${COMMON_WGSL}
${sceneDecl}
@group(0) @binding(0) var<storage, read> velocity: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read_write> accum: array<FaceAccum>;
${fractionalSolids ? "@group(0) @binding(7) var<storage, read> faceGeometry: array<vec2<f32>>;" : ""}
${FLIP_DENSITY_CORRECTION_WGSL}

fn faceValue(kind: u32, c: vec3<i32>) -> f32 {
    let index = globalFaceIndex(kind, c, p);
    ${
        fractionalSolids
            ? `let geometry = faceGeometry[index];
    return geometry.x * velocity[index].x + (1.0 - geometry.x) * geometry.y;`
            : "return velocity[index].x;"
    }
}

fn faceWeight(kind: u32, c: vec3<i32>) -> f32 {
    return decodeFixed(atomicLoad(&accum[globalFaceIndex(kind, c, p)].weight));
}

fn cellTypeAt(c: vec3<i32>) -> u32 {
    if (!inCellGrid(c, p)) {
        return CELL_SOLID;
    }
    return cellTypes[cellIndex(c, p)];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&divergence)) {
        return;
    }
    if (cellTypes[i] != CELL_FLUID) {
        divergence[i] = 0.0;
        return;
    }
    let c = faceCoord(i, gridDim(p));
    let div = faceValue(FACE_U, c + vec3<i32>(1, 0, 0)) - faceValue(FACE_U, c)
        + faceValue(FACE_V, c + vec3<i32>(0, 1, 0)) - faceValue(FACE_V, c)
        + faceValue(FACE_W, c + vec3<i32>(0, 0, 1)) - faceValue(FACE_W, c);
    let density = (
        faceWeight(FACE_U, c) + faceWeight(FACE_U, c + vec3<i32>(1, 0, 0))
        + faceWeight(FACE_V, c) + faceWeight(FACE_V, c + vec3<i32>(0, 1, 0))
        + faceWeight(FACE_W, c) + faceWeight(FACE_W, c + vec3<i32>(0, 0, 1))) / 6.0;
    // Face weights already contain a smooth trilinear marker-density estimate.
    // Bias compressed cells toward expansion so marker clustering cannot erase volume.
    let relativeDensity = density / max(1.0, f32(p.counts.y));
    let compression = max(relativeDensity - 1.0, 0.0);
    // Collision projection naturally concentrates markers in the cut-cell layer next
    // to a stationary solid. Treating that concentration as lost volume pushes the
    // neighbouring liquid inward while the projected markers remain pinned to the
    // wall, opening a cell-wide empty strip. Moving boundaries still need the density
    // correction to preserve volume while actively compressing and stirring liquid.
    let nearSolid =
        cellTypeAt(c + vec3<i32>(-1, 0, 0)) == CELL_SOLID
        || cellTypeAt(c + vec3<i32>(1, 0, 0)) == CELL_SOLID
        || cellTypeAt(c + vec3<i32>(0, -1, 0)) == CELL_SOLID
        || cellTypeAt(c + vec3<i32>(0, 1, 0)) == CELL_SOLID
        || cellTypeAt(c + vec3<i32>(0, 0, -1)) == CELL_SOLID
        || cellTypeAt(c + vec3<i32>(0, 0, 1)) == CELL_SOLID;
    var boundaryMoves = false;
    if (nearSolid) {
        let center = p.originDx.xyz + (vec3<f32>(c) + 0.5) * p.originDx.w;
        boundaryMoves = abs(sceneSdf(center, 0.002) - sceneSdf(center, 0.0)) > 1.0e-5;
    }
    let suppressBoundaryCorrection = nearSolid && !boundaryMoves;
    let expansion = select(flipDensityExpansion(compression, p.sim.x), 0.0, suppressBoundaryCorrection);
    divergence[i] = div / p.originDx.w - expansion;
}`;
}

function buildPressureWgsl(ghostFluid = false, fractionalSolids = false): string {
    return /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(2) var<storage, read> divergence: array<f32>;
@group(0) @binding(3) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(4) var<uniform> p: Params;
${ghostFluid ? "@group(0) @binding(5) var<storage, read> liquidSdf: array<f32>;" : ""}
${fractionalSolids ? "@group(0) @binding(6) var<storage, read> faceGeometry: array<vec2<f32>>;" : ""}

fn addNeighbour(c: vec3<i32>, faceIndex: u32, centerPhi: f32, sum: ptr<function, f32>, diagonal: ptr<function, f32>) {
    if (!inCellGrid(c, p)) {
        return;
    }
    let cellKind = cellTypes[cellIndex(c, p)];
    if (cellKind == CELL_SOLID) {
        return;
    }
    ${fractionalSolids ? "let faceWeight = faceGeometry[faceIndex].x;" : "let faceWeight = 1.0;"}
    if (faceWeight <= 1.0e-4) {
        return;
    }
    if (cellKind == CELL_FLUID) {
        *diagonal += faceWeight;
        *sum += faceWeight * pressureIn[cellIndex(c, p)];
    } else {
        ${
            ghostFluid
                ? `let neighbourPhi = liquidSdf[cellIndex(c, p)];
        let theta = clamp(centerPhi / min(centerPhi - neighbourPhi, -1.0e-6), 0.01, 1.0);
        *diagonal += faceWeight / theta;`
                : "*diagonal += faceWeight;"
        }
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&pressureOut)) {
        return;
    }
    if (cellTypes[i] != CELL_FLUID) {
        pressureOut[i] = 0.0;
        return;
    }
    let c = faceCoord(i, gridDim(p));
    var sum = 0.0;
    var diagonal = 0.0;
    ${ghostFluid ? "let centerPhi = liquidSdf[i];" : "let centerPhi = -p.originDx.w;"}
    addNeighbour(c + vec3<i32>(-1, 0, 0), globalFaceIndex(FACE_U, c, p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(1, 0, 0), globalFaceIndex(FACE_U, c + vec3<i32>(1, 0, 0), p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, -1, 0), globalFaceIndex(FACE_V, c, p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 1, 0), globalFaceIndex(FACE_V, c + vec3<i32>(0, 1, 0), p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 0, -1), globalFaceIndex(FACE_W, c, p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 0, 1), globalFaceIndex(FACE_W, c + vec3<i32>(0, 0, 1), p), centerPhi, &sum, &diagonal);
    if (diagonal <= 0.0) {
        pressureOut[i] = 0.0;
        return;
    }
    let rhs = divergence[i] * p.originDx.w * p.originDx.w;
    let candidate = (sum - rhs) / diagonal;
    pressureOut[i] = mix(pressureIn[i], candidate, p.solve.x);
}`;
}

function buildPressureResidualWgsl(ghostFluid = false, fractionalSolids = false): string {
    return /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(3) var<storage, read_write> residual: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
${ghostFluid ? "@group(0) @binding(5) var<storage, read> liquidSdf: array<f32>;" : ""}
${fractionalSolids ? "@group(0) @binding(6) var<storage, read> faceGeometry: array<vec2<f32>>;" : ""}

fn addNeighbour(c: vec3<i32>, faceIndex: u32, centerPhi: f32, sum: ptr<function, f32>, diagonal: ptr<function, f32>) {
    if (!inCellGrid(c, p)) {
        return;
    }
    let cellKind = cellTypes[cellIndex(c, p)];
    if (cellKind == CELL_SOLID) {
        return;
    }
    ${fractionalSolids ? "let faceWeight = faceGeometry[faceIndex].x;" : "let faceWeight = 1.0;"}
    if (faceWeight <= 1.0e-4) {
        return;
    }
    if (cellKind == CELL_FLUID) {
        *diagonal += faceWeight;
        *sum += faceWeight * pressure[cellIndex(c, p)];
    } else {
        ${
            ghostFluid
                ? `let neighbourPhi = liquidSdf[cellIndex(c, p)];
        let theta = clamp(centerPhi / min(centerPhi - neighbourPhi, -1.0e-6), 0.01, 1.0);
        *diagonal += faceWeight / theta;`
                : "*diagonal += faceWeight;"
        }
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&residual)) {
        return;
    }
    if (cellTypes[i] != CELL_FLUID) {
        residual[i] = 0.0;
        return;
    }
    let c = faceCoord(i, gridDim(p));
    var sum = 0.0;
    var diagonal = 0.0;
    ${ghostFluid ? "let centerPhi = liquidSdf[i];" : "let centerPhi = -p.originDx.w;"}
    addNeighbour(c + vec3<i32>(-1, 0, 0), globalFaceIndex(FACE_U, c, p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(1, 0, 0), globalFaceIndex(FACE_U, c + vec3<i32>(1, 0, 0), p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, -1, 0), globalFaceIndex(FACE_V, c, p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 1, 0), globalFaceIndex(FACE_V, c + vec3<i32>(0, 1, 0), p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 0, -1), globalFaceIndex(FACE_W, c, p), centerPhi, &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 0, 1), globalFaceIndex(FACE_W, c + vec3<i32>(0, 0, 1), p), centerPhi, &sum, &diagonal);
    let rhs = -divergence[i] * p.originDx.w * p.originDx.w;
    residual[i] = rhs - (diagonal * pressure[i] - sum);
}`;
}

const PRESSURE_DIAGNOSTIC_REDUCE_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(3) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&residual) || cellTypes[i] != CELL_FLUID) {
        return;
    }
    let rhs = abs(divergence[i] * p.originDx.w * p.originDx.w);
    atomicMax(&diagnostics[0], bitcast<u32>(abs(residual[i])));
    atomicMax(&diagnostics[1], bitcast<u32>(rhs));
    atomicAdd(&diagnostics[3], 1u);
}
`;

const POST_DIVERGENCE_DIAGNOSTIC_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> divergence: array<f32>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> diagnostics: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i < arrayLength(&divergence) && cellTypes[i] == CELL_FLUID) {
        atomicMax(&diagnostics[2], bitcast<u32>(abs(divergence[i])));
    }
}
`;

const MULTIGRID_COMMON_WGSL = /* wgsl */ `
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

struct MultigridParams {
    dim: vec4<u32>,
    fineDim: vec4<u32>,
    solve: vec4<f32>,
};

fn gridDim(p: MultigridParams) -> vec3<i32> {
    return vec3<i32>(p.dim.xyz);
}

fn fineGridDim(p: MultigridParams) -> vec3<i32> {
    return vec3<i32>(p.fineDim.xyz);
}

fn gridIndex(c: vec3<i32>, d: vec3<i32>) -> u32 {
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn gridCoord(index: u32, d: vec3<i32>) -> vec3<i32> {
    let x = i32(index % u32(d.x));
    let yz = i32(index / u32(d.x));
    let y = yz % d.y;
    return vec3<i32>(x, y, yz / d.y);
}

fn inGrid(c: vec3<i32>, d: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < d);
}
`;

const MULTIGRID_BUILD_RHS_WGSL = /* wgsl */ `
${MULTIGRID_COMMON_WGSL}
@group(0) @binding(0) var<storage, read> divergence: array<f32>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(3) var<uniform> p: MultigridParams;
@group(0) @binding(4) var<storage, read_write> fluidFraction: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&rhs)) {
        return;
    }
    let fluid = cellTypes[i] == CELL_FLUID;
    rhs[i] = select(0.0, -divergence[i] * p.solve.y, fluid);
    fluidFraction[i] = select(0.0, 1.0, fluid);
}`;

const MULTIGRID_SMOOTH_WGSL = /* wgsl */ `
${MULTIGRID_COMMON_WGSL}
@group(0) @binding(0) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(2) var<storage, read> rhs: array<f32>;
@group(0) @binding(3) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(4) var<uniform> p: MultigridParams;

fn addNeighbour(c: vec3<i32>, sum: ptr<function, f32>, diagonal: ptr<function, f32>) {
    let d = gridDim(p);
    if (!inGrid(c, d)) {
        return;
    }
    let cellKind = cellTypes[gridIndex(c, d)];
    if (cellKind == CELL_SOLID) {
        return;
    }
    *diagonal += 1.0;
    if (cellKind == CELL_FLUID) {
        *sum += pressureIn[gridIndex(c, d)];
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&pressureOut)) {
        return;
    }
    if (cellTypes[i] != CELL_FLUID) {
        pressureOut[i] = 0.0;
        return;
    }
    let c = gridCoord(i, gridDim(p));
    var sum = 0.0;
    var diagonal = 0.0;
    addNeighbour(c + vec3<i32>(-1, 0, 0), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(1, 0, 0), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, -1, 0), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 1, 0), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 0, -1), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 0, 1), &sum, &diagonal);
    if (diagonal <= 0.0) {
        pressureOut[i] = 0.0;
        return;
    }
    let candidate = (sum + rhs[i]) / diagonal;
    pressureOut[i] = mix(pressureIn[i], candidate, p.solve.x);
}`;

const MULTIGRID_RESIDUAL_WGSL = /* wgsl */ `
${MULTIGRID_COMMON_WGSL}
@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(3) var<storage, read_write> residual: array<f32>;
@group(0) @binding(4) var<uniform> p: MultigridParams;

fn addNeighbour(c: vec3<i32>, sum: ptr<function, f32>, diagonal: ptr<function, f32>) {
    let d = gridDim(p);
    if (!inGrid(c, d)) {
        return;
    }
    let cellKind = cellTypes[gridIndex(c, d)];
    if (cellKind == CELL_SOLID) {
        return;
    }
    *diagonal += 1.0;
    if (cellKind == CELL_FLUID) {
        *sum += pressure[gridIndex(c, d)];
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&residual)) {
        return;
    }
    if (cellTypes[i] != CELL_FLUID) {
        residual[i] = 0.0;
        return;
    }
    let c = gridCoord(i, gridDim(p));
    var sum = 0.0;
    var diagonal = 0.0;
    addNeighbour(c + vec3<i32>(-1, 0, 0), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(1, 0, 0), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, -1, 0), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 1, 0), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 0, -1), &sum, &diagonal);
    addNeighbour(c + vec3<i32>(0, 0, 1), &sum, &diagonal);
    residual[i] = rhs[i] - (diagonal * pressure[i] - sum);
}`;

const MULTIGRID_RESTRICT_WGSL = /* wgsl */ `
${MULTIGRID_COMMON_WGSL}
@group(0) @binding(0) var<storage, read> fineResidual: array<f32>;
@group(0) @binding(1) var<storage, read> fineTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> coarseRhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> coarseTypes: array<u32>;
@group(0) @binding(4) var<storage, read_write> coarsePressureA: array<f32>;
@group(0) @binding(5) var<storage, read_write> coarsePressureB: array<f32>;
@group(0) @binding(6) var<storage, read> fineFraction: array<f32>;
@group(0) @binding(7) var<storage, read_write> coarseFraction: array<f32>;
@group(0) @binding(8) var<uniform> p: MultigridParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&coarseRhs)) {
        return;
    }
    coarsePressureA[i] = 0.0;
    coarsePressureB[i] = 0.0;
    let coarseCoord = gridCoord(i, gridDim(p));
    let fineBase = coarseCoord * 2;
    let fineDim = fineGridDim(p);
    var residualSum = 0.0;
    var fractionSum = 0.0;
    var childCount = 0u;
    var solidCount = 0u;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = fineBase + vec3<i32>(x, y, z);
                if (!inGrid(c, fineDim)) {
                    continue;
                }
                let fineIndex = gridIndex(c, fineDim);
                let cellKind = fineTypes[fineIndex];
                if (cellKind == CELL_FLUID) {
                    let fraction = fineFraction[fineIndex];
                    fractionSum += fraction;
                    residualSum += fraction * fineResidual[fineIndex];
                } else if (cellKind == CELL_SOLID) {
                    solidCount += 1u;
                }
                childCount += 1u;
            }
        }
    }
    coarseFraction[i] = fractionSum / max(1.0, f32(childCount));
    if (fractionSum > 1.0e-4) {
        coarseTypes[i] = CELL_FLUID;
        coarseRhs[i] = 4.0 * residualSum / max(1.0, f32(childCount));
    } else {
        coarseTypes[i] = select(CELL_AIR, CELL_SOLID, solidCount > 0u);
        coarseRhs[i] = 0.0;
    }
}`;

const MULTIGRID_PROLONGATE_WGSL = /* wgsl */ `
${MULTIGRID_COMMON_WGSL}
@group(0) @binding(0) var<storage, read> coarsePressure: array<f32>;
@group(0) @binding(1) var<storage, read> coarseTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> finePressure: array<f32>;
@group(0) @binding(3) var<storage, read> fineTypes: array<u32>;
@group(0) @binding(4) var<uniform> p: MultigridParams;

fn coarsePressureAt(c: vec3<i32>) -> f32 {
    let d = gridDim(p);
    if (!inGrid(c, d)) {
        return 0.0;
    }
    let i = gridIndex(c, d);
    return select(0.0, coarsePressure[i], coarseTypes[i] == CELL_FLUID);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&finePressure)) {
        return;
    }
    if (fineTypes[i] != CELL_FLUID) {
        finePressure[i] = 0.0;
        return;
    }
    let fineCoord = gridCoord(i, fineGridDim(p));
    let coarsePosition = (vec3<f32>(fineCoord) + 0.5) * 0.5 - 0.5;
    let base = vec3<i32>(floor(coarsePosition));
    let fraction = coarsePosition - floor(coarsePosition);
    var correction = 0.0;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let offset = vec3<i32>(x, y, z);
                let weightAxis = select(vec3<f32>(1.0) - fraction, fraction, offset == vec3<i32>(1));
                correction += coarsePressureAt(base + offset) * weightAxis.x * weightAxis.y * weightAxis.z;
            }
        }
    }
    finePressure[i] += correction * ${MULTIGRID_CORRECTION_DAMPING};
}`;

function buildPagedMultigridAddressingWgsl(layout: FlipPageLayout): string {
    const mapOffset = 2;
    const coordOffset = mapOffset + layout.numBlocks;
    return /* wgsl */ `
@group(1) @binding(0) var pageLookup: texture_2d<u32>;
const PAGE_SIZE: u32 = ${FLIP_PAGE_SIZE}u;
const PAGE_CELLS: u32 = ${FLIP_PAGE_CELLS}u;
const PAGE_BLOCK_DIM: vec3<u32> = vec3<u32>(${layout.blockDim[0]}u, ${layout.blockDim[1]}u, ${layout.blockDim[2]}u);
const PAGE_MAP_OFFSET: u32 = ${mapOffset}u;
const PAGE_COORD_OFFSET: u32 = ${coordOffset}u;
const PAGE_LOOKUP_WIDTH: u32 = ${layout.lookupWidth}u;

fn pageStorageCells() -> u32 {
    return arrayLength(&fineTypes) - 1u;
}

fn pageMaxPages() -> u32 {
    return pageStorageCells() / PAGE_CELLS;
}

fn pageWord(index: u32) -> u32 {
    return textureLoad(pageLookup, vec2<i32>(i32(index % PAGE_LOOKUP_WIDTH), i32(index / PAGE_LOOKUP_WIDTH)), 0).x;
}

fn pagedFineIndex(c: vec3<i32>, d: vec3<i32>) -> u32 {
    if (!inGrid(c, d)) {
        return pageStorageCells();
    }
    let cell = vec3<u32>(c);
    let block = cell / vec3<u32>(PAGE_SIZE);
    let blockIndex = block.x + PAGE_BLOCK_DIM.x * (block.y + PAGE_BLOCK_DIM.y * block.z);
    let page = pageWord(PAGE_MAP_OFFSET + blockIndex);
    if (page == 0u || page > pageMaxPages()) {
        return pageStorageCells();
    }
    let local = cell % vec3<u32>(PAGE_SIZE);
    let localIndex = local.x + PAGE_SIZE * (local.y + PAGE_SIZE * local.z);
    return (page - 1u) * PAGE_CELLS + localIndex;
}

fn pagedFineCoord(index: u32) -> vec3<i32> {
    let page = index / PAGE_CELLS;
    let localIndex = index % PAGE_CELLS;
    let blockIndex = pageWord(PAGE_COORD_OFFSET + page);
    let block = vec3<u32>(
        blockIndex % PAGE_BLOCK_DIM.x,
        (blockIndex / PAGE_BLOCK_DIM.x) % PAGE_BLOCK_DIM.y,
        blockIndex / (PAGE_BLOCK_DIM.x * PAGE_BLOCK_DIM.y)
    );
    let local = vec3<u32>(
        localIndex % PAGE_SIZE,
        (localIndex / PAGE_SIZE) % PAGE_SIZE,
        localIndex / (PAGE_SIZE * PAGE_SIZE)
    );
    return vec3<i32>(block * PAGE_SIZE + local);
}
`;
}

function buildPagedMultigridRestrictWgsl(layout: FlipPageLayout): string {
    return /* wgsl */ `
${MULTIGRID_COMMON_WGSL}
${buildPagedMultigridAddressingWgsl(layout)}
@group(0) @binding(0) var<storage, read> fineResidual: array<f32>;
@group(0) @binding(1) var<storage, read> fineTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> coarseRhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> coarseTypes: array<u32>;
@group(0) @binding(4) var<storage, read_write> coarsePressureA: array<f32>;
@group(0) @binding(5) var<storage, read_write> coarsePressureB: array<f32>;
@group(0) @binding(6) var<storage, read> fineFraction: array<f32>;
@group(0) @binding(7) var<storage, read_write> coarseFraction: array<f32>;
@group(0) @binding(8) var<uniform> p: MultigridParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&coarseRhs)) {
        return;
    }
    coarsePressureA[i] = 0.0;
    coarsePressureB[i] = 0.0;
    let coarseCoord = gridCoord(i, gridDim(p));
    let fineBase = coarseCoord * 2;
    let fineDim = fineGridDim(p);
    var residualSum = 0.0;
    var fractionSum = 0.0;
    var childCount = 0u;
    var solidCount = 0u;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = fineBase + vec3<i32>(x, y, z);
                if (!inGrid(c, fineDim)) {
                    continue;
                }
                let fineIndex = pagedFineIndex(c, fineDim);
                let cellKind = fineTypes[fineIndex];
                if (cellKind == CELL_FLUID) {
                    let fraction = fineFraction[fineIndex];
                    fractionSum += fraction;
                    residualSum += fraction * fineResidual[fineIndex];
                } else if (cellKind == CELL_SOLID) {
                    solidCount += 1u;
                }
                childCount += 1u;
            }
        }
    }
    coarseFraction[i] = fractionSum / max(1.0, f32(childCount));
    if (fractionSum > 1.0e-4) {
        coarseTypes[i] = CELL_FLUID;
        coarseRhs[i] = 4.0 * residualSum / max(1.0, f32(childCount));
    } else {
        coarseTypes[i] = select(CELL_AIR, CELL_SOLID, solidCount > 0u);
        coarseRhs[i] = 0.0;
    }
}`;
}

function buildPagedMultigridProlongateWgsl(layout: FlipPageLayout): string {
    return /* wgsl */ `
${MULTIGRID_COMMON_WGSL}
${buildPagedMultigridAddressingWgsl(layout)}
@group(0) @binding(0) var<storage, read> coarsePressure: array<f32>;
@group(0) @binding(1) var<storage, read> coarseTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> finePressure: array<f32>;
@group(0) @binding(3) var<storage, read> fineTypes: array<u32>;
@group(0) @binding(4) var<uniform> p: MultigridParams;

fn coarsePressureAt(c: vec3<i32>) -> f32 {
    let d = gridDim(p);
    if (!inGrid(c, d)) {
        return 0.0;
    }
    let i = gridIndex(c, d);
    return select(0.0, coarsePressure[i], coarseTypes[i] == CELL_FLUID);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&finePressure) || fineTypes[i] != CELL_FLUID) {
        return;
    }
    let fineCoord = pagedFineCoord(i);
    if (!inGrid(fineCoord, fineGridDim(p))) {
        finePressure[i] = 0.0;
        return;
    }
    let coarsePosition = (vec3<f32>(fineCoord) + 0.5) * 0.5 - 0.5;
    let base = vec3<i32>(floor(coarsePosition));
    let fraction = coarsePosition - floor(coarsePosition);
    var correction = 0.0;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let offset = vec3<i32>(x, y, z);
                let weightAxis = select(vec3<f32>(1.0) - fraction, fraction, offset == vec3<i32>(1));
                correction += coarsePressureAt(base + offset) * weightAxis.x * weightAxis.y * weightAxis.z;
            }
        }
    }
    finePressure[i] += correction * ${MULTIGRID_CORRECTION_DAMPING};
}`;
}

function buildProjectWgsl(ghostFluid = false, fractionalSolids = false): string {
    return /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> oldVelocity: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> deltaVelocity: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(5) var<uniform> p: Params;
${ghostFluid ? "@group(0) @binding(6) var<storage, read> liquidSdf: array<f32>;" : ""}
${fractionalSolids ? "@group(0) @binding(7) var<storage, read> faceGeometry: array<vec2<f32>>;" : ""}

fn cellTypeAt(c: vec3<i32>) -> u32 {
    if (!inCellGrid(c, p)) {
        return CELL_SOLID;
    }
    return cellTypes[cellIndex(c, p)];
}

fn pressureAt(c: vec3<i32>) -> f32 {
    if (!inCellGrid(c, p)) {
        return 0.0;
    }
    return select(0.0, pressure[cellIndex(c, p)], cellTypeAt(c) == CELL_FLUID);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= totalFaceCount(p)) {
        return;
    }
    let kind = faceKind(i, p);
    let d = faceGridDim(kind, p);
    let c = faceCoord(faceLocalIndex(i, kind, p), d);
    let adjacent = adjacentCells(kind, c);
    let leftType = cellTypeAt(adjacent[0]);
    let rightType = cellTypeAt(adjacent[1]);
    ${fractionalSolids ? "let geometry = faceGeometry[i];" : "let geometry = vec2<f32>(1.0, 0.0);"}
    if (leftType == CELL_SOLID || rightType == CELL_SOLID) {
        velocity[i] = vec2<f32>(geometry.y, 0.0);
        deltaVelocity[i] = vec2<f32>(geometry.y - oldVelocity[i], 0.0);
        return;
    }
    if (leftType != CELL_FLUID && rightType != CELL_FLUID) {
        velocity[i] = vec2<f32>(0.0);
        deltaVelocity[i] = vec2<f32>(0.0);
        return;
    }
    if (geometry.x <= 1.0e-4) {
        velocity[i] = vec2<f32>(geometry.y, 0.0);
        deltaVelocity[i] = vec2<f32>(geometry.y - oldVelocity[i], 0.0);
        return;
    }
    var theta = 1.0;
    ${
        ghostFluid
            ? `if (leftType != rightType && (leftType == CELL_AIR || rightType == CELL_AIR)) {
        let leftPhi = liquidSdf[cellIndex(adjacent[0], p)];
        let rightPhi = liquidSdf[cellIndex(adjacent[1], p)];
        theta = clamp(abs(select(rightPhi, leftPhi, leftType == CELL_FLUID)) / max(abs(leftPhi) + abs(rightPhi), 1.0e-6), 0.01, 1.0);
    }`
            : ""
    }
    let rawProjected = velocity[i].x - (pressureAt(adjacent[1]) - pressureAt(adjacent[0])) / (theta * p.originDx.w);
    let finiteProjected = select(0.0, rawProjected, rawProjected == rawProjected);
    let projected = clamp(finiteProjected, -p.solve.w, p.solve.w);
    velocity[i] = vec2<f32>(projected, 1.0);
    deltaVelocity[i] = vec2<f32>(projected - oldVelocity[i], 1.0);
}`;
}

const EXTRAPOLATE_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> velocityIn: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> deltaIn: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocityOut: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> deltaOut: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(5) var<uniform> p: Params;

fn cellTypeAt(c: vec3<i32>) -> u32 {
    if (!inCellGrid(c, p)) {
        return CELL_SOLID;
    }
    return cellTypes[cellIndex(c, p)];
}

fn faceIsSolid(kind: u32, c: vec3<i32>) -> bool {
    let adjacent = adjacentCells(kind, c);
    return cellTypeAt(adjacent[0]) == CELL_SOLID || cellTypeAt(adjacent[1]) == CELL_SOLID;
}

fn addFace(kind: u32, c: vec3<i32>, d: vec3<i32>, velocitySum: ptr<function, f32>, deltaSum: ptr<function, f32>, count: ptr<function, f32>) {
    if (any(c < vec3<i32>(0)) || any(c >= d)) {
        return;
    }
    let index = globalFaceIndex(kind, c, p);
    if (velocityIn[index].y > 0.5) {
        *velocitySum += velocityIn[index].x;
        *deltaSum += deltaIn[index].x;
        *count += 1.0;
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= totalFaceCount(p)) {
        return;
    }
    if (velocityIn[i].y > 0.5) {
        velocityOut[i] = velocityIn[i];
        deltaOut[i] = deltaIn[i];
        return;
    }
    let kind = faceKind(i, p);
    let d = faceGridDim(kind, p);
    let c = faceCoord(faceLocalIndex(i, kind, p), d);
    if (faceIsSolid(kind, c)) {
        velocityOut[i] = vec2<f32>(0.0);
        deltaOut[i] = vec2<f32>(0.0);
        return;
    }
    var velocitySum = 0.0;
    var deltaSum = 0.0;
    var count = 0.0;
    addFace(kind, c + vec3<i32>(-1, 0, 0), d, &velocitySum, &deltaSum, &count);
    addFace(kind, c + vec3<i32>(1, 0, 0), d, &velocitySum, &deltaSum, &count);
    addFace(kind, c + vec3<i32>(0, -1, 0), d, &velocitySum, &deltaSum, &count);
    addFace(kind, c + vec3<i32>(0, 1, 0), d, &velocitySum, &deltaSum, &count);
    addFace(kind, c + vec3<i32>(0, 0, -1), d, &velocitySum, &deltaSum, &count);
    addFace(kind, c + vec3<i32>(0, 0, 1), d, &velocitySum, &deltaSum, &count);
    if (count > 0.0) {
        velocityOut[i] = vec2<f32>(velocitySum / count, 1.0);
        deltaOut[i] = vec2<f32>(deltaSum / count, 1.0);
    } else {
        velocityOut[i] = vec2<f32>(0.0);
        deltaOut[i] = vec2<f32>(0.0);
    }
}`;

function buildG2pWgsl(scene: SceneSdfSpec | null): string {
    const sceneBinding = scene ? 6 : -1;
    const gridBinding = scene?.sdfGrid ? 7 : -1;
    const lifecycleBinding = scene ? (scene.sdfGrid ? 8 : 7) : 6;
    const sceneDecl = scene
        ? `${scene.struct}
@group(0) @binding(${sceneBinding}) var<uniform> sceneSdfParams: SceneSdfParams;
${sceneSdfGridBindingWgsl(scene, gridBinding)}
${scene.sdf}
${SCENE_NORMAL_WGSL}`
        : `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return 1.0e30; }
fn sceneNormal(pt: vec3<f32>, dt: f32) -> vec3<f32> { return vec3<f32>(0.0, 1.0, 0.0); }`;
    return /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
${sceneDecl}
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> faceVelocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> faceDelta: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> p: Params;
@group(0) @binding(5) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(${lifecycleBinding}) var<storage, read_write> lifecycle: FluidLifecycle;

fn sampleComponent(world: vec3<f32>, kind: u32, delta: bool) -> f32 {
    let d = faceGridDim(kind, p);
    let grid = (world - p.originDx.xyz) / p.originDx.w - faceOffset(kind);
    let base = vec3<i32>(floor(grid));
    let f = grid - floor(grid);
    var weighted = 0.0;
    var weightSum = 0.0;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (any(c < vec3<i32>(0)) || any(c >= d)) {
                    continue;
                }
                let q = vec3<f32>(f32(x), f32(y), f32(z));
                let w3 = select(vec3<f32>(1.0) - f, f, q > vec3<f32>(0.5));
                let weight = w3.x * w3.y * w3.z;
                let index = globalFaceIndex(kind, c, p);
                let sample = select(faceVelocity[index], faceDelta[index], delta);
                if (sample.y > 0.5) {
                    weighted += weight * sample.x;
                    weightSum += weight;
                }
            }
        }
    }
    return select(0.0, weighted / weightSum, weightSum > 1.0e-6);
}

fn sampleVector(world: vec3<f32>, delta: bool) -> vec3<f32> {
    return vec3<f32>(
        sampleComponent(world, FACE_U, delta),
        sampleComponent(world, FACE_V, delta),
        sampleComponent(world, FACE_W, delta));
}

fn markerHash(value: u32) -> u32 {
    var hash = value;
    hash ^= hash >> 16u;
    hash *= 0x7feb352du;
    hash ^= hash >> 15u;
    hash *= 0x846ca68bu;
    hash ^= hash >> 16u;
    return hash;
}

fn markerRedistribution(world: vec3<f32>, particleIndex: u32) -> vec4<f32> {
    let c = clamp(vec3<i32>(floor((world - p.originDx.xyz) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
    let current = atomicLoad(&cellMarks[cellIndex(c, p)]);
    let crowded = u32(ceil(1.25 * f32(p.counts.y)));
    if (current <= crowded) {
        return vec4<f32>(world, 0.0);
    }
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1));
    let hash = markerHash(particleIndex ^ (cellIndex(c, p) * 0x9e3779b9u) ^ current);
    var candidateCount = 0u;
    var candidatePriority = 0u;
    var targetCenter = world;
    for (var neighbour = 0; neighbour < 6; neighbour = neighbour + 1) {
        let nc = c + offsets[neighbour];
        if (!inCellGrid(nc, p)) {
            continue;
        }
        let neighbourCenter = p.originDx.xyz + (vec3<f32>(nc) + 0.5) * p.originDx.w;
        if (sceneSdf(neighbourCenter, 0.0) <= p.solve.z) {
            continue;
        }
        let neighbourCount = atomicLoad(&cellMarks[cellIndex(nc, p)]);
        let refillEmpty = current > 2u * p.counts.y && offsets[neighbour].y >= 0;
        if (neighbourCount < current && (neighbourCount > 0u || refillEmpty)) {
            let priority = select(1u, 2u, offsets[neighbour].y == 0);
            if (priority > candidatePriority) {
                candidatePriority = priority;
                candidateCount = 1u;
                targetCenter = neighbourCenter;
            } else if (priority == candidatePriority) {
                candidateCount += 1u;
                if (markerHash(hash ^ candidateCount) % candidateCount == 0u) {
                    targetCenter = neighbourCenter;
                }
            }
        }
    }
    if (candidateCount == 0u) {
        return vec4<f32>(world, 0.0);
    }
    let moveProbability = min(0.25, f32(current - crowded) / f32(current));
    if (f32(hash & 0x00ffffffu) / 16777216.0 >= moveProbability) {
        return vec4<f32>(world, 0.0);
    }
    let jitter = vec3<f32>(
        f32(markerHash(hash ^ 0x68bc21ebu) & 0xffffu) / 65535.0,
        f32(markerHash(hash ^ 0x02e5be93u) & 0xffffu) / 65535.0,
        f32(markerHash(hash ^ 0x967a889bu) & 0xffffu) / 65535.0) - 0.5;
    return vec4<f32>(targetCenter + jitter * (0.5 * p.originDx.w), 1.0);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsActive(i)) {
        return;
    }
    let world = positions[i].xyz;
    let oldParticleVelocity = velocities[i].xyz;
    let pic = sampleVector(world, false);
    let flip = oldParticleVelocity + sampleVector(world, true);
    var velocity = mix(pic, flip, clamp(p.sim.z, 0.0, 1.0));
    velocity *= exp(-max(0.0, p.sim.w) * p.sim.x);
    velocity = clampLength(velocity, p.solve.w);
    let advectionVelocity = clampLength(pic, p.solve.w);
    let midpoint = world + advectionVelocity * (0.5 * p.sim.x);
    let midpointVelocity = clampLength(sampleVector(midpoint, false), p.solve.w);
    var next = world + midpointVelocity * p.sim.x;
    let redistributed = markerRedistribution(world, i);
    next = select(next, redistributed.xyz, redistributed.w > 0.5);
    let radius = p.solve.z;
    let lo = vec3<f32>(p.boundsMin.x + radius, max(p.boundsMin.y, p.dimGround.w) + radius, p.boundsMin.z + radius);
    let hi = p.boundsMax.xyz - vec3<f32>(radius);
    let clamped = clamp(next, lo, hi);
    if (clamped.x != next.x && velocity.x * (next.x - clamped.x) > 0.0) {
        velocity.x = -velocity.x * p.solve.y;
    }
    if (clamped.y != next.y && velocity.y * (next.y - clamped.y) > 0.0) {
        velocity.y = -velocity.y * p.solve.y;
    }
    if (clamped.z != next.z && velocity.z * (next.z - clamped.z) > 0.0) {
        velocity.z = -velocity.z * p.solve.y;
    }
    next = clamped;
    let distance = sceneSdf(next, 0.0);
    if (distance < radius) {
        let normal = safeNormal(sceneNormal(next, 0.0), vec3<f32>(0.0, 1.0, 0.0));
        let correction = min(radius - distance, p.solve.w * p.sim.x + radius);
        next += correction * normal;
        // Resolve velocity relative to the moving boundary, not a static wall.
        let boundaryNormalSpeed = clamp(-(sceneSdf(next, 0.002) - sceneSdf(next, 0.0)) / 0.002, -p.solve.w, p.solve.w);
        let boundaryVelocity = boundaryNormalSpeed * normal;
        let relativeNormalVelocity = dot(velocity - boundaryVelocity, normal);
        if (relativeNormalVelocity < 0.0) {
            velocity -= (1.0 + p.solve.y) * relativeNormalVelocity * normal;
        }
        next = clamp(next, lo, hi);
    }
    velocity = clampLength(velocity, p.solve.w);
    positions[i] = vec4<f32>(next, 1.0);
    velocities[i] = vec4<f32>(velocity, 0.0);
}`;
}

function buildReseedDeleteWgsl(overfullOnly: boolean): string {
    return /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read_write> lifecycle: FluidLifecycle;
@group(0) @binding(5) var<storage, read_write> state: array<atomic<u32>>;

fn reseedWorkBudget() -> u32 {
    let listCapacity = atomicLoad(&state[2]);
    return min(listCapacity, max(${RESEED_MIN_WORK_BUDGET}u, p.counts.x / ${RESEED_WORK_BUDGET_DIVISOR}u));
}

fn reserveRecycleRequest() -> u32 {
    let requestCount = min(atomicLoad(&state[0]), reseedWorkBudget());
    loop {
        let recycled = atomicLoad(&state[4]);
        if (recycled >= requestCount) {
            return 0xffffffffu;
        }
        if (atomicCompareExchangeWeak(&state[4], recycled, recycled + 1u).exchanged) {
            return recycled;
        }
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsActive(i)) {
        return;
    }
    let c = clamp(vec3<i32>(floor((positions[i].xyz - p.originDx.xyz) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
    let cell = cellIndex(c, p);
    loop {
        let current = atomicLoad(&cellMarks[cell]);
        if (current <= ${overfullOnly ? "p.reseed.w" : "p.reseed.z"}) {
            return;
        }
        if (atomicCompareExchangeWeak(&cellMarks[cell], current, current - 1u).exchanged) {
            let ticket = reserveRecycleRequest();
            if (ticket == 0xffffffffu) {
                atomicAdd(&cellMarks[cell], 1u);
                return;
            }
            if (fluidDeleteParticle(i)) {
                let listCapacity = atomicLoad(&state[2]);
                atomicStore(&state[${RESEED_HEADER_WORDS}u + listCapacity + ticket], i);
            } else {
                atomicAdd(&cellMarks[cell], 1u);
            }
            return;
        }
    }
}
`;
}

const RESEED_BUILD_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p: Params;

fn reseedWorkBudget() -> u32 {
    let listCapacity = atomicLoad(&state[2]);
    return min(listCapacity, max(${RESEED_MIN_WORK_BUDGET}u, p.counts.x / ${RESEED_WORK_BUDGET_DIVISOR}u));
}

fn touchesFreeSurface(c: vec3<i32>) -> bool {
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1));
    for (var neighbour = 0u; neighbour < 6u; neighbour = neighbour + 1u) {
        let nc = c + offsets[neighbour];
        if (inCellGrid(nc, p) && cellTypes[cellIndex(nc, p)] == CELL_AIR) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i == 0u) {
        atomicAdd(&state[3], 1u);
    }
    if (i >= arrayLength(&cellTypes) || cellTypes[i] != CELL_FLUID) {
        return;
    }
    let c = faceCoord(i, gridDim(p));
    if (touchesFreeSurface(c)) {
        return;
    }
    let current = atomicLoad(&cellMarks[i]);
    if (current >= p.reseed.y) {
        return;
    }
    let deficit = p.reseed.z - min(current, p.reseed.z);
    let capacity = reseedWorkBudget();
    for (var marker = 0u; marker < deficit; marker = marker + 1u) {
        let ticket = atomicAdd(&state[0], 1u);
        if (ticket < capacity) {
            atomicStore(&state[${RESEED_HEADER_WORDS}u + ticket], i);
        }
    }
}
`;

function buildReseedEmitWgsl(scene: SceneSdfSpec | null): string {
    const sceneDecl = scene
        ? `${scene.struct}
@group(0) @binding(7) var<uniform> sceneSdfParams: SceneSdfParams;
${sceneSdfGridBindingWgsl(scene, 8)}
${scene.sdf}
${SCENE_NORMAL_WGSL}`
        : `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return 1.0e30; }
fn sceneNormal(pt: vec3<f32>, dt: f32) -> vec3<f32> { return vec3<f32>(0.0, 1.0, 0.0); }`;
    return /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
${sceneDecl}
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> faceVelocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(4) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p: Params;
@group(0) @binding(6) var<storage, read_write> lifecycle: FluidLifecycle;

fn reseedWorkBudget() -> u32 {
    let listCapacity = atomicLoad(&state[2]);
    return min(listCapacity, max(${RESEED_MIN_WORK_BUDGET}u, p.counts.x / ${RESEED_WORK_BUDGET_DIVISOR}u));
}

fn hash(value: u32) -> u32 {
    var result = value;
    result ^= result >> 16u;
    result *= 0x7feb352du;
    result ^= result >> 15u;
    result *= 0x846ca68bu;
    result ^= result >> 16u;
    return result;
}

fn sampleComponent(world: vec3<f32>, kind: u32) -> f32 {
    let d = faceGridDim(kind, p);
    let grid = (world - p.originDx.xyz) / p.originDx.w - faceOffset(kind);
    let base = vec3<i32>(floor(grid));
    let f = grid - floor(grid);
    var weighted = 0.0;
    var weightSum = 0.0;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (any(c < vec3<i32>(0)) || any(c >= d)) {
                    continue;
                }
                let q = vec3<f32>(f32(x), f32(y), f32(z));
                let w3 = select(vec3<f32>(1.0) - f, f, q > vec3<f32>(0.5));
                let sample = faceVelocity[globalFaceIndex(kind, c, p)];
                if (sample.y > 0.5) {
                    let weight = w3.x * w3.y * w3.z;
                    weighted += weight * sample.x;
                    weightSum += weight;
                }
            }
        }
    }
    return select(0.0, weighted / weightSum, weightSum > 1.0e-6);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let ticket = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    let available = min(min(atomicLoad(&state[0]), atomicLoad(&state[4])), reseedWorkBudget());
    if (ticket >= available) {
        return;
    }
    let listCapacity = atomicLoad(&state[2]);
    let donor = atomicLoad(&state[${RESEED_HEADER_WORDS}u + listCapacity + ticket]);
    if (donor >= p.counts.x || !fluidParticleIsFree(donor)) {
        return;
    }
    let cell = atomicLoad(&state[${RESEED_HEADER_WORDS}u + ticket]);
    if (cellTypes[cell] != CELL_FLUID) {
        fluidActivateParticle(donor);
        return;
    }
    let c = faceCoord(cell, gridDim(p));
    let generation = atomicLoad(&state[3]);
    let seed = hash(ticket ^ (cell * 0x9e3779b9u) ^ (generation * 0x85ebca6bu));
    let jitter = vec3<f32>(
        f32(hash(seed ^ 0x68bc21ebu) & 0xffffu) / 65535.0,
        f32(hash(seed ^ 0x02e5be93u) & 0xffffu) / 65535.0,
        f32(hash(seed ^ 0x967a889bu) & 0xffffu) / 65535.0) - 0.5;
    var position = p.originDx.xyz + (vec3<f32>(c) + 0.5 + 0.7 * jitter) * p.originDx.w;
    let radius = p.solve.z;
    let distance = sceneSdf(position, 0.0);
    if (distance < radius) {
        let normal = safeNormal(sceneNormal(position, 0.0), vec3<f32>(0.0, 1.0, 0.0));
        position += (radius - distance) * normal;
        position = clamp(position, p.boundsMin.xyz + radius, p.boundsMax.xyz - radius);
        if (sceneSdf(position, 0.0) < radius) {
            fluidActivateParticle(donor);
            return;
        }
    }
    if (!fluidActivateParticle(donor)) {
        return;
    }
    let velocity = vec3<f32>(
        sampleComponent(position, FACE_U),
        sampleComponent(position, FACE_V),
        sampleComponent(position, FACE_W));
    positions[donor] = vec4<f32>(position, 1.0);
    velocities[donor] = vec4<f32>(velocity, 0.0);
}
`;
}

const SHEETING_BUILD_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> cellTypes: array<u32>;
@group(0) @binding(2) var<storage, read> liquidSdf: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p: Params;

fn sheetingWorkBudget() -> u32 {
    let listCapacity = atomicLoad(&state[3]);
    let base = max(${SHEETING_MIN_WORK_BUDGET}u, p.counts.x / ${SHEETING_WORK_BUDGET_DIVISOR}u);
    return min(listCapacity, max(1u, u32(ceil(f32(base) * clamp(p.sheeting.y, 0.0, 1.0)))));
}

fn isAir(c: vec3<i32>) -> bool {
    return inCellGrid(c, p) && cellTypes[cellIndex(c, p)] == CELL_AIR;
}

fn isThinSheet(c: vec3<i32>) -> bool {
    return
        (isAir(c - vec3<i32>(1, 0, 0)) && isAir(c + vec3<i32>(1, 0, 0))) ||
        (isAir(c - vec3<i32>(0, 1, 0)) && isAir(c + vec3<i32>(0, 1, 0))) ||
        (isAir(c - vec3<i32>(0, 0, 1)) && isAir(c + vec3<i32>(0, 0, 1)));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i == 0u) {
        atomicAdd(&state[2], 1u);
    }
    if (i >= arrayLength(&cellTypes) || cellTypes[i] != CELL_FLUID || liquidSdf[i] >= 0.0) {
        return;
    }
    let c = faceCoord(i, gridDim(p));
    if (!isThinSheet(c)) {
        return;
    }
    let current = atomicLoad(&cellMarks[i]);
    let targetCount = max(1u, u32(ceil(f32(p.counts.y) * clamp(p.sheeting.y, 0.0, 1.0))));
    if (current >= targetCount) {
        return;
    }
    let deficit = targetCount - current;
    let capacity = sheetingWorkBudget();
    for (var marker = 0u; marker < deficit; marker = marker + 1u) {
        let ticket = atomicAdd(&state[0], 1u);
        if (ticket < capacity) {
            atomicStore(&state[${SHEETING_HEADER_WORDS}u + ticket], i);
        }
    }
}
`;

function buildSheetingEmitWgsl(scene: SceneSdfSpec | null): string {
    const sceneDecl = scene
        ? `${scene.struct}
@group(0) @binding(7) var<uniform> sceneSdfParams: SceneSdfParams;
${sceneSdfGridBindingWgsl(scene, 8)}
${scene.sdf}
${SCENE_NORMAL_WGSL}`
        : `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return 1.0e30; }
fn sceneNormal(pt: vec3<f32>, dt: f32) -> vec3<f32> { return vec3<f32>(0.0, 1.0, 0.0); }`;
    return /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
${sceneDecl}
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> faceVelocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> liquidSdf: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p: Params;
@group(0) @binding(6) var<storage, read_write> lifecycle: FluidLifecycle;

fn hash(value: u32) -> u32 {
    var result = value;
    result ^= result >> 16u;
    result *= 0x7feb352du;
    result ^= result >> 15u;
    result *= 0x846ca68bu;
    result ^= result >> 16u;
    return result;
}

fn sdfAt(c: vec3<i32>) -> f32 {
    return liquidSdf[cellIndex(clamp(c, vec3<i32>(0), gridDim(p) - vec3<i32>(1)), p)];
}

fn sampleComponent(world: vec3<f32>, kind: u32) -> f32 {
    let d = faceGridDim(kind, p);
    let grid = (world - p.originDx.xyz) / p.originDx.w - faceOffset(kind);
    let base = vec3<i32>(floor(grid));
    let f = grid - floor(grid);
    var weighted = 0.0;
    var weightSum = 0.0;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (any(c < vec3<i32>(0)) || any(c >= d)) {
                    continue;
                }
                let q = vec3<f32>(f32(x), f32(y), f32(z));
                let w3 = select(vec3<f32>(1.0) - f, f, q > vec3<f32>(0.5));
                let sample = faceVelocity[globalFaceIndex(kind, c, p)];
                if (sample.y > 0.5) {
                    let weight = w3.x * w3.y * w3.z;
                    weighted += weight * sample.x;
                    weightSum += weight;
                }
            }
        }
    }
    return select(0.0, weighted / weightSum, weightSum > 1.0e-6);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsFree(i)) {
        return;
    }
    let requestCount = min(atomicLoad(&state[0]), atomicLoad(&state[3]));
    let ticket = atomicAdd(&state[1], 1u);
    if (ticket >= requestCount) {
        return;
    }
    let cell = atomicLoad(&state[${SHEETING_HEADER_WORDS}u + ticket]);
    let c = faceCoord(cell, gridDim(p));
    if (liquidSdf[cell] >= 0.0) {
        return;
    }
    let generation = atomicLoad(&state[2]);
    let seed = hash(ticket ^ (cell * 0x9e3779b9u) ^ (generation * 0x85ebca6bu));
    let random = vec2<f32>(
        f32(hash(seed ^ 0x68bc21ebu) & 0xffffu) / 65535.0,
        f32(hash(seed ^ 0x02e5be93u) & 0xffffu) / 65535.0) - 0.5;
    let gradient = vec3<f32>(
        sdfAt(c + vec3<i32>(1, 0, 0)) - sdfAt(c - vec3<i32>(1, 0, 0)),
        sdfAt(c + vec3<i32>(0, 1, 0)) - sdfAt(c - vec3<i32>(0, 1, 0)),
        sdfAt(c + vec3<i32>(0, 0, 1)) - sdfAt(c - vec3<i32>(0, 0, 1)));
    let normal = safeNormal(gradient, vec3<f32>(0.0, 1.0, 0.0));
    let tangent = safeNormal(cross(normal, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.9)), vec3<f32>(1.0, 0.0, 0.0));
    let bitangent = cross(normal, tangent);
    var position =
        p.originDx.xyz +
        (vec3<f32>(c) + vec3<f32>(0.5)) * p.originDx.w +
        (tangent * random.x + bitangent * random.y) * (0.55 * p.originDx.w) -
        normal * (0.1 * p.originDx.w);
    let radius = p.solve.z;
    position = clamp(position, p.boundsMin.xyz + radius, p.boundsMax.xyz - radius);
    let distance = sceneSdf(position, 0.0);
    if (distance < radius) {
        position += (radius - distance) * safeNormal(sceneNormal(position, 0.0), vec3<f32>(0.0, 1.0, 0.0));
        position = clamp(position, p.boundsMin.xyz + radius, p.boundsMax.xyz - radius);
        if (sceneSdf(position, 0.0) < radius) {
            return;
        }
    }
    if (!fluidActivateParticle(i)) {
        return;
    }
    let velocity = vec3<f32>(
        sampleComponent(position, FACE_U),
        sampleComponent(position, FACE_V),
        sampleComponent(position, FACE_W));
    positions[i] = vec4<f32>(position, 1.0);
    velocities[i] = vec4<f32>(velocity, 0.0);
}
`;
}

const SURFACE_NET_DOMAIN_BOUNDARY_WGSL = /* wgsl */ `
fn closeLateralDomainBoundary(i: u32, value: f32, p: Params) -> f32 {
    let c = faceCoord(i, gridDim(p));
    let maximum = gridDim(p) - vec3<i32>(1);
    let lateralBoundary = c.x == 0 || c.x == maximum.x || c.z == 0 || c.z == maximum.z;
    return select(value, max(value, 0.05 * p.originDx.w), lateralBoundary);
}
`;

const SURFACE_NET_STABILIZE_WGSL = /* wgsl */ `
${COMMON_WGSL}
${SURFACE_NET_DOMAIN_BOUNDARY_WGSL}
@group(0) @binding(0) var<storage, read> currentSdf: array<f32>;
@group(0) @binding(1) var<storage, read_write> stabilizedSdf: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&stabilizedSdf)) {
        return;
    }
    let current = currentSdf[i];
    let previous = stabilizedSdf[i];
    let nearInterface = min(abs(current), abs(previous)) < 1.5 * p.originDx.w;
    let coherentMotion = abs(current - previous) < 1.5 * p.originDx.w;
    let stabilized = select(current, mix(previous, current, 0.65), nearInterface && coherentMotion);
    stabilizedSdf[i] = closeLateralDomainBoundary(i, stabilized, p);
}
`;

const SURFACE_NET_COPY_WGSL = /* wgsl */ `
${COMMON_WGSL}
${SURFACE_NET_DOMAIN_BOUNDARY_WGSL}
@group(0) @binding(0) var<storage, read> currentSdf: array<f32>;
@group(0) @binding(1) var<storage, read_write> stabilizedSdf: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i < arrayLength(&stabilizedSdf)) {
        stabilizedSdf[i] = closeLateralDomainBoundary(i, currentSdf[i], p);
    }
}`;

const SURFACE_NET_RESET_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> drawArgs: array<u32>;
@group(0) @binding(1) var<storage, read_write> wireframeDrawArgs: array<u32>;
@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 5u; i = i + 1u) {
        drawArgs[i] = 0u;
        wireframeDrawArgs[i] = 0u;
    }
}`;

const SURFACE_NET_VERTEX_WGSL = /* wgsl */ `
${COMMON_WGSL}
struct SurfaceVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> liquidSdf: array<f32>;
@group(0) @binding(1) var<storage, read_write> vertices: array<SurfaceVertex>;
@group(0) @binding(2) var<uniform> p: Params;

fn cubeDim() -> vec3<i32> {
    return gridDim(p) - vec3<i32>(1);
}

fn sdfAt(c: vec3<i32>) -> f32 {
    return liquidSdf[cellIndex(clamp(c, vec3<i32>(0), gridDim(p) - vec3<i32>(1)), p)];
}

fn gradientAt(c: vec3<i32>) -> vec3<f32> {
    let fine = vec3<f32>(
        sdfAt(c + vec3<i32>(1, 0, 0)) - sdfAt(c - vec3<i32>(1, 0, 0)),
        sdfAt(c + vec3<i32>(0, 1, 0)) - sdfAt(c - vec3<i32>(0, 1, 0)),
        sdfAt(c + vec3<i32>(0, 0, 1)) - sdfAt(c - vec3<i32>(0, 0, 1)));
    let coarse = 0.5 * vec3<f32>(
        sdfAt(c + vec3<i32>(2, 0, 0)) - sdfAt(c - vec3<i32>(2, 0, 0)),
        sdfAt(c + vec3<i32>(0, 2, 0)) - sdfAt(c - vec3<i32>(0, 2, 0)),
        sdfAt(c + vec3<i32>(0, 0, 2)) - sdfAt(c - vec3<i32>(0, 0, 2)));
    return mix(fine, coarse, 0.65);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&vertices)) {
        return;
    }
    let c = faceCoord(i, cubeDim());
    let corners = array<vec3<i32>, 8>(
        vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
        vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1));
    let edges = array<vec2<u32>, 12>(
        vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(4u, 5u), vec2<u32>(6u, 7u),
        vec2<u32>(0u, 2u), vec2<u32>(1u, 3u), vec2<u32>(4u, 6u), vec2<u32>(5u, 7u),
        vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u));
    var positionSum = vec3<f32>(0.0);
    var normalSum = vec3<f32>(0.0);
    var crossingCount = 0u;
    for (var edge = 0u; edge < 12u; edge = edge + 1u) {
        let ca = c + corners[edges[edge].x];
        let cb = c + corners[edges[edge].y];
        let a = sdfAt(ca);
        let b = sdfAt(cb);
        if ((a < 0.0) == (b < 0.0)) {
            continue;
        }
        let t = clamp(a / (a - b), 0.0, 1.0);
        let pa = p.originDx.xyz + (vec3<f32>(ca) + vec3<f32>(0.5)) * p.originDx.w;
        let pb = p.originDx.xyz + (vec3<f32>(cb) + vec3<f32>(0.5)) * p.originDx.w;
        positionSum += mix(pa, pb, t);
        normalSum += mix(gradientAt(ca), gradientAt(cb), t);
        crossingCount += 1u;
    }
    if (crossingCount == 0u) {
        vertices[i].position = vec4<f32>(0.0);
        vertices[i].normal = vec4<f32>(0.0);
        return;
    }
    var position = positionSum / f32(crossingCount);
    var lowXNegative = false;
    var highXNegative = false;
    var lowZNegative = false;
    var highZNegative = false;
    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        let negative = sdfAt(c + corners[corner]) < 0.0;
        lowXNegative = lowXNegative || (negative && corners[corner].x == 0);
        highXNegative = highXNegative || (negative && corners[corner].x == 1);
        lowZNegative = lowZNegative || (negative && corners[corner].z == 0);
        highZNegative = highZNegative || (negative && corners[corner].z == 1);
    }
    let cubes = cubeDim();
    if (c.x == 0 && !lowXNegative && highXNegative) {
        position.x = p.originDx.x;
    } else if (c.x == cubes.x - 1 && lowXNegative && !highXNegative) {
        position.x = p.boundsMax.x;
    }
    if (c.z == 0 && !lowZNegative && highZNegative) {
        position.z = p.originDx.z;
    } else if (c.z == cubes.z - 1 && lowZNegative && !highZNegative) {
        position.z = p.boundsMax.z;
    }
    vertices[i].position = vec4<f32>(position, 1.0);
    vertices[i].normal = vec4<f32>(safeNormal(normalSum, vec3<f32>(0.0, 1.0, 0.0)), 1.0);
}
`;

const SURFACE_NET_INDEX_WGSL = /* wgsl */ `
${COMMON_WGSL}
struct SurfaceVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> liquidSdf: array<f32>;
@group(0) @binding(1) var<storage, read> vertices: array<SurfaceVertex>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> drawArgs: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> wireframeIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> wireframeDrawArgs: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> p: Params;

fn cubeDim() -> vec3<i32> {
    return gridDim(p) - vec3<i32>(1);
}

fn sdfAt(c: vec3<i32>) -> f32 {
    return liquidSdf[cellIndex(c, p)];
}

fn cubeIndex(c: vec3<i32>) -> u32 {
    let d = cubeDim();
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn emitQuad(a: vec3<i32>, b: vec3<i32>, c: vec3<i32>, d: vec3<i32>, reverse: bool) {
    let ia = cubeIndex(a);
    let ib = cubeIndex(b);
    let ic = cubeIndex(c);
    let id = cubeIndex(d);
    if (vertices[ia].normal.w < 0.5 || vertices[ib].normal.w < 0.5 || vertices[ic].normal.w < 0.5 || vertices[id].normal.w < 0.5) {
        return;
    }
    let offset = atomicAdd(&drawArgs[0], 6u);
    if (offset + 6u > arrayLength(&indices)) {
        return;
    }
    let wireOffset = atomicAdd(&wireframeDrawArgs[0], 8u);
    if (wireOffset + 8u <= arrayLength(&wireframeIndices)) {
        wireframeIndices[wireOffset] = ia;
        wireframeIndices[wireOffset + 1u] = ib;
        wireframeIndices[wireOffset + 2u] = ib;
        wireframeIndices[wireOffset + 3u] = ic;
        wireframeIndices[wireOffset + 4u] = ic;
        wireframeIndices[wireOffset + 5u] = id;
        wireframeIndices[wireOffset + 6u] = id;
        wireframeIndices[wireOffset + 7u] = ia;
    }
    if (reverse) {
        indices[offset] = ia;
        indices[offset + 1u] = ic;
        indices[offset + 2u] = ib;
        indices[offset + 3u] = ia;
        indices[offset + 4u] = id;
        indices[offset + 5u] = ic;
    } else {
        indices[offset] = ia;
        indices[offset + 1u] = ib;
        indices[offset + 2u] = ic;
        indices[offset + 3u] = ia;
        indices[offset + 4u] = ic;
        indices[offset + 5u] = id;
    }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&liquidSdf)) {
        return;
    }
    let c = faceCoord(i, gridDim(p));
    let phi = liquidSdf[i];
    let d = gridDim(p);
    if (c.x + 1 < d.x && c.y > 0 && c.y + 1 < d.y && c.z > 0 && c.z + 1 < d.z) {
        let next = sdfAt(c + vec3<i32>(1, 0, 0));
        if ((phi < 0.0) != (next < 0.0)) {
            emitQuad(
                vec3<i32>(c.x, c.y - 1, c.z - 1), vec3<i32>(c.x, c.y, c.z - 1),
                vec3<i32>(c.x, c.y, c.z), vec3<i32>(c.x, c.y - 1, c.z), phi >= 0.0);
        }
    }
    if (c.y + 1 < d.y && c.x > 0 && c.x + 1 < d.x && c.z > 0 && c.z + 1 < d.z) {
        let next = sdfAt(c + vec3<i32>(0, 1, 0));
        if ((phi < 0.0) != (next < 0.0)) {
            emitQuad(
                vec3<i32>(c.x - 1, c.y, c.z - 1), vec3<i32>(c.x - 1, c.y, c.z),
                vec3<i32>(c.x, c.y, c.z), vec3<i32>(c.x, c.y, c.z - 1), phi >= 0.0);
        }
    }
    if (c.z + 1 < d.z && c.x > 0 && c.x + 1 < d.x && c.y > 0 && c.y + 1 < d.y) {
        let next = sdfAt(c + vec3<i32>(0, 0, 1));
        if ((phi < 0.0) != (next < 0.0)) {
            emitQuad(
                vec3<i32>(c.x - 1, c.y - 1, c.z), vec3<i32>(c.x, c.y - 1, c.z),
                vec3<i32>(c.x, c.y, c.z), vec3<i32>(c.x - 1, c.y, c.z), phi >= 0.0);
        }
    }
}
`;

const SURFACE_NET_FINALIZE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> drawArgs: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> wireframeDrawArgs: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> wireframeIndices: array<u32>;
@compute @workgroup_size(1)
fn main() {
    atomicStore(&drawArgs[0], min(atomicLoad(&drawArgs[0]), arrayLength(&indices)));
    atomicStore(&drawArgs[1], 1u);
    atomicStore(&drawArgs[2], 0u);
    atomicStore(&drawArgs[3], 0u);
    atomicStore(&drawArgs[4], 0u);
    atomicStore(&wireframeDrawArgs[0], min(atomicLoad(&wireframeDrawArgs[0]), arrayLength(&wireframeIndices)));
    atomicStore(&wireframeDrawArgs[1], 1u);
    atomicStore(&wireframeDrawArgs[2], 0u);
    atomicStore(&wireframeDrawArgs[3], 0u);
    atomicStore(&wireframeDrawArgs[4], 0u);
}
`;

const SPEED_REDUCE_WGSL = /* wgsl */ `
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> debugSpeed: array<f32>;
@group(0) @binding(2) var<storage, read_write> maxSpeedBits: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> lifecycle: FluidLifecycle;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= lifecycle.capacity || !fluidParticleIsActive(i)) {
        return;
    }
    let speed = length(velocities[i].xyz);
    debugSpeed[i] = speed;
    atomicMax(&maxSpeedBits[0], bitcast<u32>(speed));
}`;

const SPEED_RESET_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> maxSpeedBits: array<atomic<u32>>;
@compute @workgroup_size(1)
fn main() {
    atomicStore(&maxSpeedBits[0], 0u);
}`;

const FOAM_ACTIVE_OVERFLOW_GATE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> pageData: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> computeArgs: array<u32>;
@group(0) @binding(2) var<storage, read_write> statusData: array<atomic<u32>>;
@compute @workgroup_size(1)
fn main() {
    if (atomicLoad(&pageData[1]) != 0u || atomicLoad(&statusData[2]) != 0u) {
        computeArgs[0] = 0u;
        computeArgs[1] = 1u;
        computeArgs[2] = 1u;
    }
}`;

const FLOW_DELETE_WGSL = /* wgsl */ `
${COMMON_WGSL}
${FLUID_FLOW_STRUCT_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> flow: FluidFlowData;
@group(0) @binding(3) var<storage, read_write> flowCounters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> lifecycle: FluidLifecycle;
${FLUID_FLOW_RUNTIME_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= flow.header.z || !fluidParticleIsActive(i)) {
        return;
    }
    let launch = fluidTryRelaunch(positions[i].xyz, i);
    if (launch.launched != 0u) {
        positions[i] = vec4<f32>(launch.position, 1.0);
        velocities[i] = vec4<f32>(launch.velocity, 0.0);
        return;
    }
    if (fluidTryDelete(positions[i].xyz, i) && fluidDeleteParticle(i)) {
        positions[i] = vec4<f32>(0.0, -1.0e5, 0.0, 1.0);
        velocities[i] = vec4<f32>(0.0);
    }
}`;

const FLOW_MARK_OCCUPANCY_WGSL = /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read_write> lifecycle: FluidLifecycle;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsActive(i)) {
        return;
    }
    let cell = vec3<i32>(floor((positions[i].xyz - p.originDx.xyz) / p.originDx.w));
    if (storageCellExists(cell, p)) {
        atomicAdd(&cellMarks[cellIndex(cell, p)], 1u);
    }
}`;

const FLOW_FILL_EMPTY_SPACE_WGSL = /* wgsl */ `
const FLOW_FILL_ATTEMPTS: u32 = 8u;
const FLOW_INVALID_CELL: u32 = 0xffffffffu;

fn fluidEmitterHasBudget(index: u32) -> bool {
    let claimed = atomicLoad(&flowCounters[index * 2u]);
    let budget = atomicLoad(&flowCounters[index * 2u + 1u]);
    return budget == 0xffffffffu || claimed < budget;
}

fn fluidChooseAvailableEmitter(seed: u32) -> u32 {
    var count = 0u;
    for (var i = 0u; i < flow.header.x; i = i + 1u) {
        let emitter = flow.emitters[i];
        if (emitter.flags.x != 0u && emitter.flags.y != 0u && fluidEmitterHasBudget(i)) {
            count = count + 1u;
        }
    }
    if (count == 0u) {
        return flow.header.x;
    }
    let wanted = fluidHash(seed) % count;
    var seen = 0u;
    for (var i = 0u; i < flow.header.x; i = i + 1u) {
        let emitter = flow.emitters[i];
        if (emitter.flags.x != 0u && emitter.flags.y != 0u && fluidEmitterHasBudget(i)) {
            if (seen == wanted) {
                return i;
            }
            seen = seen + 1u;
        }
    }
    return flow.header.x;
}

fn fluidClaimMarkerCell(world: vec3<f32>) -> u32 {
    let cell = vec3<i32>(floor((world - p.originDx.xyz) / p.originDx.w));
    if (!storageCellExists(cell, p)) {
        return FLOW_INVALID_CELL;
    }
    let index = cellIndex(cell, p);
    let markerTarget = max(1u, p.counts.y);
    loop {
        let current = atomicLoad(&cellMarks[index]);
        if (current >= markerTarget) {
            return FLOW_INVALID_CELL;
        }
        if (atomicCompareExchangeWeak(&cellMarks[index], current, current + 1u).exchanged) {
            return index;
        }
    }
}

fn fluidTryFillEmptySpace(seed: u32) -> FluidLaunch {
    for (var attempt = 0u; attempt < FLOW_FILL_ATTEMPTS; attempt = attempt + 1u) {
        let attemptSeed = seed ^ ((attempt + 1u) * 2246822519u);
        let emitterIndex = fluidChooseAvailableEmitter(attemptSeed);
        if (emitterIndex >= flow.header.x) {
            return FluidLaunch(vec3<f32>(0.0), vec3<f32>(0.0), 0u);
        }
        let launch = fluidPerParticleLaunch(flow.emitters[emitterIndex], attemptSeed);
        let markerCell = fluidClaimMarkerCell(launch.position);
        if (markerCell == FLOW_INVALID_CELL) {
            continue;
        }
        if (fluidClaimEmitter(emitterIndex)) {
            return launch;
        }
        atomicSub(&cellMarks[markerCell], 1u);
    }
    return FluidLaunch(vec3<f32>(0.0), vec3<f32>(0.0), 0u);
}
`;

const FLOW_EMIT_WGSL = /* wgsl */ `
${COMMON_WGSL}
${FLUID_FLOW_STRUCT_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> flow: FluidFlowData;
@group(0) @binding(3) var<storage, read_write> flowCounters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> lifecycle: FluidLifecycle;
@group(0) @binding(5) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> p: Params;
${FLUID_FLOW_RUNTIME_WGSL}
${FLOW_FILL_EMPTY_SPACE_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= flow.header.z || !fluidParticleIsFree(i)) {
        return;
    }
    let launch = fluidTryFillEmptySpace(flow.header.w * 2654435761u + i);
    if (launch.launched != 0u && fluidActivateParticle(i)) {
        positions[i] = vec4<f32>(launch.position, 1.0);
        velocities[i] = vec4<f32>(launch.velocity, 0.0);
    }
}`;

const FLOW_EMIT_APPEND_WGSL = /* wgsl */ `
${COMMON_WGSL}
${FLUID_FLOW_STRUCT_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
struct EmitRange { base: u32, count: u32, reserved0: u32, reserved1: u32, };
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> flow: FluidFlowData;
@group(0) @binding(3) var<storage, read_write> flowCounters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> lifecycle: FluidLifecycle;
@group(0) @binding(5) var<uniform> emitRange: EmitRange;
@group(0) @binding(6) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p: Params;
${FLUID_FLOW_RUNTIME_WGSL}
${FLOW_FILL_EMPTY_SPACE_WGSL}
fn fluidAllocateAppendIndex() -> u32 {
    loop {
        let current = atomicLoad(&lifecycle.activeCount);
        if (current >= lifecycle.capacity) {
            return lifecycle.capacity;
        }
        if (atomicCompareExchangeWeak(&lifecycle.activeCount, current, current + 1u).exchanged) {
            return current;
        }
    }
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let ticket = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (ticket >= emitRange.count) {
        return;
    }
    let launch = fluidTryFillEmptySpace(flow.header.w * 2654435761u + ticket);
    if (launch.launched == 0u) {
        return;
    }
    let i = fluidAllocateAppendIndex();
    if (i >= flow.header.z) {
        let cell = vec3<i32>(floor((launch.position - p.originDx.xyz) / p.originDx.w));
        if (storageCellExists(cell, p)) {
            atomicSub(&cellMarks[cellIndex(cell, p)], 1u);
        }
        return;
    }
    atomicStore(&lifecycle.states[i], FLUID_PARTICLE_ACTIVE);
    positions[i] = vec4<f32>(launch.position, 1.0);
    velocities[i] = vec4<f32>(launch.velocity, 0.0);
}`;

function buildFoamEmitWgsl(activeParticles: boolean): string {
    const headDecl = activeParticles
        ? "@group(0) @binding(8) var<storage, read_write> activeState: array<atomic<u32>>;"
        : "@group(0) @binding(8) var<storage, read_write> head: array<atomic<u32>>;";
    const activeDecl = activeParticles
        ? `
fn activeStride(cap: u32) -> u32 { return ((cap + 63u) / 64u) * 64u; }
fn activeListBase(side: u32, cap: u32) -> u32 { return 64u + side * activeStride(cap); }
fn activeFlagBase(cap: u32) -> u32 { return 64u + 2u * activeStride(cap); }`
        : "";
    const activate = activeParticles
        ? `
        if (atomicExchange(&activeState[activeFlagBase(cap) + slot], 1u) == 0u) {
            let side = atomicLoad(&activeState[3]);
            let dst = atomicAdd(&activeState[1u + side], 1u);
            atomicStore(&activeState[activeListBase(side, cap) + dst], slot);
        }`
        : "";
    return /* wgsl */ `
${COMMON_WGSL}
${FOAM_COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> faceVelocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> curvature: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;
@group(0) @binding(6) var<uniform> foam: Foam;
@group(0) @binding(7) var<storage, read_write> diffuse: array<Diffuse>;
${headDecl}
@group(0) @binding(9) var<storage, read_write> lifecycle: FluidLifecycle;
${activeDecl}

fn sampleSurface(world: vec3<f32>) -> vec4<f32> {
    let grid = (world - p.originDx.xyz) / p.originDx.w - vec3<f32>(0.5);
    let base = vec3<i32>(floor(grid));
    let fraction = grid - floor(grid);
    var surface = vec4<f32>(0.0);
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (!storageCellExists(c, p)) {
                    continue;
                }
                let q = vec3<f32>(f32(x), f32(y), f32(z));
                let w3 = select(vec3<f32>(1.0) - fraction, fraction, q > vec3<f32>(0.5));
                surface += normals[cellIndex(c, p)] * (w3.x * w3.y * w3.z);
            }
        }
    }
    return surface;
}

fn sampleCurvature(world: vec3<f32>) -> f32 {
    let grid = (world - p.originDx.xyz) / p.originDx.w - vec3<f32>(0.5);
    let base = vec3<i32>(floor(grid));
    let fraction = grid - floor(grid);
    var value = 0.0;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (!storageCellExists(c, p)) {
                    continue;
                }
                let q = vec3<f32>(f32(x), f32(y), f32(z));
                let w3 = select(vec3<f32>(1.0) - fraction, fraction, q > vec3<f32>(0.5));
                value += curvature[cellIndex(c, p)] * (w3.x * w3.y * w3.z);
            }
        }
    }
    return value;
}

fn faceValue(kind: u32, c: vec3<i32>) -> f32 {
    let d = faceGridDim(kind, p);
    if (any(c < vec3<i32>(0)) || any(c >= d) || !storageCellExists(c, p)) {
        return 0.0;
    }
    let sample = faceVelocity[globalFaceIndex(kind, c, p)];
    return select(0.0, sample.x, sample.y > 0.5);
}

fn cellVelocity(c: vec3<i32>) -> vec3<f32> {
    if (!storageCellExists(c, p)) {
        return vec3<f32>(0.0);
    }
    return 0.5 * vec3<f32>(
        faceValue(FACE_U, c) + faceValue(FACE_U, c + vec3<i32>(1, 0, 0)),
        faceValue(FACE_V, c) + faceValue(FACE_V, c + vec3<i32>(0, 1, 0)),
        faceValue(FACE_W, c) + faceValue(FACE_W, c + vec3<i32>(0, 0, 1))
    );
}

fn sampleTurbulence(world: vec3<f32>) -> f32 {
    let c = clamp(vec3<i32>(floor((world - p.originDx.xyz) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
    let invTwoDx = 0.5 / p.originDx.w;
    let dVdx = (cellVelocity(c + vec3<i32>(1, 0, 0)) - cellVelocity(c - vec3<i32>(1, 0, 0))) * invTwoDx;
    let dVdy = (cellVelocity(c + vec3<i32>(0, 1, 0)) - cellVelocity(c - vec3<i32>(0, 1, 0))) * invTwoDx;
    let dVdz = (cellVelocity(c + vec3<i32>(0, 0, 1)) - cellVelocity(c - vec3<i32>(0, 0, 1))) * invTwoDx;
    let curl = vec3<f32>(dVdy.z - dVdz.y, dVdz.x - dVdx.z, dVdx.y - dVdy.x);
    let sxy = 0.5 * (dVdx.y + dVdy.x);
    let sxz = 0.5 * (dVdx.z + dVdz.x);
    let syz = 0.5 * (dVdy.z + dVdz.y);
    let strainSq = dVdx.x * dVdx.x + dVdy.y * dVdy.y + dVdz.z * dVdz.z + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
    return p.originDx.w * sqrt(max(0.0, dot(curl, curl) + 2.0 * strainSq));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsActive(i)) {
        return;
    }
    let pi = positions[i].xyz;
    let vi = velocities[i].xyz;
    let speed = length(vi);
    if (speed < 1.0e-4) {
        return;
    }
    let surface = sampleSurface(pi);
    let surfaceStrength = surface.w * p.originDx.w;
    if (surfaceStrength < 0.05) {
        return;
    }
    let outward = -safeNormal(surface.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let topWeight = smoothstep(0.15, 0.65, outward.y);
    if (topWeight <= 0.0) {
        return;
    }
    let normalSpeed = dot(vi, outward);
    let trappedAir = max(0.0, -normalSpeed);
    let waveCrest = max(0.0, sampleCurvature(pi) * p.originDx.w) * max(0.0, normalSpeed);
    let ita = phi(trappedAir, foam.tauTaMin, foam.tauTaMax);
    let iwc = phi(waveCrest, foam.curvatureMin, foam.curvatureMax);
    var iturb = 0.0;
    if (foam.kTurb > 0.0) {
        iturb = phi(sampleTurbulence(pi), foam.turbulenceMin, foam.turbulenceMax);
    }
    let ik = phi(speed, foam.energySpeedMin, foam.energySpeedMax);
    let expected = topWeight * ik * (foam.kTa * ita + foam.kWc * iwc + foam.kTurb * iturb) * foam.frameDt;
    let whole = floor(expected);
    var spawnCount = i32(whole) + select(0, 1, fRnd((i * 2246822519u) ^ (foam.frameSeed * 22695477u)) < expected - whole);
    spawnCount = min(spawnCount, 8);
    if (spawnCount <= 0) {
        return;
    }
    let potential = max(ita, max(iwc, iturb));
    let life = mix(foam.tMin, foam.tMax, potential);
    let axis = vi / speed;
    var tangent = cross(axis, vec3<f32>(0.0, 1.0, 0.0));
    if (dot(tangent, tangent) <= 1.0e-6) {
        tangent = cross(axis, vec3<f32>(1.0, 0.0, 0.0));
    }
    tangent = normalize(tangent);
    let bitangent = cross(axis, tangent);
    let cap = arrayLength(&diffuse);
    let travel = speed * foam.frameDt;
    for (var sample = 0; sample < spawnCount; sample = sample + 1) {
        let seed = (i * 2654435761u) ^ (foam.frameSeed * 40503u) ^ (u32(sample) * 2246822519u);
        let radius = foam.rv * sqrt(fRnd(seed));
        let angle = 6.28318530718 * fRnd(seed * 3u + 1u);
        let offset = tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle));
        let xd = pi + offset + axis * (fRnd(seed * 7u + 5u) * travel);
        let slot = atomicAdd(&${activeParticles ? "activeState" : "head"}[0], 1u) % cap;
        diffuse[slot].p = vec4<f32>(xd, life);
        diffuse[slot].v = vec4<f32>(vi + offset, 1.0);
${activate}
    }
}`;
}

function buildFoamUpdateWgsl(activeParticles: boolean): string {
    const activeDecl = activeParticles
        ? `
@group(0) @binding(6) var<storage, read_write> activeState: array<atomic<u32>>;
fn activeStride(cap: u32) -> u32 { return ((cap + 63u) / 64u) * 64u; }
fn activeListBase(side: u32, cap: u32) -> u32 { return 64u + side * activeStride(cap); }
fn activeFlagBase(cap: u32) -> u32 { return 64u + 2u * activeStride(cap); }`
        : "";
    const slotLookup = activeParticles
        ? `
    let cap = arrayLength(&diffuse);
    let side = atomicLoad(&activeState[3]);
    let item = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (item >= atomicLoad(&activeState[1u + side])) { return; }
    let i = atomicLoad(&activeState[activeListBase(side, cap) + item]);`
        : `
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&diffuse)) { return; }`;
    const deactivate = activeParticles ? "atomicStore(&activeState[activeFlagBase(cap) + i], 0u);" : "";
    const keepActive = activeParticles
        ? `
    let nextSide = 1u - side;
    let dst = atomicAdd(&activeState[1u + nextSide], 1u);
    atomicStore(&activeState[activeListBase(nextSide, cap) + dst], i);`
        : "";
    return /* wgsl */ `
${COMMON_WGSL}
${FOAM_COMMON_WGSL}
@group(0) @binding(0) var<storage, read> faceVelocity: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cellMarks: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<uniform> foam: Foam;
@group(0) @binding(5) var<storage, read_write> diffuse: array<Diffuse>;
${activeDecl}

fn sampleComponent(world: vec3<f32>, kind: u32) -> f32 {
    let d = faceGridDim(kind, p);
    let grid = (world - p.originDx.xyz) / p.originDx.w - faceOffset(kind);
    let base = vec3<i32>(floor(grid));
    let fraction = grid - floor(grid);
    var weighted = 0.0;
    var weightSum = 0.0;
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (any(c < vec3<i32>(0)) || any(c >= d) || !storageCellExists(c, p)) {
                    continue;
                }
                let q = vec3<f32>(f32(x), f32(y), f32(z));
                let w3 = select(vec3<f32>(1.0) - fraction, fraction, q > vec3<f32>(0.5));
                let weight = w3.x * w3.y * w3.z;
                let sample = faceVelocity[globalFaceIndex(kind, c, p)];
                if (sample.y > 0.5) {
                    weighted += weight * sample.x;
                    weightSum += weight;
                }
            }
        }
    }
    return select(0.0, weighted / weightSum, weightSum > 1.0e-6);
}

fn sampleVelocity(world: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(sampleComponent(world, FACE_U), sampleComponent(world, FACE_V), sampleComponent(world, FACE_W));
}

fn sampleSurface(world: vec3<f32>) -> vec4<f32> {
    let grid = (world - p.originDx.xyz) / p.originDx.w - vec3<f32>(0.5);
    let base = vec3<i32>(floor(grid));
    let fraction = grid - floor(grid);
    var surface = vec4<f32>(0.0);
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (!storageCellExists(c, p)) {
                    continue;
                }
                let q = vec3<f32>(f32(x), f32(y), f32(z));
                let w3 = select(vec3<f32>(1.0) - fraction, fraction, q > vec3<f32>(0.5));
                surface += normals[cellIndex(c, p)] * (w3.x * w3.y * w3.z);
            }
        }
    }
    return surface;
}

fn sampleFoamLayer(world: vec3<f32>) -> vec4<f32> {
    var best = sampleSurface(world);
    var bestOutward = -safeNormal(best.xyz, vec3<f32>(0.0, 1.0, 0.0));
    var bestScore = best.w * smoothstep(0.1, 0.45, bestOutward.y);
    let depth = clamp(foam.foamLayerDepth, 0.0, 4.0);
    if (depth <= 0.0) {
        return best;
    }
    let cell = clamp(vec3<i32>(floor((world - p.originDx.xyz) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
    for (var layer = 1; layer <= 4; layer = layer + 1) {
        let distance = f32(layer);
        if (distance > depth + 0.5) {
            continue;
        }
        let candidateCell = cell + vec3<i32>(0, layer, 0);
        if (!storageCellExists(candidateCell, p)) {
            continue;
        }
        var candidate = normals[cellIndex(candidateCell, p)];
        let layerWeight = 1.0 - smoothstep(max(0.0, depth - 0.5), depth + 0.5, distance);
        candidate.w *= layerWeight;
        let outward = -safeNormal(candidate.xyz, vec3<f32>(0.0, 1.0, 0.0));
        let score = candidate.w * smoothstep(0.1, 0.45, outward.y);
        if (score > bestScore) {
            best = candidate;
            bestScore = score;
        }
    }
    return best;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
${slotLookup}
    let current = diffuse[i].p;
    if (current.w <= 0.0) {
        ${deactivate}
        return;
    }
    let position = current.xyz;
    if (any(position < p.boundsMin.xyz) || any(position > p.boundsMax.xyz)) {
        diffuse[i].p = vec4<f32>(position, 0.0);
        ${deactivate}
        return;
    }
    let cell = clamp(vec3<i32>(floor((position - p.originDx.xyz) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
    var occupancy = 0.0;
    if (storageCellExists(cell, p)) {
        occupancy = min(f32(atomicLoad(&cellMarks[cellIndex(cell, p)])) / max(1.0, f32(p.counts.y)), 1.0);
    }
    let surface = sampleFoamLayer(position);
    let surfaceStrength = surface.w * p.originDx.w;
    let outward = -safeNormal(surface.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let fluidVelocity = sampleVelocity(position);
    let dt = foam.frameDt;
    var velocity = diffuse[i].v.xyz;
    let previousKind = u32(clamp(round(diffuse[i].v.w), 0.0, 2.0));
    let enterFoam = surfaceStrength >= 0.05 && outward.y >= 0.45;
    let keepFoam = previousKind == 1u && surfaceStrength >= 0.025 && outward.y >= 0.2;
    var kind = 2u;
    if (enterFoam || keepFoam) {
        kind = 1u;
    } else if (occupancy < 0.2) {
        kind = 0u;
    }
    if (!foamKindEnabled(kind)) {
        diffuse[i].p = vec4<f32>(position, 0.0);
        ${deactivate}
        return;
    }
    var next = position;
    var life = current.w;
    if (kind == 0u) {
        velocity.y -= p.sim.y * dt;
        if (foam.sprayDrag > 0.0) {
            velocity *= exp(-foam.sprayDrag * dt);
        }
        next += velocity * dt;
    } else if (kind == 2u) {
        velocity.y += foam.kb * p.sim.y * dt;
        velocity += clamp(foam.kd, 0.0, 1.0) * (fluidVelocity - velocity);
        next += velocity * dt;
    } else {
        velocity = fluidVelocity;
        next += fluidVelocity * dt;
        life -= dt;
    }
    if (life <= 0.0 || any(next < p.boundsMin.xyz) || any(next > p.boundsMax.xyz)) {
        diffuse[i].p = vec4<f32>(next, 0.0);
        ${deactivate}
        return;
    }
    diffuse[i].p = vec4<f32>(next, life);
    diffuse[i].v = vec4<f32>(velocity, f32(kind));
${keepActive}
}`;
}

function buildForceWgsl(spec: ForceFieldSpec): string {
    return /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
${spec.struct}
${spec.wgsl}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<uniform> forceFieldParams: ForceFieldParams;
@group(0) @binding(4) var<storage, read_write> lifecycle: FluidLifecycle;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsActive(i)) {
        return;
    }
    let velocity = velocities[i].xyz + externalForce(positions[i].xyz, velocities[i].xyz, p.sim.x);
    velocities[i] = vec4<f32>(clampLength(velocity, p.solve.w), 0.0);
}`;
}

const FLIP_WARMUP_WGSL = /* wgsl */ `
${FLUID_LIFECYCLE_STRUCT_WGSL}
struct WarmupParams { start: u32, end: u32, reserved0: u32, reserved1: u32, };
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> seedPositions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> seedVelocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> lifecycle: FluidLifecycle;
@group(0) @binding(5) var<uniform> p: WarmupParams;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = p.start + gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (index >= p.end || index >= lifecycle.capacity) {
        return;
    }
    loop {
        let current = atomicLoad(&lifecycle.states[index]);
        if (current != 2u) {
            return;
        }
        if (atomicCompareExchangeWeak(&lifecycle.states[index], current, 1u).exchanged) {
            positions[index] = seedPositions[index];
            velocities[index] = seedVelocities[index];
            atomicAdd(&lifecycle.activeCount, 1u);
            return;
        }
    }
}`;

function buildPagedFlipAllocationWgsl(layout: FlipPageLayout): string {
    return /* wgsl */ `
const PAGE_SIZE_I: i32 = ${FLIP_PAGE_SIZE};
const BLOCK_DIM: vec3<i32> = vec3<i32>(${layout.blockDim[0]}, ${layout.blockDim[1]}, ${layout.blockDim[2]});
const PAGE_MAP_OFFSET: u32 = 2u;
const PAGE_COORD_OFFSET: u32 = ${2 + layout.numBlocks}u;
const PAGE_LOCK: u32 = 0xffffffffu;

fn blockIndex(block: vec3<i32>) -> u32 {
    return u32(block.x + BLOCK_DIM.x * (block.y + BLOCK_DIM.y * block.z));
}

fn assignPage(block: vec3<i32>) {
    let address = PAGE_MAP_OFFSET + blockIndex(block);
    if (atomicLoad(&pageData[address]) != 0u) {
        return;
    }
    loop {
        let claimed = atomicCompareExchangeWeak(&pageData[address], 0u, PAGE_LOCK);
        if (claimed.exchanged) {
            break;
        }
        if (claimed.old_value != 0u) {
            return;
        }
    }
    let page = atomicAdd(&pageData[0], 1u);
    if (page < config.pageCapacity) {
        atomicStore(&pageData[PAGE_COORD_OFFSET + page], blockIndex(block));
        atomicStore(&pageData[address], page + 1u);
    } else {
        atomicStore(&pageData[1], 1u);
        atomicStore(&pageData[address], config.pageCapacity + 1u);
    }
}`;
}

const FLIP_PAGE_DISPATCH_CONFIG_WGSL = /* wgsl */ `
struct DispatchConfig {
    activeParticleGroups: u32,
    allParticleGroups: u32,
    foamGroups: u32,
    frameId: u32,
    sweepDistance: f32,
    polygonCellItems: u32,
    polygonCubeItems: u32,
    pageCapacity: u32,
};`;

const FLIP_PAGE_SWEPT_ASSIGNMENT_WGSL = /* wgsl */ `
fn assignSweptPoint(world: vec3<f32>) {
    let distance = max(0.0, config.sweepDistance);
    let gridMin = p.originDx.xyz;
    let cellMin = clamp(vec3<i32>(floor((world - vec3<f32>(distance) - gridMin) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
    let cellMax = clamp(vec3<i32>(floor((world + vec3<f32>(distance) - gridMin) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
    let blockMin = max(cellMin / PAGE_SIZE_I - vec3<i32>(1), vec3<i32>(0));
    let blockMax = min(cellMax / PAGE_SIZE_I + vec3<i32>(1), BLOCK_DIM - vec3<i32>(1));
    for (var z = blockMin.z; z <= blockMax.z; z = z + 1) {
        for (var y = blockMin.y; y <= blockMax.y; y = y + 1) {
            for (var x = blockMin.x; x <= blockMax.x; x = x + 1) {
                assignPage(vec3<i32>(x, y, z));
            }
        }
    }
}`;

function buildPagedFlipDiscoveryWgsl(layout: FlipPageLayout): string {
    return /* wgsl */ `
${COMMON_WGSL}
${FLUID_LIFECYCLE_STRUCT_WGSL}
${FLUID_LIFECYCLE_RUNTIME_WGSL}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> pageData: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> lifecycle: FluidLifecycle;
${FLIP_PAGE_DISPATCH_CONFIG_WGSL}
@group(0) @binding(4) var<uniform> config: DispatchConfig;
${buildPagedFlipAllocationWgsl(layout)}
${FLIP_PAGE_SWEPT_ASSIGNMENT_WGSL}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x || !fluidParticleIsActive(i)) {
        return;
    }
    assignSweptPoint(positions[i].xyz);
}`;
}

function buildPagedFlipWarmupDiscoveryWgsl(layout: FlipPageLayout): string {
    return /* wgsl */ `
${COMMON_WGSL}
struct WarmupParams { start: u32, end: u32, reserved0: u32, reserved1: u32, };
@group(0) @binding(0) var<storage, read> seedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> pageData: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> warmup: WarmupParams;
${FLIP_PAGE_DISPATCH_CONFIG_WGSL}
@group(0) @binding(4) var<uniform> config: DispatchConfig;
${buildPagedFlipAllocationWgsl(layout)}
${FLIP_PAGE_SWEPT_ASSIGNMENT_WGSL}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = warmup.start + gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i >= warmup.end || i >= p.counts.x) {
        return;
    }
    assignSweptPoint(seedPositions[i].xyz);
}`;
}

function buildPagedFlipFlowDiscoveryWgsl(layout: FlipPageLayout): string {
    return /* wgsl */ `
${COMMON_WGSL}
${FLUID_FLOW_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read_write> pageData: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<uniform> flow: FluidFlowData;
${FLIP_PAGE_DISPATCH_CONFIG_WGSL}
@group(0) @binding(3) var<uniform> config: DispatchConfig;
${buildPagedFlipAllocationWgsl(layout)}
${FLUID_FLOW_SHAPE_WGSL}

var<workgroup> emitterBlockMin: vec3<i32>;
var<workgroup> emitterBlockMax: vec3<i32>;
var<workgroup> emitterActive: u32;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (wid.x >= flow.header.x) {
        return;
    }
    if (lid.x == 0u) {
        let emitter = flow.emitters[wid.x];
        emitterActive = 0u;
        if (emitter.flags.x != 0u && emitter.flags.y != 0u) {
            let bounds = fluidShapeWorldBounds(emitter.shape);
            let padding = max(0.0, config.sweepDistance) + f32(PAGE_SIZE_I) * p.originDx.w;
            let worldMin = bounds[0] - vec3<f32>(padding);
            let worldMax = bounds[1] + vec3<f32>(padding);
            let gridMin = p.originDx.xyz;
            let gridMax = p.originDx.xyz + vec3<f32>(gridDim(p)) * p.originDx.w;
            if (all(worldMax >= gridMin) && all(worldMin <= gridMax)) {
                let cellMin = clamp(vec3<i32>(floor((worldMin - gridMin) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
                let cellMax = clamp(vec3<i32>(floor((worldMax - gridMin) / p.originDx.w)), vec3<i32>(0), gridDim(p) - vec3<i32>(1));
                emitterBlockMin = cellMin / PAGE_SIZE_I;
                emitterBlockMax = cellMax / PAGE_SIZE_I;
                emitterActive = 1u;
            }
        }
    }
    workgroupBarrier();
    if (emitterActive == 0u) {
        return;
    }
    let span = emitterBlockMax - emitterBlockMin + vec3<i32>(1);
    let total = u32(span.x * span.y * span.z);
    for (var index = lid.x; index < total; index = index + ${WORKGROUP_SIZE}u) {
        let x = i32(index % u32(span.x));
        let yz = i32(index / u32(span.x));
        let y = yz % span.y;
        let z = yz / span.y;
        let block = emitterBlockMin + vec3<i32>(x, y, z);
        let blockWorldMin = p.originDx.xyz + vec3<f32>(block * PAGE_SIZE_I) * p.originDx.w;
        let blockWorldMax = min(blockWorldMin + vec3<f32>(f32(PAGE_SIZE_I) * p.originDx.w), p.originDx.xyz + vec3<f32>(gridDim(p)) * p.originDx.w);
        let padding = max(0.0, config.sweepDistance) + f32(PAGE_SIZE_I) * p.originDx.w;
        if (fluidShapeIntersectsWorldAabb(
            flow.emitters[wid.x].shape,
            blockWorldMin - vec3<f32>(padding),
            blockWorldMax + vec3<f32>(padding)
        )) {
            assignPage(block);
        }
    }
}`;
}

function buildPagedFlipLookupSyncWgsl(layout: FlipPageLayout): string {
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> pageData: array<atomic<u32>>;
@group(0) @binding(1) var pageLookup: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<storage, read_write> dispatchArgs: array<u32>;
${FLIP_PAGE_DISPATCH_CONFIG_WGSL}
@group(0) @binding(3) var<uniform> config: DispatchConfig;
@group(0) @binding(4) var<storage, read_write> statusData: array<atomic<u32>>;
const LOOKUP_WIDTH: u32 = ${layout.lookupWidth}u;
const PAGE_COORD_OFFSET: u32 = ${2 + layout.numBlocks}u;

fn writeDispatch(offset: u32, itemCount: u32) {
    let groups = (itemCount + ${WORKGROUP_SIZE - 1}u) / ${WORKGROUP_SIZE}u;
    let rows = max(1u, (groups + ${MAX_WORKGROUPS - 1}u) / ${MAX_WORKGROUPS}u);
    dispatchArgs[offset] = (groups + rows - 1u) / rows;
    dispatchArgs[offset + 1u] = rows;
    dispatchArgs[offset + 2u] = 1u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    if (i == 0u) {
        let pages = min(atomicLoad(&pageData[0]), config.pageCapacity);
        let overflow = atomicLoad(&pageData[1]) != 0u || atomicLoad(&statusData[2]) != 0u;
        writeDispatch(0u, select(pages * ${FLIP_PAGE_CELLS}u, 0u, overflow));
        writeDispatch(3u, select(pages * ${FLIP_PAGE_CELLS * 3}u, 0u, overflow));
        writeDispatch(6u, select(config.activeParticleGroups * ${WORKGROUP_SIZE}u, 0u, overflow));
        writeDispatch(9u, select(config.allParticleGroups * ${WORKGROUP_SIZE}u, 0u, overflow));
        writeDispatch(12u, select(${WORKGROUP_SIZE}u, 0u, overflow));
        writeDispatch(15u, select(config.foamGroups * ${WORKGROUP_SIZE}u, 0u, overflow));
        writeDispatch(18u, select(config.polygonCellItems, 0u, overflow));
        writeDispatch(21u, select(config.polygonCubeItems, 0u, overflow));
    }
    if (i >= PAGE_COORD_OFFSET + config.pageCapacity) {
        return;
    }
    textureStore(pageLookup, vec2<i32>(i32(i % LOOKUP_WIDTH), i32(i / LOOKUP_WIDTH)), vec4<u32>(atomicLoad(&pageData[i]), 0u, 0u, 0u));
}`;
}

function buildPagedFlipStatusPublishWgsl(): string {
    return /* wgsl */ `
${FLIP_PAGE_DISPATCH_CONFIG_WGSL}
@group(0) @binding(0) var<storage, read_write> pageData: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> statusData: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> config: DispatchConfig;
@compute @workgroup_size(1)
fn main() {
    let requiredPages = atomicLoad(&pageData[0]);
    atomicStore(&statusData[0], config.frameId);
    atomicStore(&statusData[1], requiredPages);
    atomicMax(&statusData[3], requiredPages);
    if (atomicLoad(&pageData[1]) != 0u) {
        loop {
            let result = atomicCompareExchangeWeak(&statusData[2], 0u, config.frameId);
            if (result.exchanged || result.old_value != 0u) {
                break;
            }
        }
    }
}`;
}

function buildPagedFlipClearWgsl(_layout: FlipPageLayout, faces: boolean): string {
    const itemsPerPage = faces ? FLIP_PAGE_CELLS * 3 : FLIP_PAGE_CELLS;
    const bindings = faces
        ? "@group(0) @binding(1) var<storage, read_write> values: array<vec2<u32>>;"
        : `@group(0) @binding(1) var<storage, read_write> marks: array<u32>;
@group(0) @binding(2) var<storage, read_write> pressureA: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressureB: array<f32>;`;
    const clear = faces
        ? "values[i] = vec2<u32>(0u);"
        : `marks[i] = 0u;
    pressureA[i] = 0.0;
    pressureB[i] = 0.0;`;
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> pageData: array<atomic<u32>>;
${bindings}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * ${WORKGROUP_SIZE}u;
    let bufferCapacity = (arrayLength(&${faces ? "values" : "marks"}) - 1u) / ${itemsPerPage}u;
    let itemCount = min(atomicLoad(&pageData[0]), bufferCapacity) * ${itemsPerPage}u;
    if (i >= itemCount) {
        return;
    }
    ${clear}
}`;
}

function replaceWgslFunction(source: string, name: string, nextName: string, replacement: string): string {
    const start = source.indexOf(`fn ${name}`);
    const end = source.indexOf(`fn ${nextName}`, start);
    if (start === -1 || end === -1) {
        throw new Error(`[FLIP] Could not rewrite WGSL function ${name} for paged storage.`);
    }
    return `${source.slice(0, start)}${replacement}\n\n${source.slice(end)}`;
}

export function rewriteFlipWgslForPagedStorage(code: string, layout: FlipPageLayout): string {
    if (!/\bstruct\s+Params\s*\{/.test(code)) {
        return code;
    }
    const mapOffset = 2;
    const coordOffset = mapOffset + layout.numBlocks;
    let source = code.replace(
        "const FIXED_POINT:",
        `@group(1) @binding(0) var pageLookup: texture_2d<u32>;
const FLIP_PAGE_SIZE_U: u32 = ${FLIP_PAGE_SIZE}u;
const FLIP_PAGE_CELLS_U: u32 = ${FLIP_PAGE_CELLS}u;
const FLIP_PAGE_BLOCK_DIM: vec3<u32> = vec3<u32>(${layout.blockDim[0]}u, ${layout.blockDim[1]}u, ${layout.blockDim[2]}u);
const FLIP_PAGE_MAP_OFFSET: u32 = ${mapOffset}u;
const FLIP_PAGE_COORD_OFFSET: u32 = ${coordOffset}u;
const FLIP_PAGE_LOOKUP_WIDTH: u32 = ${layout.lookupWidth}u;

fn flipPageMaxPages(p: Params) -> u32 {
    return u32(p.sheeting.z);
}

fn flipPageStorageCells(p: Params) -> u32 {
    return flipPageMaxPages(p) * FLIP_PAGE_CELLS_U;
}

fn flipPageStorageFaces(p: Params) -> u32 {
    return flipPageStorageCells(p) * 3u;
}

fn flipPageWord(index: u32) -> u32 {
    return textureLoad(pageLookup, vec2<i32>(i32(index % FLIP_PAGE_LOOKUP_WIDTH), i32(index / FLIP_PAGE_LOOKUP_WIDTH)), 0).x;
}

fn flipPageBlockIndex(c: vec3<i32>) -> u32 {
    let block = vec3<u32>(c) / vec3<u32>(FLIP_PAGE_SIZE_U);
    return block.x + FLIP_PAGE_BLOCK_DIM.x * (block.y + FLIP_PAGE_BLOCK_DIM.y * block.z);
}

fn flipPageLocalIndex(c: vec3<i32>) -> u32 {
    let local = vec3<u32>(c) % vec3<u32>(FLIP_PAGE_SIZE_U);
    return local.x + FLIP_PAGE_SIZE_U * (local.y + FLIP_PAGE_SIZE_U * local.z);
}

fn flipPageCellCoord(index: u32) -> vec3<i32> {
    let page = index / FLIP_PAGE_CELLS_U;
    let localIndex = index % FLIP_PAGE_CELLS_U;
    let blockIndex = flipPageWord(FLIP_PAGE_COORD_OFFSET + page);
    let block = vec3<u32>(
        blockIndex % FLIP_PAGE_BLOCK_DIM.x,
        (blockIndex / FLIP_PAGE_BLOCK_DIM.x) % FLIP_PAGE_BLOCK_DIM.y,
        blockIndex / (FLIP_PAGE_BLOCK_DIM.x * FLIP_PAGE_BLOCK_DIM.y)
    );
    let local = vec3<u32>(
        localIndex % FLIP_PAGE_SIZE_U,
        (localIndex / FLIP_PAGE_SIZE_U) % FLIP_PAGE_SIZE_U,
        localIndex / (FLIP_PAGE_SIZE_U * FLIP_PAGE_SIZE_U)
    );
    return vec3<i32>(block * FLIP_PAGE_SIZE_U + local);
}

const FIXED_POINT:`
    );
    source = replaceWgslFunction(
        source,
        "cellIndex",
        "inCellGrid",
        `fn cellIndex(c: vec3<i32>, p: Params) -> u32 {
    if (!inCellGrid(c, p)) {
                return flipPageStorageCells(p);
    }
    let page = flipPageWord(FLIP_PAGE_MAP_OFFSET + flipPageBlockIndex(c));
            if (page == 0u || page > flipPageMaxPages(p)) {
                return flipPageStorageCells(p);
    }
    return (page - 1u) * FLIP_PAGE_CELLS_U + flipPageLocalIndex(c);
}`
    );
    source = replaceWgslFunction(
        source,
        "storageCellExists",
        "uDim",
        `fn storageCellExists(c: vec3<i32>, p: Params) -> bool {
    if (!inCellGrid(c, p)) {
        return false;
    }
    let page = flipPageWord(FLIP_PAGE_MAP_OFFSET + flipPageBlockIndex(c));
    return page != 0u && page <= flipPageMaxPages(p);
}`
    );
    source = replaceWgslFunction(
        source,
        "totalFaceCount",
        "localFaceIndex",
        `fn totalFaceCount(p: Params) -> u32 {
            return flipPageStorageFaces(p);
}`
    );
    source = replaceWgslFunction(
        source,
        "globalFaceIndex",
        "faceCoord",
        `fn globalFaceIndex(kind: u32, c: vec3<i32>, p: Params) -> u32 {
    if (!inCellGrid(c, p)) {
                return flipPageStorageFaces(p);
    }
    let page = flipPageWord(FLIP_PAGE_MAP_OFFSET + flipPageBlockIndex(c));
            if (page == 0u || page > flipPageMaxPages(p)) {
                return flipPageStorageFaces(p);
    }
    let cell = (page - 1u) * FLIP_PAGE_CELLS_U + flipPageLocalIndex(c);
    return cell * 3u + kind;
}`
    );
    source = replaceWgslFunction(
        source,
        "faceCoord",
        "faceKind",
        `fn faceCoord(localIndex: u32, d: vec3<i32>) -> vec3<i32> {
    return flipPageCellCoord(localIndex);
}`
    );
    source = replaceWgslFunction(
        source,
        "faceKind",
        "faceLocalIndex",
        `fn faceKind(globalIndex: u32, p: Params) -> u32 {
    return globalIndex % 3u;
}`
    );
    source = replaceWgslFunction(
        source,
        "faceLocalIndex",
        "faceGridDim",
        `fn faceLocalIndex(globalIndex: u32, kind: u32, p: Params) -> u32 {
    return globalIndex / 3u;
}`
    );
    if (source.includes("fn scatterFace(")) {
        source = source.replace(
            /if\s*\(\s*any\s*\(\s*c\s*<\s*vec3<i32>\s*\(\s*0\s*\)\s*\)\s*\|\|\s*any\s*\(\s*c\s*>=\s*d\s*\)\s*\)\s*\{\s*continue\s*;\s*\}/,
            `if (any(c < vec3<i32>(0)) || any(c >= d)
                    || (kind == FACE_U && c.x == d.x - 1)
                    || (kind == FACE_V && c.y == d.y - 1)
                    || (kind == FACE_W && c.z == d.z - 1)) {
                    continue;
                }`
        );
    }
    return source.replace(
        /fn main([^{}]*)\{/g,
        (match) => `${match}
    if (flipPageWord(1u) != 0u) {
        return;
    }`
    );
}

function gateFlipWgslForPagedOverflow(code: string, layout: FlipPageLayout): string {
    const gate = /* wgsl */ `
@group(1) @binding(0) var flipOverflowPageLookup: texture_2d<u32>;
const FLIP_OVERFLOW_LOOKUP_WIDTH: u32 = ${layout.lookupWidth}u;
fn flipStandaloneOverflowed() -> bool {
    return textureLoad(
        flipOverflowPageLookup,
        vec2<i32>(i32(1u % FLIP_OVERFLOW_LOOKUP_WIDTH), i32(1u / FLIP_OVERFLOW_LOOKUP_WIDTH)),
        0
    ).x != 0u;
}
`;
    return `${gate}${code}`.replace(
        /fn main([^{}]*)\{/g,
        (match) => `${match}
    if (flipStandaloneOverflowed()) {
        return;
    }`
    );
}

/** @internal */
export function createFlipSim(engine: EngineContext, options: FlipOptions = {}): FluidSim {
    const device = engine._device;
    const count = Math.max(1, Math.floor(options.count ?? 80000));
    const boundsMin: [number, number, number] = options.boundsMin ? [...options.boundsMin] : [-20, 0, -20];
    const boundsMax: [number, number, number] = options.boundsMax ? [...options.boundsMax] : [20, 20, 20];
    const groundY = options.groundY ?? boundsMin[1];
    const { dx, gridDim, markersPerCell, markerVolume, particleRadius } = resolveFlipDiscretization({
        boundsMin,
        boundsMax,
        ...(options.gridResolution !== undefined ? { gridResolution: options.gridResolution } : {}),
        ...(options.dx !== undefined ? { dx: options.dx } : {}),
        ...(options.gridDim !== undefined ? { gridDim: options.gridDim } : {}),
        ...(options.markersPerCell !== undefined ? { markersPerCell: options.markersPerCell } : {}),
        ...(options.particleRadius !== undefined ? { particleRadius: options.particleRadius } : {}),
    });
    const pagedGrid = options.pagedGrid === true;
    const pageLayout = pagedGrid ? resolveFlipPageLayout(gridDim, options.pagedGridMaxPages ?? FLIP_DEFAULT_PAGE_CAPACITY, device.limits.maxTextureDimension2D) : null;
    const maxStorageBytes = storageBufferByteLimit(device.limits);
    const maxPageCapacityByFaceBuffer = Math.max(0, Math.floor((maxStorageBytes / 8 - 1) / (FLIP_PAGE_CELLS * 3)));
    if (pageLayout && pageLayout.maxPages > maxPageCapacityByFaceBuffer) {
        const capacityLabel =
            pageLayout.maxPages === pageLayout.requestedMaxPages
                ? pageLayout.maxPages.toLocaleString()
                : `${pageLayout.maxPages.toLocaleString()} effective (${pageLayout.requestedMaxPages.toLocaleString()} requested)`;
        throw new RangeError(`[FLIP] Paged-grid capacity ${capacityLabel} exceeds this device's ${maxPageCapacityByFaceBuffer.toLocaleString()}-page storage-buffer limit.`);
    }
    if (pageLayout && pageLayout.lookupHeight > device.limits.maxTextureDimension2D) {
        throw new RangeError(
            `[FLIP] Paged-grid lookup requires a ${pageLayout.lookupWidth.toLocaleString()} x ${pageLayout.lookupHeight.toLocaleString()} texture; device maximum is ${device.limits.maxTextureDimension2D.toLocaleString()}.`
        );
    }
    const storageCellCount = pageLayout?.storageCells ?? gridDim[0] * gridDim[1] * gridDim[2];
    const storageFaceCount =
        pageLayout?.storageFaces ?? (gridDim[0] + 1) * gridDim[1] * gridDim[2] + gridDim[0] * (gridDim[1] + 1) * gridDim[2] + gridDim[0] * gridDim[1] * (gridDim[2] + 1);
    const allocatedCellCount = pageLayout?.allocatedCells ?? storageCellCount;
    const allocatedFaceCount = pageLayout?.allocatedFaces ?? storageFaceCount;
    const faceBytes = allocatedFaceCount * 8;
    if (!pagedGrid && faceBytes > maxStorageBytes) {
        throw new RangeError(
            `[FLIP] Grid ${gridDim.join(" x ")}\u00a0requires\u00a0${(faceBytes / (1024 * 1024)).toFixed(1)}\u00a0MiB per packed MAC face buffer;\u00a0device per-storage-buffer binding limit is\u00a0${(
                maxStorageBytes /
                (1024 * 1024)
            ).toFixed(1)}\u00a0MiB.`
        );
    }

    const spawnMin: [number, number, number] = options.spawnMin ? [...options.spawnMin] : [-2, 4, -2];
    const spawnMax: [number, number, number] = options.spawnMax ? [...options.spawnMax] : [2, 12, 2];
    let spawnAccept: ((x: number, y: number, z: number) => boolean) | null = null;
    const initialPositions = options.initialPositions ?? null;
    const initialVelocities = options.initialVelocities ?? null;
    if (initialVelocities && (!initialPositions || initialVelocities.length !== initialPositions.length)) {
        throw new RangeError("[FLIP] initialVelocities must match initialPositions.");
    }
    let gravity = options.gravity ?? 9.8;
    let flipRatio = options.flipRatio ?? 0.95;
    let pressureIterations = Math.max(1, Math.round(options.pressureIterations ?? 40));
    let pressureRelaxation = options.pressureRelaxation ?? 0.8;
    let pressureSolver: FlipPressureSolver = options.pressureSolver === "multigrid" ? "multigrid" : "jacobi";
    let multigridCycles = Math.min(8, Math.max(1, Math.round(options.multigridCycles ?? 2)));
    let pressureTolerance = Math.max(0, options.pressureTolerance ?? 0);
    let pressureDiagnosticsEnabled = options.pressureDiagnostics === true;
    let adaptiveMultigridCycles = multigridCycles;
    let velocityDamping = options.velocityDamping ?? 0;
    let kinematicViscosity = Math.max(0, options.kinematicViscosity ?? 0);
    let viscosityIterations = Math.max(1, Math.round(options.viscosityIterations ?? 12));
    let surfaceTension = Math.max(0, options.surfaceTension ?? 0);
    let liquidSdfEnabled = options.liquidSdf === true;
    let ghostFluidEnabled = options.ghostFluid === true;
    let fractionalSolidsEnabled = options.fractionalSolids === true;
    let movingSolidBoundariesEnabled = options.movingSolidBoundaries === true;
    let reseedParticlesEnabled = options.reseedParticles === true;
    let reseedMinParticles = Math.max(1, Math.round(options.reseedMinParticles ?? markersPerCell * 0.5));
    let reseedTargetParticles = Math.max(reseedMinParticles, Math.round(options.reseedTargetParticles ?? markersPerCell));
    let reseedMaxParticles = Math.max(reseedTargetParticles, Math.round(options.reseedMaxParticles ?? markersPerCell * 1.5));
    let reseedInterval = Math.max(1, Math.round(options.reseedInterval ?? 5));
    let reseedSubstep = 0;
    let particleSheetingEnabled = options.particleSheeting === true;
    let sheetingStrength = Math.min(1, Math.max(0.05, options.sheetingStrength ?? 0.5));
    let sheetingInterval = Math.max(1, Math.round(options.sheetingInterval ?? 5));
    let sheetingSubstep = 0;
    let polygonSurfaceEnabled = options.polygonSurface === true;
    let polygonReconstructionMultiplier = Math.round(Math.min(2, Math.max(1, options.polygonReconstructionMultiplier ?? 1)) * 4) / 4;
    const surfaceMaxTriangles = Math.max(1, Math.round(options.surfaceMaxTriangles ?? DEFAULT_SURFACE_TRIANGLE_CAPACITY));
    let restitution = options.restitution ?? 0;
    let minSubsteps = Math.max(1, Math.round(options.minSubsteps ?? 1));
    let maxSubsteps = Math.max(minSubsteps, Math.round(options.maxSubsteps ?? 8));
    let cflNumber = Math.max(0, options.cflNumber ?? 2);
    let maxSubDt = Math.max(1e-4, options.maxSubDt ?? 1 / 120);
    const timestepScheduler = createFluidTimestepScheduler();
    let warmupFrames = Math.max(0, Math.floor(options.warmupFrames ?? 0));
    let warmupStep = count;
    let liveCount = count;
    let initialTargetCount = count;
    let flowSeedsInitialParticles = true;
    let configuredFlowBreaksPrefix = false;
    let activePrefixValid = true;
    let flowOccupancyValid = false;
    let resetInitialEmitterParticleCounts: ReadonlyMap<string, number> = new Map();
    let simProfiler: FluidProfiler | null = null;
    let pressureCurrentIsA = true;
    let latestPressureDiagnostics: FluidPressureDiagnostics | undefined;
    let pressureDiagnosticGeneration = 0;
    let pressureDiagnosticSampleId = 0;
    let publishedPressureDiagnosticSampleId = 0;

    function usesLiquidSdf(): boolean {
        return liquidSdfEnabled || particleSheetingEnabled;
    }

    function needsLiquidSdfResources(): boolean {
        return usesLiquidSdf() || polygonSurfaceEnabled;
    }

    const positionBuffer = device.createBuffer({
        label: "flip-particle-positions",
        size: count * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const velocityBuffer = device.createBuffer({
        label: "flip-particle-velocities",
        size: count * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const debugBuffer = device.createBuffer({
        label: "flip-particle-debug",
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const faceAccumBuffer = device.createBuffer({
        label: "flip-face-accum",
        size: faceBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const faceOldBuffer = device.createBuffer({
        label: "flip-face-old",
        size: allocatedFaceCount * 4,
        usage: GPUBufferUsage.STORAGE,
    });
    const faceVelocityA = device.createBuffer({
        label: "flip-face-velocity-a",
        size: faceBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const faceVelocityB = device.createBuffer({
        label: "flip-face-velocity-b",
        size: faceBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const viscosityRhsBuffer = device.createBuffer({ label: "flip-viscosity-rhs", size: allocatedFaceCount * 4, usage: GPUBufferUsage.STORAGE });
    const faceDeltaA = device.createBuffer({ label: "flip-face-delta-a", size: faceBytes, usage: GPUBufferUsage.STORAGE });
    const faceDeltaB = device.createBuffer({ label: "flip-face-delta-b", size: faceBytes, usage: GPUBufferUsage.STORAGE });
    const cellMarksBuffer = device.createBuffer({
        label: "flip-cell-marks",
        size: allocatedCellCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const cellTypeBuffer = device.createBuffer({ label: "flip-cell-types", size: allocatedCellCount * 4, usage: GPUBufferUsage.STORAGE });
    const divergenceBuffer = device.createBuffer({ label: "flip-divergence", size: allocatedCellCount * 4, usage: GPUBufferUsage.STORAGE });
    const surfaceNormalBuffer = device.createBuffer({ label: "flip-surface-normal", size: allocatedCellCount * 16, usage: GPUBufferUsage.STORAGE });
    const surfaceCurvatureBuffer = device.createBuffer({ label: "flip-surface-curvature", size: allocatedCellCount * 4, usage: GPUBufferUsage.STORAGE });
    const pressureA = device.createBuffer({ label: "flip-pressure-a", size: allocatedCellCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const pressureB = device.createBuffer({ label: "flip-pressure-b", size: allocatedCellCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    let multigridResources: MultigridResources | null = null;
    let liquidSdfResources: LiquidSdfResources | null = null;
    let solidFaceResources: SolidFaceResources | null = null;
    let pressureDiagnosticResources: PressureDiagnosticResources | null = null;
    let reseedResources: ReseedResources | null = null;
    let sheetingResources: SheetingResources | null = null;
    let polygonSurfaceResources: PolygonSurfaceResources | null = null;
    let polygonSurfaceRefreshPending = false;
    let sceneSdf: SceneSdfSpec | null = null;
    const paramsBuffer = device.createBuffer({ label: "flip-params", size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const emitRangeBuffer = device.createBuffer({ label: "flip-flow-emit-range", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const maxSpeedBuffer = device.createBuffer({
        label: "flip-max-speed",
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const maxSpeedReadbacks = [0, 1].map((index) =>
        device.createBuffer({
            label: `flip-max-speed-readback-${index}`,
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
    );
    const maxSpeedReadbackStates: Array<"idle" | "copied" | "mapping"> = ["idle", "idle"];
    const maxSpeedReadbackGenerations = [0, 0];
    let maxSpeedReadbackNext = 0;
    let maxSpeedGeneration = 0;
    let maxSpeedReadbackError: unknown = null;
    let lastMaxSpeed = 0;
    const flowState = createFluidFlowState(device, count, particleRadius, markerVolume);
    enableFluidActiveCountReadback(flowState);
    const warmupParamsBuffer = device.createBuffer({
        label: "flip-warmup-params",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    let warmupSeedPositionBuffer: GPUBuffer | null = null;
    let warmupSeedVelocityBuffer: GPUBuffer | null = null;
    let warmupBindGroup: GPUBindGroup | null = null;
    let pageWarmupDiscoveryBindGroup: GPUBindGroup | null = null;
    const pageDataBuffer = pageLayout
        ? device.createBuffer({
              label: "flip-page-data",
              size: pageLayout.lookupWords * 4,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          })
        : null;
    const pageLookupTexture = pageLayout
        ? device.createTexture({
              label: "flip-page-lookup",
              size: [pageLayout.lookupWidth, pageLayout.lookupHeight],
              format: "r32uint",
              usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
          })
        : null;
    const pageDispatchBuffer = pageLayout
        ? device.createBuffer({
              label: "flip-page-dispatch",
              size: PAGE_DISPATCH_BYTES,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
          })
        : null;
    const pageDispatchConfigBuffer = pageLayout
        ? device.createBuffer({
              label: "flip-page-dispatch-config",
              size: FLIP_PAGE_DISPATCH_CONFIG_BYTES,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          })
        : null;
    const pageStatusBuffer = pageLayout
        ? device.createBuffer({
              label: "flip-page-status",
              size: FLIP_PAGE_STATUS_BYTES,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          })
        : null;
    const pageStatusReadbacks = pageLayout
        ? [0, 1].map((index) =>
              device.createBuffer({
                  label: `flip-page-status-readback-${index}`,
                  size: FLIP_PAGE_STATUS_READBACK_BYTES,
                  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
              })
          )
        : [];
    const pageStatusStates: Array<"idle" | "copied" | "mapping"> = pageStatusReadbacks.map(() => "idle");
    const pageStatusGenerations = pageStatusReadbacks.map(() => 0);
    const pageStatusSampleIds = pageStatusReadbacks.map(() => 0);
    const pageStatusSnapshots = new Map<number, PagedStepMetadataSnapshot>();
    let pageStatusGeneration = 0;
    let pageStatusFrame = 0;
    let pageStatusSampleId = 0;
    let publishedPageStatusSampleId = 0;
    let latestAcceptedPageStatusFrame = 0;
    let polygonTriangleReadbackPageFrame = 0;
    let pageStatusReadbackError: { readonly generation: number; readonly sampleId: number; readonly error: unknown } | null = null;
    let pagedGridOverflowed = false;
    let disposed = false;

    function capturePagedStepMetadata(): PagedStepMetadataSnapshot {
        return {
            liveCount,
            activePrefixValid,
            flowOccupancyValid,
            reseedSubstep,
            sheetingSubstep,
            pressureCurrentIsA,
            flowActiveCount: flowState.activeCount,
            flowFrameSeed: flowState.frameSeed,
            flowEmitterCursor: flowState.emitterCursor,
            flowEmitterCarries: flowState.emitterCarries.slice(),
            flowElapsedSeconds: flowState.elapsedSeconds,
            flowSinkCarries: flowState.sinkCarries.slice(),
            flowData: new Uint8Array(flowState.data.slice(0)),
            flowCounterData: flowState.counterData.slice(),
            foamActiveSide,
            foamSeed,
            polygonSurfaceRefreshPending,
            polygonHistoryValid: polygonSurfaceEnabled ? (polygonSurfaceResources?.historyValid ?? false) : undefined,
        };
    }

    function restorePagedStepMetadata(snapshot: PagedStepMetadataSnapshot): void {
        liveCount = snapshot.liveCount;
        activePrefixValid = snapshot.activePrefixValid;
        flowOccupancyValid = snapshot.flowOccupancyValid;
        reseedSubstep = snapshot.reseedSubstep;
        sheetingSubstep = snapshot.sheetingSubstep;
        pressureCurrentIsA = snapshot.pressureCurrentIsA;
        flowState.activeCount = snapshot.flowActiveCount;
        flowState.frameSeed = snapshot.flowFrameSeed;
        flowState.emitterCursor = snapshot.flowEmitterCursor;
        flowState.emitterCarries.set(snapshot.flowEmitterCarries);
        flowState.elapsedSeconds = snapshot.flowElapsedSeconds;
        flowState.sinkCarries.set(snapshot.flowSinkCarries);
        new Uint8Array(flowState.data).set(snapshot.flowData);
        flowState.counterData.set(snapshot.flowCounterData);
        device.queue.writeBuffer(flowState.uniformBuffer, 0, flowState.data);
        device.queue.writeBuffer(flowState.counterBuffer, 0, flowState.counterData);
        foamActiveSide = snapshot.foamActiveSide;
        foamSeed = snapshot.foamSeed;
        if (diffusePool?.activeIndices) {
            diffusePool = {
                ...diffusePool,
                activeIndicesOffset: foamActiveListOffset(foamCapacity, foamActiveSide),
            };
        }
        polygonSurfaceRefreshPending = snapshot.polygonSurfaceRefreshPending;
        if (polygonSurfaceResources && snapshot.polygonHistoryValid !== undefined) {
            polygonSurfaceResources.historyValid = snapshot.polygonHistoryValid;
        }
    }

    function pollPagedGridStatus(): void {
        if (!pageLayout) {
            return;
        }
        if (pageStatusReadbackError) {
            const pending = pageStatusReadbackError;
            pageStatusReadbackError = null;
            if (pending.generation === pageStatusGeneration && pending.sampleId > publishedPageStatusSampleId) {
                throw pending.error;
            }
        }
        for (let index = 0; index < pageStatusStates.length; index++) {
            if (pageStatusStates[index] !== "copied") {
                continue;
            }
            pageStatusStates[index] = "mapping";
            const generation = pageStatusGenerations[index]!;
            const sampleId = pageStatusSampleIds[index]!;
            const staging = pageStatusReadbacks[index]!;
            staging
                .mapAsync(GPUMapMode.READ)
                .then(() => {
                    const status = new Uint32Array(staging.getMappedRange());
                    const completedFrame = status[0]!;
                    const activePages = status[1]!;
                    const overflowFrame = status[2]!;
                    const requiredPages = Math.max(activePages, status[3]!);
                    staging.unmap();
                    pageStatusStates[index] = "idle";
                    const current =
                        generation === pageStatusGeneration &&
                        pageStatusGenerations[index] === generation &&
                        pageStatusSampleIds[index] === sampleId &&
                        sampleId > publishedPageStatusSampleId;
                    if (disposed || !current) {
                        return;
                    }
                    if (overflowFrame !== 0 && !pagedGridOverflowed) {
                        const metadata = pageStatusSnapshots.get(overflowFrame);
                        if (!metadata) {
                            pageStatusReadbackError = {
                                generation,
                                sampleId,
                                error: new Error(`[FLIP] Missing rollback metadata for paged-grid overflow frame ${overflowFrame}.`),
                            };
                            pagedGridOverflowed = true;
                            return;
                        }
                        restorePagedStepMetadata(metadata);
                        pagedGridOverflowed = true;
                    } else {
                        latestAcceptedPageStatusFrame = Math.max(latestAcceptedPageStatusFrame, completedFrame);
                    }
                    publishedPageStatusSampleId = sampleId;
                    for (const frame of pageStatusSnapshots.keys()) {
                        if (pagedGridOverflowed || frame <= completedFrame) {
                            pageStatusSnapshots.delete(frame);
                        }
                    }
                    options.onPagedGridPages?.(activePages, pageLayout.maxPages);
                    if (pagedGridOverflowed) {
                        options.onPagedGridOverflow?.(requiredPages, pageLayout.maxPages);
                    }
                })
                .catch((error: unknown) => {
                    if (!disposed) {
                        if (
                            generation === pageStatusGeneration &&
                            pageStatusGenerations[index] === generation &&
                            pageStatusSampleIds[index] === sampleId &&
                            sampleId > publishedPageStatusSampleId &&
                            sampleId > (pageStatusReadbackError?.sampleId ?? 0)
                        ) {
                            pageStatusReadbackError = { generation, sampleId, error };
                        }
                        pageStatusStates[index] = "idle";
                    }
                });
        }
    }

    function retirePagedGridStatus(): void {
        pageStatusGeneration++;
        pageStatusFrame = 0;
        pageStatusReadbackError = null;
        pageStatusSnapshots.clear();
        latestAcceptedPageStatusFrame = 0;
        polygonTriangleReadbackPageFrame = 0;
        for (let index = 0; index < pageStatusStates.length; index++) {
            if (pageStatusStates[index] === "copied") {
                pageStatusStates[index] = "idle";
            }
        }
    }

    function pollMaxSpeed(): void {
        if (maxSpeedReadbackError) {
            const error = maxSpeedReadbackError;
            maxSpeedReadbackError = null;
            throw error;
        }
        for (let index = 0; index < maxSpeedReadbacks.length; index++) {
            if (maxSpeedReadbackStates[index] !== "copied") {
                continue;
            }
            maxSpeedReadbackStates[index] = "mapping";
            const buffer = maxSpeedReadbacks[index]!;
            const generation = maxSpeedReadbackGenerations[index]!;
            void buffer
                .mapAsync(GPUMapMode.READ)
                .then(() => {
                    const bits = new Uint32Array(buffer.getMappedRange())[0] ?? 0;
                    if (generation === maxSpeedGeneration) {
                        lastMaxSpeed = new Float32Array(new Uint32Array([bits]).buffer)[0] ?? lastMaxSpeed;
                    }
                    buffer.unmap();
                    maxSpeedReadbackStates[index] = "idle";
                })
                .catch((error: unknown) => {
                    maxSpeedReadbackError = error;
                    maxSpeedReadbackStates[index] = "idle";
                });
        }
    }

    function encodeMaxSpeedReadback(encoder: GPUCommandEncoder): void {
        for (let offset = 0; offset < maxSpeedReadbacks.length; offset++) {
            const index = (maxSpeedReadbackNext + offset) % maxSpeedReadbacks.length;
            if (maxSpeedReadbackStates[index] !== "idle") {
                continue;
            }
            encoder.copyBufferToBuffer(maxSpeedBuffer, 0, maxSpeedReadbacks[index]!, 0, 4);
            maxSpeedReadbackStates[index] = "copied";
            maxSpeedReadbackGenerations[index] = maxSpeedGeneration;
            maxSpeedReadbackNext = (index + 1) % maxSpeedReadbacks.length;
            return;
        }
    }

    function retirePressureDiagnostics(): void {
        pressureDiagnosticGeneration++;
        latestPressureDiagnostics = undefined;
        adaptiveMultigridCycles = multigridCycles;
        const resources = pressureDiagnosticResources;
        if (!resources) {
            return;
        }
        resources.readbackError = null;
        for (let index = 0; index < resources.readbackStates.length; index++) {
            if (resources.readbackStates[index] === "copied") {
                resources.readbackStates[index] = "idle";
            }
        }
    }

    function pollPressureDiagnostics(): void {
        const resources = pressureDiagnosticResources;
        if (!resources) {
            return;
        }
        if (resources.readbackError) {
            const pending = resources.readbackError;
            resources.readbackError = null;
            if (pending.generation === pressureDiagnosticGeneration && pending.sampleId > publishedPressureDiagnosticSampleId) {
                throw pending.error;
            }
        }
        for (let index = 0; index < resources.readbacks.length; index++) {
            if (resources.readbackStates[index] !== "copied") {
                continue;
            }
            resources.readbackStates[index] = "mapping";
            const buffer = resources.readbacks[index]!;
            const generation = resources.readbackGenerations[index]!;
            const sampleId = resources.readbackSampleIds[index]!;
            const sampledIterations = resources.readbackIterations[index]!;
            void buffer
                .mapAsync(GPUMapMode.READ)
                .then(() => {
                    const words = new Uint32Array(buffer.getMappedRange()).slice();
                    buffer.unmap();
                    resources.readbackStates[index] = "idle";
                    const current =
                        !resources.disposed &&
                        generation === pressureDiagnosticGeneration &&
                        resources.readbackGenerations[index] === generation &&
                        resources.readbackSampleIds[index] === sampleId &&
                        sampleId > publishedPressureDiagnosticSampleId;
                    if (!current) {
                        return;
                    }
                    const floats = new Float32Array(words.slice(0, 3).buffer);
                    const maxResidual = floats[0] ?? 0;
                    const maxRhs = floats[1] ?? 0;
                    const relativeResidual = maxResidual / Math.max(maxRhs, 1e-12);
                    publishedPressureDiagnosticSampleId = sampleId;
                    latestPressureDiagnostics = {
                        maxResidual,
                        maxRhs,
                        relativeResidual,
                        maxDivergence: floats[2] ?? 0,
                        fluidCellCount: words[3] ?? 0,
                        pressureIterations: sampledIterations,
                    };
                    if (pressureSolver === "multigrid" && pressureTolerance > 0 && Number.isFinite(relativeResidual)) {
                        if (relativeResidual > pressureTolerance) {
                            adaptiveMultigridCycles = Math.min(multigridCycles, adaptiveMultigridCycles + 1);
                        } else if (relativeResidual < pressureTolerance * 0.25) {
                            adaptiveMultigridCycles = Math.max(1, adaptiveMultigridCycles - 1);
                        }
                    }
                })
                .catch((error: unknown) => {
                    if (!resources.disposed) {
                        resources.readbackStates[index] = "idle";
                        if (
                            generation === pressureDiagnosticGeneration &&
                            resources.readbackGenerations[index] === generation &&
                            resources.readbackSampleIds[index] === sampleId &&
                            sampleId > publishedPressureDiagnosticSampleId &&
                            sampleId > (resources.readbackError?.sampleId ?? 0)
                        ) {
                            resources.readbackError = { generation, sampleId, error };
                        }
                    }
                });
        }
    }

    function encodePressureDiagnosticReadback(encoder: GPUCommandEncoder, resources: PressureDiagnosticResources): void {
        for (let offset = 0; offset < resources.readbacks.length; offset++) {
            const index = (resources.readbackNext + offset) % resources.readbacks.length;
            if (resources.readbackStates[index] !== "idle") {
                continue;
            }
            encoder.copyBufferToBuffer(resources.output, 0, resources.readbacks[index]!, 0, PRESSURE_DIAGNOSTIC_BYTES);
            resources.readbackStates[index] = "copied";
            resources.readbackGenerations[index] = pressureDiagnosticGeneration;
            resources.readbackSampleIds[index] = ++pressureDiagnosticSampleId;
            resources.readbackIterations[index] = pressureSolver === "multigrid" ? (pressureTolerance > 0 ? adaptiveMultigridCycles : multigridCycles) : pressureIterations;
            resources.readbackNext = (index + 1) % resources.readbacks.length;
            return;
        }
    }

    const paramsData = new ArrayBuffer(PARAMS_BYTES);
    const paramsF32 = new Float32Array(paramsData);
    const paramsU32 = new Uint32Array(paramsData);
    paramsF32[0] = boundsMin[0];
    paramsF32[1] = boundsMin[1];
    paramsF32[2] = boundsMin[2];
    paramsF32[3] = dx;
    paramsF32[4] = gridDim[0];
    paramsF32[5] = gridDim[1];
    paramsF32[6] = gridDim[2];
    paramsF32[7] = groundY;
    paramsF32[8] = boundsMin[0];
    paramsF32[9] = boundsMin[1];
    paramsF32[10] = boundsMin[2];
    paramsF32[12] = boundsMax[0];
    paramsF32[13] = boundsMax[1];
    paramsF32[14] = boundsMax[2];
    paramsU32[28] = count;
    paramsU32[29] = markersPerCell;

    function writeParams(subDt: number): void {
        paramsF32[16] = subDt;
        paramsF32[17] = gravity;
        paramsF32[18] = flipRatio;
        paramsF32[19] = velocityDamping;
        paramsF32[20] = pressureRelaxation;
        paramsF32[21] = restitution;
        paramsF32[22] = particleRadius;
        paramsF32[23] = ((cflNumber > 0 ? cflNumber : 0.9) * dx) / Math.max(subDt, 1e-6);
        paramsF32[24] = kinematicViscosity;
        paramsF32[25] = surfaceTension;
        paramsF32[26] = viscosityIterations;
        paramsF32[27] = cflNumber;
        paramsU32[30] = fractionalSolidsEnabled && movingSolidBoundariesEnabled ? 1 : 0;
        paramsU32[32] = reseedParticlesEnabled ? 1 : 0;
        paramsU32[33] = reseedMinParticles;
        paramsU32[34] = reseedTargetParticles;
        paramsU32[35] = reseedMaxParticles;
        paramsF32[36] = particleSheetingEnabled ? 1 : 0;
        paramsF32[37] = sheetingStrength;
        paramsF32[38] = pageLayout?.maxPages ?? 0;
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    }

    const seedPositions = new Float32Array(count * 4);
    const seedVelocities = new Float32Array(count * 4);

    function seed(): void {
        resetFluidFlowState(flowState);
        flowOccupancyValid = false;
        maxSpeedGeneration++;
        lastMaxSpeed = 0;
        retirePressureDiagnostics();
        reseedSubstep = 0;
        sheetingSubstep = 0;
        const flowParticles =
            initialPositions || !flowSeedsInitialParticles
                ? null
                : createFluidInitialParticles(count, flowState.config, flowState.particleVolume, { min: boundsMin, max: boundsMax }, true);
        resetInitialEmitterParticleCounts = flowParticles?.emitterCounts ?? new Map();
        const explicitCount = initialPositions ? Math.min(count, Math.floor(initialPositions.length / 3)) : count;
        initialTargetCount = initialPositions ? explicitCount : (flowParticles?.activeCount ?? count);
        activePrefixValid = !configuredFlowBreaksPrefix && !reseedParticlesEnabled && !particleSheetingEnabled;
        warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(initialTargetCount / warmupFrames)) : Math.max(1, initialTargetCount);
        liveCount = warmupFrames > 0 ? Math.min(initialTargetCount, warmupStep) : initialTargetCount;
        resetFluidParticleLifecycle(flowState, liveCount, initialTargetCount);
        seedPositions.fill(0);
        seedVelocities.fill(0);
        const renderPositions = new Float32Array(count * 4);
        const renderVelocities = new Float32Array(count * 4);
        const flowPositions = flowParticles?.positions;
        const flowVelocities = flowParticles?.velocities;
        let lastAccepted: [number, number, number] | null = null;
        for (let i = 0; i < count; i++) {
            let x = 0;
            let y = 0;
            let z = 0;
            let vx = 0;
            let vy = 0;
            let vz = 0;
            if (initialPositions && i < initialTargetCount) {
                x = initialPositions[i * 3]!;
                y = initialPositions[i * 3 + 1]!;
                z = initialPositions[i * 3 + 2]!;
                vx = initialVelocities?.[i * 3] ?? 0;
                vy = initialVelocities?.[i * 3 + 1] ?? 0;
                vz = initialVelocities?.[i * 3 + 2] ?? 0;
            } else if (flowPositions && i < initialTargetCount) {
                x = flowPositions[i * 3]!;
                y = flowPositions[i * 3 + 1]!;
                z = flowPositions[i * 3 + 2]!;
                vx = flowVelocities![i * 3]!;
                vy = flowVelocities![i * 3 + 1]!;
                vz = flowVelocities![i * 3 + 2]!;
            } else if (!flowParticles && !initialPositions) {
                let accepted = false;
                for (let attempt = 0; attempt <= SPAWN_ACCEPT_TRIES; attempt++) {
                    x = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
                    y = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
                    z = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
                    if (!spawnAccept || spawnAccept(x, y, z)) {
                        accepted = true;
                        lastAccepted = [x, y, z];
                        break;
                    }
                }
                if (!accepted && lastAccepted) {
                    [x, y, z] = lastAccepted;
                }
            }
            const offset = i * 4;
            seedPositions[offset] = x;
            seedPositions[offset + 1] = y;
            seedPositions[offset + 2] = z;
            seedPositions[offset + 3] = 1;
            seedVelocities[offset] = vx;
            seedVelocities[offset + 1] = vy;
            seedVelocities[offset + 2] = vz;
            const active = i < liveCount;
            renderPositions[offset] = active ? x : 0;
            renderPositions[offset + 1] = active ? y : -1.0e5;
            renderPositions[offset + 2] = active ? z : 0;
            renderPositions[offset + 3] = 1;
            if (active) {
                renderVelocities[offset] = vx;
                renderVelocities[offset + 1] = vy;
                renderVelocities[offset + 2] = vz;
            }
        }
        if (liveCount < initialTargetCount) {
            ensureWarmupResources();
            device.queue.writeBuffer(warmupSeedPositionBuffer!, 0, seedPositions);
            device.queue.writeBuffer(warmupSeedVelocityBuffer!, 0, seedVelocities);
        }
        device.queue.writeBuffer(positionBuffer, 0, renderPositions);
        device.queue.writeBuffer(velocityBuffer, 0, renderVelocities);
        device.queue.writeBuffer(debugBuffer, 0, new Float32Array(count));
        clearFoamPool("flip-foam-reset");
        const encoder = device.createCommandEncoder({ label: "flip-reset-grid" });
        encoder.clearBuffer(faceAccumBuffer);
        encoder.clearBuffer(cellMarksBuffer);
        encoder.clearBuffer(pressureA);
        encoder.clearBuffer(pressureB);
        if (pageDataBuffer) {
            pagedGridOverflowed = false;
            retirePagedGridStatus();
            encoder.clearBuffer(pageDataBuffer);
            encoder.clearBuffer(pageStatusBuffer!);
        }
        if (pressureDiagnosticResources) {
            encoder.clearBuffer(pressureDiagnosticResources.output);
        }
        if (multigridResources) {
            for (let index = 1; index < multigridResources.levels.length; index++) {
                encoder.clearBuffer(multigridResources.levels[index]!.pressureA);
                encoder.clearBuffer(multigridResources.levels[index]!.pressureB);
            }
        }
        device.queue.submit([encoder.finish()]);
        pressureCurrentIsA = true;
    }

    const pagedPipelines = new WeakSet<GPUComputePipeline>();
    const pageLookupBindGroups = new WeakMap<GPUComputePipeline, GPUBindGroup>();
    function rawPipeline(label: string, code: string): GPUComputePipeline {
        return cachedFlipComputePipeline(device, label, code);
    }
    function pipeline(label: string, code: string): GPUComputePipeline {
        const pipe = rawPipeline(label, pageLayout ? rewriteFlipWgslForPagedStorage(code, pageLayout) : code);
        if (pageLayout && /\bstruct\s+Params\s*\{/.test(code)) {
            pagedPipelines.add(pipe);
        }
        return pipe;
    }
    function standalonePipeline(label: string, code: string): GPUComputePipeline {
        const pipe = rawPipeline(label, pageLayout ? gateFlipWgslForPagedOverflow(code, pageLayout) : code);
        if (pageLayout) {
            pagedPipelines.add(pipe);
        }
        return pipe;
    }
    function pagedRawPipeline(label: string, code: string): GPUComputePipeline {
        const pipe = rawPipeline(label, code);
        pagedPipelines.add(pipe);
        return pipe;
    }
    function pageLookupBindGroup(pipe: GPUComputePipeline): GPUBindGroup | null {
        if (!pageLookupTexture || !pagedPipelines.has(pipe)) {
            return null;
        }
        let bindGroup = pageLookupBindGroups.get(pipe);
        if (!bindGroup) {
            bindGroup = device.createBindGroup({
                layout: pipe.getBindGroupLayout(1),
                entries: [{ binding: 0, resource: pageLookupTexture.createView() }],
            });
            pageLookupBindGroups.set(pipe, bindGroup);
        }
        return bindGroup;
    }

    const warmupPipeline = standalonePipeline("flip-warmup", FLIP_WARMUP_WGSL);
    const pageDiscoveryPipeline = pageLayout ? rawPipeline("flip-page-discovery", buildPagedFlipDiscoveryWgsl(pageLayout)) : null;
    const pageWarmupDiscoveryPipeline = pageLayout ? rawPipeline("flip-page-discovery-warmup", buildPagedFlipWarmupDiscoveryWgsl(pageLayout)) : null;
    const pageFlowDiscoveryPipeline = pageLayout ? rawPipeline("flip-page-discovery-flow", buildPagedFlipFlowDiscoveryWgsl(pageLayout)) : null;
    const pageStatusPublishPipeline = pageLayout ? rawPipeline("flip-page-status-publish", buildPagedFlipStatusPublishWgsl()) : null;
    const pageLookupSyncPipeline = pageLayout ? rawPipeline("flip-page-lookup-sync", buildPagedFlipLookupSyncWgsl(pageLayout)) : null;
    const pageClearCellsPipeline = pageLayout ? rawPipeline("flip-page-clear-cells", buildPagedFlipClearWgsl(pageLayout, false)) : null;
    const pageClearFacesPipeline = pageLayout ? rawPipeline("flip-page-clear-faces", buildPagedFlipClearWgsl(pageLayout, true)) : null;
    const pageDiscoveryBindGroup = pageLayout
        ? device.createBindGroup({
              layout: pageDiscoveryPipeline!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: positionBuffer } },
                  { binding: 1, resource: { buffer: paramsBuffer } },
                  { binding: 2, resource: { buffer: pageDataBuffer! } },
                  { binding: 3, resource: { buffer: flowState.lifecycleBuffer } },
                  { binding: 4, resource: { buffer: pageDispatchConfigBuffer! } },
              ],
          })
        : null;
    const pageFlowDiscoveryBindGroup = pageLayout
        ? device.createBindGroup({
              layout: pageFlowDiscoveryPipeline!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: pageDataBuffer! } },
                  { binding: 1, resource: { buffer: paramsBuffer } },
                  { binding: 2, resource: { buffer: flowState.uniformBuffer } },
                  { binding: 3, resource: { buffer: pageDispatchConfigBuffer! } },
              ],
          })
        : null;
    const pageStatusPublishBindGroup = pageLayout
        ? device.createBindGroup({
              layout: pageStatusPublishPipeline!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: pageDataBuffer! } },
                  { binding: 1, resource: { buffer: pageStatusBuffer! } },
                  { binding: 2, resource: { buffer: pageDispatchConfigBuffer! } },
              ],
          })
        : null;
    const pageLookupSyncBindGroup = pageLayout
        ? device.createBindGroup({
              layout: pageLookupSyncPipeline!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: pageDataBuffer! } },
                  { binding: 1, resource: pageLookupTexture!.createView() },
                  { binding: 2, resource: { buffer: pageDispatchBuffer! } },
                  { binding: 3, resource: { buffer: pageDispatchConfigBuffer! } },
                  { binding: 4, resource: { buffer: pageStatusBuffer! } },
              ],
          })
        : null;
    const pageClearCellsBindGroup = pageLayout
        ? device.createBindGroup({
              layout: pageClearCellsPipeline!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: pageDataBuffer! } },
                  { binding: 1, resource: { buffer: cellMarksBuffer } },
                  { binding: 2, resource: { buffer: pressureA } },
                  { binding: 3, resource: { buffer: pressureB } },
              ],
          })
        : null;
    const pageClearFacesBindGroup = pageLayout
        ? device.createBindGroup({
              layout: pageClearFacesPipeline!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: pageDataBuffer! } },
                  { binding: 1, resource: { buffer: faceAccumBuffer } },
              ],
          })
        : null;

    function ensureWarmupResources(): void {
        if (!warmupSeedPositionBuffer || !warmupSeedVelocityBuffer) {
            warmupSeedPositionBuffer = device.createBuffer({
                label: "flip-warmup-seed-positions",
                size: count * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            warmupSeedVelocityBuffer = device.createBuffer({
                label: "flip-warmup-seed-velocities",
                size: count * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }
        warmupBindGroup ??= device.createBindGroup({
            layout: warmupPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: positionBuffer } },
                { binding: 1, resource: { buffer: velocityBuffer } },
                { binding: 2, resource: { buffer: warmupSeedPositionBuffer } },
                { binding: 3, resource: { buffer: warmupSeedVelocityBuffer } },
                { binding: 4, resource: { buffer: flowState.lifecycleBuffer } },
                { binding: 5, resource: { buffer: warmupParamsBuffer } },
            ],
        });
        if (pageWarmupDiscoveryPipeline) {
            pageWarmupDiscoveryBindGroup ??= device.createBindGroup({
                layout: pageWarmupDiscoveryPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: warmupSeedPositionBuffer } },
                    { binding: 1, resource: { buffer: paramsBuffer } },
                    { binding: 2, resource: { buffer: pageDataBuffer! } },
                    { binding: 3, resource: { buffer: warmupParamsBuffer } },
                    { binding: 4, resource: { buffer: pageDispatchConfigBuffer! } },
                ],
            });
        }
    }

    const p2gPipeline = pipeline("flip-p2g", P2G_WGSL);
    const normalizePipeline = pipeline("flip-normalize", NORMALIZE_WGSL);
    const viscosityRhsPipeline = pipeline("flip-viscosity-rhs", VISCOSITY_RHS_WGSL);
    const viscosityPipeline = pipeline("flip-viscosity", VISCOSITY_WGSL);
    let surfaceNormalPipeline = pipeline("flip-surface-normal", usesLiquidSdf() ? SURFACE_NORMAL_SDF_WGSL : SURFACE_NORMAL_WGSL);
    const surfaceCurvaturePipeline = pipeline("flip-surface-curvature", SURFACE_CURVATURE_WGSL);
    let surfaceForcePipeline = pipeline("flip-surface-force", usesLiquidSdf() ? SURFACE_FORCE_SDF_WGSL : SURFACE_FORCE_WGSL);
    const speedResetPipeline = standalonePipeline("flip-speed-reset", SPEED_RESET_WGSL);
    const speedReducePipeline = standalonePipeline("flip-speed-reduce", SPEED_REDUCE_WGSL);
    const foamActiveOverflowGatePipeline = pageLayout ? rawPipeline("flip-foam-active-overflow-gate", FOAM_ACTIVE_OVERFLOW_GATE_WGSL) : null;
    let divergencePipeline = pipeline("flip-divergence", buildDivergenceWgsl(null, fractionalSolidsEnabled));
    let pressurePipeline = pipeline("flip-pressure", buildPressureWgsl(usesLiquidSdf() && ghostFluidEnabled, fractionalSolidsEnabled));
    const multigridBuildRhsPipeline = pipeline("flip-multigrid-build-rhs", MULTIGRID_BUILD_RHS_WGSL);
    const multigridSmoothPipeline = standalonePipeline("flip-multigrid-smooth", MULTIGRID_SMOOTH_WGSL);
    const multigridResidualPipeline = standalonePipeline("flip-multigrid-residual", MULTIGRID_RESIDUAL_WGSL);
    let multigridFineResidualPipeline: GPUComputePipeline | null = null;
    const multigridRestrictPipeline = standalonePipeline("flip-multigrid-restrict", MULTIGRID_RESTRICT_WGSL);
    const multigridProlongatePipeline = standalonePipeline("flip-multigrid-prolongate", MULTIGRID_PROLONGATE_WGSL);
    const multigridPagedRestrictPipeline = pageLayout ? pagedRawPipeline("flip-multigrid-paged-restrict", buildPagedMultigridRestrictWgsl(pageLayout)) : null;
    const multigridPagedProlongatePipeline = pageLayout ? pagedRawPipeline("flip-multigrid-paged-prolongate", buildPagedMultigridProlongateWgsl(pageLayout)) : null;
    let projectPipeline = pipeline("flip-project", buildProjectWgsl(usesLiquidSdf() && ghostFluidEnabled, fractionalSolidsEnabled));
    const extrapolatePipeline = pipeline("flip-extrapolate", EXTRAPOLATE_WGSL);
    const flowDeletePipeline = pipeline("flip-flow-delete", FLOW_DELETE_WGSL);
    const flowMarkOccupancyPipeline = pipeline("flip-flow-mark-occupancy", FLOW_MARK_OCCUPANCY_WGSL);
    const flowEmitPipeline = pipeline("flip-flow-emit", FLOW_EMIT_WGSL);
    const flowEmitAppendPipeline = pipeline("flip-flow-emit-append", FLOW_EMIT_APPEND_WGSL);
    let classifyPipeline = pipeline("flip-classify", buildClassifyWgsl(null, fractionalSolidsEnabled));
    let g2pPipeline = pipeline("flip-g2p", buildG2pWgsl(null));

    function ensureLiquidSdf(): LiquidSdfResources {
        if (liquidSdfResources) {
            return liquidSdfResources;
        }
        const sdfA = device.createBuffer({ label: "flip-liquid-sdf-a", size: allocatedCellCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const sdfB = device.createBuffer({ label: "flip-liquid-sdf-b", size: allocatedCellCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const clearPipeline = pipeline("flip-liquid-sdf-clear", LIQUID_SDF_CLEAR_WGSL);
        const scatterPipeline = pipeline("flip-liquid-sdf-scatter", LIQUID_SDF_SCATTER_WGSL);
        const finalizePipeline = pipeline("flip-liquid-sdf-finalize", LIQUID_SDF_FINALIZE_WGSL);
        const relaxPipeline = pipeline("flip-liquid-sdf-relax", LIQUID_SDF_RELAX_WGSL);
        liquidSdfResources = {
            sdfA,
            sdfB,
            clearPipeline,
            scatterPipeline,
            finalizePipeline,
            relaxPipeline,
            clear: device.createBindGroup({
                layout: clearPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: sdfA } },
                    { binding: 1, resource: { buffer: paramsBuffer } },
                ],
            }),
            scatter: device.createBindGroup({
                layout: scatterPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: positionBuffer } },
                    { binding: 1, resource: { buffer: sdfA } },
                    { binding: 2, resource: { buffer: paramsBuffer } },
                    { binding: 3, resource: { buffer: flowState.lifecycleBuffer } },
                ],
            }),
            finalize: device.createBindGroup({
                layout: finalizePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: sdfA } },
                    { binding: 1, resource: { buffer: cellTypeBuffer } },
                    { binding: 2, resource: { buffer: sdfB } },
                    { binding: 3, resource: { buffer: paramsBuffer } },
                ],
            }),
            relaxBA: device.createBindGroup({
                layout: relaxPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: sdfB } },
                    { binding: 1, resource: { buffer: sdfA } },
                    { binding: 2, resource: { buffer: cellTypeBuffer } },
                    { binding: 3, resource: { buffer: paramsBuffer } },
                ],
            }),
            relaxAB: device.createBindGroup({
                layout: relaxPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: sdfA } },
                    { binding: 1, resource: { buffer: sdfB } },
                    { binding: 2, resource: { buffer: cellTypeBuffer } },
                    { binding: 3, resource: { buffer: paramsBuffer } },
                ],
            }),
            gpuBytes: sdfA.size + sdfB.size,
        };
        return liquidSdfResources;
    }

    function buildSolidFaceBindGroup(geometry: GPUBuffer, pipe: GPUComputePipeline, scene: SceneSdfSpec | null): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: geometry } },
            { binding: 1, resource: { buffer: paramsBuffer } },
        ];
        if (scene) {
            entries.push({ binding: 2, resource: { buffer: scene.buffer } });
            if (scene.sdfGrid) {
                entries.push({ binding: 3, resource: { buffer: scene.sdfGrid } });
            }
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    function ensureSolidFaces(): SolidFaceResources {
        if (solidFaceResources) {
            return solidFaceResources;
        }
        const geometry = device.createBuffer({ label: "flip-solid-face-geometry", size: allocatedFaceCount * 8, usage: GPUBufferUsage.STORAGE });
        const solidPipeline = pipeline("flip-solid-face-geometry", buildSolidFaceGeometryWgsl(sceneSdf));
        solidFaceResources = {
            geometry,
            pipeline: solidPipeline,
            bindGroup: buildSolidFaceBindGroup(geometry, solidPipeline, sceneSdf),
            gpuBytes: geometry.size,
        };
        return solidFaceResources;
    }

    function buildReseedEmitBindGroup(pipe: GPUComputePipeline, state: GPUBuffer, scene: SceneSdfSpec | null): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: faceVelocityA } },
            { binding: 3, resource: { buffer: cellTypeBuffer } },
            { binding: 4, resource: { buffer: state } },
            { binding: 5, resource: { buffer: paramsBuffer } },
            { binding: 6, resource: { buffer: flowState.lifecycleBuffer } },
        ];
        if (scene) {
            entries.push({ binding: 7, resource: { buffer: scene.buffer } });
            if (scene.sdfGrid) {
                entries.push({ binding: 8, resource: { buffer: scene.sdfGrid } });
            }
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    function ensureReseedResources(): ReseedResources {
        if (reseedResources) {
            return reseedResources;
        }
        const state = device.createBuffer({
            label: "flip-reseed-state",
            size: RESEED_HEADER_BYTES + count * 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(state, 8, new Uint32Array([count, 0]));
        const deleteOverfullPipeline = pipeline("flip-reseed-delete-overfull", buildReseedDeleteWgsl(true));
        const deleteSurplusPipeline = pipeline("flip-reseed-delete-surplus", buildReseedDeleteWgsl(false));
        const buildPipeline = pipeline("flip-reseed-build", RESEED_BUILD_WGSL);
        const emitPipeline = pipeline("flip-reseed-emit", buildReseedEmitWgsl(sceneSdf));
        const buildDeleteBindGroup = (deletePipeline: GPUComputePipeline): GPUBindGroup =>
            device.createBindGroup({
                layout: deletePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: positionBuffer } },
                    { binding: 2, resource: { buffer: cellMarksBuffer } },
                    { binding: 3, resource: { buffer: paramsBuffer } },
                    { binding: 4, resource: { buffer: flowState.lifecycleBuffer } },
                    { binding: 5, resource: { buffer: state } },
                ],
            });
        reseedResources = {
            state,
            deleteOverfullPipeline,
            deleteSurplusPipeline,
            buildPipeline,
            emitPipeline,
            deleteOverfullBindGroup: buildDeleteBindGroup(deleteOverfullPipeline),
            deleteSurplusBindGroup: buildDeleteBindGroup(deleteSurplusPipeline),
            buildBindGroup: device.createBindGroup({
                layout: buildPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: cellMarksBuffer } },
                    { binding: 1, resource: { buffer: cellTypeBuffer } },
                    { binding: 2, resource: { buffer: state } },
                    { binding: 3, resource: { buffer: paramsBuffer } },
                ],
            }),
            emitBindGroup: buildReseedEmitBindGroup(emitPipeline, state, sceneSdf),
            gpuBytes: state.size,
        };
        return reseedResources;
    }

    function buildSheetingEmitBindGroup(pipe: GPUComputePipeline, state: GPUBuffer, scene: SceneSdfSpec | null): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: faceVelocityA } },
            { binding: 3, resource: { buffer: liquidSdfBuffer() } },
            { binding: 4, resource: { buffer: state } },
            { binding: 5, resource: { buffer: paramsBuffer } },
            { binding: 6, resource: { buffer: flowState.lifecycleBuffer } },
        ];
        if (scene) {
            entries.push({ binding: 7, resource: { buffer: scene.buffer } });
            if (scene.sdfGrid) {
                entries.push({ binding: 8, resource: { buffer: scene.sdfGrid } });
            }
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    function ensureSheetingResources(): SheetingResources {
        if (sheetingResources) {
            return sheetingResources;
        }
        ensureLiquidSdf();
        const state = device.createBuffer({
            label: "flip-sheeting-state",
            size: SHEETING_HEADER_BYTES + count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(state, 12, new Uint32Array([count]));
        const buildPipeline = pipeline("flip-sheeting-build", SHEETING_BUILD_WGSL);
        const emitPipeline = pipeline("flip-sheeting-emit", buildSheetingEmitWgsl(sceneSdf));
        sheetingResources = {
            state,
            buildPipeline,
            emitPipeline,
            buildBindGroup: device.createBindGroup({
                layout: buildPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: cellMarksBuffer } },
                    { binding: 1, resource: { buffer: cellTypeBuffer } },
                    { binding: 2, resource: { buffer: liquidSdfBuffer() } },
                    { binding: 3, resource: { buffer: state } },
                    { binding: 4, resource: { buffer: paramsBuffer } },
                ],
            }),
            emitBindGroup: buildSheetingEmitBindGroup(emitPipeline, state, sceneSdf),
            gpuBytes: state.size,
        };
        return sheetingResources;
    }

    function destroyPolygonSurfaceResources(): void {
        polygonSurfaceResources?.surface.vertexBuffer.destroy();
        polygonSurfaceResources?.surface.indexBuffer.destroy();
        polygonSurfaceResources?.surface.drawIndirect.destroy();
        polygonSurfaceResources?.surface.wireframeIndexBuffer?.destroy();
        polygonSurfaceResources?.surface.wireframeDrawIndirect?.destroy();
        polygonSurfaceResources?.paramsBuffer.destroy();
        polygonSurfaceResources?.reconstructedSdf?.destroy();
        polygonSurfaceResources?.stabilizedSdf.destroy();
        for (const buffer of polygonSurfaceResources?.triangleCountReadback.buffers ?? []) {
            buffer.destroy();
        }
        polygonSurfaceResources = null;
    }

    function pollPolygonTriangleCount(resources: PolygonSurfaceResources, encoder: GPUCommandEncoder): void {
        const state = resources.triangleCountReadback;
        if (state.error) {
            const error = state.error;
            state.error = null;
            throw error;
        }
        for (let index = 0; index < state.buffers.length; index++) {
            if (state.states[index] !== "copied" || state.copyEncoders[index] === encoder) {
                continue;
            }
            state.states[index] = "mapping";
            const buffer = state.buffers[index]!;
            const generation = state.generations[index]!;
            void buffer
                .mapAsync(GPUMapMode.READ)
                .then(() => {
                    if (generation === state.generation) {
                        const indexCount = new Uint32Array(buffer.getMappedRange())[0] ?? 0;
                        state.latest = Math.floor(indexCount / 3);
                    }
                    buffer.unmap();
                    state.states[index] = "idle";
                    state.copyEncoders[index] = null;
                })
                .catch((error: unknown) => {
                    state.error = error;
                    state.states[index] = "idle";
                    state.copyEncoders[index] = null;
                });
        }
    }

    function encodePolygonTriangleCountReadback(encoder: GPUCommandEncoder, resources: PolygonSurfaceResources): void {
        const state = resources.triangleCountReadback;
        if (state.frame++ % 15 !== 0) {
            return;
        }
        for (let offset = 0; offset < state.buffers.length; offset++) {
            const index = (state.next + offset) % state.buffers.length;
            if (state.states[index] !== "idle") {
                continue;
            }
            encoder.copyBufferToBuffer(resources.surface.drawIndirect, 0, state.buffers[index]!, 0, 4);
            state.states[index] = "copied";
            state.generations[index] = state.generation;
            state.copyEncoders[index] = encoder;
            state.next = (index + 1) % state.buffers.length;
            return;
        }
    }

    function ensurePolygonSurfaceResources(): PolygonSurfaceResources {
        if (polygonSurfaceResources) {
            return polygonSurfaceResources;
        }
        const multiplier = polygonReconstructionMultiplier;
        const dimensions = gridDim.map((value) => Math.max(4, Math.ceil(value * multiplier))) as [number, number, number];
        const spacing = dx / multiplier;
        const cellCount = dimensions[0] * dimensions[1] * dimensions[2];
        const usesSolverSdf = multiplier <= 1 && !pageLayout;
        ensureLiquidSdf();
        const cubeCount = Math.max(1, (dimensions[0] - 1) * (dimensions[1] - 1) * (dimensions[2] - 1));
        const vertexBytes = cubeCount * SURFACE_VERTEX_STRIDE;
        const maxStorageBytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
        if (vertexBytes > maxStorageBytes) {
            throw new RangeError(
                `[FLIP] Polygon reconstruction needs ${(vertexBytes / (1024 * 1024)).toFixed(1)} MiB for surface vertices; device limit is ${(
                    maxStorageBytes /
                    (1024 * 1024)
                ).toFixed(1)} MiB.`
            );
        }
        const requestedTriangleCapacity = Math.min(surfaceMaxTriangles, cubeCount * 6, Math.floor(maxStorageBytes / 16));
        const triangleCapacity = Math.max(2, Math.floor(requestedTriangleCapacity / 2) * 2);
        const vertexBuffer = device.createBuffer({
            label: "flip-polygon-surface-vertices",
            size: vertexBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
        });
        const polygonParamsBuffer = device.createBuffer({
            label: "flip-polygon-surface-params",
            size: PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const polygonParamsData = new ArrayBuffer(PARAMS_BYTES);
        const polygonParamsF32 = new Float32Array(polygonParamsData);
        const polygonParamsU32 = new Uint32Array(polygonParamsData);
        polygonParamsF32.set(boundsMin, 0);
        polygonParamsF32[3] = spacing;
        polygonParamsF32.set(dimensions, 4);
        polygonParamsF32[7] = groundY;
        polygonParamsF32.set(boundsMin, 8);
        polygonParamsF32.set(boundsMax, 12);
        polygonParamsF32.set(gridDim, 20);
        polygonParamsF32[23] = multiplier;
        polygonParamsU32[28] = count;
        polygonParamsU32[29] = markersPerCell;
        device.queue.writeBuffer(polygonParamsBuffer, 0, polygonParamsData);
        const reconstructedSdf = usesSolverSdf
            ? null
            : device.createBuffer({
                  label: "flip-polygon-surface-reconstructed-sdf",
                  size: cellCount * 4,
                  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
              });
        const stabilizedSdf = device.createBuffer({
            label: "flip-polygon-surface-stabilized-sdf",
            size: cellCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const indexBuffer = device.createBuffer({
            label: "flip-polygon-surface-indices",
            size: triangleCapacity * 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_SRC,
        });
        const wireframeIndexBuffer = device.createBuffer({
            label: "flip-polygon-surface-wireframe-indices",
            size: triangleCapacity * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_SRC,
        });
        const drawIndirect = device.createBuffer({
            label: "flip-polygon-surface-draw",
            size: SURFACE_INDIRECT_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const wireframeDrawIndirect = device.createBuffer({
            label: "flip-polygon-surface-wireframe-draw",
            size: SURFACE_INDIRECT_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const triangleCountReadback: PolygonSurfaceResources["triangleCountReadback"] = {
            buffers: [0, 1].map((index) =>
                device.createBuffer({
                    label: `flip-polygon-surface-triangle-count-${index}`,
                    size: 4,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                })
            ),
            states: ["idle", "idle"],
            generations: [0, 0],
            copyEncoders: [null, null],
            generation: 0,
            frame: 0,
            next: 0,
            latest: undefined,
            error: null,
        };
        const sdfUpsamplePipeline = usesSolverSdf
            ? null
            : pageLayout
              ? pagedRawPipeline("flip-polygon-sdf-upsample", buildPolygonSdfUpsampleWgsl(pageLayout))
              : rawPipeline("flip-polygon-sdf-upsample", buildPolygonSdfUpsampleWgsl(null));
        const sdfUpsampleBindGroup =
            sdfUpsamplePipeline && reconstructedSdf
                ? device.createBindGroup({
                      layout: sdfUpsamplePipeline.getBindGroupLayout(0),
                      entries: [
                          { binding: 0, resource: { buffer: liquidSdfBuffer() } },
                          { binding: 1, resource: { buffer: reconstructedSdf } },
                          { binding: 2, resource: { buffer: polygonParamsBuffer } },
                      ],
                  })
                : null;
        // Polygon extraction always operates on the dense reconstruction grid.
        // Rewriting these pipelines for paged storage would reinterpret dense
        // coordinates as physical page offsets.
        const copyPipeline = standalonePipeline("flip-polygon-surface-copy", SURFACE_NET_COPY_WGSL);
        const stabilizePipeline = standalonePipeline("flip-polygon-surface-stabilize", SURFACE_NET_STABILIZE_WGSL);
        const resetPipeline = standalonePipeline("flip-polygon-surface-reset", SURFACE_NET_RESET_WGSL);
        const vertexPipeline = standalonePipeline("flip-polygon-surface-vertices", SURFACE_NET_VERTEX_WGSL);
        const indexPipeline = standalonePipeline("flip-polygon-surface-indices", SURFACE_NET_INDEX_WGSL);
        const finalizePipeline = standalonePipeline("flip-polygon-surface-finalize", SURFACE_NET_FINALIZE_WGSL);
        const surface: FluidPolygonSurface = {
            vertexBuffer,
            indexBuffer,
            drawIndirect,
            wireframeIndexBuffer,
            wireframeDrawIndirect,
            indexFormat: "uint32",
            vertexStride: SURFACE_VERTEX_STRIDE,
            triangleCapacity,
            reconstructionMultiplier: multiplier,
            get triangleCount(): number | undefined {
                return triangleCountReadback.latest;
            },
            liquidSdfBuffer: stabilizedSdf,
            gridOrigin: boundsMin,
            gridDimensions: dimensions,
            gridSpacing: spacing,
        };
        const currentSdf = reconstructedSdf ?? liquidSdfBuffer();
        polygonSurfaceResources = {
            surface,
            dimensions,
            cellCount,
            spacing,
            paramsBuffer: polygonParamsBuffer,
            reconstructedSdf,
            stabilizedSdf,
            sdfUpsamplePipeline,
            sdfUpsampleBindGroup,
            copyPipeline,
            stabilizePipeline,
            resetPipeline,
            vertexPipeline,
            indexPipeline,
            finalizePipeline,
            copyBindGroup: device.createBindGroup({
                layout: copyPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: currentSdf } },
                    { binding: 1, resource: { buffer: stabilizedSdf } },
                    { binding: 2, resource: { buffer: polygonParamsBuffer } },
                ],
            }),
            stabilizeBindGroup: device.createBindGroup({
                layout: stabilizePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: currentSdf } },
                    { binding: 1, resource: { buffer: stabilizedSdf } },
                    { binding: 2, resource: { buffer: polygonParamsBuffer } },
                ],
            }),
            resetBindGroup: device.createBindGroup({
                layout: resetPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: drawIndirect } },
                    { binding: 1, resource: { buffer: wireframeDrawIndirect } },
                ],
            }),
            vertexBindGroup: device.createBindGroup({
                layout: vertexPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: stabilizedSdf } },
                    { binding: 1, resource: { buffer: vertexBuffer } },
                    { binding: 2, resource: { buffer: polygonParamsBuffer } },
                ],
            }),
            indexBindGroup: device.createBindGroup({
                layout: indexPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: stabilizedSdf } },
                    { binding: 1, resource: { buffer: vertexBuffer } },
                    { binding: 2, resource: { buffer: indexBuffer } },
                    { binding: 3, resource: { buffer: drawIndirect } },
                    { binding: 4, resource: { buffer: wireframeIndexBuffer } },
                    { binding: 5, resource: { buffer: wireframeDrawIndirect } },
                    { binding: 6, resource: { buffer: polygonParamsBuffer } },
                ],
            }),
            finalizeBindGroup: device.createBindGroup({
                layout: finalizePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: drawIndirect } },
                    { binding: 1, resource: { buffer: indexBuffer } },
                    { binding: 2, resource: { buffer: wireframeDrawIndirect } },
                    { binding: 3, resource: { buffer: wireframeIndexBuffer } },
                ],
            }),
            cubeCount,
            gpuBytes:
                vertexBuffer.size +
                polygonParamsBuffer.size +
                (reconstructedSdf?.size ?? 0) +
                stabilizedSdf.size +
                indexBuffer.size +
                drawIndirect.size +
                wireframeIndexBuffer.size +
                wireframeDrawIndirect.size +
                triangleCountReadback.buffers.reduce((sum, buffer) => sum + buffer.size, 0),
            triangleCountReadback,
            historyValid: false,
        };
        return polygonSurfaceResources;
    }

    function encodeLiquidSdf(encoder: GPUCommandEncoder, profileStage?: string): void {
        const resources = ensureLiquidSdf();
        dispatchCells(encoder, "flip-liquid-sdf-clear", resources.clearPipeline, resources.clear, profileStage);
        dispatchMutation(encoder, "flip-liquid-sdf-scatter", resources.scatterPipeline, resources.scatter, particleGroups, PAGE_DISPATCH_ALL_PARTICLES_OFFSET, profileStage);
        dispatchCells(encoder, "flip-liquid-sdf-finalize", resources.finalizePipeline, resources.finalize, profileStage);
        let currentIsA = false;
        for (let layer = 0; layer < LIQUID_SDF_LAYERS; layer++) {
            dispatchCells(encoder, "flip-liquid-sdf-relax", resources.relaxPipeline, currentIsA ? resources.relaxAB : resources.relaxBA, profileStage);
            currentIsA = !currentIsA;
        }
    }

    function encodeParticleGridClassification(encoder: GPUCommandEncoder, activeParticleGroups: number): void {
        if (pageLayout) {
            clearPagedGrid(encoder);
            pressureCurrentIsA = true;
        } else {
            encoder.clearBuffer(faceAccumBuffer);
            encoder.clearBuffer(cellMarksBuffer);
        }
        dispatchMutation(encoder, "flip-p2g", p2gPipeline, p2gBindGroup, activeParticleGroups, PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET);
        flowOccupancyValid = true;
        dispatchCells(encoder, "flip-classify", classifyPipeline, classifyBindGroup);
    }

    function liquidSdfBuffer(): GPUBuffer {
        const resources = ensureLiquidSdf();
        return LIQUID_SDF_LAYERS % 2 === 0 ? resources.sdfB : resources.sdfA;
    }

    function encodeSheeting(encoder: GPUCommandEncoder): void {
        const resources = ensureSheetingResources();
        encoder.clearBuffer(resources.state, 0, 8);
        dispatchCells(encoder, "flip-sheeting-build", resources.buildPipeline, resources.buildBindGroup);
        dispatchMutation(encoder, "flip-sheeting-emit", resources.emitPipeline, resources.emitBindGroup, particleGroups, PAGE_DISPATCH_ALL_PARTICLES_OFFSET);
    }

    function encodePolygonSurface(encoder: GPUCommandEncoder): void {
        const resources = ensurePolygonSurfaceResources();
        pollPolygonTriangleCount(resources, encoder);
        const reconstructionGroups = Math.ceil(resources.cellCount / WORKGROUP_SIZE);
        if (resources.reconstructedSdf && resources.sdfUpsamplePipeline && resources.sdfUpsampleBindGroup) {
            dispatchMutation(
                encoder,
                "flip-polygon-sdf-upsample",
                resources.sdfUpsamplePipeline,
                resources.sdfUpsampleBindGroup,
                reconstructionGroups,
                PAGE_DISPATCH_POLYGON_CELLS_OFFSET,
                "Surface"
            );
        }
        if (resources.historyValid) {
            dispatchMutation(
                encoder,
                "flip-polygon-surface-stabilize",
                resources.stabilizePipeline,
                resources.stabilizeBindGroup,
                reconstructionGroups,
                PAGE_DISPATCH_POLYGON_CELLS_OFFSET,
                "Surface"
            );
        } else {
            dispatchMutation(
                encoder,
                "flip-polygon-surface-copy",
                resources.copyPipeline,
                resources.copyBindGroup,
                reconstructionGroups,
                PAGE_DISPATCH_POLYGON_CELLS_OFFSET,
                "Surface"
            );
            resources.historyValid = true;
        }
        dispatchMutation(encoder, "flip-polygon-surface-reset", resources.resetPipeline, resources.resetBindGroup, 1, PAGE_DISPATCH_SINGLE_OFFSET, "Surface");
        dispatchMutation(
            encoder,
            "flip-polygon-surface-vertices",
            resources.vertexPipeline,
            resources.vertexBindGroup,
            Math.ceil(resources.cubeCount / WORKGROUP_SIZE),
            PAGE_DISPATCH_POLYGON_CUBES_OFFSET,
            "Surface"
        );
        dispatchMutation(
            encoder,
            "flip-polygon-surface-indices",
            resources.indexPipeline,
            resources.indexBindGroup,
            reconstructionGroups,
            PAGE_DISPATCH_POLYGON_CELLS_OFFSET,
            "Surface"
        );
        dispatchMutation(encoder, "flip-polygon-surface-finalize", resources.finalizePipeline, resources.finalizeBindGroup, 1, PAGE_DISPATCH_SINGLE_OFFSET, "Surface");
        if (!pageLayout) {
            encodePolygonTriangleCountReadback(encoder, resources);
        }
    }

    function ensureMultigrid(): MultigridResources {
        if (multigridResources) {
            return multigridResources;
        }
        multigridFineResidualPipeline = pipeline("flip-multigrid-fine-residual", buildPressureResidualWgsl(usesLiquidSdf() && ghostFluidEnabled, fractionalSolidsEnabled));
        const dimensions = multigridDimensions(gridDim);
        const rawLevels = dimensions.map((dim, index) => {
            const levelCount = index === 0 && pageLayout ? allocatedCellCount : dim[0] * dim[1] * dim[2];
            const bytes = levelCount * 4;
            if (index > 0) {
                assertDenseMultigridLevelFits(index, dim, bytes, maxStorageBytes);
            }
            const cellTypes = index === 0 ? cellTypeBuffer : device.createBuffer({ label: `flip-multigrid-types-${index}`, size: bytes, usage: GPUBufferUsage.STORAGE });
            const levelPressureA =
                index === 0
                    ? pressureA
                    : device.createBuffer({
                          label: `flip-multigrid-pressure-a-${index}`,
                          size: bytes,
                          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                      });
            const levelPressureB =
                index === 0
                    ? pressureB
                    : device.createBuffer({
                          label: `flip-multigrid-pressure-b-${index}`,
                          size: bytes,
                          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                      });
            const rhs = device.createBuffer({ label: `flip-multigrid-rhs-${index}`, size: bytes, usage: GPUBufferUsage.STORAGE });
            const residual = device.createBuffer({ label: `flip-multigrid-residual-${index}`, size: bytes, usage: GPUBufferUsage.STORAGE });
            const fluidFraction = device.createBuffer({ label: `flip-multigrid-fluid-fraction-${index}`, size: bytes, usage: GPUBufferUsage.STORAGE });
            const params = device.createBuffer({
                label: `flip-multigrid-params-${index}`,
                size: MULTIGRID_PARAMS_BYTES,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const paramsData = new ArrayBuffer(MULTIGRID_PARAMS_BYTES);
            const paramsU32 = new Uint32Array(paramsData);
            const paramsF32 = new Float32Array(paramsData);
            const fineDim = index === 0 ? dim : dimensions[index - 1]!;
            paramsU32[0] = dim[0];
            paramsU32[1] = dim[1];
            paramsU32[2] = dim[2];
            paramsU32[4] = fineDim[0];
            paramsU32[5] = fineDim[1];
            paramsU32[6] = fineDim[2];
            paramsF32[8] = MULTIGRID_RELAXATION;
            paramsF32[9] = (dx * 2 ** index) ** 2;
            device.queue.writeBuffer(params, 0, paramsData);
            return {
                dim,
                count: levelCount,
                groups: Math.ceil(levelCount / WORKGROUP_SIZE),
                cellTypes,
                rhs,
                residual,
                fluidFraction,
                pressureA: levelPressureA,
                pressureB: levelPressureB,
                params,
            };
        });
        const levels: MultigridLevel[] = rawLevels.map((level) => ({
            ...level,
            smoothAB: device.createBindGroup({
                layout: multigridSmoothPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: level.pressureA } },
                    { binding: 1, resource: { buffer: level.pressureB } },
                    { binding: 2, resource: { buffer: level.rhs } },
                    { binding: 3, resource: { buffer: level.cellTypes } },
                    { binding: 4, resource: { buffer: level.params } },
                ],
            }),
            smoothBA: device.createBindGroup({
                layout: multigridSmoothPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: level.pressureB } },
                    { binding: 1, resource: { buffer: level.pressureA } },
                    { binding: 2, resource: { buffer: level.rhs } },
                    { binding: 3, resource: { buffer: level.cellTypes } },
                    { binding: 4, resource: { buffer: level.params } },
                ],
            }),
            residualA: device.createBindGroup({
                layout: multigridResidualPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: level.pressureA } },
                    { binding: 1, resource: { buffer: level.rhs } },
                    { binding: 2, resource: { buffer: level.cellTypes } },
                    { binding: 3, resource: { buffer: level.residual } },
                    { binding: 4, resource: { buffer: level.params } },
                ],
            }),
            residualB: device.createBindGroup({
                layout: multigridResidualPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: level.pressureB } },
                    { binding: 1, resource: { buffer: level.rhs } },
                    { binding: 2, resource: { buffer: level.cellTypes } },
                    { binding: 3, resource: { buffer: level.residual } },
                    { binding: 4, resource: { buffer: level.params } },
                ],
            }),
        }));
        for (let index = 0; index < levels.length - 1; index++) {
            const fine = levels[index]!;
            const coarse = levels[index + 1]!;
            const restrictPipeline = index === 0 && pageLayout ? multigridPagedRestrictPipeline! : multigridRestrictPipeline;
            fine.restrict = device.createBindGroup({
                layout: restrictPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: fine.residual } },
                    { binding: 1, resource: { buffer: fine.cellTypes } },
                    { binding: 2, resource: { buffer: coarse.rhs } },
                    { binding: 3, resource: { buffer: coarse.cellTypes } },
                    { binding: 4, resource: { buffer: coarse.pressureA } },
                    { binding: 5, resource: { buffer: coarse.pressureB } },
                    { binding: 6, resource: { buffer: fine.fluidFraction } },
                    { binding: 7, resource: { buffer: coarse.fluidFraction } },
                    { binding: 8, resource: { buffer: coarse.params } },
                ],
            });
            const prolongEntries = (coarsePressure: GPUBuffer, finePressure: GPUBuffer): GPUBindGroup =>
                device.createBindGroup({
                    layout: (index === 0 && pageLayout ? multigridPagedProlongatePipeline! : multigridProlongatePipeline).getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: coarsePressure } },
                        { binding: 1, resource: { buffer: coarse.cellTypes } },
                        { binding: 2, resource: { buffer: finePressure } },
                        { binding: 3, resource: { buffer: fine.cellTypes } },
                        { binding: 4, resource: { buffer: coarse.params } },
                    ],
                });
            fine.prolongAA = prolongEntries(coarse.pressureA, fine.pressureA);
            fine.prolongAB = prolongEntries(coarse.pressureA, fine.pressureB);
            fine.prolongBA = prolongEntries(coarse.pressureB, fine.pressureA);
            fine.prolongBB = prolongEntries(coarse.pressureB, fine.pressureB);
        }
        const buildRhs = device.createBindGroup({
            layout: multigridBuildRhsPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: divergenceBuffer } },
                { binding: 1, resource: { buffer: cellTypeBuffer } },
                { binding: 2, resource: { buffer: levels[0]!.rhs } },
                { binding: 3, resource: { buffer: levels[0]!.params } },
                { binding: 4, resource: { buffer: levels[0]!.fluidFraction } },
            ],
        });
        let gpuBytes = 0;
        for (let index = 0; index < levels.length; index++) {
            const level = levels[index]!;
            gpuBytes += level.rhs.size + level.residual.size + level.fluidFraction.size + level.params.size;
            if (index > 0) {
                gpuBytes += level.cellTypes.size + level.pressureA.size + level.pressureB.size;
            }
        }
        multigridResources = { levels, buildRhs, gpuBytes };
        rebuildMultigridFineResidualBindGroups();
        return multigridResources;
    }

    function encodeMultigridPressure(encoder: GPUCommandEncoder): void {
        const resources = ensureMultigrid();
        const pass = encoder.beginComputePass({
            label: "flip-multigrid",
            timestampWrites: activeProfileSpans.has("Simulation") ? undefined : simProfiler?.pass("Simulation"),
        });
        const run = (pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, groups: number, indirect = false): void => {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            const lookupBindGroup = pageLookupBindGroup(pipeline);
            if (lookupBindGroup) {
                pass.setBindGroup(1, lookupBindGroup);
            }
            if (indirect) {
                pass.dispatchWorkgroupsIndirect(pageDispatchBuffer!, PAGE_DISPATCH_CELLS_OFFSET);
            } else {
                pass.dispatchWorkgroups(Math.min(groups, MAX_WORKGROUPS), Math.ceil(groups / MAX_WORKGROUPS));
            }
        };
        const smooth = (levelIndex: number, iterations: number, currentIsA: boolean): boolean => {
            const level = resources.levels[levelIndex]!;
            for (let iteration = 0; iteration < iterations; iteration++) {
                run(
                    levelIndex === 0 ? pressurePipeline : multigridSmoothPipeline,
                    levelIndex === 0 ? (currentIsA ? pressureABindGroup : pressureBBindGroup) : currentIsA ? level.smoothAB : level.smoothBA,
                    level.groups,
                    levelIndex === 0 && pageLayout !== null
                );
                currentIsA = !currentIsA;
            }
            return currentIsA;
        };
        const solveLevel = (levelIndex: number, currentIsA: boolean): boolean => {
            const level = resources.levels[levelIndex]!;
            if (levelIndex === resources.levels.length - 1) {
                return smooth(levelIndex, MULTIGRID_COARSE_SMOOTH, currentIsA);
            }
            currentIsA = smooth(levelIndex, MULTIGRID_PRE_SMOOTH, currentIsA);
            run(
                levelIndex === 0 ? multigridFineResidualPipeline! : multigridResidualPipeline,
                levelIndex === 0 ? (currentIsA ? resources.fineResidualA! : resources.fineResidualB!) : currentIsA ? level.residualA : level.residualB,
                level.groups,
                levelIndex === 0 && pageLayout !== null
            );
            const coarse = resources.levels[levelIndex + 1]!;
            run(levelIndex === 0 && pageLayout ? multigridPagedRestrictPipeline! : multigridRestrictPipeline, level.restrict!, coarse.groups);
            const coarseCurrentIsA = solveLevel(levelIndex + 1, true);
            const prolong = coarseCurrentIsA && currentIsA ? level.prolongAA! : coarseCurrentIsA ? level.prolongAB! : currentIsA ? level.prolongBA! : level.prolongBB!;
            run(levelIndex === 0 && pageLayout ? multigridPagedProlongatePipeline! : multigridProlongatePipeline, prolong, level.groups, levelIndex === 0 && pageLayout !== null);
            return smooth(levelIndex, MULTIGRID_POST_SMOOTH, currentIsA);
        };

        const fine = resources.levels[0]!;
        run(multigridBuildRhsPipeline, resources.buildRhs, fine.groups, pageLayout !== null);
        let currentIsA = pressureCurrentIsA;
        const cycleBudget = pressureTolerance > 0 ? adaptiveMultigridCycles : multigridCycles;
        for (let cycle = 0; cycle < cycleBudget; cycle++) {
            currentIsA = solveLevel(0, currentIsA);
        }
        pass.end();
        pressureCurrentIsA = currentIsA;
    }

    const p2gBindGroup = device.createBindGroup({
        layout: p2gPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: faceAccumBuffer } },
            { binding: 3, resource: { buffer: cellMarksBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
            { binding: 5, resource: { buffer: flowState.lifecycleBuffer } },
        ],
    });
    const normalizeBindGroup = device.createBindGroup({
        layout: normalizePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: faceAccumBuffer } },
            { binding: 1, resource: { buffer: faceOldBuffer } },
            { binding: 2, resource: { buffer: faceVelocityA } },
            { binding: 3, resource: { buffer: cellTypeBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ],
    });
    const viscosityRhsBindGroup = device.createBindGroup({
        layout: viscosityRhsPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: faceVelocityA } },
            { binding: 1, resource: { buffer: viscosityRhsBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ],
    });
    const viscosityABindGroup = device.createBindGroup({
        layout: viscosityPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: faceVelocityA } },
            { binding: 1, resource: { buffer: viscosityRhsBuffer } },
            { binding: 2, resource: { buffer: faceVelocityB } },
            { binding: 3, resource: { buffer: cellTypeBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ],
    });
    const viscosityBBindGroup = device.createBindGroup({
        layout: viscosityPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: faceVelocityB } },
            { binding: 1, resource: { buffer: viscosityRhsBuffer } },
            { binding: 2, resource: { buffer: faceVelocityA } },
            { binding: 3, resource: { buffer: cellTypeBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ],
    });
    function buildSurfaceNormalBindGroup(pipe: GPUComputePipeline): GPUBindGroup {
        return device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: usesLiquidSdf() ? liquidSdfBuffer() : cellMarksBuffer } },
                { binding: 1, resource: { buffer: cellTypeBuffer } },
                { binding: 2, resource: { buffer: surfaceNormalBuffer } },
                { binding: 3, resource: { buffer: paramsBuffer } },
            ],
        });
    }
    let surfaceNormalBindGroup = buildSurfaceNormalBindGroup(surfaceNormalPipeline);
    const surfaceCurvatureBindGroup = device.createBindGroup({
        layout: surfaceCurvaturePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: surfaceNormalBuffer } },
            { binding: 1, resource: { buffer: cellTypeBuffer } },
            { binding: 2, resource: { buffer: surfaceCurvatureBuffer } },
            { binding: 3, resource: { buffer: paramsBuffer } },
        ],
    });
    function buildSurfaceForceBindGroup(pipe: GPUComputePipeline): GPUBindGroup {
        return device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: faceVelocityA } },
                { binding: 1, resource: { buffer: usesLiquidSdf() ? liquidSdfBuffer() : cellMarksBuffer } },
                { binding: 2, resource: { buffer: cellTypeBuffer } },
                { binding: 3, resource: { buffer: surfaceCurvatureBuffer } },
                { binding: 4, resource: { buffer: paramsBuffer } },
            ],
        });
    }
    let surfaceForceBindGroup = buildSurfaceForceBindGroup(surfaceForcePipeline);
    const speedResetBindGroup = device.createBindGroup({
        layout: speedResetPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: maxSpeedBuffer } }],
    });
    const speedReduceBindGroup = device.createBindGroup({
        layout: speedReducePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: velocityBuffer } },
            { binding: 1, resource: { buffer: debugBuffer } },
            { binding: 2, resource: { buffer: maxSpeedBuffer } },
            { binding: 3, resource: { buffer: flowState.lifecycleBuffer } },
        ],
    });
    function buildDivergenceBindGroup(pipe: GPUComputePipeline, scene: SceneSdfSpec | null): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: faceVelocityA } },
            { binding: 1, resource: { buffer: cellTypeBuffer } },
            { binding: 2, resource: { buffer: divergenceBuffer } },
            { binding: 3, resource: { buffer: paramsBuffer } },
            { binding: 4, resource: { buffer: faceAccumBuffer } },
        ];
        if (scene) {
            entries.push({ binding: 5, resource: { buffer: scene.buffer } });
            if (scene.sdfGrid) {
                entries.push({ binding: 6, resource: { buffer: scene.sdfGrid } });
            }
        }
        if (fractionalSolidsEnabled) {
            entries.push({ binding: 7, resource: { buffer: ensureSolidFaces().geometry } });
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }
    let divergenceBindGroup = buildDivergenceBindGroup(divergencePipeline, null);
    function buildPressureBindGroup(pipe: GPUComputePipeline, input: GPUBuffer, output: GPUBuffer): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: input } },
            { binding: 1, resource: { buffer: output } },
            { binding: 2, resource: { buffer: divergenceBuffer } },
            { binding: 3, resource: { buffer: cellTypeBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ];
        if (usesLiquidSdf() && ghostFluidEnabled) {
            entries.push({ binding: 5, resource: { buffer: liquidSdfBuffer() } });
        }
        if (fractionalSolidsEnabled) {
            entries.push({ binding: 6, resource: { buffer: ensureSolidFaces().geometry } });
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }
    let pressureABindGroup = buildPressureBindGroup(pressurePipeline, pressureA, pressureB);
    let pressureBBindGroup = buildPressureBindGroup(pressurePipeline, pressureB, pressureA);

    function buildPressureDiagnosticResidualBindGroup(pipe: GPUComputePipeline, pressure: GPUBuffer, residual: GPUBuffer): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: pressure } },
            { binding: 1, resource: { buffer: divergenceBuffer } },
            { binding: 2, resource: { buffer: cellTypeBuffer } },
            { binding: 3, resource: { buffer: residual } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ];
        if (usesLiquidSdf() && ghostFluidEnabled) {
            entries.push({ binding: 5, resource: { buffer: liquidSdfBuffer() } });
        }
        if (fractionalSolidsEnabled) {
            entries.push({ binding: 6, resource: { buffer: ensureSolidFaces().geometry } });
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    function ensurePressureDiagnosticResources(): PressureDiagnosticResources {
        if (pressureDiagnosticResources) {
            return pressureDiagnosticResources;
        }
        const residual = device.createBuffer({
            label: "flip-pressure-diagnostic-residual",
            size: allocatedCellCount * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        const output = device.createBuffer({
            label: "flip-pressure-diagnostics",
            size: PRESSURE_DIAGNOSTIC_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const readbacks = [0, 1].map((index) =>
            device.createBuffer({
                label: `flip-pressure-diagnostics-readback-${index}`,
                size: PRESSURE_DIAGNOSTIC_BYTES,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            })
        );
        const residualPipeline = pipeline("flip-pressure-diagnostic-residual", buildPressureResidualWgsl(usesLiquidSdf() && ghostFluidEnabled, fractionalSolidsEnabled));
        const reducePipeline = pipeline("flip-pressure-diagnostic-reduce", PRESSURE_DIAGNOSTIC_REDUCE_WGSL);
        const postDivergencePipeline = pipeline("flip-post-divergence-diagnostic", POST_DIVERGENCE_DIAGNOSTIC_WGSL);
        pressureDiagnosticResources = {
            residual,
            output,
            readbacks,
            readbackStates: ["idle", "idle"],
            readbackGenerations: [0, 0],
            readbackSampleIds: [0, 0],
            readbackIterations: [0, 0],
            readbackNext: 0,
            readbackError: null,
            disposed: false,
            residualPipeline,
            reducePipeline,
            postDivergencePipeline,
            residualA: buildPressureDiagnosticResidualBindGroup(residualPipeline, pressureA, residual),
            residualB: buildPressureDiagnosticResidualBindGroup(residualPipeline, pressureB, residual),
            reduce: device.createBindGroup({
                layout: reducePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: residual } },
                    { binding: 1, resource: { buffer: divergenceBuffer } },
                    { binding: 2, resource: { buffer: cellTypeBuffer } },
                    { binding: 3, resource: { buffer: output } },
                    { binding: 4, resource: { buffer: paramsBuffer } },
                ],
            }),
            postDivergence: device.createBindGroup({
                layout: postDivergencePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: divergenceBuffer } },
                    { binding: 1, resource: { buffer: cellTypeBuffer } },
                    { binding: 2, resource: { buffer: output } },
                ],
            }),
            gpuBytes: residual.size + output.size + readbacks.reduce((sum, buffer) => sum + buffer.size, 0),
        };
        return pressureDiagnosticResources;
    }

    function rebuildPressureDiagnosticResources(): void {
        if (!pressureDiagnosticResources) {
            return;
        }
        retirePressureDiagnostics();
        const resources = pressureDiagnosticResources;
        resources.residualPipeline = pipeline("flip-pressure-diagnostic-residual", buildPressureResidualWgsl(usesLiquidSdf() && ghostFluidEnabled, fractionalSolidsEnabled));
        resources.residualA = buildPressureDiagnosticResidualBindGroup(resources.residualPipeline, pressureA, resources.residual);
        resources.residualB = buildPressureDiagnosticResidualBindGroup(resources.residualPipeline, pressureB, resources.residual);
    }

    function destroyPressureDiagnosticResources(): void {
        const resources = pressureDiagnosticResources;
        if (!resources) {
            return;
        }
        retirePressureDiagnostics();
        pressureDiagnosticResources = null;
        resources.disposed = true;
        resources.residual.destroy();
        resources.output.destroy();
        for (const readback of resources.readbacks) {
            readback.destroy();
        }
    }

    function buildMultigridFineResidualBindGroup(pipe: GPUComputePipeline, pressure: GPUBuffer, residual: GPUBuffer): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: pressure } },
            { binding: 1, resource: { buffer: divergenceBuffer } },
            { binding: 2, resource: { buffer: cellTypeBuffer } },
            { binding: 3, resource: { buffer: residual } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ];
        if (usesLiquidSdf() && ghostFluidEnabled) {
            entries.push({ binding: 5, resource: { buffer: liquidSdfBuffer() } });
        }
        if (fractionalSolidsEnabled) {
            entries.push({ binding: 6, resource: { buffer: ensureSolidFaces().geometry } });
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    function rebuildMultigridFineResidualBindGroups(): void {
        if (!multigridResources) {
            return;
        }
        const fineResidual = multigridResources.levels[0]!.residual;
        multigridResources.fineResidualA = buildMultigridFineResidualBindGroup(multigridFineResidualPipeline!, pressureA, fineResidual);
        multigridResources.fineResidualB = buildMultigridFineResidualBindGroup(multigridFineResidualPipeline!, pressureB, fineResidual);
    }

    function buildProjectBindGroup(pipe: GPUComputePipeline, pressure: GPUBuffer): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: pressure } },
            { binding: 1, resource: { buffer: faceOldBuffer } },
            { binding: 2, resource: { buffer: faceVelocityA } },
            { binding: 3, resource: { buffer: faceDeltaA } },
            { binding: 4, resource: { buffer: cellTypeBuffer } },
            { binding: 5, resource: { buffer: paramsBuffer } },
        ];
        if (usesLiquidSdf() && ghostFluidEnabled) {
            entries.push({ binding: 6, resource: { buffer: liquidSdfBuffer() } });
        }
        if (fractionalSolidsEnabled) {
            entries.push({ binding: 7, resource: { buffer: ensureSolidFaces().geometry } });
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }
    let projectABindGroup = buildProjectBindGroup(projectPipeline, pressureA);
    let projectBBindGroup = buildProjectBindGroup(projectPipeline, pressureB);
    const extrapolateABindGroup = device.createBindGroup({
        layout: extrapolatePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: faceVelocityA } },
            { binding: 1, resource: { buffer: faceDeltaA } },
            { binding: 2, resource: { buffer: faceVelocityB } },
            { binding: 3, resource: { buffer: faceDeltaB } },
            { binding: 4, resource: { buffer: cellTypeBuffer } },
            { binding: 5, resource: { buffer: paramsBuffer } },
        ],
    });
    const extrapolateBBindGroup = device.createBindGroup({
        layout: extrapolatePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: faceVelocityB } },
            { binding: 1, resource: { buffer: faceDeltaB } },
            { binding: 2, resource: { buffer: faceVelocityA } },
            { binding: 3, resource: { buffer: faceDeltaA } },
            { binding: 4, resource: { buffer: cellTypeBuffer } },
            { binding: 5, resource: { buffer: paramsBuffer } },
        ],
    });
    const flowEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: velocityBuffer } },
        { binding: 2, resource: { buffer: flowState.uniformBuffer } },
        { binding: 3, resource: { buffer: flowState.counterBuffer } },
        { binding: 4, resource: { buffer: flowState.lifecycleBuffer } },
    ];
    const flowDeleteBindGroup = device.createBindGroup({ layout: flowDeletePipeline.getBindGroupLayout(0), entries: flowEntries });
    const flowMarkOccupancyBindGroup = device.createBindGroup({
        layout: flowMarkOccupancyPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: cellMarksBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
            { binding: 3, resource: { buffer: flowState.lifecycleBuffer } },
        ],
    });
    const flowEmitBindGroup = device.createBindGroup({
        layout: flowEmitPipeline.getBindGroupLayout(0),
        entries: [...flowEntries, { binding: 5, resource: { buffer: cellMarksBuffer } }, { binding: 6, resource: { buffer: paramsBuffer } }],
    });
    const flowEmitAppendBindGroup = device.createBindGroup({
        layout: flowEmitAppendPipeline.getBindGroupLayout(0),
        entries: [
            ...flowEntries,
            { binding: 5, resource: { buffer: emitRangeBuffer } },
            { binding: 6, resource: { buffer: cellMarksBuffer } },
            { binding: 7, resource: { buffer: paramsBuffer } },
        ],
    });

    let classifyBindGroup = buildClassifyBindGroup(classifyPipeline, null);
    let g2pBindGroup = buildG2pBindGroup(g2pPipeline, null);
    let forcePasses: FluidForcePasses | null = null;
    function buildForceBindGroup(pipe: GPUComputePipeline, spec: ForceFieldSpec): GPUBindGroup {
        return device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: positionBuffer } },
                { binding: 1, resource: { buffer: velocityBuffer } },
                { binding: 2, resource: { buffer: paramsBuffer } },
                { binding: 3, resource: { buffer: spec.buffer } },
                { binding: 4, resource: { buffer: flowState.lifecycleBuffer } },
            ],
        });
    }

    function buildClassifyBindGroup(pipe: GPUComputePipeline, scene: SceneSdfSpec | null): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: cellMarksBuffer } },
            { binding: 1, resource: { buffer: cellTypeBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ];
        if (scene) {
            entries.push({ binding: 3, resource: { buffer: scene.buffer } });
            if (scene.sdfGrid) {
                entries.push({ binding: 4, resource: { buffer: scene.sdfGrid } });
            }
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    function buildG2pBindGroup(pipe: GPUComputePipeline, scene: SceneSdfSpec | null): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: faceVelocityA } },
            { binding: 3, resource: { buffer: faceDeltaA } },
            { binding: 4, resource: { buffer: paramsBuffer } },
            { binding: 5, resource: { buffer: cellMarksBuffer } },
        ];
        if (scene) {
            entries.push({ binding: 6, resource: { buffer: scene.buffer } });
            if (scene.sdfGrid) {
                entries.push({ binding: 7, resource: { buffer: scene.sdfGrid } });
            }
        }
        entries.push({ binding: scene ? (scene.sdfGrid ? 8 : 7) : 6, resource: { buffer: flowState.lifecycleBuffer } });
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    const activeProfileSpans = new Set<string>();

    function beginProfileSpan(encoder: GPUCommandEncoder, stage: string): (() => void) | null {
        const span = simProfiler?.stageSpan?.(stage);
        if (!span) {
            return null;
        }
        activeProfileSpans.add(stage);
        encoder.beginComputePass({ label: `${stage}-start`, timestampWrites: span.begin }).end();
        return () => {
            encoder.beginComputePass({ label: `${stage}-end`, timestampWrites: span.end }).end();
            activeProfileSpans.delete(stage);
        };
    }

    function dispatch(encoder: GPUCommandEncoder, label: string, pipe: GPUComputePipeline, bindGroup: GPUBindGroup, groups: number, profileStage?: string): void {
        if (groups <= 0) {
            return;
        }
        const stage = profileStage ?? (label.includes("foam") ? "Foam gen" : "Simulation");
        const pass = encoder.beginComputePass({ label, timestampWrites: activeProfileSpans.has(stage) ? undefined : simProfiler?.pass(stage) });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bindGroup);
        const lookupBindGroup = pageLookupBindGroup(pipe);
        if (lookupBindGroup) {
            pass.setBindGroup(1, lookupBindGroup);
        }
        if (groups > MAX_WORKGROUPS) {
            pass.dispatchWorkgroups(MAX_WORKGROUPS, Math.ceil(groups / MAX_WORKGROUPS), 1);
        } else {
            pass.dispatchWorkgroups(groups);
        }
        pass.end();
    }

    function dispatchIndirect(
        encoder: GPUCommandEncoder,
        label: string,
        pipe: GPUComputePipeline,
        bindGroup: GPUBindGroup,
        args: GPUBuffer,
        offset = 0,
        profileStage?: string
    ): void {
        const stage = profileStage ?? (label.includes("foam") ? "Foam gen" : "Simulation");
        const pass = encoder.beginComputePass({ label, timestampWrites: activeProfileSpans.has(stage) ? undefined : simProfiler?.pass(stage) });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bindGroup);
        const lookupBindGroup = pageLookupBindGroup(pipe);
        if (lookupBindGroup) {
            pass.setBindGroup(1, lookupBindGroup);
        }
        pass.dispatchWorkgroupsIndirect(args, offset);
        pass.end();
    }

    function dispatchMutation(
        encoder: GPUCommandEncoder,
        label: string,
        pipe: GPUComputePipeline,
        bindGroup: GPUBindGroup,
        groups: number,
        pagedOffset: number,
        profileStage?: string
    ): void {
        if (pageDispatchBuffer) {
            dispatchIndirect(encoder, label, pipe, bindGroup, pageDispatchBuffer, pagedOffset, profileStage);
        } else {
            dispatch(encoder, label, pipe, bindGroup, groups, profileStage);
        }
    }

    const particleGroups = Math.ceil(count / WORKGROUP_SIZE);
    const cellGroups = Math.ceil(storageCellCount / WORKGROUP_SIZE);
    const faceGroups = Math.ceil(storageFaceCount / WORKGROUP_SIZE);

    function dispatchCells(encoder: GPUCommandEncoder, label: string, pipe: GPUComputePipeline, bindGroup: GPUBindGroup, profileStage?: string): void {
        if (pageDispatchBuffer) {
            dispatchIndirect(encoder, label, pipe, bindGroup, pageDispatchBuffer, PAGE_DISPATCH_CELLS_OFFSET, profileStage);
        } else {
            dispatch(encoder, label, pipe, bindGroup, cellGroups, profileStage);
        }
    }

    function dispatchFaces(encoder: GPUCommandEncoder, label: string, pipe: GPUComputePipeline, bindGroup: GPUBindGroup): void {
        if (pageDispatchBuffer) {
            dispatchIndirect(encoder, label, pipe, bindGroup, pageDispatchBuffer, PAGE_DISPATCH_FACES_OFFSET);
        } else {
            dispatch(encoder, label, pipe, bindGroup, faceGroups);
        }
    }

    function encodePagedGridLookup(encoder: GPUCommandEncoder, activeParticleGroups: number, flowEmitterGroups = 0, reset = false, warmupGroups = 0, statusFrame = 0): void {
        if (!pageLayout) {
            return;
        }
        if (reset) {
            encoder.clearBuffer(pageDataBuffer!);
        }
        dispatch(encoder, "flip-page-discovery", pageDiscoveryPipeline!, pageDiscoveryBindGroup!, activeParticleGroups);
        dispatch(encoder, "flip-page-discovery-warmup", pageWarmupDiscoveryPipeline!, pageWarmupDiscoveryBindGroup!, warmupGroups);
        dispatch(encoder, "flip-page-discovery-flow", pageFlowDiscoveryPipeline!, pageFlowDiscoveryBindGroup!, flowEmitterGroups);
        if (statusFrame !== 0) {
            dispatch(encoder, "flip-page-status-publish", pageStatusPublishPipeline!, pageStatusPublishBindGroup!, 1);
        }
        dispatch(encoder, "flip-page-lookup-sync", pageLookupSyncPipeline!, pageLookupSyncBindGroup!, Math.ceil(pageLayout.lookupWords / WORKGROUP_SIZE));
    }

    function encodePagedGridStatusReadback(encoder: GPUCommandEncoder): void {
        const pageStatusIndex = pageStatusStates.indexOf("idle");
        if (pageStatusIndex < 0) {
            return;
        }
        encoder.copyBufferToBuffer(pageStatusBuffer!, 0, pageStatusReadbacks[pageStatusIndex]!, 0, 16);
        pageStatusStates[pageStatusIndex] = "copied";
        pageStatusGenerations[pageStatusIndex] = pageStatusGeneration;
        pageStatusSampleIds[pageStatusIndex] = ++pageStatusSampleId;
    }

    function encodeConfirmedPolygonTriangleCountReadback(encoder: GPUCommandEncoder): void {
        if (!pageLayout || !polygonSurfaceResources) {
            return;
        }
        pollPolygonTriangleCount(polygonSurfaceResources, encoder);
        if (latestAcceptedPageStatusFrame <= polygonTriangleReadbackPageFrame) {
            return;
        }
        encodePolygonTriangleCountReadback(encoder, polygonSurfaceResources);
        polygonTriangleReadbackPageFrame = latestAcceptedPageStatusFrame;
    }

    function clearPagedGrid(encoder: GPUCommandEncoder): void {
        dispatchFaces(encoder, "flip-page-clear-faces", pageClearFacesPipeline!, pageClearFacesBindGroup!);
        dispatchCells(encoder, "flip-page-clear-cells", pageClearCellsPipeline!, pageClearCellsBindGroup!);
    }

    function encodeReseed(encoder: GPUCommandEncoder, activeParticleGroups: number): void {
        const resources = ensureReseedResources();
        encoder.clearBuffer(resources.state, 0, 8);
        encoder.clearBuffer(resources.state, 16, 4);
        dispatchCells(encoder, "flip-reseed-build", resources.buildPipeline, resources.buildBindGroup);
        dispatchMutation(
            encoder,
            "flip-reseed-delete-overfull",
            resources.deleteOverfullPipeline,
            resources.deleteOverfullBindGroup,
            activeParticleGroups,
            PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET
        );
        dispatchMutation(
            encoder,
            "flip-reseed-delete-surplus",
            resources.deleteSurplusPipeline,
            resources.deleteSurplusBindGroup,
            activeParticleGroups,
            PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET
        );
        dispatchMutation(encoder, "flip-reseed-emit", resources.emitPipeline, resources.emitBindGroup, particleGroups, PAGE_DISPATCH_ALL_PARTICLES_OFFSET);
    }

    function rebuildQualityPipelines(): void {
        if (usesLiquidSdf()) {
            ensureLiquidSdf();
        }
        if (fractionalSolidsEnabled) {
            const resources = ensureSolidFaces();
            resources.pipeline = pipeline("flip-solid-face-geometry", buildSolidFaceGeometryWgsl(sceneSdf));
            resources.bindGroup = buildSolidFaceBindGroup(resources.geometry, resources.pipeline, sceneSdf);
        }
        const useGhostFluid = usesLiquidSdf() && ghostFluidEnabled;
        classifyPipeline = pipeline("flip-classify", buildClassifyWgsl(sceneSdf, fractionalSolidsEnabled));
        surfaceNormalPipeline = pipeline("flip-surface-normal", usesLiquidSdf() ? SURFACE_NORMAL_SDF_WGSL : SURFACE_NORMAL_WGSL);
        surfaceForcePipeline = pipeline("flip-surface-force", usesLiquidSdf() ? SURFACE_FORCE_SDF_WGSL : SURFACE_FORCE_WGSL);
        divergencePipeline = pipeline("flip-divergence", buildDivergenceWgsl(sceneSdf, fractionalSolidsEnabled));
        pressurePipeline = pipeline("flip-pressure", buildPressureWgsl(useGhostFluid, fractionalSolidsEnabled));
        if (multigridResources) {
            multigridFineResidualPipeline = pipeline("flip-multigrid-fine-residual", buildPressureResidualWgsl(useGhostFluid, fractionalSolidsEnabled));
        }
        projectPipeline = pipeline("flip-project", buildProjectWgsl(useGhostFluid, fractionalSolidsEnabled));
        surfaceNormalBindGroup = buildSurfaceNormalBindGroup(surfaceNormalPipeline);
        surfaceForceBindGroup = buildSurfaceForceBindGroup(surfaceForcePipeline);
        divergenceBindGroup = buildDivergenceBindGroup(divergencePipeline, sceneSdf);
        classifyBindGroup = buildClassifyBindGroup(classifyPipeline, sceneSdf);
        pressureABindGroup = buildPressureBindGroup(pressurePipeline, pressureA, pressureB);
        pressureBBindGroup = buildPressureBindGroup(pressurePipeline, pressureB, pressureA);
        rebuildPressureDiagnosticResources();
        rebuildMultigridFineResidualBindGroups();
        projectABindGroup = buildProjectBindGroup(projectPipeline, pressureA);
        projectBBindGroup = buildProjectBindGroup(projectPipeline, pressureB);
    }

    const FOAM_CAP_LIMIT = Math.min(
        Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / 32),
        MAX_WORKGROUPS * MAX_WORKGROUPS * WORKGROUP_SIZE
    );
    const foamData = new ArrayBuffer(FOAM_BYTES);
    const foamF32 = new Float32Array(foamData);
    const foamU32 = new Uint32Array(foamData);
    let foamEnabled = false;
    let foamSeed = 0;
    let foamCapacity = 0;
    let foamPoolGroups = 0;
    let foamActiveParticles = false;
    let foamActiveSide = 0;
    let diffuseBuffer: GPUBuffer | null = null;
    let diffuseHeadBuffer: GPUBuffer | null = null;
    let foamActiveStateBuffer: GPUBuffer | null = null;
    let foamActiveDispatchBuffer: GPUBuffer | null = null;
    let foamDrawIndirectBuffer: GPUBuffer | null = null;
    let foamParamsBuffer: GPUBuffer | null = null;
    let foamEmitPipeline: GPUComputePipeline | null = null;
    let foamDenseEmitPipeline: GPUComputePipeline | null = null;
    let foamActiveEmitPipeline: GPUComputePipeline | null = null;
    let foamUpdatePipeline: GPUComputePipeline | null = null;
    let foamDenseUpdatePipeline: GPUComputePipeline | null = null;
    let foamActiveUpdatePipeline: GPUComputePipeline | null = null;
    let foamActivePreparePipeline: GPUComputePipeline | null = null;
    let foamActiveFinishPipeline: GPUComputePipeline | null = null;
    let foamEmitBindGroup: GPUBindGroup | null = null;
    let foamUpdateBindGroup: GPUBindGroup | null = null;
    let foamActivePrepareBindGroup: GPUBindGroup | null = null;
    let foamActiveFinishBindGroup: GPUBindGroup | null = null;
    let foamActiveOverflowGateBindGroup: GPUBindGroup | null = null;
    let diffusePool: DiffusePool | undefined;
    let foamCountTracker: ReturnType<typeof createDiffuseCountTracker> | null = null;

    function buildFoamBindGroups(): void {
        foamEmitBindGroup = device.createBindGroup({
            layout: foamEmitPipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: positionBuffer } },
                { binding: 1, resource: { buffer: velocityBuffer } },
                { binding: 2, resource: { buffer: faceVelocityA } },
                { binding: 3, resource: { buffer: surfaceNormalBuffer } },
                { binding: 4, resource: { buffer: surfaceCurvatureBuffer } },
                { binding: 5, resource: { buffer: paramsBuffer } },
                { binding: 6, resource: { buffer: foamParamsBuffer! } },
                { binding: 7, resource: { buffer: diffuseBuffer! } },
                { binding: 8, resource: { buffer: foamActiveParticles ? foamActiveStateBuffer! : diffuseHeadBuffer! } },
                { binding: 9, resource: { buffer: flowState.lifecycleBuffer } },
            ],
        });
        const updateEntries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: faceVelocityA } },
            { binding: 1, resource: { buffer: cellMarksBuffer } },
            { binding: 2, resource: { buffer: surfaceNormalBuffer } },
            { binding: 3, resource: { buffer: paramsBuffer } },
            { binding: 4, resource: { buffer: foamParamsBuffer! } },
            { binding: 5, resource: { buffer: diffuseBuffer! } },
        ];
        if (foamActiveParticles) {
            updateEntries.push({ binding: 6, resource: { buffer: foamActiveStateBuffer! } });
        }
        foamUpdateBindGroup = device.createBindGroup({ layout: foamUpdatePipeline!.getBindGroupLayout(0), entries: updateEntries });
        foamActivePrepareBindGroup = foamActiveParticles
            ? device.createBindGroup({
                  layout: foamActivePreparePipeline!.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: foamActiveStateBuffer! } },
                      { binding: 1, resource: { buffer: foamActiveDispatchBuffer! } },
                  ],
              })
            : null;
        foamActiveFinishBindGroup = foamActiveParticles
            ? device.createBindGroup({
                  layout: foamActiveFinishPipeline!.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: foamActiveStateBuffer! } },
                      { binding: 1, resource: { buffer: foamDrawIndirectBuffer! } },
                      { binding: 2, resource: { buffer: foamActiveDispatchBuffer! } },
                  ],
              })
            : null;
    }

    function ensureFoam(config: FoamConfig): void {
        if (!foamDenseEmitPipeline) {
            const nextFoamDenseEmitPipeline = pipeline("flip-foam-emit", buildFoamEmitWgsl(false));
            const nextFoamDenseUpdatePipeline = pipeline("flip-foam-update", buildFoamUpdateWgsl(false));
            const [nextFoamParamsBuffer, nextDiffuseHeadBuffer] = createGpuBuffersAtomically(device, [
                { label: "flip-foam-params", size: FOAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
                { label: "flip-foam-head", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC },
            ]);
            foamDenseEmitPipeline = nextFoamDenseEmitPipeline;
            foamDenseUpdatePipeline = nextFoamDenseUpdatePipeline;
            foamParamsBuffer = nextFoamParamsBuffer!;
            diffuseHeadBuffer = nextDiffuseHeadBuffer!;
        }
        const nextActiveParticles = config.activeParticles ?? true;
        const activeModeChanged = nextActiveParticles !== foamActiveParticles;
        if (nextActiveParticles && !foamActiveEmitPipeline) {
            const nextFoamActiveEmitPipeline = pipeline("flip-foam-emit-active", buildFoamEmitWgsl(true));
            const nextFoamActiveUpdatePipeline = pipeline("flip-foam-update-active", buildFoamUpdateWgsl(true));
            const nextFoamActivePreparePipeline = standalonePipeline("flip-foam-active-prepare", FOAM_ACTIVE_PREPARE_WGSL);
            const nextFoamActiveFinishPipeline = standalonePipeline("flip-foam-active-finish", FOAM_ACTIVE_FINISH_WGSL);
            foamActiveEmitPipeline = nextFoamActiveEmitPipeline;
            foamActiveUpdatePipeline = nextFoamActiveUpdatePipeline;
            foamActivePreparePipeline = nextFoamActivePreparePipeline;
            foamActiveFinishPipeline = nextFoamActiveFinishPipeline;
        }
        let capacity = Math.round(count * (config.poolScale ?? 3));
        capacity = Math.max(1024, Math.min(capacity, config.poolCapMax ?? Number.POSITIVE_INFINITY, FOAM_CAP_LIMIT));
        if (nextActiveParticles) {
            capacity = Math.min(capacity, Math.max(1024, Math.floor((device.limits.maxStorageBufferBindingSize - 256) / 12)));
            while (foamActiveStateBytes(capacity) > device.limits.maxStorageBufferBindingSize) {
                capacity--;
            }
        }
        const resizePool = capacity !== foamCapacity || !diffuseBuffer;
        const rebuildActiveResources = nextActiveParticles && (resizePool || !foamActiveStateBuffer || !foamActiveDispatchBuffer || !foamDrawIndirectBuffer);
        const descriptors: GPUBufferDescriptor[] = [];
        const diffuseIndex = resizePool
            ? descriptors.push({
                  label: "flip-foam-pool",
                  size: capacity * 32,
                  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
              }) - 1
            : -1;
        const activeIndex = rebuildActiveResources
            ? descriptors.push(
                  {
                      label: "flip-foam-active-state",
                      size: foamActiveStateBytes(capacity),
                      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                  },
                  {
                      label: "flip-foam-active-dispatch",
                      size: 12,
                      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                  },
                  {
                      label: "flip-foam-draw-indirect",
                      size: 16,
                      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                  }
              ) - 3
            : -1;
        const replacements = createGpuBuffersAtomically(device, descriptors);
        const retiredDiffuseBuffer = resizePool ? diffuseBuffer : null;
        const retireActiveResources = rebuildActiveResources || (!nextActiveParticles && !!foamActiveStateBuffer);
        const retiredFoamActiveStateBuffer = retireActiveResources ? foamActiveStateBuffer : null;
        const retiredFoamActiveDispatchBuffer = retireActiveResources ? foamActiveDispatchBuffer : null;
        const retiredFoamDrawIndirectBuffer = retireActiveResources ? foamDrawIndirectBuffer : null;
        foamActiveParticles = nextActiveParticles;
        foamEmitPipeline = foamActiveParticles ? foamActiveEmitPipeline! : foamDenseEmitPipeline;
        foamUpdatePipeline = foamActiveParticles ? foamActiveUpdatePipeline! : foamDenseUpdatePipeline!;
        if (resizePool) {
            diffuseBuffer = replacements[diffuseIndex]!;
            foamCapacity = capacity;
            foamPoolGroups = Math.ceil(capacity / WORKGROUP_SIZE);
            const encoder = device.createCommandEncoder({ label: "flip-foam-clear" });
            encoder.clearBuffer(diffuseBuffer);
            encoder.clearBuffer(diffuseHeadBuffer!);
            device.queue.submit([encoder.finish()]);
        }
        if (rebuildActiveResources) {
            foamActiveStateBuffer = replacements[activeIndex]!;
            foamActiveDispatchBuffer = replacements[activeIndex + 1]!;
            foamDrawIndirectBuffer = replacements[activeIndex + 2]!;
            foamActiveOverflowGateBindGroup = foamActiveOverflowGatePipeline
                ? device.createBindGroup({
                      layout: foamActiveOverflowGatePipeline.getBindGroupLayout(0),
                      entries: [
                          { binding: 0, resource: { buffer: pageDataBuffer! } },
                          { binding: 1, resource: { buffer: foamActiveDispatchBuffer } },
                          { binding: 2, resource: { buffer: pageStatusBuffer! } },
                      ],
                  })
                : null;
            foamActiveSide = 0;
        } else if (!foamActiveParticles && foamActiveStateBuffer) {
            foamActiveStateBuffer = null;
            foamActiveDispatchBuffer = null;
            foamDrawIndirectBuffer = null;
            foamActiveOverflowGateBindGroup = null;
        }
        if (foamActiveParticles && (rebuildActiveResources || activeModeChanged)) {
            const encoder = device.createCommandEncoder({ label: "flip-foam-active-reset" });
            encoder.clearBuffer(diffuseBuffer!);
            encoder.clearBuffer(diffuseHeadBuffer!);
            encoder.clearBuffer(foamActiveStateBuffer!);
            device.queue.submit([encoder.finish()]);
            device.queue.writeBuffer(foamActiveStateBuffer!, 16, new Uint32Array([capacity]));
            foamActiveSide = 0;
        } else if (!foamActiveParticles && activeModeChanged) {
            const encoder = device.createCommandEncoder({ label: "flip-foam-dense-head-reset" });
            encoder.clearBuffer(diffuseHeadBuffer!);
            device.queue.submit([encoder.finish()]);
        }
        foamCountTracker ??= createDiffuseCountTracker(device, "flip-foam");
        foamCountTracker.configure(diffuseBuffer!, capacity, foamActiveParticles ? foamActiveStateBuffer! : undefined);
        diffusePool = {
            buffer: diffuseBuffer!,
            headBuffer: foamActiveParticles ? foamActiveStateBuffer! : diffuseHeadBuffer!,
            capacity,
            get counts() {
                return foamCountTracker!.counts;
            },
            ...(foamActiveParticles
                ? {
                      activeIndices: foamActiveStateBuffer!,
                      activeIndicesOffset: foamActiveListOffset(capacity, foamActiveSide),
                      drawIndirect: foamDrawIndirectBuffer!,
                  }
                : {}),
        };
        buildFoamBindGroups();
        foamF32[0] = 0.5;
        foamF32[1] = 4;
        foamF32[2] = 0.05;
        foamF32[3] = 1.5;
        foamF32[4] = 0.25;
        foamF32[5] = 20;
        foamF32[6] = config.kTa ?? 40;
        foamF32[7] = config.kWc ?? 40;
        foamF32[8] = config.kb ?? 0.8;
        foamF32[9] = config.kd ?? 0.5;
        foamF32[10] = config.rv ?? particleRadius;
        foamF32[12] = config.tMin ?? 0.3;
        foamF32[13] = config.tMax ?? 2;
        foamF32[16] = config.kTurb ?? 0;
        foamF32[17] = Math.max(0, config.energySpeedMin ?? Math.sqrt(0.5));
        foamF32[18] = Math.max(foamF32[17]! + 1e-4, config.energySpeedMax ?? Math.sqrt(40));
        foamF32[19] = Math.max(0, config.sprayDrag ?? 0);
        foamF32[20] = Math.max(0, config.turbulenceMin ?? 0.1);
        foamF32[21] = Math.max(foamF32[20]! + 1e-4, config.turbulenceMax ?? 2.5);
        foamF32[22] = Math.max(0, config.curvatureMin ?? 0.05);
        foamF32[23] = Math.max(foamF32[22]! + 1e-4, config.curvatureMax ?? 1.5);
        foamF32[24] = Math.max(0, config.foamLayerDepth ?? 0);
        foamU32[28] = config.generateSpray === false ? 0 : 1;
        foamU32[29] = config.generateFoam === false ? 0 : 1;
        foamU32[30] = config.generateBubbles === false ? 0 : 1;
        device.queue.writeBuffer(foamParamsBuffer!, 0, foamData);
        retiredDiffuseBuffer?.destroy();
        retiredFoamActiveStateBuffer?.destroy();
        retiredFoamActiveDispatchBuffer?.destroy();
        retiredFoamDrawIndirectBuffer?.destroy();
    }

    function clearFoamPool(label: string): void {
        if (!diffuseBuffer || !diffuseHeadBuffer) {
            return;
        }
        const encoder = device.createCommandEncoder({ label });
        encoder.clearBuffer(diffuseBuffer);
        encoder.clearBuffer(diffuseHeadBuffer);
        if (foamActiveStateBuffer) {
            encoder.clearBuffer(foamActiveStateBuffer);
            encoder.clearBuffer(foamDrawIndirectBuffer!);
        }
        device.queue.submit([encoder.finish()]);
        foamCountTracker?.reset(foamCapacity);
        if (foamActiveStateBuffer) {
            device.queue.writeBuffer(foamActiveStateBuffer, 16, new Uint32Array([foamCapacity]));
            foamActiveSide = 0;
            if (diffusePool) {
                diffusePool = { ...diffusePool, activeIndicesOffset: foamActiveListOffset(foamCapacity, foamActiveSide) };
            }
        }
    }

    function flowEmitCandidateCount(flowFrame: ReturnType<typeof prepareFluidFlowFrame>): number {
        const emitters = flowState.config?.emitters ?? [];
        let candidates = 0;
        for (let index = 0; index < emitters.length; index++) {
            const emitter = emitters[index]!;
            if (!emitter.enabled || emitter.behavior !== "inflow") {
                continue;
            }
            const sourceCapacity = Math.min(count, Math.max(0, Math.ceil(fluidShapeVolume(emitter.shape, emitter.transform) / flowState.particleVolume)));
            const budget = flowState.counterData[index * 2 + 1] ?? 0;
            candidates += emitter.volumeRate === undefined ? sourceCapacity : Math.min(sourceCapacity, budget);
        }
        return Math.min(count, flowFrame.emitUnlimited ? candidates : Math.min(candidates, flowFrame.emitCount));
    }

    if (pressureSolver === "multigrid") {
        ensureMultigrid();
    }
    if (pressureDiagnosticsEnabled || pressureTolerance > 0) {
        ensurePressureDiagnosticResources();
    }
    seed();
    writeParams(1 / 120);

    const sim: FluidSim = {
        count,
        get activeCount(): number {
            return flowState.activeCount;
        },
        get initialEmitterParticleCounts(): ReadonlyMap<string, number> {
            return resetInitialEmitterParticleCounts;
        },
        get renderCount(): number {
            return activePrefixValid ? liveCount : count;
        },
        particleRadius,
        surfaceThicknessScale: 8 / markersPerCell,
        surfaceRejectSparseMarkers: true,
        positionBuffer,
        velocityBuffer,
        debugBuffer,
        debugNorm: 1 / 8,
        get pressureDiagnostics(): FluidPressureDiagnostics | undefined {
            return latestPressureDiagnostics;
        },
        get gpuBytes(): number {
            return (
                positionBuffer.size +
                velocityBuffer.size +
                debugBuffer.size +
                faceAccumBuffer.size +
                faceOldBuffer.size +
                faceVelocityA.size +
                faceVelocityB.size +
                viscosityRhsBuffer.size +
                faceDeltaA.size +
                faceDeltaB.size +
                cellMarksBuffer.size +
                cellTypeBuffer.size +
                divergenceBuffer.size +
                surfaceNormalBuffer.size +
                surfaceCurvatureBuffer.size +
                pressureA.size +
                pressureB.size +
                (pageDataBuffer?.size ?? 0) +
                (pageStatusBuffer?.size ?? 0) +
                (pageLookupTexture ? pageLayout!.lookupWidth * pageLayout!.lookupHeight * 4 : 0) +
                (pageDispatchBuffer?.size ?? 0) +
                pageStatusReadbacks.reduce((sum, buffer) => sum + buffer.size, 0) +
                (pressureDiagnosticResources?.gpuBytes ?? 0) +
                (multigridResources?.gpuBytes ?? 0) +
                (liquidSdfResources?.gpuBytes ?? 0) +
                (solidFaceResources?.gpuBytes ?? 0) +
                (reseedResources?.gpuBytes ?? 0) +
                (sheetingResources?.gpuBytes ?? 0) +
                (polygonSurfaceResources?.gpuBytes ?? 0) +
                paramsBuffer.size +
                emitRangeBuffer.size +
                maxSpeedBuffer.size +
                maxSpeedReadbacks.reduce((sum, buffer) => sum + buffer.size, 0) +
                (warmupSeedPositionBuffer?.size ?? 0) +
                (warmupSeedVelocityBuffer?.size ?? 0) +
                warmupParamsBuffer.size +
                (pageDispatchConfigBuffer?.size ?? 0) +
                fluidFlowGpuBytes(flowState) +
                (diffuseBuffer?.size ?? 0) +
                (diffuseHeadBuffer?.size ?? 0) +
                (foamActiveStateBuffer?.size ?? 0) +
                (foamActiveDispatchBuffer?.size ?? 0) +
                (foamDrawIndirectBuffer?.size ?? 0) +
                (foamCountTracker?.gpuBytes ?? 0) +
                (foamParamsBuffer?.size ?? 0)
            );
        },
        get timestepDiagnostics() {
            return getFluidTimestepDiagnostics(timestepScheduler);
        },
        prepare(): void {
            if (needsLiquidSdfResources()) {
                ensureLiquidSdf();
            }
            if (fractionalSolidsEnabled) {
                ensureSolidFaces();
            }
            if (reseedParticlesEnabled) {
                ensureReseedResources();
            }
            if (particleSheetingEnabled) {
                ensureSheetingResources();
            }
            if (polygonSurfaceEnabled) {
                ensurePolygonSurfaceResources();
            }
        },
        step(encoder: GPUCommandEncoder, dt: number): void {
            pollPagedGridStatus();
            encodeConfirmedPolygonTriangleCountReadback(encoder);
            if (pagedGridOverflowed) {
                resetFluidTimestepScheduler(timestepScheduler);
                return;
            }
            if (pageLayout && pageStatusSnapshots.size >= MAX_PAGED_SPECULATIVE_FRAMES) {
                deferFluidTimestep(timestepScheduler, dt);
                return;
            }
            if (!Number.isFinite(dt) || dt <= 0) {
                return;
            }
            pollFluidActiveCount(flowState);
            pollMaxSpeed();
            pollPressureDiagnostics();
            const capillaryDt = surfaceTension > 0 ? 0.5 * Math.sqrt((dx * dx * dx) / surfaceTension) : Number.POSITIVE_INFINITY;
            const cflDt = cflNumber > 0 && lastMaxSpeed > 0 ? (cflNumber * dx) / lastMaxSpeed : Number.POSITIVE_INFINITY;
            const adaptiveMaxSubDt = Math.min(maxSubDt, capillaryDt, cflDt);
            const schedule = scheduleFluidTimestep(timestepScheduler, dt, minSubsteps, adaptiveMaxSubDt, maxSubsteps);
            if (!schedule) {
                return;
            }
            const frameDt = schedule.frameDeltaSeconds;
            const stepCount = schedule.substeps;
            const subDt = schedule.substepDeltaSeconds;
            const statusFrame = pageLayout ? ++pageStatusFrame : 0;
            if (pageLayout) {
                pageStatusSnapshots.set(statusFrame, capturePagedStepMetadata());
            }
            let releasedWarmupParticles = false;
            let warmupStart = liveCount;
            if (liveCount < initialTargetCount) {
                const previous = liveCount;
                liveCount = Math.min(initialTargetCount, liveCount + warmupStep);
                warmupStart = previous;
                releasedWarmupParticles = liveCount > previous;
            }
            const flowFrame = prepareFluidFlowFrame(flowState, frameDt);
            writeParams(subDt);
            encoder.pushDebugGroup("FLIP sim step");
            const endSimulationProfile = beginProfileSpan(encoder, "Simulation");
            const warmupGroups = releasedWarmupParticles ? Math.ceil((liveCount - warmupStart) / WORKGROUP_SIZE) : 0;
            if (releasedWarmupParticles) {
                device.queue.writeBuffer(warmupParamsBuffer, 0, new Uint32Array([warmupStart, liveCount, 0, 0]));
            }
            if (pageDispatchConfigBuffer) {
                const polygonResources = polygonSurfaceEnabled ? ensurePolygonSurfaceResources() : null;
                const cflDistance = (cflNumber > 0 ? cflNumber : 0.9) * dx;
                const redistributionDistance = reseedParticlesEnabled || particleSheetingEnabled ? 2.25 * dx : cflDistance;
                const sweepDistance = stepCount * (Math.max(cflDistance, redistributionDistance) + cflDistance + particleRadius);
                const configData = new ArrayBuffer(32);
                const configU32 = new Uint32Array(configData);
                const configF32 = new Float32Array(configData);
                configU32[0] = particleGroups;
                configU32[1] = particleGroups;
                configU32[2] = foamPoolGroups;
                configU32[3] = statusFrame;
                configF32[4] = sweepDistance;
                configU32[5] = polygonResources?.cellCount ?? 0;
                configU32[6] = polygonResources?.cubeCount ?? 0;
                configU32[7] = pageLayout!.maxPages;
                device.queue.writeBuffer(pageDispatchConfigBuffer, 0, configData);
            }
            const relaunchActive =
                flowFrame.deleteActive && (flowState.legacyEmitter !== null || flowState.config?.sinks.some((sink) => sink.enabled && sink.mode === "recycle") === true);
            encodePagedGridLookup(encoder, particleGroups, flowFrame.emitActive || relaunchActive ? flowState.emitterIds.length : 0, true, warmupGroups, statusFrame);
            if (releasedWarmupParticles) {
                dispatchMutation(encoder, "flip-warmup", warmupPipeline, warmupBindGroup!, warmupGroups, PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET);
            }
            if (flowFrame.deleteActive) {
                dispatchMutation(encoder, "flip-flow-delete", flowDeletePipeline, flowDeleteBindGroup, particleGroups, PAGE_DISPATCH_ALL_PARTICLES_OFFSET);
            }
            const canRefillCapacity = flowFrame.deleteActive || flowState.activeCount < count;
            if (flowFrame.emitActive && canRefillCapacity) {
                if (!flowOccupancyValid || flowFrame.deleteActive || releasedWarmupParticles) {
                    if (pageLayout) {
                        dispatchCells(encoder, "flip-page-clear-cells", pageClearCellsPipeline!, pageClearCellsBindGroup!);
                    } else {
                        encoder.clearBuffer(cellMarksBuffer);
                    }
                    const occupancyGroups = activePrefixValid ? Math.ceil(liveCount / WORKGROUP_SIZE) : particleGroups;
                    dispatchMutation(
                        encoder,
                        "flip-flow-mark-occupancy",
                        flowMarkOccupancyPipeline,
                        flowMarkOccupancyBindGroup,
                        occupancyGroups,
                        PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET
                    );
                    flowOccupancyValid = true;
                } else if (pageLayout) {
                    dispatchCells(encoder, "flip-page-clear-cells", pageClearCellsPipeline!, pageClearCellsBindGroup!);
                    const occupancyGroups = activePrefixValid ? Math.ceil(liveCount / WORKGROUP_SIZE) : particleGroups;
                    dispatchMutation(
                        encoder,
                        "flip-flow-mark-occupancy",
                        flowMarkOccupancyPipeline,
                        flowMarkOccupancyBindGroup,
                        occupancyGroups,
                        PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET
                    );
                }
                const candidateCount = flowEmitCandidateCount(flowFrame);
                const canAppendPrefix = activePrefixValid && !flowFrame.deleteActive && liveCount >= initialTargetCount;
                if (canAppendPrefix) {
                    const appendCount = candidateCount;
                    if (appendCount > 0) {
                        liveCount = Math.min(count, liveCount + appendCount);
                        device.queue.writeBuffer(emitRangeBuffer, 0, new Uint32Array([0, appendCount, 0, 0]));
                        dispatchMutation(
                            encoder,
                            "flip-flow-emit-append",
                            flowEmitAppendPipeline,
                            flowEmitAppendBindGroup,
                            Math.ceil(appendCount / WORKGROUP_SIZE),
                            PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET
                        );
                    }
                } else {
                    activePrefixValid = false;
                    dispatchMutation(encoder, "flip-flow-emit", flowEmitPipeline, flowEmitBindGroup, particleGroups, PAGE_DISPATCH_ALL_PARTICLES_OFFSET);
                }
            }
            const activeParticleGroups = activePrefixValid ? Math.ceil(liveCount / WORKGROUP_SIZE) : particleGroups;
            for (let step = 0; step < stepCount; step++) {
                if (forcePasses) {
                    for (let i = 0; i < forcePasses.commands.length; i++) {
                        const force = forcePasses.commands[i]!;
                        dispatchMutation(encoder, "flip-force", force.pipeline, force.bindGroup, activeParticleGroups, PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET);
                    }
                }
                encodeParticleGridClassification(encoder, activeParticleGroups);
                if (usesLiquidSdf()) {
                    encodeLiquidSdf(encoder);
                }
                if (fractionalSolidsEnabled) {
                    const resources = ensureSolidFaces();
                    dispatchFaces(encoder, "flip-solid-face-geometry", resources.pipeline, resources.bindGroup);
                }
                dispatchFaces(encoder, "flip-normalize", normalizePipeline, normalizeBindGroup);
                if (kinematicViscosity > 0 && viscosityIterations > 0) {
                    dispatchFaces(encoder, "flip-viscosity-rhs", viscosityRhsPipeline, viscosityRhsBindGroup);
                    for (let iteration = 0; iteration < viscosityIterations; iteration++) {
                        dispatchFaces(encoder, "flip-viscosity", viscosityPipeline, iteration % 2 === 0 ? viscosityABindGroup : viscosityBBindGroup);
                    }
                    if (viscosityIterations % 2 !== 0) {
                        encoder.copyBufferToBuffer(faceVelocityB, 0, faceVelocityA, 0, faceBytes);
                    }
                }
                if (surfaceTension > 0 || (foamEnabled && step === stepCount - 1)) {
                    dispatchCells(encoder, "flip-surface-normal", surfaceNormalPipeline, surfaceNormalBindGroup);
                    dispatchCells(encoder, "flip-surface-curvature", surfaceCurvaturePipeline, surfaceCurvatureBindGroup);
                    if (surfaceTension > 0) {
                        dispatchFaces(encoder, "flip-surface-force", surfaceForcePipeline, surfaceForceBindGroup);
                    }
                }
                dispatchCells(encoder, "flip-divergence", divergencePipeline, divergenceBindGroup);
                if (pressureSolver === "multigrid") {
                    encodeMultigridPressure(encoder);
                } else {
                    for (let iteration = 0; iteration < pressureIterations; iteration++) {
                        dispatchCells(encoder, "flip-pressure", pressurePipeline, pressureCurrentIsA ? pressureABindGroup : pressureBBindGroup);
                        pressureCurrentIsA = !pressureCurrentIsA;
                    }
                }
                const samplePressure = step === stepCount - 1 && (pressureDiagnosticsEnabled || pressureTolerance > 0);
                if (samplePressure) {
                    const resources = ensurePressureDiagnosticResources();
                    encoder.clearBuffer(resources.output);
                    dispatchCells(encoder, "flip-pressure-diagnostic-residual", resources.residualPipeline, pressureCurrentIsA ? resources.residualA : resources.residualB);
                    dispatchCells(encoder, "flip-pressure-diagnostic-reduce", resources.reducePipeline, resources.reduce);
                }
                dispatchFaces(encoder, "flip-project", projectPipeline, pressureCurrentIsA ? projectABindGroup : projectBBindGroup);
                if (samplePressure) {
                    const resources = pressureDiagnosticResources!;
                    dispatchCells(encoder, "flip-post-project-divergence", divergencePipeline, divergenceBindGroup);
                    dispatchCells(encoder, "flip-post-divergence-diagnostic", resources.postDivergencePipeline, resources.postDivergence);
                    encodePressureDiagnosticReadback(encoder, resources);
                }
                for (let layer = 0; layer < EXTRAPOLATION_LAYERS; layer++) {
                    dispatchFaces(encoder, "flip-extrapolate", extrapolatePipeline, layer % 2 === 0 ? extrapolateABindGroup : extrapolateBBindGroup);
                }
                dispatchMutation(encoder, "flip-g2p", g2pPipeline, g2pBindGroup, activeParticleGroups, PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET);
                reseedSubstep++;
                if (reseedParticlesEnabled && reseedSubstep % reseedInterval === 0) {
                    encodeReseed(encoder, activeParticleGroups);
                }
                sheetingSubstep++;
                if (particleSheetingEnabled && sheetingSubstep % sheetingInterval === 0) {
                    encodeSheeting(encoder);
                }
            }
            dispatchMutation(encoder, "flip-speed-reset", speedResetPipeline, speedResetBindGroup, 1, PAGE_DISPATCH_SINGLE_OFFSET);
            dispatchMutation(encoder, "flip-speed-reduce", speedReducePipeline, speedReduceBindGroup, activeParticleGroups, PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET);
            encodeFluidActiveCountReadback(flowState, encoder);
            encodeMaxSpeedReadback(encoder);
            endSimulationProfile?.();
            if (polygonSurfaceEnabled) {
                const endSurfaceProfile = beginProfileSpan(encoder, "Surface");
                encodeLiquidSdf(encoder, "Surface");
                encodePolygonSurface(encoder);
                polygonSurfaceRefreshPending = false;
                endSurfaceProfile?.();
            }
            if (foamEnabled && foamEmitBindGroup && foamUpdateBindGroup) {
                const endFoamProfile = beginProfileSpan(encoder, "Foam gen");
                foamF32[11] = frameDt;
                foamU32[14] = foamSeed++;
                device.queue.writeBuffer(foamParamsBuffer!, 0, foamData);
                encoder.pushDebugGroup("foam");
                dispatchMutation(encoder, "flip-foam-emit", foamEmitPipeline!, foamEmitBindGroup, activeParticleGroups, PAGE_DISPATCH_ACTIVE_PARTICLES_OFFSET);
                if (foamActiveParticles) {
                    dispatchMutation(encoder, "flip-foam-active-prepare", foamActivePreparePipeline!, foamActivePrepareBindGroup!, 1, PAGE_DISPATCH_SINGLE_OFFSET);
                    if (foamActiveOverflowGatePipeline && foamActiveOverflowGateBindGroup) {
                        dispatch(encoder, "flip-foam-active-overflow-gate", foamActiveOverflowGatePipeline, foamActiveOverflowGateBindGroup, 1);
                    }
                    dispatchIndirect(encoder, "flip-foam-update", foamUpdatePipeline!, foamUpdateBindGroup, foamActiveDispatchBuffer!);
                    dispatchMutation(encoder, "flip-foam-active-finish", foamActiveFinishPipeline!, foamActiveFinishBindGroup!, 1, PAGE_DISPATCH_SINGLE_OFFSET);
                    foamActiveSide = 1 - foamActiveSide;
                    diffusePool = {
                        buffer: diffuseBuffer!,
                        headBuffer: foamActiveStateBuffer!,
                        capacity: foamCapacity,
                        activeIndices: foamActiveStateBuffer!,
                        activeIndicesOffset: foamActiveListOffset(foamCapacity, foamActiveSide),
                        drawIndirect: foamDrawIndirectBuffer!,
                        get counts() {
                            return foamCountTracker!.counts;
                        },
                    };
                } else {
                    dispatchMutation(encoder, "flip-foam-update", foamUpdatePipeline!, foamUpdateBindGroup, foamPoolGroups, PAGE_DISPATCH_FOAM_OFFSET);
                }
                foamCountTracker?.encode(encoder, foamPoolGroups, foamActiveParticles ? foamActiveDispatchBuffer! : undefined);
                encoder.popDebugGroup();
                endFoamProfile?.();
            }
            if (pageLayout) {
                encodePagedGridStatusReadback(encoder);
            }
            encoder.popDebugGroup();
        },
        get diffuse(): DiffusePool | undefined {
            return diffusePool;
        },
        get polygonSurface(): FluidPolygonSurface | undefined {
            return polygonSurfaceEnabled ? polygonSurfaceResources?.surface : undefined;
        },
        refreshPolygonSurface(encoder: GPUCommandEncoder): void {
            pollPagedGridStatus();
            encodeConfirmedPolygonTriangleCountReadback(encoder);
            if (!polygonSurfaceEnabled) {
                return;
            }
            if (pagedGridOverflowed || (pageLayout && pageStatusSnapshots.size >= MAX_PAGED_SPECULATIVE_FRAMES)) {
                return;
            }
            if (!polygonSurfaceRefreshPending) {
                if (polygonSurfaceResources) {
                    pollPolygonTriangleCount(polygonSurfaceResources, encoder);
                }
                return;
            }
            const endSurfaceProfile = beginProfileSpan(encoder, "Surface");
            if (pageLayout) {
                const statusFrame = ++pageStatusFrame;
                pageStatusSnapshots.set(statusFrame, capturePagedStepMetadata());
                const activeParticleGroups = activePrefixValid ? Math.ceil(liveCount / WORKGROUP_SIZE) : particleGroups;
                const resources = ensurePolygonSurfaceResources();
                const configData = new ArrayBuffer(32);
                const configU32 = new Uint32Array(configData);
                configU32[0] = activeParticleGroups;
                configU32[1] = particleGroups;
                configU32[2] = foamPoolGroups;
                configU32[3] = statusFrame;
                configU32[5] = resources.cellCount;
                configU32[6] = resources.cubeCount;
                configU32[7] = pageLayout!.maxPages;
                device.queue.writeBuffer(pageDispatchConfigBuffer!, 0, configData);
                encodePagedGridLookup(encoder, activeParticleGroups, 0, true, 0, statusFrame);
            }
            const activeParticleGroups = activePrefixValid ? Math.ceil(liveCount / WORKGROUP_SIZE) : particleGroups;
            encodeParticleGridClassification(encoder, activeParticleGroups);
            encodeLiquidSdf(encoder, "Surface");
            encodePolygonSurface(encoder);
            polygonSurfaceRefreshPending = false;
            if (pageLayout) {
                encodePagedGridStatusReadback(encoder);
            }
            endSurfaceProfile?.();
        },
        reset(): void {
            resetFluidTimestepScheduler(timestepScheduler);
            if (polygonSurfaceResources) {
                polygonSurfaceResources.historyValid = false;
                polygonSurfaceResources.triangleCountReadback.generation++;
                polygonSurfaceResources.triangleCountReadback.frame = 0;
                polygonSurfaceResources.triangleCountReadback.latest = undefined;
            }
            seed();
            polygonSurfaceRefreshPending = polygonSurfaceEnabled;
        },
        setParam(key: string, value: number): void {
            switch (key) {
                case "gravity":
                    gravity = value;
                    break;
                case "flipRatio":
                    flipRatio = Math.min(1, Math.max(0, value));
                    break;
                case "pressureIterations":
                    pressureIterations = Math.max(1, Math.round(value));
                    retirePressureDiagnostics();
                    break;
                case "pressureRelaxation":
                    pressureRelaxation = Math.min(1, Math.max(0.01, value));
                    retirePressureDiagnostics();
                    break;
                case "pressureSolver":
                    pressureSolver = value >= 0.5 ? "multigrid" : "jacobi";
                    retirePressureDiagnostics();
                    if (pressureSolver === "multigrid") {
                        ensureMultigrid();
                    }
                    break;
                case "multigridCycles":
                    multigridCycles = Math.min(8, Math.max(1, Math.round(value)));
                    retirePressureDiagnostics();
                    break;
                case "pressureTolerance":
                    pressureTolerance = Math.max(0, value);
                    retirePressureDiagnostics();
                    if (pressureTolerance > 0) {
                        ensurePressureDiagnosticResources();
                    } else if (!pressureDiagnosticsEnabled) {
                        destroyPressureDiagnosticResources();
                    }
                    break;
                case "pressureDiagnostics":
                    pressureDiagnosticsEnabled = value >= 0.5;
                    retirePressureDiagnostics();
                    if (pressureDiagnosticsEnabled) {
                        ensurePressureDiagnosticResources();
                    } else if (pressureTolerance === 0) {
                        destroyPressureDiagnosticResources();
                    }
                    break;
                case "velocityDamping":
                    velocityDamping = Math.max(0, value);
                    break;
                case "kinematicViscosity":
                    kinematicViscosity = Math.max(0, value);
                    break;
                case "viscosityIterations":
                    viscosityIterations = Math.max(1, Math.round(value));
                    break;
                case "surfaceTension":
                    surfaceTension = Math.max(0, value);
                    break;
                case "liquidSdf":
                    if (liquidSdfEnabled === value >= 0.5) {
                        break;
                    }
                    {
                        const wasActive = usesLiquidSdf();
                        liquidSdfEnabled = value >= 0.5;
                        if (wasActive && !needsLiquidSdfResources() && liquidSdfResources) {
                            liquidSdfResources.sdfA.destroy();
                            liquidSdfResources.sdfB.destroy();
                            liquidSdfResources = null;
                        }
                    }
                    rebuildQualityPipelines();
                    break;
                case "ghostFluid":
                    if (ghostFluidEnabled === value >= 0.5) {
                        break;
                    }
                    ghostFluidEnabled = value >= 0.5;
                    if (usesLiquidSdf()) {
                        rebuildQualityPipelines();
                    }
                    break;
                case "fractionalSolids":
                    if (fractionalSolidsEnabled === value >= 0.5) {
                        break;
                    }
                    fractionalSolidsEnabled = value >= 0.5;
                    if (!fractionalSolidsEnabled && solidFaceResources) {
                        solidFaceResources.geometry.destroy();
                        solidFaceResources = null;
                    }
                    rebuildQualityPipelines();
                    break;
                case "movingSolidBoundaries":
                    movingSolidBoundariesEnabled = value >= 0.5;
                    break;
                case "reseedParticles":
                    reseedParticlesEnabled = value >= 0.5;
                    if (reseedParticlesEnabled) {
                        ensureReseedResources();
                        activePrefixValid = false;
                    } else if (reseedResources) {
                        reseedResources.state.destroy();
                        reseedResources = null;
                    }
                    break;
                case "reseedMinParticles":
                    reseedMinParticles = Math.max(1, Math.round(value));
                    reseedTargetParticles = Math.max(reseedMinParticles, reseedTargetParticles);
                    reseedMaxParticles = Math.max(reseedTargetParticles, reseedMaxParticles);
                    break;
                case "reseedTargetParticles":
                    reseedTargetParticles = Math.max(reseedMinParticles, Math.round(value));
                    reseedMaxParticles = Math.max(reseedTargetParticles, reseedMaxParticles);
                    break;
                case "reseedMaxParticles":
                    reseedMaxParticles = Math.max(reseedTargetParticles, Math.round(value));
                    break;
                case "reseedInterval":
                    reseedInterval = Math.max(1, Math.round(value));
                    break;
                case "particleSheeting": {
                    const wasActive = usesLiquidSdf();
                    particleSheetingEnabled = value >= 0.5;
                    if (particleSheetingEnabled) {
                        ensureSheetingResources();
                        activePrefixValid = false;
                    } else if (sheetingResources) {
                        sheetingResources.state.destroy();
                        sheetingResources = null;
                    }
                    if (wasActive !== usesLiquidSdf()) {
                        if (!needsLiquidSdfResources() && liquidSdfResources) {
                            liquidSdfResources.sdfA.destroy();
                            liquidSdfResources.sdfB.destroy();
                            liquidSdfResources = null;
                        }
                        rebuildQualityPipelines();
                    }
                    break;
                }
                case "sheetingStrength":
                    sheetingStrength = Math.min(1, Math.max(0.05, value));
                    break;
                case "sheetingInterval":
                    sheetingInterval = Math.max(1, Math.round(value));
                    break;
                case "polygonSurface": {
                    const wasNeeded = needsLiquidSdfResources();
                    polygonSurfaceEnabled = value >= 0.5;
                    if (polygonSurfaceEnabled) {
                        ensurePolygonSurfaceResources();
                        polygonSurfaceRefreshPending = true;
                    } else {
                        destroyPolygonSurfaceResources();
                        polygonSurfaceRefreshPending = false;
                    }
                    if (wasNeeded && !needsLiquidSdfResources() && liquidSdfResources) {
                        liquidSdfResources.sdfA.destroy();
                        liquidSdfResources.sdfB.destroy();
                        liquidSdfResources = null;
                    }
                    break;
                }
                case "polygonReconstructionMultiplier": {
                    const nextMultiplier = Math.round(Math.min(2, Math.max(1, value)) * 4) / 4;
                    if (nextMultiplier === polygonReconstructionMultiplier) {
                        break;
                    }
                    const wasNeeded = needsLiquidSdfResources();
                    destroyPolygonSurfaceResources();
                    polygonReconstructionMultiplier = nextMultiplier;
                    if (polygonSurfaceEnabled) {
                        ensurePolygonSurfaceResources();
                        polygonSurfaceRefreshPending = true;
                    }
                    if (wasNeeded && !needsLiquidSdfResources() && liquidSdfResources) {
                        liquidSdfResources.sdfA.destroy();
                        liquidSdfResources.sdfB.destroy();
                        liquidSdfResources = null;
                    }
                    break;
                }
                case "restitution":
                    restitution = Math.min(1, Math.max(0, value));
                    break;
                case "minSubsteps":
                    minSubsteps = Math.max(1, Math.round(value));
                    maxSubsteps = Math.max(maxSubsteps, minSubsteps);
                    break;
                case "maxSubsteps":
                    maxSubsteps = Math.max(minSubsteps, Math.round(value));
                    break;
                case "cflNumber":
                    cflNumber = Math.max(0, value);
                    break;
                case "maxSubDtMs":
                    maxSubDt = Math.max(1e-4, value / 1000);
                    break;
            }
        },
        setSceneSdf(scene: SceneSdfSpec | null): void {
            sceneSdf = scene;
            classifyPipeline = pipeline("flip-classify", buildClassifyWgsl(scene, fractionalSolidsEnabled));
            g2pPipeline = pipeline("flip-g2p", buildG2pWgsl(scene));
            classifyBindGroup = buildClassifyBindGroup(classifyPipeline, scene);
            g2pBindGroup = buildG2pBindGroup(g2pPipeline, scene);
            if (fractionalSolidsEnabled) {
                const resources = ensureSolidFaces();
                resources.pipeline = pipeline("flip-solid-face-geometry", buildSolidFaceGeometryWgsl(scene));
                resources.bindGroup = buildSolidFaceBindGroup(resources.geometry, resources.pipeline, scene);
            }
            if (reseedResources) {
                reseedResources.emitPipeline = pipeline("flip-reseed-emit", buildReseedEmitWgsl(scene));
                reseedResources.emitBindGroup = buildReseedEmitBindGroup(reseedResources.emitPipeline, reseedResources.state, scene);
            }
            if (sheetingResources) {
                sheetingResources.emitPipeline = pipeline("flip-sheeting-emit", buildSheetingEmitWgsl(scene));
                sheetingResources.emitBindGroup = buildSheetingEmitBindGroup(sheetingResources.emitPipeline, sheetingResources.state, scene);
            }
            divergencePipeline = pipeline("flip-divergence", buildDivergenceWgsl(scene, fractionalSolidsEnabled));
            divergenceBindGroup = buildDivergenceBindGroup(divergencePipeline, scene);
        },
        setEmitters(config: EmitterConfig | null): void {
            flowSeedsInitialParticles = false;
            configuredFlowBreaksPrefix = config !== null;
            activePrefixValid &&= !configuredFlowBreaksPrefix;
            setFluidFlowConfig(flowState, legacyEmitterConfigToFluidFlow(config, count));
        },
        setFlow(config: FluidFlowConfig | null): void {
            flowSeedsInitialParticles = true;
            configuredFlowBreaksPrefix = config?.sinks.some((sink) => sink.enabled) === true;
            activePrefixValid &&= !configuredFlowBreaksPrefix;
            setFluidFlowConfig(flowState, config);
        },
        updateFlowEmitter(emitter: FluidEmitter): void {
            updateFluidFlowEmitter(flowState, emitter);
        },
        setSpawn(min: [number, number, number], max: [number, number, number], accept?: ((x: number, y: number, z: number) => boolean) | null): void {
            spawnMin[0] = min[0];
            spawnMin[1] = min[1];
            spawnMin[2] = min[2];
            spawnMax[0] = max[0];
            spawnMax[1] = max[1];
            spawnMax[2] = max[2];
            spawnAccept = accept ?? null;
        },
        setWarmup(frames: number): void {
            warmupFrames = Math.max(0, Math.floor(frames));
            warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(initialTargetCount / warmupFrames)) : Math.max(1, initialTargetCount);
        },
        setForceField(spec: ForceFieldSpec | null): void {
            if (!spec && !forcePasses) {
                return;
            }
            forcePasses ??= createFluidForcePasses(buildForceWgsl, (source) => pipeline("flip-force", source), buildForceBindGroup);
            updateFluidForcePasses(forcePasses, spec);
        },
        setFoam(config: FoamConfig | null): void {
            if (!config) {
                foamEnabled = false;
                clearFoamPool("flip-foam-off-clear");
                return;
            }
            ensureFoam(config);
            foamEnabled = true;
        },
        setProfiler(profiler: FluidProfiler | null): void {
            simProfiler = profiler;
        },
        dispose(): void {
            disposed = true;
            flipSimTransferEndpoints?.delete(sim);
            pageStatusGeneration++;
            positionBuffer.destroy();
            velocityBuffer.destroy();
            debugBuffer.destroy();
            faceAccumBuffer.destroy();
            faceOldBuffer.destroy();
            faceVelocityA.destroy();
            faceVelocityB.destroy();
            viscosityRhsBuffer.destroy();
            faceDeltaA.destroy();
            faceDeltaB.destroy();
            cellMarksBuffer.destroy();
            cellTypeBuffer.destroy();
            divergenceBuffer.destroy();
            surfaceNormalBuffer.destroy();
            surfaceCurvatureBuffer.destroy();
            pressureA.destroy();
            pressureB.destroy();
            pageDataBuffer?.destroy();
            pageStatusBuffer?.destroy();
            pageDispatchBuffer?.destroy();
            pageDispatchConfigBuffer?.destroy();
            pageLookupTexture?.destroy();
            for (const readback of pageStatusReadbacks) {
                readback.destroy();
            }
            destroyPressureDiagnosticResources();
            liquidSdfResources?.sdfA.destroy();
            liquidSdfResources?.sdfB.destroy();
            solidFaceResources?.geometry.destroy();
            reseedResources?.state.destroy();
            sheetingResources?.state.destroy();
            destroyPolygonSurfaceResources();
            if (multigridResources) {
                for (let index = 0; index < multigridResources.levels.length; index++) {
                    const level = multigridResources.levels[index]!;
                    level.rhs.destroy();
                    level.residual.destroy();
                    level.fluidFraction.destroy();
                    level.params.destroy();
                    if (index > 0) {
                        level.cellTypes.destroy();
                        level.pressureA.destroy();
                        level.pressureB.destroy();
                    }
                }
            }
            paramsBuffer.destroy();
            emitRangeBuffer.destroy();
            maxSpeedBuffer.destroy();
            for (const buffer of maxSpeedReadbacks) {
                buffer.destroy();
            }
            diffuseBuffer?.destroy();
            diffuseHeadBuffer?.destroy();
            foamActiveStateBuffer?.destroy();
            foamActiveDispatchBuffer?.destroy();
            foamDrawIndirectBuffer?.destroy();
            foamCountTracker?.dispose();
            foamParamsBuffer?.destroy();
            warmupSeedPositionBuffer?.destroy();
            warmupSeedVelocityBuffer?.destroy();
            warmupParamsBuffer.destroy();
            disposeFluidFlowState(flowState);
        },
    };
    const transferLayoutKey = `${boundsMin.join(",")}:${boundsMax.join(",")}:${dx}`;
    getFlipSimTransferEndpoints().set(sim, {
        layoutKey: transferLayoutKey,
        capacity: count,
        positionBuffer,
        velocityBuffer,
        debugBuffer,
        lifecycleBuffer: flowState.lifecycleBuffer,
        diffuse: () =>
            diffuseBuffer && diffuseHeadBuffer
                ? {
                      capacity: foamCapacity,
                      active: foamActiveParticles,
                      buffer: diffuseBuffer,
                      denseHead: diffuseHeadBuffer,
                      activeState: foamActiveStateBuffer,
                      activeDispatch: foamActiveDispatchBuffer,
                      drawIndirect: foamDrawIndirectBuffer,
                  }
                : null,
        polygon: () => {
            const resources = polygonSurfaceEnabled ? ensurePolygonSurfaceResources() : null;
            return resources
                ? {
                      stabilizedSdf: resources.stabilizedSdf,
                      historyValid: resources.historyValid,
                  }
                : null;
        },
        capture: () => ({
            liveCount,
            initialTargetCount,
            warmupStep,
            activePrefixValid,
            initialEmitterParticleCounts: new Map(resetInitialEmitterParticleCounts),
            reseedSubstep,
            sheetingSubstep,
            flowActiveCount: flowState.activeCount,
            flowFrameSeed: flowState.frameSeed,
            flowEmitterCursorId: flowState.emitterIds[flowState.emitterCursor],
            flowEmitterCarries: captureFluidFlowCarries(flowState.emitterIds, flowState.emitterCarries),
            flowElapsedSeconds: flowState.elapsedSeconds,
            flowSinkCarries: captureFluidFlowCarries(flowState.sinkIds, flowState.sinkCarries),
            foamActiveSide,
            foamCounts: foamCountTracker?.counts ? { ...foamCountTracker.counts } : undefined,
            foamSeed,
        }),
        restore: (metadata, diffuseTransferred, polygonHistoryTransferred) => {
            liveCount = metadata.liveCount;
            initialTargetCount = metadata.initialTargetCount;
            warmupStep = metadata.warmupStep;
            activePrefixValid = metadata.activePrefixValid;
            resetInitialEmitterParticleCounts = new Map(metadata.initialEmitterParticleCounts);
            reseedSubstep = metadata.reseedSubstep;
            sheetingSubstep = metadata.sheetingSubstep;
            flowOccupancyValid = false;
            flowState.activeCount = metadata.flowActiveCount;
            flowState.frameSeed = metadata.flowFrameSeed;
            const emitterCursor = metadata.flowEmitterCursorId === undefined ? -1 : flowState.emitterIds.indexOf(metadata.flowEmitterCursorId);
            flowState.emitterCursor = emitterCursor >= 0 ? emitterCursor : 0;
            restoreFluidFlowCarries(flowState.emitterIds, flowState.emitterCarries, metadata.flowEmitterCarries);
            flowState.elapsedSeconds = metadata.flowElapsedSeconds;
            restoreFluidFlowCarries(flowState.sinkIds, flowState.sinkCarries, metadata.flowSinkCarries);
            if (diffuseTransferred) {
                foamSeed = metadata.foamSeed;
                foamActiveSide = metadata.foamActiveSide;
                foamCountTracker?.restore(metadata.foamCounts);
                if (diffusePool && foamActiveParticles) {
                    diffusePool = {
                        ...diffusePool,
                        activeIndicesOffset: foamActiveListOffset(foamCapacity, foamActiveSide),
                    };
                }
            }
            if (polygonSurfaceResources) {
                polygonSurfaceResources.historyValid = polygonHistoryTransferred;
            }
            polygonSurfaceRefreshPending = polygonSurfaceEnabled;
        },
    });
    return sim;
}
