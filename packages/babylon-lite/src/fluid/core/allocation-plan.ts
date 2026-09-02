import { FLIP_DEFAULT_PAGE_CAPACITY, FLIP_PAGE_CELLS, pagedFlipStorageCounts, resolveFlipPageLayout } from "../solvers/flip-sim.js";
import type { FlipPressureSolver, FlipQualityFeatures } from "../solvers/flip-sim.js";
import { FLIP_PAGE_DISPATCH_BYTES, FLIP_PAGE_DISPATCH_CONFIG_BYTES, FLIP_PAGE_STATUS_BYTES, FLIP_PAGE_STATUS_READBACK_BYTES } from "../solvers/flip-layout.js";
import { FLUID_FLOW_BYTES, FLUID_FLOW_COUNTER_BYTES, FOAM_BYTES, foamActiveStateBytes } from "./sim-common.js";

export type FluidAllocationMethod = "FLIP" | "PBF" | "MLS-MPM" | "PB-MPM";
export type FluidAllocationResourceKind = "buffer" | "texture";
export type FluidAllocationBinding = "storage" | "uniform" | "none";

export interface FluidDeviceLimitsSnapshot {
    maxStorageBufferBindingSize?: number;
    maxBufferSize?: number;
    maxTextureDimension2D?: number;
}

export interface FluidAllocationFoam {
    enabled: boolean;
    activeParticles?: boolean;
    poolScale?: number;
    poolCapMax?: number;
    capacity?: number;
}

export interface FluidAllocationFlipWarmup {
    /** Active particles immediately after reset. */
    initialLiveCount: number;
    /** Particles that reset intends to release across the warm-up. */
    initialTargetCount: number;
}

export interface FluidAllocationFlipWarmupPlan extends FluidAllocationFlipWarmup {
    readonly seedBuffersAllocated: boolean;
}

export interface FluidAllocationPlanInput {
    method: FluidAllocationMethod;
    particleCount: number;
    gridDim: readonly [number, number, number];
    pressureSolver?: FlipPressureSolver;
    quality?: FlipQualityFeatures;
    /** Requested FLIP polygon extraction capacity. */
    surfaceMaxTriangles?: number;
    /** Generic sparse-grid state. FLIP also accepts the legacy quality.pagedGrid fields. */
    pagedGrid?: boolean;
    pagedGridMaxPages?: number;
    /** MLS-MPM dense active-block execution. Paging always enables it. */
    activeBlocks?: boolean;
    foam?: FluidAllocationFoam;
    /** Initial FLIP reset state. Omit when reset starts with its full target population. */
    flipWarmup?: FluidAllocationFlipWarmup;
    limits?: FluidDeviceLimitsSnapshot;
    previousSteadyBytes?: number;
    maxParticleBudget?: number;
}

export interface FluidAllocationResource {
    readonly name: string;
    readonly kind: FluidAllocationResourceKind;
    readonly binding: FluidAllocationBinding;
    readonly bytes: number;
    readonly width?: number;
    readonly height?: number;
}

export interface FluidAllocationPages {
    readonly numBlocks: number;
    readonly requestedMaxPages: number;
    readonly maximumPageCapacity: number;
    readonly maxPages: number;
    readonly storageCells: number;
    readonly storageFaces: number;
    readonly lookupWidth: number;
    readonly lookupHeight: number;
    readonly lookupWords: number;
    readonly lookupPaddedWords: number;
    readonly pageDispatchBytes: number;
}

export interface FluidAllocationPlan {
    readonly method: FluidAllocationMethod;
    readonly dimensions: {
        readonly gridDim: [number, number, number];
        readonly cellCount: number;
        readonly faceCount: number;
        readonly particleCount: number;
    };
    readonly pages?: FluidAllocationPages;
    readonly flipWarmup?: FluidAllocationFlipWarmupPlan;
    readonly foamCapacity: number;
    readonly polygonTriangleCapacity: number;
    readonly steadyBytes: number;
    readonly rebuildPeakBytes: number;
    readonly resources: FluidAllocationResource[];
    readonly errors: string[];
}

export interface FluidAllocationCapabilities {
    readonly method: FluidAllocationMethod;
    readonly supportsPagedGrid: boolean;
    readonly supportsStaticByteEstimate: boolean;
    readonly supportsPolygonSurface: boolean;
    readonly supportsPressureDiagnostics: boolean;
}

export interface FluidBufferLimits {
    readonly maxStorageBufferBindingSize: number;
    readonly maxBufferSize: number;
}

export interface FluidFlipPaging {
    readonly enabled: boolean;
    readonly maxPages: number;
}

export interface FluidGridCompatibility {
    readonly compatible: boolean;
    readonly code?: "axis-limit" | "storage-binding-limit";
    readonly message?: string;
    readonly gridDim: readonly [number, number, number];
    readonly requiredBytes?: number;
    readonly availableBytes?: number;
}

export const PBF_PARTICLE_BYTES_PER_SLOT = 16;
export const FLIP_PARTICLE_BYTES_PER_SLOT = 16;
export const MLS_MPM_PARTICLE_BYTES_PER_SLOT = 80;
export const PB_MPM_PARTICLE_BYTES_PER_SLOT = 144;
export const DEFAULT_PARTICLE_BYTES_PER_SLOT = 144;
export const FLUID_GRID_MAX_AXIS_CELLS = 2048;
export const AQUANOVA_COMBINED_PARTICLE_CAPACITY = 600_000;

const DEFAULT_MAX_BUFFER_SIZE = 1 << 30;
const DEFAULT_MAX_TEXTURE_DIMENSION_2D = 8192;
const SCAN_WORKGROUP_SIZE = 256;
const WORKGROUP_SIZE = 64;
const MAX_WORKGROUPS = 65535;
const MPM_PAGE_SIZE = 4;
const MPM_PAGE_BYTES = MPM_PAGE_SIZE ** 3 * 16;
const FLIP_PARAMS_BYTES = 160;
const FLIP_MULTIGRID_PARAMS_BYTES = 48;
const FLIP_PRESSURE_DIAGNOSTIC_BYTES = 24;
const FLIP_RESEED_HEADER_BYTES = 32;
const FLIP_SHEETING_HEADER_BYTES = 16;
const FLIP_DEFAULT_SURFACE_TRIANGLE_CAPACITY = 1_000_000;
const FLIP_SURFACE_VERTEX_STRIDE = 32;
const PBF_SIM_BYTES = 160;
const PBF_GRID_BYTES = 32;
const MLS_PARAMS_BYTES = 256;
const PBMPM_PARAMS_BYTES = 128;
const COMBINED_RENDER_PARTICLE_BYTES = 16;

interface NormalizedLimits {
    readonly maxStorageBufferBindingSize: number;
    readonly maxBufferSize: number;
    readonly maxTextureDimension2D: number;
}

interface ResourceLedger {
    readonly resources: FluidAllocationResource[];
    buffer(name: string, bytes: number, binding?: FluidAllocationBinding): void;
    texture(name: string, width: number, height: number, bytesPerTexel: number): void;
}

export function fluidAllocationCapabilities(method: FluidAllocationMethod): FluidAllocationCapabilities {
    const flip = method === "FLIP";
    return {
        method,
        supportsPagedGrid: flip || method === "MLS-MPM",
        supportsStaticByteEstimate: true,
        supportsPolygonSurface: flip,
        supportsPressureDiagnostics: flip,
    };
}

function finiteInteger(value: number | undefined, fallback: number, minimum: number, errors: string[], name: string): number {
    const safeFallback = Number.isFinite(fallback) ? Math.min(Number.MAX_SAFE_INTEGER, Math.max(minimum, Math.floor(fallback))) : minimum;
    if (value === undefined) {
        return safeFallback;
    }
    if (!Number.isFinite(value)) {
        errors.push(`${name} must be finite (got ${String(value)}); using ${safeFallback}.`);
        return safeFallback;
    }
    const normalized = Math.min(Number.MAX_SAFE_INTEGER, Math.max(minimum, Math.floor(value)));
    if (normalized !== value) {
        errors.push(`${name} ${value} normalized to ${normalized}.`);
    }
    return normalized;
}

function finitePositive(value: number | undefined, fallback: number, errors: string[], name: string): number {
    if (value === undefined) {
        return fallback;
    }
    if (!Number.isFinite(value) || value <= 0) {
        errors.push(`${name} must be a positive finite number (got ${String(value)}); using ${fallback}.`);
        return fallback;
    }
    return value;
}

function finiteNonNegative(value: number | undefined, fallback: number, errors: string[], name: string): number {
    if (value === undefined) {
        return fallback;
    }
    if (!Number.isFinite(value)) {
        errors.push(`${name} must be finite (got ${String(value)}); using ${fallback}.`);
        return fallback;
    }
    const normalized = Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, value));
    if (normalized !== value) {
        errors.push(`${name} ${value} normalized to ${normalized}.`);
    }
    return normalized;
}

function normalizeLimits(input: FluidDeviceLimitsSnapshot | undefined, errors: string[]): NormalizedLimits {
    return {
        maxStorageBufferBindingSize: finiteInteger(input?.maxStorageBufferBindingSize, DEFAULT_MAX_BUFFER_SIZE, 1, errors, "limits.maxStorageBufferBindingSize"),
        maxBufferSize: finiteInteger(input?.maxBufferSize, DEFAULT_MAX_BUFFER_SIZE, 1, errors, "limits.maxBufferSize"),
        maxTextureDimension2D: finiteInteger(input?.maxTextureDimension2D, DEFAULT_MAX_TEXTURE_DIMENSION_2D, 1, errors, "limits.maxTextureDimension2D"),
    };
}

function normalizeGridDim(gridDim: readonly [number, number, number], errors: string[]): [number, number, number] {
    return gridDim.map((value, axis) => {
        const normalized = finiteInteger(value, 4, 4, errors, `gridDim[${axis}]`);
        if (normalized > FLUID_GRID_MAX_AXIS_CELLS) {
            errors.push(`gridDim[${axis}] ${normalized} normalized to ${FLUID_GRID_MAX_AXIS_CELLS}.`);
            return FLUID_GRID_MAX_AXIS_CELLS;
        }
        return normalized;
    }) as [number, number, number];
}

function cellAndFaceCounts(dim: readonly [number, number, number]): { cellCount: number; faceCount: number } {
    const [nx, ny, nz] = dim;
    return {
        cellCount: nx * ny * nz,
        faceCount: (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1),
    };
}

function createLedger(): ResourceLedger {
    const resources: FluidAllocationResource[] = [];
    return {
        resources,
        buffer(name, bytes, binding = "storage"): void {
            resources.push({ name, kind: "buffer", binding, bytes: Math.max(0, Math.ceil(bytes)) });
        },
        texture(name, width, height, bytesPerTexel): void {
            resources.push({
                name,
                kind: "texture",
                binding: "none",
                bytes: Math.max(0, Math.ceil(width * height * bytesPerTexel)),
                width,
                height,
            });
        },
    };
}

function addCommonResources(ledger: ResourceLedger, count: number): void {
    ledger.buffer("fluid-flow", FLUID_FLOW_BYTES, "uniform");
    ledger.buffer("fluid-flow-counters", FLUID_FLOW_COUNTER_BYTES);
    ledger.buffer("fluid-particle-lifecycle", (4 + count) * 4);
    ledger.buffer("fluid-active-count-readback-0", 4, "none");
    ledger.buffer("fluid-active-count-readback-1", 4, "none");
    ledger.buffer("fluid-warmup-params", 16, "uniform");
}

function foamDispatchCapacityLimit(limits: NormalizedLimits): number {
    return Math.min(Math.floor(Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) / 32), MAX_WORKGROUPS * MAX_WORKGROUPS * WORKGROUP_SIZE);
}

function resolveFoamCapacity(input: FluidAllocationFoam, count: number, limits: NormalizedLimits, errors: string[]): number {
    const scaled = finiteNonNegative(input.poolScale, 3, errors, "foam.poolScale");
    const requested = finiteInteger(input.capacity, Math.round(count * scaled), 1, errors, "foam.capacity");
    const capMax = input.poolCapMax === undefined ? Number.MAX_SAFE_INTEGER : finiteInteger(input.poolCapMax, requested, 1, errors, "foam.poolCapMax");
    let capacity = Math.max(1024, Math.min(requested, capMax, foamDispatchCapacityLimit(limits)));
    if (input.activeParticles ?? true) {
        const approximate = Math.max(1024, Math.floor((limits.maxStorageBufferBindingSize - 256) / 12));
        capacity = Math.min(capacity, approximate);
        while (capacity > 1 && foamActiveStateBytes(capacity) > limits.maxStorageBufferBindingSize) {
            capacity--;
        }
    }
    return capacity;
}

function addFoamResources(
    ledger: ResourceLedger,
    method: FluidAllocationMethod,
    count: number,
    foam: FluidAllocationFoam | undefined,
    limits: NormalizedLimits,
    errors: string[]
): number {
    if (!foam?.enabled) {
        return 0;
    }
    const capacity = resolveFoamCapacity(foam, count, limits, errors);
    if (method === "PBF") {
        ledger.buffer("fluid-foam-sorted-normals", count * 16);
    }
    ledger.buffer(`${method.toLowerCase()}-foam-params`, FOAM_BYTES, "uniform");
    ledger.buffer(`${method.toLowerCase()}-foam-head`, 16);
    ledger.buffer(`${method.toLowerCase()}-foam-pool`, capacity * 32);
    if (foam.activeParticles ?? true) {
        ledger.buffer(`${method.toLowerCase()}-foam-active-state`, foamActiveStateBytes(capacity));
        ledger.buffer(`${method.toLowerCase()}-foam-active-dispatch`, 12);
        ledger.buffer(`${method.toLowerCase()}-foam-draw-indirect`, 16);
    }
    ledger.buffer(`${method.toLowerCase()}-foam-counts`, 16);
    ledger.buffer(`${method.toLowerCase()}-foam-counts-readback-0`, 16, "none");
    ledger.buffer(`${method.toLowerCase()}-foam-counts-readback-1`, 16, "none");
    return capacity;
}

function addPbfResources(ledger: ResourceLedger, count: number, cellCount: number): void {
    ledger.buffer("fluid-positions", count * 16);
    ledger.buffer("fluid-velocities", count * 16);
    ledger.buffer("fluid-predicted", count * 16);
    ledger.buffer("fluid-debug", count * 4);
    ledger.buffer("fluid-cell-count", cellCount * 4);
    ledger.buffer("fluid-cell-start", cellCount * 4);
    ledger.buffer("fluid-cell-cursor", cellCount * 4);
    ledger.buffer("fluid-partial-sums", Math.ceil(cellCount / SCAN_WORKGROUP_SIZE) * 4);
    ledger.buffer("fluid-sorted-idx", count * 4);
    ledger.buffer("fluid-sorted-pos", count * 16);
    ledger.buffer("fluid-sorted-pos0", count * 16);
    ledger.buffer("fluid-sorted-lambda", count * 4);
    ledger.buffer("fluid-sorted-delta", count * 16);
    ledger.buffer("fluid-sorted-vel", count * 16);
    ledger.buffer("fluid-sim", PBF_SIM_BYTES, "uniform");
    ledger.buffer("fluid-grid", PBF_GRID_BYTES, "uniform");
}

function mpmBlockCount(dim: readonly [number, number, number]): number {
    return dim.map((value) => Math.ceil(value / MPM_PAGE_SIZE)).reduce((product, value) => product * value, 1);
}

function maximumMlsPageCapacity(dim: readonly [number, number, number], limits: NormalizedLimits): number {
    const byCellBuffer = Math.max(0, Math.floor(Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) / MPM_PAGE_BYTES) - 1);
    return Math.min(mpmBlockCount(dim), byCellBuffer);
}

function addMlsResources(
    ledger: ResourceLedger,
    count: number,
    dim: readonly [number, number, number],
    cellCount: number,
    input: FluidAllocationPlanInput,
    limits: NormalizedLimits,
    errors: string[]
): FluidAllocationPages | undefined {
    const numBlocks = mpmBlockCount(dim);
    const paged = input.pagedGrid === true;
    const maximumPageCapacity = maximumMlsPageCapacity(dim, limits);
    const requestedMaxPages = finiteInteger(input.pagedGridMaxPages, mlsMpmDefaultPageCapacity(count), 1, errors, "pagedGridMaxPages");
    const maxPages = paged ? Math.min(requestedMaxPages, maximumPageCapacity) : 0;
    const activeBlocks = paged || input.activeBlocks === true;

    ledger.buffer("mpm-particles", count * 80);
    ledger.buffer("mpm-particles-working", count * 80);
    ledger.buffer(paged ? "mpm-paged-cells" : "mpm-cells", paged ? (maxPages + 1) * MPM_PAGE_BYTES : cellCount * 16);
    ledger.buffer("mpm-render-pos", count * 16);
    ledger.buffer("mpm-render-vel", count * 16);
    ledger.buffer("mpm-debug", count * 4);
    ledger.buffer("mpm-params", MLS_PARAMS_BYTES);
    ledger.buffer("mpm-copy-params", 16, "uniform");
    ledger.buffer("mpm-lifecycle-working", (4 + count) * 4);
    ledger.buffer("mpm-flow-counters-working", FLUID_FLOW_COUNTER_BYTES);
    ledger.buffer("mpm-block-count", numBlocks * 4);
    ledger.buffer("mpm-cell-count", cellCount * 4);
    ledger.buffer("mpm-block-start", numBlocks * 4);
    ledger.buffer("mpm-block-cursor", numBlocks * 4);
    ledger.buffer("mpm-partial-sums", Math.ceil(numBlocks / SCAN_WORKGROUP_SIZE) * 4);
    ledger.buffer("mpm-sorted-idx", count * 4);
    if (activeBlocks) {
        ledger.buffer("mpm-active-block-list", numBlocks * 4);
        ledger.buffer("mpm-active-block-state", paged ? 12 : 4);
        ledger.buffer("mpm-active-block-indirect", 12);
        ledger.buffer("mpm-node-block-flags", numBlocks * 4);
        ledger.buffer("mpm-node-block-list", numBlocks * 4);
        ledger.buffer("mpm-node-block-count", 4);
        ledger.buffer("mpm-node-block-indirect", 12);
    }
    if (paged) {
        ledger.buffer("mpm-page-map", numBlocks * 4);
        ledger.buffer("mpm-page-status-readback-0", 8, "none");
        ledger.buffer("mpm-page-status-readback-1", 8, "none");
    }
    ledger.buffer("mpm-accumulation-state", 16);
    ledger.buffer("mpm-accumulation-status-readback-0", 8, "none");
    ledger.buffer("mpm-accumulation-status-readback-1", 8, "none");

    if (!paged) {
        return undefined;
    }
    if (requestedMaxPages > maximumPageCapacity) {
        errors.push(`pagedGridMaxPages ${requestedMaxPages} exceeds the authoritative MLS-MPM maximumPageCapacity ${maximumPageCapacity}; using ${maxPages}.`);
    }
    return {
        numBlocks,
        requestedMaxPages,
        maximumPageCapacity,
        maxPages,
        storageCells: maxPages * MPM_PAGE_SIZE ** 3,
        storageFaces: 0,
        lookupWidth: 0,
        lookupHeight: 0,
        lookupWords: numBlocks,
        lookupPaddedWords: numBlocks,
        pageDispatchBytes: 12,
    };
}

function addPbMpmResources(ledger: ResourceLedger, count: number, cellCount: number): void {
    ledger.buffer("pbmpm-particles", count * 144);
    ledger.buffer("pbmpm-cells", cellCount * 16);
    ledger.buffer("pbmpm-grid-volume", cellCount * 4);
    ledger.buffer("pbmpm-render-pos", count * 16);
    ledger.buffer("pbmpm-render-vel", count * 16);
    ledger.buffer("pbmpm-debug", count * 4);
    ledger.buffer("pbmpm-params", PBMPM_PARAMS_BYTES, "uniform");
}

function maximumFlipPageCapacity(dim: readonly [number, number, number], limits: NormalizedLimits): number {
    const layout = resolveFlipPageLayout(dim, 1, limits.maxTextureDimension2D);
    const storageLimit = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
    const byFaceBuffer = Math.max(0, Math.floor((storageLimit / 8 - 1) / (FLIP_PAGE_CELLS * 3)));
    const byPageData = Math.max(0, Math.floor(storageLimit / 4) - 2 - layout.numBlocks);
    const byLookupTexture = Math.max(0, limits.maxTextureDimension2D ** 2 - 2 - layout.numBlocks);
    return Math.min(layout.numBlocks, byFaceBuffer, byPageData, byLookupTexture);
}

function multigridDimensions(gridDim: readonly [number, number, number]): Array<[number, number, number]> {
    const dimensions: Array<[number, number, number]> = [[...gridDim]];
    while (dimensions.length < 8) {
        const current = dimensions[dimensions.length - 1]!;
        if (Math.min(...current) <= 4 || current[0] * current[1] * current[2] <= 64) {
            break;
        }
        dimensions.push(current.map((value) => Math.max(1, Math.ceil(value / 2))) as [number, number, number]);
    }
    return dimensions;
}

function addFlipMultigridResources(ledger: ResourceLedger, dim: readonly [number, number, number], allocatedCellCount: number): void {
    for (const [index, levelDim] of multigridDimensions(dim).entries()) {
        const levelCount = index === 0 ? allocatedCellCount : levelDim[0] * levelDim[1] * levelDim[2];
        const bytes = levelCount * 4;
        if (index > 0) {
            ledger.buffer(`flip-multigrid-types-${index}`, bytes);
            ledger.buffer(`flip-multigrid-pressure-a-${index}`, bytes);
            ledger.buffer(`flip-multigrid-pressure-b-${index}`, bytes);
        }
        ledger.buffer(`flip-multigrid-rhs-${index}`, bytes);
        ledger.buffer(`flip-multigrid-residual-${index}`, bytes);
        ledger.buffer(`flip-multigrid-fluid-fraction-${index}`, bytes);
        ledger.buffer(`flip-multigrid-params-${index}`, FLIP_MULTIGRID_PARAMS_BYTES, "uniform");
    }
}

function addFlipPolygonResources(
    ledger: ResourceLedger,
    dim: readonly [number, number, number],
    paged: boolean,
    quality: FlipQualityFeatures,
    surfaceMaxTriangles: number | undefined,
    limits: NormalizedLimits,
    errors: string[]
): number {
    if (!quality.polygonSurface) {
        return 0;
    }
    const multiplier = Math.round(Math.min(2, Math.max(1, finitePositive(quality.polygonReconstructionMultiplier, 1, errors, "polygonReconstructionMultiplier"))) * 4) / 4;
    const polygonDim = dim.map((value) => Math.max(4, Math.ceil(value * multiplier))) as [number, number, number];
    const polygonCellCount = polygonDim[0] * polygonDim[1] * polygonDim[2];
    const cubeCount = Math.max(1, (polygonDim[0] - 1) * (polygonDim[1] - 1) * (polygonDim[2] - 1));
    const requestedCapacity = finiteInteger(surfaceMaxTriangles, FLIP_DEFAULT_SURFACE_TRIANGLE_CAPACITY, 1, errors, "surfaceMaxTriangles");
    const maxStorageBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
    const bounded = Math.min(requestedCapacity, cubeCount * 6, Math.floor(maxStorageBytes / 16));
    const triangleCapacity = Math.max(2, Math.floor(bounded / 2) * 2);
    ledger.buffer("flip-polygon-surface-vertices", cubeCount * FLIP_SURFACE_VERTEX_STRIDE);
    ledger.buffer("flip-polygon-surface-params", FLIP_PARAMS_BYTES, "uniform");
    if (multiplier > 1 || paged) {
        ledger.buffer("flip-polygon-surface-reconstructed-sdf", polygonCellCount * 4);
    }
    ledger.buffer("flip-polygon-surface-stabilized-sdf", polygonCellCount * 4);
    ledger.buffer("flip-polygon-surface-indices", triangleCapacity * 12);
    ledger.buffer("flip-polygon-surface-wireframe-indices", triangleCapacity * 16);
    ledger.buffer("flip-polygon-surface-draw", 20);
    ledger.buffer("flip-polygon-surface-wireframe-draw", 20);
    ledger.buffer("flip-polygon-surface-triangle-count-0", 4, "none");
    ledger.buffer("flip-polygon-surface-triangle-count-1", 4, "none");
    return triangleCapacity;
}

function normalizeFlipWarmup(input: FluidAllocationFlipWarmup | undefined, count: number, errors: string[]): FluidAllocationFlipWarmupPlan {
    let initialTargetCount = finiteInteger(input?.initialTargetCount, count, 0, errors, "flipWarmup.initialTargetCount");
    if (initialTargetCount > count) {
        errors.push(`flipWarmup.initialTargetCount ${initialTargetCount} exceeds particleCount ${count}; using ${count}.`);
        initialTargetCount = count;
    }
    let initialLiveCount = finiteInteger(input?.initialLiveCount, initialTargetCount, 0, errors, "flipWarmup.initialLiveCount");
    if (initialLiveCount > initialTargetCount) {
        errors.push(`flipWarmup.initialLiveCount ${initialLiveCount} exceeds initialTargetCount ${initialTargetCount}; using ${initialTargetCount}.`);
        initialLiveCount = initialTargetCount;
    }
    return {
        initialLiveCount,
        initialTargetCount,
        seedBuffersAllocated: initialLiveCount < initialTargetCount,
    };
}

function addFlipResources(
    ledger: ResourceLedger,
    count: number,
    dim: readonly [number, number, number],
    cellCount: number,
    faceCount: number,
    input: FluidAllocationPlanInput,
    limits: NormalizedLimits,
    errors: string[]
): { pages?: FluidAllocationPages; flipWarmup: FluidAllocationFlipWarmupPlan; polygonTriangleCapacity: number } {
    const quality = input.quality ?? {};
    const paged = input.pagedGrid ?? quality.pagedGrid ?? false;
    const requestedPages = finiteInteger(input.pagedGridMaxPages ?? quality.pagedGridMaxPages, FLIP_DEFAULT_PAGE_CAPACITY, 1, errors, "pagedGridMaxPages");
    const maximumPageCapacity = maximumFlipPageCapacity(dim, limits);
    const effectivePages = paged ? Math.min(requestedPages, maximumPageCapacity) : 0;
    const pageLayout = paged ? resolveFlipPageLayout(dim, Math.max(1, effectivePages), limits.maxTextureDimension2D) : undefined;
    const allocatedCellCount = pageLayout?.allocatedCells ?? cellCount;
    const allocatedFaceCount = pageLayout?.allocatedFaces ?? faceCount;
    const flipWarmup = normalizeFlipWarmup(input.flipWarmup, count, errors);

    ledger.buffer("flip-particle-positions", count * 16);
    ledger.buffer("flip-particle-velocities", count * 16);
    ledger.buffer("flip-particle-debug", count * 4);
    if (flipWarmup.seedBuffersAllocated) {
        ledger.buffer("flip-warmup-seed-positions", count * 16);
        ledger.buffer("flip-warmup-seed-velocities", count * 16);
    }
    ledger.buffer("flip-face-accum", allocatedFaceCount * 8);
    ledger.buffer("flip-face-old", allocatedFaceCount * 4);
    ledger.buffer("flip-face-velocity-a", allocatedFaceCount * 8);
    ledger.buffer("flip-face-velocity-b", allocatedFaceCount * 8);
    ledger.buffer("flip-viscosity-rhs", allocatedFaceCount * 4);
    ledger.buffer("flip-face-delta-a", allocatedFaceCount * 8);
    ledger.buffer("flip-face-delta-b", allocatedFaceCount * 8);
    ledger.buffer("flip-cell-marks", allocatedCellCount * 4);
    ledger.buffer("flip-cell-types", allocatedCellCount * 4);
    ledger.buffer("flip-divergence", allocatedCellCount * 4);
    ledger.buffer("flip-surface-normal", allocatedCellCount * 16);
    ledger.buffer("flip-surface-curvature", allocatedCellCount * 4);
    ledger.buffer("flip-pressure-a", allocatedCellCount * 4);
    ledger.buffer("flip-pressure-b", allocatedCellCount * 4);
    ledger.buffer("flip-params", FLIP_PARAMS_BYTES, "uniform");
    ledger.buffer("flip-flow-emit-range", 16, "uniform");
    ledger.buffer("flip-max-speed", 4);
    ledger.buffer("flip-max-speed-readback-0", 4, "none");
    ledger.buffer("flip-max-speed-readback-1", 4, "none");

    let pages: FluidAllocationPages | undefined;
    if (pageLayout) {
        ledger.buffer("flip-page-data", pageLayout.lookupWords * 4);
        ledger.texture("flip-page-lookup", pageLayout.lookupWidth, pageLayout.lookupHeight, 4);
        ledger.buffer("flip-page-dispatch", FLIP_PAGE_DISPATCH_BYTES);
        ledger.buffer("flip-page-dispatch-config", FLIP_PAGE_DISPATCH_CONFIG_BYTES, "uniform");
        ledger.buffer("flip-page-status", FLIP_PAGE_STATUS_BYTES);
        ledger.buffer("flip-page-status-readback-0", FLIP_PAGE_STATUS_READBACK_BYTES, "none");
        ledger.buffer("flip-page-status-readback-1", FLIP_PAGE_STATUS_READBACK_BYTES, "none");
        if (requestedPages > maximumPageCapacity) {
            errors.push(`pagedGridMaxPages ${requestedPages} exceeds the authoritative FLIP maximumPageCapacity ${maximumPageCapacity}; using ${pageLayout.maxPages}.`);
        }
        pages = {
            numBlocks: pageLayout.numBlocks,
            requestedMaxPages: requestedPages,
            maximumPageCapacity,
            maxPages: pageLayout.maxPages,
            storageCells: pageLayout.storageCells,
            storageFaces: pageLayout.storageFaces,
            lookupWidth: pageLayout.lookupWidth,
            lookupHeight: pageLayout.lookupHeight,
            lookupWords: pageLayout.lookupWords,
            lookupPaddedWords: pageLayout.lookupWidth * pageLayout.lookupHeight,
            pageDispatchBytes: FLIP_PAGE_DISPATCH_BYTES,
        };
    }

    if ((input.pressureSolver ?? "jacobi") === "multigrid") {
        addFlipMultigridResources(ledger, dim, allocatedCellCount);
    }
    if (quality.pressureDiagnostics) {
        ledger.buffer("flip-pressure-diagnostic-residual", allocatedCellCount * 4);
        ledger.buffer("flip-pressure-diagnostics", FLIP_PRESSURE_DIAGNOSTIC_BYTES);
        ledger.buffer("flip-pressure-diagnostics-readback-0", FLIP_PRESSURE_DIAGNOSTIC_BYTES, "none");
        ledger.buffer("flip-pressure-diagnostics-readback-1", FLIP_PRESSURE_DIAGNOSTIC_BYTES, "none");
    }
    const needsLiquidSdf = quality.liquidSdf || quality.particleSheeting || quality.polygonSurface;
    if (needsLiquidSdf) {
        ledger.buffer("flip-liquid-sdf-a", allocatedCellCount * 4);
        ledger.buffer("flip-liquid-sdf-b", allocatedCellCount * 4);
    }
    if (quality.fractionalSolids) {
        ledger.buffer("flip-solid-face-geometry", allocatedFaceCount * 8);
    }
    if (quality.reseedParticles) {
        ledger.buffer("flip-reseed-state", FLIP_RESEED_HEADER_BYTES + count * 8);
    }
    if (quality.particleSheeting) {
        ledger.buffer("flip-sheeting-state", FLIP_SHEETING_HEADER_BYTES + count * 4);
    }
    const polygonTriangleCapacity = addFlipPolygonResources(ledger, dim, paged, quality, input.surfaceMaxTriangles, limits, errors);
    return { ...(pages ? { pages } : {}), flipWarmup, polygonTriangleCapacity };
}

function validateResources(resources: readonly FluidAllocationResource[], limits: NormalizedLimits, errors: string[]): void {
    for (const resource of resources) {
        if (resource.kind === "texture") {
            if ((resource.width ?? 0) > limits.maxTextureDimension2D || (resource.height ?? 0) > limits.maxTextureDimension2D) {
                errors.push(`${resource.name} ${resource.width} x ${resource.height} exceeds maxTextureDimension2D ${limits.maxTextureDimension2D}.`);
            }
            continue;
        }
        if (resource.bytes > limits.maxBufferSize) {
            errors.push(`${resource.name} requires ${resource.bytes} bytes, exceeding maxBufferSize ${limits.maxBufferSize}.`);
        }
        if (resource.binding === "storage" && resource.bytes > limits.maxStorageBufferBindingSize) {
            errors.push(`${resource.name} requires ${resource.bytes} bytes in one storage binding, exceeding maxStorageBufferBindingSize ${limits.maxStorageBufferBindingSize}.`);
        }
    }
}

export function resolveFluidAllocationPlan(input: FluidAllocationPlanInput): FluidAllocationPlan {
    const errors: string[] = [];
    const particleCount = finiteInteger(input.particleCount, 1, 1, errors, "particleCount");
    const gridDim = normalizeGridDim(input.gridDim, errors);
    const { cellCount, faceCount } = cellAndFaceCounts(gridDim);
    const limits = normalizeLimits(input.limits, errors);
    const ledger = createLedger();
    let pages: FluidAllocationPages | undefined;
    let flipWarmup: FluidAllocationFlipWarmupPlan | undefined;
    let polygonTriangleCapacity = 0;

    addCommonResources(ledger, particleCount);
    if (input.method === "PBF") {
        addPbfResources(ledger, particleCount, cellCount);
    } else if (input.method === "MLS-MPM") {
        pages = addMlsResources(ledger, particleCount, gridDim, cellCount, input, limits, errors);
    } else if (input.method === "PB-MPM") {
        addPbMpmResources(ledger, particleCount, cellCount);
    } else {
        const flip = addFlipResources(ledger, particleCount, gridDim, cellCount, faceCount, input, limits, errors);
        pages = flip.pages;
        flipWarmup = flip.flipWarmup;
        polygonTriangleCapacity = flip.polygonTriangleCapacity;
    }

    const foamCapacity = addFoamResources(ledger, input.method, particleCount, input.foam, limits, errors);
    if (input.maxParticleBudget !== undefined) {
        const budget = finiteInteger(input.maxParticleBudget, particleCount, 1, errors, "maxParticleBudget");
        if (particleCount > budget) {
            errors.push(`particleCount ${particleCount} exceeds budget ${budget}.`);
        }
    }
    validateResources(ledger.resources, limits, errors);
    const steadyBytes = ledger.resources.reduce((sum, resource) => sum + resource.bytes, 0);
    const previousSteadyBytes = finiteInteger(input.previousSteadyBytes, 0, 0, errors, "previousSteadyBytes");
    return {
        method: input.method,
        dimensions: { gridDim, cellCount, faceCount, particleCount },
        ...(pages ? { pages } : {}),
        ...(flipWarmup ? { flipWarmup } : {}),
        foamCapacity,
        polygonTriangleCapacity,
        steadyBytes,
        rebuildPeakBytes: steadyBytes + previousSteadyBytes,
        resources: ledger.resources,
        errors,
    };
}

/** Largest mandatory individual particle-buffer stride for the selected method. */
export const fluidParticleBytesPerSlot = (method: string): number => {
    switch (method) {
        case "PBF":
            return PBF_PARTICLE_BYTES_PER_SLOT;
        case "FLIP":
            return FLIP_PARTICLE_BYTES_PER_SLOT;
        case "MLS-MPM":
            return MLS_MPM_PARTICLE_BYTES_PER_SLOT;
        case "PB-MPM":
            return PB_MPM_PARTICLE_BYTES_PER_SLOT;
        default:
            return DEFAULT_PARTICLE_BYTES_PER_SLOT;
    }
};

export const fluidParticleBufferLimitBytes = (limits: FluidBufferLimits): number =>
    Math.min(finitePositive(limits.maxStorageBufferBindingSize, 1, [], "maxStorageBufferBindingSize"), finitePositive(limits.maxBufferSize, 1, [], "maxBufferSize"));

export const fluidDeviceParticleCapacity = (limits: FluidBufferLimits, method: string): number =>
    Math.floor(fluidParticleBufferLimitBytes(limits) / fluidParticleBytesPerSlot(method));

export interface FluidParticleCapacityPlanInput extends Omit<FluidAllocationPlanInput, "particleCount" | "previousSteadyBytes" | "maxParticleBudget"> {
    /** Optional caller policy ceiling applied after the mandatory per-binding limit. */
    maximumParticleCount?: number;
}

export interface FluidParticleCapacityPlan {
    readonly capacity: number;
    readonly bytesPerParticleResource: number;
    readonly plan: FluidAllocationPlan;
    /** A rejected plan evaluated above capacity, when one was encountered. */
    readonly rejectedPlan?: FluidAllocationPlan;
}

function allocationResourcesFit(resources: readonly FluidAllocationResource[], limits: NormalizedLimits): boolean {
    return resources.every((resource) => {
        if (resource.kind === "texture") {
            return (resource.width ?? 0) <= limits.maxTextureDimension2D && (resource.height ?? 0) <= limits.maxTextureDimension2D;
        }
        return resource.bytes <= limits.maxBufferSize && (resource.binding !== "storage" || resource.bytes <= limits.maxStorageBufferBindingSize);
    });
}

/**
 * Resolves the largest particle count whose complete contextual plan fits every individual
 * buffer, binding, and texture limit. Aggregate allocation bytes are never treated as one buffer.
 */
export function resolveFluidParticleCapacity(input: FluidParticleCapacityPlanInput): FluidParticleCapacityPlan {
    const limits = normalizeLimits(input.limits, []);
    const bytesPerParticleResource = fluidParticleBytesPerSlot(input.method);
    const baseCapacity = fluidDeviceParticleCapacity(limits, input.method);
    const maximumParticleCount = finiteInteger(input.maximumParticleCount, baseCapacity, 0, [], "maximumParticleCount");
    let low = 1;
    let high = Math.min(baseCapacity, maximumParticleCount);
    let capacity = 0;
    let acceptedPlan: FluidAllocationPlan | undefined;
    let rejectedPlan: FluidAllocationPlan | undefined;
    const planFor = (particleCount: number): FluidAllocationPlan =>
        resolveFluidAllocationPlan({
            ...input,
            particleCount,
            limits,
        });

    while (low <= high) {
        const middle = Math.floor((low + high) / 2);
        const plan = planFor(middle);
        if (allocationResourcesFit(plan.resources, limits)) {
            capacity = middle;
            acceptedPlan = plan;
            low = middle + 1;
        } else {
            rejectedPlan = plan;
            high = middle - 1;
        }
    }

    const plan = acceptedPlan ?? planFor(1);
    return {
        capacity,
        bytesPerParticleResource,
        plan,
        ...(rejectedPlan ? { rejectedPlan } : {}),
    };
}

export const aquanovaCombinedParticleCapacity = (limits: FluidBufferLimits): number =>
    Math.max(1, Math.min(AQUANOVA_COMBINED_PARTICLE_CAPACITY, Math.floor(fluidParticleBufferLimitBytes(limits) / COMBINED_RENDER_PARTICLE_BYTES)));

export const fluidGridCellCountLimit = (limits: FluidBufferLimits): number =>
    Math.floor(finitePositive(limits.maxStorageBufferBindingSize, 1, [], "maxStorageBufferBindingSize") / 16);

export const fluidFlipMacBufferBytes = (cells: readonly [number, number, number], paging?: FluidFlipPaging): number =>
    paging?.enabled ? pagedFlipStorageCounts(Number.isFinite(paging.maxPages) ? Math.max(1, Math.floor(paging.maxPages)) : 1).faces * 8 : cellAndFaceCounts(cells).faceCount * 8;

/** Validate grid dimensions and produce shared, consumer-independent device diagnostics. */
export function resolveFluidGridCompatibility(
    method: FluidAllocationMethod,
    gridDim: readonly [number, number, number],
    limits: FluidBufferLimits,
    paging?: FluidFlipPaging
): FluidGridCompatibility {
    const dim = gridDim.map((value) => Math.max(0, Math.floor(Number.isFinite(value) ? value : 0))) as [number, number, number];
    const oversizedAxis = dim.findIndex((value) => value > FLUID_GRID_MAX_AXIS_CELLS);
    if (oversizedAxis >= 0) {
        return {
            compatible: false,
            code: "axis-limit",
            message: `Grid requires ${dim[oversizedAxis]!.toLocaleString()} cells on ${"XYZ"[oversizedAxis]}; maximum is ${FLUID_GRID_MAX_AXIS_CELLS.toLocaleString()}.`,
            gridDim: dim,
        };
    }
    const availableBytes = fluidParticleBufferLimitBytes(limits);
    const requiredBytes = method === "FLIP" ? fluidFlipMacBufferBytes(dim, paging) : dim[0] * dim[1] * dim[2] * 16;
    if (requiredBytes > availableBytes) {
        const subject = method === "FLIP" && paging?.enabled ? `Page capacity ${paging.maxPages.toLocaleString()}` : `Grid ${dim.join(" \u00d7 ")}`;
        return {
            compatible: false,
            code: "storage-binding-limit",
            message: `${subject} requires ${(requiredBytes / (1024 * 1024)).toFixed(1)} MiB per ${method === "FLIP" ? "FLIP MAC" : "grid"} storage buffer; this device supports ${(availableBytes / (1024 * 1024)).toFixed(1)} MiB.`,
            gridDim: dim,
            requiredBytes,
            availableBytes,
        };
    }
    return { compatible: true, gridDim: dim, requiredBytes, availableBytes };
}

export const mlsMpmDefaultPageCapacity = (particleCount: number): number => {
    const count = Number.isFinite(particleCount) ? Math.max(1, particleCount) : 1;
    return Math.max(1000, Math.round((count * 27 * 1.5) / 64000) * 1000);
};

export function fluidMaximumPageCapacity(method: "FLIP" | "MLS-MPM", gridDim: readonly [number, number, number], limits: FluidDeviceLimitsSnapshot): number {
    const errors: string[] = [];
    const dim = normalizeGridDim(gridDim, errors);
    const normalizedLimits = normalizeLimits(limits, errors);
    return method === "FLIP" ? maximumFlipPageCapacity(dim, normalizedLimits) : maximumMlsPageCapacity(dim, normalizedLimits);
}
