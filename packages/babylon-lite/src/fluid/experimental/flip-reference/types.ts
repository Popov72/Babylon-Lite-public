import type { FluidProfiler } from "../../core/sim-common.js";

/** Parameters for the native histogram and sparse-top-outlier deletion rule, not a velocity clamp. */
export interface FlipReferenceExtremeRemovalOptions {
    /** Full physical frame duration, not the duration of one solver substep. */
    frameDtSeconds: number;
    cfl: number;
    /** Histogram bins / native maximum frame substeps. */
    maxFrameSubsteps: number;
}

/** Explicit, offline-oriented configuration for the isolated MAC/FLIP experiment. */
export interface FlipReferenceOptions {
    /**
     * Opt-in researched stage conventions: Wyvill transfer, conditioned liquid SDF, 12 extension layers,
     * corner-case face areas and averaged cube partitions, theta at least 1/26, constrained snapshots,
     * RK3 and swept collisions.
     * Also defaults the domain inset to 1.5*dx+5e-5, domain marker clearance to 0.1*dx, and obstacle
     * push-out to 0.2*dx. Explicit options override these defaults. Pressure remains original f32 Jacobi-PCG.
     * Use this preset for native-cache comparisons; the alternatives retain legacy transfer/contact behavior.
     */
    referenceNumerics?: boolean;
    gridOrigin: readonly [number, number, number];
    gridDimensions: readonly [number, number, number];
    cellSize: number;
    /** Tightly packed world-space XYZ marker coordinates. */
    initialPositions: Float32Array;
    /** Tightly packed world-space XYZ velocities, in distance units per second. */
    initialVelocities: Float32Array;
    gravity?: readonly [number, number, number];
    picFraction?: number;
    /** Relative L2 residual of the pressure-impulse system. Default: 1e-6. */
    pressureTolerance?: number;
    /** Absolute L2 residual floor, in velocity units. Default: 1e-8. */
    pressureAbsoluteTolerance?: number;
    /** Iteration bound for the original f32 Jacobi-PCG solver. Default: 400 for both presets. */
    maxPressureIterations?: number;
    /** Minimum liquid fraction on a liquid/air pressure edge. Baseline: 0.01; reference preset: 1/26. */
    ghostFluidThetaMin?: number;
    kernel?: "trilinear" | "radial" | "wyvill";
    /** Radial transfer support in world units: dx for radial, sqrt(3)/2*dx for Wyvill. Trilinear is fixed at dx. */
    transferRadius?: number;
    advection?: "rk2" | "rk3";
    /** Radius of the spheres defining the liquid level set. Default: sqrt(3)/2 * dx. */
    liquidRadius?: number;
    /** Debug marker radius; does not change the liquid level set. Default: 0.1 * dx. */
    particleRadius?: number;
    /** Obstacle projection clearance or swept push-out target. Baseline: 0.01*dx; reference preset: 0.2*dx. */
    collisionRadius?: number;
    /** Physical closed-domain boundary inset, in world units. With solidsIncludeDomain, this only locates the separate particle collision AABB. Baseline default: zero. */
    domainInset?: number;
    /** Additional marker clearance from the inset domain boundary. Baseline default: collisionRadius. */
    domainCollisionRadius?: number;
    /** Constrain both old and projected MAC fields at fully blocked faces before G2P. Baseline default: false. */
    constrainSnapshot?: boolean;
    /** Include open cell volume in the moving-solid pressure RHS. Default: true; disable only for a legacy-RHS ablation. */
    solidVolumeCorrection?: boolean;
    /** Center-triangulated fractions, or corner-case faces and averaged five-tetrahedra cube partitions. Reference fractions treat nodal zero as outside solid. */
    geometryFractions?: "center" | "reference";
    /** Sweeps use intervals at most 0.1*dx and preserve G2P velocities; projection uses half-cell trajectory segments. */
    collisionMode?: "project" | "sweep";
    /** Six-neighbor velocity-extension layers, before and after projection. Baseline: 8; reference preset: 12. */
    extrapolationLayers?: number;
    /** Per-marker trial/interval budget. Sweeps share at most this many + 1 samples per step. Default: 256; maximum: 16777215. */
    maxAdvectionSubsteps?: number;
    /** Delete markers still inside the supplied solid SDF after collision handling. Default: false. */
    removeInsideSolids?: boolean;
    /** Separate opt-in native extreme-speed deletion, evaluated before other removals. */
    extremeVelocityRemoval?: FlipReferenceExtremeRemovalOptions;
    /**
     * Supplied nodal fields already contain the closed domain. Disables all internal domain-SDF composition.
     * Supply solidDistances at construction or call updateFlipReferenceSolids before the first step.
     * The separate particle collision AABB still uses domainInset + domainCollisionRadius. Default: false.
     */
    solidsIncludeDomain?: boolean;
    /** Nodal, X-fastest signed distances: (nx+1)*(ny+1)*(nz+1). Negative means solid. */
    solidDistances?: Float32Array;
    /** Tightly packed XYZ world velocities at the same grid vertices. Default: stationary. */
    solidVelocities?: Float32Array;
}

export interface FlipReferenceDiagnostics {
    consumedDtSeconds: number;
    elapsedSeconds: number;
    particleCount: number;
    particleCapacity: number;
    particleCountBeforeRemoval: number;
    removedParticleCount: number;
    removedInsideSolids: number;
    removedExtremeVelocities: number;
    totalRemovedInsideSolids: number;
    totalRemovedExtremeVelocities: number;
    fluidCellCount: number;
    pressureIterations: number;
    /** False when the bounded pressure solve has not reached its residual tolerance. */
    converged: boolean;
    /** Same value as converged, retained for pressure-specific consumers. */
    pressureConverged: boolean;
    pressureResidualNorm: "l2";
    /** The matrix uses pressure impulse dt*p/(rho*dx), so its residual is in velocity units, not divergence units. */
    pressureResidualUnits: "velocity";
    /** True (recomputed, not recursively updated) absolute L2 pressure-system residual. */
    pressureAbsoluteResidual: number;
    pressureRelativeResidual: number;
    pressureRhsNorm: number;
    /** Maximum post-projection finite-volume divergence, in inverse seconds. */
    maxDivergence: number;
    sealedComponentCount: number;
    conditionedComponentCount: number;
    conditionedCellCount: number;
    /** Sum of the RHS in the worst closed component, before fixing its pressure gauge. */
    maxSealedComponentFlux: number;
    collisionCount: number;
    maxSpeed: number;
    /** Includes all markers after G2P, before any pruning, so removal cannot conceal an acceleration spike. */
    maxSpeedBeforeRemoval: number;
    /** Null when extreme-speed removal is disabled. */
    extremeSpeedThreshold: number | null;
    /** Swept collisions that reverted to their last nonpenetrating sample, without changing marker velocity. */
    collisionFallbackCount: number;
    referenceNumerics: boolean;
    solidsIncludeDomain: boolean;
    solidVolumeCorrection: boolean;
}

/** Plain simulation state. GPU storage is intentionally package-internal. */
export interface FlipReferenceSimulation {
    readonly gridOrigin: readonly [number, number, number];
    readonly gridDimensions: readonly [number, number, number];
    readonly cellSize: number;
    /** Live prefix length, updated by stepFlipReferenceSimulation. Do not assign directly. */
    readonly count: number;
    /** Fixed allocation size. Buffer identities and sizes do not change when markers are removed. */
    readonly capacity: number;
    totalRemovedInsideSolids: number;
    totalRemovedExtremeVelocities: number;
    readonly particleRadius: number;
    readonly referenceNumerics: boolean;
    readonly solidsIncludeDomain: boolean;
    elapsedSeconds: number;
    diagnostics: FlipReferenceDiagnostics | null;
    /** @internal Authoritative live prefix length, published through count. */
    _count: number;
    /** @internal XYZ1 f32 marker positions, one vec4 per marker. */
    _positionBuffer: GPUBuffer;
    /** @internal XYZ0 f32 marker velocities, one vec4 per marker. */
    _velocityBuffer: GPUBuffer;
    /** @internal One f32 speed per marker. */
    _debugBuffer: GPUBuffer;
    /** @internal */
    _device: GPUDevice;
    /** @internal Optional timestamp enabler; command spans never cross CPU awaits. */
    _profiler?: FluidProfiler | null;
    /** @internal Finish marker for the command encoder currently being recorded. */
    _profileEnd?: () => void;
    /** @internal */
    _buffers: GPUBuffer[];
    /** @internal */
    _uniformBuffer: GPUBuffer;
    /** @internal */
    _faceBuffer: GPUBuffer;
    /** @internal */
    _cellBuffer: GPUBuffer;
    /** @internal */
    _listBuffer: GPUBuffer;
    /** @internal */
    _solidBuffer: GPUBuffer;
    /** @internal */
    _pressureBuffer: GPUBuffer;
    /** @internal Scratch pairs: XYZ1 position, then XYZ velocity plus scalar speed. */
    _particleScratchBuffer: GPUBuffer;
    /** @internal Removal reasons, prefix scan, counters and histogram; used only when removal is enabled. */
    _particleStateBuffer: GPUBuffer;
    /** @internal Byte offset of removal counters in _particleStateBuffer. */
    _removalControlOffset: number;
    /** @internal */
    _removalReadback: Uint32Array;
    /** @internal */
    _removalEnabled: boolean;
    /** @internal */
    _extremeRemoval: boolean;
    /** @internal */
    _removalHistogramBins: number;
    /** @internal */
    _readbackBuffer: GPUBuffer;
    /** @internal */
    _pipelines: Record<string, GPUComputePipeline>;
    /** @internal */
    _bindGroups: Record<string, GPUBindGroup>;
    /** @internal */
    _params: Float32Array;
    /** @internal */
    _cells: number;
    /** @internal */
    _faces: number;
    /** @internal */
    _vertices: number;
    /** @internal */
    _reductionGroups: number;
    /** @internal */
    _cellReadback: Float32Array;
    /** @internal */
    _controlReadback: Float32Array;
    /** @internal */
    _statusReadback: Uint32Array;
    /** @internal */
    _componentVisited: Uint8Array;
    /** @internal */
    _componentQueue: Int32Array;
    /** @internal */
    _solidUpload: Float32Array;
    /** @internal */
    _hasSolidData: boolean;
    /** @internal */
    _pressureTolerance: number;
    /** @internal */
    _pressureAbsoluteTolerance: number;
    /** @internal */
    _maxPressureIterations: number;
    /** @internal */
    _extrapolationLayers: number;
    /** @internal */
    _constrainSnapshot: boolean;
    /** @internal */
    _solidVolumeCorrection: boolean;
    /** @internal */
    _ready: Promise<void>;
    /** @internal */
    _error: Error | null;
    /** @internal */
    _busy: boolean;
    /** @internal */
    _disposed: boolean;
}
