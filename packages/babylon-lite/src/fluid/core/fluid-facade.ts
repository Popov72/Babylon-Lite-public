import type { Camera } from "../../camera/camera.js";
import type { EngineContext } from "../../engine/engine.js";
import { submitGpuOperation } from "../../engine/engine.js";
import { retireGpuResources } from "../../engine/gpu-resource-retirement.js";
import type { RenderTarget } from "../../engine/render-target.js";
import { addTask, addTaskBefore } from "../../frame-graph/frame-graph-actions.js";
import type { Task } from "../../frame-graph/task.js";
import { getFrameGraph } from "../../scene/scene-core.js";
import type { SceneContext } from "../../scene/scene-core.js";
import { createFlipSim, encodeFlipSimStateTransfer } from "../solvers/flip-sim.js";
import type { FlipOptions } from "../solvers/flip-sim.js";
import { createFluidRenderCompositor } from "../rendering/fluid-render-compositor.js";
import type { FluidRenderCompositor } from "../rendering/fluid-render-compositor.js";
import { applyFluidRenderProfile, fluidRenderHexColor } from "../rendering/fluid-render-profile.js";
import type { FluidRenderProfileSettings } from "../rendering/fluid-render-profile.js";
import { createFoamRenderTask } from "../rendering/foam-render.js";
import type { FoamDebugTexture } from "../rendering/foam-render.js";
import { createCompositeSceneSdfRuntimeBinding, createForceFieldRuntimeBinding, createSceneSdfRuntimeBinding } from "./fluid-runtime-bindings.js";
import type {
    CompositeSceneSdfBounds,
    CompositeSceneSdfGridData,
    CompositeSceneSdfGridSettingsUpdate,
    CompositeSceneSdfLocalGridData,
    CompositeSceneSdfTransformUpdate,
    ForceFieldRuntimeBinding,
    SceneSdfRuntimeBinding,
} from "./fluid-runtime-bindings.js";
import type { Mat4 } from "../../math/types.js";
import {
    adoptFluidParticleChannelRuntime,
    createFluidAggregateRuntime,
    createFluidParticleChannelRuntime,
    createFluidSpatialQueryRuntime,
    createFluidWheelTorqueQueryRuntime,
    fluidSimParticleStream,
    readFluidSimPositions,
    writeFluidSimPositions,
} from "./fluid-particle-runtime.js";
import type {
    FluidAggregateRuntime,
    FluidParticleChannelRuntime,
    FluidParticleStreamRuntime,
    FluidSpatialQueryRuntime,
    FluidWheelTorqueQueryRuntime,
} from "./fluid-particle-runtime.js";
import { createFluidSurfaceTask } from "../rendering/fluid-surface-render.js";
import type { FluidDebug, FluidSurfaceTask } from "../rendering/fluid-surface-render.js";
import { createFluidProfiler as createGpuFluidProfiler } from "./gpu-profiler.js";
import type { FluidProfilerImpl } from "./gpu-profiler.js";
import { createMlsMpmSim } from "../solvers/mls-mpm-sim.js";
import type { MlsMpmOptions } from "../solvers/mls-mpm-sim.js";
import { createParticleRenderTask } from "../rendering/particle-render.js";
import type { ParticleRenderTask } from "../rendering/particle-render.js";
import { createPbfSim } from "../solvers/pbf-sim.js";
import type { PbfOptions } from "../solvers/pbf-sim.js";
import { createPbMpmSim } from "../solvers/pbmpm-sim.js";
import type { PbMpmOptions } from "../solvers/pbmpm-sim.js";
import { createFluidPolygonSurfaceTask } from "../rendering/polygon-surface-render.js";
import type { FluidPolygonSurfaceTask } from "../rendering/polygon-surface-render.js";
import type { FloatingBodySystem } from "./floating-body.js";
import type { FluidTimestepDiagnostics } from "./timestep-scheduler.js";
import type {
    DiffuseParticleCounts,
    FluidEmitter,
    FluidFlowConfig,
    FluidPressureDiagnostics,
    FluidProfiler,
    FluidSim,
    FoamConfig,
    ForceFieldSpec,
    SceneSdfSpec,
} from "./sim-common.js";
import { resolveFluidSimulationConfig } from "./simulation-config.js";
import type { FluidCompatibilityProfile, FluidGridBounds, FluidSimulationSamplingType, FluidSimulationSemantics, ResolvedFluidSimulationConfig } from "./simulation-config.js";
import type { FluidAllocationPlan, FluidDeviceLimitsSnapshot } from "./allocation-plan.js";

export interface FluidSceneSdfOptions {
    /** WGSL declaration for the parameter structure read by `sdf`. */
    readonly struct: string;
    /** WGSL `sceneSdf(point, deltaTime)` implementation. */
    readonly sdf: string;
    /** Initial packed uniform values. The allocation remains fixed-size. */
    readonly params: Float32Array;
    /** Optional dense signed-distance grid, negative inside solid geometry. */
    readonly sdfGrid?: Float32Array | Uint32Array;
    /** Storage encoding used by `sdfGrid`. Defaults to `f32`. */
    readonly sdfGridFormat?: "f32" | "packed-f16";
    /** Whether MLS-MPM should confine at grid nodes. */
    readonly gridConfine?: boolean;
    /** Whether solvers apply temporal moving-boundary velocity. */
    readonly movingBoundaries?: boolean;
}
export interface FluidSdfGridData extends CompositeSceneSdfGridData {}

export interface FluidLocalSdfData extends CompositeSceneSdfLocalGridData {}

export interface FluidSceneSdfBounds extends CompositeSceneSdfBounds {}

export interface FluidCompositeSceneSdfOptions {
    /** Optional static world-space collision grid. */
    readonly staticSdf?: FluidSdfGridData;
    /** Rigid local-space grids driven by transform updates. */
    readonly localSdfs?: readonly FluidLocalSdfData[];
    /** Optional closed container: positive inside, negative outside. */
    readonly container?: FluidSceneSdfBounds;
    /** Whether MLS-MPM should confine at grid nodes. */
    readonly gridConfine?: boolean;
    /** Whether solvers apply temporal moving-boundary velocity. */
    readonly movingBoundaries?: boolean;
}

export interface FluidSceneSdfTransformUpdate {
    readonly id: string;
    readonly localToWorld: Mat4;
}

export interface FluidSceneSdfGridSettingsUpdate {
    /** Omit for the static grid; use an animated/local grid ID otherwise. */
    readonly id?: string;
    readonly enabled?: boolean;
    readonly trilinear?: boolean;
}

/** Opaque, pure-state scene collision binding. */
export interface FluidSceneSdf {
    readonly disposed: boolean;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _binding: unknown;
    /** @internal */
    _owners: Set<object>;
}

export interface FluidForceFieldOptions {
    /** WGSL declaration for the parameter structure read by `wgsl`. */
    readonly struct: string;
    /** WGSL `externalForce(position, velocity, deltaTime)` implementation. */
    readonly wgsl: string;
    /** Initial packed uniform values. The allocation remains fixed-size. */
    readonly params: Float32Array;
}

export interface FluidImpulseForceOptions {
    readonly center: readonly [number, number, number];
    readonly radius: number;
    readonly intensity: number;
    readonly direction: readonly [number, number, number];
    readonly fallbackDirection?: readonly [number, number, number];
    readonly radialScale?: number;
    readonly directionalScale?: number;
}

export interface FluidImpulseForceResolution {
    readonly direction: readonly [number, number, number];
    readonly usedFallbackDirection: boolean;
    readonly directionalMagnitude: number;
    readonly radialMagnitude: number;
}

/** Opaque, pure-state external-force binding. */
export interface FluidForceField {
    readonly disposed: boolean;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _binding: unknown;
    /** @internal */
    _owners: Set<object>;
    /** @internal Identifies shared force layouts without a module-level registry. */
    _kind?: "impulse";
}

export interface FluidSimulationProfilerResults {
    readonly stages: Readonly<Record<string, number>>;
    readonly total: number;
    readonly frameTotal: number;
    /** Query capacity was exceeded; timings are incomplete and must not be displayed as a full sample. */
    readonly overflowed?: boolean;
}

export interface FluidSimulationProfilerOptions {
    readonly queryCapacity?: number;
}

export interface FluidSimulationProfilerFrameOptions {
    /** Omit the render-encoder envelope for a separately submitted simulation-only operation. */
    readonly captureEnvelope?: boolean;
    /** Close the render envelope now, but resolve only after separately submitted GPU work. */
    readonly deferResolve?: boolean;
}

/** Opaque, pure-state GPU profiler. Frame behavior is exposed by standalone functions. */
export interface FluidSimulationProfiler {
    readonly disposed: boolean;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _profiler: unknown;
    /** @internal */
    _owners: Set<object>;
}

export interface FluidParticleChannelOptions {
    readonly capacity: number;
    readonly components: 1 | 4;
    readonly label?: string;
    readonly initialData?: Float32Array;
}

/** Opaque scalar or RGBA particle data usable by shared render streams. */
export interface FluidParticleChannel {
    readonly capacity: number;
    readonly components: 1 | 4;
    readonly disposed: boolean;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _runtime: unknown;
    /** @internal */
    _owners: Set<object>;
}

/** Existing environment state accepted only to create an opaque fluid render environment. */
export type FluidRenderEnvironmentSource = object;

/** Non-owning opaque environment used to retarget surface and polygon reflections. */
export interface FluidRenderEnvironment {
    /** @internal */
    readonly _source: FluidRenderEnvironmentSource;
}

/** Opaque particle positions and visibility consumed by spatial queries and scene effects. */
export interface FluidParticleStream {
    readonly count: number;
    readonly capacity: number;
    /** @internal */
    _resolve: () => unknown;
}

export interface FluidParticleSpatialQueryOptions {
    readonly maximumQueries?: number;
}

export const FLUID_SPATIAL_QUERY_SAMPLE_INTERVAL_FRAMES = 6;

export interface FluidParticleSpatialQueryRequest {
    readonly key: string | number;
    readonly offset?: number;
    readonly count?: number;
    readonly bounds: FluidGridBounds;
    /** Optional propagation sphere intersected with `bounds`, used by electricity effects. */
    readonly sphere?: {
        readonly origin: readonly [number, number, number];
        readonly radius: number;
    };
}

export interface FluidParticleSpatialQueryResult {
    readonly key: string | number;
    readonly count: number;
}

/** Batched asynchronous AABB/sphere particle counter. */
export interface FluidParticleSpatialQuery {
    readonly results: readonly FluidParticleSpatialQueryResult[];
    readonly status: "idle" | "pending" | "ready" | "failed";
    readonly disposed: boolean;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _runtime: unknown;
}

export interface FluidWheelTorqueQueryOptions {
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

/** Opaque asynchronous reduction of particle weight around a wheel axis. */
export interface FluidWheelTorqueQuery {
    readonly torque: number;
    readonly status: "idle" | "pending" | "ready" | "failed";
    readonly disposed: boolean;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _runtime: unknown;
}

export interface FluidSimulationDiffuseDiagnostics extends DiffuseParticleCounts {
    readonly enabled: boolean;
}

export interface FluidSimulationPolygonDiagnostics {
    readonly triangleCount: number | undefined;
    readonly triangleCapacity: number;
    readonly reconstructionMultiplier: number;
    readonly gridOrigin: readonly [number, number, number];
    readonly gridDimensions: readonly [number, number, number];
    readonly gridSpacing: number;
}

export interface FluidSimulationDiagnostics {
    readonly method: FluidSimulationOptions["method"];
    readonly count: number;
    readonly activeCount: number;
    readonly renderCount: number;
    readonly particleRadius: number;
    readonly surfaceSizeScale: number;
    readonly surfaceThicknessScale: number;
    readonly surfaceRejectSparseMarkers: boolean;
    readonly gpuBytes: number;
    readonly initialEmitterParticleCounts: ReadonlyMap<string, number>;
    readonly pressure: FluidPressureDiagnostics | null;
    readonly timestep: FluidTimestepDiagnostics | null;
    readonly diffuse: FluidSimulationDiffuseDiagnostics | null;
    readonly polygon: FluidSimulationPolygonDiagnostics | null;
}

/** Opt-in backend provider. Intrinsic method semantics remain separate from implementation selection. */
export interface FluidSimulationBackend {
    readonly id: string;
    readonly name: string;
    readonly method: FluidSimulationOptions["method"];
    readonly steppingMode: "frame" | "async";
    readonly renderModes: readonly FluidSimulationRenderMode[];
    readonly supportsFoam: boolean;
    readonly supportsForces: boolean;
    readonly supportsContinuousFlow: boolean;
    readonly physicsParameters: readonly string[];
    readonly description: string;
    /** @internal */
    readonly _create: (engine: EngineContext, options: FluidSimulationOptions, config: ResolvedFluidSimulationConfig) => FluidSim;
    /** @internal Override intrinsic-method memory projection for a different buffer layout. */
    readonly _allocationPlan?: (options: FluidSimulationOptions, config: ResolvedFluidSimulationConfig, limits?: FluidDeviceLimitsSnapshot) => FluidAllocationPlan;
}

export interface FluidSimulationStepOptions {
    /**
     * Advance scene inputs before a CPU-scheduled submission, outside render-frame recording.
     * GPU-resident backends call once with the complete frame delta and sample temporal collisions
     * within that frame; readback-driven backends may call separately for each physical substep.
     */
    readonly beforeSubstep?: (deltaSeconds: number) => void;
}

export interface FluidSimulationOptions {
    method: "PBF" | "FLIP" | "MLS-MPM" | "PB-MPM";
    /** Explicitly loaded implementation override; omitted for the production method. */
    backend?: FluidSimulationBackend;
    particleCount: number;
    bounds: FluidGridBounds;
    physicsScale: number;
    compatibilityProfile?: FluidCompatibilityProfile;
    explicitGrid?: boolean;
    domainScale?: number;
    semantics?: FluidSimulationSemantics;
    samplingType?: FluidSimulationSamplingType;
    particleRadius?: number;
    groundY?: number;
    gridResolution?: number;
    markersPerCell?: number;
    /** Authored values. The facade applies the selected simulation semantics exactly once. */
    physics?: Readonly<Record<string, number>>;
    flow?: FluidFlowConfig;
    foam?: FoamConfig | null;
    material?: number;
    activeBlocks?: boolean;
    pagedGrid?: boolean;
    pagedGridMaxPages?: number;
    fusedBlockDiscovery?: boolean;
    initialPositions?: Float32Array;
    initialVelocities?: Float32Array;
    sceneSdf?: FluidSceneSdf | null;
    forceField?: FluidForceField | null;
    profiler?: FluidSimulationProfiler | null;
    onPagedGridPages?: (requiredPages: number, capacity: number) => void;
    onPagedGridOverflow?: (requiredPages: number, capacity: number) => void;
}

/** GPU-safe pure-state simulation handle. Behavior is provided by standalone functions below. */
export interface FluidSimulation {
    readonly method: FluidSimulationOptions["method"];
    readonly options: FluidSimulationOptions;
    readonly steppingMode: "frame" | "async";
    readonly count: number;
    readonly activeCount: number | undefined;
    readonly renderCount: number | undefined;
    readonly particleRadius: number;
    readonly surfaceSizeScale: number;
    readonly surfaceThicknessScale: number;
    readonly surfaceRejectSparseMarkers: boolean;
    readonly initialEmitterParticleCounts: ReadonlyMap<string, number>;
    readonly gpuBytes: number;
    readonly pressureDiagnostics: FluidPressureDiagnostics | undefined;
    readonly polygonTriangleCount: number | undefined;
    readonly resolvedConfig: ResolvedFluidSimulationConfig;
    readonly sceneSdf: FluidSceneSdf | null;
    readonly forceField: FluidForceField | null;
    readonly profiler: FluidSimulationProfiler | null;
    readonly renderLayers: readonly FluidSimulationRenderLayer[];
    disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _sim: unknown;
    /** @internal */
    _method: FluidSimulationOptions["method"];
    /** @internal */
    _options: FluidSimulationOptions;
    /** @internal */
    _resolvedConfig: ResolvedFluidSimulationConfig;
    /** @internal */
    _renderLayers: FluidSimulationRenderLayer[];
}

export type FluidSimulationRenderMode = "spheres" | "surface" | "polygon";
export type FluidSimulationCollectionRenderMode = FluidSimulationRenderMode | "foam";

export interface FluidFoamRenderSettings {
    readonly enabled?: boolean;
    readonly polygonSurfaceDepth?: boolean;
    readonly surfaceFiltering?: boolean;
    readonly opacity?: number;
    readonly sizeScale?: number;
    readonly spraySize?: number;
    readonly sprayIntensity?: number;
    readonly spraySeparation?: number;
    readonly debugByKind?: boolean;
    readonly softness?: number;
    readonly density?: number;
    readonly subsurfaceStrength?: number;
    readonly subsurfaceColor?: string;
    readonly blurRadius?: number;
    readonly debugTexture?: FoamDebugTexture;
    readonly lightIntensity?: number;
    readonly ambient?: number;
    readonly aoStrength?: number;
    readonly normalStrength?: number;
}

export interface FluidSimulationRenderLayerOptions {
    scene: SceneContext;
    camera: Camera;
    mode: FluidSimulationRenderMode;
    depthTarget: RenderTarget;
    colorTarget?: RenderTarget;
    backgroundTarget?: RenderTarget;
    outputTarget?: RenderTarget;
    profile?: FluidRenderProfileSettings;
    profiler?: FluidSimulationProfiler | null;
    /** Scene whose current high-level environment should be reflected by the fluid. */
    environmentScene?: SceneContext;
    /** Opaque environment for local-probe or other scene-specific reflection retargeting. */
    environment?: FluidRenderEnvironment;
    environmentRotationY?: number;
    direction?: readonly [number, number, number];
    surfaceMode?: "surface" | "blit" | "ellipsoidDebug";
    particleVelocityBrighten?: number;
    particleColorMode?: "water" | "mesh";
    useParticleColor?: boolean;
    debug?: FluidDebug;
    polygonWireframe?: boolean;
    foam?: FluidFoamRenderSettings;
}

export interface FluidSimulationRenderLayerState {
    enabled?: boolean;
    opacity?: number;
    profile?: FluidRenderProfileSettings;
    profiler?: FluidSimulationProfiler | null;
    environmentScene?: SceneContext;
    environment?: FluidRenderEnvironment;
    environmentRotationY?: number;
    direction?: readonly [number, number, number];
    surfaceMode?: "surface" | "blit" | "ellipsoidDebug";
    particleVelocityBrighten?: number;
    particleColorMode?: "water" | "mesh";
    useParticleColor?: boolean;
    debug?: FluidDebug;
    polygonWireframe?: boolean;
    foam?: FluidFoamRenderSettings;
}

/** GPU-safe pure-state render attachment. */
export interface FluidSimulationRenderLayer {
    readonly mode: FluidSimulationCollectionRenderMode;
    readonly profile: FluidRenderProfileSettings;
    readonly particleStream: FluidParticleStream | null;
    /** Whether the latest collection aggregation exceeded the layer's particle capacity. */
    readonly overflowed: boolean;
    enabled: boolean;
    opacity: number;
    disposed: boolean;
    /** @internal */
    _particleTask?: unknown;
    /** @internal */
    _surfaceTask?: unknown;
    /** @internal */
    _polygonTask?: unknown;
    /** @internal */
    _foamTask?: unknown;
    /** @internal */
    _task: unknown;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _scene: unknown;
    /** @internal */
    _profile: FluidRenderProfileSettings;
    /** @internal */
    _colorTarget?: RenderTarget;
    /** @internal */
    _profiler: FluidSimulationProfiler | null;
    /** @internal */
    _particleStream: FluidParticleStream | null;
    /** @internal */
    _aggregate?: unknown;
    /** @internal */
    _aggregationTask?: unknown;
    /** @internal */
    _useParticleColor?: boolean;
    /** @internal */
    _foamPolygonSurfaceDepth?: boolean;
}

export interface FluidSimulationRenderSource {
    readonly simulation: FluidSimulation;
    /** Contiguous particle prefix to render. Defaults to the simulation's render/active count. */
    readonly count?: number;
    /** Constant per-source opacity when no alpha channel is supplied. */
    readonly opacity?: number;
    readonly alpha?: FluidParticleChannel;
    readonly color?: FluidParticleChannel;
}

export interface FluidSimulationCollection {
    readonly simulations: readonly FluidSimulation[];
    readonly sources: readonly FluidSimulationRenderSource[];
    readonly renderLayers: readonly FluidSimulationRenderLayer[];
    readonly disposed: boolean;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _simulations: FluidSimulation[];
    /** @internal */
    _sources: FluidSimulationRenderSource[];
    /** @internal */
    _renderLayers: FluidSimulationRenderLayer[];
}

/** Aggregated collection stream for gameplay queries and scene-specific effects. */
export interface FluidSimulationCollectionParticleStream {
    readonly stream: FluidParticleStream;
    readonly overflowed: boolean;
    readonly disposed: boolean;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _aggregate: unknown;
    /** @internal */
    _task: unknown;
    /** @internal */
    _scene: unknown;
    /** @internal */
    _automatic: boolean;
}

export interface FluidSimulationCollectionRenderLayerOptions extends Omit<FluidSimulationRenderLayerOptions, "mode"> {
    readonly mode: FluidSimulationCollectionRenderMode;
    /** Shared aggregate-stream capacity for sphere/surface rendering. */
    readonly particleCapacity?: number;
    /** Surface or polygon layer whose depth classifies collection foam. */
    readonly surfaceLayer?: FluidSimulationRenderLayer;
    /** Polygon layer whose depth replaces surfaceLayer while polygon foam filtering is active. */
    readonly polygonSurfaceLayer?: FluidSimulationRenderLayer;
    /** Exclude simulations with a live polygon surface from the shared particle stream. */
    readonly excludePolygonSurfaces?: boolean;
    /** Render diffuse particles only for simulations with a live polygon surface. */
    readonly polygonSurfacesOnly?: boolean;
    /** Insert aggregation and rendering before this facade compositor. */
    readonly beforeCompositor?: FluidSimulationRenderCompositor;
}

export interface FluidSimulationCollectionDiagnostics {
    readonly simulationCount: number;
    readonly count: number;
    readonly activeCount: number;
    readonly renderCount: number;
    readonly gpuBytes: number;
    readonly pressure: FluidPressureDiagnostics | null;
    readonly diffuse: FluidSimulationDiffuseDiagnostics | null;
    readonly polygons: readonly FluidSimulationPolygonDiagnostics[];
}

/** Fully prepared backend candidate. It has no effect on the live simulation until committed. */
export interface PreparedFluidReconfiguration {
    readonly simulation: FluidSimulation;
    readonly options: FluidSimulationOptions;
    readonly resolvedConfig: ResolvedFluidSimulationConfig;
    readonly steadyBytes: number;
    readonly transitionPeakBytes: number;
    readonly status: "prepared" | "committed" | "cancelled";
    /** @internal */
    _status: "prepared" | "committed" | "cancelled";
    /** @internal */
    _candidate?: unknown;
    /** @internal */
    _bindingsRetained: boolean;
    /** @internal */
    _preserveState: boolean;
}

export interface FluidSimulationReconfigurationRequest {
    readonly simulation: FluidSimulation;
    readonly updates: Partial<FluidSimulationOptions>;
    readonly preserveState?: boolean;
}

/** Aggregate, fully prepared replacement set that commits all stable handles together. */
export interface PreparedFluidCollectionReconfiguration {
    readonly reconfigurations: readonly PreparedFluidReconfiguration[];
    readonly steadyBytes: number;
    readonly transitionPeakBytes: number;
    readonly status: "prepared" | "committed" | "cancelled";
    /** @internal */
    _status: "prepared" | "committed" | "cancelled";
}

export interface FluidSimulationRenderCompositorOptions {
    readonly scene: SceneContext;
    readonly baseColorTarget: RenderTarget;
    readonly baseLayer?: FluidSimulationRenderLayer | null;
}

/** GPU-safe pure-state depth-aware compositor attachment. */
export interface FluidSimulationRenderCompositor {
    readonly layers: readonly FluidSimulationRenderLayer[];
    readonly baseLayer: FluidSimulationRenderLayer | null;
    disposed: boolean;
    /** @internal */
    _engine: unknown;
    /** @internal */
    _task: unknown;
    /** @internal */
    _scene: unknown;
    /** @internal */
    _layers: FluidSimulationRenderLayer[];
    /** @internal */
    _baseState: { layer: FluidSimulationRenderLayer | null };
}

const backendOf = (simulation: FluidSimulation): FluidSim => simulation._sim as FluidSim;
const engineOf = (simulation: FluidSimulation): EngineContext => simulation._engine as EngineContext;
const currentFrameEncoder = (engine: EngineContext, operation: string): GPUCommandEncoder => {
    if (!engine._currentEncoder) {
        throw new Error(`[fluid] ${operation} must be recorded during an active engine frame.`);
    }
    return engine._currentEncoder;
};
const particleTaskOf = (layer: FluidSimulationRenderLayer): ParticleRenderTask | undefined => layer._particleTask as ParticleRenderTask | undefined;
const surfaceTaskOf = (layer: FluidSimulationRenderLayer): FluidSurfaceTask | undefined => layer._surfaceTask as FluidSurfaceTask | undefined;
const polygonTaskOf = (layer: FluidSimulationRenderLayer): FluidPolygonSurfaceTask | undefined => layer._polygonTask as FluidPolygonSurfaceTask | undefined;
type FoamRenderTask = ReturnType<typeof createFoamRenderTask>;
const foamTaskOf = (layer: FluidSimulationRenderLayer): FoamRenderTask | undefined => layer._foamTask as FoamRenderTask | undefined;
const taskOf = (layer: FluidSimulationRenderLayer): Task => layer._task as Task;
const profilerImplOf = (profiler: FluidSimulationProfiler): FluidProfilerImpl => profiler._profiler as FluidProfilerImpl;
const profilerHookOf = (profiler: FluidSimulationProfiler | null | undefined): FluidProfiler | null => (profiler ? profilerImplOf(profiler) : null);
const sceneSdfBindingOf = (sceneSdf: FluidSceneSdf): SceneSdfRuntimeBinding => sceneSdf._binding as SceneSdfRuntimeBinding;
const sceneSdfSpecOf = (sceneSdf: FluidSceneSdf | null | undefined): SceneSdfSpec | null => (sceneSdf ? sceneSdfBindingOf(sceneSdf).spec : null);
const forceFieldBindingOf = (forceField: FluidForceField): ForceFieldRuntimeBinding => forceField._binding as ForceFieldRuntimeBinding;
const forceFieldSpecOf = (forceField: FluidForceField | null | undefined): ForceFieldSpec | null => (forceField ? forceFieldBindingOf(forceField).spec : null);
const FLUID_IMPULSE_FORCE_WGSL = /* wgsl */ `
fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> {
    let center = forceFieldParams.center.xyz;
    let radius = max(forceFieldParams.center.w, 1.0e-4);
    let toParticle = pos - center;
    let dist = length(toParticle);
    var force = forceFieldParams.push.xyz;
    if (dist < radius) {
        let direction = select(vec3<f32>(0.0, 1.0, 0.0), toParticle / max(dist, 1.0e-4), dist > 1.0e-4);
        let core = smoothstep(0.0, 0.15, dist / radius);
        force += direction * forceFieldParams.push.w * core;
    }
    return force * dt;
}`;

export function resolveFluidImpulseForce(options: FluidImpulseForceOptions): FluidImpulseForceResolution {
    if (!options.center.every(Number.isFinite) || !Number.isFinite(options.radius) || options.radius <= 0 || !Number.isFinite(options.intensity) || options.intensity < 0) {
        throw new RangeError("[fluid] impulse force requires a finite center, positive radius, and non-negative intensity.");
    }
    const directionLength = Math.sqrt(options.direction[0] * options.direction[0] + options.direction[1] * options.direction[1] + options.direction[2] * options.direction[2]);
    const fallback = options.fallbackDirection ?? [0, 1, 0];
    const resolved = directionLength > 1e-6 ? options.direction : fallback;
    const normalized = normalizedDirection(resolved, "impulse fallback direction");
    const directionalScale = options.directionalScale ?? 6;
    const radialScale = options.radialScale ?? 18;
    if (!Number.isFinite(directionalScale) || directionalScale < 0 || !Number.isFinite(radialScale) || radialScale < 0) {
        throw new RangeError("[fluid] impulse force scales must be finite and non-negative.");
    }
    return {
        direction: normalized,
        usedFallbackDirection: directionLength <= 1e-6,
        directionalMagnitude: directionalScale * options.intensity,
        radialMagnitude: radialScale * options.intensity,
    };
}

function fluidImpulseParams(options: FluidImpulseForceOptions): Float32Array {
    const resolved = resolveFluidImpulseForce(options);
    return new Float32Array([
        options.center[0],
        options.center[1],
        options.center[2],
        options.radius,
        resolved.direction[0] * resolved.directionalMagnitude,
        resolved.direction[1] * resolved.directionalMagnitude,
        resolved.direction[2] * resolved.directionalMagnitude,
        resolved.radialMagnitude,
    ]);
}
const compositorTaskOf = (compositor: FluidSimulationRenderCompositor): FluidRenderCompositor => compositor._task as FluidRenderCompositor;
const particleChannelRuntimeOf = (channel: FluidParticleChannel): FluidParticleChannelRuntime => channel._runtime as FluidParticleChannelRuntime;
const particleStreamRuntimeOf = (stream: FluidParticleStream): FluidParticleStreamRuntime => stream._resolve() as FluidParticleStreamRuntime;

/** @internal Scene-specific effects may consume an opaque stream without exposing buffers publicly. */
export function fluidParticleStreamForSceneIntegration(stream: FluidParticleStream): FluidParticleStreamRuntime {
    return particleStreamRuntimeOf(stream);
}

/** @internal Scene-specific effects may sample a facade-owned surface depth attachment. */
export function fluidSimulationRenderLayerDepthForSceneIntegration(layer: FluidSimulationRenderLayer): GPUTextureView | null {
    return layerDepthView(layer) as GPUTextureView | null;
}

/** @internal Legacy scene extensions must stay behind this source-only bridge. */
export function fluidSimulationBackendForSceneIntegration(simulation: FluidSimulation): FluidSim {
    return backendOf(simulation);
}

/** @internal Legacy scene extensions must stay behind this source-only bridge. */
export function fluidSimulationProfilerForSceneIntegration(profiler: FluidSimulationProfiler | null): FluidProfiler | null {
    return profilerHookOf(profiler);
}
const aggregateRuntimeOf = (layer: FluidSimulationRenderLayer): FluidAggregateRuntime | undefined => layer._aggregate as FluidAggregateRuntime | undefined;
const spatialQueryRuntimeOf = (query: FluidParticleSpatialQuery): FluidSpatialQueryRuntime => query._runtime as FluidSpatialQueryRuntime;
const wheelTorqueQueryRuntimeOf = (query: FluidWheelTorqueQuery): FluidWheelTorqueQueryRuntime => query._runtime as FluidWheelTorqueQueryRuntime;
let emptyInitialEmitterCounts: ReadonlyMap<string, number> | undefined;
const getEmptyInitialEmitterCounts = (): ReadonlyMap<string, number> => (emptyInitialEmitterCounts ??= new Map());

function removeFluidTasksFromScene(scene: SceneContext, tasks: readonly (Task | undefined)[]): void {
    if (!("_frameGraph" in scene)) {
        return;
    }
    const registered = getFrameGraph(scene)._tasks;
    for (const task of tasks) {
        if (!task) {
            continue;
        }
        const index = registered.indexOf(task);
        if (index >= 0) {
            registered.splice(index, 1);
        }
    }
}

function physicsValue(physics: Readonly<Record<string, number>>, key: string): number | undefined {
    const value = physics[key];
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function finitePositive(value: number, label: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new RangeError(`[fluid] ${label} must be a finite positive number.`);
    }
    return value;
}

function normalizedDirection(value: readonly [number, number, number], label: string): [number, number, number] {
    if (!value.every(Number.isFinite)) {
        throw new TypeError(`[fluid] ${label} must contain finite values.`);
    }
    const length = Math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    if (length <= 0) {
        throw new RangeError(`[fluid] ${label} must be non-zero.`);
    }
    return [value[0] / length, value[1] / length, value[2] / length];
}

function validateBindingEngine(engine: EngineContext, binding: { readonly _engine: unknown } | null | undefined, label: string): void {
    if (binding && binding._engine !== engine) {
        throw new Error(`[fluid] ${label} belongs to a different engine.`);
    }
}

function validateSimulationBindings(engine: EngineContext, options: FluidSimulationOptions): void {
    validateBindingEngine(engine, options.sceneSdf, "scene SDF");
    validateBindingEngine(engine, options.forceField, "force field");
    validateBindingEngine(engine, options.profiler, "profiler");
}

function normalizeFluidSimulationOptions(options: FluidSimulationOptions): FluidSimulationOptions {
    if (!["PBF", "FLIP", "MLS-MPM", "PB-MPM"].includes(options.method)) {
        throw new TypeError(`[fluid] unsupported simulation method '${String(options.method)}'.`);
    }
    if (options.backend && (options.backend.method !== options.method || typeof options.backend._create !== "function")) {
        throw new TypeError("[fluid] backend provider must support the selected intrinsic method.");
    }
    if (options.backend && !options.backend.supportsFoam && options.foam) {
        throw new Error("[fluid] " + options.backend.name + " does not support foam.");
    }
    if (options.backend && !options.backend.supportsForces && options.forceField) {
        throw new Error("[fluid] " + options.backend.name + " does not support custom force fields.");
    }
    finitePositive(options.particleCount, "particleCount");
    finitePositive(options.physicsScale, "physicsScale");
    const min = [...options.bounds.min] as [number, number, number];
    const max = [...options.bounds.max] as [number, number, number];
    for (const axis of [0, 1, 2] as const) {
        if (!Number.isFinite(min[axis]) || !Number.isFinite(max[axis]) || max[axis] <= min[axis]) {
            throw new RangeError("[fluid] bounds must contain finite, increasing axis extents.");
        }
    }
    for (const [key, value] of Object.entries(options.physics ?? {})) {
        if (!Number.isFinite(value)) {
            throw new TypeError(`[fluid] physics.${key} must be finite.`);
        }
    }
    if (options.sceneSdf?.disposed) {
        throw new Error("[fluid] cannot attach a disposed scene SDF.");
    }
    if (options.forceField?.disposed) {
        throw new Error("[fluid] cannot attach a disposed force field.");
    }
    if (options.profiler?.disposed) {
        throw new Error("[fluid] cannot attach a disposed profiler.");
    }
    return {
        ...options,
        particleCount: Math.max(1, Math.floor(options.particleCount)),
        bounds: { min, max },
        physics: options.physics ? { ...options.physics } : undefined,
        flow: options.flow ? structuredClone(options.flow) : undefined,
        foam: options.foam ? { ...options.foam } : options.foam,
        initialPositions: options.initialPositions?.slice(),
        initialVelocities: options.initialVelocities?.slice(),
    };
}

function createBackend(engine: EngineContext, options: FluidSimulationOptions, config: ResolvedFluidSimulationConfig): FluidSim {
    if (options.backend) {
        const backend = options.backend._create(engine, options, config);
        if (options.backend.steppingMode === "async" && !backend.submitStep) {
            backend.dispose();
            throw new Error("[fluid] asynchronous backend provider did not supply asynchronous stepping.");
        }
        return backend;
    }
    const physics = config.physics;
    const base = {
        count: options.particleCount,
        boundsMin: [...options.bounds.min] as [number, number, number],
        boundsMax: [...options.bounds.max] as [number, number, number],
        groundY: options.groundY ?? options.bounds.min[1],
        particleRadius: config.particleRadius,
        ...(options.initialPositions ? { initialPositions: options.initialPositions } : {}),
    };
    if (options.method === "PBF") {
        const pbf: PbfOptions = {
            ...base,
            smoothingRadius: config.cellSize,
            particleVolume: config.particleVolume,
            maxPerCell: 48,
            ...(physicsValue(physics, "gravity") !== undefined ? { gravity: physicsValue(physics, "gravity") } : {}),
            ...(physicsValue(physics, "viscosity") !== undefined ? { viscosity: physicsValue(physics, "viscosity") } : {}),
            ...(physicsValue(physics, "relaxation") !== undefined ? { relaxation: physicsValue(physics, "relaxation") } : {}),
            ...(physicsValue(physics, "scorr") !== undefined ? { scorr: physicsValue(physics, "scorr") } : {}),
            ...(physicsValue(physics, "iterations") !== undefined ? { iterations: physicsValue(physics, "iterations") } : {}),
            ...(physicsValue(physics, "restDensity") !== undefined ? { restDensity: physicsValue(physics, "restDensity") } : {}),
            ...(physicsValue(physics, "boundaryDensity") !== undefined ? { boundaryDensity: physicsValue(physics, "boundaryDensity") } : {}),
        };
        return createPbfSim(engine, pbf);
    }
    if (options.method === "FLIP") {
        const flip: FlipOptions = {
            ...base,
            ...(options.initialVelocities ? { initialVelocities: options.initialVelocities } : {}),
            ...(options.gridResolution !== undefined ? { gridResolution: options.gridResolution } : { dx: config.cellSize }),
            markersPerCell: options.markersPerCell,
            pagedGrid: options.pagedGrid,
            pagedGridMaxPages: options.pagedGridMaxPages,
            onPagedGridPages: options.onPagedGridPages,
            onPagedGridOverflow: options.onPagedGridOverflow,
            gravity: physicsValue(physics, "gravity"),
            flipRatio: physicsValue(physics, "flipRatio"),
            pressureIterations: physicsValue(physics, "pressureIterations"),
            pressureRelaxation: physicsValue(physics, "pressureRelaxation"),
            pressureSolver: (physicsValue(physics, "pressureSolver") ?? 0) >= 0.5 ? "multigrid" : "jacobi",
            multigridCycles: physicsValue(physics, "multigridCycles"),
            pressureTolerance: physicsValue(physics, "pressureTolerance"),
            pressureDiagnostics: (physicsValue(physics, "pressureDiagnostics") ?? 0) >= 0.5,
            velocityDamping: physicsValue(physics, "velocityDamping"),
            kinematicViscosity: physicsValue(physics, "kinematicViscosity"),
            viscosityIterations: physicsValue(physics, "viscosityIterations"),
            surfaceTension: physicsValue(physics, "surfaceTension"),
            liquidSdf: (physicsValue(physics, "liquidSdf") ?? 0) >= 0.5,
            ghostFluid: (physicsValue(physics, "ghostFluid") ?? 0) >= 0.5,
            fractionalSolids: (physicsValue(physics, "fractionalSolids") ?? 0) >= 0.5,
            movingSolidBoundaries: (physicsValue(physics, "movingSolidBoundaries") ?? 0) >= 0.5,
            reseedParticles: (physicsValue(physics, "reseedParticles") ?? 0) >= 0.5,
            reseedMinParticles: physicsValue(physics, "reseedMinParticles"),
            reseedTargetParticles: physicsValue(physics, "reseedTargetParticles"),
            reseedMaxParticles: physicsValue(physics, "reseedMaxParticles"),
            reseedInterval: physicsValue(physics, "reseedInterval"),
            particleSheeting: (physicsValue(physics, "particleSheeting") ?? 0) >= 0.5,
            sheetingStrength: physicsValue(physics, "sheetingStrength"),
            sheetingInterval: physicsValue(physics, "sheetingInterval"),
            polygonSurface: (physicsValue(physics, "polygonSurface") ?? 0) >= 0.5,
            polygonReconstructionMultiplier: physicsValue(physics, "polygonReconstructionMultiplier"),
            restitution: physicsValue(physics, "restitution"),
            minSubsteps: physicsValue(physics, "minSubsteps"),
            maxSubsteps: physicsValue(physics, "maxSubsteps"),
            cflNumber: physicsValue(physics, "cflNumber"),
            ...(physicsValue(physics, "maxSubDtMs") !== undefined ? { maxSubDt: physicsValue(physics, "maxSubDtMs")! / 1000 } : {}),
        };
        return createFlipSim(engine, flip);
    }
    if (options.method === "MLS-MPM") {
        const mls: MlsMpmOptions = {
            ...base,
            dx: config.cellSize,
            particleVolume: config.particleVolume,
            activeBlocks: options.activeBlocks,
            pagedGrid: options.pagedGrid,
            pagedGridMaxPages: options.pagedGridMaxPages,
            fusedBlockDiscovery: options.fusedBlockDiscovery,
            onPagedGridPages: options.onPagedGridPages,
            onPagedGridOverflow: options.onPagedGridOverflow,
            gravity: physicsValue(physics, "gravity"),
            stiffness: physicsValue(physics, "stiffness"),
            viscosity: physicsValue(physics, "viscosity"),
            restDensity: physicsValue(physics, "restDensity"),
            damping: physicsValue(physics, "damping"),
            affineDamping: physicsValue(physics, "affineDamping"),
            groundDamp: physicsValue(physics, "groundDamp"),
            groundDampHeight: physicsValue(physics, "groundDampHeight"),
            restitution: physicsValue(physics, "restitution"),
            substeps: physicsValue(physics, "substeps"),
            ...(physicsValue(physics, "maxSubDtMs") !== undefined ? { maxSubDt: physicsValue(physics, "maxSubDtMs")! / 1000 } : {}),
        };
        return createMlsMpmSim(engine, mls);
    }
    const pbmpm: PbMpmOptions = {
        ...base,
        dx: config.cellSize,
        particleVolume: config.particleVolume,
        material: options.material,
        gravity: physicsValue(physics, "gravity"),
        iterations: physicsValue(physics, "iterations"),
        liquidRelaxation: physicsValue(physics, "liquidRelaxation"),
        liquidViscosity: physicsValue(physics, "liquidViscosity"),
        elasticityRatio: physicsValue(physics, "elasticityRatio"),
        elasticRelaxation: physicsValue(physics, "elasticRelaxation"),
        frictionAngle: physicsValue(physics, "frictionAngle"),
        plasticity: physicsValue(physics, "plasticity"),
        restitution: physicsValue(physics, "restitution"),
        substeps: physicsValue(physics, "substeps"),
        ...(physicsValue(physics, "maxSubDtMs") !== undefined ? { maxSubDt: physicsValue(physics, "maxSubDtMs")! / 1000 } : {}),
    };
    return createPbMpmSim(engine, pbmpm);
}

function resolveOptions(options: FluidSimulationOptions): ResolvedFluidSimulationConfig {
    return resolveFluidSimulationConfig(options.compatibilityProfile ?? "fluid", {
        method: options.method,
        physicsScale: options.physicsScale,
        samplingType: options.samplingType,
        particleRadius: options.particleRadius,
        bounds: options.bounds,
        gridResolution: options.gridResolution,
        markersPerCell: options.markersPerCell,
        explicitGrid: options.explicitGrid ?? true,
        domainScale: options.domainScale,
        physics: options.physics,
        semantics: options.semantics,
    });
}

function configureBackend(sim: FluidSim, options: FluidSimulationOptions): void {
    if (options.flow) {
        sim.setFlow(structuredClone(options.flow));
        sim.reset();
    }
    if (options.foam !== undefined) {
        sim.setFoam?.(options.foam ? { ...options.foam } : null);
    }
    sim.setSceneSdf(sceneSdfSpecOf(options.sceneSdf));
    sim.setForceField(forceFieldSpecOf(options.forceField));
    sim.setProfiler?.(profilerHookOf(options.profiler));
}

function retainSimulationBindings(simulation: FluidSimulation, options: FluidSimulationOptions): void {
    options.sceneSdf?._owners.add(simulation);
    options.forceField?._owners.add(simulation);
    options.profiler?._owners.add(simulation);
}

function releaseSimulationBindings(simulation: FluidSimulation, options: FluidSimulationOptions): void {
    options.sceneSdf?._owners.delete(simulation);
    options.forceField?._owners.delete(simulation);
    options.profiler?._owners.delete(simulation);
}

function retainPreparedBindings(prepared: PreparedFluidReconfiguration): void {
    prepared.options.sceneSdf?._owners.add(prepared);
    prepared.options.forceField?._owners.add(prepared);
    prepared.options.profiler?._owners.add(prepared);
    prepared._bindingsRetained = true;
}

function releasePreparedBindings(prepared: PreparedFluidReconfiguration): void {
    if (!prepared._bindingsRetained) {
        return;
    }
    prepared.options.sceneSdf?._owners.delete(prepared);
    prepared.options.forceField?._owners.delete(prepared);
    prepared.options.profiler?._owners.delete(prepared);
    prepared._bindingsRetained = false;
}

function setLayerBackend(layer: FluidSimulationRenderLayer, backend: FluidSim): void {
    particleTaskOf(layer)?.setSim(backend);
    surfaceTaskOf(layer)?.setSim(backend);
    polygonTaskOf(layer)?.setSim(backend);
    foamTaskOf(layer)?.setSim(backend);
}

function setLayerProfiler(layer: FluidSimulationRenderLayer, profiler: FluidSimulationProfiler | null): void {
    const hook = profilerHookOf(profiler);
    particleTaskOf(layer)?.setProfiler(hook);
    surfaceTaskOf(layer)?.setProfiler(hook);
    polygonTaskOf(layer)?.setProfiler(hook);
    foamTaskOf(layer)?.setProfiler(hook);
}

function applyPolygonRenderProfile(task: FluidPolygonSurfaceTask, profile: FluidRenderProfileSettings): void {
    const color = fluidRenderHexColor(profile.waterColor);
    if (color) {
        task.setFluidColor(color);
    }
    if (profile.polygonShader !== undefined) {
        task.setShadingMode(profile.polygonShader);
    }
    if (profile.absorption !== undefined) {
        task.setAbsorption(profile.absorption);
    }
    if (profile.refractionStrength !== undefined) {
        task.setRefractionStrength(profile.refractionStrength);
    }
    if (profile.specularPower !== undefined) {
        task.setSpecularPower(profile.specularPower);
    }
    if (profile.reflectionExposure !== undefined || profile.reflectionContrast !== undefined) {
        task.setEnvReflection(profile.reflectionExposure ?? 1, profile.reflectionContrast ?? 1.1);
    }
    if (profile.waterReflectivity !== undefined) {
        task.setFresnelF0(profile.waterReflectivity);
    }
}

function applyRenderLayerProfile(layer: FluidSimulationRenderLayer, profile: FluidRenderProfileSettings): void {
    const particle = particleTaskOf(layer);
    if (particle) {
        const color = fluidRenderHexColor(profile.waterColor);
        if (color) {
            particle.setTint(color);
        }
        if (profile.particleSize !== undefined) {
            particle.setSizeScale(profile.particleSize);
        }
    }
    const surface = surfaceTaskOf(layer);
    if (surface) {
        applyFluidRenderProfile(surface, profile);
    }
    const polygon = polygonTaskOf(layer);
    if (polygon) {
        applyPolygonRenderProfile(polygon, profile);
    }
}

function applyFoamRenderSettings(task: FoamRenderTask, settings: FluidFoamRenderSettings): void {
    if (settings.enabled !== undefined) {
        task.setEnabled(settings.enabled);
    }
    if (settings.polygonSurfaceDepth !== undefined) {
        task.setPolygonSurfaceDepth(settings.polygonSurfaceDepth);
    }
    if (settings.surfaceFiltering !== undefined) {
        task.setSurfaceFiltering(settings.surfaceFiltering);
    }
    if (settings.opacity !== undefined) {
        task.setOpacity(settings.opacity);
    }
    if (settings.sizeScale !== undefined) {
        task.setSizeScale(settings.sizeScale);
    }
    if (settings.spraySize !== undefined) {
        task.setSpraySize(settings.spraySize);
    }
    if (settings.sprayIntensity !== undefined) {
        task.setSprayIntensity(settings.sprayIntensity);
    }
    if (settings.spraySeparation !== undefined) {
        task.setSpraySeparation(settings.spraySeparation);
    }
    if (settings.debugByKind !== undefined) {
        task.setDebugByKind(settings.debugByKind);
    }
    if (settings.softness !== undefined || settings.density !== undefined) {
        task.setThresholds(settings.softness ?? 0.25, settings.density ?? 1.6);
    }
    if (settings.subsurfaceStrength !== undefined) {
        task.setSubsurfaceStrength(settings.subsurfaceStrength);
    }
    const subsurfaceColor = fluidRenderHexColor(settings.subsurfaceColor);
    if (subsurfaceColor) {
        task.setSubsurfaceColor(subsurfaceColor);
    }
    if (settings.blurRadius !== undefined) {
        task.setBlurRadius(settings.blurRadius);
    }
    if (settings.debugTexture !== undefined) {
        task.setDebugTexture(settings.debugTexture);
    }
    if (settings.lightIntensity !== undefined) {
        task.setLightIntensity(settings.lightIntensity);
    }
    if (settings.ambient !== undefined) {
        task.setAmbient(settings.ambient);
    }
    if (settings.aoStrength !== undefined) {
        task.setAOStrength(settings.aoStrength);
    }
    if (settings.normalStrength !== undefined) {
        task.setNormalStrength(settings.normalStrength);
    }
}

function applyRenderLayerEnvironment(layer: FluidSimulationRenderLayer, scene: SceneContext): void {
    const environment = scene._envTextures;
    if (!environment) {
        throw new Error("[fluid] the selected scene has no environment textures.");
    }
    const value = { view: environment.specularCubeView, sampler: environment.cubeSampler };
    surfaceTaskOf(layer)?.setEnvMap(value);
    polygonTaskOf(layer)?.setEnvMap(value);
}

function applyOpaqueRenderLayerEnvironment(layer: FluidSimulationRenderLayer, environment: FluidRenderEnvironment): void {
    const source = environment._source as { readonly specularCubeView: unknown; readonly cubeSampler: unknown };
    const value = { view: source.specularCubeView, sampler: source.cubeSampler };
    surfaceTaskOf(layer)?.setEnvMap(value as never);
    polygonTaskOf(layer)?.setEnvMap(value as never);
}

function layerDepthView(layer: FluidSimulationRenderLayer): unknown {
    return surfaceTaskOf(layer)?.surfaceDepthView() ?? polygonTaskOf(layer)?.surfaceDepthView() ?? null;
}

/** Create a root-import-only fluid simulation without exposing the underlying WebGPU backend. */
export function createFluidSimulation(engine: EngineContext, options: FluidSimulationOptions): FluidSimulation {
    const normalized = normalizeFluidSimulationOptions(options);
    validateSimulationBindings(engine, normalized);
    const resolvedConfig = resolveOptions(normalized);
    let backend: FluidSim | undefined;
    try {
        backend = createBackend(engine, normalized, resolvedConfig);
        configureBackend(backend, normalized);
        backend.prepare?.();
        const renderLayers: FluidSimulationRenderLayer[] = [];
        const simulation = {
            get method(): FluidSimulationOptions["method"] {
                return simulation._method;
            },
            get options(): FluidSimulationOptions {
                return simulation._options;
            },
            get steppingMode(): "frame" | "async" {
                return simulation._options.backend?.steppingMode ?? "frame";
            },
            get count(): number {
                return backendOf(simulation).count;
            },
            get activeCount(): number | undefined {
                return backendOf(simulation).activeCount;
            },
            get renderCount(): number | undefined {
                return backendOf(simulation).renderCount;
            },
            get particleRadius(): number {
                return backendOf(simulation).particleRadius;
            },
            get surfaceSizeScale(): number {
                return backendOf(simulation).surfaceSizeScale ?? 1;
            },
            get surfaceThicknessScale(): number {
                return backendOf(simulation).surfaceThicknessScale ?? 1;
            },
            get surfaceRejectSparseMarkers(): boolean {
                return backendOf(simulation).surfaceRejectSparseMarkers ?? false;
            },
            get initialEmitterParticleCounts(): ReadonlyMap<string, number> {
                return backendOf(simulation).initialEmitterParticleCounts ?? getEmptyInitialEmitterCounts();
            },
            get gpuBytes(): number {
                return backendOf(simulation).gpuBytes;
            },
            get pressureDiagnostics(): FluidPressureDiagnostics | undefined {
                return backendOf(simulation).pressureDiagnostics;
            },
            get polygonTriangleCount(): number | undefined {
                return backendOf(simulation).polygonSurface?.triangleCount;
            },
            get resolvedConfig(): ResolvedFluidSimulationConfig {
                return simulation._resolvedConfig;
            },
            get sceneSdf(): FluidSceneSdf | null {
                return simulation._options.sceneSdf ?? null;
            },
            get forceField(): FluidForceField | null {
                return simulation._options.forceField ?? null;
            },
            get profiler(): FluidSimulationProfiler | null {
                return simulation._options.profiler ?? null;
            },
            renderLayers,
            disposed: false,
            _engine: engine,
            _sim: backend,
            _method: normalized.method,
            _options: normalized,
            _resolvedConfig: resolvedConfig,
            _renderLayers: renderLayers,
        } satisfies FluidSimulation;
        retainSimulationBindings(simulation, normalized);
        return simulation;
    } catch (error) {
        backend?.dispose();
        throw error;
    }
}

/**
 * Validate, normalize, allocate, and configure a detached candidate. Compatible FLIP state is
 * captured at commit so preparation cannot install a stale snapshot.
 */
export function prepareFluidReconfiguration(simulation: FluidSimulation, options: FluidSimulationOptions, preserveState = false): PreparedFluidReconfiguration {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot reconfigure a disposed simulation.");
    }
    if (preserveState && (simulation._options.backend || options.backend)) {
        throw new Error("[fluid] state-preserving reconfiguration is not supported by implementation overrides; reset the selected backend instead.");
    }
    const normalized = normalizeFluidSimulationOptions(options);
    const engine = engineOf(simulation);
    validateSimulationBindings(engine, normalized);
    const resolvedConfig = resolveOptions(normalized);
    let candidate: FluidSim | undefined;
    let prepared = false;
    try {
        candidate = createBackend(engine, normalized, resolvedConfig);
        configureBackend(candidate, normalized);
        candidate.prepare?.();
        const result = {
            simulation,
            options: normalized,
            resolvedConfig,
            steadyBytes: candidate.gpuBytes,
            transitionPeakBytes: simulation.gpuBytes + candidate.gpuBytes,
            get status(): "prepared" | "committed" | "cancelled" {
                return result._status;
            },
            _status: "prepared",
            _candidate: candidate,
            _bindingsRetained: false,
            _preserveState: preserveState,
        } satisfies PreparedFluidReconfiguration;
        retainPreparedBindings(result);
        prepared = true;
        return result;
    } finally {
        if (!prepared) {
            candidate?.dispose();
        }
    }
}

/** Prepare a candidate by applying a pure-state option patch to the simulation's committed options. */
export function prepareFluidReconfigurationUpdate(simulation: FluidSimulation, updates: Partial<FluidSimulationOptions>, preserveState = false): PreparedFluidReconfiguration {
    return prepareFluidReconfiguration(
        simulation,
        {
            ...simulation._options,
            ...updates,
            method: updates.method ?? simulation._options.method,
            particleCount: updates.particleCount ?? simulation._options.particleCount,
            bounds: updates.bounds ?? simulation._options.bounds,
        },
        preserveState
    );
}

interface FluidReconfigurationCommitEntry {
    readonly prepared: PreparedFluidReconfiguration;
    readonly simulation: FluidSimulation;
    readonly previous: FluidSim;
    readonly next: FluidSim;
    readonly previousOptions: FluidSimulationOptions;
}

function validatedCommitEntries(reconfigurations: readonly PreparedFluidReconfiguration[]): FluidReconfigurationCommitEntry[] {
    const entries: FluidReconfigurationCommitEntry[] = [];
    const simulations = new Set<FluidSimulation>();
    let engine: EngineContext | null = null;
    for (const prepared of reconfigurations) {
        if (prepared._status !== "prepared") {
            throw new Error(`[fluid] cannot commit a ${prepared._status} reconfiguration.`);
        }
        const simulation = prepared.simulation;
        if (simulation.disposed) {
            throw new Error("[fluid] cannot commit into a disposed simulation.");
        }
        if (simulations.has(simulation)) {
            throw new Error("[fluid] a collection reconfiguration cannot replace the same simulation twice.");
        }
        simulations.add(simulation);
        const simulationEngine = engineOf(simulation);
        if (engine && engine !== simulationEngine) {
            throw new Error("[fluid] collection reconfiguration requires simulations from one engine.");
        }
        engine = simulationEngine;
        validateSimulationBindings(simulationEngine, prepared.options);
        const next = prepared._candidate as FluidSim | undefined;
        if (!next) {
            throw new Error("[fluid] prepared reconfiguration has no candidate backend.");
        }
        entries.push({
            prepared,
            simulation,
            previous: backendOf(simulation),
            next,
            previousOptions: simulation._options,
        });
    }
    return entries;
}

function retargetCommitEntries(entries: readonly FluidReconfigurationCommitEntry[], target: "next" | "previous"): void {
    for (const entry of entries) {
        const backend = target === "next" ? entry.next : entry.previous;
        for (const layer of entry.simulation._renderLayers) {
            setLayerBackend(layer, backend);
        }
    }
}

function commitPreparedReconfigurations(reconfigurations: readonly PreparedFluidReconfiguration[]): void {
    const entries = validatedCommitEntries(reconfigurations);
    if (entries.length === 0) {
        return;
    }
    try {
        retargetCommitEntries(entries, "next");
        const engine = engineOf(entries[0]!.simulation);
        const transferCommits: Array<() => void> = [];
        const transferEntries = entries.filter((entry) => entry.prepared._preserveState && entry.simulation.method === "FLIP" && entry.prepared.options.method === "FLIP");
        if (transferEntries.length > 0) {
            submitGpuOperation(engine, "fluid-reconfiguration-transfer", (encoder) => {
                for (const entry of transferEntries) {
                    const commitTransfer = encodeFlipSimStateTransfer(encoder, entry.previous, entry.next);
                    if (commitTransfer) {
                        transferCommits.push(commitTransfer);
                    }
                }
            });
        }
        for (const commitTransfer of transferCommits) {
            commitTransfer();
        }
    } catch (error) {
        retargetCommitEntries(entries, "previous");
        throw error;
    }

    for (const entry of entries) {
        releaseSimulationBindings(entry.simulation, entry.previousOptions);
        entry.simulation._sim = entry.next;
        entry.simulation._method = entry.prepared.options.method;
        entry.simulation._options = entry.prepared.options;
        entry.simulation._resolvedConfig = entry.prepared.resolvedConfig;
        retainSimulationBindings(entry.simulation, entry.prepared.options);
        releasePreparedBindings(entry.prepared);
        entry.prepared._candidate = undefined;
        entry.prepared._status = "committed";
    }
    for (const entry of entries) {
        retireGpuResources(engineOf(entry.simulation), () => entry.previous.dispose());
    }
}

/** Commit a prepared candidate, then retire the replaced backend behind the next submission fence. */
export function commitFluidReconfiguration(prepared: PreparedFluidReconfiguration): void {
    commitPreparedReconfigurations([prepared]);
}

/**
 * Prepare a same-engine set of simulation updates and expose its aggregate steady-state and
 * old-plus-new transition allocation. Preparation is fail-atomic.
 */
export function prepareFluidCollectionReconfiguration(requests: readonly FluidSimulationReconfigurationRequest[]): PreparedFluidCollectionReconfiguration {
    const prepared: PreparedFluidReconfiguration[] = [];
    try {
        for (const request of requests) {
            prepared.push(prepareFluidReconfigurationUpdate(request.simulation, request.updates, request.preserveState ?? false));
        }
        validatedCommitEntries(prepared);
    } catch (error) {
        for (const candidate of prepared) {
            cancelFluidReconfiguration(candidate);
        }
        throw error;
    }
    const result = {
        reconfigurations: prepared,
        steadyBytes: prepared.reduce((total, candidate) => total + candidate.steadyBytes, 0),
        transitionPeakBytes: prepared.reduce((total, candidate) => total + candidate.transitionPeakBytes, 0),
        get status(): "prepared" | "committed" | "cancelled" {
            return result._status;
        },
        _status: "prepared",
    } satisfies PreparedFluidCollectionReconfiguration;
    return result;
}

/** Retarget and replace every prepared simulation as one stable-handle commit. */
export function commitFluidCollectionReconfiguration(prepared: PreparedFluidCollectionReconfiguration): void {
    if (prepared._status !== "prepared") {
        throw new Error(`[fluid] cannot commit a ${prepared._status} collection reconfiguration.`);
    }
    commitPreparedReconfigurations(prepared.reconfigurations);
    prepared._status = "committed";
}

/** Cancel every candidate in an uncommitted collection reconfiguration. */
export function cancelFluidCollectionReconfiguration(prepared: PreparedFluidCollectionReconfiguration): void {
    if (prepared._status !== "prepared") {
        return;
    }
    for (const candidate of prepared.reconfigurations) {
        cancelFluidReconfiguration(candidate);
    }
    prepared._status = "cancelled";
}

/** Cancel an uncommitted candidate and retire all of its resources safely. */
export function cancelFluidReconfiguration(prepared: PreparedFluidReconfiguration): void {
    if (prepared._status !== "prepared") {
        return;
    }
    const candidate = prepared._candidate as FluidSim | undefined;
    prepared._candidate = undefined;
    prepared._status = "cancelled";
    if (candidate) {
        retireGpuResources(engineOf(prepared.simulation), () => {
            candidate.dispose();
            releasePreparedBindings(prepared);
        });
    } else {
        releasePreparedBindings(prepared);
    }
}

/** Dispose is the cancellation operation for a prepared candidate. */
export function disposePreparedFluidReconfiguration(prepared: PreparedFluidReconfiguration): void {
    cancelFluidReconfiguration(prepared);
}

/** Convenience rebuild implemented as prepare followed by commit, with rollback on commit failure. */
export function reconfigureFluidSimulation(simulation: FluidSimulation, options: FluidSimulationOptions, preserveState = false): void {
    const prepared = prepareFluidReconfiguration(simulation, options, preserveState);
    try {
        commitFluidReconfiguration(prepared);
    } catch (error) {
        cancelFluidReconfiguration(prepared);
        throw error;
    }
}

/** Encode one step into the engine's current frame command stream. */
export function stepFluidSimulation(simulation: FluidSimulation, deltaSeconds: number): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot step a disposed simulation.");
    }
    if (simulation.steppingMode === "async") {
        throw new Error("[fluid] this backend requires awaited submitFluidSimulationStep outside render-frame recording.");
    }

    if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) {
        return;
    }
    const engine = engineOf(simulation);
    backendOf(simulation).step(currentFrameEncoder(engine, "simulation stepping"), deltaSeconds);
}

/** Submit one fixed simulation frame outside the live render loop and resolve when its GPU work completes. */
export async function submitFluidSimulationStep(simulation: FluidSimulation, deltaSeconds: number, options: FluidSimulationStepOptions = {}): Promise<void> {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot step a disposed simulation.");
    }
    if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) {
        throw new RangeError("[fluid] offline simulation stepping requires a finite positive timestep.");
    }
    const engine = engineOf(simulation);
    if (engine._currentEncoder) {
        throw new Error("[fluid] cannot submit an offline simulation step while a render frame is being recorded.");
    }
    const backend = backendOf(simulation);
    if (backend.submitStep) {
        await backend.submitStep(deltaSeconds, options.beforeSubstep);
        return;
    }
    options.beforeSubstep?.(deltaSeconds);
    const encoder = engine._device.createCommandEncoder({ label: "fluid-offline-step" });
    backend.step(encoder, deltaSeconds);
    engine._device.queue.submit([encoder.finish()]);
    await engine._device.queue.onSubmittedWorkDone();
}

/** Submit multiple fixed production-solver frames in ordered command buffers and resolve after completion. */
export async function submitFluidSimulationSteps(simulation: FluidSimulation, deltaSeconds: number, count: number): Promise<void> {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot step a disposed simulation.");
    }
    if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0 || !Number.isInteger(count) || count <= 0) {
        throw new RangeError("[fluid] batched simulation stepping requires a finite positive timestep and positive integer count.");
    }
    const engine = engineOf(simulation);
    if (engine._currentEncoder) {
        throw new Error("[fluid] cannot submit batched simulation steps while a render frame is being recorded.");
    }
    const backend = backendOf(simulation);
    if (backend.submitStep) {
        for (let step = 0; step < count; step++) {
            await backend.submitStep(deltaSeconds);
        }
        return;
    }
    for (let step = 0; step < count; step++) {
        const encoder = engine._device.createCommandEncoder({ label: "fluid-offline-step-batch" });
        backend.step(encoder, deltaSeconds);
        engine._device.queue.submit([encoder.finish()]);
    }
    await engine._device.queue.onSubmittedWorkDone();
}

/** Clear the active backend's motion state without changing its current particle positions. */
export async function settleFluidSimulation(simulation: FluidSimulation): Promise<void> {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot settle a disposed simulation.");
    }
    const engine = engineOf(simulation);
    if (engine._currentEncoder) {
        throw new Error("[fluid] cannot settle simulation motion while a render frame is being recorded.");
    }
    const settle = backendOf(simulation).settle;
    if (!settle) {
        return;
    }
    const encoder = engine._device.createCommandEncoder({ label: "fluid-settle-motion" });
    settle(encoder);
    engine._device.queue.submit([encoder.finish()]);
    await engine._device.queue.onSubmittedWorkDone();
}

/** @internal Warm-up and scene integration may encode a facade simulation outside the live frame. */
export function stepFluidSimulationForSceneIntegration(simulation: FluidSimulation, encoder: GPUCommandEncoder, deltaSeconds: number): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot step a disposed simulation.");
    }
    if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) {
        return;
    }
    backendOf(simulation).step(encoder, deltaSeconds);
}

export function resetFluidSimulation(simulation: FluidSimulation): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot reset a disposed simulation.");
    }
    backendOf(simulation).reset();
}

export function setFluidSimulationFlow(simulation: FluidSimulation, flow: FluidFlowConfig | null, reset = false): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot configure flow on a disposed simulation.");
    }
    const snapshot = flow ? structuredClone(flow) : null;
    backendOf(simulation).setFlow(snapshot);
    if (reset) {
        backendOf(simulation).reset();
    }
    simulation._options = {
        ...simulation._options,
        ...(snapshot ? { flow: snapshot } : { flow: undefined }),
    };
}

export function updateFluidSimulationEmitter(simulation: FluidSimulation, emitter: FluidEmitter): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot update an emitter on a disposed simulation.");
    }
    const snapshot = structuredClone(emitter);
    backendOf(simulation).updateFlowEmitter(snapshot);
    const flow = simulation._options.flow;
    if (flow) {
        simulation._options = {
            ...simulation._options,
            flow: {
                ...flow,
                emitters: flow.emitters.map((entry) => (entry.id === snapshot.id ? snapshot : entry)),
            },
        };
    }
}

export function setFluidSimulationFoam(simulation: FluidSimulation, foam: FoamConfig | null): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot configure foam on a disposed simulation.");
    }
    const backend = backendOf(simulation);
    if (!backend.setFoam && foam) {
        throw new Error(`[fluid] ${simulation.method} does not support diffuse particles.`);
    }
    const snapshot = foam ? { ...foam } : null;
    backend.setFoam?.(snapshot);
    simulation._options = { ...simulation._options, foam: snapshot };
}

export function setFluidSimulationParameter(simulation: FluidSimulation, key: string, value: number): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot configure a disposed simulation.");
    }
    if (!key || !Number.isFinite(value)) {
        throw new TypeError("[fluid] simulation parameters require a key and finite value.");
    }
    const physics = { ...(simulation._options.physics ?? {}), [key]: value };
    const resolvedConfig = resolveOptions({ ...simulation._options, physics });
    backendOf(simulation).setParam(key, resolvedConfig.physics[key] ?? value);
    simulation._options = {
        ...simulation._options,
        physics,
    };
    simulation._resolvedConfig = resolvedConfig;
}

export function setFluidSimulationMaterial(simulation: FluidSimulation, material: number): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot configure material on a disposed simulation.");
    }
    const backend = backendOf(simulation);
    if (!backend.setMaterial) {
        throw new Error(`[fluid] ${simulation.method} does not support material selection.`);
    }
    if (!Number.isFinite(material)) {
        throw new TypeError("[fluid] simulation material must be finite.");
    }
    backend.setMaterial(material);
    simulation._options = { ...simulation._options, material };
}

export function refreshFluidSimulationPolygonSurface(simulation: FluidSimulation): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot refresh a disposed simulation.");
    }
    const engine = engineOf(simulation);
    const backend = backendOf(simulation);
    if (backend.refreshPolygonSurface) {
        backend.refreshPolygonSurface(currentFrameEncoder(engine, "polygon-surface refresh"));
    }
}

export function writeFluidSimulationPositions(simulation: FluidSimulation, positions: Float32Array, particleOffset = 0): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot write positions on a disposed simulation.");
    }
    writeFluidSimPositions(engineOf(simulation), backendOf(simulation), positions, particleOffset);
}

export function readFluidSimulationPositions(
    simulation: FluidSimulation,
    options: { readonly particleOffset?: number; readonly particleCount?: number } = {}
): Promise<Float32Array> {
    if (simulation.disposed) {
        return Promise.reject(new Error("[fluid] cannot read positions from a disposed simulation."));
    }
    return readFluidSimPositions(engineOf(simulation), backendOf(simulation), options.particleOffset ?? 0, options.particleCount ?? simulation.count);
}

export function createFluidSimulationParticleStream(simulation: FluidSimulation): FluidParticleStream {
    const stream = {
        get count(): number {
            return simulation.disposed ? 0 : particleStreamRuntimeOf(stream).count;
        },
        get capacity(): number {
            return simulation.disposed ? 0 : particleStreamRuntimeOf(stream).capacity;
        },
        _resolve: (): FluidParticleStreamRuntime => fluidSimParticleStream(backendOf(simulation)),
    } satisfies FluidParticleStream;
    return stream;
}

export function createFluidParticleChannel(engine: EngineContext, options: FluidParticleChannelOptions): FluidParticleChannel {
    if (!Number.isFinite(options.capacity) || options.capacity <= 0 || (options.components !== 1 && options.components !== 4)) {
        throw new RangeError("[fluid] particle channels require positive capacity and one or four components.");
    }

    if (options.initialData && options.initialData.length % options.components !== 0) {
        throw new RangeError("[fluid] initial particle-channel data must contain complete particles.");
    }
    const runtime = createFluidParticleChannelRuntime(engine, options);
    const channel = {
        capacity: runtime.capacity,
        components: runtime.components,
        get disposed(): boolean {
            return channel._disposed;
        },
        _disposed: false,
        _engine: engine,
        _runtime: runtime,
        _owners: new Set<object>(),
    } satisfies FluidParticleChannel;
    return channel;
}

/** @internal Adopt a host-created channel buffer into the shared collection runtime. */
export function adoptFluidParticleChannel(
    engine: EngineContext,
    buffer: GPUBuffer,
    options: { readonly capacity: number; readonly components: 1 | 4; readonly ownsBuffer?: boolean; readonly label?: string }
): FluidParticleChannel {
    const runtime = adoptFluidParticleChannelRuntime(engine, buffer, options);
    const channel = {
        capacity: runtime.capacity,
        components: runtime.components,
        get disposed(): boolean {
            return channel._disposed;
        },
        _disposed: false,
        _engine: engine,
        _runtime: runtime,
        _owners: new Set<object>(),
    } satisfies FluidParticleChannel;
    return channel;
}

export function writeFluidParticleChannel(channel: FluidParticleChannel, data: Float32Array, particleOffset = 0): void {
    if (channel.disposed) {
        throw new Error("[fluid] cannot write a disposed particle channel.");
    }
    particleChannelRuntimeOf(channel).write(data, particleOffset);
}

export function readFluidParticleChannel(
    channel: FluidParticleChannel,
    options: { readonly particleOffset?: number; readonly particleCount?: number } = {}
): Promise<Float32Array> {
    if (channel.disposed) {
        return Promise.reject(new Error("[fluid] cannot read a disposed particle channel."));
    }
    return particleChannelRuntimeOf(channel).read(options.particleOffset ?? 0, options.particleCount ?? channel.capacity);
}

export function fillFluidParticleChannel(
    channel: FluidParticleChannel,
    value: number | readonly [number, number, number, number],
    start = 0,
    count = channel.capacity - start
): void {
    if (channel.disposed) {
        throw new Error("[fluid] cannot fill a disposed particle channel.");
    }
    particleChannelRuntimeOf(channel).fill(value, start, count);
}

export function disposeFluidParticleChannel(channel: FluidParticleChannel): void {
    if (channel.disposed) {
        return;
    }
    if (channel._owners.size > 0) {
        throw new Error("[fluid] cannot dispose a particle channel while it belongs to a simulation collection.");
    }
    channel._disposed = true;
    retireGpuResources(channel._engine as EngineContext, () => particleChannelRuntimeOf(channel).dispose());
}

export function createFluidParticleSpatialQuery(engine: EngineContext, options: FluidParticleSpatialQueryOptions = {}): FluidParticleSpatialQuery {
    const maximumQueries = options.maximumQueries ?? 1024;
    if (!Number.isFinite(maximumQueries) || maximumQueries <= 0) {
        throw new RangeError("[fluid] spatial-query capacity must be a positive number.");
    }
    const runtime = createFluidSpatialQueryRuntime(engine, maximumQueries);
    const query = {
        get results(): readonly FluidParticleSpatialQueryResult[] {
            return spatialQueryRuntimeOf(query).results;
        },
        get status(): FluidParticleSpatialQuery["status"] {
            return spatialQueryRuntimeOf(query).status;
        },
        get disposed(): boolean {
            return query._disposed;
        },
        _disposed: false,
        _engine: engine,
        _runtime: runtime,
    } satisfies FluidParticleSpatialQuery;
    return query;
}

export function sampleFluidParticleSpatialQuery(query: FluidParticleSpatialQuery, stream: FluidParticleStream, requests: readonly FluidParticleSpatialQueryRequest[]): void {
    if (query.disposed) {
        throw new Error("[fluid] cannot sample a disposed particle query.");
    }
    for (const request of requests) {
        if (
            !request.bounds.min.every(Number.isFinite) ||
            !request.bounds.max.every(Number.isFinite) ||
            request.bounds.min.some((value, index) => value > request.bounds.max[index]!) ||
            (request.offset !== undefined && (!Number.isFinite(request.offset) || request.offset < 0)) ||
            (request.count !== undefined && (!Number.isFinite(request.count) || request.count < 0)) ||
            (request.sphere !== undefined && (!request.sphere.origin.every(Number.isFinite) || !Number.isFinite(request.sphere.radius) || request.sphere.radius < 0))
        ) {
            throw new RangeError("[fluid] spatial queries require finite ordered bounds, ranges, and spheres.");
        }
    }
    spatialQueryRuntimeOf(query).record(
        particleStreamRuntimeOf(stream),
        requests.map((request) => ({
            key: request.key,
            offset: request.offset ?? 0,
            count: request.count ?? stream.count,
            min: request.bounds.min,
            max: request.bounds.max,
            ...(request.sphere ? { sphere: request.sphere } : {}),
        }))
    );
}

export function readFluidParticleSpatialQuery(query: FluidParticleSpatialQuery): readonly FluidParticleSpatialQueryResult[] {
    if (query.disposed) {
        throw new Error("[fluid] cannot read a disposed particle query.");
    }
    const runtime = spatialQueryRuntimeOf(query);
    runtime.poll();
    return runtime.results.map((result) => ({ ...result }));
}

export function disposeFluidParticleSpatialQuery(query: FluidParticleSpatialQuery): void {
    if (query.disposed) {
        return;
    }
    query._disposed = true;
    retireGpuResources(query._engine as EngineContext, () => spatialQueryRuntimeOf(query).dispose());
}

export function createFluidWheelTorqueQuery(engine: EngineContext, options: FluidWheelTorqueQueryOptions): FluidWheelTorqueQuery {
    if (!options.center.every(Number.isFinite)) {
        throw new TypeError("[fluid] wheel-torque center must contain finite values.");
    }
    const axialHalfExtent = finitePositive(options.axialHalfExtent, "wheel-torque axialHalfExtent");
    const radialMax = finitePositive(options.radialMax, "wheel-torque radialMax");
    const radialMin = Math.max(0, options.radialMin);
    if (!Number.isFinite(radialMin) || radialMin >= radialMax) {
        throw new RangeError("[fluid] wheel-torque radialMin must be finite and less than radialMax.");
    }
    if (!Number.isFinite(options.speedThreshold) || options.speedThreshold < 0) {
        throw new RangeError("[fluid] wheel-torque speedThreshold must be finite and non-negative.");
    }
    if (options.driveSide !== -1 && options.driveSide !== 1) {
        throw new RangeError("[fluid] wheel-torque driveSide must be -1 or 1.");
    }
    const maximumParticles = Math.max(1, Math.floor(finitePositive(options.maximumParticles, "wheel-torque maximumParticles")));
    const runtime = createFluidWheelTorqueQueryRuntime(engine, {
        center: [...options.center],
        axis: normalizedDirection(options.axis, "wheel-torque axis"),
        verticalAxis: normalizedDirection(options.verticalAxis, "wheel-torque verticalAxis"),
        driveAxis: normalizedDirection(options.driveAxis, "wheel-torque driveAxis"),
        driveSide: options.driveSide,
        axialHalfExtent,
        radialMin,
        radialMax,
        speedThreshold: options.speedThreshold,
        maximumParticles,
    });
    const query = {
        get torque(): number {
            return wheelTorqueQueryRuntimeOf(query).torque;
        },
        get status(): FluidWheelTorqueQuery["status"] {
            return wheelTorqueQueryRuntimeOf(query).status;
        },
        get disposed(): boolean {
            return query._disposed;
        },
        _disposed: false,
        _engine: engine,
        _runtime: runtime,
    } satisfies FluidWheelTorqueQuery;
    return query;
}

export function sampleFluidWheelTorqueQuery(query: FluidWheelTorqueQuery, simulation: FluidSimulation): void {
    if (query.disposed || simulation.disposed) {
        throw new Error("[fluid] cannot sample wheel torque from disposed state.");
    }
    if (query._engine !== simulation._engine) {
        throw new Error("[fluid] wheel-torque query and simulation belong to different engines.");
    }
    wheelTorqueQueryRuntimeOf(query).record(backendOf(simulation));
}

export function readFluidWheelTorqueQuery(query: FluidWheelTorqueQuery): number {
    if (query.disposed) {
        throw new Error("[fluid] cannot read a disposed wheel-torque query.");
    }
    const runtime = wheelTorqueQueryRuntimeOf(query);
    runtime.poll();
    return runtime.torque;
}

export function disposeFluidWheelTorqueQuery(query: FluidWheelTorqueQuery): void {
    if (query.disposed) {
        return;
    }
    query._disposed = true;
    retireGpuResources(query._engine as EngineContext, () => wheelTorqueQueryRuntimeOf(query).dispose());
}

export function getFluidSimulationDiagnostics(simulation: FluidSimulation): FluidSimulationDiagnostics {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot read diagnostics from a disposed simulation.");
    }
    const backend = backendOf(simulation);
    const diffuse = backend.diffuse;
    const counts = diffuse?.counts;
    const polygon = backend.polygonSurface;
    return {
        method: simulation.method,
        count: backend.count,
        activeCount: backend.activeCount ?? backend.count,
        renderCount: backend.renderCount ?? backend.activeCount ?? backend.count,
        particleRadius: backend.particleRadius,
        surfaceSizeScale: backend.surfaceSizeScale ?? 1,
        surfaceThicknessScale: backend.surfaceThicknessScale ?? 1,
        surfaceRejectSparseMarkers: backend.surfaceRejectSparseMarkers ?? false,
        gpuBytes: backend.gpuBytes,
        initialEmitterParticleCounts: backend.initialEmitterParticleCounts ?? getEmptyInitialEmitterCounts(),
        pressure: backend.pressureDiagnostics ? { ...backend.pressureDiagnostics } : null,
        timestep: backend.timestepDiagnostics ? { ...backend.timestepDiagnostics } : null,
        diffuse: diffuse
            ? {
                  enabled: true,
                  total: counts?.total ?? 0,
                  spray: counts?.spray ?? 0,
                  foam: counts?.foam ?? 0,
                  bubble: counts?.bubble ?? 0,
                  capacity: diffuse.capacity,
              }
            : null,
        polygon: polygon
            ? {
                  triangleCount: polygon.triangleCount,
                  triangleCapacity: polygon.triangleCapacity,
                  reconstructionMultiplier: polygon.reconstructionMultiplier,
                  gridOrigin: [...polygon.gridOrigin],
                  gridDimensions: [...polygon.gridDimensions],
                  gridSpacing: polygon.gridSpacing,
              }
            : null,
    };
}

function normalizedCollectionSources(sources: readonly FluidSimulationRenderSource[]): FluidSimulationRenderSource[] {
    return sources.map((source) => {
        if (source.simulation.disposed || source.alpha?.disposed || source.color?.disposed) {
            throw new Error("[fluid] simulation collections cannot contain disposed state.");
        }
        if (source.alpha && source.alpha.components !== 1) {
            throw new TypeError("[fluid] render alpha channels must contain one component per particle.");
        }
        if (source.color && source.color.components !== 4) {
            throw new TypeError("[fluid] render color channels must contain four components per particle.");
        }
        const count = source.count === undefined ? undefined : Math.min(source.simulation.count, Math.max(0, Math.floor(source.count)));
        if ((source.alpha && source.alpha.capacity < (count ?? source.simulation.count)) || (source.color && source.color.capacity < (count ?? source.simulation.count))) {
            throw new RangeError("[fluid] particle-channel capacity is smaller than its render source.");
        }
        return {
            simulation: source.simulation,
            ...(count !== undefined ? { count } : {}),
            opacity: Number.isFinite(source.opacity) ? Math.min(1, Math.max(0, source.opacity!)) : 1,
            ...(source.alpha ? { alpha: source.alpha } : {}),
            ...(source.color ? { color: source.color } : {}),
        };
    });
}

function assertActiveCollection(collection: FluidSimulationCollection): void {
    if (collection.disposed) {
        throw new Error("[fluid] cannot operate on a disposed simulation collection.");
    }
}

function updateCollectionSources(collection: FluidSimulationCollection, sources: FluidSimulationRenderSource[]): void {
    for (const source of collection._sources) {
        source.alpha?._owners.delete(collection);
        source.color?._owners.delete(collection);
    }
    collection._sources.splice(0, collection._sources.length, ...sources);
    const simulations = [...new Set(sources.map((source) => source.simulation))];
    collection._simulations.splice(0, collection._simulations.length, ...simulations);
    for (const source of sources) {
        source.alpha?._owners.add(collection);
        source.color?._owners.add(collection);
    }
}

export function createFluidSimulationCollection(engine: EngineContext, simulations: readonly FluidSimulation[] = []): FluidSimulationCollection {
    const simulationState: FluidSimulation[] = [];
    const sourceState: FluidSimulationRenderSource[] = [];
    const renderLayers: FluidSimulationRenderLayer[] = [];
    const collection = {
        get simulations(): readonly FluidSimulation[] {
            return collection._simulations;
        },
        get sources(): readonly FluidSimulationRenderSource[] {
            return collection._sources;
        },
        renderLayers,
        get disposed(): boolean {
            return collection._disposed;
        },
        _disposed: false,
        _engine: engine,
        _simulations: simulationState,
        _sources: sourceState,
        _renderLayers: renderLayers,
    } satisfies FluidSimulationCollection;
    setFluidSimulationCollectionSimulations(collection, simulations);
    return collection;
}

export function setFluidSimulationCollectionSimulations(collection: FluidSimulationCollection, simulations: readonly FluidSimulation[]): void {
    setFluidSimulationCollectionSources(
        collection,
        simulations.map((simulation) => ({ simulation }))
    );
}

export function setFluidSimulationCollectionSources(collection: FluidSimulationCollection, sources: readonly FluidSimulationRenderSource[]): void {
    assertActiveCollection(collection);
    for (const source of sources) {
        if (
            source.simulation._engine !== collection._engine ||
            (source.alpha?._engine !== undefined && source.alpha._engine !== collection._engine) ||
            (source.color?._engine !== undefined && source.color._engine !== collection._engine)
        ) {
            throw new Error("[fluid] simulation collections and particle channels must share one engine.");
        }
    }
    updateCollectionSources(collection, normalizedCollectionSources(sources));
}

export function createFluidSimulationCollectionParticleStream(
    collection: FluidSimulationCollection,
    scene: SceneContext,
    capacityInput: number,
    options: { readonly update?: "frame-graph" | "manual" } = {}
): FluidSimulationCollectionParticleStream {
    assertActiveCollection(collection);
    const capacity = finitePositive(capacityInput, "collection particle-stream capacity");
    const engine = collection._engine as EngineContext;
    const aggregate = createFluidAggregateRuntime(engine, capacity, () =>
        collection.sources
            .filter((source) => !source.simulation.disposed)
            .map((source) => ({
                sim: backendOf(source.simulation),
                count: source.count,
                opacity: source.opacity ?? 1,
                ...(source.alpha ? { alpha: particleChannelRuntimeOf(source.alpha) } : {}),
                ...(source.color ? { color: particleChannelRuntimeOf(source.color) } : {}),
            }))
    );
    const stream = {
        get count(): number {
            return aggregate.stream.count;
        },
        get capacity(): number {
            return aggregate.stream.capacity;
        },
        _resolve: (): FluidParticleStreamRuntime => aggregate.stream,
    } satisfies FluidParticleStream;
    const handle = {
        stream,
        get overflowed(): boolean {
            return aggregate.overflowed;
        },
        get disposed(): boolean {
            return handle._disposed;
        },
        _disposed: false,
        _engine: engine,
        _aggregate: aggregate,
        _task: aggregate.task,
        _scene: scene,
        _automatic: options.update !== "manual",
    } satisfies FluidSimulationCollectionParticleStream;
    if (handle._automatic) {
        addTask(scene, aggregate.task);
    }
    return handle;
}

/** Record collection aggregation immediately in the current frame encoder. */
export function refreshFluidSimulationCollectionParticleStream(stream: FluidSimulationCollectionParticleStream): void {
    if (stream.disposed) {
        throw new Error("[fluid] cannot refresh a disposed collection particle stream.");
    }
    (stream._task as Task).execute!();
}

export function disposeFluidSimulationCollectionParticleStream(stream: FluidSimulationCollectionParticleStream): void {
    if (stream.disposed) {
        return;
    }
    stream._disposed = true;
    const task = stream._task as Task;
    task.executionEnabled = false;
    if (stream._automatic) {
        removeFluidTasksFromScene(stream._scene as SceneContext, [task]);
    }
    retireGpuResources(stream._engine as EngineContext, () => {
        task.dispose();
        (stream._aggregate as FluidAggregateRuntime).dispose();
    });
}

export function stepFluidSimulationCollection(collection: FluidSimulationCollection, deltaSeconds: number): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        stepFluidSimulation(simulation, deltaSeconds);
    }
}

/** @internal Update a shared floating-body reducer without exposing its simulation backend. */
export function updateFluidFloatingBodySystem(system: FloatingBodySystem, deltaSeconds: number, simulation: FluidSimulation): void {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot update floating bodies from a disposed simulation.");
    }
    system.update(deltaSeconds, backendOf(simulation));
}

export function resetFluidSimulationCollection(collection: FluidSimulationCollection): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        resetFluidSimulation(simulation);
    }
}

export function setFluidSimulationCollectionFlow(collection: FluidSimulationCollection, flow: FluidFlowConfig | null, reset = false): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        setFluidSimulationFlow(simulation, flow, reset);
    }
}

export function setFluidSimulationCollectionFoam(collection: FluidSimulationCollection, foam: FoamConfig | null): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        setFluidSimulationFoam(simulation, foam);
    }
}

export function setFluidSimulationCollectionParameter(collection: FluidSimulationCollection, key: string, value: number): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        setFluidSimulationParameter(simulation, key, value);
    }
}

export function setFluidSimulationCollectionMaterial(collection: FluidSimulationCollection, material: number): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        setFluidSimulationMaterial(simulation, material);
    }
}

export function setFluidSimulationCollectionSceneSdf(collection: FluidSimulationCollection, sceneSdf: FluidSceneSdf | null): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        setFluidSimulationSceneSdf(simulation, sceneSdf);
    }
}

export function setFluidSimulationCollectionForceField(collection: FluidSimulationCollection, forceField: FluidForceField | null): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        setFluidSimulationForceField(simulation, forceField);
    }
}

export function setFluidSimulationCollectionProfiler(collection: FluidSimulationCollection, profiler: FluidSimulationProfiler | null): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        setFluidSimulationProfiler(simulation, profiler);
    }
}

export function refreshFluidSimulationCollectionPolygonSurfaces(collection: FluidSimulationCollection): void {
    assertActiveCollection(collection);
    for (const simulation of collection.simulations) {
        refreshFluidSimulationPolygonSurface(simulation);
    }
}

export function getFluidSimulationCollectionDiagnostics(collection: FluidSimulationCollection): FluidSimulationCollectionDiagnostics {
    if (collection.disposed) {
        throw new Error("[fluid] cannot read diagnostics from a disposed simulation collection.");
    }
    const diagnostics = collection.simulations.filter((simulation) => !simulation.disposed).map(getFluidSimulationDiagnostics);
    const pressure =
        diagnostics
            .map((entry) => entry.pressure)
            .filter((entry): entry is FluidPressureDiagnostics => entry !== null)
            .sort((left, right) => right.relativeResidual - left.relativeResidual)[0] ?? null;
    const diffuseEntries = diagnostics.map((entry) => entry.diffuse).filter((entry): entry is FluidSimulationDiffuseDiagnostics => entry !== null);
    const diffuse = diffuseEntries.length
        ? diffuseEntries.reduce<FluidSimulationDiffuseDiagnostics>(
              (sum, entry) => ({
                  enabled: true,
                  total: sum.total + entry.total,
                  spray: sum.spray + entry.spray,
                  foam: sum.foam + entry.foam,
                  bubble: sum.bubble + entry.bubble,
                  capacity: sum.capacity + entry.capacity,
              }),
              { enabled: true, total: 0, spray: 0, foam: 0, bubble: 0, capacity: 0 }
          )
        : null;
    return {
        simulationCount: diagnostics.length,
        count: diagnostics.reduce((sum, entry) => sum + entry.count, 0),
        activeCount: diagnostics.reduce((sum, entry) => sum + entry.activeCount, 0),
        renderCount: diagnostics.reduce((sum, entry) => sum + entry.renderCount, 0),
        gpuBytes: diagnostics.reduce((sum, entry) => sum + entry.gpuBytes, 0),
        pressure,
        diffuse,
        polygons: diagnostics.flatMap((entry) => (entry.polygon ? [entry.polygon] : [])),
    };
}

export function createFluidSceneSdf(engine: EngineContext, options: FluidSceneSdfOptions): FluidSceneSdf {
    if (!options.struct.trim() || !options.sdf.trim() || options.params.byteLength === 0) {
        throw new TypeError("[fluid] scene SDF requires WGSL source and non-empty parameter data.");
    }

    const binding = createSceneSdfRuntimeBinding(engine, options);
    const sceneSdf = {
        get disposed(): boolean {
            return sceneSdf._disposed;
        },
        _disposed: false,
        _engine: engine,
        _binding: binding,
        _owners: new Set<FluidSimulation>(),
    } satisfies FluidSceneSdf;
    return sceneSdf;
}

/** Create one packed static-plus-animated collision binding shared by every fluid solver. */
export function createFluidCompositeSceneSdf(engine: EngineContext, options: FluidCompositeSceneSdfOptions): FluidSceneSdf {
    const binding = createCompositeSceneSdfRuntimeBinding(engine, options);
    const sceneSdf = {
        get disposed(): boolean {
            return sceneSdf._disposed;
        },
        _disposed: false,
        _engine: engine,
        _binding: binding,
        _owners: new Set<FluidSimulation>(),
    } satisfies FluidSceneSdf;
    return sceneSdf;
}

/** @internal Adopt a host-owned low-level scene SDF while migrating legacy scene orchestration. */
export function adoptFluidSceneSdf(engine: EngineContext, spec: SceneSdfSpec): FluidSceneSdf {
    const binding: SceneSdfRuntimeBinding = {
        spec,
        updateParams(data): void {
            if (data.byteLength > spec.buffer.size) {
                throw new RangeError("[fluid] scene SDF parameters exceed the adopted binding size.");
            }
            engine._device.queue.writeBuffer(spec.buffer, 0, data);
        },
        updateSdfGrid(data): void {
            if (!spec.sdfGrid) {
                throw new Error("[fluid] this adopted scene SDF has no grid binding.");
            }
            if (data.byteLength > spec.sdfGrid.size) {
                throw new RangeError("[fluid] scene SDF grid data exceeds the adopted binding size.");
            }
            engine._device.queue.writeBuffer(spec.sdfGrid, 0, data);
        },
        dispose(): void {
            // The host retains ownership of adopted buffers.
        },
    };
    const sceneSdf = {
        get disposed(): boolean {
            return sceneSdf._disposed;
        },
        _disposed: false,
        _engine: engine,
        _binding: binding,
        _owners: new Set<FluidSimulation>(),
    } satisfies FluidSceneSdf;
    return sceneSdf;
}

export function createFluidRenderEnvironment(source: FluidRenderEnvironmentSource): FluidRenderEnvironment {
    const candidate = source as { readonly specularCubeView?: unknown; readonly cubeSampler?: unknown };
    if (!candidate.specularCubeView || !candidate.cubeSampler) {
        throw new TypeError("[fluid] a render environment requires specular cubemap and sampler state.");
    }
    return { _source: source };
}

export function updateFluidSceneSdf(sceneSdf: FluidSceneSdf, params: Float32Array, sdfGrid?: Float32Array | Uint32Array): void {
    if (sceneSdf.disposed) {
        throw new Error("[fluid] cannot update a disposed scene SDF.");
    }
    sceneSdfBindingOf(sceneSdf).updateParams(params);
    if (sdfGrid) {
        sceneSdfBindingOf(sceneSdf).updateSdfGrid(sdfGrid);
    }
}

/** Update every local collision transform after scene animation has advanced. */
export function updateFluidSceneSdfTransforms(
    sceneSdf: FluidSceneSdf,
    updates: readonly FluidSceneSdfTransformUpdate[],
    elapsedSeconds: number,
    options: { readonly resetMotion?: boolean } = {}
): void {
    if (sceneSdf.disposed) {
        throw new Error("[fluid] cannot update a disposed scene SDF.");
    }
    const updateTransforms = sceneSdfBindingOf(sceneSdf).updateTransforms;
    if (!updateTransforms) {
        throw new Error("[fluid] this scene SDF does not support transform updates.");
    }
    updateTransforms(updates as readonly CompositeSceneSdfTransformUpdate[], elapsedSeconds, options.resetMotion === true);
}

/** Translate the static collision grid without reallocating or rebuilding solver pipelines. */
export function updateFluidSceneSdfStaticOffset(sceneSdf: FluidSceneSdf, offset: readonly [number, number, number], options: { readonly resetMotion?: boolean } = {}): void {
    if (sceneSdf.disposed) {
        throw new Error("[fluid] cannot update a disposed scene SDF.");
    }
    const updateStaticOffset = sceneSdfBindingOf(sceneSdf).updateStaticOffset;
    if (!updateStaticOffset) {
        throw new Error("[fluid] this scene SDF does not support static-grid offsets.");
    }
    updateStaticOffset(offset, options.resetMotion === true);
}

/** Uniformly scale the static collision grid around a world-space pivot without rebuilding its atlas. */
export function updateFluidSceneSdfStaticScale(
    sceneSdf: FluidSceneSdf,
    scale: number,
    pivot: readonly [number, number, number],
    options: { readonly resetMotion?: boolean } = {}
): void {
    if (sceneSdf.disposed) {
        throw new Error("[fluid] cannot update a disposed scene SDF.");
    }
    const updateStaticScale = sceneSdfBindingOf(sceneSdf).updateStaticScale;
    if (!updateStaticScale) {
        throw new Error("[fluid] this scene SDF does not support static-grid scaling.");
    }
    updateStaticScale(scale, pivot, options.resetMotion === true);
}

/** Update or disable the optional closed container used by a composite collision binding. */
export function updateFluidSceneSdfContainer(sceneSdf: FluidSceneSdf, bounds: FluidSceneSdfBounds | null): void {
    if (sceneSdf.disposed) {
        throw new Error("[fluid] cannot update a disposed scene SDF.");
    }
    const updateContainer = sceneSdfBindingOf(sceneSdf).updateContainer;
    if (!updateContainer) {
        throw new Error("[fluid] this scene SDF does not support container updates.");
    }
    updateContainer(bounds);
}

/** Update static or local SDF enablement and filtering without rebuilding pipelines. */
export function updateFluidSceneSdfGridSettings(sceneSdf: FluidSceneSdf, updates: readonly FluidSceneSdfGridSettingsUpdate[]): void {
    if (sceneSdf.disposed) {
        throw new Error("[fluid] cannot update a disposed scene SDF.");
    }
    const updateGridSettings = sceneSdfBindingOf(sceneSdf).updateGridSettings;
    if (!updateGridSettings) {
        throw new Error("[fluid] this scene SDF does not support per-grid settings.");
    }
    updateGridSettings(updates as readonly CompositeSceneSdfGridSettingsUpdate[]);
}

export function setFluidSimulationSceneSdf(simulation: FluidSimulation, sceneSdf: FluidSceneSdf | null): void {
    if (simulation.disposed || sceneSdf?.disposed) {
        throw new Error("[fluid] cannot attach scene SDF state to a disposed resource.");
    }
    validateBindingEngine(engineOf(simulation), sceneSdf, "scene SDF");
    backendOf(simulation).setSceneSdf(sceneSdfSpecOf(sceneSdf));
    simulation._options.sceneSdf?._owners.delete(simulation);
    simulation._options = { ...simulation._options, sceneSdf };
    sceneSdf?._owners.add(simulation);
}

export function disposeFluidSceneSdf(sceneSdf: FluidSceneSdf): void {
    if (sceneSdf.disposed) {
        return;
    }
    if (sceneSdf._owners.size > 0) {
        throw new Error("[fluid] cannot dispose a scene SDF while it is attached to a simulation.");
    }
    sceneSdf._disposed = true;
    retireGpuResources(sceneSdf._engine as EngineContext, () => sceneSdfBindingOf(sceneSdf).dispose());
}

export function createFluidForceField(engine: EngineContext, options: FluidForceFieldOptions): FluidForceField {
    if (!options.struct.trim() || !options.wgsl.trim() || options.params.byteLength === 0) {
        throw new TypeError("[fluid] force field requires WGSL source and non-empty parameter data.");
    }

    const binding = createForceFieldRuntimeBinding(engine, options);
    const forceField = {
        get disposed(): boolean {
            return forceField._disposed;
        },
        _disposed: false,
        _engine: engine,
        _binding: binding,
        _owners: new Set<FluidSimulation>(),
    } satisfies FluidForceField;
    return forceField;
}

/** Create the shared radial-plus-directional impulse used by fluid consumers. */
export function createFluidImpulseForce(engine: EngineContext, options: FluidImpulseForceOptions): FluidForceField {
    const forceField = createFluidForceField(engine, {
        struct: "struct ForceFieldParams { center: vec4<f32>, push: vec4<f32>, };",
        wgsl: FLUID_IMPULSE_FORCE_WGSL,
        params: fluidImpulseParams(options),
    });
    forceField._kind = "impulse";
    return forceField;
}

/** Update a shared impulse without exposing its uniform layout or GPU buffer. */
export function updateFluidImpulseForce(forceField: FluidForceField, options: FluidImpulseForceOptions): void {
    if (forceField._kind !== "impulse") {
        throw new TypeError("[fluid] updateFluidImpulseForce requires a shared impulse force.");
    }
    updateFluidForceField(forceField, fluidImpulseParams(options));
}

/** @internal Adopt a host-owned low-level force field while migrating legacy scene orchestration. */
export function adoptFluidForceField(engine: EngineContext, spec: ForceFieldSpec): FluidForceField {
    const binding: ForceFieldRuntimeBinding = {
        spec,
        updateParams(data): void {
            if (data.byteLength > spec.buffer.size) {
                throw new RangeError("[fluid] force-field parameters exceed the adopted binding size.");
            }
            engine._device.queue.writeBuffer(spec.buffer, 0, data);
        },
        dispose(): void {
            // The host retains ownership of adopted buffers.
        },
    };
    const forceField = {
        get disposed(): boolean {
            return forceField._disposed;
        },
        _disposed: false,
        _engine: engine,
        _binding: binding,
        _owners: new Set<FluidSimulation>(),
    } satisfies FluidForceField;
    return forceField;
}

export function updateFluidForceField(forceField: FluidForceField, params: Float32Array): void {
    if (forceField.disposed) {
        throw new Error("[fluid] cannot update a disposed force field.");
    }
    forceFieldBindingOf(forceField).updateParams(params);
}

export function setFluidSimulationForceField(simulation: FluidSimulation, forceField: FluidForceField | null): void {
    if (simulation.disposed || forceField?.disposed) {
        throw new Error("[fluid] cannot attach force-field state to a disposed resource.");
    }
    validateBindingEngine(engineOf(simulation), forceField, "force field");
    backendOf(simulation).setForceField(forceFieldSpecOf(forceField));
    simulation._options.forceField?._owners.delete(simulation);
    simulation._options = { ...simulation._options, forceField };
    forceField?._owners.add(simulation);
}

export function disposeFluidForceField(forceField: FluidForceField): void {
    if (forceField.disposed) {
        return;
    }
    if (forceField._owners.size > 0) {
        throw new Error("[fluid] cannot dispose a force field while it is attached to a simulation.");
    }
    forceField._disposed = true;
    retireGpuResources(forceField._engine as EngineContext, () => forceFieldBindingOf(forceField).dispose());
}

export function createFluidSimulationProfiler(engine: EngineContext, options: FluidSimulationProfilerOptions = {}): FluidSimulationProfiler {
    const impl = createGpuFluidProfiler(engine._device, options.queryCapacity);
    const profiler = {
        get disposed(): boolean {
            return profiler._disposed;
        },
        _disposed: false,
        _engine: engine,
        _profiler: impl,
        _owners: new Set<object>(),
    } satisfies FluidSimulationProfiler;
    return profiler;
}

export function beginFluidSimulationProfilerFrame(profiler: FluidSimulationProfiler, options: FluidSimulationProfilerFrameOptions = {}): void {
    if (profiler.disposed) {
        throw new Error("[fluid] cannot begin a disposed profiler.");
    }
    const impl = profilerImplOf(profiler);
    const encoder = options.captureEnvelope === false ? null : currentFrameEncoder(profiler._engine as EngineContext, "profiler frame begin");
    impl.beginFrame();
    if (encoder) {
        impl.frameStart(encoder);
    }
}

export function endFluidSimulationProfilerFrame(profiler: FluidSimulationProfiler, options: FluidSimulationProfilerFrameOptions = {}): void {
    if (profiler.disposed) {
        throw new Error("[fluid] cannot end a disposed profiler.");
    }
    const impl = profilerImplOf(profiler);
    const encoder = currentFrameEncoder(profiler._engine as EngineContext, "profiler frame end");
    impl.frameStop(encoder);
    if (!options.deferResolve) {
        impl.resolveInto(encoder);
    }
}

/** Resolve a profiling window after its independently submitted GPU work has been recorded. */
export async function submitFluidSimulationProfiler(profiler: FluidSimulationProfiler): Promise<void> {
    if (profiler.disposed) {
        throw new Error("[fluid] cannot submit a disposed profiler.");
    }
    const engine = profiler._engine as EngineContext;
    if (engine._currentEncoder) {
        throw new Error("[fluid] submit profiler results outside render-frame recording.");
    }
    const impl = profilerImplOf(profiler);
    const encoder = engine._device.createCommandEncoder({ label: "fluid-profiler-submit" });
    impl.resolveInto(encoder);
    engine._device.queue.submit([encoder.finish()]);
    await impl.collectSubmitted();
}

export function readFluidSimulationProfiler(profiler: FluidSimulationProfiler): FluidSimulationProfilerResults | null {
    if (profiler.disposed) {
        throw new Error("[fluid] cannot read a disposed profiler.");
    }
    const result = profilerImplOf(profiler).results();
    return result ? { stages: { ...result.stages }, total: result.total, frameTotal: result.frameTotal, ...(result.overflowed ? { overflowed: true } : {}) } : null;
}

export function setFluidSimulationProfiler(simulation: FluidSimulation, profiler: FluidSimulationProfiler | null): void {
    if (simulation.disposed || profiler?.disposed) {
        throw new Error("[fluid] cannot attach profiler state to a disposed resource.");
    }
    validateBindingEngine(engineOf(simulation), profiler, "profiler");
    const previous = simulation.profiler;
    backendOf(simulation).setProfiler?.(profilerHookOf(profiler));
    try {
        for (const layer of simulation._renderLayers) {
            setLayerProfiler(layer, profiler);
        }
    } catch (error) {
        backendOf(simulation).setProfiler?.(profilerHookOf(previous));
        for (const layer of simulation._renderLayers) {
            setLayerProfiler(layer, previous);
        }
        throw error;
    }
    previous?._owners.delete(simulation);
    for (const layer of simulation._renderLayers) {
        layer._profiler?._owners.delete(layer);
        layer._profiler = profiler;
        profiler?._owners.add(layer);
    }
    simulation._options = { ...simulation._options, profiler };
    profiler?._owners.add(simulation);
}

export function disposeFluidSimulationProfiler(profiler: FluidSimulationProfiler): void {
    if (profiler.disposed) {
        return;
    }
    if (profiler._owners.size > 0) {
        throw new Error("[fluid] cannot dispose a profiler while it is attached.");
    }
    profiler._disposed = true;
    retireGpuResources(profiler._engine as EngineContext, () => profilerImplOf(profiler).dispose());
}

export function readFluidSimulationPressureDiagnostics(simulation: FluidSimulation): FluidPressureDiagnostics | null {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot read diagnostics from a disposed simulation.");
    }
    const diagnostics = simulation.pressureDiagnostics;
    return diagnostics ? { ...diagnostics } : null;
}

/** Attach a sphere, screen-space surface, or FLIP polygon render task. */
export function attachFluidSimulationRenderLayer(simulation: FluidSimulation, options: FluidSimulationRenderLayerOptions): FluidSimulationRenderLayer {
    if (simulation.disposed) {
        throw new Error("[fluid] cannot attach a render layer to a disposed simulation.");
    }
    const profiler = options.profiler === undefined ? simulation.profiler : options.profiler;
    if (profiler?.disposed) {
        throw new Error("[fluid] cannot attach a disposed profiler.");
    }
    validateBindingEngine(engineOf(simulation), profiler, "profiler");
    let particleTask: ParticleRenderTask | undefined;
    let surfaceTask: FluidSurfaceTask | undefined;
    let polygonTask: FluidPolygonSurfaceTask | undefined;
    if (options.mode === "spheres") {
        if (!options.colorTarget) {
            throw new Error("[fluid] a sphere render layer requires colorTarget.");
        }
        particleTask = createParticleRenderTask(engineOf(simulation), options.scene, {
            colorRT: options.colorTarget,
            depthRT: options.depthTarget,
            camera: options.camera,
            sim: backendOf(simulation),
        });
    } else {
        if (!options.backgroundTarget || !options.outputTarget) {
            throw new Error(`[fluid] a ${options.mode} render layer requires backgroundTarget and outputTarget.`);
        }
        if (options.mode === "surface") {
            surfaceTask = createFluidSurfaceTask(engineOf(simulation), options.scene, {
                bgRT: options.backgroundTarget,
                outRT: options.outputTarget,
                depthRT: options.depthTarget,
                camera: options.camera,
                sim: backendOf(simulation),
            });
        } else {
            polygonTask = createFluidPolygonSurfaceTask(engineOf(simulation), options.scene, {
                bgRT: options.backgroundTarget,
                outRT: options.outputTarget,
                depthRT: options.depthTarget,
                camera: options.camera,
                sim: backendOf(simulation),
            });
            polygonTask.setEnabled(true);
        }
    }
    const task = particleTask ?? surfaceTask ?? polygonTask!;
    const profile = { ...(options.profile ?? {}) };
    const layer = {
        mode: options.mode,
        get profile(): FluidRenderProfileSettings {
            return layer._profile;
        },
        get particleStream(): FluidParticleStream | null {
            return layer._particleStream;
        },
        overflowed: false,
        enabled: true,
        opacity: 1,
        disposed: false,
        ...(particleTask ? { _particleTask: particleTask } : {}),
        ...(surfaceTask ? { _surfaceTask: surfaceTask } : {}),
        ...(polygonTask ? { _polygonTask: polygonTask } : {}),
        _task: task,
        _engine: engineOf(simulation),
        _scene: options.scene,
        _profile: profile,
        _colorTarget: options.mode === "spheres" ? options.colorTarget : options.outputTarget,
        _profiler: profiler ?? null,
        _particleStream: createFluidSimulationParticleStream(simulation),
        _useParticleColor: options.useParticleColor,
    } satisfies FluidSimulationRenderLayer;
    try {
        setLayerProfiler(layer, profiler ?? null);
        applyRenderLayerProfile(layer, profile);
        configureFluidSimulationRenderLayer(layer, options);
        addTask(options.scene, task);
    } catch (error) {
        task.dispose();
        throw error;
    }
    profiler?._owners.add(layer);
    simulation._renderLayers.push(layer);
    return layer;
}

/** Attach one shared render task for a dynamic simulation collection. */
export function attachFluidSimulationCollectionRenderLayer(
    collection: FluidSimulationCollection,
    options: FluidSimulationCollectionRenderLayerOptions
): FluidSimulationRenderLayer {
    if (collection.disposed) {
        throw new Error("[fluid] cannot render a disposed simulation collection.");
    }
    const engine = collection._engine as EngineContext;
    const firstSimulation = collection.simulations.find((simulation) => !simulation.disposed);
    const profiler = options.profiler === undefined ? (firstSimulation?.profiler ?? null) : options.profiler;
    if (profiler?.disposed) {
        throw new Error("[fluid] cannot attach a disposed profiler.");
    }
    validateBindingEngine(engine, profiler, "profiler");
    let aggregate: FluidAggregateRuntime | undefined;
    let particleTask: ParticleRenderTask | undefined;
    let surfaceTask: FluidSurfaceTask | undefined;
    let polygonTask: FluidPolygonSurfaceTask | undefined;
    let foamTask: FoamRenderTask | undefined;
    let foamPolygonSurfaceDepth = false;
    if (options.mode === "surface" || options.mode === "spheres") {
        const capacity =
            options.particleCapacity ??
            Math.max(
                1,
                collection.sources.reduce((sum, source) => sum + (source.count ?? source.simulation.count), 0)
            );
        aggregate = createFluidAggregateRuntime(engine, capacity, () =>
            collection.sources
                .filter((source) => !source.simulation.disposed && !(options.excludePolygonSurfaces && getFluidSimulationDiagnostics(source.simulation).polygon))
                .map((source) => ({
                    sim: backendOf(source.simulation),
                    count: source.count,
                    opacity: source.opacity ?? 1,
                    ...(source.alpha ? { alpha: particleChannelRuntimeOf(source.alpha) } : {}),
                    ...(source.color ? { color: particleChannelRuntimeOf(source.color) } : {}),
                }))
        );
        if (options.mode === "spheres") {
            if (!options.colorTarget) {
                aggregate.dispose();
                throw new Error("[fluid] a sphere collection layer requires colorTarget.");
            }
            particleTask = createParticleRenderTask(engine, options.scene, {
                colorRT: options.colorTarget,
                depthRT: options.depthTarget,
                camera: options.camera,
                sim: aggregate.sim,
            });
        } else {
            if (!options.backgroundTarget || !options.outputTarget) {
                aggregate.dispose();
                throw new Error("[fluid] a surface collection layer requires backgroundTarget and outputTarget.");
            }
            surfaceTask = createFluidSurfaceTask(engine, options.scene, {
                bgRT: options.backgroundTarget,
                outRT: options.outputTarget,
                depthRT: options.depthTarget,
                camera: options.camera,
                sim: aggregate.sim,
            });
            surfaceTask.setParticleAlpha(aggregate.stream.alphaBuffer ?? null);
            surfaceTask.setParticleColor(aggregate.colorBuffer);
        }
    } else {
        if (options.mode === "polygon") {
            if (!options.backgroundTarget || !options.outputTarget) {
                throw new Error("[fluid] a polygon collection layer requires backgroundTarget and outputTarget.");
            }
            polygonTask = createFluidPolygonSurfaceTask(engine, options.scene, {
                bgRT: options.backgroundTarget,
                outRT: options.outputTarget,
                depthRT: options.depthTarget,
                camera: options.camera,
                ...(firstSimulation ? { sim: backendOf(firstSimulation) } : {}),
            });
            const execute = polygonTask.execute?.bind(polygonTask);
            if (!execute) {
                polygonTask.dispose();
                throw new Error("[fluid] polygon collection rendering requires a direct task execute path.");
            }
            polygonTask.execute = (): number => {
                polygonTask!.setSims(collection.simulations.filter((simulation) => !simulation.disposed).map(backendOf));
                return execute();
            };
            polygonTask.setEnabled(true);
        } else {
            foamTask = createFoamRenderTask(engine, options.scene, {
                colorRT: options.colorTarget ?? options.outputTarget ?? engine.scRT,
                depthRT: options.depthTarget,
                camera: options.camera,
                ...(firstSimulation ? { sim: backendOf(firstSimulation) } : {}),
                getSurfaceDepth: () => {
                    const surfaceDepth = options.surfaceLayer ? (layerDepthView(options.surfaceLayer) as ReturnType<FluidSurfaceTask["surfaceDepthView"]>) : null;
                    return foamPolygonSurfaceDepth && options.polygonSurfaceLayer
                        ? ((layerDepthView(options.polygonSurfaceLayer) as ReturnType<FluidSurfaceTask["surfaceDepthView"]>) ?? surfaceDepth)
                        : surfaceDepth;
                },
            });
            const execute = foamTask.execute?.bind(foamTask);
            foamTask.execute = (): number => {
                let passes = 0;
                for (const simulation of collection.simulations) {
                    if (simulation.disposed) {
                        continue;
                    }
                    const backend = backendOf(simulation);
                    if ((options.excludePolygonSurfaces && backend.polygonSurface) || (options.polygonSurfacesOnly && !backend.polygonSurface)) {
                        continue;
                    }
                    if (!backend.diffuse) {
                        continue;
                    }
                    foamPolygonSurfaceDepth = layer._foamPolygonSurfaceDepth ?? backend.polygonSurface !== undefined;
                    foamTask!.setPolygonSurfaceDepth(foamPolygonSurfaceDepth);
                    foamTask!.setSim(backend);
                    passes += execute?.() ?? 0;
                }
                return passes;
            };
        }
    }
    const task = particleTask ?? surfaceTask ?? polygonTask ?? foamTask!;
    const profile = { ...(options.profile ?? {}) };
    const particleStream = aggregate
        ? ({
              get count(): number {
                  return aggregate!.stream.count;
              },
              get capacity(): number {
                  return aggregate!.stream.capacity;
              },
              _resolve: (): FluidParticleStreamRuntime => aggregate!.stream,
          } satisfies FluidParticleStream)
        : null;
    const layer = {
        mode: options.mode,
        get profile(): FluidRenderProfileSettings {
            return layer._profile;
        },
        get particleStream(): FluidParticleStream | null {
            return layer._particleStream;
        },
        get overflowed(): boolean {
            return aggregate?.overflowed ?? false;
        },
        enabled: true,
        opacity: 1,
        disposed: false,
        ...(particleTask ? { _particleTask: particleTask } : {}),
        ...(surfaceTask ? { _surfaceTask: surfaceTask } : {}),
        ...(polygonTask ? { _polygonTask: polygonTask } : {}),
        ...(foamTask ? { _foamTask: foamTask } : {}),
        _task: task,
        _engine: engine,
        _scene: options.scene,
        _profile: profile,
        _colorTarget: options.mode === "spheres" ? options.colorTarget : options.outputTarget,
        _profiler: profiler,
        _particleStream: particleStream,
        _useParticleColor: options.useParticleColor,
        _foamPolygonSurfaceDepth: options.foam?.polygonSurfaceDepth,
        ...(aggregate ? { _aggregate: aggregate, _aggregationTask: aggregate.task } : {}),
    } satisfies FluidSimulationRenderLayer;
    try {
        setLayerProfiler(layer, profiler);
        configureFluidSimulationRenderLayer(layer, options);
        if (aggregate) {
            aggregate.setOnUpdate(() => {
                surfaceTask?.setUseParticleColor(layer._useParticleColor ?? aggregate!.usesParticleColor);
            });
            if (options.beforeCompositor) {
                addTaskBefore(options.scene, aggregate.task, compositorTaskOf(options.beforeCompositor));
            } else {
                addTask(options.scene, aggregate.task);
            }
        }
        if (options.beforeCompositor) {
            addTaskBefore(options.scene, task, compositorTaskOf(options.beforeCompositor));
        } else {
            addTask(options.scene, task);
        }
    } catch (error) {
        task.dispose();
        aggregate?.dispose();
        throw error;
    }
    profiler?._owners.add(layer);
    collection._renderLayers.push(layer);
    return layer;
}

export function configureFluidSimulationRenderLayer(layer: FluidSimulationRenderLayer, state: FluidSimulationRenderLayerState): void {
    if (layer.disposed) {
        throw new Error("[fluid] cannot configure a disposed render layer.");
    }
    if (state.profiler?.disposed) {
        throw new Error("[fluid] cannot attach a disposed profiler.");
    }
    if (state.profiler !== undefined) {
        validateBindingEngine(layer._engine as EngineContext, state.profiler, "profiler");
    }
    if (state.enabled !== undefined) {
        layer.enabled = state.enabled;
        taskOf(layer).executionEnabled = state.enabled;
        particleTaskOf(layer)?.setEnabled(state.enabled);
        polygonTaskOf(layer)?.setEnabled(state.enabled);
        foamTaskOf(layer)?.setEnabled(state.enabled);
    }
    if (state.opacity !== undefined) {
        const opacity = Number.isFinite(state.opacity) ? Math.min(1, Math.max(0, state.opacity)) : 1;
        layer.opacity = opacity;
        particleTaskOf(layer)?.setOpacity(opacity);
        surfaceTaskOf(layer)?.setOpacity(opacity);
        polygonTaskOf(layer)?.setOpacity(opacity);
        foamTaskOf(layer)?.setOpacity(opacity);
    }
    if (state.profile !== undefined) {
        const profile = { ...state.profile };
        applyRenderLayerProfile(layer, profile);
        layer._profile = profile;
    }
    if (state.profiler !== undefined) {
        setLayerProfiler(layer, state.profiler);
        layer._profiler?._owners.delete(layer);
        layer._profiler = state.profiler;
        state.profiler?._owners.add(layer);
    }
    if (state.environmentScene !== undefined) {
        applyRenderLayerEnvironment(layer, state.environmentScene);
    }
    if (state.environment !== undefined) {
        applyOpaqueRenderLayerEnvironment(layer, state.environment);
    }
    if (state.environmentRotationY !== undefined) {
        if (!Number.isFinite(state.environmentRotationY)) {
            throw new TypeError("[fluid] environment rotation must be finite.");
        }
        surfaceTaskOf(layer)?.setEnvRotationY(state.environmentRotationY);
        polygonTaskOf(layer)?.setEnvRotationY(state.environmentRotationY);
    }
    if (state.direction !== undefined) {
        const direction = [...state.direction] as [number, number, number];
        surfaceTaskOf(layer)?.setDirLight(direction);
        polygonTaskOf(layer)?.setDirLight(direction);
    }
    if (state.particleColorMode !== undefined) {
        surfaceTaskOf(layer)?.setParticleColorMode(state.particleColorMode);
    }
    if (state.surfaceMode !== undefined) {
        surfaceTaskOf(layer)?.setMode(state.surfaceMode);
    }
    if (state.particleVelocityBrighten !== undefined) {
        if (!Number.isFinite(state.particleVelocityBrighten)) {
            throw new TypeError("[fluid] particle velocity brightening must be finite.");
        }
        particleTaskOf(layer)?.setVelocityBrighten(state.particleVelocityBrighten);
    }
    if (state.useParticleColor !== undefined) {
        layer._useParticleColor = state.useParticleColor;
        surfaceTaskOf(layer)?.setUseParticleColor(state.useParticleColor);
    }
    if (state.debug !== undefined) {
        surfaceTaskOf(layer)?.setDebug(state.debug);
    }
    if (state.polygonWireframe !== undefined) {
        polygonTaskOf(layer)?.setWireframe(state.polygonWireframe);
    }
    if (state.foam !== undefined) {
        const foam = foamTaskOf(layer);
        if (!foam) {
            throw new Error("[fluid] foam render settings require a foam collection layer.");
        }
        layer._foamPolygonSurfaceDepth = state.foam.polygonSurfaceDepth;
        applyFoamRenderSettings(foam, state.foam);
    }
}

export function detachFluidSimulationRenderLayer(simulation: FluidSimulation, layer: FluidSimulationRenderLayer): void {
    if (layer.disposed) {
        return;
    }
    taskOf(layer).executionEnabled = false;
    layer._profiler?._owners.delete(layer);
    layer._profiler = null;
    layer.disposed = true;
    const index = simulation._renderLayers.indexOf(layer);
    if (index >= 0) {
        simulation._renderLayers.splice(index, 1);
    }
    const task = taskOf(layer);
    const aggregate = aggregateRuntimeOf(layer);
    const aggregationTask = layer._aggregationTask as Task | undefined;
    removeFluidTasksFromScene(layer._scene as SceneContext, [aggregationTask, task]);
    if (aggregationTask) {
        aggregationTask.executionEnabled = false;
    }
    retireGpuResources(engineOf(simulation), () => {
        task.dispose();
        aggregate?.dispose();
    });
}

export function detachFluidSimulationCollectionRenderLayer(collection: FluidSimulationCollection, layer: FluidSimulationRenderLayer): void {
    if (layer.disposed) {
        return;
    }
    taskOf(layer).executionEnabled = false;
    const aggregationTask = layer._aggregationTask as Task | undefined;
    if (aggregationTask) {
        aggregationTask.executionEnabled = false;
    }
    layer._profiler?._owners.delete(layer);
    layer._profiler = null;
    layer.disposed = true;
    const index = collection._renderLayers.indexOf(layer);
    if (index >= 0) {
        collection._renderLayers.splice(index, 1);
    }
    const task = taskOf(layer);
    const aggregate = aggregateRuntimeOf(layer);
    removeFluidTasksFromScene(layer._scene as SceneContext, [aggregationTask, task]);
    retireGpuResources(collection._engine as EngineContext, () => {
        task.dispose();
        aggregate?.dispose();
    });
}

export function disposeFluidSimulationCollection(collection: FluidSimulationCollection, disposeSimulations = false): void {
    if (collection.disposed) {
        return;
    }
    for (const layer of [...collection._renderLayers]) {
        detachFluidSimulationCollectionRenderLayer(collection, layer);
    }
    for (const source of collection._sources) {
        source.alpha?._owners.delete(collection);
        source.color?._owners.delete(collection);
    }
    if (disposeSimulations) {
        for (const simulation of collection._simulations) {
            disposeFluidSimulation(simulation);
        }
    }
    collection._sources.length = 0;
    collection._simulations.length = 0;
    collection._disposed = true;
}

export function createFluidSimulationRenderCompositor(engine: EngineContext, options: FluidSimulationRenderCompositorOptions): FluidSimulationRenderCompositor {
    const baseState = { layer: options.baseLayer ?? null };
    const task = createFluidRenderCompositor(engine, options.scene, options.baseColorTarget, () =>
        baseState.layer ? (layerDepthView(baseState.layer) as ReturnType<FluidSurfaceTask["surfaceDepthView"]>) : null
    );
    const layers: FluidSimulationRenderLayer[] = [];
    const compositor = {
        get layers(): readonly FluidSimulationRenderLayer[] {
            return compositor._layers;
        },
        get baseLayer(): FluidSimulationRenderLayer | null {
            return compositor._baseState.layer;
        },
        disposed: false,
        _engine: engine,
        _task: task,
        _scene: options.scene,
        _layers: layers,
        _baseState: baseState,
    } satisfies FluidSimulationRenderCompositor;
    try {
        addTask(options.scene, task);
    } catch (error) {
        task.dispose();
        throw error;
    }
    return compositor;
}

export function configureFluidSimulationRenderCompositor(
    compositor: FluidSimulationRenderCompositor,
    layers: readonly FluidSimulationRenderLayer[],
    baseLayer: FluidSimulationRenderLayer | null = compositor.baseLayer
): void {
    if (compositor.disposed || layers.some((layer) => layer.disposed) || baseLayer?.disposed) {
        throw new Error("[fluid] cannot configure a compositor with disposed state.");
    }
    const attachments = layers.map((layer) => {
        if (!layer._colorTarget || (!surfaceTaskOf(layer) && !polygonTaskOf(layer))) {
            throw new Error("[fluid] composited layers must be surface or polygon attachments with an output target.");
        }
        return {
            colorTarget: layer._colorTarget,
            surfaceDepthView: () => layerDepthView(layer) as ReturnType<FluidSurfaceTask["surfaceDepthView"]>,
        };
    });
    compositorTaskOf(compositor).setLayers(attachments, baseLayer !== null);
    compositor._layers.splice(0, compositor._layers.length, ...layers);
    compositor._baseState.layer = baseLayer;
}

export function disposeFluidSimulationRenderCompositor(compositor: FluidSimulationRenderCompositor): void {
    if (compositor.disposed) {
        return;
    }
    compositor.disposed = true;
    const task = compositorTaskOf(compositor);
    task.executionEnabled = false;
    task.setLayers([], false);
    removeFluidTasksFromScene(compositor._scene as SceneContext, [task]);
    compositor._layers.length = 0;
    compositor._baseState.layer = null;
    retireGpuResources(compositor._engine as EngineContext, () => task.dispose());
}

export function disposeFluidSimulation(simulation: FluidSimulation): void {
    if (simulation.disposed) {
        return;
    }
    for (const layer of [...simulation._renderLayers]) {
        detachFluidSimulationRenderLayer(simulation, layer);
    }
    releaseSimulationBindings(simulation, simulation._options);
    const backend = backendOf(simulation);
    simulation.disposed = true;
    retireGpuResources(engineOf(simulation), () => backend.dispose());
}
