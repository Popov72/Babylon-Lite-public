import {
    cancelFluidCollectionReconfiguration,
    cancelFluidReconfiguration,
    commitFluidCollectionReconfiguration,
    commitFluidReconfiguration,
    configureFluidSimulationRenderLayer,
    getFluidSimulationCollectionDiagnostics,
    getFluidSimulationDiagnostics,
    prepareFluidCollectionReconfiguration,
    prepareFluidReconfiguration,
    setFluidSimulationFoam,
    setFluidSimulationParameter,
} from "../core/fluid-facade.js";
import type {
    FluidSimulation,
    FluidSimulationCollection,
    FluidSimulationOptions,
    FluidSimulationRenderLayer,
    FluidSimulationReconfigurationRequest,
    PreparedFluidCollectionReconfiguration,
    PreparedFluidReconfiguration,
} from "../core/fluid-facade.js";
import { fluidMaximumPageCapacity, resolveFluidAllocationPlan } from "../core/allocation-plan.js";
import type { FluidAllocationPlan, FluidDeviceLimitsSnapshot } from "../core/allocation-plan.js";
import type { FluidControlValues, FluidControlsHandle, FluidFoamValues } from "./controls-panel.js";
import { resolveFluidControlsCapabilities } from "./controls-capabilities.js";
import type { FluidControlsCapabilities } from "./controls-capabilities.js";
import { normalizeFluidFlipDiscretization, resolveFluidRenderMode } from "../core/fluid-policy.js";
import type { FluidRenderMode } from "../core/fluid-policy.js";
import { fluidInitialEmitterVolume } from "../core/initial-state-plan.js";
import { resolveFluidSimulationConfig } from "../core/simulation-config.js";
import type { FluidDebug } from "../rendering/fluid-surface-render.js";
import type { FoamDebugTexture } from "../rendering/foam-render.js";
import type { FoamConfig } from "../core/sim-common.js";

type FluidMethod = FluidSimulationOptions["method"];
type ControlKey = keyof FluidControlValues;

const RECONFIGURATION_KEYS: readonly ControlKey[] = [
    "method",
    "material",
    "count",
    "physScale",
    "gridPosition",
    "gridSize",
    "gridResolution",
    "markersPerCell",
    "activeBlocks",
    "pagedGrid",
    "pagedGridMaxPages",
    "fusedBlockDiscovery",
];

const GRID_DEFERRED_RECONFIGURATION_KEYS: readonly ControlKey[] = ["gridPosition", "gridSize"];
const FLIP_DEFERRED_RECONFIGURATION_KEYS: readonly ControlKey[] = [
    "count",
    ...GRID_DEFERRED_RECONFIGURATION_KEYS,
    "gridResolution",
    "markersPerCell",
    "pagedGrid",
    "pagedGridMaxPages",
];
const MLS_MPM_DEFERRED_RECONFIGURATION_KEYS: readonly ControlKey[] = [...GRID_DEFERRED_RECONFIGURATION_KEYS, "pagedGridMaxPages"];

const RENDER_PROFILE_KEYS: readonly ControlKey[] = [
    "polygonShader",
    "color",
    "absorption",
    "size",
    "refraction",
    "specular",
    "reflectionExposure",
    "reflectionContrast",
    "reflectivity",
    "depthBlur",
    "depthBlurThreshold",
    "thicknessBlur",
    "half",
    "surfaceFilter",
    "narrowDelta",
    "narrowMu",
    "anisotropic",
    "anisoSurfScale",
    "thicknessDownscale",
];

export interface FluidControlsPageDiagnostics {
    readonly method: "FLIP" | "MLS-MPM";
    readonly requiredPages: number;
    readonly capacity: number;
    readonly text: string;
    readonly overflow: boolean;
}

export interface FluidControlsApplicationPlan {
    readonly changedKeys: readonly ControlKey[];
    readonly reconfigure: boolean;
    readonly restartRequired: boolean;
    readonly renderProfile: boolean;
    readonly renderMode: boolean;
    readonly foamRender: boolean;
    readonly debugRender: boolean;
}

export interface FluidControlsMemoryTarget {
    readonly options: FluidSimulationOptions;
    readonly activeCount?: number;
    readonly restartActiveCount?: number;
    readonly restartActiveCountEstimated?: boolean;
    readonly currentGpuBytes?: number;
}

export interface FluidControlsMemoryProjection {
    readonly steadyBytes: number;
    readonly transitionPeakBytes: number;
    readonly foamCapacity: number;
    readonly restartActiveCount: number;
    readonly restartParticleCount: number;
    readonly restartActiveCountEstimated: boolean;
    readonly plans: readonly FluidAllocationPlan[];
}

export interface FluidControlsSceneTargetOptions
    extends
        Pick<FluidSimulationOptions, "bounds">,
        Partial<
            Pick<
                FluidSimulationOptions,
                | "particleCount"
                | "compatibilityProfile"
                | "explicitGrid"
                | "domainScale"
                | "semantics"
                | "samplingType"
                | "particleRadius"
                | "groundY"
                | "backend"
                | "flow"
                | "initialPositions"
                | "initialVelocities"
                | "sceneSdf"
                | "forceField"
                | "profiler"
            >
        > {}

export interface FluidControlsBindingTarget {
    readonly simulation: FluidSimulation;
    /** Scene-owned bounds, flow, bindings and target-specific particle layout. */
    readonly options: FluidControlsSceneTargetOptions;
    readonly preserveState?: boolean;
    readonly activeCount?: number;
}

export interface FluidControlsRenderLayers {
    readonly particles?: FluidSimulationRenderLayer;
    readonly surface?: FluidSimulationRenderLayer;
    readonly polygon?: FluidSimulationRenderLayer;
    readonly foam?: FluidSimulationRenderLayer;
}

export interface BindFluidControlsOptions<THostState = unknown> {
    readonly controls: FluidControlsHandle;
    readonly target: FluidSimulation | FluidSimulationCollection;
    readonly deviceLimits: FluidDeviceLimitsSnapshot;
    readonly renderLayers?: FluidControlsRenderLayers;
    readonly maxParticleCount?: number;
    readonly capabilities?: Partial<FluidControlsCapabilities>;
    readonly timestampQuerySupported?: boolean;
    readonly surfaceDebugActive?: () => boolean;
    readonly simulationOpacity?: () => number;
    readonly resolveTarget: (simulation: FluidSimulation, snapshot: Readonly<FluidControlValues>) => FluidControlsBindingTarget;
    readonly captureHostState?: () => THostState;
    readonly applyHostState?: (snapshot: Readonly<FluidControlValues>, changedKeys: readonly ControlKey[]) => void;
    readonly restoreHostState?: (state: THostState) => void;
    readonly onPageDiagnostics?: (diagnostics: FluidControlsPageDiagnostics) => void;
}

interface FluidControlsBindingRuntime<THostState = unknown> {
    readonly options: BindFluidControlsOptions<THostState>;
    applying: boolean;
}

interface FluidControlsRestartTarget {
    readonly simulation: FluidSimulation;
    readonly method: FluidMethod;
    readonly backend?: FluidSimulationOptions["backend"];
    readonly particleCount: number;
    readonly gridResolution?: number;
    readonly markersPerCell?: number;
    readonly pagedGrid: boolean;
    readonly pagedGridMaxPages?: number;
    readonly bounds: FluidSimulationOptions["bounds"];
}

interface FluidControlsSyncCache {
    readonly snapshot: FluidControlValues;
    readonly restartTargets: readonly FluidControlsRestartTarget[];
    readonly pageCapacityLimit: number;
}

/**
 * Pure-state shared controls binding. Behaviour is provided by the standalone functions below.
 */
export interface FluidControlsBinding {
    snapshot: FluidControlValues;
    plan: FluidControlsApplicationPlan;
    renderMode: FluidRenderMode;
    memory: FluidControlsMemoryProjection;
    pageDiagnostics: FluidControlsPageDiagnostics | null;
    capabilities: FluidControlsCapabilities;
    disposed: boolean;
    /** @internal */
    _runtime: FluidControlsBindingRuntime;
    /** @internal */
    _syncCache: FluidControlsSyncCache | null;
}

export interface SyncFluidControlsInput {
    readonly pageDiagnostics?: {
        readonly method: "FLIP" | "MLS-MPM";
        readonly requiredPages: number;
        readonly capacity: number;
        readonly overflow?: boolean;
    };
    /** @internal */
    readonly _targets?: readonly FluidControlsBindingTarget[];
}

export interface FluidControlsNormalizationOptions {
    readonly maxParticleCount?: number;
    readonly pageCapacityLimit?: (snapshot: Readonly<FluidControlValues>) => number;
    readonly capabilities?: Partial<FluidControlsCapabilities>;
    readonly timestampQuerySupported?: boolean;
}

function fluidMethod(value: string): FluidMethod {
    if (value === "PBF" || value === "FLIP" || value === "MLS-MPM" || value === "PB-MPM") {
        return value;
    }
    throw new RangeError(`[fluid] unsupported controls method "${value}".`);
}

function finite(value: number, name: string): number {
    if (!Number.isFinite(value)) {
        throw new TypeError(`[fluid] ${name} must be finite.`);
    }
    return value;
}

function positiveInteger(value: number, name: string, maximum = Number.MAX_SAFE_INTEGER): number {
    return Math.min(maximum, Math.max(1, Math.round(finite(value, name))));
}

function normalizedFoam(foam: FluidFoamValues): FluidFoamValues {
    const result = structuredClone(foam);
    result.activeParticles = true;
    result.generateSpray ??= true;
    result.generateFoam ??= true;
    result.generateBubbles ??= true;
    result.poolScale = Math.max(0, finite(result.poolScale, "foam.poolScale"));
    for (const key of [
        "kTa",
        "kWc",
        "kTurb",
        "energySpeedMin",
        "energySpeedMax",
        "curvatureMin",
        "curvatureMax",
        "turbulenceMin",
        "turbulenceMax",
        "foamLayerDepth",
        "sprayDrag",
        "kb",
        "kd",
        "tMin",
        "tMax",
        "size",
        "blurRadius",
        "lightIntensity",
        "ambient",
        "aoStrength",
        "normalStrength",
        "softness",
        "density",
        "subsurfaceStrength",
        "spraySize",
        "sprayIntensity",
        "spraySeparation",
    ] as const) {
        const fallback = key === "spraySize" ? 0.55 : key === "sprayIntensity" ? 1.4 : key === "spraySeparation" ? 1 : undefined;
        result[key] = finite(result[key] ?? fallback!, `foam.${key}`);
    }
    if (result.energySpeedMax < result.energySpeedMin) {
        [result.energySpeedMin, result.energySpeedMax] = [result.energySpeedMax, result.energySpeedMin];
    }
    if (result.curvatureMax < result.curvatureMin) {
        [result.curvatureMin, result.curvatureMax] = [result.curvatureMax, result.curvatureMin];
    }
    if (result.turbulenceMax < result.turbulenceMin) {
        [result.turbulenceMin, result.turbulenceMax] = [result.turbulenceMax, result.turbulenceMin];
    }
    if (result.tMax < result.tMin) {
        [result.tMin, result.tMax] = [result.tMax, result.tMin];
    }
    return result;
}

export function normalizeFluidControls(values: Readonly<FluidControlValues>, options: FluidControlsNormalizationOptions = {}): FluidControlValues {
    const normalized = structuredClone(values) as FluidControlValues;
    const method = fluidMethod(normalized.method);
    const capabilities = resolveFluidControlsCapabilities({
        method,
        timestampQuerySupported: options.timestampQuerySupported,
        hostCapabilities: options.capabilities,
    });
    normalized.count = positiveInteger(normalized.count, "particle count", options.maxParticleCount);
    normalized.material = Math.max(0, Math.round(finite(normalized.material, "material")));
    normalized.physScale = Math.max(0.01, finite(normalized.physScale, "physics scale"));
    normalized.gridPosition = normalized.gridPosition.map((value, axis) => finite(value, `gridPosition[${axis}]`)) as [number, number, number];
    normalized.gridSize = normalized.gridSize.map((value, axis) => {
        const size = finite(value, `gridSize[${axis}]`);
        if (size <= 0) {
            throw new RangeError(`[fluid] gridSize[${axis}] must be positive.`);
        }
        return size;
    }) as [number, number, number];
    const discretization = normalizeFluidFlipDiscretization(normalized.gridResolution, normalized.markersPerCell);
    normalized.gridResolution = discretization.gridResolution;
    normalized.markersPerCell = discretization.markersPerCell;
    normalized.pagedGridMaxPages = positiveInteger(normalized.pagedGridMaxPages, "page capacity");
    if ((method === "FLIP" || method === "MLS-MPM") && options.pageCapacityLimit) {
        normalized.pagedGridMaxPages = Math.min(normalized.pagedGridMaxPages, positiveInteger(options.pageCapacityLimit(normalized), "maximum page capacity"));
    }
    if (method === "MLS-MPM" && normalized.pagedGrid) {
        normalized.activeBlocks = true;
    }
    if (!capabilities.pagedGrid) {
        normalized.pagedGrid = false;
    }
    if (!capabilities.activeBlocks) {
        normalized.activeBlocks = false;
    }
    if (!capabilities.polygonSurface) {
        normalized.schema.polygonSurface = 0;
    }
    if (!capabilities.pressureDiagnostics) {
        normalized.schema.pressureDiagnostics = 0;
        normalized.schema.pressureTolerance = 0;
    }
    for (const [key, value] of Object.entries(normalized.schema)) {
        normalized.schema[key] = finite(value, `schema.${key}`);
    }
    if (capabilities.physicsParameters) {
        const schema: Record<string, number> = {};
        for (const key of capabilities.physicsParameters) {
            if (normalized.schema[key] !== undefined) {
                schema[key] = normalized.schema[key]!;
            }
        }
        normalized.schema = schema;
    }
    normalized.foam = normalizedFoam(normalized.foam);
    if (!capabilities.foam) {
        normalized.foam.enabled = false;
    }
    return normalized;
}

export function deriveFluidControlsApplicationPlan(
    previous: Readonly<FluidControlValues>,
    next: Readonly<FluidControlValues>,
    _changedKeys?: readonly ControlKey[]
): FluidControlsApplicationPlan {
    const changed: ControlKey[] = [];
    for (const key of Object.keys(next) as ControlKey[]) {
        if (JSON.stringify(previous[key]) !== JSON.stringify(next[key])) {
            changed.push(key);
        }
    }
    const has = (keys: readonly ControlKey[]): boolean => keys.some((key) => changed.includes(key));
    const polygonSurfaceChanged = previous.schema.polygonSurface !== next.schema.polygonSurface;
    const deferredKeys =
        next.method === "FLIP" ? FLIP_DEFERRED_RECONFIGURATION_KEYS : next.method === "MLS-MPM" ? MLS_MPM_DEFERRED_RECONFIGURATION_KEYS : GRID_DEFERRED_RECONFIGURATION_KEYS;
    const reconfigure = RECONFIGURATION_KEYS.some((key) => changed.includes(key) && !deferredKeys.includes(key));
    return {
        changedKeys: changed,
        reconfigure,
        restartRequired: has(deferredKeys) && !reconfigure,
        renderProfile: has(RENDER_PROFILE_KEYS),
        renderMode: has(["method", "renderMode", "anisotropic", "foam", "debug"]) || polygonSurfaceChanged,
        foamRender: has(["foam", "renderMode", "debug"]) || polygonSurfaceChanged,
        debugRender: has(["debug"]),
    };
}

export function fluidFoamConfigFromControls(foam: Readonly<FluidFoamValues>): FoamConfig | null {
    if (!foam.enabled) {
        return null;
    }
    return {
        activeParticles: true,
        generateSpray: foam.generateSpray ?? true,
        generateFoam: foam.generateFoam ?? true,
        generateBubbles: foam.generateBubbles ?? true,
        kTa: foam.kTa,
        kWc: foam.kWc,
        kTurb: foam.kTurb,
        energySpeedMin: foam.energySpeedMin,
        energySpeedMax: foam.energySpeedMax,
        curvatureMin: foam.curvatureMin,
        curvatureMax: foam.curvatureMax,
        turbulenceMin: foam.turbulenceMin,
        turbulenceMax: foam.turbulenceMax,
        foamLayerDepth: foam.foamLayerDepth,
        sprayDrag: foam.sprayDrag,
        kb: foam.kb,
        kd: foam.kd,
        tMin: foam.tMin,
        tMax: foam.tMax,
        poolScale: foam.poolScale,
    };
}

function commonSimulationOptions(binding: FluidControlsBinding, target: FluidControlsBindingTarget, snapshot: FluidControlValues): FluidSimulationOptions {
    const method = fluidMethod(snapshot.method);
    const base = target.options;
    const physics: Record<string, number> = {};
    if (base.backend) {
        if (snapshot.foam.enabled && !base.backend.supportsFoam) {
            throw new Error("[fluid] selected backend does not support foam.");
        }
        if ((snapshot.schema.polygonSurface ?? 0) > 0 && !base.backend.renderModes.includes("polygon")) {
            throw new Error("[fluid] selected backend does not support polygon rendering.");
        }
        for (const key of base.backend.physicsParameters) {
            if (snapshot.schema[key] !== undefined) {
                physics[key] = snapshot.schema[key]!;
            }
        }
    } else {
        Object.assign(physics, snapshot.schema);
    }
    const publish = (requiredPages: number, capacity: number, overflow: boolean): void => {
        if (binding.disposed || binding.snapshot.method !== method) {
            return;
        }
        publishPageDiagnostics(binding, method, requiredPages, capacity, overflow);
    };
    return {
        ...base,
        method,
        // Keep an explicit undefined when returning to production so collection update-merging
        // replaces, rather than accidentally retains, the currently committed provider.
        backend: base.backend,
        particleCount: positiveInteger(base.particleCount ?? snapshot.count, "target particle count"),
        physicsScale: snapshot.physScale,
        physics,
        explicitGrid: base.explicitGrid ?? true,
        gridResolution: method === "FLIP" ? snapshot.gridResolution : undefined,
        markersPerCell: method === "FLIP" ? snapshot.markersPerCell : undefined,
        material: method === "PB-MPM" ? snapshot.material : undefined,
        activeBlocks: method === "MLS-MPM" ? snapshot.activeBlocks || snapshot.pagedGrid : false,
        pagedGrid: method === "FLIP" || method === "MLS-MPM" ? snapshot.pagedGrid : false,
        pagedGridMaxPages: method === "FLIP" || method === "MLS-MPM" ? snapshot.pagedGridMaxPages : undefined,
        fusedBlockDiscovery: method === "MLS-MPM" ? snapshot.fusedBlockDiscovery : false,
        foam: fluidFoamConfigFromControls(snapshot.foam),
        onPagedGridPages: method === "FLIP" || method === "MLS-MPM" ? (requiredPages, capacity) => publish(requiredPages, capacity, false) : undefined,
        onPagedGridOverflow: method === "FLIP" || method === "MLS-MPM" ? (requiredPages, capacity) => publish(requiredPages, capacity, true) : undefined,
    };
}

function allocationPlan(options: FluidSimulationOptions, activeCount?: number, limits?: FluidDeviceLimitsSnapshot): FluidAllocationPlan {
    const config = resolveFluidSimulationConfig(options.compatibilityProfile ?? "fluid", {
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
    if (!config.gridDim) {
        throw new Error("[fluid] controls memory projection requires resolved grid dimensions.");
    }
    if (options.backend?._allocationPlan) {
        return options.backend._allocationPlan(options, config, limits);
    }
    const physics = config.physics;
    const foam = options.foam;
    return resolveFluidAllocationPlan({
        method: options.method,
        particleCount: options.particleCount,
        gridDim: config.gridDim,
        ...(options.method === "FLIP"
            ? {
                  flipWarmup: {
                      initialLiveCount: activeCount ?? options.particleCount,
                      initialTargetCount: activeCount ?? options.particleCount,
                  },
              }
            : {}),
        pressureSolver: (physics.pressureSolver ?? 0) >= 0.5 ? "multigrid" : "jacobi",
        pagedGrid: options.pagedGrid,
        pagedGridMaxPages: options.pagedGridMaxPages,
        activeBlocks: options.activeBlocks,
        quality: {
            pagedGrid: options.pagedGrid,
            pagedGridMaxPages: options.pagedGridMaxPages,
            pressureDiagnostics: (physics.pressureDiagnostics ?? 0) >= 0.5 || (physics.pressureTolerance ?? 0) > 0,
            liquidSdf: (physics.liquidSdf ?? 0) >= 0.5,
            fractionalSolids: (physics.fractionalSolids ?? 0) >= 0.5,
            reseedParticles: (physics.reseedParticles ?? 0) >= 0.5,
            particleSheeting: (physics.particleSheeting ?? 0) >= 0.5,
            polygonSurface: (physics.polygonSurface ?? 0) >= 0.5,
            polygonReconstructionMultiplier: physics.polygonReconstructionMultiplier ?? 1,
        },
        foam: foam
            ? {
                  enabled: true,
                  activeParticles: foam.activeParticles,
                  poolScale: foam.poolScale,
              }
            : { enabled: false },
        limits,
    });
}

export function projectFluidControlsMemory(targets: readonly FluidControlsMemoryTarget[], limits?: FluidDeviceLimitsSnapshot): FluidControlsMemoryProjection {
    const plans = targets.map((target) => allocationPlan(target.options, target.activeCount, limits));
    const steadyBytes = plans.reduce((sum, plan) => sum + plan.steadyBytes, 0);
    const currentBytes = targets.reduce((sum, target) => sum + (target.currentGpuBytes ?? 0), 0);
    return {
        steadyBytes,
        transitionPeakBytes: currentBytes + steadyBytes,
        foamCapacity: plans.reduce((sum, plan) => sum + plan.foamCapacity, 0),
        restartActiveCount: targets.reduce((sum, target) => sum + (target.restartActiveCount ?? target.activeCount ?? target.options.particleCount), 0),
        restartParticleCount: targets.reduce((sum, target) => sum + target.options.particleCount, 0),
        restartActiveCountEstimated: targets.some((target) => target.restartActiveCountEstimated === true),
        plans,
    };
}

function renderModeFor(binding: FluidControlsBinding, snapshot: FluidControlValues): FluidRenderMode {
    return resolveFluidRenderMode({
        method: fluidMethod(snapshot.method),
        renderSpheres: snapshot.renderMode === "spheres",
        anisotropicSurface: snapshot.anisotropic,
        polygonSurface: (snapshot.schema.polygonSurface ?? 0) >= 0.5,
        foamEnabled: snapshot.foam.enabled,
        surfaceDebugActive: binding._runtime.options.surfaceDebugActive?.() ?? snapshot.debug !== "none",
    });
}

function applyRenderState(binding: FluidControlsBinding, snapshot: FluidControlValues): FluidRenderMode {
    const mode = renderModeFor(binding, snapshot);
    const layers = binding._runtime.options.renderLayers;
    const profile = {
        waterColor: snapshot.color,
        polygonShader: snapshot.polygonShader,
        absorption: snapshot.absorption,
        particleSize: snapshot.size,
        refractionStrength: snapshot.refraction,
        specularPower: snapshot.specular,
        reflectionExposure: snapshot.reflectionExposure,
        reflectionContrast: snapshot.reflectionContrast,
        waterReflectivity: snapshot.reflectivity,
        surfaceDepthBlur: snapshot.depthBlur,
        depthBlurEdgeThreshold: snapshot.depthBlurThreshold,
        surfaceThicknessBlur: snapshot.thicknessBlur,
        halfRendering: snapshot.half,
        surfaceFilter: snapshot.surfaceFilter,
        narrowRangeDelta: snapshot.narrowDelta,
        narrowRangeMu: snapshot.narrowMu,
        anisotropicSurface: snapshot.anisotropic,
        anisoRadiusDamping: snapshot.anisoSurfScale,
        thicknessDownscale: snapshot.thicknessDownscale,
    };
    if (layers?.particles) {
        configureFluidSimulationRenderLayer(layers.particles, { enabled: mode.particleEnabled, profile });
    }
    if (layers?.surface) {
        configureFluidSimulationRenderLayer(layers.surface, {
            profile,
            surfaceMode: mode.surfaceMode,
            debug: snapshot.debug === "polygonWireframe" ? "none" : (snapshot.debug as FluidDebug),
        });
    }
    if (layers?.polygon) {
        configureFluidSimulationRenderLayer(layers.polygon, {
            enabled: mode.polygonEnabled,
            profile,
            polygonWireframe: snapshot.debug === "polygonWireframe",
        });
    }
    if (layers?.foam) {
        const foam = snapshot.foam;
        configureFluidSimulationRenderLayer(layers.foam, {
            enabled: (binding._runtime.options.simulationOpacity?.() ?? 1) > 0 && mode.foamEnabled,
            foam: {
                polygonSurfaceDepth: mode.foamPolygonSurfaceDepth,
                surfaceFiltering: foam.surfaceFiltering ?? snapshot.method === "FLIP",
                softness: foam.softness,
                density: foam.density,
                subsurfaceStrength: foam.subsurfaceStrength,
                subsurfaceColor: foam.subsurfaceColor,
                sizeScale: foam.size,
                blurRadius: foam.blurRadius,
                lightIntensity: foam.lightIntensity,
                ambient: foam.ambient,
                aoStrength: foam.aoStrength,
                normalStrength: foam.normalStrength,
                spraySize: foam.spraySize ?? 0.55,
                sprayIntensity: foam.sprayIntensity ?? 1.4,
                spraySeparation: foam.spraySeparation ?? 1,
                debugByKind: foam.debugByKind ?? false,
                debugTexture: foam.debugTexture as FoamDebugTexture,
            },
        });
    }
    return mode;
}

function targetSimulations(target: FluidSimulation | FluidSimulationCollection): readonly FluidSimulation[] {
    return "simulations" in target ? target.simulations.filter((simulation) => !simulation.disposed) : target.disposed ? [] : [target];
}

function projectedRestartActiveCount(options: FluidSimulationOptions): { count: number; estimated: boolean } {
    if (options.initialPositions) {
        return { count: Math.min(options.particleCount, Math.floor(options.initialPositions.length / 3)), estimated: false };
    }
    if (!options.flow) {
        return { count: options.particleCount, estimated: false };
    }
    if (options.flow.initialEmittersFillCapacity) {
        return { count: options.particleCount, estimated: false };
    }
    const config = resolveFluidSimulationConfig(options.compatibilityProfile ?? "fluid", {
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
    const initialVolume = fluidInitialEmitterVolume(options.flow);
    return {
        count: Math.min(options.particleCount, Math.max(0, Math.ceil(initialVolume / Math.max(config.particleVolume, 1e-12)))),
        estimated: options.method === "FLIP" && options.flow.emitters.some((emitter) => emitter.enabled && emitter.behavior === "initial" && emitter.sampling === "volume"),
    };
}

function projectionFor(
    binding: FluidControlsBinding,
    snapshot: FluidControlValues,
    targets: readonly FluidControlsBindingTarget[],
    restartRequired = false
): FluidControlsMemoryProjection {
    return projectFluidControlsMemory(
        targets.map((target) => {
            const options = commonSimulationOptions(binding, target, snapshot);
            const restartActive = restartRequired ? projectedRestartActiveCount(options) : null;
            return {
                options,
                activeCount: target.activeCount,
                restartActiveCount: restartActive?.count ?? target.activeCount,
                restartActiveCountEstimated: restartActive?.estimated ?? false,
                currentGpuBytes: target.simulation.gpuBytes,
            };
        }),
        binding._runtime.options.deviceLimits
    );
}

function pageCapacityLimitFor(binding: FluidControlsBinding, snapshot: FluidControlValues, targets: readonly FluidControlsBindingTarget[]): number {
    const method = fluidMethod(snapshot.method);
    if (method !== "FLIP" && method !== "MLS-MPM") {
        return snapshot.pagedGridMaxPages;
    }
    const limit = targets.reduce((current, target) => {
        const options = commonSimulationOptions(binding, target, snapshot);
        const config = resolveFluidSimulationConfig(options.compatibilityProfile ?? "fluid", {
            method,
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
        return config.gridDim ? Math.min(current, fluidMaximumPageCapacity(method, config.gridDim, binding._runtime.options.deviceLimits)) : current;
    }, Number.MAX_SAFE_INTEGER);
    return limit === Number.MAX_SAFE_INTEGER ? snapshot.pagedGridMaxPages : limit;
}

function clampPageCapacity(binding: FluidControlsBinding, snapshot: FluidControlValues, targets: readonly FluidControlsBindingTarget[]): FluidControlValues {
    const limit = Math.max(1, pageCapacityLimitFor(binding, snapshot, targets));
    if (snapshot.pagedGridMaxPages <= limit) {
        return snapshot;
    }
    return { ...snapshot, pagedGridMaxPages: limit };
}

function changedSchema(previous: FluidControlValues, next: FluidControlValues): [string, number][] {
    return Object.entries(next.schema).filter(([key, value]) => previous.schema[key] !== value);
}

function simulationFoamChanged(previous: FluidControlValues, next: FluidControlValues): boolean {
    return JSON.stringify(fluidFoamConfigFromControls(previous.foam)) !== JSON.stringify(fluidFoamConfigFromControls(next.foam));
}

function applyLiveSimulationState(targets: readonly FluidControlsBindingTarget[], previous: FluidControlValues, next: FluidControlValues): void {
    const schema = changedSchema(previous, next);
    const foamChanged = simulationFoamChanged(previous, next);
    const operations: Array<{ apply: () => void; rollback: () => void }> = [];
    for (const target of targets) {
        if (foamChanged) {
            const previousFoam = target.simulation.options.foam ? { ...target.simulation.options.foam } : null;
            const nextFoam = fluidFoamConfigFromControls(next.foam);
            operations.push({
                apply: () => setFluidSimulationFoam(target.simulation, nextFoam),
                rollback: () => setFluidSimulationFoam(target.simulation, previousFoam),
            });
        }
        for (const [key, value] of schema) {
            if (target.simulation.options.backend && !target.simulation.options.backend.physicsParameters.includes(key)) {
                continue;
            }
            const previousValue = target.simulation.options.physics?.[key] ?? previous.schema[key];
            if (!Number.isFinite(previousValue)) {
                throw new Error(`[fluid] cannot transactionally update simulation parameter "${key}" without its previous value.`);
            }
            operations.push({
                apply: () => setFluidSimulationParameter(target.simulation, key, value),
                rollback: () => setFluidSimulationParameter(target.simulation, key, previousValue!),
            });
        }
    }
    const applied: Array<() => void> = [];
    try {
        for (const operation of operations) {
            applied.push(operation.rollback);
            operation.apply();
        }
    } catch (error) {
        const failures: unknown[] = [error];
        for (let index = applied.length - 1; index >= 0; index--) {
            try {
                applied[index]!();
            } catch (rollbackError) {
                failures.push(rollbackError);
            }
        }
        if (failures.length > 1) {
            throw new AggregateError(failures, "[fluid] live controls update failed and could not be completely rolled back.", { cause: error });
        }
        throw error;
    }
}

function equalBounds(a: FluidSimulationOptions["bounds"], b: FluidSimulationOptions["bounds"]): boolean {
    return a.min.every((value, axis) => value === b.min[axis]) && a.max.every((value, axis) => value === b.max[axis]);
}

function backendChanged(current: FluidSimulationOptions["backend"], desired: FluidSimulationOptions["backend"]): boolean {
    return current !== desired || current?.id !== desired?.id;
}

function preserveReconfiguredState(target: FluidControlsBindingTarget): boolean {
    return target.options.backend || target.simulation.options.backend ? false : (target.preserveState ?? true);
}

function restartTargetsFor(binding: FluidControlsBinding, snapshot: FluidControlValues, targets: readonly FluidControlsBindingTarget[]): FluidControlsRestartTarget[] {
    return targets.map((target) => {
        const desired = commonSimulationOptions(binding, target, snapshot);
        return {
            simulation: target.simulation,
            method: desired.method,
            backend: desired.backend,
            particleCount: desired.particleCount,
            gridResolution: desired.gridResolution,
            markersPerCell: desired.markersPerCell,
            pagedGrid: desired.pagedGrid ?? false,
            pagedGridMaxPages: desired.pagedGridMaxPages,
            bounds: { min: [...desired.bounds.min], max: [...desired.bounds.max] },
        };
    });
}

function hasPendingRestartTargets(snapshot: FluidControlValues, targets: readonly FluidControlsRestartTarget[]): boolean {
    return targets.some((target) => {
        const current = target.simulation.options;
        if (backendChanged(current.backend, target.backend)) {
            return true;
        }
        const boundsChanged = !equalBounds(current.bounds, target.bounds);
        const pagedGridChanged = (current.pagedGrid ?? false) !== target.pagedGrid;
        const pageCapacityChanged = target.pagedGrid && current.pagedGridMaxPages !== target.pagedGridMaxPages;
        if (snapshot.method === "MLS-MPM") {
            return boundsChanged || pageCapacityChanged;
        }
        if (snapshot.method !== "FLIP") {
            return boundsChanged;
        }
        return (
            current.method !== target.method ||
            current.particleCount !== target.particleCount ||
            current.gridResolution !== target.gridResolution ||
            current.markersPerCell !== target.markersPerCell ||
            boundsChanged ||
            pagedGridChanged ||
            pageCapacityChanged
        );
    });
}

function hasPendingRestart(binding: FluidControlsBinding, snapshot: FluidControlValues, targets: readonly FluidControlsBindingTarget[]): boolean {
    return hasPendingRestartTargets(snapshot, restartTargetsFor(binding, snapshot, targets));
}

function sameTargetSimulations(targets: readonly FluidControlsRestartTarget[], simulations: readonly FluidSimulation[]): boolean {
    return targets.length === simulations.length && targets.every((target, index) => target.simulation === simulations[index]);
}

function refreshSyncProjection(binding: FluidControlsBinding, targets: readonly FluidControlsBindingTarget[], simulations: readonly FluidSimulation[]): FluidControlsSyncCache {
    const restartTargets = restartTargetsFor(binding, binding.snapshot, targets);
    if (!sameTargetSimulations(restartTargets, simulations)) {
        throw new Error("[fluid] controls target resolution did not match the live simulation set.");
    }
    const restartRequired = hasPendingRestartTargets(binding.snapshot, restartTargets);
    const method = fluidMethod(binding.snapshot.method);
    const cache: FluidControlsSyncCache = {
        snapshot: binding.snapshot,
        restartTargets,
        pageCapacityLimit: method === "FLIP" || method === "MLS-MPM" ? pageCapacityLimitFor(binding, binding.snapshot, targets) : binding.snapshot.pagedGridMaxPages,
    };
    const memory = projectionFor(binding, binding.snapshot, targets, restartRequired);
    binding.memory = memory;
    binding.plan = { ...binding.plan, restartRequired };
    binding._syncCache = cache;
    return cache;
}

function publishPageDiagnostics(binding: FluidControlsBinding, method: FluidMethod, requiredPages: number, capacity: number, overflow: boolean): void {
    if (method !== "FLIP" && method !== "MLS-MPM") {
        return;
    }
    const required = Math.max(0, Math.floor(requiredPages));
    const available = Math.max(0, Math.floor(capacity));
    const diagnostics: FluidControlsPageDiagnostics = {
        method,
        requiredPages: required,
        capacity: available,
        text: overflow
            ? `Page capacity exceeded: ${required.toLocaleString()} required, ${available.toLocaleString()} allocated.`
            : `${required.toLocaleString()}\u00a0/\u00a0${available.toLocaleString()}\u00a0pages`,
        overflow,
    };
    binding.pageDiagnostics = diagnostics;
    binding._runtime.options.controls.setPagedGridStatus(diagnostics.text, overflow);
    binding._runtime.options.onPageDiagnostics?.(diagnostics);
}

function effectiveCapabilities(options: Pick<BindFluidControlsOptions, "controls" | "capabilities">): Partial<FluidControlsCapabilities> {
    return { ...options.controls.capabilityOverrides, ...options.capabilities };
}

export function bindFluidControls<THostState = unknown>(options: BindFluidControlsOptions<THostState>): FluidControlsBinding {
    let snapshot = normalizeFluidControls(options.controls.getValues(), {
        maxParticleCount: options.maxParticleCount,
        capabilities: effectiveCapabilities(options),
        timestampQuerySupported: options.timestampQuerySupported,
    });
    const emptyPlan = deriveFluidControlsApplicationPlan(snapshot, snapshot, []);
    const binding: FluidControlsBinding = {
        snapshot,
        plan: emptyPlan,
        renderMode: resolveFluidRenderMode({
            method: fluidMethod(snapshot.method),
            renderSpheres: snapshot.renderMode === "spheres",
            anisotropicSurface: snapshot.anisotropic,
            polygonSurface: (snapshot.schema.polygonSurface ?? 0) >= 0.5,
            foamEnabled: snapshot.foam.enabled,
            surfaceDebugActive: options.surfaceDebugActive?.() ?? snapshot.debug !== "none",
        }),
        memory: { steadyBytes: 0, transitionPeakBytes: 0, foamCapacity: 0, restartActiveCount: 0, restartParticleCount: 0, restartActiveCountEstimated: false, plans: [] },
        pageDiagnostics: null,
        capabilities: resolveFluidControlsCapabilities({
            method: snapshot.method,
            timestampQuerySupported: options.timestampQuerySupported,
            hostCapabilities: effectiveCapabilities(options),
        }),
        disposed: false,
        _runtime: { options: options as BindFluidControlsOptions<unknown>, applying: false },
        _syncCache: null,
    };
    let targets = targetSimulations(options.target).map((simulation) => options.resolveTarget(simulation, snapshot));
    const boundedSnapshot = clampPageCapacity(binding, snapshot, targets);
    if (boundedSnapshot !== snapshot) {
        snapshot = boundedSnapshot;
        binding.snapshot = snapshot;
        options.applyHostState?.(snapshot, ["pagedGridMaxPages"]);
        targets = targetSimulations(options.target).map((simulation) => options.resolveTarget(simulation, snapshot));
    }
    const restartRequired = hasPendingRestart(binding, snapshot, targets);
    binding.plan = { ...binding.plan, restartRequired };
    binding.memory = projectionFor(binding, snapshot, targets, restartRequired);
    binding.renderMode = applyRenderState(binding, snapshot);
    syncFluidControls(binding, { _targets: targets });
    return binding;
}

export function applyFluidControls(binding: FluidControlsBinding, values: Readonly<FluidControlValues>, changedKeys?: readonly ControlKey[]): FluidControlsApplicationPlan {
    if (binding.disposed) {
        throw new Error("[fluid] cannot apply a disposed controls binding.");
    }
    if (binding._runtime.applying) {
        throw new Error("[fluid] controls binding application is not reentrant.");
    }
    const runtime = binding._runtime;
    let next = normalizeFluidControls(values, {
        maxParticleCount: runtime.options.maxParticleCount,
        capabilities: effectiveCapabilities(runtime.options),
        timestampQuerySupported: runtime.options.timestampQuerySupported,
    });
    let plan = deriveFluidControlsApplicationPlan(binding.snapshot, next, changedKeys);
    let hostChangedKeys = changedKeys ? [...changedKeys] : [...plan.changedKeys];
    const previousSnapshot = binding.snapshot;
    const previousRenderMode = binding.renderMode;
    const previousMemory = binding.memory;
    const previousPageDiagnostics = binding.pageDiagnostics;
    const previousCapabilities = binding.capabilities;
    const renderStateChanged = plan.renderProfile || plan.renderMode || plan.foamRender || plan.debugRender;
    const hostState = runtime.options.captureHostState?.();
    let prepared: PreparedFluidReconfiguration | PreparedFluidCollectionReconfiguration | null = null;
    let finalized = false;
    runtime.applying = true;
    try {
        runtime.options.applyHostState?.(next, hostChangedKeys);
        let targets = targetSimulations(runtime.options.target).map((simulation) => runtime.options.resolveTarget(simulation, next));
        const boundedSnapshot = clampPageCapacity(binding, next, targets);
        if (boundedSnapshot !== next) {
            next = boundedSnapshot;
            plan = deriveFluidControlsApplicationPlan(binding.snapshot, next, changedKeys);
            if (!hostChangedKeys.includes("pagedGridMaxPages")) {
                hostChangedKeys = [...hostChangedKeys, "pagedGridMaxPages"];
            }
            runtime.options.applyHostState?.(next, hostChangedKeys);
            targets = targetSimulations(runtime.options.target).map((simulation) => runtime.options.resolveTarget(simulation, next));
        }
        if (targets.some((target) => backendChanged(target.simulation.options.backend, target.options.backend))) {
            plan = { ...plan, reconfigure: true, restartRequired: false };
        }
        if (plan.reconfigure && targets.length > 0) {
            if ("simulations" in runtime.options.target) {
                const requests: FluidSimulationReconfigurationRequest[] = targets.map((target) => ({
                    simulation: target.simulation,
                    updates: commonSimulationOptions(binding, target, next),
                    preserveState: preserveReconfiguredState(target),
                }));
                prepared = prepareFluidCollectionReconfiguration(requests);
            } else {
                const target = targets[0]!;
                prepared = prepareFluidReconfiguration(target.simulation, commonSimulationOptions(binding, target, next), preserveReconfiguredState(target));
            }
        }
        const projectedRestart = !plan.reconfigure && hasPendingRestart(binding, next, targets);
        const nextMemory = projectionFor(binding, next, targets, projectedRestart);
        const nextRenderMode = renderStateChanged ? applyRenderState(binding, next) : previousRenderMode;
        if (prepared) {
            if ("reconfigurations" in prepared) {
                commitFluidCollectionReconfiguration(prepared);
            } else {
                commitFluidReconfiguration(prepared);
            }
        } else {
            applyLiveSimulationState(targets, previousSnapshot, next);
        }
        finalized = true;
        binding.snapshot = next;
        binding.plan = { ...plan, restartRequired: hasPendingRestart(binding, next, targets) };
        binding.renderMode = nextRenderMode;
        binding.memory = nextMemory;
        binding.capabilities = resolveFluidControlsCapabilities({
            method: next.method,
            timestampQuerySupported: runtime.options.timestampQuerySupported,
            hostCapabilities: effectiveCapabilities(runtime.options),
        });
        syncFluidControls(binding, { _targets: targets });
        return binding.plan;
    } catch (error) {
        if (finalized) {
            throw error;
        }
        if (prepared) {
            if ("reconfigurations" in prepared) {
                cancelFluidCollectionReconfiguration(prepared);
            } else {
                cancelFluidReconfiguration(prepared);
            }
        }
        if (runtime.options.captureHostState && runtime.options.restoreHostState) {
            runtime.options.restoreHostState(hostState);
        }
        binding.snapshot = previousSnapshot;
        binding.renderMode = previousRenderMode;
        binding.memory = previousMemory;
        binding.pageDiagnostics = previousPageDiagnostics;
        binding.capabilities = previousCapabilities;
        if (renderStateChanged) {
            applyRenderState(binding, previousSnapshot);
        }
        throw error;
    } finally {
        runtime.applying = false;
    }
}

export function applyFluidGridSettings(
    binding: FluidControlsBinding,
    position: readonly [number, number, number],
    size: readonly [number, number, number]
): FluidControlsApplicationPlan {
    return applyFluidControls(
        binding,
        {
            ...binding.snapshot,
            gridPosition: [...position],
            gridSize: [...size],
        },
        ["gridPosition", "gridSize"]
    );
}

export function syncFluidControls(binding: FluidControlsBinding, input: SyncFluidControlsInput = {}): void {
    if (binding.disposed) {
        return;
    }
    binding.capabilities = resolveFluidControlsCapabilities({
        method: binding.snapshot.method,
        timestampQuerySupported: binding._runtime.options.timestampQuerySupported,
        hostCapabilities: effectiveCapabilities(binding._runtime.options),
    });
    if (input.pageDiagnostics) {
        publishPageDiagnostics(binding, input.pageDiagnostics.method, input.pageDiagnostics.requiredPages, input.pageDiagnostics.capacity, input.pageDiagnostics.overflow ?? false);
    }
    const controls = binding._runtime.options.controls;
    const target = binding._runtime.options.target;
    const simulations = targetSimulations(target);
    const previousSyncCache = binding._syncCache;
    let syncCache = binding._syncCache;
    if (!input.pageDiagnostics) {
        if (input._targets !== undefined || syncCache === null || syncCache.snapshot !== binding.snapshot || !sameTargetSimulations(syncCache.restartTargets, simulations)) {
            const resolvedTargets = input._targets ?? simulations.map((simulation) => binding._runtime.options.resolveTarget(simulation, binding.snapshot));
            syncCache = refreshSyncProjection(binding, resolvedTargets, simulations);
        } else {
            const restartRequired = hasPendingRestartTargets(binding.snapshot, syncCache.restartTargets);
            if (restartRequired !== binding.plan.restartRequired) {
                if (restartRequired) {
                    const resolvedTargets = simulations.map((simulation) => binding._runtime.options.resolveTarget(simulation, binding.snapshot));
                    syncCache = refreshSyncProjection(binding, resolvedTargets, simulations);
                } else {
                    binding.plan = { ...binding.plan, restartRequired: false };
                }
            }
        }
    }
    const diagnostics = "simulations" in target ? getFluidSimulationCollectionDiagnostics(target) : getFluidSimulationDiagnostics(target);
    controls.setParticleUsage(
        diagnostics.activeCount,
        diagnostics.count,
        diagnostics.gpuBytes,
        binding.plan.restartRequired ? binding.memory.restartActiveCount : undefined,
        binding.plan.restartRequired ? binding.memory.restartParticleCount : undefined,
        binding.plan.restartRequired ? binding.memory.steadyBytes : undefined,
        binding.plan.restartRequired ? binding.memory.restartActiveCountEstimated : undefined
    );
    controls.setFoamParticleCounts(diagnostics.diffuse ?? undefined, binding.snapshot.foam.enabled, diagnostics.diffuse?.capacity);
    controls.setPressureDiagnostics(diagnostics.pressure ?? undefined);
    const polygonCount = "polygons" in diagnostics ? diagnostics.polygons.reduce((sum, polygon) => sum + (polygon.triangleCount ?? 0), 0) : diagnostics.polygon?.triangleCount;
    controls.setPolygonTriangleCount(polygonCount, binding.snapshot.method === "FLIP");
    if (binding.snapshot.method === "FLIP" || binding.snapshot.method === "MLS-MPM") {
        const method = binding.snapshot.method;
        const livePageLimit =
            syncCache?.snapshot === binding.snapshot
                ? syncCache.pageCapacityLimit
                : simulations.reduce((limit, simulation) => {
                      const dim = simulation.resolvedConfig.gridDim;
                      return dim ? Math.min(limit, fluidMaximumPageCapacity(method, dim, binding._runtime.options.deviceLimits)) : limit;
                  }, Number.MAX_SAFE_INTEGER);
        const pageLimit = livePageLimit === Number.MAX_SAFE_INTEGER ? binding.snapshot.pagedGridMaxPages : livePageLimit;
        if (previousSyncCache?.snapshot !== binding.snapshot || previousSyncCache?.pageCapacityLimit !== pageLimit) {
            controls.setPagedGrid(binding.snapshot.pagedGrid);
            controls.setPagedGridMaxPages(binding.snapshot.pagedGridMaxPages, pageLimit);
        }
    }
    if (binding.pageDiagnostics) {
        controls.setPagedGridStatus(binding.pageDiagnostics.text, binding.pageDiagnostics.overflow);
    }
}

export function disposeFluidControlsBinding(binding: FluidControlsBinding): void {
    if (binding.disposed) {
        return;
    }
    binding.disposed = true;
    binding.pageDiagnostics = null;
    binding._syncCache = null;
}
