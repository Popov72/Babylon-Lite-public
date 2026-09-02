// Round-trippable serialisation for a (demo, method, quality) fluid preset.
//
// The on-disk quality presets (scenes/../presets/*.json, loaded by
// quality-presets.ts) and the panel's "Export parameters" download share ONE
// human-facing shape: grouped physics / render / foam, with UI-label-derived
// keys. `exportJsonFromPairState` writes that shape from the app's internal
// PairState; `presetFromExportJson` is its exact inverse, turning a loaded file
// back into a Partial<PairState> the core merges over its defaults. Keeping both
// directions here guarantees Export → drop-in file → import round-trips cleanly.

import type { FluidEmitter, FluidSink } from "../core/sim-common.js";
import type { DemoStateValue, FluidDomainBounds, PairState } from "./authoring-state.js";
import { FLIP_DEFAULT_MARKERS_PER_CELL, cellSizeForPhysicsScale, gridPositionForBounds, gridResolutionForScale, gridSizeForBounds, gridWorldSize } from "./grid-settings.js";
import { CURRENT_FLUID_SIMULATION_SEMANTICS, resolveFluidSimulationSemantics, type FluidSimulationSemantics } from "../core/simulation-config.js";

type KnownPresetShape = true | { readonly [key: string]: KnownPresetShape };

// `true` means the corresponding value is already carried by PairState. An object recursively lists
// modeled children; omitted children are retained in forwardCompatibleFields. Empty objects mark
// shared-format sections that this mapper does not own but must preserve for another host.
const KNOWN_EXPORT_SHAPE: KnownPresetShape = {
    formatVersion: true,
    simulationSemantics: true,
    meta: { demo: true, method: true },
    source: {},
    physics: true,
    demoParams: true,
    demoState: true,
    simulationDuration: true,
    alphaDecay: true,
    simulationTimeScale: true,
    emitters: true,
    sinks: true,
    initialEmittersFillCapacity: true,
    showContainer: true,
    envIntensity: true,
    msaa: true,
    activeBlocks: true,
    pagedGrid: true,
    pagedGridMaxPages: true,
    fusedBlockDiscovery: true,
    physicsParticleSize: true,
    gridPosition: true,
    gridSize: true,
    gridCells: {},
    domain: {},
    gridResolution: true,
    markersPerCell: true,
    showGridBounds: true,
    showGridBoundsSolid: true,
    particleCount: true,
    material: true,
    camera: { alpha: true, beta: true, radius: true, target: true },
    freeCamera: { position: true, target: true },
    impulse: {},
    grid: {},
    render: {
        independentRendering: true,
        renderAsSpheres: true,
        polygonShader: true,
        waterColor: true,
        absorption: true,
        particleSize: true,
        refractionStrength: true,
        specularPower: true,
        reflectionExposure: true,
        reflectionContrast: true,
        waterReflectivity: true,
        surfaceDepthBlur: true,
        depthBlurEdgeThreshold: true,
        surfaceThicknessBlur: true,
        halfRendering: true,
        thicknessDownscale: true,
        surfaceFilter: true,
        narrowRangeDelta: true,
        narrowRangeMu: true,
        anisotropicSurface: true,
        anisoRadiusDamping: true,
    },
    foam: {
        enableFoam: true,
        activeParticles: true,
        generateSpray: true,
        generateFoam: true,
        generateBubbles: true,
        surfaceFiltering: true,
        trappedAirRate: true,
        waveCrestRate: true,
        turbulenceRate: true,
        energySpeedMin: true,
        energySpeedMax: true,
        curvatureMin: true,
        curvatureMax: true,
        turbulenceMin: true,
        turbulenceMax: true,
        foamLayerDepth: true,
        sprayDrag: true,
        foamLifetime: true,
        foamLifetimeMin: true,
        bubbleBuoyancy: true,
        bubbleDrag: true,
        poolSize: true,
        foamSoftness: true,
        foamDensity: true,
        subsurfaceBubbleStrength: true,
        subsurfaceBubbleColor: true,
        foamBlurRadius: true,
        foamLightIntensity: true,
        foamAmbient: true,
        foamAO: true,
        foamNormalStrength: true,
        foamDebug: true,
        foamSize: true,
    },
    scene: {},
};

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function unknownPresetFields(value: unknown, shape: KnownPresetShape): unknown {
    if (shape === true || !isRecord(value)) {
        return undefined;
    }
    const unknown: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value)) {
        const childShape = shape[key];
        if (childShape === undefined) {
            unknown[key] = structuredClone(child);
            continue;
        }
        const nested = unknownPresetFields(child, childShape);
        if (nested !== undefined) {
            unknown[key] = nested;
        }
    }
    return Object.keys(unknown).length > 0 ? unknown : undefined;
}

function forwardCompatibleFieldsFrom(json: FluidExportJson): Record<string, unknown> | undefined {
    return unknownPresetFields(json, KNOWN_EXPORT_SHAPE) as Record<string, unknown> | undefined;
}

/** Deeply overlay current preset-owned values over preserved data. Objects merge recursively,
 * arrays replace as complete values, and undefined leaves the preserved value unchanged. */
export function mergeFluidPresetData<T>(preserved: unknown, current: T): T {
    if (current === undefined) {
        return structuredClone(preserved) as T;
    }
    if (!isRecord(preserved) || !isRecord(current)) {
        return structuredClone(current);
    }
    const merged = structuredClone(preserved);
    for (const [key, value] of Object.entries(current)) {
        merged[key] = mergeFluidPresetData(merged[key], value);
    }
    return merged as T;
}

function truncateToThreeDecimals(value: number): number {
    return Math.trunc(value * 1000) / 1000;
}

/** The grouped, human-facing JSON shape (matches the "Export parameters" download). */
export interface FluidExportJson {
    formatVersion?: number;
    /** Explicit solver-value interpretation. Required on files written by format 14 and newer. */
    simulationSemantics?: FluidSimulationSemantics;
    meta: { demo: string; method: string };
    /** Original authoring data retained for diagnostics and future mappings. */
    source?: {
        application: string;
        version?: string;
        settings?: Record<string, unknown>;
    };
    physics: Record<string, number>;
    demoParams: Record<string, number>;
    demoState: Record<string, DemoStateValue>;
    /** Simulated seconds before fading begins. Zero runs indefinitely. */
    simulationDuration?: number;
    /** Seconds taken to fade fluid opacity to zero. */
    alphaDecay?: number;
    /** Multiplier applied to real frame time before stepping the simulation. */
    simulationTimeScale?: number;
    /** Solver-independent flow authoring data. Optional for legacy presets. */
    emitters?: FluidEmitter[];
    sinks?: FluidSink[];
    /** Fill the full particle capacity from initial emitters even when inflows exist. */
    initialEmittersFillCapacity?: boolean;
    showContainer: boolean;
    /** Image-based-lighting multiplier ("Environment intensity"), default 1. Optional so files
     *  written before it existed still load. */
    envIntensity?: number;
    /** 4× MSAA on the scene pass ("Anti-aliasing"), default false. Optional for the same reason. */
    msaa?: boolean;
    /** MLS-MPM sparse active-block execution. Optional; defaults off. */
    activeBlocks?: boolean;
    /** MLS-MPM bounded sparse grid-page storage. Optional; defaults off. */
    pagedGrid?: boolean;
    /** Maximum live 4³-cell grid pages. */
    pagedGridMaxPages?: number;
    /** MLS-MPM histogram-integrated active-block discovery. Optional; defaults off. */
    fusedBlockDiscovery?: boolean;
    /** Legacy/non-FLIP particle-size scale. FLIP derives discretization from gridResolution. */
    physicsParticleSize?: number;
    /** World-space center and exact world-space dimensions of the simulation grid. */
    gridPosition?: [number, number, number];
    gridSize?: [number, number, number];
    /** Legacy format-4 exact allocated cell counts. */
    gridCells?: [number, number, number];
    /** Legacy format v3-or-earlier simulation-domain AABB. */
    domain?: FluidDomainBounds;
    /** FLIP grid divisions along the longest domain axis; legacy formats v3-or-earlier also used this field. */
    gridResolution?: number;
    /** FLIP marker sampling density. Defaults to a 2 x 2 x 2 sub-cell layout. */
    markersPerCell?: number;
    /** Display the active solver domain wireframe. */
    showGridBounds?: boolean;
    /** Display transparent depth-tested faces instead of only domain wireframe edges. */
    showGridBoundsSolid?: boolean;
    particleCount: number;
    /** PB-MPM material enum: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. */
    material?: number;
    /** Optional ArcRotate camera framing — omitted for pure-default pairs that pin no viewpoint. */
    camera?: { alpha: number; beta: number; radius: number; target?: [number, number, number] };
    /** Optional FreeCamera pose, stored separately from the incompatible ArcRotate representation. */
    freeCamera?: { position: [number, number, number]; target: [number, number, number] };
    /**
     * Liquefaction hand-off impulse — the burst applied when a melted prop becomes fluid.
     *
     * Optional and demo-specific: the fluid demo has no impulse and omits it, and files written
     * before this existed fall back to each demo's defaults. `direction` is normalised at use, so
     * any non-zero vector is fine and its length carries no meaning — `intensity` is the only
     * magnitude. `radius` is the blast sphere around the impact point in world units; 0 (or absent)
     * keeps the automatic per-mesh radius that engulfs the mesh's own volume. Both Liquefactor and
     * Aquanova read this, which is what keeps a prop auditioned in one demo behaving the same in the
     * other.
     */
    impulse?: { intensity: number; direction: [number, number, number]; radius?: number };
    /**
     * Simulation domain size in world units — the FULL extent, not a half-width.
     *
     * `position` is optional for compatibility with older Liquefactor/Aquanova files, which centred
     * the grid automatically. Pinning the domain matters for cross-demo parity: the wall is a hard
     * boundary, so the same prop in a small grid and a large grid settles into visibly different
     * shapes even with identical physics.
     */
    grid?: { x: number; y: number; z: number; position?: [number, number, number] };
    render: {
        /** Opt into a profile-specific surface pass when combined with other simulations. */
        independentRendering?: boolean;
        renderAsSpheres: boolean;
        /** FLIP polygon shading model. Optional for presets written before the ocean mode. */
        polygonShader?: "physical" | "ocean";
        waterColor: string;
        absorption: number;
        particleSize: number;
        refractionStrength: number;
        specularPower: number;
        /** Environment-reflection tonemap and the water's head-on reflectance. Optional so files
         *  written before these were exposed still load on the shader's own defaults. */
        reflectionExposure?: number;
        reflectionContrast?: number;
        waterReflectivity?: number;
        surfaceDepthBlur: number;
        depthBlurEdgeThreshold: number;
        surfaceThicknessBlur: number;
        halfRendering: boolean;
        thicknessDownscale: number;
        surfaceFilter: "bilateral" | "narrowRange";
        narrowRangeDelta: number;
        narrowRangeMu: number;
        anisotropicSurface: boolean;
        /** Anisotropic WPCA radius damping (0..1). Optional for backward compatibility. */
        anisoRadiusDamping?: number;
    };
    foam: {
        enableFoam: boolean;
        activeParticles?: boolean;
        generateSpray?: boolean;
        generateFoam?: boolean;
        generateBubbles?: boolean;
        /** Optional for backward compatibility; legacy FLIP files default on, other methods off. */
        surfaceFiltering?: boolean;
        trappedAirRate: number;
        waveCrestRate: number;
        turbulenceRate?: number;
        energySpeedMin?: number;
        energySpeedMax?: number;
        curvatureMin?: number;
        curvatureMax?: number;
        turbulenceMin?: number;
        turbulenceMax?: number;
        foamLayerDepth?: number;
        sprayDrag?: number;
        foamLifetime: number;
        foamLifetimeMin: number;
        bubbleBuoyancy: number;
        bubbleDrag: number;
        poolSize: number;
        foamSoftness: number;
        foamDensity: number;
        subsurfaceBubbleStrength: number;
        subsurfaceBubbleColor: string;
        foamBlurRadius: number;
        foamLightIntensity: number;
        foamAmbient: number;
        foamAO: number;
        foamNormalStrength: number;
        foamDebug: string;
        foamSize: number;
    };
    /** Self-contained imported scene payload. Omitted by parameter-only presets. */
    scene?: {
        encoding: "base64";
        glb: string;
        collision: string;
        /** Grid position at which the immutable GLB and collision coordinates were authored. */
        anchorPosition?: [number, number, number];
    };
}

/** Serialise a full PairState (from a preset/default or the live UI) into the
 *  grouped export shape. Camera is written only when the state carries one. */
export function exportJsonFromPairState(demo: string, method: string, ps: PairState): FluidExportJson {
    const f = ps.foam;
    const current: FluidExportJson = {
        formatVersion: 14,
        simulationSemantics: ps.simulationSemantics ?? CURRENT_FLUID_SIMULATION_SEMANTICS,
        meta: { demo, method },
        physics: { ...ps.schema },
        demoParams: { ...ps.demoParams },
        demoState: ps.demoState ? { ...ps.demoState } : {},
        simulationDuration: ps.simulationDuration ?? 0,
        alphaDecay: ps.alphaDecay ?? 2,
        simulationTimeScale: ps.simulationTimeScale ?? 1,
        emitters: structuredClone(ps.emitters ?? []),
        sinks: structuredClone(ps.sinks ?? []).map((sink) => ({ ...sink, mode: sink.mode ?? "delete" })),
        ...(ps.initialEmittersFillCapacity !== undefined ? { initialEmittersFillCapacity: ps.initialEmittersFillCapacity } : {}),
        showContainer: ps.showContainer ?? true,
        ...(ps.envIntensity !== undefined ? { envIntensity: ps.envIntensity } : {}),
        ...(ps.msaa !== undefined ? { msaa: ps.msaa } : {}),
        ...(ps.activeBlocks !== undefined ? { activeBlocks: ps.activeBlocks } : {}),
        ...(ps.pagedGrid !== undefined ? { pagedGrid: ps.pagedGrid } : {}),
        ...(ps.pagedGridMaxPages !== undefined ? { pagedGridMaxPages: ps.pagedGridMaxPages } : {}),
        ...(ps.fusedBlockDiscovery !== undefined ? { fusedBlockDiscovery: ps.fusedBlockDiscovery } : {}),
        ...(method !== "FLIP" ? { physicsParticleSize: ps.physScale } : {}),
        ...(ps.grid ? { gridPosition: [...ps.grid.position], gridSize: [...ps.grid.size] } : {}),
        ...(method === "FLIP"
            ? {
                  gridResolution: ps.gridResolution ?? 160,
                  markersPerCell: ps.markersPerCell ?? FLIP_DEFAULT_MARKERS_PER_CELL,
              }
            : {}),
        showGridBounds: ps.showGridBounds ?? false,
        showGridBoundsSolid: ps.showGridBoundsSolid ?? false,
        particleCount: ps.count,
        ...(ps.material !== undefined ? { material: ps.material } : {}),
        ...(ps.camera ? { camera: { ...ps.camera } } : {}),
        ...(ps.freeCamera ? { freeCamera: { position: [...ps.freeCamera.position], target: [...ps.freeCamera.target] } } : {}),
        render: {
            ...(ps.independentRendering !== undefined ? { independentRendering: ps.independentRendering } : {}),
            renderAsSpheres: ps.renderMode === "spheres",
            ...(ps.polygonShader !== undefined ? { polygonShader: ps.polygonShader } : {}),
            waterColor: ps.color,
            absorption: ps.absorption,
            particleSize: ps.size,
            refractionStrength: ps.refraction ?? 0,
            specularPower: ps.specular ?? 0,
            ...(ps.reflectionExposure !== undefined ? { reflectionExposure: ps.reflectionExposure } : {}),
            ...(ps.reflectionContrast !== undefined ? { reflectionContrast: ps.reflectionContrast } : {}),
            ...(ps.reflectivity !== undefined ? { waterReflectivity: ps.reflectivity } : {}),
            surfaceDepthBlur: ps.depthBlur ?? 0,
            depthBlurEdgeThreshold: ps.depthBlurThreshold ?? 0,
            surfaceThicknessBlur: ps.thicknessBlur ?? 0,
            halfRendering: ps.half,
            thicknessDownscale: ps.thicknessDownscale,
            surfaceFilter: ps.surfaceFilter ?? "bilateral",
            narrowRangeDelta: ps.narrowDelta ?? 0,
            narrowRangeMu: ps.narrowMu ?? 0,
            anisotropicSurface: ps.anisotropic ?? false,
            anisoRadiusDamping: ps.anisoSurfScale ?? 0.5,
        },
        foam: {
            enableFoam: f?.enabled ?? false,
            activeParticles: f?.activeParticles ?? true,
            generateSpray: f?.generateSpray ?? true,
            generateFoam: f?.generateFoam ?? true,
            generateBubbles: f?.generateBubbles ?? true,
            surfaceFiltering: f?.surfaceFiltering ?? method === "FLIP",
            trappedAirRate: f?.kTa ?? 0,
            waveCrestRate: f?.kWc ?? 0,
            turbulenceRate: f?.kTurb ?? 0,
            energySpeedMin: f?.energySpeedMin ?? Math.sqrt(0.5),
            energySpeedMax: f?.energySpeedMax ?? Math.sqrt(40),
            curvatureMin: f?.curvatureMin ?? 0.05,
            curvatureMax: f?.curvatureMax ?? 1.5,
            turbulenceMin: f?.turbulenceMin ?? 0.1,
            turbulenceMax: f?.turbulenceMax ?? 2.5,
            foamLayerDepth: f?.foamLayerDepth ?? 0,
            sprayDrag: f?.sprayDrag ?? 0,
            foamLifetime: f?.tMax ?? 0,
            foamLifetimeMin: f?.tMin ?? 0,
            bubbleBuoyancy: f?.kb ?? 0,
            bubbleDrag: f?.kd ?? 0,
            poolSize: f?.poolScale ?? 0,
            foamSoftness: f?.softness ?? 0,
            foamDensity: f?.density ?? 0,
            subsurfaceBubbleStrength: f?.subsurfaceStrength ?? 0,
            subsurfaceBubbleColor: f?.subsurfaceColor ?? "#b8d1f2",
            foamBlurRadius: f?.blurRadius ?? 0,
            foamLightIntensity: f?.lightIntensity ?? 0,
            foamAmbient: f?.ambient ?? 0,
            foamAO: f?.aoStrength ?? 0,
            foamNormalStrength: f?.normalStrength ?? 0,
            foamDebug: f?.debugTexture && f.debugTexture !== "none" ? f.debugTexture : "off",
            foamSize: f?.size ?? 1,
        },
    };
    return mergeFluidPresetData(ps.forwardCompatibleFields, current);
}

/** Inverse of {@link exportJsonFromPairState}: turn a loaded quality file back into a
 *  Partial<PairState> for the core to merge over its per-method defaults. */
export function presetFromExportJson(j: FluidExportJson): Partial<PairState> {
    const r = j.render;
    const fm = j.foam;
    const forwardCompatibleFields = forwardCompatibleFieldsFrom(j);
    const hasGridDefinition = j.gridPosition !== undefined || j.gridSize !== undefined || j.gridCells !== undefined || j.domain !== undefined || j.gridResolution !== undefined;
    const meshScale = j.demoParams.meshScale ?? 1;
    const legacyParticleScale = hasGridDefinition && (j.formatVersion ?? 0) < 3 && j.meta.demo === "marbleTower" ? meshScale : 1;
    const legacyPhysScale = (j.physicsParticleSize ?? 1) * legacyParticleScale;
    const legacyBounds: FluidDomainBounds =
        j.domain ??
        (j.meta.method === "PBF" || j.meta.method === "FLIP"
            ? { min: [-20 * meshScale, 0, -20 * meshScale], max: [20 * meshScale, 20 * meshScale, 20 * meshScale] }
            : { min: [-20 * meshScale, -1 * meshScale, -20 * meshScale], max: [20 * meshScale, 20 * meshScale, 20 * meshScale] });
    const gridPosition: [number, number, number] = j.gridPosition ? [...j.gridPosition] : gridPositionForBounds(legacyBounds);
    const legacyCellSize = cellSizeForPhysicsScale(j.meta.method, legacyPhysScale);
    const gridSize: [number, number, number] = j.gridSize ? [...j.gridSize] : j.gridCells ? gridWorldSize(j.gridCells, legacyCellSize) : gridSizeForBounds(legacyBounds);
    const currentFlipResolution = j.meta.method === "FLIP" && (j.formatVersion ?? 0) >= 10 && j.gridResolution !== undefined;
    const physScale = currentFlipResolution ? 1 : legacyPhysScale;
    const localizeFlow = <T extends FluidEmitter | FluidSink>(objects: T[] | undefined): T[] | undefined => {
        if (!objects) {
            return undefined;
        }
        const copies = structuredClone(objects);
        if ((j.formatVersion ?? 0) < 4) {
            for (const object of copies) {
                object.transform.position = [
                    object.transform.position[0] - gridPosition[0],
                    object.transform.position[1] - gridPosition[1],
                    object.transform.position[2] - gridPosition[2],
                ];
            }
        }
        return copies;
    };
    const emitters = localizeFlow(j.emitters);
    const sinks = localizeFlow(j.sinks)?.map((sink) => ({
        ...sink,
        mode: sink.mode ?? ((j.formatVersion ?? 0) <= 6 ? "recycle" : "delete"),
    }));
    const schema = { ...j.physics };
    if (schema.gravity !== undefined) {
        schema.gravity = truncateToThreeDecimals(schema.gravity);
    }
    if (schema.scorr !== undefined) {
        schema.scorr = truncateToThreeDecimals(schema.scorr);
    }
    return {
        simulationSemantics: resolveFluidSimulationSemantics({
            formatVersion: j.formatVersion,
            demo: j.meta.demo,
            sourceApplication: j.source?.application,
            explicit: j.simulationSemantics,
        }),
        schema,
        demoParams: { ...j.demoParams },
        simulationDuration: j.simulationDuration ?? 0,
        alphaDecay: j.alphaDecay ?? 2,
        simulationTimeScale: j.simulationTimeScale ?? 1,
        ...(emitters ? { emitters } : {}),
        ...(sinks ? { sinks } : {}),
        ...(j.initialEmittersFillCapacity !== undefined ? { initialEmittersFillCapacity: j.initialEmittersFillCapacity } : {}),
        legacyFlow: j.emitters === undefined && j.sinks === undefined,
        color: r.waterColor,
        half: r.halfRendering,
        thicknessDownscale: r.thicknessDownscale,
        absorption: r.absorption,
        size: r.particleSize,
        physScale,
        ...(j.meta.method === "FLIP"
            ? {
                  gridResolution: currentFlipResolution ? j.gridResolution : gridResolutionForScale("FLIP", physScale, Math.max(...gridSize)),
                  markersPerCell: j.markersPerCell ?? FLIP_DEFAULT_MARKERS_PER_CELL,
              }
            : {}),
        ...(hasGridDefinition ? { grid: { position: gridPosition, size: gridSize } } : {}),
        showGridBounds: j.showGridBounds ?? false,
        showGridBoundsSolid: j.showGridBoundsSolid ?? false,
        count: j.particleCount,
        material: j.material,
        ...(j.camera ? { camera: { ...j.camera } } : {}),
        ...(j.freeCamera ? { freeCamera: { position: [...j.freeCamera.position], target: [...j.freeCamera.target] } } : {}),
        ...(r.independentRendering !== undefined ? { independentRendering: r.independentRendering } : {}),
        renderMode: r.renderAsSpheres ? "spheres" : "surface",
        ...(r.polygonShader !== undefined ? { polygonShader: r.polygonShader } : {}),
        refraction: r.refractionStrength,
        specular: r.specularPower,
        ...(r.reflectionExposure !== undefined ? { reflectionExposure: r.reflectionExposure } : {}),
        ...(r.reflectionContrast !== undefined ? { reflectionContrast: r.reflectionContrast } : {}),
        ...(r.waterReflectivity !== undefined ? { reflectivity: r.waterReflectivity } : {}),
        depthBlur: r.surfaceDepthBlur,
        depthBlurThreshold: r.depthBlurEdgeThreshold,
        thicknessBlur: r.surfaceThicknessBlur,
        surfaceFilter: r.surfaceFilter,
        narrowDelta: r.narrowRangeDelta,
        narrowMu: r.narrowRangeMu,
        anisotropic: r.anisotropicSurface ?? false,
        anisoSurfScale: r.anisoRadiusDamping ?? 0.5,
        foam: {
            enabled: fm.enableFoam,
            activeParticles: fm.activeParticles ?? true,
            generateSpray: fm.generateSpray ?? true,
            generateFoam: fm.generateFoam ?? true,
            generateBubbles: fm.generateBubbles ?? true,
            surfaceFiltering: fm.surfaceFiltering ?? j.meta.method === "FLIP",
            kTa: fm.trappedAirRate,
            kWc: fm.waveCrestRate,
            kTurb: fm.turbulenceRate ?? 0,
            energySpeedMin: fm.energySpeedMin ?? Math.sqrt(0.5),
            energySpeedMax: fm.energySpeedMax ?? Math.sqrt(40),
            curvatureMin: fm.curvatureMin ?? 0.05,
            curvatureMax: fm.curvatureMax ?? 1.5,
            turbulenceMin: fm.turbulenceMin ?? 0.1,
            turbulenceMax: fm.turbulenceMax ?? 2.5,
            foamLayerDepth: fm.foamLayerDepth ?? 0,
            sprayDrag: fm.sprayDrag ?? 0,
            kb: fm.bubbleBuoyancy,
            kd: fm.bubbleDrag,
            tMin: fm.foamLifetimeMin,
            tMax: fm.foamLifetime,
            poolScale: fm.poolSize,
            blurRadius: fm.foamBlurRadius,
            lightIntensity: fm.foamLightIntensity,
            ambient: fm.foamAmbient,
            aoStrength: fm.foamAO,
            normalStrength: fm.foamNormalStrength,
            debugTexture: fm.foamDebug && fm.foamDebug !== "none" ? fm.foamDebug : "off",
            softness: fm.foamSoftness,
            density: fm.foamDensity,
            subsurfaceStrength: fm.subsurfaceBubbleStrength,
            // Presets written before the tint was configurable have no field — fall back to
            // the pale blue that used to be hardcoded so they restore byte-identically.
            subsurfaceColor: fm.subsurfaceBubbleColor ?? "#b8d1f2",
            size: fm.foamSize ?? 1,
        },
        demoState: { ...j.demoState },
        showContainer: j.showContainer,
        ...(j.envIntensity !== undefined ? { envIntensity: j.envIntensity } : {}),
        ...(j.msaa !== undefined ? { msaa: j.msaa } : {}),
        ...(j.activeBlocks !== undefined ? { activeBlocks: j.activeBlocks } : {}),
        ...(j.pagedGrid !== undefined ? { pagedGrid: j.pagedGrid } : {}),
        ...(j.pagedGridMaxPages !== undefined ? { pagedGridMaxPages: j.pagedGridMaxPages } : {}),
        ...(j.fusedBlockDiscovery !== undefined ? { fusedBlockDiscovery: j.fusedBlockDiscovery } : {}),
        ...(forwardCompatibleFields ? { forwardCompatibleFields } : {}),
    };
}
