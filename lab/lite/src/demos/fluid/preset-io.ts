// Round-trippable serialisation for a (demo, method, quality) fluid preset.
//
// The on-disk quality presets (scenes/../presets/*.json, loaded by
// quality-presets.ts) and the panel's "Export parameters" download share ONE
// human-facing shape: grouped physics / render / foam, with UI-label-derived
// keys. `exportJsonFromPairState` writes that shape from the app's internal
// PairState; `presetFromExportJson` is its exact inverse, turning a loaded file
// back into a Partial<PairState> the core merges over its defaults. Keeping both
// directions here guarantees Export → drop-in file → import round-trips cleanly.

import type { FluidEmitter, FluidSink } from "babylon-lite";
import type { DemoStateValue, FluidDomainBounds, PairState } from "./demo.js";
import { cellSizeForPhysicsScale, gridPositionForBounds, gridSizeForBounds, gridWorldSize } from "./grid-settings.js";

/** The grouped, human-facing JSON shape (matches the "Export parameters" download). */
export interface FluidExportJson {
    formatVersion?: number;
    meta: { demo: string; method: string };
    physics: Record<string, number>;
    demoParams: Record<string, number>;
    demoState: Record<string, DemoStateValue>;
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
    physicsParticleSize: number;
    /** World-space center and exact world-space dimensions of the simulation grid. */
    gridPosition?: [number, number, number];
    gridSize?: [number, number, number];
    /** Legacy format-4 exact allocated cell counts. */
    gridCells?: [number, number, number];
    /** Legacy format <=3 simulation-domain AABB. */
    domain?: FluidDomainBounds;
    /** Legacy format <=3 longest-axis resolution. */
    gridResolution?: number;
    /** Display the active solver domain wireframe. */
    showGridBounds?: boolean;
    particleCount: number;
    /** PB-MPM material enum: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. */
    material?: number;
    /** Optional camera framing — omitted for pure-default pairs that pin no viewpoint. */
    camera?: { alpha: number; beta: number; radius: number };
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
     * The grid is centred on the sampled prop in X/Z and sits on the ground in Y, so `x`/`z` are how
     * far the puddle can spread before it piles against the domain wall and `y` is the headroom above
     * the floor. Any axis left at 0 (or absent) falls back to the demo's automatic size, derived from
     * the prop's own footprint. `y` is a MINIMUM: it is raised when needed to clear the prop, since a
     * box stopping below the prop would leave the seeded particles outside the domain. Pinning this
     * matters for cross-demo parity: the domain wall is a hard boundary, so the same prop in a small
     * grid and a large grid settles into visibly different shapes even with identical physics.
     */
    grid?: { x: number; y: number; z: number };
    render: {
        renderAsSpheres: boolean;
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
        trappedAirRate: number;
        waveCrestRate: number;
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
}

/** Serialise a full PairState (from a preset/default or the live UI) into the
 *  grouped export shape. Camera is written only when the state carries one. */
export function exportJsonFromPairState(demo: string, method: string, ps: PairState): FluidExportJson {
    const f = ps.foam;
    return {
        formatVersion: 5,
        meta: { demo, method },
        physics: { ...ps.schema },
        demoParams: { ...ps.demoParams },
        demoState: ps.demoState ? { ...ps.demoState } : {},
        emitters: structuredClone(ps.emitters ?? []),
        sinks: structuredClone(ps.sinks ?? []),
        ...(ps.initialEmittersFillCapacity !== undefined ? { initialEmittersFillCapacity: ps.initialEmittersFillCapacity } : {}),
        showContainer: ps.showContainer ?? true,
        ...(ps.envIntensity !== undefined ? { envIntensity: ps.envIntensity } : {}),
        ...(ps.msaa !== undefined ? { msaa: ps.msaa } : {}),
        ...(ps.activeBlocks !== undefined ? { activeBlocks: ps.activeBlocks } : {}),
        ...(ps.pagedGrid !== undefined ? { pagedGrid: ps.pagedGrid } : {}),
        ...(ps.pagedGridMaxPages !== undefined ? { pagedGridMaxPages: ps.pagedGridMaxPages } : {}),
        ...(ps.fusedBlockDiscovery !== undefined ? { fusedBlockDiscovery: ps.fusedBlockDiscovery } : {}),
        physicsParticleSize: ps.physScale,
        ...(ps.grid ? { gridPosition: [...ps.grid.position], gridSize: [...ps.grid.size] } : {}),
        showGridBounds: ps.showGridBounds ?? false,
        particleCount: ps.count,
        ...(ps.material !== undefined ? { material: ps.material } : {}),
        ...(ps.camera ? { camera: { ...ps.camera } } : {}),
        render: {
            renderAsSpheres: ps.renderMode === "spheres",
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
            activeParticles: f?.activeParticles ?? false,
            trappedAirRate: f?.kTa ?? 0,
            waveCrestRate: f?.kWc ?? 0,
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
            foamDebug: f?.debugTexture ?? "off",
            foamSize: f?.size ?? 1,
        },
    };
}

/** Inverse of {@link exportJsonFromPairState}: turn a loaded quality file back into a
 *  Partial<PairState> for the core to merge over its per-method defaults. */
export function presetFromExportJson(j: FluidExportJson): Partial<PairState> {
    const r = j.render;
    const fm = j.foam;
    const hasGridDefinition = j.gridPosition !== undefined || j.gridSize !== undefined || j.gridCells !== undefined || j.domain !== undefined || j.gridResolution !== undefined;
    const meshScale = j.demoParams.meshScale ?? 1;
    const legacyParticleScale = hasGridDefinition && (j.formatVersion ?? 0) < 3 && j.meta.demo === "marbleTower" ? meshScale : 1;
    const physScale = j.physicsParticleSize * legacyParticleScale;
    const legacyBounds: FluidDomainBounds =
        j.domain ??
        (j.meta.method === "PBF"
            ? { min: [-20 * meshScale, 0, -20 * meshScale], max: [20 * meshScale, 20 * meshScale, 20 * meshScale] }
            : { min: [-20 * meshScale, -1 * meshScale, -20 * meshScale], max: [20 * meshScale, 20 * meshScale, 20 * meshScale] });
    const gridPosition: [number, number, number] = j.gridPosition ? [...j.gridPosition] : gridPositionForBounds(legacyBounds);
    const cellSize = cellSizeForPhysicsScale(j.meta.method, physScale);
    const gridSize: [number, number, number] = j.gridSize ? [...j.gridSize] : j.gridCells ? gridWorldSize(j.gridCells, cellSize) : gridSizeForBounds(legacyBounds);
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
    const sinks = localizeFlow(j.sinks);
    return {
        schema: { ...j.physics },
        demoParams: { ...j.demoParams },
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
        ...(hasGridDefinition ? { grid: { position: gridPosition, size: gridSize } } : {}),
        showGridBounds: j.showGridBounds ?? false,
        count: j.particleCount,
        material: j.material,
        ...(j.camera ? { camera: { ...j.camera } } : {}),
        renderMode: r.renderAsSpheres ? "spheres" : "surface",
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
            activeParticles: fm.activeParticles ?? false,
            kTa: fm.trappedAirRate,
            kWc: fm.waveCrestRate,
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
            debugTexture: fm.foamDebug,
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
    };
}
