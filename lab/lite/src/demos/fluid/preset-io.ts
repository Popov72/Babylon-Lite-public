// Round-trippable serialisation for a (demo, method, quality) fluid preset.
//
// The on-disk quality presets (scenes/../presets/*.json, loaded by
// quality-presets.ts) and the panel's "Export parameters" download share ONE
// human-facing shape: grouped physics / render / foam, with UI-label-derived
// keys. `exportJsonFromPairState` writes that shape from the app's internal
// PairState; `presetFromExportJson` is its exact inverse, turning a loaded file
// back into a Partial<PairState> the core merges over its defaults. Keeping both
// directions here guarantees Export → drop-in file → import round-trips cleanly.

import type { PairState } from "./demo.js";

/** The grouped, human-facing JSON shape (matches the "Export parameters" download). */
export interface FluidExportJson {
    meta: { demo: string; method: string };
    physics: Record<string, number>;
    demoParams: Record<string, number>;
    demoState: Record<string, number | boolean>;
    showContainer: boolean;
    physicsParticleSize: number;
    particleCount: number;
    /** PB-MPM material enum: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. */
    material?: number;
    /** Optional camera framing — omitted for pure-default pairs that pin no viewpoint. */
    camera?: { alpha: number; beta: number; radius: number };
    render: {
        renderAsSpheres: boolean;
        waterColor: string;
        absorption: number;
        particleSize: number;
        refractionStrength: number;
        specularPower: number;
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
        meta: { demo, method },
        physics: { ...ps.schema },
        demoParams: { ...ps.demoParams },
        demoState: ps.demoState ? { ...ps.demoState } : {},
        showContainer: ps.showContainer ?? true,
        physicsParticleSize: ps.physScale,
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
    return {
        schema: { ...j.physics },
        demoParams: { ...j.demoParams },
        color: r.waterColor,
        half: r.halfRendering,
        thicknessDownscale: r.thicknessDownscale,
        absorption: r.absorption,
        size: r.particleSize,
        physScale: j.physicsParticleSize,
        count: j.particleCount,
        material: j.material,
        ...(j.camera ? { camera: { ...j.camera } } : {}),
        renderMode: r.renderAsSpheres ? "spheres" : "surface",
        refraction: r.refractionStrength,
        specular: r.specularPower,
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
            size: fm.foamSize ?? 1,
        },
        demoState: { ...j.demoState },
        showContainer: j.showContainer,
    };
}
