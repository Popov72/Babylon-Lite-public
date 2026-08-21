// Parsing of a fluid-simulation setting file (`lab/public/aquanova/fluidSim/<name>.json`).
//
// These files are the SAME shape the Liquefactor demo's "Export parameters" writes, which is what
// lets a prop be tuned in Liquefactor and behave identically here. Every block the file carries is
// parsed and carried on the setting — nothing is silently dropped at load, because a value that is
// read in one demo and ignored in the other is exactly how the two drift apart.
// A parsed fluid-simulation setting (a liquefactor/fluid export JSON): the solver method plus the raw,
// method-specific physics record and, for PB-MPM, the material enum. Consumed by buildFluidSim below.
export interface FluidSimSetting {
    method: string; // "MLS-MPM" | "PB-MPM"
    physics: Record<string, number>;
    material?: number;
    useMeshColors?: boolean; // demoParams.useMeshColors — tint the water by the liquefied mesh's texture
    particleRadius?: number; // demoParams.particleRadius — volume-sampling spacing (drives the particle count)
    /** The file's `render` block (water colour, absorption, blur, surface filter…). The surface pass is
     *  shared by every active blob, so this is applied globally by the shot that installs it. */
    render?: FluidRenderSetting;
    /** The file's `foam` block. Parsed and kept for a future foam pass; nothing consumes it yet. */
    foam?: FluidFoamSetting;
    /** The file's `impulse` block — the eruption burst applied when the melt completes. */
    impulse?: FluidImpulseSetting;
    /** The file's `grid` block — an explicit simulation-domain size, overriding the automatic one. */
    grid?: FluidGridSetting;
}
/** Explicit simulation-domain size in world units, as a FULL extent per axis (not a half-width). The
 *  box is centred on the sampled prop in X/Z and rests on the floor in Y. Any axis at 0/absent falls
 *  back to the automatic size derived from the prop's footprint. */
export interface FluidGridSetting {
    x?: number;
    y?: number;
    z?: number;
}
/** Liquefaction hand-off impulse. `direction` is normalised before use, so its length carries no
 *  meaning; `intensity` is the only magnitude. Absent leaves the demo defaults (straight up, 1×). */
export interface FluidImpulseSetting {
    intensity?: number;
    direction?: [number, number, number];
    /** Blast-sphere radius in world units around the impact point. 0/absent = derive it per mesh from
     *  the sampled volume, so the falloff scales with the prop rather than being a fixed distance. */
    radius?: number;
}
/** The subset of a setting file's `render` block the surface pass understands, matching the controls
 *  Liquefactor exposes. Anything absent leaves the current value alone. */
export interface FluidRenderSetting {
    waterColor?: string;
    absorption?: number;
    particleSize?: number;
    refractionStrength?: number;
    specularPower?: number;
    surfaceDepthBlur?: number;
    depthBlurEdgeThreshold?: number;
    surfaceThicknessBlur?: number;
    halfRendering?: boolean;
    thicknessDownscale?: number;
    surfaceFilter?: string;
    narrowRangeDelta?: number;
    narrowRangeMu?: number;
}
/** A setting file's `foam` block, parsed and carried on the setting so it is available the moment a
 *  foam pass is wired up. Nothing renders foam in this demo yet, so these values are currently only
 *  reported through the `fluidSetting()` QA hook — they are NOT silently dropped at load. */
export interface FluidFoamSetting {
    enableFoam?: boolean;
    trappedAirRate?: number;
    waveCrestRate?: number;
    foamLifetime?: number;
    foamLifetimeMin?: number;
    bubbleBuoyancy?: number;
    bubbleDrag?: number;
    poolSize?: number;
    foamSoftness?: number;
    foamDensity?: number;
    subsurfaceBubbleStrength?: number;
    subsurfaceBubbleColor?: string;
    foamBlurRadius?: number;
    foamLightIntensity?: number;
    foamAmbient?: number;
    foamAO?: number;
    foamNormalStrength?: number;
    foamDebug?: string;
    foamSize?: number;
}
/** "#16a3c3" → [r, g, b] in 0..1. Returns null for anything that is not a 6-digit hex colour. */
export function hexToRgb(hex: string | undefined): [number, number, number] | null {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex ?? "");
    if (!m) return null;
    const n = parseInt(m[1]!, 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
export async function fetchFluidSetting(name: string): Promise<FluidSimSetting | undefined> {
    if (!name || /\.json$/i.test(name)) {
        throw new Error(`[aquanova] fluidSim name "${name}" must omit the .json extension`);
    }
    try {
        const res = await fetch(`/aquanova/fluidSim/${name}.json`);
        if (!res.ok) {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] fluidSim "${name}": HTTP ${res.status} — falling back to the default water`);
            return undefined;
        }
        const j = (await res.json()) as {
            meta?: { method?: string };
            physics?: Record<string, number>;
            material?: number;
            demoParams?: { useMeshColors?: number; particleRadius?: number };
            render?: FluidRenderSetting;
            foam?: FluidFoamSetting;
            impulse?: FluidImpulseSetting;
            grid?: FluidGridSetting;
        };
        if (!j.physics || !j.meta?.method) {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] fluidSim "${name}": missing meta.method or physics — falling back to the default water`);
            return undefined;
        }
        return {
            method: j.meta.method,
            physics: j.physics,
            material: j.material,
            useMeshColors: !!j.demoParams?.useMeshColors,
            particleRadius: j.demoParams?.particleRadius,
            render: j.render,
            foam: j.foam,
            impulse: j.impulse,
            grid: j.grid,
        };
    } catch (err) {
        // A dev-server SPA fallback answers 200 with HTML, so this is the path a wrong name takes.
        // eslint-disable-next-line no-console
        console.warn(`[aquanova] fluidSim "${name}" failed to load`, err);
        return undefined;
    }
}
