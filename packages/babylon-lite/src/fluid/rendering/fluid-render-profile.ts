import type { FluidParticleColorMode, FluidSurfaceTask } from "./fluid-surface-render.js";

/** Surface-wide settings that must match for simulations to share one reconstruction pass. */
export interface FluidRenderProfileSettings {
    /** Opt into profile-specific rendering instead of the application's shared fast path. */
    independentRendering?: boolean;
    /** Six-digit hexadecimal water color, with or without a leading hash. */
    waterColor?: string;
    polygonShader?: "physical" | "ocean";
    absorption?: number;
    particleSize?: number;
    refractionStrength?: number;
    specularPower?: number;
    reflectionExposure?: number;
    reflectionContrast?: number;
    waterReflectivity?: number;
    surfaceDepthBlur?: number;
    depthBlurEdgeThreshold?: number;
    surfaceThicknessBlur?: number;
    halfRendering?: boolean;
    thicknessDownscale?: number;
    surfaceFilter?: string;
    narrowRangeDelta?: number;
    narrowRangeMu?: number;
    anisotropicSurface?: boolean;
    anisoRadiusDamping?: number;
}

/** Convert a six-digit hexadecimal color to normalized linear components. */
export function fluidRenderHexColor(hex: string | undefined): [number, number, number] | null {
    const match = /^#?([0-9a-f]{6})$/i.exec(hex ?? "");
    if (!match) {
        return null;
    }
    const value = Number.parseInt(match[1]!, 16);
    return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
}

/** Apply every supported, explicitly authored profile setting to a fluid surface task. */
/** @internal */
export function applyFluidRenderProfile(task: FluidSurfaceTask, render: FluidRenderProfileSettings): void {
    const color = fluidRenderHexColor(render.waterColor);
    if (color) {
        task.setFluidColor(color);
    }
    if (render.polygonShader !== undefined) {
        task.setShadingMode(render.polygonShader);
    }
    if (render.absorption !== undefined) {
        task.setAbsorption(render.absorption);
    }
    if (render.particleSize !== undefined) {
        task.setSizeScale(render.particleSize);
    }
    if (render.refractionStrength !== undefined) {
        task.setRefractionStrength(render.refractionStrength);
    }
    if (render.specularPower !== undefined) {
        task.setSpecularPower(render.specularPower);
    }
    if (render.reflectionExposure !== undefined || render.reflectionContrast !== undefined) {
        task.setEnvReflection(render.reflectionExposure ?? 1, render.reflectionContrast ?? 1.1);
    }
    if (render.waterReflectivity !== undefined) {
        task.setFresnelF0(render.waterReflectivity);
    }
    if (render.surfaceDepthBlur !== undefined) {
        task.setDepthBlur(render.surfaceDepthBlur, render.depthBlurEdgeThreshold ?? 0);
    }
    if (render.surfaceThicknessBlur !== undefined) {
        task.setThicknessBlur(render.surfaceThicknessBlur);
    }
    if (render.halfRendering !== undefined) {
        task.setHalfRender(render.halfRendering);
    }
    if (render.thicknessDownscale !== undefined) {
        task.setThicknessDownscale(render.thicknessDownscale);
    }
    if (render.surfaceFilter === "bilateral" || render.surfaceFilter === "narrowRange") {
        task.setSurfaceFilter(render.surfaceFilter);
    }
    if (render.narrowRangeDelta !== undefined || render.narrowRangeMu !== undefined) {
        task.setNarrowRange(render.narrowRangeDelta ?? 10, render.narrowRangeMu ?? 1);
    }
    if (render.anisotropicSurface !== undefined) {
        task.setAnisotropic(render.anisotropicSurface);
    }
    if (render.anisoRadiusDamping !== undefined) {
        task.setAnisotropySurfScale(render.anisoRadiusDamping);
    }
}

/** Stable key for simulations that can share one profile-specific surface pass. */
export function fluidRenderProfileKey(render: FluidRenderProfileSettings, particleRadius: number, surfaceSizeScale: number, particleColorMode: FluidParticleColorMode): string {
    return JSON.stringify([
        fluidRenderHexColor(render.waterColor),
        render.polygonShader ?? null,
        render.absorption ?? null,
        render.particleSize ?? null,
        render.refractionStrength ?? null,
        render.specularPower ?? null,
        render.reflectionExposure ?? null,
        render.reflectionContrast ?? null,
        render.waterReflectivity ?? null,
        render.surfaceDepthBlur ?? null,
        render.depthBlurEdgeThreshold ?? null,
        render.surfaceThicknessBlur ?? null,
        render.halfRendering ?? null,
        render.thicknessDownscale ?? null,
        render.surfaceFilter ?? null,
        render.narrowRangeDelta ?? null,
        render.narrowRangeMu ?? null,
        render.anisotropicSurface ?? null,
        render.anisoRadiusDamping ?? null,
        particleRadius,
        surfaceSizeScale,
        particleColorMode,
    ]);
}
