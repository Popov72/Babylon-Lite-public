import type { FluidExportJson } from "babylon-lite";

function record(value: unknown): Record<string, unknown> | undefined {
    return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function positive(value: unknown, fallback: number, label: string): number {
    if (value === undefined) {
        return fallback;
    }
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
        throw new Error("[FLIP Reference] invalid authored " + label + ".");
    }
    return value;
}

/** Keep physical solver time distinct from the glTF animation's authored timeline rate. */
export function fluidReferenceTiming(preset?: Pick<FluidExportJson, "source" | "simulationTimeScale">): { frameDelta: number; animationRate: number } {
    const settings = preset?.source?.settings;
    const timeline = record(settings?.timeline);
    if (!timeline) {
        return { frameDelta: 1 / 60, animationRate: 1 };
    }
    const renderFps = positive(timeline.fps, 60, "timeline FPS") / positive(timeline.fpsBase, 1, "FPS base");
    const simulation = record(record(settings?.domain)?.simulation);
    const custom = simulation?.frame_rate_mode === "FRAME_RATE_MODE_CUSTOM" ? simulation.frame_rate_custom : undefined;
    const simulationFps = positive(timeline.simulationFps ?? custom, renderFps, "simulation FPS");
    const nativeScale = positive(simulation?.time_scale ?? preset?.simulationTimeScale, 1, "simulation time scale");
    return { frameDelta: 1 / simulationFps, animationRate: simulationFps / (renderFps * nativeScale) };
}
