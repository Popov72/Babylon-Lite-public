export interface FluidSimulationLifecycle {
    opacity: number;
    stopped: boolean;
}

/** Resolve the simulation's visible lifetime from elapsed simulation time. */
export function fluidSimulationLifecycle(elapsed: number, duration: number, alphaDecay: number): FluidSimulationLifecycle {
    const safeElapsed = Math.max(0, elapsed);
    const safeDuration = Math.max(0, duration);
    const safeDecay = Math.max(0, alphaDecay);
    if (safeDuration === 0 || safeElapsed <= safeDuration) {
        return { opacity: 1, stopped: false };
    }
    if (safeDecay === 0 || safeElapsed >= safeDuration + safeDecay) {
        return { opacity: 0, stopped: true };
    }
    return { opacity: 1 - (safeElapsed - safeDuration) / safeDecay, stopped: false };
}
