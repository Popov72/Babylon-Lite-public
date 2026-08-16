export interface FluidSimulationLifecycle {
    opacity: number;
    stopped: boolean;
}

/** Resolve the simulation's visible lifetime from elapsed simulation time. */
export function fluidSimulationLifecycle(elapsed: number, duration: number, alphaDecay: number): FluidSimulationLifecycle {
    const safeElapsed = Math.max(0, elapsed);
    const safeDuration = Math.max(0, duration);
    const safeDecay = Math.max(0, alphaDecay);
    if (safeDuration === 0 || safeElapsed < safeDuration) {
        return { opacity: 1, stopped: false };
    }
    if (safeDecay === 0 || safeElapsed >= safeDuration + safeDecay) {
        return { opacity: 0, stopped: true };
    }
    return { opacity: 1 - (safeElapsed - safeDuration) / safeDecay, stopped: false };
}

/** Clamp a solver step to the exact finite-lifecycle stop time. */
export function fluidSimulationStepDelta(elapsed: number, requestedDelta: number, duration: number, alphaDecay: number): number {
    const safeDelta = Math.max(0, requestedDelta);
    const safeDuration = Math.max(0, duration);
    if (safeDuration === 0) {
        return safeDelta;
    }
    const stopTime = safeDuration + Math.max(0, alphaDecay);
    return Math.min(safeDelta, Math.max(0, stopTime - Math.max(0, elapsed)));
}

/** Return the authoritative simulated time once a deterministic capture can complete. */
export function fluidCaptureCompletionTime(elapsed: number, targetReached: boolean, lifecycleStopped: boolean): number | null {
    return targetReached || lifecycleStopped ? Math.max(0, elapsed) : null;
}
