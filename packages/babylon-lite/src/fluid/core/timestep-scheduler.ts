export interface FluidTimestepDiagnostics {
    readonly deferredSeconds: number;
    readonly droppedSeconds: number;
    readonly saturated: boolean;
}

export interface FluidTimestepSchedule {
    readonly frameDeltaSeconds: number;
    readonly substepDeltaSeconds: number;
    readonly substeps: number;
    readonly diagnostics: FluidTimestepDiagnostics;
}

export interface FluidTimestepScheduler {
    readonly maximumSubstepsPerFrame: number;
    readonly maximumDebtSeconds: number;
    debtSeconds: number;
    droppedSeconds: number;
    saturated: boolean;
}

export interface FluidTimestepSchedulerOptions {
    readonly maximumSubstepsPerFrame?: number;
    readonly maximumDebtSeconds?: number;
}

/** Bound per-frame GPU work while retaining a short, explicit real-time catch-up debt. */
export function createFluidTimestepScheduler(options: FluidTimestepSchedulerOptions = {}): FluidTimestepScheduler {
    const maximumSubstepsPerFrame = Math.max(1, Math.floor(options.maximumSubstepsPerFrame ?? 16));
    const maximumDebtSeconds = Math.max(0, options.maximumDebtSeconds ?? 0.25);
    if (!Number.isFinite(maximumSubstepsPerFrame) || !Number.isFinite(maximumDebtSeconds)) {
        throw new TypeError("[fluid] timestep scheduler limits must be finite.");
    }
    return {
        maximumSubstepsPerFrame,
        maximumDebtSeconds,
        debtSeconds: 0,
        droppedSeconds: 0,
        saturated: false,
    };
}

function retainFluidTimestepDebt(scheduler: FluidTimestepScheduler, seconds: number): void {
    if (seconds <= 0) {
        return;
    }
    const next = scheduler.debtSeconds + seconds;
    if (next > scheduler.maximumDebtSeconds) {
        scheduler.droppedSeconds += next - scheduler.maximumDebtSeconds;
        scheduler.debtSeconds = scheduler.maximumDebtSeconds;
        scheduler.saturated = true;
    } else {
        scheduler.debtSeconds = next;
    }
}

export function getFluidTimestepDiagnostics(scheduler: FluidTimestepScheduler): FluidTimestepDiagnostics {
    return {
        deferredSeconds: scheduler.debtSeconds,
        droppedSeconds: scheduler.droppedSeconds,
        saturated: scheduler.saturated,
    };
}

export function deferFluidTimestep(scheduler: FluidTimestepScheduler, deltaSeconds: number): void {
    if (Number.isFinite(deltaSeconds) && deltaSeconds > 0) {
        retainFluidTimestepDebt(scheduler, deltaSeconds);
    }
}

export function scheduleFluidTimestep(
    scheduler: FluidTimestepScheduler,
    deltaSeconds: number,
    minimumSubsteps: number,
    maximumSubstepSeconds: number,
    requestedMaximumSubsteps?: number
): FluidTimestepSchedule | null {
    if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) {
        return null;
    }
    if (!Number.isFinite(minimumSubsteps) || !Number.isFinite(maximumSubstepSeconds) || maximumSubstepSeconds <= 0) {
        throw new RangeError("[fluid] timestep scheduling requires finite positive substep limits.");
    }
    const minimum = Math.min(scheduler.maximumSubstepsPerFrame, Math.max(1, Math.floor(minimumSubsteps)));
    const requestedBudget = requestedMaximumSubsteps === undefined ? scheduler.maximumSubstepsPerFrame : Math.max(1, Math.floor(requestedMaximumSubsteps));
    if (!Number.isFinite(requestedBudget)) {
        throw new RangeError("[fluid] timestep scheduling requires a finite maximum substep count.");
    }
    const budget = Math.min(scheduler.maximumSubstepsPerFrame, Math.max(minimum, requestedBudget));
    const available = scheduler.debtSeconds + deltaSeconds;
    scheduler.debtSeconds = 0;
    const maximumAdvance = budget * maximumSubstepSeconds;
    const frameDeltaSeconds = Math.min(available, maximumAdvance);
    retainFluidTimestepDebt(scheduler, available - frameDeltaSeconds);
    const substeps = Math.min(budget, Math.max(minimum, Math.ceil(frameDeltaSeconds / maximumSubstepSeconds - 1e-9)));
    return {
        frameDeltaSeconds,
        substepDeltaSeconds: frameDeltaSeconds / substeps,
        substeps,
        diagnostics: getFluidTimestepDiagnostics(scheduler),
    };
}

export function resetFluidTimestepScheduler(scheduler: FluidTimestepScheduler): void {
    scheduler.debtSeconds = 0;
    scheduler.droppedSeconds = 0;
    scheduler.saturated = false;
}
