import { describe, expect, it } from "vitest";

import {
    createFluidTimestepScheduler,
    deferFluidTimestep,
    getFluidTimestepDiagnostics,
    resetFluidTimestepScheduler,
    scheduleFluidTimestep,
} from "../../../../packages/babylon-lite/src/fluid/core/timestep-scheduler.js";

describe("fluid timestep scheduler", () => {
    it("treats non-positive and non-finite deltas as no-ops", () => {
        const scheduler = createFluidTimestepScheduler();
        expect(scheduleFluidTimestep(scheduler, 0, 1, 1 / 120)).toBeNull();
        expect(scheduleFluidTimestep(scheduler, -1, 1, 1 / 120)).toBeNull();
        expect(scheduleFluidTimestep(scheduler, Number.NaN, 1, 1 / 120)).toBeNull();
        expect(scheduleFluidTimestep(scheduler, Number.POSITIVE_INFINITY, 1, 1 / 120)).toBeNull();
        expect(getFluidTimestepDiagnostics(scheduler)).toEqual({ deferredSeconds: 0, droppedSeconds: 0, saturated: false });
    });

    it("caps encoded work and retains only bounded catch-up debt", () => {
        const scheduler = createFluidTimestepScheduler({ maximumSubstepsPerFrame: 8, maximumDebtSeconds: 0.25 });
        const schedule = scheduleFluidTimestep(scheduler, 10, 3, 1 / 120)!;

        expect(schedule.substeps).toBe(8);
        expect(schedule.frameDeltaSeconds).toBeCloseTo(8 / 120);
        expect(schedule.diagnostics.deferredSeconds).toBe(0.25);
        expect(schedule.diagnostics.droppedSeconds).toBeCloseTo(10 - 8 / 120 - 0.25);
        expect(schedule.diagnostics.saturated).toBe(true);
    });

    it("combines blocked time with the next frame without exceeding the budget", () => {
        const scheduler = createFluidTimestepScheduler({ maximumSubstepsPerFrame: 4, maximumDebtSeconds: 0.1 });
        deferFluidTimestep(scheduler, 0.04);
        const schedule = scheduleFluidTimestep(scheduler, 0.01, 2, 0.01)!;

        expect(schedule.substeps).toBe(4);
        expect(schedule.frameDeltaSeconds).toBeCloseTo(0.04);
        expect(schedule.diagnostics.deferredSeconds).toBeCloseTo(0.01);
        resetFluidTimestepScheduler(scheduler);
        expect(getFluidTimestepDiagnostics(scheduler)).toEqual({ deferredSeconds: 0, droppedSeconds: 0, saturated: false });
    });

    it("honors a solver's lower dynamic substep ceiling", () => {
        const scheduler = createFluidTimestepScheduler({ maximumSubstepsPerFrame: 16 });
        const schedule = scheduleFluidTimestep(scheduler, 1, 2, 1 / 120, 6)!;

        expect(schedule.substeps).toBe(6);
        expect(schedule.frameDeltaSeconds).toBeCloseTo(6 / 120);
        expect(schedule.diagnostics.deferredSeconds).toBeGreaterThan(0);
    });

    it("keeps the global cap authoritative when the requested minimum is larger", () => {
        const scheduler = createFluidTimestepScheduler({ maximumSubstepsPerFrame: 16 });
        const schedule = scheduleFluidTimestep(scheduler, 1, 1000, 1 / 120)!;

        expect(schedule.substeps).toBe(16);
        expect(schedule.frameDeltaSeconds).toBeCloseTo(16 / 120);
        expect(schedule.diagnostics.deferredSeconds).toBeGreaterThan(0);
    });
});
