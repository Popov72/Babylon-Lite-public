import { describe, expect, it } from "vitest";

import { fluidCaptureCompletionTime, fluidSimulationLifecycle, fluidSimulationStepDelta } from "../../../lab/lite/src/demos/fluid/simulation-lifecycle";

describe("fluid simulation lifecycle", () => {
    it("runs indefinitely when duration is zero", () => {
        expect(fluidSimulationLifecycle(1000, 0, 2)).toEqual({ opacity: 1, stopped: false });
    });

    it("remains fully visible before the duration", () => {
        expect(fluidSimulationLifecycle(4, 5, 2)).toEqual({ opacity: 1, stopped: false });
    });

    it("decays smoothly after the duration", () => {
        expect(fluidSimulationLifecycle(6, 5, 2)).toEqual({ opacity: 0.5, stopped: false });
    });

    it("stops at zero opacity", () => {
        expect(fluidSimulationLifecycle(7, 5, 2)).toEqual({ opacity: 0, stopped: true });
        expect(fluidSimulationLifecycle(5, 5, 0)).toEqual({ opacity: 0, stopped: true });
    });

    it("clamps the final solver step to the exact lifecycle stop time", () => {
        expect(fluidSimulationStepDelta(6.9, 0.2, 5, 2)).toBeCloseTo(0.1, 8);
        expect(fluidSimulationStepDelta(10, 1 / 60, 0, 2)).toBeCloseTo(1 / 60, 8);
    });

    it("completes an overlong capture at the stopped simulation time", () => {
        const elapsed = 6.9 + fluidSimulationStepDelta(6.9, 0.2, 5, 2);
        const lifecycle = fluidSimulationLifecycle(elapsed, 5, 2);

        expect(elapsed).toBeCloseTo(7, 8);
        expect(lifecycle.stopped).toBe(true);
        expect(fluidCaptureCompletionTime(elapsed, false, lifecycle.stopped)).toBeCloseTo(7, 8);
        expect(fluidCaptureCompletionTime(7, false, false)).toBeNull();
        expect(fluidCaptureCompletionTime(10, true, false)).toBe(10);
    });
});
