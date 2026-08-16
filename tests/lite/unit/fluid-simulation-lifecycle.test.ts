import { describe, expect, it } from "vitest";

import { fluidSimulationLifecycle } from "../../../lab/lite/src/demos/fluid/simulation-lifecycle";

describe("fluid simulation lifecycle", () => {
    it("runs indefinitely when duration is zero", () => {
        expect(fluidSimulationLifecycle(1000, 0, 2)).toEqual({ opacity: 1, stopped: false });
    });

    it("fades after the duration and stops at zero opacity", () => {
        expect(fluidSimulationLifecycle(4, 5, 2)).toEqual({ opacity: 1, stopped: false });
        expect(fluidSimulationLifecycle(6, 5, 2)).toEqual({ opacity: 0.5, stopped: false });
        expect(fluidSimulationLifecycle(7, 5, 2)).toEqual({ opacity: 0, stopped: true });
    });

    it("stops immediately after the duration when decay is zero", () => {
        expect(fluidSimulationLifecycle(5.01, 5, 0)).toEqual({ opacity: 0, stopped: true });
    });
});
