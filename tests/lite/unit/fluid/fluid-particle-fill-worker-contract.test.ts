import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import type { ParticleFillWorkerResponse } from "../../../../lab/lite/src/demos/particle-fill-worker";

const ROOT = new URL("../../../../", import.meta.url);

describe("particle fill worker protocol", () => {
    it("has a discriminated structured error field", () => {
        const failure: ParticleFillWorkerResponse = {
            id: 12,
            error: {
                code: "VOLUME_SAMPLER_EMPTY",
                message: "forced volume was empty",
                requestedStrategy: "volume",
            },
        };

        expect(failure.error).toEqual(expect.objectContaining({ code: "VOLUME_SAMPLER_EMPTY" }));
        expect("positions" in failure).toBe(false);
    });

    it("delegates computation to the root API and both Aquanova consumers propagate errors", () => {
        const worker = readFileSync(new URL("lab/lite/src/demos/particle-fill-worker.ts", ROOT), "utf8");
        const game = readFileSync(new URL("lab/lite/src/demos/aquanova/main.ts", ROOT), "utf8");
        const lab = readFileSync(new URL("lab/lite/src/demos/aquanova-fluid-sim.ts", ROOT), "utf8");

        expect(worker).toContain('sampleFluidMeshParticles } from "babylon-lite"');
        expect(worker).not.toContain('from "./particle-fill.js"');
        expect(worker).toContain("error: fluidMeshSamplingErrorInfo");
        for (const consumer of [game, lab]) {
            expect(consumer).toContain("if (ev.data.error)");
            expect(consumer).toContain("failSample(entry, ev.data.error)");
            expect(consumer).toContain('code: "EMPTY_RESULT"');
        }
    });
});
