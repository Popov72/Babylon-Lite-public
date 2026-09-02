import { describe, expect, it } from "vitest";

import { fluidRenderProfileKey } from "../../../../packages/babylon-lite/src/fluid/rendering/fluid-render-profile";

describe("Aquanova independent fluid render profiles", () => {
    it("groups simulations with the same render and particle parameters", () => {
        const render = {
            independentRendering: true,
            waterColor: "#16A3C3",
            absorption: 0.4,
            refractionStrength: 0.06,
            surfaceFilter: "bilateral" as const,
        };
        expect(fluidRenderProfileKey(render, 0.035, 1, "water")).toBe(fluidRenderProfileKey({ ...render, waterColor: "16a3c3" }, 0.035, 1, "water"));
    });

    it("separates profiles when a render or particle-footprint parameter differs", () => {
        const render = { independentRendering: true, waterColor: "#16a3c3", absorption: 0.4 };
        const key = fluidRenderProfileKey(render, 0.035, 1, "water");
        expect(fluidRenderProfileKey({ ...render, absorption: 0.8 }, 0.035, 1, "water")).not.toBe(key);
        expect(fluidRenderProfileKey({ ...render, reflectionExposure: 1.2 }, 0.035, 1, "water")).not.toBe(key);
        expect(fluidRenderProfileKey(render, 0.05, 1, "water")).not.toBe(key);
        expect(fluidRenderProfileKey(render, 0.035, 1.2, "water")).not.toBe(key);
        expect(fluidRenderProfileKey(render, 0.035, 1, "mesh")).not.toBe(key);
    });
});
