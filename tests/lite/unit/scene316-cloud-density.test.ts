import { describe, expect, it } from "vitest";
import { createScene316CloudDensitySource } from "../../../lab/lite/src/shared/scene316-cloud-density";

describe("Scene 316 density source", () => {
    it("keeps the bank shape in lab WGSL and uploads only live animation time", () => {
        const state = { time: 0 };
        const source = createScene316CloudDensitySource(state);
        const data = new Float32Array(4);
        expect(source.uniformFloatCount).toBe(4);
        expect(source.shader).toContain("fn fogDensityAt(worldPosition:vec3f)");
        expect(source.shader).not.toContain("cloudWind");
        source.validate?.();
        source.writeUniforms?.(data);
        expect(Array.from(data)).toEqual([0, 0, 0, 0]);
        state.time = 3;
        source.validate?.();
        source.writeUniforms?.(data);
        expect(Array.from(data)).toEqual([3, 0, 0, 0]);
    });

    it("rejects non-finite animation time before uploading its uniform", () => {
        const state = { time: NaN };
        const source = createScene316CloudDensitySource(state);
        expect(() => source.validate?.()).toThrow("cloud time");
        state.time = Infinity;
        expect(() => source.validate?.()).toThrow("cloud time");
        state.time = 1.5;
        expect(() => source.validate?.()).not.toThrow();
    });
});
