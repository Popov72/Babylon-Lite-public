import { describe, expect, it } from "vitest";
import { DEFAULT_DYNAMIC_MASS, resolveDynamicMass } from "../../../../lab/lite/src/demos/aquanova/behaviors/dynamic";

describe("Aquanova dynamic behavior", () => {
    it("defaults mass to 10 kilograms and accepts a positive override", () => {
        expect(resolveDynamicMass({})).toBe(DEFAULT_DYNAMIC_MASS);
        expect(resolveDynamicMass({ mass: 42 })).toBe(42);
    });

    it("rejects non-positive and non-finite masses", () => {
        expect(() => resolveDynamicMass({ mass: 0 })).toThrow("dynamic.mass");
        expect(() => resolveDynamicMass({ mass: Number.POSITIVE_INFINITY })).toThrow("dynamic.mass");
    });
});
