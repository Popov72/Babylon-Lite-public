import { describe, expect, it } from "vitest";
import { normalizeSoundVolume } from "../../../../lab/lite/src/demos/aquanova/behaviors/sound-volume";
import { loadGraphicsSettings } from "../../../../lab/lite/src/demos/aquanova/settings";

describe("Aquanova sound volume", () => {
    it("clamps finite values to the master-gain range", () => {
        expect(normalizeSoundVolume(-1)).toBe(0);
        expect(normalizeSoundVolume(0.35)).toBe(0.35);
        expect(normalizeSoundVolume(2)).toBe(1);
    });

    it("rejects non-finite values", () => {
        expect(() => normalizeSoundVolume(Number.NaN)).toThrow("[aquanova] sound volume must be finite");
    });

    it("loads zero volume and clamps authored values above the slider range", () => {
        expect(loadGraphicsSettings({ soundVolume: 0 }).soundVolume).toBe(0);
        expect(loadGraphicsSettings({ soundVolume: 2 }).soundVolume).toBe(1);
    });
});
