import { describe, expect, it } from "vitest";
import { liquefyFrontDistance } from "../../../lab/lite/src/demos/liquefy-plugin";

describe("Aquanova liquefaction particle mask", () => {
    it("reduces to the world-space spherical front when noise is disabled", () => {
        expect(liquefyFrontDistance(4, 2, 3, [1, 2, 3], 0)).toBe(3);
    });

    it("keeps the shader-matched perturbation within the configured amplitude", () => {
        const hit = [1, -2, 0.5] as const;
        const rawDistance = Math.hypot(3 - hit[0], 4 - hit[1], -1 - hit[2]);
        const maskedDistance = liquefyFrontDistance(3, 4, -1, hit, 0.35, 1.2);
        expect(maskedDistance).toBeGreaterThanOrEqual(rawDistance - 0.35);
        expect(maskedDistance).toBeLessThanOrEqual(rawDistance + 0.35);
        expect(liquefyFrontDistance(3, 4, -1, hit, 0.35, 1.2)).toBe(maskedDistance);
    });
});
