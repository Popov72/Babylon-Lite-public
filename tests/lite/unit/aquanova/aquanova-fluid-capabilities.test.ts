import { describe, expect, it } from "vitest";

import {
    AQUANOVA_PRODUCTION_FLUID_CAPABILITIES,
    productionFluidCapabilityRejection,
} from "../../../../lab/lite/src/demos/aquanova/fluid-capabilities.js";

describe("Aquanova production fluid capability gates", () => {
    it("declares shared polygon rendering support without rejecting FLIP presets", () => {
        expect(AQUANOVA_PRODUCTION_FLUID_CAPABILITIES.polygonSurface).toBe(true);
        expect(productionFluidCapabilityRejection("FLIP", { polygonSurface: 1 })).toBeNull();
        expect(productionFluidCapabilityRejection("FLIP", { polygonSurface: 0 })).toBeNull();
        expect(productionFluidCapabilityRejection("MLS-MPM", { polygonSurface: 1 })).toBeNull();
    });
});
