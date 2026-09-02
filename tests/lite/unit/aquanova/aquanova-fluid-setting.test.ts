import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchFluidSetting } from "../../../../lab/lite/src/demos/aquanova/fluid-setting";
import { exportFluidPresetSession } from "../../../../packages/babylon-lite/src";

describe("Aquanova fluid settings", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("appends the JSON extension only when fetching the setting file", async () => {
        const fetchMock = vi.fn().mockResolvedValue({
            ok: true,
            json: async () => ({
                meta: { method: "MLS-MPM" },
                physics: { gravity: -9.8 },
                simulationDuration: 12,
                alphaDecay: 3,
                futureProduction: { retained: true },
            }),
        });
        vi.stubGlobal("fetch", fetchMock);

        const setting = await fetchFluidSetting("liquid-slow");
        expect(setting).toMatchObject({
            method: "MLS-MPM",
            physics: { gravity: -9.8 },
        });
        const exported = exportFluidPresetSession(setting!.session, { demo: "aquanova", method: setting!.method });
        expect(exported).toMatchObject({
            simulationDuration: 12,
            alphaDecay: 3,
            futureProduction: { retained: true },
        });
        expect(fetchMock).toHaveBeenCalledWith("/aquanova/fluidSim/liquid-slow.json");
    });

    it("rejects manifest names that still include the extension", async () => {
        await expect(fetchFluidSetting("liquid-slow.json")).rejects.toThrow('[aquanova] fluidSim name "liquid-slow.json" must omit the .json extension');
    });
});
