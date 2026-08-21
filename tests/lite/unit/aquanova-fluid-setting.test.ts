import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchFluidSetting } from "../../../lab/lite/src/demos/aquanova/fluid-setting";

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
            }),
        });
        vi.stubGlobal("fetch", fetchMock);

        await expect(fetchFluidSetting("liquid-slow")).resolves.toMatchObject({
            method: "MLS-MPM",
            physics: { gravity: -9.8 },
        });
        expect(fetchMock).toHaveBeenCalledWith("/aquanova/fluidSim/liquid-slow.json");
    });

    it("rejects manifest names that still include the extension", async () => {
        await expect(fetchFluidSetting("liquid-slow.json")).rejects.toThrow('[aquanova] fluidSim name "liquid-slow.json" must omit the .json extension');
    });
});
