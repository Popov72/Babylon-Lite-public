import { describe, expect, it } from "vitest";
import { buildSkyPortalMask } from "../../../lab/lite/src/demos/aquanova/exterior-mesh-classifier";
import type { RuntimePortal } from "../../../lab/lite/src/demos/aquanova/portal-visibility";

const skyPortal: RuntimePortal = {
    id: "outside",
    chunkA: "A",
    chunkB: "__SKYBOX__",
    centre: [0, 0, 2],
    normal: [0, 0, -1],
    corners: [
        [-1, -1, 2],
        [1, -1, 2],
        [1, 1, 2],
        [-1, 1, 2],
    ],
    enabled: true,
};

describe("Aquanova exterior mesh classifier", () => {
    it("masks a sky portal when viewed from outside", () => {
        const mask = buildSkyPortalMask(
            {
                direction: [0, 0, -1],
                right: [-1, 0, 0],
                up: [0, 1, 0],
            },
            [skyPortal],
            [0, 0, 0],
            4,
            10
        );

        expect(mask.some((value) => value !== 0)).toBe(true);
    });

    it("does not mask the portal when viewed from its interior side", () => {
        const mask = buildSkyPortalMask(
            {
                direction: [0, 0, 1],
                right: [1, 0, 0],
                up: [0, 1, 0],
            },
            [skyPortal],
            [0, 0, 0],
            4,
            10
        );

        expect(mask.some((value) => value !== 0)).toBe(false);
    });
});
