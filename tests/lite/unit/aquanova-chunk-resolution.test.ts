import { describe, expect, it } from "vitest";
import { chunkAt, type ShipChunk } from "../../../lab/lite/src/demos/aquanova/manifest";

const chunks: ShipChunk[] = [
    {
        id: "CH01_StorageC",
        aabb: { min: [-4.0017, -0.0065, -4], max: [12, 5.0046, 4.774] },
    },
    {
        id: "CH02_StorageCTL",
        aabb: { min: [3.9983, -0.0065, -4.0017], max: [12.774, 5.0046, 4.774] },
    },
    {
        id: "CH03_StorageC2",
        aabb: { min: [7.226, -0.013, -20.774], max: [16.2534, 5.006, -3.9983] },
    },
];

describe("Aquanova chunk resolution", () => {
    it("selects the narrower split chunk when exported mesh bounds overlap", () => {
        expect(chunkAt(chunks, 6, 0)?.id).toBe("CH02_StorageCTL");
    });

    it("retains the containing chunk outside the overlap", () => {
        expect(chunkAt(chunks, 0, 0)?.id).toBe("CH01_StorageC");
        expect(chunkAt(chunks, 10, -10)?.id).toBe("CH03_StorageC2");
    });

    it("returns undefined outside the ship", () => {
        expect(chunkAt(chunks, 100, 100)).toBeUndefined();
    });
});
