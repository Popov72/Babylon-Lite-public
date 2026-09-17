import { describe, expect, it } from "vitest";
import { markFlipReferenceClosedPockets } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/closed-pockets";

function twoCells(aperture: number) {
    const rows = new Float32Array(32);
    rows[3] = rows[19] = 1;
    rows[4] = rows[24] = aperture;
    return rows;
}

describe("reference closed-pocket conditioning", () => {
    it("marks a multi-cell pocket without changing pressure coefficients or RHS", () => {
        const rows = twoCells(0.2);
        rows[2] = 3;
        rows[18] = -2;
        expect(markFlipReferenceClosedPockets(rows, [2, 1, 1], new Uint8Array(2), new Int32Array(2))).toEqual({ components: 1, cells: 2 });
        expect([rows[14], rows[30]]).toEqual([1, 1]);
        expect([rows[2], rows[18], rows[4], rows[24]]).toEqual([3, -2, Math.fround(0.2), Math.fround(0.2)]);
    });

    it("does not condition a component connected to air and clears prior flags", () => {
        const rows = twoCells(0.2);
        rows[29] = 1;
        rows[14] = rows[30] = 1;
        expect(markFlipReferenceClosedPockets(rows, [2, 1, 1], new Uint8Array(2), new Int32Array(2))).toEqual({ components: 0, cells: 0 });
        expect([rows[14], rows[30]]).toEqual([0, 0]);
    });

    it("uses the same float32 threshold as the GPU and skips single-cell components", () => {
        const tiny = twoCells(0.5e-6);
        expect(markFlipReferenceClosedPockets(tiny, [2, 1, 1], new Uint8Array(2), new Int32Array(2))).toEqual({ components: 0, cells: 0 });
        const atThreshold = twoCells(1e-6);
        expect(markFlipReferenceClosedPockets(atThreshold, [2, 1, 1], new Uint8Array(2), new Int32Array(2))).toEqual({ components: 1, cells: 2 });
    });
});
