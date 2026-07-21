import { describe, expect, it } from "vitest";

import { svd3 } from "../../../packages/babylon-lite/src/fluid/svd3";

// Row-major 3×3 helpers.
type M9 = number[];

function mul(a: M9, b: M9): M9 {
    const m = new Array<number>(9).fill(0);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            m[r * 3 + c] = a[r * 3]! * b[c]! + a[r * 3 + 1]! * b[3 + c]! + a[r * 3 + 2]! * b[6 + c]!;
        }
    }
    return m;
}
function transpose(a: M9): M9 {
    return [a[0]!, a[3]!, a[6]!, a[1]!, a[4]!, a[7]!, a[2]!, a[5]!, a[8]!];
}
function diag(s: readonly [number, number, number]): M9 {
    return [s[0], 0, 0, 0, s[1], 0, 0, 0, s[2]];
}
function det(a: M9): number {
    return a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!) - a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!) + a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!);
}
const IDENT: M9 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function expectMatClose(actual: M9, expected: M9, tol: number): void {
    for (let i = 0; i < 9; i++) expect(Math.abs(actual[i]! - expected[i]!), `entry ${i}: ${actual[i]} vs ${expected[i]}`).toBeLessThan(tol);
}

// Full validation of a decomposition against its source matrix.
function checkSvd(a: M9): void {
    const { u, s, v } = svd3(a);

    // U, V are orthonormal rotations (built from quaternions → exact).
    expectMatClose(mul(transpose(u), u), IDENT, 1e-4);
    expectMatClose(mul(transpose(v), v), IDENT, 1e-4);
    expect(Math.abs(det(u) - 1), `det(U)=${det(u)}`).toBeLessThan(1e-3);
    expect(Math.abs(det(v) - 1), `det(V)=${det(v)}`).toBeLessThan(1e-3);

    // Reconstruction A ≈ U·diag(S)·Vᵀ. The fixed 4-sweep McAdams SVD is approximate: its residual
    // (the off-diagonal of R) scales with the matrix magnitude, so bound the error RELATIVE to ‖A‖ ≈ |s0|.
    const scale = Math.max(1, Math.abs(s[0]));
    const recon = mul(mul(u, diag(s)), transpose(v));
    expectMatClose(recon, a, 5e-3 * scale);

    // Singular values sorted by decreasing magnitude.
    expect(Math.abs(s[0])).toBeGreaterThanOrEqual(Math.abs(s[1]) - 1e-5);
    expect(Math.abs(s[1])).toBeGreaterThanOrEqual(Math.abs(s[2]) - 1e-5);

    // Signed singular-value product matches det(A) (U, V are rotations).
    expect(Math.abs(s[0] * s[1] * s[2] - det(a)), `prod S vs det A`).toBeLessThan(1e-2 * (1 + Math.abs(det(a))));
}

describe("svd3 (3×3 SVD)", () => {
    it("identity", () => checkSvd(IDENT));

    it("positive diagonal", () => {
        checkSvd([3, 0, 0, 0, 2, 0, 0, 0, 1]);
        checkSvd([5, 0, 0, 0, 0.25, 0, 0, 0, 4]);
    });

    it("pure rotation → all singular values ≈ 1", () => {
        // Rotation about a tilted axis.
        const c = Math.cos(0.7);
        const s = Math.sin(0.7);
        const rot: M9 = [c, -s, 0, s, c, 0, 0, 0, 1];
        const { s: sv } = svd3(rot);
        for (const val of sv) expect(Math.abs(Math.abs(val) - 1)).toBeLessThan(2e-3);
        checkSvd(rot);
    });

    it("symmetric matrix", () => {
        checkSvd([2, -1, 0, -1, 2, -1, 0, -1, 2]);
        checkSvd([4, 1, 2, 1, 3, -1, 2, -1, 5]);
    });

    it("negative determinant (reflection): smallest singular value is negative", () => {
        const a: M9 = [1, 0, 0, 0, 2, 0, 0, 0, -3];
        checkSvd(a);
        const { s } = svd3(a);
        expect(s[0] * s[1] * s[2]).toBeLessThan(0); // det < 0
    });

    it("general / rotated non-uniform scale", () => {
        // R(θ) · diag(3,1,0.5) — a rotated ellipsoid.
        const cx = Math.cos(0.5),
            sx = Math.sin(0.5);
        const rx: M9 = [1, 0, 0, 0, cx, -sx, 0, sx, cx];
        const scale: M9 = [3, 0, 0, 0, 1, 0, 0, 0, 0.5];
        checkSvd(mul(rx, scale));
        checkSvd([1, 2, 3, 4, 5, 6, 7, 8, 10]);
        checkSvd([0.2, -1.3, 0.7, 2.1, 0.4, -0.9, -1.1, 0.6, 1.8]);
    });

    it("near-singular (rank-deficient) matrix", () => {
        // Two nearly-parallel rows → one tiny singular value.
        const a: M9 = [1, 2, 3, 1.0001, 2.0001, 3.0001, 0, 1, 0];
        const { s } = svd3(a);
        expect(Math.abs(s[2])).toBeLessThan(Math.abs(s[0])); // clearly separated
        checkSvd(a);
    });

    it("randomized fuzz", () => {
        let seed = 1234567;
        const rnd = (): number => {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            return (seed / 0x7fffffff) * 4 - 2;
        };
        for (let t = 0; t < 200; t++) {
            const a: M9 = Array.from({ length: 9 }, rnd);
            checkSvd(a);
        }
    });
});
