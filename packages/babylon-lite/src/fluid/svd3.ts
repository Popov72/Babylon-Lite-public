// Branchless 3×3 Singular Value Decomposition (SVD).
//
// A = U · diag(S) · Vᵀ, with U and V pure ROTATIONS (det = +1) and S the (signed) singular values.
// Because U and V are constrained to rotations, the SMALLEST singular value carries the sign of det(A)
// (so a reflection stays a negative S[2] instead of flipping U/V) — exactly what MPM/plasticity wants
// (decompose F → clamp/modify the stretches S → recompose), e.g. the PB-MPM constitutive constraint solve.
//
// Algorithm: A. McAdams, A. Selle, R. Tamstorf, J. Teran, E. Sifakis, "Computing the Singular Value
// Decomposition of 3×3 matrices with minimal branching and elementary floating point operations",
// University of Wisconsin–Madison, tech report TR1690 (2011). Ported from Eric Jang's public-domain
// reference C++ (https://github.com/ericjang/svd3): Jacobi eigenanalysis of AᵀA (quaternion V) →
// B = A·V → sort columns → Givens QR of B → U and the diagonal S.
//
// This module ships the algorithm twice, kept structurally identical so they stay in lock-step:
//   • {@link svd3} — a CPU/TypeScript reference (row-major 9-arrays), used by tests and any CPU path.
//   • {@link SVD3_WGSL} — a WGSL string exposing `fn svd3(A: mat3x3<f32>) -> Svd3` for GPU solvers.

const SVD_GAMMA = 5.828427124; // sqrt(8) + 3  (= 4·γ², the McAdams convergence constant)
const SVD_CSTAR = 0.923879532; // cos(π/8)
const SVD_SSTAR = 0.3826834323; // sin(π/8)
const SVD_EPS = 1e-6;

/** SVD result as row-major 3×3 matrices (`u`, `v`, both 9-element) and the three singular values `s`. */
export interface Svd3Result {
    /** Left rotation U (row-major, det = +1). */
    u: number[];
    /** Singular values [σ0, σ1, σ2], sorted so |σ0| ≥ |σ1| ≥ |σ2|; σ2 may be negative (sign of det A). */
    s: [number, number, number];
    /** Right rotation V (row-major, det = +1). */
    v: number[];
}

// Approximate the Givens quaternion (ch, sh) that diagonalizes the leading 2×2 of a symmetric matrix.
function approximateGivensQuaternion(a11: number, a12: number, a22: number): [number, number] {
    let ch = 2 * (a11 - a22);
    let sh = a12;
    const b = SVD_GAMMA * sh * sh < ch * ch;
    const w = 1 / Math.sqrt(ch * ch + sh * sh);
    ch = b ? w * ch : SVD_CSTAR;
    sh = b ? w * sh : SVD_SSTAR;
    return [ch, sh];
}

// One Jacobi conjugation S ← Qᵀ S Q for the (p,q) pair encoded by (x,y,z), accumulating V into qV.
// S is the symmetric lower triangle [s11, s21, s22, s31, s32, s33]; qV is [x, y, z, w].
function jacobiConjugation(x: number, y: number, z: number, s: number[], qV: number[]): void {
    const [ch, sh] = approximateGivensQuaternion(s[0]!, s[1]!, s[2]!);
    const scale = ch * ch + sh * sh;
    const a = (ch * ch - sh * sh) / scale;
    const b = (2 * sh * ch) / scale;

    const _s11 = s[0]!,
        _s21 = s[1]!,
        _s22 = s[2]!,
        _s31 = s[3]!,
        _s32 = s[4]!,
        _s33 = s[5]!;

    // S = Q'·S·Q (Q implicit from a, b)
    const n11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
    const n21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
    const n22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
    const n31 = a * _s31 + b * _s32;
    const n32 = -b * _s31 + a * _s32;
    const n33 = _s33;

    // accumulate the cumulative rotation qV
    const tmp = [qV[0]! * sh, qV[1]! * sh, qV[2]! * sh];
    const shw = sh * qV[3]!;
    qV[0]! *= ch;
    qV[1]! *= ch;
    qV[2]! *= ch;
    qV[3]! *= ch;
    qV[z]! += shw;
    qV[3]! -= tmp[z]!;
    qV[x]! += tmp[y]!;
    qV[y]! -= tmp[x]!;

    // cyclically re-arrange S for the next (p,q) pair
    s[0] = n22;
    s[1] = n32;
    s[2] = n33;
    s[3] = n21;
    s[4] = n31;
    s[5] = n11;
}

// Diagonalize the symmetric AᵀA via cyclic Jacobi sweeps, returning V as a quaternion [x, y, z, w].
function jacobiEigenanalysis(s: number[]): number[] {
    const qV = [0, 0, 0, 1];
    for (let i = 0; i < 4; i++) {
        jacobiConjugation(0, 1, 2, s, qV);
        jacobiConjugation(1, 2, 0, s, qV);
        jacobiConjugation(2, 0, 1, s, qV);
    }
    return qV;
}

// Quaternion [x, y, z, w] → row-major rotation matrix (9-array).
function quatToMat3(q: number[]): number[] {
    const x = q[0]!,
        y = q[1]!,
        z = q[2]!,
        w = q[3]!;
    const xx = x * x,
        yy = y * y,
        zz = z * z,
        xz = x * z,
        xy = x * y,
        yz = y * z,
        wx = w * x,
        wy = w * y,
        wz = w * z;
    return [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)];
}

// Row-major 3×3 multiply M = A·B.
function multAB(A: number[], B: number[]): number[] {
    const m = new Array<number>(9);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            m[r * 3 + c] = A[r * 3]! * B[c]! + A[r * 3 + 1]! * B[3 + c]! + A[r * 3 + 2]! * B[6 + c]!;
        }
    }
    return m;
}

const dist2 = (x: number, y: number, z: number): number => x * x + y * y + z * z;

// Swap array entries i,j when c; the "Neg" variant also negates the incoming value (a column-sign flip).
function condSwap(c: boolean, a: number[], i: number, j: number): void {
    if (c) {
        const t = a[i]!;
        a[i] = a[j]!;
        a[j] = t;
    }
}
function condNegSwap(c: boolean, a: number[], i: number, j: number): void {
    if (c) {
        const t = a[i]!;
        a[i] = a[j]!;
        a[j] = -t;
    }
}

// Sort B's columns by decreasing magnitude, applying the same column swaps/sign-flips to V.
function sortSingularValues(B: number[], V: number[]): void {
    const rho = [dist2(B[0]!, B[3]!, B[6]!), dist2(B[1]!, B[4]!, B[7]!), dist2(B[2]!, B[5]!, B[8]!)];
    const swapCols = (c: boolean, ci: number, cj: number): void => {
        for (let r = 0; r < 3; r++) {
            condNegSwap(c, B, r * 3 + ci, r * 3 + cj);
            condNegSwap(c, V, r * 3 + ci, r * 3 + cj);
        }
    };
    let c = rho[0]! < rho[1]!;
    swapCols(c, 0, 1);
    condSwap(c, rho, 0, 1);
    c = rho[0]! < rho[2]!;
    swapCols(c, 0, 2);
    condSwap(c, rho, 0, 2);
    c = rho[1]! < rho[2]!;
    swapCols(c, 1, 2);
}

// Givens (ch, sh) that annihilates a2 against pivot a1.
function qrGivensQuaternion(a1: number, a2: number): [number, number] {
    const rho = Math.sqrt(a1 * a1 + a2 * a2);
    let sh = rho > SVD_EPS ? a2 : 0;
    let ch = Math.abs(a1) + Math.max(rho, SVD_EPS);
    if (a1 < 0) {
        const t = sh;
        sh = ch;
        ch = t;
    }
    const w = 1 / Math.sqrt(ch * ch + sh * sh);
    return [ch * w, sh * w];
}

// QR of B via three Givens rotations → Q (row-major) and R (row-major, upper-triangular).
function qrDecomposition(B: number[]): { Q: number[]; R: number[] } {
    let b11 = B[0]!,
        b12 = B[1]!,
        b13 = B[2]!,
        b21 = B[3]!,
        b22 = B[4]!,
        b23 = B[5]!,
        b31 = B[6]!,
        b32 = B[7]!,
        b33 = B[8]!;

    // first Givens (annihilate b21)
    const [ch1, sh1] = qrGivensQuaternion(b11, b21);
    let a = 1 - 2 * sh1 * sh1;
    let b = 2 * ch1 * sh1;
    let r11 = a * b11 + b * b21,
        r12 = a * b12 + b * b22,
        r13 = a * b13 + b * b23;
    let r21 = -b * b11 + a * b21,
        r22 = -b * b12 + a * b22,
        r23 = -b * b13 + a * b23;
    let r31 = b31,
        r32 = b32,
        r33 = b33;

    // second Givens (annihilate r31)
    const [ch2, sh2] = qrGivensQuaternion(r11, r31);
    a = 1 - 2 * sh2 * sh2;
    b = 2 * ch2 * sh2;
    b11 = a * r11 + b * r31;
    b12 = a * r12 + b * r32;
    b13 = a * r13 + b * r33;
    b21 = r21;
    b22 = r22;
    b23 = r23;
    b31 = -b * r11 + a * r31;
    b32 = -b * r12 + a * r32;
    b33 = -b * r13 + a * r33;

    // third Givens (annihilate b32)
    const [ch3, sh3] = qrGivensQuaternion(b22, b32);
    a = 1 - 2 * sh3 * sh3;
    b = 2 * ch3 * sh3;
    r11 = b11;
    r12 = b12;
    r13 = b13;
    r21 = a * b21 + b * b31;
    r22 = a * b22 + b * b32;
    r23 = a * b23 + b * b33;
    r31 = -b * b21 + a * b31;
    r32 = -b * b22 + a * b32;
    r33 = -b * b23 + a * b33;

    // Q = Q1·Q2·Q3 (assembled from the three Givens quaternions)
    const sh12 = sh1 * sh1,
        sh22 = sh2 * sh2,
        sh32 = sh3 * sh3;
    const Q = [
        (-1 + 2 * sh12) * (-1 + 2 * sh22),
        4 * ch2 * ch3 * (-1 + 2 * sh12) * sh2 * sh3 + 2 * ch1 * sh1 * (-1 + 2 * sh32),
        4 * ch1 * ch3 * sh1 * sh3 - 2 * ch2 * (-1 + 2 * sh12) * sh2 * (-1 + 2 * sh32),
        2 * ch1 * sh1 * (1 - 2 * sh22),
        -8 * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 + (-1 + 2 * sh12) * (-1 + 2 * sh32),
        -2 * ch3 * sh3 + 4 * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (-1 + 2 * sh32)),
        2 * ch2 * sh2,
        2 * ch3 * (1 - 2 * sh22) * sh3,
        (-1 + 2 * sh22) * (-1 + 2 * sh32),
    ];
    const R = [r11, r12, r13, r21, r22, r23, r31, r32, r33];
    return { Q, R };
}

/**
 * Compute the SVD of a 3×3 matrix `a` (row-major 9-array): `a = u · diag(s) · vᵀ`, with `u`, `v` pure
 * rotations. CPU reference mirroring {@link SVD3_WGSL}.
 */
export function svd3(a: ArrayLike<number>): Svd3Result {
    const a11 = a[0]!,
        a12 = a[1]!,
        a13 = a[2]!,
        a21 = a[3]!,
        a22 = a[4]!,
        a23 = a[5]!,
        a31 = a[6]!,
        a32 = a[7]!,
        a33 = a[8]!;

    // AᵀA (symmetric) lower triangle [s11, s21, s22, s31, s32, s33]
    const s = [
        a11 * a11 + a21 * a21 + a31 * a31,
        a12 * a11 + a22 * a21 + a32 * a31,
        a12 * a12 + a22 * a22 + a32 * a32,
        a13 * a11 + a23 * a21 + a33 * a31,
        a13 * a12 + a23 * a22 + a33 * a32,
        a13 * a13 + a23 * a23 + a33 * a33,
    ];

    const qV = jacobiEigenanalysis(s);
    const V = quatToMat3(qV);
    const B = multAB([a11, a12, a13, a21, a22, a23, a31, a32, a33], V);
    sortSingularValues(B, V);
    const { Q, R } = qrDecomposition(B);
    return { u: Q, s: [R[0]!, R[4]!, R[8]!], v: V };
}

/**
 * WGSL 3×3 SVD, structurally identical to {@link svd3}. Inject this string into a shader, then call
 * `let r = svd3(F);` where `F: mat3x3<f32>`; `r.U`/`r.V` are rotations and `r.S` the singular values.
 * Uses only elementary ops + `inverseSqrt`/`sqrt` (no data-dependent branching or loops beyond the fixed
 * Jacobi sweep count), so it is uniform-control-flow safe inside compute kernels.
 */
export const SVD3_WGSL = /* wgsl */ `
struct Svd3 { U: mat3x3<f32>, S: vec3<f32>, V: mat3x3<f32> };

const SVD_GAMMA: f32 = 5.828427124;
const SVD_CSTAR: f32 = 0.923879532;
const SVD_SSTAR: f32 = 0.3826834323;
const SVD_EPS: f32 = 1e-6;

fn svdApproxGivens(a11: f32, a12: f32, a22: f32) -> vec2<f32> {
    var ch = 2.0 * (a11 - a22);
    var sh = a12;
    let useApprox = SVD_GAMMA * sh * sh < ch * ch;
    let w = inverseSqrt(ch * ch + sh * sh);
    ch = select(SVD_CSTAR, w * ch, useApprox);
    sh = select(SVD_SSTAR, w * sh, useApprox);
    return vec2<f32>(ch, sh);
}

// S is the symmetric lower triangle [s11, s21, s22, s31, s32, s33]; qV is [x, y, z, w].
fn svdJacobiConj(x: u32, y: u32, z: u32, s: ptr<function, array<f32, 6>>, qV: ptr<function, array<f32, 4>>) {
    let g = svdApproxGivens((*s)[0], (*s)[1], (*s)[2]);
    let ch = g.x;
    let sh = g.y;
    let scale = ch * ch + sh * sh;
    let a = (ch * ch - sh * sh) / scale;
    let b = (2.0 * sh * ch) / scale;

    let _s11 = (*s)[0]; let _s21 = (*s)[1]; let _s22 = (*s)[2];
    let _s31 = (*s)[3]; let _s32 = (*s)[4]; let _s33 = (*s)[5];

    let n11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
    let n21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
    let n22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
    let n31 = a * _s31 + b * _s32;
    let n32 = -b * _s31 + a * _s32;
    let n33 = _s33;

    var tmp: array<f32, 3>;
    tmp[0] = (*qV)[0] * sh;
    tmp[1] = (*qV)[1] * sh;
    tmp[2] = (*qV)[2] * sh;
    let shq = sh * (*qV)[3];
    (*qV)[0] = (*qV)[0] * ch;
    (*qV)[1] = (*qV)[1] * ch;
    (*qV)[2] = (*qV)[2] * ch;
    (*qV)[3] = (*qV)[3] * ch;
    (*qV)[z] = (*qV)[z] + shq;
    (*qV)[3] = (*qV)[3] - tmp[z];
    (*qV)[x] = (*qV)[x] + tmp[y];
    (*qV)[y] = (*qV)[y] - tmp[x];

    // cyclic re-arrange for the next (p,q)
    (*s)[0] = n22;
    (*s)[1] = n32;
    (*s)[2] = n33;
    (*s)[3] = n21;
    (*s)[4] = n31;
    (*s)[5] = n11;
}

fn svdQuatToMat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xz = x * z; let xy = x * y; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    // column-major: each vec3 is a COLUMN (col j = (m1j, m2j, m3j))
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)));
}

fn svdQrGivens(a1: f32, a2: f32) -> vec2<f32> {
    let rho = sqrt(a1 * a1 + a2 * a2);
    var sh = select(0.0, a2, rho > SVD_EPS);
    var ch = abs(a1) + max(rho, SVD_EPS);
    let neg = a1 < 0.0;
    let sh2 = select(sh, ch, neg);
    let ch2 = select(ch, sh, neg);
    let w = inverseSqrt(ch2 * ch2 + sh2 * sh2);
    return vec2<f32>(ch2 * w, sh2 * w);
}

fn svd3(A: mat3x3<f32>) -> Svd3 {
    // extract row/col scalars (A is column-major: A[col][row])
    let a11 = A[0][0]; let a12 = A[1][0]; let a13 = A[2][0];
    let a21 = A[0][1]; let a22 = A[1][1]; let a23 = A[2][1];
    let a31 = A[0][2]; let a32 = A[1][2]; let a33 = A[2][2];

    // AᵀA symmetric lower triangle
    var s: array<f32, 6>;
    s[0] = a11 * a11 + a21 * a21 + a31 * a31;
    s[1] = a12 * a11 + a22 * a21 + a32 * a31;
    s[2] = a12 * a12 + a22 * a22 + a32 * a32;
    s[3] = a13 * a11 + a23 * a21 + a33 * a31;
    s[4] = a13 * a12 + a23 * a22 + a33 * a32;
    s[5] = a13 * a13 + a23 * a23 + a33 * a33;

    var qV: array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 1.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        svdJacobiConj(0u, 1u, 2u, &s, &qV);
        svdJacobiConj(1u, 2u, 0u, &s, &qV);
        svdJacobiConj(2u, 0u, 1u, &s, &qV);
    }
    var V = svdQuatToMat3(vec4<f32>(qV[0], qV[1], qV[2], qV[3]));

    // B = A · V (column-major products); track B's columns as scalars for the sort + QR
    var b = A * V;
    var b11 = b[0][0]; var b12 = b[1][0]; var b13 = b[2][0];
    var b21 = b[0][1]; var b22 = b[1][1]; var b23 = b[2][1];
    var b31 = b[0][2]; var b32 = b[1][2]; var b33 = b[2][2];

    // sort singular values (columns of B) by magnitude, applying the same column ops to V
    var rho = vec3<f32>(b11 * b11 + b21 * b21 + b31 * b31, b12 * b12 + b22 * b22 + b32 * b32, b13 * b13 + b23 * b23 + b33 * b33);
    // (col i, col j) magnitude-descending swaps with sign flip; mirror on V's columns
    var c = rho.x < rho.y;
    if (c) {
        let tb1 = b11; b11 = b12; b12 = -tb1; let tb2 = b21; b21 = b22; b22 = -tb2; let tb3 = b31; b31 = b32; b32 = -tb3;
        let tv = V[0]; V[0] = V[1]; V[1] = -tv;
        let tr = rho.x; rho.x = rho.y; rho.y = tr;
    }
    c = rho.x < rho.z;
    if (c) {
        let tb1 = b11; b11 = b13; b13 = -tb1; let tb2 = b21; b21 = b23; b23 = -tb2; let tb3 = b31; b31 = b33; b33 = -tb3;
        let tv = V[0]; V[0] = V[2]; V[2] = -tv;
        let tr = rho.x; rho.x = rho.z; rho.z = tr;
    }
    c = rho.y < rho.z;
    if (c) {
        let tb1 = b12; b12 = b13; b13 = -tb1; let tb2 = b22; b22 = b23; b23 = -tb2; let tb3 = b32; b32 = b33; b33 = -tb3;
        let tv = V[1]; V[1] = V[2]; V[2] = -tv;
    }

    // QR of B via three Givens rotations
    let g1 = svdQrGivens(b11, b21);
    var aa = 1.0 - 2.0 * g1.y * g1.y;
    var bb = 2.0 * g1.x * g1.y;
    var r11 = aa * b11 + bb * b21; var r12 = aa * b12 + bb * b22; var r13 = aa * b13 + bb * b23;
    var r21 = -bb * b11 + aa * b21; var r22 = -bb * b12 + aa * b22; var r23 = -bb * b13 + aa * b23;
    var r31 = b31; var r32 = b32; var r33 = b33;

    let g2 = svdQrGivens(r11, r31);
    aa = 1.0 - 2.0 * g2.y * g2.y;
    bb = 2.0 * g2.x * g2.y;
    b11 = aa * r11 + bb * r31; b12 = aa * r12 + bb * r32; b13 = aa * r13 + bb * r33;
    b21 = r21; b22 = r22; b23 = r23;
    b31 = -bb * r11 + aa * r31; b32 = -bb * r12 + aa * r32; b33 = -bb * r13 + aa * r33;

    let g3 = svdQrGivens(b22, b32);
    aa = 1.0 - 2.0 * g3.y * g3.y;
    bb = 2.0 * g3.x * g3.y;
    r11 = b11; r12 = b12; r13 = b13;
    r21 = aa * b21 + bb * b31; r22 = aa * b22 + bb * b32; r23 = aa * b23 + bb * b33;
    r31 = -bb * b21 + aa * b31; r32 = -bb * b22 + aa * b32; r33 = -bb * b23 + aa * b33;

    // Q = Q1·Q2·Q3
    let sh12 = g1.y * g1.y; let sh22 = g2.y * g2.y; let sh32 = g3.y * g3.y;
    let q11 = (-1.0 + 2.0 * sh12) * (-1.0 + 2.0 * sh22);
    let q12 = 4.0 * g2.x * g3.x * (-1.0 + 2.0 * sh12) * g2.y * g3.y + 2.0 * g1.x * g1.y * (-1.0 + 2.0 * sh32);
    let q13 = 4.0 * g1.x * g3.x * g1.y * g3.y - 2.0 * g2.x * (-1.0 + 2.0 * sh12) * g2.y * (-1.0 + 2.0 * sh32);
    let q21 = 2.0 * g1.x * g1.y * (1.0 - 2.0 * sh22);
    let q22 = -8.0 * g1.x * g2.x * g3.x * g1.y * g2.y * g3.y + (-1.0 + 2.0 * sh12) * (-1.0 + 2.0 * sh32);
    let q23 = -2.0 * g3.x * g3.y + 4.0 * g1.y * (g3.x * g1.y * g3.y + g1.x * g2.x * g2.y * (-1.0 + 2.0 * sh32));
    let q31 = 2.0 * g2.x * g2.y;
    let q32 = 2.0 * g3.x * (1.0 - 2.0 * sh22) * g3.y;
    let q33 = (-1.0 + 2.0 * sh22) * (-1.0 + 2.0 * sh32);

    var out: Svd3;
    out.U = mat3x3<f32>(vec3<f32>(q11, q21, q31), vec3<f32>(q12, q22, q32), vec3<f32>(q13, q23, q33));
    out.S = vec3<f32>(r11, r22, r33);
    out.V = V;
    return out;
}`;
