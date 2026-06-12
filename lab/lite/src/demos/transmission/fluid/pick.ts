// Demo-local mouse → capsule-surface picking for the fluid tank.
//
// Kept out of babylon-lite core: the demo unprojects the cursor through the
// camera's view-projection matrix and analytically intersects the resulting
// ray with the capsule tank, returning the world-space surface hit so a hole
// can be opened exactly where the user clicks. (babylon-lite exposes
// getViewProjectionMatrix but not its picking-ray helper, so the small amount
// of maths needed lives here.)

type Vec3 = [number, number, number];

/** Inverse of a 4×4 column-major matrix, or null if singular. */
function mat4Invert(m: ArrayLike<number>): number[] | null {
    const a00 = m[0]!, a01 = m[1]!, a02 = m[2]!, a03 = m[3]!;
    const a10 = m[4]!, a11 = m[5]!, a12 = m[6]!, a13 = m[7]!;
    const a20 = m[8]!, a21 = m[9]!, a22 = m[10]!, a23 = m[11]!;
    const a30 = m[12]!, a31 = m[13]!, a32 = m[14]!, a33 = m[15]!;

    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (Math.abs(det) < 1e-12) {
        return null;
    }
    det = 1 / det;

    return [
        (a11 * b11 - a12 * b10 + a13 * b09) * det,
        (a02 * b10 - a01 * b11 - a03 * b09) * det,
        (a31 * b05 - a32 * b04 + a33 * b03) * det,
        (a22 * b04 - a21 * b05 - a23 * b03) * det,
        (a12 * b08 - a10 * b11 - a13 * b07) * det,
        (a00 * b11 - a02 * b08 + a03 * b07) * det,
        (a32 * b02 - a30 * b05 - a33 * b01) * det,
        (a20 * b05 - a22 * b02 + a23 * b01) * det,
        (a10 * b10 - a11 * b08 + a13 * b06) * det,
        (a01 * b08 - a00 * b10 - a03 * b06) * det,
        (a30 * b04 - a31 * b02 + a33 * b00) * det,
        (a21 * b02 - a20 * b04 - a23 * b00) * det,
        (a11 * b07 - a10 * b09 - a12 * b06) * det,
        (a00 * b09 - a01 * b07 + a02 * b06) * det,
        (a31 * b01 - a30 * b03 - a32 * b00) * det,
        (a20 * b03 - a21 * b01 + a22 * b00) * det,
    ];
}

/** Unproject a clip-space point (NDC + depth) through an inverse VP matrix. */
function unproject(inv: number[], ndcX: number, ndcY: number, depth: number): Vec3 {
    const x = inv[0]! * ndcX + inv[4]! * ndcY + inv[8]! * depth + inv[12]!;
    const y = inv[1]! * ndcX + inv[5]! * ndcY + inv[9]! * depth + inv[13]!;
    const z = inv[2]! * ndcX + inv[6]! * ndcY + inv[10]! * depth + inv[14]!;
    const w = inv[3]! * ndcX + inv[7]! * ndcY + inv[11]! * depth + inv[15]!;
    const iw = 1 / w;
    return [x * iw, y * iw, z * iw];
}

/** Nearest positive ray–capsule intersection distance (Inigo Quilez), or -1.
 *  `rd` must be normalised. */
function capIntersect(ro: Vec3, rd: Vec3, pa: Vec3, pb: Vec3, r: number): number {
    const bax = pb[0] - pa[0], bay = pb[1] - pa[1], baz = pb[2] - pa[2];
    const oax = ro[0] - pa[0], oay = ro[1] - pa[1], oaz = ro[2] - pa[2];
    const baba = bax * bax + bay * bay + baz * baz;
    const bard = bax * rd[0] + bay * rd[1] + baz * rd[2];
    const baoa = bax * oax + bay * oay + baz * oaz;
    const rdoa = rd[0] * oax + rd[1] * oay + rd[2] * oaz;
    const oaoa = oax * oax + oay * oay + oaz * oaz;

    const a = baba - bard * bard;
    let b = baba * rdoa - baoa * bard;
    let c = baba * oaoa - baoa * baoa - r * r * baba;
    let h = b * b - a * c;
    if (h >= 0 && Math.abs(a) > 1e-12) {
        const t = (-b - Math.sqrt(h)) / a;
        const y = baoa + t * bard;
        if (y > 0 && y < baba && t > 0) {
            return t; // cylinder body
        }
        // Cap sphere (whichever end the cylinder hit fell beyond).
        const ocx = y <= 0 ? oax : ro[0] - pb[0];
        const ocy = y <= 0 ? oay : ro[1] - pb[1];
        const ocz = y <= 0 ? oaz : ro[2] - pb[2];
        b = rd[0] * ocx + rd[1] * ocy + rd[2] * ocz;
        c = ocx * ocx + ocy * ocy + ocz * ocz - r * r;
        h = b * b - c;
        if (h > 0) {
            const t = -b - Math.sqrt(h);
            if (t > 0) {
                return t;
            }
        }
    }
    return -1;
}

/** Cast the cursor into the scene and return the world-space point where it
 *  hits the capsule tank (centre segment a→b, radius r), or null on a miss. */
export function pickCapsuleHole(
    vp: ArrayLike<number>,
    x: number,
    y: number,
    width: number,
    height: number,
    a: Vec3,
    b: Vec3,
    r: number,
): Vec3 | null {
    const inv = mat4Invert(vp);
    if (!inv) {
        return null;
    }
    const ndcX = (2 * x) / width - 1;
    const ndcY = 1 - (2 * y) / height; // WebGPU: Y flipped
    // Reverse-Z: near plane maps to depth 1, far plane to 0.
    const near = unproject(inv, ndcX, ndcY, 1);
    const far = unproject(inv, ndcX, ndcY, 0);
    let dx = far[0] - near[0], dy = far[1] - near[1], dz = far[2] - near[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-10) {
        return null;
    }
    dx /= len;
    dy /= len;
    dz /= len;
    const t = capIntersect(near, [dx, dy, dz], a, b, r);
    if (t < 0) {
        return null;
    }
    return [near[0] + dx * t, near[1] + dy * t, near[2] + dz * t];
}
