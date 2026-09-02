// Per-particle UV lookup for mesh-coloured fluid.
//
// A liquefied mesh's particles are sampled INSIDE its volume, so each particle needs the UV of the
// mesh surface point closest to it. Snapping to the nearest VERTEX is only acceptable on dense
// meshes: on low-poly, atlas-mapped geometry (a sci-fi kit door is ~38 vertices whose UVs all sit on
// the border of its atlas island) every interior particle would inherit an island-CORNER texel, so
// the water reads as a flat trim colour instead of the panel it came from.
//
// Instead, find the closest TRIANGLE and interpolate its UVs with the barycentric coordinates of the
// closest point on it. A uniform grid is a poor fit here — particles fill the volume, so a particle
// deep inside a hollow prop sits many cells away from any triangle and an expanding-shell search
// degenerates to O(r³) cell visits. A median-split BVH with branch-and-bound pruning is distance-
// independent, so a 50k-triangle mesh with 120k particles stays well under a second.

/** Inputs for closest-surface UV transfer. Particle and mesh positions must use the same space. */
export interface FluidParticleUvTransferOptions {
    /** Flat xyz particle positions. */
    particles: Float32Array;
    /** Number of particles to process. */
    count: number;
    /** Flat xyz mesh vertex positions. */
    positions: Float32Array;
    /** Triangle indices into `positions`. */
    indices: Uint32Array;
    /** Per-vertex UVs aligned with `positions`. */
    uvs: Float32Array;
    /** Optional per-vertex texture indices. */
    texIndices?: Uint32Array | null;
}

/** Per-particle UVs and closest-triangle texture indices. */
export interface FluidParticleUvTransferResult {
    uvs: Float32Array;
    texIndices: Uint32Array;
}

const LEAF_TRIS = 4;

/**
 * For each particle, the UV of the closest point on the mesh surface.
 *
 * @param options - particles, source geometry, and per-vertex UV data.
 */
export function transferFluidMeshParticleUvs(options: FluidParticleUvTransferOptions): FluidParticleUvTransferResult {
    const { particles, count, positions: verts, indices, uvs: vertUvs } = options;
    const vertTex = options.texIndices ?? null;
    const uvs = new Float32Array(count * 2);
    const texIndices = new Uint32Array(count).fill(0xffffffff);
    const nTri = Math.floor(indices.length / 3);
    if (nTri === 0 || verts.length === 0) {
        return { uvs, texIndices };
    }

    const bvh = buildTriangleBvh(verts, indices, nTri);
    const bary = new Float32Array(3);
    const stack = new Int32Array(256);
    // Volume-sampled particles arrive in grid order, so consecutive queries are spatially coherent:
    // seeding the search with the previous particle's triangle gives a tight bound up front and
    // prunes nearly the whole tree.
    let seed = -1;

    for (let p = 0; p < count; p++) {
        const best = closestTriangle(bvh, verts, indices, particles[p * 3]!, particles[p * 3 + 1]!, particles[p * 3 + 2]!, bary, stack, seed);
        if (best < 0) {
            continue;
        }
        seed = best;
        const ia = indices[best * 3]!,
            ib = indices[best * 3 + 1]!,
            ic = indices[best * 3 + 2]!;
        const bu = bary[0]!,
            bv = bary[1]!,
            bw = bary[2]!;
        uvs[p * 2] = vertUvs[ia * 2]! * bu + vertUvs[ib * 2]! * bv + vertUvs[ic * 2]! * bw;
        uvs[p * 2 + 1] = vertUvs[ia * 2 + 1]! * bu + vertUvs[ib * 2 + 1]! * bv + vertUvs[ic * 2 + 1]! * bw;
        if (vertTex) {
            texIndices[p] = vertTex[ia]!;
        }
    }
    return { uvs, texIndices };
}

/** Flat-array BVH: `order` is the triangle permutation, leaves own the range [start, start+n). */
interface TriBvh {
    order: Uint32Array;
    nodeMin: Float32Array;
    nodeMax: Float32Array;
    nodeStart: Uint32Array;
    nodeCount: Uint32Array; // 0 => internal node, and nodeRight holds the right child
    nodeRight: Uint32Array; // left child is always the node index + 1
}

function buildTriangleBvh(verts: Float32Array, indices: Uint32Array, nTri: number): TriBvh {
    const triMin = new Float32Array(nTri * 3);
    const triMax = new Float32Array(nTri * 3);
    const centroid = new Float32Array(nTri * 3);
    for (let t = 0; t < nTri; t++) {
        const ia = indices[t * 3]! * 3,
            ib = indices[t * 3 + 1]! * 3,
            ic = indices[t * 3 + 2]! * 3;
        for (let k = 0; k < 3; k++) {
            const va = verts[ia + k]!,
                vb = verts[ib + k]!,
                vc = verts[ic + k]!;
            triMin[t * 3 + k] = Math.min(va, vb, vc);
            triMax[t * 3 + k] = Math.max(va, vb, vc);
            centroid[t * 3 + k] = (va + vb + vc) / 3;
        }
    }

    const maxNodes = Math.max(2, 2 * nTri);
    const order = new Uint32Array(nTri);
    for (let t = 0; t < nTri; t++) {
        order[t] = t;
    }
    const nodeMin = new Float32Array(maxNodes * 3);
    const nodeMax = new Float32Array(maxNodes * 3);
    const nodeStart = new Uint32Array(maxNodes);
    const nodeCount = new Uint32Array(maxNodes);
    const nodeRight = new Uint32Array(maxNodes);
    let used = 1;

    const fitBounds = (node: number): void => {
        let x0 = Infinity,
            y0 = Infinity,
            z0 = Infinity,
            x1 = -Infinity,
            y1 = -Infinity,
            z1 = -Infinity;
        const s = nodeStart[node]!;
        const e = s + nodeCount[node]!;
        for (let i = s; i < e; i++) {
            const t = order[i]! * 3;
            if (triMin[t]! < x0) {
                x0 = triMin[t]!;
            }
            if (triMin[t + 1]! < y0) {
                y0 = triMin[t + 1]!;
            }
            if (triMin[t + 2]! < z0) {
                z0 = triMin[t + 2]!;
            }
            if (triMax[t]! > x1) {
                x1 = triMax[t]!;
            }
            if (triMax[t + 1]! > y1) {
                y1 = triMax[t + 1]!;
            }
            if (triMax[t + 2]! > z1) {
                z1 = triMax[t + 2]!;
            }
        }
        nodeMin[node * 3] = x0;
        nodeMin[node * 3 + 1] = y0;
        nodeMin[node * 3 + 2] = z0;
        nodeMax[node * 3] = x1;
        nodeMax[node * 3 + 1] = y1;
        nodeMax[node * 3 + 2] = z1;
    };

    nodeStart[0] = 0;
    nodeCount[0] = nTri;
    fitBounds(0);

    // Explicit stack: mean-of-centroids split on the widest axis, falling back to a half-count split
    // when every centroid lands on one side (co-planar fans), so the recursion always terminates.
    const todo: number[] = [0];
    while (todo.length) {
        const node = todo.pop()!;
        const n = nodeCount[node]!;
        if (n <= LEAF_TRIS) {
            continue;
        }
        const s = nodeStart[node]!;
        const ex = nodeMax[node * 3]! - nodeMin[node * 3]!;
        const ey = nodeMax[node * 3 + 1]! - nodeMin[node * 3 + 1]!;
        const ez = nodeMax[node * 3 + 2]! - nodeMin[node * 3 + 2]!;
        const axis = ex > ey ? (ex > ez ? 0 : 2) : ey > ez ? 1 : 2;
        let mean = 0;
        for (let i = s; i < s + n; i++) {
            mean += centroid[order[i]! * 3 + axis]!;
        }
        mean /= n;
        let i = s;
        let j = s + n - 1;
        while (i <= j) {
            if (centroid[order[i]! * 3 + axis]! < mean) {
                i++;
            } else {
                const tmp = order[i]!;
                order[i] = order[j]!;
                order[j] = tmp;
                j--;
            }
        }
        let leftN = i - s;
        if (leftN === 0 || leftN === n) {
            leftN = n >> 1;
        }
        const left = used++;
        const right = used++;
        nodeStart[left] = s;
        nodeCount[left] = leftN;
        nodeStart[right] = s + leftN;
        nodeCount[right] = n - leftN;
        fitBounds(left);
        fitBounds(right);
        nodeCount[node] = 0;
        nodeRight[node] = right;
        todo.push(left, right);
    }
    return { order, nodeMin, nodeMax, nodeStart, nodeCount, nodeRight };
}

/** Index of the triangle closest to (px, py, pz); writes its barycentric weights into `bary`.
 *  `seedTri` (or -1) is an optional first candidate used only to tighten the initial pruning bound. */
function closestTriangle(
    bvh: TriBvh,
    verts: Float32Array,
    indices: Uint32Array,
    px: number,
    py: number,
    pz: number,
    bary: Float32Array,
    stack: Int32Array,
    seedTri: number
): number {
    const { order, nodeMin, nodeMax, nodeStart, nodeCount, nodeRight } = bvh;
    let sp = 0;
    stack[sp++] = 0;
    let best = -1;
    let bestD2 = Infinity;
    let bu = 0,
        bv = 0,
        bw = 0;
    if (seedTri >= 0) {
        bestD2 = closestPointOnTriangle(px, py, pz, verts, indices[seedTri * 3]! * 3, indices[seedTri * 3 + 1]! * 3, indices[seedTri * 3 + 2]! * 3, bary);
        best = seedTri;
        bu = bary[0]!;
        bv = bary[1]!;
        bw = bary[2]!;
    }
    while (sp > 0) {
        const node = stack[--sp]!;
        if (aabbDist2(nodeMin, nodeMax, node, px, py, pz) >= bestD2) {
            continue;
        }
        const n = nodeCount[node]!;
        if (n === 0) {
            // Push the farther child first so the nearer one is popped (and prunes) first.
            const l = node + 1;
            const r = nodeRight[node]!;
            if (aabbDist2(nodeMin, nodeMax, l, px, py, pz) < aabbDist2(nodeMin, nodeMax, r, px, py, pz)) {
                stack[sp++] = r;
                stack[sp++] = l;
            } else {
                stack[sp++] = l;
                stack[sp++] = r;
            }
            continue;
        }
        const s = nodeStart[node]!;
        for (let i = s; i < s + n; i++) {
            const t = order[i]!;
            const d2 = closestPointOnTriangle(px, py, pz, verts, indices[t * 3]! * 3, indices[t * 3 + 1]! * 3, indices[t * 3 + 2]! * 3, bary);
            if (d2 < bestD2) {
                bestD2 = d2;
                best = t;
                bu = bary[0]!;
                bv = bary[1]!;
                bw = bary[2]!;
            }
        }
    }
    bary[0] = bu;
    bary[1] = bv;
    bary[2] = bw;
    return best;
}

function aabbDist2(nodeMin: Float32Array, nodeMax: Float32Array, node: number, px: number, py: number, pz: number): number {
    const o = node * 3;
    const dx = Math.max(nodeMin[o]! - px, 0, px - nodeMax[o]!);
    const dy = Math.max(nodeMin[o + 1]! - py, 0, py - nodeMax[o + 1]!);
    const dz = Math.max(nodeMin[o + 2]! - pz, 0, pz - nodeMax[o + 2]!);
    return dx * dx + dy * dy + dz * dz;
}

/** Squared distance from p to triangle (a, b, c), writing the closest point's barycentric weights
 *  (for a, b, c) into `out`. Ericson, Real-Time Collision Detection §5.1.5 — the Voronoi-region
 *  form, so vertex/edge/face cases are all handled without a projection fallback. */
function closestPointOnTriangle(px: number, py: number, pz: number, v: Float32Array, ai: number, bi: number, ci: number, out: Float32Array): number {
    const ax = v[ai]!,
        ay = v[ai + 1]!,
        az = v[ai + 2]!;
    const bx = v[bi]!,
        by = v[bi + 1]!,
        bz = v[bi + 2]!;
    const cx = v[ci]!,
        cy = v[ci + 1]!,
        cz = v[ci + 2]!;
    const abx = bx - ax,
        aby = by - ay,
        abz = bz - az;
    const acx = cx - ax,
        acy = cy - ay,
        acz = cz - az;
    const apx = px - ax,
        apy = py - ay,
        apz = pz - az;
    const d1 = abx * apx + aby * apy + abz * apz;
    const d2 = acx * apx + acy * apy + acz * apz;
    let u = 1,
        vv = 0,
        w = 0;
    if (d1 <= 0 && d2 <= 0) {
        u = 1;
        vv = 0;
        w = 0;
    } else {
        const bpx = px - bx,
            bpy = py - by,
            bpz = pz - bz;
        const d3 = abx * bpx + aby * bpy + abz * bpz;
        const d4 = acx * bpx + acy * bpy + acz * bpz;
        const cpx = px - cx,
            cpy = py - cy,
            cpz = pz - cz;
        const d5 = abx * cpx + aby * cpy + abz * cpz;
        const d6 = acx * cpx + acy * cpy + acz * cpz;
        const vc = d1 * d4 - d3 * d2;
        const vb = d5 * d2 - d1 * d6;
        const va = d3 * d6 - d5 * d4;
        if (d3 >= 0 && d4 <= d3) {
            u = 0;
            vv = 1;
            w = 0;
        } else if (d6 >= 0 && d5 <= d6) {
            u = 0;
            vv = 0;
            w = 1;
        } else if (vc <= 0 && d1 >= 0 && d3 <= 0) {
            const t = d1 / (d1 - d3);
            u = 1 - t;
            vv = t;
            w = 0;
        } else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            const t = d2 / (d2 - d6);
            u = 1 - t;
            vv = 0;
            w = t;
        } else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
            const t = (d4 - d3) / (d4 - d3 + (d5 - d6));
            u = 0;
            vv = 1 - t;
            w = t;
        } else {
            const denom = 1 / (va + vb + vc);
            vv = vb * denom;
            w = vc * denom;
            u = 1 - vv - w;
        }
    }
    out[0] = u;
    out[1] = vv;
    out[2] = w;
    const qx = ax + abx * vv + acx * w - px;
    const qy = ay + aby * vv + acy * w - py;
    const qz = az + abz * vv + acz * w - pz;
    return qx * qx + qy * qy + qz * qz;
}
