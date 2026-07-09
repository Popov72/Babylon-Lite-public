// Exact signed distance to a triangle mesh + a sampled trilinear signed-distance grid.
//
// Port of the SPlisHSPlasH VolumeSampling SDF construction (Tools/VolumeSampling).
// The unsigned distance is the min point-to-triangle distance; the sign comes from
// angle-weighted pseudonormals (Baerentzen & Aanaes 2002 — the Discregrid
// `TriangleMeshDistance` algorithm):
//   • closest point on a FACE   -> face normal
//   • closest point on an EDGE  -> normalized sum of the two adjacent face normals
//   • closest point on a VERTEX -> sum of adjacent face normals, each weighted by that
//                                  triangle's interior angle at the vertex, normalized
// Sign: `u = p - closest; sign = dot(u, pseudonormal) >= 0 ? +1 : -1`
// Convention (matches TriangleMeshDistance): POSITIVE outside, NEGATIVE inside.
//
// `buildSignedDistanceGrid` samples the exact UNSIGNED distance on a (res+1)^3 trilinear grid and
// stores it INVERTED (inside POSITIVE by default), so the downstream `value > 0 == inside`
// test used by the lattice/SPH samplers holds. This replaces the 32-node cubic-Lagrange
// grid of the original with a trilinear grid: simpler, correct for the inside test and the
// boundary gradient, and a 1:1 match for the eventual GPU 3D-texture representation.
//
// The grid's inside/outside SIGN does NOT use the pseudonormals (those are only reliable for
// CLOSED manifolds); it uses the generalized (fast) WINDING NUMBER (Barill et al. 2018,
// BVH-accelerated) so OPEN / non-manifold meshes no longer generate spurious exterior "inside"
// regions. The pseudonormal path still supplies the smooth distance MAGNITUDE + gradient.
//
// Pure CPU — no GPU handles. Zero module-level side effects (all state is created lazily
// inside the factory functions below).

/** Result of a single exact signed-distance query. `distance` is +outside / -inside; `cx/cy/cz` is the closest surface point. */
export interface SignedDistanceResult {
    distance: number;
    cx: number;
    cy: number;
    cz: number;
}

/** Exact signed-distance evaluator over a triangle mesh, accelerated by a median-split AABB BVH. */
export interface MeshDistance {
    /** Exact signed distance from (px,py,pz) to the mesh surface (BVH-accelerated). */
    signedDistance(px: number, py: number, pz: number): SignedDistanceResult;
    /** Brute-force reference (iterates every triangle). Used to cross-check the BVH in tests. */
    signedDistanceBrute(px: number, py: number, pz: number): SignedDistanceResult;
    /**
     * Generalized (fast) winding number at (px,py,pz), Barill et al. 2018. ~±1 inside a closed
     * region, ~0 outside; the SIGN is robust for OPEN / non-manifold meshes where pseudonormals
     * are not. BVH-accelerated: far clusters use an order-1 dipole approximation, near leaves sum
     * exact per-triangle solid angles.
     */
    windingNumber(px: number, py: number, pz: number): number;
    /** Brute-force winding number (exact solid angle summed over ALL triangles). Cross-checks the fast path in tests. */
    windingNumberBrute(px: number, py: number, pz: number): number;
}

/** A trilinear signed-distance grid sampled over a padded domain box. Inside is POSITIVE (unless built with `invert`). */
export interface SignedDistanceGrid {
    /** Cells per axis (node count per axis is res+1). */
    readonly res: [number, number, number];
    /** World-space minimum corner of the domain box. */
    readonly domainMin: [number, number, number];
    /** World-space maximum corner of the domain box. */
    readonly domainMax: [number, number, number];
    /** World-space size of one grid cell per axis. */
    readonly cellSize: [number, number, number];
    /** Node values in x-fastest / z-slowest order: index = k*(nx+1)*(ny+1) + j*(nx+1) + i. */
    readonly data: Float32Array;
    /** Trilinear value at (px,py,pz); `Infinity` if the point is outside the domain box. */
    interpolate(px: number, py: number, pz: number): number;
    /** Trilinear value + analytic gradient of the trilinear field; value `Infinity` and zero gradient if outside the domain. */
    interpolateWithGradient(px: number, py: number, pz: number): GradientResult;
    /** Mirrors the C++ `distance(x, tol, normal, cp)`: signed distance minus `tolerance`, the surface normal, and the closest surface point along the gradient. */
    distanceWithNormal(px: number, py: number, pz: number, tolerance: number): DistanceNormalResult;
}

/** Trilinear value and its analytic world-space gradient. */
export interface GradientResult {
    value: number;
    gx: number;
    gy: number;
    gz: number;
}

/** Signed distance (minus tolerance), unit surface normal, and closest surface point. */
export interface DistanceNormalResult {
    dist: number;
    nx: number;
    ny: number;
    nz: number;
    cx: number;
    cy: number;
    cz: number;
}

// Feature codes for the closest point on a triangle.
const FEAT_FACE = 0;
const FEAT_V0 = 1;
const FEAT_V1 = 2;
const FEAT_V2 = 3;
const FEAT_E01 = 4; // edge (v0,v1) — triangle edge index 0
const FEAT_E12 = 5; // edge (v1,v2) — triangle edge index 1
const FEAT_E20 = 6; // edge (v2,v0) — triangle edge index 2

const LEAF_TRIANGLES = 4;

/**
 * Build an exact signed-distance evaluator for the given indexed triangle mesh.
 *
 * @param positions - flat xyz vertex positions.
 * @param indices - flat triangle indices (three per triangle).
 * @returns a `MeshDistance` with BVH-accelerated and brute-force query paths.
 */
export function buildMeshDistance(positions: Float32Array, indices: Uint32Array | Uint16Array): MeshDistance {
    const numTris = (indices.length / 3) | 0;
    const numVerts = (positions.length / 3) | 0;

    // Per-triangle face normals (normalized) + centroids + AABBs.
    const faceNormals = new Float32Array(numTris * 3);
    const triMin = new Float32Array(numTris * 3);
    const triMax = new Float32Array(numTris * 3);
    const centroid = new Float32Array(numTris * 3);

    // Per-triangle area-weighted normals `aN_t = 0.5*cross(v1-v0, v2-v0)` (magnitude = area,
    // direction = geometric normal; NOT normalized) and their magnitudes (triangle areas). These
    // feed the fast-winding-number BVH aggregates below.
    const triAN = new Float32Array(numTris * 3);
    const triArea = new Float32Array(numTris);

    // Angle-weighted per-vertex pseudonormals (accumulated below).
    const vertexPN = new Float32Array(numVerts * 3);

    // Edge pseudonormals: one slot per unique undirected edge; `triEdgeSlot[3t+e]` maps a
    // triangle edge (e = 0,1,2) to its slot. Each slot accumulates the adjacent face normals.
    const edgeMap = new Map<number, number>();
    const edgeAccum: number[] = [];
    const triEdgeSlot = new Int32Array(numTris * 3);

    const edgeSlot = (a: number, b: number): number => {
        const lo = a < b ? a : b;
        const hi = a < b ? b : a;
        const key = lo * numVerts + hi;
        let slot = edgeMap.get(key);
        if (slot === undefined) {
            slot = edgeAccum.length / 3;
            edgeMap.set(key, slot);
            edgeAccum.push(0, 0, 0);
        }
        return slot;
    };

    for (let t = 0; t < numTris; t++) {
        const i0 = indices[3 * t]!;
        const i1 = indices[3 * t + 1]!;
        const i2 = indices[3 * t + 2]!;
        const ax = positions[3 * i0]!;
        const ay = positions[3 * i0 + 1]!;
        const az = positions[3 * i0 + 2]!;
        const bx = positions[3 * i1]!;
        const by = positions[3 * i1 + 1]!;
        const bz = positions[3 * i1 + 2]!;
        const cx = positions[3 * i2]!;
        const cy = positions[3 * i2 + 1]!;
        const cz = positions[3 * i2 + 2]!;

        // Face normal = normalize(cross(b-a, c-a)).
        const abx = bx - ax;
        const aby = by - ay;
        const abz = bz - az;
        const acx = cx - ax;
        const acy = cy - ay;
        const acz = cz - az;
        let nx = aby * acz - abz * acy;
        let ny = abz * acx - abx * acz;
        let nz = abx * acy - aby * acx;
        const nlen = Math.hypot(nx, ny, nz);
        // Area-weighted normal (0.5*cross) captured BEFORE normalization; |aN| = triangle area.
        triAN[3 * t] = 0.5 * nx;
        triAN[3 * t + 1] = 0.5 * ny;
        triAN[3 * t + 2] = 0.5 * nz;
        triArea[t] = 0.5 * nlen;
        if (nlen > 0) {
            nx /= nlen;
            ny /= nlen;
            nz /= nlen;
        }
        faceNormals[3 * t] = nx;
        faceNormals[3 * t + 1] = ny;
        faceNormals[3 * t + 2] = nz;

        // Centroid + AABB.
        centroid[3 * t] = (ax + bx + cx) / 3;
        centroid[3 * t + 1] = (ay + by + cy) / 3;
        centroid[3 * t + 2] = (az + bz + cz) / 3;
        triMin[3 * t] = Math.min(ax, bx, cx);
        triMin[3 * t + 1] = Math.min(ay, by, cy);
        triMin[3 * t + 2] = Math.min(az, bz, cz);
        triMax[3 * t] = Math.max(ax, bx, cx);
        triMax[3 * t + 1] = Math.max(ay, by, cy);
        triMax[3 * t + 2] = Math.max(az, bz, cz);

        // Interior angles at the three vertices (between the two edges leaving each vertex).
        const a0 = vertexAngle(ax, ay, az, bx, by, bz, cx, cy, cz); // at v0
        const a1 = vertexAngle(bx, by, bz, cx, cy, cz, ax, ay, az); // at v1
        const a2 = vertexAngle(cx, cy, cz, ax, ay, az, bx, by, bz); // at v2
        vertexPN[3 * i0]! += a0 * nx;
        vertexPN[3 * i0 + 1]! += a0 * ny;
        vertexPN[3 * i0 + 2]! += a0 * nz;
        vertexPN[3 * i1]! += a1 * nx;
        vertexPN[3 * i1 + 1]! += a1 * ny;
        vertexPN[3 * i1 + 2]! += a1 * nz;
        vertexPN[3 * i2]! += a2 * nx;
        vertexPN[3 * i2 + 1]! += a2 * ny;
        vertexPN[3 * i2 + 2]! += a2 * nz;

        // Edge pseudonormals: accumulate this face normal into each of the 3 edge slots.
        const s0 = edgeSlot(i0, i1);
        const s1 = edgeSlot(i1, i2);
        const s2 = edgeSlot(i2, i0);
        triEdgeSlot[3 * t] = s0;
        triEdgeSlot[3 * t + 1] = s1;
        triEdgeSlot[3 * t + 2] = s2;
        edgeAccum[3 * s0]! += nx;
        edgeAccum[3 * s0 + 1]! += ny;
        edgeAccum[3 * s0 + 2]! += nz;
        edgeAccum[3 * s1]! += nx;
        edgeAccum[3 * s1 + 1]! += ny;
        edgeAccum[3 * s1 + 2]! += nz;
        edgeAccum[3 * s2]! += nx;
        edgeAccum[3 * s2 + 1]! += ny;
        edgeAccum[3 * s2 + 2]! += nz;
    }

    // Normalize the accumulated per-vertex and per-edge pseudonormals (only the direction
    // matters for the sign test, but we normalize for faithfulness / numeric hygiene).
    normalizeVec3Buffer(vertexPN, numVerts);
    const edgeNormals = Float32Array.from(edgeAccum);
    normalizeVec3Buffer(edgeNormals, edgeNormals.length / 3);

    // ---- Median-split AABB BVH over the triangles. ----
    const maxNodes = Math.max(1, 2 * numTris);
    const nodeMin = new Float32Array(maxNodes * 3);
    const nodeMax = new Float32Array(maxNodes * 3);
    const nodeLeft = new Int32Array(maxNodes); // internal: left child index; leaf: first tri offset
    const nodeCount = new Int32Array(maxNodes); // 0 = internal node; >0 = leaf triangle count
    // Fast-winding-number per-node aggregates (filled bottom-up after the BVH is built):
    //   nodeN = Σ aN_t (area-weighted normal / "dipole"); nodeC = area-weighted centroid;
    //   nodeR = radius bounding the node's triangles from nodeC.
    const nodeN = new Float32Array(maxNodes * 3);
    const nodeC = new Float32Array(maxNodes * 3);
    const nodeR = new Float32Array(maxNodes);
    const nodeArea = new Float64Array(maxNodes); // Σ area_t (build-time weight for combining centroids).
    const triOrder = new Uint32Array(numTris);
    for (let i = 0; i < numTris; i++) {
        triOrder[i] = i;
    }
    let nodesUsed = 1;

    const computeBounds = (node: number, first: number, count: number): void => {
        let mnx = Infinity;
        let mny = Infinity;
        let mnz = Infinity;
        let mxx = -Infinity;
        let mxy = -Infinity;
        let mxz = -Infinity;
        for (let i = 0; i < count; i++) {
            const t = triOrder[first + i]!;
            mnx = Math.min(mnx, triMin[3 * t]!);
            mny = Math.min(mny, triMin[3 * t + 1]!);
            mnz = Math.min(mnz, triMin[3 * t + 2]!);
            mxx = Math.max(mxx, triMax[3 * t]!);
            mxy = Math.max(mxy, triMax[3 * t + 1]!);
            mxz = Math.max(mxz, triMax[3 * t + 2]!);
        }
        nodeMin[3 * node] = mnx;
        nodeMin[3 * node + 1] = mny;
        nodeMin[3 * node + 2] = mnz;
        nodeMax[3 * node] = mxx;
        nodeMax[3 * node + 1] = mxy;
        nodeMax[3 * node + 2] = mxz;
    };

    const subdivide = (node: number, first: number, count: number): void => {
        computeBounds(node, first, count);
        if (count <= LEAF_TRIANGLES) {
            nodeLeft[node] = first;
            nodeCount[node] = count;
            return;
        }
        // Split axis = longest extent of the node's AABB.
        const ex = nodeMax[3 * node]! - nodeMin[3 * node]!;
        const ey = nodeMax[3 * node + 1]! - nodeMin[3 * node + 1]!;
        const ez = nodeMax[3 * node + 2]! - nodeMin[3 * node + 2]!;
        const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;

        // Median split: sort this slice by centroid along `axis`, split in half.
        const slice = Array.from(triOrder.subarray(first, first + count));
        slice.sort((p, q) => centroid[3 * p + axis]! - centroid[3 * q + axis]!);
        triOrder.set(slice, first);

        const mid = count >> 1;
        const left = nodesUsed++;
        const right = nodesUsed++;
        nodeLeft[node] = left;
        nodeCount[node] = 0;
        subdivide(left, first, mid);
        subdivide(right, first + mid, count - mid);
    };

    if (numTris > 0) {
        subdivide(0, 0, numTris);
    }

    // ---- Fast-winding-number aggregates: fill nodeN / nodeC / nodeR bottom-up. ----
    // Leaf: sum the area-weighted normals and the area-weighted centroid over its triangles, then
    // bound the triangle vertices from that centroid. Internal: combine the two children (dipole =
    // sum; centroid = area-weighted mean; radius = max child |centroidOffset| + childRadius).
    const fillAggregates = (node: number): void => {
        const count = nodeCount[node]!;
        if (count > 0) {
            const first = nodeLeft[node]!;
            let nX = 0;
            let nY = 0;
            let nZ = 0;
            let sumA = 0;
            let cX = 0;
            let cY = 0;
            let cZ = 0;
            for (let i = 0; i < count; i++) {
                const t = triOrder[first + i]!;
                nX += triAN[3 * t]!;
                nY += triAN[3 * t + 1]!;
                nZ += triAN[3 * t + 2]!;
                const a = triArea[t]!;
                sumA += a;
                cX += a * centroid[3 * t]!;
                cY += a * centroid[3 * t + 1]!;
                cZ += a * centroid[3 * t + 2]!;
            }
            if (sumA > 0) {
                cX /= sumA;
                cY /= sumA;
                cZ /= sumA;
            } else {
                // Degenerate (zero-area) triangles: fall back to the plain centroid mean.
                cX = 0;
                cY = 0;
                cZ = 0;
                for (let i = 0; i < count; i++) {
                    const t = triOrder[first + i]!;
                    cX += centroid[3 * t]!;
                    cY += centroid[3 * t + 1]!;
                    cZ += centroid[3 * t + 2]!;
                }
                cX /= count;
                cY /= count;
                cZ /= count;
            }
            // Radius = max distance from the cluster centroid to any triangle vertex.
            let r2max = 0;
            for (let i = 0; i < count; i++) {
                const t = triOrder[first + i]!;
                for (let v = 0; v < 3; v++) {
                    const vi = indices[3 * t + v]!;
                    const dx = positions[3 * vi]! - cX;
                    const dy = positions[3 * vi + 1]! - cY;
                    const dz = positions[3 * vi + 2]! - cZ;
                    const r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 > r2max) {
                        r2max = r2;
                    }
                }
            }
            nodeN[3 * node] = nX;
            nodeN[3 * node + 1] = nY;
            nodeN[3 * node + 2] = nZ;
            nodeC[3 * node] = cX;
            nodeC[3 * node + 1] = cY;
            nodeC[3 * node + 2] = cZ;
            nodeR[node] = Math.sqrt(r2max);
            nodeArea[node] = sumA;
            return;
        }
        const left = nodeLeft[node]!;
        const right = left + 1;
        fillAggregates(left);
        fillAggregates(right);
        const aL = nodeArea[left]!;
        const aR = nodeArea[right]!;
        const sumA = aL + aR;
        let cX: number;
        let cY: number;
        let cZ: number;
        if (sumA > 0) {
            cX = (nodeC[3 * left]! * aL + nodeC[3 * right]! * aR) / sumA;
            cY = (nodeC[3 * left + 1]! * aL + nodeC[3 * right + 1]! * aR) / sumA;
            cZ = (nodeC[3 * left + 2]! * aL + nodeC[3 * right + 2]! * aR) / sumA;
        } else {
            cX = (nodeC[3 * left]! + nodeC[3 * right]!) * 0.5;
            cY = (nodeC[3 * left + 1]! + nodeC[3 * right + 1]!) * 0.5;
            cZ = (nodeC[3 * left + 2]! + nodeC[3 * right + 2]!) * 0.5;
        }
        // Conservative radius: a child's triangles lie within nodeR[child] of the child's centroid,
        // which lies within |childCentroid - c| of the combined centroid.
        const dLx = nodeC[3 * left]! - cX;
        const dLy = nodeC[3 * left + 1]! - cY;
        const dLz = nodeC[3 * left + 2]! - cZ;
        const rL = Math.hypot(dLx, dLy, dLz) + nodeR[left]!;
        const dRx = nodeC[3 * right]! - cX;
        const dRy = nodeC[3 * right + 1]! - cY;
        const dRz = nodeC[3 * right + 2]! - cZ;
        const rR = Math.hypot(dRx, dRy, dRz) + nodeR[right]!;
        nodeN[3 * node] = nodeN[3 * left]! + nodeN[3 * right]!;
        nodeN[3 * node + 1] = nodeN[3 * left + 1]! + nodeN[3 * right + 1]!;
        nodeN[3 * node + 2] = nodeN[3 * left + 2]! + nodeN[3 * right + 2]!;
        nodeC[3 * node] = cX;
        nodeC[3 * node + 1] = cY;
        nodeC[3 * node + 2] = cZ;
        nodeR[node] = rL > rR ? rL : rR;
        nodeArea[node] = sumA;
    };

    if (numTris > 0) {
        fillAggregates(0);
    }

    // ---- Query machinery (allocation-free inner loop). ----
    // Scratch for the closest point + feature of the most recent triangle test.
    const cp = { x: 0, y: 0, z: 0, feat: FEAT_FACE };
    const pn = { x: 0, y: 0, z: 0 };

    // Best-so-far state for the current query (reset per query; queries are not reentrant).
    let bestD2 = Infinity;
    let bestCx = 0;
    let bestCy = 0;
    let bestCz = 0;
    let bestFeat = FEAT_FACE;
    let bestTri = -1;

    const setClosest = (px: number, py: number, pz: number, qx: number, qy: number, qz: number, feat: number): number => {
        cp.x = qx;
        cp.y = qy;
        cp.z = qz;
        cp.feat = feat;
        const dx = px - qx;
        const dy = py - qy;
        const dz = pz - qz;
        return dx * dx + dy * dy + dz * dz;
    };

    // Squared distance from p to triangle `t`; writes the closest point + feature into `cp`.
    const pointTriDistSq = (px: number, py: number, pz: number, t: number): number => {
        const i0 = indices[3 * t]!;
        const i1 = indices[3 * t + 1]!;
        const i2 = indices[3 * t + 2]!;
        const ax = positions[3 * i0]!;
        const ay = positions[3 * i0 + 1]!;
        const az = positions[3 * i0 + 2]!;
        const bx = positions[3 * i1]!;
        const by = positions[3 * i1 + 1]!;
        const bz = positions[3 * i1 + 2]!;
        const cx = positions[3 * i2]!;
        const cy = positions[3 * i2 + 1]!;
        const cz = positions[3 * i2 + 2]!;

        const abx = bx - ax;
        const aby = by - ay;
        const abz = bz - az;
        const acx = cx - ax;
        const acy = cy - ay;
        const acz = cz - az;

        const apx = px - ax;
        const apy = py - ay;
        const apz = pz - az;
        const d1 = abx * apx + aby * apy + abz * apz;
        const d2 = acx * apx + acy * apy + acz * apz;
        if (d1 <= 0 && d2 <= 0) {
            return setClosest(px, py, pz, ax, ay, az, FEAT_V0);
        }

        const bpx = px - bx;
        const bpy = py - by;
        const bpz = pz - bz;
        const d3 = abx * bpx + aby * bpy + abz * bpz;
        const d4 = acx * bpx + acy * bpy + acz * bpz;
        if (d3 >= 0 && d4 <= d3) {
            return setClosest(px, py, pz, bx, by, bz, FEAT_V1);
        }

        const cpx = px - cx;
        const cpy = py - cy;
        const cpz = pz - cz;
        const d5 = abx * cpx + aby * cpy + abz * cpz;
        const d6 = acx * cpx + acy * cpy + acz * cpz;
        if (d6 >= 0 && d5 <= d6) {
            return setClosest(px, py, pz, cx, cy, cz, FEAT_V2);
        }

        const vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
            const v = d1 / (d1 - d3);
            return setClosest(px, py, pz, ax + v * abx, ay + v * aby, az + v * abz, FEAT_E01);
        }

        const vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            const w = d2 / (d2 - d6);
            return setClosest(px, py, pz, ax + w * acx, ay + w * acy, az + w * acz, FEAT_E20);
        }

        const va = d3 * d6 - d5 * d4;
        if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
            const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
            return setClosest(px, py, pz, bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz), FEAT_E12);
        }

        // Interior of the face.
        const denom = 1 / (va + vb + vc);
        const v = vb * denom;
        const w = vc * denom;
        return setClosest(px, py, pz, ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w, FEAT_FACE);
    };

    const aabbDistSq = (px: number, py: number, pz: number, node: number): number => {
        const x0 = nodeMin[3 * node]!;
        const y0 = nodeMin[3 * node + 1]!;
        const z0 = nodeMin[3 * node + 2]!;
        const x1 = nodeMax[3 * node]!;
        const y1 = nodeMax[3 * node + 1]!;
        const z1 = nodeMax[3 * node + 2]!;
        const dx = px < x0 ? x0 - px : px > x1 ? px - x1 : 0;
        const dy = py < y0 ? y0 - py : py > y1 ? py - y1 : 0;
        const dz = pz < z0 ? z0 - pz : pz > z1 ? pz - z1 : 0;
        return dx * dx + dy * dy + dz * dz;
    };

    const stackNode = new Int32Array(maxNodes + 1);
    const stackDist = new Float64Array(maxNodes + 1);

    const query = (px: number, py: number, pz: number): void => {
        bestD2 = Infinity;
        bestFeat = FEAT_FACE;
        bestTri = -1;
        bestCx = 0;
        bestCy = 0;
        bestCz = 0;
        if (numTris === 0) {
            return;
        }
        let sp = 0;
        stackNode[sp] = 0;
        stackDist[sp] = 0;
        sp++;
        while (sp > 0) {
            sp--;
            if (stackDist[sp]! >= bestD2) {
                continue;
            }
            const node = stackNode[sp]!;
            const count = nodeCount[node]!;
            if (count > 0) {
                const first = nodeLeft[node]!;
                for (let i = 0; i < count; i++) {
                    const t = triOrder[first + i]!;
                    const d2 = pointTriDistSq(px, py, pz, t);
                    if (d2 < bestD2) {
                        bestD2 = d2;
                        bestCx = cp.x;
                        bestCy = cp.y;
                        bestCz = cp.z;
                        bestFeat = cp.feat;
                        bestTri = t;
                    }
                }
            } else {
                const left = nodeLeft[node]!;
                const right = left + 1;
                const dL = aabbDistSq(px, py, pz, left);
                const dR = aabbDistSq(px, py, pz, right);
                // Push the farther child first so the nearer child is popped (visited) first.
                if (dL < dR) {
                    if (dR < bestD2) {
                        stackNode[sp] = right;
                        stackDist[sp] = dR;
                        sp++;
                    }
                    if (dL < bestD2) {
                        stackNode[sp] = left;
                        stackDist[sp] = dL;
                        sp++;
                    }
                } else {
                    if (dL < bestD2) {
                        stackNode[sp] = left;
                        stackDist[sp] = dL;
                        sp++;
                    }
                    if (dR < bestD2) {
                        stackNode[sp] = right;
                        stackDist[sp] = dR;
                        sp++;
                    }
                }
            }
        }
    };

    // Write the pseudonormal of feature `feat` on triangle `t` into `pn`.
    const pseudonormalFor = (t: number, feat: number): void => {
        if (feat === FEAT_FACE) {
            pn.x = faceNormals[3 * t]!;
            pn.y = faceNormals[3 * t + 1]!;
            pn.z = faceNormals[3 * t + 2]!;
        } else if (feat <= FEAT_V2) {
            const vi = indices[3 * t + (feat - FEAT_V0)]!;
            pn.x = vertexPN[3 * vi]!;
            pn.y = vertexPN[3 * vi + 1]!;
            pn.z = vertexPN[3 * vi + 2]!;
        } else {
            const slot = triEdgeSlot[3 * t + (feat - FEAT_E01)]!;
            pn.x = edgeNormals[3 * slot]!;
            pn.y = edgeNormals[3 * slot + 1]!;
            pn.z = edgeNormals[3 * slot + 2]!;
        }
    };

    const signFromClosest = (px: number, py: number, pz: number, qx: number, qy: number, qz: number): number => {
        const ux = px - qx;
        const uy = py - qy;
        const uz = pz - qz;
        return ux * pn.x + uy * pn.y + uz * pn.z >= 0 ? 1 : -1;
    };

    const signedDistance = (px: number, py: number, pz: number): SignedDistanceResult => {
        query(px, py, pz);
        if (bestTri < 0) {
            return { distance: Infinity, cx: px, cy: py, cz: pz };
        }
        pseudonormalFor(bestTri, bestFeat);
        const sign = signFromClosest(px, py, pz, bestCx, bestCy, bestCz);
        return { distance: Math.sqrt(bestD2) * sign, cx: bestCx, cy: bestCy, cz: bestCz };
    };

    const signedDistanceBrute = (px: number, py: number, pz: number): SignedDistanceResult => {
        let bD2 = Infinity;
        let bcx = 0;
        let bcy = 0;
        let bcz = 0;
        let bfeat = FEAT_FACE;
        let btri = -1;
        for (let t = 0; t < numTris; t++) {
            const d2 = pointTriDistSq(px, py, pz, t);
            if (d2 < bD2) {
                bD2 = d2;
                bcx = cp.x;
                bcy = cp.y;
                bcz = cp.z;
                bfeat = cp.feat;
                btri = t;
            }
        }
        if (btri < 0) {
            return { distance: Infinity, cx: px, cy: py, cz: pz };
        }
        pseudonormalFor(btri, bfeat);
        const sign = signFromClosest(px, py, pz, bcx, bcy, bcz);
        return { distance: Math.sqrt(bD2) * sign, cx: bcx, cy: bcy, cz: bcz };
    };

    // ---- Fast winding number (Barill et al. 2018), BVH-accelerated. ----
    const INV_FOUR_PI = 1 / (4 * Math.PI);
    // Accuracy parameter: recurse until the query is farther than BETA * nodeRadius from the
    // cluster centroid, then use the order-1 dipole approximation. 2.5 balances speed (30^3 grid
    // build ~180 ms for a 10k-tri mesh) vs accuracy (fast-vs-exact WN agree to <1e-2 away from the
    // surface); the grid inside-test only needs |w|>0.5, which is robust well below that.
    const BETA = 2.5;
    const BETA2 = BETA * BETA;

    // Signed solid angle subtended by triangle `t` at (px,py,pz) (Van Oosterom & Strackee).
    const solidAngle = (px: number, py: number, pz: number, t: number): number => {
        const i0 = indices[3 * t]!;
        const i1 = indices[3 * t + 1]!;
        const i2 = indices[3 * t + 2]!;
        const ax = positions[3 * i0]! - px;
        const ay = positions[3 * i0 + 1]! - py;
        const az = positions[3 * i0 + 2]! - pz;
        const bx = positions[3 * i1]! - px;
        const by = positions[3 * i1 + 1]! - py;
        const bz = positions[3 * i1 + 2]! - pz;
        const cx = positions[3 * i2]! - px;
        const cy = positions[3 * i2 + 1]! - py;
        const cz = positions[3 * i2 + 2]! - pz;
        const la = Math.hypot(ax, ay, az);
        const lb = Math.hypot(bx, by, bz);
        const lc = Math.hypot(cx, cy, cz);
        // det[a,b,c] = a · (b × c).
        const bcx = by * cz - bz * cy;
        const bcy = bz * cx - bx * cz;
        const bcz = bx * cy - by * cx;
        const det = ax * bcx + ay * bcy + az * bcz;
        const abDot = ax * bx + ay * by + az * bz;
        const bcDot = bx * cx + by * cy + bz * cz;
        const caDot = cx * ax + cy * ay + cz * az;
        const denom = la * lb * lc + abDot * lc + bcDot * la + caDot * lb;
        return 2 * Math.atan2(det, denom);
    };

    const wnStack = new Int32Array(maxNodes + 1);

    const windingNumber = (px: number, py: number, pz: number): number => {
        if (numTris === 0) {
            return 0;
        }
        let w = 0;
        let sp = 0;
        wnStack[sp++] = 0;
        while (sp > 0) {
            const node = wnStack[--sp]!;
            const dx = nodeC[3 * node]! - px;
            const dy = nodeC[3 * node + 1]! - py;
            const dz = nodeC[3 * node + 2]! - pz;
            const r2 = dx * dx + dy * dy + dz * dz;
            const R = nodeR[node]!;
            // Far field: query well outside the cluster -> order-1 dipole approximation, stop.
            if (r2 > BETA2 * R * R && r2 > 1e-24) {
                const r = Math.sqrt(r2);
                w += ((nodeN[3 * node]! * dx + nodeN[3 * node + 1]! * dy + nodeN[3 * node + 2]! * dz) * INV_FOUR_PI) / (r2 * r);
                continue;
            }
            const count = nodeCount[node]!;
            if (count > 0) {
                const first = nodeLeft[node]!;
                for (let i = 0; i < count; i++) {
                    w += solidAngle(px, py, pz, triOrder[first + i]!) * INV_FOUR_PI;
                }
            } else {
                const left = nodeLeft[node]!;
                wnStack[sp++] = left;
                wnStack[sp++] = left + 1;
            }
        }
        return w;
    };

    const windingNumberBrute = (px: number, py: number, pz: number): number => {
        let w = 0;
        for (let t = 0; t < numTris; t++) {
            w += solidAngle(px, py, pz, t);
        }
        return w * INV_FOUR_PI;
    };

    return { signedDistance, signedDistanceBrute, windingNumber, windingNumberBrute };
}

/**
 * Sample the exact signed distance onto a (res+1)^3 trilinear grid over `[domainMin, domainMax]`.
 * Stores the INVERTED field so inside is POSITIVE by default (`store -signedDistance` unless `invert`).
 *
 * @param dist - the exact mesh distance evaluator.
 * @param domainMin - world-space minimum corner of the domain box.
 * @param domainMax - world-space maximum corner of the domain box.
 * @param res - number of cells per axis.
 * @param invert - when true, store `+signedDistance` (inside becomes negative).
 * @returns a `SignedDistanceGrid` with trilinear interpolation + gradient helpers.
 */
export function buildSignedDistanceGrid(
    dist: MeshDistance,
    domainMin: [number, number, number],
    domainMax: [number, number, number],
    res: [number, number, number],
    invert: boolean
): SignedDistanceGrid {
    const nx = res[0];
    const ny = res[1];
    const nz = res[2];
    const nvx = nx + 1;
    const nvy = ny + 1;
    const nvz = nz + 1;
    const csx = (domainMax[0] - domainMin[0]) / nx;
    const csy = (domainMax[1] - domainMin[1]) / ny;
    const csz = (domainMax[2] - domainMin[2]) / nz;

    const data = new Float32Array(nvx * nvy * nvz);
    // Inside/outside SIGN comes from the generalized (fast) winding number, which is robust for
    // OPEN / non-manifold meshes (unlike angle-weighted pseudonormals). A node is inside iff
    // |windingNumber| > 0.5 — the |·| makes it independent of triangle winding orientation, so this
    // subsumes the old domain-corner auto-orient hack. The DISTANCE MAGNITUDE still comes from the
    // exact pseudonormal path (correct unsigned distance + smooth gradient the SPH boundary needs);
    // we store it with the winding-number sign. Default (invert=false): inside POSITIVE.
    const outSign = invert ? -1 : 1;
    for (let k = 0; k < nvz; k++) {
        const z = domainMin[2] + csz * k;
        for (let j = 0; j < nvy; j++) {
            const y = domainMin[1] + csy * j;
            const rowBase = k * nvx * nvy + j * nvx;
            for (let i = 0; i < nvx; i++) {
                const x = domainMin[0] + csx * i;
                const absDist = Math.abs(dist.signedDistance(x, y, z).distance);
                const inside = Math.abs(dist.windingNumber(x, y, z)) > 0.5;
                data[rowBase + i] = outSign * (inside ? absDist : -absDist);
            }
        }
    }

    const inDomain = (px: number, py: number, pz: number): boolean =>
        px >= domainMin[0] && px <= domainMax[0] && py >= domainMin[1] && py <= domainMax[1] && pz >= domainMin[2] && pz <= domainMax[2];

    // Locate the cell + local fractions for (px,py,pz); assumes the point is inside the domain.
    // Writes the base corner index (i,j,k) and fractions into the shared scratch object `loc`.
    const loc = { i: 0, j: 0, k: 0, fx: 0, fy: 0, fz: 0 };
    const locate = (px: number, py: number, pz: number): void => {
        let gx = (px - domainMin[0]) / csx;
        let gy = (py - domainMin[1]) / csy;
        let gz = (pz - domainMin[2]) / csz;
        let i = Math.floor(gx);
        let j = Math.floor(gy);
        let k = Math.floor(gz);
        if (i < 0) {
            i = 0;
        } else if (i > nx - 1) {
            i = nx - 1;
        }
        if (j < 0) {
            j = 0;
        } else if (j > ny - 1) {
            j = ny - 1;
        }
        if (k < 0) {
            k = 0;
        } else if (k > nz - 1) {
            k = nz - 1;
        }
        gx -= i;
        gy -= j;
        gz -= k;
        loc.i = i;
        loc.j = j;
        loc.k = k;
        loc.fx = gx;
        loc.fy = gy;
        loc.fz = gz;
    };

    // The 8 corner values of the current cell (written by `corners`):
    // [c000, c100, c010, c110, c001, c101, c011, c111].
    const c = new Float64Array(8);
    const corners = (i: number, j: number, k: number): void => {
        const s0 = k * nvx * nvy + j * nvx + i;
        const s1 = s0 + nvx * nvy; // layer k+1
        c[0] = data[s0]!;
        c[1] = data[s0 + 1]!;
        c[2] = data[s0 + nvx]!;
        c[3] = data[s0 + nvx + 1]!;
        c[4] = data[s1]!;
        c[5] = data[s1 + 1]!;
        c[6] = data[s1 + nvx]!;
        c[7] = data[s1 + nvx + 1]!;
    };

    const interpolate = (px: number, py: number, pz: number): number => {
        if (!inDomain(px, py, pz)) {
            return Infinity;
        }
        locate(px, py, pz);
        corners(loc.i, loc.j, loc.k);
        const fx = loc.fx;
        const fy = loc.fy;
        const fz = loc.fz;
        const c00 = c[0]! + (c[1]! - c[0]!) * fx;
        const c10 = c[2]! + (c[3]! - c[2]!) * fx;
        const c01 = c[4]! + (c[5]! - c[4]!) * fx;
        const c11 = c[6]! + (c[7]! - c[6]!) * fx;
        const cy0 = c00 + (c10 - c00) * fy;
        const cy1 = c01 + (c11 - c01) * fy;
        return cy0 + (cy1 - cy0) * fz;
    };

    const interpolateWithGradient = (px: number, py: number, pz: number): GradientResult => {
        if (!inDomain(px, py, pz)) {
            return { value: Infinity, gx: 0, gy: 0, gz: 0 };
        }
        locate(px, py, pz);
        corners(loc.i, loc.j, loc.k);
        const fx = loc.fx;
        const fy = loc.fy;
        const fz = loc.fz;
        const c000 = c[0]!;
        const c100 = c[1]!;
        const c010 = c[2]!;
        const c110 = c[3]!;
        const c001 = c[4]!;
        const c101 = c[5]!;
        const c011 = c[6]!;
        const c111 = c[7]!;

        // Value (trilinear).
        const c00 = c000 + (c100 - c000) * fx;
        const c10 = c010 + (c110 - c010) * fx;
        const c01 = c001 + (c101 - c001) * fx;
        const c11 = c011 + (c111 - c011) * fx;
        const cy0 = c00 + (c10 - c00) * fy;
        const cy1 = c01 + (c11 - c01) * fy;
        const value = cy0 + (cy1 - cy0) * fz;

        // Analytic gradient of the trilinear field (chain rule -> divide by cell size).
        const gx = ((c100 - c000) * (1 - fy) * (1 - fz) + (c110 - c010) * fy * (1 - fz) + (c101 - c001) * (1 - fy) * fz + (c111 - c011) * fy * fz) / csx;
        const gy = ((c010 - c000) * (1 - fx) * (1 - fz) + (c110 - c100) * fx * (1 - fz) + (c011 - c001) * (1 - fx) * fz + (c111 - c101) * fx * fz) / csy;
        const gz = ((c001 - c000) * (1 - fx) * (1 - fy) + (c101 - c100) * fx * (1 - fy) + (c011 - c010) * (1 - fx) * fy + (c111 - c110) * fx * fy) / csz;
        return { value, gx, gy, gz };
    };

    const distanceWithNormal = (px: number, py: number, pz: number, tolerance: number): DistanceNormalResult => {
        const g = interpolateWithGradient(px, py, pz);
        if (g.value === Infinity) {
            return { dist: Infinity, nx: 0, ny: 0, nz: 0, cx: px, cy: py, cz: pz };
        }
        let nnx = g.gx;
        let nny = g.gy;
        let nnz = g.gz;
        const nlen = Math.hypot(nnx, nny, nnz);
        if (nlen > 0) {
            nnx /= nlen;
            nny /= nlen;
            nnz /= nlen;
        }
        return {
            dist: g.value - tolerance,
            nx: nnx,
            ny: nny,
            nz: nnz,
            cx: px - g.value * nnx,
            cy: py - g.value * nny,
            cz: pz - g.value * nnz,
        };
    };

    return {
        res: [nx, ny, nz],
        domainMin: [domainMin[0], domainMin[1], domainMin[2]],
        domainMax: [domainMax[0], domainMax[1], domainMax[2]],
        cellSize: [csx, csy, csz],
        data,
        interpolate,
        interpolateWithGradient,
        distanceWithNormal,
    };
}

// Interior angle at vertex (vx,vy,vz), between the edges leading to (ax,ay,az) and (bx,by,bz).
function vertexAngle(vx: number, vy: number, vz: number, ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
    let e1x = ax - vx;
    let e1y = ay - vy;
    let e1z = az - vz;
    let e2x = bx - vx;
    let e2y = by - vy;
    let e2z = bz - vz;
    const l1 = Math.hypot(e1x, e1y, e1z);
    const l2 = Math.hypot(e2x, e2y, e2z);
    if (l1 > 0) {
        e1x /= l1;
        e1y /= l1;
        e1z /= l1;
    }
    if (l2 > 0) {
        e2x /= l2;
        e2y /= l2;
        e2z /= l2;
    }
    let d = e1x * e2x + e1y * e2y + e1z * e2z;
    if (d < -1) {
        d = -1;
    } else if (d > 1) {
        d = 1;
    }
    return Math.acos(d);
}

// Normalize each 3-vector in a flat buffer in place (leaves zero-length vectors untouched).
function normalizeVec3Buffer(buf: Float32Array, count: number): void {
    for (let i = 0; i < count; i++) {
        const x = buf[3 * i]!;
        const y = buf[3 * i + 1]!;
        const z = buf[3 * i + 2]!;
        const len = Math.hypot(x, y, z);
        if (len > 0) {
            buf[3 * i] = x / len;
            buf[3 * i + 1] = y / len;
            buf[3 * i + 2] = z / len;
        }
    }
}
