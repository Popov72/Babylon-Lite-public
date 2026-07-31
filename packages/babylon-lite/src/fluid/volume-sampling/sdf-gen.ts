// Fast triangle-mesh signed-distance-field baker — a faithful CPU port of
// christopherbatty/SDFGen (`makelevelset3.cpp`, Bridson's narrow-band + fast sweep).
//
// The exact winding-number evaluator in `mesh-sdf.ts` is accurate but too slow for a dense
// high-resolution grid (~3s for 30k nodes). This module bakes a mesh into a dense signed-distance
// GRID in three cheap stages that scale to 128^3 in a fraction of a second:
//
//   1. Narrow band — for each triangle, visit the grid nodes inside its AABB (grown by `exactBand`
//      cells) and record the EXACT point-to-triangle distance + the closest triangle index.
//   2. Sign via ray crossing — for every grid COLUMN along +x, +y and +z, count how many triangles the
//      column pierces before each node (SDFGen's robust `point_in_triangle_2d` + exact intercept). A
//      node is INSIDE along an axis iff that column's running crossing count is ODD; the final sign is
//      the MAJORITY of the three. SDFGen votes on +x alone, which is exact for a closed surface but
//      lines out an entire ray of wrongly-signed nodes through any hole — and real level geometry has
//      doorways, coincident faces and open ends. Three axes agree on a closed mesh, so nothing changes
//      there, while a hole that breaks one direction is outvoted by the other two.
//   3. Fast sweep — a handful of Gauss-Seidel passes propagate each node's closest-triangle guess to
//      its 26-neighbourhood, filling exact distances OUTSIDE the narrow band (SDFGen's `sweep` /
//      `check_neighbour`).
//
// Convention (matches SDFGen): distance is NEGATIVE inside the solid, POSITIVE outside.
//
// Pure CPU — no GPU handles, no imports, zero module-level side effects. The point-to-triangle
// squared-distance routine is the Ericson closest-point clamp, replicated locally (the version in
// `mesh-sdf.ts` is a private closure over its own index buffers).

/** Grid definition for {@link generateMeshSdf}. Provide EITHER `max` OR `dims` alongside `min`. */
export interface MeshSdfOptions {
    /** World-space minimum corner of the mesh AABB (the grid is padded outward from here). */
    min: readonly [number, number, number];
    /** Uniform world-space grid spacing (edge length of one cell). */
    cellSize: number;
    /**
     * World-space maximum corner of the mesh AABB. With this form the grid is padded by `padding`
     * cells on every side, so `origin = min - padding*cellSize` and node counts cover `[min,max]`
     * plus the padding. Provide EITHER `max` OR `dims`.
     */
    max?: readonly [number, number, number];
    /**
     * Explicit node counts per axis. With this form `origin = min` and the counts are used verbatim
     * (`padding` is ignored — bake in any margin yourself). Provide EITHER `dims` OR `max`.
     */
    dims?: readonly [number, number, number];
    /** Extra cells added beyond the AABB on every side (max form only). Default 2. */
    padding?: number;
    /** Half-width in cells of the exact narrow band computed per triangle. Default 2. */
    exactBand?: number;
    /**
     * Number of fast-sweep rounds (each round = the 8 diagonal sweeps). Higher is more accurate far
     * from the surface; the near-surface band is already exact regardless. Default 2 (SDFGen).
     */
    sweepPasses?: number;
    /**
     * How a node's INSIDE/OUTSIDE sign is decided. Distances are identical either way.
     *
     * `"parity"` (default) votes on the odd/even triangle-crossing count along the +x, +y and +z
     * columns through the node. Exact for a closed surface and cheap, but every ray through a hole in
     * the mesh flips the parity of the whole ray behind it, so an open model grows slabs of inverted
     * sign that no amount of voting removes once two axes are wrong at once.
     *
     * `"flood"` ignores crossings and asks a topological question instead: can this node be reached
     * from OUTSIDE the grid without passing through the surface band? Reachable ⇒ outside. It cannot
     * invert an enclosed volume, but it needs the OUTER hull to be closed — one hole and "outside"
     * floods the whole interior, which then reads as empty space.
     *
     * `"normal"` is fully local and needs no closure at all: a node takes the side of the plane of its
     * nearest triangle. Correct wherever the nearest feature is a face interior, which is almost
     * everywhere at cell resolution; it can flip within a cell or two of a sharp concave edge, and it
     * requires consistent outward winding. This is the one to reach for on kit-bashed level geometry,
     * where neither the shell nor the hull is closed.
     */
    signMode?: "parity" | "flood" | "normal";
    /**
     * Flip any component smaller than this many nodes that does not touch the grid boundary.
     * Default 0 (off).
     *
     * Kit-bashed levels do not meet exactly: adjacent wall panels here are modelled 0.01–0.02 m apart,
     * and a ray slipping through such a seam flips the parity behind it, leaving a one- or two-node
     * island floating in open space. They are far too small to be real geometry at cell resolution, so
     * removing them is safe — but only for a model whose material is one big connected region (a room
     * shell, a hull). Do NOT use it on a bake of small separate props: each prop is its own small
     * component and would be erased. Applies to islands of EITHER sign, since the caller may be reading
     * this grid as a solid or (inverted) as a container.
     */
    despeckle?: number;
}

/** A dense signed-distance grid. `data[idx]` is negative inside the solid, positive outside. */
export interface MeshSdfGrid {
    /**
     * Signed distance per grid node, x-fastest / z-slowest:
     * `idx = i + dims[0] * (j + dims[1] * k)`. Length is `dims[0]*dims[1]*dims[2]`.
     */
    data: Float32Array;
    /** Node counts per axis. */
    dims: [number, number, number];
    /** World-space position of node (0,0,0). Node (i,j,k) sits at `origin + cellSize*(i,j,k)`. */
    origin: [number, number, number];
    /** Uniform world-space grid spacing. */
    cellSize: number;
}

const DEFAULT_PADDING = 2;
const DEFAULT_EXACT_BAND = 2;
const DEFAULT_SWEEP_PASSES = 2;

/**
 * Bake an indexed triangle mesh into a dense signed-distance grid (SDFGen algorithm).
 *
 * @param positions - flat xyz vertex positions.
 * @param indices - flat triangle indices (three per triangle); Uint16 or Uint32.
 * @param opts - grid definition (see {@link MeshSdfOptions}).
 * @returns a {@link MeshSdfGrid} — negative inside the solid, positive outside.
 */
export function generateMeshSdf(positions: Float32Array, indices: Uint32Array | Uint16Array, opts: MeshSdfOptions): MeshSdfGrid {
    const cellSize = opts.cellSize;
    if (!(cellSize > 0)) {
        throw new Error("generateMeshSdf: cellSize must be > 0");
    }
    const invDx = 1 / cellSize;
    const padding = opts.padding ?? DEFAULT_PADDING;
    const exactBand = Math.max(1, opts.exactBand ?? DEFAULT_EXACT_BAND);
    const sweepPasses = Math.max(1, Math.floor(opts.sweepPasses ?? DEFAULT_SWEEP_PASSES));

    // Resolve grid extents + origin from whichever form the caller used.
    let ni: number;
    let nj: number;
    let nk: number;
    let ox: number;
    let oy: number;
    let oz: number;
    if (opts.dims) {
        ni = Math.max(1, Math.floor(opts.dims[0]));
        nj = Math.max(1, Math.floor(opts.dims[1]));
        nk = Math.max(1, Math.floor(opts.dims[2]));
        ox = opts.min[0];
        oy = opts.min[1];
        oz = opts.min[2];
    } else if (opts.max) {
        ox = opts.min[0] - padding * cellSize;
        oy = opts.min[1] - padding * cellSize;
        oz = opts.min[2] - padding * cellSize;
        ni = Math.ceil((opts.max[0] - opts.min[0]) * invDx) + 1 + 2 * padding;
        nj = Math.ceil((opts.max[1] - opts.min[1]) * invDx) + 1 + 2 * padding;
        nk = Math.ceil((opts.max[2] - opts.min[2]) * invDx) + 1 + 2 * padding;
        ni = Math.max(1, ni);
        nj = Math.max(1, nj);
        nk = Math.max(1, nk);
    } else {
        throw new Error("generateMeshSdf: provide either opts.max or opts.dims");
    }

    const nij = ni * nj;
    const nodeCount = nij * nk;

    // dist2 holds the SQUARED unsigned distance to the surface (avoids a sqrt in every inner-loop
    // comparison; Float64 keeps the squares precise). closestTri is the SDFGen `closest_tri`.
    // crossings[i,j,k] counts triangles pierced in the interval (i-1, i] along +x at column (j,k).
    const dist2 = new Float64Array(nodeCount);
    const large = (ni + nj + nk) * cellSize; // upper bound on any distance (SDFGen `phi.assign`)
    const large2 = large * large;
    dist2.fill(large2);
    const closestTri = new Int32Array(nodeCount).fill(-1);
    // Crossing counts for THREE ray directions. `crossings`[i,j,k] counts triangles pierced in the
    // interval (i-1, i] along +x at column (j,k); `crossingsY` / `crossingsZ` are the same along +y and
    // +z. SDFGen uses +x alone, which is correct only for a closed surface: one hole in the mesh flips
    // the parity of every node on the rays that pass through it, so an open model grows whole LINES of
    // wrongly-signed nodes. Real level geometry — kit-bashed rooms with doorways, coincident faces and
    // open ends — is never that clean, so the sign is taken as the MAJORITY of the three axes instead. A
    // hole big enough to break one direction rarely lines up with the other two, and for a genuinely
    // closed mesh all three agree and the result is unchanged.
    const crossings = new Int32Array(nodeCount);
    const crossingsY = new Int32Array(nodeCount);
    const crossingsZ = new Int32Array(nodeCount);

    const triCount = (indices.length / 3) | 0;
    const bary = new Float64Array(3); // scratch for point_in_triangle_2d barycentric weights

    // Flat per-triangle vertex table (9 floats/tri) — read in both the narrow band and the sweep.
    // Collapsing the index→position indirection into one contiguous array is the biggest single
    // win for the sweep, which evaluates triangle distances hundreds of millions of times at 128^3.
    const triVerts = new Float64Array(9 * triCount);
    for (let t = 0; t < triCount; t++) {
        const p = indices[3 * t]!;
        const q = indices[3 * t + 1]!;
        const r = indices[3 * t + 2]!;
        const b = 9 * t;
        triVerts[b] = positions[3 * p]!;
        triVerts[b + 1] = positions[3 * p + 1]!;
        triVerts[b + 2] = positions[3 * p + 2]!;
        triVerts[b + 3] = positions[3 * q]!;
        triVerts[b + 4] = positions[3 * q + 1]!;
        triVerts[b + 5] = positions[3 * q + 2]!;
        triVerts[b + 6] = positions[3 * r]!;
        triVerts[b + 7] = positions[3 * r + 1]!;
        triVerts[b + 8] = positions[3 * r + 2]!;
    }

    // ---- Stages 1 & 2: exact narrow band + intersection counting -------------------------------
    for (let t = 0; t < triCount; t++) {
        const b = 9 * t;
        const px = triVerts[b]!;
        const py = triVerts[b + 1]!;
        const pz = triVerts[b + 2]!;
        const qx = triVerts[b + 3]!;
        const qy = triVerts[b + 4]!;
        const qz = triVerts[b + 5]!;
        const rx = triVerts[b + 6]!;
        const ry = triVerts[b + 7]!;
        const rz = triVerts[b + 8]!;

        // Skip fully degenerate (zero-area / duplicate-vertex) triangles — they contribute no
        // distance and cast no ray crossing.
        const e1x = qx - px;
        const e1y = qy - py;
        const e1z = qz - pz;
        const e2x = rx - px;
        const e2y = ry - py;
        const e2z = rz - pz;
        const cxn = e1y * e2z - e1z * e2y;
        const cyn = e1z * e2x - e1x * e2z;
        const czn = e1x * e2y - e1y * e2x;
        if (cxn * cxn + cyn * cyn + czn * czn <= 0) {
            continue;
        }

        // Vertex coordinates in grid space (high precision).
        const fip = (px - ox) * invDx;
        const fjp = (py - oy) * invDx;
        const fkp = (pz - oz) * invDx;
        const fiq = (qx - ox) * invDx;
        const fjq = (qy - oy) * invDx;
        const fkq = (qz - oz) * invDx;
        const fir = (rx - ox) * invDx;
        const fjr = (ry - oy) * invDx;
        const fkr = (rz - oz) * invDx;

        // Stage 1: exact distances for nodes within `exactBand` cells of the triangle AABB.
        const i0 = clampInt(Math.floor(min3(fip, fiq, fir)) - exactBand, 0, ni - 1);
        const i1 = clampInt(Math.floor(max3(fip, fiq, fir)) + exactBand + 1, 0, ni - 1);
        const j0 = clampInt(Math.floor(min3(fjp, fjq, fjr)) - exactBand, 0, nj - 1);
        const j1 = clampInt(Math.floor(max3(fjp, fjq, fjr)) + exactBand + 1, 0, nj - 1);
        const k0 = clampInt(Math.floor(min3(fkp, fkq, fkr)) - exactBand, 0, nk - 1);
        const k1 = clampInt(Math.floor(max3(fkp, fkq, fkr)) + exactBand + 1, 0, nk - 1);
        for (let k = k0; k <= k1; k++) {
            const gz = k * cellSize + oz;
            for (let j = j0; j <= j1; j++) {
                const gy = j * cellSize + oy;
                let idx = i0 + ni * (j + nj * k);
                for (let i = i0; i <= i1; i++, idx++) {
                    const gx = i * cellSize + ox;
                    const d2 = triDistSqAt(triVerts, b, gx, gy, gz);
                    if (d2 < dist2[idx]!) {
                        dist2[idx] = d2;
                        closestTri[idx] = t;
                    }
                }
            }
        }

        // Stage 2: intersection counts. Project onto (y,z); the ray runs along +x. For every grid
        // column (j,k) inside the triangle's (y,z) footprint, find the exact x-intercept and mark
        // the node interval it lands in.
        const cj0 = clampInt(Math.ceil(min3(fjp, fjq, fjr)), 0, nj - 1);
        const cj1 = clampInt(Math.floor(max3(fjp, fjq, fjr)), 0, nj - 1);
        const ck0 = clampInt(Math.ceil(min3(fkp, fkq, fkr)), 0, nk - 1);
        const ck1 = clampInt(Math.floor(max3(fkp, fkq, fkr)), 0, nk - 1);
        for (let k = ck0; k <= ck1; k++) {
            for (let j = cj0; j <= cj1; j++) {
                if (pointInTriangle2d(j, k, fjp, fkp, fjq, fkq, fjr, fkr, bary)) {
                    const fi = bary[0]! * fip + bary[1]! * fiq + bary[2]! * fir; // x-intercept in grid space
                    const iInterval = Math.ceil(fi); // intercept lies in (iInterval-1, iInterval]
                    const colBase = ni * (j + nj * k);
                    if (iInterval < 0) {
                        // Fold everything to the -x of the grid into the first interval.
                        crossings[colBase] = crossings[colBase]! + 1;
                    } else if (iInterval < ni) {
                        const ci = iInterval + colBase;
                        crossings[ci] = crossings[ci]! + 1;
                    }
                    // Intercepts beyond +x of the grid are ignored (SDFGen behaviour).
                }
            }
        }

        // Stage 2b/2c: the same intersection count along +y and +z, for the majority-of-three sign.
        // +y: project onto (x,z), column (i,k), nodes strided by ni.
        const ci0 = clampInt(Math.ceil(min3(fip, fiq, fir)), 0, ni - 1);
        const ci1 = clampInt(Math.floor(max3(fip, fiq, fir)), 0, ni - 1);
        for (let k = ck0; k <= ck1; k++) {
            for (let i = ci0; i <= ci1; i++) {
                if (pointInTriangle2d(i, k, fip, fkp, fiq, fkq, fir, fkr, bary)) {
                    const fj = bary[0]! * fjp + bary[1]! * fjq + bary[2]! * fjr;
                    const jInterval = Math.ceil(fj);
                    const colBase = i + ni * nj * k;
                    if (jInterval < 0) {
                        crossingsY[colBase] = crossingsY[colBase]! + 1;
                    } else if (jInterval < nj) {
                        const cIdx = colBase + ni * jInterval;
                        crossingsY[cIdx] = crossingsY[cIdx]! + 1;
                    }
                }
            }
        }
        // +z: project onto (x,y), column (i,j), nodes strided by ni*nj.
        for (let j = cj0; j <= cj1; j++) {
            for (let i = ci0; i <= ci1; i++) {
                if (pointInTriangle2d(i, j, fip, fjp, fiq, fjq, fir, fjr, bary)) {
                    const fk = bary[0]! * fkp + bary[1]! * fkq + bary[2]! * fkr;
                    const kInterval = Math.ceil(fk);
                    const colBase = i + ni * j;
                    if (kInterval < 0) {
                        crossingsZ[colBase] = crossingsZ[colBase]! + 1;
                    } else if (kInterval < nk) {
                        const cIdx = colBase + nij * kInterval;
                        crossingsZ[cIdx] = crossingsZ[cIdx]! + 1;
                    }
                }
            }
        }
    }

    // ---- Stage 3: fast sweeping — propagate closest triangles across the whole grid ------------
    // For each node, recompute the exact distance to each already-swept neighbour's closest triangle
    // and adopt the closest (SDFGen `check_neighbour` + `sweep`). Hot-loop shape:
    //   • neighbour indices via stride arithmetic (no per-neighbour index multiplications),
    //   • the node's running best distance/tri held in locals (one typed-array write-back per node),
    //   • the 7 neighbour checks inlined,
    //   • an EXACT prune (identical output): skip a neighbour whose closest triangle already equals
    //     the node's current best — re-evaluating it just reproduces the node's own stored distance.
    const sweep = (di: number, dj: number, dk: number): void => {
        const sJ = dj * ni;
        const sK = dk * nij;
        const iStart = di > 0 ? 1 : ni - 2;
        const iEnd = di > 0 ? ni : -1;
        const jStart = dj > 0 ? 1 : nj - 2;
        const jEnd = dj > 0 ? nj : -1;
        const kStart = dk > 0 ? 1 : nk - 2;
        const kEnd = dk > 0 ? nk : -1;
        const stepX = di * cellSize;
        // Offsets of the 7 backward neighbours (already visited earlier in this sweep) from idx0.
        const o0 = -di;
        const o1 = -sJ;
        const o2 = -di - sJ;
        const o3 = -sK;
        const o4 = -di - sK;
        const o5 = -sJ - sK;
        const o6 = -di - sJ - sK;
        for (let k = kStart; k !== kEnd; k += dk) {
            const gz = k * cellSize + oz;
            for (let j = jStart; j !== jEnd; j += dj) {
                const gy = j * cellSize + oy;
                let idx0 = iStart + ni * (j + nj * k);
                let gx = iStart * cellSize + ox;
                for (let i = iStart; i !== iEnd; i += di, idx0 += di, gx += stepX) {
                    let best2 = dist2[idx0]!;
                    let bestCt = closestTri[idx0]!;
                    let ct = closestTri[idx0 + o0]!;
                    if (ct >= 0 && ct !== bestCt) {
                        const d2 = triDistSqAt(triVerts, 9 * ct, gx, gy, gz);
                        if (d2 < best2) {
                            best2 = d2;
                            bestCt = ct;
                        }
                    }
                    ct = closestTri[idx0 + o1]!;
                    if (ct >= 0 && ct !== bestCt) {
                        const d2 = triDistSqAt(triVerts, 9 * ct, gx, gy, gz);
                        if (d2 < best2) {
                            best2 = d2;
                            bestCt = ct;
                        }
                    }
                    ct = closestTri[idx0 + o2]!;
                    if (ct >= 0 && ct !== bestCt) {
                        const d2 = triDistSqAt(triVerts, 9 * ct, gx, gy, gz);
                        if (d2 < best2) {
                            best2 = d2;
                            bestCt = ct;
                        }
                    }
                    ct = closestTri[idx0 + o3]!;
                    if (ct >= 0 && ct !== bestCt) {
                        const d2 = triDistSqAt(triVerts, 9 * ct, gx, gy, gz);
                        if (d2 < best2) {
                            best2 = d2;
                            bestCt = ct;
                        }
                    }
                    ct = closestTri[idx0 + o4]!;
                    if (ct >= 0 && ct !== bestCt) {
                        const d2 = triDistSqAt(triVerts, 9 * ct, gx, gy, gz);
                        if (d2 < best2) {
                            best2 = d2;
                            bestCt = ct;
                        }
                    }
                    ct = closestTri[idx0 + o5]!;
                    if (ct >= 0 && ct !== bestCt) {
                        const d2 = triDistSqAt(triVerts, 9 * ct, gx, gy, gz);
                        if (d2 < best2) {
                            best2 = d2;
                            bestCt = ct;
                        }
                    }
                    ct = closestTri[idx0 + o6]!;
                    if (ct >= 0 && ct !== bestCt) {
                        const d2 = triDistSqAt(triVerts, 9 * ct, gx, gy, gz);
                        if (d2 < best2) {
                            best2 = d2;
                            bestCt = ct;
                        }
                    }
                    if (bestCt !== closestTri[idx0]!) {
                        dist2[idx0] = best2;
                        closestTri[idx0] = bestCt;
                    }
                }
            }
        }
    };

    for (let pass = 0; pass < sweepPasses; pass++) {
        sweep(+1, +1, +1);
        sweep(-1, -1, -1);
        sweep(+1, +1, -1);
        sweep(-1, -1, +1);
        sweep(+1, -1, +1);
        sweep(-1, +1, -1);
        sweep(+1, -1, -1);
        sweep(-1, +1, +1);
    }

    // ---- Final: sqrt + sign from the MAJORITY of the three axes' crossing parities ---------------
    // Each axis' parity is a running prefix sum along its own column, so all three are accumulated in
    // one pass per axis before the vote.
    const insideX = new Uint8Array(nodeCount);
    const insideY = new Uint8Array(nodeCount);
    const insideZ = new Uint8Array(nodeCount);
    for (let k = 0; k < nk; k++) {
        for (let j = 0; j < nj; j++) {
            let total = 0;
            let idx = ni * (j + nj * k);
            for (let i = 0; i < ni; i++, idx++) {
                total += crossings[idx]!;
                insideX[idx] = total & 1;
            }
        }
    }
    for (let k = 0; k < nk; k++) {
        for (let i = 0; i < ni; i++) {
            let total = 0;
            let idx = i + nij * k;
            for (let j = 0; j < nj; j++, idx += ni) {
                total += crossingsY[idx]!;
                insideY[idx] = total & 1;
            }
        }
    }
    for (let j = 0; j < nj; j++) {
        for (let i = 0; i < ni; i++) {
            let total = 0;
            let idx = i + ni * j;
            for (let k = 0; k < nk; k++, idx += nij) {
                total += crossingsZ[idx]!;
                insideZ[idx] = total & 1;
            }
        }
    }
    const data = new Float32Array(nodeCount);
    if (opts.signMode === "normal") {
        // Side of the nearest triangle's plane. `closestTri` is exact in the narrow band and carried
        // outwards by the sweep, so every node has one.
        for (let idx = 0; idx < nodeCount; idx++) {
            const d = Math.sqrt(dist2[idx]!);
            const t = closestTri[idx]!;
            if (t < 0) {
                data[idx] = d;
                continue;
            }
            const b = 9 * t;
            const ax = triVerts[b]!,
                ay = triVerts[b + 1]!,
                az = triVerts[b + 2]!;
            const e1x = triVerts[b + 3]! - ax,
                e1y = triVerts[b + 4]! - ay,
                e1z = triVerts[b + 5]! - az;
            const e2x = triVerts[b + 6]! - ax,
                e2y = triVerts[b + 7]! - ay,
                e2z = triVerts[b + 8]! - az;
            const nx = e1y * e2z - e1z * e2y,
                ny = e1z * e2x - e1x * e2z,
                nz = e1x * e2y - e1y * e2x;
            const i = idx % ni,
                j = ((idx / ni) | 0) % nj,
                k = (idx / nij) | 0;
            const s = (i * cellSize + ox - ax) * nx + (j * cellSize + oy - ay) * ny + (k * cellSize + oz - az) * nz;
            data[idx] = s < 0 ? -d : d;
        }
    } else if (opts.signMode === "flood") {
        // Reachability from the grid boundary, blocked by the surface band. Two phases: flood the open
        // space, then hand the band cells the label of whichever side reached them first.
        const BAND = cellSize; // one cell — the exact narrow band is already at least this wide
        const OUTSIDE = 1,
            INSIDE = 2;
        const label = new Uint8Array(nodeCount);
        const queue = new Int32Array(nodeCount);
        let head = 0,
            tail = 0;
        const open = (idx: number): boolean => dist2[idx]! > BAND * BAND;
        const push = (idx: number, lab: number): void => {
            if (label[idx]) {
                return;
            }
            label[idx] = lab;
            queue[tail++] = idx;
        };
        for (let k = 0; k < nk; k++) {
            for (let j = 0; j < nj; j++) {
                for (let i = 0; i < ni; i++) {
                    if (i !== 0 && i !== ni - 1 && j !== 0 && j !== nj - 1 && k !== 0 && k !== nk - 1) {
                        continue;
                    }
                    const idx = i + ni * (j + nj * k);
                    if (open(idx)) {
                        push(idx, OUTSIDE);
                    }
                }
            }
        }
        // Phase 1: spread OUTSIDE through open space only.
        const spread = (bandToo: boolean): void => {
            while (head < tail) {
                const idx = queue[head++]!;
                const lab = label[idx]!;
                const i = idx % ni,
                    j = ((idx / ni) | 0) % nj,
                    k = (idx / nij) | 0;
                if (i > 0 && (bandToo || open(idx - 1))) {
                    push(idx - 1, lab);
                }
                if (i < ni - 1 && (bandToo || open(idx + 1))) {
                    push(idx + 1, lab);
                }
                if (j > 0 && (bandToo || open(idx - ni))) {
                    push(idx - ni, lab);
                }
                if (j < nj - 1 && (bandToo || open(idx + ni))) {
                    push(idx + ni, lab);
                }
                if (k > 0 && (bandToo || open(idx - nij))) {
                    push(idx - nij, lab);
                }
                if (k < nk - 1 && (bandToo || open(idx + nij))) {
                    push(idx + nij, lab);
                }
            }
        };
        spread(false);
        // Phase 2: every open pocket the boundary could not reach is enclosed ⇒ INSIDE.
        for (let idx = 0; idx < nodeCount; idx++) {
            if (!label[idx] && open(idx)) {
                push(idx, INSIDE);
            }
        }
        spread(false);
        // Phase 3: the band itself takes the nearest label.
        head = 0;
        tail = 0;
        for (let idx = 0; idx < nodeCount; idx++) {
            if (label[idx]) {
                queue[tail++] = idx;
            }
        }
        spread(true);
        for (let idx = 0; idx < nodeCount; idx++) {
            data[idx] = label[idx] === INSIDE ? -Math.sqrt(dist2[idx]!) : Math.sqrt(dist2[idx]!);
        }
    } else {
        for (let idx = 0; idx < nodeCount; idx++) {
            const d = Math.sqrt(dist2[idx]!);
            data[idx] = insideX[idx]! + insideY[idx]! + insideZ[idx]! >= 2 ? -d : d; // majority inside ⇒ negative
        }
    }

    // Despeckle: erase islands too small to be geometry (see MeshSdfOptions.despeckle).
    const maxSpeck = opts.despeckle ?? 0;
    if (maxSpeck > 0) {
        const seen = new Uint8Array(nodeCount);
        const stack = new Int32Array(nodeCount);
        const comp = new Int32Array(maxSpeck + 1);
        for (let start = 0; start < nodeCount; start++) {
            if (seen[start]) {
                continue;
            }
            // Sign-agnostic: a caller may be treating this grid as a solid OR (via an inversion) as a
            // container, so the speck can be on either side. Both are flipped the same way.
            const inside = data[start]! < 0;
            let sp = 0;
            let size = 0;
            let touchesEdge = false;
            stack[sp++] = start;
            seen[start] = 1;
            while (sp > 0) {
                const idx = stack[--sp]!;
                if (size < comp.length) {
                    comp[size] = idx;
                }
                size++;
                const i = idx % ni;
                const j = ((idx / ni) | 0) % nj;
                const k = (idx / nij) | 0;
                if (i === 0 || j === 0 || k === 0 || i === ni - 1 || j === nj - 1 || k === nk - 1) {
                    touchesEdge = true;
                }
                const nb = [
                    i > 0 ? idx - 1 : -1,
                    i < ni - 1 ? idx + 1 : -1,
                    j > 0 ? idx - ni : -1,
                    j < nj - 1 ? idx + ni : -1,
                    k > 0 ? idx - nij : -1,
                    k < nk - 1 ? idx + nij : -1,
                ];
                for (const t of nb) {
                    // No early exit on a big component: leaving part of it unvisited would let the
                    // remainder be re-walked later and mistaken for a speck of its own.
                    if (t >= 0 && !seen[t] && data[t]! < 0 === inside) {
                        seen[t] = 1;
                        stack[sp++] = t;
                    }
                }
            }
            if (size <= maxSpeck && !touchesEdge) {
                for (let c = 0; c < size; c++) {
                    data[comp[c]!] = -data[comp[c]!]!;
                }
            }
        }
    }

    return {
        data,
        dims: [ni, nj, nk],
        origin: [ox, oy, oz],
        cellSize,
    };
}

// --- Local math helpers (top-level, pure, allocation-free) --------------------------------------

function min3(a: number, b: number, c: number): number {
    return a < b ? (a < c ? a : c) : b < c ? b : c;
}

function max3(a: number, b: number, c: number): number {
    return a > b ? (a > c ? a : c) : b > c ? b : c;
}

function clampInt(v: number, lo: number, hi: number): number {
    return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Squared distance from point (px,py,pz) to the triangle stored at `tv[base .. base+8]` —
 * Ericson's closest-point-on-triangle clamp (vertex / edge / face Voronoi regions). Allocation-free
 * and reads from a flat vertex table, so it stays cheap in the sweep's inner loop.
 */
function triDistSqAt(tv: Float64Array, base: number, px: number, py: number, pz: number): number {
    const ax = tv[base]!;
    const ay = tv[base + 1]!;
    const az = tv[base + 2]!;
    const bx = tv[base + 3]!;
    const by = tv[base + 4]!;
    const bz = tv[base + 5]!;
    const cx = tv[base + 6]!;
    const cy = tv[base + 7]!;
    const cz = tv[base + 8]!;

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
        return apx * apx + apy * apy + apz * apz; // vertex A region
    }

    const bpx = px - bx;
    const bpy = py - by;
    const bpz = pz - bz;
    const d3 = abx * bpx + aby * bpy + abz * bpz;
    const d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) {
        return bpx * bpx + bpy * bpy + bpz * bpz; // vertex B region
    }

    const cpx = px - cx;
    const cpy = py - cy;
    const cpz = pz - cz;
    const d5 = abx * cpx + aby * cpy + abz * cpz;
    const d6 = acx * cpx + acy * cpy + acz * cpz;
    if (d6 >= 0 && d5 <= d6) {
        return cpx * cpx + cpy * cpy + cpz * cpz; // vertex C region
    }

    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const v = d1 / (d1 - d3); // edge AB region
        const dx = apx - v * abx;
        const dy = apy - v * aby;
        const dz = apz - v * abz;
        return dx * dx + dy * dy + dz * dz;
    }

    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const w = d2 / (d2 - d6); // edge AC region
        const dx = apx - w * acx;
        const dy = apy - w * acy;
        const dz = apz - w * acz;
        return dx * dx + dy * dy + dz * dz;
    }

    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        const w = (d4 - d3) / (d4 - d3 + (d5 - d6)); // edge BC region
        const dx = px - (bx + w * (cx - bx));
        const dy = py - (by + w * (cy - by));
        const dz = pz - (bz + w * (cz - bz));
        return dx * dx + dy * dy + dz * dz;
    }

    // Interior of the face — project via barycentric weights.
    const denom = 1 / (va + vb + vc);
    const v = vb * denom;
    const w = vc * denom;
    const dx = apx - (abx * v + acx * w);
    const dy = apy - (aby * v + acy * w);
    const dz = apz - (abz * v + acz * w);
    return dx * dx + dy * dy + dz * dz;
}

/**
 * Symbolic-perturbation orientation of triangle (0,0)-(x1,y1)-(x2,y2). Returns +1 / -1, or 0 only
 * for a truly degenerate 2D triangle. Writes twice the signed area into `out[outIdx]`.
 */
function orientation(x1: number, y1: number, x2: number, y2: number, out: Float64Array, outIdx: number): number {
    const twiceArea = y1 * x2 - x1 * y2;
    out[outIdx] = twiceArea;
    if (twiceArea > 0) {
        return 1;
    }
    if (twiceArea < 0) {
        return -1;
    }
    if (y2 > y1) {
        return 1;
    }
    if (y2 < y1) {
        return -1;
    }
    if (x1 > x2) {
        return 1;
    }
    if (x1 < x2) {
        return -1;
    }
    return 0;
}

/**
 * Robust test of whether (x0,y0) lies in triangle (x1,y1)-(x2,y2)-(x3,y3). On success returns true
 * and writes the normalized barycentric weights (for vertices 1,2,3) into `bary`. SDFGen's
 * `point_in_triangle_2d` — uses {@link orientation}'s symbolic perturbation for exactness on edges.
 */
function pointInTriangle2d(x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, bary: Float64Array): boolean {
    x1 -= x0;
    x2 -= x0;
    x3 -= x0;
    y1 -= y0;
    y2 -= y0;
    y3 -= y0;
    const signa = orientation(x2, y2, x3, y3, bary, 0);
    if (signa === 0) {
        return false;
    }
    const signb = orientation(x3, y3, x1, y1, bary, 1);
    if (signb !== signa) {
        return false;
    }
    const signc = orientation(x1, y1, x2, y2, bary, 2);
    if (signc !== signa) {
        return false;
    }
    const sum = bary[0]! + bary[1]! + bary[2]!;
    // The SOS signs match and are nonzero, so `sum` cannot be zero.
    bary[0] = bary[0]! / sum;
    bary[1] = bary[1]! / sum;
    bary[2] = bary[2]! / sum;
    return true;
}
