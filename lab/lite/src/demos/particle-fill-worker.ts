/**
 * Liquefactor — mesh → particle volume-sampling worker.
 *
 * `sampleMeshVolume` (the CPU volume sampler) is heavy — dense sampling of a foe can block the
 * main thread for ~1 s, freezing the frame at the moment of the shot. Running it here keeps the
 * render loop smooth: the main thread posts the (copied) mesh geometry + params, we sample, bake
 * the mesh's world offset into the points, and post the world-space seed back (transferring the
 * buffer). Pure CPU, no GPU/DOM — safe in a worker.
 *
 * When `uvs` (per-vertex texture coords, aligned with `positions`) is supplied, we also compute a
 * per-particle UV (nearest surface vertex) so the main thread can GPU-sample the diffuse texture
 * into a per-particle colour buffer. UVs are computed in the sampler's local space (before the
 * world offset is baked in), which is fine since UVs are position-independent.
 *
 * Protocol (main → worker): { id, positions, indices, uvs, radius, mode, ox, oy, oz }
 * Protocol (worker → main): { id, positions, uvs, texIndices, count, radius, boundsMin, boundsMax }
 */
import { sampleMeshVolume } from "babylon-lite/fluid/volume-sampling/index.js";
import type { VolumeSamplingMode } from "babylon-lite/fluid/volume-sampling/index.js";

const ctx = self as unknown as Worker;

// Surface-sampling particle budget. Higher = radius affects the count over a wider range (denser
// at small radius), at the cost of a heavier sim. Bounded so a huge model stays real-time.
const SURFACE_MAX_POINTS = 250000;

interface SampleRequest {
    id: number;
    positions: Float32Array;
    indices: Uint32Array;
    uvs: Float32Array | null;
    texIndices: Uint32Array | null;
    radius: number;
    mode: VolumeSamplingMode;
    surfaceOnly: boolean;
    ox: number;
    oy: number;
    oz: number;
}

// For each sampled particle, find the nearest mesh vertex and copy its UV. A uniform spatial-hash
// grid over the vertices keeps this near-linear (expanding-shell search with a correct
// termination bound), so it stays cheap even for the dense Xbot sample.
function computeParticleUvs(particles: Float32Array, verts: Float32Array, vertUvs: Float32Array, vertTex: Uint32Array | null, radius: number): { uvs: Float32Array; texIndices: Uint32Array } {
    const nP = particles.length / 3;
    const nV = verts.length / 3;
    const out = new Float32Array(nP * 2);
    const outTex = new Uint32Array(nP).fill(0xffffffff);
    if (nV === 0) return { uvs: out, texIndices: outTex };

    let minx = Infinity,
        miny = Infinity,
        minz = Infinity,
        maxx = -Infinity,
        maxy = -Infinity,
        maxz = -Infinity;
    for (let v = 0; v < nV; v++) {
        const x = verts[v * 3]!,
            y = verts[v * 3 + 1]!,
            z = verts[v * 3 + 2]!;
        if (x < minx) minx = x;
        if (y < miny) miny = y;
        if (z < minz) minz = z;
        if (x > maxx) maxx = x;
        if (y > maxy) maxy = y;
        if (z > maxz) maxz = z;
    }
    const cell = Math.max(radius * 2, 1e-4);
    const inv = 1 / cell;
    const nx = Math.max(1, Math.floor((maxx - minx) * inv) + 1);
    const ny = Math.max(1, Math.floor((maxy - miny) * inv) + 1);
    const nz = Math.max(1, Math.floor((maxz - minz) * inv) + 1);
    const cellOf = (x: number, y: number, z: number): number => {
        const ix = Math.min(nx - 1, Math.max(0, Math.floor((x - minx) * inv)));
        const iy = Math.min(ny - 1, Math.max(0, Math.floor((y - miny) * inv)));
        const iz = Math.min(nz - 1, Math.max(0, Math.floor((z - minz) * inv)));
        return ix + nx * (iy + ny * iz);
    };
    const grid = new Map<number, number[]>();
    for (let v = 0; v < nV; v++) {
        const k = cellOf(verts[v * 3]!, verts[v * 3 + 1]!, verts[v * 3 + 2]!);
        let bucket = grid.get(k);
        if (!bucket) {
            bucket = [];
            grid.set(k, bucket);
        }
        bucket.push(v);
    }

    const maxRing = Math.max(nx, ny, nz);
    for (let p = 0; p < nP; p++) {
        const px = particles[p * 3]!,
            py = particles[p * 3 + 1]!,
            pz = particles[p * 3 + 2]!;
        const ix = Math.min(nx - 1, Math.max(0, Math.floor((px - minx) * inv)));
        const iy = Math.min(ny - 1, Math.max(0, Math.floor((py - miny) * inv)));
        const iz = Math.min(nz - 1, Math.max(0, Math.floor((pz - minz) * inv)));
        let best = -1;
        let bestD2 = Infinity;
        for (let r = 0; r <= maxRing; r++) {
            const lox = ix - r,
                hix = ix + r,
                loy = iy - r,
                hiy = iy + r,
                loz = iz - r,
                hiz = iz + r;
            for (let cz = loz; cz <= hiz; cz++) {
                if (cz < 0 || cz >= nz) continue;
                const onZ = cz === loz || cz === hiz;
                for (let cy = loy; cy <= hiy; cy++) {
                    if (cy < 0 || cy >= ny) continue;
                    const onY = cy === loy || cy === hiy;
                    for (let cx = lox; cx <= hix; cx++) {
                        if (cx < 0 || cx >= nx) continue;
                        // Only scan the shell at Chebyshev distance r (avoid rescanning inner cells).
                        if (r > 0 && !onZ && !onY && cx !== lox && cx !== hix) continue;
                        const bucket = grid.get(cx + nx * (cy + ny * cz));
                        if (!bucket) continue;
                        for (const vi of bucket) {
                            const dx = px - verts[vi * 3]!,
                                dy = py - verts[vi * 3 + 1]!,
                                dz = pz - verts[vi * 3 + 2]!;
                            const d2 = dx * dx + dy * dy + dz * dz;
                            if (d2 < bestD2) {
                                bestD2 = d2;
                                best = vi;
                            }
                        }
                    }
                }
            }
            // Once a candidate is found, no closer vertex can exist beyond ring r when r·cell ≥ √bestD2.
            if (best >= 0 && r * cell * (r * cell) >= bestD2) break;
        }
        if (best >= 0) {
            out[p * 2] = vertUvs[best * 2]!;
            out[p * 2 + 1] = vertUvs[best * 2 + 1]!;
            if (vertTex) outTex[p] = vertTex[best]!;
        }
    }
    return { uvs: out, texIndices: outTex };
}

// Area-weighted surface sampling — a robust fallback for meshes that don't volume-sample (open /
// non-manifold / hollow shells like a barrel or a house). Scatters points across triangles
// proportional to area, with EXACT barycentric-interpolated UVs. Produces a thin water "shell"
// of the mesh shape (melts into a puddle), which is fine for these foes.
function sampleMeshSurface(positions: Float32Array, indices: Uint32Array, uvs: Float32Array | null, vertTex: Uint32Array | null, radius: number): { positions: Float32Array; uvs: Float32Array | null; texIndices: Uint32Array | null; count: number; renderRadius: number } {
    const nTri = Math.floor(indices.length / 3);
    let totalArea = 0;
    const triArea = new Float32Array(nTri);
    for (let t = 0; t < nTri; t++) {
        const a = indices[t * 3]! * 3,
            b = indices[t * 3 + 1]! * 3,
            c = indices[t * 3 + 2]! * 3;
        const e1x = positions[b]! - positions[a]!,
            e1y = positions[b + 1]! - positions[a + 1]!,
            e1z = positions[b + 2]! - positions[a + 2]!;
        const e2x = positions[c]! - positions[a]!,
            e2y = positions[c + 1]! - positions[a + 1]!,
            e2z = positions[c + 2]! - positions[a + 2]!;
        const cxp = e1y * e2z - e1z * e2y,
            cyp = e1z * e2x - e1x * e2z,
            czp = e1x * e2y - e1y * e2x;
        const area = 0.5 * Math.hypot(cxp, cyp, czp);
        triArea[t] = area;
        totalArea += area;
    }
    if (totalArea <= 0) return { positions: new Float32Array(0), uvs: uvs ? new Float32Array(0) : null, texIndices: vertTex ? new Uint32Array(0) : null, count: 0, renderRadius: radius };
    // One point per radius² of surface area (denser at smaller radius), capped so the sim stays
    // real-time. The RENDER radius returned below is the ACTUAL spacing, so splats always cover the
    // surface (no holes) even when the count is capped and the true spacing exceeds `radius`.
    const target = Math.min(SURFACE_MAX_POINTS, Math.max(64, Math.round(totalArea / (radius * radius))));
    const density = target / totalArea;
    const outPos: number[] = [];
    const outUv: number[] | null = uvs ? [] : null;
    const outTex: number[] | null = vertTex ? [] : null;
    for (let t = 0; t < nTri; t++) {
        const expected = density * triArea[t]!;
        let n = Math.floor(expected);
        if (Math.random() < expected - n) n++;
        if (n === 0) continue;
        const ia = indices[t * 3]!,
            ib = indices[t * 3 + 1]!,
            ic = indices[t * 3 + 2]!;
        const ti = vertTex ? vertTex[ia]! : 0; // a triangle lies within one sub-mesh → one texture
        for (let k = 0; k < n; k++) {
            let u = Math.random(),
                v = Math.random();
            if (u + v > 1) {
                u = 1 - u;
                v = 1 - v;
            }
            const w = 1 - u - v;
            outPos.push(
                positions[ia * 3]! * w + positions[ib * 3]! * u + positions[ic * 3]! * v,
                positions[ia * 3 + 1]! * w + positions[ib * 3 + 1]! * u + positions[ic * 3 + 1]! * v,
                positions[ia * 3 + 2]! * w + positions[ib * 3 + 2]! * u + positions[ic * 3 + 2]! * v
            );
            if (outUv) outUv.push(uvs![ia * 2]! * w + uvs![ib * 2]! * u + uvs![ic * 2]! * v, uvs![ia * 2 + 1]! * w + uvs![ib * 2 + 1]! * u + uvs![ic * 2 + 1]! * v);
            if (outTex) outTex.push(ti);
        }
    }
    const count = outPos.length / 3;
    // Actual surface point spacing → the render splat radius, so splats cover the surface even when
    // `count` was capped (spacing > requested radius). Equals `radius` when uncapped.
    const renderRadius = count > 0 ? Math.sqrt(totalArea / count) : radius;
    return { positions: new Float32Array(outPos), uvs: outUv ? new Float32Array(outUv) : null, texIndices: outTex ? new Uint32Array(outTex) : null, count, renderRadius };
}

ctx.addEventListener("message", (ev: MessageEvent<SampleRequest>) => {
    const { id, positions, indices, uvs, texIndices, radius, mode, surfaceOnly, ox, oy, oz } = ev.data;
    try {
        // Untouched copy of the mesh vertices (the volume sampler may read/scale `positions`; both
        // the nearest-vertex UV search and the surface fallback need the original geometry).
        const srcPos = positions.slice();
        // Hollow props (surfaceOnly) skip the slow, often-failing volume attempt and go straight to
        // the dense barycentric-UV surface shell (crisp texture detail). Otherwise try volume first;
        // it can THROW or return 0 for open / non-manifold meshes, so isolate it and fall back.
        let vres: ReturnType<typeof sampleMeshVolume> | null = null;
        let vradius = radius;
        if (!surfaceOnly) {
            try {
                vres = sampleMeshVolume({ positions, indices, radius, mode });
                vradius = vres.radius;
            } catch {
                vres = null;
            }
        }
        let outPos: Float32Array;
        let outUvs: Float32Array | null;
        let outTex: Uint32Array | null;
        let count: number;
        let bmin: [number, number, number];
        let bmax: [number, number, number];
        if (vres && vres.count > 0) {
            // Volume path: solid fill + nearest-vertex UVs + nearest-vertex texture index.
            outPos = vres.positions;
            if (uvs) {
                const r = computeParticleUvs(outPos, srcPos, uvs, texIndices, vres.radius);
                outUvs = r.uvs;
                outTex = texIndices ? r.texIndices : null;
            } else {
                outUvs = null;
                outTex = null;
            }
            count = vres.count;
            bmin = [vres.bounds.min[0], vres.bounds.min[1], vres.bounds.min[2]];
            bmax = [vres.bounds.max[0], vres.bounds.max[1], vres.bounds.max[2]];
        } else {
            // Surface fallback: works for any mesh, with exact barycentric UVs + per-triangle texture.
            const surf = sampleMeshSurface(srcPos, indices, uvs, texIndices, radius);
            outPos = surf.positions;
            outUvs = surf.uvs;
            outTex = surf.texIndices;
            count = surf.count;
            vradius = surf.renderRadius; // actual point spacing → covers the surface (no holes)
            bmin = [Infinity, Infinity, Infinity];
            bmax = [-Infinity, -Infinity, -Infinity];
            for (let i = 0; i < outPos.length; i += 3) {
                for (let k = 0; k < 3; k++) {
                    const v = outPos[i + k]!;
                    if (v < bmin[k]!) bmin[k] = v;
                    if (v > bmax[k]!) bmax[k] = v;
                }
            }
        }
        // Bake the mesh world offset into the sampled points → world-space seed.
        for (let i = 0; i < outPos.length; i += 3) {
            outPos[i] = outPos[i]! + ox;
            outPos[i + 1] = outPos[i + 1]! + oy;
            outPos[i + 2] = outPos[i + 2]! + oz;
        }
        const boundsMin = count > 0 ? [bmin[0] + ox, bmin[1] + oy, bmin[2] + oz] : [0, 0, 0];
        const boundsMax = count > 0 ? [bmax[0] + ox, bmax[1] + oy, bmax[2] + oz] : [0, 0, 0];
        const transfer: Transferable[] = [outPos.buffer];
        if (outUvs) transfer.push(outUvs.buffer);
        if (outTex) transfer.push(outTex.buffer);
        ctx.postMessage({ id, positions: outPos, uvs: outUvs, texIndices: outTex, count, radius: vradius, boundsMin, boundsMax }, transfer);
    } catch (err) {
        ctx.postMessage({ id, positions: new Float32Array(0), uvs: null, texIndices: null, count: 0, radius, boundsMin: [0, 0, 0], boundsMax: [0, 0, 0], error: String(err) });
    }
});
