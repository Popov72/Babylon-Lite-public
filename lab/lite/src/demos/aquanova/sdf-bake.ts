// Baking and reading the world-space signed-distance grids the fluid collides against.
//
// A bake is only ever as good as its SIGN. `generateMeshSdf` decides inside/outside from the parity
// of triangle crossings, which is exact for a closed surface and unreliable for kit-bashed level
// geometry — hence the `signMode` / `despeckle` knobs the callers pass. See sdf-gen.ts for what each
// mode is good for.

import { getMeshTriangles, type Mesh } from "babylon-lite";
import { generateMeshSdf } from "babylon-lite/fluid/volume-sampling/index.js";

export type BakedSdf = ReturnType<typeof generateMeshSdf>;
/** A baked grid together with the world-space centre it is posed at, and its half-extents. */
export interface BakedBody {
    grid: BakedSdf;
    centre: [number, number, number];
    half: [number, number, number];
}
/** Value at or above this means "outside this body's grid box" — the shader SKIPS the body there, so
 *  it contributes neither surface nor solid. It is NOT a large distance. */
export const SDF_OUTSIDE = 1e8;

const SDF_PAD = 2;
// Bake meshes' WORLD-space geometry into a signed-distance grid centred at the combined AABB centre.
export function bakeWorldSdf(meshes: Mesh[], cell: number, nameOf: (m: Mesh) => string, opt?: { signMode?: "parity" | "flood" | "normal"; despeckle?: number }): BakedBody | null {
    const geos: Array<{ pos: Float32Array; idx: Uint32Array; m: Mesh }> = [];
    let nV = 0;
    let nI = 0;
    const skipped: string[] = [];
    for (const m of meshes) {
        // Triangles only — a bake has no use for normals, and glTF makes them optional. Asking for
        // them (as `getMeshGeometry` does) silently drops every mesh that has none, which showed up
        // as one half of a two-primitive pipe missing from the collision.
        const g = getMeshTriangles(m);
        if (!g || !g.positions.length || !g.indices.length) {
            skipped.push(nameOf(m));
            continue;
        }
        geos.push({ pos: g.positions, idx: g.indices, m });
        nV += g.positions.length;
        nI += g.indices.length;
    }
    if (skipped.length) {
        // eslint-disable-next-line no-console
        console.warn(`[aquanova] SDF bake skipped ${skipped.length} mesh(es) with no CPU geometry: ${skipped.slice(0, 12).join(", ")}${skipped.length > 12 ? " …" : ""}`);
    }
    if (nV === 0 || nI === 0) return null;
    const world = new Float32Array(nV);
    const idx = new Uint32Array(nI);
    let pOff = 0;
    let iOff = 0;
    let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
    for (const { pos, idx: ix, m } of geos) {
        const w = m.worldMatrix;
        const base = pOff / 3;
        for (let i = 0; i < pos.length; i += 3) {
            const lx = pos[i]!, ly = pos[i + 1]!, lz = pos[i + 2]!;
            const x = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
            const y = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
            const z = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
            world[pOff++] = x; world[pOff++] = y; world[pOff++] = z;
            mnx = Math.min(mnx, x); mny = Math.min(mny, y); mnz = Math.min(mnz, z);
            mxx = Math.max(mxx, x); mxy = Math.max(mxy, y); mxz = Math.max(mxz, z);
        }
        for (let i = 0; i < ix.length; i++) idx[iOff++] = ix[i]! + base;
    }
    const centre: [number, number, number] = [(mnx + mxx) / 2, (mny + mxy) / 2, (mnz + mxz) / 2];
    for (let i = 0; i < nV; i += 3) { world[i] = world[i]! - centre[0]; world[i + 1] = world[i + 1]! - centre[1]; world[i + 2] = world[i + 2]! - centre[2]; }
    const half: [number, number, number] = [(mxx - mnx) / 2, (mxy - mny) / 2, (mxz - mnz) / 2];
    const grid = generateMeshSdf(world, idx, { min: [-half[0], -half[1], -half[2]], max: [half[0], half[1], half[2]], cellSize: cell, padding: SDF_PAD, ...opt });
    return { grid, centre, half };
}

/** Trilinear sample of a baked grid at a WORLD point, mirroring `bodiesSdf`'s `fbSampleG`.
 *  `quat` is the body's rotation (identity for rooms). Returns 1e9 outside the grid box, which is
 *  what the shader's union contributes there — i.e. "no solid", NOT "far from solid". */
export function sampleBakedSdf(g: BakedSdf, centre: readonly [number, number, number], x: number, y: number, z: number, quat?: readonly [number, number, number, number]): number {
    const { data, dims, origin, cellSize } = g;
    let dx = x - centre[0], dy = y - centre[1], dz = z - centre[2];
    if (quat) {
        // local = conj(q) · (pt − pos)
        const [qx, qy, qz, qw] = quat;
        const tx = 2 * (-qy * dz + qz * dy), ty = 2 * (-qz * dx + qx * dz), tz = 2 * (-qx * dy + qy * dx);
        dx += qw * tx + (-qy * tz + qz * ty);
        dy += qw * ty + (-qz * tx + qx * tz);
        dz += qw * tz + (-qx * ty + qy * tx);
    }
    const gx = (dx - origin[0]) / cellSize, gy = (dy - origin[1]) / cellSize, gz = (dz - origin[2]) / cellSize;
    if (gx < 0 || gy < 0 || gz < 0 || gx > dims[0] - 1 || gy > dims[1] - 1 || gz > dims[2] - 1) return 1e9;
    const i = Math.floor(gx), j = Math.floor(gy), k = Math.floor(gz);
    const fx = gx - i, fy = gy - j, fz = gz - k;
    const at = (a: number, b: number, c: number): number => data[a + dims[0] * (b + dims[1] * c)]!;
    const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
    const x00 = lerp(at(i, j, k), at(i + 1, j, k), fx);
    const x10 = lerp(at(i, j + 1, k), at(i + 1, j + 1, k), fx);
    const x01 = lerp(at(i, j, k + 1), at(i + 1, j, k + 1), fx);
    const x11 = lerp(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), fx);
    return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz);
}
