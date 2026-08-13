// World-space bounds of a group of meshes.
//
// This is all that survives of the SDF bake: the fluid no longer needs a distance grid per prop (it
// collides against the manifest's authored primitives instead), but a liquefiable prop still needs
// its own centre and half-extents — for the rigid-body proxy, the display root it is reparented
// under, and its mass/inertia.
//
// Bounds are taken from the TRIANGLES, transformed by each mesh's world matrix, rather than from
// `boundMin`/`boundMax`: loaded mesh bounds are object-local, while every caller here needs the
// world-space box.

import { getMeshTriangles, type Mesh } from "babylon-lite";

export interface MeshGroupBounds {
    centre: [number, number, number];
    half: [number, number, number];
}

/** World AABB of `meshes`, or null when none of them has readable CPU geometry. */
export function meshGroupBounds(meshes: readonly Mesh[]): MeshGroupBounds | null {
    let mnx = Infinity,
        mny = Infinity,
        mnz = Infinity;
    let mxx = -Infinity,
        mxy = -Infinity,
        mxz = -Infinity;
    let any = false;
    for (const m of meshes) {
        // Triangles only — normals are optional in glTF, and asking for them (as `getMeshGeometry`
        // does) silently drops every mesh that has none.
        const g = getMeshTriangles(m);
        if (!g || !g.positions.length) continue;
        any = true;
        const w = m.worldMatrix;
        const p = g.positions;
        for (let i = 0; i < p.length; i += 3) {
            const lx = p[i]!,
                ly = p[i + 1]!,
                lz = p[i + 2]!;
            const x = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
            const y = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
            const z = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
            if (x < mnx) mnx = x;
            if (y < mny) mny = y;
            if (z < mnz) mnz = z;
            if (x > mxx) mxx = x;
            if (y > mxy) mxy = y;
            if (z > mxz) mxz = z;
        }
    }
    if (!any) return null;
    return {
        centre: [(mnx + mxx) / 2, (mny + mxy) / 2, (mnz + mxz) / 2],
        half: [(mxx - mnx) / 2, (mxy - mny) / 2, (mxz - mnz) / 2],
    };
}
