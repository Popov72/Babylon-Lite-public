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

export interface MeshGroupAabb {
    min: [number, number, number];
    max: [number, number, number];
}

interface CachedMeshBounds {
    readonly mesh: Mesh;
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
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

/** Cache mesh-local bounds once and transform only their corners when the owner moves. */
export function meshGroupAabbProvider(meshes: readonly Mesh[]): () => MeshGroupAabb | null {
    const cached: CachedMeshBounds[] = [];
    for (const mesh of meshes) {
        const geometry = getMeshTriangles(mesh);
        if (!geometry?.positions.length) continue;
        let minX = Infinity;
        let minY = Infinity;
        let minZ = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        let maxZ = -Infinity;
        for (let index = 0; index < geometry.positions.length; index += 3) {
            const x = geometry.positions[index]!;
            const y = geometry.positions[index + 1]!;
            const z = geometry.positions[index + 2]!;
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            minZ = Math.min(minZ, z);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
            maxZ = Math.max(maxZ, z);
        }
        cached.push({
            mesh,
            min: [minX, minY, minZ],
            max: [maxX, maxY, maxZ],
        });
    }
    return () => {
        let minX = Infinity;
        let minY = Infinity;
        let minZ = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        let maxZ = -Infinity;
        for (const { mesh, min, max } of cached) {
            const world = mesh.worldMatrix;
            for (const x of [min[0], max[0]]) {
                for (const y of [min[1], max[1]]) {
                    for (const z of [min[2], max[2]]) {
                        const wx = world[0]! * x + world[4]! * y + world[8]! * z + world[12]!;
                        const wy = world[1]! * x + world[5]! * y + world[9]! * z + world[13]!;
                        const wz = world[2]! * x + world[6]! * y + world[10]! * z + world[14]!;
                        minX = Math.min(minX, wx);
                        minY = Math.min(minY, wy);
                        minZ = Math.min(minZ, wz);
                        maxX = Math.max(maxX, wx);
                        maxY = Math.max(maxY, wy);
                        maxZ = Math.max(maxZ, wz);
                    }
                }
            }
        }
        return cached.length === 0 ? null : { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
    };
}
