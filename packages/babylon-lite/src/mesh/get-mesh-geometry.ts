import type { Mesh } from "./mesh.js";

/**
 * Return caller-owned copies of a mesh's triangle soup — positions and indices only.
 *
 * Unlike {@link getMeshGeometry} this does NOT require normals, which glTF makes optional and which a
 * geometric consumer has no use for: SDF baking, volume sampling and collision proxies all want the
 * triangles and nothing else. Requiring normals silently drops those meshes from the result, which
 * shows up much later as a hole in whatever was built from it.
 *
 * Returns `null` when positions or indices are not retained on the CPU.
 */
export function getMeshTriangles(mesh: Mesh): { positions: Float32Array; indices: Uint32Array } | null {
    const positions = mesh._cpuPositions;
    const indices = mesh._cpuIndices;
    if (!positions || !indices) {
        return null;
    }
    return { positions: positions.slice(), indices: indices.slice() };
}

/** Return caller-owned copies of the CPU geometry retained by a mesh.
 *  Returns `null` when positions, normals, or indices are unavailable. */
export function getMeshGeometry(mesh: Mesh): {
    positions: Float32Array;
    normals: Float32Array;
    indices: Uint32Array;
    uvs?: Float32Array;
    uvs2?: Float32Array;
    tangents?: Float32Array;
    colors?: Float32Array;
} | null {
    const positions = mesh._cpuPositions;
    const normals = mesh._cpuNormals;
    const indices = mesh._cpuIndices;
    if (!positions || !normals || !indices) {
        return null;
    }

    const uvs = mesh._cpuUvs;
    const uvs2 = mesh._cpuUv2s;
    const tangents = mesh._cpuTangents;
    const colors = mesh._cpuColors;
    return {
        positions: positions.slice(),
        normals: normals.slice(),
        indices: indices.slice(),
        ...(uvs ? { uvs: uvs.slice() } : {}),
        ...(uvs2 ? { uvs2: uvs2.slice() } : {}),
        ...(tangents ? { tangents: tangents.slice() } : {}),
        ...(colors ? { colors: colors.slice() } : {}),
    };
}
