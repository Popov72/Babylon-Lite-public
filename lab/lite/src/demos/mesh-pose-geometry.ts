import { computeDeformedPositionToRef, type Mesh } from "babylon-lite";

export interface MeshPoseGeometry {
    positions: Float32Array;
    indices: Uint32Array;
    uvs: Float32Array | null;
}

/** Capture a mesh's current rendered pose as caller-owned world-space triangle geometry. */
export function getMeshPoseGeometry(mesh: Mesh): MeshPoseGeometry | null {
    const base = mesh._cpuPositions;
    const sourceIndices = mesh._cpuIndices;
    if (!base || !sourceIndices) {
        return null;
    }

    const positions = new Float32Array(base.length);
    const world = mesh.worldMatrix;
    const deform = !!(mesh.skeleton || mesh.morphTargets);
    const local = { x: 0, y: 0, z: 0 };
    for (let vertex = 0; vertex < base.length / 3; vertex++) {
        const source = vertex * 3;
        if (deform) {
            if (!computeDeformedPositionToRef(mesh, vertex, local)) {
                throw new Error(`Cannot capture the current pose of mesh "${mesh.name}" at vertex ${vertex}`);
            }
        } else {
            local.x = base[source]!;
            local.y = base[source + 1]!;
            local.z = base[source + 2]!;
        }
        positions[source] = world[0]! * local.x + world[4]! * local.y + world[8]! * local.z + world[12]!;
        positions[source + 1] = world[1]! * local.x + world[5]! * local.y + world[9]! * local.z + world[13]!;
        positions[source + 2] = world[2]! * local.x + world[6]! * local.y + world[10]! * local.z + world[14]!;
    }

    return {
        positions,
        indices: sourceIndices.slice(),
        uvs: mesh._cpuUvs?.slice() ?? null,
    };
}
