import { createBoxData, createMeshFromData, createStandardMaterial } from "babylon-lite";
import type { EngineContext, Mesh } from "babylon-lite";

/** Create the shared transparent grid-volume visual. Standard alpha blending keeps
 * reverse-Z depth testing enabled while disabling depth writes. */
export function createSolidGridBounds(engine: EngineContext, name: string): Mesh[] {
    const box = createBoxData(1);
    // createBoxData face order: +/-Z, +/-X, +/-Y. Axis colors keep adjoining
    // faces distinguishable even when the camera is inside the simulation domain.
    const faceColors = [
        [0.2, 0.45, 1],
        [0.2, 0.45, 1],
        [1, 0.25, 0.2],
        [1, 0.25, 0.2],
        [0.2, 1, 0.25],
        [0.2, 1, 0.25],
    ] as const;
    const faceNames = ["positive-z", "negative-z", "positive-x", "negative-x", "positive-y", "negative-y"] as const;
    const faceIndices = new Uint32Array([0, 1, 2, 0, 2, 3]);
    return faceColors.map((color, face) => {
        const vertex = face * 4;
        const mesh = createMeshFromData(
            engine,
            `${name}-${faceNames[face]}`,
            box.positions.slice(vertex * 3, (vertex + 4) * 3),
            box.normals.slice(vertex * 3, (vertex + 4) * 3),
            faceIndices,
            box.uvs.slice(vertex * 2, (vertex + 4) * 2)
        );
        const material = createStandardMaterial();
        material.diffuseColor = [1, 1, 1];
        material.emissiveColor = [color[0], color[1], color[2]];
        material.disableLighting = true;
        material.alpha = 0.3;
        material.backFaceCulling = false;
        mesh.material = material;
        mesh.pickable = false;
        mesh.renderOrder = 9_998;
        return mesh;
    });
}
