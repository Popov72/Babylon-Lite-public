import { describe, expect, it } from "vitest";

import { getMeshPoseGeometry } from "../../../../lab/lite/src/demos/mesh-pose-geometry";
import type { SkeletonData } from "../../../../packages/babylon-lite/src/animation/types";
import type { Mesh } from "../../../../packages/babylon-lite/src/mesh/mesh";

function translated(x: number, y: number, z: number): Float32Array {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

describe("liquefaction mesh pose geometry", () => {
    it("captures skeletal deformation from the paused animation frame before applying the mesh world matrix", () => {
        const mesh = {
            name: "chest",
            _cpuPositions: new Float32Array([0, 0, 0, 1, 0, 0]),
            _cpuIndices: new Uint32Array([0, 1, 1]),
            _cpuUvs: new Float32Array([0, 0, 1, 0]),
            worldMatrix: translated(10, 0, 0),
            skeleton: {
                joints: new Uint8Array([0, 0, 0, 0, 1, 0, 0, 0]),
                weights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0]),
                joints1: null,
                weights1: null,
                boneMatrices: new Float32Array([...translated(0, 0, 0), ...translated(0, 2, 0)]),
            } as SkeletonData,
        } as unknown as Mesh;

        const geometry = getMeshPoseGeometry(mesh);

        expect(geometry?.positions).toEqual(new Float32Array([10, 0, 0, 11, 2, 0]));
        expect(geometry?.indices).toEqual(mesh._cpuIndices);
        expect(geometry?.indices).not.toBe(mesh._cpuIndices);
        expect(geometry?.uvs).toEqual(mesh._cpuUvs);
        expect(geometry?.uvs).not.toBe(mesh._cpuUvs);
    });
});
