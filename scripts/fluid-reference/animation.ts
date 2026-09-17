import { mat4Invert } from "../../packages/babylon-lite/src/math/mat4-invert";
import { mat4Scale } from "../../packages/babylon-lite/src/math/mat4-scale";
import type { Mat4, Mat4Storage } from "../../packages/babylon-lite/src/math/types";

/** Match native mesh-vertex interpolation and the cache's first-substep sample convention. */
export function sampleReferenceObstacleFrames(frames: readonly Mat4[], substeps: number, simulationFps: number) {
    if (frames.length === 0 || !Number.isSafeInteger(substeps) || substeps <= 0 || !Number.isFinite(simulationFps) || simulationFps <= 0) {
        throw new Error("Obstacle sampling requires frame matrices, positive substeps, and physical FPS.");
    }
    const transforms: number[][] = [];
    const inverseTransforms: number[][] = [];
    const velocityTransforms: number[][] = [];
    for (let step = 0; step <= (frames.length - 1) * substeps; step++) {
        const frame = Math.floor(step / substeps);
        const alpha = step / substeps - frame;
        const current = frames[frame]!;
        const previous = frames[Math.max(0, frame - 1)]!;
        const next = frames[Math.min(frames.length - 1, frame + 1)]!;
        const world = mat4Scale(1, 1, 1);
        const values = world as unknown as Mat4Storage;
        const derivative: number[] = [];
        for (let j = 0; j < 16; j++) {
            values[j] = (1 - alpha) * current[j]! + alpha * next[j]!;
            derivative.push(((1 - alpha) * (current[j]! - previous[j]!) + alpha * (next[j]! - current[j]!)) * simulationFps);
        }
        const inverse = mat4Invert(world);
        if (!inverse) {
            throw new Error(`Singular interpolated collision transform at substep ${step}.`);
        }
        transforms.push(Array.from({ length: 16 }, (_, j) => world[j]!));
        inverseTransforms.push(Array.from({ length: 16 }, (_, j) => inverse[j]!));
        velocityTransforms.push(derivative);
    }
    return { transforms, inverseTransforms, velocityTransforms };
}
