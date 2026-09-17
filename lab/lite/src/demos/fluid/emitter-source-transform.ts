import { mat4Compose, mat4Decompose, mat4Invert, mat4Multiply } from "babylon-lite";
import type { FluidTransform, Mat4 } from "babylon-lite";

export interface EmitterSourceTransform {
    readonly sourceWorld: readonly number[];
    readonly authored: FluidTransform;
    readonly localShape: Mat4;
}

export function createEmitterSourceTransform(sourceWorld: Mat4, authored: FluidTransform): EmitterSourceTransform {
    const inverse = mat4Invert(sourceWorld);
    if (!inverse) {
        throw new Error("[fluid] cannot bind an emitter to a singular source-node transform.");
    }
    const [x, y, z] = authored.position;
    const [qx, qy, qz, qw] = authored.rotation;
    const length = Math.hypot(qx, qy, qz, qw);
    if (!(length > 0) || !Number.isFinite(length)) {
        throw new Error("[fluid] cannot bind an emitter with an invalid authored rotation.");
    }
    const [sx, sy, sz] = authored.scale;
    const worldShape = mat4Compose(x, y, z, qx / length, qy / length, qz / length, qw / length, sx, sy, sz);
    return {
        sourceWorld: Array.from(sourceWorld),
        authored: { position: [...authored.position], rotation: [...authored.rotation], scale: [...authored.scale] },
        localShape: mat4Multiply(inverse, worldShape),
    };
}

export function resolveEmitterSourceTransform(binding: EmitterSourceTransform, sourceWorld: Mat4): FluidTransform {
    let atAuthoredPose = true;
    for (let index = 0; index < 16; index++) {
        if (sourceWorld[index] !== binding.sourceWorld[index]) {
            atAuthoredPose = false;
            break;
        }
    }
    if (atAuthoredPose) {
        return { position: [...binding.authored.position], rotation: [...binding.authored.rotation], scale: [...binding.authored.scale] };
    }
    const { translation, rotation, scale } = mat4Decompose(mat4Multiply(sourceWorld, binding.localShape));
    return {
        position: [translation.x, translation.y, translation.z],
        rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
        scale: [scale.x, scale.y, scale.z],
    };
}
