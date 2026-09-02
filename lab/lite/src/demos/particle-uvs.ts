import { transferFluidMeshParticleUvs } from "babylon-lite";
import type { FluidParticleUvTransferResult } from "babylon-lite";

export type ParticleUvResult = FluidParticleUvTransferResult;

export function computeParticleUvs(
    particles: Float32Array,
    count: number,
    verts: Float32Array,
    indices: Uint32Array,
    vertUvs: Float32Array,
    vertTex: Uint32Array | null
): ParticleUvResult {
    return transferFluidMeshParticleUvs({
        particles,
        count,
        positions: verts,
        indices,
        uvs: vertUvs,
        texIndices: vertTex,
    });
}
