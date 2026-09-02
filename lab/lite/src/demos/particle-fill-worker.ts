/**
 * Mesh → particle sampling worker, shared by the Aquanova and Liquefactor demos.
 *
 * The CPU particle fill is heavy — densely sampling a foe can block the main thread for ~1 s,
 * freezing the frame at the moment of the shot. Running it here keeps the render loop smooth: the
 * main thread posts the (copied) mesh geometry + params, we fill, bake the mesh's world offset into
 * the points, and post the world-space seed back (transferring the buffer). Pure CPU, no GPU/DOM —
 * safe in a worker.
 *
 * Strategy selection and UV transfer live in Babylon Lite core. This worker only schedules the
 * computation, applies the requested world offset, and transports the result.
 *
 * Protocol (main → worker): { id, positions, indices, uvs, radius, mode, surfaceOnly, strategy, spacing, ox, oy, oz }
 * Protocol (worker → main): a `ParticleFillWorkerResponse`, discriminated by `error`.
 */
import { fluidMeshSamplingErrorInfo, sampleFluidMeshParticles } from "babylon-lite";
import type { FluidMeshSamplingErrorInfo, FluidMeshSamplingStrategy, FluidMeshSamplingWarning, VolumeSamplingMode } from "babylon-lite";

const ctx = self as unknown as Worker;

export interface ParticleFillWorkerRequest {
    id: number;
    positions: Float32Array;
    indices: Uint32Array;
    uvs: Float32Array | null;
    texIndices: Uint32Array | null;
    radius: number;
    mode: VolumeSamplingMode;
    surfaceOnly: boolean;
    strategy?: FluidMeshSamplingStrategy;
    spacing?: number;
    ox: number;
    oy: number;
    oz: number;
}

export interface ParticleFillWorkerSuccess {
    id: number;
    error: null;
    positions: Float32Array;
    uvs: Float32Array | null;
    texIndices: Uint32Array | null;
    count: number;
    radius: number;
    shell: boolean;
    boundsMin: [number, number, number];
    boundsMax: [number, number, number];
    warnings: FluidMeshSamplingWarning[];
}

export interface ParticleFillWorkerFailure {
    id: number;
    error: FluidMeshSamplingErrorInfo;
}

export type ParticleFillWorkerResponse = ParticleFillWorkerSuccess | ParticleFillWorkerFailure;

ctx.addEventListener("message", (ev: MessageEvent<ParticleFillWorkerRequest>) => {
    const { id, positions, indices, uvs, texIndices, radius, mode, surfaceOnly, strategy, spacing, ox, oy, oz } = ev.data;
    try {
        // Volume lattice for a solid mesh, surface shell for one with no interior to fill — the
        // ship's wall/floor panels are open single-sided geometry, so the choice is automatic unless
        // the caller forced one path via `strategy`.
        const fill = sampleFluidMeshParticles({ positions, indices, uvs, texIndices, radius, mode, surfaceOnly, strategy, spacing });
        const outPos = fill.positions;
        // Bake the mesh world offset into the sampled points → world-space seed.
        for (let i = 0; i < outPos.length; i += 3) {
            outPos[i] = outPos[i]! + ox;
            outPos[i + 1] = outPos[i + 1]! + oy;
            outPos[i + 2] = outPos[i + 2]! + oz;
        }
        const boundsMin: [number, number, number] = [fill.bounds.min[0] + ox, fill.bounds.min[1] + oy, fill.bounds.min[2] + oz];
        const boundsMax: [number, number, number] = [fill.bounds.max[0] + ox, fill.bounds.max[1] + oy, fill.bounds.max[2] + oz];
        const transfer: Transferable[] = [outPos.buffer];
        if (fill.uvs) transfer.push(fill.uvs.buffer);
        if (fill.texIndices) transfer.push(fill.texIndices.buffer);
        const response: ParticleFillWorkerSuccess = {
            id,
            error: null,
            positions: outPos,
            uvs: fill.uvs,
            texIndices: fill.texIndices,
            count: fill.count,
            radius: fill.radius,
            shell: fill.shell,
            boundsMin,
            boundsMax,
            warnings: fill.warnings,
        };
        ctx.postMessage(response, transfer);
    } catch (err) {
        const response: ParticleFillWorkerFailure = {
            id,
            error: fluidMeshSamplingErrorInfo(err, strategy ?? "auto"),
        };
        ctx.postMessage(response);
    }
});
