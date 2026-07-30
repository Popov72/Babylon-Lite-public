/**
 * Mesh → particle sampling worker, shared by the Aquanova and Liquefactor demos.
 *
 * The CPU particle fill is heavy — densely sampling a foe can block the main thread for ~1 s,
 * freezing the frame at the moment of the shot. Running it here keeps the render loop smooth: the
 * main thread posts the (copied) mesh geometry + params, we fill, bake the mesh's world offset into
 * the points, and post the world-space seed back (transferring the buffer). Pure CPU, no GPU/DOM —
 * safe in a worker.
 *
 * Strategy selection (volume lattice vs surface shell) and the per-particle UV lookup both live in
 * `particle-fill.ts`, so a seed is identical whether it was produced here or on the main thread.
 *
 * Protocol (main → worker): { id, positions, indices, uvs, radius, mode, surfaceOnly, strategy, spacing, ox, oy, oz }
 * Protocol (worker → main): { id, positions, uvs, texIndices, count, radius, shell, boundsMin, boundsMax }
 */
import type { VolumeSamplingMode } from "babylon-lite/fluid/volume-sampling/index.js";
import { fillMeshParticles } from "./particle-fill.js";
import type { MeshFillStrategy } from "./particle-fill.js";

const ctx = self as unknown as Worker;

interface SampleRequest {
    id: number;
    positions: Float32Array;
    indices: Uint32Array;
    uvs: Float32Array | null;
    texIndices: Uint32Array | null;
    radius: number;
    mode: VolumeSamplingMode;
    surfaceOnly: boolean;
    strategy?: MeshFillStrategy;
    spacing?: number;
    ox: number;
    oy: number;
    oz: number;
}

ctx.addEventListener("message", (ev: MessageEvent<SampleRequest>) => {
    const { id, positions, indices, uvs, texIndices, radius, mode, surfaceOnly, strategy, spacing, ox, oy, oz } = ev.data;
    try {
        // Volume lattice for a solid mesh, surface shell for one with no interior to fill — the
        // ship's wall/floor panels are open single-sided geometry, so the choice is automatic unless
        // the caller forced one path via `strategy`.
        const fill = fillMeshParticles({ positions, indices, uvs, texIndices, radius, mode, surfaceOnly, strategy, spacing });
        const outPos = fill.positions;
        // Bake the mesh world offset into the sampled points → world-space seed.
        for (let i = 0; i < outPos.length; i += 3) {
            outPos[i] = outPos[i]! + ox;
            outPos[i + 1] = outPos[i + 1]! + oy;
            outPos[i + 2] = outPos[i + 2]! + oz;
        }
        const boundsMin = fill.count > 0 ? [fill.bounds.min[0] + ox, fill.bounds.min[1] + oy, fill.bounds.min[2] + oz] : [0, 0, 0];
        const boundsMax = fill.count > 0 ? [fill.bounds.max[0] + ox, fill.bounds.max[1] + oy, fill.bounds.max[2] + oz] : [0, 0, 0];
        const transfer: Transferable[] = [outPos.buffer];
        if (fill.uvs) transfer.push(fill.uvs.buffer);
        if (fill.texIndices) transfer.push(fill.texIndices.buffer);
        ctx.postMessage({ id, positions: outPos, uvs: fill.uvs, texIndices: fill.texIndices, count: fill.count, radius: fill.radius, shell: fill.shell, boundsMin, boundsMax }, transfer);
    } catch (err) {
        ctx.postMessage({ id, positions: new Float32Array(0), uvs: null, texIndices: null, count: 0, radius, shell: false, boundsMin: [0, 0, 0], boundsMax: [0, 0, 0], error: String(err) });
    }
});
