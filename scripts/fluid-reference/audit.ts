import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { decodeParticleState } from "./types";
import type { ReferenceCase } from "./types";
import { sampleSdf } from "./solids";

/** Check exported collision geometry against native markers before tuning a solver around it. */
export function auditReferenceGeometry(caseFile: string, frameIndex = 0) {
    const directory = dirname(caseFile);
    const input = JSON.parse(readFileSync(caseFile, "utf8")) as ReferenceCase;
    const floatFile = (file: string): Float32Array => {
        const bytes = readFileSync(resolve(directory, file));
        return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    };
    const staticGrid = input.staticSdf ? { ...input.staticSdf, distances: floatFile(input.staticSdf.file) } : null;
    const obstacles = input.obstacles.map((o) => ({ ...o, distances: floatFile(o.file) }));
    const name = input.referencePattern?.replace("{frame}", String(input.startFrame + frameIndex)) ?? input.initialState;
    const bytes = readFileSync(resolve(directory, name));
    const state = decodeParticleState(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const distances = new Float32Array(state.positions.length / 3);
    let inside = 0;
    let deeplyInside = 0;
    for (let p = 0; p < distances.length; p++) {
        const x = state.positions[p * 3]!;
        const y = state.positions[p * 3 + 1]!;
        const z = state.positions[p * 3 + 2]!;
        let distance = staticGrid ? sampleSdf(staticGrid, x, y, z) : 1e6;
        for (const obstacle of obstacles) {
            const m = obstacle.inverseTransforms[frameIndex * input.substeps]!;
            const lx = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
            const ly = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
            const lz = m[2]! * x + m[6]! * y + m[10]! * z + m[14]!;
            const invScale = Math.max(Math.hypot(m[0]!, m[4]!, m[8]!), Math.hypot(m[1]!, m[5]!, m[9]!), Math.hypot(m[2]!, m[6]!, m[10]!));
            distance = Math.min(distance, sampleSdf(obstacle, lx, ly, lz) / invScale);
        }
        distances[p] = distance;
        inside += Number(distance < 0);
        deeplyInside += Number(distance < -input.grid.cellSize * 0.25);
    }
    distances.sort();
    return {
        frame: input.startFrame + frameIndex,
        count: distances.length,
        markersInsideExportedSolids: inside,
        markersMoreThanQuarterCellInside: deeplyInside,
        minimumSolidDistance: distances[0],
        p01SolidDistance: distances[Math.floor(distances.length * 0.01)],
        medianSolidDistance: distances[Math.floor(distances.length * 0.5)],
        gridCellSize: input.grid.cellSize,
    };
}
