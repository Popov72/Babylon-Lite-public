import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { generateMeshSdf } from "../packages/babylon-lite/src/fluid/sampling/volume/sdf-gen.js";
import { evaluateJumpingWhaleFrames } from "./jumping-whale-sdf-source.js";

const ROOT = resolve(import.meta.dirname, "..");
const INPUT = resolve(ROOT, "lab/public/jumping-whale/whale.glb");
const OUTPUT = resolve(ROOT, "lab/public/jumping-whale/whale-sdf.bin");
const METADATA = resolve(ROOT, "lab/public/jumping-whale/whale-sdf.meta.json");
const FRAME_RATE = 12;
const CELL_SIZE = 0.25;
const PADDING = 2;
const HEADER_BYTES = 64;

function float32ToFloat16(value: number, floatView: Float32Array, uintView: Uint32Array): number {
    floatView[0] = value;
    const bits = uintView[0]!;
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    const mantissa = bits & 0x7fffff;
    if (exponent === 0xff) {
        return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
    }
    const halfExponent = exponent - 127 + 15;
    if (halfExponent >= 0x1f) {
        return sign | 0x7bff;
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return sign;
        }
        const shifted = (mantissa | 0x800000) >>> (1 - halfExponent);
        return sign | ((shifted + 0x1000) >>> 13);
    }
    const rounded = mantissa + 0x1000;
    if ((rounded & 0x800000) !== 0) {
        const nextExponent = halfExponent + 1;
        return nextExponent >= 0x1f ? sign | 0x7bff : sign | (nextExponent << 10);
    }
    return sign | (halfExponent << 10) | (rounded >>> 13);
}

function packFrame(distances: Float32Array): Uint32Array {
    const packed = new Uint32Array(Math.ceil(distances.length / 2));
    const floatView = new Float32Array(1);
    const uintView = new Uint32Array(floatView.buffer);
    for (let index = 0; index < distances.length; index++) {
        const half = float32ToFloat16(distances[index]!, floatView, uintView);
        const word = index >>> 1;
        packed[word] = (index & 1) === 0 ? half : packed[word]! | (half << 16);
    }
    return packed;
}

if (!existsSync(INPUT)) {
    throw new Error(`Jumping whale source asset is missing at ${INPUT}.`);
}

const input = readFileSync(INPUT);
const source = evaluateJumpingWhaleFrames(input, FRAME_RATE);
const boundsMin: [number, number, number] = [Infinity, Infinity, Infinity];
const boundsMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
for (const positions of source.frames) {
    for (let index = 0; index < positions.length; index++) {
        const axis = index % 3;
        boundsMin[axis] = Math.min(boundsMin[axis], positions[index]!);
        boundsMax[axis] = Math.max(boundsMax[axis], positions[index]!);
    }
}
const origin = boundsMin.map((value) => Math.floor(value / CELL_SIZE) * CELL_SIZE - PADDING * CELL_SIZE) as [number, number, number];
const dims = boundsMax.map((value, axis) => Math.ceil((value - origin[axis]!) / CELL_SIZE) + PADDING + 1) as [number, number, number];
const voxelCount = dims[0] * dims[1] * dims[2];
const wordsPerFrame = Math.ceil(voxelCount / 2);
const output = Buffer.allocUnsafe(HEADER_BYTES + source.frameCount * wordsPerFrame * 4);
output.fill(0, 0, HEADER_BYTES);
output.write("BLWSDF1", 0, "ascii");
output.writeUInt32LE(source.frameCount, 8);
output.writeUInt32LE(source.frameRate, 12);
output.writeUInt32LE(dims[0], 16);
output.writeUInt32LE(dims[1], 20);
output.writeUInt32LE(dims[2], 24);
output.writeFloatLE(origin[0], 28);
output.writeFloatLE(origin[1], 32);
output.writeFloatLE(origin[2], 36);
output.writeFloatLE(CELL_SIZE, 40);
output.writeUInt32LE(wordsPerFrame, 44);
output.writeFloatLE(source.duration, 48);
for (let frame = 0; frame < source.frameCount; frame++) {
    const grid = generateMeshSdf(source.frames[frame]!, source.indices, {
        min: origin,
        dims,
        cellSize: CELL_SIZE,
        exactBand: 2,
        sweepPasses: 2,
        signMode: "parity",
    });
    const packed = packFrame(grid.data);
    Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength).copy(output, HEADER_BYTES + frame * wordsPerFrame * 4);
    console.log(`Baked deforming SDF ${frame + 1}/${source.frameCount}`);
}
writeFileSync(OUTPUT, output);
writeFileSync(
    METADATA,
    `${JSON.stringify(
        {
            formatVersion: 2,
            generator: "native-gltf-skinning",
            sourceSha256: createHash("sha256").update(input).digest("hex"),
            sdfSha256: createHash("sha256").update(output).digest("hex"),
            frameCount: source.frameCount,
            frameRate: source.frameRate,
            dims,
            origin,
            cellSize: CELL_SIZE,
        },
        null,
        2
    )}\n`
);
console.log(`Wrote ${OUTPUT} (${(output.byteLength / 1048576).toFixed(2)} MiB, ${dims.join("x")}, ${source.frameCount} frames at ${source.frameRate} fps)`);
