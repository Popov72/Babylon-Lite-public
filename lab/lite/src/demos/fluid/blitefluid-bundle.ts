import type { FluidExportJson } from "./preset-io.js";

const ZIP_LOCAL_FILE = 0x04034b50;
const ZIP_CENTRAL_FILE = 0x02014b50;
const ZIP_END = 0x06054b50;
const SDF_MAGIC = 0x46534c42;
const SDF_HEADER_BYTES = 64;
const MAX_ENTRY_BYTES = 512 * 1024 * 1024;
const MAX_SDF_VOXELS = 16 * 1024 * 1024;

export interface BliteFluidManifest {
    bundleVersion: 1;
    preset: FluidExportJson;
    scene: {
        glb: "scene.glb";
        collision: "collision.blsdf";
    };
}

export interface BliteFluidCollision {
    dims: [number, number, number];
    origin: [number, number, number];
    cellSize: number;
    distances: Float32Array;
}

export interface BliteFluidBundle {
    manifest: BliteFluidManifest;
    sceneGlb: ArrayBuffer;
    collision: BliteFluidCollision;
}

function fail(message: string): never {
    throw new Error(`Invalid .blitefluid bundle: ${message}`);
}

function crc32(bytes: Uint8Array): number {
    let crc = 0xffffffff;
    for (const byte of bytes) {
        crc ^= byte;
        for (let bit = 0; bit < 8; bit++) {
            crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
        }
    }
    return (crc ^ 0xffffffff) >>> 0;
}

function parseStoredZip(data: ArrayBuffer): Map<string, Uint8Array> {
    const bytes = new Uint8Array(data);
    const view = new DataView(data);
    const decoder = new TextDecoder();
    const entries = new Map<string, Uint8Array>();
    let offset = 0;

    while (offset + 4 <= bytes.byteLength) {
        const signature = view.getUint32(offset, true);
        if (signature === ZIP_CENTRAL_FILE || signature === ZIP_END) {
            break;
        }
        if (signature !== ZIP_LOCAL_FILE || offset + 30 > bytes.byteLength) {
            fail(`invalid ZIP record at byte ${offset}`);
        }

        const flags = view.getUint16(offset + 6, true);
        const compression = view.getUint16(offset + 8, true);
        const expectedCrc = view.getUint32(offset + 14, true);
        const compressedSize = view.getUint32(offset + 18, true);
        const uncompressedSize = view.getUint32(offset + 22, true);
        const nameLength = view.getUint16(offset + 26, true);
        const extraLength = view.getUint16(offset + 28, true);
        if ((flags & 1) !== 0) fail("encrypted ZIP entries are not supported");
        if ((flags & 8) !== 0) fail("ZIP data descriptors are not supported");
        if (compression !== 0) fail("ZIP entries must use STORE compression");
        if (compressedSize !== uncompressedSize) fail("stored ZIP entry size mismatch");
        if (uncompressedSize > MAX_ENTRY_BYTES) fail("ZIP entry exceeds the 512 MiB limit");

        const nameStart = offset + 30;
        const payloadStart = nameStart + nameLength + extraLength;
        const payloadEnd = payloadStart + uncompressedSize;
        if (payloadStart > bytes.byteLength || payloadEnd > bytes.byteLength) fail("truncated ZIP entry");

        const name = decoder.decode(bytes.subarray(nameStart, nameStart + nameLength));
        if (!name || name.includes("\\") || name.startsWith("/") || name.split("/").includes("..")) {
            fail(`unsafe ZIP entry name "${name}"`);
        }
        if (entries.has(name)) fail(`duplicate ZIP entry "${name}"`);

        const payload = bytes.subarray(payloadStart, payloadEnd);
        if (crc32(payload) !== expectedCrc) fail(`CRC mismatch for "${name}"`);
        entries.set(name, payload);
        offset = payloadEnd;
    }

    return entries;
}

function requiredEntry(entries: Map<string, Uint8Array>, name: string): Uint8Array {
    const entry = entries.get(name);
    if (!entry) fail(`missing "${name}"`);
    return entry;
}

function parseManifest(bytes: Uint8Array): BliteFluidManifest {
    let value: unknown;
    try {
        value = JSON.parse(new TextDecoder().decode(bytes));
    } catch {
        fail("manifest.json is not valid JSON");
    }
    if (!value || typeof value !== "object") fail("manifest.json must contain an object");
    const manifest = value as Partial<BliteFluidManifest>;
    if (manifest.bundleVersion !== 1) fail("unsupported bundle version");
    if (!manifest.preset || typeof manifest.preset !== "object") fail("manifest preset is missing");
    if (manifest.scene?.glb !== "scene.glb" || manifest.scene.collision !== "collision.blsdf") {
        fail("manifest scene paths are invalid");
    }
    return manifest as BliteFluidManifest;
}

function parseGlb(bytes: Uint8Array): ArrayBuffer {
    if (bytes.byteLength < 12) fail("scene.glb is truncated");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== 0x46546c67) fail("scene.glb has invalid magic");
    if (view.getUint32(4, true) !== 2) fail("scene.glb must use glTF 2");
    if (view.getUint32(8, true) !== bytes.byteLength) fail("scene.glb length does not match its header");
    return bytes.slice().buffer;
}

export function parseBliteFluidCollision(bytes: Uint8Array): BliteFluidCollision {
    if (bytes.byteLength < SDF_HEADER_BYTES) fail("collision.blsdf is truncated");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== SDF_MAGIC) fail("collision.blsdf has invalid magic");
    if (view.getUint32(4, true) !== 1) fail("unsupported collision version");

    const dims: [number, number, number] = [view.getUint32(8, true), view.getUint32(12, true), view.getUint32(16, true)];
    if (dims.some((dim) => dim < 2 || dim > 2048)) fail("collision dimensions must be between 2 and 2048");
    const voxelCount = dims[0] * dims[1] * dims[2];
    if (!Number.isSafeInteger(voxelCount) || voxelCount > MAX_SDF_VOXELS) fail("collision grid exceeds the 16M-voxel limit");
    if (bytes.byteLength !== SDF_HEADER_BYTES + voxelCount * 4) fail("collision payload length is invalid");

    const origin: [number, number, number] = [view.getFloat32(24, true), view.getFloat32(28, true), view.getFloat32(32, true)];
    const cellSize = view.getFloat32(36, true);
    if (!origin.every(Number.isFinite) || !Number.isFinite(cellSize) || cellSize <= 0) fail("collision transform is invalid");

    const distances = new Float32Array(voxelCount);
    for (let index = 0; index < voxelCount; index++) {
        const distance = view.getFloat32(SDF_HEADER_BYTES + index * 4, true);
        if (!Number.isFinite(distance)) fail(`collision distance ${index} is not finite`);
        distances[index] = distance;
    }
    return { dims, origin, cellSize, distances };
}

export function parseBliteFluidBundle(data: ArrayBuffer): BliteFluidBundle {
    const entries = parseStoredZip(data);
    const expectedEntries = new Set(["manifest.json", "scene.glb", "collision.blsdf"]);
    if (entries.size !== expectedEntries.size || [...entries.keys()].some((name) => !expectedEntries.has(name))) {
        fail("archive must contain exactly manifest.json, scene.glb, and collision.blsdf");
    }
    const manifest = parseManifest(requiredEntry(entries, "manifest.json"));
    const sceneGlb = parseGlb(requiredEntry(entries, manifest.scene.glb));
    const collision = parseBliteFluidCollision(requiredEntry(entries, manifest.scene.collision));
    return { manifest, sceneGlb, collision };
}
