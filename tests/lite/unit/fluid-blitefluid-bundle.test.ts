import { describe, expect, it } from "vitest";

import { parseBliteFluidBundle, parseBliteFluidCollision } from "../../../lab/lite/src/demos/fluid/blitefluid-bundle";

function crc32(bytes: Uint8Array): number {
    let crc = 0xffffffff;
    for (const byte of bytes) {
        crc ^= byte;
        for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
    return (crc ^ 0xffffffff) >>> 0;
}

function collisionBytes(): Uint8Array {
    const bytes = new Uint8Array(64 + 8 * 4);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x46534c42, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, 2, true);
    view.setUint32(12, 2, true);
    view.setUint32(16, 2, true);
    view.setFloat32(24, -1, true);
    view.setFloat32(28, -2, true);
    view.setFloat32(32, -3, true);
    view.setFloat32(36, 0.5, true);
    for (let index = 0; index < 8; index++) view.setFloat32(64 + index * 4, index - 4, true);
    return bytes;
}

function glbBytes(): Uint8Array {
    const bytes = new Uint8Array(12);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x46546c67, true);
    view.setUint32(4, 2, true);
    view.setUint32(8, 12, true);
    return bytes;
}

function storedZip(entries: Record<string, Uint8Array>): ArrayBuffer {
    const encoder = new TextEncoder();
    const parts: Uint8Array[] = [];
    let size = 0;
    for (const [name, payload] of Object.entries(entries)) {
        const nameBytes = encoder.encode(name);
        const header = new Uint8Array(30 + nameBytes.byteLength);
        const view = new DataView(header.buffer);
        view.setUint32(0, 0x04034b50, true);
        view.setUint16(4, 20, true);
        view.setUint32(14, crc32(payload), true);
        view.setUint32(18, payload.byteLength, true);
        view.setUint32(22, payload.byteLength, true);
        view.setUint16(26, nameBytes.byteLength, true);
        header.set(nameBytes, 30);
        parts.push(header, payload);
        size += header.byteLength + payload.byteLength;
    }
    const output = new Uint8Array(size);
    let offset = 0;
    for (const part of parts) {
        output.set(part, offset);
        offset += part.byteLength;
    }
    return output.buffer;
}

describe(".blitefluid bundle", () => {
    it("parses a stored ZIP with preset, GLB, and signed-distance data", () => {
        const manifest = new TextEncoder().encode(
            JSON.stringify({
                bundleVersion: 1,
                preset: { formatVersion: 5, meta: { demo: "blender", method: "PBF" } },
                scene: { glb: "scene.glb", collision: "collision.blsdf" },
            })
        );
        const bundle = parseBliteFluidBundle(storedZip({ "manifest.json": manifest, "scene.glb": glbBytes(), "collision.blsdf": collisionBytes() }));

        expect(bundle.manifest.preset.meta.method).toBe("PBF");
        expect(bundle.collision.dims).toEqual([2, 2, 2]);
        expect(bundle.collision.origin).toEqual([-1, -2, -3]);
        expect([...bundle.collision.distances]).toEqual([-4, -3, -2, -1, 0, 1, 2, 3]);
    });

    it("rejects truncated collision payloads", () => {
        expect(() => parseBliteFluidCollision(collisionBytes().subarray(0, -4))).toThrow("payload length");
    });

    it("rejects corrupted ZIP entries", () => {
        const archive = new Uint8Array(storedZip({ "scene.glb": glbBytes() }));
        archive[archive.length - 1] = archive[archive.length - 1]! ^ 1;
        expect(() => parseBliteFluidBundle(archive.buffer)).toThrow("CRC mismatch");
    });
});
