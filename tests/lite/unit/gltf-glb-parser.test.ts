import { describe, expect, it } from "vitest";

import { parseGlbContainer } from "../../../packages/babylon-lite/src/loader-gltf/gltf-glb-parser";

function jsonOnlyGlb(json: unknown): ArrayBuffer {
    const encoded = new TextEncoder().encode(JSON.stringify(json));
    const jsonLength = (encoded.byteLength + 3) & ~3;
    const buffer = new ArrayBuffer(20 + jsonLength);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    view.setUint32(0, 0x46546c67, true);
    view.setUint32(4, 2, true);
    view.setUint32(8, buffer.byteLength, true);
    view.setUint32(12, jsonLength, true);
    view.setUint32(16, 0x4e4f534a, true);
    bytes.fill(0x20, 20);
    bytes.set(encoded, 20);
    return buffer;
}

describe("GLB container parsing", () => {
    it("accepts a JSON-only GLB without a BIN chunk", () => {
        const parsed = parseGlbContainer(jsonOnlyGlb({ asset: { version: "2.0" }, scenes: [{ nodes: [] }] }));

        expect(parsed.json.asset.version).toBe("2.0");
        expect(parsed.binChunk.byteLength).toBe(0);
    });

    it("reports a truncated optional BIN header without leaking a DataView RangeError", () => {
        const complete = new Uint8Array(jsonOnlyGlb({ asset: { version: "2.0" } }));
        const truncated = new Uint8Array(complete.byteLength + 4);
        truncated.set(complete);

        expect(() => parseGlbContainer(truncated.buffer)).toThrow("GLB BIN chunk header is truncated");
    });
});
