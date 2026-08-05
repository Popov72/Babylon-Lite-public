/**
 * Strip `"doubleSided": true` from a GLB's materials, in place.
 *
 * WHY THIS EXISTS
 * ---------------
 * The waterfall's source models (rock + oasis) ship every material flagged `doubleSided`, which is
 * a Blender/exporter default rather than a property these assets actually need: they are closed
 * solids, and an A/B of the rendered demo showed disabling it changes less than the fluid sim's own
 * run-to-run noise (mean 3.28/255 against a 4.05/255 same-settings control).
 *
 * Carrying the flag is not free, and not only for the obvious back-face-culling reason:
 * `doubleSided` also switches the PBR shader onto its two-sided path, which flips the shading
 * normal on `!front_facing`. That flip is only correct if the pipeline's `frontFace` matches the
 * mesh's real triangle winding — and the demo mirrors these models (it overwrites the loader's
 * RH→LH `__root__` flip so the geometry lines up with `rock-heightmap.bin`, which is baked in raw
 * glTF space). Getting that combination wrong inverted the shading normal on every visible face and
 * lit the whole formation as though the sun were underneath it. Single-sided materials have no such
 * path to get wrong.
 *
 * WHAT IT PRESERVES
 * -----------------
 * Only the JSON chunk is rewritten, and it is re-padded with spaces (0x20, as the glTF spec
 * requires) back to its ORIGINAL byte length. The BIN chunk, every chunk header and the file length
 * are therefore untouched, so the mesh data hashes identically and `rock-heightmap.bin` stays valid
 * — the same guarantee the AO bake was careful to keep (see waterfall.ts).
 *
 * USAGE
 *   npx tsx lab/public/waterfall/scripts/strip-double-sided.ts <file.glb> [more.glb ...]
 *   npx tsx lab/public/waterfall/scripts/strip-double-sided.ts --check <file.glb>   (report only)
 *
 * Idempotent: a file with no `doubleSided` material is left byte-for-byte alone.
 */

import { readFileSync, writeFileSync } from "node:fs";

const GLB_MAGIC = 0x46546c67; // 'glTF'
const CHUNK_JSON = 0x4e4f534a; // 'JSON'

interface GlbLayout {
    /** Byte offset of the JSON chunk's payload. */
    _offset: number;
    /** Declared byte length of the JSON chunk's payload (including its space padding). */
    _length: number;
}

/** Locate the JSON chunk payload inside a GLB container. */
function findJsonChunk(buf: Buffer, path: string): GlbLayout {
    if (buf.length < 12 || buf.readUInt32LE(0) !== GLB_MAGIC) {
        throw new Error(`${path}: not a GLB (bad magic)`);
    }
    if (buf.readUInt32LE(8) !== buf.length) {
        throw new Error(`${path}: header length ${buf.readUInt32LE(8)} != file size ${buf.length}`);
    }
    // The glTF spec requires the JSON chunk to be first, but walk the chunk list anyway rather than
    // assume it — a malformed asset should fail loudly here, not corrupt the BIN chunk.
    let at = 12;
    while (at + 8 <= buf.length) {
        const length = buf.readUInt32LE(at);
        const type = buf.readUInt32LE(at + 4);
        if (type === CHUNK_JSON) {
            return { _offset: at + 8, _length: length };
        }
        at += 8 + length;
    }
    throw new Error(`${path}: no JSON chunk`);
}

/** Remove every `"doubleSided": true` property from a minified glTF JSON string.
 *  Textual rather than parse/re-stringify so the rest of the document stays byte-identical. */
function stripDoubleSided(json: string): { _out: string; _removed: number } {
    let removed = 0;
    const count = (s: string): void => {
        removed += s.length > 0 ? 1 : 0;
    };
    // Handle each comma arrangement separately so the surrounding object stays well-formed whether
    // the property is first, last, or the only member.
    let out = json;
    for (const pattern of [/,\s*"doubleSided"\s*:\s*true/g, /"doubleSided"\s*:\s*true\s*,/g, /"doubleSided"\s*:\s*true/g]) {
        out = out.replace(pattern, (m) => {
            count(m);
            return "";
        });
    }
    return { _out: out, _removed: removed };
}

const args = process.argv.slice(2);
const checkOnly = args.includes("--check");
const files = args.filter((a) => a !== "--check");
if (files.length === 0) {
    console.error("usage: strip-double-sided.ts [--check] <file.glb> [...]");
    process.exit(1);
}

let anyChanged = false;
for (const path of files) {
    const buf = readFileSync(path);
    const { _offset: offset, _length: length } = findJsonChunk(buf, path);
    const original = buf.subarray(offset, offset + length).toString("utf8");
    const { _out: stripped, _removed: removed } = stripDoubleSided(original);

    // Validate before writing: the result must still parse, and must declare no double-sided material.
    const parsed = JSON.parse(stripped) as { materials?: { name?: string; doubleSided?: boolean }[] };
    const stillDouble = (parsed.materials ?? []).filter((m) => m.doubleSided).length;
    if (stillDouble > 0) {
        throw new Error(`${path}: ${stillDouble} material(s) still doubleSided after strip`);
    }

    if (removed === 0) {
        console.log(`${path}: already single-sided (${(parsed.materials ?? []).length} material(s)) — unchanged`);
        continue;
    }
    anyChanged = true;
    console.log(`${path}: removed doubleSided from ${removed} material(s) of ${(parsed.materials ?? []).length}`);
    if (checkOnly) {
        continue;
    }

    // Re-pad to the ORIGINAL chunk length with spaces, which the glTF spec designates as the JSON
    // chunk's padding byte. Nothing outside this chunk moves, so the BIN chunk and the file length
    // are preserved exactly.
    const padded = Buffer.from(stripped.padEnd(length, " "), "utf8");
    if (padded.length !== length) {
        throw new Error(`${path}: repadded JSON is ${padded.length} bytes, expected ${length}`);
    }
    const before = buf.subarray(offset + length);
    padded.copy(buf, offset);
    if (!buf.subarray(offset + length).equals(before)) {
        throw new Error(`${path}: data after the JSON chunk changed — refusing to write`);
    }
    writeFileSync(path, buf);
}

if (checkOnly && anyChanged) {
    process.exit(1);
}
