/**
 * bake-rock-heightmap.ts — bake a top-surface HEIGHT MAP out of the waterfall demo's
 * rock-formation glTF so the fluid can collide with the very geometry it is rendered
 * against.
 *
 * The waterfall demo's scene SDF is a HEIGHTFIELD:
 *
 *     sceneSdf(p) = p.y - heightAt(p.x, p.z)        (positive ABOVE the rock = fluid domain)
 *
 * which is exact for a terrain-like model with no playable overhangs and is far cheaper
 * than a full 3D distance grid (1 MB at 512² vs. tens of MB). This script rasterises the
 * model's triangles into an nx × nz grid of MAX world-Y ("what does a ray fired straight
 * down hit first?"), fills interior pin-holes, optionally smooths, and writes a compact
 * self-describing binary the demo uploads verbatim into a GPU storage buffer.
 *
 * Heights are stored in the model's OWN local space (metres as authored). The demo applies
 * its uniform world scale + translation at UBO-write time, so re-framing the rock in the
 * scene never needs a re-bake.
 *
 * Binary layout (little-endian) — see `HEIGHTMAP_MAGIC`:
 *
 *     off  type   field
 *       0  u32    magic   'BLHM'
 *       4  u32    version 1
 *       8  u32    nx      grid samples along X
 *      12  u32    nz      grid samples along Z
 *      16  f32    originX model-space X of sample (0, 0)
 *      20  f32    originZ model-space Z of sample (0, 0)
 *      24  f32    cellX   model-space X step between samples
 *      28  f32    cellZ   model-space Z step between samples
 *      32  f32    minH    lowest stored height
 *      36  f32    maxH    highest stored height
 *      40  f32    baseH   height written where the model has no coverage (the floor)
 *      44  u32    0       reserved
 *      48  f32[nx*nz]     heights, X-fastest: idx = i + nx * j
 *
 * Usage:
 *   pnpm tsx lab/public/waterfall/scripts/bake-rock-heightmap.ts --src <model.glb> [options]
 *
 * Options:
 *   --src <path>      source .glb (required)
 *   --out <path>      output .bin   (default: rock-heightmap.bin beside this script's folder)
 *   --size <n>        grid resolution per axis (default 512)
 *   --pad <f>         padding around the model footprint, as a fraction of its largest
 *                     horizontal extent (default 0.08) — the border ring is pure floor so
 *                     the demo's clamp-to-edge sampling reads flat ground outside the rock
 *   --smooth <n>      box-blur passes over the rock surface (default 1) — knocks single-cell
 *                     rasterisation spikes off the surface the particles slide on
 *   --preview <path>  also write a greyscale PNG of the baked field
 *
 * No third-party deps beyond pngjs (already a devDependency, preview only): the glTF is
 * parsed directly.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/** The waterfall demo's asset folder (this script lives in its `scripts/` subfolder). */
const ASSET_DIR = join(dirname(fileURLToPath(import.meta.url)), "..");

/** 'BLHM' — Babylon Lite Height Map. Little-endian u32. */
const HEIGHTMAP_MAGIC = 0x4d484c42;
const HEIGHTMAP_VERSION = 1;
const HEADER_BYTES = 48;

// ── glTF binary reading ─────────────────────────────────────────────────────

type TypedArrayCtor = Float32Array | Uint32Array | Uint16Array | Uint8Array | Int16Array | Int8Array;

const COMPONENT_CTORS: Record<number, new (buf: ArrayBufferLike, off: number, len: number) => TypedArrayCtor> = {
    5120: Int8Array,
    5121: Uint8Array,
    5122: Int16Array,
    5123: Uint16Array,
    5125: Uint32Array,
    5126: Float32Array,
};
const COMPONENT_SIZES: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COUNTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

interface Gltf {
    scene?: number;
    scenes: { nodes: number[] }[];
    nodes: { mesh?: number; children?: number[]; translation?: number[]; rotation?: number[]; scale?: number[]; matrix?: number[]; name?: string }[];
    meshes: { primitives: { attributes: Record<string, number>; indices?: number; mode?: number }[] }[];
    accessors: { bufferView: number; byteOffset?: number; componentType: number; count: number; type: string; min?: number[]; max?: number[] }[];
    bufferViews: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
}

/** A 4×4 column-major (glTF convention) matrix as a flat 16-array. */
type Mat4 = number[];

const IDENTITY: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function multiply(a: Mat4, b: Mat4): Mat4 {
    const out = new Array<number>(16).fill(0);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            let s = 0;
            for (let k = 0; k < 4; k++) {
                s += a[k * 4 + r]! * b[c * 4 + k]!;
            }
            out[c * 4 + r] = s;
        }
    }
    return out;
}

function trsMatrix(t: number[] | undefined, q: number[] | undefined, s: number[] | undefined): Mat4 {
    const [x, y, z, w] = q ?? [0, 0, 0, 1];
    const [sx, sy, sz] = s ?? [1, 1, 1];
    const x2 = x! + x!,
        y2 = y! + y!,
        z2 = z! + z!;
    const xx = x! * x2,
        xy = x! * y2,
        xz = x! * z2;
    const yy = y! * y2,
        yz = y! * z2,
        zz = z! * z2;
    const wx = w! * x2,
        wy = w! * y2,
        wz = w! * z2;
    return [
        (1 - (yy + zz)) * sx!,
        (xy + wz) * sx!,
        (xz - wy) * sx!,
        0,
        (xy - wz) * sy!,
        (1 - (xx + zz)) * sy!,
        (yz + wx) * sy!,
        0,
        (xz + wy) * sz!,
        (yz - wx) * sz!,
        (1 - (xx + yy)) * sz!,
        0,
        t?.[0] ?? 0,
        t?.[1] ?? 0,
        t?.[2] ?? 0,
        1,
    ];
}

interface Triangles {
    /** World-space (model-root-space) vertex positions, xyz-interleaved. */
    positions: Float32Array;
    /** Triangle indices into `positions`. */
    indices: Uint32Array;
}

function parseGlb(path: string): { json: Gltf; bin: Buffer } {
    const buf = readFileSync(path);
    if (buf.readUInt32LE(0) !== 0x46546c67) {
        throw new Error(`${path} is not a binary glTF (missing 'glTF' magic)`);
    }
    const jsonLen = buf.readUInt32LE(12);
    const json = JSON.parse(buf.subarray(20, 20 + jsonLen).toString("utf8")) as Gltf;
    let off = 20 + jsonLen;
    let bin = Buffer.alloc(0);
    while (off + 8 <= buf.length) {
        const len = buf.readUInt32LE(off);
        const type = buf.readUInt32LE(off + 4);
        if (type === 0x004e4942) {
            bin = buf.subarray(off + 8, off + 8 + len);
            break;
        }
        off += 8 + len;
    }
    return { json, bin };
}

/** Read an accessor, de-striding interleaved buffer views (see GUIDANCE §1c). */
function readAccessor(json: Gltf, bin: Buffer, index: number): TypedArrayCtor {
    const acc = json.accessors[index]!;
    const view = json.bufferViews[acc.bufferView]!;
    const Ctor = COMPONENT_CTORS[acc.componentType];
    if (!Ctor) {
        throw new Error(`unsupported componentType ${acc.componentType}`);
    }
    const nc = TYPE_COUNTS[acc.type]!;
    const elemBytes = COMPONENT_SIZES[acc.componentType]! * nc;
    const base = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const stride = view.byteStride ?? 0;
    if (stride && stride !== elemBytes) {
        const out = new Ctor(new ArrayBuffer(acc.count * nc * COMPONENT_SIZES[acc.componentType]!), 0, acc.count * nc);
        for (let e = 0; e < acc.count; e++) {
            const src = new Ctor(bin.buffer, bin.byteOffset + base + e * stride, nc);
            (out as Float32Array).set(src as Float32Array, e * nc);
        }
        return out;
    }
    return new Ctor(bin.buffer, bin.byteOffset + base, acc.count * nc);
}

/** Flatten the glTF scene graph into one world-space triangle soup. */
function collectTriangles(json: Gltf, bin: Buffer): Triangles {
    const posChunks: Float32Array[] = [];
    const idxChunks: Uint32Array[] = [];
    let vertexBase = 0;

    const walk = (nodeIndex: number, parent: Mat4): void => {
        const node = json.nodes[nodeIndex]!;
        const local = node.matrix ? (node.matrix as Mat4) : trsMatrix(node.translation, node.rotation, node.scale);
        const world = multiply(parent, local);
        if (node.mesh !== undefined) {
            for (const prim of json.meshes[node.mesh]!.primitives) {
                if ((prim.mode ?? 4) !== 4) {
                    continue; // triangles only
                }
                const src = readAccessor(json, bin, prim.attributes.POSITION!) as Float32Array;
                const count = src.length / 3;
                const dst = new Float32Array(src.length);
                for (let v = 0; v < count; v++) {
                    const x = src[v * 3]!,
                        y = src[v * 3 + 1]!,
                        z = src[v * 3 + 2]!;
                    dst[v * 3] = world[0]! * x + world[4]! * y + world[8]! * z + world[12]!;
                    dst[v * 3 + 1] = world[1]! * x + world[5]! * y + world[9]! * z + world[13]!;
                    dst[v * 3 + 2] = world[2]! * x + world[6]! * y + world[10]! * z + world[14]!;
                }
                let idx: Uint32Array;
                if (prim.indices !== undefined) {
                    const raw = readAccessor(json, bin, prim.indices);
                    idx = new Uint32Array(raw.length);
                    for (let i = 0; i < raw.length; i++) {
                        idx[i] = (raw as Uint32Array)[i]! + vertexBase;
                    }
                } else {
                    idx = new Uint32Array(count);
                    for (let i = 0; i < count; i++) {
                        idx[i] = i + vertexBase;
                    }
                }
                posChunks.push(dst);
                idxChunks.push(idx);
                vertexBase += count;
            }
        }
        for (const child of node.children ?? []) {
            walk(child, world);
        }
    };

    for (const root of json.scenes[json.scene ?? 0]!.nodes) {
        walk(root, IDENTITY);
    }

    const positions = new Float32Array(vertexBase * 3);
    let po = 0;
    for (const c of posChunks) {
        positions.set(c, po);
        po += c.length;
    }
    let ic = 0;
    for (const c of idxChunks) {
        ic += c.length;
    }
    const indices = new Uint32Array(ic);
    let io = 0;
    for (const c of idxChunks) {
        indices.set(c, io);
        io += c.length;
    }
    return { positions, indices };
}

// ── Height-map rasterisation ────────────────────────────────────────────────

interface BakeResult {
    heights: Float32Array;
    covered: Uint8Array;
    nx: number;
    nz: number;
    originX: number;
    originZ: number;
    cellX: number;
    cellZ: number;
    baseH: number;
    minH: number;
    maxH: number;
}

function bakeHeightmap(tris: Triangles, size: number, padFraction: number): BakeResult {
    const { positions, indices } = tris;
    let minX = Infinity,
        minY = Infinity,
        minZ = Infinity,
        maxX = -Infinity,
        maxY = -Infinity,
        maxZ = -Infinity;
    for (let v = 0; v < positions.length; v += 3) {
        const x = positions[v]!,
            y = positions[v + 1]!,
            z = positions[v + 2]!;
        if (x < minX) {
            minX = x;
        }
        if (y < minY) {
            minY = y;
        }
        if (z < minZ) {
            minZ = z;
        }
        if (x > maxX) {
            maxX = x;
        }
        if (y > maxY) {
            maxY = y;
        }
        if (z > maxZ) {
            maxZ = z;
        }
    }
    const pad = padFraction * Math.max(maxX - minX, maxZ - minZ);
    const originX = minX - pad;
    const originZ = minZ - pad;
    const nx = size;
    const nz = size;
    const cellX = (maxX - minX + 2 * pad) / (nx - 1);
    const cellZ = (maxZ - minZ + 2 * pad) / (nz - 1);
    const baseH = minY;

    const heights = new Float32Array(nx * nz).fill(-Infinity);
    const invCellX = 1 / cellX;
    const invCellZ = 1 / cellZ;

    for (let t = 0; t < indices.length; t += 3) {
        const a = indices[t]! * 3,
            b = indices[t + 1]! * 3,
            c = indices[t + 2]! * 3;
        const ay = positions[a + 1]!,
            by = positions[b + 1]!,
            cy = positions[c + 1]!;
        const ax = (positions[a]! - originX) * invCellX,
            az = (positions[a + 2]! - originZ) * invCellZ;
        const bx = (positions[b]! - originX) * invCellX,
            bz = (positions[b + 2]! - originZ) * invCellZ;
        const cx = (positions[c]! - originX) * invCellX,
            cz = (positions[c + 2]! - originZ) * invCellZ;
        const det = (bx - ax) * (cz - az) - (cx - ax) * (bz - az);
        if (det === 0 || !Number.isFinite(det)) {
            continue;
        }
        const invDet = 1 / det;
        const i0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
        const i1 = Math.min(nx - 1, Math.ceil(Math.max(ax, bx, cx)));
        const j0 = Math.max(0, Math.floor(Math.min(az, bz, cz)));
        const j1 = Math.min(nz - 1, Math.ceil(Math.max(az, bz, cz)));
        for (let j = j0; j <= j1; j++) {
            const rowOff = j * nx;
            for (let i = i0; i <= i1; i++) {
                const w1 = ((i - ax) * (cz - az) - (cx - ax) * (j - az)) * invDet;
                const w2 = ((bx - ax) * (j - az) - (i - ax) * (bz - az)) * invDet;
                const w0 = 1 - w1 - w2;
                if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) {
                    continue;
                }
                const y = w0 * ay + w1 * by + w2 * cy;
                if (y > heights[rowOff + i]!) {
                    heights[rowOff + i] = y;
                }
            }
        }
    }

    const covered = new Uint8Array(nx * nz);
    for (let k = 0; k < heights.length; k++) {
        covered[k] = heights[k]! > -Infinity ? 1 : 0;
    }

    // Interior pin-holes: an uncovered sample the model encloses (not reachable from the
    // border) is a rasterisation miss, not real ground. Flood the true exterior from the
    // border, then fill everything else from its covered neighbours.
    const exterior = new Uint8Array(nx * nz);
    const stack: number[] = [];
    const pushIfOpen = (k: number): void => {
        if (!covered[k] && !exterior[k]) {
            exterior[k] = 1;
            stack.push(k);
        }
    };
    for (let i = 0; i < nx; i++) {
        pushIfOpen(i);
        pushIfOpen((nz - 1) * nx + i);
    }
    for (let j = 0; j < nz; j++) {
        pushIfOpen(j * nx);
        pushIfOpen(j * nx + nx - 1);
    }
    while (stack.length) {
        const k = stack.pop()!;
        const i = k % nx;
        const j = (k / nx) | 0;
        if (i > 0) {
            pushIfOpen(k - 1);
        }
        if (i < nx - 1) {
            pushIfOpen(k + 1);
        }
        if (j > 0) {
            pushIfOpen(k - nx);
        }
        if (j < nz - 1) {
            pushIfOpen(k + nx);
        }
    }

    let holes = 0;
    for (let pass = 0; pass < 8; pass++) {
        let filled = 0;
        for (let j = 0; j < nz; j++) {
            for (let i = 0; i < nx; i++) {
                const k = j * nx + i;
                if (covered[k] || exterior[k]) {
                    continue;
                }
                let sum = 0;
                let n = 0;
                const add = (kk: number): void => {
                    if (covered[kk]) {
                        sum += heights[kk]!;
                        n++;
                    }
                };
                if (i > 0) {
                    add(k - 1);
                }
                if (i < nx - 1) {
                    add(k + 1);
                }
                if (j > 0) {
                    add(k - nx);
                }
                if (j < nz - 1) {
                    add(k + nx);
                }
                if (n > 0) {
                    heights[k] = sum / n;
                    covered[k] = 1;
                    filled++;
                }
            }
        }
        holes += filled;
        if (filled === 0) {
            break;
        }
    }
    if (holes) {
        console.log(`  filled ${holes} interior pin-hole sample(s)`);
    }

    for (let k = 0; k < heights.length; k++) {
        if (!covered[k]) {
            heights[k] = baseH;
        }
    }

    let minH = Infinity,
        maxH = -Infinity;
    for (const h of heights) {
        if (h < minH) {
            minH = h;
        }
        if (h > maxH) {
            maxH = h;
        }
    }
    return { heights, covered, nx, nz, originX, originZ, cellX, cellZ, baseH, minH, maxH };
}

/** Box-blur the rock surface (covered samples only) so single-cell rasterisation spikes
 *  don't launch particles. The silhouette edge is preserved: floor samples are excluded
 *  from the average, so the vertical rock/ground step stays crisp. */
function smooth(res: BakeResult, passes: number): void {
    const { heights, covered, nx, nz } = res;
    if (passes <= 0) {
        return;
    }
    const tmp = new Float32Array(heights.length);
    for (let p = 0; p < passes; p++) {
        tmp.set(heights);
        for (let j = 0; j < nz; j++) {
            for (let i = 0; i < nx; i++) {
                const k = j * nx + i;
                if (!covered[k]) {
                    continue;
                }
                let sum = 0;
                let n = 0;
                for (let dj = -1; dj <= 1; dj++) {
                    const jj = j + dj;
                    if (jj < 0 || jj >= nz) {
                        continue;
                    }
                    for (let di = -1; di <= 1; di++) {
                        const ii = i + di;
                        if (ii < 0 || ii >= nx) {
                            continue;
                        }
                        const kk = jj * nx + ii;
                        if (!covered[kk]) {
                            continue;
                        }
                        sum += tmp[kk]!;
                        n++;
                    }
                }
                heights[k] = sum / n;
            }
        }
    }
    let minH = Infinity,
        maxH = -Infinity;
    for (const h of heights) {
        if (h < minH) {
            minH = h;
        }
        if (h > maxH) {
            maxH = h;
        }
    }
    res.minH = minH;
    res.maxH = maxH;
}

function serialize(res: BakeResult): Buffer {
    const out = Buffer.alloc(HEADER_BYTES + res.heights.length * 4);
    out.writeUInt32LE(HEIGHTMAP_MAGIC, 0);
    out.writeUInt32LE(HEIGHTMAP_VERSION, 4);
    out.writeUInt32LE(res.nx, 8);
    out.writeUInt32LE(res.nz, 12);
    out.writeFloatLE(res.originX, 16);
    out.writeFloatLE(res.originZ, 20);
    out.writeFloatLE(res.cellX, 24);
    out.writeFloatLE(res.cellZ, 28);
    out.writeFloatLE(res.minH, 32);
    out.writeFloatLE(res.maxH, 36);
    out.writeFloatLE(res.baseH, 40);
    out.writeUInt32LE(0, 44);
    Buffer.from(res.heights.buffer, res.heights.byteOffset, res.heights.byteLength).copy(out, HEADER_BYTES);
    return out;
}

async function writePreview(res: BakeResult, path: string): Promise<void> {
    const { PNG } = await import("pngjs");
    const png = new PNG({ width: res.nx, height: res.nz });
    const span = Math.max(1e-6, res.maxH - res.minH);
    for (let j = 0; j < res.nz; j++) {
        for (let i = 0; i < res.nx; i++) {
            const k = j * res.nx + i;
            // Flip Z so the preview reads like a top-down map with +Z up.
            const p = ((res.nz - 1 - j) * res.nx + i) * 4;
            const v = Math.round(((res.heights[k]! - res.minH) / span) * 255);
            png.data[p] = v;
            png.data[p + 1] = res.covered[k] ? v : Math.round(v * 0.35);
            png.data[p + 2] = res.covered[k] ? v : Math.round(v * 0.35);
            png.data[p + 3] = 255;
        }
    }
    writeFileSync(path, PNG.sync.write(png));
    console.log(`Wrote preview → ${path}`);
}

function arg(name: string, fallback?: string): string | undefined {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

export async function bakeWaterfallRock(): Promise<void> {
    const src = arg("src");
    if (!src) {
        throw new Error("--src <model.glb> is required");
    }
    const out = arg("out", join(ASSET_DIR, "rock-heightmap.bin"))!;
    const size = Number(arg("size", "512"));
    const padFraction = Number(arg("pad", "0.08"));
    const smoothPasses = Number(arg("smooth", "1"));
    const preview = arg("preview");

    console.log(`Reading ${src} …`);
    const { json, bin } = parseGlb(src);
    const tris = collectTriangles(json, bin);
    console.log(`  ${tris.positions.length / 3} vertices, ${tris.indices.length / 3} triangles`);

    console.log(`Rasterising ${size}×${size} height map (pad ${padFraction}) …`);
    const res = bakeHeightmap(tris, size, padFraction);
    smooth(res, smoothPasses);

    // Report the summit + footprint so the demo can be framed against real numbers.
    let peak = 0;
    for (let k = 1; k < res.heights.length; k++) {
        if (res.heights[k]! > res.heights[peak]!) {
            peak = k;
        }
    }
    const px = res.originX + (peak % res.nx) * res.cellX;
    const pz = res.originZ + Math.floor(peak / res.nx) * res.cellZ;
    let coveredCount = 0;
    for (const c of res.covered) {
        coveredCount += c;
    }

    mkdirSync(dirname(out), { recursive: true });
    const buf = serialize(res);
    writeFileSync(out, buf);

    console.log(
        `  footprint  X [${res.originX.toFixed(4)} … ${(res.originX + (res.nx - 1) * res.cellX).toFixed(4)}]  Z [${res.originZ.toFixed(4)} … ${(res.originZ + (res.nz - 1) * res.cellZ).toFixed(4)}]`
    );
    console.log(`  height     ${res.minH.toFixed(4)} … ${res.maxH.toFixed(4)}  (floor ${res.baseH.toFixed(4)})`);
    console.log(`  summit     (${px.toFixed(4)}, ${res.maxH.toFixed(4)}, ${pz.toFixed(4)})`);
    console.log(`  coverage   ${((coveredCount / res.heights.length) * 100).toFixed(1)}% of the grid`);
    console.log(`Wrote ${(buf.length / 1024).toFixed(0)} KB → ${out}`);

    if (preview) {
        await writePreview(res, preview);
    }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    bakeWaterfallRock().catch((err: unknown) => {
        console.error(err);
        process.exit(1);
    });
}
