/**
 * sync-baked-ship.ts — publish the Aquanova editor's baked export into `lab/public/aquanova`,
 * compressing the lightmap atlases and baked ship to runtime-ready KTX2/Meshopt assets.
 *
 * The editor (`lab/lite/src/demos/aquanova/editor`) authors the ship and drives a Blender
 * lightmap bake; both write into `editor/export`, which is a working directory full of
 * timestamped autosaves and 40 MB intermediates. The demo under `lab/public/aquanova` wants
 * exactly four things out of it, in runtime-ready form:
 *
 *     ship_baked.glb        the ship with the bake's TEXCOORD_1 atlas UVs,
 *                           KTX2 material textures, and Meshopt geometry
 *     ship_manifest.json    entity behaviours, environment settings, authored lights
 *     ship_collision.json   the collision hulls
 *     lightmaps/*.ktx2      one irradiance atlas per chunk
 *
 * ── Why KTX2 rather than the PNG the bake writes ──────────────────────────────────────
 * A PNG is decoded to RGBA8 and stays that way in VRAM: the 1024² corridor atlas costs
 * 4 MB resident (5.3 MB with mips) and every new chunk adds its own. KTX2/Basis stays
 * compressed all the way onto the GPU — UASTC transcodes to BC7/ASTC at 1 byte/texel, so
 * the same atlas is 1.4 MB resident, and the transcode replaces a PNG decode.
 *
 * UASTC (not ETC1S) is the default because of what a lightmap *is*. These atlases are
 * smooth, low-frequency irradiance ramps across walls and floors, and smooth ramps are the
 * worst case for ETC1S: it transcodes to BC1, whose 5:6:5 endpoints and 2-bit indices band
 * visibly exactly where the signal is a gentle gradient. UASTC's BC7 output is effectively
 * lossless here. That already-quantised signal cannot afford a second lossy pass either —
 * `write_png` in `bake_lightmaps.py` divides the atlas by its own 99.9th percentile to fit
 * 8 bits (the `level` the runtime multiplies back), so a lit texel commonly lands around a
 * tenth of the 0-255 range before the encoder ever sees it. `--etc1s` is available when
 * download size beats fidelity; on the corridor atlas it is ~5x smaller than UASTC.
 *
 * Encoding is delegated to glTF Transform's `uastc`/`etc1s` commands, which use Khronos
 * KTX-Software and do not impose the 12 Mpix limit of the old Node Basis encoder. Ship
 * normal maps use UASTC with Zstandard supercompression; base-colour, metallic-roughness,
 * occlusion, and emissive maps use ETC1S. The standalone Meshopt transform is used for
 * geometry so the node and material hierarchy required by the manifest is preserved.
 * Install KTX-Software with its `toktx` executable on PATH before publishing. If it is
 * installed elsewhere, set `TOKTX_PATH` to the executable or `KTX_SOFTWARE_PATH` to its
 * `bin` folder.
 *
 * ── sRGB, and why the demo is a little brighter than the editor ───────────────────────
 * Blender writes the PNG through the `Standard` view transform, i.e. sRGB-encoded, which
 * `lightmaps.json` reports as `gamma: true`. The KTX2 carries an sRGB transfer function in
 * its DFD and the runtime asks for the `*-srgb` GPU format, so the decode is the sampler's
 * EXACT sRGB curve — which is why the emitted index sets `gamma: false`: the texel arriving
 * in the shader is already linear.
 *
 * The editor's Babylon.js preview instead sets `Texture.gammaSpace = true`, and BJS's
 * `toLinearSpace` is `pow(x, 2.2)`. Those two curves are NOT the same down where a lightmap
 * lives: sRGB has a linear toe below 0.04045 and stays above the 2.2 power law all the way
 * up, so at a typical lit texel (~14/255) it returns roughly 2.5x more light. The rendered
 * difference is far smaller after `level`, exposure and tone mapping — measured at 47% mean
 * luminance on the corridor view — but it is visible, and it is the ONLY difference between
 * the two paths: forcing the runtime onto `pow(2.2)` (`--gamma22`) matches the editor's BJS
 * render to a 0.23% mean absolute pixel difference, which also puts a bound on what the
 * UASTC compression costs.
 *
 * The default is the exact sRGB curve because it is the true inverse of the transform
 * Blender wrote the file with, so it reconstructs the irradiance Cycles computed. Where the
 * two disagree it is the editor preview that is wrong (too dark in the shadows), not this.
 * `--gamma22` exists to reproduce the editor byte-for-byte when comparing the two.
 *
 * ── Orientation ───────────────────────────────────────────────────────────────────────
 * KTX2 mips upload verbatim (no V-flip anywhere in the path), so row 0 of the PNG samples
 * at v=0 — the same place Babylon.js's `new Texture(url, scene, false, false)` puts it in
 * the editor's baked preview. Nothing is flipped here; see GUIDANCE §8.
 *
 * Usage:
 *   pnpm tsx lab/public/aquanova/scripts/sync-baked-ship.ts [options]
 *
 * Options:
 *   --etc1s              encode ETC1S/BC1 instead of UASTC/BC7 (much smaller, banding risk)
 *   --quality <n>        UASTC quality level 0-3 (default 2), or ETC1S 1-255 (default 200)
 *   --gamma22            decode in the shader with pow(2.2) instead of the sampler's sRGB,
 *                        reproducing the editor's Babylon.js preview exactly
 *   --no-mips            do not generate a mip chain
 *   --force              re-encode atlases whose source PNG has not changed
 *   --no-ship-optimize   copy the baked GLB without Meshopt/KTX2 processing
 *   --ship-uastc-level <n>
 *                        UASTC level for ship normal maps (default 2)
 *   --ship-etc1s-quality <n>
 *                        ETC1S quality for ship material maps (default 200)
 *   --ship-zstd <n>      UASTC Zstandard level for ship textures (default 18)
 *   --src <dir>          editor export directory (default ../../../lite/src/demos/aquanova/editor/export)
 *   --out <dir>          demo asset directory (default the aquanova directory above this one)
 */

import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, copyFile, readFile, rm, writeFile, stat, access } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SRC = path.resolve(HERE, "../../../lite/src/demos/aquanova/editor/export");
const DEFAULT_OUT = path.resolve(HERE, "..");
const REPO_ROOT = path.resolve(HERE, "../../../..");
const execFileAsync = promisify(execFile);

interface BakeChunk {
    png: string;
    hdr?: string;
    level: number;
    lit: number;
    resolution: number;
    meshes: number;
    hash: string;
}

interface BakeIndex {
    glb: string;
    uv: number;
    gamma: boolean;
    chunks: Record<string, BakeChunk>;
}

/** What the demo loads: one KTX2 per chunk plus the scale the bake divided out. */
interface RuntimeChunk {
    url: string;
    level: number;
    resolution: number;
    meshes: number;
    bytes: number;
}

interface RuntimeIndex {
    glb: string;
    /** TEXCOORD set the atlas UVs live in. */
    uv: number;
    /** Whether the shader must sRGB-decode the sample. False: the GPU format does it. */
    gamma: boolean;
    /** Whether to request the `*-srgb` GPU format when uploading. */
    srgb: boolean;
    encoding: string;
    chunks: Record<string, RuntimeChunk>;
}

interface Options {
    src: string;
    out: string;
    etc1s: boolean;
    quality: number | undefined;
    mips: boolean;
    force: boolean;
    /** Decode with the shader's `pow(2.2)` instead of the sampler's exact sRGB — see the header. */
    gamma22: boolean;
    shipOptimize: boolean;
    shipUastcLevel: number;
    shipEtc1sQuality: number;
    shipZstd: number;
}

function parseArgs(argv: readonly string[]): Options {
    const opts: Options = {
        src: DEFAULT_SRC,
        out: DEFAULT_OUT,
        etc1s: false,
        quality: undefined,
        mips: true,
        force: false,
        gamma22: false,
        shipOptimize: true,
        shipUastcLevel: 2,
        shipEtc1sQuality: 200,
        shipZstd: 18,
    };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i]!;
        if (a === "--etc1s") {
            opts.etc1s = true;
        } else if (a === "--gamma22") {
            opts.gamma22 = true;
        } else if (a === "--no-mips") {
            opts.mips = false;
        } else if (a === "--force") {
            opts.force = true;
        } else if (a === "--no-ship-optimize") {
            opts.shipOptimize = false;
        } else if (a === "--ship-uastc-level") {
            opts.shipUastcLevel = Number(argv[++i]);
        } else if (a === "--ship-etc1s-quality") {
            opts.shipEtc1sQuality = Number(argv[++i]);
        } else if (a === "--ship-zstd") {
            opts.shipZstd = Number(argv[++i]);
        } else if (a === "--quality") {
            opts.quality = Number(argv[++i]);
        } else if (a === "--src") {
            opts.src = path.resolve(argv[++i]!);
        } else if (a === "--out") {
            opts.out = path.resolve(argv[++i]!);
        } else {
            throw Error(`unknown option ${a}`);
        }
    }
    return opts;
}

function assertIntegerRange(name: string, value: number, min: number, max: number): void {
    if (!Number.isInteger(value) || value < min || value > max) {
        throw Error(`${name} must be an integer between ${min} and ${max}`);
    }
}

function mb(bytes: number): string {
    return `${(bytes / 1048576).toFixed(2)} MB`;
}

/** Copy only when the destination is missing or differs in size/mtime — the ship glb is
 *  40 MB and this script is expected to be re-run after every bake. */
async function copyIfChanged(src: string, dst: string): Promise<boolean> {
    const s = await stat(src);
    const d = await stat(dst).catch(() => null);
    if (d && d.size === s.size && d.mtimeMs >= s.mtimeMs) {
        return false;
    }
    await copyFile(src, dst);
    return true;
}

interface SourceTexture {
    name: string;
    bytes: Uint8Array;
}

interface GltfImage {
    name?: string;
    uri?: string;
}

interface GltfTexture {
    source?: number;
    extensions?: {
        KHR_texture_basisu?: {
            source?: number;
        };
    };
}

interface GltfDocument {
    images?: GltfImage[];
    textures?: GltfTexture[];
}

async function existingFile(file: string): Promise<string | undefined> {
    try {
        await access(file);
        return file;
    } catch {
        return undefined;
    }
}

async function findToktx(): Promise<string | undefined> {
    const explicit = process.env.TOKTX_PATH;
    if (explicit) {
        return existingFile(path.resolve(explicit));
    }

    const binPath = process.env.KTX_SOFTWARE_PATH;
    const roots = [
        ...(binPath ? [binPath] : []),
        process.env.ProgramFiles ? path.join(process.env.ProgramFiles, "KTX-Software", "bin") : undefined,
        process.env["ProgramFiles(x86)"] ? path.join(process.env["ProgramFiles(x86)"]!, "KTX-Software", "bin") : undefined,
        process.env.LOCALAPPDATA ? path.join(process.env.LOCALAPPDATA, "KTX-Software", "bin") : undefined,
        process.env.LOCALAPPDATA ? path.join(process.env.LOCALAPPDATA, "Programs", "KTX-Software", "bin") : undefined,
    ].filter((value): value is string => Boolean(value));
    const executable = process.platform === "win32" ? "toktx.exe" : "toktx";

    for (const root of roots) {
        const found = await existingFile(path.join(root, executable));
        if (found) {
            return found;
        }
    }

    return undefined;
}

async function encoderEnvironment(): Promise<NodeJS.ProcessEnv> {
    const toktx = await findToktx();
    if (!toktx) {
        return process.env;
    }
    const bin = path.dirname(toktx);
    const separator = process.platform === "win32" ? ";" : ":";
    return { ...process.env, PATH: `${bin}${separator}${process.env.PATH || ""}` };
}

async function runGltfTransform(args: readonly string[], label: string): Promise<void> {
    const command = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
    try {
        const env = await encoderEnvironment();
        await execFileAsync(command, ["dlx", "--yes", "@gltf-transform/cli@4.4.2", ...args], {
            cwd: REPO_ROOT,
            maxBuffer: 16 * 1024 * 1024,
            shell: process.platform === "win32",
            env,
        });
    } catch (error) {
        const detail =
            error instanceof Error ? `${error.message}${typeof error === "object" && error !== null && "stderr" in error ? `\n${String(error.stderr)}` : ""}` : String(error);
        throw Error(`glTF Transform ${label} failed. Ensure KTX-Software's "toktx" is available, then retry. ${detail}`);
    }
}

function textureDocument(
    sources: readonly SourceTexture[],
    srgb: boolean
): {
    json: GltfDocument & Record<string, unknown>;
    buffer: Uint8Array;
} {
    const positions = new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]);
    const json: GltfDocument & Record<string, unknown> = {
        asset: { version: "2.0" },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0 }],
        meshes: [
            {
                primitives: sources.map((_source, index) => ({
                    attributes: { POSITION: 0 },
                    material: index,
                })),
            },
        ],
        accessors: [
            {
                bufferView: 0,
                componentType: 5126,
                count: 3,
                type: "VEC3",
                min: [-1, -1, 0],
                max: [1, 1, 0],
            },
        ],
        bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
        buffers: [{ uri: "mesh.bin", byteLength: positions.byteLength }],
        materials: sources.map((_source, index) => ({
            pbrMetallicRoughness: srgb ? { baseColorTexture: { index } } : { metallicRoughnessTexture: { index } },
        })),
        textures: sources.map((_source, index) => ({ source: index })),
        images: sources.map((source) => ({
            name: source.name,
            uri: `${source.name}.png`,
        })),
    };
    return { json, buffer: new Uint8Array(positions.buffer) };
}

/** Encode every atlas in one glTF Transform/KTX-Software invocation. */
async function encodeWithGltfTransform(sources: readonly SourceTexture[], opts: Pick<Options, "etc1s" | "quality" | "mips">, srgb: boolean): Promise<Map<string, Uint8Array>> {
    const temp = await mkdtemp(path.join(process.env.TEMP || process.cwd(), "aquanova-ktx2-"));
    const input = path.join(temp, "input.gltf");
    const output = path.join(temp, "output.gltf");
    try {
        const document = textureDocument(sources, srgb);
        await writeFile(path.join(temp, "mesh.bin"), document.buffer);
        for (const source of sources) {
            await writeFile(path.join(temp, `${source.name}.png`), source.bytes);
        }
        await writeFile(input, `${JSON.stringify(document.json, null, 2)}\n`);

        const mode = opts.etc1s ? "etc1s" : "uastc";
        const qualityFlag = opts.etc1s ? "--quality" : "--level";
        const quality = opts.quality ?? (opts.etc1s ? 200 : 2);
        const args = [mode, input, output, "--jobs", "1", "--mipmaps", String(opts.mips), qualityFlag, String(quality)];
        if (!opts.etc1s) {
            args.push("--zstd", "18");
        }
        await runGltfTransform(args, "KTX2 conversion");

        const result = JSON.parse(await readFile(output, "utf8")) as GltfDocument;
        const images = result.images || [];
        const textures = result.textures || [];
        const encoded = new Map<string, Uint8Array>();
        for (let index = 0; index < sources.length; index++) {
            const texture = textures[index];
            const sourceIndex = texture?.extensions?.KHR_texture_basisu?.source ?? texture?.source;
            const image = sourceIndex === undefined ? undefined : images[sourceIndex];
            if (!image?.uri || !image.uri.toLowerCase().endsWith(".ktx2")) {
                throw Error(`glTF Transform did not produce a KTX2 image for ${sources[index]!.name}`);
            }
            encoded.set(sources[index]!.name, new Uint8Array(await readFile(path.resolve(temp, image.uri))));
        }
        return encoded;
    } finally {
        await rm(temp, { recursive: true, force: true });
    }
}

async function publishShipGlb(
    source: string,
    destination: string,
    opts: Pick<Options, "shipOptimize" | "shipUastcLevel" | "shipEtc1sQuality" | "shipZstd" | "force">
): Promise<void> {
    if (!opts.shipOptimize) {
        const copied = await copyIfChanged(source, destination);
        console.log(`${copied ? "copied " : "current"}  ${path.basename(destination)}  (unoptimized)`);
        return;
    }

    const sourceBytes = await readFile(source);
    const stamp = createHash("md5")
        .update(sourceBytes)
        .update(JSON.stringify([opts.shipOptimize, opts.shipUastcLevel, opts.shipEtc1sQuality, opts.shipZstd]))
        .digest("hex");
    const stampFile = `${destination}.stamp`;
    const previous = await readFile(stampFile, "utf8").catch(() => "");
    const destinationExists = await stat(destination)
        .then(() => true)
        .catch(() => false);
    if (!opts.force && destinationExists && previous === stamp) {
        console.log(`current  ${path.basename(destination)}  (Meshopt/KTX2)`);
        return;
    }

    const temp = await mkdtemp(path.join(process.env.TEMP || process.cwd(), "aquanova-ship-"));
    const meshopt = path.join(temp, "meshopt.glb");
    const uastc = path.join(temp, "uastc.glb");
    const output = path.join(temp, "output.glb");
    try {
        await runGltfTransform(["meshopt", source, meshopt, "--level", "high"], "Meshopt geometry compression");
        await runGltfTransform(
            ["uastc", meshopt, uastc, "--slots", "normalTexture", "--level", String(opts.shipUastcLevel), "--zstd", String(opts.shipZstd), "--jobs", "1"],
            "ship UASTC texture compression"
        );
        await runGltfTransform(
            [
                "etc1s",
                uastc,
                output,
                "--slots",
                "{baseColorTexture,metallicRoughnessTexture,occlusionTexture,emissiveTexture}",
                "--quality",
                String(opts.shipEtc1sQuality),
                "--compression",
                "4",
                "--jobs",
                "1",
            ],
            "ship ETC1S texture compression"
        );

        await copyFile(output, destination);
        await writeFile(stampFile, stamp);
        const outputBytes = (await stat(destination)).size;
        console.log(`optimized  ${path.basename(destination)}  ${mb(sourceBytes.byteLength)} -> ${mb(outputBytes)}`);
    } finally {
        await rm(temp, { recursive: true, force: true });
    }
}

async function main(): Promise<void> {
    const opts = parseArgs(process.argv.slice(2));
    assertIntegerRange("--ship-uastc-level", opts.shipUastcLevel, 0, 4);
    assertIntegerRange("--ship-etc1s-quality", opts.shipEtc1sQuality, 1, 255);
    assertIntegerRange("--ship-zstd", opts.shipZstd, 0, 22);
    const bakeDir = path.join(opts.src, "lightmaps");
    const index = JSON.parse(await readFile(path.join(bakeDir, "lightmaps.json"), "utf8")) as BakeIndex;
    const outLightmaps = path.join(opts.out, "lightmaps");
    await mkdir(outLightmaps, { recursive: true });

    // The manifest and collision files are authored data; only the baked GLB is transformed.
    const shipSource = path.join(opts.src, index.glb);
    const shipDestination = path.join(opts.out, index.glb);
    await publishShipGlb(shipSource, shipDestination, opts);
    for (const name of ["ship_manifest.json", "ship_collision.json"]) {
        const copied = await copyIfChanged(path.join(opts.src, name), path.join(opts.out, name));
        console.log(`${copied ? "copied " : "current"}  ${name}`);
    }

    // `--gamma22` moves the decode back into the shader: the file must then NOT advertise an
    // sRGB transfer function, or the sampler would decode it a second time.
    const samplerDecodes = index.gamma !== false && !opts.gamma22;
    const runtime: RuntimeIndex = {
        glb: index.glb,
        uv: index.uv,
        gamma: index.gamma !== false && opts.gamma22,
        srgb: samplerDecodes,
        encoding: opts.etc1s ? "etc1s" : "uastc",
        chunks: {},
    };

    const pending: Array<{
        chunkId: string;
        chunk: BakeChunk;
        source: Uint8Array;
        dst: string;
        stamp: string;
    }> = [];
    let totalPng = 0;
    for (const [chunkId, chunk] of Object.entries(index.chunks)) {
        const srcPng = path.join(bakeDir, chunk.png);
        const source = await readFile(srcPng);
        const dstName = `${chunkId}.ktx2`;
        const dst = path.join(outLightmaps, dstName);

        // Hash of the source atlas plus the encoder settings: re-running the script after a
        // bake that only touched one chunk should not re-encode the others.
        const stamp = createHash("md5")
            .update(source)
            .update(JSON.stringify([opts.etc1s, opts.quality, opts.mips, opts.gamma22]))
            .digest("hex");
        const stampFile = `${dst}.stamp`;
        const previous = await readFile(stampFile, "utf8").catch(() => "");

        if (!opts.force && previous === stamp) {
            const bytes = (await stat(dst)).size;
            console.log(`current  ${dstName}  ${mb(bytes)}`);
            runtime.chunks[chunkId] = {
                url: `lightmaps/${dstName}`,
                level: chunk.level,
                resolution: chunk.resolution,
                meshes: chunk.meshes,
                bytes,
            };
        } else {
            pending.push({ chunkId, chunk, source: new Uint8Array(source), dst, stamp });
        }

        totalPng += source.byteLength;
    }

    if (pending.length) {
        const started = Date.now();
        const encoded = await encodeWithGltfTransform(
            pending.map(({ chunkId, source }) => ({ name: chunkId, bytes: source })),
            opts,
            samplerDecodes
        );
        for (const item of pending) {
            const bytes = encoded.get(item.chunkId);
            if (!bytes) {
                throw Error(`No KTX2 output returned for ${item.chunkId}`);
            }
            await writeFile(item.dst, bytes);
            await writeFile(`${item.dst}.stamp`, item.stamp);
            console.log(`encoded  ${item.chunkId}.ktx2  ${item.chunk.resolution}²  ${mb(item.source.byteLength)} png -> ${mb(bytes.byteLength)}`);
            runtime.chunks[item.chunkId] = {
                url: `lightmaps/${item.chunkId}.ktx2`,
                level: item.chunk.level,
                resolution: item.chunk.resolution,
                meshes: item.chunk.meshes,
                bytes: bytes.byteLength,
            };
        }
        console.log(`glTF Transform conversion: ${pending.length} atlas${pending.length === 1 ? "" : "es"} in ${Date.now() - started} ms`);
    }

    await writeFile(path.join(opts.out, "lightmaps.json"), `${JSON.stringify(runtime, null, 1)}\n`);
    const chunkCount = Object.keys(index.chunks).length;
    const totalKtx = Object.values(runtime.chunks).reduce((sum, chunk) => sum + chunk.bytes, 0);
    console.log(`\n${chunkCount} lightmap${chunkCount === 1 ? "" : "s"}: ${mb(totalPng)} png -> ${mb(totalKtx)} ktx2 (${Math.round((100 * totalKtx) / totalPng)}%)`);
}

main().catch((err: unknown) => {
    console.error(err);
    process.exit(1);
});
