/**
 * optimize-rock.ts — shrink the waterfall demo's rock-formation glTF from a raw
 * photogrammetry scan into something shippable, using glTF Transform (https://gltf-transform.dev).
 *
 * The source scan is ~72 MB: 2.45 M triangles split over 16 primitives, plus three 4096²
 * JPEG maps (base colour / normal / metallic-roughness) that cost ~89 MB of VRAM EACH once
 * uploaded uncompressed. This script takes it to ~5.3 MB / 60 k triangles / 11 MB of VRAM,
 * which is small enough to commit and cheap enough to sit in a fluid demo that is already
 * spending its budget on the simulation.
 *
 * Pipeline, and why each step is there:
 *
 *   dedup      → drop duplicate accessors/textures.
 *   flatten    → collapse the node hierarchy so `join` is allowed to merge.
 *   join       → 16 primitives (all sharing ONE material) become 1. Must happen BEFORE
 *                simplification: decimating the chunks separately lets their shared borders
 *                drift apart and opens cracks along every seam.
 *   weld       → merge bitwise-identical vertices, a prerequisite for good simplification.
 *   simplify   → TWO modes, see --error vs --tris below.
 *   resize     → 2048². The rock is set dressing; 4096² costs 4x the pixels for detail that
 *                is never resolved at the demo's framing.
 *   uastc      → the NORMAL map only. Normals are the one map where ETC1S's block artifacts
 *                show up as visible shading banding.
 *   etc1s      → base colour + metallic-roughness, where it is nearly free perceptually and
 *                roughly a third the size of UASTC.
 *
 * Choosing --error vs --tris
 * --------------------------
 * `--error` is the QUALITY-FIRST mode and the one to reach for by default: a SINGLE
 * simplification pass, decimating as far as the error budget allows (meshoptimizer's own
 * `--ratio 0`). The budget is a fraction of mesh radius, so it is a real geometric guarantee.
 *
 * `--tris` hits an exact triangle budget instead, and to do that it may run several passes
 * (see the loop below). Be aware that this COMPOUNDS error: each pass measures its budget
 * against the already-decimated mesh from the previous pass, not against the original, so the
 * total deviation from the source is much larger than any single pass suggests. Use it only
 * when a hard polygon budget matters more than fidelity.
 *
 * Both modes are bounded by the same wall: a mesh whose UV islands split vertices along every
 * seam cannot be decimated past those constraints in one pass. This scan-style content
 * saturates around 8% of its source triangles however much error budget it is handed — going
 * below that is precisely what forces `--tris` into extra passes, and what costs the detail.
 *
 * The mesh changes shape slightly, so the collision height map MUST be re-baked from the
 * OUTPUT of this script — otherwise the fluid rides a surface the renderer no longer draws:
 *
 *   pnpm tsx lab/public/waterfall/scripts/bake-rock-heightmap.ts --src lab/public/waterfall/rock.glb
 *
 * Requirements (neither is a repo dependency — both are external asset tooling):
 *   • glTF Transform CLI — npm install -g @gltf-transform/cli  (or set GLTF_TRANSFORM)
 *   • KTX-Software `ktx`  — https://github.com/KhronosGroup/KTX-Software/releases, on PATH
 *
 * Usage:
 *   pnpm tsx lab/public/waterfall/scripts/optimize-rock.ts --src <scan.glb> [options]
 *
 * Options:
 *   --src <path>      source .glb (required) — the raw scan, NOT the optimised asset
 *   --out <path>      output .glb (default: rock.glb in the demo's asset folder)
 *   --error <f>       error-driven: one pass, decimate within this fraction of mesh radius
 *                     (default 0.001). Mutually exclusive with --tris.
 *   --tris <n>        count-driven: hit this triangle count, multi-pass, compounds error
 *   --texture <px>    max texture dimension (default 2048)
 *   --keep-temp       leave the intermediate .glb files on disk for inspection
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

/** The waterfall demo's asset folder (this script lives in its `scripts/` subfolder). */
const ASSET_DIR = join(dirname(fileURLToPath(import.meta.url)), "..");

// ── args ────────────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const arg = (name: string, fallback?: string): string | undefined => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};
const src = arg("src");
if (!src) {
    console.error("error: --src <scan.glb> is required");
    process.exit(1);
}
const out = arg("out", join(ASSET_DIR, "rock.glb"))!;
const texturePx = Number(arg("texture", "2048"));
const keepTemp = argv.includes("--keep-temp");
// Two mutually exclusive simplification modes. Error-driven is the default because it is a
// single pass with a real geometric guarantee; count-driven exists for a hard polygon budget.
const errorArg = arg("error");
const trisArg = arg("tris");
if (errorArg !== undefined && trisArg !== undefined) {
    console.error("error: --error and --tris are alternative modes; pass only one");
    process.exit(1);
}
const countDriven = trisArg !== undefined;
const targetTris = Number(trisArg ?? "60000");
/** Error budget as a fraction of mesh radius. Also used as the per-pass budget in count mode. */
const SIMPLIFY_ERROR = Number(errorArg ?? (countDriven ? "0.02" : "0.001"));

/** Stop once within this fraction of the target — chasing the last percent costs a whole pass. */
const SIMPLIFY_TOLERANCE = 0.02;
const SIMPLIFY_MAX_PASSES = 4;

// ── tool discovery ──────────────────────────────────────────────────────────
const GLTF_TRANSFORM = process.env.GLTF_TRANSFORM ?? "gltf-transform";
function run(cmd: string, args: string[]): void {
    execFileSync(cmd, args, { stdio: ["ignore", "ignore", "inherit"], shell: process.platform === "win32" });
}
function requireTool(cmd: string, args: string[], hint: string): void {
    try {
        run(cmd, args);
    } catch {
        console.error(`error: '${cmd}' is not runnable. ${hint}`);
        process.exit(1);
    }
}
requireTool(GLTF_TRANSFORM, ["--version"], "Install with: npm install -g @gltf-transform/cli (or set GLTF_TRANSFORM).");
requireTool("ktx", ["--version"], "Install KTX-Software: https://github.com/KhronosGroup/KTX-Software/releases");

// ── triangle counting (parse the GLB's JSON chunk directly, like the baker does) ──
interface GlbJson {
    meshes?: { primitives: { indices?: number; attributes: Record<string, number> }[] }[];
    accessors?: { count: number }[];
}
function readGlbJson(path: string): GlbJson {
    const buf = readFileSync(path);
    if (buf.readUInt32LE(0) !== 0x46546c67) {
        throw new Error(`${path} is not a binary glTF`);
    }
    // Chunk 0 is always JSON: [u32 length][u32 type][payload].
    const len = buf.readUInt32LE(12);
    return JSON.parse(buf.subarray(20, 20 + len).toString("utf8")) as GlbJson;
}
function countTriangles(path: string): number {
    const json = readGlbJson(path);
    let tris = 0;
    for (const mesh of json.meshes ?? []) {
        for (const prim of mesh.primitives) {
            const acc = prim.indices !== undefined ? prim.indices : Object.values(prim.attributes)[0];
            tris += Math.floor((json.accessors?.[acc as number]?.count ?? 0) / 3);
        }
    }
    return tris;
}

// ── pipeline ────────────────────────────────────────────────────────────────
const tmp = mkdtempSync(join(tmpdir(), "rock-opt-"));
const step = (name: string): string => join(tmp, `${name}.glb`);
const gt = (...args: string[]): void => run(GLTF_TRANSFORM, args);

try {
    console.log(`Reading ${src} … ${countTriangles(src).toLocaleString()} triangles`);

    gt("dedup", src, step("1-dedup"));
    gt("flatten", step("1-dedup"), step("2-flatten"));
    gt("join", step("2-flatten"), step("3-join"), "--keepMeshes", "false", "--keepNamed", "false");
    gt("weld", step("3-join"), step("4-weld"));
    console.log(`  joined + welded → ${countTriangles(step("4-weld")).toLocaleString()} triangles`);

    // Simplify toward the target — see the two modes below.
    let current = step("4-weld");
    let currentTris = countTriangles(current);

    if (!countDriven) {
        // ── Error-driven: ONE pass, `--ratio 0` so meshoptimizer decimates as far as the error
        // budget allows and no further. One pass is the whole point — every extra pass re-spends
        // the budget against an already-decimated mesh, so the deviation from the ORIGINAL
        // compounds invisibly. Whatever this lands on is what the error bound permits.
        const next = step("5-simplify");
        gt("simplify", current, next, "--ratio", "0", "--error", String(SIMPLIFY_ERROR));
        current = next;
        currentTris = countTriangles(current);
        console.log(`  simplify (error ${SIMPLIFY_ERROR}) → ${currentTris.toLocaleString()} triangles`);
    } else {
        // ── Count-driven: hit an exact triangle budget, re-aiming after every pass.
        //
        // One pass is normally enough — but the scan's UV islands split vertices along every
        // seam, and meshoptimizer treats an attribute discontinuity as a constraint, so a pass
        // can stall well ABOVE the ratio it was asked for. Re-running on the already decimated
        // mesh changes the topology enough to get past that, so keep going while each pass is
        // still making progress. Targets above the stall point finish in a single pass.
        //
        // The cost is compounding error (see the header): each pass measures SIMPLIFY_ERROR
        // against the previous pass's output, not the source. If detail matters more than the
        // exact count, use --error instead.
        for (let pass = 1; pass <= SIMPLIFY_MAX_PASSES; pass++) {
            if (currentTris <= targetTris * (1 + SIMPLIFY_TOLERANCE)) {
                break;
            }
            const next = step(`5-simplify-${pass}`);
            const ratio = targetTris / currentTris;
            gt("simplify", current, next, "--ratio", ratio.toFixed(5), "--error", String(SIMPLIFY_ERROR));
            const tris = countTriangles(next);
            console.log(`  simplify pass ${pass} (ratio ${ratio.toFixed(5)}) → ${tris.toLocaleString()} triangles`);
            if (tris >= currentTris) {
                console.log("    no further reduction possible — stopping");
                break;
            }
            current = next;
            currentTris = tris;
        }
    }

    gt("resize", current, step("7-resize"), "--width", String(texturePx), "--height", String(texturePx));
    gt("uastc", step("7-resize"), step("8-uastc"), "--slots", "normalTexture", "--level", "4", "--zstd", "18");
    gt("etc1s", step("8-uastc"), step("9-etc1s"), "--slots", "{baseColorTexture,metallicRoughnessTexture}", "--quality", "255", "--compression", "4");

    mkdirSync(dirname(out), { recursive: true });
    copyFileSync(step("9-etc1s"), out);

    const before = readFileSync(src).byteLength;
    const after = readFileSync(out).byteLength;
    console.log(
        `\nWrote ${(after / 1024 / 1024).toFixed(2)} MB → ${out}\n` +
            `  ${(before / 1024 / 1024).toFixed(2)} MB → ${(after / 1024 / 1024).toFixed(2)} MB ` +
            `(${(before / after).toFixed(1)}x smaller), ${countTriangles(out).toLocaleString()} triangles\n` +
            `\nNow re-bake the collision height map from THIS file:\n` +
            `  pnpm tsx lab/public/waterfall/scripts/bake-rock-heightmap.ts --src ${out}`,
    );
} finally {
    if (keepTemp) {
        console.log(`\nIntermediates kept in ${tmp}`);
    } else if (existsSync(tmp)) {
        rmSync(tmp, { recursive: true, force: true });
    }
}
