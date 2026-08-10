/**
 * Convert the Aquanova skybox cube faces into a single equirectangular image.
 *
 * Blender's World "Environment Texture" node only accepts equirectangular (or
 * mirror-ball) projections — it has no cube-map input — so the six faces the
 * runtime samples directly have to be reprojected before Cycles can use the
 * star field as the light that reaches the ship through its hull windows.
 *
 * ── Why the axes get shuffled ─────────────────────────────────────────────────
 * The output must line up with the SHIP as Blender sees it, not with Babylon's
 * world, and the two engines disagree twice over.
 *
 *   Babylon ← glTF : Lite's synthetic `__root__` applies diag(-1, 1, 1)
 *                    (see loader-gltf/gltf-parser.ts RH_TO_LH_ROOT), so
 *                    g = (-a.x, a.y, a.z).
 *   glTF → Blender : the glTF importer restores Z-up, b = (g.x, -g.z, g.y).
 *
 * Composing them gives the round trip used below:
 *
 *   Blender b = (-a.x, -a.z,  a.y)      Babylon a = (-b.x, b.z, -b.y)
 *
 * Sanity check: Babylon's up (0, 1, 0) lands on Blender's up (0, 0, 1), and the
 * handedness flip shows up as the negated X — exactly the mirror `__root__`
 * introduced when the ship was imported.
 *
 * ── Conventions relied on ─────────────────────────────────────────────────────
 * Cube faces follow the standard D3D/GL/WebGPU cube-map rules, in the order Lite
 * uploads them (texture/cube-texture.ts): +X, -X, +Y, -Y, +Z, -Z. The lookup
 * vector is the skybox box's object-space position, straight from
 * shaders/skybox-cubemap.fragment.wgsl, so no extra rotation is in play.
 *
 * Cycles maps an equirectangular environment as (kernel/geom, equirectangular_to_direction):
 *   phi = pi * (1 - 2u),  theta = pi * (v - 0.5)
 *   dir = (cos(theta)cos(phi), cos(theta)sin(phi), sin(theta))
 * so the image centre looks down Blender +X and the top row is +Z (up).
 *
 * Output stays 8-bit sRGB PNG: the sources are 8-bit sRGB, so a float format
 * would invent range that was never captured. Brightness is meant to be dialled
 * in with the Background node's Strength at bake time, not baked in here.
 *
 * Usage: node scripts/skybox-cube-to-equirect.mjs [--width 4096] [--samples 2]
 */
import { createRequire } from "node:module";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");
const require = createRequire(resolve(repoRoot, "package.json"));

// sharp is a repo-level dev dependency but is not hoisted to the root
// node_modules, so resolve it out of the pnpm store explicitly.
const sharp = require("./node_modules/.pnpm/sharp@0.34.5/node_modules/sharp");

const SKYBOX_DIR = resolve(repoRoot, "lab/public/aquanova/skybox");
const OUT_PATH = resolve(SKYBOX_DIR, "sky_equirect.png");
/** Upload order in texture/cube-texture.ts, which is also the GPU face order. */
const FACES = ["px", "nx", "py", "ny", "pz", "nz"];

function arg(name, fallback) {
    const i = process.argv.indexOf(`--${name}`);
    return i === -1 ? fallback : Number(process.argv[i + 1]);
}

/**
 * Pick the cube face for a direction and return its face-local coordinates.
 *
 * Straight out of the cube-map spec: the largest component chooses the face, and
 * the other two — negated per face so that neighbouring faces stay continuous —
 * become s/t once divided by it. `t` runs downwards, matching image row order.
 */
function cubeLookup(x, y, z) {
    const ax = Math.abs(x);
    const ay = Math.abs(y);
    const az = Math.abs(z);
    let face, ma, sc, tc;
    if (ax >= ay && ax >= az) {
        ma = ax;
        if (x > 0) (face = 0), (sc = -z), (tc = -y);
        else (face = 1), (sc = z), (tc = -y);
    } else if (ay >= az) {
        ma = ay;
        if (y > 0) (face = 2), (sc = x), (tc = z);
        else (face = 3), (sc = x), (tc = -z);
    } else {
        ma = az;
        if (z > 0) (face = 4), (sc = x), (tc = -y);
        else (face = 5), (sc = -x), (tc = -y);
    }
    return { face, s: 0.5 * (sc / ma + 1), t: 0.5 * (tc / ma + 1) };
}

/** Bilinear tap into one face, clamped at the border.
 *
 *  Clamping rather than reaching into the neighbouring face leaves at most a
 *  half-texel error along the twelve cube seams; at 1024 px faces that is far
 *  below what a backdrop or a diffuse bake can resolve. */
function sampleFace(faces, face, s, t, size) {
    const fx = Math.min(Math.max(s * size - 0.5, 0), size - 1);
    const fy = Math.min(Math.max(t * size - 0.5, 0), size - 1);
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(x0 + 1, size - 1);
    const y1 = Math.min(y0 + 1, size - 1);
    const wx = fx - x0;
    const wy = fy - y0;
    const data = faces[face];
    const out = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
        const p00 = data[(y0 * size + x0) * 3 + c];
        const p10 = data[(y0 * size + x1) * 3 + c];
        const p01 = data[(y1 * size + x0) * 3 + c];
        const p11 = data[(y1 * size + x1) * 3 + c];
        out[c] = (p00 * (1 - wx) + p10 * wx) * (1 - wy) + (p01 * (1 - wx) + p11 * wx) * wy;
    }
    return out;
}

async function main() {
    const width = arg("width", 4096);
    const height = width / 2;
    const ss = arg("samples", 2); // supersampling grid, ss × ss taps per pixel

    const faces = [];
    let size = 0;
    for (const f of FACES) {
        const file = resolve(SKYBOX_DIR, `sky_${f}.png`);
        const { data, info } = await sharp(file).removeAlpha().raw().toBuffer({ resolveWithObject: true });
        if (info.width !== info.height) throw new Error(`${file}: cube faces must be square (got ${info.width}×${info.height})`);
        if (size && info.width !== size) throw new Error(`${file}: face size ${info.width} does not match ${size}`);
        size = info.width;
        faces.push(data);
    }
    console.log(`loaded 6 × ${size}² faces → ${width}×${height} equirectangular, ${ss}× ${ss} supersampling`);

    const out = Buffer.allocUnsafe(width * height * 3);
    const inv = 1 / (ss * ss);
    for (let py = 0; py < height; py++) {
        for (let px = 0; px < width; px++) {
            let r = 0;
            let g = 0;
            let b = 0;
            for (let sy = 0; sy < ss; sy++) {
                for (let sx = 0; sx < ss; sx++) {
                    const u = (px + (sx + 0.5) / ss) / width;
                    // Row 0 is the top of the file, which is Cycles' v = 1.
                    const v = 1 - (py + (sy + 0.5) / ss) / height;
                    const phi = Math.PI * (1 - 2 * u);
                    const theta = Math.PI * (v - 0.5);
                    const ct = Math.cos(theta);
                    // Blender-space direction …
                    const bx = ct * Math.cos(phi);
                    const by = ct * Math.sin(phi);
                    const bz = Math.sin(theta);
                    // … re-expressed in the Babylon space the cube was authored for.
                    const { face, s, t } = cubeLookup(-bx, bz, -by);
                    const c = sampleFace(faces, face, s, t, size);
                    r += c[0];
                    g += c[1];
                    b += c[2];
                }
            }
            const o = (py * width + px) * 3;
            out[o] = Math.round(r * inv);
            out[o + 1] = Math.round(g * inv);
            out[o + 2] = Math.round(b * inv);
        }
    }

    await sharp(out, { raw: { width, height, channels: 3 } })
        .png({ compressionLevel: 9 })
        .toFile(OUT_PATH);
    console.log(`wrote ${OUT_PATH}`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
