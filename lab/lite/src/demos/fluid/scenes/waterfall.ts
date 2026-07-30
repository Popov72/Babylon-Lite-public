// Waterfall demo — a real, scanned-style ROCK FORMATION (glTF) standing on flat ground,
// with water pouring off its summit and cascading down its natural terraces into the
// shallow pool that spreads around its foot.
//
// The collision surface is a HEIGHTFIELD SDF baked straight out of that very model:
//
//     sceneSdf(p) = p.y - heightAt(p.x, p.z)          (positive ABOVE the terrain = fluid)
//     heightAt    = max(rock height map, FLOOR_Y)
//
// `lab/public/waterfall/scripts/bake-rock-heightmap.ts` rasterises the model's triangles into a 512² grid of
// max world-Y ("what does a ray fired straight down hit?") and writes a compact binary
// (48-byte header + f32 heights, ~1 MB). The demo uploads it verbatim into the sim's
// `sceneSdfGrid` storage buffer and samples it BILINEARLY in WGSL, so the water collides
// with the exact silhouette it is rendered against — no procedural stand-in, no second
// authoring of the terrain. A heightfield is exact here: the formation is a mound with no
// playable overhangs, and it costs 1 MB instead of the tens of MB a 3D distance grid would.
//
// Heights are stored in the MODEL's own local space; the demo's world placement (uniform
// scale + yaw + base height) is applied at UBO-write time, so re-framing the rock — even
// live, via the "Rock yaw" slider — never needs a re-bake.
//
// Outside the rock's silhouette the height field falls back to a FLAT floor at FLOOR_Y,
// which the demo also draws as a plain ground quad — so every collidable surface in the
// scene is visible and vice-versa. Water is recirculated: a thin intake slab hugging that
// floor relaunches particles from a few SPRINGS on the rock's flat top — their sites are
// derived from the height map itself (the flattest, highest, well-separated ledges) so they
// stay right whatever model or yaw is in use — giving a seamless springs → terraces → pool
// → springs loop.

import { addToScene, createMeshFromData, createPbrMaterial, loadGltf, setMeshVisible } from "babylon-lite";
import type { Mesh, SceneNode } from "babylon-lite";
import type { EmitterConfig, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import type { DemoParam, FluidCtx, FluidDemo } from "../demo.js";
import { demoAssetUrl } from "../../demo-asset-url.js";
import { screenRay } from "../pick.js";

const ROCK_URL = demoAssetUrl("./waterfall/rock.glb", import.meta.url);
const HEIGHTMAP_URL = demoAssetUrl("./waterfall/rock-heightmap.bin", import.meta.url);
// rock.glb is the OPTIMISED asset: 60 k triangles and 2048² KTX2 maps, ~5.3 MB, produced from
// the raw photogrammetry scan by lab/public/waterfall/scripts/optimize-rock.ts. The scan itself is kept
// beside it as rock-high.glb (2.45 M triangles, ~72 MB, gitignored and deliberately NOT staged
// into the demo bundle) together with its own bake, rock-heightmap-high.bin — so the model can
// be re-derived at a different triangle budget, or compared against, without re-downloading it.
// Note that the two are NOT interchangeable at runtime: simplification moves the surface, so a
// model is only ever valid with the height map baked from that same file.
/** Open-sky HDR environment: Poly Haven "belfast_sunset_puresky" (CC0), 2k Radiance panorama.
 *  Declared HERE, next to the demo's other assets, so it is resolved against the same module
 *  URL as the rock and the height map — and served out of the same `waterfall/` folder the
 *  demo-bundle pipeline stages. The core reads it as this demo's `envUrl`: it is the
 *  waterfall's skybox background, its fluid-surface reflections AND its terrain IBL. Being a
 *  raw `.hdr` (not a pre-filtered `.env`) it is loaded via `loadHdrEnvironment`.
 *
 *  Kept as a LOCAL file rather than the Poly Haven CDN URL the other picker entries use: this
 *  one is the demo's default backdrop, so it has to be there on first paint without depending
 *  on an external host. */
export const WATERFALL_ENV_URL = demoAssetUrl("./waterfall/belfast_sunset_puresky_2k.hdr", import.meta.url);
/** The previous default (Poly Haven "quarry_04_puresky", CC0), still offered in the
 *  environment picker and still staged with the demo. */
export const WATERFALL_QUARRY_ENV_URL = demoAssetUrl("./waterfall/quarry_04_puresky_2k.hdr", import.meta.url);

// ── Baked height-map layout ────────────────────────────────────────────────────────────
// The grid resolution is fixed at build time so the GPU storage buffer can be allocated
// (and bound into the sims' pipelines) up front, BEFORE the async fetch resolves: the
// sims only inject the `sceneSdfGrid` binding when `SceneSdfSpec.sdfGrid` is present at
// `setSceneSdf` time. A zero-filled grid reads as "model height 0 everywhere" = the bare
// basin, which is exactly the right stand-in until the real heights arrive. A bake at a
// different resolution is resampled onto this grid on load rather than rejected.
const HM_N = 512;
/** 'BLHM' — must match `HEIGHTMAP_MAGIC` in lab/public/waterfall/scripts/bake-rock-heightmap.ts. */
const HM_MAGIC = 0x4d484c42;
const HM_HEADER_BYTES = 48;

// ── World placement of the rock ────────────────────────────────────────────────────────
const ROCK_SCALE = 18; // uniform model→world scale: the formation spans ≈ ±8.8, summit ≈ y 7.0
const ROCK_CX = 0; // world X of the model origin (the model is authored centred on XZ)
const ROCK_CZ = 0; // world Z of the model origin
// Orientation of the model about Y — this is a COMPOSITION choice: it picks which face of the
// formation the camera sees. The collision transform lives in the UBO rather than being baked
// into the height map, so the model and the surface the water rides can never diverge.
//
// The catch is that the summit's two terraces both drain down ONE face, and at this angle that
// face points away from the viewer: tracing steepest descent on the bake from inside the source
// outlines (scratch/wf-flow.mts) lands only ~12% of the emitted water on the visible front.
// Re-yawing into the 330deg-20deg window would fix the drainage but show a different face, so
// the water is nudged toward the camera at emission instead — see FRONT_BIAS below.
const ROCK_YAW_DEG = 204;
/** Fixed model→world size multiplier on top of ROCK_SCALE. The rock is a piece of set
 *  dressing, so its size is presentation, not a parameter: pinning it here keeps the
 *  formation identical across every quality tier and solver (the per-quality presets used to
 *  each carry their own scale, which silently resized the rock on a quality switch). */
const MESH_SCALE = 3;

// ── Front bias ─────────────────────────────────────────────────────────────────────────
// Emitted water gets a small HORIZONTAL launch velocity toward the viewer, on top of the
// downward `sourceSpeed`, so it tends to crest the terrace's front rim rather than the back
// one. Cheap and local: it only sets the velocity a particle is recycled with, so nothing
// about the collision field, the outlines or the composition changes.
//
// Camera azimuth the quality presets frame the rock from. The ArcRotate position is
// r·(cos α·sin β, cos β, sin α·sin β), so the horizontal direction from the rock TOWARD the
// viewer is just (cos α, sin α) — β drops out once normalised. World space, not model space:
// "the front" is a property of the shot, so the nudge must not rotate with ROCK_YAW_DEG.
const VIEW_ALPHA = -2.017695550245583;
const FRONT_DIR_X = Math.cos(VIEW_ALPHA);
const FRONT_DIR_Z = Math.sin(VIEW_ALPHA);

// ── The ground the rock stands on (COLLIDED only, never drawn) ─────────────────────────
// A conceptual flat plane at FLOOR_Y: outside the model's silhouette the height field falls
// back to it and water spreads out over it in a shallow sheet. FLOOR_Y is 0 because that is
// the fluid domain's own floor (the MLS/PB-MPM grids start at y = -1 and PBF clamps at 0).
// Nothing is rendered for it — the rock is presented against open sky, and a visible quad
// only ever announced itself as a hard edge once the camera dipped below the horizon.
const FLOOR_Y = 0;
/** Half-width of the initial SEED disc of water around the rock, world units at 1× mesh
 *  scale. Sized a little wider than the rock's own ≈10.2-unit footprint so the pool starts
 *  at its foot. (The pump's intake is separate — it spans the whole domain, see buildConfig.) */
const INTAKE_R = 13;
// Seat the model's base just BELOW the floor so the rock rises out of the ground rather
// than resting on it — at the silhouette the height map falls to the model's own floor, and
// max(rock, FLOOR_Y) then hands the ground back to the quad with no seam or z-fight.
const ROCK_Y0 = FLOOR_Y - 0.05;

// Start-of-sim charge: the sources are seeded as a shallow HEAD OF WATER standing in each
// outline, and the whole pool is poured into them over WARMUP_FRAMES. The terraces fill and
// overflow within a couple of seconds, so the demo opens with a heavy burst off the summit
// instead of a thin trickle — while the water still starts AT the springs, never on the ground.
//
// The head is deliberately shallow: it only has to be deep enough that a frame's worth of
// particles lands sparsely. Injection density is (count / WARMUP_FRAMES) / (area × depth), and
// at the defaults that is ~12 particles per world unit³ against the ~72 a settled pool holds —
// so the ramp can be this fast without a frame-1 pressure spike. A full-height column holding
// every particle at once would be ~42 units tall over a 16-unit rock: a tower in the sky.
const SOURCE_FILL_H = 2.0;

// Frames over which the pool pours into the sources (~1.5 s at 60 fps). Lower = harder burst,
// but injection density rises with it. Honoured by every backend.
const WARMUP_FRAMES = 90;

// ── Spring outlines on the rock's flat top ─────────────────────────────────────────────
// Each source is a POLYGON in the rock's MODEL XZ plane, given a small height and used as the
// spawn volume for one emitter — so water wells up over the whole terrace and spills off its
// real edge. Model space (not world) is what makes them free under "Rock yaw" and "Mesh
// scale": the outline rotates and resizes with the rock exactly like the height map does.
//
// AUTHORED_POLYGONS is the source of truth. Leave it empty and the demo falls back to convex
// hulls auto-derived from the height map, which is only ever a starting point — tick "Author
// polygons" to draw the real outlines by hand and paste the printed arrays back in here.
const AUTHORED_POLYGONS: [number, number][][] = [
    [
        [-0.1067, -0.1666],
        [-0.1472, -0.2033],
        [-0.1776, -0.1994],
        [-0.2711, -0.1369],
        [-0.2807, -0.1075],
        [-0.2709, -0.0506],
        [-0.1961, -0.0154],
        [-0.1832, 0.0258],
        [-0.1634, 0.0564],
        [-0.1253, 0.0976],
        [-0.0938, 0.1161],
        [-0.0613, 0.0926],
        [-0.089, 0.0296],
        [-0.0738, 0.0007],
        [-0.0973, -0.0491],
        [-0.102, -0.0767],
        [-0.1174, -0.0984],
        [-0.1004, -0.1485],
    ],
    [
        [0.1063, -0.268],
        [0.0707, -0.2589],
        [0.0385, -0.2363],
        [0.0196, -0.2467],
        [-0.007, -0.2291],
        [-0.017, -0.1993],
        [-0.0492, -0.1901],
        [-0.0644, -0.1553],
        [-0.0718, -0.1104],
        [-0.0703, -0.0733],
        [-0.0504, -0.0248],
        [-0.0271, 0.0036],
        [0.0312, 0.0376],
        [0.0666, 0.0198],
        [0.0995, -0.0162],
        [0.1157, -0.0615],
        [0.136, -0.0819],
        [0.1709, -0.1003],
        [0.189, -0.1351],
        [0.1917, -0.1745],
    ],
];

/** How many auto-derived outlines to seed when AUTHORED_POLYGONS is empty (one per terrace). */
const SHELF_COUNT = 2;
/** Only the top of this fraction of the model's height range is considered "the top". */
const SHELF_TOP_BAND = 0.28;
/** Maximum |∇h| (model height units per model unit) for a cell to count as flat. */
const SHELF_SLOPE_MAX = 0.8;
/** Reject specks: a shelf must hold at least this many cells of the (half-resolution) scan. */
const SHELF_MIN_CELLS = 24;
/** Half-height of a source's spawn prism, world units at 1× mesh scale — the "small height"
 *  given to the 2D outline. Thin, so spawns hug the terrace instead of raining from above. */
const SHELF_BOX_HALF_H = 0.01;
/** Clearance between the terrace's highest point and the BOTTOM of the prism, world units at
 *  1× mesh scale. Keeps a spawn from ever landing inside the rock on an uneven surface. */
const SHELF_LIFT = 0.005;

/** A derived source: one flat shelf on the rock's top. `pts` are its model-space cell
 *  centres (re-projected to world whenever the yaw or scale changes); `topH` is the highest
 *  model-space height in it, so the spawn box can sit clear of the whole surface. */
interface Shelf {
    pts: { mx: number; mz: number }[];
    topH: number;
}
// Fallback until the bake lands: one tiny shelf at the model's nominal high point.
let shelves: Shelf[] = [{ pts: [{ mx: -0.0076, mz: 0.012 }], topH: 0.3913 }];

// ── Collision WGSL ─────────────────────────────────────────────────────────────────────
// UBO (4 × vec4, re-written every frame — see writeSdfParams):
//   p0 = (originX, originZ, invCellX, invCellZ)   model-space grid mapping
//   p1 = (nx, nz, rockY0, rockScale)
//   p2 = (cosYaw / rockScale, sinYaw / rockScale, centreX, centreZ)
//   p3 = (floorY, unused, unused, unused)
const SDF_WGSL = `fn wfGrid(i: i32, j: i32, nx: i32, nz: i32) -> f32 {
let c = clamp(vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(nx - 1, nz - 1));
return sceneSdfGrid[c.x + nx * c.y];
}
fn heightAt(wx: f32, wz: f32) -> f32 {
let nx = i32(sceneSdfParams.p1.x);
let nz = i32(sceneSdfParams.p1.y);
let dx = wx - sceneSdfParams.p2.z;
let dz = wz - sceneSdfParams.p2.w;
// world → model: inverse yaw with the inverse uniform scale folded into the cos/sin pair.
let mx = dx * sceneSdfParams.p2.x - dz * sceneSdfParams.p2.y;
let mz = dx * sceneSdfParams.p2.y + dz * sceneSdfParams.p2.x;
let g = vec2<f32>((mx - sceneSdfParams.p0.x) * sceneSdfParams.p0.z, (mz - sceneSdfParams.p0.y) * sceneSdfParams.p0.w);
let b = floor(g);
let f = g - b;
let i = i32(b.x);
let j = i32(b.y);
let h0 = mix(wfGrid(i, j, nx, nz), wfGrid(i + 1, j, nx, nz), f.x);
let h1 = mix(wfGrid(i, j + 1, nx, nz), wfGrid(i + 1, j + 1, nx, nz), f.x);
let rock = sceneSdfParams.p1.z + mix(h0, h1, f.y) * sceneSdfParams.p1.w;
return max(rock, sceneSdfParams.p3.x);
}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
return pt.y - heightAt(pt.x, pt.z);
}`;

// ── CPU mirror of the WGSL height field ────────────────────────────────────────────────
// Used to seat the nozzles and seed the pool. Reads the SAME
// Float32Array that was uploaded to the GPU, so the two can never drift.
let hmData: Float32Array | null = null;
// Safe defaults until the bake lands: a degenerate cell size would divide by zero.
let hmOx = -0.5;
let hmOz = -0.5;
let hmInvCx = HM_N - 1;
let hmInvCz = HM_N - 1;

function rockHeightWorld(wx: number, wz: number, cosYaw: number, sinYaw: number, scale: number): number {
    if (!hmData) {
        return ROCK_Y0;
    }
    const dx = wx - ROCK_CX;
    const dz = wz - ROCK_CZ;
    const mx = (dx * cosYaw - dz * sinYaw) / scale;
    const mz = (dx * sinYaw + dz * cosYaw) / scale;
    const gx = Math.min(HM_N - 1.0001, Math.max(0, (mx - hmOx) * hmInvCx));
    const gz = Math.min(HM_N - 1.0001, Math.max(0, (mz - hmOz) * hmInvCz));
    const i = Math.floor(gx);
    const j = Math.floor(gz);
    const fx = gx - i;
    const fz = gz - j;
    const h00 = hmData[j * HM_N + i]!;
    const h10 = hmData[j * HM_N + i + 1]!;
    const h01 = hmData[(j + 1) * HM_N + i]!;
    const h11 = hmData[(j + 1) * HM_N + i + 1]!;
    const h = (h00 * (1 - fx) + h10 * fx) * (1 - fz) + (h01 * (1 - fx) + h11 * fx) * fz;
    return ROCK_Y0 + h * scale;
}

/** Find the flat SHELVES on the rock's top straight out of the baked height map. Cells in the
 *  highest {@link SHELF_TOP_BAND} of the model's height range whose gradient is below
 *  {@link SHELF_SLOPE_MAX} are marked, then grouped into connected regions (4-connectivity,
 *  iterative flood fill on a half-resolution grid). The {@link SHELF_COUNT} largest regions
 *  are the shelves. Connected components — rather than the "best N separated points" this
 *  replaced — are what make a real SURFACE available: a component carries its full extent, so
 *  the emitter can be a box that covers the terrace instead of a dot sitting on it. */
function findTopShelves(): Shelf[] {
    if (!hmData) {
        return shelves;
    }
    let maxH = -Infinity;
    let minH = Infinity;
    for (const h of hmData) {
        if (h > maxH) {
            maxH = h;
        }
        if (h < minH) {
            minH = h;
        }
    }
    const cutoff = maxH - SHELF_TOP_BAND * (maxH - minH);
    const cellX = 1 / hmInvCx;
    const cellZ = 1 / hmInvCz;
    // Half-resolution scan grid: identical regions, a quarter of the flood-fill work.
    const RN = HM_N >> 1;
    const flat = new Uint8Array(RN * RN);
    for (let rj = 1; rj < RN - 1; rj++) {
        for (let ri = 1; ri < RN - 1; ri++) {
            const i = ri << 1;
            const j = rj << 1;
            const h = hmData[j * HM_N + i]!;
            if (h < cutoff) {
                continue;
            }
            const gx = (hmData[j * HM_N + i + 1]! - hmData[j * HM_N + i - 1]!) / (2 * cellX);
            const gz = (hmData[(j + 1) * HM_N + i]! - hmData[(j - 1) * HM_N + i]!) / (2 * cellZ);
            if (Math.hypot(gx, gz) <= SHELF_SLOPE_MAX) {
                flat[rj * RN + ri] = 1;
            }
        }
    }
    // Iterative flood fill (an explicit stack — a recursive fill would blow the JS stack on a
    // region of a few thousand cells).
    const found: Shelf[] = [];
    const stack: number[] = [];
    for (let start = 0; start < flat.length; start++) {
        if (flat[start] !== 1) {
            continue;
        }
        const pts: { mx: number; mz: number }[] = [];
        let topH = -Infinity;
        stack.push(start);
        flat[start] = 2; // claimed
        while (stack.length > 0) {
            const c = stack.pop()!;
            const ri = c % RN;
            const rj = (c - ri) / RN;
            const h = hmData[(rj << 1) * HM_N + (ri << 1)]!;
            if (h > topH) {
                topH = h;
            }
            pts.push({ mx: hmOx + (ri << 1) * cellX, mz: hmOz + (rj << 1) * cellZ });
            if (ri > 0 && flat[c - 1] === 1) {
                flat[c - 1] = 2;
                stack.push(c - 1);
            }
            if (ri < RN - 1 && flat[c + 1] === 1) {
                flat[c + 1] = 2;
                stack.push(c + 1);
            }
            if (rj > 0 && flat[c - RN] === 1) {
                flat[c - RN] = 2;
                stack.push(c - RN);
            }
            if (rj < RN - 1 && flat[c + RN] === 1) {
                flat[c + RN] = 2;
                stack.push(c + RN);
            }
        }
        if (pts.length >= SHELF_MIN_CELLS) {
            found.push({ pts, topH });
        }
    }
    found.sort((a, b) => b.pts.length - a.pts.length);
    const picked = found.slice(0, SHELF_COUNT);
    console.warn(`[waterfall] shelves: ${found.length} found, using ${picked.length} — cells ${picked.map((s) => s.pts.length).join(", ")}`);
    return picked.length ? picked : shelves;
}

/** Convex hull (monotone chain) of model-space points, counter-clockwise. Used only to turn an
 *  auto-detected shelf blob into a usable starting outline — a hull is a far better first guess
 *  than the blob's bounding box, but it still can't follow a concave terrace, which is exactly
 *  why the outlines are meant to be authored by hand. */
function convexHull(pts: readonly { mx: number; mz: number }[]): [number, number][] {
    if (pts.length < 3) {
        return pts.map((p) => [p.mx, p.mz] as [number, number]);
    }
    const p = [...pts].sort((a, b) => a.mx - b.mx || a.mz - b.mz);
    const cross = (o: [number, number], a: [number, number], b: [number, number]): number => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const build = (src: typeof p): [number, number][] => {
        const out: [number, number][] = [];
        for (const q of src) {
            const v: [number, number] = [q.mx, q.mz];
            while (out.length >= 2 && cross(out[out.length - 2]!, out[out.length - 1]!, v) <= 0) {
                out.pop();
            }
            out.push(v);
        }
        out.pop();
        return out;
    };
    return [...build(p), ...build([...p].reverse())];
}

/** Area of a closed polygon in the XZ plane (shoelace, winding-independent). */
function polygonArea(poly: readonly [number, number][]): number {
    let a = 0;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        a += poly[j]![0] * poly[i]![1] - poly[i]![0] * poly[j]![1];
    }
    return Math.abs(a) / 2;
}

/** Even-odd ray-crossing test in the XZ plane. */
function pointInPolygon(x: number, z: number, poly: readonly [number, number][]): boolean {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const [xi, zi] = poly[i]!;
        const [xj, zj] = poly[j]!;
        if (zi > z !== zj > z && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) {
            inside = !inside;
        }
    }
    return inside;
}

export function createWaterfallDemo(ctx: FluidCtx): FluidDemo {
    const { engine } = ctx;

    // Live-tweakable params (exposed in the Demo-parameters UI). `rockYaw`/`meshScale` are no
    // longer among them — they are pinned presentation constants (see MESH_SCALE) — but they
    // stay in this bag because everything downstream reads the rock's placement through it.
    const wf = {
        sourceSpeed: 0.6,
        emitRate: 0.7,
        spread: 0.5,
        frontBias: 4,
        rockYaw: ROCK_YAW_DEG,
        meshScale: MESH_SCALE,
        bloom: false,
        bloomIntensity: 0.25,
        bloomThreshold: 0.0,
    };

    /** Push the three bloom params to the shared presentation stage. The stage is owned by
     *  the core and shared with every other demo, so it is only ever driven while THIS demo
     *  is on screen — onLeave forces it off again. */
    const applyBloom = (): void => {
        ctx.setBloom({ enabled: wf.bloom, intensity: wf.bloomIntensity, threshold: wf.bloomThreshold });
    };

    let cosYaw = Math.cos((wf.rockYaw * Math.PI) / 180);
    let sinYaw = Math.sin((wf.rockYaw * Math.PI) / 180);
    /** Effective model→world scale: the authored base scale × the live "Mesh scale" slider. */
    const worldScale = (): number => ROCK_SCALE * wf.meshScale;

    const terrainHeightWorld = (wx: number, wz: number): number => Math.max(rockHeightWorld(wx, wz, cosYaw, sinYaw, worldScale()), FLOOR_Y);

    /** Model-space point → world, with the current yaw + scale + base offset. */
    const modelToWorld = (mx: number, my: number, mz: number): [number, number, number] => {
        const s = worldScale();
        return [ROCK_CX + (mx * cosYaw + mz * sinYaw) * s, ROCK_Y0 + my * s, ROCK_CZ + (-mx * sinYaw + mz * cosYaw) * s];
    };

    // Height-map storage buffer: allocated (zero-filled) up front so the sims can bind it
    // from the very first setSceneSdf; the bake is written into it when the fetch lands.
    const hmBuffer = engine._device.createBuffer({
        label: "waterfall-heightmap",
        size: HM_N * HM_N * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const sdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, };",
        sdf: SDF_WGSL,
        // Per-particle push-out + restitution: the rock is an irregular, steeply terraced
        // surface the coarse MLS grid wall would mis-sample.
        gridConfine: false,
        buffer: ctx.sceneSdfBuffer,
        sdfGrid: hmBuffer,
    };

    let active = false; // true while the waterfall demo is the on-screen demo
    // Debounce state for the heavy mesh-scale path (the sim-domain rebuild). Only reachable
    // if the "Mesh scale" row below is uncommented; the scale is otherwise fixed.
    let meshScaleTimer: ReturnType<typeof setTimeout> | null = null;
    let builtMeshScale = MESH_SCALE;

    // 16 floats — see the SDF_WGSL header for the layout. The core zeroes bytes 32..159 of
    // the shared UBO on every pair switch (the capsule's drain-hole ring lives there), so
    // update() re-writes this block every frame, exactly like the marble tower does.
    const sdfData = new Float32Array(16);
    const writeSdfParams = (): void => {
        if (!active) {
            return; // never stomp another demo's params in the SHARED buffer
        }
        sdfData[0] = hmOx;
        sdfData[1] = hmOz;
        sdfData[2] = hmInvCx;
        sdfData[3] = hmInvCz;
        sdfData[4] = HM_N;
        sdfData[5] = HM_N;
        sdfData[6] = ROCK_Y0;
        sdfData[7] = worldScale();
        sdfData[8] = cosYaw / worldScale();
        sdfData[9] = sinYaw / worldScale();
        sdfData[10] = ROCK_CX;
        sdfData[11] = ROCK_CZ;
        sdfData[12] = FLOOR_Y;
        sdfData[13] = 0;
        sdfData[14] = 0;
        sdfData[15] = 0;
        engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, sdfData);
    };

    // ── Waterfall sources: one POLYGON EMITTER per terrace on the rock's top. Each spawn
    //    volume is the terrace's own outline given a small height, so water wells up across the
    //    whole surface and spills over its real edge. A box could only ever be the bounding
    //    rectangle of a terrace — it spilled off the wrong edges — and the point nozzles before
    //    that were a few cm wide however the flow was tuned, which read as a hidden pipe rather
    //    than a spring. Everything scales with "Mesh scale". ──

    /** The outlines feeding the emitters, model space, in priority order: whatever is being
     *  authored (persisted to localStorage), then the committed AUTHORED_POLYGONS, then convex
     *  hulls auto-derived from the height map. */
    const sourcePolygons = (): [number, number][][] => {
        const drawn = authorPolys.filter((p) => p.length >= 3);
        if (drawn.length) {
            return drawn;
        }
        return AUTHORED_POLYGONS.length ? AUTHORED_POLYGONS : shelves.map((s) => convexHull(s.pts));
    };

    /** The outlines drawn as a faint tracing reference — never the ones being authored, so the
     *  reference stays visible underneath the outline replacing it. */
    const referencePolygons = (): [number, number][][] => (AUTHORED_POLYGONS.length ? AUTHORED_POLYGONS : shelves.map((s) => convexHull(s.pts)));

    /** Highest terrain inside a WORLD-space outline. Scanning the interior (not just the
     *  vertices) matters because a terrace can bulge in the middle, and the prism has to clear
     *  the whole surface or spawns land inside the rock. */
    const polygonTopY = (poly: [number, number][]): number => {
        let x0 = Infinity;
        let x1 = -Infinity;
        let z0 = Infinity;
        let z1 = -Infinity;
        for (const [x, z] of poly) {
            x0 = Math.min(x0, x);
            x1 = Math.max(x1, x);
            z0 = Math.min(z0, z);
            z1 = Math.max(z1, z);
        }
        let top = -Infinity;
        for (const [x, z] of poly) {
            top = Math.max(top, terrainHeightWorld(x, z));
        }
        const N = 12;
        for (let i = 0; i <= N; i++) {
            for (let j = 0; j <= N; j++) {
                const x = x0 + ((x1 - x0) * i) / N;
                const z = z0 + ((z1 - z0) * j) / N;
                if (pointInPolygon(x, z, poly)) {
                    top = Math.max(top, terrainHeightWorld(x, z));
                }
            }
        }
        return top === -Infinity ? FLOOR_Y : top;
    };

    /** A source as actual geometry: its world-space outline and the Y span of its prism. Shared
     *  by the emitters and by the initial spawn, so the water is seeded in exactly the volume it
     *  will later be emitted from. */
    const buildPrisms = (): { poly: [number, number][]; y0: number; y1: number; cx: number; cz: number }[] => {
        const s = wf.meshScale;
        const out: { poly: [number, number][]; y0: number; y1: number; cx: number; cz: number }[] = [];
        for (const poly of sourcePolygons()) {
            if (poly.length < 3) {
                continue;
            }
            // Project the outline into WORLD space; the emitter's spawn area is the polygon
            // itself, so nothing is inflated the way a rotated bounding box would be.
            const world = poly.map((p) => {
                const [wx, , wz] = modelToWorld(p[0], 0, p[1]);
                return [wx, wz] as [number, number];
            });
            let cx = 0;
            let cz = 0;
            for (const [x, z] of world) {
                cx += x / world.length;
                cz += z / world.length;
            }
            const halfH = SHELF_BOX_HALF_H * s;
            // Sit the prism's BOTTOM a clearance above the terrace's highest point.
            const mid = polygonTopY(world) + SHELF_LIFT * s + halfH;
            out.push({ poly: world, y0: mid - halfH, y1: mid + halfH, cx, cz });
        }
        return out;
    };

    const buildEmitters = (): EmitterConfig["emitters"] => {
        const list: EmitterConfig["emitters"] = [];
        // One emitter takes a single dir+speed, so fold the downward source speed and the
        // horizontal front nudge into one launch vector and hand over its direction + length.
        const vx = FRONT_DIR_X * wf.frontBias;
        const vz = FRONT_DIR_Z * wf.frontBias;
        const vy = -wf.sourceSpeed;
        const sp = Math.hypot(vx, vy, vz);
        for (const p of buildPrisms()) {
            const halfH = (p.y1 - p.y0) / 2;
            list.push({
                pos: [p.cx, (p.y0 + p.y1) / 2, p.cz], // x/z unused for a polygon emitter; kept meaningful
                dir: sp > 1e-6 ? [vx / sp, vy / sp, vz / sp] : [0, -1, 0],
                speed: sp,
                radius: halfH,
                halfExtents: [halfH, halfH, halfH], // only Y is read once `polygon` is set
                polygon: p.poly,
            });
        }
        return list;
    };

    // Intake: a THIN slab hugging the FLOOR, spanning the ENTIRE simulated footprint.
    //
    // The slab is thin so only the bottom layer of standing water is eligible to recycle —
    // the pool then holds a resting surface above it instead of being pumped dry, and the
    // level self-regulates around the slab's top.
    //
    // The footprint, though, must cover the whole domain. The ground is a flat, rimless
    // plane, so water that runs off the rock keeps sliding outward until the sim's own
    // domain wall stops it. Anything that ends up beyond the intake can NEVER be recycled,
    // so it strands permanently — and because both the domain and the intake are AXIS-
    // ALIGNED BOXES, the gap between them is narrowest at the four edge midpoints (±X, ±Z)
    // and widest at the corners. Strays therefore escape through those four faces and pile
    // up against the wall as FOUR symmetrical dead pools. Matching the intake to the domain
    // removes the dead zone entirely: every particle on the floor stays in the loop.
    const buildConfig = (): EmitterConfig => {
        const pr = ctx.simHalfExtentXZ * wf.meshScale;
        return {
            emitters: buildEmitters(),
            intakeMin: [ROCK_CX - pr, FLOOR_Y - 1.0, ROCK_CZ - pr],
            intakeMax: [ROCK_CX + pr, FLOOR_Y + 0.3, ROCK_CZ + pr],
            rate: wf.emitRate,
            spread: wf.spread,
        };
    };

    // The demo draws NO ground: the rock is presented against open sky, and a flat plane at
    // FLOOR_Y only ever showed up as a hard edge when the camera dipped below the horizon.
    // FLOOR_Y itself stays — it is the SDF's fallback floor, so the fluid still pools and the
    // intake still recycles against an invisible ground.
    //
    // The demo also casts NO shadows. With the ground gone the only receiver left was the rock
    // shadowing itself, which is not worth what it cost: the sun's CSM generator had to be
    // attached/detached around this demo, and the shadow pipeline's "no-colour" caster module
    // is preloaded once before registerScene from the material families of the declared
    // casters — a window the rock's async glTF always missed. Lighting is now IBL + sun only.
    //
    // One invisible mesh still has to exist, though, and it is NOT about shadows. addToScene
    // buckets meshes by their material's build GROUP: a group present when the scene is built
    // gets its renderables from a boot-time deferred builder, and a mesh that arrives later can
    // only be materialised by swapping into a group that has ALREADY been built. The rock is a
    // glTF that resolves long after registerScene, so with no other PBR mesh in the scene (the
    // ground quad is gone) it opens a BRAND-NEW group whose builder never runs — it silently
    // gets no renderable and never draws, with no error anywhere. This degenerate,
    // permanently-invisible PBR triangle is added synchronously so that group exists and is
    // built up front, giving the rock something to join. Delete it and the rock disappears.
    const pbrGroupWitness = createMeshFromData(
        engine,
        "waterfall-pbr-group-witness",
        new Float32Array(9),
        new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
        new Uint32Array([0, 1, 2]),
        new Float32Array(6)
    );
    pbrGroupWitness.material = createPbrMaterial({ baseColorFactor: [1, 1, 1, 1] });
    addToScene(ctx.scene, pbrGroupWitness);
    setMeshVisible(pbrGroupWitness, false);

    let containerVisible = true; // toggled by the "Show container / nozzle meshes" checkbox
    const meshes: Mesh[] = [];
    const collectMeshes = (node: SceneNode, out: Mesh[]): void => {
        if ("_gpu" in node) {
            out.push(node as unknown as Mesh);
        }
        for (const c of node.children) {
            collectMeshes(c, out);
        }
    };

    // ── The rock formation itself ──────────────────────────────────────────────────────
    let rockRoot: SceneNode | null = null;
    /** Re-place everything the "Mesh scale" / "Rock yaw" sliders drive. */
    const applyScaleTransforms = (): void => {
        if (rockRoot) {
            const rs = worldScale();
            rockRoot.position.set(ROCK_CX, ROCK_Y0, ROCK_CZ);
            rockRoot.scaling.set(rs, rs, rs);
            rockRoot.rotation.set(0, (wf.rockYaw * Math.PI) / 180, 0);
        }
    };
    void (async (): Promise<void> => {
        const asset = await loadGltf(engine, ROCK_URL);
        asset.animationGroups = undefined; // static prop — hold a fixed pose the SDF matches
        addToScene(ctx.scene, asset);
        rockRoot = asset.entities[0] as SceneNode;
        applyScaleTransforms();
        const mine: Mesh[] = [];
        collectMeshes(rockRoot, mine);
        for (const m of mine) {
            setMeshVisible(m, active && containerVisible);
            meshes.push(m);
        }
    })().catch((e: unknown) => console.warn("[waterfall] rock model load failed", e));

    // ── The baked height map ───────────────────────────────────────────────────────────
    // Uploaded verbatim into `hmBuffer` (already bound by the sims) and kept CPU-side for
    // seeding/placement. A bake at a different resolution is resampled onto the HM_N grid.
    void (async (): Promise<void> => {
        const res = await fetch(HEIGHTMAP_URL);
        if (!res.ok) {
            throw new Error(`${res.status} ${res.statusText}`);
        }
        const buf = await res.arrayBuffer();
        const head = new DataView(buf);
        if (head.getUint32(0, true) !== HM_MAGIC) {
            throw new Error("not a Babylon Lite height map");
        }
        const nx = head.getUint32(8, true);
        const nz = head.getUint32(12, true);
        const ox = head.getFloat32(16, true);
        const oz = head.getFloat32(20, true);
        const cx = head.getFloat32(24, true);
        const cz = head.getFloat32(28, true);
        const src = new Float32Array(buf, HM_HEADER_BYTES, nx * nz);
        hmOx = ox;
        hmOz = oz;
        if (nx === HM_N && nz === HM_N) {
            hmData = src;
            hmInvCx = 1 / cx;
            hmInvCz = 1 / cz;
        } else {
            // Bilinear resample onto the fixed grid so the pre-allocated (and already bound)
            // storage buffer stays valid whatever resolution the bake used.
            console.warn(`[waterfall] height map is ${nx}×${nz}, resampling to ${HM_N}×${HM_N}`);
            const dst = new Float32Array(HM_N * HM_N);
            const sx = ((nx - 1) * cx) / (HM_N - 1);
            const sz = ((nz - 1) * cz) / (HM_N - 1);
            for (let j = 0; j < HM_N; j++) {
                const gz = Math.min(nz - 1.0001, (j * sz) / cz);
                const j0 = Math.floor(gz);
                const fz = gz - j0;
                for (let i = 0; i < HM_N; i++) {
                    const gx = Math.min(nx - 1.0001, (i * sx) / cx);
                    const i0 = Math.floor(gx);
                    const fx = gx - i0;
                    const h0 = src[j0 * nx + i0]! * (1 - fx) + src[j0 * nx + i0 + 1]! * fx;
                    const h1 = src[(j0 + 1) * nx + i0]! * (1 - fx) + src[(j0 + 1) * nx + i0 + 1]! * fx;
                    dst[j * HM_N + i] = h0 * (1 - fz) + h1 * fz;
                }
            }
            hmData = dst;
            hmInvCx = 1 / sx;
            hmInvCz = 1 / sz;
        }
        engine._device.queue.writeBuffer(hmBuffer, 0, hmData);

        // Derive the shelf boxes from the real data now that it is in.
        shelves = findTopShelves();

        if (active) {
            writeSdfParams();
            ctx.refreshEmitters(); // the outlines now cover the model's real flat terraces
            ctx.refreshSpawn(); // and the seed volume is those outlines, not the fallback pool
            ctx.resetActiveSim(); // re-seed from the springs against the real terrain
        }
    })().catch((e: unknown) => console.warn("[waterfall] height map load failed", e));

    // ── Outline authoring ──────────────────────────────────────────────────────────────────
    // Tick "Author polygons" to trace the terraces by hand: the camera swings overhead, LMB
    // drops a vertex on the rock surface, and the result is printed/copied as a ready-to-paste
    // AUTHORED_POLYGONS literal. Vertices are stored in MODEL space, so an outline authored at
    // one yaw/scale stays correct at every other one.
    //   LMB    add a vertex      Backspace  undo last      X  clear this outline
    //   C      copy all outlines to the clipboard (also printed to the console)
    let authoring = false;
    let authorSlot = 0;
    let overlay: HTMLCanvasElement | null = null;
    let savedBeta = 0;

    // Outlines survive a reload through localStorage, so a shape you draw is still there next
    // run without a rebuild. Pressing C additionally prints/copies a literal to paste into
    // AUTHORED_POLYGONS above, which is how an outline becomes permanent and committable —
    // localStorage is this browser's working copy, the constant is the shared default.
    const POLY_STORE_KEY = "babylon-lite.fluid.waterfall.polygons";
    const loadStoredPolys = (): [number, number][][] | null => {
        try {
            const raw = localStorage.getItem(POLY_STORE_KEY);
            if (!raw) {
                return null;
            }
            const parsed: unknown = JSON.parse(raw);
            if (!Array.isArray(parsed)) {
                return null;
            }
            // Validate shape: a bad/stale entry must not break the demo on load.
            const polys = parsed.filter(
                (p): p is [number, number][] => Array.isArray(p) && p.every((v) => Array.isArray(v) && v.length === 2 && v.every((n) => typeof n === "number" && isFinite(n)))
            );
            return polys.length ? polys : null;
        } catch {
            return null;
        }
    };
    const authorPolys: [number, number][][] = loadStoredPolys() ?? [[], []];
    while (authorPolys.length < 2) {
        authorPolys.push([]);
    }
    /** Persist + push the edit straight into the emitters, so an outline takes effect as it is
     *  drawn instead of only after a reload. */
    const commitPolys = (): void => {
        try {
            localStorage.setItem(POLY_STORE_KEY, JSON.stringify(authorPolys.filter((p) => p.length >= 3)));
        } catch {
            // Private mode / quota — the outline still works for this session.
        }
        ctx.refreshEmitters();
        ctx.refreshSpawn(); // so a Reset after drawing seeds from the new outlines
    };

    /** World XZ under the cursor, by marching the cursor ray against the height field. Marching
     *  (rather than intersecting a flat plane) is what lets a vertex land on the terrace it is
     *  drawn over, whatever height that terrace happens to sit at. */
    const pickTerrain = (clientX: number, clientY: number): [number, number] | null => {
        const rect = ctx.canvas.getBoundingClientRect();
        const ray = screenRay(ctx.viewProjection(), clientX - rect.left, clientY - rect.top, rect.width, rect.height);
        if (!ray) {
            return null;
        }
        const { origin, dir } = ray;
        const at = (t: number): number => origin[1] + dir[1] * t - terrainHeightWorld(origin[0] + dir[0] * t, origin[2] + dir[2] * t);
        const step = 0.2 * wf.meshScale;
        const maxT = 400 * wf.meshScale;
        let prev = at(0);
        for (let t = step; t < maxT; t += step) {
            if (at(t) <= 0 && prev > 0) {
                // Bisect the bracketing interval for a crisp hit point.
                let lo = t - step;
                let hi = t;
                for (let k = 0; k < 24; k++) {
                    const m = (lo + hi) / 2;
                    if (at(m) > 0) {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                return [origin[0] + dir[0] * hi, origin[2] + dir[2] * hi];
            }
            prev = at(t);
        }
        return null;
    };

    /** World XZ → the rock's model XZ (inverse of modelToWorld). */
    const worldToModel = (wx: number, wz: number): [number, number] => {
        const s = worldScale();
        const dx = wx - ROCK_CX;
        const dz = wz - ROCK_CZ;
        return [(dx * cosYaw - dz * sinYaw) / s, (dx * sinYaw + dz * cosYaw) / s];
    };

    const formatPolys = (): string => {
        const body = authorPolys
            .filter((p) => p.length >= 3)
            .map((p) => `    [${p.map(([x, z]) => `[${x.toFixed(4)}, ${z.toFixed(4)}]`).join(", ")}],`)
            .join("\n");
        return `const AUTHORED_POLYGONS: [number, number][][] = [\n${body}\n];`;
    };

    const ensureOverlay = (): HTMLCanvasElement | null => {
        if (overlay) {
            return overlay;
        }
        const parent = ctx.canvas.parentElement;
        if (!parent) {
            return null;
        }
        overlay = document.createElement("canvas");
        overlay.style.cssText = "position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;z-index:5";
        if (getComputedStyle(parent).position === "static") {
            parent.style.position = "relative";
        }
        parent.appendChild(overlay);
        return overlay;
    };

    /** Draw the outlines over the render canvas: the outlines actually feeding the emitters as
     *  a faint reference to trace, and the one being authored on top of them. */
    const drawOverlay = (): void => {
        const cv = ensureOverlay();
        if (!cv) {
            return;
        }
        const rect = ctx.canvas.getBoundingClientRect();
        const w = Math.max(1, Math.round(rect.width));
        const h = Math.max(1, Math.round(rect.height));
        if (cv.width !== w || cv.height !== h) {
            cv.width = w;
            cv.height = h;
        }
        const g = cv.getContext("2d");
        if (!g) {
            return;
        }
        g.clearRect(0, 0, w, h);
        const vp = ctx.viewProjection();
        // Model XZ → screen, seated on the terrain so the outline hugs the surface it traces.
        const toScreen = (mx: number, mz: number): [number, number] | null => {
            const [wx, , wz] = modelToWorld(mx, 0, mz);
            const wy = terrainHeightWorld(wx, wz) + 0.05 * wf.meshScale;
            const cw = vp[3]! * wx + vp[7]! * wy + vp[11]! * wz + vp[15]!;
            if (cw <= 1e-6) {
                return null;
            }
            const cx = vp[0]! * wx + vp[4]! * wy + vp[8]! * wz + vp[12]!;
            const cy = vp[1]! * wx + vp[5]! * wy + vp[9]! * wz + vp[13]!;
            return [((cx / cw) * 0.5 + 0.5) * w, (0.5 - (cy / cw) * 0.5) * h];
        };
        const stroke = (poly: readonly [number, number][], colour: string, closing: "solid" | "dashed", dots: boolean): void => {
            const pts = poly.map(([x, z]) => toScreen(x, z)).filter((p): p is [number, number] => p != null);
            if (pts.length === 0) {
                return;
            }
            g.strokeStyle = colour;
            g.lineWidth = 2;
            g.setLineDash([]);
            g.beginPath();
            g.moveTo(pts[0]![0], pts[0]![1]);
            for (const p of pts.slice(1)) {
                g.lineTo(p[0], p[1]);
            }
            g.stroke();
            // The outline is IMPLICITLY closed, so always draw the last→first edge. Dashed on
            // the outline being authored: it shows the shape really is closed (no need to click
            // a duplicate vertex onto the start point) while still reading as the edge the tool
            // adds for you rather than one you placed.
            if (pts.length > 2) {
                g.beginPath();
                g.moveTo(pts[pts.length - 1]![0], pts[pts.length - 1]![1]);
                g.lineTo(pts[0]![0], pts[0]![1]);
                if (closing === "dashed") {
                    g.setLineDash([6, 5]);
                }
                g.stroke();
                g.setLineDash([]);
            }
            if (!dots) {
                return;
            }
            g.fillStyle = colour;
            pts.forEach((p, i) => {
                g.beginPath();
                g.arc(p[0], p[1], i === 0 ? 6 : 4, 0, Math.PI * 2);
                g.fill();
            });
        };
        for (const poly of referencePolygons()) {
            stroke(poly, "rgba(120,200,255,0.45)", "solid", false);
        }
        authorPolys.forEach((poly, i) => {
            stroke(poly, i === authorSlot ? "#ffd400" : "rgba(255,212,0,0.4)", i === authorSlot ? "dashed" : "solid", i === authorSlot);
        });
        const hint = `Outline ${authorSlot + 1}/${authorPolys.length} · ${authorPolys[authorSlot]?.length ?? 0} pts (auto-closes) · LMB add · Backspace undo · X clear · C copy`;
        g.font = "13px system-ui, sans-serif";
        // Sit it low-left, clear of the stats overlay, on a dark pill so it stays readable
        // against both the white water and the dark rock.
        const tw = g.measureText(hint).width;
        const ty = h - 64;
        g.fillStyle = "rgba(0,0,0,0.55)";
        g.fillRect(8, ty - 16, tw + 16, 24);
        g.fillStyle = "#fff";
        g.fillText(hint, 16, ty);
    };

    const setAuthoring = (on: boolean): void => {
        authoring = on;
        if (on) {
            savedBeta = ctx.camera.beta;
            ctx.camera.beta = 0.16; // near-overhead: outlines are drawn in plan view
        } else {
            if (savedBeta > 0) {
                ctx.camera.beta = savedBeta;
            }
            overlay?.remove();
            overlay = null;
        }
    };

    return {
        key: "waterfall",
        label: "Waterfall",
        envUrl: WATERFALL_ENV_URL,
        envKey: "belfast",
        // Turns the sunset round so the sun sits behind/right of the rock as the preset camera
        // frames it, which is what puts the lit rim on the falling water.
        envRotationDeg: 297,
        // The scene is authored around PB-MPM at the high tier — that pairing is what the
        // terraces, emitter charge and bloom were tuned against — so open on it. First visit
        // only: after that the method/quality dropdowns are the user's.
        defaultMethod: "PB-MPM",
        defaultQuality: "high",
        sdf,
        writeSdfParams,
        spawn() {
            // Seed the water AS A CHARGE STANDING IN THE SOURCES, not as a pool on the ground:
            // a column of water fills each outline and collapses the instant the sim starts,
            // bursting off the summit and cascading down. Every particle is live on frame 1, so
            // this needs no warm-up ramp and behaves the same on all three backends (PB-MPM
            // seeds everything immediately and cannot ramp).
            const prisms = buildPrisms();
            if (!prisms.length) {
                // No outlines yet (height map still loading): fall back to the old rest pool.
                const pr = INTAKE_R * wf.meshScale;
                const level = FLOOR_Y + 1.6 * wf.meshScale;
                return {
                    min: [ROCK_CX - pr, FLOOR_Y, ROCK_CZ - pr] as [number, number, number],
                    max: [ROCK_CX + pr, level, ROCK_CZ + pr] as [number, number, number],
                    accept: (x: number, y: number, z: number): boolean => Math.hypot(x - ROCK_CX, z - ROCK_CZ) <= pr && y > terrainHeightWorld(x, z) + 0.08,
                    warmupFrames: WARMUP_FRAMES,
                };
            }
            const s = wf.meshScale;
            const fillH = SOURCE_FILL_H * s;
            let x0 = Infinity;
            let x1 = -Infinity;
            let z0 = Infinity;
            let z1 = -Infinity;
            let y0 = Infinity;
            let y1 = -Infinity;
            for (const p of prisms) {
                y0 = Math.min(y0, p.y0);
                y1 = Math.max(y1, p.y0 + fillH);
                for (const [x, z] of p.poly) {
                    x0 = Math.min(x0, x);
                    x1 = Math.max(x1, x);
                    z0 = Math.min(z0, z);
                    z1 = Math.max(z1, z);
                }
            }
            let area = 0;
            for (const p of prisms) {
                area += polygonArea(p.poly);
            }
            // Sit the charge ON the terrace it stands in: above the local surface (never inside
            // the rock, however uneven the terrace) and up to a flat top.
            const accept = (x: number, y: number, z: number): boolean =>
                prisms.some((p) => y <= p.y0 + fillH && y >= Math.max(p.y0, terrainHeightWorld(x, z) + SHELF_LIFT * s) && pointInPolygon(x, z, p.poly));
            return {
                min: [x0, y0, z0] as [number, number, number],
                max: [x1, y1, z1] as [number, number, number],
                accept,
                warmupFrames: WARMUP_FRAMES,
            };
        },
        emitters() {
            return buildConfig();
        },
        onEnter(): void {
            active = true;
            for (const m of meshes) {
                setMeshVisible(m, containerVisible);
            }
            setMeshVisible(ctx.ground, false); // presented against open sky — no ground plane at all
            applyBloom(); // the presentation stage is shared — claim it while we are on screen
            ctx.camera.target.x = ROCK_CX;
            ctx.camera.target.y = 3.2 * wf.meshScale;
            ctx.camera.target.z = ROCK_CZ;
        },
        onLeave(): void {
            active = false;
            if (authoring) {
                setAuthoring(false); // drop the overlay canvas with the demo
            }
            for (const m of meshes) {
                setMeshVisible(m, false);
            }
            setMeshVisible(ctx.ground, true);
            ctx.setBloom({ enabled: false, intensity: wf.bloomIntensity, threshold: wf.bloomThreshold });
            ctx.camera.target.x = 0;
            ctx.camera.target.y = 6;
            ctx.camera.target.z = 0;
        },
        /** Bloom's on/off flag rides in `demoState` rather than `demoParams`, because the pair
         *  snapshot keeps only NUMERIC demoParams — a boolean there is silently dropped, which
         *  is why exported presets used to come back with bloom off and its two sliders looking
         *  inert. The intensity/threshold numbers round-trip through demoParams as usual. */
        snapshotState(): Record<string, number | boolean> {
            return { bloom: wf.bloom };
        },
        restoreState(state: Record<string, number | boolean>): void {
            if (typeof state.bloom === "boolean") {
                wf.bloom = state.bloom;
                applyBloom();
            }
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            for (const m of meshes) {
                setMeshVisible(m, active && v);
            }
        },
        update(): void {
            // The core zeroes the shared UBO's bytes 32..159 on every pair switch, so the
            // 16-float collision block is re-written each frame (cheap: one 64-byte write).
            writeSdfParams();
            if (authoring) {
                drawOverlay();
            }
        },
        claimsPointer(e: PointerEvent): boolean {
            // Only LMB while authoring — the other buttons keep orbiting the camera.
            return authoring && e.button === 0;
        },
        onPointerDown(e: PointerEvent): void {
            if (!authoring || e.button !== 0) {
                return;
            }
            const hit = pickTerrain(e.clientX, e.clientY);
            if (!hit) {
                return;
            }
            authorPolys[authorSlot]!.push(worldToModel(hit[0], hit[1]));
            commitPolys();
        },
        onKey(e: KeyboardEvent): void {
            if (!authoring) {
                return;
            }
            const poly = authorPolys[authorSlot]!;
            if (e.key === "Backspace" || e.key === "u") {
                poly.pop();
                commitPolys();
            } else if (e.key === "x" || e.key === "X") {
                poly.length = 0;
                commitPolys();
            } else if (e.key === "c" || e.key === "C") {
                const text = formatPolys();
                console.warn(`[waterfall] paste into waterfall.ts:\n${text}`);
                void navigator.clipboard?.writeText(text).catch(() => undefined);
            }
        },
        demoParams(): DemoParam[] {
            // `hidden` retires a knob from the panel while keeping it in the pair-state bag, so
            // the preset files still drive it and it round-trips between pairs — that is what
            // "Source speed" / "Jet spread" need (both feed the emitter launch velocity, and the
            // high presets deliberately zero them). The commented rows are different: "Mesh
            // scale" / "Rock yaw" are pinned presentation constants (MESH_SCALE /
            // ROCK_YAW_DEG, and applyParam ignores both keys), and the two authoring rows drive
            // the terrace-tracing tool, only needed when AUTHORED_POLYGONS has to be re-cut.
            return [
                { key: "sourceSpeed", label: "Source speed", type: "number", min: 0, max: 4, step: 0.05, value: wf.sourceSpeed, hidden: true },
                { key: "emitRate", label: "Recirculation rate", type: "number", min: 0.05, max: 6, step: 0.05, value: wf.emitRate },
                { key: "spread", label: "Jet spread", type: "number", min: 0, max: 2, step: 0.02, value: wf.spread, hidden: true },
                { key: "frontBias", label: "Front bias", type: "number", min: 0, max: 16, step: 0.5, value: wf.frontBias, hidden: true },
                // { key: "meshScale", label: "Mesh scale", type: "number", min: 1, max: 3, step: 0.05, value: wf.meshScale },
                // { key: "rockYaw", label: "Rock yaw", type: "number", min: 0, max: 360, step: 1, value: wf.rockYaw },
                { key: "bloom", label: "Bloom", type: "boolean", value: wf.bloom },
                { key: "bloomIntensity", label: "Bloom intensity", type: "number", min: 0, max: 2, step: 0.05, value: wf.bloomIntensity },
                { key: "bloomThreshold", label: "Bloom threshold", type: "number", min: 0, max: 2, step: 0.05, value: wf.bloomThreshold },
                // { key: "authorMode", label: "Author polygons", type: "boolean", value: authoring },
                // { key: "authorSlot", label: "Author: outline #", type: "number", min: 1, max: 4, step: 1, value: authorSlot + 1 },
            ];
        },
        getDomainScale(): number {
            // The core reads this on every switchPair to size the fluid-sim bounds. A larger
            // rock gets a proportionally larger simulated domain — otherwise a 3× formation
            // runs straight through the ±20 grid wall (see ctx.setDomainScale in fluid.ts).
            return wf.meshScale;
        },
        applyParam(key: string, value: number | boolean | string): void {
            const prevScale = wf.meshScale;
            // The rock's size and heading are pinned (MESH_SCALE / ROCK_YAW_DEG) and their
            // rows are commented out of demoParams(). Preset files written before they were
            // retired still carry them — the older ones at meshScale 1, framed for a 1x rock —
            // and restoreState replays whatever keys a stored bag holds, so ignoring them here
            // is what actually keeps the rock fixed. Drop this guard when uncommenting a row.
            if (key === "meshScale" || key === "rockYaw") {
                return;
            }
            if (key === "bloom") {
                wf.bloom = value as boolean;
                applyBloom();
                return;
            }
            if (key === "authorMode") {
                setAuthoring(value as boolean);
                return;
            }
            if (key === "authorSlot") {
                authorSlot = Math.max(0, Math.round(value as number) - 1);
                while (authorPolys.length <= authorSlot) {
                    authorPolys.push([]);
                }
                return;
            }
            (wf as unknown as Record<string, number>)[key] = value as number;
            if (key === "bloomIntensity" || key === "bloomThreshold") {
                applyBloom();
                return;
            }
            if (key === "rockYaw" || key === "meshScale") {
                // Both re-place the model AND the collision height field it is sampled
                // through, so the mesh and the surface the water rides never diverge.
                cosYaw = Math.cos((wf.rockYaw * Math.PI) / 180);
                sinYaw = Math.sin((wf.rockYaw * Math.PI) / 180);
                applyScaleTransforms();
                writeSdfParams();
            }
            if (key === "meshScale") {
                if (active && prevScale > 0) {
                    // Re-frame: raise the orbit target AND pull the camera back by the same
                    // factor the model grew by. Framing is then scale-invariant (a 3× rock
                    // fills the viewport exactly like a 1× one) and whatever zoom the user had
                    // dialled in is preserved RELATIVE to the model instead of leaving them
                    // buried inside the rock. The camera has no radius limits, and 3 × the
                    // preset radius (30 → 90) still sits well inside the 200-unit far plane.
                    ctx.camera.target.y = 3.2 * wf.meshScale;
                    ctx.camera.radius *= wf.meshScale / prevScale;
                }
                // DEBOUNCE the heavy path: ctx.setDomainScale disposes and recreates all three
                // backends (and re-applies the SDF, emitters and spawn), so a slider drag must
                // not do it every tick. Fires ~250 ms after the last change; skipped entirely
                // when the scale is already the one the sims were built at (e.g. a pair-state
                // restore re-applying the same value).
                if (wf.meshScale !== builtMeshScale) {
                    if (meshScaleTimer !== null) {
                        clearTimeout(meshScaleTimer);
                    }
                    meshScaleTimer = setTimeout(() => {
                        meshScaleTimer = null;
                        builtMeshScale = wf.meshScale;
                        if (active) {
                            ctx.setDomainScale(wf.meshScale);
                        }
                    }, 250);
                } else if (meshScaleTimer !== null) {
                    clearTimeout(meshScaleTimer);
                    meshScaleTimer = null;
                }
            }
            ctx.refreshEmitters();
        },
        extraControls() {
            return [];
        },
    };
}
