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

import { addToScene, createDisc, createMeshFromData, createPbrMaterial, enableMirroredMeshes, loadGltf, setMeshVisible } from "babylon-lite";
import type { FluidEmitter, FluidFlowConfig, Mesh, SceneNode } from "babylon-lite";
import type { SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import type { DemoParam, FluidCtx, FluidDemo, DemoStateValue } from "../demo.js";
import { configureDemoDecoderBases, demoAssetUrl } from "../../demo-asset-url.js";
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
//
// rock.glb also carries a BAKED AO map, packed into the red channel of its existing
// metallicRoughness texture with `occlusionTexture` pointing at the same image (see OASIS_URLS
// for why that packing is free). Its effect is deliberately subtler than the oasis's: occlusion
// only modulates the diffuse IBL term, and this formation is convex and largely sky-exposed, so
// even a fully-occluded test map moved its mean luminance by only ~16%. Crucially, adding it did
// NOT touch geometry — the mesh hashes identical to the pre-AO file, so rock-heightmap.bin
// stays valid and the fluid still rides the surface being drawn.
//
// oasis.glb beside it is SET DRESSING ONLY — see OASIS_* below. It is never baked into the
// collision field, so it needs no height map of its own.
/** Open-sky HDR environment: Poly Haven "industrial_sunset_02_puresky" (CC0), 2k Radiance
 *  panorama. Declared HERE, next to the demo's other assets, so it is resolved against the same
 *  module URL as the rock and the height map — and served out of the same `waterfall/` folder the
 *  demo-bundle pipeline stages. The core reads it as this demo's `envUrl`: it is the
 *  waterfall's skybox background, its fluid-surface reflections AND its terrain IBL. Being a
 *  raw `.hdr` (not a pre-filtered `.env`) it is loaded via `loadHdrEnvironment`.
 *
 *  Kept as a LOCAL file rather than the Poly Haven CDN URL the other picker entries use: this
 *  one is the demo's default backdrop, so it has to be there on first paint without depending
 *  on an external host.
 *
 *  Its sun is a small, intense disc (6 texels at 2k, peak radiance 2729) rather than the broad
 *  hazy glow of the previous default — which is what lets the optional shadow generator throw a
 *  crisp, believable shadow instead of a vague smudge. */
export const WATERFALL_ENV_URL = demoAssetUrl("./waterfall/industrial_sunset_02_puresky_2k.hdr", import.meta.url);
/** The previous default (Poly Haven "belfast_sunset_puresky", CC0), still offered in the
 *  environment picker and still staged with the demo. Its sun sits at essentially the same
 *  azimuth as the industrial map (215.8° vs 215.9°, both ~2.5° above the horizon), which is why
 *  swapping the default needed no change to `envRotationDeg`. */
export const WATERFALL_BELFAST_ENV_URL = demoAssetUrl("./waterfall/belfast_sunset_puresky_2k.hdr", import.meta.url);

/** Yaw applied to the environment (skybox + IBL + fluid reflections) for this demo, degrees.
 *  Turns the sunset round so the sun sits to the right of the rock as the preset camera frames
 *  it, which is what puts the lit rim on the falling water. Also feeds the shadow sun's default
 *  bearing — the sun disc turns with the map, so the two must be derived from the same yaw. */
const ENV_ROTATION_DEG = 297;

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

// ── Background scenery: the oasis ring ─────────────────────────────────────────────────
/** The oasis island used as the surrounding shoreline, in three detail tiers. Visual only — no
 *  height map, no collision. Each is the TILED variant: the source ring's hole is far too small to
 *  seat the rock once the island is scaled to a believable size — at OASIS_SCALE the pond needed is
 *  actually wider than the island's whole outer radius. Simply pushing the mesh outward grows the
 *  hole but smears the vegetation tangentially by ~3x, and no radial remap can avoid that (only a
 *  uniform scale preserves shape, and that resizes the features we are trying to keep). So the ring
 *  is compressed into half the circle, pushed out, and stamped TWICE — seamlessly, because a closed
 *  ring's start and end are the same cross-section. Residual tangential scale is 1.87x at the bare
 *  inner beach, 1.10x through the vegetated mid band and 0.88x at the outer skirt.
 *
 *  Tiling DOUBLES the triangle count, so the in-scene figures below are twice their source
 *  decimation. Regenerate with the tile-ring asset script if OASIS_SCALE changes.
 *
 *  The tiers exist because this is pure set dressing whose cost is worth choosing: measured against
 *  the 2 M-triangle source, mean surface deviation is 0.045% at 40 k, 0.035% at 160 k and 0.026%
 *  at 500 k — the geometry converges well below the source's own density.
 *
 *  High is deliberately NOT the full 2 M source. That version was built and rejected: at 3.89 M
 *  triangles it renders WORSE here, not better. The fluid demo runs with `msaaSamples: 1` and
 *  every render target at `samples: 1` (the engine default is 4 — this demo opted out), so there
 *  is no anti-aliasing at all; once leaf blades fall below a pixel their silhouettes turn into
 *  white speckle against the bright sky. Measured in isolation, 4x MSAA removes 92% of that
 *  speckle at 3.89 M versus 55% at 160 k, which is the tell: the artifact is sub-pixel geometry,
 *  not the mesh. The 1 M tier keeps essentially all the detail (0.026% deviation) while cutting
 *  the sub-pixel edge count, and is a quarter of the download.
 *
 *  All three are Draco-compressed, and all three carry a BAKED AMBIENT-OCCLUSION map. The demo
 *  has no shadows at all (they were removed), so without AO the foliage renders flat under pure
 *  IBL — leaves in a crown all receive the same sky light and lose their separation. The AO is
 *  baked in Blender (Cycles) into the model's EXISTING UV atlas and packed into the RED channel
 *  of the metallicRoughness texture, with `occlusionTexture` pointing at that same image: glTF's
 *  ORM convention leaves R unused, so this costs no extra texture, no extra sampler, and avoids
 *  the loader's ORM-composite path (which only runs when the two slots use different images).
 *
 *  Two things the bake depends on. It must run on the UNTILED ring — the tiers stamp the ring
 *  twice onto the same UVs, so baking on a tiled mesh would average two unrelated neighbourhoods.
 *  And because all tiers are decimated from one source they share the atlas, so a single bake
 *  (taken from the 500 k mesh) serves all three — which also means the 40 k tier inherits
 *  occlusion detail its own geometry could never carry.
 *
 *  Note that babylon-lite hardcodes occlusionStrength to 1.0 whenever an occlusion image exists
 *  and ignores glTF's occlusionTexture.strength, so AO strength must be baked into the pixels.
 *
 *  The loader picks the Draco decoder up automatically from `KHR_draco_mesh_compression`;
 *  `configureDemoDecoderBases` points it at the demo-local decoder instead of the site root, so
 *  the built bundle works under any base path. */
/** Detail tier for the background island. */
type OasisQuality = "low" | "mid" | "high";
const OASIS_URLS: Record<OasisQuality, string> = {
    low: demoAssetUrl("./waterfall/oasis-low.glb", import.meta.url), // 80 k tris, 1.8 MB
    mid: demoAssetUrl("./waterfall/oasis-mid.glb", import.meta.url), // 160 k tris, 2.1 MB
    high: demoAssetUrl("./waterfall/oasis-high.glb", import.meta.url), // 1 M tris, 4.5 MB
};
const OASIS_QUALITY_LABELS: [OasisQuality, string][] = [
    ["low", "Low (80k tris)"],
    ["mid", "Middle (160k tris)"],
    ["high", "High (1M tris)"],
];
const OASIS_DEFAULT_QUALITY: OasisQuality = "mid";
/** Model→world scale. Inner radius ≈ 0.347 model units, so this leaves a hole of ≈ ±35 world —
 *  wide enough to sit the rock (±26.5) inside the ring with clearance. Deliberately oversized
 *  for now: at this scale the outer edge reaches ≈ ±90, so the preset camera (radius 63) ends up
 *  INSIDE the ring's body and will clip through it. */
const OASIS_SCALE = 30;
/** Offset of the island's centre from the rock, along the view axis (0 = concentric). */
const OASIS_DIST = 0;
/** Lateral offset, perpendicular to the view axis (0 = concentric). */
const OASIS_SIDE = 0;
/** Lowest vertex of the model, in model units — used to seat the island on FLOOR_Y. */
const OASIS_MIN_Y = -0.1733;
/** Yaw about Y. Turns which side of the ring faces the camera. */
const OASIS_YAW_DEG = 180;
/** World-space nudge for the island, on top of DIST/SIDE, to seat the waterfall in the middle of
 *  the pond. The rock sits on its own origin (rock.glb's node carries a translation that cancels
 *  its mesh's bbox offset), so it is centred at ROCK_CX/ROCK_CZ = (0, 0). The ring is not: its
 *  centre is at model (-0.043, 0.031), which at OASIS_YAW_DEG and OASIS_SCALE lands near
 *  (1.3, -0.9) in world XZ. Cancelling that is the whole correction.
 *
 *  The ISLAND moves rather than the rock deliberately: ROCK_CX/ROCK_CZ also anchor the collision
 *  field, the pump intake, the spawn disc and the camera target, and the fluid domain is fixed
 *  about the world origin — so sliding the rock would push the water toward the domain wall.
 *  The island has no simulation coupling at all. */
const OASIS_OFFSET_X = -1.3;
const OASIS_OFFSET_Z = 0.9;
/** Radius of the shadow-catcher disc over the pond, world units, measured off the model. The
 *  ring's hole is NOT circular: the closest geometry to the centre sits at 29.0 world but the
 *  hole opens out to 36.8 in other directions, so a disc sized to the minimum left the water
 *  beyond it catching nothing and cut every shadow off along a visible circle. The island's
 *  OUTER silhouette is the real bound — its nearest point is 41.4 world — so 40 covers the whole
 *  pond while still ending underneath the shore in every direction, never poking out past the
 *  island where a shadow would hang in mid-air. (OASIS_OFFSET_* re-centres the ring on the rock,
 *  so the hole is concentric with ROCK_CX/ROCK_CZ.) */
const POND_RADIUS = 40;

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
// Seat the model's base just BELOW the floor so the rock rises out of the ground rather
// than resting on it — at the silhouette the height map falls to the model's own floor, and
// max(rock, FLOOR_Y) then hands the ground back to the quad with no seam or z-fight.
const ROCK_Y0 = FLOOR_Y - 0.05;

// Start-of-sim charge: initial emitters seed a shallow head of water standing in each
// source outline while inflows reserve the remaining particle capacity for recycling.
const SOURCE_FILL_H = 2.0;

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
    let rockReady = false;
    let heightMapReady = false;
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

    const buildFlow = (): FluidFlowConfig => {
        const vx = FRONT_DIR_X * wf.frontBias;
        const vz = FRONT_DIR_Z * wf.frontBias;
        const vy = -wf.sourceSpeed;
        const prisms = buildPrisms();
        const initialTerraces: FluidEmitter[] = prisms.map((p, index) => ({
            id: `waterfall-charge-${index + 1}`,
            name: `Initial terrace ${index + 1}`,
            enabled: true,
            behavior: "initial",
            transform: { position: [0, p.y0 + SOURCE_FILL_H * wf.meshScale * 0.5, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "polygonPrism", points: p.poly, thickness: SOURCE_FILL_H * wf.meshScale },
            sampling: "volume",
            velocity: [0, 0, 0],
            velocitySpace: "world",
            spread: 0,
        }));
        const inflows: FluidEmitter[] = prisms.map((p, index) => ({
            id: `waterfall-spring-${index + 1}`,
            name: `Terrace spring ${index + 1}`,
            enabled: true,
            behavior: "inflow",
            transform: { position: [0, (p.y0 + p.y1) * 0.5, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "polygonPrism", points: p.poly, thickness: p.y1 - p.y0 },
            sampling: "volume",
            velocity: [vx, vy, vz],
            velocitySpace: "world",
            spread: wf.spread,
        }));
        const pr = ctx.simHalfExtentXZ;
        return {
            emitters: [...initialTerraces, ...inflows],
            initialEmittersFillCapacity: true,
            sinks: [
                {
                    id: "waterfall-floor-recycle",
                    name: "Waterfall floor return",
                    enabled: true,
                    transform: { position: [ROCK_CX, FLOOR_Y - 0.35, ROCK_CZ], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                    shape: { type: "box", size: [pr * 2, 1.3, pr * 2] },
                    targets: inflows.map((emitter) => emitter.id),
                    perParticleRecycleRate: wf.emitRate,
                },
            ],
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
    // The witness also has to opt into shadow RECEIVING, for the same reason it exists at all:
    // the PBR group's renderables are compiled from the meshes present when the scene is built,
    // and a mesh that receives no shadows compiles no shadow-sampling path. The rock arrives
    // later and joins this already-built group, so if the witness did not ask for the receiver
    // variant here, the rock could never sample the shadow map and the "Sun shadows" toggle
    // would render a map that nothing reads.
    pbrGroupWitness.receiveShadows = true;
    addToScene(ctx.scene, pbrGroupWitness);
    setMeshVisible(pbrGroupWitness, false);

    // ── The shadow catcher over the pond ───────────────────────────────────────────────
    // Everything the rock and the vegetation throw would otherwise land on nothing: the demo
    // draws no ground, and the water is the FLUID, whose surface is reconstructed in a
    // screen-space pass that samples no shadow map. So the only surfaces that could show a
    // shadow were the rock and the ring itself, and both are steep enough that the shadow
    // mostly hides in geometry the sun already leaves dark.
    //
    // A `shadowOnly` PBR material fixes that without introducing a visible ground: the surface
    // is fully transparent wherever it is lit and only tints where a shadow falls, so it reads
    // as a shadow ON the water rather than as a plate floating in it.
    //
    // Created SYNCHRONOUSLY here, not lazily on first enable: `shadowOnly` implies alpha
    // blending, so this disc opens its own transparent material group, and a group that is not
    // present when the scene is built never gets a renderable (the same trap `pbrGroupWitness`
    // above exists to work around). `createWaterfallDemo` runs before `registerScene`, so
    // building it now is enough — it is simply kept hidden until the toggle asks for it.
    //
    // It sits AT the floor, i.e. on the bed of the pool rather than on top of it, which is also
    // where it belongs visually: the fluid composites over the scene colour with refraction, so
    // the shadow is seen THROUGH the water.
    const shadowCatcher = createDisc(engine, { radius: POND_RADIUS, tessellation: 96 });
    // createDisc builds in the XY plane with its normals and winding facing -Z, so +90° about X
    // is what lays it flat FACE UP. (-90° also lays it flat, but back-facing — it is then culled
    // and draws nothing at all.)
    shadowCatcher.rotation.set(Math.PI / 2, 0, 0);
    // A HAIR below the floor. The disc now reaches out under the shore, and the island's own
    // lowest vertices sit exactly at FLOOR_Y — coplanar with the disc, the depth test would let
    // it paint shadow over the shoreline as well as the water. Dropping it 2 cm puts it behind
    // the terrain everywhere the terrain exists, so it only ever shows in the open pond.
    shadowCatcher.position.set(ROCK_CX, FLOOR_Y - 0.02, ROCK_CZ);
    shadowCatcher.material = createPbrMaterial({
        shadowOnly: true,
        shadowOnlyColor: [0, 0, 0],
        // Well under 1: this is a shadow on WATER, and a fully opaque black reads as a hole
        // punched in the pond rather than as shade.
        shadowOnlyOpacity: 0.45,
        shadowOnlyFalloff: 1,
    });
    shadowCatcher.receiveShadows = true; // required by shadowOnly — it IS the shadow term
    addToScene(ctx.scene, shadowCatcher);
    setMeshVisible(shadowCatcher, false);

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

    // ── Optional sun shadows ───────────────────────────────────────────────────────────
    // Off by default: the scene is authored around pure IBL plus the baked AO, and shadows are
    // an opt-in extra. The core keeps the generator permanently attached and toggles the caster
    // list (see fluid.ts), so "off" costs nothing after the first frame.
    //
    // The sun is aimed manually rather than read back from the sky. Its default bearing is
    // measured against the rendered frame (see `sunAzimuth`) so the shadows fall away from the
    // glare you can actually see in the backdrop, and the ELEVATION is deliberately raised well
    // above the map's real ~2.5°: a sunset that low throws shadows several rock-lengths long that
    // leave the frame entirely and read as a flat dark wash rather than as shape. At the default
    // the sun sits off to the right of the camera, which is where a side-light reads best — the
    // lit and shadowed halves of the formation split along its silhouette instead of hiding
    // behind it.
    //
    // Note the shadow needs somewhere to LAND. The demo draws no ground and the water is the
    // fluid, whose screen-space surface pass reads no shadow map — so the rock and the ring are
    // the only real receivers, and both are steep enough to hide most of it. The `shadowCatcher`
    // disc above is what makes the effect read: a transparent-until-shadowed plate over the pond
    // that shows the rock's and the palms' shadows on the water.
    let shadowsOn = false;
    /** Compass bearing of the sun, degrees.
     *
     *  MEASURED against the rendered frame, and worth reading before touching: this value was
     *  changed to 171 at one point and then changed back, because every measurement of it was
     *  being taken through the inverted-normal bug described in the rock loader below. With the
     *  shading normals flipped, "which way is the formation lit" reads exactly 180° out, so the
     *  camera's forward bearing measured as 240° instead of 60°. Once the winding fix landed the
     *  same sweep returned 60°, and all three independent checks agree again:
     *
     *    1. `scratch/wf-calib.mts` sweeps this value with the environment and the ambient fill at
     *       zero, so the sun is the only light, and records the formation's brightness. Clean
     *       sinusoid: darkest at 60° (backlit — the sun is straight ahead of the camera), brightest
     *       at 230°. So the camera's forward bearing is 60°, as a fact about pixels.
     *    2. `scratch/wf-discmove.mts` tracks the sun DISC — the only strongly saturated yellow
     *       thing in the sky, and therefore independent of any shading. It sits at screen centre at
     *       environment rotation 6° (x = 789 of 1600) and moves LEFT as the rotation rises (555 at
     *       16°, 304 at 26°). This demo's ENV_ROTATION_DEG is 297, i.e. 69° short of a full turn
     *       back to 6°, so the disc sits 69° to the RIGHT — measured at x = 1528, hard against the
     *       right edge, with the sky's glare brighter in the right third of the frame (248 vs 210).
     *    3. `scratch/wf-crosscheck.mts` confirms the sign: the RIGHT flank is lit most at azimuth
     *       300 and the LEFT flank at 120, so azimuths BELOW the 60° forward bearing put the sun on
     *       the right. 60 - 69 = 351 is therefore the right-hand answer; 129 would be the left.
     *
     *  Whole degrees because the azimuth slider steps in whole degrees — UI and value must agree. */
    let sunAzimuth = 351;
    /** Height of the sun above the horizon, degrees. */
    let sunElevation = 34;
    /** The core's own sun strength and ambient fill, captured before the demo touches them, so
     *  leaving the demo hands the shared lights back exactly as they were found. */
    const BASE_SUN_INTENSITY = ctx.sun.intensity;
    const BASE_AMBIENT_INTENSITY = ctx.ambient.intensity;
    /** Sun strength while this demo is on screen.
     *
     *  A shadow can only subtract the light it blocks, so how legible one is comes down to what
     *  share of a surface's brightness the sun actually owns — and under this HDR sky, at the
     *  core's 2.4, that share is a couple of percent. Removing all of it barely dents the pixel,
     *  which is exactly why the shadows read as smudges. Measured shadow depth, as a fraction of
     *  local brightness (ring foliage / rock / palm):
     *
     *      sun  2.4, fill 0.75   →   5% /  2% /  1%      (the old default: invisible)
     *      sun 15,   fill 0.20   →  19% /  8% /  5%
     *      sun 30,   fill 0.10   →  ~25% / 10% /  7%     (this default)
     *      sun 60,   fill 0.10   →  31% / 16% / 10%      (diminishing, and the lit side blows out)
     *
     *  The lit side barely changes across that range because the ambient fill comes down as the
     *  sun goes up, so this buys contrast rather than exposure. */
    let sunIntensity = 30;
    /** Hemispheric fill while this demo is on screen.
     *
     *  The core's 0.75 exists for the standard-material demos, which sample no environment map.
     *  This one is PBR under a full HDR IBL, so it already has all the ambient it needs and the
     *  hemispheric is a SECOND ambient term stacked on top — one that no shadow can attenuate, so
     *  it acts as a flat floor that shadows can never dig below. Measured: with the fill at the
     *  core's 0.75 the whole IBL only accounts for ~35 of ~105 luminance on the rock's shaded
     *  flank, the rest being that unshadowable floor. Dropping it is the single biggest thing
     *  that makes shadows visible here, and it costs nothing in look because the sky is already
     *  doing the job. */
    let ambientIntensity = 0.1;
    const applySunDirection = (): void => {
        const az = (sunAzimuth * Math.PI) / 180;
        const el = (sunElevation * Math.PI) / 180;
        // Light DIRECTION points from the sun toward the scene, hence the negations.
        const dx = -Math.cos(el) * Math.cos(az);
        const dy = -Math.sin(el);
        const dz = -Math.cos(el) * Math.sin(az);
        // The boosted sun and the dimmed fill belong to this demo, not to the shared lights: hand
        // both back at the core's own values whenever the waterfall is not the demo on screen.
        ctx.sun.intensity = active ? sunIntensity : BASE_SUN_INTENSITY;
        ctx.ambient.intensity = active ? ambientIntensity : BASE_AMBIENT_INTENSITY;
        // `intensity` is a plain field, so nothing observes it — but ObservableVec3.set always
        // fires its dirty callback (no equality check), so setting a direction is what bumps the
        // light version and gets the new intensity into the lights UBO. Keep each intensity write
        // paired with one, or it is a silent no-op.
        ctx.ambient.direction.set(0.3, 1, 0.4);
        ctx.sun.direction.set(dx, dy, dz);
        // The shadow generator builds its light camera AT `sun.position` and measures its ortho
        // depth range from there — the caster fit only sizes X/Y. That range is SYMMETRIC
        // (±250, see fluid.ts), so the light belongs IN the middle of the scene rather than stood
        // off outside it: park it on the formation's own centre and the whole island, ring
        // included, sits comfortably inside the slice at every angle the sliders allow.
        const aimY = ROCK_Y0 + 6;
        ctx.sun.position.set(ROCK_CX, aimY, ROCK_CZ);
    };
    /** Everything that casts into the shadow map: the rock plus whichever oasis tier is
     *  currently shown. The ring is included deliberately — its palms are what make the pond
     *  catcher worth having — and it is affordable: the ortho box is fit to the casters, and the
     *  ring is only ~100 world units across, so a 2048² map still spends ~20 texels per world
     *  unit and leaves the rock several hundred texels wide. */
    const shadowCasterMeshes = (): Mesh[] => (containerVisible ? [...meshes, ...sceneryMeshes] : []);
    const applyShadows = (): void => {
        const on = shadowsOn && active;
        ctx.setSunShadows(on, shadowCasterMeshes());
        // The catcher is transparent wherever it is lit, so leaving it visible with shadows off
        // would be harmless — but it is a full-pond alpha-blended draw, so hide it instead.
        setMeshVisible(shadowCatcher, on);
        applySunDirection(); // the sun's STRENGTH depends on this toggle too, not just its angle
    };
    applySunDirection();

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
        rockRoot = asset.entities[0] as SceneNode;
        const mine: Mesh[] = [];
        collectMeshes(rockRoot, mine);
        // MUST be set before addToScene: `receiveShadows` is folded into the material variant
        // when the renderable is built, so flipping it afterwards would never reach the shader.
        for (const m of mine) {
            m.receiveShadows = true;
        }
        // Both of these MUST also happen before addToScene, for the same reason — a mesh's triangle
        // winding is baked into its GPU pipeline when the renderable is built.
        //
        // `applyScaleTransforms` OVERWRITES the asset root's scaling, and that root is where the
        // loader parks its right-handed→left-handed flip (`__root__`, a diag(-1,1,1)). Replacing it
        // with a plain positive uniform scale drops the flip, so the model's world determinant comes
        // out POSITIVE (measured +157464 = 54³) while the loader recorded `_authoredSign = -1` for
        // it at load time. The mesh is therefore MIRRORED relative to the winding its geometry was
        // authored for, and its triangles are wound the opposite way round.
        //
        // Dropping the flip is DELIBERATE, not an oversight: `rock-heightmap.bin` is rasterised in
        // raw glTF space (`bake-rock-heightmap.ts` walks the node tree from IDENTITY), so the water
        // only collides with the silhouette it is drawn against while the rock is drawn unflipped
        // too. Restoring the flip would mirror the formation against its own collision field.
        //
        // So the mirroring stays and the winding must be corrected instead. `enableMirroredMeshes`
        // is the engine's opt-in for exactly this: it resolves winding from the LIVE world
        // determinant rather than the loader's one-shot load-time flag, which is what a mesh
        // mirrored AFTER load needs. It must be installed, and the transform must be final, before
        // the renderable is built — the per-scene watcher only rebuilds on a subsequent sign FLIP,
        // and seeds a late-added mesh with whatever sign it was built from.
        //
        // Getting this wrong is not subtle. These materials used to ship `doubleSided` (stripped
        // from the assets by `scripts/strip-double-sided.ts`), and the two-sided PBR path flips the
        // shading normal on `!front_facing` — which WebGPU derives from the pipeline's `frontFace`.
        // Left at the default "ccw" that test came out inverted on the VISIBLE OUTER surface, so
        // every lit face got `N = -N` and the formation shaded as though the sun were underneath it:
        // up-facing surfaces went black with the sun overhead, and shadows disagreed with the
        // lighting by 180°. Now that the materials are single-sided the failure mode is different
        // but no milder — back-face culling is live, so the wrong winding would cull the outer
        // surface and render the rock inside-out.
        await enableMirroredMeshes(ctx.scene);
        applyScaleTransforms();
        addToScene(ctx.scene, asset);
        for (const m of mine) {
            setMeshVisible(m, active && containerVisible);
            meshes.push(m);
        }
        rockReady = true;
        applyShadows(); // the casters only exist now
    })().catch((e: unknown) => console.warn("[waterfall] rock model load failed", e));

    // ── Background scenery: the oasis ring ─────────────────────────────────────────────
    // SET DRESSING ONLY. It is never rasterised into the height map and takes no part in the
    // collision SDF, so the fluid ignores it completely — which is why it needs no bake of its
    // own and can be moved or resized freely without re-deriving anything.
    //
    // It is placed CONCENTRIC with the rock, so the waterfall stands inside the ring's hole.
    //
    // Three detail tiers, switched from the panel and loaded LAZILY: high is 108 MB, so it is only
    // fetched if actually chosen. Each tier is cached once loaded, so switching back is instant,
    // and only the selected tier is ever visible.
    /** Meshes of the CURRENTLY SHOWN tier; onEnter/onLeave drive visibility through this. */
    const sceneryMeshes: Mesh[] = [];
    const oasisCache = new Map<OasisQuality, Mesh[]>();
    let oasisQuality: OasisQuality = OASIS_DEFAULT_QUALITY;
    let oasisPending: OasisQuality | null = null;
    /** Notified when a tier finishes (or fails) loading, so the panel can re-enable its select. */
    let onOasisLoaded: (() => void) | null = null;

    const showOasisTier = (q: OasisQuality): void => {
        for (const m of sceneryMeshes) {
            setMeshVisible(m, false);
        }
        sceneryMeshes.length = 0;
        const cached = oasisCache.get(q);
        if (cached) {
            sceneryMeshes.push(...cached);
            for (const m of sceneryMeshes) {
                setMeshVisible(m, active && containerVisible);
            }
            applyShadows(); // the caster set is per-tier
            return;
        }
        if (oasisPending === q) {
            return; // already in flight
        }
        // No tier is shown until the load lands, so drop the previous one's meshes from the
        // caster list rather than keep casting from geometry that is no longer drawn.
        applyShadows();
        oasisPending = q;
        void (async (): Promise<void> => {
            // The oasis GLBs are Draco-compressed; aim the decoder at the demo's own directory so
            // a bundle deployed under any base path still finds it. Idempotent and cheap on
            // repeat calls — the dynamic imports behind it are module-cached.
            await configureDemoDecoderBases(import.meta.url);
            const asset = await loadGltf(engine, OASIS_URLS[q]);
            asset.animationGroups = undefined;
            const root = asset.entities[0] as SceneNode;
            const mine: Mesh[] = [];
            collectMeshes(root, mine);
            // The oasis both CASTS and RECEIVES. Casting is the point of the pond catcher — the
            // palms are what put a recognisable shadow on the water. Receiving costs no extra
            // shadow-map area (the ortho box is fit to the casters, and a caster's shadow can
            // only land inside its own light-space footprint) and keeps the ring from being the
            // one surface in the scene that a shadow passes straight through.
            // MUST be set before addToScene — `receiveShadows` is baked into the material variant
            // when the renderable is built.
            for (const m of mine) {
                m.receiveShadows = true;
            }
            // Seat the model's own base on the floor: `position` places the ORIGIN, and this
            // model's lowest vertex sits below it, so lift by that much.
            //
            // The transform, and the mirrored-mesh winding rule, MUST both precede addToScene —
            // see the rock's loader above for why. Overwriting this root's scaling drops the
            // loader's RH→LH flip, leaving the ring mirrored (world determinant +27000 = 30³)
            // against an `_authoredSign` of -1; with a `doubleSided` glTF material that inverted
            // the shading normal on every visible face.
            root.position.set(
                FRONT_DIR_X * -OASIS_DIST - FRONT_DIR_Z * OASIS_SIDE + OASIS_OFFSET_X,
                FLOOR_Y - OASIS_MIN_Y * OASIS_SCALE,
                FRONT_DIR_Z * -OASIS_DIST + FRONT_DIR_X * OASIS_SIDE + OASIS_OFFSET_Z
            );
            root.scaling.set(OASIS_SCALE, OASIS_SCALE, OASIS_SCALE);
            root.rotation.set(0, (OASIS_YAW_DEG * Math.PI) / 180, 0);
            await enableMirroredMeshes(ctx.scene);
            addToScene(ctx.scene, asset);
            oasisCache.set(q, mine);
            // A slow tier can land after the user has already picked a different one — only show
            // it if it is still the selection.
            for (const m of mine) {
                setMeshVisible(m, false);
            }
            if (oasisQuality === q) {
                sceneryMeshes.length = 0;
                sceneryMeshes.push(...mine);
                for (const m of mine) {
                    setMeshVisible(m, active && containerVisible);
                }
                applyShadows(); // this tier's meshes only exist now
            }
        })()
            .catch((e: unknown) => console.warn(`[waterfall] oasis scenery (${q}) load failed`, e))
            .finally(() => {
                oasisPending = null;
                onOasisLoaded?.();
            });
    };
    showOasisTier(oasisQuality);

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
        heightMapReady = true;

        if (active) {
            writeSdfParams();
            ctx.refreshFlow();
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
        ctx.refreshFlow();
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

    // ── The oasis detail selector ──────────────────────────────────────────────────────
    // Tiers load lazily and the select is locked while one is in flight, because the high tier is
    // a 108 MB download — spamming the dropdown must not queue several of those at once.
    const oasisRow = document.createElement("div");
    oasisRow.style.cssText = "margin:8px 0 6px;";
    const oasisLab = document.createElement("div");
    oasisLab.textContent = "Background detail";
    oasisLab.style.cssText = "color:#9fb4cc;font-size:12px;margin-bottom:4px;";
    const oasisSel = document.createElement("select");
    oasisSel.style.cssText = "width:100%;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
    for (const [value, label] of OASIS_QUALITY_LABELS) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = label;
        oasisSel.append(opt);
    }
    oasisSel.value = oasisQuality;
    oasisSel.onchange = (): void => {
        oasisQuality = oasisSel.value as OasisQuality;
        oasisSel.disabled = true;
        showOasisTier(oasisQuality);
        // A cached tier swaps synchronously and never sets `oasisPending`, so re-enable here
        // rather than waiting on the load callback that will not come.
        if (!oasisPending) {
            oasisSel.disabled = false;
        }
    };
    onOasisLoaded = (): void => {
        oasisSel.disabled = false;
    };
    oasisRow.append(oasisLab, oasisSel);

    // ── Shadow controls ────────────────────────────────────────────────────────────────
    const shadowRow = document.createElement("label");
    shadowRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:10px 0 4px;cursor:pointer;";
    const shadowChk = document.createElement("input");
    shadowChk.type = "checkbox";
    shadowChk.checked = shadowsOn;
    const shadowTxt = document.createElement("span");
    shadowTxt.textContent = "Sun shadows";
    shadowRow.append(shadowChk, shadowTxt);

    /** Azimuth/elevation sliders, shown only while shadows are on — they do nothing otherwise
     *  (the sun's direction still lights the scene, but its contribution is dwarfed by the IBL). */
    const makeSunRow = (
        label: string,
        min: number,
        max: number,
        step: number,
        fmt: (v: number) => string,
        get: () => number,
        set: (v: number) => void
    ): { row: HTMLElement; sync: () => void } => {
        const row = document.createElement("div");
        row.style.cssText = "margin:2px 0 6px;";
        const head = document.createElement("div");
        head.style.cssText = "display:flex;justify-content:space-between;color:#9fb4cc;font-size:12px;";
        const name = document.createElement("span");
        name.textContent = label;
        const val = document.createElement("span");
        val.textContent = fmt(get());
        head.append(name, val);
        const input = document.createElement("input");
        input.type = "range";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.value = String(get());
        input.style.cssText = "width:100%;";
        input.oninput = (): void => {
            set(Number(input.value));
            val.textContent = fmt(Number(input.value));
            applySunDirection();
        };
        row.append(head, input);
        // Pull the slider back into line with the state — used when a preset restores a value
        // that the widget was not the one to change.
        return {
            row,
            sync: (): void => {
                input.value = String(get());
                val.textContent = fmt(get());
            },
        };
    };
    const degrees = (v: number): string => `${v.toFixed(0)}\u00b0`;
    const az = makeSunRow(
        "Sun azimuth",
        0,
        360,
        1,
        degrees,
        () => sunAzimuth,
        (v) => (sunAzimuth = v)
    );
    // Below ~10° the shadow stretches past the rock's own footprint and stops reading as shape.
    const el = makeSunRow(
        "Sun elevation",
        5,
        89,
        1,
        degrees,
        () => sunElevation,
        (v) => (sunElevation = v)
    );
    // How hard the sun competes with the sky. This is the shadow CONTRAST control: a shadow can
    // only subtract direct light, so the higher this is the more there is to lose.
    //
    // The floor is 0, not the core's own 2.4. It used to be 2.4 on the reasoning that below the
    // core value the shadows just vanish into the IBL, so there was nothing to see — but that made
    // the control quietly lie: dragging it fully left still left the sun at FULL core strength,
    // reading "2.4×" rather than off. With the environment intensity and the ambient fill both
    // able to reach 0, that left no way to darken the scene completely, and a fully-lit render
    // looked like the other two sliders were broken. At a true 0 the models go to black
    // silhouettes, which is both the honest reading and a genuinely useful check when working out
    // which term is lighting something.
    //
    // Step 0.1 rather than 0.5 so the newly-reachable low end is controllable, and so the core's
    // own 2.4 still lands exactly on the grid — a range input snaps its value to a step boundary,
    // so on a 0.5 grid starting at 0 a restored preset holding 2.4 would display as 2.5.
    const si = makeSunRow(
        "Sun intensity",
        0,
        60,
        0.1,
        (v) => `${v.toFixed(1)}\u00d7`,
        () => sunIntensity,
        (v) => (sunIntensity = v)
    );
    // The unshadowable ambient floor. At the core's 0.75 it swamps the shadows; at 0 the scene is
    // lit by the sky and the sun alone, which is what makes them read.
    const am = makeSunRow(
        "Ambient fill",
        0,
        BASE_AMBIENT_INTENSITY,
        0.01,
        (v) => v.toFixed(2),
        () => ambientIntensity,
        (v) => (ambientIntensity = v)
    );
    const azRow = az.row;
    const elRow = el.row;
    const siRow = si.row;
    const amRow = am.row;
    const applyShadowRowVisibility = (): void => {
        // The angle rows stay visible with shadows off: the sun still LIGHTS the scene, so aiming
        // it is useful either way (and it is how you set up a shot before turning shadows on).
        azRow.style.display = "";
        elRow.style.display = "";
    };
    /** Push the live shadow state into the widgets. Called when the panel is (re)built and
     *  whenever a preset restores values behind the widgets' backs. */
    const syncShadowUi = (): void => {
        shadowChk.checked = shadowsOn;
        az.sync();
        el.sync();
        si.sync();
        am.sync();
        applyShadowRowVisibility();
    };
    syncShadowUi();
    shadowChk.onchange = (): void => {
        shadowsOn = shadowChk.checked;
        applyShadowRowVisibility();
        applyShadows();
    };

    return {
        key: "waterfall",
        label: "Waterfall",
        envUrl: WATERFALL_ENV_URL,
        envKey: "industrial",
        envRotationDeg: ENV_ROTATION_DEG,
        // The scene is authored around PB-MPM at the high tier — that pairing is what the
        // terraces, emitter charge and bloom were tuned against — so open on it. First visit
        // only: after that the method/quality dropdowns are the user's.
        defaultMethod: "PB-MPM",
        defaultQuality: "high",
        sdf,
        writeSdfParams,
        flow() {
            return buildFlow();
        },
        onEnter(): void {
            active = true;
            for (const m of meshes) {
                setMeshVisible(m, containerVisible);
            }
            // The backdrop follows the same "show container / nozzle meshes" toggle as the rock:
            // together they are the scene dressing, and showing one without the other reads as a
            // bug rather than as a debug view.
            for (const m of sceneryMeshes) {
                setMeshVisible(m, containerVisible);
            }
            applyShadows();
            setMeshVisible(ctx.ground, false); // presented against open sky — no ground plane at all
            applyBloom(); // the presentation stage is shared — claim it while we are on screen
            ctx.camera.target.x = ROCK_CX;
            ctx.camera.target.y = 3.2 * wf.meshScale;
            ctx.camera.target.z = ROCK_CZ;
        },
        onLeave(): void {
            active = false;
            applyShadows(); // hand the shadow map back before the next demo takes over
            if (authoring) {
                setAuthoring(false); // drop the overlay canvas with the demo
            }
            for (const m of meshes) {
                setMeshVisible(m, false);
            }
            for (const m of sceneryMeshes) {
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
         *  inert. The intensity/threshold numbers round-trip through demoParams as usual.
         *
         *  The sun-shadow toggle and its two angles ride here too. They are part of the LOOK a
         *  preset pins — a scene exported with a low raking sun comes back flat without them —
         *  and `demoState` takes numbers as well as booleans, so all three fit with no schema
         *  change anywhere in the core. */
        snapshotState(): Record<string, DemoStateValue> {
            return { bloom: wf.bloom, shadows: shadowsOn, sunAzimuth, sunElevation, sunIntensity, ambientIntensity, oasis: oasisQuality };
        },
        restoreState(state: Record<string, DemoStateValue>): void {
            if (typeof state.bloom === "boolean") {
                wf.bloom = state.bloom;
                applyBloom();
            }
            // Angles BEFORE the toggle: applyShadows re-aims nothing, so the direction has to be
            // current before the map is asked to render with it.
            if (typeof state.sunAzimuth === "number") {
                sunAzimuth = state.sunAzimuth;
            }
            if (typeof state.sunElevation === "number") {
                sunElevation = state.sunElevation;
            }
            if (typeof state.sunIntensity === "number") {
                sunIntensity = state.sunIntensity;
            }
            if (typeof state.ambientIntensity === "number") {
                ambientIntensity = state.ambientIntensity;
            }
            applySunDirection();
            if (typeof state.shadows === "boolean") {
                shadowsOn = state.shadows;
            }
            applyShadows();
            // The backdrop tier is part of the exported LOOK: a preset shot against the 1 M-triangle
            // ring comes back framed by the 80 k one without it. Only act on a real change — the
            // high tier is a 4.5 MB fetch, and `showOasisTier` would otherwise re-run the whole
            // load/visibility dance on every restore. Validated against the known tiers rather than
            // cast, so a hand-edited or stale file cannot leave `oasisQuality` naming a tier that
            // has no URL (which would fetch `undefined` and quietly lose the scenery).
            if (typeof state.oasis === "string" && state.oasis in OASIS_URLS && state.oasis !== oasisQuality) {
                oasisQuality = state.oasis as OasisQuality;
                showOasisTier(oasisQuality);
            }
            oasisSel.value = oasisQuality;
            oasisSel.disabled = oasisPending !== null;
            syncShadowUi(); // the widgets did not make these changes, so pull them into line
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            for (const m of meshes) {
                setMeshVisible(m, active && v);
            }
            // The backdrop follows the same toggle: it is the other half of "show me the scene
            // dressing", and hiding the rock while a full oasis ring stays up looks like a bug.
            for (const m of sceneryMeshes) {
                setMeshVisible(m, active && v);
            }
            applyShadows(); // a hidden ring must stop casting too
        },
        update(): void {
            // The core zeroes the shared UBO's bytes 32..159 on every pair switch, so the
            // 16-float collision block is re-written each frame (cheap: one 64-byte write).
            writeSdfParams();
            if (authoring) {
                drawOverlay();
            }
        },
        isReady(): boolean {
            return rockReady && heightMapReady && oasisCache.has(oasisQuality);
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
            return [
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
            ctx.refreshFlow();
        },
        extraControls() {
            // The panel is rebuilt on every re-enter, so re-sync from live state.
            oasisSel.value = oasisQuality;
            oasisSel.disabled = oasisPending !== null;
            syncShadowUi();
            return [oasisRow, shadowRow, azRow, elRow, siRow, amRow];
        },
        /** The rock plus the CURRENTLY SHOWN oasis tier. The ring is a caster because its palms
         *  are what make the pond's shadow catcher read; the ortho box grows to ~100 world units
         *  to fit it, which a 2048² map still resolves comfortably. */
        shadowCasters(): Mesh[] {
            return shadowCasterMeshes();
        },
    };
}
