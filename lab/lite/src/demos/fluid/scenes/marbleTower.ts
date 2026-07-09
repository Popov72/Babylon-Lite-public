// Marble Tower demo — a kinetic marble-run model (glTF) standing on the ground with
// water POURING from a single nozzle at its top, cascading down, pooling on the floor,
// and being RECIRCULATED back up to the top nozzle (the fountain's intake→nozzle idea).
//
// SCAFFOLD / Part 1: the collision surface is a PLACEHOLDER analytic SDF — a generous
// capped CYLINDER domain (radius DOMAIN_R, floor..CEILING_Y) that just confines the
// water around the tower plus a floor. The real analytic tower-cascade SDF is authored
// on top of this: the struct reserves eight spare `vec4` "tier" slots so tier params
// can be added WITHOUT changing the buffer binding, and `update()` re-writes the whole
// param block every frame (like the box paddle) so those slots survive the core's
// per-switch `clearSceneHoles()` (which zeroes offset 32..159).

import { addToScene, loadGltf, setMeshVisible } from "babylon-lite";
import type { Mesh, SceneNode } from "babylon-lite";
import type { EmitterConfig, FluidSim, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { generateMeshSdf } from "babylon-lite/fluid/volume-sampling/index.js";
import { createPlane, createMeshFromData } from "babylon-lite/mesh/mesh-factories.js";
import { createShaderMaterial, setShaderTexture } from "babylon-lite/material/shader/shader-material.js";
import type { ShaderMaterial } from "babylon-lite/material/shader/shader-material.js";
import { createTexture2DFromPixels, updateTexture2DFromPixels } from "babylon-lite/texture/pixels-texture.js";
import type { Texture2D } from "babylon-lite/texture/texture-2d.js";
import { releaseTexture } from "babylon-lite/resource/gpu-pool.js";
import type { DemoParam, FluidCtx, FluidDemo, PairState } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";

const TOWER_URL = "https://assets.babylonjs.com/meshes/Marble/marbleTower/marbleTower.gltf";

// Target framing: the tower is scaled so it stands TOWER_HEIGHT tall, centred on X/Z at
// the origin, base sitting on the ground (y=0), so its top is ~y=TOWER_HEIGHT.
const TOWER_HEIGHT = 14;
const FLOOR_Y = 0;
// Placeholder confinement domain: a capped cylinder generous enough to hold the whole
// cascade. Gandalf tightens / carves the tower solid out of this next.
const DOMAIN_R = 9; // cylinder radius around the tower (X/Z half-reach)
const CEILING_Y = 16; // domain lid — keeps the top pour inside the sim grid (±20)
// Single top nozzle: just below the tower crown, pouring straight down the axis.
const TOP_EMIT_Y = TOWER_HEIGHT - 0.5;

// The pour is centred on the tower's HIGHEST structure — the small top cube whose four
// inlet holes feed the marble run. We measure that cube's X/Z footprint from the top slab
// of geometry (TOP_SLAB world units below the apex) and straddle it with 4 nozzles at
// ±HOLE_OFFSET of its half-extent (a 2×2 pattern over the four holes).
const TOP_SLAB = 1.4;
const HOLE_OFFSET = 0.5;

// Baked mesh-SDF grid resolution. cellSize ≈ 0.05 resolves the ~14-tall × ~8.5-wide tower
// at ~170×290×165 ≈ 8M nodes (≈32 MB storage buffer) — fine enough to capture the thin
// wooden troughs/rails (walls ~0.1 thick) so water is channelled instead of leaking through.
// padding adds a 2-cell margin so edge nodes read far-outside positive (the sim clamps to
// the grid edge). Bake is a one-time ~few-second CPU pass in the async load callback.
const SDF_CELL_SIZE = 0.05;
const SDF_PADDING = 2;

// ── Water wheel ANALYTIC collision SDF (the glTF "wheel" node → mesh 4): a rim torus + a
//    coaxial axle shaft running along +X, offset on the +X side of the tower. These WORLD-space
//    values were measured offline from the asset geometry AFTER the load transform (which fits
//    the tower to TOWER_HEIGHT tall, base at y=0, centred on X/Z), so they are stable as long as
//    TOWER_HEIGHT stays 14. They drive ONLY the rotation-symmetric analytic wheel SDF the fluid
//    collides with (the rim+hub the water splashes off). The DISPLAY of the wheel uses the real
//    textured mesh, spun in place about its axle via the "wheel" TransformNode (see the pivot
//    repurposing in the load callback) — it is NOT split, re-created, or carved from the bake.
//    Recon: rim band ≈[3.73,3.99], shaft radius ≈0.5 spanning world X≈[0.05,4.24]; NO spokes in
//    the geometry (rim + hub only).
const WHEEL_C: [number, number, number] = [3.78, 4.28, 0.06]; // disk centre — a point on the axle
const WHEEL_AXLE: [number, number, number] = [1, 0, 0]; // unit axle direction (horizontal, +X)
const WHEEL_R = 3.65; // rim/bucket-band centreline — bucket outer (R+T)=4.05 = measured mesh perimeter
const WHEEL_T = 0.4; // rim half-thickness (axial half-width ≈0.4; R+T is the bucket-band outer radius)
const WHEEL_HUB_R = 0.5; // axle / hub radius
const WHEEL_HUB_HALF = 2.1; // hub / axle half-length along the axle
const WHEEL_HUB_OFFSET = -1.63; // hub centre along the axle relative to C (shaft world X≈[0.05,4.24])

// Mesh node carrying CPU geometry (present on glTF Mesh leaves, absent on TransformNodes).
type CpuMeshNode = SceneNode & {
    _cpuPositions?: Float32Array;
    _cpuNormals?: Float32Array;
    _cpuUvs?: Float32Array;
    _cpuIndices?: Uint32Array;
};
type Aabb = { min: [number, number, number]; max: [number, number, number] };
// Retained CPU copy of a baked signed-distance grid (drives the SDF texture visualizer):
// data is x-fastest (index = i + j*dimX + k*dimX*dimY), NEGATIVE inside the solid.
type BakedGrid = { data: Float32Array; dims: [number, number, number]; origin: [number, number, number]; cellSize: number };

export function createMarbleTowerDemo(ctx: FluidCtx): FluidDemo {
    const { engine } = ctx;

    // ── Two scene SDFs sharing the one UBO (ctx.sceneSdfBuffer): ANALYTIC (default) and
    //    BAKED. A UI checkbox swaps between them so we can compare the placeholder boxes
    //    against the real marble-run geometry.
    //
    // ANALYTIC: capped cylinder domain + floor + two rounded "tower" boxes (keep+plinth, tier0..3)
    // + the STATIC analytic water wheel. Positive = valid fluid space (inside the cylinder AND
    // below the lid AND above the floor AND outside every solid). update() re-writes the whole
    // param block every frame so it survives clearSceneHoles() (which zeroes floats 8..39).
    //
    // Shared WGSL: the analytic water wheel = a coaxial capped-cylinder hub + N straight radial
    // SPOKES + a perimeter BUCKET band (sole floor ring + two side shrouds + M rotating radial
    // vanes forming the open pockets), all about the axle `wheelA` through centre `wheelC`, unioned
    // via min() into BOTH sceneSdf variants (negative inside the solid). wheelMisc.x toggles it (UI
    // checkbox). The hub + sole + shrouds are rotation-symmetric so θ is a no-op on them, but the
    // SPOKES and bucket VANES rotate with the wheel as a MOVING BOUNDARY (exactly like the box
    // paddle): their phase advances by -(θ + ω·dt) inside the SDF, so the core's g2p confinement
    // finite-differences sceneSdf(pt,dt) and recovers the surface velocity → the spinning
    // spokes/vanes push/carry the fluid, and water flows through the gaps between them.
    // The mesh spins by -wheelTheta; the engine-measured world rotation sense is dφ/dθ = -1, so the
    // SDF uses -θ (and -ω·dt) → collision features co-rotate with the VISIBLE wheel on screen.
    // Layout floats 24..39: wheelC=(cx,cy,cz,R), wheelA=(spokeHalfW,bucketCount@29,_,T),
    // wheelHub=(hubR,hubHalf,hubOffset,θ@35), wheelMisc=(enabled@36, ω@37, spokeN@38, φ0@39).
    // The axle direction is a compile-time constant (world +X) — hardcoded in sdWheel + the flux
    // reduction — so wheelA.yz is free; we repurpose wheelA.x for the spoke half-width and
    // wheelA.y for the (tunable) bucket count.
    const SPOKE_HALF_AX = 0.32; // spoke axial half-thickness (< WHEEL_T=0.4) — thin but tunnel-safe like the paddle
    // Overshot BUCKET band (the ~30 open pockets around the rim that catch top-poured water, carry
    // it as the wheel turns, then overflow / eject it near the bottom). Compile-time geometry; the
    // pocket COUNT is user-tunable (wheelA.y, "Wheel buckets" slider). A pocket is bounded by an
    // inner cylindrical SOLE (floor), two axial SHROUD side walls, and two radial VANES — and is
    // OPEN on the outer radial face, so gravity holds water while the pocket climbs/descends and
    // dumps it once the opening rotates to face downward. Walls are thin but tunnel-safe.
    const BUCKET_DEPTH = 0.40; // radial depth of the pocket band → bInner=3.65 = measured rim inner (spokes stop here)
    const BUCKET_WALL = 0.1; // sole / shroud wall half-thickness (tunnel-safe)
    const VANE_HALF_W = 0.09; // vane tangential half-width
    const WHEEL_SDF_FN = `fn sdWheel(pt: vec3<f32>, dt: f32) -> f32 {
    if (sceneSdfParams.wheelMisc.x < 0.5) { return 1.0e9; }
    let wc = sceneSdfParams.wheelC.xyz;
    let axis = vec3<f32>(1.0, 0.0, 0.0); // axle = world +X (compile-time constant; frees wheelA.xyz)
    let d = pt - wc;
    let a = dot(d, axis);
    let perp = d - a * axis;
    let rad = length(perp);
    let rInner = sceneSdfParams.wheelHub.x; // hub radius
    let rOuter = sceneSdfParams.wheelC.w;   // rim radius
    let T = sceneSdfParams.wheelA.w;        // rim axial half-width
    // Bucket-band radii (used to size both the spokes and the buckets so they don't overlap).
    let bOuter = rOuter + T;                       // outer edge of the wheel band
    let bInner = bOuter - ${BUCKET_DEPTH.toFixed(4)}; // sole radius = pocket floor = spoke outer end
    let bMid = 0.5 * (bInner + bOuter);
    let bHalfR = 0.5 * (bOuter - bInner);
    let bHalfA = T;                                // shrouds sit at the axial extremes |a| = T
    // Hub / axle capped cylinder (rotation-symmetric, static).
    let qx = rad - rInner;
    let qy = abs(a - sceneSdfParams.wheelHub.z) - sceneSdfParams.wheelHub.y;
    let hub = length(max(vec2<f32>(qx, qy), vec2<f32>(0.0))) + min(max(qx, qy), 0.0);
    // Angle in the disk plane + the shared moving-boundary phase. Phase = -(θ + ω·dt): the mesh
    // spins by wheelNode.rotation.x = -θ, and the engine-measured world rotation sense of the disk
    // is dφ_world/dθ = -1 (world Y-Z angle DECREASES as θ grows). The SDF spokes/vanes track that
    // with -θ, so collision + debug overlay + visible mesh all co-rotate. The core g2p confinement
    // finite-differences sceneSdf(pt,dt) and recovers their surface velocity → they push the fluid.
    let phi = atan2(perp.z, perp.y);
    let spokePhase = -(sceneSdfParams.wheelHub.w + sceneSdfParams.wheelMisc.y * dt);
    // N radial spokes (hub -> rim bars, static shape, rotating phase). They stop at the rim INNER
    // (bInner = measured felloe inner radius) — the real spokes mortise into the rim and do not cross
    // the bucket band, so the analytic wheel matches the mesh instead of poking through the rim.
    let nSpokes = max(sceneSdfParams.wheelMisc.z, 1.0);
    let sector = 6.2831853071795864 / nSpokes;
    var rel = phi - sceneSdfParams.wheelMisc.w - spokePhase;
    rel = rel - sector * round(rel / sector);
    let midR = 0.5 * (rInner + bInner);
    let halfLenR = 0.5 * (bInner - rInner);
    let sa = abs(a) - ${SPOKE_HALF_AX.toFixed(4)};
    let st = abs(rad * rel) - sceneSdfParams.wheelA.x; // spoke tangential half-width (tunable, wheelA.x)
    let sr = abs(rad - midR) - halfLenR;
    let spoke = length(max(vec3<f32>(sa, st, sr), vec3<f32>(0.0))) + min(max(sa, max(st, sr)), 0.0);
    // ── Bucket band: sole (inner floor ring) + two side shrouds + N rotating radial vanes ──────
    // Sole: a thin cylindrical wall at rad = bInner spanning the full axial width (pocket floor).
    let soleRad = abs(rad - bInner) - ${BUCKET_WALL.toFixed(4)};
    let soleAx = abs(a) - bHalfA;
    let sole = length(max(vec2<f32>(soleRad, soleAx), vec2<f32>(0.0))) + min(max(soleRad, soleAx), 0.0);
    // Shrouds: the two side walls at |a| = bHalfA, spanning the pocket band radially (pocket sides).
    let shrAx = abs(abs(a) - bHalfA) - ${BUCKET_WALL.toFixed(4)};
    let shrRad = abs(rad - bMid) - bHalfR;
    let shroud = length(max(vec2<f32>(shrRad, shrAx), vec2<f32>(0.0))) + min(max(shrRad, shrAx), 0.0);
    // Vanes: N thin radial dividers across the band, polar-repeated, rotating with the wheel. The
    // buckets are a 30-fold structure independent of the 12 spokes, so they use their OWN base phase
    // (wheelA.z, detected separately) — reusing the spoke phase misaligns the vanes vs the mesh.
    let nBuckets = max(sceneSdfParams.wheelA.y, 1.0);
    let bSector = 6.2831853071795864 / nBuckets;
    var brel = phi - sceneSdfParams.wheelA.z - spokePhase;
    brel = brel - bSector * round(brel / bSector);
    let vAx = abs(a) - bHalfA;
    let vT = abs(rad * brel) - ${VANE_HALF_W.toFixed(4)};
    let vR = abs(rad - bMid) - bHalfR;
    let vane = length(max(vec3<f32>(vAx, vT, vR), vec3<f32>(0.0))) + min(max(vAx, max(vT, vR)), 0.0);
    return min(min(min(sole, shroud), vane), min(hub, spoke));
}`;

    // tier4..tier7 (floats 24..39) are repurposed into the wheel param block (wheelC / wheelA /
    // wheelHub / wheelMisc); the analytic tower only needs tier0..tier3 (keep + plinth).
    const ANALYTIC_STRUCT =
        "struct SceneSdfParams { domain: vec4<f32>, center: vec4<f32>, tier0: vec4<f32>, tier1: vec4<f32>, tier2: vec4<f32>, tier3: vec4<f32>, wheelC: vec4<f32>, wheelA: vec4<f32>, wheelHub: vec4<f32>, wheelMisc: vec4<f32>, };";
    const ANALYTIC_SDF = `fn sdRBox(p: vec3<f32>, c: vec3<f32>, h: vec3<f32>, r: f32) -> f32 {
    let q = abs(p - c) - (h - vec3<f32>(r));
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}
fn towerBox(pt: vec3<f32>, t0: vec4<f32>, t1: vec4<f32>) -> f32 {
    // t0.xyz = centre, t0.w = halfX; t1.xyz = halfY, halfZ, cornerRadius; t1.w = enabled.
    // Disabled slot returns a large positive so it never constrains (min()).
    if (t1.w < 0.5) { return 1.0e9; }
    return sdRBox(pt, t0.xyz, vec3<f32>(t0.w, t1.x, t1.y), t1.z);
}
${WHEEL_SDF_FN}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    // Valid fluid space = inside the domain cylinder, below the lid, above the floor, AND outside
    // the two tower boxes (keep+plinth) AND outside the analytic water wheel. min() intersects all
    // constraints (any negative term => penetrating). Each solid's signed distance is positive
    // outside it.
    let radial = sceneSdfParams.domain.x - length(pt.xz - sceneSdfParams.center.xz);
    let ceiling = sceneSdfParams.domain.y - pt.y;
    let floorD = pt.y - sceneSdfParams.domain.w;
    var d = min(min(radial, ceiling), floorD);
    d = min(d, towerBox(pt, sceneSdfParams.tier0, sceneSdfParams.tier1));
    d = min(d, towerBox(pt, sceneSdfParams.tier2, sceneSdfParams.tier3));
    d = min(d, sdWheel(pt, dt));
    return d;
}`;

    // BAKED: the same cylinder/floor domain intersected with a mesh-baked signed-distance
    // grid of the REAL marble-run geometry, sampled trilinearly on the GPU (the sims inject
    // the `sampleSdfGrid` sampler + bind the storage grid; calling it keeps the binding alive
    // under layout:"auto"). The struct reuses the shared UBO but repurposes the spare "tier"
    // region for the grid params. CPU packing (see `bakedData`), floats into the 160-byte UBO:
    //   domain     (floats 0..3)   = cyl radius, ceilY, _, floorY        [survives clearSceneHoles]
    //   center     (floats 4..7)   = domain centre.xz (rest spare)       [survives clearSceneHoles]
    //   gridOrigin (floats 8..11)  = grid origin.xyz, invCell(.w)        [hole ring → re-written each frame]
    //   gridDims   (floats 12..15) = grid node counts.xyz (→ i32), spare
    //   pad0/pad1  (floats 16..23) = unused (keeps the wheel block at the SAME floats as ANALYTIC)
    //   wheelC/A/Hub/Misc (floats 24..39) = STATIC analytic wheel (unioned so it collides even in
    //                                       baked mode, where its disk has been CARVED from the grid)
    const BAKED_STRUCT =
        "struct SceneSdfParams { domain: vec4<f32>, center: vec4<f32>, gridOrigin: vec4<f32>, gridDims: vec4<f32>, pad0: vec4<f32>, pad1: vec4<f32>, wheelC: vec4<f32>, wheelA: vec4<f32>, wheelHub: vec4<f32>, wheelMisc: vec4<f32>, };";
    const BAKED_SDF = `${WHEEL_SDF_FN}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let radial = sceneSdfParams.domain.x - length(pt.xz - sceneSdfParams.center.xz);
    let ceiling = sceneSdfParams.domain.y - pt.y;
    let floorD = pt.y - sceneSdfParams.domain.w;
    let dims = vec3<i32>(i32(sceneSdfParams.gridDims.x), i32(sceneSdfParams.gridDims.y), i32(sceneSdfParams.gridDims.z));
    // Baked mesh SDF: <0 inside the marble solid (penetrating), >0 outside. Intersect with
    // the domain via min() so water stays inside the cylinder/floor AND outside the tower. The
    // wheel disk is carved from the baked grid, so union the STATIC analytic wheel back in here.
    let baked = sampleSdfGrid(pt, sceneSdfParams.gridOrigin.xyz, sceneSdfParams.gridOrigin.w, dims);
    return min(min(min(min(radial, ceiling), floorD), baked), sdWheel(pt, dt));
}`;

    // Baked grid state (uploaded after the async load bakes it) + which mode is live.
    let gridBuffer: GPUBuffer | null = null;
    let bakeReady = false; // true once the mesh SDF is baked + uploaded
    let bakedActive = false; // true when the BAKED spec is selected (only possible once bakeReady)
    // Desired baked state — set by restoreState() / the checkbox. If a preset asks for baked mode
    // BEFORE the async bake finishes, we remember it here and apply it at the end of bakeSceneSdf().
    let desiredBaked = false;
    // The baked distances, retained on the CPU so the SDF-texture visualizer can slice arbitrary
    // cross-sections without a GPU readback (the gridBuffer upload is the GPU copy the sims sample).
    let bakedGrid: BakedGrid | null = null;

    // The LIVE spec handed to the core (FluidDemo.sdf). Starts analytic; applyMode() swaps
    // its fields IN PLACE so a demo re-enter / method switch (both re-read demo.sdf via the
    // core's applySceneSdf → setSceneSdf on both sims) picks up the current choice.
    const sdf: SceneSdfSpec = { struct: ANALYTIC_STRUCT, sdf: ANALYTIC_SDF, buffer: ctx.sceneSdfBuffer };

    const applyMode = (baked: boolean): void => {
        if (baked) {
            sdf.struct = BAKED_STRUCT;
            sdf.sdf = BAKED_SDF;
            sdf.sdfGrid = gridBuffer ?? undefined; // storage grid the sims bind
            sdf.gridConfine = false; // per-particle push-out vs. the baked shape (a run isn't a closed box)
        } else {
            sdf.struct = ANALYTIC_STRUCT;
            sdf.sdf = ANALYTIC_SDF;
            sdf.sdfGrid = undefined; // no grid → sims take the analytic-only path
            sdf.gridConfine = undefined; // closed-container default (matches the placeholder cylinder)
        }
    };

    // Whole param block (40 floats = 160 bytes = the shared UBO size). domain.x = radius,
    // domain.y = ceilingY, domain.w = floorY; center.xz = domain centre (origin). The
    // spare tier slots stay zero until the real cascade uses them.
    const sdfData = new Float32Array(40);
    sdfData[0] = DOMAIN_R; // domain.x — cylinder radius
    sdfData[1] = CEILING_Y; // domain.y — lid height
    sdfData[2] = 0; // domain.z — spare
    sdfData[3] = FLOOR_Y; // domain.w — floor height
    sdfData[4] = 0; // center.x
    sdfData[5] = TOWER_HEIGHT * 0.5; // center.y — spare (tower mid-height, handy for tiers)
    sdfData[6] = 0; // center.z
    sdfData[7] = 0; // center.w — spare

    // Analytic tower solid: a stack of rounded boxes the water flows AROUND/DOWN. Each box
    // uses two tier slots — tierN = centre.xyz + halfX (w); tierN+1 = halfY, halfZ,
    // cornerRadius, enabled(1). Approximates the castle's central keep on its wider plinth
    // (the water-wheel + cantilever tracks are ignored — that's what the baked SDF is for).
    // Keep: y≈2..12, ~2.8 wide. Plinth: y≈0..2.4, ~4.6 wide.
    const setBox = (slot: number, cx: number, cy: number, cz: number, hx: number, hy: number, hz: number, round: number): void => {
        const o = 8 + slot * 8;
        sdfData[o] = cx;
        sdfData[o + 1] = cy;
        sdfData[o + 2] = cz;
        sdfData[o + 3] = hx;
        sdfData[o + 4] = hy;
        sdfData[o + 5] = hz;
        sdfData[o + 6] = round;
        sdfData[o + 7] = 1; // enabled
    };
    setBox(0, 0, 7.0, 0, 1.4, 5.0, 1.4, 0.2); // keep
    setBox(1, 0, 1.2, 0, 2.3, 1.2, 2.0, 0.2); // plinth
    // slots 2,3 (floats 24..39) are now the wheel param block (see packWheel below), not boxes.

    // ── Analytic water wheel: rim torus + hub + N rotating SPOKES (moving boundary) ────────────
    // Enabled by default; the "Water wheel (analytic)" UI checkbox toggles the SDF union (the
    // wheel mesh always renders regardless). Packed at floats 24..39 in BOTH param arrays:
    //   wheelC  (24..27) = centre.xyz, R (rim radius)
    //   wheelA  (28..31) = spokeHalfW (tangential half-width, tunable), M buckets, bucketPhase0, T
    //   wheelHub(32..35) = hubR, hubHalf, hubOffset, θ (spoke angle, written each frame)
    //   wheelMisc(36..39) = enabled(1/0), ω (rad/s, each frame), N (spoke count), φ0 (spoke base phase)
    let wheelSdfEnabled = true;
    // N spokes + spoke half-width are USER-TUNABLE (sliders) — the source of truth. The wheel's
    // spoke MIDDLES carry no vertices, so the Fourier auto-detect can't pin the count; the user
    // dials N + thickness by eye against the "Debug wheel SDF" overlay. detectSpokes() below is
    // kept only as a diagnostic hint + to seed the base phase φ0.
    let spokeCount = 12; // N straight spokes — driven by the "Wheel spokes" slider (mesh has 12, Fourier-detected)
    let spokeHalfW = 0.18; // spoke tangential half-width — driven by "Spoke thickness"; packed into wheelA.x
    let bucketCount = 30; // M perimeter buckets — driven by the "Wheel buckets" slider; packed into wheelA.y
    let spokePhase0 = 0; // base phase φ0 — world disk-plane angle of spoke 0 (seeded by detectSpokes)
    let bucketPhase0 = 0; // base phase for the 30-fold bucket vanes (seeded separately by detectSpokes)
    const packWheel = (arr: Float32Array): void => {
        arr[24] = WHEEL_C[0];
        arr[25] = WHEEL_C[1];
        arr[26] = WHEEL_C[2];
        arr[27] = WHEEL_R;
        arr[28] = spokeHalfW; // wheelA.x — spoke tangential half-width (axle dir is a WGSL constant)
        arr[29] = bucketCount; // wheelA.y — M perimeter buckets (never 0 → sdWheel guards with max(M,1))
        arr[30] = bucketPhase0; // wheelA.z — bucket-vane base phase (30-fold, detected separately)
        arr[31] = WHEEL_T;
        arr[32] = WHEEL_HUB_R;
        arr[33] = WHEEL_HUB_HALF;
        arr[34] = WHEEL_HUB_OFFSET;
        arr[35] = 0; // θ — spoke angle (mirrored from wheelTheta each frame by updateWheelSpin)
        arr[36] = wheelSdfEnabled ? 1 : 0;
        arr[37] = 0; // ω — spoke angular speed (mirrored from wheelOmega each frame; drives the FD)
        arr[38] = spokeCount; // N spokes (never 0 → sdWheel guards with max(N,1))
        arr[39] = spokePhase0; // base phase φ0
    };

    packWheel(sdfData);

    // BAKED param block (40 floats = domain + center + grid params + wheel block). domain/center
    // mirror the analytic block; the grid origin/invCell/dims (floats 8..15) are filled by
    // bakeSceneSdf(); the wheel block (24..39) mirrors sdfData so the wheel collides in baked mode.
    const bakedData = new Float32Array(40);
    bakedData[0] = DOMAIN_R; // domain.x — cylinder radius
    bakedData[1] = CEILING_Y; // domain.y — lid height
    bakedData[3] = FLOOR_Y; // domain.w — floor height
    bakedData[5] = TOWER_HEIGHT * 0.5; // center.y — spare
    packWheel(bakedData);

    const writeSdfParams = (): void => {
        // Write the ACTIVE param block into the shared UBO. clearSceneHoles() zeroes bytes
        // 32..159 (floats 8..39) on every pair switch, so update() re-writes this every frame.
        // GUARD: only touch the SHARED sceneSdfBuffer while THIS demo is on-screen. All demos
        // share ctx.sceneSdfBuffer, and marbleTower's ASYNC load path (bakeSceneSdf/detectSpokes/
        // deferred baked-mode) fires ~seconds after boot — if that ran while another demo (e.g. the
        // default Box) were active it would clobber that demo's params (floats 0..15) and wreck its
        // bounds. onEnter() sets active=true before the core's applySceneSdf re-writes us, and
        // update() re-writes every frame, so gating here loses nothing when we ARE active.
        if (!active) {
            return;
        }
        if (bakedActive && gridBuffer) {
            engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, bakedData);
        } else {
            engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, sdfData);
        }
    };

    // Live-tweakable emitter params (exposed in the Demo-parameters UI).
    const marbleParams = { centralSpeed: 6, emitRate: 0.4, nozzleRadius: 0.35 };

    // The tower's HIGHEST structure (the small top cube carrying the 4 inlet holes): its
    // X/Z centre + half-extents + apex Y, measured from geometry on load (worldAabb centre
    // is skewed by the wide base/wheel, so we can't just pour at the origin). Falls back to
    // the axis until the async load fills it.
    const topStructure = { cx: 0, cz: 0, hx: 0.5, hz: 0.5, y: TOP_EMIT_Y };

    const marbleConfig = (): EmitterConfig => {
        // FOUR nozzles straddling the top cube's inlet holes (2×2), each pouring straight
        // down from just under the apex. Multiple emitters share the recycle budget (`rate`),
        // so this splits the same flow across the four holes rather than quadrupling it.
        const ox = Math.max(topStructure.hx * HOLE_OFFSET, 0.12);
        const oz = Math.max(topStructure.hz * HOLE_OFFSET, 0.12);
        const ey = topStructure.y - 0.25;
        const speed = marbleParams.centralSpeed;
        const radius = marbleParams.nozzleRadius;
        const emitters = ([[-1, -1], [1, -1], [-1, 1], [1, 1]] as const).map(([sx, sz]) => ({
            pos: [topStructure.cx + sx * ox, ey, topStructure.cz + sz * oz] as [number, number, number],
            dir: [0, -1, 0] as [number, number, number],
            speed,
            radius,
        }));
        return {
            emitters,
            // Pump intake: a thin slab across the whole domain floor. Settled water is pulled
            // back up to the top nozzles (throttled by `rate` — a controlled trickle).
            intakeMin: [-DOMAIN_R, FLOOR_Y, -DOMAIN_R],
            intakeMax: [DOMAIN_R, FLOOR_Y + 0.8, DOMAIN_R],
            rate: marbleParams.emitRate,
            spread: 0.2,
        };
    };

    // ── The tower model: loaded async + non-blocking, kept with its own materials ──
    const towerMeshes: Mesh[] = [];
    // The glTF "wheel" TransformNode (identity transform) repurposed as the spin pivot, and its
    // child Mesh leaf — the REAL textured water-wheel geometry (mesh 4). The mesh STAYS inside
    // towerMeshes so the standard visibility loops (onEnter/onLeave/setContainerVisible) cover it;
    // we only keep separate references to (a) exclude the mesh from the SDF bake and (b) drive the
    // pivot node's rotation for the spin. Set once the async load locates the node by name.
    let wheelNode: SceneNode | null = null;
    let wheelMesh: Mesh | null = null;
    let active = false; // true while this demo is the on-screen demo
    let containerVisible = true; // toggled by the "Show container" UI checkbox

    const collectMeshes = (node: SceneNode, out: Mesh[]): void => {
        if ("_gpu" in node) {
            out.push(node as unknown as Mesh);
        }
        for (const c of node.children) {
            collectMeshes(c, out);
        }
    };

    // Depth-first search for a node by name (the loader preserves glTF node names on the
    // TransformNode hierarchy). Used to locate the "wheel" node (its child mesh is the wheel).
    const findNodeByName = (node: SceneNode, target: string): SceneNode | null => {
        if (node.name === target) {
            return node;
        }
        for (const c of node.children) {
            const found = findNodeByName(c, target);
            if (found) {
                return found;
            }
        }
        return null;
    };

    // World-space AABB folded over every mesh's CPU positions × its worldMatrix
    // (column-major Mat4: m[col*4+row], translation in m[12..14]).
    const worldAabb = (meshes: Mesh[]): Aabb => {
        let minX = Infinity,
            minY = Infinity,
            minZ = Infinity,
            maxX = -Infinity,
            maxY = -Infinity,
            maxZ = -Infinity;
        for (const mesh of meshes) {
            const pos = (mesh as CpuMeshNode)._cpuPositions;
            if (!pos || pos.length === 0) {
                continue;
            }
            const w = mesh.worldMatrix;
            const m0 = w[0]!,
                m1 = w[1]!,
                m2 = w[2]!,
                m4 = w[4]!,
                m5 = w[5]!,
                m6 = w[6]!,
                m8 = w[8]!,
                m9 = w[9]!,
                m10 = w[10]!,
                m12 = w[12]!,
                m13 = w[13]!,
                m14 = w[14]!;
            for (let i = 0; i < pos.length; i += 3) {
                const lx = pos[i]!,
                    ly = pos[i + 1]!,
                    lz = pos[i + 2]!;
                const x = m0 * lx + m4 * ly + m8 * lz + m12;
                const y = m1 * lx + m5 * ly + m9 * lz + m13;
                const z = m2 * lx + m6 * ly + m10 * lz + m14;
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                minZ = Math.min(minZ, z);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
                maxZ = Math.max(maxZ, z);
            }
        }
        return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
    };

    // ── "Baked SDF (mesh)" toggle UI (appended under Demo parameters). Built once here so
    //    bakeSceneSdf() can enable it + show the grid stats when the async bake finishes. ──
    const bakedRow = document.createElement("label");
    bakedRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:6px 0 2px;cursor:pointer;";
    const bakedChk = document.createElement("input");
    bakedChk.type = "checkbox";
    bakedChk.disabled = true; // enabled once the mesh SDF is baked
    const bakedText = document.createElement("span");
    bakedText.textContent = "Baked SDF (mesh)";
    bakedRow.append(bakedChk, bakedText);
    const bakedStatusEl = document.createElement("div");
    bakedStatusEl.style.cssText = "color:#9fb4cc;font-size:11px;margin:0 0 6px;";
    bakedStatusEl.textContent = "baking mesh SDF…";
    let bakeStatus = "baking mesh SDF…";

    // Apply the baked/analytic mode (shared by the checkbox handler + restoreState). Remembers
    // the DESIRED state in `desiredBaked` so a preset that asks for baked mode BEFORE the async
    // bake completes is honoured later (bakeSceneSdf re-invokes this once bakeReady). When the bake
    // isn't ready yet the switch is deferred (returns early) so we never select a null grid.
    const setBakedMode = (v: boolean): void => {
        desiredBaked = v;
        if (v && !bakeReady) {
            return; // deferred — applied at the end of bakeSceneSdf once the grid exists
        }
        bakedActive = v;
        applyMode(bakedActive);
        writeSdfParams(); // re-pack the UBO for the active mode
        if (active) {
            // Rebuild the active sim's collision pipes for the new SDF. The inactive backend is
            // refreshed lazily on the next method switch (the core's applySceneSdf re-reads demo.sdf).
            ctx.getActiveSim().setSceneSdf(sdf);
        }
    };

    bakedChk.onchange = (): void => {
        const want = bakedChk.checked;
        if (want && !bakeReady) {
            bakedChk.checked = false; // bake not ready → stay analytic
            return;
        }
        setBakedMode(want);
    };

    // ── "Water wheel (analytic)" toggle: unions the analytic wheel SDF (rim + hub + rotating
    //    spokes) into whichever mode is live (analytic or baked). Default ON. The textured wheel
    //    mesh renders + spins regardless; this only gates the SDF collision term (a wheelMisc.x
    //    uniform the WGSL branches on — no pipeline rebuild). The wheel mesh is excluded from the
    //    baked grid either way, so with this OFF in baked mode water passes through the wheel region.
    const wheelRow = document.createElement("label");
    wheelRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:6px 0 2px;cursor:pointer;";
    const wheelChk = document.createElement("input");
    wheelChk.type = "checkbox";
    wheelChk.checked = wheelSdfEnabled;
    const wheelText = document.createElement("span");
    wheelText.textContent = "Water wheel (analytic)";
    wheelRow.append(wheelChk, wheelText);

    // Toggle the wheel SDF union (shared by the checkbox handler + restoreState).
    const setWheelSdf = (v: boolean): void => {
        wheelSdfEnabled = v;
        packWheel(sdfData); // refresh the enabled flag (float 36) in BOTH blocks
        packWheel(bakedData);
        writeSdfParams(); // push the active block now (update() also re-writes every frame)
    };
    wheelChk.onchange = (): void => {
        setWheelSdf(wheelChk.checked);
    };

    // ── STAGE 2: flux-driven wheel spin ────────────────────────────────────────────────────────
    // Spin the REAL textured wheel mesh (via its "wheel" pivot node) driven by the amount of MOVING
    // liquid that reaches the wheel. Each frame a tiny GPU reduction counts particles that are BOTH
    // inside the wheel's CATCH cylinder (axle-local: perpendicular distance to the axle ≤ R+margin,
    // |axial offset along the axle| ≤ T+margin) AND actually moving (normalized speed > SPEED_GATE),
    // reducing to a single atomic<u32>. That count is copied to a double-buffered staging buffer and
    // read back with mapAsync (NON-blocking — never awaited in the render path; frames with no fresh
    // value reuse the last one). The count is EMA-smoothed, mapped to a target angular speed
    // ωTarget = driveStrength·n (clamped to DRIVE_OMEGA_MAX), and the actual ω relaxes toward it; θ
    // integrates ω and drives wheelNode.rotation.x (axle = local +X → the disk spins in place about
    // the axle). This is fully DECOUPLED from collision: the analytic wheel SDF stays rotation-
    // symmetric, so θ is a physical no-op there — it is still mirrored into the param block (float 35)
    // for consistency. Because only MOVING water counts, the wheel coasts to a stop when the pour is
    // off and the water pools.
    const CATCH_MARGIN_R = 0.5; // radial slack beyond the rim radius R
    const CATCH_MARGIN_A = 0.6; // axial slack beyond the disk half-thickness T
    // Only water on the UPPER rim band drives the wheel: a particle counts when its offset from the
    // axle is in the rim band [R-RIM_BAND, R+CATCH_MARGIN_R] AND it is above the axle (perp.y>0). This
    // excludes (a) the submerged lower rim sitting in the base pool and (b) the hub interior, so the
    // count reflects water actually riding/striking the top of the wheel — it falls to ~0 when the
    // pour is off and the water drains, which is what stops the wheel.
    const RIM_BAND = 1.2; // radial depth of the rim catch band inward from R
    const DRIVE_OMEGA_MAX = 3; // rad/s — hard safety clamp AND the drive-strength slider's max
    const DRIVE_RESPONSIVENESS = 2; // 1/s — how fast ω relaxes toward its target (~0.5 s time-constant)
    const COUNT_EMA_RATE = 4; // 1/s — how fast the smoothed catch count tracks the async readback
    // Proportional drive: the smoothed moving-particle count is mapped through a deadzone + span into a
    // 0..1 flow fraction, so the wheel speed reflects HOW MUCH water reaches it (not a saturated on/off).
    // Below COUNT_DEADZONE there is too little water to turn the wheel → it idles/stops; COUNT_SPAN above
    // that maps to full drive. Both scale with the ~200k particle budget of this scene.
    const COUNT_DEADZONE = 250; // moving upper-rim particles below which the wheel is not driven
    const COUNT_SPAN = 2500; // moving-particle count above the deadzone that reaches full drive
    // Normalized speed gate: a particle is only counted when it is actually MOVING (its normalized
    // world speed `speed·debugNorm` exceeds this). Settled/pooled water the lower rim sits in reads
    // ~0 and is ignored, so the wheel only turns when moving water reaches it and coasts to a stop
    // when the pour is off. Resolution-independent (debugNorm ≈ 1/typical-max-speed).
    const SPEED_GATE = 0.3;
    const TWO_PI = Math.PI * 2;
    const FLUX_WG_SIZE = 256; // reduction workgroup size
    const FLUX_STAGING = 2; // double-buffered readback so mapAsync never stalls the render path

    let spinEnabled = true; // "Spin water wheel" checkbox (default ON)
    let driveStrength = 1.0; // "Wheel drive strength" slider → the wheel's ω (rad/s) at FULL flow
    let wheelTheta = 0; // current rotation angle about the axle (rad)
    let wheelOmega = 0; // current angular speed (rad/s)
    let smoothedCount = 0; // EMA of the catch count (de-jitters the async readback cadence)
    let latestCount = 0; // last successfully read-back catch count
    let spinReadoutFrames = 0; // throttles the ω/count read-out DOM update

    // Lazily-built GPU reduction resources — created once, never per frame.
    let fluxPipeline: GPUComputePipeline | null = null;
    let fluxCountBuffer: GPUBuffer | null = null; // atomic<u32> the reduction writes
    let fluxNormBuffer: GPUBuffer | null = null; // uniform: x = active sim's debugNorm (speed→normalized)
    let fluxBindGroup: GPUBindGroup | null = null;
    let fluxBoundPos: GPUBuffer | null = null; // positionBuffer the bind group is wired to (rebind on change)
    const fluxStaging: GPUBuffer[] = []; // COPY_DST→MAP_READ readback ring
    const fluxStagingBusy: boolean[] = []; // per-staging in-flight flag (mapAsync pending)

    // Reduction shader: for each particle, add 1 to the atomic when it lies inside the catch cylinder
    // AND is moving (speed gate). Wheel geometry is compile-time constant, so it is inlined as literals
    // (single source of truth via the WHEEL_* consts). arrayLength(&positions) == the sim's particle
    // count (positionBuffer is exactly `count` vec4s), so the last workgroup self-guards without a count
    // uniform. `speeds` is the sim's f32-per-particle world-speed buffer (same indexing as positions);
    // fluxParams.x = debugNorm (1/typical-max-speed) so `speed·norm` is a resolution-independent 0..1.
    const wgslF = (n: number): string => n.toFixed(5);
    const fluxRadCap = WHEEL_R + CATCH_MARGIN_R;
    const fluxRimInner = WHEEL_R - RIM_BAND;
    const fluxAxHalf = WHEEL_T + CATCH_MARGIN_A;
    const FLUX_WGSL = `@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outCount: atomic<u32>;
@group(0) @binding(2) var<storage, read> speeds: array<f32>;
@group(0) @binding(3) var<uniform> fluxParams: vec4<f32>;
@compute @workgroup_size(${FLUX_WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&positions)) { return; }
    let d = positions[i].xyz - vec3<f32>(${wgslF(WHEEL_C[0])}, ${wgslF(WHEEL_C[1])}, ${wgslF(WHEEL_C[2])});
    let axis = vec3<f32>(${wgslF(WHEEL_AXLE[0])}, ${wgslF(WHEEL_AXLE[1])}, ${wgslF(WHEEL_AXLE[2])});
    let a = dot(d, axis);
    if (abs(a) > ${wgslF(fluxAxHalf)}) { return; }
    let perp = d - a * axis;              // offset within the disk (Y-Z) plane
    if (perp.y <= 0.0) { return; }        // upper half only — skip the submerged lower rim / base pool
    let rad = length(perp);
    if (rad < ${wgslF(fluxRimInner)} || rad > ${wgslF(fluxRadCap)}) { return; } // rim band (water on the buckets)
    if (speeds[i] * fluxParams.x <= ${wgslF(SPEED_GATE)}) { return; }
    atomicAdd(&outCount, 1u);
}`;

    const ensureFluxResources = (): void => {
        if (fluxPipeline) {
            return;
        }
        const device = engine._device;
        const module = device.createShaderModule({ code: FLUX_WGSL });
        fluxPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
        fluxCountBuffer = device.createBuffer({
            label: "marbleTower-wheel-flux-count",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        fluxNormBuffer = device.createBuffer({
            label: "marbleTower-wheel-flux-norm",
            size: 16, // vec4<f32> — only .x used (debugNorm)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        for (let s = 0; s < FLUX_STAGING; s++) {
            fluxStaging.push(device.createBuffer({ label: `marbleTower-wheel-flux-staging${s}`, size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
            fluxStagingBusy.push(false);
        }
    };

    // Dispatch the reduction over the active sim's positions and kick off a non-blocking readback.
    // Skips entirely when both staging buffers are still in flight (reuses latestCount that frame).
    const runFluxPass = (sim: FluidSim): void => {
        ensureFluxResources();
        const slot = fluxStagingBusy.indexOf(false);
        if (slot < 0) {
            return; // both readbacks pending → reuse the last count this frame (no stall)
        }
        const count = sim.count | 0;
        if (count <= 0) {
            return;
        }
        const device = engine._device;
        // Rebuild the bind group only when the active sim's positionBuffer identity changes
        // (PBF↔MLS-MPM switch, or a sim re-created by a particle-count change).
        if (fluxBoundPos !== sim.positionBuffer) {
            fluxBindGroup = device.createBindGroup({
                layout: fluxPipeline!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: sim.positionBuffer } },
                    { binding: 1, resource: { buffer: fluxCountBuffer! } },
                    { binding: 2, resource: { buffer: sim.debugBuffer } },
                    { binding: 3, resource: { buffer: fluxNormBuffer! } },
                ],
            });
            fluxBoundPos = sim.positionBuffer;
            // debugNorm is per-sim (≈1/typical-max-speed); refresh the gate uniform when the sim changes.
            device.queue.writeBuffer(fluxNormBuffer!, 0, new Float32Array([sim.debugNorm, 0, 0, 0]));
        }
        const staging = fluxStaging[slot]!;
        const enc = device.createCommandEncoder({ label: "marbleTower-wheel-flux" });
        enc.clearBuffer(fluxCountBuffer!, 0, 4); // reset the atomic before this frame's reduction
        const pass = enc.beginComputePass();
        pass.setPipeline(fluxPipeline!);
        pass.setBindGroup(0, fluxBindGroup!);
        pass.dispatchWorkgroups(Math.ceil(count / FLUX_WG_SIZE));
        pass.end();
        enc.copyBufferToBuffer(fluxCountBuffer!, 0, staging, 0, 4);
        device.queue.submit([enc.finish()]); // independent submit — decoupled from the core render encoder
        fluxStagingBusy[slot] = true;
        void staging
            .mapAsync(GPUMapMode.READ)
            .then(() => {
                latestCount = new Uint32Array(staging.getMappedRange())[0] ?? 0;
                staging.unmap();
                fluxStagingBusy[slot] = false;
            })
            .catch(() => {
                fluxStagingBusy[slot] = false; // device lost / cancelled — free the slot, keep the stale count
            });
    };

    // Per-frame drive: measure flux (when on + active), smooth it, integrate ω→θ, spin the wheel node.
    const updateWheelSpin = (dt: number): void => {
        if (spinEnabled && active) {
            runFluxPass(ctx.getActiveSim());
        }
        // EMA-smooth the catch count so the spin doesn't jitter with the async readback cadence. The
        // count is speed-gated + restricted to the UPPER rim band (only water riding/striking the top of
        // the wheel), so the base pool the lower rim sits in is excluded.
        const targetCount = spinEnabled ? latestCount : 0;
        smoothedCount += (targetCount - smoothedCount) * Math.min(COUNT_EMA_RATE * dt, 1);
        // Map the count through a deadzone + span into a 0..1 flow fraction, then scale by the drive
        // strength (ω at full flow). Below the deadzone there is too little water to turn the wheel →
        // ωTarget 0 → it coasts to a stop; more water → proportionally faster (up to the safety clamp).
        const flow = Math.max(0, Math.min(1, (smoothedCount - COUNT_DEADZONE) / COUNT_SPAN));
        const omegaTarget = spinEnabled ? Math.min(driveStrength * flow, DRIVE_OMEGA_MAX) : 0;
        wheelOmega += (omegaTarget - wheelOmega) * Math.min(DRIVE_RESPONSIVENESS * dt, 1);
        if (omegaTarget === 0 && wheelOmega < 1e-4) {
            wheelOmega = 0; // fully at rest
        }
        wheelTheta = (wheelTheta + wheelOmega * dt) % TWO_PI; // integrate + wrap
        if (wheelNode) {
            // Spin the "wheel" TransformNode (repurposed as the axle pivot) — negative sense so the
            // wheel turns the natural way for water falling onto it (the axle is local +X).
            wheelNode.rotation.x = -wheelTheta;
        }
        sdfData[35] = wheelTheta; // mirror θ into the wheel block (drives the rotating spokes)
        bakedData[35] = wheelTheta;
        sdfData[37] = wheelOmega; // mirror ω → the spoke moving-boundary finite-difference reads this
        bakedData[37] = wheelOmega;
        // Lightweight tuning read-out (throttled so it doesn't thrash layout every frame).
        if (++spinReadoutFrames >= 12) {
            spinReadoutFrames = 0;
            spinReadoutEl.textContent = `ω ${wheelOmega.toFixed(2)} rad/s · catch ${Math.round(smoothedCount)}`;
        }
    };

    // ── Stage-2 UI (built once; appended near the wheel toggle in extraControls). State persists across
    //    demo re-enters exactly like the other wheel controls (closure vars + re-sync in extraControls). ──
    const spinRow = document.createElement("label");
    spinRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:6px 0 2px;cursor:pointer;";
    const spinChk = document.createElement("input");
    spinChk.type = "checkbox";
    spinChk.checked = spinEnabled;
    const spinText = document.createElement("span");
    spinText.textContent = "Spin water wheel";
    spinRow.append(spinChk, spinText);
    // Toggle the wheel spin (shared by the checkbox handler + restoreState).
    const setSpin = (v: boolean): void => {
        spinEnabled = v; // gates the flux pass (readback stops when off) + ω→0
    };
    spinChk.onchange = (): void => {
        setSpin(spinChk.checked);
    };

    const driveRow = document.createElement("label");
    driveRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0;color:#9fb4cc;font-size:12px;";
    const driveText = document.createElement("span");
    driveText.textContent = "Wheel drive strength";
    const driveSlider = document.createElement("input");
    driveSlider.type = "range";
    driveSlider.min = "0";
    driveSlider.min = "0";
    driveSlider.max = String(DRIVE_OMEGA_MAX);
    driveSlider.step = "0.1";
    driveSlider.value = String(driveStrength);
    driveSlider.style.flex = "1";
    const driveVal = document.createElement("span");
    driveVal.style.cssText = "min-width:34px;text-align:right;";
    driveVal.textContent = driveStrength.toFixed(1);
    driveRow.append(driveText, driveSlider, driveVal);
    // Set the drive strength (shared by the slider handler + restoreState).
    const setDrive = (v: number): void => {
        driveStrength = v;
        driveVal.textContent = driveStrength.toFixed(1);
    };
    driveSlider.oninput = (): void => {
        setDrive(Number(driveSlider.value));
    };

    const spinReadoutEl = document.createElement("div");
    spinReadoutEl.style.cssText = "color:#9fb4cc;font-size:11px;margin:0 0 6px;";
    spinReadoutEl.textContent = "ω 0.00 rad/s · catch 0";

    // ── SDF texture visualizer: a movable cutting plane that reveals the baked mesh-SDF's
    //    cross-section as a coloured slice (warm red/orange = inside the solid, near-white =
    //    the surface contour, cool blue/cyan = open fluid space) — a Unity-style 3D-texture
    //    slice viewer. The slab is read off the retained CPU grid into an RGBA data texture and
    //    mapped 1:1 onto a quad placed exactly on the grid slab in world space. Everything is
    //    gated on a baked grid and only recomputed on user input (toggle / axis / slider) — no
    //    work is added to the hot update() path. ──
    let vizActive = false; // "Visualize SDF texture" toggle (only meaningful once baked)
    let sliceAxis = 2; // 0=X, 1=Y, 2=Z — the axis the cutting plane sweeps along
    let slicePos = 0.5; // normalized 0..1 position of the plane along that axis
    let sliceTexW = 0; // current slice-texture dims; a change (axis switch) forces a re-create
    let sliceTexH = 0;

    const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
    const smoothstep = (e0: number, e1: number, x: number): number => {
        const t = clamp01((x - e0) / (e1 - e0));
        return t * t * (3 - 2 * t);
    };
    // Map one signed distance (world units) to an RGBA byte quad written at offset `o`. <0 is
    // inside the solid (red at the surface → orange deeper), ≈0 is the surface (whitened into a
    // crisp ~1-cell contour band), >0 is open space (blue near the surface → cyan far out).
    // Saturation ramps over ±(8·cell); alpha stays semi-transparent so it reads as a cut plane.
    // This is the one knob to tweak the look.
    const writeSdfColor = (out: Uint8Array, o: number, d: number, cell: number): void => {
        const range = 8 * cell; // distance mapped to full saturation
        let r: number;
        let g: number;
        let b: number;
        if (d < 0) {
            const n = clamp01(-d / range); // 0 at the surface → 1 deep inside the solid
            r = 1;
            g = 0.12 + 0.68 * n; // red at the surface → orange deep inside
            b = 0.04;
        } else {
            const n = clamp01(d / range); // 0 at the surface → 1 far outside
            r = 0.05;
            g = 0.22 + 0.6 * n; // blue near the surface → cyan far out
            b = 1;
        }
        // Whiten toward the zero-crossing so the surface reads as a bright contour band.
        const contour = 1 - smoothstep(0.5 * cell, 1.5 * cell, Math.abs(d));
        r += (1 - r) * contour;
        g += (1 - g) * contour;
        b += (1 - b) * contour;
        out[o] = (r * 255) | 0;
        out[o + 1] = (g * 255) | 0;
        out[o + 2] = (b * 255) | 0;
        out[o + 3] = 200; // semi-transparent cutting plane
    };

    // Unlit textured quad: sample the slice texture and output it straight. Alpha-blended, no
    // depth write, double-sided, reverse-Z compare (this engine renders reverse-Z, so the opaque
    // tower still occludes the plane). System uniforms feed world + viewProjection; the declared
    // `sliceTex` sampler is read as textureSample(sliceTex, sliceTexSampler, uv).
    const SLICE_VS = `struct VertexOutput{@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>,};
@vertex fn mainVertex(input:VertexInput)->VertexOutput{var out:VertexOutput;out.position=shaderSystem.viewProjection*(shaderSystem.world*vec4<f32>(input.position,1.0));out.uv=input.uv;return out;}`;
    const SLICE_FS = `struct VertexOutput{@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>,};
@fragment fn mainFragment(input:VertexOutput)->@location(0) vec4<f32>{return textureSample(sliceTex,sliceTexSampler,input.uv);}`;

    // Create the material + quad EAGERLY, at boot (this factory runs before registerScene). The
    // renderer only wires a material family's single-mesh rebuilder (_rebuildSingle) when that
    // family is present at scene-build time, so a ShaderMaterial plane added lazily AFTER
    // registerScene would never get a renderable. We register it here — hidden, bound to a 1×1
    // placeholder texture so the boot bind-group build succeeds — and swap in the real slice the
    // first time the tool is turned on.
    const sliceMat: ShaderMaterial = createShaderMaterial({
        name: "sdfSlice",
        vertexSource: SLICE_VS,
        fragmentSource: SLICE_FS,
        attributes: ["position", "uv"],
        uniforms: ["world", "viewProjection"],
        samplers: [{ name: "sliceTex", sampleType: "float", viewDimension: "2d" }],
        needAlphaBlending: true,
        blendMode: "alpha",
        depthWrite: false,
        backFaceCulling: false,
        depthCompare: "greater-equal",
    });
    let sliceTex: Texture2D = createTexture2DFromPixels(engine, new Uint8Array(4), 1, 1, { minFilter: "linear", magFilter: "linear" });
    sliceTexW = 1;
    sliceTexH = 1;
    setShaderTexture(sliceMat, "sliceTex", sliceTex);
    const planeMesh = createPlane(engine, {});
    planeMesh.name = "sdfSlicePlane";
    planeMesh.material = sliceMat;
    addToScene(ctx.scene, planeMesh);
    setMeshVisible(planeMesh, false);

    // Show the plane only while THIS demo is on-screen AND the visualizer is on AND the grid is
    // baked. Called from the toggle, onEnter and onLeave.
    const updatePlaneVisibility = (): void => {
        setMeshVisible(planeMesh, active && vizActive && bakeReady);
    };

    // Rebuild the slice image for the current axis/position and place the quad on the world-space
    // slab. No-op until the grid is baked. Reuses the texture in place when only the position
    // changed; re-creates it (releasing the old one) on an axis switch, since the cross-section's
    // pixel dimensions change with the axis.
    const refreshSlice = (): void => {
        if (!bakedGrid) {
            return;
        }
        const grid = bakedGrid.data;
        const [dimX, dimY, dimZ] = bakedGrid.dims;
        const [ox, oy, oz] = bakedGrid.origin;
        const cell = bakedGrid.cellSize;
        const dimA = bakedGrid.dims[sliceAxis]!;
        const idx = Math.max(0, Math.min(dimA - 1, Math.round(slicePos * (dimA - 1))));

        // The cross-section spans the OTHER two axes (u,v). Each (col,row) → (u,v) below is chosen
        // to exactly match the quad's UV→world mapping further down, so the slice is never
        // flipped/rotated relative to the geometry. Data is row-major (col fastest), top-to-bottom.
        let uSize: number;
        let vSize: number;
        if (sliceAxis === 2) {
            // Z-slab (k = idx): u=X (i, size dimX), v=Y (j, size dimY). Same memory order as the
            // grid's own contiguous XY plane at fixed k (index = i + j*dimX + idx*dimX*dimY).
            uSize = dimX;
            vSize = dimY;
        } else if (sliceAxis === 0) {
            // X-slab (i = idx): u=Z (k, size dimZ), v=Y (j, size dimY).
            uSize = dimZ;
            vSize = dimY;
        } else {
            // Y-slab (j = idx): u=X (i, size dimX), v=Z (k, size dimZ).
            uSize = dimX;
            vSize = dimZ;
        }
        const px = new Uint8Array(uSize * vSize * 4);
        if (sliceAxis === 2) {
            const kBase = idx * dimX * dimY;
            for (let j = 0; j < dimY; j++) {
                const row = j * dimX;
                for (let i = 0; i < dimX; i++) {
                    writeSdfColor(px, (row + i) * 4, grid[kBase + row + i]!, cell);
                }
            }
        } else if (sliceAxis === 0) {
            for (let j = 0; j < dimY; j++) {
                for (let k = 0; k < dimZ; k++) {
                    writeSdfColor(px, (j * dimZ + k) * 4, grid[idx + j * dimX + k * dimX * dimY]!, cell);
                }
            }
        } else {
            const jBase = idx * dimX;
            for (let k = 0; k < dimZ; k++) {
                const kb = k * dimX * dimY;
                for (let i = 0; i < dimX; i++) {
                    writeSdfColor(px, (k * dimX + i) * 4, grid[i + jBase + kb]!, cell);
                }
            }
        }

        if (sliceTexW !== uSize || sliceTexH !== vSize) {
            releaseTexture(sliceTex); // pair the acquire inside createTexture2DFromPixels (old dims)
            sliceTex = createTexture2DFromPixels(engine, px, uSize, vSize, { minFilter: "linear", magFilter: "linear" });
            sliceTexW = uSize;
            sliceTexH = vSize;
            setShaderTexture(sliceMat, "sliceTex", sliceTex);
        } else {
            updateTexture2DFromPixels(engine, sliceTex, px);
        }

        // Place + orient the unit quad (default: XY plane, faces -Z, spans [-0.5,0.5]², uv.u∝+X,
        // uv.v∝+Y) so it covers the slab's (u,v) rectangle in world space. size[a] = dims[a]·cell;
        // the slab's world coordinate along `axis` is origin[axis] + idx·cell.
        const sizeX = dimX * cell;
        const sizeY = dimY * cell;
        const sizeZ = dimZ * cell;
        if (sliceAxis === 2) {
            // No rotation: local X→world X (u→i), local Y→world Y (v→j), at fixed Z.
            planeMesh.rotation.set(0, 0, 0);
            planeMesh.scaling.set(sizeX, sizeY, 1);
            planeMesh.position.set(ox + sizeX / 2, oy + sizeY / 2, oz + idx * cell);
        } else if (sliceAxis === 0) {
            // Rotate −90° about Y: local X→world +Z (u→k), local Y→world +Y (v→j), at fixed X.
            planeMesh.rotation.set(0, -Math.PI / 2, 0);
            planeMesh.scaling.set(sizeZ, sizeY, 1);
            planeMesh.position.set(ox + idx * cell, oy + sizeY / 2, oz + sizeZ / 2);
        } else {
            // Rotate +90° about X: local X→world +X (u→i), local Y→world +Z (v→k), at fixed Y.
            planeMesh.rotation.set(Math.PI / 2, 0, 0);
            planeMesh.scaling.set(sizeX, sizeZ, 1);
            planeMesh.position.set(ox + sizeX / 2, oy + idx * cell, oz + sizeZ / 2);
        }
    };

    const applyVisualizer = (): void => {
        if (vizActive) {
            refreshSlice();
        }
        updatePlaneVisibility();
    };

    // Visualizer UI (appended under the baked toggle in extraControls). Everything stays disabled
    // until the grid is baked; the state (vizActive / sliceAxis / slicePos) persists across demo
    // re-enters exactly like the baked toggle does.
    const vizRow = document.createElement("label");
    vizRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:6px 0 2px;cursor:pointer;";
    const vizChk = document.createElement("input");
    vizChk.type = "checkbox";
    vizChk.disabled = true; // enabled once the mesh SDF is baked
    const vizText = document.createElement("span");
    vizText.textContent = "Visualize SDF texture";
    vizRow.append(vizChk, vizText);

    const axisRow = document.createElement("label");
    axisRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0;color:#9fb4cc;font-size:12px;";
    const axisText = document.createElement("span");
    axisText.textContent = "Slice axis";
    const axisSel = document.createElement("select");
    axisSel.disabled = true;
    const axisNames = ["X", "Y", "Z"];
    for (let a = 0; a < axisNames.length; a++) {
        const opt = document.createElement("option");
        opt.value = String(a);
        opt.textContent = axisNames[a]!;
        axisSel.append(opt);
    }
    axisSel.value = String(sliceAxis);
    axisRow.append(axisText, axisSel);

    const sliceRow = document.createElement("label");
    sliceRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0 6px;color:#9fb4cc;font-size:12px;";
    const sliceText = document.createElement("span");
    sliceText.textContent = "Slice position";
    const sliceSlider = document.createElement("input");
    sliceSlider.type = "range";
    sliceSlider.min = "0";
    sliceSlider.max = "1";
    sliceSlider.step = "0.01";
    sliceSlider.value = String(slicePos);
    sliceSlider.disabled = true;
    sliceSlider.style.flex = "1";
    sliceRow.append(sliceText, sliceSlider);

    vizChk.onchange = (): void => {
        if (vizChk.checked && !bakeReady) {
            vizChk.checked = false; // bake not ready → nothing to visualize
            return;
        }
        vizActive = vizChk.checked;
        applyVisualizer();
    };
    axisSel.onchange = (): void => {
        sliceAxis = Number(axisSel.value) | 0;
        if (vizActive) {
            refreshSlice();
        }
    };
    sliceSlider.oninput = (): void => {
        slicePos = Number(sliceSlider.value);
        if (vizActive) {
            refreshSlice();
        }
    };

    // ── Analytic wheel SDF DEBUG viz: a face-on, colour-coded picture of what `sdWheel` actually
    //    looks like (hub + N rotating spokes + the M perimeter bucket vanes / pockets), evaluated
    //    on the CPU and overlaid on the REAL wheel so the user can directly compare gap counts. The
    //    wheel is EXCLUDED from the baked grid, so the SDF-texture slice viewer above can't show it
    //    — this evaluates the ANALYTIC sdWheel directly. Reuses the EXACT slice-viewer infra (the
    //    writeSdfColor ramp, createTexture2DFromPixels + the textured-quad ShaderMaterial). Works in
    //    BOTH analytic and baked mode (it does not depend on the bake). ──
    // CPU port of the WGSL `sdWheel`, evaluated in the wheel's axle frame: perpendicular offset
    // (perpY,perpZ) in the disk plane (world Y-Z), axial offset along the axle (world +X), at wheel
    // angle `theta`. Mirrors the WGSL exactly — hub capped cylinder, the polar-repeat spoke bar,
    // and the perimeter bucket band (sole ring + shrouds + M rotating vanes), all with the same
    // fold + half-widths and the -θ phase convention (dt=0 here → phase = -θ; φ0 = spokePhase0).
    const wheelSdfCpu = (perpY: number, perpZ: number, axial: number, theta: number, nSpokes: number, spokeHW: number, nBuckets: number, bucketPh0: number): number => {
        const a = axial;
        const rad = Math.hypot(perpY, perpZ);
        const rInner = WHEEL_HUB_R;
        const rOuter = WHEEL_R;
        const T = WHEEL_T;
        // Bucket-band radii (spokes stop at bInner = rim inner; the band [bInner,bOuter] is the rim+buckets).
        const bOuter = rOuter + T;
        const bInner = bOuter - BUCKET_DEPTH;
        const bMid = 0.5 * (bInner + bOuter);
        const bHalfR = 0.5 * (bOuter - bInner);
        const bHalfA = T;
        // Hub / axle capped cylinder (rotation-symmetric, static).
        const qx = rad - rInner;
        const qy = Math.abs(a - WHEEL_HUB_OFFSET) - WHEEL_HUB_HALF;
        const hub = Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0);
        // Disk-plane angle + shared moving-boundary phase; the debug viz uses dt=0 → phase = -θ.
        const phi = Math.atan2(perpZ, perpY);
        const spokePhase = -theta;
        // N radial spokes, polar-repeated, ending at the rim inner (bInner).
        const n = Math.max(nSpokes, 1);
        const sector = TWO_PI / n;
        let rel = phi - spokePhase0 - spokePhase;
        rel = rel - sector * Math.round(rel / sector);
        const midR = 0.5 * (rInner + bInner); // spokes stop at the rim inner (like the real mesh)
        const halfLenR = 0.5 * (bInner - rInner);
        const sa = Math.abs(a) - SPOKE_HALF_AX;
        const st = Math.abs(rad * rel) - spokeHW;
        const sr = Math.abs(rad - midR) - halfLenR;
        const spoke = Math.hypot(Math.max(sa, 0), Math.max(st, 0), Math.max(sr, 0)) + Math.min(Math.max(sa, Math.max(st, sr)), 0);
        // Bucket band: sole (inner floor ring) + two side shrouds + M rotating radial vanes.
        const soleRad = Math.abs(rad - bInner) - BUCKET_WALL;
        const soleAx = Math.abs(a) - bHalfA;
        const sole = Math.hypot(Math.max(soleRad, 0), Math.max(soleAx, 0)) + Math.min(Math.max(soleRad, soleAx), 0);
        const shrAx = Math.abs(Math.abs(a) - bHalfA) - BUCKET_WALL;
        const shrRad = Math.abs(rad - bMid) - bHalfR;
        const shroud = Math.hypot(Math.max(shrRad, 0), Math.max(shrAx, 0)) + Math.min(Math.max(shrRad, shrAx), 0);
        const m = Math.max(nBuckets, 1);
        const bSector = TWO_PI / m;
        let brel = phi - bucketPh0 - spokePhase; // buckets use their OWN 30-fold phase
        brel = brel - bSector * Math.round(brel / bSector);
        const vAx = Math.abs(a) - bHalfA;
        const vT = Math.abs(rad * brel) - VANE_HALF_W;
        const vR = Math.abs(rad - bMid) - bHalfR;
        const vane = Math.hypot(Math.max(vAx, 0), Math.max(vT, 0), Math.max(vR, 0)) + Math.min(Math.max(vAx, Math.max(vT, vR)), 0);
        return Math.min(Math.min(Math.min(sole, shroud), vane), Math.min(hub, spoke));
    };

    // Face-on debug quad: normal along the axle (world +X), spanning ±(WHEEL_R + margin) in the
    // world Y-Z plane, centred on WHEEL_C. Placement mirrors the slice viewer's X-slab case
    // (rotation −90° about Y → local X→world +Z (u), local Y→world +Y (v), normal→world −X), so the
    // colour map is un-mirrored vs. the real wheel (the one verified orientation, reused verbatim).
    const DBG_RES = 160; // CPU eval grid resolution (both axes)
    const DBG_MARGIN = 0.6; // world-unit slack beyond the rim radius so the whole ring is inside
    let dbgActive = false; // "Debug wheel SDF" toggle
    let dbgTexN = 1; // current debug-texture side (forces a re-create on the first real build)
    const dbgPx = new Uint8Array(DBG_RES * DBG_RES * 4); // reused RGBA scratch
    // A second textured quad, its OWN material + texture (a ShaderMaterial can only carry one slice
    // texture, so the debug quad can't share sliceMat). Created eagerly at boot — like sliceMat —
    // so the ShaderMaterial family's single-mesh rebuilder is wired at scene-build time.
    const dbgMat: ShaderMaterial = createShaderMaterial({
        name: "sdfWheelDbg",
        vertexSource: SLICE_VS,
        fragmentSource: SLICE_FS,
        attributes: ["position", "uv"],
        uniforms: ["world", "viewProjection"],
        samplers: [{ name: "sliceTex", sampleType: "float", viewDimension: "2d" }],
        needAlphaBlending: true,
        blendMode: "alpha",
        depthWrite: false,
        backFaceCulling: false,
        depthCompare: "greater-equal",
    });
    let dbgTex: Texture2D = createTexture2DFromPixels(engine, new Uint8Array(4), 1, 1, { minFilter: "linear", magFilter: "linear" });
    setShaderTexture(dbgMat, "sliceTex", dbgTex);
    const dbgPlane = createPlane(engine, {});
    dbgPlane.name = "sdfWheelDebugPlane";
    dbgPlane.material = dbgMat;
    addToScene(ctx.scene, dbgPlane);
    setMeshVisible(dbgPlane, false);

    // ── Flux-region visualiser (shown with "Debug wheel SDF") ──────────────────────────────────
    // A translucent green half-annular prism outlining EXACTLY the region the flux reduction counts
    // to derive the wheel's drive: the UPPER half (perp.y>0) of the annulus [fluxRimInner, fluxRadCap]
    // around the axle through WHEEL_C, axially |x-Cx| ≤ fluxAxHalf. Only MOVING water inside this box
    // spins the wheel — so if the pour lands outside it, the wheel won't turn (this makes that visible).
    const FLUX_VS = `struct VertexOutput{@builtin(position) position:vec4<f32>,};
@vertex fn mainVertex(input:VertexInput)->VertexOutput{var out:VertexOutput;out.position=shaderSystem.viewProjection*(shaderSystem.world*vec4<f32>(input.position,1.0));return out;}`;
    const FLUX_FS = `@fragment fn mainFragment()->@location(0) vec4<f32>{return vec4<f32>(0.15,1.0,0.35,0.24);}`;
    const fluxMat: ShaderMaterial = createShaderMaterial({
        name: "wheelFluxRegion",
        vertexSource: FLUX_VS,
        fragmentSource: FLUX_FS,
        attributes: ["position"],
        uniforms: ["world", "viewProjection"],
        needAlphaBlending: true,
        blendMode: "alpha",
        depthWrite: false,
        backFaceCulling: false,
        depthCompare: "greater-equal",
    });
    // Build the half-annular prism (local coords about WHEEL_C; axle = local/world +X). φ∈[-π/2,π/2]
    // so local Y = r·cos φ ≥ 0 → the +Y (upper) half, matching the reduction's perp.y>0 gate.
    const buildFluxRegion = (rIn: number, rOut: number, axHalf: number, seg: number): { positions: Float32Array; indices: Uint32Array } => {
        const p: number[] = [];
        const idx: number[] = [];
        const inX0: number[] = [], outX0: number[] = [], inX1: number[] = [], outX1: number[] = [];
        const push = (x: number, y: number, z: number): number => { p.push(x, y, z); return p.length / 3 - 1; };
        for (let j = 0; j <= seg; j++) {
            const phi = -Math.PI / 2 + Math.PI * (j / seg);
            const cy = Math.cos(phi), sy = Math.sin(phi);
            inX0.push(push(-axHalf, rIn * cy, rIn * sy));
            outX0.push(push(-axHalf, rOut * cy, rOut * sy));
            inX1.push(push(axHalf, rIn * cy, rIn * sy));
            outX1.push(push(axHalf, rOut * cy, rOut * sy));
        }
        const quad = (a: number, b: number, c: number, d: number): void => { idx.push(a, b, c, a, c, d); };
        for (let j = 0; j < seg; j++) {
            quad(outX0[j]!, outX0[j + 1]!, outX1[j + 1]!, outX1[j]!); // outer wall
            quad(inX1[j]!, inX1[j + 1]!, inX0[j + 1]!, inX0[j]!);     // inner wall
            quad(inX0[j]!, outX0[j]!, outX0[j + 1]!, inX0[j + 1]!);   // −X annular cap
            quad(inX1[j + 1]!, outX1[j + 1]!, outX1[j]!, inX1[j]!);   // +X annular cap
        }
        quad(inX0[0]!, inX1[0]!, outX1[0]!, outX0[0]!);             // φ=-π/2 end cap
        quad(outX0[seg]!, outX1[seg]!, inX1[seg]!, inX0[seg]!);     // φ=+π/2 end cap
        return { positions: new Float32Array(p), indices: new Uint32Array(idx) };
    };
    const fluxGeo = buildFluxRegion(fluxRimInner, fluxRadCap, fluxAxHalf, 48);
    const fluxMesh = createMeshFromData(engine, "wheelFluxRegion", fluxGeo.positions, new Float32Array(fluxGeo.positions.length), fluxGeo.indices);
    fluxMesh.name = "wheelFluxRegion";
    fluxMesh.material = fluxMat;
    addToScene(ctx.scene, fluxMesh);
    fluxMesh.position.set(WHEEL_C[0], WHEEL_C[1], WHEEL_C[2]);
    setMeshVisible(fluxMesh, false);

    // Rebuild the debug image for the CURRENT wheel θ / N / thickness and keep the quad glued to the
    // wheel disk (WHEEL_C, normal along the axle). Cheap (160×160 CPU evals) + throttled (update()).
    const refreshWheelDebug = (): void => {
        const half = WHEEL_R + DBG_MARGIN;
        const sizeD = 2 * half;
        const step = sizeD / (DBG_RES - 1); // world units per texel (≈ the disk cell for the ramp)
        for (let j = 0; j < DBG_RES; j++) {
            const perpY = -half + j * step; // v → world +Y
            const rowBase = j * DBG_RES;
            for (let k = 0; k < DBG_RES; k++) {
                const perpZ = -half + k * step; // u → world +Z
                const dist = wheelSdfCpu(perpY, perpZ, 0, wheelTheta, spokeCount, spokeHalfW, bucketCount, bucketPhase0);
                writeSdfColor(dbgPx, (rowBase + k) * 4, dist, step);
            }
        }
        if (dbgTexN !== DBG_RES) {
            releaseTexture(dbgTex); // pair the acquire inside createTexture2DFromPixels (old dims)
            dbgTex = createTexture2DFromPixels(engine, dbgPx, DBG_RES, DBG_RES, { minFilter: "linear", magFilter: "linear" });
            dbgTexN = DBG_RES;
            setShaderTexture(dbgMat, "sliceTex", dbgTex);
        } else {
            updateTexture2DFromPixels(engine, dbgTex, dbgPx);
        }
        // Face-on to the disk at WHEEL_C, normal along the axle (world −X after the −90° Y turn).
        dbgPlane.rotation.set(0, -Math.PI / 2, 0);
        dbgPlane.scaling.set(sizeD, sizeD, 1);
        dbgPlane.position.set(WHEEL_C[0], WHEEL_C[1], WHEEL_C[2]);
    };

    const updateDbgVisibility = (): void => {
        setMeshVisible(dbgPlane, active && dbgActive);
        setMeshVisible(fluxMesh, active && dbgActive); // flux region shown alongside the debug quad
    };

    // ── "Debug wheel SDF" toggle + the "Wheel spokes" / "Spoke thickness" sliders ──────────────
    const dbgRow = document.createElement("label");
    dbgRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:6px 0 2px;cursor:pointer;";
    const dbgChk = document.createElement("input");
    dbgChk.type = "checkbox";
    const dbgText = document.createElement("span");
    dbgText.textContent = "Debug wheel SDF";
    dbgRow.append(dbgChk, dbgText);

    const spokeRow = document.createElement("label");
    spokeRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0;color:#9fb4cc;font-size:12px;";
    const spokeText = document.createElement("span");
    spokeText.textContent = "Wheel spokes";
    const spokeSlider = document.createElement("input");
    spokeSlider.type = "range";
    spokeSlider.min = "6";
    spokeSlider.max = "48";
    spokeSlider.step = "1";
    spokeSlider.value = String(spokeCount);
    spokeSlider.style.flex = "1";
    const spokeVal = document.createElement("span");
    spokeVal.style.cssText = "min-width:24px;text-align:right;";
    spokeVal.textContent = String(spokeCount);
    spokeRow.append(spokeText, spokeSlider, spokeVal);

    const thickRow = document.createElement("label");
    thickRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0 6px;color:#9fb4cc;font-size:12px;";
    const thickText = document.createElement("span");
    thickText.textContent = "Spoke thickness";
    const thickSlider = document.createElement("input");
    thickSlider.type = "range";
    thickSlider.min = "0.05";
    thickSlider.max = "0.4";
    thickSlider.step = "0.01";
    thickSlider.value = String(spokeHalfW);
    thickSlider.style.flex = "1";
    const thickVal = document.createElement("span");
    thickVal.style.cssText = "min-width:34px;text-align:right;";
    thickVal.textContent = spokeHalfW.toFixed(2);
    thickRow.append(thickText, thickSlider, thickVal);

    const bucketRow = document.createElement("label");
    bucketRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0 6px;color:#9fb4cc;font-size:12px;";
    const bucketText = document.createElement("span");
    bucketText.textContent = "Wheel buckets";
    const bucketSlider = document.createElement("input");
    bucketSlider.type = "range";
    bucketSlider.min = "12";
    bucketSlider.max = "48";
    bucketSlider.step = "1";
    bucketSlider.value = String(bucketCount);
    bucketSlider.style.flex = "1";
    const bucketVal = document.createElement("span");
    bucketVal.style.cssText = "min-width:24px;text-align:right;";
    bucketVal.textContent = String(bucketCount);
    bucketRow.append(bucketText, bucketSlider, bucketVal);

    // Live setters shared by the sliders + restoreState — re-pack the wheel block for BOTH modes
    // (collision) and refresh the debug picture (viz) so the two stay in lockstep.
    const setSpokeCount = (v: number): void => {
        spokeCount = Math.max(1, Math.round(v));
        packWheel(sdfData); // N → float 38 in BOTH blocks
        packWheel(bakedData);
        writeSdfParams();
        spokeVal.textContent = String(spokeCount);
        if (active && dbgActive) {
            refreshWheelDebug();
        }
    };
    const setSpokeThickness = (v: number): void => {
        spokeHalfW = v;
        packWheel(sdfData); // spokeHalfW → wheelA.x (float 28) in BOTH blocks
        packWheel(bakedData);
        writeSdfParams();
        thickVal.textContent = spokeHalfW.toFixed(2);
        if (active && dbgActive) {
            refreshWheelDebug();
        }
    };
    const setBucketCount = (v: number): void => {
        bucketCount = Math.max(1, Math.round(v));
        packWheel(sdfData); // M → wheelA.y (float 29) in BOTH blocks
        packWheel(bakedData);
        writeSdfParams();
        bucketVal.textContent = String(bucketCount);
        if (active && dbgActive) {
            refreshWheelDebug();
        }
    };

    dbgChk.onchange = (): void => {
        dbgActive = dbgChk.checked;
        if (dbgActive) {
            refreshWheelDebug(); // build the picture immediately (don't wait for the throttle)
        }
        updateDbgVisibility();
    };
    spokeSlider.oninput = (): void => {
        setSpokeCount(Number(spokeSlider.value));
    };
    thickSlider.oninput = (): void => {
        setSpokeThickness(Number(thickSlider.value));
    };
    bucketSlider.oninput = (): void => {
        setBucketCount(Number(bucketSlider.value));
    };

    // Merge every tower mesh's CPU geometry into ONE world-space triangle soup, bake it into a
    // signed-distance grid, and upload it to a storage buffer the sims sample. One-time
    // (~sub-second) synchronous cost right after the async load; logged. Non-fatal on failure.
    const bakeSceneSdf = (): void => {
        // 1) Gather meshes that actually carry CPU positions + indices — EXCLUDING the water wheel
        //    mesh (it spins, so its world transform is dynamic; the fluid collides with the analytic
        //    wheel SDF instead). Log how many tris the wheel had so the exclusion is visible.
        let totalV = 0;
        let totalI = 0;
        let wheelTris = 0;
        const usable: Mesh[] = [];
        for (const mesh of towerMeshes) {
            const cm = mesh as CpuMeshNode;
            if (mesh === wheelMesh) {
                wheelTris = cm._cpuIndices ? cm._cpuIndices.length / 3 : 0;
                continue; // the whole wheel mesh is skipped from the bake
            }
            if (!cm._cpuPositions || cm._cpuPositions.length === 0 || !cm._cpuIndices || cm._cpuIndices.length === 0) {
                continue;
            }
            usable.push(mesh);
            totalV += cm._cpuPositions.length;
            totalI += cm._cpuIndices.length;
        }
        if (usable.length === 0) {
            // eslint-disable-next-line no-console
            console.warn("[marbleTower] no CPU geometry to bake — staying analytic");
            bakeStatus = "no CPU geometry — analytic only";
            bakedStatusEl.textContent = bakeStatus;
            return;
        }

        // 2) Concat WORLD positions (local × worldMatrix) + vertex-offset indices, tracking the
        //    merged AABB (column-major Mat4, same transform as worldAabb).
        const mergedPos = new Float32Array(totalV);
        const mergedIdx = new Uint32Array(totalI);
        let pOff = 0;
        let iOff = 0;
        let vBase = 0;
        let minX = Infinity,
            minY = Infinity,
            minZ = Infinity,
            maxX = -Infinity,
            maxY = -Infinity,
            maxZ = -Infinity;
        for (const mesh of usable) {
            const cm = mesh as CpuMeshNode;
            const src = cm._cpuPositions!;
            const idx = cm._cpuIndices!;
            const w = mesh.worldMatrix;
            const m0 = w[0]!,
                m1 = w[1]!,
                m2 = w[2]!,
                m4 = w[4]!,
                m5 = w[5]!,
                m6 = w[6]!,
                m8 = w[8]!,
                m9 = w[9]!,
                m10 = w[10]!,
                m12 = w[12]!,
                m13 = w[13]!,
                m14 = w[14]!;
            for (let i = 0; i < src.length; i += 3) {
                const lx = src[i]!,
                    ly = src[i + 1]!,
                    lz = src[i + 2]!;
                const x = m0 * lx + m4 * ly + m8 * lz + m12;
                const y = m1 * lx + m5 * ly + m9 * lz + m13;
                const z = m2 * lx + m6 * ly + m10 * lz + m14;
                mergedPos[pOff++] = x;
                mergedPos[pOff++] = y;
                mergedPos[pOff++] = z;
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                minZ = Math.min(minZ, z);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
                maxZ = Math.max(maxZ, z);
            }
            const vCount = src.length / 3;
            for (let i = 0; i < idx.length; i += 3) {
                mergedIdx[iOff++] = idx[i]! + vBase; // Uint16→Uint32 normalised by the target array
                mergedIdx[iOff++] = idx[i + 1]! + vBase;
                mergedIdx[iOff++] = idx[i + 2]! + vBase;
            }
            vBase += vCount;
        }

        // 3) Bake (SDFGen narrow-band + 1 fast-sweep round — sub-second at this size) + upload.
        try {
            const t0 = performance.now();
            const grid = generateMeshSdf(mergedPos, mergedIdx.subarray(0, iOff), {
                min: [minX, minY, minZ],
                max: [maxX, maxY, maxZ],
                cellSize: SDF_CELL_SIZE,
                padding: SDF_PADDING,
                sweepPasses: 1,
            });
            const bakeMs = performance.now() - t0;

            const buf = engine._device.createBuffer({
                label: "marbleTower-sdf-grid",
                size: grid.data.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            engine._device.queue.writeBuffer(buf, 0, grid.data as Float32Array<ArrayBuffer>);
            gridBuffer = buf;

            // Retain the raw distances on the CPU for the SDF-texture visualizer (a movable slice
            // plane). Copies references / spreads the dims + origin into fixed tuples.
            bakedGrid = {
                data: grid.data,
                dims: [grid.dims[0], grid.dims[1], grid.dims[2]],
                origin: [grid.origin[0], grid.origin[1], grid.origin[2]],
                cellSize: grid.cellSize,
            };

            // 4) Pack grid params into the baked block (floats 8..15).
            bakedData[8] = grid.origin[0];
            bakedData[9] = grid.origin[1];
            bakedData[10] = grid.origin[2];
            bakedData[11] = 1 / grid.cellSize;
            bakedData[12] = grid.dims[0];
            bakedData[13] = grid.dims[1];
            bakedData[14] = grid.dims[2];
            bakedData[15] = 0;

            bakeReady = true;
            bakedChk.disabled = false;
            // A preset / restored state may have requested baked mode BEFORE the bake finished
            // (desiredBaked). Honour it now that the grid exists, and sync the checkbox.
            if (desiredBaked) {
                setBakedMode(true);
                bakedChk.checked = true;
            }
            // The visualizer can now slice the grid — enable its controls (extraControls() also
            // re-syncs these if the bake finished while the demo was off-screen).
            vizChk.disabled = false;
            axisSel.disabled = false;
            sliceSlider.disabled = false;
            bakeStatus = `${(iOff / 3) | 0} tris → ${grid.dims[0]}×${grid.dims[1]}×${grid.dims[2]} grid, ${bakeMs.toFixed(0)} ms`;
            bakedStatusEl.textContent = bakeStatus;
            // eslint-disable-next-line no-console
            console.warn(
                `[marbleTower] baked mesh SDF: ${bakeStatus} | wheel-excluded ${wheelTris} tris | ` +
                    `origin=[${grid.origin.map((v) => v.toFixed(2)).join(", ")}] cell=${grid.cellSize}`
            );
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[marbleTower] mesh SDF bake failed — staying analytic", err);
            bakeStatus = "bake failed — analytic only";
            bakedStatusEl.textContent = bakeStatus;
        }
    };

    // Detect the wheel's spoke count N (and base phase φ0) from the mesh geometry so the collision
    // spokes track the VISIBLE wheel. Project every wheel-mesh vertex into the disk plane (world
    // frame, perpendicular to the world +X axle, around WHEEL_C), restrict to a rim-inclusive band
    // (the marbleTower wheel carries its N-fold symmetry on the rim; the mid-span is nearly empty),
    // and take the rotational-symmetry order via a Fourier scan: N maximizes |Σ exp(i·N·φ)|. The
    // argument of that sum gives N·φ0, so φ0 = arg/N. Falls back to N=12 when the peak is weak.
    const detectSpokes = (): void => {
        if (!wheelMesh) {
            return;
        }
        const pos = (wheelMesh as CpuMeshNode)._cpuPositions;
        if (!pos || pos.length === 0) {
            return;
        }
        const w = wheelMesh.worldMatrix; // column-major; θ=0 pose (detected before repurposeWheelPivot)
        // Only the Y/Z rows are needed — the disk is the world Y-Z plane (axle = world +X).
        const m1 = w[1]!,
            m2 = w[2]!,
            m5 = w[5]!,
            m6 = w[6]!,
            m9 = w[9]!,
            m10 = w[10]!,
            m13 = w[13]!,
            m14 = w[14]!;
        const cy = WHEEL_C[1];
        const cz = WHEEL_C[2];
        const rLo = WHEEL_HUB_R + 0.5; // exclude the hub / axle shaft
        const rHi = WHEEL_R + WHEEL_T + 0.5; // include the whole rim band
        const NMIN = 3;
        const NMAX = 36; // extend past the 30-fold bucket structure so its phase can be read too
        const re = new Float64Array(NMAX + 1);
        const im = new Float64Array(NMAX + 1);
        let cnt = 0;
        for (let i = 0; i < pos.length; i += 3) {
            const lx = pos[i]!,
                ly = pos[i + 1]!,
                lz = pos[i + 2]!;
            const y = m1 * lx + m5 * ly + m9 * lz + m13;
            const z = m2 * lx + m6 * ly + m10 * lz + m14;
            const dy = y - cy;
            const dz = z - cz;
            const rad = Math.hypot(dy, dz);
            if (rad < rLo || rad > rHi) {
                continue;
            }
            const ang = Math.atan2(dz, dy); // matches sdWheel's atan2(perp.z, perp.y)
            cnt++;
            for (let n = NMIN; n <= NMAX; n++) {
                re[n]! += Math.cos(n * ang);
                im[n]! += Math.sin(n * ang);
            }
        }
        let bestN = 12;
        let bestMag = 0;
        let bestPhase = 0;
        if (cnt > 0) {
            for (let n = NMIN; n <= 18; n++) { // spoke fundamental (12) lives here — exclude its 24/36 harmonics
                const mag = Math.hypot(re[n]!, im[n]!) / cnt;
                if (mag > bestMag) {
                    bestMag = mag;
                    bestN = n;
                    bestPhase = Math.atan2(im[n]!, re[n]!) / n;
                }
            }
        }
        // N is the "Wheel spokes" slider's value — NOT overwritten here, since the spoke MIDDLES
        // carry no vertices and the Fourier count is unreliable for this wheel. We keep the peak
        // only as a diagnostic hint and seed the spoke base phase φ0 from it. The 30-fold buckets
        // get their OWN base phase from the n=bucketCount harmonic (independent of the 12 spokes).
        if (bestMag >= 0.15) {
            spokePhase0 = bestPhase;
        } else {
            spokePhase0 = 0;
        }
        if (cnt > 0 && bucketCount >= NMIN && bucketCount <= NMAX) {
            const bMagN = Math.hypot(re[bucketCount]!, im[bucketCount]!) / cnt;
            bucketPhase0 = bMagN >= 0.05 ? Math.atan2(im[bucketCount]!, re[bucketCount]!) / bucketCount : spokePhase0;
        } else {
            bucketPhase0 = spokePhase0;
        }
        packWheel(sdfData); // push φ0 (float 39) into BOTH blocks (N stays the slider value)
        packWheel(bakedData);
        writeSdfParams();
        // eslint-disable-next-line no-console
        console.warn(
            `[marbleTower] wheel spokes: geometry peak N=${bestN} mag=${bestMag.toFixed(3)} (${cnt} rim verts) ` +
                `— hint only; slider N=${spokeCount}, phase0=${spokePhase0.toFixed(4)}, bucketPhase0=${bucketPhase0.toFixed(4)} (dial N by eye)`
        );
    };

    // Repurpose the glTF "wheel" TransformNode (identity transform) as the spin PIVOT, using the
    // REAL textured wheel mesh — no re-created disk, no geometry split. The node's pivot sits at the
    // mesh's local ORIGIN, which is NOT on the axle; the axle is the local-X line through the disk
    // centre (Cy,Cz) computed from the mesh's local geometry (Cy=(minY+maxY)/2, Cz=(minZ+maxZ)/2;
    // the axle runs along local X so its X is irrelevant → use 0). We move the node to that axle
    // point and counter-translate the mesh by the negative, so at θ=0 the world transform is
    // unchanged (parent·T(0,Cy,Cz)·T(0,−Cy,−Cz) = parent·I) and a rotation about the node's local X
    // (wheelNode.rotation.x = θ) spins the disk IN PLACE about the axle line. Offsets are in the
    // wheel node's local frame (glTF units, same as _cpuPositions).
    const repurposeWheelPivot = (): void => {
        if (!wheelNode || !wheelMesh) {
            // eslint-disable-next-line no-console
            console.warn("[marbleTower] wheel pivot: 'wheel' node/mesh not found — spin disabled");
            return;
        }
        const pos = (wheelMesh as CpuMeshNode)._cpuPositions;
        if (!pos || pos.length === 0) {
            return;
        }
        let minY = Infinity,
            maxY = -Infinity,
            minZ = Infinity,
            maxZ = -Infinity;
        for (let i = 0; i < pos.length; i += 3) {
            const y = pos[i + 1]!,
                z = pos[i + 2]!;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }
        const cy = (minY + maxY) / 2;
        const cz = (minZ + maxZ) / 2;
        wheelNode.position.set(0, cy, cz); // move the pivot node onto the axle line
        wheelMesh.position.set(0, -cy, -cz); // counter-translate the mesh → θ=0 is a no-op
        // eslint-disable-next-line no-console
        console.warn(`[marbleTower] wheel pivot: axle at local (Cy=${cy.toFixed(1)}, Cz=${cz.toFixed(1)}) — node +offset, mesh −offset`);
    };

    void (async (): Promise<void> => {
        try {
            const asset = await loadGltf(engine, TOWER_URL);
            // Static display — drop any (marble-run) animation clips so the tower holds a
            // fixed pose the analytic SDF can match.
            asset.animationGroups = undefined;
            addToScene(ctx.scene, asset);
            const root = asset.entities[0] as SceneNode;
            root.rotation.set(0, 0, 0);
            collectMeshes(root, towerMeshes);
            if (towerMeshes.length === 0) {
                // eslint-disable-next-line no-console
                console.warn("[marbleTower] asset loaded but contains no meshes");
                return;
            }

            // Locate the glTF "wheel" TransformNode by name; its child Mesh leaf (with `_gpu`) is
            // the real textured water-wheel geometry. Kept for (a) bake exclusion and (b) spin pivot.
            wheelNode = findNodeByName(root, "wheel");
            if (wheelNode) {
                for (const c of wheelNode.children) {
                    if ("_gpu" in c) {
                        wheelMesh = c as unknown as Mesh;
                        break;
                    }
                }
            }
            // eslint-disable-next-line no-console
            console.warn(`[marbleTower] wheel node ${wheelNode ? "found" : "NOT found"} | wheel mesh ${wheelMesh ? "found" : "NOT found"}`);

            // 1) Uniformly scale the root so the model stands TOWER_HEIGHT tall. Multiply
            //    the EXISTING root scale (preserves any handedness sign baked by the
            //    glTF RH→LH conversion) rather than overwriting it.
            const raw = worldAabb(towerMeshes);
            const rawH = raw.max[1] - raw.min[1];
            const factor = rawH > 1e-6 ? TOWER_HEIGHT / rawH : 1;
            const s = root.scaling;
            s.set(s.x * factor, s.y * factor, s.z * factor);

            // 2) Translate the root so the model is centred on X/Z and its base sits on y=0.
            //    Setting root.position shifts every world point by (new - current), so offset
            //    from the current position by the scaled AABB centre / base.
            const scaled = worldAabb(towerMeshes);
            const cx = (scaled.min[0] + scaled.max[0]) / 2;
            const cz = (scaled.min[2] + scaled.max[2]) / 2;
            const baseY = scaled.min[1];
            const p = root.position;
            const px = p.x,
                py = p.y,
                pz = p.z;
            p.set(px - cx, py - baseY, pz - cz);

            for (const m of towerMeshes) {
                setMeshVisible(m, active && containerVisible);
            }

            const finalAabb = worldAabb(towerMeshes);
            // eslint-disable-next-line no-console
            console.warn(
                `[marbleTower] loaded ${towerMeshes.length} meshes | rootScale=${root.scaling.x.toExponential(4)} | ` +
                    `worldAABB min=[${finalAabb.min.map((v) => v.toFixed(3)).join(", ")}] ` +
                    `max=[${finalAabb.max.map((v) => v.toFixed(3)).join(", ")}]`
            );

            // Locate the top cube (the inlet with the 4 holes): fold the X/Z AABB of every
            // world vertex within TOP_SLAB of the apex, so the pour nozzles sit over it rather
            // than at the whole-tower AABB centre (which the base/wheel skew off-axis).
            {
                const apexY = finalAabb.max[1];
                let mnx = Infinity,
                    mnz = Infinity,
                    mxx = -Infinity,
                    mxz = -Infinity;
                for (const mesh of towerMeshes) {
                    const pos = (mesh as CpuMeshNode)._cpuPositions;
                    if (!pos || pos.length === 0) {
                        continue;
                    }
                    const w = mesh.worldMatrix;
                    const m0 = w[0]!,
                        m1 = w[1]!,
                        m2 = w[2]!,
                        m4 = w[4]!,
                        m5 = w[5]!,
                        m6 = w[6]!,
                        m8 = w[8]!,
                        m9 = w[9]!,
                        m10 = w[10]!,
                        m12 = w[12]!,
                        m13 = w[13]!,
                        m14 = w[14]!;
                    for (let i = 0; i < pos.length; i += 3) {
                        const lx = pos[i]!,
                            ly = pos[i + 1]!,
                            lz = pos[i + 2]!;
                        const y = m1 * lx + m5 * ly + m9 * lz + m13;
                        if (y >= apexY - TOP_SLAB) {
                            const x = m0 * lx + m4 * ly + m8 * lz + m12;
                            const z = m2 * lx + m6 * ly + m10 * lz + m14;
                            if (x < mnx) mnx = x;
                            if (x > mxx) mxx = x;
                            if (z < mnz) mnz = z;
                            if (z > mxz) mxz = z;
                        }
                    }
                }
                if (mxx >= mnx && mxz >= mnz) {
                    topStructure.cx = (mnx + mxx) / 2;
                    topStructure.cz = (mnz + mxz) / 2;
                    topStructure.hx = (mxx - mnx) / 2;
                    topStructure.hz = (mxz - mnz) / 2;
                    topStructure.y = apexY;
                    ctx.refreshEmitters(); // re-pack the nozzles now that they're located
                }
                // eslint-disable-next-line no-console
                console.warn(
                    `[marbleTower] top structure centre=[${topStructure.cx.toFixed(2)}, ${topStructure.cz.toFixed(2)}] ` +
                        `half=[${topStructure.hx.toFixed(2)}, ${topStructure.hz.toFixed(2)}] apexY=${topStructure.y.toFixed(2)}`
                );
            }

            // Bake the real marble-run geometry into a signed-distance grid (off the hot path,
            // once, right after the final transform is in place). The "Baked SDF" toggle enables.
            // The water wheel mesh is EXCLUDED from the grid (it spins; the fluid collides with the
            // analytic wheel SDF instead).
            bakeSceneSdf();
            // Detect the spoke count / phase from the wheel mesh BEFORE repurposing its pivot (the
            // repurpose is a θ=0 no-op on world positions, but detecting first keeps the frame clean).
            detectSpokes();
            // Repurpose the "wheel" node as the spin pivot on the REAL wheel mesh (no split / re-create).
            repurposeWheelPivot();
        } catch (err) {
            // Non-fatal: the demo still works with just the water + placeholder floor.
            // eslint-disable-next-line no-console
            console.warn("[marbleTower] failed to load tower model", err);
        }
    })();

    // Curated first-visit presets (recirculating top-pour, framed on the tall tower).
    const presets: Record<string, Partial<PairState>> = {
        PBF: {
            schema: { gravity: 17, viscosity: 1, relaxation: 209, scorr: 0, iterations: 1, restDensity: 600, boundaryDensity: 0 },
            demoParams: { centralSpeed: 7, emitRate: 0.05, nozzleRadius: 0.3 },
            color: "#bfe9f3",
            half: true,
            size: 0.6,
            physScale: 0.8,
            count: 150000,
            camera: { alpha: -1.2, beta: 1.15, radius: 30 },
            // Open in baked mode with the wheel spinning (baked SDF captures the real tower; the
            // analytic wheel + rotating spokes/buckets are unioned on top and stir/carry the water).
            demoState: { bakedActive: true, wheelSdfEnabled: true, spinEnabled: true, driveStrength: 1.0, spokeCount: 12, spokeThickness: 0.18, bucketCount: 30 },
        },
        "MLS-MPM": {
            schema: { gravity: 11.1, stiffness: 1160, viscosity: 0.02, restDensity: 47.5, damping: 1, affineDamping: 0.84, groundDamp: 0.85, groundDampHeight: 1, restitution: 1, substeps: 2 },
            demoParams: { centralSpeed: 0.5, emitRate: 2.76, nozzleRadius: 0.25 },
            color: "#bfe9f3",
            half: true,
            thicknessDownscale: 6,
            absorption: 3.6,
            size: 0.3,
            physScale: 0.5,
            count: 200000,
            camera: { alpha: -0.4172960178148234, beta: 1.2587129923753027, radius: 13.403067826684495 },
            renderMode: "surface",
            refraction: 0.02,
            specular: 250,
            depthBlur: 20,
            depthBlurThreshold: 10,
            thicknessBlur: 18,
            surfaceFilter: "narrowRange",
            narrowDelta: 10,
            narrowMu: 1,
            showContainer: true,
            foam: {
                enabled: true,
                kTa: 120,
                kWc: 44,
                kb: 0.9,
                kd: 0.45,
                tMin: 0.3,
                tMax: 4.7,
                poolScale: 3,
                blurRadius: 0,
                lightIntensity: 0.2,
                ambient: 1,
                aoStrength: 0.18,
                normalStrength: 1,
                debugTexture: "off",
                softness: 2,
                density: 50,
                subsurfaceStrength: 0.15,
            },
            // Default demo look: baked SDF (real tower geometry) with the water wheel on and its
            // spokes/buckets spinning, so the tower opens exactly as the curated screenshot.
            demoState: { bakedActive: true, wheelSdfEnabled: true, spinEnabled: true, driveStrength: 1.0, spokeCount: 12, spokeThickness: 0.18, bucketCount: 30 },
        },
    };

    return {
        key: "marbleTower",
        label: "Marble Tower",
        envUrl: ENV_STUDIO_URL,
        sdf,
        writeSdfParams,
        spawn() {
            // A block pooled near the floor across the domain footprint — the top nozzle
            // then recirculates it upward and pours it back down.
            return { min: [-5, FLOOR_Y + 0.3, -5] as [number, number, number], max: [5, 3, 5] as [number, number, number] };
        },
        emitters() {
            return marbleConfig();
        },
        onEnter(): void {
            active = true;
            for (const m of towerMeshes) {
                setMeshVisible(m, containerVisible);
            }
            setMeshVisible(ctx.ground, true);
            updatePlaneVisibility(); // reveal the SDF slice plane if the tool is on + grid baked
            updateDbgVisibility(); // reveal the wheel-SDF debug quad if that tool is on
            if (dbgActive) {
                refreshWheelDebug(); // make the picture fresh the moment the demo comes on-screen
            }
            // Raise the orbit target to the tower's mid-height so the tall model is framed.
            ctx.camera.target.x = 0;
            ctx.camera.target.y = TOWER_HEIGHT * 0.5;
            ctx.camera.target.z = 0;
        },
        onLeave(): void {
            active = false;
            for (const m of towerMeshes) {
                setMeshVisible(m, false);
            }
            updatePlaneVisibility(); // hide the SDF slice plane along with the tower
            updateDbgVisibility(); // hide the wheel-SDF debug quad along with the tower
            ctx.camera.target.x = 0;
            ctx.camera.target.y = 6;
            ctx.camera.target.z = 0;
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            for (const m of towerMeshes) {
                setMeshVisible(m, active && v);
            }
        },
        update(dt: number): void {
            // Stage 2: measure the MOVING liquid reaching the wheel → integrate ω → spin the wheel
            // pivot node (this also refreshes θ at float 35 in BOTH param blocks before upload below).
            updateWheelSpin(dt);
            // Refresh the analytic-wheel SDF debug picture EVERY frame so it rotates as smoothly as
            // the mesh (160² CPU evals + a small texture upload — cheap). Gated on toggle + active.
            if (active && dbgActive) {
                refreshWheelDebug();
            }
            // Re-write the ACTIVE param block every frame — analytic packs the tower boxes, baked packs
            // the grid origin/dims (both now carry the fresh θ). This keeps the spare tier / grid slots
            // alive against the core's per-switch clearSceneHoles() (which zeroes floats 8..39).
            writeSdfParams();
        },
        demoParams(): DemoParam[] {
            return [
                { key: "centralSpeed", label: "Top pour speed", type: "number", min: 0, max: 16, step: 0.25, value: marbleParams.centralSpeed },
                { key: "nozzleRadius", label: "Nozzle radius", type: "number", min: 0.1, max: 0.8, step: 0.05, value: marbleParams.nozzleRadius },
                { key: "emitRate", label: "Recirculation rate", type: "number", min: 0, max: 3, step: 0.02, value: marbleParams.emitRate },
            ];
        },
        applyParam(key: string, value: number | boolean | string): void {
            (marbleParams as Record<string, number>)[key] = value as number;
            ctx.refreshEmitters();
        },
        extraControls() {
            // Sync the persistent checkbox + status to the current state (it is re-appended on
            // every demo enter / pair load; the bake may have completed while off-screen).
            bakedChk.checked = bakedActive;
            bakedChk.disabled = !bakeReady;
            bakedStatusEl.textContent = bakeStatus;
            // Same for the SDF-texture visualizer controls (state persists across re-enters).
            vizChk.checked = vizActive;
            vizChk.disabled = !bakeReady;
            axisSel.value = String(sliceAxis);
            axisSel.disabled = !bakeReady;
            sliceSlider.value = String(slicePos);
            sliceSlider.disabled = !bakeReady;
            // Wheel toggle persists across re-enters like the others.
            wheelChk.checked = wheelSdfEnabled;
            // Wheel-spoke tuning + the analytic-wheel debug toggle persist across re-enters too.
            spokeSlider.value = String(spokeCount);
            spokeVal.textContent = String(spokeCount);
            thickSlider.value = String(spokeHalfW);
            thickVal.textContent = spokeHalfW.toFixed(2);
            bucketSlider.value = String(bucketCount);
            bucketVal.textContent = String(bucketCount);
            dbgChk.checked = dbgActive;
            // Stage-2 wheel-spin controls persist across re-enters too.
            spinChk.checked = spinEnabled;
            driveSlider.value = String(driveStrength);
            driveVal.textContent = driveStrength.toFixed(1);
            return [bakedRow, bakedStatusEl, wheelRow, spokeRow, thickRow, bucketRow, dbgRow, spinRow, driveRow, spinReadoutEl, vizRow, axisRow, sliceRow];
        },
        snapshotState(): Record<string, number | boolean> {
            // The wheel/baked toggle state that is NOT already a schema/render param, so
            // "Export parameters" and presets carry it. The spoke count + thickness are collision
            // params, so they export too. (The SDF-texture visualizer and the wheel-SDF debug quad
            // are debug tools, not preset settings, so they are deliberately excluded.)
            return { bakedActive, wheelSdfEnabled, spinEnabled, driveStrength, spokeCount, spokeThickness: spokeHalfW, bucketCount };
        },
        restoreState(state: Record<string, number | boolean>): void {
            // Apply each field via the same helper the DOM handler uses (identical behaviour) and
            // sync its control. bakedActive may be DEFERRED here until the async bake completes —
            // setBakedMode remembers the desire (desiredBaked) and bakeSceneSdf applies it later.
            if (typeof state.bakedActive === "boolean") {
                setBakedMode(state.bakedActive);
                bakedChk.checked = state.bakedActive;
            }
            if (typeof state.wheelSdfEnabled === "boolean") {
                setWheelSdf(state.wheelSdfEnabled);
                wheelChk.checked = wheelSdfEnabled;
            }
            if (typeof state.spinEnabled === "boolean") {
                setSpin(state.spinEnabled);
                spinChk.checked = spinEnabled;
            }
            if (typeof state.driveStrength === "number") {
                setDrive(state.driveStrength);
                driveSlider.value = String(driveStrength);
            }
            if (typeof state.spokeCount === "number") {
                setSpokeCount(state.spokeCount);
                spokeSlider.value = String(spokeCount);
            }
            if (typeof state.spokeThickness === "number") {
                setSpokeThickness(state.spokeThickness);
                thickSlider.value = String(spokeHalfW);
            }
            if (typeof state.bucketCount === "number") {
                setBucketCount(state.bucketCount);
                bucketSlider.value = String(bucketCount);
            }
        },
        presets,
    };
}
