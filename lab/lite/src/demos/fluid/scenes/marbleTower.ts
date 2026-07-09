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

import { addToScene, createMeshFromData, createStandardMaterial, loadGltf, resizeMeshGeometry, setMeshVisible } from "babylon-lite";
import type { Mesh, SceneNode } from "babylon-lite";
import type { EmitterConfig, FluidSim, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { generateMeshSdf } from "babylon-lite/fluid/volume-sampling/index.js";
import { createPlane } from "babylon-lite/mesh/mesh-factories.js";
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

// ── Water wheel (the glTF "wheel" node = gltf_mesh_4): a rim ring + a coaxial axle shaft
//    running along +X, offset on the +X side of the tower. These WORLD-space values were
//    measured offline from the asset geometry AFTER the load transform (which fits the tower
//    to TOWER_HEIGHT tall, base at y=0, centred on X/Z), so they are stable as long as
//    TOWER_HEIGHT stays 14. They drive THREE things: (a) the STATIC analytic wheel SDF term,
//    (b) carving the wheel disk out of the baked grid, and (c) splitting the rim disk into its
//    own (Stage-2-rotatable) mesh. Recon: rim band ≈[3.73,3.99], shaft radius ≈0.5 spanning
//    world X≈[0.05,4.24]; NO spokes in the geometry (rim + hub only).
const WHEEL_C: [number, number, number] = [3.78, 4.28, 0.06]; // disk centre — a point on the axle
const WHEEL_AXLE: [number, number, number] = [1, 0, 0]; // unit axle direction (horizontal, +X)
const WHEEL_R = 3.85; // rim radius (rim centreline)
const WHEEL_T = 0.4; // rim half-thickness (axial half-width ≈0.4; also the torus minor radius)
const WHEEL_HUB_R = 0.5; // axle / hub radius
const WHEEL_HUB_HALF = 2.1; // hub / axle half-length along the axle
const WHEEL_HUB_OFFSET = -1.63; // hub centre along the axle relative to C (shaft world X≈[0.05,4.24])
// Split / bake-exclusion cylinder: a thin slab about the axle that tightly bounds the VISIBLE
// rim disk (world X≈[2.98,4.58]). Captures exactly the wheel's 7840 disk triangles and ZERO
// triangles of any other mesh — the coaxial "supports" bearing-frame (gltf_mesh_3) stays static.
// A full-length axle cylinder would wrongly swallow those supports, so the capture is bounded to
// the disk; the inner axle shaft (rotation-symmetric) is left static in the source mesh + bake.
const WHEEL_CAP_R = WHEEL_R + 0.4; // 4.25 — radial cap (≥ rim outer radius ≈3.99)
const WHEEL_CAP_T = 0.8; // axial half-thickness about C.x (disk axial half-extent ≈0.4)

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
    // Shared WGSL: the STATIC analytic water wheel (Stage 1) = a rim torus (major R, minor T) +
    // a coaxial capped-cylinder hub, both about the axle `wheelA` through centre `wheelC`, unioned
    // via min() into BOTH sceneSdf variants (negative inside the solid). wheelMisc.x toggles it
    // (UI checkbox). θ (wheelHub.w) is PACKED for Stage 2 but unused here: the rim+hub are
    // rotation-symmetric about the axle, so a fixed θ has no effect until angular features (spokes/
    // buckets) are added. Layout floats 24..39: wheelC=(cx,cy,cz,R), wheelA=(ax,ay,az,T),
    // wheelHub=(hubR,hubHalf,hubOffset,θ), wheelMisc=(enabled,_,_,_).
    const WHEEL_SDF_FN = `fn sdWheel(pt: vec3<f32>) -> f32 {
    if (sceneSdfParams.wheelMisc.x < 0.5) { return 1.0e9; }
    let wc = sceneSdfParams.wheelC.xyz;
    let axis = normalize(sceneSdfParams.wheelA.xyz);
    let d = pt - wc;
    let a = dot(d, axis);
    let rad = length(d - a * axis);
    let rim = length(vec2<f32>(rad - sceneSdfParams.wheelC.w, a)) - sceneSdfParams.wheelA.w;
    let qx = rad - sceneSdfParams.wheelHub.x;
    let qy = abs(a - sceneSdfParams.wheelHub.z) - sceneSdfParams.wheelHub.y;
    let hub = length(max(vec2<f32>(qx, qy), vec2<f32>(0.0))) + min(max(qx, qy), 0.0);
    return min(rim, hub);
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
    d = min(d, sdWheel(pt));
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
    return min(min(min(min(radial, ceiling), floorD), baked), sdWheel(pt));
}`;

    // Baked grid state (uploaded after the async load bakes it) + which mode is live.
    let gridBuffer: GPUBuffer | null = null;
    let bakeReady = false; // true once the mesh SDF is baked + uploaded
    let bakedActive = false; // true when the BAKED spec is selected (only possible once bakeReady)
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

    // ── STATIC analytic water wheel (Stage 1) ─────────────────────────────────────────────────
    // Enabled by default; the "Water wheel (analytic)" UI checkbox toggles the SDF union (the
    // split rim mesh always renders regardless). Packed at floats 24..39 in BOTH param arrays:
    //   wheelC  (24..27) = centre.xyz, R (rim radius)
    //   wheelA  (28..31) = axle.xyz (unit), T (rim half-thickness)
    //   wheelHub(32..35) = hubR, hubHalf, hubOffset, θ (θ packed for Stage 2, unused now)
    //   wheelMisc(36..39) = enabled(1/0), spare, spare, spare
    let wheelSdfEnabled = true;
    const packWheel = (arr: Float32Array): void => {
        arr[24] = WHEEL_C[0];
        arr[25] = WHEEL_C[1];
        arr[26] = WHEEL_C[2];
        arr[27] = WHEEL_R;
        arr[28] = WHEEL_AXLE[0];
        arr[29] = WHEEL_AXLE[1];
        arr[30] = WHEEL_AXLE[2];
        arr[31] = WHEEL_T;
        arr[32] = WHEEL_HUB_R;
        arr[33] = WHEEL_HUB_HALF;
        arr[34] = WHEEL_HUB_OFFSET;
        arr[35] = 0; // θ — Stage 2 (rim+hub are rotation-symmetric, so 0 is a no-op in Stage 1)
        arr[36] = wheelSdfEnabled ? 1 : 0;
        arr[37] = 0;
        arr[38] = 0;
        arr[39] = 0;
    };

    // Bounding-cylinder test (WORLD space) that tightly bounds the visible rim disk. Used by BOTH
    // the bake (skip these triangles) and the mesh split (extract them). A thin axial slab about
    // C.x plus a radial cap, in the axle frame (axle is unit → axial dist = dot, radial = perp).
    const inWheelCylinder = (x: number, y: number, z: number): boolean => {
        const dx = x - WHEEL_C[0];
        const dy = y - WHEEL_C[1];
        const dz = z - WHEEL_C[2];
        const axial = dx * WHEEL_AXLE[0] + dy * WHEEL_AXLE[1] + dz * WHEEL_AXLE[2];
        if (Math.abs(axial) > WHEEL_CAP_T) {
            return false;
        }
        const px = dx - axial * WHEEL_AXLE[0];
        const py = dy - axial * WHEEL_AXLE[1];
        const pz = dz - axial * WHEEL_AXLE[2];
        return px * px + py * py + pz * pz <= WHEEL_CAP_R * WHEEL_CAP_R;
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
    // The rim disk split out of gltf_mesh_4 into its own mesh (rotatable in Stage 2). Kept
    // separate from towerMeshes so its visibility is driven explicitly alongside the SDF toggle.
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

    bakedChk.onchange = (): void => {
        const want = bakedChk.checked;
        if (want && !bakeReady) {
            bakedChk.checked = false; // bake not ready → stay analytic
            return;
        }
        bakedActive = want;
        applyMode(bakedActive);
        writeSdfParams(); // re-pack the UBO for the active mode
        if (active) {
            // Rebuild the active sim's collision pipes for the new SDF. The inactive backend is
            // refreshed lazily on the next method switch (the core's applySceneSdf re-reads demo.sdf).
            ctx.getActiveSim().setSceneSdf(sdf);
        }
    };

    // ── "Water wheel (analytic)" toggle: unions the STATIC analytic wheel SDF into whichever mode
    //    is live (analytic or baked). Default ON. The split rim mesh renders regardless; this only
    //    gates the SDF collision term (a wheelMisc.x uniform the WGSL branches on — no pipeline
    //    rebuild). The wheel disk is carved from the baked grid either way, so with this OFF in
    //    baked mode the water simply passes through the (now empty) wheel region.
    const wheelRow = document.createElement("label");
    wheelRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:6px 0 2px;cursor:pointer;";
    const wheelChk = document.createElement("input");
    wheelChk.type = "checkbox";
    wheelChk.checked = wheelSdfEnabled;
    const wheelText = document.createElement("span");
    wheelText.textContent = "Water wheel (analytic)";
    wheelRow.append(wheelChk, wheelText);

    wheelChk.onchange = (): void => {
        wheelSdfEnabled = wheelChk.checked;
        packWheel(sdfData); // refresh the enabled flag (float 36) in BOTH blocks
        packWheel(bakedData);
        writeSdfParams(); // push the active block now (update() also re-writes every frame)
    };

    // ── STAGE 2: flux-driven wheel spin ────────────────────────────────────────────────────────
    // Spin the split rim mesh driven by the amount of liquid that reaches the wheel. Each frame a
    // tiny GPU reduction counts particles inside the wheel's CATCH cylinder (axle-local: perpendicular
    // distance to the axle ≤ R+margin, |axial offset along the axle| ≤ T+margin), reducing to a single
    // atomic<u32>. That count is copied to a double-buffered staging buffer and read back with mapAsync
    // (NON-blocking — never awaited in the render path; frames with no fresh value reuse the last one).
    // The count is EMA-smoothed, mapped to a target angular speed ωTarget = driveStrength·n (clamped),
    // and the actual ω relaxes toward it; θ integrates ω and drives mesh.rotation.x (axle = +X → the
    // disk spins in place about C). This is fully DECOUPLED from collision: the analytic wheel SDF stays
    // rotation-symmetric, so θ is a physical no-op there — it is still mirrored into the param block
    // (float 35) for consistency. Little water reaches the wheel by design, so the drive is deliberately
    // sensitive and the strength is exposed as a slider (crank it to see a clear spin from a trickle).
    const CATCH_MARGIN_R = 0.5; // radial slack beyond the rim radius R
    const CATCH_MARGIN_A = 0.6; // axial slack beyond the disk half-thickness T
    const DRIVE_OMEGA_MAX = 8; // rad/s clamp on the target speed (~1.3 rev/s — a clear, fast spin)
    const DRIVE_RESPONSIVENESS = 2; // 1/s — how fast ω relaxes toward its target (~0.5 s time-constant)
    const COUNT_EMA_RATE = 4; // 1/s — how fast the smoothed catch count tracks the async readback
    const TWO_PI = Math.PI * 2;
    const FLUX_WG_SIZE = 256; // reduction workgroup size
    const FLUX_STAGING = 2; // double-buffered readback so mapAsync never stalls the render path

    let spinEnabled = true; // "Spin water wheel" checkbox (default ON)
    let driveStrength = 0.02; // "Wheel drive strength" slider → ωTarget = driveStrength · smoothedCount
    let wheelTheta = 0; // current rotation angle about the axle (rad)
    let wheelOmega = 0; // current angular speed (rad/s)
    let smoothedCount = 0; // EMA of the catch count (de-jitters the async readback cadence)
    let latestCount = 0; // last successfully read-back catch count
    let spinReadoutFrames = 0; // throttles the ω/count read-out DOM update

    // Lazily-built GPU reduction resources — created once, never per frame.
    let fluxPipeline: GPUComputePipeline | null = null;
    let fluxCountBuffer: GPUBuffer | null = null; // atomic<u32> the reduction writes
    let fluxBindGroup: GPUBindGroup | null = null;
    let fluxBoundPos: GPUBuffer | null = null; // positionBuffer the bind group is wired to (rebind on change)
    const fluxStaging: GPUBuffer[] = []; // COPY_DST→MAP_READ readback ring
    const fluxStagingBusy: boolean[] = []; // per-staging in-flight flag (mapAsync pending)

    // Reduction shader: for each particle, add 1 to the atomic when it lies inside the catch cylinder.
    // Wheel geometry is compile-time constant, so it is inlined as literals (single source of truth via
    // the WHEEL_* consts). arrayLength(&positions) == the sim's particle count (positionBuffer is exactly
    // `count` vec4s), so the last workgroup self-guards without a count uniform.
    const wgslF = (n: number): string => n.toFixed(5);
    const fluxRadCap = WHEEL_R + CATCH_MARGIN_R;
    const fluxAxHalf = WHEEL_T + CATCH_MARGIN_A;
    const FLUX_WGSL = `@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outCount: atomic<u32>;
@compute @workgroup_size(${FLUX_WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&positions)) { return; }
    let d = positions[i].xyz - vec3<f32>(${wgslF(WHEEL_C[0])}, ${wgslF(WHEEL_C[1])}, ${wgslF(WHEEL_C[2])});
    let axis = vec3<f32>(${wgslF(WHEEL_AXLE[0])}, ${wgslF(WHEEL_AXLE[1])}, ${wgslF(WHEEL_AXLE[2])});
    let a = dot(d, axis);
    if (abs(a) > ${wgslF(fluxAxHalf)}) { return; }
    if (length(d - a * axis) > ${wgslF(fluxRadCap)}) { return; }
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
                ],
            });
            fluxBoundPos = sim.positionBuffer;
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

    // Per-frame drive: measure flux (when on + active), smooth it, integrate ω→θ, rotate the mesh.
    const updateWheelSpin = (dt: number): void => {
        if (spinEnabled && active) {
            runFluxPass(ctx.getActiveSim());
        }
        // EMA-smooth the catch count so the spin doesn't jitter with the async readback cadence.
        const targetCount = spinEnabled ? latestCount : 0;
        smoothedCount += (targetCount - smoothedCount) * Math.min(COUNT_EMA_RATE * dt, 1);
        // Target angular speed from the smoothed count; relax the actual ω toward it (fixed spin sense —
        // falling water drives a wheel one consistent way; the disk is z-symmetric so +X is as natural
        // as −X). When off, ωTarget = 0 so ω coasts to a stop and the mesh holds still.
        const omegaTarget = spinEnabled ? Math.min(driveStrength * smoothedCount, DRIVE_OMEGA_MAX) : 0;
        wheelOmega += (omegaTarget - wheelOmega) * Math.min(DRIVE_RESPONSIVENESS * dt, 1);
        if (omegaTarget === 0 && wheelOmega < 1e-4) {
            wheelOmega = 0; // fully at rest
        }
        wheelTheta = (wheelTheta + wheelOmega * dt) % TWO_PI; // integrate + wrap
        if (wheelMesh) {
            wheelMesh.rotation.x = wheelTheta; // axle = +X → spins the disk in place about C
        }
        sdfData[35] = wheelTheta; // mirror θ into the wheel block (no-op for the symmetric SDF, kept in sync)
        bakedData[35] = wheelTheta;
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
    spinChk.onchange = (): void => {
        spinEnabled = spinChk.checked; // gates the flux pass (readback stops when off) + ω→0
    };

    const driveRow = document.createElement("label");
    driveRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0;color:#9fb4cc;font-size:12px;";
    const driveText = document.createElement("span");
    driveText.textContent = "Wheel drive strength";
    const driveSlider = document.createElement("input");
    driveSlider.type = "range";
    driveSlider.min = "0";
    driveSlider.max = "0.3";
    driveSlider.step = "0.005";
    driveSlider.value = String(driveStrength);
    driveSlider.style.flex = "1";
    const driveVal = document.createElement("span");
    driveVal.style.cssText = "min-width:34px;text-align:right;";
    driveVal.textContent = driveStrength.toFixed(3);
    driveRow.append(driveText, driveSlider, driveVal);
    driveSlider.oninput = (): void => {
        driveStrength = Number(driveSlider.value);
        driveVal.textContent = driveStrength.toFixed(3);
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

    // Merge every tower mesh's CPU geometry into ONE world-space triangle soup, bake it into a
    // signed-distance grid, and upload it to a storage buffer the sims sample. One-time
    // (~sub-second) synchronous cost right after the async load; logged. Non-fatal on failure.
    const bakeSceneSdf = (): void => {
        // 1) Gather meshes that actually carry CPU positions + indices.
        let totalV = 0;
        let totalI = 0;
        const usable: Mesh[] = [];
        for (const mesh of towerMeshes) {
            const cm = mesh as CpuMeshNode;
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
        let excluded = 0; // wheel-disk triangles skipped from the bake (carved so the wheel can spin)
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
            // Copy indices triangle-by-triangle, SKIPPING any triangle whose world centroid lies
            // inside the wheel-disk cylinder. This carves the wheel out of the baked grid so a
            // (Stage-2) spinning wheel won't fight a static baked copy; the analytic wheel SDF is
            // unioned back in. Positions above stay in mergedPos (unreferenced verts are harmless
            // and keep the merged AABB / grid bounds unchanged so the carved region reads outside).
            for (let i = 0; i < idx.length; i += 3) {
                const a = idx[i]! + vBase;
                const b = idx[i + 1]! + vBase;
                const c = idx[i + 2]! + vBase;
                const gx = (mergedPos[a * 3]! + mergedPos[b * 3]! + mergedPos[c * 3]!) / 3;
                const gy = (mergedPos[a * 3 + 1]! + mergedPos[b * 3 + 1]! + mergedPos[c * 3 + 1]!) / 3;
                const gz = (mergedPos[a * 3 + 2]! + mergedPos[b * 3 + 2]! + mergedPos[c * 3 + 2]!) / 3;
                if (inWheelCylinder(gx, gy, gz)) {
                    excluded++;
                    continue;
                }
                mergedIdx[iOff++] = a; // Uint16→Uint32 normalised by the target array
                mergedIdx[iOff++] = b;
                mergedIdx[iOff++] = c;
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
            // The visualizer can now slice the grid — enable its controls (extraControls() also
            // re-syncs these if the bake finished while the demo was off-screen).
            vizChk.disabled = false;
            axisSel.disabled = false;
            sliceSlider.disabled = false;
            bakeStatus = `${(iOff / 3) | 0} tris → ${grid.dims[0]}×${grid.dims[1]}×${grid.dims[2]} grid, ${bakeMs.toFixed(0)} ms`;
            bakedStatusEl.textContent = bakeStatus;
            // eslint-disable-next-line no-console
            console.warn(
                `[marbleTower] baked mesh SDF: ${bakeStatus} | wheel-excluded ${excluded} tris | ` +
                    `origin=[${grid.origin.map((v) => v.toFixed(2)).join(", ")}] cell=${grid.cellSize}`
            );
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[marbleTower] mesh SDF bake failed — staying analytic", err);
            bakeStatus = "bake failed — analytic only";
            bakedStatusEl.textContent = bakeStatus;
        }
    };

    // Split the wheel's rim disk out of its source mesh into a standalone mesh so Stage 2 can
    // rotate it about the axle. Runs AFTER bakeSceneSdf (which already carved the same triangles
    // from the grid via the SAME inWheelCylinder test). Every triangle whose WORLD centroid lies
    // inside the wheel cylinder is MOVED into a new de-indexed mesh (positions stored in axle-local
    // space = world − C, so the mesh's origin sits on the axle at C and a Stage-2 rotation spins the
    // disk in place). The source mesh is rebuilt WITHOUT those triangles (same vertex arrays +
    // filtered indices), so the untouched geometry (the coaxial axle shaft) renders identically.
    const splitWheelMesh = (): void => {
        const wPos: number[] = [];
        const wNrm: number[] = [];
        const wUv: number[] = [];
        let moved = 0;
        for (const mesh of towerMeshes) {
            const cm = mesh as CpuMeshNode;
            const pos = cm._cpuPositions;
            const idx = cm._cpuIndices;
            const nrm = cm._cpuNormals;
            // Only meshes with positions + indices + normals can be rebuilt (normals are mandatory
            // for resizeMeshGeometry). Every glTF leaf here carries normals, so this never skips the
            // wheel; it just guards against removing triangles we couldn't cleanly rebuild without.
            if (!pos || pos.length === 0 || !idx || idx.length === 0 || !nrm) {
                continue;
            }
            const uv = cm._cpuUvs;
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
            const kept: number[] = [];
            let removedHere = 0;
            for (let t = 0; t < idx.length; t += 3) {
                const tri = [idx[t]!, idx[t + 1]!, idx[t + 2]!];
                // World centroid of the triangle.
                let gx = 0,
                    gy = 0,
                    gz = 0;
                for (const vi of tri) {
                    const lx = pos[vi * 3]!,
                        ly = pos[vi * 3 + 1]!,
                        lz = pos[vi * 3 + 2]!;
                    gx += m0 * lx + m4 * ly + m8 * lz + m12;
                    gy += m1 * lx + m5 * ly + m9 * lz + m13;
                    gz += m2 * lx + m6 * ly + m10 * lz + m14;
                }
                if (!inWheelCylinder(gx / 3, gy / 3, gz / 3)) {
                    kept.push(tri[0]!, tri[1]!, tri[2]!);
                    continue;
                }
                // Move: emit 3 de-indexed verts (world − C for position, world dir for normal).
                for (const vi of tri) {
                    const lx = pos[vi * 3]!,
                        ly = pos[vi * 3 + 1]!,
                        lz = pos[vi * 3 + 2]!;
                    const wx = m0 * lx + m4 * ly + m8 * lz + m12;
                    const wy = m1 * lx + m5 * ly + m9 * lz + m13;
                    const wz = m2 * lx + m6 * ly + m10 * lz + m14;
                    wPos.push(wx - WHEEL_C[0], wy - WHEEL_C[1], wz - WHEEL_C[2]);
                    const nx0 = nrm[vi * 3]!,
                        ny0 = nrm[vi * 3 + 1]!,
                        nz0 = nrm[vi * 3 + 2]!;
                    let nx = m0 * nx0 + m4 * ny0 + m8 * nz0;
                    let ny = m1 * nx0 + m5 * ny0 + m9 * nz0;
                    let nz = m2 * nx0 + m6 * ny0 + m10 * nz0;
                    const nl = Math.hypot(nx, ny, nz) || 1;
                    nx /= nl;
                    ny /= nl;
                    nz /= nl;
                    wNrm.push(nx, ny, nz);
                    if (uv) {
                        wUv.push(uv[vi * 2]!, uv[vi * 2 + 1]!);
                    } else {
                        wUv.push(0, 0);
                    }
                }
                moved++;
                removedHere++;
            }
            // Rebuild this source mesh WITHOUT the moved triangles (only if any were removed).
            if (removedHere > 0) {
                resizeMeshGeometry(engine, mesh, pos, nrm, new Uint32Array(kept), uv);
            }
        }
        if (moved === 0) {
            // eslint-disable-next-line no-console
            console.warn("[marbleTower] wheel split: no triangles matched the wheel cylinder");
            return;
        }
        const positions = new Float32Array(wPos);
        const normals = new Float32Array(wNrm);
        const uvs = new Float32Array(wUv);
        const indices = new Uint32Array(positions.length / 3);
        for (let i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        const mesh = createMeshFromData(engine, "marbleTower-wheel", positions, normals, indices, uvs);
        const mat = createStandardMaterial();
        mat.diffuseColor = [0.28, 0.18, 0.1]; // dark wood (single representative material for the rim)
        mat.specularColor = [0.05, 0.05, 0.05];
        mat.backFaceCulling = false; // world-space rebuild loses the source's reflected winding → draw both sides
        mesh.material = mat;
        mesh.position.set(WHEEL_C[0], WHEEL_C[1], WHEEL_C[2]); // origin on the axle → Stage-2 rotation spins in place
        addToScene(ctx.scene, mesh);
        wheelMesh = mesh;
        setMeshVisible(mesh, active && containerVisible);
        // eslint-disable-next-line no-console
        console.warn(`[marbleTower] wheel split: moved ${moved} tris into a standalone mesh at C=[${WHEEL_C.map((v) => v.toFixed(2)).join(", ")}]`);
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
            // Excludes the wheel disk from the grid.
            bakeSceneSdf();
            // Split the wheel's rim disk into its own mesh (Stage-2-rotatable). AFTER the bake so
            // the same triangles are carved from the grid and moved out of the source mesh together.
            splitWheelMesh();
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
        },
        "MLS-MPM": {
            schema: { gravity: 19.4, stiffness: 1020, viscosity: 0.02, restDensity: 73.5, damping: 0.998, affineDamping: 0.84, groundDamp: 0.85, groundDampHeight: 1, restitution: 1, substeps: 2 },
            demoParams: { centralSpeed: 2, emitRate: 1.5, nozzleRadius: 0.2 },
            color: "#bfe9f3",
            half: true,
            thicknessDownscale: 6,
            absorption: 3.6,
            size: 0.3,
            physScale: 0.5,
            count: 150000,
            camera: { alpha: -7.113995853380378, beta: 1.1888512326265948, radius: 23.126190088504494 },
            renderMode: "surface",
            refraction: 0.02,
            specular: 250,
            depthBlur: 20,
            depthBlurThreshold: 10,
            thicknessBlur: 5,
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
            if (wheelMesh) {
                setMeshVisible(wheelMesh, containerVisible);
            }
            setMeshVisible(ctx.ground, true);
            updatePlaneVisibility(); // reveal the SDF slice plane if the tool is on + grid baked
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
            if (wheelMesh) {
                setMeshVisible(wheelMesh, false);
            }
            updatePlaneVisibility(); // hide the SDF slice plane along with the tower
            ctx.camera.target.x = 0;
            ctx.camera.target.y = 6;
            ctx.camera.target.z = 0;
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            for (const m of towerMeshes) {
                setMeshVisible(m, active && v);
            }
            if (wheelMesh) {
                setMeshVisible(wheelMesh, active && v);
            }
        },
        update(dt: number): void {
            // Stage 2: measure the liquid reaching the wheel → integrate ω → spin the split wheel mesh
            // (this also refreshes θ at float 35 in BOTH param blocks before they are uploaded below).
            updateWheelSpin(dt);
            // Re-write the ACTIVE param block every frame — analytic packs the tower boxes, baked packs
            // the grid origin/dims (both now carry the fresh θ). This keeps the spare tier / grid slots
            // alive against the core's per-switch clearSceneHoles() (which zeroes floats 8..39).
            writeSdfParams();
        },
        demoParams(): DemoParam[] {
            return [
                { key: "centralSpeed", label: "Top pour speed", type: "number", min: 0.5, max: 16, step: 0.25, value: marbleParams.centralSpeed },
                { key: "nozzleRadius", label: "Nozzle radius", type: "number", min: 0.1, max: 0.8, step: 0.05, value: marbleParams.nozzleRadius },
                { key: "emitRate", label: "Recirculation rate", type: "number", min: 0.02, max: 3, step: 0.02, value: marbleParams.emitRate },
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
            // Stage-2 wheel-spin controls persist across re-enters too.
            spinChk.checked = spinEnabled;
            driveSlider.value = String(driveStrength);
            driveVal.textContent = driveStrength.toFixed(3);
            return [bakedRow, bakedStatusEl, wheelRow, spinRow, driveRow, spinReadoutEl, vizRow, axisRow, sliceRow];
        },
        presets,
    };
}
