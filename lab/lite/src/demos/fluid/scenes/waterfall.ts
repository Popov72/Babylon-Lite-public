// Waterfall demo — a floating, irregular rocky slope with water cascading down a
// long series of ASYMMETRIC tiers into a channel that WIDENS toward the front, then
// spilling off the front lip as a foamy curtain. The collision surface is a
// procedural HEIGHTFIELD SDF:  sceneSdf(pt) = pt.y - heightAt(pt.x, pt.z)  (positive
// ABOVE the terrain = fluid domain). No overhangs → a heightfield is the ideal SDF.
//
// The VISIBLE rock terrain is a dense triangle mesh generated from the SAME
// heightfield (`heightAtJS` is a byte-faithful CPU port of the WGSL `heightAt`,
// hash value-noise included), so the water visibly flows over the actual rocks it
// collides with. A few decorative boulders sit embedded on the wall crests.
//
// Channel runs along Z: back/top = high +Z, front/bottom = low -Z. The TOP is a
// closed BASIN (a bowl carved into the top shelf, walled on the back + both sides,
// open only toward the front over the first tier lip). The seed FILLS this basin
// with a resting block of water so it begins as a STILL POND that gently overflows
// forward; a near-vertical source drips into the pond to keep it topped up. Water
// cascades down the seven irregular ledges at uneven speeds (skewed/tilted floor),
// spreads as the valley opens, and spills off the front lip. A wide intake box below
// the lip catches the spill and relaunches it from the source → a seamless
// recirculating loop (the fountain's intake→nozzle idea).

import { addToScene, createMeshFromData, createPbrMaterial, createPolyhedron, createSolidTexture2D, createSphere, loadGltf, loadTexture2D, rebuildMaterial, setMeshVisible, setShadowTaskCasterMeshes } from "babylon-lite";
import type { Mesh, SceneNode } from "babylon-lite";
import type { EmitterConfig, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import type { DemoParam, FluidCtx, FluidDemo, PairState } from "../demo.js";
import { ENV_COUNTRY_URL } from "../demo.js";

// ── Crique (rocky cove) heightfield parameters — the single source of truth for BOTH
//    the WGSL `heightAt` and the CPU `heightAtJS` that builds the terrain MESH. The 8
//    "primary" scalars are packed into the shared scene-SDF UBO (offset 0, 2×vec4) and
//    read back in WGSL; the rest are baked as WGSL literals so each number lives in
//    exactly one place. The two ports (WGSL `heightAt` + JS `heightAtJS`) MUST stay in
//    lock-step (same formula + constants) or the fluid collides with an invisible surface.
//
// Layout: an enclosed oval LAKE dished below the water plane, ROCK WALLS wrapping the
// left/back/right (~270°), a gentle SANDY BEACH on the near (front, -Z) side the camera
// looks over, and a NOTCH carved into the back wall down which the waterfall pours into
// the lake. Everything is expressed against the elliptical radius
//   rr = |(x / RADIUS_X, (z - LAKE_CZ) / RADIUS_Z)|
// where rr<1 = lake bed (below water), rr==1 = shoreline (terrain == water plane),
// rr>1 = rim / land rising out of the water.
const WATER_Y = 3.0; // rest water-surface height (the lake fills to here)
const LAKE_CZ = 1.5; // cove centre in Z (pulled back so the beach fits in front of it)
const LAKE_DEPTH = 3.2; // how far the lake bed dishes BELOW the water plane at the centre
const NOISE_AMP = 0.6; // rocky value-noise amplitude added to the height everywhere
const RADIUS_X = 7.8; // lake ellipse half-width (X) — shoreline sits at rr = 1
const RADIUS_Z = 6.8; // lake ellipse half-depth (Z)
const WALL_H = 6.4; // rock-wall rise above the water plane around the enclosed 270°
const BEACH_H = 2.2; // gentle rise above the water plane on the front beach arc

// Rim shaping (baked as WGSL literals; JS mirrors the same constants).
const WALL_RAMP = 0.72; // elliptical distance past the shoreline the walls reach full height
const BEACH_RAMP = 1.2; // elliptical distance for the gentle front beach slope (wide, shallow)
const BEACH_LO = 0.28; // frontness (= -nz) below which the rim is pure wall …
const BEACH_HI = 0.82; // … and above which it is pure beach (smoothstep band)
const LAND_LO = 0.85; // rr band over which crest noise fades in (0 in the lake …)
const LAND_HI = 1.2; //  … 1 on land) so the lake bed itself stays smooth
const CREST_JITTER = 1.3; // extra jagged value-noise on the rock-wall crests
// Waterfall NOTCH: a lowered spillway channel around x=0 in the BACK wall.
const NOTCH_HW = 2.6; // half-width (world X) of the notch
const NOTCH_DROP = 3.0; // how far the back-wall crest is lowered inside the notch
const NOTCH_OVER = 0.5; // elliptical distance past the shoreline the notch cut fades in

// Hash-noise magic constants (shared between WGSL u32 math and the JS port).
const HASH_A = 1597334677;
const HASH_B = 3812015801;
const HASH_C = 2246822519;
const HASH_D = 3266489917;

// Waterfall SOURCE (top of the back notch) + pump geometry.
const SRC_Z = 10.4; // just behind the notch crest, over the back rock face
const SRC_Y = WATER_Y + WALL_H - 1.4; // near the top of the back wall → the jet free-falls into the lake
const SRC_FWD = 0.55; // forward (-Z) bias so the source arcs off the crest and pours into the lake

// Start-of-sim warm-up: release the lake seed gradually over this many frames (~2.5s at
// 60fps) so the pool settles without a frame-1 pressure spike. MLS-MPM only (PBF ignores).
const WARMUP_FRAMES = 150;

// Terrain-mesh footprint — covers the walls (rr≈1.5 → x≈±11.3, back z≈11.3), the source
// ledge behind, and the beach that runs off the front toward the camera.
const MESH_HALF_X = 13.0; // terrain footprint half-width
const MESH_MAX_Z = 14.5; // covers the back wall + the source ledge
const MESH_MIN_Z = -10.0; // the beach runs down toward the camera
const MESH_NX = 190; // grid columns
const MESH_NZ = 190; // grid rows
const H_FLOOR = -9; // clamp the bed so the mesh stays finite
const ROCK_TILE = 3.0; // world units per rock-texture repeat (terrain UV tiling)
const SDF_WGSL = `fn wfHash(ix: i32, iz: i32) -> f32 {
var h: u32 = u32(ix) * ${HASH_A}u + u32(iz) * ${HASH_B}u;
h = (h ^ (h >> 16u)) * ${HASH_C}u;
h = (h ^ (h >> 13u)) * ${HASH_D}u;
h = h ^ (h >> 16u);
return f32(h) * (1.0 / 4294967295.0);
}
fn wfVnoise(x: f32, z: f32) -> f32 {
let xf = floor(x);
let zf = floor(z);
let ix = i32(xf);
let iz = i32(zf);
let fx = x - xf;
let fz = z - zf;
let ux = fx * fx * (3.0 - 2.0 * fx);
let uz = fz * fz * (3.0 - 2.0 * fz);
let a = wfHash(ix, iz);
let b = wfHash(ix + 1, iz);
let c = wfHash(ix, iz + 1);
let d = wfHash(ix + 1, iz + 1);
return mix(mix(a, b, ux), mix(c, d, ux), uz);
}
fn wfFbm(x: f32, z: f32) -> f32 {
return wfVnoise(x * 0.55, z * 0.55) * 0.65 + wfVnoise(x * 1.3 + 11.3, z * 1.3 + 7.7) * 0.35;
}
fn heightAt(x: f32, z: f32) -> f32 {
let waterY = sceneSdfParams.p0.x;
let lakeCz = sceneSdfParams.p0.y;
let lakeDepth = sceneSdfParams.p0.z;
let noiseAmp = sceneSdfParams.p0.w;
let radX = sceneSdfParams.p1.x;
let radZ = sceneSdfParams.p1.y;
let wallH = sceneSdfParams.p1.z;
let beachH = sceneSdfParams.p1.w;
let ex = x / radX;
let ez = (z - lakeCz) / radZ;
let rr = sqrt(ex * ex + ez * ez);
let nz = ez / max(rr, 0.0001);
let dish = -lakeDepth * clamp(1.0 - rr * rr, 0.0, 1.0);
let frontness = clamp(-nz, 0.0, 1.0);
let beach = smoothstep(${BEACH_LO.toFixed(3)}, ${BEACH_HI.toFixed(3)}, frontness);
let over = max(0.0, rr - 1.0);
let wallRise = wallH * smoothstep(0.0, ${WALL_RAMP.toFixed(3)}, over);
let beachRise = beachH * smoothstep(0.0, ${BEACH_RAMP.toFixed(3)}, over);
var rise = mix(wallRise, beachRise, beach);
let backness = clamp(nz, 0.0, 1.0);
let backw = backness * backness;
let nx0 = clamp(1.0 - (x * x) / (${NOTCH_HW.toFixed(3)} * ${NOTCH_HW.toFixed(3)}), 0.0, 1.0);
let notch = nx0 * nx0 * (3.0 - 2.0 * nx0);
let notchCut = ${NOTCH_DROP.toFixed(3)} * notch * backw * smoothstep(0.0, ${NOTCH_OVER.toFixed(3)}, over);
rise = rise - notchCut;
let land = smoothstep(${LAND_LO.toFixed(3)}, ${LAND_HI.toFixed(3)}, rr);
let crest = land * ${CREST_JITTER.toFixed(3)} * (wfVnoise(x * 0.7 - 4.0, z * 0.7 + 3.0) - 0.5);
let noise = noiseAmp * (wfFbm(x, z) - 0.5);
return waterY + dish + rise + crest + noise;
}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
return pt.y - heightAt(pt.x, pt.z);
}`;

// ── CPU height field (byte-faithful port used to build the terrain mesh) ──
function jSmooth(a: number, b: number, x: number): number {
    const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
}
function jHash(ix: number, iz: number): number {
    let h = (Math.imul(ix | 0, HASH_A) + Math.imul(iz | 0, HASH_B)) >>> 0;
    h = Math.imul(h ^ (h >>> 16), HASH_C) >>> 0;
    h = Math.imul(h ^ (h >>> 13), HASH_D) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    return h * (1.0 / 4294967295.0);
}
function jVnoise(x: number, z: number): number {
    const xf = Math.floor(x);
    const zf = Math.floor(z);
    const ix = xf | 0;
    const iz = zf | 0;
    const fx = x - xf;
    const fz = z - zf;
    const ux = fx * fx * (3 - 2 * fx);
    const uz = fz * fz * (3 - 2 * fz);
    const a = jHash(ix, iz);
    const b = jHash(ix + 1, iz);
    const c = jHash(ix, iz + 1);
    const d = jHash(ix + 1, iz + 1);
    const mx0 = a * (1 - ux) + b * ux;
    const mx1 = c * (1 - ux) + d * ux;
    return mx0 * (1 - uz) + mx1 * uz;
}
function jFbm(x: number, z: number): number {
    return jVnoise(x * 0.55, z * 0.55) * 0.65 + jVnoise(x * 1.3 + 11.3, z * 1.3 + 7.7) * 0.35;
}
function heightAtJS(x: number, z: number): number {
    const waterY = WATER_Y;
    const lakeCz = LAKE_CZ;
    const lakeDepth = LAKE_DEPTH;
    const noiseAmp = NOISE_AMP;
    const radX = RADIUS_X;
    const radZ = RADIUS_Z;
    const wallH = WALL_H;
    const beachH = BEACH_H;
    const ex = x / radX;
    const ez = (z - lakeCz) / radZ;
    const rr = Math.sqrt(ex * ex + ez * ez);
    const nz = ez / Math.max(rr, 0.0001);
    const dish = -lakeDepth * Math.max(0, Math.min(1, 1 - rr * rr));
    const frontness = Math.max(0, Math.min(1, -nz));
    const beach = jSmooth(BEACH_LO, BEACH_HI, frontness);
    const over = Math.max(0, rr - 1);
    const wallRise = wallH * jSmooth(0, WALL_RAMP, over);
    const beachRise = beachH * jSmooth(0, BEACH_RAMP, over);
    let rise = wallRise + (beachRise - wallRise) * beach;
    const backness = Math.max(0, Math.min(1, nz));
    const backw = backness * backness;
    const nx0 = Math.max(0, Math.min(1, 1 - (x * x) / (NOTCH_HW * NOTCH_HW)));
    const notch = nx0 * nx0 * (3 - 2 * nx0);
    const notchCut = NOTCH_DROP * notch * backw * jSmooth(0, NOTCH_OVER, over);
    rise -= notchCut;
    const land = jSmooth(LAND_LO, LAND_HI, rr);
    const crest = land * CREST_JITTER * (jVnoise(x * 0.7 - 4.0, z * 0.7 + 3.0) - 0.5);
    const noise = noiseAmp * (jFbm(x, z) - 0.5);
    return waterY + dish + rise + crest + noise;
}

// Deterministic PRNG so the decorative boulders are identical across reloads.
function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return (): number => {
        a |= 0;
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function createWaterfallDemo(ctx: FluidCtx): FluidDemo {
    const { engine } = ctx;

    // Live-tweakable emitter params (exposed in the Demo-parameters UI). The SDF is
    // STATIC (the terrain mesh matches it exactly), so these only shape the flow.
    // Defaults are tuned for the WATERFALL: a source at the top of the back notch aimed
    // DOWN + slightly FORWARD so recycled water pours off the crest, cascades down the
    // back rock face, and lands in the lake pool (which the pump keeps recirculating).
    const wf = { sourceSpeed: 1.8, emitRate: 7.0, sourceDown: 1.0, spread: 0.35 };

    // Heightfield SDF: positive ABOVE the terrain (the fluid domain). The generic
    // central-difference sceneNormal gives ≈ normalize(-∂h/∂x, 1, -∂h/∂z).
    const sdf: SceneSdfSpec = {
        // p0 = (waterY, lakeCz, lakeDepth, noiseAmp); p1 = (radiusX, radiusZ, wallH, beachH)
        struct: "struct SceneSdfParams { p0: vec4<f32>, p1: vec4<f32>, };",
        sdf: SDF_WGSL,
        // Per-particle push-out + restitution (like the capsule shell): the terrain is
        // an irregular dished/walled surface the coarse MLS grid wall would mis-sample.
        gridConfine: false,
        buffer: ctx.sceneSdfBuffer,
    };

    const writeSdfParams = (): void => {
        engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, new Float32Array([WATER_Y, LAKE_CZ, LAKE_DEPTH, NOISE_AMP, RADIUS_X, RADIUS_Z, WALL_H, BEACH_H]));
    };

    // ── Waterfall source: a ROW of nozzles across the back notch, at the top of the back
    //    rock face, each aimed DOWN + slightly FORWARD (-Z) so recycled water arcs off the
    //    crest and pours down the face into the lake. Spreading the relaunch across several
    //    nozzles lets the pump return a high flux WITHOUT piling recycled particles onto one
    //    nozzle and over-densifying it into an explosion. ──
    const buildEmitters = (): EmitterConfig["emitters"] => {
        const dl = Math.hypot(0, wf.sourceDown, SRC_FWD);
        const dir: [number, number, number] = [0, -wf.sourceDown / dl, -SRC_FWD / dl];
        const list: EmitterConfig["emitters"] = [];
        for (const xk of [-1.5, -0.5, 0.5, 1.5]) {
            list.push({ pos: [xk, SRC_Y, SRC_Z], dir, speed: wf.sourceSpeed, radius: 0.55 });
        }
        return list;
    };

    // Intake: a broad slab across the BOTTOM of the lake basin. Water that has cascaded in
    // and sunk toward the bed is pumped back up to the waterfall source, so the lake level
    // holds steady while a continuous stream cycles falls → lake → bed → falls.
    const buildConfig = (): EmitterConfig => ({
        emitters: buildEmitters(),
        intakeMin: [-5.0, WATER_Y - LAKE_DEPTH - 0.5, LAKE_CZ - 4.0],
        intakeMax: [5.0, WATER_Y - LAKE_DEPTH + 2.2, LAKE_CZ + 4.0],
        rate: wf.emitRate,
        spread: wf.spread,
    });

    // ── Visible rock terrain mesh (generated from heightAtJS === WGSL heightAt) ──
    const cols = MESH_NX + 1;
    const rows = MESH_NZ + 1;
    const positions = new Float32Array(cols * rows * 3);
    const normals = new Float32Array(cols * rows * 3);
    const uvs = new Float32Array(cols * rows * 2);
    const indices = new Uint32Array(MESH_NX * MESH_NZ * 6);
    const eN = 0.06;
    for (let row = 0; row < rows; row++) {
        const z = MESH_MAX_Z - (row / MESH_NZ) * (MESH_MAX_Z - MESH_MIN_Z);
        for (let col = 0; col < cols; col++) {
            const x = -MESH_HALF_X + (col / MESH_NX) * (2 * MESH_HALF_X);
            const vi = (row * cols + col) * 3;
            positions[vi] = x;
            positions[vi + 1] = Math.max(H_FLOOR, heightAtJS(x, z));
            positions[vi + 2] = z;
            // Analytic normal (matches the collision central-difference normal).
            let nx = -(heightAtJS(x + eN, z) - heightAtJS(x - eN, z)) / (2 * eN);
            let nz = -(heightAtJS(x, z + eN) - heightAtJS(x, z - eN)) / (2 * eN);
            const inv = 1 / Math.hypot(nx, 1, nz);
            normals[vi] = nx * inv;
            normals[vi + 1] = inv;
            normals[vi + 2] = nz * inv;
            const ui = (row * cols + col) * 2;
            uvs[ui] = x / ROCK_TILE;
            uvs[ui + 1] = z / ROCK_TILE;
        }
    }
    let ii = 0;
    for (let row = 0; row < MESH_NZ; row++) {
        for (let col = 0; col < MESH_NX; col++) {
            const tl = row * cols + col;
            const tr = tl + 1;
            const bl = (row + 1) * cols + col;
            const br = bl + 1;
            indices[ii++] = br;
            indices[ii++] = tr;
            indices[ii++] = tl;
            indices[ii++] = bl;
            indices[ii++] = br;
            indices[ii++] = tl;
        }
    }
    const terrain = createMeshFromData(engine, "waterfall-terrain", positions, normals, indices, uvs);
    // Rock/terrain look via PBR + image-based lighting (the scene's outdoor HDR env):
    // matte dielectric stone (metal 0, high roughness) so the cliffs catch the sky's
    // ambient + a soft specular sheen instead of the old flat Lambert. A shared 1×1
    // ORM (occ 1, rough 1, metal 0) lets each material dial roughness via roughnessFactor.
    const stoneOrm = createSolidTexture2D(engine, 1.0, 1.0, 0.0);
    const whiteTex = createSolidTexture2D(engine, 1.0, 1.0, 1.0); // placeholder until the rock albedo loads
    const terrainMat = createPbrMaterial({
        baseColorTexture: whiteTex,
        baseColorFactor: [1.12, 1.2, 1.38, 1], // brighten + cool the brown rock albedo toward light grey stone
        ormTexture: stoneOrm,
        roughnessFactor: 0.9,
        environmentIntensity: 1.2,
    });
    terrain.material = terrainMat;
    // Receive the sun's CSM shadows. (Einstein's sandbox GPU rendered receiveShadows
    // meshes uniformly black, but that reproduces on neither the headless CI GPU nor
    // the user's hardware — both show lit textured stone — so it's a sandbox-only
    // driver quirk; shadow-receiving is kept ON.)
    terrain.receiveShadows = true;
    addToScene(ctx.scene, terrain);
    setMeshVisible(terrain, false);

    // ── Decorative boulders EMBEDDED on the wall crests (deterministic). Seated on the
    //    FLAT crest beyond the wall ramp and sunk ~40% into the rock so they read as
    //    piled boulders, never floating. Only placed along the solid back/mid rims
    //    (skipping the widened front + spill face where the terrain drops away). ──
    const rockMats = [
        { c: [1.15, 1.22, 1.4], r: 0.88 },
        { c: [1.28, 1.35, 1.52], r: 0.82 },
        { c: [1.02, 1.1, 1.3], r: 0.92 },
    ].map(({ c, r }) =>
        createPbrMaterial({
            baseColorTexture: whiteTex,
            baseColorFactor: [c[0]!, c[1]!, c[2]!, 1],
            ormTexture: stoneOrm,
            roughnessFactor: r,
            environmentIntensity: 1.2,
        })
    );
    // Load a tiled rock albedo (reliable Babylon playground texture) and swap it onto
    // every stone material once ready — adds real surface detail over the PBR/IBL base.
    // Non-fatal on failure: the solid-tinted placeholder simply stays.
    void loadTexture2D(engine, "https://playground.babylonjs.com/textures/rock.png", {
        srgb: true,
        addressModeU: "repeat",
        addressModeV: "repeat",
    })
        .then((albedo) => {
            for (const m of [terrainMat, ...rockMats]) {
                m.baseColorTexture = albedo;
                rebuildMaterial(ctx.scene, m);
            }
        })
        .catch((e: unknown) => console.warn("[waterfall] rock texture load failed", e));
    const rng = mulberry32(0x9e3779b9);
    const meshes: Mesh[] = [terrain];
    const addRock = (cx: number, cz: number, base: number): void => {
        const t = rng();
        const m = t < 0.55 ? createPolyhedron(engine, { type: 3, size: 1 }) : createSphere(engine, { diameter: 1, segments: 6 });
        m.material = rockMats[(rng() * rockMats.length) | 0]!;
        m.receiveShadows = true;
        const k = base / 0.6;
        m.scaling.set(k * (0.8 + 0.4 * rng()), k * (0.7 + 0.4 * rng()), k * (0.8 + 0.4 * rng()));
        m.rotation.set(rng() * Math.PI, rng() * Math.PI, rng() * Math.PI);
        // Sink the boulder INTO the terrain crest at (cx, cz) so it sits partly buried.
        m.position.set(cx, heightAtJS(cx, cz) + base * 0.18, cz);
        addToScene(ctx.scene, m);
        setMeshVisible(m, false);
        meshes.push(m);
    };
    // Boulders around the cove: seated on the rim just above the shoreline along the
    // enclosing wall arc (back + both flanks), plus a few stones resting in the front
    // shallows / on the beach. Position is expressed in the same elliptical frame as the
    // heightfield: (x, z) = (RADIUS_X·rr·sin θ, LAKE_CZ + RADIUS_Z·rr·cos θ), θ=0 at the
    // back (+Z), θ≈π at the front beach.
    const placeBoulder = (theta: number, rrPos: number, base: number): void => {
        const x = RADIUS_X * rrPos * Math.sin(theta);
        const z = LAKE_CZ + RADIUS_Z * rrPos * Math.cos(theta);
        addRock(x, z, base);
    };
    // Rim boulders around the enclosing wall arc (θ ∈ [-109°, 109°], skipping the front
    // beach gap + the notch centre so the waterfall stays clear).
    const N_RIM = 9;
    for (let i = 0; i < N_RIM; i++) {
        const theta = -1.9 + (i / (N_RIM - 1)) * 3.8;
        if (Math.abs(theta) < 0.28) {
            continue; // leave the notch mouth open for the falls
        }
        const rr = 1.12 + rng() * 0.28;
        placeBoulder(theta, rr, 0.6 + rng() * 0.7);
    }
    // A few stones half-sunk in the front shallows / on the beach.
    for (let i = 0; i < 3; i++) {
        const theta = Math.PI + (rng() - 0.5) * 1.2;
        const rr = 1.05 + rng() * 0.4;
        placeBoulder(theta, rr, 0.4 + rng() * 0.4);
    }

    let containerVisible = true; // toggled by the "Show container / nozzle meshes" checkbox

    // ── Vegetation (Phase B): CC0 Poly Haven glTF plants around the cove rim + beach.
    //    Loaded ASYNC and NON-FATALLY (a failed fetch just logs a warning and leaves
    //    the cove bare). Each plant is seated on heightAtJS(x,z) so it never floats,
    //    placed with its OWN seeded RNG (so it doesn't perturb the boulder layout).
    //    Poly Haven models are real-world-metre-scale and Y-up, matching our axes, so
    //    only a uniform scale + a yaw are applied. Vegetation meshes live in their own
    //    visibility list (toggled with the demo) and are deliberately NOT registered as
    //    CSM shadow casters: the host shadow-receiving path is broken (see terrain note)
    //    so they'd cast nothing visible, and async-added casters would race the caster
    //    pipeline warm-up that fluid.ts does before registerScene.
    let active = false; // true while the waterfall demo is the on-screen demo
    const vegMeshes: Mesh[] = [];
    const vegRng = mulberry32(0x51ed2a17);
    const collectMeshes = (node: SceneNode, out: Mesh[]): void => {
        if ("_gpu" in node) {
            out.push(node as unknown as Mesh);
        }
        for (const c of node.children) {
            collectMeshes(c, out);
        }
    };
    // Poly Haven serves the glTF + .bin under Models/gltf/1k/<name>/ but its textures
    // under a SEPARATE flat dir (Models/jpg/1k/<name>/), so the glTF's relative
    // `textures/*.jpg` URIs 404 through our base-relative loader. Their files API
    // returns an authoritative include-map (relative-path → real CDN url); we fetch it,
    // rewrite every buffer/image URI to its absolute CDN url, and hand loadGltf a blob:
    // URL of the corrected glTF (all URIs now absolute, so base resolution is moot).
    const PH_API = "https://api.polyhaven.com/files";
    // Resolve a Poly Haven model to a self-consistent blob: glTF URL, MEMOISED per model
    // so the manifest + glTF JSON are fetched exactly once; repeats reuse the blob URL and
    // the browser HTTP cache serves the shared .bin/textures, so scattering many copies is
    // cheap. The blob URL is intentionally never revoked (it must survive for every reuse).
    const blobCache = new Map<string, Promise<string>>();
    const resolvePolyHavenGltf = (name: string): Promise<string> => {
        let p = blobCache.get(name);
        if (!p) {
            p = (async (): Promise<string> => {
                const files = (await fetch(`${PH_API}/${name}`).then((r) => r.json())) as {
                    gltf: Record<string, { gltf: { url: string; include: Record<string, { url: string }> } }>;
                };
                const entry = files.gltf["1k"]!.gltf;
                const include = entry.include;
                const gltf = (await fetch(entry.url).then((r) => r.json())) as {
                    buffers?: { uri?: string }[];
                    images?: { uri?: string }[];
                };
                const remap = (uri: string | undefined): string | undefined => (uri && include[uri] ? include[uri]!.url : uri);
                for (const b of gltf.buffers ?? []) {
                    b.uri = remap(b.uri);
                }
                for (const im of gltf.images ?? []) {
                    im.uri = remap(im.uri);
                }
                return URL.createObjectURL(new Blob([JSON.stringify(gltf)], { type: "model/gltf+json" }));
            })();
            blobCache.set(name, p);
        }
        return p;
    };
    const loadPlant = (name: string, x: number, z: number, scale: number, yRot: number, sink = 0): void => {
        void (async (): Promise<void> => {
            const url = await resolvePolyHavenGltf(name);
            const asset = await loadGltf(engine, url);
            const root = asset.entities[0] as SceneNode;
            addToScene(ctx.scene, asset);
            root.position.set(x, heightAtJS(x, z) - sink, z);
            root.scaling.set(scale, scale, scale);
            root.rotation.set(0, yRot, 0);
            const mine: Mesh[] = [];
            collectMeshes(root, mine);
            for (const m of mine) {
                m.receiveShadows = true;
                setMeshVisible(m, active && containerVisible);
                vegMeshes.push(m);
            }
        })().catch((e: unknown) => console.warn("[waterfall] vegetation load failed", name, e));
    };
    // Place a plant in the SAME elliptical frame as the boulders: θ=0 at the back (+Z, the
    // waterfall notch), θ≈π at the front beach (camera side). rr>1 is up on the rim/wall.
    const placePlant = (name: string, theta: number, rrPos: number, scale: number, sink = 0): void => {
        const x = RADIUS_X * rrPos * Math.sin(theta);
        const z = LAKE_CZ + RADIUS_Z * rrPos * Math.cos(theta);
        loadPlant(name, x, z, scale, vegRng() * Math.PI * 2, sink);
    };
    // Trees on the clifftops that enclose the cove: two flanking the back notch (framing the
    // waterfall), one on each side wall, and one on a front beach flank for foreground depth.
    const treeSpots: [number, number, number][] = [
        [-0.8, 1.28, 1.7], // back-left clifftop
        [0.82, 1.28, 1.6], // back-right clifftop
        [-1.55, 1.18, 1.5], // left side clifftop
        [1.55, 1.18, 1.5], // right side clifftop
        [Math.PI - 1.15, 1.08, 1.5], // front-right beach flank
    ];
    for (const [th, rr, sc] of treeSpots) {
        placePlant("tree_small_02", th + (vegRng() - 0.5) * 0.08, rr, sc + vegRng() * 0.3);
    }
    // Ferns hugging the shoreline: swept across the front beach arc (θ around π) plus the two
    // near flanks, seated just above the waterline (rr ≈ 1) so they read as lush cove growth.
    const N_FERN = 14;
    for (let i = 0; i < N_FERN; i++) {
        const th = 1.85 + (i / (N_FERN - 1)) * 2.6; // 1.85 → 4.45 (front arc, skips the notch)
        const rr = 0.99 + vegRng() * 0.19;
        placePlant("fern_02", th, rr, 1.05 + vegRng() * 0.7, 0.05);
    }

    // Curated first-visit presets (merged over the core defaults on first visit).
    // Front 3/4 camera looking at the waterfall face; light-blue water; foam ON.
    // Foam rates are DELIBERATELY lower than the other demos: the recirculating
    // impact/pool churn generates a lot of foam BELOW the visible surface (occluded →
    // wasted). Lower kTa/kWc + a smaller poolScale spend the pool on the visible
    // whitewater (cascade + front curtain) instead of saturating it with hidden bubbles.
    const foam = {
        enabled: true,
        kTa: 20,
        kWc: 20,
        kb: 0.6,
        kd: 0.3,
        tMin: 0.45,
        tMax: 2.5,
        poolScale: 2,
        blurRadius: 3,
        lightIntensity: 0.25,
        ambient: 1,
        aoStrength: 0.15,
        normalStrength: 1.5,
        debugTexture: "off",
        softness: 0.1,
        density: 24,
        subsurfaceStrength: 0.05,
    };
    const camera = { alpha: -1.3, beta: 1.12, radius: 45 };
    // MLS-MPM defaults imported from an exported preset (fluid-waterfall-MLS-MPM.json):
    // deeper/looser water (restDensity 2.5) with gentler recirculation, its own camera
    // framing, and heavier foam (much higher trapped-air/wave-crest rates + a bigger,
    // more buoyant, subsurface-tinted pool). Kept separate from the PBF preset above.
    const mlsCamera = { alpha: -1.3, beta: 1.12, radius: 45 };
    const mlsFoam = {
        enabled: true,
        kTa: 113,
        kWc: 108,
        kb: 1.65,
        kd: 0.8,
        tMin: 0.45,
        tMax: 2.5,
        poolScale: 3.5,
        blurRadius: 4,
        lightIntensity: 0.25,
        ambient: 1,
        aoStrength: 0.15,
        normalStrength: 1.5,
        debugTexture: "off",
        softness: 0.12,
        density: 23.4,
        subsurfaceStrength: 1,
    };
    const presets: Record<string, Partial<PairState>> = {
        PBF: {
            schema: { gravity: 16, viscosity: 0.45, relaxation: 45, scorr: 0.02, iterations: 4, restDensity: 400, boundaryDensity: 0 },
            demoParams: { sourceSpeed: 1.8, emitRate: 9.0, sourceDown: 1.0, spread: 0.35 },
            color: "#cfe8f2",
            half: true,
            size: 0.4,
            physScale: 0.5,
            count: 150000,
            absorption: 0.4,
            camera,
            foam,
        },
        "MLS-MPM": {
            schema: { gravity: 21, stiffness: 260, viscosity: 0.04, restDensity: 2.5, damping: 0.997, affineDamping: 0.75, groundDamp: 0.9, groundDampHeight: 1, restitution: 0.4, substeps: 2 },
            demoParams: { sourceSpeed: 1.8, emitRate: 7.0, sourceDown: 1.0, spread: 0.35 },
            color: "#cfe8f2",
            half: true,
            size: 0.5,
            physScale: 0.5,
            count: 150000,
            absorption: 0.5,
            thicknessDownscale: 6,
            camera: mlsCamera,
            foam: mlsFoam,
        },
    };

    return {
        key: "waterfall",
        label: "Waterfall",
        envUrl: ENV_COUNTRY_URL,
        sdf,
        writeSdfParams,
        spawn() {
            // Seed the whole LAKE at rest: fill the dished basin (elliptical rr<1) from
            // the bed up to the water plane, so the demo opens on a still pool the
            // waterfall then feeds. The warm-up ramp releases the pool gradually so it
            // settles without a frame-1 pressure spike; the surrounding walls + beach
            // hold it in (the lake pools, it does not drain away).
            const accept = (x: number, y: number, z: number): boolean => {
                const ex = x / RADIUS_X;
                const ez = (z - LAKE_CZ) / RADIUS_Z;
                if (ex * ex + ez * ez > 0.93 * 0.93) {
                    return false;
                }
                const h = heightAtJS(x, z);
                return y > h + 0.1 && y < WATER_Y - 0.05;
            };
            return {
                min: [-RADIUS_X, WATER_Y - LAKE_DEPTH, LAKE_CZ - RADIUS_Z] as [number, number, number],
                max: [RADIUS_X, WATER_Y, LAKE_CZ + RADIUS_Z] as [number, number, number],
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
            for (const m of vegMeshes) {
                setMeshVisible(m, containerVisible);
            }
            // Attach the sun's CSM shadow generator + register the terrain/boulders as
            // casters — only while this demo is active (other demos keep the sun but no
            // shadow map, so their custom frame graph is untouched).
            ctx.sun.shadowGenerator = ctx.sunShadow;
            setShadowTaskCasterMeshes(ctx.sunShadow, meshes);
            setMeshVisible(ctx.ground, false); // the rocky cove reads as floating (no ground)
            // Frame the cove: look into the lake from the front, over the beach, toward
            // the waterfall at the back.
            ctx.camera.target.x = 0;
            ctx.camera.target.y = 3.0;
            ctx.camera.target.z = 1.5;
        },
        onLeave(): void {
            active = false;
            for (const m of meshes) {
                setMeshVisible(m, false);
            }
            for (const m of vegMeshes) {
                setMeshVisible(m, false);
            }
            // Detach the shadow generator + clear casters so box/capsule/fountain render
            // no shadow map (and don't get the waterfall's shadows).
            ctx.sun.shadowGenerator = undefined;
            setShadowTaskCasterMeshes(ctx.sunShadow, []);
            setMeshVisible(ctx.ground, true);
            ctx.camera.target.x = 0;
            ctx.camera.target.y = 6;
            ctx.camera.target.z = 0;
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            for (const m of meshes) {
                setMeshVisible(m, v);
            }
            for (const m of vegMeshes) {
                setMeshVisible(m, active && v);
            }
        },
        shadowCasters(): Mesh[] {
            return meshes; // terrain + boulders cast the sun's CSM shadows
        },
        update(): void {
            /* emitters refresh on param change, not per-frame; SDF is static */
        },
        demoParams(): DemoParam[] {
            return [
                { key: "sourceSpeed", label: "Source speed", type: "number", min: 0, max: 12, step: 0.5, value: wf.sourceSpeed },
                { key: "emitRate", label: "Recirculation rate", type: "number", min: 0.2, max: 12, step: 0.1, value: wf.emitRate },
                { key: "sourceDown", label: "Source downward", type: "number", min: 0, max: 6, step: 0.1, value: wf.sourceDown },
                { key: "spread", label: "Jet spread", type: "number", min: 0, max: 2, step: 0.05, value: wf.spread },
            ];
        },
        applyParam(key: string, value: number | boolean | string): void {
            (wf as Record<string, number>)[key] = value as number;
            ctx.refreshEmitters();
        },
        extraControls() {
            return [];
        },
        presets,
    };
}
