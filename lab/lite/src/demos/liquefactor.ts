// Liquefactor demo — VOLUME-SAMPLING VERIFICATION with a screen-space fluid surface.
//
// Fills the interior of a procedural "enemy" mesh with particles using the
// `sampleMeshVolume` CPU volume sampler (packages/babylon-lite/src/fluid/
// volume-sampling), then "liquefies" the fill into flowing GPU fluid rendered
// through demo-fluid's screen-space surface pipeline (refraction + Beer-Lambert
// absorption + fresnel env reflection + specular).
//
// Flow:
//   • Pick an enemy shape / sampling mode / radius. A LIVE particle-count PREVIEW
//     (`createVolumeSampler` → SDF + lattice seed, no SPH) shows the predicted
//     count WITHOUT liquefying — so you can dial the fill before committing.
//   • Liquefy → runs the full sampler, seeds a PAUSED MLS-MPM sim from the sampled
//     points (the static fill in the mesh's exact shape — THIS is the
//     verification) and CACHES the world-space seed.
//   • Melt → steps the sim (gravity pulls the fill into a puddle); pressing again
//     pauses. There is NO auto-melt.
//   • Reset → rebuilds the sim from the CACHED seed and returns to the paused fill
//     (no re-sample, no solid mesh). With no cache yet it falls back to the solid
//     mesh.
//
// Render setup mirrors demo-fluid: the scene (ground + enemy + HDR skybox) draws
// into an OFFSCREEN colour target (`sceneColorRT`) that a `createFluidSurfaceTask`
// composites into the swapchain, reconstructing the liquid surface. MSAA/depth
// handling matches fluid.ts (msaaSamples:1 + a shared single-sample depth buffer).

import {
    addTask,
    addToScene,
    attachControl,
    createArcRotateCamera,
    createDirectionalLight,
    createEngine,
    createGround,
    createHemisphericLight,
    createMeshFromData,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createSphere,
    createStandardMaterial,
    createTorus,
    createTorusKnot,
    loadBabylon,
    loadEnvironment,
    loadGltf,
    onBeforeRender,
    registerScene,
    setMeshVisible,
    startEngine,
} from "babylon-lite";
import type { AssetContainer, EnvironmentTextures, Mesh, Renderable, SceneNode } from "babylon-lite";
import { createMlsMpmSim } from "babylon-lite/fluid/mls-mpm-sim.js";
import { createPbfSim } from "babylon-lite/fluid/pbf-sim.js";
import type { FluidSim, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { createParticleRenderTask } from "babylon-lite/fluid/particle-render.js";
import { createFluidSurfaceTask } from "babylon-lite/fluid/fluid-surface-render.js";
import { createVolumeSampler, sampleMeshVolume } from "babylon-lite/fluid/volume-sampling/index.js";
import type { VolumeSamplingMode } from "babylon-lite/fluid/volume-sampling/index.js";
import { buildHdrSkyboxRenderable } from "babylon-lite/material/pbr/background-hdr-skybox.js";
import { createFluidControlsPanel, DEFAULT_FLUID_SCHEMAS } from "babylon-lite/fluid/controls-panel.js";
import type { PhysSchemaEntry } from "babylon-lite/fluid/controls-panel.js";
import { demoAssetUrl } from "./demo-asset-url.js";

// World layout. The enemy floats at MESH_CENTER_Y above a ground plane at y = 0;
// the sim grid drops one unit below ground so the ground BC isn't fighting the
// grid's own domain-border wall (mirrors the fluid demo's rationale).
const MESH_CENTER_Y = 5;
const GROUND_Y = 0;
const DOMAIN_HALF = 11; // horizontal half-extent of the MLS grid (puddle spread room)

// Studio HDR environment — drives the fluid-surface reflections + the skybox
// background (loaded exactly as demo-fluid does).
const ENV_STUDIO_URL = "https://playground.babylonjs.com/textures/environment.env";
const SUN_DIR: [number, number, number] = [-0.4, -0.82, -0.45];

// LOADED-ASSET enemies — real models fetched over the network and merged into ONE
// origin-centred Mesh each (see mergeAssetToEnemyMesh) so they plug into the exact
// same sample → fill → melt flow as the procedural shapes.
//   • Dude          — a skinned .babylon character (sampled at its bind pose).
//   • Haunted House  — a multi-mesh .glb building (open/non-manifold; fill is best-effort).
const DUDE_URL = "https://assets.babylonjs.com/meshes/Dude/dude.babylon";
const HAUNTED_HOUSE_URL = "https://assets.babylonjs.com/meshes/haunted_house.glb";
// Largest-extent (world units) each loaded model is uniformly scaled to — comparable
// to the procedural enemies (~5–6 units) — before being centred at the local origin.
const LOADED_MODEL_EXTENT = 6.5;

// Box-preset render defaults (fluid/scenes/box.ts MLS-MPM preset).
const DEF_COLOR = "#16a3c3";
const DEF_ABSORPTION = 0.4;
const DEF_SIZE = 0.7;
const DEF_REFRACTION = 0.06;
const DEF_SPECULAR = 41;
const DEF_DEPTH_BLUR = 40;
const DEF_DEPTH_BLUR_THRESHOLD = 41;
const DEF_THICKNESS_BLUR = 16;
const DEF_HALF = true;
const DEF_THICKNESS_DOWNSCALE = 6;
const DEF_SURFACE_FILTER: "bilateral" | "narrowRange" = "narrowRange";
const DEF_NARROW_DELTA = 10;
const DEF_NARROW_MU = 1;

type EnemyKey = "torusKnot" | "sphere" | "torus" | "dude" | "hauntedHouse";

const ENEMY_LABELS: Record<EnemyKey, string> = {
    torusKnot: "Torus Knot",
    sphere: "Sphere",
    torus: "Torus",
    dude: "Dude",
    hauntedHouse: "Haunted House",
};

// Loaded enemies whose <option> is disabled until their async load resolves.
const LOADED_ENEMY_KEYS: EnemyKey[] = ["dude", "hauntedHouse"];

function hexToRgb(hex: string): [number, number, number] {
    return [parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255, parseInt(hex.slice(5, 7), 16) / 255];
}

/** Recompute smooth vertex normals from merged triangles (fallback for source meshes
 *  that ship no normals). Accumulates face normals per vertex, then normalizes —
 *  matching the engine's ComputeNormals for the default left-handed case. */
function computeMergedNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
    const normals = new Float32Array(positions.length);
    for (let f = 0; f < indices.length; f += 3) {
        const i0 = indices[f]! * 3;
        const i1 = indices[f + 1]! * 3;
        const i2 = indices[f + 2]! * 3;
        const ax = positions[i0]!,
            ay = positions[i0 + 1]!,
            az = positions[i0 + 2]!;
        const e1x = positions[i1]! - ax,
            e1y = positions[i1 + 1]! - ay,
            e1z = positions[i1 + 2]! - az;
        const e2x = positions[i2]! - ax,
            e2y = positions[i2 + 1]! - ay,
            e2z = positions[i2 + 2]! - az;
        const nx = e1y * e2z - e1z * e2y;
        const ny = e1z * e2x - e1x * e2z;
        const nz = e1x * e2y - e1y * e2x;
        normals[i0] = normals[i0]! + nx;
        normals[i0 + 1] = normals[i0 + 1]! + ny;
        normals[i0 + 2] = normals[i0 + 2]! + nz;
        normals[i1] = normals[i1]! + nx;
        normals[i1 + 1] = normals[i1 + 1]! + ny;
        normals[i1 + 2] = normals[i1 + 2]! + nz;
        normals[i2] = normals[i2]! + nx;
        normals[i2 + 1] = normals[i2 + 1]! + ny;
        normals[i2 + 2] = normals[i2 + 2]! + nz;
    }
    for (let i = 0; i < normals.length; i += 3) {
        const x = normals[i]!,
            y = normals[i + 1]!,
            z = normals[i + 2]!;
        const len = Math.hypot(x, y, z) || 1;
        normals[i] = x / len;
        normals[i + 1] = y / len;
        normals[i + 2] = z / len;
    }
    return normals;
}

async function main(): Promise<void> {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;

    // Request the adapter's max storage/buffer limits so fine grids (small radius →
    // many cells) still fit; harmless when they're not needed.
    let requiredLimits: Record<string, number> | undefined;
    try {
        const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
        if (adapter) {
            requiredLimits = {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            };
        }
    } catch {
        // Fall back to default limits.
    }

    // Single-sample so the particle task can share one single-sample depth buffer
    // with the scene pass (same as demo-fluid).
    const engine = await createEngine(canvas, { msaaSamples: 1, requiredLimits });
    const scene = createSceneContext(engine, { defaultRenderTask: false });

    const cam = createArcRotateCamera(-Math.PI / 2, 1.05, 18, { x: 0, y: MESH_CENTER_Y, z: 0 });
    cam.nearPlane = 0.1;
    cam.farPlane = 200;
    scene.camera = cam;
    attachControl(cam, canvas, scene);

    addToScene(scene, createHemisphericLight([0.2, 1, 0.3], 0.8));
    const sun = createDirectionalLight(SUN_DIR, 2.0);
    sun.position.set(12, 20, 10);
    addToScene(scene, sun);

    const ground = createGround(engine, { width: 40, height: 40, subdivisions: 1 });
    const groundMat = createStandardMaterial();
    groundMat.diffuseColor = [0.22, 0.24, 0.28];
    groundMat.specularColor = [0.04, 0.04, 0.05];
    ground.material = groundMat;
    addToScene(scene, ground);

    // Menacing "enemy" material — an opaque, sickly-green alien surface.
    const enemyMat = createStandardMaterial();
    enemyMat.diffuseColor = [0.16, 0.62, 0.24];
    enemyMat.specularColor = [0.35, 0.45, 0.35];

    // Build all three enemy shapes up front and add them to the scene; only one is
    // visible at a time. This keeps their CPU geometry (`_cpuPositions/_cpuIndices`)
    // resident for sampling + preview and avoids scene add/remove churn on selection.
    function buildEnemy(key: EnemyKey): Mesh {
        let mesh: Mesh;
        if (key === "sphere") {
            mesh = createSphere(engine, { diameter: 5, segments: 32 });
        } else if (key === "torus") {
            mesh = createTorus(engine, { diameter: 6, thickness: 2, tessellation: 32 });
        } else {
            mesh = createTorusKnot(engine, { radius: 2, tube: 0.6, radialSegments: 32, tubularSegments: 64 });
        }
        mesh.material = enemyMat;
        mesh.position.set(0, MESH_CENTER_Y, 0);
        return mesh;
    }

    // Node shape used to read a loaded asset's CPU geometry (present on Mesh nodes,
    // absent on pure TransformNodes / lights).
    type CpuMeshNode = SceneNode & {
        _cpuPositions?: Float32Array;
        _cpuNormals?: Float32Array;
        _cpuIndices?: Uint32Array | Uint16Array;
    };

    // ── Merge a loaded AssetContainer into ONE origin-centred enemy Mesh ─────────
    // Walk the loaded node tree, bake every sub-mesh's vertices into world space via
    // its `worldMatrix`, concatenate them, then uniformly scale + centre the result
    // at the local origin. The output is an ordinary Mesh that RETAINS
    // `_cpuPositions`/`_cpuIndices` (via createMeshFromData) and sits at y=MESH_CENTER_Y
    // — indistinguishable from a procedural enemy to runSample()/computePreview().
    //
    // World-matrix convention (VERIFIED against math/compute-aabb.ts): Mat4 is
    // COLUMN-MAJOR, m[col*4 + row], translation in m[12..14]. A position (w=1) maps to
    //   x' = m0*x + m4*y + m8*z + m12   (and likewise for y'/z' using rows 1/2)
    // — identical to computeAabb's transform path. Normals use the upper-3×3; for a
    // reflection (glTF's __root__ bakes scale [-1,1,1] for RH→LH) the plain 3×3 gives
    // the wrong SIGN, so we multiply by sign(det(3×3)) to recover outward normals
    // (exact for rotation + uniform-scale + reflection).
    function mergeAssetToEnemyMesh(asset: AssetContainer, name: string): Mesh {
        type Part = { pos: Float32Array; nrm: Float32Array | null; idx: Uint32Array | Uint16Array; w: Mesh["worldMatrix"] };
        const parts: Part[] = [];

        const visit = (node: SceneNode): void => {
            const cm = node as CpuMeshNode;
            const pos = cm._cpuPositions;
            const idx = cm._cpuIndices;
            if (pos && pos.length > 0 && idx && idx.length > 0) {
                const nrm = cm._cpuNormals;
                // Read the FULL world matrix (parent chain already wired below).
                parts.push({ pos, nrm: nrm && nrm.length === pos.length ? nrm : null, idx, w: node.worldMatrix });
            }
            const kids = node.children;
            if (Array.isArray(kids)) {
                for (const child of kids) {
                    // glTF leaves parent links for addToScene(); wire them here so
                    // worldMatrix chains correctly. The .babylon loader already sets
                    // them — the setter is idempotent. We never addToScene these source
                    // nodes (only the merged mesh), so this has no scene side-effect.
                    child.parent = node;
                    visit(child);
                }
            }
        };
        for (const entity of asset.entities) {
            const sn = entity as Partial<SceneNode>;
            if (Array.isArray(sn.children)) {
                visit(entity as SceneNode);
            }
        }
        if (parts.length === 0) {
            throw new Error("asset contains no meshes with CPU geometry");
        }

        // Size the combined buffers.
        let totalVerts = 0;
        let totalIndices = 0;
        for (const p of parts) {
            totalVerts += p.pos.length / 3;
            totalIndices += p.idx.length;
        }
        const positions = new Float32Array(totalVerts * 3);
        const normals = new Float32Array(totalVerts * 3);
        const indices = new Uint32Array(totalIndices);

        let vBase = 0; // vertices written so far
        let iCur = 0; // index-write cursor
        let missingNormals = false;

        for (const p of parts) {
            const w = p.w;
            const m0 = w[0]!,
                m1 = w[1]!,
                m2 = w[2]!;
            const m4 = w[4]!,
                m5 = w[5]!,
                m6 = w[6]!;
            const m8 = w[8]!,
                m9 = w[9]!,
                m10 = w[10]!;
            const m12 = w[12]!,
                m13 = w[13]!,
                m14 = w[14]!;
            // Sign of the upper-3×3 determinant: negative for a reflected transform.
            const det = m0 * (m5 * m10 - m9 * m6) - m4 * (m1 * m10 - m9 * m2) + m8 * (m1 * m6 - m5 * m2);
            const nSign = det < 0 ? -1 : 1;

            const vCount = p.pos.length / 3;
            const nrm = p.nrm;
            if (!nrm) {
                missingNormals = true;
            }
            for (let v = 0; v < vCount; v++) {
                const s = v * 3;
                const lx = p.pos[s]!,
                    ly = p.pos[s + 1]!,
                    lz = p.pos[s + 2]!;
                const o = (vBase + v) * 3;
                positions[o] = m0 * lx + m4 * ly + m8 * lz + m12;
                positions[o + 1] = m1 * lx + m5 * ly + m9 * lz + m13;
                positions[o + 2] = m2 * lx + m6 * ly + m10 * lz + m14;
                if (nrm) {
                    const nx = nrm[s]!,
                        ny = nrm[s + 1]!,
                        nz = nrm[s + 2]!;
                    const tx = m0 * nx + m4 * ny + m8 * nz;
                    const ty = m1 * nx + m5 * ny + m9 * nz;
                    const tz = m2 * nx + m6 * ny + m10 * nz;
                    const len = Math.hypot(tx, ty, tz) || 1;
                    const k = nSign / len;
                    normals[o] = tx * k;
                    normals[o + 1] = ty * k;
                    normals[o + 2] = tz * k;
                }
            }
            const idx = p.idx;
            for (let k = 0; k < idx.length; k++) {
                indices[iCur + k] = vBase + idx[k]!;
            }
            vBase += vCount;
            iCur += idx.length;
        }

        // Uniform scale + centre so the largest extent ≈ LOADED_MODEL_EXTENT and the
        // model is centred at the LOCAL origin (like the procedural enemies' geometry).
        let minX = Infinity,
            minY = Infinity,
            minZ = Infinity;
        let maxX = -Infinity,
            maxY = -Infinity,
            maxZ = -Infinity;
        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i]!,
                y = positions[i + 1]!,
                z = positions[i + 2]!;
            if (x < minX) {
                minX = x;
            }
            if (x > maxX) {
                maxX = x;
            }
            if (y < minY) {
                minY = y;
            }
            if (y > maxY) {
                maxY = y;
            }
            if (z < minZ) {
                minZ = z;
            }
            if (z > maxZ) {
                maxZ = z;
            }
        }
        const cx = (minX + maxX) / 2,
            cy = (minY + maxY) / 2,
            cz = (minZ + maxZ) / 2;
        const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
        const scale = LOADED_MODEL_EXTENT / extent;
        for (let i = 0; i < positions.length; i += 3) {
            positions[i] = (positions[i]! - cx) * scale;
            positions[i + 1] = (positions[i + 1]! - cy) * scale;
            positions[i + 2] = (positions[i + 2]! - cz) * scale;
        }
        // (Uniform positive scale + translation leaves normal directions unchanged.)

        const mergedNormals = missingNormals ? computeMergedNormals(positions, indices) : normals;
        const mesh = createMeshFromData(engine, name, positions, mergedNormals, indices);
        mesh.material = enemyMat;
        mesh.position.set(0, MESH_CENTER_Y, 0);
        return mesh;
    }

    const enemies: Partial<Record<EnemyKey, Mesh>> = {
        torusKnot: buildEnemy("torusKnot"),
        sphere: buildEnemy("sphere"),
        torus: buildEnemy("torus"),
    };
    for (const key of ["torusKnot", "sphere", "torus"] as EnemyKey[]) {
        const m = enemies[key]!;
        addToScene(scene, m);
        setMeshVisible(m, false);
    }
    let currentKey: EnemyKey = "torusKnot";
    let currentEnemy: Mesh = enemies.torusKnot!;
    setMeshVisible(currentEnemy, true);

    // ── Render pipeline (demo-fluid screen-space fluid surface) ──────────────
    // Depth buffer owned by the scene task; the particle + surface tasks load +
    // test against it so the fluid depth-tests against the ground.
    const depthRT = createRenderTarget({ lbl: "liq-depth", dFormat: "depth24plus", samples: 1, size: "canvas" });
    // The scene renders to an OFFSCREEN colour target (not the swapchain) so the
    // fluid surface pass can SAMPLE it for refraction. clr:true keeps a valid
    // background even if the HDR skybox fails to load; the skybox (order-0
    // renderable) overwrites every pixel when present.
    const sceneColorRT = createRenderTarget({ lbl: "liq-scene-color", format: engine.format, samples: 1, size: "canvas" });
    const sceneTask = createRenderTask(
        { name: "scene", rt: sceneColorRT, depth: depthRT, clr: true, clrColor: { r: 0.05, g: 0.06, b: 0.1, a: 1 } },
        engine,
        scene
    );
    addTask(scene, sceneTask);

    // Ground-plane scene SDF for the sim: positive above the floor, negative below.
    const groundSdfBuffer = engine._device.createBuffer({
        label: "liq-scene-sdf",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    engine._device.queue.writeBuffer(groundSdfBuffer, 0, new Float32Array([GROUND_Y, 0, 0, 0]));
    const groundSdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { ground: vec4<f32>, };",
        sdf: "fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return pt.y - sceneSdfParams.ground.x; }",
        buffer: groundSdfBuffer,
    };

    // A tiny placeholder sim so the particle/surface render tasks have valid buffers
    // to bind at boot. Disabled + parked off-screen until the first Liquefy replaces
    // it with the real, volume-seeded sim via `setSim`.
    const placeholderSim = createMlsMpmSim(engine, {
        count: 1,
        particleRadius: 0.08,
        initialPositions: new Float32Array([0, -1e5, 0]),
        boundsMin: [-2, -2, -2],
        boundsMax: [2, 2, 2],
        dx: 1,
    });

    // Particle impostors draw into the OFFSCREEN colour (only shown in "Spheres"
    // mode; in "Surface" mode this task is disabled and the surface pass
    // reconstructs the liquid from the sim's particle buffer directly).
    const particleTask = createParticleRenderTask(engine, scene, { colorRT: sceneColorRT, depthRT, camera: cam, sim: placeholderSim });
    particleTask.setEnabled(false);
    addTask(scene, particleTask);

    // Fluid surface renderer + frame presenter: reads the offscreen scene colour and
    // writes the swapchain. Surface mode reconstructs + shades the liquid (refraction
    // + absorption + reflection); blit mode just blits the scene (with impostors).
    const surfaceTask = createFluidSurfaceTask(engine, scene, { bgRT: sceneColorRT, outRT: engine.scRT, depthRT, camera: cam, sim: placeholderSim });
    addTask(scene, surfaceTask);

    // Initialize the surface task + particle size to the box-preset defaults.
    surfaceTask.setDirLight(SUN_DIR);
    surfaceTask.setFluidColor(hexToRgb(DEF_COLOR));
    surfaceTask.setAbsorption(DEF_ABSORPTION);
    surfaceTask.setSizeScale(DEF_SIZE);
    particleTask.setSizeScale(DEF_SIZE);
    surfaceTask.setRefractionStrength(DEF_REFRACTION);
    surfaceTask.setSpecularPower(DEF_SPECULAR);
    surfaceTask.setDepthBlur(DEF_DEPTH_BLUR, DEF_DEPTH_BLUR_THRESHOLD);
    surfaceTask.setThicknessBlur(DEF_THICKNESS_BLUR);
    surfaceTask.setHalfRender(DEF_HALF);
    surfaceTask.setThicknessDownscale(DEF_THICKNESS_DOWNSCALE);
    surfaceTask.setSurfaceFilter(DEF_SURFACE_FILTER);
    surfaceTask.setNarrowRange(DEF_NARROW_DELTA, DEF_NARROW_MU);
    surfaceTask.setMode("surface");

    // Load the studio HDR env (non-fatal): feeds the surface reflections and pushes
    // an order-0 HDR skybox renderable that the scene task draws as the background
    // into sceneColorRT. demoAssetUrl resolves the BRDF LUT next to the bundled demo.
    const brdfUrl = demoAssetUrl("./brdf-lut.png", import.meta.url);
    let studioSky: Renderable | null = null;
    const envReady = loadEnvironment(scene, ENV_STUDIO_URL, { brdfUrl, skipGround: true, skipSkybox: true })
        .then((env: EnvironmentTextures) => {
            scene.imageProcessing.exposure = 1.0;
            scene.imageProcessing.contrast = 1.1;
            studioSky = buildHdrSkyboxRenderable(scene, env, 10, [0, 0, 0], [0, 0, 0]);
            scene._renderables.push(studioSky);
            scene._renderableVersion++;
            surfaceTask.setEnvMap({ view: env.specularCubeView, sampler: env.cubeSampler });
        })
        .catch((err) => {
            // eslint-disable-next-line no-console
            console.warn("[liquefactor] env load failed", err);
        });

    // ── Demo state ────────────────────────────────────────────────────────
    type State = "solid" | "sampling" | "filled" | "melting";
    let state: State = "solid";
    let realSim: FluidSim | null = null;
    let melting = false;
    let particleCount = 0;
    let radiusValue = 0.08;
    let modeValue: VolumeSamplingMode = "dense";
    let renderSpheres = false; // false = fluid surface (default), true = sphere impostors

    // Active solver for the melt. Both backends are seeded from the SAME cached fill
    // (exact per-particle positions), so switching method re-melts the identical blob.
    // Default MLS-MPM (grid-transfer; robust for a free blob on the ground).
    let currentMethod = "MLS-MPM";

    // Liquefactor tunes the shared PBF defaults for a FREE blob-on-ground melt (not a
    // confined tank): more XSPH viscosity so the fluid settles into a puddle instead of
    // splashing into a thin fast sheet on impact. MLS-MPM keeps the shared defaults (its
    // ground damping already contains the melt). These become the demo's slider defaults
    // AND the "Reset" targets; the shared DEFAULT_FLUID_SCHEMAS (fluid demo) is untouched.
    const LIQ_SCHEMAS: Record<string, PhysSchemaEntry[]> = Object.fromEntries(
        Object.entries(DEFAULT_FLUID_SCHEMAS).map(([m, entries]) => [
            m,
            entries.map((e) => (m === "PBF" && e.key === "viscosity" ? { ...e, value: 0.35 } : { ...e })),
        ])
    );

    // Per-method physics-slider values (seeded from the schema defaults). The physics
    // panel writes here live (onPhysicsParam) and a freshly (re)built sim reads these so
    // it starts from the current slider values. SCHEMA_DEFAULTS holds the pristine
    // defaults the physics "Reset" button restores.
    const SCHEMA_DEFAULTS: Record<string, Record<string, number>> = {};
    const physValues: Record<string, Record<string, number>> = {};
    for (const m of Object.keys(LIQ_SCHEMAS)) {
        SCHEMA_DEFAULTS[m] = {};
        physValues[m] = {};
        for (const p of LIQ_SCHEMAS[m]!) {
            SCHEMA_DEFAULTS[m]![p.key] = p.value;
            physValues[m]![p.key] = p.value;
        }
    }

    // Cached world-space seed from the last successful Liquefy — replayed by Reset
    // (no re-sample). Everything buildRealSim needs.
    let cachedPositions: Float32Array | null = null;
    let cachedCount = 0;
    let cachedRadius = 0.08;
    let cachedWorldTop = 8;

    // Live pre-Liquefy particle-count preview.
    let previewCount = 0;
    let previewTimer = 0;

    // Toggle between the sphere-impostor renderer and the screen-space surface.
    // Impostors only draw when a real sim exists AND spheres mode is selected.
    function applyRenderMode(spheres: boolean): void {
        renderSpheres = spheres;
        particleTask.setEnabled(spheres && realSim !== null);
        surfaceTask.setMode(spheres ? "blit" : "surface");
        canvas.dataset.render = spheres ? "spheres" : "surface";
    }

    function disposeRealSim(): void {
        if (realSim) {
            // Rebind BOTH render tasks OFF the real sim before destroying its buffers.
            particleTask.setSim(placeholderSim);
            surfaceTask.setSim(placeholderSim);
            realSim.dispose();
            realSim = null;
        }
    }

    // Build the active-method sim from a world-space seed and point both render tasks
    // at it. Both backends are seeded with the EXACT sampled positions (initialPositions)
    // so the melt starts from the identical blob regardless of solver; the per-method
    // physics-slider values drive the solver params (physValues[currentMethod]).
    function buildRealSim(positions: Float32Array, count: number, radius: number, worldTop: number): void {
        disposeRealSim();
        // Grid cell / smoothing radius ≈ 2.4× particle radius (the fluid demo's
        // dx/radius ratio), floored so a very small radius can't explode the cell
        // count. Used as the MLS grid cell AND the PBF smoothing radius h (both scale
        // with particle spacing).
        const dx = Math.max(radius * 2.4, 0.18);
        const phys = physValues[currentMethod]!;
        // Bounds cover the fall + spread region and drop one unit BELOW ground so the
        // ground BC (scene SDF) isn't fighting the grid's own domain-border wall.
        const boundsMin: [number, number, number] = [-DOMAIN_HALF, -1, -DOMAIN_HALF];
        const boundsMax: [number, number, number] = [DOMAIN_HALF, Math.max(worldTop + 3, 8), DOMAIN_HALF];
        if (currentMethod === "PBF") {
            // Position-Based Fluids configured for a FREE blob-on-ground melt (PBF is
            // normally used for confined tanks): the scene SDF confines each particle
            // against the ground per-step; there is no tank wall. PBF needs a WIDER
            // kernel than the MLS grid cell for a stable poly6 density estimate — ~4×
            // particle radius (matching the fluid demo's h≈0.4 at radius 0.09), so each
            // particle sees ~40+ neighbours. Too small an h (e.g. the MLS dx≈2.4r) leaves
            // the density noisy → the constraint solver over-corrects and the dense
            // lattice fill EXPLODES. restDensity comes from the slider (default 341 ≈ the
            // poly6 rest density n≈1/cellVol for this packing) — see report notes.
            const pbfH = Math.max(radius * 4.0, 0.3);
            realSim = createPbfSim(engine, {
                count,
                particleRadius: radius,
                initialPositions: positions, // world-space exact fill (no random draw)
                smoothingRadius: pbfH,
                boundsMin,
                boundsMax,
                groundY: GROUND_Y,
                maxPerCell: 48,
                gravity: phys.gravity,
                viscosity: phys.viscosity,
                relaxation: phys.relaxation,
                scorr: phys.scorr,
                iterations: phys.iterations,
                restDensity: phys.restDensity,
                boundaryDensity: phys.boundaryDensity,
            });
        } else {
            // MLS-MPM (grid-transfer): cell size dx set at creation; the ground SDF
            // floors the puddle. Slider values drive the EOS + damping tuning.
            realSim = createMlsMpmSim(engine, {
                count,
                particleRadius: radius,
                initialPositions: positions, // world-space (offset applied by caller)
                boundsMin,
                boundsMax,
                dx,
                groundY: GROUND_Y,
                gravity: phys.gravity,
                stiffness: phys.stiffness,
                viscosity: phys.viscosity,
                restDensity: phys.restDensity,
                substeps: phys.substeps,
                damping: phys.damping,
                affineDamping: phys.affineDamping,
                groundDamp: phys.groundDamp,
                groundDampHeight: phys.groundDampHeight,
                restitution: phys.restitution,
            });
        }
        realSim.setSceneSdf(groundSdf);
        particleTask.setSim(realSim);
        surfaceTask.setSim(realSim);
        applyRenderMode(renderSpheres); // (re)enable impostors iff spheres mode
    }

    // Enter the paused "filled" state: liquid shown, sim held static, Melt armed.
    function showFilled(count: number): void {
        setMeshVisible(currentEnemy, false);
        melting = false;
        particleCount = count;
        canvas.dataset.particleCount = String(count);
        countValue.textContent = String(count);
        state = "filled";
        liquefyBtn.disabled = false;
        meltBtn.disabled = false;
        meltBtn.textContent = "Melt";
        setStatus(`filled (paused): ${count} pts — press Melt`);
    }

    function resetToSolid(): void {
        melting = false;
        disposeRealSim();
        particleTask.setEnabled(false);
        setMeshVisible(currentEnemy, true);
        state = "solid";
        particleCount = 0;
        canvas.dataset.particleCount = "0";
        countValue.textContent = "0";
        meltBtn.textContent = "Melt";
        meltBtn.disabled = true;
        liquefyBtn.disabled = false;
        setStatus("solid — pick a shape, then Liquefy");
    }

    function selectEnemy(key: EnemyKey): void {
        const mesh = enemies[key];
        if (!mesh) {
            // Loaded enemy not ready (or failed) — ignore and keep the current shape.
            enemySelect.value = currentKey;
            return;
        }
        resetToSolid();
        setMeshVisible(currentEnemy, false);
        currentKey = key;
        currentEnemy = mesh;
        setMeshVisible(currentEnemy, true);
        cachedPositions = null; // a new shape invalidates the cached fill
        computePreview();
    }

    function toggleMelt(): void {
        if (!realSim) {
            return;
        }
        melting = !melting;
        state = melting ? "melting" : "filled";
        meltBtn.textContent = melting ? "Pause" : "Melt";
        setStatus(`${melting ? "melting" : "paused"}: ${particleCount} pts`);
    }

    // The full CPU volume fill (Liquefy). Recomputes from the CURRENT enemy / mode /
    // radius, refreshes the cache, and enters the paused "filled" state.
    function runSample(): void {
        const mesh = currentEnemy;
        const positions = mesh._cpuPositions!;
        const indices = mesh._cpuIndices!;
        const t0 = performance.now();
        // Sample in the mesh's LOCAL space (factory geometry is centred at the
        // origin; the enemy is only translated, never scaled/rotated).
        const result = sampleMeshVolume({ positions, indices, radius: radiusValue, mode: modeValue });
        const elapsed = performance.now() - t0;
        // eslint-disable-next-line no-console
        console.info(`[liquefactor] ${currentKey}: sampled ${result.count} particles (mode=${modeValue}, radius=${radiusValue.toFixed(3)}) in ${elapsed.toFixed(1)} ms`);

        if (result.count === 0) {
            state = "solid";
            liquefyBtn.disabled = false;
            setStatus("0 particles — decrease radius and retry");
            return;
        }

        // Bake the mesh world offset into the sampled positions → world-space seed.
        const off = mesh.position;
        const p = result.positions;
        for (let i = 0; i < p.length; i += 3) {
            p[i] = p[i]! + off.x;
            p[i + 1] = p[i + 1]! + off.y;
            p[i + 2] = p[i + 2]! + off.z;
        }
        const worldTop = result.bounds.max[1] + off.y;

        // Cache a PRISTINE copy of the world-space seed for Reset (no re-sample).
        cachedPositions = new Float32Array(p);
        cachedCount = result.count;
        cachedRadius = result.radius;
        cachedWorldTop = worldTop;

        buildRealSim(p, result.count, result.radius, worldTop);
        showFilled(result.count);
    }

    function liquefy(): void {
        if (state === "sampling") {
            return; // re-entrancy guard during the synchronous CPU sample
        }
        state = "sampling";
        liquefyBtn.disabled = true;
        setStatus("sampling… (CPU volume fill)");
        // Defer so the "sampling…" label paints before the synchronous CPU work blocks.
        requestAnimationFrame(() => requestAnimationFrame(() => runSample()));
    }

    // Reset replays the CACHED sample (no re-sample, no solid mesh). Falls back to the
    // solid mesh only when nothing has been liquefied yet.
    function reset(): void {
        if (!cachedPositions) {
            resetToSolid();
            return;
        }
        // Rebuild from a fresh copy so the cache stays pristine for future resets.
        buildRealSim(new Float32Array(cachedPositions), cachedCount, cachedRadius, cachedWorldTop);
        showFilled(cachedCount);
    }

    // Switch the fluid solver (PBF ↔ MLS-MPM). If already liquefied, dispose the old
    // sim and rebuild the new-method sim from the CACHED fill (exact re-seed), pointing
    // both render tasks at it and returning to the paused "filled" state (user presses
    // Melt). If not yet liquefied, just record the method for the next Liquefy. Always
    // resyncs the physics sliders to the new method.
    function switchMethod(method: string): void {
        if (method === currentMethod) {
            return;
        }
        currentMethod = method;
        canvas.dataset.method = method;
        controls.setMethod(method); // sync the component's internal current method
        controls.rebuildPhysics(method); // rebuild the physics-slider block for the method
        if (cachedPositions) {
            // Rebuild from a fresh copy so the cache stays pristine.
            buildRealSim(new Float32Array(cachedPositions), cachedCount, cachedRadius, cachedWorldTop);
            showFilled(cachedCount); // fresh sim is paused → filled; user presses Melt
        }
    }

    // Predicted particle count for the current enemy / mode / radius, WITHOUT
    // liquefying. createVolumeSampler builds only the SDF + lattice seed (no SPH),
    // so this is fast for every mode.
    function computePreview(): void {
        const mesh = currentEnemy;
        const positions = mesh._cpuPositions!;
        const indices = mesh._cpuIndices!;
        try {
            previewCount = createVolumeSampler({ positions, indices, radius: radiusValue, mode: modeValue }).count;
        } catch (err) {
            previewCount = 0;
            // eslint-disable-next-line no-console
            console.warn("[liquefactor] preview sample failed", err);
        }
        previewValue.textContent = `≈ ${previewCount.toLocaleString()} particles`;
        canvas.dataset.previewCount = String(previewCount);
    }

    // ── Control panel ──────────────────────────────────────────────────────
    // The RENDER section is provided by the shared reusable component
    // (./fluid/controls-panel.ts); this demo owns only the mesh-sampling controls
    // (Enemy / mode / radius / preview / Liquefy / Melt / Reset / count), mounted into
    // the component's top "Demo" slot below.
    const PANEL_STYLE =
        "position:fixed;top:12px;left:12px;z-index:10;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;" +
        "font-size:0.8rem;color:#dfe6ee;background:rgba(12,16,24,0.82);padding:12px 14px;border-radius:10px;" +
        "width:232px;max-height:92vh;overflow-y:auto;box-shadow:0 6px 22px rgba(0,0,0,0.4);";

    const title = document.createElement("div");
    title.textContent = "Liquefactor";
    title.style.cssText = "font-weight:700;font-size:0.95rem;margin-bottom:2px;";
    const subtitle = document.createElement("div");
    subtitle.textContent = "sampleMeshVolume → GPU fluid surface";
    subtitle.style.cssText = "color:#8fa4bc;margin-bottom:10px;";

    function labelledRow(text: string, control: HTMLElement): HTMLDivElement {
        const row = document.createElement("div");
        row.style.cssText = "margin-bottom:8px;";
        const lab = document.createElement("div");
        lab.textContent = text;
        lab.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
        row.append(lab, control);
        return row;
    }

    function styleSelect(sel: HTMLSelectElement): void {
        sel.style.cssText = "width:100%;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
    }

    // Enemy selector. Procedural shapes are ready immediately; the two LOADED-ASSET
    // enemies start DISABLED (" (loading…)") and become selectable once their async
    // fetch + merge completes (or flip to " (unavailable)" if the load fails).
    const enemySelect = document.createElement("select");
    styleSelect(enemySelect);
    const enemyOptions = {} as Record<EnemyKey, HTMLOptionElement>;
    for (const key of Object.keys(ENEMY_LABELS) as EnemyKey[]) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = ENEMY_LABELS[key];
        if (LOADED_ENEMY_KEYS.includes(key)) {
            opt.disabled = true;
            opt.textContent = `${ENEMY_LABELS[key]} (loading…)`;
        }
        enemyOptions[key] = opt;
        enemySelect.append(opt);
    }
    enemySelect.onchange = () => selectEnemy(enemySelect.value as EnemyKey);

    function markEnemyReady(key: EnemyKey): void {
        const opt = enemyOptions[key];
        opt.disabled = false;
        opt.textContent = ENEMY_LABELS[key];
    }
    function markEnemyUnavailable(key: EnemyKey): void {
        const opt = enemyOptions[key];
        opt.disabled = true;
        opt.textContent = `${ENEMY_LABELS[key]} (unavailable)`;
    }

    // Kick off the two model loads in the BACKGROUND — startup stays synchronous on the
    // torus knot; each loaded enemy is merged into ONE mesh, added to the (already
    // registered) scene hidden, and unlocked when ready. Non-blocking: never awaited by
    // main(), so registerScene/startEngine are not gated on the network fetch.
    function loadEnemyAsset(key: EnemyKey, load: () => Promise<AssetContainer>): void {
        void (async () => {
            try {
                const asset = await load();
                const mesh = mergeAssetToEnemyMesh(asset, `enemy_${key}`);
                addToScene(scene, mesh); // dynamic add after boot → materialized via material-swap drain
                setMeshVisible(mesh, false);
                enemies[key] = mesh;
                markEnemyReady(key);
            } catch (err) {
                // eslint-disable-next-line no-console
                console.warn(`[liquefactor] failed to load "${key}" enemy`, err);
                markEnemyUnavailable(key);
            }
        })();
    }
    loadEnemyAsset("dude", () => loadBabylon(engine, DUDE_URL));
    loadEnemyAsset("hauntedHouse", () => loadGltf(engine, HAUNTED_HOUSE_URL));

    // Sampling mode selector.
    const modeSelect = document.createElement("select");
    styleSelect(modeSelect);
    for (const m of ["dense", "regular", "kugelstadt2021"] as VolumeSamplingMode[]) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m === "kugelstadt2021" ? "kugelstadt2021 (SPH, slow)" : m;
        modeSelect.append(opt);
    }
    modeSelect.value = modeValue;
    modeSelect.onchange = () => {
        modeValue = modeSelect.value as VolumeSamplingMode;
        computePreview(); // mode change recomputes immediately
    };

    // Particle radius slider — updates the label live, DEBOUNCES the count preview so
    // dragging stays smooth (and recomputes on release via `change`).
    const radiusInput = document.createElement("input");
    radiusInput.type = "range";
    radiusInput.min = "0.01";
    radiusInput.max = "0.2";
    radiusInput.step = "0.005";
    radiusInput.value = String(radiusValue);
    radiusInput.style.cssText = "width:100%;";
    const radiusVal = document.createElement("span");
    radiusVal.style.cssText = "color:#9fb4cc;float:right;";
    radiusVal.textContent = radiusValue.toFixed(3);
    radiusInput.oninput = () => {
        radiusValue = parseFloat(radiusInput.value);
        radiusVal.textContent = radiusValue.toFixed(3);
        window.clearTimeout(previewTimer);
        previewTimer = window.setTimeout(() => computePreview(), 180);
    };
    radiusInput.onchange = () => {
        window.clearTimeout(previewTimer);
        computePreview();
    };
    const radiusLabelWrap = document.createElement("div");
    radiusLabelWrap.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    radiusLabelWrap.append(document.createTextNode("Particle radius"), radiusVal);
    const radiusRow = document.createElement("div");
    radiusRow.style.cssText = "margin-bottom:8px;";
    radiusRow.append(radiusLabelWrap, radiusInput);

    // Live pre-Liquefy count preview (distinct from the post-Liquefy "Particles:").
    const previewRow = document.createElement("div");
    previewRow.style.cssText = "margin:4px 0 6px;padding:5px 7px;background:rgba(43,108,176,0.18);border-radius:6px;";
    const previewValue = document.createElement("span");
    previewValue.style.cssText = "color:#8fd0ff;font-weight:700;";
    previewValue.textContent = "≈ 0 particles";
    previewRow.append(previewValue);

    function makeButton(text: string): HTMLButtonElement {
        const b = document.createElement("button");
        b.textContent = text;
        b.style.cssText =
            "width:100%;padding:6px;margin-top:4px;border:0;border-radius:6px;cursor:pointer;" + "background:#2b6cb0;color:#fff;font-weight:600;";
        return b;
    }

    const liquefyBtn = makeButton("Liquefy");
    liquefyBtn.id = "liq-liquefy";
    liquefyBtn.onclick = () => liquefy();

    const meltBtn = makeButton("Melt");
    meltBtn.id = "liq-melt";
    meltBtn.style.background = "#805ad5";
    meltBtn.disabled = true;
    meltBtn.onclick = () => toggleMelt();

    const resetBtn = makeButton("Reset");
    resetBtn.id = "liq-reset";
    resetBtn.style.background = "#4a5568";
    resetBtn.onclick = () => reset();

    const countRow = document.createElement("div");
    countRow.style.cssText = "margin-top:10px;color:#b6c4d6;";
    const countValue = document.createElement("span");
    countValue.style.cssText = "color:#e6edf5;font-weight:700;";
    countValue.textContent = "0";
    countRow.append(document.createTextNode("Particles: "), countValue);

    const status = document.createElement("div");
    status.style.cssText = "margin-top:6px;color:#8fa4bc;min-height:1.1em;";
    function setStatus(text: string): void {
        status.textContent = text;
        canvas.dataset.state = state;
    }

    // ── Shared RENDER + GENERAL + PHYSICS controls (reusable component) ───────
    // The surface-render tunables (Water color / Absorption / Particle size /
    // Refraction / Specular / depth+thickness blur / Surface filter / narrow-range /
    // Half rendering / Thickness downscale / Render-as-spheres) plus the GENERAL
    // method dropdown (PBF ↔ MLS-MPM) and the per-method PHYSICS sliders are provided
    // by the shared component. The "Particles" dropdown, container toggle, Foam, Debug
    // and GPU panel are hidden; the "Physics particle size" control is hidden too
    // (Liquefactor's own "Particle radius" is its particle size). The mesh-sampling
    // controls above are mounted into the component's top "Demo" slot.
    const controls = createFluidControlsPanel({
        hideParticles: true,
        hideMethod: false,
        hideContainerToggle: true,
        hideFoam: true,
        hidePhysics: false,
        hidePhysScale: true,
        hideDebug: true,
        hideGpuTiming: true,
        panelStyle: PANEL_STYLE,
        schemas: LIQ_SCHEMAS,
        methods: ["PBF", "MLS-MPM"],
        particleCounts: [],
        initial: {
            method: currentMethod,
            count: 0,
            physScale: 1,
            color: DEF_COLOR,
            absorption: DEF_ABSORPTION,
            size: DEF_SIZE,
            refraction: DEF_REFRACTION,
            specular: DEF_SPECULAR,
            depthBlur: DEF_DEPTH_BLUR,
            depthBlurThreshold: DEF_DEPTH_BLUR_THRESHOLD,
            thicknessBlur: DEF_THICKNESS_BLUR,
            half: DEF_HALF,
            thicknessDownscale: DEF_THICKNESS_DOWNSCALE,
            surfaceFilter: DEF_SURFACE_FILTER,
            narrowDelta: DEF_NARROW_DELTA,
            narrowMu: DEF_NARROW_MU,
            anisotropic: false,
            renderMode: "surface",
            debug: "none",
            showContainer: true,
            foam: {
                enabled: false,
                kTa: 40,
                kWc: 40,
                kb: 0.8,
                kd: 0.5,
                tMin: 0.3,
                tMax: 2.0,
                poolScale: 3,
                size: 1,
                blurRadius: 4,
                lightIntensity: 0.9,
                ambient: 0.5,
                aoStrength: 0.5,
                normalStrength: 6,
                debugTexture: "off",
                softness: 0.25,
                density: 1.6,
                subsurfaceStrength: 0.4,
            },
        },
        on: {
            onMethod: (m) => switchMethod(m),
            onRenderMode: (spheres) => applyRenderMode(spheres),
            onColor: (rgb) => surfaceTask.setFluidColor(rgb),
            onAbsorption: (v) => surfaceTask.setAbsorption(v),
            onParticleSize: (s) => {
                surfaceTask.setSizeScale(s);
                particleTask.setSizeScale(s);
            },
            onRefraction: (v) => surfaceTask.setRefractionStrength(v),
            onSpecular: (v) => surfaceTask.setSpecularPower(v),
            onDepthBlur: (size, threshold) => surfaceTask.setDepthBlur(size, threshold),
            onThicknessBlur: (v) => surfaceTask.setThicknessBlur(v),
            onHalf: (on) => surfaceTask.setHalfRender(on),
            onSurfaceFilter: (m) => surfaceTask.setSurfaceFilter(m),
            onNarrowRange: (delta, mu) => surfaceTask.setNarrowRange(delta, mu),
            onThicknessDownscale: (v) => surfaceTask.setThicknessDownscale(v),
            // Physics sliders: apply LIVE to the running sim AND remember per-method so a
            // freshly (re)built sim starts from the current slider values.
            onPhysicsParam: (key, value) => {
                physValues[currentMethod]![key] = value;
                realSim?.setParam(key, value);
            },
            // Physics "Reset" button: restore this method's physics to the schema
            // defaults, push them to the live sim, and refresh the sliders.
            onReset: () => {
                const defaults = SCHEMA_DEFAULTS[currentMethod]!;
                physValues[currentMethod] = { ...defaults };
                for (const [k, v] of Object.entries(defaults)) {
                    realSim?.setParam(k, v);
                }
                controls.setPhysics(defaults); // sync the component's internal schema values
                controls.rebuildPhysics(currentMethod); // refresh the DOM sliders to the defaults
            },
        },
    });

    // Mount the demo-specific mesh-sampling controls into the component's top "Demo" slot.
    controls.demoSlot.append(
        title,
        subtitle,
        labelledRow("Enemy", enemySelect),
        labelledRow("Sampling mode", modeSelect),
        radiusRow,
        previewRow,
        liquefyBtn,
        meltBtn,
        resetBtn,
        countRow,
        status
    );
    document.body.append(controls.root);
    canvas.dataset.method = currentMethod;
    setStatus("solid — pick a shape, then Liquefy");
    computePreview(); // initial pre-Liquefy count preview

    // Expose handlers for headless smoke tests / programmatic driving.
    (window as unknown as { __liquefactor?: unknown }).__liquefactor = {
        liquefy,
        toggleMelt,
        reset: () => reset(),
        setMethod: (m: string) => switchMethod(m),
        getMethod: () => currentMethod,
        getCount: () => particleCount,
        getState: () => state,
        getPreviewCount: () => previewCount,
        getEnemy: () => currentKey,
        getTriCount: () => (currentEnemy._cpuIndices ? currentEnemy._cpuIndices.length / 3 : 0),
    };

    // ── Per-frame loop ─────────────────────────────────────────────────────
    onBeforeRender(scene, (deltaMs: number) => {
        if (melting && realSim) {
            const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
            realSim.step(engine._currentEncoder, dt);
        }
    });

    // Ensure the env finished loading (skybox + surface reflections wired) before we
    // build the scene, so the HDR skybox renders as the background from frame 0.
    await envReady;
    await registerScene(engine, scene);
    await startEngine(engine);

    canvas.dataset.particleCount = String(particleCount);
    canvas.dataset.ready = "true";
}

main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error(err);
    const c = document.getElementById("renderCanvas") as HTMLCanvasElement | null;
    if (c) {
        c.dataset.error = String(err);
    }
});
