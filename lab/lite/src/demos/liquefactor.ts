// Liquefactor demo — MULTI-TARGET shooting gallery with concurrent GPU fluid sims.
//
// Three "enemy" meshes float over the ground. Click one to LIQUEFY it: its surface
// dissolves radially from the hit point (the material clips inside a growing world-space
// "front"), the interior is CPU volume-sampled into particles, and an independent GPU
// fluid sim (grid centred on THAT mesh, at ground level) launches the blob upward and
// melts it into a puddle. Multiple meshes can be dissolving / running at the same time —
// each mesh owns its own sim (grid, particle count, physics). A few seconds after a mesh
// liquefies, its sim fades away (sinks through the floor) and the mesh is gone for good.
// "Restart" stops every sim and restores all three solid meshes.
//
// Rendering keeps demo-fluid's screen-space surface but composites ALL live sims in ONE
// pass: every frame each sim's render positions are copied into a shared "combined" buffer
// and a single fluid-surface task reconstructs the water from it. The scene (ground + HDR
// skybox, minus any dissolving foe) draws into `sceneColorRT`; the surface pass composites
// that into the swapchain; then a single foe-overlay pass draws every dissolving foe on top,
// each clipping inside its own front to reveal the one water render beneath.

import {
    addAnimationGroups,
    addTask,
    addToScene,
    attachControl,
    computeDeformedPositions,
    createAnimationManager,
    createArcRotateCamera,
    createDirectionalLight,
    createEngine,
    createGpuPicker,
    createGround,
    createHemisphericLight,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createStandardMaterial,
    createTransformNode,
    enableMaterialPlugins,
    isPbrMaterial,
    loadEnvironment,
    loadGltf,
    onBeforeRender,
    pauseAnimation,
    pickAsync,
    playAnimation,
    registerScene,
    setMeshVisible,
    startEngine,
    updateAnimationManager,
} from "babylon-lite";
import type { AnimationGroup, EnvironmentTextures, Material, Mesh, Renderable, SceneNode } from "babylon-lite";
import { createLiquefyPlugin } from "./liquefy-plugin.js";
import type { LiquefyState } from "./liquefy-plugin.js";
import { createMlsMpmSim } from "babylon-lite/fluid/mls-mpm-sim.js";
import { createPbfSim } from "babylon-lite/fluid/pbf-sim.js";
import { createPbMpmSim, pbmpmParamKeysForMaterial } from "babylon-lite/fluid/pbmpm-sim.js";
import type { FluidSim, ForceFieldSpec, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { createFluidSurfaceTask } from "babylon-lite/fluid/fluid-surface-render.js";
import { sampleMeshVolume } from "babylon-lite/fluid/volume-sampling/index.js";
import type { VolumeSamplingMode } from "babylon-lite/fluid/volume-sampling/index.js";
import { buildHdrSkyboxRenderable } from "babylon-lite/material/pbr/background-hdr-skybox.js";
import { createFluidControlsPanel, DEFAULT_FLUID_SCHEMAS } from "babylon-lite/fluid/controls-panel.js";
import type { PhysSchemaEntry } from "babylon-lite/fluid/controls-panel.js";
import { createFluidProfiler } from "./fluid/gpu-profiler.js";
import type { FluidProfilerImpl } from "./fluid/gpu-profiler.js";
import { demoAssetUrl } from "./demo-asset-url.js";

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

// World layout. Foes rest on the ground plane at y = 0; each sim grid drops one unit below
// ground so the ground BC isn't fighting the grid's own border.
const GROUND_Y = 0;
const SPREAD_MARGIN = 9; // extra half-width (world units) around a mesh footprint for puddle spread
const LIQUEFY_SPEED = 6; // dissolve front growth (world units / s)
const LIFETIME = 5.0; // seconds a running sim lives before it starts fading
const FADE_DUR = 1.2; // seconds the alpha fade-out takes before dispose
const WRIGGLE_AMP = 0.04; // cartoon "pain" jitter amplitude (world units)
const IMPULSE_RADIAL_BASE = 18; // outward explosion accel from the volume centre, all directions (× impulse intensity)
const IMPULSE_UP_BASE = 6; // gentle uniform upward lift so the burst arcs up a little (× impulse intensity)
const MAX_TOTAL = 600000; // combined render-buffer capacity (particles across all live sims)

// Studio HDR environment — drives the fluid-surface reflections + the skybox background.
const ENV_STUDIO_URL = "https://playground.babylonjs.com/textures/environment.env";
const SUN_DIR: [number, number, number] = [-0.4, -0.82, -0.45];

// Textured glTF foes — each is loaded, auto-fit to a common size, sat on the ground, and (when the
// model is skinned/animated) played + sampled in its CURRENT deformed pose. Their diffuse textures
// drive the per-particle "Use mesh colours" render. Alien + CesiumMan are animated (skinned).
const MODEL_FOES: { key: string; url: string; x: number; ry?: number; surfaceOnly?: boolean; scale?: number }[] = [
    { key: "alien", url: "https://playground.babylonjs.com/scenes/Alien/Alien.gltf", x: -15 },
    { key: "barrel", url: "https://assets.babylonjs.com/meshes/ExplodingBarrel.glb", x: -5, ry: -Math.PI / 2, surfaceOnly: true },
    { key: "house", url: "https://assets.babylonjs.com/meshes/haunted_house.glb", x: 5, surfaceOnly: true, scale: 4 },
    { key: "cesium", url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb", x: 15 },
];
const MODEL_TARGET_SIZE = 7; // auto-fit: scale each model so its largest dimension ≈ this many world units

// PB-MPM material (0 liquid, 1 elastic, 2 sand, 3 viscoelastic). Only PB-MPM branches on it.
const PBMPM_MATERIAL_LABELS: [string, number][] = [
    ["Liquid", 0],
    ["Elastic", 1],
    ["Sand", 2],
    ["Viscoelastic", 3],
];

// Impulse force field: an EXPLOSION FROM THE INSIDE. Every particle inside the blast radius is
// pushed radially OUTWARD from the volume centre (push.x, ~uniform through the core), plus a gentle
// uniform upward lift (push.y) so the burst arcs up a little. center.xyz = volume centre, center.w =
// blast radius (set well beyond the blob so the whole volume is in the flat region).
const IMPULSE_WGSL = /* wgsl */ `
fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> {
    let center = forceFieldParams.center.xyz;
    let radius = max(forceFieldParams.center.w, 1.0e-4);
    let toParticle = pos - center;
    let dist = length(toParticle);
    var f = vec3<f32>(0.0, forceFieldParams.push.y, 0.0); // uniform upward lift
    if (dist < radius) {
        let dir = select(vec3<f32>(0.0, 1.0, 0.0), toParticle / max(dist, 1.0e-4), dist > 1.0e-4);
        // ~uniform outward blast through the core; ramp up over the innermost 15% to avoid a hard
        // direction flip on particles sitting right at the centre.
        let core = smoothstep(0.0, 0.15, dist / radius);
        f += dir * forceFieldParams.push.x * core;
    }
    return f * dt;
}`;

function hexToRgb(hex: string): [number, number, number] {
    return [parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255, parseInt(hex.slice(5, 7), 16) / 255];
}

type InstancePhase = "solid" | "dissolving" | "fluid" | "fading" | "gone";

type TexInfo = { view: GPUTextureView; width: number; height: number };

interface Instance {
    readonly key: string;
    readonly meshes: Mesh[]; // display sub-meshes (1 procedural, N for the skinned model)
    readonly materials: Material[]; // clip-plugin hosts + per-frame UBO bump
    readonly root: SceneNode; // wriggle / placement target (the mesh itself, or a parent transform)
    readonly deformable: boolean; // sample the CURRENT animated pose (skinned model) instead of static geometry
    readonly surfaceOnly: boolean; // skip volume sampling (hollow prop) → dense barycentric-UV surface shell
    readonly x: number;
    readonly baseY: number; // world Y the mesh rests at (used for the static-sample world offset)
    readonly homePos: [number, number, number]; // resting root position (auto-fit centred); Restart resets here
    readonly diffuseTexs: TexInfo[]; // distinct base-colour textures across the sub-meshes (per-particle colour source)
    readonly meshTexIndex: number[]; // per display sub-mesh: its index into diffuseTexs (-1 if untextured)
    readonly baseColor: [number, number, number]; // fallback per-particle colour when there's no texture
    colorBuffer: GPUBuffer | null; // per-particle RGBA colour (filled once at shot time), fed to the surface renderer
    animation: AnimationGroup | null; // looping walk (paused on shot, resumed on Restart)
    readonly liquefyState: LiquefyState;
    readonly impulseBuffer: GPUBuffer;
    readonly impulseSpec: ForceFieldSpec;
    sim: FluidSim | null;
    phase: InstancePhase;
    sampling: boolean; // true while the worker is volume-sampling this foe (before dissolve starts)
    sampleId: number; // id of the in-flight sample request (stale results are ignored)
    maxR: number;
    volCenter: [number, number, number]; // world-space centre of the sampled volume (explosion origin)
    volRadius: number; // half-diagonal of the sampled AABB (explosion reach)
    count: number;
    radius: number;
    impulseRemaining: number;
    fluidElapsed: number;
    fadeElapsed: number;
    // wriggle
    wriggling: boolean; // mesh pain-shake active (starts on click, before the sim exists)
    wriggleBase: [number, number, number];
    wriggleWaterBase: Float32Array | null;
    wriggleWaterScratch: Float32Array | null;
}

async function main(): Promise<void> {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;

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

    const engine = await createEngine(canvas, { msaaSamples: 1, requiredLimits });
    const scene = createSceneContext(engine, { defaultRenderTask: false });

    const cam = createArcRotateCamera(-Math.PI / 2, 1.05, 46, { x: -4, y: 3.5, z: 0 });
    cam.nearPlane = 0.1;
    cam.farPlane = 200;
    scene.camera = cam;
    attachControl(cam, canvas, scene);

    addToScene(scene, createHemisphericLight([0.2, 1, 0.3], 0.8));
    const sun = createDirectionalLight(SUN_DIR, 2.0);
    sun.position.set(12, 20, 10);
    addToScene(scene, sun);

    const ground = createGround(engine, { width: 60, height: 60, subdivisions: 1 });
    const groundMat = createStandardMaterial();
    groundMat.diffuseColor = [0.22, 0.24, 0.28];
    groundMat.specularColor = [0.04, 0.04, 0.05];
    ground.material = groundMat;
    addToScene(scene, ground);

    const device = engine._device;

    // Opt-in GPU timing (timestamp-query). Null when the host GPU lacks the feature — the panel's
    // GPU section then shows "unavailable". Wired onto every sim + the surface task.
    let profiler: FluidProfilerImpl | null = null;
    try {
        profiler = createFluidProfiler(device);
    } catch {
        profiler = null;
    }

    // ── Enemy instances (loaded glTF models) ─────────────────────────────────
    // Each instance owns its OWN material + liquefy state so its dissolve clip is independent.
    function makeImpulse(key: string): { impulseBuffer: GPUBuffer; impulseSpec: ForceFieldSpec } {
        const impulseBuffer = device.createBuffer({ label: `liq-impulse-${key}`, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const impulseSpec: ForceFieldSpec = { struct: "struct ForceFieldParams { center: vec4<f32>, push: vec4<f32>, };", wgsl: IMPULSE_WGSL, buffer: impulseBuffer };
        return { impulseBuffer, impulseSpec };
    }

    const isMeshNode = (node: SceneNode): node is Mesh => "_gpu" in node && "material" in node;

    const instances: Instance[] = [];
    const meshToInstance = new Map<Mesh, Instance>();
    const instanceOf = (m: unknown): Instance | undefined => meshToInstance.get(m as Mesh);
    const animManager = createAnimationManager({ engine });

    // Load a textured glTF model as a foe: parent it under an auto-fit root (scaled so its largest
    // dimension ≈ MODEL_TARGET_SIZE, centred on cfg.x and sat on the ground), attach the radial-clip
    // liquefy plugin per PBR material, play its first animation if any, and capture its diffuse
    // texture for per-particle colouring. Sampled in its CURRENT deformed pose on a shot.
    async function loadModelInstance(cfg: { key: string; url: string; x: number; ry?: number; surfaceOnly?: boolean; scale?: number }): Promise<void> {
        let asset;
        try {
            asset = await loadGltf(engine, cfg.url);
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn(`[liquefactor] ${cfg.key} load failed`, err);
            return;
        }
        const liquefyState: LiquefyState = { hit: [0, 0, 0], frontR: 0, edge: 0.6, enabled: false };
        const ry = cfg.ry ?? 0; // optional Y spin so a labelled face (e.g. the barrel logo) points at the camera
        const root = createTransformNode(`${cfg.key}_root`, 0, 0, 0, 0, Math.sin(ry / 2), 0, Math.cos(ry / 2), 1, 1, 1);
        const meshes: Mesh[] = [];
        const materials = new Set<Material>();
        const visit = (node: SceneNode): void => {
            if (isMeshNode(node) && node._cpuPositions && node._cpuIndices) {
                meshes.push(node);
                // Attach the radial-clip liquefy plugin (PBR host). This works on SKINNED models
                // because they are materialized in the INITIAL scene build (loaded before registerScene);
                // a dynamic post-boot add leaves the plugin UBO uninitialised and the mesh invisible.
                if (node.material && isPbrMaterial(node.material) && !node.material.plugins?.some((p) => p.name === "liquefy")) {
                    node.material.plugins = [...(node.material.plugins ?? []), createLiquefyPlugin(() => liquefyState, "pbr")];
                }
                if (node.material) materials.add(node.material);
            }
            for (const c of node.children ?? []) visit(c);
        };
        for (const e of asset.entities) {
            if (!("position" in e)) continue; // skip non-node entities (e.g. lights)
            e.parent = root;
            root.children.push(e);
            visit(e);
        }
        if (meshes.length === 0) {
            // eslint-disable-next-line no-console
            console.warn(`[liquefactor] ${cfg.key} has no CPU-geometry meshes`);
            return;
        }
        // Add to the scene FIRST so the world-matrix state is wired up — reading worldMatrix before
        // this yields a partial (pre-hierarchy) transform and mis-fits the model.
        addToScene(scene, root);
        // Auto-fit: world AABB at scale 1 (root at origin) → uniform scale so the largest extent ≈
        // MODEL_TARGET_SIZE, then centre on cfg.x and sit the model's base on the ground.
        let minx = Infinity,
            miny = Infinity,
            minz = Infinity,
            maxx = -Infinity,
            maxy = -Infinity,
            maxz = -Infinity;
        for (const m of meshes) {
            const p = m._cpuPositions!;
            const w = m.worldMatrix as unknown as ArrayLike<number>;
            for (let i = 0; i < p.length; i += 3) {
                const lx = p[i]!,
                    ly = p[i + 1]!,
                    lz = p[i + 2]!;
                const wx = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
                const wy = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
                const wz = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
                if (wx < minx) minx = wx;
                if (wx > maxx) maxx = wx;
                if (wy < miny) miny = wy;
                if (wy > maxy) maxy = wy;
                if (wz < minz) minz = wz;
                if (wz > maxz) maxz = wz;
            }
        }
        const extent = Math.max(maxx - minx, maxy - miny, maxz - minz, 1e-3);
        const scale = (MODEL_TARGET_SIZE / extent) * (cfg.scale ?? 1); // optional per-model size multiplier
        const cx = (minx + maxx) / 2;
        const cz = (minz + maxz) / 2;
        const homePos: [number, number, number] = [cfg.x - cx * scale, GROUND_Y - miny * scale, -cz * scale];
        root.scaling.set(scale, scale, scale);
        root.position.set(homePos[0], homePos[1], homePos[2]);
        const anim = asset.animationGroups?.find((g) => /walk|run|idle/i.test(g.name)) ?? asset.animationGroups?.[0] ?? null;
        if (anim) {
            anim.loopAnimation = true;
            playAnimation(anim);
            addAnimationGroups(animManager, [anim]);
        }
        const { impulseBuffer, impulseSpec } = makeImpulse(cfg.key);
        // Collect the DISTINCT base-colour textures across sub-meshes (a model may use several — the
        // haunted house has two) and record which texture each sub-mesh uses, so every particle can be
        // coloured from ITS OWN mesh's texture. Only real images (>1x1) count; a 1x1 is a flat factor.
        const diffuseTexs: TexInfo[] = [];
        const meshTexIndex: number[] = [];
        for (const m of meshes) {
            const t = (m.material as unknown as { baseColorTexture?: TexInfo } | undefined)?.baseColorTexture;
            if (!t?.view || t.width <= 1 || t.height <= 1) {
                meshTexIndex.push(-1);
                continue;
            }
            let idx = diffuseTexs.findIndex((d) => d.view === t.view);
            if (idx < 0) {
                idx = diffuseTexs.length;
                diffuseTexs.push({ view: t.view, width: t.width, height: t.height });
            }
            meshTexIndex.push(idx);
        }
        const inst: Instance = {
            key: cfg.key,
            meshes,
            materials: [...materials],
            root,
            deformable: true,
            surfaceOnly: cfg.surfaceOnly ?? false,
            x: cfg.x,
            baseY: GROUND_Y,
            homePos,
            diffuseTexs,
            meshTexIndex,
            baseColor: [0.8, 0.8, 0.8],
            colorBuffer: null,
            animation: anim,
            liquefyState,
            impulseBuffer,
            impulseSpec,
            sim: null,
            phase: "solid",
            sampling: false,
            sampleId: 0,
            maxR: 0,
            volCenter: [0, 0, 0],
            volRadius: 1,
            count: 0,
            radius: 0.08,
            impulseRemaining: 0,
            fluidElapsed: 0,
            fadeElapsed: 0,
            wriggling: false,
            wriggleBase: [0, 0, 0],
            wriggleWaterBase: null,
            wriggleWaterScratch: null,
        };
        instances.push(inst);
        for (const m of meshes) meshToInstance.set(m, inst);
        invalidateFilteredSceneTasks();
        setStatus();
    }

    const picker = createGpuPicker(scene);

    // ── Render pipeline ──────────────────────────────────────────────────────
    const depthRT = createRenderTarget({ lbl: "liq-depth", dFormat: "depth24plus", samples: 1, size: engine });
    const sceneColorRT = createRenderTarget({ lbl: "liq-scene-color", format: engine.format, samples: 1, size: engine });
    // Scene bg excludes any DISSOLVING foe (drawn later as a clipping overlay); solid instances
    // render normally so the water refracts them; fluid/fading/gone instances have hidden meshes.
    const sceneTask = createRenderTask(
        {
            name: "scene",
            rt: sceneColorRT,
            depth: depthRT,
            clr: true,
            clrColor: { r: 0.05, g: 0.06, b: 0.1, a: 1 },
            _filterRenderable: (renderable) => instanceOf(renderable.mesh)?.phase !== "dissolving",
        },
        engine,
        scene
    );
    addTask(scene, sceneTask);

    // Ground-plane scene SDF for every sim: positive above the floor, negative below.
    const groundSdfBuffer = device.createBuffer({ label: "liq-scene-sdf", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(groundSdfBuffer, 0, new Float32Array([GROUND_Y, 0, 0, 0]));
    const groundSdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { ground: vec4<f32>, };",
        sdf: "fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return pt.y - sceneSdfParams.ground.x; }",
        buffer: groundSdfBuffer,
    };

    // Combined render buffer aggregating every live sim's particles into ONE surface pass.
    const combinedPos = device.createBuffer({ label: "liq-combined-pos", size: MAX_TOTAL * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const combinedDebug = device.createBuffer({ label: "liq-combined-debug", size: MAX_TOTAL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Per-particle alpha (opt-in on the surface renderer) — 1 for running blobs, ramped 1→0
    // for a blob that is fading out, so ONLY the fading blob's water fades (the shared render
    // can't fade globally). Seeded to 1 so untouched slots stay opaque.
    const combinedAlpha = device.createBuffer({ label: "liq-combined-alpha", size: MAX_TOTAL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const alphaScratch = new Float32Array(MAX_TOTAL).fill(1);
    device.queue.writeBuffer(combinedAlpha, 0, alphaScratch);
    // Per-particle RGBA colour aggregated across sims (opt-in "Use mesh colours" render toggle).
    const combinedColor = device.createBuffer({ label: "liq-combined-color", size: MAX_TOTAL * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    let useMeshColors = false; // UI toggle: tint the water by the liquefied mesh's texture/vertex colours
    // A virtual sim the surface renderer reads live (count + positionBuffer refreshed each frame).
    // When count is 0 (no active sims) the surface task skips all fluid passes and just presents
    // the scene, so the GPU "Surface" timing is 0 while idle.
    const virtualSim = {
        count: 0,
        particleRadius: 0.08,
        surfaceSizeScale: 1,
        positionBuffer: combinedPos,
        velocityBuffer: combinedPos,
        debugBuffer: combinedDebug,
        debugNorm: 1,
        gpuBytes: 0,
        step: () => {},
        reset: () => {},
        setParam: () => {},
        setSceneSdf: () => {},
        setEmitters: () => {},
        setSpawn: () => {},
        setForceField: () => {},
        dispose: () => {},
    };

    const surfaceTask = createFluidSurfaceTask(engine, scene, { bgRT: sceneColorRT, outRT: engine.scRT, depthRT, camera: cam, sim: virtualSim as unknown as FluidSim });
    surfaceTask.setSim(virtualSim as unknown as FluidSim);
    surfaceTask.setProfiler(profiler);
    surfaceTask.setParticleAlpha(combinedAlpha);
    surfaceTask.setParticleColor(combinedColor);
    addTask(scene, surfaceTask);

    // Per-particle colour: for each DISTINCT sub-mesh texture, a compute pass samples that texture at
    // the particle's UV (from the worker) for the particles that belong to it (texIdx == T), gamma-
    // encoding into the instance colour buffer. Particles with no texture keep a baseColor fill. The
    // buffers are aggregated into combinedColor each frame (same layout as combinedPos).
    const COLOR_SAMPLE_WGSL = /* wgsl */ `
struct P { count: u32, tsel: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> uvs: array<vec2<f32>>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outCol: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> texIdx: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    if (texIdx[i] != p.tsel) { return; }          // this pass only fills particles using texture tsel
    let dims = vec2<f32>(textureDimensions(tex, 0));
    let w = fract(uvs[i]);                         // repeat-wrap
    let coord = vec2<i32>(clamp(w * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));
    let c = textureLoad(tex, coord, 0);            // sRGB views decode to linear on load
    // Gamma-ENCODE back to display space: the fluid composite outputs gamma-encoded colour, but the
    // Beer-Lambert tint + overlay use this value directly, so a linear sample renders too dark.
    outCol[i] = vec4<f32>(pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}`;
    const colorPipe = device.createComputePipeline({
        label: "liq-color-sample",
        layout: "auto",
        compute: { module: device.createShaderModule({ label: "liq-color-sample", code: COLOR_SAMPLE_WGSL }), entryPoint: "main" },
    });

    function fillInstanceColor(inst: Instance, uvs: Float32Array | null, texIndices: Uint32Array | null, count: number): void {
        inst.colorBuffer?.destroy();
        const buf = device.createBuffer({ label: `liq-color-${inst.key}`, size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        inst.colorBuffer = buf;
        // baseColor fill first: untextured particles (texIdx == NO_TEX) and any not overwritten below.
        const [r, g, b] = inst.baseColor;
        const scratch = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
            scratch[i * 4] = r;
            scratch[i * 4 + 1] = g;
            scratch[i * 4 + 2] = b;
            scratch[i * 4 + 3] = 1;
        }
        device.queue.writeBuffer(buf, 0, scratch);
        if (!inst.diffuseTexs.length || !uvs || !texIndices || uvs.length < count * 2 || texIndices.length < count) return;
        const uvBuf = device.createBuffer({ label: `liq-uv-${inst.key}`, size: count * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(uvBuf, 0, uvs, 0, count * 2);
        const tiBuf = device.createBuffer({ label: `liq-ti-${inst.key}`, size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(tiBuf, 0, texIndices, 0, count);
        const enc = device.createCommandEncoder();
        const cbufs: GPUBuffer[] = [];
        for (let t = 0; t < inst.diffuseTexs.length; t++) {
            const cbuf = device.createBuffer({ label: `liq-colcount-${inst.key}-${t}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(cbuf, 0, new Uint32Array([count, t, 0, 0]));
            cbufs.push(cbuf);
            const bg = device.createBindGroup({
                layout: colorPipe.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: cbuf } },
                    { binding: 1, resource: { buffer: uvBuf } },
                    { binding: 2, resource: inst.diffuseTexs[t]!.view },
                    { binding: 3, resource: { buffer: buf } },
                    { binding: 4, resource: { buffer: tiBuf } },
                ],
            });
            const pass = enc.beginComputePass();
            pass.setPipeline(colorPipe);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(count / 64));
            pass.end();
        }
        device.queue.submit([enc.finish()]);
        uvBuf.destroy();
        tiBuf.destroy();
        for (const c of cbufs) c.destroy();
    }


    // Foe overlay: after the single water render presents into scRT, draw ALL dissolving foes
    // on top (each clips inside its own front). Depth-aliases the scene-minus-foes depth.
    const foeDepth = createRenderTarget({ lbl: "liq-foe-depth", dFormat: depthRT._descriptor.dFormat, samples: 1, size: engine });
    foeDepth._eager = true;
    foeDepth._ownsDepthTexture = false;
    const foeTask = createRenderTask({ name: "liq-foe", rt: engine.scRT, depth: foeDepth, clr: false, _filterRenderable: (renderable) => instanceOf(renderable.mesh)?.phase === "dissolving" }, engine, scene);
    const foeRecord = foeTask.record.bind(foeTask);
    foeTask.record = (): void => {
        foeDepth._depthTexture = depthRT._depthTexture;
        foeDepth._depthView = depthRT._depthView;
        foeDepth._width = depthRT._width;
        foeDepth._height = depthRT._height;
        foeRecord();
    };
    addTask(scene, foeTask);

    // Encoder-level GPU-timing resolve: after every other pass, close the whole-frame envelope and
    // resolve the timestamp queries. Draws nothing. Must be the LAST task added.
    if (profiler) {
        const prof = profiler;
        addTask(scene, {
            name: "liq-timing-resolve",
            engine,
            scene,
            _passes: [],
            record: (): void => {},
            execute: (): number => {
                prof.frameStop(engine._currentEncoder);
                prof.resolveInto(engine._currentEncoder);
                return 0;
            },
            dispose: (): void => {},
        });
    }

    const invalidateFilteredSceneTasks = (): void => {
        for (const task of [sceneTask, foeTask]) {
            task._lastVersion = -1;
            task._opaqueBundles.length = 0;
        }
    };

    surfaceTask.setDirLight(SUN_DIR);
    surfaceTask.setFluidColor(hexToRgb(DEF_COLOR));
    surfaceTask.setAbsorption(DEF_ABSORPTION);
    surfaceTask.setSizeScale(DEF_SIZE);
    surfaceTask.setRefractionStrength(DEF_REFRACTION);
    surfaceTask.setSpecularPower(DEF_SPECULAR);
    surfaceTask.setDepthBlur(DEF_DEPTH_BLUR, DEF_DEPTH_BLUR_THRESHOLD);
    surfaceTask.setThicknessBlur(DEF_THICKNESS_BLUR);
    surfaceTask.setHalfRender(DEF_HALF);
    surfaceTask.setThicknessDownscale(DEF_THICKNESS_DOWNSCALE);
    surfaceTask.setSurfaceFilter(DEF_SURFACE_FILTER);
    surfaceTask.setNarrowRange(DEF_NARROW_DELTA, DEF_NARROW_MU);
    surfaceTask.setMode("surface");

    enableMaterialPlugins(scene);

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

    // ── Global tuning state (applied to newly built sims) ────────────────────
    let radiusValue = 0.08;
    let modeValue: VolumeSamplingMode = "dense";
    let impulseIntensity = 1.0;
    let currentMethod = "MLS-MPM";
    let currentMaterial = 0;

    const LIQ_SCHEMAS: Record<string, PhysSchemaEntry[]> = Object.fromEntries(
        Object.entries(DEFAULT_FLUID_SCHEMAS).map(([m, entries]) => [
            m,
            entries.map((e) => (m === "PBF" && e.key === "viscosity" ? { ...e, value: 0.35 } : { ...e })),
        ])
    );
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

    // ── Per-instance sim construction (grid centred on the sampled foe, at ground) ──
    function buildInstanceSim(inst: Instance, positions: Float32Array, count: number, radius: number, wMin: readonly [number, number, number], wMax: readonly [number, number, number]): void {
        const dx = Math.max(radius * 2.4, 0.18);
        const phys = physValues[currentMethod]!;
        const cx = (wMin[0] + wMax[0]) / 2;
        const cz = (wMin[2] + wMax[2]) / 2;
        const half = Math.max(wMax[0] - wMin[0], wMax[2] - wMin[2]) / 2 + SPREAD_MARGIN;
        const boundsMin: [number, number, number] = [cx - half, -1, cz - half];
        const boundsMax: [number, number, number] = [cx + half, Math.max(wMax[1] + 3, 8), cz + half];
        let sim: FluidSim;
        if (currentMethod === "PBF") {
            const pbfH = Math.max(radius * 4.0, 0.3);
            sim = createPbfSim(engine, {
                count,
                particleRadius: radius,
                initialPositions: positions,
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
        } else if (currentMethod === "PB-MPM") {
            sim = createPbMpmSim(engine, {
                count,
                particleRadius: radius,
                initialPositions: positions,
                boundsMin,
                boundsMax,
                dx,
                groundY: GROUND_Y,
                material: currentMaterial,
                gravity: phys.gravity,
                substeps: phys.substeps,
                iterations: phys.iterations,
                liquidRelaxation: phys.liquidRelaxation,
                liquidViscosity: phys.liquidViscosity,
                elasticityRatio: phys.elasticityRatio,
                elasticRelaxation: phys.elasticRelaxation,
                frictionAngle: phys.frictionAngle,
                plasticity: phys.plasticity,
                restitution: phys.restitution,
            });
        } else {
            sim = createMlsMpmSim(engine, {
                count,
                particleRadius: radius,
                initialPositions: positions,
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
        sim.setSceneSdf(groundSdf);
        sim.setProfiler?.(profiler);
        inst.sim = sim;
        inst.count = count;
        inst.radius = radius;
        virtualSim.particleRadius = radius;
        virtualSim.surfaceSizeScale = sim.surfaceSizeScale ?? 1;
    }

    type SampleGeom = { positions: Float32Array; indices: Uint32Array; uvs: Float32Array | null; texIndices: Uint32Array | null; ox: number; oy: number; oz: number };
    type SampledFill = { positions: Float32Array; count: number; radius: number; boundsMin: [number, number, number]; boundsMax: [number, number, number]; uvs?: Float32Array | null; texIndices?: Uint32Array | null };
    const LIQUEFY_EDGE = 0.6;
    const NO_TEX = 0xffffffff; // per-vertex/particle texIndex sentinel: no texture → baseColor fill

    // Geometry to feed the volume sampler: LOCAL + a world offset (procedural), or the CURRENT
    // deformed pose already in WORLD space with a zero offset (skinned model). Also emits per-vertex
    // UVs + a per-vertex texture index (which sub-mesh texture to sample) when the instance is textured.
    function sampleGeometry(inst: Instance): SampleGeom | null {
        const wantUv = inst.diffuseTexs.length > 0;
        if (!inst.deformable) {
            const m = inst.meshes[0]!;
            if (!m._cpuPositions || !m._cpuIndices) return null;
            const uvs = wantUv && m._cpuUvs ? m._cpuUvs.slice() : null;
            return { positions: m._cpuPositions.slice(), indices: (m._cpuIndices as Uint32Array).slice(), uvs, texIndices: null, ox: inst.x, oy: inst.baseY, oz: 0 };
        }
        let totalV = 0;
        let totalI = 0;
        let allHaveUv = wantUv;
        const parts: { pos: Float32Array; idx: Uint32Array; uv: Float32Array | null; ti: number; w: ArrayLike<number> }[] = [];
        for (let mi = 0; mi < inst.meshes.length; mi++) {
            const m = inst.meshes[mi]!;
            if (!m._cpuPositions || !m._cpuIndices) continue;
            const local = computeDeformedPositions(m) ?? m._cpuPositions;
            const uv = m._cpuUvs ?? null;
            if (!uv) allHaveUv = false;
            parts.push({ pos: local, idx: m._cpuIndices as Uint32Array, uv, ti: inst.meshTexIndex[mi] ?? -1, w: m.worldMatrix as unknown as ArrayLike<number> });
            totalV += local.length / 3;
            totalI += m._cpuIndices.length;
        }
        if (totalV === 0) return null;
        const positions = new Float32Array(totalV * 3);
        const indices = new Uint32Array(totalI);
        const uvs = allHaveUv ? new Float32Array(totalV * 2) : null;
        const texIndices = wantUv ? new Uint32Array(totalV) : null;
        let vb = 0;
        let ic = 0;
        for (const p of parts) {
            const w = p.w;
            const n = p.pos.length / 3;
            const ti = p.ti >= 0 ? p.ti >>> 0 : NO_TEX;
            for (let v = 0; v < n; v++) {
                const s = v * 3;
                const lx = p.pos[s]!,
                    ly = p.pos[s + 1]!,
                    lz = p.pos[s + 2]!;
                const o = (vb + v) * 3;
                positions[o] = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
                positions[o + 1] = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
                positions[o + 2] = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
                if (uvs && p.uv) {
                    uvs[(vb + v) * 2] = p.uv[v * 2]!;
                    uvs[(vb + v) * 2 + 1] = p.uv[v * 2 + 1]!;
                }
                if (texIndices) texIndices[vb + v] = ti;
            }
            for (let k = 0; k < p.idx.length; k++) indices[ic + k] = vb + p.idx[k]!;
            vb += n;
            ic += p.idx.length;
        }
        return { positions, indices, uvs, texIndices, ox: 0, oy: 0, oz: 0 };
    }

    // Bake a sampler result's world offset into the points → world-space seed + world AABB.
    function bakeFill(ox: number, oy: number, oz: number, result: ReturnType<typeof sampleMeshVolume>): SampledFill {
        const p = result.positions;
        for (let i = 0; i < p.length; i += 3) {
            p[i] = p[i]! + ox;
            p[i + 1] = p[i + 1]! + oy;
            p[i + 2] = p[i + 2]! + oz;
        }
        return {
            positions: p,
            count: result.count,
            radius: result.radius,
            boundsMin: [result.bounds.min[0] + ox, result.bounds.min[1] + oy, result.bounds.min[2] + oz],
            boundsMax: [result.bounds.max[0] + ox, result.bounds.max[1] + oy, result.bounds.max[2] + oz],
        };
    }

    function computeMaxR(hit: readonly [number, number, number], bMin: readonly [number, number, number], bMax: readonly [number, number, number]): number {
        let maxD = 0;
        for (const cx of [bMin[0], bMax[0]]) for (const cy of [bMin[1], bMax[1]]) for (const cz of [bMin[2], bMax[2]]) maxD = Math.max(maxD, Math.hypot(cx - hit[0], cy - hit[1], cz - hit[2]));
        return maxD + LIQUEFY_EDGE + 0.5;
    }

    // ── Wriggle (cartoon pain shake) ─────────────────────────────────────────
    // The MESH jitter starts the instant the foe is clicked (before the async volume-sample
    // finishes) for immediate feedback; the WATER jitter is added once the sim exists.
    function startWriggle(inst: Instance): void {
        inst.wriggleBase[0] = inst.root.position.x;
        inst.wriggleBase[1] = inst.root.position.y;
        inst.wriggleBase[2] = inst.root.position.z;
        inst.wriggleWaterBase = null;
        inst.wriggleWaterScratch = null;
        inst.wriggling = true;
    }

    function setWaterWriggle(inst: Instance, sampled: Float32Array, count: number): void {
        const base = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
            base[i * 4] = sampled[i * 3]!;
            base[i * 4 + 1] = sampled[i * 3 + 1]!;
            base[i * 4 + 2] = sampled[i * 3 + 2]!;
            base[i * 4 + 3] = 1;
        }
        inst.wriggleWaterBase = base;
        inst.wriggleWaterScratch = new Float32Array(count * 4);
    }

    function stopWriggle(inst: Instance): void {
        if (inst.wriggling) inst.root.position.set(inst.wriggleBase[0], inst.wriggleBase[1], inst.wriggleBase[2]);
        inst.wriggling = false;
        inst.wriggleWaterBase = null;
        inst.wriggleWaterScratch = null;
    }

    function applyWriggle(inst: Instance): void {
        const ox = (Math.random() * 2 - 1) * WRIGGLE_AMP;
        const oy = (Math.random() * 2 - 1) * WRIGGLE_AMP;
        const oz = (Math.random() * 2 - 1) * WRIGGLE_AMP;
        inst.root.position.set(inst.wriggleBase[0] + ox, inst.wriggleBase[1] + oy, inst.wriggleBase[2] + oz);
        const base = inst.wriggleWaterBase;
        const scratch = inst.wriggleWaterScratch;
        if (base && scratch && inst.sim) {
            for (let i = 0; i < base.length; i += 4) {
                scratch[i] = base[i]! + ox;
                scratch[i + 1] = base[i + 1]! + oy;
                scratch[i + 2] = base[i + 2]! + oz;
                scratch[i + 3] = 1;
            }
            device.queue.writeBuffer(inst.sim.positionBuffer, 0, scratch);
        }
    }

    // ── Impulse / fade force fields ──────────────────────────────────────────
    const impulseData = new Float32Array(8);
    function startImpulse(inst: Instance): void {
        if (!inst.sim) return;
        // Explode from INSIDE: centre on the volume, radius reaches past the whole blob so every
        // particle is in the ~uniform-outward core (see IMPULSE_WGSL).
        impulseData[0] = inst.volCenter[0];
        impulseData[1] = inst.volCenter[1];
        impulseData[2] = inst.volCenter[2];
        impulseData[3] = Math.max(inst.volRadius * 2.0, inst.radius * 8, 1);
        impulseData[4] = IMPULSE_RADIAL_BASE * impulseIntensity;
        impulseData[5] = IMPULSE_UP_BASE * impulseIntensity;
        impulseData[6] = 0;
        impulseData[7] = 0;
        device.queue.writeBuffer(inst.impulseBuffer, 0, impulseData);
        inst.sim.setForceField(inst.impulseSpec);
        inst.impulseRemaining = 0.35;
    }

    function beginFade(inst: Instance): void {
        // Fade the blob out in place by ramping its per-particle alpha 1→0 (handled in the
        // per-frame loop); the sim keeps settling under gravity meanwhile. No sink force.
        inst.phase = "fading";
        inst.fadeElapsed = 0;
        inst.impulseRemaining = 0;
        inst.sim?.setForceField(null);
    }

    const bumpUbo = (inst: Instance): void => {
        for (const m of inst.materials) m._uboVersion++;
    };
    const setVisible = (inst: Instance, v: boolean): void => {
        for (const m of inst.meshes) setMeshVisible(m, v);
    };

    function disposeInstanceSim(inst: Instance): void {
        inst.sim?.setForceField(null);
        inst.sim?.dispose();
        inst.sim = null;
        inst.colorBuffer?.destroy();
        inst.colorBuffer = null;
        inst.phase = "gone";
        inst.count = 0;
    }

    // ── Shot lifecycle ───────────────────────────────────────────────────────
    // Volume-sampling runs in a worker (see liquefactor-worker.ts) so the ~1 s dense sample
    // doesn't freeze the frame at the shot. requestSample() fires the job (foe stays solid,
    // marked `sampling`); applySample() builds the sim + starts the dissolve when it returns.
    let liveShots = 0;
    let sampleSeq = 0;
    const pendingSamples = new Map<number, { inst: Instance; hit: [number, number, number] }>();

    // A small pool of sampling workers so several foes can convert to particles IN PARALLEL: a
    // second shot while one conversion is in flight goes to a free worker instead of queueing behind
    // it. `pendingSamples` (keyed by id) routes each reply to the right foe regardless of worker.
    type WorkerMsg = { id: number; positions: Float32Array; uvs: Float32Array | null; texIndices: Uint32Array | null; count: number; radius: number; boundsMin: [number, number, number]; boundsMax: [number, number, number] };
    interface PoolWorker {
        worker: Worker;
        pending: number;
    }
    const workerPool: PoolWorker[] = [];
    const onSampleMessage = (ev: MessageEvent<WorkerMsg>): void => {
        const { id, positions, uvs, texIndices, count, radius, boundsMin, boundsMax } = ev.data;
        const entry = pendingSamples.get(id);
        pendingSamples.delete(id);
        if (!entry) return;
        if (count === 0) {
            entry.inst.sampling = false;
            stopWriggle(entry.inst);
            setStatus();
            return;
        }
        applySample(entry.inst, id, entry.hit, { positions, count, radius, boundsMin, boundsMax, uvs, texIndices });
    };
    try {
        if (typeof Worker !== "undefined") {
            const poolSize = Math.min(3, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
            for (let i = 0; i < poolSize; i++) {
                const worker = new Worker(new URL("./liquefactor-worker.ts", import.meta.url), { type: "module" });
                const pw: PoolWorker = { worker, pending: 0 };
                worker.addEventListener("message", (ev: MessageEvent<WorkerMsg>) => {
                    pw.pending = Math.max(0, pw.pending - 1);
                    onSampleMessage(ev);
                });
                worker.addEventListener("error", (e) => {
                    // eslint-disable-next-line no-console
                    console.warn("[liquefactor] sample worker error", e.message);
                    const idx = workerPool.indexOf(pw);
                    if (idx >= 0) workerPool.splice(idx, 1); // drop the faulty worker; sync fallback once the pool empties
                });
                workerPool.push(pw);
            }
        }
    } catch {
        workerPool.length = 0;
    }
    // Least-loaded worker (fewest in-flight jobs) so concurrent conversions spread across the pool.
    const pickWorker = (): PoolWorker | null => {
        let best: PoolWorker | null = null;
        for (const pw of workerPool) if (!best || pw.pending < best.pending) best = pw;
        return best;
    };

    function requestSample(inst: Instance, hit: readonly [number, number, number]): boolean {
        if (inst.phase !== "solid" || inst.sampling) return false;
        const id = ++sampleSeq;
        inst.sampling = true;
        inst.sampleId = id;
        const h: [number, number, number] = [hit[0], hit[1], hit[2]];
        pendingSamples.set(id, { inst, hit: h });
        startWriggle(inst); // pain shake begins immediately on click, before the async sample finishes
        setStatus();
        const geom = sampleGeometry(inst);
        if (!geom) {
            pendingSamples.delete(id);
            inst.sampling = false;
            stopWriggle(inst);
            setStatus();
            return false;
        }
        // Freeze a walking foe so the dissolve shows the exact pose we just sampled.
        if (inst.animation) pauseAnimation(inst.animation);
        const pw = pickWorker();
        if (pw) {
            const transfer: Transferable[] = [geom.positions.buffer, geom.indices.buffer];
            if (geom.uvs) transfer.push(geom.uvs.buffer);
            if (geom.texIndices) transfer.push(geom.texIndices.buffer);
            pw.pending++;
            pw.worker.postMessage({ id, positions: geom.positions, indices: geom.indices, uvs: geom.uvs, texIndices: geom.texIndices, radius: radiusValue, mode: modeValue, surfaceOnly: inst.surfaceOnly, ox: geom.ox, oy: geom.oy, oz: geom.oz }, transfer);
        } else {
            // No worker available — sample synchronously on the main thread (blocks). No per-particle
            // UVs are computed here, so colours fall back to the instance baseColor.
            pendingSamples.delete(id);
            const result = sampleMeshVolume({ positions: geom.positions, indices: geom.indices, radius: radiusValue, mode: modeValue });
            if (result.count > 0) {
                applySample(inst, id, h, { ...bakeFill(geom.ox, geom.oy, geom.oz, result), uvs: null, texIndices: null });
            } else {
                inst.sampling = false;
                stopWriggle(inst);
                setStatus();
            }
        }
        return true;
    }

    function applySample(inst: Instance, id: number, hit: readonly [number, number, number], fill: SampledFill): void {
        // Ignore stale results (a Restart or re-shape happened while sampling).
        if (inst.sampleId !== id || !inst.sampling || inst.phase !== "solid") {
            inst.sampling = false;
            return;
        }
        inst.sampling = false;
        buildInstanceSim(inst, fill.positions, fill.count, fill.radius, fill.boundsMin, fill.boundsMax);
        fillInstanceColor(inst, fill.uvs ?? null, fill.texIndices ?? null, fill.count); // per-particle mesh colours (per-mesh texture sample or baseColor)
        inst.liquefyState.hit = [hit[0], hit[1], hit[2]];
        inst.liquefyState.frontR = 0;
        inst.liquefyState.enabled = true;
        inst.maxR = computeMaxR(hit, fill.boundsMin, fill.boundsMax);
        // Explosion origin = centre of the sampled volume; reach = half the AABB diagonal. The sim
        // does NOT step during "dissolving", so these still match the particles at explosion time.
        inst.volCenter = [(fill.boundsMin[0] + fill.boundsMax[0]) / 2, (fill.boundsMin[1] + fill.boundsMax[1]) / 2, (fill.boundsMin[2] + fill.boundsMax[2]) / 2];
        inst.volRadius = Math.max(0.5 * Math.hypot(fill.boundsMax[0] - fill.boundsMin[0], fill.boundsMax[1] - fill.boundsMin[1], fill.boundsMax[2] - fill.boundsMin[2]), 0.5);
        inst.phase = "dissolving";
        setWaterWriggle(inst, fill.positions, fill.count); // add the water jitter; the mesh shake is already running
        bumpUbo(inst);
        liveShots++;
        invalidateFilteredSceneTasks();
        setStatus();
    }

    function finishShot(inst: Instance): void {
        stopWriggle(inst);
        setVisible(inst, false);
        inst.liquefyState.enabled = false;
        inst.liquefyState.frontR = inst.maxR;
        bumpUbo(inst);
        inst.phase = "fluid";
        inst.fluidElapsed = 0;
        startImpulse(inst);
        invalidateFilteredSceneTasks();
        setStatus();
    }

    function restart(): void {
        pendingSamples.clear();
        for (const inst of instances) {
            inst.sim?.setForceField(null);
            inst.sim?.dispose();
            inst.sim = null;
            inst.colorBuffer?.destroy();
            inst.colorBuffer = null;
            inst.phase = "solid";
            inst.sampling = false;
            inst.liquefyState.enabled = false;
            inst.liquefyState.frontR = 0;
            inst.impulseRemaining = 0;
            inst.fluidElapsed = 0;
            inst.fadeElapsed = 0;
            inst.wriggling = false;
            inst.wriggleWaterBase = null;
            inst.wriggleWaterScratch = null;
            inst.root.position.set(inst.homePos[0], inst.homePos[1], inst.homePos[2]);
            setVisible(inst, true);
            bumpUbo(inst);
            if (inst.animation) playAnimation(inst.animation); // resume the walk on the restored foe
        }
        liveShots = 0;
        virtualSim.count = 0;
        invalidateFilteredSceneTasks();
        setStatus();
    }

    // ── Control panel ────────────────────────────────────────────────────────
    const PANEL_STYLE =
        "position:fixed;top:12px;left:12px;z-index:10;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;" +
        "font-size:0.8rem;color:#dfe6ee;background:rgba(12,16,24,0.82);padding:12px 14px;border-radius:10px;" +
        "width:232px;max-height:92vh;overflow-y:auto;box-shadow:0 6px 22px rgba(0,0,0,0.4);";

    const title = document.createElement("div");
    title.textContent = "Liquefactor";
    title.style.cssText = "font-weight:700;font-size:0.95rem;margin-bottom:2px;";
    const subtitle = document.createElement("div");
    subtitle.textContent = "click a foe → its own fluid sim";
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
    };

    // PB-MPM material selector (only meaningful for PB-MPM; applied to the next shot).
    const materialSelect = document.createElement("select");
    styleSelect(materialSelect);
    for (const [label, value] of PBMPM_MATERIAL_LABELS) {
        const opt = document.createElement("option");
        opt.value = String(value);
        opt.textContent = label;
        materialSelect.append(opt);
    }
    materialSelect.value = String(currentMaterial);
    const materialRow = labelledRow("PB-MPM material", materialSelect);
    materialSelect.onchange = () => {
        currentMaterial = parseInt(materialSelect.value, 10) || 0;
        for (const inst of instances) inst.sim?.setMaterial?.(currentMaterial);
        refreshPhysicsParamVisibility();
    };
    const refreshMaterialRow = (): void => {
        materialRow.style.display = currentMethod === "PB-MPM" ? "" : "none";
    };

    // Particle radius slider.
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
    };
    const radiusLabelWrap = document.createElement("div");
    radiusLabelWrap.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    radiusLabelWrap.append(document.createTextNode("Particle radius"), radiusVal);
    const radiusRow = document.createElement("div");
    radiusRow.style.cssText = "margin-bottom:8px;";
    radiusRow.append(radiusLabelWrap, radiusInput);

    // Impulse intensity slider — scales the hand-off launch (0 = none, 1 = tuned default).
    const impulseInput = document.createElement("input");
    impulseInput.type = "range";
    impulseInput.id = "liq-impulse";
    impulseInput.min = "0";
    impulseInput.max = "3";
    impulseInput.step = "0.05";
    impulseInput.value = String(impulseIntensity);
    impulseInput.style.cssText = "width:100%;";
    const impulseVal = document.createElement("span");
    impulseVal.style.cssText = "color:#9fb4cc;float:right;";
    impulseVal.textContent = `${impulseIntensity.toFixed(2)}×`;
    impulseInput.oninput = () => {
        impulseIntensity = parseFloat(impulseInput.value);
        impulseVal.textContent = `${impulseIntensity.toFixed(2)}×`;
    };
    const impulseLabelWrap = document.createElement("div");
    impulseLabelWrap.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    impulseLabelWrap.append(document.createTextNode("Impulse intensity"), impulseVal);
    const impulseRow = document.createElement("div");
    impulseRow.style.cssText = "margin-bottom:8px;";
    impulseRow.append(impulseLabelWrap, impulseInput);

    // "Use mesh colours" — tint the water by the liquefied mesh's texture/vertex colours
    // (per-particle) instead of the uniform water colour.
    const meshColorInput = document.createElement("input");
    meshColorInput.type = "checkbox";
    meshColorInput.id = "liq-mesh-colors";
    meshColorInput.checked = useMeshColors;
    meshColorInput.style.cssText = "margin-right:6px;vertical-align:middle;";
    meshColorInput.onchange = () => {
        useMeshColors = meshColorInput.checked;
        surfaceTask.setUseParticleColor(useMeshColors);
    };
    const meshColorRow = document.createElement("label");
    meshColorRow.style.cssText = "display:block;margin-bottom:8px;color:#b6c4d6;cursor:pointer;";
    meshColorRow.append(meshColorInput, document.createTextNode("Use mesh colours"));

    const restartBtn = document.createElement("button");
    restartBtn.id = "liq-restart";
    restartBtn.textContent = "Restart (reset all foes)";
    restartBtn.style.cssText = "width:100%;padding:6px;margin-top:4px;border:0;border-radius:6px;cursor:pointer;background:#4a5568;color:#fff;font-weight:600;";
    restartBtn.onclick = () => restart();

    const status = document.createElement("div");
    status.style.cssText = "margin-top:8px;color:#8fa4bc;min-height:1.1em;";
    const partCount = document.createElement("div");
    partCount.style.cssText = "margin-top:2px;color:#8fa4bc;min-height:1.1em;";
    partCount.textContent = "0 particles";
    function setStatus(): void {
        const solid = instances.filter((i) => i.phase === "solid" && !i.sampling).length;
        const active = instances.filter((i) => i.sampling || i.phase === "dissolving" || i.phase === "fluid" || i.phase === "fading").length;
        status.textContent = `${solid} solid · ${active} liquefying · click a foe`;
        canvas.dataset.solid = String(solid);
        canvas.dataset.active = String(active);
    }

    function refreshPhysicsParamVisibility(): void {
        controls.setVisiblePhysicsParams(currentMethod === "PB-MPM" ? pbmpmParamKeysForMaterial(currentMaterial) : null);
    }

    function switchMethod(method: string): void {
        if (method === currentMethod) return;
        currentMethod = method;
        canvas.dataset.method = method;
        controls.setMethod(method);
        controls.rebuildPhysics(method);
        refreshMaterialRow();
        refreshPhysicsParamVisibility();
    }

    const controls = createFluidControlsPanel({
        hideParticles: true,
        hideMethod: false,
        hideContainerToggle: true,
        hideFoam: true,
        hidePhysics: false,
        hidePhysScale: true,
        hideDebug: true,
        hideGpuTiming: false,
        panelStyle: PANEL_STYLE,
        schemas: LIQ_SCHEMAS,
        methods: ["PBF", "MLS-MPM", "PB-MPM"],
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
        gpu: { stages: ["Simulation", "Surface"], supported: profiler !== null },
        on: {
            onMethod: (m) => switchMethod(m),
            onRenderMode: () => {}, // surface only in the multi-target demo
            onColor: (rgb) => surfaceTask.setFluidColor(rgb),
            onAbsorption: (v) => surfaceTask.setAbsorption(v),
            onParticleSize: (s) => surfaceTask.setSizeScale(s),
            onRefraction: (v) => surfaceTask.setRefractionStrength(v),
            onSpecular: (v) => surfaceTask.setSpecularPower(v),
            onDepthBlur: (size, threshold) => surfaceTask.setDepthBlur(size, threshold),
            onThicknessBlur: (v) => surfaceTask.setThicknessBlur(v),
            onHalf: (on) => surfaceTask.setHalfRender(on),
            onSurfaceFilter: (m) => surfaceTask.setSurfaceFilter(m),
            onNarrowRange: (delta, mu) => surfaceTask.setNarrowRange(delta, mu),
            onThicknessDownscale: (v) => surfaceTask.setThicknessDownscale(v),
            // Physics sliders apply LIVE to every running sim AND seed the next shot.
            onPhysicsParam: (key, value) => {
                physValues[currentMethod]![key] = value;
                for (const inst of instances) inst.sim?.setParam(key, value);
            },
            onReset: () => {
                const defaults = SCHEMA_DEFAULTS[currentMethod]!;
                physValues[currentMethod] = { ...defaults };
                for (const inst of instances) for (const [k, v] of Object.entries(defaults)) inst.sim?.setParam(k, v);
                controls.setPhysics(defaults);
                controls.rebuildPhysics(currentMethod);
            },
        },
    });

    controls.demoSlot.append(title, subtitle, labelledRow("Sampling mode", modeSelect), materialRow, radiusRow, impulseRow, meshColorRow, restartBtn, status, partCount);
    document.body.append(controls.root);
    if (controls.gpu) {
        // The demo's own panel is top-left, so pin the GPU pane top-right to avoid overlap.
        controls.gpu.panel.style.left = "auto";
        controls.gpu.panel.style.right = "12px";
        document.body.appendChild(controls.gpu.panel);
    }
    canvas.dataset.timing = profiler ? "on" : "unavailable";
    canvas.dataset.method = currentMethod;
    refreshMaterialRow();
    refreshPhysicsParamVisibility();
    setStatus();

    // ── Programmatic hooks for headless QA ───────────────────────────────────
    (window as unknown as { __liquefactor?: unknown }).__liquefactor = {
        getInstances: () => instances.map((i) => ({ key: i.key, phase: i.phase, count: i.count, sampling: i.sampling, tex: i.diffuseTexs.map((t) => [t.width, t.height]) })),
        usesWorker: () => workerPool.length > 0,
        workerCount: () => workerPool.length,
        getTotalParticles: () => virtualSim.count,
        getInstancePos: (key: string) => {
            const inst = instances.find((i) => i.key === key);
            return inst ? ([inst.root.position.x, inst.root.position.y, inst.root.position.z] as [number, number, number]) : null;
        },
        restart: () => restart(),
        setImpulse: (v: number) => {
            impulseIntensity = v;
            impulseInput.value = String(v);
            impulseVal.textContent = `${v.toFixed(2)}×`;
        },
        setUseMeshColors: (on: boolean) => {
            useMeshColors = on;
            meshColorInput.checked = on;
            surfaceTask.setUseParticleColor(on);
        },
        readColors: async (key: string, n = 12, stride = 1): Promise<number[] | null> => {
            const inst = instances.find((i) => i.key === key);
            if (!inst || !inst.colorBuffer || inst.count === 0) return null;
            const step = Math.max(1, Math.floor(stride));
            const count = Math.min(n, Math.floor(inst.count / step) || 1);
            const rb = device.createBuffer({ label: "liq-color-readback", size: inst.count * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const enc = device.createCommandEncoder();
            enc.copyBufferToBuffer(inst.colorBuffer, 0, rb, 0, inst.count * 16);
            device.queue.submit([enc.finish()]);
            await rb.mapAsync(GPUMapMode.READ);
            const all = new Float32Array(rb.getMappedRange().slice(0));
            rb.unmap();
            rb.destroy();
            const out: number[] = [];
            for (let i = 0; i < count; i++) {
                const p = i * step * 4;
                out.push(all[p]!, all[p + 1]!, all[p + 2]!, all[p + 3]!);
            }
            return out;
        },
        shootAt: (px: number, py: number) =>
            pickAsync(picker, px, py, { filter: (m) => instanceOf(m)?.phase === "solid" && !instanceOf(m)?.sampling }).then((info) => {
                const inst = info.pickedMesh ? instanceOf(info.pickedMesh) : undefined;
                const onEnemy = !!info.hit && !!info.pickedPoint && !!inst && inst.phase === "solid" && !inst.sampling;
                let dissolveStarted = false;
                if (onEnemy && info.pickedPoint) dissolveStarted = requestSample(inst!, info.pickedPoint);
                return { hit: info.hit, onEnemy, dissolveStarted, key: inst?.key ?? null };
            }),
    };

    // Click a foe to liquefy it from the hit point (left-drag still orbits via attachControl).
    canvas.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        const rect = canvas.getBoundingClientRect();
        const px = (ev.clientX - rect.left) * (canvas.width / rect.width);
        const py = (ev.clientY - rect.top) * (canvas.height / rect.height);
        void pickAsync(picker, px, py, { filter: (m) => instanceOf(m)?.phase === "solid" && !instanceOf(m)?.sampling }).then((info) => {
            if (!info.hit || !info.pickedPoint || !info.pickedMesh) return;
            const inst = instanceOf(info.pickedMesh);
            if (inst && inst.phase === "solid" && !inst.sampling) requestSample(inst, info.pickedPoint);
        });
    });

    // ── Per-frame loop ───────────────────────────────────────────────────────
    let lastRenderableCount = -1;
    let fpsAccumMs = 0;
    let fpsFrames = 0;
    onBeforeRender(scene, (deltaMs: number) => {
        // Open the whole-frame GPU-timing envelope BEFORE any pass is encoded this frame.
        if (profiler) {
            profiler.beginFrame();
            profiler.frameStart(engine._currentEncoder);
        }
        // The model is a dynamic (post-boot) add; its renderables materialize a few frames later.
        // Rebuild the filtered scene/foe draw lists whenever the renderable set changes so the
        // model appears in sceneColorRT (not just in the picker) once it drains in.
        if (scene._renderables.length !== lastRenderableCount) {
            lastRenderableCount = scene._renderables.length;
            invalidateFilteredSceneTasks();
        }
        updateAnimationManager(animManager, deltaMs); // advance the model's walk (skeleton pose)
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        const growDt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        for (const inst of instances) {
            if (inst.wriggling) applyWriggle(inst); // pain shake — active from click through the dissolve
            if (inst.phase === "dissolving") {
                inst.liquefyState.frontR = Math.min(inst.liquefyState.frontR + LIQUEFY_SPEED * growDt, inst.maxR);
                bumpUbo(inst);
                if (inst.liquefyState.frontR >= inst.maxR) finishShot(inst);
            } else if (inst.phase === "fluid" || inst.phase === "fading") {
                // Dispose BEFORE stepping so we never destroy a sim's buffers after having
                // already encoded a step into this frame's (not-yet-submitted) encoder.
                if (inst.phase === "fading") {
                    inst.fadeElapsed += dt;
                    if (inst.fadeElapsed >= FADE_DUR) {
                        disposeInstanceSim(inst);
                        liveShots = Math.max(0, liveShots - 1);
                        setStatus();
                        continue;
                    }
                }
                inst.sim?.step(engine._currentEncoder, dt);
                if (inst.impulseRemaining > 0) {
                    inst.impulseRemaining = Math.max(0, inst.impulseRemaining - dt);
                    if (inst.impulseRemaining === 0) inst.sim?.setForceField(null);
                }
                if (inst.phase === "fluid") {
                    inst.fluidElapsed += dt;
                    if (inst.fluidElapsed >= LIFETIME) {
                        beginFade(inst);
                        setStatus();
                    }
                }
            }
        }

        // Aggregate every live sim's render positions into the combined buffer (after stepping),
        // and fill the matching per-particle alpha (1 for running blobs, ramped for fading ones).
        // When "Use mesh colours" is on, also aggregate each sim's per-particle colour buffer.
        let off = 0;
        for (const inst of instances) {
            if (inst.sim && inst.phase !== "solid" && inst.phase !== "gone") {
                const n = inst.sim.count;
                if (off + n <= MAX_TOTAL) {
                    engine._currentEncoder.copyBufferToBuffer(inst.sim.positionBuffer, 0, combinedPos, off * 16, n * 16);
                    if (useMeshColors && inst.colorBuffer) {
                        engine._currentEncoder.copyBufferToBuffer(inst.colorBuffer, 0, combinedColor, off * 16, n * 16);
                    }
                    const a = inst.phase === "fading" ? Math.max(0, 1 - inst.fadeElapsed / FADE_DUR) : 1;
                    alphaScratch.fill(a, off, off + n);
                    off += n;
                }
            }
        }
        if (off > 0) {
            // Keep the whole active range in sync so a slot reused after a fade isn't left dim.
            device.queue.writeBuffer(combinedAlpha, 0, alphaScratch, 0, off);
            virtualSim.count = off;
        } else {
            virtualSim.count = 0;
        }
        canvas.dataset.particleCount = String(off);
        partCount.textContent = `${off.toLocaleString()} particles`;

        // GPU timing + memory read-outs on a ~2 Hz cadence (same as the FPS counter).
        fpsAccumMs += deltaMs;
        fpsFrames++;
        if (fpsAccumMs >= 500) {
            const gpu = controls.gpu;
            if (gpu) {
                gpu.fpsLabel.textContent = `${Math.round((fpsFrames * 1000) / fpsAccumMs)}`;
                gpu.refreshTiming(profiler ? profiler.results() : null);
                let simBytes = 0;
                for (const inst of instances) if (inst.sim) simBytes += inst.sim.gpuBytes;
                gpu.refreshMemory(simBytes, engine.canvas.width, engine.canvas.height);
            }
            fpsAccumMs = 0;
            fpsFrames = 0;
        }
    });

    await envReady;
    // Load all glTF foes BEFORE the scene build so they materialize reliably (a dynamic post-boot
    // add leaves the clip-plugin UBO uninitialised). The 39 MB haunted house dominates this wait.
    await Promise.all(MODEL_FOES.map((cfg) => loadModelInstance(cfg)));
    await registerScene(scene);
    await startEngine(engine);

    canvas.dataset.ready = "true";
}

main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error(err);
    const c = document.getElementById("renderCanvas") as HTMLCanvasElement | null;
    if (c) c.dataset.error = String(err);
});
