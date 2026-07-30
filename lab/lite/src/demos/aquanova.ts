// Aquanova demo — first-person sci-fi shooter tech demo built on Lite.
//
// The player traverses a confined spaceship wielding the "Liquefactor": melt flying alien foes
// and ship props into GPU fluid. This slice LOADS THE SHIP: a chunked modular interior authored
// in Blender from the CC0 Quaternius "Modular SciFi MegaKit" and exported to a single glTF
// (ship.glb) with a companion ship_manifest.json describing chunks, portals, doors and spawns
// (see lab/public/aquanova/ASSET-LICENSES.md and SciFiShip/README.md).
//
// What this slice does:
//   • loads ship.glb and lights it with an HDR environment (IBL);
//   • builds one static trimesh collider from the whole ship so the player collides with every
//     wall, floor and prop — no hand-authored boxes;
//   • spawns a Havok first-person character-controller capsule at the manifest player spawn, the
//     camera riding at eye height (WASD/arrows walk in the horizontal plane, mouse-drag looks).
//
// The ship runs back → front along glTF +X (storage → corridor → junction → cargo bay). Lite's
// glTF loader mirrors handedness on the __root__ (scale x = -1), so glTF (x,y,z) renders at Lite
// (-x, y, z); every manifest coordinate is converted through `toLite()` before use.
//
// Per-chunk portal culling, animated sliding doors, the weapon pickup, and the foes build on top.

import HavokPhysics from "@babylonjs/havok";
import {
    addTask,
    addToScene,
    createBox,
    createEngine,
    createFreeCamera,
    createGpuPicker,
    createHavokWorld,
    createPhysicsBody,
    createPhysicsCharacterController,
    createPhysicsShape,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createStandardMaterial,
    createTransformNode,
    CharacterSupportedState,
    enableMaterialPlugins,
    getMeshGeometry,
    getPhysicsBodyAngularVelocity,
    getPhysicsBodyLinearVelocity,
    getProjectionMatrix,
    getViewMatrix,
    getViewProjectionMatrix,
    isPbrMaterial,
    loadGltf,
    loadHdrEnvironment,
    onBeforeRender,
    onPhysicsAfterStep,
    physicsRaycast,
    pickAsync,
    PhysicsMotionType,
    PhysicsShapeType,
    registerScene,
    removeFromScene,
    removePhysicsBody,
    setMeshVisible,
    setParent,
    setPhysicsBodyShape,
    setPhysicsTimestepMs,
    startEngine,
} from "babylon-lite";
import type { Material, Mesh, PhysicsBody, PhysicsWorld, SceneNode } from "babylon-lite";
import { createFloatingBodySystem } from "babylon-lite/fluid/floating-body.js";
import { generateMeshSdf } from "babylon-lite/fluid/volume-sampling/index.js";
import { fillMeshParticles } from "./particle-fill.js";
import { createMlsMpmSim } from "babylon-lite/fluid/mls-mpm-sim.js";
import { createPbMpmSim } from "babylon-lite/fluid/pbmpm-sim.js";
import { createFluidSurfaceTask } from "babylon-lite/fluid/fluid-surface-render.js";
import type { FluidSim, ForceFieldSpec, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { createLiquefyPlugin } from "./liquefy-plugin.js";
import { buildLitParticleColors } from "./particle-lit-colors.js";
import type { LitColorScene } from "./particle-lit-colors.js";
import {
    DEFAULT_SHIP_IBL_STRENGTH,
    excludedSdfMeshNames,
    findEntityWithBehavior,
    isDynamicBehavior,
    isLiquefiableBehavior,
    linkedMeshNames,
    PLAYER_START_BEHAVIOR,
    resolveBehavior,
    resolveExposure,
    resolveToneMapping,
    WEAPON_START_BEHAVIOR,
} from "./ship-manifest.js";
import type { ShipBehavior, ShipBehaviorLibrary, ShipEntities, ShipEnvironment } from "./ship-manifest.js";
import type { LiquefyState } from "./liquefy-plugin.js";

const SHIP_URL = "/aquanova/ship.glb";
const MANIFEST_URL = "/aquanova/ship_manifest.json";
// Image-based lighting: the ship is lit by this HDRI (Poly Haven "bank_vault" 2k, CC0) plus its own
// emissive ceiling fixtures — the glTF export carries no punctual lights.
const ENV_URL = "/aquanova/bank_vault_2k.hdr";

type Vec3 = readonly [number, number, number];

// glTF (right-handed) → Lite scene (left-handed): the loader's __root__ negates X.
function toLite([x, y, z]: Vec3): [number, number, number] {
    return [-x, y, z];
}


interface Aabb {
    min: Vec3;
    max: Vec3;
}
// `behaviors` is a library of named behaviour definitions; `entities` assigns them to MESH NAMES,
// optionally overriding parameters per entity (see resolveBehavior). Mesh names are shared across
// rooms, so an entity applies to every mesh carrying that name. `liquefiable` implies `dynamic`; a
// behaviour without `fluidSim` inherits the manifest's global list.
interface ShipChunk {
    id: string;
    node?: string;
    aabb: Aabb;
}
interface ShipManifest {
    fluidSim?: string[]; // fluid-simulation setting files (in aquanova/fluidSim/) a liquefied mesh may use
    behaviors?: ShipBehaviorLibrary; // behaviour name → definition
    entities?: ShipEntities; // mesh name → assigned behaviours
    chunks: ShipChunk[];
    portals: { centre: Vec3; normal?: Vec3; corners?: Vec3[] }[];
    environment?: ShipEnvironment;
    spawns?: { player?: Vec3; weapon?: Vec3; startChunk?: string };
}

async function fetchManifest(): Promise<ShipManifest | undefined> {
    try {
        const res = await fetch(MANIFEST_URL);
        if (!res.ok) return undefined;
        return (await res.json()) as ShipManifest;
    } catch {
        return undefined;
    }
}

// A parsed fluid-simulation setting (a liquefactor/fluid export JSON): the solver method plus the raw,
// method-specific physics record and, for PB-MPM, the material enum. Consumed by buildFluidSim below.
interface FluidSimSetting {
    method: string; // "MLS-MPM" | "PB-MPM"
    physics: Record<string, number>;
    material?: number;
    useMeshColors?: boolean; // demoParams.useMeshColors — tint the water by the liquefied mesh's texture
    particleRadius?: number; // demoParams.particleRadius — volume-sampling spacing (drives the particle count)
}
async function fetchFluidSetting(name: string): Promise<FluidSimSetting | undefined> {
    try {
        const res = await fetch(`/aquanova/fluidSim/${name}.json`);
        if (!res.ok) return undefined;
        const j = (await res.json()) as {
            meta?: { method?: string };
            physics?: Record<string, number>;
            material?: number;
            demoParams?: { useMeshColors?: number; particleRadius?: number };
        };
        if (!j.physics || !j.meta?.method) return undefined;
        return {
            method: j.meta.method,
            physics: j.physics,
            material: j.material,
            useMeshColors: !!j.demoParams?.useMeshColors,
            particleRadius: j.demoParams?.particleRadius,
        };
    } catch {
        return undefined;
    }
}

// Ship-shell collider geometry (metres). Rooms are uniform 0..5 m tall boxes on a 4 m grid.
const WALL_T = 0.4;
const FLOOR_Y = 0;
const CEIL_Y = 5;
const WALL_INSET = 0.3; // pull perimeter walls to the inner wall face so the capsule can't clip them

// Collide the player against clean per-chunk box shells (floor + ceiling + 4 walls) built from the
// manifest, rather than a trimesh of the raw ship. The 568 overlapping kit modules make a trimesh
// riddled with coplanar/degenerate triangles that pins the character controller; axis-aligned boxes
// give smooth motion and fit the chunked, portal-cell layout exactly. Each portal doorway is cut
// out of the wall it sits on (see addXWall) so the player can walk through once its door is liquefied.
/** One static collider box, in LITE space, tagged with the chunk that produced it (debug overlay). */
interface ColliderBox {
    chunk: string;
    c: [number, number, number];
    e: [number, number, number];
}

function buildShipColliders(world: PhysicsWorld, manifest: ShipManifest, out: ColliderBox[] = []): void {
    let n = 0;
    let curChunk = "";
    // addBox takes a glTF-space centre + full extents; X is negated into Lite scene space.
    // Zero/negative-extent boxes (a doorway that meets a wall edge) are skipped.
    const addBox = (gx: number, gy: number, gz: number, ex: number, ey: number, ez: number): void => {
        if (ex <= 1e-3 || ey <= 1e-3 || ez <= 1e-3) return;
        const node = createTransformNode(`shipCol_${n++}`, -gx, gy, gz);
        const shape = createPhysicsShape(world, { type: PhysicsShapeType.BOX, parameters: { extents: { x: ex, y: ey, z: ez } } });
        setPhysicsBodyShape(world, createPhysicsBody(world, node, PhysicsMotionType.STATIC), shape);
        out.push({ chunk: curChunk, c: [-gx, gy, gz], e: [ex, ey, ez] });
    };
    const midY = (FLOOR_Y + CEIL_Y) / 2;
    const roomH = CEIL_Y - FLOOR_Y;

    // Portal openings on X-facing walls (every ship portal faces ±X). Each doorway is cut OUT of the
    // wall it lies on — into two side panels + a lintel — so the player can pass once the door leaves
    // are liquefied; a sealed wall would otherwise still block the corridor. glTF-space (z/y shared).
    interface XOpening { x: number; z0: number; z1: number; y0: number; y1: number; }
    const xOpenings: XOpening[] = [];
    for (const p of manifest.portals ?? []) {
        if (!p.corners || Math.abs(p.normal?.[0] ?? 0) < 0.5) continue;
        const zs = p.corners.map((c) => c[2]!);
        const ys = p.corners.map((c) => c[1]!);
        xOpenings.push({ x: p.centre[0], z0: Math.min(...zs), z1: Math.max(...zs), y0: Math.min(...ys), y1: Math.max(...ys) });
    }
    const openingOnX = (boundaryX: number, z0: number, z1: number): XOpening | null =>
        xOpenings.find((o) => Math.abs(o.x - boundaryX) < 0.6 && o.z0 >= z0 - 0.5 && o.z1 <= z1 + 0.5) ?? null;

    // One X-facing wall (Lite x = −wallX) across the chunk's z-span, with an optional doorway gap.
    const addXWall = (wallX: number, z0: number, z1: number, o: XOpening | null): void => {
        if (!o) { addBox(wallX, midY, (z0 + z1) / 2, WALL_T, roomH, z1 - z0); return; }
        addBox(wallX, midY, (z0 + o.z0) / 2, WALL_T, roomH, o.z0 - z0); // side panel (−Z of doorway)
        addBox(wallX, midY, (o.z1 + z1) / 2, WALL_T, roomH, z1 - o.z1); // side panel (+Z of doorway)
        addBox(wallX, (o.y1 + CEIL_Y) / 2, (o.z0 + o.z1) / 2, WALL_T, CEIL_Y - o.y1, o.z1 - o.z0); // lintel above
        addBox(wallX, (FLOOR_Y + o.y0) / 2, (o.z0 + o.z1) / 2, WALL_T, o.y0 - FLOOR_Y, o.z1 - o.z0); // sill below (usually none)
    };

    for (const { id, aabb } of manifest.chunks) {
        curChunk = id;
        const [x0, , z0] = aabb.min;
        const [x1, , z1] = aabb.max;
        const cx = (x0 + x1) / 2;
        const cz = (z0 + z1) / 2;
        const w = x1 - x0;
        const d = z1 - z0;
        addBox(cx, FLOOR_Y - WALL_T / 2, cz, w, WALL_T, d); // floor (top at y = 0)
        addBox(cx, CEIL_Y + WALL_T / 2, cz, w, WALL_T, d); // ceiling (bottom at y = 5)
        addBox(cx, midY, z0 + WALL_INSET, w, roomH, WALL_T); // −Z wall
        addBox(cx, midY, z1 - WALL_INSET, w, roomH, WALL_T); // +Z wall
        addXWall(x0 + WALL_INSET, z0, z1, openingOnX(x0, z0, z1)); // −X wall
        addXWall(x1 - WALL_INSET, z0, z1, openingOnX(x1, z0, z1)); // +X wall
    }
}

async function main(): Promise<void> {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const engine = await createEngine(canvas, { msaaSamples: 1 });
    const scene = createSceneContext(engine, { defaultRenderTask: false });

    const cam = createFreeCamera({ x: 0, y: 1.5, z: 0 }, { x: -1, y: 1.5, z: 0 });
    cam.nearPlane = 0.1;
    cam.farPlane = 400;
    scene.camera = cam;

    // Fetch the ship manifest up front — it drives the IBL strength, tone mapping, colliders, and spawn.
    const manifest = await fetchManifest();

    // Preload the fluid-simulation settings the manifest lists (parsed once). At liquefy time a mesh
    // uses its pinned override if it has one, else a random pick from this global list.
    const fluidSettingNames = manifest?.fluidSim ?? [];
    const fluidSettings = new Map<string, FluidSimSetting>();
    await Promise.all(
        fluidSettingNames.map(async (name) => {
            const s = await fetchFluidSetting(name);
            if (s) fluidSettings.set(name, s);
        })
    );

    // ── Load the ship + image-based lighting ──────────────────────────────────────────────
    // Lighting is fully image-based (the export carries no punctual lights): the bank_vault HDRI
    // supplies ambient + specular, and the trim/fixtures are HDR-emissive geometry (M_Light etc.,
    // via KHR_materials_emissive_strength). entities[0] is the __root__ (RH→LH mirror baked in)
    // with all geometry beneath it. Note: Lite applies IBL per-surface with no occlusion, so the
    // HDRI lights even these sealed compartments (Blender needs baked GI for the equivalent look).
    const ship = await loadGltf(engine, SHIP_URL);
    const shipRoot = ship.entities[0] as SceneNode;
    addToScene(scene, shipRoot);

    // Build the HDR IBL (prefiltered specular cubemap + SH irradiance + BRDF LUT) from the .hdr.
    const env = await loadHdrEnvironment(scene, ENV_URL, { skipGround: true });

    // Honour the manifest's IBL strength: environmentIntensity scales only the IBL contribution
    // (diffuse SH + specular) per material, leaving the emissive fixtures at full brightness.
    const iblStrength = manifest?.environment?.strength ?? DEFAULT_SHIP_IBL_STRENGTH;
    const seenMats = new Set<object>();
    const applyIblStrength = (node: SceneNode): void => {
        const mat = (node as { material?: { environmentIntensity?: number } }).material;
        if (mat && !seenMats.has(mat)) {
            seenMats.add(mat);
            mat.environmentIntensity = iblStrength;
        }
        for (const child of node.children) applyIblStrength(child as SceneNode);
    };
    applyIblStrength(shipRoot);

    // ── Classify ship meshes: collect all, plus global dynamic / liquefiable and per-room static sets ─
    // Entities are keyed by the glTF NODE name (what the Babylon sandbox shows, e.g. "Door_D00_L"),
    // NOT the glTF mesh name Lite puts on `mesh.name`. The loader parents each node's primitives under
    // a TransformNode carrying the node name, so the walk below records that name per mesh and groups
    // one node's primitives together.
    //
    // `_primitiveN` WRAPPERS: an asset round-tripped through Babylon carries its loader's split
    // convention baked in as real nodes — a mesh-less parent "player" holding a child node
    // "player_primitive0" that owns the geometry (195 of 292 nodes in the current ship). Those
    // wrappers are collapsed into their parent so the author's node name is what the manifest keys,
    // and so a node's primitives regroup into ONE entity instead of N single-primitive ones. The test
    // is exact — the child's name must be the parent's plus `_primitive<digits>` — so a genuinely
    // authored node is never absorbed.
    //
    // `chunkStaticMeshes` (everything NOT dynamic/liquefiable, grouped by the chunk whose AABB contains
    // the mesh centre) feeds the per-room SDF bake; manifest AABBs are glTF-space, so the Lite-space
    // mesh centre is converted back.
    const PRIMITIVE_WRAPPER = /_primitive\d+$/;
    const allShipMeshes: Mesh[] = [];
    const nodeNameOfMesh = new Map<Mesh, string>(); // mesh → owning glTF node name
    const nodePrimitives = new Map<Mesh, Mesh[]>(); // mesh → every primitive of the SAME node (incl. itself)
    const meshesByNodeName = new Map<string, Mesh[]>(); // node name → all its primitives, ship-wide
    const primitivesByOwner = new Map<SceneNode, Mesh[]>();
    const collectMeshes = (node: SceneNode, owner: SceneNode): void => {
        for (const child of node.children) {
            const c = child as SceneNode;
            if ((c as Mesh).material) {
                const m = c as Mesh;
                allShipMeshes.push(m);
                nodeNameOfMesh.set(m, owner.name);
                const group = primitivesByOwner.get(owner);
                if (group) group.push(m);
                else primitivesByOwner.set(owner, [m]);
                const byName = meshesByNodeName.get(owner.name);
                if (byName) byName.push(m);
                else meshesByNodeName.set(owner.name, [m]);
                continue;
            }
            // A wrapper keeps the parent as owner; any other node owns its own subtree.
            const wrapper = PRIMITIVE_WRAPPER.test(c.name) && c.name.replace(PRIMITIVE_WRAPPER, "") === node.name;
            collectMeshes(c, wrapper ? owner : c);
        }
    };
    collectMeshes(shipRoot, shipRoot);
    for (const group of primitivesByOwner.values()) for (const m of group) nodePrimitives.set(m, group);

    // Portal meshes ("Portal_*") are doorway markers the exporter emits for culling / door pairing —
    // NOT real geometry. They render as a visible pane spanning the doorway (seen from the corridor)
    // AND sit coplanar with the door leaves, so the weapon pick can hit the portal instead of the door
    // behind it (the door then won't liquefy). Hide them, make them non-pickable, and skip them in the
    // mesh classification below so they never enter the SDF bake either.
    const isPortalMesh = (m: Mesh): boolean => m.name.startsWith("Portal_");
    for (const m of allShipMeshes) {
        if (isPortalMesh(m)) {
            setMeshVisible(m, false);
            (m as { pickable?: boolean }).pickable = false;
        }
    }

    // ── Placement markers (player / weapon start) ────────────────────────────────────────
    // An entity carrying a `*_startpos` behaviour is a PLACEMENT MARKER, not scenery: something
    // spawns at its node and the marker itself is DISABLED — hidden, non-pickable, and skipped by
    // every classification pass below so it never enters physics, the SDF bake or the target sets.
    // Its `direction` parameter is the glTF-space vector the spawned thing initially faces. The
    // library definitions are empty, so a marker is matched by NAME on the entity's reference.
    const markerMeshes = new Set<Mesh>();
    interface Marker {
        entity: string;
        min: number[] | null;
        max: number[] | null;
        direction?: number[];
    }
    const resolveMarker = (behaviorName: string): Marker | null => {
        const found = findEntityWithBehavior(manifest?.entities, behaviorName);
        if (!found) return null;
        const meshes = meshesByNodeName.get(found.name) ?? [];
        if (!meshes.length) {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] ${behaviorName} entity "${found.name}" matches no ship node — ignoring it`);
            return null;
        }
        // World AABB — already Lite space, since it comes from the loaded ship (mirror applied).
        let min: number[] | null = null;
        let max: number[] | null = null;
        for (const m of meshes) {
            markerMeshes.add(m);
            setMeshVisible(m, false);
            (m as { pickable?: boolean }).pickable = false;
            const a = m.boundMin;
            const b = m.boundMax;
            if (!a || !b) continue;
            min = min ? [Math.min(min[0]!, a[0]!), Math.min(min[1]!, a[1]!), Math.min(min[2]!, a[2]!)] : [a[0]!, a[1]!, a[2]!];
            max = max ? [Math.max(max[0]!, b[0]!), Math.max(max[1]!, b[1]!), Math.max(max[2]!, b[2]!)] : [b[0]!, b[1]!, b[2]!];
        }
        return { entity: found.name, min, max, direction: found.ref.direction };
    };
    const playerMarker = resolveMarker(PLAYER_START_BEHAVIOR);
    const weaponMarker = resolveMarker(WEAPON_START_BEHAVIOR);
    // Disabled marker geometry is treated exactly like a portal: excluded everywhere below.
    const isDisabledMesh = (m: Mesh): boolean => isPortalMesh(m) || markerMeshes.has(m);

    // Weapon pickup spot: the marker's centre, else the legacy glTF-space spawn. Exposed for the
    // pickup prop (and QA) rather than being read from the removed `spawns.weapon`.
    const weaponSpawn: Vec3 =
        weaponMarker?.min && weaponMarker.max
            ? [(weaponMarker.min[0]! + weaponMarker.max[0]!) / 2, weaponMarker.max[1]!, (weaponMarker.min[2]! + weaponMarker.max[2]!) / 2]
            : toLite(manifest?.spawns?.weapon ?? [7, 0.95, 2.8]);
    canvas.dataset.weaponSpawn = weaponSpawn.map((v) => v.toFixed(3)).join(",");

    // Disable KHR_materials_transmission on the ship: the fluid surface composites into the swapchain
    // AFTER the scene renders, but transmission retargets the scene task to an offscreen HDR buffer and
    // appends a final tonemap pass that would overwrite (hide) the fluid. Dropping the refraction
    // sub-feature keeps the scene on our sceneColorRT (like the Liquefactor demo) so the water shows;
    // the glass then renders as a plain surface instead of refracting.
    for (const m of allShipMeshes) {
        const mat = m.material;
        if (mat && isPbrMaterial(mat)) {
            const ss = (mat as unknown as { subsurface?: { refraction?: unknown } }).subsurface;
            if (ss?.refraction) ss.refraction = undefined;
        }
    }

    const meshCentreGltf = (m: Mesh): [number, number, number] | null => {
        const mn = m.boundMin;
        const mx = m.boundMax;
        if (!mn || !mx) return null;
        return [-(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2]; // Lite → glTF (negate X)
    };
    const inChunk = (p: [number, number, number], a: Aabb): boolean =>
        p[0] >= a.min[0] && p[0] <= a.max[0] && p[2] >= a.min[2] && p[2] <= a.max[2] && p[1] >= a.min[1] - 0.5 && p[1] <= a.max[1] + 0.5;
    // Which chunk (room) contains a mesh, by its centre (same glTF-space test as the classification below).
    const roomOfMesh = (m: Mesh): string => {
        const p = meshCentreGltf(m);
        if (p) for (const c of manifest?.chunks ?? []) if (inChunk(p, c.aabb)) return c.id;
        return "unknown";
    };

    const dynamicMeshes = new Set<Mesh>();
    const liquefiableMeshes = new Set<Mesh>(); // SHOOTABLE targets (their own behaviour says liquefiable)
    const dissolvableMeshes = new Set<Mesh>(); // liquefiable + everything reachable through `linked`
    const behaviorByMesh = new Map<Mesh, ShipBehavior>(); // mesh → its effective manifest behaviour
    const chunkStaticMeshes = new Map<string, Mesh[]>();

    // Node names that can dissolve: every liquefiable entity plus the transitive closure of its
    // `linked` names. A linked node need NOT be liquefiable itself (a door's frame melts with the
    // door but can't be shot on its own), yet it still needs the clip plugin, a rigid body and
    // exclusion from the static SDF — so it is classified exactly like a liquefiable node, minus
    // being a valid pick target.
    const behaviorOfName = (name: string): ShipBehavior | undefined => resolveBehavior(manifest?.behaviors, manifest?.entities, name);
    const dissolvableNames = new Set<string>();
    const nameStack = Object.keys(manifest?.entities ?? {}).filter((n) => isLiquefiableBehavior(behaviorOfName(n)));
    while (nameStack.length) {
        const n = nameStack.pop()!;
        if (dissolvableNames.has(n)) continue;
        dissolvableNames.add(n);
        for (const l of linkedMeshNames(behaviorOfName(n))) nameStack.push(l);
    }

    // Pass 1 — classify by NODE name. `entities` assigns behaviours once for the whole ship, so a
    // named node qualifies wherever it sits (including nodes outside every chunk AABB), and ALL of a
    // node's primitives are classified together. Anything that dissolves is dynamic too — it must be
    // physically present before it melts.
    for (const m of allShipMeshes) {
        if (isDisabledMesh(m)) continue; // portals + placement markers: not collidable/SDF/liquefiable geometry
        const nodeName = nodeNameOfMesh.get(m) ?? m.name;
        const b = behaviorOfName(nodeName);
        if (b) behaviorByMesh.set(m, b);
        const dissolvable = dissolvableNames.has(nodeName);
        if (dissolvable) dissolvableMeshes.add(m);
        if (dissolvable || isDynamicBehavior(b)) dynamicMeshes.add(m);
        if (isLiquefiableBehavior(b)) liquefiableMeshes.add(m);
    }

    // Pass 2 — per-room STATIC geometry (everything that neither moves nor melts) for the SDF bake.
    for (const c of manifest?.chunks ?? []) {
        const staticList: Mesh[] = [];
        for (const m of allShipMeshes) {
            if (isDisabledMesh(m)) continue;
            if (dynamicMeshes.has(m) || dissolvableMeshes.has(m)) continue;
            const p = meshCentreGltf(m);
            if (!p || !inChunk(p, c.aabb)) continue;
            staticList.push(m);
        }
        chunkStaticMeshes.set(c.id, staticList);
    }
    canvas.dataset.dynamicCount = String(dynamicMeshes.size);
    canvas.dataset.liquefiableCount = String(liquefiableMeshes.size);

    // Match the Blender view transform the ship was authored against, straight from the manifest:
    // tone mapping (Khronos PBR Neutral keeps the strength-boosted emissive trim from clipping to
    // white and losing its colour) and exposure, converted from Blender stops to a linear multiplier.
    // Set before registerScene so the first PBR build + the deferred skybox snapshot pick it up.
    const tone = resolveToneMapping(manifest?.environment?.toneMapping);
    scene.imageProcessing.toneMappingEnabled = tone !== null;
    if (tone) scene.imageProcessing.toneMapping = tone;
    scene.imageProcessing.exposure = resolveExposure(manifest?.environment?.exposure);
    scene.imageProcessing.contrast = 1.05;

    // ── Havok physics: clean per-chunk box-shell colliders (see buildShipColliders) ───────
    const hknp = await HavokPhysics({ locateFile: () => "/HavokPhysics.wasm" });
    const world = createHavokWorld(scene, hknp, { x: 0, y: -9.8, z: 0 });
    // FIXED simulation step. With the default variable step the world advances by the real frame
    // delta, so how far the player capsule sinks before the solver stops it depends on how slow the
    // first frames after load happen to be — and it stays wherever it came to rest. Measured on the
    // same build: the capsule settles at y 0.909 at full speed but 1.036 under 3x CPU throttling, i.e.
    // a 13 cm eye-height difference between machines. Babylon.js fixes its step at 1/60 for the same
    // reason. This also keeps the dynamic props' rest poses reproducible.
    setPhysicsTimestepMs(world, 1000 / 60);

    // Static collider boxes, recorded so the B overlay can draw exactly what the player collides with.
    const colliderBoxes: ColliderBox[] = [];
    // The raw ship geometry is one glTF hierarchy of 568 overlapping modules; a trimesh collider
    // built from it pins the character controller on countless coplanar/overlapping triangles.
    // Instead we collide against clean per-chunk box shells from the manifest — smooth for the
    // capsule, and a natural fit since each chunk is already an axis-aligned room.
    if (manifest) {
        buildShipColliders(world, manifest, colliderBoxes);
    } else {
        // Fallback floor so the player at least stands if the manifest failed to load.
        const node = createTransformNode("shipFloor", -30, -0.2, 0);
        const shape = createPhysicsShape(world, { type: PhysicsShapeType.BOX, parameters: { extents: { x: 140, y: 0.4, z: 24 } } });
        setPhysicsBodyShape(world, createPhysicsBody(world, node, PhysicsMotionType.STATIC), shape);
    }

    // ── First-person player at the manifest spawn ─────────────────────────────────────────
    const CAP_H = 1.8;
    const CAP_R = 0.4;
    const EYE = 0.62; // camera offset above the capsule centre → ~1.5 m eye level
    // Marker present → stand on top of it; otherwise fall back to the legacy glTF-space spawn.
    const fallback = toLite(manifest?.spawns?.player ?? [-0.5, 0, 0]);
    const pMin = playerMarker?.min;
    const pMax = playerMarker?.max;
    const sx = pMin && pMax ? (pMin[0]! + pMax[0]!) / 2 : fallback[0];
    const sz = pMin && pMax ? (pMin[2]! + pMax[2]!) / 2 : fallback[2];
    const sy = pMax ? pMax[1]! + CAP_H / 2 + 0.1 : CAP_H / 2 + 0.1;
    const character = createPhysicsCharacterController(world, { x: sx, y: sy, z: sz }, { capsuleHeight: CAP_H, capsuleRadius: CAP_R });
    character.characterStrength = 0; // player collides with dynamic bodies but cannot push them

    // ── Dynamic meshes + per-room SDFs → one shared FloatingBodySystem ────────────────────
    // All fluid boundaries live in ONE FloatingBodySystem (SceneSdfSpec has a single sdfGrid buffer;
    // the system's `bodiesSdf` unions every body, each with its own dims/origin in the header):
    //   • per-room STATIC SDFs (baked from room static meshes) — added in the room-SDF pass (C);
    //   • per dynamic mesh: a baked SDF driven by a Havok DYNAMIC box body. The player collides but
    //     can't push it (characterStrength 0); gravity/fluid move it; each step we push its Havok pose
    //     to the floating body so the SDF boundary tracks the visible mesh (reparented under a display
    //     root so the system can pose it, glTF __root__ mirror preserved via displayScale).
    const MIRROR: [number, number, number] = [-1, 1, 1];
    const ROOM_CELL = 0.18; // room SDF cell size (m)
    const DYN_CELL = 0.06; // dynamic-mesh SDF cell size (m)
    const SDF_PAD = 2;

    type BakedSdf = ReturnType<typeof generateMeshSdf>;
    // Bake meshes' WORLD-space geometry into a signed-distance grid centred at the combined AABB centre.
    const bakeWorldSdf = (meshes: Mesh[], cell: number): { grid: BakedSdf; centre: [number, number, number]; half: [number, number, number] } | null => {
        const geos: Array<{ pos: Float32Array; idx: Uint32Array; m: Mesh }> = [];
        let nV = 0;
        let nI = 0;
        for (const m of meshes) {
            const g = getMeshGeometry(m);
            if (!g) continue;
            geos.push({ pos: g.positions, idx: g.indices, m });
            nV += g.positions.length;
            nI += g.indices.length;
        }
        if (nV === 0 || nI === 0) return null;
        const world = new Float32Array(nV);
        const idx = new Uint32Array(nI);
        let pOff = 0;
        let iOff = 0;
        let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
        for (const { pos, idx: ix, m } of geos) {
            const w = m.worldMatrix;
            const base = pOff / 3;
            for (let i = 0; i < pos.length; i += 3) {
                const lx = pos[i]!, ly = pos[i + 1]!, lz = pos[i + 2]!;
                const x = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
                const y = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
                const z = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
                world[pOff++] = x; world[pOff++] = y; world[pOff++] = z;
                mnx = Math.min(mnx, x); mny = Math.min(mny, y); mnz = Math.min(mnz, z);
                mxx = Math.max(mxx, x); mxy = Math.max(mxy, y); mxz = Math.max(mxz, z);
            }
            for (let i = 0; i < ix.length; i++) idx[iOff++] = ix[i]! + base;
        }
        const centre: [number, number, number] = [(mnx + mxx) / 2, (mny + mxy) / 2, (mnz + mxz) / 2];
        for (let i = 0; i < nV; i += 3) { world[i] = world[i]! - centre[0]; world[i + 1] = world[i + 1]! - centre[1]; world[i + 2] = world[i + 2]! - centre[2]; }
        const half: [number, number, number] = [(mxx - mnx) / 2, (mxy - mny) / 2, (mxz - mnz) / 2];
        const grid = generateMeshSdf(world, idx, { min: [-half[0], -half[1], -half[2]], max: [half[0], half[1], half[2]], cellSize: cell, padding: SDF_PAD });
        return { grid, centre, half };
    };

    // Bake dynamic-mesh SDFs, and size the arena for the (larger) room grids added later in pass C.
    const dynBaked = new Map<Mesh, { grid: BakedSdf; centre: [number, number, number]; half: [number, number, number] }>();
    for (const m of dynamicMeshes) {
        const b = bakeWorldSdf([m], DYN_CELL);
        if (b) dynBaked.set(m, b);
    }
    // Bake each SDF room's STATIC geometry (walls/floor/props — excludes dynamic + liquefiable meshes)
    // so the fluid is confined to the room and collides with its static props. Baked once at load.
    const SDF_ROOMS = ["CH00_Storage", "CH01_CorridorA"];
    const roomBaked = new Map<string, { grid: BakedSdf; centre: [number, number, number] }>();
    for (const id of SDF_ROOMS) {
        const meshes = chunkStaticMeshes.get(id);
        if (!meshes || !meshes.length) continue;
        const b = bakeWorldSdf(meshes, ROOM_CELL);
        if (b) roomBaked.set(id, { grid: b.grid, centre: b.centre });
    }
    let maxGridFloats = 1;
    for (const b of dynBaked.values()) maxGridFloats = Math.max(maxGridFloats, b.grid.data.length);
    for (const b of roomBaked.values()) maxGridFloats = Math.max(maxGridFloats, b.grid.data.length);

    const floatingSystem = createFloatingBodySystem(engine._device, {
        maxBodies: roomBaked.size + dynBaked.size + 2,
        gridFloats: maxGridFloats,
        externalPose: true,
    });
    floatingSystem.setEnabled(true);

    // Room SDFs = never-moved bodies (no display node; identity pose). bodiesSdf unions them.
    const roomBodies = new Map<string, number>();
    for (const [id, { grid, centre }] of roomBaked) {
        const index = floatingSystem.addBody({ grid, mass: 1, inertia: [1, 1, 1], half: [1, 1, 1], position: centre, scale: 1, displayScale: [1, 1, 1], centre, externalPose: true });
        floatingSystem.setBodyPose(index, centre, [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]);
        roomBodies.set(id, index);
    }
    canvas.dataset.roomBodies = String(roomBodies.size);

    interface DynBody {
        index: number;
        proxy: SceneNode;
        body: PhysicsBody;
        mesh: Mesh;
        disp: SceneNode;
        wriggling?: boolean; // true while THIS body is dissolving: its Havok body still collides/supports,
                             // but the wriggle owns the display node — so skip the pose sync below.
    }
    const dynBodies: DynBody[] = [];
    const dynBodyByMesh = new Map<Mesh, DynBody>();
    for (const [m, baked] of dynBaked) {
        const { grid, centre, half } = baked;
        // Display root at the rest pose the floating system reproduces (mirror preserved); reparent the
        // mesh under it (world-preserving) so the system poses the visible mesh through the root.
        const r = createTransformNode(`dyn_disp_${dynBodies.length}`);
        r.scaling.set(MIRROR[0], MIRROR[1], MIRROR[2]);
        r.position.set(centre[0] - MIRROR[0] * centre[0], centre[1] - MIRROR[1] * centre[1], centre[2] - MIRROR[2] * centre[2]);
        addToScene(scene, r);
        setParent(m, r);
       // Havok DYNAMIC proxy (box from the mesh AABB).
        const proxy = createTransformNode(`dyn_proxy_${dynBodies.length}`, centre[0], centre[1], centre[2]);
        const shape = createPhysicsShape(world, { type: PhysicsShapeType.BOX, parameters: { extents: { x: Math.max(half[0] * 2, 0.05), y: Math.max(half[1] * 2, 0.05), z: Math.max(half[2] * 2, 0.05) } } });
        const body = createPhysicsBody(world, proxy, PhysicsMotionType.DYNAMIC, true);
        setPhysicsBodyShape(world, body, shape);
        const mass = Math.max(0.5, half[0] * half[1] * half[2] * 8 * 200);
        const inertia: [number, number, number] = [
            (mass * (half[1] * half[1] + half[2] * half[2])) / 3,
            (mass * (half[0] * half[0] + half[2] * half[2])) / 3,
            (mass * (half[0] * half[0] + half[1] * half[1])) / 3,
        ];
        const index = floatingSystem.addBody({ grid, mass, inertia, half, position: centre, display: r, scale: 1, displayScale: MIRROR, centre, externalPose: true });
        floatingSystem.setBodyPose(index, centre, [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]);
        const dyn: DynBody = { index, proxy, body, mesh: m, disp: r };
        dynBodies.push(dyn);
        dynBodyByMesh.set(m, dyn);
    }

    // Each physics step, push every dynamic mesh's Havok pose into its floating-body SDF header.
    onPhysicsAfterStep(world, () => {
        for (const d of dynBodies) {
            if (d.wriggling) continue; // dissolving: the wriggle owns the display; the body still collides
            const p = d.proxy.position;
            const q = d.proxy.rotationQuaternion;
            const lv = getPhysicsBodyLinearVelocity(world, d.body);
            const av = getPhysicsBodyAngularVelocity(world, d.body);
            floatingSystem.setBodyPose(d.index, [p.x, p.y, p.z], [q.x, q.y, q.z, q.w], [lv.x, lv.y, lv.z], [av.x, av.y, av.z]);
        }
    });
    canvas.dataset.dynBodies = String(dynBodies.length);

    // Look = mouse (Pointer Lock); move = WASD/arrows in the HORIZONTAL plane. The initial facing
    // comes from the player-start marker's `direction` (glTF space → Lite negates X); with the
    // forward vector being (sin yaw, ·, cos yaw), that is atan2(x, z). Default −π/2 faces Lite −X,
    // down the ship toward the first door.
    const startDir = playerMarker?.direction;
    let yaw = startDir && (startDir[0] || startDir[2]) ? Math.atan2(-startDir[0]!, startDir[2]!) : -Math.PI / 2;
    let pitch = 0;
    const keys = new Set<string>();

    // Free-fly (noclip): fly the camera straight through geometry with no collision or gravity.
    // Toggling on snapshots the current eye position; toggling off drops the character controller
    // back in at that spot (it re-grounds via gravity).
    let noclip = false;
    const freePos = { x: 0, y: 0, z: 0 };
    // Smoothed velocities (m/s) so movement eases in/out instead of snapping on/off (inertia).
    const walkVel = { x: 0, z: 0 };
    const flyVel = { x: 0, y: 0, z: 0 };
    let vy = 0; // vertical velocity (m/s) for gravity + jumping
    let jumpQueued = false; // set on a Space press while grounded; consumed next step
    const toggleNoclip = (): void => {
        noclip = !noclip;
        if (noclip) {
            const p = character.getPosition();
            freePos.x = p.x;
            freePos.y = p.y + EYE;
            freePos.z = p.z;
            flyVel.x = flyVel.y = flyVel.z = 0;
        } else {
            character.setPosition({ x: freePos.x, y: Math.max(freePos.y - EYE, CAP_H / 2), z: freePos.z });
            walkVel.x = walkVel.z = 0;
        }
        canvas.dataset.noclip = String(noclip);
    };

    // ── Inspect mode (press I) ────────────────────────────────────────────────────────────
    // A debug overlay, useful with the mouse released (not pointer-locked): hovering a surface
    // GPU-picks the mesh under the cursor, highlights the whole owning NODE with a translucent box
    // sized to its world AABB, and shows that node's name (the name the Babylon sandbox displays and
    // the key ship_manifest.json's `entities` use), its glTF mesh name, and the room the camera is
    // in. Toggling it on frees the mouse.
    let inspect = false;
    let inspectPicking = false; // one in-flight GPU readback at a time
    let picker: ReturnType<typeof createGpuPicker> | null = null;

    // Reusable highlight box: unlit cyan, alpha-blended so the mesh shows through. Parked far away
    // when hidden; pickable=false so it never picks itself. Added before registerScene so its
    // standard-material pipeline is built with the scene.
    const hlMat = createStandardMaterial();
    hlMat.disableLighting = true;
    hlMat.diffuseColor = [1, 1, 1]; // unlit output = emissiveColor × diffuseColor, so keep this white
    hlMat.emissiveColor = [0.2, 0.9, 1.2];
    hlMat.alpha = 0.35;
    hlMat.backFaceCulling = false;
    const hlBox = createBox(engine, 1);
    hlBox.material = hlMat;
    hlBox.pickable = false;
    const hideHl = (): void => hlBox.position.set(0, -1e6, 0);
    hideHl();
    addToScene(scene, hlBox);

    const panel = document.createElement("div");
    panel.id = "aq-inspect";
    panel.style.cssText =
        "position:fixed;left:12px;top:12px;z-index:20;display:none;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;color:#d6eefc;background:rgba(0,0,0,.6);padding:8px 11px;border-radius:6px;pointer-events:none;white-space:pre;max-width:60vw;";
    document.body.appendChild(panel);

    // Which chunk (room) contains the camera. Camera is Lite-space; convert X back to glTF to test
    // against the manifest AABBs (which are glTF-space).
    const roomAt = (): string => {
        const gx = -cam.position.x;
        const gz = cam.position.z;
        for (const c of manifest?.chunks ?? []) {
            const [x0, , z0] = c.aabb.min;
            const [x1, , z1] = c.aabb.max;
            if (gx >= x0 && gx <= x1 && gz >= z0 && gz <= z1) return c.id;
        }
        return "—";
    };

    let hoverNode = "";
    let hoverMesh = "";
    // ── Collider overlay (B) ─────────────────────────────────────────────────────────────
    // Draws EXACTLY what the player collides with in the current chunk: the manifest-built static box
    // shell (green, with each portal cut out of its wall into side panels + lintel) plus every dynamic
    // body's Havok proxy box (orange). Dynamic proxies are AABBs of their mesh, so a hollow prop such as
    // a door frame shows up as a solid slab over its own opening — which is the point of being able to
    // see them. Meshes are built here, BEFORE registerScene, so their material pipeline is compiled
    // with the scene; they are simply parked/hidden until the overlay is switched on.
    const colStaticMat = createStandardMaterial();
    colStaticMat.disableLighting = true;
    colStaticMat.diffuseColor = [1, 1, 1];
    colStaticMat.emissiveColor = [0.15, 0.95, 0.35];
    colStaticMat.alpha = 0.18;
    colStaticMat.backFaceCulling = false;
    const colDynMat = createStandardMaterial();
    colDynMat.disableLighting = true;
    colDynMat.diffuseColor = [1, 1, 1];
    colDynMat.emissiveColor = [1.3, 0.45, 0.08];
    colDynMat.alpha = 0.3;
    colDynMat.backFaceCulling = false;
    const PARKED = -1e6;
    const makeColBox = (mat: Material): Mesh => {
        const m = createBox(engine, 1);
        m.material = mat;
        m.pickable = false;
        m.position.set(0, PARKED, 0);
        addToScene(scene, m);
        return m;
    };
    const staticColMeshes = colliderBoxes.map(() => makeColBox(colStaticMat));
    const dynColMeshes = dynBodies.map(() => makeColBox(colDynMat));
    let colliderMode = 0; // 0 = off, 1 = dynamic proxies only, 2 = + the chunk's static shell
    const refreshColliders = (): void => {
        const room = roomAt();
        for (let i = 0; i < colliderBoxes.length; i++) {
            const cb = colliderBoxes[i]!;
            const m = staticColMeshes[i]!;
            if (colliderMode < 2 || cb.chunk !== room) {
                m.position.set(0, PARKED, 0);
                continue;
            }
            m.position.set(cb.c[0], cb.c[1], cb.c[2]);
            m.scaling.set(cb.e[0], cb.e[1], cb.e[2]);
        }
        for (let i = 0; i < dynBodies.length; i++) {
            const d = dynBodies[i]!;
            const m = dynColMeshes[i]!;
            const half = dynBaked.get(d.mesh)?.half;
            // Follows the Havok proxy, so a body that is being pushed out is visibly moving.
            if (colliderMode === 0 || !half) {
                m.position.set(0, PARKED, 0);
                continue;
            }
            m.position.set(d.proxy.position.x, d.proxy.position.y, d.proxy.position.z);
            m.scaling.set(Math.max(half[0] * 2, 0.05), Math.max(half[1] * 2, 0.05), Math.max(half[2] * 2, 0.05));
        }
    };
    const toggleColliders = (): void => {
        // Cycles rather than toggles: the static shell is a room-sized box you stand INSIDE, so its back
        // faces wash the whole screen. Stage 1 shows just the dynamic proxies, which is what you want
        // when chasing a prop that is being pushed around.
        colliderMode = (colliderMode + 1) % 3;
        refreshColliders();
        canvas.dataset.colliderOverlay = ["off", "dynamic", "all"][colliderMode]!;
        if (colliderMode === 0) return;
        const room = roomAt();
        const shown = colliderMode === 2 ? colliderBoxes.filter((b) => b.chunk === room) : [];
        // eslint-disable-next-line no-console
        console.log(
            `[aquanova] colliders (${["off", "dynamic", "all"][colliderMode]}) in ${room}: ${shown.length} static box(es)` +
                shown.map((b) => `\n    static  ${b.e.map((v) => v.toFixed(2)).join(" x ")} m  at [${b.c.map((v) => v.toFixed(2)).join(", ")}]`).join("") +
                dynBodies
                    .map((d) => {
                        const h = dynBaked.get(d.mesh)?.half;
                        return h ? `\n    dynamic ${h.map((v) => (v * 2).toFixed(2)).join(" x ")} m  ${nodeNameOfMesh.get(d.mesh) ?? d.mesh.name}` : "";
                    })
                    .join("")
        );
    };

    const refreshPanel = (): void => {
        const what = hoverNode ? `${hoverNode}\nMesh: ${hoverMesh}` : "—  (hover a surface)";
        // Eye/capsule height is here because it is the one number that tells you whether the character
        // controller has settled onto the floor (eye 1.53) or is still at its spawn pose (eye 1.72).
        // The downward ray names whatever is actually supporting the capsule — the chunk floor box is
        // `shipCol_*`, a prop's Havok proxy is `dyn_proxy_*` — so an unexpected perch is self-evident.
        const cp = character.getPosition();
        const footY = cp.y - CAP_H / 2;
        const hit = physicsRaycast(world, { x: cp.x, y: footY + 0.05, z: cp.z }, { x: cp.x, y: footY - 3, z: cp.z });
        const ground = hit.hasHit ? `${hit.body?.node?.name ?? "?"} @ y ${hit.hitPoint.y.toFixed(3)}` : "nothing within 3 m";
        const cam3 = `${cam.position.x.toFixed(2)}, ${cam.position.y.toFixed(2)}, ${cam.position.z.toFixed(2)}`;
        panel.textContent = `INSPECT (I)   Room: ${roomAt()}\nEye: ${cam3}   capsule y ${cp.y.toFixed(3)}   foot y ${footY.toFixed(3)}${noclip ? "   [FREE-FLY]" : ""}\nStanding on: ${ground}\nNode: ${what}`;
    };

    const inspectAt = async (x: number, y: number): Promise<void> => {
        if (!inspect || inspectPicking) return;
        picker ??= createGpuPicker(scene);
        inspectPicking = true;
        try {
            const info = await pickAsync(picker, x, y);
            const mesh = info.hit ? (info.pickedMesh as Mesh | null) : null;
            if (mesh) {
                // Report the glTF NODE name — that's what the Babylon sandbox shows and what
                // ship_manifest.json's `entities` are keyed by. `mesh.name` is the glTF MESH name,
                // shared by every instance (all six door leaves are "Cube.009"), so it is shown
                // second for reference only.
                hoverNode = nodeNameOfMesh.get(mesh) || mesh.name || "(unnamed)";
                hoverMesh = mesh.name || "(unnamed)";
                // Highlight the WHOLE node: a multi-primitive node is one object to the player (it
                // liquefies as a unit), so union the AABBs of all its primitives.
                let mn: number[] | null = null;
                let mx: number[] | null = null;
                for (const prim of nodePrimitives.get(mesh) ?? [mesh]) {
                    const a = prim.boundMin;
                    const b = prim.boundMax;
                    if (!a || !b) continue;
                    mn = mn ? [Math.min(mn[0]!, a[0]!), Math.min(mn[1]!, a[1]!), Math.min(mn[2]!, a[2]!)] : [a[0]!, a[1]!, a[2]!];
                    mx = mx ? [Math.max(mx[0]!, b[0]!), Math.max(mx[1]!, b[1]!), Math.max(mx[2]!, b[2]!)] : [b[0]!, b[1]!, b[2]!];
                }
                if (mn && mx) {
                    hlBox.position.set((mn[0]! + mx[0]!) / 2, (mn[1]! + mx[1]!) / 2, (mn[2]! + mx[2]!) / 2);
                    hlBox.scaling.set(Math.max(mx[0]! - mn[0]!, 0.02) * 1.03, Math.max(mx[1]! - mn[1]!, 0.02) * 1.03, Math.max(mx[2]! - mn[2]!, 0.02) * 1.03);
                } else {
                    hideHl();
                }
            } else {
                hoverNode = "";
                hoverMesh = "";
                hideHl();
            }
            refreshPanel();
        } finally {
            inspectPicking = false;
        }
    };

    const toggleInspect = (): void => {
        inspect = !inspect;
        if (inspect) {
            if (document.pointerLockElement === canvas) document.exitPointerLock(); // free the mouse to hover
            refreshPanel();
            panel.style.display = "block";
        } else {
            panel.style.display = "none";
            hoverNode = "";
            hoverMesh = "";
            hideHl();
        }
    };

    const LOOK_SENS = 1 / 600; // radians per pixel of mouse motion

    // ── Liquefactor weapon: crosshair + centre-screen targeting ───────────────────────────
    // The crosshair marks where the weapon points (screen centre). LMB while the mouse is captured
    // fires: GPU-pick the mesh at the centre and, if it is in the liquefiable set, liquefy it.
    const crosshair = document.createElement("div");
    crosshair.id = "aq-crosshair";
    crosshair.style.cssText = "position:fixed;left:50%;top:50%;width:24px;height:24px;transform:translate(-50%,-50%);z-index:15;pointer-events:none;";
    const bar = (s: string): string => `<div style="position:absolute;background:rgba(120,232,255,.92);${s}"></div>`;
    crosshair.innerHTML =
        bar("left:50%;top:0;width:2px;height:8px;margin-left:-1px;") +
        bar("left:50%;bottom:0;width:2px;height:8px;margin-left:-1px;") +
        bar("top:50%;left:0;height:2px;width:8px;margin-top:-1px;") +
        bar("top:50%;right:0;height:2px;width:8px;margin-top:-1px;");
    document.body.appendChild(crosshair);

    let firing = false;
    // Assigned by the Milestone D fluid block below; called when the crosshair is on a liquefiable mesh.
    let liquefyMesh: (mesh: Mesh, hitPoint?: readonly [number, number, number] | null) => void = () => {};
    const fireLiquefactor = async (): Promise<void> => {
        if (firing) return;
        picker ??= createGpuPicker(scene);
        firing = true;
        try {
            const info = await pickAsync(picker, canvas.clientWidth / 2, canvas.clientHeight / 2);
            const mesh = info.hit ? (info.pickedMesh as Mesh | null) : null;
            canvas.dataset.target = mesh ? `${nodeNameOfMesh.get(mesh) ?? mesh.name}|${liquefiableMeshes.has(mesh) ? "liq" : "no"}` : "none";
            if (mesh && liquefiableMeshes.has(mesh)) liquefyMesh(mesh, info.pickedPoint);
        } finally {
            firing = false;
        }
    };

    // Mouse look via Pointer Lock: clicking captures the pointer (hiding the cursor); moving the
    // mouse then rotates the view with NO button held, leaving LMB free for shooting later. Esc
    // releases the lock. The keyboard is bound on window so movement works without canvas focus.
    canvas.addEventListener("click", () => {
        if (inspect) return; // in inspect mode a click must not re-capture the mouse
        if (document.pointerLockElement !== canvas) {
            void Promise.resolve(canvas.requestPointerLock()).catch(() => {});
        }
    });
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return; // LMB only
        if (document.pointerLockElement !== canvas || inspect || noclip) return; // only while playing
        void fireLiquefactor();
    });
    canvas.addEventListener("pointermove", (e) => {
        if (document.pointerLockElement === canvas) {
            yaw += e.movementX * LOOK_SENS;
            pitch = Math.max(-1.45, Math.min(1.45, pitch - e.movementY * LOOK_SENS));
            return;
        }
        if (inspect) void inspectAt(e.offsetX, e.offsetY); // hover-pick when the mouse is free
    });
    window.addEventListener("keydown", (e) => {
        if (e.code === "KeyV" && !e.repeat) toggleNoclip(); // V → toggle free-fly (noclip)
        if (e.code === "KeyI" && !e.repeat) toggleInspect(); // I → toggle inspect overlay
        if (e.code === "KeyB" && !e.repeat) toggleColliders(); // B → toggle collision-box overlay
        if (e.code === "Space" && !e.repeat && !noclip) jumpQueued = true; // Space → jump (walk mode)
        keys.add(e.code);
    });
    window.addEventListener("keyup", (e) => keys.delete(e.code));

    const SPEED = 4; // m/s walk speed (×2 while Shift is held)
    const FLY_SPEED = 8; // m/s free-fly base speed
    const MOVE_ACCEL = 12; // 1/s velocity-smoothing rate → gentle ease-in/out (higher = snappier)
    const JUMP_SPEED = 6; // m/s upward launch (≈1.1 m apex)
    const JUMP_GRAVITY = 16; // m/s² fall acceleration while airborne
    const DOWN = { x: 0, y: -1, z: 0 };
    let fpFrozen = false; // debug: when true, stop driving the camera (so QA can place it freely)
    onPhysicsAfterStep(world, (dt) => {
        const inZ = (keys.has("KeyW") || keys.has("ArrowUp") ? 1 : 0) - (keys.has("KeyS") || keys.has("ArrowDown") ? 1 : 0);
        const inX = (keys.has("KeyD") || keys.has("ArrowRight") ? 1 : 0) - (keys.has("KeyA") || keys.has("ArrowLeft") ? 1 : 0);
        const cos = Math.cos(yaw);
        const sin = Math.sin(yaw);
        const cp = Math.cos(pitch);
        const sp = Math.sin(pitch);
        const run = keys.has("ShiftLeft") || keys.has("ShiftRight") ? 2 : 1; // Shift = go faster
        // Frame-rate-independent exponential smoothing toward the target velocity → soft inertia.
        const k = 1 - Math.exp(-dt * MOVE_ACCEL);

        // Free-fly / noclip: move the camera directly along the view axis (Space/Ctrl or C for
        // vertical), ignoring collisions and gravity.
        if (noclip) {
            const up = (keys.has("Space") ? 1 : 0) - (keys.has("ControlLeft") || keys.has("KeyC") ? 1 : 0);
            const spd = FLY_SPEED * run;
            flyVel.x += ((sin * cp * inZ + cos * inX) * spd - flyVel.x) * k;
            flyVel.y += ((sp * inZ + up) * spd - flyVel.y) * k;
            flyVel.z += ((cos * cp * inZ - sin * inX) * spd - flyVel.z) * k;
            freePos.x += flyVel.x * dt;
            freePos.y += flyVel.y * dt;
            freePos.z += flyVel.z * dt;
            cam.position.set(freePos.x, freePos.y, freePos.z);
            cam.target.set(freePos.x + sin * cp, freePos.y + sp, freePos.z + cos * cp);
            canvas.dataset.pos = `${freePos.x.toFixed(2)},${freePos.y.toFixed(2)},${freePos.z.toFixed(2)}`;
            return;
        }

        // Walk: ease the horizontal velocity toward the input direction.
        const spd = SPEED * run;
        walkVel.x += ((inX * cos + inZ * sin) * spd - walkVel.x) * k;
        walkVel.z += ((-inX * sin + inZ * cos) * spd - walkVel.z) * k;
        // Vertical: grounded → stick to the floor (or launch a queued jump); airborne → integrate
        // gravity so the jump arcs.
        const grounded = character.checkSupport(dt, DOWN).supportedState === CharacterSupportedState.SUPPORTED;
        if (grounded && vy <= 0) {
            vy = jumpQueued ? JUMP_SPEED : -2;
        } else {
            vy -= JUMP_GRAVITY * dt;
        }
        jumpQueued = false;
        character.moveWithCollisions({ x: walkVel.x * dt, y: vy * dt, z: walkVel.z * dt });
        const p = character.getPosition();
        if (fpFrozen) {
            canvas.dataset.pos = `${p.x.toFixed(2)},${p.y.toFixed(2)},${p.z.toFixed(2)}`;
            return;
        }
        cam.position.set(p.x, p.y + EYE, p.z);
        cam.target.set(p.x + sin * cp, p.y + EYE + sp, p.z + cos * cp);
        canvas.dataset.pos = `${p.x.toFixed(2)},${p.y.toFixed(2)},${p.z.toFixed(2)}`;
    });

    // Test hook (QA): read the player position + nudge look/movement programmatically.
    (window as unknown as { __aquanova?: unknown }).__aquanova = {
        getPos: (): { x: number; y: number; z: number } => character.getPosition(),
        setYaw: (y: number): void => {
            yaw = y;
        },
        look: (dx: number, dy: number): void => {
            yaw += dx * LOOK_SENS;
            pitch = Math.max(-1.45, Math.min(1.45, pitch - dy * LOOK_SENS));
        },
        press: (code: string): void => {
            keys.add(code);
        },
        release: (code: string): void => {
            keys.delete(code);
        },
        toggleNoclip: (): void => toggleNoclip(),
        toggleInspect: (): void => toggleInspect(),
        toggleColliders: (): void => toggleColliders(),
        colliderBoxes: (): Array<{ chunk: string; c: number[]; e: number[] }> => colliderBoxes.map((b) => ({ chunk: b.chunk, c: [...b.c], e: [...b.e] })),
        roomAt: (): string => roomAt(),
        camState: (): Record<string, number> => ({
            px: cam.position.x,
            py: cam.position.y,
            pz: cam.position.z,
            tx: cam.target.x,
            ty: cam.target.y,
            tz: cam.target.z,
            fov: (cam as unknown as { fov?: number }).fov ?? -1,
            near: cam.nearPlane,
            far: cam.farPlane,
        }),
        grading: (): Record<string, unknown> => ({
            exposure: scene.imageProcessing.exposure,
            contrast: scene.imageProcessing.contrast,
            toneMapping: scene.imageProcessing.toneMapping,
            toneMappingEnabled: scene.imageProcessing.toneMappingEnabled,
            ibl: (allShipMeshes.find((m) => m.material)?.material as unknown as { environmentIntensity?: number } | undefined)?.environmentIntensity,
        }),
        fire: (): void => {
            void fireLiquefactor();
        },
        liqTargets: (): Array<{ name: string; c: number[] | null }> =>
            [...liquefiableMeshes].map((m) => ({
                name: nodeNameOfMesh.get(m) ?? m.name,
                c: m.boundMin && m.boundMax ? [(m.boundMin[0] + m.boundMax[0]) / 2, (m.boundMin[1] + m.boundMax[1]) / 2, (m.boundMin[2] + m.boundMax[2]) / 2] : null,
            })),
        dynPos: (): Array<{ name: string; pos: number[]; half: number[] }> =>
            dynBodies.map((d) => ({
                name: nodeNameOfMesh.get(d.mesh) ?? d.mesh.name,
                pos: [d.proxy.position.x, d.proxy.position.y, d.proxy.position.z],
                half: dynBaked.get(d.mesh)?.half ?? [0, 0, 0],
            })),
        liquefyNamed: (name: string): string | null => {
            for (const m of liquefiableMeshes) {
                if ((nodeNameOfMesh.get(m) ?? m.name) === name) {
                    liquefyMesh(m);
                    return name;
                }
            }
            return null;
        },
        warp: (x: number, y: number, z: number): void => character.setPosition({ x, y, z }),
        freeze: (): void => {
            fpFrozen = true;
        },
        meltState: (): Array<{ name: string; phase: string; frontR: number; maxR: number }> =>
            activeSims.map((a) => ({ name: nodeNameOfMesh.get(a.mesh) ?? a.mesh.name, phase: a.phase, frontR: a.state ? a.state.frontR : -1, maxR: a.maxR })),
        sdfExcluded: (): number[] => [...sdfExcludeRefs.keys()].sort((x, y) => x - y),
        dynBodyIndexOf: (nodeName: string): number[] =>
            [...(meshesByNodeName.get(nodeName) ?? [])].map((m) => dynBodyByMesh.get(m)?.index ?? -1).filter((i) => i >= 0),
        camTo: (px: number, py: number, pz: number, tx: number, ty: number, tz: number): void => {
            cam.position.set(px, py, pz);
            cam.target.set(tx, ty, tz);
        },
    };

    // ── Fluid render path: scene → sceneColorRT, then the fluid surface composites to the swapchain ─
    // (Liquefactor pattern. Single-sample scene target → MSAA is off on this path.) The scene colour
    // target bakes its OWN depth (dFormat) rather than taking an external depth attachment: with
    // transmissive materials + a skybox, the transmission base-task's opaque bundle reads its depth
    // format from the colour RT descriptor, so an externally-supplied depth would desync the bundle
    // from the pipelines and drop every draw. The surface samples this same baked depth for compositing.
    const device = engine._device;
    const sceneColorRT = createRenderTarget({ lbl: "aq-scene-color", format: engine.format, dFormat: "depth24plus", samples: 1, size: engine });
    addTask(scene, createRenderTask({ name: "scene", rt: sceneColorRT, clr: true, clrColor: scene.clearColor }, engine, scene));

    // Where the ship was last drawn, plus that draw's matrices — the source for per-particle LIT
    // colours (see particle-lit-colors.ts). The scene target is written every frame before the fluid
    // composite, so at liquefy time it still holds the solid prop we are about to melt.
    const litScene = (): LitColorScene | null => {
        const rt = sceneColorRT as unknown as { _colorView: GPUTextureView | null; _depthView: GPUTextureView | null };
        // The scene colour/depth textures are allocated lazily on first render. Until then there is
        // no rendered image to promote the albedo against, so skip the lit pass for that shot rather
        // than binding null views (which throws inside createBindGroup).
        if (!rt._colorView || !rt._depthView) return null;
        const aspect = engine.canvas.width / Math.max(1, engine.canvas.height);
        const proj = getProjectionMatrix(cam, aspect);
        return {
            colorView: rt._colorView,
            depthView: rt._depthView,
            view: getViewMatrix(cam),
            viewProj: getViewProjectionMatrix(cam, aspect),
            projZW: proj[14]!,
            projZZ: proj[10]!,
        };
    };

    const MAX_TOTAL = 200000;
    const combinedPos = device.createBuffer({ label: "aq-combined-pos", size: MAX_TOTAL * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const combinedDebug = device.createBuffer({ label: "aq-combined-debug", size: MAX_TOTAL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const combinedAlpha = device.createBuffer({ label: "aq-combined-alpha", size: MAX_TOTAL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const alphaScratch = new Float32Array(MAX_TOTAL).fill(1);
    device.queue.writeBuffer(combinedAlpha, 0, alphaScratch);
    // Per-particle RGBA colour — filled per shot from the liquefied mesh's base-colour texture when its
    // fluidSim setting has useMeshColors (else the base water tint). Aggregated like combinedPos each
    // frame; the surface tints the water by it. WATER_RGB doubles as the non-mesh-colour fallback fill.
    const combinedColor = device.createBuffer({ label: "aq-combined-color", size: MAX_TOTAL * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const WATER_RGB: [number, number, number] = [0x16 / 255, 0xa3 / 255, 0xc3 / 255]; // #16a3c3 cyan water
    const colorScratch = new Float32Array(MAX_TOTAL * 4);
    for (let i = 0; i < MAX_TOTAL; i++) {
        colorScratch[i * 4] = WATER_RGB[0];
        colorScratch[i * 4 + 1] = WATER_RGB[1];
        colorScratch[i * 4 + 2] = WATER_RGB[2];
        colorScratch[i * 4 + 3] = 1;
    }
    const virtualSim = {
        count: 0,
        particleRadius: 0.08,
        surfaceSizeScale: 1,
        positionBuffer: combinedPos,
        velocityBuffer: combinedPos,
        debugBuffer: combinedDebug,
        debugNorm: 1,
        gpuBytes: 0,
        step: (): void => {},
        reset: (): void => {},
        setParam: (): void => {},
        setSceneSdf: (): void => {},
        setEmitters: (): void => {},
        setSpawn: (): void => {},
        setForceField: (): void => {},
        dispose: (): void => {},
    };
    const surfaceTask = createFluidSurfaceTask(engine, scene, { bgRT: sceneColorRT, outRT: engine.scRT, depthRT: sceneColorRT, camera: cam, sim: virtualSim as unknown as FluidSim });
    surfaceTask.setSim(virtualSim as unknown as FluidSim);
    surfaceTask.setParticleAlpha(combinedAlpha);
    surfaceTask.setParticleColor(combinedColor);
    // Water look (mirrors the Liquefactor demo defaults): cyan-blue refractive fluid lit by the HDR
    // environment. Without setEnvMap the surface samples garbage for reflections and renders opaque green.
    surfaceTask.setDirLight([-0.4, -0.82, -0.45]);
    surfaceTask.setFluidColor(WATER_RGB); // #16a3c3 cyan water
    surfaceTask.setAbsorption(0.4);
    surfaceTask.setSizeScale(0.7);
    surfaceTask.setRefractionStrength(0.06);
    surfaceTask.setSpecularPower(41);
    if (env.specularCubeView && env.cubeSampler) surfaceTask.setEnvMap({ view: env.specularCubeView, sampler: env.cubeSampler });
    addTask(scene, surfaceTask);

    // ── Per-particle mesh-colour path (honours a fluidSim setting's useMeshColors) ─────────────────
    // Colour each water particle from the liquefied mesh's base-colour texture: nearest mesh vertex →
    // its UV → textureLoad, gamma-encoded (the sRGB view decodes to linear on load) to match the
    // display-space fluid composite. Ported from the Liquefactor demo's per-particle colour render.
    const COLOR_SAMPLE_WGSL = /* wgsl */ `
struct P { count: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> uvs: array<vec2<f32>>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outCol: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    let dims = vec2<f32>(textureDimensions(tex, 0));
    let w = fract(uvs[i]);                          // repeat-wrap
    let coord = vec2<i32>(clamp(w * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));
    let c = textureLoad(tex, coord, 0);             // sRGB view → linear
    outCol[i] = vec4<f32>(pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}`;
    const colorPipe = device.createComputePipeline({ label: "aq-color-sample", layout: "auto", compute: { module: device.createShaderModule({ label: "aq-color-sample", code: COLOR_SAMPLE_WGSL }), entryPoint: "main" } });

    // Build a per-particle RGBA colour buffer by texture-sampling each particle's UV. The UVs are
    // computed alongside the particles (in the worker when one is available), so nothing here touches
    // the mesh geometry. Falls back to the base water tint when the mesh lacks UVs or a base-colour
    // texture.
    const buildMeshColorBuffer = (count: number, particleUvs: Float32Array | null, texView: GPUTextureView | undefined): GPUBuffer => {
        const buf = device.createBuffer({ label: "aq-shot-color", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(buf, 0, colorScratch, 0, count * 4); // water-tint fallback (overwritten below when textured)
        if (!particleUvs || particleUvs.length < count * 2 || !texView) return buf;
        const uvBuf = device.createBuffer({ label: "aq-shot-uv", size: count * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(uvBuf, 0, particleUvs, 0, count * 2);
        const cbuf = device.createBuffer({ label: "aq-shot-ccount", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(cbuf, 0, new Uint32Array([count, 0, 0, 0]));
        const enc = device.createCommandEncoder();
        const bg = device.createBindGroup({
            layout: colorPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cbuf } },
                { binding: 1, resource: { buffer: uvBuf } },
                { binding: 2, resource: texView },
                { binding: 3, resource: { buffer: buf } },
            ],
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(colorPipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(count / 64));
        pass.end();
        device.queue.submit([enc.finish()]);
        uvBuf.destroy();
        cbuf.destroy();
        return buf;
    };

    // ── Milestone D: Liquefaction ────────────────────────────────────────────────────────────────
    // Shared scene SDF for every liquefaction sim: the UNION of every room + dynamic-mesh floating
    // body (bodiesSdf reads the pose header + baked grids from the floating system's storage buffer,
    // so Havok-driven pose changes on dynamic meshes flow straight into the fluid collision each step).
    const sceneSdfUbo = device.createBuffer({ label: "aq-scene-sdf", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sceneSdfUbo, 0, new Float32Array([0, 0, 0, 0]));
    const sceneSdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { pad: vec4<f32>, };",
        sdf: `${floatingSystem.sdfWgsl}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return bodiesSdf(pt, dt) + sceneSdfParams.pad.x * 0.0; }`,
        buffer: sceneSdfUbo,
        sdfGrid: floatingSystem.sdfBuffer,
    };

    // A radial burst from the volume centre plus a gentle uniform lift, so a liquefied prop erupts
    // outward before settling (imported from the Liquefactor demo).
    const IMPULSE_WGSL = /* wgsl */ `
fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> {
    let center = forceFieldParams.center.xyz;
    let radius = max(forceFieldParams.center.w, 1.0e-4);
    let toParticle = pos - center;
    let dist = length(toParticle);
    var f = vec3<f32>(0.0, forceFieldParams.push.y, 0.0);
    if (dist < radius) {
        let dir = select(vec3<f32>(0.0, 1.0, 0.0), toParticle / max(dist, 1.0e-4), dist > 1.0e-4);
        let core = smoothstep(0.0, 0.15, dist / radius);
        f += dir * forceFieldParams.push.x * core;
    }
    return f * dt;
}`;
    const LIFETIME = 5.0; // seconds a settled blob lives before it fades out
    const FADE_DUR = 1.2; // seconds the alpha fade-out takes before dispose + mesh removal
    const IMPULSE_RADIAL_BASE = 8; // outward burst from the volume centre (× impulse)
    const IMPULSE_UP_BASE = 10; // upward lift so the fluid erupts into view before settling
    const SPREAD_MARGIN = 1.2; // extra half-width (world units) around the mesh footprint for the sim grid
    // Fallback MLS-MPM water used when the manifest lists no fluidSim settings (or a name fails to load).
    const FLUID_SETTING = { gravity: 9.8, stiffness: 350, viscosity: 0.3, restDensity: 3, damping: 0.995, affineDamping: 0.9, groundDamp: 0.85, groundDampHeight: 1.5, restitution: 0.3, substeps: 3 };
    // Build the solver a liquefied prop erupts into, honouring the chosen setting's method + params. The
    // settings can be MLS-MPM or PB-MPM. Render (colour/absorption/…) is a SHARED surface pass across all
    // active blobs, so it stays global — only the simulation varies per setting.
    const buildFluidSim = (
        setting: FluidSimSetting | undefined,
        o: { count: number; particleRadius: number; initialPositions: Float32Array; boundsMin: [number, number, number]; boundsMax: [number, number, number]; dx: number }
    ): FluidSim => {
        const base = { count: o.count, particleRadius: o.particleRadius, initialPositions: o.initialPositions, boundsMin: o.boundsMin, boundsMax: o.boundsMax, dx: o.dx, groundY: FLOOR_Y };
        if (setting?.method === "PB-MPM") {
            const p = setting.physics;
            return createPbMpmSim(engine, {
                ...base,
                material: setting.material ?? 0,
                gravity: p.gravity, substeps: p.substeps, iterations: p.iterations,
                liquidRelaxation: p.liquidRelaxation, liquidViscosity: p.liquidViscosity,
                elasticityRatio: p.elasticityRatio, elasticRelaxation: p.elasticRelaxation,
                frictionAngle: p.frictionAngle, plasticity: p.plasticity, restitution: p.restitution,
            });
        }
        const p = setting?.method === "MLS-MPM" ? setting.physics : FLUID_SETTING;
        return createMlsMpmSim(engine, {
            ...base,
            gravity: p.gravity, stiffness: p.stiffness, viscosity: p.viscosity, restDensity: p.restDensity,
            substeps: p.substeps, damping: p.damping, affineDamping: p.affineDamping,
            groundDamp: p.groundDamp, groundDampHeight: p.groundDampHeight, restitution: p.restitution,
        });
    };
    const WRIGGLE_AMP = 0.016; // per-frame "pain" jitter amplitude (world units)
    // Dissolve-front growth (world units / s). Matched to Liquefactor's SHIP-mode front (its gallery
    // speed of 6 divided by 3): ship modules read as architecture rather than a single prop, so a faster
    // sweep pops instead of melting. Liquefactor loads this same ship.glb to audition liquefaction, so
    // the two must agree or what the audition shows is not what the game does.
    const LIQUEFY_SPEED = 2;
    const DEFAULT_SAMPLE_RADIUS = 0.03; // volume-sampling spacing when a fluidSim setting omits demoParams.particleRadius
    const LIQUEFY_EDGE = 0.6; // fire-glow band width at the dissolving boundary

    // Dissolve clip: give each liquefiable mesh its OWN material instance (a shallow clone — textures
    // and build group are shared by reference) carrying a private liquefy plugin + state. The ship's
    // materials are shared level-wide (many walls/props reuse e.g. MI_Trim_02), so attaching the clip
    // to the shared material would make EVERY mesh using it clip/glow around the same hit point. Cloning
    // per mesh isolates the dissolve to the shot target. Done BEFORE registerScene so each clone's
    // pipeline is compiled with the plugin; disabled (no-op) until a shot fires.
    type PluginMat = { plugins?: unknown[]; _uboVersion: number };
    const liquefyStates = new Map<Mesh, LiquefyState>();
    for (const m of dissolvableMeshes) {
        const mat = m.material;
        if (!mat || !isPbrMaterial(mat)) continue;
        const st: LiquefyState = { hit: [0, 0, 0], frontR: 0, edge: LIQUEFY_EDGE, enabled: false };
        const src = mat as unknown as PluginMat;
        const clone = { ...(mat as object), _uboVersion: 0, plugins: [...(src.plugins ?? []), createLiquefyPlugin(() => st, "pbr")] } as unknown as Material;
        m.material = clone;
        liquefyStates.set(m, st);
    }
    const bumpMat = (mat: Material): void => { (mat as unknown as PluginMat)._uboVersion++; };
    // Front radius that fully engulfs the mesh AABB from the hit point (+ edge band + margin).
    const computeMaxR = (hit: readonly [number, number, number], bMin: readonly [number, number, number], bMax: readonly [number, number, number]): number => {
        let maxD = 0;
        for (const px of [bMin[0], bMax[0]]) for (const py of [bMin[1], bMax[1]]) for (const pz of [bMin[2], bMax[2]]) maxD = Math.max(maxD, Math.hypot(px - hit[0]!, py - hit[1]!, pz - hit[2]!));
        return maxD + LIQUEFY_EDGE + 0.5;
    };

    interface ActiveSim {
        sim: FluidSim;
        mesh: Mesh;
        disp: SceneNode | null; // dynamic-mesh display root (dispose the whole subtree on fade-out)
        dyn: DynBody | null; // dynamic-body handle: its Havok collider + SDF body are torn down at finishShot (null for a static prop)
        phase: "dissolving" | "fluid" | "fading";
        fluidElapsed: number;
        fadeElapsed: number;
        impulseRemaining: number;
        impulseBuffer: GPUBuffer;
        impulseSpec: ForceFieldSpec; // applied at the END of the dissolve (finishShot), not before
        // Dissolve/wriggle state:
        state: LiquefyState | null; // material clip state (null if the mesh had no PBR material)
        material: Material | null;
        maxR: number;
        wriggleNode: SceneNode; // node jittered during the dissolve (display root, or the mesh itself)
        wriggleBase: [number, number, number];
        waterBase: Float32Array; // sampled particle positions (xyzw), jittered into the sim each frame
        waterScratch: Float32Array;
        colorBuffer: GPUBuffer | null; // per-particle mesh colours (present only when the setting has useMeshColors)
        useMeshColors: boolean;
        group: ShotGroup; // members erupt together, not as each one finishes melting
        dissolved: boolean; // front fully grown; waiting on the rest of the group
    }
    const activeSims: ActiveSim[] = [];

    // One shot = one GROUP (the melted node's primitives + every linked node's, transitively). The
    // members must erupt together, so the fluid sims are held in the dissolving phase until the LAST
    // of them has finished melting: `sampling` counts members still waiting on a seed and `sims`
    // collects the ones that produced a sim.
    interface ShotGroup {
        sampling: number;
        sims: ActiveSim[];
        /** Floating-body indices this shot removes from the SDF union while its water is alive. */
        excluded: number[];
        /** True once the exclusions are actually held, so the release runs exactly once. */
        held: boolean;
    }

    // `excludeSDF` refcount: two shots may exclude the same body (a door melted while its twin's water
    // is still running), so a body is only restored once the LAST shot holding it is gone.
    const sdfExcludeRefs = new Map<number, number>();
    const holdSdfExclusions = (group: ShotGroup): void => {
        if (group.held) return;
        group.held = true;
        for (const i of group.excluded) {
            const n = (sdfExcludeRefs.get(i) ?? 0) + 1;
            sdfExcludeRefs.set(i, n);
            if (n === 1) floatingSystem.setBodySdfActive(i, false);
        }
    };
    const releaseSdfExclusions = (group: ShotGroup): void => {
        if (!group.held) return;
        group.held = false;
        for (const i of group.excluded) {
            const n = (sdfExcludeRefs.get(i) ?? 1) - 1;
            if (n > 0) {
                sdfExcludeRefs.set(i, n);
            } else {
                sdfExcludeRefs.delete(i);
                floatingSystem.setBodySdfActive(i, true);
            }
        }
    };

    // ── Sampling worker pool ─────────────────────────────────────────────────────────────
    // The CPU particle fill is 100–400 ms for a typical prop and scales with the group size (a
    // 4-primitive node samples four times, a linked door twice), so running it inline froze the
    // frame on every shot. It now runs in workers: `requestSample` posts the WORLD-space geometry
    // and `applySample` builds the sim when the seed comes back. Everything else (sim creation,
    // colour buffers) is GPU work measured at 0–14 ms, so it stays on the main thread.
    type SampleResult = { positions: Float32Array; count: number; radius: number; min: [number, number, number]; max: [number, number, number]; uvs: Float32Array | null };
    interface PendingSample {
        mesh: Mesh;
        hit: readonly [number, number, number] | null;
        setting: FluidSimSetting | undefined;
        settingName: string | undefined;
        dyn: DynBody | null;
        wriggleNode: SceneNode; // jittered from the moment the shot lands, before the seed arrives
        wriggleBase: [number, number, number];
        group: ShotGroup;
    }
    type SampleMsg = { id: number; positions: Float32Array; uvs: Float32Array | null; count: number; radius: number; boundsMin: [number, number, number]; boundsMax: [number, number, number] };
    interface PoolWorker {
        worker: Worker;
        pending: number;
    }
    const workerPool: PoolWorker[] = [];
    const pendingSamples = new Map<number, PendingSample>();
    // Shots whose seed is still being sampled. The frame loop pain-shakes these so the hit reads as
    // instant feedback instead of the prop sitting still until the worker replies.
    const samplingShots = new Set<PendingSample>();
    let sampleSeq = 0;

    const abortSample = (entry: PendingSample): void => {
        samplingShots.delete(entry);
        entry.group.sampling--;
        entry.wriggleNode.position.set(entry.wriggleBase[0], entry.wriggleBase[1], entry.wriggleBase[2]);
        if (entry.dyn) entry.dyn.wriggling = false;
        // Nothing sampled — hand the mesh back so it stays shootable rather than becoming an inert
        // solid that can never be liquefied.
        liquefiableMeshes.add(entry.mesh);
        dissolvableMeshes.add(entry.mesh);
        eruptIfReady(entry.group); // the rest of the group may have been waiting on this one
    };

    const onSampleMessage = (ev: MessageEvent<SampleMsg>): void => {
        const { id, positions, uvs, count, radius, boundsMin, boundsMax } = ev.data;
        const entry = pendingSamples.get(id);
        pendingSamples.delete(id);
        if (!entry) return;
        if (!count) {
            abortSample(entry);
            return;
        }
        applySample(entry, { positions, count, radius, min: boundsMin, max: boundsMax, uvs });
    };
    try {
        if (typeof Worker !== "undefined") {
            const poolSize = Math.min(3, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
            for (let i = 0; i < poolSize; i++) {
                const worker = new Worker(new URL("./particle-fill-worker.ts", import.meta.url), { type: "module" });
                const pw: PoolWorker = { worker, pending: 0 };
                worker.addEventListener("message", (ev: MessageEvent<SampleMsg>) => {
                    pw.pending = Math.max(0, pw.pending - 1);
                    onSampleMessage(ev);
                });
                worker.addEventListener("error", (e) => {
                    // eslint-disable-next-line no-console
                    console.warn("[aquanova] sample worker error", e.message);
                    const idx = workerPool.indexOf(pw);
                    if (idx >= 0) workerPool.splice(idx, 1); // drop the faulty worker; sync fallback once the pool empties
                });
                workerPool.push(pw);
            }
        }
    } catch {
        workerPool.length = 0;
    }
    // Least-loaded worker, so a multi-primitive node's primitives sample in parallel.
    const pickWorker = (): PoolWorker | null => {
        let best: PoolWorker | null = null;
        for (const pw of workerPool) if (!best || pw.pending < best.pending) best = pw;
        return best;
    };
    canvas.dataset.sampleWorkers = String(workerPool.length);

    const requestSample = (mesh: Mesh, hitPoint: readonly [number, number, number] | null, settingName: string | undefined, group: ShotGroup): void => {
        const g = getMeshGeometry(mesh);
        if (!g) return;
        // Transform the mesh's LOCAL geometry into WORLD space (the fluid lives in world coords).
        // Captured NOW, so a Havok-posed prop is sampled in the pose it had when it was shot.
        const w = mesh.worldMatrix;
        const src = g.positions;
        const worldPos = new Float32Array(src.length);
        for (let i = 0; i < src.length; i += 3) {
            const lx = src[i]!, ly = src[i + 1]!, lz = src[i + 2]!;
            worldPos[i] = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
            worldPos[i + 1] = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
            worldPos[i + 2] = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
        }
        // The setting is chosen ONCE per shot by the caller and shared across the whole group; its
        // demoParams.particleRadius drives the volume-sampling spacing (and therefore the particle
        // count), so the same setting file yields the same count here as in the Liquefactor demo.
        const setting = settingName ? fluidSettings.get(settingName) : undefined;
        const radius = setting?.particleRadius ?? DEFAULT_SAMPLE_RADIUS;
        // Retire the mesh as a target IMMEDIATELY: sampling is async now, so leaving it in the sets
        // would let a second shot (or a linked cascade) start a duplicate sim for the same mesh.
        liquefiableMeshes.delete(mesh);
        dissolvableMeshes.delete(mesh);
        // Start the pain shake NOW, on the click, rather than when the seed arrives — the sample takes
        // 100–400 ms in the worker and the prop would otherwise stand still through it. `wriggling`
        // decouples the display node from its Havok pose so the shake doesn't fight physics; the base
        // is captured here and reused by applySample, which must not read the already-jittered pose.
        const dyn = dynBodyByMesh.get(mesh) ?? null;
        if (dyn) dyn.wriggling = true;
        const wriggleNode = dyn ? dyn.disp : (mesh as unknown as SceneNode);
        const entry: PendingSample = {
            mesh,
            hit: hitPoint ?? null,
            setting,
            settingName,
            dyn,
            wriggleNode,
            wriggleBase: [wriggleNode.position.x, wriggleNode.position.y, wriggleNode.position.z],
            group,
        };
        group.sampling++;
        samplingShots.add(entry);
        const uvs = g.uvs ? g.uvs.slice() : null;
        const pw = pickWorker();
        if (pw) {
            const id = ++sampleSeq;
            pendingSamples.set(id, entry);
            pw.pending++;
            const indices = g.indices.slice();
            const transfer: Transferable[] = [worldPos.buffer, indices.buffer];
            if (uvs) transfer.push(uvs.buffer);
            pw.worker.postMessage({ id, positions: worldPos, indices, uvs, texIndices: null, radius, mode: "dense", surfaceOnly: false, ox: 0, oy: 0, oz: 0 }, transfer);
            return;
        }
        // No worker available — sample synchronously on the main thread (blocks).
        const s = fillMeshParticles({ positions: worldPos, indices: g.indices, uvs, radius, mode: "dense" });
        if (!s.count) {
            abortSample(entry);
            return;
        }
        applySample(entry, { positions: s.positions, count: s.count, radius: s.radius, min: s.bounds.min, max: s.bounds.max, uvs: s.uvs });
    };

    function applySample(entry: PendingSample, sample: SampleResult): void {
        const { mesh, setting, settingName } = entry;
        samplingShots.delete(entry); // the ActiveSim below takes over the shake
        entry.group.sampling--;
        // eslint-disable-next-line no-console
        console.log(`[aquanova] liquefy — room: ${roomOfMesh(mesh)}, node: ${nodeNameOfMesh.get(mesh) ?? mesh.name} (mesh ${mesh.name}), particles: ${sample.count}, fluidSim: ${settingName ?? "(default)"}`);
        const { min, max } = sample;
        const cx = (min[0] + max[0]) / 2, cy = (min[1] + max[1]) / 2, cz = (min[2] + max[2]) / 2;
        const half = Math.max(max[0] - min[0], max[2] - min[2]) / 2 + SPREAD_MARGIN;
        const boundsMin: [number, number, number] = [cx - half, FLOOR_Y - 0.5, cz - half];
        const boundsMax: [number, number, number] = [cx + half, Math.max(max[1] + 3, CEIL_Y), cz + half];
        const dx = Math.max(sample.radius * 2.4, 0.18);
        const sim = buildFluidSim(setting, { count: sample.count, particleRadius: sample.radius, initialPositions: sample.positions, boundsMin, boundsMax, dx });
        sim.setSceneSdf(sceneSdf);
        // Per-particle mesh colours when the chosen setting asks for them (else the water stays uniform).
        // sample.positions (the seed) and worldPos (mesh vertices) share world space; g.uvs aligns with worldPos.
        const wantColor = !!setting?.useMeshColors;
        const texView = wantColor ? (mesh.material as unknown as { baseColorTexture?: { view?: GPUTextureView } } | undefined)?.baseColorTexture?.view : undefined;
        let colorBuffer = wantColor ? buildMeshColorBuffer(sample.count, sample.uvs, texView) : null;
        // Promote the albedo to the mesh's CURRENTLY RENDERED colours: the scene target still holds the
        // solid, lit and tone-mapped, so the water inherits how the prop actually looked rather than a
        // raw (unlit, far brighter) texture read. Hidden particles keep their albedo, rescaled to match.
        if (colorBuffer) {
            const ls = litScene();
            if (ls) {
                const lit = buildLitParticleColors(device, sample.count, sample.positions, colorBuffer, ls);
                colorBuffer.destroy();
                colorBuffer = lit;
            }
        }
        virtualSim.particleRadius = 0.05; // render impostor radius (thicker than the fine sample spacing → solid surface)
        virtualSim.surfaceSizeScale = sim.surfaceSizeScale ?? 1;

        // Explosion force field — PREPARED now, APPLIED at the end of the dissolve (finishShot) so the
        // water only erupts once the solid has fully melted.
        const impulseBuffer = device.createBuffer({ label: "aq-impulse", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const impulseSpec: ForceFieldSpec = { struct: "struct ForceFieldParams { center: vec4<f32>, push: vec4<f32>, };", wgsl: IMPULSE_WGSL, buffer: impulseBuffer };
        const volRadius = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2;
        device.queue.writeBuffer(impulseBuffer, 0, new Float32Array([cx, cy, cz, Math.max(volRadius * 2, sample.radius * 8, 1), IMPULSE_RADIAL_BASE, IMPULSE_UP_BASE, 0, 0]));

        // The Havok body stays in the world so the solid still collides with the player AND keeps
        // supporting any prop resting on top DURING the dissolve — it is only torn down at finishShot,
        // once the solid has actually melted. The display node was already decoupled (`wriggling`) and
        // its rest pose captured at request time, so the shake carries straight over.
        const dyn = entry.dyn;

        // The dissolve proper. The front grows from the group's melt ORIGIN — the weapon hit point,
        // shared by every mesh the cascade pulled in, so a door's two leaves burn from the same spot
        // instead of each starting at its own centre — falling back to this mesh's volume centre when
        // a programmatic fire supplied none. The sim is NOT stepped during this phase: the water stays
        // mesh-shaped and is revealed as the solid clips away.
        const state = liquefyStates.get(mesh) ?? null;
        const material = state ? mesh.material ?? null : null;
        const hitPoint = entry.hit;
        const hit: [number, number, number] = hitPoint ? [hitPoint[0]!, hitPoint[1]!, hitPoint[2]!] : [cx, cy, cz];
        const maxR = computeMaxR(hit, min, max);
        if (state) { state.hit = hit; state.frontR = 0; state.enabled = true; }
        if (material) bumpMat(material);
        const wriggleNode = entry.wriggleNode;
        const wriggleBase = entry.wriggleBase;
        const waterBase = new Float32Array(sample.count * 4);
        for (let i = 0; i < sample.count; i++) {
            waterBase[i * 4] = sample.positions[i * 3]!;
            waterBase[i * 4 + 1] = sample.positions[i * 3 + 1]!;
            waterBase[i * 4 + 2] = sample.positions[i * 3 + 2]!;
            waterBase[i * 4 + 3] = 1;
        }
        activeSims.push({
            sim, mesh, disp: dyn ? dyn.disp : null, dyn, phase: "dissolving",
            fluidElapsed: 0, fadeElapsed: 0, impulseRemaining: 0, impulseBuffer, impulseSpec,
            state, material, maxR, wriggleNode, wriggleBase, waterBase, waterScratch: new Float32Array(sample.count * 4),
            colorBuffer, useMeshColors: wantColor, group: entry.group, dissolved: false,
        });
        entry.group.sims.push(activeSims[activeSims.length - 1]!);
    }
    liquefyMesh = (mesh: Mesh, hitPoint?: readonly [number, number, number] | null): void => {
        // Melting a node melts it WHOLE: every primitive of that node goes at once (a node with 4
        // primitives is one object to the player). A liquefiable behaviour may also name LINKED
        // nodes that have to melt at the same moment (e.g. a door's two leaves); linked names are
        // matched SHIP-WIDE and the cascade is transitive, following each linked node's own links
        // and their primitives in turn.
        //
        // The group is enumerated FIRST, then dispatched, so the whole shot shares one decision:
        //   • one melt ORIGIN — the weapon hit point, or (for a programmatic fire with no pick
        //     point) the shot mesh's centre. Letting each member start at its own centre made a
        //     door's two leaves burn from two separate spots.
        //   • one fluid SETTING, drawn once from the shot mesh's candidates. Picking per mesh meant
        //     a 4-primitive pod could melt into four different fluids.
        // Enumeration is synchronous; only the sampling itself is async, so the members reach the
        // worker pool together and melt in step.
        const group: Mesh[] = [];
        const seen = new Set<Mesh>();
        const queue: Mesh[] = [mesh];
        while (queue.length) {
            const next = queue.shift()!;
            if (seen.has(next)) continue;
            if (next !== mesh && !dissolvableMeshes.has(next)) continue; // already melting / not dissolvable
            seen.add(next);
            group.push(next);
            for (const s of nodePrimitives.get(next) ?? []) queue.push(s);
            for (const name of linkedMeshNames(behaviorByMesh.get(next))) for (const other of meshesByNodeName.get(name) ?? []) queue.push(other);
        }

        const bMin = mesh.boundMin;
        const bMax = mesh.boundMax;
        const origin: readonly [number, number, number] | null = hitPoint
            ? [hitPoint[0]!, hitPoint[1]!, hitPoint[2]!]
            : bMin && bMax
              ? [(bMin[0]! + bMax[0]!) / 2, (bMin[1]! + bMax[1]!) / 2, (bMin[2]! + bMax[2]!) / 2]
              : null;
        // Candidates come from the SHOT mesh's behaviour (its own `fluidSim` list when it has one,
        // otherwise the manifest's global list); the winner then applies to every member.
        const ownSettings = behaviorByMesh.get(mesh)?.fluidSim;
        const candidates = ownSettings?.length ? ownSettings : fluidSettingNames;
        const settingName = candidates.length ? candidates[Math.floor(Math.random() * candidates.length)]! : undefined;

        const shot: ShotGroup = { sampling: 0, sims: [], excluded: excludedBodies(group), held: false };
        for (const m of group) requestSample(m, origin, settingName, shot);
    };

    // Floating-body indices named by `excludeSDF` across a shot's members. A door names its frame: the
    // water is seeded in the door's own volume, which sits INSIDE the frame, so leaving the frame in the
    // SDF union would eject the seeded particles and dam the opening the door just left behind.
    // Resolved before any member melts, while every body is still registered.
    function excludedBodies(group: readonly Mesh[]): number[] {
        const out = new Set<number>();
        for (const m of group) {
            for (const name of excludedSdfMeshNames(behaviorByMesh.get(m))) {
                for (const other of meshesByNodeName.get(name) ?? []) {
                    const dyn = dynBodyByMesh.get(other);
                    if (dyn) out.add(dyn.index);
                }
            }
        }
        return [...out];
    }

    // finishShot: this member's solid is fully melted → hide it, stop colliding, erupt its water.
    function finishShot(a: ActiveSim): void {
        a.wriggleNode.position.set(a.wriggleBase[0], a.wriggleBase[1], a.wriggleBase[2]);
        if (a.state) { a.state.enabled = false; a.state.frontR = a.maxR; }
        if (a.material) bumpMat(a.material);
        setMeshVisible(a.mesh, false);
        if (a.dyn) {
            // Solid has fully melted: NOW remove its Havok collider — any prop that was resting on it
            // drops into the erupting water (not before) — and park its SDF body far below so the
            // fluid no longer collides with the vanished solid.
            removePhysicsBody(world, a.dyn.body);
            dynBodyByMesh.delete(a.dyn.mesh);
            const di = dynBodies.indexOf(a.dyn);
            if (di >= 0) dynBodies.splice(di, 1);
            floatingSystem.setBodyPose(a.dyn.index, [0, -1e6, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]);
        }
        a.sim.setForceField(a.impulseSpec);
        a.impulseRemaining = 0.35;
        a.phase = "fluid";
        a.fluidElapsed = 0;
    }

    // A shot erupts as ONE event: nothing may start simulating until every member has been sampled
    // AND has finished melting. Without this a door's first leaf would burst while the second was
    // still dissolving.
    function eruptIfReady(group: ShotGroup): void {
        if (group.sampling > 0) return;
        for (const s of group.sims) if (s.phase === "dissolving" && !s.dissolved) return;
        // The sims start STEPPING now (they are frozen while dissolving), so this is the first moment
        // the scene SDF matters — and the last moment every body is still resolvable.
        holdSdfExclusions(group);
        for (const s of group.sims) if (s.phase === "dissolving") finishShot(s);
    }

    // Per-frame: advance each shot. DISSOLVING — wriggle the solid + water and grow the clip front
    // (the sim does NOT step; the water stays mesh-shaped and is revealed as the solid clips away; the
    // Havok body stays live so props on top keep their support); at full front → finishShot (hide the
    // solid, remove its Havok collider + park its SDF body, fire the impulse). FLUID/FADING — step the
    // sim, drive the impulse + lifetime, then fade out and dispose. Encoded BEFORE the surface task reads
    // combinedPos.
    onBeforeRender(scene, (deltaMs: number) => {
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        const growDt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        // Shots still waiting on their seed: shake the solid so the hit reads instantly. There is no
        // water to shake in lock-step yet — that starts with the ActiveSim below.
        for (const s of samplingShots) {
            const ox = (Math.random() * 2 - 1) * WRIGGLE_AMP, oy = (Math.random() * 2 - 1) * WRIGGLE_AMP, oz = (Math.random() * 2 - 1) * WRIGGLE_AMP;
            s.wriggleNode.position.set(s.wriggleBase[0] + ox, s.wriggleBase[1] + oy, s.wriggleBase[2] + oz);
        }
        canvas.dataset.sampling = String(samplingShots.size);
        if (colliderMode > 0) refreshColliders(); // dynamic proxies move; the chunk changes as you walk
        for (let k = activeSims.length - 1; k >= 0; k--) {
            const a = activeSims[k]!;
            if (a.phase === "dissolving") {
                // Pain-shake the solid + the mesh-shaped water in lock-step.
                const ox = (Math.random() * 2 - 1) * WRIGGLE_AMP, oy = (Math.random() * 2 - 1) * WRIGGLE_AMP, oz = (Math.random() * 2 - 1) * WRIGGLE_AMP;
                a.wriggleNode.position.set(a.wriggleBase[0] + ox, a.wriggleBase[1] + oy, a.wriggleBase[2] + oz);
                const wb = a.waterBase, ws = a.waterScratch;
                for (let i = 0; i < wb.length; i += 4) { ws[i] = wb[i]! + ox; ws[i + 1] = wb[i + 1]! + oy; ws[i + 2] = wb[i + 2]! + oz; ws[i + 3] = 1; }
                device.queue.writeBuffer(a.sim.positionBuffer, 0, ws);
                if (a.state) a.state.frontR = Math.min(a.state.frontR + LIQUEFY_SPEED * growDt, a.maxR);
                if (a.material) bumpMat(a.material);
                if (!a.dissolved && (!a.state || a.state.frontR >= a.maxR)) {
                    // This member's solid is gone, but the group erupts as one: hold here (fully
                    // clipped, water still shaking in lock-step) until every member has melted.
                    a.dissolved = true;
                    eruptIfReady(a.group);
                }
                continue; // do not step the sim while dissolving
            }
            if (a.phase === "fading") {
                a.fadeElapsed += dt;
                if (a.fadeElapsed >= FADE_DUR) {
                    // Dispose BEFORE stepping so we never free a sim's buffers after encoding its step.
                    a.sim.setForceField(null);
                    a.sim.dispose();
                    a.impulseBuffer.destroy();
                    a.colorBuffer?.destroy();
                    // Fully remove the (already-hidden) solid: a dynamic prop's display root takes its
                    // reparented mesh with it; a static prop is the mesh itself.
                    removeFromScene(scene, a.disp ?? a.mesh);
                    activeSims.splice(k, 1);
                    // Last of this shot's water gone → put any SDF bodies it excluded back in the union.
                    if (!activeSims.some((s) => s.group === a.group)) releaseSdfExclusions(a.group);
                    continue;
                }
            }
            a.sim.step(engine._currentEncoder, dt);
            if (a.impulseRemaining > 0) {
                a.impulseRemaining = Math.max(0, a.impulseRemaining - dt);
                if (a.impulseRemaining === 0) a.sim.setForceField(null);
            }
            if (a.phase === "fluid") {
                a.fluidElapsed += dt;
                if (a.fluidElapsed >= LIFETIME) {
                    a.phase = "fading";
                    a.fadeElapsed = 0;
                    a.impulseRemaining = 0;
                    a.sim.setForceField(null);
                }
            }
        }
        // Aggregate live particles into the shared surface buffer + per-particle alpha (fading ramps 1→0)
        // + per-particle colour (mesh-colour blobs copy their texture colours; others get the water tint).
        let off = 0;
        let anyColor = false;
        for (const a of activeSims) {
            const n = a.sim.count;
            if (off + n <= MAX_TOTAL) {
                engine._currentEncoder.copyBufferToBuffer(a.sim.positionBuffer, 0, combinedPos, off * 16, n * 16);
                if (a.useMeshColors && a.colorBuffer) {
                    engine._currentEncoder.copyBufferToBuffer(a.colorBuffer, 0, combinedColor, off * 16, n * 16);
                    anyColor = true;
                } else {
                    device.queue.writeBuffer(combinedColor, off * 16, colorScratch, 0, n * 4); // plain water tint
                }
                const alpha = a.phase === "fading" ? Math.max(0, 1 - a.fadeElapsed / FADE_DUR) : 1;
                alphaScratch.fill(alpha, off, off + n);
                off += n;
            }
        }
        surfaceTask.setUseParticleColor(anyColor); // shared toggle: on when any active blob carries mesh colours
        if (off > 0) {
            device.queue.writeBuffer(combinedAlpha, 0, alphaScratch, 0, off);
            virtualSim.count = off;
        } else {
            virtualSim.count = 0;
        }
        canvas.dataset.particleCount = String(off);
        canvas.dataset.activeSims = String(activeSims.length);
    });

    enableMaterialPlugins(scene);
    await registerScene(scene);
    await startEngine(engine);
    canvas.dataset.ready = "true";
}

main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error("[aquanova] fatal", err);
});
