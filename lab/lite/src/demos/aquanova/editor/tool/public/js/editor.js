// Viewport, grid, camera, selection and the placement store.
//
// There are no transform gizmos: authoring goes through the mouse-follow ghost
// in interact.js, so this module only owns scene setup and the data stores.

import { instantiate, getModule } from "./kit.js";

const {
  Engine, Scene, UniversalCamera, HemisphericLight, Vector3,
  Color3, Color4, Quaternion, Matrix, MeshBuilder,
  HDRCubeTexture, ImageProcessingConfiguration, PointerEventTypes,
} = BABYLON;

export const GRID_MAJOR = 4;      // kit tile size, metres
export const GRID_MINOR = 1;
const GRID_EXTENT = 120;

// The ship's own HDRI, served from the env folder. Named here rather than
// written out at each use, so there is one place to change it.
export const ENV_HDRI = "env/bank_vault_2k.hdr";
const DOUBLE_CLICK_MS = 320;
// Camera smoothing, fixed rather than exposed: it is a feel setting, not a
// legibility one.
const CAMERA_INERTIA = 0.75;
// Longest left press that still counts as a click. Anything held longer is
// treated as a camera gesture and selects nothing, so the left button can be
// used to swing the view without picking things up on release.
export const CLICK_MS = 300;

/**
 * The chunk staged elements are parked in.
 *
 * Never in `state.chunks`, so it cannot be picked, isolated or assigned to, and
 * every `p.chunk === id` loop skips it for free. `shipPlacements()` is the
 * belt to that braces: the handful of loops that walk *every* placement - the
 * manifest's instance list and the .glb exporter - must use it, or the staging
 * area would end up in the ship.
 */
export const STAGE_CHUNK = "__collision_stage";

/** Every placement that is part of the ship, excluding the staging area. */
export function shipPlacements() {
  return [...state.placements.values()].filter((p) => !p.stage);
}

/**
 * Ship-wide constants the layout carries with it.
 *
 * These are authoring decisions, not preferences: a ship fitted with a 8 mm
 * collision shell and reloaded on another machine has to come back with the
 * same shell, or its collision would silently change. So they live in the
 * layout, go through the undo stack, and are editable in the Settings pane
 * rather than being buried as literals in the code.
 */
export const CONFIG_DEFAULTS = {
  // Thickness given to a module with no depth of its own when fitting
  // collision - the kit's floors and ceilings are single planes. Matches the
  // 8 mm the kit's walls actually measure, so a room comes out uniform.
  shellThickness: 0.008,
  // How often the editor writes a recovery copy, in minutes. 0 turns it off.
  autoSaveMinutes: 2,
};

/**
 * What each setting will accept.
 *
 * A thickness of zero is a shape with no shape; an auto-save interval of zero
 * is a perfectly reasonable "don't". One rule for both would have to be wrong
 * for one of them.
 */
const CONFIG_RANGE = {
  shellThickness: { min: 1e-4, max: 10 },
  autoSaveMinutes: { min: 0, max: 240 },
};

export const state = {
  scene: null,
  engine: null,
  camera: null,
  placements: new Map(),   // id -> { id, module, chunk, node }
  markers: new Map(),      // id -> door / spawn marker (see markers.js)
  colliders: new Map(),    // id -> collision primitive (see colliders.js)
  chunks: ["CH00_Storage"],
  activeChunk: "CH00_Storage",
  selection: [],
  brush: null,             // module id armed for placement
  markerBrush: null,       // "door" - the only marker kind left
  snap: { pos: 1, rot: 90, scale: 0.1 },
  gridY: 0,                // elevation of the build plane
  rotAxis: "y",            // axis Q/E turn around
  scaleAxis: "all",        // axis Ctrl+wheel scales
  isolate: false,
  hidden: new Map(),       // id -> "ghost" | "hidden", see hideSelected
  veilAlpha: 0.5,          // how see-through a ghosted element is
  behaviors: new Map(),    // behaviour name -> definition body, see setBehaviorDef
  entities: new Map(),     // node name -> [{ name, linked: [] }]
  fluidSim: [],            // the global sim list from config.json
  unlit: false,            // show raw albedo, no lighting
  runtimeLight: false,     // drop the authoring rig, see what the game sees
  toneMapping: "Khronos PBR Neutral",
  // Two independent pairs. The editor's rig adds four analytic lights the game
  // does not have, so one pair of Env/Exposure values cannot serve both: what
  // reads well while building is nothing like what the game needs. The active
  // pair is mirrored into envIntensity/exposure above, which is what the scene
  // actually gets.
  lightSets: {
    editor: { strength: 1.5, exposure: 0.55 },
    runtime: { strength: 1.5, exposure: 0.55 },
  },
  exposure: 0.55,          // see EXPOSURE_DEFAULT
  envIntensity: 1.5,       // see ENV_INTENSITY_DEFAULT
  walk: false,             // see setWalk
  selectMode: false,       // LMB draws a selection rectangle, see setSelectMode
  moveSpeed: 42,           // m/s; right button + wheel adjusts it
  dragAxis: "xz",          // "xz" | "y" | "x" | "z" - which axis a move runs on (V)
  moveSpace: "world",      // "world" | "local" - whose axis that is (Y)
  collisionMode: false,    // the collision staging area is open, see colliders.js
  // module id -> shapes authored on it, in the module's own local space. The
  // one authoritative record: what is on the staging area is a working copy.
  moduleCollision: new Map(),
  // What was on the staging area when it was last closed, so re-opening it
  // finds the same modules in the same places rather than a blank stage.
  stageLayout: [],
  showLayer: "both",       // "both" | "geometry" | "collision", see setShowLayer
  config: { ...CONFIG_DEFAULTS },
  nextId: 1,
};

/** Selection holds ids from any store; resolve without caring which. */
export function entryOf(id) {
  return state.placements.get(id) || state.markers.get(id)
    || state.colliders.get(id) || null;
}

const listeners = new Map();
export function on(evt, fn) {
  if (!listeners.has(evt)) listeners.set(evt, []);
  listeners.get(evt).push(fn);
}
export function emit(evt, payload) {
  for (const fn of listeners.get(evt) || []) fn(payload);
}

let gridNode = null;
let undoStack = [];
let redoStack = [];
// The staging area's own history, so undoing a box there does not restore a
// ship snapshot that knows nothing about the bench - see pushUndo.
let stageUndo = [];
let stageRedo = [];

// ------------------------------------------------------------------ setup

export async function initScene(canvas) {
  const engine = new Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
  const scene = new Scene(engine);
  scene.clearColor = new Color4(0.055, 0.063, 0.075, 1);
  // Babylon rewrites canvas.style.cursor on every pointer move, which would
  // undo hiding the cursor while a ghost is being placed.
  scene.doNotHandleCursors = true;
  window.__scene = scene;
  state.engine = engine;
  state.scene = scene;

  // One free-flying camera. An arc-rotate camera is always tethered to a pivot,
  // which fights you the moment you want to walk down a corridor.
  const cam = new UniversalCamera("cam", new Vector3(-6, 9, -14), scene);
  cam.setTarget(new Vector3(6, 1.5, 0));
  cam.minZ = 0.05;
  cam.maxZ = 800;
  cam.angularSensibility = 2600;
  // movement is driven per-frame below, so the built-in key bindings are off
  cam.keysUp = []; cam.keysDown = [];
  cam.keysLeft = []; cam.keysRight = [];
  cam.keysUpward = []; cam.keysDownward = [];
  // setupPointer registers first on purpose: it needs to be able to swallow a
  // pointer-down that starts a drag before the camera turns it into a look
  setupPointer(scene);
  cam.attachControl(canvas, true);
  // The camera looks on the *right* button only. Left is reserved entirely for
  // editing - select, drag, place - so a mis-aimed edit can never swing the
  // view, and a look can never grab a module.
  const mouseInput = cam.inputs.attached.mouse;
  if (mouseInput) mouseInput.buttons = [2];
  state.camera = cam;
  scene.activeCamera = cam;

  // Camera smoothing. Fixed rather than authorable: it is a feel setting, not a
  // legibility one, and 0.75 is the value that stuck.
  cam.inertia = CAMERA_INERTIA;

  // Readability first: this is a layout tool, so the lighting exists only so
  // that every module is legible from any angle. It is deliberately fixed and
  // not authorable - the real lighting lives in the render/runtime pipeline.
  const hemi = new HemisphericLight("hemi", new Vector3(0.3, 1, 0.2), scene);
  hemi.intensity = 0.75;
  hemi.groundColor = new Color3(0.20, 0.23, 0.28);
  // A hemisphere aimed the other way, to light what the first one cannot.
  // Babylon's hemi direction points at its "sky", so the main one only ever
  // gives a downward-facing surface its (dark) ground colour - which is every
  // ceiling panel, pipe run and platform underside in the kit. Its own ground
  // colour is black so it does not double-light the floors.
  const hemiUp = new HemisphericLight("hemiUp", new Vector3(-0.2, -1, -0.15), scene);
  hemiUp.intensity = 0.55;
  hemiUp.groundColor = new Color3(0, 0, 0);
  const key = new BABYLON.DirectionalLight("key", new Vector3(-0.5, -1, 0.65), scene);
  key.intensity = 1.1;
  const fill = new BABYLON.DirectionalLight("fill", new Vector3(0.7, -0.35, -0.6), scene);
  fill.intensity = 0.4;
  // Remembered so the rig can be switched off to preview the runtime, and back
  // on again at exactly the values it was authored with.
  for (const l of [hemi, hemiUp, key, fill]) authoredIntensity.set(l, l.intensity);
  applyRuntimeLighting();

  scene.imageProcessingConfiguration.toneMappingEnabled = true;
  scene.imageProcessingConfiguration.toneMappingType = resolveToneMapping(state.toneMapping);
  scene.imageProcessingConfiguration.exposure = state.exposure;

  // Metals read as black without an IBL. The ship's own HDRI is a dark vault,
  // so it is used for reflections only and pushed above its render value.
  try {
    const env = new HDRCubeTexture(`/${ENV_HDRI}`, scene, 128, false, true, false, true);
    scene.environmentTexture = env;
    scene.environmentIntensity = state.envIntensity;
  } catch (e) {
    console.warn("no HDRI, metals will look flat", e);
  }

  // Every material in the editor is double-sided. Kit modules are mostly
  // authored that way already, but the ghost's material copies and the editor
  // helpers are not, and a single-sided wall turned away from the camera simply
  // vanishes - fine in game, useless while building.
  scene.onNewMaterialAddedObservable.add(makeTwoSided);
  scene.materials.forEach(makeTwoSided);
  // The observable can fire from the base Material constructor, before a
  // subclass has applied its own backFaceCulling default, so re-sweep whenever
  // the material count changes. Cheap: it only runs when one is added.
  let lastMaterialCount = scene.materials.length;
  scene.onBeforeRenderObservable.add(() => {
    if (scene.materials.length === lastMaterialCount) return;
    lastMaterialCount = scene.materials.length;
    scene.materials.forEach(makeTwoSided);
  });

  buildGrid(scene);
  setupCameraMove(scene);

  engine.runRenderLoop(() => scene.render());
  window.addEventListener("resize", () => engine.resize());
  return scene;
}

// The kit ships almost everything doubleSided already, but the editor forces it
// on regardless. Remember what each material was authored as so the exporter
// can hand the runtime exactly what the kit shipped.
const authoredCulling = new WeakMap();
const authoredEmissive = new WeakMap();

// Unlit shows raw albedo, and a good part of the kit is painted very dark, so a
// flat emissive lift is added to keep those modules readable. Emissive is
// additive, which is what lifts a near-black texture - scaling the albedo would
// leave it near-black.
export const UNLIT_LIFT = 0.16;

/**
 * Viewport exposure.
 *
 * The pale kit panels (`MI_Trim_03` - Platform_Simple, Door_Simple) are bright
 * dielectrics, and at exposure 1.0 they render around 240/255: deep in the
 * KHR PBR Neutral highlight shoulder, where the curve is almost flat and all
 * the surface detail - dirt, seams, bolt strips - is compressed out of
 * existence. Pulling exposure down moves them back onto the straight part of
 * the curve and the detail returns; measured pixel contrast over such a panel
 * roughly doubles between 1.0 and 0.5.
 *
 * This is the same effect that made a dragged ghost look sharper than the same
 * module once placed: the ghost's 0.7 alpha was quietly acting as an exposure
 * cut. Adjustable, because lower exposure trades away brightness on genuinely
 * dark props.
 */
export const EXPOSURE_DEFAULT = 0.55;

/**
 * Strength of the image-based lighting.
 *
 * Metals have no diffuse term, so the analytic lights barely touch them and
 * almost all their brightness comes from here - this is the lever for reading
 * metal ceilings and platform undersides, which no hemisphere can help.
 */
export const ENV_INTENSITY_DEFAULT = 1.5;

// ------------------------------------------------------------ runtime light
//
// The four analytic lights are an authoring aid: they exist so every module
// stays legible from any angle while you build. The game has none of them - it
// lights the ship from the HDRI and its own emissive fixtures - so they are
// measurably the reason the editor looks nothing like the runtime. Measured on
// the real ship: mean 165.7 with the rig, 35.0 without, so the rig supplies
// about four fifths of what you see here and none of what you will ship.
//
// Turning them off is therefore the only honest preview, and the only sound way
// to judge the Env and Exposure sliders, which *are* written to the manifest.

const authoredIntensity = new WeakMap();

function applyRuntimeLighting() {
  if (!state.scene) return;
  for (const l of state.scene.lights) {
    if (!authoredIntensity.has(l)) continue;
    l.intensity = state.runtimeLight ? 0 : authoredIntensity.get(l);
  }
}

export function setRuntimeLighting(on) {
  state.runtimeLight = !!on;
  applyRuntimeLighting();
  // Swap in the pair that belongs to this mode. Both are kept whole, so
  // switching back and forth never costs you the values you tuned.
  const set = activeLightSet();
  setEnvIntensity(set.strength);
  setExposure(set.exposure);
  emit("modes");
  return state.runtimeLight;
}

export function setEnvIntensity(v) {
  const n = Number(v);
  state.envIntensity = Number.isFinite(n)
    ? Math.min(6, Math.max(0, n))
    : ENV_INTENSITY_DEFAULT;
  activeLightSet().strength = state.envIntensity;
  if (state.scene) state.scene.environmentIntensity = state.envIntensity;
  emit("modes");
}

export function setExposure(v) {
  const n = Number(v);
  // Number.isFinite, not `|| DEFAULT`: a literal 0 is falsy and would silently
  // jump back to the default instead of clamping to the floor.
  state.exposure = Number.isFinite(n)
    ? Math.min(2, Math.max(0.15, n))
    : EXPOSURE_DEFAULT;
  activeLightSet().exposure = state.exposure;
  if (state.scene) state.scene.imageProcessingConfiguration.exposure = state.exposure;
  emit("modes");
}

/** The Env/Exposure pair the sliders are currently editing. */
export function activeLightSet() {
  return state.lightSets[state.runtimeLight ? "runtime" : "editor"];
}

/**
 * Tone mapping by name, matched the way the runtime matches it - loosely, so
 * "Khronos PBR Neutral", "neutral" and "KHR_PBR_NEUTRAL" all land in the same
 * place. Returning null means "no tone mapping at all", which is a real answer
 * and not a failure.
 */
export function resolveToneMapping(name) {
  const key = String(name ?? "").toLowerCase().replace(/[^a-z]/g, "");
  const IPC = ImageProcessingConfiguration;
  if (!key) return IPC.TONEMAPPING_KHR_PBR_NEUTRAL;
  if (key.includes("none") || key.includes("off") || key.includes("linear")) return null;
  if (key.includes("aces")) return IPC.TONEMAPPING_ACES;
  if (key.includes("standard") || key.includes("exponential")) return IPC.TONEMAPPING_STANDARD;
  if (key.includes("neutral")) return IPC.TONEMAPPING_KHR_PBR_NEUTRAL;
  console.warn(`unknown toneMapping "${name}" — using Khronos PBR Neutral`);
  return IPC.TONEMAPPING_KHR_PBR_NEUTRAL;
}

/**
 * Apply the view transform the manifest names, rather than assuming one.
 * The demos read the same field, so the editor hard-coding it was the one way
 * the two could still disagree after the exposure was made literal.
 */
export function setToneMapping(name) {
  state.toneMapping = String(name || "Khronos PBR Neutral");
  if (!state.scene) return state.toneMapping;
  const tone = resolveToneMapping(state.toneMapping);
  const ipc = state.scene.imageProcessingConfiguration;
  ipc.toneMappingEnabled = tone !== null;
  if (tone !== null) ipc.toneMappingType = tone;
  emit("modes");
  return state.toneMapping;
}

function makeTwoSided(mat) {
  if (!authoredCulling.has(mat)) authoredCulling.set(mat, mat.backFaceCulling);
  mat.backFaceCulling = false;
  if ("twoSidedLighting" in mat) mat.twoSidedLighting = true;
  applyViewportMode(mat);
}

/**
 * Record a material's authored emissive so unlit mode can lift it and put it
 * back. Kit materials get their emissive set *after* construction, so the kit
 * loader re-notes with force once its own values are in place.
 */
export function noteAuthoredEmissive(mat, force = false) {
  if (!("unlit" in mat)) return;
  if (!force && authoredEmissive.has(mat)) return;
  authoredEmissive.set(mat, {
    color: mat.emissiveColor ? mat.emissiveColor.clone() : null,
    intensity: mat.emissiveIntensity,
  });
}

/** Put one material into the current viewport mode. */
export function applyViewportMode(mat) {
  if (!("unlit" in mat)) return;
  mat.unlit = state.unlit;
  noteAuthoredEmissive(mat);
  const authored = authoredEmissive.get(mat);

  // Materials carrying an emissive texture are the light strips: their emissive
  // colour multiplies that texture, so overwriting it would break them rather
  // than lift anything.
  if (state.unlit && !mat.emissiveTexture) {
    mat.emissiveColor = new Color3(UNLIT_LIFT, UNLIT_LIFT, UNLIT_LIFT);
    mat.emissiveIntensity = 1;
  } else if (authored?.color) {
    mat.emissiveColor = authored.color.clone();
    mat.emissiveIntensity = authored.intensity;
  }
}

/**
 * Show raw albedo with no lighting at all. Much of the kit is fully metallic,
 * which means no diffuse term: under a dim IBL those modules collapse to a flat
 * wash and you cannot read the texture. Unlit shows what the artist painted.
 */
export function setUnlit(on) {
  state.unlit = !!on;
  for (const mat of state.scene.materials) applyViewportMode(mat);
  emit("modes");
}

/**
 * A translucent stand-in for a material, cached.
 *
 * PBRMaterial.clone() re-creates every texture, so a naive variant would
 * duplicate the kit's 2-4 MB atlases; the copies are re-pointed at the
 * originals and the duplicates thrown away.
 *
 * Never mutate the original instead: kit materials are shared across modules,
 * so turning one translucent turns off depth writes for everything drawn with
 * it - which is exactly the bug this replaced.
 *
 * Keyed by source material *and* purpose: the placement ghost and the veil want
 * different alphas and are often on screen together.
 */
const TEXTURE_SLOTS = [
  "albedoTexture", "diffuseTexture", "ambientTexture", "opacityTexture",
  "reflectionTexture", "emissiveTexture", "reflectivityTexture", "specularTexture",
  "metallicTexture", "microSurfaceTexture", "bumpTexture", "lightmapTexture",
  "refractionTexture", "metallicReflectanceTexture", "detailMap",
];
const ghostMats = new Map();
// Set while the glTF exporter walks the scene: the veil is a way of looking,
// and the file must be written from the ship as authored.
let veilSuspended = false;

export function ghostMaterialFor(mat, tag = "GHOST", alpha = 0.7) {
  if (!mat) return null;
  const key = `${mat.uniqueId}:${tag}`;
  let g = ghostMats.get(key);
  if (!g) {
    g = mat.clone(`${tag}_${mat.name}`);
    for (const slot of TEXTURE_SLOTS) {
      const orig = mat[slot];
      const dup = g[slot];
      if (orig && dup && dup !== orig) { dup.dispose(); g[slot] = orig; }
    }
    g.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
    // Never cull: a single-sided wall would otherwise vanish the moment it is
    // turned to face away from the camera.
    g.backFaceCulling = false;
    g.twoSidedLighting = true;
    g.forceDepthWrite = false;
    g.zOffset = -2;
    ghostMats.set(key, g);
  }
  g.alpha = alpha;
  return g;
}

/**
 * Run `fn` with materials back to what the kit authored: original culling, no
 * unlit flag and no emissive lift. All three would otherwise leak into the
 * glTF export - `unlit` as KHR_materials_unlit, `backFaceCulling` as
 * doubleSided, and the lift as a grey glow on every surface.
 */
export async function withAuthoredMaterials(fn) {
  const wasUnlit = state.unlit;
  state.unlit = false;
  // Every material must have its viewport mode cleared, not only the ones the
  // editor swept for culling - otherwise a material that skipped that sweep
  // would carry its emissive lift straight into the file.
  const mats = [...state.scene.materials];
  for (const m of mats) {
    if (authoredCulling.has(m)) m.backFaceCulling = authoredCulling.get(m);
    applyViewportMode(m);
  }
  try {
    return await fn();
  } finally {
    state.unlit = wasUnlit;
    for (const m of mats) {
      if (authoredCulling.has(m)) m.backFaceCulling = false;
      applyViewportMode(m);
    }
  }
}

// ------------------------------------------------------------- walk mode

/** Eye height of the player the ship is being built for. */
export const EYE_HEIGHT = 1.8;
// How far above the feet the ground ray starts. Anything lower than this is
// stepped onto rather than walked into, and starting *above* the eye instead
// would find the ceiling of the room you are standing in.
const STEP_UP = 0.6;
const FALL_LIMIT = 200;
// Entering walk mode searches from above everything instead, so it works even
// when the camera is parked below a floor or nowhere near one.
const WORLD_TOP = 500;

/**
 * Walk the camera at eye level instead of flying it.
 *
 * The point is to judge the ship the way it will actually be seen: corridor
 * heights, sight lines through doors and how much of a room is visible from the
 * floor are all things a free camera flatters.
 */
export function setWalk(on) {
  state.walk = !!on;
  if (state.walk) groundCamera(true);
  emit("modes");
}

/**
 * Rectangle-select mode: the left button drags out a marquee instead of moving
 * whatever it started on.
 *
 * A drag from empty space is a marquee either way - there is nothing else it
 * could mean. The mode exists for the case that *is* ambiguous: starting the
 * rectangle on top of a module, which would otherwise pick it up and move it.
 */
export function setSelectMode(on) {
  state.selectMode = !!on;
  emit("modes");
  return state.selectMode;
}

/**
 * Screen-space bounding box of a node, in canvas pixels.
 *
 * All eight corners of the world box are projected, not just min and max: a
 * rotated element's screen extent is not the projection of its world extent,
 * and using two corners would under-report it badly at 45 degrees.
 */
export function screenBoundsOf(node) {
  const b = worldBounds(node);
  if (!b) return null;
  const scene = state.scene;
  const engine = scene.getEngine();
  const vp = state.camera.viewport.toGlobal(engine.getRenderWidth(), engine.getRenderHeight());
  const view = scene.getTransformMatrix();
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < 8; i++) {
    const c = new Vector3(
      i & 1 ? b.max.x : b.min.x,
      i & 2 ? b.max.y : b.min.y,
      i & 4 ? b.max.z : b.min.z);
    const p = Vector3.Project(c, Matrix.Identity(), view, vp);
    // behind the camera projects to a mirrored point that would wreck the box
    if (p.z < 0 || p.z > 1) continue;
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  if (minX === Infinity) return null;
  return { minX, minY, maxX, maxY };
}

/**
 * Every element whose screen box overlaps the given canvas-space rect.
 *
 * Markers count: the doors are elements you select, drag and delete like any
 * other, so a rectangle that goes round one has to catch it. Anything currently
 * hidden - isolated away, or parked with `Shift+H` - is skipped, on the grounds
 * that a rubber band selects what you can see.
 */
export function elementsInRect(rect) {
  const out = [];
  const overlaps = (node) => {
    if (!node.isEnabled()) return false;
    const b = screenBoundsOf(node);
    if (!b) return false;
    return !(b.maxX < rect.x0 || b.minX > rect.x1 || b.maxY < rect.y0 || b.minY > rect.y1);
  };
  for (const e of state.placements.values()) if (overlaps(e.node)) out.push(e.id);
  for (const m of state.markers.values()) if (overlaps(m.node)) out.push(m.id);
  for (const c of state.colliders.values()) if (overlaps(c.node)) out.push(c.id);
  return out;
}

/** Height of the walkable surface under `x, z`, or null if there is none. */
export function groundHeightAt(x, z, fromY, reach = FALL_LIMIT) {
  const scene = state.scene;
  if (!scene) return null;
  const ray = new BABYLON.Ray(new Vector3(x, fromY, z), Vector3.Down(), reach);
  const hit = scene.pickWithRay(ray, (m) =>
    m.isPickable && m.isEnabled() && !!m.metadata?.placementRoot);
  return hit?.hit ? hit.pickedPoint.y : null;
}

/**
 * Put the camera at eye height over whatever is underfoot.
 *
 * The two cases want different searches. **Entering** walk mode has to land you
 * somewhere no matter where the camera was left - below a floor, high above one,
 * or over a gap - so it searches from above the whole world and falls back to
 * the build plane. **While walking**, a miss means you have stepped over a hole,
 * and a half-built ship is full of them: keep the current height rather than
 * dropping out of the world.
 */
function groundCamera(entering = false) {
  const cam = state.camera;
  if (!cam) return;
  const { x, z } = cam.position;
  const feet = cam.position.y - EYE_HEIGHT;

  let y = groundHeightAt(x, z, feet + STEP_UP);
  if (y === null && entering) {
    y = groundHeightAt(x, z, WORLD_TOP, WORLD_TOP * 2);
    if (y === null) y = state.gridY;
  }
  if (y === null) return;

  cam.position.y = y + EYE_HEIGHT;
  // Vertical inertia left over from flying would fight the clamp every frame.
  cam.cameraDirection.y = 0;
}

function buildGrid(scene) {
  const minor = [], major = [], axis = [];
  for (let i = -GRID_EXTENT; i <= GRID_EXTENT; i += GRID_MINOR) {
    const bucket = i === 0 ? axis : (i % GRID_MAJOR === 0 ? major : minor);
    bucket.push([new Vector3(i, 0, -GRID_EXTENT), new Vector3(i, 0, GRID_EXTENT)]);
    bucket.push([new Vector3(-GRID_EXTENT, 0, i), new Vector3(GRID_EXTENT, 0, i)]);
  }
  gridNode = new BABYLON.TransformNode("GRID", scene);
  const mk = (name, lines, color) => {
    const ls = MeshBuilder.CreateLineSystem(name, { lines }, scene);
    ls.color = color;
    ls.isPickable = false;
    ls.parent = gridNode;
    ls.doNotSyncBoundingInfo = true;
    return ls;
  };
  mk("grid_minor", minor, new Color3(0.13, 0.15, 0.18));
  mk("grid_major", major, new Color3(0.26, 0.30, 0.36));
  mk("grid_axis", axis, new Color3(0.55, 0.36, 0.18));
}

export function setGridVisible(v) { gridNode.setEnabled(v); }

// -------------------------------------------------------------------- axes
//
// A world-space axis gizmo for one element, toggled with X. World, not local:
// a drag moves an element along the world axes - `node.position` is world
// position, since placements have no parent in the editor - so a gizmo drawn
// along the element's own axes would point one way while the drag went another.
//
// It carries all three modal axis settings at once, so "what will the next key
// do to this element" is one glance rather than three readouts:
//
//   bright arrow    the axes a drag moves along     (V)
//   curved arrow    the axis a turn goes about      (Shift+R)
//   cube on the tip the axis a scale acts on        (Shift+F)
//
// and two DOM chips carry the *values*: the move step at the origin, the turn
// angle inside the curved arrow.

export const AXIS_COLOR = {
  x: new Color3(0.90, 0.22, 0.27),
  y: new Color3(0.36, 0.78, 0.34),
  z: new Color3(0.26, 0.52, 0.96),
};

/** Which axes a drag currently moves along. */
export function liveAxes(mode = state.dragAxis) {
  return mode === "y" ? ["y"] : mode === "x" ? ["x"] : mode === "z" ? ["z"] : ["x", "z"];
}

/**
 * The frame a move runs in: the world's axes, or the given element's own.
 *
 * `moveSpace` says whose axes the *move axis* means. In world space "X only"
 * slides along world X; in local space it slides along the element's own X, so
 * a wall turned 90 degrees still slides along its length rather than across it.
 * Null means the world, which every caller treats as "no basis at all".
 *
 * The axes come from the world matrix's normalised rows, not from the rotation
 * quaternion, so they are the axes Shift+X actually draws - mirroring included.
 * A mirrored element's basis is left-handed, which is fine here: each axis is
 * used on its own.
 *
 * Whose element is always the caller's to decide, and it is always the one
 * being *acted on*: the piece under the cursor for a drag, the anchor of a
 * carry, the first of the selection for an arrow nudge - which is also the one
 * `X` puts its gizmo on. With several selected they all move by one delta
 * measured in that element's frame, the way a set of objects moves in Blender.
 */
export function moveBasis(node) {
  if (state.moveSpace !== "local" || !node || node.isDisposed()) return null;
  // Forced, not read from the cache: the frame is taken once at the start of a
  // gesture, and a turn earlier in the same frame - R, the inspector, a load -
  // has not been through a render yet, so the cached matrix still holds the
  // rotation before it. That put the axes one turn behind.
  const m = node.computeWorldMatrix(true);
  const out = {};
  for (const [a, row] of [["x", 0], ["y", 1], ["z", 2]]) {
    const r = m.getRow(row);
    const v = new Vector3(r.x, r.y, r.z);
    if (v.lengthSquared() < 1e-12) return null;    // degenerate, fall back to world
    out[a] = v.normalize();
  }
  return out;
}

/**
 * Take a world-space movement and let through only what the current axis and
 * the given frame allow, snapping along each axis that survives.
 *
 * World space is the same expression with the world's own axes, so both modes
 * run one code path: project onto each live axis, snap that distance, and add
 * the axis back scaled by it.
 */
export function constrainMove(delta, snap = (v) => v, basis = null) {
  const out = Vector3.Zero();
  for (const a of liveAxes()) {
    if (basis) out.addInPlace(basis[a].scale(snap(Vector3.Dot(delta, basis[a]))));
    else out[a] = snap(delta[a]);
  }
  return out;
}

/** Which axes a scale acts on - "all" means every one of them. */
export function scaleAxes(mode = state.scaleAxis) {
  return mode === "all" ? ["x", "y", "z"] : [mode];
}

let axes = null;      // { root, id, space, arms, mats, marks, observer }

/** The element currently showing its axes, or null. */
export function axesTarget() { return axes?.id || null; }

/** Which space the axes are drawn in - "world" or "local". */
export function axesSpace() { return axes?.space || null; }

/** Whether the turn arrow is drawn reversed, i.e. the step is negative. */
export function axesRotArrowFlipped() {
  const ring = axes?.marks.rot[state.rotAxis];
  return ring ? ring.scaling.x < 0 : null;
}

/** The id used for the armed ghost, which is not a placement and has no id. */
export const GHOST_AXES = "__ghost__";

/**
 * The node the axes should follow.
 *
 * The ghost lives in interact.js, which imports this module, so it registers
 * itself through `hooks` rather than being imported back - the same way markers
 * do.
 */
function axesNode(id) {
  if (id === GHOST_AXES) return hooks.ghostNode() || null;
  // entryOf covers placements, markers *and* collision shapes. Looking in two
  // of the three by hand is what stopped X working on a collision shape - the
  // same omission that once made them unpickable.
  const e = entryOf(id);
  return e && !e.node.isDisposed() && e.node.isEnabled() ? e.node : null;
}

function buildAxes(scene, length) {
  const root = new BABYLON.TransformNode("AXES", scene);
  const arms = {};
  const mats = {};
  const marks = { rot: {}, scale: {} };
  // Every thickness on the gizmo - shaft, arrowhead, the turn ring, the scale
  // cube - is a multiple of this one radius, and every *length* is a multiple
  // of `length`. So the whole thing is made slimmer or fatter here without
  // touching how far the arms reach.
  const r = Math.max(0.006, length * 0.01);
  // Geometry is authored along +Z once; each arm is simply turned to face its
  // own world axis, and never moves again - the axes are world-aligned.
  const facing = { x: [0, Math.PI / 2, 0], y: [-Math.PI / 2, 0, 0], z: [0, 0, 0] };

  for (const a of ["x", "y", "z"]) {
    const arm = new BABYLON.TransformNode(`AXES_${a}`, scene);
    arm.parent = root;
    arm.rotation.set(...facing[a]);

    // Standard, not PBR: applyViewportMode() only touches materials with an
    // `unlit` property, so a gizmo built from these can never pick up the
    // editor's unlit mode or its emissive lift.
    const mk = (suffix, dim) => {
      const m = new BABYLON.StandardMaterial(`AXES_MAT_${a}${suffix}`, scene);
      m.disableLighting = true;
      m.diffuseColor = new Color3(0, 0, 0);
      m.emissiveColor = AXIS_COLOR[a].clone();
      return m;
    };
    mats[a] = mk("");
    // The markers keep full brightness even on a dimmed arm: the scale axis is
    // often not one of the drag axes, and a dim square would read as "off".
    mats[`${a}_mark`] = mk("_mark");

    const shaft = MeshBuilder.CreateCylinder(`AXES_${a}_shaft`,
      { height: length * 0.82, diameter: r * 2, tessellation: 10 }, scene);
    shaft.position.z = length * 0.41;
    const head = MeshBuilder.CreateCylinder(`AXES_${a}_head`,
      { height: length * 0.18, diameterTop: 0, diameterBottom: r * 5, tessellation: 12 }, scene);
    head.position.z = length * 0.91;
    // "a turn goes about this one": a curved arrow encircling the shaft, the
    // universal reading for rotation - and unmistakable against the straight
    // arrows it sits on.
    const rotMark = buildRotationArrow(scene, `AXES_${a}_rot`, length, r);
    // "a scale acts on this one": a cube sitting on the very tip
    const scaleMark = MeshBuilder.CreateBox(`AXES_${a}_scale`,
      { size: r * 5.5 }, scene);
    scaleMark.position.z = length * 1.04;

    for (const m of [shaft, head, scaleMark]) {
      m.rotation.x = Math.PI / 2;      // Babylon builds cylinders along +Y
      m.material = m === scaleMark ? mats[`${a}_mark`] : mats[a];
      m.parent = arm;
      m.isPickable = false;
      m.doNotSyncBoundingInfo = true;
      // draw over the ship: an arrow buried inside the element it belongs to
      // would be exactly as useful as no arrow
      m.renderingGroupId = 1;
    }
    for (const m of rotMark.getChildMeshes()) {
      m.material = mats[`${a}_mark`];
      m.isPickable = false;
      m.doNotSyncBoundingInfo = true;
      m.renderingGroupId = 1;
    }
    rotMark.parent = arm;
    marks.rot[a] = rotMark;
    marks.scale[a] = scaleMark;
    arms[a] = arm;
  }
  return { root, arms, mats, marks };
}

/**
 * A curved arrow encircling the arm's own +Z, as one node holding a tube and a
 * cone. Three quarters of a turn rather than a full ring, so it reads as a
 * direction of travel and not as a collar.
 */
function buildRotationArrow(scene, name, length, r) {
  const node = new BABYLON.TransformNode(name, scene);
  const radius = length * 0.12;
  const sweep = Math.PI * 1.5;
  const steps = 24;
  // The node sits *at* the arc's centre, with the geometry built around the
  // origin, rather than at the arm's root with the arc pushed out along Z. That
  // makes getAbsolutePosition() the ring's own position, which is where the
  // angle readout has to go - otherwise it lands on the gizmo origin and sits
  // on top of the move-step chip.
  node.position.z = length * 0.46;

  const path = [];
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * sweep;
    path.push(new Vector3(radius * Math.cos(t), radius * Math.sin(t), 0));
  }
  const tube = MeshBuilder.CreateTube(`${name}_arc`,
    { path, radius: r * 0.45, tessellation: 8, cap: BABYLON.Mesh.CAP_ALL }, scene);
  tube.parent = node;

  // The head goes on the end of the arc, pointing along the tangent there.
  const head = MeshBuilder.CreateCylinder(`${name}_head`,
    { height: length * 0.065, diameterTop: 0, diameterBottom: r * 2, tessellation: 12 }, scene);
  head.position.copyFrom(path[path.length - 1]);
  const tangent = new Vector3(-Math.sin(sweep), Math.cos(sweep), 0).normalize();
  // Cones are built along +Y; turn that onto the tangent. Done by hand rather
  // than with lookAt(), which aims +Z and would need a second correction.
  const axis = Vector3.Cross(Vector3.Up(), tangent);
  head.rotationQuaternion = axis.lengthSquared() < 1e-12
    ? Quaternion.Identity()
    : Quaternion.RotationAxis(axis.normalize(),
      Math.acos(Math.min(1, Math.max(-1, Vector3.Dot(Vector3.Up(), tangent)))));
  head.parent = node;
  return node;
}

/** Repaint from the modal state: which axes drag, turn and scale. */
function paintAxes() {
  if (!axes) return;
  const live = new Set(liveAxes());
  const scaled = new Set(scaleAxes());
  for (const a of ["x", "y", "z"]) {
    const on = live.has(a);
    const c = AXIS_COLOR[a];
    axes.mats[a].emissiveColor.copyFromFloats(
      on ? c.r : c.r * 0.3, on ? c.g : c.g * 0.3, on ? c.b : c.b * 0.3);
    axes.mats[a].alpha = on ? 1 : 0.45;
    axes.marks.rot[a].setEnabled(a === state.rotAxis);
    axes.marks.scale[a].setEnabled(scaled.has(a));
  }
}

/**
 * Point the arms down the element's own axes.
 *
 * Scaling is **local**, so a world-aligned gizmo cannot answer "which way does
 * X grow?" for anything that has been turned - which is most of a ship built
 * from a modular kit. The arms are aimed individually from the world matrix's
 * basis rows rather than by rotating the whole gizmo, because a mirrored
 * element (negative scale) has no rotation that expresses it: its local +X
 * genuinely points the other way, and each arm can simply be turned round.
 *
 * Length is normalised out - the gizmo is a direction indicator, and a 3x
 * scaled element should not get 3x arrows.
 */
function orientAxes(node) {
  if (!axes || axes.space !== "local") return;
  const m = node.getWorldMatrix();
  const rows = { x: m.getRow(0), y: m.getRow(1), z: m.getRow(2) };
  const basis = Matrix.Identity();
  for (const a of ["x", "y", "z"]) {
    const r = rows[a];
    const d = new Vector3(r.x, r.y, r.z);
    if (d.lengthSquared() < 1e-12) continue;
    d.normalize();
    // The basis is built by hand rather than with Quaternion.FromLookDirectionLH,
    // which stores **-forward** in its Z row - a camera "back" convention that
    // aims the arm 180 degrees the wrong way.
    const up = Math.abs(d.y) > 0.99 ? Vector3.Forward() : Vector3.Up();
    const xAxis = Vector3.Cross(up, d).normalize();
    const yAxis = Vector3.Cross(d, xAxis).normalize();
    Matrix.FromXYZAxesToRef(xAxis, yAxis, d, basis);
    axes.arms[a].rotationQuaternion = Quaternion.FromRotationMatrix(basis);
  }
}

export function showAxes(id, space = "world") {
  const node = axesNode(id);
  if (!node) return false;
  hideAxes();

  const b = worldBounds(node);
  const span = b ? Math.max(b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z) : 2;
  const built = buildAxes(state.scene, Math.min(8, Math.max(1.5, span * 0.9)));

  // The move step decides how far every nudge and drag of this element will
  // jump, the turn angle how far one R will swing it, and the scale step how
  // far one Shift+wheel notch will grow it. Each belongs on the marker it
  // governs, where you are already looking.
  const chip = (cls) => {
    const el = document.createElement("div");
    el.className = cls;
    document.getElementById("viewport")?.appendChild(el);
    return el;
  };
  const label = chip("axis-snap");
  const rotLabel = chip("axis-snap axis-rot");
  const scaleLabels = { x: chip("axis-snap axis-scale"), y: chip("axis-snap axis-scale"),
                        z: chip("axis-snap axis-scale") };

  axes = { ...built, id, space, label, rotLabel, scaleLabels, observer: null };
  paintAxes();
  orientAxes(node);
  // Position is re-read every frame rather than parented: parenting would
  // inherit the element's scale - and a 3x element must not get 3x arrows -
  // while re-deriving it also covers drags, undo, the ghost following the
  // cursor, and deletion, without any event plumbing at all.
  axes.observer = state.scene.onBeforeRenderObservable.add(() => {
    const cur = axesNode(id);
    if (!cur) { hideAxes(); return; }
    axes.root.position.copyFrom(cur.getWorldMatrix().getRow(3).toVector3());
    orientAxes(cur);
    paintAxes();
    placeAxisLabel();
  });
  emit("axes");
  return true;
}

/**
 * Park a chip over a world point, in canvas pixels.
 *
 * DOM overlays rather than scene geometry, for the same reason the marquee is
 * one: text has to be crisp at any distance, and these are readouts, not part
 * of the ship.
 */
function placeChip(el, at, text) {
  if (!el) return;
  const scene = state.scene;
  const engine = scene.getEngine();
  const w = engine.getRenderWidth(), h = engine.getRenderHeight();
  const p = Vector3.Project(at, Matrix.Identity(), scene.getTransformMatrix(),
    state.camera.viewport.toGlobal(w, h));
  // behind the camera projects to a mirrored point, which would park the chip
  // on the opposite side of the screen from the thing it belongs to
  if (p.z < 0 || p.z > 1) { el.hidden = true; return; }
  // Off the side of the canvas is just as wrong: the chips hang off the arrow
  // *tips*, which swing outside the viewport at close range, and #viewport does
  // not clip - one was measured sitting on a palette tile.
  if (p.x < 0 || p.x > w || p.y < 0 || p.y > h) { el.hidden = true; return; }
  const box = engine.getRenderingCanvas().getBoundingClientRect();
  const host = document.getElementById("viewport")?.getBoundingClientRect();
  if (!host) return;
  el.hidden = false;
  el.textContent = text;
  el.style.left = `${p.x + box.left - host.left}px`;
  el.style.top = `${p.y + box.top - host.top}px`;
}

/**
 * The move step at the origin, the turn angle in the curved arrow, and the
 * scale step on each cube.
 *
 * Every one of these decides how far the *next* keystroke will move this
 * element, and each was otherwise only visible in a toolbar combo at the far
 * edge of the screen. Putting each value on the marker it governs means the
 * gizmo answers "how far", not just "which way".
 *
 * The scale step gets one chip per lit cube rather than a single shared one:
 * with the axis set to `all` the three cubes are the statement that all three
 * axes grow, and a value on only one of them would read as "just this one".
 */
function placeAxisLabel() {
  if (!axes) return;
  placeChip(axes.label, axes.root.position,
    state.snap.pos ? `${state.snap.pos} m` : "free");

  const ring = axes.marks.rot[state.rotAxis];
  if (ring && axes.rotLabel) {
    if (!ring.isEnabled()) { axes.rotLabel.hidden = true; } else {
      // The curved arrow has a head, so it states a *direction* - and a
      // negative step turns the other way. Mirroring the arc reverses which
      // way it circulates, so the picture cannot disagree with the key.
      ring.scaling.x = state.snap.rot < 0 ? -1 : 1;
      ring.computeWorldMatrix(true);
      placeChip(axes.rotLabel, ring.getAbsolutePosition(),
        state.snap.rot ? `${state.snap.rot}°` : "free");
    }
  }

  for (const a of ["x", "y", "z"]) {
    const el = axes.scaleLabels?.[a];
    const cube = axes.marks.scale[a];
    if (!el || !cube) continue;
    if (!cube.isEnabled()) { el.hidden = true; continue; }
    cube.computeWorldMatrix(true);
    placeChip(el, cube.getAbsolutePosition(),
      state.snap.scale ? `${state.snap.scale}` : "free");
  }
}

export function hideAxes() {
  if (!axes) return false;
  if (axes.observer) state.scene.onBeforeRenderObservable.remove(axes.observer);
  axes.label?.remove();
  axes.rotLabel?.remove();
  for (const el of Object.values(axes.scaleLabels || {})) el.remove();
  for (const a of ["x", "y", "z"]) {
    axes.marks.rot[a].getChildMeshes().forEach((m) => m.dispose());
    axes.marks.rot[a].dispose();
    axes.arms[a].getChildMeshes().forEach((m) => m.dispose());
    axes.arms[a].dispose();
    axes.mats[a].dispose();
    axes.mats[`${a}_mark`].dispose();
  }
  axes.root.dispose();
  axes = null;
  emit("axes");
  return true;
}

/**
 * X toggles the axes; Shift+X toggles them in local space.
 *
 * The same element in the same space turns them off; a different element, or
 * the same one in the other space, moves or re-aims them.
 */
export function toggleAxes(id, space = "world") {
  if (axes && axes.id === id && axes.space === space) { hideAxes(); return null; }
  return showAxes(id, space) ? id : null;
}

/**
 * Of several ids, the one whose screen position is nearest the cursor.
 *
 * Screen distance rather than world distance: "the one closer to the mouse" is
 * a question about what you are looking at, and two elements the same distance
 * away on screen can be far apart in the ship.
 */
export function nearestToCursor(ids) {
  const scene = state.scene;
  let best = null;
  let bestD = Infinity;
  for (const id of ids) {
    const e = entryOf(id);
    if (!e) continue;
    const b = screenBoundsOf(e.node);
    if (!b) continue;
    const d = Math.hypot((b.minX + b.maxX) / 2 - scene.pointerX,
      (b.minY + b.maxY) / 2 - scene.pointerY);
    if (d < bestD) { bestD = d; best = id; }
  }
  return best;
}

/** Raise or lower the build plane. */
export function setGridElevation(y) {
  state.gridY = Math.round(y * 1000) / 1000;
  gridNode.position.y = state.gridY;
  emit("grid");
}

export function nudgeGridElevation(dir) {
  const step = state.snap.pos || 1;
  setGridElevation(state.gridY + dir * step);
}

/**
 * Where the cursor meets a horizontal plane at height `y`, by ray/plane
 * intersection.
 *
 * There is deliberately no pick-plane mesh: one would sit in front of the
 * geometry whenever the build plane is raised above it, making every element
 * unselectable. Solving the plane analytically also means it can never tie with
 * a floor tile resting exactly on it.
 */
export function cursorOnPlane(y) {
  const scene = state.scene;
  const ray = scene.createPickingRay(scene.pointerX, scene.pointerY,
    Matrix.Identity(), state.camera);
  if (Math.abs(ray.direction.y) < 1e-5) return null;
  const t = (y - ray.origin.y) / ray.direction.y;
  if (t <= 0) return null;
  return ray.origin.add(ray.direction.scale(t));
}

export function cursorOnGrid() { return cursorOnPlane(state.gridY); }

/**
 * Where the cursor lands on a *vertical* plane through `origin`, used for
 * dragging elements up and down.
 *
 * The plane faces the camera - its normal is the view direction flattened onto
 * the horizontal - so the cursor tracks the element no matter which way you are
 * looking. A fixed plane (say, world XY) would invert or go edge-on and stop
 * responding as soon as you orbited round.
 */
export function cursorOnVerticalPlane(origin) {
  const scene = state.scene;
  const ray = scene.createPickingRay(scene.pointerX, scene.pointerY,
    Matrix.Identity(), state.camera);
  const n = state.camera.getDirection(Vector3.Forward());
  n.y = 0;
  if (n.lengthSquared() < 1e-6) return null;      // looking straight down
  n.normalize();
  const denom = Vector3.Dot(ray.direction, n);
  if (Math.abs(denom) < 1e-5) return null;
  const t = Vector3.Dot(origin.subtract(ray.origin), n) / denom;
  if (t <= 0) return null;
  return ray.origin.add(ray.direction.scale(t));
}

// --------------------------------------------------------------- pointer
//
// There are no gizmos: placing and moving both go through the ghost in
// interact.js. This layer only decides what a click means and forwards it.

function setupPointer(scene) {
  const canvas = scene.getEngine().getRenderingCanvas();

  const down = new Map();
  let lastTapAt = 0;
  let lastTapPos = { x: 0, y: 0 };

  scene.onPointerObservable.add((pi, es) => {
    if (pi.type === PointerEventTypes.POINTERMOVE) {
      emit("pointermove", pi.event);
      return;
    }
    if (pi.type === PointerEventTypes.POINTERDOWN) {
      down.set(pi.event.button, { x: pi.event.clientX, y: pi.event.clientY, t: performance.now() });
      if (pi.event.button === 2) {
        rmbDown = true;
        rmbGesture = false;
        emit("rmbdown", pi.event);
      }
      if (pi.event.button === 0) {
        // interact.js claims the gesture for a drag; nothing downstream needs
        // it, but stopping here keeps a drag from reaching any other observer
        const req = { event: pi.event, capture: false };
        emit("pointerdown", req);
        if (req.capture) es.skipNextObservers = true;
      }
      return;
    }
    if (pi.type !== PointerEventTypes.POINTERUP) return;
    const start = down.get(pi.event.button);
    down.delete(pi.event.button);
    if (pi.event.button === 2) { rmbDown = false; emit("rmbup", pi.event); }
    if (pi.event.button === 0) emit("pointerup", pi.event);
    if (!start) return;
    const moved = Math.hypot(pi.event.clientX - start.x, pi.event.clientY - start.y);
    if (moved > 4) return;                       // that was a drag, not a click

    if (pi.event.button === 2) {
      // a right button that was steering the camera is a gesture, not a click
      if (!rmbGesture) emit("cancel", pi.event);
      return;
    }
    if (pi.event.button !== 0) return;

    // A press that was held is a camera gesture, not a click. Movement alone is
    // not enough to tell them apart: holding the left button still and then
    // releasing would otherwise select whatever the view happened to swing
    // over. Only a quick tap selects.
    if (performance.now() - start.t > CLICK_MS) return;

    // Detected here rather than via Babylon's POINTERDOUBLETAP so that the
    // first click always acts immediately - a placement tool cannot afford to
    // wait out a double-click timeout on every click.
    const now = performance.now();
    const isDouble = now - lastTapAt < DOUBLE_CLICK_MS
      && Math.hypot(pi.event.clientX - lastTapPos.x, pi.event.clientY - lastTapPos.y) < 6;
    lastTapAt = isDouble ? 0 : now;
    lastTapPos = { x: pi.event.clientX, y: pi.event.clientY };

    emit("click", pi.event);
    if (isDouble) emit("dblclick", pi.event);
  });

  // a button released outside the canvas never reaches the observable
  window.addEventListener("pointerup", (e) => {
    if (e.button === 2) rmbDown = false;
  });
  window.addEventListener("blur", () => { rmbDown = false; });
}

// ------------------------------------------------------ camera navigation
//
// WASD + Space/C fly the camera, Shift doubles the speed and Ctrl halves it.
// Everything is fed through cameraDirection/cameraRotation so the inertia
// slider still smooths it, with a (1 - inertia) factor so the steady-state
// speed is the one asked for regardless of the damping.

const held = new Set();
let fastMove = false;
let rmbDown = false;
// Set whenever the right button does anything other than click: drive with
// WASD, or change the fly speed with the wheel. Its release must then not also
// fire the cancel gesture.
let rmbGesture = false;

const MOVE_KEYS = new Set(["KeyW", "KeyS", "KeyA", "KeyD", "Space", "KeyC"]);
const MOVE_SPEED = 42;       // metres per second, the starting fly speed
const LOOK_SPEED = 1.6;      // radians per second
const PITCH_LIMIT = 1.5;

export function noteKey(code, isDown, ev) {
  fastMove = !!ev?.shiftKey;
  if (!MOVE_KEYS.has(code)) return false;
  if (isDown) {
    held.add(code);
    // holding the right button and driving counts as a camera gesture, not a
    // right-click, so the release must not also cancel the current action
    if (rmbDown) rmbGesture = true;
  } else {
    held.delete(code);
  }
  return true;
}

/** True while the right button is held - the camera-gesture modifier. */
export function isRmbDown() { return rmbDown; }

/**
 * Fly speed, adjusted by the wheel while the right button is held - the same
 * gesture Unreal uses. Ctrl+WASD cannot do this job: Ctrl+W closes the browser
 * tab (reserved by Chrome, and a page cannot cancel it), while Ctrl+S and
 * Ctrl+D are already Save and Duplicate here.
 *
 * Steps are multiplicative so the control feels the same at 3 m/s and 100 m/s.
 */
const SPEED_STEP = 1.15;
const SPEED_MIN = 1.5;
const SPEED_MAX = 300;

export function nudgeMoveSpeed(dir) {
  const next = state.moveSpeed * (dir > 0 ? SPEED_STEP : 1 / SPEED_STEP);
  state.moveSpeed = Math.min(SPEED_MAX, Math.max(SPEED_MIN, Math.round(next * 10) / 10));
  // Turning the wheel is what the right button was held for, so releasing it
  // must not also cancel the selection or the ghost.
  rmbGesture = true;
  emit("modes");
  return state.moveSpeed;
}

export function releaseAllKeys() {
  held.clear();
  fastMove = false;
  const cam = state.camera;
  if (!cam) return;
  // Movement runs through cameraDirection, so the camera coasts to a stop
  // under the inertia setting rather than halting dead. Clearing these stops
  // any *new* motion; the residual decays geometrically within a few frames.
  cam.cameraDirection?.setAll(0);
  cam.cameraRotation?.set(0, 0);
}

/** Move the camera along its own view direction, for Ctrl+wheel. */
export function dollyCamera(dir, step = 2) {
  const cam = state.camera;
  const fwd = cam.getDirection(Vector3.Forward());
  cam.cameraDirection.addInPlace(fwd.scale(dir * step * (1 - cam.inertia)));
}

function setupCameraMove(scene) {
  scene.onBeforeRenderObservable.add(() => {
    // Grounding runs even when standing still: the floor can move out from
    // under you when a module is deleted or dragged away.
    if (state.walk) groundCamera();
    if (!held.size) return;
    const cam = state.camera;
    const dt = Math.min(scene.getEngine().getDeltaTime() / 1000, 0.1);
    const damp = 1 - cam.inertia;

    const move = Vector3.Zero();
    const fwd = cam.getDirection(Vector3.Forward());
    const right = cam.getDirection(Vector3.Right());
    if (state.walk) {
      // Walking follows the floor, so the forward vector is flattened: looking
      // at your feet would otherwise drive you into the ground.
      fwd.y = 0;
      right.y = 0;
      if (fwd.lengthSquared() > 1e-6) fwd.normalize();
      if (right.lengthSquared() > 1e-6) right.normalize();
    }
    if (held.has("KeyW")) move.addInPlace(fwd);
    if (held.has("KeyS")) move.subtractInPlace(fwd);
    if (held.has("KeyD")) move.addInPlace(right);
    if (held.has("KeyA")) move.subtractInPlace(right);
    // Space/C are the only way to change height while flying; walking takes its
    // height from the floor, so they would just fight the grounding.
    if (!state.walk) {
      if (held.has("Space")) move.addInPlace(Vector3.Up());
      if (held.has("KeyC")) move.subtractInPlace(Vector3.Up());
    }
    if (move.lengthSquared() < 1e-6) return;

    const speed = state.moveSpeed * (fastMove ? 2 : 1) * dt * damp;
    cam.cameraDirection.addInPlace(move.normalize().scale(speed));
  });
}

/**
 * Whatever is under the cursor right now: a placement, a marker, or the grid.
 *
 * A door's portal quad sits exactly where its panels are and turns to face the
 * camera, so it covers the very geometry you have to select to register it as a
 * leaf. Shift used to make those quads transparent to the pick; `Shift+H` now
 * does it for anything at all - a ghosted element is unpickable - which is one
 * mechanism instead of two, and works on walls as well as doors.
 */
export function pickUnderCursor() {
  const scene = state.scene;
  const pick = scene.pick(scene.pointerX, scene.pointerY,
    (m) => m.isPickable && m.isEnabled());
  const id = pick?.hit
    ? (pick.pickedMesh?.metadata?.placementRoot?.name
      || pick.pickedMesh?.metadata?.markerRoot?.name
      || pick.pickedMesh?.metadata?.colliderRoot?.name)
    : null;
  if (id && entryOf(id)) return { kind: "entry", id, entry: entryOf(id), pick };

  const point = cursorOnGrid();
  return point ? { kind: "ground", point } : { kind: "none" };
}

export function snapPoint(p) {
  const s = state.snap.pos || 0;
  if (!s) return p.clone();
  return new Vector3(Math.round(p.x / s) * s, p.y, Math.round(p.z / s) * s);
}

// ------------------------------------------------------------ placements

export async function placeAt(moduleId, position, opts = {}) {
  if (!opts.silent) pushUndo();
  const id = opts.id || `P${String(state.nextId++).padStart(4, "0")}`;
  if (opts.id) {
    const n = parseInt(String(opts.id).replace(/\D/g, ""), 10);
    if (Number.isFinite(n) && n >= state.nextId) state.nextId = n + 1;
  }
  const node = await instantiate(moduleId, id);
  node.position.copyFrom(position);
  if (opts.rotation) {
    node.rotationQuaternion = Quaternion.FromEulerAngles(
      opts.rotation[0] * Math.PI / 180,
      opts.rotation[1] * Math.PI / 180,
      opts.rotation[2] * Math.PI / 180);
  }
  if (opts.scale) node.scaling.set(opts.scale[0], opts.scale[1], opts.scale[2]);

  const entry = {
    id,
    module: moduleId,
    chunk: opts.stage ? STAGE_CHUNK : (opts.chunk || state.activeChunk),
    name: opts.name || "",        // optional label, see renamePlacement
    // A stand-in on the collision staging area rather than part of the ship.
    // It is a real placement so that selection, the gizmo, hiding, dragging and
    // Ctrl+D all work on it unchanged - and filtered out at the two boundaries
    // that walk every placement, so it can never reach the ship.
    stage: !!opts.stage,
    node,
  };
  node.metadata = { placement: entry };
  state.placements.set(id, entry);
  if (!entry.stage && !state.chunks.includes(entry.chunk)) state.chunks.push(entry.chunk);
  applyVisibility();
  if (!opts.silent) { emit("placements"); select([id]); }
  return entry;
}

export function removeSelected() {
  if (!state.selection.length) return;
  pushUndo();
  for (const id of state.selection) {
    const p = state.placements.get(id);
    // Taking a staged element off the area must not lose what was fitted to
    // it: unstageModule reads its shapes into the per-module record first, so
    // staging the module again brings them straight back.
    if (p?.stage) hooks.unstageModule(id);
    else if (p) removePlacement(id);
    else if (state.colliders.has(id)) hooks.removeCollider(id);
    else removeMarkerNode(id);
  }
  select([]);
  hooks.harvestStage();
  emit("placements");
  emit("markers");
  emit("colliders");
}

function removeMarkerNode(id) {
  const m = state.markers.get(id);
  if (!m) return;
  m.mesh?.dispose();
  m.node.dispose();
  state.markers.delete(id);
}

export function removePlacement(id) {
  const entry = state.placements.get(id);
  if (!entry) return;
  entry.node.getChildMeshes().forEach((m) => m.dispose());
  entry.node.dispose();
  state.placements.delete(id);
}

export async function duplicateSelected() {
  if (!state.selection.length) return;
  pushUndo();
  const made = [];
  for (const id of state.selection) {
    const e = state.placements.get(id);
    if (!e) continue;
    const rot = eulerOf(e.node);
    const copy = await placeAt(e.module,
      e.node.position.add(new Vector3(state.snap.pos || 1, 0, 0)),
      { rotation: rot, scale: e.node.scaling.asArray(), chunk: e.chunk, name: e.name, silent: true });
    made.push(copy.id);
  }
  emit("placements");
  select(made);
}

export function clearAll() {
  hideAxes();
  for (const id of [...state.placements.keys()]) removePlacement(id);
  for (const id of [...state.markers.keys()]) removeMarkerNode(id);
  for (const id of [...state.colliders.keys()]) hooks.removeCollider(id);
  state.selection = [];
  state.nextId = 1;
  // ids restart at P0001, so a leftover entry would hide a brand new element
  state.hidden.clear();
  // keyed by name rather than id, but a load must not merge the old ship's
  // behaviours into the new one
  state.behaviors.clear();
  state.entities.clear();
  emit("placements");
  emit("markers");
  emit("selection");
}

// -------------------------------------------------------------- selection

export function select(ids) {
  state.selection = ids.filter((id) => entryOf(id));
  // Axes already on screen follow the selection: having asked to see them, you
  // almost never want them left behind on the element you just moved away from.
  // Only for a single pick - a multi-selection has no one element to sit on.
  if (axes && state.selection.length === 1 && state.selection[0] !== axes.id) {
    showAxes(state.selection[0]);
  }
  emit("selection");
}

export function toggleSelect(id) {
  const i = state.selection.indexOf(id);
  if (i >= 0) state.selection.splice(i, 1);
  else state.selection.push(id);
  emit("selection");
}

/**
 * Shift every selected node by a delta.
 *
 * The delta arrives in the world's terms - one arrow key is one step along one
 * axis - and is turned into the current move space here, so an arrow nudge and
 * a mouse drag agree about which way "X" points. Unlike a drag it ignores the
 * *axis* setting: the key already names the axis.
 */
export function nudgeSelection(delta) {
  if (!state.selection.length) return;
  const basis = moveBasis(entryOf(state.selection[0])?.node);
  const step = basis
    ? basis.x.scale(delta.x).add(basis.y.scale(delta.y)).add(basis.z.scale(delta.z))
    : delta;
  pushUndo();
  for (const id of state.selection) {
    const e = entryOf(id);
    if (e) e.node.position.addInPlace(step);
  }
  emit("transform");
}

export function focusSelection() {
  const nodes = state.selection.map((id) => entryOf(id)?.node).filter(Boolean);
  if (!nodes.length) return;
  focusNodes(nodes);
}

/** Frame these nodes, keeping the camera's current orientation. */
export function focusNodes(nodes) {
  if (!nodes?.length) return;
  let min = null, max = null;
  for (const n of nodes) {
    const b = worldBounds(n);
    if (!b) continue;
    min = min ? Vector3.Minimize(min, b.min) : b.min.clone();
    max = max ? Vector3.Maximize(max, b.max) : b.max.clone();
  }
  const c = min ? min.add(max).scale(0.5)
    : nodes[0].getAbsolutePosition().clone();
  const size = min ? max.subtract(min).length() : 4;
  const dist = Math.max(4, size * 1.3);

  const cam = state.camera;
  // back off along the current view direction so framing keeps the orientation
  const dir = cam.getDirection(Vector3.Forward()).normalize();
  cam.cameraDirection.setAll(0);
  cam.position.copyFrom(c.subtract(dir.scale(dist)));
  cam.setTarget(c);
}

// ----------------------------------------------------------------- chunks

export function addChunk(name) {
  if (!name || state.chunks.includes(name)) return false;
  state.chunks.push(name);
  emit("chunks");
  return true;
}

/**
 * Rename a chunk everywhere it is referenced.
 *
 * A chunk id is not just a label: placements carry it, doors name the two
 * chunks they join, and the active chunk is one. Renaming the list entry alone
 * would orphan every placement and silently break the portals, so all four are
 * rewritten together.
 */
export function renameChunk(from, to) {
  const name = String(to || "").trim();
  if (!from || !name || from === name) return false;
  if (!state.chunks.includes(from) || state.chunks.includes(name)) return false;
  pushUndo();

  state.chunks = state.chunks.map((c) => (c === from ? name : c));
  for (const e of state.placements.values()) if (e.chunk === from) e.chunk = name;
  for (const m of state.markers.values()) {
    if (m.chunkA === from) m.chunkA = name;
    if (m.chunkB === from) m.chunkB = name;
  }
  if (state.activeChunk === from) state.activeChunk = name;

  applyVisibility();
  emit("chunks");
  emit("placements");
  emit("markers");
  return true;
}

/**
 * Give a placement a name.
 *
 * The name is the element's *node* name in ship.glb - the parent node, with its
 * primitives numbered off it - and the runtime keys its `behaviors` map off
 * exactly that. Deliberately by name and not by id, so that naming six crates
 * "crate" makes one behaviour entry govern all six. Names are therefore **not**
 * unique, and must not be made unique.
 *
 * Ids stay machine-generated and stable, and are the editor's own handle. An
 * unnamed element falls back to its id in the .glb, for want of anything
 * better to call it.
 */
export function renamePlacement(id, name) {
  const e = state.placements.get(id);
  if (!e) return false;
  const next = String(name || "").trim();
  if (e.name === next) return false;
  pushUndo();
  e.name = next;
  emit("placements");
  emit("current");
  emit("behaviors");
  return true;
}

// ------------------------------------------------------------- behaviours
//
// Two halves, matching the manifest:
//
//   behaviors   a *library* of named definitions - "door_liquefiable" is
//               `{ liquefiable: true, fluidSim: [...] }`. The body is arbitrary
//               JSON, deliberately: the runtime owns which flags exist, and a
//               tool that only understood the ones it knew about today would
//               have to be edited every time one was added.
//   entities    which of those a node name carries, plus the `linked` node
//               names some of them need - a door half links to its other half.
//
// Both are keyed by node name, and node names are shared on purpose, so one
// entry can govern every element carrying it.

/** Definition body for a behaviour name, or null. */
export function getBehaviorDef(name) {
  const b = state.behaviors.get(String(name || "").trim());
  return b ? JSON.parse(JSON.stringify(b)) : null;
}

/** Every behaviour in the library, in insertion order. */
export function behaviorNames() { return [...state.behaviors.keys()]; }

/**
 * Create or replace a definition. The body must be a JSON *object*: the
 * manifest maps a name to a bag of flags, and an array or a bare number there
 * would be silently ignored by the runtime rather than rejected.
 */
export function setBehaviorDef(name, body) {
  const key = String(name || "").trim();
  if (!key) return false;
  if (!body || typeof body !== "object" || Array.isArray(body)) return false;
  const next = JSON.parse(JSON.stringify(body));
  if (JSON.stringify(state.behaviors.get(key) ?? null) === JSON.stringify(next)) return false;
  pushUndo();
  state.behaviors.set(key, next);
  emit("behaviors");
  return true;
}

/**
 * Rename a definition, carrying every reference with it. A rename that left the
 * entities pointing at the old name would silently drop their behaviour.
 */
export function renameBehaviorDef(from, to) {
  const next = String(to || "").trim();
  if (!from || !next || from === next) return false;
  if (!state.behaviors.has(from) || state.behaviors.has(next)) return false;
  pushUndo();
  // rebuilt rather than set(), so the library keeps its order
  state.behaviors = new Map([...state.behaviors].map(([k, v]) => [k === from ? next : k, v]));
  for (const list of state.entities.values()) {
    for (const b of list) if (b.name === from) b.name = next;
  }
  emit("behaviors");
  return true;
}

/** Delete a definition and strip it from every entity that carried it. */
export function deleteBehaviorDef(name) {
  if (!state.behaviors.has(name)) return false;
  pushUndo();
  state.behaviors.delete(name);
  for (const [node, list] of [...state.entities]) {
    const kept = list.filter((b) => b.name !== name);
    if (kept.length) state.entities.set(node, kept);
    else state.entities.delete(node);
  }
  emit("behaviors");
  return true;
}

/** The behaviours a node name carries, as a copy. */
export function entityBehaviors(nodeName) {
  const list = state.entities.get(String(nodeName || "").trim()) || [];
  return list.map((b) => ({
    name: b.name,
    linked: [...(b.linked || [])],
    excludeSDF: [...(b.excludeSDF || [])],
    ...(b.direction ? { direction: [...b.direction] } : {}),
  }));
}

/** Attach a library behaviour to a node name. Names cannot carry one twice. */
export function addEntityBehavior(nodeName, behaviorName) {
  const node = String(nodeName || "").trim();
  if (!node || !state.behaviors.has(behaviorName)) return false;
  const list = state.entities.get(node) || [];
  if (list.some((b) => b.name === behaviorName)) return false;
  pushUndo();
  state.entities.set(node, [...list, { name: behaviorName, linked: [], excludeSDF: [] }]);
  emit("behaviors");
  return true;
}

export function removeEntityBehavior(nodeName, behaviorName) {
  const node = String(nodeName || "").trim();
  const list = state.entities.get(node);
  if (!list) return false;
  const kept = list.filter((b) => b.name !== behaviorName);
  if (kept.length === list.length) return false;
  pushUndo();
  if (kept.length) state.entities.set(node, kept);
  else state.entities.delete(node);
  emit("behaviors");
  return true;
}

/** The other nodes this one drags along with it - a door half links its twin. */
export function setEntityLinked(nodeName, behaviorName, linked) {
  const node = String(nodeName || "").trim();
  const list = state.entities.get(node);
  const entry = list?.find((b) => b.name === behaviorName);
  if (!entry) return false;
  const next = [...new Set(linked.map((s) => String(s).trim()).filter(Boolean))]
    .filter((n) => n !== node);          // linking a node to itself says nothing
  if (JSON.stringify(entry.linked) === JSON.stringify(next)) return false;
  pushUndo();
  entry.linked = next;
  emit("behaviors");
  return true;
}

/** True when a definition asks for liquefaction, which is what needs `linked`. */
export function isLiquefiable(behaviorName) {
  return getBehaviorDef(behaviorName)?.liquefiable === true;
}

/**
 * Names whose baked SDF the fluid sim must drop while this shot's water lives.
 *
 * A door names its frame: the water is seeded inside the door's own volume,
 * which sits *within* the frame, so leaving the frame in the SDF union would
 * eject the particles and dam the opening the door just left behind.
 */
export function setEntityExcludeSDF(nodeName, behaviorName, names) {
  const node = String(nodeName || "").trim();
  const entry = state.entities.get(node)?.find((b) => b.name === behaviorName);
  if (!entry) return false;
  const next = [...new Set(names.map((s) => String(s).trim()).filter(Boolean))];
  if (JSON.stringify(entry.excludeSDF || []) === JSON.stringify(next)) return false;
  pushUndo();
  entry.excludeSDF = next;
  emit("behaviors");
  return true;
}

/**
 * Whether a node carries a behaviour that gives it a rigid body.
 *
 * The runtime's own rule: `dynamic`, or `liquefiable` - which implies it, since
 * a mesh has to be physically present before it can melt. Only such a node has
 * an SDF in the simulation to drop, so only such a node is worth offering.
 */
export function isDynamicNode(nodeName) {
  return entityBehaviors(nodeName).some((b) => {
    const def = getBehaviorDef(b.name);
    return def?.dynamic === true || def?.liquefiable === true;
  });
}

/** Dynamic node names in a chunk - the candidates for `excludeSDF`. */
export function dynamicNodeNamesInChunk(chunk, exclude = "") {
  return nodeNamesInChunk(chunk, exclude).filter(isDynamicNode);
}

/**
 * The default direction a definition suggests, if it names one.
 *
 * `direction` is optional on *every* applied behaviour, so this is only a
 * starting value for the fields - not a gate on whether they appear. Gating on
 * it was the first design and it was wrong: it made an optional parameter
 * invisible until you knew to declare it, which is precisely the thing the
 * person editing does not know.
 */
export function defaultDirection(behaviorName) {
  const d = getBehaviorDef(behaviorName)?.direction;
  return Array.isArray(d) && d.length === 3 && d.every(Number.isFinite) ? d.map(Number) : null;
}

/**
 * The way an entity faces. Stored as typed rather than normalised: normalising
 * on every commit fights you as you fill the three fields in - typing 1 into Y
 * after X would turn both into 0.707 before you reached Z.
 */
export function setEntityDirection(nodeName, behaviorName, dir) {
  const node = String(nodeName || "").trim();
  const list = state.entities.get(node);
  const entry = list?.find((b) => b.name === behaviorName);
  if (!entry) return false;
  const next = Array.isArray(dir) && dir.length === 3 && dir.every(Number.isFinite)
    && dir.some((v) => v !== 0)
    ? dir.map((v) => Math.round(v * 1e4) / 1e4)
    : null;
  if (JSON.stringify(entry.direction ?? null) === JSON.stringify(next)) return false;
  pushUndo();
  if (next) entry.direction = next;
  else delete entry.direction;
  emit("behaviors");
  return true;
}

/**
 * Node names in a chunk - the candidates for `linked`.
 *
 * The current room, not the whole ship: linking is for pieces that behave as
 * one, which in practice are neighbours, and a ship-wide list would be hundreds
 * of entries long. Unnamed elements are left out, having nothing to link by.
 */
export function nodeNamesInChunk(chunk, exclude = "") {
  const names = new Set();
  for (const e of state.placements.values()) {
    if (e.chunk !== chunk) continue;
    const n = String(e.name || "").trim();
    if (n && n !== exclude) names.add(n);
  }
  return [...names].sort((a, b) => a.localeCompare(b));
}

/** How many elements carry a node name. Zero means the entry is orphaned. */
export function nodesNamed(name) {
  const key = String(name || "").trim();
  if (!key) return 0;
  let n = 0;
  for (const e of state.placements.values()) if (e.name === key) n++;
  return n;
}

export function assignSelectionToChunk(chunk) {  if (!state.selection.length) return;
  pushUndo();
  for (const id of state.selection) {
    const e = state.placements.get(id);
    if (e) e.chunk = chunk;
  }
  applyVisibility();
  emit("placements");
}

/**
 * One place decides what is on screen, because two would fight.
 *
 * Chunk isolation and H-hiding both work by disabling nodes, so neither can own
 * `setEnabled` alone: isolating a chunk would reveal everything you had hidden,
 * and unhiding would reveal the chunks you had isolated away. Visibility is
 * derived from both instead, and always recomputed from scratch.
 *
 * A ghosted element is drawn at `state.veilAlpha` *and made unpickable*. Half
 * alpha alone would be no use for the thing hiding is for - reaching into a
 * room without flying round the wall in front of it - because every click would
 * still land on the wall you can now see through.
 */
const veilClones = new Map();          // element id -> the translucent stand-ins

export function isVeilClone(mesh) { return !!mesh?.metadata?.veilClone; }

function realMeshes(node) {
  return node.getChildMeshes().filter((m) => !isVeilClone(m));
}

/**
 * Swap an element's meshes for translucent copies.
 *
 * Placements are hardware instances sharing one source mesh - and one
 * *material* - per module, so the two obvious routes are both closed:
 * Babylon refuses per-instance `visibility` outright ("Setting visibility on an
 * instanced mesh has no effect"), and forcing the shared material to
 * ALPHABLEND, which an earlier version of this did, moves **every mesh drawn
 * with that material** into the transparent pass. The kit shares materials
 * across modules, so that quietly turned off depth writes for most of the ship
 * and things behind walls started showing through.
 *
 * Cloning is the only way to make one element translucent without touching what
 * anything else is drawn with. The clone shares its geometry (`clone(.., true)`)
 * and its textures (see ghostMaterialFor), so the cost is a draw call, not an
 * atlas.
 */
function buildVeilClones(entry) {
  const out = [];
  for (const m of realMeshes(entry.node)) {
    const src = m.sourceMesh || m;
    const c = src.clone(`${m.name}__veil`, entry.node, true);
    if (!c) continue;
    c.setEnabled(true);
    c.isVisible = true;
    c.isPickable = false;
    // the instance carries the part's placement within the module; the
    // prototype it was cloned from carries the prototype's own
    c.position.copyFrom(m.position);
    c.rotationQuaternion = m.rotationQuaternion ? m.rotationQuaternion.clone() : null;
    if (!c.rotationQuaternion) c.rotation.copyFrom(m.rotation);
    c.scaling.copyFrom(m.scaling);
    c.material = ghostMaterialFor(src.material, "VEIL", state.veilAlpha);
    c.metadata = { veilClone: true };
    out.push(c);
    m.setEnabled(false);
  }
  return out;
}

function dropVeilClones(id) {
  const clones = veilClones.get(id);
  if (!clones) return;
  for (const c of clones) c.dispose();
  veilClones.delete(id);
}

/** Rebuild an element's veil, for when its geometry changed underneath it. */
export function refreshVeil(id) {
  if (!veilClones.has(id)) return;
  dropVeilClones(id);
  applyVisibility();
}

function setVeil(entry, on) {
  if (on && !veilClones.has(entry.id)) {
    veilClones.set(entry.id, buildVeilClones(entry));
  } else if (!on && veilClones.has(entry.id)) {
    dropVeilClones(entry.id);
  }
  if (!on) for (const m of realMeshes(entry.node)) m.setEnabled(true);
}

export function applyVisibility() {
  const veilOf = (id) => (veilSuspended ? undefined : state.hidden.get(id));
  // The staging area is a separate world: while it is open the ship is hidden,
  // never touched, so closing it puts everything back. Hiding and the veil work
  // on both, which is what makes H behave the same in either place.
  const staging = state.collisionMode;
  // The layer switch composes with everything else rather than fighting it: it
  // can only ever take things *off* screen, so chunk isolation and the Shift+H
  // veil keep the last word on what is left.
  const geometryOn = state.showLayer !== "collision";
  const collisionOn = state.showLayer !== "geometry";

  for (const e of state.placements.values()) {
    const veil = veilOf(e.id);
    const on = (e.stage ? staging : !staging && geometryOn
      && (!state.isolate || e.chunk === state.activeChunk)) && veil !== "hidden";
    e.node.setEnabled(on);
    setVeil(e, veil === "ghost");
    for (const m of realMeshes(e.node)) m.isPickable = veil !== "ghost";
  }
  // A door is not in a chunk, but it *joins* two - so isolation does have
  // something to say about it: show it when the active chunk is one of its two
  // sides. Sides left on "(auto)" are resolved the same way the manifest
  // resolves them, by nearest chunk volume, so what you see under isolation is
  // what will be written.
  const sidesOf = doorSideResolver();
  for (const mk of state.markers.values()) {
    const veil = veilOf(mk.id);
    const joins = mk.type !== "door" || !state.isolate
      || sidesOf(mk).includes(state.activeChunk);
    mk.node.setEnabled(!staging && geometryOn && joins && veil !== "hidden");
    setVeil(mk, veil === "ghost");
    for (const m of realMeshes(mk.node)) m.isPickable = veil !== "ghost";
  }
  // A room's primitives belong to a chunk and follow isolation like placements.
  // The staging area's are a working copy of what each module carries, and only
  // exist while it is open.
  for (const c of state.colliders.values()) {
    const veil = veilOf(c.id);
    const on = (c.stage ? staging : collisionOn && !staging
      && (!state.isolate || c.chunk === state.activeChunk)) && veil !== "hidden";
    c.node.setEnabled(on);
    if (c.mesh) c.mesh.isPickable = veil !== "ghost";
  }
  // Collision a placement inherits from its module is drawn from the record,
  // not stored as elements - see refreshCollisionPreview.
  hooks.refreshCollisionPreview?.();
}

/** What the viewport shows: the ship, its collision, or both. */
export const SHOW_LAYERS = ["both", "geometry", "collision"];

export function setShowLayer(layer) {
  if (!SHOW_LAYERS.includes(layer) || layer === state.showLayer) return false;
  state.showLayer = layer;
  // Selecting something and then hiding its layer would leave the gizmo and the
  // inspector acting on an element nobody can see.
  if (state.selection.some((id) => {
    const e = entryOf(id);
    return e && (e.type === "collider" ? layer === "geometry" : layer === "collision");
  })) select([]);
  applyVisibility();
  emit("modes");
  return true;
}

/**
 * The two chunks a door joins, for isolation.
 *
 * Returns a function rather than a value so the chunk volumes - which cost a
 * world-bounds pass over every placement - are measured only if some door
 * actually needs them, and then only once per call. A door with both sides set
 * by hand needs nothing measured at all.
 */
function doorSideResolver() {
  let boxes = null;
  const volumes = () => {
    if (boxes) return boxes;
    boxes = [];
    for (const id of state.chunks) {
      let min = null, max = null;
      for (const p of state.placements.values()) {
        if (p.stage || p.chunk !== id) continue;
        const b = worldBounds(p.node);
        if (!b) continue;
        min = min ? Vector3.Minimize(min, b.min) : b.min.clone();
        max = max ? Vector3.Maximize(max, b.max) : b.max.clone();
      }
      if (min) boxes.push({ id, min, max });
    }
    return boxes;
  };
  const gap = (p, b) => {
    const dx = Math.max(b.min.x - p.x, 0, p.x - b.max.x);
    const dy = Math.max(b.min.y - p.y, 0, p.y - b.max.y);
    const dz = Math.max(b.min.z - p.z, 0, p.z - b.max.z);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };
  return (mk) => {
    let a = mk.chunkA || "";
    let b = mk.chunkB || "";
    if (a && b) return [a, b];
    const near = volumes()
      .map((v) => ({ id: v.id, d: gap(mk.node.getAbsolutePosition(), v) }))
      .sort((x, y) => x.d - y.d);
    if (!a) a = near.find((n) => n.id !== b)?.id || "";
    if (!b) b = near.find((n) => n.id !== a)?.id || "";
    return [a, b];
  };
}

/**
 * How see-through a ghosted element is. Live: the cached veil materials are
 * shared by every ghosted element, so moving the slider repaints them all.
 */
export function setVeilAlpha(a) {
  const v = Math.min(0.95, Math.max(0.05, Number(a) || 0));
  state.veilAlpha = v;
  for (const [key, mat] of ghostMats) if (key.endsWith(":VEIL")) mat.alpha = v;
  emit("modes");
}

/**
 * Change a ship-wide constant.
 *
 * Undoable, because it changes what the ship *is*: fitting collision with a
 * different shell thickness produces different geometry, and a slip of the
 * keyboard has to be as recoverable as any other edit.
 */
export function setConfig(key, value) {
  if (!(key in CONFIG_DEFAULTS)) return false;
  const v = Number(value);
  const range = CONFIG_RANGE[key] || { min: 0, max: Infinity };
  if (!Number.isFinite(v) || v < range.min || v > range.max) return false;
  if (Math.abs(state.config[key] - v) < 1e-9) return false;
  pushUndo();
  state.config = { ...state.config, [key]: v };
  emit("config");
  return true;
}

/** Restore a ship-wide constant to the value the editor ships with. */
export function resetConfig(key) {
  return setConfig(key, CONFIG_DEFAULTS[key]);
}

/**
 * Run `fn` with nothing veiled, so the exporter sees the ship as authored.
 *
 * Wraps the *whole* of an export, not just the write: the allowlist and the
 * primitive renaming are built from `getChildMeshes()`, so a stand-in still in
 * the tree at that moment would take a primitive index and be written out.
 */
export async function withVeilSuspended(fn) {
  if (veilSuspended) return await fn();
  veilSuspended = true;
  applyVisibility();
  try {
    return await fn();
  } finally {
    veilSuspended = false;
    applyVisibility();
  }
}

/**
 * Park the selection out of the way, one step further each press.
 *
 * `Shift+H` cycles **ghost -> hidden -> ghost**: half alpha to see past
 * something while still knowing it is there, then gone entirely. It does not
 * cycle back to fully opaque, because `H` already does that for everything at
 * once and a third state in the cycle would just make the useful two harder to
 * reach.
 *
 * Undoable, and part of the snapshot, so `Ctrl+Z` and `Ctrl+Y` are symmetric.
 * They were not: a restore begins with clearAll(), which drops the hidden set,
 * so *any* undo revealed everything and no redo could hide it again. It is
 * still kept out of the manifest - hiding is a way of looking, not a property
 * of the ship - so a reload always starts with everything visible.
 *
 * The selection is *kept*, unlike before. It has to be: the cycle is driven by
 * pressing the same key again, and dropping the selection would strand whatever
 * you just hid with no way back except `H`. Ghosted elements are unpickable, so
 * a selection is the only handle left on them.
 *
 * Returns the level everything landed on, for the status line.
 */
export function hideSelected() {
  const ids = state.selection.filter((id) => entryOf(id));
  if (!ids.length) return null;
  pushUndo();
  // One level for the whole selection, taken from the first element, so a
  // mixed selection converges instead of scattering further apart on each
  // press. Anything not yet veiled starts the cycle at "ghost".
  const next = state.hidden.get(ids[0]) === "ghost" ? "hidden" : "ghost";
  for (const id of ids) state.hidden.set(id, next);
  applyVisibility();
  emit("placements");
  return { level: next, count: ids.length };
}

export function unhideAll() {
  const n = state.hidden.size;
  if (!n) return 0;
  pushUndo();
  state.hidden.clear();
  applyVisibility();
  emit("placements");
  return n;
}

export function hiddenCount() { return state.hidden.size; }

/** How many are at each level, for the status bar. */
export function veilCounts() {
  let ghost = 0, hidden = 0;
  for (const level of state.hidden.values()) {
    if (level === "hidden") hidden++; else ghost++;
  }
  return { ghost, hidden };
}

// ------------------------------------------------------------------- undo

// markers.js registers itself here so editor.js can serialise markers without
// importing it back and forming an import cycle.
export const hooks = {
  serializeMarkers: () => [],
  deserializeMarkers: () => {},
  serializeColliders: () => [],
  deserializeColliders: () => {},
  removeCollider: () => {},
  reconcileCollider: () => false,
  // interact.js owns the placement ghost; the axes need its node, and importing
  // interact.js back would close a cycle
  ghostNode: () => null,
};

/**
 * The camera, for the manifest only.
 *
 * Deliberately *not* part of serialize(): that feeds the undo stack, and
 * undoing an edit should not also throw the view somewhere else. Position and
 * rotation are stored rather than a look-at target, because they are what the
 * camera actually holds - deriving a target and calling setTarget() would round
 * -trip through a decomposition for no gain.
 */
export function serializeView() {
  const cam = state.camera;
  if (!cam) return null;
  return {
    position: round3(cam.position.asArray()),
    rotation: round3([cam.rotation.x, cam.rotation.y, cam.rotation.z]),
    // for anyone reading the manifest by hand
    target: round3(cam.position.add(cam.getDirection(Vector3.Forward()).scale(10)).asArray()),
  };
}

export function applyView(view) {
  const cam = state.camera;
  if (!cam || !view?.position || !view?.rotation) return false;
  cam.position.copyFrom(Vector3.FromArray(view.position));
  cam.rotation.set(view.rotation[0], view.rotation[1], view.rotation[2]);
  // any coasting from before the load would drag the restored view off again
  cam.cameraDirection?.setAll(0);
  cam.cameraRotation?.set(0, 0);
  return true;
}

/**
 * The lighting the ship is authored under, for the manifest.
 *
 * Unlike the camera, this **is** part of serialize() and so rides the undo
 * stack: it is an authored value the manifest carries and the runtime reads, so
 * getting it wrong is as much an edit to take back as moving a wall. The camera
 * stays off the stack, because where you happen to be standing is not an edit.
 *
 * These are the **Runtime light** values - the pair tuned with the editor's own
 * lights switched off - because that is the picture the demos render. `strength`
 * is `scene.environmentIntensity`, `exposure` is the linear multiplier exactly
 * as the slider shows it, and `toneMapping` names the view transform both ends
 * apply.
 */
export function serializeEnvironment() {
  const set = state.lightSets.runtime;
  return {
    strength: round3([set.strength])[0],
    toneMapping: state.toneMapping,
    exposure: round3([set.exposure])[0],
  };
}

/**
 * The editor's own pair, kept apart from `environment` because it describes
 * this tool and not the ship. The authoring rig adds four analytic lights the
 * game does not have, so the two need different numbers to look right, and
 * writing the editor's into `environment` would blow the demos out.
 */
export function serializeEditorEnvironment() {
  const set = state.lightSets.editor;
  return {
    strength: round3([set.strength])[0],
    exposure: round3([set.exposure])[0],
  };
}

/**
 * Put the lighting back on load. Anything missing is left alone, so a manifest
 * written before this section existed loads without disturbing the viewport.
 */
export function applyEnvironment(env, editorEnv) {
  if (!env && !editorEnv) return false;
  if (env && typeof env === "object") {
    if (Number.isFinite(env.strength)) state.lightSets.runtime.strength = env.strength;
    if (Number.isFinite(env.exposure)) state.lightSets.runtime.exposure = asLinearExposure(env.exposure);
    if (env.toneMapping) setToneMapping(env.toneMapping);
  }
  if (editorEnv && typeof editorEnv === "object") {
    if (Number.isFinite(editorEnv.strength)) state.lightSets.editor.strength = editorEnv.strength;
    if (Number.isFinite(editorEnv.exposure)) {
      state.lightSets.editor.exposure = asLinearExposure(editorEnv.exposure);
    }
  } else if (env && typeof env === "object") {
    // A manifest from before the split has one pair; give it to both rather
    // than leaving the editor on defaults that have nothing to do with it.
    state.lightSets.editor = { ...state.lightSets.runtime };
  }
  const set = activeLightSet();
  setEnvIntensity(set.strength);
  setExposure(set.exposure);
  emit("environment");
  return true;
}

/**
 * Exposure used to be written in Blender STOPS and raised to a power at the
 * other end. It is the plain multiplier now - what the slider shows is what
 * every consumer applies - but a *negative* value can only have come from the
 * old format, since the slider has never gone below 0.15. That makes the
 * conversion unambiguous, and worth doing rather than clamping a real setting
 * up to the floor.
 */
function asLinearExposure(v) {
  if (v >= 0) return v;
  const linear = Math.round(2 ** v * 1e4) / 1e4;
  console.warn(`exposure ${v} looks like the old stops format — reading it as ${linear}`);
  return linear;
}

export function serialize() {
  return {
    chunks: [...state.chunks],
    activeChunk: state.activeChunk,
    // The Env/Exposure pairs and the tone mapping. On the stack because they
    // are *authored* values that the manifest carries and the runtime reads -
    // getting the lighting wrong is as much an edit to undo as moving a wall.
    // Both pairs travel together: the Runtime-light toggle only decides which
    // one the sliders edit, so restoring one and not the other would leave the
    // hidden set behind.
    lightSets: {
      editor: { ...state.lightSets.editor },
      runtime: { ...state.lightSets.runtime },
    },
    toneMapping: state.toneMapping,
    // Ship data, so it belongs on the undo stack with everything else - and it
    // survives the clearAll() that a restore begins with.
    behaviors: Object.fromEntries(
      [...state.behaviors].map(([k, v]) => [k, JSON.parse(JSON.stringify(v))])),
    // Not ship data, but on the stack all the same: a restore clears it, so
    // leaving it out made every undo reveal what H had parked away.
    hidden: [...state.hidden],    entities: Object.fromEntries([...state.entities]
      .map(([k, v]) => [k, v.map((b) => ({
        name: b.name,
        linked: [...b.linked],
        excludeSDF: [...(b.excludeSDF || [])],
        // glTF space, matching the manifest - readBehaviorExtras() flips it back
        ...(b.direction ? { direction: flipX(b.direction) } : {}),
      }))])),
    instances: shipPlacements().map((e) => ({
      id: e.id,
      module: e.module,
      chunk: e.chunk,
      ...(e.name ? { name: e.name } : {}),
      position: round3(e.node.position.asArray()),
      rotation: round3(eulerOf(e.node)),
      scale: round3(e.node.scaling.asArray()),
    })),
    colliders: hooks.serializeColliders(),
    markers: hooks.serializeMarkers(),
    // Authored per kit module, in the module's own local space and the editor's
    // own coordinates. This is the *source*; the manifest's `moduleCollision`
    // is the mirrored, Havok-shaped copy derived from it - exactly the way
    // `colliders` relates to `collision`. They were once the same key, and a
    // reload quietly dropped every shape because the reader expected the other
    // form and filtered them all out.
    moduleShapes: serializeModuleCollision(),
    // Live while the bench is open: state.stageLayout is only written when it
    // closes, so reading it directly made the dirty check blind to anything
    // staged since - and a save from the bench wrote the previous roster.
    stageLayout: (hooks.stageLayoutNow?.() || state.stageLayout)
      .map((s) => ({ module: s.module, position: [...s.position] })),
    config: { ...state.config },
  };
}

/** The per-module shapes, as plain data. */
export function serializeModuleCollision() {
  const out = {};
  for (const [moduleId, shapes] of state.moduleCollision) {
    if (!shapes?.length) continue;
    out[moduleId] = shapes.map((s) => ({
      kind: s.kind,
      position: [...s.position],
      rotation: [...s.rotation],
      scale: [...s.scale],
    }));
  }
  return out;
}

/** Replace the per-module shapes wholesale. */
export function loadModuleCollision(data, stageLayout) {
  state.moduleCollision = new Map();
  for (const [moduleId, shapes] of Object.entries(data || {})) {
    if (!Array.isArray(shapes) || !shapes.length) continue;
    state.moduleCollision.set(moduleId, shapes
      .filter((s) => s && s.kind && Array.isArray(s.position))
      .map((s) => ({
        kind: s.kind,
        position: [...s.position],
        rotation: [...(s.rotation || [0, 0, 0])],
        scale: [...(s.scale || [1, 1, 1])],
      })));
  }
  if (Array.isArray(stageLayout)) {
    state.stageLayout = stageLayout
      .filter((s) => s && typeof s.module === "string" && Array.isArray(s.position))
      .map((s) => ({ module: s.module, position: [...s.position] }));
  }
  applyVisibility();      // the inherited-collision preview is built from this
  emit("colliders");
}

/**
 * Read module shapes back out of the manifest's runtime block.
 *
 * Only for a manifest written before the authoring form had a key of its own.
 * The runtime block is mirrored on X and speaks Havok's parameters, so it is
 * undone here rather than fed to the reader that expects editor coordinates -
 * which is what silently emptied it before.
 */
export function moduleShapesFromRuntime(block) {
  const out = {};
  for (const [moduleId, shapes] of Object.entries(block || {})) {
    if (!Array.isArray(shapes)) continue;
    const made = [];
    for (const s of shapes) {
      if (!s?.kind || !Array.isArray(s.centre)) continue;
      const position = [-s.centre[0], s.centre[1], s.centre[2]];
      if (s.kind === "box") {
        const q = Array.isArray(s.rotation)
          ? new Quaternion(-s.rotation[0], s.rotation[1], s.rotation[2], -s.rotation[3])
          : Quaternion.Identity();
        const e = q.toEulerAngles();
        made.push({
          kind: "box",
          position,
          rotation: [e.x, e.y, e.z].map((r) => r * 180 / Math.PI),
          scale: (s.halfExtents || [0.5, 0.5, 0.5]).map((v) => v * 2),
        });
      } else if (s.kind === "sphere") {
        const d = (s.radius ?? 0.5) * 2;
        made.push({ kind: "sphere", position, rotation: [0, 0, 0], scale: [d, d, d] });
      } else {
        // the segment carries the axis, so the turn is whatever takes +Y to it
        const d = (s.radius ?? 0.5) * 2;
        const h = s.height ?? 1;
        let rotation = [0, 0, 0];
        if (Array.isArray(s.pointA) && Array.isArray(s.pointB)) {
          const a = new Vector3(-s.pointA[0], s.pointA[1], s.pointA[2]);
          const b = new Vector3(-s.pointB[0], s.pointB[1], s.pointB[2]);
          const axis = b.subtract(a);
          if (axis.length() > 1e-6) {
            axis.normalize();
            const up = Vector3.Up();
            const dot = Vector3.Dot(up, axis);
            let q;
            if (dot > 1 - 1e-6) q = Quaternion.Identity();
            else if (dot < -1 + 1e-6) q = Quaternion.RotationAxis(Vector3.Right(), Math.PI);
            else q = Quaternion.RotationAxis(Vector3.Cross(up, axis).normalize(), Math.acos(dot));
            const e = q.toEulerAngles();
            rotation = [e.x, e.y, e.z].map((r) => r * 180 / Math.PI);
          }
        }
        made.push({ kind: s.kind, position, rotation, scale: [d, h, d] });
      }
    }
    if (made.length) out[moduleId] = made;
  }
  return out;
}

export async function deserialize(data) {
  // Nothing a restore does is itself an undoable edit. Without this guard a
  // single stray pushUndo() anywhere in the restore path wipes the redo stack
  // and pushes a half-restored snapshot - which is exactly how restoring a
  // spawn marker used to break redo and lose the player start.
  restoring = true;
  try {
    await restoreFrom(data);
  } finally {
    restoring = false;
  }
}

async function restoreFrom(data) {
  clearAll();
  // Defaults first, so a layout saved before a setting existed comes back with
  // that setting's default rather than undefined.
  state.config = { ...CONFIG_DEFAULTS, ...(data.config || {}) };
  // `moduleShapes` is the authoring form. A manifest written before it had a
  // key of its own carries only the runtime one, so convert rather than lose it.
  loadModuleCollision(
    data.moduleShapes || moduleShapesFromRuntime(data.moduleCollision),
    data.stageLayout);
  // buildManifest() writes chunks as rich objects; serialize() writes plain
  // ids. Accept either so a saved manifest reloads cleanly.
  const ids = (data.chunks || []).map((c) => (typeof c === "string" ? c : c.id)).filter(Boolean);
  state.chunks = ids.length ? ids : ["CH00_Storage"];
  state.activeChunk = data.activeChunk && state.chunks.includes(data.activeChunk)
    ? data.activeChunk : state.chunks[0];
  // Only from an undo snapshot: a *manifest* carries its lighting in the
  // `environment` / `editorEnvironment` blocks, which loadLayout() applies
  // through applyEnvironment() instead.
  if (data.lightSets) {
    for (const k of ["editor", "runtime"]) {
      const set = data.lightSets[k];
      if (!set) continue;
      if (Number.isFinite(set.strength)) state.lightSets[k].strength = set.strength;
      if (Number.isFinite(set.exposure)) state.lightSets[k].exposure = set.exposure;
    }
    if (data.toneMapping) setToneMapping(data.toneMapping);
    const live = activeLightSet();
    setEnvIntensity(live.strength);
    setExposure(live.exposure);
    emit("environment");            // the sliders and their readouts follow
  }
  for (const inst of data.instances || []) {
    if (!getModule(inst.module)) {
      console.warn("skipping unknown module", inst.module);
      continue;
    }
    await placeAt(inst.module, Vector3.FromArray(inst.position), {
      id: inst.id, rotation: inst.rotation, scale: inst.scale,
      chunk: inst.chunk, name: inst.name, silent: true,
    });
  }
  applyVisibility();
  hooks.deserializeMarkers(data.markers || []);
  hooks.deserializeColliders(data.colliders || []);
  // Restored before the final applyVisibility(), so an undo puts back exactly
  // what was on screen. Absent in a manifest, which never carries it.
  for (const item of data.hidden || []) {
    const [id, level] = Array.isArray(item) ? item : [item, "hidden"];
    state.hidden.set(String(id), level === "ghost" ? "ghost" : "hidden");
  }
  applyVisibility();
  // The ship's own sim list wins over the tool's config.json: config is what a
  // *new* ship starts from, but a saved one carries the list it was authored
  // against, and the two drifting apart would silently repoint its behaviours.
  if (Array.isArray(data.fluidSim)) state.fluidSim = [...data.fluidSim];
  for (const [name, body] of Object.entries(data.behaviors || {})) {
    const key = String(name || "").trim();
    if (!key || !body || typeof body !== "object" || Array.isArray(body)) continue;
    state.behaviors.set(key, JSON.parse(JSON.stringify(body)));
  }
  // A manifest written before `entities` existed keyed `behaviors` by NODE
  // name, so each entry meant "this node has these flags". Keeping the bodies
  // as definitions but dropping the application would silently un-liquefy the
  // ship, so apply each one to the node it was named after. Detected by the
  // key being absent, not empty: serialize() always writes both.
  if (!("entities" in data)) {
    for (const name of state.behaviors.keys()) {
      state.entities.set(name, [{ name, linked: [] }]);
    }
  }
  for (const [node, list] of Object.entries(data.entities || {})) {
    const key = String(node || "").trim();
    if (!key || !Array.isArray(list?.behaviors ?? list)) continue;
    // accept both the manifest's { behaviors: [...] } and a bare array
    const raw = Array.isArray(list) ? list : list.behaviors;
    const kept = [];
    for (const b of raw) {
      if (!b || typeof b !== "object") continue;
      // A hand-edit that put `direction` in a sibling entry of its own:
      //   [ { name: "player_startpos" }, { direction: [...] } ]
      // An entry with no name means nothing to the runtime, and folding it into
      // the one above is the only reading under which it means anything at all.
      if (!b.name) {
        const prev = kept[kept.length - 1];
        if (prev) {
          console.warn(`merged a nameless behaviour entry on "${key}" into "${prev.name}"`, b);
          Object.assign(prev, readBehaviorExtras(b));
        }
        continue;
      }
      if (!state.behaviors.has(b.name)) continue;
      kept.push({ name: b.name, linked: [], excludeSDF: [], ...readBehaviorExtras(b) });
    }
    if (kept.length) state.entities.set(key, kept);
  }
  emit("chunks");
  emit("placements");
  emit("markers");
  emit("behaviors");
  emit("selection");
}

let restoring = false;

/**
 * The manifest and the undo snapshot are both glTF space; the editor is not.
 * One helper for both directions - the transform is its own inverse - so the
 * two can never drift apart.
 */
function flipX(v) { return [-Number(v[0]), Number(v[1]), Number(v[2])]; }

/** The optional per-entity fields of an applied behaviour, validated. */
function readBehaviorExtras(b) {
  const out = {};
  if (Array.isArray(b.linked)) out.linked = b.linked.map(String);
  if (Array.isArray(b.excludeSDF)) out.excludeSDF = b.excludeSDF.map(String);
  if (Array.isArray(b.direction) && b.direction.length === 3
    && b.direction.every(Number.isFinite)) {
    out.direction = flipX(b.direction);
  }
  return out;
}

/** True while an undo/redo/load is rebuilding the scene - for tests. */
export function isRestoring() { return restoring; }

// ------------------------------------------------------------------- busy
//
// A load rebuilds the whole scene asynchronously - every module is fetched and
// instantiated one at a time - so there is a long window in which the ship is
// half there. An edit landing in that window acts on a scene that does not
// exist yet: it can select an id that is about to be recreated, drag a wall
// that is a frame away from being disposed, or push an undo snapshot of a
// half-restored ship. Nothing about that is recoverable, so input is shut off
// for the duration rather than defended against case by case.
//
// A counter, not a flag: loads nest (the boot autoload runs inside the same
// turn as the catalogue load) and the first one to finish must not reopen the
// door on the others.

let busyDepth = 0;
let busyMessage = "";

export function isBusy() { return busyDepth > 0; }
export function busyLabel() { return busyMessage; }

/** Run `fn` with the editor locked. Always unlocks, including on a throw. */
export async function whileBusy(message, fn) {
  busyDepth++;
  busyMessage = message || busyMessage || "working…";
  emit("busy");
  try {
    return await fn();
  } finally {
    busyDepth--;
    if (!busyDepth) busyMessage = "";
    emit("busy");
  }
}

/**
 * How much history to keep.
 *
 * A count on its own is the wrong knob. Measured on the real ship: a snapshot
 * is 13.5 KB (145 bytes an instance), so the old 80-deep stack held **1.1 MB**
 * - nothing beside the 11 MB of glb already in the page, and a cap that low
 * meant typing a few numbers into the inspector could push a real edit off the
 * bottom. But the same 80 on a ship ten times the size is a different number
 * entirely, and any count generous enough for this ship is reckless for that
 * one.
 *
 * So the budget is what is capped, and the count is only a rail against a
 * near-empty layout keeping half a million entries. In practice: this ship gets
 * the full 1000 steps, a 1000-instance ship ~230, a 5000-instance ship ~45 -
 * deep where it is cheap, bounded where it is not.
 *
 * Chars, not bytes: JS strings are UTF-16, so this is ~64 MB of actual memory.
 */
const HISTORY_MAX_ENTRIES = 1000;
const HISTORY_MAX_CHARS = 32 * 1024 * 1024;

/**
 * Drop the oldest entries until the stack is back inside both limits.
 *
 * A pure function on the array, and exported, because the byte path is
 * unreachable from the UI on a ship small enough to build in a test: 1000
 * snapshots of this ship is 13.5 M chars, well under the budget. The only way
 * to cover it is to hand it a synthetic stack.
 *
 * The `length > 1` guard matters: one snapshot of a ship bigger than the whole
 * budget must still be kept, or undo would silently do nothing on exactly the
 * ships that need it most.
 */
export function trimHistory(stack) {
  let chars = 0;
  for (const s of stack) chars += s.length;
  while (stack.length > HISTORY_MAX_ENTRIES
         || (stack.length > 1 && chars > HISTORY_MAX_CHARS)) {
    chars -= stack[0].length;
    stack.shift();
  }
}

/** The history limits - for tests and diagnostics. */
export function historyLimits() {
  return { entries: HISTORY_MAX_ENTRIES, chars: HISTORY_MAX_CHARS };
}

export function pushUndo() {
  if (restoring) return;
  // The staging area keeps its own history. Its contents are deliberately not
  // in serialize() - they must never reach the ship - so a ship snapshot taken
  // there restores as "no bench at all", which is precisely how Ctrl+Z used to
  // wipe it. A separate stack also means a bench edit does not rebuild 116
  // placements to undo one box.
  if (state.collisionMode) {
    stageUndo.push(JSON.stringify(hooks.serializeStage()));
    trimHistory(stageUndo);
    stageRedo.length = 0;
    return;
  }
  undoStack.push(JSON.stringify(serialize()));
  trimHistory(undoStack);
  redoStack.length = 0;
}

/** Depth of each history stack - for tests and diagnostics. */
export function historyDepth() {
  return state.collisionMode
    ? { undo: stageUndo.length, redo: stageRedo.length }
    : { undo: undoStack.length, redo: redoStack.length };
}

export async function undo() {
  if (state.collisionMode) {
    if (!stageUndo.length) return;
    stageRedo.push(JSON.stringify(hooks.serializeStage()));
    trimHistory(stageRedo);
    await hooks.restoreStage(JSON.parse(stageUndo.pop()));
    return;
  }
  if (!undoStack.length) return;
  redoStack.push(JSON.stringify(serialize()));
  trimHistory(redoStack);
  await deserialize(JSON.parse(undoStack.pop()));
}

export async function redo() {
  if (state.collisionMode) {
    if (!stageRedo.length) return;
    stageUndo.push(JSON.stringify(hooks.serializeStage()));
    trimHistory(stageUndo);
    await hooks.restoreStage(JSON.parse(stageRedo.pop()));
    return;
  }
  if (!redoStack.length) return;
  undoStack.push(JSON.stringify(serialize()));
  trimHistory(undoStack);
  await deserialize(JSON.parse(redoStack.pop()));
}

/** Start the staging area's history clean, so it cannot reach past itself. */
export function resetStageHistory() {
  stageUndo.length = 0;
  stageRedo.length = 0;
}

// ---------------------------------------------------------------- helpers

export function eulerOf(node) {
  const q = node.rotationQuaternion || Quaternion.FromEulerVector(node.rotation);
  const e = q.toEulerAngles();
  return [e.x * 180 / Math.PI, e.y * 180 / Math.PI, e.z * 180 / Math.PI];
}

export function setEuler(node, deg) {
  node.rotationQuaternion = Quaternion.FromEulerAngles(
    deg[0] * Math.PI / 180, deg[1] * Math.PI / 180, deg[2] * Math.PI / 180);
}

function round3(a) { return a.map((v) => Math.round(v * 1000) / 1000); }

export function worldBounds(node) {
  let min = null, max = null;
  for (const m of node.getChildMeshes()) {
    m.computeWorldMatrix(true);
    const bb = m.getBoundingInfo().boundingBox;
    min = min ? Vector3.Minimize(min, bb.minimumWorld) : bb.minimumWorld.clone();
    max = max ? Vector3.Maximize(max, bb.maximumWorld) : bb.maximumWorld.clone();
  }
  return min ? { min, max } : null;
}
