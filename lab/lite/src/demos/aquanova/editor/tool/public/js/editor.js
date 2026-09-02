// Viewport, grid, camera, selection and the placement store.
//
// There are no transform gizmos: authoring goes through the mouse-follow ghost
// in interact.js, so this module only owns scene setup and the data stores.

import { instantiate, getModule } from "./kit.js";
import { patchKhronosPbrNeutralShader } from "./shader-patches.js";
import { behaviorMetadata, behaviorMetadataNames, defaultBehaviorDefinition, renameEntityReferences, validateBehaviorConfig } from "./behavior-metadata.js";

const {
    Engine,
    Scene,
    UniversalCamera,
    HemisphericLight,
    Vector3,
    Color3,
    Color4,
    Quaternion,
    Matrix,
    MeshBuilder,
    HDRCubeTexture,
    ImageProcessingConfiguration,
    PointerEventTypes,
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

/**
 * The chunk the compound bench parks its members in.
 *
 * Same trick, second bench: not in `state.chunks`, and carrying `stage: true`
 * so `shipPlacements()` filters it out. A chunk of its own rather than sharing
 * STAGE_CHUNK because the two benches must be able to tell their own contents
 * apart - `stagedElements()` in colliders.js walks by chunk for exactly that
 * reason.
 */
export const COMPOUND_CHUNK = "__compound_bench";

/**
 * The editing modes, and the world each one edits.
 *
 * `ship` is the ship itself. The other two are benches: their contents are
 * deliberately absent from serialize() and can never reach the ship. Adding a
 * mode means adding it here and giving it a history entry - see `histories`.
 */
export const EDITOR_MODES = ["ship", "collision", "compound"];

/** Every placement that is part of the ship, excluding either bench. */
export function shipPlacements() {
  return [...state.placements.values()].filter((p) => !p.stage);
}

/**
 * The placements of the world currently on screen - the ship, or whichever
 * bench is open.
 *
 * Anything the inspector offers as a *choice* has to come from here rather than
 * from `shipPlacements()`, or a bench asks you to pick from a list of things
 * you cannot see. The light panel's Owner row was the case that found it:
 * fitting a lamp to a bench member listed the whole ship and not the piece the
 * lamp was actually on.
 */
export function modePlacements() {
  if (state.mode === "ship") return shipPlacements();
  const chunk = state.mode === "compound" ? COMPOUND_CHUNK : STAGE_CHUNK;
  return [...state.placements.values()].filter((p) => p.stage && p.chunk === chunk);
}

/**
 * Which bench a placement is parked on - "" for ship geometry.
 *
 * The two benches want opposite answers in a few places, and `stage` alone
 * cannot tell them apart. Lights are the sharpest case: a lamp on the collision
 * bench is meaningless and is refused, while a lamp on the compound bench is
 * the entire reason the feature exists.
 */
export function benchOf(entry) {
  if (!entry?.stage) return "";
  return entry.chunk === COMPOUND_CHUNK ? "compound" : "collision";
}

/** The chunk a bench parks things in, for whichever bench is open. */
function defaultBenchChunk() {
  return state.mode === "compound" ? COMPOUND_CHUNK : STAGE_CHUNK;
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
  // How far a fitted hull may stray from the art before that counts as wrong,
  // in metres. The single dial for how finely Fit a hull approximates a shape:
  // a rounded prop worth one box at 20 cm is worth several at 3 cm.
  hullTolerance: 0.1,
  // The depth a fitted hull is given along its thinnest axis. The kit's walls
  // are millimetres thick and a hull that thin is something a fast-moving body
  // goes straight through.
  hullThickness: 0.35,
  // Where that depth goes relative to the art: "centered" splits it either
  // side, "negative" tucks it behind the visible surface (the convention this
  // ship uses), "positive" stands it in front.
  hullOffset: "centered",
    // Face width and height of every local cubemap, in pixels. One number for
    // the whole ship rather than one per probe: the runtime keeps the captured
    // environments in a cube texture ARRAY, and an array has a single dimension
    // for all of its slices - a probe captured at another size could not be put
    // in it. So it is a ship-wide constant, and changing it makes every probe
    // stale at once.
    probeResolution: 512,
};

/** The cubemap sizes the Settings pane offers, smallest first. */
export const PROBE_RESOLUTIONS = [128, 256, 512, 1024, 2048];

/**
 * What each setting will accept.
 *
 * A thickness of zero is a shape with no shape; an auto-save interval of zero
 * is a perfectly reasonable "don't". One rule for both would have to be wrong
 * for one of them. A setting with `choices` is a word rather than a number,
 * and is checked against the list instead of a range; one with `values` is a
 * number off a list, for a dial that has no meaningful values in between.
 */
const CONFIG_RANGE = {
  shellThickness: { min: 1e-4, max: 10 },
  autoSaveMinutes: { min: 0, max: 240 },
  hullTolerance: { min: 0.01, max: 1 },
  hullThickness: { min: 0.01, max: 5 },
  hullOffset: { choices: ["centered", "negative", "positive"] },
    probeResolution: { values: PROBE_RESOLUTIONS },
};

const MOVE_SPEED = 5;       // metres per second, the starting fly speed

export const state = {
  scene: null,
  engine: null,
  camera: null,
  placements: new Map(),   // id -> { id, module, chunk, node }
  markers: new Map(),      // id -> door / spawn marker (see markers.js)
  colliders: new Map(),    // id -> collision primitive (see colliders.js)
  lights: new Map(),       // id -> authored light, riding a placement (see lights.js)
  chunks: ["CH00_Storage"],
  activeChunk: "CH00_Storage",
  // Explicit local environments, independent from render chunks. A single
  // volume may cover several chunks or only part of one.
  environmentProbes: new Map(),
  selection: [],
  brush: null,             // module id armed for placement
  markerBrush: null,       // "door" - the only marker kind left
  snap: { pos: 1, rot: 90, scale: 0.1 },
  gridY: 0,                // elevation of the build plane
  rotAxis: "y",            // axis Q/E turn around
  scaleAxis: "all",        // axis Ctrl+wheel scales
  isolate: false,
  hidden: new Map(),       // id -> "ghost" | "hidden", see hideSelected
  veilAlpha: 0.5,          // how see-through a ghosted element is, see VEIL_ALPHA_DEFAULT
  bigPalette: true,        // double-width palette with double-size tiles, see BIG_PALETTE_DEFAULT
  // Warn about elements that look like they were left in the wrong chunk. On by
  // default: a mis-assigned piece is invisible in the viewport and only shows up
  // as a hole in the portal graph much later. See strayChunkMembers.
  strayChunkCheck: true,
  // Play the authored animations of elements carrying `playAnimation`, in the
  // runtime view only. On by default: the runtime view claims to be what the
  // game draws, and a game whose fans are stopped is not that. See
  // syncBehaviorAnimations in runtime.js and RUN_BEHAVIORS_DEFAULT.
  runBehaviors: true,
  // Whether **Start demo** compresses the ship on the way out. Off by default:
  // the KTX2/Meshopt pass is minutes of toktx over every texture, and a publish
  // is usually there to look at the wall you just moved. See
  // SHIP_OPTIMIZE_DEFAULT and doStartDemo in main.js.
  shipOptimize: false,
    behaviors: new Map(), // preset name -> { base, ...parameter overrides }
  entities: new Map(),     // node name -> [{ name, linked: [] }]
  fluidSim: [],            // the global sim list from config.json
  // How the viewport is lit. The two together spell one of the three named
  // view modes - see VIEW_MODES and viewMode() - and nothing else sets them.
  unlit: false,            // show raw albedo, no lighting
  runtime: false,          // the runtime preview is on, see runtime.js
  runtimeSpecularAA: true, // Babylon.js specular anti-aliasing, see RUNTIME_SPECULAR_AA_DEFAULT
  runtimeRoughnessFactor: 1, // multiplier over authored metallic roughness, see RUNTIME_ROUGHNESS_FACTOR_DEFAULT
  toneMapping: "Khronos PBR Neutral",   // the active set's, mirrored — see lightSets
  // Two independent pairs. The editor's rig adds four analytic lights the game
  // does not have, so one pair of Env/Exposure values cannot serve both: what
  // reads well while building is nothing like what the game needs. A runtime
  // view mode picks the runtime set; the active one is mirrored into
  // envIntensity/exposure/toneMapping above, which is what the scene gets.
  lightSets: {
    editor: { strength: 1.5, exposure: 0.55, toneMapping: "Khronos PBR Neutral" },
    runtime: { strength: 1.5, exposure: 0.55, toneMapping: "Khronos PBR Neutral" },
  },
  exposure: 0.55,          // see EXPOSURE_DEFAULT
  envIntensity: 1.5,       // see ENV_INTENSITY_DEFAULT
  walk: false,             // see setWalk
  selectMode: false,       // LMB draws a selection rectangle, see setSelectMode
  moveSpeed: MOVE_SPEED,   // m/s; right button + wheel adjusts it
  dragAxis: "xz",          // "xz" | "y" | "x" | "z" - which axis a move runs on (V)
  axisSpace: "world",      // "world" | "local" - whose axes a move or turn uses (Y)
  // Which world the editor is currently editing - see EDITOR_MODES. One string
  // rather than a flag per bench: the modes are mutually exclusive by nature
  // (there is one camera and one viewport), and a flag apiece would let two of
  // them be true at once, which nothing downstream could make sense of.
  mode: "ship",
  // module id -> shapes authored on it, in the module's own local space. The
  // one authoritative record: what is on the staging area is a working copy.
  moduleCollision: new Map(),
  // What was on the staging area when it was last closed, so re-opening it
  // finds the same modules in the same places rather than a blank stage.
  stageLayout: [],
  showLayer: "geometry",   // "both" | "geometry" | "collision", see setShowLayer
  config: { ...CONFIG_DEFAULTS },
  nextId: 1,
  nextGroup: 1,            // compound instance ids, see nextGroupId
};

/** Selection holds ids from any store; resolve without caring which. */
export function entryOf(id) {
    return state.placements.get(id) || state.markers.get(id) || state.colliders.get(id) || state.lights.get(id) || hooks.environmentProbeEntry(id) || null;
}

/**
 * Which element a mesh belongs to, whoever is drawing it.
 *
 * Every editor mesh carries a back-pointer to the node it hangs off.
 *
 * `placementsOnly` is for the walk-mode floor probe, which wants something to
 * stand on and not a door marker's portal quad.
 */
export function ownerIdOf(mesh, placementsOnly = false) {
  const md = mesh?.metadata;
  if (!md) return null;
    const id = md.placementRoot?.name || (placementsOnly ? null : md.markerRoot?.name || md.colliderRoot?.name || md.lightRoot?.name || md.environmentProbeRoot?.metadata?.probe);
  return id || null;
}

// ------------------------------------------------------------- compounds
//
// A compound is a *recipe*, not a mesh. Placing one expands it into ordinary
// placements that share a `group` id and remember the `compound` they came
// from, and nothing downstream - the manifest, the .glb exporter, the runtime -
// ever learns that compounds exist.
//
// That is deliberate and load-bearing. The case the feature was asked for is a
// wall *with its lamp*, and a lamp is a `state.lights` entry rather than
// geometry, so a compound could never have been a baked .glb in the first
// place. Recipes also keep each member's own authored collision hull, which
// baking would have thrown away, and they make "delete one component" and
// "break apart" fall out for free rather than needing a rebuild path each.

/** Which compound instance a placement belongs to, if any. */
export function groupOf(id) {
  return state.placements.get(id)?.group || "";
}

/** Every placement in one compound instance. */
export function groupMembers(group) {
  if (!group) return [];
  return [...state.placements.values()].filter((p) => p.group === group);
}

/**
 * Grow a set of ids so that touching one member of a compound touches all of it.
 *
 * Selection is where this happens, and it is the only place it happens: once
 * the whole compound is in `state.selection`, every multi-selection path the
 * editor already has - drag, grab, rotate, scale, mirror, arrow nudge, Ctrl+D -
 * moves it as one piece without knowing what a compound is. Drilling into a
 * single member is then simply *not* calling this.
 */
export function groupExpand(ids) {
  const out = new Set();
  for (const id of ids) {
    out.add(id);
    for (const m of groupMembers(groupOf(id))) out.add(m.id);
  }
  return [...out];
}

/**
 * The member a compound turns about, when the whole of one is selected.
 *
 * A compound is placed relative to its **first** member - that is the piece the
 * recipe's coordinates are measured from - so that is the piece it should turn
 * about too. Anything else and a compound would come back from a quarter turn
 * somewhere other than where it was dropped, and four quarter turns would not
 * be the identity.
 *
 * Null unless the selection is exactly one whole compound: a mixed bag, or half
 * of one, has no anchor and falls back to the ordinary rules.
 */
export function groupAnchor(entries) {
    const list = entries.map((e) => (typeof e === "string" ? state.placements.get(e) : e)).filter(Boolean);
  if (list.length < 2) return null;
  const group = list[0].group;
  if (!group || list.some((e) => e.group !== group)) return null;
  const members = groupMembers(group);
  if (members.length !== list.length) return null;
  return members[0];
}

/**
 * Compound instances are numbered like placements, and for the same reason: an
 * id reading `G0007` in a diff is worth far more than a random token on the day
 * a member turns up in the wrong group.
 */
export function nextGroupId() {
  return `G${String(state.nextGroup++).padStart(4, "0")}`;
}

/** Keep the counter ahead of any id a file brought with it. */
function noteGroupId(group) {
  const n = parseInt(String(group).replace(/\D/g, ""), 10);
  if (Number.isFinite(n) && n >= state.nextGroup) state.nextGroup = n + 1;
}

/**
 * Dissolve every compound the given ids touch into ordinary placements.
 *
 * Only the two fields are cleared: the members are already independent
 * placements with their own transform, chunk, lights and collision, so from
 * here on they behave exactly as if they had been dropped one by one. Undoable
 * like any other edit, because rebuilding a compound by hand is the only way
 * back otherwise.
 */
export function breakApart(ids = state.selection) {
  const groups = new Set(ids.map(groupOf).filter(Boolean));
  if (!groups.size) return { groups: 0, members: 0 };
  pushUndo();
  let members = 0;
  for (const g of groups) {
    for (const m of groupMembers(g)) {
      m.group = "";
      m.compound = "";
      members++;
    }
  }
  emit("placements");
  emit("selection");
  return { groups: groups.size, members };
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

// ------------------------------------------------------------------ setup

export async function initScene(canvas) {
  patchKhronosPbrNeutralShader();
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
    cam.keysUp = [];
    cam.keysDown = [];
    cam.keysLeft = [];
    cam.keysRight = [];
    cam.keysUpward = [];
    cam.keysDownward = [];
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
  hemi.groundColor = new Color3(0.2, 0.23, 0.28);
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
export const RUNTIME_SPECULAR_AA_DEFAULT = true;
export const RUNTIME_ROUGHNESS_FACTOR_DEFAULT = 1;
export const TONE_MAPPING_DEFAULT = "Khronos PBR Neutral";
export const VEIL_ALPHA_DEFAULT = 0.5;
export const BIG_PALETTE_DEFAULT = true;
/** See `state.strayChunkCheck` and `strayChunkMembers`. */
export const STRAY_CHUNK_CHECK_DEFAULT = true;
/** See `state.runBehaviors` and `syncBehaviorAnimations` in runtime.js. */
export const RUN_BEHAVIORS_DEFAULT = true;
/**
 * See `state.shipOptimize` and `doStartDemo` in main.js.
 *
 * Off, because the alternative is waiting minutes for a texture compressor
 * every time you want to walk the corridor you just changed. Turn it on when
 * what you are checking *is* the shipped asset - load time, memory, banding.
 */
export const SHIP_OPTIMIZE_DEFAULT = false;

/**
 * The three ways of looking at the ship.
 *
 * These were three independent checkboxes whose eight combinations included
 * four that mean nothing and were only kept sane by disabling boxes from other
 * boxes' handlers. Naming the states that actually exist is both smaller and
 * impossible to put in a position nobody designed.
 *
 * `runtime` is the load-bearing flag: it says the viewport is lighting the ship
 * the way the game does - the editor's authoring rig switched off, the authored
 * lamps rebuilt over every mesh, the captured environment probes supplying the
 * image-based half, and the runtime Env/Exposure pair in charge.
 *
 *   * `editor`       - the authoring rig, for building in.
 *   * `editor-unlit` - raw albedo, for reading a module's own colours.
 *   * `runtime`      - what the game renders.
 */
export const VIEW_MODES = {
    editor: { runtime: false, unlit: false },
  "editor-unlit": { runtime: false, unlit: true },
    runtime: { runtime: true, unlit: false },
};

export const VIEW_MODE_DEFAULT = "editor";

/** The flags a mode stands for, or the plain editor's when the name is unknown. */
export function viewModeFlags(mode) {
  return VIEW_MODES[mode] || VIEW_MODES[VIEW_MODE_DEFAULT];
}

/**
 * Which mode the viewport is in, read back off the flags rather than stored.
 *
 * Deliberately derived: a remembered `state.viewMode` beside the flags it
 * stands for is two truths about one thing, and the moment anything sets a flag
 * on its own - setRuntimePreview does, on both edges, and so does every test
 * that pokes `state.runtime` - the two would disagree and the combo box would be
 * the one lying.
 */
export function viewMode() {
  if (state.runtime) return "runtime";
  return state.unlit ? "editor-unlit" : "editor";
}

// -------------------------------------------------------- the authoring rig
//
// The four analytic lights are an authoring aid: they exist so every module
// stays legible from any angle while you build. The game has none of them - it
// lights the ship from the authored runtime lamps and the captured environment
// probes - so they are measurably the reason the editor looks nothing like the
// runtime. Measured on the real ship: mean 165.7 with the rig, 35.0 without, so
// the rig supplies about four fifths of what you see here and none of what you
// will ship.
//
// The rig is therefore tied to **Runtime**, and to nothing else. There used to
// be a separate "Runtime light" checkbox that silenced the rig but changed
// nothing else, which was only ever half a truth - the ship was then lit by the
// HDRI alone, a picture the game never renders either. Runtime mode shows the
// real thing (the authored lamps over every mesh, each room's own probe on its
// materials), so it is the only state in which silencing the rig means
// anything, and the checkbox was one more thing to get wrong.
//
// The two rigs survive the merge unchanged: `runtime` is what the manifest's
// `environment` block carries and the demos read, `editor` is what the rig
// wants, and the view mode decides which one the scene renders through. Both
// are on the Settings pane at once, so neither has to be guessed at.

const authoredIntensity = new WeakMap();

function applyRuntimeLighting() {
  if (!state.scene) return;
  for (const l of state.scene.lights) {
    if (!authoredIntensity.has(l)) continue;
    // The runtime preview supplies its own clustered/regular lights. The editor
    // rig must be disabled, not merely set to zero: enabled rig lights still
    // consume Babylon material light slots and can evict the clustered
    // container from the compiled shader.
    l.setEnabled(!state.runtime);
    l.intensity = state.runtime ? 0 : authoredIntensity.get(l);
  }
}

/**
 * Put the viewport lighting in step with `state.runtime`.
 *
 * Called by setRuntimePreview on both edges, after it has moved the flag. Kept
 * separate from that function rather than folded into it because the Env and
 * Exposure sliders have to be refreshed by the caller afterwards, and a mode
 * switch that half-applied itself would be worse than one that did nothing.
 */
export function syncLightingMode() {
  applyRuntimeLighting();
  // Swap in the pair that belongs to this mode. Both are kept whole, so
  // switching back and forth never costs you the values you tuned.
  const set = activeLightSet();
  setEnvIntensity(set.strength);
  setExposure(set.exposure);
  setToneMapping(set.toneMapping ?? TONE_MAPPING_DEFAULT);
  emit("modes");
}

/** Which of the two sets the viewport is currently rendering through. */
export function activeLightSetName() {
  return state.runtime ? "runtime" : "editor";
}

/** Clamp one lighting value the way its own setter would. */
function lightSettingValue(key, value) {
  const n = Number(value);
  if (key === "toneMapping") return String(value || TONE_MAPPING_DEFAULT);
  if (key === "exposure") return Number.isFinite(n) ? Math.min(4, Math.max(0.15, n)) : EXPOSURE_DEFAULT;
  return Number.isFinite(n) ? Math.min(6, Math.max(0, n)) : ENV_INTENSITY_DEFAULT;
}

/**
 * Edit one lighting set by name, whether or not it is the one on screen.
 *
 * The Env/Exposure/Tone controls used to be a single row that silently meant
 * whichever set the view mode had selected, so the same slider was two
 * settings and you could not see the other one without changing what you were
 * looking at. The Settings pane now shows both sets at once, which needs a way
 * to write to the set you are *not* rendering: that is this. Only the active
 * set reaches the scene; the other takes effect when its mode is entered,
 * through syncLightingMode().
 */
export function setLightSetting(which, key, value) {
  const set = state.lightSets[which];
  if (!set || !(key in set)) return false;
  const next = lightSettingValue(key, value);
  if (set[key] === next) return false;
  set[key] = next;
  if (which !== activeLightSetName()) {
    // Nothing to apply - the set is not the one being rendered - but the pane
    // shows both at once and has to keep up with the one it is not showing.
    emit("environment");
    return true;
  }
  if (key === "strength") setEnvIntensity(next);
  else if (key === "exposure") setExposure(next);
  else if (key === "toneMapping") setToneMapping(next);
  return true;
}

export function setEnvIntensity(v) {
  const n = Number(v);
    state.envIntensity = Number.isFinite(n) ? Math.min(6, Math.max(0, n)) : ENV_INTENSITY_DEFAULT;
  activeLightSet().strength = state.envIntensity;
  // The runtime preview drives Env per material - each mesh reflects its own
  // room's probe, and the strength has to ride the material that holds it - so
  // Babylon's scene-wide multiplier is kept neutral there.
  if (state.scene) {
    state.scene.environmentIntensity = state.runtime ? 1 : state.envIntensity;
  }
  emit("environment");
}

/** Toggle Babylon.js PBR specular anti-aliasing on the ship's materials. */
export function setRuntimeSpecularAA(on) {
  const next = !!on;
  if (state.runtimeSpecularAA === next) return false;
  state.runtimeSpecularAA = next;
  emit("reflection");
  return true;
}

/**
 * Scale the authored roughness of the ship's materials.
 *
 * The exported .glb keeps the authored values - this is a manifest setting the
 * game applies on load, exactly as the preview does, so the two agree without
 * rewriting any material. Values above 1 deliberately make glossy metal less
 * likely to shimmer by broadening its IBL lobe.
 */
export function setRuntimeRoughnessFactor(v) {
  const n = Number(v);
    const next = Number.isFinite(n) ? Math.min(2, Math.max(0.5, n)) : RUNTIME_ROUGHNESS_FACTOR_DEFAULT;
  if (state.runtimeRoughnessFactor === next) return false;
  state.runtimeRoughnessFactor = next;
  emit("reflection");
  return true;
}

export function setExposure(v) {
  const n = Number(v);
  // Number.isFinite, not `|| DEFAULT`: a literal 0 is falsy and would silently
  // jump back to the default instead of clamping to the floor.
  //
  // The ceiling is 4 rather than 2 because of the runtime view. There the ship
  // is lit by its own lamps alone, which are dim enough that the picture needs
  // lifting well past the 0.55 the authoring rig wants. Exposure is the only
  // honest knob for that - the alternative is more lamp power, which is a
  // different decision.
    state.exposure = Number.isFinite(n) ? Math.min(4, Math.max(0.15, n)) : EXPOSURE_DEFAULT;
  activeLightSet().exposure = state.exposure;
  if (state.scene) state.scene.imageProcessingConfiguration.exposure = state.exposure;
  emit("modes");
}

/** The Env/Exposure pair the viewport is rendering through. */
export function activeLightSet() {
  return state.lightSets[activeLightSetName()];
}

/**
 * Tone mapping by name, matched the way the runtime matches it - loosely, so
 * "Khronos PBR Neutral", "neutral" and "KHR_PBR_NEUTRAL" all land in the same
 * place. Returning null means "no tone mapping at all", which is a real answer
 * and not a failure.
 */
export function resolveToneMapping(name) {
    const key = String(name ?? "")
        .toLowerCase()
        .replace(/[^a-z]/g, "");
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
  state.toneMapping = String(name || TONE_MAPPING_DEFAULT);
  activeLightSet().toneMapping = state.toneMapping;
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
  // The runtime preview's per-probe clones do their own compositing and must be
  // left out of this sweep: their PBR lighting is intentional, and the scene
  // environment they carry is their own room's probe rather than the global
  // one. They are preview-only and never reach the export, so nothing
  // downstream needs them swept either.
  if (mat.metadata?.runtimePreview) return;
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
    "albedoTexture",
    "diffuseTexture",
    "ambientTexture",
    "opacityTexture",
    "reflectionTexture",
    "emissiveTexture",
    "reflectivityTexture",
    "specularTexture",
    "metallicTexture",
    "microSurfaceTexture",
    "bumpTexture",
    "lightmapTexture",
    "refractionTexture",
    "metallicReflectanceTexture",
    "detailMap",
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
            if (orig && dup && dup !== orig) {
                dup.dispose();
                g[slot] = orig;
            }
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
    let minX = Infinity,
        minY = Infinity,
        maxX = -Infinity,
        maxY = -Infinity;
  for (let i = 0; i < 8; i++) {
        const c = new Vector3(i & 1 ? b.max.x : b.min.x, i & 2 ? b.max.y : b.min.y, i & 4 ? b.max.z : b.min.z);
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
 * The element's outline on screen, as a convex polygon.
 *
 * The corners of each of its meshes' *oriented* boxes, projected and hulled.
 * Two approximations are dropped here, and both matter:
 *
 * `screenBoundsOf` squares the result off into an axis-aligned rectangle,
 * which for anything seen at an angle is enormously bigger than the thing
 * itself - a slab lying diagonally across the view has a screen box covering
 * the viewport corner to corner, nearly all of it empty air. That is fine for
 * finding a point to aim at, and useless for asking what a rectangle caught.
 *
 * And `worldBounds` is *axis-aligned in world space*, so a collision box that
 * has been turned to follow a curve reports the upright box that contains it.
 * `vectorsWorld` is the box's own eight corners, which for a box collider is
 * the shape exactly.
 */
export function screenHullOf(node) {
  const scene = state.scene;
  if (!scene || !state.camera) return null;
  const engine = scene.getEngine();
  const vp = state.camera.viewport.toGlobal(engine.getRenderWidth(), engine.getRenderHeight());
  const view = scene.getTransformMatrix();
  const pts = [];
  for (const m of node.getChildMeshes()) {
    if (isRuntimeStandIn(m)) continue;
    m.computeWorldMatrix(true);
    for (const c of m.getBoundingInfo().boundingBox.vectorsWorld) {
      const p = Vector3.Project(c, Matrix.Identity(), view, vp);
      // behind the camera projects to a mirrored point that would wreck the hull
      if (p.z < 0 || p.z > 1) continue;
      pts.push([p.x, p.y]);
    }
  }
  if (pts.length < 3) return pts.length ? pts : null;

  // monotone chain; a handful of points, so the sort costs nothing worth saving
  pts.sort((u, v) =>u[0] - v[0] ||u[1] - v[1]);
  const cross = (o, a, c) =>
    (a[0] - o[0]) * (c[1] - o[1]) - (a[1] - o[1]) * (c[0] - o[0]);
  const half = (src) => {
    const h = [];
    for (const p of src) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop();
      h.push(p);
    }
    h.pop();
    return h;
  };
  const hull = [...half(pts), ...half([...pts].reverse())];
  return hull.length >= 3 ? hull : pts;
}

/** Does a convex polygon meet an axis-aligned rect? Separating axis, both ways. */
function hullMeetsRect(hull, rect) {
  if (!hull || !hull.length) return false;
  if (hull.length < 3) {
        return hull.some(([x, y]) => x >= rect.x0 && x <= rect.x1 && y >= rect.y0 && y <= rect.y1);
  }
  // the rect's own two axes
    let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
  for (const [x, y] of hull) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (maxX < rect.x0 || minX > rect.x1 || maxY < rect.y0 || minY > rect.y1) return false;

  // and one per hull edge
    const corners = [
        [rect.x0, rect.y0],
        [rect.x1, rect.y0],
        [rect.x1, rect.y1],
        [rect.x0, rect.y1],
    ];
  for (let i = 0; i < hull.length; i++) {
        const a = hull[i],
            b = hull[(i + 1) % hull.length];
        const ax = -(b[1] - a[1]),
            ay = b[0] - a[0];
        let hLo = Infinity,
            hHi = -Infinity,
            rLo = Infinity,
            rHi = -Infinity;
    for (const [x, y] of hull) {
      const d = x * ax + y * ay;
      if (d < hLo) hLo = d;
      if (d > hHi) hHi = d;
    }
    for (const [x, y] of corners) {
      const d = x * ax + y * ay;
      if (d < rLo) rLo = d;
      if (d > rHi) rHi = d;
    }
    if (hHi < rLo || rHi < hLo) return false;
  }
  return true;
}

/**
 * Every element the given canvas-space rect actually touches.
 *
 * Against the element's *outline*, not the rectangle its outline sits in. A
 * screen-space box round a slab lying diagonally across the view is mostly
 * empty space, and testing that meant a small rectangle dropped in a gap
 * selected everything around it — one drawn in clear air next to a corner hull
 * picked up all three of its boxes.
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
    return hullMeetsRect(screenHullOf(node), rect);
  };
  for (const e of state.placements.values()) if (overlaps(e.node)) out.push(e.id);
  for (const m of state.markers.values()) if (overlaps(m.node)) out.push(m.id);
  for (const c of state.colliders.values()) if (overlaps(c.node)) out.push(c.id);
  for (const id of environmentProbeIds()) {
    const probe = hooks.environmentProbeEntry(id);
    if (probe && overlaps(probe.node)) out.push(id);
  }
  return out;
}

/**
 * Height of the walkable surface under `x, z`, or null if there is none.
 *
 * `skip` drops meshes from the search - used when the thing being placed is
 * itself in the way, and would otherwise report its own roof as the floor.
 */
export function groundHeightAt(x, z, fromY, reach = FALL_LIMIT, skip = null) {
  const scene = state.scene;
  if (!scene) return null;
  const ray = new BABYLON.Ray(new Vector3(x, fromY, z), Vector3.Down(), reach);
    const hit = scene.pickWithRay(ray, (m) => m.isPickable && m.isEnabled() && !!ownerIdOf(m, true) && !(skip && skip(m)));
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
    const bucket = i === 0 ? axis :i % GRID_MAJOR === 0 ? major : minor;
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
  mk("grid_major", major, new Color3(0.26, 0.3, 0.36));
  mk("grid_axis", axis, new Color3(0.55, 0.36, 0.18));
}

export function setGridVisible(v) {
    gridNode.setEnabled(v);
}

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
//   cube on the tip the axis a scale acts on        (F)
//
// and two DOM chips carry the *values*: the move step at the origin, the turn
// angle inside the curved arrow.

export const AXIS_COLOR = {
  x: new Color3(0.9, 0.22, 0.27),
  y: new Color3(0.36, 0.78, 0.34),
  z: new Color3(0.26, 0.52, 0.96),
};

/**
 * How far the axis gizmo's arms reach, in metres. One number for every element.
 *
 * The length used to be taken from the element's own bounding span, clamped
 * between 1.5 m and 8 m. The intent was that the arrows should suit the thing
 * they describe, but the effect was that they never meant the same thing twice:
 * a crate and a corridor got gizmos more than five times apart, so the arrows
 * could not be read as a measure of anything, and stepping between two selected
 * elements resized the whole gizmo under the cursor.
 *
 * A fixed length makes it an instrument instead - the same ruler held up to
 * whatever is selected, and a constant reference for how far a snap step moves
 * something. Every thickness in `buildAxes` is a multiple of this, so the whole
 * gizmo is retuned by this one number.
 */
const AXIS_GIZMO_LENGTH = 2;

/**
 * The step value each of the three lists calls `free`.
 *
 * Filled from the markup by `main.js`, which owns the combos - the same reason
 * `cycleSnap` reads its values off the select rather than repeating them in
 * code: the list is declared in one place, and adding an option there is the
 * whole change. `null` means nobody has said, which only happens before the
 * toolbar is wired.
 *
 * The gizmo chips need it because `free` is not the same number on all three.
 * Move's is a real zero, so "no step" and "free" coincide; Rot's and Scale's
 * are the *finest* step there is - `0.5°` and `0.01` - because a wheel notch is
 * discrete and a step of nothing would simply do nothing. Without this the
 * toolbar read `free` while the chip on the arrow read `0.01`, and the two
 * looked like different settings.
 */
export const freeSnap = { pos: null, rot: null, scale: null };

/**
 * How close the camera has to get, in metres, before the gizmo gives ground,
 * and how much of its length is left when it does — measured to the point the
 * gizmo hangs on, which is the thing you are looking at.
 *
 * A fixed length is right across the working range, but the arms are world
 * geometry and the camera is not held at a polite distance: leaning in to seat
 * a light against a wall puts the eye a couple of metres off the model, where
 * 2 m of arrow fills the viewport and buries the very detail being aimed. Close
 * in there is also less need for reach — nothing else is on screen to measure
 * against — so the arms give way rather than the model.
 *
 * Steps rather than a ramp, deliberately: at any distance the gizmo is at one
 * of a few stated fractions of its length, so the arms stay a ruler you can
 * read the snap step off. A continuous falloff would make every arm a
 * different, unknowable length again, which is the problem this table exists to
 * avoid. Each tier halves the one outside it, so the fractions stay easy to
 * hold in your head and an arm is always a clean multiple of the last.
 *
 * The tightest matching tier wins, so the order here does not matter.
 */
const AXIS_GIZMO_NEAR_STEPS = [
  { within: 5, scale: 0.5 },
  { within: 2, scale: 0.25 },
];

/** Which axes a drag currently moves along. */
export function liveAxes(mode = state.dragAxis) {
  return mode === "y" ? ["y"] : mode === "x" ? ["x"] : mode === "z" ? ["z"] : ["x", "z"];
}

/**
 * The frame a move or a turn runs in: the world's axes, or the given element's
 * own.
 *
 * `axisSpace` says whose axes the *move axis* and the *rotation axis* mean. In
 * world space "X only" slides along world X and a turn goes about world Y; in
 * local space they use the element's own, so a wall turned 90 degrees slides
 * along its length rather than across it, and a tilted panel turns about its
 * own edge. Null means the world, which every caller treats as "no basis".
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
 * A turn is the exception, and takes each element's own axis, because it
 * already turns each about its own origin.
 */
export function axisBasis(node) {
  return state.axisSpace === "local" ? nodeBasis(node) : null;
}

/**
 * An element's own three axes, in world space, whatever space the tools are in.
 *
 * Separate from `axisBasis` because the two answer different questions.
 * `axisBasis` asks "whose axes should this gesture use?", and in world space the
 * answer is nobody's - hence the null. This asks "which way does this element's
 * own X point?", and that has an answer in either mode. A world-space *scale*
 * needs both at once: the axis comes from the world, but a node can only be
 * scaled along its own axes, so the world axis has to be matched against these
 * to find which of the element's three numbers to touch.
 */
export function nodeBasis(node) {
  if (!node || node.isDisposed()) return null;
  // Forced, not read from the cache: the frame is taken once at the start of a
  // gesture, and a turn earlier in the same frame - R, the inspector, a load -
  // has not been through a render yet, so the cached matrix still holds the
  // rotation before it. That put the axes one turn behind.
  const m = node.computeWorldMatrix(true);
  const out = {};
    for (const [a, row] of [
        ["x", 0],
        ["y", 1],
        ["z", 2],
    ]) {
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

let axes = null;      // { root, id, space, anchor, arms, mats, marks, observer }

/** The element currently showing its axes, or null. */
export function axesTarget() {
    return axes?.id || null;
}

/** Which space the axes are drawn in - "world" or "local". */
export function axesSpace() {
    return axes?.space || null;
}

/** Where the gizmo sits - "centre" of the visible mesh, or the node "origin". */
export function axesAnchor() {
    return axes?.anchor || null;
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

/**
 * A gizmo part is never culled, and always knows where it is.
 *
 * The gizmo does not move by being re-parented: its root is *re-positioned
 * every frame* onto whatever element it belongs to. `doNotSyncBoundingInfo`
 * was set on every part as a micro-optimisation, and that is exactly the flag
 * that stops Babylon updating a mesh's bounding box when its world matrix
 * changes - so the boxes stayed wherever the gizmo was built. Select an element
 * 50 m away and the boxes are 50 m behind the arrows (measured: a drift of
 * exactly 50), which leaves the frustum test culling arms, arrowheads and turn
 * arcs on their old position rather than their real one.
 *
 * That is what "the arrows get clipped, and if I go forward the lines get
 * clipped too" was - not depth, not the near plane - and why clicking away and
 * back fixed it: re-showing rebuilds the meshes, so their boxes start correct
 * again and drift away from there.
 *
 * Fifteen tiny meshes are not worth culling at all, so they are marked as
 * always active. Bounding info is left to sync normally, because a stale box is
 * a trap for anything else that ever asks one of these where it is.
 */
function keepAlwaysDrawn(mesh) {
  mesh.alwaysSelectAsActiveMesh = true;
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

        const shaft = MeshBuilder.CreateCylinder(`AXES_${a}_shaft`, { height: length * 0.82, diameter: r * 2, tessellation: 10 }, scene);
    shaft.position.z = length * 0.41;
        const head = MeshBuilder.CreateCylinder(`AXES_${a}_head`, { height: length * 0.18, diameterTop: 0, diameterBottom: r * 5, tessellation: 12 }, scene);
    head.position.z = length * 0.91;
    // "a turn goes about this one": a curved arrow encircling the shaft, the
    // universal reading for rotation - and unmistakable against the straight
    // arrows it sits on.
    const rotMark = buildRotationArrow(scene, `AXES_${a}_rot`, length, r);
    // "a scale acts on this one": a cube sitting on the very tip
        const scaleMark = MeshBuilder.CreateBox(`AXES_${a}_scale`, { size: r * 5.5 }, scene);
    scaleMark.position.z = length * 1.04;

    for (const m of [shaft, head, scaleMark]) {
      m.rotation.x = Math.PI / 2;      // Babylon builds cylinders along +Y
      m.material = m === scaleMark ? mats[`${a}_mark`] : mats[a];
      m.parent = arm;
      m.isPickable = false;
      // draw over the ship: an arrow buried inside the element it belongs to
      // would be exactly as useful as no arrow
      m.renderingGroupId = 1;
      keepAlwaysDrawn(m);
    }
    for (const m of rotMark.getChildMeshes()) {
      m.material = mats[`${a}_mark`];
      m.isPickable = false;
      m.renderingGroupId = 1;
      keepAlwaysDrawn(m);
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
    const tube = MeshBuilder.CreateTube(`${name}_arc`, { path, radius: r * 0.45, tessellation: 8, cap: BABYLON.Mesh.CAP_ALL }, scene);
  tube.parent = node;

  // The head goes on the end of the arc, pointing along the tangent there.
    const head = MeshBuilder.CreateCylinder(`${name}_head`, { height: length * 0.065, diameterTop: 0, diameterBottom: r * 2, tessellation: 12 }, scene);
  head.position.copyFrom(path[path.length - 1]);
  const tangent = new Vector3(-Math.sin(sweep), Math.cos(sweep), 0).normalize();
  // Cones are built along +Y; turn that onto the tangent. Done by hand rather
  // than with lookAt(), which aims +Z and would need a second correction.
  const axis = Vector3.Cross(Vector3.Up(), tangent);
    head.rotationQuaternion =
        axis.lengthSquared() < 1e-12 ? Quaternion.Identity() : Quaternion.RotationAxis(axis.normalize(), Math.acos(Math.min(1, Math.max(-1, Vector3.Dot(Vector3.Up(), tangent)))));
  head.parent = node;
  return node;
}

/** Repaint from the modal state: which axes drag, turn and scale. */
function paintAxes() {
  if (!axes) return;
  const live = new Set(liveAxes());
  const scaled = new Set(scaleAxes());
  const rotationEnabled = entryOf(axes.id)?.canRotate !== false;
  for (const a of ["x", "y", "z"]) {
    const on = live.has(a);
    const c = AXIS_COLOR[a];
        axes.mats[a].emissiveColor.copyFromFloats(on ? c.r : c.r * 0.3, on ? c.g : c.g * 0.3, on ? c.b : c.b * 0.3);
    axes.mats[a].alpha = on ? 1 : 0.45;
    axes.marks.rot[a].setEnabled(rotationEnabled && a === state.rotAxis);
    axes.marks.scale[a].setEnabled(scaled.has(a));
  }
}

/**
 * Point the arms down the element's own axes.
 *
 * A node's scale numbers act along its own axes whatever space the tools are
 * in, so a local gizmo is what says which way `scaling.x` will actually grow a
 * piece that has been turned - which is most of a ship built from a modular
 * kit. (A world-space scale reaches the same place by matching the world axis
 * against these; the gizmo is where you see why it picked the one it did.) The
 * arms are aimed individually from the world matrix's basis rows rather than by
 * rotating the whole gizmo, because a mirrored element (negative scale) has no
 * rotation that expresses it: its local +X genuinely points the other way, and
 * each arm can simply be turned round.
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

/**
 * Where on the element the gizmo hangs, as an offset in the element's *own*
 * coordinates.
 *
 * The node's origin is wherever the kit's author left it, and on this kit that
 * is regularly nowhere near the piece: a corridor section modelled off to one
 * side puts its arrows metres away from the wall they describe, or inside the
 * next room. So the gizmo is anchored on the middle of what you can actually
 * see, and the node origin becomes the *other* reading, on Ctrl.
 *
 * Measured once and kept as a local offset rather than re-derived each frame,
 * for two reasons. It is a bounding box, so it is axis-aligned to the *world*,
 * and re-measuring it every frame would make the anchor swim across the mesh
 * while the element turns - the box grows and shrinks under rotation even
 * though nothing about the element has moved. Held as a local offset it stays
 * pinned to the same point of the model through turns, drags, scales and
 * mirrors, and costs one matrix multiply a frame instead of a walk over every
 * child mesh.
 *
 * `worldBounds` does the choosing of what counts as visible - it already skips
 * Runtime stand-ins and other elements' light gizmos - so "the centre" here
 * means the same thing as it does everywhere else in the editor.
 */
function centreOffset(node, bounds) {
  const b = bounds ?? worldBounds(node);
  if (!b) return null;
  const world = node.getWorldMatrix();
  // A flattened element (a zero scale on some axis) has no invertible frame and
  // no thickness to centre on either; the origin is the only honest answer.
  if (Math.abs(world.determinant()) < 1e-12) return null;
    return Vector3.TransformCoordinates(b.min.add(b.max).scale(0.5), Matrix.Invert(world));
}

/**
 * The world point the gizmo root belongs on this frame.
 *
 * The offset is keyed to the node it was measured on, because the observer
 * re-resolves the id every frame and nothing about an id guarantees the same
 * object behind it: an undo rebuilds every placement from its snapshot, and
 * arming a different module builds a new ghost. Measured, both of those blank
 * the node for a frame first and so take the whole gizmo with them - the
 * re-measure has no observable effect today. It costs one reference compare a
 * frame, and it is what keeps "this offset belongs to that node" a fact rather
 * than an assumption about code somewhere else.
 */
function anchorPoint(node) {
  const world = node.getWorldMatrix();
  if (axes.anchor !== "centre") return world.getRow(3).toVector3();
  if (node !== axes.anchorNode) {
    axes.anchorNode = node;
    axes.anchorLocal = centreOffset(node);
  }
    return axes.anchorLocal ? Vector3.TransformCoordinates(axes.anchorLocal, world) : world.getRow(3).toVector3();
}

/**
 * Shrink the gizmo as the camera closes on it.
 *
 * Applied to the root as a uniform scale, so one number governs shafts, heads,
 * turn rings, scale cubes and the gaps between them together, and the arms keep
 * their directions — the labels ride the meshes' own absolute positions, so
 * they follow without being told.
 */
function sizeAxes() {
  // An unparented camera's `position` is already world space, and the editor's
  // never has a parent. `globalPosition` is a cache a camera only fills in once
  // its view matrix has been recomputed, so it still reads as the origin on the
  // first frame after a saved view is restored - which is exactly the frame a
  // gizmo restored with it would be sized on.
  const eye = state.camera?.position;
  let scale = 1;
  if (eye) {
    const away = Vector3.Distance(eye, axes.root.position);
    // The smallest fraction any tier asks for, so tiers nest without the list
    // having to be kept in order.
    for (const step of AXIS_GIZMO_NEAR_STEPS) {
      if (away < step.within) scale = Math.min(scale, step.scale);
    }
  }
  axes.root.scaling.setAll(scale);
}

export function showAxes(id, space = "world", anchor = "origin") {
  const node = axesNode(id);
  if (!node) return false;
  hideAxes();

  const bounds = anchor === "centre" ? worldBounds(node) : null;
  const built = buildAxes(state.scene, AXIS_GIZMO_LENGTH);

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
    const scaleLabels = { x: chip("axis-snap axis-scale"), y: chip("axis-snap axis-scale"), z: chip("axis-snap axis-scale") };

    axes = { ...built, id, space, anchor, label, rotLabel, scaleLabels, observer: null, anchorNode: node, anchorLocal: anchor === "centre" ? centreOffset(node, bounds) : null };
  paintAxes();
  orientAxes(node);
  // Position is re-read every frame rather than parented: parenting would
  // inherit the element's scale - and a 3x element must not get 3x arrows -
  // while re-deriving it also covers drags, undo, the ghost following the
  // cursor, and deletion, without any event plumbing at all.
  axes.observer = state.scene.onBeforeRenderObservable.add(() => {
    const cur = axesNode(id);
        if (!cur) {
            hideAxes();
            return;
        }
    axes.root.position.copyFrom(anchorPoint(cur));
    sizeAxes();
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
    const w = engine.getRenderWidth(),
        h = engine.getRenderHeight();
    const p = Vector3.Project(at, Matrix.Identity(), scene.getTransformMatrix(), state.camera.viewport.toGlobal(w, h));
  // behind the camera projects to a mirrored point, which would park the chip
  // on the opposite side of the screen from the thing it belongs to
    if (p.z < 0 || p.z > 1) {
        el.hidden = true;
        return;
    }
  // Off the side of the canvas is just as wrong: the chips hang off the arrow
  // *tips*, which swing outside the viewport at close range, and #viewport does
  // not clip - one was measured sitting on a palette tile.
    if (p.x < 0 || p.x > w || p.y < 0 || p.y > h) {
        el.hidden = true;
        return;
    }
  const box = engine.getRenderingCanvas().getBoundingClientRect();
  const host = document.getElementById("viewport")?.getBoundingClientRect();
  if (!host) return;
  el.hidden = false;
  el.textContent = text;
  el.style.left = `${p.x + box.left - host.left}px`;
  el.style.top = `${p.y + box.top - host.top}px`;
}

/**
 * The move step at the gizmo root, the turn angle in the curved arrow, and the
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
/**
 * What one of the step chips says: the value with its unit, or `free` when it
 * is the loosest its list offers. See `freeSnap` for why that is not simply a
 * zero on two of the three.
 */
function stepChipText(key, value, unit = "") {
  if (!value || value === freeSnap[key]) return "free";
  return `${value}${unit}`;
}

function placeAxisLabel() {
  if (!axes) return;
    placeChip(axes.label, axes.root.position, stepChipText("pos", state.snap.pos, " m"));

  const ring = axes.marks.rot[state.rotAxis];
  if (ring && axes.rotLabel) {
        if (!ring.isEnabled()) {
            axes.rotLabel.hidden = true;
        } else {
      // The curved arrow used to be mirrored for a negative step, back when a
      // step carried its own direction. The wheel carries it now - one way
      // turns, the other way turns back - so the step is a magnitude again and
      // the arrow has nothing to disagree with.
            placeChip(axes.rotLabel, ring.getAbsolutePosition(), stepChipText("rot", state.snap.rot, "°"));
    }
  }

  for (const a of ["x", "y", "z"]) {
    const el = axes.scaleLabels?.[a];
    const cube = axes.marks.scale[a];
    if (!el || !cube) continue;
        if (!cube.isEnabled()) {
            el.hidden = true;
            continue;
        }
    cube.computeWorldMatrix(true);
        placeChip(el, cube.getAbsolutePosition(), stepChipText("scale", state.snap.scale));
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
 * The same element in the same space *at the same anchor* turns them off;
 * anything else about the request differing - a different element, the other
 * space, or the other anchor - moves, re-aims or re-hangs them instead. So
 * `X` followed by `Ctrl+X` walks the gizmo from the element's centre to its
 * origin rather than blinking it off, which is the only reading that lets you
 * compare the two.
 */
export function toggleAxes(id, space = "world", anchor = "origin") {
  if (axes && axes.id === id && axes.space === space && axes.anchor === anchor) {
    hideAxes();
    return null;
  }
  return showAxes(id, space, anchor) ? id : null;
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
        const d = Math.hypot((b.minX + b.maxX) / 2 - scene.pointerX, (b.minY + b.maxY) / 2 - scene.pointerY);
        if (d < bestD) {
            bestD = d;
            best = id;
        }
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
    const ray = scene.createPickingRay(scene.pointerX, scene.pointerY, Matrix.Identity(), state.camera);
  if (Math.abs(ray.direction.y) < 1e-5) return null;
  const t = (y - ray.origin.y) / ray.direction.y;
  if (t <= 0) return null;
  return ray.origin.add(ray.direction.scale(t));
}

export function cursorOnGrid() {
    return cursorOnPlane(state.gridY);
}

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
    const ray = scene.createPickingRay(scene.pointerX, scene.pointerY, Matrix.Identity(), state.camera);
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

  const requestLookPointerLock = () => {
    if (!canvas.requestPointerLock || document.pointerLockElement === canvas) return;
    try {
      const pending = canvas.requestPointerLock();
      pending?.catch((error) => console.warn("Could not lock the camera pointer.", error));
    } catch (error) {
      console.warn("Could not lock the camera pointer.", error);
    }
  };

  const releaseLookPointerLock = () => {
    if (document.pointerLockElement === canvas) document.exitPointerLock();
  };

  scene.onPointerObservable.add((pi, es) => {
    if (pi.type === PointerEventTypes.POINTERMOVE) {
      if (rmbDown) {
        const start = down.get(2);
        if (start) {
          start.travel += Math.hypot(pi.event.movementX || 0, pi.event.movementY || 0);
          if (start.travel > 4) rmbGesture = true;
        }
      }
      emit("pointermove", pi.event);
      return;
    }
    if (pi.type === PointerEventTypes.POINTERDOWN) {
      down.set(pi.event.button, {
                x: pi.event.clientX,
                y: pi.event.clientY,
                t: performance.now(),
                travel: 0,
      });
      if (pi.event.button === 2) {
        rmbDown = true;
        rmbGesture = false;
        requestLookPointerLock();
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
    if (pi.event.button === 2) {
      rmbDown = false;
      releaseLookPointerLock();
      emit("rmbup", pi.event);
    }
    if (pi.event.button === 0) emit("pointerup", pi.event);
    if (!start) return;
        const moved = Math.max(start.travel, Math.hypot(pi.event.clientX - start.x, pi.event.clientY - start.y));
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
        const isDouble = now - lastTapAt < DOUBLE_CLICK_MS && Math.hypot(pi.event.clientX - lastTapPos.x, pi.event.clientY - lastTapPos.y) < 6;
    lastTapAt = isDouble ? 0 : now;
    lastTapPos = { x: pi.event.clientX, y: pi.event.clientY };

    emit("click", pi.event);
    if (isDouble) emit("dblclick", pi.event);
  });

  document.addEventListener("pointerlockchange", () => {
    if (document.pointerLockElement === canvas) {
      // A quick click can release RMB before the asynchronous lock request
      // completes. Do not leave that late lock active after the gesture.
      if (!rmbDown) releaseLookPointerLock();
      return;
    }
    if (!rmbDown) return;
    // Escape can release pointer lock without a pointerup. Give Babylon and
    // the editor the missing release so neither camera look nor hover sticks.
    rmbGesture = true;
    const start = down.get(2);
        canvas.dispatchEvent(
            new PointerEvent("pointerup", {
      button: 2,
      buttons: 0,
      clientX: start?.x || 0,
      clientY: start?.y || 0,
      bubbles: true,
            })
        );
  });

  document.addEventListener("pointerlockerror", () => {
    console.warn("The browser rejected camera pointer lock; edge-limited look remains available.");
  });

  // a button released outside the canvas never reaches the observable
  window.addEventListener("pointerup", (e) => {
    if (e.button !== 2 || !rmbDown) return;
    rmbDown = false;
    releaseLookPointerLock();
    emit("rmbup", e);
  });
  window.addEventListener("blur", () => {
    if (rmbDown) emit("rmbup");
    rmbDown = false;
    releaseLookPointerLock();
  });
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
export function isRmbDown() {
    return rmbDown;
}

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
    const pick = scene.pick(scene.pointerX, scene.pointerY, (m) => m.isPickable && m.isEnabled());
  const id = pick?.hit ? ownerIdOf(pick.pickedMesh) : null;
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
  // Which world this lands in. A caller that says nothing gets the world it is
  // looking at: placing while a bench is open has to land on the bench, or it
  // would quietly add an element to the ship hidden behind it - in a chunk the
  // author never chose, and invisible until they closed the bench. The two
  // callers that mean the *other* world say so outright: a ship load is ship
  // data by definition, and the bench restores name their own bench.
  const benched = opts.stage === undefined ? state.mode !== "ship" : !!opts.stage;
  let id = opts.id;
  if (!id) {
    do {
      id = `P${String(state.nextId++).padStart(4, "0")}`;
    } while (state.environmentProbes.has(id));
  }
  if (opts.id) {
    const n = parseInt(String(opts.id).replace(/\D/g, ""), 10);
    if (Number.isFinite(n) && n >= state.nextId) state.nextId = n + 1;
  }
  if (opts.group) noteGroupId(opts.group);
  const node = await instantiate(moduleId, id);
  node.position.copyFrom(position);
  if (opts.rotation) {
        node.rotationQuaternion = Quaternion.FromEulerAngles((opts.rotation[0] * Math.PI) / 180, (opts.rotation[1] * Math.PI) / 180, (opts.rotation[2] * Math.PI) / 180);
  }
  if (opts.scale) node.scaling.set(opts.scale[0], opts.scale[1], opts.scale[2]);

  const entry = {
    id,
    module: moduleId,
    chunk: benched ?opts.stageChunk || defaultBenchChunk() :opts.chunk || state.activeChunk,
    name: opts.name || "",        // optional label, see renamePlacement
    // A stand-in on one of the two benches rather than part of the ship. It is
    // a real placement so that selection, the gizmo, hiding, dragging and
    // Ctrl+D all work on it unchanged - and filtered out at the two boundaries
    // that walk every placement, so it can never reach the ship.
    stage: benched,
    // Which compound instance this placement is a member of, and which
    // definition it was expanded from. Both empty for an ordinary placement.
    // A member is an ordinary placement in every other respect - see
    // groupMembers() for what the pair buys.
    group: opts.group || "",
    compound: opts.compound || "",
    node,
  };
  node.metadata = { placement: entry };
  state.placements.set(id, entry);
  if (!entry.stage && !state.chunks.includes(entry.chunk)) state.chunks.push(entry.chunk);
  // A ceiling panel arrives lit: kit_lights.json says where the lamp sits in
  // the module's own space. Skipped by the two callers that bring lights with
  // them - a duplicate copies the original's, a load restores what was saved -
  // because seeding those would add a second lamp beside every one of them.
  const seeded = opts.noLights ? [] : hooks.seedKitLights(id);
  applyVisibility();
    if (!opts.silent) {
        emit("placements");
        select([id]);
    }
  // Not folded into the branch above: a run of tiles dropped in one go places
  // every one but the first silently, and the lights panel still has to see
  // the lamps that came with them.
  if (seeded.length) emit("lights");
  return entry;
}

export function removeSelected() {
  if (!state.selection.length) return;
  pushUndo();
  for (const id of state.selection) {
    const p = state.placements.get(id);
    // Taking a staged element off the collision area must not lose what was
    // fitted to it: unstageModule reads its shapes into the per-module record
    // first, so staging the module again brings them straight back. A compound
    // bench member has no such record - it is only ever an ordinary placement.
    if (p && benchOf(p) === "collision") hooks.unstageModule(id);
    else if (p) removePlacement(id);
    else if (state.colliders.has(id)) hooks.removeCollider(id);
    else if (state.lights.has(id)) hooks.removeLight(id);
    else if (state.environmentProbes.has(id)) removeEnvironmentProbe(id, false);
    else removeMarkerNode(id);
  }
  select([]);
  hooks.harvestStage();
  emit("placements");
  emit("markers");
  emit("colliders");
  emit("lights");
}

function removeMarkerNode(id) {
  const m = state.markers.get(id);
  if (!m) return;
  m.mesh?.dispose();
  m.node.dispose();
  state.markers.delete(id);
  // A door's behaviours hang off its id, and `nextDoorId` hands a freed id to
  // the next door placed - so an entry left behind here would quietly become
  // that door's. This is the one way an id gets re-adopted, and why a marker
  // cannot be treated like a placement, whose named entries are worth keeping
  // precisely because no other element will ever carry that id.
  state.entities.delete(id);
}

export function removePlacement(id) {
  const entry = state.placements.get(id);
  if (!entry) return;
  // Before the node goes: the light nodes are its children and would be
  // disposed with it, leaving entries pointing at scene nodes that no longer
  // exist.
  hooks.removeLightsOf(id);
  entry.node.getChildMeshes().forEach((m) => m.dispose());
  entry.node.dispose();
  state.placements.delete(id);
  // An entry under a *name* is left alone even when nothing carries it any
  // more: a name is a thing you author, and deleting the last crate to put a
  // better one down should not throw away how crates behave. An entry under an
  // id cannot be re-adopted - no other element will ever carry that id - so
  // leaving it would only put a node the .glb has never heard of in the
  // manifest.
  if (!String(entry.name || "").trim()) state.entities.delete(id);
}

/**
 * Copy the selection one snap step along X.
 *
 * A copy is a new element, not a second handle on the old one, so everything
 * that says *which* element this is is minted fresh and everything that says
 * what it is like is brought across:
 *
 *  - **The name is dropped.** Names are shared on purpose, so keeping it would
 *    hand the copy the original's identity - one behaviour entry governing
 *    both, `linked` unable to name one without the other, and the Live checks
 *    counting two elements where the manifest sees one node. The copy exports
 *    under its id until it is given a name of its own.
 *  - **The behaviours come with it**, under that id, unless `behaviors: false`
 *    asks for a bare copy. Dropping the name would otherwise quietly strip a
 *    duplicated fan of the very thing that makes it a fan.
 *  - **Lights come across** for the same reason: a copied ceiling panel that
 *    arrived dark would be a trap, since the light is part of what the element
 *    IS, the same way its collision shapes are.
 */
export async function duplicateSelected({ behaviors = true } = {}) {
  if (!state.selection.length) return;
  pushUndo();
  const made = [];
  // One fresh group id per source compound, minted on first sight. A copy has
  // to be its own instance - sharing the original's id would make selecting
  // either one select both, and moving one move the other.
  const regroup = new Map();
  for (const id of state.selection) {
    const e = state.placements.get(id);
    if (!e) continue;
    if (e.group && !regroup.has(e.group)) regroup.set(e.group, nextGroupId());
    const rot = eulerOf(e.node);
        const copy = await placeAt(e.module, e.node.position.add(new Vector3(state.snap.pos || 1, 0, 0)), {
            rotation: rot,
            scale: e.node.scaling.asArray(),
            chunk: e.chunk,
            name: "",
            group: regroup.get(e.group) || "",
            compound: e.compound,
        // A copy belongs wherever its original does. Read off the source rather
        // than off the mode so that a bench member cannot be duplicated into a
        // ship element sitting in a chunk that does not exist.
            stage: e.stage,
            stageChunk: e.chunk,
            silent: true,
            noLights: true,
        });
    hooks.copyLightsTo(id, copy.id);
    copyBehaviorsTo(id, copy.id, { copy: behaviors });
    made.push(copy.id);
  }
  emit("placements");
  emit("lights");
  emit("behaviors");
  select(made);
}

export function clearAll() {
  hideAxes();
  for (const id of [...state.placements.keys()]) removePlacement(id);
  for (const id of [...state.markers.keys()]) removeMarkerNode(id);
  for (const id of [...state.colliders.keys()]) hooks.removeCollider(id);
  // removePlacement drops the lights it owns, so this only catches a light that
  // somehow outlived its owner - cheap insurance against a leak across a load.
  state.lights.clear();
  state.selection = [];
  state.nextId = 1;
  state.nextGroup = 1;
  // ids restart at P0001, so a leftover entry would hide a brand new element
  state.hidden.clear();
  // keyed by name rather than id, but a load must not merge the old ship's
  // behaviours into the new one
  state.behaviors.clear();
  state.entities.clear();
  state.environmentProbes.clear();
  emit("placements");
  emit("markers");
  emit("environment-probes");
  emit("selection");
}

// -------------------------------------------------------------- selection

export function select(ids) {
  state.selection = ids.filter((id) => entryOf(id));
  // Axes already on screen follow the selection: having asked to see them, you
  // almost never want them left behind on the element you just moved away from.
  // Only for a single pick - a multi-selection has no one element to sit on.
  //
  // And they follow in the flavour they were already in. Re-showing them took
  // the default, so a local gizmo - the one that matters, since a node's scale
  // numbers act along its own axes and a turned piece has its own idea of which
  // way X grows - silently reverted to world on the next click, and `Shift+X`
  // had to be pressed again for every element. The anchor rides along for the
  // same reason.
  if (axes && state.selection.length === 1 && state.selection[0] !== axes.id) {
    showAxes(state.selection[0], axes.space, axes.anchor);
  }
  emit("selection");
}

export function toggleSelect(ids) {
  // A list toggles as a unit. A compound is either in the selection or out of
  // it - never half in - so every member follows whichever way the one under
  // the cursor would have gone on its own.
  const list = Array.isArray(ids) ? ids : [ids];
  if (!list.length) return;
  const removing = state.selection.includes(list[0]);
  for (const id of list) {
    const i = state.selection.indexOf(id);
        if (removing) {
            if (i >= 0) state.selection.splice(i, 1);
        } else if (i < 0) state.selection.push(id);
  }
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
  const basis = axisBasis(entryOf(state.selection[0])?.node);
    const step = basis ? basis.x.scale(delta.x).add(basis.y.scale(delta.y)).add(basis.z.scale(delta.z)) : delta;
  pushUndo();
  for (const id of state.selection) {
    const e = entryOf(id);
    // A probe's inner blend box has no centre of its own - it rides the outer
    // one's - so there is nothing here for an arrow key to write.
    if (e && e.canMove !== false) e.node.position.addInPlace(step);
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
    let min = null,
        max = null;
  for (const n of nodes) {
    const b = worldBounds(n);
    if (!b) continue;
    min = min ? Vector3.Minimize(min, b.min) : b.min.clone();
    max = max ? Vector3.Maximize(max, b.max) : b.max.clone();
  }
    const c = min ? min.add(max).scale(0.5) : nodes[0].getAbsolutePosition().clone();
  const size = min ? max.subtract(min).length() : 4;
  const dist = Math.max(4, size * 1.3);

  const cam = state.camera;
  // back off along the current view direction so framing keeps the orientation
  const dir = cam.getDirection(Vector3.Forward()).normalize();
  cam.cameraDirection.setAll(0);
  cam.position.copyFrom(c.subtract(dir.scale(dist)));
  cam.setTarget(c);
}

// How far in front of the camera a brought piece lands, in metres. Anything
// under BRING_NEAR ends up inside your face; anything past BRING_FAR is no
// longer "close to us", which is the whole request.
const BRING_NEAR = 3;
const BRING_FAR = 8;
const BRING_DROP = 6;        // how far below the spot a floor still counts

/**
 * A spot just in front of the camera to drop something on.
 *
 * Two searches, because a ship interior and open space want different answers.
 * **Forward** first, so the spot is never pushed through the wall you are
 * facing - in a corridor the preferred distance would otherwise put the piece
 * in the next room. **Down** from there, so it lands on the deck you are
 * standing on rather than floating at eye height. With neither - out in the
 * open, or over a gap - the raw point in front of the camera is still somewhere
 * you can see, which is all that was asked for.
 *
 * `size` is the piece's diagonal: a 4 m wall wants standing-back room, a
 * handrail wants to be within reach. `skip` excludes the piece itself, which
 * would otherwise be the thing the forward ray hits.
 *
 * Deliberately *not* snapped. Modules are not modelled around their origin, so
 * only the caller knows which offset has to come off before the grid applies -
 * snapping here would align the origin and leave the body off-grid.
 */
export function cameraDropPoint(size = 2, skip = null) {
    const cam = state.camera,
        scene = state.scene;
  if (!cam || !scene) return null;
  const fwd = cam.getDirection(Vector3.Forward()).normalize();
  const want = Math.min(Math.max(size * 0.9, BRING_NEAR), BRING_FAR);
    const ahead = scene.pickWithRay(new BABYLON.Ray(cam.position.clone(), fwd, want), (m) => m.isPickable && m.isEnabled() && !(skip && skip(m)));
  // Stop just short of what is in the way rather than on its surface, so the
  // piece does not start out intersecting the wall you were looking at.
  const dist = ahead?.hit ? Math.max(ahead.distance - 0.25, 0.5) : want;
  const point = cam.position.add(fwd.scale(dist));
  const floor = groundHeightAt(point.x, point.z, point.y + 0.1, BRING_DROP + 0.1, skip);
  if (floor != null) point.y = floor;
  return { point, floor: floor != null };
}

// ----------------------------------------------------------------- chunks

/**
 * The far side of a door that opens onto space rather than onto a room.
 *
 * A window looking out of the hull is still a portal - the renderer has to know
 * there is an opening and what shape it is - but there is no chunk on the other
 * side to draw. So it is expressed as a reserved chunk id rather than as yet
 * another boolean: everything that already reasons about a door's two sides -
 * isolation, validation, the portal record, the manifest - keeps working
 * unchanged, and a side is still just a name.
 *
 * Double-underscored so it cannot be confused with a room, and `addChunk` and
 * `renameChunk` both refuse it, so no real chunk can ever collide with it.
 */
export const SKYBOX_CHUNK = "__SKYBOX__";

/** True for the one side value that names space instead of a room. */
export function isSkyboxChunk(id) {
    return id === SKYBOX_CHUNK;
}

export function addChunk(name) {
  if (!name || name === SKYBOX_CHUNK || state.chunks.includes(name)) return false;
  state.chunks.push(name);
  emit("chunks");
  return true;
}

/**
 * Everything that would be orphaned if a chunk went away.
 *
 * A chunk id is not a label: placements carry it and doors name it as one of
 * their two sides. Deleting the list entry alone would leave both pointing at
 * a room that no longer exists - placements invisible to isolation, doors with
 * a portal to nowhere - so `removeChunk` refuses while this finds anything.
 */
export function chunkUsers(name) {
    const placements = shipPlacements()
        .filter((e) => e.chunk === name)
        .map((e) => e.id);
    const doors = [...state.markers.values()].filter((m) => m.chunkA === name || m.chunkB === name).map((m) => m.id);
  return { placements, doors };
}

/**
 * Delete a chunk, but only once nothing refers to it.
 *
 * Refusing is the whole design. The alternatives were to drag the contents into
 * some other room - which silently rewrites the ship's layout to service a
 * button press - or to delete them with it, which turns one keystroke into an
 * unbounded loss. Emptying the room first is a deliberate act, and `Assign`
 * already exists to do it.
 *
 * Returns `{ ok }` or `{ ok: false, reason, users }`, so the caller can say
 * exactly what is still in the way rather than just failing.
 */
export function removeChunk(name) {
  if (!name || !state.chunks.includes(name)) return { ok: false, reason: "unknown" };
  // The ship is always in some chunk: `activeChunk` is what new placements
  // join, and there has to be one for them to join.
  if (state.chunks.length < 2) return { ok: false, reason: "last" };
  const users = chunkUsers(name);
  if (users.placements.length || users.doors.length) {
    return { ok: false, reason: "in use", users };
  }
  pushUndo();
  state.chunks = state.chunks.filter((c) => c !== name);
  if (state.activeChunk === name) state.activeChunk = state.chunks[0];
  applyVisibility();
  emit("chunks");
  return { ok: true };
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
  if (name === SKYBOX_CHUNK) return false;
  if (!state.chunks.includes(from) || state.chunks.includes(name)) return false;
  pushUndo();

  state.chunks = state.chunks.map((c) => (c === from ? name : c));
  for (const e of state.placements.values()) if (e.chunk === from) e.chunk = name;
  for (const m of state.markers.values()) {
    if (m.chunkA === from) m.chunkA = name;
    if (m.chunkB === from) m.chunkB = name;
  }
  if (state.activeChunk === from) state.activeChunk = name;
  // A chunk id is a node name too - the room holder itself - so behaviours can
  // watch it, and they must follow it here as they do for any other rename.
  const carried = state.entities.get(from);
  if (carried && !state.entities.has(name)) state.entities.set(name, carried);
  if (carried) state.entities.delete(from);
  renameEntityRefs(from, name);

  applyVisibility();
  emit("chunks");
  emit("placements");
  emit("markers");
  emit("behaviors");
  return true;
}

/** A probe volume component: three finite numbers, optionally all positive. */
function validProbeVector(value, positive = false) {
  const vector = value?.map(Number);
    return vector?.length === 3 && vector.every((n) => Number.isFinite(n) && (!positive || n > 0)) ? vector : null;
}

export function validEnvironmentProbeId(value) {
  const id = String(value || "").trim();
  return /^[A-Za-z0-9._-]{1,128}$/.test(id) ? id : null;
}

/**
 * A probe's two blending volumes are selected as PARTS of it.
 *
 * A part is an **id**, not a second kind of element: `"ENV0001#influence"` runs
 * through selection, hover, dragging and the inspector unchanged, and is
 * resolved back to the probe it belongs to here. That keeps direct manipulation
 * of the volumes out of every generic path in the tool, which is what made the
 * capture box editable in the first place.
 *
 * `#` is deliberately outside the grammar validEnvironmentProbeId accepts - and
 * outside every id the tool generates - so a part id can never shadow a real
 * element, whichever store it lives in.
 */
export const PROBE_PARTS = ["influence", "inner"];

/**
 * The three volumes a probe draws, in pane order.
 *
 * Wider than PROBE_PARTS on purpose: the capture box is drawn and hidden like
 * the other two, but it is selected under the probe's own id rather than a part
 * id, so it has no place in the list above.
 */
export const PROBE_VOLUMES = ["box", "influence", "inner"];

export function environmentProbePartId(id, part) {
    return `${id}#${part}`;
}

export function environmentProbePartOf(id) {
  const text = String(id ?? "");
  const at = text.indexOf("#");
  if (at < 0) return null;
  const probe = text.slice(0, at);
  const part = text.slice(at + 1);
    return PROBE_PARTS.includes(part) && state.environmentProbes.has(probe) ? { probe, part } : null;
}

/** Whether an id can identify a probe without shadowing another editor entry. */
export function environmentProbeIdAvailable(value, currentId = null) {
  const id = validEnvironmentProbeId(value);
  if (!id) return false;
    if (state.placements.has(id) || state.markers.has(id) || state.colliders.has(id) || state.lights.has(id)) return false;
  return id === currentId || !state.environmentProbes.has(id);
}

/**
 * How far past the projection box a probe reaches by default, in metres.
 *
 * The runtime blends probes with Lagarde's normalized distance field: full
 * strength inside an inner box, fading to nothing at an outer one. Nothing
 * about the room says where those two should sit, so the default is a margin
 * either side of the box the author DID draw - which is what the runtime falls
 * back to on its own (DEFAULT_BLEND_DISTANCE in local-environments.ts), and
 * agreeing with it means a probe authored before these fields existed does not
 * change how it blends the day it is re-saved.
 */
const PROBE_BLEND_MARGIN = 1.5;

/**
 * The influence volumes a probe gets when nobody has said otherwise: centred on
 * the box, one margin out and one margin in. The inner box floors at zero,
 * where a room narrower than two margins simply has no full-strength core.
 */
function defaultProbeInfluence(boxPosition, boxSize) {
  return {
    influenceBoxPosition: [...boxPosition],
    influenceBoxSize: boxSize.map((n) => n + PROBE_BLEND_MARGIN * 2),
    influenceInnerBoxSize: boxSize.map((n) => Math.max(0, n - PROBE_BLEND_MARGIN * 2)),
  };
}

function defaultSphereProbeInfluence(spherePosition, sphereRadius) {
  return {
    influenceSpherePosition: [...spherePosition],
    influenceSphereRadius: sphereRadius + PROBE_BLEND_MARGIN,
    influenceInnerSphereRadius: Math.max(0, sphereRadius - PROBE_BLEND_MARGIN),
  };
}

/**
 * Which of a probe's three volumes the viewport draws.
 *
 * `asked` wins where it says a boolean, `standing` is what the probe already
 * had, and a volume neither of them mentions is shown - so a record written
 * before these existed, or a caller that only edited the numbers, draws all
 * three. Built through PROBE_VOLUMES so the key order is fixed: the no-op check
 * in setEnvironmentProbe compares serialized records.
 */
function probeVisibleParts(asked, standing) {
  const parts = {};
  for (const volume of PROBE_VOLUMES) {
        parts[volume] = typeof asked?.[volume] === "boolean" ? asked[volume] : standing?.[volume] !== false;
  }
  return parts;
}

function cloneEnvironmentProbe(probe) {
  if (!probe) return null;
  const shape = probe.shape === "sphere" ? "sphere" : "box";
  return {
    id: probe.id,
    shape,
    ...(shape === "sphere"
      ? {
          spherePosition: [...probe.spherePosition],
          sphereRadius: probe.sphereRadius,
          influenceSpherePosition: [...probe.influenceSpherePosition],
          influenceSphereRadius: probe.influenceSphereRadius,
          influenceInnerSphereRadius: probe.influenceInnerSphereRadius,
        }
      : {
          boxPosition: [...probe.boxPosition],
          boxSize: [...probe.boxSize],
          angle: probe.angle,
          influenceBoxPosition: [...probe.influenceBoxPosition],
          influenceBoxSize: [...probe.influenceBoxSize],
          influenceInnerBoxSize: [...probe.influenceInnerBoxSize],
        }),
    capturePosition: [...probe.capturePosition],
    // Authored capture policy. Missing means clipped so manifests written
    // before the option existed adopt the room-bounded behaviour.
    clipCapture: probe.clipCapture !== false,
    // View state, not ship data - see setEnvironmentProbeView. Carried here all
    // the same, because serialize() writes probes through this function and an
    // undo rebuilds the whole map: leaving them out would make every Ctrl+Z put
    // the probes you had on screen back down.
    alwaysVisible: !!probe.alwaysVisible,
    envFaces: !!probe.envFaces,
    visibleParts: probeVisibleParts(probe.visibleParts),
  };
}

/** One authored local-environment volume, in editor world space. */
export function environmentProbeOf(id) {
  return cloneEnvironmentProbe(state.environmentProbes.get(id));
}

export function environmentProbeIds() {
  return [...state.environmentProbes.keys()];
}

export function nextEnvironmentProbeId() {
  for (let n = 1; ; n++) {
    const id = `ENV${String(n).padStart(4, "0")}`;
    if (environmentProbeIdAvailable(id)) return id;
  }
}

/**
 * Add, update, or rename an explicit local-environment volume.
 *
 * `history` is the escape hatch for the pane, which commits on every keystroke:
 * one snapshot per visit to a field is the rule everywhere in the tool, so the
 * caller pushes its own and asks this not to push a second.
 */
export function setEnvironmentProbe(id, probe, previousId = id, { history = true } = {}) {
  const key = validEnvironmentProbeId(id);
  const previous = String(previousId || "").trim();
  const capturePosition = validProbeVector(probe?.capturePosition);
  const standing = state.environmentProbes.get(previous) || state.environmentProbes.get(key);
  const shape = probe?.shape === "sphere" ? "sphere" : "box";
  if (!key || !capturePosition) return false;
  if (!environmentProbeIdAvailable(key, previous)) return false;
  let volume;
  if (shape === "sphere") {
    const spherePosition = validProbeVector(probe?.spherePosition);
    const sphereRadius = Number(probe?.sphereRadius);
    if (!spherePosition || !Number.isFinite(sphereRadius) || sphereRadius <= 0) return false;
    const defaults = defaultSphereProbeInfluence(spherePosition, sphereRadius);
        const influenceSpherePosition = validProbeVector(probe?.influenceSpherePosition) || defaults.influenceSpherePosition;
    const influenceSphereRadius = Number(probe?.influenceSphereRadius ?? defaults.influenceSphereRadius);
    const influenceInnerSphereRadius = Number(probe?.influenceInnerSphereRadius ?? defaults.influenceInnerSphereRadius);
        if (
            !Number.isFinite(influenceSphereRadius) ||
            influenceSphereRadius <= 0 ||
            !Number.isFinite(influenceInnerSphereRadius) ||
            influenceInnerSphereRadius < 0 ||
            influenceInnerSphereRadius > influenceSphereRadius
        )
            return false;
    volume = {
            shape,
            spherePosition,
            sphereRadius,
            capturePosition,
            influenceSpherePosition,
            influenceSphereRadius,
            influenceInnerSphereRadius,
    };
  } else {
    const boxPosition = validProbeVector(probe?.boxPosition);
    const boxSize = validProbeVector(probe?.boxSize, true);
    const angle = Number(probe?.angle ?? standing?.angle ?? 0);
    if (!boxPosition || !boxSize || !Number.isFinite(angle)) return false;
    const influence = probeInfluence(probe, boxPosition, boxSize);
    if (!influence) return false;
    volume = { shape, boxPosition, boxSize, capturePosition, angle, ...influence };
  }
  // A caller editing the numbers says nothing about the view flags, and must
  // not silently put the probe's boxes away: they stay as the record has them.
  const next = {
        id: key,
        ...volume,
    clipCapture:
      typeof probe?.clipCapture === "boolean"
        ? probe.clipCapture
        : standing?.clipCapture !== false,
    alwaysVisible: !!(probe?.alwaysVisible ?? standing?.alwaysVisible),
    envFaces: !!(probe?.envFaces ?? standing?.envFaces),
    visibleParts: probeVisibleParts(probe?.visibleParts, standing?.visibleParts),
  };
    if (previous === key && JSON.stringify(state.environmentProbes.get(key) || null) === JSON.stringify(next)) {
    return false;
  }
  if (history) pushUndo();
  if (previous && previous !== key) state.environmentProbes.delete(previous);
  state.environmentProbes.set(key, next);
  if (previous && previous !== key) {
    state.selection = state.selection.map((selected) => ( selected === previous ? key : selected));
  }
  emit("environment-probes");
  if (previous && previous !== key) emit("selection");
  return true;
}

/**
 * Turn one probe's view flags on or off.
 *
 * `alwaysVisible` keeps a probe's boxes on screen while another one is being
 * edited, `envFaces` draws its captured cubemap on the box instead of a
 * wireframe, and `visibleParts` says which of its three volumes are drawn at
 * all - a partial patch, so an eye names only the volume it toggles. None of
 * them changes the ship, so none pushes an undo entry: they are the probe
 * window's equivalent of parking an element out of the way.
 */
export function setEnvironmentProbeView(id, patch) {
  const probe = state.environmentProbes.get(id);
  if (!probe || !patch || typeof patch !== "object") return false;
  const next = {
    ...probe,
    alwaysVisible: typeof patch.alwaysVisible === "boolean" ? patch.alwaysVisible : probe.alwaysVisible,
    envFaces: typeof patch.envFaces === "boolean" ? patch.envFaces : probe.envFaces,
    visibleParts: probeVisibleParts(patch.visibleParts, probe.visibleParts),
  };
    if (next.alwaysVisible === probe.alwaysVisible && next.envFaces === probe.envFaces && JSON.stringify(next.visibleParts) === JSON.stringify(probe.visibleParts)) return false;
  state.environmentProbes.set(id, next);
  emit("environment-probes");
  return true;
}

/**
 * The influence volumes off a caller's record, or the defaults for the box.
 *
 * Absent is not the same as wrong: a manifest written before these fields
 * existed, and every caller that only cares about the projection box, get the
 * derived pair. Present but impossible is refused, because the runtime divides
 * by `outer - inner` per axis - an inner box outside its outer one does not
 * blend backwards, it produces a gradient pointing the wrong way.
 */
function probeInfluence(probe, boxPosition, boxSize) {
  const defaults = defaultProbeInfluence(boxPosition, boxSize);
    const has = probe?.influenceBoxPosition || probe?.influenceBoxSize || probe?.influenceInnerBoxSize;
  if (!has) return defaults;
    const influenceBoxPosition = validProbeVector(probe?.influenceBoxPosition) || defaults.influenceBoxPosition;
    const influenceBoxSize = validProbeVector(probe?.influenceBoxSize, true) || defaults.influenceBoxSize;
    const influenceInnerBoxSize = validProbeVector(probe?.influenceInnerBoxSize) || defaults.influenceInnerBoxSize;
  if (influenceInnerBoxSize.some((n, axis) => n < 0 || n > influenceBoxSize[axis])) return null;
  return { influenceBoxPosition, influenceBoxSize, influenceInnerBoxSize };
}

/**
 * Copy a displayed probe transform back to authored state.
 *
 * The drag/scale gesture already pushed its undo snapshot, so this deliberately
 * does not create another history entry.
 *
 * The camera and the influence volumes ride along: dragging the box across the
 * room is "move this probe", not "leave its capture point and its blend region
 * behind". A resize keeps the margins the author set rather than the ratio -
 * blending is a distance in metres, so growing a corridor by 2 m should not
 * silently widen the fade with it.
 */
export function syncEnvironmentProbeTransform(id, position, size) {
  const probe = state.environmentProbes.get(id);
  const nextPosition = validProbeVector(position);
  const nextSize = validProbeVector(size, true);
  if (!probe || !nextPosition || !nextSize) return false;
  if (probe.shape === "sphere") {
    const sphereRadius = Math.max(...nextSize) * 0.5;
    const delta = nextPosition.map((value, axis) => value - probe.spherePosition[axis]);
    const grow = sphereRadius - probe.sphereRadius;
    const next = {
      ...probe,
      spherePosition: nextPosition,
      sphereRadius,
      capturePosition: probe.capturePosition.map((value, axis) => value + delta[axis]),
      influenceSpherePosition: probe.influenceSpherePosition.map((value, axis) => value + delta[axis]),
      influenceSphereRadius: Math.max(0.01, probe.influenceSphereRadius + grow),
      influenceInnerSphereRadius: Math.max(0, probe.influenceInnerSphereRadius + grow),
    };
    if (JSON.stringify(next) === JSON.stringify(probe)) return false;
    state.environmentProbes.set(id, next);
    emit("environment-probes");
    return true;
  }
  const boxPosition = nextPosition;
  const boxSize = nextSize;
  const delta = boxPosition.map((value, axis) => value - probe.boxPosition[axis]);
  const grow = boxSize.map((value, axis) => value - probe.boxSize[axis]);
  const next = {
    ...probe,
    boxPosition,
    boxSize,
    capturePosition: probe.capturePosition.map((value, axis) => value + delta[axis]),
    influenceBoxPosition: probe.influenceBoxPosition.map((value, axis) => value + delta[axis]),
    influenceBoxSize: probe.influenceBoxSize.map((value, axis) => Math.max(0.01, value + grow[axis])),
    influenceInnerBoxSize: probe.influenceInnerBoxSize.map((value, axis) => Math.max(0, value + grow[axis])),
  };
  if (JSON.stringify(next) === JSON.stringify(probe)) return false;
  state.environmentProbes.set(id, next);
  emit("environment-probes");
  return true;
}

/**
 * Copy a dragged influence volume back to authored state.
 *
 * The gesture is direct manipulation of the outer blend box: its centre is the
 * pair's centre, so moving it takes the inner box with it, and its size is set
 * outright rather than by carrying a margin - unlike a capture-box resize,
 * which is a move of the thing the influence was measured from.
 *
 * A shrunken outer box takes the inner one down with it instead of refusing the
 * gesture. The runtime divides by `outer - inner` per axis, so an inner box
 * left sticking out of its outer one is not a tighter blend, it is a gradient
 * pointing the wrong way - and a drag that silently stops halfway is worse than
 * one that is honest about the volume it is squeezing.
 *
 * Like syncEnvironmentProbeTransform, this deliberately adds no history entry:
 * the gesture pushed its own snapshot before it started moving anything.
 */
export function syncEnvironmentProbeInfluence(id, position, size) {
  const probe = state.environmentProbes.get(id);
  const influencePosition = validProbeVector(position);
  const influenceSize = validProbeVector(size, true);
  if (!probe || !influencePosition || !influenceSize) return false;
  if (probe.shape === "sphere") {
    const influenceSphereRadius = Math.max(...influenceSize) * 0.5;
    const next = {
      ...probe,
      influenceSpherePosition: influencePosition,
      influenceSphereRadius,
      influenceInnerSphereRadius: Math.min(probe.influenceInnerSphereRadius, influenceSphereRadius),
    };
    if (JSON.stringify(next) === JSON.stringify(probe)) return false;
    state.environmentProbes.set(id, next);
    emit("environment-probes");
    return true;
  }
  const influenceBoxPosition = influencePosition;
  const influenceBoxSize = influenceSize;
  const next = {
    ...probe,
    influenceBoxPosition,
    influenceBoxSize,
        influenceInnerBoxSize: probe.influenceInnerBoxSize.map((value, axis) => Math.min(value, influenceBoxSize[axis])),
  };
  if (JSON.stringify(next) === JSON.stringify(probe)) return false;
  state.environmentProbes.set(id, next);
  emit("environment-probes");
  return true;
}

/**
 * Copy a resized inner volume back to authored state.
 *
 * Size only: the inner box has no centre of its own - it shares the outer one's,
 * because a probe that faded out asymmetrically would have to be two probes -
 * so there is nothing for a move gesture to write, which is why the entry is
 * marked immovable rather than being allowed to drift and be corrected after.
 */
export function syncEnvironmentProbeInnerSize(id, size) {
  const probe = state.environmentProbes.get(id);
  const inner = validProbeVector(size);
  if (!probe || !inner) return false;
  if (probe.shape === "sphere") {
    const influenceInnerSphereRadius = Math.min(Math.max(0, Math.max(...inner) * 0.5), probe.influenceSphereRadius);
    const next = { ...probe, influenceInnerSphereRadius };
    if (JSON.stringify(next) === JSON.stringify(probe)) return false;
    state.environmentProbes.set(id, next);
    emit("environment-probes");
    return true;
  }
    const influenceInnerBoxSize = inner.map((value, axis) => Math.min(Math.max(0, value), probe.influenceBoxSize[axis]));
  const next = { ...probe, influenceInnerBoxSize };
  if (JSON.stringify(next) === JSON.stringify(probe)) return false;
  state.environmentProbes.set(id, next);
  emit("environment-probes");
  return true;
}

export function removeEnvironmentProbe(id, history = true) {
  if (!state.environmentProbes.has(id)) return false;
  if (history) pushUndo();
  state.environmentProbes.delete(id);
  // Its blending volumes go with it: they are parts of this probe, and a
  // selection still holding one would resolve to nothing from here on.
  const parts = new Set(PROBE_PARTS.map((part) => environmentProbePartId(id, part)));
  if (state.selection.some((selected) => selected === id || parts.has(selected))) {
        state.selection = state.selection.filter((selected) => selected !== id && !parts.has(selected));
    emit("selection");
  }
  emit("environment-probes");
  return true;
}

/**
 * What an element is called in ship.glb: its own name, or its id when it has
 * none.
 *
 * The single answer to "which node is this", and the key everything the runtime
 * resolves goes through - behaviours, door leaves, the manifest's node
 * references - so the tool can never name a node the export does not.
 *
 * Every element therefore always has a node name. A *given* name is the way to
 * make several elements share one: naming six crates "crate" makes one
 * behaviour entry govern all six, which is the whole point of names and is why
 * they are **not** unique and must not be made unique. An element left unnamed
 * falls back to its id, which no other element can ever carry - so "no name"
 * means "on its own", not "nothing to attach to".
 */
export function nodeNameOf(placement) {
  return String(placement?.name || "").trim() || placement?.id || "";
}

/**
 * What can carry behaviours, and under which key: a placed element under its
 * node name, a **door** under its id, and nothing else.
 *
 * A door's key is its id and can be nothing else, which is also why the
 * inspector offers a marker no Name field. That id is what the manifest writes
 * as the door's `node`, what the portal graph joins two chunks by, and what an
 * `entity` parameter names when one behaviour opens, seals or hides a door from
 * somewhere else - so a door that could be renamed would be a door whose
 * behaviours and whose events answered to two different strings.
 *
 * Lights, probes and colliders answer "": they are not nodes the runtime
 * resolves behaviours against, and a panel offered for them would write entries
 * the manifest has nowhere to put.
 */
export function entityNameOf(entry) {
  if (!entry) return "";
  if (entry.type === "door") return String(entry.id || "");
  return entry.type ? "" : nodeNameOf(entry);
}

/** The elements carrying a node name - several for a shared one, 1 for an id. */
function placementsCarrying(nodeName) {
  const key = String(nodeName || "").trim();
  if (!key) return [];
  return [...state.placements.values()].filter((e) => nodeNameOf(e) === key);
}

/**
 * Give a placement a name, or take its name away.
 *
 * Behaviours hang off the node name, and renaming changes which one this
 * element carries - so they are carried over when, and only when, doing so
 * cannot contradict what a name means:
 *
 *  - **Nothing else carries the old name.** Then the entry belongs to this
 *    element alone and following it along is the only reading that loses
 *    nothing. This is the case that matters for a fresh copy, whose behaviours
 *    sit under its id until it is given a name.
 *  - **The new name is free.** Otherwise the element is joining a name that
 *    already governs others, and the entry it joins is the one that wins -
 *    sharing is what a name is for. Its old entry is dropped rather than left
 *    behind, since by the first rule nothing is carrying it any more.
 *
 * Rename one of six crates and the other five stay governed, exactly as before:
 * the old entry stays put because five elements still carry it.
 */
export function renamePlacement(id, name) {
  const e = state.placements.get(id);
  if (!e) return false;
  const next = String(name || "").trim();
  if (e.name === next) return false;
  pushUndo();
  const before = nodeNameOf(e);
  e.name = next;
  const after = nodeNameOf(e);
  const carried = state.entities.get(before);
  // Only when the old name has nothing left answering to it. While other
  // elements still carry it, every reference to it is still correct, and the
  // new name is simply a name nothing points at yet.
  if (before !== after && !placementsCarrying(before).length) {
    if (carried && !state.entities.has(after)) state.entities.set(after, carried);
    if (carried) state.entities.delete(before);
    renameEntityRefs(before, after);
  }
  emit("placements");
  emit("current");
  emit("behaviors");
  return true;
}

/**
 * Follow a rename into every behaviour that named the old node.
 *
 * A `linked` group, a subscription's source, the target of a raised event: all
 * of them are node names written down, and a rename that left them behind would
 * turn a working ship into one whose faults only show at runtime, as behaviours
 * that quietly do nothing. The schema says which fields hold a node name - see
 * renameEntityReferences - so this cannot fall behind the behaviours the
 * metadata file grows.
 *
 * No undo entry of its own: it is part of the rename, and the caller has
 * already pushed one.
 */
function renameEntityRefs(before, after) {
  if (!behaviorCatalog) return;
  for (const [name, body] of state.behaviors) {
        renameEntityReferences(behaviorCatalog, behaviorBaseName(name), body, before, after);
  }
  for (const [node, list] of state.entities) {
    for (const assignment of list) {
            renameEntityReferences(behaviorCatalog, behaviorBaseName(assignment.name), assignment, before, after);
      // The rename can have pointed a group at its own owner, which says
      // nothing and is the one thing `linked` may not contain.
      if (Array.isArray(assignment.linked)) assignment.linked = cleanLinked(assignment.linked, node);
    }
  }
}

/**
 * The behaviour schema, once it has been fetched.
 *
 * Held rather than imported because it is loaded over the network by the UI
 * layer; until it arrives a rename still moves its entity entry, it simply has
 * nothing to say about which fields hold node names.
 */
let behaviorCatalog = null;
export function setBehaviorCatalog(catalog) {
    behaviorCatalog = catalog;
}

function assertValidBehaviorConfig(name, value, partial, path) {
  if (!behaviorCatalog) {
    throw new Error("behavior metadata is not loaded");
  }
  const errors = validateBehaviorConfig(behaviorCatalog, name, value, { partial });
  if (errors.length) {
    throw new Error(`${path}: ${errors.join(" ")}`);
  }
}

// ------------------------------------------------------------- behaviours
//
// Two halves, matching the manifest:
//
//   behaviors   named presets derived from one catalog behavior.
//   entities    which base behaviors or presets a node name carries, plus the
//               `linked` node names some of them need.
//
// Both are keyed by NODE NAME - `nodeNameOf`, so a named element by its name
// and an unnamed one by its id. Names are shared on purpose, so one entry can
// govern every element carrying it; an id is carried by exactly one element, so
// leaving something unnamed is how it comes to have behaviours of its own.

/** Parameter overrides inherited from a preset, or base-behavior constants. */
export function getBehaviorDef(name) {
    const key = String(name || "").trim();
    const preset = state.behaviors.get(key);
    if (preset) {
        const body = JSON.parse(JSON.stringify(preset));
        delete body.base;
        return body;
    }
    const metadata = behaviorMetadata(behaviorCatalog, key);
    return metadata ? defaultBehaviorDefinition(metadata) : null;
}

/** Raw named preset, including its base behavior identity. */
export function getBehaviorPreset(name) {
    const preset = state.behaviors.get(String(name || "").trim());
    return preset ? JSON.parse(JSON.stringify(preset)) : null;
}

/** Every preset in insertion order. */
export function behaviorNames() {
    return [...state.behaviors.keys()];
}

/** Base catalog behaviors and named presets, sorted by the caller for display. */
export function availableBehaviorNames() {
    return [...new Set([...behaviorMetadataNames(behaviorCatalog), ...behaviorNames()])];
}

/** Executable catalog behavior behind a base name or preset name. */
export function behaviorBaseName(name) {
    const key = String(name || "").trim();
    return state.behaviors.get(key)?.base ?? key;
}

/**
 * Create or replace a named preset. `base` identifies one catalog behavior;
 * the remaining fields are its shared parameter overrides.
 */
export function setBehaviorDef(name, body) {
  const key = String(name || "").trim();
  if (!key) return false;
  if (!body || typeof body !== "object" || Array.isArray(body)) return false;
  const next = JSON.parse(JSON.stringify(body));
    const base = String(next.base || "").trim();
    if (!base || !behaviorMetadata(behaviorCatalog, base) || behaviorMetadata(behaviorCatalog, key)) return false;
    next.base = base;
    if (base === "liquefaction") delete next.liquefiable;
    const params = { ...next };
    delete params.base;
    assertValidBehaviorConfig(base, params, true, `behavior preset "${key}"`);
  if (JSON.stringify(state.behaviors.get(key) ?? null) === JSON.stringify(next)) return false;
  pushUndo();
  state.behaviors.set(key, next);
  emit("behaviors");
  return true;
}

/** Delete a preset and strip it from every entity that carried it. */
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

/**
 * The behaviours a node name carries, as a copy.
 *
 * Every parameter has already been validated against the metadata catalogue.
 */
export function entityBehaviors(nodeName) {
  const list = state.entities.get(String(nodeName || "").trim()) || [];
  return list.map((b) => ({
    ...JSON.parse(JSON.stringify(b)),
    name: b.name,
    linked: [...(b.linked || [])],
  }));
}

/**
 * Give one element the behaviours another carries, as its own.
 *
 * What a duplicate needs. Behaviours hang off the node name, so a copy that
 * kept its source's name would *share* the source's entry rather than have one
 * - move the original's fan out of the room and the copy's would follow, and
 * `linked` could no longer tell the two apart. A copy is therefore made
 * nameless and given its own entry under its id, which nothing else can carry.
 *
 * The target's list is replaced rather than added to, so this says exactly what
 * the copy ends up with: pass `copy: false` and it ends up with nothing. Either
 * way an entry already sitting under that key is cleared, which also sweeps up
 * the one an element deleted before a reload could have left behind.
 *
 * `linked` comes across as written. The copy of a door leaf therefore still
 * names the original's twin, which is the only honest answer - the leaf it
 * ought to pair with does not exist yet - and the panel shows it plainly.
 */
export function copyBehaviorsTo(fromId, toId, { copy = true } = {}) {
  const to = state.placements.get(toId);
  if (!to) return false;
  const toKey = nodeNameOf(to);
  const from = state.placements.get(fromId);
  const fromKey = from ? nodeNameOf(from) : "";
  // Sharing a name is the one case with nothing to do: the entry already
  // governs both, and writing a second copy under the same key is a no-op at
  // best and a way to lose the source's edits at worst.
  if (fromKey && fromKey === toKey) return false;
  const carried = copy ? state.entities.get(fromKey) : null;
  if (!carried?.length) return state.entities.delete(toKey);
  state.entities.set(toKey, JSON.parse(JSON.stringify(carried)));
  return true;
}

/**
 * Attach a library behaviour to a node name.
 *
 * The same behaviour may be attached **more than once**. An entity's behaviours
 * are a *list* of assignments, not a set keyed by name: the runtime walks the
 * list and builds one instance per entry, so two entries of the same behaviour
 * with different parameters are two different things happening to one prop.
 *
 * Which is why nothing below identifies an assignment by name - a name is not
 * unique within a node - and every editing call takes its **index** instead.
 */
export function addEntityBehavior(nodeName, behaviorName) {
  const node = String(nodeName || "").trim();
    if (!node || !availableBehaviorNames().includes(behaviorName)) return false;
  const list = state.entities.get(node) || [];
  pushUndo();
  state.entities.set(node, [...list, { name: behaviorName, linked: [] }]);
  emit("behaviors");
  return true;
}

/** Where an assignment sits in a node's list, or -1 if that is not a slot. */
function assignmentAt(list, index) {
  const at = Number(index);
  return Array.isArray(list) && Number.isInteger(at) && at >= 0 && at < list.length ? at : -1;
}

/** Detach one assignment, by its position in the node's list. */
export function removeEntityBehavior(nodeName, index) {
  const node = String(nodeName || "").trim();
  const list = state.entities.get(node);
  const at = assignmentAt(list, index);
  if (at < 0) return false;
  pushUndo();
  const kept = list.filter((_, i) => i !== at);
  if (kept.length) state.entities.set(node, kept);
  else state.entities.delete(node);
  emit("behaviors");
  return true;
}

/** The other nodes this one drags along with it - a door half links its twin. */
export function setEntityLinked(nodeName, index, linked) {
  const node = String(nodeName || "").trim();
  const list = state.entities.get(node);
  const at = assignmentAt(list, index);
  if (at < 0) return false;
  const entry = list[at];
  const next = cleanLinked(linked, node);
  if (JSON.stringify(entry.linked) === JSON.stringify(next)) return false;
  pushUndo();
  entry.linked = next;
  emit("behaviors");
  return true;
}

/** A `linked` list with the duplicates, the blanks and the self-reference out. */
function cleanLinked(linked, node) {
  if (!Array.isArray(linked)) return [];
    return [...new Set(linked.map((s) => String(s).trim()).filter(Boolean))].filter((n) => n !== node); // linking a node to itself says nothing
}

/** True when a definition asks for liquefaction, which is what needs `linked`. */
export function isLiquefiable(behaviorName) {
    return behaviorBaseName(behaviorName) === "liquefaction";
}

/**
 * Whether a node carries a behaviour that gives it a rigid body.
 *
 * The runtime's own rule: `dynamic` alone. `liquefiable` does NOT imply it - a
 * mesh can be authored to melt without ever having been a rigid body.
 */
export function isDynamicNode(nodeName) {
    return entityBehaviors(nodeName).some((b) => behaviorBaseName(b.name) === "dynamic");
}

/**
 * The behaviour the runtime recognises as the first-person weapon.
 *
 * Matched by NAME rather than by a flag because that is the runtime's own
 * contract: `behavior-manager.ts` switches on `assignment.name`, so the name is
 * the thing that decides, and a definition body invented here to mirror it
 * would be a second source of truth that nothing enforces.
 */
const WEAPON_BEHAVIORS = ["weaponLiquefactor", "weaponAntiGravityGun", "weaponPistol"];
export const PLAY_ANIMATION_BEHAVIOR = "playAnimation";
export const HIDE_ENTITY_BEHAVIOR = "hideEntity";

/**
 * The parameters an assignment actually carries - everything but its identity.
 *
 * Two keys are not parameters. `name` is the assignment's identity, and an
 * EMPTY `linked` is the absence of one: setEntityParams and addEntityBehavior
 * both normalise that field, so every assignment ever touched carries a
 * `linked: []` that the manifest then omits when it writes it out. Counting
 * either would make "this has parameters" true for everything.
 *
 * This is the panel's own rule - the typed controls edit exactly this object,
 * so anything asking "was this left blank?" has to ask here rather than inspect
 * the assignment itself.
 */
export function behaviorParams(assignment) {
  const out = {};
  for (const [key, value] of Object.entries(assignment || {})) {
    if (key === "name") continue;
    if (key === "linked" && !value?.length) continue;
    out[key] = value;
  }
  return out;
}

/**
 * Whether a node is one the game hides before the player ever sees it.
 *
 * `hideEntity` with no parameters is the runtime's "hide me, now":
 * `entity-toggle.ts` emits the hide event against its own entity as soon as it
 * starts. An `events` array instead delays that action until a subscription
 * matches.
 *
 * So this is a statement about the ship, not about the editor: it is true of a
 * trap that starts buried whether or not anything is currently previewing it.
 * The callers are what decide when it BITES - the Runtime view and the probe
 * captures, both only while Run behaviours is on.
 */
export function isHiddenAtStartNode(nodeName) {
    return entityBehaviors(nodeName).some((b) => behaviorBaseName(b.name) === HIDE_ENTITY_BEHAVIOR && !Object.keys(behaviorParams(b)).length);
}

/** The ship elements isHiddenAtStartNode speaks for, for the status line. */
export function behaviorHiddenPlacements() {
  return shipPlacements().filter((e) => isHiddenAtStartNode(nodeNameOf(e)));
}

/**
 * Whether a node's geometry must be kept OUT of an environment probe.
 *
 * A probe is a photograph of a room's fixed geometry, taken once and worn by
 * every material in it. Anything that will not be standing exactly there for
 * the life of the level has to be left out, or the room reflects a crate that
 * has since been pushed over and a weapon that is really in the player's hands.
 *
 * Four ways to earn it, and all four say the same thing:
 *
 *  - `dynamic: true` — a rigid body. Its authored pose is a starting position,
 *    not a fact about the room.
 *  - `liquefaction` — it is going to melt. What the probe would record is
 *    its shape before the game begins.
 *  - `playAnimation` — its exported transform changes at runtime.
 *  - the weapon behaviour — a first-person viewmodel rides the camera, so it is
 *    never in the room at all; the placement is only where it is picked up.
 *
 * plus the authored opt-out, `reflectionProbe: "exclude"`, for anything fixed
 * that still must not be photographed.
 *
 * Excluded geometry drops out of the capture's DIGEST too - see
 * meshesInProbeBox, which is the single list both are built from - so nudging a
 * crate no longer marks every probe in its room as stale.
 */
export function isProbeExcludedNode(nodeName) {
  return entityBehaviors(nodeName).some((b) => {
        const base = behaviorBaseName(b.name);
        if (WEAPON_BEHAVIORS.includes(base) || base === PLAY_ANIMATION_BEHAVIOR) return true;
        return base === "probeExcluded" || base === "dynamic" || base === "liquefaction";
  });
}

/**
 * The parameters an applied behaviour carries, replaced wholesale.
 *
 * The metadata-driven panel edits a local object and replaces the assignment
 * when its typed values are valid.
 *
 * Only three things are normalised, and each is a rule this tool owns rather
 * than the runtime:
 *
 *  - `name` is the identity of the assignment, never a parameter, so a `name`
 *    supplied by authoring data is ignored.
 *  - `linked` is a node-name list this tool builds the picker from, so it gets
 *    the same cleaning the picker applies.
 *  - `direction` is stored as typed rather than normalised - normalising on
 *    every commit fights you as you fill it in - but rounded, and an all-zero
 *    vector is dropped: it names no direction, and writing it out would ask the
 *    runtime to face nowhere.
 */
export function setEntityParams(nodeName, index, params) {
  const node = String(nodeName || "").trim();
  const list = state.entities.get(node);
  const at = assignmentAt(list, index);
  if (at < 0 || !params || typeof params !== "object" || Array.isArray(params)) return false;
  const entry = list[at];
  const next = { name: entry.name, linked: cleanLinked(params.linked, node) };
  for (const [key, value] of Object.entries(params)) {
    if (key === "name" || key === "linked") continue;
    if (value === undefined) continue;
    next[key] = JSON.parse(JSON.stringify(value));
  }
  if (isVector3(next.direction)) {
    if (next.direction.some((v) => v !== 0)) next.direction = next.direction.map((v) => Math.round(v * 1e4) / 1e4);
    else delete next.direction;
  }
    assertValidBehaviorConfig(
        behaviorBaseName(entry.name),
        {
            ...(getBehaviorDef(entry.name) ?? {}),
    ...behaviorParams(next),
        },
        false,
        `entity "${node}" behavior "${entry.name}"`
    );
  if (JSON.stringify(entry) === JSON.stringify(next)) return false;
  pushUndo();
  list[at] = next;
  emit("behaviors");
  return true;
}

/** A finite 3-number array - what both `direction` and a position look like. */
function isVector3(v) {
  return Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
}

/**
 * Node names in a chunk - the candidates for `linked`.
 *
 * The current room, not the whole ship: linking is for pieces that behave as
 * one, which in practice are neighbours, and a ship-wide list would be hundreds
 * of entries long.
 *
 * Named elements, plus unnamed ones that carry behaviours of their own. An
 * unnamed element has a node name - its id - so it *can* be linked to, but
 * offering every one of them would bury the handful worth linking under a room
 * full of `P0042`s. Carrying a behaviour is what makes one a participant.
 *
 * Several chunks may be asked for at once, which is what a door needs: a door
 * is not *in* a room, it joins two, and both sides are equally its neighbours.
 */
export function nodeNamesInChunk(chunk, exclude = "") {
  const rooms = new Set((Array.isArray(chunk) ? chunk : [chunk]).filter(Boolean));
  const names = new Set();
  for (const e of state.placements.values()) {
    if (!rooms.has(e.chunk)) continue;
    const named = String(e.name || "").trim();
    const n = named || (state.entities.has(e.id) ? e.id : "");
    if (n && n !== exclude) names.add(n);
  }
  return [...names].sort((a, b) => a.localeCompare(b));
}

/** How many elements carry a node name. Zero means the entry is orphaned. */
export function nodesNamed(name) {
  const key = String(name || "").trim();
  if (!key) return 0;
  return placementsCarrying(key).length;
}

export function assignSelectionToChunk(chunk) {
    if (!state.selection.length) return;
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

export function isVeilClone(mesh) {
    return !!mesh?.metadata?.veilClone;
}

/**
 * Editor furniture hanging off an element's node rather than part of its art.
 *
 * Only a light's gizmo, so far - and only lights are parented to another
 * element at all. Everything that walks a placement's meshes has to skip it: it
 * is not a primitive to export, not a shape to measure, and veiling a wall must
 * not clone the lamp inside it.
 */
export function isGizmoMesh(m) {
    return !!m?.metadata?.gizmo;
}

/**
 * A stand-in the Runtime view draws in place of an authored instance.
 *
 * Same reason as a veil clone, and the same treatment: it is a copy that
 * exists only to be looked at (see runtime.js dressMeshes), so nothing that
 * walks an element's meshes should see it - not the exporter, not the veil,
 * not the picker.
 */
export function isRuntimeStandIn(m) {
    return !!m?.metadata?.runtimePreview;
}

function realMeshes(node) {
  return node.getChildMeshes().filter((m) => !isVeilClone(m) && !isGizmoMesh(m) && !isRuntimeStandIn(m));
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
  // A bench is a separate world: while one is open the ship is hidden, never
  // touched, so closing it puts everything back. Hiding and the veil work on
  // both, which is what makes H behave the same in either place. The mode name
  // doubles as the bench name, so only the bench that is actually open shows -
  // the other one's leftovers, if any, stay dark.
  const bench = state.mode === "ship" ? "" : state.mode;
  // The layer switch composes with everything else rather than fighting it: it
  // can only ever take things *off* screen, so chunk isolation and the Shift+H
  // veil keep the last word on what is left.
  const geometryOn = state.showLayer !== "collision";
  const collisionOn = state.showLayer !== "geometry";
  // What the game hides before the player arrives is hidden here too, but only
  // where that claim is being made: the Runtime view, with Run behaviours on.
  // The editor view has to keep drawing it, or a trap would be unselectable in
  // the only view you can author it in.
    const hiddenAtStart = state.runtime && state.runBehaviors ? (e) => isHiddenAtStartNode(nodeNameOf(e)) : () => false;

  for (const e of state.placements.values()) {
    const veil = veilOf(e.id);
        const on = (e.stage ? benchOf(e) === bench : !bench && geometryOn && (!state.isolate || e.chunk === state.activeChunk) && !hiddenAtStart(e)) && veil !== "hidden";
    e.node.setEnabled(on);
    setVeil(e, veil === "ghost");
    for (const m of realMeshes(e.node)) {
      // A ghosted element is drawn by its veil clones *alone* - buildVeilClones
      // turned the real meshes off, and turning them back on here would draw
      // the solid original through its own ghost.
      if (veil !== "ghost") m.setEnabled(true);
      m.isPickable = veil !== "ghost";
    }
  }
  // A door is not in a chunk, but it *joins* two - so isolation does have
  // something to say about it: show it when the active chunk is one of its two
  // sides. Sides left on "(auto)" are resolved the same way the manifest
  // resolves them, by nearest chunk volume, so what you see under isolation is
  // what will be written.
  const sidesOf = doorSideResolver();
  for (const mk of state.markers.values()) {
    const veil = veilOf(mk.id);
        const joins = mk.type !== "door" || !state.isolate || sidesOf(mk).includes(state.activeChunk);
    mk.node.setEnabled(!bench && geometryOn && joins && veil !== "hidden");
    setVeil(mk, veil === "ghost");
    for (const m of realMeshes(mk.node)) m.isPickable = veil !== "ghost";
  }
  // A room's primitives belong to a chunk and follow isolation like placements.
  // The staging area's are a working copy of what each module carries, and only
  // exist while it is open.
  for (const c of state.colliders.values()) {
    const veil = veilOf(c.id);
        const on = (c.stage ? bench === "collision" : collisionOn && !bench && (!state.isolate || c.chunk === state.activeChunk)) && veil !== "hidden";
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
    if (
        state.selection.some((id) => {
    const e = entryOf(id);
    return e && (e.type === "collider" ? layer === "geometry" : layer === "collision");
        })
    )
        select([]);
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
            let min = null,
                max = null;
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
  const range = CONFIG_RANGE[key] || { min: 0, max: Infinity };
  if (range.choices) {
    const v = String(value);
    if (!range.choices.includes(v) || state.config[key] === v) return false;
    pushUndo();
    state.config = { ...state.config, [key]: v };
    emit("config");
    return true;
  }
  const v = Number(value);
    if (range.values) {
        if (!range.values.includes(v) || state.config[key] === v) return false;
        pushUndo();
        state.config = { ...state.config, [key]: v };
        emit("config");
        return true;
    }
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

export function hiddenCount() {
    return state.hidden.size;
}

/** How many are at each level, for the status bar. */
export function veilCounts() {
    let ghost = 0,
        hidden = 0;
  for (const level of state.hidden.values()) {
        if (level === "hidden") hidden++;
        else ghost++;
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
  serializeLights: () => [],
  deserializeLights: () => {},
  removeLightsOf: () => {},
  removeLight: () => {},
  copyLightsTo: () => {},
  seedKitLights: () => [],
  lightsForExport: () => [],
  reconcileCollider: () => false,
  // interact.js owns the placement ghost; the axes need its node, and importing
  // interact.js back would close a cycle
  ghostNode: () => null,
  // palette.js owns the brush, and imports interact.js - so putting the tile
  // back down after a one-shot drop has to come back through here
  clearBrush: () => {},
  // runtime.js registers this: the probe volume gizmo is its own, and the
  // inspector needs to be able to select and edit it like any other entry.
  environmentProbeEntry: () => null,
  // runtime.js registers this too. The behaviour preview drives the skinned
  // props' bones, and a GLB taken mid-clip would bake that pose in as the
  // exported rest transform. Returns the function that puts playback back.
  pauseBehaviorAnimations: () => () => {},
  // compounds.js registers these: the compound bench keeps its own history, and
  // the registry above has to be able to snapshot it without importing it back.
  serializeBench: () => ({ members: [] }),
  restoreBench: () => {},
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
 * These are the values tuned in **Runtime** mode - the authoring rig switched
 * off, the authored lamps and the captured probes in charge - because that is
 * the picture the demos render. `strength` is `scene.environmentIntensity`,
 * `exposure` is the linear multiplier exactly as the slider shows it, and
 * `toneMapping` names the view transform both ends apply.
 *
 * `specularAA` and `reflectionRoughness` are the same kind of value one step
 * further in: they land on the materials rather than the scene. They are here
 * and not in the editor block because the game reads them - `specularAA` as
 * the authored default for a toggle the player may still override, and
 * `reflectionRoughness` as a multiplier over every ship material's authored
 * roughness, exactly as the preview applies it.
 */
export function serializeEnvironment() {
  const set = state.lightSets.runtime;
  return {
    strength: round3([set.strength])[0],
    toneMapping: set.toneMapping ?? TONE_MAPPING_DEFAULT,
    exposure: round3([set.exposure])[0],
    specularAA: !!state.runtimeSpecularAA,
    reflectionRoughness: round3([state.runtimeRoughnessFactor])[0],
  };
}

/**
 * The editor's own rig, kept apart from `environment` because it describes
 * this tool and not the ship. The authoring rig adds four analytic lights the
 * game does not have, so the two need different numbers to look right, and
 * writing the editor's into `environment` would blow the demos out.
 */
export function serializeEditorEnvironment() {
  const set = state.lightSets.editor;
  return {
    strength: round3([set.strength])[0],
    toneMapping: set.toneMapping ?? TONE_MAPPING_DEFAULT,
    exposure: round3([set.exposure])[0],
  };
}

/**
 * The editor's own view preferences, saved with the ship.
 *
 * Not lighting, so not in `editorEnvironment`, and nothing the demos may read.
 * They are in the manifest all the same: a ship reopened with the palette half
 * the size you left it, or ghosts at someone else's opacity, is the tool having
 * forgotten how you were working on *this* ship. The localStorage copy stays as
 * the fallback for a ship whose manifest predates the block.
 */
export function serializeEditorPrefs() {
  // How each probe was being looked at: whether it stays on screen, whether it
  // draws its captured faces, and which of its three volumes are drawn at all.
  // Written here rather than in `environmentProbes` because they are how you
  // were *looking* at the ship, not part of it: the game reads that array, and
  // the editor's viewport has no business in it. Only probes with something to
  // say are listed, so a ship nobody has fiddled with writes an empty object.
  const probes = {};
  for (const [id, probe] of state.environmentProbes) {
    const hiding = PROBE_VOLUMES.some((volume) => probe.visibleParts?.[volume] === false);
    if (!probe.alwaysVisible && !probe.envFaces && !hiding) continue;
    probes[id] = {
      alwaysVisible: !!probe.alwaysVisible,
      envFaces: !!probe.envFaces,
      visibleParts: probeVisibleParts(probe.visibleParts),
    };
  }
  return {
    veilAlpha: round3([state.veilAlpha])[0],
    bigPalette: !!state.bigPalette,
    strayChunkCheck: !!state.strayChunkCheck,
    runBehaviors: !!state.runBehaviors,
    shipOptimize: !!state.shipOptimize,
    probes,
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
    // Stored on the set, not through setToneMapping: that writes whichever set
    // is live, which on a load from an editor view is the wrong one.
    if (env.toneMapping) state.lightSets.runtime.toneMapping = String(env.toneMapping);
    // Through their setters, which clamp and announce "reflection" - the event
    // the preview materials and the Settings pane both listen for.
    if (typeof env.specularAA === "boolean") setRuntimeSpecularAA(env.specularAA);
    if (Number.isFinite(env.reflectionRoughness)) setRuntimeRoughnessFactor(env.reflectionRoughness);
  }
  if (editorEnv && typeof editorEnv === "object") {
    if (Number.isFinite(editorEnv.strength)) state.lightSets.editor.strength = editorEnv.strength;
    if (Number.isFinite(editorEnv.exposure)) {
      state.lightSets.editor.exposure = asLinearExposure(editorEnv.exposure);
    }
    if (editorEnv.toneMapping) state.lightSets.editor.toneMapping = String(editorEnv.toneMapping);
  } else if (env && typeof env === "object") {
    // A manifest from before the split has one rig; give it to both rather
    // than leaving the editor on defaults that have nothing to do with it.
    state.lightSets.editor = { ...state.lightSets.runtime };
  }
  const set = activeLightSet();
  setEnvIntensity(set.strength);
  setExposure(set.exposure);
  setToneMapping(set.toneMapping ?? TONE_MAPPING_DEFAULT);
  emit("environment");
  return true;
}

/**
 * Put the editor's view preferences back on load.
 *
 * Returns false when the manifest carries no block, which is the signal to keep
 * whatever boot read out of localStorage rather than snapping to the defaults.
 * `bigPalette` is stored and announced only - what it means on screen is the
 * palette's business, and the palette lives in the UI layer.
 */
export function applyEditorPrefs(prefs) {
  if (!prefs || typeof prefs !== "object") return false;
  if (Number.isFinite(prefs.veilAlpha)) setVeilAlpha(prefs.veilAlpha);
  if (typeof prefs.bigPalette === "boolean") state.bigPalette = prefs.bigPalette;
  if (typeof prefs.strayChunkCheck === "boolean") state.strayChunkCheck = prefs.strayChunkCheck;
  if (typeof prefs.runBehaviors === "boolean") state.runBehaviors = prefs.runBehaviors;
  if (typeof prefs.shipOptimize === "boolean") state.shipOptimize = prefs.shipOptimize;
  // Probes are already in by the time this runs; an id the block names but the
  // ship no longer has is simply ignored, which is how a deleted probe stops
  // being mentioned without anybody having to prune the block.
  if (prefs.probes && typeof prefs.probes === "object") {
    for (const [id, view] of Object.entries(prefs.probes)) {
      setEnvironmentProbeView(id, {
        alwaysVisible: !!view?.alwaysVisible,
        envFaces: !!view?.envFaces,
        // A block written before the eyes were per probe names no volumes, and
        // leaving them alone is what draws all three.
        visibleParts: view?.visibleParts,
      });
    }
  }
  emit("prefs");
  return true;
}

/**
 * Exposure used to be written in STOPS and raised to a power at the other end.
 * It is the plain multiplier now - what the slider shows is what every consumer
 * applies - but a *negative* value can only have come from the old format,
 * since the slider has never gone below 0.15. That makes the conversion
 * unambiguous, and worth doing rather than clamping a real setting up to the
 * floor.
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
    probeVolumes: [...state.environmentProbes.values()].map(cloneEnvironmentProbe),
    // The runtime rig's Env/Exposure/tone. On the stack because they are
    // *authored* values that the manifest carries and the runtime reads -
    // getting the lighting wrong is as much an edit to undo as moving a wall.
    // It travels whether or not the viewport is currently rendering it, or an
    // undo made in an editor view would leave the runtime rig behind.
    //
    // The *editor* rig is deliberately absent, for the same reason the camera
    // is (see serializeView): it is how you happen to be looking at the ship in
    // this browser, not part of the ship. Its slider does not push an undo
    // entry, so restoring it here would take back a change no entry recorded.
    lightSets: { runtime: { ...state.lightSets.runtime } },
    // The two material dials the Runtime section carries. Authored values the
    // manifest saves and the game reads, so they undo like the rig above them;
    // the editor rig's rows, next to them in the pane, still do not.
    runtimeSpecularAA: state.runtimeSpecularAA,
    runtimeRoughnessFactor: state.runtimeRoughnessFactor,
    // Ship data, so it belongs on the undo stack with everything else - and it
    // survives the clearAll() that a restore begins with.
        behaviorPresets: Object.fromEntries([...state.behaviors].map(([k, v]) => [k, JSON.parse(JSON.stringify(v))])),
    // Not ship data, but on the stack all the same: a restore clears it, so
    // leaving it out made every undo reveal what H had parked away.
        hidden: [...state.hidden],
        entities: Object.fromEntries([...state.entities].map(([k, v]) => [k, { behaviors: v.map((b) => ({ name: b.name, ...writeBehaviorParams(b) })) }])),
    instances: shipPlacements().map((e) => ({
      id: e.id,
      module: e.module,
      chunk: e.chunk,
      ...(e.name ? { name: e.name } : {}),
      // Only written when they are set, so an ordinary ship reads exactly as it
      // did before compounds existed and its diffs stay legible.
      ...(e.group ? { group: e.group } : {}),
      ...(e.compound ? { compound: e.compound } : {}),
      position: round3(e.node.position.asArray()),
      rotation: round3(eulerOf(e.node)),
      scale: round3(e.node.scaling.asArray()),
    })),
    colliders: hooks.serializeColliders(),
    markers: hooks.serializeMarkers(),
    lights: hooks.serializeLights(),
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
        stageLayout: (hooks.stageLayoutNow?.() || state.stageLayout).map((s) => ({ module: s.module, position: [...s.position] })),
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
        state.moduleCollision.set(
            moduleId,
            shapes
      .filter((s) => s && s.kind && Array.isArray(s.position))
      .map((s) => ({
        kind: s.kind,
        position: [...s.position],
        rotation: [...(s.rotation || [0, 0, 0])],
        scale: [...(s.scale || [1, 1, 1])],
                }))
        );
  }
  if (Array.isArray(stageLayout)) {
        state.stageLayout = stageLayout.filter((s) => s && typeof s.module === "string" && Array.isArray(s.position)).map((s) => ({ module: s.module, position: [...s.position] }));
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
                const q = Array.isArray(s.rotation) ? new Quaternion(-s.rotation[0], s.rotation[1], s.rotation[2], -s.rotation[3]) : Quaternion.Identity();
        const e = q.toEulerAngles();
        made.push({
          kind: "box",
          position,
          rotation: [e.x, e.y, e.z].map((r) => ( r * 180) / Math.PI),
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
            rotation = [e.x, e.y, e.z].map((r) => ( r * 180) / Math.PI);
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

/**
 * The cubemap size a ship authored before the setting existed was built at.
 *
 * Every probe used to carry its own, and the runtime now holds the captures in
 * one cube texture array - a single dimension for every slice. Taking the
 * LARGEST of what was authored is the only answer that does not depend on the
 * order the probes happen to be written in and never quietly downsamples a
 * room; those old values were free-form, so it is rounded up to the nearest
 * size the setting offers. Null when nothing readable was written.
 */
function legacyProbeResolution(probes) {
    let largest = 0;
    for (const probe of probes) {
        const n = Math.round(Number(probe?.resolution));
        if (Number.isFinite(n)) largest = Math.max(largest, n);
    }
    if (largest <= 0) return null;
    return PROBE_RESOLUTIONS.find((n) => n >= largest) ?? PROBE_RESOLUTIONS.at(-1);
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
  state.environmentProbes.clear();
  const hasExplicitProbes = Array.isArray(data.probeVolumes)
    || Array.isArray(data.environmentProbes);
  let probes = Array.isArray(data.probeVolumes)
    ? data.probeVolumes
    :Array.isArray(data.environmentProbes)
      ? data.environmentProbes.map((probe) => ({
        id: probe.id,
        shape: probe.shape === "sphere" ? "sphere" : "box",
        ...(probe.shape === "sphere"
          ? {
              spherePosition: Array.isArray(probe.spherePosition)
                              ? [-Number(probe.spherePosition[0]), Number(probe.spherePosition[1]), Number(probe.spherePosition[2])]
                              : null,
              sphereRadius: probe.sphereRadius,
              influenceSpherePosition: Array.isArray(probe.influenceSpherePosition)
                              ? [-Number(probe.influenceSpherePosition[0]), Number(probe.influenceSpherePosition[1]), Number(probe.influenceSpherePosition[2])]
                              : null,
              influenceSphereRadius: probe.influenceSphereRadius,
              influenceInnerSphereRadius: probe.influenceInnerSphereRadius,
            }
          : {
                          boxPosition: Array.isArray(probe.boxPosition) ? [-Number(probe.boxPosition[0]), Number(probe.boxPosition[1]), Number(probe.boxPosition[2])] : null,
              boxSize: probe.boxSize,
              angle: Number.isFinite(Number(probe.angle)) ? -Number(probe.angle) : 0,
              // Absent on anything written before influence volumes were authored,
              // where the record takes the defaults instead - which are what that
              // ship was already blending with.
              influenceBoxPosition: Array.isArray(probe.influenceBoxPosition)
                              ? [-Number(probe.influenceBoxPosition[0]), Number(probe.influenceBoxPosition[1]), Number(probe.influenceBoxPosition[2])]
                              : null,
              influenceBoxSize: probe.influenceBoxSize,
              influenceInnerBoxSize: probe.influenceInnerBoxSize,
            }),
        capturePosition: Array.isArray(probe.capturePosition)
                    ? [-Number(probe.capturePosition[0]), Number(probe.capturePosition[1]), Number(probe.capturePosition[2])]
                    : null,
                clipCapture: probe.clipCapture !== false,
                // Not a probe field any more - see legacyProbeResolution, which is the
                // only thing that still reads it.
        resolution: probe.resolution,
      }))
      : [];
  // Manifests written before explicit volumes stored one projection override
  // on each chunk. Promote those records instead of dropping authored work.
  if (!hasExplicitProbes) {
    let generated = {};
    try {
      const response = await fetch("/api/local-environments", { cache: "no-store" });
      if (response.ok) generated = (await response.json()).chunks || {};
    } catch {
      // The generated index is optional; chunk bounds remain a safe fallback.
    }
    probes = (data.chunks || [])
      .filter((chunk) => chunk && typeof chunk === "object" && chunk.environmentProbe)
      .map((chunk, index) => {
        const legacy = chunk.environmentProbe;
                const position = Array.isArray(legacy.boxPosition) ? [-Number(legacy.boxPosition[0]), Number(legacy.boxPosition[1]), Number(legacy.boxPosition[2])] : null;
        const generatedPosition = generated[chunk.id]?.position;
        const bounds = chunk.aabb;
        const automatic = Array.isArray(generatedPosition)
          ? generatedPosition
          :Array.isArray(bounds?.min) && Array.isArray(bounds?.max)
            ? bounds.min.map((value, axis) => (Number(value) + Number(bounds.max[axis])) * 0.5)
            : null;
        const capturePosition = Array.isArray(automatic)
          ? [-Number(automatic[0]), Number(automatic[1]), Number(automatic[2])]
          : position;
        return {
          id: `ENV${String(index + 1).padStart(4, "0")}`,
          boxPosition: position,
          boxSize: legacy.boxSize,
          capturePosition,
                    angle: Number(generated[chunk.id]?.angle) || 0,
          resolution: Number(generated[chunk.id]?.resolution) || 512,
        };
      });
    }
    // A ship authored before the cubemap size became a ship-wide setting carries
    // it on every probe instead, so read the setting back out of them. The test
    // is on what the FILE said rather than on the merged config, which has
    // already taken the default; and it also catches a hand-edited value the
    // texture array could not hold.
    if (!PROBE_RESOLUTIONS.includes(Number(data.config?.probeResolution))) {
        state.config = {
            ...state.config,
            probeResolution: legacyProbeResolution(probes) ?? CONFIG_DEFAULTS.probeResolution,
        };
  }
  for (const probe of probes) {
    const id = validEnvironmentProbeId(probe?.id);
        const capturePosition = validProbeVector(probe?.capturePosition || probe?.spherePosition || probe?.boxPosition);
    if (!id || !capturePosition || !environmentProbeIdAvailable(id)) continue;
    if (probe?.shape === "sphere") {
      const spherePosition = validProbeVector(probe.spherePosition);
      const sphereRadius = Number(probe.sphereRadius);
      if (!spherePosition || !Number.isFinite(sphereRadius) || sphereRadius <= 0) continue;
      const defaults = defaultSphereProbeInfluence(spherePosition, sphereRadius);
            const influenceSpherePosition = validProbeVector(probe.influenceSpherePosition) || defaults.influenceSpherePosition;
      const outer = Number(probe.influenceSphereRadius);
            const influenceSphereRadius = Number.isFinite(outer) && outer > 0 ? outer : defaults.influenceSphereRadius;
      const inner = Number(probe.influenceInnerSphereRadius);
            const influenceInnerSphereRadius =
                Number.isFinite(inner) && inner >= 0 && inner <= influenceSphereRadius ? inner : Math.min(defaults.influenceInnerSphereRadius, influenceSphereRadius);
            setEnvironmentProbe(
                id,
                {
                    ...probe,
                    shape: "sphere",
                    spherePosition,
                    sphereRadius,
                    capturePosition,
                    influenceSpherePosition,
                    influenceSphereRadius,
                    influenceInnerSphereRadius,
                },
                id,
                { history: false }
            );
      continue;
    }
    const boxPosition = validProbeVector(probe?.boxPosition);
    const boxSize = validProbeVector(probe?.boxSize, true);
    if (!boxPosition || !boxSize) continue;
        const influence = probeInfluence(probe, boxPosition, boxSize) || defaultProbeInfluence(boxPosition, boxSize);
        setEnvironmentProbe(
            id,
            {
                ...probe,
                shape: "box",
                boxPosition,
                boxSize,
                capturePosition,
                ...influence,
            },
            id,
            { history: false }
        );
  }
  // Only from an undo snapshot: a *manifest* carries its lighting in the
  // `environment` / `editorEnvironment` blocks, which loadLayout() applies
  // through applyEnvironment() instead. Only the runtime rig is on the stack -
  // the editor's is a per-browser view preference, like the camera - but the
  // loop still reads by name so an older snapshot carrying both is harmless.
  if (data.lightSets) {
    for (const k of ["editor", "runtime"]) {
      const set = data.lightSets[k];
      if (!set) continue;
      if (Number.isFinite(set.strength)) state.lightSets[k].strength = set.strength;
      if (Number.isFinite(set.exposure)) state.lightSets[k].exposure = set.exposure;
      if (set.toneMapping) state.lightSets[k].toneMapping = String(set.toneMapping);
    }
    const live = activeLightSet();
    setEnvIntensity(live.strength);
    setExposure(live.exposure);
    setToneMapping(live.toneMapping ?? TONE_MAPPING_DEFAULT);
    emit("environment");            // the sliders and their readouts follow
  }
  // Alongside the rig, and separate from it: these reach the materials rather
  // than the scene, so the setters announce them as "reflection" and not as
  // "environment". Absent from an older snapshot means "leave them alone".
  if (typeof data.runtimeSpecularAA === "boolean") setRuntimeSpecularAA(data.runtimeSpecularAA);
  if (Number.isFinite(data.runtimeRoughnessFactor)) setRuntimeRoughnessFactor(data.runtimeRoughnessFactor);
  for (const inst of data.instances || []) {
    if (!getModule(inst.module)) {
      console.warn("skipping unknown module", inst.module);
      continue;
    }
    await placeAt(inst.module, Vector3.FromArray(inst.position), {
            id: inst.id,
            rotation: inst.rotation,
            scale: inst.scale,
            chunk: inst.chunk,
            name: inst.name,
            group: inst.group,
            compound: inst.compound,
      // A layout is ship data whatever mode the editor happens to be in.
            stage: false,
            silent: true,
            noLights: true,
    });
  }
  applyVisibility();
  hooks.deserializeMarkers(data.markers || []);
  hooks.deserializeColliders(data.colliders || []);
  // After the instances loop: a light is attached to a placement and is dropped
  // when its owner is missing, so restoring it any earlier would lose every one.
  hooks.deserializeLights(data.lights || []);
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
  if (Array.isArray(data.fluidSim)) {
    for (const name of data.fluidSim) {
      if (typeof name !== "string" || !name || /\.json$/i.test(name)) {
        throw new Error(`fluidSim name "${String(name)}" must omit the .json extension`);
      }
    }
    state.fluidSim = [...data.fluidSim];
  }
    function legacyBehaviorName(name) {
        return name === "anyLiquefaction" ? "liquefaction" : name;
    }

    function legacyBehaviorPresets(definitions) {
        const presets = {};
        for (const name of ["stdLiquefaction", "explosiveLiquefaction"]) {
            const body = definitions?.[name];
            if (body && typeof body === "object" && !Array.isArray(body)) {
                const preset = JSON.parse(JSON.stringify(body));
                delete preset.liquefiable;
                presets[name] = { base: "liquefaction", ...preset };
            }
        }
        return presets;
    }

    function legacyDirectBehaviorParams(definitions, name) {
        if (!definitions || name === "stdLiquefaction" || name === "explosiveLiquefaction") return {};
        const body = definitions[name];
        if (!body || typeof body !== "object" || Array.isArray(body)) return {};
        const params = JSON.parse(JSON.stringify(body));
        if (legacyBehaviorName(name) === "liquefaction") delete params.liquefiable;
        return params;
    }

    const legacyDefinitions = data.behaviorPresets ? null : data.behaviors || {};
    const presetData = data.behaviorPresets || legacyBehaviorPresets(legacyDefinitions);
    for (const [name, body] of Object.entries(presetData)) {
    const key = String(name || "").trim();
    if (!key || !body || typeof body !== "object" || Array.isArray(body)) continue;
        const preset = JSON.parse(JSON.stringify(body));
        const base = String(preset.base || "").trim();
        if (!base || !behaviorMetadata(behaviorCatalog, base)) {
            throw new Error(`behavior preset "${key}" references unknown base behavior "${base}"`);
        }
        if (behaviorMetadata(behaviorCatalog, key)) {
            throw new Error(`behavior preset "${key}" conflicts with a base behavior of the same name`);
        }
        delete preset.base;
        if (base === "liquefaction") delete preset.liquefiable;
        assertValidBehaviorConfig(base, preset, true, `behavior preset "${key}"`);
        state.behaviors.set(key, { base, ...preset });
  }
  for (const [node, list] of Object.entries(data.entities || {})) {
    const key = String(node || "").trim();
    if (!key) continue;
    if (!Array.isArray(list?.behaviors)) {
      throw new Error(`entity "${key}" must contain a behaviors array`);
    }
    const kept = [];
    for (const b of list.behaviors) {
      if (!b || typeof b !== "object" || Array.isArray(b)) {
        throw new Error(`entity "${key}" contains an invalid behavior assignment`);
      }
            const sourceName = String(b.name || "").trim();
            if (!sourceName) {
        throw new Error(`entity "${key}" contains a behavior assignment without a name`);
      }
            const name = legacyBehaviorName(sourceName);
            if (!availableBehaviorNames().includes(name)) {
                throw new Error(`entity "${key}" references unknown behavior "${name}"`);
      }
            const params = {
                ...legacyDirectBehaviorParams(legacyDefinitions, sourceName),
                ...readBehaviorParams(b),
            };
            if (behaviorBaseName(name) === "liquefaction") delete params.liquefiable;
            assertValidBehaviorConfig(
                behaviorBaseName(name),
                {
                    ...(getBehaviorDef(name) ?? {}),
        ...behaviorParams(params),
                },
                false,
                `entity "${key}" behavior "${name}"`
            );
            kept.push({ name, linked: [], ...params });
    }

    if (kept.length) state.entities.set(key, kept);
  }
  emit("chunks");
  emit("placements");
  emit("markers");
  emit("behaviors");
  emit("environment-probes");
  emit("selection");
}

let restoring = false;

/**
 * The manifest and the undo snapshot are both glTF space; the editor is not.
 * One helper for both directions - the transform is its own inverse - so the
 * two can never drift apart.
 */
function flipX(v) {
    return [-Number(v[0]), Number(v[1]), Number(v[2])];
}

/**
 * The parameters of an applied behaviour, on the way IN from the wire.
 *
 * `direction` is a vector, so it is mirrored into editor space.
 *
 * The inverse of writeBehaviorParams - the pair has to stay symmetric, so keep
 * them next to each other.
 */
function readBehaviorParams(b) {
  const out = {};
  for (const [key, value] of Object.entries(b)) {
    if (key === "name" || value === undefined) continue;
    out[key] = JSON.parse(JSON.stringify(value));
  }
  if (isVector3(out.direction)) out.direction = flipX(out.direction);
  return out;
}

/**
 * The parameters of an applied behaviour, on the way OUT to the wire.
 *
 * The inverse of readBehaviorParams: same parameters, same two special cases,
 * plus one asymmetry - an empty `linked` is dropped rather than written, since
 * every behaviour carries the key in state and only a few ever fill it.
 */
export function writeBehaviorParams(b) {
  const out = {};
  for (const [key, value] of Object.entries(b)) {
    if (key === "name" || value === undefined) continue;
    out[key] = JSON.parse(JSON.stringify(value));
  }
  if (Array.isArray(out.linked) && out.linked.length) out.linked = [...out.linked];
  else delete out.linked;
  if (isVector3(out.direction)) out.direction = flipX(out.direction);
  return out;
}

/** True while an undo/redo/load is rebuilding the scene - for tests. */
export function isRestoring() {
    return restoring;
}

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

export function isBusy() {
    return busyDepth > 0;
}
export function busyLabel() {
    return busyMessage;
}

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
 * Say what the editor is busy WITH, part-way through being busy.
 *
 * A capture takes a room at a time and each one is seconds long, so a label
 * fixed when the lock was taken would sit there saying nothing for a minute.
 * Ignored when nothing is locked: a message with no overlay under it would be
 * a claim about a state the editor is not in.
 */
export function setBusyMessage(message) {
  if (!busyDepth || !message || message === busyMessage) return;
  busyMessage = message;
  emit("busy");
}

/**
 * Fail a step that never finishes, instead of holding the editor for ever.
 *
 * Every long await in the probe capture path is a promise that Babylon settles
 * from inside the render loop - a shader that becomes ready, a read-back, a
 * prefilter pass - and each one of them can be waited on for ever if the thing
 * it is waiting for cannot happen. A material whose effect fails to link never
 * reports ready; a tab moved to the background stops producing frames; a fetch
 * to a server that has gone away hangs on the socket. None of those throw.
 *
 * That matters far beyond the capture itself, because the capture holds the
 * busy lock: a stall does not merely leave a probe untaken, it leaves the whole
 * editor inert with an overlay over it and no way back except a reload. Turning
 * every such await into one that CAN fail is what lets `whileBusy` unwind and
 * hand the tool back with an error that names the step.
 *
 * `Promise.race` leaves the loser running - there is no cancelling a Babylon
 * readiness poll - which is fine: it resolves into nothing once the caller has
 * given up on it.
 */
export function withDeadline(promise, ms, what) {
  let timer = null;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${what} did not finish within ${Math.round(ms / 1000)}s`)), ms);
    }),
  ]).finally(() => clearTimeout(timer));
}

// --------------------------------------------------------- viewport suspend
//
// There is deliberately no such thing. Stopping the editor's frame during a
// probe capture looks obviously right - the capture drives the scene with its
// own camera and its own image processing, and a frame drawn in the middle of
// that is a picture of no state the editor is meant to have - and it does not
// work. Babylon compiles a material's shader as part of rendering, so with the
// loop stopped the effects a capture dirties (the light set, the image
// processing define) never rebuild, and `whenReadyAsync` waits for a readiness
// that can no longer arrive. Measured: the capture hung indefinitely.
//
// The busy overlay is what keeps those frames private. It covers the viewport
// and makes every panel inert, so the editor may as well keep drawing.

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
    while (stack.length > HISTORY_MAX_ENTRIES || (stack.length > 1 && chars > HISTORY_MAX_CHARS)) {
    chars -= stack[0].length;
    stack.shift();
  }
}

/** The history limits - for tests and diagnostics. */
export function historyLimits() {
  return { entries: HISTORY_MAX_ENTRIES, chars: HISTORY_MAX_CHARS };
}

/**
 * One pair of stacks per editing mode, and how each mode snapshots itself.
 *
 * A mode edits a different world, and a snapshot of one is meaningless in
 * another: both benches are deliberately absent from serialize() - they must
 * never reach the ship - so a ship snapshot taken on a bench restores as "no
 * bench at all", which is precisely how Ctrl+Z used to wipe the collision
 * staging area. A stack apiece also means a bench edit does not rebuild 116
 * placements to undo one box.
 *
 * A registry rather than a branch: this began as `if (state.collisionMode)`
 * repeated in pushUndo, undo, redo and historyDepth, and a second bench would
 * have doubled every one of those into a three-way ladder. Now a new mode is
 * one entry here and nothing else.
 */
const histories = {
  ship: {
        undo: [],
        redo: [],
    snapshot: () => serialize(),
    restore: (data) => deserialize(data),
  },
  collision: {
        undo: [],
        redo: [],
    snapshot: () => hooks.serializeStage(),
    restore: (data) => hooks.restoreStage(data),
  },
  compound: {
        undo: [],
        redo: [],
    snapshot: () => hooks.serializeBench(),
    restore: (data) => hooks.restoreBench(data),
  },
};

/** The stacks the active mode writes to. */
function history() {
  return histories[state.mode] || histories.ship;
}

export function pushUndo() {
  pushUndoFor(state.mode);
}

/**
 * Push a snapshot onto a named mode's stack, whatever mode is active.
 *
 * Nearly every edit belongs to the world you are looking at, which is what
 * `pushUndo` assumes. Pushing an update from the compound bench out to the
 * copies in the ship is the exception: the change lands in the ship, so it has
 * to be undoable *there*. Recorded on the bench's stack instead it would be
 * lost the moment the bench closed - the bench's history does not outlive it -
 * and a later ship undo would revert the sync as a side effect of undoing
 * something else entirely.
 */
export function pushUndoFor(mode) {
  if (restoring) return;
  const h = histories[mode] || histories.ship;
  h.undo.push(JSON.stringify(h.snapshot()));
  trimHistory(h.undo);
  h.redo.length = 0;
}

/** Depth of the active mode's history stacks - for tests and diagnostics. */
export function historyDepth() {
  const h = history();
  return { undo: h.undo.length, redo: h.redo.length };
}

export async function undo() {
    await stepHistory("undo", "redo");
}
export async function redo() {
    await stepHistory("redo", "undo");
}

/**
 * Move one step between the active mode's two stacks.
 *
 * Undo and redo are the same operation with the stacks swapped, and writing it
 * once is what keeps them symmetrical: the pair used to be two near-identical
 * bodies per mode, and the trimHistory call on the receiving stack was easy to
 * leave out of one of the four.
 */
async function stepHistory(from, to) {
  const h = history();
  if (!h[from].length) return;
  h[to].push(JSON.stringify(h.snapshot()));
  trimHistory(h[to]);
  await h.restore(JSON.parse(h[from].pop()));
}

/**
 * Start a mode's history clean, so it cannot reach past its own opening.
 *
 * Called on the way into a bench and again on the way out: a bench's history
 * must not outlive the bench, or re-opening it would offer to undo edits to
 * contents that are no longer there.
 */
export function resetModeHistory(mode = state.mode) {
  const h = histories[mode];
  if (!h) return;
  h.undo.length = 0;
  h.redo.length = 0;
}

// ---------------------------------------------------------------- helpers

export function eulerOf(node) {
  const q = node.rotationQuaternion || Quaternion.FromEulerVector(node.rotation);
  const e = q.toEulerAngles();
  return [(e.x * 180) / Math.PI, ( e.y * 180) / Math.PI, ( e.z * 180) / Math.PI];
}

export function setEuler(node, deg) {
    node.rotationQuaternion = Quaternion.FromEulerAngles((deg[0] * Math.PI) / 180, (deg[1] * Math.PI) / 180, (deg[2] * Math.PI) / 180);
}

function round3(a) {
    return a.map((v) => Math.round(v * 1000) / 1000);
}

export function worldBounds(node) {
    let min = null,
        max = null;
  for (const m of node.getChildMeshes()) {
    // A Runtime-view stand-in sits exactly on the mesh it stands in for, so it
    // cannot widen these bounds - but it can be the only mesh left when its
    // original is invisible, and measuring a copy is measuring nothing new.
    if (isRuntimeStandIn(m)) continue;
    // A light's gizmo rides the placement it lights. Measured with it, a wall
    // panel would report the lamp's size as its own and "Drop to plane" would
    // lift the panel off the floor by however far the lamp hangs below it.
        if (isGizmoMesh(m) && m.metadata.lightRoot !== node && m.metadata.environmentProbeRoot !== node) continue;
    m.computeWorldMatrix(true);
    const bb = m.getBoundingInfo().boundingBox;
    min = min ? Vector3.Minimize(min, bb.minimumWorld) : bb.minimumWorld.clone();
    max = max ? Vector3.Maximize(max, bb.maximumWorld) : bb.maximumWorld.clone();
  }
  return min ? { min, max } : null;
}

/**
 * How far a piece has to stand clear of its chunk before that counts as a
 * mistake.
 *
 * Wide on purpose. The question is not "does this touch?" - decals, trim and
 * door frames are all mounted a few centimetres proud of the surface they
 * belong to, and on this ship the widest such gap measured 25 cm. The question
 * is "was this left in the wrong room?", and a piece in the wrong room is a
 * kit tile away at the very least: the kit's grid is 4 m. Half a metre clears
 * every mounting offset and is still eight times inside the smallest real
 * mistake.
 */
const STRAY_CHUNK_SLACK = 0.5;

function boxesTouch(a, b, slack = STRAY_CHUNK_SLACK) {
    return ["x", "y", "z"].every((k) => Math.min(a.max[k], b.max[k]) - Math.max(a.min[k], b.min[k]) >= -slack);
}

function growBounds(into, bounds) {
  if (!into) return { min: bounds.min.clone(), max: bounds.max.clone() };
  return {
    min: Vector3.Minimize(into.min, bounds.min),
    max: Vector3.Maximize(into.max, bounds.max),
  };
}

/**
 * A chunk's members grouped by what touches what, largest group first.
 *
 * Breadth-first over the touch relation rather than a straight pass, because a
 * corridor is one room even though its two ends are nowhere near each other:
 * what makes it one thing is the unbroken run of tiles between them.
 */
function touchingGroups(list) {
  const taken = new Array(list.length).fill(false);
  const groups = [];
  for (let i = 0; i < list.length; i++) {
    if (taken[i]) continue;
    taken[i] = true;
    const group = [list[i]];
    for (let head = 0; head < group.length; head++) {
      for (let j = 0; j < list.length; j++) {
        if (taken[j] || !boxesTouch(group[head].bounds, list[j].bounds)) continue;
        taken[j] = true;
        group.push(list[j]);
      }
    }
    groups.push(group);
  }
  return groups.sort((a, b) => b.length - a.length);
}

/**
 * Elements that look like they were left in the wrong chunk.
 *
 * A chunk has no authored volume - its box is the union of whatever is assigned
 * to it - so asking "is this element inside its chunk?" is true by construction
 * and worth nothing. What the chunk does have is a shape: the pieces that make
 * up a room are stuck to one another, tile against tile. So the question worth
 * asking is which of a chunk's members hang together and which hang off on
 * their own, and the answer is the largest group of them that touch. Anything
 * outside it is only actionable when some *other* chunk's volume reaches the
 * piece: that names both the mistake and the chunk it was meant for. An entity
 * standing alone is legal authored content - a trigger, a simulation trap, a
 * pickup - and connectivity alone cannot call it misplaced.
 *
 * Grouping rather than measuring each piece against the rest of its chunk in
 * turn, which is the obvious way and is wrong: two pieces left behind in the
 * same chunk hide each other, since the "rest" that each is measured against
 * contains the other and stretches over the ground between them.
 *
 * Deliberately not an error. The geometry makes the intended chunk very likely,
 * but the assignment remains an authoring decision.
 *
 * @returns [{ id, name, chunk, host }] - `host` is the other chunk whose
 *   settled volume reaches the element.
 */
export function strayChunkMembers() {
  const members = new Map();
  for (const placement of shipPlacements()) {
    const bounds = worldBounds(placement.node);
    if (!bounds) continue;
    if (!members.has(placement.chunk)) members.set(placement.chunk, []);
    members.get(placement.chunk).push({ placement, bounds });
  }
  const strays = [];
  const settled = new Map();
  for (const [chunk, list] of members) {
    const groups = touchingGroups(list);
    // Nothing to say about a chunk that has only just been started, or one
    // split evenly: with no group bigger than the rest there is no telling
    // which of them is the room and which was left behind.
    if (groups.length > 1 && groups[1].length === groups[0].length) {
      settled.set(chunk, list);
      continue;
    }
    settled.set(chunk, groups[0]);
    for (let g = 1; g < groups.length; g++) {
      for (const entry of groups[g]) strays.push({ chunk, entry });
    }
  }
  if (!strays.length) return [];
  // Volumes for naming the chunk a stray was meant for, built from what stayed
  // put. Including the strays would let one piece dropped across the ship
  // stretch its chunk over every other room and be reported as belonging to
  // all of them.
  const volumes = new Map();
  for (const [chunk, list] of settled) {
    let box = null;
    for (const entry of list) box = growBounds(box, entry.bounds);
    if (box) volumes.set(chunk, box);
  }
  return strays.flatMap(({ chunk, entry }) => {
    const host = [...volumes].find(([id, box]) => id !== chunk && boxesTouch(entry.bounds, box));
    if (!host) return [];
    return [{
      id: entry.placement.id,
      name: entry.placement.name || entry.placement.id,
      chunk,
      host: host[0],
    }];
  });
}
