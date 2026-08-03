// Palette thumbnails.
//
// 300-odd modules is too many to render up front, so a thumbnail is only
// produced when its tile scrolls into view, and the result is pushed to the
// server's disk cache so later sessions load it straight from /api/thumb.

import { materialKey, applyKitTransparency } from "./kit.js";

const {
  Engine, Scene, ArcRotateCamera, HemisphericLight, DirectionalLight,
  Vector3, Color3, Color4, SceneLoader,
} = BABYLON;

const SIZE = 192;
// Canonical framing, re-applied after every setTarget (see renderThumb).
const THUMB_ALPHA = Math.PI * 0.3;
const THUMB_BETA = 1.15;
// Frames in a turntable sprite sheet - one full 360 deg turn.
export const TURN_FRAMES = 12;

let engine = null;
let scene = null;
let camera = null;
let cached = new Set();
const queue = [];
const inflight = new Map();
let running = false;

export async function initThumbs() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = SIZE;
  engine = new Engine(canvas, true, { preserveDrawingBuffer: true }, false);
  scene = new Scene(engine);
  scene.clearColor = new Color4(0.06, 0.07, 0.086, 1);
  scene.autoClear = true;

  // +180 deg from the obvious angle: the kit models face away from -Z, so the
  // default view showed every module from behind.
  camera = new ArcRotateCamera("t", THUMB_ALPHA, THUMB_BETA, 6, Vector3.Zero(), scene);
  camera.minZ = 0.01;

  const hemi = new HemisphericLight("th", new Vector3(0.2, 1, 0.1), scene);
  hemi.intensity = 1.0;
  hemi.groundColor = new Color3(0.22, 0.24, 0.29);
  const dir = new DirectionalLight("td", new Vector3(-0.6, -1, 0.7), scene);
  dir.intensity = 1.6;

  buildBackdrop(scene);

  // Same highlight-shoulder trap as the main viewport: the pale panels
  // (MI_Trim_03) sit near the top of the KHR PBR Neutral curve at high
  // exposure and flatten into a featureless wash - which is exactly why these
  // tiles used to look like uniform blocks of colour. Keep the exposure low
  // enough that they stay on the straight part of the curve.
  scene.imageProcessingConfiguration.toneMappingEnabled = true;
  scene.imageProcessingConfiguration.toneMappingType =
    BABYLON.ImageProcessingConfiguration.TONEMAPPING_KHR_PBR_NEUTRAL;
  scene.imageProcessingConfiguration.exposure = 0.5;
  try {
    scene.environmentTexture =
      new BABYLON.HDRCubeTexture("/env/bank_vault_2k.hdr", scene, 64, false, true, false, true);
    scene.environmentIntensity = 1.6;
  } catch { /* thumbnails degrade to silhouettes */ }

  try {
    const r = await fetch("/api/thumbs");
    cached = new Set((await r.json()).cached);
  } catch { /* first run */ }

  try {
    const r = await fetch("/api/turns");
    turnCached = new Set((await r.json()).cached);
  } catch { /* first run */ }
}

/**
 * Bumped whenever a change alters what a thumbnail *looks like*.
 *
 * The cache is on the server's disk and outlives any reload, so without this a
 * rendering fix is invisible until someone deletes the folder by hand — and
 * the whole point of the fix that added this was a tile disagreeing with the
 * ship, which is exactly the sort of thing you would then still be staring at.
 */
const THUMB_VERSION = 5;

const keyOf = (id) => `v${THUMB_VERSION}_${id.replace(/[^A-Za-z0-9_.-]/g, "_")}`;

/**
 * Checkerboard backdrop. Plenty of kit pieces have alpha-cut or genuinely
 * transparent parts (grilles, glass, cables); against a flat fill those read as
 * solid background and you cannot tell what is actually there. A checker makes
 * every see-through region obvious at a glance.
 */
function buildBackdrop(scene) {
  const tex = new BABYLON.DynamicTexture("thumb_bg",
    { width: SIZE, height: SIZE }, scene, false);
  const ctx = tex.getContext();
  const cells = 8;
  const s = SIZE / cells;
  for (let y = 0; y < cells; y++) {
    for (let x = 0; x < cells; x++) {
      ctx.fillStyle = (x + y) % 2 ? "#39414f" : "#272d38";
      ctx.fillRect(x * s, y * s, s, s);
    }
  }
  tex.update(false);
  const layer = new BABYLON.Layer("thumb_bg", null, scene, true);
  layer.texture = tex;
}

/** Framing actually used by the last thumbnail render - exposed for tests. */
export function lastThumbView() {
  return camera ? { alpha: camera.alpha, beta: camera.beta, target: camera.target.asArray() } : null;
}
export const THUMB_VIEW = { alpha: THUMB_ALPHA, beta: THUMB_BETA };

// The thumbnail scene lives on its own engine, so it needs its own material
// cache. Without it every module re-uploads the same 2-4 MB atlas. Keyed the
// same way as the ship's, transparency included - two caches filled in
// different orders is exactly how a window came out see-through on its tile
// and solid in the ship.
const thumbMaterials = new Map();

function shareMaterials(meshes) {
  for (const mesh of meshes) {
    const mat = mesh.material;
    if (!mat) continue;
    applyKitTransparency(mat);          // the same authored values the ship gets
    const key = materialKey(mat);
    const shared = thumbMaterials.get(key);
    if (shared && shared !== mat) {
      mesh.material = shared;
      mat.dispose(false, true);
    } else if (!shared) {
      mat.backFaceCulling = false;      // thumbnails are viewed from one angle
      mat.twoSidedLighting = true;
      thumbMaterials.set(key, mat);
    }
  }
}

export function cachedUrl(moduleId) {
  return cached.has(keyOf(moduleId)) ? `/api/thumb/${keyOf(moduleId)}` : null;
}

// ------------------------------------------------------------- turntables
//
// Turntables are far heavier than stills (12 renders, ~12x the bytes), so they
// are never produced up front: one is built the first time a tile is actually
// hovered, and cached on the server from then on.

let turnCached = new Set();
const turnInflight = new Map();

export function cachedTurnUrl(moduleId) {
  return turnCached.has(keyOf(moduleId)) ? `/api/turn/${keyOf(moduleId)}` : null;
}

/**
 * Drop a module from the local "already have it" sets so the next request
 * re-renders it. Used by the tests to exercise the render path itself rather
 * than the cache; the server copies are simply overwritten on the way out.
 */
export function forgetCached(moduleId) {
  cached.delete(keyOf(moduleId));
  turnCached.delete(keyOf(moduleId));
}

/** Resolve to a sprite-sheet URL, rendering it first if this is the first ask. */
export function requestTurntable(mod) {
  const ready = cachedTurnUrl(mod.id);
  if (ready) return Promise.resolve(ready);
  if (turnInflight.has(mod.id)) return turnInflight.get(mod.id);
  const job = (async () => {
    const dataUrl = await renderTurntable(mod);
    turnCached.add(keyOf(mod.id));
    const blob = await (await fetch(dataUrl)).blob();
    fetch(`/api/turn/${keyOf(mod.id)}`, {
      method: "PUT", headers: { "Content-Type": "image/png" }, body: blob,
    }).catch(() => { /* cache is best-effort */ });
    return dataUrl;
  })().finally(() => turnInflight.delete(mod.id));

  turnInflight.set(mod.id, job);
  return job;
}

export function request(mod, imgEl) {
  const url = cachedUrl(mod.id);
  if (url) { imgEl.src = url; return; }
  if (inflight.has(mod.id)) { inflight.get(mod.id).push(imgEl); return; }
  inflight.set(mod.id, [imgEl]);
  queue.push(mod);
  pump();
}

async function pump() {
  if (running) return;
  running = true;
  while (queue.length) {
    const mod = queue.shift();
    try {
      const dataUrl = await renderThumb(mod);
      for (const el of inflight.get(mod.id) || []) el.src = dataUrl;
      cached.add(keyOf(mod.id));
      fetch(`/api/thumb/${keyOf(mod.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "image/png" },
        body: await (await fetch(dataUrl)).blob(),
      }).catch(() => { /* cache is best-effort */ });
    } catch (e) {
      console.warn("thumbnail failed", mod.id, e);
    }
    inflight.delete(mod.id);
  }
  running = false;
}

async function renderThumb(mod) {
  return withModule(mod, () => {
    scene.render();
    return engine.getRenderingCanvas().toDataURL("image/png");
  });
}

/**
 * A full 360 deg turn as a single horizontal sprite sheet.
 *
 * A still tile is a poor description of a modular part - you cannot tell a
 * corner from a straight, or which side a cable run is on. Spinning it on
 * hover answers both instantly. It is one sheet rather than 12 files so a
 * preview is a single request and cannot tear halfway through a turn.
 */
async function renderTurntable(mod) {
  return withModule(mod, () => {
    const sheet = document.createElement("canvas");
    sheet.width = SIZE * TURN_FRAMES;
    sheet.height = SIZE;
    const ctx = sheet.getContext("2d");
    const src = engine.getRenderingCanvas();
    for (let i = 0; i < TURN_FRAMES; i++) {
      camera.alpha = THUMB_ALPHA + (i * 2 * Math.PI) / TURN_FRAMES;
      scene.render();
      ctx.drawImage(src, i * SIZE, 0, SIZE, SIZE);
    }
    camera.alpha = THUMB_ALPHA;
    return sheet.toDataURL("image/png");
  });
}

/**
 * Load a module into the thumbnail scene, frame it, wait for its shaders, hand
 * it to `fn`, then drop the geometry again. Shared by the still and the
 * turntable so both always agree on framing.
 *
 * Serialised, because stills and turntables share one scene and one camera and
 * are driven by two independent triggers - scrolling and hovering. A turntable
 * holds its module in the scene across 12 renders, and any still that happened
 * to be rendered in that window came out with both modules in the picture.
 */
async function withModule(mod, fn) {
  return exclusive(async () => {
    active++;
    peakActive = Math.max(peakActive, active);
    try {
      return await loadFrameAndRun(mod, fn);
    } finally {
      active--;
    }
  });
}

let sceneLock = Promise.resolve();
let active = 0;
let peakActive = 0;

/** Queue `job` behind every other use of the thumbnail scene. */
function exclusive(job) {
  // `.then(job, job)` so one failed render cannot wedge the queue forever.
  const run = sceneLock.then(job, job);
  sceneLock = run.then(() => {}, () => {});
  return run;
}

/** How many renders are in flight, and the worst ever seen - for tests. */
export function thumbConcurrency() { return { active, peak: peakActive }; }

async function loadFrameAndRun(mod, fn) {
  // Belt and braces: anything left over from an earlier render would be
  // photographed along with this module. Safe here because the lock guarantees
  // nothing else is using the scene.
  for (const m of [...scene.meshes]) m.dispose(false, false);

  const dir = mod.url.slice(0, mod.url.lastIndexOf("/") + 1);
  const file = mod.url.slice(mod.url.lastIndexOf("/") + 1);
  const container = await SceneLoader.LoadAssetContainerAsync(dir, file, scene);

  const meshes = container.meshes.filter((m) => m.getTotalVertices() > 0);
  shareMaterials(meshes);
  container.addAllToScene();

  let min = null, max = null;
  for (const m of meshes) {
    m.computeWorldMatrix(true);
    const bb = m.getBoundingInfo().boundingBox;
    min = min ? Vector3.Minimize(min, bb.minimumWorld) : bb.minimumWorld.clone();
    max = max ? Vector3.Maximize(max, bb.maximumWorld) : bb.maximumWorld.clone();
  }
  if (min) {
    const centre = min.add(max).scale(0.5);
    const diag = max.subtract(min).length();
    camera.setTarget(centre);
    // setTarget re-derives alpha/beta from wherever the camera happens to be,
    // so a module modelled high above its origin (TopCables_Corner_* sits at
    // y = 4) ends up viewed from underneath, and every tile gets a different
    // angle. Put the canonical framing back.
    camera.alpha = THUMB_ALPHA;
    camera.beta = THUMB_BETA;
    // A turntable sees the module from every side, so frame it on the widest
    // horizontal extent rather than the one angle a still happens to use -
    // otherwise a long piece swings out of frame halfway through the turn.
    const half = Math.hypot(max.x - min.x, max.z - min.z) * 0.5;
    camera.radius = Math.max(0.6, Math.max(diag, half * 2.1) * 1.15);
  }

  // Shaders only compile while rendering, so drive the scene until it reports
  // ready - waiting without rendering would spin forever and yield a blank tile.
  await new Promise((resolve) => {
    let tries = 0;
    const tick = () => {
      scene.render();
      if (scene.isReady(true) || ++tries > 90) return resolve();
      requestAnimationFrame(tick);
    };
    tick();
  });

  try {
    return fn();
  } finally {
    // Dispose the geometry only: the materials and textures are shared.
    for (const m of container.meshes) m.dispose(false, false);
    for (const t of container.transformNodes) t.dispose();
    container.meshes.length = 0;
    container.transformNodes.length = 0;
    container.materials.length = 0;
    container.textures.length = 0;
  }
}
