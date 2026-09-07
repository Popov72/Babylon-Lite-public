// Zero-dependency static server + kit API for the SciFi ship layout tool.
//
//   node server.mjs            -> http://localhost:5173
//
// It does five things:
//   1. serves public/ (the app itself)
//   2. serves the Quaternius kits, from a local BabylonAssets checkout or the CDN
//   3. exposes the module catalogue and a thumbnail cache
//   4. writes the layout manifest and the exported ship.glb to disk
//   5. holds the environment probe cubemaps the editor captures, and the index
//      that says which of them are still stale
//   6. publishes the export folder into the demo, by running sync-ship.ts

import http from "node:http";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONFIG = JSON.parse(fs.readFileSync(path.join(HERE, "config.json"), "utf8"));

// Every path in config.json is resolved against the *config file*, not the
// working directory, so the relative defaults keep pointing at this copy's own
// folders wherever the editor is checked out or copied to. An absolute path
// still wins, for anyone pointing the tool at a kit held somewhere else.
//
// ------------------------------------------------------------------- the kits
//
// The kits are NOT bundled with the editor any more. They live in the
// BabylonAssets repository, whose root maps 1:1 onto https://assets.babylonjs.com,
// and the editor reads them from one of two places:
//
//   "local"  - a checkout of BabylonAssets on this machine. Mounted read-only
//              at /assets/, so the app is same-origin with its own assets and
//              nobody has to start a second web server or think about CORS.
//   "online" - https://assets.babylonjs.com directly. The catalogue still has
//              to be built from a local listing, because a static CDN cannot be
//              asked what files it holds - so `localDir` must be present even in
//              online mode. What changes is only where the *module URLs* point.
//
// SHIP_KITS_SOURCE / SHIP_ASSETS_DIR override both, so a run can be flipped
// without editing the config.
const KITS = CONFIG.kits || {};
const KITS_SOURCE = (process.env.SHIP_KITS_SOURCE || KITS.source || "local").toLowerCase();
const ASSETS_DIR = path.resolve(HERE, process.env.SHIP_ASSETS_DIR || KITS.localDir || "../../../../../../../../BabylonAssets");
// Trailing slash normalised once: every URL below is built by concatenation.
const ONLINE_BASE = String(KITS.onlineBase || "https://assets.babylonjs.com/").replace(/\/*$/, "/");
// Where the kits sit inside the assets tree. One string, so local and online
// cannot drift apart.
const KITS_PREFIX = String(KITS.prefix || "kits").replace(/^\/+|\/+$/g, "");
const KITS_DIR = path.join(ASSETS_DIR, KITS_PREFIX);
// Production writes collision beside the kit models. Tests override only this
// root so they can keep using the real catalogue without touching BabylonAssets.
const COLLISION_KITS_DIR = process.env.SHIP_COLLISION_KITS_DIR
  ? path.resolve(HERE, process.env.SHIP_COLLISION_KITS_DIR)
  : KITS_DIR;
// Which kit the palette opens on, first installed name wins. The list itself
// is alphabetical; this only picks the one to start on.
const KIT_FOLDERS = Array.isArray(KITS.folders) ? KITS.folders : [];
const EXPORT_DIR = path.resolve(HERE,
  // Overridable so the test suite can point at a scratch directory instead of
  // the real export folder - a test run must never touch a real ship.
  process.env.SHIP_EXPORT_DIR || CONFIG.exportDir);
// The env folder always travels with the real project, never with a scratch
// export directory used by tests.
const ENV_DIR = path.resolve(HERE, CONFIG.envDir
  || path.join(path.resolve(HERE, CONFIG.exportDir), "..", "env"));
const PORT = Number(process.env.SHIP_PORT || CONFIG.port);
const PUBLIC_DIR = path.join(HERE, "public");
const FLUID_SIM_DIR = path.resolve(HERE, "../../../../../../public/aquanova/fluidSim");
// The game's sound folder. Listed rather than configured for the same reason
// the fluid-sim folder is: it is the game's own data, and the editor's job is
// to report what is in it, not to keep a second copy of the answer.
const SOUNDS_DIR = path.resolve(HERE, "../../../../../../public/aquanova/sounds");
const THUMB_DIR = path.join(HERE, "cache", "thumbs");
const TURN_DIR = path.join(HERE, "cache", "turntable");
const LAYOUT_DIR = path.join(HERE, "layouts");

const MANIFEST = path.join(EXPORT_DIR, "ship_manifest.json");
const LEGACY_COLLISION = path.join(EXPORT_DIR, "ship_collision.json");
// Compound definitions - the recipes the compound editor saves. Beside the
// manifest because they are project authoring data, not a property of any one
// kit module. Collision differs: it belongs beside the kit that owns it.
const COMPOUNDS = path.join(EXPORT_DIR, "ship_compounds.json");
const AUTOSAVE = path.join(EXPORT_DIR, "ship_autosave.json");
const GLB = path.join(EXPORT_DIR, "ship.glb");

// ------------------------------------------------------------------- the demo
//
// The editor writes into `export/`; the game reads from `lab/public/aquanova/`.
// sync-ship.ts is what carries one to the other - copying the manifest, the
// collision hulls and the captured probes, and compressing the glb - so the
// **Start demo** button is that script followed by opening the page.
//
// Both live in config.json rather than in this file: which port the lab's dev
// server is on and where the script sits are facts about this checkout, and a
// checkout that moves either one should not need a code change.
//
// SHIP_SYNC_SCRIPT / SHIP_DEMO_URL override them, for the same reason
// SHIP_EXPORT_DIR exists: the real script writes into the real game folder, and
// a test run must be able to exercise this route without doing that.
const DEMO = CONFIG.demo || {};
const DEMO_URL = String(process.env.SHIP_DEMO_URL || DEMO.url || "");
const SYNC_SCRIPT_REL = process.env.SHIP_SYNC_SCRIPT || DEMO.syncScript;
const SYNC_SCRIPT = SYNC_SCRIPT_REL ? path.resolve(HERE, SYNC_SCRIPT_REL) : "";
/** How long a publish may run before it is killed, in ms. */
const SYNC_TIMEOUT_MS = 20 * 60 * 1000;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".gltf": "model/gltf+json",
  ".glb": "model/gltf-binary",
  ".bin": "application/octet-stream",
  // No registered media type exists for FBX, and the loader reads it as an
  // ArrayBuffer, so the honest answer is "bytes".
  ".fbx": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".hdr": "application/octet-stream",
  ".env": "application/octet-stream",
  ".svg": "image/svg+xml",
};

function send(res, code, body, type = "text/plain; charset=utf-8", extra = {}) {
  res.writeHead(code, { "Content-Type": type, "Cache-Control": "no-cache", ...extra });
  res.end(body);
}

function sendJson(res, code, obj) {
  send(res, code, JSON.stringify(obj), MIME[".json"]);
}

// Resolve `rel` under `root`, refusing anything that escapes it.
function safeJoin(root, rel) {
  const clean = decodeURIComponent(rel).replace(/^\/+/, "");
  const full = path.resolve(root, clean);
  if (full !== root && !full.startsWith(root + path.sep)) return null;
  return full;
}

async function serveFile(res, file) {
  let stat;
  try {
    stat = await fsp.stat(file);
  } catch {
    return send(res, 404, "not found");
  }
  if (stat.isDirectory()) return send(res, 404, "not found");
  const type = MIME[path.extname(file).toLowerCase()] || "application/octet-stream";
  res.writeHead(200, {
    "Content-Type": type,
    "Content-Length": stat.size,
    "Cache-Control": "no-cache",
  });
  fs.createReadStream(file).pipe(res);
}

function readBody(req, limit = 256 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

// ---------------------------------------------------------------- catalogue

// Category order across every kit. Anything unlisted sorts after, alphabetically.
const CATEGORY_ORDER = ["Walls", "Platforms", "Columns", "Props", "Decals", "Aliens", "Enemies", "Guns"];

/**
 * Model files a kit may ship, best first.
 *
 * Babylon's loaders bundle registers a plugin for each, so a module loads the
 * same way whatever it was authored in; the order only decides which copy to
 * list when a pack ships the same models in more than one format.
 */
const MODEL_EXT = [".gltf", ".glb", ".fbx"];
const extensionOf = (name) => MODEL_EXT.find((e) => name.toLowerCase().endsWith(e)) || null;

/**
 * Folder names that name a *file format*, not a category.
 *
 * Quaternius ships some packs as `<Kit>/glTF/…` or `<Kit>/FBX/…`, sometimes
 * both at once. Such a folder says how the pack was exported, not what is in
 * it, so reading it as a category gives a palette with one tab called "glTF" -
 * and reading both would offer every model twice. The set is deliberately
 * wider than the formats we can actually load: `Blend/` has to be recognised
 * as packaging too, or a pack shipping it would look like a foldered kit and
 * grow a "Blend" tab holding nothing.
 *
 * The kits in BabylonAssets are curated - a pack copied in gets its models
 * dispatched into category folders and the wrapper dropped - so this is the
 * path a pack takes on the day it is dropped in, before anyone has sorted it,
 * rather than the path any kit stays on.
 */
const FORMAT_DIRS = new Set([
  "gltf", "glb", "fbx", "obj", "dae", "usd", "usdz", "blend", "blender", "source",
]);

/**
 * The category of a module in a kit that has no category folders.
 *
 * Every kit in the library ships its modules in Walls/, Platforms/, Props/…
 * and the folder IS the category. A pack freshly dropped in has not been
 * sorted yet, and several arrive with the grouping carried in the filename
 * instead - `Enemy_Raptor`, `Gun_Shotgun`, `Prop_Crate` - so the prefix before
 * the first underscore is read as the category and pluralised. A file with no
 * underscore has nothing to group by and lands in "Other" rather than
 * inventing a category per model.
 *
 * The prefix is only a category when it actually groups something, which is
 * why this counts before it names. In some packs the same underscore marks a
 * *variant* - `Potion1_Empty`, `Potion1_Filled`, `Sword_Golden` - and taking
 * it at face value gave the RPG pack thirty tabs called "Potion1s", "Potion2s"…
 * holding two models each. A pair of variants is not a category; below the
 * threshold the model goes to "Other", which says what is true: the filenames
 * of that pack carry no grouping, and if it needs one it has to be foldered.
 */
const MIN_DERIVED_CATEGORY = 3;
const IMAGE_EXT = new Set([".png", ".jpg", ".jpeg", ".webp", ".ktx2", ".dds", ".basis"]);
const prefixOf = (name) => {
  const cut = name.indexOf("_");
  return cut > 0 ? name.slice(0, cut) : "";
};

/**
 * Enough English to name a tab.
 *
 * Blind `+ "s"` is not enough once kits beyond the sci-fi ones are in: the
 * Pirate pack's `Characters_*` came out as "Characterss", the Nature pack's
 * `Bush_*` as "Bushs" and `Cactus_*` as "Cactuss". A prefix that is already
 * plural is left alone, a sibilant takes "es", and a consonant + y becomes
 * "ies" - which is also where "Enemies" now comes from, so there is no table
 * of exceptions to keep in step with the kits.
 */
function pluralise(word) {
  if (/(s|es)$/i.test(word)) return word;
  if (/(sh|ch|x|z)$/i.test(word)) return `${word}es`;
  if (/[^aeiou]y$/i.test(word)) return `${word.slice(0, -1)}ies`;
  return `${word}s`;
}

function categoriseByName(names) {
  const counts = new Map();
  for (const n of names) {
    const p = prefixOf(n);
    if (p) counts.set(p, (counts.get(p) || 0) + 1);
  }
  const out = new Map();
  for (const n of names) {
    const p = prefixOf(n);
    out.set(n, p && counts.get(p) >= MIN_DERIVED_CATEGORY ? pluralise(p) : "Other");
  }
  return out;
}

/** Categories in palette order: the known ones first, then the rest alphabetically. */
function orderCategories(names) {
  return [...names].sort((a, b) => {
    const ia = CATEGORY_ORDER.indexOf(a);
    const ib = CATEGORY_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
}

/**
 * The kits to offer, by name.
 *
 * Alphabetical rather than config order, because the list is read in a combo
 * box: an order somebody chose in a file the reader cannot see is an order
 * they have to scan the whole list to search. A kit dropped into `kits/` shows
 * up without a config edit, in its place among the rest.
 */
async function kitFolders() {
  let present;
  try {
    present = (await fsp.readdir(KITS_DIR, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name);
  } catch {
    return [];
  }
  return present.sort((a, b) => a.localeCompare(b));
}

/**
 * The kit the palette opens on, when nothing is remembered yet.
 *
 * This is what `kits.folders` in the config is for now that the list itself is
 * sorted: the ship is built out of one pack with a second for trimmings, and
 * which one that is cannot be read off an alphabetical list. First name in the
 * config that is actually installed; failing that, the first kit there is.
 */
function defaultKit(names) {
  return KIT_FOLDERS.find((k) => names.includes(k)) || names[0] || null;
}

/**
 * Where a kit file is served from, as a URL the browser can hand to the loader.
 *
 * The path is the same on both sides - it is the BabylonAssets layout either
 * way - so only the origin differs. Encoded per segment, because kit folders
 * have spaces in their names ("Modular SciFi MegaKit").
 */
function kitUrl(relative) {
  const encoded = relative.split("/").map(encodeURIComponent).join("/");
  return KITS_SOURCE === "online" ? `${ONLINE_BASE}${KITS_PREFIX}/${encoded}` : `/assets/${KITS_PREFIX}/${encoded}`;
}

/**
 * Where an unsorted kit's models are, and the files there.
 *
 * The kit root wins whenever it holds models at all - a pack that ships flat.
 * Otherwise the best format wrapper does, ranked by MODEL_EXT, so a pack
 * shipping both `glTF/` and `FBX/` is read as glTF and each model is listed
 * once. Everything else under the kit (Blend/, loose textures, the licence) is
 * left alone.
 */
async function flatSource(root, entries) {
  const here = entries.filter((e) => e.isFile() && extensionOf(e.name)).map((e) => e.name);
  if (here.length) return { dir: "", files: here };

  let best = null;
  for (const e of entries) {
    if (!e.isDirectory() || !FORMAT_DIRS.has(e.name.toLowerCase())) continue;
    const files = (await fsp.readdir(path.join(root, e.name))).filter((f) => extensionOf(f));
    if (!files.length) continue;
    const rank = Math.min(...files.map((f) => MODEL_EXT.indexOf(extensionOf(f))));
    if (!best || rank < best.rank) best = { dir: e.name, files, rank };
  }
  return best || { dir: "", files: [] };
}

/**
 * One kit's modules, grouped into categories, and where its textures live.
 *
 * Three layouts are read:
 *
 *   Walls/, Platforms/, Props/…    the folder IS the category - every kit in
 *                                  the library, and the one to aim for
 *   Enemy_Raptor.gltf, …           flat, with the category in the name
 *   glTF/…  or  FBX/…              a format wrapper
 *
 * The last two are how packs arrive from Quaternius, before anyone has sorted
 * them; a pack is usable the moment it is dropped into `kits/`, and sorting it
 * into folders is a separate step. The wrapper is read as the flat case: it is
 * packaging, not meaning, so it is stepped through and the categories come
 * from the filenames.
 *
 * A kit is scanned as flat only when it has no category folders. Sorting a
 * pack into folders is therefore what fixes its module ids - and a kit that
 * grows a folder later does not silently change every id it already
 * published, because there was no folder to disagree with.
 *
 * `rootTextures` is the shipped-as-is quirk of a kit whose models sit in a
 * subfolder: the model files name their textures with a bare filename, and
 * those textures sit one level up, at the kit root. Sorting a pack keeps them
 * there rather than copying the atlas set into every folder - a glTF URI
 * cannot say "../", and the MegaKit's 27 MB of atlases would become 99 MB
 * across its six folders - so the list travels to the client, which teaches
 * the loader where to look. See `kit.js`.
 */
async function scanKit(kit) {
  const root = path.join(KITS_DIR, kit);
  const entries = await fsp.readdir(root, { withFileTypes: true });
  const categoryDirs = entries
    .filter((e) => e.isDirectory() && !FORMAT_DIRS.has(e.name.toLowerCase()))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b));
  const byCategory = new Map();
  const add = (category, name, relative) => {
    const list = byCategory.get(category);
    // The id is kit-qualified so two kits may hold a `Props/Crate` without one
    // shadowing the other. It is what the manifest stores, so it is a permanent
    // contract, not a display string.
    const module = { id: `${kit}/${category}/${name}`, name, kit, category, url: kitUrl(relative) };
    if (list) list.push(module);
    else byCategory.set(category, [module]);
  };

  // The folders the models are actually served from - one per category, or the
  // single format wrapper. Empty when they sit at the kit root.
  let modelDirs = [];
  if (categoryDirs.length) {
    modelDirs = categoryDirs;
    for (const dir of categoryDirs) {
      for (const f of await fsp.readdir(path.join(root, dir))) {
        const ext = extensionOf(f);
        if (!ext) continue;
        add(dir, f.slice(0, -ext.length), `${kit}/${dir}/${f}`);
      }
    }
  } else {
    const { dir, files } = await flatSource(root, entries);
    if (dir) modelDirs = [dir];
    const names = files.map((f) => f.slice(0, -extensionOf(f).length));
    const category = categoriseByName(names);
    for (const f of files) {
      const name = f.slice(0, -extensionOf(f).length);
      add(category.get(name), name, dir ? `${kit}/${dir}/${f}` : `${kit}/${f}`);
    }
  }

  // Only a kit served out of subfolders needs the redirect: when the models sit
  // at the kit root the textures already sit beside them.
  const rootTextures = modelDirs.length
    ? entries.filter((e) => e.isFile() && IMAGE_EXT.has(path.extname(e.name).toLowerCase())).map((e) => e.name)
    : [];
  return {
    byCategory,
    info: {
      name: kit,
      base: kitUrl(kit) + "/",
      // Display categories, in palette order - the palette shows one kit at a
      // time, so these are its tabs. `modelDirs` is the URL side of the same
      // kit and is not interchangeable with them: under a format wrapper the
      // tab says "Props" while the folder says "glTF".
      categories: orderCategories(byCategory.keys()),
      count: [...byCategory.values()].reduce((n, list) => n + list.length, 0),
      modelDirs,
      rootTextures,
    },
  };
}

/**
 * The saved compound definitions, or an empty list when there are none.
 *
 * Read from disk per request rather than cached, exactly like the kit scan
 * above: the catalogue is rebuilt on every `/api/modules`, so saving a compound
 * and refreshing is all it takes to see it, and two editor tabs cannot disagree
 * about what exists.
 */
function readCompounds() {
  try {
    const parsed = JSON.parse(fs.readFileSync(COMPOUNDS, "utf8"));
    return Array.isArray(parsed?.compounds) ? parsed.compounds : [];
  } catch {
    return [];                          // no file yet is the normal first run
  }
}

/**
 * Is this a compound definition we are willing to write?
 *
 * Checked on the way in rather than trusted, because the file is read straight
 * back into the palette: a member naming no module would become a tile that
 * throws when clicked, and a name with a slash in it would collide with the
 * `@compound/<name>` id space.
 */
function validCompound(c) {
  if (!c || typeof c.name !== "string" || !c.name.trim()) return false;
  if (/[/\\]/.test(c.name)) return false;
  if (typeof c.kit !== "string" || !c.kit.trim()) return false;
  if (typeof c.category !== "string" || !c.category.trim()) return false;
  if (!Array.isArray(c.members) || !c.members.length) return false;
  return c.members.every((m) => m && typeof m.module === "string" && m.module.includes("/"));
}

/**
 * Fold the saved compounds into the catalogue as ordinary-looking modules.
 *
 * A compound tile has no `url`: it is a recipe, and the palette expands it into
 * real placements rather than loading a file. `compound: true` is what tells
 * every consumer that - the thumbnailer builds its preview from the members,
 * and the palette arms a multi-item ghost instead of a single one.
 *
 * They are filed under the kit and category chosen when saving, so a compound
 * built from a kit's walls sits with that kit's walls rather than in a bin of
 * its own. The kit's own tab list is extended where it has to be: a category
 * that exists only because a compound was filed there still needs a tab, or the
 * tile would be unreachable.
 */
function mergeCompounds(categories, seen, kits) {
  for (const c of readCompounds()) {
    if (!validCompound(c)) continue;
    const entry = seen.get(c.category) || { name: c.category, count: 0, modules: [] };
    if (!seen.has(c.category)) {
      seen.set(c.category, entry);
      categories.push(entry);
    }
    entry.modules.push({
      id: `@compound/${c.name}`,
      name: c.name,
      kit: c.kit,
      category: c.category,
      compound: true,
      members: c.members,
    });
    const kit = kits.find((k) => k.name === c.kit);
    if (kit) {
      if (!kit.categories.includes(c.category)) {
        kit.categories = orderCategories([...kit.categories, c.category]);
      }
      kit.count++;
    }
  }
}

const COLLISION_KINDS = new Set(["box", "sphere", "cylinder", "capsule"]);

function validCollisionVector(value) {
  return Array.isArray(value) && value.length === 3
    && value.every((component) => Number.isFinite(Number(component)));
}

function validModuleShape(shape) {
  return shape && COLLISION_KINDS.has(shape.kind)
    && validCollisionVector(shape.position)
    && validCollisionVector(shape.rotation)
    && validCollisionVector(shape.scale);
}

function qualifyCollisionKey(kit, key) {
  const normalized = String(key || "").replaceAll("\\", "/").replace(/^\.?\//, "");
  if (!normalized || normalized.split("/").some((part) => !part || part === "." || part === "..")) return null;
  return normalized.startsWith(`${kit}/`) ? normalized : `${kit}/${normalized}`;
}

function relativeCollisionKey(kit, key) {
  const full = qualifyCollisionKey(kit, key);
  if (!full || !full.startsWith(`${kit}/`)) return null;
  const relative = full.slice(kit.length + 1);
  return relative.split("/").length === 2 ? relative : null;
}

function validateModuleShapes(moduleShapes, kits) {
  if (!moduleShapes || typeof moduleShapes !== "object" || Array.isArray(moduleShapes)) {
    throw new Error("moduleShapes must be an object");
  }
  const installed = new Set(kits);
  const grouped = new Map();
  for (const [moduleId, shapes] of Object.entries(moduleShapes)) {
    const normalized = String(moduleId).replaceAll("\\", "/").replace(/^\.?\//, "");
    const cut = normalized.indexOf("/");
    const kit = cut > 0 ? normalized.slice(0, cut) : "";
    const relative = kit ? relativeCollisionKey(kit, normalized) : null;
    if (!installed.has(kit) || !relative) {
      throw new Error(`collision module is not in an installed kit: ${JSON.stringify(moduleId)}`);
    }
    if (!Array.isArray(shapes) || !shapes.length || shapes.some((shape) => !validModuleShape(shape))) {
      throw new Error(`invalid collision shapes for ${JSON.stringify(moduleId)}`);
    }
    if (!grouped.has(kit)) grouped.set(kit, {});
    grouped.get(kit)[relative] = shapes;
  }
  return grouped;
}

async function readKitCollision() {
  const moduleShapes = {};
  const files = [];
  for (const kit of await kitFolders()) {
    const file = path.join(COLLISION_KITS_DIR, kit, "collision.json");
    let parsed;
    try {
      parsed = JSON.parse(await fsp.readFile(file, "utf8"));
    } catch (error) {
      if (error?.code === "ENOENT") continue;
      throw new Error(`could not read ${file}: ${error.message || error}`);
    }
    const shapes = parsed?.moduleShapes;
    if (!shapes || typeof shapes !== "object" || Array.isArray(shapes)) {
      throw new Error(`${file} has no moduleShapes object`);
    }
    let count = 0;
    for (const [key, records] of Object.entries(shapes)) {
      const full = qualifyCollisionKey(kit, key);
      if (!full || !relativeCollisionKey(kit, full)
        || !Array.isArray(records) || !records.length
        || records.some((shape) => !validModuleShape(shape))) {
        throw new Error(`invalid collision record ${JSON.stringify(key)} in ${file}`);
      }
      moduleShapes[full] = records;
      count++;
    }
    files.push({ kit, path: file, count });
  }

  // One release of the editor used a ship-wide aggregate. Reading it only when
  // no kit file exists gives old checkouts a migration path without allowing
  // the legacy copy to override the new owners.
  if (!files.length) {
    try {
      const legacy = JSON.parse(await fsp.readFile(LEGACY_COLLISION, "utf8"));
      const shapes = legacy?.moduleShapes;
      if (shapes && typeof shapes === "object" && !Array.isArray(shapes)) {
        Object.assign(moduleShapes, shapes);
        return {
          schema: 1,
          moduleShapes,
          legacy: true,
          legacyStageLayout: legacy.stageLayout,
          legacyStageView: legacy.stageView,
          files: [],
        };
      }
    } catch (error) {
      if (error?.code !== "ENOENT") {
        throw new Error(`could not read legacy collision file ${LEGACY_COLLISION}: ${error.message || error}`);
      }
    }
  }
  return { schema: 1, moduleShapes, files };
}

async function writeKitCollision(moduleShapes) {
  const kits = await kitFolders();
  const grouped = validateModuleShapes(moduleShapes, kits);

  const files = [];
  for (const kit of kits) {
    const file = path.join(COLLISION_KITS_DIR, kit, "collision.json");
    const records = Object.fromEntries(Object.entries(grouped.get(kit) || {}).sort(([a], [b]) => a.localeCompare(b)));
    let current = null;
    try {
      current = JSON.parse(await fsp.readFile(file, "utf8"));
    } catch (error) {
      if (error?.code !== "ENOENT") {
        throw new Error(`could not read ${file} before writing: ${error.message || error}`);
      }
    }
    const currentRecords = current?.moduleShapes && typeof current.moduleShapes === "object"
      ? Object.fromEntries(Object.entries(current.moduleShapes).sort(([a], [b]) => a.localeCompare(b)))
      : null;
    if (currentRecords && JSON.stringify(currentRecords) === JSON.stringify(records)) {
      files.push({
        kit,
        path: file,
        count: Object.keys(records).length,
        bytes: (await fsp.stat(file)).size,
        previous: null,
        unchanged: true,
      });
      continue;
    }
    const text = `${JSON.stringify({
      generator: "SciFiShip layout tool",
      schema: 1,
      savedAt: new Date().toISOString(),
      units: "metres",
      kit,
      note: "Collision authored in module-local editor space. Keys are relative to this kit root.",
      moduleShapes: records,
    }, null, 2)}\n`;
    await fsp.mkdir(path.dirname(file), { recursive: true });
    const previous = await rotatePrevious(file);
    await fsp.writeFile(file, text);
    files.push({
      kit,
      path: file,
      count: Object.keys(records).length,
      bytes: Buffer.byteLength(text),
      previous: previous ? path.basename(previous) : null,
      unchanged: false,
    });
  }
  return {
    ok: true,
    count: Object.keys(moduleShapes).length,
    files,
  };
}

async function buildCatalogue() {
  const names = await kitFolders();
  const categories = [];
  const seen = new Map();
  const kits = [];
  for (const name of names) {
    let scan;
    try {
      scan = await scanKit(name);
    } catch (e) {
      console.warn(`could not scan kit "${name}":`, e.message);
      continue;
    }
    kits.push(scan.info);
    for (const [category, modules] of scan.byCategory) {
      // Kits share category names - several have Props - and the catalogue is
      // the whole library, so the modules are merged under one entry rather
      // than the name appearing twice. The palette shows one kit at a time and
      // takes its tabs from that kit's own `categories`; this merged list is
      // what `byId` is built from, so a manifest can name a module from any
      // kit whatever the palette happens to be showing.
      const entry = seen.get(category) || { name: category, count: 0, modules: [] };
      if (!seen.has(category)) {
        seen.set(category, entry);
        categories.push(entry);
      }
      entry.modules.push(...modules);
    }
  }
  mergeCompounds(categories, seen, kits);
  for (const c of categories) {
    c.modules.sort((a, b) => a.kit.localeCompare(b.kit) || a.name.localeCompare(b.name));
    c.count = c.modules.length;
  }
  const order = orderCategories(categories.map((c) => c.name));
  categories.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));

  // fluidSim is the global list a liquefied mesh may draw from; it travels with
  // the catalogue so the app needs no extra round trip on boot. Re-read per
  // request rather than taken from the boot-time CONFIG, so adding a sim to
  // config.json only needs a page refresh.
  return {
    kitsSource: KITS_SOURCE,
    kitsBase: KITS_SOURCE === "online" ? `${ONLINE_BASE}${KITS_PREFIX}/` : KITS_DIR,
    kits,
    // Computed from the kits that survived scanning, not from the folder
    // listing: a kit that could not be read is not one to open on.
    defaultKit: defaultKit(kits.map((k) => k.name)),
    categories,
    fluidSim: readFluidSim(),
    fluidSimFlow: readFluidSimFlow(),
    sounds: readSounds(),
  };
}

function readFluidSim() {
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(HERE, "config.json"), "utf8"));
    if (Array.isArray(cfg.fluidSim)) return cfg.fluidSim;
  } catch (e) {
    console.warn("could not re-read config.json for fluidSim:", e.message);
  }

  return Array.isArray(CONFIG.fluidSim) ? CONFIG.fluidSim : [];
}

function readFluidSimFlow() {
  const flow = {};
  let names = [];
  try {
    names = fs.readdirSync(FLUID_SIM_DIR)
      .filter((name) => name.endsWith(".json"))
      .map((name) => name.slice(0, -5))
      .sort((a, b) => a.localeCompare(b));
  } catch (e) {
    console.warn("could not list fluidSim settings:", e.message);
  }
  for (const name of names) {
    const file = safeJoin(FLUID_SIM_DIR, `${name}.json`);
    if (!file) continue;
    try {
      const setting = JSON.parse(fs.readFileSync(file, "utf8"));
      flow[name] = {
        emitters: flowObjectNames(setting.emitters),
        sinks: flowObjectNames(setting.sinks),
      };
    } catch (e) {
      console.warn(`could not read fluidSim "${name}":`, e.message);
    }
  }
  return flow;
}

/**
 * The clips a behaviour may name, taken from the game's own sound folder.
 *
 * The runtime's contract is an **MP3 base name** under `/aquanova/sounds/` -
 * `sound.ts`, `pick-entity.ts` and `weapon-liquefactor.ts` all reject a name
 * carrying an extension or a slash - so the folder is the only honest source
 * for this list, and other audio formats are deliberately not offered.
 *
 * It used to be typed into behavior-definitions.json beside the schema that
 * refers to it, and had already drifted: two clips on disk were missing from
 * it, and nothing anywhere could have noticed. Rebuilt per request like the
 * rest of the catalogue, so dropping an MP3 into the folder only needs a page
 * refresh.
 */
function readSounds() {
  try {
    return fs.readdirSync(SOUNDS_DIR)
      .filter((name) => name.toLowerCase().endsWith(".mp3"))
      .map((name) => name.slice(0, -4))
      .sort((a, b) => a.localeCompare(b));
  } catch (e) {
    console.warn("could not list sounds:", e.message);
    return [];
  }
}

function flowObjectNames(value) {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.map((item) => (typeof item?.name === "string" ? item.name.trim() : "")).filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

/**
 * Drop earlier-version copies of a cached image.
 *
 * The client stamps a version into every cache key so a change to how a
 * thumbnail is *rendered* invalidates the pictures already on disk. Without
 * this the superseded ones would sit there for ever, and 300 stills plus 300
 * twelve-frame sheets is not a small folder to leave behind.
 */
async function dropOlderVersions(dir, key) {
  const m = /^v(\d+)_(.*)$/.exec(key);
  if (!m) return;
  const [, version, id] = m;
  let names = [];
  try { names = await fsp.readdir(dir); } catch { return; }
  await Promise.all(names.map(async (f) => {
    if (!f.endsWith(".png")) return;
    const o = /^v(\d+)_(.*)\.png$/.exec(f);
    // a file with no version at all predates the scheme, and is superseded by
    // definition - the caches in the wild are entirely unversioned, so without
    // this the very first bump would orphan every one of them
    const stale = o ? (o[2] === id && o[1] !== version) : f.slice(0, -4) === id;
    if (stale) await fsp.rm(path.join(dir, f), { force: true }).catch(() => {});
  }));
}

// ------------------------------------------------- the environment probes
//
// A probe is a box the ship author draws around a room, plus the point inside
// it the cubemap is captured from. The capture itself happens in the browser -
// Babylon renders the six faces, prefilters them and serialises the .env - so
// this end owns only the two things a browser cannot: the folder the files
// live in, and the index that says what is in it.
//
// The index is not authored here and is not a second copy of the ship. It is a
// record of what has been GENERATED: one entry per probe, carrying the volume
// it projects onto, the point it was taken from, and the digest of the scene
// state it was taken of. `sync-ship.ts` reads exactly this file, and the stamp
// beside each .env is what lets it refuse a half-copied folder.

const ENVIRONMENT_DIR = path.join(EXPORT_DIR, "environments");
const LOCAL_ENV_INDEX = path.join(ENVIRONMENT_DIR, "local-environments.json");

/** The .env a probe id is written to. Ids are validated before they get here. */
function environmentFileName(id) {
  return `local_environment_${id}.env`;
}

/**
 * A probe id has to be safe as a file name and stable as a JSON key, so it is
 * restricted rather than escaped: an id that cannot be written verbatim is
 * refused, and the editor never offers one.
 */
function validProbeId(id) {
  const ident = String(id || "").trim();
  if (!ident || ident.length > 128) return null;
  return /^[A-Za-z0-9._-]+$/.test(ident) ? ident : null;
}

function readLocalEnvironmentIndex() {
  try {
    const data = JSON.parse(fs.readFileSync(LOCAL_ENV_INDEX, "utf8"));
    if (!data || typeof data !== "object") return { probes: {}, chunks: {} };
    const chunks = data.chunks && typeof data.chunks === "object" ? data.chunks : {};
    const probes = data.probes && typeof data.probes === "object" ? data.probes : null;
    // A folder written before explicit probes only has `chunks`, keyed by what
    // was then each chunk's own probe id - which is exactly what a probe id
    // looks like, so it doubles as the `probes` map.
    return { probes: probes || chunks, chunks };
  } catch {
    return { probes: {}, chunks: {} };
  }
}

// The same `probes` primary / `chunks` legacy-alias fallback the runtime and
// sync-ship.ts use.
function localEnvironmentEntry(index, id) {
  return (index.probes && index.probes[id]) || (index.chunks && index.chunks[id]) || null;
}

/**
 * What is on disk, and which probes still owe a capture.
 *
 * `pending` is decided against the index alone - a probe whose .env is missing,
 * or whose stamp does not match the digest recorded beside it. Whether the SHIP
 * has moved since is a question only the editor can answer, because the digest
 * describes a live scene rather than a file: the browser compares its own
 * digest with `hash` and declares what it needs before it starts.
 */
function localEnvironmentStatus() {
  const index = readLocalEnvironmentIndex();
  const pending = [];
  for (const [id, entry] of Object.entries(index.probes)) {
    const env = safeJoin(ENVIRONMENT_DIR, String(entry?.env || ""));
    const stamp = env ? `${env}.stamp` : null;
    const current = stamp && fs.existsSync(stamp)
      ? fs.readFileSync(stamp, "utf8").trim() : "";
    if (!env || !fs.existsSync(env) || current !== entry?.hash) {
      pending.push({
        id,
        env: entry?.env,
        position: entry?.position,
        shape: entry?.shape,
        boxPosition: entry?.boxPosition,
        boxSize: entry?.boxSize,
        spherePosition: entry?.spherePosition,
        sphereRadius: entry?.sphereRadius,
        angle: entry?.angle,
        resolution: entry?.resolution,
        hash: entry?.hash,
      });
    }
  }
  return { probes: index.probes, chunks: index.chunks, pending };
}

async function writeLocalEnvironmentIndex(index) {
  await fsp.mkdir(ENVIRONMENT_DIR, { recursive: true });
  await fsp.writeFile(LOCAL_ENV_INDEX, `${JSON.stringify(index, null, 1)}\n`);
}

/**
 * Declare the probes the ship currently has, and get back what still owes a
 * capture.
 *
 * The editor sends its whole authored list on every generate, so this is also
 * where a deleted probe leaves the index: an id that is not declared is
 * dropped, along with the .env and stamp it left behind. Nothing else prunes
 * them, and a stale entry would otherwise be published to the runtime for ever.
 *
 * A declaration never invents a capture. It records the digest the editor is
 * about to capture AT, which is what makes the file on disk stale the moment
 * the ship moves - the stamp still holds the digest of the render that
 * happened, and the two only agree again once the probe is taken again.
 *
 * `force` deletes the stamps rather than special-casing the comparison, so
 * there is still exactly one rule for what is pending: the stamp beside a file
 * has to match the digest declared for it. A forced probe simply has no stamp
 * to match, which is also true after a crash mid-capture - and it means an
 * interrupted force leaves the remaining probes still pending rather than
 * looking done.
 */
async function declareLocalEnvironments(declared, force = false) {
  const index = readLocalEnvironmentIndex();
  const probes = {};
  const seen = new Set();
  for (const probe of Array.isArray(declared) ? declared : []) {
    const id = validProbeId(probe?.id);
    if (!id) continue;
    seen.add(id);
    const previous = localEnvironmentEntry(index, id) || {};
    const unchanged = previous.hash === probe.hash;
    probes[id] = {
      id,
      env: environmentFileName(id),
      position: probe.position,
      shape: probe.shape === "sphere" ? "sphere" : undefined,
      boxPosition: probe.boxPosition,
      boxSize: probe.boxSize,
      spherePosition: probe.spherePosition,
      sphereRadius: probe.sphereRadius,
      angle: probe.angle,
      resolution: probe.resolution,
      hash: probe.hash,
      // Carried over so an untouched probe stays byte-identical in the index
      // and only a real recapture rewrites its row.
      generatedAt: unchanged && !force ? previous.generatedAt ?? null : null,
      bytes: unchanged && !force ? previous.bytes ?? null : null,
    };
  }
  await fsp.mkdir(ENVIRONMENT_DIR, { recursive: true });
  if (force) {
    for (const entry of Object.values(probes)) {
      const env = safeJoin(ENVIRONMENT_DIR, String(entry.env || ""));
      if (env) await fsp.rm(`${env}.stamp`, { force: true }).catch(() => {});
    }
  }
  for (const [id, entry] of Object.entries(index.probes)) {
    if (seen.has(id)) continue;
    const env = safeJoin(ENVIRONMENT_DIR, String(entry?.env || ""));
    if (!env) continue;
    await fsp.rm(env, { force: true }).catch(() => {});
    await fsp.rm(`${env}.stamp`, { force: true }).catch(() => {});
  }
  await writeLocalEnvironmentIndex({ probes });
  return localEnvironmentStatus();
}

// ------------------------------------------------------ publishing the demo

/**
 * The checkout root, found by the tool tsx lives in rather than by counting.
 *
 * `../../../../../../..` would be the same answer today and a silent breakage
 * the day this folder moves. What the publish actually needs is the tsx CLI, so
 * looking for that is both the search and the check: no `node_modules/tsx`
 * means `pnpm install` has not been run, which is worth saying plainly rather
 * than failing later inside a spawn.
 */
function findTsxCli() {
  for (let dir = HERE; ; dir = path.dirname(dir)) {
    const cli = path.join(dir, "node_modules", "tsx", "dist", "cli.mjs");
    if (fs.existsSync(cli)) return { cli, root: dir };
    if (path.dirname(dir) === dir) return null;
  }
}

/**
 * One publish at a time.
 *
 * The script writes into `lab/public/aquanova`, so two of them interleaved
 * would be two writers over one folder. The editor disables its own button for
 * the duration, but that is one tab's opinion. Concurrent requests join the
 * active publish so another editor tab does not fail with HTTP 409.
 */
let activeSync = null;

/**
 * Run sync-ship.ts and report what it said.
 *
 * Spawned through `process.execPath` and tsx's own CLI rather than through
 * `pnpm tsx`: no shell to quote a path into, nothing on `PATH` to find, and the
 * same trick the lab's own dev server uses to run TypeDoc.
 *
 * Never throws and never sends a non-200 for a script that merely failed: the
 * output *is* the answer, and the editor shows it. A 4xx/5xx is reserved for
 * this server being unable to run it at all.
 */
async function runSyncShip(optimize) {
  if (!SYNC_SCRIPT) return { ok: false, error: "config.json has no demo.syncScript" };
  if (!fs.existsSync(SYNC_SCRIPT)) return { ok: false, error: `${SYNC_SCRIPT} does not exist` };
  if (!fs.existsSync(GLB)) return { ok: false, error: `${GLB} does not exist` };
  const tsx = findTsxCli();
  if (!tsx) return { ok: false, error: "node_modules/tsx not found — run pnpm install in the repository" };

  const source = fs.statSync(GLB, { bigint: true });
  const shipName = `ship-${source.size}-${source.mtimeNs}-${optimize ? "opt" : "raw"}.glb`;
  const args = [tsx.cli, SYNC_SCRIPT, "--ship-output-name", shipName];
  // The flag the script actually takes. Optimising is opt-IN because it runs
  // toktx over every texture on the ship, which is minutes rather than the
  // second a plain copy costs - and what a publish is usually for is looking at
  // the room you have just moved a wall in.
  if (!optimize) args.push("--no-ship-optimize");
  const command = `node ${args.map((a) => (a.includes(" ") ? `"${a}"` : a)).join(" ")}`;

  return await new Promise((resolve) => {
    const child = spawn(process.execPath, args, { cwd: tsx.root });
    let output = "";
    // Interleaved into one string on purpose: the script reports progress on
    // stdout and its errors on stderr, and which line came after which is
    // most of what makes a failure readable.
    const collect = (chunk) => { output += chunk.toString(); };
    child.stdout.on("data", collect);
    child.stderr.on("data", collect);
    const timer = setTimeout(() => {
      output += `\n[server] no answer after ${Math.round(SYNC_TIMEOUT_MS / 60000)} minutes — killed\n`;
      child.kill();
    }, SYNC_TIMEOUT_MS);
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({ ok: false, error: String(err && err.message || err), command, output, shipName });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ ok: code === 0, code, command, output, optimized: !!optimize, shipName });
    });
  });
}

function runOrJoinSyncShip(optimize) {
  if (activeSync) return activeSync;
  activeSync = runSyncShip(optimize).finally(() => {
    activeSync = null;
  });
  return activeSync;
}

// ------------------------------------------------------------------ routes

async function handle(req, res) {
  const url = new URL(req.url, "http://localhost");
  const p = url.pathname;

  if (p === "/favicon.ico") return send(res, 204, "");

  if (p === "/api/modules") {
    // Deliberately not cached: scanning six directories takes a couple of
    // milliseconds and this is hit once per page load, whereas a stale cache
    // silently hides files added or renamed in the kit folder.
    return sendJson(res, 200, await buildCatalogue());
  }

  // BabylonAssets, mounted read-only at the same paths the CDN uses, so a module
  // URL differs between local and online only in its origin. Serving it here
  // rather than from a second web server keeps the app same-origin with its own
  // assets: no CORS, no extra process to remember to start.
  //
  // Present in online mode too. The catalogue is still built by listing the
  // local folder - a static CDN cannot be asked what it holds - and the
  // thumbnailer and any hand-written URL keep working while the CDN catches up.
  if (p.startsWith("/assets/")) {
    const file = safeJoin(ASSETS_DIR, p.slice("/assets/".length));
    if (!file) return send(res, 403, "forbidden");
    return serveFile(res, file);
  }

  if (p.startsWith("/env/")) {
    const file = safeJoin(ENV_DIR, p.slice("/env/".length));
    if (!file) return send(res, 403, "forbidden");
    return serveFile(res, file);
  }

  // Thumbnails are rendered client-side once, then cached here so later
  // sessions get them instantly instead of re-loading 277 glTF files.
  if (p.startsWith("/api/thumb/")) {
    const key = p.slice("/api/thumb/".length).replace(/[^A-Za-z0-9_.-]/g, "_");
    const file = path.join(THUMB_DIR, key + ".png");
    if (req.method === "GET") return serveFile(res, file);
    if (req.method === "PUT") {
      const body = await readBody(req, 4 * 1024 * 1024);
      await fsp.mkdir(THUMB_DIR, { recursive: true });
      await fsp.writeFile(file, body);
      await dropOlderVersions(THUMB_DIR, key);
      return sendJson(res, 200, { ok: true });
    }
    return send(res, 405, "method not allowed");
  }

  if (p === "/api/thumbs") {
    let names = [];
    try {
      names = (await fsp.readdir(THUMB_DIR)).filter((f) => f.endsWith(".png"))
        .map((f) => f.slice(0, -4));
    } catch { /* no cache yet */ }
    return sendJson(res, 200, { cached: names });
  }

  // Turntable sprite sheets: one 12-frame 360 deg turn per module, shown while
  // a palette tile is hovered. Same client-renders-once, server-caches deal as
  // the stills, but a sheet is 12 frames wide so it needs a larger body cap.
  if (p.startsWith("/api/turn/")) {
    const key = p.slice("/api/turn/".length).replace(/[^A-Za-z0-9_.-]/g, "_");
    const file = path.join(TURN_DIR, key + ".png");
    if (req.method === "GET") return serveFile(res, file);
    if (req.method === "PUT") {
      const body = await readBody(req, 24 * 1024 * 1024);
      await fsp.mkdir(TURN_DIR, { recursive: true });
      await fsp.writeFile(file, body);
      await dropOlderVersions(TURN_DIR, key);
      return sendJson(res, 200, { ok: true });
    }
    return send(res, 405, "method not allowed");
  }

  if (p === "/api/turns") {
    let names = [];
    try {
      names = (await fsp.readdir(TURN_DIR)).filter((f) => f.endsWith(".png"))
        .map((f) => f.slice(0, -4));
    } catch { /* no cache yet */ }
    return sendJson(res, 200, { cached: names });
  }

  // The manifest is the source of truth for the layout: it holds every
  // placement as module id + transform, so the editor can be restored exactly.
  if (p === "/api/layout") {
    if (req.method === "GET") {
      const name = url.searchParams.get("name");
      const file = name ? safeJoin(LAYOUT_DIR, name + ".json") : MANIFEST;
      if (!file) return send(res, 403, "forbidden");
      try {
        return send(res, 200, await fsp.readFile(file), MIME[".json"]);
      } catch {
        // Not an error: a fresh project simply has nothing saved yet. Returning
        // 404 makes the browser log a console error on every clean start.
        return sendJson(res, 200, {});
      }
    }
    if (req.method === "POST") {
      const body = await readBody(req);
      const name = url.searchParams.get("name");
      const file = name ? safeJoin(LAYOUT_DIR, name + ".json") : MANIFEST;
      if (!file) return send(res, 403, "forbidden");
      await fsp.mkdir(path.dirname(file), { recursive: true });
      const previous = await rotatePrevious(file);
      await fsp.writeFile(file, body);
      return sendJson(res, 200, {
        ok: true, path: file, bytes: body.length,
        previous: previous ? path.basename(previous) : null,
      });
    }
    return send(res, 405, "method not allowed");
  }

  if (p === "/api/collision") {
    // Each kit owns one collision.json at its root. The browser still sees one
    // aggregate map, so previews and the collision bench do not care which kit
    // a module came from; this route qualifies keys on read and splits them on
    // write.
    if (req.method === "GET") {
      return sendJson(res, 200, await readKitCollision());
    }
    if (req.method === "POST") {
      let parsed;
      try {
        parsed = JSON.parse((await readBody(req)).toString("utf8"));
      } catch {
        return sendJson(res, 400, { ok: false, error: "body is not JSON" });
      }
      try {
        return sendJson(res, 200, await writeKitCollision(parsed?.moduleShapes));
      } catch (error) {
        return sendJson(res, 400, { ok: false, error: error.message || String(error) });
      }
    }
    return send(res, 405, "method not allowed");
  }

  if (p === "/api/compounds") {
    // The whole list, written in one go. A compound is small and there are
    // never many, so read-modify-write in the client and post the result is
    // simpler and less racy than per-name routes - and it makes "delete" the
    // same operation as "save", which is one code path to get right instead of
    // three.
    if (req.method === "GET") {
      return sendJson(res, 200, { compounds: readCompounds() });
    }
    if (req.method === "POST") {
      let list;
      try {
        const body = await readBody(req, 8 * 1024 * 1024);
        list = JSON.parse(body.toString("utf8"))?.compounds;
      } catch { return sendJson(res, 400, { ok: false, error: "body is not JSON" }); }
      if (!Array.isArray(list)) {
        return sendJson(res, 400, { ok: false, error: "compounds must be an array" });
      }
      const bad = list.find((c) => !validCompound(c));
      if (bad) {
        return sendJson(res, 400, {
          ok: false,
          error: `invalid compound: ${JSON.stringify(bad?.name ?? null)}`,
        });
      }
      const names = list.map((c) => c.name);
      if (new Set(names).size !== names.length) {
        return sendJson(res, 400, { ok: false, error: "two compounds share a name" });
      }
      const text = `${JSON.stringify({ schema: 1, compounds: list }, null, 2)}\n`;
      await fsp.mkdir(path.dirname(COMPOUNDS), { recursive: true });
      // Rotated because these are hand-built and there is no second copy of
      // them anywhere.
      const previous = await rotatePrevious(COMPOUNDS);
      await fsp.writeFile(COMPOUNDS, text);
      return sendJson(res, 200, {
        ok: true, path: COMPOUNDS, count: list.length, bytes: Buffer.byteLength(text),
        previous: previous ? path.basename(previous) : null,
      });
    }
    return send(res, 405, "method not allowed");
  }

  if (p === "/api/autosave") {
    // A recovery file, written every couple of minutes while you work.
    //
    // Rotated, so every one of those couple-of-minute states is kept. That is
    // hundreds of small files over a long session, and that is the point: the
    // whole value of an auto-save is having the state from *before* whatever
    // went wrong, and you cannot know in advance which one that is. They are a
    // few kilobytes each and deleting them is one command.
    //
    // Deliberately not the manifest, so a background write can never overwrite
    // the ship you last chose to save.
    if (req.method === "GET") {
      try {
        return send(res, 200, await fsp.readFile(AUTOSAVE), MIME[".json"]);
      } catch {
        return sendJson(res, 200, {});
      }
    }
    if (req.method === "POST") {
      const body = await readBody(req);
      await fsp.mkdir(path.dirname(AUTOSAVE), { recursive: true });
      const previous = await rotatePrevious(AUTOSAVE);
      await fsp.writeFile(AUTOSAVE, body);
      return sendJson(res, 200, {
        ok: true, path: AUTOSAVE, bytes: body.length,
        previous: previous ? path.basename(previous) : null,
      });
    }
    return send(res, 405, "method not allowed");
  }

  if (p === "/api/layouts") {
    let names = [];
    try {
      names = (await fsp.readdir(LAYOUT_DIR)).filter((f) => f.endsWith(".json"))
        .map((f) => f.slice(0, -5)).sort();
    } catch { /* none yet */ }
    return sendJson(res, 200, { layouts: names });
  }

  if (p === "/api/export" && req.method === "POST") {
    const body = await readBody(req);
    await fsp.mkdir(EXPORT_DIR, { recursive: true });
    await fsp.writeFile(GLB, body);
    return sendJson(res, 200, { ok: true, path: GLB, bytes: body.length });
  }

  if (p === "/api/local-environments" && req.method === "GET") {
    return sendJson(res, 200, localEnvironmentStatus());
  }

  if (p === "/api/local-environments" && req.method === "POST") {
    let declared = [];
    let force = false;
    try {
      const body = await readBody(req, 4 * 1024 * 1024);
      const parsed = JSON.parse(body.toString("utf8"));
      declared = parsed?.probes;
      force = !!parsed?.force;
    } catch { return sendJson(res, 400, { ok: false, error: "body is not JSON" }); }
    if (!Array.isArray(declared)) {
      return sendJson(res, 400, { ok: false, error: "probes must be an array" });
    }
    return sendJson(res, 200, await declareLocalEnvironments(declared, force));
  }

  if (p.startsWith("/api/local-environment/") && req.method === "PUT") {
    const id = decodeURIComponent(p.slice("/api/local-environment/".length));
    const index = readLocalEnvironmentIndex();
    const entry = localEnvironmentEntry(index, id);
    if (!entry) return sendJson(res, 404, { ok: false, error: "unknown probe" });
    const hash = url.searchParams.get("hash") || "";
    if (!hash || hash !== entry.hash) {
      return sendJson(res, 409, { ok: false, error: "probe changed during capture" });
    }
    const file = safeJoin(ENVIRONMENT_DIR, String(entry.env || ""));
    if (!file || path.extname(file).toLowerCase() !== ".env") {
      return sendJson(res, 400, { ok: false, error: "invalid environment filename" });
    }
    const body = await readBody(req, 128 * 1024 * 1024);
    // Re-read: an authoring edit during the upload invalidates these bytes.
    const fresh = readLocalEnvironmentIndex();
    if (localEnvironmentEntry(fresh, id)?.hash !== hash) {
      return sendJson(res, 409, { ok: false, error: "probe changed during capture" });
    }
    await fsp.mkdir(path.dirname(file), { recursive: true });
    await fsp.writeFile(file, body);
    await fsp.writeFile(`${file}.stamp`, hash);
    // The row is only complete once the bytes are down: `generatedAt`/`bytes`
    // are what a reader looks at to tell a declared probe from a captured one.
    if (fresh.probes[id]) {
      fresh.probes[id] = { ...fresh.probes[id], bytes: body.length, generatedAt: new Date().toISOString() };
      await writeLocalEnvironmentIndex({ probes: fresh.probes });
    }
    return sendJson(res, 200, { ok: true, bytes: body.length, file: path.basename(file) });
  }

  // Publish the export folder into the demo and hand back the page to open.
  // The editor opens the tab itself, so a failed publish opens nothing.
  if (p === "/api/sync-ship" && req.method === "POST") {
    if (!DEMO_URL) return sendJson(res, 501, { ok: false, error: "config.json has no demo.url" });
    let optimize = false;
    try {
      const body = await readBody(req, 64 * 1024);
      const text = body.toString("utf8").trim();
      if (text) optimize = !!JSON.parse(text)?.optimize;
    } catch { return sendJson(res, 400, { ok: false, error: "body is not JSON" }); }
    const result = await runOrJoinSyncShip(optimize);
    const demoUrl = new URL(DEMO_URL);
    if (result.ok && result.shipName) demoUrl.searchParams.set("ship", result.shipName);
    return sendJson(res, 200, { ...result, url: demoUrl.href });
  }

  if (p.startsWith("/environments/")) {
    const file = safeJoin(ENVIRONMENT_DIR, p.slice("/environments/".length));
    if (!file) return send(res, 403, "forbidden");
    return serveFile(res, file);
  }

  // Read-only: the editor writes here through /api/export, never through a URL.
  if (p.startsWith("/export/") && req.method === "GET") {
    const file = safeJoin(EXPORT_DIR, p.slice("/export/".length));
    if (!file) return send(res, 403, "forbidden");
    return serveFile(res, file);
  }

  // static app
  const rel = p === "/" ? "index.html" : p;
  const file = safeJoin(PUBLIC_DIR, rel);
  if (!file) return send(res, 403, "forbidden");
  return serveFile(res, file);
}

/**
 * Move the previous manifest aside as ship_manifest.<timestamp>.json before a
 * save. The manifest is the only non-derivable artefact in the project, so
 * every save keeps a full history rather than a single rolling backup.
 */
async function rotatePrevious(file) {
  if (!fs.existsSync(file)) return null;
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`
    + `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;

  let dest = file.replace(/\.json$/, `.${stamp}.json`);
  for (let n = 2; fs.existsSync(dest); n++) {         // two saves in one second
    dest = file.replace(/\.json$/, `.${stamp}-${n}.json`);
  }
  await fsp.rename(file, dest);
  return dest;
}

const server = http.createServer((req, res) => {
  handle(req, res).catch((err) => {
    console.error(req.method, req.url, err);
    if (!res.headersSent) send(res, 500, String(err && err.message || err));
  });
});

server.listen(PORT, () => {
  console.log(`ship layout tool   http://localhost:${PORT}`);
  console.log(`kits (${KITS_SOURCE})${" ".repeat(Math.max(1, 12 - KITS_SOURCE.length))}${KITS_SOURCE === "online" ? `${ONLINE_BASE}${KITS_PREFIX}/` : KITS_DIR}`);
  if (COLLISION_KITS_DIR !== KITS_DIR) console.log(`kit collision      ${COLLISION_KITS_DIR}`);
  console.log(`export             ${EXPORT_DIR}`);
  if (!fs.existsSync(KITS_DIR)) {
    console.warn(`WARNING: ${KITS_DIR} does not exist - set kits.localDir in config.json or SHIP_ASSETS_DIR.`);
    console.warn("         A local BabylonAssets checkout is needed to LIST the kits even in online mode.");
  }
});
