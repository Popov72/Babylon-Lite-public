// Zero-dependency static server + kit API for the SciFi ship layout tool.
//
//   node server.mjs            -> http://localhost:5173
//
// It does four things:
//   1. serves public/ (the app itself)
//   2. mounts the Modular SciFi MegaKit glTF folder read-only at /kit/...
//   3. exposes the module catalogue and a thumbnail cache
//   4. writes the layout manifest and the exported ship.glb to disk

import http from "node:http";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONFIG = JSON.parse(fs.readFileSync(path.join(HERE, "config.json"), "utf8"));

// Every path in config.json is resolved against the *config file*, not the
// working directory, so the relative defaults keep pointing at this copy's own
// folders wherever the editor is checked out or copied to. An absolute path
// still wins, for anyone pointing the tool at a kit held somewhere else.
const KIT_DIR = path.resolve(HERE, CONFIG.kitDir);
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
const THUMB_DIR = path.join(HERE, "cache", "thumbs");
const TURN_DIR = path.join(HERE, "cache", "turntable");
const LAYOUT_DIR = path.join(HERE, "layouts");

const MANIFEST = path.join(EXPORT_DIR, "ship_manifest.json");
const COLLISION = path.join(EXPORT_DIR, "ship_collision.json");
const AUTOSAVE = path.join(EXPORT_DIR, "ship_autosave.json");
const GLB = path.join(EXPORT_DIR, "ship.glb");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".gltf": "model/gltf+json",
  ".glb": "model/gltf-binary",
  ".bin": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".hdr": "application/octet-stream",
  ".env": "application/octet-stream",
  ".svg": "image/svg+xml",
};

const CATEGORY_ORDER = ["Walls", "Platforms", "Columns", "Props", "Decals", "Aliens"];

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

async function buildCatalogue() {
  const entries = await fsp.readdir(KIT_DIR, { withFileTypes: true });
  const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);
  dirs.sort((a, b) => {
    const ia = CATEGORY_ORDER.indexOf(a);
    const ib = CATEGORY_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });

  const categories = [];
  for (const dir of dirs) {
    const files = await fsp.readdir(path.join(KIT_DIR, dir));
    const modules = files
      .filter((f) => f.toLowerCase().endsWith(".gltf"))
      .map((f) => {
        const name = f.slice(0, -".gltf".length);
        return { id: `${dir}/${name}`, name, category: dir, url: `/kit/${dir}/${f}` };
      })
      .sort((a, b) => a.name.localeCompare(b.name));
    if (modules.length) categories.push({ name: dir, count: modules.length, modules });
  }
  // fluidSim is the global list a liquefied mesh may draw from; it travels with
  // the catalogue so the app needs no extra round trip on boot. Re-read per
  // request rather than taken from the boot-time CONFIG, so adding a sim to
  // config.json only needs a page refresh.
  return { kitDir: KIT_DIR, categories, fluidSim: readFluidSim() };
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

  if (p.startsWith("/kit/")) {
    const rel = p.slice("/kit/".length);
    const file = safeJoin(KIT_DIR, rel);
    if (!file) return send(res, 403, "forbidden");
    // The kit's .gltf files reference their textures by bare filename
    // ("T_Trim_02_ORM.png") but the PNGs live once at the glTF root rather than
    // beside each module, so fall back to the root before giving up.
    if (!fs.existsSync(file)) {
      const atRoot = safeJoin(KIT_DIR, path.basename(rel));
      if (atRoot && fs.existsSync(atRoot)) return serveFile(res, atRoot);
    }
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
      await backupIfForeign(file);
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
    // Collision authored per kit module, kept in its own file so it can be
    // shipped and reused: a new ship built from the same kit gets every hull
    // back without authoring any of it again. The manifest carries the same
    // data for the runtime, but this is the copy meant to travel.
    if (req.method === "GET") {
      try {
        return send(res, 200, await fsp.readFile(COLLISION), MIME[".json"]);
      } catch {
        return sendJson(res, 200, {});
      }
    }
    if (req.method === "POST") {
      const body = await readBody(req);
      await fsp.mkdir(path.dirname(COLLISION), { recursive: true });
      // Not rotated, unlike the manifest. It is written on every save, so a
      // timestamped copy each time buried the export folder - and it is not the
      // only copy: every manifest carries the same hulls in `moduleShapes`, so
      // one can be rebuilt from any saved ship.
      await fsp.writeFile(COLLISION, body);
      return sendJson(res, 200, { ok: true, path: COLLISION, bytes: body.length });
    }
    return send(res, 405, "method not allowed");
  }

  if (p === "/api/autosave") {
    // A rolling recovery file. Deliberately *not* rotated: the manifest keeps a
    // timestamped copy of every deliberate save, and doing the same here would
    // bury the export folder under a file every couple of minutes. It is also
    // deliberately not the manifest, so a background write can never overwrite
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
      await fsp.writeFile(AUTOSAVE, body);
      return sendJson(res, 200, { ok: true, path: AUTOSAVE, bytes: body.length });
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
    await backupGlbIfForeign();
    await fsp.writeFile(GLB, body);
    return sendJson(res, 200, { ok: true, path: GLB, bytes: body.length });
  }

  // static app
  const rel = p === "/" ? "index.html" : p;
  const file = safeJoin(PUBLIC_DIR, rel);
  if (!file) return send(res, 403, "forbidden");
  return serveFile(res, file);
}

// The Blender pipeline already wrote an export/ship_manifest.json. It has no
// `instances` array, so it cannot be reloaded by the editor - preserve it
// once rather than silently destroying a known-good artefact.
async function backupIfForeign(file) {
  try {
    const existing = JSON.parse(await fsp.readFile(file, "utf8"));
    if (existing.instances) return;
    const bak = file.replace(/\.json$/, ".blender.bak.json");
    if (!fs.existsSync(bak)) await fsp.copyFile(file, bak);
    console.log("preserved pre-existing manifest ->", bak);
  } catch { /* absent or unparseable, nothing to preserve */ }
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

// Same idea for the geometry: the first ship.glb we are asked to overwrite was
// not written by us, so keep a copy before clobbering it.
async function backupGlbIfForeign() {
  const bak = GLB.replace(/\.glb$/, ".blender.bak.glb");
  if (!fs.existsSync(GLB) || fs.existsSync(bak)) return;
  await fsp.copyFile(GLB, bak);
  console.log("preserved pre-existing ship.glb ->", bak);
}

const server = http.createServer((req, res) => {
  handle(req, res).catch((err) => {
    console.error(req.method, req.url, err);
    if (!res.headersSent) send(res, 500, String(err && err.message || err));
  });
});

server.listen(PORT, () => {
  console.log(`ship layout tool   http://localhost:${PORT}`);
  console.log(`kit                ${KIT_DIR}`);
  console.log(`export             ${EXPORT_DIR}`);
  if (!fs.existsSync(KIT_DIR)) console.warn("WARNING: kitDir does not exist");
});
