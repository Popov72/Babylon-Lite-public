// Headless smoke test for the layout tool. Run with:
//   npm test
// which starts an isolated server; the suite refuses to guess one, because it
// is destructive and the editor you build in is on 5180.
// Uses the Playwright already present in the Babylon.js checkout and drives
// installed Edge, so nothing new has to be downloaded.

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { toolUrl } from "./target.mjs";

const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const HERE = path.dirname(fileURLToPath(import.meta.url));
const URL = toolUrl();

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });

const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));

await page.goto(URL, { waitUntil: "domcontentloaded" });
// networkidle never settles: the palette streams thumbnails continuously.
await page.waitForFunction(
  () => document.querySelectorAll("#palette-list .item").length > 0,
  null, { timeout: 60000 });
await page.waitForTimeout(3000);

if (errors.length) console.log("boot errors :\n  " + [...new Set(errors)].join("\n  "));
console.log("status      :", await page.textContent("#status-text"));
console.log("palette     :", await page.evaluate(
  () => document.querySelectorAll("#palette-list .item").length), "items");

const placed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new BABYLON.Vector3(0, 0, 0));
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new BABYLON.Vector3(4, 0, 0));
  await ed.placeAt("Modular SciFi MegaKit/Platforms/Door_Frame_A", new BABYLON.Vector3(8, 0, 0));
  return {
    placements: ed.state.placements.size,
    meshes: window.__scene.meshes.length,
    materials: window.__scene.materials.length,
    textures: window.__scene.textures.length,
  };
}).catch((e) => ({ error: String(e) }));
console.log("after place :", JSON.stringify(placed));

// Placing the same module twice must not duplicate its materials or textures.
const dedupe = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const before = { m: window.__scene.materials.length, t: window.__scene.textures.length };
  for (let i = 0; i < 6; i++) {
    await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new BABYLON.Vector3(i * 4, 0, 8));
  }
  return { before, after: { m: window.__scene.materials.length, t: window.__scene.textures.length } };
});
console.log("dedupe      :", JSON.stringify(dedupe));

// Materials are shared by name *and* transparency, and kit_materials.json can
// override that transparency. Two things to hold:
//
//  - M_Decal_White is MASK in thirty-one files and OPAQUE in twenty-six, with
//    no override, so it must stay two materials. Keyed by name alone whichever
//    module loaded first decided it for everything, and the palette's own
//    cache - filled in a different order - could disagree with the ship about
//    the very same module.
//  - M_Glass is BLEND in two and OPAQUE in twelve, but the kit's own shaders
//    make it see-through and the .gltf export flattened that, so the override
//    settles it. Once it has, both spellings are the same material again and
//    collapse back to one copy.
const glass = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const kit = await import("/js/kit.js");
  const V = BABYLON.Vector3;
  const seen = {};
  for (const [tag, id] of [["blend", "Modular SciFi MegaKit/Walls/TopWindow_Corner_Curve_Inner"],
    ["opaque", "Modular SciFi MegaKit/Walls/WallWindow_Straight"]]) {
    const e = await ed.placeAt(id, new V(tag === "blend" ? 40 : 48, 0, 40));
    for (const m of e.node.getChildMeshes()) {
      const src = m.sourceMesh || m;
      if (src.material?.name === "M_Glass") {
        seen[tag] = { alpha: +src.material.alpha.toFixed(3),
          mode: src.material.transparencyMode, uid: src.material.uniqueId,
          ior: src.material.indexOfRefraction,
          albedo: src.material.albedoColor?.asArray().map((v) => +v.toFixed(3)) };
      }
    }
  }
  const names = new Map();
  for (const [, m] of kit.materialRegistry) names.set(m.name, (names.get(m.name) || 0) + 1);
  const want = kit.getKitMaterials()?.transparency?.M_Glass;
  return { seen, want, materials: kit.materialRegistry.size, names: names.size,
    glassCopies: names.get("M_Glass") || 0 };
});
console.log("glass       :", JSON.stringify(glass));
const both = glass.seen.blend && glass.seen.opaque;
const tinted = both && glass.want.tint
  && glass.seen.blend.albedo.every((v, i) => Math.abs(v - glass.want.tint[i]) < 0.002);
const glassBad = !both
  || glass.seen.blend.alpha !== glass.want.alpha
  || glass.seen.opaque.alpha !== glass.want.alpha
  || glass.seen.blend.mode !== 2 || !tinted
  // an ior of 1 is what takes the white sheen off the pane
  || (glass.want.ior !== undefined && glass.seen.blend.ior !== glass.want.ior)
  // the override settles the difference, so the two spellings share one copy
  || glass.seen.blend.uid !== glass.seen.opaque.uid || glass.glassCopies !== 1;
// and the sharing still has to be doing its job
const sharingBad = glass.materials > glass.names * 2;
if (glassBad || sharingBad) {
  await browser.close();
  throw new Error(glassBad
    ? `glass must take the authored alpha and share one copy: ${JSON.stringify(glass)}`
    : `materials stopped being shared: ${glass.materials} over ${glass.names} names`);
}

const man = await page.evaluate(async () => {
  const m = await import("/js/manifest.js");
  const d = m.buildManifest();
  return { chunks: d.chunks.length, instances: d.instances.length, sample: d.instances[0] };
}).catch((e) => ({ error: String(e) }));
console.log("manifest    :", JSON.stringify(man));

// doors and derived portals
const markers = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const mf = await import("/js/manifest.js");

  ed.addChunk("CH01_CorridorA");
  // push a couple of placements into the second chunk so it has a volume
  const ids = [...ed.state.placements.keys()].slice(0, 3);
  ed.select(ids);
  ed.assignSelectionToChunk("CH01_CorridorA");
  ed.select([]);

  const leafIds = [...ed.state.placements.keys()].slice(3, 5);
  ed.select(leafIds);
  const door = mk.doorFromSelection();

  const m = mf.buildManifest();
  return {
    doorId: door?.id,
    doorLeaves: door?.leaves.length,
    doors: m.doors.length,
    portals: m.portals.length,
    portalCorners: m.portals[0]?.corners.length,
    portalChunks: [m.portals[0]?.chunkA, m.portals[0]?.chunkB],
    adjacency: m.adjacency,
    fluidSim: m.fluidSim,
    markers: m.markers.length,
  };
}).catch((e) => ({ error: String(e) }));
console.log("markers     :", JSON.stringify(markers));

const markerRound = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  await mf.saveLayout();
  const before = JSON.stringify(ed.serialize());
  ed.clearAll();
  await mf.loadLayout();
  return {
    same: before === JSON.stringify(ed.serialize()),
    markers: ed.state.markers.size,
    placements: ed.state.placements.size,
  };
}).catch((e) => ({ error: String(e) }));
console.log("marker rt   :", JSON.stringify(markerRound));

const roundTrip = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const m = await import("/js/manifest.js");
  await m.saveLayout();
  const before = ed.serialize();
  ed.clearAll();
  const loaded = await m.loadLayout();
  const after = ed.serialize();
  return {
    loaded: !!loaded,
    same: JSON.stringify(before.instances) === JSON.stringify(after.instances),
    n: after.instances.length,
  };
}).catch((e) => ({ error: String(e) }));
console.log("round-trip  :", JSON.stringify(roundTrip));

// every save must move the previous manifest aside, so nothing is ever lost
const rotation = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const m = await import("/js/manifest.js");
  const first = await m.saveLayout();
  const before = JSON.stringify(m.buildManifest().instances);
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
    new BABYLON.Vector3(40, 0, 40), { silent: true });
  await new Promise((r) => setTimeout(r, 1100));      // distinct second
  const second = await m.saveLayout();
  return { firstPrev: first.previous, secondPrev: second.previous, before };
});
const exportDir = process.env.SHIP_EXPORT_DIR || "D:/alexis/TombRaider/Popov72/SciFiShip/export";
const backupPath = exportDir + "/" + rotation.secondPrev;
const backup = JSON.parse(fs.readFileSync(backupPath, "utf8"));
console.log("rotation    :", JSON.stringify({
  previous: rotation.secondPrev,
  holdsPriorState: JSON.stringify(backup.instances) === rotation.before,
}));

// Thumbnails are the slow path: confirm the first tiles actually get pixels.
const thumbs = await page.evaluate(async () => {
  const t0 = performance.now();
  const ok = await new Promise((resolve) => {
    const check = () => {
      const imgs = [...document.querySelectorAll("#palette-list .item img")].slice(0, 12);
      const done = imgs.filter((i) => i.naturalWidth > 0).length;
      if (done >= 8 || performance.now() - t0 > 90000) return resolve(done);
      setTimeout(check, 500);
    };
    check();
  });
  return { withPixels: ok, elapsedMs: Math.round(performance.now() - t0) };
});
console.log("thumbnails  :", JSON.stringify(thumbs));

// Beside this file, not beside whatever directory it was launched from: the
// .gitignore covers `tool/test/shot-*.png`, so a run from anywhere else used to
// drop an untracked screenshot wherever it landed - which is how one ended up
// in the repository root. `URL` above shadows the global constructor, so the
// path comes from HERE rather than `new URL(..., import.meta.url)`.
await page.screenshot({ path: path.join(HERE, "shot-app.png") });

const glb = await page.evaluate(async () => {
  const m = await import("/js/manifest.js");
  try { return await m.exportGlb(); } catch (e) { return { error: String(e) }; }
});
console.log("export      :", JSON.stringify(glb));

// the editor forces every material double-sided; the export must not inherit it
const cullingRestored = await page.evaluate(() =>
  window.__scene.materials.filter((m) => m.backFaceCulling).map((m) => m.name));
console.log("culling after export:", cullingRestored.length === 0
  ? "all back to editor two-sided (ok)" : cullingRestored.join(", "));

console.log("\n--- console errors ---");
console.log(errors.length ? [...new Set(errors)].join("\n") : "(none)");

await browser.close();
