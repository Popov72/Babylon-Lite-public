// Headless smoke test for the layout tool. Run with:
//   node test/smoke.mjs
// Uses the Playwright already present in the Babylon.js checkout and drives
// installed Edge, so nothing new has to be downloaded.

import { createRequire } from "node:module";
import fs from "node:fs";

const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const URL = process.env.TOOL_URL || "http://localhost:5180/";

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
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new BABYLON.Vector3(0, 0, 0));
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new BABYLON.Vector3(4, 0, 0));
  await ed.placeAt("Platforms/Door_Frame_A", new BABYLON.Vector3(8, 0, 0));
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
    await ed.placeAt("Walls/ShortWall_Band2_Straight", new BABYLON.Vector3(i * 4, 0, 8));
  }
  return { before, after: { m: window.__scene.materials.length, t: window.__scene.textures.length } };
});
console.log("dedupe      :", JSON.stringify(dedupe));

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
  await ed.placeAt("Walls/ShortWall_Band2_Straight",
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

await page.screenshot({ path: "test/shot-app.png" });

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
