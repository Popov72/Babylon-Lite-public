// End-to-end check from a clean export/ directory:
// the tool must ignore the Blender-era manifest, and must preserve both
// Blender artefacts the first time it writes over them.

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const EXPORT = process.env.SHIP_EXPORT_DIR || "D:/alexis/TombRaider/Popov72/SciFiShip/export";
const before = Object.fromEntries(fs.readdirSync(EXPORT)
  .map((f) => [f, fs.statSync(path.join(EXPORT, f)).size]));
console.log("export before:", JSON.stringify(before));

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

await page.goto(process.env.TOOL_URL || "http://localhost:5180/", { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => document.querySelectorAll("#palette-list .item").length > 0,
  null, { timeout: 60000 });
await page.waitForTimeout(2500);

console.log("boot status  :", await page.textContent("#status-text"));

// Build a two-chunk slice with a door, the way a user would.
const built = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const V = BABYLON.Vector3;

  ed.addChunk("CH01_CorridorA");

  const floor = "Platforms/Floor_Metal_4x4";
  const wall = "Walls/ShortWall_Band2_Straight";
  const cat = (await import("/js/kit.js")).getCatalogue();
  const pick = (want, fallbackCat) =>
    cat.byId.has(want) ? want
      : cat.categories.find((c) => c.name === fallbackCat).modules[0].id;

  const F = pick(floor, "Platforms");
  const W = pick(wall, "Walls");

  ed.state.activeChunk = "CH00_Storage";
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      await ed.placeAt(F, new V(i * 4, 0, j * 4), { silent: true });
    }
  }
  await ed.placeAt(W, new V(0, 0, -4), { silent: true });

  // leave a gap at x = 8 for the doorway so the two chunk volumes stay disjoint
  ed.state.activeChunk = "CH01_CorridorA";
  for (let i = 3; i < 6; i++) await ed.placeAt(F, new V(i * 4, 0, 0), { silent: true });

  ed.state.activeChunk = "CH00_Storage";
  const doorLeaf = await ed.placeAt(W, new V(8, 0, 0), { silent: true });
  ed.select([doorLeaf.id]);
  const door = mk.doorFromSelection();

  ed.select([]);

  // A non-default ship-wide constant, so the round-trip check below has teeth:
  // buildManifest() assembles its own object rather than writing serialize()'s,
  // so anything it forgets to copy is lost on save and silently comes back on
  // its default. Leaving this at 0.008 would not have caught that.
  ed.setConfig("shellThickness", 0.0075);

  return { placements: ed.state.placements.size, markers: ed.state.markers.size, door: door?.id };
});
console.log("built        :", JSON.stringify(built));

const saved = await page.evaluate(async () => {
  const m = await import("/js/manifest.js");
  const s = await m.saveLayout();
  const g = await m.exportGlb();
  return { manifest: s.bytes, glb: g.bytes };
});
console.log("written      :", JSON.stringify(saved));

const reload = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const m = await import("/js/manifest.js");
  const a = JSON.stringify(ed.serialize());
  ed.clearAll();
  const d = await m.loadLayout();
  return {
    identical: a === JSON.stringify(ed.serialize()),
    chunks: d.chunks.map((c) => c.id),
    portals: d.portals.map((p) => `${p.chunkA}->${p.chunkB}`),
    fluidSim: d.fluidSim,
    environment: d.environment,
    config: d.config,
    checks: document.getElementById("validation").textContent.trim(),
  };
});
console.log("reload       :", JSON.stringify(reload));

await page.evaluate(() => document.getElementById("btn-focus").click());
await page.waitForTimeout(1200);
await page.screenshot({ path: "test/shot-final.png" });

const after = Object.fromEntries(fs.readdirSync(EXPORT)
  .map((f) => [f, fs.statSync(path.join(EXPORT, f)).size]));
console.log("export after :", JSON.stringify(after));
console.log("blender glb preserved:",
  after["ship.blender.bak.glb"] === before["ship.glb"]);
console.log("blender manifest preserved:",
  after["ship_manifest.blender.bak.json"] === before["ship_manifest.json"]);

// ---- every write keeps what it replaced -----------------------------------
// The manifest, the collision file and the auto-save all rotate a timestamped
// copy. The collision file was briefly exempted on the reasoning that every
// manifest carries the same hulls in `moduleShapes` - but that argument is only
// as good as its source, and when the file was overwritten with nothing the
// manifest had been emptied in the same breath. The timestamped copies were the
// only thing that got the work back. This is here so nobody removes them again.
const rotationFailures = [];
for (const [route, stem] of [["collision", "ship_collision"], ["autosave", "ship_autosave"]]) {
  const body = (n) => JSON.stringify({ probe: stem, n });
  const post = async (n) => page.evaluate(async ([r, b]) =>
    (await fetch(`/api/${r}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: b,
    })).json(), [route, body(n)]);
  const first = await post(1);
  await new Promise((r) => setTimeout(r, 1100));   // the stamp is per second
  const second = await post(2);
  const backups = fs.readdirSync(EXPORT)
    .filter((f) => f.startsWith(`${stem}.`) && f !== `${stem}.json`);
  const kept = backups.some((f) =>
    JSON.parse(fs.readFileSync(path.join(EXPORT, f), "utf8")).n === 1);
  console.log(`rotation ${stem}: ${backups.length} backup(s), first write kept: ${kept}`);
  if (!second.previous || !kept) {
    rotationFailures.push(`${stem}: previous=${second.previous}, first write kept=${kept}`);
  }
  void first;
}
if (rotationFailures.length) {
  console.log("ROTATION BROKEN:", rotationFailures.join(" | "));
  errors.push("rotation: " + rotationFailures.join(" | "));
}

console.log("\nerrors:", errors.length ? [...new Set(errors)].join("\n") : "(none)");
await browser.close();
// Losing the ability to recover work is worth failing the run over.
if (rotationFailures.length) process.exit(1);
