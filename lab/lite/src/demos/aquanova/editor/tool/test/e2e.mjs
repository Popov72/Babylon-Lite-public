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

// ---- every exported node says which instance it is -------------------------
// A name is not an identity: two placements may carry the same one, so a
// runtime matching by name cannot tell which Havok body belongs to which mesh -
// and taking a mesh out of the scene, which is what liquefying one does, would
// be a guess as to which shape to drop with it. The placement id goes into the
// glTF node's `extras`, which Babylon and Babylon-Lite both hand back at
// `metadata.gltf.extras`.
const glbFailures = [];
{
  const buf = fs.readFileSync(path.join(EXPORT, "ship.glb"));
  const magic = buf.readUInt32LE(0) === 0x46546c67;
  const json = magic
    ? JSON.parse(buf.toString("utf8", 20, 20 + buf.readUInt32LE(12)))
    : { nodes: [] };
  const man = JSON.parse(fs.readFileSync(path.join(EXPORT, "ship_manifest.json"), "utf8"));
  const byId = new Map((man.instances || []).map((i) => [i.id, i]));
  const tagged = (json.nodes || []).filter((n) => n.extras?.id);
  const ids = tagged.map((n) => n.extras.id);
  const resolvable = tagged.filter((n) => {
    const inst = byId.get(n.extras.id);
    return inst && inst.module === n.extras.module && inst.chunk === n.extras.chunk;
  });
  console.log(`glb extras   : ${tagged.length} node(s) tagged,`
    + ` ${new Set(ids).size} distinct id(s), ${resolvable.length} resolve to an instance`);
  if (!magic) glbFailures.push("ship.glb is not a glb");
  if (!tagged.length) glbFailures.push("no exported node carries an id in extras");
  if (new Set(ids).size !== ids.length) glbFailures.push("two nodes share an id");
  if (resolvable.length !== tagged.length) {
    glbFailures.push(`${tagged.length - resolvable.length} node(s) name an instance`
      + " the manifest does not have");
  }
  if (glbFailures.length) {
    console.log("GLB EXTRAS BROKEN:", glbFailures.join(" | "));
    errors.push("glb extras: " + glbFailures.join(" | "));
  }
}

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

// ---- a save made from the bench belongs to both sides ----------------------
// One camera serves two rooms. `view: serializeView()` read whichever room was
// on screen, so saving without closing the collision bench wrote the *bench's*
// viewpoint into the ship's `view` - and a stale one into `stageView`, and last
// session's roster into `stageLayout`. Three records, one mistake: reading the
// live thing when the live thing is the other side's.
const benchSave = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const m = await import("/js/manifest.js");
  const aim = (pos, rot) => {
    const c = ed.state.camera;
    c.cameraDirection.setAll(0); c.cameraRotation.set(0, 0);
    c.position.set(...pos); c.rotation.set(...rot);
  };
  aim([61, 17, -43], [0.31, -1.07, 0]);
  const ship = ed.serializeView();
  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  aim([-9, 4, -12], [0.19, 0.88, 0]);
  const bench = ed.serializeView();
  const pick = [...(kit.getCatalogue()?.byId?.keys() || [])].find((id) =>
    ![...ed.state.placements.values()].some((e) => e.stage && e.module === id));
  if (pick) await co.stageModule(pick, kit.instantiate, kit.moduleBounds, [60, 0, 60]);
  const manifest = m.buildManifest();
  const roster = ed.hooks.stageLayoutNow().map((s) => s.module);
  const stageView = ed.hooks.stageViewpoint();
  co.exitCollisionMode();
  return { ship, bench, staged: pick, roster, stageView, view: manifest.view };
});
const sameView = (a, b) => a && b
  && JSON.stringify(a.position) === JSON.stringify(b.position)
  && JSON.stringify(a.rotation) === JSON.stringify(b.rotation);
console.log("bench save   :", JSON.stringify({
  ship: benchSave.ship.position, wrote: benchSave.view?.position,
  bench: benchSave.bench.position, stageView: benchSave.stageView?.position,
}));
const benchFailures = [];
if (!sameView(benchSave.view, benchSave.ship)) {
  benchFailures.push(`the ship's view came out as ${JSON.stringify(benchSave.view?.position)}`
    + `, not ${JSON.stringify(benchSave.ship.position)}`);
}
if (!sameView(benchSave.stageView, benchSave.bench)) {
  benchFailures.push(`the bench's view came out as ${JSON.stringify(benchSave.stageView?.position)}`
    + `, not ${JSON.stringify(benchSave.bench.position)}`);
}
if (benchSave.staged && !benchSave.roster.includes(benchSave.staged)) {
  benchFailures.push(`the roster is stale: ${benchSave.staged} was staged and is missing`);
}
if (benchFailures.length) {
  console.log("BENCH SAVE BROKEN:", benchFailures.join(" | "));
  errors.push("bench save: " + benchFailures.join(" | "));
}

console.log("\nerrors:", errors.length ? [...new Set(errors)].join("\n") : "(none)");
await browser.close();
// Losing the ability to recover work - or the viewpoint you saved from - is
// worth failing the run over.
if (rotationFailures.length || benchFailures.length || glbFailures.length) process.exit(1);
