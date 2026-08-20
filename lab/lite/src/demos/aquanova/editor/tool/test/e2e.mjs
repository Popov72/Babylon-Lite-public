// End-to-end check from a clean export/ directory: build a two-room slice the
// way a user would, save it, reload it, and then look at the whole runtime
// pipeline - the probe capture and the Runtime preview - through the editor.

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { toolUrl } from "./target.mjs";

const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const HERE = path.dirname(fileURLToPath(import.meta.url));
const URL = toolUrl();
const EXPORT = process.env.SHIP_EXPORT_DIR || "D:/alexis/TombRaider/Popov72/SciFiShip/export";
const before = Object.fromEntries(fs.readdirSync(EXPORT).map((f) => [f, fs.statSync(path.join(EXPORT, f)).size]));
console.log("export before:", JSON.stringify(before));

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
// `errors` is the printed log, and it also collects whatever the page logs to
// console.error - CDN noise included - so it is deliberately not fatal on its
// own. Blocks that assert something worth failing over push into their own
// array and add it to the exit condition at the bottom. `assertFailures` is
// the shared one for blocks that had no array of their own, which is how a
// whole block of preview checks came to print failures and still let the run
// exit 0 - a preview bug shipped straight through it.
const assertFailures = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});

await page.goto(URL, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => document.querySelectorAll("#palette-list .item").length > 0, null, { timeout: 60000 });
await page.waitForTimeout(2500);

console.log("boot status  :", await page.textContent("#status-text"));

// Build a two-chunk slice with a door, the way a user would.
const built = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const V = BABYLON.Vector3;

  ed.addChunk("CH01_CorridorA");

  const floor = "Modular SciFi MegaKit/Platforms/Floor_Metal_4x4";
  const wall = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  const cat = (await import("/js/kit.js")).getCatalogue();
  const pick = (want, fallbackCat) => (cat.byId.has(want) ? want : cat.categories.find((c) => c.name === fallbackCat).modules[0].id);

  const F = pick(floor, "Platforms");
  const W = pick(wall, "Walls");

  ed.state.activeChunk = "CH00_Storage";
  let storageFloor = null;
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const tile = await ed.placeAt(F, new V(i * 4, 0, j * 4), { silent: true });
      storageFloor ??= tile;
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

  // A prop the runtime moves, so the manifest has a dynamic element in it.
  ed.setBehaviorDef("SlidingLeaf", { dynamic: true, axis: "x", distance: 2 });
  ed.addEntityBehavior(doorLeaf.node.name, "SlidingLeaf");
  ed.state.activeChunk = "CH01_CorridorA";
  const dynamicCorridorProp = await ed.placeAt("Modular SciFi MegaKit/Props/Prop_Light_Small", new V(16, 2, -4), { silent: true });
  ed.addEntityBehavior(dynamicCorridorProp.node.name, "SlidingLeaf");
  ed.state.activeChunk = "CH00_Storage";

  // A real kit lamp, whose M_Light material genuinely emits and which seeds its
  // own authored light from kit_lights.json.
  const lamp = await ed.placeAt("Modular SciFi MegaKit/Props/Prop_Light_Small", new V(4, 2, -4), { silent: true });

  // A window onto space in the corridor, so there is a portal to find. The far
  // side is deliberately the skybox.
  mk.addDoor(new V(16, 0, -2), {
    chunkA: "CH01_CorridorA",
    chunkB: ed.SKYBOX_CHUNK,
    silent: true,
  });

  // A lamp in each chunk, so the glb check below has both a light to find and a
  // second one to tell it apart from - and one of them is switched off, which is
  // the shape that creates no Babylon light at all.
  const lt = await import("/js/lights.js");
  const rooms = [...ed.state.placements.values()];
  // The storage lamps hang off a tile THIS fixture placed, not off whatever
  // happens to be first in the map: e2e shares its export folder with the
  // suites before it, so it boots on a restored ship and `rooms[0]` is not
  // reproducible. The probe orientation check downstream reads how the room
  // leans about the horizon, and that only means something if the lamps are in
  // a known place relative to the probe.
  const inA = storageFloor ?? rooms.find((p) => p.chunk === "CH00_Storage");
  const inB = rooms.find((p) => p.chunk === "CH01_CorridorA");
  lt.addLight(inA.id, {
    offset: [0, 2.5, 0],
    // Bright, and reaching the whole room: the probe orientation check below
    // reads which way each bearing leans, and a room lit to 1e-5 quantises to
    // nothing in a 16-pixel cubemap face.
    runtime: { type: "spot", clustered: false, angle: 70, intensity: 15, range: 20 },
    silent: true,
  });
  lt.addLight(inA.id, {
    offset: [0, 1.5, 0],
    runtime: { type: "point", clustered: false, intensity: 15, range: 20 },
    silent: true,
  });
  lt.addLight(inB.id, {
    offset: [0, 2, 0],
    runtime: { type: "none" },
    silent: true,
  });
  lt.addLight(inB.id, {
    offset: [0, 1.5, 0],
    runtime: { type: "point", clustered: false },
    silent: true,
  });

  // Reflection probes are independent from the two chunks. The face size is a
  // ship-wide setting rather than a probe field - the runtime keeps them in one
  // cube texture array - so it is set below, at the smallest size offered, to
  // keep the real capture quick.
  ed.setEnvironmentProbe("ENV0001", {
    boxPosition: [2, 2, 0],
    boxSize: [12, 6, 12],
    capturePosition: [2, 2, 0],
  });
  ed.setEnvironmentProbe("ENV0002", {
    boxPosition: [16, 2, 0],
    boxSize: [12, 6, 12],
    capturePosition: [15, 2.5, 0],
  });

  // A non-default ship-wide constant, so the round-trip check below has teeth:
  // buildManifest() assembles its own object rather than writing serialize()'s,
  // so anything it forgets to copy is lost on save and silently comes back on
  // its default. Leaving this at 0.008 would not have caught that.
  ed.setConfig("shellThickness", 0.0075);
  ed.setConfig("probeResolution", 128);

  return {
    placements: ed.state.placements.size,
    markers: ed.state.markers.size,
    door: door?.id,
    lights: ed.state.lights.size,
    lamp: lamp?.id,
    moving: doorLeaf.node.name,
    probes: ed.state.environmentProbes.size,
  };
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
// Beside this file, not beside the launch directory - see the note in
// smoke.mjs. `URL` above shadows the global constructor, hence HERE.
await page.screenshot({ path: path.join(HERE, "shot-final.png") });

const after = Object.fromEntries(fs.readdirSync(EXPORT).map((f) => [f, fs.statSync(path.join(EXPORT, f)).size]));
console.log("export after :", JSON.stringify(after));

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
  const json = magic ? JSON.parse(buf.toString("utf8", 20, 20 + buf.readUInt32LE(12))) : { nodes: [] };
  const man = JSON.parse(fs.readFileSync(path.join(EXPORT, "ship_manifest.json"), "utf8"));
  const byId = new Map((man.instances || []).map((i) => [i.id, i]));
  const tagged = (json.nodes || []).filter((n) => n.extras?.id && n.extras.kind !== "light");
  const ids = tagged.map((n) => n.extras.id);
  const resolvable = tagged.filter((n) => {
    const inst = byId.get(n.extras.id);
    return inst && inst.module === n.extras.module && inst.chunk === n.extras.chunk;
  });
  console.log(`glb extras   : ${tagged.length} node(s) tagged,` + ` ${new Set(ids).size} distinct id(s), ${resolvable.length} resolve to an instance`);
  if (!magic) glbFailures.push("ship.glb is not a glb");
  if (!tagged.length) glbFailures.push("no exported node carries an id in extras");
  if (new Set(ids).size !== ids.length) glbFailures.push("two nodes share an id");
  if (resolvable.length !== tagged.length) {
    glbFailures.push(`${tagged.length - resolvable.length} node(s) name an instance` + " the manifest does not have");
  }

  // ---- lights leave as bare nodes, whole ----------------------------------
  // Nothing is drawn for a light: the runtime builds a Babylon light from
  // `extras` alone, so a half-written extras block is a lamp that is quietly
  // wrong rather than one that fails to load. The node also has to be a CHILD
  // of the element it rides, which is what carries the authored offset across
  // in the space it was authored in.
  const nodes = json.nodes || [];
  const lightNodes = nodes.filter((n) => n.extras?.kind === "light");
  const parentOf = new Map();
  nodes.forEach((n, i) => (n.children || []).forEach((c) => parentOf.set(c, i)));
  const whole = lightNodes.filter((n) => {
    const r = n.extras.runtime;
    return (
      r &&
      typeof r.type === "string" &&
      typeof r.intensity === "number" &&
      typeof r.range === "number" &&
      typeof r.angle === "number" &&
      Array.isArray(r.color) &&
      typeof r.clustered === "boolean" &&
      typeof r.castsShadows === "boolean"
    );
  });
  const ridden = lightNodes.filter((n) => {
    const p = nodes[parentOf.get(nodes.indexOf(n))];
    return p && p.extras?.id === n.extras.owner;
  });
  const named = lightNodes.filter((n) => n.name === `LIGHT_${n.extras.id}`);
  // The offset is carried by the node's own translation, not by the extras -
  // so a node that arrived with its record intact and its transform dropped
  // would light every lamp at the foot of the wall it belongs to. Checked
  // against what the manifest says was authored, in glTF space: the exporter
  // mirrors X, so the manifest's own conversion has to hold here too.
  const authored = new Map((man.lights || []).map((l) => [l.id, l.offset]));
  const placed = lightNodes.filter((n) => {
    const o = authored.get(n.extras.id);
    const t = n.translation || [0, 0, 0];
    return o && [-o[0], o[1], o[2]].every((v, i) => Math.abs(v - (t[i] || 0)) < 1e-4);
  });
  const gizmos = nodes.filter((n) => /_gizmo$|_stub$/.test(n.name || ""));
  console.log(
    `glb lights   : ${lightNodes.length} node(s), ${whole.length} carry a whole record,` +
      ` ${ridden.length} hang off the element they ride,` +
      ` types: ${lightNodes.map((n) => n.extras.runtime?.type).join("+") || "(none)"},` +
      ` at y ${lightNodes.map((n) => n.translation?.[1]).join("+")}`
  );
  if (lightNodes.length !== built.lights) {
    glbFailures.push(`${built.lights} light(s) authored but ${lightNodes.length} exported`);
  }
  if (whole.length !== lightNodes.length) {
    glbFailures.push(`${lightNodes.length - whole.length} light node(s) lost part of their record`);
  }
  if (ridden.length !== lightNodes.length) {
    glbFailures.push(`${lightNodes.length - ridden.length} light node(s) are not children` + " of the element they ride");
  }
  if (named.length !== lightNodes.length) glbFailures.push("a light node is not named LIGHT_<id>");
  if (placed.length !== lightNodes.length) {
    glbFailures.push(`${lightNodes.length - placed.length} light node(s) lost their offset`);
  }
  if (gizmos.length) glbFailures.push(`${gizmos.length} gizmo mesh(es) reached the glb`);
  // ---- the glass never asks for transmission -------------------------------
  // A window here is a green base colour and an alpha. `KHR_materials_ior` is
  // welcome - it is what takes the white sheen off the pane - but
  // `KHR_materials_transmission` is not: Babylon-Lite's refraction path
  // retargets the scene to an offscreen HDR buffer, which hides the fluid
  // surface that composites after it, and aquanova/main.ts already has to strip
  // the sub-feature at load time to get the water back. Nothing in the editor
  // sets `subSurface.isRefractionEnabled`, so the serializer has no reason to
  // emit it - this is here to catch the day something does.
  const glass = (json.materials || []).filter((m) => /glass/i.test(m.name || ""));
  const transmissive = glass.filter((m) => m.extensions?.KHR_materials_transmission);
  const green = glass.filter((m) => {
    const c = m.pbrMetallicRoughness?.baseColorFactor;
    return c && c[1] > c[0] && c[1] > c[2] && c[3] < 1 && m.alphaMode === "BLEND";
  });
  const exts = [...new Set(glass.flatMap((m) => Object.keys(m.extensions || {})))];
  console.log(`glb glass    : ${glass.length} material(s), green and see-through:` + ` ${green.length}, extensions: ${exts.join("+") || "(none)"}`);
  if (transmissive.length) {
    glbFailures.push(`${transmissive.length} glass material(s) ask for KHR_materials_transmission`);
  }
  if ((json.extensionsUsed || []).includes("KHR_materials_transmission")) {
    glbFailures.push("the glb declares KHR_materials_transmission in extensionsUsed");
  }
  if (glass.length && green.length !== glass.length) {
    glbFailures.push(`${glass.length - green.length} glass material(s) not green and blended`);
  }

  if (glbFailures.length) {
    console.log("EXPORTED GLB BROKEN:", glbFailures.join(" | "));
    errors.push("exported glb: " + glbFailures.join(" | "));
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
for (const [route, stem] of [
  ["collision", "ship_collision"],
  ["autosave", "ship_autosave"],
]) {
  const body = (n) => JSON.stringify({ probe: stem, n });
  const post = async (n) =>
    page.evaluate(
      async ([r, b]) =>
        (
          await fetch(`/api/${r}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: b,
          })
        ).json(),
      [route, body(n)]
    );
  const first = await post(1);
  await new Promise((r) => setTimeout(r, 1100)); // the stamp is per second
  const second = await post(2);
  const backups = fs.readdirSync(EXPORT).filter((f) => f.startsWith(`${stem}.`) && f !== `${stem}.json`);
  const kept = backups.some((f) => JSON.parse(fs.readFileSync(path.join(EXPORT, f), "utf8")).n === 1);
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
    c.cameraDirection.setAll(0);
    c.cameraRotation.set(0, 0);
    c.position.set(...pos);
    c.rotation.set(...rot);
  };
  aim([61, 17, -43], [0.31, -1.07, 0]);
  const ship = ed.serializeView();
  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  aim([-9, 4, -12], [0.19, 0.88, 0]);
  const bench = ed.serializeView();
  const pick = [...(kit.getCatalogue()?.byId?.keys() || [])].find((id) => ![...ed.state.placements.values()].some((e) => e.stage && e.module === id));
  if (pick) await co.stageModule(pick, kit.instantiate, kit.moduleBounds, [60, 0, 60]);
  const manifest = m.buildManifest();
  const roster = ed.hooks.stageLayoutNow().map((s) => s.module);
  const stageView = ed.hooks.stageViewpoint();
  co.exitCollisionMode();
  return { ship, bench, staged: pick, roster, stageView, view: manifest.view };
});
const sameView = (a, b) => a && b && JSON.stringify(a.position) === JSON.stringify(b.position) && JSON.stringify(a.rotation) === JSON.stringify(b.rotation);
console.log(
  "bench save   :",
  JSON.stringify({
    ship: benchSave.ship.position,
    wrote: benchSave.view?.position,
    bench: benchSave.bench.position,
    stageView: benchSave.stageView?.position,
  })
);
const benchFailures = [];
if (!sameView(benchSave.view, benchSave.ship)) {
  benchFailures.push(`the ship's view came out as ${JSON.stringify(benchSave.view?.position)}` + `, not ${JSON.stringify(benchSave.ship.position)}`);
}
if (!sameView(benchSave.stageView, benchSave.bench)) {
  benchFailures.push(`the bench's view came out as ${JSON.stringify(benchSave.stageView?.position)}` + `, not ${JSON.stringify(benchSave.bench.position)}`);
}
if (benchSave.staged && !benchSave.roster.includes(benchSave.staged)) {
  benchFailures.push(`the roster is stale: ${benchSave.staged} was staged and is missing`);
}
if (benchFailures.length) {
  console.log("BENCH SAVE BROKEN:", benchFailures.join(" | "));
  errors.push("bench save: " + benchFailures.join(" | "));
}

// ---- the manifest says which space each of its halves is in -----------------
// Runtime-facing fields are glTF space, matching ship.glb; tool-facing ones are
// editor space, because they exist to rebuild the editor. Compose a glTF-space
// module hull onto an editor-space instance and the collider lands on the wrong
// side of the prop, still looking plausible - so the split is named in the file
// rather than left in a source comment no runtime will ever read.
//
// This check is about *coverage*: a new top-level key cannot ship without
// someone saying which space it is in.
const spaceFailures = [];
{
  const man = JSON.parse(fs.readFileSync(path.join(EXPORT, "ship_manifest.json"), "utf8"));
  const sp = man.space;
  if (!sp) {
    spaceFailures.push("the manifest does not say which space its fields are in");
  } else {
    const declared = new Set([...(sp.gltf || []), ...(sp.editor || []), ...(sp.none || [])].map((k) => k.split("[")[0]));
    const undeclared = Object.keys(man).filter((k) => !declared.has(k));
    const phantom = [...declared].filter((k) => !(k in man));
    console.log(`manifest space: ${declared.size} field(s) declared,` + ` ${undeclared.length} undeclared, ${phantom.length} phantom`);
    if (undeclared.length) spaceFailures.push(`undeclared: ${undeclared.join(", ")}`);
    if (phantom.length) spaceFailures.push(`declared but absent: ${phantom.join(", ")}`);
    if (!sp.convert?.quaternion) spaceFailures.push("the conversion rule omits the quaternion");
    // The two that actually caught someone out.
    if (!(sp.editor || []).includes("instances")) {
      spaceFailures.push("instances is not declared as editor space");
    }
    if (!(sp.gltf || []).includes("moduleCollision")) {
      spaceFailures.push("moduleCollision is not declared as glTF space");
    }
  }
  if (spaceFailures.length) {
    console.log("MANIFEST SPACE BROKEN:", spaceFailures.join(" | "));
    errors.push("manifest space: " + spaceFailures.join(" | "));
  }
}

// ---- the probes are captured from the editor, not from a terminal ----------
// Six renders and a prefilter per probe, driven from the page: what is under
// test is the whole round trip - declare, render, upload, serve - at the
// smallest cubemap size the editor offers, because it is the wiring that rots,
// not the filtering.
const probeFailures = [];
{
  const capture = await page.evaluate(async () => {
    const m = await import("/js/main.js");
    const before = await (await fetch("/api/local-environments")).json();
    const first = await m.captureProbes();
    const after = await (await fetch("/api/local-environments")).json();
    // Nothing has moved since, so the second pass must find every probe's
    // stamp still matching and capture none of them. That is the whole point
    // of the digest: a room you have not touched is not re-rendered.
    const again = await m.captureProbes();
    // ...and a forced pass must ignore the stamps and take them all.
    const forced = await m.captureProbes(true);
    // One probe by name: re-rendered even though the forced pass just brought
    // it up to date, and - the point of the button - without disturbing the
    // rooms it did not touch, which must still read as up to date afterwards.
    const ed = await import("/js/editor.js");
    const target = [...ed.state.environmentProbes.keys()][0] || null;
    const one = target ? await m.captureProbes(false, target) : { converted: -1 };
    const afterOne = await (await fetch("/api/local-environments")).json();
    // An id nothing authored is a bug in the caller, not an empty capture.
    let rejected = "no error";
    try { await m.captureProbes(false, "not_a_probe"); }
    catch (e) { rejected = String((e && e.message) || e); }
    return {
      declaredBefore: Object.keys(before.probes || {}).length,
      pendingBefore: (before.pending || []).length,
      first,
      declared: Object.keys(after.probes || {}).length,
      pendingAfter: (after.pending || []).length,
      files: Object.values(after.probes || {}).map((p) => p.env),
      again,
      forced,
      target,
      one,
      pendingAfterOne: (afterOne.pending || []).length,
      rejected,
    };
  });
  console.log(
    `probe capture: ${capture.first.converted} captured, ${capture.first.bytes} byte(s),` +
      ` ${capture.declared} declared, ${capture.pendingAfter} still pending,` +
      ` again ${capture.again.converted}, forced ${capture.forced.converted}`
  );
  if (capture.declared !== built.probes) {
    probeFailures.push(`${built.probes} probe(s) authored but ${capture.declared} declared`);
  }
  if (capture.first.converted !== built.probes) {
    probeFailures.push(`${capture.first.converted} probe(s) captured of ${built.probes}`);
  }
  if (capture.pendingAfter !== 0) {
    probeFailures.push(`${capture.pendingAfter} probe(s) still owe a capture afterwards`);
  }
  if (capture.again.converted !== 0) {
    probeFailures.push(`an untouched ship re-captured ${capture.again.converted} probe(s)`);
  }
  if (capture.forced.converted !== built.probes) {
    probeFailures.push(`a forced capture took ${capture.forced.converted} probe(s) of ${built.probes}`);
  }
  console.log(
    `probe capture: one probe (${capture.target}) took ${capture.one.converted},` +
      ` ${capture.pendingAfterOne} pending after, unknown id → "${capture.rejected}"`
  );
  if (capture.one.converted !== 1) {
    probeFailures.push(`capturing ${capture.target} alone took ${capture.one.converted} probe(s)`);
  }
  if (capture.pendingAfterOne !== 0) {
    probeFailures.push(
      `a single-probe capture left ${capture.pendingAfterOne} other probe(s) pending`);
  }
  if (!/not an authored probe/.test(capture.rejected)) {
    probeFailures.push(`capturing an unknown probe id said "${capture.rejected}"`);
  }
  // Written where sync-ship.ts will look for them, and reachable over the same
  // server the editor is served from.
  for (const name of capture.files) {
    const onDisk = path.join(EXPORT, "environments", name);
    if (!fs.existsSync(onDisk)) probeFailures.push(`no .env at ${onDisk}`);
    else if (fs.statSync(onDisk).size < 1024) probeFailures.push(`${name} is ${fs.statSync(onDisk).size} bytes`);
  }
  const served = capture.files.length
    ? await page.evaluate(async (name) => (await fetch(`/environments/${name}`)).status, capture.files[0])
    : 0;
  if (served !== 200) probeFailures.push(`/environments/ serves ${served}, not the cubemap`);

  // What is actually IN the file. Every check above passed for months while the
  // cubemaps held nothing but zeroes: a reflection probe's target is not drawn
  // by scene.render(), so the capture wrote six black faces of the right size,
  // to the right place, with the right digest beside them. Size alone does not
  // catch it either - a black 1024 probe is still 200 KB. So load them back.
  const content = await page.evaluate(async (names) => {
    const ed = await import("/js/editor.js");
    const out = [];
    for (const name of names) {
      const texture = BABYLON.CubeTexture.CreateFromPrefilteredData(`/environments/${name}?t=${Date.now()}`, ed.state.scene);
      try {
        await new Promise((resolve, reject) => {
          const t = setTimeout(() => reject(new Error("timed out loading")), 30000);
          BABYLON.Texture.WhenAllReady([texture], () => { clearTimeout(t); resolve(); });
        });
        let max = 0;
        for (let face = 0; face < 6; face++) {
          const buf = await texture.readPixels(face, 0);
          for (let i = 0; i < buf.length; i += 4) {
            const v = Math.max(buf[i], buf[i + 1], buf[i + 2]);
            if (v > max) max = v;
          }
        }
        out.push({ name, max });
      } catch (e) {
        out.push({ name, error: String(e && e.message || e) });
      } finally {
        texture.dispose();
      }
    }
    return out;
  }, capture.files);
  const brightest = Math.max(0, ...content.map((c) => c.max || 0));
  console.log(`probe content: brightest ${brightest.toFixed(3)} across ${content.length} cubemap(s)`);
  for (const c of content) {
    if (c.error) probeFailures.push(`${c.name} would not load back: ${c.error}`);
    else if (!(c.max > 0)) probeFailures.push(`${c.name} is black`);
  }
  // Exposure and tone mapping have to stay out of a capture - a cubemap is
  // radiance, and the runtime exposes it itself. Tone mapping clamps, so when
  // it leaks in the brightest texel in the whole set lands on exactly 1. Real
  // radiance hitting 1.000000 dead on, across every face of every room, does
  // not happen; a clamp guarantees it.
  if (brightest === 1) {
    probeFailures.push("every cubemap peaks at exactly 1.0 - image processing leaked into the capture");
  }

  // Which way up. A vertically mirrored cubemap passes every check above and
  // most of the ones you would think to write: a mirror leaves the centre of
  // each face fixed, so the poles land in the right slots and every
  // axis-aligned reading still matches. Only an OFF-AXIS direction moves. It
  // does not survive eye contact either - a corridor whose lit truss is on the
  // ceiling looks perfectly plausible upside down.
  //
  // So take the readings in pairs: the same horizontal bearing tilted 45 up and
  // 45 down. Those two ARE each other's Y-mirror, so if the cubemap is upside
  // down their values simply swap. Compare which way each pair leans in the
  // cubemap against which way it leans in the live room - never the levels
  // themselves, which are on different scales because a 16-pixel face has been
  // prefiltered for roughness and the room has not. Here the fixture leans
  // hard: lit floor below the capture point, open space above it.
  //
  // The scene reading is the centre patch of a narrow-field render, symmetric
  // about the centre and so unable to carry a row-order mistake of its own; the
  // cubemap is sampled through a shader with a constant direction, so there are
  // no derivatives and no mip is chosen for us.
  const orient = await page.evaluate(async (file) => {
    const B = window.BABYLON;
    const ed = await import("/js/editor.js");
    const rt = await import("/js/runtime.js");
    const scene = ed.state.scene;
    const engine = scene.getEngine();
    const [id] = [...ed.state.environmentProbes.keys()].filter((k) => file.includes(k));
    const probe = ed.state.environmentProbes.get(id);
    const at = new B.Vector3(...probe.capturePosition);

    const cube = B.CubeTexture.CreateFromPrefilteredData(`/environments/${file}?t=${Date.now()}`, scene);
    cube.coordinatesMode = B.Texture.SKYBOX_MODE;
    await new Promise((res, rej) => {
      const t = setTimeout(() => rej(new Error("cube never loaded")), 60000);
      B.BaseTexture.WhenAllReady([cube], () => { clearTimeout(t); res(); });
    });

    const MASK = 0x10000000;
    const mat = new B.ShaderMaterial("ORIENT_mat", scene, {
      vertexSource: `precision highp float;
attribute vec3 position;
uniform mat4 worldViewProjection;
void main(void) { gl_Position = worldViewProjection * vec4(position, 1.0); }`,
      fragmentSource: `precision highp float;
uniform samplerCube env;
uniform vec3 dir;
uniform mat4 reflectionMatrix;
uniform float oppositeZ;
uniform float decodeRGBD;
void main(void) {
  vec3 d = (reflectionMatrix * vec4(normalize(dir), 0.0)).xyz;
  d.z *= oppositeZ;
  vec4 c = textureCube(env, d);
  vec3 lin = c.rgb;
  if (decodeRGBD > 0.5) lin = pow(lin, vec3(2.2)) / max(c.a, 1e-6);
  gl_FragColor = vec4(lin, 1.0);
}`,
    }, {
      attributes: ["position"],
      uniforms: ["worldViewProjection", "dir", "reflectionMatrix", "oppositeZ", "decodeRGBD"],
      samplers: ["env"],
    });
    mat.setTexture("env", cube);
    mat.setMatrix("reflectionMatrix", cube.getReflectionTextureMatrix());
    mat.setFloat("oppositeZ", (scene.useRightHandedSystem ? !cube.invertZ : cube.invertZ) ? -1 : 1);
    mat.setFloat("decodeRGBD", cube.isRGBD ? 1 : 0);
    mat.setVector3("dir", new B.Vector3(0, 0, 1));

    const quad = B.MeshBuilder.CreatePlane("ORIENT_quad", { size: 8 }, scene);
    quad.material = mat;
    quad.layerMask = MASK;
    quad.isPickable = false;
    const quadCam = new B.FreeCamera("ORIENT_quadcam", new B.Vector3(0, 0, -2), scene, false);
    quadCam.layerMask = MASK;
    quadCam.mode = B.Camera.ORTHOGRAPHIC_CAMERA;
    quadCam.orthoLeft = -1; quadCam.orthoRight = 1; quadCam.orthoTop = 1; quadCam.orthoBottom = -1;
    quadCam.setTarget(B.Vector3.Zero());
    const quadRt = new B.RenderTargetTexture("ORIENT_quadrt", 8, scene, {
      type: B.Constants.TEXTURETYPE_FLOAT, generateMipMaps: false,
      samplingMode: B.Constants.TEXTURE_NEAREST_SAMPLINGMODE,
    });
    quadRt.clearColor = new B.Color4(0, 0, 0, 1);
    quadRt.activeCamera = quadCam;
    quadRt.renderList = [quad];
    const sampleCube = async (d) => {
      mat.setVector3("dir", d);
      for (let i = 0; i < 600 && !quad.isReady(true); i++) await new Promise((r) => requestAnimationFrame(r));
      quadRt.render();
      const p = await quadRt.readPixels(0, 0, null, true, false);
      return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    };

    class FaceCam extends B.FreeCamera {
      constructor(n, p, s) { super(n, p, s, false); this._v = B.Matrix.Identity(); this._t = new B.Vector3(); }
      look(f, u) { this.position.addToRef(f, this._t); B.Matrix.LookAtLHToRef(this.position, this._t, u, this._v); }
      _getViewMatrix() { return this._v; }
      _isSynchronizedViewMatrix() { return false; }
    }

    // 30 degrees, and the shallow angle is the whole point. A texel row flip is
    // only an up/down mirror on the four SIDE faces, where a face's vertical
    // axis is the world's; on the +Y and -Y faces it mirrors world Z instead.
    // Tilt past 45 and the sample lands on a pole face, where an upside-down
    // cubemap reads exactly the same as an upright one along this pair - which
    // is how a first version of this check passed a deliberately broken build.
    const H = Math.sqrt(3) / 2, V = 0.5;
    const bearings = [["+Z", [0, 0, 1]], ["-Z", [0, 0, -1]], ["+X", [1, 0, 0]], ["-X", [-1, 0, 0]]];
    const pairs = [];
    await rt.withRuntimeCapture(async () => {
      rt.setCaptureViewpoint(at);
      const list = rt.renderListFor(rt.meshesInProbeBox(probe.boxPosition, probe.boxSize));
      const cam = new FaceCam("ORIENT_cam", at.clone(), scene);
      cam.minZ = 0.05; cam.maxZ = 1000;
      // Wide enough to average over a patch of room rather than one lucky
      // texel, narrow enough that the patch stays a cone about the bearing.
      cam.freezeProjectionMatrix(B.Matrix.PerspectiveFovLH(0.6, 1, cam.minZ, cam.maxZ, engine.isNDCHalfZRange));
      const rtt = new B.RenderTargetTexture("ORIENT_rt", 16, scene, {
        type: B.Constants.TEXTURETYPE_HALF_FLOAT, generateMipMaps: false,
        samplingMode: B.Constants.TEXTURE_BILINEAR_SAMPLINGMODE,
        enableClusteredLights: true,
      });
      rtt.gammaSpace = false;
      rtt.clearColor = new B.Color4(0, 0, 0, 1);
      rtt.activeCamera = cam;
      rtt.renderList = [...list];
      const sampleScene = async (d) => {
        cam.look(d, new B.Vector3(0, 1, 0));
        for (let i = 0; i < 600 && !rtt.renderList.every((m) => m.isReady(true)); i++) {
          await new Promise((r) => requestAnimationFrame(r));
        }
        scene.incrementRenderId();
        rtt.render();
        const buf = await rtt.readPixels(0, 0, null, true, false);
        let l = 0;
        for (let y = 4; y < 12; y++) for (let x = 4; x < 12; x++) {
          const i = (y * 16 + x) * 4;
          l += 0.2126 * buf[i] + 0.7152 * buf[i + 1] + 0.0722 * buf[i + 2];
        }
        return l / 64;
      };
      for (const [name, b] of bearings) {
        const up = new B.Vector3(b[0] * H, V, b[2] * H);
        const down = new B.Vector3(b[0] * H, -V, b[2] * H);
        pairs.push({
          name,
          sceneUp: await sampleScene(up), sceneDown: await sampleScene(down),
          cubeUp: await sampleCube(up), cubeDown: await sampleCube(down),
        });
      }
      rtt.dispose(); cam.dispose();
      rt.setCaptureViewpoint(null);
    });

    quadRt.dispose(); quad.dispose(); mat.dispose(); quadCam.dispose(); cube.dispose();
    return { id, pairs };
  }, capture.files[0]);

  // A pair only votes when both the room and the cubemap actually lean: where
  // one of them is flat about the horizon there is nothing to compare.
  const LEAN = 1.5;
  const leans = (a, b) => (a > b * LEAN ? 1 : b > a * LEAN ? -1 : 0);
  let upright = 0, mirrored = 0;
  const report = [];
  for (const p of orient.pairs) {
    const room = leans(p.sceneDown, p.sceneUp);
    const cubed = leans(p.cubeDown, p.cubeUp);
    if (room && cubed) { if (room === cubed) upright++; else mirrored++; }
    report.push(`${p.name} room ${p.sceneUp.toExponential(1)}/${p.sceneDown.toExponential(1)}` +
      ` cube ${p.cubeUp.toExponential(1)}/${p.cubeDown.toExponential(1)}`);
  }
  console.log(`probe orient : ${orient.id} upright ${upright}, mirrored ${mirrored} [${report.join(", ")}]`);
  if (upright + mirrored < 1) {
    probeFailures.push("no bearing tells the cubemap from its Y-mirror - the orientation check has lost its teeth");
  } else if (mirrored) {
    probeFailures.push(`the cubemap is upside down: ${mirrored} bearing(s) lean the opposite way to the room`);
  }

  if (probeFailures.length) {
    console.log("PROBE CAPTURE BROKEN:", probeFailures.join(" | "));
    errors.push("probe capture: " + probeFailures.join(" | "));
  }
}

// ---- what a probe must not photograph, and when it is taken ----------------
// A probe records a room's FIXED geometry. Anything that will not still be
// standing exactly there - a rigid body, something that melts, the weapon, or
// an authored opt-out - has to be out of the render list AND out of the digest,
// because a crate that cannot appear in a cubemap must not be able to make one
// stale either. This also pins down that entering the Runtime view captures
// nothing: it used to, and a look at the game's lighting cost a full render.
{
  const exclusionFailures = [];
  const excl = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const b = await import("/js/runtime.js");
    const m = await import("/js/main.js");
    const probe = ed.state.environmentProbes.get("ENV0001");
    const box = [probe.boxPosition, probe.boxSize];

    await b.setRuntimePreview(true);
    const before = b.meshesInProbeBox(...box);
    // A lamp owner is no good as the subject: the digest covers every authored
    // lamp, so moving one would change it for a reason that has nothing to do
    // with the exclusion under test.
    const lampOwners = new Set([...ed.state.lights.values()].map((l) => l.owner));
    const subjectId = before
      .map((mesh) => ed.ownerIdOf(mesh, true))
      .find((id) => id && !lampOwners.has(id));
    await b.setRuntimePreview(false);
    if (!subjectId) return { subjectId: null };

    const subject = ed.state.placements.get(subjectId);
    const wasName = subject.name;
    ed.renamePlacement(subjectId, "probeSubject");
    ed.setBehaviorDef("probeExcluded", { reflectionProbe: "exclude" });
    ed.addEntityBehavior("probeSubject", "probeExcluded");

    await b.setRuntimePreview(true);
    const after = b.meshesInProbeBox(...box);
    const ownMeshes = new Set(before.filter((mesh) => ed.ownerIdOf(mesh, true) === subjectId));
    await b.setRuntimePreview(false);

    // Entering the Runtime view with a probe owing a render must not take it:
    // the capture below is what proves nothing beat it to it.
    const sel = document.getElementById("view-mode");
    // applyViewMode disables the combo for the duration, so that plus the busy
    // lock is what "it has finished" means here.
    const settle = async () => {
      for (let i = 0; i < 600 && (ed.isBusy() || sel.disabled); i++) {
        await new Promise((r) => setTimeout(r, 50));
      }
      await new Promise((r) => setTimeout(r, 250));
    };
    sel.value = "runtime";
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    await settle();
    const modeAfterSwitch = sel.value;
    sel.value = "editor";
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    await settle();

    // Taking something out of a room IS a change to the room, so this one is
    // still owed...
    const excluded = await m.captureProbes();
    // ...but from here on the element is invisible to the digest as well, so
    // dragging it across the room cannot mark anything stale.
    subject.node.position.x += 1.5;
    const dragged = await m.captureProbes();
    subject.node.position.x -= 1.5;

    ed.removeEntityBehavior("probeSubject", 0);
    ed.deleteBehaviorDef("probeExcluded");
    ed.renamePlacement(subjectId, wasName);
    // Put the stamps back in step with the ship for the rest of the run.
    const restored = await m.captureProbes();
    return {
      subjectId,
      before: before.length,
      after: after.length,
      own: ownMeshes.size,
      leftBehind: [...ownMeshes].filter((mesh) => after.includes(mesh)).length,
      modeAfterSwitch,
      excluded: excluded.converted,
      dragged: dragged.converted,
      restored: restored.converted,
    };
  });
  console.log("probe exclude:", JSON.stringify(excl));
  if (!excl.subjectId) {
    exclusionFailures.push("no lamp-free element inside ENV0001 to exclude");
  } else {
    if (excl.own < 1) exclusionFailures.push("the subject owns no mesh in the probe box");
    if (excl.leftBehind !== 0) {
      exclusionFailures.push(`${excl.leftBehind} mesh(es) of an excluded element are still in the render list`);
    }
    if (excl.after !== excl.before - excl.own) {
      exclusionFailures.push(`the box went from ${excl.before} to ${excl.after} mesh(es), dropping ${excl.own}`);
    }
    if (excl.modeAfterSwitch !== "runtime") {
      exclusionFailures.push(`the view mode combo came back on ${excl.modeAfterSwitch}`);
    }
    if (excl.excluded < 1) {
      exclusionFailures.push("entering the Runtime view captured the probes by itself");
    }
    if (excl.dragged !== 0) {
      exclusionFailures.push(`moving an excluded element made ${excl.dragged} probe(s) stale`);
    }
    if (excl.restored < 1) {
      exclusionFailures.push("putting the element back did not mark its probe stale again");
    }
  }
  if (exclusionFailures.length) {
    console.log("PROBE EXCLUSION BROKEN:", exclusionFailures.join(" | "));
    errors.push("probe exclusion: " + exclusionFailures.join(" | "));
  }
}

// ---- the Runtime view, on the live ship ------------------------------------
// The runtime has no light rig, no global HDRI and no lightmaps: every mesh is
// lit by the authored lamps and reflects its own room's cubemap. The preview
// puts exactly that on the authored scene, by swapping each mesh's material for
// a clone carrying its room's probe - so the ship stays editable underneath it,
// which is the whole reason it is not a second glb.
{
  const preview = await page.evaluate(async () => {
    const b = await import("/js/runtime.js");
    const ed = await import("/js/editor.js");
    const engine = ed.state.scene.getEngine();
    const texBefore = ed.state.scene.textures.length;
    const frame = () => new Promise((r) => ed.state.scene.onAfterRenderObservable.addOnce(r));

    // A placement's own kit meshes. Its node can carry other things - a lamp's
    // gizmo rides the placement it lights, and under the preview a stand-in
    // rides each instance it draws for - and neither is the model.
    const ownKitMeshes = (e) =>
      e.node.getChildMeshes().filter((m) => m.metadata?.placementRoot === e.node && !m.metadata?.runtimePreview);
    const anyPlacement = [...ed.state.placements.values()].find((e) => !e.stage && ownKitMeshes(e).length);
    const authoredBefore = new Map(anyPlacement ? ownKitMeshes(anyPlacement).map((m) => [m, m.material]) : []);

    ed.state.camera.position.copyFrom(new BABYLON.Vector3(2, 2, 0));
    await b.setRuntimePreview(true);
    const p = b.runtimePreview();
    // `p.materials` is a live Set - the clone factory keeps adding to it as
    // meshes are dressed - so snapshot it at each use rather than binding once.
    const mats = () => [...p.materials];

    // An authored kit mesh is an InstancedMesh, which cannot carry a material
    // of its own - so "dressed" means the preview drew a stand-in for it and
    // took the original off screen, not that its own material changed.
    const isDressed = (m) => {
      const live = b.runtimePreview();
      const stand = live?.previewOf?.get(m);
      if (stand) return !stand.isDisposed() && stand.material?.metadata?.runtimePreview === true && m.isVisible === false;
      return m.material?.metadata?.runtimePreview === true;
    };

    const cloned = mats().find((m) => typeof m.roughness === "number") || mats()[0];
    const authoredMaterial = authoredBefore.values().next().value || null;
    const serializedBeforeReflection = JSON.stringify(ed.serialize());
    const authoredBeforeReflection = authoredMaterial
      ? { roughness: authoredMaterial.roughness, specularAA: authoredMaterial.enableSpecularAntiAliasing }
      : null;
    const aaControl = document.getElementById("runtime-specular-aa");
    const roughnessControl = document.getElementById("runtime-roughness");
    const roughnessValue = document.getElementById("runtime-roughness-val");
    const reflection = {
      controls: !!aaControl && !!roughnessControl && !!roughnessValue,
      cloned: !!cloned,
      authoredMaterial: !!authoredMaterial,
      aaOff: false,
      aaOn: false,
      roughnessChanged: false,
      authoredStable: false,
      serializedTracked: false,
    };
    if (aaControl && roughnessControl && roughnessValue && cloned) {
      aaControl.checked = false;
      aaControl.dispatchEvent(new Event("change", { bubbles: true }));
      reflection.aaOff = mats().every((m) => m.enableSpecularAntiAliasing === false);
      aaControl.checked = true;
      aaControl.dispatchEvent(new Event("change", { bubbles: true }));
      reflection.aaOn = mats().every((m) => m.enableSpecularAntiAliasing === true);

      const roughnessBefore = cloned.roughness;
      roughnessControl.value = "1.5";
      roughnessControl.dispatchEvent(new Event("input", { bubbles: true }));
      reflection.roughnessChanged = cloned.roughness !== roughnessBefore && roughnessValue.textContent === "1.50×";
      reflection.authoredStable =
        !authoredMaterial ||
        (authoredMaterial.roughness === authoredBeforeReflection.roughness &&
          authoredMaterial.enableSpecularAntiAliasing === authoredBeforeReflection.specularAA);
      // Both dials are authored ship values now, so they belong in the snapshot
      // the undo stack takes - and the authored materials above still must not
      // move, because the game applies the multiplier itself on load.
      const snapshot = ed.serialize();
      reflection.serializedTracked = JSON.stringify(snapshot) !== serializedBeforeReflection
        && snapshot.runtimeRoughnessFactor === 1.5
        && snapshot.runtimeSpecularAA === true;

      // Leave the controls at their defaults for the rest of the run.
      roughnessControl.value = "1";
      roughnessControl.dispatchEvent(new Event("input", { bubbles: true }));
    }

    // The viewport's own unlit toggle sweeps every material in the scene. Run
    // it both ways: the clones must come out the other side still lit.
    ed.setUnlit(true);
    const survivesUnlitOn = mats().every((m) => !m.unlit);
    ed.setUnlit(false);
    const survivesUnlitOff = mats().every((m) => !m.unlit);

    // Named so a failure says WHICH mesh and what it was wearing: every one of
    // these has a per-mesh cause, and a bare `false` sends the next reader
    // back to the browser to find out which.
    const undressed = p.meshes.filter((m) => m.material?.metadata?.runtimePreview !== true);
    const unbound = p.meshes.filter((m) => {
      const element = [...p.elementGroups].find(([, group]) => group.includes(m));
      const probe = element ? p.probeOf.get(element[0]) : null;
      const tex = m.material?.reflectionTexture;
      return !(probe ? !!tex && String(tex.name || "").startsWith("LocalEnvironment_") : !tex);
    });
    const offEnv = mats().filter((m) => !(Math.abs(m.environmentIntensity - ed.state.envIntensity) < 1e-6));
    const describe = (list) => list.slice(0, 4).map((m) => `${m.name}[${m.material?.name ?? "(none)"}]`);

    const out = {
      dressed: p.meshes.length,
      inProbe: p.inProbe,
      outsideProbe: p.outsideProbe,
      probesLoaded: p.localEnvironments.size,
      // Every dressed mesh wears a clone, and only a clone: handing a mesh its
      // authored material back would leave it reflecting the editor's HDRI.
      allCloned: undressed.length === 0,
      undressedSample: describe(undressed),
      // ...and no authored material was touched on the way past. An authored
      // kit mesh is an instance and cannot carry a material of its own, so
      // "untouched" is literally that: it still wears what it came in with.
      authoredUntouched: [...authoredBefore].every(([m, mat]) => m.material === mat),
      // A mesh inside a box reflects that box's cubemap; one outside reflects
      // nothing, which is what the runtime does with it.
      probeBound: unbound.length === 0,
      unboundSample: unbound.slice(0, 4).map((m) => {
        const element = [...p.elementGroups].find(([, group]) => group.includes(m));
        const probe = element ? p.probeOf.get(element[0]) : null;
        return `${m.name} probe=${probe ?? "(none)"} tex=${m.material?.reflectionTexture?.name ?? "(none)"}`;
      }),
      envWanted: ed.state.envIntensity,
      offEnvSample: offEnv.slice(0, 4).map((m) => `${m.name}=${m.environmentIntensity}`),
      // No lightmaps anywhere: the whole point of the rebuild.
      noLightmaps: p.meshes.every((m) => !m.material?.lightmapTexture),
      maxPreviewLights: Math.max(0, ...p.meshes.map((m) => m.material?.maxSimultaneousLights || 0)),
      // Physical falloff is Babylon's default and the wrong one here twice
      // over: it ignores `range` outright, and the clustered container still
      // sizes and culls each light proxy BY `range`, so a lamp was cut off in a
      // straight line where its proxy ended. glTF is the curve Babylon-Lite's
      // clustered shader hardcodes, so it is also what the game will draw.
      physicalFalloff: mats().filter((m) => m.usePhysicalLightFalloff !== false).length,
      gltfFalloff: mats().every((m) => m.useGLTFLightFalloff === true),
      // The editor's rig and its global HDRI are both things the game does not
      // have, and both would sit on top of exactly what is being inspected.
      rig: ed.state.runtime,
      rigDisabled: ["hemi", "hemiUp", "key", "fill"].every((name) => ed.state.scene.getLightByName(name)?.isEnabled() === false),
      environmentTexture: !!ed.state.scene.environmentTexture,
      sceneNeutral: Math.abs(ed.state.scene.environmentIntensity - 1) < 1e-6,
      envOnMaterials: offEnv.length === 0,
      lamps: p.lights.length,
      lampStats: p.lightStats,
      clusteredExpected: p.clusteredLights.length,
      clusteredContainer: p.clusteredContainer?.getClassName() || null,
      clusteredActual: p.clusteredContainer?.lights.length || 0,
      clusteredContainerUnscoped: !p.clusteredContainer || p.clusteredContainer.includedOnlyMeshes.length === 0,
      // A scoped lamp is a scarce UBO slot, so it is held to its own room.
      regularScopedToChunk: p.lightSpecs.every((spec, i) => spec.r.clustered || p.lights[i].includedOnlyMeshes.length > 0),
      // The ship stays on screen and editable: there is no second copy of it to
      // stand in for anything.
      authoredVisible: anyPlacement ? ownKitMeshes(anyPlacement).every((m) => m.isEnabled() && m.isPickable) : null,
      authoredHiddenSample: anyPlacement
        ? ownKitMeshes(anyPlacement)
            .filter((m) => !(m.isEnabled() && m.isPickable))
            .slice(0, 4)
            .map((m) => `${m.name} enabled=${m.isEnabled()} pickable=${m.isPickable}`)
        : [],
      ownKit: anyPlacement ? ownKitMeshes(anyPlacement).length : 0,
      reflection,
      survivesUnlitOn,
      survivesUnlitOff,
    };

    // ---- the scoped lamps follow the room the camera is in -----------------
    // Stand in the middle of a room rather than trusting wherever the editor's
    // default view happens to sit: `nearestChunk` is only meaningful against a
    // known position, and chunk boxes overlap, so the centre of a chunk's own
    // box is the one point most clearly inside it.
    const regularLightState = () => {
      const live = b.runtimePreview();
      return live.lightSpecs
        .map((spec, i) => ({ chunk: spec.chunk, clustered: spec.r.clustered, enabled: live.lights[i].isEnabled() }))
        .filter((spec) => !spec.clustered);
    };
    const centreOf = (id) => {
      const box = p.chunkBounds.find((c) => c.id === id);
      return new BABYLON.Vector3((box.min.x + box.max.x) / 2, (box.min.y + box.max.y) / 2, (box.min.z + box.max.z) / 2);
    };
    out.chunkBounds = p.chunkBounds.map((c) => ({
      id: c.id,
      min: [c.min.x, c.min.y, c.min.z],
      max: [c.max.x, c.max.y, c.max.z],
    }));
    const cameraPosition = ed.state.camera.position.clone();
    ed.state.camera.position.copyFrom(centreOf("CH00_Storage"));
    await frame();
    out.nearestAtStart = b.runtimePreview().nearestChunk;
    out.regularAtStart = regularLightState();
    ed.state.camera.position.copyFrom(centreOf("CH01_CorridorA"));
    await frame();
    out.nearestAfterMove = b.runtimePreview().nearestChunk;
    out.regularAfterMove = regularLightState();
    ed.state.camera.position.copyFrom(cameraPosition);
    await frame();

    // ---- the ship is still editable underneath the preview ------------------
    // Dragging an element has to carry its dressed meshes, its lamps and its
    // probe assignment with it, because all three are recomputed per frame.
    const editable = [...ed.state.placements.values()].filter((e) => !e.stage);
    const subject = editable[editable.length - 1];
    if (subject) {
      const boxOf = (e) => {
        let lo = [Infinity, Infinity, Infinity];
        let hi = [-Infinity, -Infinity, -Infinity];
        for (const m of ownKitMeshes(e)) {
          if (!m.getTotalVertices?.()) continue;
          m.computeWorldMatrix(true);
          const bb = m.getBoundingInfo().boundingBox;
          const a = bb.minimumWorld.asArray();
          const c = bb.maximumWorld.asArray();
          for (let i = 0; i < 3; i++) {
            lo[i] = Math.min(lo[i], a[i]);
            hi[i] = Math.max(hi[i], c[i]);
          }
        }
        return Number.isFinite(lo[0]) ? lo.concat(hi) : null;
      };
      const at = boxOf(subject);
      subject.node.position.x += 2;
      await frame();
      const moved = boxOf(subject);
      out.dragTracked = at && moved ? +(moved[0] - at[0]).toFixed(2) : null;
      out.dragKeptDressed = ownKitMeshes(subject).every(isDressed);
      out.dragSample = ownKitMeshes(subject)
        .filter((m) => !isDressed(m))
        .slice(0, 4)
        .map((m) => `${m.name}[${m.material?.name ?? "(none)"}] visible=${m.isVisible} stand=${!!b.runtimePreview()?.previewOf?.get(m)}`);
      subject.node.position.x -= 2;
      await frame();

      // Delete and undo: the preview re-dresses off the `placements` event, so
      // a rebuilt node must come back wearing a clone rather than its authored
      // material - the bug being guarded is a room that goes matte after undo.
      const id = subject.id;
      ed.pushUndo();
      ed.removePlacement(id);
      await frame();
      out.deletedGone = !ed.state.placements.has(id);
      await ed.undo();
      await frame();
      const back = ed.state.placements.get(id);
      out.undoRedressed = !!back && ownKitMeshes(back).every(isDressed);
    }

    // ---- the lamps answer to the inspector ---------------------------------
    // The preview builds its lamps from the editor's live records, so an edit
    // has to reach the screen without a reload.
    const lit2 = await import("/js/lights.js");
    const spec = b.runtimePreview().lightSpecs?.[0];
    if (spec) {
      const lamp = b.runtimePreview().lights[0];
      const lid = spec.light.id;
      const original = {
        type: spec.r.type,
        range: spec.r.range,
        intensity: spec.r.intensity,
        angle: spec.r.angle,
        color: [...spec.r.color],
      };
      lit2.setLightPart(lid, "runtime", { range: 41.5, intensity: 6.25, color: [0.2, 0.4, 0.6] });
      const changed = b.runtimePreview().lights[0];
      out.lampRange = changed.range;
      out.lampIntensity = changed.intensity;
      out.lampColor = changed.diffuse.asArray().map((v) => +v.toFixed(3));
      // Poked in place, not rebuilt: a slider drag emits on every tick, and
      // disposing a light dirties every material it touched.
      out.lampReused = b.runtimePreview().lights[0] === lamp;
      // ...but a change of shape has to rebuild, because the constructor and
      // `includedOnlyMeshes` are not settable after the fact.
      const sig = b.runtimePreview().lightSig;
      lit2.setLightPart(lid, "runtime", { type: original.type === "point" ? "spot" : "point" });
      out.lampRebuilt = b.runtimePreview().lightSig !== sig;
      lit2.setLightPart(lid, "runtime", { type: "spot", angle: 37 });
      out.lampCone = +b.runtimePreview().lights[0].angle.toFixed(4);
      lit2.setLightPart(lid, "runtime", original);
      // The lamp rides its owner, so a drag must carry it - a lamp left behind
      // would light the place the fitting used to be.
      const owner = ed.state.placements.get(spec.light.owner);
      const at = b.runtimePreview().lights[0].position?.clone();
      if (owner && at) {
        owner.node.position.x += 3;
        await frame();
        out.lampFollowed = +(b.runtimePreview().lights[0].position.x - at.x).toFixed(2);
        owner.node.position.x -= 3;
        await frame();
      }

      // The live regression: a clustered lamp must actually reach the meshes it
      // was built for, through a compiled clustered shader rather than through
      // Babylon's regular light loop.
      const live = b.runtimePreview();
      const clustered = live.lightSpecs.find((s) => s.r.clustered);
      const target = live.meshes.find((m) => m.material?.metadata?.runtimePreview);
      const targetSource = target?._sourceMesh || target;
      out.clusteredTarget = !!clustered && !!target;
      out.clusteredBound = !!live.clusteredContainer && !!targetSource?.lightSources?.includes(live.clusteredContainer);
      out.clusteredShader = false;
      if (clustered && target) {
        target.computeWorldMatrix(true);
        const centre = target.getBoundingInfo().boundingBox.centerWorld.clone();
        const camera = ed.state.camera;
        const wasAt = camera.position.clone();
        const wasRotation = camera.rotation.clone();
        camera.position.copyFrom(centre.add(new BABYLON.Vector3(0, 0.8, 3)));
        camera.setTarget(centre);
        await frame();
        await frame();
        // Parallel shader compilation may finish after the first frame at a
        // newly focused camera, so let the material settle before reading its
        // generated source.
        await new Promise((r) => setTimeout(r, 1200));
        const effect = targetSource.material?._activeEffect;
        out.clusteredShader =
          !!effect && String(effect.defines).includes("CLUSTLIGHT_SLICES") && String(effect._fragmentSourceCode).includes("CLUSTLIGHT");
        camera.position.copyFrom(wasAt);
        camera.rotation.copyFrom(wasRotation);
        await frame();
      }
    }

    // ---- the outline goes on whatever is actually drawn ---------------------
    // Here the authored instance is invisible and a stand-in draws in its
    // place, and Babylon dispatches an edges renderer from the ACTIVE mesh
    // list - so edges hung on the hidden instance are built, updated and never
    // reach the screen. Picking is untouched either way: it resolves through
    // metadata.placementRoot, which is why this looked fine for a while.
    if (anyPlacement) {
      // Re-resolve: the delete/undo above hands back a fresh record for
      // whichever element it acted on, and a stale one has no meshes.
      const outlined = ed.state.placements.get(anyPlacement.id);
      ed.select([]);
      await frame();
      ed.select([outlined.id]);
      await frame();
      const authored = ownKitMeshes(outlined);
      const pairs = authored
        .map((mesh) => [mesh, b.runtimePreview()?.previewOf?.get(mesh)])
        .filter(([, stand]) => !!stand);
      const stand = pairs.map(([, s]) => s);
      out.outlineStandIns = stand.length;
      out.outlineDrawn = stand.length > 0 && stand.every((mesh) => !!mesh.edgesRenderer && mesh.isVisible !== false);
      // ...and not on the instance it draws for, which is off screen
      out.outlineNotOnHidden = pairs.every(([mesh]) => !mesh.edgesRenderer);
      ed.select([]);
      await frame();
      out.outlineCleared = stand.every((mesh) => !mesh.edgesRenderer);
    }

    await b.setRuntimePreview(false);
    return {
      ...out,
      offPreview: b.runtimePreview(),
      offRestored: [...authoredBefore].every(([m, mat]) => m.material === mat),
      offRigEnabled: ["hemi", "hemiUp", "key", "fill"].every((name) => ed.state.scene.getLightByName(name)?.isEnabled() === true),
      offEnvironmentTexture: !!ed.state.scene.environmentTexture,
      texBefore,
      texAfter: ed.state.scene.textures.length,
      engineCached: (engine._internalTexturesCache || []).length,
    };
  });

  const preFail = [];
  if (!preview.dressed) preFail.push("the preview dressed no meshes");
  if (!preview.probesLoaded) preFail.push("no captured probe was loaded");
  if (!preview.inProbe) preFail.push("not one element resolved to a probe");
  if (!preview.allCloned) preFail.push("a dressed mesh is still wearing its authored material");
  if (!preview.authoredUntouched) preFail.push("the preview edited an authored material in place");
  if (!preview.probeBound) preFail.push("a mesh is not reflecting the probe its element resolved to");
  if (!preview.noLightmaps) preFail.push("a preview material still carries a lightmap");
  if (preview.maxPreviewLights > 4) {
    preFail.push(`preview materials use ${preview.maxPreviewLights} light slots`);
  }
  if (preview.physicalFalloff) {
    preFail.push(`${preview.physicalFalloff} preview material(s) still fall off as 1/d², which ignores Range`);
  }
  if (!preview.gltfFalloff) preFail.push("a preview material is not using the glTF light falloff");
  if (!preview.rig) preFail.push("the runtime flag was not set");
  if (!preview.rigDisabled) preFail.push("Runtime mode left the editor rig lights enabled");
  if (preview.environmentTexture) preFail.push("Runtime mode kept the editor's global HDRI");
  if (!preview.sceneNeutral) preFail.push("scene.environmentIntensity is a second, invisible factor");
  if (!preview.envOnMaterials) preFail.push("Env did not reach the preview materials");
  if (!preview.lamps) preFail.push("the preview built no lamps at all");
  if (preview.clusteredExpected > 0 && (preview.clusteredContainer !== "ClusteredLightContainer" || preview.clusteredActual !== preview.clusteredExpected)) {
    preFail.push(`clustered lamps were not aggregated (${preview.clusteredActual}/${preview.clusteredExpected})`);
  }
  if (!preview.clusteredContainerUnscoped) {
    preFail.push("the clustered container is mesh-scoped and can miss instances");
  }
  if (!preview.regularScopedToChunk) preFail.push("a scoped lamp reaches the whole ship");
  const regularAtStart = preview.regularAtStart || [];
  const regularAfterMove = preview.regularAfterMove || [];
  if (regularAtStart.length && (preview.nearestAtStart !== "CH00_Storage" || regularAtStart.some((l) => l.enabled !== (l.chunk === "CH00_Storage")))) {
    preFail.push("scoped lamps were not limited to the nearest start chunk");
  }
  if (regularAfterMove.length && (preview.nearestAfterMove !== "CH01_CorridorA" || regularAfterMove.some((l) => l.enabled !== (l.chunk === "CH01_CorridorA")))) {
    preFail.push("scoped lamps did not follow the nearest chunk");
  }
  if (!preview.reflection?.controls) preFail.push("the runtime reflection controls are missing");
  if (!preview.reflection?.cloned || !preview.reflection?.authoredMaterial) {
    preFail.push("the reflection test found no materials");
  }
  if (!preview.reflection?.aaOff || !preview.reflection?.aaOn) {
    preFail.push("the Specular AA control did not update the preview materials");
  }
  if (!preview.reflection?.roughnessChanged) {
    preFail.push("the Reflection roughness control did not update the preview materials");
  }
  if (!preview.reflection?.authoredStable) preFail.push("reflection controls changed an authored material");
  if (!preview.reflection?.serializedTracked) preFail.push("the reflection controls are missing from the undo snapshot");
  if (!preview.survivesUnlitOn || !preview.survivesUnlitOff) {
    preFail.push("the viewport's unlit toggle clobbered the preview materials");
  }

  // ---- editing under the preview -----------------------------------------
  if (!preview.ownKit) preFail.push("no placement mesh carries placementRoot");
  if (preview.authoredVisible !== true) preFail.push("the ship is not visible and pickable under the preview");
  if (preview.dragTracked !== 2) preFail.push(`a dragged element moved ${preview.dragTracked}m of 2`);
  if (!preview.dragKeptDressed) preFail.push("dragging an element undressed it");
  if (!preview.deletedGone) preFail.push("deleting an element left it in the scene");
  if (!preview.undoRedressed) preFail.push("undo brought an element back undressed");

  // ---- selecting still shows you what you selected ------------------------
  if (preview.outlineStandIns !== undefined) {
    if (!preview.outlineStandIns) preFail.push("the selected element has no stand-in to outline");
    if (!preview.outlineDrawn) preFail.push("the selection outline is not on the mesh being drawn");
    if (!preview.outlineNotOnHidden) preFail.push("the outline went on the hidden authored instance");
    if (!preview.outlineCleared) preFail.push("deselecting left the outline behind");
  }

  // ---- the lamps answer to the inspector ---------------------------------
  if (preview.lampRange !== undefined) {
    if (preview.lampRange !== 41.5 || preview.lampIntensity !== 6.25) {
      preFail.push(`editing a lamp did not reach the preview (range ${preview.lampRange}, intensity ${preview.lampIntensity})`);
    }
    if (JSON.stringify(preview.lampColor) !== JSON.stringify([0.2, 0.4, 0.6])) {
      preFail.push(`editing a lamp colour did not reach the preview (${preview.lampColor})`);
    }
    if (Math.abs(preview.lampCone - (37 * Math.PI) / 180) > 1e-4) {
      preFail.push(`editing a lamp cone did not reach the preview (${preview.lampCone})`);
    }
    if (!preview.lampReused) preFail.push("a scalar lamp edit rebuilt the light");
    if (!preview.lampRebuilt) preFail.push("changing a lamp's type did not rebuild it");
    if (preview.lampFollowed !== 3) {
      preFail.push(`a dragged element left its lamp behind (${preview.lampFollowed}m of 3)`);
    }
    if (preview.clusteredTarget && (!preview.clusteredBound || !preview.clusteredShader)) {
      preFail.push("a dressed mesh did not compile with the clustered container");
    }
  }

  // ---- and turning it off puts the editor back exactly -------------------
  // The bug this guards: `Material.clone` deep-copies every texture slot, so
  // each preview material owns private wrappers the authored material never
  // sees. Left behind they leak a full set of the ship's textures per switch.
  if (preview.offPreview !== null) preFail.push("turning it off kept the preview");
  if (!preview.offRestored) preFail.push("turning it off did not give the meshes their materials back");
  if (!preview.offRigEnabled) preFail.push("turning it off did not restore the rig lights");
  if (!preview.offEnvironmentTexture) preFail.push("turning it off did not restore the editor's HDRI");
  if (preview.texAfter !== preview.texBefore) {
    preFail.push(`the preview leaked ${preview.texAfter - preview.texBefore} texture(s)`);
  }

  console.log(
    `runtime view : ${preFail.length ? preFail.join(" | ") : "ok"},` +
      ` ${preview.dressed} mesh(es), ${preview.inProbe} in a probe / ${preview.outsideProbe} outside,` +
      ` ${preview.lamps} lamp(s) ${JSON.stringify(preview.lampStats)}`
  );
  if (preFail.length) {
    console.log(
      "  diag       :",
      JSON.stringify({
        chunkBounds: preview.chunkBounds,
        nearestAtStart: preview.nearestAtStart,
        nearestAfterMove: preview.nearestAfterMove,
        regularAtStart: preview.regularAtStart,
        regularAfterMove: preview.regularAfterMove,
        reflection: preview.reflection,
        undressed: preview.undressedSample,
        unbound: preview.unboundSample,
        env: { wanted: preview.envWanted, off: preview.offEnvSample },
        drag: preview.dragSample,
        authoredHidden: preview.authoredHiddenSample,
        clustered: {
          target: preview.clusteredTarget,
          bound: preview.clusteredBound,
          shader: preview.clusteredShader,
        },
      })
    );
    errors.push("runtime preview: " + preFail.join(" | "));
    assertFailures.push("runtime preview: " + preFail.join(" | "));
  }
}

console.log("\nerrors:", errors.length ? [...new Set(errors)].join("\n") : "(none)");
await browser.close();
// Losing the ability to recover work - or the viewpoint you saved from - is
// worth failing the run over. So is anything in `assertFailures`, which is
// what the runtime-preview block asserts: a preview that leaks textures, keeps
// the editor's rig on top of the authored lamps, or cannot be edited under is
// a broken editor, not a warning.
if (rotationFailures.length || benchFailures.length || glbFailures.length || spaceFailures.length || probeFailures.length || assertFailures.length) process.exit(1);
