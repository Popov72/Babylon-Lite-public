// End-to-end check from a clean export/ directory:
// the tool must ignore the Blender-era manifest, and must preserve both
// Blender artefacts the first time it writes over them.

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { toolUrl } from "./target.mjs";

const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const URL = toolUrl();
const EXPORT = process.env.SHIP_EXPORT_DIR || "D:/alexis/TombRaider/Popov72/SciFiShip/export";
const before = Object.fromEntries(fs.readdirSync(EXPORT)
  .map((f) => [f, fs.statSync(path.join(EXPORT, f)).size]));
console.log("export before:", JSON.stringify(before));

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
// `errors` is the printed log, and it also collects whatever the page logs to
// console.error - CDN noise included - so it is deliberately not fatal on its
// own. Blocks that assert something worth failing over push into their own
// array and add it to the exit condition at the bottom. `assertFailures` is
// the shared one for blocks that had no array of their own, which is how a
// whole block of baked-preview checks came to print failures and still let the
// run exit 0 - the mirrored-stand-in bug shipped straight through it.
const assertFailures = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

await page.goto(URL, { waitUntil: "domcontentloaded" });
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

  // A prop the runtime moves, so the bake has something to leave out. A leaf
  // that slides open would otherwise paint its own shadow across the floor it
  // used to cover, and that shadow would stay there once it opened.
  ed.setBehaviorDef("SlidingLeaf", { dynamic: true, axis: "x", distance: 2 });
  ed.addEntityBehavior(doorLeaf.node.name, "SlidingLeaf");
  ed.state.activeChunk = "CH01_CorridorA";
  const dynamicCorridorProp = await ed.placeAt(
    "Props/Prop_Light_Small", new V(16, 2, -4), { silent: true });
  ed.addEntityBehavior(dynamicCorridorProp.node.name, "SlidingLeaf");
  ed.state.activeChunk = "CH00_Storage";

  // A real kit lamp, whose M_Light material genuinely emits. It seeds its own
  // area light from kit_lights.json, which is the exact pair the bake has to
  // stop double-counting: an emissive mesh AND an authored light on top of it.
  const lamp = await ed.placeAt("Props/Prop_Light_Small", new V(4, 2, -4), { silent: true });

  // A window onto space in the corridor, so the bake suite has a portal to
  // find. The far side is deliberately the skybox: a door between two rooms
  // gets no world light through it and must NOT become a portal.
  mk.addDoor(new V(16, 0, -2), {
    chunkA: "CH01_CorridorA", chunkB: ed.SKYBOX_CHUNK, silent: true,
  });

  // A lamp in each chunk, so the glb check below has both a light to find and a
  // second one to tell it apart from - and one of them is bake-only, which is
  // the shape that carries no runtime light at all.
  const lt = await import("/js/lights.js");
  const rooms = [...ed.state.placements.values()];
  const inA = rooms.find((p) => p.chunk === "CH00_Storage");
  const inB = rooms.find((p) => p.chunk === "CH01_CorridorA");
  lt.addLight(inA.id, {
    offset: [0, 2.5, 0],
    bake: { shape: "rectangle", sizeX: 1.2, sizeY: 0.2, watts: 55 },
    silent: true,
  });
  lt.addLight(inA.id, {
    offset: [0, 1.5, 0],
    runtime: { type: "point", clustered: false },
    silent: true,
  });
  lt.addLight(inB.id, {
    offset: [0, 2, 0],
    bake: { shape: "disk", sizeX: 0.8 },
    runtime: { type: "none" },
    silent: true,
  });
  lt.addLight(inB.id, {
    offset: [0, 1.5, 0],
    runtime: { type: "point", clustered: false },
    silent: true,
  });

  // A non-default ship-wide constant, so the round-trip check below has teeth:
  // buildManifest() assembles its own object rather than writing serialize()'s,
  // so anything it forgets to copy is lost on save and silently comes back on
  // its default. Leaving this at 0.008 would not have caught that.
  ed.setConfig("shellThickness", 0.0075);

  return {
    placements: ed.state.placements.size, markers: ed.state.markers.size,
    door: door?.id, lights: ed.state.lights.size, lamp: lamp?.id,
    moving: doorLeaf.node.name,
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
  const tagged = (json.nodes || []).filter((n) => n.extras?.id && n.extras.kind !== "light");
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

  // ---- lights leave as bare nodes, whole ----------------------------------
  // Nothing is drawn for a light: the Blender bake script rebuilds each one as
  // a Cycles Area light from `extras`, and the runtime builds a Babylon light
  // from the same record - so a half-written extras block is a bake that is
  // quietly wrong rather than a bake that fails. The node also has to be a
  // CHILD of the element it rides, which is what carries the authored offset
  // across in the space it was authored in.
  const nodes = json.nodes || [];
  const lightNodes = nodes.filter((n) => n.extras?.kind === "light");
  const parentOf = new Map();
  nodes.forEach((n, i) => (n.children || []).forEach((c) => parentOf.set(c, i)));
  const whole = lightNodes.filter((n) => {
    const b = n.extras.bake, r = n.extras.runtime;
    return b && r && typeof b.shape === "string" && typeof b.watts === "number"
      && Array.isArray(b.color) && typeof r.type === "string"
      && typeof r.clustered === "boolean" && typeof r.castsShadows === "boolean";
  });
  const ridden = lightNodes.filter((n) => {
    const p = nodes[parentOf.get(nodes.indexOf(n))];
    return p && p.extras?.id === n.extras.owner;
  });
  const named = lightNodes.filter((n) => n.name === `LIGHT_${n.extras.id}`);
  // The offset is carried by the node's own translation, not by the extras -
  // so a node that arrived with its record intact and its transform dropped
  // would bake every lamp at the foot of the wall it belongs to. Checked
  // against what the manifest says was authored, in glTF space: the exporter
  // mirrors X, so the manifest's own conversion has to hold here too.
  const authored = new Map((man.lights || []).map((l) => [l.id, l.offset]));
  const placed = lightNodes.filter((n) => {
    const o = authored.get(n.extras.id);
    const t = n.translation || [0, 0, 0];
    return o && [-o[0], o[1], o[2]].every((v, i) => Math.abs(v - (t[i] || 0)) < 1e-4);
  });
  const gizmos = nodes.filter((n) => /_gizmo$|_stub$/.test(n.name || ""));
  console.log(`glb lights   : ${lightNodes.length} node(s), ${whole.length} carry both halves,`
    + ` ${ridden.length} hang off the element they ride,`
    + ` shapes: ${lightNodes.map((n) => n.extras.bake?.shape).join("+") || "(none)"},`
    + ` at y ${lightNodes.map((n) => n.translation?.[1]).join("+")}`);
  if (lightNodes.length !== built.lights) {
    glbFailures.push(`${built.lights} light(s) authored but ${lightNodes.length} exported`);
  }
  if (whole.length !== lightNodes.length) {
    glbFailures.push(`${lightNodes.length - whole.length} light node(s) lost half their record`);
  }
  if (ridden.length !== lightNodes.length) {
    glbFailures.push(`${lightNodes.length - ridden.length} light node(s) are not children`
      + " of the element they ride");
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
  console.log(`glb glass    : ${glass.length} material(s), green and see-through:`
    + ` ${green.length}, extensions: ${exts.join("+") || "(none)"}`);
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
    const declared = new Set([...(sp.gltf || []), ...(sp.editor || []), ...(sp.none || [])]
      .map((k) => k.split("[")[0]));
    const undeclared = Object.keys(man).filter((k) => !declared.has(k));
    const phantom = [...declared].filter((k) => !(k in man));
    console.log(`manifest space: ${declared.size} field(s) declared,`
      + ` ${undeclared.length} undeclared, ${phantom.length} phantom`);
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

// ---- the bake runs from the editor, not from a terminal ---------------------
// A bake is minutes of Cycles, so /api/bake starts it and reports on it rather
// than answering with it - the browser would time out long before Blender
// finished. This drives the whole round trip at a deliberately tiny resolution:
// what is under test is the wiring, not the render.
const bakeFailures = [];
{
  const status = await page.evaluate(async () => (await fetch("/api/bake")).json());
  if (!status.available) {
    console.log("bake api     : no Blender on this machine - skipping");
  } else {
    const done = await page.evaluate(async () => {
      const m = await import("/js/main.js");
      return m.bakeNow({ samples: 4, resolution: 64 });
    });
    const maps = done.report?.baked || [];
    console.log(`bake api     : ${done.ok ? "ok" : "failed"},`
      + ` ${maps.length} map(s), ${Math.round(((done.finishedAt || 0)
        - (done.startedAt || 0)) / 1000)}s, ${done.error || "no error"}`);
    if (!done.ok) bakeFailures.push(`the bake failed: ${done.error}`);
    if (!maps.length) bakeFailures.push("the bake wrote no lightmaps");
    // Written where the runtime will look for them, and reachable over the
    // same server the editor is served from.
    for (const m of maps) {
      if (!fs.existsSync(m.image)) bakeFailures.push(`${m.chunk}: no image at ${m.image}`);
    }
    const served = maps.length
      ? await page.evaluate(async (name) =>
        (await fetch(`/lightmaps/${name}`)).status, path.basename(maps[0].image))
      : 0;
    if (served !== 200) bakeFailures.push(`/lightmaps/ serves ${served}, not the map`);
    if (bakeFailures.length) {
      console.log("BAKE BROKEN:", bakeFailures.join(" | "));
      errors.push("bake: " + bakeFailures.join(" | "));
    }

    // ---- and the bake can be looked at without leaving the editor ----------
    // The lightmaps cannot go on the authored meshes - they have one UV set and
    // their materials are shared across the whole ship - so the preview is
    // ship_baked.glb itself, shown instead of the ship being edited.
    const preview = await page.evaluate(async () => {
      const b = await import("/js/baked.js");
      const ed = await import("/js/editor.js");
      const served = (await fetch("/export/ship_baked.glb")).status;
      const engine = ed.state.scene.getEngine();
      const glbCached = () => (engine._internalTexturesCache || [])
        .filter((t) => String(t._originalUrl || t.url || "").includes("ship_baked.glb")).length;
      const texBefore = ed.state.scene.textures.length;
      ed.state.camera.position.copyFrom(new BABYLON.Vector3(2, 2, 0));
      await b.setBakedPreview(true);
      const p = b.bakedPreview();
      const anyPlacement = [...ed.state.placements.values()][0];
      const bakedMaterial = p.materials.find((m) =>
        typeof m.roughness === "number" && m.roughness > 0) || p.materials[0];
      const authoredMaterial = ed.state.scene.meshes
        .map((m) => m.material)
        .find((m) => m && m !== bakedMaterial && !m.metadata?.bakedPreview);
      const serializedBeforeReflection = JSON.stringify(ed.serialize());
      const authoredBeforeReflection = authoredMaterial ? {
        roughness: authoredMaterial.roughness,
        specularAA: authoredMaterial.enableSpecularAntiAliasing,
      } : null;
      const aaControl = document.getElementById("baked-specular-aa");
      const roughnessControl = document.getElementById("baked-roughness");
      const roughnessValue = document.getElementById("baked-roughness-val");
      const reflection = {
        controls: !!aaControl && !!roughnessControl && !!roughnessValue,
        bakedMaterial: !!bakedMaterial,
        authoredMaterial: !!authoredMaterial,
        aaOff: false,
        aaOn: false,
        roughnessChanged: false,
        authoredStable: false,
        serializedStable: false,
      };
      if (aaControl && roughnessControl && roughnessValue && bakedMaterial) {
        aaControl.checked = false;
        aaControl.dispatchEvent(new Event("change", { bubbles: true }));
        reflection.aaOff = p.materials.every((m) =>
          m.enableSpecularAntiAliasing === false);
        aaControl.checked = true;
        aaControl.dispatchEvent(new Event("change", { bubbles: true }));
        reflection.aaOn = p.materials.every((m) =>
          m.enableSpecularAntiAliasing === true);

        const roughnessBefore = bakedMaterial.roughness;
        roughnessControl.value = "1.5";
        roughnessControl.dispatchEvent(new Event("input", { bubbles: true }));
        reflection.roughnessChanged = bakedMaterial.roughness !== roughnessBefore
          && roughnessValue.textContent === "1.50×";
        reflection.authoredStable = !authoredMaterial || (
          authoredMaterial.roughness === authoredBeforeReflection.roughness
          && authoredMaterial.enableSpecularAntiAliasing
            === authoredBeforeReflection.specularAA
        );
        reflection.serializedStable =
          JSON.stringify(ed.serialize()) === serializedBeforeReflection;

        // Leave the preview controls at their defaults for the rest of the
        // existing baked-preview checks and for the next test run.
        roughnessControl.value = "1";
        roughnessControl.dispatchEvent(new Event("input", { bubbles: true }));
      }
      // A placement's own kit meshes, which is what the stand-in replaces. Its
      // node can carry other things - a lamp's gizmo rides the placement it
      // lights - and those stay on screen, or the lamp could not be selected.
      const ownKitMeshes = (e) =>
        e.node.getChildMeshes().filter((m) => m.metadata?.placementRoot === e.node);
      // The viewport's own unlit toggle sweeps every material in the scene.
      // Run it both ways over the preview: the clones must come out the other
      // side still carrying their own compositing.
      ed.setUnlit(true);
      const survivesUnlitOn = p.materials.every((m) => !m.unlit && m.useLightmapAsShadowmap);
      ed.setUnlit(false);
      const survivesUnlitOff = p.materials.every((m) => !m.unlit && m.useLightmapAsShadowmap);
      const dynamicLightMeshes = new Set(p.dynamic.map((m) => m._sourceMesh || m));
      const bakedMeshes = ed.state.scene
        .getTransformNodeByName("BAKED_PREVIEW")
        .getChildMeshes()
        .filter((m) => m.material?.metadata?.bakedPreview);
      const on = {
        served,
        lit: p.lit,
        maps: p.textures.length,
        uv: p.textures.map((t) => t.coordinatesIndex),
        level: p.textures.map((t) => t.level),
        // Baked mode is what silences the four analytic lights now - there is
        // no separate switch to forget to flip.
        rig: ed.state.baked,
        rigDisabled: ["hemi", "hemiUp", "key", "fill"].every((name) =>
          ed.state.scene.getLightByName(name)?.isEnabled() === false),
        // the authored runtime lamps, rebuilt over the props the bake skipped
        lamps: p.lights.length,
        clusteredExpected: p.clusteredLights.length,
        clusteredContainer: p.clusteredContainer?.getClassName() || null,
        clusteredActual: p.clusteredContainer?.lights.length || 0,
        regularLamps: p.lights.length - p.clusteredLights.length,
        maxPreviewLights: Math.max(0, ...p.dynamic
          .map((m) => m.material?.maxSimultaneousLights || 0)),
        dynamic: p.skipped,
        dynamicVisible: p.dynamic.filter((m) =>
          m.isEnabled() && m.isVisible && m.visibility > 0).length,
        dynamicAllHaveNoUv2: p.dynamic.every((m) =>
          !m.isVerticesDataPresent(BABYLON.VertexBuffer.UV2Kind)),
        bakedAnalyticLightingDisabled: bakedMeshes.every((m) =>
          m.material?.disableLighting === true),
        clusteredContainerUnscoped: !p.clusteredContainer
          || p.clusteredContainer.includedOnlyMeshes.length === 0,
        regularScopesDynamicOnly: p.lights.every((light) =>
          p.clusteredLights.includes(light)
          || (light.includedOnlyMeshes.length > 0
            && light.includedOnlyMeshes.every((m) => dynamicLightMeshes.has(m)))),
        // The baked diffuse remains multiplied by the lightmap, while the
        // normal PBR path keeps environment reflections for metallic surfaces.
        composited: p.materials.every((m) => m.unlit === false
          && m.disableLighting === true
          && m.useLightmapAsShadowmap === true
          && m.metadata?.bakedPreview === true),
        reflection,
        environment: {
          texture: !!ed.state.scene.environmentTexture,
          intensity: ed.state.scene.environmentIntensity,
        },
        survivesUnlitOn,
        survivesUnlitOff,
        // Nothing wearing a lightmap may be missing the UV set that samples
        // it: the multiply would read zero and paint the mesh black.
        everyLitMeshHasUv2: ed.state.scene
          .getTransformNodeByName("BAKED_PREVIEW")
          .getChildMeshes()
          .filter((m) => m.material?.metadata?.bakedPreview)
          .every((m) => m.isVerticesDataPresent(BABYLON.VertexBuffer.UV2Kind)),
        // The element's node stays enabled now - it is editable through the
        // bake - so what must be off is its own kit geometry, which the
        // stand-in is drawn in place of.
        authoredHidden: anyPlacement
          ? ownKitMeshes(anyPlacement).every((m) => !m.isEnabled()) : null,
        // ...and that the filter above actually found something, or the two
        // assertions built on it would pass on an empty list.
        ownKit: anyPlacement ? ownKitMeshes(anyPlacement).length : 0,
        rootInScene: !!ed.state.scene.getTransformNodeByName("BAKED_PREVIEW"),
        ...(await (async () => {
          // Editing through the bake: the exported node stands in for the
          // element, and has to answer for it under a pick, a drag and a
          // delete. See "Editing through the bake" in the README.
          const out = { standIns: p.standIns.size, ship: 0, standOwner: null };
          out.flags = { isolate: ed.state.isolate, layer: ed.state.showLayer,
            hidden: ed.state.hidden.size, staging: ed.state.collisionMode };
          const ids = [];
          for (const e of ed.state.placements.values()) {
            if (!e.stage) { out.ship++; ids.push(e.id); }
          }
          out.standDrift = b.bakedDrift();
          // Deliberately the last one: `anyPlacement` above is the first, and
          // this block deletes and restores its element, which would leave that
          // reference pointing at a disposed node.
          const id = [...ids].reverse().find((x) => p.standIns.has(x));
          if (!id) return out;
          const e = ed.state.placements.get(id);
          const s = p.standIns.get(id);
          // a click on baked geometry must resolve to the element behind it
          out.standOwner = ed.ownerIdOf(s.meshes[0]);
          out.standPickable = s.meshes.every((m) => m.isPickable);
          // ...and the element's own meshes must be off, or the two z-fight
          out.standSwapped = ownKitMeshes(e).every((m) => !m.isEnabled())
            && s.node.isEnabled();
          out.why = { id, nodeOn: e.node.isEnabled(), standOn: s.node.isEnabled(),
            own: e.node.getChildMeshes().map((m) => `${m.name}:${m.isEnabled()}`),
            veil: ed.state.hidden.get(id) || null, chunk: e.chunk };
          const frame = () =>
            new Promise((r) => ed.state.scene.onAfterRenderObservable.addOnce(r));
          const regularLightState = () => {
            const preview = b.bakedPreview();
            return preview.lightSpecs.map((spec, i) => ({
              chunk: spec.chunk,
              clustered: spec.r.clustered,
              enabled: preview.lights[i].isEnabled(),
            })).filter((spec) => !spec.clustered);
          };
          const cameraPosition = ed.state.camera.position.clone();
          out.nearestAtStart = p.nearestChunk;
          out.regularAtStart = regularLightState();
          ed.state.camera.position.copyFrom(new BABYLON.Vector3(16, 2, 0));
          await frame();
          out.nearestAfterMove = b.bakedPreview().nearestChunk;
          out.regularAfterMove = regularLightState();
          ed.state.camera.position.copyFrom(cameraPosition);
          await frame();

          // World bounds of what is actually drawn. The whole point: a
          // stand-in whose *node* matrix matches the element can still render
          // mirrored, because the baked vertices live in glTF's right-handed
          // frame and only come out right under the reflection the loader left
          // above them. Only the geometry can answer for that.
          const worldBox = (meshes) => {
            const lo = [Infinity, Infinity, Infinity];
            const hi = [-Infinity, -Infinity, -Infinity];
            for (const m of meshes) {
              if (!m.getTotalVertices?.()) continue;
              m.computeWorldMatrix(true);
              const bb = m.getBoundingInfo().boundingBox;
              const a2 = bb.minimumWorld.asArray(), b2 = bb.maximumWorld.asArray();
              for (let i = 0; i < 3; i++) {
                lo[i] = Math.min(lo[i], a2[i]); hi[i] = Math.max(hi[i], b2[i]);
              }
            }
            return Number.isFinite(lo[0]) ? lo.concat(hi) : null;
          };
          const boxGap = (e2, s2) => {
            const a2 = worldBox(ownKitMeshes(e2)), b2 = worldBox(s2.meshes);
            if (!a2 || !b2) return null;
            return Math.max(...a2.map((v, i) => Math.abs(v - b2[i])));
          };

          // Every stand-in, not just the one this block goes on to drag: a
          // mirrored module whose silhouette happens to be symmetric hides the
          // fault, so a single sample proves nothing.
          //
          // The frame is not optional. Stand-ins are synced before a render,
          // and until that happens they are still sitting exactly where the
          // file drew them - which is right by construction. Measuring here
          // without rendering first only ever confirms the exporter.
          await frame();
          out.standOffset = 0;
          out.standWorst = 0;
          out.standWorstId = null;
          for (const [sid, st] of p.standIns) {
            const el = ed.state.placements.get(sid);
            if (!el) continue;
            const gap = boxGap(el, st);
            if (gap === null) continue;
            if (gap > 0.01) out.standOffset++;
            if (gap > out.standWorst) { out.standWorst = +gap.toFixed(3); out.standWorstId = sid; }
          }

          e.node.position.x += 2;
          e.node.rotation.y += 0.6;
          await frame();
          // Dragging must carry the stand-in's geometry with it - still landing
          // on the element, still the right way round.
          out.standTrackGap = boxGap(e, s);
          out.standTracks = out.standTrackGap !== null && out.standTrackGap < 0.01;
          out.standDriftMoved = b.bakedDrift().moved;
          e.node.position.x -= 2;
          e.node.rotation.y -= 0.6;
          await frame();
          // Deleting must take the stand-in down with it, and undo must find
          // the *rebuilt* node - the map is keyed by id for exactly this.
          ed.pushUndo();
          ed.removePlacement(id);
          out.standGone = !s.node.isEnabled() && b.bakedDrift().gone === 1;
          await ed.undo();
          await frame();
          const back = ed.state.placements.get(id);
          out.standRestored = !!back && back.node !== e.node && s.node.isEnabled()
            && b.bakedDrift().gone === 0;

          // ---- the runtime lamps answer to the inspector --------------------
          // The bug this guards: the preview built its lamps from the LIGHT_*
          // extras in ship_baked.glb, which is the bake's record of them. That
          // is frozen at bake time, so turning Range or Intensity up in the
          // inspector changed the authored light and nothing on screen - and
          // the runtime half is precisely the half no bake consumes, so there
          // was no re-bake that would have shown it either.
          const lit2 = await import("/js/lights.js");
          const spec = b.bakedPreview().lightSpecs?.[0];
          if (spec) {
            const lamp = b.bakedPreview().lights[0];
            const lid = spec.light.id;
            const original = {
              type: spec.r.type,
              range: spec.r.range,
              intensity: spec.r.intensity,
              angle: spec.r.angle,
              color: [...spec.r.color],
            };
            const changedColor = [0.2, 0.4, 0.6];
            lit2.setLightPart(lid, "runtime", {
              range: 41.5, intensity: 6.25, color: changedColor,
            });
            const changed = b.bakedPreview().lights[0];
            out.lampRange = changed.range;
            out.lampIntensity = changed.intensity;
            out.lampColor = changed.diffuse.asArray().map((v) => +v.toFixed(3));
            // Poked in place, not rebuilt: a slider drag emits on every tick,
            // and disposing a light dirties every material it touched.
            out.lampReused = b.bakedPreview().lights[0] === lamp;
            // ...but a change of shape has to rebuild, because the constructor
            // and `includedOnlyMeshes` are not settable after the fact.
            const sig = b.bakedPreview().lightSig;
            lit2.setLightPart(lid, "runtime",
              { type: original.type === "point" ? "spot" : "point" });
            out.lampRebuilt = b.bakedPreview().lightSig !== sig;
            lit2.setLightPart(lid, "runtime", { type: "spot", angle: 37 });
            out.lampCone = +b.bakedPreview().lights[0].angle.toFixed(4);
            lit2.setLightPart(lid, "runtime", original);
            // The lamp rides its owner, and the stand-ins beside it follow a
            // drag every frame - a lamp left behind would light the place the
            // fitting used to be.
            const owner = ed.state.placements.get(spec.light.owner);
            const at = b.bakedPreview().lights[0].position?.clone();
            if (owner && at) {
              owner.node.position.x += 3;
              await frame();
              out.lampFollowed = +(b.bakedPreview().lights[0].position.x - at.x).toFixed(2);
              owner.node.position.x -= 3;
              await frame();
            }
            // And it must actually be bound to the props it was built for.
            const named = b.bakedPreview().lights.map((l) => l.name);
            const preview = b.bakedPreview();
            const sourceMeshes = [...new Set(preview.dynamic.map((m) =>
              m._sourceMesh || m))];
            out.lampBound = sourceMeshes.filter((m) =>
              (m.lightSources || []).some((l) => named.includes(l.name))).length;

            // The live regression: L0010 is clustered and must affect the
            // dynamic P0306 barrel when its intensity changes. Instances
            // delegate lightSources to their source mesh, so inspect both the
            // source binding and the compiled effect rather than an instance's
            // private list.
            const target = preview.dynamic.find((m) =>
              m.name === "Barrel2_primitive0");
            const targetSource = target?._sourceMesh || target;
            target?.computeWorldMatrix(true);
            const clustered = preview.lightSpecs.find((s) =>
              s.light.id === "L0010" && s.r.clustered);
            out.clusteredBound = !!preview.clusteredContainer
              && !!targetSource?.lightSources?.includes(preview.clusteredContainer);
            out.clusteredShader = false;
            out.l0010Target = !!clustered && !!target;
            out.lightIntensityTarget = clustered?.light.id || null;
            if (clustered && target) {
              const camera = ed.state.camera;
              const cameraPosition = camera.position.clone();
              const cameraRotation = camera.rotation.clone();
              const targetMeshes = preview.dynamic.filter((m) =>
                m.name.startsWith("Barrel2_"));
              const visibleMeshes = new Set([
                ...targetMeshes,
                ...targetMeshes.map((m) => m._sourceMesh).filter(Boolean),
              ]);
              const visibility = new Map(preview.root.getChildMeshes()
                .map((m) => [m, m.visibility]));
              for (const [mesh] of visibility) {
                mesh.visibility = visibleMeshes.has(mesh) ? 1 : 0;
              }
              target.computeWorldMatrix(true);
              const center = target.getBoundingInfo().boundingBox.centerWorld.clone();
              const frame = () =>
                new Promise((r) => ed.state.scene.onAfterRenderObservable.addOnce(r));
              const pixels = async () => {
                await frame();
                await frame();
                const canvas = ed.state.engine.getRenderingCanvas();
                return ed.state.engine.readPixels(0, 0, canvas.width, canvas.height);
              };
              camera.position.copyFrom(center.add(new BABYLON.Vector3(0, 0.8, 3)));
              camera.setTarget(center);
              const beforePixels = await pixels();
              // Parallel shader compilation may finish after the first frame
              // at a newly focused camera, so allow the target material to
              // settle before inspecting its generated source.
              await new Promise((r) => setTimeout(r, 1200));
              const targetEffect = targetSource.material?._activeEffect;
              out.clusteredShader = !!targetEffect
                && String(targetEffect.defines).includes("CLUSTLIGHT_SLICES")
                && String(targetEffect._fragmentSourceCode).includes("CLUSTLIGHT");
              const originalIntensity = clustered.r.intensity;
              lit2.setLightPart(clustered.light.id, "runtime", {
                intensity: originalIntensity * 4,
              });
              const afterPixels = await pixels();
              let pixelDelta = 0;
              let changedPixels = 0;
              for (let i = 0; i < beforePixels.length; i++) {
                const delta = Math.abs(beforePixels[i] - afterPixels[i]);
                pixelDelta += delta;
                if (delta > 2) changedPixels++;
              }
              out.l0010PixelDelta = pixelDelta;
              out.l0010ChangedPixels = changedPixels;
              lit2.setLightPart(clustered.light.id, "runtime", {
                intensity: originalIntensity,
              });
              for (const [mesh, value] of visibility) mesh.visibility = value;
              camera.position.copyFrom(cameraPosition);
              camera.rotation.copyFrom(cameraRotation);
              await frame();
            }
          }
          return out;
        })()),
      };
      await b.setBakedPreview(false);
      return {
        ...on,
        offRoot: !!ed.state.scene.getTransformNodeByName("BAKED_PREVIEW"),
        offAuthored: anyPlacement
          ? ownKitMeshes(anyPlacement).every((m) => m.isEnabled()) : null,
        offRigEnabled: ["hemi", "hemiUp", "key", "fill"].every((name) =>
          ed.state.scene.getLightByName(name)?.isEnabled() === true),
        offPreview: b.bakedPreview(),
        texBefore,
        texAfter: ed.state.scene.textures.length,
        glbCachedAfter: glbCached(),
      };
    });
    const preFail = [];
    if (preview.served !== 200) preFail.push(`/export/ serves ${preview.served}`);
    if (!preview.lit) preFail.push("the preview lit no meshes");
    if (preview.maps !== maps.length) {
      preFail.push(`${preview.maps} lightmap(s) bound, ${maps.length} baked`);
    }
    // TEXCOORD_1 and nothing else: sampling the kit's own UVs would show the
    // lightmap tiled across every wall panel.
    if (!preview.uv.every((u) => u === 1)) preFail.push(`uv set ${preview.uv.join()}`);
    if (!preview.level.every((l) => l >= 1)) preFail.push(`level ${preview.level.join()}`);
    if (!preview.rig) preFail.push("the editor's light rig was left on top of the bake");
    if (!preview.rigDisabled) preFail.push("Baked mode left editor rig lights enabled");
    // A prop with no UV2 got no lightmap, so an analytic light is the only
    // thing that can show it at all: without one it renders as a silhouette.
    if (preview.dynamic > 0 && preview.lamps === 0) {
      preFail.push(`${preview.dynamic} dynamic prop(s) and not one runtime lamp over them`);
    }
    if (preview.dynamic > 0 && preview.dynamicVisible !== preview.dynamic) {
      preFail.push(`${preview.dynamic - preview.dynamicVisible} dynamic prop mesh(es) disappeared`);
    }
    if (preview.clusteredExpected > 0
      && (preview.clusteredContainer !== "ClusteredLightContainer"
        || preview.clusteredActual !== preview.clusteredExpected)) {
      preFail.push(`clustered lamps were not aggregated (${preview.clusteredActual}/`
        + `${preview.clusteredExpected})`);
    }
    const regularAtStart = preview.regularAtStart || [];
    const regularAfterMove = preview.regularAfterMove || [];
    if (regularAtStart.length
      && (preview.nearestAtStart !== "CH00_Storage"
        || regularAtStart.some((light) =>
          light.enabled !== (light.chunk === "CH00_Storage")))) {
      preFail.push("regular baked-preview lights were not limited to the nearest start chunk");
    }
    if (regularAfterMove.length
      && (preview.nearestAfterMove !== "CH01_CorridorA"
        || regularAfterMove.some((light) =>
          light.enabled !== (light.chunk === "CH01_CorridorA")))) {
      preFail.push("regular baked-preview lights did not follow the nearest chunk");
    }
    if (preview.maxPreviewLights > 4) {
      preFail.push(`preview materials use ${preview.maxPreviewLights} regular light slots`);
    }
    if (!preview.dynamicAllHaveNoUv2) {
      preFail.push("a dynamic prop was classified with UV2");
    }
    if (!preview.bakedAnalyticLightingDisabled) {
      preFail.push("a baked material still accepts analytic runtime lights");
    }
    if (!preview.clusteredContainerUnscoped) {
      preFail.push("the clustered container is mesh-scoped and can miss dynamic instances");
    }
    if (!preview.regularScopesDynamicOnly) {
      preFail.push("a regular runtime light reaches beyond dynamic meshes");
    }
    // The baked lightmap must multiply the PBR result, while the normal PBR
    // path remains active so the scene environment can reflect on metals.
    if (!preview.composited) preFail.push("the lightmap is not multiplied into the albedo");
    if (!preview.reflection?.controls) {
      preFail.push("the baked reflection controls are missing");
    }
    if (!preview.reflection?.bakedMaterial || !preview.reflection?.authoredMaterial) {
      preFail.push("the baked reflection test found no materials");
    }
    if (!preview.reflection?.aaOff || !preview.reflection?.aaOn) {
      preFail.push("the Specular AA control did not update baked materials");
    }
    if (!preview.reflection?.roughnessChanged) {
      preFail.push("the Reflection roughness control did not update baked materials");
    }
    if (!preview.reflection?.authoredStable) {
      preFail.push("reflection controls changed an authored material");
    }
    if (!preview.reflection?.serializedStable) {
      preFail.push("reflection controls changed exported editor state");
    }
    if (!preview.survivesUnlitOn || !preview.survivesUnlitOff) {
      preFail.push("the viewport's unlit toggle clobbered baked environment lighting");
    }
    if (!preview.environment.texture || !(preview.environment.intensity > 0)) {
      preFail.push("the baked preview has no active environment lighting");
    }
    // The bug this guards: the dynamic props are kept out of the atlas on
    // purpose, so they have no TEXCOORD_1 and dressing them in a lightmap
    // material renders them black.
    if (!preview.everyLitMeshHasUv2) {
      preFail.push("a mesh with no UV2 was given a lightmap material");
    }
    // The bug this guards: `Material.clone` deep-copies every texture slot, so
    // each preview material owns private wrappers that the glb's AssetContainer
    // never sees. Left behind they pin the glb's InternalTextures in the
    // engine's url cache, and the next load - after a fresh bake - is handed
    // the previous export's images instead of the new ones.
    if (preview.texAfter !== preview.texBefore) {
      preFail.push(`the preview leaked ${preview.texAfter - preview.texBefore} texture(s)`);
    }
    if (preview.glbCachedAfter !== 0) {
      preFail.push(`${preview.glbCachedAfter} baked-glb texture(s) pinned in the engine cache`);
    }
    if (preview.authoredHidden !== true) preFail.push("the authored ship is still drawn");
    if (!preview.ownKit) preFail.push("no placement mesh carries placementRoot");
    if (!preview.rootInScene) preFail.push("no BAKED_PREVIEW root in the scene");
    if (preview.offRoot) preFail.push("turning it off left the baked ship behind");
    if (preview.offAuthored !== true) preFail.push("turning it off left the ship hidden");
    if (!preview.offRigEnabled) preFail.push("turning Baked mode off did not restore rig lights");
    if (preview.offPreview !== null) preFail.push("turning it off kept the preview");

    // ---- editing through the bake -----------------------------------------
    // The bug this guards: the preview used to disable the whole authored ship,
    // so nothing could be picked or moved while it was up. Now each exported
    // node stands in for its element, and has to answer for it.
    if (preview.standIns !== preview.ship) {
      preFail.push(`${preview.standIns} stand-in(s) for ${preview.ship} element(s)`);
    }
    if (preview.standDrift?.moved || preview.standDrift?.gone) {
      preFail.push("an untouched ship reports drift from its own bake: "
        + JSON.stringify(preview.standDrift));
    }
    if (preview.standIns > 0) {
      if (!preview.standOwner) preFail.push("a pick on baked geometry owns nothing");
      if (!preview.standPickable) preFail.push("the baked geometry is not pickable");
      // Both drawn at once is a z-fight over the whole vessel.
      if (!preview.standSwapped) preFail.push("the element and its stand-in are both drawn");
      // The bug this guards: the stand-in was placed on the element's own world
      // matrix, which throws away the reflection the glTF loader hangs off
      // `__root__` to turn glTF's right-handed data into Babylon's left-handed
      // scene. The node then sits in the right place with its basis mirrored,
      // so every asymmetric module renders flipped about its own origin - read
      // on screen as "some walls are in the wrong place".
      if (preview.standOffset) {
        preFail.push(`${preview.standOffset} stand-in(s) not on their element`
          + ` (worst ${preview.standWorstId} by ${preview.standWorst}m)`);
      }
      if (!preview.standTracks) {
        preFail.push("the stand-in did not follow the element it stands for"
          + ` (off by ${preview.standTrackGap}m)`);
      }
      if (preview.standDriftMoved !== 1) {
        preFail.push(`a moved element counted as ${preview.standDriftMoved} moved`);
      }
      if (!preview.standGone) preFail.push("deleting an element left its stand-in on screen");
      // Undo rebuilds every placement node, which is why the map is keyed by id.
      if (!preview.standRestored) preFail.push("undo did not bring the stand-in back");
      // The runtime lamps are the editor's live record, not the bake's copy.
      if (preview.lampRange !== undefined) {
        if (preview.lampRange !== 41.5 || preview.lampIntensity !== 6.25) {
          preFail.push("editing a runtime lamp did not reach the preview"
            + ` (range ${preview.lampRange}, intensity ${preview.lampIntensity})`);
        }
        if (JSON.stringify(preview.lampColor) !== JSON.stringify([0.2, 0.4, 0.6])) {
          preFail.push(`editing a runtime lamp colour did not reach the preview (${preview.lampColor})`);
        }
        if (Math.abs(preview.lampCone - (37 * Math.PI / 180)) > 1e-4) {
          preFail.push(`editing a runtime lamp cone did not reach the preview (${preview.lampCone})`);
        }
        if (!preview.lampReused) preFail.push("a scalar lamp edit rebuilt the light");
        if (!preview.lampRebuilt) preFail.push("changing a lamp's type did not rebuild it");
        if (preview.lampFollowed !== 3) {
          preFail.push(`a dragged element left its lamp behind (${preview.lampFollowed}m of 3)`);
        }
        if (preview.dynamic > 0 && !preview.lampBound) {
          preFail.push("no dynamic prop is lit by a preview lamp");
        }
        if (preview.l0010Target
          && (!preview.clusteredBound || !preview.clusteredShader)) {
          preFail.push("P0306 did not compile with the clustered L0010 light");
        }
        if (preview.l0010Target
          && !(preview.l0010PixelDelta > 10000
            && preview.l0010ChangedPixels > 500)) {
          preFail.push("raising L0010 intensity did not change P0306 pixels");
        }
      }
    }
    console.log(`baked view   : ${preFail.length ? preFail.join(" | ") : "ok"},`
      + ` ${preview.lit} mesh(es), ${preview.maps} map(s)`);
    if (preFail.length) {
      console.log("  stand-in diag:", JSON.stringify({
        flags: preview.flags, why: preview.why, trackGap: preview.standTrackGap,
        standIns: preview.standIns, ship: preview.ship, drift: preview.standDrift,
      }));
    }
    if (preFail.length) {
      errors.push("baked preview: " + preFail.join(" | "));
      assertFailures.push("baked preview: " + preFail.join(" | "));
    }

    // ---- and a Blender session can send its light powers back -------------
    // A live session exists to settle on a wattage by eye. That number is worth
    // nothing if it stays in Blender, so the session writes it out and the
    // editor reads it in. No Blender is started here: what is under test is the
    // file contract between the two, which is the part that can silently rot.
    const powersFile = path.join(process.env.SHIP_EXPORT_DIR, "light_powers.json");
    const before = await page.evaluate(async () =>
      (await fetch("/api/light-powers")).json());
    const target = await page.evaluate(async () =>
      [...(await import("/js/editor.js")).state.lights.keys()][0]);
    fs.writeFileSync(powersFile, JSON.stringify({
      multiplier: 3, env: 1,
      watts: { [target]: 123.5, L9999_gone: 7 },
    }));
    const after = await page.evaluate(async (id) => {
      const m = await import("/js/main.js");
      const lt = await import("/js/editor.js");
      const was = lt.state.lights.get(id).bake.watts;
      await m.syncLightPowers();
      const light = lt.state.lights.get(id);
      return { was, now: light.bake.watts, runtime: light.runtime.intensity };
    }, target);
    fs.rmSync(powersFile, { force: true });

    const powFail = [];
    if (before?.watts != null) powFail.push("a missing file did not read as empty");
    if (after.now !== 123.5) powFail.push(`${target} is ${after.now} W, not 123.5`);
    if (after.was === after.now) powFail.push("the watts did not actually change");
    // The session has no opinion about the clustered lights the runtime draws.
    if (after.runtime !== 1) powFail.push(`the runtime half moved to ${after.runtime}`);
    console.log(`light powers : ${powFail.length ? powFail.join(" | ") : "ok"},`
      + ` ${after.was} W -> ${after.now} W`);
    if (powFail.length) {
      bakeFailures.push("light powers: " + powFail.join(" | "));
      errors.push("light powers: " + powFail.join(" | "));
    }
  }
}

console.log("\nerrors:", errors.length ? [...new Set(errors)].join("\n") : "(none)");
await browser.close();
// Losing the ability to recover work - or the viewpoint you saved from - is
// worth failing the run over. So is anything in `assertFailures`, which is
// what the baked-preview block asserts: a preview that leaks textures, hides
// the wrong ship, or draws its stand-ins in the wrong place is a broken
// editor, not a warning.
if (rotationFailures.length || benchFailures.length || glbFailures.length
  || spaceFailures.length || bakeFailures.length
  || assertFailures.length) process.exit(1);
