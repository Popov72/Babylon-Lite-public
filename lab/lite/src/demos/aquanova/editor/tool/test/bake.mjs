// The Blender half of the lightmap pipeline.
//
// bake_lightmaps.py is the other end of the glTF `extras` contract: the editor
// writes lights as bare nodes, Blender rebuilds them as Cycles Area lights.
// Nothing in the browser can check that, so this suite drives the real Blender
// against the real ship.glb that e2e.mjs just exported into the scratch folder.
//
// Blender is not a dependency of the editor - it is a tool the pipeline shells
// out to - so a machine without it skips this suite rather than failing it.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, "..");
const SCRIPT = path.join(ROOT, "bake_lightmaps.py");
const EXPORT = process.env.SHIP_EXPORT_DIR;
const GLB = path.join(EXPORT || "", "ship.glb");

const CANDIDATES = [
  process.env.BLENDER,
  "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe",
  "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe",
  "/usr/bin/blender",
  "/Applications/Blender.app/Contents/MacOS/Blender",
].filter(Boolean);

const blender = CANDIDATES.find((p) => fs.existsSync(p));
if (!blender) {
  console.log("blender      : not installed - skipping the bake suite");
  console.log("  (set BLENDER=<path to blender> to run it)");
  process.exit(0);
}
if (!fs.existsSync(GLB)) {
  console.log(`blender      : no ship.glb at ${GLB} - skipping`);
  process.exit(0);
}
console.log("blender      :", blender);

const results = [];
function check(label, ok, detail = "") {
  results.push(`${ok ? "PASS" : "FAIL"}  ${label.padEnd(60)} ${detail}`);
}

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "aquanova-bake-"));

function bake(extra = []) {
  return run(["--no-bake", ...extra]);
}

/** The same run with the renderer actually let loose. */
function bakeReal(extra = []) {
  return run(extra);
}

function run(extra) {
  const out = execFileSync(blender, [
    "-b", "--factory-startup", "--python", SCRIPT, "--",
    "--glb", GLB, "--out", path.join(scratch, "lightmaps"),
    ...extra,
  ], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, timeout: 900000 });
  const line = out.split(/\r?\n/).find((l) => l.startsWith("BAKE_REPORT "));
  if (!line) throw new Error(`no report in blender output:\n${out.slice(-2000)}`);
  return JSON.parse(line.slice("BAKE_REPORT ".length));
}

/** A 1x1 PNG, so the world test does not need the real 4 MB star field. */
function tinyPng() {
  return Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    + "hQGAhKmMIQAAAABJRU5ErkJggg==", "base64");
}

/** The JSON chunk of a .glb, which is all these checks read. */
function readGlbJson(file) {
  const buf = fs.readFileSync(file);
  if (buf.readUInt32LE(0) !== 0x46546c67) throw new Error(`${file} is not a glb`);
  return buf.toString("utf8", 20, 20 + buf.readUInt32LE(12));
}

// ---- the ship arrives, and the lights arrive with it ------------------------
// The glb e2e.mjs wrote holds two chunks and three lights: a rectangle that
// also exists at runtime, a bake-only disk, and the one a real kit lamp module
// seeded for itself. All three must be rebuilt - the runtime half is none of
// Blender's business.
const report = bake([]);
console.log("bake report  :", JSON.stringify({
  chunks: report.chunks, placements: report.placements, lights: report.lights,
}));

check("blender reads the ship the editor exported",
  report.placements > 0 && report.chunks.length === 2,
  `${report.placements} placement(s) in ${report.chunks.join("+")}`);
check("every light node is found by its extras, not by its name",
  report.lights.found === 3, `${report.lights.found} found`);
check("and each one is rebuilt as a Cycles area light",
  report.lights.built.length === 3 && report.lights.skipped.length === 0,
  `built ${report.lights.built.join("+")}, skipped ${report.lights.skipped.join("+") || "(none)"}`);

// ---- the axis contract, which is the whole reason -Y was chosen ------------
// A light emits along its own -Y in the editor. The glTF importer turns Y-up
// into Z-up, which lands that on Blender's -Z - the axis an Area light emits
// along - so a ceiling panel authored with no rotation must come out pointing
// at the floor with no fix-up anywhere. Nothing else in the pipeline can see
// this, and getting it wrong bakes a ship lit through its ceiling.
const rect = report.lamps.find((l) => l.size.join() === "1.2,0.2");
const disk = report.lamps.find((l) => l.shape === "DISK");
const down = (l) => l && Math.abs(l.aim[0]) < 1e-3 && Math.abs(l.aim[1]) < 1e-3
  && l.aim[2] < -0.999;
check("a ceiling panel comes out of the importer pointing at the floor",
  down(rect) && down(disk),
  report.lamps.map((l) => `${l.shape} aims [${l.aim}]`).join(", "));
// The aim check above is only worth anything if the matrix it was read from is
// the evaluated one. Parenting does not move a lamp until the depsgraph runs,
// and an unevaluated matrix is the identity - whose -Z is [0,0,-1], so "points
// at the floor" passed for every lamp in the ship while the report placed them
// all at the origin. A lamp that is somewhere is the proof that it is aimed.
check("and it is placed where its empty is, not left at the origin",
  report.lamps.length > 0
    && report.lamps.every((l) => l.world.some((c) => Math.abs(c) > 1e-3)),
  report.lamps.map((l) => `${l.id} at [${l.world}]`).join(", "));
check("its rectangle keeps the two sizes it was authored with",
  rect?.size.join() === "1.2,0.2" && disk?.size[0] === 0.8,
  `${rect?.size.join("x")} and ${disk?.size.join("x")}`);
check("and the power is the total watts the editor asked for",
  rect?.watts === 55 && rect?.spread === 180,
  `${rect?.watts} W over ${rect?.spread} deg`);
check("each lamp knows the chunk it belongs to, so the bake can go room by room",
  report.lamps.every((l) => report.chunks.includes(l.chunk)),
  report.lamps.map((l) => `${l.id}@${l.chunk}`).join(", "));

// ---- the bake is diffuse, and colour-free ----------------------------------
// What is wanted is how much light reaches a surface, not what that surface
// looks like: leaving the colour pass in multiplies the albedo into the map and
// then again at runtime, and every wall comes out twice as brown as it should.
check("it bakes diffuse direct + indirect, with no colour in the map",
  report.cycles.bakeType === "DIFFUSE" && report.cycles.passes.direct
    && report.cycles.passes.indirect && report.cycles.passes.color === false,
  JSON.stringify(report.cycles));

// ---- the star field outside, and the windows it comes through --------------
// A sealed hull lit through a few small panes is the worst case for a path
// tracer: fire rays at random and almost none of them find a window. A Cycles
// portal is the fix - it marks the opening so the sampler aims at it - and it
// only makes sense on a door onto space. The corridor window e2e.mjs authored
// must become one; the door between the two rooms must not.
const sky = report.portals.find((p) => p.chunk === "CH01_CorridorA");
check("a door onto space becomes a portal, and a door between rooms does not",
  report.portals.length === 1 && !!sky,
  report.portals.map((p) => `${p.id}@${p.chunk}`).join(", ") || "(none)");
check("the portal covers the opening the door left",
  !!sky && sky.size[0] > 0.1 && sky.size[1] > 0.1,
  sky ? `${sky.size.join(" x ")} m at [${sky.world}]` : "no portal");
// Backwards is not a smaller win, it is a loss: it guides every sample out into
// space, and the room bakes noisier than with no portal at all.
check("and it faces into the room, not out at the stars",
  !!sky && sky.inward > 0, sky ? `aim [${sky.aim}], inward ${sky.inward}` : "no portal");

// The equirect is what the runtime shows outside; without one the interior is
// lit by its own lamps against black, never against Blender's default grey -
// a uniform grey world is an ambient term the runtime does not have.
check("with no skybox the world is black, so nothing invents ambient light",
  report.world.skybox === null, JSON.stringify(report.world));

const equirect = path.join(scratch, "sky.png");
fs.writeFileSync(equirect, tinyPng());
const lit = bake(["--skybox", equirect, "--env-strength", "0.6"]);
const samePath = (a, b) => !!a && !!b
  && path.resolve(a).toLowerCase() === path.resolve(b).toLowerCase();
check("an equirect is wired straight into the world, at the strength asked for",
  samePath(lit.world.skybox, equirect) && lit.world.strength === 0.6,
  JSON.stringify(lit.world));

let noSky = false;
try {
  bake(["--skybox", path.join(scratch, "not-here.png")]);
} catch {
  noSky = true;
}
check("a skybox path that is not there stops the bake", noSky,
  noSky ? "refused" : "bakes in the dark, quietly");

// ---- an emissive lamp that also carries a light is counted once ------------
// The kit's lamp modules are an emissive M_Light strip, and the editor seeds an
// area light onto that same strip. Bake both and every fixture is counted
// twice: once with the watts that were authored, once with an emission nobody
// tuned. The area light wins - it is what the runtime lights with.
const put = report.emission;
check("a lamp that carries an area light stops glowing, so it is not counted twice",
  put.materials.length >= 1 && put.materials.every((m) => m.was > 0) && put.meshes >= 1,
  `${put.meshes} mesh(es), ${put.materials.map((m) => `${m.material}@${m.was}`).join(", ")
  || "(nothing suppressed)"}`);
// Kit materials are shared across every placement of a module, so zeroing the
// original would put out every other lamp in the ship - including the ones with
// no area light, which are exactly the ones that still have to glow.
check("and it is a copy that is dimmed, not the shared kit material",
  put.materials.every((m) => !m.material.endsWith("_NoEmit")),
  put.materials.map((m) => m.material).join(", ") || "(none)");
const kept = bake(["--keep-emission"]);
check("--keep-emission leaves it alone, for comparing the two bakes",
  kept.emission.materials.length === 0 && kept.emission.meshes === 0,
  JSON.stringify(kept.emission));

// ---- one packed UV2 atlas per chunk ----------------------------------------
// The runtime binds one lightmap per chunk, not one per wall panel, so every
// mesh of a chunk is unwrapped together in one multi-object session and packed
// into a single shared 0-1 square. Per chunk and not per ship is what lets one
// room be re-baked without re-rendering the whole vessel.
check("each chunk gets its own atlas, in a second UV layer",
  report.atlases.length === 2
    && report.atlases.every((a) => a.layer === "UV2" && a.unwrapped === a.meshes
      && a.meshes > 0),
  report.atlases.map((a) => `${a.chunk}: ${a.unwrapped}/${a.meshes}`).join(", "));
// The runtime samples TEXCOORD_1, which is layer 1 and nothing else. A few kit
// meshes arrive with no UVs at all, and on those the atlas would otherwise land
// on TEXCOORD_0 and be sampled as if it were the kit's own texture atlas.
check("and it is layer 1 on every mesh, which is the set TEXCOORD_1 comes from",
  report.atlases.every((a) => a.misplaced.length === 0),
  report.atlases.flatMap((a) => a.misplaced).join(", ") || "none misplaced");
// A modular kit is the same wall panel forty times over, and Blender keeps all
// forty pointing at ONE mesh datablock. That is free until something wants a
// PER-OBJECT vertex attribute, which is exactly what an atlas coordinate is:
// share the datablock and the forty panels share one island, so Cycles bakes
// each of them over the last and they all end up wearing the same lighting.
// The fixture places its floor module four times, so this has teeth.
check("and no two objects share one datablock, which would share one island",
  report.atlases.every((a) => a.shared === 0) && report.atlases.some((a) => a.unshared > 0),
  report.atlases.map((a) => `${a.chunk}: ${a.unshared} unshared, ${a.shared} left`).join(", "));
// De-instancing multiplies the island count, and every island pays the margin
// around its whole perimeter, so the bill lands on atlas coverage: the very
// first de-instanced bake packed a corridor into four percent of its square,
// which is a silent 5x resolution loss that nothing else here would have
// caught. The floor is deliberately low - the fixture is two tiny rooms in a
// 0-1 square and will never pack densely - but four percent has to fail.
check("and the islands cover the atlas, rather than mostly gutter",
  report.atlases.every((a) => a.coverage >= 0.15),
  report.atlases.map((a) => `${a.chunk}: ${(a.coverage * 100).toFixed(1)}%`).join(", "));
// The kit's own atlas is the first layer; overwriting it would untexture the
// ship, which is the kind of bug that only shows up in the runtime.
const raw = bake(["--no-unwrap"]);
check("and it is added, never written over the kit's own UVs",
  raw.atlases.every((a) => a.unwrapped === 0),
  raw.atlases.map((a) => `${a.chunk}: ${a.unwrapped}`).join(", "));

// ---- what the runtime moves or melts is not baked into the walls ------------
// A door leaf that slides open would leave its own shadow painted across the
// floor it used to cover, and a crate that liquefies when shot was never the
// shape the map was rendered for. The flags live in the manifest, not in the
// glb - `behaviors` says which behaviour is dynamic or liquefiable, `entities`
// says which node carries it - so this is the runtime's own rule, read back.
const manFile = path.join(EXPORT, "ship_manifest.json");
const man = JSON.parse(fs.readFileSync(manFile, "utf8"));
let variant = 0;
/** The same ship with one field changed, so what matters can be isolated. */
function withManifest(mutate, extra = []) {
  const copy = JSON.parse(JSON.stringify(man));
  mutate(copy);
  const file = path.join(scratch, `manifest-${variant++}.json`);
  fs.writeFileSync(file, JSON.stringify(copy));
  return bake(["--manifest", file, ...extra]);
}
const meshTotal = (r) => r.atlases.reduce((n, a) => n + a.meshes, 0);
const staticOnly = withManifest(() => {});
const everything = withManifest((m) => {
  for (const b of Object.values(m.behaviors || {})) { delete b.dynamic; delete b.liquefiable; }
});
check("a prop the runtime moves is kept out of the atlas and the bake",
  staticOnly.unbaked.length === 1
    && meshTotal(staticOnly) < meshTotal(everything),
  `${staticOnly.unbaked.join()}: ${meshTotal(staticOnly)}`
    + ` of ${meshTotal(everything)} mesh(es) baked`);

// A liquefiable prop is replaced by a fluid the moment it is hit, and is a
// rigid body the player can shove before that, so a map baked onto its rest
// pose is wrong twice over. The bug this guards: the rule read `dynamic` only,
// and `liquefiable` deliberately does NOT imply `dynamic` in this editor.
// Same node, same everything - only which flag its behaviour carries.
const meltsInstead = withManifest((m) => {
  for (const b of Object.values(m.behaviors || {})) {
    if (b.dynamic) { delete b.dynamic; b.liquefiable = true; }
  }
});
check("and so is one the runtime melts, which is not the same flag",
  meltsInstead.unbaked.length === 1
    && meshTotal(meltsInstead) === meshTotal(staticOnly)
    && meshTotal(meltsInstead) < meshTotal(everything),
  `${meltsInstead.unbaked.join()}: ${meshTotal(meltsInstead)}`
    + ` of ${meshTotal(everything)} mesh(es) baked`);

// Liquefaction spreads down `linked` - that is how a pair of door leaves comes
// apart together - and the runtime puts every linked node in `dynamicMeshes`,
// so it loses its lightmap there whatever the bake decided. One left in the
// bake would have had its shadow painted in and then be lit as if it had not.
const linked = withManifest((m) => {
  for (const [name, b] of Object.entries(m.behaviors || {})) {
    if (!b.dynamic) continue;
    delete b.dynamic;
    b.liquefiable = true;
    for (const entry of Object.values(m.entities || {})) {
      for (const a of entry.behaviors || []) if (a.name === name) a.linked = ["melts_with_me"];
    }
  }
});
check("and everything it is linked to melts with it, so that goes too",
  linked.unbaked.includes("melts_with_me")
    && linked.unbaked.length === meltsInstead.unbaked.length + 1,
  linked.unbaked.join());

// The rule reads the behaviour *definitions*, so flipping a flag changes no
// node's behaviour NAME list. The bug this guards: the chunk hash recorded
// only those names, so the stale map - with the prop's shadow still painted
// into it - was reused and the change was invisible. `everything` and
// `meltsInstead` differ by nothing else at all.
check("and flipping the flag on an existing behaviour invalidates the map",
  everything.unbaked.length === 0
    && JSON.stringify(everything.hashes) !== JSON.stringify(meltsInstead.hashes),
  `${Object.values(everything.hashes).map((h) => h.slice(0, 8)).join()}`
    + ` vs ${Object.values(meltsInstead.hashes).map((h) => h.slice(0, 8)).join()}`);

// ---- the author's override of all of the above ------------------------------
// `entities[node].bake` is the editor's Force baking setting. The rule above is
// right almost always, and these are the two cases where it is not: an emissive
// panel driven at runtime must come OUT of a map it would otherwise get, and a
// liquefiable fixture that never moves until it is destroyed must go back IN.
const anyBaked = (man.instances || [])
  .map((i) => i.node).find((n) => n && !staticOnly.unbaked.includes(n));
const forcedOut = withManifest((m) => {
  m.entities = { ...(m.entities || {}) };
  m.entities[anyBaked] = { ...(m.entities[anyBaked] || {}), bake: "exclude" };
});
check("an author can force a mesh out of the atlas without inventing a behaviour",
  forcedOut.unbaked.includes(anyBaked)
    && meshTotal(forcedOut) < meshTotal(staticOnly),
  `${anyBaked}: ${meshTotal(forcedOut)} of ${meshTotal(staticOnly)} mesh(es) baked`);

// The other direction, and the harder one: the exclusion is on the NODE, and
// the meshes are its primitives underneath, so an inclusion only works if the
// parent walk in `is_unbaked` stops at the nearest instruction rather than
// carrying the ancestor's verdict all the way down.
const forcedIn = withManifest((m) => {
  m.entities = { ...(m.entities || {}) };
  for (const node of staticOnly.unbaked) {
    m.entities[node] = { ...(m.entities[node] || {}), bake: "include" };
  }
});
check("and force one back in, beating the behaviour that dropped it",
  forcedIn.unbaked.length === 0
    && forcedIn.forcedIn.join() === [...staticOnly.unbaked].sort().join()
    && meshTotal(forcedIn) === meshTotal(everything),
  `${forcedIn.forcedIn.join()}: ${meshTotal(forcedIn)}`
    + ` of ${meshTotal(everything)} mesh(es) baked`);

// An override changes no behaviour and moves nothing, so the same trap as the
// flag flip above: without the resolved verdict in the hash, the stale map -
// with the forced-out prop's shadow still in it - would be reused. It reaches
// the room next door as well, and should: the prop was bouncing light back
// through the portal, so that map is wrong now too.
const invalidated = Object.keys(staticOnly.hashes)
  .filter((c) => forcedOut.hashes[c] !== staticOnly.hashes[c]);
check("and setting one invalidates the maps it changed, portals included",
  invalidated.length > 0
    && JSON.stringify(forcedIn.hashes) !== JSON.stringify(staticOnly.hashes),
  invalidated.join());

// ---- each room is baked at the size the editor asked for --------------------
// Samples, size and margin are per chunk, because a two-metre airlock and a
// corridor are not worth the same texels. Four sources decide it, in order: a
// command-line flag beats the chunk's own `bake`, which beats the ship's
// `bakeDefaults`, which beats the built-ins.
const roomNames = (staticOnly.chunks || []).filter((c) => c !== "__SKYBOX__");
const firstRoom = roomNames[0];
const sized = withManifest((m) => {
  m.bakeDefaults = { samples: 32, width: 256, height: 256, margin: 2 };
  for (const c of m.chunks || []) if (c.id === firstRoom) c.bake = { samples: 64, width: 128 };
});
check("a chunk's own bake settings override the ship's defaults, field by field",
  sized.plan[firstRoom]?.samples === 64 && sized.plan[firstRoom]?.width === 128
    // Untouched fields keep following the defaults rather than the built-ins.
    && sized.plan[firstRoom]?.height === 256 && sized.plan[firstRoom]?.margin === 2,
  JSON.stringify(sized.plan[firstRoom]));
check("and a room that asked for nothing gets the ship's defaults",
  roomNames.slice(1).every((n) => sized.plan[n]?.samples === 32
    && sized.plan[n]?.width === 256 && sized.plan[n]?.height === 256),
  JSON.stringify(sized.plan));

// The flag is what makes "--resolution 64" a usable "show me something now",
// and it is what this very suite bakes at, so it has to beat everything.
const forcedSize = withManifest((m) => {
  m.bakeDefaults = { samples: 32, width: 256, height: 256, margin: 2 };
  for (const c of m.chunks || []) if (c.id === firstRoom) c.bake = { samples: 64, width: 128 };
}, ["--samples", "8", "--resolution", "32"]);
check("a command-line flag beats both, for every chunk",
  Object.values(forcedSize.plan).every((s) => s.samples === 8 && s.width === 32
    && s.height === 32)
    // --resolution is the square shorthand; the margin is not part of it.
    && Object.values(forcedSize.plan).every((s) => s.margin === 2),
  JSON.stringify(forcedSize.plan[firstRoom]));

// Retuning one room must not re-render the ship: its settings go in its OWN
// hash and not its neighbours'. A neighbour's resolution cannot change how
// much light reaches this room.
const retuned = withManifest((m) => {
  for (const c of m.chunks || []) if (c.id === firstRoom) c.bake = { samples: 999 };
});
const movedHashes = Object.keys(staticOnly.hashes || {})
  .filter((n) => staticOnly.hashes[n] !== retuned.hashes[n]);
check("re-tuning one room invalidates that room's map and no other",
  movedHashes.length === 1 && movedHashes[0] === firstRoom,
  movedHashes.join() || "nothing moved");

// ---- the bake writes one HDR per chunk -------------------------------------
// Radiance HDR and not PNG: bounced light in a lit corridor runs well past 1.0,
// and clamping it here would bake in the clipping the runtime's tone mapping is
// there to do properly.
const outDir = path.join(scratch, "real");
const real = bakeReal(["--out", outDir, "--samples", "4", "--resolution", "64",
  "--baked-glb", path.join(scratch, "ship_baked.glb")]);
const images = real.baked.map((b) => b.image);
check("baking writes one lightmap per chunk, as HDR",
  real.baked.length === 2 && images.every((p) => fs.existsSync(p) && /\.hdr$/.test(p)),
  images.map((p) => `${path.basename(p)} ${fs.existsSync(p)
    ? fs.statSync(p).size : 0}b`).join(", "));
// "Not empty" used to be measured as HDR file size, which is not a measure of
// anything: a map that is mostly unbaked gutter and a map that came out black
// both compress to a few hundred bytes, so the check passed for years and then
// failed for the right reason at the wrong time. `lit` is the fraction of
// texels carrying any light at all - gutter is exactly zero - so this asks the
// question directly.
//
// The floor is low on purpose and is not a coverage check: this bake gets no
// skybox, so the fixture is a sealed hull lit by three small lamps against a
// black world, and the overwhelming majority of it is *correctly* pitch black.
// A percent is what "the lamps reached something" looks like here; zero is what
// a broken bake looks like. Atlas coverage is measured on its own, above.
check("and the map is not empty, so something actually reached the walls",
  real.baked.every((b) => b.lit > 0.005),
  real.baked.map((b) => `${b.chunk} ${(b.lit * 100).toFixed(1)}% lit`).join(", "));
// The real ship, though, is a lit room built out of the kit's metal panels, and
// on it "not empty" is nowhere near enough. A Cycles DIFFUSE bake reads the
// diffuse lobe, and a metal has none: with `Metallic` at 1.0 the pass is
// exactly zero however bright the room is. The kit drives `Metallic` from an
// ORM map that is 1.0 across every panel, so every wall, floor and ceiling
// baked to pure black while the few dielectric decals baked correctly - and the
// map still passed every check above, because the islands were packed, the UVs
// were right, the meshes were unshared and the file was not empty. On the real
// storage room this was the difference between 3.2% and 65% of the atlas lit.
//
// So it is measured as a difference, not as a level: the same bake with
// `--keep-metals` is the broken one, and it has to come out darker. An absolute
// floor would only measure how bright this particular fixture happens to be.
check("and metalness is neutralised for the bake, or metal bakes black",
  real.baked.every((b) => b.metals > 0),
  real.baked.map((b) => `${b.chunk} ${b.metals} metal input(s)`).join(", "));
const metal = bakeReal(["--keep-metals", "--force"]);
const litOf = (r) => Object.fromEntries(r.baked.map((b) => [b.chunk, b.lit]));
const flattened = litOf(real);
const metalKept = litOf(metal);
check("and leaving it alone measurably darkens the map, which is the bug",
  Object.keys(flattened).every((c) => flattened[c] > metalKept[c] * 1.25),
  Object.keys(flattened).map((c) =>
    `${c} ${(metalKept[c] * 100).toFixed(1)}% -> ${(flattened[c] * 100).toFixed(1)}%`).join(", "));
// Cycles denoises a render and never a bake: `scene.cycles.use_denoising` has
// no effect on `bpy.ops.object.bake`, and there is no denoise setting on the
// bake at all. So the grain the sample count could not resolve goes straight
// into the file, and it dominates at high resolutions - a 4096 texel integrates
// a sixteenth of the surface a 1024 texel does, so it needs sixteen times the
// samples for the same noise. Pushing the map back through the compositor's
// OpenImageDenoise node afterwards took the storage room's 4096 map from 5.26%
// texel-to-neighbour deviation down to 1.07%.
//
// It is checked as a flag rather than as a noise measurement because the
// failure mode is silent: the compositor moved out of `scene.node_tree` in
// Blender 5, and a version that cannot build the tree would go on writing
// perfectly valid, perfectly grainy maps.
check("and the map is denoised, because a Cycles bake never is",
  real.baked.every((b) => b.denoised === true),
  real.baked.map((b) => `${b.chunk} denoised=${b.denoised}`).join(", "));
const grainy = bakeReal(["--no-denoise", "--force"]);
check("and asking for the raw bake instead is honoured, and re-bakes",
  grainy.baked.every((b) => b.denoised === false)
    && grainy.baked.every((b) => !b.reused),
  grainy.baked.map((b) => `${b.chunk} denoised=${b.denoised} reused=${b.reused}`).join(", "));
// A browser texture is 8-bit, and bounced light runs past 1.0 - so the map is
// divided by its own peak on the way out and the peak is written down as the
// level the runtime multiplies back. Clamping would flatten every highlight to
// white; a fixed exposure would clip a bright room and crush a dim one.
check("each map also lands as 8-bit sRGB, with the scale it was divided by",
  real.baked.every((b) => fs.existsSync(b.png) && b.level >= 1),
  real.baked.map((b) => `${path.basename(b.png)} @ ${b.level}`).join(", "));

// A non-square atlas is the reason width and height are separate fields: a long
// corridor wastes half a square map on nothing. The risk is that the packer
// works in the 0-1 square and knows nothing about the image it will be sampled
// from, so an island packed square comes out stretched; the UVs are pre-squeezed
// by the aspect before packing to cancel exactly that, and island rotation is
// switched off because a cardinal rotation would undo the squeeze for that
// island alone.
//
// The squeeze must not cost the atlas its fill - it is a uniform scale away
// from what the packer would have done anyway - but switching rotation off
// genuinely does cost, and quite a lot: the fixture packs to about 60% square
// and about a third of that back, because a rect packer leans on 90-degree
// swaps to fit an L-shaped wall panel against its neighbour.
//
// So the question worth asking is not "what fraction is filled" - that is the
// price, and it is documented - but "does asking for four times the pixels
// actually buy texels". Coverage alone would also pass at 3%, which is what a
// broken squeeze looks like: islands shrunk into a strip along one edge.
const wideAtlas = withManifest((m) => {
  m.bakeDefaults = { samples: 4, width: 2048, height: 512, margin: 4 };
});
const squareAtlas = withManifest((m) => {
  m.bakeDefaults = { samples: 4, width: 512, height: 512, margin: 4 };
});
const squareCover = Object.fromEntries(squareAtlas.atlases.map((a) => [a.chunk, a.coverage]));
const texelGain = (a) => (a.coverage * 2048 * 512) / (squareCover[a.chunk] * 512 * 512);
check("a non-square atlas still packs, so four times the pixels buys texels",
  wideAtlas.atlases.length > 0 && wideAtlas.atlases.every((a) => texelGain(a) > 2),
  wideAtlas.atlases.map((a) =>
    `${a.chunk} ${(squareCover[a.chunk] * 100).toFixed(0)}% square`
    + ` -> ${(a.coverage * 100).toFixed(0)}% wide, ${texelGain(a).toFixed(1)}x texels`)
    .join(", "));

// And the image really comes out at the size that was asked for, which only a
// render can answer. Tiny, because this is about the dimensions and not the light.
const wideFile = path.join(scratch, "manifest-wide.json");
{
  const copy = JSON.parse(JSON.stringify(man));
  copy.bakeDefaults = { samples: 4, width: 128, height: 32, margin: 2 };
  fs.writeFileSync(wideFile, JSON.stringify(copy));
}
const wideDir = path.join(scratch, "wide");
const wide = bakeReal(["--manifest", wideFile, "--out", wideDir,
  "--baked-glb", path.join(scratch, "ship_wide.glb")]);
check("a chunk can be baked to a non-square map, at the size it asked for",
  wide.baked.length > 0
    && wide.baked.every((b) => b.resolution === 128 && b.height === 32),
  wide.baked.map((b) => `${b.chunk} ${b.resolution}x${b.height}`).join(", "));

// ---- the baked ship comes back out as its own glb --------------------------
// A second .glb and not an edit of the first: ship.glb is what the editor
// writes and the bake reads, so overwriting it would make the input of the next
// bake the output of the last one, and any error would compound.
const baked = real.export;
const bakedGlb = baked && fs.existsSync(baked.glb)
  ? JSON.parse(readGlbJson(baked.glb)) : null;
check("the bake writes ship_baked.glb beside the ship it came from",
  !!bakedGlb && /ship_baked\.glb$/.test(baked.glb),
  baked ? `${baked.glb} ${fs.existsSync(baked.glb)
    ? fs.statSync(baked.glb).size : 0}b` : "not written");
// The whole point of the second UV layer. Without TEXCOORD_1 the runtime has
// nothing to sample the lightmap with. The props the bake leaves out are the
// exception and the proof: they are kept out of the atlas on purpose, so they
// must be the ONLY meshes without it. Before every instance got its own
// geometry a moving crate shared a datablock with a bolted-down one and
// inherited its UV2 by accident, which read as "every mesh carries it" and
// hid the split.
const unbakedMeshes = new Set((bakedGlb?.nodes || [])
  .filter((n) => n.mesh !== undefined
    && real.unbaked.some((d) => (n.name || "").startsWith(d)))
  .map((n) => n.mesh));
const wantUv2 = (bakedGlb?.meshes || [])
  .map((m, i) => i).filter((i) => !unbakedMeshes.has(i));
const hasUv2 = (i) => (bakedGlb.meshes[i].primitives || [])
  .some((pr) => pr.attributes?.TEXCOORD_1 !== undefined);
check("every baked mesh carries the UV2 atlas as TEXCOORD_1",
  !!bakedGlb && wantUv2.length > 0 && wantUv2.every(hasUv2),
  `${wantUv2.filter(hasUv2).length}/${wantUv2.length} static, `
    + `${unbakedMeshes.size} unbaked skipped`);
// And the split has to be real in the other direction too, or "unbaked" is
// just a word: a prop the runtime moves or melts must carry no atlas at all.
check("and nothing the runtime moves or melts carries one",
  unbakedMeshes.size > 0 && ![...unbakedMeshes].some(hasUv2),
  `${unbakedMeshes.size} unbaked mesh(es), `
    + `${[...unbakedMeshes].filter(hasUv2).length} wrongly unwrapped`);
// Not receiving a lightmap is only half of it. Left in the light transport, a
// crate the player pushes still blocks shadow rays and still bounces colour,
// so its shadow is painted onto the floor it happened to be standing on and
// stays there. The bug this guards: the props were dropped as bake *targets*
// and left in the scene as occluders.
const hiddenDyn = new Set(real.unbakedHidden || []);
const unbakedNodeNames = (bakedGlb?.nodes || [])
  .filter((n) => n.mesh !== undefined
    && real.unbaked.some((d) => (n.name || "").startsWith(d)))
  .map((n) => n.name);
check("and it is out of the light transport too, not just off the atlas",
  hiddenDyn.size > 0 && unbakedNodeNames.length > 0
    && unbakedNodeNames.every((n) => hiddenDyn.has(n)),
  `${hiddenDyn.size} hidden, `
    + `${unbakedNodeNames.filter((n) => !hiddenDyn.has(n)).length} still casting`);
// And put back afterwards, so the props export like anything else.
check("and it is back in the scene for the export",
  (real.unbakedRestored || []).length === hiddenDyn.size,
  `${(real.unbakedRestored || []).length} restored of ${hiddenDyn.size}`);
// The light records and the placement ids have to survive the round trip, or
// the baked ship is a ship the runtime cannot wire anything to.
const bakedLights = (bakedGlb?.nodes || []).filter((n) => n.extras?.kind === "light");
check("and the extras come back out, both halves intact",
  bakedLights.length === 3
    && bakedLights.every((n) => n.extras.bake?.shape && n.extras.runtime?.type),
  `${bakedLights.length} light node(s)`);
// Area lights have no glTF equivalent, and the portals were scaffolding.
check("the Cycles lamps and the portals stay behind, being bake-only",
  !(bakedGlb?.extensions?.KHR_lights_punctual)
    && (real.restored || []).length >= 4,
  `${(real.restored || []).length} light object(s) dropped`);
// The lightmap only replaced the *light* the fixture cast, not the look of the
// fixture: the strip still has to glow in the runtime.
const emissive = (bakedGlb?.materials || []).filter((m) =>
  (m.emissiveFactor || []).some((v) => v > 0));
check("and the lamp strip glows again, because only its light was baked",
  emissive.length >= 1, `${emissive.length} emissive material(s)`);

const index = baked && fs.existsSync(baked.index)
  ? JSON.parse(fs.readFileSync(baked.index, "utf8")) : null;
check("lightmaps.json says which map goes on which chunk, and at what level",
  !!index && Object.keys(index.chunks).length === 2 && index.uv === 1
    && Object.values(index.chunks).every((c) => c.png && c.level >= 1),
  index ? JSON.stringify(index.chunks) : "not written");

// ---- only what changed gets re-baked ---------------------------------------
// A full ship is minutes of Cycles, and most edits touch one room. Each chunk
// is hashed over its own contents plus everything one portal away, and a map
// whose hash still matches is kept rather than rendered again.
const t0 = real.baked.map((b) => fs.statSync(b.png).mtimeMs);
const again = bakeReal(["--out", outDir, "--samples", "4", "--resolution", "64",
  "--baked-glb", path.join(scratch, "ship_baked.glb")]);
check("a second bake with nothing changed re-renders nothing",
  (again.reused || []).length === 2 && again.baked.every((b) => b.reused === true)
    && again.baked.every((b, i) => fs.statSync(b.png).mtimeMs === t0[i]),
  `${(again.reused || []).length} reused`);
// The images are still indexed - a bake that reused everything must still leave
// a lightmaps.json the runtime can load, not an empty one.
const idx2 = JSON.parse(fs.readFileSync(again.export.index, "utf8"));
check("and the index still names every map, reused or not",
  Object.keys(idx2.chunks).length === 2
    && Object.values(idx2.chunks).every((c) => c.hash && c.png),
  Object.keys(idx2.chunks).join("+"));

const forced = bakeReal(["--out", outDir, "--samples", "4", "--resolution", "64",
  "--force", "--baked-glb", path.join(scratch, "ship_baked.glb")]);
check("--force renders anyway, for when the images on disk are suspect",
  (forced.reused || []).length === 0
    && forced.baked.every((b) => b.reused === false),
  `${forced.baked.length} re-baked`);

// The cheap half of the same contract: the hashes come out of the manifest, so
// what does and does not invalidate a map can be checked without rendering.
const hashesFor = (mutate, extra = []) => withManifest(mutate, extra).hashes || {};
/** Which chunk a placement id sits in, which is how a light finds its room. */
function chunkOf(m, owner) {
  return (m.instances.find((i) => i.id === owner) || {}).chunk;
}
const base = hashesFor(() => {});
check("every chunk gets a hash of what it was baked from",
  Object.keys(base).length === 2
    && Object.values(base).every((h) => typeof h === "string" && h.length === 32),
  Object.entries(base).map(([k, v]) => `${k}=${v.slice(0, 8)}`).join(" "));

// A lamp is a bake input; the runtime half of the same record is not. Reading
// it would re-render the ship every time somebody nudged a flicker speed.
const lampRoom = chunkOf(man, man.lights[0].owner);
const moved = hashesFor((m) => { m.lights[0].offset[1] += 0.5; });
const runtimeOnly = hashesFor((m) => { m.lights[0].runtime.intensity += 3; });
check("moving a light invalidates its chunk, retuning the runtime one does not",
  !!lampRoom && moved[lampRoom] !== base[lampRoom]
    && JSON.stringify(runtimeOnly) === JSON.stringify(base),
  lampRoom || "no chunk for L0001");

// Light does not stop at a doorway: a lamp moved in the corridor changes the
// storage room's map too, so a chunk's hash covers its neighbours as well.
const next = (man.adjacency[lampRoom] || []).map((e) => e.to)
  .find((c) => c !== lampRoom && base[c]);
check("and it invalidates the rooms on the other side of a portal",
  !!next && moved[next] !== base[next], `${lampRoom} → ${next}`);

// Samples are the noise floor, which is part of the image; the device is not.
check("bake settings are part of the hash, but which machine rendered it is not",
  JSON.stringify(hashesFor(() => {}, ["--samples", "999"])) !== JSON.stringify(base)
    && JSON.stringify(hashesFor(() => {}, ["--device", "GPU"])) === JSON.stringify(base),
  "samples in, device out");

// ---- baking one chunk at a time --------------------------------------------
// A ship is baked per chunk so an edit to one room does not cost a re-render of
// the whole vessel. Naming a chunk that is not there is a typo, and a typo that
// silently bakes nothing is worse than one that stops.
const one = bake(["--chunk", report.chunks[0]]);
check("a single chunk can be asked for by name",
  one.chunks.length === 2 && one.atlases.length === 1
    && one.atlases[0].chunk === report.chunks[0],
  one.atlases.map((a) => a.chunk).join("+"));

let refused = false;
try {
  bake(["--chunk", "CH99_NoSuchRoom"]);
} catch {
  refused = true;
}
check("and a chunk that is not in the ship stops the bake", refused,
  refused ? "refused" : "bakes nothing, quietly");

// ---- a live session, set up but not baked ----------------------------------
// `--interactive` is the same pipeline stopped one step short: everything is
// built, nothing is rendered, and the panel's buttons run the very functions
// the headless path would have run next. Headless it registers and returns,
// which is enough to catch a broken operator or a property that will not
// register - the two ways this can fail without anybody opening Blender.
//
// The unwrap is deliberately NOT done on the way in. `smart_project` over a
// whole ship is minutes, and the reason to open a window is the rendered
// viewport, which needs no UV2 at all; the first bake pays for it instead.
const live = run(["--interactive"]);
check("--interactive sets the ship up and stops before rendering",
  live.interactive === true && live.baked.length === 0 && live.export === null,
  `${live.session?.chunks?.length ?? 0} chunk(s) offered`);

check("and reports whether the Aquanova sidebar tab was focused",
  typeof live.session?.panelTab === "boolean",
  String(live.session?.panelTab));

check("and the session is handed every chunk the headless path would bake",
  JSON.stringify(live.session?.chunks) === JSON.stringify(
    report.atlases.map((a) => a.chunk)),
  (live.session?.chunks || []).join("+"));

check("opening a session does not pay for the unwrap",
  live.atlases.length > 0 && live.atlases.every((a) => a.layer === null
    && a.unwrapped === 0),
  live.atlases.map((a) => `${a.chunk}:${a.unwrapped}`).join(" "));

// The lamps still have to be real: the whole point is judging them by eye.
check("but the lights are built, because that is what there is to look at",
  live.lamps.length === report.lamps.length && live.lamps.every((l) => l.watts > 0),
  `${live.lamps.length} lamp(s)`);

// A ship judged from outside its hull is a ship judged wrong: a ceiling panel
// looks even from above and blinding from under it. So the session opens
// standing in the first room, at the height the player's eyes will be, and the
// chunk dropdown follows so Bake means that room rather than the whole vessel.
const floorUp = live.session && live.session.eye
  ? live.session.eye[2] - live.session.floor : null;
check("and the session opens standing in the first room, at head height",
  live.session?.start === report.atlases[0].chunk
    && Math.abs(floorUp - 1.8) < 1e-3,
  `${live.session?.start} at ${floorUp?.toFixed(2)} m over the floor`);

// Where a room *is* is not the box around everything it owns. A chunk owns
// whatever the editor parented to it, and one barrel left standing out in the
// corridor doubles that box, drags its centre into a wall and leaves the whole
// room behind the camera. So the box is trimmed and the eye goes at the median.
// This fixture is honest, so the two spans should agree and nothing should have
// strayed - if they ever diverge here, the trimming has started eating rooms.
const span = live.session?.span, rawSpan = live.session?.raw_span;
check("and the room it stands in is the room, not the box around its strays",
  live.session?.strays === 0 && !!span
    && span.every((s, i) => s <= rawSpan[i] + 1e-6),
  `${span?.map((s) => s.toFixed(1)).join("x")} of `
    + `${rawSpan?.map((s) => s.toFixed(1)).join("x")}, ${live.session?.strays} stray(s)`);

// ---- the panel's cache check ------------------------------------------------
// The queue decision behind the Bake button cannot be reached from here - the
// operator wants a window, a timer and a modal handler - so it lives in
// `session_queue`, which is pure, and this drives that directly inside
// Blender's own interpreter. `--python-expr` cannot see a `--python` script's
// globals, so the probe is a file that imports the module instead; the
// `__main__` guard is what makes that safe.
const probe = path.join(scratch, "probe_queue.py");
fs.writeFileSync(probe, `
import json, os, sys
sys.path.insert(0, ${JSON.stringify(ROOT.replace(/\\/g, "/"))})
import bake_lightmaps as bl

tmp = ${JSON.stringify(scratch.replace(/\\/g, "/"))}
def touch(name):
    p = os.path.join(tmp, name)
    open(p, "wb").write(b"x")
    return p

hdr, png = touch("a.hdr"), touch("a.png")
gone = os.path.join(tmp, "not-there.png")
bl.SESSION["per_chunk"] = {"fresh": [], "stale": [], "lost": [], "half": []}
bl.SESSION["baked"] = [
    {"chunk": "fresh", "hash": "h1", "image": hdr, "png": png},
    {"chunk": "stale", "hash": "old", "image": hdr, "png": png},
    {"chunk": "half", "hash": "h4", "image": hdr, "png": gone},
]
hashes = {"fresh": "h1", "stale": "new", "lost": "h3", "half": "h4"}

out = {
    "fresh": bl.session_fresh("fresh", "h1"),
    "rehashed": bl.session_fresh("stale", "new"),
    "never": bl.session_fresh("lost", "h3"),
    "deleted": bl.session_fresh("half", "h4"),
    "nohash": bl.session_fresh("fresh", None),
}
names = ["fresh", "stale", "lost", "half", "vanished"]
out["queue"], out["spared"] = bl.session_queue(names, hashes)
out["forced"], out["forced_spared"] = bl.session_queue(names, hashes, True)
bl.SESSION["uv2_dirty"] = {"lost"};
out["uv2_selected_queue"], out["uv2_selected_spared"] = bl.session_queue(
    ["fresh"], hashes)
out["uv2_all_queue"], out["uv2_all_spared"] = bl.session_queue(
    names, hashes)
out["progress"] = bl.bake_progress(1, 3, "CH01_StorageCorridor", 2);
base_manifest = {
    "savedAt": "old",
    "chunks": [{"id": "A", "aabb": {"min": [0], "max": [1]},
                "bake": {"samples": 128}}],
    "bakeDefaults": {"samples": 128},
    "bakeLighting": {"power": 1, "sky": 1},
}
timestamp_manifest = dict(base_manifest, savedAt="new")
bake_manifest = dict(timestamp_manifest, bakeDefaults={"samples": 256},
                      chunks=[{"id": "A", "aabb": {"min": [0], "max": [1]},
                               "bake": {"samples": 256}}])
scene_manifest = dict(bake_manifest,
                      chunks=[{"id": "A", "aabb": {"min": [2], "max": [3]},
                               "bake": {"samples": 256}}])
merged = bl.merge_bake_manifest(base_manifest, scene_manifest)
out["manifest_reload"] = {
    "timestamp_only": bl.manifest_session_signature(base_manifest)
        == bl.manifest_session_signature(timestamp_manifest),
    "bake_only": bl.manifest_session_signature(base_manifest)
        == bl.manifest_session_signature(bake_manifest),
    "scene_changed": bl.manifest_session_signature(base_manifest)
        != bl.manifest_session_signature(scene_manifest),
    "kept_scene": merged["chunks"][0]["aabb"] == base_manifest["chunks"][0]["aabb"],
    "updated_bake": merged["chunks"][0]["bake"]["samples"] == 256,
}
print("PROBE " + json.dumps(out))
`);
const probeOut = execFileSync(blender,
  ["-b", "--factory-startup", "--python", probe],
  { encoding: "utf8", maxBuffer: 16 * 1024 * 1024, timeout: 300000 });
const q = JSON.parse((probeOut.split(/\r?\n/)
  .find((l) => l.startsWith("PROBE ")) || "PROBE null").slice(6));

// "Up to date" has to mean the map on disk, not the bookkeeping. A hash that
// still matches proves nothing if somebody emptied the output folder between
// two clicks, and that is exactly the case where being told "up to date" and
// shown last week's lighting would be maddening.
check("a room counts as fresh only if its hash matches AND its files are there",
  q?.fresh === true && q.rehashed === false && q.never === false
    && q.deleted === false && q.nohash === false,
  q ? `fresh=${q.fresh} rehashed=${q.rehashed} deleted=${q.deleted}` : "no probe");

// The point of the whole thing: All chunks costs the rooms that moved.
check("so Bake renders what changed and spares what did not",
  JSON.stringify(q?.queue) === JSON.stringify(["stale", "lost", "half"])
    && JSON.stringify(q?.spared) === JSON.stringify(["fresh"]),
  `${(q?.queue || []).join("+")} over ${(q?.spared || []).join("+")}`);

// And the escape hatch, for when the hashes are right but the images are not.
check("and Re-bake up-to-date rooms renders them anyway",
  JSON.stringify(q?.forced) === JSON.stringify(
    ["fresh", "stale", "lost", "half"]) && q?.forced_spared.length === 0,
  (q?.forced || []).join("+"));

check("the live bake progress names the current chunk and remaining textures",
  q?.progress ===
    "Baking CH01_StorageCorridor (2/3) - 2 texture(s) remaining",
  q?.progress || "no progress");

check("a selected bake stays independent from another dirty chunk",
  JSON.stringify(q?.uv2_selected_queue) === JSON.stringify([])
    && JSON.stringify(q?.uv2_selected_spared) === JSON.stringify(["fresh"]),
  `${(q?.uv2_selected_queue || []).join("+")} over `
    + `${(q?.uv2_selected_spared || []).join("+")}`);
check("baking all chunks still includes the dirty chunk",
  JSON.stringify(q?.uv2_all_queue) === JSON.stringify(["half", "lost", "stale"])
    && JSON.stringify(q?.uv2_all_spared) === JSON.stringify(["fresh"]),
  `${(q?.uv2_all_queue || []).join("+")} over ${(q?.uv2_all_spared || []).join("+")}`);

check("manifest timestamps and bake settings do not look like scene changes",
  q?.manifest_reload?.timestamp_only && q.manifest_reload.bake_only
    && q.manifest_reload.scene_changed,
  JSON.stringify(q?.manifest_reload));
check("scene edits stay with the imported scene while bake settings refresh",
  q?.manifest_reload?.kept_scene && q.manifest_reload.updated_bake,
  JSON.stringify(q?.manifest_reload));

// ---- the report is the contract --------------------------------------------
const reportFile = path.join(scratch, "report.json");
bake(["--report", reportFile]);
const written = fs.existsSync(reportFile)
  ? JSON.parse(fs.readFileSync(reportFile, "utf8")) : null;
check("--report writes the same summary to disk, for the editor to read back",
  !!written && written.lights.found === 3 && written.blender.startsWith("5."),
  written ? `blender ${written.blender}` : "not written");

fs.rmSync(scratch, { recursive: true, force: true });

console.log(results.join("\n"));
const failures = results.filter((r) => r.startsWith("FAIL")).length;
console.log("\nfailures:", failures);
process.exit(failures ? 1 : 0);
