// Interaction test for the ghost-driven workflow: ghost follow, click-to-place,
// grab/drop, hover outline, wheel rotate/scale, grid elevation and axis modes.

import { createRequire } from "node:module";
const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const URL = process.env.TOOL_URL || "http://localhost:5180/";

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1700, height: 950 } });
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

await page.goto(URL, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => document.querySelectorAll("#palette-list .item").length > 0,
  null, { timeout: 60000 });
await page.waitForTimeout(2500);
await page.evaluate(async () => (await import("/js/editor.js")).clearAll());

const results = [];
const check = (name, pass, detail = "") => {
  const line = `${pass ? "PASS" : "FAIL"}  ${name.padEnd(38)} ${detail}`;
  results.push(line);
  console.log(line);          // stream, so a mid-run crash still shows progress
};

// the kit tile is 4 m; the default move snap is 1 m, so ask for 4 explicitly
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.snap.pos = 4;
  document.getElementById("snap-pos").value = "4";
});

// project a world point to canvas coordinates so we can aim the real mouse
async function screenOf(world) {
  const r = await page.evaluate((w) => {
    const s = window.__scene;
    const e = s.getEngine();
    s.render();                       // make sure the view matrix is current
    const p = BABYLON.Vector3.Project(
      BABYLON.Vector3.FromArray(w),
      BABYLON.Matrix.Identity(),
      s.getTransformMatrix(),
      s.activeCamera.viewport.toGlobal(e.getRenderWidth(), e.getRenderHeight()));
    const rect = e.getRenderingCanvas().getBoundingClientRect();
    return {
      x: rect.x + p.x, y: rect.y + p.y,
      inside: p.x >= 0 && p.y >= 0 && p.x <= rect.width && p.y <= rect.height,
      camPos: s.activeCamera.position.asArray().map((v) => +v.toFixed(1)),
    };
  }, world);
  if (!r.inside) {
    throw new Error(`world ${JSON.stringify(world)} projects off-canvas ` +
      `(${r.x.toFixed(0)},${r.y.toFixed(0)}) from camera ${JSON.stringify(r.camPos)}`);
  }
  return r;
}

const MODULE = "Walls/ShortWall_Band2_Straight";

// ---- 1. arming the palette creates a ghost that follows the cursor ----------
await page.evaluate((m) => import("/js/palette.js").then((p) => p.setBrush(m)), MODULE);
await page.waitForTimeout(1200);

let pt = await screenOf([0, 0, 0]);
await page.mouse.move(pt.x, pt.y, { steps: 4 });
await page.waitForTimeout(250);

let g = await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  const n = window.__scene.getTransformNodeByName("GHOST");
  return { active: i.ghostActive(), module: i.ghostModule(), pos: n?.position.asArray() };
});
check("ghost armed from palette", g.active && g.module === MODULE, g.module || "");
check("ghost snapped to 4 m grid",
  g.pos && g.pos.every((v) => Math.abs(v / 4 - Math.round(v / 4)) < 1e-6), JSON.stringify(g.pos));

// ---- 1b. the ghost body must sit under the cursor, not at its own origin ----
// Measured in screen space with snapping off, since "under the cursor" is a
// screen property and grid quantisation would otherwise dominate.
const offGridAim = await screenOf([2.4, 0, 2.4]);
await page.mouse.move(offGridAim.x, offGridAim.y, { steps: 4 });
await page.waitForTimeout(250);

const centred = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const snapWas = ed.state.snap.pos;
  ed.state.snap.pos = 0;
  const s = window.__scene;
  // nudge so the ghost re-solves with snapping off
  s.onPointerObservable.notifyObservers({
    type: BABYLON.PointerEventTypes.POINTERMOVE,
    event: { clientX: 0, clientY: 0 },
  });
  await new Promise((r) => requestAnimationFrame(r));

  const root = s.getTransformNodeByName("GHOST");
  let min = null, max = null;
  for (const m of root.getChildMeshes()) {
    m.computeWorldMatrix(true); m.refreshBoundingInfo();
    const bb = m.getBoundingInfo().boundingBox;
    min = min ? BABYLON.Vector3.Minimize(min, bb.minimumWorld) : bb.minimumWorld.clone();
    max = max ? BABYLON.Vector3.Maximize(max, bb.maximumWorld) : bb.maximumWorld.clone();
  }
  const body = min.add(max).scale(0.5);

  const e = s.getEngine();
  const vp = s.activeCamera.viewport.toGlobal(e.getRenderWidth(), e.getRenderHeight());
  const px = (v) => BABYLON.Vector3.Project(v, BABYLON.Matrix.Identity(), s.getTransformMatrix(), vp);
  const pb = px(body), po = px(root.position);
  const proto = await (await import("/js/kit.js")).getProto("Walls/ShortWall_Band2_Straight");
  ed.state.snap.pos = snapWas;
  return {
    bodyGap: Math.hypot(pb.x - s.pointerX, pb.y - s.pointerY),
    originGap: Math.hypot(po.x - s.pointerX, po.y - s.pointerY),
    localCentre: proto._centre
      ? [+proto._centre.x.toFixed(2), +proto._centre.y.toFixed(2), +proto._centre.z.toFixed(2)]
      : null,
  };
});
check("ghost body lands under the cursor", centred.bodyGap < 12,
  `body ${centred.bodyGap.toFixed(0)} px from the pointer`);
check("compensation actually changes the result", centred.originGap > centred.bodyGap + 20,
  `origin is ${centred.originGap.toFixed(0)} px away; module body sits ` +
  `${JSON.stringify(centred.localCentre)} from its own origin`);

const cursorHidden = await page.evaluate(() =>
  window.__scene.getEngine().getRenderingCanvas().style.cursor);
check("cursor hidden while placing", cursorHidden === "none", `cursor="${cursorHidden}"`);

// ---- 1b2. a module whose body sits well above its origin ------------------
// TopCables_Corner_* are ceiling pieces modelled ~4 m above their own origin.
// Following the build plane would leave the ghost floating far up-screen.
const highModule = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  const id = "Walls/TopCables_Corner_Square_Inner";
  if (!kit.getModule(id)) return null;
  const b = await kit.moduleBounds(id);
  return { id, centreY: (b.min.y + b.max.y) / 2 };
});
if (highModule) {
  await page.evaluate((m) => import("/js/palette.js").then((p) => p.setBrush(m)), highModule.id);
  await page.waitForTimeout(1200);
  const aim = await screenOf([0, 0, 0]);
  await page.mouse.move(aim.x, aim.y, { steps: 5 });
  await page.waitForTimeout(300);
  const highFit = await page.evaluate(async () => {
    const s = window.__scene;
    const root = s.getTransformNodeByName("GHOST");
    let min = null, max = null;
    for (const m of root.getChildMeshes()) {
      m.computeWorldMatrix(true); m.refreshBoundingInfo();
      const bb = m.getBoundingInfo().boundingBox;
      min = min ? BABYLON.Vector3.Minimize(min, bb.minimumWorld) : bb.minimumWorld.clone();
      max = max ? BABYLON.Vector3.Maximize(max, bb.maximumWorld) : bb.maximumWorld.clone();
    }
    const body = min.add(max).scale(0.5);
    const e = s.getEngine();
    const vp = s.activeCamera.viewport.toGlobal(e.getRenderWidth(), e.getRenderHeight());
    const pr = BABYLON.Vector3.Project(body, BABYLON.Matrix.Identity(), s.getTransformMatrix(), vp);
    return {
      screenGap: Math.hypot(pr.x - s.pointerX, pr.y - s.pointerY),
      bodyY: +body.y.toFixed(2),
    };
  });
  // the body has to land near the pointer *on screen*, which is the thing that
  // breaks when a tall module is tracked on the build plane instead
  check("a high module still lands under the cursor", highFit.screenGap < 90,
    `${highModule.id} body at y=${highFit.bodyY}, ${highFit.screenGap.toFixed(0)} px from the pointer`);

  // its thumbnail must be framed from above too: setTarget re-derives alpha and
  // beta, which used to leave tall modules viewed from underneath
  const thumbView = await page.evaluate(async (id) => {
    const t = await import("/js/thumbs.js");
    const kit = await import("/js/kit.js");
    await new Promise((r) => {
      const img = new Image();
      t.request(kit.getModule(id), img);
      const wait = () => (img.naturalWidth > 0 || img.src ? r() : setTimeout(wait, 200));
      setTimeout(wait, 300);
    });
    return { used: t.lastThumbView(), want: t.THUMB_VIEW };
  }, highModule.id);
  check("thumbnails keep a canonical framing",
    Math.abs(thumbView.used.beta - thumbView.want.beta) < 1e-6
    && Math.abs(thumbView.used.alpha - thumbView.want.alpha) < 1e-6
    && thumbView.used.beta < Math.PI / 2,
    `alpha ${thumbView.used.alpha.toFixed(2)}, beta ${thumbView.used.beta.toFixed(2)} ` +
    `(want ${thumbView.want.beta.toFixed(2)}, must stay under ${(Math.PI / 2).toFixed(2)} to look down)`);
  await page.evaluate((m) => import("/js/palette.js").then((p) => p.setBrush(m)), MODULE);
  await page.waitForTimeout(900);
  await page.mouse.move(aim.x, aim.y, { steps: 4 });
  await page.waitForTimeout(200);
}

// ---- 1c. the ghost is textured and translucent, and shares its textures -----
const ghostMat = await page.evaluate(async () => {
  const s = window.__scene;
  const root = s.getTransformNodeByName("GHOST");
  const m = root.getChildMeshes()[0];
  const mat = m.material;
  const proto = (await import("/js/kit.js")).getProto
    ? await (await import("/js/kit.js")).getProto("Walls/ShortWall_Band2_Straight") : null;
  const srcMat = proto?.parts[0].mesh.material;
  const tex = mat.getActiveTextures().map((t) => t.name);
  const srcTex = srcMat ? srcMat.getActiveTextures().map((t) => t.name) : [];
  return {
    cls: mat.getClassName(),
    alpha: mat.alpha,
    textures: tex.length,
    shared: srcMat ? mat.getActiveTextures().every((t) => srcMat.getActiveTextures().includes(t)) : false,
    sameAsSource: mat === srcMat,
    sceneTextures: s.textures.length,
    srcTex: srcTex.length,
  };
});
check("ghost uses the real textured material",
  ghostMat.cls === "PBRMaterial" && ghostMat.textures > 0 && !ghostMat.sameAsSource,
  `${ghostMat.cls}, ${ghostMat.textures} texture(s)`);
check("ghost is translucent", ghostMat.alpha > 0 && ghostMat.alpha < 1, `alpha=${ghostMat.alpha}`);
check("ghost material shares source textures", ghostMat.shared,
  `${ghostMat.textures} vs source ${ghostMat.srcTex}, scene has ${ghostMat.sceneTextures}`);

// ---- 1d. nothing may be backface-culled in the editor -----------------------
const culling = await page.evaluate(async () => {
  const s = window.__scene;
  const kit = await import("/js/kit.js");
  const culled = s.materials.filter((m) => m.backFaceCulling);
  return {
    total: s.materials.length,
    culled: culled.map((m) => m.name),
    registry: [...kit.materialRegistry.values()].filter((m) => m.backFaceCulling).length,
  };
});
check("no material is backface-culled", culling.culled.length === 0,  `${culling.total} materials, culled: ${culling.culled.join(", ") || "none"}`);

// ---- 1d-bis. the ceiling gets lit ------------------------------------------
// Babylon's hemi direction points at its "sky", so the main hemisphere only
// ever gives a downward-facing surface its (dark) ground colour - which is
// every ceiling panel and platform underside in the kit.
const rig = await page.evaluate(() => {
  const s = window.__scene;
  const up = s.getLightByName("hemi");
  const down = s.getLightByName("hemiUp");
  return {
    hasBoth: !!up && !!down,
    upDir: up?.direction.asArray(),
    downDir: down?.direction.asArray(),
    downIntensity: down?.intensity,
    downGround: down?.groundColor.asArray(),
    downEnabled: down?.isEnabled(),
  };
});
check("a second hemisphere aims down, to light the undersides",
  rig.hasBoth && rig.upDir[1] > 0 && rig.downDir[1] < 0
    && rig.downIntensity > 0 && rig.downEnabled,
  `up ${rig.upDir?.map((v) => v.toFixed(1))}, down ${rig.downDir?.map((v) => v.toFixed(1))} ` +
  `at ${rig.downIntensity}`);
check("the up-light's ground colour is black, so floors are not lit twice",
  rig.downGround.every((v) => v === 0), `groundColor [${rig.downGround}]`);

// ---- 1d-ter. rotation composes, and the viewport knobs ---------------------
// Rotation used to be accumulated in Euler space: read the triple, add a step,
// write it back. That works for Y and Z and quietly fails for X, because
// toEulerAngles() decomposes YXZ with X clamped to +/-90 deg. From a yawed
// element, four 90 deg X steps used to give (90,90,0) -> (0,-90,180) ->
// (90,-270,0) -> (0,-90,180): it oscillates instead of coming back round.
const axisSpin = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.state.snap.rot = 90;
  const e = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.select([e.id]);
  const up = () => {
    const m = new BABYLON.Matrix();
    e.node.rotationQuaternion.toRotationMatrix(m);
    return V.TransformCoordinates(V.Up(), m).asArray().map((v) => +v.toFixed(2));
  };
  const box = () => {
    const b = ed.worldBounds(e.node);
    return [b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z].map((v) => +v.toFixed(2));
  };
  const out = {};
  ed.state.rotAxis = "y"; i.rotateCurrent(1);       // yaw first: this is what broke X
  const start = up();
  ed.state.rotAxis = "x";
  const boxBefore = box();
  const cyc = [];
  for (let k = 0; k < 4; k++) { i.rotateCurrent(1); cyc.push(up()); }
  out.x = { start, cyc, distinct: new Set(cyc.map(JSON.stringify)).size,
            back: JSON.stringify(cyc[3]) === JSON.stringify(start) };
  i.rotateCurrent(1);
  out.tipped = { before: boxBefore, after: box() };
  for (let k = 0; k < 3; k++) i.rotateCurrent(1);

  for (const axis of ["y", "z"]) {
    ed.state.rotAxis = axis;
    const s = up();
    const c = [];
    for (let k = 0; k < 4; k++) { i.rotateCurrent(1); c.push(up()); }
    out[axis] = { back: JSON.stringify(c[3]) === JSON.stringify(s) };
  }
  ed.state.rotAxis = "y";
  ed.clearAll(); ed.select([]);
  return out;
});
check("four 90° turns about X come back to the start",
  axisSpin.x.back && axisSpin.x.distinct === 4,
  `${axisSpin.x.distinct} distinct orientations, ends at [${axisSpin.x.cyc[3]}]`);
check("an X turn actually tips the element over",
  axisSpin.tipped.before[1] > 1 && axisSpin.tipped.after[1] < 0.5 && axisSpin.tipped.after[2] > 1,
  `${axisSpin.tipped.before} -> ${axisSpin.tipped.after}`);
check("Y and Z still come back round too", axisSpin.y.back && axisSpin.z.back);

const knobs = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const before = { state: ed.state.envIntensity, scene: window.__scene.environmentIntensity };
  ed.setEnvIntensity(3.2);
  const raised = { state: ed.state.envIntensity, scene: window.__scene.environmentIntensity };
  ed.setEnvIntensity(9);   const high = ed.state.envIntensity;
  ed.setEnvIntensity(-1);  const low = ed.state.envIntensity;
  ed.setEnvIntensity(before.state);
  return { before, raised, high, low,
           inertiaSlider: !!document.getElementById("inertia"),
           camInertia: ed.state.camera.inertia,
           rotOptions: [...document.getElementById("snap-rot").options].map((o) => o.value),
           scaleOptions: [...document.getElementById("snap-scale").options].map((o) => o.value) };
});
check("the Env slider drives the scene's IBL strength",
  knobs.raised.state === 3.2 && knobs.raised.scene === 3.2,
  `${knobs.before.scene} -> ${knobs.raised.scene}`);
check("Env intensity is clamped", knobs.high === 6 && knobs.low === 0,
  `high=${knobs.high}, low=${knobs.low}`);
check("the inertia slider is gone and the camera is fixed at 0.75",
  !knobs.inertiaSlider && knobs.camInertia === 0.75,
  `slider=${knobs.inertiaSlider}, inertia=${knobs.camInertia}`);
// "off" is a real option for Move (free positioning) but meaningless for a
// keyboard step: it fell back to a hidden default instead of doing nothing.
check("Rot and Scale no longer offer a meaningless 'off'",
  !knobs.rotOptions.includes("0") && !knobs.scaleOptions.includes("0"),
  `rot [${knobs.rotOptions}], scale [${knobs.scaleOptions}]`);

// ---- 1d-quater. undo / redo ------------------------------------------------
const history = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const base = ed.state.placements.size;

  ed.pushUndo();
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(4, 0, 0), { silent: true });
  const added = ed.state.placements.size;
  await ed.undo();
  const undone = ed.state.placements.size;
  await ed.redo();
  const redone = ed.state.placements.size;

  // a move is undoable too, and restores the exact position
  await ed.undo();
  ed.select([a.id]);
  const home = a.node.position.clone();
  ed.pushUndo();
  ed.nudgeSelection(new V(8, 0, 0));
  const moved = ed.state.placements.get(a.id).node.position.x;
  await ed.undo();
  const backHome = ed.state.placements.get(a.id).node.position.x;
  ed.clearAll(); ed.select([]);
  return { base, added, undone, redone, homeX: home.x, moved, backHome };
});
check("undo removes what was just added, redo puts it back",
  history.added === history.base + 1 && history.undone === history.base
    && history.redone === history.base + 1,
  `${history.base} -> ${history.added} -> ${history.undone} -> ${history.redone}`);
check("undo restores an exact position, not an approximation",
  Math.abs(history.moved - (history.homeX + 8)) < 1e-6
    && Math.abs(history.backHome - history.homeX) < 1e-6,
  `${history.homeX} -> ${history.moved} -> ${history.backHome}`);

// A restore is not itself an edit. Restoring a spawn marker used to call
// pushUndo(), which wiped the redo stack (so Ctrl+Y did nothing) and pushed a
// half-restored snapshot onto the undo stack (so a second Ctrl+Z lost the
// player start). It only showed up on layouts that actually had a spawn.
const withSpawn = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  mk.addDoor(new V(2, 0, 2), { silent: true });
  const base = { placements: ed.state.placements.size, markers: ed.state.markers.size };

  // two placements, exactly as a run of palette clicks would
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(4, 0, 0));
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(8, 0, 0));
  const twice = ed.state.placements.size;

  await ed.undo();
  const afterUndo = { n: ed.state.placements.size, ...ed.historyDepth() };
  await ed.redo();
  const afterRedo = { n: ed.state.placements.size, markers: ed.state.markers.size };
  await ed.undo();
  await ed.undo();
  const afterTwo = { n: ed.state.placements.size, markers: ed.state.markers.size,
                     ids: [...ed.state.markers.keys()] };

  const guarded = ed.isRestoring() === false;
  ed.clearAll(); ed.select([]);
  return { base, twice, afterUndo, afterRedo, afterTwo, guarded };
});
check("redo works on a layout that has a marker",
  withSpawn.afterUndo.n === withSpawn.twice - 1 && withSpawn.afterUndo.redo === 1
    && withSpawn.afterRedo.n === withSpawn.twice,
  `${withSpawn.twice} -> undo ${withSpawn.afterUndo.n} (redo depth ${withSpawn.afterUndo.redo})` +
  ` -> redo ${withSpawn.afterRedo.n}`);
check("undoing twice does not lose the marker",
  withSpawn.afterTwo.n === withSpawn.base.placements
    && withSpawn.afterTwo.markers === withSpawn.base.markers,
  `${withSpawn.afterTwo.n} placements, markers [${withSpawn.afterTwo.ids}]`);
check("the restore guard is released again", withSpawn.guarded);

// ---- 1d-quinquies. renaming chunks and elements ----------------------------
// A chunk id is not a label: placements carry it, doors name the two chunks
// they join, and the active chunk is one. Renaming only the list entry would
// orphan every placement and silently break the portals.
const naming = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mk = await import("/js/markers.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.addChunk("CH_A"); ed.addChunk("CH_B");
  ed.state.activeChunk = "CH_A";
  const a = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.state.activeChunk = "CH_B";
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(8, 0, 0), { silent: true });
  mk.addDoor(new V(4, 0, 0), { chunkA: "CH_A", chunkB: "CH_B", silent: true });
  ed.state.activeChunk = "CH_A";

  const ok = ed.renameChunk("CH_A", "CH_Storage");
  const after = {
    listed: ed.state.chunks.includes("CH_Storage") && !ed.state.chunks.includes("CH_A"),
    placement: ed.state.placements.get(a.id).chunk,
    door: [...ed.state.markers.values()].find((m) => m.type === "door").chunkA,
    active: ed.state.activeChunk,
  };
  const dup = ed.renameChunk("CH_B", "CH_Storage");     // already taken
  const blank = ed.renameChunk("CH_B", "   ");          // nothing to rename to

  ed.renamePlacement(a.id, "  weapon locker  ");
  const named = ed.state.placements.get(a.id).name;
  const inManifest = mf.buildManifest().instances.find((x) => x.id === a.id)?.name;
  await ed.deserialize(JSON.parse(JSON.stringify(ed.serialize())));
  const survived = ed.state.placements.get(a.id)?.name;
  ed.select([a.id]);
  const label = i.currentElement()?.module;
  ed.clearAll(); ed.select([]);
  return { ok, after, dup, blank, named, inManifest, survived, label };
});
check("renaming a chunk follows every reference to it",
  naming.ok && naming.after.listed && naming.after.placement === "CH_Storage"
    && naming.after.door === "CH_Storage" && naming.after.active === "CH_Storage",
  `placement=${naming.after.placement}, door=${naming.after.door}, active=${naming.after.active}`);
check("a chunk cannot be renamed onto an existing one, or to nothing",
  naming.dup === false && naming.blank === false,
  `duplicate=${naming.dup}, blank=${naming.blank}`);
check("an element name is trimmed and kept in the manifest",
  naming.named === "weapon locker" && naming.inManifest === "weapon locker",
  `"${naming.named}" / manifest "${naming.inManifest}"`);
check("an element name survives a save/load round-trip",
  naming.survived === "weapon locker", `"${naming.survived}"`);
check("a named element shows its name rather than its module",
  naming.label === "weapon locker", `"${naming.label}"`);

// ---- 1d-sexies. names carry into the .glb ----------------------------------
// glTF node names come straight off the Babylon nodes, so the export renames
// them and puts them back. Two things have to hold: the file has to be
// readable (names, not P0007s) and every name has to stay unique, because the
// runtime resolves door leaves by node name.
const glbNames = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mk = await import("/js/markers.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  const M = "Walls/ShortWall_Band2_Straight";
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  const a = await ed.placeAt(M, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(M, new V(4, 0, 0), { silent: true });
  const c = await ed.placeAt(M, new V(8, 0, 0), { silent: true });
  const d = await ed.placeAt(M, new V(12, 0, 0), { silent: true });
  ed.renamePlacement(a.id, "weapon locker");
  ed.renamePlacement(b.id, "door");            // deliberately the same name
  ed.renamePlacement(c.id, "door");            // d is left unnamed
  mk.addDoor(new V(2, 0, 0), { silent: true, leaves: [a.id] });

  // capture the bytes instead of writing them: the test must not depend on,
  // or touch, whatever export directory the server happens to be pointed at
  const realFetch = window.fetch;
  let body = null;
  window.fetch = (url, opts) => {
    if (String(url).includes("/api/export")) {
      body = opts.body;
      return Promise.resolve(new Response('{"ok":true,"bytes":0}',
        { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
  try { await mf.exportGlb(); } finally { window.fetch = realFetch; }

  const buf = await body.arrayBuffer();
  const dv = new DataView(buf);
  const json = JSON.parse(new TextDecoder()
    .decode(new Uint8Array(buf, 20, dv.getUint32(12, true))));

  const man = mf.buildManifest();
  const out = {
    nodes: (json.nodes || []).map((n) => n.name),
    // how many glTF nodes carry each name, so shared names can be proved shared
    tally: (json.nodes || []).reduce((acc, n) => {
      acc[n.name] = (acc[n.name] || 0) + 1;
      return acc;
    }, {}),
    leaf: man.doors[0]?.leaves[0],
    instanceNodes: Object.fromEntries(man.instances.map((x) => [x.id, x.node])),
    // the editor must look exactly as it did before the export
    sceneNames: [...ed.state.placements.values()]
      .flatMap((p) => [p.node.name, ...p.node.getChildMeshes().map((m) => m.name)]),
    ids: { a: a.id, b: b.id, c: c.id, d: d.id },
  };
  ed.clearAll(); ed.select([]);
  return out;
});

const gNodes = glbNames.nodes;
check("a named element's parent node takes the name",
  glbNames.tally["weapon locker"] === 1
    && gNodes.includes("weapon locker_primitive0")
    && gNodes.includes("weapon locker_primitive1"),
  JSON.stringify(gNodes.filter((n) => n.startsWith("weapon"))));
check("elements sharing a name share the node name, so one behaviour governs both",
  glbNames.tally.door === 2 && !gNodes.some((n) => n.startsWith("door (")),
  `${glbNames.tally.door} nodes called "door"`);
check("an unnamed element still exports under its id",
  glbNames.tally[glbNames.ids.d] === 1
    && gNodes.includes(`${glbNames.ids.d}_primitive0`),
  glbNames.ids.d);
check("the manifest names each element as the .glb does",
  glbNames.instanceNodes[glbNames.ids.a] === "weapon locker"
    && glbNames.instanceNodes[glbNames.ids.b] === "door"
    && glbNames.instanceNodes[glbNames.ids.d] === glbNames.ids.d,
  JSON.stringify(glbNames.instanceNodes));
check("a door leaf points at the leaf's node name",
  glbNames.leaf?.node === "weapon locker", JSON.stringify(glbNames.leaf));

// A window onto space: the portal still renders, but the far side cannot be
// walked to. Written on the door for the collision work to read later.
const sealedDoor = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const mf = await import("/js/manifest.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const d = mk.addDoor(new V(4, 0, 0), { silent: true });
  const fresh = mf.buildManifest().doors[0];

  ed.select([d.id]);
  document.getElementById("door-sealed").checked = true;
  document.getElementById("door-sealed").dispatchEvent(new Event("change", { bubbles: true }));
  const built = mf.buildManifest();

  // it must survive a save/load round trip, and undo
  const layout = ed.serialize();
  await ed.deserialize(JSON.parse(JSON.stringify(layout)));
  const reloaded = [...ed.state.markers.values()][0]?.sealed;
  ed.select([d.id]);
  await ed.undo();
  const afterUndo = [...ed.state.markers.values()][0]?.sealed;

  const out = {
    byDefault: fresh.sealed,
    onDoor: built.doors[0].sealed,
    // not on the portal: only the door carries it for now
    onPortal: "sealed" in built.portals[0],
    reloaded, afterUndo,
    checkbox: document.getElementById("door-sealed").checked,
  };
  ed.clearAll(); ed.select([]);
  return out;
});
check("a door is an ordinary doorway unless sealed",
  sealedDoor.byDefault === false, `${sealedDoor.byDefault}`);
check("the Sealed box reaches the manifest's door record",
  sealedDoor.onDoor === true && sealedDoor.onPortal === false,
  `door=${sealedDoor.onDoor}, on the portal too=${sealedDoor.onPortal}`);
check("sealed survives a reload, and undo takes it back",
  sealedDoor.reloaded === true && sealedDoor.afterUndo === false,
  `reloaded=${sealedDoor.reloaded}, after undo=${sealedDoor.afterUndo}`);

// ---- 1d-quatervicies. collision primitives ---------------------------------
// Unit shapes sized by scaling - but only the box takes an arbitrary scale.
// Havok's sphere is one radius and its capsule a radius plus two endpoints, so
// an ellipsoid has no representation and the runtime would have to silently
// resize it. The editor pulls the scale back onto something buildable instead.
const coll = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const mf = await import("/js/manifest.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  const made = {};
  for (const k of co.COLLIDER_KINDS) made[k] = co.addCollider(k, new V(0, 0, 0), { silent: true });

  made.sphere.node.scaling.set(2, 1, 1);  co.reconcileCollider(made.sphere);
  made.capsule.node.scaling.set(3, 5, 1); co.reconcileCollider(made.capsule);
  made.box.node.scaling.set(4, 2, 0.2);   co.reconcileCollider(made.box);
  made.cylinder.node.scaling.set(2, 6, 2);
  made.cylinder.node.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(Math.PI / 2, 0, 0);

  const shapes = mf.buildManifest().collision[ed.state.activeChunk] || [];
  const by = Object.fromEntries(shapes.map((s) => [s.kind, s]));

  // a collider is an element like any other: entryOf finds it, so selection,
  // dragging, hiding and the gizmo all work without knowing about it
  const viaEntryOf = !!ed.entryOf(made.box.id);
  const layout = ed.serialize();
  await ed.deserialize(JSON.parse(JSON.stringify(layout)));
  const out = {
    kinds: [...ed.state.colliders.values()].map((c) => c.kind).sort().join(),
    viaEntryOf,
    sphere: made.sphere.node.scaling.asArray(),
    capsule: made.capsule.node.scaling.asArray(),
    box: made.box.node.scaling.asArray(),
    mBox: by.box, mSphere: by.sphere, mCyl: by.cylinder,
    inGlb: null,
  };
  ed.clearAll(); ed.select([]);
  return out;
});
check("all four primitives exist and survive a reload",
  coll.kinds === "box,capsule,cylinder,sphere" && coll.viaEntryOf, coll.kinds);
check("a sphere is forced round, and a capsule to one radius",
  coll.sphere[0] === coll.sphere[1] && coll.sphere[1] === coll.sphere[2]
    && coll.capsule[0] === coll.capsule[2] && coll.capsule[1] === 5,
  `sphere [${coll.sphere}], capsule [${coll.capsule}]`);
check("a box keeps whatever scale it is given",
  coll.box.join() === "4,2,0.2", `[${coll.box}]`);
check("the manifest speaks Havok's own parameters",
  coll.mBox.halfExtents.join() === "2,1,0.1" && coll.mBox.rotation.length === 4
    && Math.abs(coll.mSphere.radius - 0.6667) < 1e-3
    && !("rotation" in coll.mSphere),
  `box ${JSON.stringify(coll.mBox.halfExtents)}, sphere r=${coll.mSphere.radius}`);
check("a turned capsule or cylinder carries its axis as two points",
  // rotated 90 degrees about X, so the segment runs along Z, not Y - which is
  // how Havok expresses an arbitrarily oriented capsule
  Math.abs(coll.mCyl.pointA[2] + 3) < 1e-3 && Math.abs(coll.mCyl.pointB[2] - 3) < 1e-3
    && Math.abs(coll.mCyl.pointA[1]) < 1e-3,
  `A=${coll.mCyl.pointA} B=${coll.mCyl.pointB}`);

// ---- 1d-sexvicies. a collider is placed and moved like anything else -------
// The Collision pane arms the ghost the same way the palette does: nothing
// exists until you click. And a collider is pickable - it used to be created
// and then be unselectable for good, because the pick resolved a mesh through
// `placementRoot`/`markerRoot` only and never looked at `colliderRoot`.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
});
await page.waitForTimeout(200);

const collCanvas = await page.evaluate(() =>
  window.__scene.getEngine().getRenderingCanvas().getBoundingClientRect().toJSON());
const collMid = {
  x: Math.round(collCanvas.x + collCanvas.width / 2),
  y: Math.round(collCanvas.y + collCanvas.height / 2),
};

await page.click('#collider-buttons button[data-kind="box"]');
await page.mouse.move(collMid.x, collMid.y);
await page.waitForTimeout(300);
const armedColl = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return {
    ghost: i.ghostActive(), kind: i.ghostCollider(), made: ed.state.colliders.size,
    lit: document.querySelector('#collider-buttons button[data-kind="box"]')
      .classList.contains("active"),
  };
});
check("the Collision pane arms a ghost, it does not drop a shape",
  armedColl.ghost && armedColl.kind === "box" && armedColl.made === 0 && armedColl.lit,
  `ghost=${armedColl.ghost} kind=${armedColl.kind} made=${armedColl.made}`);

await page.mouse.click(collMid.x, collMid.y);
await page.waitForTimeout(300);
const droppedColl = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const c = [...ed.state.colliders.values()][0];
  return {
    n: ed.state.colliders.size, id: c?.id, kind: c?.kind,
    pos: c ? c.node.position.asArray().map((v) => +v.toFixed(2)) : null,
    armed: i.ghostActive(),
  };
});
check("clicking lands one collider and stays armed for the next",
  droppedColl.n === 1 && droppedColl.kind === "box" && droppedColl.armed,
  `${droppedColl.n} placed, still armed=${droppedColl.armed}`);

await page.keyboard.press("Escape");
await page.evaluate(async () => (await import("/js/editor.js")).select([]));
await page.waitForTimeout(200);

const collAt = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const c = ed.state.colliders.get(id);
  ed.state.scene.render();
  const e = ed.state.engine;
  const p = BABYLON.Vector3.Project(c.node.getAbsolutePosition(),
    BABYLON.Matrix.Identity(), ed.state.scene.getTransformMatrix(),
    ed.state.camera.viewport.toGlobal(e.getRenderWidth(), e.getRenderHeight()));
  const r = e.getRenderingCanvas().getBoundingClientRect();
  return { x: Math.round(r.x + p.x), y: Math.round(r.y + p.y) };
}, droppedColl.id);

await page.mouse.move(collAt.x, collAt.y);
await page.waitForTimeout(250);
await page.mouse.click(collAt.x, collAt.y);
await page.waitForTimeout(300);
const pickedColl = await page.evaluate(async () =>
  [...(await import("/js/editor.js")).state.selection]);
check("a collider can still be picked after it is deselected",
  pickedColl.length === 1 && pickedColl[0] === droppedColl.id,
  `selection=${JSON.stringify(pickedColl)}`);

await page.mouse.move(collMid.x + 150, collMid.y - 60);
await page.waitForTimeout(150);
await page.keyboard.press("m");
await page.waitForTimeout(400);
const carriedColl = await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  return { ghost: i.ghostActive(), mode: i.ghostMode() };
});
await page.mouse.move(collMid.x + 200, collMid.y - 30);
await page.waitForTimeout(150);
await page.mouse.click(collMid.x + 200, collMid.y - 30);
await page.waitForTimeout(350);
const landedColl = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const c = ed.state.colliders.get(id);
  return {
    n: ed.state.colliders.size, enabled: !!c?.node.isEnabled(),
    pos: c ? c.node.position.asArray().map((v) => +v.toFixed(2)) : null,
  };
}, droppedColl.id);
check("M carries a collider and drops the same one somewhere else",
  carriedColl.mode === "move" && landedColl.n === 1 && landedColl.enabled
    && landedColl.pos.join() !== droppedColl.pos.join(),
  `mode=${carriedColl.mode}, ${droppedColl.pos} -> ${landedColl.pos}`);

// Ctrl+D on a collider arms a copy of the same primitive rather than falling
// back to duplicateSelected(), which is the marker path.
await page.mouse.move(collCanvas.x + 40, collCanvas.y + 40);  // hover beats selection
await page.waitForTimeout(200);
await page.evaluate(async (id) => (await import("/js/editor.js")).select([id]),
  droppedColl.id);
await page.keyboard.press("Control+d");
await page.waitForTimeout(400);
const dupColl = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return { kind: i.ghostCollider(), n: ed.state.colliders.size };
});
check("Ctrl+D on a collider arms a copy of the same primitive",
  dupColl.kind === "box" && dupColl.n === 1, `kind=${dupColl.kind}, ${dupColl.n} existing`);

await page.keyboard.press("Escape");
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
});
await page.waitForTimeout(200);

// ---- 1d-septvicies. a fresh primitive lands corner first -------------------
// A kit module is modelled from its origin, so it rests on the build plane and
// fills whole cells. A primitive is centred on its origin, because Havok wants
// a centre - so without a shift it lands half buried with its faces through the
// middle of a cell.
await page.evaluate(async () => (await import("/js/editor.js")).setGridElevation(0));
const cornerLands = [];
for (const [dx, dy] of [[0, 0], [37, -23], [-91, 61]]) {
  await page.click('#collider-buttons button[data-kind="box"]');
  await page.mouse.move(collMid.x + dx, collMid.y + dy);
  await page.waitForTimeout(250);
  await page.mouse.click(collMid.x + dx, collMid.y + dy);
  await page.waitForTimeout(250);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(120);
  cornerLands.push(await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const list = [...ed.state.colliders.values()];
    const c = list[list.length - 1];
    const p = c.node.position, s = c.node.scaling;
    return { id: c.id, min: [p.x - s.x / 2, p.y - s.y / 2, p.z - s.z / 2].map((v) => +v.toFixed(4)) };
  }));
}
check("a fresh primitive rests on the build plane",
  cornerLands.every((l) => Math.abs(l.min[1]) < 1e-6),
  `base heights ${cornerLands.map((l) => l.min[1]).join(", ")}`);
check("its corner lands on a grid intersection",
  cornerLands.every((l) => Math.abs(l.min[0] - Math.round(l.min[0])) < 1e-6
    && Math.abs(l.min[2] - Math.round(l.min[2])) < 1e-6),
  cornerLands.map((l) => `[${l.min}]`).join(" "));

// The build plane means a primitive's *base*, not its origin, or referencing
// the centre would raise it by half its height on every single grab.
const driftId = cornerLands[0].id;
for (let round = 0; round < 3; round++) {
  await page.evaluate(async (id) => (await import("/js/editor.js")).select([id]), driftId);
  await page.mouse.move(collMid.x + 60, collMid.y + 20);
  await page.waitForTimeout(150);
  await page.keyboard.press("m");
  await page.waitForTimeout(350);
  await page.mouse.click(collMid.x + 60, collMid.y + 20);
  await page.waitForTimeout(300);
}
const drifted = await page.evaluate(async (id) => {
  const c = (await import("/js/editor.js")).state.colliders.get(id);
  return +(c.node.position.y - c.node.scaling.y / 2).toFixed(4);
}, driftId);
check("carrying one does not raise it off the plane", Math.abs(drifted) < 1e-6,
  `base ${drifted} m after three grab/drop rounds`);

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
});
await page.waitForTimeout(200);

// ---- 1d-duodetricies. the shell thickness a flat module is given -----------
// The kit models floors and ceilings as single planes with no depth at all, and
// its walls only 7.5 mm. Both used to come out wrong: a zero axis fell through
// `Math.abs(v) || 1` and became a *metre*, and the shell only ever filled a
// zero axis, so it did nothing at all to anything solid.
const shell = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); co.exitCollisionMode(); ed.clearAll(); ed.select([]);
  ed.loadModuleCollision({}, []);

  const W = "Walls/ShortWall_Band2_Straight";
  const P = "Platforms/Platform_3Plates";
  const flatBounds = await kit.moduleBounds(P);
  const wallBounds = await kit.moduleBounds(W);

  // Fit a box is the one fitter now, and it works on the staging area.
  const fitAt = async (moduleId, thickness) => {
    ed.setConfig("shellThickness", thickness);
    const { entry } = await co.stageModule(moduleId, kit.instantiate, kit.moduleBounds);
    ed.select([entry.id]);
    const r = await co.fitBoxToSelection(kit.moduleBounds);
    const size = r.collider.node.scaling.asArray().map((v) => +v.toFixed(4));
    co.unstageModule(entry.id);
    ed.state.moduleCollision.delete(moduleId);
    return size;
  };
  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  const flatThin = await fitAt(P, 0.008);
  const wallThin = await fitAt(W, 0.008);
  const flatThick = await fitAt(P, 0.05);
  co.exitCollisionMode();

  await ed.undo();                     // undo() restores asynchronously
  await ed.undo();
  const undone = ed.state.config.shellThickness;

  ed.setConfig("shellThickness", 0.033);
  const layout = ed.serialize();
  await ed.deserialize(JSON.parse(JSON.stringify(layout)));
  const roundTrip = ed.state.config.shellThickness;
  const older = JSON.parse(JSON.stringify(layout)); delete older.config;
  await ed.deserialize(older);
  const legacy = ed.state.config.shellThickness;

  ed.setConfig("shellThickness", 0.008);
  ed.clearAll(); ed.select([]); ed.loadModuleCollision({}, []);
  return {
    flatThin, wallThin, flatThick, undone, roundTrip, legacy,
    flatDepth: +(flatBounds.max.y - flatBounds.min.y).toFixed(6),
    wallDepth: +(wallBounds.max.x - wallBounds.min.x).toFixed(6),
    savedConfig: layout.config,
  };
});
check("the kit really does model a floor as a bare plane",
  shell.flatDepth === 0, `local depth ${shell.flatDepth} m`);
check("a flat module gets the shell thickness, not a metre",
  Math.abs(shell.flatThin[1] - 0.008) < 1e-6, `[${shell.flatThin}]`);
check("the shell is a minimum, so a 7.5 mm wall is brought up to it",
  shell.wallDepth < 0.008 && Math.abs(shell.wallThin[0] - 0.008) < 1e-6,
  `a real ${shell.wallDepth} m fitted as [${shell.wallThin}]`);
check("but a module thicker than the shell keeps its own size",
  Math.abs(shell.wallThin[2] - 4) < 1e-3 && Math.abs(shell.wallThin[1] - 1.9981) < 1e-3,
  `wall fitted [${shell.wallThin}]`);
check("the setting drives the fit",
  Math.abs(shell.flatThick[1] - 0.05) < 1e-6, `at 0.05 m: [${shell.flatThick}]`);
check("changing a setting is undoable", Math.abs(shell.undone - 0.008) < 1e-9,
  `undo -> ${shell.undone}`);
check("a setting round-trips through the layout",
  Math.abs(shell.roundTrip - 0.033) < 1e-9 && Math.abs(shell.legacy - 0.008) < 1e-9,
  `saved ${JSON.stringify(shell.savedConfig)}, reloaded ${shell.roundTrip},`
  + ` a layout without one ${shell.legacy}`);
// ---- 1d-undetricies. the collision staging area ---------------------------
// A mode, not a property of the selection. It opens empty, you stage whatever
// modules you want to fit shapes to, and which element a shape belongs to is
// decided by where it sits - which is what makes copying a similar hull work.
const PROP = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  return [...kit.getCatalogue().byId.keys()].filter((id) => id.startsWith("Props/"));
});
const [PROP_A, PROP_B] = PROP;

const areaOpen = await page.evaluate(async (a) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); co.exitCollisionMode(); ed.clearAll(); ed.select([]);
  ed.loadModuleCollision({}, []);
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  await ed.placeAt(a, new V(8, 0, 0), { silent: true });
  co.addCollider("box", new V(-6, 0.5, 0), { silent: true });
  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  return {
    mode: ed.state.collisionMode,
    staged: [...ed.state.placements.values()].filter((p) => p.stage).length,
    shipShown: [...ed.state.placements.values()].filter((p) => !p.stage && p.node.isEnabled()).length,
    roomShapesShown: [...ed.state.colliders.values()].filter((c) => !c.stage && c.node.isEnabled()).length,
  };
}, PROP_A);
check("the collision area opens empty, with the ship off screen",
  areaOpen.mode && areaOpen.staged === 0 && areaOpen.shipShown === 0
    && areaOpen.roomShapesShown === 0, JSON.stringify(areaOpen));

const stagedTwo = await page.evaluate(async ([a, b]) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const first = await co.stageModule(a, kit.instantiate, kit.moduleBounds);
  await co.stageModule(b, kit.instantiate, kit.moduleBounds);
  const again = await co.stageModule(a, kit.instantiate, kit.moduleBounds);
  const list = [...ed.state.placements.values()].filter((p) => p.stage);
  const boxes = list.map((p) => ed.worldBounds(p.node));
  let gap = Infinity;
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      gap = Math.min(gap, Math.max(boxes[i].min.x, boxes[j].min.x)
        - Math.min(boxes[i].max.x, boxes[j].max.x));
    }
  }
  return {
    n: list.length, addedAgain: again.added, same: again.entry.id === first.entry.id,
    gap: Math.round(gap * 100) / 100,
    chunkHidden: !ed.state.chunks.includes(ed.STAGE_CHUNK),
  };
}, [PROP_A, PROP_B]);
check("one instance per module, and re-staging returns the one already there",
  stagedTwo.n === 2 && stagedTwo.addedAgain === false && stagedTwo.same,
  `${stagedTwo.n} staged`);
check("staged elements are spaced beyond twice the association margin",
  stagedTwo.gap > 2 * 0.5, `${stagedTwo.gap} m clear, margin is 0.5 m`);
check("the stage chunk never joins the ship's chunk list", stagedTwo.chunkHidden);

// Fit a box acts on the selection and says plainly when it cannot
const fitRules = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const out = {};
  ed.select([]);
  out.none = (await co.fitBoxToSelection(kit.moduleBounds)).error;
  const list = [...ed.state.placements.values()].filter((p) => p.stage);
  ed.select(list.map((p) => p.id));
  out.many = (await co.fitBoxToSelection(kit.moduleBounds)).error;
  const shape = co.addCollider("sphere", new BABYLON.Vector3(300, 0, 0), { stage: true, silent: true });
  ed.select([shape.id]);
  out.onShape = (await co.fitBoxToSelection(kit.moduleBounds)).error;
  co.removeCollider(shape.id, true);

  // the shell is a MINIMUM thickness, so it applies to every axis - it used to
  // fill a zero-depth axis only, which made it look broken on anything solid
  ed.setConfig("shellThickness", 0.5);
  ed.select([list[0].id]);
  const r = await co.fitBoxToSelection(kit.moduleBounds);
  const bounds = await kit.moduleBounds(list[0].module);
  out.ok = r.ok;
  out.fitted = r.collider.node.scaling.asArray().map((v) => +v.toFixed(3));
  out.raw = bounds.max.subtract(bounds.min).asArray().map((v) => +v.toFixed(3));
  ed.setConfig("shellThickness", 0.008);
  return out;
});
check("fitting with nothing, or several, selected explains itself",
  /select the element/.test(fitRules.none) && /single element/.test(fitRules.many),
  `${fitRules.none} / ${fitRules.many}`);
check("fitting onto a shape rather than an element explains itself",
  /collision shape/.test(fitRules.onShape), fitRules.onShape);
check("the collision shell is a minimum thickness on every axis",
  fitRules.ok && fitRules.fitted.every((v) => v >= 0.5 - 1e-6)
    && fitRules.fitted.some((v, i) => Math.abs(v - fitRules.raw[i]) > 1e-6),
  `raw ${JSON.stringify(fitRules.raw)} -> ${JSON.stringify(fitRules.fitted)} at shell 0.5`);

// association is by position, which is what makes copying a hull work
const assoc = await page.evaluate(async ([a, b]) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const list = [...ed.state.placements.values()].filter((p) => p.stage);
  const ea = list.find((p) => p.module === a), eb = list.find((p) => p.module === b);
  const src = co.stageColliders()[0];
  const delta = eb.node.position.subtract(ea.node.position);
  co.addCollider(src.kind, src.node.position.add(delta),
    { stage: true, silent: true, scale: src.node.scaling.asArray() });
  const h = co.harvestStage();
  const stray = co.addCollider("box", new BABYLON.Vector3(0, 0, 400), { stage: true, silent: true });
  const orphans = co.orphanCount();
  co.removeCollider(stray.id, true);
  return {
    h, orphans,
    aLocal: (ed.state.moduleCollision.get(a) || [])[0]?.position,
    bLocal: (ed.state.moduleCollision.get(b) || [])[0]?.position,
    counts: [(ed.state.moduleCollision.get(a) || []).length,
      (ed.state.moduleCollision.get(b) || []).length],
  };
}, [PROP_A, PROP_B]);
check("a shape is claimed by the element it sits on",
  assoc.counts.join() === "1,1" && assoc.h.orphans === 0, JSON.stringify(assoc.counts));
check("stored relative to it, so a copy dropped on another element matches",
  JSON.stringify(assoc.aLocal) === JSON.stringify(assoc.bLocal),
  `${JSON.stringify(assoc.aLocal)} vs ${JSON.stringify(assoc.bLocal)}`);
check("a shape belonging to nothing is counted rather than hidden",
  assoc.orphans === 1, `${assoc.orphans}`);

// taking an element off the area must not lose what was fitted to it
const persist = await page.evaluate(async (a) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const el = [...ed.state.placements.values()].find((p) => p.stage && p.module === a);
  const before = (ed.state.moduleCollision.get(a) || []).length;
  ed.select([el.id]);
  ed.removeSelected();                       // the Del path, not the API
  const kept = (ed.state.moduleCollision.get(a) || []).length;
  const leftOnArea = co.stageColliders().length;
  await co.stageModule(a, kit.instantiate, kit.moduleBounds);
  return { before, kept, leftOnArea, restored: co.stageColliders().length };
}, PROP_A);
check("deleting a staged element keeps its shapes on record",
  persist.before === 1 && persist.kept === 1, `${persist.before} -> ${persist.kept}`);
check("and takes only its own shapes off the area",
  persist.leftOnArea === 1, `${persist.leftOnArea} left`);
check("staging the module again brings them back",
  persist.restored === 2, `${persist.restored} on the area`);

// Moving or turning a stand-in is a *view* operation: the hull is authored in
// the module's own frame, so the shapes come with it and the record does not
// change. Left alone the element slid out from under its shapes, which then
// belonged to nothing - and deleting it afterwards could not find them either.
const follow = await page.evaluate(async (prop) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const V = BABYLON.Vector3;
  const frame = () => new Promise((r) =>
    ed.state.scene.onAfterRenderObservable.addOnce(() => r()));
  const el = [...ed.state.placements.values()].find((p) => p.stage && p.module === prop);
  const shapes = co.stageColliders().filter((c) => c.host === el.id);
  await frame(); await frame();
  const was = {
    shape: shapes[0].node.position.asArray().map((v) => +v.toFixed(3)),
    record: JSON.stringify(ed.state.moduleCollision.get(prop)),
    hosted: shapes.length,
  };

  el.node.position.addInPlace(new V(5, 0, 3));
  await frame(); await frame();
  const moved = {
    shape: shapes[0].node.position.asArray().map((v) => +v.toFixed(3)),
    record: JSON.stringify(ed.state.moduleCollision.get(prop)),
  };

  el.node.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(0, Math.PI / 2, 0);
  await frame(); await frame();
  const turned = {
    q: shapes[0].node.rotationQuaternion.asArray().map((v) => +v.toFixed(3)),
    record: JSON.stringify(ed.state.moduleCollision.get(prop)),
  };

  ed.select([el.id]);
  ed.removeSelected();
  await frame();
  const deleted = {
    staged: [...ed.state.placements.values()].filter((p) => p.stage && p.module === prop).length,
    left: co.stageColliders().filter((c) => c.host === el.id).length,
    stored: (ed.state.moduleCollision.get(prop) || []).length,
  };
  // put the bench back as the blocks after this one expect to find it
  const kit = await import("/js/kit.js");
  await co.stageModule(prop, kit.instantiate, kit.moduleBounds);
  await frame();
  return { was, moved, turned, deleted };
}, PROP_A);
check("a shape knows which staged element it belongs to",
  follow.was.hosted >= 1, `${follow.was.hosted} hosted`);
check("moving a staged element carries its shapes with it",
  follow.moved.shape[0] === +(follow.was.shape[0] + 5).toFixed(3)
    && follow.moved.shape[2] === +(follow.was.shape[2] + 3).toFixed(3),
  `${JSON.stringify(follow.was.shape)} -> ${JSON.stringify(follow.moved.shape)}`);
check("turning it turns them too",
  Math.abs(Math.abs(follow.turned.q[1]) - 0.707) < 0.01, `q=${JSON.stringify(follow.turned.q)}`);
check("and neither changes the hull that was authored",
  follow.moved.record === follow.was.record && follow.turned.record === follow.was.record,
  "record held through a move and a turn");
check("deleting a moved element still takes its shapes off the bench",
  follow.deleted.staged === 0 && follow.deleted.left === 0 && follow.deleted.stored >= 1,
  JSON.stringify(follow.deleted));

// the whole point: none of this reaches the ship
const stageLeak = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const man = mf.buildManifest();
  const layout = ed.serialize();
  const realFetch = window.fetch;
  let body = null;
  window.fetch = (url, opts) => {
    if (String(url).includes("/api/export")) {
      body = opts.body;
      return Promise.resolve(new Response('{"ok":true,"bytes":0}',
        { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
  try { await mf.exportGlb(); } finally { window.fetch = realFetch; }
  const buf = await body.arrayBuffer();
  const dv = new DataView(buf);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, dv.getUint32(12, true))));
  const ids = [...ed.state.placements.values()].filter((p) => p.stage).map((p) => p.id);
  const names = (json.nodes || []).map((n) => String(n.name));
  return {
    staged: ids.length,
    inInstances: man.instances.filter((i) => ids.includes(i.id)).length,
    inLayout: layout.instances.filter((i) => ids.includes(i.id)).length,
    inChunks: man.chunks.filter((c) => String(c.id).includes("stage")).length,
    inGlb: names.filter((n) => ids.some((id) => n.startsWith(id))).length,
    roomColliders: (layout.colliders || []).length,
    modules: Object.keys(man.moduleCollision || {}).length,
    objectCount: document.getElementById("status-counts").textContent,
  };
});
check("the staging area reaches neither the manifest, the layout nor the .glb",
  stageLeak.staged === 2 && stageLeak.inInstances === 0 && stageLeak.inLayout === 0
    && stageLeak.inChunks === 0 && stageLeak.inGlb === 0,
  `${stageLeak.staged} staged, ${stageLeak.inInstances}/${stageLeak.inLayout}/${stageLeak.inGlb} leaked`);
check("staged shapes are not serialized as the ship's own",
  stageLeak.roomColliders === 1, `${stageLeak.roomColliders}`);
check("the per-module record is what the manifest carries",
  stageLeak.modules === 2, `${stageLeak.modules} modules`);
check("and the object count still counts the ship, not the area",
  /^2 objects/.test(stageLeak.objectCount), stageLeak.objectCount.split("·")[0].trim());

// X and H work here like anywhere else
const modeTools = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const el = [...ed.state.placements.values()].find((p) => p.stage);
  const shape = co.stageColliders()[0];
  // Count the gizmo's own arms, and take showAxes at its word. Counting meshes
  // whose name merely contains "axis" was too weak to notice that X did nothing
  // at all on a collision shape - axesNode() looked in placements and markers
  // and never in colliders.
  const arms = () => ["x", "y", "z"]
    .every((a) => ed.state.scene.transformNodes.some((n) => n.name === `AXES_${a}`));
  ed.hideAxes();
  const onElement = ed.showAxes(el.id) && arms();
  ed.hideAxes();
  const onShape = ed.showAxes(shape.id) && arms();
  ed.hideAxes();
  const viaToggle = !!ed.toggleAxes(shape.id) && arms();   // the path X takes
  ed.hideAxes();
  ed.select([el.id]); ed.hideSelected("ghost"); const veiled = ed.veilCounts().ghost;
  ed.select([el.id]); ed.hideSelected("hidden"); const gone = !el.node.isEnabled();
  ed.unhideAll();
  return { onElement, onShape, viaToggle, veiled, gone, back: el.node.isEnabled() };
});
check("X draws axes on staged elements and on staged shapes",
  modeTools.onElement && modeTools.onShape && modeTools.viaToggle,
  `element=${modeTools.onElement} shape=${modeTools.onShape} toggle=${modeTools.viaToggle}`);

// The right button is also the camera button. It cancels what is in your hand,
// like Escape, but must never close the bench you are working on.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.select([]);
});
await page.mouse.move(collMid.x, collMid.y);
await page.mouse.down({ button: "right" });
await page.mouse.up({ button: "right" });
await page.waitForTimeout(400);
const benchAfterRmb = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return {
    mode: ed.state.collisionMode,
    staged: [...ed.state.placements.values()].filter((p) => p.stage).length,
  };
});
check("the right button does not close the collision area",
  benchAfterRmb.mode === true && benchAfterRmb.staged === 2, JSON.stringify(benchAfterRmb));

// but it still puts down an armed shape
await page.click('#collider-buttons button[data-kind="sphere"]');
await page.mouse.move(collMid.x, collMid.y);
await page.waitForTimeout(300);
const wasArmed = await page.evaluate(async () => (await import("/js/interact.js")).ghostActive());
await page.mouse.down({ button: "right" });
await page.mouse.up({ button: "right" });
await page.waitForTimeout(400);
const benchAfterCancel = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return { ghost: i.ghostActive(), mode: ed.state.collisionMode };
});
check("the right button still puts down an armed shape",
  wasArmed === true && benchAfterCancel.ghost === false && benchAfterCancel.mode === true,
  `armed=${wasArmed}, ${JSON.stringify(benchAfterCancel)}`);
check("H veils and hides staged elements, and gives them back",
  modeTools.veiled >= 1 && modeTools.gone && modeTools.back,
  `veiled=${modeTools.veiled} hidden=${modeTools.gone} restored=${modeTools.back}`);

// leaving keeps everything, and collision travels in its own file
const closed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const mf = await import("/js/manifest.js");
  co.exitCollisionMode();
  const after = {
    mode: ed.state.collisionMode,
    staged: [...ed.state.placements.values()].filter((p) => p.stage).length,
    stageShapes: co.stageColliders().length,
    shipShown: [...ed.state.placements.values()].filter((p) => p.node.isEnabled()).length,
    stored: ed.state.moduleCollision.size,
  };

  // The collision file is server state shared by the whole run, so put back
  // exactly what was there. Saving the *ship* here would be worse still: it
  // would become what every later boot restores, and the suite would stop
  // being repeatable - which is precisely how it first went wrong.
  const original = await (await fetch("/api/collision")).json();
  await mf.saveCollision();
  const onDisk = await (await fetch("/api/collision")).json();
  ed.loadModuleCollision({});
  const wiped = ed.state.moduleCollision.size;
  const back = await mf.loadCollision();
  const restored = ed.state.moduleCollision.size;
  await fetch("/api/collision", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(original),
  });

  ed.clearAll(); ed.select([]); ed.loadModuleCollision({});
  return {
    after, wiped, restored,
    onDisk: Object.keys(onDisk.moduleShapes || {}).length,
    schema: onDisk.schema,
    backKeys: back ? Object.keys(back).length : 0,
  };
});
check("leaving clears the area and puts the ship back",
  !closed.after.mode && closed.after.staged === 0 && closed.after.stageShapes === 0
    && closed.after.shipShown === 2, JSON.stringify(closed.after));
check("everything fitted survives the way out", closed.after.stored === 2);
check("collision is written to its own file, apart from the ship",
  closed.onDisk === 2 && closed.schema === 1, `${closed.onDisk} modules, schema ${closed.schema}`);
check("and that file alone can restore it",
  closed.wiped === 0 && closed.restored === 2 && closed.backKeys === 2,
  `wiped to ${closed.wiped}, back to ${closed.restored}`);

// ---- 1d-trestricies. the bench keeps what you left on it ------------------
// Coming back to a blank stage after stepping out to look at the ship was the
// wrong default: the area is a workbench. The roster rides in the collision
// file, so it survives a reload as well as a trip back to the ship.
const bench = await page.evaluate(async ([a, b]) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]); ed.loadModuleCollision({}, []);
  await ed.placeAt(a, new V(0, 0, 0), { silent: true });
  await ed.placeAt(a, new V(8, 0, 0), { silent: true });

  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  await co.stageModule(a, kit.instantiate, kit.moduleBounds);
  await co.stageModule(b, kit.instantiate, kit.moduleBounds);
  const first = [...ed.state.placements.values()].filter((p) => p.stage);
  ed.select([first[0].id]);
  await co.fitBoxToSelection(kit.moduleBounds);
  const placed = first.map((p) => p.node.position.asArray().map((v) => +v.toFixed(3)));

  co.exitCollisionMode();
  const roster = ed.state.stageLayout.map((s) => s.module);
  const rosterAt = ed.state.stageLayout.map((s) => s.position);

  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  const back = [...ed.state.placements.values()].filter((p) => p.stage);
  const out = {
    roster, rosterAt, placed,
    backModules: back.map((p) => p.module),
    backAt: back.map((p) => p.node.position.asArray().map((v) => +v.toFixed(3))),
    shapes: co.stageColliders().length,
  };
  co.exitCollisionMode();
  return out;
}, [PROP_A, PROP_B]);
const sameSpots = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length
  && a.every((v, i) => (Array.isArray(v)
    ? sameSpots(v, b[i]) : Math.abs(v - b[i]) < 1e-2));
check("closing the area records what was on it",
  bench.roster.length === 2 && sameSpots(bench.rosterAt, bench.placed),
  `${JSON.stringify(bench.roster)} at ${JSON.stringify(bench.rosterAt)}`);
check("re-opening puts the same modules back in the same places",
  JSON.stringify(bench.backModules) === JSON.stringify(bench.roster)
    && sameSpots(bench.backAt, bench.placed),
  `${JSON.stringify(bench.backAt)} vs ${JSON.stringify(bench.placed)}`);
check("and their shapes come back with them", bench.shapes === 1, `${bench.shapes}`);

// ---- 1d-quattuortricies. inherited collision is visible on the ship -------
// A module's shapes are instanced onto every placement at export time, so the
// ship carried collision that was drawn nowhere but the staging area. On a ship
// whose collision is all inherited, "Collision only" showed an empty room -
// which reads exactly like a broken switch.
const inherited = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const settle = () => new Promise((r) => setTimeout(r, 60));
  ed.setShowLayer("both"); ed.applyVisibility(); await settle();
  const both = co.previewCount();
  ed.setShowLayer("collision"); ed.applyVisibility(); await settle();
  const collisionOnly = {
    preview: co.previewCount(),
    geo: [...ed.state.placements.values()].filter((p) => p.node.isEnabled()).length,
  };
  ed.setShowLayer("geometry"); ed.applyVisibility(); await settle();
  const geometryOnly = co.previewCount();

  // it follows isolation like everything else, being built through applyVisibility
  ed.setShowLayer("both");
  ed.state.isolate = true;
  ed.state.chunks = [...new Set([...ed.state.chunks, "CH_Nowhere"])];
  ed.state.activeChunk = "CH_Nowhere";
  ed.applyVisibility(); await settle();
  const isolatedAway = co.previewCount();
  ed.state.isolate = false;
  ed.state.activeChunk = ed.state.chunks[0];
  ed.applyVisibility(); await settle();

  // and never reaches the .glb
  const mf = await import("/js/manifest.js");
  const realFetch = window.fetch;
  let body = null;
  window.fetch = (url, opts) => {
    if (String(url).includes("/api/export")) {
      body = opts.body;
      return Promise.resolve(new Response('{"ok":true,"bytes":0}',
        { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
  try { await mf.exportGlb(); } finally { window.fetch = realFetch; }
  const buf = await body.arrayBuffer();
  const dv = new DataView(buf);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, dv.getUint32(12, true))));
  const stowaways = (json.nodes || []).map((n) => String(n.name))
    .filter((n) => /PREVIEW/i.test(n)).length;

  ed.clearAll(); ed.select([]); ed.loadModuleCollision({}, []);
  ed.applyVisibility(); await settle();
  return { both, collisionOnly, geometryOnly, isolatedAway, stowaways, cleared: co.previewCount() };
});
check("inherited collision is drawn on every placement of its module",
  inherited.both === 2, `${inherited.both} shapes for 2 placements`);
check("Collision only shows it with the ship off screen",
  inherited.collisionOnly.preview === 2 && inherited.collisionOnly.geo === 0,
  JSON.stringify(inherited.collisionOnly));
check("Ship only takes it off screen too", inherited.geometryOnly === 0);
check("it follows chunk isolation, being built through applyVisibility",
  inherited.isolatedAway === 0, `${inherited.isolatedAway} still drawn`);
check("it is a preview, not data: never exported, and gone when the record is",
  inherited.stowaways === 0 && inherited.cleared === 0,
  `${inherited.stowaways} in the .glb, ${inherited.cleared} left after clearing`);

// ---- 1d-quintricies. the palette says which modules are done --------------
// Fitting a kit is a job you do a few modules at a time and come back to, and
// there are 277 of them. Without a mark, the only way to tell which were done
// was to stage each one and look.
const marks = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const settle = () => new Promise((r) => setTimeout(r, 120));
  ed.clearAll(); ed.select([]); ed.loadModuleCollision({}, []);
  await settle();
  const target = [...kit.getCatalogue().byId.keys()].find((m) => m.startsWith("Props/"));
  const tile = () => document.querySelector(`#palette-list .item[data-id="${target}"]`);
  const lit = () => !!tile()?.classList.contains("has-collision");

  const before = lit();
  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  const { entry } = await co.stageModule(target, kit.instantiate, kit.moduleBounds);
  ed.select([entry.id]);
  await co.fitBoxToSelection(kit.moduleBounds);
  await settle();
  const after = lit();
  const title = tile()?.title;

  // and it goes out again when the shapes do
  for (const c of co.stageColliders()) co.removeCollider(c.id, true);
  co.harvestStage();
  ed.emit("colliders");
  await settle();
  const cleared = lit();
  co.exitCollisionMode();
  ed.clearAll(); ed.select([]); ed.loadModuleCollision({}, []);
  await settle();
  return { target, before, after, cleared, title, stillLit: lit() };
});
check("a module with collision is marked in the palette",
  marks.before === false && marks.after === true,
  `${marks.target}: ${marks.before} -> ${marks.after}`);
check("and the tile says how many shapes it carries",
  /1 collision shape\b/.test(marks.title || ""), marks.title);
check("the mark goes out when the shapes do",
  marks.cleared === false && marks.stillLit === false,
  `after removing ${marks.cleared}, after clearing ${marks.stillLit}`);


// ---- 1d-duotricies. the geometry / collision layer switch -----------------
// It can only ever take things off screen, so chunk isolation and the Shift+H
// veil keep the last word - which is why it goes through applyVisibility()
// rather than reaching for setEnabled itself.
const layers = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  co.addCollider("box", new V(2, 0.5, 0), { silent: true });
  co.addCollider("sphere", new V(4, 0.5, 0), { silent: true, scale: [2, 2, 2] });
  const shown = () => ({
    geo: [...ed.state.placements.values()].filter((e) => e.node.isEnabled()).length,
    col: [...ed.state.colliders.values()].filter((c) => c.node.isEnabled()).length,
  });
  const both = shown();
  ed.setShowLayer("geometry"); const geoOnly = shown();
  ed.setShowLayer("collision"); const colOnly = shown();

  // selecting something and then hiding its layer must drop it, or the gizmo
  // and the inspector act on an element nobody can see
  ed.setShowLayer("both");
  ed.select([[...ed.state.placements.keys()][0]]);
  ed.setShowLayer("collision");
  const afterHidingGeometry = [...ed.state.selection];
  ed.select([[...ed.state.colliders.keys()][0]]);
  ed.setShowLayer("geometry");
  const afterHidingCollision = [...ed.state.selection];

  // and it composes with isolation rather than overriding it
  ed.setShowLayer("both");
  ed.state.isolate = true;
  ed.state.activeChunk = "CH_Elsewhere";
  ed.state.chunks = [...new Set([...ed.state.chunks, "CH_Elsewhere"])];
  ed.applyVisibility();
  const isolatedElsewhere = shown();
  ed.state.isolate = false;
  ed.state.activeChunk = ed.state.chunks[0];
  ed.setShowLayer("both");
  ed.applyVisibility();
  ed.clearAll(); ed.select([]);
  return { both, geoOnly, colOnly, afterHidingGeometry, afterHidingCollision, isolatedElsewhere };
});
check("both layers show by default",
  layers.both.geo === 1 && layers.both.col === 2, JSON.stringify(layers.both));
check("ship only takes the collision off screen",
  layers.geoOnly.geo === 1 && layers.geoOnly.col === 0, JSON.stringify(layers.geoOnly));
check("collision only takes the ship off screen",
  layers.colOnly.geo === 0 && layers.colOnly.col === 2, JSON.stringify(layers.colOnly));
check("hiding a layer drops any selection it hides",
  layers.afterHidingGeometry.length === 0 && layers.afterHidingCollision.length === 0,
  `${JSON.stringify(layers.afterHidingGeometry)} / ${JSON.stringify(layers.afterHidingCollision)}`);
check("isolation still gets the last word over the switch",
  layers.isolatedElsewhere.geo === 0 && layers.isolatedElsewhere.col === 0,
  JSON.stringify(layers.isolatedElsewhere));

// ---- 1d-tertricies. what the inspector measures for a primitive -----------
// Havok's own parameters, not a bounding box: a world AABB would show a sphere
// as three identical sides, and a turned capsule as something with no relation
// to the radius it is actually built from.
const primDims = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const read = () => [
    document.getElementById("dim-x").textContent,
    document.getElementById("dim-y").textContent,
    document.getElementById("dim-z").textContent,
    document.getElementById("dim-note").textContent,
  ];
  const out = {};
  const box = co.addCollider("box", new V(0, 0, 0), { silent: true, scale: [4, 2, 0.2] });
  ed.select([box.id]); out.box = read();
  const sph = co.addCollider("sphere", new V(6, 0, 0), { silent: true, scale: [3, 3, 3] });
  ed.select([sph.id]); out.sphere = read();
  const cap = co.addCollider("capsule", new V(12, 0, 0), { silent: true, scale: [2, 5, 2] });
  cap.node.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(Math.PI / 2, 0, 0);
  ed.select([cap.id]); out.capsule = read();
  const wall = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 20), { silent: true });
  ed.select([wall.id]); out.wall = read();
  ed.clearAll(); ed.select([]);
  return out;
});
check("a box reports its three sides",
  primDims.box[0] === "4.00" && primDims.box[1] === "2.00" && primDims.box[2] === "0.20"
    && primDims.box[3].includes("X/Y/Z"), JSON.stringify(primDims.box));
check("a sphere reports one radius and says so",
  primDims.sphere[0] === "1.50" && primDims.sphere[1] === "—" && primDims.sphere[3].includes("radius"),
  JSON.stringify(primDims.sphere));
check("a turned capsule still reports radius and height",
  primDims.capsule[0] === "1.00" && primDims.capsule[1] === "5.00" && primDims.capsule[2] === "—",
  JSON.stringify(primDims.capsule));
check("a kit module still reports its bounding box",
  primDims.wall[3].includes("X/Y/Z") && primDims.wall[1] === "2.00", JSON.stringify(primDims.wall));

// ---- 1d-terdecies. the behaviour library and the entities that use it ------
// `behaviors` is a library of named definitions; `entities` says which node
// names carry which, plus the `linked` names some of them need. Both are keyed
// by node name, which is why names must NOT be made unique.
const bhv = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  const M = "Walls/ShortWall_Band2_Straight";
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  const a = await ed.placeAt(M, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(M, new V(4, 0, 0), { silent: true });
  const c = await ed.placeAt(M, new V(8, 0, 0), { silent: true });
  ed.renamePlacement(a.id, "crate");
  ed.renamePlacement(b.id, "crate");       // same name on purpose
  ed.renamePlacement(c.id, "doorL");

  // the body is free-form JSON: the runtime owns which flags exist
  const blank = ed.setBehaviorDef("   ", { dynamic: true });
  const notObject = ed.setBehaviorDef("bad", [1, 2, 3]);
  ed.setBehaviorDef("any_liquefiable", { liquefiable: true });
  ed.setBehaviorDef("door_liquefiable",
    { liquefiable: true, fluidSim: ["viscosity-inplace"], whatever: { nested: 1 } });
  ed.setBehaviorDef("dynamic", { dynamic: true });

  ed.addEntityBehavior("crate", "any_liquefiable");
  const twice = ed.addEntityBehavior("crate", "any_liquefiable");   // no duplicates
  const unknown = ed.addEntityBehavior("crate", "nope");
  ed.addEntityBehavior("doorL", "door_liquefiable");
  ed.setEntityLinked("doorL", "door_liquefiable", ["crate", "doorL", "crate"]);

  const man = mf.buildManifest();
  const written = JSON.parse(JSON.stringify(man.behaviors));
  const entities = JSON.parse(JSON.stringify(man.entities));

  // renaming a definition has to carry every reference with it
  ed.renameBehaviorDef("any_liquefiable", "meltable");
  const afterRename = ed.entityBehaviors("crate").map((x) => x.name);
  // and deleting one has to strip it from the entities that carried it
  ed.deleteBehaviorDef("meltable");
  const afterDelete = { list: ed.behaviorNames(), crate: ed.entityBehaviors("crate") };

  // round trip through the undo snapshot
  const snapshot = JSON.parse(JSON.stringify(ed.serialize()));
  ed.deleteBehaviorDef("door_liquefiable");
  const wiped = ed.entityBehaviors("doorL").length;
  await ed.deserialize(snapshot);
  const restored = ed.entityBehaviors("doorL");

  const inRoom = ed.nodeNamesInChunk(ed.state.placements.get(c.id).chunk, "doorL");
  const liquefies = [ed.isLiquefiable("door_liquefiable"), ed.isLiquefiable("dynamic")];
  const global = man.fluidSim;
  ed.clearAll(); ed.select([]);
  const afterClearAll = { defs: ed.behaviorNames(), ents: ed.entityBehaviors("crate") };
  return { blank, notObject, twice, unknown, written, entities, afterRename, afterDelete,
    wiped, restored, inRoom, liquefies, global, afterClearAll };
});
check("a definition needs a name, and a JSON object for a body",
  bhv.blank === false && bhv.notObject === false,
  `blank=${bhv.blank}, array=${bhv.notObject}`);
check("definition bodies are written through untouched",
  JSON.stringify(bhv.written.door_liquefiable)
    === JSON.stringify({ liquefiable: true, fluidSim: ["viscosity-inplace"], whatever: { nested: 1 } }),
  JSON.stringify(bhv.written.door_liquefiable));
check("entities list the behaviours they carry",
  bhv.entities.crate.behaviors[0].name === "any_liquefiable"
    && !("linked" in bhv.entities.crate.behaviors[0]),
  JSON.stringify(bhv.entities.crate));
check("linked is de-duplicated, drops self, and is omitted when empty",
  bhv.entities.doorL.behaviors[0].linked.join() === "crate",
  JSON.stringify(bhv.entities.doorL.behaviors[0]));
check("a behaviour cannot be attached twice, or when it does not exist",
  bhv.twice === false && bhv.unknown === false,
  `twice=${bhv.twice}, unknown=${bhv.unknown}`);
check("renaming a definition carries every reference with it",
  bhv.afterRename.join() === "meltable", `[${bhv.afterRename}]`);
check("deleting a definition strips it from the entities that used it",
  !bhv.afterDelete.list.includes("meltable") && bhv.afterDelete.crate.length === 0,
  `library [${bhv.afterDelete.list}], crate ${JSON.stringify(bhv.afterDelete.crate)}`);
check("behaviours ride the undo stack",
  bhv.wiped === 0 && bhv.restored[0]?.name === "door_liquefiable"
    && bhv.restored[0].linked.join() === "crate",
  `wiped=${bhv.wiped}, restored=${JSON.stringify(bhv.restored)}`);
check("only liquefiable definitions ask for linked names",
  bhv.liquefies[0] === true && bhv.liquefies[1] === false, JSON.stringify(bhv.liquefies));
check("the linked candidates are the other named nodes in the room",
  bhv.inRoom.join() === "crate", `[${bhv.inRoom}]`);
check("the manifest carries the global sim list",
  Array.isArray(bhv.global) && bhv.global.includes("liquid-slow.json"),
  JSON.stringify(bhv.global));
check("clearing the layout forgets the library and the entities",
  bhv.afterClearAll.defs.length === 0 && bhv.afterClearAll.ents.length === 0,
  JSON.stringify(bhv.afterClearAll));

// a manifest from before `entities` existed keyed behaviours by node name, so
// the bodies must be kept AND applied to the node they were named after
const legacy = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
  await ed.deserialize({
    chunks: ["CH00_Storage"], activeChunk: "CH00_Storage", instances: [], markers: [],
    behaviors: { crate2: { liquefiable: true }, crate4: { liquefiable: true } },
  });
  const migrated = {
    defs: ed.behaviorNames(),
    crate2: ed.entityBehaviors("crate2").map((b) => b.name),
  };
  // …but a snapshot that *has* the key, even empty, is the new format
  await ed.deserialize({
    chunks: ["CH00_Storage"], activeChunk: "CH00_Storage", instances: [], markers: [],
    behaviors: { spare: { dynamic: true } }, entities: {},
  });
  const modern = { defs: ed.behaviorNames(), spare: ed.entityBehaviors("spare").length };
  ed.clearAll(); ed.select([]);
  return { migrated, modern };
});
check("an old node-keyed behaviours block is applied, not just kept",
  legacy.migrated.defs.join() === "crate2,crate4"
    && legacy.migrated.crate2.join() === "crate2",
  JSON.stringify(legacy.migrated));
check("an empty entities block means the new format, and applies nothing",
  legacy.modern.defs.join() === "spare" && legacy.modern.spare === 0,
  JSON.stringify(legacy.modern));

// the ship's own sim list wins over config.json: config is what a *new* ship
// starts from, but a saved one carries the list it was authored against
const simList = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const fromConfig = [...ed.state.fluidSim];
  const base = { chunks: ["CH00_Storage"], activeChunk: "CH00_Storage", instances: [], markers: [] };

  await ed.deserialize({ ...base, fluidSim: ["ship-only.json", "thick.json"] });
  const loaded = [...ed.state.fluidSim];
  const written = mf.buildManifest().fluidSim;

  // an undo snapshot carries no list, and must not wipe the one in force
  await ed.deserialize(base);
  const afterUndo = [...ed.state.fluidSim];

  ed.state.fluidSim = fromConfig;
  ed.clearAll(); ed.select([]);
  return { fromConfig, loaded, written, afterUndo };
});
check("the manifest's fluidSim list is exported",
  simList.written.join() === "ship-only.json,thick.json", `[${simList.written}]`);
check("a loaded list takes precedence over the one from config.json",
  simList.loaded.join() === "ship-only.json,thick.json"
    && simList.fromConfig.join() !== simList.loaded.join(),
  `config [${simList.fromConfig}] -> ship [${simList.loaded}]`);
check("a snapshot without a list leaves the one in force alone",
  simList.afterUndo.join() === "ship-only.json,thick.json", `[${simList.afterUndo}]`);

// ---- 1d-quaterdecies. the behaviour panel and the library dialog ----------
// The panel edits the *name* in the field, not the element, so giving a second
// element the same name has to bring the first one's behaviours up on blur.
const bhvIds = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const M = "Walls/ShortWall_Band2_Straight";
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt(M, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(M, new V(6, 0, 0), { silent: true });
  ed.select([a.id]);
  return { a: a.id, b: b.id };
});
await page.waitForTimeout(200);

const readPanel = () => page.evaluate(() => ({
  shown: !document.getElementById("behavior-fields").hidden,
  applied: [...document.querySelectorAll("#bhv-applied .item .n")].map((e) => e.textContent),
  linked: [...document.querySelectorAll("#bhv-applied [data-linked]")]
    .map((s) => [...s.selectedOptions].map((o) => o.value).join("|")),
  hasLinkedPicker: !!document.querySelector("#bhv-applied [data-linked]"),
  addOff: document.getElementById("btn-bhv-add").disabled,
  count: document.getElementById("bhv-count").textContent,
  hint: document.getElementById("bhv-hint").textContent,
}));

const unnamed = await readPanel();
check("an unnamed element cannot be given a behaviour",
  unnamed.shown && unnamed.addOff && /Name the element first/.test(unnamed.hint),
  `add disabled=${unnamed.addOff}, hint="${unnamed.hint}"`);

// define two behaviours through the dialog, exactly as a user would
await page.fill("#insp-name", "crate");
await page.locator("#insp-name").blur();
await page.waitForTimeout(200);
await page.click("#btn-bhv-library");
await page.waitForTimeout(150);
await page.click("#btn-bhv-new");
await page.fill("#bhv-name", "meltable");
await page.fill("#bhv-json", '{ "liquefiable": true }');
await page.click("#btn-bhv-save");
await page.waitForTimeout(150);
await page.click("#btn-bhv-new");
await page.fill("#bhv-name", "heavy");
await page.fill("#bhv-json", "{ not json }");
await page.click("#btn-bhv-save");
await page.waitForTimeout(150);
const badJson = await page.evaluate(async () => ({
  error: document.getElementById("bhv-error").textContent,
  names: (await import("/js/editor.js")).behaviorNames(),
}));
check("a body that is not JSON is refused, with the parser's own complaint",
  /not valid json/i.test(badJson.error) && !badJson.names.includes("heavy"),
  `"${badJson.error}"`);

await page.fill("#bhv-json", '{ "dynamic": true }');
await page.click("#btn-bhv-save");
await page.waitForTimeout(150);
await page.click("#btn-bhv-close");
await page.waitForTimeout(200);
const defined = await page.evaluate(async () =>
  (await import("/js/editor.js")).behaviorNames());
check("the dialog creates definitions", defined.join() === "meltable,heavy", `[${defined}]`);

const named = await readPanel();
check("naming the element unlocks Add, and the element count is shown",
  !named.addOff && /"crate" — 1 element\b/.test(named.count), `"${named.count}"`);

await page.selectOption("#bhv-add", "meltable");
await page.click("#btn-bhv-add");
await page.waitForTimeout(250);
const attached = await readPanel();
check("Add attaches the behaviour to the node name",
  attached.applied.join() === "meltable", JSON.stringify(attached.applied));
check("a liquefiable behaviour brings up the linked picker",
  attached.hasLinkedPicker, `pickers=${attached.hasLinkedPicker}`);

// a second element, named differently, is a linkable neighbour in the same room
await page.evaluate(async (ids) => (await import("/js/editor.js")).select([ids.b]), bhvIds);
await page.waitForTimeout(200);
await page.fill("#insp-name", "crateB");
await page.locator("#insp-name").blur();
await page.waitForTimeout(200);
await page.evaluate(async (ids) => (await import("/js/editor.js")).select([ids.a]), bhvIds);
await page.waitForTimeout(250);
const options = await page.evaluate(() =>
  [...document.querySelector("#bhv-applied [data-linked]").options].map((o) => o.value));
check("the linked picker offers the other named nodes in the room",
  options.join() === "crateB", `[${options}]`);

await page.selectOption("#bhv-applied [data-linked]", ["crateB"]);
await page.waitForTimeout(250);
const linkedNow = await page.evaluate(async () =>
  (await import("/js/editor.js")).entityBehaviors("crate")[0].linked);
check("picking a linked node stores it", linkedNow.join() === "crateB", `[${linkedNow}]`);

// a dynamic-only behaviour has nothing to link, so no picker
await page.selectOption("#bhv-add", "heavy");
await page.click("#btn-bhv-add");
await page.waitForTimeout(250);
const both = await page.evaluate(() => ({
  applied: [...document.querySelectorAll("#bhv-applied .item .n")].map((e) => e.textContent),
  pickers: document.querySelectorAll("#bhv-applied [data-linked]").length,
}));
check("only the liquefiable one gets a linked picker",
  both.applied.join() === "meltable,heavy" && both.pickers === 1,
  `${JSON.stringify(both.applied)}, ${both.pickers} picker(s)`);

// excludeSDF: the same shape as linked, but only dynamic nodes are worth
// offering - a node with no rigid body has no SDF in the sim to drop
const sdf = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  // crateB carries nothing yet, so it must not be offered
  const before = [...document.querySelectorAll("#bhv-applied [data-sdf] option")]
    .map((o) => o.value || o.textContent);
  ed.addEntityBehavior("crateB", "heavy");          // heavy is { dynamic: true }
  ed.select([]);
  ed.select([[...ed.state.placements.values()].find((p) => p.name === "crate").id]);
  const after = [...document.querySelectorAll("#bhv-applied [data-sdf] option")]
    .map((o) => o.value);
  return { before, after, dynamic: ed.isDynamicNode("crateB"),
    notDynamic: ed.isDynamicNode("nothing") };
});
check("excludeSDF offers only the dynamic nodes in the room",
  sdf.before.join() === "(nothing dynamic in this room)"
    && sdf.after.join() === "crateB" && sdf.dynamic && !sdf.notDynamic,
  `before ${JSON.stringify(sdf.before)} -> after ${JSON.stringify(sdf.after)}`);

await page.selectOption('#bhv-applied [data-sdf="meltable"]', ["crateB"]);
await page.waitForTimeout(250);
const sdfStored = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  return {
    entity: ed.entityBehaviors("crate").find((b) => b.name === "meltable").excludeSDF,
    written: mf.buildManifest().entities.crate.behaviors.find((b) => b.name === "meltable"),
    heavy: mf.buildManifest().entities.crateB.behaviors.find((b) => b.name === "heavy"),
  };
});
check("the chosen node is written as excludeSDF",
  sdfStored.entity.join() === "crateB" && sdfStored.written.excludeSDF.join() === "crateB",
  JSON.stringify(sdfStored.written));
check("an empty excludeSDF is left out, like linked",
  !("excludeSDF" in sdfStored.heavy) && !("linked" in sdfStored.heavy),
  JSON.stringify(sdfStored.heavy));

await page.click("#bhv-applied [data-remove='heavy']");
await page.waitForTimeout(250);
const removed = await page.evaluate(async () =>
  (await import("/js/editor.js")).entityBehaviors("crate").map((b) => b.name));
check("Remove detaches it again", removed.join() === "meltable", `[${removed}]`);

// direction: optional on every applied behaviour, no declaration needed
const dirFields = await page.evaluate(() => ({
  // one set per attached behaviour - "meltable" is attached, and takes one too
  fields: document.querySelectorAll("#bhv-applied [data-dir]").length,
  onMeltable: !!document.querySelector('#bhv-applied [data-dir="meltable"]'),
}));
check("every applied behaviour offers a direction, declared or not",
  dirFields.fields === 3 && dirFields.onMeltable, JSON.stringify(dirFields));

await page.fill('#bhv-applied [data-dir="meltable"][data-axis="0"]', "-1");
await page.locator('#bhv-applied [data-dir="meltable"][data-axis="0"]').blur();
await page.waitForTimeout(250);
const dirStored = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const b = ed.entityBehaviors("crate").find((x) => x.name === "meltable");
  return { direction: b?.direction, written: mf.buildManifest().entities.crate.behaviors };
});
check("the direction is stored on the applied behaviour, not the definition",
  dirStored.direction?.join() === "-1,0,0",
  JSON.stringify(dirStored.direction));
check("the manifest writes it beside the name",
  JSON.stringify(dirStored.written.find((b) => b.name === "meltable"))
    // X is negated: the manifest is glTF space, the editor's gizmo is not
    === JSON.stringify({
      name: "meltable", linked: ["crateB"], excludeSDF: ["crateB"], direction: [1, 0, 0],
    }),
  JSON.stringify(dirStored.written));

// the flip must be its own inverse, or undo would mirror the facing each time
const dirRoundTrip = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const before = ed.entityBehaviors("crate").find((b) => b.name === "meltable").direction;
  const snapshot = JSON.parse(JSON.stringify(ed.serialize()));
  await ed.deserialize(snapshot);
  const after = ed.entityBehaviors("crate").find((b) => b.name === "meltable")?.direction;
  // a restore clears the selection; put it back so the panel below still has
  // something to render
  const crate = [...ed.state.placements.values()].find((p) => p.name === "crate");
  if (crate) ed.select([crate.id]);
  return { before, snapshot: snapshot.entities.crate[0].direction, after };
});
check("a direction survives undo without mirroring itself",
  dirRoundTrip.before.join() === "-1,0,0" && dirRoundTrip.snapshot.join() === "1,0,0"
    && dirRoundTrip.after.join() === "-1,0,0",
  `editor ${dirRoundTrip.before} -> stored ${dirRoundTrip.snapshot} -> back ${dirRoundTrip.after}`);

// a definition that names a direction supplies the starting value
const seeded = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.setBehaviorDef("facing", { direction: [0, 0, 1] });
  ed.addEntityBehavior("crate", "facing");
  const shown = [...document.querySelectorAll('#bhv-applied [data-dir="facing"]')]
    .map((i) => i.value);
  // …but it is only a default: nothing is written until the entity says so
  const written = (await import("/js/manifest.js")).buildManifest()
    .entities.crate.behaviors.find((b) => b.name === "facing");
  return { shown, written };
});
check("a definition's direction seeds the fields without being written",
  seeded.shown.join() === "0,0,1" && !("direction" in seeded.written),
  `${JSON.stringify(seeded.shown)} -> ${JSON.stringify(seeded.written)}`);

// clearing every field drops it rather than writing a zero vector
await page.fill('#bhv-applied [data-dir="meltable"][data-axis="0"]', "");
await page.locator('#bhv-applied [data-dir="meltable"][data-axis="0"]').blur();
await page.waitForTimeout(250);
const dirCleared = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  return {
    entity: ed.entityBehaviors("crate").find((x) => x.name === "meltable"),
    written: mf.buildManifest().entities.crate.behaviors.find((b) => b.name === "meltable"),
  };
});
check("an all-zero direction is dropped, not written",
  !("direction" in dirCleared.entity) && !("direction" in dirCleared.written),
  JSON.stringify(dirCleared.written));

// a hand-edit that split `direction` into a sibling entry of its own
const merged = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
  await ed.deserialize({
    chunks: ["CH00_Storage"], activeChunk: "CH00_Storage", instances: [], markers: [],
    behaviors: { player_startpos: {} },
    entities: {
      player: { behaviors: [{ name: "player_startpos" }, { direction: [-1, 0, 0] }] },
    },
  });
  const out = ed.entityBehaviors("player");
  ed.clearAll(); ed.select([]);
  return out;
});
check("a nameless sibling entry is folded into the one above it",
  merged.length === 1 && merged[0].name === "player_startpos"
    // the manifest is glTF space, so the editor sees it mirrored
    && merged[0].direction?.join() === "1,0,0",
  JSON.stringify(merged));

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
  document.activeElement?.blur();
});
// Park the pointer back over the viewport. The wheel bindings live there, so
// leaving the cursor on the inspector quietly turns every later wheel test into
// a no-op - which is exactly what happened the first time this ran.
{
  const c = await page.evaluate(() =>
    window.__scene.getEngine().getRenderingCanvas().getBoundingClientRect().toJSON());
  await page.mouse.move(c.x + c.width / 2, c.y + c.height / 2, { steps: 3 });
}

// ---- 1d-sexdecies. X shows one element's axes -------------------------------
// World by default, because a drag moves along the world axes. Shift+X shows
// the element's own, which is what scaling acts on. The gizmo also dims the
// axes a drag cannot move along, so it answers "what will a drag do now".
const gizmo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const M = "Walls/ShortWall_Band2_Straight";
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.state.dragAxis = "xz";

  const a = await ed.placeAt(M, new V(0, 0, 0), { rotation: [0, 90, 0], silent: true });
  const b = await ed.placeAt(M, new V(20, 0, 0), { silent: true });
  ed.state.camera.position = new V(10, 14, -18);
  ed.state.camera.setTarget(new V(10, 0, 0));
  ed.state.camera.cameraDirection.setAll(0);
  ed.state.scene.render();

  const armDir = (axis) => {
    const arm = ed.state.scene.getTransformNodeByName(`AXES_${axis}`);
    if (!arm) return null;
    arm.computeWorldMatrix(true);
    const m = arm.getWorldMatrix();
    return m.getRow(2).toVector3().normalize().asArray().map((v) => Math.round(v * 100) / 100);
  };
  const lit = () => ["x", "y", "z"].map((k) =>
    ed.state.scene.getMaterialByName(`AXES_MAT_${k}`)?.alpha);
  const marks = (kind) => ["x", "y", "z"]
    .map((k) => (kind === "rot"
      ? ed.state.scene.getTransformNodeByName(`AXES_${k}_rot`)
      : ed.state.scene.getMeshByName(`AXES_${k}_${kind}`))?.isEnabled());

  const before = !!ed.axesTarget();
  ed.select([a.id]);
  const on = ed.toggleAxes(a.id);
  ed.state.scene.render();
  // a is turned 90 degrees about Y, so a local gizmo would point elsewhere
  const turned = { x: armDir("x"), y: armDir("y"), z: armDir("z"), alpha: lit() };

  // dragging on Y only: the gizmo has to follow the mode
  ed.state.dragAxis = "y";
  ed.state.scene.render();
  const yMode = { alpha: lit() };
  ed.state.dragAxis = "xz";
  ed.state.scene.render();

  // the curved arrow marks the ROTATION axis, which V does not touch
  ed.state.rotAxis = "y";
  ed.state.scene.render();
  const rotY = marks("rot");
  ed.state.rotAxis = "x";
  ed.state.dragAxis = "y";              // deliberately not the rotation axis
  ed.state.scene.render();
  const rotX = { ring: marks("rot"), alpha: lit() };
  ed.state.dragAxis = "xz";
  ed.state.rotAxis = "y";

  // and the scale axis puts a cube on its tip, whether or not it also drags
  ed.state.scaleAxis = "y";
  ed.state.scene.render();
  const scaleY = marks("scale");
  ed.state.scaleAxis = "all";
  ed.state.scene.render();
  const scaleAll = marks("scale");
  ed.state.scaleAxis = "all";

  const same = ed.toggleAxes(a.id);            // same element toggles off
  const gone = !ed.state.scene.getTransformNodeByName("AXES");
  ed.toggleAxes(a.id);
  const moved = ed.toggleAxes(b.id);           // a different one moves them
  const movedTo = ed.axesTarget();
  ed.state.scene.render();                     // the observer places it
  const pos = ed.state.scene.getTransformNodeByName("AXES").position.asArray();

  // deleting the element it is attached to must take the gizmo with it
  ed.select([b.id]);
  ed.removeSelected();
  ed.state.scene.render();
  const afterDelete = { target: ed.axesTarget(), node: !!ed.state.scene.getTransformNodeByName("AXES") };
  const leftover = ed.state.scene.meshes.filter((m) => m.name.startsWith("AXES_")).length;
  ed.clearAll(); ed.select([]);
  return { before, on, turned, yMode, rotY, rotX, scaleY, scaleAll, same, gone, moved,
    movedTo, pos, afterDelete, leftover };
});
check("X shows the axes, and X again hides them",
  gizmo.before === false && gizmo.on !== null && gizmo.same === null && gizmo.gone,
  `shown=${gizmo.on}, toggled off=${gizmo.same === null}`);
check("the arrows are world-aligned, whatever the element is turned to",
  // the element carries a 90-degree Y turn; the gizmo must ignore it, because
  // a drag moves along the world axes and not the element's own
  Math.abs(gizmo.turned.x[0] - 1) < 0.02 && Math.abs(gizmo.turned.y[1] - 1) < 0.02
    && Math.abs(gizmo.turned.z[2] - 1) < 0.02,
  `X->[${gizmo.turned.x}] Y->[${gizmo.turned.y}] Z->[${gizmo.turned.z}]`);
check("the live drag axes are bright and the locked ones dimmed",
  gizmo.turned.alpha[0] === 1 && gizmo.turned.alpha[2] === 1 && gizmo.turned.alpha[1] < 1
    && gizmo.yMode.alpha[1] === 1 && gizmo.yMode.alpha[0] < 1 && gizmo.yMode.alpha[2] < 1,
  `X/Z mode ${JSON.stringify(gizmo.turned.alpha)}, Y mode ${JSON.stringify(gizmo.yMode.alpha)}`);
check("a curved arrow encircles the rotation axis, which V does not touch",
  gizmo.rotY.join() === "false,true,false" && gizmo.rotX.ring.join() === "true,false,false"
    // rotation on X while dragging on Y: the ring is on the dimmed arm
    && gizmo.rotX.alpha[0] < 1,
  `rot Y ${JSON.stringify(gizmo.rotY)}, rot X ${JSON.stringify(gizmo.rotX.ring)} ` +
  `on a dimmed arm (alpha ${gizmo.rotX.alpha[0]})`);
check("a cube on the tip marks the scale axis, all three when it is 'all'",
  gizmo.scaleY.join() === "false,true,false" && gizmo.scaleAll.join() === "true,true,true",
  `Y ${JSON.stringify(gizmo.scaleY)}, all ${JSON.stringify(gizmo.scaleAll)}`);
check("X on a different element moves the axes there",
  gizmo.moved !== null && gizmo.pos[0] > 15,
  `now on ${gizmo.movedTo} at x=${gizmo.pos[0]}`);
check("deleting the element takes its axes with it, leaving no meshes behind",
  gizmo.afterDelete.target === null && !gizmo.afterDelete.node && gizmo.leftover === 0,
  `target=${gizmo.afterDelete.target}, ${gizmo.leftover} meshes left`);

// ---- 1d-septendecies. Shift+X shows the element's own axes ------------------
// Scaling is local, so on anything that has been turned a world gizmo cannot
// say which way X grows.
const localGizmo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0),
    { rotation: [0, 90, 0], silent: true });
  const dirs = () => {
    const out = {};
    for (const k of ["x", "y", "z"]) {
      const arm = ed.state.scene.getTransformNodeByName(`AXES_${k}`);
      arm.computeWorldMatrix(true);
      const d = BABYLON.Vector3.TransformNormal(new V(0, 0, 1), arm.getWorldMatrix()).normalize();
      out[k] = [Math.round(d.x), Math.round(d.y), Math.round(d.z)];
    }
    return out;
  };
  ed.toggleAxes(a.id, "world");
  const world = { dirs: dirs(), space: ed.axesSpace() };
  ed.toggleAxes(a.id, "local");
  const local = { dirs: dirs(), space: ed.axesSpace() };
  // a mirrored element's local +X really does point the other way
  a.node.scaling.x = -1;
  a.node.computeWorldMatrix(true);
  ed.toggleAxes(a.id, "world"); ed.toggleAxes(a.id, "local");
  const mirrored = dirs();
  // same element, same space toggles off; the other space re-aims instead
  const off = ed.toggleAxes(a.id, "local");
  ed.toggleAxes(a.id, "world");
  const swapped = ed.toggleAxes(a.id, "local");
  const stillOn = ed.axesTarget();
  ed.hideAxes(); ed.clearAll(); ed.select([]);
  return { world, local, mirrored, off, swapped, stillOn };
});
check("X stays world-aligned whatever the element is turned to",
  localGizmo.world.space === "world"
    && localGizmo.world.dirs.x.join() === "1,0,0"
    && localGizmo.world.dirs.y.join() === "0,1,0"
    && localGizmo.world.dirs.z.join() === "0,0,1",
  JSON.stringify(localGizmo.world.dirs));
check("Shift+X turns the arrows onto the element's own axes",
  localGizmo.local.space === "local"
    // a 90-degree Y turn sends local X to world -Z and local Z to world +X,
    // and leaves Y alone
    && localGizmo.local.dirs.x.join() === "0,0,-1"
    && localGizmo.local.dirs.y.join() === "0,1,0"
    && localGizmo.local.dirs.z.join() === "1,0,0",
  JSON.stringify(localGizmo.local.dirs));
check("a mirrored element's local X points the other way",
  localGizmo.mirrored.x.join() === "0,0,1" && localGizmo.mirrored.z.join() === "1,0,0",
  JSON.stringify(localGizmo.mirrored));
check("the same space toggles off, the other space re-aims",
  localGizmo.off === null && localGizmo.swapped !== null && localGizmo.stillOn !== null,
  `off=${localGizmo.off}, swapped=${localGizmo.swapped}`);

// ---- 1d-octodecies. no target hides, and the ghost counts as a target -------
const axesNoTarget = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.select([a.id]);
  ed.toggleAxes(a.id, "world");
  const shown = ed.axesTarget();
  ed.select([]);
  ed.state.scene.pointerX = -500; ed.state.scene.pointerY = -500;
  return { id: a.id, shown };
});
await page.evaluate(() => window.dispatchEvent(
  new KeyboardEvent("keydown", { key: "x", code: "KeyX", bubbles: true })));
await page.waitForTimeout(250);
const axesCleared = await page.evaluate(async () => ({
  target: (await import("/js/editor.js")).axesTarget(),
  status: document.getElementById("status-text").textContent,
}));
check("X with nothing selected or hovered hides the axes",
  axesNoTarget.shown === axesNoTarget.id && axesCleared.target === null
    && /hidden/.test(axesCleared.status),
  `was ${axesNoTarget.shown}, now ${axesCleared.target} — "${axesCleared.status}"`);

// the armed ghost has a rotation and a mirroring you set before dropping it,
// which is exactly when the axes are worth seeing
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.clearAll(); ed.select([]);
  // put the cursor back on screen: the ghost follows it, and one parked
  // off-canvas leaves it outside the frustum and rendering nothing
  const canvas = ed.state.engine.getRenderingCanvas();
  ed.state.scene.pointerX = canvas.width / 2;
  ed.state.scene.pointerY = canvas.height / 2;
  await i.armGhost("Walls/ShortWall_Band2_Straight");
});
await page.waitForTimeout(300);
await page.evaluate(() => window.dispatchEvent(
  new KeyboardEvent("keydown", { key: "x", code: "KeyX", bubbles: true })));
await page.waitForTimeout(350);
const ghostAxes = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return {
    target: ed.axesTarget(),
    ghostId: ed.GHOST_AXES,
    node: !!ed.state.scene.getTransformNodeByName("AXES"),
    status: document.getElementById("status-text").textContent,
  };
});
check("X shows the axes on the armed ghost",
  ghostAxes.target === ghostAxes.ghostId && ghostAxes.node
    && /Walls\/ShortWall/.test(ghostAxes.status),
  `target=${ghostAxes.target}, status="${ghostAxes.status}"`);

await page.evaluate(async () => (await import("/js/interact.js")).cancelGhost());
await page.waitForTimeout(350);
const afterGhostCancel = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const out = { target: ed.axesTarget(),
                leftover: ed.state.scene.meshes.filter((m) => m.name.startsWith("AXES_")).length };
  ed.clearAll(); ed.select([]);
  return out;
});
check("cancelling the ghost takes its axes with it",
  afterGhostCancel.target === null && afterGhostCancel.leftover === 0,
  `target=${afterGhostCancel.target}, ${afterGhostCancel.leftover} meshes left`);

// ---- 1d-novendecies. T cycles the move snap, and the gizmo shows it --------
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const p = await ed.placeAt("Walls/ShortWall_Band2_Straight",
    new BABYLON.Vector3(0, 0, 0), { silent: true });
  ed.state.camera.position = new BABYLON.Vector3(0, 8, -12);
  ed.state.camera.setTarget(new BABYLON.Vector3(0, 0, 0));
  ed.state.camera.cameraDirection.setAll(0);
  // the suite runs at a 4 m snap; borrow it and hand it back afterwards
  window.__snapWas = ed.state.snap.pos;
  ed.state.snap.pos = 1;
  document.getElementById("snap-pos").value = "1";
  ed.select([p.id]);
  ed.toggleAxes(p.id);
});
await page.waitForTimeout(500);
const snapLabel = await page.evaluate(() => {
  const el = document.querySelector("#viewport .axis-snap:not(.axis-rot)");
  const host = document.getElementById("viewport").getBoundingClientRect();
  const r = el?.getBoundingClientRect();
  return {
    text: el?.textContent,
    onScreen: !!r && r.width > 0 && r.left > host.left && r.right < host.right,
  };
});
check("the gizmo carries the move step at its origin",
  snapLabel.text === "1 m" && snapLabel.onScreen, JSON.stringify(snapLabel));

// The turn angle rides the curved arrow, the way the move step rides the
// origin. It has to sit on the *ring*, not on the origin: the ring's node used
// to be at the arm's root with the arc pushed out along Z, so a naive
// getAbsolutePosition() parked this chip on top of the move chip.
const rotLabel = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const el = document.querySelector("#viewport .axis-rot");
  const snap = document.querySelector("#viewport .axis-snap:not(.axis-rot)");
  const host = document.getElementById("viewport").getBoundingClientRect();
  const at = (n) => { const r = n.getBoundingClientRect(); return [r.left - host.left, r.top - host.top]; };
  const ring = ed.state.scene.getTransformNodeByName(`AXES_${ed.state.rotAxis}_rot`);
  ring?.computeWorldMatrix(true);
  return {
    text: el?.textContent, hidden: el?.hidden,
    apart: el && snap ? Math.hypot(...at(el).map((v, i) => v - at(snap)[i])) : 0,
    ringOffOrigin: ring ? ring.getAbsolutePosition().length() : 0,
  };
});
check("the turn angle rides the curved arrow",
  rotLabel.text === "90°" && rotLabel.hidden === false, JSON.stringify(rotLabel));
check("it sits on the ring, not on top of the move step chip",
  rotLabel.apart > 40 && rotLabel.ringOffOrigin > 0.5,
  `${rotLabel.apart.toFixed(0)} px apart, ring ${rotLabel.ringOffOrigin.toFixed(2)} m from the origin`);

// and it follows both the angle and the axis
await page.keyboard.press("Control+r");
await page.waitForTimeout(250);
await page.keyboard.press("Shift+R");
await page.waitForTimeout(400);
const rotFollowed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const ring = ed.state.scene.getTransformNodeByName(`AXES_${ed.state.rotAxis}_rot`);
  ring?.computeWorldMatrix(true);
  return {
    text: document.querySelector("#viewport .axis-rot")?.textContent,
    axis: ed.state.rotAxis,
    onAxisArm: ring ? ring.getAbsolutePosition().asArray().map((v) => Math.abs(v) > 0.5) : null,
  };
});
check("the angle chip follows Ctrl+R, and moves to the arm Shift+R picks",
  rotFollowed.text === "5°" && rotFollowed.axis === "x"
    && rotFollowed.onAxisArm.join() === "true,false,false",
  `${rotFollowed.text} on ${rotFollowed.axis}, at ${rotFollowed.onAxisArm}`);
// put the angle back for the tests below
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.snap.rot = 90; ed.state.rotAxis = "y";
  document.getElementById("snap-rot").value = "90";
  document.getElementById("rot-axis").value = "y";
});

// The scale step rides the cube on each lit tip - one chip per cube, because
// with the axis set to `all` the three cubes are the statement that all three
// grow, and a value on only one would read as "just this one".
const scaleChips = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.scaleAxis = "all";
  ed.state.snap.scale = 0.1;
  document.getElementById("scale-axis").value = "all";
  document.getElementById("snap-scale").value = "0.1";
  // pull the camera back so all three tips are on canvas - the chips are
  // clamped to it, which is the whole point of the check below
  ed.state.camera.position = new BABYLON.Vector3(14, 12, -18);
  ed.state.camera.setTarget(BABYLON.Vector3.Zero());
  ed.state.camera.cameraDirection.setAll(0);
  return true;
});
await page.waitForTimeout(600);
const sclShown = await page.evaluate(() => {
  const els = [...document.querySelectorAll("#viewport .axis-scale")];
  return { total: els.length, shown: els.filter((e) => !e.hidden).length,
           texts: [...new Set(els.filter((e) => !e.hidden).map((e) => e.textContent))] };
});
check("every lit scale cube carries the scale step",
  sclShown.total === 3 && sclShown.shown === 3 && sclShown.texts.join() === "0.1",
  JSON.stringify(sclShown));

await page.keyboard.press("Shift+F");          // all -> x
await page.waitForTimeout(400);
await page.keyboard.press("Control+f");        // 0.1 -> 0.25
await page.waitForTimeout(400);
const sclOne = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const els = [...document.querySelectorAll("#viewport .axis-scale")];
  const lit = els.filter((e) => !e.hidden);
  return { axis: ed.state.scaleAxis, step: ed.state.snap.scale,
           shown: lit.length, text: lit[0]?.textContent };
});
check("one axis lights one chip, and Ctrl+F changes what it says",
  sclOne.axis === "x" && sclOne.shown === 1 && sclOne.text === "0.25",
  JSON.stringify(sclOne));

// The chips hang off the arrow *tips*, which swing outside the viewport at
// close range - and #viewport does not clip, so one was measured sitting on a
// palette tile.
const offCanvas = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.scaleAxis = "all";
  const cam = ed.state.camera;
  cam.position = new BABYLON.Vector3(0, 0.2, -0.6);   // nose against the origin
  cam.setTarget(new BABYLON.Vector3(0, 0.2, 0));
  cam.cameraDirection.setAll(0);
  return true;
});
await page.waitForTimeout(600);
const strayChips = await page.evaluate(() => {
  const host = document.getElementById("viewport").getBoundingClientRect();
  const strays = [...document.querySelectorAll("#viewport .axis-snap")]
    .filter((e) => !e.hidden)
    .map((e) => e.getBoundingClientRect())
    .filter((r) => r.left < host.left || r.right > host.right
      || r.top < host.top || r.bottom > host.bottom);
  return strays.length;
});
check("no chip escapes the viewport onto the palette or the inspector",
  strayChips === 0, `${strayChips} stray chip(s)`);

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.scaleAxis = "all"; ed.state.snap.scale = 0.1;
  document.getElementById("scale-axis").value = "all";
  document.getElementById("snap-scale").value = "0.1";
});

const cycled = [];
for (let k = 0; k < 3; k++) {
  await page.keyboard.press("Shift+V");
  await page.waitForTimeout(150);
  cycled.push(await page.evaluate(async () => ({
    snap: (await import("/js/editor.js")).state.snap.pos,
    combo: document.getElementById("snap-pos").value,
    label: document.querySelector("#viewport .axis-snap:not(.axis-rot)")?.textContent,
  })));
}
await page.keyboard.press("Control+v");
await page.waitForTimeout(150);
const back = await page.evaluate(async () => (await import("/js/editor.js")).state.snap.pos);
check("Shift+V walks the Move options forward",
  cycled.map((c) => c.snap).join() === "2,4,0"
    && cycled.every((c) => c.combo === String(c.snap)),
  cycled.map((c) => c.snap).join(" -> "));
check("the readout follows, and says what snapping off means",
  cycled[0].label === "2 m" && cycled[2].label === "free",
  `${cycled.map((c) => c.label).join(" -> ")}`);
check("Ctrl+V walks them back", back === 4, `${back}`);

// hiding the gizmo takes the readout with it
const noLabel = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.hideAxes();
  const gone = !document.querySelector("#viewport .axis-snap");
  ed.state.snap.pos = window.__snapWas;
  document.getElementById("snap-pos").value = String(window.__snapWas);
  ed.clearAll(); ed.select([]);
  return gone;
});
check("hiding the axes removes the readout too", noLabel);

// with several selected, the nearest to the cursor wins
const nearest = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const M = "Walls/ShortWall_Band2_Straight";
  const left = await ed.placeAt(M, new V(-14, 0, 0), { silent: true });
  const right = await ed.placeAt(M, new V(14, 0, 0), { silent: true });
  ed.state.camera.position = new V(0, 12, -22);
  ed.state.camera.setTarget(new V(0, 0, 0));
  ed.state.camera.cameraDirection.setAll(0);
  ed.state.scene.render();
  ed.select([left.id, right.id]);

  const at = (id) => {
    const b = ed.screenBoundsOf(ed.state.placements.get(id).node);
    return { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2 };
  };
  // the scene's pointer is global state that later tests read, so put it back
  const wasPointer = { x: ed.state.scene.pointerX, y: ed.state.scene.pointerY };
  const aim = (p) => { ed.state.scene.pointerX = p.x; ed.state.scene.pointerY = p.y; };
  aim(at(left.id));
  const nearLeft = ed.nearestToCursor(ed.state.selection);
  aim(at(right.id));
  const nearRight = ed.nearestToCursor(ed.state.selection);
  aim(wasPointer);
  ed.clearAll(); ed.select([]);
  return { left: left.id, right: right.id, nearLeft, nearRight };
});
check("with several selected, the element nearest the cursor gets the axes",
  nearest.nearLeft === nearest.left && nearest.nearRight === nearest.right,
  `cursor left -> ${nearest.nearLeft}, cursor right -> ${nearest.nearRight}`);

// visible axes follow a single pick, so they never sit on the element you have
// just moved away from
const follows = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const M = "Walls/ShortWall_Band2_Straight";
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt(M, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(M, new V(9, 0, 0), { silent: true });
  const c = await ed.placeAt(M, new V(18, 0, 0), { silent: true });

  ed.select([a.id]);
  ed.toggleAxes(a.id);
  const started = ed.axesTarget();
  ed.select([b.id]);
  const followed = ed.axesTarget();
  ed.select([b.id, c.id]);                 // a multi-pick has no single home
  const onMulti = ed.axesTarget();
  ed.select([]);
  const onNone = ed.axesTarget();

  ed.hideAxes();
  ed.select([a.id]);
  const whenHidden = ed.axesTarget();      // hidden stays hidden
  ed.clearAll(); ed.select([]);
  return { started, followed, onMulti, onNone, whenHidden, a: a.id, b: b.id };
});
check("visible axes follow a single click to the new element",
  follows.started === follows.a && follows.followed === follows.b,
  `${follows.started} -> ${follows.followed}`);
check("a multi-selection or an empty one leaves them where they are",
  follows.onMulti === follows.b && follows.onNone === follows.b,
  `multi=${follows.onMulti}, none=${follows.onNone}`);
check("selecting does not conjure axes that were never shown",
  follows.whenHidden === null, `${follows.whenHidden}`);

// ---- 1d-octodecies. Shift/Ctrl on E, F and V drive the axis modes ----------
// One letter per action, its settings behind the modifiers: Shift picks the
// axis, Ctrl picks the step. Ctrl+R, Ctrl+F and Ctrl+V must all be claimed from
// the browser - reload, the find bar and paste respectively.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.rotAxis = "y";
  ed.state.scaleAxis = "all";
  ed.state.snap.rot = 90;
  ed.state.snap.scale = 0.1;
  document.getElementById("snap-rot").value = "90";
  document.getElementById("snap-scale").value = "0.1";
  window.__prevented = [];
  addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && "rfv".includes(e.key.toLowerCase())) {
      setTimeout(() => window.__prevented.push([e.key.toLowerCase(), e.defaultPrevented]), 0);
    }
  });
});
const rotCycle = [];
for (let k = 0; k < 3; k++) {
  await page.keyboard.press("Shift+R");
  await page.waitForTimeout(120);
  rotCycle.push(await page.evaluate(async () => (await import("/js/editor.js")).state.rotAxis));
}
const sclCycle = [];
for (let k = 0; k < 4; k++) {
  await page.keyboard.press("Shift+F");
  await page.waitForTimeout(120);
  sclCycle.push(await page.evaluate(async () => (await import("/js/editor.js")).state.scaleAxis));
}
check("Shift+R cycles the rotation axis",
  rotCycle.join() === "x,z,y", rotCycle.join(" -> "));
check("Shift+F cycles the scale axis",
  sclCycle.join() === "x,y,z,all", sclCycle.join(" -> "));

const rotStep = [];
for (let k = 0; k < 2; k++) {
  await page.keyboard.press("Control+r");
  await page.waitForTimeout(120);
  rotStep.push(await page.evaluate(async () => ({
    v: (await import("/js/editor.js")).state.snap.rot,
    combo: document.getElementById("snap-rot").value,
  })));
}
await page.keyboard.press("Control+Shift+r");
await page.waitForTimeout(120);
const rotStepBack = await page.evaluate(async () => (await import("/js/editor.js")).state.snap.rot);
check("Ctrl+R cycles the rotation angle, Ctrl+Shift+R goes back",
  rotStep.map((r) => r.v).join() === "5,15" && rotStep.every((r) => r.combo === String(r.v))
    && rotStepBack === 5,
  `90 -> ${rotStep.map((r) => r.v).join(" -> ")} -> back ${rotStepBack}`);

const sclStep = [];
for (let k = 0; k < 2; k++) {
  await page.keyboard.press("Control+f");
  await page.waitForTimeout(120);
  sclStep.push(await page.evaluate(async () => ({
    v: (await import("/js/editor.js")).state.snap.scale,
    combo: document.getElementById("snap-scale").value,
  })));
}
check("Ctrl+F cycles the scale step",
  sclStep.map((r) => r.v).join() === "0.25,0.05"
    && sclStep.every((r) => r.combo === String(r.v)),
  `0.1 -> ${sclStep.map((r) => r.v).join(" -> ")}`);

// Ctrl+V and Ctrl+R are the ones that most need claiming - unclaimed it pastes - so press
// it here, where the listener can actually see it. It walks the move step, so
// put that back: numpad +/- steps the build plane by it, further down.
const snapWas = await page.evaluate(async () => (await import("/js/editor.js")).state.snap.pos);
await page.keyboard.press("Control+v");
await page.waitForTimeout(120);
await page.evaluate(async (v) => {
  const ed = await import("/js/editor.js");
  ed.state.snap.pos = v;
  document.getElementById("snap-pos").value = String(v);
}, snapWas);

const claimed = await page.evaluate(() => window.__prevented);
check("Ctrl+R, Ctrl+F and Ctrl+V are all claimed from the browser",
  claimed.length > 0 && claimed.every(([, p]) => p === true)
    && new Set(claimed.map(([k]) => k)).size === 3,
  JSON.stringify(claimed));

// the numpad no longer cycles anything: one way to reach a setting, not two
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.rotAxis = "y"; ed.state.scaleAxis = "all";
  document.getElementById("rot-axis").value = "y";
  document.getElementById("scale-axis").value = "all";
});
for (const code of ["NumpadDivide", "NumpadMultiply"]) {
  await page.evaluate((c) => window.dispatchEvent(
    new KeyboardEvent("keydown", { key: c === "NumpadDivide" ? "/" : "*", code: c, bubbles: true })), code);
  await page.waitForTimeout(100);
}
const numpadGone = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { rot: ed.state.rotAxis, scl: ed.state.scaleAxis };
});
check("numpad / and * no longer cycle the axes",
  numpadGone.rot === "y" && numpadGone.scl === "all", JSON.stringify(numpadGone));

// restore for the tests that follow
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.snap.rot = 90; ed.state.snap.scale = 0.1;
  document.getElementById("snap-rot").value = "90";
  document.getElementById("snap-scale").value = "0.1";
});

// ---- 1d-duovicies. the manifest speaks glTF space -------------------------
// The editor is a left-handed Babylon scene; the exporter mirrors X on the way
// out (measured: an element at [7,3,5] lands in the .glb at [-7,3,5]). The
// runtime reads this manifest as glTF space and negates X in a dozen places -
// colliders, portal openings, room tests, the player's facing - so everything
// it consumes has to be written mirrored, or the ship fights its own geometry.
const handed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mk = await import("/js/markers.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const M = "Walls/ShortWall_Band2_Straight";
  const a = await ed.placeAt(M, new V(7, 0, 5), { silent: true });
  await ed.placeAt(M, new V(11, 0, 5), { silent: true });
  mk.addDoor(new V(9, 0, 5), { silent: true, leaves: [a.id] });

  // the chunk box spans every element in it, not just the first
  let lo = Infinity, hi = -Infinity;
  for (const p of ed.state.placements.values()) {
    const b = ed.worldBounds(p.node);
    if (!b) continue;
    lo = Math.min(lo, b.min.x);
    hi = Math.max(hi, b.max.x);
  }
  const man = (await import("/js/manifest.js")).buildManifest();
  const out = {
    editorMinX: Math.round(lo * 1e4) / 1e4,
    editorMaxX: Math.round(hi * 1e4) / 1e4,
    aabb: man.chunks.find((c) => c.aabb)?.aabb,
    doorX: man.doors[0]?.position[0],
    portalCentreX: man.portals[0]?.centre[0],
    // the tool's own reload data stays in editor space
    instanceX: man.instances.find((x) => x.id === a.id).position[0],
    markerX: man.markers[0]?.position[0],
  };
  ed.clearAll(); ed.select([]);
  return out;
});
check("chunk boxes are mirrored, and min/max swap with them",
  Math.abs(handed.aabb.min[0] + handed.editorMaxX) < 1e-3
    && Math.abs(handed.aabb.max[0] + handed.editorMinX) < 1e-3
    && handed.aabb.min[0] < handed.aabb.max[0],
  `editor X ${handed.editorMinX}..${handed.editorMaxX} -> glTF ${handed.aabb.min[0]}..${handed.aabb.max[0]}`);
check("doors and portals are mirrored too",
  handed.doorX === -9 && handed.portalCentreX === -9,
  `door ${handed.doorX}, portal ${handed.portalCentreX}`);
check("the tool's own reload data stays in editor space",
  handed.instanceX === 7 && handed.markerX === 9,
  `instance ${handed.instanceX}, marker ${handed.markerX}`);

// ---- 1d-quinquies. the view is saved with the layout -----------------------
// In the manifest, not in serialize(): that feeds the undo stack, and undoing
// an edit must not also throw the view somewhere else.
const view = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  ed.state.camera.position = new V(12, 7, -33);
  ed.state.camera.rotation.set(0.21, -1.05, 0);
  const saved = mf.buildManifest().view;

  // undo must not touch it
  ed.pushUndo();
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  await ed.undo();
  const afterUndo = ed.state.camera.position.asArray();

  // move away, then restore
  ed.state.camera.position = new V(-80, 40, 80);
  ed.state.camera.rotation.set(-0.4, 2.2, 0);
  const applied = ed.applyView(saved);
  const back = {
    pos: ed.state.camera.position.asArray(),
    rot: [ed.state.camera.rotation.x, ed.state.camera.rotation.y, ed.state.camera.rotation.z],
  };
  const ignoredOld = ed.applyView(undefined);       // manifests from before this
  const inUndoState = "view" in (JSON.parse(JSON.stringify(ed.serialize())));
  ed.clearAll(); ed.select([]);
  return { saved, afterUndo, applied, back, ignoredOld, inUndoState };
});
check("the manifest carries the camera position and rotation",
  !!view.saved?.position && !!view.saved?.rotation
    && Math.abs(view.saved.position[0] - 12) < 0.01,
  `position [${view.saved?.position}], rotation [${view.saved?.rotation}]`);
check("loading restores exactly where you were standing",
  view.applied
    && Math.abs(view.back.pos[0] - 12) < 0.01 && Math.abs(view.back.pos[2] + 33) < 0.01
    && Math.abs(view.back.rot[1] + 1.05) < 0.01,
  `[${view.back.pos.map((v) => v.toFixed(1))}] rot ${view.back.rot[1].toFixed(2)}`);
check("undo does not move the camera",
  Math.abs(view.afterUndo[0] - 12) < 0.01 && !view.inUndoState,
  `camera at [${view.afterUndo.map((v) => v.toFixed(1))}], view in undo snapshot=${view.inUndoState}`);
check("a manifest saved before this simply has no view to apply",
  view.ignoredOld === false);

// ---- 1d-duodecies. the lighting is saved with the layout -------------------
// Exposure and IBL strength are properties of the ship, not of this browser, so
// they ride in the manifest and outrank the sliders' remembered values.
const envRound = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  // `environment` carries the RUNTIME pair - the one the demos read - so the
  // values have to be set with that mode active
  ed.setRuntimeLighting(true);
  ed.setEnvIntensity(2.4);
  ed.setExposure(0.82);
  const saved = mf.buildManifest().environment;

  // a save/load round trip, served from memory so no export folder is touched
  const realFetch = window.fetch;
  let stored = null;
  window.fetch = (url, opts) => {
    if (String(url).includes("/api/layout")) {
      if (opts?.method === "POST") {
        stored = opts.body;
        return Promise.resolve(new Response('{"ok":true,"bytes":0}',
          { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      return Promise.resolve(new Response(stored,
        { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
  let loaded, sliders;
  try {
    await mf.saveLayout();
    ed.setEnvIntensity(0.2);          // wander off before loading it back
    ed.setExposure(1.9);
    loaded = await mf.loadLayout();
    sliders = {
      env: document.getElementById("env-intensity").value,
      exposure: document.getElementById("exposure").value,
      envLabel: document.getElementById("env-intensity-val").textContent,
      exposureLabel: document.getElementById("exposure-val").textContent,
      envStored: localStorage.getItem("envIntensity"),
      exposureStored: localStorage.getItem("exposure"),
    };
  } finally { window.fetch = realFetch; }

  const scene = {
    strength: ed.state.scene.environmentIntensity,
    exposure: ed.state.scene.imageProcessingConfiguration.exposure,
  };
  const inUndoState = "lightSets" in JSON.parse(JSON.stringify(ed.serialize()));
  const ignoredOld = ed.applyEnvironment(undefined);

  ed.setEnvIntensity(ed.ENV_INTENSITY_DEFAULT);
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  ed.setRuntimeLighting(false);
  ed.setEnvIntensity(ed.ENV_INTENSITY_DEFAULT);
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  ed.clearAll(); ed.select([]);
  return { saved, loaded: loaded?.environment, sliders, scene, inUndoState, ignoredOld };
});
check("the manifest carries an environment section",
  envRound.saved?.strength === 2.4
    && envRound.saved.toneMapping === "Khronos PBR Neutral"
    // the exposure is the linear multiplier, exactly as the slider shows it:
    // the demos apply it as-is, so there is only one number to know
    && envRound.saved.exposure === 0.82
    && Object.keys(envRound.saved).sort().join() === "exposure,strength,toneMapping",
  JSON.stringify(envRound.saved));
check("loading puts the lighting back on the scene",
  Math.abs(envRound.scene.strength - 2.4) < 1e-6
    && Math.abs(envRound.scene.exposure - 0.82) < 1e-3,
  `strength=${envRound.scene.strength}, exposure=${envRound.scene.exposure}`);
check("the sliders follow the loaded lighting",
  parseFloat(envRound.sliders.env) === 2.4
    && parseFloat(envRound.sliders.envStored) === 2.4
    && Math.abs(parseFloat(envRound.sliders.exposureStored) - 0.82) < 1e-3
    // 0.82 is deliberately off the slider's 0.05 step: the readout and the
    // stored value stay exact, only the thumb rounds
    && envRound.sliders.exposureLabel === "0.82"
    && Math.abs(parseFloat(envRound.sliders.exposure) - 0.82) <= 0.05,
  JSON.stringify(envRound.sliders));
check("the environment now rides the undo snapshot",
  envRound.inUndoState === true);
check("a manifest saved before this simply has no environment to apply",
  envRound.ignoredOld === false);

// ---- 1d-vicies. the lighting is undoable, one entry per gesture -------------
// It used to be deliberately off the stack ("not an edit to the ship"). It is
// an authored value the manifest carries and the runtime reads, so getting it
// wrong is as much an edit to take back as moving a wall.
const lightUndo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.setEnvIntensity(1.5); ed.setExposure(0.55);
  document.getElementById("env-intensity").value = "1.5";
  return { start: ed.state.envIntensity, depth: ed.historyDepth().undo };
});
// one gesture: press, then several input events as the thumb travels
await page.evaluate(() => {
  const el = document.getElementById("env-intensity");
  el.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
  for (const v of ["2.0", "2.5", "3.0"]) {
    el.value = v;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
});
await page.waitForTimeout(250);
const afterDrag = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { env: ed.state.envIntensity, depth: ed.historyDepth().undo };
});
check("dragging the Env slider costs exactly one undo entry",
  Math.abs(afterDrag.env - 3.0) < 1e-6 && afterDrag.depth === lightUndo.depth + 1,
  `${lightUndo.start} -> ${afterDrag.env}, stack ${lightUndo.depth} -> ${afterDrag.depth}`);

await page.evaluate(async () => (await import("/js/editor.js")).undo());
await page.waitForTimeout(250);
const undone = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return {
    env: ed.state.envIntensity,
    scene: ed.state.scene.environmentIntensity,
    slider: parseFloat(document.getElementById("env-intensity").value),
    readout: document.getElementById("env-intensity-val").textContent,
  };
});
check("undo puts the Env back, on the scene and on the slider",
  Math.abs(undone.env - 1.5) < 1e-6 && Math.abs(undone.scene - 1.5) < 1e-6
    && Math.abs(undone.slider - 1.5) < 1e-6 && undone.readout === "1.5",
  JSON.stringify(undone));

await page.evaluate(async () => (await import("/js/editor.js")).redo());
await page.waitForTimeout(250);
const redone = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.envIntensity);
check("redo brings it back", Math.abs(redone - 3.0) < 1e-6, `${redone}`);

// the *inactive* light set must travel too, or switching Runtime light after an
// undo would surface a value that was never restored
const bothSets = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.setRuntimeLighting(false);
  ed.setEnvIntensity(1.2);
  ed.pushUndo();
  ed.setRuntimeLighting(true);
  ed.setEnvIntensity(3.9);                 // edits the runtime set
  ed.setRuntimeLighting(false);
  const before = { ...ed.state.lightSets.runtime };
  await ed.undo();
  const after = { ...ed.state.lightSets.runtime };
  ed.setRuntimeLighting(false);
  ed.setEnvIntensity(1.5); ed.setExposure(0.55);
  ed.clearAll(); ed.select([]);
  return { before: before.strength, after: after.strength };
});
check("undo restores the light set the sliders are not editing",
  Math.abs(bothSets.before - 3.9) < 1e-6 && Math.abs(bothSets.after - 1.5) < 1e-6,
  `runtime strength ${bothSets.before} -> ${bothSets.after}`);

// ---- 1d-unvicies. Load warns before discarding unsaved work -----------------
// Load throws away the whole scene and sits one button from Save. Nothing else
// in the tool destroys unsaved work in a single click.
const guard = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  window.__confirms = [];
  window.__realConfirm = window.confirm;
  window.confirm = (msg) => { window.__confirms.push(msg); return window.__answer; };
  return true;
});

// dirty: prompt, and answering no leaves the scene alone
const dirty = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  window.__answer = false;
  window.__confirms.length = 0;
  return ed.state.placements.size;
});
await page.click("#btn-load");
await page.waitForTimeout(700);
const refused = await page.evaluate(async () => ({
  prompts: window.__confirms.length,
  msg: window.__confirms[0] || "",
  placements: (await import("/js/editor.js")).state.placements.size,
  status: document.getElementById("status-text").textContent,
}));
check("Load with unsaved changes asks first",
  refused.prompts === 1 && /unsaved/i.test(refused.msg), `"${refused.msg}"`);
check("answering no keeps the scene and says so",
  refused.placements === dirty && /cancelled/i.test(refused.status),
  `${refused.placements} placements, "${refused.status}"`);

// saving clears the warning: the changes are no longer unsaved
await page.evaluate(async () => {
  const realFetch = window.fetch;
  window.fetch = (url, opts) => (String(url).includes("/api/layout") && opts?.method === "POST"
    ? Promise.resolve(new Response('{"ok":true,"bytes":10,"path":"x"}',
      { status: 200, headers: { "Content-Type": "application/json" } }))
    : realFetch(url, opts));
  document.getElementById("btn-save").click();
  await new Promise((r) => setTimeout(r, 500));
  window.fetch = realFetch;
  window.__confirms.length = 0;
  window.__answer = true;               // let this one through
});
await page.click("#btn-load");
await page.waitForTimeout(900);
const savedThenLoad = await page.evaluate(() => window.__confirms.length);
check("saving clears the unsaved-changes warning", savedThenLoad === 0,
  `${savedThenLoad} prompts after a save`);

// and a load re-baselines too, so loading twice in a row never asks
await page.evaluate(() => { window.__confirms.length = 0; window.__answer = false; });
await page.click("#btn-load");
await page.waitForTimeout(900);
const loadTwice = await page.evaluate(() => window.__confirms.length);
check("a load re-baselines, so loading again does not ask", loadTwice === 0,
  `${loadTwice} prompts on the second load`);

// Closing or reloading the tab gets the same guard. The browser owns the
// wording, so all we control is whether the event is cancelled.
const unloadClean = await page.evaluate(() => {
  const e = new Event("beforeunload", { cancelable: true });
  dispatchEvent(e);
  return e.defaultPrevented;
});
check("closing a saved ship does not warn", unloadClean === false);

const unloadDirty = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(8, 0, 0), { silent: true });
  const e = new Event("beforeunload", { cancelable: true });
  dispatchEvent(e);
  return { prevented: e.defaultPrevented, returnValue: e.returnValue };
});
check("closing with unsaved changes warns",
  // A synthetic Event cannot model BeforeUnloadEvent exactly: its `returnValue`
  // is the legacy *boolean* alias for "not cancelled", so assigning "" reads
  // back as false rather than "". Either way it is falsy, which is precisely
  // what arms the browser's prompt - and `defaultPrevented` is the part that
  // actually matters.
  unloadDirty.prevented === true && !unloadDirty.returnValue,
  JSON.stringify(unloadDirty));

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  window.confirm = window.__realConfirm;
  ed.clearAll(); ed.select([]);
});

// ---- 1d-vicies. the runtime-light preview ---------------------------------
// The editor's four analytic lights are an authoring aid the game does not
// have, and they are the reason it looks nothing like the runtime - not the
// exposure conversion, which round-trips exactly.
const runtimeLight = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const rig = () => ed.state.scene.lights
    .filter((l) => ["hemi", "hemiUp", "key", "fill"].includes(l.name))
    .map((l) => l.intensity);

  const before = rig();
  ed.setRuntimeLighting(true);
  const off = rig();
  ed.setRuntimeLighting(false);
  const back = rig();

  // the numbers the demos read must be the ones the slider shows
  ed.setRuntimeLighting(true);
  ed.setExposure(0.55);
  ed.setEnvIntensity(1.7);
  const env = mf.buildManifest().environment;
  ed.setRuntimeLighting(false);
  return { before, off, back, env, mode: ed.state.runtimeLight };
});
check("runtime light silences the authoring rig, and only it",
  runtimeLight.before.length === 4 && runtimeLight.off.every((v) => v === 0)
    && runtimeLight.before.some((v) => v > 0),
  `${JSON.stringify(runtimeLight.before)} -> ${JSON.stringify(runtimeLight.off)}`);
check("switching back restores the authored intensities exactly",
  runtimeLight.back.join() === runtimeLight.before.join() && !runtimeLight.mode,
  JSON.stringify(runtimeLight.back));
check("the exposure is written as the slider shows it, no conversion",
  runtimeLight.env.exposure === 0.55 && runtimeLight.env.strength === 1.7,
  `exposure ${runtimeLight.env.exposure}, strength ${runtimeLight.env.strength}`);

// ---- 1d-unvicies. two light sets, one per mode ----------------------------
// The rig adds four lights the game does not have, so one pair of Env/Exposure
// values cannot serve both. The runtime pair goes to `environment`, which the
// demos read; the editor pair goes to `editorEnvironment`, which they must not.
const lightSets = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  ed.setRuntimeLighting(false);
  ed.setEnvIntensity(1.5); ed.setExposure(0.55);        // editor pair
  ed.setRuntimeLighting(true);
  ed.setEnvIntensity(3.2); ed.setExposure(1.1);         // runtime pair
  const inRuntime = { env: ed.state.envIntensity, exp: ed.state.exposure };
  ed.setRuntimeLighting(false);
  const backInEditor = { env: ed.state.envIntensity, exp: ed.state.exposure };

  const man = mf.buildManifest();
  const written = { environment: man.environment, editorEnvironment: man.editorEnvironment };

  // a round trip must keep both pairs apart
  ed.setEnvIntensity(0.1); ed.setExposure(0.2);
  ed.applyEnvironment(written.environment, written.editorEnvironment);
  const restored = {
    editor: { ...ed.state.lightSets.editor },
    runtime: { ...ed.state.lightSets.runtime },
    active: { env: ed.state.envIntensity, exp: ed.state.exposure },
  };

  // an old single-pair manifest gives its values to both
  ed.applyEnvironment({ strength: 2, exposure: 0.9 }, undefined);
  const legacy = {
    editor: { ...ed.state.lightSets.editor },
    runtime: { ...ed.state.lightSets.runtime },
  };

  ed.setRuntimeLighting(false);
  ed.setEnvIntensity(ed.ENV_INTENSITY_DEFAULT);
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  return { inRuntime, backInEditor, written, restored, legacy };
});
check("each mode keeps its own Env/Exposure",
  lightSets.inRuntime.env === 3.2 && lightSets.inRuntime.exp === 1.1
    && lightSets.backInEditor.env === 1.5 && lightSets.backInEditor.exp === 0.55,
  `runtime ${JSON.stringify(lightSets.inRuntime)}, editor ${JSON.stringify(lightSets.backInEditor)}`);
check("the demos get the runtime pair, the editor pair is filed separately",
  lightSets.written.environment.strength === 3.2
    && lightSets.written.environment.exposure === 1.1
    && lightSets.written.editorEnvironment.strength === 1.5
    && lightSets.written.editorEnvironment.exposure === 0.55,
  JSON.stringify(lightSets.written));
check("a round trip keeps the two pairs apart",
  lightSets.restored.runtime.strength === 3.2 && lightSets.restored.editor.strength === 1.5
    && lightSets.restored.active.env === 1.5,
  JSON.stringify(lightSets.restored));
check("a manifest with one pair gives it to both, rather than to neither",
  lightSets.legacy.editor.strength === 2 && lightSets.legacy.runtime.strength === 2
    && lightSets.legacy.editor.exposure === 0.9,
  JSON.stringify(lightSets.legacy));

// a negative exposure can only be the old stops format - the slider has never
// gone below 0.15 - so it is converted rather than clamped up to the floor
const oldStops = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.applyEnvironment({ strength: 1.7, exposure: -0.862 }, undefined);
  const migrated = { ...ed.state.lightSets.runtime };
  ed.applyEnvironment({ strength: 1.7, exposure: 0.55 }, undefined);
  const literal = { ...ed.state.lightSets.runtime };
  ed.setRuntimeLighting(false);
  ed.setEnvIntensity(ed.ENV_INTENSITY_DEFAULT);
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  return { migrated, literal };
});
check("a negative exposure is read as the old stops and converted",
  Math.abs(oldStops.migrated.exposure - 0.55) < 1e-3
    && oldStops.literal.exposure === 0.55,
  `-0.862 -> ${oldStops.migrated.exposure}, 0.55 -> ${oldStops.literal.exposure}`);

// the view transform comes from the manifest rather than being assumed
const tone = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const ipc = ed.state.scene.imageProcessingConfiguration;
  const read = () => ({ on: ipc.toneMappingEnabled, type: ipc.toneMappingType });
  ed.setToneMapping("ACES");
  const aces = read();
  ed.setToneMapping("none");
  const off = read();
  ed.setToneMapping("KHR_PBR_NEUTRAL");
  const neutral = read();
  ed.setToneMapping("Khronos PBR Neutral");
  return { aces, off, neutral, IPC: {
    aces: BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES,
    neutral: BABYLON.ImageProcessingConfiguration.TONEMAPPING_KHR_PBR_NEUTRAL,
  } };
});
check("the tone mapping named in the manifest is the one applied",
  tone.aces.type === tone.IPC.aces && tone.aces.on
    && tone.neutral.type === tone.IPC.neutral && !tone.off.on,
  `ACES=${tone.aces.type}, neutral=${tone.neutral.type}, none enabled=${tone.off.on}`);

// ---- 1d-septendecies. the inspector never shows a stale element ------------
// `hidden` only works through the UA rule `[hidden] { display: none }`, and any
// author `display` rule beats it: `.row { display: flex }` left the Name and
// Chunk rows on screen carrying the *previous* element's values.
const stale = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mk = await import("/js/markers.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const p = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.renamePlacement(p.id, "weaponHolder");
  mk.addDoor(new V(8, 0, 0), { silent: true });

  const read = () => {
    const row = document.getElementById("insp-name").parentElement;
    const chunkRow = document.getElementById("insp-chunk").parentElement;
    return {
      name: document.getElementById("insp-name").value,
      nameShown: row.getBoundingClientRect().height > 0,
      chunkShown: chunkRow.getBoundingClientRect().height > 0,
      behaviour: document.getElementById("behavior-fields").getBoundingClientRect().height > 0,
      id: document.getElementById("insp-id").textContent,
      module: document.getElementById("insp-module").textContent,
    };
  };
  ed.select([p.id]);
  const onPlacement = read();
  ed.select([[...ed.state.markers.keys()].find((k) => k.startsWith("Door_"))]);
  const onDoor = read();
  ed.select([p.id]);
  const backAgain = read();
  // the Id names *which* element the transform below belongs to
  const q = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(4, 0, 0), { silent: true });
  ed.select([p.id, q.id]);
  const onMulti = { ...read(), posX: document.getElementById("pos-x").value };
  ed.clearAll(); ed.select([]);
  return { onPlacement, onDoor, backAgain, onMulti, pId: p.id, qId: q.id };
});
check("a placement shows its name, chunk and behaviour",
  stale.onPlacement.name === "weaponHolder" && stale.onPlacement.nameShown
    && stale.onPlacement.chunkShown && stale.onPlacement.behaviour,
  JSON.stringify(stale.onPlacement));
check("selecting a door drops the previous element's name",
  stale.onDoor.name === "" && !stale.onDoor.nameShown
    && !stale.onDoor.chunkShown && !stale.onDoor.behaviour,
  JSON.stringify(stale.onDoor));
check("selecting a placement again brings the fields back",
  stale.backAgain.name === "weaponHolder" && stale.backAgain.nameShown
    && stale.backAgain.chunkShown,
  JSON.stringify(stale.backAgain));

// The id is read-only and always present - it is what doors, portals and
// behaviours reference, and the only way to tell a renamed element from the
// module it came from.
check("the inspector shows the selected element's id",
  stale.onPlacement.id === stale.pId && stale.onDoor.id.startsWith("Door_")
    && stale.backAgain.id === stale.pId,
  `placement "${stale.onPlacement.id}", door "${stale.onDoor.id}"`);
check("a marker shows its type rather than repeating its id",
  stale.onDoor.module === "door", `module="${stale.onDoor.module}"`);
check("in a multi-selection the id names the element the transform belongs to",
  stale.onMulti.id === stale.pId && stale.onMulti.module === "2 selected"
    && stale.onMulti.posX === "0",
  `id=${stale.onMulti.id}, module="${stale.onMulti.module}", pos-x=${stale.onMulti.posX}`);
check("the id is not editable",
  await page.evaluate(() => {
    const el = document.getElementById("insp-id");
    return el.tagName === "SPAN" && !el.isContentEditable;
  }));

// ---- 1d-quindecies. the editor is locked while a load runs ----------------
// A load rebuilds the scene one module at a time, so there is a long window in
// which the ship is half there. Anything that lands in it acts on a scene that
// does not exist yet.
const lockShape = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });

  const seen = [];
  let inside = null;
  await ed.whileBusy("loading ship…", async () => {
    inside = {
      busy: ed.isBusy(),
      label: ed.busyLabel(),
      overlay: !document.getElementById("busy").hidden,
      message: document.getElementById("busy-msg").textContent,
      inert: ["toolbar", "palette", "viewport", "inspector"]
        .every((id) => document.getElementById(id).inert),
    };
    // nested, the way the boot autoload sits inside the boot itself
    await ed.whileBusy("inner", async () => seen.push(ed.isBusy()));
    seen.push(ed.isBusy());          // must still be locked after the inner one
  });
  return {
    inside, seen,
    after: {
      busy: ed.isBusy(),
      overlay: !document.getElementById("busy").hidden,
      inert: ["toolbar", "palette", "viewport", "inspector"]
        .some((id) => document.getElementById(id).inert),
    },
  };
});
check("a load raises the overlay and makes the panels inert",
  lockShape.inside.busy && lockShape.inside.overlay && lockShape.inside.inert
    && lockShape.inside.message === "loading ship…",
  `overlay=${lockShape.inside.overlay}, inert=${lockShape.inside.inert}, ` +
  `"${lockShape.inside.message}"`);
check("nested loads do not unlock early",
  lockShape.seen.every(Boolean), JSON.stringify(lockShape.seen));
check("the lock is lifted when the load finishes",
  !lockShape.after.busy && !lockShape.after.overlay && !lockShape.after.inert,
  `busy=${lockShape.after.busy}, overlay=${lockShape.after.overlay}`);

const lockThrows = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  try { await ed.whileBusy("boom", () => { throw new Error("nope"); }); } catch { /* expected */ }
  return { busy: ed.isBusy(), overlay: !document.getElementById("busy").hidden };
});
check("a load that fails still unlocks",
  !lockThrows.busy && !lockThrows.overlay, JSON.stringify(lockThrows));

// and now for real: hold the lock open and try to break the scene through it
const held = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  let release;
  const gate = new Promise((r) => { release = r; });
  window.__release = release;
  window.__locked = ed.whileBusy("loading ship…", () => gate);
  await new Promise((r) => setTimeout(r, 50));
  const p = [...ed.state.placements.values()][0];
  return { id: p.id, pos: p.node.position.asArray(), count: ed.state.placements.size };
});
const lockCanvas = await page.evaluate(() =>
  window.__scene.getEngine().getRenderingCanvas().getBoundingClientRect().toJSON());
const lockMid = {
  x: lockCanvas.x + lockCanvas.width / 2,
  y: lockCanvas.y + lockCanvas.height / 2,
};
await page.mouse.move(lockMid.x, lockMid.y, { steps: 4 });
await page.mouse.down();
await page.mouse.move(lockMid.x + 120, lockMid.y + 60, { steps: 8 });
await page.mouse.up();
await page.keyboard.press("Delete");
await page.keyboard.press("r");
await page.mouse.wheel(0, -240);
await page.waitForTimeout(300);
const survived = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const p = [...ed.state.placements.values()][0];
  return {
    count: ed.state.placements.size,
    pos: p ? p.node.position.asArray() : null,
    rot: p ? ed.eulerOf(p.node) : null,
    hovered: i.hoveredId(),
    selection: ed.state.selection.length,
    marquee: i.isMarqueeing(),
  };
});
check("no click, drag, Del, turn or hover gets through the lock",
  survived.count === held.count
    && JSON.stringify(survived.pos) === JSON.stringify(held.pos)
    && survived.rot.every((v) => Math.abs(v) < 1e-6)
    && survived.hovered === null && survived.selection === 0 && !survived.marquee,
  `${survived.count} left at [${survived.pos}], rot [${survived.rot}], ` +
  `hover=${survived.hovered}, ${survived.selection} selected`);

await page.evaluate(async () => {
  window.__release();
  await window.__locked;
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
});
await page.waitForTimeout(200);
const unlocked = await page.evaluate(async () =>
  (await import("/js/editor.js")).isBusy());
check("releasing the load unlocks the editor again", unlocked === false);

// The editor forces every material two-sided, but the glTF exporter must hand
// the runtime what the kit authored. Prove the save/restore, using a fresh
// single-sided material (kit materials are all authored doubleSided already).
const cullRestore = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const probe = new BABYLON.StandardMaterial("CULL_PROBE", window.__scene);
  const authored = probe.backFaceCulling;                  // what it was born as
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const forcedOff = probe.backFaceCulling;                 // editor sweep set false
  const duringExport = await ed.withAuthoredMaterials(() => probe.backFaceCulling);
  const afterExport = probe.backFaceCulling;
  probe.dispose();
  return { authored, forcedOff, duringExport, afterExport };
});
check("authored culling restored for export",
  cullRestore.authored === true && cullRestore.forcedOff === false
    && cullRestore.duringExport === true && cullRestore.afterExport === false,
  `born=${cullRestore.authored} editor=${cullRestore.forcedOff} ` +
  `export=${cullRestore.duringExport} after=${cullRestore.afterExport}`);

// ---- 1f. unlit viewport mode -----------------------------------------------
const unlit = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const pbr = () => window.__scene.materials.filter((m) => "unlit" in m);
  ed.setUnlit(true);
  const on = pbr().every((m) => m.unlit);
  // a material created *after* the toggle must pick the mode up too
  const later = new BABYLON.PBRMaterial("UNLIT_PROBE", window.__scene);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const inherited = later.unlit;
  // and it must never reach the glTF export as KHR_materials_unlit
  const duringExport = await ed.withAuthoredMaterials(() => pbr().every((m) => !m.unlit));
  const restored = pbr().filter((m) => m.name !== "UNLIT_PROBE").every((m) => m.unlit);
  later.dispose();
  ed.setUnlit(false);
  const off = pbr().every((m) => !m.unlit);
  return { on, inherited, duringExport, restored, off, count: pbr().length };
});
check("Unlit switches every material to raw albedo",
  unlit.on && unlit.off, `${unlit.count} PBR materials toggled`);
check("materials created later inherit the unlit mode", unlit.inherited);
check("unlit never leaks into the export",
  unlit.duringExport && unlit.restored,
  `cleared during export=${unlit.duringExport}, restored after=${unlit.restored}`);

// ---- 1f-bis. emissive lift ---------------------------------------------------
// Raw albedo alone leaves the darkest kit modules almost black, so unlit mode
// adds a flat emissive lift. The light strips must be spared: their emissive
// colour *multiplies* an emissive texture, so overwriting it breaks them.
const lift = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const s = window.__scene;
  const L = ed.UNLIT_LIFT;

  const plain = new BABYLON.PBRMaterial("LIFT_PLAIN", s);
  plain.emissiveColor = new BABYLON.Color3(0.25, 0, 0);
  plain.emissiveIntensity = 2;
  ed.noteAuthoredEmissive(plain, true);

  const strip = new BABYLON.PBRMaterial("LIFT_STRIP", s);
  strip.emissiveTexture = new BABYLON.DynamicTexture("t", 4, s, false);
  strip.emissiveColor = new BABYLON.Color3(0, 0.7, 1);
  ed.noteAuthoredEmissive(strip, true);

  ed.setUnlit(true);
  const lifted = plain.emissiveColor.equalsFloats(L, L, L) && plain.emissiveIntensity === 1;
  const stripKept = strip.emissiveColor.equalsFloats(0, 0.7, 1);
  const noLiftInExport = await ed.withAuthoredMaterials(
    () => plain.emissiveColor.equalsFloats(0.25, 0, 0) && plain.emissiveIntensity === 2);
  const liftBack = plain.emissiveColor.equalsFloats(L, L, L);

  ed.setUnlit(false);
  const restored = plain.emissiveColor.equalsFloats(0.25, 0, 0) && plain.emissiveIntensity === 2;

  strip.emissiveTexture.dispose();
  plain.dispose(); strip.dispose();
  return { L, lifted, stripKept, noLiftInExport, liftBack, restored };
});
check("unlit lifts dark materials with a flat emissive",
  lift.lifted, `lift=${lift.L}`);
check("materials with an emissive texture are left alone", lift.stripKept);
check("the authored emissive comes back when unlit is turned off", lift.restored);
check("the emissive lift never leaks into the export",
  lift.noLiftInExport && lift.liftBack,
  `cleared during export=${lift.noLiftInExport}, restored after=${lift.liftBack}`);

// the kit sets its own emissive values *after* the material is constructed, so
// the snapshot has to be re-taken then or unlit would restore a stale black.
const kitEmissive = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const named = window.__scene.materials.find((m) => m.emissiveTexture && "unlit" in m);
  if (!named) return { skip: true };
  const before = named.emissiveColor.asArray();
  ed.setUnlit(true); ed.setUnlit(false);
  return { skip: false, name: named.name, before, after: named.emissiveColor.asArray() };
});
check("a kit light keeps its authored emissive across an unlit round-trip",
  kitEmissive.skip || kitEmissive.before.every((v, i) => Math.abs(v - kitEmissive.after[i]) < 1e-6),
  kitEmissive.skip ? "(no emissive-texture material loaded)"
    : `${kitEmissive.name} ${kitEmissive.before} -> ${kitEmissive.after}`);

// ---- 1e. a rotated ghost must not vanish -----------------------------------
// The blocks above clear the scene, so re-arm the ghost these tests need.
await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  await i.armGhost("Walls/ShortWall_Band2_Straight");
});
await page.waitForTimeout(900);
const spin = await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  const s = window.__scene;
  const seen = [];
  for (let step = 0; step < 4; step++) {
    s.render();
    const meshes = s.getTransformNodeByName("GHOST").getChildMeshes();
    const active = s.getActiveMeshes();
    seen.push({
      deg: step * 90,
      visible: meshes.every((m) => m.isVisible && m.isEnabled() && m.material.alpha > 0),
      inActive: meshes.filter((m) => {
        for (let k = 0; k < active.length; k++) if (active.data[k] === m) return true;
        return false;
      }).length,
      total: meshes.length,
    });
    i.rotateCurrent(1);           // +90 deg about Y
  }
  return seen;
});
check("ghost stays rendered through a full turn",
  spin.every((f) => f.visible && f.inActive === f.total),
  spin.map((f) => `${f.deg}:${f.inActive}/${f.total}`).join(" "));

// ---- 2. E turns, Shift+wheel scales, wheel zooms ---------------------------
// The wheel no longer rotates: it collided with zoom, which is what a wheel is
// for in a 3D view. Rotation lives on Q/E.
await page.mouse.wheel(0, -120);
await page.waitForTimeout(150);
let rot = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").rotationQuaternion.toEulerAngles().y * 180 / Math.PI);
check("the wheel does not rotate the ghost", Math.abs(rot) < 0.01, `${rot.toFixed(2)}°`);

await page.keyboard.press("r");
await page.waitForTimeout(200);
rot = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").rotationQuaternion.toEulerAngles().y * 180 / Math.PI);
check("E turns the ghost 90°", Math.abs(Math.abs(rot) - 90) < 0.01, `${rot.toFixed(2)}°`);

await page.keyboard.down("Shift");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Shift");
await page.waitForTimeout(150);
let scl = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").scaling.asArray());
check("Shift+wheel scales ghost +0.1", scl.every((v) => Math.abs(v - 1.1) < 1e-6), JSON.stringify(scl));

// Shift+wheel is an edit, so it must not also dolly
const camHeld = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
await page.keyboard.down("Shift");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Shift");
await page.waitForTimeout(400);
const camStill = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
check("Shift+wheel resizes without moving the camera",
  JSON.stringify(camHeld) === JSON.stringify(camStill),
  `${camHeld.map((v) => v.toFixed(1))} -> ${camStill.map((v) => v.toFixed(1))}`);
await page.keyboard.down("Shift");
await page.mouse.wheel(0, 120);            // undo the extra scale
await page.keyboard.up("Shift");
await page.waitForTimeout(150);

// Ctrl+wheel dollies the camera
const posBefore = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
await page.keyboard.down("Control");
await page.mouse.wheel(0, -240);
await page.keyboard.up("Control");
await page.waitForTimeout(600);
const posAfter = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
const moved = Math.hypot(posAfter[0] - posBefore[0], posAfter[1] - posBefore[1], posAfter[2] - posBefore[2]);
check("Ctrl+wheel dollies the camera", moved > 0.5, `moved ${moved.toFixed(2)} m`);

// ---- 2b. the dolly step never shrinks with distance ------------------------
const zoom = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const cam = window.__scene.activeCamera;
  const settle = () => new Promise((r) => setTimeout(r, 900));
  const notch = async () => {
    const before = cam.position.clone();
    ed.dollyCamera(1);
    await settle();
    return BABYLON.Vector3.Distance(before, cam.position);
  };
  const home = cam.position.clone();
  cam.position.set(0, 6, -200); cam.cameraDirection.setAll(0); await settle();
  const far = await notch();
  cam.position.set(0, 2, -6); cam.cameraDirection.setAll(0); await settle();
  const near = await notch();
  // a free camera has no pivot to collide with, so it keeps going at any range
  cam.position.set(0, 0.4, -0.4); cam.cameraDirection.setAll(0); await settle();
  const veryNear = await notch();
  cam.position.copyFrom(home); cam.cameraDirection.setAll(0);
  return { far, near, veryNear };
});
check("dolly step is the same far and near",
  // Relative, not absolute: what this guards against is a step that scales with
  // distance, which would make these differ by ~30x across 200 m vs 6 m. An
  // absolute 0.05 m window on a 2 m step was tight enough to fail on residual
  // camera inertia alone - measured 2.04 against 1.99, exactly on the boundary.
  Math.abs(zoom.far - zoom.near) / Math.max(zoom.far, zoom.near) < 0.05
    && zoom.far > 0.5,
  `${zoom.far.toFixed(2)} m at 200 m out vs ${zoom.near.toFixed(2)} m at 6 m`);
check("dolly still works when very close", zoom.veryNear > 0.5,
  `moved ${zoom.veryNear.toFixed(2)} m from 0.4 m out`);

// The wheel tests above dolly the camera around, so put it back somewhere it
// can actually see the origin before anything projects world points again.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.camera.position = new BABYLON.Vector3(-6, 9, -14);
  ed.state.camera.setTarget(new BABYLON.Vector3(0, 0, 0));
  ed.state.camera.cameraDirection.setAll(0);
  ed.state.camera.cameraRotation.set(0, 0);
});
await page.waitForTimeout(800);

// ---- 3. clicking places, and the brush stays armed --------------------------
pt = await screenOf([0, 0, 0]);
await page.mouse.move(pt.x, pt.y, { steps: 4 });
await page.waitForTimeout(200);
await page.mouse.down(); await page.mouse.up();
await page.waitForTimeout(700);

let placed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const e = [...ed.state.placements.values()][0];
  return {
    count: ed.state.placements.size,
    stillArmed: i.ghostActive(),
    brush: ed.state.brush,
    radius: window.__scene.activeCamera.radius,
    rotY: e ? ed.eulerOf(e.node)[1] : null,
    scale: e ? e.node.scaling.asArray() : null,
  };
});
check("click places the ghost", placed.count === 1,
  `${placed.count} placement(s), armed=${placed.stillArmed}, brush=${placed.brush}, r=${placed.radius?.toFixed(1)}`);
check("brush stays armed after placing", placed.stillArmed);
check("placement keeps ghost rotation",
  placed.rotY !== null && Math.abs(Math.abs(placed.rotY) - 90) < 0.01, `${placed.rotY}`);
check("placement keeps ghost scale",
  !!placed.scale && placed.scale.every((v) => Math.abs(v - 1.1) < 1e-6), JSON.stringify(placed.scale));

// ---- 4. grid elevation ------------------------------------------------------
await page.evaluate(() => import("/js/interact.js").then((i) => i.cancelGhost()));
await page.evaluate(() => import("/js/palette.js").then((p) => p.setBrush(null)));
await page.waitForTimeout(200);

await page.click("#render-canvas", { position: { x: 5, y: 5 } }).catch(() => {});
await page.keyboard.press("NumpadAdd");
await page.keyboard.press("NumpadAdd");
await page.waitForTimeout(150);
let elev = await page.evaluate(async () => ({
  y: (await import("/js/editor.js")).state.gridY,
  hud: document.getElementById("hud-elev").textContent,
}));
check("numpad + raises grid by snap", elev.y === 8, `${elev.y} m, HUD "${elev.hud}"`);
await page.keyboard.press("NumpadSubtract");
await page.waitForTimeout(120);
elev = await page.evaluate(async () => (await import("/js/editor.js")).state.gridY);
check("numpad - lowers grid", elev === 4, `${elev} m`);

// ---- 5. axis cycling --------------------------------------------------------
await page.evaluate(async () => {
  const s = (await import("/js/editor.js")).state;
  s.rotAxis = "y"; s.scaleAxis = "all";
});
await page.keyboard.press("Shift+R");
await page.waitForTimeout(120);
let axes = await page.evaluate(async () => {
  const s = (await import("/js/editor.js")).state;
  return { rot: s.rotAxis, scl: s.scaleAxis };
});
check("Shift+R cycles rotation axis", axes.rot === "x", `y -> ${axes.rot}`);
await page.keyboard.press("Shift+F");
await page.waitForTimeout(120);
axes = await page.evaluate(async () => {
  const s = (await import("/js/editor.js")).state;
  return { rot: s.rotAxis, scl: s.scaleAxis };
});
check("Shift+F cycles scale axis", axes.scl === "x", `all -> ${axes.scl}`);

// a toolbar control that keeps focus used to swallow every shortcut
const focusTrap = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.setGridElevation(4);
  document.getElementById("snap-pos").focus();
  return { focused: document.activeElement.id, before: ed.state.gridY };
});
await page.keyboard.press("NumpadSubtract");
await page.waitForTimeout(120);
const whileFocused = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.gridY);
const box2 = await page.locator("#render-canvas").boundingBox();
await page.mouse.move(box2.x + 40, box2.y + 40, { steps: 2 });
await page.mouse.down(); await page.mouse.up();
await page.waitForTimeout(150);
await page.keyboard.press("NumpadSubtract");
await page.waitForTimeout(150);
const afterClickAway = await page.evaluate(async () => ({
  y: (await import("/js/editor.js")).state.gridY,
  focused: document.activeElement.id || document.activeElement.tagName,
}));
check("clicking the viewport frees the keyboard from the toolbar",
  focusTrap.focused === "snap-pos" && afterClickAway.y < whileFocused,
  `focus was ${focusTrap.focused}, grid ${focusTrap.before} -> ${whileFocused} (trapped) -> ${afterClickAway.y}`);

// main-row minus works too, for keyboards without a numpad
const rowMinus = await page.evaluate(async () => (await import("/js/editor.js")).state.gridY);
await page.keyboard.press("Minus");
await page.waitForTimeout(150);
const afterRowMinus = await page.evaluate(async () => (await import("/js/editor.js")).state.gridY);
check("main-row minus also lowers the grid", afterRowMinus < rowMinus,
  `${rowMinus} -> ${afterRowMinus}`);
await page.evaluate(async () => {
  const s = (await import("/js/editor.js")).state;
  s.rotAxis = "y"; s.scaleAxis = "all";
  (await import("/js/editor.js")).setGridElevation(0);
});

// ---- 6. hover outline -------------------------------------------------------
const target = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const e = [...ed.state.placements.values()][0];
  e.node.position.set(0, 0, 0);
  const b = ed.worldBounds(e.node);
  return { id: e.id, centre: b.min.add(b.max).scale(0.5).asArray() };
});
pt = await screenOf(target.centre);
await page.mouse.move(pt.x + 1, pt.y + 1, { steps: 4 });
await page.waitForTimeout(400);
let hov = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const s = window.__scene;
  const e = [...ed.state.placements.values()][0];
  const edged = e.node.getChildMeshes().filter((m) => !!m.edgesRenderer);
  return {
    id: i.hoveredId(),
    edged: edged.length,
    total: e.node.getChildMeshes().length,
    effectLayers: (s.effectLayers || []).length,
    current: i.currentElement()?.kind,
    dbgPointer: [Math.round(s.pointerX), Math.round(s.pointerY)],
    dbgPick: ed.pickUnderCursor().kind,
    dbgGhost: i.ghostActive(),
  };
});
check("hover picks the element", hov.id === target.id,
  `${hov.id} | pointer ${JSON.stringify(hov.dbgPointer)} pick=${hov.dbgPick} ghost=${hov.dbgGhost}`);
check("outline uses per-instance edges", hov.edged === hov.total && hov.edged > 0,
  `${hov.edged}/${hov.total} instances edged`);
check("no HighlightLayer in the scene", hov.effectLayers === 0, `${hov.effectLayers} effect layer(s)`);
// selection now outranks hover, so clear it to test the hover fallback
const hoverIsCurrent = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.select([]);
  return i.currentElement()?.kind;
});
check("hovered element is 'current'", hoverIsCurrent === "hover", hoverIsCurrent || "none");

// ---- 7. the wheel leaves a merely-hovered element alone ---------------------
const rotBefore = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.select([]);
  return ed.eulerOf([...ed.state.placements.values()][0].node)[1];
});
const camBeforeHoverWheel = await page.evaluate(() =>
  window.__scene.activeCamera.position.asArray());
await page.mouse.wheel(0, 120);
await page.waitForTimeout(500);
const hoverWheel = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return {
    rot: ed.eulerOf([...ed.state.placements.values()][0].node)[1],
    hovered: i.hoveredId(),
    cam: window.__scene.activeCamera.position.asArray(),
  };
});
const camMovedOnHover = Math.hypot(
  ...hoverWheel.cam.map((v, k) => v - camBeforeHoverWheel[k]));
check("wheel over a hovered element dollies, does not rotate it",
  !!hoverWheel.hovered && Math.abs(hoverWheel.rot - rotBefore) < 1e-6 && camMovedOnHover > 0.5,
  `hovering ${hoverWheel.hovered}, rotation held at ${rotBefore.toFixed(1)}, camera moved ${camMovedOnHover.toFixed(2)} m`);

// ---- 2c. right button + wheel sets the fly speed ---------------------------
// Ctrl+WASD cannot do this: Ctrl+W closes the browser tab (reserved by Chrome,
// and a page cannot cancel it), and Ctrl+S / Ctrl+D are Save and Duplicate.
const speedStart = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.select([...ed.state.placements.keys()].slice(0, 1));
  return { speed: ed.state.moveSpeed, selection: ed.state.selection.length,
           cam: ed.state.camera.position.asArray() };
});
await page.mouse.down({ button: "right" });
for (let k = 0; k < 5; k++) { await page.mouse.wheel(0, -120); await page.waitForTimeout(50); }
await page.mouse.up({ button: "right" });
await page.waitForTimeout(300);
const speedUp = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { speed: ed.state.moveSpeed, selection: ed.state.selection.length,
           cam: ed.state.camera.position.asArray(),
         };
});
check("right button + wheel raises the fly speed",
  speedUp.speed > speedStart.speed * 1.5,
  `${speedStart.speed} -> ${speedUp.speed} m/s`);
check("changing speed does not dolly the camera",
  Math.hypot(...speedUp.cam.map((v, k) => v - speedStart.cam[k])) < 1e-6,
  `camera moved ${Math.hypot(...speedUp.cam.map((v, k) => v - speedStart.cam[k])).toFixed(4)} m`);
// the release is not a click: it must not fire the cancel gesture
check("changing speed does not cancel the selection",
  speedUp.selection === speedStart.selection,
  `${speedStart.selection} -> ${speedUp.selection} selected`);
// the HUD lost its speed row, so this control's only feedback is the status line
check("the status line reports the fly speed",
  new RegExp(`fly speed ${speedUp.speed} m/s`).test(
    await page.evaluate(() => document.getElementById("status-text").textContent)),
  await page.evaluate(() => document.getElementById("status-text").textContent));

await page.mouse.down({ button: "right" });
for (let k = 0; k < 40; k++) { await page.mouse.wheel(0, 120); await page.waitForTimeout(20); }
await page.mouse.up({ button: "right" });
await page.waitForTimeout(300);
const speedFloor = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.moveSpeed);
check("the fly speed is clamped above zero", speedFloor >= 1.5 && speedFloor < 5,
  `bottomed out at ${speedFloor} m/s`);
await page.evaluate(async () => { (await import("/js/editor.js")).state.moveSpeed = 42; });

// a plain right-click, with no wheel, must still cancel
await page.mouse.down({ button: "right" });
await page.mouse.up({ button: "right" });
await page.waitForTimeout(250);
const afterPlainRight = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.selection.length);
check("a plain right-click still cancels", afterPlainRight === 0,
  `${afterPlainRight} selected`);

// R still turns whatever is under the cursor - no camera binding to collide with
await page.evaluate(() => window.dispatchEvent(
  new KeyboardEvent("keydown", { key: "r", code: "KeyR", bubbles: true })));
await page.waitForTimeout(200);
const rotAfterHover = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return ed.eulerOf([...ed.state.placements.values()][0].node)[1];
});
check("R still rotates the hovered element",
  Math.abs(Math.abs(rotAfterHover - rotBefore) - 90) < 0.01,
  `${rotBefore.toFixed(1)} -> ${rotAfterHover.toFixed(1)}`);

const outlineSynced = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const e = [...ed.state.placements.values()][0];
  // edges rendering reads the instance world matrix every frame, so an edit
  // can never leave the outline behind
  const m = e.node.getChildMeshes()[0];
  return { ok: !!m.edgesRenderer && m.edgesRenderer._source === m };
});
check("the outline follows an edit", outlineSynced.ok, "");

// rotating moved the wall, so re-aim before grabbing it
const aim = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const e = [...ed.state.placements.values()][0];
  const b = ed.worldBounds(e.node);
  return b.min.add(b.max).scale(0.5).asArray();
});
pt = await screenOf(aim);
await page.mouse.move(pt.x, pt.y, { steps: 4 });
await page.waitForTimeout(300);

// ---- 8. click selects, drag moves ------------------------------------------
const beforeGrab = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.select([]);
  return [...ed.state.placements.values()][0].node.position.asArray();
});

await page.mouse.down(); await page.mouse.up();
await page.waitForTimeout(500);
const singleClick = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return { selection: ed.state.selection.length, dragging: i.isDragging() };
});
check("single click selects without moving",
  singleClick.selection === 1 && !singleClick.dragging,
  `${singleClick.selection} selected, dragging=${singleClick.dragging}`);

// drag it to a new cell
const dragTo = await screenOf([8, 0, 8]);
await page.mouse.down();
await page.mouse.move(dragTo.x, dragTo.y, { steps: 8 });
const midDrag = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return {
    dragging: i.isDragging(),
    pos: [...ed.state.placements.values()][0].node.position.asArray(),
    visible: [...ed.state.placements.values()][0].node.isEnabled(),
    camPos: ed.state.camera.position.asArray(),
    cursor: window.__scene.getEngine().getRenderingCanvas().style.cursor,
  };
});
await page.mouse.up();
await page.waitForTimeout(300);
const dropped = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const e = [...ed.state.placements.values()][0];
  return {
    dragging: i.isDragging(),
    pos: e.node.position.asArray(),
    count: ed.state.placements.size,
    visible: e.node.isEnabled(),
    cursor: window.__scene.getEngine().getRenderingCanvas().style.cursor,
  };
});
check("drag moves the element live", midDrag.dragging && midDrag.visible
  && JSON.stringify(midDrag.pos) !== JSON.stringify(beforeGrab),
  `${JSON.stringify(beforeGrab)} -> ${JSON.stringify(midDrag.pos)} while dragging`);
check("cursor hidden while dragging", midDrag.cursor === "none", `cursor="${midDrag.cursor}"`);
check("drop ends the drag without duplicating",
  !dropped.dragging && dropped.count === 1 && dropped.visible
  && JSON.stringify(dropped.pos) !== JSON.stringify(beforeGrab),
  `${dropped.count} placement at ${JSON.stringify(dropped.pos)}`);
check("cursor restored after the drop", dropped.cursor !== "none", `cursor="${dropped.cursor}"`);
check("dragging an element does not move the camera",
  JSON.stringify(midDrag.camPos) === JSON.stringify(
    await page.evaluate(async () => (await import("/js/editor.js")).state.camera.position.asArray())),
  "camera untouched by the drag");

// Esc mid-drag puts everything back
const home = dropped.pos;
await page.mouse.move(dragTo.x, dragTo.y, { steps: 3 });
await page.waitForTimeout(200);
const escFrom = await screenOf([0, 0, 0]);
await page.mouse.down();
await page.mouse.move(escFrom.x, escFrom.y, { steps: 8 });
await page.keyboard.press("Escape");
await page.mouse.up();
await page.waitForTimeout(300);
const restored = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const e = [...ed.state.placements.values()][0];
  return { dragging: i.isDragging(), pos: e.node.position.asArray(), visible: e.node.isEnabled() };
});
check("Esc cancels a drag and restores the position",
  !restored.dragging && restored.visible
  && JSON.stringify(restored.pos) === JSON.stringify(home),
  `${JSON.stringify(restored.pos)} vs ${JSON.stringify(home)}`);

// ---- 8b. a drag moves the whole selection ----------------------------------
const multiDrag = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  ed.state.snap.pos = 4;
  const cam = ed.state.camera;
  cam.position.set(0, 16, -24); cam.cameraDirection.setAll(0);
  cam.setTarget(new V(0, 0, 0));
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(8, 0, 4), { silent: true });
  const c = await ed.placeAt(moduleId, new V(-12, 0, 0), { silent: true });
  ed.select([a.id, b.id]);
  const bb = ed.worldBounds(a.node);
  return {
    ids: [a.id, b.id, c.id],
    before: [a, b, c].map((e) => e.node.position.asArray()),
    grabPoint: bb.min.add(bb.max).scale(0.5).asArray(),
  };
}, MODULE);

const from = await screenOf(multiDrag.grabPoint);
const to = await screenOf([multiDrag.grabPoint[0] + 8, multiDrag.grabPoint[1], multiDrag.grabPoint[2] + 8]);
await page.mouse.move(from.x, from.y, { steps: 4 });
await page.waitForTimeout(200);
await page.mouse.down();
await page.mouse.move(to.x, to.y, { steps: 10 });
await page.mouse.up();
await page.waitForTimeout(300);

const multiAfter = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  return ids.map((id) => ed.state.placements.get(id).node.position.asArray());
}, multiDrag.ids);
const d0 = multiAfter[0].map((v, i) => v - multiDrag.before[0][i]);
const d1 = multiAfter[1].map((v, i) => v - multiDrag.before[1][i]);
const d2 = multiAfter[2].map((v, i) => v - multiDrag.before[2][i]);
check("drag moves every selected element",
  Math.hypot(...d0) > 1 && JSON.stringify(d0) === JSON.stringify(d1),
  `both moved by ${JSON.stringify(d0)}`);
check("unselected elements stay put", Math.hypot(...d2) === 0, `moved ${JSON.stringify(d2)}`);
check("relative spacing is preserved",
  JSON.stringify(multiAfter[1].map((v, i) => v - multiAfter[0][i]))
  === JSON.stringify(multiDrag.before[1].map((v, i) => v - multiDrag.before[0][i])),
  "offset between the two is unchanged");
check("drag delta lands on the grid",
  d0.every((v) => Math.abs(v / 4 - Math.round(v / 4)) < 1e-6), `delta ${JSON.stringify(d0)}`);

// dragging an unselected element takes over the selection
const takeover = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const e = ed.state.placements.get(ids[2]);
  const bb = ed.worldBounds(e.node);
  return { centre: bb.min.add(bb.max).scale(0.5).asArray(), id: ids[2] };
}, multiDrag.ids);
const tFrom = await screenOf(takeover.centre);
const tTo = await screenOf([takeover.centre[0], takeover.centre[1], takeover.centre[2] + 8]);
await page.mouse.move(tFrom.x, tFrom.y, { steps: 4 });
await page.waitForTimeout(200);
await page.mouse.down();
await page.mouse.move(tTo.x, tTo.y, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(300);
const afterTakeover = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { selection: [...ed.state.selection] };
});
check("dragging an unselected element selects just it",
  afterTakeover.selection.length === 1 && afterTakeover.selection[0] === takeover.id,
  `selection is ${JSON.stringify(afterTakeover.selection)}`);

// ---- 10. no leaked helper meshes -------------------------------------------
const box = await page.locator("#render-canvas").boundingBox();
await page.mouse.move(box.x + box.width - 40, box.y + 30, { steps: 4 });
await page.evaluate(async () => (await import("/js/editor.js")).select([]));
await page.waitForTimeout(400);
const leaks = await page.evaluate(async () => ({
  hovered: (await import("/js/interact.js")).hoveredId(),
  ghost: window.__scene.meshes.filter((m) => m.name.startsWith("GHOST_")).length,
  hover: window.__scene.meshes.filter((m) => m.name.startsWith("HOVER_")).length,
  edged: window.__scene.meshes.filter((m) => !!m.edgesRenderer).length,
  geometries: window.__scene.geometries.length,
  cursor: window.__scene.getEngine().getRenderingCanvas().style.cursor,
}));
check("no leaked ghost/hover/outline meshes",
  leaks.ghost === 0 && leaks.hover === 0 && leaks.edged === 0 && !leaks.hovered,
  JSON.stringify(leaks));

// ---- 11. a tile lying exactly on the build plane must still be hoverable ----
const flat = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const cat = (await import("/js/kit.js")).getCatalogue();
  const floor = cat.categories.find((c) => c.name === "Platforms").modules
    .find((m) => /floor|platform/i.test(m.name)) || cat.categories[1].modules[0];
  ed.clearAll();
  const e = await ed.placeAt(floor.id, new BABYLON.Vector3(0, 0, 0), { silent: true });
  const b = ed.worldBounds(e.node);
  return { id: e.id, module: floor.id, centre: b.min.add(b.max).scale(0.5).asArray(), minY: b.min.y };
});
pt = await screenOf(flat.centre);
await page.mouse.move(pt.x, pt.y, { steps: 4 });
await page.waitForTimeout(400);
const flatHover = await page.evaluate(async () =>
  (await import("/js/interact.js")).hoveredId());
check("floor tile at y=0 beats the pick plane", flatHover === flat.id,
  `${flat.module} minY=${flat.minY.toFixed(3)} -> ${flatHover}`);

// ---- 12. a raised build plane must not block selection ----------------------
const raised = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.setGridElevation(20);                       // well above the tile
  return {
    pickables: window.__scene.meshes.filter((m) => m.isPickable && m.name.includes("PICK")).length,
    grid: !!window.__scene.getTransformNodeByName("GRID"),
  };
});
await page.mouse.move(pt.x + 3, pt.y + 3, { steps: 4 });
await page.waitForTimeout(400);
const throughGrid = await page.evaluate(async () =>
  (await import("/js/interact.js")).hoveredId());
check("raised grid does not block picking", throughGrid === flat.id,
  `grid at y=20, hovered=${throughGrid}, pick-plane meshes=${raised.pickables}`);

const groundStillWorks = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  // a plane above the camera while looking down has no intersection at all,
  // which is correct - test at an elevation the camera can actually see
  ed.setGridElevation(2);
  const off = ed.cursorOnGrid();
  const hit = ed.pickUnderCursor();
  ed.setGridElevation(0);
  return { kind: hit.kind, hasPoint: !!off, y: off ? +off.y.toFixed(3) : null };
});
check("build plane still solved analytically",
  groundStillWorks.hasPoint && Math.abs(groundStillWorks.y - 2) < 1e-3,
  `cursorOnGrid y=${groundStillWorks.y} (expected 2), pick kind=${groundStillWorks.kind}`);

// ---- 13. drop to plane uses the current elevation ---------------------------
const dropTo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const e = [...ed.state.placements.values()][0];
  ed.select([e.id]);
  ed.setGridElevation(5);
  document.getElementById("btn-ground").click();
  const b = ed.worldBounds(e.node);
  return { bottom: +b.min.y.toFixed(3), gridY: ed.state.gridY };
});
check("drop to plane uses the build plane", Math.abs(dropTo.bottom - dropTo.gridY) < 1e-3,
  `bottom=${dropTo.bottom} plane=${dropTo.gridY}`);

// ---- 14. flying the free camera --------------------------------------------
const nav = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const cam = ed.state.camera;
  const V = BABYLON.Vector3;
  ed.setGridElevation(0);

  const fly = async (code, mods, ms) => {
    cam.cameraDirection.setAll(0);
    const from = cam.position.clone();
    ed.noteKey(code, true, mods);
    await new Promise((r) => setTimeout(r, ms));
    ed.noteKey(code, false, {});
    await new Promise((r) => setTimeout(r, 400));
    const d = cam.position.subtract(from);
    cam.cameraDirection.setAll(0);
    return d;
  };

  cam.position.set(0, 8, -20);
  cam.rotation.set(0.3, 0.2, 0);
  const fwd = cam.getDirection(V.Forward()).normalize();
  const right = cam.getDirection(V.Right()).normalize();
  const home = cam.position.clone();

  const dW = await fly("KeyW", {}, 400); cam.position.copyFrom(home);
  const dD = await fly("KeyD", {}, 400); cam.position.copyFrom(home);
  const dA = await fly("KeyA", {}, 400); cam.position.copyFrom(home);
  const dSpace = await fly("Space", {}, 400); cam.position.copyFrom(home);
  const dC = await fly("KeyC", {}, 400); cam.position.copyFrom(home);
  const slow = await fly("KeyW", {}, 500); cam.position.copyFrom(home);
  const fast = await fly("KeyW", { shiftKey: true }, 500); cam.position.copyFrom(home);

  return {
    forward: dW.length(),
    wDotFwd: V.Dot(dW.normalize(), fwd),
    dDotRight: V.Dot(dD.normalize(), right),
    aDotRight: V.Dot(dA.normalize(), right),
    spaceUp: dSpace.y, spaceLateral: Math.hypot(dSpace.x, dSpace.z),
    cDown: dC.y,
    boost: fast.length() / Math.max(slow.length(), 1e-6),
  };
});
check("W flies along the view direction", nav.forward > 1 && nav.wDotFwd > 0.99,
  `moved ${nav.forward.toFixed(2)} m, W·forward=${nav.wDotFwd.toFixed(3)}`);
check("translation speed is ~42 m/s", nav.forward > 12 && nav.forward < 25,
  `${nav.forward.toFixed(1)} m in 0.4 s`);
check("D goes right, A goes left", nav.dDotRight > 0.99 && nav.aDotRight < -0.99,
  `D·right=${nav.dDotRight.toFixed(3)}, A·right=${nav.aDotRight.toFixed(3)}`);
check("Space rises, C descends",
  nav.spaceUp > 1 && nav.cDown < -1 && nav.spaceLateral < 1e-6,
  `Space +${nav.spaceUp.toFixed(2)} m, C ${nav.cDown.toFixed(2)} m, lateral ${nav.spaceLateral.toFixed(4)}`);
check("Shift doubles the move speed", nav.boost > 1.8 && nav.boost < 2.2,
  `${nav.boost.toFixed(2)}x`);

// Alt must not reach the browser: on Windows it focuses the menu bar, which
// then swallows the following keystrokes and kills WASD entirely.
await page.evaluate(() => {
  window.__altOut = {};
  window.__altSpy = (e) => { if (e.key === "Alt") window.__altOut[e.type] = e.defaultPrevented; };
  window.addEventListener("keydown", window.__altSpy);
  window.addEventListener("keyup", window.__altSpy);
});
await page.keyboard.down("Alt");
await page.keyboard.up("Alt");
await page.waitForTimeout(150);
const altFlags = await page.evaluate(() => {
  window.removeEventListener("keydown", window.__altSpy);
  window.removeEventListener("keyup", window.__altSpy);
  return window.__altOut;
});
check("Alt is claimed from the browser menu",
  altFlags.keydown === true && altFlags.keyup === true,
  `keydown prevented=${altFlags.keydown}, keyup prevented=${altFlags.keyup}`);

const altThenMove = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const cam = ed.state.camera;
  cam.position.set(0, 8, -20);
  cam.cameraDirection.setAll(0);
  await new Promise((r) => setTimeout(r, 200));
  window.__altFrom = cam.position.clone();
  return true;
});
await page.keyboard.down("w");
await page.waitForTimeout(400);
await page.keyboard.up("w");
await page.waitForTimeout(400);
const altMoved = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return BABYLON.Vector3.Distance(window.__altFrom, ed.state.camera.position);
});
check("WASD still works after tapping Alt", altMoved > 5, `moved ${altMoved.toFixed(2)} m`);

// ---- 14b. the right button is a modifier for nothing ------------------------
// WASD translates whether or not the right button is down: the two bindings do
// not overlap, so both can run at once.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.select([...ed.state.placements.keys()]);
  const cam = ed.state.camera;
  cam.position.set(0, 6, -18);
  cam.rotation.set(0.2, 0, 0);
  cam.cameraDirection.setAll(0);
  cam.cameraRotation.set(0, 0);
});
const canvasBox = await page.locator("#render-canvas").boundingBox();
const mid = { x: canvasBox.x + canvasBox.width / 2, y: canvasBox.y + canvasBox.height / 2 };
await page.mouse.move(mid.x, mid.y, { steps: 3 });

const camBefore = await page.evaluate(async () => {
  const c = (await import("/js/editor.js")).state.camera;
  return { rot: c.rotation.asArray(), pos: c.position.asArray() };
});
await page.mouse.down({ button: "right" });
await page.keyboard.down("d");
await page.waitForTimeout(350);
await page.keyboard.up("d");
await page.keyboard.down("w");
await page.waitForTimeout(350);
await page.keyboard.up("w");
await page.mouse.up({ button: "right" });
await page.waitForTimeout(500);

const camAfter = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const c = ed.state.camera;
  return { rot: c.rotation.asArray(), pos: c.position.asArray(), selection: ed.state.selection.length };
});
check("WASD translates even with the right button held",
  Math.hypot(...camAfter.pos.map((v, i) => v - camBefore.pos[i])) > 2,
  `moved ${Math.hypot(...camAfter.pos.map((v, i) => v - camBefore.pos[i])).toFixed(2)} m`);
check("the right button alone does not turn the view",
  Math.hypot(...camAfter.rot.map((v, i) => v - camBefore.rot[i])) < 1e-6,
  `rotation moved ${Math.hypot(...camAfter.rot.map((v, i) => v - camBefore.rot[i])).toFixed(5)}`);
check("driving with the right button held does not cancel", camAfter.selection === 1,
  `${camAfter.selection} still selected`);

// a plain right-click with no steering must still cancel
await page.mouse.click(mid.x, mid.y, { button: "right" });
await page.waitForTimeout(300);
const afterPlainRmb = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.selection.length);
check("plain RMB still cancels", afterPlainRmb === 0, `${afterPlainRmb} selected`);

// ---- 14c. right-drag looks, left-drag never touches the camera --------------
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const cam = ed.state.camera;
  cam.position.set(0, 6, -18);
  cam.rotation.set(0.15, 0.25, 0);
  cam.cameraDirection.setAll(0);
  cam.cameraRotation.set(0, 0);
  // a right-drag is a camera gesture, so it must leave the selection alone
  ed.select([...ed.state.placements.keys()].slice(0, 1));
});
const panBefore = await page.evaluate(async () => {
  const c = (await import("/js/editor.js")).state.camera;
  return { pos: c.position.asArray(), rot: c.rotation.asArray() };
});
await page.mouse.move(mid.x, mid.y, { steps: 2 });
await page.mouse.down({ button: "right" });
await page.mouse.move(mid.x + 120, mid.y, { steps: 8 });
await page.mouse.up({ button: "right" });
await page.waitForTimeout(300);
const panAfter = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const c = ed.state.camera;
  return { pos: c.position.asArray(), rot: c.rotation.asArray(), selection: ed.state.selection.length };
});
const moveDelta = Math.hypot(...panAfter.pos.map((v, i) => v - panBefore.pos[i]));
const yawDelta = panAfter.rot[1] - panBefore.rot[1];
check("right-drag looks around", Math.abs(yawDelta) > 0.05,
  `yaw moved ${yawDelta.toFixed(3)}`);
check("dragging right turns the view right", yawDelta > 0.05,
  `yaw delta = ${yawDelta.toFixed(3)}`);
check("looking does not move the camera", moveDelta < 1e-6,
  `position moved ${moveDelta.toFixed(5)}`);
check("right-drag does not cancel the selection", panAfter.selection === 1,
  `${panAfter.selection} still selected`);

// vertical right-drag pitches
const pitchBefore = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.camera.rotation.asArray());
await page.mouse.move(mid.x, mid.y, { steps: 2 });
await page.mouse.down({ button: "right" });
await page.mouse.move(mid.x, mid.y + 120, { steps: 8 });
await page.mouse.up({ button: "right" });
await page.waitForTimeout(300);
const pitchAfter = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.camera.rotation.asArray());
check("dragging down pitches the view down", pitchAfter[0] - pitchBefore[0] > 0.05,
  `pitch delta = ${(pitchAfter[0] - pitchBefore[0]).toFixed(3)}`);

// The whole point of moving look onto the right button: the left one is for
// editing only, so a drag in empty space must not disturb the view at all.
// Let the previous gesture's inertia decay first, or its residual would be
// mistaken for the left-drag doing something.
await page.evaluate(async () => {
  const c = (await import("/js/editor.js")).state.camera;
  c.cameraRotation.set(0, 0);
  c.cameraDirection.setAll(0);
});
await page.waitForTimeout(400);
const lookBefore = await page.evaluate(async () => {
  const c = (await import("/js/editor.js")).state.camera;
  return { pos: c.position.asArray(), rot: c.rotation.asArray() };
});
await page.mouse.move(mid.x, mid.y, { steps: 2 });
await page.mouse.down();
await page.mouse.move(mid.x + 120, mid.y, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(400);
const lookAfter = await page.evaluate(async () => {
  const c = (await import("/js/editor.js")).state.camera;
  return { pos: c.position.asArray(), rot: c.rotation.asArray() };
});
check("left-drag never moves the camera",
  Math.hypot(...lookAfter.rot.map((v, i) => v - lookBefore.rot[i])) < 1e-6
  && Math.hypot(...lookAfter.pos.map((v, i) => v - lookBefore.pos[i])) < 1e-6,
  `rot moved ${Math.hypot(...lookAfter.rot.map((v, i) => v - lookBefore.rot[i])).toFixed(5)}, ` +
  `pos moved ${Math.hypot(...lookAfter.pos.map((v, i) => v - lookBefore.pos[i])).toFixed(5)}`);

const drift = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const cam = ed.state.camera;
  const V = BABYLON.Vector3;
  ed.noteKey("KeyW", true, {});
  await new Promise((r) => setTimeout(r, 250));
  ed.releaseAllKeys();                       // simulates the window losing focus
  const a = cam.position.clone();
  await new Promise((r) => setTimeout(r, 400));
  const b = cam.position.clone();
  await new Promise((r) => setTimeout(r, 800));
  return { coast: V.Distance(a, b), after: V.Distance(b, cam.position.clone()) };
});
// movement runs through cameraDirection, so the camera coasts under the
// inertia setting instead of stopping dead - what matters is that it converges
check("camera coasts to a stop, does not run away",
  drift.after < drift.coast * 0.2 && drift.after < 0.1,
  `coasted ${drift.coast.toFixed(3)} m, then ${drift.after.toFixed(4)} m over the next 0.8 s`);

// ---- 15. right mouse button behaves like Escape -----------------------------
await page.evaluate((m) => import("/js/palette.js").then((p) => p.setBrush(m)), MODULE);
await page.waitForTimeout(900);
const beforeRmb = await page.evaluate(async () =>
  (await import("/js/interact.js")).ghostActive());
await page.mouse.click(pt.x, pt.y, { button: "right" });
await page.waitForTimeout(400);
const afterRmb = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return { ghost: i.ghostActive(), brush: ed.state.brush, selection: ed.state.selection.length };
});
check("RMB cancels like Escape",
  beforeRmb && !afterRmb.ghost && !afterRmb.brush && afterRmb.selection === 0,
  `ghost ${beforeRmb} -> ${afterRmb.ghost}, brush=${afterRmb.brush}`);

// ---- 16. selection beats hover, and the wheel edits the whole selection -----
const multi = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(8, 0, 0), { silent: true });
  const c = await ed.placeAt(moduleId, new V(16, 0, 0), { silent: true });
  ed.select([a.id, b.id]);
  const cur = i.currentElement();
  const before = [a, b, c].map((e) => ed.eulerOf(e.node)[1]);
  i.rotateCurrent(1);
  const after = [a, b, c].map((e) => ed.eulerOf(e.node)[1]);
  const beforeScale = [a, b, c].map((e) => e.node.scaling.x);
  i.scaleCurrent(1);
  const afterScale = [a, b, c].map((e) => e.node.scaling.x);
  return { kind: cur?.kind, label: cur?.module, ids: cur?.ids?.length, before, after, beforeScale, afterScale };
}, MODULE);
check("selection is the current element", multi.kind === "selection" && multi.ids === 2,
  `${multi.kind}, "${multi.label}"`);
check("E turns every selected element",
  Math.abs(multi.after[0] - multi.before[0]) > 89 &&
  Math.abs(multi.after[1] - multi.before[1]) > 89 &&
  Math.abs(multi.after[2] - multi.before[2]) < 1e-6,
  `${multi.before.map((v) => v.toFixed(0))} -> ${multi.after.map((v) => v.toFixed(0))} (third unselected)`);
check("wheel scales every selected element",
  Math.abs(multi.afterScale[0] - multi.beforeScale[0] - 0.1) < 1e-6 &&
  Math.abs(multi.afterScale[1] - multi.beforeScale[1] - 0.1) < 1e-6 &&
  Math.abs(multi.afterScale[2] - multi.beforeScale[2]) < 1e-6,
  `${multi.afterScale.map((v) => v.toFixed(2))}`);

// The old test only claimed to hover; it never moved the pointer, so it was
// really just asserting the no-hover fallback. Drive the real mouse instead.
const precedenceSetup = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(8, 0, 0), { silent: true });
  // frame both, or projecting B's centre lands off-canvas
  ed.state.camera.position = new V(5, 16, -13);
  ed.state.camera.setTarget(new V(5, 0, 0));
  const bb = ed.worldBounds(b.node);
  return { aId: a.id, bId: b.id, bCentre: bb.min.add(bb.max).scale(0.5).asArray() };
}, MODULE);
await page.waitForTimeout(500);

const overB = await screenOf(precedenceSetup.bCentre);
await page.mouse.move(overB.x, overB.y, { steps: 4 });
await page.waitForTimeout(350);
const precedence = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.select([ids.aId]);                       // select A, hover B
  const hovered = i.hoveredId();
  const kind = i.currentElement()?.kind;
  const aBefore = ed.eulerOf(ed.state.placements.get(ids.aId).node)[1];
  const bBefore = ed.eulerOf(ed.state.placements.get(ids.bId).node)[1];
  i.rotateCurrent(1);
  return {
    hovered, kind,
    aMoved: Math.abs(ed.eulerOf(ed.state.placements.get(ids.aId).node)[1] - aBefore) > 1e-6,
    bMoved: Math.abs(ed.eulerOf(ed.state.placements.get(ids.bId).node)[1] - bBefore) > 1e-6,
  };
}, precedenceSetup);
check("hovered takes precedence over selected",
  precedence.hovered === precedenceSetup.bId && precedence.kind === "hover"
    && precedence.bMoved && !precedence.aMoved,
  `current=${precedence.kind}, hovered moved=${precedence.bMoved}, selected moved=${precedence.aMoved}`);

// pointer off everything -> the selection is the current element again
await page.mouse.move(canvasBox.x + 12, canvasBox.y + 12, { steps: 4 });
await page.waitForTimeout(350);
const fallback = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const kind = i.currentElement()?.kind;
  const aBefore = ed.eulerOf(ed.state.placements.get(ids.aId).node)[1];
  i.rotateCurrent(1);
  return { kind,
    aMoved: Math.abs(ed.eulerOf(ed.state.placements.get(ids.aId).node)[1] - aBefore) > 1e-6 };
}, precedenceSetup);
check("with nothing hovered the selection is the current element",
  fallback.kind === "selection" && fallback.aMoved,
  `current=${fallback.kind}, selected moved=${fallback.aMoved}`);

// ---- 17. Alt+E rotates about a shared, grid-snapped pivot ------------------
// This used to be Alt+wheel; the wheel gave up rotation entirely, so the pivot
// swing moved onto the modifier of the key that still turns things.
const pivotRot = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  ed.state.snap.pos = 4;
  ed.state.rotAxis = "y";
  ed.state.snap.rot = 90;
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(8, 0, 0), { silent: true });
  ed.select([a.id, b.id]);

  const pivot = i.sharedPivot([a, b]);
  const before = [a, b].map((e) => e.node.position.asArray());
  i.rotateCurrent(1, true);                       // Alt+E
  const after = [a, b].map((e) => e.node.position.asArray());

  // a 90 deg turn about the pivot must preserve each element's distance to it
  const dist = (p) => Math.hypot(p[0] - pivot.x, p[2] - pivot.z);
  return {
    pivot: [pivot.x, pivot.y, pivot.z],
    onGrid: [pivot.x, pivot.y, pivot.z].every((v) => Math.abs(v / 4 - Math.round(v / 4)) < 1e-6),
    moved: before.some((p, k) => JSON.stringify(p) !== JSON.stringify(after[k])),
    radiiKept: before.every((p, k) => Math.abs(dist(p) - dist(after[k])) < 1e-4),
    yaw: ed.eulerOf(a.node)[1],
    before, after,
  };
}, MODULE);
check("the Alt+E pivot is snapped to the grid", pivotRot.onGrid,
  `pivot ${JSON.stringify(pivotRot.pivot)} with 4 m snap`);
check("Alt+E swings the group about the pivot",
  pivotRot.moved && pivotRot.radiiKept && Math.abs(Math.abs(pivotRot.yaw) - 90) < 0.01,
  `${JSON.stringify(pivotRot.before)} -> ${JSON.stringify(pivotRot.after)}, yaw ${pivotRot.yaw.toFixed(0)}`);

const noPivot = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const entries = [...ed.state.placements.values()];
  const before = entries.map((e) => e.node.position.asArray());
  i.rotateCurrent(1, false);                      // plain E
  const after = entries.map((e) => e.node.position.asArray());
  return before.every((p, k) => JSON.stringify(p) === JSON.stringify(after[k]));
});
check("plain E turns in place, no pivot swing", noPivot, "positions unchanged");

// ---- 18. Del: selection first, hovered as the fallback, never the ghost ----
const delSetup = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost();
  ed.clearAll();
  ed.select([]);
  const cam = ed.state.camera;
  cam.position.set(0, 10, -18);
  cam.cameraDirection.setAll(0);
  cam.setTarget(new V(0, 0, 0));
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(12, 0, 0), { silent: true });
  const bb = ed.worldBounds(a.node);
  return {
    started: ed.state.placements.size,
    aId: a.id, bId: b.id,
    aCentre: bb.min.add(bb.max).scale(0.5).asArray(),
  };
}, MODULE);

// hover one with nothing selected -> Del removes it
const hoverPt = await screenOf(delSetup.aCentre);
await page.mouse.move(hoverPt.x, hoverPt.y, { steps: 4 });
await page.waitForTimeout(400);
const hovering = await page.evaluate(async () =>
  (await import("/js/interact.js")).hoveredId());
await page.keyboard.press("Delete");
await page.waitForTimeout(300);
const afterHoverDel = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.placements.size);
check("Del removes the hovered element when nothing is selected",
  hovering === delSetup.aId && afterHoverDel === delSetup.started - 1,
  `hovered ${hovering}, ${delSetup.started} -> ${afterHoverDel}`);

const delRules = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  ed.select([]);
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(12, 0, 0), { silent: true });
  const bb = ed.worldBounds(a.node);
  return {
    started: ed.state.placements.size,
    aId: a.id, bId: b.id,
    aCentre: bb.min.add(bb.max).scale(0.5).asArray(),
  };
}, MODULE);

// mid-drag, Del must do nothing
const dragFrom = await screenOf(delRules.aCentre);
const dragTo2 = await screenOf([delRules.aCentre[0] - 8, delRules.aCentre[1], delRules.aCentre[2]]);
await page.mouse.move(dragFrom.x, dragFrom.y, { steps: 4 });
await page.waitForTimeout(200);
await page.mouse.down();
await page.mouse.move(dragTo2.x, dragTo2.y, { steps: 8 });
const midDragDel = await page.evaluate(async () =>
  (await import("/js/interact.js")).isDragging());
await page.keyboard.press("Delete");
await page.waitForTimeout(200);
const afterMidDragDel = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.placements.size);
await page.mouse.up();
await page.waitForTimeout(200);
check("Del does nothing while dragging",
  midDragDel && afterMidDragDel === delRules.started,
  `dragging=${midDragDel}, ${delRules.started} kept`);

// Hover outranks the selection: pointing at something is a more immediate
// statement of intent than a selection made earlier. Fresh scene - the drag
// above moved A, so its old centre no longer points at anything.
const delPrec = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(8, 0, 0), { silent: true });
  ed.state.camera.position = new V(5, 16, -13);
  ed.state.camera.setTarget(new V(5, 0, 0));
  const bb = ed.worldBounds(a.node);
  return { started: ed.state.placements.size, aId: a.id, bId: b.id,
           aCentre: bb.min.add(bb.max).scale(0.5).asArray() };
}, MODULE);
await page.waitForTimeout(500);

const hoverA = await screenOf(delPrec.aCentre);
await page.mouse.move(hoverA.x, hoverA.y, { steps: 4 });
await page.waitForTimeout(350);
const hoverWins = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.select([ids.bId]);                       // select one, hover the other
  const hovered = i.hoveredId();
  window.dispatchEvent(new KeyboardEvent("keydown",
    { key: "Delete", code: "Delete", bubbles: true }));
  return { hovered, left: ed.state.placements.size,
           survivor: [...ed.state.placements.values()][0]?.id,
           selection: [...ed.state.selection] };
}, delPrec);
check("Del takes the hovered element over the selection",
  hoverWins.hovered === delPrec.aId
    && hoverWins.left === delPrec.started - 1
    && hoverWins.survivor === delPrec.bId,
  `hovered ${hoverWins.hovered}, kept ${hoverWins.survivor}`);
check("deleting the hovered element keeps the rest of the selection",
  hoverWins.selection.length === 1 && hoverWins.selection[0] === delPrec.bId,
  `selection now [${hoverWins.selection}]`);

// with nothing under the cursor it falls back to the selection
await page.mouse.move(canvasBox.x + 12, canvasBox.y + 12, { steps: 4 });
await page.waitForTimeout(350);
const selDel = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.select([ids.bId]);
  const hovered = i.hoveredId();
  window.dispatchEvent(new KeyboardEvent("keydown",
    { key: "Delete", code: "Delete", bubbles: true }));
  return { hovered, left: ed.state.placements.size };
}, delPrec);
check("Del falls back to the selection with nothing hovered",
  selDel.hovered === null && selDel.left === delPrec.started - 2,
  `hovered=${selDel.hovered}, ${delPrec.started} -> ${selDel.left}`);

// ---- 18b. the inspector reports bounding-box dimensions --------------------
const dims = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  ed.state.snap.rot = 90;
  ed.state.rotAxis = "y";
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(12, 0, 0), { silent: true });

  const read = () => ["dim-x", "dim-y", "dim-z"]
    .map((id) => parseFloat(document.getElementById(id).textContent));
  const truth = (nodes) => {
    let mn = null, mx = null;
    for (const n of nodes) {
      const w = ed.worldBounds(n);
      mn = mn ? V.Minimize(mn, w.min) : w.min.clone();
      mx = mx ? V.Maximize(mx, w.max) : w.max.clone();
    }
    return mx.subtract(mn).asArray();
  };

  ed.select([a.id]);
  const single = read(), singleTruth = truth([a.node]);

  // the world box must follow a rotation
  ed.setEuler(a.node, [0, 90, 0]);
  ed.emit("transform");
  const rotated = read(), rotatedTruth = truth([a.node]);

  ed.select([a.id, b.id]);
  const combined = read(), combinedTruth = truth([a.node, b.node]);
  const note = document.getElementById("dim-note").textContent;

  return { single, singleTruth, rotated, rotatedTruth, combined, combinedTruth, note };
}, MODULE);

const near = (shown, real, tol = 0.011) =>
  shown.every((v, i) => Math.abs(v - real[i]) <= tol);
check("inspector shows the world bounding box",
  near(dims.single, dims.singleTruth),
  `${dims.single.join(" x ")} vs ${dims.singleTruth.map((v) => v.toFixed(2)).join(" x ")}`);
check("dimensions follow a rotation",
  near(dims.rotated, dims.rotatedTruth)
  && Math.abs(dims.rotated[0] - dims.single[0]) > 0.5,
  `${dims.single.join(" x ")} -> ${dims.rotated.join(" x ")} after 90 deg`);
check("multi-selection shows the combined box",
  near(dims.combined, dims.combinedTruth) && dims.note.includes("2 objects"),
  `${dims.combined.join(" x ")}, note "${dims.note}"`);

// ---- 18c. negative (mirrored) scale ----------------------------------------
const mirror = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  ed.select([a.id]);

  const field = document.getElementById("scl-x");
  const type = (text) => {
    field.focus();
    field.value = text;
    field.dispatchEvent(new Event("input", { bubbles: true }));
  };
  document.getElementById("scl-uniform").checked = false;

  // typing a negative goes through an intermediate bare "-"
  type("-");
  const afterMinus = a.node.scaling.x;
  type("-1");
  const afterMinusOne = a.node.scaling.x;

  // and the wheel must resize a mirrored element instead of flipping it back
  ed.state.scaleAxis = "x";
  ed.state.snap.scale = 0.1;
  i.scaleCurrent(1);
  const afterWheelUp = a.node.scaling.x;
  i.scaleCurrent(-1);
  const afterWheelDown = a.node.scaling.x;

  ed.state.scaleAxis = "all";
  document.getElementById("scl-uniform").checked = true;
  return { afterMinus, afterMinusOne, afterWheelUp, afterWheelDown };
}, MODULE);
check("a partial '-' does not clobber the scale", mirror.afterMinus === 1,
  `held at ${mirror.afterMinus} while typing`);
check("inspector accepts a negative scale", mirror.afterMinusOne === -1,
  `scale.x = ${mirror.afterMinusOne}`);
check("wheel resizes a mirrored element, keeping the sign",
  Math.abs(mirror.afterWheelUp - -1.1) < 1e-6 && Math.abs(mirror.afterWheelDown - -1) < 1e-6,
  `-1 -> ${mirror.afterWheelUp} -> ${mirror.afterWheelDown}`);

// ---- 18c-bis. decimals can actually be typed -------------------------------
// Editing a field applies the value and re-selects, which refreshes the whole
// inspector. Rewriting the *focused* field with the parsed number destroys
// whatever is half-typed: "1." parses as 1 and comes straight back as "1", so
// the decimal point vanished as fast as it was typed.
const decimalId = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  document.getElementById("scl-uniform").checked = false;
  const e = await ed.placeAt("Walls/ShortWall_Band2_Straight",
    new BABYLON.Vector3(0, 0, 0), { silent: true });
  ed.select([e.id]);
  return e.id;
});
await page.waitForTimeout(400);

async function typeField(sel, text) {
  await page.focus(sel);
  await page.keyboard.press("Control+a");
  await page.keyboard.press("Delete");
  for (const ch of text) { await page.keyboard.type(ch); await page.waitForTimeout(60); }
  return page.inputValue(sel);
}
const typedX = await typeField("#pos-x", "2.75");
const typedS = await typeField("#scl-x", "0.05");
const typedR = await typeField("#rot-y", "22.5");
await page.evaluate(() => document.activeElement.blur());
await page.waitForTimeout(300);
const decimals = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const e = ed.state.placements.get(id);
  return { x: e.node.position.x, sx: +e.node.scaling.x.toFixed(3),
           rotY: +ed.eulerOf(e.node)[1].toFixed(2) };
}, decimalId);
check("a decimal point survives being typed into the inspector",
  typedX === "2.75" && typedS === "0.05" && typedR === "22.5",
  `pos-x "${typedX}", scl-x "${typedS}", rot-y "${typedR}"`);
check("the typed decimals reach the element",
  Math.abs(decimals.x - 2.75) < 1e-6 && Math.abs(decimals.sx - 0.05) < 1e-6
    && Math.abs(decimals.rotY - 22.5) < 0.01,
  `x=${decimals.x}, scale.x=${decimals.sx}, rotY=${decimals.rotY}`);

// `input` fires per character, so "2.75" used to push four snapshots and cost
// four Ctrl+Z. One visit to a field is now one entry.
const undoOnce = await (async () => {
  const before = await page.evaluate(async (id) =>
    (await import("/js/editor.js")).state.placements.get(id).node.position.x, decimalId);
  await typeField("#pos-x", "9.25");
  await page.evaluate(() => document.activeElement.blur());
  const typed = await page.evaluate(async (id) =>
    (await import("/js/editor.js")).state.placements.get(id).node.position.x, decimalId);
  await page.evaluate(async () => { await (await import("/js/editor.js")).undo(); });
  const after = await page.evaluate(async (id) =>
    (await import("/js/editor.js")).state.placements.get(id).node.position.x, decimalId);
  return { before, typed, after };
})();
check("four keystrokes in one field cost a single undo",
  Math.abs(undoOnce.typed - 9.25) < 1e-6 && Math.abs(undoOnce.after - undoOnce.before) < 1e-6,
  `${undoOnce.before} -> typed ${undoOnce.typed} -> undo ${undoOnce.after}`);
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  document.getElementById("scl-uniform").checked = true;
  ed.clearAll(); ed.select([]);
});

// ---- 18d. F flips on the current scale axis --------------------------------
const press = (key, code) => page.evaluate(([k, c]) => window.dispatchEvent(
  new KeyboardEvent("keydown", { key: k, code: c, bubbles: true })), [key, code]);

const flip = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  ed.select([]);
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(moduleId, new V(8, 0, 0), { silent: true });
  const f = (axis) => {
    ed.state.scaleAxis = axis;
    return i.flipCurrent();
  };

  ed.select([a.id]);
  f("x");
  const oneX = a.node.scaling.asArray();
  f("z");
  const oneZ = a.node.scaling.asArray();

  // "all" is a point inversion rather than a mirror, so it maps to X
  ed.select([b.id]);
  const allResult = f("all");
  const allScale = b.node.scaling.asArray();

  // a whole selection flips together
  ed.select([a.id, b.id]);
  ed.state.scaleAxis = "y";
  const multi = i.flipCurrent();
  const both = [a.node.scaling.y, b.node.scaling.y];

  ed.state.scaleAxis = "all";
  return { oneX, oneZ, allResult, allScale, multi, both };
}, MODULE);
check("F flips on the chosen axis",
  flip.oneX[0] === -1 && flip.oneX[2] === 1 && flip.oneZ[0] === -1 && flip.oneZ[2] === -1,
  `x-flip ${JSON.stringify(flip.oneX)}, then z-flip ${JSON.stringify(flip.oneZ)}`);
check("scale axis 'all' flips X, not all three",
  flip.allResult.axis === "x" && JSON.stringify(flip.allScale) === JSON.stringify([-1, 1, 1]),
  `axis=${flip.allResult.axis}, scale ${JSON.stringify(flip.allScale)}`);
check("F flips the whole selection",
  flip.multi.count === 2 && flip.both.every((v) => v === -1),
  `${flip.multi.count} objects, scale.y ${JSON.stringify(flip.both)}`);

// and it flips whatever is mid-drag
const dragFlip = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  const cam = ed.state.camera;
  cam.position.set(0, 12, -18); cam.cameraDirection.setAll(0);
  cam.setTarget(new V(0, 0, 0));
  const a = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  ed.select([]);
  const bb = ed.worldBounds(a.node);
  return { id: a.id, centre: bb.min.add(bb.max).scale(0.5).asArray() };
}, MODULE);
const dFrom = await screenOf(dragFlip.centre);
const dTo = await screenOf([dragFlip.centre[0] - 8, dragFlip.centre[1], dragFlip.centre[2]]);
await page.mouse.move(dFrom.x, dFrom.y, { steps: 4 });
await page.waitForTimeout(200);
await page.mouse.down();
await page.mouse.move(dTo.x, dTo.y, { steps: 8 });
await press("f", "KeyF");
await page.waitForTimeout(150);
const midFlip = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return { dragging: i.isDragging(), scale: [...ed.state.placements.values()][0].node.scaling.asArray() };
});
await page.mouse.up();
await page.waitForTimeout(200);
check("F flips the element being dragged",
  midFlip.dragging && midFlip.scale[0] === -1,
  `dragging=${midFlip.dragging}, scale ${JSON.stringify(midFlip.scale)}`);

// and the ghost, before it is even placed
await page.evaluate((m) => import("/js/palette.js").then((p) => p.setBrush(m)), MODULE);
await page.waitForTimeout(1000);
await press("f", "KeyF");
await page.waitForTimeout(150);
const ghostFlip = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").scaling.asArray());
check("F flips the ghost before placing", ghostFlip[0] === -1,
  `ghost scale ${JSON.stringify(ghostFlip)}`);
await page.evaluate(() => import("/js/palette.js").then((p) => p.setBrush(null)));

// ---- 19. rapid clicks while placing must place, not grab -------------------
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost();
  ed.clearAll();
  ed.select([]);
  // the navigation tests left the camera somewhere arbitrary
  const cam = ed.state.camera;
  cam.position.set(0, 10, -18);
  cam.cameraDirection.setAll(0);
  cam.setTarget(new BABYLON.Vector3(0, 0, 0));
});
await page.evaluate((m) => import("/js/palette.js").then((p) => p.setBrush(m)), MODULE);
await page.waitForTimeout(1000);
const placeSpot = await screenOf([0, 0, 0]);
await page.mouse.move(placeSpot.x, placeSpot.y, { steps: 4 });
await page.waitForTimeout(250);
await page.mouse.down(); await page.mouse.up();
await page.waitForTimeout(60);
await page.mouse.down(); await page.mouse.up();       // same spot, double-click speed
await page.waitForTimeout(700);
const rapid = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return {
    placements: ed.state.placements.size,
    stillArmed: i.ghostActive(),
    grabbed: i.currentElement()?.kind === "ghost" && !!i.ghostModule(),
  };
});
check("double-click speed still places twice while armed",
  rapid.placements === 2 && rapid.stillArmed,
  `${rapid.placements} placed, still armed=${rapid.stillArmed}`);
await page.evaluate(() => import("/js/palette.js").then((p) => p.setBrush(null)));

// ---- 1g. viewport exposure ---------------------------------------------------
// The pale kit panels (MI_Trim_03) sit at ~240/255 at exposure 1.0, deep in the
// KHR PBR Neutral shoulder where the curve flattens and all surface detail is
// squeezed out. Lowering exposure moves them back onto the straight part.
const exposure = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const s = window.__scene;
  const startedAt = s.imageProcessingConfiguration.exposure;
  const isDefault = Math.abs(startedAt - ed.EXPOSURE_DEFAULT) < 1e-6;

  ed.clearAll();
  await ed.placeAt("Platforms/Platform_Simple", new BABYLON.Vector3(0, 0, 0));
  await new Promise((r) => setTimeout(r, 1200));
  ed.select([]);
  ed.setGridVisible(false);
  ed.state.camera.position = new BABYLON.Vector3(0, 3.0, -2.4);
  ed.state.camera.setTarget(new BABYLON.Vector3(0, 0, 0));

  const contrast = () => {
    s.render(); s.render();
    const cv = s.getEngine().getRenderingCanvas();
    const g = document.createElement("canvas"); g.width = cv.width; g.height = cv.height;
    g.getContext("2d").drawImage(cv, 0, 0);
    const d = g.getContext("2d").getImageData(
      Math.floor(cv.width * 0.35), Math.floor(cv.height * 0.4),
      Math.floor(cv.width * 0.3), Math.floor(cv.height * 0.3)).data;
    let n = 0, sum = 0, sq = 0;
    for (let k = 0; k < d.length; k += 4) {
      const v = 0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2];
      if (v > 40) { n++; sum += v; sq += v * v; }
    }
    const mean = sum / Math.max(n, 1);
    return { mean, sd: Math.sqrt(Math.max(sq / Math.max(n, 1) - mean * mean, 0)) };
  };

  ed.setExposure(1.0);
  const hot = contrast();
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  const tuned = contrast();

  ed.setExposure(9);   const clampHi = ed.state.exposure;
  ed.setExposure(0);   const clampLo = ed.state.exposure;
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  ed.setGridVisible(true);
  ed.clearAll();
  return { isDefault, startedAt, hot, tuned, clampHi, clampLo,
           applied: s.imageProcessingConfiguration.exposure };
});
check("scene starts at the tuned default exposure",
  exposure.isDefault, `exposure = ${exposure.startedAt}`);
check("lower exposure restores contrast on pale panels",
  exposure.tuned.sd > exposure.hot.sd * 1.3,
  `sd ${exposure.hot.sd.toFixed(2)} @1.0 -> ${exposure.tuned.sd.toFixed(2)} @${exposure.applied} ` +
  `(mean ${exposure.hot.mean.toFixed(0)} -> ${exposure.tuned.mean.toFixed(0)})`);
check("exposure is clamped to a sane range",
  exposure.clampHi === 2 && exposure.clampLo === 0.15,
  `high=${exposure.clampHi} low=${exposure.clampLo}`);

// ---- 1h. palette turntables --------------------------------------------------
// Thumbnails are lit again - unlit threw away too much - and hovering a tile
// spins the module through a full turn so a corner can be told from a straight.
const litThumbs = await page.evaluate(async () => {
  const t = await import("/js/thumbs.js");
  return { frames: t.TURN_FRAMES };
});
check("a turntable is a full 12-frame turn", litThumbs.frames === 12);

await page.fill("#palette-search", "Platform_Simple");
await page.waitForTimeout(2500);
const firstTile = page.locator("#palette-list .item").first();
await firstTile.hover();
await page.waitForTimeout(9000);
const spinning = await page.evaluate(async () => {
  const pal = await import("/js/palette.js");
  const el = document.querySelector("#palette-list .item");
  const film = el?.querySelector(".turn");
  const seen = new Set();
  for (let i = 0; i < 26; i++) {
    seen.add(el?.querySelector(".turn")?.style.backgroundPositionX);
    await new Promise((r) => setTimeout(r, 120));
  }
  return { ...pal.turntableState(), hasFilm: !!film,
           size: film?.style.backgroundSize, distinct: seen.size };
});
check("hovering a tile starts a turntable",
  spinning.running && spinning.hasFilm && spinning.size === "1200% 100%",
  `tile=${spinning.tile} size=${spinning.size}`);
check("the turntable actually advances through frames",
  spinning.distinct >= 6, `${spinning.distinct} distinct frame offsets seen`);

const sheet = await page.evaluate(async () => {
  const r = await fetch("/api/turns");
  const { cached } = await r.json();
  if (!cached.length) return { cached: 0 };
  const img = new Image();
  await new Promise((ok, no) => { img.onload = ok; img.onerror = no; img.src = `/api/turn/${cached[0]}`; });
  return { cached: cached.length, w: img.naturalWidth, h: img.naturalHeight, key: cached[0] };
});
check("the sheet is cached server-side as 12 square frames",
  sheet.cached > 0 && sheet.w === sheet.h * 12,
  `${sheet.key} ${sheet.w}x${sheet.h}, ${sheet.cached} cached`);

await page.mouse.move(1200, 700, { steps: 4 });
await page.waitForTimeout(500);
const stopped = await page.evaluate(async () => {
  const pal = await import("/js/palette.js");
  return { ...pal.turntableState(), films: document.querySelectorAll("#palette-list .turn").length };
});
check("leaving the tile stops the turntable and cleans up",
  !stopped.running && stopped.tile === null && stopped.films === 0,
  `running=${stopped.running} films=${stopped.films}`);

// Stills and turntables share one thumbnail scene but are triggered
// independently - scrolling and hovering. A turntable keeps its module in the
// scene across 12 renders, so an unsynchronised still rendered in that window
// came out with two different modules in the picture.
const race = await page.evaluate(async () => {
  const t = await import("/js/thumbs.js");
  const kit = await import("/js/kit.js");
  const cat = kit.getCatalogue();
  const find = (n) => {
    for (const c of cat.categories) for (const m of c.modules) if (m.name === n) return m;
    return null;
  };
  const still = find("ShortWall_Simple1_Corner_Outer");
  const spin = find("ShortWall_MetalPlates_Corner_Outer");
  if (!still || !spin) return { skip: true };

  // force both to actually render rather than come back from the cache
  t.forgetCached(still.id);
  t.forgetCached(spin.id);
  const thumbScene = BABYLON.EngineStore.Instances.map((e) => e.scenes[0])
    .find((s) => s && s !== window.__scene);

  let peakRoots = 0, worst = [];
  const spy = setInterval(() => {
    const roots = new Set(thumbScene.meshes.filter((m) => m.getTotalVertices() > 0)
      .map((m) => m.name.replace(/_primitive\d+$/, "")));
    if (roots.size > peakRoots) { peakRoots = roots.size; worst = [...roots]; }
  }, 15);

  const img = document.createElement("img");
  const stillDone = new Promise((r) => {
    const iv = setInterval(() => {
      if (img.src && img.src.length > 100) { clearInterval(iv); r(); }
    }, 40);
  });
  t.request(still, img);
  const turn = t.requestTurntable(spin);
  await Promise.all([stillDone, turn]);
  await new Promise((r) => setTimeout(r, 200));
  clearInterval(spy);
  return { skip: false, peakRoots, worst, ...t.thumbConcurrency() };
});
check("a still and a turntable never share the thumbnail scene",
  race.skip || (race.peakRoots <= 1 && race.peak <= 1),
  race.skip ? "(modules not in catalogue)"
    : `peak ${race.peak} render(s), ${race.peakRoots} module(s) on stage [${race.worst}]`);

// ---- 1i. the button split ----------------------------------------------------
// Look lives on the right button and editing on the left, with no overlap, so
// there is no gesture that could do the wrong one of the two.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.state.camera.position = new BABYLON.Vector3(0, 3, -10);
  ed.state.camera.setTarget(new BABYLON.Vector3(0, 0, 0));
});
const navSpot = { x: 1100, y: 500 };

async function cameraDelta(gesture) {
  const before = await page.evaluate(() => {
    const c = window.__scene.activeCamera;
    return { pos: c.position.asArray(), rot: [c.rotation.x, c.rotation.y] };
  });
  await gesture();
  await page.waitForTimeout(320);
  return page.evaluate((b) => {
    const c = window.__scene.activeCamera;
    const dp = Math.hypot(c.position.x - b.pos[0], c.position.y - b.pos[1], c.position.z - b.pos[2]);
    const dr = Math.abs(c.rotation.x - b.rot[0]) + Math.abs(c.rotation.y - b.rot[1]);
    return { dp: +dp.toFixed(3), dr: +dr.toFixed(3) };
  }, before);
}

const dragLmb = () => (async () => {
  await page.mouse.move(navSpot.x, navSpot.y, { steps: 3 });
  await page.mouse.down();
  await page.mouse.move(navSpot.x + 160, navSpot.y + 60, { steps: 6 });
  await page.mouse.up();
})();
const dragRmb = () => (async () => {
  await page.mouse.move(navSpot.x, navSpot.y, { steps: 3 });
  await page.mouse.down({ button: "right" });
  await page.mouse.move(navSpot.x + 160, navSpot.y + 60, { steps: 6 });
  await page.mouse.up({ button: "right" });
})();
const rmbWasd = () => (async () => {
  await page.mouse.move(navSpot.x, navSpot.y, { steps: 3 });
  await page.mouse.down({ button: "right" });
  await page.keyboard.down("d");
  await page.waitForTimeout(320);
  await page.keyboard.up("d");
  await page.mouse.up({ button: "right" });
})();
// The two bindings do not overlap, so they have to work at the same time:
// turning with the mouse while walking with the keys is the normal way to move.
const rmbDragAndWasd = () => (async () => {
  await page.mouse.move(navSpot.x, navSpot.y, { steps: 3 });
  await page.mouse.down({ button: "right" });
  await page.keyboard.down("w");
  await page.mouse.move(navSpot.x + 150, navSpot.y + 50, { steps: 8 });
  await page.waitForTimeout(320);
  await page.keyboard.up("w");
  await page.mouse.up({ button: "right" });
})();

const lmb = await cameraDelta(dragLmb);
const rmb = await cameraDelta(dragRmb);
const wasd = await cameraDelta(rmbWasd);
const together = await cameraDelta(rmbDragAndWasd);
check("right-drag is the only mouse gesture that looks",
  rmb.dr > 0.02 && lmb.dr < 0.005,
  `RMB drot=${rmb.dr}, LMB drot=${lmb.dr}`);
check("no mouse gesture moves the camera any more",
  rmb.dp < 0.05 && lmb.dp < 0.05,
  `RMB dpos=${rmb.dp}, LMB dpos=${lmb.dp}`);
check("the right button does nothing special to WASD - they translate",
  wasd.dp > 0.5 && wasd.dr < 0.005,
  `dpos=${wasd.dp}, drot=${wasd.dr}`);
check("turning with the mouse and moving with WASD work at the same time",
  together.dp > 0.5 && together.dr > 0.02,
  `dpos=${together.dp}, drot=${together.dr}`);

// ---- 1i-bis. walk mode -------------------------------------------------------
// Judging corridor heights and sight lines needs the player's eye, not a free
// camera: walk mode pins the height to whatever floor is underfoot.
const walkSetup = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  for (let k = 0; k < 12; k++) {
    await ed.placeAt("Platforms/Platform_3Plates", new V(k * 4, 0, 0));
  }
  ed.select([]);
  ed.state.camera.position = new V(0, 25, 0);
  ed.state.camera.rotation.set(0, Math.PI / 2, 0);       // face +X, level
  ed.state.camera.cameraDirection.setAll(0);
  return { eye: ed.EYE_HEIGHT, from: ed.state.camera.position.y };
});
await page.waitForTimeout(600);

await page.check("#walk");
await page.waitForTimeout(400);
const grounded = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { walk: ed.state.walk, y: +ed.state.camera.position.y.toFixed(3) };
});
check("Walk drops the camera to eye height on the floor",
  grounded.walk && Math.abs(grounded.y - walkSetup.eye) < 0.01,
  `${walkSetup.from} m -> ${grounded.y} m (eye ${walkSetup.eye})`);

// pitched hard down, forward must still go forward - not into the floor
await page.evaluate(async () => {
  (await import("/js/editor.js")).state.camera.rotation.x = 1.0;
});
const beforeWalk = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.camera.position.asArray());
await page.mouse.move(1100, 500, { steps: 2 });
await page.keyboard.down("w");
await page.waitForTimeout(500);
await page.keyboard.up("w");
await page.waitForTimeout(500);
const afterWalk = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.camera.position.asArray());
check("walking looks down without walking into the floor",
  Math.abs(afterWalk[1] - walkSetup.eye) < 0.01
    && Math.abs(afterWalk[0] - beforeWalk[0]) > 2,
  `moved ${(afterWalk[0] - beforeWalk[0]).toFixed(2)} m along X, height ${afterWalk[1].toFixed(2)} m`);

// walking covers ground at the same rate as flying - it is a viewpoint, not a
// speed limit
const walkPace = Math.abs(afterWalk[0] - beforeWalk[0]) / 0.5;
check("walking is as quick as flying", walkPace > 15,
  `about ${walkPace.toFixed(0)} m/s`);

// Space/C must not fight the grounding
await page.keyboard.down(" ");
await page.waitForTimeout(600);
await page.keyboard.up(" ");
await page.waitForTimeout(400);
const afterSpace = await page.evaluate(async () =>
  +(await import("/js/editor.js")).state.camera.position.y.toFixed(3));
check("Space does not fly while walking",
  Math.abs(afterSpace - walkSetup.eye) < 0.01, `height ${afterSpace} m`);

// stepping off the end of the floor must not drop the camera out of the world
const offEdge = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const c = ed.state.camera;
  c.position.set(400, ed.EYE_HEIGHT, 400);              // nothing underneath
  c.cameraDirection.setAll(0);
  const ground = ed.groundHeightAt(400, 400, ed.EYE_HEIGHT);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return { ground, y: c.position.y };
});
check("no floor underfoot leaves the camera where it is",
  offEdge.ground === null && Math.abs(offEdge.y - walkSetup.eye) < 0.01,
  `ground=${offEdge.ground}, height ${offEdge.y.toFixed(2)} m`);

// Entering walk mode is the opposite case and must never be a no-op: a camera
// parked over a gap, or below a floor, still has to end up at eye height.
// This is the bug that made the toggle look like it did nothing on a real,
// half-built ship, where most of the bounding box is empty space.
const entering = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const c = ed.state.camera;
  const out = {};

  ed.setWalk(false);
  c.position.set(500, 40, 500);                   // nothing anywhere near
  c.cameraDirection.setAll(0);
  ed.setWalk(true);
  out.overNothing = +c.position.y.toFixed(2);
  out.gridY = ed.state.gridY;

  ed.setWalk(false);
  c.position.set(4, 0.2, 0);                      // *below* the floor surface
  c.cameraDirection.setAll(0);
  ed.setWalk(true);
  out.fromBelow = +c.position.y.toFixed(2);

  ed.setWalk(false);
  c.position.set(4, 120, 0);                      // far above it
  c.cameraDirection.setAll(0);
  ed.setWalk(true);
  out.fromHigh = +c.position.y.toFixed(2);
  return out;
});
check("entering Walk over empty space falls back to the build plane",
  Math.abs(entering.overNothing - (entering.gridY + walkSetup.eye)) < 0.01,
  `height ${entering.overNothing} m, build plane at ${entering.gridY} m`);
check("entering Walk from below the floor still stands on it",
  Math.abs(entering.fromBelow - walkSetup.eye) < 0.01,
  `0.2 m -> ${entering.fromBelow} m`);
check("entering Walk from high above lands on the floor",
  Math.abs(entering.fromHigh - walkSetup.eye) < 0.01,
  `120 m -> ${entering.fromHigh} m`);

await page.uncheck("#walk");
await page.waitForTimeout(300);
const flyAgain = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.camera.position.set(0, 6, 0);
  ed.state.camera.cameraDirection.setAll(0);
  return ed.state.walk;
});
await page.keyboard.down(" ");
await page.waitForTimeout(600);
await page.keyboard.up(" ");
await page.waitForTimeout(400);
const flewUp = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.camera.position.y);
check("unticking Walk gives Space back",
  flyAgain === false && flewUp > 6.5, `rose to ${flewUp.toFixed(2)} m`);
await page.evaluate(async () => (await import("/js/editor.js")).clearAll());

// ---- 1i-ter. Ctrl+D hands you the copy ---------------------------------------
// A duplicate dropped beside the original has to be dragged where you wanted it
// anyway, so Ctrl+D arms a ghost instead - and it must carry the source's
// rotation and mirroring, or a turned or flipped piece copies wrong.
const dupSetup = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const e = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0),
    { rotation: [0, 90, 0], scale: [-1, 1, 1] });
  ed.select([e.id]);
  ed.state.camera.position = new V(0, 14, -10);
  ed.state.camera.setTarget(new V(0, 0, 0));
  return { id: e.id, module: e.module, count: ed.state.placements.size };
});
await page.waitForTimeout(600);
await page.mouse.move(1150, 520, { steps: 3 });
await page.keyboard.press("Control+d");
await page.waitForTimeout(900);
const dupGhost = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const node = window.__scene.getTransformNodeByName("GHOST");
  return {
    module: i.ghostModule(), kind: i.currentElement()?.kind, brush: ed.state.brush,
    count: ed.state.placements.size,
    scaling: node?.scaling.asArray(),
    rotY: node ? +(node.rotationQuaternion.toEulerAngles().y * 180 / Math.PI).toFixed(1) : null,
  };
});
check("Ctrl+D arms a ghost instead of dropping a copy",
  dupGhost.kind === "ghost" && dupGhost.module === dupSetup.module
    && dupGhost.count === dupSetup.count,
  `ghost=${dupGhost.module}, still ${dupGhost.count} placement(s)`);
check("the duplicate ghost keeps the source rotation and mirroring",
  Math.abs(dupGhost.rotY - 90) < 0.5 && dupGhost.scaling[0] === -1,
  `rotY=${dupGhost.rotY}, scale=[${dupGhost.scaling}]`);
check("the duplicate arms the palette brush like any other module",
  dupGhost.brush === dupSetup.module, `brush=${dupGhost.brush}`);

await page.mouse.down(); await page.mouse.up();
await page.waitForTimeout(800);
const dupPlaced = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const list = [...ed.state.placements.values()];
  const last = list[list.length - 1];
  return { count: list.length, module: last.module,
           rotY: +ed.eulerOf(last.node)[1].toFixed(1), scale: last.node.scaling.asArray() };
});
check("clicking places the duplicate with its transform intact",
  dupPlaced.count === dupSetup.count + 1 && Math.abs(dupPlaced.rotY - 90) < 0.5
    && dupPlaced.scale[0] === -1,
  `${dupPlaced.count} placements, rotY=${dupPlaced.rotY}, scale=[${dupPlaced.scale}]`);

// ---- 1d-duovicies. the ghost carries a whole selection ---------------------
// It used to hold exactly one module, which is why Ctrl+D on a multi-selection
// duplicated in place and why moving anything needed a second mechanism.
const carrySet = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const M = "Walls/ShortWall_Band2_Straight";
  const a = await ed.placeAt(M, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(M, new V(8, 0, 0), { rotation: [0, 90, 0], silent: true });
  const c = await ed.placeAt(M, new V(0, 0, 8), { scale: [-1, 1, 1], silent: true });
  ed.state.scene.pointerX = ed.state.engine.getRenderWidth() / 2;
  ed.state.scene.pointerY = ed.state.engine.getRenderHeight() / 2;
  ed.select([a.id, b.id, c.id]);
  const g = await i.grabSelection();
  return {
    ids: [a.id, b.id, c.id],
    carrying: i.ghostCount(), mode: i.ghostMode(),
    // the originals go out of sight while the ghost stands in for them
    originalsOn: [a, b, c].map((e) => e.node.isEnabled()),
    // and the set keeps its shape: distinct offsets from the anchor
    spread: g ? g.items.map((it) => it.node.position.asArray().map((v) => +v.toFixed(2)).join()) : [],
  };
});
check("G picks the whole selection up", carrySet.carrying === 3 && carrySet.mode === "move",
  `${carrySet.carrying} items, mode ${carrySet.mode}`);
check("the originals step aside while it is carried",
  carrySet.originalsOn.join() === "false,false,false", `${carrySet.originalsOn}`);
check("the set keeps its shape on the cursor",
  // one at the anchor, one 8 m along X, one 8 m along Z - lengths alone would
  // not tell the last two apart
  new Set(carrySet.spread).size === 3 && carrySet.spread.includes("0,0,0"),
  `offsets [${carrySet.spread.join(" | ")}]`);

// Esc puts them back exactly, having moved nothing
const carryEsc = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost();
  const es = ids.map((id) => ed.state.placements.get(id));
  return {
    on: es.map((e) => e.node.isEnabled()),
    at: es.map((e) => +e.node.position.x.toFixed(2)),
    ghost: i.ghostActive(),
  };
}, carrySet.ids);
check("Esc puts a carried selection back untouched",
  carryEsc.on.join() === "true,true,true" && carryEsc.at.join() === "0,8,0"
    && carryEsc.ghost === false,
  `enabled ${carryEsc.on}, x ${carryEsc.at}`);

// dropping moves the originals rather than duplicating them
const carryDropped = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.select(ids);
  await i.grabSelection();
  // stand the ghost somewhere definite, then land it
  const g = ed.state.scene.getTransformNodeByName("GHOST");
  g.position.set(20, 0, 20);
  await i.dropGhost();
  const es = ids.map((id) => ed.state.placements.get(id));
  return {
    count: ed.state.placements.size,
    on: es.map((e) => e.node.isEnabled()),
    // relative shape preserved: b was 8 m along X from a, c 8 m along Z
    dx: +(es[1].node.position.x - es[0].node.position.x).toFixed(2),
    dz: +(es[2].node.position.z - es[0].node.position.z).toFixed(2),
    rotB: +ed.eulerOf(es[1].node)[1].toFixed(1),
    sclC: es[2].node.scaling.asArray(),
    ghost: i.ghostActive(),
  };
}, carrySet.ids);
check("dropping a carried selection moves it, and does not copy it",
  carryDropped.count === 3 && carryDropped.on.join() === "true,true,true" && carryDropped.ghost === false,
  `${carryDropped.count} placements, enabled ${carryDropped.on}`);
check("the set lands with its shape, turns and mirroring intact",
  carryDropped.dx === 8 && carryDropped.dz === 8 && Math.abs(carryDropped.rotB - 90) < 0.5
    && carryDropped.sclC[0] === -1,
  `dx=${carryDropped.dx}, dz=${carryDropped.dz}, rotB=${carryDropped.rotB}, scaleC=[${carryDropped.sclC}]`);

// ---- a grab is relative: it starts in place and follows how far you move ---
// It used to teleport the elements onto the cursor the moment the key went
// down, which threw a piece halfway across the room and re-snapped anything
// deliberately placed off the grid.
const relGrab = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  // Deliberately does not clear the scene: the blocks around this one share
  // their placements, and wiping them here left the next test reaching into
  // elements that no longer existed.
  i.cancelGhost(); ed.select([]);
  // The move step is a global other blocks cycle, and a grab snaps its *delta*
  // to it - at 4 m a 3 m drag would land 4 m away and this would read as a bug
  // in the grab rather than in the test.
  const step = ed.state.snap.pos;
  ed.state.snap.pos = 1;
  const camWas = {
    pos: ed.state.camera.position.asArray(),
    target: ed.state.camera.getTarget().asArray(),
  };
  // off the 1 m grid on purpose, so a re-snap would show
  const e = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(10.27, 0, 3.4),
    { silent: true });
  // Look almost straight down, so the plane the ghost tracks (one through the
  // module's body, a metre up on a wall) and the y = 0 plane this test projects
  // onto agree to within millimetres instead of a metre of parallax. Inertia
  // is cleared too, or the camera drifts between the two measurements and the
  // delta comes out a grid step long.
  ed.state.camera.cameraDirection.setAll(0);
  ed.state.camera.cameraRotation.setAll(0);
  ed.state.camera.position.set(10.27, 30, 3.5);
  ed.state.camera.setTarget(new V(10.27, 0, 3.4));
  ed.state.scene.render();
  return { id: e.id, at: e.node.position.asArray().map((v) => +v.toFixed(3)), step, camWas };
});

const grabFrom = await screenOf([10.27, 0, 3.4]);
await page.mouse.move(grabFrom.x, grabFrom.y);
await page.waitForTimeout(200);
await page.evaluate(async (id) => (await import("/js/editor.js")).select([id]), relGrab.id);
await page.keyboard.press("m");
await page.waitForTimeout(400);
const grabbedAt = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const n = ed.hooks.ghostNode();
  return n ? n.position.asArray().map((v) => +v.toFixed(3)) : null;
});
check("a grab starts exactly where the element already was",
  JSON.stringify(grabbedAt) === JSON.stringify(relGrab.at),
  `${JSON.stringify(relGrab.at)} -> ${JSON.stringify(grabbedAt)}`);

// drive the cursor to a point exactly 3 m along X and Z from where it started
const grabTo = await screenOf([13.27, 0, 6.4]);
await page.mouse.move(grabTo.x, grabTo.y, { steps: 6 });
await page.waitForTimeout(300);
const movedTo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const n = ed.hooks.ghostNode();
  return n ? n.position.asArray().map((v) => +v.toFixed(3)) : null;
});
check("and then follows the cursor by exactly that far, offset and all",
  movedTo && Math.abs(movedTo[0] - 13.27) < 0.01 && Math.abs(movedTo[2] - 6.4) < 0.01,
  `${JSON.stringify(relGrab.at)} -> ${JSON.stringify(movedTo)} (wanted 13.27, 6.4)`);
await page.evaluate(async (was) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost();
  ed.removePlacement(was.id);              // only the one this block made
  ed.select([]);
  ed.state.snap.pos = was.step;            // and put the globals back
  ed.state.camera.cameraDirection.setAll(0);
  ed.state.camera.cameraRotation.setAll(0);
  ed.state.camera.position.set(...was.camWas.pos);
  ed.state.camera.setTarget(BABYLON.Vector3.FromArray(was.camWas.target));
}, relGrab);


// Ctrl+D on several: copies, originals untouched
const dupMany = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.select(ids);
  const g = await i.grabSelection({ copy: true });
  const out = {
    carrying: i.ghostCount(), mode: i.ghostMode(),
    originalsOn: ids.map((id) => ed.state.placements.get(id).node.isEnabled()),
  };
  const root = ed.state.scene.getTransformNodeByName("GHOST");
  root.position.set(40, 0, 40);
  await i.dropGhost();
  out.after = ed.state.placements.size;
  out.selected = ed.state.selection.length;
  ed.clearAll(); ed.select([]);
  return out;
}, carrySet.ids);
check("Ctrl+D on a multi-selection carries copies, leaving the originals put",
  dupMany.carrying === 3 && dupMany.mode === "copy"
    && dupMany.originalsOn.join() === "true,true,true",
  `${dupMany.carrying} items, mode ${dupMany.mode}, originals ${dupMany.originalsOn}`);
check("dropping the copies adds them and selects them",
  dupMany.after === 6 && dupMany.selected === 3,
  `${dupMany.after} placements, ${dupMany.selected} selected`);

// ---- 1d-tervicies. the middle button deletes -------------------------------
const mmb = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Platforms/Platform_Simple", new V(0, 0, 0), { silent: true });
  ed.state.camera.position = new V(0, 8, -0.2);
  ed.state.camera.setTarget(new V(0, 0, 0));
  ed.state.camera.cameraDirection.setAll(0);
  return { id: a.id, before: ed.state.placements.size };
});
await page.waitForTimeout(600);
const mmbAt = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const b = ed.screenBoundsOf(ed.state.placements.get([...ed.state.placements.keys()][0]).node);
  const r = ed.state.engine.getRenderingCanvas().getBoundingClientRect();
  return { x: r.x + (b.minX + b.maxX) / 2, y: r.y + (b.minY + b.maxY) / 2 };
});
await page.mouse.move(mmbAt.x, mmbAt.y, { steps: 4 });
await page.waitForTimeout(250);
await page.mouse.click(mmbAt.x, mmbAt.y, { button: "middle" });
await page.waitForTimeout(500);
const mmbAfter = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.placements.size);
check("the middle button deletes what the cursor is over",
  mmb.before === 1 && mmbAfter === 0, `${mmb.before} -> ${mmbAfter} placements`);
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
});

// The ghost rides the build plane, so duplicating something on an upper deck
// has to bring the plane up with it or the copy reappears at ground level.
const dupHigh = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.setGridElevation(0);
  const e = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 12, 0), { silent: true });
  ed.select([e.id]);
  ed.state.camera.position = new V(0, 20, -22);
  ed.state.camera.setTarget(new V(0, 12, 0));
  return { id: e.id, sourceY: e.node.position.y, gridBefore: ed.state.gridY };
});
await page.waitForTimeout(600);
await page.mouse.move(1150, 500, { steps: 3 });
await page.keyboard.press("Control+d");
await page.waitForTimeout(900);
const highGhost = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const n = window.__scene.getTransformNodeByName("GHOST");
  return { gridY: ed.state.gridY, ghostY: n ? n.position.y : null,
           hud: document.getElementById("hud-elev").textContent };
});
check("duplicating raises the build plane to the source's level",
  Math.abs(highGhost.gridY - dupHigh.sourceY) < 1e-6
    && Math.abs(highGhost.ghostY - dupHigh.sourceY) < 1e-6,
  `plane ${dupHigh.gridBefore} -> ${highGhost.gridY} m, ghost at ${highGhost.ghostY} m`);

await page.mouse.down(); await page.mouse.up();
await page.waitForTimeout(800);
const highPlaced = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const list = [...ed.state.placements.values()];
  return list.map((e) => e.node.position.y);
});
check("the copy lands at the same elevation as its source",
  highPlaced.length === 2 && highPlaced.every((y) => Math.abs(y - dupHigh.sourceY) < 1e-6),
  `heights [${highPlaced.map((y) => y.toFixed(2))}]`);
await page.evaluate(async () => {
  (await import("/js/interact.js")).cancelGhost();
  (await import("/js/palette.js")).setBrush(null);
  (await import("/js/editor.js")).setGridElevation(0);
});

// a ghost holds one module, so several selected still duplicate in place
const dupMulti = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost();
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(8, 0, 0), { silent: true });
  ed.select([a.id, b.id]);
  const before = ed.state.placements.size;
  await ed.duplicateSelected();
  return { before, after: ed.state.placements.size, ghost: i.ghostActive() };
});
check("several selected still duplicate in place",
  dupMulti.after === dupMulti.before * 2 && !dupMulti.ghost,
  `${dupMulti.before} -> ${dupMulti.after}, ghost=${dupMulti.ghost}`);
await page.evaluate(async () => {
  (await import("/js/interact.js")).cancelGhost();
  (await import("/js/palette.js")).setBrush(null);
  (await import("/js/editor.js")).clearAll();
});

// ---- 1i-quater. Q flips the drag plane --------------------------------------
// Modular kits are laid out on the horizontal, so dragging across the floor is
// the default; stacking decks and hanging ceiling parts needs the other axis.
const qSetup = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.state.snap.pos = 1;
  ed.state.dragAxis = "xz";
  const e = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.select([]);
  ed.state.camera.position = new V(0, 6, -14);
  ed.state.camera.setTarget(new V(0, 1, 0));
  const bb = ed.worldBounds(e.node);
  return { id: e.id, centre: bb.min.add(bb.max).scale(0.5).asArray(),
           start: e.node.position.asArray(), axis: ed.state.dragAxis };
});
await page.waitForTimeout(600);
const qPt = await screenOf(qSetup.centre);
await page.mouse.move(qPt.x, qPt.y, { steps: 4 });
await page.waitForTimeout(300);

check("the drag plane starts on the floor", qSetup.axis === "xz", qSetup.axis);

// Q cannot be the shortcut: WASD is matched by physical position (so it stays
// under the same fingers on any layout), and on AZERTY the key labelled Q sits
// where QWERTY has A - it arrives as code "KeyA", is claimed as strafe-left and
// never reaches the shortcut switch. V is the same key on both layouts.
const layout = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const fire = (key, code) => {
    ed.state.dragAxis = "xz";
    window.dispatchEvent(new KeyboardEvent("keydown", { key, code, bubbles: true, cancelable: true }));
    // KeyA is strafe-left: without the matching keyup it stays held down and
    // the camera flies off, taking the rest of the suite with it.
    window.dispatchEvent(new KeyboardEvent("keyup", { key, code, bubbles: true, cancelable: true }));
    return ed.state.dragAxis;
  };
  const out = {
    qOnQwerty: fire("q", "KeyQ"),      // label Q at the Q position
    qOnAzerty: fire("q", "KeyA"),      // label Q at the A position
    vOnQwerty: fire("v", "KeyV"),
    vOnAzerty: fire("v", "KeyV"),      // V is the same key on both layouts
  };
  ed.state.dragAxis = "xz";
  ed.releaseAllKeys();
  return out;
});
check("V flips the drag plane on either keyboard layout",
  layout.vOnQwerty === "y" && layout.vOnAzerty === "y",
  `QWERTY=${layout.vOnQwerty}, AZERTY=${layout.vOnAzerty}`);
check("Q is unavailable on AZERTY, where it is the strafe-left key",
  layout.qOnAzerty === "xz",
  `code KeyA left the axis at ${layout.qOnAzerty}`);

// that probe drove the camera a little; put it back before aiming the mouse
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.camera.position = new BABYLON.Vector3(0, 6, -14);
  ed.state.camera.setTarget(new BABYLON.Vector3(0, 1, 0));
  ed.state.camera.cameraDirection.setAll(0);
});
await page.waitForTimeout(500);

await page.keyboard.press("v");
await page.waitForTimeout(200);
const qOn = await page.evaluate(async () => ({
  axis: (await import("/js/editor.js")).state.dragAxis,
  combo: document.getElementById("drag-axis").value,
  options: [...document.getElementById("drag-axis").options].map((o) => o.value),
}));
check("V flips the drag plane to up/down", qOn.axis === "y", qOn.axis);
check("the toolbar combo shows the drag axis, and offers all four",
  qOn.combo === "y" && qOn.options.join() === "xz,y,x,z",
  `"${qOn.combo}" of [${qOn.options}]`);

// Every toolbar control says what its shortcut is, or says it has none. The
// keys are the whole point of the tool - the combos are a readout of modal
// state you are meant to drive from the keyboard - and a control that does not
// mention its key is a key nobody finds.
const tips = await page.evaluate(() => {
  const bar = document.getElementById("toolbar");
  const controls = [...bar.querySelectorAll("button, select, input[type=range], label")];
  const missing = [];
  const bare = [];
  for (const el of controls) {
    // a label wrapping a checkbox is one control with the checkbox, not two
    const t = (el.title || el.closest("label")?.title || "").trim();
    const id = el.id || el.textContent.trim().slice(0, 20);
    if (!t) { missing.push(id); continue; }
    // either it names a key, or it says outright that there is none
    if (!/(Ctrl|Shift|Alt|numpad|\b[A-Z]\b|no shortcut)/i.test(t)) bare.push(id);
  }
  const named = (id) => document.getElementById(id)?.title || "";
  return {
    count: controls.length, missing, bare,
    move: named("snap-pos"),
    moveLabel: [...bar.querySelectorAll("label")].find((l) => l.textContent.trim() === "Move")?.title || "",
    rotAxis: named("rot-axis"),
    scaleAxis: named("scale-axis"),
    dragAxis: named("drag-axis"),
    save: named("btn-save"),
  };
});
check("every toolbar control carries a tooltip",
  tips.missing.length === 0 && tips.count > 20,
  `${tips.count} controls, missing: [${tips.missing}]`);
check("every tooltip names its shortcut, or says there is none",
  tips.bare.length === 0, `no key mentioned on: [${tips.bare}]`);
check("the Move step tooltip names its keys, which were undiscoverable",
  /Shift\+V/.test(tips.move) && /Ctrl\+V/.test(tips.move) && /Shift\+V/.test(tips.moveLabel),
  `"${tips.move}"`);
check("the axis combos name their cycling keys",
  /Shift\+R/.test(tips.rotAxis) && /\bR\b/.test(tips.rotAxis)
    && /Shift\+F/.test(tips.scaleAxis) && /\bF\b/.test(tips.scaleAxis)
    && /\bV\b/.test(tips.dragAxis),
  `rot "${tips.rotAxis}" · scale "${tips.scaleAxis}"`);
check("Save still names Ctrl+S", /Ctrl\+S/.test(tips.save), `"${tips.save}"`);

await page.mouse.down();
await page.mouse.move(qPt.x + 10, qPt.y - 150, { steps: 10 });
await page.waitForTimeout(200);
await page.mouse.up();
await page.waitForTimeout(300);
const qMoved = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  return ed.state.placements.get(id).node.position.asArray();
}, qSetup.id);
check("a drag in Y mode only changes the height",
  qMoved[1] - qSetup.start[1] > 0.5
    && Math.abs(qMoved[0] - qSetup.start[0]) < 1e-6
    && Math.abs(qMoved[2] - qSetup.start[2]) < 1e-6,
  `[${qSetup.start}] -> [${qMoved.map((v) => v.toFixed(2))}]`);

// V cycles rather than toggles now: xz -> y -> x -> z -> xz
const qCycle = [];
for (let k = 0; k < 3; k++) {
  await page.keyboard.press("v");
  await page.waitForTimeout(150);
  qCycle.push(await page.evaluate(async () => ({
    axis: (await import("/js/editor.js")).state.dragAxis,
    combo: document.getElementById("drag-axis").value,
  })));
}
const qBack = qCycle[qCycle.length - 1];
check("V cycles through every drag axis and back to the floor",
  qCycle.map((s) => s.axis).join() === "x,z,xz"
    && qCycle.every((s) => s.combo === s.axis),
  qCycle.map((s) => s.axis).join(" -> "));
check("V flips back to the floor plane, combo and all",
  qBack.axis === "xz" && qBack.combo === "xz", `${qBack.axis}, "${qBack.combo}"`);

const heightNow = qMoved[1];
const qPt2 = await screenOf(await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const bb = ed.worldBounds(ed.state.placements.get(id).node);
  return bb.min.add(bb.max).scale(0.5).asArray();
}, qSetup.id));
await page.mouse.move(qPt2.x, qPt2.y, { steps: 4 });
await page.waitForTimeout(250);
await page.mouse.down();
await page.mouse.move(qPt2.x + 140, qPt2.y + 10, { steps: 10 });
await page.waitForTimeout(200);
await page.mouse.up();
await page.waitForTimeout(300);
const qFloor = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  return ed.state.placements.get(id).node.position.asArray();
}, qSetup.id);
check("a drag in X/Z mode leaves the height alone",
  Math.abs(qFloor[1] - heightNow) < 1e-6
    && Math.hypot(qFloor[0] - qMoved[0], qFloor[2] - qMoved[2]) > 0.5,
  `height held at ${qFloor[1].toFixed(2)}, moved ` +
  `${Math.hypot(qFloor[0] - qMoved[0], qFloor[2] - qMoved[2]).toFixed(2)} m across`);

// ---- single-axis drags: one coordinate moves, the others do not -------------
// A diagonal drag is the test that matters: it would move on both axes in X/Z
// mode, so anything left on the locked axis is the constraint failing.
for (const axis of ["x", "z"]) {
  const before = await page.evaluate(async ([id, a]) => {
    const ed = await import("/js/editor.js");
    const i = await import("/js/interact.js");
    i.setDragAxis(a);
    const bb = ed.worldBounds(ed.state.placements.get(id).node);
    return {
      pos: ed.state.placements.get(id).node.position.asArray(),
      centre: bb.min.add(bb.max).scale(0.5).asArray(),
      combo: document.getElementById("drag-axis").value,
    };
  }, [qSetup.id, axis]);
  const aim = await screenOf(before.centre);
  await page.mouse.move(aim.x, aim.y, { steps: 4 });
  await page.waitForTimeout(200);
  await page.mouse.down();
  await page.mouse.move(aim.x + 150, aim.y + 90, { steps: 12 });   // diagonal
  await page.waitForTimeout(150);
  await page.mouse.up();
  await page.waitForTimeout(300);
  const after = await page.evaluate(async (id) =>
    (await import("/js/editor.js")).state.placements.get(id).node.position.asArray(),
  qSetup.id);
  const d = [0, 1, 2].map((k) => after[k] - before.pos[k]);
  const moved = axis === "x" ? d[0] : d[2];
  const locked = axis === "x" ? d[2] : d[0];
  check(`a drag in ${axis.toUpperCase()}-only mode moves on ${axis.toUpperCase()} alone`,
    Math.abs(moved) > 0.5 && Math.abs(locked) < 1e-6 && Math.abs(d[1]) < 1e-6,
    `delta [${d.map((v) => v.toFixed(2))}]`);
}
await page.evaluate(async () => (await import("/js/interact.js")).setDragAxis("xz"));

// ---- V also constrains the ghost -------------------------------------------
// In Y mode the cursor drives the *build plane*, not the ghost's own height:
// the grid, the ghost and numpad +/- all hang off that one value, so raising it
// keeps them in step and switching back to X/Z resumes at the new height.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.setGridElevation(0); ed.state.snap.pos = 1; ed.state.dragAxis = "xz";
  // well above the height the plane will be raised to: a build plane level with
  // the camera makes the X/Z ray parallel to it, and nothing can track that
  ed.state.camera.position = new BABYLON.Vector3(0, 26, -20);
  ed.state.camera.setTarget(new BABYLON.Vector3(0, 2, 0));
  ed.state.camera.cameraDirection.setAll(0);
  await i.armGhost("Walls/ShortWall_Band2_Straight");
});
await page.waitForTimeout(1200);
const gCv = await page.evaluate(() => {
  const r = window.__scene.getEngine().getRenderingCanvas().getBoundingClientRect();
  return { x: r.x, y: r.y, w: r.width, h: r.height };
});
const ghostAt = () => page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const n = window.__scene.getTransformNodeByName("GHOST");
  return { y: n?.position.y, x: n?.position.x, grid: ed.state.gridY };
});
await page.mouse.move(gCv.x + gCv.w * 0.5, gCv.y + gCv.h * 0.6, { steps: 5 });
await page.waitForTimeout(350);
const gFlat = await ghostAt();

await page.keyboard.press("v");
await page.waitForTimeout(250);
await page.mouse.move(gCv.x + gCv.w * 0.5, gCv.y + gCv.h * 0.25, { steps: 12 });
await page.waitForTimeout(350);
const gUp = await ghostAt();
await page.mouse.move(gCv.x + gCv.w * 0.78, gCv.y + gCv.h * 0.25, { steps: 8 });
await page.waitForTimeout(350);
const gSide = await ghostAt();

await page.keyboard.press("v");
await page.waitForTimeout(250);
await page.mouse.move(gCv.x + gCv.w * 0.35, gCv.y + gCv.h * 0.55, { steps: 10 });
await page.waitForTimeout(350);
const gBack = await ghostAt();

check("V lifts the ghost, and the build plane with it",
  gUp.y > gFlat.y + 0.5 && Math.abs(gUp.grid - gUp.y) < 1e-6,
  `ghost ${gFlat.y} -> ${gUp.y} m, build plane ${gFlat.grid} -> ${gUp.grid} m`);
check("a ghost constrained to Y does not drift sideways",
  Math.abs(gSide.x - gUp.x) < 1e-6, `x ${gUp.x} -> ${gSide.x}`);
check("flipping back to X/Z keeps the height the ghost was raised to",
  Math.abs(gBack.grid - gSide.grid) < 1e-6 && Math.abs(gBack.x - gSide.x) > 0.5,
  `plane held at ${gBack.grid} m, x ${gSide.x} -> ${gBack.x}`);
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.setGridElevation(0); ed.state.dragAxis = "xz";
  ed.clearAll(); ed.select([]);
  (await import("/js/palette.js")).setBrush(null);
});

// ---- 1i-sexies. tall elements drag from any grab point ----------------------
// A drag used to reference a horizontal plane at the element's *origin*, which
// sits at its base. Grab a column near the top from a camera near the floor and
// that plane is behind you (t <= 0), so there was no reference and the piece
// refused to move - only a grab low enough to look *down* at the base worked.
const tall = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.state.snap.pos = 1; ed.state.dragAxis = "xz";
  const e = await ed.placeAt("Columns/Column_Large3", new V(0, 0, 0), { silent: true });
  ed.select([]);
  const bb = ed.worldBounds(e.node);
  const cx = (bb.min.x + bb.max.x) / 2, cz = (bb.min.z + bb.max.z) / 2;
  const grabY = bb.min.y + (bb.max.y - bb.min.y) * 0.75;
  ed.state.camera.position = new V(cx, 1.7, cz - 9);     // eye level
  ed.state.camera.setTarget(new V(cx, grabY, cz));       // looking up at the grab
  ed.state.camera.cameraDirection.setAll(0);
  return { id: e.id, start: e.node.position.asArray(), grab: [cx, grabY, cz],
           height: +(bb.max.y - bb.min.y).toFixed(1) };
});
await page.waitForTimeout(800);
const planes = await page.evaluate(async (g) => {
  const ed = await import("/js/editor.js");
  const s = window.__scene, e = s.getEngine();
  const v = BABYLON.Vector3.Project(BABYLON.Vector3.FromArray(g), BABYLON.Matrix.Identity(),
    s.getTransformMatrix(), s.activeCamera.viewport.toGlobal(e.getRenderWidth(), e.getRenderHeight()));
  s.pointerX = v.x; s.pointerY = v.y;
  return { atBase: !!ed.cursorOnPlane(0), atGrab: !!ed.cursorOnPlane(g[1]) };
}, tall.grab);
check("looking up, the plane at the base is unreachable but the grab point is not",
  planes.atBase === false && planes.atGrab === true,
  `base=${planes.atBase}, grab=${planes.atGrab}`);

const tallPt = await screenOf(tall.grab);
await page.mouse.move(tallPt.x, tallPt.y, { steps: 4 });
await page.waitForTimeout(300);
await page.mouse.down();
await page.mouse.move(tallPt.x + 130, tallPt.y + 10, { steps: 10 });
await page.waitForTimeout(200);
const tallDragging = await page.evaluate(async () =>
  (await import("/js/interact.js")).isDragging());
await page.mouse.up();
await page.waitForTimeout(300);
const tallMoved = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  return ed.state.placements.get(id).node.position.asArray();
}, tall.id);
check("a tall element drags when grabbed near the top from a low camera",
  tallDragging && Math.abs(tallMoved[0] - tall.start[0]) > 0.5,
  `${tall.height} m column moved ${(tallMoved[0] - tall.start[0]).toFixed(2)} m`);

// Direction matters as much as movement. The plane mapping *inverts* once the
// plane is above the camera - looking up, higher on screen is nearer - so
// pushing the mouse forward pulled the piece towards you. An eye-level grab
// therefore uses the screen-space mapping instead.
async function tallDrag(id, grab, dxPx, dyPx) {
  await page.evaluate(async (i) => {
    const ed = await import("/js/editor.js");
    ed.state.placements.get(i).node.position.set(0, 0, 0);
  }, id);
  const at = await screenOf(grab);
  await page.mouse.move(at.x, at.y, { steps: 4 });
  await page.waitForTimeout(250);
  await page.mouse.down();
  await page.mouse.move(at.x + dxPx, at.y + dyPx, { steps: 10 });
  await page.waitForTimeout(180);
  const mode = await page.evaluate(async () => (await import("/js/interact.js")).dragMode());
  await page.mouse.up();
  await page.waitForTimeout(220);
  const pos = await page.evaluate(async (i) => (await import("/js/editor.js"))
    .state.placements.get(i).node.position.asArray(), id);
  return { mode, pos };
}
// camera sits at -z looking towards +z, so "away" is +z
const pushed = await tallDrag(tall.id, tall.grab, 0, -120);
const pulled = await tallDrag(tall.id, tall.grab, 0, 120);
const sided = await tallDrag(tall.id, tall.grab, 150, 0);
check("an eye-level grab uses the screen-space mapping",
  pushed.mode === "screen", `mode=${pushed.mode}`);
check("mouse away pushes the element away, and back pulls it back",
  pushed.pos[2] > 0.5 && pulled.pos[2] < -0.5,
  `up -> z=${pushed.pos[2].toFixed(2)}, down -> z=${pulled.pos[2].toFixed(2)}`);
check("mouse right moves the element right",
  sided.pos[0] > 0.5, `x=${sided.pos[0].toFixed(2)}`);

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
});

// ---- 1i-quinquies. selection and hover are mutually exclusive ---------------
// One element, one state: otherwise the outline colour and "what will this key
// act on?" can disagree about the same element.
const exclusive = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const e = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.state.camera.position = new V(0, 10, -10);
  ed.state.camera.setTarget(new V(0, 1, 0));
  ed.state.camera.cameraDirection.setAll(0);
  return { id: e.id, hoveredBefore: i.hoveredId() };
});
await page.waitForTimeout(700);
const exclusiveId = exclusive.id;
const stillPt = await screenOf(await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const bb = ed.worldBounds(ed.state.placements.get(id).node);
  return bb.min.add(bb.max).scale(0.5).asArray();
}, exclusiveId));
await page.mouse.move(stillPt.x, stillPt.y, { steps: 4 });
await page.waitForTimeout(350);
const hoveredUnselected = await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  return { hovered: i.hoveredId(), kind: i.currentElement()?.kind };
});
check("an unselected element under the cursor is hovered",
  hoveredUnselected.hovered === exclusive.id && hoveredUnselected.kind === "hover",
  `hovered=${hoveredUnselected.hovered}, current=${hoveredUnselected.kind}`);

const afterSelect = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.select([id]);                       // select the very element under the cursor
  return { hovered: i.hoveredId(), kind: i.currentElement()?.kind,
           selection: ed.state.selection.length };
}, exclusiveId);
check("selecting the hovered element drops the hover at once",
  afterSelect.hovered === null && afterSelect.kind === "selection"
    && afterSelect.selection === 1,
  `hovered=${afterSelect.hovered}, current=${afterSelect.kind}`);

// and it stays carryDropped when the pointer moves over it again
await page.mouse.move(stillPt.x + 30, stillPt.y + 20, { steps: 3 });
await page.mouse.move(stillPt.x, stillPt.y, { steps: 3 });
await page.waitForTimeout(350);
const reHover = await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  return { hovered: i.hoveredId(), kind: i.currentElement()?.kind };
});
check("a selected element is never re-hovered",
  reHover.hovered === null && reHover.kind === "selection",
  `hovered=${reHover.hovered}, current=${reHover.kind}`);

// ---- 1i-septies. no hover while driving the camera -------------------------
// Looking around sweeps the cursor over the whole scene, so hovering would
// strobe the outline and keep changing what the next key would act on.
const rmbHover = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const e = await ed.placeAt("Platforms/Platform_3Plates", new V(0, 0, 0), { silent: true });
  ed.select([]);
  ed.state.camera.position = new V(0, 10, -10);
  ed.state.camera.setTarget(new V(0, 0, 0));
  ed.state.camera.cameraDirection.setAll(0);
  const bb = ed.worldBounds(e.node);
  return { id: e.id, centre: bb.min.add(bb.max).scale(0.5).asArray() };
});
await page.waitForTimeout(700);
const rmbPt = await screenOf(rmbHover.centre);
const hoveredNow = async () => page.evaluate(async () =>
  (await import("/js/interact.js")).hoveredId());

await page.mouse.move(rmbPt.x, rmbPt.y, { steps: 4 });
await page.waitForTimeout(350);
const rmbBefore = await hoveredNow();
await page.mouse.down({ button: "right" });
await page.waitForTimeout(150);
const rmbOnPress = await hoveredNow();
await page.mouse.move(rmbPt.x + 12, rmbPt.y + 8, { steps: 4 });   // still on it
await page.waitForTimeout(250);
const rmbHeld = await hoveredNow();
await page.mouse.up({ button: "right" });
await page.waitForTimeout(350);
const rmbAfter = await hoveredNow();

check("pressing the right button drops the hover at once",
  rmbBefore === rmbHover.id && rmbOnPress === null,
  `${rmbBefore} -> ${rmbOnPress}`);
check("moving over an element with the right button held does not hover it",
  rmbHeld === null, `hovered=${rmbHeld}`);
check("releasing the right button brings the hover back without a mouse move",
  rmbAfter === rmbHover.id, `hovered=${rmbAfter}`);

// ---- 1i-octies. cursor hidden while driving, and rectangle select ----------
const rmbCursor = await page.evaluate(() =>
  window.__scene.getEngine().getRenderingCanvas().style.cursor);
await page.mouse.down({ button: "right" });
await page.waitForTimeout(200);
const heldCursor = await page.evaluate(() =>
  window.__scene.getEngine().getRenderingCanvas().style.cursor);
await page.mouse.up({ button: "right" });
await page.waitForTimeout(250);
const freeCursor = await page.evaluate(() =>
  window.__scene.getEngine().getRenderingCanvas().style.cursor);
check("the cursor is hidden while the right button is held",
  rmbCursor !== "none" && heldCursor === "none" && freeCursor !== "none",
  `"${rmbCursor}" -> "${heldCursor}" -> "${freeCursor}"`);

// ---- 1i-nonies. no context menu anywhere in the editor ---------------------
// The right button is a camera control, so the menu is never wanted. It is also
// not enough to guard the canvas: contextmenu arrives after pointerup, aimed at
// whatever the cursor drifted over rather than at the pointer-capturing canvas,
// so a look that ended on a palette thumbnail used to raise "Save image as".
await page.evaluate(() => {
  window.__menus = [];
  // deferred so the read happens once every other listener has had its say
  addEventListener("contextmenu", (e) => setTimeout(() => window.__menus.push({
    target: e.target.tagName + (e.target.id ? `#${e.target.id}` : ""),
    prevented: e.defaultPrevented,
  }), 0), true);
});
const menuPts = await page.evaluate(() => {
  const box = (sel) => {
    const r = document.querySelector(sel).getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  };
  return {
    mid: box("#render-canvas"),
    search: box("#palette-search"),
    tile: box("#palette-list .item"),
    toolbar: { x: box("#toolbar").x, y: box("#toolbar").y },
    inspector: box("#inspector"),
  };
});

await page.mouse.move(menuPts.mid.x, menuPts.mid.y, { steps: 3 });
await page.mouse.down({ button: "right" });
await page.mouse.move(menuPts.mid.x + 30, menuPts.mid.y + 15, { steps: 4 });
await page.mouse.up({ button: "right" });
await page.waitForTimeout(200);

// the release that actually bites: outside the canvas, over the palette
await page.mouse.move(menuPts.mid.x, menuPts.mid.y, { steps: 3 });
await page.mouse.down({ button: "right" });
await page.mouse.move(menuPts.tile.x, menuPts.tile.y, { steps: 8 });
await page.mouse.up({ button: "right" });
await page.waitForTimeout(200);

// and every other area, including a plain right-click that never went near the
// canvas: Ctrl+C/Ctrl+V still work, only the menu is gone
for (const at of [menuPts.search, menuPts.toolbar, menuPts.inspector]) {
  await page.mouse.click(at.x, at.y, { button: "right" });
  await page.waitForTimeout(150);
}

const menus = await page.evaluate(() => window.__menus);
check("a right-drag released on the canvas raises no menu",
  menus[0]?.prevented === true, JSON.stringify(menus[0]));
check("a right-drag released off the canvas raises no menu",
  menus[1] && menus[1].target !== "CANVAS#render-canvas" && menus[1].prevented === true,
  JSON.stringify(menus[1]));
check("no editor area raises a menu, text fields included",
  menus.length === 5 && menus.every((m) => m.prevented === true),
  JSON.stringify(menus.map((m) => `${m.target}:${m.prevented}`)));

// the loading overlay is the one place there is no scene to hang a handler off
const busyMenu = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  window.__menus.length = 0;
  let release;
  const gate = new Promise((r) => { release = r; });
  const done = ed.whileBusy("loading ship…", () => gate);
  await new Promise((r) => setTimeout(r, 50));
  const overlay = document.getElementById("busy").getBoundingClientRect();
  document.getElementById("busy").dispatchEvent(new MouseEvent("contextmenu", {
    bubbles: true, cancelable: true,
    clientX: overlay.x + overlay.width / 2, clientY: overlay.y + overlay.height / 2,
  }));
  release();
  await done;
  await new Promise((r) => setTimeout(r, 20));
  return window.__menus;
});
check("the loading overlay raises no menu either",
  busyMenu.length === 1 && busyMenu[0].prevented === true && busyMenu[0].target === "DIV#busy",
  JSON.stringify(busyMenu));
await page.evaluate(() => document.querySelector("#palette-search").blur());
await page.keyboard.press("Escape");

const rectIds = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const made = [];
  for (let k = 0; k < 4; k++) {
    made.push((await ed.placeAt("Walls/ShortWall_Band2_Straight",
      new V(k * 6, 0, 0), { silent: true })).id);
  }
  ed.select([]);
  ed.state.camera.position = new V(9, 16, -16);
  ed.state.camera.setTarget(new V(9, 0, 0));
  ed.state.camera.cameraDirection.setAll(0);
  return made;
});
await page.waitForTimeout(700);
const cv = await page.evaluate(() => {
  const r = window.__scene.getEngine().getRenderingCanvas().getBoundingClientRect();
  return { x: r.x, y: r.y, w: r.width, h: r.height };
});

// a drag from empty space is always a rectangle - nothing else it could mean
await page.mouse.move(cv.x + 20, cv.y + 20, { steps: 3 });
await page.mouse.down();
await page.mouse.move(cv.x + cv.w * 0.52, cv.y + cv.h * 0.9, { steps: 12 });
await page.waitForTimeout(220);
const banding = await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  const el = document.querySelector("#viewport .marquee");
  return { marqueeing: i.isMarqueeing(), visible: !!el };
});
await page.mouse.up();
await page.waitForTimeout(350);
const banded = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { sel: [...ed.state.selection], leftover: !!document.querySelector("#viewport .marquee") };
});
check("dragging from empty space rubber-bands a selection",
  banding.marqueeing && banding.visible && banded.sel.length === 2
    && banded.sel[0] === rectIds[0] && !banded.leftover,
  `selected [${banded.sel}] of ${rectIds.length}, overlay cleaned up=${!banded.leftover}`);

// Ctrl adds to what is already selected
await page.keyboard.down("Control");
await page.mouse.move(cv.x + cv.w * 0.55, cv.y + 20, { steps: 3 });
await page.mouse.down();
await page.mouse.move(cv.x + cv.w * 0.98, cv.y + cv.h * 0.9, { steps: 12 });
await page.mouse.up();
await page.keyboard.up("Control");
await page.waitForTimeout(350);
const added = await page.evaluate(async () =>
  [...(await import("/js/editor.js")).state.selection]);
check("Ctrl + rectangle adds to the selection rather than replacing it",
  added.length > banded.sel.length && banded.sel.every((id) => added.includes(id)),
  `[${banded.sel}] -> [${added}]`);

// with the mode on, a rectangle can start on top of a module
const modeOn = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.select([]);
  document.getElementById("btn-select-rect").click();
  const b = ed.screenBoundsOf(ed.state.placements.get([...ed.state.placements.keys()][0]).node);
  return { on: ed.state.selectMode, box: b };
});
const startOn = { x: cv.x + (modeOn.box.minX + modeOn.box.maxX) / 2,
                  y: cv.y + (modeOn.box.minY + modeOn.box.maxY) / 2 };
await page.mouse.move(startOn.x, startOn.y, { steps: 3 });
await page.mouse.down();
await page.mouse.move(cv.x + cv.w * 0.55, cv.y + cv.h * 0.9, { steps: 12 });
await page.waitForTimeout(200);
const onModule = await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  return { marqueeing: i.isMarqueeing(), dragging: i.isDragging() };
});
await page.mouse.up();
await page.waitForTimeout(350);
const modeSel = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const moved = ed.state.placements.get([...ed.state.placements.keys()][0]).node.position.x;
  document.getElementById("btn-select-rect").click();
  return { sel: ed.state.selection.length, firstX: moved, off: ed.state.selectMode };
});
check("Rect select mode bands instead of moving the module under the cursor",
  modeOn.on && onModule.marqueeing && !onModule.dragging
    && modeSel.sel > 0 && Math.abs(modeSel.firstX) < 1e-6,
  `marqueeing=${onModule.marqueeing}, dragging=${onModule.dragging}, ` +
  `${modeSel.sel} selected, module still at x=${modeSel.firstX}`);
check("the toggle turns back off", modeSel.off === false);

// Esc abandons a rectangle in progress
await page.evaluate(async () => (await import("/js/editor.js")).select([]));
await page.mouse.move(cv.x + 20, cv.y + 20, { steps: 3 });
await page.mouse.down();
await page.mouse.move(cv.x + cv.w * 0.6, cv.y + cv.h * 0.9, { steps: 8 });
await page.waitForTimeout(180);
await page.keyboard.press("Escape");
await page.waitForTimeout(250);
const escaped = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { sel: ed.state.selection.length, leftover: !!document.querySelector("#viewport .marquee") };
});
await page.mouse.up();
await page.waitForTimeout(200);
check("Esc abandons the rectangle and selects nothing",
  escaped.sel === 0 && !escaped.leftover,
  `${escaped.sel} selected, overlay left=${escaped.leftover}`);

// ---- 1i-decies. a rectangle catches markers too ----------------------------
// The player spawn is an element you select, drag and delete like any other,
// ---- 1i-decies. a rectangle catches markers too ----------------------------
// A door marker is an element you select, drag and delete like any other, so a
// band drawn round it has to pick it up. It lives in state.markers rather than
// state.placements, which is exactly what the rectangle used to miss.
const spawnRect = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mk = await import("/js/markers.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const wall = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  mk.addDoor(new V(8, 0, 0), { silent: true });
  ed.state.camera.position = new V(4, 12, -14);
  ed.state.camera.setTarget(new V(4, 1, 0));
  ed.state.camera.cameraDirection.setAll(0);
  return { wall: wall.id };
});
await page.waitForTimeout(700);
await page.mouse.move(cv.x + 8, cv.y + 8, { steps: 3 });
await page.mouse.down();
await page.mouse.move(cv.x + cv.w - 8, cv.y + cv.h - 8, { steps: 12 });
await page.mouse.up();
await page.waitForTimeout(350);
const caught = await page.evaluate(async () =>
  [...(await import("/js/editor.js")).state.selection]);
check("a rectangle selects markers, not just placements",
  caught.some((id) => id.startsWith("Door_")) && caught.includes(spawnRect.wall)
    && caught.length === 2,
  `[${caught}]`);

// ---- 1i-undecies. Shift+H hides the selection, H brings it back ------------
const hiding = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const made = [];
  for (let k = 0; k < 3; k++) {
    made.push((await ed.placeAt("Walls/ShortWall_Band2_Straight",
      new V(k * 6, 0, 0), { silent: true })).id);
  }
  ed.select([made[0], made[1]]);
  return { ids: made };
});

// hiding is the shifted one on purpose: a bare H must never cost you a
// selection you cannot see any more
await page.keyboard.press("h");
await page.waitForTimeout(200);
const bareH = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  return {
    enabled: ids.map((id) => ed.state.placements.get(id).node.isEnabled()),
    count: ed.hiddenCount(),
    sel: ed.state.selection.length,
  };
}, hiding.ids);
check("plain H cannot hide anything",
  bareH.enabled.every(Boolean) && bareH.count === 0 && bareH.sel === 2,
  `${bareH.count} hidden, ${bareH.sel} still selected`);

await page.keyboard.press("Shift+H");
await page.waitForTimeout(250);
const ghosted = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const parts = (id) => {
    const kids = ed.state.placements.get(id).node.getChildMeshes();
    return {
      real: kids.filter((m) => !ed.isVeilClone(m)),
      veil: kids.filter((m) => ed.isVeilClone(m)),
    };
  };
  const a = parts(ids[0]), c = parts(ids[2]);
  return {
    enabled: ids.map((id) => ed.state.placements.get(id).node.isEnabled()),
    veilCount: ids.map((id) => parts(id).veil.length),
    realOn: ids.map((id) => parts(id).real.some((m) => m.isEnabled())),
    alpha: a.veil[0]?.material?.alpha,
    veilPickable: a.veil.some((m) => m.isPickable),
    counts: document.getElementById("status-counts").textContent,
    sel: [...ed.state.selection],
    // THE regression: the shared kit material must not be touched. Forcing it
    // to ALPHABLEND moved every mesh drawn with it into the transparent pass,
    // where nothing writes depth, and half the ship stopped occluding.
    sharedMat: a.real[0]?.sourceMesh?.material?.transparencyMode ?? null,
    sharedBlends: a.real[0]?.sourceMesh?.material
      ?.needAlphaBlendingForMesh(c.real[0]) ?? null,
    otherVeiled: c.veil.length,
  };
}, hiding.ids);
check("the first Shift+H ghosts rather than hides",
  ghosted.enabled.every(Boolean) && ghosted.veilCount.join() === "2,2,0"
    && ghosted.realOn.join() === "false,false,true",
  `enabled=${JSON.stringify(ghosted.enabled)}, veils=${ghosted.veilCount}, real=${ghosted.realOn}`);
check("the stand-in carries the veil alpha",
  Math.abs(ghosted.alpha - 0.5) < 1e-6, `alpha=${ghosted.alpha}`);
check("a ghosted element is click-through, so you can reach what is behind it",
  ghosted.veilPickable === false, `veil pickable=${ghosted.veilPickable}`);
check("ghosting one instance leaves the shared material alone, so depth still writes",
  ghosted.sharedMat !== 2 && ghosted.sharedBlends === false && ghosted.otherVeiled === 0,
  `transparencyMode=${ghosted.sharedMat}, blends=${ghosted.sharedBlends}, other veils=${ghosted.otherVeiled}`);
check("the selection survives, or the cycle could not be driven",
  ghosted.sel.length === 2, `[${ghosted.sel}]`);
check("the status bar says what is at 50%",
  /2 at 50%/.test(ghosted.counts), ghosted.counts);

await page.keyboard.press("Shift+H");
await page.waitForTimeout(250);
const hid = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const wide = { x0: -1e5, y0: -1e5, x1: 1e5, y1: 1e5 };
  return {
    enabled: ids.map((id) => ed.state.placements.get(id).node.isEnabled()),
    count: ed.hiddenCount(),
    sel: [...ed.state.selection],
    banded: ed.elementsInRect(wide),
    counts: document.getElementById("status-counts").textContent,
  };
}, hiding.ids);
check("the second Shift+H hides outright",
  hid.enabled[0] === false && hid.enabled[1] === false && hid.enabled[2] === true
    && hid.count === 2,
  `enabled=${JSON.stringify(hid.enabled)}, ${hid.count} hidden`);
check("a hidden element is out of reach of the rectangle",
  hid.banded.length === 1 && hid.banded[0] === hiding.ids[2], `[${hid.banded}]`);
check("the hidden count is on the status bar",
  /2 hidden/.test(hid.counts), hid.counts);

// the cycle is two-state: H is the only way back to fully opaque
await page.keyboard.press("Shift+H");
await page.waitForTimeout(250);
const veilCycled = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  return {
    enabled: ids.map((id) => ed.state.placements.get(id).node.isEnabled()),
    veils: ids.map((id) => ed.state.placements.get(id).node.getChildMeshes()
      .filter((m) => ed.isVeilClone(m)).length),
  };
}, hiding.ids);
check("a third Shift+H returns to 50% rather than to fully opaque",
  veilCycled.enabled.every(Boolean) && veilCycled.veils.join() === "2,2,0",
  `enabled=${JSON.stringify(veilCycled.enabled)}, veils=${veilCycled.veils}`);

// back to hidden for the isolation and export checks below
await page.keyboard.press("Shift+H");
await page.waitForTimeout(250);

// isolation and hiding both work by disabling nodes: neither may clobber the
// other, or isolating a chunk would silently reveal what you had hidden
const withIsolate = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  ed.addChunk("CH_Other");
  ed.state.placements.get(ids[2]).chunk = "CH_Other";   // so isolation has work to do
  ed.state.isolate = true;
  ed.applyVisibility();
  const isolated = ids.map((id) => ed.state.placements.get(id).node.isEnabled());
  ed.state.isolate = false;
  ed.applyVisibility();
  return { isolated, after: ids.map((id) => ed.state.placements.get(id).node.isEnabled()) };
}, hiding.ids);
check("isolating a chunk hides the other chunks without revealing what H hid",
  withIsolate.isolated.join() === "false,false,false"
    && withIsolate.after.join() === "false,false,true",
  `isolated=${JSON.stringify(withIsolate.isolated)}, after=${JSON.stringify(withIsolate.after)}`);

// hiding is a viewport aid, not an edit: the ship on disk must be unchanged
const hiddenExport = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  // two hidden and one ghosted, so the export meets both kinds of veil
  ed.select([ids[2]]); ed.hideSelected(); ed.select([]);
  const veilsBefore = ed.state.placements.get(ids[2]).node.getChildMeshes()
    .filter((m) => ed.isVeilClone(m)).length;
  const realFetch = window.fetch;
  let body = null;
  window.fetch = (url, opts) => {
    if (String(url).includes("/api/export")) {
      body = opts.body;
      return Promise.resolve(new Response('{"ok":true,"bytes":0}',
        { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
  try { await mf.exportGlb(); } finally { window.fetch = realFetch; }
  const buf = await body.arrayBuffer();
  const dv = new DataView(buf);
  const json = JSON.parse(new TextDecoder()
    .decode(new Uint8Array(buf, 20, dv.getUint32(12, true))));
  return {
    nodes: (json.nodes || []).map((n) => n.name),
    instances: mf.buildManifest().instances.length,
    hidden: ids.slice(0, 2),
    // no "__veil" node, and the stand-ins are back afterwards
    veilNodes: (json.nodes || []).map((n) => n.name).filter((n) => /__veil/.test(n)),
    mats: (json.materials || []).map((m) => m.name).filter((n) => /^VEIL_|^GHOST_/.test(n)),
    meshCount: mf.buildManifest().chunks.reduce((a, c) => a + c.meshCount, 0),
    veilsBefore,
    // suspending the veil for the export must not leave it suspended
    veilsAfter: ed.state.placements.get(ids[2]).node.getChildMeshes()
      .filter((m) => ed.isVeilClone(m)).length,
  };
}, hiding.ids);
check("a hidden element is still exported and still in the manifest",
  hiddenExport.hidden.every((id) => hiddenExport.nodes.includes(id))
    && hiddenExport.instances === 3,
  `${hiddenExport.instances} instances, nodes ${JSON.stringify(hiddenExport.nodes)}`);
check("the translucent stand-ins reach neither the .glb nor the manifest",
  hiddenExport.veilsBefore === 2 && hiddenExport.veilNodes.length === 0
    && hiddenExport.mats.length === 0 && hiddenExport.meshCount === 6,
  `${hiddenExport.veilsBefore} stand-ins, veil nodes ${JSON.stringify(hiddenExport.veilNodes)}, mats ${JSON.stringify(hiddenExport.mats)}, meshCount ${hiddenExport.meshCount}`);
check("the veil comes back after the export",
  hiddenExport.veilsAfter === 2, `${hiddenExport.veilsAfter} stand-ins`);

await page.keyboard.press("h");
await page.waitForTimeout(250);
const unhid = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  return {
    enabled: ids.map((id) => ed.state.placements.get(id).node.isEnabled()),
    count: ed.hiddenCount(),
    counts: document.getElementById("status-counts").textContent,
  };
}, hiding.ids);
check("H brings everything back",
  unhid.enabled.every(Boolean) && unhid.count === 0 && !/hidden/.test(unhid.counts),
  `enabled=${JSON.stringify(unhid.enabled)}, ${unhid.count} hidden`);

// Doors were excluded from both scale paths: scaleCurrent() filtered markers
// out, and applyInspector() guarded the scale write with `if (!e.type)`.
const doorScale = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const i = await import("/js/interact.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const d = mk.addDoor(new V(8, 0, 0), { silent: true });
  ed.select([d.id]);

  ed.state.scaleAxis = "all";
  const before = d.node.scaling.x;
  i.scaleCurrent(1);
  const wheeled = +d.node.scaling.x.toFixed(3);

  d.node.scaling.set(2, 3, 1);
  const built = mf.buildManifest();
  const rec = built.doors.find((x) => x.id === d.id);
  const p = built.portals.find((x) => x.id === `Portal_${d.id}`);
  const span = Math.hypot(p.corners[1][0] - p.corners[0][0], p.corners[1][2] - p.corners[0][2]);
  const tall = p.corners[2][1] - p.corners[1][1];

  const layout = ed.serialize();
  await ed.deserialize(JSON.parse(JSON.stringify(layout)));
  const reloaded = [...ed.state.markers.values()][0]?.node.scaling.asArray();

  ed.clearAll(); ed.select([]);
  ed.state.scaleAxis = "all";
  return { before, wheeled, authored: [d.width, d.height],
           exported: [rec.width, rec.height],
           span: +span.toFixed(3), tall: +tall.toFixed(3), reloaded };
});
check("Shift+wheel scales a door",
  doorScale.before === 1 && doorScale.wheeled === 1.1,
  `${doorScale.before} -> ${doorScale.wheeled}`);
check("a scaled door exports its effective width and height",
  Math.abs(doorScale.exported[0] - doorScale.authored[0] * 2) < 1e-3
  && Math.abs(doorScale.exported[1] - doorScale.authored[1] * 3) < 1e-3,
  `authored ${doorScale.authored} x (2,3) -> exported ${doorScale.exported}`);
check("the exported portal agrees with the exported door size",
  Math.abs(doorScale.span - doorScale.exported[0]) < 1e-3
  && Math.abs(doorScale.tall - doorScale.exported[1]) < 1e-3,
  `portal ${doorScale.span} x ${doorScale.tall} vs door ${doorScale.exported}`);
check("a door's scale survives a reload",
  doorScale.reloaded?.join() === "2,3,1", `${doorScale.reloaded}`);

// the inspector's scale fields, driven through the real DOM on a door
const doorInsp = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const d = mk.addDoor(new V(8, 0, 0), { silent: true });
  ed.select([d.id]);
  return { id: d.id, shown: document.getElementById("scl-x").parentElement
    .getBoundingClientRect().height > 0 };
});
await page.fill("#scl-x", "2.5");
await page.waitForTimeout(250);
const doorInspOut = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const out = [...ed.state.markers.values()][0].node.scaling.asArray();
  ed.clearAll(); ed.select([]);
  return out;
});
check("the inspector scale fields work on a door",
  doorInsp.shown && Math.abs(doorInspOut[0] - 2.5) < 1e-6,
  `shown=${doorInsp.shown}, scale=${doorInspOut}`);

// the Ghost slider drives the veil live, on stand-ins that already exist
const veilSlider = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.select([a.id]);
  ed.hideSelected();                       // ghost, with the slider at its default
  const veilOf = () => a.node.getChildMeshes().find((m) => ed.isVeilClone(m));
  return { id: a.id, before: veilOf()?.material?.alpha };
});
await page.fill("#veil-alpha", "0.2");
await page.dispatchEvent("#veil-alpha", "input");
await page.waitForTimeout(200);
const veilSliderOut = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const node = ed.state.placements.get(id).node;
  const out = {
    after: node.getChildMeshes().find((m) => ed.isVeilClone(m))?.material?.alpha,
    readout: document.getElementById("veil-alpha-val").textContent,
    counts: document.getElementById("status-counts").textContent,
    stored: localStorage.getItem("veilAlpha"),
  };
  ed.unhideAll(); ed.clearAll(); ed.select([]);
  ed.setVeilAlpha(0.5);
  document.getElementById("veil-alpha").value = "0.5";
  document.getElementById("veil-alpha-val").textContent = "50%";
  localStorage.setItem("veilAlpha", "0.5");
  return out;
}, veilSlider.id);
check("the Ghost slider repaints stand-ins that are already on screen",
  Math.abs(veilSlider.before - 0.5) < 1e-6 && Math.abs(veilSliderOut.after - 0.2) < 1e-6,
  `${veilSlider.before} -> ${veilSliderOut.after}`);
check("the slider readout, the status bar and localStorage all follow",
  veilSliderOut.readout === "20%" && /1 at 20%/.test(veilSliderOut.counts)
    && veilSliderOut.stored === "0.2",
  `readout "${veilSliderOut.readout}", stored ${veilSliderOut.stored}, counts "${veilSliderOut.counts}"`);

// ids restart at P0001 after a clear, so a leftover entry would hide a new one
const afterClear = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const one = (await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true })).id;
  ed.select([one]);
  ed.hideSelected();
  ed.clearAll();
  const fresh = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const out = { one, fresh: fresh.id, enabled: fresh.node.isEnabled(), count: ed.hiddenCount() };
  ed.clearAll(); ed.select([]);
  return out;
});
check("clearing forgets what was hidden, so recycled ids stay visible",
  afterClear.fresh === afterClear.one && afterClear.enabled && afterClear.count === 0,
  `${afterClear.one} reused as ${afterClear.fresh}, visible=${afterClear.enabled}`);

// Undo used to unhide as a side effect (restoreFrom clears the set) with no way
// back, because `hidden` was not in the snapshot. Both directions now work.
const hideUndo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const a = (await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true })).id;
  const b = (await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(4, 0, 0), { silent: true })).id;
  const on = () => [a, b].map((id) => ed.state.placements.get(id).node.isEnabled());
  ed.select([a]);
  ed.hideSelected(); ed.hideSelected();       // ghost, then hidden
  const hidden = on();
  await ed.undo();
  const undone = on();
  await ed.redo();
  const redone = on();
  // the ghost level has to survive the round trip too, not just "hidden"
  await ed.undo(); await ed.undo();
  ed.select([a]); ed.hideSelected();          // ghost
  const ghostLevel = ed.state.hidden.get(a);
  await ed.undo();
  const afterGhostUndo = ed.state.hidden.get(a);
  await ed.redo();
  const afterGhostRedo = ed.state.hidden.get(a);
  const out = { hidden, undone, redone, count: ed.hiddenCount(),
                ghostLevel, afterGhostUndo, afterGhostRedo };
  ed.clearAll(); ed.select([]);
  return out;
});
check("undo takes back a hide and redo puts it back",
  hideUndo.hidden.join() === "false,true" && hideUndo.undone.join() === "true,true"
  && hideUndo.redone.join() === "false,true",
  `hidden=${hideUndo.hidden} undo=${hideUndo.undone} redo=${hideUndo.redone}`);
check("the ghost level survives undo and redo, not just the hidden one",
  hideUndo.ghostLevel === "ghost" && hideUndo.afterGhostUndo === undefined
  && hideUndo.afterGhostRedo === "ghost",
  `${hideUndo.ghostLevel} -> undo ${hideUndo.afterGhostUndo} -> redo ${hideUndo.afterGhostRedo}`);

// The old bug the other way round: undoing anything at all revealed the lot.
const hideKept = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const a = (await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true })).id;
  ed.select([a]);
  ed.hideSelected(); ed.hideSelected();       // ghost, then hidden
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(4, 0, 0));   // not silent: pushes undo
  await ed.undo();                             // takes back the placement, not the hide
  const out = {
    enabled: ed.state.placements.get(a).node.isEnabled(),
    count: ed.hiddenCount(),
    placements: ed.state.placements.size,
  };
  ed.clearAll(); ed.select([]);
  return out;
});
check("an unrelated undo leaves hidden elements hidden",
  hideKept.enabled === false && hideKept.count === 1 && hideKept.placements === 1,
  `enabled=${hideKept.enabled}, ${hideKept.count} hidden, ${hideKept.placements} placements`);

// The cap used to be 80 entries; it is now a memory budget with a count rail,
// so both ends need checking - and a snapshot too big for the budget on its own
// must still be kept, or undo would do nothing at all on a huge ship.
const histCap = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const limits = ed.historyLimits();
  await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const one = JSON.stringify(ed.serialize()).length;

  const before = ed.historyDepth().undo;
  for (let i = 0; i < 200; i++) ed.pushUndo();          // well past the old 80
  const past80 = ed.historyDepth().undo;

  for (let i = 0; i < limits.entries + 50; i++) ed.pushUndo();
  const capped = ed.historyDepth().undo;

  // The byte path is unreachable from the UI on a ship small enough to build
  // in a test - 1000 snapshots of this one is ~0.3 M chars against a 32 M
  // budget - so the trim is handed a synthetic stack instead.
  const big = "x".repeat(Math.ceil(limits.chars / 4));
  const byBytes = [big, big, big, big, big, big];         // 6 x quarter-budget
  ed.trimHistory(byBytes);
  const huge = ["y".repeat(limits.chars * 3)];            // one, over budget
  ed.trimHistory(huge);

  const out = { limits, one, before, past80, capped,
                byBytes: byBytes.length, huge: huge.length };
  ed.clearAll(); ed.select([]);
  return out;
});
check("history goes far deeper than the old 80 entries",
  histCap.past80 === Math.min(histCap.before + 200, histCap.limits.entries)
  && histCap.past80 > 80 && histCap.capped === histCap.limits.entries,
  `${histCap.before} + 200 -> ${histCap.past80} deep, then capped at ${histCap.capped}`);
check("the budget is generous for a ship this size",
  histCap.one * histCap.limits.entries < histCap.limits.chars,
  `${histCap.one} chars x ${histCap.limits.entries} < ${histCap.limits.chars} budget`);
check("a fat stack is trimmed by memory, not by count",
  histCap.byBytes === 4, `6 quarter-budget entries -> ${histCap.byBytes} kept`);
check("a single snapshot over the whole budget is still kept",
  histCap.huge === 1, `${histCap.huge} kept`);

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
});
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.select([]); ed.clearAll();
});
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.select([]); ed.clearAll(); ed.state.dragAxis = "xz";
});
check("the camera input claims the right button only", await page.evaluate(async () => {
  const b = (await import("/js/editor.js")).state.camera.inputs.attached.mouse?.buttons || [];
  return b.length === 1 && b[0] === 2;
}));

// ---- 1i-duodecies. reaching the panel behind a portal ----------------------
// A portal is a flat quad sitting exactly where the door panels are, and it
// turns to face the camera, so no amount of flying gets you behind it. Shift
// used to make it transparent to picking; Shift+H on the door does that now,
// for anything and not only for doors.
const portalSetup = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mk = await import("/js/markers.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const pane = await ed.placeAt("Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.select([pane.id]);
  const door = mk.doorFromSelection();
  ed.select([]);
  // straight at the door, so the portal quad is squarely between us and it
  ed.state.camera.position = new V(0, 1.5, -9);
  ed.state.camera.setTarget(new V(0, 1.5, 0));
  ed.state.camera.cameraDirection.setAll(0);
  return { pane: pane.id, door: door?.id };
});
await page.waitForTimeout(700);
const doorAim = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const b = ed.screenBoundsOf(ed.state.markers.get(ids.door).node);
  const r = ed.state.engine.getRenderingCanvas().getBoundingClientRect();
  return { x: r.x + (b.minX + b.maxX) / 2, y: r.y + (b.minY + b.maxY) / 2 };
}, portalSetup);

await page.mouse.move(doorAim.x, doorAim.y, { steps: 5 });
await page.waitForTimeout(200);
const plainHover = await page.evaluate(async () =>
  (await import("/js/interact.js")).hoveredId());
check("the portal is what you point at", plainHover === portalSetup.door,
  `hovered=${plainHover}`);

// Shift is no longer special: it adds, like Ctrl. (mouse.click has no
// `modifiers` option - that is page.click - so the key is held explicitly, and
// clicks are spaced out or they arrive as a double-click, which frames the
// camera and moves everything out from under the cursor.)
await page.evaluate(async (ids) => (await import("/js/editor.js")).select([ids.pane]), portalSetup);
await page.keyboard.down("Shift");
await page.mouse.click(doorAim.x, doorAim.y);
await page.keyboard.up("Shift");
await page.waitForTimeout(450);
const shiftClick = await page.evaluate(async () =>
  [...(await import("/js/editor.js")).state.selection]);
check("Shift+click adds rather than seeing through",
  shiftClick.length === 2 && shiftClick.includes(portalSetup.door)
    && shiftClick.includes(portalSetup.pane), `[${shiftClick}]`);

await page.keyboard.down("Shift");
await page.mouse.click(doorAim.x, doorAim.y);
await page.keyboard.up("Shift");
await page.waitForTimeout(450);
const shiftAgain = await page.evaluate(async () =>
  [...(await import("/js/editor.js")).state.selection]);
check("Shift+click toggles back off, the same as Ctrl",
  shiftAgain.length === 1 && shiftAgain[0] === portalSetup.pane, `[${shiftAgain}]`);

await page.evaluate(async (ids) => (await import("/js/editor.js")).select([ids.door]), portalSetup);
await page.keyboard.down("Control");
await page.mouse.click(doorAim.x, doorAim.y);
await page.keyboard.up("Control");
await page.waitForTimeout(450);
const ctrlToggled = await page.evaluate(async () =>
  [...(await import("/js/editor.js")).state.selection]);
check("Ctrl+click still toggles the selection", ctrlToggled.length === 0,
  `[${ctrlToggled}]`);

// the replacement: ghost the door and the pick falls through to the panel
const ghostAim = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  ed.select([ids.door]);
  ed.hideSelected();                      // ghost: visible, unpickable
  ed.select([]);
  // re-aim: nothing has moved, but read it fresh so a stray camera nudge in an
  // earlier step cannot silently make this test measure empty space
  const b = ed.screenBoundsOf(ed.state.placements.get(ids.pane).node);
  const r = ed.state.engine.getRenderingCanvas().getBoundingClientRect();
  return { x: r.x + (b.minX + b.maxX) / 2, y: r.y + (b.minY + b.maxY) / 2 };
}, portalSetup);
await page.mouse.move(ghostAim.x + 4, ghostAim.y + 4, { steps: 3 });
await page.mouse.move(ghostAim.x, ghostAim.y, { steps: 3 });
await page.waitForTimeout(300);
const throughGhost = await page.evaluate(async () =>
  (await import("/js/interact.js")).hoveredId());
check("Shift+H on the door lets the pick reach the panel behind it",
  throughGhost === portalSetup.pane, `hovered=${throughGhost}`);

await page.mouse.click(ghostAim.x, ghostAim.y);
await page.waitForTimeout(300);
const ghostClick = await page.evaluate(async () =>
  [...(await import("/js/editor.js")).state.selection]);
check("and a click lands on the panel, not the portal",
  ghostClick.length === 1 && ghostClick[0] === portalSetup.pane, `[${ghostClick}]`);

// a rectangle catches the door again - there is no portal exception left
const band = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const wide = { x0: -1e5, y0: -1e5, x1: 1e5, y1: 1e5 };
  ed.unhideAll();
  const all = ed.elementsInRect(wide);
  return { all, arity: ed.elementsInRect.length };
});
check("a rectangle catches doors like anything else",
  band.all.includes(portalSetup.door) && band.all.includes(portalSetup.pane),
  `[${band.all}]`);
check("elementsInRect no longer takes an options argument", band.arity === 1,
  `arity=${band.arity}`);
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
});

// ---- 1j. only a quick click selects -----------------------------------------
// The left button no longer moves the camera at all, but a press that is held
// still must not select: the drag candidate used to select on pointer-down, so
// merely resting the button on an element picked it up.
const quick = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const id = await ed.placeAt("Platforms/Platform_Simple", new BABYLON.Vector3(0, 0, 0));
  ed.state.camera.position = new BABYLON.Vector3(0, 6, -0.2);
  ed.state.camera.setTarget(new BABYLON.Vector3(0, 0, 0));
  return { id, clickMs: ed.CLICK_MS };
});
const onTile = await screenOf([0, 0, 0]);

async function pressAndHold(holdMs) {
  await page.evaluate(async () => (await import("/js/editor.js")).select([]));
  await page.mouse.move(onTile.x, onTile.y, { steps: 4 });
  await page.waitForTimeout(120);
  await page.mouse.down();
  await page.waitForTimeout(holdMs);
  const during = await page.evaluate(async () =>
    (await import("/js/editor.js")).state.selection.length);
  await page.mouse.up();
  await page.waitForTimeout(200);
  const after = await page.evaluate(async () =>
    (await import("/js/editor.js")).state.selection.length);
  return { during, after };
}

const tap = await pressAndHold(80);
check("a quick click still selects", tap.after === 1,
  `during press=${tap.during}, after release=${tap.after}`);

const hold = await pressAndHold(quick.clickMs + 350);
check("a long press selects nothing",
  hold.during === 0 && hold.after === 0,
  `during press=${hold.during}, after release=${hold.after} (CLICK_MS=${quick.clickMs})`);

// a real drag must still select what it moves, however long it takes to start
await page.evaluate(async () => (await import("/js/editor.js")).select([]));
await page.mouse.move(onTile.x, onTile.y, { steps: 4 });
await page.mouse.down();
await page.waitForTimeout(500);                       // deliberately slow start
await page.mouse.move(onTile.x + 90, onTile.y + 40, { steps: 6 });
await page.waitForTimeout(150);
const slowDrag = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return { dragging: i.isDragging(), sel: ed.state.selection.length };
});
await page.mouse.up();
await page.waitForTimeout(200);
check("a slow-starting drag still selects and moves what it drags",
  slowDrag.dragging && slowDrag.sel === 1,
  `dragging=${slowDrag.dragging}, selection=${slowDrag.sel}`);
await page.evaluate(async () => (await import("/js/editor.js")).clearAll());

// ---- 1k. the Chunk button isolates -------------------------------------------
// The chunk label doubles as the isolate toggle: pressed, every chunk but the
// active one is hidden, and the button has to look pressed while that is true.
const isolate = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  ed.addChunk("CH_ISO_A");
  ed.addChunk("CH_ISO_B");
  ed.state.activeChunk = "CH_ISO_A";
  await ed.placeAt("Platforms/Platform_3Plates", new BABYLON.Vector3(0, 0, 0));
  ed.state.activeChunk = "CH_ISO_B";
  await ed.placeAt("Platforms/Platform_3Plates", new BABYLON.Vector3(8, 0, 0));
  ed.state.activeChunk = "CH_ISO_A";
  document.getElementById("chunk-select").value = "CH_ISO_A";

  const shown = () => [...ed.state.placements.values()]
    .filter((e) => e.node.isEnabled()).map((e) => e.chunk).sort();
  return { total: ed.state.placements.size, before: shown() };
});
const btnBefore = await page.getAttribute("#btn-isolate", "aria-pressed");

await page.click("#btn-isolate");
await page.waitForTimeout(250);
const isoOn = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const btn = document.getElementById("btn-isolate");
  const cs = getComputedStyle(btn);
  return {
    isolate: ed.state.isolate,
    pressed: btn.getAttribute("aria-pressed"),
    shown: [...ed.state.placements.values()].filter((e) => e.node.isEnabled())
      .map((e) => e.chunk).sort(),
    bold: cs.fontWeight, bg: cs.backgroundColor,
  };
});
check("the Chunk button starts unpressed and shows every chunk",
  btnBefore === "false" && isolate.before.length === isolate.total,
  `pressed=${btnBefore}, ${isolate.before.length}/${isolate.total} shown`);
check("pressing Chunk hides every chunk but the active one",
  isoOn.isolate === true && isoOn.shown.length === 1 && isoOn.shown[0] === "CH_ISO_A",
  `showing [${isoOn.shown}]`);
check("a pressed Chunk button is visibly different",
  isoOn.pressed === "true" && Number(isoOn.bold) >= 600 && isoOn.bg !== "rgba(0, 0, 0, 0)",
  `aria-pressed=${isoOn.pressed}, weight=${isoOn.bold}, bg=${isoOn.bg}`);

await page.selectOption("#chunk-select", "CH_ISO_B");
await page.waitForTimeout(250);
const followed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return [...ed.state.placements.values()].filter((e) => e.node.isEnabled())
    .map((e) => e.chunk);
});
check("isolation follows the chunk dropdown",
  followed.length === 1 && followed[0] === "CH_ISO_B", `showing [${followed}]`);

await page.click("#btn-isolate");
await page.waitForTimeout(250);
const isoOff = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { isolate: ed.state.isolate,
           pressed: document.getElementById("btn-isolate").getAttribute("aria-pressed"),
           shown: [...ed.state.placements.values()].filter((e) => e.node.isEnabled()).length };
});
check("pressing Chunk again brings every chunk back",
  isoOff.isolate === false && isoOff.pressed === "false" && isoOff.shown === isolate.total,
  `pressed=${isoOff.pressed}, ${isoOff.shown}/${isolate.total} shown`);
await page.evaluate(async () => (await import("/js/editor.js")).clearAll());

await page.fill("#palette-search", "");
await page.waitForTimeout(400);

await page.screenshot({ path: "test/shot-ux.png" });

console.log(results.join("\n"));
console.log("\nfailures:", results.filter((r) => r.startsWith("FAIL")).length);console.log("errors  :", errors.length ? [...new Set(errors)].join("\n  ") : "(none)");
await browser.close();
process.exit(results.some((r) => r.startsWith("FAIL")) || errors.length ? 1 : 0);