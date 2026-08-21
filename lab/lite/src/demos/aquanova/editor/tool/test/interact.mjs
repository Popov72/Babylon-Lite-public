// Interaction test for the ghost-driven workflow: ghost follow, click-to-place,
// grab/drop, hover outline, wheel rotate/scale, grid elevation and axis modes.

import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { toolUrl } from "./target.mjs";
const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const HERE = path.dirname(fileURLToPath(import.meta.url));
const URL = toolUrl();

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1700, height: 950 } });
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
// Chromium's own console line for a bad response is just "Failed to load
// resource: the server responded with a status of 404" - no URL, which is
// useless when it fires somewhere in a suite this long. Record the response
// itself and drop the blind console echo of it.
//
// Before the first capture there is no .env on disk, and the Runtime view asks
// for one anyway - the miss is what makes a probe fall back to no reflection
// rather than to a stale one. Those 404s are the editor working, not breaking.
const EXPECTED_MISSING = /\/environments\//;
page.on("response", (r) => {
  if (r.status() >= 400 && !EXPECTED_MISSING.test(r.url())) {
    errors.push(`HTTP ${r.status()} ${r.url()}`);
  }
});
page.on("console", (m) => {
  if (m.type() !== "error") return;
  if (/Failed to load resource/i.test(m.text())) return;
  errors.push(m.text());
});

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

const MODULE = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";

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
  const proto = await (await import("/js/kit.js")).getProto("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight");
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
  const id = "Modular SciFi MegaKit/Walls/TopCables_Corner_Square_Inner";
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
    ? await (await import("/js/kit.js")).getProto("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight") : null;
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
           rotLabels: [...document.getElementById("snap-rot").options].map((o) => o.text),
           scaleOptions: [...document.getElementById("snap-scale").options].map((o) => o.value),
           scaleLabels: [...document.getElementById("snap-scale").options].map((o) => o.text),
           posOptions: [...document.getElementById("snap-pos").options].map((o) => o.value),
           posLabels: [...document.getElementById("snap-pos").options].map((o) => o.text) };
});
check("the Env slider drives the scene's IBL strength",
  knobs.raised.state === 3.2 && knobs.raised.scene === 3.2,
  `${knobs.before.scene} -> ${knobs.raised.scene}`);
check("Env intensity is clamped", knobs.high === 6 && knobs.low === 0,
  `high=${knobs.high}, low=${knobs.low}`);
check("the inertia slider is gone and the camera is fixed at 0.75",
  !knobs.inertiaSlider && knobs.camInertia === 0.75,
  `slider=${knobs.inertiaSlider}, inertia=${knobs.camInertia}`);
// A zero step is real for Move (free positioning while dragging) but
// meaningless for a keyboard step: a step of zero simply does nothing, and it
// fell back to a hidden default instead. `free` is the useful reading of the
// same idea - a step small enough to dial in any value with the keys you
// already use, rather than no step at all. All three lists say `free`; only
// Move's is actually zero.
check("Rot and Scale no longer offer a meaningless 'off'",
  !knobs.rotOptions.includes("0") && !knobs.scaleOptions.includes("0"),
  `rot [${knobs.rotOptions}], scale [${knobs.scaleOptions}]`);
// The step used to carry a sign, back when a key turned and only ever one way.
// The wheel turns both ways, so the sign went with it: the list is magnitudes
// again, running from a fine step up to a quarter turn.
check("the rotation step runs from a fine step up to a quarter turn",
  knobs.rotOptions.join() === "0.5,5,15,45,90",
  `[${knobs.rotOptions}]`);
check("and the scale step starts at a fine one",
  knobs.scaleOptions.join() === "0.01,0.05,0.1,0.25", `[${knobs.scaleOptions}]`);
check("all three step lists call their loosest setting 'free'",
  knobs.rotLabels.filter((t) => /free/.test(t)).length === 1
    && knobs.scaleLabels.filter((t) => /free/.test(t)).length === 1
    && knobs.posLabels.filter((t) => /free/.test(t)).length === 1
    && knobs.posLabels[0] === "free" && knobs.posOptions[0] === "0",
  `move [${knobs.posLabels}], rot [${knobs.rotLabels}], scale [${knobs.scaleLabels}]`);

const rotSign = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const c = co.addCollider("box", new V(0, 0, 0), { silent: true, scale: [3, 0.4, 0.8] });
  ed.select([c.id]);
  const wasStep = ed.state.snap.rot, wasAxis = ed.state.rotAxis;
  ed.state.rotAxis = "y";
  const run = (step, dir) => {
    c.node.rotationQuaternion = BABYLON.Quaternion.Identity();
    ed.state.snap.rot = step;
    i.rotateCurrent(dir);
    return +(c.node.rotationQuaternion.toEulerAngles().y * 180 / Math.PI).toFixed(2);
  };
  const out = { p90: run(90, 1), m90: run(90, -1), p45: run(45, 1), m45: run(45, -1),
    p5: run(5, 1), m5: run(5, -1), free: run(0.5, 1), freeBack: run(0.5, -1) };
  ed.state.snap.rot = wasStep; ed.state.rotAxis = wasAxis;
  ed.clearAll(); ed.select([]);
  return out;
});
check("the direction comes from the wheel, not the step",
  rotSign.p90 === -rotSign.m90 && rotSign.p45 === -rotSign.m45 && rotSign.p5 === -rotSign.m5
    && rotSign.p90 === 90,
  `${rotSign.p90}/${rotSign.m90}, ${rotSign.p45}/${rotSign.m45}, ${rotSign.p5}/${rotSign.m5}`);
// `free` is a fine step, not no step - a step of zero would simply do nothing,
// which is why "off" was taken off these two lists in the first place.
check("'free' turns by half a degree, either way",
  Math.abs(rotSign.free - 0.5) < 1e-6 && Math.abs(rotSign.freeBack + 0.5) < 1e-6,
  `${rotSign.free} / ${rotSign.freeBack}`);

// The wheel already goes both ways, so the scale step needs no sign - but its
// floor did need lowering. It sat at 5 cm, which is thicker than most of what
// the kit is made of: a collision shell fitted to its 7.5 mm walls could not be
// nudged at all, and a decal sits well under it.
const scaleFree = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const c = co.addCollider("box", new V(0, 0, 0), { silent: true, scale: [1, 1, 1] });
  ed.select([c.id]);
  const wasStep = ed.state.snap.scale, wasAxis = ed.state.scaleAxis;
  ed.state.scaleAxis = "y";
  const run = (step, dir, from) => {
    c.node.scaling.set(1, from, 1);
    ed.state.snap.scale = step;
    i.scaleCurrent(dir);
    return +c.node.scaling.y.toFixed(4);
  };
  const out = {
    up: run(0.01, 1, 1), down: run(0.01, -1, 1),
    coarse: run(0.1, 1, 1),
    // and it can go below the old fixed floor, one fine step at a time
    thin: run(0.01, -1, 0.04),
    // while a coarse step keeps the floor it always had
    coarseFloor: run(0.1, -1, 0.06),
  };
  ed.state.snap.scale = wasStep; ed.state.scaleAxis = wasAxis;
  co.removeCollider(c.id, true); ed.clearAll(); ed.select([]);
  return out;
});
check("'free' resizes by 0.01 a notch, up and down",
  Math.abs(scaleFree.up - 1.01) < 1e-4 && Math.abs(scaleFree.down - 0.99) < 1e-4
    && Math.abs(scaleFree.coarse - 1.1) < 1e-4,
  `up ${scaleFree.up}, down ${scaleFree.down}, coarse ${scaleFree.coarse}`);
check("and the floor is 1 cm, whatever step you got there with",
  Math.abs(scaleFree.thin - 0.03) < 1e-4 && Math.abs(scaleFree.coarseFloor - 0.01) < 1e-4,
  `fine 0.04 -> ${scaleFree.thin}, coarse 0.06 -> ${scaleFree.coarseFloor}`);

// ---- World/Local governs turning too ----------------------------------------
// `R` turned about a world axis whatever the element was doing, so on a wall
// already yawed 90 degrees "turn about X" tumbled it about the room rather than
// about its own length. Whose axis it is now comes from the same combo as
// moving, and the axis is taken per element, matching the origin: each turns
// about its own origin, so each turns about its own axis.
const rotSpace = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3, Q = BABYLON.Quaternion;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const c = co.addCollider("box", new V(0, 0, 0), { silent: true, scale: [3, 0.4, 1] });
  ed.select([c.id]);
  const wasStep = ed.state.snap.rot, wasAxis = ed.state.rotAxis;
  ed.state.snap.rot = 90;
  const rows = () => {
    c.node.computeWorldMatrix(true);
    const m = c.node.getWorldMatrix();
    const row = (k) => {
      const q = m.getRow(k);
      return new V(q.x, q.y, q.z).normalize().asArray().map((v) => +v.toFixed(3));
    };
    return { x: row(0), y: row(1), z: row(2) };
  };
  const run = (space, yawDeg, rotAxis) => {
    c.node.rotationQuaternion = Q.FromEulerAngles(0, yawDeg * Math.PI / 180, 0);
    c.node.computeWorldMatrix(true);
    i.setAxisSpace(space);
    ed.state.rotAxis = rotAxis;
    i.rotateCurrent(1);
    return rows();
  };
  const out = {
    // wheelTargets prefers whatever is under the cursor, so a stray hover would
    // turn a different element and quietly make all of this vacuous
    hovered: i.hoveredId?.() ?? null,
    yawedWorld: run("world", 90, "x"),
    yawedLocal: run("local", 90, "x"),
    flatWorld: run("world", 0, "x"),
    flatLocal: run("local", 0, "x"),
    aboutYWorld: run("world", 90, "y"),
    aboutYLocal: run("local", 90, "y"),
  };
  i.setAxisSpace("world");
  ed.state.snap.rot = wasStep; ed.state.rotAxis = wasAxis;
  co.removeCollider(c.id, true); ed.clearAll(); ed.select([]);
  return out;
});
const rotSame = (a, b) => JSON.stringify(a) === JSON.stringify(b);
check("nothing was hovered, so the checks below turned the element they meant to",
  rotSpace.hovered === null, `${rotSpace.hovered}`);
check("on an unturned element the two spaces agree exactly",
  rotSame(rotSpace.flatWorld, rotSpace.flatLocal), JSON.stringify(rotSpace.flatWorld));
check("but on a yawed wall, turning about X differs between them",
  !rotSame(rotSpace.yawedWorld, rotSpace.yawedLocal),
  `world ${JSON.stringify(rotSpace.yawedWorld.x)}, local ${JSON.stringify(rotSpace.yawedLocal.x)}`);
// The definitive one: a turn leaves its own axis alone, so in local space the
// element's own x row has to come out unchanged - it *is* the axis.
check("a local X turn leaves the element's own X where it was, being the axis",
  rotSame(rotSpace.yawedLocal.x, [0, 0, -1]), JSON.stringify(rotSpace.yawedLocal.x));
check("while a world X turn does not",
  !rotSame(rotSpace.yawedWorld.x, [0, 0, -1]), JSON.stringify(rotSpace.yawedWorld.x));
check("about Y they agree on a yawed wall, its own Y being the world's",
  rotSame(rotSpace.aboutYWorld, rotSpace.aboutYLocal), JSON.stringify(rotSpace.aboutYLocal));

// ---- 1d-quater. undo / redo ------------------------------------------------
const history = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const base = ed.state.placements.size;

  ed.pushUndo();
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(4, 0, 0), { silent: true });
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  mk.addDoor(new V(2, 0, 2), { silent: true });
  const base = { placements: ed.state.placements.size, markers: ed.state.markers.size };

  // two placements, exactly as a run of palette clicks would
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(4, 0, 0));
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(8, 0, 0));
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
  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.state.activeChunk = "CH_B";
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(8, 0, 0), { silent: true });
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

// ---- 1d-quinquies-bis. the chunks pane -------------------------------------
// Deleting a chunk that still holds anything would turn one keystroke into
// unbounded loss, so it refuses and names what is in the way.
const pane = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  // The chunk list survives `clearAll` - it is the ship's layout, not its
  // contents - so it is emptied by hand to make the "last chunk" case reachable.
  ed.state.chunks = [];
  ed.addChunk("CH_P_A"); ed.addChunk("CH_P_B");
  ed.state.activeChunk = "CH_P_A";
  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });

  const busy = ed.removeChunk("CH_P_A");
  const users = ed.chunkUsers("CH_P_A");
  ed.removePlacement(a.id);
  const freed = ed.removeChunk("CH_P_A");
  const gone = !ed.state.chunks.includes("CH_P_A");

  // A room reaches the manifest whether or not anything is in it: the runtime
  // streams collision per chunk, and a room that vanished when it was emptied
  // would take its portals with it.
  const inManifest = mf.buildManifest().chunks.map((c) => c.id);

  await ed.deserialize(JSON.parse(JSON.stringify(ed.serialize())));
  const survived = [...ed.state.chunks];

  ed.renameChunk("CH_P_B", "CH_P_C");
  const rekeyed = ed.state.chunks.includes("CH_P_C");
  const last = ed.removeChunk("CH_P_C");
  ed.clearAll(); ed.select([]);
  return { busy, users, freed, gone, inManifest, survived, rekeyed, last };
});
check("a chunk that still holds something refuses to be deleted, and says what",
  pane.busy.ok === false && pane.busy.reason === "in use"
    && pane.users.placements.length === 1,
  `${pane.busy.reason}, ${pane.users.placements.length} placement(s)`);
check("an emptied chunk can be deleted, and the last one cannot",
  pane.freed.ok === true && pane.gone && pane.last.ok === false
    && pane.last.reason === "last",
  `freed=${pane.freed.ok}, gone=${pane.gone}, last=${pane.last.reason}`);
check("an empty room still reaches the manifest",
  pane.inManifest.includes("CH_P_B"), pane.inManifest.join());
check("the chunk list survives a save/load round-trip",
  pane.survived.includes("CH_P_B"), pane.survived.join());
check("renaming a chunk rekeys it",
  pane.rekeyed === true, `rekeyed=${pane.rekeyed}`);

// The pane itself, driven through the DOM rather than through the module: the
// wiring between the two is where the buttons that were removed from the
// toolbar actually went, and none of the checks above would notice a form that
// never reaches the state it edits.
const paneUi = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const $ = (id) => document.getElementById(id);
  let refreshedAlone = false;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.state.chunks = [];
  ed.addChunk("CH_UI_A");

  $("btn-chunks").click();
  const opened = !$("chunk-modal").hidden;
  $("btn-chunk-new").click();
  const added = ed.state.chunks.length === 2;
  const listed = $("chunk-list").options.length;

  // Pick the first room and rename it through the form.
  $("chunk-list").value = "CH_UI_A";
  $("chunk-list").dispatchEvent(new Event("change"));
  $("chunk-name").value = "CH_UI_Renamed";
  $("btn-chunk-apply").click();
  const renamed = ed.state.chunks.includes("CH_UI_Renamed");

  // Deleting a room that still holds something has to say what, in the pane.
  ed.state.activeChunk = "CH_UI_Renamed";
  const obj = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
    new BABYLON.Vector3(0, 0, 0), { silent: true });
  $("chunk-list").value = "CH_UI_Renamed";
  $("chunk-list").dispatchEvent(new Event("change"));
  // What the room holds is shown rather than left to be discovered by a
  // refused Delete: the count is the whole reason to open the pane at all.
  const holds = $("chunk-holds").textContent;
  $("btn-chunk-delete").click();
  const refused = $("chunk-error").textContent;
  const survived = ed.state.chunks.includes("CH_UI_Renamed");

  // And the last room cannot go at all, which the pane says by going grey
  // rather than by waiting for the click and then refusing it.
  ed.removePlacement(obj.id);
  ed.removeChunk("CH01_New");
  refreshedAlone = $("btn-chunk-delete").disabled;

  $("btn-chunk-close").click();
  const closed = $("chunk-modal").hidden;
  ed.clearAll(); ed.select([]);
  return { opened, added, listed, renamed, holds, refused, survived,
           refusedId: obj.id, lastIsGrey: refreshedAlone, closed };
});
check("the Chunks pane opens, lists the ship's rooms and adds one",
  paneUi.opened && paneUi.added && paneUi.listed === 2 && paneUi.closed,
  `open=${paneUi.opened}, ${paneUi.listed} listed, closed=${paneUi.closed}`);
check("Apply renames the selected room",
  paneUi.renamed === true, `renamed=${paneUi.renamed}`);
check("the pane says what a room holds before you try to delete it",
  /1\b/.test(paneUi.holds || ""), `"${paneUi.holds}"`);
check("and Delete explains what is in the way instead of just failing",
  /still holds/i.test(paneUi.refused) && paneUi.refused.includes(paneUi.refusedId)
    && paneUi.survived,
  `"${paneUi.refused}"`);
check("and the last remaining chunk cannot be deleted at all",
  paneUi.lastIsGrey === true, `disabled=${paneUi.lastIsGrey}`);

// ---- 1d-quinquies-ter. independent environment probes ---------------------
// Reflection volumes are authored independently from chunks: one probe may
// span several rooms, and splitting a room must not split its cubemap. The
// capture camera is deliberately separate from the projection box centre.
const probes = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const interact = await import("/js/interact.js");
  const mf = await import("/js/manifest.js");
  const runtime = await import("/js/runtime.js");
  const $ = (id) => document.getElementById(id);
  // Probe gizmos are per probe now, so their nodes carry the id they belong to.
  const gizmo = (part, probeId) => `LOCAL_ENVIRONMENT_${part}#${probeId}`;
  // The probe window's buttons kick off their own async re-show, and
  // showEnvironmentProbes is last-request-wins: a call made from here right
  // after a click is cancelled by the one already in flight. So wait for the
  // window to finish dressing the gizmo rather than racing it.
  const settle = async (probeId) => {
    for (let i = 0; i < 400 && !ed.entryOf(probeId)?.node; i++) {
      await new Promise((r) => setTimeout(r, 25));
    }
    return ed.entryOf(probeId)?.node || null;
  };
  // Typing is what commits now: there is no Apply, so a test types the way a
  // user does and lets the pane write the record.
  const type = (field, value, event = "input") => {
    $(field).value = value;
    $(field).dispatchEvent(new Event(event));
  };
  ed.state.environmentProbes.clear();

  $("btn-probes").click();
  const opened = !$("probe-modal").hidden;
  // Capture acts on the selection, so with the list empty it has to open inert
  // - and come back the moment there is a probe to point at.
  const captureButtons = {
    labels: [$("btn-capture-one").textContent.trim(), $("btn-capture-all").textContent.trim()],
    emptyDisabled: $("btn-capture-one").disabled,
        allDisabled: $("btn-capture-all").disabled,
        // Capture all takes no selection, so it does not belong to a probe and is
        // not in the window: it is a ship-wide action, filed with the ship-wide
        // cubemap size it is the answer to.
        allInSettings: !!$("btn-capture-all").closest("#settings-pane .settings-section") && !$("btn-capture-all").closest("#probe-modal"),
        oneInWindow: !!$("btn-capture-one").closest("#probe-modal"),
    };
    $("btn-probe-new").click();
  captureButtons.pickedDisabled = $("btn-capture-one").disabled;
  const id = $("probe-list").value;
  // A brand-new probe never had influence volumes typed for it, so they are
  // derived from the box it does have. The box itself depends on the camera,
  // so the check is the relationship rather than the numbers.
  const created = ed.environmentProbeOf(id);
  const derivedInfluence = created.influenceBoxPosition.join() === created.boxPosition.join()
    && created.influenceBoxSize.every((n, axis) => n === created.boxSize[axis] + 3)
    && created.influenceInnerBoxSize.every(
      (n, axis) => n === Math.max(0, created.boxSize[axis] - 3));

  const values = {
    "probe-box-x": "1", "probe-box-y": "2", "probe-box-z": "3",
    "probe-size-x": "8", "probe-size-y": "5", "probe-size-z": "12",
    "probe-camera-x": "7", "probe-camera-y": "8", "probe-camera-z": "9",
    "probe-influence-x": "1.5", "probe-influence-y": "2", "probe-influence-z": "3",
    "probe-influence-size-x": "12", "probe-influence-size-y": "9", "probe-influence-size-z": "16",
    "probe-inner-size-x": "4", "probe-inner-size-y": "1",
        "probe-inner-size-z": "8",
    };
    for (const [field, value] of Object.entries(values)) type(field, value);
  await settle(id);
  await runtime.showEnvironmentProbes(id);

  const centre = ed.state.scene.getMeshByName(gizmo("BOX_CENTRE", id));
  const camera = ed.state.scene.getMeshByName(gizmo("CAMERA", id));
  // The volumes are a root plus a box, like the capture volume: the root is
  // what carries the transform, the box is what is picked and outlined.
  const outerRoot = ed.state.scene.getTransformNodeByName(gizmo("INFLUENCE_BOX_ROOT", id));
  const innerRoot = ed.state.scene.getTransformNodeByName(gizmo("INNER_BOX_ROOT", id));
  const outerBox = ed.state.scene.getMeshByName(gizmo("INFLUENCE_BOX", id));
  const innerBox = ed.state.scene.getMeshByName(gizmo("INNER_BOX", id));
  const live = {
    centre: centre?.position.asArray(),
    camera: camera?.position.asArray(),
    separate: centre && camera && !centre.position.equals(camera.position),
    outer: outerRoot?.isEnabled() ? outerRoot.scaling.asArray() : null,
    inner: innerRoot?.isEnabled() ? innerRoot.scaling.asArray() : null,
    influenceCentre: outerRoot?.position.asArray(),
    innerCentre: innerRoot?.position.asArray(),
    influencePickable: !!outerBox?.isPickable && !!innerBox?.isPickable,
    // Each volume is its own selectable part of the probe, and picking one of
    // its meshes has to resolve to that part rather than to the capture box.
    influenceOwners: [ed.ownerIdOf(outerBox), ed.ownerIdOf(innerBox)],
    influencePartIds: [
      ed.environmentProbePartId(id, "influence"), ed.environmentProbePartId(id, "inner"),
    ],
  };

  // An inner box outside its outer one would make the runtime's normalized
  // distance field run backwards, so the pane refuses it rather than storing it.
  // Typing is quiet - "0.5" is unusable for a moment on its way in - so the
  // complaint arrives when the field is left, along with the good value back.
  type("probe-inner-size-x", "99");
  const innerHeldWhileTyped =
    ed.environmentProbeOf(id).influenceInnerBoxSize.join() === "4,1,8";
  type("probe-inner-size-x", "99", "blur");
  const innerRefused = innerHeldWhileTyped
    && /inner size/i.test($("probe-error").textContent)
    && ed.environmentProbeOf(id).influenceInnerBoxSize.join() === "4,1,8";
  await new Promise((r) => setTimeout(r, 200));
  const innerFieldRestored = $("probe-inner-size-x").value === "4";

  ed.setEnvironmentProbe("ENV_OTHER", {
    boxPosition: [30, 2, 0], boxSize: [2, 2, 2],
        capturePosition: [30, 2, 0],
    });
    type("probe-id", "ENV_OTHER", "change");
  const duplicateRefused = /already used/i.test($("probe-error").textContent)
    && !!ed.environmentProbeOf(id);

  // Any editor entry's id is off limits for a probe, not just another probe's.
  // The suite clears the ship before this block, so make something to collide
  // with rather than depending on whatever an earlier block left lying around.
  // A collider is the cheapest entry to conjure: generated geometry, no kit
  // module to load.
  const colliders = await import("/js/colliders.js");
  const sceneId = colliders.addCollider(
    "box", new BABYLON.Vector3(40, 1, 40), { silent: true })?.id;
  type("probe-id", sceneId, "change");
  const sceneConflictRefused = !!sceneId
    && /already used/i.test($("probe-error").textContent)
    && !!ed.environmentProbeOf(id);
  colliders.removeCollider(sceneId, true);

  const renamedId = "storage_main";
  type("probe-id", renamedId, "change");
  await settle(renamedId);
  ed.select([renamedId]);
  const renamed = !ed.environmentProbeOf(id) && !!ed.environmentProbeOf(renamedId);

  const root = ed.entryOf(renamedId)?.node;
  root.position.set(3, 5, 7);
  root.scaling.set(10, 6, 14);
  ed.emit("transform");
  const transformed = ed.environmentProbeOf(renamedId);
  const beforeRotation = root.rotationQuaternion?.asArray() || root.rotation.asArray();
  interact.rotateCurrent(1);
  const afterRotation = root.rotationQuaternion?.asArray() || root.rotation.asArray();
  const axes = ed.toggleAxes(renamedId, "world");

  const authored = ed.environmentProbeOf(renamedId);
  const inManifest = mf.buildManifest().environmentProbes.find((probe) => probe.id === renamedId);
  const snapshot = JSON.parse(JSON.stringify(ed.serialize()));
  await ed.deserialize(snapshot);
  const survived = ed.environmentProbeOf(renamedId);

  // Deserialize re-emits "environment-probes", which re-renders the list
  // asynchronously; setting .value before that lands would silently select
  // nothing and make the delete below a no-op.
  for (let i = 0; i < 400 && !$("probe-list").querySelector(`option[value="${renamedId}"]`); i++) {
    await new Promise((r) => setTimeout(r, 25));
  }
  await runtime.showEnvironmentProbes(renamedId);

  // ---- direct manipulation of the two blend volumes ---------------------
  // They are parts of the probe rather than entries of their own, so the same
  // selection, drag, wheel and inspector paths that move the capture box work
  // on them - which is the whole point of giving them synthetic ids.
  const influenceId = ed.environmentProbePartId(renamedId, "influence");
  const innerId = ed.environmentProbePartId(renamedId, "inner");
  ed.select([influenceId]);
  const outerNode = ed.entryOf(influenceId)?.node;
  outerNode.position.set(4, 6, 8);
  // Deliberately shorter on Z than the inner box already is: shrinking the
  // outer volume has to take the inner one with it, not leave a blend gradient
  // pointing the wrong way.
  outerNode.scaling.set(9, 8, 7);
  ed.emit("transform");
  const influenceDragged = ed.environmentProbeOf(renamedId);

  await runtime.showEnvironmentProbes(renamedId);
  ed.select([innerId]);
  const innerEntry = ed.entryOf(innerId);
  const innerCanMove = innerEntry?.canMove;
  const innerPositionRow = $("position-fields").hidden;
  const innerBefore = innerEntry.node.position.asArray();
  ed.nudgeSelection(new BABYLON.Vector3(5, 0, 0));
  const innerAfterNudge = innerEntry.node.position.asArray();
  const snapScale = ed.state.snap.scale, savedAxis = ed.state.scaleAxis;
  ed.state.snap.scale = 0.5;
  ed.state.scaleAxis = "all";
  interact.scaleCurrent(1);
  ed.state.snap.scale = snapScale;
  ed.state.scaleAxis = savedAxis;
  const innerResized = ed.environmentProbeOf(renamedId);

  ed.select([renamedId]);
  $("probe-list").value = renamedId;
  $("probe-list").dispatchEvent(new Event("change"));
  $("btn-probe-delete").click();
  const deleted = !ed.environmentProbeOf(renamedId);
  ed.removeEnvironmentProbe("ENV_OTHER");
  captureButtons.emptiedDisabled = $("btn-capture-one").disabled;
  $("btn-probe-close").click();
  return {
    opened, id, renamedId, live, authored, inManifest, survived, deleted,
    renamed, duplicateRefused, sceneConflictRefused, sceneId,
    transformed, beforeRotation, afterRotation, axes, captureButtons,
    derivedInfluence, innerRefused, innerFieldRestored,
    influenceDragged, innerResized, innerCanMove, innerPositionRow,
    innerMoveRefused: innerBefore.join() === innerAfterNudge.join(),
    closed: $("probe-modal").hidden,
  };
});
check("the Probes pane creates and deletes an independent probe volume",
  probes.opened && /^ENV\d{4}$/.test(probes.id) && probes.deleted && probes.closed,
  `${probes.id}, opened=${probes.opened}, deleted=${probes.deleted}, closed=${probes.closed}`);
check("probe IDs can be renamed but remain unique",
  probes.renamed && probes.duplicateRefused && probes.sceneConflictRefused
    && probes.authored?.id === probes.renamedId,
  `${probes.id} -> ${probes.renamedId}, probe=${probes.duplicateRefused}, scene=${probes.sceneConflictRefused} (vs ${probes.sceneId})`);
check("the probe box centre and capture camera are edited and drawn separately",
  probes.live.separate
    && probes.live.centre?.join() === "1,2,3"
    && probes.live.camera?.join() === "7,8,9",
  `${JSON.stringify(probes.live.centre)} / ${JSON.stringify(probes.live.camera)}`);
check("the regular transform tools move and scale the probe box and carry its camera",
  probes.transformed?.boxPosition?.join() === "3,5,7"
    && probes.transformed?.boxSize?.join() === "10,6,14"
    && probes.transformed?.capturePosition?.join() === "9,11,13"
    && probes.axes === probes.renamedId,
  JSON.stringify(probes.transformed));
check("a new probe derives its influence volumes from the box it was given",
  probes.derivedInfluence === true, JSON.stringify(probes.derivedInfluence));
check("the influence volumes are authored, drawn, and selectable in their own right",
  probes.live.outer?.join() === "12,9,16"
    && probes.live.inner?.join() === "4,1,8"
    && probes.live.influenceCentre?.join() === "1.5,2,3"
    && probes.live.influenceCentre?.join() === probes.live.innerCentre?.join()
    && probes.live.influencePickable
    && probes.live.influenceOwners?.join() === probes.live.influencePartIds?.join(),
  `${JSON.stringify(probes.live.outer)} / ${JSON.stringify(probes.live.inner)} @ ${JSON.stringify(probes.live.influenceCentre)}`
  + ` owners ${JSON.stringify(probes.live.influenceOwners)}`);
check("an inner influence box larger than its outer one is refused, and the field put back",
  probes.innerRefused === true && probes.innerFieldRestored === true,
  `refused=${probes.innerRefused}, field restored=${probes.innerFieldRestored}`);
check("the outer influence volume is moved and resized with the mouse, and takes the inner one down with it",
  probes.influenceDragged?.influenceBoxPosition?.join() === "4,6,8"
    && probes.influenceDragged?.influenceBoxSize?.join() === "9,8,7"
    && probes.influenceDragged?.influenceInnerBoxSize?.join() === "6,2,7"
    && probes.influenceDragged?.boxPosition?.join() === "3,5,7"
    && probes.influenceDragged?.boxSize?.join() === "10,6,14",
  JSON.stringify(probes.influenceDragged));
check("the inner influence volume resizes but never moves, and offers no Position row",
  probes.innerCanMove === false && probes.innerMoveRefused === true
    && probes.innerPositionRow === true
    && probes.innerResized?.influenceInnerBoxSize?.join() === "6.5,2.5,7"
    && probes.innerResized?.influenceBoxPosition?.join() === "4,6,8"
    && probes.innerResized?.influenceBoxSize?.join() === "9,8,7",
  `canMove=${probes.innerCanMove}, moved=${!probes.innerMoveRefused},`
  + ` row hidden=${probes.innerPositionRow}, ${JSON.stringify(probes.innerResized)}`);
check("a probe drag carries its influence volumes and a resize keeps their margins",
  probes.transformed?.influenceBoxPosition?.join() === "3.5,5,7"
    && probes.transformed?.influenceBoxSize?.join() === "14,10,18"
    && probes.transformed?.influenceInnerBoxSize?.join() === "6,2,10",
  JSON.stringify(probes.transformed));
check("environment probe boxes remain axis-aligned when rotation is requested",
  JSON.stringify(probes.beforeRotation) === JSON.stringify(probes.afterRotation),
  `${JSON.stringify(probes.beforeRotation)} -> ${JSON.stringify(probes.afterRotation)}`);
check(
    "probe size and camera survive an editor round-trip",
    probes.authored?.boxSize?.join() === "10,6,14" && probes.authored?.capturePosition?.join() === "9,11,13" && JSON.stringify(probes.authored) === JSON.stringify(probes.survived),
    `${JSON.stringify(probes.authored)} / ${JSON.stringify(probes.survived)}`
);
check("the manifest stores probes at the root and converts both positions to glTF space",
  probes.inManifest?.boxPosition?.join() === "-3,5,7"
    && probes.inManifest?.capturePosition?.join() === "-9,11,13"
    && probes.inManifest?.influenceBoxPosition?.join() === "-3.5,5,7"
    && probes.inManifest?.boxSize?.join() === "10,6,14"
    && probes.inManifest?.influenceBoxSize?.join() === "14,10,18" &&
        probes.inManifest?.influenceInnerBoxSize?.join() === "6,2,10" &&
        probes.inManifest?.angle === 0,
    JSON.stringify(probes.inManifest)
);
check("Capture follows the selected probe, Capture all never needs one",
  probes.captureButtons?.labels.join("|") === "Capture|Capture all"
    && probes.captureButtons.emptyDisabled === true
    && probes.captureButtons.pickedDisabled === false
    && probes.captureButtons.emptiedDisabled === true
    && probes.captureButtons.allDisabled === false,
  JSON.stringify(probes.captureButtons)
);
check(
    "so Capture all sits with the settings and Capture with the probe list",
    probes.captureButtons?.allInSettings === true && probes.captureButtons.oneInWindow === true,
    JSON.stringify({
        all: probes.captureButtons?.allInSettings,
        one: probes.captureButtons?.oneInWindow,
    })
);

// ---- 1d-quinquies-ter-bis. one cubemap size for the whole ship -------------
// The runtime holds the captured environments in a cube texture ARRAY, and
// every slice of an array shares one dimension - so the face size cannot be a
// per-probe field. It is a ship-wide setting on the Settings pane, saved in
// `config`, and a ship authored before it existed adopts the largest size its
// probes were carrying.
const cubemapSize = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const mf = await import("/js/manifest.js");
    const $ = (id) => document.getElementById(id);
    ed.state.environmentProbes.clear();
    ed.setEnvironmentProbe("ENV_SIZE", {
        boxPosition: [0, 2, 0],
        boxSize: [6, 4, 6],
        capturePosition: [0, 2, 0],
    });

    const row = $("cfg-probe-res");
    const offered = [...row.options].map((o) => o.value).join(",");
    const fromRange = offered === ed.PROBE_RESOLUTIONS.join(",");
    row.value = "1024";
    row.dispatchEvent(new Event("change"));
    const applied = ed.state.config.probeResolution;
    const manifest = mf.buildManifest();
    const inConfig = manifest.config?.probeResolution;
    const onProbe = Object.prototype.hasOwnProperty.call(manifest.environmentProbes[0] || {}, "resolution");
    // A size the array cannot be built at is refused, whatever asks for it.
    const oddRefused = ed.setConfig("probeResolution", 300) === false;

    await ed.undo(); // undo() restores asynchronously
    const undone = ed.state.config.probeResolution;

    ed.setConfig("probeResolution", 1024);
    const layout = ed.serialize();
    await ed.deserialize(JSON.parse(JSON.stringify(layout)));
    const roundTrip = ed.state.config.probeResolution;

    // What an older ship looks like: no setting, and a size on every probe. The
    // largest wins, so no room is quietly downsampled by the migration.
    const legacyOf = (sizes) => {
        const older = JSON.parse(JSON.stringify(layout));
        delete older.config.probeResolution;
        older.probeVolumes = sizes.map((resolution, index) => ({
            ...older.probeVolumes[0],
            id: `ENV_OLD_${index}`,
            resolution,
        }));
        return older;
    };
    await ed.deserialize(legacyOf([1024, 256]));
    const legacyLargest = ed.state.config.probeResolution;
    // Those old values were free-form, so one that is not on the list is rounded
    // up rather than refused - the ship still has to load.
    await ed.deserialize(legacyOf([300]));
    const legacySnapped = ed.state.config.probeResolution;
    await ed.deserialize(legacyOf([]));
    const legacyEmpty = ed.state.config.probeResolution;

    await ed.deserialize(JSON.parse(JSON.stringify(layout)));
    ed.setConfig("probeResolution", 512);
    ed.state.environmentProbes.clear();
    return {
        offered,
        fromRange,
        applied,
        inConfig,
        onProbe,
        oddRefused,
        undone,
        roundTrip,
        legacyLargest,
        legacySnapped,
        legacyEmpty,
    };
});
check(
    "the cubemap size is a ship-wide setting, not a probe field",
    cubemapSize.offered === "128,256,512,1024,2048" &&
        cubemapSize.fromRange === true &&
        cubemapSize.applied === 1024 &&
        cubemapSize.inConfig === 1024 &&
        cubemapSize.onProbe === false &&
        cubemapSize.oddRefused === true,
    `offered ${cubemapSize.offered}, applied ${cubemapSize.applied},` +
        ` config ${cubemapSize.inConfig}, on probe ${cubemapSize.onProbe},` +
        ` 300 refused ${cubemapSize.oddRefused}`
);
check(
    "changing the cubemap size is undoable and survives a round-trip",
    cubemapSize.undone === 512 && cubemapSize.roundTrip === 1024,
    `undone ${cubemapSize.undone}, round-trip ${cubemapSize.roundTrip}`
);
check(
    "a ship authored per probe adopts the largest size it was built at",
    cubemapSize.legacyLargest === 1024 && cubemapSize.legacySnapped === 512 && cubemapSize.legacyEmpty === 512,
    `largest ${cubemapSize.legacyLargest}, snapped ${cubemapSize.legacySnapped},` + ` empty ${cubemapSize.legacyEmpty}`
);

// ---- 1d-quinquies-quater. the probe pane's three sections ------------------
// Three near-identical vector triples read as one block of numbers, so each
// volume gets a heading, an eye that shows and hides it on its own, and a mark
// when it is the box the viewport has selected.
const probeSections = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const $ = (id) => document.getElementById(id);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const enabled = (name) => {
    const s = ed.state.scene;
    const node = s.getTransformNodeByName(name) || s.getMeshByName(name);
    return !!node?.isEnabled();
  };
  const gizmos = (probeId) => ({
    box: enabled(`LOCAL_ENVIRONMENT_BOX#${probeId}`),
    centre: enabled(`LOCAL_ENVIRONMENT_BOX_CENTRE#${probeId}`),
    camera: enabled(`LOCAL_ENVIRONMENT_CAMERA#${probeId}`),
    influence: enabled(`LOCAL_ENVIRONMENT_INFLUENCE_BOX#${probeId}`),
    inner: enabled(`LOCAL_ENVIRONMENT_INNER_BOX#${probeId}`),
  });
  const heads = () => [...document.querySelectorAll("#probe-modal .probe-head")].map((h) => ({
    id: h.id,
    title: h.querySelector("span").textContent,
    eye: h.querySelector(".eye").getAttribute("aria-pressed"),
    selected: h.classList.contains("selected"),
    badge: !h.querySelector(".sel").hidden,
  }));

  ed.state.environmentProbes.clear();
  $("btn-probes").click();
  $("btn-probe-new").click();
  const id = $("probe-list").value;
  for (let i = 0; i < 400 && !ed.entryOf(id)?.node; i++) await wait(25);
  await wait(150);

  // Reading order: the pane's own children, so a row moved in the markup shows
  // up here rather than in a screenshot nobody looks at.
  const order = [...document.querySelectorAll("#probe-modal .editor > *")].map((el) =>
    (el.tagName === "H3"
      ? `H3:${el.querySelector("span").textContent}`
      : (el.querySelector("label")?.textContent?.trim() || el.id)));

  const opening = { heads: heads(), gizmos: gizmos(id) };

  $("btn-probe-eye-influence").click();
  await wait(150);
  const influenceHidden = { heads: heads(), gizmos: gizmos(id) };

  // Hiding the capture box has to take the selection with it: a gizmo that is
  // not drawn cannot be dragged, framed or outlined.
  $("btn-probe-eye-box").click();
  await wait(150);
  const boxHidden = { gizmos: gizmos(id), selection: ed.state.selection.slice() };

  // A view choice, so it stays out of the ship the game reads - but it is still
  // worth keeping, so it is written beside it.
  const mf = await import("/js/manifest.js");
  const manifest = mf.buildManifest();
  const inShip = JSON.stringify(manifest.environmentProbes[0] || {});
  const inPrefs = JSON.stringify(manifest.editorPrefs?.probes?.[id] || {});

  $("btn-probe-eye-box").click();
  $("btn-probe-eye-influence").click();
  await wait(250);
  const restored = gizmos(id);

  ed.select([ed.environmentProbePartId(id, "inner")]);
  await wait(150);
  const innerSelected = heads();
  ed.select([ed.environmentProbePartId(id, "influence")]);
  await wait(150);
  const influenceSelected = heads();

  // What the manifest was written with is what it puts back.
  ed.applyEditorPrefs(manifest.editorPrefs);
  await wait(250);
  const reloaded = gizmos(id);

  ed.removeEnvironmentProbe(id);
  $("btn-probe-close").click();
  return {
    order, opening, influenceHidden, boxHidden, restored,
    innerSelected, influenceSelected, inShip, inPrefs, reloaded,
  };
});
check("the probe pane is split into Probe box, Influence box and Inner box",
    probeSections.order.join("|") ===
        ["ID", "Always visible", "Env faces", "H3:Probe box", "Centre", "Size", "Camera", "H3:Influence box", "Centre", "Size", "H3:Inner box", "Size", "probe-resolved"].join("|"),
    probeSections.order.join(" · ")
);
check("each probe section owns an eye that shows and hides only its own volume",
  probeSections.opening.heads.every((h) => h.eye === "true")
    && Object.values(probeSections.opening.gizmos).every(Boolean)
    && probeSections.influenceHidden.gizmos.influence === false
    && probeSections.influenceHidden.gizmos.box === true
    && probeSections.influenceHidden.gizmos.inner === true
    && probeSections.influenceHidden.heads[1].eye === "false"
    && probeSections.influenceHidden.heads[0].eye === "true"
    && Object.values(probeSections.restored).every(Boolean),
  `${JSON.stringify(probeSections.influenceHidden.gizmos)} → ${JSON.stringify(probeSections.restored)}`);
check("hiding the probe box takes its centre, camera and selection with it",
  probeSections.boxHidden.gizmos.box === false
    && probeSections.boxHidden.gizmos.centre === false
    && probeSections.boxHidden.gizmos.camera === false
    && probeSections.boxHidden.gizmos.inner === true
    && probeSections.boxHidden.selection.length === 0,
  `${JSON.stringify(probeSections.boxHidden.gizmos)}, selection ${JSON.stringify(probeSections.boxHidden.selection)}`);
check("the pane marks whichever box the viewport has selected",
  probeSections.opening.heads[0].selected === true
    && probeSections.opening.heads[0].badge === true
    && probeSections.innerSelected.map((h) => h.selected).join() === "false,false,true"
    && probeSections.innerSelected[2].badge === true
    && probeSections.influenceSelected.map((h) => h.selected).join() === "false,true,false"
    && probeSections.influenceSelected[1].badge === true,
  `open=${JSON.stringify(probeSections.opening.heads.map((h) => h.selected))},`
  + ` inner=${JSON.stringify(probeSections.innerSelected.map((h) => h.selected))},`
  + ` influence=${JSON.stringify(probeSections.influenceSelected.map((h) => h.selected))}`);
check("per-box visibility stays out of the ship and is saved beside it",
  /"box":false/.test(probeSections.inPrefs)
    && /"influence":false/.test(probeSections.inPrefs)
    && /"inner":true/.test(probeSections.inPrefs)
    && !/visibleParts/.test(probeSections.inShip)
    && probeSections.reloaded.box === false
    && probeSections.reloaded.influence === false
    && probeSections.reloaded.inner === true,
  `prefs ${probeSections.inPrefs} · ship ${probeSections.inShip}`
  + ` · reloaded ${JSON.stringify(probeSections.reloaded)}`);

// ---- 1d-quinquies-quinquies. per-probe visibility and live editing ---------
// Reading one probe against its neighbours means holding several on screen at
// once, so which probes are drawn is a per-probe choice rather than one switch
// for the pane - and the one being worked on has to stay legible among them.
const probeView = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const $ = (id) => document.getElementById(id);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const node = (part, probeId) => {
    const s = ed.state.scene;
    const name = `LOCAL_ENVIRONMENT_${part}#${probeId}`;
    return s.getMeshByName(name) || s.getTransformNodeByName(name);
  };
  const material = (part, probeId) => node(part, probeId)?.material?.name || null;
  const drawn = (probeId) => !!node("BOX", probeId)?.isEnabled();
  // A lit face needs a captured cubemap, and nothing has been captured in a
  // suite that builds its ship from nothing - so what is checked is that the
  // faces were asked for, which is the per-probe choice under test: the mesh
  // exists as soon as a probe wants them and goes with the gizmos when that
  // probe stops being drawn.
  const faces = (probeId) => !!node("BOX_SURFACE", probeId);
  const settle = async (probeId) => {
    for (let i = 0; i < 400 && !drawn(probeId); i++) await wait(25);
    await wait(100);
  };

  ed.state.environmentProbes.clear();
  const near = {
    boxPosition: [0, 2, 0], boxSize: [6, 4, 6],
        capturePosition: [0, 2, 0],
    };
    const far = {
    boxPosition: [20, 2, 0], boxSize: [6, 4, 6],
        capturePosition: [20, 2, 0],
    };
    ed.setEnvironmentProbe("ENV_NEAR", near);
  ed.setEnvironmentProbe("ENV_FAR", far);
  $("btn-probes").click();
  $("probe-list").value = "ENV_NEAR";
  $("probe-list").dispatchEvent(new Event("change"));
  await settle("ENV_NEAR");
  // Nothing is asked to stay, so only the probe being worked on is drawn.
  const selectedOnly = { near: drawn("ENV_NEAR"), far: drawn("ENV_FAR") };

  // Always visible is per probe: ticking it for the far one has to leave it on
  // screen after the near one takes the selection back.
  $("probe-list").value = "ENV_FAR";
  $("probe-list").dispatchEvent(new Event("change"));
  await settle("ENV_FAR");
  $("probe-show").checked = true;
  $("probe-show").dispatchEvent(new Event("change"));
  $("probe-env").checked = true;
  $("probe-env").dispatchEvent(new Event("change"));
  await wait(300);
  const farFlags = { ...ed.environmentProbeOf("ENV_FAR") };
  const nearFlags = { ...ed.environmentProbeOf("ENV_NEAR") };

  $("probe-list").value = "ENV_NEAR";
  $("probe-list").dispatchEvent(new Event("change"));
  await settle("ENV_NEAR");
  await wait(300);
  const together = {
    near: drawn("ENV_NEAR"), far: drawn("ENV_FAR"),
    nearBox: material("BOX", "ENV_NEAR"), farBox: material("BOX", "ENV_FAR"),
    nearInner: material("INNER_BOX", "ENV_NEAR"), farInner: material("INNER_BOX", "ENV_FAR"),
    // Env faces is per probe too, and a probe that is on screen without the
    // selection is exactly the one worth putting its own cubemap on.
    nearFaces: faces("ENV_NEAR"), farFaces: faces("ENV_FAR"),
    // The checkboxes follow the probe, not the pane.
    showTicked: $("probe-show").checked, envTicked: $("probe-env").checked,
  };

  // Picking a probe that is only on screen because it was asked to stay has to
  // resolve to that probe, not to the selected one.
  const dimEntry = ed.entryOf("ENV_FAR");
  const dimPickable = !!node("BOX", "ENV_FAR")?.isPickable && !!dimEntry?.node;

  // An eye belongs to a probe too: pulling one room's influence box out of the
  // way is about that room, and must leave the neighbour it is being placed
  // against exactly as it was.
  const influenceOn = (probeId) => !!node("INFLUENCE_BOX", probeId)?.isEnabled();
  $("btn-probe-eye-influence").click();
  await wait(300);
  const eyes = {
    nearInfluence: influenceOn("ENV_NEAR"), farInfluence: influenceOn("ENV_FAR"),
    nearEye: $("btn-probe-eye-influence").getAttribute("aria-pressed"),
  };
  // And the eye follows the probe the pane moves to, rather than staying where
  // the last probe left it.
  $("probe-list").value = "ENV_FAR";
  $("probe-list").dispatchEvent(new Event("change"));
  await settle("ENV_FAR");
  await wait(200);
  eyes.farEye = $("btn-probe-eye-influence").getAttribute("aria-pressed");
  $("probe-list").value = "ENV_NEAR";
  $("probe-list").dispatchEvent(new Event("change"));
  await settle("ENV_NEAR");
  await wait(200);
  eyes.backOnNear = $("btn-probe-eye-influence").getAttribute("aria-pressed");
  $("btn-probe-eye-influence").click();
  await wait(200);

  // No Apply: a keystroke is the commit, and one visit to a field is one undo
  // entry however many characters it took.
  const before = ed.historyDepth().undo;
  $("probe-size-x").dispatchEvent(new Event("focus"));
  for (const value of ["1", "1.", "1.2", "1.25"]) {
    $("probe-size-x").value = value;
    $("probe-size-x").dispatchEvent(new Event("input"));
  }
  await wait(200);
  const typed = {
    size: ed.environmentProbeOf("ENV_NEAR").boxSize.join(),
    entries: ed.historyDepth().undo - before,
  };
  await ed.undo();
  await wait(300);
  const undone = ed.environmentProbeOf("ENV_NEAR").boxSize.join();
  // A restore empties the probe map before it refills it: the pane must not
  // take that instant as "your probe is gone" and come back on another one,
  // with the caret still in a field about to edit it.
  const undoKeptPick = $("probe-list").value;
  // Undo restores the whole record, so a probe asked to stay on screen must not
  // come down with it.
  const undoKeptFlags = ed.environmentProbeOf("ENV_FAR").alwaysVisible === true
    && drawn("ENV_FAR");

  const manifest = mf.buildManifest();
  const inProbes = JSON.stringify(manifest.environmentProbes);
  const prefs = JSON.stringify(manifest.editorPrefs?.probes || {});
  // A round-trip through the manifest has to put the same probes back up.
  ed.setEnvironmentProbeView("ENV_FAR", { alwaysVisible: false, envFaces: false });
  await wait(200);
  const cleared = drawn("ENV_FAR");
  ed.applyEditorPrefs(manifest.editorPrefs);
  await wait(300);
  const restored = { drawn: drawn("ENV_FAR"), faces: faces("ENV_FAR") };

  // The list is the tall thing in the pane, so it is what a taller window is
  // for: it has to take the room the window gains and give it back.
  const panel = document.querySelector("#probe-modal .panel");
  const listHeight = () => Math.round($("probe-list").getBoundingClientRect().height);
  const savedHeight = panel.style.height;
  panel.style.height = "460px";
  await wait(120);
  const shortList = listHeight();
  panel.style.height = "860px";
  await wait(120);
  const tallList = listHeight();
  panel.style.height = savedHeight;

  // Always visible means "while I am working on the probes", not "for ever".
  $("btn-probe-close").click();
  await wait(300);
  const left = ed.state.scene.meshes.filter((m) => m.name.startsWith("LOCAL_ENVIRONMENT_")).length
    + ed.state.scene.transformNodes.filter((n) => n.name.startsWith("LOCAL_ENVIRONMENT_")).length;
  ed.removeEnvironmentProbe("ENV_NEAR");
  ed.removeEnvironmentProbe("ENV_FAR");
  return {
    selectedOnly, farFlags, nearFlags, together, dimPickable, eyes,
    typed, undone, undoKeptFlags, undoKeptPick, inProbes, prefs, cleared, restored, left,
    shortList, tallList,
  };
});
check("only the selected probe is drawn until another is asked to stay",
  probeView.selectedOnly.near === true && probeView.selectedOnly.far === false
    && probeView.together.near === true && probeView.together.far === true,
  `${JSON.stringify(probeView.selectedOnly)} → ${JSON.stringify(probeView.together)}`);
check("Always visible and Env faces belong to a probe, not to the pane",
  probeView.farFlags.alwaysVisible === true && probeView.farFlags.envFaces === true
    && probeView.nearFlags.alwaysVisible !== true && probeView.nearFlags.envFaces !== true
    && probeView.together.showTicked === false && probeView.together.envTicked === false,
  `far=${JSON.stringify([probeView.farFlags.alwaysVisible, probeView.farFlags.envFaces])},`
  + ` near=${JSON.stringify([probeView.nearFlags.alwaysVisible, probeView.nearFlags.envFaces])},`
  + ` ticks=${JSON.stringify([probeView.together.showTicked, probeView.together.envTicked])}`);
check("the selected probe stays bright while the probes kept beside it go dim",
  /_BRIGHT_/.test(probeView.together.nearBox) && /_DIM_/.test(probeView.together.farBox)
    && /_BRIGHT_/.test(probeView.together.nearInner) && /_DIM_/.test(probeView.together.farInner)
    && probeView.dimPickable === true,
  `near=${probeView.together.nearBox}/${probeView.together.nearInner},`
  + ` far=${probeView.together.farBox}/${probeView.together.farInner},`
  + ` pickable=${probeView.dimPickable}`);
check("env faces are asked for per probe, including one kept on screen without the selection",
  probeView.together.farFaces === true && probeView.together.nearFaces === false,
  `far=${probeView.together.farFaces}, near=${probeView.together.nearFaces}`);
check("hiding one probe's influence box leaves the probe beside it alone",
  probeView.eyes.nearInfluence === false && probeView.eyes.farInfluence === true
    && probeView.eyes.nearEye === "false" && probeView.eyes.farEye === "true"
    && probeView.eyes.backOnNear === "false",
  `near=${probeView.eyes.nearInfluence}/${probeView.eyes.nearEye},`
  + ` far=${probeView.eyes.farInfluence}/${probeView.eyes.farEye},`
  + ` back on near=${probeView.eyes.backOnNear}`);
check("typing into a probe field commits, and one visit to it is one undo entry",
  probeView.typed.size === "1.25,4,6" && probeView.typed.entries === 1
    && probeView.undone === "6,4,6",
  `${JSON.stringify(probeView.typed)} → ${probeView.undone}`);
check("undo puts a probe record back without taking its boxes off screen",
  probeView.undoKeptFlags === true && probeView.undoKeptPick === "ENV_NEAR",
  `flags=${probeView.undoKeptFlags}, pane on ${probeView.undoKeptPick}`);
check("which probes are held on screen is saved beside the ship, not inside it",
  /"ENV_FAR":\{"alwaysVisible":true,"envFaces":true,"visibleParts":\{"box":true,"influence":true,"inner":true\}\}/
    .test(probeView.prefs)
    && !/alwaysVisible|envFaces|visibleParts/.test(probeView.inProbes)
    && probeView.cleared === false
    && probeView.restored.drawn === true && probeView.restored.faces === true,
  `prefs=${probeView.prefs}, cleared=${probeView.cleared},`
  + ` restored=${JSON.stringify(probeView.restored)}`);
check("closing the probes window takes every probe gizmo down with it",
  probeView.left === 0, `${probeView.left} nodes left`);
check("the probe list takes the room a taller window gains and gives it back",
  probeView.tallList - probeView.shortList >= 350 && probeView.shortList > 0,
  `${probeView.shortList}px at 460 → ${probeView.tallList}px at 860`);

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
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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

const enabledDoor = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  ed.clearAll();
  const d = mk.addDoor(new V(0, 0, 0), { chunkA: "CH00", chunkB: "CH01", silent: true });
  const byDefault = mf.buildManifest();

  ed.select([d.id]);
  document.getElementById("door-enabled").checked = false;
  document.getElementById("door-enabled").dispatchEvent(new Event("change", { bubbles: true }));
  const disabled = mf.buildManifest();

  const layout = ed.serialize();
  await ed.deserialize(JSON.parse(JSON.stringify(layout)));
  const reloaded = [...ed.state.markers.values()][0]?.enabled;
  ed.select([d.id]);
  await ed.undo();
  const afterUndo = [...ed.state.markers.values()][0]?.enabled;

  const out = {
    defaultDoor: byDefault.doors[0].enabled,
    defaultPortal: byDefault.portals[0].enabled,
    disabledDoor: disabled.doors[0].enabled,
    disabledPortal: disabled.portals[0].enabled,
    reloaded, afterUndo,
  };
  ed.clearAll(); ed.select([]);
  return out;
});
check("doors and their portals are enabled by default",
  enabledDoor.defaultDoor === true && enabledDoor.defaultPortal === true,
  JSON.stringify(enabledDoor));
check("the Enabled box reaches both manifest records",
  enabledDoor.disabledDoor === false && enabledDoor.disabledPortal === false,
  JSON.stringify(enabledDoor));
check("enabled survives a reload, and undo restores it",
  enabledDoor.reloaded === false && enabledDoor.afterUndo === true,
  JSON.stringify(enabledDoor));

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

// ---- 1d-quinvicies. a capsule is drawn as a capsule ------------------------
// A tube closed by two hemispheres, not a stretched sphere. Every other kind is
// one unit mesh under a scale; a capsule is not, because its caps are
// hemispheres of the tube's radius and a non-uniform scale turns them into an
// ellipsoid. The old unit mesh was `height: 1, radius: 0.5`, and Babylon's
// capsule height *includes* the caps - so `height - 2 x radius` left no tube at
// all and every capsule in the editor was a lozenge.
const capsProbe = await page.evaluate(async (sizes) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const mf = await import("/js/manifest.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  const out = { cases: [], armed: co.COLLIDER_DEFAULT_SCALE.capsule };
  for (const [d, h] of sizes) {
    const c = co.addCollider("capsule", new V(0, 0, 0), { silent: true, scale: [d, h, d] });
    ed.state.scene.render();                       // let the watcher settle the shape
    c.mesh.computeWorldMatrix(true);
    const raw = c.mesh.getVerticesData("position");
    const wm = c.mesh.getWorldMatrix();
    const pts = [];
    for (let k = 0; k < raw.length; k += 3)
      pts.push(V.TransformCoordinates(new V(raw[k], raw[k + 1], raw[k + 2]), wm));

    const s = c.node.scaling, rad = s.x / 2, half = Math.max(s.y / 2 - rad, 0);
    // distance from every vertex to the capsule's own segment, less the radius:
    // zero everywhere is the definition of the surface
    let worst = 0;
    for (const q of pts) {
      const y = Math.max(-half, Math.min(half, q.y));
      worst = Math.max(worst, Math.abs(Math.hypot(q.x, q.y - y, q.z) - rad));
    }
    const ys = pts.map((q) => q.y), xs = pts.map((q) => q.x), zs = pts.map((q) => q.z);
    const ring = (sign) => pts.filter((q) => Math.abs(q.y - sign * half) < 1e-4)
      .map((q) => Math.hypot(q.x, q.z));
    const rings = [...ring(1), ...ring(-1)];
    const wsm = wm.m;
    out.cases.push({
      asked: [d, h],
      scale: s.asArray().map((v) => +v.toFixed(4)),
      size: [Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys),
        Math.max(...zs) - Math.min(...zs)].map((v) => +v.toFixed(4)),
      offSurface: +worst.toFixed(5),
      tube: rings.length ? [Math.min(...rings), Math.max(...rings)].map((v) => +v.toFixed(5)) : null,
      // the mesh counter-scales its parent's Y, so what it ends up wearing is a
      // uniform scale - which is exactly why the caps stay round
      worldScale: [Math.hypot(wsm[0], wsm[1], wsm[2]), Math.hypot(wsm[4], wsm[5], wsm[6]),
        Math.hypot(wsm[8], wsm[9], wsm[10])].map((v) => +v.toFixed(4)),
    });
    co.removeCollider(c.id, true);
  }

  // and what Havok is told has to describe the same pill
  const cCap = co.addCollider("capsule", new V(0, 0, 0), { silent: true, scale: [1, 3, 1] });
  const cCyl = co.addCollider("cylinder", new V(6, 0, 0), { silent: true, scale: [1, 3, 1] });
  const capShapes = mf.buildManifest().collision[ed.state.activeChunk] || [];
  out.rec = Object.fromEntries(capShapes.map((s) => [s.kind, s]));
  ed.clearAll(); ed.select([]);
  return out;
}, [[1, 3], [1.6, 2.2], [1, 0.2]]);

for (const c of capsProbe.cases) {
  const [d, h] = c.asked;
  const wantH = Math.max(h, d);
  check(`a ${d} x ${h} capsule is ${d} wide and ${wantH} tall, caps included`,
    Math.abs(c.size[0] - d) < 1e-3 && Math.abs(c.size[2] - d) < 1e-3
      && Math.abs(c.size[1] - wantH) < 1e-3, `[${c.size}] from scale [${c.scale}]`);
  check(`every vertex of the ${d} x ${h} capsule lies on a true capsule`,
    c.offSurface < 3e-3, `worst off-surface = ${c.offSurface} m`);
  check(`the ${d} x ${h} capsule's mesh ends up uniformly scaled, so its caps are round`,
    new Set(c.worldScale).size === 1, `[${c.worldScale}]`);
  if (h > d) {
    check(`the ${d} x ${h} capsule has a straight tube of exactly its radius`,
      c.tube && Math.abs(c.tube[0] - d / 2) < 1e-3 && Math.abs(c.tube[1] - d / 2) < 1e-3,
      `tube radius ${c.tube} vs ${d / 2}`);
  }
}
// A capsule shorter than it is wide has no tube left, so it *is* a sphere -
// which is also the only thing Havok can make of it, its capsule being a
// segment plus a radius.
check("a capsule cannot be squashed below a sphere",
  capsProbe.cases[2].scale.join() === "1,1,1", `[${capsProbe.cases[2].scale}]`);
check("and it arms as a pill rather than as a sphere",
  capsProbe.armed[1] > capsProbe.armed[0], `[${capsProbe.armed}]`);

const capSeg = (s) => Math.hypot(s.pointB[0] - s.pointA[0], s.pointB[1] - s.pointA[1],
  s.pointB[2] - s.pointA[2]);
check("Havok's capsule segment is a diameter shorter than the drawn height",
  Math.abs(capSeg(capsProbe.rec.capsule) + 2 * capsProbe.rec.capsule.radius
    - capsProbe.rec.capsule.height) < 1e-6,
  `segment ${capSeg(capsProbe.rec.capsule).toFixed(3)} + 2r ${2 * capsProbe.rec.capsule.radius}`
  + ` vs height ${capsProbe.rec.capsule.height}`);
check("but a cylinder's segment is still its whole height",
  Math.abs(capSeg(capsProbe.rec.cylinder) - capsProbe.rec.cylinder.height) < 1e-6,
  `segment ${capSeg(capsProbe.rec.cylinder).toFixed(3)} vs height ${capsProbe.rec.cylinder.height}`);

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

  const W = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  const P = "Modular SciFi MegaKit/Platforms/Platform_3Plates";
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

// ---- 1d-sesquitricies. auto-save ------------------------------------------
// A recovery copy written *beside* the ship, never over it, and only when
// something has actually changed. It keeps its own baseline: a background write
// must not clear "you have unsaved work", because the ship you last chose to
// save still does not have those changes.
const auto = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const main = await import("/js/main.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  const i = await import("/js/interact.js");
  const co = await import("/js/colliders.js");
  i.cancelGhost(); co.exitCollisionMode(); ed.clearAll(); ed.select([]);
  ed.setConfig("autoSaveMinutes", 2);

  const dirtyNow = () => {
    const e = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(e);
    return e.defaultPrevented;
  };

  // The manifest is server state the whole run shares, so put back whatever is
  // there rather than leaving this block's ship behind for the next one.
  const manifestWas = await (await fetch("/api/layout")).json();
  main.markSaved();
  const quiet = await main.autoSaveNow();

  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const afterChange = await main.autoSaveNow();
  const file = await (await fetch("/api/autosave")).json();
  const stillWarns = dirtyNow();
  const again = await main.autoSaveNow();

  // and the manifest never moved
  const manifestNow = await (await fetch("/api/layout")).json();
  const manifestTouched = JSON.stringify(manifestNow.instances || [])
    !== JSON.stringify(manifestWas.instances || []);

  ed.setConfig("autoSaveMinutes", 0);
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(8, 0, 0), { silent: true });
  const whenOff = await main.autoSaveNow();

  const defaults = ed.CONFIG_DEFAULTS.autoSaveMinutes;
  ed.setConfig("autoSaveMinutes", 2);
  ed.clearAll(); ed.select([]);
  return {
    defaults, quiet, afterChange, again, whenOff, stillWarns, manifestTouched,
    marked: !!file.autoSaved, instances: (file.instances || []).length,
  };
});
check("auto-save defaults to two minutes", auto.defaults === 2, `${auto.defaults}`);
check("a tick with nothing changed writes nothing",
  auto.quiet.wrote === false && auto.quiet.armed === true, JSON.stringify(auto.quiet));
check("a tick after a change writes, and marks the file as an auto-save",
  auto.afterChange.wrote === true && auto.marked && auto.instances >= 1,
  `${JSON.stringify(auto.afterChange)}, ${auto.instances} instances`);
check("a second tick with nothing further changed writes nothing",
  auto.again.wrote === false, JSON.stringify(auto.again));
check("an auto-save never clears the unsaved-work warning",
  auto.stillWarns === true, `${auto.stillWarns}`);
check("and never writes over the ship you last saved",
  auto.manifestTouched === false, `manifest changed: ${auto.manifestTouched}`);
check("zero minutes turns it off, timer and tick alike",
  auto.whenOff.armed === false && auto.whenOff.wrote === false,
  JSON.stringify(auto.whenOff));
// ---- 1d-undetricies. the collision staging area ---------------------------
// A mode, not a property of the selection. It opens empty, you stage whatever
// modules you want to fit shapes to, and which element a shape belongs to is
// decided by where it sits - which is what makes copying a similar hull work.
const PROP = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  return [...kit.getCatalogue().byId.keys()].filter((id) => id.startsWith("Modular SciFi MegaKit/Props/"));
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  await ed.placeAt(a, new V(8, 0, 0), { silent: true });
  co.addCollider("box", new V(-6, 0.5, 0), { silent: true });
  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  return {
    mode: ed.state.mode === "collision",
    staged: [...ed.state.placements.values()].filter((p) => p.stage).length,
    shipShown: [...ed.state.placements.values()].filter((p) => !p.stage && p.node.isEnabled()).length,
        roomShapesShown: [...ed.state.colliders.values()].filter((c) => !c.stage && c.node.isEnabled()).length,
        // Capture all lives on the palette now, which is on screen in here too -
        // and the ship it would photograph is not. Disabled is only half of it:
        // the pane paints its own colours over the browser's greying, so a button
        // that is off has to be *seen* to be off or it reads as a dead editor.
        captureAll: document.getElementById("btn-capture-all").disabled,
        captureAllPaint: getComputedStyle(document.getElementById("btn-capture-all")).color,
        livePaint: getComputedStyle(document.getElementById("btn-cfg-reset")).color,
        captureAllCursor: getComputedStyle(document.getElementById("btn-capture-all")).cursor,
    };
}, PROP_A);
check("the collision area opens empty, with the ship off screen",
  areaOpen.mode && areaOpen.staged === 0 && areaOpen.shipShown === 0
    && areaOpen.roomShapesShown === 0, JSON.stringify(areaOpen)
);
check(
    "and Capture all is inert in here, and looks it, since a capture would photograph nothing",
    areaOpen.captureAll === true && areaOpen.captureAllPaint !== areaOpen.livePaint && areaOpen.captureAllCursor === "not-allowed",
    `disabled ${areaOpen.captureAll}, ${areaOpen.captureAllPaint} against a live ${areaOpen.livePaint},` + ` cursor ${areaOpen.captureAllCursor}`
);

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

// Fit a hull reads the element's own triangles rather than its bounding box,
// and lays out as many boxes as the shape asks for. It is measured through the
// saved records, not the live colliders: collider ids are recycled, so an id
// diff sees nothing at all when a fit replaces an existing hull.
const hullFit = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const V = BABYLON.Vector3;
  const out = {};
  ed.select([]);
  out.none = (await co.fitHullToSelection()).error;
  const list = [...ed.state.placements.values()].filter((p) => p.stage);
  ed.select(list.map((p) => p.id));
  out.many = (await co.fitHullToSelection()).error;
  const shape = co.addCollider("sphere", new V(320, 0, 0), { stage: true, silent: true });
  ed.select([shape.id]);
  out.onShape = (await co.fitHullToSelection()).error;
  co.removeCollider(shape.id, true);

  const target = list[0];
  ed.select([target.id]);
  const before = (ed.state.moduleCollision.get(target.module) || []).length;
  const r = await co.fitHullToSelection();
  out.ok = r.ok;
  out.module = target.module;
  out.how = r.how;
  out.coverage = r.coverage;
  out.confident = r.confident;
  const shapes = ed.state.moduleCollision.get(target.module) || [];
  out.recorded = shapes.length;
  out.made = r.colliders.length;
  out.kinds = [...new Set(shapes.map((s) => s.kind))];

  // compose the records onto the element, the way the runtime has to
  target.node.computeWorldMatrix(true);
  const parent = target.node.getWorldMatrix();
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const s of shapes) {
    const w = BABYLON.Matrix.Compose(
      new V(s.scale[0], s.scale[1], s.scale[2]),
      BABYLON.Quaternion.FromEulerAngles(...s.rotation.map((d) => (d * Math.PI) / 180)),
      new V(s.position[0], s.position[1], s.position[2])).multiply(parent);
    for (const sx of [-0.5, 0.5]) {
      for (const sy of [-0.5, 0.5]) {
        for (const sz of [-0.5, 0.5]) {
          const p = V.TransformCoordinates(new V(sx, sy, sz), w);
          for (const [k, v] of [[0, p.x], [1, p.y], [2, p.z]]) {
            if (v < min[k]) min[k] = v;
            if (v > max[k]) max[k] = v;
          }
        }
      }
    }
  }
  const mb = target.node.getHierarchyBoundingVectors(true);
  out.hull = { min, max };
  out.mesh = { min: [mb.min.x, mb.min.y, mb.min.z], max: [mb.max.x, mb.max.y, mb.max.z] };

  ed.undo();
  await new Promise((res) => setTimeout(res, 400));
  out.afterUndo = (ed.state.moduleCollision.get(target.module) || []).length;
  out.before = before;
  return out;
});
check("fitting a hull with nothing, or several, selected explains itself",
  /select the element/.test(hullFit.none) && /single element/.test(hullFit.many),
  `${hullFit.none} / ${hullFit.many}`);
check("fitting a hull onto a shape rather than an element explains itself",
  /collision shape/.test(hullFit.onShape), hullFit.onShape);
check("a fitted hull is recorded exactly as it was made",
  hullFit.ok && hullFit.recorded === hullFit.made && hullFit.made > 0
    && hullFit.kinds.length === 1 && hullFit.kinds[0] === "box",
  `${hullFit.made} made, ${hullFit.recorded} recorded, kinds ${hullFit.kinds}`);
check("a fitted hull covers the geometry it was fitted to",
  hullFit.coverage >= 0.97, `${(100 * hullFit.coverage).toFixed(1)}% of ${hullFit.module}`);
check("a fitted hull spans its element rather than landing elsewhere",
  [0, 1, 2].every((k) => hullFit.hull.min[k] <= hullFit.mesh.min[k] + 0.15
    && hullFit.hull.max[k] >= hullFit.mesh.max[k] - 0.15),
  `hull ${hullFit.hull.min.map((v) => v.toFixed(2))}..${hullFit.hull.max.map((v) => v.toFixed(2))}`
  + ` mesh ${hullFit.mesh.min.map((v) => v.toFixed(2))}..${hullFit.mesh.max.map((v) => v.toFixed(2))}`);
check("a confident hull is a tight one",
  !hullFit.confident || [0, 1, 2].every((k) =>
    hullFit.mesh.min[k] - hullFit.hull.min[k] < 0.35
    && hullFit.hull.max[k] - hullFit.mesh.max[k] < 0.35),
  `confident ${hullFit.confident}, by ${hullFit.how}`);
check("undoing a fitted hull puts the old one back",
  hullFit.afterUndo === hullFit.before, `${hullFit.recorded} -> ${hullFit.afterUndo},`
  + ` wanted ${hullFit.before}`);

// The two hull settings. The tolerance is the single dial for how finely a
// shape is approximated; the offset decides which side of the art the
// thickness goes. Both are measured through the saved records, composed onto
// the element, because that is what the runtime will do with them.
const hullDials = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const V = BABYLON.Vector3;
  const target = [...ed.state.placements.values()].find((p) => p.stage);
  ed.select([target.id]);

  /** Refit at these settings and measure the hull, along its own thin axis. */
  const at = async (cfg) => {
    for (const [k, v] of Object.entries(cfg)) ed.setConfig(k, v);
    const r = await co.fitHullToSelection();
    const shapes = ed.state.moduleCollision.get(target.module) || [];
    let vol = 0, escape = -Infinity, shift = [0, 0, 0], slack = [0, 0, 0];
    const pts = [];
    for (const m of target.node.getChildMeshes()) {
      const pos = m.getVerticesData && m.getVerticesData("position");
      if (!pos) continue;
      m.computeWorldMatrix(true);
      const w = m.getWorldMatrix().multiply(target.node.getWorldMatrix().clone().invert());
      for (let i = 0; i < pos.length; i += 3) {
        pts.push(V.TransformCoordinates(new V(pos[i], pos[i + 1], pos[i + 2]), w));
      }
    }
    for (const s of shapes) {
      vol += Math.abs(s.scale[0] * s.scale[1] * s.scale[2]);
      if (shapes.length !== 1) continue;
      const m = new BABYLON.Matrix();
      BABYLON.Matrix.FromQuaternionToRef(BABYLON.Quaternion.FromEulerAngles(
        ...s.rotation.map((d) => (d * Math.PI) / 180)), m);
      const ax = [0, 1, 2].map((k) => [m.m[k * 4], m.m[k * 4 + 1], m.m[k * 4 + 2]]);
      const half = [0, 1, 2].map((k) => Math.abs(s.scale[k]) / 2);
      // in the hull's own frame: where the art sits, and whether it fits
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const p of pts) {
        for (let k = 0; k < 3; k++) {
          const d = (p.x - s.position[0]) * ax[k][0] + (p.y - s.position[1]) * ax[k][1]
            + (p.z - s.position[2]) * ax[k][2];
          if (d < lo[k]) lo[k] = d;
          if (d > hi[k]) hi[k] = d;
          escape = Math.max(escape, Math.abs(d) - half[k]);
        }
      }
      // the hull's centre is the origin of that frame, so the art's centre is
      // exactly how far the hull was pushed off the art
      shift = [0, 1, 2].map((k) => +(-(lo[k] + hi[k]) / 2).toFixed(4));
      slack = [0, 1, 2].map((k) => +(2 * half[k] - (hi[k] - lo[k])).toFixed(3));
    }
    return { n: shapes.length, ok: r.ok, vol: +vol.toFixed(3),
      escape: +escape.toFixed(4), shift, slack };
  };

  const out = { module: target.module };
  out.coarse = await at({ hullTolerance: 0.3, hullThickness: 0.35, hullOffset: "centered" });
  out.fine = await at({ hullTolerance: 0.03 });
  // a thickness well past the art's own, so there is padding for the offset to
  // put somewhere - on a prop already thicker than the setting there is no
  // slack, and all three offsets are correctly the same hull
  out.mid = await at({ hullTolerance: 0.2, hullThickness: 1, hullOffset: "centered" });
  out.neg = await at({ hullOffset: "negative" });
  out.pos = await at({ hullOffset: "positive" });
  out.again = await at({ hullOffset: "negative" });
  out.rejected = ed.setConfig("hullOffset", "sideways");
  for (const k of ["hullTolerance", "hullThickness", "hullOffset"]) ed.resetConfig(k);
  return out;
});
check("a tighter hull tolerance buys a closer approximation",
  hullDials.fine.n > hullDials.coarse.n || hullDials.fine.vol < hullDials.coarse.vol,
  `${hullDials.module}: ${hullDials.coarse.n} box/${hullDials.coarse.vol} m3`
  + ` -> ${hullDials.fine.n} box/${hullDials.fine.vol} m3`);
const far = (v) => Math.max(...v.map(Math.abs));
check("centered leaves the hull sitting on the middle of the art",
  hullDials.mid.n !== 1 || far(hullDials.mid.shift) < 0.005,
  `shift ${JSON.stringify(hullDials.mid.shift)}, slack ${JSON.stringify(hullDials.mid.slack)}`);
check("an offset moves the hull off the art, by the slack it has to give",
  hullDials.neg.n !== 1
    || Math.abs(far(hullDials.neg.shift) - far(hullDials.neg.slack) / 2) < 0.01,
  `shift ${JSON.stringify(hullDials.neg.shift)}, slack ${JSON.stringify(hullDials.neg.slack)}`);
check("and the other offset is its mirror image",
  hullDials.neg.n !== 1 || [0, 1, 2].every((k) =>
    Math.abs(hullDials.pos.shift[k] + hullDials.neg.shift[k]) < 0.005),
  `${JSON.stringify(hullDials.pos.shift)} vs ${JSON.stringify(hullDials.neg.shift)}`);
check("an offset hull still contains its art",
  hullDials.neg.escape <= 1e-3 && hullDials.pos.escape <= 1e-3
    && hullDials.mid.escape <= 1e-3,
  `escape ${hullDials.mid.escape} / ${hullDials.neg.escape} / ${hullDials.pos.escape}`);
check("asking for the same offset twice does not move it twice",
  far(hullDials.again.shift.map((v, k) => v - hullDials.neg.shift[k])) < 0.005,
  `${JSON.stringify(hullDials.neg.shift)} then ${JSON.stringify(hullDials.again.shift)}`);
check("a setting that is a word rejects a word that is not on the list",
  hullDials.rejected === false, `setConfig returned ${hullDials.rejected}`);

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

// The bench has its own undo history. Its contents are deliberately not in
// serialize(), so a *ship* snapshot restores as "no bench at all" - which is
// exactly how Ctrl+Z used to wipe it.
const benchUndo = await page.evaluate(async (prop) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const el = [...ed.state.placements.values()].find((p) => p.stage && p.module === prop);
  const shipBefore = ed.state.placements.size;
  const shapesBefore = co.stageColliders().length;
  ed.select([el.id]);
  await co.fitBoxToSelection(kit.moduleBounds);
  const fitted = co.stageColliders().length;

  await ed.undo();
  const afterUndo = {
    staged: [...ed.state.placements.values()].filter((p) => p.stage).length,
    shapes: co.stageColliders().length,
    ship: ed.state.placements.size,
    mode: ed.state.mode === "collision",
  };
  await ed.redo();
  return { shipBefore, shapesBefore, fitted, afterUndo,
    afterRedo: co.stageColliders().length };
}, PROP_A);
check("Ctrl+Z on the bench undoes the edit and leaves the bench standing",
  benchUndo.afterUndo.mode && benchUndo.afterUndo.staged === 2
    && benchUndo.afterUndo.shapes === benchUndo.shapesBefore,
  JSON.stringify(benchUndo.afterUndo));
check("and never touches the ship",
  benchUndo.afterUndo.ship === benchUndo.shipBefore,
  `${benchUndo.shipBefore} -> ${benchUndo.afterUndo.ship}`);
check("redo puts the bench edit back",
  benchUndo.afterRedo === benchUndo.fitted,
  `${benchUndo.afterRedo} of ${benchUndo.fitted}`);

// Each side of the switch keeps its own viewpoint.
const views = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  // Aim by rotation, the way free look does. Position and a *target* was the
  // earlier shape of this, and getTarget() on a FreeCamera only reports the
  // last point something explicitly aimed it at - so the camera came back in
  // the right place looking the wrong way.
  const put = (p, r) => {
    const c = ed.state.camera;
    c.cameraDirection.setAll(0);
    c.cameraRotation.set(0, 0);
    c.position.set(...p);
    c.rotation.set(...r);
  };
  const at = () => {
    const c = ed.state.camera;
    return {
      pos: c.position.asArray().map((v) => +v.toFixed(2)),
      rot: [c.rotation.x, c.rotation.y, c.rotation.z].map((v) => +v.toFixed(3)),
    };
  };
  put([-7, 6, -9], [0.31, 0.82, 0]);
  const bench = at();
  co.exitCollisionMode();
  put([100, 40, -100], [0.44, -1.2, 0]);
  const ship = at();
  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  const backOnBench = at();
  co.exitCollisionMode();
  const backOnShip = at();
  await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
  return { bench, ship, backOnBench, backOnShip };
});
check("the bench keeps its own viewpoint across a switch, aim and all",
  JSON.stringify(views.backOnBench) === JSON.stringify(views.bench),
  `${JSON.stringify(views.bench)} -> ${JSON.stringify(views.backOnBench)}`);
check("and the ship keeps its own",
  JSON.stringify(views.backOnShip) === JSON.stringify(views.ship),
  `${JSON.stringify(views.ship)} -> ${JSON.stringify(views.backOnShip)}`);

// Ctrl+D on the bench: a shape copies, a module stand-in does not.
await page.mouse.move(collCanvas.x + 8, collCanvas.y + 8);   // selection, not hover
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  ed.select([co.stageColliders()[0].id]);
});
await page.keyboard.press("Control+d");
await page.waitForTimeout(500);
const dupShape = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const n = ed.hooks.ghostNode();
  const ms = n ? n.getChildMeshes() : [];
  return {
    kind: i.ghostCollider(), meshes: ms.length,
    // green, translucent, and edged - a bare mesh came out in the scene's
    // default grey, because ghostMaterialFor() returns null for no material
    green: ms.some((m) => (m.material?.emissiveColor?.g ?? 0) > 0.5
      && (m.material?.emissiveColor?.r ?? 1) < 0.4),
    clear: ms.every((m) => (m.material?.alpha ?? 1) < 1),
    edged: ms.some((m) => !!m.material?.wireframe),
  };
});
check("Ctrl+D on a shape arms a green, translucent, edged ghost",
  dupShape.kind === "box" && dupShape.meshes === 2 && dupShape.green
    && dupShape.clear && dupShape.edged, JSON.stringify(dupShape));
await page.evaluate(async () => (await import("/js/interact.js")).cancelGhost());
await page.waitForTimeout(200);

const dupStaged = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const el = [...ed.state.placements.values()].find((p) => p.stage);
  ed.select([el.id]);
  return [...ed.state.placements.values()].filter((p) => p.stage).length;
});
await page.keyboard.press("Control+d");
await page.waitForTimeout(500);
const benchRefused = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return {
    ghost: i.ghostActive(), brush: ed.state.brush,
    staged: [...ed.state.placements.values()].filter((p) => p.stage).length,
    status: document.getElementById("status-text").textContent,
  };
});
check("Ctrl+D on a staged module is refused, and says why",
  !benchRefused.ghost && !benchRefused.brush && benchRefused.staged === dupStaged
    && /only be on the bench once/.test(benchRefused.status),
  JSON.stringify(benchRefused));

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
    mode: ed.state.mode === "collision",
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
  return { ghost: i.ghostActive(), mode: ed.state.mode === "collision" };
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
    mode: ed.state.mode === "collision",
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

// The manifest must carry a module's hull once, not once per placement: the
// runtime instances it from `instances`, which already has every placement's
// module, chunk and transform. Expanding it here as well grew as
// placements x shapes rather than modules x shapes.
const noDupes = await page.evaluate(async ([a]) => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const co = await import("/js/colliders.js");
  const V = BABYLON.Vector3;
  co.exitCollisionMode();
  ed.clearAll(); ed.select([]);
  ed.loadModuleCollision({ [a]: [
    { kind: "box", position: [0, 0.5, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
  ] }, []);
  // six placements of the module, and one room shape that has no module at all
  for (let i = 0; i < 6; i++) await ed.placeAt(a, new V(i * 6, 0, 0), { silent: true });
  co.addCollider("sphere", new V(-8, 1, 0), { silent: true });

  const man = mf.buildManifest();
  const flat = Object.values(man.collision || {}).flat();
  const out = {
    placements: man.instances.filter((x) => x.module === a).length,
    inCollision: flat.length,
    kinds: flat.map((s) => s.kind),
    inModuleCollision: (man.moduleCollision[a] || []).length,
    inModuleShapes: (man.moduleShapes[a] || []).length,
    // everything the runtime needs to place the hull itself
    instanceHasAll: man.instances.every((x) =>
      x.module && x.chunk && x.position && x.rotation && x.scale),
  };
  ed.clearAll(); ed.select([]); ed.loadModuleCollision({}, []);
  return out;
}, [PROP_A]);
check("a module's hull is written once, not once per placement",
  noDupes.placements === 6 && noDupes.inModuleCollision === 1,
  `${noDupes.placements} placements, ${noDupes.inModuleCollision} record in moduleCollision`);
check("and collision[chunk] carries only the room's own shapes",
  noDupes.inCollision === 1 && noDupes.kinds.join() === "sphere",
  `${noDupes.inCollision} record(s): ${JSON.stringify(noDupes.kinds)}`);
check("the authoring form is there too, once per module",
  noDupes.inModuleShapes === 1, `${noDupes.inModuleShapes}`);
check("and every instance carries what it takes to place the hull",
  noDupes.instanceHasAll === true);

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

// ---- and a capsule hull is drawn as a capsule out on the ship ---------------
// A capsule is the one kind whose geometry depends on its proportions, so it
// counter-scales its own Y to keep its caps round. The preview wrote the size
// straight onto the *mesh*, which looked equivalent to what a real collider
// does and was not: it flattened that counter-scale, and every inherited
// capsule came out a sphere. The size goes on a holder now, as it does
// everywhere else.
const previewCapsule = await page.evaluate(async (prop) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const V = BABYLON.Vector3;
  const settle = () => new Promise((r) => setTimeout(r, 80));
  ed.clearAll(); ed.select([]);
  const measure = async (placementScale) => {
    ed.clearAll(); ed.select([]);
    ed.loadModuleCollision({ [prop]: [
      { kind: "capsule", position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 3, 1] },
    ] }, []);
    await ed.placeAt(prop, new V(0, 0, 0), { scale: placementScale, silent: true });
    ed.setShowLayer("both");
    ed.applyVisibility();
    await settle();
    ed.state.scene.render();
    await settle();
    ed.state.scene.render();
    // buildMesh suffixes the shape's name; the holder takes the plain one
    const mesh = ed.state.scene.meshes.find((m) => m.name.startsWith("PREVIEW_0_fill"));
    if (!mesh) return { missing: true };
    mesh.computeWorldMatrix(true);
    const raw = mesh.getVerticesData("position");
    const wm = mesh.getWorldMatrix();
    const c = wm.getRow(3);
    const origin = new V(c.x, c.y, c.z);
    const pts = [];
    for (let i = 0; i < raw.length; i += 3) {
      pts.push(V.TransformCoordinates(new V(raw[i], raw[i + 1], raw[i + 2]), wm).subtract(origin));
    }
    const xs = pts.map((q) => q.x), ys = pts.map((q) => q.y), zs = pts.map((q) => q.z);
    const width = Math.max(...xs) - Math.min(...xs);
    const depth = Math.max(...zs) - Math.min(...zs);
    const height = Math.max(...ys) - Math.min(...ys);
    const rad = width / 2, half = Math.max(height / 2 - rad, 0);
    let worst = 0;
    for (const q of pts) {
      const y = Math.max(-half, Math.min(half, q.y));
      worst = Math.max(worst, Math.abs(Math.hypot(q.x, q.y - y, q.z) - rad));
    }
    const wsm = wm.m;
    return {
      size: [width, height, depth].map((v) => +v.toFixed(3)),
      offSurface: +worst.toFixed(4),
      worldScale: [Math.hypot(wsm[0], wsm[1], wsm[2]), Math.hypot(wsm[4], wsm[5], wsm[6]),
        Math.hypot(wsm[8], wsm[9], wsm[10])].map((v) => +v.toFixed(3)),
      holder: mesh.parent?.getClassName?.() ?? null,
    };
  };
  const plain = await measure([1, 1, 1]);
  // A placement may be scaled unevenly - this ship has doors at [0.65,0.85,1] -
  // and a capsule composed with that is an ellipsoid, which Havok has no shape
  // for. The preview shows what Havok gets, not a promise it cannot keep.
  const squashed = await measure([0.5, 1, 1]);
  ed.clearAll(); ed.select([]);
  ed.loadModuleCollision({}, []);
  return { plain, squashed };
}, PROP_A);
check("an inherited capsule is drawn 1 wide and 3 tall, as its hull says",
  !previewCapsule.plain.missing
    && previewCapsule.plain.size.join() === "1,3,1", JSON.stringify(previewCapsule.plain.size));
check("every vertex of it lies on a true capsule, not a stretched sphere",
  previewCapsule.plain.offSurface < 3e-3, `worst ${previewCapsule.plain.offSurface} m`);
check("the size rides a holder, so the mesh keeps the counter-scale that rounds its caps",
  previewCapsule.plain.holder === "TransformNode"
    && new Set(previewCapsule.plain.worldScale).size === 1,
  `${previewCapsule.plain.holder}, world scale ${JSON.stringify(previewCapsule.plain.worldScale)}`);
check("and on an unevenly scaled placement it is still a capsule, radius averaged",
  previewCapsule.squashed.offSurface < 3e-3
    && previewCapsule.squashed.size[0] === previewCapsule.squashed.size[2]
    && Math.abs(previewCapsule.squashed.size[0] - 0.75) < 1e-3,
  `${JSON.stringify(previewCapsule.squashed.size)}, worst ${previewCapsule.squashed.offSurface}`);

// And it keeps up with the ship. It is drawn from each placement's world
// matrix and rebuilt only through applyVisibility(), which a move, a turn or a
// delete never calls - so the hulls sat where the elements used to be, and
// outlived the elements entirely.
const previewFollows = await page.evaluate(async (prop) => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  const frame = () => new Promise((r) =>
    ed.state.scene.onAfterRenderObservable.addOnce(() => r()));
  ed.clearAll(); ed.select([]);
  ed.loadModuleCollision({ [prop]: [
    { kind: "box", position: [0, 0.5, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
  ] }, []);
  const el = await ed.placeAt(prop, new V(0, 0, 0), { silent: true });
  ed.applyVisibility();
  await frame(); await frame();

  const shapes = () => ed.state.scene.transformNodes
    .find((n) => n.name === "__COLLISION_PREVIEW")?.getChildMeshes() || [];
  const nearest = () => {
    let d = Infinity;
    for (const m of shapes()) d = Math.min(d, V.Distance(m.getAbsolutePosition(), el.node.getAbsolutePosition()));
    return +d.toFixed(2);
  };
  const before = { n: shapes().length, d: nearest() };

  el.node.position.addInPlace(new V(30, 0, 20));
  await frame(); await frame(); await frame();
  const afterMove = nearest();

  el.node.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(0, Math.PI / 2, 0);
  await frame(); await frame(); await frame();
  const afterTurn = nearest();

  ed.select([el.id]);
  ed.removeSelected();
  await frame(); await frame(); await frame();
  const afterDelete = shapes().length;

  ed.clearAll(); ed.select([]); ed.loadModuleCollision({}, []);
  ed.applyVisibility();
  await frame(); await frame();
  return { before, afterMove, afterTurn, afterDelete };
}, PROP_A);
check("an inherited hull is drawn on its element",
  previewFollows.before.n > 0 && previewFollows.before.d < 1.5,
  `${previewFollows.before.n} meshes, ${previewFollows.before.d} m away`);
check("moving a ship element carries its inherited hull",
  previewFollows.afterMove < 1.5, `${previewFollows.afterMove} m away after a 36 m move`);
check("turning it keeps the hull with it",
  previewFollows.afterTurn < 1.5, `${previewFollows.afterTurn} m away after a 90 degree turn`);
check("deleting it takes the hull with it",
  previewFollows.afterDelete === 0, `${previewFollows.afterDelete} meshes left`);

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
  const target = [...kit.getCatalogue().byId.keys()].find((m) => m.startsWith("Modular SciFi MegaKit/Props/"));
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
  const wall = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 20), { silent: true });
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
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  const a = await ed.placeAt(M, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(M, new V(4, 0, 0), { silent: true });
  const c = await ed.placeAt(M, new V(8, 0, 0), { silent: true });
  ed.renamePlacement(a.id, "crate");
  ed.renamePlacement(b.id, "crate");       // same name on purpose
  ed.renamePlacement(c.id, "doorL");

  const blank = ed.setBehaviorDef("   ", { dynamic: true });
  const notObject = ed.setBehaviorDef("bad", [1, 2, 3]);
  let unknownDefinition = false;
  try {
    ed.setBehaviorDef("bad", { dynamic: true });
  } catch {
    unknownDefinition = true;
  }
  ed.setBehaviorDef("anyLiquefaction", { liquefiable: true });
  ed.setBehaviorDef("stdLiquefaction",
    { liquefiable: true, fluidSim: ["viscosity-inplace"] });
  ed.setBehaviorDef("dynamic", { dynamic: true });

  ed.addEntityBehavior("crate", "anyLiquefaction");
  // The same behaviour twice is legal - the runtime builds one instance per
  // entry - so the second attach lands beside the first, not on top of it.
  const twice = ed.addEntityBehavior("crate", "anyLiquefaction");
  const twiceListed = ed.entityBehaviors("crate").map((x) => x.name);
  ed.removeEntityBehavior("crate", 1);
  const afterOneRemoved = ed.entityBehaviors("crate").map((x) => x.name);
  const unknown = ed.addEntityBehavior("crate", "nope");
  ed.addEntityBehavior("doorL", "stdLiquefaction");
  ed.setEntityLinked("doorL", 0, ["crate", "doorL", "crate"]);
  ed.state.entities.get("doorL")[0].sound = "longSplash";

  const man = mf.buildManifest();
  const written = JSON.parse(JSON.stringify(man.behaviors));
  const entities = JSON.parse(JSON.stringify(man.entities));

  // deleting one has to strip it from the entities that carried it
  ed.deleteBehaviorDef("anyLiquefaction");
  const afterDelete = { list: ed.behaviorNames(), crate: ed.entityBehaviors("crate") };

  // round trip through the undo snapshot
  const snapshot = JSON.parse(JSON.stringify(ed.serialize()));
  ed.deleteBehaviorDef("stdLiquefaction");
  const wiped = ed.entityBehaviors("doorL").length;
  await ed.deserialize(snapshot);
  const restored = ed.entityBehaviors("doorL");

  const inRoom = ed.nodeNamesInChunk(ed.state.placements.get(c.id).chunk, "doorL");
  const liquefies = [ed.isLiquefiable("stdLiquefaction"), ed.isLiquefiable("dynamic")];
  const global = man.fluidSim;
  ed.clearAll(); ed.select([]);
  const afterClearAll = { defs: ed.behaviorNames(), ents: ed.entityBehaviors("crate") };
  return { blank, notObject, unknownDefinition, twice, twiceListed, afterOneRemoved, unknown, written,
    entities, afterDelete,
    wiped, restored, inRoom, liquefies, global, afterClearAll };
});
check("a definition needs a metadata name and a JSON object for a body",
  bhv.blank === false && bhv.notObject === false && bhv.unknownDefinition,
  `blank=${bhv.blank}, array=${bhv.notObject}, unknown=${bhv.unknownDefinition}`);
check("validated definition bodies are written unchanged",
  JSON.stringify(bhv.written.stdLiquefaction)
    === JSON.stringify({ liquefiable: true, fluidSim: ["viscosity-inplace"] }),
  JSON.stringify(bhv.written.stdLiquefaction));
check("entities list the behaviours they carry",
  bhv.entities.crate.behaviors[0].name === "anyLiquefaction"
    && !("linked" in bhv.entities.crate.behaviors[0]),
  JSON.stringify(bhv.entities.crate));
check("linked is de-duplicated, drops self, and is omitted when empty",
  bhv.entities.doorL.behaviors[0].linked.join() === "crate",
  JSON.stringify(bhv.entities.doorL.behaviors[0]));
check("per-entity sound categories are written beside the behavior name",
  bhv.entities.doorL.behaviors[0].sound === "longSplash",
  JSON.stringify(bhv.entities.doorL.behaviors[0]));
check("a behaviour can be attached twice and removed by position, but not when it does not exist",
  bhv.twice === true && bhv.unknown === false
    && bhv.twiceListed?.join() === "anyLiquefaction,anyLiquefaction"
    && bhv.afterOneRemoved?.join() === "anyLiquefaction",
  `twice=${bhv.twice}, listed=${JSON.stringify(bhv.twiceListed)},`
  + ` after remove=${JSON.stringify(bhv.afterOneRemoved)}, unknown=${bhv.unknown}`);
check("deleting a definition strips it from the entities that used it",
  !bhv.afterDelete.list.includes("anyLiquefaction") && bhv.afterDelete.crate.length === 0,
  `library [${bhv.afterDelete.list}], crate ${JSON.stringify(bhv.afterDelete.crate)}`);
check("per-entity sound categories survive undo snapshots",
  bhv.restored[0]?.sound === "longSplash",
  JSON.stringify(bhv.restored));
check("behaviours ride the undo stack",
  bhv.wiped === 0 && bhv.restored[0]?.name === "stdLiquefaction"
    && bhv.restored[0].linked.join() === "crate",
  `wiped=${bhv.wiped}, restored=${JSON.stringify(bhv.restored)}`);
check("only liquefiable definitions ask for linked names",
  bhv.liquefies[0] === true && bhv.liquefies[1] === false, JSON.stringify(bhv.liquefies));
check("the linked candidates are the other named nodes in the room",
  bhv.inRoom.join() === "crate", `[${bhv.inRoom}]`);
check("the manifest carries the global sim list",
  Array.isArray(bhv.global) && bhv.global.includes("liquid-slow"),
  JSON.stringify(bhv.global));
check("clearing the layout forgets the library and the entities",
  bhv.afterClearAll.defs.length === 0 && bhv.afterClearAll.ents.length === 0,
  JSON.stringify(bhv.afterClearAll));

// the ship's own sim list wins over config.json: config is what a *new* ship
// starts from, but a saved one carries the list it was authored against
const simList = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const fromConfig = [...ed.state.fluidSim];
  const base = { chunks: ["CH00_Storage"], activeChunk: "CH00_Storage", instances: [], markers: [] };

  let extensionError = "";
  try {
    await ed.deserialize({ ...base, fluidSim: ["ship-only.json"] });
  } catch (error) {
    extensionError = String(error);
  }
  await ed.deserialize({ ...base, fluidSim: ["ship-only", "thick"] });
  const loaded = [...ed.state.fluidSim];
  const written = mf.buildManifest().fluidSim;

  // an undo snapshot carries no list, and must not wipe the one in force
  await ed.deserialize(base);
  const afterUndo = [...ed.state.fluidSim];

  ed.state.fluidSim = fromConfig;
  ed.clearAll(); ed.select([]);
  return { fromConfig, loaded, written, afterUndo, extensionError };
});
check("fluidSim names omit the file extension",
  simList.extensionError.includes('fluidSim name "ship-only.json" must omit the .json extension'),
  simList.extensionError);
check("the manifest's fluidSim list is exported",
  simList.written.join() === "ship-only,thick", `[${simList.written}]`);
check("a loaded list takes precedence over the one from config.json",
  simList.loaded.join() === "ship-only,thick"
    && simList.fromConfig.join() !== simList.loaded.join(),
  `config [${simList.fromConfig}] -> ship [${simList.loaded}]`);
check("a snapshot without a list leaves the one in force alone",
  simList.afterUndo.join() === "ship-only,thick", `[${simList.afterUndo}]`);

// ---- 1d-quaterdecies. the behaviour panel and the library dialog ----------
// The panel edits the *node name* - the one in the field, or the element's id
// when the field is empty - not the element itself, so giving a second element
// the same name has to bring the first one's behaviours up on blur.
const bhvIds = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
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
  hasLinkedPicker: !!document.querySelector('#bhv-applied [data-behavior-key="linked"]'),
  addOff: document.getElementById("btn-bhv-add").disabled,
  count: document.getElementById("bhv-count").textContent,
  hint: document.getElementById("bhv-hint").textContent,
}));

// Behaviours open folded, so anything reaching into a form has to open its box
// first - which is what a user does too. Idempotent: a box the panel already
// opened (Add opens what it attaches) is left alone.
const unfoldBehavior = async (at = 0) => {
  await page.evaluate((i) => {
    const box = document.querySelector(`#bhv-applied .bhv-entry[data-entry="${i}"]`);
    if (box?.classList.contains("folded")) box.querySelector("[data-fold]").click();
  }, at);
  await page.waitForTimeout(80);
};

const unnamed = await readPanel();
check("an unnamed element carries behaviours under its id, and the panel says so",
  unnamed.shown && /unnamed, so it stands alone under its id/.test(unnamed.count)
    && new RegExp(`^${bhvIds.a} —`).test(unnamed.count),
  `count="${unnamed.count}", hint="${unnamed.hint}"`);

// Define two known behaviours through the metadata-driven dialog, exactly as a
// user would. Unknown names are refused because there is no schema to edit.
await page.fill("#insp-name", "crate");
await page.locator("#insp-name").blur();
await page.waitForTimeout(200);
await page.click("#btn-bhv-library");
await page.waitForTimeout(150);
await page.click("#btn-bhv-new");
await page.selectOption("#bhv-name", "stdLiquefaction");
await page.click("#btn-bhv-save");
await page.waitForTimeout(150);
await page.click("#btn-bhv-new");
await page.selectOption("#bhv-name", "dynamic");
await page.click("#btn-bhv-save");
await page.waitForTimeout(150);

// The library is a window, not a modal: definitions pile up in the order the
// ship was built, so the list has to sort them to be a list you can look
// something up in - and the window has to get out of the way of the ship it is
// describing, which means a titlebar to drag and a corner to pull.
const libBefore = await page.evaluate(() => {
  const win = document.getElementById("bhv-modal");
  const rect = win.getBoundingClientRect();
  return {
    listed: [...document.getElementById("bhv-list").options].map((o) => o.value),
    offered: [...document.getElementById("bhv-add").options].map((o) => o.value),
    resize: getComputedStyle(win.querySelector(".panel")).resize,
    left: rect.left, top: rect.top, width: rect.width, height: rect.height,
    fields: document.getElementById("bhv-fields").getBoundingClientRect().height,
    covers: rect.width >= window.innerWidth - 8,
  };
});
check("the library lists its definitions alphabetically, whatever order they were written in",
  libBefore.listed.join() === "dynamic,stdLiquefaction"
    && libBefore.offered.join() === "dynamic,stdLiquefaction",
  `list=[${libBefore.listed}], add=[${libBefore.offered}]`);

const bhvBar = await page.locator("#bhv-window-handle").boundingBox();
await page.mouse.move(bhvBar.x + 40, bhvBar.y + bhvBar.height / 2);
await page.mouse.down();
await page.mouse.move(bhvBar.x + 170, bhvBar.y + bhvBar.height / 2 + 70, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(100);
const libMoved = await page.evaluate(() => {
  const rect = document.getElementById("bhv-modal").getBoundingClientRect();
  return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
});
check("the library window is dragged by its titlebar, and does not cover the ship",
  !libBefore.covers
    && Math.abs(libMoved.left - libBefore.left - 130) < 2
    && Math.abs(libMoved.top - libBefore.top - 70) < 2
    && libMoved.width === libBefore.width,
  `(${libBefore.left},${libBefore.top}) -> (${libMoved.left},${libMoved.top})`);

// The grip is the browser's own, so this is also the check that the panel is
// the element carrying the size: pulling it has to take the window with it.
const bhvPanel = await page.locator("#bhv-modal .panel").boundingBox();
await page.mouse.move(bhvPanel.x + bhvPanel.width - 4, bhvPanel.y + bhvPanel.height - 4);
await page.mouse.down();
await page.mouse.move(bhvPanel.x + bhvPanel.width + 76, bhvPanel.y + bhvPanel.height + 54,
  { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(100);
const libResized = await page.evaluate(() => {
  const rect = document.getElementById("bhv-modal").getBoundingClientRect();
  return {
    width: rect.width, height: rect.height,
    fields: document.getElementById("bhv-fields").getBoundingClientRect().height,
  };
});
check("and pulled bigger from its corner, with the height going to the typed fields",
  libBefore.resize === "both"
    && libResized.width > libMoved.width + 60
    && libResized.height > libMoved.height + 40
    && libResized.fields > libBefore.fields + 40,
  `resize=${libBefore.resize}, ${libMoved.width}x${libMoved.height}`
  + ` -> ${libResized.width}x${libResized.height}, fields ${libBefore.fields} -> ${libResized.fields}`);

await page.click("#btn-bhv-close");
await page.waitForTimeout(200);
const defined = await page.evaluate(async () =>
  (await import("/js/editor.js")).behaviorNames());
check("the dialog creates typed definitions",
  defined.join() === "stdLiquefaction,dynamic", `[${defined}]`);

const named = await readPanel();
check("naming the element unlocks Add, and the element count is shown",
  !named.addOff && /"crate" — 1 element\b/.test(named.count), `"${named.count}"`);

await page.selectOption("#bhv-add", "stdLiquefaction");
await page.click("#btn-bhv-add");
await page.waitForTimeout(250);
const attached = await readPanel();
check("Add attaches the behaviour to the node name",
  attached.applied.join() === "stdLiquefaction", JSON.stringify(attached.applied));
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
await page.click('#bhv-applied [data-behavior-key="linked"] .behavior-array-add');
await page.waitForTimeout(250);
const linkedNow = await page.evaluate(async () => {
  const select = document.querySelector(
    '#bhv-applied [data-behavior-key="linked"] .behavior-array-row select');
  return {
    linked: (await import("/js/editor.js")).entityBehaviors("crate")[0].linked,
    options: [...(select?.options ?? [])].map((option) => option.value).filter(Boolean),
  };
});
check("the linked picker offers the other named nodes in the room",
  linkedNow.options.join() === "crateB", `[${linkedNow.options}]`);
check("picking a linked node stores it",
  linkedNow.linked.join() === "crateB", `[${linkedNow.linked}]`);

// a dynamic-only behaviour has nothing to link, so no picker
await page.selectOption("#bhv-add", "dynamic");
await page.click("#btn-bhv-add");
await page.waitForTimeout(250);
const both = await page.evaluate(() => ({
  applied: [...document.querySelectorAll("#bhv-applied .item .n")].map((e) => e.textContent),
  pickers: document.querySelectorAll('#bhv-applied [data-behavior-key="linked"]').length,
}));
check("only the liquefiable one gets a linked picker",
  both.applied.join() === "stdLiquefaction,dynamic" && both.pickers === 1,
  `${JSON.stringify(both.applied)}, ${both.pickers} picker(s)`);

// Each behaviour is one box you can fold shut, and what it is for is on its
// name rather than printed above its fields.
const folded = await page.evaluate(async () => {
  const boxes = () => [...document.querySelectorAll("#bhv-applied .bhv-entry")];
  const shut = () => boxes().map((box) => box.classList.contains("folded"));
  const foldAll = document.getElementById("btn-bhv-fold");
  const opened = shut();
  const tips = [...document.querySelectorAll("#bhv-applied .item .n")].map((n) => n.title);
  const prose = document.querySelectorAll("#bhv-applied .behavior-description").length;
  boxes()[0].querySelector("[data-fold]").click();
  const one = shut();
  const hidden = getComputedStyle(boxes()[0].querySelector("[data-behavior-form]")).display;
  const label = foldAll.textContent;
  foldAll.click();
  const all = shut();
  const backLabel = foldAll.textContent;

  // A fold is how the panel is being looked at, so it has to survive the panel
  // rebuilding itself - which it does after every edit.
  const ed = await import("/js/editor.js");
  const id = [...ed.state.placements.values()].find((p) => p.name === "crate").id;
  ed.select([]); ed.select([id]);
  await new Promise((r) => setTimeout(r, 50));
  const rebuilt = shut();
  document.getElementById("btn-bhv-fold").click();
  return { opened, one, all, rebuilt, back: shut(), tips, prose, hidden, label, backLabel };
});
check("a behaviour folds on its own and every one folds at once",
  folded.opened.join() === "false,false" && folded.one.join() === "true,false"
    && folded.hidden === "none" && folded.all.join() === "true,true"
    && folded.back.join() === "false,false",
  JSON.stringify(folded));
check("the fold-all button says what it will do next",
  folded.label === "Fold all" && folded.backLabel === "Unfold all", JSON.stringify(folded));
check("a fold survives the panel rebuilding itself",
  folded.rebuilt.join() === "true,true", JSON.stringify(folded.rebuilt));
check("a behaviour's description is on its name, not above its fields",
  folded.prose === 0 && /liquef/i.test(folded.tips[0] ?? ""), JSON.stringify(folded.tips));

// Folded is the default - an element answers "what do you do" as a list of
// names, not as every form it carries - and Add is the one exception.
const foldDefault = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const shut = () => [...document.querySelectorAll("#bhv-applied .bhv-entry")]
    .map((box) => box.classList.contains("folded"));
  const idOf = (name) => [...ed.state.placements.values()].find((p) => p.name === name).id;
  const settle = () => new Promise((r) => setTimeout(r, 60));

  ed.addEntityBehavior("crateB", "stdLiquefaction");
  ed.select([]); ed.select([idOf("crateB")]);
  await settle();
  const attached = shut();

  document.getElementById("bhv-add").value = "dynamic";
  document.getElementById("btn-bhv-add").click();
  await settle();
  const afterAdd = shut();

  // Removing the one above it has to take the folds down with it: they are
  // kept by position, and the box that stays open must be the same box.
  // The strip is the handle, not just the arrow, so the name folds it too.
  document.querySelector("#bhv-applied .bhv-entry .item .n").click();
  await settle();
  const afterStrip = shut();
  document.querySelector("#bhv-applied .bhv-entry .item .n").click();
  await settle();
  const afterStripBack = shut();

  document.querySelector("#bhv-applied [data-remove='0']").click();
  await settle();
  const afterRemove = shut();

  ed.removeEntityBehavior("crateB", 0);
  ed.select([]); ed.select([idOf("crate")]);
  await settle();
  return { attached, afterAdd, afterStrip, afterStripBack, afterRemove };
});
check("a behaviour opens folded, and only the one just added opens",
  foldDefault.attached.join() === "true" && foldDefault.afterAdd.join() === "true,false",
  JSON.stringify(foldDefault));
check("the whole title strip folds it, and Remove still removes",
  foldDefault.afterStrip.join() === "false,false"
    && foldDefault.afterStripBack.join() === "true,false",
  JSON.stringify(foldDefault));
check("removing a behaviour slides the folds under it up with it",
  foldDefault.afterRemove.join() === "false", JSON.stringify(foldDefault.afterRemove));

// isDynamicNode: only `dynamic: true` counts - liquefiable no longer implies it
const dyn = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  ed.addEntityBehavior("crateB", "dynamic");
  ed.select([]);
  ed.select([[...ed.state.placements.values()].find((p) => p.name === "crate").id]);
  return {
    dynamic: ed.isDynamicNode("crateB"),
    unknown: ed.isDynamicNode("nothing"),
    dynamic: mf.buildManifest().entities.crateB.behaviors.find((b) => b.name === "dynamic"),
  };
});
check("`dynamic: true` makes a node dynamic",
  dyn.dynamic && !dyn.unknown, JSON.stringify(dyn));
check("an empty linked list is left out of the manifest",
  !("linked" in dyn.dynamic), JSON.stringify(dyn.dynamic));

await page.click("#bhv-applied [data-remove='1']");
await page.waitForTimeout(250);
const removed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { names: ed.entityBehaviors("crate").map((b) => b.name),
    // stdLiquefaction is liquefiable, which no longer implies dynamic
    stillDynamic: ed.isDynamicNode("crate") };
});
check("Remove detaches it again",
  removed.names.join() === "stdLiquefaction", `[${removed.names}]`);
check("liquefiable alone does not make a node dynamic",
  !removed.stillDynamic, JSON.stringify(removed));

// Assignment parameters are typed overrides. The definition stays untouched,
// and the manifest keeps the same wire shape.
// A splash category is chosen from the ones the weapon defines rather than
// typed, so one has to exist first: a category the weapon does not know plays
// nothing at runtime, which is exactly what the list is there to prevent.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.setBehaviorDef("weaponLiquefactor", { sounds: { longSplash: ["waterLongSplash"] } });
  const id = [...ed.state.placements.values()].find((p) => p.name === "crate").id;
  ed.select([]);
  ed.select([id]);
});
await page.waitForTimeout(250);
await page.waitForTimeout(200);
await unfoldBehavior(0);
await page.selectOption('#bhv-applied [data-behavior-key="sound"] select', "longSplash");
await page.waitForTimeout(250);
const paramsStored = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const b = ed.entityBehaviors("crate").find((x) => x.name === "stdLiquefaction");
  return {
    entity: b,
    definition: ed.getBehaviorDef("stdLiquefaction"),
    written: mf.buildManifest().entities.crate.behaviors,
  };
});
const paramsWritten = paramsStored.written.find((b) => b.name === "stdLiquefaction");
check("typed parameters are stored on the applied behaviour, not the definition",
  paramsStored.entity?.sound === "longSplash"
    && paramsStored.definition?.sound === undefined,
  JSON.stringify(paramsStored.entity));
check("typed parameters reach the unchanged manifest shape",
  paramsWritten?.sound === "longSplash" && paramsWritten?.linked?.join() === "crateB",
  JSON.stringify(paramsWritten));

// Vector controls still edit in editor space, while the manifest writes glTF
// space. The flip must be its own inverse, or undo would mirror the facing.
const dirRoundTrip = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.setBehaviorDef("player", {});
  ed.addEntityBehavior("crate", "player");
  return ed.entityBehaviors("crate").findIndex((b) => b.name === "player");
});
await page.waitForTimeout(200);
await unfoldBehavior(dirRoundTrip);
const directionInputs = page.locator(
  `[data-behavior-form="${dirRoundTrip}"] [data-behavior-key="direction"] .behavior-vector input`);
await directionInputs.nth(0).fill("-1");
await directionInputs.nth(0).blur();
await page.waitForTimeout(150);
const directionRoundTrip = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const before = ed.entityBehaviors("crate").find((b) => b.name === "player");
  const snapshot = JSON.parse(JSON.stringify(ed.serialize()));
  await ed.deserialize(snapshot);
  const after = ed.entityBehaviors("crate").find((b) => b.name === "player");
  // a restore clears the selection; put it back so the panel below still has
  // something to render
  const crate = [...ed.state.placements.values()].find((p) => p.name === "crate");
  if (crate) ed.select([crate.id]);
  return {
    before,
    snapshot: snapshot.entities.crate.behaviors.find((b) => b.name === "player"),
    after,
  };
});
check("a direction survives undo without mirroring itself",
  directionRoundTrip.before.direction.join() === "-1,0,0"
    && directionRoundTrip.snapshot.direction.join() === "1,0,0"
    && directionRoundTrip.after?.direction.join() === "-1,0,0",
  `editor ${directionRoundTrip.before.direction} -> stored ${directionRoundTrip.snapshot.direction}`
  + ` -> back ${directionRoundTrip.after?.direction}`);

// isProbeExcludedNode: what an environment probe must not photograph. Five
// separate reasons, because they are five separate facts about an element -
// and one of them is matched by name, since the weapon behaviour's contract
// with the runtime is its name, not a flag in its body.
const probeOut = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.setBehaviorDef("probeExcluded", { reflectionProbe: "exclude" });
  ed.setBehaviorDef("weaponLiquefactor", { sounds: { fire: ["zap"] } });
  ed.setBehaviorDef("weaponAntiGravityGun", {});
  ed.setBehaviorDef("playAnimation", {});
  ed.setBehaviorDef("dynamic", { dynamic: true });
  ed.setBehaviorDef("anyLiquefaction", { liquefiable: true });
  ed.setBehaviorDef("enableEntity", {});
  ed.addEntityBehavior("optedOut", "probeExcluded");
  ed.addEntityBehavior("gun", "weaponLiquefactor");
  ed.addEntityBehavior("gravityGun", "weaponAntiGravityGun");
  ed.addEntityBehavior("fan", "playAnimation");
  ed.addEntityBehavior("barrel", "dynamic");
  ed.addEntityBehavior("icicle", "anyLiquefaction");
  ed.addEntityBehavior("plainDoor", "enableEntity");
  // the last one carries a harmless behaviour first, so this also proves the
  // test is over the whole list rather than just the first entry
  ed.addEntityBehavior("plainDoor", "probeExcluded");
  return {
    optedOut: ed.isProbeExcludedNode("optedOut"),
    gun: ed.isProbeExcludedNode("gun"),
    gravityGun: ed.isProbeExcludedNode("gravityGun"),
    fan: ed.isProbeExcludedNode("fan"),
    barrel: ed.isProbeExcludedNode("barrel"),
    icicle: ed.isProbeExcludedNode("icicle"),
    second: ed.isProbeExcludedNode("plainDoor"),
    plain: ed.isProbeExcludedNode("crateB2"),
    unnamed: ed.isProbeExcludedNode(""),
  };
});
check(
    "a probe leaves out opt-outs, weapons, animations, dynamics and liquefiables",
    probeOut.optedOut && probeOut.gun && probeOut.gravityGun && probeOut.fan && probeOut.barrel && probeOut.icicle,
    JSON.stringify(probeOut)
);
check("any behaviour in the list is enough, not just the first", probeOut.second, JSON.stringify(probeOut));
check("everything else stays in the probe", !probeOut.plain && !probeOut.unnamed, JSON.stringify(probeOut));

// ---- hideEntity: what the game hides before the player arrives -------------
// `hideEntity` with no parameters is entity-toggle.ts's "hide me, now", so the
// Runtime view has to draw the ship without it and a probe capture must not
// photograph it. With an events subscription it waits instead. Run behaviours
// governs both.
await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.clearAll();
    ed.select([]);
    await ed.deserialize({
        chunks: ["CH00_Storage"],
        activeChunk: "CH00_Storage",
        markers: [],
        instances: [
            { id: "T1", module: "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", chunk: "CH00_Storage", name: "trap", position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
            { id: "T2", module: "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", chunk: "CH00_Storage", name: "plainWall", position: [4, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
            { id: "T3", module: "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", chunk: "CH00_Storage", name: "trapTrigger", position: [8, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
        ],
        behaviors: { hideEntity: {} },
        entities: {
            trap: { behaviors: [{ name: "hideEntity" }] },
            trapTrigger: { behaviors: [{ name: "hideEntity", events: [{ name: "activated", source: "plainWall" }] }] },
        },
    });
});
await page.waitForTimeout(2000);

// One reader for the whole block: what is on screen, and what a capture of a
// box around all three would render AND digest - meshesInProbeBox is the one
// list both are built from, and it answers with nothing outside the Runtime
// view because there is no preview to photograph.
const hideState = () =>
    page.evaluate(async () => {
        const ed = await import("/js/editor.js");
        const rt = await import("/js/runtime.js");
        const inBox = new Set(rt.meshesInProbeBox([4, 1, 0], [40, 20, 40]).map((m) => m.uniqueId));
        const out = { shown: {}, inProbe: {} };
        for (const p of ed.shipPlacements()) {
            out.shown[p.name] = p.node.isEnabled();
            out.inProbe[p.name] = p.node.getChildMeshes().some((m) => inBox.has(m.uniqueId));
        }
        return out;
    });

const hideRule = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    return {
        bare: ed.behaviorParams({ name: "hideEntity" }),
        linkedEmpty: ed.behaviorParams({ name: "hideEntity", linked: [] }),
        linkedFull: ed.behaviorParams({ name: "hideEntity", linked: ["a"] }),
        withEvent: ed.behaviorParams({ name: "hideEntity", events: [{ name: "activated", source: "plainWall" }] }),
        hidden: ed.behaviorHiddenPlacements().map((p) => p.name),
    };
});
check(
    "an empty parameters box is what says 'hide me' — name and an empty linked are not parameters",
    Object.keys(hideRule.bare).length === 0 && Object.keys(hideRule.linkedEmpty).length === 0
        && Object.keys(hideRule.linkedFull).length === 1 && Object.keys(hideRule.withEvent).length === 1
        && hideRule.hidden.join() === "trap",
    JSON.stringify(hideRule)
);

const hideEditor = await hideState();
check(
    "the editor view still draws it — a trap you cannot see is one you cannot move",
    hideEditor.shown.trap && hideEditor.shown.plainWall && hideEditor.shown.trapTrigger,
    JSON.stringify(hideEditor.shown)
);

await page.selectOption("#view-mode", "runtime");
await page.waitForFunction(
    async () => {
        const ed = await import("/js/editor.js");
        return ed.state.runtime === true && !document.getElementById("view-mode").disabled;
    },
    null,
    { timeout: 60000 }
);
await page.waitForTimeout(500);

const hideRun = await hideState();
check(
    "the runtime view takes it off screen, and leaves the rest of the room alone",
    !hideRun.shown.trap && hideRun.shown.plainWall && hideRun.shown.trapTrigger,
    JSON.stringify(hideRun.shown)
);
check(
    "and a probe neither renders nor digests it",
    !hideRun.inProbe.trap && hideRun.inProbe.plainWall && hideRun.inProbe.trapTrigger,
    JSON.stringify(hideRun.inProbe)
);
check(
    "an event-driven hideEntity waits, so its own element stays",
    hideRun.shown.trapTrigger && hideRun.inProbe.trapTrigger,
    JSON.stringify(hideRun)
);

await page.uncheck("#run-behaviors");
await page.waitForTimeout(400);
const hideOff = await hideState();
const hideOffStatus = await page.textContent("#status-text");
check(
    "Run behaviours off puts it back — on screen, in the render list and in the digest",
    hideOff.shown.trap && hideOff.inProbe.trap,
    JSON.stringify(hideOff)
);
check("and the status line says how many came back", /1 element\(s\) no longer hidden/.test(hideOffStatus), hideOffStatus);

await page.check("#run-behaviors");
await page.waitForTimeout(400);
const hideOn = await hideState();
const hideOnStatus = await page.textContent("#status-text");
check("checking it hides them again", !hideOn.shown.trap && !hideOn.inProbe.trap, JSON.stringify(hideOn));
check("and the status line counts them", /1 element\(s\) hidden by hideEntity/.test(hideOnStatus), hideOnStatus);

// Editing the assignment is what turns one reading into the other, and the
// view has to follow it without a mode switch.
await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.setEntityParams("trap", 0, { events: [{ name: "activated", source: "plainWall" }] });
});
await page.waitForTimeout(400);
const hideParamed = await hideState();
check(
    "giving it an event subscription delays the action, and the element comes straight back",
    hideParamed.shown.trap && hideParamed.inProbe.trap,
    JSON.stringify(hideParamed)
);

await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.setEntityParams("trap", 0, {});
});
await page.waitForTimeout(400);
const hideBlanked = await hideState();
check("and emptying the box hides it again", !hideBlanked.shown.trap, JSON.stringify(hideBlanked.shown));

await page.selectOption("#view-mode", "editor");
await page.waitForFunction(
    async () => {
        const ed = await import("/js/editor.js");
        return ed.state.runtime === false && !document.getElementById("view-mode").disabled;
    },
    null,
    { timeout: 60000 }
);
await page.waitForTimeout(400);
const hideExit = await hideState();
check(
    "leaving the runtime view shows it again, setting or no setting",
    hideExit.shown.trap && (await page.isChecked("#run-behaviors")),
    JSON.stringify(hideExit.shown)
);

// ---- Run behaviours: the runtime view plays what the game would play -------
// A `playAnimation` element is animated on screen the way `play-animation.ts`
// animates it - same clip, same loop rule - but only in the Runtime view, and
// only while the setting is on. Everything below drives the real controls.
await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.clearAll();
    ed.select([]);
    await ed.deserialize({
        chunks: ["CH00_Storage"],
        activeChunk: "CH00_Storage",
        markers: [],
        instances: [
            { id: "F1", module: "Modular SciFi MegaKit/Props/Prop_Fan_Small", chunk: "CH00_Storage", name: "fanA", position: [0, 2, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
            { id: "F2", module: "Modular SciFi MegaKit/Props/Prop_Fan_Small", chunk: "CH00_Storage", name: "fanB", position: [4, 2, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
            {
                id: "W9",
                module: "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
                chunk: "CH00_Storage",
                name: "stillWall",
                position: [8, 0, 0],
                rotation: [0, 0, 0],
                scale: [1, 1, 1],
            },
        ],
        behaviors: { playAnimation: {} },
        entities: {
            fanA: { behaviors: [{ name: "playAnimation" }] },
            fanB: { behaviors: [{ name: "playAnimation" }] },
            stillWall: { behaviors: [{ name: "playAnimation" }] },
        },
    });
});
await page.waitForTimeout(3000);

// One reader for the whole block: which clips exist, whether they are running,
// and where the animated bones are standing right now.
const readAnims = () =>
    page.evaluate(async () => {
        const ed = await import("/js/editor.js");
        const out = {};
        for (const p of ed.shipPlacements()) {
            out[p.name] = {
                clips: (p.node._shipAnimationGroups || []).map((g) => ({
                    name: g.name,
                    playing: !!g.isPlaying,
                    loop: !!g.loopAnimation,
                })),
                pose: (p.node._shipAnimationNodes || []).map((n) =>
                    (n.rotationQuaternion || n.rotation)
                        .asArray()
                        .map((v) => +v.toFixed(4))
                        .join(",")
                ),
            };
        }
        return out;
    });

const animIdle = await readAnims();
check(
    "an animated module gets one clip, a static one none",
    animIdle.fanA.clips.length === 1 && animIdle.fanA.clips[0].name === "Fan_Idle" && animIdle.stillWall.clips.length === 0,
    JSON.stringify(animIdle.fanA.clips) + " / " + JSON.stringify(animIdle.stillWall.clips)
);
check("nothing plays while the editor view is up", !animIdle.fanA.clips[0].playing && !animIdle.fanB.clips[0].playing, JSON.stringify(animIdle.fanA.clips));

await page.selectOption("#view-mode", "runtime");
await page.waitForFunction(
    async () => {
        const ed = await import("/js/editor.js");
        return ed.state.runtime === true && !document.getElementById("view-mode").disabled;
    },
    null,
    { timeout: 60000 }
);
await page.waitForTimeout(500);

const animRun = await readAnims();
check(
    "the runtime view spins every fan, looping",
    animRun.fanA.clips[0].playing && animRun.fanB.clips[0].playing && animRun.fanA.clips[0].loop && animRun.fanB.clips[0].loop,
    JSON.stringify(animRun.fanA.clips) + " / " + JSON.stringify(animRun.fanB.clips)
);

await page.waitForTimeout(900);
const animMoved = await readAnims();
check(
    "the bones actually turn, not merely report as playing",
    animMoved.fanA.pose.join() !== animIdle.fanA.pose.join(),
    `${animIdle.fanA.pose.join(" ")} → ${animMoved.fanA.pose.join(" ")}`
);

// Held still while the exporter reads the ship: the GLB records each node's
// transform as it stands, so a fan caught halfway round would be written out
// as the authored rest pose.
const animExport = await page.evaluate(async () => {
    const mf = await import("/js/manifest.js");
    const ed = await import("/js/editor.js");
    const realExport = BABYLON.GLTF2Export.GLBAsync;
    const realFetch = window.fetch;
    let sawPlaying = null;
    let sawPose = null;
    BABYLON.GLTF2Export.GLBAsync = async () => {
        const fan = ed.shipPlacements().find((p) => p.name === "fanA");
        sawPlaying = fan.node._shipAnimationGroups.some((g) => g.isPlaying);
        sawPose = fan.node._shipAnimationNodes.map((n) =>
            (n.rotationQuaternion || n.rotation)
                .asArray()
                .map((v) => +v.toFixed(4))
                .join(",")
        );
        return { glTFFiles: { "ship.glb": new Blob([new Uint8Array(4)]) } };
    };
    window.fetch = async (url, opts) => {
        if (opts?.method === "POST" && String(url).includes("/api/export")) {
            return new Response(JSON.stringify({ ok: true, bytes: 4 }), { status: 200, headers: { "Content-Type": "application/json" } });
        }
        return realFetch(url, opts);
    };
    try {
        await mf.exportGlb();
    } finally {
        BABYLON.GLTF2Export.GLBAsync = realExport;
        window.fetch = realFetch;
    }
    await new Promise((r) => setTimeout(r, 400));
    const fan = ed.shipPlacements().find((p) => p.name === "fanA");
    return { sawPlaying, sawPose, after: fan.node._shipAnimationGroups.some((g) => g.isPlaying) };
});
check(
    "an export is taken with the animations held at their first frame",
    animExport.sawPlaying === false && animExport.sawPose.join() === animIdle.fanA.pose.join(),
    `${JSON.stringify(animExport.sawPlaying)} — ${animExport.sawPose?.join(" ")}`
);
check("and playback comes back once the export is done", animExport.after === true, JSON.stringify(animExport));

await page.uncheck("#run-behaviors");
await page.waitForTimeout(400);
const animOff = await readAnims();
const animOffStatus = await page.textContent("#status-text");
await page.waitForTimeout(600);
const animSettled = await readAnims();
check(
    "unchecking Run behaviours stops them and rewinds the bones",
    !animOff.fanA.clips[0].playing &&
        !animOff.fanB.clips[0].playing &&
        animOff.fanA.pose.join() === animIdle.fanA.pose.join() &&
        animSettled.fanA.pose.join() === animOff.fanA.pose.join(),
    `${animOff.fanA.pose.join(" ")} vs authored ${animIdle.fanA.pose.join(" ")}`
);
check("and says so on the status line", /behaviours off/.test(animOffStatus), animOffStatus);

await page.check("#run-behaviors");
await page.waitForTimeout(400);
const animOnStatus = await page.textContent("#status-text");
check("checking it starts them again, and counts them", (await readAnims()).fanA.clips[0].playing && /2 animation\(s\) playing/.test(animOnStatus), animOnStatus);

// The assignment's own JSON is the config, exactly as the runtime reads it.
await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.setEntityParams("fanA", 0, { loop: false });
    ed.setEntityParams("fanB", 0, { animation: "Fan_Idle" });
});
await page.waitForTimeout(400);
const animParams = await readAnims();
check(
    "loop and animation come from the assignment",
    animParams.fanA.clips[0].loop === false && animParams.fanB.clips[0].playing && animParams.fanB.clips[0].loop === true,
    JSON.stringify(animParams.fanA.clips) + " / " + JSON.stringify(animParams.fanB.clips)
);

await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.setEntityParams("fanB", 0, { animation: "NoSuchClip" });
});
await page.waitForTimeout(400);
const animMissing = await readAnims();
check(
    "a clip that does not exist stops that one and leaves the rest running",
    !animMissing.fanB.clips[0].playing && animMissing.fanA.clips[0].playing,
    JSON.stringify(animMissing.fanB.clips)
);

await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.setEntityParams("fanA", 0, {});
    ed.setEntityParams("fanB", 0, {});
    ed.state.mode = "collision";
    ed.emit("mode");
});
await page.waitForTimeout(400);
const animBench = await readAnims();
await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.state.mode = "ship";
    ed.emit("mode");
});
await page.waitForTimeout(400);
const animShip = await readAnims();
check(
    "a bench holds them still, and coming back to the ship releases them",
    !animBench.fanA.clips[0].playing && animShip.fanA.clips[0].playing,
    JSON.stringify(animBench.fanA.clips) + " → " + JSON.stringify(animShip.fanA.clips)
);

await page.selectOption("#view-mode", "editor");
await page.waitForFunction(
    async () => {
        const ed = await import("/js/editor.js");
        return ed.state.runtime === false && !document.getElementById("view-mode").disabled;
    },
    null,
    { timeout: 60000 }
);
await page.waitForTimeout(400);
const animExit = await readAnims();
check(
    "leaving the runtime view stops them, setting or no setting",
    !animExit.fanA.clips[0].playing && !animExit.fanB.clips[0].playing && (await page.isChecked("#run-behaviors")),
    JSON.stringify(animExit.fanA.clips)
);

// An editor preference: saved with the ship, restored on load, and reset by
// the Settings pane's Reset like the two beside it.
const animPrefs = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const on = ed.serializeEditorPrefs().runBehaviors;
    ed.state.runBehaviors = false;
    const off = ed.serializeEditorPrefs().runBehaviors;
    ed.applyEditorPrefs({ runBehaviors: true });
    const restored = ed.state.runBehaviors;
    ed.state.runBehaviors = false;
    document.getElementById("btn-cfg-reset").click();
    await new Promise((r) => setTimeout(r, 200));
    return {
        on,
        off,
        restored,
        afterReset: ed.state.runBehaviors,
        box: document.getElementById("run-behaviors").checked,
        default: ed.RUN_BEHAVIORS_DEFAULT,
    };
});
check(
    "Run behaviours is saved, restored and reset like the settings beside it",
    animPrefs.on === true && animPrefs.off === false && animPrefs.restored === true && animPrefs.afterReset === true && animPrefs.box === true && animPrefs.default === true,
    JSON.stringify(animPrefs)
);

// The library button moved out of the inspector, where it only appeared when
// exactly one element was selected, into the pane that is always on screen.
const animButton = await page.evaluate(() => {
    const btn = document.getElementById("btn-bhv-library");
    return {
        inInspector: !!btn.closest("#behavior-fields"),
        section: btn.closest("#settings-pane .settings-section")?.querySelector("h3")?.textContent,
    };
});
await page.click("#btn-bhv-library");
await page.waitForTimeout(300);
const animLibraryOpen = await page.isVisible("#bhv-modal");
await page.click("#btn-bhv-close");
await page.waitForTimeout(200);
check(
    "Edit behaviours… sits in Settings → Runtime and still opens the library",
    !animButton.inInspector && animButton.section === "Runtime" && animLibraryOpen,
    JSON.stringify(animButton) + ` open=${animLibraryOpen}`
);

await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.clearAll();
    ed.select([]);
});
await page.waitForTimeout(300);

// ---- Start demo: publish the saved ship, then open it ----------------------
// The button runs `sync-ship.ts` server-side and opens the demo page. Both are
// stubbed here: the real script writes into lab/public/aquanova, which a test
// run must no more touch than it touches a real export folder. What is checked
// is everything on this side of the wire - which flag the request carries, that
// the tab is opened only on success, and that the failure is readable.
await page.evaluate(() => {
    window.__demo = { posts: [], opened: [], reply: { ok: true, code: 0, optimized: false, output: "copied 3 files\n", url: "http://localhost:5174/lite/demo-aquanova.html" } };
    window.__realFetch = window.fetch;
    window.fetch = (url, opts) => {
        if (String(url).includes("/api/sync-ship")) {
            window.__demo.posts.push({ method: opts?.method, body: JSON.parse(opts?.body || "{}") });
            return Promise.resolve(new Response(JSON.stringify(window.__demo.reply),
                { status: 200, headers: { "Content-Type": "application/json" } }));
        }
        return window.__realFetch(url, opts);
    };
    window.__realOpen = window.open;
    window.open = (u, name) => { window.__demo.opened.push([u, name]); return {}; };
});

// Off by default, and off is what the script has to be *told* - its own default
// is to optimize - so the flag being absent from the request is the bug this
// check exists for.
const demoOff = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    document.getElementById("btn-demo").click();
    await new Promise((r) => setTimeout(r, 500));
    return {
        posts: window.__demo.posts.slice(),
        opened: window.__demo.opened.slice(),
        status: document.getElementById("status-text").textContent,
        box: document.getElementById("ship-optimize").checked,
        setting: ed.state.shipOptimize,
        default: ed.SHIP_OPTIMIZE_DEFAULT,
    };
});
check(
    "Start demo publishes unoptimized by default, and opens the demo in a named tab",
    demoOff.posts.length === 1 && demoOff.posts[0].method === "POST" && demoOff.posts[0].body.optimize === false
        && demoOff.opened.length === 1 && demoOff.opened[0][0] === "http://localhost:5174/lite/demo-aquanova.html"
        && demoOff.opened[0][1] === "aquanova-demo"
        && demoOff.box === false && demoOff.setting === false && demoOff.default === false,
    JSON.stringify(demoOff)
);
check("and says which of the two it published", /published \(unoptimized\)/.test(demoOff.status), demoOff.status);

await page.check("#ship-optimize");
await page.waitForTimeout(200);
const demoOn = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    window.__demo.posts.length = 0;
    window.__demo.opened.length = 0;
    window.__demo.reply = { ...window.__demo.reply, optimized: true };
    document.getElementById("btn-demo").click();
    await new Promise((r) => setTimeout(r, 500));
    return {
        posts: window.__demo.posts.slice(),
        status: document.getElementById("status-text").textContent,
        setting: ed.state.shipOptimize,
    };
});
check(
    "ticking Optimize ship asks for the compressed publish instead",
    demoOn.setting === true && demoOn.posts.length === 1 && demoOn.posts[0].body.optimize === true
        && /published \(optimized\)/.test(demoOn.status),
    JSON.stringify(demoOn)
);

// A script that failed must not be followed by a tab: an opened demo is a
// claim that what it loads is the ship you just published.
const demoFailed = await page.evaluate(async () => {
    window.__demo.posts.length = 0;
    window.__demo.opened.length = 0;
    window.__demo.reply = { ok: false, code: 1, output: "Error: toktx not found\n", url: "http://localhost:5174/lite/demo-aquanova.html" };
    document.getElementById("btn-demo").click();
    await new Promise((r) => setTimeout(r, 500));
    return { opened: window.__demo.opened.length, status: document.getElementById("status-text").textContent };
});
check(
    "a failed publish opens nothing and says so, pointing at the script's output",
    demoFailed.opened === 0 && /demo NOT started/.test(demoFailed.status)
        && /exited with code 1/.test(demoFailed.status) && /console/.test(demoFailed.status),
    JSON.stringify(demoFailed)
);

// A server that cannot run the script at all reports its own reason, not an
// exit code it never got.
const demoNoScript = await page.evaluate(async () => {
    window.__demo.opened.length = 0;
    window.__demo.reply = { ok: false, error: "node_modules/tsx not found — run pnpm install in the repository" };
    document.getElementById("btn-demo").click();
    await new Promise((r) => setTimeout(r, 500));
    return { opened: window.__demo.opened.length, status: document.getElementById("status-text").textContent };
});
check(
    "and a server that cannot run it at all quotes its own reason",
    demoNoScript.opened === 0 && /tsx not found/.test(demoNoScript.status)
        && !/exited with code/.test(demoNoScript.status),
    JSON.stringify(demoNoScript)
);

await page.evaluate(() => {
    window.fetch = window.__realFetch;
    window.open = window.__realOpen;
    delete window.__realFetch;
    delete window.__realOpen;
    delete window.__demo;
});

// An editor preference like Run behaviours: saved with the ship, restored on
// load, off the undo stack and put back by Reset to defaults.
const demoPrefs = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    ed.state.shipOptimize = true;
    const on = ed.serializeEditorPrefs().shipOptimize;
    ed.applyEditorPrefs({ shipOptimize: false });
    const restored = ed.state.shipOptimize;
    const undoable = "shipOptimize" in JSON.parse(JSON.stringify(ed.serialize()));
    ed.state.shipOptimize = true;
    document.getElementById("btn-cfg-reset").click();
    await new Promise((r) => setTimeout(r, 200));
    return {
        on,
        restored,
        undoable,
        afterReset: ed.state.shipOptimize,
        box: document.getElementById("ship-optimize").checked,
    };
});
check(
    "Optimize ship is saved, restored, reset — and never on the undo stack",
    demoPrefs.on === true && demoPrefs.restored === false && demoPrefs.undoable === false
        && demoPrefs.afterReset === false && demoPrefs.box === false,
    JSON.stringify(demoPrefs)
);

const demoSection = await page.evaluate(() => {
    const box = document.getElementById("ship-optimize");
    return {
        section: box.closest("#settings-pane .settings-section")?.querySelector("h3")?.textContent,
        inToolbar: !!document.getElementById("btn-demo").closest("#toolbar"),
        afterLoad: document.getElementById("btn-demo").previousElementSibling?.id,
    };
});
check(
    "the button sits in the toolbar beside Load, and its setting under Demo",
    demoSection.section === "Demo" && demoSection.inToolbar && demoSection.afterLoad === "btn-load",
    JSON.stringify(demoSection)
);

// ---- 1d-quaterdecies-bis. a door carries behaviours under its id -----------
// A door is a marker, not a placement, so the panel used to skip it entirely -
// yet the manifest writes marker and element behaviours to the same `entities`
// map. Its key is its id and can be nothing else, which is why a door has no
// Name field: the id is what the portal graph and the runtime's door events
// already refer to.
const doorBhv = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  ed.state.chunks = ["CH_DA", "CH_DB", "CH_DC"];
  ed.state.activeChunk = "CH_DA";
  ed.setBehaviorDef("stdLiquefaction", { liquefiable: true });
  // one named element per room, to prove the picker reads *both* sides
  const a = await ed.placeAt(M, new V(0, 0, 0), { chunk: "CH_DA", name: "leafL", silent: true });
  const b = await ed.placeAt(M, new V(6, 0, 0), { chunk: "CH_DB", name: "leafR", silent: true });
  await ed.placeAt(M, new V(30, 0, 0), { chunk: "CH_DC", name: "elsewhere", silent: true });
  const door = mk.addDoor(new V(3, 0, 0),
    { chunkA: "CH_DA", chunkB: "CH_DB", leaves: [a.id, b.id], silent: true });
  return { door: door.id, key: ed.entityNameOf(door), name: door.name ?? null };
});
await page.evaluate((id) => import("/js/editor.js").then((ed) => ed.select([id])), doorBhv.door);
await page.waitForTimeout(200);

const doorPanel = await readPanel();
check("the behaviour panel opens on a door, keyed by its id",
  doorPanel.shown && doorBhv.key === doorBhv.door && doorBhv.name === null
    && new RegExp(`^${doorBhv.door} — a door, so it stands alone under its id$`)
      .test(doorPanel.count) && !doorPanel.addOff,
  `key="${doorBhv.key}", count="${doorPanel.count}"`);

const doorNameRow = await page.evaluate(() =>
  document.getElementById("insp-name").parentElement.hidden);
check("a door is deliberately not nameable", doorNameRow === true, String(doorNameRow));

// attach through the UI, exactly as a user would
await page.selectOption("#bhv-add", "stdLiquefaction");
await page.click("#btn-bhv-add");
await page.waitForTimeout(200);
await page.click('#bhv-applied [data-behavior-key="linked"] .behavior-array-add');
await page.waitForTimeout(150);
const doorAttached = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const id = ed.state.selection[0];
  return {
    applied: ed.entityBehaviors(id).map((b) => b.name),
    candidates: (() => {
      const select = document.querySelector(
        '#bhv-applied [data-behavior-key="linked"] .behavior-array-row select');
      return [...(select?.options ?? [])].map((option) => option.value).filter(Boolean);
    })(),
    exported: Object.keys(mf.buildManifest().entities || {}),
  };
});
check("a behaviour attached to a door is exported under the door's id",
  doorAttached.applied.join() === "stdLiquefaction"
    && doorAttached.exported.includes(doorBhv.door),
  JSON.stringify(doorAttached.applied) + " -> " + JSON.stringify(doorAttached.exported));
check("the linked picker offers both of the door's sides, and nothing else",
  doorAttached.candidates.slice().sort().join() === "leafL,leafR",
  JSON.stringify(doorAttached.candidates));

// A door id is a slot the editor hands back out, so deleting one has to take
// its behaviours with it - or the next door placed would silently inherit them.
const doorGone = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const V = BABYLON.Vector3;
  const id = ed.state.selection[0];
  ed.select([id]);
  ed.removeSelected();
  const afterDelete = ed.state.entities.has(id);
  const reused = mk.addDoor(new V(3, 0, 0), { silent: true });
  const inherited = ed.entityBehaviors(reused.id).map((b) => b.name);
  ed.select([reused.id]); ed.removeSelected();
  // addDoor is silent, so it left no entry of its own: the two removals are the
  // whole stack, and undoing both is what puts the original door back.
  await ed.undo(); await ed.undo();
  return { id, afterDelete, reusedId: reused.id, inherited,
    afterUndo: ed.entityBehaviors(id).map((b) => b.name) };
});
check("deleting a door drops its behaviours, and the reused id inherits nothing",
  doorGone.afterDelete === false && doorGone.reusedId === doorGone.id
    && doorGone.inherited.length === 0,
  JSON.stringify(doorGone));
check("undo brings a deleted door's behaviours back",
  doorGone.afterUndo.join() === "stdLiquefaction", JSON.stringify(doorGone.afterUndo));

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
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
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
  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0),
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

// Showing the axes says which space you are thinking in, so moving and turning
// follow it. You press Shift+X to see which way the element's own X grows
// *because* you are about to work along it. Hiding says nothing, and Y still
// overrides afterwards.
const axesSetSpace = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  ed.select([a.id]);
  ed.state.camera.position = new V(4, 6, -7);
  ed.state.camera.setTarget(new V(0, 0.5, 0));
  i.setAxisSpace("world");
  const key = (shift) => window.dispatchEvent(new KeyboardEvent("keydown",
    { key: shift ? "X" : "x", code: "KeyX", shiftKey: shift, bubbles: true }));
  const now = () => ({ space: ed.state.axisSpace, combo: document.getElementById("axis-space").value,
    drawn: ed.axesSpace(), shown: ed.axesTarget() });

  const start = now();
  key(true);  const toLocal = now();          // Shift+X -> local axes, local space
  key(false); const toWorld = now();          // X re-aims -> world, world space
  key(false); const hidden = now();           // same key again hides
  i.setAxisSpace("local");
  key(false); const afterHiddenThenShow = now();
  // and hiding must not touch it
  i.setAxisSpace("local");
  const beforeHide = ed.state.axisSpace;
  key(false); const afterHide = now();
  ed.hideAxes(); ed.clearAll(); ed.select([]);
  i.setAxisSpace("world");
  return { start, toLocal, toWorld, hidden, afterHiddenThenShow, beforeHide, afterHide };
});
check("Shift+X shows local axes and puts moving and turning in local space",
  axesSetSpace.start.space === "world" && axesSetSpace.toLocal.drawn === "local"
    && axesSetSpace.toLocal.space === "local" && axesSetSpace.toLocal.combo === "local",
  `${axesSetSpace.start.space} -> ${axesSetSpace.toLocal.space},`
  + ` gizmo ${axesSetSpace.toLocal.drawn}`);
check("and X re-aims them to world and takes the space with it",
  axesSetSpace.toWorld.drawn === "world" && axesSetSpace.toWorld.space === "world"
    && axesSetSpace.toWorld.combo === "world",
  `${axesSetSpace.toLocal.space} -> ${axesSetSpace.toWorld.space}`);
check("hiding them says nothing about which space you want",
  axesSetSpace.afterHide.shown === null
    && axesSetSpace.afterHide.space === axesSetSpace.beforeHide,
  `${axesSetSpace.beforeHide} -> ${axesSetSpace.afterHide.space}`);

// ---- the gizmo hangs on the middle of the mesh; Ctrl asks for the origin ----
// A kit's authors put the node origin wherever suited the export, and on this
// kit that is regularly nowhere near the piece: a short wall's visible mesh is
// centred 2.2 m from its own origin, so the arrows floated clear of the thing
// they describe - "some meshes are quite off-centred from the origin, making it
// difficult to see the axis". The bare keys now measure what you can see and
// hang the gizmo on the middle of it. The origin stays reachable on Ctrl: it is
// the number the inspector's Position field writes and the point a snap lands
// on, so it is still worth being able to look at.
const anchorKeys = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const i = await import("/js/interact.js");
    const V = BABYLON.Vector3;
    i.cancelGhost();
    ed.clearAll();
    ed.select([]);
    const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(3, 0, 0), { silent: true });
    ed.select([a.id]);
    ed.state.camera.position = new V(8, 8, -12);
    ed.state.camera.setTarget(new V(3, 0.5, 0));

    const b = ed.worldBounds(a.node);
    const centre = b.min.add(b.max).scale(0.5);
    const origin = a.node.getWorldMatrix().getRow(3).toVector3();
    const key = (opts) => {
        window.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyX", bubbles: true, ...opts }));
        ed.state.scene.render();
        const n = ed.state.scene.getTransformNodeByName("AXES");
        return {
            anchor: ed.axesAnchor(),
            space: ed.axesSpace(),
            shown: ed.axesTarget(),
            offCentre: n ? +n.position.subtract(centre).length().toFixed(3) : null,
            offOrigin: n ? +n.position.subtract(origin).length().toFixed(3) : null,
        };
    };

    const plain = key({ key: "x" });
    const ctrl = key({ key: "x", ctrlKey: true });
    const shift = key({ key: "X", shiftKey: true });
    const ctrlShift = key({ key: "X", shiftKey: true, ctrlKey: true });
    const again = key({ key: "X", shiftKey: true, ctrlKey: true });

    // and the anchor is rigid: it is held as an offset in the element's own
    // frame, so a turn carries it round with the model rather than letting it
    // swim across the mesh while the world-aligned box grows and shrinks
    ed.showAxes(a.id, "world", "centre");
    ed.state.scene.render();
    const at = () => ed.state.scene.getTransformNodeByName("AXES").position.subtract(a.node.getWorldMatrix().getRow(3).toVector3());
    const rel = at();
    a.node.rotationQuaternion = (a.node.rotationQuaternion || BABYLON.Quaternion.Identity()).multiply(BABYLON.Quaternion.RotationAxis(BABYLON.Axis.Y, Math.PI / 2));
    a.node.computeWorldMatrix(true);
    ed.state.scene.render();
    // a 90-degree Y turn sends (x, z) to (z, -x)
    const drift = +at()
        .subtract(new V(rel.z, rel.y, -rel.x))
        .length()
        .toFixed(3);

    ed.hideAxes();
    ed.clearAll();
    ed.select([]);
    i.setAxisSpace("world");
    return { apart: +centre.subtract(origin).length().toFixed(3), plain, ctrl, shift, ctrlShift, again, drift };
});
check("the module used here really does have its origin off the mesh", anchorKeys.apart > 1, `${anchorKeys.apart} m apart`);
check(
    "X hangs the gizmo on the middle of the visible mesh",
    anchorKeys.plain.anchor === "centre" && anchorKeys.plain.space === "world" && anchorKeys.plain.offCentre < 0.01,
    `${anchorKeys.plain.anchor}, ${anchorKeys.plain.offCentre} m off the centre`
);
check(
    "Ctrl+X moves it to the node's own origin rather than hiding it",
    anchorKeys.ctrl.anchor === "origin" && anchorKeys.ctrl.space === "world" && anchorKeys.ctrl.shown !== null && anchorKeys.ctrl.offOrigin < 0.01,
    `${anchorKeys.ctrl.anchor}, ${anchorKeys.ctrl.offOrigin} m off the origin`
);
check(
    "Shift+X and Ctrl+Shift+X are the same pair in local space",
    anchorKeys.shift.anchor === "centre" &&
        anchorKeys.shift.space === "local" &&
        anchorKeys.shift.offCentre < 0.01 &&
        anchorKeys.ctrlShift.anchor === "origin" &&
        anchorKeys.ctrlShift.space === "local" &&
        anchorKeys.ctrlShift.offOrigin < 0.01,
    `Shift+X ${anchorKeys.shift.anchor}/${anchorKeys.shift.space},` + ` Ctrl+Shift+X ${anchorKeys.ctrlShift.anchor}/${anchorKeys.ctrlShift.space}`
);
check("and the same key twice still hides them", anchorKeys.again.shown === null, `${anchorKeys.again.shown}`);
check("the centre anchor turns with the element instead of swimming across it", anchorKeys.drift < 0.01, `${anchorKeys.drift} m adrift after a 90-degree turn`);

// ---- and the arms are the same length on every element ----------------------
// The length used to be taken from the element's own bounding span, clamped
// between 1.5 m and 8 m, so a prop and a platform got gizmos several times
// apart: "it looks like the size of the axis coordinates depends on the mesh
// size: please use a fixed size for all meshes". An arm is a direction
// indicator, and the ruler for the snap step printed along it, so it has to
// mean the same thing wherever it is hung.
const gizmoSize = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const i = await import("/js/interact.js");
    const V = BABYLON.Vector3;
    i.cancelGhost();
    ed.clearAll();
    ed.select([]);

    // every length on the gizmo is a fixed multiple of the arm length, so the
    // tip cube's local z reads it straight back out
    const armLength = () => +(ed.state.scene.getMeshByName("AXES_x_scale").position.z / 1.04).toFixed(3);

    const on = async (module, at) => {
        const e = await ed.placeAt(module, at, { silent: true });
        const b = ed.worldBounds(e.node);
        ed.showAxes(e.id, "world", "centre");
        ed.state.scene.render();
        return {
            span: +Math.max(b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z).toFixed(2),
            arm: armLength(),
        };
    };

    const small = await on("Modular SciFi MegaKit/Props/Prop_Light_Corner", new V(0, 0, 0));
    const big = await on("Modular SciFi MegaKit/Platforms/Platform_3Plates", new V(24, 0, 0));

    // and growing one does not grow its arms either - the very span that used to
    // drive the length, multiplied by five on a single element
    const huge = [...ed.state.placements.values()].find((p) => p.id === ed.axesTarget());
    huge.node.scaling.setAll(5);
    huge.node.computeWorldMatrix(true);
    ed.showAxes(huge.id, "world", "centre");
    ed.state.scene.render();
    const hb = ed.worldBounds(huge.node);
    const scaled = {
        span: +Math.max(hb.max.x - hb.min.x, hb.max.y - hb.min.y, hb.max.z - hb.min.z).toFixed(2),
        arm: armLength(),
    };

    ed.hideAxes();
    ed.clearAll();
    ed.select([]);
    return { small, big, scaled };
});
check(
    "scaling the element 5x really does change the span that used to size them",
    gizmoSize.scaled.span > gizmoSize.big.span * 4,
    `${gizmoSize.big.span} m -> ${gizmoSize.scaled.span} m`
);
check(
    "the axis arms are the same fixed length on both modules and when 5x scaled",
    gizmoSize.small.arm === gizmoSize.big.arm && gizmoSize.scaled.arm === gizmoSize.small.arm && Math.abs(gizmoSize.small.arm - 2) < 1e-6,
    `${gizmoSize.small.span} m module ${gizmoSize.small.arm} m, ${gizmoSize.big.span} m module ` +
        `${gizmoSize.big.arm} m, ${gizmoSize.scaled.span} m element ${gizmoSize.scaled.arm} m`
);

// ---- except right up close, where they halve, and halve again ----------------
// The arms are world geometry, so leaning in to seat something against a wall
// puts 2 m of arrow across the whole viewport and buries the detail being
// aimed. Inside 5 m of the point the gizmo hangs on they drop to half length,
// and inside 2 m to a quarter - steps, not a ramp, so an arm is still a ruler
// you can read the snap step off.
const gizmoNear = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const i = await import("/js/interact.js");
    const V = BABYLON.Vector3;
    i.cancelGhost();
    ed.clearAll();
    ed.select([]);
    const a = await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_Simple", new V(0, 0, 0), { silent: true });
    ed.select([a.id]);
    ed.showAxes(a.id, "world", "centre");
    ed.state.scene.render();
    const root = ed.state.scene.getTransformNodeByName("AXES");

    // the arm's length *in the world* this time, not as built: the tip cube sits
    // at a fixed multiple of it, so its distance from the root reads it back out
    const armWorld = () => +(ed.state.scene.getMeshByName("AXES_x_scale").getAbsolutePosition().subtract(root.position).length() / 1.04).toFixed(3);

    const from = (metres) => {
        const at = root.position.clone();
        ed.state.camera.position = at.add(new V(0, metres * 0.6, -metres * 0.8));
        ed.state.camera.setTarget(at);
        ed.state.scene.render();
        ed.state.scene.render();
        return { dist: +V.Distance(ed.state.camera.position, root.position).toFixed(2), arm: armWorld() };
    };
    const far = from(12);
    const close = from(3);
    const closer = from(1.5);
    const back = from(12); // and it comes back when you pull away again

    ed.hideAxes();
    ed.clearAll();
    ed.select([]);
    return { far, close, closer, back };
});
check("the camera really did move either side of the 5 m mark", gizmoNear.far.dist > 5 && gizmoNear.close.dist < 5, `${gizmoNear.far.dist} m then ${gizmoNear.close.dist} m`);
check(
    "the arms halve inside 5 m and come back when you pull away",
    Math.abs(gizmoNear.far.arm - 2) < 1e-3 && Math.abs(gizmoNear.close.arm - 1) < 1e-3 && Math.abs(gizmoNear.back.arm - 2) < 1e-3,
    `${gizmoNear.far.dist} m -> ${gizmoNear.far.arm} m arms, ` + `${gizmoNear.close.dist} m -> ${gizmoNear.close.arm} m, back at ${gizmoNear.back.arm} m`
);
check("the camera really did get inside the 2 m mark too", gizmoNear.closer.dist < 2, `${gizmoNear.closer.dist} m`);
check(
    "and they halve again inside 2 m, to a quarter length",
    Math.abs(gizmoNear.closer.arm - 0.5) < 1e-3,
    `${gizmoNear.closer.dist} m -> ${gizmoNear.closer.arm} m arms`
);

// ---- 1d-octodecies. no target hides, and the ghost counts as a target -------
const axesNoTarget = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
  await i.armGhost("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight");
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
  const p = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
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

// ---- the gizmo is never culled, and knows where it is -----------------------
// It does not move by being re-parented: its root is re-positioned every frame
// onto whatever element it belongs to. `doNotSyncBoundingInfo` was set on every
// part as a micro-optimisation, and that is precisely the flag that stops a
// bounding box following the world matrix - so the boxes stayed where the gizmo
// was *built*. On an element 50 m out the boxes trailed 50 m behind, and the
// frustum test culled arms, heads and turn arcs on their old position. That is
// what "the arrows get clipped, and going forward clips the lines too" was, and
// why clicking away and back cured it: re-showing rebuilds the meshes.
const gizmoCull = await page.evaluate(async (moduleId) => {
  const ed = await import("/js/editor.js");
  const V = BABYLON.Vector3;
  const scene = ed.state.scene;
  const cam = ed.state.camera;
  // Blocks share the scene, and the chip checks below expect the gizmo this
  // one found. So: borrow it, and put everything back.
  const had = { target: ed.axesTarget(), space: ed.axesSpace(),
    selection: [...ed.state.selection],
    pos: cam.position.clone(), rot: cam.rotation.clone() };
  const e = await ed.placeAt(moduleId, new V(0, 0, 0), { silent: true });
  ed.select([e.id]);
  ed.showAxes(e.id, "world");
  const look = (at, back) => {
    cam.cameraDirection.setAll(0);
    cam.cameraRotation.set(0, 0);
    cam.position.set(at[0], at[1] + 1.5, at[2] - back);
    cam.rotation.set(0.08, 0, 0);
  };
  const survey = () => {
    scene.render();
    const parts = scene.meshes.filter((m) => m.name.startsWith("AXES_") && m.isEnabled());
    const active = new Set(scene.getActiveMeshes().data.filter(Boolean).map((m) => m.uniqueId));
    const drift = parts.map((m) => +m.getBoundingInfo().boundingBox.centerWorld
      .subtract(m.getAbsolutePosition()).length().toFixed(2));
    return { enabled: parts.length, drawn: parts.filter((m) => active.has(m.uniqueId)).length,
      drift: Math.max(...drift) };
  };
  look([0, 0, 0], 8);
  const atOrigin = survey();
  // out where a real ship is, with the camera following it
  e.node.position.set(40, 0, -30);
  look([40, 0, -30], 8);
  const away = survey();
  look([40, 0, -30], 2.2);        // "go forward"
  const near = survey();
  look([40, 0, -30], 0.8);
  const closer = survey();

  ed.hideAxes();
  ed.removePlacement(e.id);
  ed.select(had.selection);
  if (had.target) ed.showAxes(had.target, had.space);
  cam.cameraDirection.setAll(0);
  cam.cameraRotation.set(0, 0);
  cam.position.copyFrom(had.pos);
  cam.rotation.copyFrom(had.rot);
  scene.render();
  return { atOrigin, away, near, closer, restored: ed.axesTarget() === had.target };
}, PROP_A);
check("a gizmo on an element far from the origin keeps its bounding boxes with it",
  gizmoCull.away.drift < 0.01, `${gizmoCull.away.drift} m behind the arrows`);
for (const where of ["atOrigin", "away", "near", "closer"]) {
  const r = gizmoCull[where];
  check(`every part of the gizmo is drawn (${where})`,
    r.enabled > 0 && r.drawn === r.enabled, `${r.drawn} of ${r.enabled}`);
}
check("and the block put back the gizmo it borrowed", gizmoCull.restored);

// and it follows both the angle and the axis
await page.keyboard.press("Shift+r");
await page.waitForTimeout(250);
await page.keyboard.press("r");
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
check("the angle chip follows Shift+R, and moves to the arm R picks",
  rotFollowed.text === "free" && rotFollowed.axis === "x"
    && rotFollowed.onAxisArm.join() === "true,false,false",
  `${rotFollowed.text} on ${rotFollowed.axis}, at ${rotFollowed.onAxisArm}`);
// A chip says what its combo says. `free` is a real zero on Move but the finest
// step there is on Rot and Scale - 0.5 degrees and 0.01 - so a chip that printed
// the number while the toolbar said `free` made one setting look like two.
const freeChips = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const pick = (id, key, v) => {
    ed.state.snap[key] = v;
    document.getElementById(id).value = String(v);
  };
  pick("snap-rot", "rot", 0.5);
  pick("snap-scale", "scale", 0.01);
  // all three cubes lit and the camera far enough back that their chips are on
  // canvas: a chip clamped off-screen is hidden, and would read as no chip
  ed.state.scaleAxis = "all";
  document.getElementById("scale-axis").value = "all";
  ed.state.camera.position = new BABYLON.Vector3(14, 12, -18);
  ed.state.camera.setTarget(BABYLON.Vector3.Zero());
  ed.state.camera.cameraDirection.setAll(0);
  return { free: { ...ed.freeSnap } };
});
await page.waitForTimeout(500);
const freeText = await page.evaluate(() => ({
  rot: document.querySelector("#viewport .axis-rot")?.textContent,
  scale: [...new Set([...document.querySelectorAll("#viewport .axis-scale")]
    .filter((e) => !e.hidden).map((e) => e.textContent))],
}));
check("the loosest step of each list is read off the combo, not written out again",
  freeChips.free.pos === 0 && freeChips.free.rot === 0.5 && freeChips.free.scale === 0.01,
  JSON.stringify(freeChips.free));
check("so the rot and scale chips say 'free' where the combo does",
  freeText.rot === "free" && freeText.scale.join() === "free",
  JSON.stringify(freeText));
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.snap.rot = 5; ed.state.snap.scale = 0.05;
  document.getElementById("snap-rot").value = "5";
  document.getElementById("snap-scale").value = "0.05";
});
await page.waitForTimeout(500);
const coarse = await page.evaluate(() => ({
  rot: document.querySelector("#viewport .axis-rot")?.textContent,
  scale: [...new Set([...document.querySelectorAll("#viewport .axis-scale")]
    .filter((e) => !e.hidden).map((e) => e.textContent))],
}));
check("and go back to the number for any other step",
  coarse.rot === "5°" && coarse.scale.join() === "0.05",
  JSON.stringify(coarse));
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

await page.keyboard.press("f");                // all -> x
await page.waitForTimeout(400);
await page.keyboard.press("Shift+f");          // 0.1 -> 0.25
await page.waitForTimeout(400);
const sclOne = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const els = [...document.querySelectorAll("#viewport .axis-scale")];
  const lit = els.filter((e) => !e.hidden);
  return { axis: ed.state.scaleAxis, step: ed.state.snap.scale,
           shown: lit.length, text: lit[0]?.textContent };
});
check("one axis lights one chip, and Shift+F changes what it says",
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
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
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
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
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

  // ...and they follow in the flavour they were already in. Re-showing them
  // took the default, so a local gizmo reverted to world on the next click and
  // Shift+X had to be pressed again for every element - which is the one that
  // matters, scaling being local and a turned piece having its own idea of
  // which way X grows.
  ed.showAxes(a.id, "local");
  const localStart = ed.axesSpace();
  ed.select([b.id]);
  const localKept = { on: ed.axesTarget(), space: ed.axesSpace() };
  ed.select([c.id]);
  const localTwice = ed.axesSpace();
  ed.showAxes(a.id, "world");
  ed.select([b.id]);
    const worldKept = ed.axesSpace();
    // the anchor rides along with the flavour, for the same reason: having asked
    // for the middle of the mesh you did not ask to be sent back to the origin
    ed.showAxes(a.id, "world", "centre");
    ed.select([c.id]);
    const anchorKept = { on: ed.axesTarget(), anchor: ed.axesAnchor() };
    // and X/Shift+X still say which flavour outright, whatever is on screen
    ed.showAxes(a.id, "local");
  ed.toggleAxes(a.id, "world");
  const xForcesWorld = ed.axesSpace();
  ed.toggleAxes(a.id, "local");
  const shiftXForcesLocal = ed.axesSpace();

  ed.hideAxes();
  ed.select([a.id]);
  const whenHidden = ed.axesTarget();      // hidden stays hidden
  ed.clearAll();
    ed.select([]);
    return { started, followed, onMulti, onNone, whenHidden, a: a.id, b: b.id, c: c.id, localStart, localKept, localTwice, worldKept, anchorKept, xForcesWorld, shiftXForcesLocal };
});
check("visible axes follow a single click to the new element", follows.started === follows.a && follows.followed === follows.b, `${follows.started} -> ${follows.followed}`);
check("a multi-selection or an empty one leaves them where they are",
  follows.onMulti === follows.b && follows.onNone === follows.b,
  `multi=${follows.onMulti}, none=${follows.onNone}`);
check("selecting does not conjure axes that were never shown",
  follows.whenHidden === null, `${follows.whenHidden}`);
check("a local gizmo is still local on the element you click next",
  follows.localStart === "local" && follows.localKept.on === follows.b
    && follows.localKept.space === "local" && follows.localTwice === "local",
  `${follows.localStart} -> ${follows.localKept.space} -> ${follows.localTwice}`);
check("and a world one is still world", follows.worldKept === "world", follows.worldKept);
check(
    "and a centre-hung one is still hung on the centre",
    follows.anchorKept.on === follows.c && follows.anchorKept.anchor === "centre",
    `${follows.anchorKept.on} / ${follows.anchorKept.anchor}`
);
check(
    "X and Shift+X still name the flavour outright",
  follows.xForcesWorld === "world" && follows.shiftXForcesLocal === "local",
  `X -> ${follows.xForcesWorld}, Shift+X -> ${follows.shiftXForcesLocal}`);

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
  await page.keyboard.press("r");
  await page.waitForTimeout(120);
  rotCycle.push(await page.evaluate(async () => (await import("/js/editor.js")).state.rotAxis));
}
const sclCycle = [];
for (let k = 0; k < 4; k++) {
  await page.keyboard.press("f");
  await page.waitForTimeout(120);
  sclCycle.push(await page.evaluate(async () => (await import("/js/editor.js")).state.scaleAxis));
}
check("R cycles the rotation axis",
  rotCycle.join() === "x,z,y", rotCycle.join(" -> "));
check("F cycles the scale axis",
  sclCycle.join() === "x,y,z,all", sclCycle.join(" -> "));

const rotStep = [];
for (let k = 0; k < 2; k++) {
  await page.keyboard.press("Shift+r");
  await page.waitForTimeout(120);
  rotStep.push(await page.evaluate(async () => ({
    v: (await import("/js/editor.js")).state.snap.rot,
    combo: document.getElementById("snap-rot").value,
  })));
}
await page.keyboard.press("Control+r");
await page.waitForTimeout(120);
const rotStepBack = await page.evaluate(async () => (await import("/js/editor.js")).state.snap.rot);
// The list runs 0.5 to 90, so past 90 the cycle wraps round to the far end.
check("Shift+R cycles the rotation angle, Ctrl+R goes back",
  rotStep.map((r) => r.v).join() === "0.5,5" && rotStep.every((r) => r.combo === String(r.v))
    && rotStepBack === 0.5,
  `90 -> ${rotStep.map((r) => r.v).join(" -> ")} -> back ${rotStepBack}`);

const sclStep = [];
for (let k = 0; k < 2; k++) {
  await page.keyboard.press("Shift+f");
  await page.waitForTimeout(120);
  sclStep.push(await page.evaluate(async () => ({
    v: (await import("/js/editor.js")).state.snap.scale,
    combo: document.getElementById("snap-scale").value,
  })));
}
await page.keyboard.press("Control+f");
await page.waitForTimeout(120);
const sclStepBack = await page.evaluate(async () => (await import("/js/editor.js")).state.snap.scale);
check("Shift+F cycles the scale step, Ctrl+F goes back",
  sclStep.map((r) => r.v).join() === "0.25,0.01"
    && sclStep.every((r) => r.combo === String(r.v)) && sclStepBack === 0.25,
  `0.1 -> ${sclStep.map((r) => r.v).join(" -> ")} -> back ${sclStepBack}`);

// Put the four settings back where the rest of the run expects them. Leaving a
// half-degree step behind is invisible here and turns every later rotation
// check into a no-op that still reads as a pass on the wrong grounds.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.state.snap.rot = 90; ed.state.rotAxis = "y";
  ed.state.snap.scale = 0.1; ed.state.scaleAxis = "all";
  document.getElementById("snap-rot").value = "90";
  document.getElementById("snap-scale").value = "0.1";
  document.getElementById("rot-axis").value = "y";
  document.getElementById("scale-axis").value = "all";
});

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
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
  // The Runtime view is what selects the RUNTIME pair - the one the demos read
  // - so the values have to be set with it active. Dressing the whole ship is
  // not this suite's business, so the flag is moved directly and
  // syncLightingMode() is asked to follow it: that is exactly what
  // setRuntimePreview does, minus the materials.
  const runtimeMode = (on) => { ed.state.runtime = on; ed.syncLightingMode(); };
  runtimeMode(true);
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
      env: document.getElementById("runtime-env").value,
      exposure: document.getElementById("runtime-exposure").value,
      envLabel: document.getElementById("runtime-env-val").textContent,
      exposureLabel: document.getElementById("runtime-exposure-val").textContent,
      // The runtime rig lives in the manifest and nowhere else. Only the
      // editor's is mirrored here, so that a fresh tab opens looking the way
      // this one was left before a manifest has been read.
      envStored: localStorage.getItem("editorEnv"),
      exposureStored: localStorage.getItem("editorExposure"),
      editorEnv: document.getElementById("editor-env").value,
    };
  } finally { window.fetch = realFetch; }

  const scene = {
    strength: ed.state.scene.environmentIntensity,
    authored: ed.state.envIntensity,
    exposure: ed.state.scene.imageProcessingConfiguration.exposure,
  };
  const inUndoState = "lightSets" in JSON.parse(JSON.stringify(ed.serialize()));
  const ignoredOld = ed.applyEnvironment(undefined);

  ed.setEnvIntensity(ed.ENV_INTENSITY_DEFAULT);
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  runtimeMode(false);
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
    // the two material dials ride here too - the game reads them as the
    // authored defaults, so they are ship values, not view preferences
    && envRound.saved.specularAA === true
    && envRound.saved.reflectionRoughness === 1
    && Object.keys(envRound.saved).sort().join()
      === "exposure,reflectionRoughness,specularAA,strength,toneMapping",
  JSON.stringify(envRound.saved));
check("loading puts the lighting back on the scene",
  Math.abs(envRound.scene.authored - 2.4) < 1e-6
    // Babylon's scene-wide multiplier stays at 1 in the Runtime view on
    // purpose: Env rides each material there, because each one carries its own
    // room's probe, so a scene multiplier would be a second, invisible factor
    // on top of it. See setEnvIntensity.
    && Math.abs(envRound.scene.strength - 1) < 1e-6
    && Math.abs(envRound.scene.exposure - 0.82) < 1e-3,
  `authored=${envRound.scene.authored}, scene=${envRound.scene.strength}, exposure=${envRound.scene.exposure}`);
check("the sliders follow the loaded lighting",
  parseFloat(envRound.sliders.env) === 2.4
    // 0.82 is deliberately off the slider's 0.05 step: the readout and the
    // stored value stay exact, only the thumb rounds
    && envRound.sliders.exposureLabel === "0.82"
    && Math.abs(parseFloat(envRound.sliders.exposure) - 0.82) <= 0.05,
  JSON.stringify(envRound.sliders));
check("the runtime rig is not written to this browser's storage",
  // It belongs to the ship. Copying it here is what used to let a session
  // spent in a runtime view overwrite the brightness the editor came up in:
  // storage tracks the *editor* slider, whatever the runtime one says.
  parseFloat(envRound.sliders.envStored) === parseFloat(envRound.sliders.editorEnv)
    && Math.abs(parseFloat(envRound.sliders.envStored) - 2.4) > 1e-6
    && Math.abs(parseFloat(envRound.sliders.exposureStored) - 0.82) > 1e-3,
  JSON.stringify(envRound.sliders));
check("the environment now rides the undo snapshot",
  envRound.inUndoState === true);
check("a manifest saved before this simply has no environment to apply",
  envRound.ignoredOld === false);

// ---- 1d-vicies. the lighting is undoable, one entry per gesture -------------
// It used to be deliberately off the stack ("not an edit to the ship"). The
// *runtime* rig is an authored value the manifest carries and the runtime
// reads, so getting it wrong is as much an edit to take back as moving a wall.
// The editor's rig is the opposite and must stay off the stack - see below.
const lightUndo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.setLightSetting("runtime", "strength", 1.5);
  ed.setLightSetting("runtime", "exposure", 0.55);
  document.getElementById("runtime-env").value = "1.5";
  return { start: ed.state.lightSets.runtime.strength, depth: ed.historyDepth().undo };
});
// one gesture: press, then several input events as the thumb travels
await page.evaluate(() => {
  const el = document.getElementById("runtime-env");
  el.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
  for (const v of ["2.0", "2.5", "3.0"]) {
    el.value = v;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
});
await page.waitForTimeout(250);
const afterDrag = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { env: ed.state.lightSets.runtime.strength, depth: ed.historyDepth().undo };
});
check("dragging the runtime Env slider costs exactly one undo entry",
  Math.abs(afterDrag.env - 3.0) < 1e-6 && afterDrag.depth === lightUndo.depth + 1,
  `${lightUndo.start} -> ${afterDrag.env}, stack ${lightUndo.depth} -> ${afterDrag.depth}`);

// The editor's own rig is how you like to look at the ship, not part of it.
// Spending the undo stack on it would push out the edits worth taking back.
await page.evaluate(() => {
  const el = document.getElementById("editor-env");
  el.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
  for (const v of ["2.0", "2.6"]) {
    el.value = v;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
});
await page.waitForTimeout(250);
const editorDrag = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return {
    env: ed.state.lightSets.editor.strength,
    live: ed.state.envIntensity,
    depth: ed.historyDepth().undo,
    stored: localStorage.getItem("editorEnv"),
  };
});
check("dragging the editor Env slider changes the view and nothing else",
  Math.abs(editorDrag.env - 2.6) < 1e-6 && Math.abs(editorDrag.live - 2.6) < 1e-6
    && editorDrag.depth === afterDrag.depth && parseFloat(editorDrag.stored) === 2.6,
  JSON.stringify(editorDrag));

await page.evaluate(async () => (await import("/js/editor.js")).undo());
await page.waitForTimeout(250);
const undone = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return {
    env: ed.state.lightSets.runtime.strength,
    slider: parseFloat(document.getElementById("runtime-env").value),
    readout: document.getElementById("runtime-env-val").textContent,
    // the editor rig is live, so it is what reached the scene
    scene: ed.state.scene.environmentIntensity,
    editor: ed.state.lightSets.editor.strength,
  };
});
check("undo puts the runtime Env back, on the slider and its readout",
  Math.abs(undone.env - 1.5) < 1e-6
    && Math.abs(undone.slider - 1.5) < 1e-6 && undone.readout === "1.5"
    // and leaves the editor's rig, which the undo entry never covered, alone
    && Math.abs(undone.editor - 2.6) < 1e-6 && Math.abs(undone.scene - 2.6) < 1e-6,
  JSON.stringify(undone));

await page.evaluate(async () => (await import("/js/editor.js")).redo());
await page.waitForTimeout(250);
const redone = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.lightSets.runtime.strength);
check("redo brings it back", Math.abs(redone - 3.0) < 1e-6, `${redone}`);

// the *inactive* light set must travel too, or switching to a runtime view
// after an undo would surface a value that was never restored
const bothSets = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.setLightSetting("runtime", "strength", 1.5);
  ed.setLightSetting("editor", "strength", 1.2);
  ed.pushUndo();
  ed.setLightSetting("runtime", "strength", 3.9);   // the set nothing is rendering
  const before = { ...ed.state.lightSets.runtime };
  await ed.undo();
  const after = { ...ed.state.lightSets.runtime };
  ed.setLightSetting("editor", "strength", 1.5);
  ed.setLightSetting("editor", "exposure", 0.55);
  ed.setLightSetting("runtime", "strength", 1.5);
  ed.setLightSetting("runtime", "exposure", 0.55);
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(8, 0, 0), { silent: true });
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

// ---- 1d-vicies. the Runtime view silences the authoring rig ----------------
// The editor's four analytic lights are an authoring aid the game does not
// have, and they are the reason it looks nothing like the runtime - not the
// exposure conversion, which round-trips exactly. The Runtime view is the one
// state where dropping them says something true, so it owns the switch.
const runtimeRig = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const runtimeMode = (on) => { ed.state.runtime = on; ed.syncLightingMode(); };
  const rig = () => ed.state.scene.lights
    .filter((l) => ["hemi", "hemiUp", "key", "fill"].includes(l.name))
    .map((l) => l.intensity);

  const before = rig();
  runtimeMode(true);
  const off = rig();
  runtimeMode(false);
  const back = rig();

  // the numbers the demos read must be the ones the slider shows
  runtimeMode(true);
  ed.setExposure(0.55);
  ed.setEnvIntensity(1.7);
  const env = mf.buildManifest().environment;
  runtimeMode(false);
  return { before, off, back, env, mode: ed.state.runtime };
});
check("the Runtime view silences the authoring rig, and only it",
  runtimeRig.before.length === 4 && runtimeRig.off.every((v) => v === 0)
    && runtimeRig.before.some((v) => v > 0),
  `${JSON.stringify(runtimeRig.before)} -> ${JSON.stringify(runtimeRig.off)}`);
check("switching back restores the authored intensities exactly",
  runtimeRig.back.join() === runtimeRig.before.join() && !runtimeRig.mode,
  JSON.stringify(runtimeRig.back));
check("the exposure is written as the slider shows it, no conversion",
  runtimeRig.env.exposure === 0.55 && runtimeRig.env.strength === 1.7,
  `exposure ${runtimeRig.env.exposure}, strength ${runtimeRig.env.strength}`);

// ---- 1d-unvicies. two light sets, one per mode ----------------------------
// The rig adds four lights the game does not have, so one pair of Env/Exposure
// values cannot serve both. The runtime pair goes to `environment`, which the
// demos read; the editor pair goes to `editorEnvironment`, which they must not.
const lightSets = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  const runtimeMode = (on) => { ed.state.runtime = on; ed.syncLightingMode(); };
  runtimeMode(false);
  ed.setEnvIntensity(1.5); ed.setExposure(0.55);        // editor pair
  runtimeMode(true);
  ed.setEnvIntensity(3.2); ed.setExposure(1.1);         // runtime pair
  const inRuntime = { env: ed.state.envIntensity, exp: ed.state.exposure };
  runtimeMode(false);
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

  ed.state.runtime = false; ed.syncLightingMode();
  ed.setEnvIntensity(ed.ENV_INTENSITY_DEFAULT);
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  return { inRuntime, backInEditor, written, restored, legacy };
});
check("each mode keeps its own Env/Exposure",
  lightSets.inRuntime.env === 3.2 && lightSets.inRuntime.exp === 1.1
    && lightSets.backInEditor.env === 1.5
    && lightSets.backInEditor.exp === 0.55,
  `runtime ${JSON.stringify(lightSets.inRuntime)}, editor ${JSON.stringify(lightSets.backInEditor)}`);
check("the demos get the runtime pair, the editor pair is filed separately",
  lightSets.written.environment.strength === 3.2
    && lightSets.written.environment.exposure === 1.1
    && lightSets.written.editorEnvironment.strength === 1.5
    && lightSets.written.editorEnvironment.exposure === 0.55,
  JSON.stringify(lightSets.written));
check("a round trip keeps the two pairs apart",
  lightSets.restored.runtime.strength === 3.2
    && lightSets.restored.editor.strength === 1.5
    && lightSets.restored.active.env === 1.5,
  JSON.stringify(lightSets.restored));
check("a manifest with one pair gives it to both, rather than to neither",
  lightSets.legacy.editor.strength === 2 && lightSets.legacy.runtime.strength === 2
    && lightSets.legacy.editor.exposure === 0.9,
  JSON.stringify(lightSets.legacy));

// ---- 1d-duovicies. every row on the Settings pane survives a save ----------
// The pane is the ship's global settings, so nothing on it may be a value you
// have to set again next session. The two material dials ride `environment`,
// where the game reads them; the editor's view preferences ride `editorPrefs`,
// where it must not. A manifest predating either block leaves both alone.
const settingsRoundTrip = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  ed.setRuntimeSpecularAA(false);
  ed.setRuntimeRoughnessFactor(1.35);
  ed.setVeilAlpha(0.25);
  ed.state.bigPalette = false;
  ed.state.strayChunkCheck = false;

  const man = mf.buildManifest();
  const written = { environment: man.environment, editorPrefs: man.editorPrefs };

  // move every one of them, then load what was written back
  ed.setRuntimeSpecularAA(true);
  ed.setRuntimeRoughnessFactor(2);
  ed.setVeilAlpha(0.9);
  ed.state.bigPalette = true;
  ed.state.strayChunkCheck = true;
  ed.applyEnvironment(written.environment, man.editorEnvironment);
  ed.applyEditorPrefs(written.editorPrefs);
  const restored = {
    specularAA: ed.state.runtimeSpecularAA,
    roughness: ed.state.runtimeRoughnessFactor,
    veilAlpha: ed.state.veilAlpha,
    bigPalette: ed.state.bigPalette,
    strayChunkCheck: ed.state.strayChunkCheck,
  };

  // a manifest from before the blocks existed must not reset what is on screen
  ed.applyEnvironment({ strength: 1.5, exposure: 0.55 }, undefined);
  ed.applyEditorPrefs(undefined);
  const legacy = {
    specularAA: ed.state.runtimeSpecularAA,
    roughness: ed.state.runtimeRoughnessFactor,
    veilAlpha: ed.state.veilAlpha,
    bigPalette: ed.state.bigPalette,
    strayChunkCheck: ed.state.strayChunkCheck,
  };

  ed.setRuntimeSpecularAA(ed.RUNTIME_SPECULAR_AA_DEFAULT);
  ed.setRuntimeRoughnessFactor(ed.RUNTIME_ROUGHNESS_FACTOR_DEFAULT);
  // through applyEditorPrefs, so the palette and the slider follow state back
  ed.applyEditorPrefs({
    veilAlpha: ed.VEIL_ALPHA_DEFAULT,
    bigPalette: ed.BIG_PALETTE_DEFAULT,
    strayChunkCheck: ed.STRAY_CHUNK_CHECK_DEFAULT,
  });
  ed.state.runtime = false; ed.syncLightingMode();
  ed.setEnvIntensity(ed.ENV_INTENSITY_DEFAULT);
  ed.setExposure(ed.EXPOSURE_DEFAULT);
  return { written, restored, legacy };
});
check("the material dials are written where the game reads them",
  settingsRoundTrip.written.environment.specularAA === false
    && settingsRoundTrip.written.environment.reflectionRoughness === 1.35,
  JSON.stringify(settingsRoundTrip.written.environment));
check("the editor's view preferences are written apart from the ship's",
  settingsRoundTrip.written.editorPrefs.veilAlpha === 0.25
    && settingsRoundTrip.written.editorPrefs.bigPalette === false
    && settingsRoundTrip.written.editorPrefs.strayChunkCheck === false,
  JSON.stringify(settingsRoundTrip.written.editorPrefs));
check("every one of them comes back on load",
  settingsRoundTrip.restored.specularAA === false
    && settingsRoundTrip.restored.roughness === 1.35
    && settingsRoundTrip.restored.veilAlpha === 0.25
    && settingsRoundTrip.restored.bigPalette === false
    && settingsRoundTrip.restored.strayChunkCheck === false,
  JSON.stringify(settingsRoundTrip.restored));
check("a manifest without the blocks leaves them where they are",
  settingsRoundTrip.legacy.specularAA === false
    && settingsRoundTrip.legacy.roughness === 1.35
    && settingsRoundTrip.legacy.veilAlpha === 0.25
    && settingsRoundTrip.legacy.bigPalette === false
    && settingsRoundTrip.legacy.strayChunkCheck === false,
  JSON.stringify(settingsRoundTrip.legacy));

// ---- 1d-duovicies-bis. elements left in the wrong chunk --------------------
// A chunk has no authored volume - its box is the union of what is assigned to
// it - so "is this inside its chunk?" is true by construction. The question the
// check actually asks is the same one with the element taken out of that union.
const strays = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const W = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  // The panel runs 4 m along z and is a few millimetres thick along x, so a
  // run of them 4 m apart in z is stuck together the way a real wall is.
  const wall = (x, z, chunk) => ed.placeAt(W, new V(x, 0, z), { silent: true, chunk });
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.addChunk("CH_SA"); ed.addChunk("CH_SB");

  // Two rooms, a long way apart.
  for (const z of [0, 4, 8]) await wall(0, z, "CH_SA");
  for (const z of [40, 44, 48]) await wall(0, z, "CH_SB");
  const quiet = ed.strayChunkMembers();

  // A piece left in CH_SA while building CH_SB's wall: the classic slip of
  // carrying on without moving the active chunk on.
  const misplaced = await wall(0, 52, "CH_SA");
  const caught = ed.strayChunkMembers();

  // Something out on its own belongs to no chunk at all and has to say so
  // rather than naming one at random. Two strays in the same chunk also have
  // to be reported separately rather than vouching for each other.
  const lost = await wall(300, 300, "CH_SA");
  const both = ed.strayChunkMembers();
  const stillMine = both.every((s) => s.chunk === "CH_SA");
  ed.removePlacement(lost.id);

  // Mounting offsets are not mistakes: trim and decals sit a hand's breadth
  // proud of the surface they belong to.
  const proud = await wall(0.4, 44, "CH_SB");
  const withProud = ed.strayChunkMembers().length;
  ed.removePlacement(proud.id);

  // A chunk that has only just been started has nothing to be outside of, and
  // one split evenly has no room to call the odd one out from.
  ed.addChunk("CH_SOLO");
  const first = await wall(0, -200, "CH_SOLO");
  const solo = ed.strayChunkMembers().length;
  const second = await wall(0, -100, "CH_SOLO");
  const even = ed.strayChunkMembers().length;
  ed.removePlacement(first.id); ed.removePlacement(second.id);

  // The setting silences it everywhere, including the Live checks panel.
  document.getElementById("stray-chunk-check").click();
  await new Promise((r) => setTimeout(r, 50));
  const panelOff = document.getElementById("validation").textContent;
  const stateOff = ed.state.strayChunkCheck;
  document.getElementById("stray-chunk-check").click();
  await new Promise((r) => setTimeout(r, 50));
  const panelOn = document.getElementById("validation").textContent;

  ed.clearAll(); ed.select([]);
  return {
    quiet, caught, both, stillMine, misplaced: misplaced.id,
    proud: withProud, solo, even,
    panelOff, panelOn, stateOff, stateOn: ed.state.strayChunkCheck,
  };
});
check("a chunk whose members touch reports no strays",
  strays.quiet.length === 0, JSON.stringify(strays.quiet));
check("an element left in the wrong chunk is caught and the right chunk named",
  strays.caught.length === 1
    && strays.caught[0].id === strays.misplaced
    && strays.caught[0].chunk === "CH_SA"
    && strays.caught[0].host === "CH_SB",
  JSON.stringify(strays.caught));
check("two strays in one chunk are both reported, and one clear of every chunk names none",
  strays.both.length === 2 && strays.stillMine
    && strays.both.filter((s) => s.host === "CH_SB").length === 1
    && strays.both.filter((s) => s.host === null).length === 1,
  JSON.stringify(strays.both));
check("a mounting offset is not a stray",
  strays.proud === 1, `${strays.proud} stray(s), expected only the misplaced one`);
check("a chunk just started, or split evenly, is left alone",
  strays.solo === 1 && strays.even === 1,
  `solo=${strays.solo} even=${strays.even}, expected only the misplaced one in both`);
check("the stray-chunk setting silences the check and the Live checks panel",
  strays.stateOff === false && strays.stateOn === true
    && !/wrong chunk|touch nothing|sit in/i.test(strays.panelOff)
    && /sit in CH_SB/.test(strays.panelOn),
  `off="${strays.panelOff}" on="${strays.panelOn}"`);

// A warning that names a piece is only worth reading if you can get to that
// piece: every name a check prints is a way there.
const strayLink = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const W = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  const wall = (x, z, chunk) => ed.placeAt(W, new V(x, 0, z), { silent: true, chunk });
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.addChunk("CH_LA"); ed.addChunk("CH_LB");
  for (const z of [0, 4, 8]) await wall(0, z, "CH_LA");
  for (const z of [40, 44, 48]) await wall(0, z, "CH_LB");
  // The last one loudly, so the panel is looking at this ship and not at
  // whatever the section before left behind.
  const misplaced = await ed.placeAt(W, new V(0, 0, 52), { chunk: "CH_LA" });
  await new Promise((r) => setTimeout(r, 50));
  ed.select([]);
  const link = [...document.querySelectorAll("#validation .check-ref")]
    .find((node) => node.textContent === misplaced.id);
  const camera = ed.state.camera.position.clone();
  link?.click();
  await new Promise((r) => setTimeout(r, 50));
  return {
    named: !!link,
    // the prose stays prose: only the element's own name is a control
    text: document.getElementById("validation").textContent,
    selected: [...ed.state.selection],
    misplaced: misplaced.id,
    framed: BABYLON.Vector3.Distance(camera, ed.state.camera.position) > 1,
  };
});
check("a stray named in the checks panel selects and frames that element",
  strayLink.named && strayLink.selected.join() === strayLink.misplaced && strayLink.framed
    && /sit in CH_LB/.test(strayLink.text),
  JSON.stringify(strayLink));

// A door names itself in its own warnings, and that name goes to the door.
const doorLink = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  const door = mk.addDoor(new V(0, 0, 0));
  await new Promise((r) => setTimeout(r, 50));
  const link = [...document.querySelectorAll("#validation .check-ref")]
    .find((node) => node.textContent === door.id);
  link?.click();
  await new Promise((r) => setTimeout(r, 50));
  const selected = [...ed.state.selection];
  ed.clearAll(); ed.select([]);
  return { named: !!link, selected, door: door.id };
});
check("a door named in the checks panel is a way to that door",
  doorLink.named && doorLink.selected.join() === doorLink.door, JSON.stringify(doorLink));

// a negative exposure can only be the old stops format - the slider has never
// gone below 0.15 - so it is converted rather than clamped up to the floor
const oldStops = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.applyEnvironment({ strength: 1.7, exposure: -0.862 }, undefined);
  const migrated = { ...ed.state.lightSets.runtime };
  ed.applyEnvironment({ strength: 1.7, exposure: 0.55 }, undefined);
  const literal = { ...ed.state.lightSets.runtime };
  ed.state.runtime = false; ed.syncLightingMode();
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

// each rig keeps its own transform, or picking one in a runtime view would
// silently restyle the editor too
const tonePerSet = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  ed.setLightSetting("editor", "toneMapping", "ACES");
  ed.setLightSetting("runtime", "toneMapping", "Standard");
  const live = ed.state.toneMapping;                    // editor is the live one
  const man = mf.buildManifest();
  ed.setLightSetting("editor", "toneMapping", "None");
  ed.setLightSetting("runtime", "toneMapping", "None");
  ed.applyEnvironment(man.environment, man.editorEnvironment);
  const restored = {
    editor: ed.state.lightSets.editor.toneMapping,
    runtime: ed.state.lightSets.runtime.toneMapping,
  };
  // read while the restore is still showing - resetting first would only
  // prove the combos follow the reset
  const combos = {
    editor: document.getElementById("editor-tone").value,
    runtime: document.getElementById("runtime-tone").value,
  };
  ed.setLightSetting("editor", "toneMapping", "Khronos PBR Neutral");
  ed.setLightSetting("runtime", "toneMapping", "Khronos PBR Neutral");
  return {
    live, restored, combos,
    written: man.environment.toneMapping,
    writtenEditor: man.editorEnvironment.toneMapping,
  };
});
check("the two rigs keep their own view transform, through the manifest",
  tonePerSet.live === "ACES" && tonePerSet.written === "Standard"
    && tonePerSet.writtenEditor === "ACES"
    && tonePerSet.restored.editor === "ACES" && tonePerSet.restored.runtime === "Standard"
    && tonePerSet.combos.editor === "ACES" && tonePerSet.combos.runtime === "Standard",
  JSON.stringify(tonePerSet));

// ---- 1d-duovicies. one combo names the view, and the flags agree with it ----
// The modes used to be checkboxes that could spell states nothing rendered.
// The flags stayed; the combo is read back off them, so it cannot claim a mode
// the viewport is not in.
const viewModes = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const sel = document.getElementById("view-mode");
  const derived = {};
  for (const [mode, flags] of Object.entries(ed.VIEW_MODES)) {
    ed.state.runtime = flags.runtime;
    ed.state.unlit = flags.unlit;
    derived[mode] = ed.viewMode();
  }
  ed.state.runtime = false; ed.state.unlit = false;
  ed.syncLightingMode();
  return {
    derived,
    options: [...sel.options].map((o) => o.value),
    value: sel.value,
    // the controls it replaced
    gone: ["unlit", "baked", "baked-diffuse-only", "env-intensity", "exposure",
           "tone-mapping", "dynamic-env-intensity", "btn-export"]
      .filter((id) => document.getElementById(id)),
    // both rigs on the Settings pane, neither on the toolbar
    inSettings: ["editor-env", "editor-exposure", "editor-tone", "runtime-env",
                 "runtime-exposure", "runtime-tone"]
      .filter((id) => !document.getElementById("settings-pane")?.contains(document.getElementById(id))),
  };
});
check("the combo offers exactly the three view modes",
  viewModes.options.join() === "editor,editor-unlit,runtime"
    && viewModes.value === "editor",
  `${viewModes.options.join()} — showing ${viewModes.value}`);
check("every mode's flags name that mode back",
  Object.entries(viewModes.derived).every(([k, v]) => k === v),
  JSON.stringify(viewModes.derived));
check("the toolbar controls it replaced are gone",
  viewModes.gone.length === 0, `still present: ${viewModes.gone.join()}`);
check("both lighting rigs live on the Settings pane",
  viewModes.inSettings.length === 0, `not in Settings: ${viewModes.inSettings.join()}`);

// ---- 1d-tervicies. Save writes the manifest and the glb together ------------
// They describe one thing: the runtime loads the geometry from the glb and
// everything else - rooms, portals, collision, lamps - out of the manifest. A
// manifest saved without its glb is a ship whose description and geometry
// disagree.
const saveBoth = await page.evaluate(async () => {
  const realFetch = window.fetch;
  const posted = [];
  window.fetch = (url, opts) => {
    const u = String(url);
    if (opts?.method === "POST" && /\/api\/(layout|export|collision)/.test(u)) {
      posted.push(u.replace(location.origin, ""));
      return Promise.resolve(new Response(
        JSON.stringify({ ok: true, bytes: 1048576, path: "x" }),
        { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
  try {
    document.getElementById("btn-save").click();
    for (let i = 0; i < 200 && !posted.some((u) => u.includes("export")); i++) {
      await new Promise((r) => setTimeout(r, 50));
    }
  } finally { window.fetch = realFetch; }
  return { posted, status: document.getElementById("status-text").textContent };
});
check("one Save writes both files and says so",
  saveBoth.posted.some((u) => u.includes("/api/layout"))
    && saveBoth.posted.some((u) => u.includes("/api/export"))
    && /saved \d+ bytes/.test(saveBoth.status) && /MB →/.test(saveBoth.status),
  `${saveBoth.posted.join(" ")} — "${saveBoth.status}"`);

// ---- 1d-tervicies-bis. Save drops behaviour entries nothing carries --------
// Behaviours outlive the element that carried them while you work - deleting
// the last crate to put a better one down must not throw away how crates behave
// - but a save is the moment that stops being a working state and becomes the
// file the game reads. The names the exporter *derives* count as carried: an
// entry on `crate_primitive0` is how one part of a module gets a behaviour of
// its own, and anything that only knew about placement names would eat it.
const pruneSetup = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const l = await import("/js/lights.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.state.chunks = ["CH_PRUNE"]; ed.state.activeChunk = "CH_PRUNE";
  ed.setBehaviorDef("enableEntity", {});

  const named = await ed.placeAt(M, new V(0, 0, 0),
    { chunk: "CH_PRUNE", name: "crate", silent: true });
  const bare = await ed.placeAt(M, new V(6, 0, 0), { chunk: "CH_PRUNE", silent: true });
  const door = mk.addDoor(new V(3, 0, 0), { chunkA: "CH_PRUNE", silent: true });
  const lamp = l.addLight(named.id, { silent: true });

  for (const key of ["crate", "crate_primitive0", bare.id, door.id, "CH_PRUNE",
    `LIGHT_${lamp.id}`, "ghost", "oldName"]) {
    ed.addEntityBehavior(key, "enableEntity");
  }
  return { named: named.id, bare: bare.id, door: door.id, lamp: lamp.id,
    before: [...ed.state.entities.keys()].sort() };
});
const pruned = await page.evaluate(async () => {
  const realFetch = window.fetch;
  const posted = [];
  window.fetch = (url, opts) => {
    const u = String(url);
    if (opts?.method === "POST" && /\/api\/(layout|export|collision)/.test(u)) {
      posted.push(u.replace(location.origin, ""));
      return Promise.resolve(new Response(
        JSON.stringify({ ok: true, bytes: 1048576, path: "x" }),
        { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
  try {
    document.getElementById("btn-save").click();
    for (let i = 0; i < 200 && !posted.some((u) => u.includes("export")); i++) {
      await new Promise((r) => setTimeout(r, 50));
    }
  } finally { window.fetch = realFetch; }
  const ed = await import("/js/editor.js");
  const mf = await import("/js/manifest.js");
  return {
    after: [...ed.state.entities.keys()].sort(),
    exported: Object.keys(mf.buildManifest().entities || {}).sort(),
    status: document.getElementById("status-text").textContent,
  };
});
check("a save drops the entries no node answers to, and names them",
  !pruned.after.includes("ghost") && !pruned.after.includes("oldName")
    && / — dropped 2 unused behaviour entries: ghost, oldName/.test(pruned.status),
  `${JSON.stringify(pruned.after)} — "${pruned.status}"`);
check("and keeps every name the export will actually carry",
  ["crate", "crate_primitive0", pruneSetup.bare, pruneSetup.door, "CH_PRUNE",
    `LIGHT_${pruneSetup.lamp}`].every((k) => pruned.after.includes(k))
    && pruned.exported.includes("crate_primitive0"),
  JSON.stringify(pruned.after));

const pruneUndo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  await ed.undo();
  const back = [...ed.state.entities.keys()].sort();
  await ed.redo();
  return back;
});
check("undo brings a dropped entry back",
  pruneUndo.includes("ghost") && pruneUndo.includes("oldName"),
  JSON.stringify(pruneUndo));

const pruneAuto = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const main = await import("/js/main.js");
  const realFetch = window.fetch;
  window.fetch = (url, opts) => {
    if (opts?.method === "POST" && String(url).includes("/api/autosave")) {
      return Promise.resolve(new Response(
        JSON.stringify({ ok: true, bytes: 42, path: "ship_autosave.json" }),
        { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
  ed.addEntityBehavior("ghostAgain", "enableEntity");
  const minutes = ed.state.config.autoSaveMinutes;
  try {
    ed.setConfig("autoSaveMinutes", 2);
    await main.autoSaveNow();
  } finally {
    window.fetch = realFetch;
    ed.setConfig("autoSaveMinutes", minutes);
  }
  const held = ed.state.entities.has("ghostAgain");
  ed.clearAll(); ed.select([]);
  return held;
});
check("an auto-save is a recovery copy, so it drops nothing",
  pruneAuto === true, String(pruneAuto));

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
  const p = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
      bhvCount: document.getElementById("bhv-count").textContent,
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
  const q = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(4, 0, 0), { silent: true });
  ed.select([p.id, q.id]);
  const onMulti = { ...read(), posX: document.getElementById("pos-x").value };
  ed.clearAll(); ed.select([]);
  return { onPlacement, onDoor, backAgain, onMulti, pId: p.id, qId: q.id };
});
check("a placement shows its name, chunk and behaviour",
  stale.onPlacement.name === "weaponHolder" && stale.onPlacement.nameShown
    && stale.onPlacement.chunkShown && stale.onPlacement.behaviour
    && /^"weaponHolder" —/.test(stale.onPlacement.bhvCount),
  JSON.stringify(stale.onPlacement));
// The behaviour panel stays - a door carries behaviours under its id - but it
// has to be keyed to the door, which is the staleness this whole block is about.
check("selecting a door drops the previous element's name",
  stale.onDoor.name === "" && !stale.onDoor.nameShown
    && !stale.onDoor.chunkShown && stale.onDoor.behaviour
    && /^Door_\w+ — a door,/.test(stale.onDoor.bhvCount),
  JSON.stringify(stale.onDoor));
check("selecting a placement again brings the fields back",
  stale.backAgain.name === "weaponHolder" && stale.backAgain.nameShown
    && stale.backAgain.chunkShown
    && /^"weaponHolder" —/.test(stale.backAgain.bhvCount),
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });

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
  await i.armGhost("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight");
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

// ---- 2. Shift+wheel turns, Ctrl+wheel resizes, the bare wheel zooms --------
// The bare wheel is always the camera, which is what a wheel is for in a 3D
// view. The two edits it can do sit on the two modifiers.
await page.mouse.wheel(0, -120);
await page.waitForTimeout(150);
let rot = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").rotationQuaternion.toEulerAngles().y * 180 / Math.PI);
check("the bare wheel does not rotate the ghost", Math.abs(rot) < 0.01, `${rot.toFixed(2)}°`);

await page.keyboard.down("Shift");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Shift");
await page.waitForTimeout(200);
rot = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").rotationQuaternion.toEulerAngles().y * 180 / Math.PI);
check("Shift+wheel turns the ghost 90°", Math.abs(Math.abs(rot) - 90) < 0.01, `${rot.toFixed(2)}°`);

// and the other way, which is what retired the negative steps
await page.keyboard.down("Shift");
await page.mouse.wheel(0, 120);
await page.keyboard.up("Shift");
await page.waitForTimeout(200);
const rotBack = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").rotationQuaternion.toEulerAngles().y * 180 / Math.PI);
check("and back the other way, which is why a step needs no sign",
  Math.abs(rotBack) < 0.01, `${rot.toFixed(2)}° -> ${rotBack.toFixed(2)}°`);

// leave it turned, which is what the placement checks below read
await page.keyboard.down("Shift");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Shift");
await page.waitForTimeout(200);

await page.keyboard.down("Control");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Control");
await page.waitForTimeout(150);
let scl = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").scaling.asArray());
check("Ctrl+wheel scales ghost +0.1", scl.every((v) => Math.abs(v - 1.1) < 1e-6), JSON.stringify(scl));

// an edit, so it must not also dolly
const camHeld = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
await page.keyboard.down("Control");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Control");
await page.waitForTimeout(400);
const camStill = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
check("Ctrl+wheel resizes without moving the camera",
  JSON.stringify(camHeld) === JSON.stringify(camStill),
  `${camHeld.map((v) => v.toFixed(1))} -> ${camStill.map((v) => v.toFixed(1))}`);
await page.keyboard.down("Control");
await page.mouse.wheel(0, 120);            // undo the extra scale
await page.keyboard.up("Control");
await page.waitForTimeout(150);

// the bare wheel dollies the camera
const posBefore = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
await page.mouse.wheel(0, -240);
await page.waitForTimeout(600);
const posAfter = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
const moved = Math.hypot(posAfter[0] - posBefore[0], posAfter[1] - posBefore[1], posAfter[2] - posBefore[2]);
check("the bare wheel dollies the camera", moved > 0.5, `moved ${moved.toFixed(2)} m`);

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
await page.keyboard.press("r");
await page.waitForTimeout(120);
let axes = await page.evaluate(async () => {
  const s = (await import("/js/editor.js")).state;
  return { rot: s.rotAxis, scl: s.scaleAxis };
});
check("R cycles rotation axis", axes.rot === "x", `y -> ${axes.rot}`);
await page.keyboard.press("f");
await page.waitForTimeout(120);
axes = await page.evaluate(async () => {
  const s = (await import("/js/editor.js")).state;
  return { rot: s.rotAxis, scl: s.scaleAxis };
});
check("F cycles scale axis", axes.scl === "x", `all -> ${axes.scl}`);

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
// Hover no longer redirects an edit, so with nothing selected there is nothing
// to act on - pointing at a wall must not make it the thing R turns.
const hoverIsCurrent = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  ed.select([]);
  return { kind: i.currentElement()?.kind ?? null, hovered: !!i.hoveredId() };
});
check("a merely hovered element is not what an edit acts on",
  hoverIsCurrent.hovered && hoverIsCurrent.kind === null, JSON.stringify(hoverIsCurrent));

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

// A turn leaves a merely hovered element alone: edits go to the selection, so
// that resting the pointer somewhere cannot decide what the next one turns.
await page.evaluate(async () =>
  (await import("/js/interact.js")).rotateCurrent(1));
await page.waitForTimeout(200);
const rotAfterHover = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return { rot: ed.eulerOf([...ed.state.placements.values()][0].node)[1],
    hovered: i.hoveredId(), selected: ed.state.selection.length };
});
check("a turn does not reach a merely hovered element",
  !!rotAfterHover.hovered && rotAfterHover.selected === 0
    && Math.abs(rotAfterHover.rot - rotBefore) < 1e-6,
  `hovering ${rotAfterHover.hovered}, held at ${rotBefore.toFixed(1)}`);
// ...and turns it once it is actually selected
const rotWhenSelected = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const e = [...ed.state.placements.values()][0];
  ed.select([e.id]);
  i.rotateCurrent(1);
  return ed.eulerOf(e.node)[1];
});
check("but it does turn the selection",
  Math.abs(Math.abs(rotWhenSelected - rotBefore) - 90) < 0.01,
  `${rotBefore.toFixed(1)} -> ${rotWhenSelected.toFixed(1)}`);

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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
check("a turn goes to every selected element",
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
check("the selection is edited even with something else under the cursor",
  precedence.hovered === precedenceSetup.bId && precedence.kind === "selection"
    && precedence.aMoved && !precedence.bMoved,
  `current=${precedence.kind}, hovered moved=${precedence.bMoved}, selected moved=${precedence.aMoved}`);

// pointer off everything -> still the selection, unchanged
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
check("and with nothing hovered it is the selection just the same",
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
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
const press = (key, code, opts = {}) => page.evaluate(([k, c, o]) => window.dispatchEvent(
  new KeyboardEvent("keydown", { key: k, code: c, bubbles: true, ...o })), [key, code, opts]);

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
check("a mirror flips on the chosen axis",
  flip.oneX[0] === -1 && flip.oneX[2] === 1 && flip.oneZ[0] === -1 && flip.oneZ[2] === -1,
  `x-flip ${JSON.stringify(flip.oneX)}, then z-flip ${JSON.stringify(flip.oneZ)}`);
check("scale axis 'all' flips X, not all three",
  flip.allResult.axis === "x" && JSON.stringify(flip.allScale) === JSON.stringify([-1, 1, 1]),
  `axis=${flip.allResult.axis}, scale ${JSON.stringify(flip.allScale)}`);
check("a mirror flips the whole selection",
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
await press("f", "KeyF", { altKey: true });
await page.waitForTimeout(150);
const midFlip = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  return { dragging: i.isDragging(), scale: [...ed.state.placements.values()][0].node.scaling.asArray() };
});
await page.mouse.up();
await page.waitForTimeout(200);
check("Alt+F flips the element being dragged",
  midFlip.dragging && midFlip.scale[0] === -1,
  `dragging=${midFlip.dragging}, scale ${JSON.stringify(midFlip.scale)}`);

// and the ghost, before it is even placed
await page.evaluate((m) => import("/js/palette.js").then((p) => p.setBrush(m)), MODULE);
await page.waitForTimeout(1000);
await press("f", "KeyF", { altKey: true });
await page.waitForTimeout(150);
const ghostFlip = await page.evaluate(() =>
  window.__scene.getTransformNodeByName("GHOST").scaling.asArray());
check("Alt+F flips the ghost before placing", ghostFlip[0] === -1,
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
  await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_Simple", new BABYLON.Vector3(0, 0, 0));
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
  exposure.clampHi === 4 && exposure.clampLo === 0.15,
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

// ---- 1h-bis. the kit picker --------------------------------------------------
// The library spans five unrelated packs. Merged, their categories are one
// strip of 27 tabs where Walls, Rocks and Potions sit side by side and group
// nothing, so the palette shows one kit at a time and the tabs are that kit's.
await page.fill("#palette-search", "");
await page.waitForTimeout(400);

const kitPicker = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  const sel = document.getElementById("palette-kit");
  const head = document.querySelector(".palette-head");
  return {
    exists: !!sel,
    // above the search box: it decides what the search is searching
    beforeSearch: [...head.children].indexOf(sel)
      < [...head.children].indexOf(document.getElementById("palette-search")),
    options: [...sel.options].map((o) => o.value),
        value: sel.value,
        fallback: kit.defaultKit(),
        tabs: [...document.getElementById("palette-tabs").children].map((b) => b.textContent),
        tiles: document.querySelectorAll("#palette-list .item").length,
    expected: kit.getCatalogue().kits.find((k) => k.name === sel.value)?.count,
    tip: sel.title,
  };
});
check("the palette has a kit picker, above the search box", kitPicker.exists && kitPicker.beforeSearch, JSON.stringify(kitPicker.exists));
// Alphabetical, not config order: the picker is read to find a kit, and an
// order decided in config.json is one nobody at the keyboard can predict.
check(
    "it offers every kit on disk, by name",
    kitPicker.options.length >= 2 && String(kitPicker.options) === String([...kitPicker.options].sort((a, b) => a.localeCompare(b))),
    `[${kitPicker.options}]`
);
// What config.json still decides: which of them to land in. "Aquanova" sorts
// first, and is not the pack the ship is built from.
check(
    "and starts on the kit the config names, not the first letter",
    kitPicker.value === "Modular SciFi MegaKit" && kitPicker.fallback === "Modular SciFi MegaKit",
    `on "${kitPicker.value}", catalogue says "${kitPicker.fallback}"`
);
check("the tabs are the chosen kit's categories, not every kit's",
  kitPicker.tabs[0] === "All" && kitPicker.tabs.includes("Walls")
  && !kitPicker.tabs.includes("Enemies") && !kitPicker.tabs.includes("Rocks"),
  `[${kitPicker.tabs}]`);
check("and every tile shown belongs to that kit",
  kitPicker.tiles > 0 && kitPicker.tiles === kitPicker.expected,
  `${kitPicker.tiles} tiles, kit holds ${kitPicker.expected}`);

// A category belongs to the kit it came from: leaving "Walls" selected while
// showing a pack that has none would empty the palette with no clue why.
const kitSwitch = await page.evaluate(async () => {
    const pal = await import("/js/palette.js");
    const cat = (await import("/js/kit.js")).getCatalogue();
    const sel = document.getElementById("palette-kit");
    const tabs = () => [...document.getElementById("palette-tabs").children].map((b) => b.textContent);
  const pick = (name) => {
    sel.value = name;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    };
    // Deliberately a kit with no Walls of its own, since what is being watched
    // is a tab the new kit has no answer for. Which kit that is depends on what
    // is installed - the picker is alphabetical, and the name next to the
    // MegaKit's is an accident of spelling - so it is looked up rather than
    // taken to be the one after it.
    const info = (n) => cat.kits.find((k) => k.name === n);
    const other = [...sel.options].map((o) => o.value).find((n) => n !== "Modular SciFi MegaKit" && info(n)?.count > 0 && !info(n).categories.includes("Walls"));

    [...document.getElementById("palette-tabs").children].find((b) => b.textContent === "Walls").click();
  const narrowed = document.querySelectorAll("#palette-list .item").length;

  pick(other);
  const after = {
    tabs: tabs(),
    active: [...document.getElementById("palette-tabs").children]
      .find((b) => b.classList.contains("active"))?.textContent,
    tiles: document.querySelectorAll("#palette-list .item").length,
    kits: new Set([...document.querySelectorAll("#palette-list .item")]
      .map((el) => el.dataset.id.split("/")[0])),
    stored: localStorage.getItem("paletteKit"),
  };
  pick("Modular SciFi MegaKit");
  return {
    other, narrowed, ...after, kits: [...after.kits],
    back: document.querySelectorAll("#palette-list .item").length,
    pal: typeof pal.refreshPalette,
  };
});
check("a tab narrows the list to its category",
  kitSwitch.narrowed > 0 && kitSwitch.narrowed < kitPicker.expected,
  `${kitSwitch.narrowed} tiles`);
check("switching kit shows only that kit's modules",
  kitSwitch.kits.length === 1 && kitSwitch.kits[0] === kitSwitch.other && kitSwitch.tiles > 0,
  `${kitSwitch.tiles} tiles from [${kitSwitch.kits}]`);
check("and drops a category the new kit does not have, rather than showing nothing",
  kitSwitch.active === "All" && !kitSwitch.tabs.includes("Walls"),
  `active "${kitSwitch.active}" of [${kitSwitch.tabs}]`);
check("the choice is remembered for the next session",
  kitSwitch.stored === kitSwitch.other, `stored "${kitSwitch.stored}"`);
check("and switching back restores the first kit's list",
  kitSwitch.back === kitPicker.expected, `${kitSwitch.back} tiles`);

// A module nobody has previewed has to be loaded and rendered before there is
// a picture at all, and a fresh kit queues a hundred and fifty of them. The
// tile used to sit blank throughout, which reads as a broken image.
const pending = await page.evaluate(async () => {
  const t = await import("/js/thumbs.js");
  const kit = await import("/js/kit.js");
  const mods = [];
  for (const c of kit.getCatalogue().categories) {
    for (const m of c.modules) if (m.kit === "Modular SciFi MegaKit") mods.push(m);
  }
  for (const m of mods.slice(0, 40)) t.forgetCached(m.id);

  const pal = await import("/js/palette.js");
  pal.refreshPalette();
  await new Promise((r) => setTimeout(r, 120));
  const tile = document.querySelector("#palette-list .item.thumb-pending");
  const msg = tile ? getComputedStyle(tile, "::before").content : null;
  const covers = tile
    ? getComputedStyle(tile, "::before").position === "absolute" : false;
  // and it goes away again once the picture is there
  const gone = await new Promise((resolve) => {
    const until = Date.now() + 30000;
    const iv = setInterval(() => {
      const still = document.querySelectorAll("#palette-list .item.thumb-pending").length;
      if (!still || Date.now() > until) { clearInterval(iv); resolve(still); }
    }, 200);
  });
  const withSrc = [...document.querySelectorAll("#palette-list .item img.thumb")]
    .filter((i) => i.src).length;
  return { msg, covers, gone, withSrc };
});
check("a tile being rendered says so instead of showing a blank thumbnail",
  /Generating preview/.test(pending.msg || "") && pending.covers,
  `${pending.msg}`);
check("and the message goes away when the picture arrives",
  pending.gone === 0 && pending.withSrc > 0,
  `${pending.gone} still pending, ${pending.withSrc} filled`);

// FBX packs are read in the file's own units - Babylon parses UnitScaleFactor
// and never applies it - so a tree arrives 248 units tall. Everything else here
// is metres, so placing one dropped a 248-metre tree beside a 3-metre corridor.
const fbxUnits = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  const cm = new Uint8Array(64).fill(0);
  const fbx = kit.getCatalogue().kits.find((k) => k.name === "Ultimate Nature Pack");
  let size = null;
  if (fbx) {
    let id = null;
    for (const c of kit.getCatalogue().categories) {
      for (const m of c.modules) if (m.kit === fbx.name && m.name === "CommonTree_1") id = m.id;
    }
    if (id) {
      const b = await kit.moduleBounds(id);
      const s = b.max.subtract(b.min);
      size = [+s.x.toFixed(2), +s.y.toFixed(2), +s.z.toFixed(2)];
    }
  }
  return {
    hasKit: !!fbx,
    size,
    // a file that says nothing falls back to the FBX default, the centimetre
    fallback: kit.fbxMetresPerUnit(cm.buffer),
    // a format is packaging, not a category: a kit read through its `glTF/` or
    // `FBX/` folder would show one tab holding everything
    formatTabs: kit.getCatalogue().kits.flatMap((k) =>
      k.categories.filter((c) => /^(gltf|glb|fbx|obj|blend|source)$/i.test(c)).map((c) => `${k.name}/${c}`)),
  };
});
check("an FBX module is brought into metres from the unit its file declares",
  !fbxUnits.hasKit || (fbxUnits.size && fbxUnits.size[1] > 1 && fbxUnits.size[1] < 6),
  `CommonTree_1 is ${fbxUnits.size} m`);
check("and a file that declares nothing is read as centimetres, the FBX default",
  fbxUnits.fallback === 0.01, `${fbxUnits.fallback}`);
check("no kit offers a tab named after a file format",
  fbxUnits.formatTabs.length === 0, `[${fbxUnits.formatTabs}]`);

// Every kit keeps its texture atlases at the kit root while its models sit in
// category folders, so each one leans on the loader rule that redirects a bare
// filename back up. Sorting a kit into folders is what turns that rule on for
// it, and nothing else in these suites loads a module from a second kit. The
// turntable is rendered too, because that is a second load of the same module
// through the thumbnail scene - any miss is a 404 the harness records.
const foldered = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  const thumbs = await import("/js/thumbs.js");
  const out = {};
  for (const name of ["Prop_Crate", "Prop_Barrel"]) {
    let mod = null;
    for (const c of kit.getCatalogue().categories) {
      for (const m of c.modules) if (m.name === name && m.kit !== "Modular SciFi MegaKit") mod = m;
    }
    if (!mod) continue;
    const proto = await kit.getProto(mod.id);
    const textures = proto.parts
      .map((p) => p.mesh.material && (p.mesh.material.albedoTexture || p.mesh.material.diffuseTexture))
      .filter(Boolean);
    await thumbs.requestTurntable(mod);
    out[name] = { id: mod.id, ready: textures.length > 0 && textures.every((t) => t.isReady()) };
  }
  return out;
});
check("a module from another kit finds the atlas its kit keeps at the root",
  Object.values(foldered).length > 0 && Object.values(foldered).every((m) => m.ready),
  JSON.stringify(foldered));

// A module built for this ship out of parts of a bought pack wears that pack's
// trim, and Blender writes the path from the .gltf to the image: a URI that
// climbs. Babylon rejects any ".." outright, so kit.js resolves those itself,
// under one rule - the reference has to land inside a kit. Checked as a
// function first, because the interesting cases are the ones that must be
// refused and none of them can be authored into a real pack.
const crossKit = await page.evaluate(async () => {
    const kit = await import("/js/kit.js");
    const bases = ["/assets/kits/Aquanova/", "/assets/kits/Modular%20SciFi%20MegaKit/"];
    const from = (uri) => kit.resolveKitUri("/assets/kits/Aquanova/Walls/", uri, bases);
    const mega = kit.getCatalogue().kits.find((k) => k.name === "Modular SciFi MegaKit");
    return {
        climb: from("../../Modular SciFi MegaKit/T_Trim_03_Normal.png"),
        up: from("../T.png"),
        beside: from("T.png"),
        escape: from("../../../../index.html"),
        offsite: from("https://example.com/T.png"),
        // Off the live catalogue, whose bases are already encoded, and in the
        // shape the app uses: a path locally, a full URL against the CDN.
        live: kit.resolveKitUri(`${mega.base}Walls/`, "../T_Trim_01_ORM.png"),
        want: `${mega.base}T_Trim_01_ORM.png`,
    };
});
check(
    "a texture reference may climb into another kit, and no further",
    crossKit.climb === "/assets/kits/Modular%20SciFi%20MegaKit/T_Trim_03_Normal.png" &&
        crossKit.up === "/assets/kits/Aquanova/T.png" &&
        crossKit.beside === "/assets/kits/Aquanova/Walls/T.png" &&
        crossKit.escape === null &&
        crossKit.offsite === null &&
        crossKit.live === crossKit.want,
    JSON.stringify(crossKit)
);

// And end to end, because the loader hook is `_loadUriAsync` and an extension
// that spells it `loadUriAsync` registers, lists, and is never consulted - a
// mistake nothing but a real load can catch. The file is built here rather
// than kept in a kit: it is the loader that is under test, and a fixture in
// BabylonAssets would be one more thing to keep in step.
const crossKitLoad = await page.evaluate(async () => {
    const kit = await import("/js/kit.js");
    const mega = kit.getCatalogue().kits.find((k) => k.name === "Modular SciFi MegaKit");
    const data = new Uint8Array(60);
    new Float32Array(data.buffer, 0, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    new Float32Array(data.buffer, 36, 6).set([0, 0, 1, 0, 0, 1]);
    const b64 = btoa(String.fromCharCode(...data));
    const build = (uri) =>
        JSON.stringify({
            asset: { version: "2.0" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ mesh: 0 }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
            materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
            textures: [{ source: 0 }],
            images: [{ mimeType: "image/png", uri }],
            buffers: [{ byteLength: 60, uri: `data:application/octet-stream;base64,${b64}` }],
            bufferViews: [
                { buffer: 0, byteOffset: 0, byteLength: 36 },
                { buffer: 0, byteOffset: 36, byteLength: 24 },
            ],
            accessors: [
                { bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] },
                { bufferView: 1, componentType: 5126, count: 3, type: "VEC2" },
            ],
        });
    const load = (uri) => BABYLON.SceneLoader.LoadAssetContainerAsync(`${mega.base}Walls/`, new File([build(uri)], "probe.gltf"), window.__scene);

    const out = {};
    const box = await load("../T_Trim_01_ORM.png");
    const tex = box.materials[0]?.albedoTexture || null;
    for (let i = 0; i < 40 && tex && !tex.isReady(); i++) {
        await new Promise((done) => setTimeout(done, 100));
    }
    out.size = tex?.isReady() ? tex.getSize() : null;
    box.dispose();

    // Same file, one folder too far: the message has to name what it refused.
    try {
        (await load("../../../../index.html")).dispose();
        out.refused = "loaded anyway";
    } catch (e) {
        out.refused = e.message;
    }
    return out;
});
check(
    "a climbing reference loads the other kit's atlas, and a wilder one is refused",
    crossKitLoad.size && crossKitLoad.size.width >= 1024 && crossKitLoad.size.width === crossKitLoad.size.height && /points outside the kits/.test(crossKitLoad.refused),
    `${JSON.stringify(crossKitLoad.size)}, refused with "${String(crossKitLoad.refused).slice(-90)}"`
);

// The editor server is started once and left running for days, and it serves
// public/ off disk - so the page can end up newer than the process answering
// it. That skew took the texture redirect out silently: a catalogue without
// `modelDirs` built no rules, and every kit 404'd its atlases while looking
// like a broken kit rather than a stale server.
const shape = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  const ok = { categories: [], kits: [{ name: "K", modelDirs: [], rootTextures: [] }] };
  const say = (cat) => { try { kit.assertCatalogueShape(cat); return null; } catch (e) { return e.message; } };
  return {
    current: say(kit.getCatalogue()),
    good: say(ok),
    noField: say({ categories: [], kits: [{ name: "K", rootTextures: [] }] }),
    noKits: say({ categories: [] }),
  };
});
check("the catalogue the server is serving is one this page can read",
  shape.current === null && shape.good === null, `${shape.current || shape.good}`);
check("and a catalogue from an older server is refused, naming the cure",
  shape.noField && shape.noField.includes("modelDirs") && /restart it/i.test(shape.noField)
  && shape.noKits && shape.noKits.includes('"kits"'),
  `${shape.noField}`);

// Two materials are the same material when they read the same textures, not
// when they share a name. Every Pirate model calls its material `Atlas` and
// embeds its own 32x32 slice of the palette, so a name-keyed cache handed the
// first-loaded module's atlas to all of them: the characters came out grey
// except for the prop in their hand, which reads from a row the props' atlas
// also fills in. Five more names - MI_Trim_01 and friends - are shared between
// the MegaKit and the Essentials Kit over different atlases.
//
// The prop is loaded first here on purpose: that is the order that used to
// break, and the check is worthless in the other one.
const identity = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  const find = (n, k) => {
    for (const c of kit.getCatalogue().categories) {
      for (const m of c.modules) if (m.name === n && (!k || m.kit === k)) return m;
    }
    return null;
  };
  const out = { pirate: null, mega: null };

  const prop = find("Prop_Barrel", "Pirate Kit");
  const character = find("Characters_Anne", "Pirate Kit");
  if (prop && character) {
    const a = await kit.getProto(prop.id);
    const b = await kit.getProto(character.id);
    const urlOf = (proto) => proto.parts.map((p) => p.mesh.material.albedoTexture?.url);
    const propUrls = urlOf(a), charUrls = urlOf(b);
    out.pirate = {
      // the character's own atlas, on every one of its parts - body and the
      // weapon in its hand alike
      character: charUrls.every((u) => u && u.includes("Characters_Anne")),
      // and not the prop's, which is what it used to be given
      notTheProps: !charUrls.some((u) => propUrls.includes(u)),
      shared: b.parts.every((p) => p.mesh.material === b.parts[0].mesh.material),
      urls: [propUrls[0], charUrls[0]],
    };
  }

  // What the sharing is for has to keep working: the MegaKit names the same
  // files at its root from every module, so those must still be one material.
  const mega = [];
  for (const c of kit.getCatalogue().categories) {
    for (const m of c.modules) if (m.kit === "Modular SciFi MegaKit") mega.push(m);
  }
  const before = kit.materialRegistry.size;
  for (const m of mega.slice(0, 12)) await kit.getProto(m.id);
  const seen = new Map();
  for (const m of mega.slice(0, 12)) {
    for (const p of (await kit.getProto(m.id)).parts) {
      const name = p.mesh.material.name;
      if (!seen.has(name)) seen.set(name, new Set());
      seen.get(name).add(p.mesh.material);
    }
  }
  out.mega = {
    added: kit.materialRegistry.size - before,
    names: seen.size,
    // one material per name across all twelve modules, as before
    copies: Math.max(...[...seen.values()].map((s) => s.size)),
  };
  return out;
});
check("a material is shared over its textures, not its name",
  identity.pirate?.character && identity.pirate.notTheProps && identity.pirate.shared,
  JSON.stringify(identity.pirate?.urls));
check("and the kit that names one atlas everywhere still shares it",
  identity.mega && identity.mega.copies === 1 && identity.mega.names > 0,
  JSON.stringify(identity.mega));

// Quaternius' .fbx packs are exported with every face flat - the normals in
// the file itself are as many as the mesh has corners, up to 159 degrees apart
// - so the smoothing is ours to do. It is opt-in per kit, because doing it to
// the sci-fi kits would change the ship and every probe captured from it.
const shading = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  const splitFraction = async (mod) => {
    const proto = await kit.getProto(mod.id);
    const m = proto.parts[0].mesh;
    const pos = m.getVerticesData("position"), nor = m.getVerticesData("normal");
    const at = new Map();
    for (let i = 0; i < pos.length; i += 3) {
      const k = [pos[i], pos[i + 1], pos[i + 2]].map((v) => Math.round(v * 1e4)).join(",");
      if (!at.has(k)) at.set(k, []);
      at.get(k).push(i / 3);
    }
    let split = 0;
    for (const g of at.values()) {
      let diff = false;
      for (const a of g) for (const b of g) {
        const d = nor[a * 3] * nor[b * 3] + nor[a * 3 + 1] * nor[b * 3 + 1] + nor[a * 3 + 2] * nor[b * 3 + 2];
        if (Math.acos(Math.min(1, Math.max(-1, d))) > 0.02) diff = true;
      }
      if (diff) split++;
    }
    return { positions: at.size, split, frac: +(split / at.size).toFixed(2) };
  };
  const find = (n, k) => {
    for (const c of kit.getCatalogue().categories) {
      for (const m of c.modules) if (m.name === n && (!k || m.kit === k)) return m;
    }
    return null;
  };
  const tree = find("CommonTree_1", "Ultimate Nature Pack");
  const wall = [];
  for (const c of kit.getCatalogue().categories) {
    for (const m of c.modules) if (m.kit === "Modular SciFi MegaKit") wall.push(m);
  }
  return {
    configured: kit.getKitReading("Ultimate Nature Pack")?.smoothNormalsBelowDeg ?? null,
    megaKitLeftAlone: kit.getKitReading("Modular SciFi MegaKit"),
    tree: tree ? await splitFraction(tree) : null,
  };
});
check("a pack exported flat is smoothed, creases kept",
  shading.configured > 0 && shading.megaKitLeftAlone === null
  && shading.tree && shading.tree.frac > 0 && shading.tree.frac < 0.2,
  JSON.stringify(shading.tree));

// The RPG pack's six see-through materials are the only ones in it that arrive
// with no colour: Blender's FBX exporter writes Phong properties out of a
// Principled BSDF and has nothing to write for a transparent shader. Babylon
// then gave all of them its default 0.8 grey, which made a filled potion
// identical to an empty one - liquid, glass and air all the same colour.
const potions = await page.evaluate(async () => {
  const kit = await import("/js/kit.js");
  const find = (n) => {
    for (const c of kit.getCatalogue().categories) {
      for (const m of c.modules) if (m.name === n && m.kit === "Ultimate RPG Items Pack") return m;
    }
    return null;
  };
  const materials = async (name) => {
    const mod = find(name); if (!mod) return null;
    const out = {};
    for (const p of (await kit.getProto(mod.id)).parts) {
      for (const sub of p.mesh.material.subMaterials || [p.mesh.material]) {
        const c = sub.diffuseColor || sub.albedoColor;
        out[sub.name] = { rgb: c && [c.r, c.g, c.b].map((v) => +v.toFixed(3)), alpha: sub.alpha };
      }
    }
    return out;
  };
  return { filled: await materials("Potion1_Filled"), empty: await materials("Potion1_Empty") };
});
const liquid = potions.filled?.Liquid_Red;
check("a liquid the .fbx forgot to colour is coloured, and the glass sees through",
  liquid && liquid.rgb[0] > 0.5 && liquid.rgb[1] < 0.2 && liquid.alpha === 1
  && potions.filled.Glass.alpha < 1 && !potions.empty.Liquid_Red,
  JSON.stringify(potions.filled));

await page.fill("#palette-search", "");
await page.waitForTimeout(300);

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
    await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_3Plates", new V(k * 4, 0, 0));
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0),
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

// ---- ... and the lamps riding it -------------------------------------------
// A light is part of what an element IS, so a duplicate has to bring the lamps
// the source is wearing *now*. The ghost carries the id it came from for
// exactly this: without it the drop was an ordinary placement, so a lit panel
// came back with the kit's default lamp - every tuned intensity lost - and a
// lamp added by hand to something the kit does not light came back with none.
const dupLitSetup = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  // deliberately no clearAll: the axis section below still needs the wall the
  // block above placed
  i.cancelGhost(); ed.select([]);
  // a module the kit lights on its own, so a re-seed would show up as a lamp
  // back at its default numbers
  const src = await ed.placeAt("Modular SciFi MegaKit/Props/Prop_Light_Wide", new V(12, 0, 0),
    { silent: true, name: "port lamp" });
  const seeded = lt.lightsOf(src.id);
  lt.setLightPart(seeded[0].id, "runtime", { intensity: 7, range: 3, color: [1, 0, 0] });
  lt.setLightTransform(seeded[0].id, { offset: [0, 1.25, 0] });
  // and a second the kit would never have put there
  lt.addLight(src.id, {
    offset: [0, -1, 0], rotation: [0, 0, 180],
    runtime: { type: "spot", intensity: 42, angle: 33 }, silent: true,
  });
  ed.select([src.id]);
  ed.state.camera.position = new V(0, 14, -10);
  ed.state.camera.setTarget(new V(0, 0, 0));
  const record = (l) => ({ offset: lt.lightOffset(l), rotation: lt.lightRotation(l), runtime: l.runtime });
  return { id: src.id, ids: lt.lightsOf(src.id).map((l) => l.id), lamps: lt.lightsOf(src.id).map(record) };
});
await page.waitForTimeout(600);
await page.mouse.move(1150, 520, { steps: 3 });
await page.keyboard.press("Control+d");
await page.waitForTimeout(900);
await page.mouse.down(); await page.mouse.up();
await page.waitForTimeout(900);
const dupLit = await page.evaluate(async ({ srcId, srcLampIds }) => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const copyId = ed.state.selection[0];
  const copy = ed.entryOf(copyId);
  const record = (l) => ({ offset: lt.lightOffset(l), rotation: lt.lightRotation(l), runtime: l.runtime });
  const lamps = lt.lightsOf(copyId);
  return {
    fresh: copyId !== srcId,
    name: copy?.name,
    lamps: lamps.map(record),
    // entries of the copy's own, not a second handle on the source's
    distinct: lamps.length > 0 && lamps.every((l) => l.owner === copyId && !srcLampIds.includes(l.id)),
    source: lt.lightsOf(srcId).map(record),
  };
}, { srcId: dupLitSetup.id, srcLampIds: dupLitSetup.ids });
check("Ctrl+D copies the lamps the source is wearing, not the kit's defaults",
  dupLit.fresh && dupLit.lamps.length === 2 && dupLit.distinct
    && JSON.stringify(dupLit.lamps) === JSON.stringify(dupLitSetup.lamps),
  JSON.stringify({ carried: dupLit.lamps, wanted: dupLitSetup.lamps }));
check("the copy is nameless, so it is its own node, and leaves the source alone",
  dupLit.name === ""
    && JSON.stringify(dupLit.source) === JSON.stringify(dupLitSetup.lamps),
  `name=${JSON.stringify(dupLit.name)}, source=${JSON.stringify(dupLit.source)}`);

// ---- Ctrl+D always arms on the floor plane ----------------------------------
// In Y mode the cursor drives the *build plane* rather than the ghost's own
// height, and clears `baseY` to take it over - which is the very height the copy
// was given so it would appear beside its source. The copy landed on the plane
// instead: measured 9.5 m below, and from a camera at deck level that is off the
// top of the screen. Ctrl+D sets the axis back rather than arming something you
// cannot see, and says so, because a mode that changes itself quietly is worse
// than one you change by hand.
const dupAxis = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost();
  const e = ed.state.placements.get(id);
  e.node.position.set(4, 9.5, 6);
  ed.select([id]);
  ed.setGridElevation(0);                    // the plane, far below the source
  i.setDragAxis("y");
  const c = ed.state.camera;
  c.cameraDirection.setAll(0); c.cameraRotation.set(0, 0);
  c.position.set(4, 26, 6.001); c.rotation.set(Math.PI / 2, 0, 0);   // straight down
  ed.state.scene.render();
  return { was: ed.state.dragAxis, sourceY: e.node.position.y, gridY: ed.state.gridY };
}, dupSetup.id);
await page.waitForTimeout(400);
const dupAxisCanvas = await page.evaluate(() =>
  document.getElementById("render-canvas").getBoundingClientRect().toJSON());
const dupAxisMid = {
  x: Math.round(dupAxisCanvas.x + dupAxisCanvas.width / 2),
  y: Math.round(dupAxisCanvas.y + dupAxisCanvas.height / 2),
};
await page.mouse.move(dupAxisMid.x, dupAxisMid.y, { steps: 3 });
await page.waitForTimeout(250);
await page.keyboard.press("Control+d");
await page.waitForTimeout(800);
await page.mouse.move(dupAxisMid.x + 3, dupAxisMid.y + 3);
await page.waitForTimeout(300);
const dupAxisGhost = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const n = ed.hooks.ghostNode?.();
  return {
    armed: !!n, y: n ? +n.position.y.toFixed(2) : null,
    axis: ed.state.dragAxis, combo: document.getElementById("drag-axis").value,
    gridY: +ed.state.gridY.toFixed(2),
    status: (document.getElementById("status")?.textContent || "").trim(),
  };
});
await page.keyboard.press("Escape");
await page.waitForTimeout(300);
check("Ctrl+D from Y mode puts the drag axis back on the floor",
  dupAxis.was === "y" && dupAxisGhost.axis === "xz" && dupAxisGhost.combo === "xz",
  `${dupAxis.was} -> ${dupAxisGhost.axis}, combo ${dupAxisGhost.combo}`);
check("so the copy arms at its source's height, not down on the build plane",
  dupAxisGhost.armed && Math.abs(dupAxisGhost.y - 9.5) < 0.01,
  `ghost at ${dupAxisGhost.y} m, plane at ${dupAxisGhost.gridY} m`);
check("and the status line says the axis moved",
  /X\/Z/.test(dupAxisGhost.status.split("\n")[0]), dupAxisGhost.status.split("\n")[0].trim());

// Every way of arming a ghost does it, not only Ctrl+D. The reset lives in
// interact.js beside the three functions that assign the ghost, because there
// are four routes in - a palette tile, the eyedropper, a shape button, a
// duplicate - and putting it in the click handlers missed the palette, which is
// the one the bug was reported against.
const armAxis = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const pal = await import("/js/palette.js");
  const module = ed.state.placements.get(id).module;
  const arm = async (how) => {
    i.cancelGhost();
    i.setDragAxis("y");
    ed.select([id]);
    await how();
    // setBrush fires armGhost without awaiting it, and armGhost loads the
    // module's prototype - so the ghost turns up a few frames later even though
    // the axis is set at once.
    for (let k = 0; k < 60 && !ed.hooks.ghostNode?.(); k++) {
      await new Promise((r) => setTimeout(r, 50));
    }
    const out = { axis: ed.state.dragAxis, combo: document.getElementById("drag-axis").value,
      ghost: !!ed.hooks.ghostNode?.() };
    i.cancelGhost();
    return out;
  };
  return {
    palette: await arm(() => pal.setBrush(module)),
    shape: await arm(() => i.armColliderGhost("box", { scaling: [1, 1, 1] })),
    copy: await arm(() => i.grabSelection({ copy: true })),
    // ...but a carry is moving what is already there, and raising it is a
    // perfectly good reason to be in Y mode
    carry: await arm(() => i.grabSelection()),
  };
}, dupSetup.id);
for (const [how, r] of [["a palette tile", armAxis.palette], ["a collision shape button", armAxis.shape],
  ["a Ctrl+D copy of a selection", armAxis.copy]]) {
  check(`arming from ${how} puts the axis back on the floor`,
    r.ghost && r.axis === "xz" && r.combo === "xz", JSON.stringify(r));
}
check("but an M carry is left in Y mode, raising being the whole point of it",
  armAxis.carry.ghost && armAxis.carry.axis === "y", JSON.stringify(armAxis.carry));

await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.setDragAxis("xz");
  ed.state.placements.get(id)?.node.position.set(0, 0, 0);
  ed.select([]);
}, dupSetup.id);

// ---- 1d-duovicies. the ghost carries a whole selection ---------------------
// It used to hold exactly one module, which is why Ctrl+D on a multi-selection
// duplicated in place and why moving anything needed a second mechanism.
const carrySet = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(10.27, 0, 3.4),
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
  const a = await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_Simple", new V(0, 0, 0), { silent: true });
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 12, 0), { silent: true });
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
check("duplicating keeps the source's height without moving the build plane",
  // Moving the plane was the old behaviour, and it meant Ctrl+D silently
  // changed where *everything placed afterwards* would land.
  Math.abs(highGhost.gridY - dupHigh.gridBefore) < 1e-6
    && Math.abs(highGhost.ghostY - dupHigh.sourceY) < 1e-6,
  `plane held at ${highGhost.gridY} m, ghost at ${highGhost.ghostY} m`
  + ` from a source at ${dupHigh.sourceY} m`);

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
  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(8, 0, 0), { silent: true });
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
  /Shift\+wheel/.test(tips.rotAxis) && /\bR\b/.test(tips.rotAxis)
    && /Ctrl\+wheel/.test(tips.scaleAxis) && /Alt\+F/.test(tips.scaleAxis)
    && /\bF\b/.test(tips.scaleAxis) && /\bV\b/.test(tips.dragAxis),
  `rot "${tips.rotAxis}" · scale "${tips.scaleAxis}"`);
check("Save still names Ctrl+S", /Ctrl\+S/.test(tips.save), `"${tips.save}"`);

// ---- a press has to look like a press ---------------------------------------
// Save, Load and Export glb all do their work somewhere else - a file on disk,
// a line in the status bar - so a press that missed looked exactly like one
// that worked. The measurement is taken in the *same task* as the click, so a
// transition cannot hide behind it: a fade-in is what broke the first attempt,
// the colour still climbing out of grey when the timer took the class off.
//
// Each button is swapped for a clone of itself for the duration. The clone
// carries the same id, classes and styling but none of the listeners, so the
// flash is measured on the real thing without Save actually saving over the
// test server's ship or Drop-to-plane actually moving the selection.
const ACCENT = "rgb(255, 138, 61)";
const tapped = await page.evaluate((accent) => {
  const flash = (id) => {
    const real = document.getElementById(id);
    if (!real) return { id, missing: true };
    const stunt = real.cloneNode(true);
    real.replaceWith(stunt);
    const idle = getComputedStyle(stunt).backgroundColor;
    stunt.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    const during = getComputedStyle(stunt).backgroundColor;
    const marked = stunt.classList.contains("tapped");
    stunt.replaceWith(real);
    return { id, idle, during, marked, lit: during === accent };
  };
  return {
    actions: ["btn-save", "btn-load", "btn-focus", "btn-ground"].map(flash),
    toggle: flash("btn-isolate"),
  };
}, ACCENT);
check("Save, Load, Export and the rest go orange the instant they are pressed",
  tapped.actions.every((r) => r.lit && r.marked),
  tapped.actions.map((r) => `${r.id} ${r.idle}->${r.during}`).join(", "));
// A toggle latches solid orange and stays there, which says the same thing for
// longer - flashing a slightly different orange first only muddies it.
check("a toggle button is left to its own latched state",
  tapped.toggle.marked === false, `class marked=${tapped.toggle.marked}`);

const tapClears = await page.evaluate(() => {
  const real = document.getElementById("btn-focus");
  const stunt = real.cloneNode(true);
  stunt.id = "tap-probe";
  real.after(stunt);
  stunt.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  return stunt.classList.contains("tapped");
});
await page.waitForTimeout(500);
const tapGone = await page.evaluate(() => {
  const stunt = document.getElementById("tap-probe");
  const still = stunt.classList.contains("tapped");
  stunt.remove();
  return still;
});
check("and the flash takes itself off again",
  tapClears && !tapGone, `on at click=${tapClears}, still on later=${tapGone}`);

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

// ---- World / Local: whose axes the move axis means --------------------------
// A modular kit turns every second wall 90 degrees, so "slide it along its
// length" is world Z on one and world X on the next. Local space says the axis
// belongs to the element, and the pair of combos reads as "which axis, whose".
//
// The wall is put back on world axes first: the checks below turn it, and every
// later block expects it where it started.
const spSetup = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const e = ed.state.placements.get(id);
  e.node.position.set(0, 0, 0);
  e.node.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(0, Math.PI / 2, 0);
  ed.select([id]);
  ed.state.snap.pos = 0;                       // measure the raw travel
  i.setDragAxis("x");
  ed.state.camera.cameraDirection.setAll(0);
  ed.state.camera.cameraRotation.set(0, 0);
  ed.state.camera.position = new V(0, 16, 0.001);
  ed.state.camera.setTarget(V.Zero());          // straight down: screen +x is world +x
  const bb = ed.worldBounds(e.node);
  return { centre: bb.min.add(bb.max).scale(0.5).asArray(),
    space: ed.state.axisSpace, combo: document.getElementById("axis-space").value,
    options: [...document.getElementById("axis-space").options].map((o) => o.value) };
}, qSetup.id);
await page.waitForTimeout(500);

check("the move space starts on the world's axes",
  spSetup.space === "world" && spSetup.combo === "world"
    && spSetup.options.join() === "world,local", JSON.stringify(spSetup));

// One gesture, four combinations: across the screen is world X, up the screen
// is world Z, and the wall's own X lies along world Z because it is turned.
async function spDrag(space, dx, dy) {
  const from = await page.evaluate(async ([id, s]) => {
    const ed = await import("/js/editor.js");
    const i = await import("/js/interact.js");
    const e = ed.state.placements.get(id);
    e.node.position.set(0, 0, 0);
    i.setAxisSpace(s);
    const bb = ed.worldBounds(e.node);
    return { pos: e.node.position.asArray(), centre: bb.min.add(bb.max).scale(0.5).asArray() };
  }, [qSetup.id, space]);
  const at = await screenOf(from.centre);
  await page.mouse.move(at.x, at.y, { steps: 4 });
  await page.waitForTimeout(250);
  await page.mouse.down();
  await page.mouse.move(at.x + dx, at.y + dy, { steps: 10 });
  await page.waitForTimeout(150);
  await page.mouse.up();
  await page.waitForTimeout(250);
  const to = await page.evaluate(async (id) =>
    (await import("/js/editor.js")).state.placements.get(id).node.position.asArray(), qSetup.id);
  return to.map((v, k) => +(v - from.pos[k]).toFixed(3));
}

const spWorldAcross = await spDrag("world", 150, 0);
const spWorldUp = await spDrag("world", 0, -150);
const spLocalAcross = await spDrag("local", 150, 0);
const spLocalUp = await spDrag("local", 0, -150);

check("world + X only: dragging across the screen slides on world X",
  Math.abs(spWorldAcross[0]) > 0.5 && Math.abs(spWorldAcross[2]) < 1e-6,
  `[${spWorldAcross}]`);
check("world + X only: dragging up the screen moves nothing at all",
  Math.hypot(...spWorldUp) < 1e-6, `[${spWorldUp}]`);
// The two are exact opposites, which is the whole claim: the axis really turned
// with the element rather than the constraint simply being dropped.
check("local + X only on a turned wall: up the screen slides it on world Z",
  Math.abs(spLocalUp[0]) < 1e-6 && Math.abs(spLocalUp[2]) > 0.5, `[${spLocalUp}]`);
check("local + X only on a turned wall: across the screen moves nothing",
  Math.hypot(...spLocalAcross) < 1e-6, `[${spLocalAcross}]`);
check("and the travel is the same length either way, only turned",
  Math.abs(Math.hypot(...spWorldAcross) - Math.hypot(...spLocalUp)) < 1e-2,
  `${Math.hypot(...spWorldAcross).toFixed(3)} vs ${Math.hypot(...spLocalUp).toFixed(3)}`);

// An arrow nudge names its own axis, so it ignores the axis combo - but it must
// still agree with a drag about which way that axis points.
const spNudge = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  const e = ed.state.placements.get(id);
  const run = (space, yawDeg) => {
    e.node.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(0, yawDeg * Math.PI / 180, 0);
    e.node.position.set(0, 0, 0);
    i.setAxisSpace(space);
    ed.nudgeSelection(new V(1, 0, 0));
    return e.node.position.asArray().map((v) => +v.toFixed(4));
  };
  const out = { worldTurned: run("world", 90), localFlat: run("local", 0),
    localTurned: run("local", 90), localDiagonal: run("local", 45) };
  i.setAxisSpace("world");
  return out;
}, qSetup.id);
check("an arrow nudge in world space is world X however the wall is turned",
  spNudge.worldTurned.join() === "1,0,0", `[${spNudge.worldTurned}]`);
check("in local space on an unturned element it is the same thing",
  spNudge.localFlat.join() === "1,0,0", `[${spNudge.localFlat}]`);
check("but on a turned wall it runs along the wall instead",
  Math.abs(spNudge.localTurned[0]) < 1e-3
    && Math.abs(Math.abs(spNudge.localTurned[2]) - 1) < 1e-3, `[${spNudge.localTurned}]`);
check("at 45 degrees it splits the step without changing its length",
  Math.abs(Math.hypot(...spNudge.localDiagonal) - 1) < 1e-3
    && Math.abs(spNudge.localDiagonal[0] - Math.abs(spNudge.localDiagonal[2])) < 1e-3,
  `[${spNudge.localDiagonal}]`);

// Y is the key, next to V, and the combo has to follow it - a mode you cannot
// see is a mode you will forget you are in.
await page.evaluate(() => document.getElementById("render-canvas").focus());
await page.keyboard.press("y");
await page.waitForTimeout(150);
const spAfterY = await page.evaluate(async () => ({
  state: (await import("/js/editor.js")).state.axisSpace,
  combo: document.getElementById("axis-space").value,
  status: document.getElementById("status")?.textContent || "",
}));
await page.keyboard.press("y");
await page.waitForTimeout(150);
const spBackY = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.axisSpace);
// Ctrl+Y is redo and is claimed before the switch: it must not toggle as well.
await page.keyboard.press("Control+y");
await page.waitForTimeout(150);
const spCtrlY = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.axisSpace);

check("Y switches to local space and the combo says so",
  spAfterY.state === "local" && spAfterY.combo === "local", JSON.stringify(spAfterY));
check("and it says which key goes back",
  /\bY\b/.test(spAfterY.status) && /local|own/i.test(spAfterY.status), spAfterY.status);
check("Y again returns to the world's axes", spBackY === "world", spBackY);
check("Ctrl+Y is still redo and leaves the space alone", spCtrlY === "world", spCtrlY);

// A combo change has to reach the state, not just the status line.
const spCombo = await page.evaluate(async () => {
  const el = document.getElementById("axis-space");
  el.value = "local";
  el.dispatchEvent(new Event("change", { bubbles: true }));
  const local = (await import("/js/editor.js")).state.axisSpace;
  el.value = "world";
  el.dispatchEvent(new Event("change", { bubbles: true }));
  return { local, world: (await import("/js/editor.js")).state.axisSpace };
});
check("the combo drives the mode as well as the key",
  spCombo.local === "local" && spCombo.world === "world", JSON.stringify(spCombo));

await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const e = ed.state.placements.get(id);
  e.node.rotationQuaternion = BABYLON.Quaternion.Identity();
  e.node.position.set(0, 0, 0);
  i.setAxisSpace("world"); i.setDragAxis("xz");
  ed.state.snap.pos = 1;
  ed.state.camera.position = new BABYLON.Vector3(0, 6, -14);
  ed.state.camera.setTarget(new BABYLON.Vector3(0, 1, 0));
}, qSetup.id);
await page.waitForTimeout(400);

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
  await i.armGhost("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight");
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Columns/Column_Large3", new V(0, 0, 0), { silent: true });
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
check("an unselected element under the cursor is hovered, but not edited",
  hoveredUnselected.hovered === exclusive.id && hoveredUnselected.kind === undefined,
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
  const e = await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_3Plates", new V(0, 0, 0), { silent: true });
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
const heldPointer = await page.evaluate(() => {
  const canvas = window.__scene.getEngine().getRenderingCanvas();
  return {
    cursor: canvas.style.cursor,
    locked: document.pointerLockElement === canvas,
  };
});
await page.mouse.up({ button: "right" });
await page.waitForTimeout(250);
const freePointer = await page.evaluate(() => ({
  cursor: window.__scene.getEngine().getRenderingCanvas().style.cursor,
  locked: document.pointerLockElement !== null,
}));
check("the cursor is hidden while the right button is held",
  rmbCursor !== "none" && heldPointer.cursor === "none" && freePointer.cursor !== "none",
  `"${rmbCursor}" -> "${heldPointer.cursor}" -> "${freePointer.cursor}"`);
check("RMB look locks the pointer until release",
  heldPointer.locked && !freePointer.locked,
  `held=${heldPointer.locked}, released=${!freePointer.locked}`);
await page.mouse.click(rmbPt.x, rmbPt.y, { button: "right" });
await page.waitForTimeout(200);
check("a quick RMB click cannot leave a late pointer lock behind",
  await page.evaluate(() => document.pointerLockElement === null));

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
const canvasMenuCount = await page.evaluate(() => window.__menus.length);

// the release that actually bites: outside the canvas, over the palette
await page.mouse.move(menuPts.mid.x, menuPts.mid.y, { steps: 3 });
await page.mouse.down({ button: "right" });
await page.mouse.move(menuPts.tile.x, menuPts.tile.y, { steps: 8 });
await page.mouse.up({ button: "right" });
await page.waitForTimeout(200);
const edgeMenuCount = await page.evaluate(() => window.__menus.length);

// and every other area, including a plain right-click that never went near the
// canvas: Ctrl+C/Ctrl+V still work, only the menu is gone
for (const at of [menuPts.search, menuPts.toolbar, menuPts.inspector]) {
  await page.mouse.click(at.x, at.y, { button: "right" });
  await page.waitForTimeout(150);
}

const menus = await page.evaluate(() => window.__menus);
check("a right-drag released on the canvas raises no menu",
  menus.slice(0, canvasMenuCount).every((m) => m.prevented === true),
  JSON.stringify(menus.slice(0, canvasMenuCount)));
check("a right-drag driven past the canvas edge raises no menu",
  menus.slice(canvasMenuCount, edgeMenuCount).every((m) => m.prevented === true),
  JSON.stringify(menus.slice(canvasMenuCount, edgeMenuCount)));
check("no editor area raises a menu, text fields included",
  menus.length === edgeMenuCount + 3
    && menus.slice(edgeMenuCount).every((m) => m.prevented === true),
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
    made.push((await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
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
  const wall = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
    made.push((await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
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

// ---- a copy is its own node ------------------------------------------------
// Behaviours hang off a node NAME, and names are shared on purpose - naming six
// crates "crate" is how one entry governs all six. A copy that kept its
// source's name therefore inherited the source's identity rather than its
// properties: one entry for both, `linked` unable to name one without the
// other, and no way to give the copy a behaviour of its own. So a copy arrives
// nameless, under its id, carrying its own deep copy of what the source had -
// unless Ctrl+Shift+D asks for the bare shape.
const dupBhv = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  ed.setBehaviorDef("playAnimation", { });
  const fan = await ed.placeAt("Modular SciFi MegaKit/Props/Prop_Fan_Small", new V(0, 2, 0), { silent: true });
  ed.renamePlacement(fan.id, "fanA");
  ed.addEntityBehavior("fanA", "playAnimation");
  ed.setEntityParams("fanA", 0, { animation: "Fan_Idle", loop: false });

  ed.select([fan.id]);
  await ed.duplicateSelected();
  const copyId = ed.state.selection[0];
  const copy = ed.entryOf(copyId);
  const carried = ed.entityBehaviors(copyId);
  // its own copy, not a second handle: editing the copy must not reach back
  ed.setEntityParams(copyId, 0, { animation: "Fan_Fast" });
  const sourceStill = ed.entityBehaviors("fanA")[0]?.animation;

  // the manifest names both nodes, and the .glb names the copy after its id
  const written = JSON.parse(JSON.stringify(mf.buildManifest().entities));
  const exported = mf.nodeNameOf(copy);

  // Ctrl+Shift+D is the same copy without them
  ed.select([fan.id]);
  await ed.duplicateSelected({ behaviors: false });
  const bareId = ed.state.selection[0];
  const bare = ed.entityBehaviors(bareId).length;

  // naming the copy carries its behaviours over, or they would be orphaned the
  // moment you gave it the name that makes it findable
  ed.renamePlacement(copyId, "fanB");
  const afterName = { byId: ed.entityBehaviors(copyId).length, byName: ed.entityBehaviors("fanB").length };
  // and joining a name that already governs others hands it over to that entry
  ed.renamePlacement(bareId, "fanA");
  const joined = { onFanA: ed.entityBehaviors("fanA").length, strays: ed.entityBehaviors(bareId).length };

  // an id entry cannot be re-adopted, so deleting its element takes it with it
  ed.renamePlacement(copyId, "");
  const backOnId = ed.entityBehaviors(copyId).length;
  ed.removePlacement(copyId);
  const afterDelete = ed.entityBehaviors(copyId).length;
  // a NAMED entry survives its last element: it is authored, not derived
  ed.removePlacement(fan.id);
  const namedSurvives = ed.entityBehaviors("fanA").length;

  ed.clearAll(); ed.select([]);
  return { name: copy?.name, carried, sourceStill, written, exported, copyId,
    bare, afterName, joined, backOnId, afterDelete, namedSurvives };
});
check("a copy is nameless, so the manifest sees two nodes rather than one",
  dupBhv.name === "" && dupBhv.exported === dupBhv.copyId
    && !!dupBhv.written.fanA && !!dupBhv.written[dupBhv.copyId],
  `name=${JSON.stringify(dupBhv.name)}, exported as ${dupBhv.exported}, entities [${Object.keys(dupBhv.written)}]`);
check("and it brings the source's behaviours, parameters and all",
  dupBhv.carried.length === 1 && dupBhv.carried[0].name === "playAnimation"
    && dupBhv.carried[0].animation === "Fan_Idle" && dupBhv.carried[0].loop === false,
  JSON.stringify(dupBhv.carried));
check("as a copy, not a second handle on the source's",
  dupBhv.sourceStill === "Fan_Idle", `source now says ${JSON.stringify(dupBhv.sourceStill)}`);
check("Ctrl+Shift+D copies the element without them",
  dupBhv.bare === 0, `${dupBhv.bare} carried`);
check("naming a copy carries its behaviours onto the name",
  dupBhv.afterName.byId === 0 && dupBhv.afterName.byName === 1,
  JSON.stringify(dupBhv.afterName));
check("but joining a name that already governs others defers to that entry",
  dupBhv.joined.onFanA === 1 && dupBhv.joined.strays === 0,
  JSON.stringify(dupBhv.joined));
check("an id entry dies with its element, where a named one outlives its last",
  dupBhv.backOnId === 1 && dupBhv.afterDelete === 0 && dupBhv.namedSurvives === 1,
  `back on id ${dupBhv.backOnId}, after delete ${dupBhv.afterDelete}, named ${dupBhv.namedSurvives}`);

// A scale runs along the world's axes when the tools are in world space. It
// used to be local whatever the setting said, so "grow this along world X" grew
// a wall turned 90 degrees along world Z instead - the one gesture of the three
// that ignored Y.
const worldScale = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const saved = { space: ed.state.axisSpace, axis: ed.state.scaleAxis,
                  scale: ed.state.snap.scale, rot: ed.state.rotAxis, step: ed.state.snap.rot };
  ed.state.snap.scale = 0.1;
  ed.state.scaleAxis = "x";
  ed.state.rotAxis = "y";
  ed.state.snap.rot = 90;

  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
    new V(0, 0, 0), { silent: true });
  ed.select([a.id]);
  // a quarter turn about Y puts the wall's own Z along world X
  ed.state.axisSpace = "world";
  i.rotateCurrent(1);

  i.scaleCurrent(1);
  const world = a.node.scaling.asArray().map((v) => +v.toFixed(3));

  a.node.scaling.set(1, 1, 1);
  ed.state.axisSpace = "local";
  i.scaleCurrent(1);
  const local = a.node.scaling.asArray().map((v) => +v.toFixed(3));

  // turned to an odd angle there is no axis to act on: stretching it along a
  // world axis would shear it, and no position/rotation/scale triple says that
  a.node.scaling.set(1, 1, 1);
  ed.state.axisSpace = "world";
  ed.state.snap.rot = 45;
  i.rotateCurrent(1);
  i.scaleCurrent(1);
  const offAxis = a.node.scaling.asArray().map((v) => +v.toFixed(3));
  const said = document.getElementById("status-text").textContent;

  // but a uniform scale is the same in every space, so it never declines
  ed.state.scaleAxis = "all";
  i.scaleCurrent(1);
  const uniform = a.node.scaling.asArray().map((v) => +v.toFixed(3));

  ed.state.axisSpace = saved.space; ed.state.scaleAxis = saved.axis;
  ed.state.snap.scale = saved.scale; ed.state.rotAxis = saved.rot;
  ed.state.snap.rot = saved.step;
  ed.clearAll(); ed.select([]);
  return { world, local, offAxis, said, uniform };
});
check("a world-space scale grows the element along the world's axis",
  worldScale.world[0] === 1 && worldScale.world[1] === 1 && worldScale.world[2] === 1.1,
  `world X on a quarter-turned wall gave ${JSON.stringify(worldScale.world)}`);
check("a local-space scale still grows it along its own",
  worldScale.local[0] === 1.1 && worldScale.local[1] === 1 && worldScale.local[2] === 1,
  `local X gave ${JSON.stringify(worldScale.local)}`);
check("an off-axis element declines a world-space scale",
  JSON.stringify(worldScale.offAxis) === JSON.stringify([1, 1, 1])
    && /shear/.test(worldScale.said) && /\bY\b/.test(worldScale.said),
  `${JSON.stringify(worldScale.offAxis)}, said "${worldScale.said}"`);
check("a uniform scale needs no axis and acts anyway",
  JSON.stringify(worldScale.uniform) === JSON.stringify([1.1, 1.1, 1.1]),
  `${JSON.stringify(worldScale.uniform)}`);

// A flip is a scale by -1, so it asked the same question and gave the same wrong
// answer: Alt+F mirrored about the element's *own* plane whatever Y said. Unlike
// a stretch, a mirror is exact at any angle - the rotation absorbs the
// difference - so this one never has to decline.
const worldFlip = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const saved = { space: ed.state.axisSpace, axis: ed.state.scaleAxis,
                  rot: ed.state.rotAxis, step: ed.state.snap.rot };
  const WALL = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  ed.state.scaleAxis = "x";
  ed.state.rotAxis = "y";

  // an element's own axes as the world sees them
  const rows = (n) => {
    const m = n.computeWorldMatrix(true);
    return [0, 1, 2].map((r) => {
      const v = m.getRow(r);
      return new V(v.x, v.y, v.z).normalize().asArray().map((c) => +c.toFixed(4));
    });
  };

  // ---- turned to an odd angle: no axis lines up, and it mirrors anyway
  const a = await ed.placeAt(WALL, new V(0, 0, 0), { silent: true });
  ed.select([a.id]);
  ed.state.axisSpace = "world";
  ed.state.snap.rot = 45;
  i.rotateCurrent(1);
  const parented = !!a.node.parent;
  const before = rows(a.node);
  i.flipCurrent();
  const after = rows(a.node);
  // mirrored about the world YZ plane: every direction's X negates, Y and Z hold
  const mirrored = before.every((row, r) =>
    Math.abs(after[r][0] + row[0]) < 1e-3
    && Math.abs(after[r][1] - row[1]) < 1e-3
    && Math.abs(after[r][2] - row[2]) < 1e-3);

  // ---- on the grid: the wall's own Z is what lies along world X
  const b = await ed.placeAt(WALL, new V(8, 0, 0), { silent: true });
  ed.select([b.id]);
  ed.state.snap.rot = 90;
  i.rotateCurrent(1);
  const quatBefore = b.node.rotationQuaternion.asArray();
  i.flipCurrent();
  const gridScale = b.node.scaling.asArray();
  // and it keeps its exact right angles - the rotation is not rebuilt at all
  const untouched = quatBefore.every((v, n) => v === b.node.rotationQuaternion.asArray()[n]);

  // ---- local space still mirrors about the element's own plane
  const c = await ed.placeAt(WALL, new V(16, 0, 0), { silent: true });
  ed.select([c.id]);
  i.rotateCurrent(1);
  ed.state.axisSpace = "local";
  i.flipCurrent();
  const localScale = c.node.scaling.asArray();

  ed.state.axisSpace = saved.space; ed.state.scaleAxis = saved.axis;
  ed.state.rotAxis = saved.rot; ed.state.snap.rot = saved.step;
  ed.clearAll(); ed.select([]);
  return { parented, mirrored, before, after, gridScale, untouched, localScale };
});
check("a placement root is top-level, which the mirror rebuild relies on",
  worldFlip.parented === false, `parent=${worldFlip.parented}`);
check("a world-space flip mirrors an off-axis element about the world plane",
  worldFlip.mirrored,
  `${JSON.stringify(worldFlip.before)} -> ${JSON.stringify(worldFlip.after)}`);
check("on the grid it flips the axis that lies along the world's",
  JSON.stringify(worldFlip.gridScale) === JSON.stringify([1, 1, -1]),
  `world X on a quarter-turned wall gave ${JSON.stringify(worldFlip.gridScale)}`);
check("and leaves an aligned element's rotation untouched to the bit",
  worldFlip.untouched, `quaternion was rebuilt`);
check("a local-space flip still mirrors about the element's own plane",
  JSON.stringify(worldFlip.localScale) === JSON.stringify([-1, 1, 1]),
  `local X gave ${JSON.stringify(worldFlip.localScale)}`);

// Doors were excluded from both scale paths: scaleCurrent() filtered markers
// out, and applyInspector() guarded the scale write with `if (!e.type)`.
const doorScale = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const i = await import("/js/interact.js");
  const mf = await import("/js/manifest.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
  const a = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
  const one = (await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true })).id;
  ed.select([one]);
  ed.hideSelected();
  ed.clearAll();
  const fresh = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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
  const a = (await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true })).id;
  const b = (await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(4, 0, 0), { silent: true })).id;
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
  const a = (await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true })).id;
  ed.select([a]);
  ed.hideSelected(); ed.hideSelected();       // ghost, then hidden
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(4, 0, 0));   // not silent: pushes undo
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
  await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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

// ---- a door marker drops through whatever is in the way ---------------------
// A doorway is a wall module with a hole in it, so the ray hits that wall
// first. Dropping the marker only where the pick found bare ground therefore
// refused it at exactly the spot it is wanted, which made the Door button
// nearly useless. The click now goes straight to the build plane under the
// cursor whatever is in front of it - nothing is lost, because the marker takes
// the grid elevation rather than the height of what it was clicked through.
const doorDropSetup = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const wall = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
    new V(0, 0, 0), { silent: true });
  ed.select([]);
  // above and in front, so a ray through the wall carries on down to the grid
  ed.state.camera.position = new V(0, 6, -8);
  ed.state.camera.setTarget(new V(0, 0, 0));
  ed.state.camera.cameraDirection.setAll(0);
  return { wall: wall.id, doors: [...ed.state.markers.values()].filter((m) => m.type === "door").length };
});
await page.waitForTimeout(700);
const wallAim = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const scene = ed.state.scene;
  const b = ed.screenBoundsOf(ed.state.placements.get(ids.wall).node);
  const r = scene.getEngine().getRenderingCanvas().getBoundingClientRect();
  const at = (fx, fy) => ({
    x: b.minX + (b.maxX - b.minX) * fx,
    y: b.minY + (b.maxY - b.minY) * fy,
  });
  // Hunt for a point that really is on the panel rather than trusting its
  // screen centre: a wall module can have an opening through the middle, and a
  // marker aimed at fresh air would be testing the old behaviour by accident.
  for (const fy of [0.5, 0.3, 0.7, 0.15, 0.85]) {
    for (const fx of [0.5, 0.2, 0.8, 0.35, 0.65]) {
      const p = at(fx, fy);
      const pick = scene.pick(p.x, p.y, (m) => m.isPickable && m.isEnabled());
      if (pick?.hit && ed.ownerIdOf(pick.pickedMesh) === ids.wall) {
        return { x: r.x + p.x, y: r.y + p.y, found: true };
      }
    }
  }
  const c = at(0.5, 0.5);
  return { x: r.x + c.x, y: r.y + c.y, found: false };
}, doorDropSetup);

await page.mouse.move(wallAim.x, wallAim.y, { steps: 5 });
await page.waitForTimeout(250);
const overWall = await page.evaluate(async () =>
  (await import("/js/interact.js")).hoveredId());
check("the spot a door is wanted at is squarely on a wall",
  wallAim.found && overWall === doorDropSetup.wall,
  `found=${wallAim.found}, hovered=${overWall}`);

await page.click("#btn-door");
await page.mouse.move(wallAim.x + 5, wallAim.y + 5, { steps: 3 });
await page.mouse.move(wallAim.x, wallAim.y, { steps: 3 });
await page.waitForTimeout(250);
const armedHover = await page.evaluate(async () => ({
  hovered: (await import("/js/interact.js")).hoveredId(),
  armed: (await import("/js/editor.js")).state.markerBrush,
}));
check("arming the marker stops the hover promising a selection it will not make",
  armedHover.armed === "door" && armedHover.hovered === null,
  JSON.stringify(armedHover));

await page.mouse.click(wallAim.x, wallAim.y);
await page.waitForTimeout(450);
const doorDropped = await page.evaluate(async (ids) => {
  const ed = await import("/js/editor.js");
  const doors = [...ed.state.markers.values()].filter((m) => m.type === "door");
  const d = doors[doors.length - 1];
  return {
    doors: doors.length,
    selection: [...ed.state.selection],
    onDoor: !!d && ed.state.selection.length === 1 && ed.state.selection[0] === d.id,
    y: d ? Number(d.node.position.y.toFixed(3)) : null,
    gridY: ed.state.gridY,
    armed: ed.state.markerBrush,
    wallStands: ed.state.placements.has(ids.wall),
  };
}, doorDropSetup);
check("clicking on the wall drops the marker instead of selecting the wall",
  doorDropped.doors === doorDropSetup.doors + 1 && doorDropped.onDoor
    && doorDropped.wallStands,
  JSON.stringify(doorDropped));
check("and it lands on the build plane, not on the wall it was clicked through",
  Math.abs(doorDropped.y - doorDropped.gridY) < 1e-3 && doorDropped.armed === null,
  `y=${doorDropped.y}, gridY=${doorDropped.gridY}, armed=${doorDropped.armed}`);

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
  const pane = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
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

// A rectangle is tested against each element's *outline*, not the rectangle
// its outline sits in. A screen box round a slab lying diagonally across the
// view is mostly empty air, so a small rectangle dropped in a gap used to
// select everything around it - one drawn in clear air beside a corner hull
// picked up all of its boxes.
const marquee = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  await new Promise((r) => setTimeout(r, 300));

  // three long boxes, turned, laid out like the boxes of a corner hull: this
  // is the shape the old test was worst on, because a turned slab's screen box
  // is mostly empty air
  const made = [
    co.addCollider("box", new V(0, 1, 0),
      { rotation: [0, 35, 0], scale: [4, 2, 0.4], silent: true }),
    co.addCollider("box", new V(2.6, 1, 1.8),
      { rotation: [0, -55, 0], scale: [4, 2, 0.4], silent: true }),
    co.addCollider("box", new V(-2.2, 1, 2.2),
      { rotation: [0, 20, 0], scale: [3, 2, 0.4], silent: true }),
  ];
  ed.state.camera.position = new V(5, 6, -7);
  ed.state.camera.setTarget(new V(0, 1, 0.7));
  ed.state.scene.render();
  await new Promise((r) => setTimeout(r, 300));

  const total = ed.state.colliders.size;

  // a patch of canvas with nothing drawn on it
  const eng = ed.state.engine;
  ed.state.scene.render();
  const w = eng.getRenderWidth(), h = eng.getRenderHeight();
  const buf = await eng.readPixels(0, 0, w, h);
  const green = (x, yTop) => {
    const i = (((h - 1 - yTop) * w) + x) * 4;
    return buf[i + 1] > 70 && buf[i + 1] > buf[i] + 25;
  };
  let clear = null;
  for (let y = 40; y < h - 140 && !clear; y += 20) {
    for (let x = 40; x < w - 140 && !clear; x += 20) {
      let ok = true;
      for (let dy = 0; dy < 90 && ok; dy += 10) {
        for (let dx = 0; dx < 90 && ok; dx += 10) if (green(x + dx, y + dy)) ok = false;
      }
      if (ok) clear = { x, y };
    }
  }

  // and how much tighter the outline is than the box it replaces
  const areaOf = (poly) => {
    let a = 0;
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i], q = poly[(i + 1) % poly.length];
      a += p[0] * q[1] - q[0] * p[1];
    }
    return Math.abs(a) / 2;
  };
  let hullArea = 0, boxArea = 0;
  for (const s of made) {
    const hull = ed.screenHullOf(s.node), b = ed.screenBoundsOf(s.node);
    if (!hull || !b || hull.length < 3) continue;
    hullArea += areaOf(hull);
    boxArea += (b.maxX - b.minX) * (b.maxY - b.minY);
  }

  const out = { total, clear,
    inAir: clear ? ed.elementsInRect(
      { x0: clear.x, y0: clear.y, x1: clear.x + 80, y1: clear.y + 70 }).length : null,
    everything: ed.elementsInRect({ x0: -1e5, y0: -1e5, x1: 1e5, y1: 1e5 }).length,
    tightness: boxArea ? +(hullArea / boxArea).toFixed(3) : null };
  for (const s of made) co.removeCollider(s.id, true);
  return out;
});
check("a rectangle drawn in clear air selects nothing",
  marquee.inAir === 0,
  `caught ${marquee.inAir} of ${marquee.total}, rect at ${JSON.stringify(marquee.clear)}`);
check("a rectangle over everything still catches everything",
  marquee.everything === marquee.total,
  `caught ${marquee.everything} of ${marquee.total}`);
check("the outline is far tighter than the screen box it replaces",
  marquee.tightness !== null && marquee.tightness < 0.75,
  `${(100 * marquee.tightness).toFixed(0)}% of the old area`);
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
});
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
  const id = await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_Simple", new BABYLON.Vector3(0, 0, 0));
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
  await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_3Plates", new BABYLON.Vector3(0, 0, 0));
  ed.state.activeChunk = "CH_ISO_B";
  await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_3Plates", new BABYLON.Vector3(8, 0, 0));
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

// A door is not *in* a chunk, but it joins two, so isolation does have
// something to say about it. Markers used to be exempt outright, which left a
// door hanging in a room it has nothing to do with.
const doorIso = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const W = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  ed.addChunk("CH_D_A"); ed.addChunk("CH_D_B"); ed.addChunk("CH_D_C");
  ed.state.activeChunk = "CH_D_A";
  await ed.placeAt(W, new V(0, 0, 0), { silent: true });
  ed.state.activeChunk = "CH_D_B";
  await ed.placeAt(W, new V(8, 0, 0), { silent: true });
  ed.state.activeChunk = "CH_D_C";
  await ed.placeAt(W, new V(60, 0, 60), { silent: true });

  const named = mk.addDoor(new V(4, 0, 0), { chunkA: "CH_D_A", chunkB: "CH_D_B", silent: true });
  const auto = mk.addDoor(new V(58, 0, 58), { silent: true });      // sides left on (auto)

  const shownIn = (chunk) => {
    ed.state.activeChunk = chunk;
    ed.state.isolate = true;
    ed.applyVisibility();
    return [...ed.state.markers.values()].filter((m) => m.node.isEnabled()).map((m) => m.id);
  };
  const out = {
    inA: shownIn("CH_D_A"), inB: shownIn("CH_D_B"), inC: shownIn("CH_D_C"),
    named: named.id, auto: auto.id,
  };
  ed.state.isolate = false;
  ed.applyVisibility();
  out.off = [...ed.state.markers.values()].filter((m) => m.node.isEnabled()).map((m) => m.id);
  ed.clearAll(); ed.select([]);
  return out;
});
check("an isolated chunk shows the doors that reach it",
  doorIso.inA.includes(doorIso.named) && doorIso.inB.includes(doorIso.named),
  `A ${JSON.stringify(doorIso.inA)}, B ${JSON.stringify(doorIso.inB)}`);
check("and hides the ones that do not",
  !doorIso.inC.includes(doorIso.named), `C ${JSON.stringify(doorIso.inC)}`);
check("a door left on (auto) is placed by the same rule the manifest uses",
  doorIso.inC.includes(doorIso.auto) && !doorIso.inA.includes(doorIso.auto),
  `auto door in C=${doorIso.inC.includes(doorIso.auto)}, in A=${doorIso.inA.includes(doorIso.auto)}`);
check("every door is back once isolation is off",
  doorIso.off.includes(doorIso.named) && doorIso.off.includes(doorIso.auto),
  JSON.stringify(doorIso.off));
await page.evaluate(async () => (await import("/js/editor.js")).clearAll());

// ---- 1w. a door onto the skybox ------------------------------------------
//
// A window through the hull is still a portal - the renderer needs the opening
// and its shape - but there is no room behind it. It is expressed as a reserved
// chunk id rather than another boolean, so everything that already reasons
// about a door's two sides keeps working; the price is that the id must be
// unusable as a real chunk, and that "opens onto space" and "not sealed" must
// not both be true at once.
const sky = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mk = await import("/js/markers.js");
  const man = await import("/js/manifest.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const W = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  ed.addChunk("CH_SKY_A");
  ed.state.activeChunk = "CH_SKY_A";
  await ed.placeAt(W, new V(0, 0, 0), { silent: true });

  const out = { id: ed.SKYBOX_CHUNK };
  // The reserved id cannot be taken by a room, whichever way it is reached.
  out.addRefused = ed.addChunk(ed.SKYBOX_CHUNK) === false
    && !ed.state.chunks.includes(ed.SKYBOX_CHUNK);
  out.renameRefused = ed.renameChunk("CH_SKY_A", ed.SKYBOX_CHUNK) === false
    && ed.state.chunks.includes("CH_SKY_A");

  // Set through the inspector, the way a user would.
  const door = mk.addDoor(new V(4, 0, 0), { chunkA: "CH_SKY_A", silent: true });
  ed.select([door.id]);
  const sel = document.getElementById("door-b");
  out.offered = [...sel.options].map((o) => o.value);
  out.offeredOnA = [...document.getElementById("door-a").options].map((o) => o.value);
  sel.value = ed.SKYBOX_CHUNK;
  sel.dispatchEvent(new Event("change", { bubbles: true }));

  const box = document.getElementById("door-sealed");
  out.afterPick = { chunkB: door.chunkB, sealed: door.sealed,
                    checked: box.checked, disabled: box.disabled };

  // The manifest is the contract: the side is written through, and the portal
  // edge is one-way because space is not a room with an aabb to come back from.
  const m = man.buildManifest();
  const d = m.doors.find((x) => x.id === door.id);
  const p = m.portals.find((x) => x.door === door.id);
  out.written = { chunkB: d?.chunkB, sealed: d?.sealed, portalB: p?.chunkB };
  out.chunkList = m.chunks.map((c) => c.id);
  out.adjacency = m.adjacency;

  // ...and it survives a round trip through serialize/deserialize.
  const saved = mk.serializeMarkers();
  mk.deserializeMarkers(saved);
  const back = [...ed.state.markers.values()][0];
  out.roundTrip = { chunkB: back?.chunkB, sealed: back?.sealed };

  // An unsealed window onto space is not a state the ship can be in, even if a
  // hand-edited file says so.
  const forced = mk.addDoor(new V(9, 0, 0),
    { chunkA: "CH_SKY_A", chunkB: ed.SKYBOX_CHUNK, sealed: false, silent: true });
  out.forcedOnLoad = forced.sealed;

  ed.clearAll(); ed.select([]);
  return out;
});
check("the skybox id cannot be taken by a real chunk",
  sky.addRefused && sky.renameRefused,
  `add refused=${sky.addRefused}, rename refused=${sky.renameRefused}`);
check("Chunk B offers Skybox and Chunk A does not",
  sky.offered.includes(sky.id) && !sky.offeredOnA.includes(sky.id),
  `B ${JSON.stringify(sky.offered)}, A ${JSON.stringify(sky.offeredOnA)}`);
check("picking it seals the door and takes the choice away",
  sky.afterPick.chunkB === sky.id && sky.afterPick.sealed === true
    && sky.afterPick.checked === true && sky.afterPick.disabled === true,
  JSON.stringify(sky.afterPick));
check("the manifest writes the side through, sealed",
  sky.written.chunkB === sky.id && sky.written.sealed === true
    && sky.written.portalB === sky.id,
  JSON.stringify(sky.written));
check("space is not listed as a chunk of the ship",
  !sky.chunkList.includes(sky.id), JSON.stringify(sky.chunkList));
check("the portal leads out to space, and nothing leads back",
  sky.adjacency["CH_SKY_A"]?.some((e) => e.to === sky.id)
    && !(sky.id in sky.adjacency),
  JSON.stringify(sky.adjacency));
check("the side survives a save and a load",
  sky.roundTrip.chunkB === sky.id && sky.roundTrip.sealed === true,
  JSON.stringify(sky.roundTrip));
check("and a door onto space is sealed even if the file says otherwise",
  sky.forcedOnLoad === true, `sealed=${sky.forcedOnLoad}`);
await page.evaluate(async () => (await import("/js/editor.js")).clearAll());

// ---- 1w-bis. B brings what is in hand to the camera ------------------------
//
// Arming a module does not choose a position: the ghost lands wherever the
// cursor ray happens to cross the build plane. From inside a finished room,
// with the plane still down on the deck you started from, that is routinely
// behind you - `cursorOnPlane` returns null for anything at or above the eye
// and the ghost simply stays where it last was - or hundreds of metres off.
// Neither can be dragged back into view, because it is not in view to drag.
//
// So B fetches it, and takes the **build plane** with it. That second half is
// the one that matters: a one-shot teleport would be undone by the very next
// mouse move, which re-derives the ghost's position from the plane.
const WALL = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
const bringGhost = await page.evaluate(async (WALL) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  // One deck, scaled up, a long way from the origin - so "wherever the ghost
  // already was" is unmistakably the wrong answer.
  await ed.placeAt("Modular SciFi MegaKit/Platforms/Platform_Simple", new V(96, 0, 96),
    { scale: [8, 1, 8], silent: true });
  ed.state.scene.pointerX = ed.state.engine.getRenderWidth() / 2;
  ed.state.scene.pointerY = ed.state.engine.getRenderHeight() / 2;

  // the broken case exactly: standing on the deck, the build plane left up on
  // another one, looking very slightly down
  ed.setGridElevation(20);
  ed.state.camera.position.set(100, 1.6, 100);
  ed.state.camera.setTarget(new V(104, -1.2, 104));
  ed.state.scene.render();

  await i.armGhost(WALL);
  const g = ed.state.scene.getTransformNodeByName("GHOST");
  const mid = () => { const b = ed.worldBounds(g); return b.min.add(b.max).scale(0.5); };
  const away = V.Distance(mid(), ed.state.camera.position);
  const reachedBefore = !!ed.cursorOnPlane(ed.state.gridY);

  const r = i.bringToCamera();
  ed.state.scene.render();
  const after = ed.cursorOnPlane(ed.state.gridY);
  const step = ed.state.snap.pos || 0;
  const onGrid = (v) => !step || Math.abs(v / step - Math.round(v / step)) < 1e-6;
  return {
    r, step, reachedBefore,
    away: +away.toFixed(1),
    near: +V.Distance(mid(), ed.state.camera.position).toFixed(2),
    gridY: +ed.state.gridY.toFixed(3),
    deck: ed.groundHeightAt(g.position.x, g.position.z, 10, 30),
    onGrid: onGrid(g.position.x) && onGrid(g.position.z),
    origin: g.position.asArray().map((v) => +v.toFixed(3)),
    reachable: after ? +V.Distance(after, ed.state.camera.position).toFixed(1) : null,
  };
}, WALL);
check("before B the ghost is nowhere near you, and the plane is out of reach",
  bringGhost.away > 50 && bringGhost.reachedBefore === false,
  `${bringGhost.away} m away, plane reachable: ${bringGhost.reachedBefore}`);
check("B brings it to arm's length",
  bringGhost.r?.kind === "ghost" && bringGhost.near < 10,
  `${bringGhost.near} m away, ${JSON.stringify(bringGhost.r)}`);
check("it lands on the deck under your feet, not at eye height",
  bringGhost.r?.floor === true && bringGhost.deck != null
    && Math.abs(bringGhost.r.y - bringGhost.deck) < 0.01,
  `dropped at y ${bringGhost.r?.y}, deck at ${bringGhost.deck}`);
check("still aligned on the grid",
  bringGhost.onGrid, `step ${bringGhost.step}, origin [${bringGhost.origin}]`);
// the crux: without this the next mouse move sends it straight back
check("and the build plane comes with it, so it stays reachable",
  Math.abs(bringGhost.gridY - bringGhost.r.y) < 1e-6 && bringGhost.reachable < 15,
  `plane 20 -> ${bringGhost.gridY}, cursor meets it ${bringGhost.reachable} m away`);

// Facing something close, it stops short rather than landing through it: the
// preferred distance would put the piece in the next room every time you built
// against a wall.
const bringClose = await page.evaluate(async (WALL) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost();
  ed.state.camera.position.set(100, 1.0, 100);
  ed.state.camera.setTarget(new V(100.02, 0, 100.02));   // all but straight down
  ed.state.scene.render();
  await i.armGhost(WALL);
  const r = i.bringToCamera();
  const g = ed.state.scene.getTransformNodeByName("GHOST");
  const b = ed.worldBounds(g), mid = b.min.add(b.max).scale(0.5);
  const cam = ed.state.camera.position;
  i.cancelGhost();
  return { r, flat: +Math.hypot(mid.x - cam.x, mid.z - cam.z).toFixed(2) };
}, WALL);
check("what you are looking at stops it, so it never lands through a wall",
  bringClose.flat < 2.5, `${bringClose.flat} m out, the free-air distance is ~4 m`);

// A placed selection moves too - same spot, one undo entry, shape intact.
const bringSel = await page.evaluate(async (WALL) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost();
  const was = ed.state.placements.size;
  const a = await ed.placeAt(WALL, new V(0, 0, 0), { silent: true });
  const b = await ed.placeAt(WALL, new V(0, 0, 6), { silent: true });
  ed.select([a.id, b.id]);
  ed.setGridElevation(20);
  ed.state.camera.position.set(100, 1.6, 100);
  ed.state.camera.setTarget(new V(104, -1.2, 104));
  ed.state.scene.render();

  const r = i.bringToCamera();
  const step = ed.state.snap.pos || 0;
  const cam = ed.state.camera.position;
  const bb = ed.worldBounds(a.node);
  const out = {
    r, added: ed.state.placements.size - was,
    dist: +Math.hypot(a.node.position.x - cam.x, a.node.position.z - cam.z).toFixed(2),
    dz: +(b.node.position.z - a.node.position.z).toFixed(3),
    bottom: +bb.min.y.toFixed(2),
    gridY: +ed.state.gridY.toFixed(3),
    onGrid: !step || Math.abs(a.node.position.x / step - Math.round(a.node.position.x / step)) < 1e-6,
  };
  // undo rebuilds the placements, so the node has to be looked up again
  await ed.undo();
  const back = ed.state.placements.get(a.id).node.position;
  out.undone = [+back.x.toFixed(2), +back.z.toFixed(2)];
  return out;
}, WALL);
check("a placed selection is fetched the same way, and moved rather than copied",
  bringSel.r?.kind === "selection" && bringSel.r.count === 2 && bringSel.added === 2
    && bringSel.dist < 10,
  `${bringSel.dist} m away, ${bringSel.added} new placements, ${JSON.stringify(bringSel.r)}`);
check("the set keeps its spacing and its grid, and rests on the deck",
  bringSel.dz === 6 && bringSel.onGrid && Math.abs(bringSel.bottom - bringSel.r.y) < 0.05
    && Math.abs(bringSel.gridY - bringSel.r.y) < 1e-6,
  `dz ${bringSel.dz}, on grid ${bringSel.onGrid}, bottom ${bringSel.bottom} vs y ${bringSel.r.y}`);
check("and it is one undo away",
  bringSel.undone.join() === "0,0", `back at [${bringSel.undone}]`);

// Out in the open there is no deck to find, and it still has to land somewhere
// you can see rather than refusing.
const bringSpace = await page.evaluate(async (WALL) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  ed.state.camera.position.set(500, 30, 500);
  ed.state.camera.setTarget(new V(508, 30, 508));
  ed.state.scene.render();
  await i.armGhost(WALL);
  const r = i.bringToCamera();
  const g = ed.state.scene.getTransformNodeByName("GHOST");
  const b = ed.worldBounds(g), mid = b.min.add(b.max).scale(0.5);
  i.cancelGhost();
  return { r, near: +V.Distance(mid, ed.state.camera.position).toFixed(2) };
}, WALL);
check("with no floor under it, it lands in front of you at your own height",
  bringSpace.r?.floor === false && Math.abs(bringSpace.r.y - 30) < 0.01
    && bringSpace.near < 10,
  `y ${bringSpace.r?.y}, ${bringSpace.near} m out`);

const bringNone = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost(); ed.select([]);
  return i.bringToCamera();
});
check("with nothing in hand it does nothing at all",
  bringNone === null, `returned ${JSON.stringify(bringNone)}`);
await page.evaluate(async () => (await import("/js/editor.js")).clearAll());

// ---- 1x. an outline is the same thickness however close the camera is ------
//
// `edgesWidth` is not a pixel width: the line shader offsets the vertex in
// clip space, *before* the perspective divide, so what you see is
// `edgesWidth * renderHeight / (100 * viewDepth)` - it doubles every time you
// halve your distance. A fixed width read as a fine line across a room and as
// a slab of colour with the camera against a crate, hiding the thing it was
// drawn to point at.
const edgeScale = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const kit = await import("/js/kit.js");
  const id = [...kit.getCatalogue().byId.keys()].find((k) => k.includes("Crate1"))
    || [...kit.getCatalogue().byId.keys()][0];
  const e = await ed.placeAt(id, new BABYLON.Vector3(0, 0, 0), {});
  ed.select([e.id]);
  await new Promise((r) => setTimeout(r, 300));

  const mid = (() => {
    const c = e.node.getHierarchyBoundingVectors(true);
    return c.min.add(c.max).scale(0.5);
  })();
  const dir = new BABYLON.Vector3(0.6, 0.45, 0.66).normalize();

  /** Frame the crate from `d` metres, then measure the outline off the buffer. */
  const at = async (d) => {
    ed.state.camera.position = mid.add(dir.scale(d));
    ed.state.camera.setTarget(mid);
    ed.state.scene.render();
    await new Promise((r) => setTimeout(r, 200));
    ed.state.scene.render();
    const eng = ed.state.engine;
    const w = eng.getRenderWidth(), h = eng.getRenderHeight();
    const buf = await eng.readPixels(0, 0, w, h);
    const runs = [];
    for (let row = 0; row < h; row += 2) {
      let run = 0;
      for (let x = 0; x < w; x++) {
        const i = ((row * w) + x) * 4;
        // clearly blue: the select colour is (64, 191, 255)
        const blue = buf[i + 2] > 150 && buf[i + 2] - buf[i] > 80
          && buf[i + 1] > 90 && buf[i + 1] < buf[i + 2];
        if (blue) run++;
        else { if (run) runs.push(run); run = 0; }
      }
      if (run) runs.push(run);
    }
    runs.sort((a, b) => a - b);
    return { width: +e.node.getChildMeshes()[0].edgesWidth.toFixed(3),
      px: runs.length ? runs[Math.floor(runs.length / 2)] : 0, runs: runs.length };
  };
  const far = await at(12);
  const near = await at(2.5);
  ed.removePlacement(e.id);
  return { far, near };
});
check("the outline width tracks the distance, which is what keeps it constant",
  Math.abs(edgeScale.far.width / edgeScale.near.width - 12 / 2.5) < 0.3,
  `${edgeScale.far.width} at 12 m vs ${edgeScale.near.width} at 2.5 m`
  + ` — ratio ${(edgeScale.far.width / edgeScale.near.width).toFixed(2)}, wanted 4.80`);
check("so the drawn outline does not fatten as the camera closes in",
  edgeScale.near.px > 0 && edgeScale.far.px > 0
    && edgeScale.near.px <= edgeScale.far.px * 1.6 && edgeScale.near.px <= 8,
  `${edgeScale.far.px} px at 12 m -> ${edgeScale.near.px} px at 2.5 m,`
  + ` over a 4.8x change in distance`);

await page.evaluate(async () => (await import("/js/editor.js")).clearAll());

// ---- 1y. a module dropped on the collision bench is a one-shot -------------
//
// On the ship a palette tile stays armed, so a row of panels is just repeated
// clicks. The bench is the opposite: a module goes there once, to have a hull
// fitted to it, and staging the same one twice is refused anyway - so staying
// armed only ever left a second stand-in on the cursor to be dismissed.
const benchShot = await (async () => {
  await page.evaluate(async () => {
    const co = await import("/js/colliders.js");
    const kit = await import("/js/kit.js");
    if (!co.isCollisionMode?.() && (await import("/js/editor.js")).state.mode !== "collision") {
      await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
    }
  });
  await page.waitForTimeout(700);

  const read = () => page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const it = await import("/js/interact.js");
    return { brush: ed.state.brush, armed: it.ghostActive(),
      staged: [...ed.state.placements.values()].filter((p) => p.stage).length,
      colliders: ed.state.colliders.size,
      hint: document.getElementById("hint").textContent,
      lit: !!document.querySelector("#palette-list .item.active") };
  });
  const clickAt = async (dx, dy) => {
    const b = await page.locator("#viewport").boundingBox();
    await page.mouse.move(b.x + b.width / 2 + dx, b.y + b.height / 2 + dy);
    await page.waitForTimeout(200);
    await page.mouse.down();
    await page.mouse.up();
    await page.waitForTimeout(800);
  };

  await page.evaluate(async (id) => (await import("/js/palette.js")).setBrush(id), PROP_A);
  await page.waitForFunction(async () => (await import("/js/interact.js")).ghostActive(),
    null, { timeout: 20000 });
  await page.waitForTimeout(250);
  const armed = await read();
  await clickAt(0, 0);
  const one = await read();
  await clickAt(180, 0);
  const two = await read();

  // a collision primitive is a different matter: a hull really is a run of boxes
  await page.evaluate(async () => (await import("/js/interact.js")).armColliderGhost("box"));
  await page.waitForTimeout(400);
  const cBefore = await read();
  await clickAt(-220, 70);
  await clickAt(-280, 70);
  const cAfter = await read();
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
  return { armed, one, two, cBefore, cAfter };
})();
check("the hint says a bench click puts the module on the bench",
  /bench/.test(benchShot.armed.hint), JSON.stringify(benchShot.armed.hint));
check("one click on the bench stages one module and puts the ghost down",
  benchShot.one.staged === benchShot.armed.staged + 1
    && !benchShot.one.armed && benchShot.one.brush === null,
  `staged ${benchShot.armed.staged} -> ${benchShot.one.staged},`
  + ` armed ${benchShot.one.armed}, brush ${JSON.stringify(benchShot.one.brush)}`);
check("the tile goes out and the hint is cleared with it",
  !benchShot.one.lit && benchShot.one.hint === "",
  `lit ${benchShot.one.lit}, hint ${JSON.stringify(benchShot.one.hint)}`);
check("so a second click stages nothing",
  benchShot.two.staged === benchShot.one.staged,
  `${benchShot.one.staged} -> ${benchShot.two.staged}`);
check("a collision primitive still stays armed for a run of boxes",
  benchShot.cAfter.colliders === benchShot.cBefore.colliders + 2,
  `${benchShot.cBefore.colliders} -> ${benchShot.cAfter.colliders}`);

// Ctrl+D there copies shapes, never the module. The rule lives in
// grabSelection rather than only at the key, which used to filter a list it
// then did not pass on - so a module selected alongside a shape was copied
// anyway, and the bench got a second stand-in of a module that may only be on
// it once.
const benchCopy = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const it = await import("/js/interact.js");
  const mod = [...ed.state.placements.values()].find((p) => p.stage);
  const a = co.addCollider("box", new BABYLON.Vector3(2, 0, 2), { stage: true, silent: true });
  const b = co.addCollider("box", new BABYLON.Vector3(3, 0, 2), { stage: true, silent: true });

  const carried = async (ids, opts) => {
    ed.select(ids);
    const g = await it.grabSelection(opts);
    const n = it.ghostCount();
    const mode = it.ghostMode();
    it.cancelGhost();
    return { got: !!g, n, mode };
  };
  const mixed = await carried([mod.id, a.id, b.id], { copy: true });
  const alone = await carried([mod.id], { copy: true });
  const move = await carried([mod.id], {});
  const shapes = await carried([a.id, b.id], { copy: true });
  co.removeCollider(a.id, true);
  co.removeCollider(b.id, true);
  ed.select([]);
  return { mixed, alone, move, shapes };
});
check("Ctrl+D on the bench leaves a module behind and takes only its shapes",
  benchCopy.mixed.n === 2, `carried ${benchCopy.mixed.n} of 3 selected (2 shapes + 1 module)`);
check("and a module on its own copies nothing at all",
  !benchCopy.alone.got && benchCopy.alone.n === 0, JSON.stringify(benchCopy.alone));
check("but M still picks a staged module up to move it",
  benchCopy.move.got && benchCopy.move.n === 1 && benchCopy.move.mode === "move",
  JSON.stringify(benchCopy.move));
check("and copying shapes alone is untouched",
  benchCopy.shapes.n === 2, JSON.stringify(benchCopy.shapes));

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  if (ed.state.mode === "collision") co.exitCollisionMode();
  ed.clearAll();
});
await page.waitForTimeout(500);

// ---- 1y. compound objects --------------------------------------------------
//
// The third editing mode. A compound is a *recipe*: saving one records its
// members and their lamps, and placing one expands that into ordinary
// placements sharing a `group` id. These follow the whole round trip - bench,
// save, palette, place, select, copy, break apart, delete - and check the two
// boundaries a bench must never cross: the ship file and the ship's undo stack.

const CNAME = "ZZ Test Compound";
const CID = `@compound/${CNAME}`;
const PLATE = "Modular SciFi MegaKit/Platforms/Platform_3Plates";

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const co = await import("/js/colliders.js");
  i.cancelGhost();
  if (ed.state.mode === "collision") co.exitCollisionMode();
  ed.clearAll(); ed.select([]);
  localStorage.removeItem("compoundBench");
});
await page.waitForTimeout(200);

// A ship element first, so "the bench never reaches the ship" has something to
// be true *about*: an empty serialize() would pass that check by accident.
const benchOpen = await page.evaluate(async ([M, P]) => {
  const ed = await import("/js/editor.js");
  const cp = await import("/js/compounds.js");
  const lg = await import("/js/lights.js");
  const V = BABYLON.Vector3;
  const ship = await ed.placeAt(M, new V(-20, 0, -20), { silent: true, noLights: true });

  await cp.enterCompoundMode();
  const a = await ed.placeAt(M, new V(-2, 0, 0), { silent: true, noLights: true });
  const b = await ed.placeAt(P, new V(4, 0, 0), { silent: true, noLights: true });
  const lamp = lg.addLight(a.id, { silent: true });
  return {
    mode: ed.state.mode,
    members: cp.benchMembers().length,
    staged: cp.benchMembers().every((p) => p.stage && p.chunk === "__compound_bench"),
    shipHidden: !ship.node.isEnabled(),
    benchShown: a.node.isEnabled() && b.node.isEnabled(),
    lamp: !!lamp,
    inShip: ed.serialize().instances.length,
    shipLights: ed.serialize().lights?.length ?? 0,
  };
}, [MODULE, PLATE]);
check("the compound bench is a third mode of its own",
  benchOpen.mode === "compound" && benchOpen.members === 2 && benchOpen.staged,
  `mode ${benchOpen.mode}, ${benchOpen.members} member(s) staged ${benchOpen.staged}`);
check("it hides the ship and shows only what is on it",
  benchOpen.shipHidden && benchOpen.benchShown,
  `ship hidden ${benchOpen.shipHidden}, bench shown ${benchOpen.benchShown}`);
check("a bench piece and its lamp are absent from the ship file",
  benchOpen.lamp && benchOpen.inShip === 1 && benchOpen.shipLights === 0,
  `${benchOpen.inShip} instance(s), ${benchOpen.shipLights} light(s) in serialize()`);

// The bench keeps its own undo stack. A ship snapshot taken here would restore
// as "no bench at all", which is exactly how Ctrl+Z used to wipe a staging area.
const benchStack = await page.evaluate(async ([M]) => {
  const ed = await import("/js/editor.js");
  const cp = await import("/js/compounds.js");
  const V = BABYLON.Vector3;
  const before = cp.benchMembers().length;
  await ed.placeAt(M, new V(0, 0, 6), {});     // not silent: pushes undo
  const after = cp.benchMembers().length;
  await ed.undo();
  return { before, after, undone: cp.benchMembers().length, ship: ed.state.placements.size };
}, [MODULE]);
check("the bench has its own undo stack",
  benchStack.after === benchStack.before + 1 && benchStack.undone === benchStack.before,
  `${benchStack.before} -> ${benchStack.after} -> ${benchStack.undone}`);

const cpSaved = await page.evaluate(async (name) => {
  const ed = await import("/js/editor.js");
  const cp = await import("/js/compounds.js");
  const kit = await import("/js/kit.js");
  // What the origin is *supposed* to be: the first member's own origin, the
  // piece the compound was started from. Read off the bench before it is saved
  // rather than written down as a number, so the check is the rule and not a
  // copy of the arithmetic.
  const before = cp.benchMembers().map((p) => ({ id: p.id, at: p.node.position.asArray() }));
  const origin = before[0].at;
  const r = await cp.saveCompound({ name, kit: "Modular SciFi MegaKit", category: "Compounds" });
  const tile = kit.getModule(`@compound/${name}`);
  const server = await (await fetch("/api/compounds")).json();
  return {
    r, origin,
    tile: tile ? { compound: !!tile.compound, url: tile.url ?? null, members: tile.members.length } : null,
    onDisk: (server.compounds || []).filter((c) => c.name === name).length,
    // Measured from the first member, so the recipe's coordinates say where the
    // rest sit relative to it rather than where it was built on the bench.
    offsets: (tile?.members || []).map((m) => m.position.map((v) => +v.toFixed(3))),
    want: before.map((m) => m.at.map((v, k) => +(v - origin[k]).toFixed(3))),
    lamps: (tile?.members || []).map((m) => (m.lights || []).length),
  };
}, CNAME);
check("saving files a tile with no model file of its own",
  cpSaved.r.ok && cpSaved.tile?.compound === true && cpSaved.tile.url === null
    && cpSaved.tile.members === 2 && cpSaved.onDisk === 1,
  `${JSON.stringify(cpSaved.r)}, tile ${JSON.stringify(cpSaved.tile)}`);
check("its members are measured from the first one, which sits at the origin",
  cpSaved.offsets.length === 2
    && JSON.stringify(cpSaved.offsets) === JSON.stringify(cpSaved.want)
    && JSON.stringify(cpSaved.offsets[0]) === JSON.stringify([0, 0, 0])
    // and that really is a rebase, not the world coordinates left alone
    && cpSaved.origin.some((v) => v !== 0),
  `offsets ${JSON.stringify(cpSaved.offsets)} about ${JSON.stringify(cpSaved.origin)},`
  + ` expected ${JSON.stringify(cpSaved.want)}`);
check("and a member's lamps travel in the recipe",
  cpSaved.lamps.reduce((a, b) => a + b, 0) === 1, `lamps per member ${JSON.stringify(cpSaved.lamps)}`);

const benchClosed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const cp = await import("/js/compounds.js");
  cp.exitCompoundMode();
  const kept = JSON.parse(localStorage.getItem("compoundBench") || "{}");
  return {
    mode: ed.state.mode,
    left: cp.benchMembers().length,
    ship: ed.state.placements.size,
    remembered: (kept.members || []).length,
  };
});
check("closing the bench puts the ship back and remembers what was on it",
  benchClosed.mode === "ship" && benchClosed.left === 0 && benchClosed.ship === 1
    && benchClosed.remembered === 2,
  `mode ${benchClosed.mode}, ${benchClosed.left} left, ship ${benchClosed.ship},`
  + ` ${benchClosed.remembered} remembered`);

const cpPlaced = await page.evaluate(async (id) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const lg = await import("/js/lights.js");
  await i.armGhost(id);
  const carried = i.ghostActive();
  await i.dropGhost();
  const mine = [...ed.state.placements.values()].filter((p) => p.compound);
  const groups = new Set(mine.map((p) => p.group));
  return {
    carried, n: mine.length, groups: groups.size,
    group: mine[0]?.group || "",
    lamps: mine.reduce((a, p) => a + lg.lightsOf(p.id).length, 0),
    selected: ed.state.selection.length,
    inShip: ed.serialize().instances.filter((x) => x.compound).length,
  };
}, CID);
check("placing a compound expands it into ordinary elements sharing one group",
  cpPlaced.carried && cpPlaced.n === 2 && cpPlaced.groups === 1 && /^G\d{4}$/.test(cpPlaced.group),
  `${cpPlaced.n} element(s) in ${cpPlaced.groups} group(s), id ${cpPlaced.group}`);
check("its lamps come with it, exactly once",
  cpPlaced.lamps === 1, `${cpPlaced.lamps} lamp(s)`);
check("the whole compound is selected on landing, and it is ship data",
  cpPlaced.selected === 2 && cpPlaced.inShip === 2,
  `${cpPlaced.selected} selected, ${cpPlaced.inShip} written`);

// Dropping keeps the brush armed, the same as any other tile, so that a run of
// them can be laid down in one go. Put it away before the clicking checks, or
// every click below would place another compound instead of selecting one.
const cpStillArmed = await page.evaluate(async () => {
  const i = await import("/js/interact.js");
  const armed = i.ghostActive();
  i.cancelGhost();
  return { armed, after: i.ghostActive() };
});
check("a compound brush stays armed for a run, and Esc puts it away",
  cpStillArmed.armed && !cpStillArmed.after,
  `armed ${cpStillArmed.armed} -> ${cpStillArmed.after}`);

// Clicking. Selection is the only place a compound is expanded - once every
// member is selected, every transform path the editor already has moves it as
// one - so this is the check that matters most.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.select([]);
  ed.focusSelection();
});
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mine = [...ed.state.placements.values()].filter((p) => p.compound);
  ed.focusNodes(mine.map((p) => p.node));
});
await page.waitForTimeout(400);
const memberAt = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const mine = [...ed.state.placements.values()].filter((p) => p.compound);
  return mine.map((p) => {
    const b = ed.worldBounds(p.node);
    return { id: p.id, at: b.min.add(b.max).scale(0.5).asArray() };
  });
});
const firstPt = await screenOf(memberAt[0].at);
await page.mouse.click(firstPt.x, firstPt.y);
await page.waitForTimeout(200);
const wholeClick = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { sel: [...ed.state.selection].sort() };
});
await page.keyboard.down("Control");
await page.keyboard.down("Alt");
await page.mouse.click(firstPt.x, firstPt.y);
await page.keyboard.up("Alt");
await page.keyboard.up("Control");
await page.waitForTimeout(200);
const drillClick = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { sel: [...ed.state.selection] };
});
check("clicking one member selects the whole compound",
  wholeClick.sel.length === 2, `selected ${JSON.stringify(wholeClick.sel)}`);
check("Ctrl+Alt+click drills in to the single member under the cursor",
  drillClick.sel.length === 1 && drillClick.sel[0] === memberAt[0].id,
  `selected ${JSON.stringify(drillClick.sel)}, aimed at ${memberAt[0].id}`);

const cpCopied = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const lg = await import("/js/lights.js");
  const mine = [...ed.state.placements.values()].filter((p) => p.compound);
  ed.select(mine.map((p) => p.id));
  await i.grabSelection({ copy: true });
  await i.dropGhost();
  const all = [...ed.state.placements.values()].filter((p) => p.compound);
  const groups = [...new Set(all.map((p) => p.group))];
  return {
    n: all.length, groups: groups.length,
    lamps: all.reduce((a, p) => a + lg.lightsOf(p.id).length, 0),
    named: all.every((p) => p.compound),
  };
});
check("copying a compound mints a new instance rather than a second handle",
  cpCopied.n === 4 && cpCopied.groups === 2 && cpCopied.named,
  `${cpCopied.n} element(s) in ${cpCopied.groups} group(s)`);
check("and the copy brings the lamps with it",
  cpCopied.lamps === 2, `${cpCopied.lamps} lamp(s) across both`);

const cpBroken = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const all = [...ed.state.placements.values()].filter((p) => p.group);
  const one = all[0].group;
  ed.select(ed.groupMembers(one).map((p) => p.id));
  const r = ed.breakApart();
  const left = [...ed.state.placements.values()].filter((p) => p.group).length;
  const still = ed.state.placements.size;
  await ed.undo();
  return {
    r, left, still,
    back: [...ed.state.placements.values()].filter((p) => p.group).length,
  };
});
check("break apart dissolves the group and leaves the pieces where they are",
  cpBroken.r.groups === 1 && cpBroken.r.members === 2 && cpBroken.left === 2 && cpBroken.still === 5,
  `${JSON.stringify(cpBroken.r)}, ${cpBroken.left} still grouped of ${cpBroken.still} elements`);
check("and it is undoable",
  cpBroken.back === 4, `${cpBroken.back} grouped again`);

const cpRound = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const layout = ed.serialize();
  const groups = layout.instances.filter((x) => x.group).length;
  await ed.deserialize(JSON.parse(JSON.stringify(layout)));
  const after = [...ed.state.placements.values()].filter((p) => p.group);
  const distinct = new Set(after.map((p) => p.group)).size;
  const next = ed.state.nextGroup;
  return { groups, after: after.length, distinct, next };
});
check("group and compound survive a save and load",
  cpRound.groups === 4 && cpRound.after === 4 && cpRound.distinct === 2
    && cpRound.next > 2,
  `${cpRound.groups} written, ${cpRound.after} read back in ${cpRound.distinct}`
  + ` group(s), counter at ${cpRound.next}`);

const cpDeleted = await page.evaluate(async (name) => {
  const cp = await import("/js/compounds.js");
  const ed = await import("/js/editor.js");
  const kit = await import("/js/kit.js");
  const r = await cp.deleteCompound(name);
  const server = await (await fetch("/api/compounds")).json();
  return {
    r,
    tile: !!kit.getModule(`@compound/${name}`),
    onDisk: (server.compounds || []).filter((c) => c.name === name).length,
    survivors: [...ed.state.placements.values()].filter((p) => p.compound === name).length,
  };
}, CNAME);
check("deleting a compound forgets the recipe, not the copies already placed",
  cpDeleted.r.ok && !cpDeleted.tile && cpDeleted.onDisk === 0 && cpDeleted.survivors === 4,
  `${JSON.stringify(cpDeleted.r)}, tile ${cpDeleted.tile}, ${cpDeleted.survivors} still in the ship`);

// ---- the follow-ups: rigid turns, whole-drop announcements, instance updates

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost();
  ed.clearAll(); ed.select([]);
  localStorage.removeItem("compoundBench");
});
await page.waitForTimeout(200);

// Built at whole-metre positions on purpose: the offsets a compound records are
// what decides whether a copy of it lands on the grid or half a metre off it.
// The bounding-box origin it used to record centred a 4 m wall against a 12 m
// platform at 5, so every member came out half a metre off the grid.
const cpBuilt = await page.evaluate(async ([M, P, name]) => {
  const ed = await import("/js/editor.js");
  const cp = await import("/js/compounds.js");
  const i = await import("/js/interact.js");
  const lg = await import("/js/lights.js");
  const V = BABYLON.Vector3;
  ed.state.rotAxis = "y";
  ed.state.axisSpace = "world";
  ed.state.snap.pos = 1;
  await cp.enterCompoundMode();
  const wall = await ed.placeAt(M, new V(-2, 0, 0), { silent: true, noLights: true });
  await ed.placeAt(P, new V(-2, 0, 4), { silent: true, noLights: true });
  lg.addLight(wall.id, { silent: true });
  const saved = await cp.saveCompound({
    name, kit: "Modular SciFi MegaKit", category: "Compounds",
  });
  const tile = (await import("/js/kit.js")).getModule(`@compound/${name}`);
  cp.exitCompoundMode();
  await i.armGhost(`@compound/${name}`);
  i.bringToCamera();                    // snapped exactly as a cursor drop is
  return {
    saved, armed: i.ghostActive(),
    offsets: (tile?.members || []).map((m) => m.position),
  };
}, [MODULE, PLATE, `${CNAME} rigid`]);

const cpRigid = await page.evaluate(async (name) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const lg = await import("/js/lights.js");
  await i.dropGhost();
  i.cancelGhost();
  const mine = () => [...ed.state.placements.values()].filter((p) => p.compound === name);
  const placed = mine();
  const at = (p) => p.node.position.asArray().map((v) => +v.toFixed(4));
  const before = placed.map(at);
  if (placed.length < 2) return { n: placed.length, where: before };
  ed.select(placed.map((p) => p.id));
  const anchorBefore = before[0];
  i.rotateCurrent(1);                    // one snap step about the current axis
  const after = mine().map(at);
  // The anchor stays put and the other member swings about it, at the same
  // distance - which is what "rigid" means and what per-element spinning broke.
  const span = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  return {
    n: placed.length,
    // Move is at 1 m, so the grid coordinates a drop snaps - X and Z - come out
    // whole unless the recipe itself carried a fraction.
    onGrid: Number.isInteger(before[0][0]) && Number.isInteger(before[0][2]),
    spaced: before.every((p, k) => [0, 1, 2].every((c) =>
      Number.isInteger(+(p[c] - before[0][c]).toFixed(4)))),
    anchorHeld: JSON.stringify(after[0]) === JSON.stringify(anchorBefore),
    moved: span(before[1], after[1]) > 0.5,
    keptRadius: Math.abs(span(before[0], before[1]) - span(after[0], after[1])) < 1e-3,
    lamps: mine().reduce((a, p) => a + lg.lightsOf(p.id).length, 0),
    where: before,
  };
}, `${CNAME} rigid`);
check("a compound records its members from the first one, at the origin",
  cpBuilt.saved.ok && cpBuilt.armed
    && JSON.stringify(cpBuilt.offsets[0]) === JSON.stringify([0, 0, 0])
    && cpBuilt.offsets.every((o) => o.every((v) => Number.isInteger(v))),
  `${JSON.stringify(cpBuilt.saved)}, offsets ${JSON.stringify(cpBuilt.offsets)}`);
check("so dropping one with Move at 1 m lands it on whole numbers",
  cpRigid.n === 2 && cpRigid.onGrid && cpRigid.spaced,
  `at ${JSON.stringify(cpRigid.where)}`);
check("turning a whole compound turns it rigidly about its first member",
  cpRigid.anchorHeld && cpRigid.moved && cpRigid.keptRadius,
  `anchor held ${cpRigid.anchorHeld}, other moved ${cpRigid.moved},`
  + ` distance kept ${cpRigid.keptRadius}`);

// The inspector used to write to state.selection[0] alone, which tore a
// compound apart one keystroke at a time.
const cpInspector = await page.evaluate(async (name) => {
  const ed = await import("/js/editor.js");
  const mine = [...ed.state.placements.values()].filter((p) => p.compound === name);
  if (mine.length < 2) return { n: mine.length };
  ed.select(mine.map((p) => p.id));
  await new Promise((r) => setTimeout(r, 80));
  const at = (p) => p.node.position.asArray().map((v) => +v.toFixed(3));
  const before = mine.map(at);
  const field = document.getElementById("pos-x");
  field.focus();
  field.value = String(+before[0][0] + 3);
  field.dispatchEvent(new Event("input", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 80));
  const after = mine.map(at);
  return {
    n: mine.length,
    anchorMoved: +(after[0][0] - before[0][0]).toFixed(3),
    otherMoved: +(after[1][0] - before[1][0]).toFixed(3),
    stillSelected: ed.state.selection.length,
  };
}, `${CNAME} rigid`);
check("the inspector moves a whole compound, not just the element it shows",
  cpInspector.anchorMoved === 3 && cpInspector.otherMoved === 3
    && cpInspector.stillSelected === 2,
  `anchor +${cpInspector.anchorMoved}, other +${cpInspector.otherMoved},`
  + ` ${cpInspector.stillSelected} still selected`);

// The Runtime view rebuilds its meshes on "placements" and its lamps on
// "lights". A drop that announced itself after the first member left the rest
// undressed and the compound's lamp doing nothing.
const cpAnnounced = await page.evaluate(async (name) => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const seen = { placements: 0, lights: 0, atPlacements: 0 };
  ed.on("placements", () => {
    seen.placements++;
    seen.atPlacements = [...ed.state.placements.values()].filter((p) => p.compound === name).length;
  });
  ed.on("lights", () => seen.lights++);
  const was = [...ed.state.placements.values()].filter((p) => p.compound === name).length;
  await i.armGhost(`@compound/${name}`);
  await i.dropGhost();
  i.cancelGhost();
  const now = [...ed.state.placements.values()].filter((p) => p.compound === name).length;
  return { ...seen, was, now };
}, `${CNAME} rigid`);
check("a compound drop is announced once, with all of it in place",
  cpAnnounced.placements === 1 && cpAnnounced.lights === 1
    && cpAnnounced.atPlacements === cpAnnounced.now && cpAnnounced.now === cpAnnounced.was + 2,
  `${cpAnnounced.placements} placement event(s), ${cpAnnounced.lights} light event(s),`
  + ` ${cpAnnounced.atPlacements} of ${cpAnnounced.now} present when announced`);

// Pushing a bench edit out to the copies. Asked for rather than assumed: a
// compound is a macro, and this is the one action that treats it as a prefab.
const cpSync = await page.evaluate(async ([name, P]) => {
  const ed = await import("/js/editor.js");
  const cp = await import("/js/compounds.js");
  const lg = await import("/js/lights.js");
  const V = BABYLON.Vector3;
  const mine = () => [...ed.state.placements.values()].filter((p) => p.compound === name);
  const copies = new Set(mine().map((p) => p.group)).size;
  const anchors = () => [...cp.compoundInstances(name).entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([g, list]) => [g, list[0].node.position.asArray().map((v) => +v.toFixed(3))]);
  const anchorsBefore = anchors();

  // Edit the recipe: a third piece, and a brighter lamp.
  await cp.editCompound(name);
  const extra = await ed.placeAt(P, new V(-2, 0, -4), { silent: true, noLights: true });
  const lamp = lg.lightsOf(cp.benchMembers()[0]?.id || "")[0];
  if (!lamp) { cp.exitCompoundMode(); return { copies, members: cp.benchMembers().length }; }
  lg.setLightPart(lamp.id, "runtime", { intensity: 9 });
  const r = await cp.quickSaveCompound({ updateInstances: true });
  cp.exitCompoundMode();

  const after = mine();
  const bright = after.reduce((a, p) => a
    + lg.lightsOf(p.id).filter((l) => l.runtime.intensity === 9).length, 0);
  const anchorsAfter = anchors();
  await ed.undo();
  return {
    r, copies, extra: !!extra,
    n: after.length,
    groups: new Set(after.map((p) => p.group)).size,
    anchorsHeld: JSON.stringify(anchorsBefore) === JSON.stringify(anchorsAfter),
    bright,
    undone: [...ed.state.placements.values()].filter((p) => p.compound === name).length,
  };
}, [`${CNAME} rigid`, PLATE]);
check("updating copies rebuilds every instance from the bench",
  cpSync.r.ok && cpSync.r.instances === cpSync.copies && cpSync.n === cpSync.copies * 3
    && cpSync.groups === cpSync.copies,
  `${JSON.stringify(cpSync.r)}, ${cpSync.n} element(s) in ${cpSync.groups} group(s)`);
check("each copy keeps its place, and the edit reaches its lamps",
  cpSync.anchorsHeld && cpSync.bright === cpSync.copies,
  `anchors held ${cpSync.anchorsHeld}, ${cpSync.bright} brightened lamp(s)`);
check("and the push is one undo step, on the ship's stack",
  cpSync.undone === cpSync.copies * 2,
  `${cpSync.undone} element(s) after undo, was ${cpSync.copies * 2} before the push`);

// A bench member's chunk is a private fiction, and a lamp being fitted to one
// has to be able to name it as its owner.
const cpBenchPanel = await page.evaluate(async (name) => {
  const ed = await import("/js/editor.js");
  const cp = await import("/js/compounds.js");
  const lg = await import("/js/lights.js");
  await cp.editCompound(name);
  const member = cp.benchMembers()[0];
  const lamp = member ? lg.lightsOf(member.id)[0] : null;
  if (!lamp) { cp.exitCompoundMode(); return { members: cp.benchMembers().length }; }
  ed.select([member.id]);
  await new Promise((r) => setTimeout(r, 80));
  const chunkRow = document.getElementById("insp-chunk").parentElement.hidden;
  ed.select([lamp.id]);
  await new Promise((r) => setTimeout(r, 80));
  const owners = [...document.getElementById("lgt-owner").options].map((o) => o.value);
  const benchIds = cp.benchMembers().map((p) => p.id);
  cp.exitCompoundMode();
  ed.select([]);
  return {
    chunkRow,
    listsBench: benchIds.every((id) => owners.includes(id)),
    listsShip: owners.some((id) => !benchIds.includes(id)),
    owners: owners.length,
  };
}, `${CNAME} rigid`);
check("the Chunk row is hidden on a bench, where a chunk means nothing",
  cpBenchPanel.chunkRow === true, `hidden ${cpBenchPanel.chunkRow}`);
check("and a lamp on the bench is offered the bench's pieces as owners",
  cpBenchPanel.listsBench && !cpBenchPanel.listsShip,
  `${cpBenchPanel.owners} owner(s) offered, ship pieces among them ${cpBenchPanel.listsShip}`);

await page.evaluate(async (name) => {
  const cp = await import("/js/compounds.js");
  await cp.deleteCompound(name);
}, `${CNAME} rigid`);

await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  i.cancelGhost();
  ed.clearAll(); ed.select([]);
  localStorage.removeItem("compoundBench");
});
await page.waitForTimeout(300);

// ---- 1z. the palette panes fold, and the brush label survives a short window
//
// The two panes are sized to their content, and once they were taller than the
// room left they pushed the brush label out of the bottom of the palette and
// under the status bar - so on anything below about 615 px there was no sign
// of what you were holding. Measured at the end of the run: it resizes the
// window, and puts it back before the screenshot.
const paneFold = await page.evaluate(() => {
  const box = (sel) => {
    const b = document.querySelector(sel).getBoundingClientRect();
    return { top: Math.round(b.top), bottom: Math.round(b.bottom), h: Math.round(b.height) };
  };
  const label = document.getElementById("brush-label");
  const lb = label.getBoundingClientRect();
  const hit = document.elementFromPoint(lb.left + 4, lb.top + lb.height / 2);
  return { palette: box("#palette"), foot: box(".palette-foot"), status: box("#status"),
    list: box("#palette-list"),
    open: [...document.querySelectorAll("#palette-panes details")].map((d) => d.open),
    onTop: hit === label || !!hit?.contains(label) };
});
check("all palette panes start open, with the brush label under them",
  paneFold.open.length === 3 && paneFold.open.every(Boolean)
    && paneFold.foot.bottom <= paneFold.palette.bottom + 1 && paneFold.onTop,
  `${JSON.stringify(paneFold.open)}, foot ends ${paneFold.foot.bottom},`
  + ` palette ends ${paneFold.palette.bottom}, painted on top ${paneFold.onTop}`);

// A pane wider than the palette hangs a scrollbar along the bottom of the
// menu, which is never anything but a mistake: every row down here is built to
// the palette's width, whatever that width is. The read-outs in the last
// column do lean into the pane's right padding on purpose - "1.00x" was never
// going to fit a 16 px track - so what is checked is that nothing spills past
// the padding, not that each label sits inside its own column.
const paneWide = await page.evaluate(() => {
    const read = (sel) => {
        const el = document.querySelector(sel);
        return { over: el.scrollWidth - el.clientWidth, w: el.clientWidth };
    };
    const worst = [...document.querySelectorAll("#palette-panes .cfg-row")]
        .map((el) => ({
            row: (el.textContent || "").trim().split("\n")[0].slice(0, 24),
            over: el.scrollWidth - el.clientWidth,
        }))
        .sort((a, b) => b.over - a.over)[0];
    return { panes: read("#palette-panes"), list: read("#palette-list"), worst };
});
check(
    "no pane outgrows the palette, so the menu never scrolls sideways",
    paneWide.panes.over <= 0 && paneWide.list.over <= 0,
    `panes ${paneWide.panes.over} px past ${paneWide.panes.w}, list ${paneWide.list.over} px,` + ` widest row "${paneWide.worst?.row}" +${paneWide.worst?.over} px`
);

await page.click("#settings-pane > summary");
await page.waitForTimeout(250);
const paneShut = await page.evaluate(() => ({
  settings: Math.round(document.querySelector("#settings-pane").getBoundingClientRect().height),
  list: Math.round(document.querySelector("#palette-list").getBoundingClientRect().height),
  shown: document.getElementById("cfg-hull-tol").checkVisibility(),
  stored: localStorage.getItem("pane.settings-pane"),
}));
check("folding a pane leaves its header and hands the room back to the list",
  paneShut.settings < 45 && paneShut.list > paneFold.list.h && paneShut.shown === false,
  `settings ${paneShut.settings} px, list ${paneFold.list.h} -> ${paneShut.list},`
  + ` controls shown ${paneShut.shown}`);
check("and the fold is remembered outside the ship, in localStorage",
  paneShut.stored === "0", `pane.settings-pane = ${paneShut.stored}`);

await page.setViewportSize({ width: 1700, height: 560 });
await page.waitForTimeout(400);
const paneTight = await page.evaluate(() => {
  const box = (sel) => {
    const b = document.querySelector(sel).getBoundingClientRect();
    return { top: Math.round(b.top), bottom: Math.round(b.bottom), h: Math.round(b.height) };
  };
  const label = document.getElementById("brush-label");
  const lb = label.getBoundingClientRect();
  const hit = document.elementFromPoint(lb.left + 4, lb.top + lb.height / 2);
  return { palette: box("#palette"), foot: box(".palette-foot"), status: box("#status"),
    list: box("#palette-list"), panes: box("#palette-panes"),
    onTop: hit === label || !!hit?.contains(label) };
});
await page.setViewportSize({ width: 1700, height: 950 });
await page.waitForTimeout(400);
check("on a short window the brush label keeps its place and the panes scroll",
  paneTight.foot.bottom <= paneTight.palette.bottom + 1
    && paneTight.foot.bottom <= paneTight.status.top + 1
    && paneTight.onTop && paneTight.list.h >= 90,
  `foot ${paneTight.foot.top}-${paneTight.foot.bottom},`
  + ` palette ends ${paneTight.palette.bottom}, status starts ${paneTight.status.top},`
  + ` list ${paneTight.list.h} px, panes ${paneTight.panes.h} px`);

await page.click("#settings-pane > summary");
await page.waitForTimeout(250);
const paneBack = await page.evaluate(() => ({
  open: document.querySelector("#settings-pane").open,
  shown: document.getElementById("cfg-hull-tol").checkVisibility(),
  h: Math.round(document.querySelector("#viewport").getBoundingClientRect().height),
}));
check("unfolding brings the controls back, and the window is as it was",
  paneBack.open && paneBack.shown && paneBack.h > 700,
  `open ${paneBack.open}, controls ${paneBack.shown}, viewport ${paneBack.h} px`);

// ---- both side panels are draggable ----------------------------------------
// The palette carries a fixed number of columns, so its width is the tile size,
// and the inspector's rows are long enough to wrap at 232 px. Neither was
// adjustable, and both are read for hours at a stretch.
const gripBefore = await page.evaluate(() => ({
  palette: Math.round(document.getElementById("palette").getBoundingClientRect().width),
  inspector: Math.round(document.getElementById("inspector").getBoundingClientRect().width),
  viewport: Math.round(document.getElementById("viewport").getBoundingClientRect().width),
  grips: ["resize-palette", "resize-inspector"].filter((id) => document.getElementById(id)).length,
}));

async function dragGrip(id, dx) {
  const b = await page.locator(`#${id}`).boundingBox();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2 + dx, b.y + b.height / 2, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(200);
}

await dragGrip("resize-palette", 140);
await dragGrip("resize-inspector", -100);
const gripAfter = await page.evaluate(() => ({
  palette: Math.round(document.getElementById("palette").getBoundingClientRect().width),
  inspector: Math.round(document.getElementById("inspector").getBoundingClientRect().width),
  viewport: Math.round(document.getElementById("viewport").getBoundingClientRect().width),
  storedPalette: localStorage.getItem("paletteWidth.big"),
  storedInspector: localStorage.getItem("inspectorWidth"),
  // the canvas has to be told, or it renders at the old size, stretched
  canvas: Math.round(document.getElementById("render-canvas").width / devicePixelRatio),
}));
check("both panels have a grip, and dragging it moves them",
  gripBefore.grips === 2
    && Math.abs(gripAfter.palette - gripBefore.palette - 140) <= 4
    && Math.abs(gripAfter.inspector - gripBefore.inspector - 100) <= 4,
  `palette ${gripBefore.palette}->${gripAfter.palette}, `
  + `inspector ${gripBefore.inspector}->${gripAfter.inspector}`);
check("the viewport gives up exactly what the panels took, and re-renders at it",
  Math.abs(gripAfter.viewport - (gripBefore.viewport - 240)) <= 6
    && Math.abs(gripAfter.canvas - gripAfter.viewport) <= 6,
  `viewport ${gripBefore.viewport}->${gripAfter.viewport}, canvas ${gripAfter.canvas}`);
check("the widths are remembered here, not in the ship",
  Math.abs(parseFloat(gripAfter.storedPalette) - gripAfter.palette) <= 1
    && Math.abs(parseFloat(gripAfter.storedInspector) - gripAfter.inspector) <= 1,
  `stored ${gripAfter.storedPalette} / ${gripAfter.storedInspector}`);

// Big icons is the palette's other width. One remembered number for both
// would mean one drag permanently defeated the toggle.
const gripBig = await page.evaluate(async () => {
  const b = document.getElementById("big-palette");
  b.checked = false; b.dispatchEvent(new Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 120));
  const small = Math.round(document.getElementById("palette").getBoundingClientRect().width);
  b.checked = true; b.dispatchEvent(new Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 120));
  const big = Math.round(document.getElementById("palette").getBoundingClientRect().width);
  // and double-click puts whichever one you are on back to its default
  document.getElementById("resize-palette").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
  document.getElementById("resize-inspector").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 120));
  const reset = Math.round(document.getElementById("palette").getBoundingClientRect().width);
  const resetInspector = Math.round(document.getElementById("inspector").getBoundingClientRect().width);
  for (const k of ["paletteWidth", "paletteWidth.big", "inspectorWidth"]) localStorage.removeItem(k);
  return { small, big, reset, resetInspector };
});
check("the two icon sizes keep separate widths, and double-click restores",
  gripBig.small === 260 && Math.abs(gripBig.big - gripAfter.palette) <= 1
    && gripBig.reset === 520 && gripBig.resetInspector === 232,
  `small ${gripBig.small}, big ${gripBig.big}, after double-click `
  + `${gripBig.reset} / ${gripBig.resetInspector}`);

const editorCleanup = await page.evaluate(() => {
  const settings = document.getElementById("settings-pane");
  const sections = [...document.querySelectorAll("#settings-pane .settings-section")];
  const titleOf = (s) => s.querySelector("h3")?.textContent.trim();
  const runtimeSection = sections.find((s) => titleOf(s) === "Runtime");
  const source = BABYLON.ShaderStore?.IncludesShadersStore?.imageProcessingFunctions || "";
  const clamp = source.indexOf("result.rgb = max(result.rgb, vec3(1e-7));");
  const neutral = source.lastIndexOf("PBRNeutralToneMapping(result.rgb);");
  return {
    ghostInSettings: settings?.contains(document.getElementById("veil-alpha")),
    bigIconsInSettings: settings?.contains(document.getElementById("big-palette")),
    sections: sections.map(titleOf),
    aaInRuntime: !!runtimeSection?.contains(document.getElementById("runtime-specular-aa")),
    roughnessInRuntime: !!runtimeSection?.contains(document.getElementById("runtime-roughness")),
    taaRemoved: !document.getElementById("taa"),
    neutralPatchBeforeCall: clamp >= 0 && neutral > clamp,
  };
});
check("Ghost and Big icons live in global Settings",
  editorCleanup.ghostInSettings && editorCleanup.bigIconsInSettings,
  `ghost ${editorCleanup.ghostInSettings}, big icons ${editorCleanup.bigIconsInSettings}`);
// Both dials are saved in the manifest and read by the game, so the section
// that told you they were preview-only is gone and they sit with the rest of
// the ship's runtime settings.
check("Specular AA and Reflection roughness sit under Runtime, with no preview section left",
  editorCleanup.aaInRuntime && editorCleanup.roughnessInRuntime
    && !editorCleanup.sections.includes("Runtime preview"),
  `sections ${JSON.stringify(editorCleanup.sections)}, `
  + `AA ${editorCleanup.aaInRuntime}, roughness ${editorCleanup.roughnessInRuntime}`);
check("TAA is removed from the editor",
  editorCleanup.taaRemoved, `TAA control present: ${!editorCleanup.taaRemoved}`);
check("Khronos PBR Neutral shader patch is installed before the tone mapper",
  editorCleanup.neutralPatchBeforeCall,
  `clamp before call: ${editorCleanup.neutralPatchBeforeCall}`);

// ---- authored lights -------------------------------------------------------
// A light rides a placement, so the module's transform carries it, and the
// record holds one Babylon light: what the runtime will create at that spot.
const lightModel = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
  const owner = await ed.placeAt(M, new V(10, 0, 0), { silent: true });
  const bench = await ed.placeAt(M, new V(0, 0, 0), { silent: true, stage: true });

  const l = lt.addLight(owner.id, { offset: [0, 2.5, 0], silent: true });
  return {
    id: l.id,
    parented: l.node.parent === owner.node,
    // the offset is the OWNER's local space, so it lands beside the owner
    world: l.node.getAbsolutePosition().asArray().map((v) => +v.toFixed(3)),
    runtime: l.runtime,
    onBench: lt.addLight(bench.id, { silent: true }),
    unknown: lt.addLight("nope", { silent: true }),
    listed: lt.lightsOf(owner.id).length,
  };
});
check("a light attaches to a placement and hangs off its node",
  lightModel.parented && lightModel.listed === 1
    && lightModel.world.join() === "10,2.5,0",
  `${lightModel.id} at [${lightModel.world}], parented ${lightModel.parented}`);
check("it defaults to a ceiling lamp: a clustered point light",
  lightModel.runtime.type === "point" && lightModel.runtime.clustered
    && lightModel.runtime.intensity === 1,
  JSON.stringify(lightModel.runtime));
check("nothing that cannot reach the ship can own one",
  lightModel.onBench === null && lightModel.unknown === null,
  `bench ${lightModel.onBench}, unknown ${lightModel.unknown}`);

// The combinations the engine has no meaning for are settled on the way in,
// rather than trusted to the inspector, a loaded manifest and kit_lights.json
// each getting it right on their own.
const lightRules = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const id = [...ed.state.lights.keys()][0];
  const run = (patch) => ({ ...lt.setLightPart(id, "runtime", patch).runtime });
  return {
    // no cube shadow generator exists, so a point light can never cast
    point: run({ type: "point", clustered: false, castsShadows: true }),
    // a clustered light is packed into a data texture with no shadow map
    clusteredSpot: run({ type: "spot", clustered: true, castsShadows: true }),
    // ... but an ordinary spot can
    spot: run({ type: "spot", clustered: false, castsShadows: true }),
    // a directional has no position to bin and no falloff to cluster
    dir: run({ type: "directional", clustered: true, castsShadows: true }),
    bogus: run({ type: "lava-lamp" }),
    // a lamp that reaches nowhere and a colour off the end of the scale
    clamped: run({ range: -3, intensity: -1, angle: 500, color: [2, -1, 0.5] }),
  };
});
check("a point light cannot cast a shadow, and neither can a clustered one",
  !lightRules.point.castsShadows && !lightRules.clusteredSpot.castsShadows,
  `point ${lightRules.point.castsShadows}, clustered spot ${lightRules.clusteredSpot.castsShadows}`);
check("an ordinary spot can, which is the whole reason for the flag",
  lightRules.spot.castsShadows && lightRules.spot.type === "spot",
  JSON.stringify(lightRules.spot));
check("a directional light is never clustered",
  !lightRules.dir.clustered && lightRules.dir.castsShadows,
  JSON.stringify(lightRules.dir));
check("an unknown kind falls back rather than reaching the scene",
  lightRules.bogus.type === "point", JSON.stringify(lightRules.bogus));
check("out-of-range values are clamped on the way in, not stored",
  lightRules.clamped.range > 0 && lightRules.clamped.intensity === 0
    && lightRules.clamped.angle === 179
    && lightRules.clamped.color.join() === "1,0,0.5",
  JSON.stringify(lightRules.clamped));

// A light is part of what the element IS - it goes with a copy and dies with
// the original, the same way its collision shapes do.
const lightLife = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const ownerId = [...ed.state.lights.values()][0].owner;
  ed.select([ownerId]);
  await ed.duplicateSelected();
  const copyId = ed.state.selection[0];
  const copied = lt.lightsOf(copyId);
  const snapshot = JSON.parse(JSON.stringify(ed.serialize()));
  ed.removePlacement(ownerId);
  const afterDelete = ed.state.lights.size;
  await ed.deserialize(snapshot);
  const back = lt.serializeLights();
  // a light whose owner did not come back has nowhere to hang
  const orphaned = { ...snapshot, lights: snapshot.lights.map((l) => ({ ...l, owner: "P9999" })) };
  await ed.deserialize(orphaned);
  return {
    copies: copied.length,
    distinct: copied.length === 1 && copied[0].id !== snapshot.lights[0].id,
    sameSpot: copied[0] && lt.lightOffset(copied[0]).join(),
    saved: snapshot.lights.length,
    afterDelete,
    restored: back.length,
    // field for field, not just the count
    identical: JSON.stringify(back) === JSON.stringify(snapshot.lights),
    edited: `${back[0]?.runtime.type} ${back[0]?.runtime.intensity} ${back[0]?.runtime.angle}`,
    orphans: ed.state.lights.size,
  };
});
check("a copied element brings its lights, as new entries of its own",
  lightLife.copies === 1 && lightLife.distinct && lightLife.sameSpot === "0,2.5,0",
  JSON.stringify(lightLife));
check("deleting the element it rides takes the light with it",
  lightLife.afterDelete === 1, `${lightLife.afterDelete} left`);
check("lights ride the undo stack, field for field",
  lightLife.saved === 2 && lightLife.restored === 2 && lightLife.identical
    // the edits above, not the defaults a fresh light would come back with
    && lightLife.edited === "point 0 179",
  JSON.stringify(lightLife));
check("and one whose element is gone is dropped, not stranded",
  lightLife.orphans === 0, `${lightLife.orphans} orphan(s)`);

// Ctrl+D has a direct duplicate path for a light: it has no module to put in a
// ghost, so the copy is placed beside it in the owner's local space.
const lightDuplicate = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const V = BABYLON.Vector3;
  ed.clearAll(); ed.select([]);
  ed.state.snap.pos = 1;
  const owner = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0),
    { silent: true });
  const source = lt.addLight(owner.id, { offset: [0, 2, 0], silent: true });
  ed.select([source.id]);
  document.getElementById("btn-duplicate").click();
  const selected = ed.state.selection.map((id) => ed.entryOf(id));
  return {
    count: ed.state.lights.size,
    selected: selected.length,
    owner: selected[0]?.owner,
    offset: selected[0] ? lt.lightOffset(selected[0]).join() : null,
    status: document.getElementById("status-text").textContent,
  };
});
check("Ctrl+D duplicates a selected light beside its source",
  lightDuplicate.count === 2 && lightDuplicate.selected === 1
    && lightDuplicate.owner?.startsWith("P")
    && lightDuplicate.offset === "1,2,0"
    && /copy of light/.test(lightDuplicate.status),
  JSON.stringify(lightDuplicate));

// The undo stack is not what persists - the manifest is. A light that rode the
// snapshot but never reached buildManifest() would survive every undo and be
// lost by the next save.
const lightManifest = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const mf = await import("/js/manifest.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const owner = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(4, 0, 0), { silent: true });
  lt.addLight(owner.id, {
    offset: [0, 3, 0],
    runtime: { type: "spot", clustered: false, castsShadows: true, angle: 45 },
    silent: true,
  });
  const man = JSON.parse(JSON.stringify(mf.buildManifest()));
  await ed.deserialize(man);
  const back = lt.serializeLights();
  return {
    written: man.lights,
    declared: (man.space?.editor || []).includes("lights"),
    identical: JSON.stringify(back) === JSON.stringify(man.lights),
  };
});
check("the manifest carries the lights, in the space it declares them in",
  lightManifest.written?.length === 1 && lightManifest.declared,
  JSON.stringify(lightManifest.written));
check("and a saved ship reloads them unchanged",
  lightManifest.identical, JSON.stringify(lightManifest));

// kit_lights.json: a light module arrives lit, so nobody has to find the spot
// under its face by hand for every panel in the ship.
const kitLights = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const mf = await import("/js/manifest.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  const wide = await ed.placeAt("Modular SciFi MegaKit/Props/Prop_Light_Wide", new V(0, 3, 0), { silent: true });
  const seeded = lt.lightsOf(wide.id);
  // an arc cannot be one flat area light, so the corner strip is three
  const corner = await ed.placeAt("Modular SciFi MegaKit/Props/Prop_Light_Corner", new V(20, 3, 0), { silent: true });
  // nothing in the kit lights a plain wall
  const wall = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(40, 0, 0), { silent: true });
  // ... and the collision bench is a stand-in, not ship geometry
  const bench = await ed.placeAt("Modular SciFi MegaKit/Props/Prop_Light_Wide", new V(0, 0, 0), { silent: true, stage: true });

  // the seed is a starting point, not a derivation: edit it, then copy and save
  lt.setLightPart(seeded[0].id, "runtime", { intensity: 7 });
  ed.select([wide.id]);
  await ed.duplicateSelected();
  const copied = lt.lightsOf(ed.state.selection[0]);

  const before = ed.state.lights.size;
  const man = JSON.parse(JSON.stringify(mf.buildManifest()));
  await ed.deserialize(man);
  const reloaded = [...ed.state.lights.values()];

  return {
    wideCount: seeded.length,
    offset: seeded[0] && lt.lightOffset(seeded[0]).join(),
    kind: seeded[0]?.runtime.type,
    range: seeded[0]?.runtime.range,
    clustered: seeded[0]?.runtime.clustered,
    cornerCount: lt.lightsOf(corner.id).length,
    cornerTurned: lt.lightsOf(corner.id).map((l) => lt.lightRotation(l)[1]).join(),
    wallCount: lt.lightsOf(wall.id).length,
    benchCount: lt.lightsOf(bench.id).length,
    copies: copied.length,
    copiedIntensity: copied[0]?.runtime.intensity,
    before,
    after: reloaded.length,
    keptEdit: reloaded.filter((l) => l.runtime.intensity === 7).length,
  };
});
check("placing a light module brings its lamp, on the face of its strip",
  kitLights.wideCount === 1 && kitLights.offset === "0.58,-0.1,0"
    && kitLights.kind === "point" && kitLights.range === 8 && kitLights.clustered,
  JSON.stringify(kitLights));
check("a curved strip gets one lamp per segment, each turned to its tangent",
  kitLights.cornerCount === 3 && kitLights.cornerTurned === "-17.3,-45.7,-74.1",
  `${kitLights.cornerCount} at [${kitLights.cornerTurned}]`);
check("a module with nothing to light gets nothing, and neither does the bench",
  kitLights.wallCount === 0 && kitLights.benchCount === 0,
  `wall ${kitLights.wallCount}, bench ${kitLights.benchCount}`);
check("a copy carries the edited light rather than a fresh default",
  kitLights.copies === 1 && kitLights.copiedIntensity === 7,
  `${kitLights.copies} copy(ies) at intensity ${kitLights.copiedIntensity}`);
check("and a reload restores what was saved instead of seeding a second lamp",
  kitLights.after === kitLights.before && kitLights.keptEdit === 2,
  `${kitLights.before} -> ${kitLights.after}, ${kitLights.keptEdit} edited`);

// The inspector, and the plate that stands in for the light in the viewport.
// A light is the only thing in this editor parented to another element, so the
// gizmo has to be visible to the eye and invisible to everything that walks an
// element's meshes.
const lightUi = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);

  const owner = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(0, 0, 0), { silent: true });
  const bare = ed.worldBounds(owner.node);
  const light = lt.addLight(owner.id, { offset: [0, 6, 0], silent: true });
  const withLight = ed.worldBounds(owner.node);
  ed.select([light.id]);
  const shown = (id) => !document.getElementById(id).hidden;
  const val = (id) => document.getElementById(id).value;
  const off = (id) => document.getElementById(id).disabled;

  const panel = {
    light: shown("light-fields"), scale: shown("scale-fields"),
    behavior: shown("behavior-fields"), door: shown("door-fields"),
    ownerId: document.getElementById("lgt-owner").value,
    ownerLabel: document.getElementById("lgt-owner").selectedOptions[0]?.textContent,
    kind: val("lgt-type"), intensity: val("lgt-intensity"),
    // a point light has no cone, and cannot cast whatever the checkbox says
    angleOff: off("lgt-angle"), shadowsOff: off("lgt-shadows"),
    // ...and every lamp with a position has a Range the engine reads
    rangeOff: off("lgt-range"),
    // the offset row is the light's own, in its owner's space
    posY: val("pos-y"),
  };

  // editing a field lands on the record, through the same normalisation
  const intensityEl = document.getElementById("lgt-intensity");
  intensityEl.value = "12";
  intensityEl.dispatchEvent(new Event("input", { bubbles: true }));
  const clusteredEl = document.getElementById("lgt-clustered");
  clusteredEl.checked = false;
  clusteredEl.dispatchEvent(new Event("change", { bubbles: true }));
  const typeEl = document.getElementById("lgt-type");
  typeEl.value = "spot";
  typeEl.dispatchEvent(new Event("change", { bubbles: true }));

  const newOwner = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", new V(4, 0, 0),
    { silent: true });
  const beforeRehome = light.node.getAbsolutePosition().clone();
  ed.select([light.id]);
  const ownerEl = document.getElementById("lgt-owner");
  ownerEl.value = newOwner.id;
  ownerEl.dispatchEvent(new Event("change", { bubbles: true }));
  const afterRehome = light.node.getAbsolutePosition().clone();

  // A directional light is the one lamp Range still means nothing to: no
  // position, so no distance to fall off over. Probe it and put the light back
  // the way the checks below expect to find it.
  typeEl.value = "directional";
  typeEl.dispatchEvent(new Event("change", { bubbles: true }));
  const rangeDirectional = off("lgt-range");
  const hintDirectional = document.getElementById("lgt-hint").textContent;
  typeEl.value = "spot";
  typeEl.dispatchEvent(new Event("change", { bubbles: true }));

  // Only the gizmos that belong to a *light*: the probe window leaves its own
  // box/centre/camera gizmos parked (disabled) in the scene, and they are none
  // of this check's business. Keeping every light's gizmo in scope still
  // catches one leaking from a lamp that was edited earlier.
  const gizmo = ed.state.scene.meshes
    .filter((m) => ed.isGizmoMesh(m) && m.metadata.lightRoot);
  const shipMeshes = owner.node.getChildMeshes()
    .filter((m) => !ed.isGizmoMesh(m)).length;

  // selecting a placement again must not leave the light panel on screen
  ed.select([owner.id]);
  const afterOwner = shown("light-fields");

  return {
    panel,
    editedIntensity: light.runtime.intensity,
    editedClustered: light.runtime.clustered,
    editedType: light.runtime.type,
    rehomed: light.owner === newOwner.id && light.node.parent === newOwner.node,
    rehomeWorldGap: BABYLON.Vector3.Distance(beforeRehome, afterRehome),
    // a spot has a cone, and an unclustered one can cast
    shadowsNow: off("lgt-shadows") === false,
    angleNow: off("lgt-angle") === false,
    // ...and unclustering no longer takes Range away
    rangeNow: off("lgt-range"),
    rangeHint: document.getElementById("lgt-hint").textContent,
    rangeDirectional,
    hintDirectional,
    gizmos: gizmo.length,
    gizmoOwned: gizmo.every((m) => m.metadata.lightRoot === light.node),
    shipMeshes,
    boundsUnchanged: bare && withLight
      && bare.max.y.toFixed(3) === withLight.max.y.toFixed(3),
    afterOwner,
  };
});
check("selecting a light shows its two forms and drops the ones it has none of",
  lightUi.panel.light && !lightUi.panel.scale && !lightUi.panel.behavior
    && !lightUi.panel.door && lightUi.panel.posY === "6",
  JSON.stringify(lightUi.panel));
check("the panel says what the light rides, and starts on the defaults",
  lightUi.panel.ownerId === "P0001"
    && lightUi.panel.ownerLabel === "P0001 — Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight"
    && lightUi.panel.kind === "point" && lightUi.panel.intensity === "1",
  JSON.stringify(lightUi.panel));
check("the light owner is editable and reparenting keeps its world position",
  lightUi.rehomed && lightUi.rehomeWorldGap < 1e-5,
  JSON.stringify({ rehomed: lightUi.rehomed, worldGap: lightUi.rehomeWorldGap }));
check("rows the engine has no meaning for are disabled, not silently ignored",
  lightUi.panel.angleOff && lightUi.panel.shadowsOff,
  JSON.stringify(lightUi.panel));
check("editing a field lands on the record, and reopens the rows it unlocks",
  lightUi.editedIntensity === 12 && lightUi.editedClustered === false
    && lightUi.editedType === "spot" && lightUi.shadowsNow && lightUi.angleNow,
  JSON.stringify(lightUi));
// Range reaches the shader on every lamp that has a position, and only because
// the preview materials ask Babylon for the glTF falloff instead of the PBR
// physical one. Physical is a plain 1/d² that never cuts off, so Range authored
// a number nothing read - and on a clustered lamp it was worse still, because
// the cluster sizes and culls the light proxy by Range and the light stopped
// dead where the proxy ended.
check("Range is live on a clustered lamp and on an unclustered one alike",
  lightUi.panel.rangeOff === false && lightUi.rangeNow === false,
  JSON.stringify({ clustered: lightUi.panel.rangeOff, unclustered: lightUi.rangeNow }));
check("an unclustered lamp is told the game ramps it down to Range differently",
  /straight ramp rather than this preview/.test(lightUi.rangeHint),
  JSON.stringify(lightUi.rangeHint));
// A directional light is the one lamp with no distance to fall off over.
check("Range is dead on a directional light, with the reason under it",
  lightUi.rangeDirectional === true && /no distance falloff/.test(lightUi.hintDirectional),
  JSON.stringify({ off: lightUi.rangeDirectional, hint: lightUi.hintDirectional }));
check("the light has a plate and a stub in the viewport, both hung off its node",
  lightUi.gizmos === 2 && lightUi.gizmoOwned, `${lightUi.gizmos} gizmo mesh(es)`);
check("neither reaches the ship: not its meshes, not its size, not the export",
  lightUi.shipMeshes > 0 && lightUi.boundsUnchanged,
  `${lightUi.shipMeshes} art mesh(es), bounds unchanged ${lightUi.boundsUnchanged}`);
check("and selecting the element it rides puts the light panel away", !lightUi.afterOwner, `still shown: ${lightUi.afterOwner}`);

// ---- L steps between an element and the lamps riding it ---------------------
// A lamp sits inside the thing it lights, so its gizmo and the element's surface
// are the same few pixels and clicking either means missing the other. L walks
// the element and everything riding it, then returns - so every press lands
// somewhere and a single-lamp element is simply a toggle.
const lightKey = await page.evaluate(async () => {
    const ed = await import("/js/editor.js");
    const lt = await import("/js/lights.js");
    const i = await import("/js/interact.js");
    const V = BABYLON.Vector3;
    i.cancelGhost();
    ed.clearAll();
    ed.select([]);
    const M = "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight";
    const lit = await ed.placeAt(M, new V(40, 0, 0), { silent: true });
    const dark = await ed.placeAt(M, new V(48, 0, 0), { silent: true });
    const one = lt.addLight(lit.id, { silent: true });
    const two = lt.addLight(lit.id, { silent: true });

    const press = () => {
        window.dispatchEvent(new KeyboardEvent("keydown", { key: "l", code: "KeyL", bubbles: true }));
        return { sel: [...ed.state.selection], status: (document.getElementById("status-text")?.textContent || "").trim() };
    };

    ed.select([lit.id]);
    const walk = [press(), press(), press(), press()];

    // an element with no lamp says so and leaves the selection alone
    ed.select([dark.id]);
    const none = press();

    // and so does nothing at all selected
    ed.select([]);
    const empty = press();

    ed.clearAll();
    ed.select([]);
    return { owner: lit.id, dark: dark.id, one: one.id, two: two.id, walk, none, empty };
});
check("L steps from the element onto its first lamp", lightKey.walk[0].sel.join() === lightKey.one, `${JSON.stringify(lightKey.walk[0].sel)} — "${lightKey.walk[0].status}"`);
check(
    "L again visits the second lamp, then returns to the element and round again",
    lightKey.walk[1].sel.join() === lightKey.two && lightKey.walk[2].sel.join() === lightKey.owner && lightKey.walk[3].sel.join() === lightKey.one,
    lightKey.walk.map((w) => w.sel.join() || "-").join(" -> ")
);
check(
    "on an element with no lamp it says so and leaves the selection alone",
    lightKey.none.sel.join() === lightKey.dark && /no light on /.test(lightKey.none.status),
    `${JSON.stringify(lightKey.none.sel)} — "${lightKey.none.status}"`
);
check(
    "and with nothing selected it asks for something to step from",
    lightKey.empty.sel.length === 0 && /select an element/.test(lightKey.empty.status),
    `"${lightKey.empty.status}"`
);

// ---- a lamp starts at the numbers its own kind needs ------------------------
// Intensity, range and cone mean different things per kind: 1 is bright for a
// point light filling a small room and invisible for a spot throwing a cone at
// a wall six metres away. Choosing "spot" and getting a point light's numbers
// is what made spots look like they did not work at all.
const spotDefaults = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  const owner = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
    new V(0, 0, 0), { silent: true, noLights: true });
  const take = (l) => [l.runtime.intensity, l.runtime.range, l.runtime.angle];

  // asked for outright, the way a kit seed asks
  const born = lt.addLight(owner.id, { runtime: { type: "spot" }, silent: true });
  const asked = take(born);

  // ...and arrived at through the panel, which is how one is really made
  const lamp = lt.addLight(owner.id, { silent: true });
  const fresh = take(lamp);
  ed.select([lamp.id]);
  await new Promise((r) => setTimeout(r, 60));
  const typeEl = document.getElementById("lgt-type");
  const pick = async (kind) => {
    typeEl.value = kind;
    typeEl.dispatchEvent(new Event("change", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 60));
  };
  await pick("spot");
  const spot = take(lamp);
  const fields = ["lgt-intensity", "lgt-range", "lgt-angle"]
    .map((id) => document.getElementById(id).value);

  // a number typed by hand is an opinion, and survives the next change of kind
  const intensityEl = document.getElementById("lgt-intensity");
  intensityEl.value = "50";
  intensityEl.dispatchEvent(new Event("input", { bubbles: true }));
  await pick("none");
  await pick("spot");
  const tuned = take(lamp);

  // and going back the other way hands the untouched fields to the new kind
  await pick("point");
  const back = take(lamp);
  return { asked, fresh, spot, fields, tuned, back };
});
check("a lamp asked for as a spot is born with a spot's numbers",
  JSON.stringify(spotDefaults.asked) === JSON.stringify([80, 6, 120]),
  JSON.stringify(spotDefaults.asked));
check("choosing spot in the panel moves an untouched lamp onto them too",
  JSON.stringify(spotDefaults.fresh) === JSON.stringify([1, 8, 90])
    && JSON.stringify(spotDefaults.spot) === JSON.stringify([80, 6, 120])
    && JSON.stringify(spotDefaults.fields) === JSON.stringify(["80", "6", "120"]),
  `${JSON.stringify(spotDefaults.fresh)} -> ${JSON.stringify(spotDefaults.spot)},`
  + ` fields ${JSON.stringify(spotDefaults.fields)}`);
check("a number typed by hand outlives a change of kind, and the rest follow it",
  JSON.stringify(spotDefaults.tuned) === JSON.stringify([50, 6, 120])
    && JSON.stringify(spotDefaults.back) === JSON.stringify([50, 8, 90]),
  `tuned ${JSON.stringify(spotDefaults.tuned)}, back on point ${JSON.stringify(spotDefaults.back)}`);

// ---- a field keeps only the keys it can use ---------------------------------
// Standing aside for a focused control is what typing is - but a number field
// cannot spell a letter, so R after typing a position did nothing at all, and
// nothing said why. That is the whole of "sometimes R takes several presses".
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll();
  const el = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
    new V(0, 0, 0), { silent: true });
  ed.select([el.id]);
  ed.state.rotAxis = "y";
});
await page.waitForTimeout(150);
await page.focus("#pos-x");
await page.keyboard.press("r");
await page.waitForTimeout(120);
const blindField = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { axis: ed.state.rotAxis, focus: document.activeElement.id };
});
check("a number field hands back a letter it could only have dropped",
  blindField.axis !== "y" && blindField.focus === "pos-x", JSON.stringify(blindField));

await page.fill("#palette-search", "");
await page.focus("#palette-search");
await page.evaluate(async () => { (await import("/js/editor.js")).state.rotAxis = "y"; });
await page.keyboard.press("r");
await page.waitForTimeout(120);
const textField = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return { axis: ed.state.rotAxis, value: document.getElementById("palette-search").value };
});
check("a text field keeps it, because there the letter is the point",
  textField.axis === "y" && textField.value === "r", JSON.stringify(textField));

// Enter is "done with this field": it hands the keyboard back, which is the way
// out of a search box, where the letters really were the field's.
await page.keyboard.press("Enter");
await page.waitForTimeout(120);
const afterEnter = await page.evaluate(() => document.activeElement.id);
await page.keyboard.press("r");
await page.waitForTimeout(120);
const afterEnterAxis = await page.evaluate(async () =>
  (await import("/js/editor.js")).state.rotAxis);
check("Enter releases a field, so the next shortcut lands on the ship",
  afterEnter !== "palette-search" && afterEnterAxis !== "y",
  `${afterEnter || "(body)"}, axis ${afterEnterAxis}`);
await page.fill("#palette-search", "");
await page.waitForTimeout(300);

// ---- the wheel aims a lamp --------------------------------------------------
// A lamp is aimed rather than built, so while the selection is lights and
// nothing else the wheel drives the light instead of the camera and the
// transform tools: bare intensity, Ctrl range, Alt cone, Shift a 0.5° turn.
const lamp = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const i = await import("/js/interact.js");
  const V = BABYLON.Vector3;
  i.cancelGhost(); ed.clearAll(); ed.select([]);
  // The settings the lamp turn must ignore: a world space and a 90° snap are
  // for laying walls out on a grid, and are not aiming.
  i.setAxisSpace("world");
  ed.state.rotAxis = "y";
  ed.state.snap.rot = 90;
  const owner = await ed.placeAt("Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight",
    new V(0, 0, 0), { silent: true });
  // Turned, so a step about a *world* axis would land on a different local one.
  ed.setEuler(owner.node, [0, 0, 90]);
  const light = lt.addLight(owner.id, {
    offset: [0, 2, 0],
    runtime: { type: "spot", clustered: false, intensity: 1, range: 8, angle: 60 },
    silent: true,
  });
  ed.select([light.id]);
  return { id: light.id, owner: owner.id };
});
{
  const c = await page.evaluate(() =>
    window.__scene.getEngine().getRenderingCanvas().getBoundingClientRect().toJSON());
  await page.mouse.move(c.x + c.width / 2, c.y + c.height / 2, { steps: 3 });
}
const lampCamBefore = await page.evaluate(() => window.__scene.activeCamera.position.asArray());
await page.mouse.wheel(0, -120);
await page.waitForTimeout(150);
const lampUp = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return {
    intensity: ed.entryOf(ed.state.selection[0]).runtime.intensity,
    field: document.getElementById("lgt-intensity").value,
    status: document.getElementById("status-text").textContent,
    cam: window.__scene.activeCamera.position.asArray(),
  };
});
check("the bare wheel steps a selected lamp's intensity, a whole unit on a spot",
  Math.abs(lampUp.intensity - 2) < 1e-9 && lampUp.field === "2"
    && /intensity 2\b/.test(lampUp.status),
  JSON.stringify(lampUp));
check("and it is an edit, so the camera stays where it was",
  JSON.stringify(lampCamBefore) === JSON.stringify(lampUp.cam),
  `${lampCamBefore.map((v) => v.toFixed(1))} -> ${lampUp.cam.map((v) => v.toFixed(1))}`);

// A light of negative intensity is not a shadow, it is a broken manifest.
for (let k = 0; k < 12; k++) await page.mouse.wheel(0, 120);
await page.waitForTimeout(200);
const lampFloor = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return ed.entryOf(ed.state.selection[0]).runtime.intensity;
});
check("winding it down stops at zero rather than going negative",
  lampFloor === 0, `${lampFloor}`);

await page.keyboard.down("Control");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Control");
await page.keyboard.down("Alt");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Alt");
await page.keyboard.down("Shift");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Shift");
await page.waitForTimeout(250);
const lampTuned = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  const e = ed.entryOf(ed.state.selection[0]);
  return {
    range: e.runtime.range, angle: e.runtime.angle,
    rotation: lt.lightRotation(e).join(),
    space: ed.state.axisSpace, snap: ed.state.snap.rot,
    rangeField: document.getElementById("lgt-range").value,
    status: document.getElementById("status-text").textContent,
  };
});
check("Ctrl takes the range and Alt the cone, both by the panel's step",
  Math.abs(lampTuned.range - 8.5) < 1e-9 && lampTuned.angle === 65
    && lampTuned.rangeField === "8.5",
  JSON.stringify(lampTuned));
// The lamp rides an element turned 90° about Z, so a step about the *world* Y
// spinNode would have taken lands on the node's local X instead - which aims
// the beam somewhere nobody asked for.
check("Shift turns it 0.5° about its own current axis, whatever the space says",
  lampTuned.rotation === "0,0.5,0" && lampTuned.space === "world" && lampTuned.snap === 90,
  `${lampTuned.rotation} with space ${lampTuned.space}, snap ${lampTuned.snap}`);

// A wheel notch is a CDP round trip, and the one-entry-per-gesture rule is a
// 400 ms wall-clock window - so the runs of notches that have to collapse into
// single undo entries are dispatched in the page, where they cannot be starved.
//
// Counting the stack would not answer this: it is capped by memory, so at the
// cap a push also drops the oldest entry and the depth never moves. Two
// separated gestures and two undos say the same thing and survive the cap.
const lampUndo = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const id = [...ed.state.lights.keys()][0];
  const canvas = window.__scene.getEngine().getRenderingCanvas();
  const intensity = () => ed.state.lights.get(id).runtime.intensity;
  const gap = () => new Promise((r) => setTimeout(r, 500));
  const notches = (n) => {
    for (let k = 0; k < n; k++) {
      canvas.dispatchEvent(new WheelEvent("wheel", { deltaY: -120, bubbles: true, cancelable: true }));
    }
  };
  await gap();
  const start = intensity();
  notches(5);
  const first = intensity();
  await gap();
  notches(3);
  const second = intensity();
  await ed.undo();
  const back = intensity();
  await ed.undo();
  const backAgain = intensity();
  // The undos rebuilt every record, so the selection is re-taken by id rather
  // than trusted to have survived them.
  ed.select([id]);
  return { start, first, second, back, backAgain };
});
check("a run of notches is one undo entry, and the next gesture is another",
  Math.abs(lampUndo.first - (lampUndo.start + 5)) < 1e-9
    && Math.abs(lampUndo.second - (lampUndo.start + 8)) < 1e-9
    && Math.abs(lampUndo.back - lampUndo.first) < 1e-9
    && Math.abs(lampUndo.backAgain - lampUndo.start) < 1e-9,
  JSON.stringify(lampUndo));

// A lamp that has no such setting says so, rather than silently doing nothing.
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  lt.setLightPart(ed.state.selection[0], "runtime", { type: "directional" });
});
await page.keyboard.down("Control");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Control");
await page.waitForTimeout(150);
const lampNoRange = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return {
    range: ed.entryOf(ed.state.selection[0]).runtime.range,
    status: document.getElementById("status-text").textContent,
  };
});
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const lt = await import("/js/lights.js");
  lt.setLightPart(ed.state.selection[0], "runtime", { type: "point" });
});
await page.keyboard.down("Alt");
await page.mouse.wheel(0, -120);
await page.keyboard.up("Alt");
await page.waitForTimeout(150);
const lampNoCone = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return {
    angle: ed.entryOf(ed.state.selection[0]).runtime.angle,
    status: document.getElementById("status-text").textContent,
  };
});
check("a directional lamp has no range and a point light no cone, and both say so",
  Math.abs(lampNoRange.range - 8.5) < 1e-9 && /no range/.test(lampNoRange.status)
    && lampNoCone.angle === 65 && /only a spot light/.test(lampNoCone.status),
  `${lampNoRange.status} | ${lampNoCone.status}`);

// One notch does not mean one number: the lamp above is now a point light,
// which fills a small room from the inside, where the spot's whole unit would
// blow the room out in a single click.
await page.waitForTimeout(500);
await page.mouse.wheel(0, -120);
await page.waitForTimeout(150);
const lampPointStep = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  return ed.entryOf(ed.state.selection[0]).runtime.intensity;
});
check("the intensity step follows the kind of lamp: 0.05 on a point light",
  Math.abs(lampPointStep - 0.05) < 1e-9, `${lampPointStep}`);

// The bindings arm on a selection that is lamps and *nothing else*: mixed with
// the element it rides, the wheel is the camera dolly again.
const lampMixedStart = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const light = ed.state.lights.values().next().value;
  ed.select([light.id, light.owner]);
  return { intensity: light.runtime.intensity, cam: window.__scene.activeCamera.position.asArray() };
});
await page.mouse.wheel(0, -240);
await page.waitForTimeout(600);
const lampMixed = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const light = ed.state.lights.values().next().value;
  return { intensity: light.runtime.intensity, cam: window.__scene.activeCamera.position.asArray() };
});
{
  const d = Math.hypot(...lampMixed.cam.map((v, k) => v - lampMixedStart.cam[k]));
  check("a mixed selection leaves the wheel to the camera",
    d > 0.5 && lampMixed.intensity === lampMixedStart.intensity,
    `camera moved ${d.toFixed(2)} m, intensity ${lampMixedStart.intensity} -> ${lampMixed.intensity}`);
}
await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  ed.clearAll(); ed.select([]);
});

await page.fill("#palette-search", "");
await page.waitForTimeout(400);

// Beside this file, not beside the launch directory - see the note in
// smoke.mjs. `URL` above shadows the global constructor, hence HERE.
await page.screenshot({ path: path.join(HERE, "shot-ux.png") });

console.log(results.join("\n"));
console.log("\nfailures:", results.filter((r) => r.startsWith("FAIL")).length);console.log("errors  :", errors.length ? [...new Set(errors)].join("\n  ") : "(none)");
await browser.close();
process.exit(results.some((r) => r.startsWith("FAIL")) || errors.length ? 1 : 0);
