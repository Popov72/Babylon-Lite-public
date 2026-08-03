/**
 * Dump each module's triangles, in the module's own space, so the fitter can be
 * developed and re-run without a browser in the loop.
 *
 * Geometry is only readable for a module that exists in the scene, so anything
 * not placed in the ship is put on the collision bench to be read. Nothing is
 * ever saved from this session: staging rewrites `moduleCollision` through
 * harvestStage, so a save here would be destructive.
 */
import { createRequire } from "node:module";
import fs from "node:fs";
const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const URL_ = process.env.DUMP_URL || "http://localhost:5210/";
const OUT = process.env.DUMP_OUT || "D:/alexis/TombRaider/Popov72/fitter/modules.json";
fs.mkdirSync(OUT.replace(/\/[^/]+$/, ""), { recursive: true });

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
page.on("pageerror", (e) => console.log("PAGE ERROR:", e.message));
await page.goto(URL_, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => document.querySelectorAll("#palette-list .item").length > 0,
  null, { timeout: 60000 });
await page.waitForFunction(async () => !(await import("/js/editor.js")).isBusy(),
  null, { timeout: 60000 });
let prev = -1;
for (let k = 0; k < 30; k++) {
  await page.waitForTimeout(500);
  const n = await page.evaluate(async () => (await import("/js/editor.js")).state.placements.size);
  if (n === prev && n > 0) break;
  prev = n;
}

const wanted = await page.evaluate(async () => {
  const ed = await import("/js/editor.js");
  const placed = new Set([...ed.state.placements.values()].filter((e) => !e.stage)
    .map((e) => e.module));
  return {
    hulled: [...ed.state.moduleCollision.keys()],
    placed: [...placed],
    shapes: Object.fromEntries([...ed.state.moduleCollision.entries()]),
  };
});
console.log(`${wanted.hulled.length} modules carry a hull; ${wanted.placed.length} are placed`);

// read straight off placements first - no staging needed
const dump = { authored: wanted.shapes, modules: {} };
const readFrom = async (module, viaBench) => page.evaluate(async ({ module, viaBench }) => {
  const ed = await import("/js/editor.js");
  const co = await import("/js/colliders.js");
  const kit = await import("/js/kit.js");
  const V = BABYLON.Vector3;
  let e = [...ed.state.placements.values()].find((x) => x.module === module);
  if (!e && viaBench) {
    if (!ed.state.collisionMode) await co.enterCollisionMode(kit.instantiate, kit.moduleBounds);
    await co.stageModule(module, kit.instantiate, kit.moduleBounds, [0, 0, 0]);
    await new Promise((r) => setTimeout(r, 900));
    e = [...ed.state.placements.values()].find((x) => x.module === module);
  }
  if (!e) return null;
  e.node.computeWorldMatrix(true);
  const inv = e.node.getWorldMatrix().clone().invert();
  const tris = [];
  for (const m of e.node.getChildMeshes()) {
    const pos = m.getVerticesData && m.getVerticesData("position");
    const idx = m.getIndices && m.getIndices();
    if (!pos || !idx) continue;
    m.computeWorldMatrix(true);
    const w = m.getWorldMatrix().multiply(inv);
    const at = (i) => {
      const q = V.TransformCoordinates(new V(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]), w);
      return [+q.x.toFixed(4), +q.y.toFixed(4), +q.z.toFixed(4)];
    };
    for (let i = 0; i < idx.length; i += 3) tris.push([at(idx[i]), at(idx[i + 1]), at(idx[i + 2])]);
  }
  return tris;
}, { module, viaBench });

for (const module of [...new Set([...wanted.hulled, ...wanted.placed])]) {
  const viaBench = !wanted.placed.includes(module);
  const tris = await readFrom(module, viaBench);
  if (!tris || !tris.length) { console.log(`   ${module}: no geometry`); continue; }
  dump.modules[module] = tris;
  process.stdout.write(`\r   ${Object.keys(dump.modules).length} read`);
}
console.log("");
fs.writeFileSync(OUT, JSON.stringify(dump));
const sizes = Object.entries(dump.modules).map(([m, t]) => `${t.length}`);
console.log(`wrote ${OUT}: ${Object.keys(dump.modules).length} modules,`
  + ` ${sizes.reduce((s, v) => s + Number(v), 0)} triangles`);
await browser.close();
