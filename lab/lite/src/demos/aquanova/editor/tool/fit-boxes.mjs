/**
 * Score the hull fitter against the hand-authored hulls.
 *
 * The fitter itself lives in public/js/hullfit.js, shared with the editor, so
 * the thresholds baked into it are the ones this harness measured. Reads the
 * geometry dumped by fit-dump.mjs; writes nothing back to the ship.
 */
import fs from "node:fs";
import { fit, surfaceIndex, solidFraction } from "./public/js/hullfit.js";

const IN = process.env.FIT_IN || "D:/alexis/TombRaider/Popov72/fitter/modules.json";
const OUT = process.env.FIT_OUT || "D:/alexis/TombRaider/Popov72/fitter/fitted.json";

/** Is a point inside this box? Mirrors hullfit's own test, for the authored hulls. */
function inside(box, p) {
  const d = [0, 1, 2].map((k) => p[k] - box.centre[k]);
  for (let k = 0; k < 3; k++) {
    const v = d[0] * box.basis[k][0] + d[1] * box.basis[k][1] + d[2] * box.basis[k][2];
    if (Math.abs(v) > box.half[k] + 1e-4) return false;
  }
  return true;
}
if (process.argv[1] && process.argv[1].endsWith("fit-boxes.mjs")) {
  const data = JSON.parse(fs.readFileSync(IN, "utf8"));
  const opts = {};
  for (const a of process.argv.slice(2)) {
    const [k, v] = a.replace(/^--/, "").split("=");
    opts[k] = Number(v);
  }

  /** An authored shape in the same box form the fitter emits. */
  const toBox = (s) => {
    const [rx, ry, rz] = s.rotation.map((d) => (d * Math.PI) / 180);
    // Babylon's FromEulerAngles is Ry * Rx * Rz, in the row-vector convention
    const cy = Math.cos(ry), sy = Math.sin(ry);
    const cx = Math.cos(rx), sx = Math.sin(rx);
    const cz = Math.cos(rz), sz = Math.sin(rz);
    const Ry = [[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]];
    const Rx = [[1, 0, 0], [0, cx, sx], [0, -sx, cx]];
    const Rz = [[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]];
    const mm = (A, B) => A.map((r) => [0, 1, 2].map((j) =>
      r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
    return { centre: s.position, half: s.scale.map((v) => Math.abs(v) / 2),
      basis: mm(mm(Rz, Rx), Ry) };
  };

  const authoredCoverage = (module, tris) => {
    const shapes = data.authored[module] || [];
    const verts = [];
    for (const [a, b, c] of tris) verts.push(a, b, c);
    const boxes = shapes.map(toBox);
    let n = 0;
    for (const p of verts) if (boxes.some((bx) => inside(bx, p))) n++;
    return { cov: n / verts.length, count: shapes.length };
  };

  const rows = [];
  for (const [module, tris] of Object.entries(data.modules)) {
    const mine = fit(tris, opts);
    const theirs = authoredCoverage(module, tris);
    rows.push({ module, mineCov: mine.coverage, mineN: mine.boxes.length,
      theirCov: +theirs.cov.toFixed(4), theirN: theirs.count, bins: mine.bins,
      mineVol: +mine.boxes.reduce((s, b) => s + 8 * b.half[0] * b.half[1] * b.half[2], 0).toFixed(2),
      theirVol: +(data.authored[module] || [])
        .reduce((s, x) => s + Math.abs(x.scale[0] * x.scale[1] * x.scale[2]), 0).toFixed(2),
      mineSolid: mine.solid, how: mine.how, confident: mine.confident,
      theirSolid: solidFraction(surfaceIndex(tris), (data.authored[module] || []).map(toBox)) });
  }
  const f = (v) => (100 * v).toFixed(1).padStart(5);
  rows.sort((a, b) => (a.mineCov - a.theirCov) - (b.mineCov - b.theirCov));
  console.log("module".padEnd(42), "         fitter                    authored");
  console.log("".padEnd(42), "  cov  n   vol  solid  how     cov  n   vol  solid");
  for (const r of rows) {
    console.log(r.module.padEnd(42),
      `${f(r.mineCov)}% ${String(r.mineN).padStart(2)} ${String(r.mineVol).padStart(6)} ${f(r.mineSolid)}% ${r.how.padEnd(5)}`
      + ` ${f(r.theirCov)}% ${String(r.theirN).padStart(2)} ${String(r.theirVol).padStart(6)} ${f(r.theirSolid)}%`);
  }
  const avg = (k) => rows.reduce((s, r) => s + r[k], 0) / rows.length;
  console.log("\n" + "-".repeat(96));
  console.log(`fitter  : coverage ${(100 * avg("mineCov")).toFixed(1)}%,`
    + ` ${avg("mineN").toFixed(1)} boxes, ${avg("mineVol").toFixed(1)} m3, solid ${(100 * avg("mineSolid")).toFixed(1)}%`);
  console.log(`authored: coverage ${(100 * avg("theirCov")).toFixed(1)}%,`
    + ` ${avg("theirN").toFixed(1)} boxes, ${avg("theirVol").toFixed(1)} m3, solid ${(100 * avg("theirSolid")).toFixed(1)}%`);
  console.log(`\ncoverage at least as good on `
    + `${rows.filter((r) => r.mineCov + 0.02 >= r.theirCov).length} of ${rows.length};`
    + ` and no bulkier on ${rows.filter((r) => r.mineVol <= r.theirVol * 1.25).length}`);
  const sure = rows.filter((r) => r.confident);
  console.log(`confident on ${sure.length} of ${rows.length}`
    + ` (coverage ${(100 * sure.reduce((s, r) => s + r.mineCov, 0) / sure.length).toFixed(1)}%,`
    + ` solid ${(100 * sure.reduce((s, r) => s + r.mineSolid, 0) / sure.length).toFixed(1)}%);`
    + ` hands off the other ${rows.length - sure.length}:`);
  for (const r of rows.filter((x) => !x.confident)) console.log(`    ${r.module}`);
  fs.writeFileSync(OUT, JSON.stringify(Object.fromEntries(
    Object.entries(data.modules).map(([m, t]) => [m, fit(t, opts)])), null, 1));
}
