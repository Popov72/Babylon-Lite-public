/** Why do some rounded corners only ever get axis-aligned boxes? */
import fs from "node:fs";
import { fit, surfaceIndex, solidFraction } from "./public/js/hullfit.js";

const data = JSON.parse(fs.readFileSync(
  process.env.FIT_IN || "D:/alexis/TombRaider/Popov72/fitter/corners.json", "utf8"));

const NAMES = (process.env.FIT_ONLY || [
  "Modular SciFi MegaKit/Walls/ShortWall_WhitePlate2_Corner_Inner",
  "Modular SciFi MegaKit/Walls/TopAstra_Corner_Round_Inner",
  "Modular SciFi MegaKit/Walls/TopCables_Corner_Round_Outer",
  "Modular SciFi MegaKit/Walls/TopCables_Corner_Round_Inner",
].join(",")).split(",");

const axisAligned = (b) => b.basis.every((r, i) =>
  r.every((v, j) => Math.abs(v - (i === j ? 1 : 0)) < 1e-4 || Math.abs(v + (i === j ? 1 : 0)) < 1e-4));

for (const name of NAMES) {
  const tris = data.modules[name];
  if (!tris) { console.log(`\n${name}: not dumped`); continue; }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const t of tris) {
    for (const p of t) {
      for (let k = 0; k < 3; k++) {
        if (p[k] < lo[k]) lo[k] = p[k];
        if (p[k] > hi[k]) hi[k] = p[k];
      }
    }
  }
  console.log(`\n=== ${name}`);
  console.log(`    ${tris.length} triangles, extent `
    + [0, 1, 2].map((k) => (hi[k] - lo[k]).toFixed(2)).join(" x ") + " m");
  for (const tolerance of [0.4, 0.25, 0.15, 0.1]) {
    const r = fit(tris, { tolerance, thickness: 0.35 });
    const obb = r.boxes.filter((b) => !axisAligned(b)).length;
    console.log(`    tol ${String(tolerance).padEnd(5)} -> ${String(r.boxes.length).padStart(2)}`
      + ` box by ${r.how.padEnd(5)} (${obb} oriented), cov ${(100 * r.coverage).toFixed(1)}%`
      + ` solid ${(100 * r.solid).toFixed(1)}% vol ${r.volume} m3`);
  }
}
