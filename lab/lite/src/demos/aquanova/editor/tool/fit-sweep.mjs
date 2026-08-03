/**
 * Sweep the tolerance and the offset, to see what each setting actually does.
 *
 * The point of the tolerance is that it should be the *only* knob for how
 * finely a rounded shape is approximated, so what matters here is that box
 * counts rise monotonically as it tightens - and that flat things stay at one.
 */
import fs from "node:fs";
import { fit } from "./public/js/hullfit.js";

const IN = process.env.FIT_IN || "D:/alexis/TombRaider/Popov72/fitter/modules.json";
const data = JSON.parse(fs.readFileSync(IN, "utf8"));

const ROUND = ["Props/Prop_Barrel_Small", "Props/Prop_Pipe_Thick_Straight",
  "Platforms/Platform_Round1", "Walls/WallAstra_Corner_Round_Outer",
  "Walls/WallAstra_Corner_Round_Inner", "Walls/TopPadded_Incline_Curve_Round_Inner"];
const FLAT = ["Props/Prop_Crate1", "Platforms/Platform_Metal", "Walls/WallWideBand_Straight",
  "Platforms/Door_Metal"];
const TOL = [0.25, 0.15, 0.1, 0.06, 0.04, 0.025];
const THICK = Number(process.env.FIT_THICK || 0.35);

const show = (title, names) => {
  console.log(`\n${title}`);
  console.log("module".padEnd(40) + TOL.map((t) => `${t}m`.padStart(13)).join(""));
  for (const m of names) {
    const tris = data.modules[m];
    if (!tris) { console.log(`${m.padEnd(40)} (not dumped)`); continue; }
    const cells = TOL.map((tolerance) => {
      const t0 = Date.now();
      const r = fit(tris, { tolerance, thickness: THICK });
      const tag = { box: "b", slabs: "L", split: "X" }[r.how] || "?";
      const vol = r.boxes.reduce((s, b) => s + 8 * b.half[0] * b.half[1] * b.half[2], 0);
      return `${r.boxes.length}${tag} ${vol.toFixed(1)}m3 ${Date.now() - t0}ms`.padStart(13);
    });
    console.log(m.padEnd(40) + cells.join(""));
  }
};

show("rounded — box count should climb as the tolerance tightens", ROUND);
show("flat — should stay at one box whatever the tolerance", FLAT);

// the offset must keep the art inside the hull, and put it on the near face
console.log("\noffset — where the art sits inside its hull (thickness 0.35 m)");
console.log("module".padEnd(44) + "centered".padStart(22) + "negative".padStart(22)
  + "positive".padStart(22));
for (const m of ["Walls/WallWideBand_Straight", "Platforms/Platform_Metal",
  "Decals/Decal_Caution", "Props/Prop_Crate1"]) {
  const tris = data.modules[m];
  if (!tris) continue;
  const cells = ["centered", "negative", "positive"].map((offset) => {
    const r = fit(tris, { tolerance: 0.1, thickness: 0.35, offset });
    // measure the deepest the art pokes out of its hull, and the clearance
    // between the art and each face along the hull's thin axis
    let worst = -Infinity, front = Infinity, back = Infinity;
    for (const b of r.boxes) {
      const thin = b.half.indexOf(Math.min(...b.half));
      const ax = b.basis[thin];
      for (const t of tris) {
        for (const p of t) {
          const d = (p[0] - b.centre[0]) * ax[0] + (p[1] - b.centre[1]) * ax[1]
            + (p[2] - b.centre[2]) * ax[2];
          if (r.boxes.length === 1) {
            worst = Math.max(worst, Math.abs(d) - b.half[thin]);
            front = Math.min(front, b.half[thin] - d);
            back = Math.min(back, d + b.half[thin]);
          }
        }
      }
    }
    if (r.boxes.length !== 1) return "(multi)".padStart(22);
    return `out ${worst.toFixed(3)} gap ${back.toFixed(2)}/${front.toFixed(2)}`.padStart(22);
  });
  console.log(m.padEnd(44) + cells.join(""));
}
console.log("\n'out' is how far the art escapes its hull — must be negative.");
console.log("'gap a/b' is the clearance to the near and far face along the thin axis.");
