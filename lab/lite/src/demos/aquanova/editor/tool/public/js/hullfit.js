/**
 * Fit collision boxes to a mesh.
 *
 * Not a learner: the problem has a cheap objective - how much of the mesh does
 * the hull cover, and how much of the hull sits on real surface - so it is
 * searched rather than trained. The 53 hand-authored hulls are far too few to
 * learn from, and the geometry fully determines a good answer anyway.
 *
 * Three candidates are fitted and the best-scoring one wins:
 *
 *   box    the whole module in one box - a crate, a floor plate, a door
 *   slabs  a slab per surface patch, laid on it and extending backwards -
 *          a wall with a lip. Triangle normals point out of the solid, so
 *          "backwards" is away from the play space for free
 *   split  cut in half and recurse until each piece is worth boxing -
 *          a corner, a door frame
 *
 * Letting the score choose means no single heuristic has to be right about
 * which kind of module it is looking at. Measured against the hand-authored
 * hulls it covers more (99.7% vs 88.4%) in less volume (5.3 vs 7.4 m3) with
 * the same number of boxes; see fit-boxes.mjs for that harness.
 *
 * Rotations come out as basis vectors, not Euler angles: converting to
 * Babylon's YXZ convention is easy to get subtly wrong, and the caller
 * already has proven code for it.
 */

// ---------------------------------------------------------------- vector bits
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** An orthonormal frame with `n` as its third axis and the second as upright as it can be. */
function frameFrom(n) {
  const z = norm(n);
  const up = Math.abs(z[1]) > 0.95 ? [0, 0, 1] : [0, 1, 0];
  const x = norm(cross(up, z));
  const y = norm(cross(z, x));
  return [x, y, z];
}

/** Is a point inside this box? */
function inside(box, p) {
  const d = sub(p, box.centre);
  for (let k = 0; k < 3; k++) {
    if (Math.abs(dot(d, box.basis[k])) > box.half[k] + 1e-4) return false;
  }
  return true;
}

// ------------------------------------------------------------------ the fitter
/** Lay a slab on each surface patch, biggest patch first. */
function fitSlabs(tris, opts = {}) {
  const {
    binDeg = 28,          // how far a triangle's normal may be from its bin's
    thickness = 0.35,     // how far a slab reaches behind its surface
    maxDepth = 0.6,       // ...and how far it may reach at the very most
    minAreaFrac = 0.04,   // ignore a patch smaller than this share of the total
    maxBoxes = 10,
    target = 0.97,        // stop once this much of the mesh is covered
    gainFloor = 0.015,    // ...or once a box adds less than this
  } = opts;

  const faces = [];
  let totalArea = 0;
  for (const [a, b, c] of tris) {
    const n = cross(sub(b, a), sub(c, a));
    const area = len(n) / 2;
    if (area < 1e-9) continue;
    faces.push({ n: norm(n), area, pts: [a, b, c] });
    totalArea += area;
  }
  if (!faces.length) return { boxes: [], coverage: 0 };

  // bin the normals, biggest patches first so a bin is seeded by real surface
  faces.sort((p, q) => q.area - p.area);
  const cosBin = Math.cos((binDeg * Math.PI) / 180);
  const bins = [];
  for (const f of faces) {
    let home = null;
    for (const bin of bins) if (dot(bin.n, f.n) >= cosBin) { home = bin; break; }
    if (home) {
      home.faces.push(f);
      home.area += f.area;
      home.n = norm(add(mul(home.n, 1 - f.area / home.area), mul(f.n, f.area / home.area)));
    } else {
      bins.push({ n: f.n, area: f.area, faces: [f] });
    }
  }
  bins.sort((p, q) => q.area - p.area);

  // every vertex, as the thing to be covered
  const verts = [];
  for (const [a, b, c] of tris) verts.push(a, b, c);

  const boxes = [];
  const covered = new Uint8Array(verts.length);
  let coveredN = 0;
  const coverage = () => coveredN / verts.length;

  for (const bin of bins) {
    if (boxes.length >= maxBoxes) break;
    if (bin.area / totalArea < minAreaFrac) break;
    if (coverage() >= target) break;

    const basis = frameFrom(bin.n);
    const pts = bin.faces.flatMap((f) => f.pts);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) {
      for (let k = 0; k < 3; k++) {
        const v = dot(p, basis[k]);
        if (v < lo[k]) lo[k] = v;
        if (v > hi[k]) hi[k] = v;
      }
    }
    // in-plane the slab spans the patch; along the normal it reaches back a
    // bounded thickness. Using the patch's own extent instead would let a
    // curved patch — whose points wrap right around an arc — grow a slab that
    // swallows everything the arc encloses.
    const depth = Math.min(Math.max(thickness, hi[2] - lo[2]), maxDepth);
    const half = [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, depth / 2];
    const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] - half[2]];
    const centre = add(add(mul(basis[0], mid[0]), mul(basis[1], mid[1])), mul(basis[2], mid[2]));
    const box = { centre, half, basis, area: bin.area };

    // does it earn its place?
    let gained = 0;
    for (let i = 0; i < verts.length; i++) if (!covered[i] && inside(box, verts[i])) gained++;
    if (gained / verts.length < gainFloor && boxes.length) continue;
    for (let i = 0; i < verts.length; i++) {
      if (!covered[i] && inside(box, verts[i])) { covered[i] = 1; coveredN++; }
    }
    boxes.push(box);
  }

  return { boxes, bins: bins.length };
}

/** A voxel set of the mesh surface, so solidity can be asked cheaply and often. */
export function surfaceIndex(tris, cell = 0.2) {
  const set = new Set();
  const key = (x, y, z) => `${Math.round(x / cell)},${Math.round(y / cell)},${Math.round(z / cell)}`;
  for (const [a, b, c] of tris) {
    for (let u = 0; u <= 1; u += 0.25) {
      for (let v = 0; u + v <= 1; v += 0.25) {
        const p = [0, 1, 2].map((k) => a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]));
        set.add(key(p[0], p[1], p[2]));
      }
    }
  }
  const near = (p) => {
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          if (set.has(key(p[0] + dx * cell, p[1] + dy * cell, p[2] + dz * cell))) return true;
        }
      }
    }
    return false;
  };
  return { cell, near };
}

/**
 * How much of a hull's volume actually hugs the mesh.
 *
 * Coverage alone is a bad judge: one enormous box covers every vertex and
 * fills the room the player is meant to walk in. This samples the hull and
 * asks what share of it lies near real surface, so a slab laid on a wall
 * scores high and a box swallowing a corner scores low.
 */
export function solidFraction(idx, boxes) {
  if (!boxes.length) return 0;
  const cell = idx.cell;
  let hit = 0, total = 0;
  for (const bx of boxes) {
    const n = bx.half.map((h) => Math.max(1, Math.round((2 * h) / cell)));
    for (let i = 0; i < n[0]; i++) {
      for (let j = 0; j < n[1]; j++) {
        for (let k = 0; k < n[2]; k++) {
          const local = [(i + 0.5) / n[0] - 0.5, (j + 0.5) / n[1] - 0.5, (k + 0.5) / n[2] - 0.5];
          const p = [0, 1, 2].map((a) => bx.centre[a]
            + 2 * bx.half[0] * local[0] * bx.basis[0][a]
            + 2 * bx.half[1] * local[1] * bx.basis[1][a]
            + 2 * bx.half[2] * local[2] * bx.basis[2][a]);
          total++;
          if (idx.near(p)) hit++;
        }
      }
    }
  }
  return total ? +(hit / total).toFixed(4) : 0;
}

/** The whole module in one axis-aligned box — the answer 42 of 53 hulls use. */
function oneBox(tris) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const t of tris) {
    for (const p of t) {
      for (let k = 0; k < 3; k++) {
        if (p[k] < lo[k]) lo[k] = p[k];
        if (p[k] > hi[k]) hi[k] = p[k];
      }
    }
  }
  return {
    centre: [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2),
    half: [0, 1, 2].map((k) => Math.max((hi[k] - lo[k]) / 2, 0.01)),
    basis: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  };
}

/**
 * Cut the module up until each piece is worth boxing.
 *
 * An L-shaped corner suits neither one box (which fills the elbow) nor a slab
 * per normal (a curved face has no single normal). Halving the bounding box
 * along its longest side and boxing each half separately handles both, and it
 * stops as soon as a piece is mostly solid, so a flat plate is never split.
 */
function fitSplit(tris, idx, opts = {}) {
  const { splitDepth = 4, splitSolid = 0.75, minPiece = 0.15 } = opts;

  const extents = (ts, basis) => {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const t of ts) {
      for (const p of t) {
        for (let k = 0; k < 3; k++) {
          const v = dot(p, basis[k]);
          if (v < lo[k]) lo[k] = v;
          if (v > hi[k]) hi[k] = v;
        }
      }
    }
    const half = [0, 1, 2].map((k) => Math.max((hi[k] - lo[k]) / 2, 0.01));
    const mid = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
    const centre = add(add(mul(basis[0], mid[0]), mul(basis[1], mid[1])), mul(basis[2], mid[2]));
    return { centre, half, basis, vol: 8 * half[0] * half[1] * half[2] };
  };

  const AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  /**
   * The tightest box round a piece.
   *
   * An axis-aligned box is right for most of the kit, but a curved corner is
   * cut into pieces that lie at an angle, and boxing those squarely leaves a
   * staircase of oversized blocks. Turning the box to face the piece's own
   * dominant surface costs one extra fit and shrinks it a great deal.
   */
  const boxOf = (ts) => {
    const square = extents(ts, AXES);
    let n = [0, 0, 0];
    for (const [a, b, c] of ts) {
      const f = cross(sub(b, a), sub(c, a));
      // face the same way as the first normal, so opposite faces don't cancel
      n = add(n, dot(f, n) < 0 ? mul(f, -1) : f);
    }
    if (len(n) < 1e-9) return square;
    const turned = extents(ts, frameFrom(norm(n)));
    return turned.vol < square.vol * 0.9 ? turned : square;
  };

  const out = [];
  const walk = (ts, depth) => {
    if (!ts.length) return;
    const bx = boxOf(ts);
    const longest = bx.half.indexOf(Math.max(...bx.half));
    if (depth >= splitDepth || 2 * bx.half[longest] < 2 * minPiece
        || solidFraction(idx, [bx]) >= splitSolid) {
      out.push(bx);
      return;
    }
    // cut across the box's own long axis, and where the material is rather
    // than at the middle of empty space
    const axis = bx.basis[longest];
    const at = (t) => (dot(t[0], axis) + dot(t[1], axis) + dot(t[2], axis)) / 3;
    const cut = ts.map(at).sort((a, b) => a - b)[Math.floor(ts.length / 2)];
    const lo = [], hi = [];
    for (const t of ts) (at(t) <= cut ? lo : hi).push(t);
    if (!lo.length || !hi.length) { out.push(bx); return; }
    walk(lo, depth + 1);
    walk(hi, depth + 1);
  };
  walk(tris, 0);
  return out;
}

/**
 * Pick between the candidates.
 *
 * A hull is good when it covers the mesh AND stays near it; and between two
 * that do, the one with fewer boxes is the one a person would have drawn.
 */
function score(tris, idx, boxes) {
  if (!boxes.length) return null;
  const verts = [];
  for (const [a, b, c] of tris) verts.push(a, b, c);
  let n = 0;
  for (const p of verts) if (boxes.some((bx) => inside(bx, p))) n++;
  const cov = n / verts.length;
  const solid = solidFraction(idx, boxes);
  return { cov, solid, value: cov * solid - 0.005 * boxes.length };
}


/**
 * Fit a hull to a module.
 *
 * Three candidates are offered and the best one wins: the whole thing in a
 * single box, a slab per surface patch, and a recursive split. The single box
 * is what a person draws for a crate, a floor plate or a door; the slabs are
 * what they draw for a wall with a lip; the split is what they draw for a
 * corner. Letting the score choose means no heuristic has to be right about
 * which kind of module it is looking at.
 */
export function fit(tris, opts = {}) {
  const idx = surfaceIndex(tris);
  const slabs = fitSlabs(tris, opts);
  const candidates = [
    { name: "box", boxes: [oneBox(tris)], bins: 1 },
    { name: "slabs", boxes: slabs.boxes, bins: slabs.bins },
    { name: "split", boxes: fitSplit(tris, idx, opts), bins: 0 },
  ];
  let best = null;
  for (const c of candidates) {
    const s = score(tris, idx, c.boxes);
    if (!s) continue;
    Object.assign(c, s);
    if (!best || s.value > best.value) best = c;
  }
  if (!best) return { boxes: [], coverage: 0, solid: 0, how: "none", bins: 0, confident: false };
  return {
    boxes: best.boxes.map((b) => ({
      centre: b.centre.map((v) => +v.toFixed(4)),
      half: b.half.map((v) => +v.toFixed(4)),
      basis: b.basis.map((r) => r.map((v) => +v.toFixed(6))),
    })),
    coverage: +best.cov.toFixed(4),
    solid: +best.solid.toFixed(4),
    how: best.name,
    bins: best.bins,
    // A hull that covers the mesh, hugs it, and did not need a pile of boxes
    // can go straight in. Anything else — in practice the curved corners —
    // is worth a person's eye, and saying so is more use than a bad hull.
    confident: best.cov >= 0.97 && best.solid >= 0.8 && best.boxes.length <= 4,
  };
}


