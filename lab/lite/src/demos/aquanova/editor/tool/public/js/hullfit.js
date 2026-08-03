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
/**
 * Lay a slab on each surface patch, biggest patch first.
 *
 * The bin angle decides how finely a curve is broken up, so it is *derived*
 * from the tolerance rather than fixed: a patch is only allowed to be as
 * curved as the tolerance permits. Leaving it at a constant was what stopped
 * a tighter tolerance from ever improving a rounded corner - the split pass
 * would produce more boxes, but the slab pass kept winning with the same
 * coarse three.
 */
function fitSlabs(tris, opts = {}) {
  const {
    tolerance = 0.1,      // how far a slab may stray from its own surface
    thickness = 0.35,     // how far a slab reaches behind its surface
    maxDepth = 0.6,       // ...and how far it may reach at the very most
    minAreaFrac = 0.04,   // ignore a patch smaller than this share of the total
    maxBoxes = 32,
    target = 0.995,       // stop once this much of the mesh is covered
    gainFloor = 0.004,    // ...or once a box adds less than this
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
  const binAt = (deg) => {
    const cosBin = Math.cos((deg * Math.PI) / 180);
    const out = [];
    for (const f of faces) {
      let home = null;
      for (const bin of out) if (dot(bin.n, f.n) >= cosBin) { home = bin; break; }
      if (home) {
        home.faces.push(f);
        home.area += f.area;
        home.n = norm(add(mul(home.n, 1 - f.area / home.area), mul(f.n, f.area / home.area)));
      } else {
        out.push({ n: f.n, area: f.area, faces: [f] });
      }
    }
    out.sort((p, q) => q.area - p.area);
    return out;
  };

  /**
   * How far a patch strays from the plane its slab will lie on.
   *
   * For a flat wall this is nothing; for an arc it is the sagitta, which is
   * exactly the distance the slab ends up jutting into the room. Only patches
   * worth a slab are measured - a sliver of trim should not force the whole
   * module to be cut finely.
   */
  const sag = (bins) => {
    let worst = 0;
    for (const bin of bins) {
      if (bin.area / totalArea < minAreaFrac) continue;
      let lo = Infinity, hi = -Infinity;
      for (const f of bin.faces) {
        for (const p of f.pts) {
          const v = dot(p, bin.n);
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
      worst = Math.max(worst, hi - lo);
    }
    return worst;
  };

  let bins = binAt(28);
  for (let deg = 14; deg >= 1.5 && sag(bins) > tolerance; deg /= 2) {
    const finer = binAt(deg);
    if (finer.length > maxBoxes * 2) break;
    bins = finer;
  }
  // the floor on a patch's size has to give way as the bins get finer, or
  // cutting a curve into more pieces just pushes every piece under it and the
  // slab count stops responding to the tolerance at all
  const areaFloor = Math.min(minAreaFrac, 0.7 / Math.max(1, bins.length));

  // every vertex, as the thing to be covered
  const verts = [];
  for (const [a, b, c] of tris) verts.push(a, b, c);

  const boxes = [];
  const covered = new Uint8Array(verts.length);
  let coveredN = 0;
  const coverage = () => coveredN / verts.length;

  for (const bin of bins) {
    if (boxes.length >= maxBoxes) break;
    if (bin.area / totalArea < areaFloor) break;
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

/**
 * Where the solid is: inside the mesh, by ray parity along X.
 *
 * Surface proximity alone is not enough to judge a hull. A crate is a hollow
 * shell, so every point in the middle of it is far from any triangle — and a
 * hull that fills the crate would be marked as mostly empty air, which is the
 * opposite of the truth. What matters is whether a point is *in the solid*.
 *
 * Crossings along each grid line are paired in order, and an odd one left
 * over is dropped. The kit is full of open shapes — a floor is a single plane
 * with no inside at all — and plain parity would mark everything beyond such
 * a plane as solid. Pairing means an unclosed surface simply contributes
 * nothing, while a wall's 8 mm shell and a crate's interior both come out
 * right.
 */
function insideTest(tris, cell) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const t of tris) {
    for (const p of t) {
      for (let k = 0; k < 3; k++) {
        if (p[k] < lo[k]) lo[k] = p[k];
        if (p[k] > hi[k]) hi[k] = p[k];
      }
    }
  }
  if (!Number.isFinite(lo[0])) return () => false;
  const ny = Math.max(1, Math.ceil((hi[1] - lo[1]) / cell));
  const nz = Math.max(1, Math.ceil((hi[2] - lo[2]) / cell));
  if (ny * nz > 4e5) return () => false;   // too fine to be worth the memory
  const at = (iy, iz) => iy * nz + iz;
  const hits = new Map();

  for (const [a, b, c] of tris) {
    // project along X; the ray crosses where the point is inside the triangle
    const area = (b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2]);
    if (Math.abs(area) < 1e-12) continue;
    const yLo = Math.max(0, Math.floor((Math.min(a[1], b[1], c[1]) - lo[1]) / cell - 1));
    const yHi = Math.min(ny - 1, Math.ceil((Math.max(a[1], b[1], c[1]) - lo[1]) / cell));
    const zLo = Math.max(0, Math.floor((Math.min(a[2], b[2], c[2]) - lo[2]) / cell - 1));
    const zHi = Math.min(nz - 1, Math.ceil((Math.max(a[2], b[2], c[2]) - lo[2]) / cell));
    for (let iy = yLo; iy <= yHi; iy++) {
      const Y = lo[1] + (iy + 0.5) * cell;
      for (let iz = zLo; iz <= zHi; iz++) {
        const Z = lo[2] + (iz + 0.5) * cell;
        const w1 = ((Y - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (Z - a[2])) / area;
        if (w1 < 0 || w1 > 1) continue;
        const w2 = ((b[1] - a[1]) * (Z - a[2]) - (Y - a[1]) * (b[2] - a[2])) / area;
        if (w2 < 0 || w1 + w2 > 1) continue;
        const x = a[0] + w1 * (b[0] - a[0]) + w2 * (c[0] - a[0]);
        const k = at(iy, iz);
        const list = hits.get(k);
        if (list) list.push(x); else hits.set(k, [x]);
      }
    }
  }

  const spans = new Map();
  for (const [k, list] of hits) {
    list.sort((p, q) => p - q);
    // a shared edge is crossed once per triangle; left in, that breaks parity
    const uniq = list.filter((v, i) => i === 0 || v - list[i - 1] > 1e-6);
    const out = [];
    for (let i = 0; i + 1 < uniq.length; i += 2) out.push(uniq[i], uniq[i + 1]);
    if (out.length) spans.set(k, out);
  }

  return (p) => {
    const iy = Math.floor((p[1] - lo[1]) / cell);
    const iz = Math.floor((p[2] - lo[2]) / cell);
    if (iy < 0 || iy >= ny || iz < 0 || iz >= nz) return false;
    const s = spans.get(at(iy, iz));
    if (!s) return false;
    for (let i = 0; i < s.length; i += 2) {
      if (p[0] >= s[i] && p[0] <= s[i + 1]) return true;
    }
    return false;
  };
}

/**
 * A voxel set of the mesh surface, so solidity can be asked cheaply and often.
 *
 * The cell size is the tolerance: it is the distance at which the hull is
 * still considered to be on the art. Triangles are sampled finely enough for
 * that cell, so a single large triangle does not leave holes the hull can
 * appear to fall into.
 */
export function surfaceIndex(tris, cell = 0.2) {
  const set = new Set();
  const key = (x, y, z) => `${Math.round(x / cell)},${Math.round(y / cell)},${Math.round(z / cell)}`;
  for (const [a, b, c] of tris) {
    const span = Math.max(len(sub(b, a)), len(sub(c, a)), len(sub(c, b)));
    const n = Math.min(24, Math.max(2, Math.ceil(span / cell) + 1));
    const step = 1 / n;
    for (let u = 0; u <= 1 + 1e-9; u += step) {
      for (let v = 0; u + v <= 1 + 1e-9; v += step) {
        const p = [0, 1, 2].map((k) => a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]));
        set.add(key(p[0], p[1], p[2]));
      }
    }
  }
  const within = insideTest(tris, cell);
  const near = (p) => {
    if (within(p)) return true;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          if (set.has(key(p[0] + dx * cell, p[1] + dy * cell, p[2] + dz * cell))) return true;
        }
      }
    }
    return false;
  };
  return { cell, near, within };
}

/**
 * How much of a hull's volume is doing a job.
 *
 * Coverage alone is a bad judge: one enormous box covers every vertex and
 * fills the room the player is meant to walk in. This samples the hull and
 * asks what share of it is either inside the art or within the tolerance of
 * its surface, so a slab laid on a wall scores high, a box filling a crate
 * scores high, and a box swallowing the space beside a corner scores low.
 *
 * Because the tolerance is the cell size, tightening it is what makes a
 * rounded shape want more boxes: at 20 cm a single box round an arc looks
 * solid enough, at 3 cm most of it is plainly empty air.
 */
export function solidFraction(idx, boxes, budget = 20000) {
  if (!boxes.length) return 0;
  const cell = idx.cell;
  let hit = 0, total = 0;
  for (const bx of boxes) {
    let n = bx.half.map((h) => Math.max(1, Math.round((2 * h) / cell)));
    // a fine tolerance on a big box is a lot of samples; thin the grid evenly
    // rather than refusing, so the number stays comparable between candidates
    const want = n[0] * n[1] * n[2];
    if (want > budget) {
      const f = Math.cbrt(budget / want);
      n = n.map((v) => Math.max(1, Math.round(v * f)));
    }
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
 * Cut the module up, and offer every depth as a candidate.
 *
 * An L-shaped corner suits neither one box (which fills the elbow) nor a slab
 * per normal (a curved face has no single normal). Halving the bounding box
 * along its longest side and boxing each half separately handles both.
 *
 * How deep to go is deliberately *not* decided here. Cutting axis-aligned
 * means the parts nest inside their parent, so each extra level can only
 * shrink the hull — which makes "how deep" a question the score can answer
 * directly, by being handed the whole ladder. Stopping on a threshold instead
 * is what made a tighter tolerance sometimes return a bulkier hull than a
 * looser one: the threshold would happen to be met one cut too early.
 */
function fitSplit(tris, idx, opts = {}) {
  const {
    // a piece thinner than the tolerance cannot be improved by cutting it
    minPiece = Math.max(2 * idx.cell, 0.05),
    splitDepth = 6,
    maxBoxes = 24,
  } = opts;

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
   * The tightest box round a piece, axis aligned.
   *
   * Turning each piece to face its own surface makes a single piece smaller,
   * but neighbouring pieces then overlap — and a decomposition whose parts
   * overlap gets *bulkier* the more finely it is cut, which is the opposite
   * of what the tolerance is for. Axis-aligned cuts nest inside their parent,
   * so more cuts can only ever help. Angled surfaces are the slab pass's job.
   */
  const boxOf = (ts) => extents(ts, AXES);

  const nodes = [];
  const walk = (ts, depth) => {
    if (!ts.length) return;
    const bx = boxOf(ts);
    const longest = bx.half.indexOf(Math.max(...bx.half));
    if (depth >= splitDepth || 2 * bx.half[longest] < 2 * minPiece) {
      nodes.push({ box: bx, depth, leaf: true });
      return;
    }
    // cut across the box's own long axis, and where the material is rather
    // than at the middle of empty space
    const axis = bx.basis[longest];
    const at = (t) => (dot(t[0], axis) + dot(t[1], axis) + dot(t[2], axis)) / 3;
    const cut = ts.map(at).sort((a, b) => a - b)[Math.floor(ts.length / 2)];
    const lo = [], hi = [];
    for (const t of ts) (at(t) <= cut ? lo : hi).push(t);
    if (!lo.length || !hi.length) { nodes.push({ box: bx, depth, leaf: true }); return; }
    nodes.push({ box: bx, depth, leaf: false });
    walk(lo, depth + 1);
    walk(hi, depth + 1);
  };
  walk(tris, 0);

  const levels = [];
  for (let d = 0; d <= splitDepth; d++) {
    const set = nodes.filter((n) => n.depth === d || (n.leaf && n.depth < d)).map((n) => n.box);
    if (!set.length || set.length > maxBoxes) break;
    const prev = levels[levels.length - 1];
    if (prev && prev.length === set.length) continue;   // nothing new to offer
    levels.push(set);
  }
  return levels;
}

/**
 * Thicken a box to the minimum, and push it clear of the play space.
 *
 * A wall panel is 10 cm of art but wants a hull you cannot walk through, and
 * a hull centred on the panel juts as far into the room as it does behind it.
 * The ship's convention is the other way round: the hull's *face* sits on the
 * visible surface and its body extends away, so you stand exactly where the
 * art says you do.
 *
 * The push is along the box's thinnest axis, by half the thickness less half
 * the art's own, less a millimetre — which puts the near face just clear of
 * the visible surface. The axis is oriented to the surface normal, so it
 * points out of the solid: `negative` therefore tucks the hull behind the
 * art, which is the convention this ship uses, and `positive` stands it in
 * front.
 *
 * Subtracting the art's own half-thickness is what keeps the art strictly
 * inside the hull. Pushing by half the thickness alone would be right only
 * for something with no thickness of its own — a decal — and would leave a
 * 10 cm wall panel hanging half out of the back of its own collision.
 */
export function placeBox(box, tris, thickness = 0, offset = "centered") {
  const MARGIN = 0.001;
  const thin = box.half.indexOf(Math.min(...box.half));
  const axis0 = box.basis[thin];
  // The art's own extent along that axis, measured from the triangles this box
  // actually covers. Measuring over the whole module instead is a fine way to
  // make every box in a decomposition as deep as the module is wide.
  let lo = Infinity, hi = -Infinity;
  for (const t of tris) {
    const mid = [0, 1, 2].map((k) => (t[0][k] + t[1][k] + t[2][k]) / 3);
    if (!inside(box, mid)) continue;
    for (const p of t) {
      const v = dot(p, axis0);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const out = { centre: box.centre.slice(), half: box.half.slice(),
    basis: box.basis.map((r) => r.slice()) };
  const covers = Number.isFinite(lo);
  const art = covers ? Math.min(hi - lo, 2 * box.half[thin]) : 2 * box.half[thin];
  out.half[thin] = Math.max(Math.max(art, thickness), 2 * box.half[thin]) / 2;
  if (!covers || (offset !== "positive" && offset !== "negative")) return out;

  // point the thin axis out of the solid, so the two names mean the same
  // thing on every module rather than depending on how the box was fitted
  let n = [0, 0, 0];
  for (const [a, b, c] of tris) {
    const f = cross(sub(b, a), sub(c, a));
    n = dot(f, axis0) > 0 ? add(n, f) : sub(n, f);
  }
  const flip = dot(n, axis0) < 0;
  const axis = flip ? mul(axis0, -1) : axis0;
  const [near, far] = flip ? [-hi, -lo] : [lo, hi];

  // Place the face, do not nudge the box. A slab is fitted lying against its
  // own surface already, so pushing it "by the padding" pushes it a second
  // time and the art comes out of the back. Naming where the face goes is the
  // same answer whatever the box started as, and asking twice changes nothing.
  //
  // The clamp is what keeps the promise that the art stays inside: when the
  // hull is no thicker than the art there is no slack to give, and the
  // millimetre of clearance asked for at one face would have to come out of
  // the other.
  const half = out.half[thin];
  const wanted = offset === "positive" ? far + MARGIN - half : near - MARGIN + half;
  const at = Math.min(Math.max(wanted, far - half), near + half);
  out.centre = add(out.centre, mul(axis, at - dot(out.centre, axis)));
  return out;
}

/**
 * Pick between the candidates.
 *
 * Two questions in order. Is the hull good enough — does it contain the art,
 * and is it within the tolerance of it nearly everywhere? And of the ones
 * that are, which is the smallest? Volume is the honest measure of a
 * collision hull: every cubic metre of it that is not art is somewhere the
 * player cannot stand. The surcharge per box stops a pile of slivers winning
 * by a hair.
 *
 * Splitting it this way is what makes the tolerance a dial. Tightening it
 * fails the coarse hulls, so a finer one has to take over; and because the
 * winner among those is still the smallest, tightening can never hand back
 * something bulkier than the setting before it.
 */
const QUALIFY_COVERAGE = 0.95;
const QUALIFY_SOLID = 0.85;
const BOX_SURCHARGE = 0.02;

function score(tris, idx, raw, placed) {
  if (!raw.length) return null;
  const verts = [];
  for (const [a, b, c] of tris) verts.push(a, b, c);
  let n = 0;
  for (const p of verts) if (raw.some((bx) => inside(bx, p))) n++;
  const cov = n / verts.length;
  // Solidity judges the *shape* the fitter chose, so it is measured before the
  // thickness is added. Measuring after would have a thin floor plate padded
  // to a walkable depth fail its own test — most of that hull is deliberately
  // not near the art — and the fitter would chop the plate up trying to fix it.
  const solid = solidFraction(idx, raw, 8000);
  // Volume, on the other hand, is what the user pays for, padding and all.
  const vol = placed.reduce((s, b) => s + 8 * b.half[0] * b.half[1] * b.half[2], 0);
  return { cov, solid, vol,
    fits: cov >= QUALIFY_COVERAGE,
    ok: cov >= QUALIFY_COVERAGE && solid >= QUALIFY_SOLID,
    cost: vol * (1 + BOX_SURCHARGE * (placed.length - 1)) };
}

/** Better of two scored candidates: good enough first, then cheapest. */
function better(a, b) {
  if (!b) return true;
  if (a.fits !== b.fits) return a.fits;
  if (!a.fits) return a.cov > b.cov;
  if (a.ok !== b.ok) return a.ok;
  if (!a.ok) return a.solid > b.solid;
  return a.cost < b.cost;
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
 *
 * `tolerance` is how far the hull may stray from the art before that counts
 * as wrong, in metres. It is deliberately the *only* knob for how finely a
 * shape is approximated: it sets the resolution the score judges at, so a
 * rounded prop that is worth one box at 20 cm is worth eight at 3 cm, with
 * no separate "how many boxes" setting to keep in step.
 */
export function fit(tris, opts = {}) {
  const { tolerance = 0.1, thickness = 0, offset = "centered" } = opts;
  const idx = surfaceIndex(tris, Math.max(tolerance, 0.005));
  const slabs = fitSlabs(tris, opts);
  const candidates = [
    { name: "box", boxes: [oneBox(tris)], bins: 1 },
    { name: "slabs", boxes: slabs.boxes, bins: slabs.bins },
    ...fitSplit(tris, idx, opts).map((boxes) => ({ name: "split", boxes, bins: 0 })),
  ];
  let best = null;
  for (const c of candidates) {
    c.placed = c.boxes.map((b) => placeBox(b, tris, thickness, "centered"));
    const s = score(tris, idx, c.boxes, c.placed);
    if (!s) continue;
    Object.assign(c, s);
    if (better(c, best)) best = c;
  }
  if (!best) {
    return { boxes: [], coverage: 0, solid: 0, how: "none", bins: 0, confident: false };
  }
  // the offset is a deliberate displacement, applied only to the winner: it
  // would move every candidate the same way, and judging fit through it would
  // punish them all equally for no purpose
  const placed = offset === "centered" ? best.placed
    : best.boxes.map((b) => placeBox(b, tris, thickness, offset));
  return {
    boxes: placed.map((b) => ({
      centre: b.centre.map((v) => +v.toFixed(4)),
      half: b.half.map((v) => +v.toFixed(4)),
      basis: b.basis.map((r) => r.map((v) => +v.toFixed(6))),
    })),
    coverage: +best.cov.toFixed(4),
    solid: +best.solid.toFixed(4),
    volume: +best.vol.toFixed(3),
    how: best.name,
    bins: best.bins,
    tolerance,
    // A hull that covers the mesh, hugs it, and did not need a pile of boxes
    // can go straight in. Anything else — in practice the curved corners —
    // is worth a person's eye, and saying so is more use than a bad hull.
    confident: best.ok && best.boxes.length <= 4,
  };
}


