// The `F` / `Shift+F` debug overlay: a voxel view of what the fluid ACTUALLY collides with.
//
// The scene SDF is invisible, and a wrong sign looks exactly like a right one until the water
// misbehaves — which is how an inverted room shell went unnoticed while it shoved the fluid down
// through its own floor. This draws the field instead of trusting it.

import { addToScene, createBox, createStandardMaterial, type Mesh, type SceneContext, type EngineContext } from "babylon-lite";
import { flushThinInstances, setThinInstanceCount, setThinInstances } from "babylon-lite/mesh/thin-instance.js";
import { sampleBakedSdf, SDF_OUTSIDE, type BakedBody, type BakedSdf } from "../sdf-bake.js";

/** One dynamic body as the overlay needs to see it: its baked grid plus its live Havok pose. */
export interface SdfOverlayDynBody {
    baked: BakedBody;
    position: { x: number; y: number; z: number };
    rotationQuaternion: { x: number; y: number; z: number; w: number };
}

export interface SdfOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    /** Cell size of the room bake (m) — the lattice samples every other cell. */
    roomCell: number;
    /** Cell size of a dynamic-body bake (m). */
    dynCell: number;
    /** Baked room/ship bodies, drawn GREEN. Every one is drawn: `bodiesSdf` unions them all, so the
     *  fluid collides with the lot regardless of which chunk you are standing in. */
    rooms: () => Iterable<{ grid: BakedSdf; centre: [number, number, number] }>;
    /** Live dynamic bodies, drawn ORANGE. */
    dynBodies: () => Iterable<SdfOverlayDynBody>;
    /** Current room id, for the console readout only. */
    roomAt: () => string;
}

export interface SdfOverlay {
    /** Set the mode outright: 0 off, 1 surface (zero-crossing), 2 solid (sdf < 0). */
    setMode(want: number): void;
    /** Toggle `want` against the current mode, so `F` and `Shift+F` each switch their own view off. */
    toggle(want: number): void;
    /** Per frame: re-voxelise the dynamic bodies if any of them actually moved. */
    onFrame(): void;
    state(): Record<string, unknown>;
}

export function createSdfOverlay(opts: SdfOverlayOptions): SdfOverlay {
    const { engine, scene, canvas, roomCell, dynCell, rooms, dynBodies, roomAt } = opts;
// ── SDF debug overlay (F / Shift+F) ─────────────────────────────────────────────────────────
// Voxelises what the fluid's `sceneSdf` actually returns. GREEN = the room body, ORANGE = a dynamic
// body.
//
//   F        SURFACE  — a cube on every zero-crossing of the field: the collision skin the fluid
//                       actually slides along, exactly one cell thick whatever the alignment.
//   Shift+F  SOLID    — a cube wherever sdf < 0, the volume the solver refuses to admit fluid into.
//                       Signed, so this is the view that catches an inside-out body: it must fill
//                       the walls, floor slab and props and leave the air empty.
//
// Keying off the DISTANCE rather than neighbour topology matters. The first version marked any solid
// cell touching a non-solid one, and since the lattice spans the whole grid box — whose outer layer
// is solid, being outside the room — cells at the box edge counted as exposed and emitted a complete
// room-sized shell around the player. That is what washed the screen green.
//
// The baked field is STATIC, so the overlay must be too: the lattice is snapped to a fixed world grid
// and spans the whole room, never the camera or the player. Anchoring the samples to the player (an
// earlier version) moved the lattice PHASE with him, so every rebuild put the voxels at different
// world positions and the overlay appeared to change as you walked — for a field that had not moved.
const SDF_VOX_STEP = roomCell * 2; // sample every other room cell
const SDF_VOX_MAX = 90000; // instance cap per layer
const DYN_VOX_STEP = dynCell * 2; // dynamic bodies get their own bake's resolution — they are small

const makeSdfLayer = (r: number, g: number, b: number): Mesh => {
    const mat = createStandardMaterial();
    mat.disableLighting = true;
    mat.diffuseColor = [1, 1, 1];
    mat.emissiveColor = [r, g, b];
    mat.alpha = 0.25;
    // Unlike the collider overlay's room-sized boxes (which you stand inside, so their back faces
    // are the only thing you would see), these are small closed cubes — drawing back faces just
    // doubles the overdraw and makes a wall of them twice as opaque.
    mat.backFaceCulling = true;
    const m = createBox(engine, 1);
    m.material = mat;
    m.pickable = false;
    // Thin instances must exist BEFORE registerScene so the pipeline is built with the instanced
    // path; the overlay then just varies the live count.
    setThinInstances(m, new Float32Array(SDF_VOX_MAX * 16), SDF_VOX_MAX);
    setThinInstanceCount(m, 0);
    addToScene(scene, m);
    return m;
};
const sdfRoomVox = makeSdfLayer(0.15, 0.95, 0.35);
const sdfDynVox = makeSdfLayer(1.3, 0.45, 0.08);
let sdfMode = 0; // 0 = off, 1 = surface (zero-crossing), 2 = solid (sdf < 0)

/**
 * Fill `mesh` with one voxel per accepted cell of a snapped world lattice, starting at instance `n`.
 *
 * The lattice origin is a global multiple of `step`, so a given world cell is always sampled at the
 * same point no matter what triggered the rebuild — that is what makes the overlay stand still.
 *
 * `sample` returns the body's RAW baked value; `invert` is its `invertSdf` flag, applied here so
 * both modes reason about the field the shader ends up with rather than the raw bake. Values at or
 * above {@link SDF_OUTSIDE} mean the point is outside this body's grid box, where the shader skips
 * the body entirely — it has neither surface nor solid there, and a cell next to one cannot be a
 * crossing either.
 *
 * SURFACE mode marks a cell whose sign DIFFERS from its +X/+Y/+Z neighbour — the isosurface passes
 * between them — keeping whichever of the pair is nearer the surface. A zero-crossing is used rather
 * than a `|sdf| < eps` threshold because a threshold is an aliasing trap: with eps at half a step, a
 * plane lying on a lattice plane produces ONE layer but a plane halfway between produces TWO, so
 * flat surfaces came out as doubled or offset rows depending on where they happened to fall. A
 * crossing is exactly one cell per surface, whatever the alignment, with nothing to tune.
 */
const emitCells = (mesh: Mesh, n: number, lo: readonly number[], hi: readonly number[], step: number, sample: (x: number, y: number, z: number) => number, invert: boolean): number => {
    const o0 = [Math.floor(lo[0]! / step) * step, Math.floor(lo[1]! / step) * step, Math.floor(lo[2]! / step) * step];
    const nx = Math.ceil((hi[0]! - o0[0]!) / step) + 1, ny = Math.ceil((hi[1]! - o0[1]!) / step) + 1, nz = Math.ceil((hi[2]! - o0[2]!) / step) + 1;
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx * ny * nz > 4e6) return n;
    const at = (i: number, j: number, k: number): number => i + nx * (j + ny * k);
    const f = new Float32Array(nx * ny * nz);
    for (let i = 0; i < nx; i++)
        for (let j = 0; j < ny; j++)
            for (let k = 0; k < nz; k++) {
                const v = sample(o0[0]! + i * step, o0[1]! + j * step, o0[2]! + k * step);
                f[at(i, j, k)] = v >= SDF_OUTSIDE ? SDF_OUTSIDE : invert ? -v : v;
            }
    const mark = new Uint8Array(nx * ny * nz);
    if (sdfMode === 1) {
        const cross = (a: number, b: number): void => {
            const va = f[a]!, vb = f[b]!;
            if (va >= SDF_OUTSIDE || vb >= SDF_OUTSIDE || va < 0 === vb < 0) return;
            mark[Math.abs(va) <= Math.abs(vb) ? a : b] = 1;
        };
        for (let i = 0; i < nx; i++)
            for (let j = 0; j < ny; j++)
                for (let k = 0; k < nz; k++) {
                    const c = at(i, j, k);
                    if (i + 1 < nx) cross(c, at(i + 1, j, k));
                    if (j + 1 < ny) cross(c, at(i, j + 1, k));
                    if (k + 1 < nz) cross(c, at(i, j, k + 1));
                }
    } else {
        for (let c = 0; c < f.length; c++) if (f[c]! < 0) mark[c] = 1;
    }
    const mats = mesh.thinInstances!.matrices as Float32Array;
    const s = step * 0.9;
    for (let i = 0; i < nx && n < SDF_VOX_MAX; i++)
        for (let j = 0; j < ny && n < SDF_VOX_MAX; j++)
            for (let k = 0; k < nz && n < SDF_VOX_MAX; k++) {
                if (!mark[at(i, j, k)]) continue;
                const o = n * 16;
                mats.fill(0, o, o + 16);
                mats[o] = s;
                mats[o + 5] = s;
                mats[o + 10] = s;
                mats[o + 12] = o0[0]! + i * step;
                mats[o + 13] = o0[1]! + j * step;
                mats[o + 14] = o0[2]! + k * step;
                mats[o + 15] = 1;
                n++;
            }
    return n;
};

/** Dynamic-body voxels, split out because these bodies genuinely MOVE — unlike the room, whose
 *  voxels must stay put. A lattice PER BODY over its own AABB at the finer dynamic bake resolution:
 *  sampling them on the room lattice missed them (a 0.1 m door panel almost never lands on a 0.36 m
 *  grid), and a fine lattice over the whole room would be 8× the samples for nothing. */
let sdfDynPoseKey = "";
const refreshSdfDynVoxels = (force: boolean): void => {
    // Every dynamic body, not just the current room's: `bodiesSdf` unions them all, so the fluid
    // collides with the lot regardless of which chunk you happen to be standing in.
    const live = sdfMode === 0 ? [] : [...dynBodies()];
    // Resting crates never move, so re-voxelising every frame would be pure waste.
    const key = `${sdfMode}|${live.map((d) => `${d.position.x.toFixed(2)},${d.position.y.toFixed(2)},${d.position.z.toFixed(2)},${d.rotationQuaternion.w.toFixed(3)}`).join("|")}`;
    if (!force && key === sdfDynPoseKey) return;
    sdfDynPoseKey = key;
    let n = 0;
    for (const d of live) {
                const bk = d.baked;
            const pos = d.position;
        const q = d.rotationQuaternion;
        const quat: [number, number, number, number] = [q.x, q.y, q.z, q.w];
        const m = DYN_VOX_STEP;
        n = emitCells(
            sdfDynVox,
            n,
            [pos.x - bk.half[0] - m, pos.y - bk.half[1] - m, pos.z - bk.half[2] - m],
            [pos.x + bk.half[0] + m, pos.y + bk.half[1] + m, pos.z + bk.half[2] + m],
            DYN_VOX_STEP,
            (x, y, z) => sampleBakedSdf(bk.grid, [pos.x, pos.y, pos.z], x, y, z, quat),
            false
        );
    }
    setThinInstanceCount(sdfDynVox, n);
    flushThinInstances(sdfDynVox);
    canvas.dataset.sdfDynVoxels = String(n);
};

const refreshSdfVoxels = (): void => {
    // EVERY room body, not just the one you are standing in. A chunk boundary is a bookkeeping line,
    // not a physical one: the storage room's right-hand half is built from meshes whose centres fall
    // in the neighbouring chunk's AABB, so drawing only the current room cut the overlay in half at
    // the doorway while the fluid — which unions every body in `bodiesSdf` — collided with both.
    let nRoom = 0;
    if (sdfMode !== 0)
        for (const { grid: g, centre: c } of rooms()) {
            const lo = [c[0] + g.origin[0], c[1] + g.origin[1], c[2] + g.origin[2]];
            nRoom = emitCells(
                sdfRoomVox,
                nRoom,
                lo,
                [lo[0]! + g.dims[0] * g.cellSize, lo[1]! + g.dims[1] * g.cellSize, lo[2]! + g.dims[2] * g.cellSize],
                SDF_VOX_STEP,
                (x, y, z) => sampleBakedSdf(g, c, x, y, z),
                true
            );
        }
    setThinInstanceCount(sdfRoomVox, nRoom);
    flushThinInstances(sdfRoomVox);
    refreshSdfDynVoxels(true);
    canvas.dataset.sdfVoxels = `${nRoom}/${canvas.dataset.sdfDynVoxels ?? 0}`;
};

/** `want` is the mode the pressed key stands for; pressing it again turns the overlay off, so F and
 *  Shift+F each toggle their own view and switching between them is a single press. */
const setSdfMode = (want: number): void => {
    sdfMode = ((want % 3) + 3) % 3;
    refreshSdfVoxels();
    canvas.dataset.sdfOverlay = ["off", "surface", "solid"][sdfMode]!;
    // eslint-disable-next-line no-console
    console.log(
        `[aquanova] SDF overlay: ${["off", "surface (zero-crossing)", "solid sdf < 0"][sdfMode]}` +
            (sdfMode ? ` — room ${roomAt()}, ${canvas.dataset.sdfVoxels} voxels (green = room, orange = dynamic), lattice ${SDF_VOX_STEP.toFixed(2)} m` : "")
    );
};

    return {
        setMode: setSdfMode,
        toggle: (want: number) => setSdfMode(sdfMode === want ? 0 : want),
        onFrame: () => {
            // The room voxels cover EVERY baked room and are world-anchored, so neither walking around
            // nor crossing a chunk boundary changes them. Only the dynamic bodies actually move.
            if (sdfMode > 0) {
                refreshSdfDynVoxels(false);
            }
        },
        state: () => {
            const box = (m: Mesh): number[] | null => {
                const ti = m.thinInstances;
                if (!ti?.count) {
                    return null;
                }
                const v = ti.matrices as Float32Array;
                const a = [1e9, 1e9, 1e9];
                const b = [-1e9, -1e9, -1e9];
                for (let i = 0; i < ti.count; i++) {
                    for (let k = 0; k < 3; k++) {
                        a[k] = Math.min(a[k]!, v[i * 16 + 12 + k]!);
                        b[k] = Math.max(b[k]!, v[i * 16 + 12 + k]!);
                    }
                }
                return [...a, ...b];
            };
            return {
                mode: sdfMode,
                room: roomAt(),
                roomVoxels: sdfRoomVox.thinInstances?.count ?? 0,
                dynVoxels: sdfDynVox.thinInstances?.count ?? 0,
                roomBox: box(sdfRoomVox),
                dynBox: box(sdfDynVox),
                step: SDF_VOX_STEP,
            };
        },
    };
}