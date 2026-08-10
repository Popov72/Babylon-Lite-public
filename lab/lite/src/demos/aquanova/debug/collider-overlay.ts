// The `B` debug overlay: the ship's collision, as both the player and the fluid see it.
//
// Draws the AUTHORED manifest primitives in their true form (an OBB as an oriented box, a barrel as
// a capsule, the pod as a cylinder), plus every dissolvable prop's live rigid-body box and the
// per-chunk static shell.
//
// There is only ONE collision representation now: the same primitives feed Havok and the fluid.
// (Until recently the fluid collided against an SDF baked from the raw triangles, so water and the
// player genuinely saw different worlds and each needed its own overlay.)

import {
    addToScene,
    createBox,
    createCapsule,
    createCylinder,
    createSphere,
    createStandardMaterial,
    removeFromScene,
    type EngineContext,
    type Material,
    type Mesh,
    type SceneContext,
} from "babylon-lite";
import type { WorldCollisionShape } from "../collision-shapes.js";
import type { FluidPrimitive } from "../collision-field.js";

/** One dynamic body as the overlay needs it: its live proxy pose and the half-extents of its box. */
export interface ColliderOverlayDynBody {
    name: string;
    position: { x: number; y: number; z: number };
    half: readonly [number, number, number] | undefined;
}

export interface ColliderOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    /** Every AUTHORED collision primitive from the manifest, with the placement it came from. */
    manifestShapes: ReadonlyArray<{ id: string; node: string; shape: WorldCollisionShape }>;
    /** True once that placement's prop has melted — its shape stops being drawn. */
    isRemoved: (id: string) => boolean;
    /** The primitives actually packed into the RUNNING fluid simulations' buffers, live. Each entry
     *  is one simulation's set — the same subset its shader loops over this frame. */
    injectedPrims: () => ReadonlyArray<{ sim: string; prims: readonly FluidPrimitive[] }>;
    /** Live dynamic proxies, in a stable order (the meshes are allocated once, up front). */
    dynBodies: () => readonly ColliderOverlayDynBody[];
    /** Current room id — the static shell is drawn for this chunk only. */
    roomAt: () => string;
}

export interface ColliderOverlay {
    /** Advance the cycle: off → authored manifest primitives → + the dissolvable props' rigid bodies. */
    cycle(): void;
    /** Per frame: dynamic proxies move, and the chunk changes as you walk. */
    onFrame(): void;
}

const PARKED = -1e6;
const MODES = ["off", "authored", "authored+dynamic", "fluid-injected"] as const;

/** Quaternion rotating +Y (the axis every capsule/cylinder is built along) onto `d`. */
function quatFromYTo(d: readonly [number, number, number]): [number, number, number, number] {
    const len = Math.hypot(d[0], d[1], d[2]);
    if (len < 1e-9) return [0, 0, 0, 1];
    const [x, y, z] = [d[0] / len, d[1] / len, d[2] / len];
    if (y > 0.999999) return [0, 0, 0, 1];
    if (y < -0.999999) return [1, 0, 0, 0]; // 180° about X
    // axis = (+Y) × d, angle = acos(dot) — built directly in half-angle form.
    const ax = z,
        az = -x; // cross([0,1,0], [x,y,z]) = (1*z - 0*y, 0*x - 0*z, 0*y - 1*x)
    const s = Math.sqrt((1 + y) * 2);
    return [ax / s, 0, az / s, s * 0.5];
}

export function createColliderOverlay(opts: ColliderOverlayOptions): ColliderOverlay {
    const { engine, scene, canvas, manifestShapes, isRemoved, injectedPrims, dynBodies, roomAt } = opts;

    // Meshes are built HERE, before registerScene, so their material pipeline is compiled with the
    // scene; they are simply parked out of sight until the overlay is switched on.
    const colStaticMat = createStandardMaterial();
    colStaticMat.disableLighting = true;
    colStaticMat.diffuseColor = [1, 1, 1];
    colStaticMat.emissiveColor = [0.15, 0.95, 0.35];
    colStaticMat.alpha = 0.18;
    colStaticMat.backFaceCulling = false;
    const colDynMat = createStandardMaterial();
    colDynMat.disableLighting = true;
    colDynMat.diffuseColor = [1, 1, 1];
    colDynMat.emissiveColor = [1.3, 0.45, 0.08];
    colDynMat.alpha = 0.3;
    colDynMat.backFaceCulling = false;
    const makeColBox = (mat: Material): Mesh => {
        const m = createBox(engine, 1);
        m.material = mat;
        m.pickable = false;
        m.position.set(0, PARKED, 0);
        addToScene(scene, m);
        return m;
    };
    const dynColMeshes = dynBodies().map(() => makeColBox(colDynMat));

    // Authored manifest primitives, drawn with the RIGHT shape: an OBB as an oriented box, a barrel
    // as a capsule, the pod as a cylinder. This is the view for checking what the editor exported —
    // a box drawn around a capsule would hide exactly the mistakes it exists to catch.
    const colShapeMat = createStandardMaterial();
    colShapeMat.disableLighting = true;
    colShapeMat.diffuseColor = [1, 1, 1];
    colShapeMat.emissiveColor = [0.25, 0.6, 1.4];
    colShapeMat.alpha = 0.32;
    colShapeMat.backFaceCulling = false;
    const shapeMeshes = manifestShapes.map(({ shape }) => {
        let m: Mesh;
        if (shape.kind === "sphere") {
            m = createSphere(engine, { diameter: Math.max((shape.radius ?? 0.1) * 2, 1e-3) });
        } else if (shape.kind === "cylinder" || shape.kind === "capsule") {
            const a = shape.pointA ?? shape.centre;
            const b = shape.pointB ?? shape.centre;
            const h = Math.max(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]), 1e-3);
            const r = Math.max(shape.radius ?? 0.1, 1e-3);
            m = shape.kind === "capsule" ? createCapsule(engine, { radius: r, height: h + 2 * r }) : createCylinder(engine, { height: h, diameter: r * 2 });
        } else {
            m = createBox(engine, 1);
        }
        m.material = colShapeMat;
        m.pickable = false;
        m.position.set(0, PARKED, 0);
        addToScene(scene, m);
        return m;
    });
    let colliderMode = 0; // index into MODES

    // ── Stage 3: what the RUNNING simulations actually collide against ───────────────────────────
    // Each live sim was handed the subset of primitives whose bounds met its domain, packed into its
    // own buffer. Linked multi-mesh props can create several simulations with identical sets, so the
    // overlay draws their deduplicated union; otherwise the same translucent mesh is drawn repeatedly
    // and appears increasingly pink, falsely suggesting that some colliders are more active than others.
    //
    // Meshes are rebuilt only when the SET changes (a liquefaction starts or ends), not per frame: a
    // capsule's proportions are baked into its geometry, so a moving prop can be re-posed cheaply but
    // a different primitive needs a different mesh.
    const injMat = createStandardMaterial();
    injMat.disableLighting = true;
    injMat.diffuseColor = [1, 1, 1];
    injMat.emissiveColor = [1.4, 0.2, 0.9];
    injMat.alpha = 0.35;
    injMat.backFaceCulling = false;
    let injMeshes: Mesh[] = [];
    let injSignature = "";
    const vecKey = (v: readonly number[] | undefined): string => v?.map((n) => n.toFixed(3)).join(",") ?? "";
    /** Full visual identity, used to collapse the same primitive packed into several linked sims. */
    const injVisualKey = (p: FluidPrimitive): string => `${p.kind}:${vecKey(p.a)}:${vecKey(p.b)}:${p.radius?.toFixed(3) ?? ""}:${vecKey(p.rotation)}`;
    /** Geometry identity of a primitive — two prims with the same signature can share a mesh. */
    const injKey = (p: FluidPrimitive): string => {
        if (p.kind === "box") return `b:${p.b?.map((v) => v.toFixed(3)).join(",")}`;
        if (p.kind === "sphere") return `s:${p.radius?.toFixed(3)}`;
        const a = p.a,
            b = p.b ?? p.a;
        return `${p.kind[0]}:${p.radius?.toFixed(3)}:${Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]).toFixed(3)}`;
    };
    const buildInjMesh = (p: FluidPrimitive): Mesh => {
        let m: Mesh;
        if (p.kind === "sphere") {
            m = createSphere(engine, { diameter: Math.max((p.radius ?? 0.1) * 2, 1e-3) });
        } else if (p.kind === "capsule" || p.kind === "cylinder") {
            const a = p.a,
                b = p.b ?? p.a;
            const h = Math.max(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]), 1e-3);
            const r = Math.max(p.radius ?? 0.1, 1e-3);
            m = p.kind === "capsule" ? createCapsule(engine, { radius: r, height: h + 2 * r }) : createCylinder(engine, { height: h, diameter: r * 2 });
        } else {
            m = createBox(engine, 1);
        }
        m.material = injMat;
        m.pickable = false;
        addToScene(scene, m);
        return m;
    };
    const refreshInjected = (): void => {
        const sets = colliderMode === 3 ? injectedPrims() : [];
        const unique = new Map<string, FluidPrimitive>();
        for (const { prims } of sets) {
            for (const primitive of prims) unique.set(injVisualKey(primitive), primitive);
        }
        const flat = [...unique.values()];
        const sig = flat.map(injKey).join("|");
        if (sig !== injSignature) {
            for (const m of injMeshes) removeFromScene(scene, m);
            injMeshes = flat.map(buildInjMesh);
            injSignature = sig;
        }
        for (let i = 0; i < flat.length; i++) {
            const p = flat[i]!;
            const m = injMeshes[i]!;
            if (p.kind === "box") {
                const h = p.b ?? [0.1, 0.1, 0.1];
                m.position.set(p.a[0], p.a[1], p.a[2]);
                m.scaling.set(Math.max(h[0] * 2, 1e-3), Math.max(h[1] * 2, 1e-3), Math.max(h[2] * 2, 1e-3));
                const q = p.rotation ?? [0, 0, 0, 1];
                m.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
            } else if (p.kind === "sphere") {
                m.position.set(p.a[0], p.a[1], p.a[2]);
            } else {
                const a = p.a,
                    b = p.b ?? p.a;
                m.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
                const q = quatFromYTo([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
                m.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
            }
        }
    };

    const refresh = (): void => {
        // Authored manifest primitives (stage 1+).
        for (let i = 0; i < manifestShapes.length; i++) {
            const s = manifestShapes[i]!.shape;
            const m = shapeMeshes[i]!;
            // A melted prop's body is gone, so its authored shape must go too — otherwise it hangs
            // in the air exactly where the prop used to be, which reads as a decoding bug.
            if (colliderMode < 1 || colliderMode === 3 || isRemoved(manifestShapes[i]!.id)) {
                m.position.set(0, PARKED, 0);
                continue;
            }
            m.position.set(s.centre[0], s.centre[1], s.centre[2]);
            if (s.kind === "box") {
                const h = s.halfExtents ?? [0.1, 0.1, 0.1];
                m.scaling.set(Math.max(h[0] * 2, 1e-3), Math.max(h[1] * 2, 1e-3), Math.max(h[2] * 2, 1e-3));
                const q = s.rotation ?? [0, 0, 0, 1];
                m.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
            } else if (s.kind === "cylinder" || s.kind === "capsule") {
                const a = s.pointA ?? s.centre;
                const b = s.pointB ?? s.centre;
                const q = quatFromYTo([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
                m.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
                // Midpoint of the axis, which is where the built mesh is centred.
                m.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
            }
        }
        refreshInjected();
        const live = dynBodies();
        for (let i = 0; i < live.length; i++) {
            const d = live[i]!;
            const m = dynColMeshes[i]!;
            const half = d.half;
            // Follows the Havok proxy, so a body that is being pushed out is visibly moving.
            if (colliderMode !== 2 || !half) {
                m.position.set(0, PARKED, 0);
                continue;
            }
            m.position.set(d.position.x, d.position.y, d.position.z);
            m.scaling.set(Math.max(half[0] * 2, 0.05), Math.max(half[1] * 2, 0.05), Math.max(half[2] * 2, 0.05));
        }
    };

    return {
        cycle(): void {
            // Cycles rather than toggles: stage 1 is the authored collision on its own — what the editor
            // exported, and what you normally want to check — with the live rigid bodies added on top at
            // stage 2 when you are chasing a prop that is being pushed around. Stage 3 drops both and
            // shows ONLY what the running simulations were handed, which is a different question: not
            // "is this shape right" but "did this simulation actually get it".
            colliderMode = (colliderMode + 1) % MODES.length;
            refresh();
            canvas.dataset.colliderOverlay = MODES[colliderMode]!;
            if (colliderMode === 0) {
                return;
            }
            const room = roomAt();
            if (colliderMode === 3) {
                const sets = injectedPrims();
                const unique = new Set<string>();
                for (const { prims } of sets) {
                    for (const primitive of prims) unique.add(injVisualKey(primitive));
                }
                // eslint-disable-next-line no-console
                console.log(
                    sets.length
                        ? `[aquanova] fluid-injected collision — ${sets.length} running sim(s), ${unique.size} unique primitive(s):` +
                              sets
                                  .map((s) => {
                                      const k: Record<string, number> = {};
                                      for (const p of s.prims) k[p.kind] = (k[p.kind] ?? 0) + 1;
                                      const moving = s.prims.filter((p) => p.velocity && (p.velocity[0] || p.velocity[1] || p.velocity[2])).length;
                                      return `\n    ${s.sim}: ${s.prims.length} primitive(s) (${Object.entries(k)
                                          .map(([kk, n]) => `${n} ${kk}`)
                                          .join(", ")})${moving ? `, ${moving} moving` : ""}`;
                                  })
                                  .join("")
                        : "[aquanova] fluid-injected collision: no simulation running — liquefy something to see its set"
                );
                return;
            }
            const kinds: Record<string, number> = {};
            for (const { id, shape } of manifestShapes) if (!isRemoved(id)) kinds[shape.kind] = (kinds[shape.kind] ?? 0) + 1;
            // eslint-disable-next-line no-console
            console.log(
                `[aquanova] colliders (${MODES[colliderMode]}) in ${room}: ${manifestShapes.length} authored ` +
                    `(${Object.entries(kinds)
                        .map(([k, n]) => `${n} ${k}`)
                        .join(", ")})` +
                    (colliderMode >= 2
                        ? dynBodies()
                              .map((d) => (d.half ? `\n    dynamic ${d.half.map((v) => (v * 2).toFixed(2)).join(" x ")} m  ${d.name}` : ""))
                              .join("")
                        : "")
            );
        },
        onFrame(): void {
            if (colliderMode > 0) {
                refresh();
            }
        },
    };
}
