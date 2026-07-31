// The `B` debug overlay: what the PLAYER collides with.
//
// Draws the manifest-built static box shell for the chunk you are in (green, each portal cut out of
// its wall into side panels + a lintel) plus every dynamic body's Havok proxy box (orange). Those
// proxies are AABBs of their mesh, so a hollow prop such as a door frame shows up as a solid slab
// over its own opening — which is exactly the sort of thing this exists to make visible.
//
// This is a DIFFERENT representation from the fluid's collision, which is an SDF baked from the real
// triangles (see debug/sdf-overlay.ts). Water flows around pipes and greebles the player walks
// straight through; showing both is how you see that.

import { addToScene, createBox, createStandardMaterial, type EngineContext, type Material, type Mesh, type SceneContext } from "babylon-lite";
import type { ColliderBox } from "../colliders.js";

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
    /** Static shell boxes, tagged with the chunk that produced them. */
    colliderBoxes: readonly ColliderBox[];
    /** Live dynamic proxies, in a stable order (the meshes are allocated once, up front). */
    dynBodies: () => readonly ColliderOverlayDynBody[];
    /** Current room id — the static shell is drawn for this chunk only. */
    roomAt: () => string;
}

export interface ColliderOverlay {
    /** Advance the 3-stage cycle: off → dynamic proxies → + the chunk's static shell. */
    cycle(): void;
    /** Per frame: dynamic proxies move, and the chunk changes as you walk. */
    onFrame(): void;
}

const PARKED = -1e6;

export function createColliderOverlay(opts: ColliderOverlayOptions): ColliderOverlay {
    const { engine, scene, canvas, colliderBoxes, dynBodies, roomAt } = opts;

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
    const staticColMeshes = colliderBoxes.map(() => makeColBox(colStaticMat));
    const dynColMeshes = dynBodies().map(() => makeColBox(colDynMat));
    let colliderMode = 0; // 0 = off, 1 = dynamic proxies only, 2 = + the chunk's static shell

    const refresh = (): void => {
        const room = roomAt();
        for (let i = 0; i < colliderBoxes.length; i++) {
            const cb = colliderBoxes[i]!;
            const m = staticColMeshes[i]!;
            if (colliderMode < 2 || cb.chunk !== room) {
                m.position.set(0, PARKED, 0);
                continue;
            }
            m.position.set(cb.c[0], cb.c[1], cb.c[2]);
            m.scaling.set(cb.e[0], cb.e[1], cb.e[2]);
        }
        const live = dynBodies();
        for (let i = 0; i < live.length; i++) {
            const d = live[i]!;
            const m = dynColMeshes[i]!;
            const half = d.half;
            // Follows the Havok proxy, so a body that is being pushed out is visibly moving.
            if (colliderMode === 0 || !half) {
                m.position.set(0, PARKED, 0);
                continue;
            }
            m.position.set(d.position.x, d.position.y, d.position.z);
            m.scaling.set(Math.max(half[0] * 2, 0.05), Math.max(half[1] * 2, 0.05), Math.max(half[2] * 2, 0.05));
        }
    };

    return {
        cycle(): void {
            // Cycles rather than toggles: the static shell is a room-sized box you stand INSIDE, so its
            // back faces wash the whole screen. Stage 1 shows just the dynamic proxies, which is what
            // you want when chasing a prop that is being pushed around.
            colliderMode = (colliderMode + 1) % 3;
            refresh();
            canvas.dataset.colliderOverlay = ["off", "dynamic", "all"][colliderMode]!;
            if (colliderMode === 0) {
                return;
            }
            const room = roomAt();
            const shown = colliderMode === 2 ? colliderBoxes.filter((b) => b.chunk === room) : [];
            // eslint-disable-next-line no-console
            console.log(
                `[aquanova] colliders (${["off", "dynamic", "all"][colliderMode]}) in ${room}: ${shown.length} static box(es)` +
                    shown.map((b) => `\n    static  ${b.e.map((v) => v.toFixed(2)).join(" x ")} m  at [${b.c.map((v) => v.toFixed(2)).join(", ")}]`).join("") +
                    dynBodies()
                        .map((d) => (d.half ? `\n    dynamic ${d.half.map((v) => (v * 2).toFixed(2)).join(" x ")} m  ${d.name}` : ""))
                        .join("")
            );
        },
        onFrame(): void {
            if (colliderMode > 0) {
                refresh();
            }
        },
    };
}
