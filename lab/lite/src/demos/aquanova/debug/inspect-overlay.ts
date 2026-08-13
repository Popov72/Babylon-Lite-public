// The `I` debug overlay: identify what you are looking at.
//
// Useful with the mouse released (not pointer-locked): hovering a surface GPU-picks the mesh under
// the cursor, highlights the whole owning NODE with a translucent box sized to its world AABB, and
// reports that node's name — the name the Babylon sandbox shows and the key `ship_manifest.json`'s
// `entities` are written against — plus its glTF mesh name and every room touched by the player
// capsule.
//
// The panel also carries the capsule/foot heights and a downward raycast naming whatever supports the
// player, because those are the numbers that say whether the character controller has actually
// settled on the floor or is still at its spawn pose.

import {
    addToScene,
    createBox,
    createGpuPicker,
    createStandardMaterial,
    pickAsync,
    physicsRaycast,
    type EngineContext,
    type Mesh,
    type PhysicsWorld,
    type SceneContext,
} from "babylon-lite";
import { meshGroupBounds } from "../mesh-bounds.js";

export interface InspectOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    world: PhysicsWorld;
    /** Camera, read for the eye position shown in the panel. */
    cam: { position: { x: number; y: number; z: number } };
    /** Player capsule, for the foot height and the support raycast. */
    character: { getPosition(): { x: number; y: number; z: number } };
    getCapsuleHeight: () => number;
    roomsAt: () => readonly string[];
    /** The demo's shared GPU picker, created lazily — the weapon needs one anyway, and two pickers
     *  would mean two sets of GPU targets for no benefit. */
    getPicker: () => ReturnType<typeof createGpuPicker>;
    /** glTF NODE name of a mesh (what the manifest is keyed by), falling back to the mesh name. */
    nodeNameOf: (m: Mesh) => string | undefined;
    /** Every primitive of the mesh's owning node — a multi-primitive node is ONE object to the player
     *  (it liquefies as a unit), so the highlight unions their AABBs. */
    nodePrimitivesOf: (m: Mesh) => readonly Mesh[] | undefined;
    /** Manifest chunk for a static node, or every intersected chunk for a dynamic node. */
    chunkIdsOf: (m: Mesh) => readonly string[];
    /** Whether portal visibility updates the owning node's chunk membership every frame. */
    isDynamic: (m: Mesh) => boolean;
    /** Whether free-fly is on, annotated in the panel. */
    isNoclip: () => boolean;
}

export interface InspectOverlay {
    toggle(): void;
    /** Whether the overlay is on — the click/fire handlers must not act while it is. */
    isOn(): boolean;
    /** Refresh the player and room readout while the overlay is visible. */
    onFrame(): void;
    /** Hover handler — a no-op unless the overlay is on. Keeps one GPU readback in flight. */
    pickAt(x: number, y: number): void;
}

export function createInspectOverlay(opts: InspectOverlayOptions): InspectOverlay {
    const { engine, scene, canvas, world, cam, character, getCapsuleHeight, roomsAt, getPicker, nodeNameOf, nodePrimitivesOf, chunkIdsOf, isDynamic, isNoclip } = opts;

    let inspect = false;
    let inspectPicking = false; // one in-flight GPU readback at a time
    let hoverNode = "";
    let hoverMesh = "";
    let hoverChunks: readonly string[] = [];
    let hoverDynamic = false;

    // Reusable highlight box: unlit cyan, alpha-blended so the mesh shows through. Parked far away
    // when hidden; pickable=false so it never picks itself. Added before registerScene so its
    // material pipeline is compiled with the scene.
    const hlMat = createStandardMaterial();
    hlMat.disableLighting = true;
    hlMat.diffuseColor = [1, 1, 1];
    hlMat.emissiveColor = [0.25, 0.85, 1];
    hlMat.alpha = 0.28;
    hlMat.backFaceCulling = false;
    const hlBox = createBox(engine, 1);
    hlBox.material = hlMat;
    hlBox.pickable = false;
    const hideHl = (): void => hlBox.position.set(0, -1e6, 0);
    hideHl();
    addToScene(scene, hlBox);

    const panel = document.createElement("div");
    panel.id = "aq-inspect";
    panel.style.cssText =
        "position:fixed;left:12px;top:12px;z-index:20;display:none;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;color:#d6eefc;background:rgba(0,0,0,.6);padding:8px 11px;border-radius:6px;pointer-events:none;white-space:pre;max-width:60vw;";
    document.body.appendChild(panel);

    const refreshPanel = (): void => {
        const what = hoverNode
            ? `${hoverNode}\nMesh: ${hoverMesh}\nChunks: ${hoverChunks.length > 0 ? hoverChunks.join(", ") : "none"}\nMobility: ${hoverDynamic ? "dynamic" : "static"}`
            : "—  (hover a surface)";
        const cp = character.getPosition();
        const footY = cp.y - getCapsuleHeight() / 2;
        const hit = physicsRaycast(world, { x: cp.x, y: footY + 0.05, z: cp.z }, { x: cp.x, y: footY - 3, z: cp.z });
        const ground = hit.hasHit ? `${hit.body?.node?.name ?? "?"} @ y ${hit.hitPoint.y.toFixed(3)}` : "nothing within 3 m";
        const cam3 = `${cam.position.x.toFixed(2)}, ${cam.position.y.toFixed(2)}, ${cam.position.z.toFixed(2)}`;
        const rooms = roomsAt();
        panel.textContent = `INSPECT (I)   Rooms: ${rooms.length > 0 ? rooms.join(", ") : "—"}\nEye: ${cam3}   capsule y ${cp.y.toFixed(3)}   foot y ${footY.toFixed(3)}${isNoclip() ? "   [FREE-FLY]" : ""}\nStanding on: ${ground}\nNode: ${what}`;
    };

    const inspectAt = async (x: number, y: number): Promise<void> => {
        if (!inspect || inspectPicking) {
            return;
        }
        inspectPicking = true;
        try {
            const info = await pickAsync(getPicker(), x, y);
            const mesh = info.hit ? (info.pickedMesh as Mesh | null) : null;
            if (mesh) {
                // The glTF NODE name is what the sandbox shows and what `entities` are keyed by;
                // `mesh.name` is the glTF MESH name, shared by every instance (all six door leaves are
                // "Cube.009"), so it is shown second for reference only.
                hoverNode = nodeNameOf(mesh) || mesh.name || "(unnamed)";
                hoverMesh = mesh.name || "(unnamed)";
                hoverChunks = chunkIdsOf(mesh);
                hoverDynamic = isDynamic(mesh);
                const bounds = meshGroupBounds(nodePrimitivesOf(mesh) ?? [mesh]);
                if (bounds) {
                    hlBox.position.set(bounds.centre[0], bounds.centre[1], bounds.centre[2]);
                    hlBox.scaling.set(Math.max(bounds.half[0] * 2, 0.02) * 1.03, Math.max(bounds.half[1] * 2, 0.02) * 1.03, Math.max(bounds.half[2] * 2, 0.02) * 1.03);
                } else {
                    hideHl();
                }
            } else {
                hoverNode = "";
                hoverMesh = "";
                hoverChunks = [];
                hoverDynamic = false;
                hideHl();
            }
            refreshPanel();
        } finally {
            inspectPicking = false;
        }
    };

    return {
        isOn: () => inspect,
        onFrame(): void {
            if (inspect) refreshPanel();
        },
        toggle(): void {
            inspect = !inspect;
            if (inspect) {
                if (document.pointerLockElement === canvas) {
                    document.exitPointerLock(); // free the mouse to hover
                }
                refreshPanel();
                panel.style.display = "block";
            } else {
                panel.style.display = "none";
                hoverNode = "";
                hoverMesh = "";
                hoverChunks = [];
                hoverDynamic = false;
                hideHl();
            }
        },
        pickAt(x: number, y: number): void {
            void inspectAt(x, y);
        },
    };
}
