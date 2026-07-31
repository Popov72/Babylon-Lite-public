// The `I` debug overlay: identify what you are looking at.
//
// Useful with the mouse released (not pointer-locked): hovering a surface GPU-picks the mesh under
// the cursor, highlights the whole owning NODE with a translucent box sized to its world AABB, and
// reports that node's name — the name the Babylon sandbox shows and the key `ship_manifest.json`'s
// `entities` are written against — plus its glTF mesh name and the room the camera is in.
//
// The panel also carries the capsule/foot heights and a downward raycast naming whatever supports the
// player, because those are the numbers that say whether the character controller has actually
// settled on the floor or is still at its spawn pose.

import { addToScene, createBox, createGpuPicker, createStandardMaterial, pickAsync, physicsRaycast, type EngineContext, type Mesh, type PhysicsWorld, type SceneContext } from "babylon-lite";

export interface InspectOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    world: PhysicsWorld;
    /** Camera, read for the eye position shown in the panel. */
    cam: { position: { x: number; y: number; z: number } };
    /** Player capsule, for the foot height and the support raycast. */
    character: { getPosition(): { x: number; y: number; z: number } };
    capsuleHeight: number;
    roomAt: () => string;
    /** The demo's shared GPU picker, created lazily — the weapon needs one anyway, and two pickers
     *  would mean two sets of GPU targets for no benefit. */
    getPicker: () => ReturnType<typeof createGpuPicker>;
    /** glTF NODE name of a mesh (what the manifest is keyed by), falling back to the mesh name. */
    nodeNameOf: (m: Mesh) => string | undefined;
    /** Every primitive of the mesh's owning node — a multi-primitive node is ONE object to the player
     *  (it liquefies as a unit), so the highlight unions their AABBs. */
    nodePrimitivesOf: (m: Mesh) => readonly Mesh[] | undefined;
    /** Whether free-fly is on, annotated in the panel. */
    isNoclip: () => boolean;
}

export interface InspectOverlay {
    toggle(): void;
    /** Whether the overlay is on — the click/fire handlers must not act while it is. */
    isOn(): boolean;
    /** Hover handler — a no-op unless the overlay is on. Keeps one GPU readback in flight. */
    pickAt(x: number, y: number): void;
}

export function createInspectOverlay(opts: InspectOverlayOptions): InspectOverlay {
    const { engine, scene, canvas, world, cam, character, capsuleHeight, roomAt, getPicker, nodeNameOf, nodePrimitivesOf, isNoclip } = opts;

    let inspect = false;
    let inspectPicking = false; // one in-flight GPU readback at a time
    let hoverNode = "";
    let hoverMesh = "";

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
        const what = hoverNode ? `${hoverNode}\nMesh: ${hoverMesh}` : "—  (hover a surface)";
        const cp = character.getPosition();
        const footY = cp.y - capsuleHeight / 2;
        const hit = physicsRaycast(world, { x: cp.x, y: footY + 0.05, z: cp.z }, { x: cp.x, y: footY - 3, z: cp.z });
        const ground = hit.hasHit ? `${hit.body?.node?.name ?? "?"} @ y ${hit.hitPoint.y.toFixed(3)}` : "nothing within 3 m";
        const cam3 = `${cam.position.x.toFixed(2)}, ${cam.position.y.toFixed(2)}, ${cam.position.z.toFixed(2)}`;
        panel.textContent = `INSPECT (I)   Room: ${roomAt()}\nEye: ${cam3}   capsule y ${cp.y.toFixed(3)}   foot y ${footY.toFixed(3)}${isNoclip() ? "   [FREE-FLY]" : ""}\nStanding on: ${ground}\nNode: ${what}`;
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
                let mn: number[] | null = null;
                let mx: number[] | null = null;
                for (const prim of nodePrimitivesOf(mesh) ?? [mesh]) {
                    const a = prim.boundMin;
                    const b = prim.boundMax;
                    if (!a || !b) {
                        continue;
                    }
                    mn = mn ? [Math.min(mn[0]!, a[0]!), Math.min(mn[1]!, a[1]!), Math.min(mn[2]!, a[2]!)] : [a[0]!, a[1]!, a[2]!];
                    mx = mx ? [Math.max(mx[0]!, b[0]!), Math.max(mx[1]!, b[1]!), Math.max(mx[2]!, b[2]!)] : [b[0]!, b[1]!, b[2]!];
                }
                if (mn && mx) {
                    hlBox.position.set((mn[0]! + mx[0]!) / 2, (mn[1]! + mx[1]!) / 2, (mn[2]! + mx[2]!) / 2);
                    hlBox.scaling.set(Math.max(mx[0]! - mn[0]!, 0.02) * 1.03, Math.max(mx[1]! - mn[1]!, 0.02) * 1.03, Math.max(mx[2]! - mn[2]!, 0.02) * 1.03);
                } else {
                    hideHl();
                }
            } else {
                hoverNode = "";
                hoverMesh = "";
                hideHl();
            }
            refreshPanel();
        } finally {
            inspectPicking = false;
        }
    };

    return {
        isOn: () => inspect,
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
                hideHl();
            }
        },
        pickAt(x: number, y: number): void {
            void inspectAt(x, y);
        },
    };
}
