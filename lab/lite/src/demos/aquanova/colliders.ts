// Havok collision shell for the ship.
//
// Collide the player against clean per-chunk box shells (floor + ceiling + 4 walls) built from the
// MANIFEST, rather than a trimesh of the raw ship. The 568 overlapping kit modules make a trimesh
// riddled with coplanar/degenerate triangles that pins the character controller; boxes give smooth
// motion and fit the chunked, portal-cell layout exactly. Each portal doorway is cut out of the wall
// it sits on (see addXWall) so the player can walk through once its door is liquefied.
//
// NOTE: this is a SEPARATE representation from the fluid's collision, which is an SDF baked from the
// real triangles. The player and the water genuinely see different worlds — the `B` and `F` debug
// overlays show each.

import { createPhysicsBody, createPhysicsShape, createTransformNode, PhysicsMotionType, PhysicsShapeType, setPhysicsBodyShape, type PhysicsWorld } from "babylon-lite";
import { CEIL_Y, FLOOR_Y, WALL_INSET, WALL_T } from "./constants.js";
import type { ShipManifest } from "./manifest.js";

/** One static collider box, in LITE space, tagged with the chunk that produced it (debug overlay). */
export interface ColliderBox {
    chunk: string;
    c: [number, number, number];
    e: [number, number, number];
}
export function buildShipColliders(world: PhysicsWorld, manifest: ShipManifest, out: ColliderBox[] = []): void {
    let n = 0;
    let curChunk = "";
    // addBox takes a glTF-space centre + full extents; X is negated into Lite scene space.
    // Zero/negative-extent boxes (a doorway that meets a wall edge) are skipped.
    const addBox = (gx: number, gy: number, gz: number, ex: number, ey: number, ez: number): void => {
        if (ex <= 1e-3 || ey <= 1e-3 || ez <= 1e-3) return;
        const node = createTransformNode(`shipCol_${n++}`, -gx, gy, gz);
        const shape = createPhysicsShape(world, { type: PhysicsShapeType.BOX, parameters: { extents: { x: ex, y: ey, z: ez } } });
        setPhysicsBodyShape(world, createPhysicsBody(world, node, PhysicsMotionType.STATIC), shape);
        out.push({ chunk: curChunk, c: [-gx, gy, gz], e: [ex, ey, ez] });
    };
    const midY = (FLOOR_Y + CEIL_Y) / 2;
    const roomH = CEIL_Y - FLOOR_Y;

    // Portal openings on X-facing walls (every ship portal faces ±X). Each doorway is cut OUT of the
    // wall it lies on — into two side panels + a lintel — so the player can pass once the door leaves
    // are liquefied; a sealed wall would otherwise still block the corridor. glTF-space (z/y shared).
    interface XOpening { x: number; z0: number; z1: number; y0: number; y1: number; }
    const xOpenings: XOpening[] = [];
    for (const p of manifest.portals ?? []) {
        if (!p.corners || Math.abs(p.normal?.[0] ?? 0) < 0.5) continue;
        const zs = p.corners.map((c) => c[2]!);
        const ys = p.corners.map((c) => c[1]!);
        xOpenings.push({ x: p.centre[0], z0: Math.min(...zs), z1: Math.max(...zs), y0: Math.min(...ys), y1: Math.max(...ys) });
    }
    const openingOnX = (boundaryX: number, z0: number, z1: number): XOpening | null =>
        xOpenings.find((o) => Math.abs(o.x - boundaryX) < 0.6 && o.z0 >= z0 - 0.5 && o.z1 <= z1 + 0.5) ?? null;

    // One X-facing wall (Lite x = −wallX) across the chunk's z-span, with an optional doorway gap.
    const addXWall = (wallX: number, z0: number, z1: number, o: XOpening | null): void => {
        if (!o) { addBox(wallX, midY, (z0 + z1) / 2, WALL_T, roomH, z1 - z0); return; }
        addBox(wallX, midY, (z0 + o.z0) / 2, WALL_T, roomH, o.z0 - z0); // side panel (−Z of doorway)
        addBox(wallX, midY, (o.z1 + z1) / 2, WALL_T, roomH, z1 - o.z1); // side panel (+Z of doorway)
        addBox(wallX, (o.y1 + CEIL_Y) / 2, (o.z0 + o.z1) / 2, WALL_T, CEIL_Y - o.y1, o.z1 - o.z0); // lintel above
        addBox(wallX, (FLOOR_Y + o.y0) / 2, (o.z0 + o.z1) / 2, WALL_T, o.y0 - FLOOR_Y, o.z1 - o.z0); // sill below (usually none)
    };

    for (const { id, aabb } of manifest.chunks) {
        curChunk = id;
        const [x0, , z0] = aabb.min;
        const [x1, , z1] = aabb.max;
        const cx = (x0 + x1) / 2;
        const cz = (z0 + z1) / 2;
        const w = x1 - x0;
        const d = z1 - z0;
        addBox(cx, FLOOR_Y - WALL_T / 2, cz, w, WALL_T, d); // floor (top at y = 0)
        addBox(cx, CEIL_Y + WALL_T / 2, cz, w, WALL_T, d); // ceiling (bottom at y = 5)
        addBox(cx, midY, z0 + WALL_INSET, w, roomH, WALL_T); // −Z wall
        addBox(cx, midY, z1 - WALL_INSET, w, roomH, WALL_T); // +Z wall
        addXWall(x0 + WALL_INSET, z0, z1, openingOnX(x0, z0, z1)); // −X wall
        addXWall(x1 - WALL_INSET, z0, z1, openingOnX(x1, z0, z1)); // +X wall
    }
}
