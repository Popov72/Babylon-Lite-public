// Havok collision for the ship, built entirely from the manifest's authored primitives.
//
// There is no synthesised stand-in any more. The demo used to collide the player against a box shell
// derived from each chunk's AABB (floor + ceiling + four walls, with the portal doorways cut out of
// them), because the raw kit geometry made a useless trimesh and nothing else described the rooms.
// Now every placed module carries its own OBB / sphere / capsule / cylinder, so the manifest is the
// single source of collision for the player AND the fluid — if a surface has no authored shape, it
// is not solid, and that is a gap to fill in the editor rather than to paper over here.

import {
    createPhysicsBody,
    createPhysicsShape,
    createTransformNode,
    PhysicsMotionType,
    PhysicsShapeType,
    setPhysicsBodyShape,
    type PhysicsBody,
    type PhysicsWorld,
} from "babylon-lite";
import type { WorldCollisionShape } from "./collision-shapes.js";

export interface ManifestColliderSource {
    readonly instanceId: string;
    readonly shape: WorldCollisionShape;
}

export interface ManifestCollider {
    readonly instanceId: string;
    readonly shape: WorldCollisionShape;
    readonly body: PhysicsBody;
}
/** Build a manifest collision primitive (given in Lite WORLD space) as a shape on a body whose node
 *  sits at `origin`. Everything is re-expressed relative to that origin, so the caller is free to
 *  put the body node wherever it needs it — the dynamic-prop path anchors it at the SDF bake centre
 *  so the fluid boundary and the rigid body stay in agreement. */
export function createWorldCollisionShape(world: PhysicsWorld, s: WorldCollisionShape, origin: readonly [number, number, number] = s.centre): ReturnType<typeof createPhysicsShape> {
    const rel = (v: readonly [number, number, number]): { x: number; y: number; z: number } => ({ x: v[0] - origin[0], y: v[1] - origin[1], z: v[2] - origin[2] });
    if (s.kind === "sphere") {
        return createPhysicsShape(world, { type: PhysicsShapeType.SPHERE, parameters: { radius: Math.max(s.radius ?? 0.1, 1e-3), center: rel(s.centre) } });
    }
    if (s.kind === "cylinder" || s.kind === "capsule") {
        return createPhysicsShape(world, {
            type: s.kind === "capsule" ? PhysicsShapeType.CAPSULE : PhysicsShapeType.CYLINDER,
            parameters: { radius: Math.max(s.radius ?? 0.1, 1e-3), pointA: rel(s.pointA ?? s.centre), pointB: rel(s.pointB ?? s.centre) },
        });
    }
    const h = s.halfExtents ?? [0.1, 0.1, 0.1];
    const q = s.rotation ?? [0, 0, 0, 1];
    // Havok's box takes FULL extents plus its own orientation, which is what makes an OBB
    // expressible without a separate shape type.
    return createPhysicsShape(world, {
        type: PhysicsShapeType.BOX,
        parameters: {
            extents: { x: Math.max(h[0] * 2, 1e-3), y: Math.max(h[1] * 2, 1e-3), z: Math.max(h[2] * 2, 1e-3) },
            rotation: { x: q[0], y: q[1], z: q[2], w: q[3] },
            center: rel(s.centre),
        },
    });
}

/**
 * Static Havok bodies for authored collision.
 *
 * This is what makes props solid without anyone marking them `dynamic` — a collision shape is all it
 * takes to be Havok-aware. Shapes arrive already resolved into world space from their placement
 * node, so this only has to build bodies. Dissolvable placements are excluded by the caller: they
 * get their own removable body in the dynamic-prop pass, and a duplicate static body here would
 * outlive the melt as an invisible wall.
 */
export function buildManifestColliders(world: PhysicsWorld, sources: Iterable<ManifestColliderSource>): ManifestCollider[] {
    const made: ManifestCollider[] = [];
    let n = 0;
    for (const { instanceId, shape } of sources) {
        const tn = createTransformNode(`mfCol_${n++}`, shape.centre[0], shape.centre[1], shape.centre[2]);
        const body = createPhysicsBody(world, tn, PhysicsMotionType.STATIC);
        setPhysicsBodyShape(world, body, createWorldCollisionShape(world, shape));
        made.push({ instanceId, shape, body });
    }
    return made;
}
