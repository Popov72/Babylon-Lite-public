// Manifest collision primitives → world-space shapes.
//
// The editor authors one or more collision primitives per KIT MODULE, in module-local space, and
// places that module many times. This turns (placement node, module primitives) into world-space
// shapes in the loaded scene's coordinates, ready to hand to Havok.
//
// The transform comes from the LOADED glTF NODE, not from `instances[]`, and that is the whole
// trick. The manifest's own `space` block spells out why: `moduleCollision` is glTF space while
// `instances[]` is editor space, so composing those two directly mixes spaces — it looks perfect on
// anything symmetrical and is wrong on everything turned. The loaded node is already the same
// placement expressed in the scene's own space, so `worldMatrix · shape` needs no conversion at all:
//
//     shape_lite = W_lite · p_gltf ,  where  W_lite = M · N_gltf  (M = the loader's X mirror)
//
// which is exactly the composition the renderer uses to draw the mesh. Nothing here has to know
// about handedness, Euler order or negative scales — the loader has already resolved all three.
//
// The node is found from its `extras.id` (the exporter stamps `{ id, module, chunk }` on every
// placement node), which is also the only unique key: node NAMES repeat across placements.

import type { Vec3 } from "./constants.js";

export type CollisionKind = "box" | "sphere" | "cylinder" | "capsule";

interface ShapeCommon {
    id?: string;
    kind: CollisionKind;
    centre: Vec3;
}
export interface BoxCollisionShape extends ShapeCommon {
    kind: "box";
    halfExtents: Vec3;
    /** Quaternion [x, y, z, w]. The exporter writes identity as [0,0,0,-1]; -q is the same rotation. */
    rotation?: [number, number, number, number];
}
export interface SphereCollisionShape extends ShapeCommon {
    kind: "sphere";
    radius: number;
}
export interface AxialCollisionShape extends ShapeCommon {
    kind: "cylinder" | "capsule";
    radius: number;
    height?: number;
    pointA: Vec3;
    pointB: Vec3;
}
export type ShipCollisionShape = BoxCollisionShape | SphereCollisionShape | AxialCollisionShape;

/** One placed kit module. `node` is the glTF node name, which is what entity behaviours key off. */
export interface ShipInstance {
    id: string;
    module: string;
    chunk?: string;
    name?: string;
    node?: string;
    position: Vec3;
    /** Euler angles in DEGREES, Babylon yaw-pitch-roll order (see file header). */
    rotation?: Vec3;
    scale?: Vec3;
}

/** A collision primitive resolved into Lite world space. */
export interface WorldCollisionShape {
    kind: CollisionKind;
    /** Lite-space centre. */
    centre: [number, number, number];
    /** Box only: half extents along its own (rotated) axes. */
    halfExtents?: [number, number, number];
    /** Box only: orientation in Lite space, [x, y, z, w]. */
    rotation?: [number, number, number, number];
    /** Sphere / cylinder / capsule. */
    radius?: number;
    /** Cylinder / capsule: the axis end points, in Lite space. */
    pointA?: [number, number, number];
    pointB?: [number, number, number];
}

type Mat3 = [number, number, number, number, number, number, number, number, number]; // column-major

/**
 * The node's linear map and translation, taken straight from its world matrix (column-major 4×4).
 *
 * `A` is generally LEFT-handed here — the loader mirrors the ship's root — and may carry non-uniform
 * or negative scale. Callers must not assume it is a rotation.
 */
function matrixParts(world: ArrayLike<number>): { A: Mat3; t: [number, number, number] } {
    const A: Mat3 = [world[0]!, world[1]!, world[2]!, world[4]!, world[5]!, world[6]!, world[8]!, world[9]!, world[10]!];
    return { A, t: [world[12]!, world[13]!, world[14]!] };
}

function mat3FromQuat([x, y, z, w]: readonly [number, number, number, number]): Mat3 {
    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const wx = w * x, wy = w * y, wz = w * z;
    // Column-major: column i is the image of basis vector i.
    return [1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)];
}

function mat3Mul(a: Mat3, b: Mat3): Mat3 {
    const o = new Array<number>(9) as Mat3;
    for (let c = 0; c < 3; c++) {
        for (let r = 0; r < 3; r++) {
            o[c * 3 + r] = a[r]! * b[c * 3]! + a[3 + r]! * b[c * 3 + 1]! + a[6 + r]! * b[c * 3 + 2]!;
        }
    }
    return o;
}

function mat3Apply(m: Mat3, v: Vec3): [number, number, number] {
    return [m[0]! * v[0] + m[3]! * v[1] + m[6]! * v[2], m[1]! * v[0] + m[4]! * v[1] + m[7]! * v[2], m[2]! * v[0] + m[5]! * v[1] + m[8]! * v[2]];
}

/** Orthonormal (right-handed) rotation matrix → quaternion.
 *
 *  `m` is column-major, so the destructured names are mRC = row R, column C. The difference terms
 *  must be taken as (lower − upper), i.e. `m21 - m12`: writing them the other way round yields the
 *  CONJUGATE, which is the inverse rotation. That is invisible on identity and on 180° turns —
 *  both are self-inverse — and wrong on everything else, so it survives any check that only
 *  compares shape centres. */
function quatFromMat3(m: Mat3): [number, number, number, number] {
    const [m00, m10, m20, m01, m11, m21, m02, m12, m22] = m;
    const tr = m00! + m11! + m22!;
    if (tr > 0) {
        const s = Math.sqrt(tr + 1) * 2;
        return [(m21! - m12!) / s, (m02! - m20!) / s, (m10! - m01!) / s, 0.25 * s];
    }
    if (m00! > m11! && m00! > m22!) {
        const s = Math.sqrt(1 + m00! - m11! - m22!) * 2;
        return [0.25 * s, (m10! + m01!) / s, (m20! + m02!) / s, (m21! - m12!) / s];
    }
    if (m11! > m22!) {
        const s = Math.sqrt(1 + m11! - m00! - m22!) * 2;
        return [(m10! + m01!) / s, 0.25 * s, (m21! + m12!) / s, (m02! - m20!) / s];
    }
    const s = Math.sqrt(1 + m22! - m00! - m11!) * 2;
    return [(m20! + m02!) / s, (m21! + m12!) / s, 0.25 * s, (m10! - m01!) / s];
}

/** Column `i` of a 3×3, as a vector. */
const col = (m: Mat3, i: number): [number, number, number] => [m[i * 3]!, m[i * 3 + 1]!, m[i * 3 + 2]!];
const norm = (v: readonly [number, number, number]): number => Math.hypot(v[0], v[1], v[2]);

/**
 * Resolve one module's collision primitives into Lite world space for a given placement.
 *
 * Non-uniform scale on a primitive whose own rotation is not axis-aligned is not exactly
 * representable (a sheared box is not a box); the axis lengths are taken from the transformed
 * columns, which is exact for the axis-aligned and uniformly-scaled cases the kit actually uses.
 * Radii take the largest axis scale, so a scaled sphere/capsule errs toward being too fat rather
 * than letting the player clip into it.
 */
/**
 * Resolve one module's collision primitives into world space for a placement, given that
 * placement's LOADED node world matrix (column-major 4×4).
 *
 * The shapes are used exactly as exported — they are glTF-local and the node matrix carries the
 * glTF→scene mirror, so the composition is already correct (see the file header). Non-uniform scale
 * on a primitive whose own rotation is not axis-aligned is not exactly representable (a sheared box
 * is not a box); axis lengths are taken from the transformed columns, which is exact for the
 * axis-aligned and uniformly-scaled cases the kit uses. Radii take the largest axis scale, so a
 * scaled sphere/capsule errs toward being too fat rather than letting the player clip into it.
 */
export function worldShapesForMatrix(world: ArrayLike<number>, shapes: ShipCollisionShape | readonly ShipCollisionShape[]): WorldCollisionShape[] {
    const { A, t } = matrixParts(world);
    const toWorld = (v: Vec3): [number, number, number] => {
        const a = mat3Apply(A, v);
        return [t[0] + a[0], t[1] + a[1], t[2] + a[2]];
    };
    const axisScale = Math.max(norm(col(A, 0)), norm(col(A, 1)), norm(col(A, 2)));
    const out: WorldCollisionShape[] = [];

    for (const s of Array.isArray(shapes) ? shapes : [shapes as ShipCollisionShape]) {
        if (s.kind === "box") {
            const local = s.rotation ? mat3FromQuat(s.rotation) : ([1, 0, 0, 0, 1, 0, 0, 0, 1] as Mat3);
            const M = mat3Mul(A, local); // the box's own axes, scaled and mirrored into Lite space
            const lens: [number, number, number] = [norm(col(M, 0)), norm(col(M, 1)), norm(col(M, 2))];
            const R: Mat3 = [...M] as Mat3;
            for (let i = 0; i < 3; i++) {
                const l = lens[i] || 1;
                R[i * 3] = M[i * 3]! / l;
                R[i * 3 + 1] = M[i * 3 + 1]! / l;
                R[i * 3 + 2] = M[i * 3 + 2]! / l;
            }
            // The loader's root mirror (and any negative placement scale) makes the frame
            // left-handed, which is not a rotation. Flipping one axis restores right-handedness and
            // leaves the box itself unchanged — a box is symmetric about each of its own axes.
            const det =
                R[0]! * (R[4]! * R[8]! - R[5]! * R[7]!) - R[3]! * (R[1]! * R[8]! - R[2]! * R[7]!) + R[6]! * (R[1]! * R[5]! - R[2]! * R[4]!);
            if (det < 0) {
                R[0] = -R[0]!;
                R[1] = -R[1]!;
                R[2] = -R[2]!;
            }
            out.push({
                kind: "box",
                centre: toWorld(s.centre),
                halfExtents: [Math.abs(s.halfExtents[0]) * lens[0], Math.abs(s.halfExtents[1]) * lens[1], Math.abs(s.halfExtents[2]) * lens[2]],
                rotation: quatFromMat3(R),
            });
        } else if (s.kind === "sphere") {
            out.push({ kind: "sphere", centre: toWorld(s.centre), radius: s.radius * axisScale });
        } else {
            out.push({ kind: s.kind, centre: toWorld(s.centre), radius: s.radius * axisScale, pointA: toWorld(s.pointA), pointB: toWorld(s.pointB) });
        }
    }
    return out;
}
