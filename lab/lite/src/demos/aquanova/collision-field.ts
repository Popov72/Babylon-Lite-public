// Analytic collision field for the fluid — the replacement for baked SDF grids.
//
// The ship's collision is authored as primitives (OBB / sphere / capsule / cylinder) in the manifest,
// so the fluid can evaluate them directly instead of sampling a distance grid baked from triangles.
// That removes the whole bake step, the per-room grids and their memory, and — more importantly — the
// class of bugs that came with them: a bake is only ever as good as its SIGN, and kit-bashed level
// geometry (open shells, coplanar duplicates, meshes modelled 1 cm apart) has no reliable inside.
//
// The primitives live in a STORAGE BUFFER, never baked into the WGSL. The solvers cache compute
// pipelines keyed on shader source, so emitting a different shader per liquefaction would miss the
// cache every time and pay a full pipeline compile (~450 ms, measured) on each shot. One shader, a
// buffer that changes, and every sim shares the same compiled pipeline.
//
// Sign convention matches the rest of the fluid: NEGATIVE inside the solid, positive outside, and the
// union of solids is `min`.

/** Floats per packed primitive. 16 keeps each one 64-byte aligned and leaves room for velocity. */
export const PRIM_STRIDE = 16;
/** Floats before the first primitive: [solid count, subtraction count, _, _]. */
export const PRIM_HEADER = 4;

export const PRIM_BOX = 0;
export const PRIM_SPHERE = 1;
export const PRIM_CAPSULE = 2;
export const PRIM_CYLINDER = 3;
export const PRIM_HOLLOW_CYLINDER = 4;
/** Offset of the active flag inside one packed primitive. */
export const PRIM_ACTIVE_OFFSET = 15;

/** One collision primitive in world space, as the fluid sees it. */
export interface FluidPrimitive {
    kind: "box" | "sphere" | "capsule" | "cylinder" | "hollowCylinder";
    /** Box/sphere centre, or the first end point of an axial primitive. */
    a: readonly [number, number, number];
    /** Box HALF extents, or the second axis end point. Unused for a sphere. */
    b?: readonly [number, number, number];
    /** Sphere/capsule/cylinder radius; outer radius for a hollow cylinder. */
    radius?: number;
    /** Hollow-cylinder inner radius. `radius` is its outer radius. */
    innerRadius?: number;
    /** Box orientation [x, y, z, w]. */
    rotation?: readonly [number, number, number, number];
    /** Linear velocity (m/s). Lets the solver recover boundary motion from -∂sdf/∂t. */
    velocity?: readonly [number, number, number];
    /** False makes the shader skip this slot without rebuilding or compacting the collision set. */
    active?: boolean;
}

/** Fixed-capacity set of primitives subtracted from the solid collision field. */
export interface FluidHoleRing {
    readonly holes: FluidHole[];
}

/** One subtraction primitive and the solid primitive slot it is allowed to carve. */
export interface FluidHole {
    readonly primitive: FluidPrimitive;
    readonly targetSolidSlot: number;
}

const KIND_CODE: Record<FluidPrimitive["kind"], number> = {
    box: PRIM_BOX,
    sphere: PRIM_SPHERE,
    capsule: PRIM_CAPSULE,
    cylinder: PRIM_CYLINDER,
    hollowCylinder: PRIM_HOLLOW_CYLINDER,
};

/** Write one primitive into `out` at primitive index `i`. */
export function packPrimitive(out: Float32Array, i: number, p: FluidPrimitive): void {
    const o = PRIM_HEADER + i * PRIM_STRIDE;
    out[o] = KIND_CODE[p.kind];
    out[o + 1] = p.a[0];
    out[o + 2] = p.a[1];
    out[o + 3] = p.a[2];
    const b = p.b ?? [0, 0, 0];
    out[o + 4] = b[0];
    out[o + 5] = b[1];
    out[o + 6] = b[2];
    out[o + 7] = p.radius ?? 0;
    if (p.kind === "hollowCylinder") {
        out[o + 8] = p.innerRadius ?? 0;
        out[o + 9] = 0;
        out[o + 10] = 0;
        out[o + 11] = 1;
    } else {
        const q = p.rotation ?? [0, 0, 0, 1];
        out[o + 8] = q[0];
        out[o + 9] = q[1];
        out[o + 10] = q[2];
        out[o + 11] = q[3];
    }
    const v = p.velocity ?? [0, 0, 0];
    out[o + 12] = v[0];
    out[o + 13] = v[1];
    out[o + 14] = v[2];
    out[o + PRIM_ACTIVE_OFFSET] = p.active === false ? 0 : 1;
}

/** Update only one packed primitive's active flag. */
export function setPackedPrimitiveActive(out: Float32Array, i: number, active: boolean): void {
    out[PRIM_HEADER + i * PRIM_STRIDE + PRIM_ACTIVE_OFFSET] = active ? 1 : 0;
}

/** Pack a whole set, writing the count into the header. `out` must hold `PRIM_HEADER + n*PRIM_STRIDE`. */
export function packPrimitives(out: Float32Array, prims: readonly FluidPrimitive[]): void {
    out[0] = prims.length;
    out[1] = 0;
    for (let i = 0; i < prims.length; i++) packPrimitive(out, i, prims[i]!);
}

/** Pack one subtraction primitive immediately after the regular primitive slots. */
export function packSubtractionPrimitive(out: Float32Array, solidCount: number, index: number, hole: FluidHole): void {
    const packedSlot = solidCount + index;
    packPrimitive(out, packedSlot, hole.primitive);
    out[PRIM_HEADER + packedSlot * PRIM_STRIDE + 8] = hole.targetSolidSlot;
}

/** Pack subtraction primitives immediately after `solidCount` regular primitives. */
export function packSubtractionPrimitives(out: Float32Array, solidCount: number, holes: readonly FluidHole[]): void {
    out[1] = holes.length;
    for (let i = 0; i < holes.length; i++) packSubtractionPrimitive(out, solidCount, i, holes[i]!);
}

/** Byte size of a buffer holding `capacity` primitives. */
export const primBufferBytes = (capacity: number): number => (PRIM_HEADER + capacity * PRIM_STRIDE) * 4;

/** Build a finite cylinder centred on a pistol impact and aligned with the shot. */
export function shotHolePrimitive(center: readonly [number, number, number], direction: readonly [number, number, number], halfLength: number, radius: number): FluidPrimitive {
    const length = Math.hypot(direction[0], direction[1], direction[2]);
    if (
        !center.every(Number.isFinite) ||
        !Number.isFinite(length) ||
        length <= 1e-8 ||
        !Number.isFinite(halfLength) ||
        halfLength <= 0 ||
        !Number.isFinite(radius) ||
        radius <= 0
    ) {
        throw new Error("[aquanova] pistol fluid hole requires a finite centre, direction, half-length, and radius");
    }
    const scale = halfLength / length;
    const axis: [number, number, number] = [direction[0] * scale, direction[1] * scale, direction[2] * scale];
    return {
        kind: "cylinder",
        a: [center[0] - axis[0], center[1] - axis[1], center[2] - axis[2]],
        b: [center[0] + axis[0], center[1] + axis[1], center[2] + axis[2]],
        radius,
    };
}

/** Preallocate inactive hole slots; subsequent writes replace the highest active hole. */
export function createFluidHoleRing(capacity: number): FluidHoleRing {
    if (!Number.isInteger(capacity) || capacity <= 0) {
        throw new Error(`[aquanova] fluid hole capacity must be a positive integer, received ${String(capacity)}`);
    }
    return {
        holes: Array.from({ length: capacity }, () => ({
            primitive: {
                kind: "cylinder" as const,
                a: [0, 0, 0] as const,
                b: [0, 0, 0] as const,
                radius: 0,
                active: false,
            },
            targetSolidSlot: 0,
        })),
    };
}

/** Fill an inactive slot, or replace the active hole whose cylinder centre is highest. */
export function addFluidHole(ring: FluidHoleRing, primitive: FluidPrimitive, targetSolidSlot: number): number {
    if (!Number.isInteger(targetSolidSlot) || targetSolidSlot < 0) {
        throw new Error(`[aquanova] fluid hole target must be a non-negative solid slot, received ${String(targetSolidSlot)}`);
    }
    let slot = ring.holes.findIndex((hole) => hole.primitive.active === false);
    if (slot < 0) {
        slot = 0;
        let highestY = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < ring.holes.length; i++) {
            const hole = ring.holes[i]!.primitive;
            const centerY = hole.b ? (hole.a[1] + hole.b[1]) * 0.5 : hole.a[1];
            if (centerY > highestY) {
                highestY = centerY;
                slot = i;
            }
        }
    }
    ring.holes[slot] = { primitive, targetSolidSlot };
    return slot;
}

/** Transform a local +Y hollow cylinder into the world-space axial primitive used by the solver. */
export function hollowCylinderPrimitiveForMatrix(
    world: ArrayLike<number>,
    start: readonly [number, number, number],
    height: number,
    innerRadius: number,
    outerRadius: number
): FluidPrimitive {
    const point = (x: number, y: number, z: number): [number, number, number] => [
        world[0]! * x + world[4]! * y + world[8]! * z + world[12]!,
        world[1]! * x + world[5]! * y + world[9]! * z + world[13]!,
        world[2]! * x + world[6]! * y + world[10]! * z + world[14]!,
    ];
    const a = point(start[0], start[1], start[2]);
    const b = point(start[0], start[1] + height, start[2]);
    const axisLength = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const x: [number, number, number] = [world[0]!, world[1]!, world[2]!];
    const y: [number, number, number] = [world[4]!, world[5]!, world[6]!];
    const z: [number, number, number] = [world[8]!, world[9]!, world[10]!];
    const xScale = Math.hypot(...x);
    const yScale = Math.hypot(...y);
    const zScale = Math.hypot(...z);
    if (axisLength <= 1e-8 || xScale <= 1e-8 || yScale <= 1e-8 || zScale <= 1e-8) {
        throw new Error("[aquanova] setCollisionShape.fluidSimShape resolves to a degenerate cylinder");
    }
    const dot = (left: readonly number[], right: readonly number[]): number => left[0]! * right[0]! + left[1]! * right[1]! + left[2]! * right[2]!;
    const tolerance = 1e-5;
    if (
        Math.abs(xScale - zScale) > Math.max(xScale, zScale) * tolerance ||
        Math.abs(dot(x, y)) > xScale * yScale * tolerance ||
        Math.abs(dot(x, z)) > xScale * zScale * tolerance ||
        Math.abs(dot(y, z)) > yScale * zScale * tolerance
    ) {
        throw new Error("[aquanova] setCollisionShape.fluidSimShape requires a mesh transform with equal X/Z scale and no shear");
    }
    const radialScale = (xScale + zScale) * 0.5;
    return {
        kind: "hollowCylinder",
        a,
        b,
        innerRadius: innerRadius * radialScale,
        radius: outerRadius * radialScale,
    };
}

/**
 * Re-express a world-space primitive relative to `centre`, so a moving body can re-pose it by adding
 * its live position back.
 *
 * The subtlety this exists to contain: `a` is always a POINT, but `b` is a point only for a capsule
 * or cylinder (the second axis end) — for a box it is HALF EXTENTS, a size, which must be left
 * alone. Re-basing a box's half extents on the centre silently produces a zero/negative-sized box,
 * which has no interior at all: fluid pours straight through it and debug overlays draw nothing.
 *
 * @param p - Primitive in world space.
 * @param centre - Origin of the local frame (typically the body's rest centre).
 * @returns A copy with point fields made relative; sizes, radii and rotations untouched.
 */
export function localizePrimitive(p: FluidPrimitive, centre: readonly [number, number, number]): FluidPrimitive {
    const rel = (v: readonly [number, number, number]): [number, number, number] => [v[0] - centre[0], v[1] - centre[1], v[2] - centre[2]];
    const out: FluidPrimitive = { ...p, a: rel(p.a) };
    if (p.b) {
        out.b = p.kind === "box" ? p.b : rel(p.b);
    }
    return out;
}

// ── CPU mirror ──────────────────────────────────────────────────────────────────────────────────
// The same maths as the WGSL below, so the distance field can be unit-tested and probed from the
// host without a GPU readback. Any change to one MUST be made to the other; the tests compare them
// against known distances precisely so a divergence shows up as a failure rather than as water
// quietly sinking through a floor.

/** Rotate `v` by the INVERSE of quaternion `q` (world → box local). */
function qRotInv(q: readonly [number, number, number, number], v: readonly [number, number, number]): [number, number, number] {
    const ux = -q[0],
        uy = -q[1],
        uz = -q[2],
        s = q[3];
    const d = ux * v[0] + uy * v[1] + uz * v[2];
    const uu = ux * ux + uy * uy + uz * uz;
    const cx = uy * v[2] - uz * v[1];
    const cy = uz * v[0] - ux * v[2];
    const cz = ux * v[1] - uy * v[0];
    const k = s * s - uu;
    return [2 * d * ux + k * v[0] + 2 * s * cx, 2 * d * uy + k * v[1] + 2 * s * cy, 2 * d * uz + k * v[2] + 2 * s * cz];
}

/** Signed distance to one primitive. Negative inside. `dt` advances the primitive along its velocity. */
export function primitiveSdf(p: FluidPrimitive, pt: readonly [number, number, number], dt = 0): number {
    if (p.active === false) return 1e9;
    const v = p.velocity ?? [0, 0, 0];
    const a: [number, number, number] = [p.a[0] + v[0] * dt, p.a[1] + v[1] * dt, p.a[2] + v[2] * dt];
    if (p.kind === "sphere") {
        return Math.hypot(pt[0] - a[0], pt[1] - a[1], pt[2] - a[2]) - (p.radius ?? 0);
    }
    if (p.kind === "box") {
        const h = p.b ?? [0, 0, 0];
        const l = qRotInv(p.rotation ?? [0, 0, 0, 1], [pt[0] - a[0], pt[1] - a[1], pt[2] - a[2]]);
        const ex = Math.abs(l[0]) - h[0],
            ey = Math.abs(l[1]) - h[1],
            ez = Math.abs(l[2]) - h[2];
        const outside = Math.hypot(Math.max(ex, 0), Math.max(ey, 0), Math.max(ez, 0));
        return outside + Math.min(Math.max(ex, Math.max(ey, ez)), 0);
    }
    const bb = p.b ?? [0, 0, 0];
    const b: [number, number, number] = [bb[0] + v[0] * dt, bb[1] + v[1] * dt, bb[2] + v[2] * dt];
    const r = p.radius ?? 0;
    const bax = b[0] - a[0],
        bay = b[1] - a[1],
        baz = b[2] - a[2];
    const pax = pt[0] - a[0],
        pay = pt[1] - a[1],
        paz = pt[2] - a[2];
    const baba = bax * bax + bay * bay + baz * baz;
    const paba = pax * bax + pay * bay + paz * baz;
    if (p.kind === "capsule") {
        const t = baba > 1e-12 ? Math.max(0, Math.min(1, paba / baba)) : 0;
        return Math.hypot(pax - bax * t, pay - bay * t, paz - baz * t) - r;
    }
    if (p.kind === "hollowCylinder") {
        const axisLength = Math.sqrt(baba);
        const along = paba / baba;
        const radial = Math.hypot(pax - bax * along, pay - bay * along, paz - baz * along);
        const inner = p.innerRadius ?? 0;
        const middle = (inner + r) * 0.5;
        const halfThickness = (r - inner) * 0.5;
        const radialBand = Math.abs(radial - middle) - halfThickness;
        const axialBand = Math.abs(along - 0.5) * axisLength - axisLength * 0.5;
        return Math.hypot(Math.max(radialBand, 0), Math.max(axialBand, 0)) + Math.min(Math.max(radialBand, axialBand), 0);
    }
    // Capped solid cylinder (Inigo Quilez): flat ends, unlike the capsule's rounded caps.
    if (baba < 1e-12) return Math.hypot(pax, pay, paz) - r;
    const px = pax * baba - bax * paba,
        py = pay * baba - bay * paba,
        pz = paz * baba - baz * paba;
    const x = Math.hypot(px, py, pz) - r * baba;
    const y = Math.abs(paba - baba * 0.5) - baba * 0.5;
    const x2 = x * x;
    const y2 = y * y * baba;
    const d = Math.max(x, y) < 0 ? -Math.min(x2, y2) : (x > 0 ? x2 : 0) + (y > 0 ? y2 : 0);
    return (Math.sign(d) * Math.sqrt(Math.abs(d))) / baba;
}

/** Union of a primitive set (min), matching the WGSL `primitivesSdf`. */
export function primitivesSdf(prims: readonly FluidPrimitive[], pt: readonly [number, number, number], dt = 0): number {
    let d = 1e9;
    for (const p of prims) d = Math.min(d, primitiveSdf(p, pt, dt));
    return d;
}

/** Subtract each hole from only its target solid before unioning the solids. */
export function subtractedPrimitivesSdf(solids: readonly FluidPrimitive[], holes: readonly FluidHole[], pt: readonly [number, number, number], dt = 0): number {
    let result = 1e9;
    for (let solidSlot = 0; solidSlot < solids.length; solidSlot++) {
        let solidDistance = primitiveSdf(solids[solidSlot]!, pt, dt);
        for (const hole of holes) {
            if (hole.targetSolidSlot === solidSlot) {
                solidDistance = Math.max(solidDistance, -primitiveSdf(hole.primitive, pt, dt));
            }
        }
        result = Math.min(result, solidDistance);
    }
    return result;
}

// ── WGSL ────────────────────────────────────────────────────────────────────────────────────────
// Reads the primitive list out of `sceneSdfGrid` — the storage buffer the solvers already bind for
// `SceneSdfSpec.sdfGrid`. Reusing that binding is why this needs no engine change at all.

export const PRIMITIVES_WGSL = /* wgsl */ `
fn primQRotInv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = -q.xyz;
    return 2.0 * dot(u, v) * u + (q.w * q.w - dot(u, u)) * v + 2.0 * q.w * cross(u, v);
}

fn primSdf(o: u32, pt: vec3<f32>, dt: f32) -> f32 {
    let kind = sceneSdfGrid[o];
    let vel = vec3<f32>(sceneSdfGrid[o + 12u], sceneSdfGrid[o + 13u], sceneSdfGrid[o + 14u]);
    let a = vec3<f32>(sceneSdfGrid[o + 1u], sceneSdfGrid[o + 2u], sceneSdfGrid[o + 3u]) + vel * dt;
    if (kind < 0.5) {
        let h = vec3<f32>(sceneSdfGrid[o + 4u], sceneSdfGrid[o + 5u], sceneSdfGrid[o + 6u]);
        let q = vec4<f32>(sceneSdfGrid[o + 8u], sceneSdfGrid[o + 9u], sceneSdfGrid[o + 10u], sceneSdfGrid[o + 11u]);
        let e = abs(primQRotInv(q, pt - a)) - h;
        return length(max(e, vec3<f32>(0.0))) + min(max(e.x, max(e.y, e.z)), 0.0);
    }
    let r = sceneSdfGrid[o + 7u];
    if (kind < 1.5) {
        return length(pt - a) - r;
    }
    let b = vec3<f32>(sceneSdfGrid[o + 4u], sceneSdfGrid[o + 5u], sceneSdfGrid[o + 6u]) + vel * dt;
    let ba = b - a;
    let pa = pt - a;
    let baba = max(dot(ba, ba), 1e-12);
    let paba = dot(pa, ba);
    if (kind < 2.5) {
        return length(pa - ba * clamp(paba / baba, 0.0, 1.0)) - r;
    }
    if (kind > 3.5) {
        let axisLength = sqrt(baba);
        let along = paba / baba;
        let radial = length(pa - ba * along);
        let inner = sceneSdfGrid[o + 8u];
        let middle = (inner + r) * 0.5;
        let halfThickness = (r - inner) * 0.5;
        let e = vec2<f32>(
            abs(radial - middle) - halfThickness,
            abs(along - 0.5) * axisLength - axisLength * 0.5
        );
        return length(max(e, vec2<f32>(0.0))) + min(max(e.x, e.y), 0.0);
    }
    let x = length(pa * baba - ba * paba) - r * baba;
    let y = abs(paba - baba * 0.5) - baba * 0.5;
    let x2 = x * x;
    let y2 = y * y * baba;
    var d = 0.0;
    if (max(x, y) < 0.0) {
        d = -min(x2, y2);
    } else {
        d = select(0.0, x2, x > 0.0) + select(0.0, y2, y > 0.0);
    }
    return sign(d) * sqrt(abs(d)) / baba;
}

/** Union of every packed primitive. Returns a large positive value when the set is empty. */
fn primitivesSdf(pt: vec3<f32>, dt: f32) -> f32 {
    var d = 1e9;
    let n = u32(sceneSdfGrid[0]);
    for (var i = 0u; i < n; i = i + 1u) {
        let o = ${PRIM_HEADER}u + i * ${PRIM_STRIDE}u;
        if (sceneSdfGrid[o + ${PRIM_ACTIVE_OFFSET}u] > 0.5) {
            d = min(d, primSdf(o, pt, dt));
        }
    }
    return d;
}

/** Union the solids after applying each subtraction only to its target solid slot. */
fn carvedPrimitivesSdf(pt: vec3<f32>, dt: f32) -> f32 {
    var d = 1e9;
    let solidCount = u32(sceneSdfGrid[0]);
    let holeCount = u32(sceneSdfGrid[1]);
    for (var solidSlot = 0u; solidSlot < solidCount; solidSlot = solidSlot + 1u) {
        let solidOffset = ${PRIM_HEADER}u + solidSlot * ${PRIM_STRIDE}u;
        if (sceneSdfGrid[solidOffset + ${PRIM_ACTIVE_OFFSET}u] > 0.5) {
            var solidDistance = primSdf(solidOffset, pt, dt);
            for (var holeSlot = 0u; holeSlot < holeCount; holeSlot = holeSlot + 1u) {
                let holeOffset = ${PRIM_HEADER}u + (solidCount + holeSlot) * ${PRIM_STRIDE}u;
                if (sceneSdfGrid[holeOffset + ${PRIM_ACTIVE_OFFSET}u] > 0.5 && u32(sceneSdfGrid[holeOffset + 8u]) == solidSlot) {
                    solidDistance = max(solidDistance, -primSdf(holeOffset, pt, dt));
                }
            }
            d = min(d, solidDistance);
        }
    }
    return d;
}`;
