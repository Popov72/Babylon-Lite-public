// Ready-made external force field for the fluid solvers: an interactive push that
// accelerates particles within `radius` of a ray (origin, dir) along `push`, with
// linear falloff to the ray edge. This is the exact "mouse-stir" behaviour that
// used to be hardcoded in both solvers, now packaged as an opt-in `ForceFieldSpec`
// so the core solvers stay app-agnostic and pay nothing when no force is injected.
//
// The caller owns one small uniform buffer (12 floats) and drives the push via
// `setRay` / `clear`; the solver compiles `spec.wgsl` into its integration pass
// and binds `spec.buffer`.

import type { ForceFieldSpec } from "./sim-common.js";

/** @internal */
export interface RayForce {
    readonly spec: ForceFieldSpec;
    /** Push particles within `radius` of the ray (origin,dir) along `push`,
     *  accel-scaled with linear falloff. accel = 0 disables (no push this step). */
    setRay(origin: [number, number, number], dir: [number, number, number], push: [number, number, number], radius: number, accel: number): void;
    /** Disable the push (accel = 0). */
    clear(): void;
    dispose(): void;
}

// forceO = (origin.xyz, radius), forceD = (dir.xyz, accel), forceP = (push.xyz, pad).
const FORCE_STRUCT = "struct ForceFieldParams { forceO: vec4<f32>, forceD: vec4<f32>, forceP: vec4<f32>, };";

// Ray-push velocity delta (dt-scaled). Reproduces the removed in-solver math:
// old = push * (accel * (1 - dist/radius) * dt), falloff linear, gated on the
// ray-relative distance being inside the cylinder radius and ahead of the origin.
const FORCE_WGSL = /* wgsl */ `
fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> {
    let accel = forceFieldParams.forceD.w;
    if (accel <= 0.0) { return vec3<f32>(0.0); }
    let o = forceFieldParams.forceO.xyz;
    let dir = forceFieldParams.forceD.xyz;
    let t = dot(pos - o, dir);
    if (t <= 0.0) { return vec3<f32>(0.0); }
    let dist = length(pos - (o + t * dir));
    if (dist >= forceFieldParams.forceO.w) { return vec3<f32>(0.0); }
    return forceFieldParams.forceP.xyz * (accel * (1.0 - dist / forceFieldParams.forceO.w) * dt);
}`;

/** @internal */
export function createRayForce(device: GPUDevice): RayForce {
    const buffer = device.createBuffer({ label: "ray-force", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new Float32Array(12);

    function upload(): void {
        device.queue.writeBuffer(buffer, 0, data);
    }
    // Start disabled (accel = 0) so the field is a no-op until the first setRay.
    upload();

    return {
        spec: { struct: FORCE_STRUCT, wgsl: FORCE_WGSL, buffer },
        setRay(origin, dir, push, radius, accel): void {
            data[0] = origin[0];
            data[1] = origin[1];
            data[2] = origin[2];
            data[3] = radius;
            data[4] = dir[0];
            data[5] = dir[1];
            data[6] = dir[2];
            data[7] = accel;
            data[8] = push[0];
            data[9] = push[1];
            data[10] = push[2];
            data[11] = 0;
            upload();
        },
        clear(): void {
            data[7] = 0; // accel = 0 disables the push
            upload();
        },
        dispose(): void {
            buffer.destroy();
        },
    };
}
