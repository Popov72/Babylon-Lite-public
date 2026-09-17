import type { ForceFieldSpec } from "../../core/sim-common.js";

/** Per-particle forces enter before P2G, with the actual GPU-selected physical substep duration. */
export function flipReferenceForceWgsl(spec: ForceFieldSpec): string {
    return /* wgsl */ `
${spec.struct}
@group(0) @binding(0) var<uniform> forceFieldParams: ForceFieldParams;
@group(0) @binding(1) var<storage, read> forcePositions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> forceVelocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> forceRuntime: array<atomic<u32>>;
${spec.wgsl}
@compute @workgroup_size(128)
fn applyFlipReferenceForce(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= atomicLoad(&forceRuntime[0])) { return; }
    let dt = bitcast<f32>(atomicLoad(&forceRuntime[4]));
    let previous = forceVelocities[id.x].xyz;
    let velocity = previous + externalForce(forcePositions[id.x].xyz, previous, dt);
    if (!all(abs(velocity) < vec3<f32>(3.0e38)) || !(length(velocity) < 3.0e38)) {
        atomicOr(&forceRuntime[1], 1u);
        atomicStore(&forceRuntime[2], 0u);
        return;
    }
    forceVelocities[id.x] = vec4<f32>(velocity, 0.0);
}`;
}
