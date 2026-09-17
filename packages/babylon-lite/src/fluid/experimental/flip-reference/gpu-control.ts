/** GPU-owned frame clock, failure latch, indirect scheduling, and completed-state publication. */
export const FLIP_REFERENCE_GPU_CONTROL_WGSL = /* wgsl */ `
@group(0) @binding(13) var<uniform> frameSettings: vec4<f32>;
@group(0) @binding(14) var<storage, read_write> dispatchArgs: array<u32>;
@group(0) @binding(15) var<storage, read_write> publishedPositions: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> publishedVelocities: array<vec4<f32>>;
@group(0) @binding(17) var<storage, read_write> publishedSpeeds: array<f32>;
@group(0) @binding(18) var<storage, read_write> publishedDraw: array<u32>;

fn runtimeFailure(bit: u32) {
    atomicOr(&runtime[1], bit);
    atomicStore(&runtime[2], 0u);
}
@compute @workgroup_size(128)
fn initializeGpuMaximumSpeed(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < particleCount()) { atomicMax(&runtime[7], bitcast<u32>(speeds[id.x])); }
}
@compute @workgroup_size(1)
fn beginGpuFrame() {
    atomicStore(&runtime[2], 0u);
    atomicStore(&runtime[3], 0u);
    atomicStore(&runtime[5], bitcast<u32>(frameSettings.x));
    atomicStore(&runtime[6], bitcast<u32>(frameSettings.x));
}
@compute @workgroup_size(1)
fn beginGpuSubstep() {
    atomicStore(&runtime[2], 0u);
    let remaining = bitcast<f32>(atomicLoad(&runtime[6]));
    if (atomicLoad(&runtime[1]) != 0u || remaining <= 0.0) { return; }
    let required = max(1.0, frameSettings.z - f32(atomicLoad(&runtime[3])));
    var dt = min(remaining / required, frameSettings.y);
    let speed = bitcast<f32>(atomicLoad(&runtime[7]));
    if (frameSettings.w > 0.0 && speed > 0.0) { dt = min(dt, frameSettings.w * params.origin.w / speed); }
    if (!(dt > 0.0) || !(dt < 3.0e38) || remaining - dt == remaining) {
        runtimeFailure(64u);
        return;
    }
    atomicStore(&runtime[2], 1u);
    atomicStore(&runtime[4], bitcast<u32>(dt));
    for (var i = 8u; i <= 12u; i++) { atomicStore(&runtime[i], 0u); }
    for (var i = 0u; i < 8u; i++) { atomicStore(&lists[statusIndex(i)], 0u); }
    pcg[controlIndex()] = vec4<f32>(0.0);
    pcg[controlIndex() + 1u] = vec4<f32>(0.0);
}
@compute @workgroup_size(1)
fn updateGpuDispatch() {
    if (atomicLoad(&runtime[2]) != 0u && (pcg[controlIndex()].w != 0.0 || pcg[controlIndex() + 1u].w != 0.0)) {
        runtimeFailure(16u);
    }
    let enabled = atomicLoad(&runtime[1]) == 0u && atomicLoad(&runtime[2]) != 0u && particleCount() > 0u;
    let vertices = (params.grid.x + 1u) * (params.grid.y + 1u) * (params.grid.z + 1u);
    let counts = array<u32, 6>(params.counts.x, params.counts.y, vertices, particleCount(), 1u, max(params.counts.x, max(8u, params.particles.z)));
    for (var kind = 0u; kind < 6u; kind++) {
        dispatchArgs[kind * 3u] = select(0u, (counts[kind] + 127u) / 128u, enabled);
        dispatchArgs[kind * 3u + 1u] = 1u;
        dispatchArgs[kind * 3u + 2u] = 1u;
    }
}
fn particleFailureBits() -> u32 {
    var bits = 0u;
    if (atomicLoad(&lists[statusIndex(0u)]) != 0u) { bits |= 1u; }
    if (atomicLoad(&lists[statusIndex(1u)]) != 0u) { bits |= 2u; }
    if (atomicLoad(&lists[statusIndex(5u)]) != 0u) { bits |= 4u; }
    if (atomicLoad(&lists[statusIndex(6u)]) != 0u) { bits |= 8u; }
    return bits;
}
@compute @workgroup_size(1)
fn finishGpuSubstep() {
    if (atomicLoad(&runtime[2]) == 0u || atomicLoad(&runtime[1]) != 0u) { return; }
    let bits = particleFailureBits();
    if (bits != 0u) { runtimeFailure(bits); return; }
    let before = particleCount();
    var count = before;
    var speed = bitcast<f32>(atomicLoad(&lists[statusIndex(4u)]));
    var inside = 0u;
    var extreme = 0u;
    if (params.particles.w != 0u && before > 0u) {
        count = atomicLoad(&particleState[removalControl(0u)]);
        inside = atomicLoad(&particleState[removalControl(1u)]);
        extreme = atomicLoad(&particleState[removalControl(2u)]);
        speed = bitcast<f32>(atomicLoad(&particleState[removalControl(6u)]));
    }
    if (count + inside + extreme != before || count > params.particles.x || !(speed >= 0.0) || !(speed < 3.0e38)) {
        runtimeFailure(1u);
        return;
    }
    atomicStore(&runtime[0], count);
    atomicStore(&runtime[7], bitcast<u32>(speed));
    atomicAdd(&runtime[14], inside);
    atomicAdd(&runtime[15], extreme);
    let remaining = bitcast<f32>(atomicLoad(&runtime[6]));
    atomicStore(&runtime[6], bitcast<u32>(max(0.0, remaining - stepDt())));
    atomicAdd(&runtime[3], 1u);
    atomicStore(&runtime[2], 0u);
}
@compute @workgroup_size(1)
fn finishGpuFrame() {
    if (atomicLoad(&runtime[1]) != 0u) { return; }
    let skipped = max(0.0, bitcast<f32>(atomicLoad(&runtime[6])));
    let consumed = max(0.0, frameSettings.x - skipped);
    let totalSkipped = bitcast<f32>(atomicLoad(&runtime[17]));
    atomicStore(&runtime[17], bitcast<u32>(totalSkipped + skipped));
    atomicStore(&runtime[18], bitcast<u32>(skipped));
    atomicStore(&runtime[19], bitcast<u32>(consumed));
    atomicAdd(&runtime[13], 1u);
    let elapsed = bitcast<f32>(atomicLoad(&runtime[16]));
    atomicStore(&runtime[16], bitcast<u32>(elapsed + consumed));
}
@compute @workgroup_size(128)
fn publishGpuParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (atomicLoad(&runtime[1]) != 0u) { return; }
    if (id.x == 0u) {
        publishedDraw[0] = 6u;
        publishedDraw[1] = particleCount();
        publishedDraw[2] = 0u;
        publishedDraw[3] = 0u;
    }
    if (id.x >= params.particles.x) { return; }
    if (id.x < particleCount()) {
        publishedPositions[id.x] = positions[id.x];
        publishedVelocities[id.x] = velocities[id.x];
        publishedSpeeds[id.x] = speeds[id.x];
    } else {
        publishedPositions[id.x] = vec4<f32>(0.0);
        publishedVelocities[id.x] = vec4<f32>(0.0);
        publishedSpeeds[id.x] = 0.0;
    }
}
`;

export const FLIP_REFERENCE_GPU_CONTROL_BINDINGS: Readonly<Record<string, readonly number[]>> = {
    initializeGpuMaximumSpeed: [3, 11],
    beginGpuFrame: [11, 13],
    beginGpuSubstep: [0, 6, 8, 11, 13],
    updateGpuDispatch: [0, 8, 11, 14],
    finishGpuSubstep: [0, 6, 10, 11],
    finishGpuFrame: [11, 13],
    publishGpuParticles: [0, 1, 2, 3, 11, 15, 16, 17, 18],
};
