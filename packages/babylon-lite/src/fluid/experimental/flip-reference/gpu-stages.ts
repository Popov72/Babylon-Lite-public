/** Independent operations share a dispatch; the oracle uses the same helper bodies separately. */
export const FLIP_REFERENCE_GPU_STAGES_WGSL = /* wgsl */ `
@compute @workgroup_size(128)
fn clearGpuParticleStage(@builtin(global_invocation_id) gid: vec3<u32>) {
    clearParticleLists(gid.x);
    if (params.particles.w != 0u) { clearParticleRemoval(gid.x); }
}
fn gpuConditionedFace(c: vec3<i32>, axis: u32) -> bool {
    let other = c - axisVector(axis);
    var conditioned = false;
    if (inGrid(c)) { conditioned = cells[cellIndex(c)].geometry.z != 0.0; }
    if (inGrid(other)) { conditioned = conditioned || cells[cellIndex(other)].geometry.z != 0.0; }
    return conditioned;
}
@compute @workgroup_size(128)
fn conditionGpuMatrix(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (atomicLoad(&runtime[9]) == 0u) { return; }
    if (gid.x < params.counts.y) {
        let face = faceCoordinate(gid.x);
        if (gpuConditionedFace(face.cell, face.axis)) { faces[gid.x].solid = 0.0; }
    }
    if (gid.x >= params.counts.x || cells[gid.x].state.w == 0.0) { return; }
    let c = cellCoordinate(gid.x);
    let volume = cells[gid.x].geometry.x;
    var rhs = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        for (var sign = -1; sign <= 1; sign += 2) {
            var fcoord = c;
            if (sign > 0) { fcoord += axisVector(axis); }
            let index = faceIndex(fcoord, axis);
            let area = faces[index].area;
            let velocity = faces[index].velocity;
            var solid = 0.0;
            if (!gpuConditionedFace(fcoord, axis)) { solid = faces[index].solid; }
            rhs += f32(sign) * ((area - volume) * solid - area * velocity);
        }
    }
    cells[gid.x].state.z = rhs;
}
`;

export const FLIP_REFERENCE_GPU_STAGES_BINDINGS: Readonly<Record<string, readonly number[]>> = {
    clearGpuParticleStage: [0, 6, 10],
    conditionGpuMatrix: [0, 4, 5, 11],
};
