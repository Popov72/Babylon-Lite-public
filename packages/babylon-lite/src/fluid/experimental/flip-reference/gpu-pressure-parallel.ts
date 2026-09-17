export const FLIP_REFERENCE_PARALLEL_PRESSURE_MIN_PARTICLES = 65536;
export const FLIP_REFERENCE_PARALLEL_PRESSURE_ITERATIONS = 128;
export const FLIP_REFERENCE_PARALLEL_PRESSURE_DISTRIBUTED_PARTICLES = 262144;

/** Bounded GPU PCG prefix; shares the cached operator, controls and idle compaction scratch. */
export const FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_WGSL = /* wgsl */ `
var<workgroup> parallelPressureEnabled: u32;
var<workgroup> parallelPressureControl: array<vec4<f32>, 2>;

fn parallelPressureActive() -> bool {
    return atomicLoad(&runtime[2]) != 0u && atomicLoad(&runtime[1]) == 0u
        && pcg[controlIndex()].w != 0.0 && pcg[controlIndex() + 1u].w == 0.0;
}
fn parallelPressureDispatch(enabled: bool) {
    dispatchArgs[0] = select(0u, (atomicLoad(&runtime[8]) + 127u) / 128u, enabled);
}
@compute @workgroup_size(1)
fn beginParallelPressure() {
    parallelPressureDispatch(parallelPressureActive());
}
@compute @workgroup_size(1)
fn finishParallelPressure() {
    let enabled = atomicLoad(&runtime[2]) != 0u && atomicLoad(&runtime[1]) == 0u && particleCount() > 0u;
    dispatchArgs[0] = select(0u, (params.counts.x + 127u) / 128u, enabled);
    dispatchArgs[12] = select(0u, 1u, enabled);
}
fn nextParallelPressureDirection(index: u32, beta: f32) -> f32 {
    return pcg[index].y * gpuPressureRows[index].inverse + beta * pcg[index].z;
}
@compute @workgroup_size(128)
fn continueParallelPressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= atomicLoad(&runtime[8]) || pcg[controlIndex()].w == 0.0) { return; }
    let index = atomicLoad(&lists[gid.x]);
    pcg[index].z = nextParallelPressureDirection(index, pcg[controlIndex() + 1u].y);
}
@compute @workgroup_size(128)
fn applyParallelPressure(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>
) {
    var products = vec4<f32>(0.0);
    if (gid.x < atomicLoad(&runtime[8])) {
        let index = atomicLoad(&lists[gid.x]);
        let row = gpuPressureRows[index];
        let beta = pcg[controlIndex() + 1u].y;
        let direction = nextParallelPressureDirection(index, beta);
        var product = row.diagonal * direction;
        for (var axis = 0u; axis < 3u; axis++) {
            if (row.positive[axis] > 0.0) {
                product -= row.positive[axis] * nextParallelPressureDirection(row.positiveNeighbor[axis], beta);
            }
            if (row.negative[axis] > 0.0) {
                product -= row.negative[axis] * nextParallelPressureDirection(row.negativeNeighbor[axis], beta);
            }
        }
        let dotProduct = direction * product;
        particleScratch[gid.x].x = direction;
        pcg[index].w = product;
        if (pressureFinite(direction) && pressureFinite(product) && pressureFinite(dotProduct)) {
            products.z = dotProduct;
        } else {
            products.w = 1.0;
        }
    }
    sums[local] = products;
    reduceLocal(local);
    if (local == 0u) {
        pcg[params.counts.x + group.x] = sums[0];
    }
}
fn reduceParallelPressure(local: u32) {
    var value = vec4<f32>(0.0);
    let groups = (atomicLoad(&runtime[8]) + 127u) / 128u;
    for (var index = local; index < groups; index += 128u) {
        value += pcg[params.counts.x + index];
    }
    sums[local] = value;
    reduceLocal(local);
}
@compute @workgroup_size(128)
fn alphaParallelPressure(@builtin(local_invocation_index) local: u32) {
    if (local == 0u) {
        parallelPressureEnabled = u32(parallelPressureActive());
    }
    if (workgroupUniformLoad(&parallelPressureEnabled) == 0u) {
        if (local == 0u) { parallelPressureDispatch(false); }
        return;
    }
    reduceParallelPressure(local);
    if (local != 0u) { return; }
    let total = sums[0];
    let control = pcg[controlIndex()];
    let iterations = u32(pcg[controlIndex() + 1u].z);
    if (total.w != 0.0 || !(total.z > 0.0) || !pressureFinite(total.z)) {
        failGpuPressure(local, select(1.0, 2.0, total.w != 0.0 || !pressureFinite(total.z)), control.x, control.y, control.z, iterations);
        parallelPressureDispatch(false);
        return;
    }
    let alpha = control.x / total.z;
    if (!pressureFinite(alpha)) {
        failGpuPressure(local, 2.0, control.x, control.y, control.z, iterations);
        parallelPressureDispatch(false);
        return;
    }
    pcg[controlIndex() + 1u].x = alpha;
}
fn advanceParallelPressure(item: u32, alpha: f32) -> vec4<f32> {
    let index = atomicLoad(&lists[item]);
    let direction = particleScratch[item].x;
    let value = pcg[index];
    let nextX = pressurePairAdd(readPressurePair(index), pressurePairScale(vec2<f32>(direction, 0.0), alpha));
    let residual = value.y - alpha * value.w;
    let preconditioned = residual * gpuPressureRows[index].inverse;
    let rz = residual * preconditioned;
    let squared = residual * residual;
    pcg[index].x = nextX.x;
    pressureLow[index] = nextX.y;
    pcg[index].y = residual;
    pcg[index].z = direction;
    if (pressureFinite(nextX.x) && pressureFinite(nextX.y) && pressureFinite(residual) && pressureFinite(preconditioned) && pressureFinite(rz) && pressureFinite(squared)) {
        return vec4<f32>(rz, squared, 0.0, 0.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
@compute @workgroup_size(128)
fn updateParallelPressure(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>
) {
    var products = vec4<f32>(0.0);
    if (gid.x < atomicLoad(&runtime[8])) {
        products = advanceParallelPressure(gid.x, pcg[controlIndex() + 1u].x);
    }
    sums[local] = products;
    reduceLocal(local);
    if (local == 0u) {
        pcg[params.counts.x + group.x] = sums[0];
    }
}
@compute @workgroup_size(128)
fn betaParallelPressure(@builtin(local_invocation_index) local: u32) {
    if (local == 0u) {
        parallelPressureEnabled = u32(parallelPressureActive());
    }
    if (workgroupUniformLoad(&parallelPressureEnabled) == 0u) {
        if (local == 0u) { parallelPressureDispatch(false); }
        return;
    }
    reduceParallelPressure(local);
    if (local != 0u) { return; }
    let total = sums[0];
    let control = pcg[controlIndex()];
    let iterations = u32(pcg[controlIndex() + 1u].z) + 1u;
    if (total.w != 0.0 || !(total.x >= 0.0) || !(total.y >= 0.0) || !pressureFinite(total.x) || !pressureFinite(total.y)) {
        failGpuPressure(local, 2.0, control.x, control.y, control.z, iterations);
        parallelPressureDispatch(false);
        return;
    }
    let beta = total.x / control.x;
    if (!(control.x > 0.0) || !pressureFinite(beta)) {
        failGpuPressure(local, 2.0, control.x, control.y, control.z, iterations);
        parallelPressureDispatch(false);
        return;
    }
    let needsIteration = total.y > max(params.tolerances.y, params.tolerances.x * control.z);
    pcg[controlIndex()] = vec4<f32>(total.x, total.y, control.z, f32(needsIteration));
    pcg[controlIndex() + 1u].y = beta;
    pcg[controlIndex() + 1u].z = f32(iterations);
    parallelPressureDispatch(needsIteration);
}

@compute @workgroup_size(pressureWorkgroupSize)
fn finishParallelPressureIteration(@builtin(local_invocation_index) local: u32) {
    if (local == 0u) {
        pressureState[0] = u32(parallelPressureActive());
        pressureState[1] = atomicLoad(&runtime[8]);
        parallelPressureControl[0] = pcg[controlIndex()];
        parallelPressureControl[1] = pcg[controlIndex() + 1u];
    }
    let enabled = workgroupUniformLoad(&pressureState[0]);
    if (enabled == 0u) {
        if (local == 0u) { parallelPressureDispatch(false); }
        return;
    }
    let activeCount = workgroupUniformLoad(&pressureState[1]);
    let control = workgroupUniformLoad(&parallelPressureControl[0]);
    let previous = workgroupUniformLoad(&parallelPressureControl[1]);
    let iterations = u32(previous.z);
    var products = vec4<f32>(0.0);
    let groups = (activeCount + 127u) / 128u;
    for (var group = local; group < groups; group += pressureWorkgroupSize) {
        products += pcg[params.counts.x + group];
    }
    let dotProduct = reducePressure(local, products);
    if (dotProduct.w != 0.0 || !(dotProduct.z > 0.0) || !pressureFinite(dotProduct.z)) {
        failGpuPressure(local, select(1.0, 2.0, dotProduct.w != 0.0 || !pressureFinite(dotProduct.z)), control.x, control.y, control.z, iterations);
        if (local == 0u) { parallelPressureDispatch(false); }
        return;
    }
    let alpha = control.x / dotProduct.z;
    if (!pressureFinite(alpha)) {
        failGpuPressure(local, 2.0, control.x, control.y, control.z, iterations);
        if (local == 0u) { parallelPressureDispatch(false); }
        return;
    }
    products = vec4<f32>(0.0);
    for (var item = local; item < activeCount; item += pressureWorkgroupSize) {
        products += advanceParallelPressure(item, alpha);
    }
    storageBarrier();
    let total = reducePressure(local, products);
    if (total.w != 0.0 || !(total.x >= 0.0) || !(total.y >= 0.0) || !pressureFinite(total.x) || !pressureFinite(total.y)) {
        failGpuPressure(local, 2.0, control.x, control.y, control.z, iterations + 1u);
        if (local == 0u) { parallelPressureDispatch(false); }
        return;
    }
    let beta = total.x / control.x;
    if (!(control.x > 0.0) || !pressureFinite(beta)) {
        failGpuPressure(local, 2.0, control.x, control.y, control.z, iterations + 1u);
        if (local == 0u) { parallelPressureDispatch(false); }
        return;
    }
    if (local == 0u) {
        let needsIteration = total.y > max(params.tolerances.y, params.tolerances.x * control.z);
        pcg[controlIndex()] = vec4<f32>(total.x, total.y, control.z, f32(needsIteration));
        pcg[controlIndex() + 1u] = vec4<f32>(alpha, beta, f32(iterations + 1u), 0.0);
        parallelPressureDispatch(needsIteration);
    }
}
`;

export const FLIP_REFERENCE_GPU_PARALLEL_PRESSURE_BINDINGS: Readonly<Record<string, readonly number[]>> = {
    beginParallelPressure: [0, 8, 11, 14],
    finishParallelPressure: [0, 11, 14],
    continueParallelPressure: [0, 6, 8, 11, 19],
    applyParallelPressure: [0, 6, 8, 9, 11, 19],
    alphaParallelPressure: [0, 8, 11, 14],
    updateParallelPressure: [0, 6, 8, 9, 11, 19, 21],
    betaParallelPressure: [0, 8, 11, 14],
    finishParallelPressureIteration: [0, 6, 8, 9, 11, 14, 19, 21],
};
