const GPU_COMPONENT_BYTES = 32;

/** @internal Eight atomic u32 fields per pressure cell. */
export function flipReferenceGpuComponentBytes(cells: number): number {
    return cells * GPU_COMPONENT_BYTES;
}

/**
 * GPU pressure-component processing. Dispatch order:
 * buildMatrix, initialize, link, aggregate, optional pocket marking and solid conditioning, rebuildMatrix,
 * sealed flux sum, gauge finalization, then pressure initialization and solve.
 */
export const FLIP_REFERENCE_GPU_COMPONENT_WGSL = /* wgsl */ `
struct GpuComponent {
    conditioningParent: atomic<u32>,
    gaugeParent: atomic<u32>,
    conditioningCount: atomic<u32>,
    gaugeCount: atomic<u32>,
    conditioningAir: atomic<u32>,
    gaugeAir: atomic<u32>,
    fluxBits: atomic<u32>,
    rhsSquaredBits: atomic<u32>,
}
@group(0) @binding(12) var<storage, read_write> gpuComponents: array<GpuComponent>;

fn gpuComponentsEnabled() -> bool {
    return atomicLoad(&runtime[2]) != 0u && atomicLoad(&runtime[1]) == 0u;
}
fn conditioningRoot(index: u32) -> u32 {
    var root = index;
    loop {
        let parent = atomicLoad(&gpuComponents[root].conditioningParent);
        if (parent == root) { return root; }
        root = parent;
    }
}
fn gaugeRoot(index: u32) -> u32 {
    var root = index;
    loop {
        let parent = atomicLoad(&gpuComponents[root].gaugeParent);
        if (parent == root) { return root; }
        root = parent;
    }
}
fn unionConditioning(a: u32, b: u32) {
    loop {
        let aRoot = conditioningRoot(a);
        let bRoot = conditioningRoot(b);
        if (aRoot == bRoot) { return; }
        let higher = max(aRoot, bRoot);
        let lower = min(aRoot, bRoot);
        if (atomicCompareExchangeWeak(&gpuComponents[higher].conditioningParent, higher, lower).exchanged) { return; }
    }
}
fn unionGauge(a: u32, b: u32) {
    loop {
        let aRoot = gaugeRoot(a);
        let bRoot = gaugeRoot(b);
        if (aRoot == bRoot) { return; }
        let higher = max(aRoot, bRoot);
        let lower = min(aRoot, bRoot);
        if (atomicCompareExchangeWeak(&gpuComponents[higher].gaugeParent, higher, lower).exchanged) { return; }
    }
}
fn compressConditioning(index: u32) -> u32 {
    let root = conditioningRoot(index);
    var current = index;
    loop {
        let parent = atomicLoad(&gpuComponents[current].conditioningParent);
        if (parent == current) { break; }
        atomicStore(&gpuComponents[current].conditioningParent, root);
        current = parent;
    }
    return root;
}
fn compressGauge(index: u32) -> u32 {
    let root = gaugeRoot(index);
    var current = index;
    loop {
        let parent = atomicLoad(&gpuComponents[current].gaugeParent);
        if (parent == current) { break; }
        atomicStore(&gpuComponents[current].gaugeParent, root);
        current = parent;
    }
    return root;
}
fn finiteComponentValue(value: f32) -> bool {
    return value == value && abs(value) < 3.0e38;
}
fn atomicAddFinite(accumulator: ptr<storage, atomic<u32>, read_write>, value: f32) -> bool {
    var oldBits = atomicLoad(accumulator);
    loop {
        let oldValue = bitcast<f32>(oldBits);
        let newValue = oldValue + value;
        if (!finiteComponentValue(oldValue) || !finiteComponentValue(newValue)) {
            runtimeFailure(16u);
            return false;
        }
        let result = atomicCompareExchangeWeak(accumulator, oldBits, bitcast<u32>(newValue));
        if (result.exchanged) { return true; }
        oldBits = result.old_value;
    }
}
fn componentCell(activeIndex: u32) -> u32 {
    return atomicLoad(&lists[activeIndex]);
}

@compute @workgroup_size(128)
fn initializeGpuComponents(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!gpuComponentsEnabled() || gid.x >= params.counts.x) { return; }
    cells[gid.x].geometry.z = 0.0;
    cells[gid.x].positive.w = 0.0;
    if (cells[gid.x].state.w == 0.0) { return; }
    atomicStore(&gpuComponents[gid.x].conditioningParent, gid.x);
    atomicStore(&gpuComponents[gid.x].gaugeParent, gid.x);
    atomicStore(&gpuComponents[gid.x].conditioningCount, 0u);
    atomicStore(&gpuComponents[gid.x].gaugeCount, 0u);
    atomicStore(&gpuComponents[gid.x].conditioningAir, 0u);
    atomicStore(&gpuComponents[gid.x].gaugeAir, 0u);
    atomicStore(&gpuComponents[gid.x].fluxBits, bitcast<u32>(0.0));
    atomicStore(&gpuComponents[gid.x].rhsSquaredBits, bitcast<u32>(0.0));
    let activeIndex = atomicAdd(&runtime[8], 1u);
    atomicStore(&lists[activeIndex], gid.x);
}

@compute @workgroup_size(128)
fn linkGpuComponents(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!gpuComponentsEnabled() || gid.x >= atomicLoad(&runtime[8])) { return; }
    let index = componentCell(gid.x);
    let c = cellCoordinate(index);
    let row = cells[index];
    for (var axis = 0u; axis < 3u; axis++) {
        let neighborCoordinate = c + axisVector(axis);
        if (!inGrid(neighborCoordinate)) { continue; }
        let neighbor = cellIndex(neighborCoordinate);
        if (cells[neighbor].state.w == 0.0) { continue; }
        let coefficient = row.positive[axis];
        if (coefficient >= 1.0e-6) { unionConditioning(index, neighbor); }
        if (coefficient > 0.0) { unionGauge(index, neighbor); }
    }
}

@compute @workgroup_size(128)
fn aggregateGpuComponents(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!gpuComponentsEnabled() || gid.x >= atomicLoad(&runtime[8])) { return; }
    let index = componentCell(gid.x);
    let conditioning = compressConditioning(index);
    let gauge = compressGauge(index);
    atomicAdd(&gpuComponents[conditioning].conditioningCount, 1u);
    atomicAdd(&gpuComponents[gauge].gaugeCount, 1u);
    if (cells[index].geometry.y != 0.0) { atomicStore(&gpuComponents[conditioning].conditioningAir, 1u); }
    if (cells[index].negative.w != 0.0) { atomicStore(&gpuComponents[gauge].gaugeAir, 1u); }
}

@compute @workgroup_size(128)
fn markGpuClosedPockets(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!gpuComponentsEnabled() || gid.x >= atomicLoad(&runtime[8])) { return; }
    let index = componentCell(gid.x);
    let root = conditioningRoot(index);
    let count = atomicLoad(&gpuComponents[root].conditioningCount);
    let closed = count > 1u && atomicLoad(&gpuComponents[root].conditioningAir) == 0u;
    cells[index].geometry.z = select(0.0, 1.0, closed);
    if (closed && index == root) {
        atomicAdd(&runtime[9], 1u);
        atomicAdd(&runtime[10], count);
    }
}

@compute @workgroup_size(128)
fn sumGpuSealedFlux(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!gpuComponentsEnabled() || gid.x >= atomicLoad(&runtime[8])) { return; }
    let index = componentCell(gid.x);
    let row = cells[index];
    var finite = finiteComponentValue(row.state.y) && finiteComponentValue(row.state.z);
    for (var axis = 0u; axis < 3u; axis++) {
        finite = finite && finiteComponentValue(row.positive[axis]) && finiteComponentValue(row.negative[axis]);
    }
    if (!finite) {
        runtimeFailure(16u);
        return;
    }
    let root = gaugeRoot(index);
    if (atomicLoad(&gpuComponents[root].gaugeAir) != 0u) { return; }
    let rhsSquared = row.state.z * row.state.z;
    if (!finiteComponentValue(rhsSquared)) {
        runtimeFailure(16u);
        return;
    }
    if (!atomicAddFinite(&gpuComponents[root].fluxBits, row.state.z)) { return; }
    _ = atomicAddFinite(&gpuComponents[root].rhsSquaredBits, rhsSquared);
}

@compute @workgroup_size(128)
fn fixGpuPressureGauges(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!gpuComponentsEnabled() || gid.x >= atomicLoad(&runtime[8])) { return; }
    let index = componentCell(gid.x);
    if (gaugeRoot(index) != index || atomicLoad(&gpuComponents[index].gaugeAir) != 0u) { return; }
    let count = atomicLoad(&gpuComponents[index].gaugeCount);
    let flux = bitcast<f32>(atomicLoad(&gpuComponents[index].fluxBits));
    let rhsSquared = bitcast<f32>(atomicLoad(&gpuComponents[index].rhsSquaredBits));
    if (count == 0u || !finiteComponentValue(flux) || !finiteComponentValue(rhsSquared) || rhsSquared < 0.0) {
        runtimeFailure(16u);
        return;
    }
    atomicAdd(&runtime[11], 1u);
    atomicMax(&runtime[12], bitcast<u32>(abs(flux)));
    let relativeTolerance = sqrt(max(params.tolerances.x, 0.0));
    let absoluteTolerance = sqrt(max(params.tolerances.y, 0.0));
    let compatibleTolerance = 4.0 * sqrt(f32(count)) * max(absoluteTolerance, relativeTolerance * sqrt(rhsSquared));
    if (abs(flux) > compatibleTolerance) {
        runtimeFailure(32u);
        return;
    }
    cells[index].positive.w = 1.0;
}
`;

export const FLIP_REFERENCE_GPU_COMPONENT_BINDINGS: Readonly<Record<string, readonly number[]>> = {
    initializeGpuComponents: [0, 5, 6, 11, 12],
    linkGpuComponents: [0, 5, 6, 11, 12],
    aggregateGpuComponents: [5, 6, 11, 12],
    markGpuClosedPockets: [5, 6, 11, 12],
    sumGpuSealedFlux: [5, 6, 11, 12],
    fixGpuPressureGauges: [0, 5, 6, 11, 12],
};
