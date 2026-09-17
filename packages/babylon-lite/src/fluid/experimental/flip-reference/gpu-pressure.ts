const GPU_PRESSURE_ROW_BYTES = 64;

/** @internal Dense cached pressure operator: four aligned 16-byte rows per cell. */
export function flipReferenceGpuPressureBytes(cells: number): number {
    return cells * GPU_PRESSURE_ROW_BYTES;
}

/** @internal The reduction requires a power-of-two workgroup that fits its shared scratch. */
export function flipReferenceGpuPressureWorkgroupSize(
    limits: Pick<GPUSupportedLimits, "maxComputeInvocationsPerWorkgroup" | "maxComputeWorkgroupSizeX" | "maxComputeWorkgroupStorageSize">
): number {
    const maximum = Math.min(limits.maxComputeInvocationsPerWorkgroup, limits.maxComputeWorkgroupSizeX, Math.floor((limits.maxComputeWorkgroupStorageSize - 48) / 16));
    return maximum >= 1024 ? 1024 : maximum >= 512 ? 512 : 256;
}

/**
 * Cached, warm-started single-workgroup Jacobi-PCG pressure solve.
 *
 * The GPU-resident base shader supplies Params, cells, lists, pcg, runtime,
 * matrixValue(), rowRhs(), inverseDiagonal(), controlIndex(), and stepDt().
 */
export const FLIP_REFERENCE_GPU_PRESSURE_WGSL = /* wgsl */ `
struct GpuPressureRow {
    positive: vec3<f32>,
    diagonal: f32,
    negative: vec3<f32>,
    inverse: f32,
    positiveNeighbor: vec3<u32>,
    rhs: f32,
    negativeNeighbor: vec3<u32>,
    flags: u32,
}
@group(0) @binding(19) var<storage, read_write> gpuPressureRows: array<GpuPressureRow>;

override pressureWorkgroupSize: u32 = 256u;
var<workgroup> pressureSums: array<vec4<f32>, pressureWorkgroupSize>;
var<workgroup> pressureState: array<u32, 3>;

fn pressureFinite(value: f32) -> bool {
    return value == value && abs(value) < 3.0e38;
}
fn reducePressure(local: u32, value: vec4<f32>) -> vec4<f32> {
    pressureSums[local] = value;
    workgroupBarrier();
    for (var stride = pressureWorkgroupSize / 2u; stride > 0u; stride /= 2u) {
        if (local < stride) {
            pressureSums[local] += pressureSums[local + stride];
        }
        workgroupBarrier();
    }
    return workgroupUniformLoad(&pressureSums[0]);
}
fn cachedPressureValue(index: u32, component: u32) -> f32 {
    let row = gpuPressureRows[index];
    var value = row.diagonal * pcg[index][component];
    for (var axis = 0u; axis < 3u; axis++) {
        if (row.positive[axis] > 0.0) {
            value -= row.positive[axis] * pcg[row.positiveNeighbor[axis]][component];
        }
        if (row.negative[axis] > 0.0) {
            value -= row.negative[axis] * pcg[row.negativeNeighbor[axis]][component];
        }
    }
    return value;
}
fn cachedPressureResidual(index: u32) -> f32 {
    let row = gpuPressureRows[index];
    var value = pressurePairScale(readPressurePair(index), row.diagonal);
    for (var axis = 0u; axis < 3u; axis++) {
        if (row.positive[axis] > 0.0) {
            value = pressurePairAdd(value, -pressurePairScale(readPressurePair(row.positiveNeighbor[axis]), row.positive[axis]));
        }
        if (row.negative[axis] > 0.0) {
            value = pressurePairAdd(value, -pressurePairScale(readPressurePair(row.negativeNeighbor[axis]), row.negative[axis]));
        }
    }
    let residual = pressurePairAdd(vec2<f32>(row.rhs, 0.0), -value);
    return residual.x + residual.y;
}
fn failGpuPressure(local: u32, code: f32, rz: f32, residualSquared: f32, rhsSquared: f32, iterations: u32) {
    if (local == 0u) {
        pcg[controlIndex()] = vec4<f32>(rz, residualSquared, rhsSquared, 1.0);
        pcg[controlIndex() + 1u] = vec4<f32>(0.0, 0.0, f32(iterations), code);
        atomicOr(&runtime[1], 16u);
        atomicStore(&runtime[2], 0u);
    }
}
fn publishPressureControl(local: u32, rz: f32, residualSquared: f32, rhsSquared: f32, iterations: u32, needsIteration: bool) {
    if (local == 0u) {
        pcg[controlIndex()] = vec4<f32>(rz, residualSquared, rhsSquared, f32(needsIteration));
        pcg[controlIndex() + 1u] = vec4<f32>(0.0, 0.0, f32(iterations), 0.0);
    }
}

@compute @workgroup_size(128)
fn prepareGpuPressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (atomicLoad(&runtime[2]) == 0u || atomicLoad(&runtime[1]) != 0u || gid.x >= params.counts.x) {
        return;
    }
    let source = cells[gid.x];
    let currentDt = stepDt();
    let previousDt = bitcast<f32>(atomicLoad(&runtime[20]));
    var flags = u32(source.state.w != 0.0);
    var positive = vec3<f32>(0.0);
    var negative = vec3<f32>(0.0);
    var positiveNeighbor = vec3<u32>(0xffffffffu);
    var negativeNeighbor = vec3<u32>(0xffffffffu);
    var diagonal = 1.0;
    var inverse = 1.0;
    var rhs = 0.0;

    if ((flags & 1u) != 0u && source.positive.w == 0.0) {
        diagonal = source.state.y;
        rhs = source.state.z;
        if (!(diagonal > 0.0) || !pressureFinite(diagonal) || !pressureFinite(rhs)) {
            flags |= 4u;
        } else {
            inverse = 1.0 / diagonal;
            let coordinate = cellCoordinate(gid.x);
            for (var axis = 0u; axis < 3u; axis++) {
                let step = axisVector(axis);
                let positiveCoordinate = coordinate + step;
                let negativeCoordinate = coordinate - step;
                var positiveCoefficient = source.positive[axis];
                var negativeCoefficient = source.negative[axis];
                if (
                    !(positiveCoefficient >= 0.0) || !pressureFinite(positiveCoefficient) || !(negativeCoefficient >= 0.0)
                    || !pressureFinite(negativeCoefficient)
                ) {
                    flags |= 4u;
                }
                if (positiveCoefficient > 0.0) {
                    if (!inGrid(positiveCoordinate)) {
                        flags |= 4u;
                    } else {
                        let neighbor = cellIndex(positiveCoordinate);
                        if (cells[neighbor].state.w == 0.0) {
                            flags |= 4u;
                        } else if (cells[neighbor].positive.w == 0.0) {
                            positive[axis] = positiveCoefficient;
                            positiveNeighbor[axis] = neighbor;
                        }
                    }
                }
                if (negativeCoefficient > 0.0) {
                    if (!inGrid(negativeCoordinate)) {
                        flags |= 4u;
                    } else {
                        let neighbor = cellIndex(negativeCoordinate);
                        if (cells[neighbor].state.w == 0.0) {
                            flags |= 4u;
                        } else if (cells[neighbor].positive.w == 0.0) {
                            negative[axis] = negativeCoefficient;
                            negativeNeighbor[axis] = neighbor;
                        }
                    }
                }
            }
            if (!pressureFinite(inverse)) {
                flags |= 4u;
            }
        }
    }
    if ((flags & 4u) != 0u) {
        positive = vec3<f32>(0.0);
        negative = vec3<f32>(0.0);
        positiveNeighbor = vec3<u32>(0xffffffffu);
        negativeNeighbor = vec3<u32>(0xffffffffu);
        diagonal = 1.0;
        inverse = 1.0;
        rhs = 0.0;
    }

    var guess = vec2<f32>(0.0);
    if ((flags & 1u) != 0u && source.positive.w == 0.0 && previousDt > 0.0 && pressureFinite(previousDt) && currentDt > 0.0 && pressureFinite(currentDt)) {
        guess = pressurePairScale(readPressurePair(gid.x), currentDt / previousDt);
        if (!pressureFinite(guess.x) || !pressureFinite(guess.y)) {
            guess = vec2<f32>(0.0);
            flags |= 2u;
        }
    } else if (!(currentDt > 0.0) || !pressureFinite(currentDt) || (previousDt != 0.0 && !pressureFinite(previousDt))) {
        flags |= 2u;
    }
    gpuPressureRows[gid.x] = GpuPressureRow(positive, diagonal, negative, inverse, positiveNeighbor, rhs, negativeNeighbor, flags);
    pcg[gid.x] = vec4<f32>(guess.x, 0.0, 0.0, 0.0);
    pressureLow[gid.x] = guess.y;
}

@compute @workgroup_size(128)
fn initializeGpuPressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (atomicLoad(&runtime[2]) == 0u || atomicLoad(&runtime[1]) != 0u || gid.x >= params.counts.x) {
        return;
    }
    var row = gpuPressureRows[gid.x];
    if ((row.flags & 1u) == 0u || (row.flags & 4u) != 0u) {
        return;
    }
    let residual = cachedPressureResidual(gid.x);
    let preconditioned = residual * row.inverse;
    if (!pressureFinite(residual) || !pressureFinite(preconditioned)) {
        row.flags |= 2u;
        gpuPressureRows[gid.x].flags = row.flags;
        return;
    }
    // Neighbors read x during this dispatch; preparation already cleared the other components.
    pcg[gid.x].y = residual;
    pcg[gid.x].z = preconditioned;
}

// Phase 0 solves normally; 1 initializes an eligible parallel prefix; 2 preserves its CG recurrence.
fn runGpuPressure(local: u32, phase: u32) {
    if (local == 0u) {
        pressureState[0] = u32(atomicLoad(&runtime[2]) != 0u && atomicLoad(&runtime[1]) == 0u);
        pressureState[1] = atomicLoad(&runtime[8]);
        pressureState[2] = select(0u, u32(pcg[controlIndex() + 1u].z), phase == 2u);
        pressureSums[0] = pcg[controlIndex()];
    }
    let enabled = workgroupUniformLoad(&pressureState[0]);
    let activeCount = workgroupUniformLoad(&pressureState[1]);
    let incomingControl = workgroupUniformLoad(&pressureSums[0]);
    if (enabled == 0u) {
        return;
    }
    let maximumIterations = u32(params.removal.w);
    if (activeCount > params.counts.x || maximumIterations == 0u) {
        failGpuPressure(local, 5.0, 0.0, 0.0, 0.0, 0u);
        return;
    }

    var checks = vec4<f32>(0.0);
    var initialProducts = vec4<f32>(0.0);
    for (var item = local; item < activeCount; item += pressureWorkgroupSize) {
        let index = atomicLoad(&lists[item]);
        if (index >= params.counts.x) {
            checks.x += 1.0;
            continue;
        }
        let row = gpuPressureRows[index];
        if ((row.flags & 1u) == 0u || (row.flags & 4u) != 0u) {
            checks.x += 1.0;
            continue;
        }
        if ((row.flags & 2u) != 0u && phase != 2u) {
            checks.y += 1.0;
        }
        let value = pcg[index];
        let preconditioned = select(value.z, value.y * row.inverse, phase == 2u);
        let rz = value.y * preconditioned;
        let residualSquared = value.y * value.y;
        let rhsSquared = row.rhs * row.rhs;
        if (!pressureFinite(value.x) || !pressureFinite(value.y) || !pressureFinite(value.z) || !pressureFinite(rz) || !pressureFinite(residualSquared)) {
            checks.y += 1.0;
        } else {
            initialProducts.x += rz;
            initialProducts.y += residualSquared;
        }
        if (!pressureFinite(rhsSquared)) {
            checks.x += 1.0;
        } else {
            initialProducts.z += rhsSquared;
        }
    }
    let checkTotal = reducePressure(local, checks);
    let initial = reducePressure(local, initialProducts);
    if (checkTotal.x != 0.0 || !(initial.z >= 0.0) || !pressureFinite(initial.z)) {
        failGpuPressure(local, 2.0, 0.0, 0.0, initial.z, 0u);
        return;
    }

    var rz = select(initial.x, incomingControl.x, phase == 2u);
    var residualSquared = select(initial.y, incomingControl.y, phase == 2u);
    let rhsSquared = select(initial.z, incomingControl.z, phase == 2u);
    let invalidInitial = checkTotal.y != 0.0 || !(rz >= 0.0) || !pressureFinite(rz) || !(residualSquared >= 0.0) || !pressureFinite(residualSquared)
        || !(rhsSquared >= 0.0) || !pressureFinite(rhsSquared) || (phase == 2u && incomingControl.w != 0.0 && incomingControl.w != 1.0);
    if (phase == 2u && invalidInitial) {
        failGpuPressure(local, 2.0, rz, residualSquared, rhsSquared, u32(pcg[controlIndex() + 1u].z));
        return;
    }
    let rejectWarm = phase != 2u && (invalidInitial || residualSquared > rhsSquared);
    if (rejectWarm) {
        var coldProducts = vec4<f32>(0.0);
        for (var item = local; item < activeCount; item += pressureWorkgroupSize) {
            let index = atomicLoad(&lists[item]);
            let row = gpuPressureRows[index];
            let preconditioned = row.rhs * row.inverse;
            let coldRz = row.rhs * preconditioned;
            if (!pressureFinite(preconditioned) || !pressureFinite(coldRz)) {
                coldProducts.w += 1.0;
            } else {
                coldProducts.x += coldRz;
                coldProducts.y += row.rhs * row.rhs;
            }
            pcg[index] = vec4<f32>(0.0, row.rhs, preconditioned, 0.0);
            pressureLow[index] = 0.0;
        }
        storageBarrier();
        let cold = reducePressure(local, coldProducts);
        if (cold.w != 0.0 || !(cold.x >= 0.0) || !pressureFinite(cold.x) || !(cold.y >= 0.0) || !pressureFinite(cold.y)) {
            failGpuPressure(local, 2.0, 0.0, 0.0, rhsSquared, 0u);
            return;
        }
        rz = cold.x;
        residualSquared = cold.y;
    }

    let threshold = max(params.tolerances.y, params.tolerances.x * rhsSquared);
    var iterations = workgroupUniformLoad(&pressureState[2]);
    var needsIteration = select(residualSquared > threshold, incomingControl.w != 0.0, phase == 2u);
    if (phase == 1u && activeCount >= 8192u && activeCount <= arrayLength(&particleScratch)) {
        publishPressureControl(local, rz, residualSquared, rhsSquared, iterations, needsIteration);
        return;
    }
    var previousRestart = 0xffffffffu;
    loop {
        while (needsIteration && iterations < maximumIterations) {
            var directionProducts = vec4<f32>(0.0);
            for (var item = local; item < activeCount; item += pressureWorkgroupSize) {
                let index = atomicLoad(&lists[item]);
                let direction = pcg[index].z;
                let product = cachedPressureValue(index, 2u);
                pcg[index].w = product;
                let dotProduct = direction * product;
                if (!pressureFinite(product) || !pressureFinite(dotProduct)) {
                    directionProducts.w += 1.0;
                } else {
                    directionProducts.z += dotProduct;
                }
            }
            storageBarrier();
            let direction = reducePressure(local, directionProducts);
            if (direction.w != 0.0 || !(direction.z > 0.0) || !pressureFinite(direction.z)) {
                failGpuPressure(local, select(1.0, 2.0, direction.w != 0.0 || !pressureFinite(direction.z)), rz, residualSquared, rhsSquared, iterations);
                return;
            }
            let alpha = rz / direction.z;
            if (!pressureFinite(alpha)) {
                failGpuPressure(local, 2.0, rz, residualSquared, rhsSquared, iterations);
                return;
            }

            var residualProducts = vec4<f32>(0.0);
            for (var item = local; item < activeCount; item += pressureWorkgroupSize) {
                let index = atomicLoad(&lists[item]);
                let row = gpuPressureRows[index];
                let value = pcg[index];
                let nextX = pressurePairAdd(readPressurePair(index), pressurePairScale(vec2<f32>(value.z, 0.0), alpha));
                let nextResidual = value.y - alpha * value.w;
                let preconditioned = nextResidual * row.inverse;
                let nextRz = nextResidual * preconditioned;
                let nextResidualSquared = nextResidual * nextResidual;
                pcg[index].x = nextX.x;
                pressureLow[index] = nextX.y;
                pcg[index].y = nextResidual;
                if (
                    !pressureFinite(nextX.x) || !pressureFinite(nextX.y) || !pressureFinite(nextResidual) || !pressureFinite(preconditioned) || !pressureFinite(nextRz)
                    || !pressureFinite(nextResidualSquared)
                ) {
                    residualProducts.w += 1.0;
                } else {
                    residualProducts.x += nextRz;
                    residualProducts.y += nextResidualSquared;
                }
            }
            storageBarrier();
            let residual = reducePressure(local, residualProducts);
            if (
                residual.w != 0.0 || !(residual.x >= 0.0) || !pressureFinite(residual.x) || !(residual.y >= 0.0)
                || !pressureFinite(residual.y)
            ) {
                failGpuPressure(local, 2.0, rz, residualSquared, rhsSquared, iterations);
                return;
            }
            let previousRz = rz;
            rz = residual.x;
            residualSquared = residual.y;
            iterations++;
            needsIteration = residualSquared > threshold;

            if (needsIteration) {
                if (!(previousRz > 0.0)) {
                    failGpuPressure(local, 1.0, rz, residualSquared, rhsSquared, iterations);
                    return;
                }
                let beta = rz / previousRz;
                if (!pressureFinite(beta)) {
                    failGpuPressure(local, 2.0, rz, residualSquared, rhsSquared, iterations);
                    return;
                }
                for (var item = local; item < activeCount; item += pressureWorkgroupSize) {
                    let index = atomicLoad(&lists[item]);
                    let value = pcg[index];
                    pcg[index].z = value.y * gpuPressureRows[index].inverse + beta * value.z;
                }
                storageBarrier();
            }
        }

        var trueProducts = vec4<f32>(0.0);
        for (var item = local; item < activeCount; item += pressureWorkgroupSize) {
            let index = atomicLoad(&lists[item]);
            let trueResidual = pressureResidualValue(index);
            let preconditioned = trueResidual * inverseDiagonal(index);
            let trueRz = trueResidual * preconditioned;
            let trueResidualSquared = trueResidual * trueResidual;
            pcg[index].y = trueResidual;
            pcg[index].z = preconditioned;
            if (
                !pressureFinite(trueResidual) || !pressureFinite(preconditioned) || !pressureFinite(trueRz)
                || !pressureFinite(trueResidualSquared)
            ) {
                trueProducts.w += 1.0;
            } else {
                trueProducts.x += trueRz;
                trueProducts.y += trueResidualSquared;
            }
        }
        storageBarrier();
        let replacement = reducePressure(local, trueProducts);
        if (
            replacement.w != 0.0 || !(replacement.x >= 0.0) || !pressureFinite(replacement.x) || !(replacement.y >= 0.0)
            || !pressureFinite(replacement.y)
        ) {
            failGpuPressure(local, 2.0, rz, residualSquared, rhsSquared, iterations);
            return;
        }
        rz = replacement.x;
        residualSquared = replacement.y;
        needsIteration = residualSquared > threshold;
        publishPressureControl(local, rz, residualSquared, rhsSquared, iterations, needsIteration);
        if (!needsIteration) {
            if (local == 0u) {
                atomicStore(&runtime[20], bitcast<u32>(stepDt()));
            }
            return;
        }
        if (iterations >= maximumIterations) {
            failGpuPressure(local, 4.0, rz, residualSquared, rhsSquared, iterations);
            return;
        }
        if (iterations == previousRestart) {
            failGpuPressure(local, 3.0, rz, residualSquared, rhsSquared, iterations);
            return;
        }
        previousRestart = iterations;
    }
}

@compute @workgroup_size(pressureWorkgroupSize)
fn solveGpuPressure(@builtin(local_invocation_index) local: u32) {
    runGpuPressure(local, 0u);
}
@compute @workgroup_size(pressureWorkgroupSize)
fn initializeParallelGpuPressure(@builtin(local_invocation_index) local: u32) {
    runGpuPressure(local, 1u);
}
@compute @workgroup_size(pressureWorkgroupSize)
fn resumeGpuPressure(@builtin(local_invocation_index) local: u32) {
    runGpuPressure(local, 2u);
}
`;

/** Storage/uniform bindings reachable by each pressure entry point. */
export const FLIP_REFERENCE_GPU_PRESSURE_BINDINGS: Readonly<Record<string, readonly number[]>> = {
    prepareGpuPressure: [0, 5, 8, 11, 19, 21],
    initializeGpuPressure: [0, 8, 11, 19, 21],
    solveGpuPressure: [0, 5, 6, 8, 9, 11, 19, 21],
    initializeParallelGpuPressure: [0, 5, 6, 8, 9, 11, 19, 21],
    resumeGpuPressure: [0, 5, 6, 8, 9, 11, 19, 21],
    project: [0, 4, 5, 8, 21],
    trueResidual: [0, 5, 8, 21],
};
