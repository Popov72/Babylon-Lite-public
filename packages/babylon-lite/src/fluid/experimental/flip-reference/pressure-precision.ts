/** Compensated pressure accumulation and independent cell-operator residual evaluation. */
export const FLIP_REFERENCE_PRESSURE_PRECISION_WGSL = /* wgsl */ `
@group(0) @binding(21) var<storage, read_write> pressureLow: array<f32>;

fn pressureTwoSum(a: f32, b: f32) -> vec2<f32> {
    let sum = a + b;
    let virtualB = sum - a;
    return vec2<f32>(sum, (a - (sum - virtualB)) + (b - virtualB));
}
fn pressurePairAdd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let sum = pressureTwoSum(a.x, b.x);
    return pressureTwoSum(sum.x, sum.y + (a.y + b.y));
}
fn pressurePairScale(a: vec2<f32>, scale: f32) -> vec2<f32> {
    let product = a.x * scale;
    let remainder = fma(a.x, scale, -product) + a.y * scale;
    return pressureTwoSum(product, remainder);
}
fn readPressurePair(index: u32) -> vec2<f32> {
    return vec2<f32>(pcg[index].x, pressureLow[index]);
}
fn pressurePairResidual(index: u32) -> f32 {
    let row = cells[index];
    if (row.positive.w != 0.0 || row.state.w == 0.0) {
        let value = readPressurePair(index);
        return -(value.x + value.y);
    }
    var value = pressurePairScale(readPressurePair(index), row.state.y);
    let coordinate = cellCoordinate(index);
    for (var axis = 0u; axis < 3u; axis++) {
        if (row.positive[axis] > 0.0) {
            let neighbor = cellIndex(coordinate + axisVector(axis));
            if (cells[neighbor].positive.w == 0.0) {
                value = pressurePairAdd(value, -pressurePairScale(readPressurePair(neighbor), row.positive[axis]));
            }
        }
        if (row.negative[axis] > 0.0) {
            let neighbor = cellIndex(coordinate - axisVector(axis));
            if (cells[neighbor].positive.w == 0.0) {
                value = pressurePairAdd(value, -pressurePairScale(readPressurePair(neighbor), row.negative[axis]));
            }
        }
    }
    let residual = pressurePairAdd(vec2<f32>(row.state.z, 0.0), -value);
    return residual.x + residual.y;
}
`;
