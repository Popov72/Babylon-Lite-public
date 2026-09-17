/** Independent dense MAC implementation; pressure is stored as dt * p / (density * dx). */
export function flipReferenceWgsl(gpuResident = false): string {
    return /* wgsl */ `
struct Params {
    grid: vec4<u32>,
    origin: vec4<f32>,
    gravity: vec4<f32>,
    settings: vec4<f32>,
    counts: vec4<u32>,
    switches: vec4<u32>,
    tolerances: vec4<f32>,
    geometry: vec4<f32>,
    modes: vec4<u32>, // y bits: reference fractions, constrain old grid, supplied domain.
    particles: vec4<u32>, // Capacity, scan-group capacity, histogram bins, removal flags.
    removal: vec4<f32>, // Histogram speed interval: CFL * dx / full-frame dt.
}
struct Face {
    velocity: f32,
    previous: f32,
    area: f32,
    solid: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
struct Cell {
    state: vec4<f32>,
    positive: vec4<f32>,
    negative: vec4<f32>,
    geometry: vec4<f32>,
}
struct FaceCoordinate {
    cell: vec3<i32>,
    axis: u32,
}
struct SolidSample {
    distance: f32,
    gradient: vec3<f32>,
    velocity: vec3<f32>,
}
struct Correction {
    position: vec3<f32>,
    velocity: vec3<f32>,
    contacted: bool,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> speeds: array<f32>;
@group(0) @binding(4) var<storage, read_write> faces: array<Face>;
@group(0) @binding(5) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(6) var<storage, read_write> lists: array<atomic<u32> >;
@group(0) @binding(7) var<storage, read> solids: array<vec4f>;
// Per-cell x, residual, search direction, A*direction; then reduction scratch and two controls.
@group(0) @binding(8) var<storage, read_write> pcg: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> particleScratch: array<vec4f>;
@group(0) @binding(10) var<storage, read_write> particleState: array<atomic<u32> >;
${gpuResident ? "@group(0) @binding(11) var<storage, read_write> runtime: array<atomic<u32>>;" : ""}
var<workgroup> sums: array<vec4<f32>, 128>;
var<workgroup> survivorScan: array<u32, 128>;

fn particleCount() -> u32 { return ${gpuResident ? "atomicLoad(&runtime[0])" : "params.grid.w"}; }
fn stepDt() -> f32 { return ${gpuResident ? "bitcast<f32>(atomicLoad(&runtime[4]))" : "params.gravity.w"}; }
fn cellIndex(c: vec3<i32>) -> u32 {
    return u32(c.x) + params.grid.x * (u32(c.y) + params.grid.y * u32(c.z));
}
fn cellCoordinate(i: u32) -> vec3<i32> {
    return vec3<i32>(i32(i % params.grid.x), i32((i / params.grid.x) % params.grid.y), i32(i / (params.grid.x * params.grid.y)));
}
fn inGrid(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(params.grid.xyz));
}
fn axisVector(axis: u32) -> vec3<i32> {
    var v = vec3<i32>(0);
    v[axis] = 1;
    return v;
}
fn faceDimensions(axis: u32) -> vec3<i32> {
    return vec3<i32>(params.grid.xyz) + axisVector(axis);
}
fn faceIndex(c: vec3<i32>, axis: u32) -> u32 {
    let d = faceDimensions(axis);
    var offset = 0u;
    if (axis == 1u) { offset = params.counts.z; }
    if (axis == 2u) { offset = params.counts.z + params.counts.w; }
    return offset + u32(c.x + d.x * (c.y + d.y * c.z));
}
fn faceCoordinate(index: u32) -> FaceCoordinate {
    var i = index;
    var axis = 0u;
    if (i >= params.counts.z + params.counts.w) {
        i -= params.counts.z + params.counts.w;
        axis = 2u;
    } else if (i >= params.counts.z) {
        i -= params.counts.z;
        axis = 1u;
    }
    let d = vec3<u32>(faceDimensions(axis));
    return FaceCoordinate(vec3<i32>(i32(i % d.x), i32((i / d.x) % d.y), i32(i / (d.x * d.y))), axis);
}
fn vertexIndex(c: vec3<i32>) -> u32 {
    let d = params.grid.xyz + vec3<u32>(1u);
    return u32(c.x) + d.x * (u32(c.y) + d.y * u32(c.z));
}
fn solidNode(c: vec3<i32>) -> vec4<f32> {
    let obstacle = solids[vertexIndex(c)];
    if (params.geometry.y == 0.0 || (params.modes.y & 4u) != 0u) { return obstacle; }
    let halfSize = 0.5 * vec3<f32>(params.grid.xyz) * params.origin.w;
    let q = abs(vec3<f32>(c) * params.origin.w - halfSize) - halfSize + vec3<f32>(params.geometry.y);
    let domain = -length(max(q, vec3<f32>(0.0))) - min(max(q.x, max(q.y, q.z)), 0.0);
    if (domain < obstacle.x) { return vec4<f32>(domain, 0.0, 0.0, 0.0); }
    return obstacle;
}
fn statusIndex(i: u32) -> u32 { return params.counts.x + params.particles.x + i; }
fn removalControl(i: u32) -> u32 { return 2u * params.particles.x + params.particles.y + i; }
fn histogramIndex(i: u32) -> u32 { return removalControl(8u) + i; }
fn controlIndex() -> u32 { return params.counts.x + params.switches.w; }
fn liquid(c: vec3<i32>) -> bool {
    if (!inGrid(c)) { return false; }
    return cells[cellIndex(c)].state.w != 0.0;
}
fn theta(a: f32, b: f32) -> f32 {
    return clamp(a / (a - b), params.settings.y, 1.0);
}
fn triangleOpen(a: f32, b: f32, c: f32) -> f32 {
    // An exactly zero-level face is a wall, not an open connection to the solid's interior.
    let positive = u32(a > 0.0) + u32(b > 0.0) + u32(c > 0.0);
    if (positive == 0u) { return 0.0; }
    if (positive == 3u) { return 1.0; }
    if (positive == 1u) {
        if (a > 0.0) { return (a / (a - b)) * (a / (a - c)); }
        if (b > 0.0) { return (b / (b - a)) * (b / (b - c)); }
        return (c / (c - a)) * (c / (c - b));
    }
    if (a <= 0.0) { return 1.0 - (a / (a - b)) * (a / (a - c)); }
    if (b <= 0.0) { return 1.0 - (b / (b - a)) * (b / (b - c)); }
    return 1.0 - (c / (c - a)) * (c / (c - b));
}
fn quadOpen(s: vec4<f32>) -> f32 {
    let middle = 0.25 * (s.x + s.y + s.z + s.w);
    if ((params.modes.y & 1u) == 0u) {
        return 0.25 * (triangleOpen(s.x, s.y, middle) + triangleOpen(s.y, s.z, middle)
            + triangleOpen(s.z, s.w, middle) + triangleOpen(s.w, s.x, middle));
    }
    let inside = u32(s.x < 0.0) + u32(s.y < 0.0) + u32(s.z < 0.0) + u32(s.w < 0.0);
    if (inside == 0u) { return 1.0; }
    if (inside == 4u) { return 0.0; }
    if (inside == 1u || inside == 3u) {
        for (var i = 0u; i < 4u; i++) {
            if ((s[i] < 0.0) == (inside == 1u)) {
                let area = 0.5 * (s[i] / (s[i] - s[(i + 1u) & 3u])) * (s[i] / (s[i] - s[(i + 3u) & 3u]));
                return select(area, 1.0 - area, inside == 1u);
            }
        }
    }
    for (var i = 0u; i < 4u; i++) {
        let next = (i + 1u) & 3u;
        if (s[i] < 0.0 && s[next] < 0.0) {
            let left = s[i] / (s[i] - s[(i + 3u) & 3u]);
            let right = s[next] / (s[next] - s[(i + 2u) & 3u]);
            return 1.0 - 0.5 * (left + right);
        }
    }
    // A saddle face uses two isolated corner triangles on the side opposite the mean sign.
    var corners = 0.0;
    for (var i = 0u; i < 4u; i++) {
        if ((s[i] < 0.0) == (middle >= 0.0)) {
            corners += 0.5 * (s[i] / (s[i] - s[(i + 1u) & 3u])) * (s[i] / (s[i] - s[(i + 3u) & 3u]));
        }
    }
    return select(corners, 1.0 - corners, middle >= 0.0);
}
fn faceBoundary(c: vec3<i32>, axis: u32) -> vec2<f32> {
    let a = axisVector((axis + 1u) % 3u);
    let b = axisVector((axis + 2u) % 3u);
    let s0 = solidNode(c);
    let s1 = solidNode(c + a);
    let s2 = solidNode(c + a + b);
    let s3 = solidNode(c + b);
    let open = quadOpen(vec4<f32>(s0.x, s1.x, s2.x, s3.x));
    let area = clamp(open, 0.0, 1.0);
    let velocity = 0.25 * (s0[axis + 1u] + s1[axis + 1u] + s2[axis + 1u] + s3[axis + 1u]);
    // The fluid domain is closed, but an obstacle may cross it. Preserve that covered solid-volume flux.
    if (c[axis] == 0 || c[axis] == i32(params.grid[axis])) { return vec2<f32>(0.0, (1.0 - area) * velocity); }
    return vec2<f32>(area, velocity);
}
fn tetrahedronOpen(values: vec4<f32>) -> f32 {
    var positive: array<f32, 4>;
    var negative: array<f32, 4>;
    var count = 0u;
    var other = 0u;
    for (var i = 0u; i < 4u; i++) {
        if (values[i] > 0.0) { positive[count] = values[i]; count++; }
        else { negative[other] = values[i]; other++; }
    }
    if (count == 0u) { return 0.0; }
    if (count == 4u) { return 1.0; }
    if (count == 1u) {
        let a = positive[0];
        return (a / (a - negative[0])) * (a / (a - negative[1])) * (a / (a - negative[2]));
    }
    if (count == 3u) {
        let a = negative[0];
        return 1.0 - (a / (a - positive[0])) * (a / (a - positive[1])) * (a / (a - positive[2]));
    }
    // The two-positive clipped tetrahedron is a triangular prism, partitioned into three tetrahedra.
    let ac = positive[0] / (positive[0] - negative[0]);
    let ad = positive[0] / (positive[0] - negative[1]);
    let bc = positive[1] / (positive[1] - negative[0]);
    let bd = positive[1] / (positive[1] - negative[1]);
    return ac * ad + ac * bd * (1.0 - ad) + bc * bd * (1.0 - ac);
}
fn cellOpenVolume(c: vec3<i32>) -> f32 {
    if (params.modes.z == 0u) { return 1.0; }
    var values: array<f32, 8>;
    var center = 0.0;
    var minimum = 3.0e38;
    var maximum = -3.0e38;
    for (var corner = 0u; corner < 8u; corner++) {
        let d = vec3<i32>(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
        let value = solidNode(c + d).x;
        values[corner] = value;
        center += value * 0.125;
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }
    if ((params.modes.y & 1u) != 0u) {
        if (minimum >= 0.0) { return 1.0; }
        if (maximum < 0.0) { return 0.0; }
        // Each parity partition has four corner tetrahedra of volume 1/6 and one central tetrahedron of volume 1/3.
        var solidVolume = 2.0 * tetrahedronOpen(-vec4<f32>(values[0], values[3], values[5], values[6]));
        solidVolume += 2.0 * tetrahedronOpen(-vec4<f32>(values[1], values[2], values[4], values[7]));
        solidVolume += tetrahedronOpen(-vec4<f32>(values[1], values[0], values[3], values[5]));
        solidVolume += tetrahedronOpen(-vec4<f32>(values[2], values[0], values[3], values[6]));
        solidVolume += tetrahedronOpen(-vec4<f32>(values[4], values[0], values[5], values[6]));
        solidVolume += tetrahedronOpen(-vec4<f32>(values[7], values[3], values[5], values[6]));
        solidVolume += tetrahedronOpen(-vec4<f32>(values[0], values[1], values[2], values[4]));
        solidVolume += tetrahedronOpen(-vec4<f32>(values[3], values[1], values[2], values[7]));
        solidVolume += tetrahedronOpen(-vec4<f32>(values[5], values[1], values[4], values[7]));
        solidVolume += tetrahedronOpen(-vec4<f32>(values[6], values[2], values[4], values[7]));
        return clamp(1.0 - solidVolume / 12.0, 0.0, 1.0);
    }
    if (minimum > 0.0) { return 1.0; }
    if (maximum <= 0.0) { return 0.0; }
    var volume = 0.0;
    // The 24 tetrahedra meet at the cube center and use the same center-triangulated faces as aperture areas.
    for (var axis = 0u; axis < 3u; axis++) {
        let a = 1u << ((axis + 1u) % 3u);
        let b = 1u << ((axis + 2u) % 3u);
        for (var side = 0u; side < 2u; side++) {
            let base = side << axis;
            let s0 = values[base];
            let s1 = values[base | a];
            let s2 = values[base | a | b];
            let s3 = values[base | b];
            let middle = 0.25 * (s0 + s1 + s2 + s3);
            volume += tetrahedronOpen(vec4<f32>(center, middle, s0, s1));
            volume += tetrahedronOpen(vec4<f32>(center, middle, s1, s2));
            volume += tetrahedronOpen(vec4<f32>(center, middle, s2, s3));
            volume += tetrahedronOpen(vec4<f32>(center, middle, s3, s0));
        }
    }
    return clamp(volume / 24.0, 0.0, 1.0);
}

fn clearParticleLists(index: u32) {
    if (index < params.counts.x) { atomicStore(&lists[index], 0xffffffffu); }
    if (index < 8u) { atomicStore(&lists[statusIndex(index)], 0u); }
}
@compute @workgroup_size(128)
fn clearLists(@builtin(global_invocation_id) gid: vec3<u32>) { clearParticleLists(gid.x); }
@compute @workgroup_size(128)
fn linkParticles(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= particleCount()) { return; }
    let p = (positions[gid.x].xyz - params.origin.xyz) / params.origin.w;
    if (!all(p >= vec3<f32>(0.0)) || !all(p < vec3<f32>(params.grid.xyz))) {
        atomicAdd(&lists[statusIndex(0u)], 1u);
        return;
    }
    let c = vec3<i32>(floor(p));
    let previous = atomicExchange(&lists[cellIndex(c)], gid.x);
    atomicStore(&lists[params.counts.x + gid.x], previous);
}
@compute @workgroup_size(128)
fn liquidLevelSet(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x) { return; }
    let c = cellCoordinate(gid.x);
    let center = vec3<f32>(c) + vec3<f32>(0.5);
    let radius = params.settings.z;
    let reference = params.modes.x != 0u;
    let reach = i32(ceil(radius + 1.0));
    var phi = 1.0;
    var lower = max(c - vec3<i32>(reach), vec3<i32>(0));
    var upper = min(c + vec3<i32>(reach), vec3<i32>(params.grid.xyz) - vec3<i32>(1));
    if (reference) {
        phi = 3.0;
        lower = max(vec3<i32>(floor(center - vec3<f32>(2.0 * radius))), vec3<i32>(0));
        upper = min(vec3<i32>(floor(center + vec3<f32>(2.0 * radius))), vec3<i32>(params.grid.xyz) - vec3<i32>(1));
    }
    for (var z = lower.z; z <= upper.z; z++) {
        for (var y = lower.y; y <= upper.y; y++) {
            for (var x = lower.x; x <= upper.x; x++) {
                var marker = atomicLoad(&lists[cellIndex(vec3<i32>(x, y, z))]);
                while (marker != 0xffffffffu) {
                    let p = (positions[marker].xyz - params.origin.xyz) / params.origin.w;
                    if (!reference || all(abs(p - center) <= vec3<f32>(2.0 * radius))) {
                        phi = min(phi, length(p - center) - radius);
                    }
                    marker = atomicLoad(&lists[params.counts.x + marker]);
                }
            }
        }
    }
    // Solid samples drive reference phi repair and the legacy solid-cell rejection.
    var solidMaximum = -3.0e38;
    var solidCenter = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let d = vec3<i32>(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
        let distance = solidNode(c + d).x;
        solidMaximum = max(solidMaximum, distance);
        solidCenter += distance * 0.125;
    }
    if (reference) {
        if (phi < 0.5 && solidCenter < 0.0) { phi = -0.5; }
        if (abs(phi) < 0.005) { phi = select(0.005, -0.005, phi <= 0.0); }
    } else if (solidMaximum <= 0.0) {
        phi = 1.0;
    }
    // Include zero-level cells so the default radius covers markers exactly at lattice vertices.
    let interior = all(c > vec3<i32>(0)) && all(c < vec3<i32>(params.grid.xyz) - vec3<i32>(1));
    let pressureCell = phi <= 0.0 && (!reference || solidMaximum >= 0.0 || interior);
    cells[gid.x] = Cell(vec4<f32>(phi, 0.0, 0.0, f32(pressureCell)), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
}
@compute @workgroup_size(128)
fn particleToGrid(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.y) { return; }
    let fc = faceCoordinate(gid.x);
    var point = vec3<f32>(fc.cell) + vec3<f32>(0.5);
    point[fc.axis] -= 0.5;
    let boundary = faceBoundary(fc.cell, fc.axis);
    var sum = 0.0;
    var weight = 0.0;
    let radius = params.geometry.x;
    let lower = max(vec3<i32>(floor(point - vec3<f32>(radius))), vec3<i32>(0));
    let upper = min(vec3<i32>(floor(point + vec3<f32>(radius))), vec3<i32>(params.grid.xyz) - vec3<i32>(1));
    for (var z = lower.z; z <= upper.z; z++) {
        for (var y = lower.y; y <= upper.y; y++) {
            for (var x = lower.x; x <= upper.x; x++) {
                var marker = atomicLoad(&lists[cellIndex(vec3<i32>(x, y, z))]);
                while (marker != 0xffffffffu) {
                    let p = (positions[marker].xyz - params.origin.xyz) / params.origin.w;
                    let delta = abs(p - point);
                    var w = 0.0;
                    if (params.switches.x == 0u) {
                        let basis = max(vec3<f32>(0.0), vec3<f32>(1.0) - delta);
                        w = basis.x * basis.y * basis.z;
                    } else {
                        let q2 = dot(delta, delta) / (radius * radius);
                        let basis = max(0.0, 1.0 - q2);
                        if (params.switches.x == 2u) {
                            if (q2 < 1.0) { w = basis * basis * (1.0 - (4.0 / 9.0) * q2); }
                        } else {
                            w = basis * basis * basis;
                        }
                    }
                    weight += w;
                    sum += w * velocities[marker][fc.axis];
                    marker = atomicLoad(&lists[params.counts.x + marker]);
                }
            }
        }
    }
    var value = 0.0;
    var valid = 0u;
    if (weight > params.geometry.w) { value = sum / weight; valid = 1u; }
    faces[gid.x] = Face(value, 0.0, boundary.x, boundary.y, valid, 0u, 0u, 0u);
}
fn extendFace(i: u32, source: u32, destination: u32, unconstrained: bool) {
    if (i >= params.counts.y) { return; }
    var f = faces[source + i];
    if (f.valid == 0u && (unconstrained || f.area > 0.0)) {
        let fc = faceCoordinate(i);
        let dimensions = faceDimensions(fc.axis);
        var value = 0.0;
        var weight = 0.0;
        for (var axis = 0u; axis < 3u; axis++) {
            for (var sign = -1; sign <= 1; sign += 2) {
                let q = fc.cell + sign * axisVector(axis);
                if (all(q >= vec3<i32>(0)) && all(q < dimensions)) {
                    let neighbor = faces[source + faceIndex(q, fc.axis)];
                    if (neighbor.valid != 0u && (unconstrained || neighbor.area > 0.0)) {
                        value += neighbor.velocity;
                        weight += 1.0;
                    }
                }
            }
        }
        if (weight > 0.0) { f.velocity = value / weight; f.valid = 1u; }
    }
    faces[destination + i] = f;
}
@compute @workgroup_size(128)
fn extendBeforeToB(@builtin(global_invocation_id) gid: vec3<u32>) { extendFace(gid.x, 0u, params.counts.y, true); }
@compute @workgroup_size(128)
fn extendBeforeToA(@builtin(global_invocation_id) gid: vec3<u32>) { extendFace(gid.x, params.counts.y, 0u, true); }
@compute @workgroup_size(128)
fn extendToB(@builtin(global_invocation_id) gid: vec3<u32>) { extendFace(gid.x, 0u, params.counts.y, false); }
@compute @workgroup_size(128)
fn extendToA(@builtin(global_invocation_id) gid: vec3<u32>) { extendFace(gid.x, params.counts.y, 0u, false); }
@compute @workgroup_size(128)
fn copyToA(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.counts.y) { faces[gid.x] = faces[params.counts.y + gid.x]; }
}
@compute @workgroup_size(128)
fn snapshotAndForce(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.y) { return; }
    let fc = faceCoordinate(gid.x);
    var f = faces[gid.x];
    // Save pre-force transfer; reference solid constraints are applied to both snapshots later.
    f.previous = f.velocity;
    if (f.area > 0.0 && (liquid(fc.cell) || liquid(fc.cell - axisVector(fc.axis)))) {
        f.velocity += params.gravity[fc.axis] * stepDt();
        f.valid = 1u;
    }
    faces[gid.x] = f;
}
@compute @workgroup_size(128)
fn buildMatrix(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x) { return; }
    var row = cells[gid.x];
    if (row.state.w == 0.0) { return; }
    let c = cellCoordinate(gid.x);
    let volume = cellOpenVolume(c);
    var diagonal = 0.0;
    var rhs = 0.0;
    var conditioningAir = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        for (var sign = -1; sign <= 1; sign += 2) {
            let q = c + sign * axisVector(axis);
            var fcoord = c;
            if (sign > 0) { fcoord += axisVector(axis); }
            let f = faces[faceIndex(fcoord, axis)];
            rhs += f32(sign) * ((f.area - volume) * f.solid - f.area * f.velocity);
            if (f.area > 0.0 && inGrid(q)) {
                var coefficient = f.area;
                if (liquid(q)) {
                    if (sign > 0) { row.positive[axis] = coefficient; }
                    else { row.negative[axis] = coefficient; }
                } else {
                    coefficient /= theta(row.state.x, cells[cellIndex(q)].state.x);
                    row.negative.w = 1.0;
                    if (f.area >= 1.0e-6) { conditioningAir = 1.0; }
                }
                diagonal += coefficient;
            }
        }
    }
    row.state.y = diagonal;
    row.state.z = rhs;
    cells[gid.x].state.y = row.state.y;
    cells[gid.x].state.z = row.state.z;
    cells[gid.x].positive = row.positive;
    cells[gid.x].negative = row.negative;
    cells[gid.x].geometry.x = volume;
    cells[gid.x].geometry.y = conditioningAir;
}

@compute @workgroup_size(128)
fn conditionSolidVelocities(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.y) { return; }
    let face = faceCoordinate(gid.x);
    let other = face.cell - axisVector(face.axis);
    var conditioned = false;
    if (inGrid(face.cell)) { conditioned = cells[cellIndex(face.cell)].geometry.z != 0.0; }
    if (inGrid(other)) { conditioned = conditioned || cells[cellIndex(other)].geometry.z != 0.0; }
    if (conditioned) { faces[gid.x].solid = 0.0; }
}

fn matrixValue(i: u32, direction: bool) -> f32 {
    let row = cells[i];
    var component = 0u;
    if (direction) { component = 2u; }
    if (row.positive.w != 0.0 || row.state.w == 0.0) { return pcg[i][component]; }
    var value = row.state.y * pcg[i][component];
    let c = cellCoordinate(i);
    for (var axis = 0u; axis < 3u; axis++) {
        if (row.positive[axis] > 0.0) {
            let neighbor = cellIndex(c + axisVector(axis));
            if (cells[neighbor].positive.w == 0.0) { value -= row.positive[axis] * pcg[neighbor][component]; }
        }
        if (row.negative[axis] > 0.0) {
            let neighbor = cellIndex(c - axisVector(axis));
            if (cells[neighbor].positive.w == 0.0) { value -= row.negative[axis] * pcg[neighbor][component]; }
        }
    }
    return value;
}
fn rowRhs(i: u32) -> f32 {
    if (cells[i].positive.w != 0.0 || cells[i].state.w == 0.0) { return 0.0; }
    return cells[i].state.z;
}
fn inverseDiagonal(i: u32) -> f32 {
    let diagonal = cells[i].state.y;
    if (diagonal > 0.0 && cells[i].positive.w == 0.0) { return 1.0 / diagonal; }
    return 1.0;
}
@compute @workgroup_size(128)
fn initializeCg(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x) { return; }
    let rhs = rowRhs(gid.x);
    pcg[gid.x] = vec4<f32>(0.0, rhs, rhs * inverseDiagonal(gid.x), 0.0);
}
@compute @workgroup_size(128)
fn applyDirection(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x || pcg[controlIndex()].w == 0.0) { return; }
    pcg[gid.x].w = matrixValue(gid.x, true);
}
@compute @workgroup_size(128)
fn trueResidual(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x) { return; }
    // Only x is read from neighbors here; writes to r/d/q are invocation-local.
    let residual = rowRhs(gid.x) - matrixValue(gid.x, false);
    pcg[gid.x].y = residual;
    pcg[gid.x].z = residual * inverseDiagonal(gid.x);
}
@compute @workgroup_size(128)
fn updateResidual(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x || pcg[controlIndex()].w == 0.0) { return; }
    let alpha = pcg[controlIndex() + 1u].x;
    let v = pcg[gid.x];
    pcg[gid.x] = vec4<f32>(v.x + alpha * v.z, v.y - alpha * v.w, v.z, v.w);
}
@compute @workgroup_size(128)
fn updateDirection(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x || pcg[controlIndex()].w == 0.0) { return; }
    pcg[gid.x].z = pcg[gid.x].y * inverseDiagonal(gid.x) + pcg[controlIndex() + 1u].y * pcg[gid.x].z;
}
fn reduceLocal(local: u32) {
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride /= 2u) {
        if (local < stride) { sums[local] += sums[local + stride]; }
        workgroupBarrier();
    }
}
@compute @workgroup_size(128)
fn reduceProducts(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var product = vec4<f32>(0.0);
    if (gid.x < params.counts.x) {
        let v = pcg[gid.x];
        product = vec4<f32>(v.y * v.y * inverseDiagonal(gid.x), v.y * v.y, v.z * v.w, 0.0);
    }
    sums[local] = product;
    reduceLocal(local);
    if (local == 0u) { pcg[params.counts.x + group.x] = sums[0]; }
}
fn finishReduction(local: u32, mode: u32) {
    var value = vec4<f32>(0.0);
    for (var i = local; i < params.switches.w; i += 128u) { value += pcg[params.counts.x + i]; }
    sums[local] = value;
    reduceLocal(local);
    if (local != 0u) { return; }
    let total = sums[0];
    let index = controlIndex();
    if (mode == 0u) {
        pcg[index] = vec4<f32>(total.x, total.y, total.y, 1.0);
        pcg[index + 1u] = vec4<f32>(0.0);
    } else if (mode == 1u) {
        if (pcg[index].w == 0.0) { return; }
        if (!(total.z > 0.0) || !(total.z < 3.0e38)) {
            pcg[index].w = 0.0;
            pcg[index + 1u].w = 1.0;
            return;
        }
        pcg[index + 1u].x = pcg[index].x / total.z;
        return;
    } else if (mode == 2u) {
        if (pcg[index].w == 0.0) { return; }
        pcg[index + 1u].z += 1.0;
        pcg[index + 1u].y = total.x / pcg[index].x;
    } else {
        pcg[index + 1u].w = 0.0;
    }
    pcg[index].x = total.x;
    pcg[index].y = total.y;
    if (!(total.x >= 0.0) || !(total.y >= 0.0) || !(total.x < 3.0e38) || !(total.y < 3.0e38)) {
        pcg[index].w = 0.0;
        pcg[index + 1u].w = 2.0;
        return;
    }
    let threshold = max(params.tolerances.y, params.tolerances.x * pcg[index].z);
    pcg[index].w = f32(total.y > threshold);
}
@compute @workgroup_size(128)
fn finishInitialize(@builtin(local_invocation_index) local: u32) { finishReduction(local, 0u); }
@compute @workgroup_size(128)
fn finishDot(@builtin(local_invocation_index) local: u32) { finishReduction(local, 1u); }
@compute @workgroup_size(128)
fn finishResidual(@builtin(local_invocation_index) local: u32) { finishReduction(local, 2u); }
@compute @workgroup_size(128)
fn finishRestart(@builtin(local_invocation_index) local: u32) { finishReduction(local, 3u); }

@compute @workgroup_size(128)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.y) { return; }
    let fc = faceCoordinate(gid.x);
    let left = fc.cell - axisVector(fc.axis);
    let right = fc.cell;
    var f = faces[gid.x];
    if (f.area == 0.0) {
        if (params.modes.x != 0u) {
            f.velocity = 0.0;
            f.valid = 0u;
        } else {
            f.velocity = f.solid;
            f.valid = 1u;
        }
    } else {
        let leftLiquid = liquid(left);
        let rightLiquid = liquid(right);
        if (leftLiquid || rightLiquid) {
            var pl = 0.0;
            var pr = 0.0;
            var fraction = 1.0;
            if (leftLiquid) { pl = pcg[cellIndex(left)].x; }
            if (rightLiquid) { pr = pcg[cellIndex(right)].x; }
            if (leftLiquid && !rightLiquid) { fraction = theta(cells[cellIndex(left)].state.x, cells[cellIndex(right)].state.x); }
            if (rightLiquid && !leftLiquid) { fraction = theta(cells[cellIndex(right)].state.x, cells[cellIndex(left)].state.x); }
            f.velocity -= (pr - pl) / fraction;
            f.valid = 1u;
        } else {
            f.velocity = 0.0;
            f.valid = 0u;
        }
    }
    faces[gid.x] = f;
}
@compute @workgroup_size(128)
fn measureDivergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x || cells[gid.x].state.w == 0.0) { return; }
    let c = cellCoordinate(gid.x);
    let volume = cells[gid.x].geometry.x;
    var flux = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let lo = faces[faceIndex(c, axis)];
        let hi = faces[faceIndex(c + axisVector(axis), axis)];
        flux += hi.area * hi.velocity - (hi.area - volume) * hi.solid - lo.area * lo.velocity + (lo.area - volume) * lo.solid;
    }
    atomicMax(&lists[statusIndex(3u)], bitcast<u32>(abs(flux) / params.origin.w));
}
@compute @workgroup_size(128)
fn constrainSnapshots(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.y) { return; }
    if (faces[gid.x].area == 0.0) {
        faces[gid.x].velocity = faces[gid.x].solid;
        if ((params.modes.y & 2u) != 0u) { faces[gid.x].previous = faces[gid.x].solid; }
        faces[gid.x].valid = 1u;
    }
}

fn sampleFace(point: vec3<f32>, axis: u32, previous: bool) -> f32 {
    var q = (point - params.origin.xyz) / params.origin.w - vec3<f32>(0.5);
    q[axis] += 0.5;
    let dimensions = faceDimensions(axis);
    q = clamp(q, vec3<f32>(0.0), vec3<f32>(dimensions - vec3<i32>(1)));
    let lower = vec3<i32>(floor(q));
    let fraction = q - vec3<f32>(lower);
    var value = 0.0;
    var support = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let d = vec3<i32>(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
        let c = min(lower + d, dimensions - vec3<i32>(1));
        let w = select(vec3<f32>(1.0) - fraction, fraction, d != vec3<i32>(0));
        let weight = w.x * w.y * w.z;
        let f = faces[faceIndex(c, axis)];
        if (previous) { value += weight * f.previous; }
        else {
            value += weight * f.velocity;
            if (f.valid != 0u) { support += weight; }
        }
    }
    if (!previous && support < 0.999) { atomicOr(&lists[statusIndex(6u)], 1u); }
    return value;
}
fn sampleVelocity(p: vec3<f32>, previous: bool) -> vec3<f32> {
    return vec3<f32>(sampleFace(p, 0u, previous), sampleFace(p, 1u, previous), sampleFace(p, 2u, previous));
}
fn sampleObstacle(point: vec3<f32>) -> SolidSample {
    let q = clamp((point - params.origin.xyz) / params.origin.w, vec3<f32>(0.0), vec3<f32>(params.grid.xyz));
    let lower = min(vec3<i32>(floor(q)), vec3<i32>(params.grid.xyz) - vec3<i32>(1));
    let f = q - vec3<f32>(lower);
    var distance = 0.0;
    var gradient = vec3<f32>(0.0);
    var velocity = vec3<f32>(0.0);
    for (var corner = 0u; corner < 8u; corner++) {
        let d = vec3<i32>(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
        let w = select(vec3<f32>(1.0) - f, f, d != vec3<i32>(0));
        let sign = 2.0 * vec3<f32>(d) - vec3<f32>(1.0);
        let sample = solids[vertexIndex(lower + d)];
        let weight = w.x * w.y * w.z;
        distance += weight * sample.x;
        velocity += weight * sample.yzw;
        gradient += sample.x * vec3<f32>(sign.x * w.y * w.z, w.x * sign.y * w.z, w.x * w.y * sign.z) / params.origin.w;
    }
    return SolidSample(distance, gradient, velocity);
}
fn sampleSolid(point: vec3<f32>) -> SolidSample {
    let obstacle = sampleObstacle(point);
    var distance = obstacle.distance;
    var gradient = obstacle.gradient;
    var velocity = obstacle.velocity;
    let padding = params.geometry.y + params.geometry.z - params.settings.w;
    let lo = point - params.origin.xyz - vec3<f32>(padding);
    let hi = params.origin.xyz + vec3<f32>(params.grid.xyz) * params.origin.w - point - vec3<f32>(padding);
    for (var axis = 0u; axis < 3u; axis++) {
        if (lo[axis] < distance) {
            distance = lo[axis];
            gradient = vec3<f32>(axisVector(axis));
            velocity = vec3<f32>(0.0);
        }
        if (hi[axis] < distance) {
            distance = hi[axis];
            gradient = -vec3<f32>(axisVector(axis));
            velocity = vec3<f32>(0.0);
        }
    }
    return SolidSample(distance, gradient, velocity);
}
fn reportCollisionFailure(point: vec3<f32>) {
    // Only a genuinely inside marker may be handed to the explicitly enabled removal stage.
    if ((params.particles.w & 1u) != 0u && sampleObstacle(point).distance < 0.0) { return; }
    atomicOr(&lists[statusIndex(1u)], 1u);
}
fn correctParticle(point: vec3<f32>, velocity: vec3<f32>) -> Correction {
    var p = point;
    var v = velocity;
    var contacted = false;
    let clearance = params.settings.w;
    let tolerance = params.origin.w * 1.0e-5;
    for (var iteration = 0u; iteration < 24u; iteration++) {
        let solid = sampleSolid(p);
        if (solid.distance >= clearance - tolerance) { return Correction(p, v, contacted); }
        contacted = true;
        let magnitude = length(solid.gradient);
        if (!(magnitude > 1.0e-6)) {
            reportCollisionFailure(p);
            return Correction(p, v, contacted);
        }
        let normal = solid.gradient / magnitude;
        p += min(params.origin.w, (clearance - solid.distance) / magnitude) * normal;
        v -= min(0.0, dot(v - solid.velocity, normal)) * normal;
    }
    if (sampleSolid(p).distance < clearance - tolerance) { reportCollisionFailure(p); }
    return Correction(p, v, contacted);
}
fn advectGrid(point: vec3<f32>, h: f32, first: vec3<f32>) -> vec3<f32> {
    let second = sampleVelocity(point + 0.5 * h * first, false);
    if (params.switches.y == 0u) { return point + h * second; }
    let third = sampleVelocity(point + 0.75 * h * second, false);
    return point + h * ((2.0 / 9.0) * first + (1.0 / 3.0) * second + (4.0 / 9.0) * third);
}
fn sweepParticle(original: vec3<f32>, proposed: vec3<f32>, velocity: vec3<f32>) -> Correction {
    let padding = vec3<f32>(params.geometry.y + params.geometry.z);
    let lower = params.origin.xyz + padding;
    let upper = params.origin.xyz + vec3<f32>(params.grid.xyz) * params.origin.w - padding;
    let candidate = clamp(proposed, lower, upper);
    let distance = length(candidate - original);
    let spacing = 0.1 * params.origin.w;
    if (!(distance <= f32(params.switches.z) * spacing)) {
        atomicOr(&lists[statusIndex(5u)], 1u);
        return Correction(original, velocity, false);
    }
    let intervals = max(1u, u32(ceil(distance / spacing)));
    var lastSafe = clamp(original, lower, upper);
    for (var i = 0u; i <= intervals; i++) {
        let point = mix(original, candidate, f32(i) / f32(intervals));
        let solid = sampleObstacle(point);
        if (solid.distance < 0.0) {
            let magnitude = length(solid.gradient);
            var corrected = lastSafe;
            var fallback = true;
            if (magnitude > 1.0e-6) {
                corrected = clamp(point + (params.settings.w - solid.distance) * solid.gradient / magnitude, lower, upper);
                fallback = sampleObstacle(corrected).distance < 0.0 || length(corrected - point) > 5.0 * params.origin.w;
            }
            if (fallback) {
                corrected = lastSafe;
                atomicAdd(&lists[statusIndex(7u)], 1u);
                if (sampleObstacle(corrected).distance < 0.0) { reportCollisionFailure(corrected); }
            }
            return Correction(corrected, velocity, true);
        }
        lastSafe = clamp(point, lower, upper);
    }
    return Correction(candidate, velocity, any(candidate != proposed));
}
@compute @workgroup_size(128)
fn gridToParticles(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= particleCount()) { return; }
    let original = positions[gid.x].xyz;
    let pic = sampleVelocity(original, false);
    let delta = pic - sampleVelocity(original, true);
    var velocity = mix(velocities[gid.x].xyz + delta, pic, params.settings.x);
    var position = original;
    var contacted = false;
    if (params.modes.w != 0u) {
        let correction = sweepParticle(original, advectGrid(original, stepDt(), pic), velocity);
        position = correction.position;
        contacted = correction.contacted;
    } else {
        var correction = correctParticle(original, velocity);
        position = correction.position;
        velocity = correction.velocity;
        contacted = correction.contacted;
        var remaining = stepDt();
        for (var segment = 0u; segment < params.switches.z && remaining > 0.0; segment++) {
            let first = sampleVelocity(position, false);
            let speed = length(first);
            let h = min(remaining, 0.5 * params.origin.w / max(speed, 1.0e-12));
            correction = correctParticle(advectGrid(position, h, first), velocity);
            position = correction.position;
            velocity = correction.velocity;
            contacted = contacted || correction.contacted;
            remaining -= h;
        }
        if (remaining > 0.0) { atomicOr(&lists[statusIndex(5u)], 1u); }
    }
    let gridPosition = (position - params.origin.xyz) / params.origin.w;
    let speed = length(velocity);
    if (!all(gridPosition >= vec3<f32>(0.0)) || !all(gridPosition < vec3<f32>(params.grid.xyz))
        || !all(abs(velocity) < vec3<f32>(3.0e38)) || !(speed < 3.0e38)) {
        atomicAdd(&lists[statusIndex(0u)], 1u);
    }
    if (contacted) { atomicAdd(&lists[statusIndex(2u)], 1u); }
    positions[gid.x] = vec4<f32>(position, 1.0);
    velocities[gid.x] = vec4<f32>(velocity, 0.0);
    speeds[gid.x] = speed;
    atomicMax(&lists[statusIndex(4u)], bitcast<u32>(speed));
    if (params.particles.w != 0u) {
        var reason = 0u;
        if ((params.particles.w & 1u) != 0u && sampleObstacle(position).distance < 0.0) { reason = 1u; }
        atomicStore(&particleState[gid.x], reason);
        if ((params.particles.w & 2u) != 0u) {
            let bin = min(u32(floor(speed / params.removal.x)), params.particles.z - 1u);
            atomicAdd(&particleState[histogramIndex(bin)], 1u);
        }
    }
}

fn clearParticleRemoval(index: u32) {
    if (index < 8u) { atomicStore(&particleState[removalControl(index)], 0u); }
    if (index < params.particles.z) { atomicStore(&particleState[histogramIndex(index)], 0u); }
}
@compute @workgroup_size(128)
fn clearRemoval(@builtin(global_invocation_id) gid: vec3<u32>) { clearParticleRemoval(gid.x); }
@compute @workgroup_size(128)
fn countExtremeGroups(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= particleCount()) { return; }
    let maximum = bitcast<f32>(atomicLoad(&lists[statusIndex(4u)]));
    let speed = speeds[gid.x];
    if (speed >= 0.99999 * maximum) {
        atomicAdd(&particleState[removalControl(4u)], 1u);
    } else if (speed >= 0.9 * maximum) {
        atomicAdd(&particleState[removalControl(5u)], 1u);
    }
}
@compute @workgroup_size(1)
fn chooseExtremeThreshold() {
    let bins = params.particles.z;
    let interval = params.removal.x;
    let allowance = min(u32(floor(0.0005 * f32(particleCount()))), 35u);
    var accepted = 0u;
    var limit = f32(bins) * interval;
    for (var bin = bins - 1u; bin > 0u; bin--) {
        let count = atomicLoad(&particleState[histogramIndex(bin)]);
        if (accepted + count > allowance) { break; }
        accepted += count;
        // Native policy deliberately uses max here, not a descending minimum.
        limit = f32(max(bin + 4u, bins)) * interval;
    }
    let maximum = bitcast<f32>(atomicLoad(&lists[statusIndex(4u)]));
    if (atomicLoad(&particleState[removalControl(4u)]) <= 6u && atomicLoad(&particleState[removalControl(5u)]) <= 6u) {
        limit = min(limit, 0.99999 * maximum);
    }
    atomicStore(&particleState[removalControl(3u)], bitcast<u32>(limit));
}
@compute @workgroup_size(128)
fn scanSurvivors(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var keep = 0u;
    if (gid.x < particleCount()) {
        var reason = atomicLoad(&particleState[gid.x]);
        if (reason == 0u && (params.particles.w & 2u) != 0u) {
            let limit = bitcast<f32>(atomicLoad(&particleState[removalControl(3u)]));
            let velocity = velocities[gid.x].xyz;
            if (dot(velocity, velocity) > limit * limit) { reason = 2u; }
        }
        atomicStore(&particleState[gid.x], reason);
        if (reason == 1u) { atomicAdd(&particleState[removalControl(1u)], 1u); }
        if (reason == 2u) { atomicAdd(&particleState[removalControl(2u)], 1u); }
        keep = u32(reason == 0u);
    }
    survivorScan[local] = keep;
    workgroupBarrier();
    for (var offset = 1u; offset < 128u; offset *= 2u) {
        var preceding = 0u;
        if (local >= offset) { preceding = survivorScan[local - offset]; }
        workgroupBarrier();
        survivorScan[local] += preceding;
        workgroupBarrier();
    }
    if (gid.x < particleCount()) {
        atomicStore(&particleState[params.particles.x + gid.x], survivorScan[local] - keep);
    }
    if (local == 127u) {
        atomicStore(&particleState[2u * params.particles.x + group.x], survivorScan[127u]);
    }
}
@compute @workgroup_size(1)
fn scanSurvivorGroups() {
    let groups = (particleCount() + 127u) / 128u;
    var total = 0u;
    for (var group = 0u; group < groups; group++) {
        let index = 2u * params.particles.x + group;
        let count = atomicLoad(&particleState[index]);
        atomicStore(&particleState[index], total);
        total += count;
    }
    atomicStore(&particleState[removalControl(0u)], total);
    atomicStore(&particleState[removalControl(7u)], atomicLoad(&lists[statusIndex(4u)]));
}
@compute @workgroup_size(128)
fn packSurvivors(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= particleCount() || atomicLoad(&particleState[gid.x]) != 0u) { return; }
    let index = atomicLoad(&particleState[params.particles.x + gid.x]) + atomicLoad(&particleState[2u * params.particles.x + gid.x / 128u]);
    particleScratch[2u * index] = positions[gid.x];
    particleScratch[2u * index + 1u] = vec4<f32>(velocities[gid.x].xyz, speeds[gid.x]);
}
@compute @workgroup_size(128)
fn commitSurvivors(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= particleCount()) { return; }
    if (gid.x < atomicLoad(&particleState[removalControl(0u)])) {
        let position = particleScratch[2u * gid.x];
        let velocityAndSpeed = particleScratch[2u * gid.x + 1u];
        positions[gid.x] = position;
        velocities[gid.x] = vec4<f32>(velocityAndSpeed.xyz, 0.0);
        let speed = velocityAndSpeed.w;
        speeds[gid.x] = speed;
        atomicMax(&particleState[removalControl(6u)], bitcast<u32>(speed));
    } else {
        positions[gid.x] = vec4<f32>(0.0);
        velocities[gid.x] = vec4<f32>(0.0);
        speeds[gid.x] = 0.0;
    }
}
`;
}

/** Entry points whose generated count/timestep helpers read GPU runtime storage. */
export const FLIP_REFERENCE_RUNTIME_ENTRIES: readonly string[] = [
    "linkParticles",
    "snapshotAndForce",
    "gridToParticles",
    "countExtremeGroups",
    "chooseExtremeThreshold",
    "scanSurvivors",
    "scanSurvivorGroups",
    "packSurvivors",
    "commitSurvivors",
];

/** Storage bindings actually reachable by each entry point (automatic pipeline layouts). */
export const FLIP_REFERENCE_BINDINGS: Readonly<Record<string, readonly number[]>> = {
    clearLists: [0, 6],
    linkParticles: [0, 1, 6],
    liquidLevelSet: [0, 1, 5, 6, 7],
    particleToGrid: [0, 1, 2, 4, 6, 7],
    extendBeforeToB: [0, 4],
    extendBeforeToA: [0, 4],
    extendToB: [0, 4],
    extendToA: [0, 4],
    copyToA: [0, 4],
    snapshotAndForce: [0, 4, 5],
    buildMatrix: [0, 4, 5, 7],
    conditionSolidVelocities: [0, 4, 5],
    initializeCg: [0, 5, 8],
    applyDirection: [0, 5, 8],
    trueResidual: [0, 5, 8],
    updateResidual: [0, 8],
    updateDirection: [0, 5, 8],
    reduceProducts: [0, 5, 8],
    finishInitialize: [0, 8],
    finishDot: [0, 8],
    finishResidual: [0, 8],
    finishRestart: [0, 8],
    project: [0, 4, 5, 8],
    measureDivergence: [0, 4, 5, 6],
    constrainSnapshots: [0, 4],
    gridToParticles: [0, 1, 2, 3, 4, 6, 7, 10],
    clearRemoval: [0, 10],
    countExtremeGroups: [0, 3, 6, 10],
    chooseExtremeThreshold: [0, 6, 10],
    scanSurvivors: [0, 2, 10],
    scanSurvivorGroups: [0, 6, 10],
    packSurvivors: [0, 1, 2, 3, 9, 10],
    commitSurvivors: [0, 1, 2, 3, 9, 10],
};
