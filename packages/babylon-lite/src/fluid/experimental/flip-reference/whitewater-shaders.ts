import { FOAM_COMMON_WGSL } from "../../core/sim-common.js";
import { flipReferenceWgsl } from "./shaders.js";

const WHITEWATER_WGSL = /* wgsl */ `
${FOAM_COMMON_WGSL}
struct WhitewaterField {
    normalCurvature: vec4<f32>,
    phiTurbulenceInterfaceOpen: vec4<f32>,
}
@group(0) @binding(12) var<uniform> foam: Foam;
@group(0) @binding(13) var<storage, read_write> whitewaterField: array<WhitewaterField>;
@group(0) @binding(14) var<storage, read_write> workingPool: array<Diffuse>;
@group(0) @binding(15) var<storage, read_write> workingState: array<atomic<u32>>;
@group(0) @binding(16) var<storage, read_write> workingDispatch: array<u32>;
@group(0) @binding(17) var<storage, read_write> publishedPool: array<Diffuse>;
@group(0) @binding(18) var<storage, read_write> publishedState: array<atomic<u32>>;
@group(0) @binding(19) var<storage, read_write> publishedDraw: array<u32>;
@group(0) @binding(20) var<storage, read_write> publishedDispatch: array<u32>;

var<workgroup> publicationCounts: array<atomic<u32>, 4>;

fn wwActiveStride(capacity: u32) -> u32 {
    return ((capacity + 63u) / 64u) * 64u;
}
fn wwListBase(side: u32, capacity: u32) -> u32 {
    return 64u + side * wwActiveStride(capacity);
}
fn wwFreeBase(capacity: u32) -> u32 {
    return 64u + 2u * wwActiveStride(capacity);
}
fn wwFrameDt() -> f32 {
    return bitcast<f32>(atomicLoad(&runtime[19]));
}
fn wwFrameValid() -> bool {
    let dt = wwFrameDt();
    return atomicLoad(&runtime[1]) == 0u && dt > 0.0 && dt < 3.0e38;
}
fn wwInvocationIndex(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 64u;
}
fn wwWriteDispatch(offset: u32, count: u32) {
    let groups = (count + 63u) / 64u;
    let limit = max(1u, atomicLoad(&workingState[13]));
    let x = min(groups, limit);
    workingDispatch[offset] = x;
    workingDispatch[offset + 1u] = 1u;
    if (x > 0u) {
        workingDispatch[offset + 1u] = (groups + x - 1u) / x;
    }
    workingDispatch[offset + 2u] = 1u;
}
fn wwWritePublishedDispatch(count: u32) {
    let groups = (count + 63u) / 64u;
    let limit = max(1u, atomicLoad(&workingState[13]));
    let x = min(groups, limit);
    publishedDispatch[0] = x;
    publishedDispatch[1] = 1u;
    if (x > 0u) {
        publishedDispatch[1] = (groups + x - 1u) / x;
    }
    publishedDispatch[2] = 1u;
}
fn wwPhiAt(c: vec3<i32>, fallback: f32) -> f32 {
    if (!inGrid(c)) {
        return max(3.0, fallback);
    }
    return cells[cellIndex(c)].state.x;
}
fn wwRawNormal(c: vec3<i32>) -> vec4<f32> {
    if (!inGrid(c)) {
        return vec4<f32>(0.0);
    }
    let center = cells[cellIndex(c)].state.x;
    let gradient = 0.5 * vec3<f32>(
        wwPhiAt(c + vec3<i32>(1, 0, 0), center) - wwPhiAt(c - vec3<i32>(1, 0, 0), center),
        wwPhiAt(c + vec3<i32>(0, 1, 0), center) - wwPhiAt(c - vec3<i32>(0, 1, 0), center),
        wwPhiAt(c + vec3<i32>(0, 0, 1), center) - wwPhiAt(c - vec3<i32>(0, 0, 1), center)
    );
    let magnitude = length(gradient);
    let nearInterface = abs(center) <= 1.5;
    return vec4<f32>(select(vec3<f32>(0.0), -gradient / magnitude, nearInterface && magnitude > 1.0e-6), select(0.0, magnitude, nearInterface));
}
fn wwFaceVelocity(c: vec3<i32>, axis: u32) -> f32 {
    let dimensions = faceDimensions(axis);
    if (any(c < vec3<i32>(0)) || any(c >= dimensions)) {
        return 0.0;
    }
    let face = faces[faceIndex(c, axis)];
    return select(0.0, face.velocity, face.valid != 0u);
}
fn wwCellVelocity(c: vec3<i32>) -> vec3<f32> {
    if (!inGrid(c)) {
        return vec3<f32>(0.0);
    }
    return 0.5 * vec3<f32>(
        wwFaceVelocity(c, 0u) + wwFaceVelocity(c + vec3<i32>(1, 0, 0), 0u),
        wwFaceVelocity(c, 1u) + wwFaceVelocity(c + vec3<i32>(0, 1, 0), 1u),
        wwFaceVelocity(c, 2u) + wwFaceVelocity(c + vec3<i32>(0, 0, 1), 2u)
    );
}
fn wwCellTurbulence(c: vec3<i32>) -> f32 {
    if (foam.kTurb <= 0.0) {
        return 0.0;
    }
    let invTwoDx = 0.5 / params.origin.w;
    let dVdx = (wwCellVelocity(c + vec3<i32>(1, 0, 0)) - wwCellVelocity(c - vec3<i32>(1, 0, 0))) * invTwoDx;
    let dVdy = (wwCellVelocity(c + vec3<i32>(0, 1, 0)) - wwCellVelocity(c - vec3<i32>(0, 1, 0))) * invTwoDx;
    let dVdz = (wwCellVelocity(c + vec3<i32>(0, 0, 1)) - wwCellVelocity(c - vec3<i32>(0, 0, 1))) * invTwoDx;
    let curl = vec3<f32>(dVdy.z - dVdz.y, dVdz.x - dVdx.z, dVdx.y - dVdy.x);
    let sxy = 0.5 * (dVdx.y + dVdy.x);
    let sxz = 0.5 * (dVdx.z + dVdz.x);
    let syz = 0.5 * (dVdy.z + dVdz.y);
    let strainSquared = dVdx.x * dVdx.x + dVdy.y * dVdy.y + dVdz.z * dVdz.z + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
    return params.origin.w * sqrt(max(0.0, dot(curl, curl) + 2.0 * strainSquared));
}
fn wwSafeNormal(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let squared = dot(value, value);
    return select(fallback, value * inverseSqrt(squared), squared > 1.0e-12);
}
fn wwSampleField(world: vec3<f32>) -> WhitewaterField {
    if (atomicLoad(&runtime[0]) == 0u) {
        return WhitewaterField(vec4<f32>(0.0), vec4<f32>(3.0, 0.0, 0.0, 0.0));
    }
    let grid = (world - params.origin.xyz) / params.origin.w - vec3<f32>(0.5);
    let base = vec3<i32>(floor(grid));
    let fraction = grid - floor(grid);
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    for (var corner = 0u; corner < 8u; corner++) {
        let d = vec3<i32>(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
        let w3 = select(vec3<f32>(1.0) - fraction, fraction, d != vec3<i32>(0));
        let weight = w3.x * w3.y * w3.z;
        let c = base + d;
        if (inGrid(c)) {
            let sample = whitewaterField[cellIndex(c)];
            a += sample.normalCurvature * weight;
            b += sample.phiTurbulenceInterfaceOpen * weight;
        } else {
            b.x += 3.0 * weight;
        }
    }
    return WhitewaterField(a, b);
}
fn wwSampleFoamLayer(world: vec3<f32>) -> WhitewaterField {
    var best = wwSampleField(world);
    var outward = -wwSafeNormal(best.normalCurvature.xyz, vec3<f32>(0.0, 1.0, 0.0));
    var score = best.phiTurbulenceInterfaceOpen.z * smoothstep(0.1, 0.45, outward.y);
    if (foam.foamLayerDepth <= 0.0) {
        return best;
    }
    let cell = clamp(vec3<i32>(floor((world - params.origin.xyz) / params.origin.w)), vec3<i32>(0), vec3<i32>(params.grid.xyz) - vec3<i32>(1));
    for (var layer = 1; layer <= 4; layer++) {
        let distance = f32(layer);
        if (distance > foam.foamLayerDepth + 0.5) {
            continue;
        }
        let candidateCell = cell + vec3<i32>(0, layer, 0);
        if (!inGrid(candidateCell)) {
            continue;
        }
        var candidate = whitewaterField[cellIndex(candidateCell)];
        let layerWeight = 1.0 - smoothstep(max(0.0, foam.foamLayerDepth - 0.5), foam.foamLayerDepth + 0.5, distance);
        candidate.phiTurbulenceInterfaceOpen.z *= layerWeight;
        outward = -wwSafeNormal(candidate.normalCurvature.xyz, vec3<f32>(0.0, 1.0, 0.0));
        let candidateScore = candidate.phiTurbulenceInterfaceOpen.z * smoothstep(0.1, 0.45, outward.y);
        if (candidateScore > score) {
            best = candidate;
            score = candidateScore;
        }
    }
    return best;
}
fn wwSampleComponent(world: vec3<f32>, axis: u32) -> f32 {
    if (atomicLoad(&runtime[0]) == 0u) {
        return 0.0;
    }
    var grid = (world - params.origin.xyz) / params.origin.w - vec3<f32>(0.5);
    grid[axis] += 0.5;
    let base = vec3<i32>(floor(grid));
    let fraction = grid - floor(grid);
    let dimensions = faceDimensions(axis);
    var weighted = 0.0;
    var weightSum = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let d = vec3<i32>(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
        let c = base + d;
        if (any(c < vec3<i32>(0)) || any(c >= dimensions)) {
            continue;
        }
        let w3 = select(vec3<f32>(1.0) - fraction, fraction, d != vec3<i32>(0));
        let weight = w3.x * w3.y * w3.z;
        let face = faces[faceIndex(c, axis)];
        if (face.valid != 0u) {
            weighted += weight * face.velocity;
            weightSum += weight;
        }
    }
    return select(0.0, weighted / weightSum, weightSum > 1.0e-6);
}
fn wwSampleVelocity(world: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(wwSampleComponent(world, 0u), wwSampleComponent(world, 1u), wwSampleComponent(world, 2u));
}
fn wwInsideDomain(world: vec3<f32>) -> bool {
    let upper = params.origin.xyz + vec3<f32>(params.grid.xyz) * params.origin.w;
    return all(world >= params.origin.xyz) && all(world <= upper);
}
fn wwClassify(world: vec3<f32>, previousKind: u32) -> u32 {
    let field = wwSampleField(world);
    let surface = wwSampleFoamLayer(world);
    let outward = -wwSafeNormal(surface.normalCurvature.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let strength = surface.phiTurbulenceInterfaceOpen.z;
    let enterFoam = strength >= 0.05 && outward.y >= 0.45;
    let keepFoam = previousKind == 1u && strength >= 0.025 && outward.y >= 0.2;
    if (enterFoam || keepFoam) {
        return 1u;
    }
    if (field.phiTurbulenceInterfaceOpen.x >= 0.0 || field.phiTurbulenceInterfaceOpen.w <= 1.0e-5) {
        return 0u;
    }
    return 2u;
}
fn wwPushFree(slot: u32) {
    let capacity = atomicLoad(&workingState[4]);
    let destination = atomicAdd(&workingState[5], 1u);
    if (destination < capacity) {
        atomicStore(&workingState[wwFreeBase(capacity) + destination], slot);
    } else {
        atomicSub(&workingState[5], 1u);
        atomicAdd(&workingState[7], 1u);
    }
}
fn wwPopFree() -> u32 {
    loop {
        let available = atomicLoad(&workingState[5]);
        if (available == 0u) {
            atomicAdd(&workingState[7], 1u);
            return 0xffffffffu;
        }
        let result = atomicCompareExchangeWeak(&workingState[5], available, available - 1u);
        if (result.exchanged) {
            let capacity = atomicLoad(&workingState[4]);
            return atomicLoad(&workingState[wwFreeBase(capacity) + available - 1u]);
        }
    }
}
fn wwAppend(slot: u32, side: u32) -> bool {
    let capacity = atomicLoad(&workingState[4]);
    let destination = atomicAdd(&workingState[1u + side], 1u);
    if (destination < capacity) {
        atomicStore(&workingState[wwListBase(side, capacity) + destination], slot);
        return true;
    }
    atomicSub(&workingState[1u + side], 1u);
    atomicAdd(&workingState[7], 1u);
    return false;
}
fn wwKill(slot: u32, position: vec3<f32>) {
    workingPool[slot].p = vec4<f32>(position, 0.0);
    wwPushFree(slot);
}

@compute @workgroup_size(64)
fn initializeFlipReferenceWhitewater(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = wwInvocationIndex(gid, groups);
    let capacity = arrayLength(&workingPool);
    if (index < capacity) {
        atomicStore(&workingState[wwFreeBase(capacity) + index], index);
        atomicStore(&publishedState[wwListBase(0u, capacity) + index], index);
    }
    if (index == 0u) {
        atomicStore(&workingState[0], 0u);
        atomicStore(&workingState[1], 0u);
        atomicStore(&workingState[2], 0u);
        atomicStore(&workingState[3], 0u);
        atomicStore(&workingState[4], capacity);
        atomicStore(&workingState[5], capacity);
        atomicStore(&workingState[6], 0u);
        atomicStore(&workingState[7], 0u);
        atomicStore(&publishedState[1], 0u);
        atomicStore(&publishedState[2], 0u);
        atomicStore(&publishedState[3], 0u);
        atomicStore(&publishedState[4], capacity);
        publishedDraw[0] = 6u;
        publishedDraw[1] = 0u;
        publishedDraw[2] = 0u;
        publishedDraw[3] = 0u;
    }
}

@compute @workgroup_size(1)
fn prepareFlipReferenceWhitewater() {
    if (!wwFrameValid()) {
        wwWriteDispatch(0u, 0u);
        wwWriteDispatch(3u, 0u);
        wwWriteDispatch(6u, 0u);
        wwWritePublishedDispatch(0u);
        return;
    }
    atomicAdd(&workingState[0], 1u);
    let side = atomicLoad(&workingState[3]);
    let compactMode = atomicLoad(&workingState[12]) != 0u;
    let updateCount = select(atomicLoad(&workingState[4]), atomicLoad(&workingState[1u + side]), compactMode);
    wwWriteDispatch(0u, arrayLength(&whitewaterField));
    wwWriteDispatch(3u, updateCount);
    wwWriteDispatch(6u, atomicLoad(&runtime[0]));
    wwWritePublishedDispatch(atomicLoad(&workingState[4]));
}

@compute @workgroup_size(64)
fn prepareFlipReferenceWhitewaterField(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = wwInvocationIndex(gid, groups);
    if (index >= arrayLength(&whitewaterField) || !wwFrameValid()) {
        return;
    }
    let c = cellCoordinate(index);
    if (atomicLoad(&runtime[0]) == 0u) {
        whitewaterField[index] = WhitewaterField(vec4<f32>(0.0), vec4<f32>(3.0, 0.0, 0.0, 0.0));
        return;
    }
    let center = cells[index].state.x;
    let normal = wwRawNormal(c);
    let openVolume = cellOpenVolume(c);
    let divergence =
        wwRawNormal(c + vec3<i32>(1, 0, 0)).x - wwRawNormal(c - vec3<i32>(1, 0, 0)).x
        + wwRawNormal(c + vec3<i32>(0, 1, 0)).y - wwRawNormal(c - vec3<i32>(0, 1, 0)).y
        + wwRawNormal(c + vec3<i32>(0, 0, 1)).z - wwRawNormal(c - vec3<i32>(0, 0, 1)).z;
    let interfaceStrength = select(0.0, normal.w, openVolume > 1.0e-5);
    let curvature = select(0.0, -0.5 * divergence, interfaceStrength > 1.0e-6);
    whitewaterField[index] = WhitewaterField(vec4<f32>(normal.xyz, curvature), vec4<f32>(center, wwCellTurbulence(c), interfaceStrength, openVolume));
}

@compute @workgroup_size(64)
fn updateFlipReferenceWhitewater(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let item = wwInvocationIndex(gid, groups);
    if (!wwFrameValid()) {
        return;
    }
    let capacity = atomicLoad(&workingState[4]);
    let oldSide = atomicLoad(&workingState[3]);
    let compactMode = atomicLoad(&workingState[12]) != 0u;
    var slot = item;
    if (compactMode) {
        let count = atomicLoad(&workingState[1u + oldSide]);
        if (item >= count) {
            return;
        }
        slot = atomicLoad(&workingState[wwListBase(oldSide, capacity) + item]);
    } else if (slot >= capacity) {
        return;
    }
    let current = workingPool[slot].p;
    if (current.w <= 0.0) {
        return;
    }
    let position = current.xyz;
    if (!wwInsideDomain(position)) {
        wwKill(slot, position);
        return;
    }
    let previousKind = u32(clamp(round(workingPool[slot].v.w), 0.0, 2.0));
    let kind = wwClassify(position, previousKind);
    if (!foamKindEnabled(kind)) {
        wwKill(slot, position);
        return;
    }
    let dt = wwFrameDt();
    let fluidVelocity = wwSampleVelocity(position);
    var velocity = workingPool[slot].v.xyz;
    var next = position;
    var lifetime = current.w;
    if (kind == 0u) {
        velocity += params.gravity.xyz * dt;
        if (foam.sprayDrag > 0.0) {
            velocity *= exp(-foam.sprayDrag * dt);
        }
        next += velocity * dt;
    } else if (kind == 2u) {
        velocity -= foam.kb * params.gravity.xyz * dt;
        velocity += foam.kd * (fluidVelocity - velocity);
        next += velocity * dt;
    } else {
        velocity = fluidVelocity;
        next += fluidVelocity * dt;
        lifetime -= dt;
    }
    if (!wwInsideDomain(next) || lifetime <= 0.0) {
        wwKill(slot, next);
        return;
    }
    let obstacle = sampleObstacle(next);
    if (obstacle.distance < 0.02 * params.origin.w) {
        let magnitude = length(obstacle.gradient);
        if (magnitude <= 1.0e-6) {
            wwKill(slot, next);
            return;
        }
        let normal = obstacle.gradient / magnitude;
        next += (0.02 * params.origin.w - obstacle.distance) * normal;
        velocity -= min(0.0, dot(velocity - obstacle.velocity, normal)) * normal;
    }
    if (!wwInsideDomain(next)) {
        wwKill(slot, next);
        return;
    }
    workingPool[slot].p = vec4<f32>(next, lifetime);
    workingPool[slot].v = vec4<f32>(velocity, f32(kind));
    if (!wwAppend(slot, 1u - oldSide)) {
        wwKill(slot, next);
    }
}

@compute @workgroup_size(64)
fn emitFlipReferenceWhitewater(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = wwInvocationIndex(gid, groups);
    if (!wwFrameValid() || index >= atomicLoad(&runtime[0])) {
        return;
    }
    if (foam.generateSpray == 0u && foam.generateFoam == 0u && foam.generateBubbles == 0u) {
        return;
    }
    let position = positions[index].xyz;
    let velocity = velocities[index].xyz;
    let speed = length(velocity);
    if (speed < 1.0e-4) {
        return;
    }
    let surface = wwSampleField(position);
    let interfaceStrength = surface.phiTurbulenceInterfaceOpen.z;
    if (interfaceStrength < 0.05) {
        return;
    }
    let outward = -wwSafeNormal(surface.normalCurvature.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let topWeight = smoothstep(0.15, 0.65, outward.y);
    if (topWeight <= 0.0) {
        return;
    }
    let normalSpeed = dot(velocity, outward);
    let trappedAir = max(0.0, -normalSpeed);
    let waveCrest = max(0.0, surface.normalCurvature.w) * max(0.0, normalSpeed);
    let trappedPotential = phi(trappedAir, foam.tauTaMin, foam.tauTaMax);
    let crestPotential = phi(waveCrest, foam.curvatureMin, foam.curvatureMax);
    var turbulencePotential = 0.0;
    if (foam.kTurb > 0.0) {
        turbulencePotential = phi(surface.phiTurbulenceInterfaceOpen.y, foam.turbulenceMin, foam.turbulenceMax);
    }
    let energy = phi(speed, foam.energySpeedMin, foam.energySpeedMax);
    if (energy <= 0.0) {
        return;
    }
    let expected = min(8.0, topWeight * energy * (foam.kTa * trappedPotential + foam.kWc * crestPotential + foam.kTurb * turbulencePotential) * wwFrameDt());
    let whole = floor(expected);
    let frameSeed = atomicLoad(&workingState[0]);
    let births = i32(whole) + select(0, 1, fRnd((index * 2246822519u) ^ (frameSeed * 22695477u)) < expected - whole);
    if (births <= 0) {
        return;
    }
    let potential = max(trappedPotential, max(crestPotential, turbulencePotential));
    let lifetime = mix(foam.tMin, foam.tMax, potential);
    let axis = velocity / speed;
    var tangent = cross(axis, vec3<f32>(0.0, 1.0, 0.0));
    if (dot(tangent, tangent) <= 1.0e-6) {
        tangent = cross(axis, vec3<f32>(1.0, 0.0, 0.0));
    }
    tangent = normalize(tangent);
    let bitangent = cross(axis, tangent);
    let oldSide = atomicLoad(&workingState[3]);
    let nextSide = 1u - oldSide;
    let travel = speed * wwFrameDt();
    for (var sample = 0; sample < births; sample++) {
        let seed = (index * 2654435761u) ^ (frameSeed * 40503u) ^ (u32(sample) * 2246822519u);
        let radius = foam.rv * sqrt(fRnd(seed));
        let angle = 6.28318530718 * fRnd(seed * 3u + 1u);
        let offset = tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle));
        let candidate = position + offset + axis * (fRnd(seed * 7u + 5u) * travel);
        if (!wwInsideDomain(candidate)) {
            continue;
        }
        let kind = wwClassify(candidate, 1u);
        if (!foamKindEnabled(kind)) {
            continue;
        }
        let slot = wwPopFree();
        if (slot == 0xffffffffu) {
            return;
        }
        workingPool[slot].p = vec4<f32>(candidate, lifetime);
        workingPool[slot].v = vec4<f32>(velocity + offset, f32(kind));
        if (!wwAppend(slot, nextSide)) {
            workingPool[slot].p.w = 0.0;
            wwPushFree(slot);
            return;
        }
    }
}

@compute @workgroup_size(1)
fn finishFlipReferenceWhitewater() {
    if (!wwFrameValid()) {
        return;
    }
    let oldSide = atomicLoad(&workingState[3]);
    let nextSide = 1u - oldSide;
    let count = atomicLoad(&workingState[1u + nextSide]);
    atomicStore(&workingState[3], nextSide);
    atomicStore(&workingState[1u + oldSide], 0u);
    atomicAdd(&workingState[6], 1u);
    atomicStore(&publishedState[1], count);
    atomicStore(&publishedState[2], 0u);
    atomicStore(&publishedState[3], 0u);
    for (var index = 8u; index < 12u; index++) {
        atomicStore(&publishedState[index], 0u);
    }
    publishedDraw[0] = 6u;
    publishedDraw[1] = count;
    publishedDraw[2] = 0u;
    publishedDraw[3] = 0u;
}

@compute @workgroup_size(64)
fn publishFlipReferenceWhitewater(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>
) {
    let frameValid = wwFrameValid();
    if (lid.x < 4u) {
        atomicStore(&publicationCounts[lid.x], 0u);
    }
    workgroupBarrier();
    let index = wwInvocationIndex(gid, groups);
    let capacity = atomicLoad(&workingState[4]);
    let side = atomicLoad(&workingState[3]);
    let count = atomicLoad(&workingState[1u + side]);
    if (frameValid && index < capacity) {
        if (index < count) {
            let slot = atomicLoad(&workingState[wwListBase(side, capacity) + index]);
            let particle = workingPool[slot];
            publishedPool[index] = particle;
            let kind = u32(clamp(round(particle.v.w), 0.0, 2.0));
            atomicAdd(&publicationCounts[0], 1u);
            atomicAdd(&publicationCounts[1u + kind], 1u);
        } else {
            publishedPool[index] = Diffuse(vec4<f32>(0.0), vec4<f32>(0.0));
        }
    }
    workgroupBarrier();
    if (frameValid && lid.x < 4u) {
        atomicAdd(&publishedState[8u + lid.x], atomicLoad(&publicationCounts[lid.x]));
    }
}
`;

/** @internal Shared Reference solver helpers plus isolated whitewater entry points. */
export function flipReferenceWhitewaterWgsl(): string {
    return flipReferenceWgsl(true) + WHITEWATER_WGSL;
}

/** @internal Per-entry bindings remain below WebGPU's eight-storage-buffer stage floor. */
export const FLIP_REFERENCE_WHITEWATER_BINDINGS: Readonly<Record<string, readonly number[]>> = {
    initializeFlipReferenceWhitewater: [14, 15, 18, 19],
    prepareFlipReferenceWhitewater: [11, 13, 15, 16, 20],
    prepareFlipReferenceWhitewaterField: [0, 4, 5, 7, 11, 12, 13],
    updateFlipReferenceWhitewater: [0, 4, 7, 11, 12, 13, 14, 15],
    emitFlipReferenceWhitewater: [0, 1, 2, 11, 12, 13, 14, 15],
    finishFlipReferenceWhitewater: [11, 15, 18, 19],
    publishFlipReferenceWhitewater: [11, 14, 15, 17, 18],
};
