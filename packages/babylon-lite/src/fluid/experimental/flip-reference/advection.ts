/** Shared by the CPU-controlled oracle and GPU-resident Reference backend. */
export const FLIP_REFERENCE_ADVECTION_WGSL = /* wgsl */ `
struct AdvectionTrial {
    position: vec3<f32>,
    supported: bool,
    finite: bool,
    excursion: f32,
    error: f32,
}
struct SweepResult {
    correction: Correction,
    samples: u32,
    complete: bool,
}
fn finiteAdvectionVector(value: vec3<f32>) -> bool {
    return all(abs(value) < vec3<f32>(3.0e38));
}
fn trialGridAdvection(point: vec3<f32>, h: f32, first: vec3<f32>, refined: bool) -> AdvectionTrial {
    let secondPoint = point + 0.5 * h * first;
    if (!finiteAdvectionVector(secondPoint)) {
        return AdvectionTrial(point, true, false, 0.0, 0.0);
    }
    let second = sampleVelocityWithSupport(secondPoint, false);
    var delta = h * second.xyz;
    var supported = second.w >= 0.999;
    var finite = finiteAdvectionVector(second.xyz);
    var excursion = 0.0;
    var error = 0.0;
    if (refined) {
        excursion = max(length(h * first), length(delta));
    }
    if (params.switches.y != 0u) {
        let thirdPoint = point + 0.75 * h * second.xyz;
        if (!finiteAdvectionVector(thirdPoint)) {
            return AdvectionTrial(point, supported, false, 0.0, 0.0);
        }
        let third = sampleVelocityWithSupport(thirdPoint, false);
        delta = h * ((2.0 / 9.0) * first + (1.0 / 3.0) * second.xyz + (4.0 / 9.0) * third.xyz);
        supported = supported && third.w >= 0.999;
        finite = finite && finiteAdvectionVector(third.xyz);
        if (refined) {
            excursion = max(excursion, length(h * third.xyz));
        }
        error = length(delta - h * second.xyz);
    } else {
        error = length(delta - h * first);
    }
    let proposed = point + delta;
    finite = finite && finiteAdvectionVector(proposed) && excursion < 3.0e38 && error < 3.0e38;
    return AdvectionTrial(proposed, supported, finite, excursion, error);
}
fn advectGrid(point: vec3<f32>, h: f32, first: vec3<f32>) -> vec3<f32> {
    let trial = trialGridAdvection(point, h, first, false);
    if (!trial.supported) { atomicOr(&lists[statusIndex(6u)], 1u); }
    if (!trial.finite) { atomicOr(&lists[statusIndex(0u)], 1u); }
    return trial.position;
}
fn sweptCandidate(proposed: vec3<f32>) -> vec3<f32> {
    let padding = vec3<f32>(params.geometry.y + params.geometry.z);
    return clamp(proposed, params.origin.xyz + padding, params.origin.xyz + vec3<f32>(params.grid.xyz) * params.origin.w - padding);
}
fn sweepParticle(original: vec3<f32>, proposed: vec3<f32>, velocity: vec3<f32>, sampleBudget: u32) -> SweepResult {
    let candidate = sweptCandidate(proposed);
    let distance = length(candidate - original);
    let intervals = max(1.0, ceil(distance / (0.1 * params.origin.w)));
    var lastSafe = sweptCandidate(original);
    for (var i = 0u; i < sampleBudget; i++) {
        let point = mix(original, candidate, f32(i) / intervals);
        let solid = sampleObstacle(point);
        if (solid.distance < 0.0) {
            let magnitude = length(solid.gradient);
            var corrected = lastSafe;
            var fallback = true;
            if (magnitude > 1.0e-6) {
                corrected = sweptCandidate(point + (params.settings.w - solid.distance) * solid.gradient / magnitude);
                fallback = sampleObstacle(corrected).distance < 0.0 || length(corrected - point) > 5.0 * params.origin.w;
            }
            if (fallback) {
                corrected = lastSafe;
                atomicAdd(&lists[statusIndex(7u)], 1u);
                if (sampleObstacle(corrected).distance < 0.0) { reportCollisionFailure(corrected); }
            }
            return SweepResult(Correction(corrected, velocity, true), i + 1u, true);
        }
        lastSafe = sweptCandidate(point);
        if (f32(i) >= intervals) {
            return SweepResult(Correction(candidate, velocity, any(candidate != proposed)), i + 1u, true);
        }
    }
    return SweepResult(Correction(original, velocity, false), sampleBudget, false);
}
fn advectSweptParticle(original: vec3<f32>, dt: f32, initialSample: vec4<f32>, velocity: vec3<f32>) -> Correction {
    if (!finiteAdvectionVector(initialSample.xyz)) {
        atomicOr(&lists[statusIndex(0u)], 1u);
        return Correction(original, velocity, false);
    }
    if (!(initialSample.w >= 0.999)) {
        atomicOr(&lists[statusIndex(6u)], 1u);
        return Correction(original, velocity, false);
    }
    var position = original;
    var first = initialSample.xyz;
    var remaining = dt;
    var h = dt;
    var samplesLeft = params.switches.z + 1u;
    var refined = false;
    var lastFailure = 5u;
    for (var attempt = 0u; attempt < params.switches.z; attempt++) {
        if (!(h > 0.0) || !(remaining - h < remaining)) { break; }
        let trial = trialGridAdvection(position, h, first, refined);
        if (!trial.finite || !trial.supported) {
            lastFailure = select(6u, 0u, !trial.finite);
            refined = true;
            h *= 0.5;
            continue;
        }
        lastFailure = 5u;
        if (refined && (trial.excursion > 0.5 * params.origin.w || trial.error > 0.01 * params.origin.w)) {
            h *= 0.5;
            continue;
        }
        let distance = length(sweptCandidate(trial.position) - position);
        if (!(distance >= 0.0) || !(distance < 3.0e38)) {
            lastFailure = 0u;
            refined = true;
            h *= 0.5;
            continue;
        }
        let requiredSamples = max(1.0, ceil(distance / (0.1 * params.origin.w))) + 1.0;
        var budget = samplesLeft;
        let probe = !refined && requiredSamples > f32(samplesLeft) && trial.error > 0.01 * params.origin.w;
        if (probe) { budget = min(budget, 8u); }
        let sweep = sweepParticle(position, trial.position, velocity, budget);
        samplesLeft -= sweep.samples;
        if (!sweep.complete) {
            if (!probe || samplesLeft == 0u) { break; }
            refined = true;
            h *= 0.5;
            continue;
        }
        position = sweep.correction.position;
        if (sweep.correction.contacted) {
            return Correction(position, velocity, true);
        }
        remaining -= h;
        if (remaining <= 0.0) {
            return Correction(position, velocity, false);
        }
        if (samplesLeft == 0u) { break; }
        let next = sampleVelocityWithSupport(position, false);
        if (!finiteAdvectionVector(next.xyz) || !(next.w >= 0.999)) {
            lastFailure = select(6u, 0u, !finiteAdvectionVector(next.xyz));
            break;
        }
        first = next.xyz;
        h = min(remaining, 2.0 * h);
    }
    atomicOr(&lists[statusIndex(lastFailure)], 1u);
    return Correction(original, velocity, false);
}
`;
