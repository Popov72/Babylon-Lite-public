import type { EngineContext } from "../../engine/engine.js";
import { createFluidForceField, type FluidForceField } from "../core/fluid-facade.js";
import type { ForceFieldRuntimeBinding } from "../core/fluid-runtime-bindings.js";
import { validateFluidForceFields, type FluidForceFieldDefinition } from "./force-field-config.js";

const HEADER_FLOATS = 4;
const FIELD_FLOATS = 16;

interface ConfiguredForceBinding extends ForceFieldRuntimeBinding {
    readonly data: Float32Array;
    readonly capacity: number;
}

/** Maximum authored fields supported by this engine's uniform-buffer limits. */
export function fluidForceFieldCapacity(engine: EngineContext): number {
    const limits = engine._device.limits;
    return Math.max(0, Math.floor((Math.min(limits.maxUniformBufferBindingSize, limits.maxBufferSize) - HEADER_FLOATS * 4) / (FIELD_FLOATS * 4)));
}

const FORCE_WGSL = /* wgsl */ `
fn configuredForceVector(direction: vec3<f32>, strength: f32, radius: f32, power: f32, limitFactor: f32) -> vec3<f32> {
    let magnitude = abs(strength);
    let directionLength = length(direction);
    if (magnitude == 0.0 || limitFactor == 0.0 || directionLength < 1.0e-6) { return vec3<f32>(0.0); }
    var exponent = log2(magnitude) + log2(directionLength);
    if (power != 0.0) { exponent -= power * log2(radius); }
    exponent = min(exponent, log2(magnitude * limitFactor));
    if (exponent < -126.0) { return vec3<f32>(0.0); }
    return (direction / directionLength) * (sign(strength) * exp2(exponent));
}
fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> {
    var acceleration = vec3<f32>(0.0);
    for (var index = 0u; index < forceFieldParams.count.x; index++) {
        let field = forceFieldParams.fields[index];
        var delta = pos - field.originStrength.xyz;
        if (field.kind == 1u) {
            let along = dot(delta, field.axisLength.xyz);
            if ((field.flags & 4u) == 0u && (along < 0.0 || along > field.axisLength.w)) { continue; }
            delta -= clamp(along, 0.0, field.axisLength.w) * field.axisLength.xyz;
        }
        let distance = length(delta);
        var radius = distance;
        if ((field.flags & 1u) != 0u) { radius = max(radius, field.range.x); }
        if (distance < 1.0e-6 || radius < 1.0e-6 || ((field.flags & 2u) != 0u && radius > field.range.y)) { continue; }
        let radial = delta / distance;
        acceleration += configuredForceVector(radial, field.originStrength.w, radius, field.strengths.z, field.strengths.w);
        if (field.kind == 1u) {
            acceleration += configuredForceVector(field.axisLength.xyz, field.strengths.x, radius, field.strengths.z, field.strengths.w);
            acceleration += configuredForceVector(cross(field.axisLength.xyz, radial), field.strengths.y, radius, field.strengths.z, field.strengths.w);
        }
    }
    return acceleration * dt;
}`;

function packFields(data: Float32Array, fields: readonly FluidForceFieldDefinition[]): Float32Array {
    data.fill(0);
    const integers = new Uint32Array(data.buffer, data.byteOffset, data.length);
    let count = 0;
    for (const field of fields) {
        if (!field.enabled) {
            continue;
        }
        const offset = HEADER_FLOATS + count * FIELD_FLOATS;
        data.set(field.type === "point" ? field.position : field.start, offset);
        data[offset + 3] = field.strength;
        if (field.type === "guide") {
            const dx = Math.fround(field.end[0]) - Math.fround(field.start[0]);
            const dy = Math.fround(field.end[1]) - Math.fround(field.start[1]);
            const dz = Math.fround(field.end[2]) - Math.fround(field.start[2]);
            const length = Math.hypot(dx, dy, dz);
            data.set([dx / length, dy / length, dz / length, length, field.flowStrength, field.spinStrength], offset + 4);
        }
        data[offset + 10] = field.falloffPower;
        data[offset + 11] = field.maxForceLimitFactor;
        data[offset + 12] = field.minDistance;
        data[offset + 13] = field.maxDistance;
        integers[offset + 14] = (field.useMinDistance ? 1 : 0) | (field.useMaxDistance ? 2 : 0) | (field.type === "guide" && field.endCaps ? 4 : 0);
        integers[offset + 15] = field.type === "guide" ? 1 : 0;
        count++;
    }
    integers[0] = count;
    return data.subarray(0, HEADER_FLOATS + count * FIELD_FLOATS);
}

/** Create world-space authored forces, optionally following an existing caller-owned force. */
export function createFluidConfiguredForceField(engine: EngineContext, fields: readonly FluidForceFieldDefinition[], baseForce: FluidForceField | null = null): FluidForceField {
    if (baseForce?.disposed || (baseForce && baseForce._engine !== engine)) {
        throw new Error("[fluid] the base force must be live and belong to the same engine.");
    }
    const validated = validateFluidForceFields(fields);
    const capacity = fluidForceFieldCapacity(engine);
    if (capacity < 1 || validated.length > capacity) {
        throw new RangeError(`[fluid] this device supports at most ${capacity} configured force fields.`);
    }
    const data = new Float32Array(HEADER_FLOATS + capacity * FIELD_FLOATS);
    packFields(data, validated);
    const force = createFluidForceField(engine, {
        struct: `struct ConfiguredFluidForce {
    originStrength: vec4<f32>,
    axisLength: vec4<f32>,
    strengths: vec4<f32>,
    range: vec2<f32>,
    flags: u32,
    kind: u32,
}
struct ForceFieldParams {
    count: vec4<u32>,
    fields: array<ConfiguredFluidForce, ${capacity}>,
};`,
        wgsl: FORCE_WGSL,
        params: data,
    });
    const owned = force._binding as ForceFieldRuntimeBinding;
    const base = baseForce?._binding as ForceFieldRuntimeBinding | undefined;
    const binding: ConfiguredForceBinding = {
        spec: base ? { ...base.spec, additional: [...(base.spec.additional ?? []), owned.spec] } : owned.spec,
        capacity,
        data,
        updateParams: (params) => owned.updateParams(params),
        validateImplementation(method, backendId): void {
            base?.validateImplementation?.(method, backendId);
            if (backendId !== undefined) {
                throw new Error("[fluid] configured force fields currently support production fluid methods only.");
            }
        },
        releaseOwners(): void {
            baseForce?._owners.delete(force);
        },
        dispose: () => owned.dispose(),
    };
    force._kind = "configured";
    force._binding = binding;
    baseForce?._owners.add(force);
    return force;
}

/** Update configured fields without replacing the uniform allocation or shader. */
export function updateFluidConfiguredForceField(force: FluidForceField, fields: readonly FluidForceFieldDefinition[]): void {
    if (force.disposed || force._kind !== "configured") {
        throw new Error("[fluid] a live configured force field is required.");
    }
    const binding = force._binding as ConfiguredForceBinding;
    const validated = validateFluidForceFields(fields);
    if (validated.length > binding.capacity) {
        throw new RangeError(`[fluid] this device supports at most ${binding.capacity} configured force fields.`);
    }
    binding.updateParams(packFields(binding.data, validated));
}
