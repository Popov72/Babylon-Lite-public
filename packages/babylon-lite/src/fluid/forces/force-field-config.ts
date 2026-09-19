export type FluidForceFieldKind = "point" | "guide";
export type FluidForceFieldVector = readonly [number, number, number];

interface FluidForceFieldCommon {
    readonly id: string;
    readonly name: string;
    readonly enabled: boolean;
    /** Signed acceleration: negative attracts, positive repels. */
    readonly strength: number;
    readonly falloffPower: number;
    readonly useMinDistance: boolean;
    readonly minDistance: number;
    readonly useMaxDistance: boolean;
    readonly maxDistance: number;
    readonly maxForceLimitFactor: number;
}

export interface FluidPointForceField extends FluidForceFieldCommon {
    readonly type: "point";
    readonly position: FluidForceFieldVector;
}

export interface FluidGuideForceField extends FluidForceFieldCommon {
    readonly type: "guide";
    readonly start: FluidForceFieldVector;
    readonly end: FluidForceFieldVector;
    readonly flowStrength: number;
    readonly spinStrength: number;
    readonly endCaps: boolean;
}

export type FluidForceFieldDefinition = FluidPointForceField | FluidGuideForceField;

export function createDefaultFluidForceField(type: "point", id: string, position?: FluidForceFieldVector): FluidPointForceField;
export function createDefaultFluidForceField(type: "guide", id: string, position?: FluidForceFieldVector): FluidGuideForceField;
export function createDefaultFluidForceField(type: FluidForceFieldKind, id: string, position?: FluidForceFieldVector): FluidForceFieldDefinition;
export function createDefaultFluidForceField(type: FluidForceFieldKind, id: string, position: FluidForceFieldVector = [0, 1, 0]): FluidForceFieldDefinition {
    const common = {
        id,
        name: type === "point" ? "Point force" : "Guide force",
        enabled: true,
        strength: -9.81,
        falloffPower: 1,
        useMinDistance: true,
        minDistance: 0.5,
        useMaxDistance: true,
        maxDistance: 5,
        maxForceLimitFactor: 3,
    };
    return type === "point"
        ? { ...common, type, position: [...position] }
        : {
              ...common,
              type,
              start: [position[0], position[1] - 1, position[2]],
              end: [position[0], position[1] + 1, position[2]],
              flowStrength: 5,
              spinStrength: 0,
              endCaps: true,
          };
}

function number(value: unknown, label: string, minimum = -Infinity, maximum = Infinity): number {
    if (
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        !Number.isFinite(Math.fround(value)) ||
        (value !== 0 && Math.fround(value) === 0) ||
        value < minimum ||
        value > maximum
    ) {
        throw new RangeError(`[fluid] ${label} must be a finite float32 value in [${minimum}, ${maximum}].`);
    }
    return value;
}

function boolean(value: unknown, label: string): boolean {
    if (typeof value !== "boolean") {
        throw new TypeError(`[fluid] ${label} must be boolean.`);
    }
    return value;
}

function text(value: unknown, label: string): string {
    if (typeof value !== "string" || value.trim().length === 0) {
        throw new TypeError(`[fluid] ${label} must be a non-empty string.`);
    }
    return value;
}

function vector(value: unknown, label: string): [number, number, number] {
    if (!Array.isArray(value) || value.length !== 3) {
        throw new TypeError(`[fluid] ${label} must contain three coordinates.`);
    }
    return [number(value[0], `${label}.x`), number(value[1], `${label}.y`), number(value[2], `${label}.z`)];
}

/** Validate untrusted authoring data while retaining forward-compatible object metadata. */
export function validateFluidForceFields(value: unknown): FluidForceFieldDefinition[] {
    if (!Array.isArray(value)) {
        throw new TypeError("[fluid] forceFields must be an array.");
    }
    const ids = new Set<string>();
    let maximumAcceleration = 0;
    return value.map((entry: unknown, index) => {
        if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
            throw new TypeError(`[fluid] forceFields[${index}] must be an object.`);
        }
        const source = entry as Record<string, unknown>;
        const label = `forceFields[${index}]`;
        const id = text(source.id, `${label}.id`);
        if (ids.has(id)) {
            throw new Error(`[fluid] duplicate force-field id '${id}'.`);
        }
        ids.add(id);
        const common = {
            ...structuredClone(source),
            id,
            name: text(source.name, `${label}.name`),
            enabled: boolean(source.enabled, `${label}.enabled`),
            strength: number(source.strength, `${label}.strength`),
            falloffPower: number(source.falloffPower, `${label}.falloffPower`, 0, 16),
            useMinDistance: boolean(source.useMinDistance, `${label}.useMinDistance`),
            minDistance: number(source.minDistance, `${label}.minDistance`, 0),
            useMaxDistance: boolean(source.useMaxDistance, `${label}.useMaxDistance`),
            maxDistance: number(source.maxDistance, `${label}.maxDistance`, 0),
            maxForceLimitFactor: number(source.maxForceLimitFactor, `${label}.maxForceLimitFactor`, 0),
        };
        if (common.useMinDistance && common.useMaxDistance && common.minDistance > common.maxDistance) {
            throw new RangeError(`[fluid] ${label}.minDistance must not exceed maxDistance when both limits are enabled.`);
        }
        let field: FluidForceFieldDefinition;
        if (source.type === "point") {
            field = { ...common, type: "point", position: vector(source.position, `${label}.position`) };
        } else if (source.type === "guide") {
            const start = vector(source.start, `${label}.start`);
            const end = vector(source.end, `${label}.end`);
            const length = Math.hypot(...end.map((v, axis) => Math.fround(v) - Math.fround(start[axis]!)));
            if (!(length > 1e-6) || !Number.isFinite(Math.fround(length * length))) {
                throw new RangeError(`[fluid] ${label} requires distinct, float32-representable guide endpoints.`);
            }
            field = {
                ...common,
                type: "guide",
                start,
                end,
                flowStrength: number(source.flowStrength, `${label}.flowStrength`),
                spinStrength: number(source.spinStrength, `${label}.spinStrength`),
                endCaps: boolean(source.endCaps, `${label}.endCaps`),
            };
        } else {
            throw new TypeError(`[fluid] unsupported force-field type '${String(source.type)}'.`);
        }
        if (field.enabled) {
            maximumAcceleration +=
                (Math.abs(field.strength) + (field.type === "guide" ? Math.abs(field.flowStrength) + Math.abs(field.spinStrength) : 0)) * field.maxForceLimitFactor;
            if (!Number.isFinite(Math.fround(maximumAcceleration * maximumAcceleration))) {
                throw new RangeError("[fluid] combined force limits exceed the safe float32 magnitude range.");
            }
        }
        return field;
    });
}

/** Analytic acceleration in world units; definitions must have passed validation. */
export function evaluateFluidForceFields(fields: readonly FluidForceFieldDefinition[], point: FluidForceFieldVector): [number, number, number] {
    const acceleration: [number, number, number] = [0, 0, 0];
    for (const field of fields) {
        if (!field.enabled || field.maxForceLimitFactor === 0) {
            continue;
        }
        const origin = field.type === "point" ? field.position : field.start;
        let x = point[0] - origin[0];
        let y = point[1] - origin[1];
        let z = point[2] - origin[2];
        let axis: [number, number, number] = [0, 0, 0];
        if (field.type === "guide") {
            axis = [field.end[0] - field.start[0], field.end[1] - field.start[1], field.end[2] - field.start[2]];
            const length = Math.hypot(...axis);
            axis = [axis[0] / length, axis[1] / length, axis[2] / length];
            const along = x * axis[0] + y * axis[1] + z * axis[2];
            if (!field.endCaps && (along < 0 || along > length)) {
                continue;
            }
            const clamped = Math.min(length, Math.max(0, along));
            x -= clamped * axis[0];
            y -= clamped * axis[1];
            z -= clamped * axis[2];
        }
        const distance = Math.hypot(x, y, z);
        const radius = Math.max(distance, field.useMinDistance ? field.minDistance : 0);
        if (distance < 1e-6 || radius < 1e-6 || (field.useMaxDistance && radius > field.maxDistance)) {
            continue;
        }
        const radial: [number, number, number] = [x / distance, y / distance, z / distance];
        const add = (direction: FluidForceFieldVector, strength: number): void => {
            const length = Math.hypot(...direction);
            if (length < 1e-6 || strength === 0) {
                return;
            }
            const magnitude = Math.sign(strength) * Math.min((Math.abs(strength) * length) / radius ** field.falloffPower, Math.abs(strength) * field.maxForceLimitFactor);
            for (let axis = 0; axis < 3; axis++) {
                acceleration[axis]! += (direction[axis]! / length) * magnitude;
            }
        };
        add(radial, field.strength);
        if (field.type === "guide") {
            add(axis, field.flowStrength);
            add([axis[1] * radial[2] - axis[2] * radial[1], axis[2] * radial[0] - axis[0] * radial[2], axis[0] * radial[1] - axis[1] * radial[0]], field.spinStrength);
        }
    }
    return acceleration;
}
