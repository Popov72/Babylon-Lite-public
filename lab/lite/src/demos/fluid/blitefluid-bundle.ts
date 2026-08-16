import type { FluidExportJson } from "./preset-io.js";
import {
    MAX_FLUID_EMITTERS,
    MAX_FLUID_POLYGON_POINTS,
    MAX_FLUID_POLYGON_TRIANGLES,
    MAX_FLUID_SINKS,
    type FluidEmitter,
    type FluidShape,
    type FluidSink,
    type FluidTransform,
} from "babylon-lite";
import { PHYS_MAX_SCALE, PHYS_MIN_SCALE, cellSizeForPhysicsScale, gridCellsForSize } from "./grid-settings.js";

const ZIP_LOCAL_FILE = 0x04034b50;
const ZIP_CENTRAL_FILE = 0x02014b50;
const ZIP_END = 0x06054b50;
const SDF_MAGIC = 0x46534c42;
const SDF_HEADER_BYTES = 64;
const MAX_ENTRY_BYTES = 512 * 1024 * 1024;
const MAX_SDF_VOXELS = 16 * 1024 * 1024;
const MAX_MANIFEST_BYTES = 1024 * 1024;
const MAX_PARTICLE_COUNT = 2_000_000;
const MAX_GRID_AXIS_CELLS = 2048;
const MAX_GRID_CELL_COUNT = 8 * 1024 * 1024;
const MAX_ABS_POSITION = 1_000_000;
const MAX_EXTENT = 10_000;
const MAX_VELOCITY = 100_000;
const MAX_RATE = 1_000_000_000_000;
const MAX_TEXT_LENGTH = 256;
const MAX_TARGETS = MAX_FLUID_EMITTERS;
const HEX_COLOR = /^#[0-9a-f]{6}$/i;
const METHODS = new Set(["PBF", "MLS-MPM", "PB-MPM"]);
const PHYSICS_LIMITS: Record<string, Record<string, readonly [number, number]>> = {
    PBF: {
        gravity: [0, 200],
        viscosity: [0, 3],
        relaxation: [1, 1000],
        scorr: [0, 0.5],
        iterations: [1, 8],
        restDensity: [100, 2000],
        boundaryDensity: [0, 1],
    },
    "MLS-MPM": {
        gravity: [0, 200],
        stiffness: [10, 5000],
        viscosity: [0, 1],
        restDensity: [1, 100],
        damping: [0.9, 1],
        affineDamping: [0.1, 1],
        groundDamp: [0.7, 1],
        groundDampHeight: [0, 10],
        restitution: [0, 1],
        substeps: [1, 8],
        maxSubDtMs: [2, 20],
    },
    "PB-MPM": {
        gravity: [0, 200],
        iterations: [1, 12],
        liquidRelaxation: [0.1, 3],
        liquidViscosity: [0, 0.2],
        elasticityRatio: [0, 1],
        elasticRelaxation: [0.05, 1],
        frictionAngle: [0, 60],
        plasticity: [0, 1],
        restitution: [0, 1],
        substeps: [1, 8],
        maxSubDtMs: [2, 20],
    },
};

export interface BliteFluidManifest {
    bundleVersion: 1;
    preset: FluidExportJson;
    scene: {
        glb: "scene.glb";
        collision: "collision.blsdf";
    };
}

export interface BliteFluidCollision {
    dims: [number, number, number];
    origin: [number, number, number];
    cellSize: number;
    distances: Float32Array;
}

export interface BliteFluidBundle {
    manifest: BliteFluidManifest;
    sceneGlb: ArrayBuffer;
    collision: BliteFluidCollision;
}

function fail(message: string): never {
    throw new Error(`Invalid .blitefluid bundle: ${message}`);
}

function crc32(bytes: Uint8Array): number {
    let crc = 0xffffffff;
    for (const byte of bytes) {
        crc ^= byte;
        for (let bit = 0; bit < 8; bit++) {
            crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
        }
    }
    return (crc ^ 0xffffffff) >>> 0;
}

function parseStoredZip(data: ArrayBuffer): Map<string, Uint8Array> {
    const bytes = new Uint8Array(data);
    const view = new DataView(data);
    const decoder = new TextDecoder();
    const entries = new Map<string, Uint8Array>();
    let offset = 0;

    while (offset + 4 <= bytes.byteLength) {
        const signature = view.getUint32(offset, true);
        if (signature === ZIP_CENTRAL_FILE || signature === ZIP_END) {
            break;
        }
        if (signature !== ZIP_LOCAL_FILE || offset + 30 > bytes.byteLength) {
            fail(`invalid ZIP record at byte ${offset}`);
        }

        const flags = view.getUint16(offset + 6, true);
        const compression = view.getUint16(offset + 8, true);
        const expectedCrc = view.getUint32(offset + 14, true);
        const compressedSize = view.getUint32(offset + 18, true);
        const uncompressedSize = view.getUint32(offset + 22, true);
        const nameLength = view.getUint16(offset + 26, true);
        const extraLength = view.getUint16(offset + 28, true);
        if ((flags & 1) !== 0) {
            fail("encrypted ZIP entries are not supported");
        }
        if ((flags & 8) !== 0) {
            fail("ZIP data descriptors are not supported");
        }
        if (compression !== 0) {
            fail("ZIP entries must use STORE compression");
        }
        if (compressedSize !== uncompressedSize) {
            fail("stored ZIP entry size mismatch");
        }
        if (uncompressedSize > MAX_ENTRY_BYTES) {
            fail("ZIP entry exceeds the 512 MiB limit");
        }

        const nameStart = offset + 30;
        const payloadStart = nameStart + nameLength + extraLength;
        const payloadEnd = payloadStart + uncompressedSize;
        if (payloadStart > bytes.byteLength || payloadEnd > bytes.byteLength) {
            fail("truncated ZIP entry");
        }

        const name = decoder.decode(bytes.subarray(nameStart, nameStart + nameLength));
        if (!name || name.includes("\\") || name.startsWith("/") || name.split("/").includes("..")) {
            fail(`unsafe ZIP entry name "${name}"`);
        }
        if (entries.has(name)) {
            fail(`duplicate ZIP entry "${name}"`);
        }

        const payload = bytes.subarray(payloadStart, payloadEnd);
        if (crc32(payload) !== expectedCrc) {
            fail(`CRC mismatch for "${name}"`);
        }
        entries.set(name, payload);
        offset = payloadEnd;
    }

    return entries;
}

function requiredEntry(entries: Map<string, Uint8Array>, name: string): Uint8Array {
    const entry = entries.get(name);
    if (!entry) {
        fail(`missing "${name}"`);
    }
    return entry;
}

function record(value: unknown, path: string): Record<string, unknown> {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        fail(`${path} must be an object`);
    }
    return value as Record<string, unknown>;
}

function array(value: unknown, path: string): unknown[] {
    if (!Array.isArray(value)) {
        fail(`${path} must be an array`);
    }
    return value;
}

function text(value: unknown, path: string, allowEmpty = false): string {
    if (typeof value !== "string" || (!allowEmpty && value.length === 0) || value.length > MAX_TEXT_LENGTH) {
        fail(`${path} must be ${allowEmpty ? "a" : "a non-empty"} string no longer than ${MAX_TEXT_LENGTH} characters`);
    }
    return value;
}

function bool(value: unknown, path: string): boolean {
    if (typeof value !== "boolean") {
        fail(`${path} must be a boolean`);
    }
    return value;
}

function finiteNumber(value: unknown, path: string, min: number, max: number): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
        fail(`${path} must be a finite number between ${min} and ${max}`);
    }
    return value;
}

function integer(value: unknown, path: string, min: number, max: number): number {
    const parsed = finiteNumber(value, path, min, max);
    if (!Number.isInteger(parsed)) {
        fail(`${path} must be an integer`);
    }
    return parsed;
}

function vector(value: unknown, path: string, length: number, min: number, max: number): number[] {
    const values = array(value, path);
    if (values.length !== length) {
        fail(`${path} must contain exactly ${length} numbers`);
    }
    return values.map((component, index) => finiteNumber(component, `${path}[${index}]`, min, max));
}

function numericRecord(value: unknown, path: string, maxEntries = 64): Record<string, number> {
    const values = record(value, path);
    const entries = Object.entries(values);
    if (entries.length > maxEntries) {
        fail(`${path} has too many fields`);
    }
    for (const [key, entry] of entries) {
        text(key, `${path} key`);
        finiteNumber(entry, `${path}.${key}`, -1_000_000, 1_000_000);
    }
    return values as Record<string, number>;
}

function validateTransform(value: unknown, path: string): FluidTransform {
    const transform = record(value, path);
    const position = vector(transform.position, `${path}.position`, 3, -MAX_ABS_POSITION, MAX_ABS_POSITION) as [number, number, number];
    const rotation = vector(transform.rotation, `${path}.rotation`, 4, -1, 1) as [number, number, number, number];
    if (Math.hypot(...rotation) < 1e-8) {
        fail(`${path}.rotation must be non-zero`);
    }
    const scale = vector(transform.scale, `${path}.scale`, 3, -MAX_EXTENT, MAX_EXTENT) as [number, number, number];
    if (scale.some((component) => Math.abs(component) < 1e-8)) {
        fail(`${path}.scale components must be non-zero`);
    }
    return { position, rotation, scale };
}

function validatePolygon(points: [number, number][], path: string): void {
    const pointKeys = new Set(points.map((point) => `${point[0]},${point[1]}`));
    if (pointKeys.size !== points.length) {
        fail(`${path} must not contain duplicate points`);
    }
    let area2 = 0;
    for (let index = 0; index < points.length; index++) {
        const point = points[index]!;
        const next = points[(index + 1) % points.length]!;
        area2 += point[0] * next[1] - next[0] * point[1];
    }
    if (Math.abs(area2) < 1e-8) {
        fail(`${path} must enclose a non-zero area`);
    }
    const orientation = (a: [number, number], b: [number, number], c: [number, number]): number => Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
    const intersects = (a: [number, number], b: [number, number], c: [number, number], d: [number, number]): boolean => {
        const abC = orientation(a, b, c);
        const abD = orientation(a, b, d);
        const cdA = orientation(c, d, a);
        const cdB = orientation(c, d, b);
        return abC !== abD && cdA !== cdB;
    };
    for (let first = 0; first < points.length; first++) {
        const firstNext = (first + 1) % points.length;
        for (let second = first + 1; second < points.length; second++) {
            const secondNext = (second + 1) % points.length;
            if (first === secondNext || firstNext === second) {
                continue;
            }
            if (intersects(points[first]!, points[firstNext]!, points[second]!, points[secondNext]!)) {
                fail(`${path} must define a simple, non-self-intersecting polygon`);
            }
        }
    }
}

function validateShape(value: unknown, path: string): FluidShape {
    const shape = record(value, path);
    const kind = text(shape.type, `${path}.type`);
    switch (kind) {
        case "box":
            return { type: kind, size: vector(shape.size, `${path}.size`, 3, Number.MIN_VALUE, MAX_EXTENT) as [number, number, number] };
        case "sphere":
            return { type: kind, radius: finiteNumber(shape.radius, `${path}.radius`, Number.MIN_VALUE, MAX_EXTENT) };
        case "cylinder": {
            const radius = finiteNumber(shape.radius, `${path}.radius`, Number.MIN_VALUE, MAX_EXTENT);
            const height = finiteNumber(shape.height, `${path}.height`, Number.MIN_VALUE, MAX_EXTENT);
            if (shape.innerRadius === undefined) {
                return { type: kind, radius, height };
            }
            const innerRadius = finiteNumber(shape.innerRadius, `${path}.innerRadius`, 0, radius);
            if (innerRadius >= radius) {
                fail(`${path}.innerRadius must be smaller than radius`);
            }
            return { type: kind, radius, height, innerRadius };
        }
        case "cone":
            return {
                type: kind,
                bottomRadius: finiteNumber(shape.bottomRadius, `${path}.bottomRadius`, Number.MIN_VALUE, MAX_EXTENT),
                topRadius: finiteNumber(shape.topRadius, `${path}.topRadius`, 0, MAX_EXTENT),
                height: finiteNumber(shape.height, `${path}.height`, Number.MIN_VALUE, MAX_EXTENT),
            };
        case "capsule":
            return {
                type: kind,
                radius: finiteNumber(shape.radius, `${path}.radius`, Number.MIN_VALUE, MAX_EXTENT),
                height: finiteNumber(shape.height, `${path}.height`, Number.MIN_VALUE, MAX_EXTENT),
            };
        case "polygonPrism": {
            const points = array(shape.points, `${path}.points`);
            if (points.length < 3 || points.length > MAX_FLUID_POLYGON_POINTS) {
                fail(`${path}.points must contain between 3 and ${MAX_FLUID_POLYGON_POINTS} points`);
            }
            const validatedPoints = points.map((point, index) => vector(point, `${path}.points[${index}]`, 2, -MAX_EXTENT, MAX_EXTENT) as [number, number]);
            validatePolygon(validatedPoints, `${path}.points`);
            return {
                type: kind,
                points: validatedPoints,
                thickness: finiteNumber(shape.thickness, `${path}.thickness`, Number.MIN_VALUE, MAX_EXTENT),
            };
        }
        default:
            fail(`${path}.type "${kind}" is not supported`);
    }
}

function validateEmitter(value: unknown, path: string): FluidEmitter {
    const emitter = record(value, path);
    const behavior = text(emitter.behavior, `${path}.behavior`);
    if (behavior !== "initial" && behavior !== "inflow") {
        fail(`${path}.behavior must be "initial" or "inflow"`);
    }
    const sampling = text(emitter.sampling, `${path}.sampling`);
    if (sampling !== "volume" && sampling !== "surface") {
        fail(`${path}.sampling must be "volume" or "surface"`);
    }
    const velocitySpace = text(emitter.velocitySpace, `${path}.velocitySpace`);
    if (velocitySpace !== "local" && velocitySpace !== "world") {
        fail(`${path}.velocitySpace must be "local" or "world"`);
    }
    const result: FluidEmitter = {
        id: text(emitter.id, `${path}.id`),
        name: text(emitter.name, `${path}.name`, true),
        enabled: bool(emitter.enabled, `${path}.enabled`),
        behavior,
        transform: validateTransform(emitter.transform, `${path}.transform`),
        shape: validateShape(emitter.shape, `${path}.shape`),
        sampling,
        velocity: vector(emitter.velocity, `${path}.velocity`, 3, -MAX_VELOCITY, MAX_VELOCITY) as [number, number, number],
        velocitySpace,
        spread: finiteNumber(emitter.spread, `${path}.spread`, 0, MAX_VELOCITY),
    };
    if (emitter.volumeRate !== undefined) {
        result.volumeRate = finiteNumber(emitter.volumeRate, `${path}.volumeRate`, Number.MIN_VALUE, MAX_RATE);
    }
    return result;
}

function validateSink(value: unknown, path: string): FluidSink {
    const sink = record(value, path);
    const targets = array(sink.targets, `${path}.targets`);
    if (targets.length > MAX_TARGETS) {
        fail(`${path}.targets supports at most ${MAX_TARGETS} entries`);
    }
    const result: FluidSink = {
        id: text(sink.id, `${path}.id`),
        name: text(sink.name, `${path}.name`, true),
        enabled: bool(sink.enabled, `${path}.enabled`),
        transform: validateTransform(sink.transform, `${path}.transform`),
        shape: validateShape(sink.shape, `${path}.shape`),
        targets: targets.map((target, index) => text(target, `${path}.targets[${index}]`)),
    };
    if (new Set(result.targets).size !== result.targets.length) {
        fail(`${path}.targets must not contain duplicates`);
    }
    if (sink.volumeRate !== undefined && sink.perParticleRecycleRate !== undefined) {
        fail(`${path} cannot define both volumeRate and perParticleRecycleRate`);
    }
    if (sink.volumeRate !== undefined) {
        result.volumeRate = finiteNumber(sink.volumeRate, `${path}.volumeRate`, Number.MIN_VALUE, MAX_RATE);
    }
    if (sink.perParticleRecycleRate !== undefined) {
        result.perParticleRecycleRate = finiteNumber(sink.perParticleRecycleRate, `${path}.perParticleRecycleRate`, 0, MAX_RATE);
    }
    return result;
}

function validatePhysics(value: unknown, method: string): void {
    const physics = numericRecord(value, "manifest.preset.physics");
    const limits = PHYSICS_LIMITS[method]!;
    const allowed = new Set(Object.keys(limits));
    for (const key of Object.keys(physics)) {
        if (!allowed.has(key)) {
            fail(`manifest.preset.physics.${key} is not valid for ${method}`);
        }
    }
    for (const [key, [min, max]] of Object.entries(limits)) {
        finiteNumber(physics[key], `manifest.preset.physics.${key}`, min, max);
    }
}

function validateDemoState(value: unknown): void {
    const state = record(value, "manifest.preset.demoState");
    if (Object.keys(state).length > 64) {
        fail("manifest.preset.demoState has too many fields");
    }
    for (const [key, entry] of Object.entries(state)) {
        text(key, "manifest.preset.demoState key");
        if (typeof entry === "number") {
            finiteNumber(entry, `manifest.preset.demoState.${key}`, -1_000_000, 1_000_000);
        } else if (typeof entry === "string") {
            text(entry, `manifest.preset.demoState.${key}`, true);
        } else if (typeof entry !== "boolean") {
            fail(`manifest.preset.demoState.${key} must be a finite number, boolean, or string`);
        }
    }
}

function validateRender(value: unknown): void {
    const render = record(value, "manifest.preset.render");
    bool(render.renderAsSpheres, "manifest.preset.render.renderAsSpheres");
    const waterColor = text(render.waterColor, "manifest.preset.render.waterColor");
    if (!HEX_COLOR.test(waterColor)) {
        fail("manifest.preset.render.waterColor must be a #RRGGBB color");
    }
    for (const key of [
        "absorption",
        "particleSize",
        "refractionStrength",
        "specularPower",
        "surfaceDepthBlur",
        "depthBlurEdgeThreshold",
        "surfaceThicknessBlur",
        "narrowRangeDelta",
        "narrowRangeMu",
    ]) {
        finiteNumber(render[key], `manifest.preset.render.${key}`, 0, 1_000_000);
    }
    finiteNumber(render.thicknessDownscale, "manifest.preset.render.thicknessDownscale", 1, 16);
    bool(render.halfRendering, "manifest.preset.render.halfRendering");
    bool(render.anisotropicSurface, "manifest.preset.render.anisotropicSurface");
    const filter = text(render.surfaceFilter, "manifest.preset.render.surfaceFilter");
    if (filter !== "bilateral" && filter !== "narrowRange") {
        fail('manifest.preset.render.surfaceFilter must be "bilateral" or "narrowRange"');
    }
    for (const key of ["reflectionExposure", "reflectionContrast"]) {
        if (render[key] !== undefined) {
            finiteNumber(render[key], `manifest.preset.render.${key}`, 0, 100);
        }
    }
    if (render.waterReflectivity !== undefined) {
        finiteNumber(render.waterReflectivity, "manifest.preset.render.waterReflectivity", 0, 1);
    }
    if (render.anisoRadiusDamping !== undefined) {
        finiteNumber(render.anisoRadiusDamping, "manifest.preset.render.anisoRadiusDamping", 0, 1);
    }
}

function validateFoam(value: unknown): void {
    const foam = record(value, "manifest.preset.foam");
    bool(foam.enableFoam, "manifest.preset.foam.enableFoam");
    if (foam.activeParticles !== undefined) {
        bool(foam.activeParticles, "manifest.preset.foam.activeParticles");
    }
    for (const key of [
        "trappedAirRate",
        "waveCrestRate",
        "foamLifetime",
        "foamLifetimeMin",
        "bubbleBuoyancy",
        "bubbleDrag",
        "poolSize",
        "foamSoftness",
        "foamDensity",
        "subsurfaceBubbleStrength",
        "foamBlurRadius",
        "foamLightIntensity",
        "foamAmbient",
        "foamAO",
        "foamNormalStrength",
        "foamSize",
    ]) {
        finiteNumber(foam[key], `manifest.preset.foam.${key}`, 0, 1_000_000);
    }
    if ((foam.foamLifetimeMin as number) > (foam.foamLifetime as number)) {
        fail("manifest.preset.foam.foamLifetimeMin must not exceed foamLifetime");
    }
    const bubbleColor = text(foam.subsurfaceBubbleColor, "manifest.preset.foam.subsurfaceBubbleColor");
    if (!HEX_COLOR.test(bubbleColor)) {
        fail("manifest.preset.foam.subsurfaceBubbleColor must be a #RRGGBB color");
    }
    text(foam.foamDebug, "manifest.preset.foam.foamDebug");
}

function validatePreset(value: unknown): FluidExportJson {
    const preset = record(value, "manifest.preset");
    if (preset.formatVersion !== 5) {
        fail("manifest preset must use formatVersion 5");
    }
    const meta = record(preset.meta, "manifest.preset.meta");
    if (meta.demo !== "blender") {
        fail('manifest.preset.meta.demo must be "blender"');
    }
    const method = text(meta.method, "manifest.preset.meta.method");
    if (!METHODS.has(method)) {
        fail(`manifest.preset.meta.method "${method}" is not supported`);
    }
    validatePhysics(preset.physics, method);
    numericRecord(preset.demoParams, "manifest.preset.demoParams");
    validateDemoState(preset.demoState);
    const particleSize = finiteNumber(preset.physicsParticleSize, "manifest.preset.physicsParticleSize", PHYS_MIN_SCALE, PHYS_MAX_SCALE);
    integer(preset.particleCount, "manifest.preset.particleCount", 1, MAX_PARTICLE_COUNT);
    vector(preset.gridPosition, "manifest.preset.gridPosition", 3, -MAX_ABS_POSITION, MAX_ABS_POSITION);
    const gridSize = vector(preset.gridSize, "manifest.preset.gridSize", 3, Number.MIN_VALUE, MAX_EXTENT) as [number, number, number];
    const cells = gridCellsForSize(gridSize, cellSizeForPhysicsScale(method, particleSize));
    if (cells.some((count) => count > MAX_GRID_AXIS_CELLS) || cells[0] * cells[1] * cells[2] > MAX_GRID_CELL_COUNT) {
        fail("manifest preset grid exceeds the supported allocation limits");
    }
    const emitterValues = array(preset.emitters, "manifest.preset.emitters");
    const sinkValues = array(preset.sinks, "manifest.preset.sinks");
    if (emitterValues.length > MAX_FLUID_EMITTERS) {
        fail(`manifest.preset.emitters supports at most ${MAX_FLUID_EMITTERS} entries`);
    }
    if (sinkValues.length > MAX_FLUID_SINKS) {
        fail(`manifest.preset.sinks supports at most ${MAX_FLUID_SINKS} entries`);
    }
    const emitters = emitterValues.map((emitter, index) => validateEmitter(emitter, `manifest.preset.emitters[${index}]`));
    const sinks = sinkValues.map((sink, index) => validateSink(sink, `manifest.preset.sinks[${index}]`));
    const ids = new Set<string>();
    for (const object of [...emitters, ...sinks]) {
        if (ids.has(object.id)) {
            fail(`manifest preset flow ID "${object.id}" is duplicated`);
        }
        ids.add(object.id);
    }
    const emitterIds = new Set(emitters.filter((emitter) => emitter.behavior === "inflow").map((emitter) => emitter.id));
    for (const [index, sink] of sinks.entries()) {
        for (const target of sink.targets) {
            if (!emitterIds.has(target)) {
                fail(`manifest.preset.sinks[${index}].targets references unknown or non-inflow emitter "${target}"`);
            }
        }
    }
    const polygons = [...emitters, ...sinks].map((object) => object.shape).filter((shape): shape is Extract<FluidShape, { type: "polygonPrism" }> => shape.type === "polygonPrism");
    const pointCount = polygons.reduce((total, shape) => total + shape.points.length, 0);
    const triangleCount = polygons.reduce((total, shape) => total + shape.points.length - 2, 0);
    if (pointCount > MAX_FLUID_POLYGON_POINTS || triangleCount > MAX_FLUID_POLYGON_TRIANGLES) {
        fail("manifest preset polygon shapes exceed the shared flow-buffer capacity");
    }

    for (const key of ["showContainer", "msaa", "activeBlocks", "pagedGrid", "fusedBlockDiscovery", "showGridBounds", "initialEmittersFillCapacity"]) {
        if (preset[key] !== undefined) {
            bool(preset[key], `manifest.preset.${key}`);
        }
    }
    if (preset.envIntensity !== undefined) {
        finiteNumber(preset.envIntensity, "manifest.preset.envIntensity", 0, 100);
    }
    if (preset.pagedGridMaxPages !== undefined) {
        integer(preset.pagedGridMaxPages, "manifest.preset.pagedGridMaxPages", 1, 1_000_000);
    }
    if (preset.material !== undefined) {
        integer(preset.material, "manifest.preset.material", 0, 3);
    }
    if (preset.camera !== undefined) {
        const camera = record(preset.camera, "manifest.preset.camera");
        finiteNumber(camera.alpha, "manifest.preset.camera.alpha", -1_000_000, 1_000_000);
        finiteNumber(camera.beta, "manifest.preset.camera.beta", -1_000_000, 1_000_000);
        finiteNumber(camera.radius, "manifest.preset.camera.radius", Number.MIN_VALUE, MAX_EXTENT);
    }
    validateRender(preset.render);
    validateFoam(preset.foam);
    return preset as unknown as FluidExportJson;
}

function parseManifest(bytes: Uint8Array): BliteFluidManifest {
    if (bytes.byteLength > MAX_MANIFEST_BYTES) {
        fail("manifest.json exceeds the 1 MiB limit");
    }
    let value: unknown;
    try {
        value = JSON.parse(new TextDecoder().decode(bytes));
    } catch {
        fail("manifest.json is not valid JSON");
    }
    const manifest = record(value, "manifest.json");
    if (manifest.bundleVersion !== 1) {
        fail("unsupported bundle version");
    }
    const preset = validatePreset(manifest.preset);
    const scene = record(manifest.scene, "manifest.scene");
    if (scene.glb !== "scene.glb" || scene.collision !== "collision.blsdf") {
        fail("manifest scene paths are invalid");
    }
    return { bundleVersion: 1, preset, scene: { glb: "scene.glb", collision: "collision.blsdf" } };
}

function parseGlb(bytes: Uint8Array): ArrayBuffer {
    if (bytes.byteLength < 12) {
        fail("scene.glb is truncated");
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== 0x46546c67) {
        fail("scene.glb has invalid magic");
    }
    if (view.getUint32(4, true) !== 2) {
        fail("scene.glb must use glTF 2");
    }
    if (view.getUint32(8, true) !== bytes.byteLength) {
        fail("scene.glb length does not match its header");
    }
    return bytes.slice().buffer;
}

export function parseBliteFluidCollision(bytes: Uint8Array): BliteFluidCollision {
    if (bytes.byteLength < SDF_HEADER_BYTES) {
        fail("collision.blsdf is truncated");
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== SDF_MAGIC) {
        fail("collision.blsdf has invalid magic");
    }
    if (view.getUint32(4, true) !== 1) {
        fail("unsupported collision version");
    }

    const dims: [number, number, number] = [view.getUint32(8, true), view.getUint32(12, true), view.getUint32(16, true)];
    if (dims.some((dim) => dim < 2 || dim > 2048)) {
        fail("collision dimensions must be between 2 and 2048");
    }
    const voxelCount = dims[0] * dims[1] * dims[2];
    if (!Number.isSafeInteger(voxelCount) || voxelCount > MAX_SDF_VOXELS) {
        fail("collision grid exceeds the 16M-voxel limit");
    }
    if (bytes.byteLength !== SDF_HEADER_BYTES + voxelCount * 4) {
        fail("collision payload length is invalid");
    }

    const origin: [number, number, number] = [view.getFloat32(24, true), view.getFloat32(28, true), view.getFloat32(32, true)];
    const cellSize = view.getFloat32(36, true);
    if (
        !origin.every((component) => Number.isFinite(component) && Math.abs(component) <= MAX_ABS_POSITION) ||
        !Number.isFinite(cellSize) ||
        cellSize <= 0 ||
        cellSize > MAX_EXTENT
    ) {
        fail("collision transform is invalid");
    }

    const distances = new Float32Array(voxelCount);
    for (let index = 0; index < voxelCount; index++) {
        const distance = view.getFloat32(SDF_HEADER_BYTES + index * 4, true);
        if (!Number.isFinite(distance)) {
            fail(`collision distance ${index} is not finite`);
        }
        distances[index] = distance;
    }
    return { dims, origin, cellSize, distances };
}

export function parseBliteFluidBundle(data: ArrayBuffer): BliteFluidBundle {
    const entries = parseStoredZip(data);
    const expectedEntries = new Set(["manifest.json", "scene.glb", "collision.blsdf"]);
    if (entries.size !== expectedEntries.size || [...entries.keys()].some((name) => !expectedEntries.has(name))) {
        fail("archive must contain exactly manifest.json, scene.glb, and collision.blsdf");
    }
    const manifest = parseManifest(requiredEntry(entries, "manifest.json"));
    const sceneGlb = parseGlb(requiredEntry(entries, manifest.scene.glb));
    const collision = parseBliteFluidCollision(requiredEntry(entries, manifest.scene.collision));
    return { manifest, sceneGlb, collision };
}
