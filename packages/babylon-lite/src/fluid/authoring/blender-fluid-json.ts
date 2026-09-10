import type { FluidExportJson } from "./preset-io.js";
import { Unzlib, zlibSync } from "fflate";
import {
    MAX_FLUID_EMITTERS,
    MAX_FLUID_POLYGON_POINTS,
    MAX_FLUID_POLYGON_TRIANGLES,
    MAX_FLUID_SINKS,
    type FluidEmitter,
    type FluidShape,
    type FluidSink,
    type FluidTransform,
} from "../core/sim-common.js";
import { fluidSimulationCellSize } from "../core/simulation-config.js";
import { PHYS_MAX_SCALE, PHYS_MIN_SCALE, gridCellsForSize } from "./grid-settings.js";

const SDF_MAGIC = 0x46534c42;
const SDF_HEADER_BYTES = 64;
const MAX_ENTRY_BYTES = 512 * 1024 * 1024;
const MAX_SDF_VOXELS = 16 * 1024 * 1024;
const MAX_JSON_BYTES = 768 * 1024 * 1024;
const MAX_GRID_AXIS_CELLS = 2048;
const MAX_GRID_CELL_COUNT = 8 * 1024 * 1024;
const MAX_ABS_POSITION = 1_000_000;
const MAX_EXTENT = 10_000;
const MAX_VELOCITY = 100_000;
const MAX_RATE = 1_000_000_000_000;
const MAX_DELAY = 86_400;
const MAX_TEXT_LENGTH = 256;
const MAX_TARGETS = MAX_FLUID_EMITTERS;
const MAX_ANIMATED_COLLISIONS = 16;
const HEX_COLOR = /^#[0-9a-f]{6}$/i;
const METHODS = ["PBF", "FLIP", "MLS-MPM", "PB-MPM"] as const;
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
    FLIP: {
        gravity: [0, 200],
        flipRatio: [0, 1],
        kinematicViscosity: [0, 5],
        surfaceTension: [0, 5],
        minSubsteps: [1, 16],
        maxSubsteps: [1, 32],
        cflNumber: [0, 10],
        restitution: [0, 1],
        velocityDamping: [0, 10],
        pressureSolver: [0, 1],
        pressureIterations: [1, 100],
        pressureRelaxation: [0.1, 1],
        multigridCycles: [1, 8],
        pressureTolerance: [0, 0.1],
        pressureDiagnostics: [0, 1],
        liquidSdf: [0, 1],
        ghostFluid: [0, 1],
        fractionalSolids: [0, 1],
        movingSolidBoundaries: [0, 1],
        reseedParticles: [0, 1],
        reseedMinParticles: [1, 64],
        reseedTargetParticles: [1, 64],
        reseedMaxParticles: [1, 96],
        reseedInterval: [1, 30],
        particleSheeting: [0, 1],
        sheetingStrength: [0.05, 1],
        sheetingInterval: [1, 30],
        polygonSurface: [0, 1],
        polygonReconstructionMultiplier: [1, 2],
        viscosityIterations: [1, 40],
        maxSubDtMs: [1, 20],
    },
    "MLS-MPM": {
        gravity: [0, 200],
        stiffness: [10, 5000],
        viscosity: [0, 1],
        restDensity: [1, 100],
        damping: [0.9, 1],
        affineDamping: [0.1, 1],
        groundDamp: [0, 1],
        groundDampHeight: [0, 10],
        restitution: [0, 1],
        substeps: [1, 8],
        maxSubDtMs: [2, 20],
    },
    "PB-MPM": {
        gravity: [0, 200],
        iterations: [1, 12],
        liquidRelaxation: [0.1, 3],
        liquidViscosity: [0, 1],
        elasticityRatio: [0, 1],
        elasticRelaxation: [0.05, 1],
        frictionAngle: [0, 60],
        plasticity: [0, 1],
        restitution: [0, 1],
        substeps: [1, 8],
        maxSubDtMs: [2, 20],
    },
};
const OPTIONAL_PHYSICS_KEYS: Record<string, readonly string[]> = {
    FLIP: [
        "pressureSolver",
        "multigridCycles",
        "pressureTolerance",
        "pressureDiagnostics",
        "liquidSdf",
        "ghostFluid",
        "fractionalSolids",
        "movingSolidBoundaries",
        "reseedParticles",
        "reseedMinParticles",
        "reseedTargetParticles",
        "reseedMaxParticles",
        "reseedInterval",
        "particleSheeting",
        "sheetingStrength",
        "sheetingInterval",
        "polygonSurface",
        "polygonReconstructionMultiplier",
    ],
};

export interface BlenderFluidScene {
    preset: FluidExportJson;
    sceneGlb: ArrayBuffer;
    collision: BlenderFluidCollision;
    collisionEnabled: boolean;
    collisionTrilinear: boolean;
    animatedCollisions: BlenderFluidAnimatedCollision[];
}

export type BlenderFluidExternalResources = ReadonlyMap<string, ArrayBufferLike>;

export interface BlenderFluidCollision {
    dims: [number, number, number];
    origin: [number, number, number];
    cellSize: number;
    distances: Float32Array;
}

export interface BlenderFluidAnimatedCollision {
    id: string;
    node: string;
    space: "node-local";
    resolution: number;
    bakeFrame: number;
    presentation: boolean;
    enabled: boolean;
    trilinear: boolean;
    collision: BlenderFluidCollision;
}

export interface BlenderFluidScenePayloadOptions {
    /** Preserve external filenames instead of embedding the already-resolved bytes. Default false. */
    preserveExternal?: boolean;
}

export function scenePayloadFromBlenderFluidJson(scene: BlenderFluidScene, options: BlenderFluidScenePayloadOptions = {}): NonNullable<FluidExportJson["scene"]> {
    const payload = scene.preset.scene;
    if (!payload) {
        fail("self-contained fluid JSON is missing scene data");
    }
    if (payload.encoding === "external") {
        if (options.preserveExternal) {
            return {
                encoding: "external",
                glb: payload.glb,
                collision: payload.collision,
                ...(payload.sdfCompression ? { sdfCompression: payload.sdfCompression } : {}),
                ...(payload.collisionEnabled !== undefined ? { collisionEnabled: payload.collisionEnabled } : {}),
                ...(payload.collisionTrilinear !== undefined ? { collisionTrilinear: payload.collisionTrilinear } : {}),
                ...(payload.collisionByteLength !== undefined ? { collisionByteLength: payload.collisionByteLength } : {}),
                ...(payload.anchorPosition ? { anchorPosition: [...payload.anchorPosition] as [number, number, number] } : {}),
                ...(payload.animatedCollisions
                    ? {
                          animatedCollisions: payload.animatedCollisions.map((entry) => ({
                              id: entry.id,
                              node: entry.node,
                              sdf: entry.sdf,
                              ...(entry.byteOffset !== undefined ? { byteOffset: entry.byteOffset } : {}),
                              ...(entry.byteLength !== undefined ? { byteLength: entry.byteLength } : {}),
                              space: entry.space,
                              resolution: entry.resolution,
                              bakeFrame: entry.bakeFrame,
                              presentation: entry.presentation,
                              ...(entry.enabled !== undefined ? { enabled: entry.enabled } : {}),
                              ...(entry.trilinear !== undefined ? { trilinear: entry.trilinear } : {}),
                          })),
                      }
                    : {}),
            };
        }
        return {
            encoding: "base64",
            glb: encodeBase64(new Uint8Array(scene.sceneGlb)),
            collision: encodeBase64(compressSdfBytes(encodeBlenderFluidCollision(scene.collision))),
            sdfCompression: "zlib",
            collisionEnabled: scene.collisionEnabled,
            collisionTrilinear: scene.collisionTrilinear,
            ...(payload.anchorPosition ? { anchorPosition: [...payload.anchorPosition] as [number, number, number] } : {}),
            ...(scene.animatedCollisions.length
                ? {
                      animatedCollisions: scene.animatedCollisions.map((entry) => ({
                          id: entry.id,
                          node: entry.node,
                          sdf: encodeBase64(compressSdfBytes(encodeBlenderFluidCollision(entry.collision))),
                          space: entry.space,
                          resolution: entry.resolution,
                          bakeFrame: entry.bakeFrame,
                          presentation: entry.presentation,
                          enabled: entry.enabled,
                          trilinear: entry.trilinear,
                      })),
                  }
                : {}),
        };
    }
    return {
        encoding: "base64",
        glb: payload.glb,
        collision: payload.collision,
        ...(payload.sdfCompression ? { sdfCompression: payload.sdfCompression } : {}),
        ...(payload.collisionEnabled !== undefined ? { collisionEnabled: payload.collisionEnabled } : {}),
        ...(payload.collisionTrilinear !== undefined ? { collisionTrilinear: payload.collisionTrilinear } : {}),
        ...(payload.anchorPosition ? { anchorPosition: [...payload.anchorPosition] as [number, number, number] } : {}),
        ...(payload.animatedCollisions
            ? {
                  animatedCollisions: payload.animatedCollisions.map((entry) => ({ ...entry })),
              }
            : {}),
    };
}

function fail(message: string): never {
    throw new Error(`Invalid fluid export: ${message}`);
}

function encodeBase64(bytes: Uint8Array): string {
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(offset, Math.min(bytes.length, offset + 0x8000)));
    }
    return btoa(binary);
}

function encodeBlenderFluidCollision(collision: BlenderFluidCollision): Uint8Array {
    const bytes = new Uint8Array(SDF_HEADER_BYTES + collision.distances.length * 4);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, SDF_MAGIC, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, collision.dims[0], true);
    view.setUint32(12, collision.dims[1], true);
    view.setUint32(16, collision.dims[2], true);
    view.setFloat32(24, collision.origin[0], true);
    view.setFloat32(28, collision.origin[1], true);
    view.setFloat32(32, collision.origin[2], true);
    view.setFloat32(36, collision.cellSize, true);
    for (let index = 0; index < collision.distances.length; index++) {
        view.setFloat32(SDF_HEADER_BYTES + index * 4, collision.distances[index]!, true);
    }
    return bytes;
}

function compressSdfBytes(bytes: Uint8Array): Uint8Array {
    return zlibSync(bytes, { level: 6 });
}

function decompressSdfBytes(bytes: Uint8Array, path: string, compressed: boolean): Uint8Array {
    if (!compressed) {
        return bytes;
    }
    const chunks: Uint8Array[] = [];
    let total = 0;
    let complete = false;
    let limitExceeded = false;
    try {
        const decoder = new Unzlib((chunk, final) => {
            total += chunk.byteLength;
            if (total > MAX_ENTRY_BYTES) {
                limitExceeded = true;
                throw new RangeError("SDF decompression limit exceeded");
            }
            chunks.push(chunk);
            complete = final;
        });
        decoder.push(bytes, true);
    } catch {
        if (limitExceeded) {
            fail(`${path} decompressed data exceeds the 512 MiB limit`);
        }
        fail(`${path} is not valid zlib data`);
    }
    if (!complete) {
        fail(`${path} zlib stream is truncated`);
    }
    const output = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
        output.set(chunk, offset);
        offset += chunk.byteLength;
    }
    return output;
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
    if (Math.sqrt(rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2] + rotation[3] * rotation[3]) < 1e-8) {
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
    if (emitter.sourceNode !== undefined) {
        result.sourceNode = text(emitter.sourceNode, `${path}.sourceNode`);
    }
    if (emitter.sourcePresentation !== undefined) {
        if (typeof emitter.sourcePresentation !== "boolean") {
            fail(`${path}.sourcePresentation must be a boolean`);
        }
        result.sourcePresentation = emitter.sourcePresentation;
    }
    if (emitter.sourceVelocity !== undefined) {
        result.sourceVelocity = vector(emitter.sourceVelocity, `${path}.sourceVelocity`, 3, -MAX_VELOCITY, MAX_VELOCITY) as [number, number, number];
    }
    if (emitter.sourceVelocityFactor !== undefined) {
        result.sourceVelocityFactor = finiteNumber(emitter.sourceVelocityFactor, `${path}.sourceVelocityFactor`, -MAX_VELOCITY, MAX_VELOCITY);
    }
    if (emitter.normalVelocity !== undefined) {
        result.normalVelocity = finiteNumber(emitter.normalVelocity, `${path}.normalVelocity`, -MAX_VELOCITY, MAX_VELOCITY);
    }
    if (emitter.volumeRate !== undefined) {
        result.volumeRate = finiteNumber(emitter.volumeRate, `${path}.volumeRate`, Number.MIN_VALUE, MAX_RATE);
    }
    if (emitter.delayBeforeStart !== undefined) {
        result.delayBeforeStart = finiteNumber(emitter.delayBeforeStart, `${path}.delayBeforeStart`, 0, MAX_DELAY);
    }
    return result;
}

function validateSink(value: unknown, path: string, formatVersion: number): FluidSink {
    const sink = record(value, path);
    const mode = sink.mode === undefined && formatVersion <= 6 ? "recycle" : sink.mode;
    if (mode !== "delete" && mode !== "recycle") {
        fail(`${path}.mode must be "delete" or "recycle"`);
    }
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
        mode,
        targets: targets.map((target, index) => text(target, `${path}.targets[${index}]`)),
    };
    if (new Set(result.targets).size !== result.targets.length) {
        fail(`${path}.targets must not contain duplicates`);
    }
    if (sink.delayBeforeStart !== undefined) {
        result.delayBeforeStart = finiteNumber(sink.delayBeforeStart, `${path}.delayBeforeStart`, 0, MAX_DELAY);
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
        if (physics[key] === undefined && OPTIONAL_PHYSICS_KEYS[method]?.includes(key)) {
            continue;
        }
        finiteNumber(physics[key], `manifest.preset.physics.${key}`, min, max);
    }
}

function validateDemoState(value: unknown): Record<string, unknown> {
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
    return state;
}

function validateRender(value: unknown): void {
    const render = record(value, "manifest.preset.render");
    bool(render.renderAsSpheres, "manifest.preset.render.renderAsSpheres");
    if (render.polygonShader !== undefined && render.polygonShader !== "physical" && render.polygonShader !== "ocean") {
        fail('manifest.preset.render.polygonShader must be "physical" or "ocean"');
    }
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
    if (render.independentRendering !== undefined) {
        bool(render.independentRendering, "manifest.preset.render.independentRendering");
    }
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
    for (const key of ["generateSpray", "generateFoam", "generateBubbles"]) {
        if (foam[key] !== undefined) {
            bool(foam[key], `manifest.preset.foam.${key}`);
        }
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
    for (const [key, max] of [
        ["turbulenceRate", 500],
        ["energySpeedMin", 20],
        ["energySpeedMax", 40],
        ["curvatureMin", 4],
        ["curvatureMax", 8],
        ["turbulenceMin", 20],
        ["turbulenceMax", 40],
        ["foamLayerDepth", 4],
        ["sprayDrag", 10],
    ] as const) {
        if (foam[key] !== undefined) {
            finiteNumber(foam[key], `manifest.preset.foam.${key}`, 0, max);
        }
    }
    if ((foam.foamLifetimeMin as number) > (foam.foamLifetime as number)) {
        fail("manifest.preset.foam.foamLifetimeMin must not exceed foamLifetime");
    }
    for (const [minKey, maxKey] of [
        ["energySpeedMin", "energySpeedMax"],
        ["curvatureMin", "curvatureMax"],
        ["turbulenceMin", "turbulenceMax"],
    ] as const) {
        if (foam[minKey] !== undefined && foam[maxKey] !== undefined && (foam[minKey] as number) >= (foam[maxKey] as number)) {
            fail(`manifest.preset.foam.${minKey} must be less than ${maxKey}`);
        }
    }
    const bubbleColor = text(foam.subsurfaceBubbleColor, "manifest.preset.foam.subsurfaceBubbleColor");
    if (!HEX_COLOR.test(bubbleColor)) {
        fail("manifest.preset.foam.subsurfaceBubbleColor must be a #RRGGBB color");
    }
    const foamDebug = text(foam.foamDebug, "manifest.preset.foam.foamDebug", true);
    if (foamDebug === "" || foamDebug === "none") {
        foam.foamDebug = "off";
    } else if (
        foamDebug !== "off" &&
        foamDebug !== "accum" &&
        foamDebug !== "foamR" &&
        foamDebug !== "bubbleG" &&
        foamDebug !== "sprayB" &&
        foamDebug !== "blurred" &&
        foamDebug !== "foamAlpha" &&
        foamDebug !== "normals"
    ) {
        fail("manifest.preset.foam.foamDebug is not supported");
    }
}

function validatePreset(value: unknown): FluidExportJson {
    const preset = record(value, "manifest.preset");
    if (
        preset.formatVersion !== 5 &&
        preset.formatVersion !== 6 &&
        preset.formatVersion !== 7 &&
        preset.formatVersion !== 8 &&
        preset.formatVersion !== 9 &&
        preset.formatVersion !== 10 &&
        preset.formatVersion !== 11 &&
        preset.formatVersion !== 12 &&
        preset.formatVersion !== 13 &&
        preset.formatVersion !== 14 &&
        preset.formatVersion !== 15
    ) {
        fail("manifest preset must use formatVersion 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, or 15");
    }
    if (preset.formatVersion >= 14) {
        const semantics = record(preset.simulationSemantics, "manifest.preset.simulationSemantics");
        if (semantics.version !== 1) {
            fail("manifest.preset.simulationSemantics.version must be 1");
        }
        const profile = text(semantics.profile, "manifest.preset.simulationSemantics.profile");
        if (profile !== "normalized-v1" && profile !== "legacy-fluid" && profile !== "legacy-aquanova") {
            fail("manifest.preset.simulationSemantics.profile is not supported");
        }
        const pbfPhysics = text(semantics.pbfPhysics, "manifest.preset.simulationSemantics.pbfPhysics");
        if (pbfPhysics !== "scale-adjusted" && pbfPhysics !== "literal") {
            fail("manifest.preset.simulationSemantics.pbfPhysics is not supported");
        }
    }
    const meta = record(preset.meta, "manifest.preset.meta");
    if (meta.demo !== "blender") {
        fail('manifest.preset.meta.demo must be "blender"');
    }
    const method = text(meta.method, "manifest.preset.meta.method");
    if (!METHODS.includes(method as (typeof METHODS)[number])) {
        fail(`manifest.preset.meta.method "${method}" is not supported`);
    }
    validatePhysics(preset.physics, method);
    const demoParams = numericRecord(preset.demoParams, "manifest.preset.demoParams");
    const demoState = validateDemoState(preset.demoState);
    const resolution = method === "FLIP" && preset.gridResolution !== undefined ? integer(preset.gridResolution, "manifest.preset.gridResolution", 16, 2048) : undefined;
    const particleSize =
        preset.physicsParticleSize !== undefined
            ? finiteNumber(preset.physicsParticleSize, "manifest.preset.physicsParticleSize", PHYS_MIN_SCALE, PHYS_MAX_SCALE)
            : resolution !== undefined
              ? 1
              : finiteNumber(preset.physicsParticleSize, "manifest.preset.physicsParticleSize", PHYS_MIN_SCALE, PHYS_MAX_SCALE);
    integer(preset.particleCount, "manifest.preset.particleCount", 1, Number.MAX_SAFE_INTEGER);
    vector(preset.gridPosition, "manifest.preset.gridPosition", 3, -MAX_ABS_POSITION, MAX_ABS_POSITION);
    const gridSize = vector(preset.gridSize, "manifest.preset.gridSize", 3, Number.MIN_VALUE, MAX_EXTENT) as [number, number, number];
    if (preset.markersPerCell !== undefined) {
        integer(preset.markersPerCell, "manifest.preset.markersPerCell", 1, 64);
    }
    const simulationType = demoState.simulationType;
    const cellSize = resolution
        ? Math.max(...gridSize) / resolution
        : fluidSimulationCellSize(method, {
              physicsParticleSize: particleSize,
              samplingType: simulationType === "mesh" ? "mesh" : "fluid",
              particleRadius: demoParams.particleRadius,
          });
    const cells = gridCellsForSize(gridSize, cellSize);
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
    const sinks = sinkValues.map((sink, index) => validateSink(sink, `manifest.preset.sinks[${index}]`, preset.formatVersion as number));
    preset.emitters = emitters;
    preset.sinks = sinks;
    const ids = new Set<string>();
    for (const object of [...emitters, ...sinks]) {
        if (ids.has(object.id)) {
            fail(`manifest preset flow ID "${object.id}" is duplicated`);
        }
        ids.add(object.id);
    }
    const emitterIds = new Set(emitters.filter((emitter) => emitter.behavior === "inflow").map((emitter) => emitter.id));
    for (const [index, sink] of sinks.entries()) {
        for (const target of sink.mode === "recycle" ? sink.targets : []) {
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

    for (const key of ["showContainer", "msaa", "activeBlocks", "pagedGrid", "fusedBlockDiscovery", "showGridBounds", "showGridBoundsSolid", "initialEmittersFillCapacity"]) {
        if (preset[key] !== undefined) {
            bool(preset[key], `manifest.preset.${key}`);
        }
    }
    if (preset.envIntensity !== undefined) {
        finiteNumber(preset.envIntensity, "manifest.preset.envIntensity", 0, 100);
    }
    if (preset.simulationDuration !== undefined) {
        finiteNumber(preset.simulationDuration, "manifest.preset.simulationDuration", 0, 120);
    }
    if (preset.alphaDecay !== undefined) {
        finiteNumber(preset.alphaDecay, "manifest.preset.alphaDecay", 0, 10);
    }
    if (preset.simulationTimeScale !== undefined) {
        finiteNumber(preset.simulationTimeScale, "manifest.preset.simulationTimeScale", 0.01, 100);
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
        if (camera.target !== undefined) {
            vector(camera.target, "manifest.preset.camera.target", 3, -MAX_EXTENT, MAX_EXTENT);
        }
    }
    validateRender(preset.render);
    validateFoam(preset.foam);
    return preset as unknown as FluidExportJson;
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

function decodeBase64(value: unknown, path: string): Uint8Array {
    if (typeof value !== "string" || value.length === 0) {
        fail(`${path} must be a non-empty base64 string`);
    }
    const encoded = value;
    if (encoded.length > Math.ceil((MAX_ENTRY_BYTES * 4) / 3) + 4 || encoded.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)) {
        fail(`${path} is not valid bounded base64 data`);
    }
    let binary: string;
    try {
        binary = atob(encoded);
    } catch {
        fail(`${path} is not valid base64 data`);
    }
    if (binary.length > MAX_ENTRY_BYTES) {
        fail(`${path} exceeds the 512 MiB limit`);
    }
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function externalResourceName(value: unknown, path: string): string {
    const name = text(value, path).replaceAll("\\", "/");
    if (name.startsWith("/") || /^[a-z][a-z0-9+.-]*:/i.test(name) || name.split("/").includes("..")) {
        fail(`${path} must be a relative resource path without parent traversal`);
    }
    return name;
}

function externalResourceBytes(resources: BlenderFluidExternalResources | undefined, value: unknown, path: string): Uint8Array {
    const name = externalResourceName(value, path);
    const resource = resources?.get(name);
    if (!resource) {
        fail(`external resource "${name}" was not provided; select the JSON, GLB, and SDF files together`);
    }
    if (resource.byteLength > MAX_ENTRY_BYTES) {
        fail(`external resource "${name}" exceeds the 512 MiB limit`);
    }
    return new Uint8Array(resource);
}

function resourceSlice(bytes: Uint8Array, path: string, byteOffset: number, byteLength: number): Uint8Array {
    if (byteOffset > bytes.byteLength || byteLength > bytes.byteLength - byteOffset) {
        fail(`${path} byte range exceeds the provided external resource`);
    }
    return bytes.subarray(byteOffset, byteOffset + byteLength);
}

export function parseBlenderFluidCollision(bytes: Uint8Array): BlenderFluidCollision {
    if (bytes.byteLength < SDF_HEADER_BYTES) {
        fail("collision.sdf is truncated");
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== SDF_MAGIC) {
        fail("collision.sdf has invalid magic");
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

/** Parse an embedded or external-resource format-6 through format-15 fluid JSON export. */
export function parseBlenderFluidJson(contents: string, externalResources?: BlenderFluidExternalResources): BlenderFluidScene {
    if (contents.length > MAX_JSON_BYTES) {
        fail("JSON export exceeds the 768 MiB limit");
    }
    let value: unknown;
    try {
        value = JSON.parse(contents);
    } catch {
        fail("export is not valid JSON");
    }
    const preset = validatePreset(value);
    if (
        preset.formatVersion !== 6 &&
        preset.formatVersion !== 7 &&
        preset.formatVersion !== 8 &&
        preset.formatVersion !== 9 &&
        preset.formatVersion !== 10 &&
        preset.formatVersion !== 11 &&
        preset.formatVersion !== 12 &&
        preset.formatVersion !== 13 &&
        preset.formatVersion !== 14 &&
        preset.formatVersion !== 15
    ) {
        fail("self-contained fluid JSON must use formatVersion 6, 7, 8, 9, 10, 11, 12, 13, 14, or 15");
    }
    const scene = record(preset.scene, "manifest.preset.scene");
    if (scene.sdfCompression !== undefined && scene.sdfCompression !== "zlib") {
        fail('manifest.preset.scene.sdfCompression must be "zlib"');
    }
    const sdfCompressed = scene.sdfCompression === "zlib";
    const collisionEnabled = scene.collisionEnabled === undefined ? true : bool(scene.collisionEnabled, "manifest.preset.scene.collisionEnabled");
    const collisionTrilinear = scene.collisionTrilinear === undefined ? true : bool(scene.collisionTrilinear, "manifest.preset.scene.collisionTrilinear");
    scene.collisionEnabled = collisionEnabled;
    scene.collisionTrilinear = collisionTrilinear;
    if (scene.anchorPosition !== undefined) {
        vector(scene.anchorPosition, "manifest.preset.scene.anchorPosition", 3, -MAX_ABS_POSITION, MAX_ABS_POSITION);
    }
    const animatedEntries = scene.animatedCollisions === undefined ? [] : array(scene.animatedCollisions, "manifest.preset.scene.animatedCollisions");
    if (animatedEntries.length > MAX_ANIMATED_COLLISIONS) {
        fail(`manifest.preset.scene.animatedCollisions supports at most ${MAX_ANIMATED_COLLISIONS} entries`);
    }
    const ids = new Set<string>();
    const nodes = new Set<string>();
    const animatedCollisions: BlenderFluidAnimatedCollision[] = [];
    let glbBytes: Uint8Array;
    let collisionBytes: Uint8Array;
    const externalSdfResources = new Map<string, Uint8Array>();
    const readExternalSdf = (value: unknown, path: string): Uint8Array => {
        const name = externalResourceName(value, path);
        let bytes = externalSdfResources.get(name);
        if (!bytes) {
            bytes = decompressSdfBytes(externalResourceBytes(externalResources, name, path), path, sdfCompressed);
            externalSdfResources.set(name, bytes);
        }
        return bytes;
    };
    if (scene.encoding === "base64") {
        if (scene.collisionByteLength !== undefined) {
            fail("manifest.preset.scene.collisionByteLength is only valid for external resources");
        }
        glbBytes = decodeBase64(scene.glb, "manifest.preset.scene.glb");
        collisionBytes = decompressSdfBytes(decodeBase64(scene.collision, "manifest.preset.scene.collision"), "manifest.preset.scene.collision", sdfCompressed);
    } else if (scene.encoding === "external") {
        glbBytes = externalResourceBytes(externalResources, scene.glb, "manifest.preset.scene.glb");
        const collisionResource = readExternalSdf(scene.collision, "manifest.preset.scene.collision");
        collisionBytes =
            scene.collisionByteLength === undefined
                ? collisionResource
                : resourceSlice(
                      collisionResource,
                      "manifest.preset.scene.collision",
                      0,
                      integer(scene.collisionByteLength, "manifest.preset.scene.collisionByteLength", SDF_HEADER_BYTES, MAX_ENTRY_BYTES)
                  );
    } else {
        fail('manifest.preset.scene.encoding must be "base64" or "external"');
    }
    const sceneGlb = parseGlb(glbBytes);
    const collision = parseBlenderFluidCollision(collisionBytes);
    let totalVoxels = collision.distances.length;
    for (let index = 0; index < animatedEntries.length; index++) {
        const path = `manifest.preset.scene.animatedCollisions[${index}]`;
        const entry = record(animatedEntries[index], path);
        const id = text(entry.id, `${path}.id`);
        const node = text(entry.node, `${path}.node`);
        if (ids.has(id)) {
            fail(`${path}.id must be unique`);
        }
        if (nodes.has(node)) {
            fail(`${path}.node must be unique`);
        }
        ids.add(id);
        nodes.add(node);
        if (entry.space !== "node-local") {
            fail(`${path}.space must be "node-local"`);
        }
        const resolution = integer(entry.resolution, `${path}.resolution`, 1, 2048);
        const bakeFrame = finiteNumber(entry.bakeFrame, `${path}.bakeFrame`, -1_000_000, 1_000_000);
        if (typeof entry.presentation !== "boolean") {
            fail(`${path}.presentation must be a boolean`);
        }
        const enabled = entry.enabled === undefined ? true : bool(entry.enabled, `${path}.enabled`);
        const trilinear = entry.trilinear === undefined ? true : bool(entry.trilinear, `${path}.trilinear`);
        entry.enabled = enabled;
        entry.trilinear = trilinear;
        const hasByteOffset = entry.byteOffset !== undefined;
        const hasByteLength = entry.byteLength !== undefined;
        if (hasByteOffset !== hasByteLength) {
            fail(`${path}.byteOffset and byteLength must be provided together`);
        }
        let animatedBytes: Uint8Array;
        if (scene.encoding === "base64") {
            if (hasByteOffset) {
                fail(`${path} byte ranges are only valid for external resources`);
            }
            animatedBytes = decompressSdfBytes(decodeBase64(entry.sdf, `${path}.sdf`), `${path}.sdf`, sdfCompressed);
        } else if (hasByteOffset) {
            animatedBytes = resourceSlice(
                readExternalSdf(entry.sdf, `${path}.sdf`),
                `${path}.sdf`,
                integer(entry.byteOffset, `${path}.byteOffset`, 0, MAX_ENTRY_BYTES),
                integer(entry.byteLength, `${path}.byteLength`, SDF_HEADER_BYTES, MAX_ENTRY_BYTES)
            );
        } else {
            animatedBytes = readExternalSdf(entry.sdf, `${path}.sdf`);
        }
        const animatedCollision = parseBlenderFluidCollision(animatedBytes);
        totalVoxels += animatedCollision.distances.length;
        if (totalVoxels > MAX_SDF_VOXELS) {
            fail("combined static and animated collision grids exceed the 16M-voxel limit");
        }
        animatedCollisions.push({
            id,
            node,
            space: "node-local",
            resolution,
            bakeFrame,
            presentation: entry.presentation,
            enabled,
            trilinear,
            collision: animatedCollision,
        });
    }
    return { preset, sceneGlb, collision, collisionEnabled, collisionTrilinear, animatedCollisions };
}
