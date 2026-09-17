import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { basename, dirname, isAbsolute, relative, resolve } from "node:path";
import { createHash } from "node:crypto";
import { parseBlenderFluidJson } from "../../packages/babylon-lite/src/fluid/authoring/blender-fluid-json";
import type { BlenderFluidCollision } from "../../packages/babylon-lite/src/fluid/authoring/blender-fluid-json";
import { parseGlbContainer } from "../../packages/babylon-lite/src/loader-gltf/gltf-glb-parser";
import { computeNodeWorldMatrix, resolveAccessor } from "../../packages/babylon-lite/src/loader-gltf/gltf-parser";
import { evaluateSampler } from "../../packages/babylon-lite/src/animation/evaluate";
import type { AnimationSampler } from "../../packages/babylon-lite/src/animation/types";
import { mat4Multiply } from "../../packages/babylon-lite/src/math/mat4-multiply";
import { mat4Scale } from "../../packages/babylon-lite/src/math/mat4-scale";
import type { Mat4 } from "../../packages/babylon-lite/src/math/types";
import { sampleReferenceObstacleFrames } from "./animation";
import type { ParticleState, ReferenceCase, ReferenceGrid, ReferenceSdf, Triple } from "./types";

interface GlbNode {
    name?: string;
    children?: number[];
    mesh?: number;
    translation?: number[];
    rotation?: number[];
    scale?: number[];
    matrix?: number[];
}

interface GlbDocument {
    nodes: GlbNode[];
    accessors: Array<{ bufferView?: number; sparse?: unknown; type: string; componentType: number }>;
    bufferViews: Array<{ byteStride?: number }>;
    meshes?: Array<{ primitives: Array<{ attributes: Record<string, number>; indices?: number; mode?: number }> }>;
    animations?: Array<{
        samplers: Array<{ input: number; output: number; interpolation?: string }>;
        channels: Array<{ sampler: number; target: { node?: number; path: string } }>;
    }>;
}

interface SourceManifest {
    source?: {
        settings?: {
            timeline?: { frameStart?: number; frameEnd?: number; fps?: number; fpsBase?: number; simulationFps?: number };
            domain?: {
                simulation?: { frame_rate_custom?: number; frame_rate_mode?: string };
                advanced?: { enable_extreme_velocity_removal?: boolean };
                world?: { world_scale_relative?: number; time_scale?: number };
            };
        };
    };
    scene?: { encoding?: string; glb?: string; collision?: string; animatedCollisions?: Array<{ sdf?: string }>; initialState?: { data?: string } };
}

interface BoundsFile {
    x: number;
    y: number;
    z: number;
    width: number;
    height: number;
    depth: number;
    dx: number;
    isize: number;
    jsize: number;
    ksize: number;
}

export function writeParticleState(path: string, state: ParticleState): void {
    const count = state.positions.length / 3;
    if (!Number.isSafeInteger(count) || count < 0 || state.velocities.length !== state.positions.length) {
        throw new Error("Cannot encode unaligned particle snapshot.");
    }
    const buffer = new ArrayBuffer(8 + count * 24);
    new DataView(buffer).setUint32(0, count, true);
    new Float32Array(buffer, 8, count * 3).set(state.positions);
    new Float32Array(buffer, 8 + count * 12, count * 3).set(state.velocities);
    writeFileSync(path, new Uint8Array(buffer));
}

export function readReferenceParticles(cache: string, frame: number): ParticleState {
    const suffix = String(frame).padStart(6, "0");
    const positions = readFileSync(resolve(cache, "bakefiles", `fluidparticles${suffix}.ffp3`));
    const velocities = readFileSync(resolve(cache, "bakefiles", `fluidparticlesvelocity${suffix}.ffp3`));
    if (positions.length < 16 || velocities.length < 16) {
        throw new Error(`Missing FFP3 header at frame ${frame}.`);
    }
    const count = positions.readUInt32LE(0) + positions.readUInt32LE(4) + positions.readUInt32LE(8);
    const offset = 16 + 12 * positions.readUInt32LE(12);
    if (positions.length !== offset + count * 12 || velocities.length !== positions.length || !positions.subarray(0, offset).equals(velocities.subarray(0, offset))) {
        throw new Error(`Position/velocity FFP3 streams do not have matching category counts and ID tables at frame ${frame}.`);
    }
    const result = { positions: new Float32Array(count * 3), velocities: new Float32Array(count * 3) };
    for (let i = 0; i < count; i++) {
        for (const [source, destination] of [
            [positions, result.positions],
            [velocities, result.velocities],
        ] as const) {
            destination[i * 3] = source.readFloatLE(offset + i * 12);
            destination[i * 3 + 1] = source.readFloatLE(offset + i * 12 + 8);
            destination[i * 3 + 2] = -source.readFloatLE(offset + i * 12 + 4);
        }
    }
    return result;
}

function positive(value: number | undefined, name: string): number {
    if (value === undefined || !Number.isFinite(value) || value <= 0) {
        throw new Error(`${name} must be declared and positive.`);
    }
    return value;
}

function referenceGrid(bounds: BoundsFile): ReferenceGrid {
    const dimensions: Triple = [bounds.isize, bounds.ksize, bounds.jsize];
    const cellSize = positive(bounds.dx, "Cache cell size");
    if (dimensions.some((d) => !Number.isSafeInteger(d) || d < 1)) {
        throw new Error("Invalid cached grid dimensions.");
    }
    for (const [extent, cells] of [
        [bounds.width, bounds.isize],
        [bounds.height, bounds.jsize],
        [bounds.depth, bounds.ksize],
    ]) {
        if (!Number.isFinite(extent) || Math.abs(extent! - cells! * cellSize) > 1e-5 * cellSize) {
            throw new Error("Cached grid extents and spacing disagree.");
        }
    }
    const origin: Triple = [bounds.x, bounds.z, -bounds.y - bounds.height];
    if (origin.some((v) => !Number.isFinite(v))) {
        throw new Error("Invalid cached grid origin.");
    }
    return { origin, dimensions, cellSize };
}

function externalResource(input: string, name: string): string {
    const directory = dirname(input);
    const path = resolve(directory, name);
    const child = relative(directory, path);
    if (isAbsolute(name) || isAbsolute(child) || child.startsWith("..")) {
        throw new Error(`Resource must be inside the fluid bundle directory: ${name}.`);
    }
    return path;
}

function floatAccessor(json: GlbDocument, bin: DataView, index: number): Float32Array {
    const accessor = json.accessors[index];
    if (!accessor || accessor.componentType !== 5126 || accessor.sparse || (accessor.bufferView !== undefined && json.bufferViews[accessor.bufferView]?.byteStride)) {
        throw new Error("The comparison fixture requires tightly packed nonsparse float accessors.");
    }
    const result = resolveAccessor(json, bin, index)._data;
    if (!(result instanceof Float32Array)) {
        throw new Error("Expected float glTF accessor.");
    }
    return result;
}

export function prepareBlenderCase(input: string, cache: string, output: string, frames: number, substepsOverride?: number): ReferenceCase {
    mkdirSync(output, { recursive: true });
    const text = readFileSync(input, "utf8");
    const manifest = JSON.parse(text) as SourceManifest;
    const resources = new Map<string, ArrayBuffer>();
    if (manifest.scene?.encoding === "external") {
        const names = [manifest.scene.glb, manifest.scene.collision, manifest.scene.initialState?.data, ...(manifest.scene.animatedCollisions ?? []).map((s) => s.sdf)];
        for (const name of names) {
            if (name && !resources.has(name)) {
                const bytes = readFileSync(externalResource(input, name));
                resources.set(name, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
            }
        }
    }
    const bundle = parseBlenderFluidJson(text, resources);
    const timing = manifest.source?.settings?.timeline;
    const simulation = manifest.source?.settings?.domain?.simulation;
    const timelineFps = positive(timing?.fps, "Timeline FPS") / positive(timing?.fpsBase ?? 1, "Timeline FPS base");
    const simulationFps = positive(
        timing?.simulationFps ?? (simulation?.frame_rate_mode === "FRAME_RATE_MODE_CUSTOM" ? simulation.frame_rate_custom : timelineFps),
        "Simulation FPS"
    );
    const startFrame = timing?.frameStart;
    if (!Number.isSafeInteger(startFrame) || !Number.isSafeInteger(timing?.frameEnd) || startFrame === undefined || startFrame + frames - 1 > timing!.frameEnd!) {
        throw new Error("Requested frames exceed the explicitly authored frame range.");
    }
    if (manifest.source?.settings?.domain?.world?.world_scale_relative !== 1 || bundle.preset.simulationTimeScale !== 1) {
        throw new Error("This first reference experiment requires unit world scale and simulation time scale.");
    }
    const boundsPath = resolve(cache, "bakefiles", `bounds${String(startFrame).padStart(6, "0")}.bbox`);
    const bounds = JSON.parse(readFileSync(boundsPath, "utf8")) as BoundsFile;
    const grid = referenceGrid(bounds);
    const initial = readReferenceParticles(cache, startFrame);
    if (!bundle.initialState || bundle.initialState.frame !== startFrame || bundle.initialState.positions.length !== initial.positions.length) {
        throw new Error("Export a baked exact initial state matching this cache before running the comparison.");
    }
    let seedError = 0;
    for (let i = 0; i < initial.positions.length; i++) {
        seedError = Math.max(seedError, Math.abs(bundle.initialState.positions[i]! - initial.positions[i]!), Math.abs(bundle.initialState.velocities[i]! - initial.velocities[i]!));
    }
    if (seedError !== 0) {
        throw new Error(`Bundle initial particles differ from the selected reference cache (max error ${seedError}).`);
    }
    const physics = bundle.preset.physics;
    const substeps = substepsOverride ?? positive(physics.minSubsteps, "Substeps");
    if (!Number.isSafeInteger(substeps) || substeps < 1) {
        throw new Error("Substeps must be a positive integer.");
    }
    const camera = bundle.preset.camera;
    if (!camera?.target || !camera.fov) {
        throw new Error("An exported perspective camera is required for the comparison.");
    }
    const flipRatio = physics.flipRatio;
    if (flipRatio === undefined || !Number.isFinite(flipRatio) || flipRatio < 0 || flipRatio > 1) {
        throw new Error("The fixture must explicitly declare its FLIP/PIC ratio.");
    }
    const removeExtreme = manifest.source?.settings?.domain?.advanced?.enable_extreme_velocity_removal;
    if (typeof removeExtreme !== "boolean") {
        throw new Error("Native extreme-velocity removal must be explicitly declared in the source metadata.");
    }
    const cfl = positive(physics.cflNumber, "Native CFL number");
    const maxFrameSubsteps = positive(physics.maxSubsteps, "Native maximum substeps");
    if (!Number.isSafeInteger(maxFrameSubsteps)) {
        throw new Error("Native maximum substeps must be an integer.");
    }
    const result: ReferenceCase = {
        version: 1,
        label: basename(input),
        grid,
        domainInset: 1.5 * grid.cellSize + 0.00005,
        startFrame,
        frames,
        simulationFps,
        timelineFps,
        substeps,
        gravity: [0, -positive(physics.gravity, "Gravity"), 0],
        picFraction: 1 - flipRatio,
        ...(removeExtreme ? { extremeVelocityRemoval: { cfl, maxFrameSubsteps } } : {}),
        initialState: "initial.bin",
        obstacles: [],
        meshes: [],
        referencePattern: "reference-{frame}.bin",
        camera: { ...camera, target: [...camera.target], fov: camera.fov, mirrorX: camera.mirrorX ?? false },
        provenance: {
            input,
            cache,
            manifestSha256: createHash("sha256").update(text).digest("hex"),
            seedMaxError: seedError,
            coordinateBasis: "Blender (x,y,z) -> (x,z,-y)",
            cachePhase: "visible cache is written after first substep; seed continuation starts at timeline start + 1/substeps",
            exportedGridSize: bundle.preset.gridSize,
            voxelSnappedGridSize: grid.dimensions.map((n) => n * grid.cellSize),
            disabledFeatures: ["whitewater", "reseeding", "surface tension", "viscosity"],
        },
    };
    if (physics.kinematicViscosity || physics.surfaceTension || bundle.preset.sinks?.length || bundle.preset.emitters?.some((e) => e.behavior !== "initial")) {
        throw new Error("This experiment supports inviscid liquid with initial-only emitters and no sinks.");
    }
    writeParticleState(resolve(output, result.initialState), initial);
    for (let f = startFrame; f < startFrame + frames; f++) {
        writeParticleState(resolve(output, `reference-${f}.bin`), f === startFrame ? initial : readReferenceParticles(cache, f));
    }
    const saveSdf = (collision: BlenderFluidCollision, file: string): ReferenceSdf => {
        writeFileSync(resolve(output, file), new Uint8Array(collision.distances.buffer, collision.distances.byteOffset, collision.distances.byteLength));
        return { dimensions: collision.dims, origin: collision.origin, cellSize: collision.cellSize, file };
    };
    if (bundle.collisionEnabled) {
        result.staticSdf = saveSdf(bundle.collision, "static-sdf.f32");
    }
    const parsed = parseGlbContainer(bundle.sceneGlb);
    const json: GlbDocument = parsed.json;
    const parents = new Map<number, number>();
    json.nodes.forEach((node, parent) => {
        for (const child of node.children ?? []) {
            if (parents.has(child)) {
                throw new Error("Comparison glTF hierarchy contains a multiply parented node.");
            }
            parents.set(child, parent);
        }
    });
    const samplers: Array<{ node: number; path: "translation" | "rotation" | "scale"; sampler: AnimationSampler }> = [];
    const animatedPaths = new Set<string>();
    for (const animation of json.animations ?? []) {
        for (const channel of animation.channels) {
            const path = channel.target.path;
            const node = channel.target.node;
            if (node === undefined || (path !== "translation" && path !== "rotation" && path !== "scale")) {
                throw new Error("Unsupported non-transform animation in comparison fixture.");
            }
            const key = `${node}:${path}`;
            if (animatedPaths.has(key)) {
                throw new Error("Ambiguous competing glTF clips in comparison fixture.");
            }
            animatedPaths.add(key);
            const source = animation.samplers[channel.sampler]!;
            const interpolation = source.interpolation ?? "LINEAR";
            if (!["LINEAR", "STEP", "CUBICSPLINE"].includes(interpolation)) {
                throw new Error(`Unsupported interpolation: ${interpolation}.`);
            }
            samplers.push({
                node,
                path,
                sampler: {
                    input: floatAccessor(json, parsed.binChunk, source.input),
                    output: floatAccessor(json, parsed.binChunk, source.output),
                    interpolation: interpolation === "STEP" ? 1 : interpolation === "CUBICSPLINE" ? 2 : 0,
                },
            });
        }
    }
    const scratch = new Float32Array(4);
    const undoLoaderReflection = mat4Scale(-1, 1, 1);
    const matricesAt = (time: number): Map<number, Mat4> => {
        for (const channel of samplers) {
            const arity = channel.path === "rotation" ? 4 : 3;
            evaluateSampler(channel.sampler, time, arity, arity === 4, scratch, 0);
            json.nodes[channel.node]![channel.path] = Array.from(scratch.subarray(0, arity));
        }
        const matrices = new Map<number, Mat4>();
        json.nodes.forEach((_, index) => computeNodeWorldMatrix(json, index, parents, matrices));
        // Particle/SDF exports already use the bundle basis, not the general glTF loader's LH root.
        for (const [index, matrix] of matrices) {
            matrices.set(index, mat4Multiply(undoLoaderReflection, matrix));
        }
        return matrices;
    };
    const frameWorlds = Array.from({ length: frames }, (_, index) => matricesAt(index / timelineFps));
    for (const [i, obstacle] of bundle.animatedCollisions.entries()) {
        if (!obstacle.enabled) {
            continue;
        }
        const node = json.nodes.findIndex((n) => n.name === obstacle.node);
        if (node < 0) {
            throw new Error(`Missing animated collision node: ${obstacle.node}.`);
        }
        const definition = {
            ...saveSdf(obstacle.collision, `local-sdf-${i}.f32`),
            name: obstacle.node,
            ...sampleReferenceObstacleFrames(
                frameWorlds.map((matrices) => matrices.get(node)!),
                substeps,
                simulationFps
            ),
        };
        result.obstacles.push(definition);
    }
    const worlds = matricesAt(0);
    for (const [nodeIndex, node] of json.nodes.entries()) {
        if (node.mesh === undefined || node.name === "Water" || node.name?.startsWith("fluid_particles")) {
            continue;
        }
        for (const [primitiveIndex, primitive] of (json.meshes?.[node.mesh]?.primitives ?? []).entries()) {
            if (primitive.mode !== undefined && primitive.mode !== 4) {
                continue;
            }
            const positionIndex = primitive.attributes.POSITION;
            if (positionIndex === undefined || primitive.indices === undefined) {
                continue;
            }
            const positions = floatAccessor(json, parsed.binChunk, positionIndex);
            const sourceIndices = resolveAccessor(json, parsed.binChunk, primitive.indices)._data;
            if (!(sourceIndices instanceof Uint16Array || sourceIndices instanceof Uint32Array || sourceIndices instanceof Uint8Array)) {
                throw new Error("Invalid triangle index accessor.");
            }
            const indices = Uint32Array.from(sourceIndices);
            const prefix = `mesh-${nodeIndex}-${primitiveIndex}`;
            writeFileSync(resolve(output, `${prefix}.f32`), new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength));
            writeFileSync(resolve(output, `${prefix}.u32`), new Uint8Array(indices.buffer));
            const obstacle = result.obstacles.findIndex((o) => o.name === node.name);
            const world = worlds.get(nodeIndex)!;
            result.meshes.push({
                name: node.name ?? prefix,
                positions: `${prefix}.f32`,
                indices: `${prefix}.u32`,
                color: node.name === "Sand" ? [0.58, 0.45, 0.3] : node.name === "rock" ? [0.85, 0.85, 0.85] : [0.22, 0.23, 0.25],
                transform: Array.from({ length: 16 }, (_, j) => world[j]!),
                ...(obstacle >= 0 ? { obstacle } : {}),
            });
        }
    }
    writeFileSync(resolve(output, "case.json"), JSON.stringify(result));
    return result;
}

export function prepareAnalyticCase(output: string, frames: number): ReferenceCase {
    mkdirSync(output, { recursive: true });
    const positions: number[] = [];
    const dx = 0.1;
    for (let z = 0; z < 12; z++) {
        for (let y = 0; y < 12; y++) {
            for (let x = 0; x < 12; x++) {
                positions.push(0.15 + ((x + 0.5) * dx) / 2, 0.15 + ((y + 0.5) * dx) / 2, 0.15 + ((z + 0.5) * dx) / 2);
            }
        }
    }
    const result: ReferenceCase = {
        version: 1,
        label: "Analytic dam break (not a Blender reference)",
        grid: { origin: [0, 0, 0], dimensions: [20, 12, 20], cellSize: dx },
        domainInset: 0,
        startFrame: 0,
        frames,
        simulationFps: 50,
        timelineFps: 25,
        substeps: 2,
        gravity: [0, -9.81, 0],
        picFraction: 0.05,
        initialState: "initial.bin",
        obstacles: [],
        meshes: [],
        camera: { alpha: 0.85, beta: 1.12, radius: 4, target: [1, 0.4, 1], fov: 0.7, mirrorX: false },
        provenance: { purpose: "numerical smoke case; no native reference and no claim of parity" },
    };
    writeParticleState(resolve(output, "initial.bin"), { positions: Float32Array.from(positions), velocities: new Float32Array(positions.length) });
    writeFileSync(resolve(output, "case.json"), JSON.stringify(result));
    return result;
}
