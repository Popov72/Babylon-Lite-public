/**
 * Opt-in PBR box-projected local cubemap reflections.
 *
 * Local environment state, shader code, and GPU binding logic live entirely in
 * this feature chunk. PBR scenes that do not call enablePbrLocalCubemap() retain
 * the ordinary material shape, shader path, and renderable update loop.
 */

import { BU, TU } from "../../engine/gpu-flags.js";
import { createMappedBuffer, createUniformBuffer } from "../../resource/gpu-buffers.js";
import type { SceneContext } from "../../scene/scene.js";
import { _registerPbrExt } from "./pbr-flags.js";
import {
    _initializePbrLocalCubemapLimits,
    _PBR_LOCAL_ENVIRONMENT_CANDIDATE_CAPACITY,
    _PBR_LOCAL_ENVIRONMENT_DEBUG_COLOR_FLAG,
    _PBR_LOCAL_ENVIRONMENT_HEADER_U32,
    _PBR_LOCAL_ENVIRONMENT_PARALLAX_FLAG,
    _PBR_LOCAL_ENVIRONMENT_PROBE_FLOATS,
    _PBR_LOCAL_ENVIRONMENT_UNIFORM_FLOATS,
    MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES,
    MAX_PBR_LOCAL_ENVIRONMENT_PROBES,
} from "./pbr-local-cubemap-limits.js";
import type { PbrMaterialProps } from "./pbr-material.js";
import { _setPbrLocalEnvironment, type PbrLocalEnvironmentOptions, type PbrLocalEnvironmentProbe, type PbrLocalEnvironmentProbeSet } from "./pbr-local-cubemap-state.js";
import type { EnvironmentTextures } from "../../loader-env/load-env.js";

export { MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES, MAX_PBR_LOCAL_ENVIRONMENT_PROBES } from "./pbr-local-cubemap-limits.js";
export type { PbrLocalEnvironmentOptions, PbrLocalEnvironmentProbe, PbrLocalEnvironmentProbeSet } from "./pbr-local-cubemap-state.js";

export interface PbrLocalCubemapInitOptions {
    /** Maximum probes evaluated per fragment. Defaults to 4 and cannot change after initialization. */
    readonly maxCandidates?: number;
}

export interface PbrLocalEnvironmentProbeSetOptions {
    readonly probes: readonly PbrLocalEnvironmentProbe[];
    readonly voxelGrid: PbrLocalEnvironmentProbeGridOptions;
    readonly parallaxCorrection?: boolean;
}

export interface PbrLocalEnvironmentProbeGridOptions {
    /** Inclusive world-space minimum corner. */
    readonly minimum: readonly [number, number, number];
    /** World-space maximum corner. Dimensions are rounded up to whole cells. */
    readonly maximum: readonly [number, number, number];
    /** Uniform voxel edge length. */
    readonly cellSize: number;
}

export interface PbrLocalEnvironmentProbeGridCell {
    readonly coordinates: readonly [number, number, number];
    readonly probeIndices: readonly number[];
    /** True when the queried point was clamped to a boundary voxel. */
    readonly outside: boolean;
}

/**
 * Assign one bounded local environment to a PBR material.
 *
 * Configure before scene registration. If bindings already exist, call
 * `rebuildMaterial(scene, material)` after changing or clearing the assignment.
 */
export function setPbrLocalEnvironment(material: PbrMaterialProps, environment: EnvironmentTextures, options: PbrLocalEnvironmentOptions): void {
    finiteVec3(options.projectionPosition, "local environment projectionPosition");
    finiteVec3(options.projectionSize, "local environment projectionSize", true);
    _setPbrLocalEnvironment(material, {
        kind: "single",
        environment,
        projectionPosition: [...options.projectionPosition],
        projectionSize: [...options.projectionSize],
    });
}

/**
 * Assign a fragment-blended probe set to a PBR material.
 *
 * Configure before scene registration. If bindings already exist, call
 * `rebuildMaterial(scene, material)` after changing or clearing the assignment.
 */
export function setPbrLocalEnvironmentProbeSet(material: PbrMaterialProps, set: PbrLocalEnvironmentProbeSet): void {
    _setPbrLocalEnvironment(material, { kind: "probes", set });
}

/** Clear an opt-in local environment assignment from a PBR material. */
export function clearPbrLocalEnvironment(material: PbrMaterialProps): void {
    _setPbrLocalEnvironment(material, null);
}

function finiteVec3(value: readonly number[], name: string, positive = false): void {
    if (value.length !== 3 || value.some((component) => !Number.isFinite(component) || (positive && component <= 0))) {
        throw new Error(`[babylon-lite] ${name} must contain three finite${positive ? " positive" : ""} values`);
    }
}

function validateProbe(probe: PbrLocalEnvironmentProbe, index: number): void {
    finiteVec3(probe.capturePosition, `local probe ${index} capturePosition`);
    finiteVec3(probe.projectionPosition, `local probe ${index} projectionPosition`);
    finiteVec3(probe.projectionSize, `local probe ${index} projectionSize`, true);
    finiteVec3(probe.influencePosition, `local probe ${index} influencePosition`);
    finiteVec3(probe.influenceInnerSize, `local probe ${index} influenceInnerSize`);
    finiteVec3(probe.influenceOuterSize, `local probe ${index} influenceOuterSize`, true);
    for (let axis = 0; axis < 3; axis++) {
        if (probe.influenceInnerSize[axis]! < 0 || probe.influenceInnerSize[axis]! > probe.influenceOuterSize[axis]!) {
            throw new Error(`[babylon-lite] local probe ${index} influenceInnerSize must be non-negative and no larger than influenceOuterSize`);
        }
    }
    if (probe.angleRadians !== undefined && !Number.isFinite(probe.angleRadians)) {
        throw new Error(`[babylon-lite] local probe ${index} angleRadians must be finite`);
    }
    if (probe.debugColor !== undefined && (probe.debugColor.length !== 3 || probe.debugColor.some((component) => !Number.isFinite(component) || component < 0 || component > 1))) {
        throw new Error(`[babylon-lite] local probe ${index} debugColor must contain three finite values from 0 to 1`);
    }
}

function powerOfTwoRatio(value: number, base: number): number {
    const ratio = value / base;
    const offset = Math.log2(ratio);
    if (!Number.isInteger(offset) || offset < 0) {
        throw new Error(`[babylon-lite] local probe cubemap dimensions must differ only by power-of-two mip levels`);
    }
    return offset;
}

function packDebugColor(color: readonly [number, number, number] | undefined): number {
    const value = color ?? [1, 0, 1];
    return Math.round(value[0] * 255) | (Math.round(value[1] * 255) << 8) | (Math.round(value[2] * 255) << 16);
}

function writeProbe(data: Float32Array, u32: Uint32Array, probe: PbrLocalEnvironmentProbe, index: number, sourceMipOffset: number): void {
    const base = _PBR_LOCAL_ENVIRONMENT_HEADER_U32 + index * _PBR_LOCAL_ENVIRONMENT_PROBE_FLOATS;
    const projectionHalf = probe.projectionSize.map((value) => value * 0.5);
    const influenceInnerHalf = probe.influenceInnerSize.map((value) => value * 0.5);
    const influenceOuterHalf = probe.influenceOuterSize.map((value) => value * 0.5);
    const angle = probe.angleRadians ?? 0;
    const lodScale = probe.environment._lodGenerationScale ?? 0.8;
    const lodBias = sourceMipOffset * (lodScale - 1);

    data.set(probe.projectionPosition, base);
    data[base + 3] = index;
    data.set(projectionHalf, base + 4);
    data[base + 7] = lodScale;
    data.set(probe.capturePosition, base + 8);
    data[base + 11] = lodBias;
    data.set(probe.influencePosition, base + 12);
    data[base + 15] = Math.cos(angle);
    data.set(influenceInnerHalf, base + 16);
    data[base + 19] = Math.sin(angle);
    data.set(influenceOuterHalf, base + 20);
    u32[base + 23] = packDebugColor(probe.debugColor);
}

const GRID_HEADER_U32 = 8;
const GRID_EPSILON = 1e-6;

interface BuiltProbeGrid {
    readonly data: Uint32Array;
    readonly minimum: readonly [number, number, number];
    readonly cellSize: number;
    readonly dimensions: readonly [number, number, number];
    readonly stride: number;
}

function validateGrid(options: PbrLocalEnvironmentProbeGridOptions): void {
    finiteVec3(options.minimum, "local probe voxelGrid.minimum");
    finiteVec3(options.maximum, "local probe voxelGrid.maximum");
    if (!Number.isFinite(options.cellSize) || options.cellSize <= 0) {
        throw new Error("[babylon-lite] local probe voxelGrid.cellSize must be finite and positive");
    }
    for (let axis = 0; axis < 3; axis++) {
        if (options.maximum[axis]! <= options.minimum[axis]!) {
            throw new Error("[babylon-lite] local probe voxelGrid.maximum must be greater than minimum on every axis");
        }
    }
}

function probeOuterWorldExtent(probe: PbrLocalEnvironmentProbe): [number, number, number] {
    const halfX = probe.influenceOuterSize[0] * 0.5;
    const halfY = probe.influenceOuterSize[1] * 0.5;
    const halfZ = probe.influenceOuterSize[2] * 0.5;
    const angle = probe.angleRadians ?? 0;
    const cosine = Math.abs(Math.cos(angle));
    const sine = Math.abs(Math.sin(angle));
    return [cosine * halfX + sine * halfZ, halfY, sine * halfX + cosine * halfZ];
}

function intersectsProbeOuterBox(probe: PbrLocalEnvironmentProbe, cellCentre: readonly number[], cellHalfSize: number): boolean {
    const dx = cellCentre[0]! - probe.influencePosition[0];
    const dy = cellCentre[1]! - probe.influencePosition[1];
    const dz = cellCentre[2]! - probe.influencePosition[2];
    const probeHalfX = probe.influenceOuterSize[0] * 0.5;
    const probeHalfY = probe.influenceOuterSize[1] * 0.5;
    const probeHalfZ = probe.influenceOuterSize[2] * 0.5;
    if (Math.abs(dy) > cellHalfSize + probeHalfY + GRID_EPSILON) {
        return false;
    }

    const angle = probe.angleRadians ?? 0;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const absCosine = Math.abs(cosine);
    const absSine = Math.abs(sine);
    if (Math.abs(dx) > cellHalfSize + absCosine * probeHalfX + absSine * probeHalfZ + GRID_EPSILON) {
        return false;
    }
    if (Math.abs(dz) > cellHalfSize + absSine * probeHalfX + absCosine * probeHalfZ + GRID_EPSILON) {
        return false;
    }

    const localX = cosine * dx - sine * dz;
    const localZ = sine * dx + cosine * dz;
    const projectedCellHalfSize = (absCosine + absSine) * cellHalfSize;
    return Math.abs(localX) <= probeHalfX + projectedCellHalfSize + GRID_EPSILON && Math.abs(localZ) <= probeHalfZ + projectedCellHalfSize + GRID_EPSILON;
}

function probeNdfAtPoint(probe: PbrLocalEnvironmentProbe, point: readonly number[]): number {
    const dx = point[0]! - probe.influencePosition[0];
    const dy = point[1]! - probe.influencePosition[1];
    const dz = point[2]! - probe.influencePosition[2];
    const angle = probe.angleRadians ?? 0;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const local = [cosine * dx - sine * dz, dy, sine * dx + cosine * dz];
    let ndf = Number.NEGATIVE_INFINITY;
    for (let axis = 0; axis < 3; axis++) {
        const inner = probe.influenceInnerSize[axis]! * 0.5;
        const outer = probe.influenceOuterSize[axis]! * 0.5;
        ndf = Math.max(ndf, (Math.abs(local[axis]!) - inner) / Math.max(outer - inner, 0.00001));
    }
    return ndf;
}

function buildProbeGrid(probes: readonly PbrLocalEnvironmentProbe[], options: PbrLocalEnvironmentProbeGridOptions): BuiltProbeGrid {
    validateGrid(options);
    const minimum: [number, number, number] = [...options.minimum];
    const dimensions: [number, number, number] = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
        dimensions[axis] = Math.ceil((options.maximum[axis]! - minimum[axis]!) / options.cellSize);
    }
    const cellCount = dimensions[0] * dimensions[1] * dimensions[2];
    if (!Number.isSafeInteger(cellCount) || cellCount < 1) {
        throw new Error("[babylon-lite] local probe voxel grid dimensions are too large");
    }

    const cells = Array.from({ length: cellCount }, () => [] as number[]);
    const cellHalfSize = options.cellSize * 0.5;
    const linearIndex = (x: number, y: number, z: number): number => (z * dimensions[1] + y) * dimensions[0] + x;
    for (let probeIndex = 0; probeIndex < probes.length; probeIndex++) {
        const probe = probes[probeIndex]!;
        const extent = probeOuterWorldExtent(probe);
        const starts: [number, number, number] = [0, 0, 0];
        const ends: [number, number, number] = [0, 0, 0];
        for (let axis = 0; axis < 3; axis++) {
            starts[axis] = Math.max(0, Math.floor((probe.influencePosition[axis]! - extent[axis]! - minimum[axis]!) / options.cellSize));
            ends[axis] = Math.min(dimensions[axis]! - 1, Math.floor((probe.influencePosition[axis]! + extent[axis]! - minimum[axis]!) / options.cellSize));
        }
        for (let z = starts[2]; z <= ends[2]; z++) {
            for (let y = starts[1]; y <= ends[1]; y++) {
                for (let x = starts[0]; x <= ends[0]; x++) {
                    const centre = [minimum[0] + (x + 0.5) * options.cellSize, minimum[1] + (y + 0.5) * options.cellSize, minimum[2] + (z + 0.5) * options.cellSize];
                    if (!intersectsProbeOuterBox(probe, centre, cellHalfSize)) {
                        continue;
                    }
                    const cell = cells[linearIndex(x, y, z)]!;
                    cell.push(probeIndex);
                    if (cell.length > MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES) {
                        throw new Error(
                            `[babylon-lite] local probe voxel (${x}, ${y}, ${z}) intersects ${cell.length} probes, exceeding maxCandidates ${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}`
                        );
                    }
                }
            }
        }
    }

    for (let z = 0; z < dimensions[2]; z++) {
        for (let y = 0; y < dimensions[1]; y++) {
            for (let x = 0; x < dimensions[0]; x++) {
                const cell = cells[linearIndex(x, y, z)]!;
                if (cell.length) {
                    continue;
                }
                const centre = [minimum[0] + (x + 0.5) * options.cellSize, minimum[1] + (y + 0.5) * options.cellSize, minimum[2] + (z + 0.5) * options.cellSize];
                let nearestIndex = 0;
                let nearestNdf = Number.POSITIVE_INFINITY;
                for (let probeIndex = 0; probeIndex < probes.length; probeIndex++) {
                    const ndf = probeNdfAtPoint(probes[probeIndex]!, centre);
                    if (ndf < nearestNdf) {
                        nearestNdf = ndf;
                        nearestIndex = probeIndex;
                    }
                }
                cell.push(nearestIndex);
            }
        }
    }

    const stride = 1 + MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES;
    const data = new Uint32Array(GRID_HEADER_U32 + cellCount * stride);
    const floats = new Float32Array(data.buffer);
    floats.set(minimum, 0);
    floats[3] = 1 / options.cellSize;
    data.set([dimensions[0], dimensions[1], dimensions[2], stride], 4);
    for (let cellIndex = 0; cellIndex < cells.length; cellIndex++) {
        const cell = cells[cellIndex]!;
        const base = GRID_HEADER_U32 + cellIndex * stride;
        data[base] = cell.length;
        data.set(cell, base + 1);
    }
    return { data, minimum, cellSize: options.cellSize, dimensions, stride };
}

/** Resolve the voxel and probe indices used by a world-space point. */
export function getPbrLocalEnvironmentProbeGridCell(set: PbrLocalEnvironmentProbeSet, position: readonly [number, number, number]): PbrLocalEnvironmentProbeGridCell {
    finiteVec3(position, "local probe grid query position");
    const coordinates: [number, number, number] = [0, 0, 0];
    let outside = false;
    for (let axis = 0; axis < 3; axis++) {
        const raw = Math.floor((position[axis]! - set._gridMinimum[axis]!) / set._gridCellSize);
        outside ||= raw < 0 || raw >= set._gridDimensions[axis]!;
        coordinates[axis] = Math.max(0, Math.min(set._gridDimensions[axis]! - 1, raw));
    }
    const cellIndex = (coordinates[2] * set._gridDimensions[1] + coordinates[1]) * set._gridDimensions[0] + coordinates[0];
    const base = GRID_HEADER_U32 + cellIndex * set._gridStride;
    const count = set._gridData[base]!;
    return {
        coordinates,
        probeIndices: Array.from(set._gridData.subarray(base + 1, base + 1 + count)),
        outside,
    };
}

/** Create a scene-owned local-probe texture array and shared uniform buffer. */
export function createPbrLocalEnvironmentProbeSet(scene: SceneContext, options: PbrLocalEnvironmentProbeSetOptions): PbrLocalEnvironmentProbeSet {
    const probes = options.probes.slice();
    if (!probes.length || probes.length > MAX_PBR_LOCAL_ENVIRONMENT_PROBES) {
        throw new Error(`[babylon-lite] local probe sets require 1..${MAX_PBR_LOCAL_ENVIRONMENT_PROBES} probes`);
    }
    probes.forEach(validateProbe);

    const textures = probes.map((probe) => probe.environment._specularCube);
    const targetSize = Math.min(...textures.map((texture) => texture.width));
    const format = textures[0]!.format;
    const sourceMipOffsets = textures.map((texture) => {
        if (texture.width !== texture.height || texture.depthOrArrayLayers !== 6 || texture.format !== format) {
            throw new Error("[babylon-lite] local probe cubemaps must be square six-face textures with one shared format");
        }
        return powerOfTwoRatio(texture.width, targetSize);
    });
    const mipLevelCount = Math.min(...textures.map((texture, index) => texture.mipLevelCount - sourceMipOffsets[index]!));
    if (mipLevelCount < 1) {
        throw new Error("[babylon-lite] local probe cubemaps have no common mip range");
    }

    const engine = scene.surface.engine;
    const device = engine._device;
    const maxTextureProbes = Math.floor(device.limits.maxTextureArrayLayers / 6);
    if (probes.length > maxTextureProbes) {
        throw new Error(`[babylon-lite] local probe set has ${probes.length} probes, but this device supports at most ${maxTextureProbes} cube-array probes`);
    }
    const uniformBytes = _PBR_LOCAL_ENVIRONMENT_UNIFORM_FLOATS * 4;
    if (device.limits.maxUniformBufferBindingSize < uniformBytes) {
        throw new Error(`[babylon-lite] local probe UBO requires ${uniformBytes} bytes, but this device supports ${device.limits.maxUniformBufferBindingSize}`);
    }
    const grid = buildProbeGrid(probes, options.voxelGrid);
    if (grid.data.byteLength > device.limits.maxStorageBufferBindingSize || grid.data.byteLength > device.limits.maxBufferSize) {
        throw new Error(`[babylon-lite] local probe voxel grid requires ${grid.data.byteLength} bytes, exceeding this device's storage-buffer limits`);
    }
    const texture = device.createTexture({
        label: "pbr-local-environment-probes",
        size: [targetSize, targetSize, probes.length * 6],
        format,
        dimension: "2d",
        mipLevelCount,
        usage: TU.TEXTURE_BINDING | TU.COPY_DST,
    });
    const encoder = device.createCommandEncoder({ label: "pbr-local-environment-probe-copy" });
    for (let probeIndex = 0; probeIndex < probes.length; probeIndex++) {
        const source = textures[probeIndex]!;
        const sourceMipOffset = sourceMipOffsets[probeIndex]!;
        for (let mip = 0; mip < mipLevelCount; mip++) {
            const size = Math.max(1, targetSize >> mip);
            for (let face = 0; face < 6; face++) {
                encoder.copyTextureToTexture(
                    { texture: source, mipLevel: mip + sourceMipOffset, origin: [0, 0, face] },
                    { texture, mipLevel: mip, origin: [0, 0, probeIndex * 6 + face] },
                    [size, size, 1]
                );
            }
        }
    }
    device.queue.submit([encoder.finish()]);

    const uniformData = new Float32Array(_PBR_LOCAL_ENVIRONMENT_UNIFORM_FLOATS);
    const uniformU32 = new Uint32Array(uniformData.buffer);
    uniformU32[0] = probes.length;
    uniformU32[3] = options.parallaxCorrection === false ? 0 : _PBR_LOCAL_ENVIRONMENT_PARALLAX_FLAG;
    for (let index = 0; index < probes.length; index++) {
        writeProbe(uniformData, uniformU32, probes[index]!, index, sourceMipOffsets[index]!);
    }

    const set: PbrLocalEnvironmentProbeSet = {
        probes,
        _uniformBuffer: null as unknown as GPUBuffer,
        _uniformData: uniformData,
        _uniformU32: uniformU32,
        _texture: texture,
        _textureView: texture.createView({ dimension: "cube-array", baseArrayLayer: 0, arrayLayerCount: probes.length * 6 }),
        _sampler: probes[0]!.environment._cubeSampler,
        _gridBuffer: createMappedBuffer(engine, grid.data, BU.STORAGE, "pbr-local-environment-probe-grid"),
        _gridData: grid.data,
        _gridMinimum: grid.minimum,
        _gridCellSize: grid.cellSize,
        _gridDimensions: grid.dimensions,
        _gridStride: grid.stride,
        _device: device,
    };
    (set as { _uniformBuffer: GPUBuffer })._uniformBuffer = createUniformBuffer(engine, uniformData, "pbr-local-environment-probes");
    scene._disposables.push(() => {
        set._uniformBuffer.destroy();
        set._gridBuffer.destroy();
        set._texture.destroy();
    });
    return set;
}

/** Replace local-probe PBR output with the probes' per-fragment blended debug colors. */
export function setPbrLocalEnvironmentProbeDebug(set: PbrLocalEnvironmentProbeSet, enabled: boolean): void {
    const previous = set._uniformU32[3]!;
    const next = enabled ? previous | _PBR_LOCAL_ENVIRONMENT_DEBUG_COLOR_FLAG : previous & ~_PBR_LOCAL_ENVIRONMENT_DEBUG_COLOR_FLAG;
    if (next === previous) {
        return;
    }
    set._uniformU32[3] = next;
    const byteOffset = 3 * Uint32Array.BYTES_PER_ELEMENT;
    set._device.queue.writeBuffer(set._uniformBuffer, byteOffset, set._uniformData.buffer, set._uniformData.byteOffset + byteOffset, Uint32Array.BYTES_PER_ELEMENT);
}

let _enabled: Promise<void> | null = null;

/** Enable bounded single-probe projection and initialize fragment-weighted probe arrays. */
export function enablePbrLocalCubemap(options: PbrLocalCubemapInitOptions = {}): Promise<void> {
    _initializePbrLocalCubemapLimits(options.maxCandidates);
    return (_enabled ??= import("./fragments/local-cubemap-fragment.js").then((mod) => {
        mod.registerPbrLocalCubemapExt(_registerPbrExt);
    }));
}
