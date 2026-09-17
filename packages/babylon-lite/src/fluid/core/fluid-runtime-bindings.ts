import type { EngineContext } from "../../engine/engine.js";
import { mat4Invert } from "../../math/mat4-invert.js";
import type { Mat4 } from "../../math/types.js";
import type { ForceFieldSpec, SceneSdfSpec } from "./sim-common.js";

export const MAX_FLUID_LOCAL_SDFS = 16;

interface SceneSdfBindingOptions {
    readonly struct: string;
    readonly sdf: string;
    readonly params: Float32Array;
    readonly sdfGrid?: Float32Array | Uint32Array;
    readonly sdfGridFormat?: "f32" | "packed-f16";
    readonly gridConfine?: boolean;
    readonly movingBoundaries?: boolean;
}

interface ForceFieldBindingOptions {
    readonly struct: string;
    readonly wgsl: string;
    readonly params: Float32Array;
}

export interface SceneSdfRuntimeBinding {
    readonly spec: SceneSdfSpec;
    updateParams(params: Float32Array): void;
    updateSdfGrid(data: Float32Array | Uint32Array): void;
    updateTransforms?(updates: readonly CompositeSceneSdfTransformUpdate[], elapsedSeconds: number, resetMotion: boolean): void;
    updateStaticOffset?(offset: readonly [number, number, number], resetMotion: boolean): void;
    updateStaticScale?(scale: number, pivot: readonly [number, number, number], resetMotion: boolean): void;
    updateContainer?(bounds: CompositeSceneSdfBounds | null): void;
    updateGridSettings?(updates: readonly CompositeSceneSdfGridSettingsUpdate[]): void;
    dispose(): void;
}

export interface ForceFieldRuntimeBinding {
    readonly spec: ForceFieldSpec;
    updateParams(params: Float32Array): void;
    dispose(): void;
}

function alignedUniformSize(byteLength: number): number {
    return Math.max(16, Math.ceil(byteLength / 16) * 16);
}

export interface CompositeSceneSdfGridData {
    readonly dims: readonly [number, number, number];
    readonly origin: readonly [number, number, number];
    readonly cellSize: number;
    readonly distances: Float32Array;
    readonly enabled?: boolean;
    readonly trilinear?: boolean;
}

export interface CompositeSceneSdfLocalGridData extends CompositeSceneSdfGridData {
    readonly id: string;
}

export interface CompositeSceneSdfBounds {
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
}

export interface CompositeSceneSdfBindingOptions {
    readonly staticSdf?: CompositeSceneSdfGridData;
    readonly localSdfs?: readonly CompositeSceneSdfLocalGridData[];
    readonly container?: CompositeSceneSdfBounds;
    readonly gridConfine?: boolean;
    readonly movingBoundaries?: boolean;
}

export interface CompositeSceneSdfTransformUpdate {
    readonly id: string;
    readonly localToWorld: Mat4;
}

export interface CompositeSceneSdfGridSettingsUpdate {
    /** Omit for the static grid; use a local-grid ID otherwise. */
    readonly id?: string;
    readonly enabled?: boolean;
    readonly trilinear?: boolean;
}

const COMPOSITE_HEADER_FLOATS = 24;
const COMPOSITE_LOCAL_STRIDE = 44;
const COMPOSITE_PARAMS_FLOATS = COMPOSITE_HEADER_FLOATS + MAX_FLUID_LOCAL_SDFS * COMPOSITE_LOCAL_STRIDE;

const COMPOSITE_SCENE_SDF_STRUCT = /* wgsl */ `
struct LocalSdfParams {
    grid: vec4<f32>,
    dimsOffset: vec4<f32>,
    options: vec4<f32>,
    currentWorldToLocal: mat4x4<f32>,
    previousWorldToLocal: mat4x4<f32>,
};
struct SceneSdfParams {
    staticGrid: vec4<f32>,
    staticDimsOffset: vec4<f32>,
    state: vec4<f32>,
    staticOptions: vec4<f32>,
    containerLo: vec4<f32>,
    containerHi: vec4<f32>,
    locals: array<LocalSdfParams, ${MAX_FLUID_LOCAL_SDFS}>,
};`;

const COMPOSITE_SCENE_SDF_WGSL = /* wgsl */ `
fn packedSdfLoad(base: u32, i: i32, j: i32, k: i32, dims: vec3<i32>) -> f32 {
    let c = clamp(vec3<i32>(i, j, k), vec3<i32>(0), dims - vec3<i32>(1));
    let halfIndex = base + u32(c.x + dims.x * (c.y + dims.y * c.z));
    let values = unpack2x16float(sceneSdfGrid[halfIndex >> 1u]);
    return select(values.x, values.y, (halfIndex & 1u) != 0u);
}
fn sampleNearestSdfGrid(base: u32, g: vec3<f32>, dims: vec3<i32>) -> f32 {
    let nearest = vec3<i32>(floor(g + 0.5));
    let center = packedSdfLoad(base, nearest.x, nearest.y, nearest.z, dims);
    let gradient = 0.5 * vec3<f32>(
        packedSdfLoad(base, nearest.x + 1, nearest.y, nearest.z, dims) - packedSdfLoad(base, nearest.x - 1, nearest.y, nearest.z, dims),
        packedSdfLoad(base, nearest.x, nearest.y + 1, nearest.z, dims) - packedSdfLoad(base, nearest.x, nearest.y - 1, nearest.z, dims),
        packedSdfLoad(base, nearest.x, nearest.y, nearest.z + 1, dims) - packedSdfLoad(base, nearest.x, nearest.y, nearest.z - 1, dims)
    );
    return center + dot(g - vec3<f32>(nearest), gradient);
}
fn samplePackedSdfGrid(base: u32, pt: vec3<f32>, origin: vec3<f32>, invCell: f32, dims: vec3<i32>, trilinear: bool) -> f32 {
    let g = (pt - origin) * invCell;
    if (!trilinear) {
        return sampleNearestSdfGrid(base, g, dims) / invCell;
    }
    let b = floor(g);
    let f = g - b;
    let i = i32(b.x); let j = i32(b.y); let k = i32(b.z);
    let c000 = packedSdfLoad(base, i, j, k, dims); let c100 = packedSdfLoad(base, i + 1, j, k, dims);
    let c010 = packedSdfLoad(base, i, j + 1, k, dims); let c110 = packedSdfLoad(base, i + 1, j + 1, k, dims);
    let c001 = packedSdfLoad(base, i, j, k + 1, dims); let c101 = packedSdfLoad(base, i + 1, j, k + 1, dims);
    let c011 = packedSdfLoad(base, i, j + 1, k + 1, dims); let c111 = packedSdfLoad(base, i + 1, j + 1, k + 1, dims);
    let x00 = mix(c000, c100, f.x); let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x); let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z) / invCell;
}
fn localSdfWorldDistance(entry: LocalSdfParams, pt: vec3<f32>, dt: f32) -> f32 {
    let elapsed = sceneSdfParams.state.z;
    // Raw nearest-cell fields are not temporally differentiable: a small pose change can cross
    // one voxel and look like an enormous wall velocity. Keep nearest-mode motion discrete per
    // frame; collision still follows the current node pose, but moving-boundary velocity is zero.
    let alpha = select(0.0, clamp(dt / elapsed, -1.0, 1.0), entry.options.y > 0.5 && elapsed > 1.0e-6);
    let worldToLocal = entry.currentWorldToLocal + alpha * (entry.currentWorldToLocal - entry.previousWorldToLocal);
    let localPoint = (worldToLocal * vec4<f32>(pt, 1.0)).xyz;
    let dims = vec3<i32>(entry.dimsOffset.xyz);
    let gridLo = entry.grid.xyz;
    let gridHi = gridLo + (vec3<f32>(dims) - 1.0) / entry.grid.w;
    let clampedPoint = clamp(localPoint, gridLo, gridHi);
    let sampled = samplePackedSdfGrid(u32(entry.dimsOffset.w), clampedPoint, gridLo, entry.grid.w, dims, entry.options.y > 0.5);
    var localDistance = sampled;
    if (any(localPoint != clampedPoint)) {
        localDistance = max(sampled, 0.0) + distance(localPoint, clampedPoint);
    }
    let inverseScale = max(
        length(vec3<f32>(worldToLocal[0].x, worldToLocal[1].x, worldToLocal[2].x)),
        max(
            length(vec3<f32>(worldToLocal[0].y, worldToLocal[1].y, worldToLocal[2].y)),
            length(vec3<f32>(worldToLocal[0].z, worldToLocal[1].z, worldToLocal[2].z))
        )
    );
    return localDistance / max(inverseScale, 1.0e-6);
}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    var distanceToSolid = 1.0e30;
    if (sceneSdfParams.staticOptions.x > 0.5) {
        distanceToSolid = samplePackedSdfGrid(
            u32(sceneSdfParams.staticDimsOffset.w),
            pt,
            sceneSdfParams.staticGrid.xyz,
            sceneSdfParams.staticGrid.w,
            vec3<i32>(sceneSdfParams.staticDimsOffset.xyz),
            sceneSdfParams.staticOptions.y > 0.5
        );
    }
    for (var index = 0u; index < ${MAX_FLUID_LOCAL_SDFS}u; index++) {
        if (f32(index) >= sceneSdfParams.state.y) {
            break;
        }
        let entry = sceneSdfParams.locals[index];
        if (entry.options.x > 0.5) {
            distanceToSolid = min(distanceToSolid, localSdfWorldDistance(entry, pt, dt));
        }
    }
    if (sceneSdfParams.state.w > 0.5) {
        let fromLo = pt - sceneSdfParams.containerLo.xyz;
        let fromHi = sceneSdfParams.containerHi.xyz - pt;
        let container = min(min(min(fromLo.x, fromLo.y), fromLo.z), min(min(fromHi.x, fromHi.y), fromHi.z));
        distanceToSolid = min(distanceToSolid, container);
    }
    return distanceToSolid;
}`;

function validateGrid(grid: CompositeSceneSdfGridData, label: string): number {
    const [x, y, z] = grid.dims;
    const voxelCount = x * y * z;
    if (
        !Number.isInteger(x) ||
        !Number.isInteger(y) ||
        !Number.isInteger(z) ||
        x < 2 ||
        y < 2 ||
        z < 2 ||
        x > 2048 ||
        y > 2048 ||
        z > 2048 ||
        !Number.isSafeInteger(voxelCount) ||
        voxelCount !== grid.distances.length
    ) {
        throw new RangeError(`[fluid] ${label} has invalid dimensions or distance data.`);
    }
    if (!grid.origin.every(Number.isFinite) || !Number.isFinite(grid.cellSize) || grid.cellSize <= 0) {
        throw new RangeError(`[fluid] ${label} has an invalid transform.`);
    }
    return voxelCount;
}

function validateBounds(bounds: CompositeSceneSdfBounds, label: string): void {
    if (
        !bounds.min.every(Number.isFinite) ||
        !bounds.max.every(Number.isFinite) ||
        bounds.min[0] >= bounds.max[0] ||
        bounds.min[1] >= bounds.max[1] ||
        bounds.min[2] >= bounds.max[2]
    ) {
        throw new RangeError(`[fluid] ${label} must have finite, increasing bounds.`);
    }
}

function writeMatrix(target: Float32Array, offset: number, matrix: Mat4): void {
    for (let index = 0; index < 16; index++) {
        const value = matrix[index]!;
        if (!Number.isFinite(value)) {
            throw new RangeError("[fluid] scene SDF transform matrices must be finite.");
        }
        target[offset + index] = value;
    }
}

function identityMatrix(): Mat4 {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) as unknown as Mat4;
}

function float32ToFloat16(value: number, floatView: Float32Array, uintView: Uint32Array): number {
    floatView[0] = value;
    const bits = uintView[0]!;
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    const mantissa = bits & 0x7fffff;
    if (exponent === 0xff) {
        return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
    }
    const halfExponent = exponent - 127 + 15;
    if (halfExponent >= 0x1f) {
        return sign | 0x7bff;
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return sign;
        }
        const shifted = (mantissa | 0x800000) >>> (1 - halfExponent);
        return sign | ((shifted + 0x1000) >>> 13);
    }
    const rounded = mantissa + 0x1000;
    if ((rounded & 0x800000) !== 0) {
        const nextExponent = halfExponent + 1;
        return nextExponent >= 0x1f ? sign | 0x7bff : sign | (nextExponent << 10);
    }
    return sign | (halfExponent << 10) | (rounded >>> 13);
}

function packSdfGrid(atlas: Uint32Array, halfOffset: number, grid: CompositeSceneSdfGridData, floatView: Float32Array, uintView: Uint32Array): void {
    const inverseCellSize = 1 / grid.cellSize;
    for (let index = 0; index < grid.distances.length; index++) {
        const halfIndex = halfOffset + index;
        const half = float32ToFloat16(grid.distances[index]! * inverseCellSize, floatView, uintView);
        const wordIndex = halfIndex >>> 1;
        atlas[wordIndex] = (halfIndex & 1) === 0 ? (atlas[wordIndex]! & 0xffff0000) | half : (atlas[wordIndex]! & 0x0000ffff) | (half << 16);
    }
}

export function createCompositeSceneSdfRuntimeBinding(engine: EngineContext, options: CompositeSceneSdfBindingOptions): SceneSdfRuntimeBinding {
    const localSdfs = options.localSdfs ?? [];
    if (localSdfs.length > MAX_FLUID_LOCAL_SDFS) {
        throw new RangeError(`[fluid] composite scene SDF supports at most ${MAX_FLUID_LOCAL_SDFS} local grids.`);
    }
    if (options.container) {
        validateBounds(options.container, "scene SDF container");
    }

    const ids = new Map<string, number>();
    let totalVoxels = options.staticSdf ? validateGrid(options.staticSdf, "static scene SDF") : 0;
    for (let index = 0; index < localSdfs.length; index++) {
        const local = localSdfs[index]!;
        if (!local.id || ids.has(local.id)) {
            throw new TypeError("[fluid] local scene SDF identifiers must be non-empty and unique.");
        }
        ids.set(local.id, index);
        totalVoxels += validateGrid(local, `local scene SDF "${local.id}"`);
    }
    const atlasBytes = Math.ceil(totalVoxels / 2) * 4;
    const storageLimit = Math.min(engine._device.limits.maxStorageBufferBindingSize, engine._device.limits.maxBufferSize);
    if (atlasBytes <= 0 || atlasBytes > storageLimit) {
        throw new RangeError("[fluid] packed scene SDF atlas exceeds this device's storage-buffer limit.");
    }

    const atlas = new Uint32Array(Math.ceil(totalVoxels / 2));
    const floatView = new Float32Array(1);
    const uintView = new Uint32Array(floatView.buffer);
    const params = new Float32Array(COMPOSITE_PARAMS_FLOATS);
    const currentMatrices = localSdfs.map(() => identityMatrix());
    const previousMatrices = localSdfs.map(() => identityMatrix());
    const staticOrigin: [number, number, number] = options.staticSdf ? [...options.staticSdf.origin] : [0, 0, 0];
    let staticOffset: [number, number, number] = [0, 0, 0];
    let staticScale = 1;
    let staticScalePivot: [number, number, number] = [0, 0, 0];
    let atlasOffset = 0;
    if (options.staticSdf) {
        packSdfGrid(atlas, atlasOffset, options.staticSdf, floatView, uintView);
        params.set([staticOrigin[0], staticOrigin[1], staticOrigin[2], 1 / options.staticSdf.cellSize], 0);
        params.set([options.staticSdf.dims[0], options.staticSdf.dims[1], options.staticSdf.dims[2], atlasOffset], 4);
        atlasOffset += options.staticSdf.distances.length;
    }
    params.set([0, localSdfs.length, 0, options.container ? 1 : 0], 8);
    params.set([options.staticSdf?.enabled === false ? 0 : options.staticSdf ? 1 : 0, options.staticSdf?.trilinear === false ? 0 : 1, 0, 0], 12);
    if (options.container) {
        params.set(options.container.min, 16);
        params.set(options.container.max, 20);
    }
    for (let index = 0; index < localSdfs.length; index++) {
        const local = localSdfs[index]!;
        const offset = COMPOSITE_HEADER_FLOATS + index * COMPOSITE_LOCAL_STRIDE;
        packSdfGrid(atlas, atlasOffset, local, floatView, uintView);
        params.set([local.origin[0], local.origin[1], local.origin[2], 1 / local.cellSize], offset);
        params.set([local.dims[0], local.dims[1], local.dims[2], atlasOffset], offset + 4);
        params.set([local.enabled === false ? 0 : 1, local.trilinear === false ? 0 : 1, 0, 0], offset + 8);
        writeMatrix(params, offset + 12, currentMatrices[index]!);
        writeMatrix(params, offset + 28, previousMatrices[index]!);
        atlasOffset += local.distances.length;
    }

    const binding = createSceneSdfRuntimeBinding(engine, {
        struct: COMPOSITE_SCENE_SDF_STRUCT,
        sdf: COMPOSITE_SCENE_SDF_WGSL,
        params,
        sdfGrid: atlas,
        sdfGridFormat: "packed-f16",
        gridConfine: options.gridConfine,
        movingBoundaries: options.movingBoundaries,
    });
    const writeParams = (): void => binding.updateParams(params);
    const updateStaticGridTransform = (resetMotion: boolean): void => {
        params[0] = staticScalePivot[0] + (staticOrigin[0] - staticScalePivot[0]) * staticScale + staticOffset[0];
        params[1] = staticScalePivot[1] + (staticOrigin[1] - staticScalePivot[1]) * staticScale + staticOffset[1];
        params[2] = staticScalePivot[2] + (staticOrigin[2] - staticScalePivot[2]) * staticScale + staticOffset[2];
        params[3] = options.staticSdf ? 1 / (options.staticSdf.cellSize * staticScale) : 0;
        if (resetMotion) {
            params[10] = 0;
        }
        writeParams();
    };
    binding.updateTransforms = (updates, elapsedSeconds, resetMotion): void => {
        if (!Number.isFinite(elapsedSeconds) || elapsedSeconds < 0) {
            throw new RangeError("[fluid] scene SDF transform elapsed time must be finite and non-negative.");
        }
        if (updates.length !== localSdfs.length) {
            throw new RangeError(`[fluid] scene SDF transform update requires all ${localSdfs.length} local grids.`);
        }
        const next = params.slice();
        const seen = new Set<number>();
        const inverses: Array<{ index: number; inverse: Mat4 }> = [];
        for (const update of updates) {
            const index = ids.get(update.id);
            if (index === undefined) {
                throw new Error(`[fluid] unknown local scene SDF "${update.id}".`);
            }
            if (seen.has(index)) {
                throw new Error(`[fluid] duplicate local scene SDF transform "${update.id}".`);
            }
            seen.add(index);
            const inverse = mat4Invert(update.localToWorld);
            if (!inverse) {
                throw new RangeError(`[fluid] local scene SDF "${update.id}" has a singular transform.`);
            }
            for (let component = 0; component < 16; component++) {
                if (!Number.isFinite(inverse[component])) {
                    throw new RangeError(`[fluid] local scene SDF "${update.id}" has a non-finite transform.`);
                }
            }
            inverses.push({ index, inverse });
        }
        next[10] = resetMotion ? 0 : Math.min(elapsedSeconds, 0.25);
        for (const { index, inverse } of inverses) {
            const offset = COMPOSITE_HEADER_FLOATS + index * COMPOSITE_LOCAL_STRIDE;
            const previous = resetMotion ? inverse : currentMatrices[index]!;
            writeMatrix(next, offset + 12, inverse);
            writeMatrix(next, offset + 28, previous);
        }
        params.set(next);
        for (const { index, inverse } of inverses) {
            previousMatrices[index] = resetMotion ? inverse : currentMatrices[index]!;
            currentMatrices[index] = inverse;
        }
        writeParams();
    };
    binding.updateStaticOffset = (offset, resetMotion): void => {
        if (!offset.every(Number.isFinite)) {
            throw new RangeError("[fluid] scene SDF static offset must be finite.");
        }
        staticOffset = [offset[0], offset[1], offset[2]];
        updateStaticGridTransform(resetMotion);
    };
    binding.updateStaticScale = (scale, pivot, resetMotion): void => {
        if (!Number.isFinite(scale) || scale <= 0) {
            throw new RangeError("[fluid] scene SDF static scale must be finite and positive.");
        }
        if (!pivot.every(Number.isFinite)) {
            throw new RangeError("[fluid] scene SDF static scale pivot must be finite.");
        }
        staticScale = scale;
        staticScalePivot = [pivot[0], pivot[1], pivot[2]];
        updateStaticGridTransform(resetMotion);
    };
    binding.updateContainer = (bounds): void => {
        if (bounds) {
            validateBounds(bounds, "scene SDF container");
            params.set(bounds.min, 16);
            params.set(bounds.max, 20);
            params[11] = 1;
        } else {
            params[11] = 0;
        }
        writeParams();
    };
    binding.updateGridSettings = (updates): void => {
        const next = params.slice();
        const seen = new Set<string>();
        for (const update of updates) {
            const key = update.id ?? "__static__";
            if (seen.has(key)) {
                throw new Error(`[fluid] duplicate scene SDF settings update "${key}".`);
            }
            seen.add(key);
            let offset: number;
            if (update.id === undefined) {
                if (!options.staticSdf) {
                    throw new Error("[fluid] this composite scene SDF has no static grid.");
                }
                offset = 12;
            } else {
                const index = ids.get(update.id);
                if (index === undefined) {
                    throw new Error(`[fluid] unknown local scene SDF "${update.id}".`);
                }
                offset = COMPOSITE_HEADER_FLOATS + index * COMPOSITE_LOCAL_STRIDE + 8;
            }
            if (update.enabled !== undefined) {
                next[offset] = update.enabled ? 1 : 0;
            }
            if (update.trilinear !== undefined) {
                next[offset + 1] = update.trilinear ? 1 : 0;
            }
        }
        params.set(next);
        writeParams();
    };
    return binding;
}

export function createSceneSdfRuntimeBinding(engine: EngineContext, options: SceneSdfBindingOptions): SceneSdfRuntimeBinding {
    const device = engine._device;
    const params = device.createBuffer({
        label: "fluid-scene-sdf-params",
        size: alignedUniformSize(options.params.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    let sdfGrid: GPUBuffer | undefined;
    try {
        device.queue.writeBuffer(params, 0, options.params);
        if (options.sdfGrid) {
            sdfGrid = device.createBuffer({
                label: "fluid-scene-sdf-grid",
                size: Math.max(4, options.sdfGrid.byteLength),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(sdfGrid, 0, options.sdfGrid);
        }
    } catch (error) {
        params.destroy();
        sdfGrid?.destroy();
        throw error;
    }

    const spec: SceneSdfSpec = {
        struct: options.struct,
        sdf: options.sdf,
        buffer: params,
        ...(sdfGrid ? { sdfGrid } : {}),
        ...(sdfGrid ? { sdfGridFormat: options.sdfGridFormat ?? "f32" } : {}),
        ...(options.gridConfine !== undefined ? { gridConfine: options.gridConfine } : {}),
        ...(options.movingBoundaries !== undefined ? { movingBoundaries: options.movingBoundaries } : {}),
    };
    return {
        spec,
        updateParams(data): void {
            if (data.byteLength > params.size) {
                throw new RangeError("[fluid] scene SDF parameters exceed the allocated binding size.");
            }
            device.queue.writeBuffer(params, 0, data);
        },
        updateSdfGrid(data): void {
            if (!sdfGrid) {
                throw new Error("[fluid] this scene SDF has no grid binding.");
            }
            if (data.byteLength > sdfGrid.size) {
                throw new RangeError("[fluid] scene SDF grid data exceeds the allocated binding size.");
            }
            device.queue.writeBuffer(sdfGrid, 0, data);
        },
        dispose(): void {
            params.destroy();
            sdfGrid?.destroy();
        },
    };
}

export function createForceFieldRuntimeBinding(engine: EngineContext, options: ForceFieldBindingOptions): ForceFieldRuntimeBinding {
    const device = engine._device;
    const params = device.createBuffer({
        label: "fluid-force-field-params",
        size: alignedUniformSize(options.params.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    try {
        device.queue.writeBuffer(params, 0, options.params);
    } catch (error) {
        params.destroy();
        throw error;
    }
    return {
        spec: {
            struct: options.struct,
            wgsl: options.wgsl,
            buffer: params,
        },
        updateParams(data): void {
            if (data.byteLength > params.size) {
                throw new RangeError("[fluid] force-field parameters exceed the allocated binding size.");
            }
            device.queue.writeBuffer(params, 0, data);
        },
        dispose(): void {
            params.destroy();
        },
    };
}
