import type { EngineContext } from "../../engine/engine.js";
import { mat4Invert } from "../../math/mat4-invert.js";
import type { Mat4 } from "../../math/types.js";
import { createFluidSceneSdf, updateFluidSceneSdf } from "./fluid-facade.js";
import type { FluidSceneSdf, FluidSceneSdfBounds } from "./fluid-facade.js";

export interface FluidDeformingSceneSdfOptions {
    readonly dims: readonly [number, number, number];
    readonly origin: readonly [number, number, number];
    readonly cellSize: number;
    /** One frame packed as IEEE-754 half floats, two values per uint32 word. */
    readonly initialFrame: Uint32Array;
    readonly trilinear?: boolean;
    readonly gridConfine?: boolean;
    readonly movingBoundaries?: boolean;
}

export interface FluidDeformingSceneSdf {
    readonly sceneSdf: FluidSceneSdf;
    /** @internal */
    readonly _params: Float32Array;
    /** @internal */
    readonly _atlas: Uint32Array;
    /** @internal */
    readonly _frameWords: number;
    /** @internal */
    readonly _trilinear: boolean;
    /** @internal */
    _firstFrameIndex: number;
    /** @internal */
    _secondFrameIndex: number;
    /** @internal */
    _currentWorldToLocal: Mat4;
}

export interface FluidDeformingSceneSdfUpdate {
    readonly firstFrameIndex: number;
    readonly secondFrameIndex: number;
    readonly firstFrame: Uint32Array;
    readonly secondFrame: Uint32Array;
    /** Interpolation factor from the first frame to the second frame. */
    readonly blend: number;
    /** Number of baked SDF frames advanced per real-time second. */
    readonly frameRate: number;
    readonly localToWorld: Mat4;
    readonly elapsedSeconds: number;
    readonly velocityScale?: number;
    readonly resetMotion?: boolean;
}

const PARAM_FLOATS = 56;

const STRUCT = /* wgsl */ `
struct SceneSdfParams {
    grid: vec4<f32>,
    dims: vec4<f32>,
    state: vec4<f32>,
    containerState: vec4<f32>,
    currentWorldToLocal: mat4x4<f32>,
    previousWorldToLocal: mat4x4<f32>,
    containerLo: vec4<f32>,
    containerHi: vec4<f32>,
};`;

const WGSL = /* wgsl */ `
fn deformingPackedLoad(base: u32, i: i32, j: i32, k: i32, dims: vec3<i32>) -> f32 {
    let c = clamp(vec3<i32>(i, j, k), vec3<i32>(0), dims - vec3<i32>(1));
    let halfIndex = base + u32(c.x + dims.x * (c.y + dims.y * c.z));
    let values = unpack2x16float(sceneSdfGrid[halfIndex >> 1u]);
    return select(values.x, values.y, (halfIndex & 1u) != 0u);
}
fn deformingNearest(base: u32, g: vec3<f32>, dims: vec3<i32>) -> f32 {
    let nearest = vec3<i32>(floor(g + 0.5));
    let center = deformingPackedLoad(base, nearest.x, nearest.y, nearest.z, dims);
    let gradient = 0.5 * vec3<f32>(
        deformingPackedLoad(base, nearest.x + 1, nearest.y, nearest.z, dims) - deformingPackedLoad(base, nearest.x - 1, nearest.y, nearest.z, dims),
        deformingPackedLoad(base, nearest.x, nearest.y + 1, nearest.z, dims) - deformingPackedLoad(base, nearest.x, nearest.y - 1, nearest.z, dims),
        deformingPackedLoad(base, nearest.x, nearest.y, nearest.z + 1, dims) - deformingPackedLoad(base, nearest.x, nearest.y, nearest.z - 1, dims)
    );
    return center + dot(g - vec3<f32>(nearest), gradient);
}
fn deformingSample(base: u32, g: vec3<f32>, dims: vec3<i32>, trilinear: bool) -> f32 {
    if (!trilinear) {
        return deformingNearest(base, g, dims);
    }
    let b = floor(g);
    let f = g - b;
    let i = i32(b.x); let j = i32(b.y); let k = i32(b.z);
    let c000 = deformingPackedLoad(base, i, j, k, dims); let c100 = deformingPackedLoad(base, i + 1, j, k, dims);
    let c010 = deformingPackedLoad(base, i, j + 1, k, dims); let c110 = deformingPackedLoad(base, i + 1, j + 1, k, dims);
    let c001 = deformingPackedLoad(base, i, j, k + 1, dims); let c101 = deformingPackedLoad(base, i + 1, j, k + 1, dims);
    let c011 = deformingPackedLoad(base, i, j + 1, k + 1, dims); let c111 = deformingPackedLoad(base, i + 1, j + 1, k + 1, dims);
    return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y), mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let transformAlpha = select(0.0, clamp(dt / sceneSdfParams.state.z, -1.0, 1.0), sceneSdfParams.state.z > 1.0e-6);
    let worldToLocal = sceneSdfParams.currentWorldToLocal + transformAlpha * (sceneSdfParams.currentWorldToLocal - sceneSdfParams.previousWorldToLocal);
    let localPoint = (worldToLocal * vec4<f32>(pt, 1.0)).xyz;
    let dims = vec3<i32>(sceneSdfParams.dims.xyz);
    let gridLo = sceneSdfParams.grid.xyz;
    let gridHi = gridLo + (vec3<f32>(dims) - 1.0) / sceneSdfParams.grid.w;
    let clampedPoint = clamp(localPoint, gridLo, gridHi);
    let g = (clampedPoint - gridLo) * sceneSdfParams.grid.w;
    let frameStride = u32(sceneSdfParams.dims.w);
    let trilinear = sceneSdfParams.state.w > 0.5;
    let first = deformingSample(0u, g, dims, trilinear);
    let second = deformingSample(frameStride, g, dims, trilinear);
    let frameBlend = clamp(sceneSdfParams.state.x + dt * sceneSdfParams.state.y, -1.0, 2.0);
    var localDistance = mix(first, second, frameBlend) / sceneSdfParams.grid.w;
    if (any(localPoint != clampedPoint)) {
        localDistance = max(localDistance, 0.0) + distance(localPoint, clampedPoint);
    }
    let inverseScale = max(
        length(vec3<f32>(worldToLocal[0].x, worldToLocal[1].x, worldToLocal[2].x)),
        max(
            length(vec3<f32>(worldToLocal[0].y, worldToLocal[1].y, worldToLocal[2].y)),
            length(vec3<f32>(worldToLocal[0].z, worldToLocal[1].z, worldToLocal[2].z))
        )
    );
    var worldDistance = localDistance / max(inverseScale, 1.0e-6);
    if (sceneSdfParams.containerState.x > 0.5) {
        let fromLo = pt - sceneSdfParams.containerLo.xyz;
        let fromHi = sceneSdfParams.containerHi.xyz - pt;
        let container = min(min(min(fromLo.x, fromLo.y), fromLo.z), min(min(fromHi.x, fromHi.y), fromHi.z));
        worldDistance = min(worldDistance, container);
    }
    return worldDistance;
}`;

function identityMatrix(): Mat4 {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) as unknown as Mat4;
}

function writeMatrix(target: Float32Array, offset: number, matrix: Mat4): void {
    for (let index = 0; index < 16; index++) {
        const value = matrix[index]!;
        if (!Number.isFinite(value)) {
            throw new RangeError("[fluid] deforming SDF transform matrices must be finite.");
        }
        target[offset + index] = value;
    }
}

export function createFluidDeformingSceneSdf(engine: EngineContext, options: FluidDeformingSceneSdfOptions): FluidDeformingSceneSdf {
    const [x, y, z] = options.dims;
    const voxelCount = x * y * z;
    const frameWords = Math.ceil(voxelCount / 2);
    if (
        !Number.isInteger(x) ||
        !Number.isInteger(y) ||
        !Number.isInteger(z) ||
        x < 2 ||
        y < 2 ||
        z < 2 ||
        !Number.isSafeInteger(voxelCount) ||
        options.initialFrame.length !== frameWords
    ) {
        throw new RangeError("[fluid] deforming SDF has invalid dimensions or frame data.");
    }
    if (!options.origin.every(Number.isFinite) || !Number.isFinite(options.cellSize) || options.cellSize <= 0) {
        throw new RangeError("[fluid] deforming SDF has an invalid grid transform.");
    }
    const atlas = new Uint32Array(frameWords * 2);
    atlas.set(options.initialFrame, 0);
    atlas.set(options.initialFrame, frameWords);
    const params = new Float32Array(PARAM_FLOATS);
    params.set([options.origin[0], options.origin[1], options.origin[2], 1 / options.cellSize], 0);
    params.set([x, y, z, frameWords * 2], 4);
    params.set([0, 0, 0, options.trilinear === false ? 0 : 1], 8);
    const identity = identityMatrix();
    writeMatrix(params, 16, identity);
    writeMatrix(params, 32, identity);
    return {
        sceneSdf: createFluidSceneSdf(engine, {
            struct: STRUCT,
            sdf: WGSL,
            params,
            sdfGrid: atlas,
            sdfGridFormat: "packed-f16",
            gridConfine: options.gridConfine,
            movingBoundaries: options.movingBoundaries,
        }),
        _params: params,
        _atlas: atlas,
        _frameWords: frameWords,
        _trilinear: options.trilinear !== false,
        _firstFrameIndex: -1,
        _secondFrameIndex: -1,
        _currentWorldToLocal: identity,
    };
}

export function updateFluidDeformingSceneSdf(deforming: FluidDeformingSceneSdf, update: FluidDeformingSceneSdfUpdate): void {
    if (update.firstFrame.length !== deforming._frameWords || update.secondFrame.length !== deforming._frameWords) {
        throw new RangeError("[fluid] deforming SDF update frame size does not match its grid.");
    }
    if (
        !Number.isInteger(update.firstFrameIndex) ||
        !Number.isInteger(update.secondFrameIndex) ||
        !Number.isFinite(update.blend) ||
        update.blend < 0 ||
        update.blend > 1 ||
        !Number.isFinite(update.frameRate) ||
        update.frameRate < 0 ||
        !Number.isFinite(update.elapsedSeconds) ||
        update.elapsedSeconds < 0
    ) {
        throw new RangeError("[fluid] deforming SDF update has invalid animation state.");
    }
    const velocityScale = update.velocityScale ?? 1;
    if (!Number.isFinite(velocityScale) || velocityScale < 0) {
        throw new RangeError("[fluid] deforming SDF velocity scale must be finite and non-negative.");
    }
    const inverse = mat4Invert(update.localToWorld);
    if (!inverse) {
        throw new RangeError("[fluid] deforming SDF has a singular transform.");
    }
    const framesChanged = update.firstFrameIndex !== deforming._firstFrameIndex || update.secondFrameIndex !== deforming._secondFrameIndex;
    if (framesChanged) {
        deforming._atlas.set(update.firstFrame, 0);
        deforming._atlas.set(update.secondFrame, deforming._frameWords);
        deforming._firstFrameIndex = update.firstFrameIndex;
        deforming._secondFrameIndex = update.secondFrameIndex;
    }
    const resetMotion = update.resetMotion === true || velocityScale === 0;
    deforming._params[8] = update.blend;
    deforming._params[9] = resetMotion ? 0 : update.frameRate * velocityScale;
    deforming._params[10] = resetMotion ? 0 : Math.min(update.elapsedSeconds / velocityScale, 0.25);
    deforming._params[11] = deforming._trilinear ? 1 : 0;
    writeMatrix(deforming._params, 16, inverse);
    writeMatrix(deforming._params, 32, resetMotion ? inverse : deforming._currentWorldToLocal);
    deforming._currentWorldToLocal = inverse;
    updateFluidSceneSdf(deforming.sceneSdf, deforming._params, framesChanged ? deforming._atlas : undefined);
}

export function updateFluidDeformingSceneSdfContainer(deforming: FluidDeformingSceneSdf, bounds: FluidSceneSdfBounds | null): void {
    if (bounds) {
        if (
            !bounds.min.every(Number.isFinite) ||
            !bounds.max.every(Number.isFinite) ||
            bounds.min[0] >= bounds.max[0] ||
            bounds.min[1] >= bounds.max[1] ||
            bounds.min[2] >= bounds.max[2]
        ) {
            throw new RangeError("[fluid] deforming SDF container must have finite, increasing bounds.");
        }
        deforming._params[12] = 1;
        deforming._params.set(bounds.min, 48);
        deforming._params.set(bounds.max, 52);
    } else {
        deforming._params[12] = 0;
    }
    updateFluidSceneSdf(deforming.sceneSdf, deforming._params);
}
