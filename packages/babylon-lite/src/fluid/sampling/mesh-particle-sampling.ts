/**
 * Shared mesh-to-fluid particle sampling.
 *
 * Picks the right strategy for the mesh and returns a world-ready particle seed:
 *
 *   - **Volume** (`sampleMeshVolume`) for a SOLID mesh — a dense interior lattice.
 *   - **Shell** (area-weighted surface scatter) for a mesh that has no interior to fill.
 *
 * Why the choice has to be automatic: volume sampling is only defined for a mesh that encloses a
 * region. Most architectural kit geometry (the Aquanova ship is ~93% of this) is modelled as OPEN,
 * single-sided panels — a wall is two flat quads, not a slab. Such a mesh encloses nothing, and the
 * generalized winding number the volume sampler relies on is then unreliable in both directions:
 * the empty space beside an L-shaped corner wall reads |w| ~ 0.74 (accepted as "inside") while the
 * panel's own interior reads only ~0.39 (rejected), so the fill lands in mid-air instead of in the
 * wall. Scattering points across the surface gives a water shell of the right shape, which is what
 * these props should melt into.
 *
 * Two gates decide, in order:
 *   1. **Openness** — normalized vector-area magnitude, 0 for a watertight mesh. Anything clearly
 *      open has no interior to fill and goes straight to the shell path. This is a magnitude, so it
 *      is independent of triangle winding (the glTF loader mirrors the ship on X and flips winding).
 *   2. **Thickness** — even a closed mesh can be too thin to hold a particle. Treating each particle
 *      as occupying a `2r` cube, `V ≈ count·(2r)³`; a shell of surface area `A` has ~`A/2` of
 *      "footprint", so mean thickness `≈ V/(A/2)`, which in particle diameters is `8·count·r²/A`.
 *
 * Both paths sample at the SAME point spacing (see `FluidMeshParticleSamplingOptions.spacing`,
 * default one
 * particle diameter), so their counts are directly comparable: a solid 1 m crate at r = 0.02 yields
 * ~22k volume particles against ~3.7k for its skin. The shell used to scatter one point per `radius²`
 * — 4x denser per unit area than the lattice's own packing — which made the two look equivalent.
 *
 * Measured on the ship: a door slab scores openness 0.04 / thickness 3.26 (fill it), the L-shaped
 * corner wall scores openness 0.59 (shell it), and a closed-but-32mm-thin panel flips from volume
 * to shell exactly when the particle diameter grows past its thickness.
 */
import { sampleMeshVolume, type VolumeSamplingMode } from "./volume/index.js";
import { transferFluidMeshParticleUvs } from "./particle-uv-transfer.js";

/** Surface-scatter particle budget, so a huge shell still simulates in real time. */
export const FLUID_MESH_SURFACE_MAX_POINTS = 250000;

/**
 * Minimum mean fill thickness, in particle DIAMETERS, for a volume fill to be considered real.
 * Below this the mesh is a shell (a solid thinner than one particle cannot be volume-filled).
 */
export const FLUID_MESH_MIN_FILL_THICKNESS = 1;

/**
 * Largest fraction of its enclosing surface a mesh may be missing and still be volume-sampled.
 * Measured as normalized vector-area magnitude, which is 0 for a watertight mesh. The ship's door
 * slab scores 0.04 (a couple of missing edge quads — still a solid), while an L-shaped wall panel
 * scores 0.59 and a box with a whole face removed scores 0.20.
 */
export const FLUID_MESH_MAX_OPENNESS = 0.1;

/**
 * How {@link sampleFluidMeshParticles} chooses between the volume lattice and surface shell.
 *
 * - `auto` — the openness + thickness gates decide (and a caller's `surfaceOnly` hint is honoured).
 * - `volume` — always use the volume lattice, gates bypassed. A failure is reported and never
 *   converted to a shell result.
 * - `surface` — always scatter across the surface, even for a solid mesh.
 */
export type FluidMeshSamplingStrategy = "auto" | "volume" | "surface";

export type FluidMeshSamplingSelectionReason = "forced-volume" | "forced-surface" | "surface-only" | "open-mesh" | "thin-mesh" | "volume-sampled" | "volume-fallback";

export type FluidMeshSamplingWarningCode = "AUTO_VOLUME_FAILED" | "AUTO_VOLUME_EMPTY" | "AUTO_VOLUME_INVALID_RESULT";

export interface FluidMeshSamplingWarning {
    code: FluidMeshSamplingWarningCode;
    message: string;
}

export type FluidMeshSamplingErrorCode =
    | "INVALID_POSITIONS"
    | "INVALID_INDICES"
    | "INVALID_UVS"
    | "INVALID_TEXTURE_INDICES"
    | "INVALID_RADIUS"
    | "INVALID_MODE"
    | "INVALID_STRATEGY"
    | "INVALID_SPACING"
    | "DEGENERATE_MESH"
    | "ALLOCATION_FAILED"
    | "VOLUME_SAMPLING_FAILED"
    | "VOLUME_SAMPLER_INVALID_RESULT"
    | "VOLUME_SAMPLER_EMPTY"
    | "SURFACE_SAMPLING_FAILED"
    | "SURFACE_SAMPLER_EMPTY"
    | "UV_TRANSFER_FAILED"
    | "EMPTY_RESULT"
    | "UNKNOWN";

/** Serializable sampling failure information, including failures transported by a worker. */
export interface FluidMeshSamplingErrorInfo {
    code: FluidMeshSamplingErrorCode;
    message: string;
    requestedStrategy: FluidMeshSamplingStrategy;
}

/** Typed failure thrown when mesh sampling cannot produce a valid result. */
export class FluidMeshSamplingError extends Error {
    readonly code: FluidMeshSamplingErrorCode;
    readonly requestedStrategy: FluidMeshSamplingStrategy;

    constructor(info: FluidMeshSamplingErrorInfo, options?: ErrorOptions) {
        super(info.message, options);
        this.name = "FluidMeshSamplingError";
        this.code = info.code;
        this.requestedStrategy = info.requestedStrategy;
    }

    toJSON(): FluidMeshSamplingErrorInfo {
        return { code: this.code, message: this.message, requestedStrategy: this.requestedStrategy };
    }
}

/** Convert any thrown value to serializable mesh-sampling failure information. */
export function fluidMeshSamplingErrorInfo(error: unknown, requestedStrategy: FluidMeshSamplingStrategy = "auto"): FluidMeshSamplingErrorInfo {
    if (error instanceof FluidMeshSamplingError) {
        return error.toJSON();
    }
    return {
        code: "UNKNOWN",
        message: error instanceof Error ? error.message : String(error),
        requestedStrategy,
    };
}

export interface FluidMeshSamplingMetrics {
    /** Sum of triangle areas. */
    area: number;
    /** Magnitude of summed triangle vector areas divided by `area`. */
    openness: number;
    /** Estimated volume-fill thickness in particle diameters, when volume sampling was attempted. */
    thickness: number | null;
    /** Requested world-space point spacing shared by volume and surface paths. */
    spacing: number;
}

/** A particle seed produced by {@link sampleFluidMeshParticles}. */
export interface FluidMeshParticleSamplingResult {
    /** Flat xyz particle positions, in the same space as the supplied geometry. */
    positions: Float32Array;
    /** Particle count. */
    count: number;
    /** Particle radius: the requested radius for a volume fill, the ACTUAL point spacing for a shell. */
    radius: number;
    /** AABB of the produced points. */
    bounds: { min: [number, number, number]; max: [number, number, number] };
    /** Per-particle UVs (only when `uvs` was supplied). */
    uvs: Float32Array | null;
    /** Per-particle sub-mesh texture index (only when `texIndices` was supplied). */
    texIndices: Uint32Array | null;
    /** True when the surface-shell path produced this fill. */
    shell: boolean;
    /** Strategy that actually produced the particles. */
    strategy: "volume" | "surface";
    /** Why that strategy was selected. */
    selectionReason: FluidMeshSamplingSelectionReason;
    /** Explicit automatic-fallback diagnostics. Empty for normal policy selection. */
    warnings: FluidMeshSamplingWarning[];
    /** Geometry and policy values used to select the result. */
    metrics: FluidMeshSamplingMetrics;
}

export interface FluidMeshParticleSamplingOptions {
    positions: Float32Array;
    indices: Uint32Array;
    radius: number;
    mode: VolumeSamplingMode;
    /** Per-vertex UVs aligned with `positions`; enables per-particle UV output. */
    uvs?: Float32Array | null;
    /** Per-vertex sub-mesh texture index aligned with `positions`. */
    texIndices?: Uint32Array | null;
    /** Skip the volume attempt entirely (a prop already known to be hollow). Honoured by `auto` only. */
    surfaceOnly?: boolean;
    /** Override the automatic volume-vs-shell choice. Defaults to `auto`. */
    strategy?: FluidMeshSamplingStrategy;
    /**
     * Distance between adjacent sampled points, in units of the particle RADIUS. Applied identically
     * to both paths, so their counts are directly comparable:
     *
     *   - `2` (default) — one particle DIAMETER, i.e. neighbouring particles just touch. This is the
     *     convention the volume lattice already uses (it steps by `2 * radius`).
     *   - smaller — denser, overlapping particles; larger — sparser.
     *
     * The two paths used to disagree here: the lattice stepped by a diameter while the shell scattered
     * one point per `radius²`, making the shell 4x denser per unit area and hiding the fact that a
     * solid's interior holds far more particles than its skin.
     */
    spacing?: number;
}

/** Total surface area and openness of an indexed triangle mesh (single pass). */
function meshShellStats(positions: Float32Array, indices: Uint32Array): { area: number; openness: number } {
    let area = 0;
    let vx = 0;
    let vy = 0;
    let vz = 0;
    for (let t = 0; t + 2 < indices.length; t += 3) {
        const a = indices[t]! * 3;
        const b = indices[t + 1]! * 3;
        const c = indices[t + 2]! * 3;
        const e1x = positions[b]! - positions[a]!;
        const e1y = positions[b + 1]! - positions[a + 1]!;
        const e1z = positions[b + 2]! - positions[a + 2]!;
        const e2x = positions[c]! - positions[a]!;
        const e2y = positions[c + 1]! - positions[a + 1]!;
        const e2z = positions[c + 2]! - positions[a + 2]!;
        const nx = e1y * e2z - e1z * e2y;
        const ny = e1z * e2x - e1x * e2z;
        const nz = e1x * e2y - e1y * e2x;
        vx += 0.5 * nx;
        vy += 0.5 * ny;
        vz += 0.5 * nz;
        area += 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
    }
    // A CLOSED surface has exactly zero vector area, so the normalized magnitude measures roughly
    // what fraction of the enclosing surface is missing. Using the MAGNITUDE makes this independent
    // of triangle winding, which matters because the glTF loader mirrors the ship on X and flips
    // winding with it.
    return { area, openness: area > 0 ? Math.sqrt(vx * vx + vy * vy + vz * vz) / area : 1 };
}

/**
 * Area-weighted surface scatter with EXACT barycentric-interpolated UVs. Works for any mesh
 * (open, non-manifold, single-sided), producing a thin water shell of the mesh's shape.
 *
 * @param positions - flat xyz vertex positions.
 * @param indices - flat triangle indices.
 * @param uvs - per-vertex UVs aligned with `positions`, or null.
 * @param vertTex - per-vertex sub-mesh texture index aligned with `positions`, or null.
 * @param spacing - desired distance between adjacent points.
 * @returns the scattered points, their UVs, and the ACTUAL spacing achieved.
 */
export function sampleFluidMeshSurface(
    positions: Float32Array,
    indices: Uint32Array,
    uvs: Float32Array | null,
    vertTex: Uint32Array | null,
    spacing: number
): { positions: Float32Array; uvs: Float32Array | null; texIndices: Uint32Array | null; count: number; renderRadius: number } {
    const nTri = Math.floor(indices.length / 3);
    let totalArea = 0;
    const triArea = new Float32Array(nTri);
    for (let t = 0; t < nTri; t++) {
        const a = indices[t * 3]! * 3;
        const b = indices[t * 3 + 1]! * 3;
        const c = indices[t * 3 + 2]! * 3;
        const e1x = positions[b]! - positions[a]!;
        const e1y = positions[b + 1]! - positions[a + 1]!;
        const e1z = positions[b + 2]! - positions[a + 2]!;
        const e2x = positions[c]! - positions[a]!;
        const e2y = positions[c + 1]! - positions[a + 1]!;
        const e2z = positions[c + 2]! - positions[a + 2]!;
        const nx = e1y * e2z - e1z * e2y;
        const ny = e1z * e2x - e1x * e2z;
        const nz = e1x * e2y - e1y * e2x;
        const area = 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
        triArea[t] = area;
        totalArea += area;
    }
    if (totalArea <= 0) {
        return { positions: new Float32Array(0), uvs: uvs ? new Float32Array(0) : null, texIndices: vertTex ? new Uint32Array(0) : null, count: 0, renderRadius: spacing };
    }
    // One point per spacing² of surface area, capped so the sim stays real-time. The RENDER radius
    // returned below is the ACTUAL spacing achieved: a random scatter (unlike a lattice) needs splats
    // as wide as the spacing to close its gaps, so this keeps the surface hole-free even when the
    // count is capped and the true spacing exceeds `spacing`.
    const target = Math.min(FLUID_MESH_SURFACE_MAX_POINTS, Math.max(64, Math.round(totalArea / (spacing * spacing))));
    const density = target / totalArea;
    const outPos: number[] = [];
    const outUv: number[] | null = uvs ? [] : null;
    const outTex: number[] | null = vertTex ? [] : null;
    for (let t = 0; t < nTri; t++) {
        const expected = density * triArea[t]!;
        let n = Math.floor(expected);
        if (Math.random() < expected - n) {
            n++;
        }
        if (n === 0) {
            continue;
        }
        const ia = indices[t * 3]!;
        const ib = indices[t * 3 + 1]!;
        const ic = indices[t * 3 + 2]!;
        const ti = vertTex ? vertTex[ia]! : 0; // a triangle lies within one sub-mesh → one texture
        for (let k = 0; k < n; k++) {
            let u = Math.random();
            let v = Math.random();
            if (u + v > 1) {
                u = 1 - u;
                v = 1 - v;
            }
            const w = 1 - u - v;
            outPos.push(
                positions[ia * 3]! * w + positions[ib * 3]! * u + positions[ic * 3]! * v,
                positions[ia * 3 + 1]! * w + positions[ib * 3 + 1]! * u + positions[ic * 3 + 1]! * v,
                positions[ia * 3 + 2]! * w + positions[ib * 3 + 2]! * u + positions[ic * 3 + 2]! * v
            );
            if (outUv) {
                outUv.push(uvs![ia * 2]! * w + uvs![ib * 2]! * u + uvs![ic * 2]! * v, uvs![ia * 2 + 1]! * w + uvs![ib * 2 + 1]! * u + uvs![ic * 2 + 1]! * v);
            }
            if (outTex) {
                outTex.push(ti);
            }
        }
    }
    const count = outPos.length / 3;
    // Actual surface point spacing → the render splat radius, so splats cover the surface even when
    // `count` was capped (spacing > requested radius). Equals `radius` when uncapped.
    const renderRadius = count > 0 ? Math.sqrt(totalArea / count) : spacing;
    return { positions: new Float32Array(outPos), uvs: outUv ? new Float32Array(outUv) : null, texIndices: outTex ? new Uint32Array(outTex) : null, count, renderRadius };
}

function boundsOf(p: Float32Array): { min: [number, number, number]; max: [number, number, number] } {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) {
        for (let k = 0; k < 3; k++) {
            const v = p[i + k]!;
            if (v < min[k]!) {
                min[k] = v;
            }
            if (v > max[k]!) {
                max[k] = v;
            }
        }
    }
    if (!Number.isFinite(min[0])) {
        return { min: [0, 0, 0], max: [0, 0, 0] };
    }
    return { min, max };
}

/**
 * Sample a mesh into fluid particles, choosing a volume lattice or surface shell automatically.
 *
 * Automatic mode first applies the openness and thickness policy. If the volume sampler throws,
 * returns no particles, or returns malformed data, automatic mode may use the surface path and
 * records that fallback in `warnings` and `selectionReason`. Forced volume mode never falls back:
 * the same conditions throw {@link FluidMeshSamplingError}.
 *
 * @param opts - the geometry, particle radius and sampling mode.
 * @returns the particle seed, flagged with which strategy produced it.
 */
export function sampleFluidMeshParticles(opts: FluidMeshParticleSamplingOptions): FluidMeshParticleSamplingResult {
    const { positions, indices, radius, mode } = opts;
    const requestedStrategy = opts.strategy ?? "auto";
    validateSamplingInput(opts, requestedStrategy);
    const uvs = opts.uvs ?? null;
    const texIndices = opts.texIndices ?? null;
    // The volume sampler may read/scale `positions`; the UV search and the shell path both need the
    // ORIGINAL geometry, so keep an untouched copy.
    let srcPos: Float32Array;
    try {
        srcPos = positions.slice();
    } catch (error) {
        throw samplingError("ALLOCATION_FAILED", "Could not copy mesh positions for fluid sampling.", requestedStrategy, error);
    }
    const { area, openness } = meshShellStats(srcPos, indices);
    if (!(area > 0) || !Number.isFinite(area)) {
        throw samplingError("DEGENERATE_MESH", "Fluid mesh sampling requires at least one triangle with finite non-zero area.", requestedStrategy);
    }

    // Volume sampling needs a mesh that ENCLOSES something. The interior test is the generalized
    // winding number, and on an open shell that field is unreliable in BOTH directions: the empty
    // space beside an L-shaped wall panel reads |w| ~ 0.74 (accepted) while the panel's own interior
    // reads only ~0.39 (rejected), and which of those comes out positive depends on the triangle
    // winding. So an open mesh is sent straight to the shell path rather than being volume-sampled
    // and second-guessed afterwards — unless the caller explicitly asks to see it happen.
    const strategy = requestedStrategy;
    // Point spacing shared by BOTH paths (see `spacing`). The lattice steps by `2 * radius`, so it is
    // handed half the spacing; the shell takes the spacing directly.
    const spacing = Math.max(1e-4, (opts.spacing ?? 2) * radius);
    const warnings: FluidMeshSamplingWarning[] = [];
    let selectionReason: FluidMeshSamplingSelectionReason =
        strategy === "surface"
            ? "forced-surface"
            : opts.surfaceOnly && strategy === "auto"
              ? "surface-only"
              : openness > FLUID_MESH_MAX_OPENNESS && strategy === "auto"
                ? "open-mesh"
                : "volume-sampled";
    let thickness: number | null = null;
    const tryVolume = strategy === "volume" || (strategy === "auto" && !opts.surfaceOnly && openness <= FLUID_MESH_MAX_OPENNESS);
    if (tryVolume) {
        const vr = spacing / 2;
        let volume: ReturnType<typeof sampleMeshVolume> | null = null;
        let volumeFailure: FluidMeshSamplingWarning | null = null;
        try {
            volume = sampleMeshVolume({ positions, indices, radius: vr, mode });
        } catch (error) {
            if (strategy === "volume") {
                throw samplingError("VOLUME_SAMPLING_FAILED", `Forced volume sampling failed: ${errorMessage(error)}`, strategy, error);
            }
            volumeFailure = { code: "AUTO_VOLUME_FAILED", message: `Automatic volume sampling failed; used surface sampling instead: ${errorMessage(error)}` };
        }
        if (volume) {
            const invalidVolume = invalidVolumeResult(volume);
            if (invalidVolume) {
                if (strategy === "volume") {
                    throw samplingError("VOLUME_SAMPLER_INVALID_RESULT", `Forced volume sampler returned invalid data: ${invalidVolume}`, strategy);
                }
                volumeFailure = { code: "AUTO_VOLUME_INVALID_RESULT", message: `Automatic volume sampler returned invalid data; used surface sampling instead: ${invalidVolume}` };
                volume = null;
            } else if (volume.count === 0) {
                if (strategy === "volume") {
                    throw samplingError("VOLUME_SAMPLER_EMPTY", "Forced volume sampling produced no particles.", strategy);
                }
                volumeFailure = { code: "AUTO_VOLUME_EMPTY", message: "Automatic volume sampling produced no particles; used surface sampling instead." };
                volume = null;
            }
        }
        // Mean fill thickness in particle diameters (see the module header). A forced volume fill
        // bypasses this policy gate but still requires a valid non-empty sampler result.
        thickness = volume ? (8 * volume.count * vr * vr) / area : 0;
        if (volume && (strategy === "volume" || thickness >= FLUID_MESH_MIN_FILL_THICKNESS)) {
            let outUvs: Float32Array | null = null;
            let outTex: Uint32Array | null = null;
            if (uvs) {
                try {
                    const transferred = transferFluidMeshParticleUvs({
                        particles: volume.positions,
                        count: volume.count,
                        positions: srcPos,
                        indices,
                        uvs,
                        texIndices,
                    });
                    outUvs = transferred.uvs;
                    outTex = texIndices ? transferred.texIndices : null;
                } catch (error) {
                    throw samplingError("UV_TRANSFER_FAILED", `Closest-surface UV transfer failed: ${errorMessage(error)}`, strategy, error);
                }
            }
            return {
                positions: volume.positions,
                count: volume.count,
                radius: volume.radius,
                bounds: { min: [volume.bounds.min[0], volume.bounds.min[1], volume.bounds.min[2]], max: [volume.bounds.max[0], volume.bounds.max[1], volume.bounds.max[2]] },
                uvs: outUvs,
                texIndices: outTex,
                shell: false,
                strategy: "volume",
                selectionReason: strategy === "volume" ? "forced-volume" : "volume-sampled",
                warnings,
                metrics: { area, openness, thickness, spacing },
            };
        }
        if (volumeFailure) {
            warnings.push(volumeFailure);
            selectionReason = "volume-fallback";
        } else if (volume) {
            selectionReason = "thin-mesh";
        }
    }

    let surf: ReturnType<typeof sampleFluidMeshSurface>;
    try {
        surf = sampleFluidMeshSurface(srcPos, indices, uvs, texIndices, spacing);
    } catch (error) {
        throw samplingError("SURFACE_SAMPLING_FAILED", `Surface sampling failed: ${errorMessage(error)}`, strategy, error);
    }
    if (surf.count === 0) {
        throw samplingError("SURFACE_SAMPLER_EMPTY", "Surface sampling produced no particles.", strategy);
    }
    return {
        positions: surf.positions,
        count: surf.count,
        radius: surf.renderRadius,
        bounds: boundsOf(surf.positions),
        uvs: surf.uvs,
        texIndices: surf.texIndices,
        shell: true,
        strategy: "surface",
        selectionReason,
        warnings,
        metrics: { area, openness, thickness, spacing },
    };
}

function isVolumeMode(value: string): boolean {
    return value === "regular" || value === "almostDense" || value === "dense" || value === "jiang2015" || value === "kugelstadt2021";
}

function isSamplingStrategy(value: string): boolean {
    return value === "auto" || value === "volume" || value === "surface";
}

function validateSamplingInput(options: FluidMeshParticleSamplingOptions, strategy: FluidMeshSamplingStrategy): void {
    const { positions, indices, radius, mode } = options;
    if (positions.length < 9 || positions.length % 3 !== 0 || !allFinite(positions)) {
        throw samplingError("INVALID_POSITIONS", "Mesh positions must contain finite xyz values for at least three vertices.", strategy);
    }
    if (indices.length < 3 || indices.length % 3 !== 0) {
        throw samplingError("INVALID_INDICES", "Mesh indices must contain one or more complete triangles.", strategy);
    }
    const vertexCount = positions.length / 3;
    for (const index of indices) {
        if (index >= vertexCount) {
            throw samplingError("INVALID_INDICES", `Mesh index ${index} is outside the ${vertexCount}-vertex position buffer.`, strategy);
        }
    }
    if (options.uvs && (options.uvs.length !== vertexCount * 2 || !allFinite(options.uvs))) {
        throw samplingError("INVALID_UVS", "Mesh UVs must contain finite uv values for every vertex.", strategy);
    }
    if (options.texIndices && options.texIndices.length !== vertexCount) {
        throw samplingError("INVALID_TEXTURE_INDICES", "Mesh texture indices must contain one value for every vertex.", strategy);
    }
    if (!Number.isFinite(radius) || radius <= 0) {
        throw samplingError("INVALID_RADIUS", "Fluid particle radius must be finite and greater than zero.", strategy);
    }
    if (!isVolumeMode(mode)) {
        throw samplingError("INVALID_MODE", `Unknown volume sampling mode: ${String(mode)}.`, strategy);
    }
    if (!isSamplingStrategy(strategy)) {
        throw samplingError("INVALID_STRATEGY", `Unknown fluid mesh sampling strategy: ${String(strategy)}.`, strategy);
    }
    const spacing = options.spacing ?? 2;
    if (!Number.isFinite(spacing) || spacing <= 0) {
        throw samplingError("INVALID_SPACING", "Fluid mesh sampling spacing must be finite and greater than zero.", strategy);
    }
}

function allFinite(values: Float32Array): boolean {
    for (const value of values) {
        if (!Number.isFinite(value)) {
            return false;
        }
    }
    return true;
}

function invalidVolumeResult(volume: ReturnType<typeof sampleMeshVolume>): string | null {
    if (!Number.isInteger(volume.count) || volume.count < 0) {
        return "count is not a non-negative integer";
    }
    if (!(volume.positions instanceof Float32Array) || volume.positions.length !== volume.count * 3) {
        return "positions length does not match count";
    }
    if (!allFinite(volume.positions)) {
        return "positions contain a non-finite value";
    }
    if (!Number.isFinite(volume.radius) || volume.radius <= 0) {
        return "radius is not finite and positive";
    }
    const bounds = [...volume.bounds.min, ...volume.bounds.max];
    if (!bounds.every(Number.isFinite)) {
        return "bounds contain a non-finite value";
    }
    return null;
}

function samplingError(code: FluidMeshSamplingErrorCode, message: string, requestedStrategy: FluidMeshSamplingStrategy, cause?: unknown): FluidMeshSamplingError {
    return new FluidMeshSamplingError({ code, message, requestedStrategy }, cause === undefined ? undefined : { cause });
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}
