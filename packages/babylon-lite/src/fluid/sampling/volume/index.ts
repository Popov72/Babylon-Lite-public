// Public API for mesh -> particle volume sampling (SPlisHSPlasH VolumeSampling port).
//
// Given an indexed triangle mesh, fills its interior with particles. The deterministic lattice
// modes (regular / almostDense / dense) sample once and are `done` immediately. The SPH-relaxed
// blue-noise modes (jiang2015 / kugelstadt2021) build the SDF, overseed a mode-1 seed at
// 0.95*radius, then run a resumable relaxation: each `step()` advances one SPH iteration, mutating
// the live `positions` in place, until `stepIndex` reaches `totalSteps`.
//
// Pipeline (see spec sections A/B/F):
//   1. copy positions, apply `scale`
//   2. mesh AABB, clamped to `region` if given
//   3. domain = [bbMin - 4*radius, bbMax + 4*radius]  (4-radius padding per side)
//   4. exact mesh distance -> inverted trilinear SDF grid (inside > 0)
//   5. lattice sample (modes 0/1/2) OR overseed mode-1 lattice at 0.95*radius (modes 3/4)
//   6. modes 3/4 only: relax the seed with the matching SPH stepper (see sph-relax.ts)
//
// Pure CPU — no GPU handles, zero module-level side effects.

import type { SignedDistanceGrid } from "./mesh-sdf.js";
import { buildMeshDistance, buildSignedDistanceGrid } from "./mesh-sdf.js";
import { sampleLattice } from "./lattice.js";
import type { SphRelaxer } from "./sph-relax.js";
import { createJiang2015Relaxer, createKugelstadtRelaxer } from "./sph-relax.js";

/** Sampling strategy. Deterministic lattices: `regular`/`almostDense`/`dense`. SPH-relaxed: `jiang2015`/`kugelstadt2021` (Phase 2). */
export type VolumeSamplingMode = "regular" | "almostDense" | "dense" | "jiang2015" | "kugelstadt2021";

/** Options for {@link createVolumeSampler} / {@link sampleMeshVolume} (mirrors the CLI tool). */
export interface VolumeSamplingOptions {
    /** Flat xyz vertex positions of the source mesh. */
    positions: Float32Array;
    /** Flat triangle indices (three per triangle). */
    indices: Uint32Array | Uint16Array;
    /** Particle radius (diameter = 2*radius). Default 0.025. */
    radius?: number;
    /** Sampling mode. Default "kugelstadt2021". */
    mode?: VolumeSamplingMode;
    /** Per-axis scale applied to a copy of the mesh before sampling. Default [1,1,1]. */
    scale?: [number, number, number];
    /** SDF grid cell resolution per axis. Default [30,30,30]. */
    resolution?: [number, number, number];
    /** World-space clamp box [minX,minY,minZ, maxX,maxY,maxZ] applied to the bounding box. */
    region?: [number, number, number, number, number, number];
    /** Invert inside/outside (store +signedDistance). Default false. */
    invert?: boolean;
    /** SPH relaxation step count (modes 3/4). Default 100. */
    steps?: number;
    /** SPH CFL factor (mode 4). Default 0.25. */
    cflFactor?: number;
    /** SPH cohesion coefficient. Defaults per mode (4 for mode 4, 20 for mode 3). */
    cohesion?: number;
    /** SPH adhesion coefficient (mode 4). Default 2. */
    adhesion?: number;
    /** WCSPH stiffness (mode 3). Default 10000. */
    stiffness?: number;
    /** SPH time step (mode 3). Default 0.0001. */
    dt?: number;
    /** SPH rest density. Default 1000. */
    density0?: number;
    /** Optional progress callback in [0,1]. */
    onProgress?: (frac: number) => void;
}

/** Result of a completed {@link sampleMeshVolume}. */
export interface VolumeSamplingResult {
    /** Flat xyz particle positions. */
    positions: Float32Array;
    /** Number of particles (positions.length / 3). */
    count: number;
    /** Particle radius used. */
    radius: number;
    /** The signed-distance grid built for the mesh. */
    sdf: SignedDistanceGrid;
    /** Sampled bounding box (mesh AABB, clamped to `region`). */
    bounds: { min: [number, number, number]; max: [number, number, number] };
}

/** A resumable volume sampler. Lattice modes are `done` immediately; SPH modes step to relax. */
export interface VolumeSampler {
    /** Live flat xyz particle positions (mutated in place as `step()` relaxes SPH modes). */
    readonly positions: Float32Array;
    /** Current particle count (constant — relaxation never adds/removes particles). */
    readonly count: number;
    /** Particle radius used. */
    readonly radius: number;
    /** Steps completed so far. */
    readonly stepIndex: number;
    /** Total steps to run (0 for lattice modes). */
    readonly totalSteps: number;
    /** True once sampling is complete. */
    readonly done: boolean;
    /** The signed-distance grid built for the mesh. */
    readonly sdf: SignedDistanceGrid;
    /** Advance one step. No-op for lattice modes / once `done`; one SPH iteration otherwise. */
    step(): void;
}

const DEFAULT_RADIUS = 0.025;
const DEFAULT_RESOLUTION: [number, number, number] = [30, 30, 30];
const DEFAULT_STEPS = 100;
const DEFAULT_CFL_FACTOR = 0.25;
const DEFAULT_ADHESION = 2;
const DEFAULT_STIFFNESS = 10000;
const DEFAULT_DT = 0.0001;
const DEFAULT_DENSITY0 = 1000;
const DOMAIN_PADDING_RADII = 4;

const MODE_NUMBER: Record<VolumeSamplingMode, number> = {
    regular: 0,
    almostDense: 1,
    dense: 2,
    jiang2015: 3,
    kugelstadt2021: 4,
};

interface PreparedSampling {
    radius: number;
    modeNum: number;
    totalSteps: number;
    sdf: SignedDistanceGrid;
    bbMin: [number, number, number];
    bbMax: [number, number, number];
}

// Shared setup: scale the mesh, compute the (region-clamped) bounding box, build the SDF grid.
function prepareSampling(opts: VolumeSamplingOptions): PreparedSampling {
    const radius = opts.radius ?? DEFAULT_RADIUS;
    const mode = opts.mode ?? "kugelstadt2021";
    const modeNum = MODE_NUMBER[mode];
    const scale = opts.scale ?? [1, 1, 1];
    const resolution = opts.resolution ?? DEFAULT_RESOLUTION;
    const invert = opts.invert ?? false;

    // Copy positions and apply the per-axis scale (never mutate the caller's buffer).
    const src = opts.positions;
    const scaled = new Float32Array(src.length);
    for (let i = 0; i < src.length; i += 3) {
        scaled[i] = src[i]! * scale[0];
        scaled[i + 1] = src[i + 1]! * scale[1];
        scaled[i + 2] = src[i + 2]! * scale[2];
    }

    // Mesh AABB from the scaled positions.
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < scaled.length; i += 3) {
        minX = Math.min(minX, scaled[i]!);
        minY = Math.min(minY, scaled[i + 1]!);
        minZ = Math.min(minZ, scaled[i + 2]!);
        maxX = Math.max(maxX, scaled[i]!);
        maxY = Math.max(maxY, scaled[i + 1]!);
        maxZ = Math.max(maxZ, scaled[i + 2]!);
    }

    // Region clamp (spec F.1): bbMin = max(scale*regionMin, meshMin); bbMax = min(scale*regionMax, meshMax).
    const region = opts.region;
    if (region) {
        minX = Math.max(scale[0] * region[0], minX);
        minY = Math.max(scale[1] * region[1], minY);
        minZ = Math.max(scale[2] * region[2], minZ);
        maxX = Math.min(scale[0] * region[3], maxX);
        maxY = Math.min(scale[1] * region[4], maxY);
        maxZ = Math.min(scale[2] * region[5], maxZ);
    }

    const bbMin: [number, number, number] = [minX, minY, minZ];
    const bbMax: [number, number, number] = [maxX, maxY, maxZ];

    // Domain padded by 4 radii per side (spec A.1).
    const pad = radius * DOMAIN_PADDING_RADII;
    const domainMin: [number, number, number] = [minX - pad, minY - pad, minZ - pad];
    const domainMax: [number, number, number] = [maxX + pad, maxY + pad, maxZ + pad];

    const meshDist = buildMeshDistance(scaled, opts.indices);
    const sdf = buildSignedDistanceGrid(meshDist, domainMin, domainMax, resolution, invert);

    return {
        radius,
        modeNum,
        totalSteps: modeNum <= 2 ? 0 : (opts.steps ?? DEFAULT_STEPS),
        sdf,
        bbMin,
        bbMax,
    };
}

/**
 * Build a resumable volume sampler. Lattice modes (0/1/2) sample immediately and are `done`;
 * SPH modes (3/4) produce the overseeded mode-1 seed and expose `step()` to relax it in place.
 *
 * @param opts - sampling options.
 * @returns a {@link VolumeSampler}.
 */
export function createVolumeSampler(opts: VolumeSamplingOptions): VolumeSampler {
    const prep = prepareSampling(opts);
    const { radius, modeNum, totalSteps, sdf, bbMin, bbMax } = prep;

    let positions: Float32Array;
    let relaxer: SphRelaxer | null = null;
    if (modeNum <= 2) {
        // Deterministic lattice — sampled once, complete immediately.
        positions = Float32Array.from(sampleLattice(sdf, bbMin, bbMax, radius, modeNum));
    } else {
        // SPH seed: overseeded mode-1 lattice at 0.95*radius (spec D.2 / E.4).
        positions = Float32Array.from(sampleLattice(sdf, bbMin, bbMax, 0.95 * radius, 1));
        const count = positions.length / 3;
        const density0 = opts.density0 ?? DEFAULT_DENSITY0;
        const cohesion = opts.cohesion ?? (modeNum === 3 ? 20 : 4);
        if (modeNum === 4) {
            relaxer = createKugelstadtRelaxer(positions, count, radius, sdf, {
                density0,
                cohesion,
                adhesion: opts.adhesion ?? DEFAULT_ADHESION,
                cflFactor: opts.cflFactor ?? DEFAULT_CFL_FACTOR,
            });
        } else {
            relaxer = createJiang2015Relaxer(positions, count, radius, sdf, {
                density0,
                cohesion,
                stiffness: opts.stiffness ?? DEFAULT_STIFFNESS,
                dt: opts.dt ?? DEFAULT_DT,
            });
        }
    }

    // stepIndex / done are live: getters expose the mutable backing while satisfying `readonly`.
    let stepIndex = 0;
    let done = relaxer === null || totalSteps <= 0;

    const sampler: VolumeSampler = {
        positions,
        count: positions.length / 3,
        radius,
        get stepIndex() {
            return stepIndex;
        },
        totalSteps,
        get done() {
            return done;
        },
        sdf,
        step(): void {
            if (done || relaxer === null) {
                return;
            }
            relaxer.step();
            stepIndex++;
            if (stepIndex >= totalSteps) {
                done = true;
            }
        },
    };
    return sampler;
}

/**
 * Convenience: build a sampler and run it to completion, returning the particle set. For SPH
 * modes this runs the full relaxation (`steps` iterations), reporting progress via `onProgress`.
 *
 * @param opts - sampling options.
 * @returns a {@link VolumeSamplingResult}.
 */
export function sampleMeshVolume(opts: VolumeSamplingOptions): VolumeSamplingResult {
    const sampler = createVolumeSampler(opts);
    while (!sampler.done) {
        sampler.step();
        opts.onProgress?.(sampler.totalSteps > 0 ? sampler.stepIndex / sampler.totalSteps : 1);
    }
    opts.onProgress?.(1);

    // Recover the (region-clamped) mesh AABB from the padded SDF domain.
    const pad = sampler.radius * DOMAIN_PADDING_RADII;
    const dMin = sampler.sdf.domainMin;
    const dMax = sampler.sdf.domainMax;
    return {
        positions: sampler.positions,
        count: sampler.count,
        radius: sampler.radius,
        sdf: sampler.sdf,
        bounds: {
            min: [dMin[0] + pad, dMin[1] + pad, dMin[2] + pad],
            max: [dMax[0] - pad, dMax[1] - pad, dMax[2] - pad],
        },
    };
}

export type { SignedDistanceGrid, MeshDistance, SignedDistanceResult, GradientResult, DistanceNormalResult } from "./mesh-sdf.js";
export { buildMeshDistance, buildSignedDistanceGrid } from "./mesh-sdf.js";
export { sampleLattice } from "./lattice.js";
export type { MeshSdfGrid, MeshSdfOptions } from "./sdf-gen.js";
export { generateMeshSdf } from "./sdf-gen.js";
