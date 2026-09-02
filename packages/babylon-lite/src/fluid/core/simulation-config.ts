import { fluidShapeVolume } from "./sim-common.js";
import type { FluidFlowConfig } from "./sim-common.js";
import { cellSizeForPhysicsScale, FLIP_DEFAULT_MARKERS_PER_CELL, gridCellsForSize, gridSizeForBounds } from "../authoring/grid-settings.js";
import { resolveFlipDiscretization } from "../solvers/flip-sim.js";
import type { FlipDiscretization } from "../solvers/flip-sim.js";

const FLUID_BASE_PARTICLE_RADIUS = 0.08;
const MPM_MIN_CELL_SIZE = 0.18;
const MPM_CELL_RADIUS_RATIO = 2.4;
const PBF_MIN_CELL_SIZE = 0.3;
const PBF_CELL_RADIUS_RATIO = 4;

export type FluidSimulationSamplingType = "fluid" | "mesh";

export interface FluidSimulationDiscretization {
    physicsParticleSize: number;
    samplingType: FluidSimulationSamplingType;
    particleRadius?: number;
}

export const fluidParticleRadiusForPhysicsScale = (physicsScale: number): number => FLUID_BASE_PARTICLE_RADIUS * physicsScale;

export const fluidCellSizeForParticleRadius = (method: string, particleRadius: number): number =>
    method === "PBF" ? Math.max(particleRadius * PBF_CELL_RADIUS_RATIO, PBF_MIN_CELL_SIZE) : Math.max(particleRadius * MPM_CELL_RADIUS_RATIO, MPM_MIN_CELL_SIZE);

export function fluidSimulationParticleRadius(config: FluidSimulationDiscretization): number {
    const authoredRadius = config.particleRadius;
    return config.samplingType === "fluid" || typeof authoredRadius !== "number" || !Number.isFinite(authoredRadius) || authoredRadius <= 0
        ? fluidParticleRadiusForPhysicsScale(config.physicsParticleSize)
        : authoredRadius;
}

export const fluidSimulationCellSize = (method: string, config: FluidSimulationDiscretization): number =>
    fluidCellSizeForParticleRadius(method, fluidSimulationParticleRadius(config));

export function fluidSimulationParticleCapacity(method: string, requestedCount: number, flow: FluidFlowConfig, particleVolume: number): number {
    const emitters = flow.emitters.filter((emitter) => emitter.enabled);
    const effectiveParticleVolume = Math.max(particleVolume, 1e-6);
    const initialDemand = emitters
        .filter((emitter) => emitter.behavior === "initial")
        .reduce((sum, emitter) => sum + Math.ceil(fluidShapeVolume(emitter.shape, emitter.transform) / effectiveParticleVolume), 0);
    const inflowDemand = emitters.some((emitter) => emitter.behavior === "inflow") ? Math.max(initialDemand * 2, 20000) : 0;
    const automatic = Math.max(initialDemand + inflowDemand, initialDemand || 20000);
    const requested = requestedCount > 0 ? Math.round(requestedCount) : automatic;
    return Math.max(1, method === "FLIP" ? Math.max(initialDemand, requested) : requested);
}

/**
 * Adjust a PBF physics-parameter value for the particle-size rescaling.
 *
 * PBF sliders author values at the base (1x) particle scale, but two of them must track the
 * particle-size multiplier or the rescaled fluid leaves its stable regime: rest density falls with
 * the cube of the scale (the poly6 kernel sum of the scaled radius shrinks by scale^3) and the
 * constraint relaxation epsilon falls with its square. Every other key is returned unchanged. This
 * is the single home for that coupling, shared by solver construction and live parameter updates.
 */
export function pbfScaleAdjustedParam(key: string, baseValue: number, scale: number): number {
    if (key === "restDensity") {
        return baseValue / (scale * scale * scale);
    }
    if (key === "relaxation") {
        return baseValue / (scale * scale);
    }
    return baseValue;
}

// ─── Compatibility profiles ─────────────────────────────────────────────────
//
// The fluid demo host and the Aquanova host feed the SAME solvers, but they resolve a
// handful of discretization quantities with different constants and policies. Rather than
// silently canonicalizing those divergences into one behavior, they are captured here as two
// explicit, named compatibility profiles so a shared resolver can reproduce EITHER host
// exactly. New consumers pick a profile deliberately; the hosts themselves are migrated
// incrementally (see resolveFluidSimulationConfig docs).

/** Named host-behavior profile. Each reproduces one current host's discretization policy. */
export type FluidCompatibilityProfile = "fluid" | "aquanova";

export type FluidSimulationSemanticsProfile = "normalized-v1" | "legacy-fluid" | "legacy-aquanova";
export type FluidPbfPhysicsSemantics = "scale-adjusted" | "literal";

/** Serialized interpretation of authored solver values. New presets always write this record. */
export interface FluidSimulationSemantics {
    readonly version: 1;
    readonly profile: FluidSimulationSemanticsProfile;
    readonly pbfPhysics: FluidPbfPhysicsSemantics;
}

export const CURRENT_FLUID_SIMULATION_SEMANTICS: FluidSimulationSemantics = {
    version: 1,
    profile: "normalized-v1",
    pbfPhysics: "scale-adjusted",
};

export interface LegacyFluidSimulationSemanticsInput {
    readonly formatVersion?: number;
    readonly demo?: string;
    readonly sourceApplication?: string;
    readonly explicit?: FluidSimulationSemantics;
}

/**
 * Resolve persisted solver semantics. Format 14 and newer must carry an explicit record; older
 * Aquanova-authored files retain literal PBF values, while other legacy files retain Fluid's
 * scale-adjusted interpretation.
 */
export function resolveFluidSimulationSemantics(input: LegacyFluidSimulationSemanticsInput): FluidSimulationSemantics {
    if (input.explicit) {
        if (
            input.explicit.version !== 1 ||
            (input.explicit.profile !== "normalized-v1" && input.explicit.profile !== "legacy-fluid" && input.explicit.profile !== "legacy-aquanova") ||
            (input.explicit.pbfPhysics !== "scale-adjusted" && input.explicit.pbfPhysics !== "literal")
        ) {
            throw new TypeError("[fluid] unsupported simulation semantics.");
        }
        return {
            version: 1,
            profile: input.explicit.profile,
            pbfPhysics: input.explicit.pbfPhysics,
        };
    }
    if ((input.formatVersion ?? 0) >= 14) {
        throw new TypeError("[fluid] format 14 presets require explicit simulationSemantics.");
    }
    const source = `${input.demo ?? ""} ${input.sourceApplication ?? ""}`.toLowerCase();
    const aquanova = source.includes("aquanova");
    return aquanova
        ? { version: 1, profile: "legacy-aquanova", pbfPhysics: "literal" }
        : { version: 1, profile: "legacy-fluid", pbfPhysics: "scale-adjusted" };
}

/** The policy constants that differ between {@link FluidCompatibilityProfile}s. */
export interface FluidCompatibilityProfileSpec {
    readonly profile: FluidCompatibilityProfile;
    /** Whether the profile has a GRIDLESS legacy fallback path at all. The fluid demo does (see the
     *  radius/cell-size constants below); Aquanova does not - it always uses the shared
     *  discretization ({@link fluidSimulationParticleRadius} / {@link fluidSimulationCellSize}). */
    readonly hasGridlessFallback: boolean;
    /** Non-FLIP marker radius (m) per unit physics scale on the GRIDLESS legacy fallback path
     *  (no explicit world grid, fluid sampling). The fluid demo pins 0.09; Aquanova reuses the
     *  shared {@link FLUID_BASE_PARTICLE_RADIUS} (0.08). Ignored when an explicit grid or an
     *  authored mesh radius exists - both hosts then use the shared discretization. */
    readonly gridlessFallbackRadiusPerScale: number;
    /** Whether the gridless fallback multiplies radius and cell size by the legacy domain
     *  multiplier. The fluid demo does; Aquanova has no gridless path and does not. */
    readonly gridlessFallbackAppliesDomainScale: boolean;
    /** Lower bound applied to the non-FLIP per-particle volume (particle diameter cubed). The
     *  fluid demo floors it at 1e-6; Aquanova applies no explicit floor. */
    readonly nonFlipParticleVolumeFloor: number;
    /** FLIP markers-per-cell default when the caller supplies none. Both hosts use 8. */
    readonly defaultFlipMarkersPerCell: number;
}

const FLUID_COMPATIBILITY_PROFILE: FluidCompatibilityProfileSpec = {
    profile: "fluid",
    hasGridlessFallback: true,
    gridlessFallbackRadiusPerScale: 0.09,
    gridlessFallbackAppliesDomainScale: true,
    nonFlipParticleVolumeFloor: 1e-6,
    defaultFlipMarkersPerCell: FLIP_DEFAULT_MARKERS_PER_CELL,
};

const AQUANOVA_COMPATIBILITY_PROFILE: FluidCompatibilityProfileSpec = {
    profile: "aquanova",
    hasGridlessFallback: false,
    gridlessFallbackRadiusPerScale: FLUID_BASE_PARTICLE_RADIUS,
    gridlessFallbackAppliesDomainScale: false,
    nonFlipParticleVolumeFloor: 0,
    defaultFlipMarkersPerCell: FLIP_DEFAULT_MARKERS_PER_CELL,
};

/** Resolve the immutable policy constants for a named compatibility profile. */
export function fluidCompatibilityProfile(profile: FluidCompatibilityProfile): FluidCompatibilityProfileSpec {
    return profile === "aquanova" ? AQUANOVA_COMPATIBILITY_PROFILE : FLUID_COMPATIBILITY_PROFILE;
}

/** World-space axis-aligned simulation grid. */
export interface FluidGridBounds {
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
}

/** Inputs to {@link resolveFluidSimulationConfig}. All world-space geometry is optional so
 *  the resolver also serves gridless legacy demos. */
export interface FluidSimulationConfigInput {
    /** Solver method: "PBF", "FLIP", "MLS-MPM" or "PB-MPM". */
    method: string;
    /** Authored physics-particle-size scale (physScale). */
    physicsScale: number;
    /** "fluid" derives the radius from the physics scale; "mesh" honors an authored radius. Default "fluid". */
    samplingType?: FluidSimulationSamplingType;
    /** Authored particle radius (mesh sampling). Ignored for fluid sampling. */
    particleRadius?: number;
    /** World-space simulation-domain AABB. Required for FLIP; drives grid dimensions otherwise. */
    bounds?: FluidGridBounds;
    /** Explicit world-space grid extent (alternative to {@link bounds} for non-FLIP dimensions). */
    gridSize?: readonly [number, number, number];
    /** FLIP pressure-cell divisions along the longest domain axis. */
    gridResolution?: number;
    /** FLIP marker sampling density. Defaults to the profile's markers-per-cell. */
    markersPerCell?: number;
    /** Whether an explicit world grid is authored (selects the shared discretization path). */
    explicitGrid?: boolean;
    /** Legacy gridless domain multiplier (fluid profile only). Default 1. */
    domainScale?: number;
    /** Authored physics values to normalize for the selected semantics. */
    physics?: Readonly<Record<string, number>>;
    /** Persisted value interpretation. Defaults to the selected host's legacy behavior. */
    semantics?: FluidSimulationSemantics;
}

/** Fully-resolved, host-independent discretization for one (method, profile, input). */
export interface ResolvedFluidSimulationConfig {
    readonly profile: FluidCompatibilityProfile;
    readonly method: string;
    readonly semantics: FluidSimulationSemantics;
    /** Solver-ready physics values after applying explicit semantics. */
    readonly physics: Readonly<Record<string, number>>;
    /** Marker/particle radius in world units. */
    readonly particleRadius: number;
    /** MAC/background grid cell width in world units. */
    readonly cellSize: number;
    /** Per-particle volume used for capacity math (FLIP marker volume; else diameter cubed). */
    readonly particleVolume: number;
    /** Per-axis cell dimensions, when a world grid extent is known. */
    readonly gridDim?: [number, number, number];
    /** Full FLIP discretization detail, present only for the FLIP method. */
    readonly flip?: FlipDiscretization;
}

/**
 * Centralized, profile-aware resolution of particle radius, cell size, grid dimensions and
 * per-particle volume for any fluid backend.
 *
 * This is the single home for the discretization policy the hosts currently inline. It
 * reproduces both current hosts exactly via the named {@link FluidCompatibilityProfile}s:
 *
 * - "fluid": the lab fluid demo. Non-FLIP gridless demos fall back to a "0.09 * scale *
 *   domainScale" radius and a "cellSizeForPhysicsScale(method, scale) * domainScale" cell size;
 *   the non-FLIP particle volume is floored at 1e-6.
 * - "aquanova": the Aquanova host. Always uses the shared {@link fluidSimulationParticleRadius}
 *   / {@link fluidSimulationCellSize} (mesh-authored or physics-scale derived) with no gridless
 *   fallback and no volume floor.
 *
 * FLIP always delegates to {@link resolveFlipDiscretization} (the authoritative FLIP resolver)
 * and requires {@link FluidSimulationConfigInput.bounds}. When {@link FluidSimulationConfigInput.gridResolution}
 * is omitted it falls back to the profile's physics-scale cell size (`cellSizeForPhysicsScale`),
 * reproducing the Aquanova production FLIP path.
 *
 * Adoption: the Aquanova host (`lab/lite/src/demos/aquanova/*`) resolves its solver-creation
 * discretization through this resolver with the "aquanova" profile; the fluid demo uses the
 * "fluid" profile for its non-FLIP path. This is the single home for the discretization policy.
 */
export function resolveFluidSimulationConfig(profile: FluidCompatibilityProfile, input: FluidSimulationConfigInput): ResolvedFluidSimulationConfig {
    const spec = fluidCompatibilityProfile(profile);
    const { method } = input;
    const scale = input.physicsScale;
    const samplingType: FluidSimulationSamplingType = input.samplingType ?? "fluid";
    const domainScale = input.domainScale ?? 1;
    const semantics =
        input.semantics ??
        (profile === "fluid"
            ? { version: 1, profile: "legacy-fluid", pbfPhysics: "scale-adjusted" }
            : { version: 1, profile: "legacy-aquanova", pbfPhysics: "literal" });
    const physics = { ...(input.physics ?? {}) };
    if (method === "PBF" && semantics.pbfPhysics === "scale-adjusted") {
        if (physics.restDensity !== undefined) {
            physics.restDensity = pbfScaleAdjustedParam("restDensity", physics.restDensity, scale);
        }
        if (physics.relaxation !== undefined) {
            physics.relaxation = pbfScaleAdjustedParam("relaxation", physics.relaxation, scale);
        }
    }

    if (method === "FLIP") {
        if (!input.bounds) {
            throw new RangeError("[fluid-config] FLIP requires world-space bounds.");
        }
        // When no explicit grid resolution is authored, fall back to the profile's physics-scale
        // cell size (the fluid profile additionally multiplies by its gridless domain scale). This
        // reproduces the Aquanova production FLIP path, which passes cellSizeForPhysicsScale(method,
        // scale) as dx when the preset omits gridResolution.
        const fallbackCellSize = cellSizeForPhysicsScale(method, scale) * (spec.gridlessFallbackAppliesDomainScale ? domainScale : 1);
        const flip = resolveFlipDiscretization({
            boundsMin: [...input.bounds.min],
            boundsMax: [...input.bounds.max],
            ...(input.gridResolution !== undefined ? { gridResolution: input.gridResolution } : { dx: fallbackCellSize }),
            markersPerCell: input.markersPerCell ?? spec.defaultFlipMarkersPerCell,
            ...(input.particleRadius !== undefined ? { particleRadius: input.particleRadius } : {}),
        });
        return {
            profile: spec.profile,
            method,
            semantics,
            physics,
            particleRadius: flip.particleRadius,
            cellSize: flip.dx,
            particleVolume: flip.markerVolume,
            gridDim: [...flip.gridDim],
            flip,
        };
    }

    // Aquanova has no gridless fallback, so it always uses the shared discretization; the fluid
    // demo only does so with an explicit grid or an authored mesh radius.
    const usesSharedDiscretization = !spec.hasGridlessFallback || input.explicitGrid === true || samplingType === "mesh";
    const discretization: FluidSimulationDiscretization = {
        physicsParticleSize: scale,
        samplingType,
        ...(input.particleRadius !== undefined ? { particleRadius: input.particleRadius } : {}),
    };
    const particleRadius = usesSharedDiscretization
        ? fluidSimulationParticleRadius(discretization)
        : spec.gridlessFallbackRadiusPerScale * scale * (spec.gridlessFallbackAppliesDomainScale ? domainScale : 1);
    const cellSize = usesSharedDiscretization
        ? fluidSimulationCellSize(method, discretization)
        : cellSizeForPhysicsScale(method, scale) * (spec.gridlessFallbackAppliesDomainScale ? domainScale : 1);
    const particleVolume = Math.max((particleRadius * 2) ** 3, spec.nonFlipParticleVolumeFloor);
    const size = input.gridSize ? ([...input.gridSize] as [number, number, number]) : input.bounds ? gridSizeForBounds(input.bounds) : undefined;
    const gridDim = size ? gridCellsForSize(size, cellSize) : undefined;
    return {
        profile: spec.profile,
        method,
        semantics,
        physics,
        particleRadius,
        cellSize,
        particleVolume,
        ...(gridDim ? { gridDim } : {}),
    };
}
