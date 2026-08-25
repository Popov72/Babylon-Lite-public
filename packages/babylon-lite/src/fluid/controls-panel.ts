// Reusable fluid-controls panel — the SINGLE source of truth for the shared
// control UI used by BOTH the fluid demo (fluid.ts) and the Liquefactor demo
// (liquefactor.ts). It builds the collapsible GENERAL / RENDER / FOAM / PHYSICS
// sections (and the optional top-left GPU-timing panel), exactly mirroring the
// look, styling and semantics the fluid demo grew inline. The scene-specific
// "Demo" section is NOT built here: each host prepends its own via `demoSlot`.
//
// Design contract (behaviour preservation of the fluid demo is paramount):
//   • Every control fires a HOST callback (opts.on.*) that applies the effect —
//     identical to the inline handlers the fluid demo used.
//   • Every control also has a PROGRAMMATIC setter on the handle. Setters that
//     the fluid demo's `loadPairState` used to APPLY the effect (color, absorption,
//     size, refraction, …, foam) call the same effect callback; setters whose
//     effect is driven elsewhere (method, particle count, phys scale, show-container)
//     only update the DOM value + read-out label WITHOUT firing a callback — matching
//     `loadPairState` line-for-line so per-(scene,method) pair-state restore is exact.
//   • The physics-slider block is rebuilt per method via `rebuildPhysics`.

import type { FluidDebug } from "./fluid-surface-render.js";
import type { FoamDebugTexture } from "./foam-render.js";
import type { DiffuseParticleCounts, FluidPressureDiagnostics } from "./sim-common.js";

/** One per-method physics slider definition (mirrors the fluid demo's `SCHEMAS`). */
export interface PhysSchemaEntry {
    key: string;
    label: string;
    min: number;
    max: number;
    step: number;
    value: number;
    /** Render as a range slider, dropdown, or boolean checkbox. */
    control?: "slider" | "select" | "checkbox";
    /** Numeric dropdown choices used when control is "select". */
    options?: readonly { label: string; value: number }[];
    /** Show this control only while another numeric parameter has the given value. */
    visibleWhen?: { key: string; equals: number };
    /** Optional visual group used to separate material, timestep, collision, and numerical controls. */
    group?: "liquid" | "timestep" | "collision" | "advanced";
    /** One-line explanation shown in a hover tooltip behind an "i" next to the label. */
    info?: string;
}

/**
 * Default per-method physics-slider definitions shared by every fluid app. Each
 * entry mirrors the corresponding solver default and remembers the last value the
 * user set. Both the fluid demo and the Liquefactor demo pass this as `schemas`
 * (the component deep-copies it, so the shared constant is never mutated). Kept as
 * an exported constant so any host can reuse the exact same per-backend tunables.
 */
export const DEFAULT_FLUID_SCHEMAS: Record<string, PhysSchemaEntry[]> = {
    PBF: [
        {
            key: "gravity",
            label: "Gravity",
            min: 0,
            max: 200,
            step: 0.1,
            value: 9.8,
            info: "Downward acceleration applied to every particle, in world units per second squared. Raising it makes the fluid fall and settle faster.",
        },
        {
            key: "viscosity",
            label: "Viscosity (XSPH)",
            min: 0,
            max: 3,
            step: 0.005,
            value: 0.08,
            info: "XSPH velocity smoothing: each particle is nudged toward the average velocity of its neighbours. Higher = thicker, more syrup-like, less splashy. 0 disables the smoothing pass entirely.",
        },
        {
            key: "relaxation",
            label: "Relaxation \u03b5",
            min: 1,
            max: 1000,
            step: 1,
            value: 50,
            info: "Constraint-force denominator softening. Larger values make the incompressibility solve gentler and more stable, but let the fluid compress more.",
        },
        {
            key: "scorr",
            label: "Artificial pressure",
            min: 0,
            max: 0.5,
            step: 0.001,
            value: 0.02,
            info: "Repulsion added between close neighbours to stop particles clumping into strings (the classic PBF surface-tension hack). Too high looks grainy; 0 turns the correction off.",
        },
        {
            key: "iterations",
            label: "Solver iterations",
            min: 1,
            max: 8,
            step: 1,
            value: 3,
            info: "Density-constraint solves per frame. More iterations = less compressible, better-behaved fluid, at a directly proportional GPU cost.",
        },
        {
            key: "restDensity",
            label: "Rest density",
            min: 100,
            max: 2000,
            step: 10,
            value: 341,
            info: "The density the solver drives the fluid toward. It sets the natural particle spacing, so it must be consistent with the particle size.",
        },
        {
            key: "boundaryDensity",
            label: "Boundary density",
            min: 0,
            max: 1,
            step: 0.05,
            value: 0,
            info: "Phantom density contributed by walls, compensating the SPH density deficit there. Above zero it stops particles piling up against a boundary, at the cost of a thin gap. 0 disables the correction entirely.",
        },
    ],
    FLIP: [
        {
            key: "gravity",
            label: "Gravity",
            min: 0,
            max: 200,
            step: 0.1,
            value: 9.8,
            group: "liquid",
            info: "Downward acceleration applied on the MAC grid, in world units per second squared.",
        },
        {
            key: "flipRatio",
            label: "FLIP ratio",
            min: 0,
            max: 1,
            step: 0.01,
            value: 0.95,
            info: "Blend between dissipative PIC velocity (0) and energy-preserving FLIP velocity change (1). Values near 1 retain splashes and vortices; lower values suppress noise.",
        },
        {
            key: "kinematicViscosity",
            label: "Kinematic viscosity",
            min: 0,
            max: 5,
            step: 0.01,
            value: 0,
            info: "Physical diffusion of velocity gradients on the MAC grid, in world-units squared per second. 0 disables the implicit viscosity solve.",
        },
        {
            key: "surfaceTension",
            label: "Surface tension",
            min: 0,
            max: 5,
            step: 0.01,
            value: 0,
            info: "Cohesion force applied at the liquid-air interface. Higher values form rounder drops and beads and may require more substeps through the capillary stability limit.",
        },
        {
            key: "minSubsteps",
            label: "Minimum substeps",
            min: 1,
            max: 16,
            step: 1,
            value: 1,
            group: "timestep",
            info: "Minimum FLIP steps per rendered frame. Increase this when forces, viscosity, or collisions need more temporal accuracy even at low velocity.",
        },
        {
            key: "maxSubsteps",
            label: "Maximum substeps",
            min: 1,
            max: 32,
            step: 1,
            value: 8,
            info: "Maximum adaptive FLIP steps per rendered frame. This bounds GPU cost when CFL or surface tension requests very small timesteps.",
        },
        {
            key: "cflNumber",
            label: "CFL number",
            min: 0,
            max: 10,
            step: 0.1,
            value: 2,
            info: "Maximum grid cells the fastest marker may travel per substep. Lower values improve collision and free-surface accuracy; 0 disables adaptive CFL for legacy behavior.",
        },
        {
            key: "restitution",
            label: "Restitution (bounce)",
            min: 0,
            max: 1,
            step: 0.05,
            value: 0,
            group: "collision",
            info: "Fraction of inward normal velocity reflected when a particle hits the domain or scene SDF. 0 removes penetration without bounce; 1 is elastic.",
        },
        {
            key: "velocityDamping",
            label: "Velocity damping",
            min: 0,
            max: 10,
            step: 0.05,
            value: 0,
            group: "advanced",
            info: "Non-physical exponential drag applied uniformly to marker velocity. This is separate from viscosity and should normally remain 0.",
        },
        {
            key: "pressureSolver",
            label: "Pressure solver",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "select",
            options: [
                { label: "Weighted Jacobi", value: 0 },
                { label: "Multigrid", value: 1 },
            ],
            group: "advanced",
            info: "Selects the incompressibility solver. Weighted Jacobi is the legacy fixed-iteration path; multigrid removes pressure error across several grid resolutions.",
        },
        {
            key: "pressureIterations",
            label: "Pressure iterations",
            min: 1,
            max: 100,
            step: 1,
            value: 40,
            visibleWhen: { key: "pressureSolver", equals: 0 },
            info: "Weighted-Jacobi iterations used to project the MAC grid to an approximately divergence-free velocity field. More iterations improve incompressibility at proportional GPU cost.",
        },
        {
            key: "pressureRelaxation",
            label: "Pressure relaxation",
            min: 0.1,
            max: 1,
            step: 0.01,
            value: 0.8,
            visibleWhen: { key: "pressureSolver", equals: 0 },
            info: "Weighted-Jacobi relaxation factor. Hidden while multigrid is selected because multigrid uses an internally tuned smoother.",
        },
        {
            key: "multigridCycles",
            label: "Maximum multigrid cycles",
            min: 1,
            max: 8,
            step: 1,
            value: 2,
            visibleWhen: { key: "pressureSolver", equals: 1 },
            info: "Maximum geometric multigrid V-cycles per substep. With Pressure tolerance enabled, completed residual samples adapt the cycle budget up to this cap.",
        },
        {
            key: "pressureTolerance",
            label: "Pressure tolerance",
            min: 0,
            max: 0.1,
            step: 0.0001,
            value: 0,
            visibleWhen: { key: "pressureSolver", equals: 1 },
            info: "Target relative pressure residual. 0 preserves fixed-cycle behavior. Nonzero values adapt the multigrid cycle budget asynchronously without stalling the GPU.",
        },
        {
            key: "pressureDiagnostics",
            label: "Pressure diagnostics",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "checkbox",
            group: "advanced",
            info: "Asynchronously samples pressure residual and post-projection divergence. Pressure tolerance enables the same sampling automatically.",
        },
        {
            key: "liquidSdf",
            label: "Liquid SDF",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "checkbox",
            group: "advanced",
            info: "Builds a particle-derived narrow-band liquid signed-distance field for smoother interface normals and optional ghost-fluid pressure. Disabled by default and records no SDF passes.",
        },
        {
            key: "ghostFluid",
            label: "Ghost-fluid pressure",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "checkbox",
            visibleWhen: { key: "liquidSdf", equals: 1 },
            group: "advanced",
            info: "Uses the liquid SDF to place the zero-pressure free surface between cell centres. Improves thin surfaces and volume behavior at additional GPU cost.",
        },
        {
            key: "fractionalSolids",
            label: "Fractional solid faces",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "checkbox",
            group: "advanced",
            info: "Samples the scene SDF at MAC-face corners and uses the open-area fraction in divergence and pressure projection. Disabled by default.",
        },
        {
            key: "movingSolidBoundaries",
            label: "Moving solid velocity",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "checkbox",
            visibleWhen: { key: "fractionalSolids", equals: 1 },
            group: "advanced",
            info: "Includes SDF-derived obstacle velocity in fractional boundary fluxes. Enable for animated obstacles; leave off for static scenes.",
        },
        {
            key: "reseedParticles",
            label: "Particle reseeding",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "checkbox",
            group: "advanced",
            info: "Redistributes markers from dense cells into sparse interior cells without increasing the global active count. Disabled by default and allocates no reseeding buffers.",
        },
        {
            key: "reseedMinParticles",
            label: "Reseed minimum",
            min: 1,
            max: 64,
            step: 1,
            value: 4,
            visibleWhen: { key: "reseedParticles", equals: 1 },
            info: "Interior cells below this marker count request recycled markers. Partially filled free-surface cells are left unchanged.",
        },
        {
            key: "reseedTargetParticles",
            label: "Reseed target",
            min: 1,
            max: 64,
            step: 1,
            value: 8,
            visibleWhen: { key: "reseedParticles", equals: 1 },
            info: "Marker count requested for an under-populated interior liquid cell, subject to the available recycled-marker budget.",
        },
        {
            key: "reseedMaxParticles",
            label: "Reseed maximum",
            min: 1,
            max: 96,
            step: 1,
            value: 12,
            visibleWhen: { key: "reseedParticles", equals: 1 },
            info: "Markers above this count are removed first; additional markers above the target may be redistributed to sparse interior cells.",
        },
        {
            key: "reseedInterval",
            label: "Reseed interval",
            min: 1,
            max: 30,
            step: 1,
            value: 5,
            visibleWhen: { key: "reseedParticles", equals: 1 },
            info: "Number of FLIP substeps between rate-limited marker redistribution passes.",
        },
        {
            key: "particleSheeting",
            label: "Particle sheeting",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "checkbox",
            group: "advanced",
            info: "Adds markers from unused capacity only in under-sampled, one-cell-thin free-surface sheets. This preserves splash curtains without densifying the bulk liquid.",
        },
        {
            key: "sheetingStrength",
            label: "Sheeting strength",
            min: 0.05,
            max: 1,
            step: 0.05,
            value: 0.5,
            visibleWhen: { key: "particleSheeting", equals: 1 },
            info: "Fraction of the normal marker density restored in detected thin sheets. It also scales the rate-limited number of particles that may be added per pass.",
        },
        {
            key: "sheetingInterval",
            label: "Sheeting interval",
            min: 1,
            max: 30,
            step: 1,
            value: 5,
            visibleWhen: { key: "particleSheeting", equals: 1 },
            info: "Number of FLIP substeps between thin-sheet detection and insertion passes.",
        },
        {
            key: "polygonSurface",
            label: "Polygon surface",
            min: 0,
            max: 1,
            step: 1,
            value: 0,
            control: "checkbox",
            group: "advanced",
            info: "Reconstructs an indexed surface-net mesh from the liquid SDF on the GPU. Refraction ray-marches the SDF and opaque scene depth; reflections use screen-space Hi-Z with an environment fallback. No mesh data is read back to the CPU.",
        },
        {
            key: "polygonReconstructionMultiplier",
            label: "Reconstruction multiplier",
            min: 1,
            max: 2,
            step: 0.25,
            value: 1,
            group: "advanced",
            visibleWhen: { key: "polygonSurface", equals: 1 },
            info: "Render-only samples per FLIP cell axis. Values above 1 interpolate the coherent solver liquid SDF onto a finer polygon grid without changing pressure, whitewater, or timestep resolution. Cost grows cubically: 2× uses roughly 8× as many reconstruction cells.",
        },
        {
            key: "viscosityIterations",
            label: "Viscosity iterations",
            min: 1,
            max: 40,
            step: 1,
            value: 12,
            info: "Jacobi iterations for the implicit viscosity solve. This has no GPU cost when Kinematic viscosity is 0.",
        },
        {
            key: "maxSubDtMs",
            label: "Hard max sub-step (ms)",
            min: 1,
            max: 20,
            step: 0.1,
            value: 8.4,
            info: "Absolute timestep safety cap applied in addition to CFL and surface-tension limits.",
        },
    ],
    "MLS-MPM": [
        {
            key: "gravity",
            label: "Gravity",
            min: 0,
            max: 200,
            step: 0.1,
            value: 9.8,
            info: "Downward acceleration applied to every particle, in world units per second squared. Raising it makes the fluid fall and settle faster.",
        },
        {
            key: "stiffness",
            label: "Stiffness (EOS)",
            min: 10,
            max: 5000,
            step: 10,
            value: 350,
            info: "Equation-of-state pressure gain: how hard the fluid pushes back when compressed. Higher resists compression but needs more substeps to stay stable.",
        },
        {
            key: "viscosity",
            label: "Viscosity",
            min: 0,
            max: 1,
            step: 0.01,
            value: 0.3,
            info: "Resistance to shear. Higher = thicker and more sluggish, and it damps out fine splashes. 0 removes the viscous stress term entirely.",
        },
        {
            key: "restDensity",
            label: "Rest density (/cell)",
            min: 1,
            max: 100,
            step: 0.5,
            value: 3,
            info: "Target mass per grid cell. Together with stiffness it sets the pressure response — it is a per-CELL quantity, not the SPH rest density.",
        },
        {
            key: "damping",
            label: "Velocity damping",
            min: 0.9,
            max: 1,
            step: 0.001,
            value: 0.995,
            info: "Per-substep velocity multiplier that bleeds off energy. 1 disables damping, and the fluid sloshes forever; drop it too far and the fluid goes visibly sluggish.",
        },
        {
            key: "affineDamping",
            label: "Affine damping (\u2192PIC)",
            min: 0.1,
            max: 1,
            step: 0.005,
            value: 0.9,
            info: "Damps each particle's affine velocity field, blending APIC toward plain PIC. 1 disables it (pure APIC); lower = smoother and more dissipative, losing swirl detail.",
        },
        {
            key: "groundDamp",
            label: "Ground damping",
            min: 0.7,
            max: 1,
            step: 0.01,
            value: 0.85,
            info: "Extra velocity damping applied near the floor, used to stop a pool jittering forever once it should have settled. 1 disables it.",
        },
        {
            key: "groundDampHeight",
            label: "Ground damp height",
            min: 0,
            max: 10,
            step: 0.1,
            value: 1.5,
            info: "Height above the floor over which ground damping fades out, in world units. 0 collapses the layer, which disables ground damping altogether.",
        },
        {
            key: "restitution",
            label: "Restitution (bounce)",
            min: 0,
            max: 1,
            step: 0.05,
            value: 0.3,
            info: "How much normal velocity survives a collision with the scene. 0 sticks, 1 bounces perfectly.",
        },
        {
            key: "substeps",
            label: "Substeps / frame",
            min: 1,
            max: 8,
            step: 1,
            value: 3,
            info: "MINIMUM simulation steps per rendered frame. More substeps allow higher stiffness and faster flow without blowing up, at a proportional GPU cost. The solver runs more than this when a step would exceed the max sub-step below.",
        },
        {
            key: "maxSubDtMs",
            label: "Max sub-step (ms)",
            min: 2,
            max: 20,
            step: 0.1,
            value: 8.4,
            info: "Largest slice of time one substep may integrate. A frame always advances by the real elapsed time, so this decides how many substeps that takes: at 60 fps (16.7 ms) a 8.3 ms cap needs 2 steps, a 16.7 ms cap needs 1 (half the cost, less stable). It never changes playback speed.",
        },
    ],
    "PB-MPM": [
        {
            key: "gravity",
            label: "Gravity",
            min: 0,
            max: 200,
            step: 0.1,
            value: 9.8,
            info: "Downward acceleration applied to every particle, in world units per second squared. Raising it makes the fluid fall and settle faster.",
        },
        {
            key: "iterations",
            label: "PB iterations",
            min: 1,
            max: 12,
            step: 1,
            value: 5,
            info: "Position-based constraint iterations per substep. More = stiffer, less compressible material, at a directly proportional GPU cost.",
        },
        {
            key: "liquidRelaxation",
            label: "Liquid relaxation",
            min: 0.1,
            max: 3,
            step: 0.05,
            value: 1.5,
            info: "Over-relaxation factor for the liquid volume constraint. Higher converges faster toward incompressibility but can overshoot and ring.",
        },
        {
            key: "liquidViscosity",
            label: "Liquid viscosity",
            min: 0,
            max: 0.2,
            step: 0.005,
            value: 0.01,
            info: "Resistance to shear in the liquid phase. Higher = thicker and more sluggish flow. 0 disables the viscosity projection.",
        },
        {
            key: "elasticityRatio",
            label: "Elasticity ratio",
            min: 0,
            max: 1,
            step: 0.01,
            value: 0.3,
            info: "Blend between fluid and elastic-solid behaviour. 0 is pure liquid, 1 is a springy solid. Only meaningful for the non-liquid materials.",
        },
        {
            key: "elasticRelaxation",
            label: "Elastic relaxation",
            min: 0.05,
            max: 1,
            step: 0.01,
            value: 0.3,
            info: "How strongly the elastic material is pulled back toward its rest shape each iteration.",
        },
        {
            key: "frictionAngle",
            label: "Sand friction angle",
            min: 0,
            max: 60,
            step: 1,
            value: 35,
            info: "Internal friction angle of the sand material, in degrees — effectively the steepest slope a pile can hold before it collapses.",
        },
        {
            key: "plasticity",
            label: "Visco plasticity",
            min: 0,
            max: 1,
            step: 0.01,
            value: 0.8,
            info: "How readily the viscoelastic material forgets its rest shape and stays deformed, rather than springing back.",
        },
        {
            key: "restitution",
            label: "Restitution (bounce)",
            min: 0,
            max: 1,
            step: 0.05,
            value: 0,
            info: "How much normal velocity survives a collision with the scene. 0 sticks, 1 bounces perfectly.",
        },
        {
            key: "substeps",
            label: "Substeps / frame",
            min: 1,
            max: 8,
            step: 1,
            value: 3,
            info: "MINIMUM simulation steps per rendered frame. More substeps allow stiffer settings and faster flow without blowing up, at a proportional GPU cost. The solver runs more than this when a step would exceed the max sub-step below.",
        },
        {
            key: "maxSubDtMs",
            label: "Max sub-step (ms)",
            min: 2,
            max: 20,
            step: 0.1,
            value: 8.4,
            info: "Largest slice of time one substep may integrate. A frame always advances by the real elapsed time, so this decides how many substeps that takes: at 60 fps (16.7 ms) a 8.3 ms cap needs 2 steps, a 16.7 ms cap needs 1 (half the cost, less stable). It never changes playback speed.",
        },
    ],
};

/** Full foam (diffuse-particle) look/config snapshot. */
export interface FluidFoamValues {
    enabled: boolean;
    activeParticles?: boolean;
    generateSpray?: boolean;
    generateFoam?: boolean;
    generateBubbles?: boolean;
    /** Reject near-surface spray and foam that does not follow the reconstructed surface. */
    surfaceFiltering?: boolean;
    kTa: number;
    kWc: number;
    kTurb: number;
    energySpeedMin: number;
    energySpeedMax: number;
    curvatureMin: number;
    curvatureMax: number;
    turbulenceMin: number;
    turbulenceMax: number;
    foamLayerDepth: number;
    sprayDrag: number;
    kb: number;
    kd: number;
    tMin: number;
    tMax: number;
    poolScale: number;
    /** Visual foam splat-size multiplier (× the foam renderer's base splat radius). */
    size: number;
    blurRadius: number;
    lightIntensity: number;
    ambient: number;
    aoStrength: number;
    normalStrength: number;
    debugTexture: string;
    softness: number;
    density: number;
    subsurfaceStrength: number;
    /** Submerged-bubble tint as an sRGB hex string (e.g. "#b8d1f2"). */
    subsurfaceColor: string;
}

/** Snapshot of every control the component owns (for pair-state capture / export). */
export interface FluidControlValues {
    method: string;
    material: number;
    schema: Record<string, number>;
    simulationDuration: number;
    alphaDecay: number;
    color: string;
    half: boolean;
    thicknessDownscale: number;
    absorption: number;
    size: number;
    physScale: number;
    gridPosition: [number, number, number];
    gridSize: [number, number, number];
    cellSize: number;
    gridResolution: number;
    markersPerCell: number;
    showGridBounds: boolean;
    count: number;
    renderMode: "surface" | "spheres";
    polygonShader: "physical" | "ocean";
    refraction: number;
    specular: number;
    reflectionExposure: number;
    reflectionContrast: number;
    reflectivity: number;
    depthBlur: number;
    depthBlurThreshold: number;
    thicknessBlur: number;
    surfaceFilter: "bilateral" | "narrowRange";
    narrowDelta: number;
    narrowMu: number;
    anisotropic: boolean;
    anisoSurfScale: number;
    activeBlocks: boolean;
    pagedGrid: boolean;
    pagedGridMaxPages: number;
    fusedBlockDiscovery: boolean;
    debug: string;
    showContainer: boolean;
    foam: FluidFoamValues;
}

/** Initial value for every control. */
export interface FluidControlsInitial {
    method: string;
    /** PB-MPM material enum: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. */
    material?: number;
    count: number;
    /** Simulated seconds before particles begin fading. Zero runs indefinitely. */
    simulationDuration?: number;
    /** Seconds taken to fade particle opacity from one to zero after the duration. */
    alphaDecay?: number;
    physScale: number;
    /** World-space center of the active simulation grid. */
    gridPosition?: [number, number, number];
    /** Exact world-space simulation-domain extent along X/Y/Z. */
    gridSize?: [number, number, number];
    /** Derived world-space cubic cell size. */
    cellSize?: number;
    /** FLIP grid divisions along the longest domain side. */
    gridResolution?: number;
    /** FLIP marker samples represented by one full MAC cell. */
    markersPerCell?: number;
    /** Initial visibility of the simulation-domain wireframe. */
    showGridBounds?: boolean;
    color: string;
    absorption: number;
    size: number;
    refraction: number;
    specular: number;
    /** Reflection tonemap exposure. Optional so hosts predating it keep the shader default. */
    reflectionExposure?: number;
    /** Reflection tonemap contrast. Optional for the same reason. */
    reflectionContrast?: number;
    /** Fresnel reflectance at normal incidence (water ≈ 0.02). Optional for the same reason. */
    reflectivity?: number;
    depthBlur: number;
    depthBlurThreshold: number;
    thicknessBlur: number;
    half: boolean;
    thicknessDownscale: number;
    surfaceFilter: "bilateral" | "narrowRange";
    narrowDelta: number;
    narrowMu: number;
    anisotropic: boolean;
    /** Anisotropic WPCA radius damping (0..1); defaults to 0.5 when omitted. */
    anisoSurfScale?: number;
    /** MLS-MPM sparse active-block execution. Defaults off. */
    activeBlocks?: boolean;
    /** Bounded sparse page pool for FLIP or MLS-MPM. Defaults off. */
    pagedGrid?: boolean;
    /** Maximum live grid pages in paged-grid mode (8³ for FLIP, 4³ for MLS-MPM). */
    pagedGridMaxPages?: number;
    /** Append active blocks directly from the particle histogram. Defaults off. */
    fusedBlockDiscovery?: boolean;
    renderMode: "surface" | "spheres";
    /** Shading model used by the FLIP polygon renderer. */
    polygonShader?: "physical" | "ocean";
    debug: string;
    showContainer: boolean;
    foam: FluidFoamValues;
}

/** Host effect callbacks — the component fires these; the host applies the effect. */
export interface FluidControlsCallbacks {
    onMethod?(method: string): void;
    onMaterial?(material: number): void;
    onParticleCount?(count: number): void;
    onSimulationDuration?(seconds: number): void;
    onAlphaDecay?(seconds: number): void;
    onRenderMode?(spheres: boolean): void;
    onPolygonShader?(mode: "physical" | "ocean"): void;
    onColor?(rgb: [number, number, number]): void;
    onAbsorption?(v: number): void;
    onParticleSize?(v: number): void;
    onRefraction?(v: number): void;
    onSpecular?(v: number): void;
    /** Reflection tonemap changed — exposure and contrast, matching the scene's image processing. */
    onReflection?(exposure: number, contrast: number): void;
    /** Fresnel reflectance at normal incidence changed. */
    onReflectivity?(v: number): void;
    onDepthBlur?(size: number, threshold: number): void;
    onThicknessBlur?(v: number): void;
    onHalf?(on: boolean): void;
    onSurfaceFilter?(m: "bilateral" | "narrowRange"): void;
    onNarrowRange?(delta: number, mu: number): void;
    onAnisotropic?(on: boolean): void;
    onAnisotropySurfScale?(share: number): void;
    onThicknessDownscale?(v: number): void;
    onShowContainer?(visible: boolean): void;
    onDebug?(mode: FluidDebug): void;
    onPhysicsParam?(key: string, value: number): void;
    /** Host renderer integration for the FLIP polygon-surface checkbox. */
    onPolygonSurface?(enabled: boolean): void;
    onPhysScale?(scale: number): void;
    onGridResolution?(resolution: number): void;
    onMarkersPerCell?(markersPerCell: number): void;
    onFlipParticleCapacity?(capacity: number): void;
    /** Return an error message to reject proposed grid settings without installing them. */
    onGridSettings?(position: [number, number, number], size: [number, number, number]): string | void;
    onGridGizmo?(visible: boolean): void;
    onShowGridBounds?(visible: boolean): void;
    onActiveBlocks?(enabled: boolean): void;
    onPagedGrid?(enabled: boolean): void;
    onPagedGridMaxPages?(pages: number): void;
    onFusedBlockDiscovery?(enabled: boolean): void;
    onReset?(preserveSceneAnimations?: boolean): void;
    // Foam config (generation) — gated on "enabled" by the host.
    onFoamEnable?(enabled: boolean): void;
    onFoamActiveParticles?(enabled: boolean): void;
    onFoamKinds?(): void;
    onFoamSurfaceFiltering?(enabled: boolean): void;
    onFoamKta?(v: number): void;
    onFoamKwc?(v: number): void;
    onFoamAdvanced?(): void;
    onFoamLifetime?(v: number): void;
    onFoamBuoyancy?(v: number): void;
    onFoamDrag?(v: number): void;
    onFoamPool?(v: number): void;
    onFoamSubColor?(rgb: [number, number, number]): void;
    // Foam screen-space look — always applied to the foam renderer.
    onFoamThresholds?(t0: number, t1: number): void;
    onFoamSubsurface?(v: number): void;
    onFoamSize?(v: number): void;
    onFoamBlur?(v: number): void;
    onFoamLight?(v: number): void;
    onFoamAmbient?(v: number): void;
    onFoamAO?(v: number): void;
    onFoamNormal?(v: number): void;
    onFoamDebugByKind?(on: boolean): void;
    onFoamDebugTexture?(v: FoamDebugTexture): void;
}

/** Optional top-left GPU-timing panel configuration. */
export interface FluidGpuOptions {
    /** Per-stage timing row labels (order preserved). */
    stages: readonly string[];
    /** Whether the GPU supports timestamp-query (a profiler was created). */
    supported: boolean;
}

export interface FluidControlsOptions {
    // ── Visibility flags (each defaults to SHOWN) ──
    hideParticles?: boolean;
    hideMethod?: boolean;
    hideRenderAsSpheres?: boolean;
    hideContainerToggle?: boolean;
    hideFoam?: boolean;
    hideDebug?: boolean;
    hideGpuTiming?: boolean;
    hidePhysics?: boolean;
    /** Show the MLS-MPM active-block execution checkbox. */
    showActiveBlocks?: boolean;
    /** Show grid position, world-space XYZ size and derived cell-size controls. */
    showGridControls?: boolean;
    /** Show duration and alpha-decay lifecycle controls in the General section. */
    showSimulationTiming?: boolean;
    /** When true, the "Physics simulation" section OMITS the "Physics particle size"
     *  row but KEEPS the per-method sliders + reset button. Use
     *  when the host owns its own particle-size control (so physScale would conflict).
     *  The reported physScale (getValues / setPhysScale) still reflects the initial /
     *  last value — only the DOM row is dropped. */
    hidePhysScale?: boolean;

    /** Per-method physics slider definitions (the physics section is driven by these). */
    schemas: Record<string, PhysSchemaEntry[]>;
    /** Method names for the "Fluid method" dropdown (e.g. ["PBF","MLS-MPM"]). */
    methods: string[];
    /** Options for the "Particles" dropdown. */
    particleCounts: number[];
    /** Use a free-form positive integer input instead of the particle-count dropdown. */
    particleCountInput?: boolean;
    /** Initial value for every control. */
    initial: FluidControlsInitial;
    /** "Physics particle size" slider range (defaults 0.5 … 3). */
    physScaleMin?: number;
    physScaleMax?: number;
    /** Maximum legal FLIP particle capacity for the current WebGPU device. */
    flipParticleCapacityMax?: number;
    /** Host effect callbacks. */
    on: FluidControlsCallbacks;
    /** Override the outer panel `cssText` (default = the fluid demo's right-side panel). */
    panelStyle?: string;
    /** Allow the user to resize the panel in both directions (defaults to true). */
    resizable?: boolean;
    /** GPU-timing panel config (only built when provided AND !hideGpuTiming). */
    gpu?: FluidGpuOptions;
}

/** Live handle for the GPU-timing panel (host drives it each frame). */
export interface FluidGpuHandle {
    /** The pinned top-left GPU panel div (host mounts it). */
    panel: HTMLElement;
    /** FPS read-out element (host writes its text each frame). */
    fpsLabel: HTMLElement;
    /** Refresh the memory read-out; the component adds its own texture estimate. */
    refreshMemory(simBytes: number, canvasW: number, canvasH: number): void;
    /** Refresh the per-stage timing rows from the latest profiler results (or null). */
    refreshTiming(res: { stages: Record<string, number>; total: number; frameTotal: number } | null): void;
}

export interface FluidControlsHandle {
    /** The panel div (host mounts it). */
    root: HTMLElement;
    /** Empty slot at the very top the host fills with its scene-specific "Demo" section. */
    demoSlot: HTMLElement;
    /** The "Show container / nozzle meshes" toggle row (null when hidden). Host places it. */
    containerToggleRow: HTMLElement | null;
    /** Build a collapsible section with the shared header styling (for the host's Demo/Export). */
    makeSection(title: string, items: HTMLElement[]): HTMLElement[];
    /** Show or hide a shared section by its title. */
    setSectionVisible(title: string, visible: boolean): void;

    // ── Programmatic setters (see the module contract for which fire callbacks) ──
    setMethod(method: string): void;
    setMaterial(material: number): void;
    setParticleCount(count: number): void;
    setParticleCountVisible(visible: boolean): void;
    setActiveParticleCount(count: number): void;
    setParticleUsage(activeCount: number, totalCount: number, gpuBytes: number, restartActiveCount?: number, restartTotalCount?: number, restartGpuBytes?: number): void;
    setPolygonTriangleCount(count: number | undefined, visible: boolean): void;
    setPressureDiagnostics(diagnostics: FluidPressureDiagnostics | undefined): void;
    setSimulationDuration(seconds: number): void;
    setAlphaDecay(seconds: number): void;
    setRenderMode(spheres: boolean): void;
    setPolygonShader(mode: "physical" | "ocean"): void;
    setColor(hex: string): void;
    setAbsorption(v: number): void;
    setParticleSize(v: number): void;
    setRefraction(v: number): void;
    setSpecular(v: number): void;
    setReflection(exposure: number, contrast: number): void;
    setReflectivity(v: number): void;
    setDepthBlur(size: number, threshold: number): void;
    setThicknessBlur(v: number): void;
    setHalf(on: boolean): void;
    setSurfaceFilter(m: "bilateral" | "narrowRange"): void;
    setNarrowRange(delta: number, mu: number): void;
    setAnisotropic(on: boolean): void;
    setAnisotropySurfScale(v: number): void;
    setThicknessDownscale(v: number): void;
    setShowContainer(on: boolean): void;
    setDebug(mode: string): void;
    setPhysics(schema: Record<string, number>): void;
    setPhysScale(scale: number): void;
    setGridResolution(resolution: number): void;
    setMarkersPerCell(markersPerCell: number): void;
    setFlipParticleCapacity(capacity: number): void;
    setMarkerDensityWarning(message: string): void;
    setGridSettings(position: [number, number, number], size: [number, number, number], cellSize: number): void;
    setGridStatus(message: string): void;
    setShowGridBounds(visible: boolean): void;
    setActiveBlocks(enabled: boolean): void;
    setPagedGrid(enabled: boolean): void;
    setPagedGridMaxPages(pages: number): void;
    setPagedGridStatus(message: string, error?: boolean): void;
    setFusedBlockDiscovery(enabled: boolean): void;
    setFoam(foam: FluidFoamValues): void;
    setFoamParticleCounts(counts: DiffuseParticleCounts | undefined, enabled: boolean, capacity?: number): void;

    /** Snapshot every control value (for pair-state capture / export). */
    getValues(): FluidControlValues;
    /** Physics-slider values for a given method (source of truth for applying to the sim). */
    getPhysicsValues(method: string): Record<string, number>;
    /** Rebuild the physics-slider block for a method (called on method change). */
    rebuildPhysics(method: string): void;
    /** Restrict the physics sliders to the given param keys (e.g. those a PB-MPM material uses);
     *  pass null to show all. Applies to the currently-built sliders. */
    setVisiblePhysicsParams(keys: string[] | null): void;

    /** GPU-timing panel handle, or null when hidden / not configured. */
    gpu: FluidGpuHandle | null;
}

/** Default right-side panel style (the fluid demo's). */
const DEFAULT_PANEL_STYLE =
    "position:fixed;top:12px;right:12px;z-index:20;width:272px;min-width:220px;min-height:160px;max-width:calc(100vw - 24px);max-height:calc(100vh - 24px);" +
    "overflow:auto;resize:both;box-sizing:border-box;font:12px system-ui,-apple-system,'Segoe UI',sans-serif;" +
    "color:#dfe6ee;background:rgba(10,14,20,0.85);padding:10px 12px;border-radius:8px;pointer-events:auto;user-select:none;";
const SELECT_STYLE = "width:100%;margin-bottom:8px;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
const FLIP_GRID_RESOLUTION_MIN = 16;
const FLIP_GRID_RESOLUTION_MAX = 2000;

function hexToRgb(hex: string): [number, number, number] {
    return [parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255, parseInt(hex.slice(5, 7), 16) / 255];
}

export function createFluidControlsPanel(opts: FluidControlsOptions): FluidControlsHandle {
    const on = opts.on;
    const init = opts.initial;
    const physMin = opts.physScaleMin ?? 0.5;
    const physMax = opts.physScaleMax ?? 3;

    // Deep-copy the schemas so the component owns each ParamDef's mutable `value`.
    const schemas: Record<string, PhysSchemaEntry[]> = {};
    for (const m of Object.keys(opts.schemas)) {
        schemas[m] = opts.schemas[m]!.map((p) => ({ ...p }));
    }
    let currentMethod = init.method;
    let applyFoamMethodVisibility = (): void => {};

    // ── Labelled render-slider helper (mirrors the fluid demo's makeRenderSlider). ──
    type RenderSliderRow = HTMLDivElement & { set(v: number): void; get(): number };
    /** A hoverable "i" appended after a setting's name, explaining what the setting does.
     *  Uses the native `title` tooltip: no positioning code, no stacking-context fights with
     *  the panel's own scroll container, and it works unchanged if the panel is ever reparented. */
    function infoIcon(text: string): HTMLSpanElement {
        const i = document.createElement("span");
        i.textContent = "ⓘ";
        i.title = text;
        i.style.cssText = "margin-left:5px;color:#6d7f95;cursor:help;";
        return i;
    }
    /** Label text plus its info icon, for the left-hand side of a control's header row. */
    function labelWithInfo(text: string, info?: string): HTMLSpanElement {
        const lab = document.createElement("span");
        lab.textContent = text;
        if (info) {
            lab.appendChild(infoIcon(info));
        }
        return lab;
    }

    function makeRenderSlider(
        label: string,
        min: number,
        max: number,
        step: number,
        value: number,
        fmt: (v: number) => string,
        onInput: (v: number) => void,
        info?: string
    ): RenderSliderRow {
        const row = document.createElement("div") as RenderSliderRow;
        row.style.cssText = "margin:2px 0 8px;";
        const head = document.createElement("div");
        head.style.cssText = "display:flex;justify-content:space-between;";
        const lab = labelWithInfo(label, info);
        const val = document.createElement("span");
        val.style.cssText = "color:#9fb4cc;";
        val.textContent = fmt(value);
        head.append(lab, val);
        const input = document.createElement("input");
        input.type = "range";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.value = String(value);
        input.style.cssText = "width:100%;";
        input.oninput = () => {
            const v = parseFloat(input.value);
            val.textContent = fmt(v);
            onInput(v);
        };
        row.set = (v: number): void => {
            input.value = String(v);
            val.textContent = fmt(v);
            onInput(v);
        };
        row.get = (): number => parseFloat(input.value);
        row.append(head, input);
        return row;
    }

    // ── Collapsible section builder (shared header styling). ──
    const sections = new Map<string, readonly [HTMLElement, HTMLElement]>();
    const makeSection = (text: string, items: HTMLElement[]): HTMLElement[] => {
        const h = document.createElement("div");
        h.style.cssText =
            "font-weight:700;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#7fb0e0;margin:12px 0 8px;padding-top:9px;border-top:1px solid #2a3647;cursor:pointer;display:flex;justify-content:space-between;align-items:center;user-select:none;";
        const label = document.createElement("span");
        label.textContent = text;
        const caret = document.createElement("span");
        caret.textContent = "\u25be"; // ▾
        caret.style.cssText = "transition:transform 0.15s;";
        h.append(label, caret);
        const body = document.createElement("div");
        body.append(...items);
        h.onclick = () => {
            const collapsed = body.style.display === "none";
            body.style.display = collapsed ? "" : "none";
            caret.style.transform = collapsed ? "" : "rotate(-90deg)";
        };
        sections.set(text, [h, body]);
        return [h, body];
    };

    // ── GENERAL: method + PB-MPM material + particles ───────────────────────
    const methodTitle = document.createElement("div");
    methodTitle.textContent = "Fluid method";
    methodTitle.style.cssText = "font-weight:600;margin-bottom:6px;";
    const methodSel = document.createElement("select");
    methodSel.style.cssText = SELECT_STYLE;
    methodSel.dataset.fluidMethod = "true";
    for (const name of opts.methods) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name === "PBF" ? "SPH (PBF)" : name;
        methodSel.appendChild(opt);
    }
    methodSel.value = init.method;

    const materialRow = document.createElement("div");
    materialRow.style.cssText = "margin:2px 0 8px;";
    const materialTitle = document.createElement("div");
    materialTitle.textContent = "PB-MPM material";
    materialTitle.style.cssText = "font-weight:600;margin-bottom:6px;";
    const materialSel = document.createElement("select");
    materialSel.style.cssText = SELECT_STYLE;
    for (const material of [
        { value: 0, label: "Liquid" },
        { value: 1, label: "Elastic (jelly)" },
        { value: 2, label: "Sand" },
        { value: 3, label: "Viscoelastic" },
    ]) {
        const opt = document.createElement("option");
        opt.value = String(material.value);
        opt.textContent = material.label;
        materialSel.appendChild(opt);
    }
    materialSel.value = String(init.material ?? 0);
    materialSel.onchange = () => on.onMaterial?.(parseInt(materialSel.value, 10));
    materialRow.append(materialTitle, materialSel);
    const applyMaterialVisibility = (): void => {
        materialRow.style.display = currentMethod === "PB-MPM" ? "block" : "none";
    };
    methodSel.onchange = () => {
        currentMethod = methodSel.value;
        applyMaterialVisibility();
        applyActiveBlocksVisibility();
        applyFlipControlVisibility();
        applyFoamMethodVisibility();
        on.onMethod?.(currentMethod);
    };
    applyMaterialVisibility();

    const particlesTitle = document.createElement("div");
    particlesTitle.textContent = "Particles";
    particlesTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const particlesControl = document.createElement(opts.particleCountInput ? "input" : "select");
    particlesControl.style.cssText = SELECT_STYLE;
    if (particlesControl instanceof HTMLInputElement) {
        particlesControl.type = "number";
        particlesControl.min = "1";
        particlesControl.step = "1";
    }
    const particlesRow = document.createElement("div");
    particlesRow.dataset.fluidParticleCountControl = "true";
    particlesRow.append(particlesTitle, particlesControl);
    let particleCountVisible = !opts.hideParticles;
    let committedParticleCount = Math.max(0, Math.round(init.count));
    const particleCountLabel = (count: number): string =>
        count >= 1000000
            ? `${(count / 1000000).toFixed(count % 1000000 === 0 ? 0 : 1)}M`
            : count >= 1000
              ? `${(count / 1000).toFixed(count % 1000 === 0 ? 0 : 1)}k`
              : String(count);
    const ensureParticleCountOption = (count: number): void => {
        if (!(particlesControl instanceof HTMLSelectElement) || particlesControl.querySelector(`option[value="${count}"]`)) {
            return;
        }
        const opt = document.createElement("option");
        opt.value = String(count);
        // "750k" below 1M and "1M" / "1.5M" at/above so larger counts read sensibly.
        opt.textContent = particleCountLabel(count);
        const next = Array.from(particlesControl.options).find((candidate) => Number(candidate.value) > count);
        particlesControl.insertBefore(opt, next ?? null);
    };
    for (const count of opts.particleCounts) {
        ensureParticleCountOption(count);
    }
    ensureParticleCountOption(init.count);
    particlesControl.value = String(init.count);
    particlesControl.onchange = () => {
        const count = Math.round(Number(particlesControl.value));
        if (!Number.isFinite(count) || count < 1) {
            particlesControl.value = String(committedParticleCount);
            return;
        }
        committedParticleCount = count;
        particlesControl.value = String(count);
        on.onParticleCount?.(count);
    };
    const formatParticleCount = (count: number): string => Math.max(0, Math.floor(count)).toLocaleString("en-US");
    const formatGpuBytes = (bytes: number): string => {
        const value = Math.max(0, bytes);
        if (value >= 1024 * 1024 * 1024) {
            return (value / (1024 * 1024 * 1024)).toFixed(2) + " GiB";
        }
        if (value >= 1024 * 1024) {
            return (value / (1024 * 1024)).toFixed(1) + " MiB";
        }
        return (value / 1024).toFixed(1) + " KiB";
    };
    const particleUsageRow = document.createElement("div");
    particleUsageRow.style.cssText = "margin:8px 0;font-size:11px;line-height:1.45;";
    particleUsageRow.dataset.fluidParticleUsage = "true";
    const particleUsageLabel = labelWithInfo(
        "Particle usage",
        "Active is the number of particles currently simulated; total is the number of allocated particle slots. In FLIP, each marker is one simulation particle. Simulation GPU memory counts buffers owned by the active solver only; before a solver starts, the projected allocation is shown in the After restart rows. The GPU panel also estimates fluid render targets."
    );
    particleUsageLabel.style.cssText = "display:block;margin-bottom:3px;";
    const currentParticleUsageValue = document.createElement("div");
    currentParticleUsageValue.style.cssText = "color:#9fb4cc;font-variant-numeric:tabular-nums;";
    const polygonTriangleCountValue = document.createElement("div");
    polygonTriangleCountValue.style.cssText = "display:none;color:#9fb4cc;font-variant-numeric:tabular-nums;";
    polygonTriangleCountValue.dataset.fluidPolygonTriangleCount = "true";
    const restartParticleUsageValue = document.createElement("div");
    restartParticleUsageValue.style.cssText = "display:none;color:#e5bd68;font-variant-numeric:tabular-nums;";
    const particleGpuMemoryValue = document.createElement("div");
    particleGpuMemoryValue.style.cssText = "color:#7c8aa0;font-variant-numeric:tabular-nums;";
    const restartGpuMemoryValue = document.createElement("div");
    restartGpuMemoryValue.style.cssText = "display:none;color:#e5bd68;font-variant-numeric:tabular-nums;";
    particleUsageRow.append(particleUsageLabel, currentParticleUsageValue, polygonTriangleCountValue, restartParticleUsageValue, particleGpuMemoryValue, restartGpuMemoryValue);
    const pressureDiagnosticsRow = document.createElement("div");
    pressureDiagnosticsRow.style.cssText = "display:none;margin:6px 0;font-size:11px;line-height:1.45;color:#9fb4cc;font-variant-numeric:tabular-nums;";
    pressureDiagnosticsRow.dataset.fluidPressureDiagnostics = "true";
    let displayedPressureDiagnostics: FluidPressureDiagnostics | undefined;
    const updatePressureDiagnostics = (): void => {
        const diagnostics = displayedPressureDiagnostics;
        pressureDiagnosticsRow.style.display = currentMethod === "FLIP" && diagnostics ? "block" : "none";
        if (!diagnostics) {
            pressureDiagnosticsRow.textContent = "";
            return;
        }
        pressureDiagnosticsRow.replaceChildren(
            Object.assign(document.createElement("div"), {
                textContent: `Pressure residual:\u00a0${diagnostics.relativeResidual.toExponential(2)}\u00a0relative\u00a0(${diagnostics.pressureIterations}\u00a0iterations)`,
            }),
            Object.assign(document.createElement("div"), {
                textContent: `Post-project divergence:\u00a0${diagnostics.maxDivergence.toExponential(2)}`,
            })
        );
    };
    let displayedActiveParticleCount = init.count;
    let displayedParticleCount = init.count;
    let displayedGpuBytes = 0;
    let restartParticleUsage: { activeCount: number; totalCount: number; gpuBytes: number } | null = null;
    const updateParticleUsage = (): void => {
        const countLabel = currentMethod === "FLIP" ? "capacity" : "total";
        currentParticleUsageValue.textContent =
            "Particles:\u00a0" +
            formatParticleCount(displayedActiveParticleCount) +
            "\u00a0active\u00a0/\u00a0" +
            formatParticleCount(displayedParticleCount) +
            "\u00a0" +
            countLabel;
        particleGpuMemoryValue.textContent =
            displayedParticleCount === 0
                ? "Simulation GPU memory:\u00a0Not running"
                : displayedGpuBytes > 0
                  ? "Simulation GPU memory:\u00a0" + formatGpuBytes(displayedGpuBytes)
                  : "Simulation GPU memory:\u00a0Calculating...";
        const restartDiffers =
            restartParticleUsage !== null &&
            (restartParticleUsage.activeCount !== displayedActiveParticleCount ||
                restartParticleUsage.totalCount !== displayedParticleCount ||
                restartParticleUsage.gpuBytes !== displayedGpuBytes);
        restartParticleUsageValue.style.display = restartDiffers ? "block" : "none";
        restartGpuMemoryValue.style.display = restartDiffers ? "block" : "none";
        if (restartDiffers) {
            restartParticleUsageValue.textContent =
                "After restart:\u00a0" +
                formatParticleCount(restartParticleUsage!.activeCount) +
                "\u00a0active\u00a0/\u00a0" +
                formatParticleCount(restartParticleUsage!.totalCount) +
                "\u00a0" +
                countLabel;
            restartGpuMemoryValue.textContent = "After restart GPU memory:\u00a0" + formatGpuBytes(restartParticleUsage!.gpuBytes);
        }
    };
    const setActiveParticleCount = (count: number): void => {
        displayedActiveParticleCount = count;
        updateParticleUsage();
    };
    const setParticleCapacity = (count: number): void => {
        displayedParticleCount = Math.max(0, count);
        updateParticleUsage();
    };
    updateParticleUsage();
    const formatSeconds = (v: number): string => `${Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1)} s`;
    const simulationDurationRow = makeRenderSlider(
        "Simulation duration",
        0,
        120,
        0.5,
        init.simulationDuration ?? 0,
        (v) => (v === 0 ? "Indefinite" : formatSeconds(v)),
        (v) => on.onSimulationDuration?.(v),
        "How long the simulation advances before particles begin fading. Zero keeps it running indefinitely."
    );
    const alphaDecayRow = makeRenderSlider(
        "Alpha decay",
        0,
        10,
        0.1,
        init.alphaDecay ?? 2,
        formatSeconds,
        (v) => on.onAlphaDecay?.(v),
        "Time taken to fade from fully visible to zero after the duration. At zero opacity, simulation and fluid rendering stop."
    );

    // ── RENDER controls ─────────────────────────────────────────────────────
    // Controls that ONLY affect the screen-space fluid surface (depth/thickness/refraction
    // shading). They are hidden in "Render as spheres" mode where they do nothing. Populated
    // at panel-assembly time (all rows exist by then) and toggled by applySurfaceVisibility.
    const surfaceOnlyRows: HTMLElement[] = [];
    /** Each row's ORIGINAL inline `display`, captured the first time it is toggled.
     *
     *  Restoring `""` instead would DELETE the property rather than restore it, dropping the
     *  element back to its default display. Several of these rows are <label>s carrying
     *  `display:flex`, and a label defaults to `inline` — so un-hiding them that way silently
     *  collapsed "Half rendering" and "Anisotropic surface" onto one shared line. */
    const originalDisplay = new WeakMap<HTMLElement, string>();
    const setRowVisible = (row: HTMLElement, visible: boolean): void => {
        if (!originalDisplay.has(row)) {
            originalDisplay.set(row, row.style.display);
        }
        row.style.display = visible ? originalDisplay.get(row)! : "none";
    };
    const applySurfaceVisibility = (spheres: boolean): void => {
        // Most of these rows sit ABOVE the "Render as spheres" checkbox that drives them, so
        // collapsing them shortens the panel above the click point and yanks the toggle — and
        // everything the user was looking at — hundreds of pixels up (or down, on the way
        // back). Browsers only sometimes absorb that with scroll anchoring, so pin it here:
        // measure the toggle, apply the change, then take the scroll position back by however
        // far it moved. A no-op when the panel is not scrollable or nothing shifted.
        const before = renderRow.getBoundingClientRect().top;
        for (const r of surfaceOnlyRows) {
            setRowVisible(r, !spheres);
        }
        // Showing the surface rows again must not resurrect the ones the CURRENT surface
        // filter / anisotropic state says are irrelevant — this is the single place that
        // decides what is on screen, so it re-applies those rules on the way out.
        if (!spheres) {
            applyFilterVisibility(surfFilterSel.value);
            applyAnisoVisibility(anisoChk.checked);
        }
        const delta = renderRow.getBoundingClientRect().top - before;
        if (delta !== 0) {
            root.scrollTop += delta;
        }
    };

    // Water color picker (Beer-Lambert diffuse tint; sRGB hex → non-sRGB UNORM RGB).
    const colorRow = document.createElement("label");
    colorRow.style.cssText = "display:flex;align-items:center;gap:8px;margin:2px 0 8px;cursor:pointer;";
    const colorLab = document.createElement("span");
    colorLab.textContent = "Water color";
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = init.color;
    colorInput.style.cssText = "width:36px;height:22px;padding:0;border:1px solid #33415a;border-radius:4px;background:#1a2230;cursor:pointer;";
    colorRow.append(colorLab, colorInput);
    colorInput.oninput = () => on.onColor?.(hexToRgb(colorInput.value));

    // Absorption slider (Beer-Lambert strength over thickness).
    const absorbRow = document.createElement("div");
    absorbRow.style.cssText = "margin:2px 0 8px;";
    const absorbHead = document.createElement("div");
    absorbHead.style.cssText = "display:flex;justify-content:space-between;";
    const absorbLab = document.createElement("span");
    absorbLab.textContent = "Absorption (Beer-Lambert)";
    const absorbVal = document.createElement("span");
    absorbVal.style.cssText = "color:#9fb4cc;";
    absorbVal.textContent = init.absorption.toFixed(1);
    absorbHead.append(absorbLab, absorbVal);
    const absorbInput = document.createElement("input");
    absorbInput.type = "range";
    absorbInput.min = "0";
    absorbInput.max = "40";
    absorbInput.step = "0.1";
    absorbInput.value = String(init.absorption);
    absorbInput.style.cssText = "width:100%;";
    absorbInput.oninput = () => {
        const v = parseFloat(absorbInput.value);
        absorbVal.textContent = v.toFixed(1);
        on.onAbsorption?.(v);
    };
    absorbRow.append(absorbHead, absorbInput);

    // Particle size (visual multiplier for impostors + surface splats).
    const sizeRow = document.createElement("div");
    sizeRow.style.cssText = "margin:2px 0 8px;";
    const sizeHead = document.createElement("div");
    sizeHead.style.cssText = "display:flex;justify-content:space-between;";
    const sizeLab = document.createElement("span");
    sizeLab.textContent = "Particle size";
    const sizeVal = document.createElement("span");
    sizeVal.style.cssText = "color:#9fb4cc;";
    sizeVal.textContent = `${init.size.toFixed(2)}\u00d7`;
    sizeHead.append(sizeLab, sizeVal);
    const sizeInput = document.createElement("input");
    sizeInput.type = "range";
    sizeInput.min = "0.1";
    sizeInput.max = "3";
    sizeInput.step = "0.01";
    sizeInput.value = String(init.size);
    sizeInput.style.cssText = "width:100%;";
    sizeInput.oninput = () => {
        const s = parseFloat(sizeInput.value);
        sizeVal.textContent = `${s.toFixed(2)}\u00d7`;
        on.onParticleSize?.(s);
    };
    sizeRow.append(sizeHead, sizeInput);

    const polygonShaderRow = document.createElement("div");
    polygonShaderRow.style.cssText = "margin:2px 0 8px;";
    const polygonShaderLabel = document.createElement("div");
    polygonShaderLabel.textContent = "Surface shader";
    polygonShaderLabel.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const polygonShaderSelect = document.createElement("select");
    polygonShaderSelect.style.cssText = SELECT_STYLE;
    polygonShaderSelect.dataset.fluidSurfaceShader = "true";
    polygonShaderSelect.dataset.fluidPolygonShader = "true";
    for (const option of [
        { value: "physical", label: "Physical refraction" },
        { value: "ocean", label: "Ocean PBR" },
    ]) {
        const element = document.createElement("option");
        element.value = option.value;
        element.textContent = option.label;
        polygonShaderSelect.appendChild(element);
    }
    polygonShaderSelect.value = init.polygonShader ?? "physical";
    polygonShaderSelect.onchange = () => {
        on.onPolygonShader?.(polygonShaderSelect.value as "physical" | "ocean");
    };
    polygonShaderRow.append(polygonShaderLabel, polygonShaderSelect);

    // Surface-shading sliders (the two depth sliders share one setter, so track live).
    let surfRefraction = init.refraction;
    const refractionRow = makeRenderSlider(
        "Refraction strength",
        0,
        1,
        0.01,
        surfRefraction,
        (v) => v.toFixed(2),
        (v) => {
            surfRefraction = v;
            on.onRefraction?.(v);
        }
    );
    let surfSpecular = init.specular;
    const specularRow = makeRenderSlider(
        "Specular power",
        1,
        1000,
        5,
        surfSpecular,
        (v) => String(Math.round(v)),
        (v) => {
            surfSpecular = v;
            on.onSpecular?.(v);
        }
    );
    // Reflection shaping. The first two are the tonemap the environment reflection is put through
    // before it is mixed in, and want to match the scene's own image processing — they are exposed
    // rather than pinned because a demo is free to change that, and a mismatch shows up directly as
    // water whose reflection is brighter or flatter than the sky above it. Defaults come from the
    // shader's own values so an omitted init leaves the surface untouched.
    let surfReflExposure = init.reflectionExposure ?? 1;
    let surfReflContrast = init.reflectionContrast ?? 1.1;
    const reflExposureRow = makeRenderSlider(
        "Reflection exposure",
        0,
        3,
        0.05,
        surfReflExposure,
        (v) => v.toFixed(2),
        (v) => {
            surfReflExposure = v;
            on.onReflection?.(surfReflExposure, surfReflContrast);
        }
    );
    const reflContrastRow = makeRenderSlider(
        "Reflection contrast",
        0,
        3,
        0.05,
        surfReflContrast,
        (v) => v.toFixed(2),
        (v) => {
            surfReflContrast = v;
            on.onReflection?.(surfReflExposure, surfReflContrast);
        }
    );
    let surfReflectivity = init.reflectivity ?? 0.02;
    const reflectivityRow = makeRenderSlider(
        "Water reflectivity",
        0,
        1,
        0.01,
        surfReflectivity,
        (v) => v.toFixed(2),
        (v) => {
            surfReflectivity = v;
            on.onReflectivity?.(v);
        }
    );
    let surfDepthFilter = init.depthBlur;
    let surfDepthThreshold = init.depthBlurThreshold;
    const surfDepthBlurRow = makeRenderSlider(
        "Surface depth blur",
        0,
        100,
        1,
        surfDepthFilter,
        (v) => String(Math.round(v)),
        (v) => {
            surfDepthFilter = v;
            on.onDepthBlur?.(surfDepthFilter, surfDepthThreshold);
        }
    );
    const surfDepthThreshRow = makeRenderSlider(
        "Depth blur edge threshold",
        0,
        100,
        1,
        surfDepthThreshold,
        (v) => String(Math.round(v)),
        (v) => {
            surfDepthThreshold = v;
            on.onDepthBlur?.(surfDepthFilter, surfDepthThreshold);
        }
    );
    let surfThicknessBlur = init.thicknessBlur;
    const surfThickBlurRow = makeRenderSlider(
        "Surface thickness blur",
        0,
        40,
        1,
        surfThicknessBlur,
        (v) => String(Math.round(v)),
        (v) => {
            surfThicknessBlur = v;
            on.onThicknessBlur?.(v);
        }
    );

    // Surface depth smoother selector + narrow-range δ/µ (the two share one setter).
    const surfFilterTitle = document.createElement("div");
    surfFilterTitle.textContent = "Surface filter";
    surfFilterTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const surfFilterSel = document.createElement("select");
    surfFilterSel.style.cssText = SELECT_STYLE;
    for (const o of [
        { value: "bilateral", label: "Bilateral" },
        { value: "narrowRange", label: "Narrow-range" },
    ]) {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        surfFilterSel.appendChild(opt);
    }
    surfFilterSel.value = init.surfaceFilter;
    surfFilterSel.onchange = () => {
        applyFilterVisibility(surfFilterSel.value);
        on.onSurfaceFilter?.(surfFilterSel.value as "bilateral" | "narrowRange");
    };
    let nrDelta = init.narrowDelta;
    let nrMu = init.narrowMu;
    const nrDeltaRow = makeRenderSlider(
        "Narrow range \u03b4 (\u00d7size)",
        1,
        30,
        1,
        nrDelta,
        (v) => String(Math.round(v)),
        (v) => {
            nrDelta = v;
            on.onNarrowRange?.(nrDelta, nrMu);
        }
    );
    const nrMuRow = makeRenderSlider(
        "Narrow range \u00b5 (\u00d7size)",
        0,
        5,
        0.1,
        nrMu,
        (v) => v.toFixed(1),
        (v) => {
            nrMu = v;
            on.onNarrowRange?.(nrDelta, nrMu);
        }
    );

    // Half-resolution toggle (perf).
    const halfRow = document.createElement("label");
    halfRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:8px;cursor:pointer;";
    const halfChk = document.createElement("input");
    halfChk.type = "checkbox";
    halfChk.checked = init.half;
    const halfText = document.createElement("span");
    halfText.textContent = "Half rendering (perf)";
    halfRow.append(halfChk, halfText);
    halfChk.onchange = () => on.onHalf?.(halfChk.checked);

    // Anisotropic surface toggle (Yu & Turk ellipsoidal splatting; default OFF).
    const anisoRow = document.createElement("label");
    anisoRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:8px;cursor:pointer;";
    const anisoChk = document.createElement("input");
    anisoChk.type = "checkbox";
    anisoChk.checked = init.anisotropic;
    const anisoText = document.createElement("span");
    anisoText.textContent = "Anisotropic surface";
    anisoRow.append(anisoChk, anisoText);
    anisoChk.onchange = () => {
        applyAnisoVisibility(anisoChk.checked);
        on.onAnisotropic?.(anisoChk.checked);
    };

    // Anisotropic WPCA radius damping (0 = ignore surfaceSizeScale → tightest neighbourhood,
    // fastest, most sphere-like discs; 1 = full radius → widest/strongest, slowest). Only
    // affects backends with surfaceSizeScale != 1 (MLS-MPM). Live tuning of ANISO_SURFSCALE_RADIUS.
    const anisoDampInit = init.anisoSurfScale ?? 0.5;
    const anisoDampRow = document.createElement("div");
    anisoDampRow.style.cssText = "margin:2px 0 8px;";
    const anisoDampHead = document.createElement("div");
    anisoDampHead.style.cssText = "display:flex;justify-content:space-between;";
    const anisoDampLab = document.createElement("span");
    anisoDampLab.textContent = "Aniso radius damping";
    const anisoDampVal = document.createElement("span");
    anisoDampVal.style.cssText = "color:#9fb4cc;";
    anisoDampVal.textContent = anisoDampInit.toFixed(2);
    anisoDampHead.append(anisoDampLab, anisoDampVal);
    const anisoDampInput = document.createElement("input");
    anisoDampInput.type = "range";
    anisoDampInput.min = "0";
    anisoDampInput.max = "1";
    anisoDampInput.step = "0.05";
    anisoDampInput.value = String(anisoDampInit);
    anisoDampInput.style.cssText = "width:100%;";
    anisoDampInput.oninput = () => {
        const v = parseFloat(anisoDampInput.value);
        anisoDampVal.textContent = v.toFixed(2);
        on.onAnisotropySurfScale?.(v);
    };
    anisoDampRow.append(anisoDampHead, anisoDampInput);

    // Thickness-texture downscale (independent of half rendering).
    const thickDownRow = document.createElement("div");
    thickDownRow.style.cssText = "margin:2px 0 8px;";
    const thickDownHead = document.createElement("div");
    thickDownHead.style.cssText = "display:flex;justify-content:space-between;";
    const thickDownLab = document.createElement("span");
    thickDownLab.textContent = "Thickness downscale";
    const thickDownVal = document.createElement("span");
    thickDownVal.style.cssText = "color:#9fb4cc;";
    thickDownVal.textContent = `${init.thicknessDownscale}\u00d7`;
    thickDownHead.append(thickDownLab, thickDownVal);
    const thickDownInput = document.createElement("input");
    thickDownInput.type = "range";
    thickDownInput.min = "1";
    thickDownInput.max = "16";
    thickDownInput.step = "1";
    thickDownInput.value = String(init.thicknessDownscale);
    thickDownInput.style.cssText = "width:100%;";
    thickDownInput.oninput = () => {
        const v = parseInt(thickDownInput.value, 10);
        thickDownVal.textContent = `${v}\u00d7`;
        on.onThicknessDownscale?.(v);
    };
    thickDownRow.append(thickDownHead, thickDownInput);

    // "Render as spheres" toggle (unchecked = fluid surface).
    const renderRow = document.createElement("label");
    renderRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:8px;cursor:pointer;";
    const renderChk = document.createElement("input");
    renderChk.type = "checkbox";
    renderChk.checked = init.renderMode === "spheres";
    const renderChkText = document.createElement("span");
    renderChkText.textContent = "Render as spheres";
    renderRow.append(renderChk, renderChkText);
    renderChk.onchange = () => {
        on.onRenderMode?.(renderChk.checked);
        applySurfaceVisibility(renderChk.checked);
    };

    // Surface debug (feature) dropdown.
    const debugTitle = document.createElement("div");
    debugTitle.textContent = "Debug (feature)";
    debugTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const debugSel = document.createElement("select");
    debugSel.style.cssText = SELECT_STYLE;
    debugSel.title = "Thickness views compress additive HDR values into grayscale. Polygon wireframe overlays the GPU-reconstructed Surface Nets edges for topology debugging.";
    debugSel.dataset.fluidDebug = "true";
    for (const o of [
        { value: "none", label: "None (final render)" },
        { value: "depth", label: "Depth" },
        { value: "depthBlur", label: "Depth (blurred)" },
        { value: "thickness", label: "Thickness" },
        { value: "thicknessBlur", label: "Thickness (blurred)" },
        { value: "normals", label: "Normals" },
        { value: "polygonWireframe", label: "Polygon wireframe" },
    ]) {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        debugSel.appendChild(opt);
    }
    debugSel.value = init.debug;
    debugSel.onchange = () => on.onDebug?.(debugSel.value as FluidDebug);

    // ── "Show container / nozzle meshes" toggle (host-placed) ────────────────
    const containerRow = document.createElement("label");
    containerRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:8px;cursor:pointer;";
    const containerChk = document.createElement("input");
    containerChk.type = "checkbox";
    containerChk.checked = init.showContainer;
    const containerText = document.createElement("span");
    containerText.textContent = "Show container / nozzle meshes";
    containerRow.append(containerChk, containerText);
    containerChk.onchange = () => on.onShowContainer?.(containerChk.checked);

    // ── PHYSICS ─────────────────────────────────────────────────────────────
    // Laid out like every other slider: ONE flex head carrying the label on the left and the
    // value on the right, then the input. This used to be a separate bold title div above a
    // head that held only the right-aligned value, which pushed the name a whole line higher
    // than every neighbouring row and made it read as a section heading.
    const physRow = document.createElement("div");
    physRow.style.cssText = "margin:2px 0 8px;";
    const physHead = document.createElement("div");
    physHead.style.cssText = "display:flex;justify-content:space-between;";
    const physLab = labelWithInfo(
        "Physics particle size",
        "Scales the SIMULATION particle radius and the neighbour-grid spacing derived from it. Smaller resolves finer detail at a steeply higher cost; changing it rebuilds both backends, so the fluid restarts."
    );
    const physVal = document.createElement("span");
    physVal.style.cssText = "color:#9fb4cc;";
    physVal.textContent = `${init.physScale.toFixed(2)}\u00d7`;
    physHead.append(physLab, physVal);
    const physInput = document.createElement("input");
    physInput.type = "range";
    physInput.min = String(physMin);
    physInput.max = String(physMax);
    physInput.step = "0.01";
    physInput.value = String(init.physScale);
    physInput.style.cssText = "width:100%;";
    physInput.oninput = () => {
        physVal.textContent = `${parseFloat(physInput.value).toFixed(2)}\u00d7`;
    };
    physInput.onchange = () => on.onPhysScale?.(parseFloat(physInput.value));
    physRow.append(physHead, physInput);
    let gridResolution = Math.max(FLIP_GRID_RESOLUTION_MIN, Math.min(FLIP_GRID_RESOLUTION_MAX, Math.round(init.gridResolution ?? 160)));
    const flipResolutionRow = document.createElement("div");
    flipResolutionRow.style.cssText = "display:none;margin:2px 0 8px;";
    flipResolutionRow.dataset.fluidGridResolution = "true";
    const flipResolutionHead = document.createElement("div");
    flipResolutionHead.style.cssText = "display:flex;justify-content:space-between;";
    const flipResolutionLabel = labelWithInfo(
        "Resolution divisions",
        "FLIP grid voxels along the longest side of the domain. Higher values reduce cell size and increase grid memory and marker count cubically. Changes are previewed and applied on Reset simulation."
    );
    const flipResolutionValue = document.createElement("span");
    flipResolutionValue.style.cssText = "color:#9fb4cc;";
    flipResolutionValue.textContent = String(gridResolution);
    flipResolutionHead.append(flipResolutionLabel, flipResolutionValue);
    const flipResolutionInput = document.createElement("input");
    flipResolutionInput.type = "range";
    flipResolutionInput.min = String(FLIP_GRID_RESOLUTION_MIN);
    flipResolutionInput.max = String(FLIP_GRID_RESOLUTION_MAX);
    flipResolutionInput.step = "1";
    flipResolutionInput.value = String(gridResolution);
    flipResolutionInput.style.cssText = "width:100%;";
    flipResolutionInput.oninput = () => {
        flipResolutionValue.textContent = flipResolutionInput.value;
    };
    flipResolutionInput.onchange = () => on.onGridResolution?.(parseInt(flipResolutionInput.value, 10));
    flipResolutionRow.append(flipResolutionHead, flipResolutionInput);

    let markersPerCell = Math.max(1, Math.round(init.markersPerCell ?? 8));
    const flipMarkersRow = document.createElement("div");
    flipMarkersRow.style.cssText = "display:none;align-items:center;justify-content:space-between;gap:8px;margin:8px 0;";
    const flipMarkersLabel = labelWithInfo(
        "Markers per cell",
        "Marker sampling density used to derive active particles from fluid volume. Eight markers per full cell form a 2 x 2 x 2 sub-cell layout. Changes are previewed and applied on Reset simulation."
    );
    const flipMarkersInput = document.createElement("input");
    flipMarkersInput.type = "number";
    flipMarkersInput.min = "1";
    flipMarkersInput.max = "64";
    flipMarkersInput.step = "1";
    flipMarkersInput.value = String(markersPerCell);
    flipMarkersInput.style.cssText = "width:72px;box-sizing:border-box;";
    flipMarkersInput.onchange = () => {
        markersPerCell = Math.max(1, Math.min(64, Math.round(Number.parseFloat(flipMarkersInput.value) || 8)));
        flipMarkersInput.value = String(markersPerCell);
        on.onMarkersPerCell?.(markersPerCell);
    };
    flipMarkersRow.append(flipMarkersLabel, flipMarkersInput);
    const flipParticleCapacityMax = Math.max(1, Math.floor(opts.flipParticleCapacityMax ?? Number.MAX_SAFE_INTEGER));
    let flipParticleCapacity = Math.max(1, Math.min(flipParticleCapacityMax, Math.round(init.count)));
    const flipParticleCapacityRow = document.createElement("div");
    flipParticleCapacityRow.style.cssText = "display:none;align-items:center;justify-content:space-between;gap:8px;margin:8px 0;";
    flipParticleCapacityRow.dataset.fluidFlipParticleCapacity = "true";
    const flipParticleCapacityLabel = labelWithInfo(
        "Particle capacity",
        "Maximum FLIP markers preallocated in GPU buffers. Initial markers are derived from fluid volume; inflows append markers up to this capacity. Changes are previewed and applied on Reset simulation."
    );
    const flipParticleCapacityInput = document.createElement("input");
    flipParticleCapacityInput.type = "number";
    flipParticleCapacityInput.min = "1";
    flipParticleCapacityInput.max = String(flipParticleCapacityMax);
    flipParticleCapacityInput.step = "1";
    flipParticleCapacityInput.value = String(flipParticleCapacity);
    flipParticleCapacityInput.style.cssText = "width:108px;box-sizing:border-box;";
    flipParticleCapacityInput.onchange = () => {
        flipParticleCapacity = Math.max(1, Math.min(flipParticleCapacityMax, Math.round(Number.parseFloat(flipParticleCapacityInput.value) || init.count)));
        flipParticleCapacityInput.value = String(flipParticleCapacity);
        on.onFlipParticleCapacity?.(flipParticleCapacity);
    };
    flipParticleCapacityRow.append(flipParticleCapacityLabel, flipParticleCapacityInput);
    const markerDensityStatus = document.createElement("div");
    markerDensityStatus.style.cssText = "display:none;margin:4px 0 8px;font-size:11px;line-height:1.35;color:#ff5f56;font-weight:600;";
    const applyMarkerDensityStatusVisibility = (): void => {
        markerDensityStatus.style.display = currentMethod === "FLIP" && markerDensityStatus.textContent ? "block" : "none";
    };

    const createGridVectorRow = (
        label: string,
        info: string,
        values: [number, number, number]
    ): { row: HTMLElement; inputs: [HTMLInputElement, HTMLInputElement, HTMLInputElement] } => {
        const row = document.createElement("div");
        row.style.cssText = "margin:8px 0;";
        row.dataset.fluidGridVector = label;
        row.appendChild(labelWithInfo(label, info));
        const fields = document.createElement("div");
        fields.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:5px;";
        const inputs = values.map((value, index) => {
            const field = document.createElement("label");
            field.style.cssText = "display:flex;align-items:center;gap:4px;min-width:0;";
            const axis = document.createElement("span");
            axis.textContent = "XYZ"[index]!;
            axis.style.cssText = "color:#7c8aa0;font-size:11px;";
            const input = document.createElement("input");
            input.type = "number";
            input.step = "0.1";
            input.value = String(value);
            input.style.cssText = "width:100%;min-width:0;box-sizing:border-box;";
            field.append(axis, input);
            fields.appendChild(field);
            return input;
        }) as [HTMLInputElement, HTMLInputElement, HTMLInputElement];
        row.appendChild(fields);
        return { row, inputs };
    };
    const gridPositionControl = createGridVectorRow(
        "Grid position",
        "World-space center of the simulation grid. Emitters and sinks use positions relative to this center. A self-contained imported scene and its collision SDF translate with it; built-in demo geometry remains fixed. FLIP changes are applied on Reset simulation.",
        init.gridPosition ?? [0, 9.5, 0]
    );
    const gridSizeControl = createGridVectorRow(
        "Grid size",
        "Exact world-space X/Y/Z extents of the simulation domain, centered around Grid position. FLIP changes are previewed and applied on Reset simulation; other methods restart immediately.",
        init.gridSize ?? [40, 21, 40]
    );
    const gridStatus = document.createElement("div");
    gridStatus.style.cssText = "display:none;margin:4px 0 8px;font-size:11px;color:#ff8a80;";
    const readGridVector = (inputs: [HTMLInputElement, HTMLInputElement, HTMLInputElement]): [number, number, number] =>
        inputs.map((input) => Number.parseFloat(input.value)) as [number, number, number];
    const setGridStatus = (message: string): void => {
        gridStatus.textContent = message;
        gridStatus.style.display = message ? "block" : "none";
    };
    const commitGridSettings = (): void => {
        const position = readGridVector(gridPositionControl.inputs);
        const size = readGridVector(gridSizeControl.inputs);
        if (!position.every(Number.isFinite)) {
            setGridStatus("Grid position must contain finite numbers.");
            return;
        }
        if (!size.every((value) => Number.isFinite(value) && value > 0)) {
            setGridStatus("Grid size must contain positive finite world-space dimensions.");
            return;
        }
        const error = on.onGridSettings?.(position, size);
        setGridStatus(typeof error === "string" ? error : "");
    };
    for (const input of [...gridPositionControl.inputs, ...gridSizeControl.inputs]) {
        input.onchange = commitGridSettings;
    }
    let gridCellSize = init.cellSize ?? 0;
    const cellSizeRow = document.createElement("div");
    cellSizeRow.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin:8px 0;";
    cellSizeRow.dataset.fluidCellSize = "true";
    const cellSizeLabel = labelWithInfo(
        "Cell size",
        "Derived world-space size of one cubic simulation cell. FLIP derives it from Resolution divisions; other methods derive it from Physics particle size."
    );
    const cellSizeValue = document.createElement("span");
    cellSizeValue.style.cssText = "color:#9fb4cc;font-variant-numeric:tabular-nums;";
    const updateCellSizeValue = (): void => {
        cellSizeValue.textContent = gridCellSize.toFixed(gridCellSize < 0.01 ? 5 : gridCellSize < 0.1 ? 4 : 3);
    };
    updateCellSizeValue();
    cellSizeRow.append(cellSizeLabel, cellSizeValue);

    const applyFlipControlVisibility = (): void => {
        const flip = currentMethod === "FLIP";
        particlesRow.style.display = particleCountVisible && !flip ? "" : "none";
        physRow.style.display = flip ? "none" : "";
        flipResolutionRow.hidden = !flip;
        flipMarkersRow.hidden = !flip;
        flipParticleCapacityRow.hidden = !flip;
        flipResolutionRow.style.display = flip ? "block" : "none";
        flipMarkersRow.style.display = flip ? "flex" : "none";
        flipParticleCapacityRow.style.display = flip ? "flex" : "none";
        applyMarkerDensityStatusVisibility();
        updateParticleUsage();
        updatePressureDiagnostics();
    };
    applyFlipControlVisibility();

    const gridBoundsRow = document.createElement("label");
    gridBoundsRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:8px 0;cursor:pointer;";
    const gridBoundsChk = document.createElement("input");
    gridBoundsChk.type = "checkbox";
    gridBoundsChk.checked = init.showGridBounds ?? false;
    gridBoundsRow.append(
        gridBoundsChk,
        labelWithInfo("Show grid bounds", "Displays the active solver's simulation-domain bounding box. This visualization does not draw every cell or affect the simulation.")
    );
    gridBoundsChk.onchange = () => on.onShowGridBounds?.(gridBoundsChk.checked);

    const gridGizmoRow = document.createElement("label");
    gridGizmoRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:8px 0;cursor:pointer;";
    const gridGizmoCheckbox = document.createElement("input");
    gridGizmoCheckbox.type = "checkbox";
    gridGizmoCheckbox.onchange = () => on.onGridGizmo?.(gridGizmoCheckbox.checked);
    gridGizmoRow.append(gridGizmoCheckbox, labelWithInfo("Gizmo", "Shows position and scale gizmos together. Scaling is rounded to 0.1 world unit when the drag ends."));

    const sliderHost = document.createElement("div");
    const activeBlocksRow = document.createElement("label");
    activeBlocksRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:8px 0;cursor:pointer;";
    const activeBlocksChk = document.createElement("input");
    activeBlocksChk.type = "checkbox";
    activeBlocksChk.checked = init.activeBlocks ?? false;
    const activeBlocksText = labelWithInfo(
        "Active grid blocks",
        "MLS-MPM only. Dispatches particle-to-grid transfer and grid clear/update over occupied blocks and their node halo instead of the full dense grid. Changing it rebuilds the simulations."
    );
    activeBlocksRow.append(activeBlocksChk, activeBlocksText);
    const makeActiveBlockOption = (label: string, info: string, checked: boolean, changed: (enabled: boolean) => void): [HTMLLabelElement, HTMLInputElement] => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex;align-items:center;gap:6px;margin:8px 0 8px 18px;cursor:pointer;";
        const chk = document.createElement("input");
        chk.type = "checkbox";
        chk.checked = checked;
        row.append(chk, labelWithInfo(label, info));
        chk.onchange = () => changed(chk.checked);
        return [row, chk];
    };
    const [fusedBlockDiscoveryRow, fusedBlockDiscoveryChk] = makeActiveBlockOption(
        "Fused block discovery",
        "MLS-MPM active-block mode only. Appends a block when its histogram count changes from zero to one, removing the separate full block-list compaction pass.",
        init.fusedBlockDiscovery ?? false,
        (enabled) => on.onFusedBlockDiscovery?.(enabled)
    );
    const [pagedGridRow, pagedGridChk] = makeActiveBlockOption(
        "Paged grid",
        "Opt-in sparse storage for FLIP and MLS-MPM. Allocates only pages near active fluid instead of every cell in the full domain. The dense backend remains the default fast path. The simulation freezes and reports an error if capacity is exceeded.",
        init.pagedGrid ?? false,
        (enabled) => {
            applyActiveBlockDependencies();
            on.onPagedGrid?.(enabled);
        }
    );
    pagedGridRow.style.margin = "8px 0";
    const pagedGridCapacityRow = document.createElement("label");
    pagedGridCapacityRow.style.cssText = "display:block;margin:8px 0 8px 36px;";
    const pagedGridCapacityHead = document.createElement("div");
    pagedGridCapacityHead.style.cssText = "display:flex;justify-content:space-between;gap:8px;";
    const pagedGridCapacityLabel = labelWithInfo(
        "Page capacity",
        "Maximum number of live grid pages (8×8×8 cells for FLIP, 4×4×4 for MLS-MPM). Higher values use more memory; exceeding the cap freezes the solver instead of integrating against a partial grid."
    );
    const pagedGridCapacityValue = document.createElement("span");
    pagedGridCapacityHead.append(pagedGridCapacityLabel, pagedGridCapacityValue);
    const pagedGridCapacityInput = document.createElement("input");
    pagedGridCapacityInput.type = "range";
    pagedGridCapacityInput.min = "1";
    pagedGridCapacityInput.max = "2000";
    pagedGridCapacityInput.step = "1";
    pagedGridCapacityInput.value = String(Math.max(1, Math.round((init.pagedGridMaxPages ?? 40000) / 1000)));
    pagedGridCapacityInput.style.cssText = "width:100%;";
    const updatePagedGridCapacityValue = (): void => {
        pagedGridCapacityValue.textContent = `${pagedGridCapacityInput.value}k`;
    };
    updatePagedGridCapacityValue();
    pagedGridCapacityInput.oninput = updatePagedGridCapacityValue;
    pagedGridCapacityInput.onchange = () => on.onPagedGridMaxPages?.(parseInt(pagedGridCapacityInput.value, 10) * 1000);
    pagedGridCapacityRow.append(pagedGridCapacityHead, pagedGridCapacityInput);
    const pagedGridStatus = document.createElement("div");
    pagedGridStatus.style.cssText = "display:none;margin:4px 0 8px 36px;font-size:11px;color:#9fb3c8;";
    const applyActiveBlockDependencies = (): void => {
        const flip = currentMethod === "FLIP";
        fusedBlockDiscoveryChk.disabled = !activeBlocksChk.checked;
        pagedGridChk.disabled = !flip && !activeBlocksChk.checked;
        activeBlocksChk.disabled = !flip && pagedGridChk.checked;
        pagedGridCapacityInput.disabled = !pagedGridChk.checked;
        fusedBlockDiscoveryRow.style.opacity = activeBlocksChk.checked ? "1" : "0.5";
        pagedGridRow.style.opacity = flip || activeBlocksChk.checked ? "1" : "0.5";
        pagedGridCapacityRow.style.opacity = pagedGridChk.checked ? "1" : "0.5";
    };
    activeBlocksChk.onchange = () => {
        applyActiveBlockDependencies();
        on.onActiveBlocks?.(activeBlocksChk.checked);
    };
    const applyActiveBlocksVisibility = (): void => {
        const mpmDisplay = opts.showActiveBlocks && currentMethod === "MLS-MPM" ? "flex" : "none";
        const pagedDisplay = opts.showActiveBlocks && (currentMethod === "MLS-MPM" || currentMethod === "FLIP") ? "flex" : "none";
        activeBlocksRow.style.display = mpmDisplay;
        fusedBlockDiscoveryRow.style.display = mpmDisplay;
        pagedGridRow.style.display = pagedDisplay;
        pagedGridCapacityRow.style.display = pagedDisplay;
        pagedGridStatus.style.display = pagedDisplay !== "none" && pagedGridStatus.textContent ? "block" : "none";
        applyActiveBlockDependencies();
    };
    applyActiveBlockDependencies();
    applyActiveBlocksVisibility();

    // Per-method physics sliders can be filtered to only those relevant to the host's current
    // config (e.g. a PB-MPM material only uses a subset). null = show all. paramRows maps each
    // slider's param key → its DOM row/input so filtering and dependencies are cheap updates.
    const paramRows = new Map<string, HTMLElement>();
    const paramInputs = new Map<string, HTMLInputElement | HTMLSelectElement>();
    let visibleParamKeys: Set<string> | null = null;
    const applyParamVisibility = (): void => {
        for (const [key, row] of paramRows) {
            const definition = (schemas[currentMethod] ?? []).find((entry) => entry.key === key);
            const dependency = definition?.visibleWhen;
            const dependencyApplies = !dependency || (schemas[currentMethod] ?? []).find((entry) => entry.key === dependency.key)?.value === dependency.equals;
            const hostVisible = !visibleParamKeys || visibleParamKeys.has(key);
            const hostSupported = on.onPhysicsParam !== undefined && (key !== "polygonSurface" || on.onPolygonSurface !== undefined);
            const keepVisible = definition?.control === "checkbox";
            row.style.display = hostVisible && (dependencyApplies || keepVisible) ? "" : "none";
            const input = paramInputs.get(key);
            if (input) {
                if (!dependencyApplies && definition?.control === "checkbox" && input instanceof HTMLInputElement && input.checked) {
                    input.checked = false;
                    definition.value = 0;
                    on.onPhysicsParam?.(key, 0);
                }
                input.disabled = !dependencyApplies || !hostSupported;
            }
            row.style.opacity = dependencyApplies && hostSupported ? "1" : "0.5";
        }
    };
    function buildSliders(name: string): void {
        sliderHost.replaceChildren();
        paramRows.clear();
        paramInputs.clear();
        let activeGroup: PhysSchemaEntry["group"];
        let groupHost: HTMLElement = sliderHost;
        const groupLabels: Record<NonNullable<PhysSchemaEntry["group"]>, string> = {
            liquid: "Liquid",
            timestep: "Time stepping",
            collision: "Collision",
            advanced: "Advanced numerical",
        };
        for (const p of schemas[name] ?? []) {
            if (p.group && p.group !== activeGroup) {
                activeGroup = p.group;
                if (p.group === "advanced") {
                    const details = document.createElement("details");
                    details.open = true;
                    details.dataset.fluidPhysicsGroup = p.group;
                    const summary = document.createElement("summary");
                    summary.textContent = groupLabels[p.group];
                    summary.style.cssText = "font-weight:600;margin:10px 0 4px;color:#c7d7ea;cursor:pointer;user-select:none;";
                    details.appendChild(summary);
                    sliderHost.appendChild(details);
                    groupHost = details;
                } else {
                    const heading = document.createElement("div");
                    heading.textContent = groupLabels[p.group];
                    heading.style.cssText = "font-weight:600;margin:10px 0 4px;color:#c7d7ea;";
                    sliderHost.appendChild(heading);
                    groupHost = sliderHost;
                }
            }
            const row = document.createElement("div");
            row.style.cssText = "margin:6px 0;";
            row.dataset.fluidPhysicsParam = p.key;
            if (p.control === "checkbox") {
                row.style.cssText = "display:flex;align-items:center;gap:6px;margin:8px 0;cursor:pointer;";
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.checked = p.value >= 0.5;
                checkbox.onchange = () => {
                    p.value = checkbox.checked ? 1 : 0;
                    on.onPhysicsParam?.(p.key, p.value);
                    if (p.key === "polygonSurface") {
                        on.onPolygonSurface?.(checkbox.checked);
                    }
                    applyParamVisibility();
                };
                const label = document.createElement("label");
                label.style.cssText = "display:flex;align-items:center;gap:6px;cursor:pointer;";
                label.append(checkbox, labelWithInfo(p.label, p.info));
                row.appendChild(label);
                groupHost.appendChild(row);
                paramRows.set(p.key, row);
                paramInputs.set(p.key, checkbox);
                continue;
            }
            if (p.control === "select") {
                const select = document.createElement("select");
                select.style.cssText = SELECT_STYLE;
                for (const option of p.options ?? []) {
                    const element = document.createElement("option");
                    element.value = String(option.value);
                    element.textContent = option.label;
                    select.appendChild(element);
                }
                select.value = String(p.value);
                select.onchange = () => {
                    const value = Number.parseFloat(select.value);
                    p.value = value;
                    on.onPhysicsParam?.(p.key, value);
                    applyParamVisibility();
                };
                row.append(labelWithInfo(p.label, p.info), select);
                groupHost.appendChild(row);
                paramRows.set(p.key, row);
                paramInputs.set(p.key, select);
                continue;
            }
            const head = document.createElement("div");
            head.style.cssText = "display:flex;justify-content:space-between;";
            const lab = labelWithInfo(p.label, p.info);
            const val = document.createElement("span");
            val.style.cssText = "color:#9fb4cc;";
            val.textContent = String(p.value);
            head.append(lab, val);
            const input = document.createElement("input");
            input.type = "range";
            input.min = String(p.min);
            input.max = String(p.max);
            input.step = String(p.step);
            input.value = String(p.value);
            input.style.cssText = "width:100%;";
            input.oninput = () => {
                const v = parseFloat(input.value);
                p.value = v;
                val.textContent = String(v);
                on.onPhysicsParam?.(p.key, v);
            };
            row.append(head, input);
            groupHost.appendChild(row);
            paramRows.set(p.key, row);
            paramInputs.set(p.key, input);
        }
        applyParamVisibility();
    }
    buildSliders(currentMethod);

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset simulation";
    resetBtn.style.cssText = "width:100%;margin-top:8px;padding:5px;cursor:pointer;background:#26415f;color:#eef3f8;border:1px solid #3a567a;border-radius:4px;";
    resetBtn.onclick = (event) => on.onReset?.(event.shiftKey);

    // ── FOAM controls ───────────────────────────────────────────────────────
    let foamEnabled = init.foam.enabled;
    let foamTMin = init.foam.tMin;
    // Foam generation config (mirrors the fluid demo's foamCfg fields).
    const foamCfg = {
        kTa: init.foam.kTa,
        kWc: init.foam.kWc,
        kTurb: init.foam.kTurb ?? 0,
        energySpeedMin: init.foam.energySpeedMin ?? Math.sqrt(0.5),
        energySpeedMax: init.foam.energySpeedMax ?? Math.sqrt(40),
        curvatureMin: init.foam.curvatureMin ?? 0.05,
        curvatureMax: init.foam.curvatureMax ?? 1.5,
        turbulenceMin: init.foam.turbulenceMin ?? 0.1,
        turbulenceMax: init.foam.turbulenceMax ?? 2.5,
        foamLayerDepth: init.foam.foamLayerDepth ?? 0,
        sprayDrag: init.foam.sprayDrag ?? 0,
        kb: init.foam.kb,
        kd: init.foam.kd,
        tMax: init.foam.tMax,
        poolScale: init.foam.poolScale,
    };
    let foamT0 = init.foam.softness;
    let foamT1 = init.foam.density;
    let foamSubStrength = init.foam.subsurfaceStrength;
    let foamSize = init.foam.size;
    let foamBlurRadius = init.foam.blurRadius;
    let foamLightIntensity = init.foam.lightIntensity;
    let foamAmbient = init.foam.ambient;
    let foamAOStrength = init.foam.aoStrength;
    let foamNormalStrength = init.foam.normalStrength;
    let foamGenerateFoam = init.foam.generateFoam ?? true;
    let foamGenerateSpray = init.foam.generateSpray ?? true;
    let foamGenerateBubbles = init.foam.generateBubbles ?? true;
    let foamSurfaceFiltering = foamEnabled && (init.foam.surfaceFiltering ?? currentMethod === "FLIP");

    const foamUsageRow = document.createElement("div");
    foamUsageRow.dataset.fluidFoamCounts = "true";
    foamUsageRow.style.cssText = "margin:0 0 8px 18px;color:#aebdca;font-size:11px;line-height:1.45;";
    const updateFoamParticleCounts = (counts: DiffuseParticleCounts | undefined, enabled: boolean, capacity?: number): void => {
        const lines: string[] = [];
        if (!enabled) {
            lines.push("Disabled");
        } else if (!counts) {
            lines.push(capacity ? `Calculating…\u00a0/\u00a0${capacity.toLocaleString()}\u00a0particles` : "Calculating…");
        } else {
            lines.push(`${counts.total.toLocaleString()}\u00a0/\u00a0${counts.capacity.toLocaleString()}\u00a0particles`);
            if (foamGenerateFoam) {
                lines.push(`Foam:\u00a0${counts.foam.toLocaleString()}`);
            }
            if (foamGenerateSpray) {
                lines.push(`Spray:\u00a0${counts.spray.toLocaleString()}`);
            }
            if (foamGenerateBubbles) {
                lines.push(`Bubbles:\u00a0${counts.bubble.toLocaleString()}`);
            }
        }
        foamUsageRow.replaceChildren(
            ...lines.map((text) => {
                const line = document.createElement("div");
                line.textContent = text;
                return line;
            })
        );
    };

    const foamEnableRow = document.createElement("label");
    foamEnableRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:8px;cursor:pointer;";
    const foamEnableChk = document.createElement("input");
    foamEnableChk.type = "checkbox";
    foamEnableChk.checked = foamEnabled;
    const foamEnableText = document.createElement("span");
    foamEnableText.textContent = "Enable foam";
    foamEnableRow.append(foamEnableChk, foamEnableText);
    foamEnableChk.onchange = () => {
        foamEnabled = foamEnableChk.checked;
        if (!foamEnabled && foamSurfaceFiltering) {
            foamSurfaceFiltering = false;
            foamSurfaceFilteringChk.checked = false;
            on.onFoamSurfaceFiltering?.(false);
        }
        applyFoamKindAvailability();
        updateFoamParticleCounts(undefined, foamEnabled);
        on.onFoamEnable?.(foamEnabled);
    };
    const makeFoamKindToggle = (label: string, checked: boolean, info: string): [HTMLLabelElement, HTMLInputElement] => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex;align-items:center;gap:6px;margin:0 0 6px 18px;cursor:pointer;";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = checked;
        row.append(input, labelWithInfo(label, info));
        return [row, input];
    };
    const [foamGenerateFoamRow, foamGenerateFoamChk] = makeFoamKindToggle(
        "Generate foam",
        foamGenerateFoam,
        "Allows diffuse particles attached to the liquid surface. This is independent of the overall Enable foam switch."
    );
    const [foamGenerateSprayRow, foamGenerateSprayChk] = makeFoamKindToggle(
        "Generate spray",
        foamGenerateSpray,
        "Allows detached low-density diffuse particles. Disabling it removes existing spray and prevents it from consuming pool slots."
    );
    const [foamGenerateBubblesRow, foamGenerateBubblesChk] = makeFoamKindToggle(
        "Generate bubbles",
        foamGenerateBubbles,
        "Allows submerged diffuse particles. Subsurface bubble strength only changes their rendering; this switch controls whether they exist."
    );
    foamGenerateFoamChk.onchange = () => {
        foamGenerateFoam = foamGenerateFoamChk.checked;
        on.onFoamKinds?.();
    };
    foamGenerateSprayChk.onchange = () => {
        foamGenerateSpray = foamGenerateSprayChk.checked;
        on.onFoamKinds?.();
    };
    foamGenerateBubblesChk.onchange = () => {
        foamGenerateBubbles = foamGenerateBubblesChk.checked;
        on.onFoamKinds?.();
    };
    const foamSurfaceFilteringRow = document.createElement("label");
    foamSurfaceFilteringRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:0 0 8px 18px;cursor:pointer;";
    const foamSurfaceFilteringChk = document.createElement("input");
    foamSurfaceFilteringChk.type = "checkbox";
    foamSurfaceFilteringChk.checked = foamSurfaceFiltering;
    foamSurfaceFilteringRow.append(
        foamSurfaceFilteringChk,
        labelWithInfo(
            "Strict surface filtering",
            "Rejects spray that has not separated from the reconstructed liquid and foam that does not follow its local surface. Useful for suppressing persistent boundary artifacts, but may remove foam from vertical cascades."
        )
    );
    foamSurfaceFilteringChk.onchange = () => {
        foamSurfaceFiltering = foamSurfaceFilteringChk.checked;
        on.onFoamSurfaceFiltering?.(foamSurfaceFiltering);
    };
    const foamKindRows = [foamGenerateFoamRow, foamGenerateSprayRow, foamGenerateBubblesRow];
    const foamKindChecks = [foamGenerateFoamChk, foamGenerateSprayChk, foamGenerateBubblesChk];
    const applyFoamKindAvailability = (): void => {
        for (let index = 0; index < foamKindRows.length; index++) {
            foamKindChecks[index]!.disabled = !foamEnabled;
            foamKindRows[index]!.style.opacity = foamEnabled ? "1" : "0.5";
        }
        foamSurfaceFilteringChk.disabled = !foamEnabled;
        foamSurfaceFilteringRow.style.opacity = foamEnabled ? "1" : "0.5";
    };
    applyFoamKindAvailability();
    updateFoamParticleCounts(undefined, foamEnabled);
    const foamKtaRow = makeRenderSlider(
        "Trapped-air rate",
        0,
        500,
        1,
        foamCfg.kTa,
        (v) => String(Math.round(v)),
        (v) => {
            foamCfg.kTa = v;
            on.onFoamKta?.(v);
        }
    );
    const foamKwcRow = makeRenderSlider(
        "Wave-crest rate",
        0,
        500,
        1,
        foamCfg.kWc,
        (v) => String(Math.round(v)),
        (v) => {
            foamCfg.kWc = v;
            on.onFoamKwc?.(v);
        }
    );
    const foamAdvancedTitle = document.createElement("div");
    foamAdvancedTitle.textContent = "FLIP advanced whitewater";
    foamAdvancedTitle.dataset.fluidFlipFoamAdvanced = "true";
    foamAdvancedTitle.style.cssText = "font-weight:600;color:#9fb4cc;margin:10px 0 6px;";
    const foamTurbulenceRateRow = makeRenderSlider(
        "Turbulence rate",
        0,
        500,
        1,
        foamCfg.kTurb,
        (v) => String(Math.round(v)),
        (v) => {
            foamCfg.kTurb = v;
            on.onFoamAdvanced?.();
        }
    );
    const foamEnergyMinRow = makeRenderSlider(
        "Energy speed min",
        0,
        20,
        0.1,
        foamCfg.energySpeedMin,
        (v) => v.toFixed(1),
        (v) => {
            foamCfg.energySpeedMin = Math.min(v, foamCfg.energySpeedMax - 0.1);
            on.onFoamAdvanced?.();
        }
    );
    const foamEnergyMaxRow = makeRenderSlider(
        "Energy speed max",
        0.1,
        40,
        0.1,
        foamCfg.energySpeedMax,
        (v) => v.toFixed(1),
        (v) => {
            foamCfg.energySpeedMax = Math.max(v, foamCfg.energySpeedMin + 0.1);
            on.onFoamAdvanced?.();
        }
    );
    const foamCurvatureMinRow = makeRenderSlider(
        "Curvature min",
        0,
        4,
        0.01,
        foamCfg.curvatureMin,
        (v) => v.toFixed(2),
        (v) => {
            foamCfg.curvatureMin = Math.min(v, foamCfg.curvatureMax - 0.01);
            on.onFoamAdvanced?.();
        }
    );
    const foamCurvatureMaxRow = makeRenderSlider(
        "Curvature max",
        0.01,
        8,
        0.01,
        foamCfg.curvatureMax,
        (v) => v.toFixed(2),
        (v) => {
            foamCfg.curvatureMax = Math.max(v, foamCfg.curvatureMin + 0.01);
            on.onFoamAdvanced?.();
        }
    );
    const foamTurbulenceMinRow = makeRenderSlider(
        "Turbulence min",
        0,
        20,
        0.05,
        foamCfg.turbulenceMin,
        (v) => v.toFixed(2),
        (v) => {
            foamCfg.turbulenceMin = Math.min(v, foamCfg.turbulenceMax - 0.05);
            on.onFoamAdvanced?.();
        }
    );
    const foamTurbulenceMaxRow = makeRenderSlider(
        "Turbulence max",
        0.05,
        40,
        0.05,
        foamCfg.turbulenceMax,
        (v) => v.toFixed(2),
        (v) => {
            foamCfg.turbulenceMax = Math.max(v, foamCfg.turbulenceMin + 0.05);
            on.onFoamAdvanced?.();
        }
    );
    const foamLayerDepthRow = makeRenderSlider(
        "Foam layer depth",
        0,
        4,
        0.1,
        foamCfg.foamLayerDepth,
        (v) => `${v.toFixed(1)} cells`,
        (v) => {
            foamCfg.foamLayerDepth = v;
            on.onFoamAdvanced?.();
        }
    );
    const foamSprayDragRow = makeRenderSlider(
        "Spray air drag",
        0,
        10,
        0.1,
        foamCfg.sprayDrag,
        (v) => `${v.toFixed(1)} /s`,
        (v) => {
            foamCfg.sprayDrag = v;
            on.onFoamAdvanced?.();
        }
    );
    const foamLifeRow = makeRenderSlider(
        "Foam lifetime (s)",
        0.3,
        20,
        0.1,
        foamCfg.tMax,
        (v) => v.toFixed(1),
        (v) => {
            foamCfg.tMax = v;
            on.onFoamLifetime?.(v);
        }
    );
    const foamBuoyRow = makeRenderSlider(
        "Bubble buoyancy",
        0,
        8,
        0.05,
        foamCfg.kb,
        (v) => v.toFixed(2),
        (v) => {
            foamCfg.kb = v;
            on.onFoamBuoyancy?.(v);
        }
    );
    const foamDragRow = makeRenderSlider(
        "Bubble drag",
        0,
        1,
        0.05,
        foamCfg.kd,
        (v) => v.toFixed(2),
        (v) => {
            foamCfg.kd = v;
            on.onFoamDrag?.(v);
        }
    );
    const foamPoolRow = makeRenderSlider(
        "Pool size (\u00d7 fluid)",
        1,
        6,
        0.5,
        foamCfg.poolScale,
        (v) => `${v.toFixed(1)}\u00d7`,
        (v) => {
            foamCfg.poolScale = v;
            on.onFoamPool?.(v);
        }
    );
    const foamSoftRow = makeRenderSlider(
        "Foam softness",
        0,
        10,
        0.02,
        foamT0,
        (v) => v.toFixed(2),
        (v) => {
            foamT0 = Math.min(v, foamT1 - 0.02);
            on.onFoamThresholds?.(foamT0, foamT1);
        }
    );
    const foamDensityRow = makeRenderSlider(
        "Foam density",
        0.2,
        200,
        0.05,
        foamT1,
        (v) => v.toFixed(2),
        (v) => {
            foamT1 = Math.max(v, foamT0 + 0.02);
            on.onFoamThresholds?.(foamT0, foamT1);
        }
    );
    const foamSubRow = makeRenderSlider(
        "Subsurface bubble strength",
        0,
        1,
        0.05,
        foamSubStrength,
        (v) => v.toFixed(2),
        (v) => {
            foamSubStrength = v;
            on.onFoamSubsurface?.(v);
        }
    );
    // Submerged-bubble tint. A swatch rather than a slider triple: it is a look choice, and
    // it sits directly under the strength slider it modulates.
    const foamSubColorRow = document.createElement("label");
    foamSubColorRow.style.cssText = "display:flex;align-items:center;gap:8px;margin:2px 0 8px;cursor:pointer;";
    const foamSubColorLab = document.createElement("span");
    foamSubColorLab.textContent = "Subsurface bubble color";
    const foamSubColorInput = document.createElement("input");
    foamSubColorInput.type = "color";
    foamSubColorInput.value = init.foam.subsurfaceColor;
    foamSubColorInput.style.cssText = "width:36px;height:22px;padding:0;border:1px solid #33415a;border-radius:4px;background:#1a2230;cursor:pointer;";
    foamSubColorRow.append(foamSubColorLab, foamSubColorInput);
    foamSubColorInput.oninput = () => on.onFoamSubColor?.(hexToRgb(foamSubColorInput.value));

    const foamSizeRow = makeRenderSlider(
        "Foam size",
        0.1,
        3,
        0.05,
        foamSize,
        (v) => `${v.toFixed(2)}\u00d7`,
        (v) => {
            foamSize = v;
            on.onFoamSize?.(v);
        }
    );
    const foamBlurRow = makeRenderSlider(
        "Foam blur radius",
        0,
        12,
        1,
        foamBlurRadius,
        (v) => String(Math.round(v)),
        (v) => {
            foamBlurRadius = v;
            on.onFoamBlur?.(v);
        }
    );
    const foamLightRow = makeRenderSlider(
        "Foam light intensity",
        0,
        2,
        0.05,
        foamLightIntensity,
        (v) => v.toFixed(2),
        (v) => {
            foamLightIntensity = v;
            on.onFoamLight?.(v);
        }
    );
    const foamAmbientRow = makeRenderSlider(
        "Foam ambient",
        0,
        2,
        0.02,
        foamAmbient,
        (v) => v.toFixed(2),
        (v) => {
            foamAmbient = v;
            on.onFoamAmbient?.(v);
        }
    );
    const foamAORow = makeRenderSlider(
        "Foam AO / shadow",
        0,
        1,
        0.02,
        foamAOStrength,
        (v) => v.toFixed(2),
        (v) => {
            foamAOStrength = v;
            on.onFoamAO?.(v);
        }
    );
    const foamNormalRow = makeRenderSlider(
        "Foam normal strength",
        0,
        16,
        0.5,
        foamNormalStrength,
        (v) => v.toFixed(1),
        (v) => {
            foamNormalStrength = v;
            on.onFoamNormal?.(v);
        }
    );
    const foamDebugRow = document.createElement("label");
    foamDebugRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:6px;cursor:pointer;";
    const foamDebugChk = document.createElement("input");
    foamDebugChk.type = "checkbox";
    const foamDebugText = document.createElement("span");
    foamDebugText.textContent = "Debug: colour by kind";
    foamDebugRow.append(foamDebugChk, foamDebugText);
    foamDebugChk.onchange = () => on.onFoamDebugByKind?.(foamDebugChk.checked);

    const foamDebugTexTitle = document.createElement("div");
    foamDebugTexTitle.textContent = "Foam debug";
    foamDebugTexTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const foamDebugTexSel = document.createElement("select");
    foamDebugTexSel.style.cssText = SELECT_STYLE;
    for (const o of [
        { value: "off", label: "Off (normal)" },
        { value: "accum", label: "Accumulation (raw)" },
        { value: "foamR", label: "Foam channel (R)" },
        { value: "bubbleG", label: "Bubble channel (G)" },
        { value: "sprayB", label: "Spray channel (B)" },
        { value: "blurred", label: "Blurred accumulation" },
        { value: "foamAlpha", label: "Foam alpha" },
        { value: "normals", label: "Fake normals" },
    ]) {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        foamDebugTexSel.appendChild(opt);
    }
    foamDebugTexSel.value = init.foam.debugTexture;
    foamDebugTexSel.onchange = () => on.onFoamDebugTexture?.(foamDebugTexSel.value as FoamDebugTexture);

    const foamControls: HTMLElement[] = [
        foamEnableRow,
        foamUsageRow,
        foamGenerateFoamRow,
        foamGenerateSprayRow,
        foamGenerateBubblesRow,
        foamSurfaceFilteringRow,
        foamKtaRow,
        foamKwcRow,
        foamAdvancedTitle,
        foamTurbulenceRateRow,
        foamEnergyMinRow,
        foamEnergyMaxRow,
        foamCurvatureMinRow,
        foamCurvatureMaxRow,
        foamTurbulenceMinRow,
        foamTurbulenceMaxRow,
        foamLayerDepthRow,
        foamSprayDragRow,
        foamLifeRow,
        foamBuoyRow,
        foamDragRow,
        foamSubRow,
        foamSubColorRow,
        foamPoolRow,
        foamSoftRow,
        foamDensityRow,
        foamSizeRow,
        foamBlurRow,
        foamLightRow,
        foamAmbientRow,
        foamAORow,
        // foamNormalRow is deliberately NOT listed: the fake normals it scales barely register
        // in the final image, so the slider was dead weight in the panel. The row is still
        // built and still round-trips through the foam values/presets — only the control is
        // hidden, so a preset written before this change still restores exactly.
        foamDebugRow,
        foamDebugTexTitle,
        foamDebugTexSel,
    ];
    const foamAdvancedControls = [
        foamAdvancedTitle,
        foamTurbulenceRateRow,
        foamEnergyMinRow,
        foamEnergyMaxRow,
        foamCurvatureMinRow,
        foamCurvatureMaxRow,
        foamTurbulenceMinRow,
        foamTurbulenceMaxRow,
        foamLayerDepthRow,
        foamSprayDragRow,
    ];
    applyFoamMethodVisibility = (): void => {
        const flip = currentMethod === "FLIP";
        for (const control of foamAdvancedControls) {
            control.hidden = !flip;
            control.style.display = flip ? "" : "none";
        }
    };
    applyFoamMethodVisibility();

    // ── Assemble the panel ──────────────────────────────────────────────────
    const root = document.createElement("div");
    root.style.cssText = opts.panelStyle ?? DEFAULT_PANEL_STYLE;
    if (opts.resizable !== false) {
        root.style.resize = "both";
        root.style.overflow = "auto";
        root.style.boxSizing = "border-box";
    }
    const demoSlot = document.createElement("div");
    root.appendChild(demoSlot);

    const generalItems: HTMLElement[] = [];
    if (!opts.hideMethod) {
        generalItems.push(methodTitle, methodSel);
    }
    if (opts.methods.includes("PB-MPM")) {
        generalItems.push(materialRow);
    }
    if (!opts.hideParticles) {
        generalItems.push(particlesRow);
    }
    if (opts.showSimulationTiming) {
        generalItems.push(simulationDurationRow, alphaDecayRow);
    }
    if (generalItems.length > 0) {
        root.append(...makeSection("General", generalItems));
    }

    // ── Tooltips ────────────────────────────────────────────────────────────
    // Attached after construction rather than threaded through every factory call, so the
    // explanations live in one readable table instead of being scattered across ~25 call
    // sites. Each row's FIRST <span> is its label (the value read-out is the second), and
    // that holds for the slider rows, the checkbox labels and the colour row alike.
    for (const [row, text] of [
        [colorRow, "Beer-Lambert absorption tint of the water body. This is the colour light is TINTED toward as it travels through the fluid, not a surface paint."],
        [
            sizeRow,
            "Visual radius multiplier for the impostors the surface is built from. Larger blobs merge into a smoother, fatter surface; smaller ones read as more separate droplets. Purely cosmetic — the simulation is unaffected.",
        ],
        [absorbRow, "How strongly the water colour saturates with depth. Higher makes thin sheets read as tinted and deep water go opaque; 0 leaves the fluid clear."],
        [
            polygonShaderRow,
            "Shading model for both screen-space and FLIP polygon surfaces. Physical refraction transmits the opaque scene through the water. Ocean PBR adapts Babylon.js Playground YX6IB8#758: opaque dark water with Beer-Lambert absorption, environment Fresnel, distance-dependent gloss, directional specular, and splash back-lighting.",
        ],
        [refractionRow, "How far the background is displaced when seen through the fluid, scaled by the water's thickness. 0 disables refraction."],
        [specularRow, "Tightness of the specular highlight. Higher values give a smaller, sharper glint; lower values spread it into a broad sheen."],
        [
            reflExposureRow,
            "Exposure applied to the environment reflection before it is mixed into the water. Match it to the scene's own exposure so the reflection is as bright as the sky it is reflecting; raise it for a brighter, more mirror-like sheet.",
        ],
        [
            reflContrastRow,
            "Contrast applied to the environment reflection, as a smoothstep about mid-grey. 1 leaves it linear; higher deepens the reflection's darks and brightens its highlights, matching the skybox's own contrast.",
        ],
        [
            reflectivityRow,
            "How reflective the water is when looked at HEAD-ON (Fresnel F0). Real water is about 0.02, so it only turns mirror-like at grazing angles; raising this makes it reflective from every angle, toward a chrome look.",
        ],
        [
            surfDepthBlurRow,
            "Radius of the blur applied to the fluid's depth buffer. This is the main smoothness control: low values leave a bumpy blob surface, high values give a calm, glassy one. At 0 the bilateral filter becomes a pass-through; narrow-range still applies its fixed 5\u00d75 clean-up pass.",
        ],
        [thickDownRow, "Resolution divisor for the thickness buffer. Thickness is low-frequency, so a higher divisor costs little visually and saves fill rate."],
        [
            surfFilterSel,
            "Algorithm used to smooth the depth buffer. Bilateral is a depth-weighted blur; Narrow-range is edge-aware and holds thin features and silhouettes better.",
        ],
        [
            surfDepthThreshRow,
            "BILATERAL ONLY. How far apart in depth two samples can be and still be blurred together. Lower preserves edges more sharply but leaves the surface noisier.",
        ],
        [nrDeltaRow, "NARROW-RANGE ONLY. Depth window, in impostor radii, that counts as the same surface. Wider smooths more but starts merging separate sheets of water."],
        [
            nrMuRow,
            "NARROW-RANGE ONLY. How far a front-facing outlier is clamped toward the centre sample, in impostor radii. Suppresses spikes from stray particles; 0 clamps them exactly onto the centre depth.",
        ],
        [surfThickBlurRow, "Radius of the blur applied to the thickness buffer, which drives absorption and refraction. 0 skips the pass entirely."],
        [halfRow, "Render the depth/thickness buffers at half resolution. Much cheaper, at the cost of a slightly softer surface and coarser silhouettes."],
        [anisoRow, "Stretch each particle's impostor along the local flow (Yu & Turk). Thin sheets and jets read as sheets instead of strings of beads, at extra GPU cost."],
        [anisoDampRow, "How strongly the anisotropic stretch is reined in. Lower allows longer, flatter ellipsoids; higher keeps them closer to spheres."],
        [renderRow, "Draw the raw particles as shaded spheres instead of building a fluid surface. Useful for seeing what the simulation is actually doing."],
        [
            foamKtaRow,
            "How much foam is generated by air being dragged under, e.g. where a jet plunges into a pool. The main source of churn in a waterfall. 0 turns this source off; with the wave-crest rate also at 0 no foam is created at all.",
        ],
        [
            foamKwcRow,
            "How much foam is generated at wave crests \u2014 sharply curved, fast-moving surfaces. Raise it for spray off breaking waves. 0 turns this source off; with the trapped-air rate also at 0 no foam is created at all.",
        ],
        [
            foamTurbulenceRateRow,
            "FLIP ONLY. Generation rate driven by vorticity and shear in the final MAC velocity field. 0 disables this advanced source and preserves the previous whitewater behaviour.",
        ],
        [foamEnergyMinRow, "FLIP ONLY. Fluid speed where all whitewater emission begins to pass the kinetic-energy gate."],
        [foamEnergyMaxRow, "FLIP ONLY. Fluid speed where the kinetic-energy gate reaches full strength."],
        [foamCurvatureMinRow, "FLIP ONLY. Dimensionless upward-surface curvature where wave-crest emission begins."],
        [foamCurvatureMaxRow, "FLIP ONLY. Dimensionless upward-surface curvature where wave-crest emission reaches full strength."],
        [foamTurbulenceMinRow, "FLIP ONLY. MAC-grid vorticity/shear magnitude where turbulence emission begins."],
        [foamTurbulenceMaxRow, "FLIP ONLY. MAC-grid vorticity/shear magnitude where turbulence emission reaches full strength."],
        [foamLayerDepthRow, "FLIP ONLY. Retains diffuse particles as surface foam this many MAC cells below an upward-facing free surface. 0 uses only the immediate interface."],
        [foamSprayDragRow, "FLIP ONLY. Exponential aerodynamic damping applied to detached spray. 0 leaves spray ballistic apart from gravity."],
        [foamLifeRow, "How long foam particles survive, in seconds. Longer leaves persistent trails and rafts of foam; shorter makes it flash and vanish."],
        [
            foamBuoyRow,
            "Upward acceleration on SUBMERGED foam (bubbles), as a fraction of gravity. Only affects particles currently classified as bubbles, so it does nothing in a scene without submerged churn.",
        ],
        [foamDragRow, "How strongly submerged bubbles are pulled toward the surrounding fluid's velocity. At 1 they simply follow the flow."],
        [
            foamPoolRow,
            "Size of the foam particle pool, as a multiple of the fluid particle count. Caps how much foam can exist at once. Changing it reallocates the buffer (32 bytes per slot), so the GPU panel's memory read-out moves with it.",
        ],
        [foamSoftRow, "Accumulation level at which foam starts to appear. Raise it to keep sparse spray from fogging the image."],
        [foamDensityRow, "Accumulation level at which foam becomes fully opaque. Bring it closer to the softness value for a harder, more defined foam edge."],
        [foamSubRow, "How visible bubbles below the surface are through the water. 0 hides the submerged component entirely."],
        [
            foamSubColorRow,
            "Tint of the submerged bubbles seen through the water. Independent of the surface foam, which stays white — use it to match the bubbles to the water colour.",
        ],
        [foamSizeRow, "Splat radius multiplier for foam particles. Larger merges foam into continuous sheets; smaller keeps it granular."],
        [foamBlurRow, "Blur radius applied to the foam accumulation buffer. Higher turns speckle into smooth mist. 0 skips the blur passes entirely."],
        [foamLightRow, "Strength of directional lighting on foam. Higher gives foam more shaded, three-dimensional relief. 0 leaves foam lit only by the ambient floor."],
        [foamAmbientRow, "Ambient light floor for foam, filling the parts the directional term leaves dark."],
        [foamAORow, "Self-shadowing within thick foam. Higher darkens the interior of dense clumps and adds depth. 0 disables the darkening."],
    ] as [HTMLElement, string][]) {
        row.querySelector("span")?.appendChild(infoIcon(text));
    }

    // Order here is the order in the panel: colour, then the particle/surface toggle pair, then
    // the surface-shading knobs, each blur next to the buffer it acts on, and the filter
    // immediately followed by the parameters that belong to it.
    const renderItems: HTMLElement[] = [colorRow, sizeRow];
    if (!opts.hideRenderAsSpheres) {
        renderItems.push(renderRow);
    }
    renderItems.push(
        polygonShaderRow,
        absorbRow,
        refractionRow,
        specularRow,
        reflExposureRow,
        reflContrastRow,
        reflectivityRow,
        surfDepthBlurRow,
        thickDownRow,
        surfThickBlurRow,
        surfFilterTitle,
        surfFilterSel,
        surfDepthThreshRow,
        nrDeltaRow,
        nrMuRow,
        halfRow,
        anisoRow,
        anisoDampRow
    );
    if (!opts.hideDebug) {
        renderItems.push(debugTitle, debugSel);
    }
    root.append(...makeSection("Render", renderItems));

    // Rows that only apply to ONE surface filter, hidden when the other one is selected: the
    // bilateral blur reads the edge threshold and ignores delta/mu, and the narrow-range
    // filter does the reverse (see writeBilateral / writeNarrow in fluid-surface-render).
    const applyFilterVisibility = (filter: string): void => {
        const narrow = filter === "narrowRange";
        setRowVisible(surfDepthThreshRow, !narrow);
        setRowVisible(nrDeltaRow, narrow);
        setRowVisible(nrMuRow, narrow);
    };
    /** The damping slider only means anything while the anisotropic surface is on. */
    const applyAnisoVisibility = (on: boolean): void => {
        setRowVisible(anisoDampRow, on);
    };
    // Single entry point, now that all three rules exist: hides the surface rows in spheres
    // mode and otherwise applies the filter + anisotropic rules.
    applySurfaceVisibility(init.renderMode === "spheres");

    // Surface-only rows (hidden in "Render as spheres" mode). Water color + particle size affect
    // both renderers and the spheres toggle itself must stay visible, so they are excluded.
    surfaceOnlyRows.push(
        polygonShaderRow,
        absorbRow,
        refractionRow,
        specularRow,
        reflExposureRow,
        reflContrastRow,
        reflectivityRow,
        surfDepthBlurRow,
        surfDepthThreshRow,
        surfThickBlurRow,
        surfFilterTitle,
        surfFilterSel,
        nrDeltaRow,
        nrMuRow,
        halfRow,
        anisoRow,
        anisoDampRow,
        thickDownRow
    );
    if (!opts.hideDebug) {
        surfaceOnlyRows.push(debugTitle, debugSel);
    }

    if (!opts.hideFoam) {
        root.append(...makeSection("Foam", foamControls));
    }
    if (!opts.hidePhysics) {
        // The "Physics particle size" row is dropped when the host owns its own particle-size
        // slider; the per-method sliders + reset stay.
        const activeBlockRows = [activeBlocksRow, pagedGridRow, pagedGridCapacityRow, pagedGridStatus, fusedBlockDiscoveryRow];
        const gridRows = opts.showGridControls
            ? [gridPositionControl.row, gridSizeControl.row, gridStatus, cellSizeRow, particleUsageRow, pressureDiagnosticsRow, gridBoundsRow, gridGizmoRow]
            : [particleUsageRow, pressureDiagnosticsRow];
        const physItems = opts.hidePhysScale
            ? [flipResolutionRow, flipMarkersRow, flipParticleCapacityRow, markerDensityStatus, ...gridRows, ...activeBlockRows, sliderHost, resetBtn]
            : [physRow, flipResolutionRow, flipMarkersRow, flipParticleCapacityRow, markerDensityStatus, ...gridRows, ...activeBlockRows, sliderHost, resetBtn];
        root.append(...makeSection("Physics simulation", physItems));
    }

    // ── GPU-timing panel (optional, pinned top-left) ────────────────────────
    let gpu: FluidGpuHandle | null = null;
    if (!opts.hideGpuTiming && opts.gpu) {
        const stages = opts.gpu.stages;
        const timingValueEls: Record<string, HTMLElement> = {};

        const makeSubHeader = (text: string): HTMLElement => {
            const h = document.createElement("div");
            h.textContent = text;
            h.style.cssText = "font-weight:700;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#7fb0e0;margin:10px 0 6px;";
            return h;
        };
        const makeStatRow = (label: string, valueEl?: HTMLElement): { row: HTMLElement; val: HTMLElement } => {
            const row = document.createElement("div");
            row.style.cssText = "display:flex;justify-content:space-between;margin:2px 0;font-variant-numeric:tabular-nums;";
            const lab = document.createElement("span");
            lab.textContent = label;
            const val = valueEl ?? document.createElement("span");
            if (!valueEl) {
                val.style.cssText = "color:#9fb4cc;";
                val.textContent = "\u2014";
            }
            row.append(lab, val);
            return { row, val };
        };
        const makeTimingRow = (label: string): HTMLElement => {
            const { row, val } = makeStatRow(label);
            row.dataset.fluidGpuStage = label;
            val.textContent = "\u2014 ms";
            timingValueEls[label] = val;
            return row;
        };

        const fpsLabel = document.createElement("div");
        fpsLabel.textContent = "\u2014";
        fpsLabel.style.cssText = "color:#7fd68a;font-weight:600;font-variant-numeric:tabular-nums;";

        const generalHeader = makeSubHeader("General");
        const fpsRow = makeStatRow("FPS", fpsLabel).row;
        const memRow = makeStatRow("Memory estimate");
        const memValueEl = memRow.val;
        memRow.row.title =
            "Active solver buffers plus estimated fluid render targets. This is not total graphics-card memory and excludes scene assets, inactive solver backends, and driver overhead.";

        // Fluid render-target / texture bytes estimate (dominant terms only). Reads the
        // component-owned half / thickness-downscale / foam-enabled controls.
        function estimateTextureBytes(w: number, h: number): number {
            const px = w * h;
            const dW = halfChk.checked ? Math.ceil(w / 2) : w;
            const dH = halfChk.checked ? Math.ceil(h / 2) : h;
            const surfaceDepth = dW * dH * (3 * 8 + 4);
            const td = Math.max(1, parseInt(thickDownInput.value, 10) || 1);
            const surfaceThick = Math.ceil(w / td) * Math.ceil(h / td) * (3 * 8);
            const foamBytes = foamEnabled ? px * (3 * 8) : 0;
            const polygonSurfaceEnabled = currentMethod === "FLIP" && ((schemas.FLIP ?? []).find((entry) => entry.key === "polygonSurface")?.value ?? 0) > 0.5;
            // RG32 eye depth, R32 back depth, and two full-resolution depth buffers.
            // Eye depth (rg32f), private front depth, and the full r32f Hi-Z mip chain.
            const polygonSurfaceBytes = polygonSurfaceEnabled ? px * 18 : 0;
            const sceneRT = px * (4 + 4);
            return surfaceDepth + surfaceThick + foamBytes + polygonSurfaceBytes + sceneRT;
        }

        const timingHeader = makeSubHeader("Timing");
        const timingNote = document.createElement("div");
        timingNote.style.cssText = "color:#7c8aa0;font-size:11px;margin:2px 0 6px;";
        const timingBody = document.createElement("div");
        for (const s of stages) {
            timingBody.appendChild(makeTimingRow(s));
        }
        timingBody.appendChild(makeTimingRow("Other"));
        const timingTotalRow = makeTimingRow("Total");
        timingTotalRow.style.cssText += "border-top:1px solid #2a3647;margin-top:4px;padding-top:4px;font-weight:700;";
        timingBody.appendChild(timingTotalRow);

        const panel = document.createElement("div");
        panel.style.cssText =
            "position:fixed;top:12px;left:12px;z-index:20;min-width:186px;font:12px system-ui,-apple-system,'Segoe UI',sans-serif;" +
            "color:#dfe6ee;background:rgba(10,14,20,0.85);padding:10px 12px;border-radius:8px;pointer-events:auto;user-select:none;";
        const panelTitle = document.createElement("div");
        panelTitle.textContent = "GPU";
        panelTitle.style.cssText = "font-weight:700;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#7fb0e0;margin-bottom:8px;";
        panel.append(panelTitle, generalHeader, fpsRow, memRow.row, timingHeader);
        if (opts.gpu.supported) {
            timingNote.textContent = "Per-stage GPU time (Chrome quantizes resolution).";
            panel.append(timingNote, timingBody);
        } else {
            timingNote.textContent = "GPU timing unavailable (no timestamp-query feature)";
            panel.append(timingNote);
        }

        gpu = {
            panel,
            fpsLabel,
            refreshMemory(simBytes: number, canvasW: number, canvasH: number): void {
                const bytes = simBytes + estimateTextureBytes(canvasW, canvasH);
                memValueEl.textContent = "~" + formatGpuBytes(bytes);
            },
            refreshTiming(res): void {
                if (!res) {
                    return;
                }
                for (const s of stages) {
                    timingValueEls[s]!.textContent = `${(res.stages[s] ?? 0).toFixed(2)} ms`;
                }
                timingValueEls["Other"]!.textContent = `${Math.max(0, res.frameTotal - res.total).toFixed(2)} ms`;
                timingValueEls["Total"]!.textContent = `${res.frameTotal.toFixed(2)} ms`;
            },
        };
    }

    return {
        root,
        demoSlot,
        containerToggleRow: opts.hideContainerToggle ? null : containerRow,
        makeSection,
        setSectionVisible(title: string, visible: boolean): void {
            const section = sections.get(title);
            if (!section) {
                return;
            }
            section[0].style.display = visible ? "" : "none";
            section[1].style.display = visible ? "" : "none";
        },

        setMethod(method: string): void {
            currentMethod = method;
            methodSel.value = method;
            applyMaterialVisibility();
            applyActiveBlocksVisibility();
            applyFlipControlVisibility();
            applyFoamMethodVisibility();
        },
        setMaterial(material: number): void {
            materialSel.value = String(material);
        },
        setParticleCount(count: number): void {
            ensureParticleCountOption(count);
            committedParticleCount = Math.max(0, Math.round(count));
            particlesControl.value = String(committedParticleCount);
            setParticleCapacity(count);
        },
        setParticleCountVisible(visible: boolean): void {
            particleCountVisible = visible;
            applyFlipControlVisibility();
        },
        setActiveParticleCount,
        setParticleUsage(activeCount: number, totalCount: number, gpuBytes: number, restartActiveCount?: number, restartTotalCount?: number, restartGpuBytes?: number): void {
            displayedActiveParticleCount = Math.max(0, Math.floor(activeCount));
            displayedParticleCount = Math.max(0, Math.floor(totalCount));
            displayedGpuBytes = Math.max(0, gpuBytes);
            restartParticleUsage =
                restartActiveCount !== undefined && restartTotalCount !== undefined && restartGpuBytes !== undefined
                    ? {
                          activeCount: Math.max(0, Math.floor(restartActiveCount)),
                          totalCount: Math.max(1, Math.floor(restartTotalCount)),
                          gpuBytes: Math.max(0, restartGpuBytes),
                      }
                    : null;
            updateParticleUsage();
        },
        setPolygonTriangleCount(count: number | undefined, visible: boolean): void {
            polygonTriangleCountValue.style.display = visible ? "block" : "none";
            polygonTriangleCountValue.textContent = visible
                ? count === undefined
                    ? "Triangles:\u00a0Calculating..."
                    : "Triangles:\u00a0" + formatParticleCount(Math.max(0, Math.floor(count)))
                : "";
        },
        setPressureDiagnostics(diagnostics: FluidPressureDiagnostics | undefined): void {
            displayedPressureDiagnostics = diagnostics;
            updatePressureDiagnostics();
        },
        setFoamParticleCounts(counts: DiffuseParticleCounts | undefined, enabled: boolean, capacity?: number): void {
            updateFoamParticleCounts(counts, enabled, capacity);
        },
        setSimulationDuration(seconds: number): void {
            simulationDurationRow.set(seconds);
        },
        setAlphaDecay(seconds: number): void {
            alphaDecayRow.set(seconds);
        },
        setRenderMode(spheres: boolean): void {
            renderChk.checked = spheres;
            on.onRenderMode?.(spheres);
            applySurfaceVisibility(spheres);
        },
        setPolygonShader(mode: "physical" | "ocean"): void {
            polygonShaderSelect.value = mode;
            on.onPolygonShader?.(mode);
        },
        setColor(hex: string): void {
            colorInput.value = hex;
            on.onColor?.(hexToRgb(hex));
        },
        setAbsorption(v: number): void {
            absorbInput.value = String(v);
            absorbVal.textContent = v.toFixed(1);
            on.onAbsorption?.(v);
        },
        setParticleSize(v: number): void {
            sizeInput.value = String(v);
            sizeVal.textContent = `${v.toFixed(2)}\u00d7`;
            on.onParticleSize?.(v);
        },
        setRefraction(v: number): void {
            refractionRow.set(v);
        },
        setSpecular(v: number): void {
            specularRow.set(v);
        },
        setReflection(exposure: number, contrast: number): void {
            // Each `set` fires its own onInput, which calls onReflection with BOTH tracked values —
            // so setting the exposure first would briefly push the old contrast. Harmless (the
            // second call corrects it immediately) and it keeps both rows and both tracked vars in
            // step, which is what getValues and the next slider drag read.
            reflExposureRow.set(exposure);
            reflContrastRow.set(contrast);
        },
        setReflectivity(v: number): void {
            reflectivityRow.set(v);
        },
        setDepthBlur(size: number, threshold: number): void {
            surfDepthBlurRow.set(size);
            surfDepthThreshRow.set(threshold);
        },
        setThicknessBlur(v: number): void {
            surfThickBlurRow.set(v);
        },
        setHalf(onFlag: boolean): void {
            halfChk.checked = onFlag;
            on.onHalf?.(onFlag);
        },
        setSurfaceFilter(m: "bilateral" | "narrowRange"): void {
            surfFilterSel.value = m;
            on.onSurfaceFilter?.(m);
        },
        setNarrowRange(delta: number, mu: number): void {
            nrDeltaRow.set(delta);
            nrMuRow.set(mu);
        },
        setAnisotropic(onFlag: boolean): void {
            anisoChk.checked = onFlag;
            on.onAnisotropic?.(onFlag);
        },
        setAnisotropySurfScale(v: number): void {
            anisoDampInput.value = String(v);
            anisoDampVal.textContent = v.toFixed(2);
            on.onAnisotropySurfScale?.(v);
        },
        setThicknessDownscale(v: number): void {
            thickDownInput.value = String(v);
            thickDownVal.textContent = `${v}\u00d7`;
            on.onThicknessDownscale?.(v);
        },
        setShowContainer(onFlag: boolean): void {
            containerChk.checked = onFlag;
        },
        setDebug(mode: string): void {
            debugSel.value = mode;
            on.onDebug?.(mode as FluidDebug);
        },
        setPhysics(schema: Record<string, number>): void {
            for (const p of schemas[currentMethod] ?? []) {
                if (schema[p.key] !== undefined) {
                    p.value = schema[p.key]!;
                }
            }
        },
        setPhysScale(scale: number): void {
            physInput.value = String(scale);
            physVal.textContent = `${scale.toFixed(2)}\u00d7`;
        },
        setGridResolution(resolution: number): void {
            gridResolution = Math.max(FLIP_GRID_RESOLUTION_MIN, Math.min(FLIP_GRID_RESOLUTION_MAX, Math.round(resolution)));
            flipResolutionInput.value = String(gridResolution);
            flipResolutionValue.textContent = String(gridResolution);
        },
        setMarkersPerCell(value: number): void {
            markersPerCell = Math.max(1, Math.min(64, Math.round(value)));
            flipMarkersInput.value = String(markersPerCell);
        },
        setFlipParticleCapacity(capacity: number): void {
            flipParticleCapacity = Math.max(1, Math.min(flipParticleCapacityMax, Math.round(capacity)));
            flipParticleCapacityInput.value = String(flipParticleCapacity);
        },
        setMarkerDensityWarning(message: string): void {
            markerDensityStatus.textContent = message;
            applyMarkerDensityStatusVisibility();
        },
        setGridSettings(position: [number, number, number], size: [number, number, number], cellSize: number): void {
            for (let i = 0; i < 3; i++) {
                gridPositionControl.inputs[i]!.value = String(position[i]);
                gridSizeControl.inputs[i]!.value = String(size[i]);
            }
            gridCellSize = cellSize;
            updateCellSizeValue();
            setGridStatus("");
        },
        setGridStatus,
        setShowGridBounds(visible: boolean): void {
            gridBoundsChk.checked = visible;
        },
        setActiveBlocks(enabled: boolean): void {
            activeBlocksChk.checked = enabled;
            applyActiveBlockDependencies();
        },
        setPagedGrid(enabled: boolean): void {
            pagedGridChk.checked = enabled;
            applyActiveBlockDependencies();
        },
        setPagedGridMaxPages(pages: number): void {
            pagedGridCapacityInput.value = String(Math.max(1, Math.min(2000, Math.round(pages / 1000))));
            updatePagedGridCapacityValue();
        },
        setPagedGridStatus(message: string, error = false): void {
            pagedGridStatus.textContent = message;
            pagedGridStatus.style.color = error ? "#ff8a80" : "#9fb3c8";
            pagedGridStatus.style.display = message && opts.showActiveBlocks && (currentMethod === "MLS-MPM" || currentMethod === "FLIP") ? "block" : "none";
        },
        setFusedBlockDiscovery(enabled: boolean): void {
            fusedBlockDiscoveryChk.checked = enabled;
        },
        setFoam(foam: FluidFoamValues): void {
            // Enable state + carried tMin + debug texture (set the DOM; the host re-pushes
            // the config to the sim via its applyFoam() after the sim rebuild).
            foamEnabled = foam.enabled;
            foamEnableChk.checked = foam.enabled;
            foamGenerateFoam = foam.generateFoam ?? true;
            foamGenerateSpray = foam.generateSpray ?? true;
            foamGenerateBubbles = foam.generateBubbles ?? true;
            foamSurfaceFiltering = foam.enabled && (foam.surfaceFiltering ?? currentMethod === "FLIP");
            foamGenerateFoamChk.checked = foamGenerateFoam;
            foamGenerateSprayChk.checked = foamGenerateSpray;
            foamGenerateBubblesChk.checked = foamGenerateBubbles;
            foamSurfaceFilteringChk.checked = foamSurfaceFiltering;
            on.onFoamSurfaceFiltering?.(foamSurfaceFiltering);
            applyFoamKindAvailability();
            foamTMin = foam.tMin;
            foamDebugTexSel.value = foam.debugTexture;
            on.onFoamDebugTexture?.(foam.debugTexture as FoamDebugTexture);
            // Config sliders (fire their effect callbacks — gated on "enabled" by the host).
            foamKtaRow.set(foam.kTa);
            foamKwcRow.set(foam.kWc);
            foamTurbulenceRateRow.set(foam.kTurb);
            foamEnergyMinRow.set(foam.energySpeedMin);
            foamEnergyMaxRow.set(foam.energySpeedMax);
            foamCurvatureMinRow.set(foam.curvatureMin);
            foamCurvatureMaxRow.set(foam.curvatureMax);
            foamTurbulenceMinRow.set(foam.turbulenceMin);
            foamTurbulenceMaxRow.set(foam.turbulenceMax);
            foamLayerDepthRow.set(foam.foamLayerDepth);
            foamSprayDragRow.set(foam.sprayDrag);
            foamBuoyRow.set(foam.kb);
            foamDragRow.set(foam.kd);
            foamLifeRow.set(foam.tMax);
            foamPoolRow.set(foam.poolScale);
            // Look sliders.
            foamSizeRow.set(foam.size);
            foamBlurRow.set(foam.blurRadius);
            foamLightRow.set(foam.lightIntensity);
            foamAmbientRow.set(foam.ambient);
            foamAORow.set(foam.aoStrength);
            foamNormalRow.set(foam.normalStrength);
            // Density (t1) BEFORE softness (t0) so the t0 ≤ t1 clamp restores the exact pair.
            foamDensityRow.set(foam.density);
            foamSoftRow.set(foam.softness);
            foamSubRow.set(foam.subsurfaceStrength);
            foamSubColorInput.value = foam.subsurfaceColor;
            on.onFoamSubColor?.(hexToRgb(foam.subsurfaceColor));
        },

        getValues(): FluidControlValues {
            const schema: Record<string, number> = {};
            for (const p of schemas[currentMethod] ?? []) {
                schema[p.key] = p.value;
            }
            return {
                method: currentMethod,
                material: parseInt(materialSel.value, 10),
                schema,
                simulationDuration: simulationDurationRow.get(),
                alphaDecay: alphaDecayRow.get(),
                color: colorInput.value,
                half: halfChk.checked,
                thicknessDownscale: parseInt(thickDownInput.value, 10),
                absorption: parseFloat(absorbInput.value),
                size: parseFloat(sizeInput.value),
                physScale: parseFloat(physInput.value),
                gridPosition: readGridVector(gridPositionControl.inputs),
                gridSize: readGridVector(gridSizeControl.inputs),
                cellSize: gridCellSize,
                gridResolution,
                markersPerCell,
                showGridBounds: gridBoundsChk.checked,
                count: currentMethod === "FLIP" ? flipParticleCapacity : committedParticleCount,
                renderMode: renderChk.checked ? "spheres" : "surface",
                polygonShader: polygonShaderSelect.value as "physical" | "ocean",
                refraction: surfRefraction,
                specular: surfSpecular,
                reflectionExposure: surfReflExposure,
                reflectionContrast: surfReflContrast,
                reflectivity: surfReflectivity,
                depthBlur: surfDepthFilter,
                depthBlurThreshold: surfDepthThreshold,
                thicknessBlur: surfThicknessBlur,
                surfaceFilter: surfFilterSel.value as "bilateral" | "narrowRange",
                narrowDelta: nrDelta,
                narrowMu: nrMu,
                anisotropic: anisoChk.checked,
                anisoSurfScale: parseFloat(anisoDampInput.value),
                activeBlocks: activeBlocksChk.checked,
                pagedGrid: pagedGridChk.checked,
                pagedGridMaxPages: parseInt(pagedGridCapacityInput.value, 10) * 1000,
                fusedBlockDiscovery: fusedBlockDiscoveryChk.checked,
                debug: debugSel.value,
                showContainer: containerChk.checked,
                foam: {
                    enabled: foamEnabled,
                    activeParticles: true,
                    generateSpray: foamGenerateSpray,
                    generateFoam: foamGenerateFoam,
                    generateBubbles: foamGenerateBubbles,
                    surfaceFiltering: foamSurfaceFiltering,
                    kTa: foamCfg.kTa,
                    kWc: foamCfg.kWc,
                    kTurb: foamCfg.kTurb,
                    energySpeedMin: foamCfg.energySpeedMin,
                    energySpeedMax: foamCfg.energySpeedMax,
                    curvatureMin: foamCfg.curvatureMin,
                    curvatureMax: foamCfg.curvatureMax,
                    turbulenceMin: foamCfg.turbulenceMin,
                    turbulenceMax: foamCfg.turbulenceMax,
                    foamLayerDepth: foamCfg.foamLayerDepth,
                    sprayDrag: foamCfg.sprayDrag,
                    kb: foamCfg.kb,
                    kd: foamCfg.kd,
                    tMin: foamTMin,
                    tMax: foamCfg.tMax,
                    poolScale: foamCfg.poolScale,
                    size: foamSize,
                    blurRadius: foamBlurRadius,
                    lightIntensity: foamLightIntensity,
                    ambient: foamAmbient,
                    aoStrength: foamAOStrength,
                    normalStrength: foamNormalStrength,
                    debugTexture: foamDebugTexSel.value,
                    softness: foamT0,
                    density: foamT1,
                    subsurfaceStrength: foamSubStrength,
                    subsurfaceColor: foamSubColorInput.value,
                },
            };
        },
        getPhysicsValues(method: string): Record<string, number> {
            const out: Record<string, number> = {};
            for (const p of schemas[method] ?? []) {
                out[p.key] = p.value;
            }
            return out;
        },
        rebuildPhysics(method: string): void {
            buildSliders(method);
            applyActiveBlocksVisibility();
        },
        setVisiblePhysicsParams(keys: string[] | null): void {
            visibleParamKeys = keys ? new Set(keys) : null;
            applyParamVisibility();
        },

        gpu,
    };
}
