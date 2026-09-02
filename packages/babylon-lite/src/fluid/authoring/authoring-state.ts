// Plain-data fluid authoring/preset state shared by the reusable fluid core and the
// thin scene/orchestration layers that drive it.
//
// These types describe WHAT a fluid pair is (physics slider values, flow authoring,
// surface look, grid placement, foam knobs) with no reference to any engine/scene/GPU
// object — so preset I/O, migrations, and method-independent carry-over can all operate
// on them without importing a host module. The host-facing service interfaces that DO
// hold engine/scene/GPU handles (FluidCtx, FluidDemo) stay in the lab's scene layer and
// re-export these types unchanged.

import type { FluidEmitter, FluidSink } from "../core/sim-common.js";
import type { FluidSimulationSemantics } from "../core/simulation-config.js";

// A demo exposes a typed list of live tunables; the host's "Demo parameters"
// section renders a control per type (number -> slider, boolean -> checkbox,
// color -> picker) and calls the demo's handler on change.
/** A `hidden` param keeps its place in the pair-state bag - so preset files still drive it and
 *  it round-trips when switching between (demo, method) pairs - but gets no control in the
 *  panel. Use it to retire a knob from the UI without freezing its value. */
export type DemoParam = { hidden?: boolean } & (
    | { key: string; label: string; type: "number"; min: number; max: number; step: number; value: number }
    | { key: string; label: string; type: "boolean"; value: boolean }
    | { key: string; label: string; type: "color"; value: string }
);

/** A value a demo may stash in its opaque `demoState` bag.
 *
 *  Strings are allowed alongside numbers and booleans so a demo can pin a NAMED choice rather than
 *  encode it as a magic index - the waterfall's "Background detail" tier rides here as
 *  `"low" | "mid" | "high"`, which keeps the exported JSON readable and, more importantly, stable:
 *  an index would silently re-point at a different tier the day the list is reordered. */
export type DemoStateValue = number | boolean | string;

export interface FluidDomainBounds {
    min: [number, number, number];
    max: [number, number, number];
}

export interface FluidGridSettings {
    /** World-space center of the axis-aligned simulation grid. */
    position: [number, number, number];
    /** Exact world-space extent along X/Y/Z. */
    size: [number, number, number];
}

// Per-(demo, simulation) parameter snapshot. The host stores one of these per
// (demo, method) pair so e.g. SPH-fountain and MLS-fountain keep independent
// values. `demoParams` is a generic bag captured from `FluidDemo.demoParams()`
// (numeric params only) and restored via `FluidDemo.applyParam()`.
export interface PairState {
    /** Explicit interpretation of authored solver values. Inferred once for legacy files. */
    simulationSemantics?: FluidSimulationSemantics;
    /** Physics-slider values for the pair's method (core-owned SCHEMAS). */
    schema: Record<string, number>;
    /** Generic per-demo tunables (empty for demos with no `demoParams`). */
    demoParams: Record<string, number>;
    /** Simulated seconds before the fluid starts fading. Zero runs indefinitely. */
    simulationDuration?: number;
    /** Seconds taken to fade fluid opacity to zero after the duration. */
    alphaDecay?: number;
    /** Multiplier applied to real frame time before stepping the simulation. */
    simulationTimeScale?: number;
    /** Solver-independent fluid sources, stored in grid-local coordinates. */
    emitters?: FluidEmitter[];
    /** Solver-independent recycling volumes, stored in grid-local coordinates. */
    sinks?: FluidSink[];
    /** Fill the full particle capacity from initial emitters even when inflows exist. */
    initialEmittersFillCapacity?: boolean;
    /** Legacy presets without explicit flow arrays rebuild the demo-owned graph after restoring demo state. */
    legacyFlow?: boolean;
    color: string;
    half: boolean;
    /** Thickness-texture downscale factor (thickness size = canvas / factor). */
    thicknessDownscale: number;
    absorption: number;
    size: number;
    physScale: number;
    /** Explicit simulation grid in world space. Omitted by legacy presets that use demo-authored bounds. */
    grid?: FluidGridSettings;
    /** Whether the active simulation-domain wireframe is visible. */
    showGridBounds?: boolean;
    /** Whether visible simulation-domain bounds use transparent depth-tested faces. */
    showGridBoundsSolid?: boolean;
    /** Legacy format v3-or-earlier simulation-domain AABB. */
    domain?: FluidDomainBounds;
    /** Legacy format v3-or-earlier divisions along the longest domain axis. */
    gridResolution?: number;
    /** FLIP marker sampling density. Eight markers form a 2 x 2 x 2 sub-cell layout. */
    markersPerCell?: number;
    count: number;
    /** PB-MPM material enum: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. */
    material?: number;
    /** Optional authored ArcRotate camera framing. */
    camera?: { alpha: number; beta: number; radius: number; target?: [number, number, number] };
    /** Optional authored FreeCamera pose. Kept separate because position/target do not map to ArcRotate alpha/beta/radius. */
    freeCamera?: { position: [number, number, number]; target: [number, number, number] };
    // -- Surface-render settings (per-pair, restored on switch). All optional so old
    //    presets/states without them fall back to the core render defaults. --
    /** Opt into profile-specific rendering when this simulation is combined with others. */
    independentRendering?: boolean;
    renderMode?: "surface" | "spheres";
    polygonShader?: "physical" | "ocean";
    refraction?: number;
    specular?: number;
    /** Reflection tonemap (exposure + contrast) applied to the environment reflection, and the
     *  water's Fresnel reflectance at normal incidence. Part of the surface LOOK a preset pins. */
    reflectionExposure?: number;
    reflectionContrast?: number;
    reflectivity?: number;
    depthBlur?: number;
    depthBlurThreshold?: number;
    thicknessBlur?: number;
    /** Surface depth smoother: "bilateral" (default) or "narrowRange" (Truong-Yuksel
     *  Narrow-Range Filter), plus its delta/mu params (x impostor size). Per-pair so the
     *  choice is captured in export and restored on switch. */
    surfaceFilter?: "bilateral" | "narrowRange";
    narrowDelta?: number;
    narrowMu?: number;
    /** Anisotropic surface (Yu & Turk ellipsoidal splatting). Default OFF; per-pair. */
    anisotropic?: boolean;
    /** Anisotropic WPCA radius damping (0..1 share of surfaceSizeScale). Default 0.5; per-pair. */
    anisoSurfScale?: number;
    /** MLS-MPM active-block execution. Optional and defaults off. */
    activeBlocks?: boolean;
    /** Sparse bounded grid-page storage for FLIP or MLS-MPM. */
    pagedGrid?: boolean;
    /** Maximum live grid pages (8 cubed FLIP cells or 4 cubed MLS-MPM nodes). */
    pagedGridMaxPages?: number;
    /** Append active particle blocks directly during histogram construction. */
    fusedBlockDiscovery?: boolean;
    /** Foam (diffuse-particle) config for this pair - generation, pool and screen-space
     *  look. When present, foam is restored (enabled/disabled) with these knobs on switch. */
    foam?: {
        enabled: boolean;
        activeParticles?: boolean;
        generateSpray?: boolean;
        generateFoam?: boolean;
        generateBubbles?: boolean;
        /** Strict reconstructed-surface rejection for foam and spray rendering. */
        surfaceFiltering?: boolean;
        kTa: number;
        kWc: number;
        kTurb?: number;
        energySpeedMin?: number;
        energySpeedMax?: number;
        curvatureMin?: number;
        curvatureMax?: number;
        turbulenceMin?: number;
        turbulenceMax?: number;
        foamLayerDepth?: number;
        sprayDrag?: number;
        kb: number;
        kd: number;
        tMin: number;
        tMax: number;
        poolScale: number;
        blurRadius: number;
        lightIntensity: number;
        ambient: number;
        aoStrength: number;
        normalStrength: number;
        debugTexture: string;
        /** Screen-space froth softness/coverage thresholds ("Foam softness t0" /
         *  "Foam density t1") and the subsurface-bubble tint strength. Optional so
         *  presets/states predating them fall back to the core foam defaults. */
        softness?: number;
        density?: number;
        subsurfaceStrength?: number;
        /** Submerged-bubble tint as an sRGB hex string. Optional so presets/states
         *  predating it fall back to the pale blue that used to be hardcoded. */
        subsurfaceColor?: string;
        /** Visual foam splat-size multiplier ("Foam size"). Optional so presets/states
         *  predating it fall back to the core foam default (1). */
        size?: number;
    };
    /** Opaque per-demo extra-control state (box: size / paddle), captured via
     *  {@link FluidDemo.snapshotState} and restored via {@link FluidDemo.restoreState}. */
    demoState?: Record<string, DemoStateValue>;
    /** Whether the demo's container / nozzle meshes are shown ("Show container" toggle). */
    showContainer?: boolean;
    /** Image-based-lighting multiplier ("Environment intensity"). Optional so presets/states
     *  predating it fall back to the shader's own 1.0. Core-owned rather than per-demo, but it
     *  is pinned per pair because it is part of the exported LOOK. */
    envIntensity?: number;
    /** 4x MSAA on the scene pass ("Anti-aliasing"). Optional, defaults off. */
    msaa?: boolean;
    /** Unknown/forward-compatible preset data preserved recursively through import then export.
     *  The tree may contain top-level fields, unknown children of known sections, and shared
     *  host-owned sections that this PairState does not model. Known exported values always win
     *  when the tree is merged back. Never surfaced as a control. */
    forwardCompatibleFields?: Record<string, unknown>;
}
