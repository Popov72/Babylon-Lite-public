// Shared contract between the generic fluid CORE (fluid.ts) and the per-demo
// modules (scenes/*.ts). The core owns the engine/scene/camera, the shared
// scene-SDF UBO + hole ring, both sims + their lifecycle, the render tasks, the
// whole panel UI and the per-(demo, method) parameter store. Each demo is a
// plain data+behaviour object (a `FluidDemo`) built from a `FluidCtx` of the
// services the core hands it. No demo references the core module directly.

import type { ArcRotateCamera, DirectionalLight, EngineContext, HemisphericLight, Mat4, Mesh, SceneContext } from "babylon-lite";
import type { EmitterConfig, FluidProfiler, FluidSim, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";

/** Default (capsule / box) spawn box: a tall central column that drops in to
 *  fill the tank. The fountain overrides this with a wide shallow basin block. */
export const DEFAULT_SPAWN_MIN: [number, number, number] = [-2, 6, -2];
export const DEFAULT_SPAWN_MAX: [number, number, number] = [2, 12, 2];

// Per-demo HDR environment maps. Each demo declares which one it wants via
// {@link FluidDemo.envUrl}; the core loads both up front and swaps the skybox
// background + the fluid-surface reflection cube when the active demo changes.
/** Neutral studio HDR — used by the capsule / box / fountain / marble-tower demos. */
export const ENV_STUDIO_URL = "https://playground.babylonjs.com/textures/environment.env";
// The waterfall's open-sky `.hdr` is declared next to that demo's own assets, as
// `WATERFALL_ENV_URL` in scenes/waterfall.ts.

// A demo exposes a typed list of live tunables; the core's "Demo parameters"
// section renders a control per type (number → slider, boolean → checkbox,
// color → picker) and calls the demo's handler on change.
/** A `hidden` param keeps its place in the pair-state bag — so preset files still drive it and
 *  it round-trips when switching between (demo, method) pairs — but gets no control in the
 *  panel. Use it to retire a knob from the UI without freezing its value. */
export type DemoParam = { hidden?: boolean } & (
    | { key: string; label: string; type: "number"; min: number; max: number; step: number; value: number }
    | { key: string; label: string; type: "boolean"; value: boolean }
    | { key: string; label: string; type: "color"; value: string }
);

/** A value a demo may stash in its opaque `demoState` bag.
 *
 *  Strings are allowed alongside numbers and booleans so a demo can pin a NAMED choice rather than
 *  encode it as a magic index — the waterfall's "Background detail" tier rides here as
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

// Per-(demo, simulation) parameter snapshot. The core stores one of these per
// (demo, method) pair so e.g. SPH-fountain and MLS-fountain keep independent
// values. `demoParams` is a generic bag captured from `FluidDemo.demoParams()`
// (numeric params only) and restored via `FluidDemo.applyParam()`.
export interface PairState {
    /** Physics-slider values for the pair's method (core-owned SCHEMAS). */
    schema: Record<string, number>;
    /** Generic per-demo tunables (empty for demos with no `demoParams`). */
    demoParams: Record<string, number>;
    color: string;
    half: boolean;
    /** Thickness-texture downscale factor (thickness size = canvas / factor). */
    thicknessDownscale: number;
    absorption: number;
    size: number;
    physScale: number;
    /** Explicit simulation grid in world space. Omitted by legacy presets that use demo-authored bounds. */
    grid?: FluidGridSettings;
    /** Legacy format <=3 simulation-domain AABB. */
    domain?: FluidDomainBounds;
    /** Legacy format <=3 divisions along the longest domain axis. */
    gridResolution?: number;
    count: number;
    /** PB-MPM material enum: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. */
    material?: number;
    /** Optional ArcRotate camera framing (alpha/beta/radius). A demo's preset can set
     *  it to frame the scene on the first visit to that (demo, method) pair; it is also
     *  captured live so switching pairs remembers each one's viewpoint. */
    camera?: { alpha: number; beta: number; radius: number };
    // ── Surface-render settings (per-pair, restored on switch). All optional so old
    //    presets/states without them fall back to the core render defaults. ──
    renderMode?: "surface" | "spheres";
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
     *  Narrow-Range Filter), plus its δ/µ params (× impostor size). Per-pair so the
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
    /** Sparse bounded grid-page storage. */
    pagedGrid?: boolean;
    /** Maximum live 4³-cell grid pages. */
    pagedGridMaxPages?: number;
    /** Append active particle blocks directly during histogram construction. */
    fusedBlockDiscovery?: boolean;
    /** Foam (diffuse-particle) config for this pair — generation, pool and screen-space
     *  look. Present → foam is restored (enabled/disabled) with these knobs on switch. */
    foam?: {
        enabled: boolean;
        activeParticles?: boolean;
        kTa: number;
        kWc: number;
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
    /** 4× MSAA on the scene pass ("Anti-aliasing"). Optional, defaults off. */
    msaa?: boolean;
}

/** Interactive push force applied for exactly one frame (box mouse-stir). */
export interface PendingForce {
    origin: [number, number, number];
    dir: [number, number, number];
    push: [number, number, number];
    radius: number;
    accel: number;
}

// Services the core gives each demo. Demos read/write the shared scene-SDF UBO
// (offset 0 region) and drive collision/interaction through these helpers; the
// core owns the concrete engine/sim state behind them.
export interface FluidCtx {
    readonly engine: EngineContext;
    readonly scene: SceneContext;
    readonly canvas: HTMLCanvasElement;
    readonly camera: ArcRotateCamera;
    /** Shared ground plane the escaping liquid falls onto (hidden by the box). */
    readonly ground: Mesh;
    /** Shared scene-SDF uniform buffer. Demos pack their params into offset 0..;
     *  the hole ring (offset 32) is managed by the core via addSceneHole/clear. */
    readonly sceneSdfBuffer: GPUBuffer;
    /** The currently-selected backend (PBF, MLS-MPM or PB-MPM). */
    getActiveSim(): FluidSim;
    /** Re-seed the active sim (does NOT clear holes — call clearSceneHoles too). */
    resetActiveSim(): void;
    /** Push new emitters (from the active demo) to BOTH backends. */
    refreshEmitters(): void;
    /** Re-read the active demo's `spawn()` and push it (plus its warm-up) to every backend.
     *  Needed when a demo's seed volume is only known asynchronously — the waterfall derives
     *  its from a height map that lands after the demo is already on screen, and without this
     *  a following `resetActiveSim()` would re-seed against the stale volume. */
    refreshSpawn(): void;
    /** Carve a drain hole into the scene-SDF hole ring (offset 32 region). */
    addSceneHole(center: [number, number, number], radius: number): void;
    /** Clear all drain holes (zero the hole ring). */
    clearSceneHoles(): void;
    /** Directional "sun" light (always in the scene). It only carries a shadow generator while a
     *  demo has explicitly enabled shadows via {@link setSunShadows}; otherwise no shadow map is
     *  rendered at all. Its `direction` may be written at runtime — the generator recomputes its
     *  light matrix (and re-fits the ortho box to the casters) on every shadow-map render. */
    readonly sun: DirectionalLight;
    /** Hemispheric ambient fill, shared by every demo. It is there for the standard-material
     *  demos, which sample no environment map — a PBR demo under a full HDR IBL gets ambient
     *  from the IBL already, so for those this is a SECOND ambient term, and one no shadow can
     *  attenuate. A demo that wants readable shadows should dim it in `onEnter` and put it back
     *  in `onLeave`. `intensity` is a plain field with no observer, so a write only reaches the
     *  lights UBO once something bumps the light version — set `direction` (ObservableVec3.set
     *  always fires) in the same breath. */
    readonly ambient: HemisphericLight;
    /** Attach or detach the sun's shadow map, with the meshes that should cast into it.
     *  The generator itself stays permanently attached to the light (see fluid.ts) — this only
     *  swaps the caster list, which is what makes the toggle free when off. Demos MUST call
     *  `setSunShadows(false, [])` in `onLeave`, or their casters keep drawing into the map
     *  under the next demo. */
    setSunShadows(on: boolean, casters: Mesh[]): void;

    /** Largest absolute X/Z coordinate of the active fluid-sim domain.
     *  Exposed so a demo can size a pump intake to the whole simulated floor without
     *  hard-coding the core's bounds: particles that drift outside the intake can never be
     *  recycled, and since the domain wall stops them they pile up against it forever. */
    readonly simHalfExtentXZ: number;
    /** View-projection matrix for screen picking / rays. */
    viewProjection(): Mat4;
    /** The active GPU timing profiler (or null when timing is off / unsupported). A demo
     *  that encodes its own passes can tag them by forwarding this to their encode() so
     *  they appear in the GPU panel. */
    getProfiler(): FluidProfiler | null;
    /** Scale the active simulation domain and rebuild both backends. Explicit grids scale their
     *  world-space position/size and Physics particle size together; gridless legacy demos retain
     *  their historical hidden domain multiplier. */
    setDomainScale(s: number): void;

    /** Configure the shared bloom post-process. The whole fluid chain composites into an
     *  offscreen target, which is then presented to the swapchain either through bloom or a
     *  plain blit — so `enabled: false` costs nothing beyond that blit. `intensity` is the
     *  merge weight (how much glow is added on top) and `threshold` is the luminance above
     *  which a pixel starts to bloom. Demo-scoped: a demo that offers bloom must turn it off
     *  again in `onLeave`, since the stage itself is shared by every demo. */
    setBloom(cfg: { enabled: boolean; intensity: number; threshold: number }): void;
}

// A single fluid demo (capsule / box / fountain). The core drives the active
// demo through this interface — it contains no demo-specific branches itself.
export interface FluidDemo {
    /** Stable id used for pair-state keys and the dropdown value. */
    readonly key: string;
    /** Dropdown label. */
    readonly label: string;
    /** HDR environment this demo shows as its skybox background AND reflects in the
     *  fluid surface (one of {@link ENV_STUDIO_URL} / the waterfall's own
     *  `WATERFALL_ENV_URL`). The
     *  core loads both up front and swaps to this one when the demo becomes active. */
    readonly envUrl: string;
    /** Key of the environment-picker entry this demo defaults to (see the core's
     *  `ENV_CHOICES`). Omit → the studio environment. A picker selection overrides it. */
    readonly envKey?: string;
    /** Environment yaw this demo wants, in degrees. Applied (and shown on the
     *  "Environment rotation" slider) whenever the demo becomes active, so a scene can
     *  aim the sun/horizon of its backdrop. Omit → 0°. */
    readonly envRotationDeg?: number;
    /** Method this demo should open on the FIRST time it is picked from the dropdown (e.g. the
     *  waterfall is authored around PB-MPM). Omit to carry the current method over. Later
     *  visits keep whatever the user last chose, and the method/quality selectors themselves
     *  are never overridden. */
    readonly defaultMethod?: string;
    /** Quality tier this demo should open on the first time it is picked. Omit to carry the
     *  current tier over. Same first-visit-only rule as {@link defaultMethod}. */
    readonly defaultQuality?: "low" | "middle" | "high";
    /** Injected scene SDF. `gridConfine` affects the MLS-MPM backend ONLY (the PBF/SPH
     *  backend ignores it and always confines per-particle): for MLS-MPM, a spec with
     *  `gridConfine === false` uses per-particle push-out confinement (for thin curved
     *  shells like the capsule); otherwise (`gridConfine` true or absent) it's a CLOSED
     *  container confined at the grid with a separating wall. */
    readonly sdf: SceneSdfSpec;
    /** Pack this demo's static params into the shared UBO (offset 0 region). */
    writeSdfParams(): void;
    /** Re-seed spawn box used by the sim's reset(). `accept`, when returned,
     *  restricts seeding (CPU reject-sampling) so particles fit a non-box shape. */
    spawn(): { min: [number, number, number]; max: [number, number, number]; accept?: (x: number, y: number, z: number) => boolean; warmupFrames?: number };
    /** Recirculating jet emitters (fountain), or null. */
    emitters(): EmitterConfig | null;
    /** Entering this demo: show meshes, set camera mode, etc. */
    onEnter(): void;
    /** Leaving this demo: hide meshes, undo camera mode / any force. */
    onLeave(): void;
    /** Per-frame hook (box: spin paddle + write paddle SDF block). */
    update(dt: number): void;
    /** Whether asynchronously loaded assets required for deterministic capture are ready. */
    isReady?(): boolean;
    /** Optional world/domain scale for the fluid-sim bounds (the marble tower's "Mesh scale").
     *  The core reads this on every `switchPair` (default 1 when omitted) and, if it differs
     *  from the currently-built domain scale, rebuilds the sims with scaled bounds. Demos that
     *  do not resize their world omit it → the sims always use the base bounds. */
    getDomainScale?(): number;
    /** Live tunables shown in "Demo parameters" (empty if none). */
    demoParams(): DemoParam[];
    /** Apply a live demo-param change. */
    applyParam(key: string, value: number | boolean | string): void;
    /** Demo-specific panel controls appended under "Physics simulation". */
    extraControls(): HTMLElement[];
    /** Meshes this demo casts shadows from. Informational: the core does not call it during
     *  startup — caster sets are registered through {@link setSunShadows} and preloaded
     *  lazily. Omit → this demo never casts shadows. */
    shadowCasters?(): Mesh[];
    /** Show/hide this demo's container / nozzle meshes (box glass, capsule shell,
     *  or fountain nozzle spouts). Optional: demos without any such mesh omit it. */
    setContainerVisible?(visible: boolean): void;
    /** Translucent container meshes (alpha < 1, no depth write) that must be drawn
     *  as a post-fluid OVERLAY into the swapchain instead of the offscreen scene-
     *  colour target — otherwise the fluid surface composites over them. Opaque
     *  container meshes (e.g. the box paddle, fountain nozzles) write depth and are
     *  correctly handled by the normal scene pass, so they are NOT returned here.
     *  Omit → this demo has no translucent overlay meshes. The core wires an overlay
     *  render task from these and strips them out of the scene-colour pass. */
    containerMeshes?(): Mesh[];
    /** Snapshot this demo's extra-control state (box: size/paddle) as a flat bag. */
    snapshotState?(): Record<string, DemoStateValue>;
    /** Restore extra-control state; MUST also update the extra-control UI to match. */
    restoreState?(state: Record<string, DemoStateValue>): void;

    /** Return true if this demo will handle the pointerdown itself, so the
     *  built-in arc-camera control should ignore it (e.g. capsule: LMB over the
     *  tank punches a hole instead of rotating). Omit → camera always handles. */
    claimsPointer?(e: PointerEvent): boolean;
    onPointerDown?(e: PointerEvent): void;
    onPointerMove?(e: PointerEvent): void;
    onPointerUp?(e: PointerEvent): void;
    onKey?(e: KeyboardEvent): void;
}
