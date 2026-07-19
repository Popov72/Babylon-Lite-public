// Shared contract between the generic fluid CORE (fluid.ts) and the per-demo
// modules (scenes/*.ts). The core owns the engine/scene/camera, the shared
// scene-SDF UBO + hole ring, both sims + their lifecycle, the render tasks, the
// whole panel UI and the per-(demo, method) parameter store. Each demo is a
// plain data+behaviour object (a `FluidDemo`) built from a `FluidCtx` of the
// services the core hands it. No demo references the core module directly.

import type { ArcRotateCamera, DirectionalLight, EngineContext, Mat4, Mesh, SceneContext, ShadowGenerator } from "babylon-lite";
import type { EmitterConfig, FluidProfiler, FluidSim, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";

/** Default (capsule / box) spawn box: a tall central column that drops in to
 *  fill the tank. The fountain overrides this with a wide shallow basin block. */
export const DEFAULT_SPAWN_MIN: [number, number, number] = [-2, 6, -2];
export const DEFAULT_SPAWN_MAX: [number, number, number] = [2, 12, 2];

// Per-demo HDR environment maps. Each demo declares which one it wants via
// {@link FluidDemo.envUrl}; the core loads BOTH up front and swaps the skybox
// background + the fluid-surface reflection cube when the active demo changes.
/** Neutral studio HDR — used by the capsule / box / fountain demos. */
export const ENV_STUDIO_URL = "https://playground.babylonjs.com/textures/environment.env";
/** Green countryside HDR — used by the waterfall demo (also feeds its terrain IBL). */
export const ENV_COUNTRY_URL = "https://playground.babylonjs.com/textures/country.env";

// A demo exposes a typed list of live tunables; the core's "Demo parameters"
// section renders a control per type (number → slider, boolean → checkbox,
// color → picker) and calls the demo's handler on change.
export type DemoParam =
    | { key: string; label: string; type: "number"; min: number; max: number; step: number; value: number }
    | { key: string; label: string; type: "boolean"; value: boolean }
    | { key: string; label: string; type: "color"; value: string };

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
    count: number;
    /** Optional ArcRotate camera framing (alpha/beta/radius). A demo's preset can set
     *  it to frame the scene on the first visit to that (demo, method) pair; it is also
     *  captured live so switching pairs remembers each one's viewpoint. */
    camera?: { alpha: number; beta: number; radius: number };
    // ── Surface-render settings (per-pair, restored on switch). All optional so old
    //    presets/states without them fall back to the core render defaults. ──
    renderMode?: "surface" | "spheres";
    refraction?: number;
    specular?: number;
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
    /** Foam (diffuse-particle) config for this pair — generation, pool and screen-space
     *  look. Present → foam is restored (enabled/disabled) with these knobs on switch. */
    foam?: {
        enabled: boolean;
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
        /** Visual foam splat-size multiplier ("Foam size"). Optional so presets/states
         *  predating it fall back to the core foam default (1). */
        size?: number;
    };
    /** Opaque per-demo extra-control state (box: size / paddle), captured via
     *  {@link FluidDemo.snapshotState} and restored via {@link FluidDemo.restoreState}. */
    demoState?: Record<string, number | boolean>;
    /** Whether the demo's container / nozzle meshes are shown ("Show container" toggle). */
    showContainer?: boolean;
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
    /** The currently-selected backend (PBF or MLS-MPM). */
    getActiveSim(): FluidSim;
    /** Re-seed the active sim (does NOT clear holes — call clearSceneHoles too). */
    resetActiveSim(): void;
    /** Push new emitters (from the active demo) to BOTH backends. */
    refreshEmitters(): void;
    /** Carve a drain hole into the scene-SDF hole ring (offset 32 region). */
    addSceneHole(center: [number, number, number], radius: number): void;
    /** Clear all drain holes (zero the hole ring). */
    clearSceneHoles(): void;
    /** Directional "sun" light (always in the scene). A demo that wants cast shadows
     *  attaches `sunShadow` to it on enter (`sun.shadowGenerator = ctx.sunShadow`) and
     *  detaches it on leave (`sun.shadowGenerator = undefined`) so other demos never
     *  render a shadow map. */
    readonly sun: DirectionalLight;
    /** Sun (directional) CSM shadow generator, created up-front but attached to the
     *  sun only while a shadow-casting demo is active. Register caster meshes via
     *  `setShadowTaskCasterMeshes(ctx.sunShadow, ...)` in `onEnter`. */
    readonly sunShadow: ShadowGenerator;
    /** View-projection matrix for screen picking / rays. */
    viewProjection(): Mat4;
    /** The active GPU timing profiler (or null when timing is off / unsupported). A demo
     *  that encodes its own passes can tag them by forwarding this to their encode() so
     *  they appear in the GPU panel. */
    getProfiler(): FluidProfiler | null;
    /** Set the fluid-sim DOMAIN (world) scale. Rebuilds BOTH backends with the sim bounds,
     *  grid cell `dx`, particle/smoothing radius and spawn box all multiplied by `s` — so the
     *  grid dimensions (bounds/dx) stay constant and GPU memory is unchanged while the domain
     *  physically grows/shrinks. Re-applies the active demo's scene SDF. 1 = base domain. Used
     *  by the marble-tower "Mesh scale" slider so a larger tower gets a proportionally larger
     *  water domain instead of hitting the fixed-grid cap. */
    setDomainScale(s: number): void;
}

// A single fluid demo (capsule / box / fountain). The core drives the active
// demo through this interface — it contains no demo-specific branches itself.
export interface FluidDemo {
    /** Stable id used for pair-state keys and the dropdown value. */
    readonly key: string;
    /** Dropdown label. */
    readonly label: string;
    /** HDR environment this demo shows as its skybox background AND reflects in the
     *  fluid surface (one of {@link ENV_STUDIO_URL} / {@link ENV_COUNTRY_URL}). The
     *  core loads both up front and swaps to this one when the demo becomes active. */
    readonly envUrl: string;
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
    /** Meshes this demo wants the sun to cast CSM shadows from (terrain + boulders in
     *  the waterfall). The core collects these BEFORE registerScene to warm up the
     *  shadow-caster pipeline preload; the demo itself re-registers them (and attaches
     *  `ctx.sunShadow` to `ctx.sun`) in onEnter. Omit → this demo casts no shadows. */
    shadowCasters?(): Mesh[];
    /** Snapshot this demo's extra-control state (box: size/paddle) as a flat bag. */
    snapshotState?(): Record<string, number | boolean>;
    /** Restore extra-control state; MUST also update the extra-control UI to match. */
    restoreState?(state: Record<string, number | boolean>): void;
    /** Return true if this demo will handle the pointerdown itself, so the
     *  built-in arc-camera control should ignore it (e.g. capsule: LMB over the
     *  tank punches a hole instead of rotating). Omit → camera always handles. */
    claimsPointer?(e: PointerEvent): boolean;
    onPointerDown?(e: PointerEvent): void;
    onPointerMove?(e: PointerEvent): void;
    onPointerUp?(e: PointerEvent): void;
    onKey?(e: KeyboardEvent): void;
}
