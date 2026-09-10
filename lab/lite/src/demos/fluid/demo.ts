// Shared contract between the generic fluid CORE (fluid.ts) and the per-demo
// modules (scenes/*.ts). The core owns the engine/scene/camera, the shared
// scene-SDF UBO + hole ring, both sims + their lifecycle, the render tasks, the
// whole panel UI and the per-(demo, method) parameter store. Each demo is a
// plain data+behaviour object (a `FluidDemo`) built from a `FluidCtx` of the
// services the core hands it. No demo references the core module directly.

import type {
    ArcRotateCamera,
    DirectionalLight,
    EngineContext,
    FluidFlowConfig,
    FluidSimulation,
    ForceFieldSpec,
    HemisphericLight,
    Mat4,
    Mesh,
    SceneContext,
    SceneSdfSpec,
} from "babylon-lite";

/** Default (capsule / box) spawn box: a tall central column that drops in to
 *  fill the tank. The fountain overrides this with a wide shallow basin block. */
export const DEFAULT_SPAWN_MIN: [number, number, number] = [-2, 6, -2];
export const DEFAULT_SPAWN_MAX: [number, number, number] = [2, 12, 2];

// Per-demo HDR environment maps. Each demo declares which one it wants via
// {@link FluidDemo.envUrl}; the core loads both up front and swaps the skybox
// background + the fluid-surface reflection cube when the active demo changes.
/** Neutral studio HDR — used by the capsule / box / fountain / marble-tower demos. */
export const ENV_STUDIO_URL = "https://playground.babylonjs.com/textures/environment.env";
export const INTERACTIVE_FORCE_SAMPLE_HOLD_MS = 100;
// The waterfall's open-sky `.hdr` is declared next to that demo's own assets, as
// `WATERFALL_ENV_URL` in scenes/waterfall.ts.

// The plain-data authoring/preset types now live in the reusable fluid core so preset I/O,
// migrations, and method-independent carry-over can share them without importing this scene
// module. They are re-exported here unchanged so existing `./demo.js` imports keep working.
import type { DemoParam, DemoStateValue, FluidDomainBounds, FluidGridSettings, PairState } from "babylon-lite";
export type { DemoParam, DemoStateValue, FluidDomainBounds, FluidGridSettings, PairState };

/** Latest interactive push sample, kept briefly so deferred solver steps do not lose it. */
export interface PendingForce {
    origin: [number, number, number];
    dir: [number, number, number];
    push: [number, number, number];
    radius: number;
    accel: number;
    expiresAt: number;
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
    /** The currently-selected opaque simulation. */
    getActiveSimulation(): FluidSimulation;
    /** Rebind the active demo's current scene SDF through the shared facade. */
    rebindSceneSdf(): void;
    /** Re-seed the active sim (does NOT clear holes — call clearSceneHoles too). */
    resetActiveSim(): void;
    /** Replace the active pair's authored flow with the demo's current defaults. */
    refreshFlow(): void;
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
    /** Scale the active simulation domain and rebuild the active backend. Explicit grids scale their
     *  world-space position/size and Physics particle size together; gridless legacy demos retain
     *  their historical hidden domain multiplier. Returns whether a rebuild occurred. */
    setDomainScale(s: number): boolean;

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
    /** Demo-specific interaction text appended to the shared camera and simulation controls. */
    readonly helperText?: string;
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
    /** Preserve solver-independent authoring state when switching methods. The target method
     *  keeps its own specialized physics schema while shared state (including gravity) carries over. */
    readonly methodIndependentAuthoring?: boolean;
    /** Whether runtime quality preset files apply to this demo. Defaults to true. */
    readonly usesQualityPresets?: boolean;
    /** Use the editable grid's lower Y bound as the solver safety floor instead of world Y=0. */
    readonly useGridFloor?: boolean;
    /** Injected scene SDF. `gridConfine` affects the MLS-MPM backend ONLY (the PBF/SPH
     *  backend ignores it and always confines per-particle): for MLS-MPM, a spec with
     *  `gridConfine === false` uses per-particle push-out confinement (for thin curved
     *  shells like the capsule); otherwise (`gridConfine` true or absent) it's a CLOSED
     *  container confined at the grid with a separating wall. */
    readonly sdf: SceneSdfSpec;
    /** Pack this demo's static params into the shared UBO (offset 0 region). */
    writeSdfParams(): void;
    /** Default, solver-independent initial volumes, inflows and recycling sinks. */
    flow(): FluidFlowConfig;
    /** Optional notification after the generic authoring UI restores or edits the flow. */
    onFlowChanged?(flow: FluidFlowConfig): void;
    /** Persistent demo-local force field. The core's interactive mouse force temporarily overrides it. */
    forceField?(): ForceFieldSpec | null;
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
    /** Commit deferred work after a complete pair-state parameter restore and before the
     *  target simulation is resolved. Omit when every parameter applies synchronously. */
    commitRestoredParams?(): void;
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
