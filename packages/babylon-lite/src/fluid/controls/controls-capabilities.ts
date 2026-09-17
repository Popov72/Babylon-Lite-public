// Shared fluid-controls capability contract.
//
// Which control groups a fluid host may expose depends on the selected backend method (and, for
// timing, on the device). Historically each host inlined scattered `method === "FLIP"` /
// `method === "PB-MPM"` conditionals across its panel, so the same capability question was answered
// in many places with no single source of truth. This resolver states the answer ONCE, as plain
// data, so a host can disable/hide unsupported controls uniformly instead of rendering visible
// no-ops. The method facts here are aligned with `fluidAllocationCapabilities` (paging / polygon /
// pressure diagnostics) and extend it with the UI-only groups (material, active blocks, FLIP tuning,
// independent rendering, timing, grid visuals, authoring surfaces).

/** Inputs the controls-capability contract depends on. */
export interface FluidControlsCapabilitiesInput {
    /** Selected backend method: "PBF", "FLIP", "MLS-MPM" or "PB-MPM". */
    method: string;
    /** Whether the device exposes the "timestamp-query" feature that the GPU timing panel needs.
     *  Defaults to true (assume available; the host refines it with the real device capability). */
    timestampQuerySupported?: boolean;
    /** Capabilities implemented by the embedding host. Backend support is intersected with these
     *  values; omitted entries default to supported for backward compatibility. */
    hostCapabilities?: Partial<FluidControlsCapabilities>;
}

/** Per-(method, device) support for each fluid control group. `true` means the host should render
 *  the control; `false` means it must be hidden/disabled rather than shown as a no-op. */
export interface FluidControlsCapabilities {
    /** Sparse bounded page-storage controls (FLIP or MLS-MPM). */
    readonly pagedGrid: boolean;
    /** GPU polygon-surface reconstruction toggle (FLIP only). */
    readonly polygonSurface: boolean;
    /** Diffuse-particle generation and foam-render controls. */
    readonly foam: boolean;
    /** Physics parameter keys accepted by an implementation override. null means every intrinsic
     *  method parameter remains available. */
    readonly physicsParameters: readonly string[] | null;
    /** Per-simulation independent render-profile control (every backend). */
    readonly independentRendering: boolean;
    /** GPU timing panel (requires device timestamp-query support). */
    readonly timing: boolean;
    /** Simulation-domain grid visuals: bounds wireframe and placement gizmo (every backend). */
    readonly gridVisuals: boolean;
    /** Flow authoring surfaces: the emitter/sink editor (every backend). */
    readonly authoringSurfaces: boolean;
    /** PB-MPM material selector (PB-MPM only). */
    readonly material: boolean;
    /** MLS-MPM active-block execution controls (MLS-MPM only). */
    readonly activeBlocks: boolean;
    /** FLIP-only tuning: grid resolution, markers-per-cell, pressure solver, advanced whitewater. */
    readonly flipTuning: boolean;
    /** Asynchronous pressure-residual diagnostics (FLIP only). */
    readonly pressureDiagnostics: boolean;
}

export interface FluidRuntimeCapabilities {
    /** Whether the host attaches the FLIP polygon mesh to its render graph. */
    readonly polygonSurface: boolean;
    /** Whether the host supports per-simulation surface render groups. */
    readonly independentRendering: boolean;
}

export interface FluidCapabilityRequest {
    readonly method: string;
    readonly physics?: Readonly<Record<string, number>>;
    readonly independentRendering?: boolean;
}

/** Explain why declarative fluid state cannot run on a host, or return null when supported. */
export function fluidRuntimeCapabilityRejection(capabilities: FluidRuntimeCapabilities, request: FluidCapabilityRequest): string | null {
    if (request.method === "FLIP" && (request.physics?.polygonSurface ?? 0) >= 0.5 && !capabilities.polygonSurface) {
        return "enables FLIP polygon-surface reconstruction, which this host does not render";
    }
    if (request.independentRendering === true && !capabilities.independentRendering) {
        return "requires independent fluid rendering, which this host does not support";
    }
    return null;
}

/** Resolve the control groups a fluid host should expose for a backend method and device. */
export function resolveFluidControlsCapabilities(input: FluidControlsCapabilitiesInput): FluidControlsCapabilities {
    const flip = input.method === "FLIP";
    const backend: FluidControlsCapabilities = {
        pagedGrid: flip || input.method === "MLS-MPM",
        polygonSurface: flip,
        foam: true,
        physicsParameters: null,
        pressureDiagnostics: flip,
        flipTuning: flip,
        material: input.method === "PB-MPM",
        activeBlocks: input.method === "MLS-MPM",
        independentRendering: true,
        gridVisuals: true,
        authoringSurfaces: true,
        timing: input.timestampQuerySupported ?? true,
    };
    const host = input.hostCapabilities;
    return {
        pagedGrid: backend.pagedGrid && host?.pagedGrid !== false,
        polygonSurface: backend.polygonSurface && host?.polygonSurface !== false,
        foam: backend.foam && host?.foam !== false,
        physicsParameters: host?.physicsParameters ? [...host.physicsParameters] : backend.physicsParameters,
        pressureDiagnostics: backend.pressureDiagnostics && host?.pressureDiagnostics !== false,
        flipTuning: backend.flipTuning && host?.flipTuning !== false,
        material: backend.material && host?.material !== false,
        activeBlocks: backend.activeBlocks && host?.activeBlocks !== false,
        independentRendering: backend.independentRendering && host?.independentRendering !== false,
        gridVisuals: backend.gridVisuals && host?.gridVisuals !== false,
        authoringSurfaces: backend.authoringSurfaces && host?.authoringSurfaces !== false,
        timing: backend.timing && host?.timing !== false,
    };
}
