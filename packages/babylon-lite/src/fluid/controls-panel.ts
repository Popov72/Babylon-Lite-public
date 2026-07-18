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

/** One per-method physics slider definition (mirrors the fluid demo's `SCHEMAS`). */
export interface PhysSchemaEntry {
    key: string;
    label: string;
    min: number;
    max: number;
    step: number;
    value: number;
}

/**
 * Default per-method physics-slider definitions shared by every fluid app. Each
 * entry mirrors the corresponding solver default and remembers the last value the
 * user set. Both the fluid demo and the Liquefactor demo pass this as `schemas`
 * (the component deep-copies it, so the shared constant is never mutated). Kept as
 * an exported constant so any host can reuse the exact same PBF / MLS-MPM tunables.
 */
export const DEFAULT_FLUID_SCHEMAS: Record<string, PhysSchemaEntry[]> = {
    PBF: [
        { key: "gravity", label: "Gravity", min: 0, max: 200, step: 0.1, value: 9.8 },
        { key: "viscosity", label: "Viscosity (XSPH)", min: 0, max: 1, step: 0.005, value: 0.08 },
        { key: "relaxation", label: "Relaxation \u03b5", min: 1, max: 300, step: 1, value: 50 },
        { key: "scorr", label: "Artificial pressure", min: 0, max: 0.1, step: 0.001, value: 0.02 },
        { key: "iterations", label: "Solver iterations", min: 1, max: 8, step: 1, value: 3 },
        { key: "restDensity", label: "Rest density", min: 100, max: 2000, step: 10, value: 341 },
        { key: "boundaryDensity", label: "Boundary density", min: 0, max: 1, step: 0.05, value: 0 },
    ],
    "MLS-MPM": [
        { key: "gravity", label: "Gravity", min: 0, max: 200, step: 0.1, value: 9.8 },
        { key: "stiffness", label: "Stiffness (EOS)", min: 10, max: 5000, step: 10, value: 350 },
        { key: "viscosity", label: "Viscosity", min: 0, max: 1, step: 0.01, value: 0.3 },
        { key: "restDensity", label: "Rest density (/cell)", min: 1, max: 100, step: 0.5, value: 3 },
        { key: "damping", label: "Velocity damping", min: 0.9, max: 1, step: 0.001, value: 0.995 },
        { key: "affineDamping", label: "Affine damping (\u2192PIC)", min: 0.1, max: 1, step: 0.005, value: 0.9 },
        { key: "groundDamp", label: "Ground damping", min: 0.7, max: 1, step: 0.01, value: 0.85 },
        { key: "groundDampHeight", label: "Ground damp height", min: 0, max: 10, step: 0.1, value: 1.5 },
        { key: "restitution", label: "Restitution (bounce)", min: 0, max: 1, step: 0.05, value: 0.3 },
        { key: "substeps", label: "Substeps / frame", min: 1, max: 8, step: 1, value: 3 },
    ],
};

/** Full foam (diffuse-particle) look/config snapshot. */
export interface FluidFoamValues {
    enabled: boolean;
    kTa: number;
    kWc: number;
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
}

/** Snapshot of every control the component owns (for pair-state capture / export). */
export interface FluidControlValues {
    method: string;
    schema: Record<string, number>;
    color: string;
    half: boolean;
    thicknessDownscale: number;
    absorption: number;
    size: number;
    physScale: number;
    count: number;
    renderMode: "surface" | "spheres";
    refraction: number;
    specular: number;
    depthBlur: number;
    depthBlurThreshold: number;
    thicknessBlur: number;
    surfaceFilter: "bilateral" | "narrowRange";
    narrowDelta: number;
    narrowMu: number;
    debug: string;
    showContainer: boolean;
    foam: FluidFoamValues;
}

/** Initial value for every control. */
export interface FluidControlsInitial {
    method: string;
    count: number;
    physScale: number;
    color: string;
    absorption: number;
    size: number;
    refraction: number;
    specular: number;
    depthBlur: number;
    depthBlurThreshold: number;
    thicknessBlur: number;
    half: boolean;
    thicknessDownscale: number;
    surfaceFilter: "bilateral" | "narrowRange";
    narrowDelta: number;
    narrowMu: number;
    renderMode: "surface" | "spheres";
    debug: string;
    showContainer: boolean;
    foam: FluidFoamValues;
}

/** Host effect callbacks — the component fires these; the host applies the effect. */
export interface FluidControlsCallbacks {
    onMethod?(method: string): void;
    onParticleCount?(count: number): void;
    onRenderMode?(spheres: boolean): void;
    onColor?(rgb: [number, number, number]): void;
    onAbsorption?(v: number): void;
    onParticleSize?(v: number): void;
    onRefraction?(v: number): void;
    onSpecular?(v: number): void;
    onDepthBlur?(size: number, threshold: number): void;
    onThicknessBlur?(v: number): void;
    onHalf?(on: boolean): void;
    onSurfaceFilter?(m: "bilateral" | "narrowRange"): void;
    onNarrowRange?(delta: number, mu: number): void;
    onThicknessDownscale?(v: number): void;
    onShowContainer?(visible: boolean): void;
    onDebug?(mode: FluidDebug): void;
    onPhysicsParam?(key: string, value: number): void;
    onPhysScale?(scale: number): void;
    onReset?(): void;
    // Foam config (generation) — gated on "enabled" by the host.
    onFoamEnable?(enabled: boolean): void;
    onFoamKta?(v: number): void;
    onFoamKwc?(v: number): void;
    onFoamLifetime?(v: number): void;
    onFoamBuoyancy?(v: number): void;
    onFoamDrag?(v: number): void;
    onFoamPool?(v: number): void;
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
    /** When true, the "Physics simulation" section OMITS the "Physics particle size"
     *  control (title + slider) but KEEPS the per-method sliders + reset button. Use
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
    /** Initial value for every control. */
    initial: FluidControlsInitial;
    /** "Physics particle size" slider range (defaults 0.5 … 3). */
    physScaleMin?: number;
    physScaleMax?: number;
    /** Host effect callbacks. */
    on: FluidControlsCallbacks;
    /** Override the outer panel `cssText` (default = the fluid demo's right-side panel). */
    panelStyle?: string;
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

    // ── Programmatic setters (see the module contract for which fire callbacks) ──
    setMethod(method: string): void;
    setParticleCount(count: number): void;
    setRenderMode(spheres: boolean): void;
    setColor(hex: string): void;
    setAbsorption(v: number): void;
    setParticleSize(v: number): void;
    setRefraction(v: number): void;
    setSpecular(v: number): void;
    setDepthBlur(size: number, threshold: number): void;
    setThicknessBlur(v: number): void;
    setHalf(on: boolean): void;
    setSurfaceFilter(m: "bilateral" | "narrowRange"): void;
    setNarrowRange(delta: number, mu: number): void;
    setThicknessDownscale(v: number): void;
    setShowContainer(on: boolean): void;
    setDebug(mode: string): void;
    setPhysics(schema: Record<string, number>): void;
    setPhysScale(scale: number): void;
    setFoam(foam: FluidFoamValues): void;

    /** Snapshot every control value (for pair-state capture / export). */
    getValues(): FluidControlValues;
    /** Physics-slider values for a given method (source of truth for applying to the sim). */
    getPhysicsValues(method: string): Record<string, number>;
    /** Rebuild the physics-slider block for a method (called on method change). */
    rebuildPhysics(method: string): void;

    /** GPU-timing panel handle, or null when hidden / not configured. */
    gpu: FluidGpuHandle | null;
}

/** Default right-side panel style (the fluid demo's). */
const DEFAULT_PANEL_STYLE =
    "position:fixed;top:12px;right:12px;z-index:20;width:248px;max-height:calc(100vh - 24px);overflow-y:auto;font:12px system-ui,-apple-system,'Segoe UI',sans-serif;" +
    "color:#dfe6ee;background:rgba(10,14,20,0.85);padding:10px 12px;border-radius:8px;pointer-events:auto;user-select:none;";
const SELECT_STYLE = "width:100%;margin-bottom:8px;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";

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

    // ── Labelled render-slider helper (mirrors the fluid demo's makeRenderSlider). ──
    type RenderSliderRow = HTMLDivElement & { set(v: number): void };
    function makeRenderSlider(label: string, min: number, max: number, step: number, value: number, fmt: (v: number) => string, onInput: (v: number) => void): RenderSliderRow {
        const row = document.createElement("div") as RenderSliderRow;
        row.style.cssText = "margin:2px 0 8px;";
        const head = document.createElement("div");
        head.style.cssText = "display:flex;justify-content:space-between;";
        const lab = document.createElement("span");
        lab.textContent = label;
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
        row.append(head, input);
        return row;
    }

    // ── Collapsible section builder (shared header styling). ──
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
        return [h, body];
    };

    // ── GENERAL: method + particles ─────────────────────────────────────────
    const methodTitle = document.createElement("div");
    methodTitle.textContent = "Fluid method";
    methodTitle.style.cssText = "font-weight:600;margin-bottom:6px;";
    const methodSel = document.createElement("select");
    methodSel.style.cssText = SELECT_STYLE;
    for (const name of opts.methods) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name === "PBF" ? "SPH (PBF)" : name;
        methodSel.appendChild(opt);
    }
    methodSel.value = init.method;
    methodSel.onchange = () => on.onMethod?.(methodSel.value);

    const particlesTitle = document.createElement("div");
    particlesTitle.textContent = "Particles";
    particlesTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const particlesSel = document.createElement("select");
    particlesSel.style.cssText = SELECT_STYLE;
    for (const c of opts.particleCounts) {
        const opt = document.createElement("option");
        opt.value = String(c);
        // "750k" below 1M and "1M" / "1.5M" at/above so larger counts read sensibly.
        opt.textContent = c >= 1000000 ? `${(c / 1000000).toFixed(c % 1000000 === 0 ? 0 : 1)}M` : `${(c / 1000).toFixed(0)}k`;
        if (c === init.count) {
            opt.selected = true;
        }
        particlesSel.appendChild(opt);
    }
    particlesSel.onchange = () => on.onParticleCount?.(parseInt(particlesSel.value, 10));

    // ── RENDER controls ─────────────────────────────────────────────────────
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
    surfFilterSel.onchange = () => on.onSurfaceFilter?.(surfFilterSel.value as "bilateral" | "narrowRange");
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
    thickDownInput.max = "8";
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
    renderChk.onchange = () => on.onRenderMode?.(renderChk.checked);

    // Surface debug (feature) dropdown.
    const debugTitle = document.createElement("div");
    debugTitle.textContent = "Debug (feature)";
    debugTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const debugSel = document.createElement("select");
    debugSel.style.cssText = SELECT_STYLE;
    for (const o of [
        { value: "none", label: "None (final render)" },
        { value: "depth", label: "Depth" },
        { value: "depthBlur", label: "Depth (blurred)" },
        { value: "thickness", label: "Thickness" },
        { value: "thicknessBlur", label: "Thickness (blurred)" },
        { value: "normals", label: "Normals" },
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
    const physTitle = document.createElement("div");
    physTitle.textContent = "Physics particle size";
    physTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const physRow = document.createElement("div");
    physRow.style.cssText = "margin:2px 0 8px;";
    const physHead = document.createElement("div");
    physHead.style.cssText = "display:flex;justify-content:flex-end;";
    const physVal = document.createElement("span");
    physVal.style.cssText = "color:#9fb4cc;";
    physVal.textContent = `${init.physScale.toFixed(1)}\u00d7`;
    physHead.append(physVal);
    const physInput = document.createElement("input");
    physInput.type = "range";
    physInput.min = String(physMin);
    physInput.max = String(physMax);
    physInput.step = "0.1";
    physInput.value = String(init.physScale);
    physInput.style.cssText = "width:100%;";
    physInput.oninput = () => {
        physVal.textContent = `${parseFloat(physInput.value).toFixed(1)}\u00d7`;
    };
    physInput.onchange = () => on.onPhysScale?.(parseFloat(physInput.value));
    physRow.append(physHead, physInput);

    const sliderHost = document.createElement("div");
    function buildSliders(name: string): void {
        sliderHost.replaceChildren();
        for (const p of schemas[name] ?? []) {
            const row = document.createElement("div");
            row.style.cssText = "margin:6px 0;";
            const head = document.createElement("div");
            head.style.cssText = "display:flex;justify-content:space-between;";
            const lab = document.createElement("span");
            lab.textContent = p.label;
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
            sliderHost.appendChild(row);
        }
    }
    buildSliders(currentMethod);

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset simulation";
    resetBtn.style.cssText = "width:100%;margin-top:8px;padding:5px;cursor:pointer;background:#26415f;color:#eef3f8;border:1px solid #3a567a;border-radius:4px;";
    resetBtn.onclick = () => on.onReset?.();

    // ── FOAM controls ───────────────────────────────────────────────────────
    let foamEnabled = init.foam.enabled;
    let foamTMin = init.foam.tMin;
    // Foam generation config (mirrors the fluid demo's foamCfg fields).
    const foamCfg = {
        kTa: init.foam.kTa,
        kWc: init.foam.kWc,
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
        on.onFoamEnable?.(foamEnabled);
    };
    const foamKtaRow = makeRenderSlider(
        "Trapped-air rate k_ta",
        0,
        120,
        1,
        foamCfg.kTa,
        (v) => String(Math.round(v)),
        (v) => {
            foamCfg.kTa = v;
            on.onFoamKta?.(v);
        }
    );
    const foamKwcRow = makeRenderSlider(
        "Wave-crest rate k_wc",
        0,
        120,
        1,
        foamCfg.kWc,
        (v) => String(Math.round(v)),
        (v) => {
            foamCfg.kWc = v;
            on.onFoamKwc?.(v);
        }
    );
    const foamLifeRow = makeRenderSlider(
        "Foam lifetime (s)",
        0.3,
        6,
        0.1,
        foamCfg.tMax,
        (v) => v.toFixed(1),
        (v) => {
            foamCfg.tMax = v;
            on.onFoamLifetime?.(v);
        }
    );
    const foamBuoyRow = makeRenderSlider(
        "Bubble buoyancy k_b",
        0,
        3,
        0.05,
        foamCfg.kb,
        (v) => v.toFixed(2),
        (v) => {
            foamCfg.kb = v;
            on.onFoamBuoyancy?.(v);
        }
    );
    const foamDragRow = makeRenderSlider(
        "Bubble drag k_d",
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
        "Foam softness t0 (edge)",
        0,
        2,
        0.02,
        foamT0,
        (v) => v.toFixed(2),
        (v) => {
            foamT0 = Math.min(v, foamT1 - 0.02);
            on.onFoamThresholds?.(foamT0, foamT1);
        }
    );
    const foamDensityRow = makeRenderSlider(
        "Foam density t1 (opaque)",
        0.2,
        50,
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
        1,
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
        foamKtaRow,
        foamKwcRow,
        foamLifeRow,
        foamBuoyRow,
        foamDragRow,
        foamPoolRow,
        foamSoftRow,
        foamDensityRow,
        foamSubRow,
        foamSizeRow,
        foamBlurRow,
        foamLightRow,
        foamAmbientRow,
        foamAORow,
        foamNormalRow,
        foamDebugRow,
        foamDebugTexTitle,
        foamDebugTexSel,
    ];

    // ── Assemble the panel ──────────────────────────────────────────────────
    const root = document.createElement("div");
    root.style.cssText = opts.panelStyle ?? DEFAULT_PANEL_STYLE;
    const demoSlot = document.createElement("div");
    root.appendChild(demoSlot);

    const generalItems: HTMLElement[] = [];
    if (!opts.hideMethod) {
        generalItems.push(methodTitle, methodSel);
    }
    if (!opts.hideParticles) {
        generalItems.push(particlesTitle, particlesSel);
    }
    if (generalItems.length > 0) {
        root.append(...makeSection("General", generalItems));
    }

    const renderItems: HTMLElement[] = [
        colorRow,
        absorbRow,
        sizeRow,
        refractionRow,
        specularRow,
        surfDepthBlurRow,
        surfDepthThreshRow,
        surfThickBlurRow,
        surfFilterTitle,
        surfFilterSel,
        nrDeltaRow,
        nrMuRow,
        halfRow,
        thickDownRow,
    ];
    if (!opts.hideRenderAsSpheres) {
        renderItems.push(renderRow);
    }
    if (!opts.hideDebug) {
        renderItems.push(debugTitle, debugSel);
    }
    root.append(...makeSection("Render", renderItems));

    if (!opts.hideFoam) {
        root.append(...makeSection("Foam", foamControls));
    }
    if (!opts.hidePhysics) {
        // The "Physics particle size" control (physTitle + physRow) is dropped when the
        // host owns its own particle-size slider; the per-method sliders + reset stay.
        const physItems = opts.hidePhysScale ? [sliderHost, resetBtn] : [physTitle, physRow, sliderHost, resetBtn];
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
            val.textContent = "\u2014 ms";
            timingValueEls[label] = val;
            return row;
        };

        const fpsLabel = document.createElement("div");
        fpsLabel.textContent = "\u2014";
        fpsLabel.style.cssText = "color:#7fd68a;font-weight:600;font-variant-numeric:tabular-nums;";

        const generalHeader = makeSubHeader("General");
        const fpsRow = makeStatRow("FPS", fpsLabel).row;
        const memRow = makeStatRow("Memory");
        const memValueEl = memRow.val;

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
            const sceneRT = px * (4 + 4);
            return surfaceDepth + surfaceThick + foamBytes + sceneRT;
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
                memValueEl.textContent = `~${(bytes / 1048576).toFixed(1)} MB`;
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

        setMethod(method: string): void {
            currentMethod = method;
            methodSel.value = method;
        },
        setParticleCount(count: number): void {
            particlesSel.value = String(count);
        },
        setRenderMode(spheres: boolean): void {
            renderChk.checked = spheres;
            on.onRenderMode?.(spheres);
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
            physVal.textContent = `${scale.toFixed(1)}\u00d7`;
        },
        setFoam(foam: FluidFoamValues): void {
            // Enable state + carried tMin + debug texture (set the DOM; the host re-pushes
            // the config to the sim via its applyFoam() after the sim rebuild).
            foamEnabled = foam.enabled;
            foamEnableChk.checked = foam.enabled;
            foamTMin = foam.tMin;
            foamDebugTexSel.value = foam.debugTexture;
            on.onFoamDebugTexture?.(foam.debugTexture as FoamDebugTexture);
            // Config sliders (fire their effect callbacks — gated on "enabled" by the host).
            foamKtaRow.set(foam.kTa);
            foamKwcRow.set(foam.kWc);
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
        },

        getValues(): FluidControlValues {
            const schema: Record<string, number> = {};
            for (const p of schemas[currentMethod] ?? []) {
                schema[p.key] = p.value;
            }
            return {
                method: currentMethod,
                schema,
                color: colorInput.value,
                half: halfChk.checked,
                thicknessDownscale: parseInt(thickDownInput.value, 10),
                absorption: parseFloat(absorbInput.value),
                size: parseFloat(sizeInput.value),
                physScale: parseFloat(physInput.value),
                count: parseInt(particlesSel.value, 10),
                renderMode: renderChk.checked ? "spheres" : "surface",
                refraction: surfRefraction,
                specular: surfSpecular,
                depthBlur: surfDepthFilter,
                depthBlurThreshold: surfDepthThreshold,
                thicknessBlur: surfThicknessBlur,
                surfaceFilter: surfFilterSel.value as "bilateral" | "narrowRange",
                narrowDelta: nrDelta,
                narrowMu: nrMu,
                debug: debugSel.value,
                showContainer: containerChk.checked,
                foam: {
                    enabled: foamEnabled,
                    kTa: foamCfg.kTa,
                    kWc: foamCfg.kWc,
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
        },

        gpu,
    };
}
