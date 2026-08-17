// Aquanova user-facing graphics settings.
//
// These are the knobs a player gets to turn to make the demo run on their machine, kept apart from
// the demo's internal tuning constants (constants.ts) because they are *chosen at runtime*: read on
// load, changed while playing, and persisted. `main.ts` owns the wiring — a setting here only says
// what exists, what it defaults to and how it is described; nothing in this file touches the GPU.
//
// The Aquanova control panel uses these definitions for persisted graphics options.

export const LIQUEFACTOR_MODELS = ["20k", "80k", "350k"] as const;
export type LiquefactorModel = (typeof LIQUEFACTOR_MODELS)[number];

/** Everything the player can tune. Add fields as the control panel grows. */
export interface GraphicsSettings {
    /** 4× MSAA on the scene (ship geometry) pass. Costs a canvas-sized 4-sample colour + depth pair. */
    msaa: boolean;
    /** SMAA post-process on the final image. Catches the aliasing MSAA cannot: the hard lines inside
     *  the ship's panel/grate textures, which are most of what reads as "jaggy" here. */
    smaa: boolean;
    /** PBR specular anti-aliasing. Increases effective roughness where the screen-space normal
     *  changes rapidly, reducing metallic reflection shimmer without blurring the whole frame. */
    specularAA: boolean;
    /** TAA — jittered temporal accumulation. Supersamples everything, including inside textures, but
     *  core TAA resets on camera movement, so today it only pays off while standing still. */
    taa: boolean;
    /** SSAA — `1` is off, `2` renders at twice the display dimensions and downscales. */
    ssaa: number;
    /** Geometry detail used by the first-person Liquefactor viewmodel. */
    liquefactorModel: LiquefactorModel;
    /** Cosmetic low-frequency balancing motion on the held weapon. */
    weaponSway: boolean;
    /** Gameplay sound effects for pickups and the Liquefactor. */
    soundsEnabled: boolean;
    /** Blend the two nearest local cubemaps. Disabled uses one dominant box-projected probe. */
    localCubemapBlending: boolean;
    /** Render through an sRGB swapchain view so the GPU encodes linear→sRGB on store.
     *
     *  MEASURED, and the answer is "don't": this demo's image-processing stage already outputs
     *  display-encoded values, so the sRGB view applies the transfer function a SECOND time — the
     *  image washes out (mean luma 19.6 → 70.2, blacks lift to grey). It also makes no difference to
     *  anti-aliasing: measured in linear space, every AA mode scored within 0.9 points of its
     *  non-sRGB result (MSAA 4.9/4.9, SMAA 4.3/4.3, TAA 13.9/13.5, SSAA 2x 55.3/54.4 % better than
     *  no AA). Kept only as a diagnostic (`?srgb=1`); leave it off. */
    srgb: boolean;
}

export const DEFAULT_GRAPHICS: GraphicsSettings = {
    // On by default: the ship is a hard-edged modular kit — panel seams, railings and door frames are
    // exactly the near-vertical high-contrast edges MSAA fixes, and they are most of the screen. The
    // cost is one extra canvas-sized 4× colour+depth pair, which any device that can already run the
    // GPU fluid can afford. Weak devices turn it off from the control panel (or ?msaa=0).
    msaa: true,
    // On by default for the complementary half of the problem: measurement on this content showed
    // most of the residual error is INSIDE textures (straight lines in the panel/grate art), which
    // MSAA cannot touch because it only supersamples polygon coverage. SMAA is an image-space
    // filter, so it treats those the same as a silhouette. Three cheap fullscreen passes.
    smaa: true,
    // On by default, matching the glTF loader. This specifically targets shader-generated metallic
    // reflection/highlight aliasing, which coverage and image-space edge filters cannot prevent.
    specularAA: true,
    // Off by default: core TAA resets to the raw frame whenever the camera moves, so while walking
    // it costs three passes and changes nothing. It is worth turning on to look at a static view,
    // where the jittered accumulation supersamples texture interiors that no spatial filter can.
    taa: false,
    // Off by default: quality is excellent but the cost is quadratic — 2x means four times the
    // pixels for every canvas-sized target, including the 4-sample MSAA pair and the fluid's
    // screen-space buffers. It is here to measure the others against.
    ssaa: 1,
    // The middle tier is visually smooth in first person while keeping startup and GPU cost modest.
    liquefactorModel: "80k",
    weaponSway: true,
    soundsEnabled: true,
    // On by default for smooth room transitions. Older devices can use one dominant probe, retaining
    // box projection while avoiding the second cubemap sample.
    localCubemapBlending: true,
    // Off, and it should stay off: measured, it double-encodes (the image-processing stage already
    // writes display-space values) and changes AA quality by less than a percentage point.
    srgb: false,
};

/** Describes a setting for a generic config UI: no UI code needs to know the field names. */
export interface GraphicsSettingDef {
    readonly key: keyof GraphicsSettings;
    /** `toggle` is boolean; `scale` and `choice` select a value from `options`. */
    readonly kind: "toggle" | "scale" | "choice";
    readonly label: string;
    readonly help: string;
    /** For `scale` / `choice`: the values a UI should offer. */
    readonly options?: readonly (number | string)[];
}

export const GRAPHICS_SETTING_DEFS: readonly GraphicsSettingDef[] = [
    { key: "msaa", kind: "toggle", label: "Anti-aliasing (4× MSAA)", help: "Smooths jagged polygon edges. Costs GPU memory and some fill rate." },
    { key: "smaa", kind: "toggle", label: "Anti-aliasing (SMAA)", help: "Smooths hard lines inside textures, which MSAA cannot. Costs a little fill rate." },
    {
        key: "specularAA",
        kind: "toggle",
        label: "Specular anti-aliasing",
        help: "Reduces metallic reflection shimmer by filtering rapid normal changes. Rebuilds PBR shaders when changed.",
    },
    {
        key: "taa",
        kind: "toggle",
        label: "Anti-aliasing (TAA)",
        help: "Accumulates jittered frames. Best quality when standing still; pauses while the camera moves. Replaces MSAA and SMAA.",
    },
    {
        key: "ssaa",
        kind: "scale",
        label: "Supersampling (SSAA)",
        help: "Renders at twice the display dimensions and downscales. Best quality of all, but costs four times the pixels.",
        options: [1, 2],
    },
    {
        key: "liquefactorModel",
        kind: "choice",
        label: "Liquefactor model",
        help: "Selects the first-person weapon geometry detail.",
        options: LIQUEFACTOR_MODELS,
    },
    { key: "weaponSway", kind: "toggle", label: "Weapon sway", help: "Adds subtle idle balancing motion to the held weapon without moving the crosshair." },
    { key: "soundsEnabled", kind: "toggle", label: "Sounds", help: "Enables pickup and Liquefactor sound effects." },
    {
        key: "localCubemapBlending",
        kind: "toggle",
        label: "Local cubemap blending",
        help: "Blends two box-projected probes across room boundaries. Disable on older devices to sample only the dominant probe.",
    },
    {
        key: "srgb",
        kind: "toggle",
        label: "sRGB swapchain (diagnostic)",
        help: "Double-encodes on this demo — the image washes out and AA is unaffected. Leave off. Needs a reload.",
    },
];

const STORAGE_KEY = "aquanova.graphics";

/** Raw query-string value for a key, or undefined when absent. */
function queryRaw(key: string): string | null | undefined {
    if (typeof window === "undefined") return undefined;
    const params = new URLSearchParams(window.location.search);
    if (!params.has(key)) return undefined;
    return params.get(key);
}

/** Parse a query-string flag: `?msaa=0` / `?msaa=false` disables, `?msaa=1` / `?msaa` enables. */
function queryOverride(key: string): boolean | undefined {
    const raw = queryRaw(key);
    if (raw === undefined) return undefined;
    // A bare `?msaa` (no value) reads as the empty string and means "on".
    return raw === null || raw === "" ? true : raw !== "0" && raw.toLowerCase() !== "false";
}

/** Parse a positive numeric query-string value, e.g. `?ssaa=1.5`. */
function queryScale(key: string): number | undefined {
    const raw = queryRaw(key);
    if (raw === undefined || raw === null || raw === "") return undefined;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : undefined;
}

/**
 * Resolve the settings for this session: built-in defaults, replaced by anything the ship authored,
 * overlaid with anything previously saved, overlaid with query-string overrides. The query string
 * wins so a device-specific link (or a QA run) can pin a setting without disturbing what the player
 * saved; the player's own saved choice in turn wins over the ship, so tuning the manifest never
 * silently takes a setting away from someone who has already picked it.
 *
 * Driven by GRAPHICS_SETTING_DEFS so a new setting needs no changes here.
 *
 * @param authored Per-key defaults from the manifest. Keys that are absent or `undefined` leave the
 *                 built-in default in place.
 */
export function loadGraphicsSettings(authored?: Partial<GraphicsSettings>): GraphicsSettings {
    const settings: GraphicsSettings = { ...DEFAULT_GRAPHICS };
    if (authored) {
        // GraphicsSettings mixes booleans, numbers and strings; the widened view keeps this
        // key-agnostic, the same way the override loop below stays data-driven.
        const base = settings as unknown as Record<string, boolean | number | string>;
        for (const [key, value] of Object.entries(authored)) {
            if (value !== undefined && key in base) base[key] = value;
        }
    }
    let saved: Partial<GraphicsSettings> | null = null;
    try {
        const raw = typeof localStorage !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
        if (raw) saved = JSON.parse(raw) as Partial<GraphicsSettings>;
    } catch {
        // Private-mode / disabled storage / corrupt JSON: fall back to defaults rather than failing
        // to boot. A graphics preference is never worth breaking the demo over.
    }
    const legacy = saved as (Partial<GraphicsSettings> & { localCubemapParallax?: unknown }) | null;
    if (typeof saved?.localCubemapBlending !== "boolean" && typeof legacy?.localCubemapParallax === "boolean") {
        settings.localCubemapBlending = legacy.localCubemapParallax;
    }
    for (const def of GRAPHICS_SETTING_DEFS) {
        // GraphicsSettings mixes booleans and numbers; go through a widened view so the loop can
        // stay data-driven rather than naming each field.
        const out = settings as unknown as Record<string, boolean | number | string>;
        if (def.kind === "scale") {
            const stored = saved?.[def.key];
            if (typeof stored === "number" && Number.isFinite(stored) && stored > 0) out[def.key] = stored;
            const override = queryScale(def.key);
            if (override !== undefined) out[def.key] = override;
        } else if (def.kind === "choice") {
            const choices = def.options?.filter((value): value is string => typeof value === "string") ?? [];
            const stored = saved?.[def.key];
            if (typeof stored === "string" && choices.includes(stored)) out[def.key] = stored;
            const override = queryRaw(def.key);
            if (override && choices.includes(override)) out[def.key] = override;
        } else {
            const stored = saved?.[def.key];
            if (typeof stored === "boolean") out[def.key] = stored;
            const override = queryOverride(def.key);
            if (override !== undefined) out[def.key] = override;
        }
    }
    if (queryRaw("localCubemapBlending") === undefined) {
        const legacyOverride = queryOverride("localCubemapParallax");
        if (legacyOverride !== undefined) settings.localCubemapBlending = legacyOverride;
    }
    settings.ssaa = settings.ssaa > 1 ? 2 : 1;
    if (!LIQUEFACTOR_MODELS.includes(settings.liquefactorModel)) settings.liquefactorModel = DEFAULT_GRAPHICS.liquefactorModel;
    return applyExclusivity(settings);
}

/**
 * TAA is an ALTERNATIVE to the spatial filters, not an addition to them: it supersamples by jittering
 * the camera, so running it on top of MSAA or SMAA means accumulating frames that have already been
 * spatially filtered — double-smoothing, and the jitter fights the filters' edge detection. Enabling
 * TAA therefore turns the other two off, and vice versa.
 */
export function applyExclusivity(settings: GraphicsSettings): GraphicsSettings {
    if (settings.taa) {
        settings.msaa = false;
        settings.smaa = false;
    }
    return settings;
}

/** Persist the player's choices. Never throws — storage can be unavailable. */
export function saveGraphicsSettings(settings: GraphicsSettings): void {
    try {
        if (typeof localStorage !== "undefined") localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
        // See loadGraphicsSettings.
    }
}
