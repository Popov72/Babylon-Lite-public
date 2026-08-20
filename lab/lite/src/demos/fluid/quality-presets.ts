// Loads the per-(demo, method, quality) fluid presets at RUNTIME and indexes them for the
// core's first-visit lookup. The files live in lab/public/fluid-presets/<demo>.<sph|mlsmpm>
// .<low|middle|high>.json (grouped "Export parameters" shape) and are served statically by
// the dev server, so editing one + reloading the page picks up the change with NO bundle
// rebuild. Each file is converted back to a Partial<PairState> (via preset-io) that the core
// merges over its per-method defaults.
//
// `loadQualityPresets()` MUST be awaited once (before the first switchPair) to populate the
// registry; `getQualityPreset()` then reads it synchronously.

import { presetFromExportJson, type FluidExportJson } from "./preset-io.js";
import type { PairState } from "./demo.js";

/** Quality tiers shown in the panel's quality dropdown (order = display order). */
export type Quality = "low" | "middle" | "high";
export const QUALITIES: Quality[] = ["low", "middle", "high"];
export const DEFAULT_QUALITY: Quality = "middle";

/** Internal method name → filename slug. */
const METHOD_TO_SLUG: Record<string, string> = { PBF: "sph", FLIP: "flip", "MLS-MPM": "mlsmpm", "PB-MPM": "pbmpm" };
const METHODS = Object.keys(METHOD_TO_SLUG);

/** PB-MPM material enum value → filename slug (index = material value: 0 liquid, 1 elastic, 2 sand, 3 visco). */
export const MATERIAL_SLUGS = ["liquid", "elastic", "sand", "visco"];

/** Static URL the dev server maps to lab/public/fluid-presets/. Fetched fresh (no-store) so a
 *  page reload always re-reads the current file contents. */
const PRESET_BASE = "/fluid-presets";

// registry[key] = Partial<PairState>, where key = `${demo}|${method}|${quality}|${materialSlug}`.
// materialSlug is "" for non-PB-MPM methods and the material slug (liquid/elastic/sand/visco) for PB-MPM,
// so PB-MPM carries one preset PER MATERIAL (their physics/render/colour differ).
let registry: Record<string, Partial<PairState>> = {};

const keyOf = (demo: string, method: string, quality: string, materialSlug: string): string => `${demo}|${method}|${quality}|${materialSlug}`;

/** Fetch + index every (demo × method × quality [× material for PB-MPM]) preset that exists on disk.
 *  Missing files (404) are skipped so a combo with no file falls back to core defaults. Safe to call
 *  again to hot-reload — it rebuilds the registry from scratch. */
export async function loadQualityPresets(demoKeys: string[], materialDemoKeys: string[] = []): Promise<void> {
    const reg: Record<string, Partial<PairState>> = {};
    const jobs: Promise<void>[] = [];
    const fetchInto = (demo: string, method: string, quality: string, materialSlug: string, file: string): Promise<void> =>
        (async (): Promise<void> => {
            try {
                const res = await fetch(`${PRESET_BASE}/${file}`, { cache: "no-store" });
                if (!res.ok) {
                    return;
                }
                const json = (await res.json()) as FluidExportJson;
                reg[keyOf(demo, method, quality, materialSlug)] = presetFromExportJson(json);
            } catch {
                // Network/parse error → treat as "no preset" (core defaults apply).
            }
        })();
    for (const demo of demoKeys) {
        for (const method of METHODS) {
            const slug = METHOD_TO_SLUG[method]!;
            for (const quality of QUALITIES) {
                if (method === "PB-MPM") {
                    // Only material-capable demos (e.g. the box container) carry per-material presets;
                    // every other demo is liquid-only.
                    const mats = materialDemoKeys.includes(demo) ? MATERIAL_SLUGS : ["liquid"];
                    for (const mat of mats) {
                        jobs.push(fetchInto(demo, method, quality, mat, `${demo}.${slug}.${mat}.${quality}.json`));
                    }
                } else {
                    jobs.push(fetchInto(demo, method, quality, "", `${demo}.${slug}.${quality}.json`));
                }
            }
        }
    }
    await Promise.all(jobs);
    registry = reg;
}

/** First-visit preset for a (demo, method, quality [, material]), or null if none was loaded (the core
 *  then falls back to its built-in defaults). For PB-MPM a `material` (0..3) selects the material preset;
 *  if that material has no file, it falls back to the LIQUID preset of the same (demo, quality) so the
 *  render params stay sensible. Call {@link loadQualityPresets} first. */
export function getQualityPreset(demo: string, method: string, quality: string, material = 0): Partial<PairState> | null {
    if (method !== "PB-MPM") {
        return registry[keyOf(demo, method, quality, "")] ?? null;
    }
    const slug = MATERIAL_SLUGS[material] ?? "liquid";
    return registry[keyOf(demo, method, quality, slug)] ?? registry[keyOf(demo, method, quality, "liquid")] ?? null;
}
