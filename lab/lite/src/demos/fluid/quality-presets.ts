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
const METHOD_TO_SLUG: Record<string, string> = { PBF: "sph", "MLS-MPM": "mlsmpm" };
const METHODS = Object.keys(METHOD_TO_SLUG);

/** Static URL the dev server maps to lab/public/fluid-presets/. Fetched fresh (no-store) so a
 *  page reload always re-reads the current file contents. */
const PRESET_BASE = "/fluid-presets";

// registry[demoKey][method][quality] = Partial<PairState>. Populated by loadQualityPresets().
let registry: Record<string, Record<string, Record<string, Partial<PairState>>>> = {};

/** Fetch + index every (demo × method × quality) preset that exists on disk. Missing files
 *  (404) are skipped so a (demo, method, quality) with no file falls back to core defaults.
 *  Safe to call again (e.g. to hot-reload) — it rebuilds the registry from scratch. */
export async function loadQualityPresets(demoKeys: string[]): Promise<void> {
    const reg: Record<string, Record<string, Record<string, Partial<PairState>>>> = {};
    const jobs: Promise<void>[] = [];
    for (const demo of demoKeys) {
        for (const method of METHODS) {
            const slug = METHOD_TO_SLUG[method]!;
            for (const quality of QUALITIES) {
                jobs.push(
                    (async (): Promise<void> => {
                        try {
                            const res = await fetch(`${PRESET_BASE}/${demo}.${slug}.${quality}.json`, { cache: "no-store" });
                            if (!res.ok) {
                                return;
                            }
                            const json = (await res.json()) as FluidExportJson;
                            ((reg[demo] ??= {})[method] ??= {})[quality] = presetFromExportJson(json);
                        } catch {
                            // Network/parse error → treat as "no preset" (core defaults apply).
                        }
                    })()
                );
            }
        }
    }
    await Promise.all(jobs);
    registry = reg;
}

/** First-visit preset for a (demo, method, quality), or null if none was loaded (the core
 *  then falls back to its built-in defaults). Call {@link loadQualityPresets} first. */
export function getQualityPreset(demo: string, method: string, quality: string): Partial<PairState> | null {
    return registry[demo]?.[method]?.[quality] ?? null;
}
