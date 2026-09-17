/**
 * Build Bundle Demos — builds each lab "demo" plus required demo support
 * bundles as standalone, tree-shaken, minified production bundles into
 * lab/public/bundle/demos/, writes the demo HTML needed to serve those bundles,
 * then measures each configured demo's runtime JS size with a headless browser.
 *
 * Demos are showcase-only pages (pure Lite, no BJS comparison, no parity/golden
 * obligations) that exist to advertise how thin a Lite-powered page can be.
 * They are intentionally kept OUT of scene-config.json so they don't inherit
 * parity / bundle-ceiling test requirements.
 *
 * Sizes are written to lab/public/bundle/demos-manifest.json which the lab
 * "Demos" tab reads to render a size badge per demo.
 *
 * NOTE: The Vite build config below mirrors the lite branch of `buildScene`
 * in bundle-scenes-core.ts so demo sizes are measured the exact same way as
 * scenes. Keep the two in sync.
 *
 * Usage: npx tsx scripts/build-bundle-demos.ts
 */
import { build, type Plugin } from "vite";
import { createHash } from "crypto";
import { spawnSync } from "child_process";
import { resolve, dirname, basename } from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";
import { cpSync, readdirSync, readFileSync, writeFileSync, mkdirSync, existsSync, rmSync, statSync } from "fs";
import {
    labDir,
    srcDir,
    outDir,
    terserPropertyManglePlugin,
    isLiteBundleExternal,
    writeBundleInfo,
    startStaticServer,
    measurementBrowserArgs,
    measurePage,
    LITE_BUNDLE_TARGET,
    NAME_POLYFILL,
    atomicWriteJson,
} from "./bundle-scenes-core";
import { wgslMinifyPlugin } from "./wgsl-minify-plugin";
import { fetchDemoAssets } from "./demo-fetchers";
import { demoOwnsBundleFile } from "./demo-bundle-name";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PAGES_SRC = resolve(ROOT, "pages");
const THUMBS_SRC = resolve(labDir, "public/thumbnails");
const DOOM_SRC = resolve(labDir, "public/doom");
const LIBREQUAKE_SRC = resolve(labDir, "public/librequake");
const MINECRAFT_SRC = resolve(labDir, "public/minecraft");
const FREECIV_SRC = resolve(labDir, "public/freeciv");
const LITTLEST_TOKYO_SRC = resolve(labDir, "public/littlest-tokyo");
const TETRIS_SRC = resolve(labDir, "public/tetris");
const PLATFORMER_SRC = resolve(labDir, "public/platformer");
const SANDBLOX_SRC = resolve(labDir, "public/sandblox");
const WATERFALL_SRC = resolve(labDir, "public/waterfall");
const JUMPING_WHALE_SRC = resolve(labDir, "public/jumping-whale");
const RACER_SRC = resolve(labDir, "public/racer");
const ANTIGRAVITY_RACER_SRC = resolve(labDir, "public/antigravity-racer");
const SCREEN_SPACE_EFFECTS_SRC = resolve(labDir, "public/screen-space-effects");
const DRACO_FILES = ["draco_decoder.js", "draco_decoder.wasm"];

const _demoRequire = createRequire(import.meta.url);

function ensureJumpingWhaleSdfCurrent(): void {
    const glb = resolve(JUMPING_WHALE_SRC, "whale.glb");
    const sdf = resolve(JUMPING_WHALE_SRC, "whale-sdf.bin");
    const metadataFile = resolve(JUMPING_WHALE_SRC, "whale-sdf.meta.json");
    const sourceSha256 = createHash("sha256").update(readFileSync(glb)).digest("hex");
    let metadata: { formatVersion?: number; generator?: string; sourceSha256?: string } | null = null;
    try {
        metadata = JSON.parse(readFileSync(metadataFile, "utf8")) as { formatVersion?: number; generator?: string; sourceSha256?: string };
    } catch {
        // Missing or malformed metadata is handled as a stale bake below.
    }
    if (existsSync(sdf) && metadata?.formatVersion === 2 && metadata.generator === "native-gltf-skinning" && metadata.sourceSha256 === sourceSha256) {
        return;
    }
    console.log("Jumping whale GLB changed; regenerating its deforming SDF...");
    const pnpm = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
    const result = spawnSync(pnpm, ["exec", "tsx", resolve(ROOT, "scripts/bake-jumping-whale-sdf.ts")], {
        cwd: ROOT,
        stdio: "inherit",
    });
    if (result.status !== 0) {
        throw new Error("Jumping whale SDF regeneration failed. Run `pnpm exec tsx scripts/bake-jumping-whale-sdf.ts` for the detailed error.");
    }
}

/** Absolute path to the ESM build of `@babylonjs/havok`, or null if unavailable.
 *  Scenes externalize Havok to `/vendor/havok.js` via an import map, but standalone
 *  demo bundles have no import map — so we alias Havok to this ESM file and bundle
 *  it inline (its WASM is still fetched at runtime via the caller's `locateFile`). */
function havokEsmEntry(): string | null {
    try {
        const havokMain = _demoRequire.resolve("@babylonjs/havok");
        const esm = resolve(dirname(dirname(havokMain)), "esm/HavokPhysics_es.js");
        return existsSync(esm) ? esm : null;
    } catch {
        return null;
    }
}

interface DemoConfigEntry {
    slug: string;
    name: string;
    description: string;
    tags?: string[];
    /** When false, the demo is hidden on mobile-oriented demo listings. */
    mobile?: boolean;
    /** Optional id of the asset fetcher for this demo (see scripts/demo-fetchers.ts). */
    fetch?: string;
}

interface DemoManifestEntry {
    rawKB: number;
    gzipKB: number;
}

const demosDir = resolve(outDir, "demos");
const DEMOS_MANIFEST_FILE = resolve(outDir, "demos-manifest.json");
const DEMO_SUPPORT_BUNDLES = ["landing-bg"] as const;
const DEMO_SOURCE_BASE_URL = "https://github.com/BabylonJS/Babylon-Lite/blob/master/lab/lite/src/demos/";

/** Stub Vite's preload helper so it doesn't add bytes to measured bundles. */
function minimalVitePreloadPlugin(): Plugin {
    const id = "\0minimal-vite-preload";
    return {
        name: "minimal-vite-preload",
        enforce: "pre",
        resolveId(source) {
            return source === "vite/preload-helper.js" ? id : null;
        },
        load(source) {
            return source === id ? "export const __vitePreload = (baseModule) => baseModule();" : null;
        },
        transform(_code, source) {
            return source.endsWith("vite/preload-helper.js") ? "export const __vitePreload = (baseModule) => baseModule();" : null;
        },
    };
}

function loadDemosConfig(): DemoConfigEntry[] {
    return JSON.parse(readFileSync(resolve(ROOT, "demos-config.json"), "utf-8")) as DemoConfigEntry[];
}

function escapeHtml(value: string): string {
    return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function rewriteDemoHtmlForBundle(html: string): string {
    return html.replace(/(["'])\/(?:lite\/)?bundle\/demos\//g, "$1./");
}

function readDemoManifest(): Record<string, DemoManifestEntry> {
    return existsSync(DEMOS_MANIFEST_FILE) ? (JSON.parse(readFileSync(DEMOS_MANIFEST_FILE, "utf-8")) as Record<string, DemoManifestEntry>) : {};
}

function writeDemoManifest(manifest: Record<string, DemoManifestEntry>): void {
    atomicWriteJson(DEMOS_MANIFEST_FILE, `${JSON.stringify(manifest, null, 2)}\n`);
}

export async function measureAndPersistDemoManifestEntry(
    manifest: Record<string, DemoManifestEntry>,
    slug: string,
    measure: () => Promise<DemoManifestEntry>,
    persist: (manifest: Record<string, DemoManifestEntry>) => void = writeDemoManifest
): Promise<DemoManifestEntry> {
    const previous = manifest[slug];
    try {
        const next = await measure();
        manifest[slug] = next;
        persist(manifest);
        return next;
    } catch (error) {
        if (previous === undefined) {
            delete manifest[slug];
        } else {
            manifest[slug] = previous;
        }
        throw error;
    }
}

/**
 * Inject the measured engine/code size (the same KB shown on the gallery card)
 * into a built demo page so its loading overlay can reiterate it next to the
 * asset estimate. `installFetchProgress` reads `window.__DEMO_ENGINE_KB`; the
 * pre-hydration `.loading-size` text is rewritten to match.
 */
function injectDemoEngineSize(html: string, rawKB: number): string {
    const tag = `<script>window.__DEMO_ENGINE_KB=${rawKB};</script>`;
    const out = html.includes("</head>") ? html.replace("</head>", `  ${tag}\n</head>`) : `${tag}\n${html}`;
    return out.replace(/Estimated demo assets:\s*/g, `Engine ${rawKB} KB · Assets `);
}

function renderCard(demo: DemoConfigEntry, size: DemoManifestEntry | undefined): string {
    const tagList = demo.tags ?? [];
    const tags = tagList.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("");
    const sizeRow = size
        ? `<div class="size" title="Engine + demo code only — excludes external assets (textures, game data, etc.)"><strong>${size.rawKB} KB</strong> · ${size.gzipKB} KB gzip</div>`
        : "";
    const sourceHref = `${DEMO_SOURCE_BASE_URL}${encodeURIComponent(demo.slug)}.ts`;
    return [
        `<article class="card" data-tags="${escapeHtml(tagList.join(" "))}" data-mobile="${demo.mobile === false ? "false" : "true"}">`,
        `<a class="card-main" href="./demo-${demo.slug}.html" aria-label="Open ${escapeHtml(demo.name)} demo">`,
        `<div class="card-image">`,
        `<img src="thumbnails/demo-${demo.slug}.jpg" alt="${escapeHtml(demo.name)} thumbnail" loading="lazy" decoding="async" onerror="this.remove()" />`,
        `</div>`,
        `<div class="card-body">`,
        `<h2>${escapeHtml(demo.name)}</h2>`,
        `<p>${escapeHtml(demo.description)}</p>`,
        tags ? `<div class="tags">${tags}</div>` : "",
        sizeRow,
        `<span class="card-disabled-badge">Requires WebGPU</span>`,
        `</div></a>`,
        `<div class="card-links"><a class="source-link" href="${escapeHtml(sourceHref)}" target="_blank" rel="noopener noreferrer">Source code</a></div>`,
        `</article>`,
    ].join("");
}

function renderFilters(demos: DemoConfigEntry[]): string {
    const tags = Array.from(new Set(demos.flatMap((demo) => demo.tags ?? []))).sort();
    if (tags.length === 0) {
        return "";
    }
    const pills = [
        `<button type="button" class="filter-pill is-active" data-filter="all" aria-pressed="true">All</button>`,
        ...tags.map((tag) => `<button type="button" class="filter-pill" data-filter="${escapeHtml(tag)}" aria-pressed="false">${escapeHtml(tag)}</button>`),
    ].join("");
    return `<nav class="filters" aria-label="Filter demos by tag">${pills}</nav>`;
}

function renderDemoIndex(demos: DemoConfigEntry[], manifest: Record<string, DemoManifestEntry>): string {
    const template = readFileSync(resolve(PAGES_SRC, "index.template.html"), "utf-8");
    const cards = demos.map((demo) => renderCard(demo, manifest[demo.slug])).join("\n                ");
    return template
        .replace("<!--FILTERS-->", renderFilters(demos))
        .replace("<!--CARDS-->", cards)
        .replace(/(["'])bundle\/demos\/landing-bg\.js\1/g, "$1./landing-bg.js$1");
}

function copyDemoIndexAssets(demos: DemoConfigEntry[]): void {
    cpSync(resolve(PAGES_SRC, "babylon-logo.svg"), resolve(demosDir, "babylon-logo.svg"));

    const thumbsOut = resolve(demosDir, "thumbnails");
    rmSync(thumbsOut, { recursive: true, force: true });
    mkdirSync(thumbsOut, { recursive: true });
    for (const demo of demos) {
        const thumb = resolve(THUMBS_SRC, `demo-${demo.slug}.jpg`);
        if (existsSync(thumb)) {
            cpSync(thumb, resolve(thumbsOut, `demo-${demo.slug}.jpg`));
        }
    }
}

function copyRequiredDir(source: string, target: string, label: string, include?: (file: string) => boolean): void {
    if (!existsSync(source)) {
        throw new Error(`Missing ${label} assets at ${source}`);
    }
    rmSync(target, { recursive: true, force: true });
    cpSync(source, target, { recursive: true, filter: include ? (src) => include(basename(src)) : undefined });
}

function copyDemoRuntimeAssets(demos: DemoConfigEntry[]): void {
    if (demos.some((demo) => demo.slug === "doom")) {
        if (!existsSync(DOOM_SRC)) {
            throw new Error(`Missing DOOM assets at ${DOOM_SRC}`);
        }
        const doomOut = resolve(demosDir, "doom");
        rmSync(doomOut, { recursive: true, force: true });
        mkdirSync(doomOut, { recursive: true });
        for (const file of readdirSync(DOOM_SRC)) {
            if (file === "freedoom2.wad") continue;
            cpSync(resolve(DOOM_SRC, file), resolve(doomOut, file));
        }
    }

    if (demos.some((demo) => demo.slug === "quake")) {
        copyRequiredDir(LIBREQUAKE_SRC, resolve(demosDir, "librequake"), "LibreQuake");
    }

    if (demos.some((demo) => demo.slug === "minecraft")) {
        copyRequiredDir(MINECRAFT_SRC, resolve(demosDir, "minecraft"), "Minecraft voxel pack");
    }

    if (demos.some((demo) => demo.slug === "freeciv")) {
        copyRequiredDir(FREECIV_SRC, resolve(demosDir, "freeciv"), "Freeciv");
    }

    if (demos.some((demo) => demo.slug === "platformer")) {
        // Committed CC0 Kenney sprite sheets + backgrounds, copied under the demo's own
        // subpath (the demo resolves them via `demoAssetUrl("./platformer/...")`), so the
        // deployed demos site (which serves ONLY lab/public/bundle/demos/) finds them.
        copyRequiredDir(PLATFORMER_SRC, resolve(demosDir, "platformer"), "Platformer");
    }

    if (demos.some((demo) => demo.slug === "littlest-tokyo")) {
        copyRequiredDir(LITTLEST_TOKYO_SRC, resolve(demosDir, "littlest-tokyo"), "Littlest Tokyo");
    }

    if (demos.some((demo) => demo.slug === "sandblox")) {
        // Default world map JSON, fetched at runtime via demoAssetUrl.
        copyRequiredDir(SANDBLOX_SRC, resolve(demosDir, "sandblox"), "Sandblox");
    }

    if (demos.some((demo) => demo.slug === "fluid")) {
        // Waterfall scene: the rock-formation glTF plus the height map baked from it
        // (lab/public/waterfall/scripts/bake-rock-heightmap.ts), both fetched at runtime via demoAssetUrl.
        // `rock-high.glb` / `rock-heightmap-high.bin` — the raw 72 MB photogrammetry scan and its own
        // bake — are the SOURCE the shipped rock.glb is derived from (lab/public/waterfall/scripts/optimize-rock.ts),
        // kept next to it for future re-exports but never served, so they are skipped here rather
        // than copied into every demo build. `scripts/` is that tooling itself — build-time
        // Node code that has no business being downloaded by a browser.
        //
        // Two more intermediates are skipped for the same reason. `oasis.glb` is the un-tiled
        // island the shipped `oasis-*.glb` tiers are derived from, and `oasis-heightmap.bin` was
        // baked when the oasis was briefly the collision model — it is set dressing now, so
        // nothing ever fetches it. Together they were 6.7 MB of dead weight per demo build.
        //
        // All three `oasis-<tier>.glb` ARE shipped: the scene's detail selector loads them lazily,
        // and Draco keeps the set to 8.4 MB. Note that `oasis-high.glb` must NOT be caught by the
        // rock's `-high.` source-file rule, hence the explicit keep below.
        const WATERFALL_SKIP = new Set(["scripts", "oasis.glb", "oasis-heightmap.bin"]);
        const isUnservedSource = (file: string): boolean => file.includes("-high.") && !file.startsWith("oasis-");
        copyRequiredDir(WATERFALL_SRC, resolve(demosDir, "waterfall"), "Waterfall rock", (file) => !WATERFALL_SKIP.has(file) && !isUnservedSource(file));
        ensureJumpingWhaleSdfCurrent();
        copyRequiredDir(JUMPING_WHALE_SRC, resolve(demosDir, "jumping-whale"), "Jumping whale", (file) => file !== "whale-sdf.meta.json");
    }

    if (demos.some((demo) => demo.slug === "racer")) {
        // CC0 Kenney car / track / prop GLBs + smoke sprite + audio, fetched by
        // fetch-racer.ts and resolved at runtime via demoAssetUrl("./racer/...").
        copyRequiredDir(RACER_SRC, resolve(demosDir, "racer"), "Racer");
    }

    if (demos.some((demo) => demo.slug === "screen-space-effects")) {
        copyRequiredDir(SCREEN_SPACE_EFFECTS_SRC, resolve(demosDir, "screen-space-effects"), "Screen-space effects");
    }

    if (demos.some((demo) => demo.slug === "antigravity-racer")) {
        const required = ["track/road-straight.png", "track/road-curve.png", "track/road-emissive.png", "track/boost-arrow.png", "rhs-x.glb", "rock.glb"];
        for (const file of required) {
            if (!existsSync(resolve(ANTIGRAVITY_RACER_SRC, file))) {
                throw new Error(`Missing Antigravity Racer asset ${file} at ${ANTIGRAVITY_RACER_SRC}`);
            }
        }
        copyRequiredDir(ANTIGRAVITY_RACER_SRC, resolve(demosDir, "antigravity-racer"), "Antigravity Racer");
    }

    if (demos.some((demo) => demo.slug === "bath-day")) {
        const glb = resolve(labDir, "public", "bath_day.glb");
        if (existsSync(glb)) {
            cpSync(glb, resolve(demosDir, "bath_day.glb"));
        }
    }

    if (demos.some((demo) => demo.slug === "calculator")) {
        const glb = resolve(labDir, "public", "models", "Calculator.glb");
        if (existsSync(glb)) {
            cpSync(glb, resolve(demosDir, "Calculator.glb"));
        }
    }

    if (demos.some((demo) => demo.slug === "tetris")) {
        // Tetris geometry/texture assets (consolidated under lab/public/tetris/)
        // plus its local studio HDR environment, copied flat so the demo resolves
        // them relative to its own module (subpath-safe).
        copyRequiredDir(TETRIS_SRC, resolve(demosDir, "tetris"), "Tetris");
        const env = resolve(labDir, "public", "textures", "environment.env");
        if (existsSync(env)) {
            cpSync(env, resolve(demosDir, "environment.env"));
        }
    }

    for (const file of [...DRACO_FILES, "meshopt_decoder.js", "brdf-lut.png", "HavokPhysics.wasm"]) {
        const src = resolve(labDir, "public", file);
        if (existsSync(src)) {
            cpSync(src, resolve(demosDir, file));
        }
    }
}

function writeDemoHtml(demos: DemoConfigEntry[], manifest: Record<string, DemoManifestEntry>): void {
    for (const demo of demos) {
        const source = resolve(labDir, "lite", `demo-${demo.slug}.html`);
        if (!existsSync(source)) {
            throw new Error(`Missing demo HTML: ${source}`);
        }
        const html = rewriteDemoHtmlForBundle(readFileSync(source, "utf-8"));
        const rawKB = manifest[demo.slug]?.rawKB;
        writeFileSync(resolve(demosDir, `demo-${demo.slug}.html`), rawKB != null ? injectDemoEngineSize(html, rawKB) : html);
    }
    copyDemoIndexAssets(demos);
    writeFileSync(resolve(demosDir, "index.html"), renderDemoIndex(demos, manifest));
}

/**
 * Lite demo bundle sizes are only valid once the demo reports ready. A failed
 * boot can otherwise look like a tiny/0 KB "success" because the lazy runtime
 * chunks never load.
 */
export function demoRequiresReady(_slug: string): boolean {
    return true;
}

type MeasureDemoPage = (
    browser: any,
    port: number,
    scene: string,
    htmlFile: string,
    bundlePath: string,
    requireReady?: boolean,
    readyTimeoutMs?: number
) => Promise<{ rawKB: number; gzipKB: number }>;

export async function measureDemoManifestEntry(browser: any, port: number, slug: string, measure: MeasureDemoPage = measurePage): Promise<DemoManifestEntry> {
    const { rawKB, gzipKB } = await measure(browser, port, `demo-${slug}`, `lite/demo-${slug}.html`, "/bundle/demos/", demoRequiresReady(slug));
    return { rawKB, gzipKB };
}

/**
 * Whether demo bundles keep their debug tooling. OFF unless asked for, so the default artefact is
 * the one a visitor downloads. Enabled by `--debug` on the build command or `LAB_DEMO_DEBUG=1`.
 */
function demoDebugEnabled(): boolean {
    return process.argv.includes("--debug") || process.env.LAB_DEMO_DEBUG === "1";
}

export async function buildDemo(slug: string): Promise<void> {
    const demoOutDir = resolve(demosDir, slug);
    rmSync(demoOutDir, { recursive: true, force: true });

    // Standalone demos have no import map, so Havok can't be externalized to
    // /vendor/havok.js like scenes do — bundle its ESM build inline instead.
    const havokEsm = havokEsmEntry();

    const buildResult = await build({
        root: labDir,
        configFile: false,
        base: "./",
        publicDir: false,
        logLevel: "warn",
        // Compile-time switch for a demo's debug tooling (overlays, QA hooks). OFF by default so a
        // shipped demo carries none of it — the bundler folds `if (LAB_DEBUG)` to `false` and drops
        // the `debug/` modules. Turn it on with `--debug` or LAB_DEMO_DEBUG=1 to ship a build you can
        // actually inspect. The dev server injects nothing and defaults to ON (see debug-flag.ts).
        define: { __LAB_DEBUG__: JSON.stringify(demoDebugEnabled()) },
        plugins: [wgslMinifyPlugin(), terserPropertyManglePlugin(), minimalVitePreloadPlugin()],
        resolve: {
            // Demos resolve `babylon-lite` to the TS SOURCE (not `build/lib`) on purpose:
            // demos have no bundle-size ceilings, and using source keeps the dev iteration
            // loop fast (no package rebuild required to see demo changes). Demo sizes could
            // therefore differ slightly from a real consumer's, but the scene bundle-size
            // tests (which DO build against `build/lib`) are what guard against size drift.
            alias: { "babylon-lite": srcDir, ...(havokEsm ? { "@babylonjs/havok": havokEsm } : {}) },
            dedupe: ["@babylonjs/core"],
        },
        build: {
            outDir: demoOutDir,
            emptyOutDir: true,
            target: LITE_BUNDLE_TARGET,
            minify: "esbuild",
            sourcemap: "hidden",
            modulePreload: { polyfill: false, resolveDependencies: () => [] },
            rollupOptions: {
                input: { [slug]: resolve(labDir, `lite/src/demos/${slug}.ts`) },
                // Bundle Havok inline (aliased above); keep the other vendor runtimes external.
                external: (id: string) => id !== "@babylonjs/havok" && isLiteBundleExternal(id),
                output: {
                    format: "es",
                    entryFileNames: "[name].js",
                    chunkFileNames: `${slug}-[name]-[hash].js`,
                    banner: NAME_POLYFILL,
                },
            },
        },
        // Demos may spawn a module Web Worker via `new Worker(new URL("./x.ts", import.meta.url), { type: "module" })`
        // (see the offscreen demo). Build the worker with the same tagged-WGSL/property-mangle
        // pipeline and emit its chunks prefixed with the slug so the copy + stale-cleanup
        // logic below picks them up alongside the main entry.
        worker: {
            format: "es",
            plugins: () => [wgslMinifyPlugin(), terserPropertyManglePlugin()],
            rollupOptions: {
                output: {
                    entryFileNames: `${slug}-worker-[hash].js`,
                    chunkFileNames: `${slug}-worker-[name]-[hash].js`,
                    banner: NAME_POLYFILL,
                },
            },
        },
    });

    // Bundle-info keyed as `demo-<slug>` so size accounting can read it during measurement.
    writeBundleInfo(`demo-${slug}`, buildResult);

    // Atomically replace this demo's files in outDir/demos:
    // 1. Write all new files. 2. Remove stale chunks from a previous build.
    mkdirSync(demosDir, { recursive: true });
    const newNames = new Set<string>();
    for (const f of readdirSync(demoOutDir)) {
        if (f.endsWith(".map")) continue;
        if (!statSync(resolve(demoOutDir, f)).isFile()) continue;
        newNames.add(f);
        writeFileSync(resolve(demosDir, f), readFileSync(resolve(demoOutDir, f)));
    }
    const demoSlugs = [...loadDemosConfig().map((demo) => demo.slug), ...DEMO_SUPPORT_BUNDLES];
    for (const existing of readdirSync(demosDir)) {
        if (demoOwnsBundleFile(existing, slug, demoSlugs) && !newNames.has(existing)) {
            rmSync(resolve(demosDir, existing));
        }
    }
    rmSync(demoOutDir, { recursive: true, force: true });
}

export async function buildDemoSupportBundles(): Promise<void> {
    for (const slug of DEMO_SUPPORT_BUNDLES) {
        console.log(`Building demo support bundle ${slug}...`);
        await buildDemo(slug);
    }
}

/**
 * Build a SINGLE demo by slug — its bundle, runtime assets, and standalone
 * HTML — without rebuilding the other demos. This is the fast iteration path
 * while working on one demo (`pnpm build:bundle-demo <slug>`); the full
 * `buildDemoBundles()` rebuilds and re-measures every configured demo.
 *
 * The headless size measurement (the slowest step, it needs a browser) is
 * skipped by default. Pass `{ measure: true }` to measure this demo and refresh
 * only its entry in demos-manifest.json.
 */
export async function buildSingleDemo(slug: string, options: { measure?: boolean } = {}): Promise<void> {
    const demos = loadDemosConfig();
    const demo = demos.find((d) => d.slug === slug);
    if (!demo) {
        throw new Error(`Unknown demo slug "${slug}". Known demos: ${demos.map((d) => d.slug).join(", ")}`);
    }

    // Ensure this demo's runtime assets are present (idempotent; a no-op for
    // demos like the platformer that ship a committed asset subset).
    await fetchDemoAssets([demo]);
    mkdirSync(demosDir, { recursive: true });
    console.log(`Building demo ${slug}...`);
    await buildDemo(slug);
    copyDemoRuntimeAssets([demo]);
    const manifest = readDemoManifest();

    if (options.measure) {
        const { chromium } = await import("@playwright/test");
        const { server, port } = await startStaticServer(labDir);
        try {
            const browser = await chromium.launch({ channel: "chrome", headless: true, args: measurementBrowserArgs() });
            try {
                const { rawKB, gzipKB } = await measureAndPersistDemoManifestEntry(manifest, slug, () => measureDemoManifestEntry(browser, port, slug));
                console.log(`  measured ${slug}: ${rawKB} KB raw, ${gzipKB} KB gzip`);
            } finally {
                await browser.close();
            }
        } finally {
            server.close();
        }
    }

    const source = resolve(labDir, "lite", `demo-${slug}.html`);
    if (existsSync(source)) {
        const html = rewriteDemoHtmlForBundle(readFileSync(source, "utf-8"));
        const rawKB = manifest[slug]?.rawKB;
        writeFileSync(resolve(demosDir, `demo-${slug}.html`), rawKB != null ? injectDemoEngineSize(html, rawKB) : html);
    }

    console.log(`Demo "${slug}" ready → lab/public/bundle/demos/${slug}.js`);
}

export async function buildDemoBundles(): Promise<void> {
    const demos = loadDemosConfig();
    if (demos.length === 0) {
        console.log("No demos configured; skipping demo bundle build.");
        return;
    }

    // Make sure every demo's runtime assets (IWAD, textures, tilesets, …) are
    // present locally before bundling. Each fetcher is idempotent.
    await fetchDemoAssets(demos);

    mkdirSync(demosDir, { recursive: true });

    for (const demo of demos) {
        console.log(`Building demo ${demo.slug}...`);
        await buildDemo(demo.slug);
    }

    await buildDemoSupportBundles();
    // Measurement loads demos through their source HTML, so runtime assets must
    // already exist in the bundled output. The ready-signal requirement below
    // turns a missing asset/WASM/model into a loud build failure instead of a
    // bogus tiny bundle measurement.
    copyDemoRuntimeAssets(demos);

    // Measure runtime-fetched JS size for each demo.
    const { chromium } = await import("@playwright/test");
    const { server, port } = await startStaticServer(labDir);
    const manifest = readDemoManifest();
    try {
        const browser = await chromium.launch({ channel: "chrome", headless: true, args: measurementBrowserArgs() });
        try {
            for (const demo of demos) {
                const { rawKB, gzipKB } = await measureAndPersistDemoManifestEntry(manifest, demo.slug, () => measureDemoManifestEntry(browser, port, demo.slug));
                console.log(`  measured ${demo.slug}: ${rawKB} KB raw, ${gzipKB} KB gzip`);
            }
        } finally {
            await browser.close();
        }
    } finally {
        server.close();
    }

    // Drop manifest entries for demos that no longer exist.
    const slugs = new Set(demos.map((d) => d.slug));
    let changed = false;
    for (const key of Object.keys(manifest)) {
        if (!slugs.has(key)) {
            delete manifest[key];
            changed = true;
        }
    }
    if (changed) {
        writeDemoManifest(manifest);
    }

    writeDemoHtml(demos, manifest);

    console.log(`✓ Demo bundles, manifest, and HTML built to ${demosDir}`);
}

/**
 * Build all demo bundles and write the flat, self-contained demo site (demo
 * HTML, runtime assets, landing index) into lab/public/bundle/demos/ — the same
 * artifact `build:bundle-demos` deploys — but WITHOUT the Playwright size
 * measurement. Size badges come from the committed demos-manifest.json if
 * present. Returns the output directory.
 *
 * Used by build:pages-site so that build stays browser-free; build:bundle-demos
 * uses buildDemoBundles() instead (which also measures sizes).
 */
export async function buildFlatDemoSite(): Promise<string> {
    const demos = loadDemosConfig();
    if (demos.length === 0) {
        throw new Error("No demos configured in demos-config.json");
    }
    await fetchDemoAssets(demos);

    // Clean rebuild so removed demos / stale chunks never linger in the output.
    rmSync(demosDir, { recursive: true, force: true });
    mkdirSync(demosDir, { recursive: true });

    for (const demo of demos) {
        console.log(`Building demo ${demo.slug}...`);
        await buildDemo(demo.slug);
    }
    await buildDemoSupportBundles();
    copyDemoRuntimeAssets(demos);

    const manifest = readDemoManifest();
    writeDemoHtml(demos, manifest);

    console.log(`✓ Flat demo site built to ${demosDir}`);
    return demosDir;
}
