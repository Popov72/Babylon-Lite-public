import { mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { resolve, dirname, relative, isAbsolute, extname } from "node:path";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { build } from "vite";
import { chromium } from "@playwright/test";
import { measurementBrowserArgs, NAME_POLYFILL } from "./bundle-scenes-core";
import { prepareAnalyticCase, prepareBlenderCase, writeParticleState } from "./fluid-reference/prepare";
import type { Browser } from "@playwright/test";
import type { ComparisonFrame } from "./fluid-reference/browser";
import type { ReferenceCase } from "./fluid-reference/types";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2).filter((a) => a !== "--");
function value(name: string): string | undefined {
    const i = args.indexOf(name);
    if (i < 0) {
        return undefined;
    }
    const result = args[i + 1];
    if (!result || result.startsWith("--")) {
        throw new Error(`Missing value for ${name}.`);
    }
    return result;
}
function integer(name: string, fallback: number): number {
    const number = Number(value(name) ?? fallback);
    if (!Number.isSafeInteger(number) || number <= 0) {
        throw new Error(`${name} must be a positive integer.`);
    }
    return number;
}

async function serveDirectory(directory: string) {
    const mime: Record<string, string> = {
        ".html": "text/html",
        ".js": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    };
    const server = createServer((request, response) => {
        void (async () => {
            const name = decodeURIComponent(new URL(request.url ?? "/", "http://localhost").pathname);
            const file = resolve(directory, `.${name === "/" ? "/index.html" : name}`);
            const child = relative(directory, file);
            if (request.method !== "GET" || child.startsWith("..") || isAbsolute(child)) {
                response.writeHead(403).end();
                return;
            }
            const data = await readFile(file);
            response.writeHead(200, { "Content-Type": mime[extname(file)] ?? "application/octet-stream" }).end(data);
        })().catch((error: unknown) => {
            const missing = error instanceof Error && "code" in error && error.code === "ENOENT";
            if (!missing) {
                console.error("Comparison server request failed:", error);
            }
            response.writeHead(missing ? 404 : 500).end();
        });
    });
    await new Promise<void>((done, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", done);
    });
    const address = server.address();
    if (!address || typeof address === "string") {
        server.close();
        throw new Error("Comparison server did not acquire a local TCP port.");
    }
    return { server, port: address.port };
}

async function encodeComparisonVideo(output: string, input: ReferenceCase, backend: string, frames: number): Promise<void> {
    const standard = resolve(process.env.ProgramFiles ?? "C:\\Program Files", "ffmpeg", "bin", "ffmpeg.exe");
    const executable = value("--ffmpeg") ?? process.env.FFMPEG_PATH ?? (process.platform === "win32" && existsSync(standard) ? standard : "ffmpeg");
    const run = (arguments_: string[]): Promise<void> =>
        new Promise((done, reject) => {
            const process = spawn(executable, ["-hide_banner", "-loglevel", "error", ...arguments_], { stdio: ["ignore", "ignore", "pipe"] });
            let error = "";
            process.stderr.on("data", (chunk: Buffer) => {
                error += chunk.toString();
            });
            process.on("error", reject);
            process.on("close", (code) => (code === 0 ? done() : reject(new Error(`ffmpeg failed (${code}): ${error}`))));
        });
    const names = input.referencePattern ? [backend, "blender"] : [backend];
    for (const name of names) {
        await run([
            "-framerate",
            String(input.timelineFps),
            "-start_number",
            String(input.startFrame),
            "-i",
            resolve(output, `frame-%06d-${name}.jpg`),
            "-frames:v",
            String(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "19",
            "-movflags",
            "+faststart",
            resolve(output, `${name}.mp4`),
        ]);
    }
    if (input.referencePattern) {
        await run([
            "-i",
            resolve(output, `${backend}.mp4`),
            "-i",
            resolve(output, "blender.mp4"),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "19",
            "-movflags",
            "+faststart",
            resolve(output, "side-by-side.mp4"),
        ]);
    }
}

async function main(): Promise<void> {
    const valueFlags = ["--input", "--cache", "--output", "--frames", "--backend", "--sample-every", "--substeps", "--iterations", "--tolerance", "--ffmpeg", "--prepared"];
    const switches = ["--help", "-h", "--analytic", "--no-images", "--video", "--headed", "--no-extreme-removal"];
    for (let i = 0; i < args.length; i++) {
        const flag = args[i]!;
        if (valueFlags.includes(flag)) {
            value(flag);
            i++;
        } else if (!switches.includes(flag)) {
            throw new Error(`Unknown comparison argument: ${flag}.`);
        }
    }
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage:
  pnpm exec tsx scripts\\compare-fluid-flip.ts --input <exact.json> --cache <FLIP-cache> --output <new-directory>
  pnpm exec tsx scripts\\compare-fluid-flip.ts --analytic --output <new-directory>

Options:
  --frames <count>       Number of cache frames, including the initial state (default: 51)
  --backend <name>       reference or production (default: reference)
  --sample-every <n>     Numerical/image output cadence in frames (default: 10)
  --substeps <n>         Explicit physical substeps per cache frame (default: authored minimum)
  --iterations <n>       Reference pressure iteration cap (default: 400)
  --tolerance <n>        Reference relative pressure target (default: 1e-5 for float32)
  --no-extreme-removal   Disable native extreme-speed pruning for a contact-only comparison
  --no-images           Numerical results only
  --video               Encode marker videos and a side-by-side MP4 (requires --sample-every 1)
  --ffmpeg <path>        Explicit ffmpeg executable
  --headed              Show the Chrome window
  --prepared <path>      Reuse an existing prepared case.json without re-reading Blender assets

The runner builds only its isolated debug viewer, uses a private ephemeral server, and
writes results.json plus matching small-sphere views. It never changes Blender source/cache
files or switches the Fluid demo's production backend.`);
        return;
    }
    const outputValue = value("--output");
    if (!outputValue) {
        throw new Error("--output <new-directory> is required; comparison artifacts are never written over source assets.");
    }
    const output = resolve(outputValue);
    if (existsSync(output) && readdirSync(output).length > 0) {
        throw new Error("The output directory is not empty; choose a new run directory.");
    }
    mkdirSync(output, { recursive: true });
    const viewerDirectory = resolve(output, "viewer");
    const frames = integer("--frames", 51);
    const sampleEvery = integer("--sample-every", 10);
    if (args.includes("--video") && (sampleEvery !== 1 || args.includes("--no-images"))) {
        throw new Error("--video requires images and --sample-every 1 so every encoded frame has the same cadence.");
    }
    const backend = value("--backend") ?? "reference";
    if (backend !== "reference" && backend !== "production") {
        throw new Error("--backend must be reference or production.");
    }
    if (backend === "production" && (value("--tolerance") !== undefined || value("--iterations") !== undefined)) {
        throw new Error("--tolerance and --iterations configure only the reference backend, not production FLIP.");
    }
    const tolerance = Number(value("--tolerance") ?? 1e-5);
    if (!(tolerance > 0) || !Number.isFinite(tolerance)) {
        throw new Error("--tolerance must be finite and positive.");
    }
    let input: ReferenceCase;
    const prepared = value("--prepared");
    let dataDirectory = viewerDirectory;
    if (prepared) {
        if (value("--substeps") || value("--input") || value("--cache") || args.includes("--analytic")) {
            throw new Error("--prepared cannot be combined with source preparation or substep overrides.");
        }
        input = JSON.parse(readFileSync(resolve(prepared), "utf8")) as ReferenceCase;
        dataDirectory = dirname(resolve(prepared));
        if (input.frames < frames) {
            throw new Error("The prepared fixture does not contain the requested number of frames.");
        }
    } else if (args.includes("--analytic")) {
        if (value("--input") || value("--cache") || value("--substeps")) {
            throw new Error("--analytic has its own fixed preparation and cannot be combined with source/substep options.");
        }
        input = prepareAnalyticCase(viewerDirectory, frames);
    } else {
        const source = value("--input");
        const cache = value("--cache");
        if (!source || !cache) {
            throw new Error("--input and --cache are required unless --analytic or --prepared is used.");
        }
        input = prepareBlenderCase(resolve(source), resolve(cache), viewerDirectory, frames, value("--substeps") ? integer("--substeps", 2) : undefined);
    }
    await build({
        configFile: false,
        root: resolve(root, "scripts", "fluid-reference"),
        base: "./",
        logLevel: "warn",
        resolve: { alias: { "babylon-lite": resolve(root, "packages", "babylon-lite", "src", "index.ts") } },
        build: {
            outDir: viewerDirectory,
            emptyOutDir: false,
            minify: false,
            sourcemap: true,
            target: "es2022",
            rollupOptions: { output: { banner: NAME_POLYFILL } },
        },
    });
    const server = await serveDirectory(viewerDirectory);
    let browser: Browser | null = null;
    const rows: ComparisonFrame[] = [];
    let failure: string | null = null;
    const solverSettings = {
        solidGeometry: "exported",
        numerics:
            backend === "reference"
                ? {
                      referenceNumerics: true,
                      pressureTolerance: tolerance,
                      maxPressureIterations: integer("--iterations", 400),
                      removeInsideSolids: true,
                      extremeVelocityRemoval:
                          input.extremeVelocityRemoval && !args.includes("--no-extreme-removal")
                              ? { frameDtSeconds: 1 / input.simulationFps, ...input.extremeVelocityRemoval }
                              : null,
                  }
                : { pressureSolver: "multigrid", pressureTolerance: 0.001, multigridCycles: 4, reseeding: false },
    };
    const saveResults = (complete: boolean): void => {
        writeFileSync(resolve(output, "results.json"), JSON.stringify({ input, backend, solverSettings, complete, failure, frames: rows }, null, 2));
    };
    try {
        browser = await chromium.launch({ channel: "chrome", headless: !args.includes("--headed"), args: measurementBrowserArgs() });
        const page = await browser.newPage({ viewport: { width: 960, height: 540 }, deviceScaleFactor: 1 });
        page.setDefaultTimeout(300_000);
        if (prepared) {
            const allowed = new Set<string>([
                "case.json",
                input.initialState,
                ...(input.staticSdf ? [input.staticSdf.file] : []),
                ...input.obstacles.map((o) => o.file),
                ...input.meshes.flatMap((m) => [m.positions, m.indices]),
            ]);
            for (let f = input.startFrame; f < input.startFrame + frames; f++) {
                if (input.referencePattern) {
                    allowed.add(input.referencePattern.replace("{frame}", String(f)));
                }
            }
            await page.route("**/*", async (route) => {
                const name = new URL(route.request().url()).pathname.slice(1);
                if (allowed.has(name)) {
                    await route.fulfill({
                        body: readFileSync(resolve(dataDirectory, name)),
                        contentType: name.endsWith(".json") ? "application/json" : "application/octet-stream",
                    });
                } else {
                    await route.continue();
                }
            });
        }
        const search = new URLSearchParams({
            backend,
            images: args.includes("--no-images") ? "0" : "1",
            iterations: String(integer("--iterations", 400)),
            tolerance: String(tolerance),
            "extreme-removal": args.includes("--no-extreme-removal") ? "0" : "1",
        });
        await page.goto(`http://127.0.0.1:${server.port}/index.html?${search}`, { waitUntil: "domcontentloaded" });
        await page.waitForFunction(() => window.fluidReference || window.fluidReferenceError);
        const startupError = await page.evaluate(() => window.fluidReferenceError);
        if (startupError) {
            throw new Error(startupError);
        }
        const indices = [0];
        for (let index = sampleEvery; index < frames - 1; index += sampleEvery) {
            indices.push(index);
        }
        if (frames > 1) {
            indices.push(frames - 1);
        }
        for (const index of indices) {
            const row = await page.evaluate((frame) => window.fluidReference!.stepTo(frame), index);
            rows.push(row);
            const snapshot = Float32Array.from(await page.evaluate(() => window.fluidReference!.snapshot()));
            const half = snapshot.length / 2;
            writeParticleState(resolve(output, `particles-${row.frame}.bin`), { positions: snapshot.subarray(0, half), velocities: snapshot.subarray(half) });
            console.log(JSON.stringify({ frame: row.frame, seconds: row.physicalSeconds, particles: row.actual.count, comparison: row.comparison, pressure: row.pressure }));
            saveResults(false);
            if (!args.includes("--no-images")) {
                for (const kind of (row.reference ? ["actual", "reference"] : ["actual"]) as Array<"actual" | "reference">) {
                    await page.evaluate((which) => window.fluidReference!.show(which), kind);
                    await page.screenshot({
                        path: resolve(output, `frame-${String(row.frame).padStart(6, "0")}-${kind === "actual" ? backend : "blender"}.jpg`),
                        type: "jpeg",
                        quality: 60,
                    });
                }
            }
        }
        await page.evaluate(() => window.fluidReference!.dispose());
        if (args.includes("--video")) {
            await encodeComparisonVideo(output, input, backend, frames);
        }
    } catch (error) {
        failure = error instanceof Error ? (error.stack ?? error.message) : String(error);
        throw error;
    } finally {
        saveResults(failure === null);
        await browser?.close();
        await new Promise<void>((done, reject) => server.server.close((error) => (error ? reject(error) : done())));
    }
    console.log(`Comparison artifacts: ${output}`);
}

main().catch((error: unknown) => {
    console.error(error);
    process.exitCode = 1;
});
