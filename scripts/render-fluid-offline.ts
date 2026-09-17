import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { basename, dirname, extname, resolve } from "node:path";
import type { Server } from "node:http";
import { spawn } from "node:child_process";

import { chromium } from "@playwright/test";
import type { Browser } from "@playwright/test";

import { labDir, measurementBrowserArgs, startStaticServer } from "./bundle-scenes-core";

interface OfflineRenderOptions {
    input: string;
    output: string;
    frames: number;
    fixedDt: number;
    stepsPerFrame: number;
    outputFps: number;
    width: number;
    height: number;
    headed: boolean;
    encodeVideo: boolean;
    ffmpeg: string;
    url?: string;
}

interface FluidSceneManifest {
    encoding?: string;
    glb?: string;
    collision?: string;
    animatedCollisions?: Array<{ sdf?: string }>;
}

interface FluidManifest {
    scene?: FluidSceneManifest;
    source?: {
        settings?: {
            timeline?: {
                fps?: number;
                fpsBase?: number;
                simulationFps?: number;
            };
            domain?: {
                simulation?: {
                    frame_rate_mode?: string;
                    frame_rate_custom?: number;
                };
            };
        };
    };
}

interface OfflineFluidSummary {
    method: string;
    gridResolution: number;
    gridCells: [number, number, number];
    gridSize: [number, number, number];
    gridCellSize: number;
    markersPerCell: number | null;
    activeParticles: number;
    particleCapacity: number;
    simulationGpuBytes: number;
    deviceBufferLimitBytes: number;
    deviceParticleCapacity: number;
    deviceParticleBytesPerSlot: number;
    pagedGrid: boolean;
    residentPages: number | null;
    pageCapacity: number;
    pressureSolver: string | null;
    renderMode: string;
    polygonSurface: boolean;
    foamEnabled: boolean;
    foamCapacity: number;
}

function argumentValue(args: readonly string[], name: string): string | undefined {
    const index = args.indexOf(name);
    return index >= 0 ? args[index + 1] : undefined;
}

function positiveInteger(value: string | undefined, fallback: number, name: string): number {
    const parsed = value === undefined ? fallback : Number(value);
    if (!Number.isSafeInteger(parsed) || parsed <= 0) {
        throw new RangeError(`${name} must be a positive integer.`);
    }
    return parsed;
}

function positiveNumber(value: string | undefined, fallback: number, name: string): number {
    const parsed = value === undefined ? fallback : Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new RangeError(`${name} must be a finite positive number.`);
    }
    return parsed;
}

function parseArguments(args: readonly string[]): OfflineRenderOptions {
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage:
  pnpm render:fluid-offline -- --input <fluid.json> [options]

Options:
  --output <directory>       Output directory (default: <input>-frames)
  --frames <count>           Number of PNG frames (default: 300)
  --fixed-dt <seconds>       Simulation timestep (default: exported simulation FPS or 1/60)
  --steps-per-frame <count>  Simulation steps between PNG frames (default: 1)
  --output-fps <rate>        Encoded video frame rate (default: exported render FPS or simulation cadence)
  --width <pixels>           Output width (default: 1280)
  --height <pixels>          Output height (default: 720)
  --url <demo-url>           Use an existing Fluid demo server instead of an ephemeral server
  --ffmpeg <executable>      ffmpeg executable path (default: PATH or standard Windows install)
  --no-video                 Keep only the PNG sequence; skip automatic ffmpeg encoding
  --headed                    Show the Chrome window`);
        process.exit(0);
    }
    const inputValue = argumentValue(args, "--input");
    if (!inputValue) {
        throw new Error("--input <fluid.json> is required.");
    }
    const input = resolve(inputValue);
    if (!existsSync(input) || extname(input).toLowerCase() !== ".json") {
        throw new Error(`Fluid JSON was not found: ${input}`);
    }
    const manifest = JSON.parse(readFileSync(input, "utf8")) as FluidManifest;
    const timeline = manifest.source?.settings?.timeline;
    const renderFps =
        typeof timeline?.fps === "number" && Number.isFinite(timeline.fps) && timeline.fps > 0
            ? timeline.fps / (typeof timeline.fpsBase === "number" && Number.isFinite(timeline.fpsBase) && timeline.fpsBase > 0 ? timeline.fpsBase : 1)
            : undefined;
    const rawSimulationFps =
        typeof timeline?.simulationFps === "number"
            ? timeline.simulationFps
            : manifest.source?.settings?.domain?.simulation?.frame_rate_mode === "FRAME_RATE_MODE_CUSTOM"
              ? manifest.source.settings.domain.simulation.frame_rate_custom
              : renderFps;
    const simulationFps = typeof rawSimulationFps === "number" && Number.isFinite(rawSimulationFps) && rawSimulationFps > 0 ? rawSimulationFps : undefined;
    const outputValue = argumentValue(args, "--output");
    const ffmpegValue = argumentValue(args, "--ffmpeg");
    const standardWindowsFfmpeg = process.platform === "win32" ? resolve(process.env.ProgramFiles ?? "C:\\Program Files", "ffmpeg", "bin", "ffmpeg.exe") : undefined;
    const ffmpeg = ffmpegValue
        ? resolve(ffmpegValue)
        : process.env.FFMPEG_PATH
          ? resolve(process.env.FFMPEG_PATH)
          : standardWindowsFfmpeg && existsSync(standardWindowsFfmpeg)
            ? standardWindowsFfmpeg
            : "ffmpeg";
    if (ffmpegValue && !existsSync(ffmpeg)) {
        throw new Error(`ffmpeg executable was not found: ${ffmpeg}`);
    }
    const fixedDt = positiveNumber(argumentValue(args, "--fixed-dt"), simulationFps ? 1 / simulationFps : 1 / 60, "--fixed-dt");
    const stepsPerFrame = positiveInteger(argumentValue(args, "--steps-per-frame"), 1, "--steps-per-frame");
    return {
        input,
        output: resolve(outputValue ?? resolve(dirname(input), `${basename(input, extname(input))}-frames`)),
        frames: positiveInteger(argumentValue(args, "--frames"), 300, "--frames"),
        fixedDt,
        stepsPerFrame,
        outputFps: positiveNumber(argumentValue(args, "--output-fps"), renderFps ?? 1 / (fixedDt * stepsPerFrame), "--output-fps"),
        width: positiveInteger(argumentValue(args, "--width"), 1280, "--width"),
        height: positiveInteger(argumentValue(args, "--height"), 720, "--height"),
        headed: args.includes("--headed"),
        encodeVideo: !args.includes("--no-video"),
        ffmpeg,
        ...(argumentValue(args, "--url") ? { url: argumentValue(args, "--url") } : {}),
    };
}

function selectedFluidFiles(input: string): string[] {
    const contents = readFileSync(input, "utf8");
    const manifest = JSON.parse(contents) as FluidManifest;
    const references = new Set<string>();
    const scene = manifest.scene;
    if (scene?.encoding === "external") {
        for (const value of [scene.glb, scene.collision, ...(scene.animatedCollisions ?? []).map((entry) => entry.sdf)]) {
            if (typeof value === "string" && value) {
                references.add(value.replaceAll("\\", "/"));
            }
        }
    }
    const files = [input];
    for (const reference of references) {
        if (reference.startsWith("/") || /^[a-z][a-z0-9+.-]*:/i.test(reference) || reference.split("/").includes("..")) {
            throw new Error(`External resource must be a safe relative path: ${reference}`);
        }
        const path = resolve(dirname(input), reference);
        if (!existsSync(path)) {
            throw new Error(`External fluid resource was not found: ${path}`);
        }
        files.push(path);
    }
    return files;
}

function framePath(output: string, frame: number): string {
    return resolve(output, `frame-${String(frame).padStart(6, "0")}.png`);
}

function formatInteger(value: number): string {
    return Math.max(0, Math.floor(value)).toLocaleString("en-US");
}

function formatScalar(value: number): string {
    return Number(value.toPrecision(12)).toString();
}

function formatBytes(bytes: number): string {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB (${formatInteger(bytes)} bytes)`;
}

function printOfflineSummary(summary: OfflineFluidSummary, options: OfflineRenderOptions): void {
    const totalCells = summary.gridCells[0] * summary.gridCells[1] * summary.gridCells[2];
    console.log("Fluid offline startup:");
    console.log(`  Method: ${summary.method}`);
    console.log(`  Resolution divisions: ${formatInteger(summary.gridResolution)}`);
    console.log(`  Grid: ${summary.gridCells.join(" x ")} cells (${formatInteger(totalCells)} total)`);
    console.log(`  Grid size: ${summary.gridSize.map(formatScalar).join(" x ")}`);
    console.log(`  Cell size: ${formatScalar(summary.gridCellSize)}`);
    if (summary.markersPerCell !== null) {
        console.log(`  Markers per cell: ${formatInteger(summary.markersPerCell)}`);
    }
    console.log(`  Particles: ${formatInteger(summary.activeParticles)} active / ${formatInteger(summary.particleCapacity)} total capacity`);
    console.log(`  Simulation GPU allocation: ${formatBytes(summary.simulationGpuBytes)}`);
    console.log(
        `  Device per-buffer limit: ${formatBytes(summary.deviceBufferLimitBytes)}; ${formatInteger(summary.deviceParticleCapacity)} ${summary.method} particle slots at ${formatInteger(
            summary.deviceParticleBytesPerSlot
        )} bytes/slot`
    );
    console.log(
        summary.pagedGrid
            ? `  Grid storage: paged (${summary.residentPages === null ? "resident pages pending" : `${formatInteger(summary.residentPages)} resident pages`}, ${formatInteger(
                  summary.pageCapacity
              )} page capacity)`
            : "  Grid storage: dense"
    );
    if (summary.pressureSolver) {
        console.log(`  Pressure solver: ${summary.pressureSolver}`);
    }
    console.log(
        `  Rendering: ${summary.renderMode}${summary.polygonSurface ? ", polygon surface" : ""}${summary.foamEnabled ? `, foam capacity ${formatInteger(summary.foamCapacity)}` : ", foam disabled"}`
    );
    console.log(
        `  Output: ${formatInteger(options.frames)} ${options.frames === 1 ? "frame" : "frames"}, ${formatInteger(options.stepsPerFrame)} ${
            options.stepsPerFrame === 1 ? "step" : "steps"
        }/frame, fixed dt ${formatScalar(options.fixedDt)} s, video ${formatScalar(options.outputFps)} FPS, ${formatInteger(options.width)} x ${formatInteger(options.height)}`
    );
}

async function encodeOfflineVideo(options: OfflineRenderOptions, frameCount: number): Promise<string> {
    const output = resolve(options.output, "fluid-offline.mp4");
    const args = [
        "-y",
        "-framerate",
        options.outputFps.toFixed(9),
        "-start_number",
        "1",
        "-i",
        resolve(options.output, "frame-%06d.png"),
        "-frames:v",
        String(frameCount),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output,
    ];
    console.log(`Encoding ${output}`);
    await new Promise<void>((resolveEncoding, rejectEncoding) => {
        const child = spawn(options.ffmpeg, args, { stdio: "inherit", windowsHide: true });
        child.once("error", (error: NodeJS.ErrnoException) => {
            rejectEncoding(
                error.code === "ENOENT"
                    ? new Error(`ffmpeg was not found at "${options.ffmpeg}". Add it to PATH, pass --ffmpeg <path>, or rerun with --no-video to keep only the PNG sequence.`)
                    : error
            );
        });
        child.once("close", (code, signal) => {
            if (code === 0) {
                resolveEncoding();
            } else {
                rejectEncoding(new Error(`ffmpeg failed${signal ? ` with signal ${signal}` : ` with exit code ${code ?? "unknown"}`}. PNG frames remain in ${options.output}.`));
            }
        });
    });
    return output;
}

async function closeServer(server: Server | undefined): Promise<void> {
    if (!server) {
        return;
    }
    await new Promise<void>((resolveClose, reject) => {
        server.close((error) => (error ? reject(error) : resolveClose()));
    });
}

async function main(): Promise<void> {
    const options = parseArguments(process.argv.slice(2));
    const files = selectedFluidFiles(options.input);
    mkdirSync(options.output, { recursive: true });

    let server: Server | undefined;
    let browser: Browser | undefined;
    let capturedFrames = 0;
    try {
        let demoUrl = options.url;
        if (!demoUrl) {
            const bundle = resolve(labDir, "public/bundle/demos/fluid.js");
            if (!existsSync(bundle)) {
                throw new Error("The Fluid demo bundle is missing. Run `pnpm build:bundle-demo fluid --debug` first.");
            }
            const started = await startStaticServer(labDir);
            server = started.server;
            demoUrl = `http://127.0.0.1:${started.port}/lite/demo-fluid.html`;
        }

        const url = new URL(demoUrl);
        url.searchParams.set("offline", "1");
        url.searchParams.set("fixedDt", String(options.fixedDt));
        url.searchParams.set("demo", "whiteboard");
        url.searchParams.set("method", "PBF");
        url.searchParams.set("quality", "middle");

        browser = await chromium.launch({
            channel: "chrome",
            headless: !options.headed,
            args: measurementBrowserArgs(),
        });
        const page = await browser.newPage({ viewport: { width: options.width, height: options.height }, deviceScaleFactor: 1 });
        page.on("dialog", (dialog) => void dialog.dismiss());
        page.on("console", (message) => {
            if (message.type() === "error" && !message.text().startsWith("Failed to load resource:")) {
                console.error(`[browser] ${message.text()}`);
            }
        });
        page.on("pageerror", (error) => console.error(`[browser] ${error.message}`));

        await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 120_000 });
        await page.waitForFunction(() => document.querySelector<HTMLCanvasElement>("#renderCanvas")?.dataset.offlineReady === "true", undefined, { timeout: 120_000 });
        const input = page.locator('input[type="file"][accept*=".sdf"]');
        await input.setInputFiles(files);
        await page.waitForFunction(
            () => {
                const status = document.querySelector<HTMLCanvasElement>("#renderCanvas")?.dataset.presetImportStatus;
                return status === "ready" || status === "error";
            },
            undefined,
            { timeout: 0 }
        );
        const importState = await page.locator("#renderCanvas").evaluate((canvas: HTMLCanvasElement) => ({
            status: canvas.dataset.presetImportStatus,
            error: canvas.dataset.presetImportError,
        }));
        if (importState.status !== "ready") {
            throw new Error(`Fluid preset import failed: ${importState.error ?? "unknown error"}`);
        }
        const summaryJson = await page.locator("#renderCanvas").evaluate((canvas: HTMLCanvasElement) => canvas.dataset.offlineSummary);
        if (!summaryJson) {
            throw new Error("The Fluid demo did not publish its offline startup summary. Rebuild the Fluid demo bundle in debug mode.");
        }
        const summary = JSON.parse(summaryJson) as OfflineFluidSummary;
        printOfflineSummary(summary, options);

        const startedAt = performance.now();
        for (let frame = 1; frame <= options.frames; frame++) {
            const token = `frame-${frame}`;
            await page.locator("#renderCanvas").evaluate(
                (canvas: HTMLCanvasElement, detail) => {
                    canvas.dispatchEvent(new CustomEvent("fluid-offline-run", { detail }));
                },
                { token, steps: options.stepsPerFrame, render: true }
            );
            await page.waitForFunction(
                (expectedToken) => {
                    const data = document.querySelector<HTMLCanvasElement>("#renderCanvas")?.dataset;
                    return data?.offlineCompletedToken === expectedToken || data?.offlineErrorToken === expectedToken;
                },
                token,
                { timeout: 0 }
            );
            const state = await page.locator("#renderCanvas").evaluate((canvas: HTMLCanvasElement) => ({
                completedToken: canvas.dataset.offlineCompletedToken,
                errorToken: canvas.dataset.offlineErrorToken,
                error: canvas.dataset.offlineError,
                simulationTime: Number(canvas.dataset.offlineSimulationTime ?? 0),
                skippedSeconds: Number(canvas.dataset.referenceDroppedSeconds ?? 0),
                stopped: canvas.dataset.simulationStopped === "true",
            }));
            if (state.errorToken === token) {
                throw new Error(`Offline step failed: ${state.error ?? "unknown error"}`);
            }
            await page.locator("#renderCanvas").screenshot({ path: framePath(options.output, frame) });
            capturedFrames = frame;
            const elapsedSeconds = (performance.now() - startedAt) / 1000;
            const simulationProgress =
                state.skippedSeconds > 0
                    ? `requested ${state.simulationTime.toFixed(3)}s  simulated ${Math.max(0, state.simulationTime - state.skippedSeconds).toFixed(3)}s  skipped ${state.skippedSeconds.toFixed(3)}s (substep budget)`
                    : `simulation ${state.simulationTime.toFixed(3)}s`;
            console.log(`frame ${frame}/${options.frames}  ${simulationProgress}  elapsed ${elapsedSeconds.toFixed(1)}s`);
            if (state.stopped && frame < options.frames) {
                console.warn("Simulation lifecycle stopped before the requested output frame count.");
                break;
            }
        }
    } finally {
        await browser?.close();
        await closeServer(server);
    }
    console.log(`${capturedFrames} frame${capturedFrames === 1 ? "" : "s"} written to ${options.output}`);
    if (options.encodeVideo) {
        const video = await encodeOfflineVideo(options, capturedFrames);
        console.log(`Video written to ${video}`);
    }
}

main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
});
