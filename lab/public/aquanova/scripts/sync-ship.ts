/**
 * Publish the Aquanova editor's ship export into `lab/public/aquanova`.
 *
 * The ship is compressed to runtime-ready Meshopt/KTX2 data, while the authored manifest,
 * collision hulls, and converted local environment probes are copied alongside it.
 *
 * Usage:
 *   pnpm tsx lab/public/aquanova/scripts/sync-ship.ts [options]
 */

import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, copyFile, readFile, rm, writeFile, stat, access } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SRC = path.resolve(HERE, "../../../lite/src/demos/aquanova/editor/export");
const DEFAULT_OUT = path.resolve(HERE, "..");
const REPO_ROOT = path.resolve(HERE, "../../../..");
const execFileAsync = promisify(execFile);

interface SourceLocalEnvironmentBase {
    env: string;
    position: [number, number, number];
    resolution: number;
    hash: string;
}

type SourceLocalEnvironment = SourceLocalEnvironmentBase &
    (
        | {
              shape?: "box";
              boxPosition: [number, number, number];
              boxSize: [number, number, number];
              angle?: number;
          }
        | {
              shape: "sphere";
              spherePosition: [number, number, number];
              sphereRadius: number;
          }
    );

interface SourceLocalEnvironmentIndex {
    probes?: Record<string, SourceLocalEnvironment>;
    chunks?: Record<string, SourceLocalEnvironment>;
}

interface RuntimeLocalEnvironmentBase {
    url: string;
    position: [number, number, number];
    resolution: number;
    bytes: number;
}

type RuntimeLocalEnvironment = RuntimeLocalEnvironmentBase &
    (
        | {
              shape?: "box";
              boxPosition: [number, number, number];
              boxSize: [number, number, number];
              influenceBoxPosition: [number, number, number];
              influenceBoxSize: [number, number, number];
              influenceInnerBoxSize: [number, number, number];
              angle: number;
          }
        | {
              shape: "sphere";
              spherePosition: [number, number, number];
              sphereRadius: number;
              influenceSpherePosition: [number, number, number];
              influenceSphereRadius: number;
              influenceInnerSphereRadius: number;
          }
    );

interface RuntimeLocalEnvironmentIndex {
    probes: Record<string, RuntimeLocalEnvironment>;
}

const DEFAULT_BLEND_DISTANCE = 1.5;

interface ShipManifest {
    environmentProbes?: Array<{
        id?: string;
        shape?: "box" | "sphere";
        boxPosition?: [number, number, number];
        boxSize?: [number, number, number];
        spherePosition?: [number, number, number];
        sphereRadius?: number;
        capturePosition?: [number, number, number];
        angle?: number;
        influenceBoxPosition?: [number, number, number];
        influenceBoxSize?: [number, number, number];
        influenceInnerBoxSize?: [number, number, number];
        influenceSpherePosition?: [number, number, number];
        influenceSphereRadius?: number;
        influenceInnerSphereRadius?: number;
    }>;
    chunks?: Array<{
        id?: string;
        environmentProbe?: {
            boxPosition?: [number, number, number];
            boxSize?: [number, number, number];
        };
    }>;
}

/**
 * What the manifest says about a probe, which is the authored truth. The
 * generated index only records what was captured, in editor space, so
 * everything the author can type belongs here instead.
 */
interface AuthoredProbeBase {
    capturePosition?: [number, number, number];
}

type AuthoredProbe = AuthoredProbeBase &
    (
        | {
              shape: "box";
              boxPosition: [number, number, number];
              boxSize: [number, number, number];
              angle?: number;
              influenceBoxPosition?: [number, number, number];
              influenceBoxSize?: [number, number, number];
              influenceInnerBoxSize?: [number, number, number];
          }
        | {
              shape: "sphere";
              spherePosition: [number, number, number];
              sphereRadius: number;
              influenceSpherePosition?: [number, number, number];
              influenceSphereRadius?: number;
              influenceInnerSphereRadius?: number;
          }
    );

interface Options {
    src: string;
    out: string;
    force: boolean;
    shipOptimize: boolean;
    shipUastcLevel: number;
    shipEtc1sQuality: number;
    shipZstd: number;
}

function parseArgs(argv: readonly string[]): Options {
    const opts: Options = {
        src: DEFAULT_SRC,
        out: DEFAULT_OUT,
        force: false,
        shipOptimize: true,
        shipUastcLevel: 2,
        shipEtc1sQuality: 200,
        shipZstd: 18,
    };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i]!;
        if (a === "--force") {
            opts.force = true;
        } else if (a === "--no-ship-optimize") {
            opts.shipOptimize = false;
        } else if (a === "--ship-uastc-level") {
            opts.shipUastcLevel = Number(argv[++i]);
        } else if (a === "--ship-etc1s-quality") {
            opts.shipEtc1sQuality = Number(argv[++i]);
        } else if (a === "--ship-zstd") {
            opts.shipZstd = Number(argv[++i]);
        } else if (a === "--src") {
            opts.src = path.resolve(argv[++i]!);
        } else if (a === "--out") {
            opts.out = path.resolve(argv[++i]!);
        } else {
            throw Error(`unknown option ${a}`);
        }
    }
    return opts;
}

function assertIntegerRange(name: string, value: number, min: number, max: number): void {
    if (!Number.isInteger(value) || value < min || value > max) {
        throw Error(`${name} must be an integer between ${min} and ${max}`);
    }
}

function mb(bytes: number): string {
    return `${(bytes / 1048576).toFixed(2)} MB`;
}

/** Copy only when the destination is missing or differs in size/mtime. */
async function copyIfChanged(src: string, dst: string): Promise<boolean> {
    const s = await stat(src);
    const d = await stat(dst).catch(() => null);
    if (d && d.size === s.size && d.mtimeMs >= s.mtimeMs) {
        return false;
    }
    await copyFile(src, dst);
    return true;
}

async function readLocalEnvironmentIndex(sourceDir: string): Promise<SourceLocalEnvironmentIndex | null> {
    try {
        return JSON.parse(await readFile(path.join(sourceDir, "local-environments.json"), "utf8")) as SourceLocalEnvironmentIndex;
    } catch (err) {
        if ((err as NodeJS.ErrnoException).code === "ENOENT") {
            return null;
        }
        throw err;
    }
}

async function readPublishedLocalEnvironmentIndex(outputDir: string): Promise<RuntimeLocalEnvironmentIndex | null> {
    try {
        return JSON.parse(await readFile(path.join(outputDir, "local-environments.json"), "utf8")) as RuntimeLocalEnvironmentIndex;
    } catch (err) {
        if ((err as NodeJS.ErrnoException).code === "ENOENT") {
            return null;
        }
        throw err;
    }
}

function defaultInfluenceSizes(boxSize: readonly number[]): {
    inner: [number, number, number];
    outer: [number, number, number];
} {
    return {
        inner: [
            Math.max(0, boxSize[0]! - DEFAULT_BLEND_DISTANCE * 2),
            Math.max(0, boxSize[1]! - DEFAULT_BLEND_DISTANCE * 2),
            Math.max(0, boxSize[2]! - DEFAULT_BLEND_DISTANCE * 2),
        ],
        outer: [boxSize[0]! + DEFAULT_BLEND_DISTANCE * 2, boxSize[1]! + DEFAULT_BLEND_DISTANCE * 2, boxSize[2]! + DEFAULT_BLEND_DISTANCE * 2],
    };
}

function defaultInfluenceRadii(radius: number): { inner: number; outer: number } {
    return {
        inner: Math.max(0, radius - DEFAULT_BLEND_DISTANCE),
        outer: radius + DEFAULT_BLEND_DISTANCE,
    };
}

function authoredVector(value: readonly number[] | undefined, bound?: "positive" | "nonNegative"): [number, number, number] | undefined {
    if (!value || value.length !== 3 || !value.every(Number.isFinite)) {
        return undefined;
    }
    if (bound === "positive" && !value.every((n) => n > 0)) {
        return undefined;
    }
    if (bound === "nonNegative" && !value.every((n) => n >= 0)) {
        return undefined;
    }
    return [value[0]!, value[1]!, value[2]!];
}

function authoredLocalEnvironmentProbes(manifest: ShipManifest): Map<string, AuthoredProbe> {
    const probes = new Map<string, AuthoredProbe>();
    const explicit = manifest.environmentProbes;
    if (explicit) {
        for (const probe of explicit) {
            if (!probe.id) {
                continue;
            }
            if (probe.shape === "sphere") {
                const position = authoredVector(probe.spherePosition);
                const radius = Number(probe.sphereRadius);
                const influencePosition = authoredVector(probe.influenceSpherePosition);
                const influenceRadius = Number(probe.influenceSphereRadius);
                const innerRadius = Number(probe.influenceInnerSphereRadius);
                if (!position || !Number.isFinite(radius) || radius <= 0) {
                    continue;
                }
                probes.set(probe.id, {
                    shape: "sphere",
                    spherePosition: position,
                    sphereRadius: radius,
                    capturePosition: authoredVector(probe.capturePosition),
                    influenceSpherePosition: influencePosition,
                    influenceSphereRadius: Number.isFinite(influenceRadius) && influenceRadius > 0 ? influenceRadius : undefined,
                    influenceInnerSphereRadius:
                        Number.isFinite(innerRadius) && innerRadius >= 0 && (!Number.isFinite(influenceRadius) || innerRadius <= influenceRadius) ? innerRadius : undefined,
                });
                continue;
            }
            const position = probe.boxPosition;
            const size = probe.boxSize;
            if (
                !position ||
                !size ||
                position.length !== 3 ||
                size.length !== 3 ||
                ![...position, ...size].every(Number.isFinite) ||
                !size.every((value) => value > 0)
            ) {
                continue;
            }
            const influenceBoxSize = authoredVector(probe.influenceBoxSize, "positive");
            const influenceInnerBoxSize = authoredVector(probe.influenceInnerBoxSize, "nonNegative");
            const angle = Number(probe.angle ?? 0);
            probes.set(probe.id, {
                shape: "box",
                boxPosition: [...position],
                boxSize: [...size],
                capturePosition: authoredVector(probe.capturePosition),
                angle: Number.isFinite(angle) ? angle : 0,
                influenceBoxPosition: authoredVector(probe.influenceBoxPosition),
                influenceBoxSize,
                // An inner box larger than the outer one would make the runtime
                // divide by a negative width, so it is dropped rather than
                // published; the derived default takes over.
                influenceInnerBoxSize: influenceBoxSize && influenceInnerBoxSize?.every((n, axis) => n <= influenceBoxSize[axis]!) ? influenceInnerBoxSize : undefined,
            });
        }
        return probes;
    }
    for (const chunk of manifest.chunks || []) {
        const position = chunk.environmentProbe?.boxPosition;
        const size = chunk.environmentProbe?.boxSize;
        if (!chunk.id || !position || !size || position.length !== 3 || size.length !== 3 || ![...position, ...size].every(Number.isFinite) || !size.every((value) => value > 0)) {
            continue;
        }
        probes.set(chunk.id, {
            shape: "box",
            boxPosition: [...position],
            boxSize: [...size],
        });
    }
    return probes;
}

async function assertLocalEnvironmentsReady(sourceDir: string, index: SourceLocalEnvironmentIndex): Promise<void> {
    const problems: string[] = [];
    for (const [probeId, probe] of Object.entries(index.probes ?? index.chunks ?? {})) {
        const source = path.join(sourceDir, probe.env);
        try {
            await access(source);
        } catch (err) {
            if ((err as NodeJS.ErrnoException).code !== "ENOENT") {
                throw err;
            }
            problems.push(`${probeId} (missing ${probe.env})`);
            continue;
        }
        const stamp = await readFile(`${source}.stamp`, "utf8").catch((err: NodeJS.ErrnoException) => {
            if (err.code === "ENOENT") {
                return "";
            }
            throw err;
        });
        if (stamp.trim() !== probe.hash) {
            problems.push(`${probeId} (stale or unstamped ${probe.env})`);
        }
    }
    if (problems.length) {
        throw Error(
            `Local environment capture is incomplete: ${problems.join(", ")}. ` + "Open the ship editor, press Capture to render the missing probes, then run this sync again."
        );
    }
}

async function existingFile(file: string): Promise<string | undefined> {
    try {
        await access(file);
        return file;
    } catch {
        return undefined;
    }
}

async function findToktx(): Promise<string | undefined> {
    const explicit = process.env.TOKTX_PATH;
    if (explicit) {
        return existingFile(path.resolve(explicit));
    }

    const binPath = process.env.KTX_SOFTWARE_PATH;
    const roots = [
        ...(binPath ? [binPath] : []),
        process.env.ProgramFiles ? path.join(process.env.ProgramFiles, "KTX-Software", "bin") : undefined,
        process.env["ProgramFiles(x86)"] ? path.join(process.env["ProgramFiles(x86)"]!, "KTX-Software", "bin") : undefined,
        process.env.LOCALAPPDATA ? path.join(process.env.LOCALAPPDATA, "KTX-Software", "bin") : undefined,
        process.env.LOCALAPPDATA ? path.join(process.env.LOCALAPPDATA, "Programs", "KTX-Software", "bin") : undefined,
    ].filter((value): value is string => Boolean(value));
    const executable = process.platform === "win32" ? "toktx.exe" : "toktx";

    for (const root of roots) {
        const found = await existingFile(path.join(root, executable));
        if (found) {
            return found;
        }
    }

    return undefined;
}

async function encoderEnvironment(): Promise<NodeJS.ProcessEnv> {
    const toktx = await findToktx();
    if (!toktx) {
        return process.env;
    }
    const bin = path.dirname(toktx);
    const separator = process.platform === "win32" ? ";" : ":";
    return { ...process.env, PATH: `${bin}${separator}${process.env.PATH || ""}` };
}

async function runGltfTransform(args: readonly string[], label: string): Promise<void> {
    const command = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
    try {
        const env = await encoderEnvironment();
        await execFileAsync(command, ["dlx", "--yes", "@gltf-transform/cli@4.4.2", ...args], {
            cwd: REPO_ROOT,
            maxBuffer: 16 * 1024 * 1024,
            shell: process.platform === "win32",
            env,
        });
    } catch (error) {
        const detail =
            error instanceof Error ? `${error.message}${typeof error === "object" && error !== null && "stderr" in error ? `\n${String(error.stderr)}` : ""}` : String(error);
        throw Error(`glTF Transform ${label} failed. Ensure KTX-Software's "toktx" is available, then retry. ${detail}`);
    }
}

async function publishShipGlb(
    source: string,
    destination: string,
    opts: Pick<Options, "shipOptimize" | "shipUastcLevel" | "shipEtc1sQuality" | "shipZstd" | "force">
): Promise<void> {
    if (!opts.shipOptimize) {
        const copied = await copyIfChanged(source, destination);
        console.log(`${copied ? "copied " : "current"}  ${path.basename(destination)}  (unoptimized)`);
        return;
    }

    const sourceBytes = await readFile(source);
    const stamp = createHash("md5")
        .update(sourceBytes)
        .update(JSON.stringify([opts.shipOptimize, opts.shipUastcLevel, opts.shipEtc1sQuality, opts.shipZstd]))
        .digest("hex");
    const stampFile = `${destination}.stamp`;
    const previous = await readFile(stampFile, "utf8").catch(() => "");
    const destinationExists = await stat(destination)
        .then(() => true)
        .catch(() => false);
    if (!opts.force && destinationExists && previous === stamp) {
        console.log(`current  ${path.basename(destination)}  (Meshopt/KTX2)`);
        return;
    }

    const temp = await mkdtemp(path.join(process.env.TEMP || process.cwd(), "aquanova-ship-"));
    const meshopt = path.join(temp, "meshopt.glb");
    const uastc = path.join(temp, "uastc.glb");
    const output = path.join(temp, "output.glb");
    try {
        await runGltfTransform(["meshopt", source, meshopt, "--level", "high"], "Meshopt geometry compression");
        await runGltfTransform(
            ["uastc", meshopt, uastc, "--slots", "normalTexture", "--level", String(opts.shipUastcLevel), "--zstd", String(opts.shipZstd), "--jobs", "1"],
            "ship UASTC texture compression"
        );
        await runGltfTransform(
            [
                "etc1s",
                uastc,
                output,
                "--slots",
                "{baseColorTexture,metallicRoughnessTexture,occlusionTexture,emissiveTexture}",
                "--quality",
                String(opts.shipEtc1sQuality),
                "--compression",
                "4",
                "--jobs",
                "1",
            ],
            "ship ETC1S texture compression"
        );

        await copyFile(output, destination);
        await writeFile(stampFile, stamp);
        const outputBytes = (await stat(destination)).size;
        console.log(`optimized  ${path.basename(destination)}  ${mb(sourceBytes.byteLength)} -> ${mb(outputBytes)}`);
    } finally {
        await rm(temp, { recursive: true, force: true });
    }
}

async function main(): Promise<void> {
    const opts = parseArgs(process.argv.slice(2));
    assertIntegerRange("--ship-uastc-level", opts.shipUastcLevel, 0, 4);
    assertIntegerRange("--ship-etc1s-quality", opts.shipEtc1sQuality, 1, 255);
    assertIntegerRange("--ship-zstd", opts.shipZstd, 0, 22);
    const generatedDir = path.join(opts.src, "environments");
    const localSource = await readLocalEnvironmentIndex(generatedDir);
    const manifest = JSON.parse(await readFile(path.join(opts.src, "ship_manifest.json"), "utf8")) as ShipManifest;
    const authoredProbes = authoredLocalEnvironmentProbes(manifest);
    if (localSource) {
        await assertLocalEnvironmentsReady(generatedDir, localSource);
    }

    const shipName = "ship.glb";
    const shipSource = path.join(opts.src, shipName);
    const shipDestination = path.join(opts.out, shipName);
    await publishShipGlb(shipSource, shipDestination, opts);
    for (const name of ["ship_manifest.json", "ship_collision.json"]) {
        const copied = await copyIfChanged(path.join(opts.src, name), path.join(opts.out, name));
        console.log(`${copied ? "copied " : "current"}  ${name}`);
    }

    if (localSource) {
        const outEnvironments = path.join(opts.out, "environments");
        await mkdir(outEnvironments, { recursive: true });
        const published = await readPublishedLocalEnvironmentIndex(opts.out);
        const localRuntime: RuntimeLocalEnvironmentIndex = { probes: {} };
        for (const [probeId, probe] of Object.entries(localSource.probes ?? localSource.chunks ?? {})) {
            const authored = authoredProbes.get(probeId);
            const source = path.join(generatedDir, probe.env);
            const destination = path.join(outEnvironments, `${probeId}.env`);
            const copied = await copyIfChanged(source, destination);
            const bytes = (await stat(destination)).size;
            console.log(`${copied ? "copied " : "current"}  environments/${probeId}.env  ${mb(bytes)}`);
            const previous = published?.probes[probeId];
            const common = {
                url: `environments/${probeId}.env`,
                // The generated index is in editor space, where X runs the
                // other way; the runtime mirrors what it reads here, so the
                // manifest - already glTF space - is the only safe source.
                position: authored?.capturePosition ?? probe.position,
                resolution: probe.resolution,
                bytes,
            };
            const shape = authored?.shape ?? probe.shape ?? "box";
            if (shape === "sphere") {
                const spherePosition =
                    authored?.shape === "sphere" ? authored.spherePosition : probe.shape === "sphere" ? probe.spherePosition : undefined;
                const sphereRadius =
                    authored?.shape === "sphere" ? authored.sphereRadius : probe.shape === "sphere" ? probe.sphereRadius : undefined;
                if (!spherePosition || !Number.isFinite(sphereRadius) || sphereRadius! <= 0) {
                    throw new Error(`environment probe ${probeId} has invalid spherical projection data`);
                }
                const defaults = defaultInfluenceRadii(sphereRadius!);
                const previousSphere = previous?.shape === "sphere" ? previous : undefined;
                const outerRadius =
                    (authored?.shape === "sphere" ? authored.influenceSphereRadius : undefined) ?? previousSphere?.influenceSphereRadius ?? defaults.outer;
                const innerRadius =
                    (authored?.shape === "sphere" ? authored.influenceInnerSphereRadius : undefined) ?? previousSphere?.influenceInnerSphereRadius ?? defaults.inner;
                localRuntime.probes[probeId] = {
                    ...common,
                    shape: "sphere",
                    spherePosition,
                    sphereRadius: sphereRadius!,
                    influenceSpherePosition:
                        (authored?.shape === "sphere" ? authored.influenceSpherePosition : undefined) ?? previousSphere?.influenceSpherePosition ?? spherePosition,
                    influenceSphereRadius: outerRadius,
                    influenceInnerSphereRadius: Math.min(innerRadius, outerRadius),
                };
            } else {
                const boxPosition = authored?.shape === "box" ? authored.boxPosition : probe.shape !== "sphere" ? probe.boxPosition : undefined;
                const boxSize = authored?.shape === "box" ? authored.boxSize : probe.shape !== "sphere" ? probe.boxSize : undefined;
                if (!boxPosition || !boxSize) {
                    throw new Error(`environment probe ${probeId} has invalid box projection data`);
                }
                const defaults = defaultInfluenceSizes(boxSize);
                const previousBox = previous?.shape !== "sphere" ? previous : undefined;
                localRuntime.probes[probeId] = {
                    ...common,
                    boxPosition,
                    boxSize,
                    influenceBoxPosition: (authored?.shape === "box" ? authored.influenceBoxPosition : undefined) ?? previousBox?.influenceBoxPosition ?? boxPosition,
                    influenceBoxSize: (authored?.shape === "box" ? authored.influenceBoxSize : undefined) ?? previousBox?.influenceBoxSize ?? defaults.outer,
                    influenceInnerBoxSize:
                        (authored?.shape === "box" ? authored.influenceInnerBoxSize : undefined) ?? previousBox?.influenceInnerBoxSize ?? defaults.inner,
                    angle:
                        (authored?.shape === "box" ? authored.angle : undefined) ??
                        previousBox?.angle ??
                        (probe.shape !== "sphere" && Number.isFinite(probe.angle) ? -probe.angle! : 0),
                };
            }
        }
        await writeFile(path.join(opts.out, "local-environments.json"), `${JSON.stringify(localRuntime, null, 1)}\n`);
    }
}

main().catch((err: unknown) => {
    console.error(err);
    process.exit(1);
});
