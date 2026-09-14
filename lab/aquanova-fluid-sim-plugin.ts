import { copyFileSync, mkdirSync, existsSync, readFileSync, readdirSync, renameSync, unlinkSync, writeFileSync } from "fs";
import type { IncomingMessage, ServerResponse } from "http";
import { resolve } from "path";
import type { Plugin } from "vite";
import { isValidAquanovaFluidSimName, normalizeAquanovaFluidSimName, renameAquanovaFluidSimReferences } from "./aquanova-fluid-sim-authoring.js";

export function aquanovaFluidSimPlugin(): Plugin {
    const configuredProjectDir = process.env.AQUANOVA_SCENE_PROJECT_DIR;
    const defaultProjectDir = resolve(__dirname, "../../Babylon-Scene-Editor/projects/aquanova");
    const projectDir = resolve(configuredProjectDir || defaultProjectDir);
    const projectAvailable = existsSync(resolve(projectDir, "project.json"));
    const publicDir = resolve(__dirname, "public/aquanova");
    const presetDir = projectAvailable ? resolve(projectDir, "fluidSim") : resolve(publicDir, "fluidSim");
    const publicPresetDir = resolve(publicDir, "fluidSim");
    const manifestPaths = [...new Set([projectAvailable ? resolve(projectDir, "scene.json") : "", resolve(publicDir, "scene.json")].filter(Boolean))];
    const readPresetNames = (): string[] =>
        readdirSync(presetDir, { withFileTypes: true })
            .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
            .map((entry) => entry.name.slice(0, -5))
            .filter(isValidAquanovaFluidSimName)
            .sort((a, b) => a.localeCompare(b));
    const sendJson = (res: ServerResponse, status: number, value: unknown): void => {
        res.statusCode = status;
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.setHeader("Cache-Control", "no-store");
        res.end(JSON.stringify(value));
    };
    const readJsonBody = async (req: IncomingMessage): Promise<Record<string, unknown>> => {
        const chunks: Buffer[] = [];
        let size = 0;
        for await (const rawChunk of req) {
            const chunk = Buffer.isBuffer(rawChunk) ? rawChunk : Buffer.from(rawChunk);
            size += chunk.length;
            if (size > 16 * 1024 * 1024) {
                throw new Error("Preset exceeds the 16 MB authoring limit.");
            }
            chunks.push(chunk);
        }
        const parsed = JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            throw new Error("Preset body must be a JSON object.");
        }
        return parsed as Record<string, unknown>;
    };
    const writeJsonAtomic = (path: string, value: unknown): void => {
        const tempPath = `${path}.tmp`;
        writeFileSync(tempPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
        renameSync(tempPath, path);
    };
    const refreshOptionSources = (): void => {
        if (!projectAvailable) return;
        const emitters: Record<string, string[]> = {};
        const sinks: Record<string, string[]> = {};
        for (const name of readPresetNames()) {
            const preset = JSON.parse(readFileSync(resolve(presetDir, `${name}.json`), "utf8")) as {
                emitters?: Array<{ name?: string }>;
                sinks?: Array<{ name?: string }>;
            };
            emitters[name] = [...new Set((preset.emitters ?? []).map((entry) => String(entry.name || "").trim()).filter(Boolean))].sort();
            sinks[name] = [...new Set((preset.sinks ?? []).map((entry) => String(entry.name || "").trim()).filter(Boolean))].sort();
        }
        writeJsonAtomic(resolve(projectDir, "fluid-emitters.json"), emitters);
        writeJsonAtomic(resolve(projectDir, "fluid-sinks.json"), sinks);

        const publishedBehaviors = resolve(publicDir, "behaviors.json");
        if (existsSync(publishedBehaviors)) {
            const metadata = JSON.parse(readFileSync(publishedBehaviors, "utf8")) as { options?: Record<string, unknown> };
            metadata.options = {
                ...(metadata.options ?? {}),
                "Fluid Simulations": readPresetNames(),
                fluidEmitters: emitters,
                fluidSinks: sinks,
            };
            writeJsonAtomic(publishedBehaviors, metadata);
        }
    };
    const mirrorPreset = (name: string): void => {
        if (presetDir === publicPresetDir) return;
        mkdirSync(publicPresetDir, { recursive: true });
        copyFileSync(resolve(presetDir, `${name}.json`), resolve(publicPresetDir, `${name}.json`));
    };
    const renameFluidSimInManifest = (path: string, previousName: string, nextName: string): void => {
        const source = readFileSync(path, "utf8");
        const renamed = renameAquanovaFluidSimReferences(source, previousName, nextName);
        const tempPath = `${path}.tmp`;
        writeFileSync(tempPath, renamed, "utf8");
        renameSync(tempPath, path);
    };

    return {
        name: "aquanova-fluid-sim-authoring",
        configureServer(server) {
            server.middlewares.use(async (req, res, next) => {
                const requestUrl = new URL(req.url ?? "", "http://localhost");
                const pathname = requestUrl.pathname;
                const match = pathname.match(/^\/lab-api\/aquanova-fluid-sims(?:\/([^/]+))?$/);
                if (!match) {
                    next();
                    return;
                }
                const remoteAddress = req.socket.remoteAddress ?? "";
                if (!["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(remoteAddress)) {
                    sendJson(res, 403, { error: "Fluid-simulation authoring is available only from this machine." });
                    return;
                }
                const origin = String(req.headers.origin || "");
                if (origin && !/^https?:\/\/(?:localhost|127\.0\.0\.1)(?::\d+)?$/i.test(origin)) {
                    sendJson(res, 403, { error: "Cross-origin fluid-simulation writes are not allowed." });
                    return;
                }
                try {
                    const encodedName = match[1];
                    if (req.method === "GET" && !encodedName) {
                        sendJson(res, 200, { names: readPresetNames() });
                        return;
                    }
                    if (!encodedName) {
                        sendJson(res, 405, { error: "A simulation name is required." });
                        return;
                    }
                    const name = normalizeAquanovaFluidSimName(decodeURIComponent(encodedName));
                    if (!isValidAquanovaFluidSimName(name)) {
                        sendJson(res, 400, { error: "Simulation names may only contain letters, numbers, hyphens, and underscores." });
                        return;
                    }
                    const presetPath = resolve(presetDir, `${name}.json`);
                    if (req.method === "PUT") {
                        const preset = await readJsonBody(req);
                        const names = readPresetNames();
                        const renameToValue = requestUrl.searchParams.get("renameTo");
                        const targetName = normalizeAquanovaFluidSimName(renameToValue ?? name);
                        if (!isValidAquanovaFluidSimName(targetName)) {
                            sendJson(res, 400, { error: "Simulation names may only contain letters, numbers, hyphens, and underscores." });
                            return;
                        }
                        const renaming = targetName !== name;
                        const targetPresetPath = resolve(presetDir, `${targetName}.json`);
                        if (renaming && (!names.includes(name) || !existsSync(presetPath))) {
                            sendJson(res, 404, { error: `Simulation "${name}" does not exist.` });
                            return;
                        }
                        if (renaming && (names.includes(targetName) || existsSync(targetPresetPath))) {
                            sendJson(res, 409, { error: `Simulation "${targetName}" already exists.` });
                            return;
                        }
                        const nextNames = renaming ? names.map((entry) => (entry === name ? targetName : entry)) : [...names];
                        if (!names.includes(name)) {
                            nextNames.push(targetName);
                        }
                        mkdirSync(presetDir, { recursive: true });
                        const previousPreset = existsSync(presetPath) ? readFileSync(presetPath) : null;
                        const previousTargetPreset = renaming && existsSync(targetPresetPath) ? readFileSync(targetPresetPath) : null;
                        const publicTargetPresetPath = resolve(publicPresetDir, `${targetName}.json`);
                        const previousPublicTargetPreset = presetDir !== publicPresetDir && existsSync(publicTargetPresetPath) ? readFileSync(publicTargetPresetPath) : null;
                        const previousManifests = renaming ? manifestPaths.map((path) => [path, readFileSync(path)] as const) : [];
                        try {
                            writeJsonAtomic(targetPresetPath, preset);
                            mirrorPreset(targetName);
                            if (renaming) {
                                for (const path of manifestPaths) renameFluidSimInManifest(path, name, targetName);
                                unlinkSync(presetPath);
                                const oldPublicPreset = resolve(publicPresetDir, `${name}.json`);
                                if (presetDir !== publicPresetDir && existsSync(oldPublicPreset)) unlinkSync(oldPublicPreset);
                            }
                            refreshOptionSources();
                        } catch (error) {
                            if (previousPreset) {
                                writeFileSync(presetPath, previousPreset);
                            } else if (existsSync(presetPath)) {
                                unlinkSync(presetPath);
                            }
                            if (renaming) {
                                if (previousTargetPreset) {
                                    writeFileSync(targetPresetPath, previousTargetPreset);
                                } else if (existsSync(targetPresetPath)) {
                                    unlinkSync(targetPresetPath);
                                }
                            }
                            if (presetDir !== publicPresetDir) {
                                if (previousPublicTargetPreset) {
                                    writeFileSync(publicTargetPresetPath, previousPublicTargetPreset);
                                } else if (existsSync(publicTargetPresetPath)) {
                                    unlinkSync(publicTargetPresetPath);
                                }
                                if (renaming && previousPreset) writeFileSync(resolve(publicPresetDir, `${name}.json`), previousPreset);
                            }
                            for (const [path, contents] of previousManifests) writeFileSync(path, contents);
                            throw error;
                        }
                        sendJson(res, 200, { name: targetName, names: nextNames });
                        return;
                    }
                    if (req.method === "DELETE") {
                        if (!existsSync(presetPath)) {
                            sendJson(res, 404, { error: `Simulation "${name}" does not exist.` });
                            return;
                        }
                        const names = readPresetNames().filter((entry) => entry !== name);
                        const previousPreset = readFileSync(presetPath);
                        const publicPreset = resolve(publicPresetDir, `${name}.json`);
                        const previousPublicPreset = presetDir !== publicPresetDir && existsSync(publicPreset) ? readFileSync(publicPreset) : null;
                        try {
                            unlinkSync(presetPath);
                            if (presetDir !== publicPresetDir && existsSync(publicPreset)) unlinkSync(publicPreset);
                            refreshOptionSources();
                        } catch (error) {
                            writeFileSync(presetPath, previousPreset);
                            if (previousPublicPreset) writeFileSync(publicPreset, previousPublicPreset);
                            throw error;
                        }
                        sendJson(res, 200, { name, names });
                        return;
                    }
                    sendJson(res, 405, { error: `Method ${req.method ?? "unknown"} is not supported.` });
                } catch (error) {
                    sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
                }
            });
        },
    };
}
