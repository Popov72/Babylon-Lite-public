import { mkdirSync, existsSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "fs";
import type { IncomingMessage, ServerResponse } from "http";
import { resolve } from "path";
import type { Plugin } from "vite";
import {
    aquanovaFluidSimNames,
    isValidAquanovaFluidSimName,
    normalizeAquanovaFluidSimName,
    renameAquanovaFluidSimReferences,
    type AquanovaFluidSimManifest,
} from "./aquanova-fluid-sim-authoring.js";

export function aquanovaFluidSimPlugin(): Plugin {
    const sourceManifestPath = resolve(__dirname, "lite/src/demos/aquanova/editor/export/ship_manifest.json");
    const publicManifestPath = resolve(__dirname, "public/aquanova/ship_manifest.json");
    const presetDir = resolve(__dirname, "public/aquanova/fluidSim");
    const readManifest = (): Record<string, unknown> & AquanovaFluidSimManifest =>
        JSON.parse(readFileSync(sourceManifestPath, "utf8")) as Record<string, unknown> & AquanovaFluidSimManifest;
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
    const writeManifestFluidSims = (path: string, names: readonly string[]): void => {
        const source = readFileSync(path, "utf8");
        const current = aquanovaFluidSimNames(JSON.parse(source) as AquanovaFluidSimManifest);
        if (current.length === names.length && current.every((name, index) => name === names[index])) {
            return;
        }
        const pattern = /^(\s*)"fluidSim"\s*:\s*\[[^\]]*\]/m;
        const match = pattern.exec(source);
        if (!match) {
            throw new Error("ship_manifest.json does not contain a fluidSim array.");
        }
        const indent = match[1] ?? "";
        const itemIndent = /\n([ \t]+)"/.exec(match[0])?.[1] ?? `${indent}  `;
        const replacement =
            match[0].includes("\n") && names.length
                ? `${indent}"fluidSim": [\n${names.map((name) => `${itemIndent}${JSON.stringify(name)}`).join(",\n")}\n${indent}]`
                : `${indent}"fluidSim": [${names.map((name) => JSON.stringify(name)).join(", ")}]`;
        const next = source.replace(pattern, replacement);
        const tempPath = `${path}.tmp`;
        writeFileSync(tempPath, next, "utf8");
        renameSync(tempPath, path);
    };
    const writeFluidSimManifests = (names: readonly string[]): void => {
        writeManifestFluidSims(sourceManifestPath, names);
        writeManifestFluidSims(publicManifestPath, names);
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
                try {
                    const encodedName = match[1];
                    if (req.method === "GET" && !encodedName) {
                        sendJson(res, 200, { names: aquanovaFluidSimNames(readManifest()) });
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
                        const manifest = readManifest();
                        const names = aquanovaFluidSimNames(manifest);
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
                        const previousSourceManifest = readFileSync(sourceManifestPath);
                        const previousPublicManifest = readFileSync(publicManifestPath);
                        try {
                            writeJsonAtomic(targetPresetPath, preset);
                            if (renaming) {
                                renameFluidSimInManifest(sourceManifestPath, name, targetName);
                                renameFluidSimInManifest(publicManifestPath, name, targetName);
                                unlinkSync(presetPath);
                            } else {
                                writeFluidSimManifests(nextNames);
                            }
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
                            writeFileSync(sourceManifestPath, previousSourceManifest);
                            writeFileSync(publicManifestPath, previousPublicManifest);
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
                        const manifest = readManifest();
                        const names = aquanovaFluidSimNames(manifest).filter((entry) => entry !== name);
                        const previousSourceManifest = readFileSync(sourceManifestPath);
                        const previousPublicManifest = readFileSync(publicManifestPath);
                        try {
                            writeFluidSimManifests(names);
                            unlinkSync(presetPath);
                        } catch (error) {
                            writeFileSync(sourceManifestPath, previousSourceManifest);
                            writeFileSync(publicManifestPath, previousPublicManifest);
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
