export interface AquanovaFluidSimManifest {
    fluidSim?: string[];
}

export function normalizeAquanovaFluidSimName(value: string): string {
    return value.replace(/\.json$/i, "").toLowerCase();
}

export function isValidAquanovaFluidSimName(value: string): boolean {
    return /^[a-z0-9][a-z0-9_-]{0,63}$/i.test(value);
}

export function aquanovaFluidSimNames(manifest: AquanovaFluidSimManifest): string[] {
    return Array.from(new Set((manifest.fluidSim ?? []).map(normalizeAquanovaFluidSimName).filter(isValidAquanovaFluidSimName)));
}

export function renameAquanovaFluidSimReferences(source: string, previousName: string, nextName: string): string {
    const previous = normalizeAquanovaFluidSimName(previousName);
    const next = normalizeAquanovaFluidSimName(nextName);
    if (!isValidAquanovaFluidSimName(previous) || !isValidAquanovaFluidSimName(next)) {
        throw new Error("Cannot rename an invalid fluid simulation name.");
    }
    if (previous === next) {
        return source;
    }

    let replacements = 0;
    const renameToken = (token: string): string => {
        const value = JSON.parse(token) as unknown;
        if (typeof value === "string" && normalizeAquanovaFluidSimName(value) === previous) {
            replacements++;
            return JSON.stringify(next);
        }
        return token;
    };
    const renamed = source.replace(/("fluidSim"\s*:\s*)("(?:\\.|[^"\\])*"|\[[\s\S]*?\])/g, (_match, prefix: string, value: string) => {
        if (value.startsWith("[")) {
            return prefix + value.replace(/"(?:\\.|[^"\\])*"/g, renameToken);
        }
        return prefix + renameToken(value);
    });
    if (replacements === 0) {
        throw new Error(`No fluidSim reference to "${previous}" was found in the manifest.`);
    }
    JSON.parse(renamed);
    return renamed;
}
