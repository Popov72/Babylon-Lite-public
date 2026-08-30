const SOUND_ROOT = "/aquanova/sounds";
const SOUND_ASSET_VERSION = "20260813-1";

export function aquanovaSoundUrl(soundName: string): string {
    return `${SOUND_ROOT}/${encodeURIComponent(soundName)}.mp3?v=${SOUND_ASSET_VERSION}`;
}

export function validateAquanovaSoundName(path: string, soundName: string): void {
    if (typeof soundName !== "string" || !soundName || soundName.endsWith(".mp3") || soundName.includes("/") || soundName.includes("\\")) {
        throw new Error(`[aquanova] ${path} "${soundName}" must be an MP3 file name without its extension`);
    }
}
