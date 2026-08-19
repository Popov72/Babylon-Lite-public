export function normalizeSoundVolume(volume: number): number {
    if (!Number.isFinite(volume)) {
        throw new Error(`[aquanova] sound volume must be finite, received ${String(volume)}`);
    }
    return Math.max(0, Math.min(1, volume));
}
