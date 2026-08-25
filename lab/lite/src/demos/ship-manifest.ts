/**
 * Shared reader for `ship_manifest.json`'s `environment` block.
 *
 * The manifest records the view transform the Aquanova ship was authored against. Both the
 * Aquanova demo and the Liquefactor demo's `?foes=ship` mode grade the scene from it, so the two
 * render the same interior at the same brightness and the water look can be compared directly.
 */
import { AcesToneMapping, NeutralToneMapping, StandardToneMapping } from "babylon-lite";
import type { ToneMapping } from "babylon-lite";

/** The manifest's `environment` block (only the fields the demos consume). */
export interface ShipEnvironment {
    /** HDRI path, relative to the manifest. Informational — the demos load a known URL. */
    hdri?: string;
    /** IBL intensity applied as each PBR material's `environmentIntensity`. */
    strength?: number;
    /** Linear exposure multiplier, used exactly as authored (see {@link resolveExposure}). */
    exposure?: number;
    /** Human-readable view-transform name (see {@link resolveToneMapping}). */
    toneMapping?: string;
    /**
     * Authored default for PBR specular anti-aliasing on the ship's materials. A player who has
     * chosen it themselves in the graphics panel keeps their choice — this only moves the default.
     */
    specularAA?: boolean;
    /**
     * Multiplier applied to every ship material's authored `roughnessFactor` on load. Above 1 it
     * broadens metallic reflections, which is the cheapest cure for specular shimmer. The .glb keeps
     * the authored values; this is the one place the multiplier lives, so the editor's Runtime
     * preview and the game apply the identical number.
     */
    reflectionRoughness?: number;
}

/**
 * Behaviour marking the entity that fixes the player's start position. The entity is a placement
 * marker only: the demo spawns the player at its node and then DISABLES it (hidden, non-pickable and
 * excluded from physics/SDF). Its optional `direction` parameter is the glTF-space vector the player
 * initially looks along.
 */
export const PLAYER_START_BEHAVIOR = "player_startpos";

/**
 * Behaviour marking the entity that fixes the weapon pickup's start position. Handled exactly like
 * {@link PLAYER_START_BEHAVIOR}: a placement marker that is disabled at runtime, with an optional
 * glTF-space `direction`.
 */
export const WEAPON_START_BEHAVIOR = "weapon_startpos";

/** Default linear exposure when the manifest omits `environment.exposure`. */
export const DEFAULT_SHIP_EXPOSURE = 1.3;
/** Default IBL intensity when the manifest omits `environment.strength`. */
export const DEFAULT_SHIP_IBL_STRENGTH = 0.45;

/**
 * Resolve the manifest's view-transform name to an engine tone mapping.
 *
 * Matched loosely, so "Khronos PBR Neutral", "neutral" and "KHR_PBR_NEUTRAL" all resolve. An
 * explicit "none" / "off" / "linear" disables tone mapping; anything unrecognised warns and falls
 * back to Neutral, which is what the ship is authored for.
 *
 * @param name - the manifest's `environment.toneMapping`.
 * @returns the tone mapping to apply, or `null` to disable tone mapping entirely.
 */
export function resolveToneMapping(name: string | undefined): ToneMapping | null {
    const key = (name ?? "").toLowerCase().replace(/[^a-z]/g, "");
    if (!key) return NeutralToneMapping;
    if (key.includes("none") || key.includes("off") || key.includes("linear")) return null;
    if (key.includes("aces")) return AcesToneMapping;
    if (key.includes("standard") || key.includes("exponential")) return StandardToneMapping;
    if (key.includes("neutral")) return NeutralToneMapping;
    // eslint-disable-next-line no-console
    console.warn(`[ship-manifest] unknown toneMapping "${name}" — using Khronos PBR Neutral`);
    return NeutralToneMapping;
}

/**
 * Resolve the manifest's exposure to a linear multiplier.
 *
 * The value is the multiplier itself, used exactly as authored — the same
 * number the layout tool's Exposure slider shows, so what you set there is what
 * the demo renders. It used to be stored in Blender STOPS and raised to a power
 * here; that made a plausible-looking `0.3` mean `2^0.3 = 1.23`, more than
 * twice what it appeared to say, and nobody could hold both readings in mind.
 *
 * @param exposure - the manifest's `environment.exposure`.
 * @returns the linear exposure multiplier, or {@link DEFAULT_SHIP_EXPOSURE} when unset.
 */
export function resolveExposure(exposure: number | undefined): number {
    return exposure === undefined ? DEFAULT_SHIP_EXPOSURE : exposure;
}
