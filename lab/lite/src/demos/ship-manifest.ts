/**
 * Shared reader for `ship_manifest.json`'s `environment` block.
 *
 * The manifest records the Blender view transform the Aquanova ship was authored against. Both the
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
    /** IBL intensity for meshes without baked lightmap UVs, such as dynamic props. */
    dynamicStrength?: number;
    /** Linear exposure multiplier, used exactly as authored (see {@link resolveExposure}). */
    exposure?: number;
    /** Human-readable view-transform name (see {@link resolveToneMapping}). */
    toneMapping?: string;
}

/**
 * A behaviour definition from the manifest's global `behaviors` block, or the effective behaviour of
 * a mesh once its entity references have been flattened by {@link resolveBehavior}.
 */
export interface ShipBehavior {
    /** The mesh can be shot and turned into fluid. Implies {@link ShipBehavior.dynamic}. */
    liquefiable?: boolean;
    /** The mesh is driven by a Havok rigid body. Implied by `liquefiable`, so it need not be repeated. */
    dynamic?: boolean;
    /** Candidate fluidSim setting names; absent/empty inherits the manifest's global `fluidSim` list. */
    fluidSim?: string[];
    /**
     * Mesh names that must liquefy at the same moment as this one (liquefiable behaviours only).
     * Applied SHIP-WIDE: every mesh carrying a listed name melts, not just nearby instances.
     */
    linked?: string[];
    /** Facing direction, in glTF space, for the marker behaviours (see {@link PLAYER_START_BEHAVIOR}). */
    direction?: number[];
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

/**
 * An entity's reference to a named behaviour. Every key other than `name` is a parameter merged on
 * top of that behaviour, so one shared definition can be reused with per-entity tweaks.
 */
export interface ShipBehaviorRef extends ShipBehavior {
    /** Key into the manifest's global `behaviors` block. */
    name: string;
}

/** The manifest's global `behaviors` block: behaviour name → definition. */
export type ShipBehaviorLibrary = Record<string, ShipBehavior>;

/** The manifest's `entities` block: MESH NAME → the behaviours assigned to it. */
export type ShipEntities = Record<string, { behaviors?: ShipBehaviorRef[] }>;

/**
 * Flatten the behaviours assigned to one mesh into a single effective behaviour.
 *
 * Each reference names a definition in the global `behaviors` block; the reference's own keys
 * (everything but `name`) are merged ON TOP, so an entity can parameterise a shared behaviour
 * without redefining it. A mesh may list several behaviours — they merge in order, later winning.
 *
 * @param library - the manifest's global `behaviors` block.
 * @param entities - the manifest's `entities` block.
 * @param meshName - the mesh name to resolve.
 * @returns the effective behaviour, or undefined when the mesh has none.
 */
export function resolveBehavior(library: ShipBehaviorLibrary | undefined, entities: ShipEntities | undefined, meshName: string): ShipBehavior | undefined {
    const refs = entities?.[meshName]?.behaviors;
    if (!refs?.length) return undefined;
    let out: ShipBehavior | undefined;
    for (const ref of refs) {
        const base = library?.[ref.name];
        if (!base) {
            // eslint-disable-next-line no-console
            console.warn(`[ship-manifest] entity "${meshName}" references unknown behavior "${ref.name}"`);
            continue;
        }
        const params: Partial<ShipBehaviorRef> = { ...ref };
        delete params.name;
        out = { ...(out ?? {}), ...base, ...params };
    }
    return out;
}

/**
 * Mesh names that must liquefy together with the owning mesh.
 *
 * @param b - the mesh's effective behaviour.
 * @returns the linked mesh names, with blank placeholder entries dropped.
 */
export function linkedMeshNames(b: ShipBehavior | undefined): string[] {
    return (b?.linked ?? []).filter((n) => typeof n === "string" && n.length > 0);
}

/**
 * Find the entity carrying a named behaviour — used for the singleton placement markers, whose
 * library definitions are empty, so the behaviour is identified by its NAME rather than by any flag
 * on the merged result.
 *
 * @param entities - the manifest's `entities` block.
 * @param behaviorName - the behaviour to look for.
 * @returns the entity name and the reference (carrying any parameters such as `direction`), or
 *          undefined when no entity declares it.
 */
export function findEntityWithBehavior(entities: ShipEntities | undefined, behaviorName: string): { name: string; ref: ShipBehaviorRef } | undefined {
    for (const [name, entity] of Object.entries(entities ?? {})) {
        const ref = entity.behaviors?.find((r) => r.name === behaviorName);
        if (ref) return { name, ref };
    }
    return undefined;
}

/**
 * Whether a mesh needs a rigid body.
 *
 * A liquefiable mesh is ALWAYS dynamic — it has to be shootable and physically present before it
 * melts — so the manifest doesn't repeat `dynamic: true` for it and this rule is applied here once
 * for every consumer.
 *
 * @param b - the mesh's behaviour entry, if any.
 * @returns true when the mesh should get a Havok body.
 */
export function isDynamicBehavior(b: ShipBehavior | undefined): boolean {
    return b !== undefined && (b.dynamic === true || b.liquefiable === true);
}

/**
 * Whether a mesh can be liquefied.
 *
 * @param b - the mesh's behaviour entry, if any.
 * @returns true when the mesh is liquefiable.
 */
export function isLiquefiableBehavior(b: ShipBehavior | undefined): boolean {
    return b?.liquefiable === true;
}

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
