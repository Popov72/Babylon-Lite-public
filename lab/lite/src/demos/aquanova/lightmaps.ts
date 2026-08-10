// Baked lightmaps for the Aquanova ship.
//
// The ship is lit by a Blender/Cycles bake, not by runtime lights. `bake_lightmaps.py` renders one
// irradiance atlas per chunk and `lab/public/aquanova/scripts/sync-baked-ship.ts` compresses those
// atlases to KTX2 and writes the index this module reads.
//
// ── What the atlas contains, and how it composes with the environment ─────────────────────────
// The bake is a Cycles DIFFUSE pass with `use_pass_color = False`: direct + indirect irradiance and
// NOTHING else — no albedo, no texture, no decals. So the surface colour is
//
//     PBR lighting × lightmap + emissive
//
// which is `useLightmapAsShadowmap` (multiply, not add). The material remains on the normal PBR
// path so the environment can contribute metallic reflections; static geometry is still excluded
// from authored runtime lights below.
//
//   • without `useLightmapAsShadowmap` the atlas is ADDED instead, so every wall glows with raw
//     irradiance and the kit's albedo, panel lines and decals disappear under it.
//
// Emissive stays on top of the product, matching Babylon.js — `pbrBlockFinalColorComposition.fx`
// adds `finalEmissive` outside its `UNLIT` guard. That is what keeps the light strips and the
// signage glowing after the multiply, and it is why the bake replaces the emissive materials with
// Cycles lamps rather than letting them contribute twice.
//
// ── level ─────────────────────────────────────────────────────────────────────────────────────
// Bounced light in a lit corridor runs well past 1.0, so the bake divides each atlas by its own
// 99.9th percentile before writing 8 bits and records the divisor as `level`; `lightmapLevel`
// multiplies it back. Nothing here should clamp before that happens.
//
// ── Which meshes get one ──────────────────────────────────────────────────────────────────────
// Only meshes the bake actually unwrapped: `chunk_meshes` in `bake_lightmaps.py` drops anything the
// runtime redraws — the meshes it moves (`dynamic`) and the ones it melts (`liquefiable`, plus
// everything they are `linked` to) — because a prop that will not stay put cannot have its shadow
// painted into the room. Those keep their glTF material and are lit by the runtime lamps instead
// (see `lights.ts`). `isBakeExcluded` below is the single test both halves share; its doc explains
// why it reads TEXCOORD_1 rather than the behaviour list.
//
// ── Material cloning ──────────────────────────────────────────────────────────────────────────
// Blender shares one material across every chunk that uses the kit module, but each chunk has its
// OWN atlas. One clone per (chunk, material) pair keeps them apart; without it the last chunk
// processed would win and every earlier chunk would sample the wrong atlas.

import { enablePbrLightmap, loadKtx2Texture2D, setPbrLightmap, type EngineContext, type Mesh, type PbrMaterialProps } from "babylon-lite";
import { LIGHTMAPS_URL } from "./constants.js";

/** One chunk's baked atlas, as written by `sync-baked-ship.ts`. */
export interface LightmapChunk {
    /** Relative to the index file's own directory. */
    url: string;
    /** The scale the bake divided out to fit 8 bits — `lightmapLevel` puts it back. */
    level: number;
    resolution: number;
    meshes: number;
    bytes: number;
}

/** `lab/public/aquanova/lightmaps.json`. */
export interface LightmapIndex {
    glb: string;
    /** TEXCOORD set holding the atlas UVs (1). */
    uv: number;
    /** Whether the SHADER must sRGB-decode the sample. False when `srgb` moved that to the sampler. */
    gamma: boolean;
    /** Whether to upload as an `*-srgb` GPU format, so the hardware does the decode. */
    srgb: boolean;
    encoding: string;
    chunks: Record<string, LightmapChunk>;
}

export interface LightmapStats {
    /** Meshes that received an atlas. */
    lit: number;
    /** Meshes skipped for want of TEXCOORD_1 — the props the bake left out. */
    skipped: number;
    /** Chunks named in the index that the loaded ship has no meshes for. */
    missing: string[];
    bytes: number;
}

/**
 * Whether the bake left this mesh out of its chunk's atlas, and it therefore has to be lit at
 * runtime instead. This is the ONE rule that decides where a mesh gets its light from, and both
 * halves read it from here: {@link applyBakedLightmaps} skips these, `buildRuntimeLights` scopes the
 * authored lamps to exactly them.
 *
 * The test is the absence of TEXCOORD_1 rather than a behaviour list, because that is what the
 * shader actually needs — sampling a vertex buffer that is not there reads zero, and the atlas
 * multiply would render the mesh black. It also means this follows whatever `chunk_meshes` in
 * `bake_lightmaps.py` decided, for free: today that is `dynamic` plus `liquefiable` and everything
 * they are `linked` to, and if that list changes the runtime does not have to be taught about it.
 *
 * Keying off the behaviours instead is what left the liquefiable-only doors dark: they are excluded
 * from the bake but are not `dynamic`, so a `movableMeshes`-scoped lamp set skipped them and they
 * ended up with no lightmap AND no lamp.
 * @param mesh - A ship mesh.
 */
export function isBakeExcluded(mesh: Mesh): boolean {
    return !mesh._gpu.uv2Buffer;
}

export async function fetchLightmapIndex(): Promise<LightmapIndex | null> {
    try {
        const resp = await fetch(LIGHTMAPS_URL);
        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }
        return (await resp.json()) as LightmapIndex;
    } catch (err) {
        console.warn("[aquanova] no baked lightmaps —", err);
        return null;
    }
}

/**
 * Dress every static ship mesh with its chunk's baked atlas. Call **before `registerScene`**:
 * `enablePbrLightmap` has to register the PBR extension before the first pipeline is composed, and
 * the material swaps below have to be visible when the renderables are built.
 * @param engine - Engine the ship was loaded on.
 * @param index - Parsed `lightmaps.json`.
 * @param meshes - Every mesh of the ship.
 * @param chunkOfMesh - Which chunk each mesh belongs to, from the `CHUNK_*` ancestor walk.
 */
export async function applyBakedLightmaps(engine: EngineContext, index: LightmapIndex, meshes: readonly Mesh[], chunkOfMesh: ReadonlyMap<Mesh, string>): Promise<LightmapStats> {
    await enablePbrLightmap();

    const base = new URL(LIGHTMAPS_URL, location.href);
    const byChunk = new Map<string, Mesh[]>();
    for (const mesh of meshes) {
        const chunk = chunkOfMesh.get(mesh);
        if (!chunk || !index.chunks[chunk]) {
            continue;
        }
        const group = byChunk.get(chunk);
        if (group) group.push(mesh);
        else byChunk.set(chunk, [mesh]);
    }

    const stats: LightmapStats = { lit: 0, skipped: 0, missing: [], bytes: 0 };
    for (const chunkId of Object.keys(index.chunks)) {
        if (!byChunk.has(chunkId)) stats.missing.push(chunkId);
    }

    // Atlases load in parallel: each is a fetch plus a Basis transcode, and the transcode of the
    // 1024² corridor atlas alone is longer than every material swap in the ship put together.
    const entries = [...byChunk];
    const atlases = await Promise.all(entries.map(([chunkId]) => loadKtx2Texture2D(engine, new URL(index.chunks[chunkId]!.url, base).href, index.srgb)));

    for (let i = 0; i < entries.length; i++) {
        const [chunkId, group] = entries[i]!;
        const chunk = index.chunks[chunkId]!;
        const atlas = atlases[i]!;
        stats.bytes += chunk.bytes;
        // Keyed by the SOURCE material, scoped to this chunk — see the header.
        const clones = new Map<PbrMaterialProps, PbrMaterialProps>();
        for (const mesh of group) {
            if (isBakeExcluded(mesh)) {
                stats.skipped++;
                continue;
            }
            const source = mesh.material as PbrMaterialProps | undefined;
            if (!source) {
                continue;
            }
            let clone = clones.get(source);
            if (!clone) {
                clone = { ...source, unlit: false };
                // Drop the two caches the spread would otherwise inherit from a material that has
                // already been through a build: `_renderFeatures` is a memo of the flag bits and
                // would hide `lightmap` from the composer, and `_clusteredLightState` opts the
                // material into the clustered-light block. Static geometry must keep the normal
                // PBR path for environment reflections, but it must not receive authored runtime
                // lights, so the clustered-light stamp is still removed.
                // `detect` and `bind` both read the field off the same material, so the bind-group
                // layout stays consistent.
                delete (clone as { _renderFeatures?: unknown })._renderFeatures;
                delete (clone as { _clusteredLightState?: unknown })._clusteredLightState;
                setPbrLightmap(clone, atlas, { coordIndex: index.uv === 1 ? 1 : 0, level: chunk.level, useAsShadowmap: true, gamma: index.gamma });
                clones.set(source, clone);
            }
            mesh.material = clone;
            stats.lit++;
        }
    }
    return stats;
}
