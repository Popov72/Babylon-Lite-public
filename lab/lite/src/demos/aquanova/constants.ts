// Shared constants and the glTF→Lite coordinate conversion for the Aquanova demo.
//
// The ship runs back → front along glTF +X. Lite's glTF loader mirrors handedness on the `__root__`
// (scale x = -1), so glTF (x, y, z) renders at Lite (-x, y, z) — every manifest coordinate goes
// through `toLite` before use.

export const SHIP_URL = "/aquanova/ship.glb";
export const MANIFEST_URL = "/aquanova/ship_manifest.json";
/** Image-based lighting: the ship is lit by this HDRI (Poly Haven "bank_vault" 2k, CC0) plus its own
 *  emissive ceiling fixtures — the glTF export carries no punctual lights. */
export const ENV_URL = "/aquanova/bank_vault_2k.hdr";

export type Vec3 = readonly [number, number, number];

/** glTF (right-handed) → Lite scene (left-handed): the loader's `__root__` negates X. */
export function toLite([x, y, z]: Vec3): [number, number, number] {
    return [-x, y, z];
}

/** Room floor height. The fluid domain, the collider shell and the SDF overlay all key off this. */
export const FLOOR_Y = 0;
/** Room ceiling height. */
export const CEIL_Y = 5;
/** Thickness of a synthesised collider slab (floor / ceiling / wall). */
export const WALL_T = 0.4;
/** Pull perimeter walls in to the inner wall face so the capsule can't clip them. */
export const WALL_INSET = 0.3;
