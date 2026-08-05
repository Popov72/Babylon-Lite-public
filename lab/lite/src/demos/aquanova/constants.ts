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
/** Skybox cube faces, as `${SKYBOX_URL}_px${SKYBOX_EXT}` … `_nz`. Purely a backdrop seen through the
 *  ship's openings — the IBL that actually lights the ship still comes from ENV_URL. */
export const SKYBOX_URL = "/aquanova/skybox/sky";
export const SKYBOX_EXT = ".png";
/** Edge length of the skybox cube. Constrained at both ends: the box is static at the origin (it
 *  does not follow the camera), so it must comfortably enclose anywhere the player reaches, yet its
 *  CORNERS — at (size/2)·√3 — have to stay inside the camera's 400 m far plane or the corners of the
 *  view clip to the clear colour. 300 puts the faces 150 m out and the corners at ~260 m. */
export const SKYBOX_SIZE = 300;

export type Vec3 = readonly [number, number, number];

/** glTF (right-handed) → Lite scene (left-handed): the loader's `__root__` negates X. */
export function toLite([x, y, z]: Vec3): [number, number, number] {
    return [-x, y, z];
}

/** Room floor height. The fluid domain and the ground-plane backstop key off this. */
export const FLOOR_Y = 0;
/** Room ceiling height. Used as the fallback top of a fluid domain. */
export const CEIL_Y = 5;
