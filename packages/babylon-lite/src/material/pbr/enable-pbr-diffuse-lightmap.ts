import type { PbrMaterialProps } from "./pbr-material.js";
import type { Texture2D } from "../../texture/texture-2d.js";

export interface PbrDiffuseLightmapOptions {
    coordIndex?: 0 | 1;
    level?: number;
    gamma?: boolean;
}

type DiffuseLightmapMaterial = PbrMaterialProps & {
    _diffuseLightmapTexture?: Texture2D;
    _diffuseLightmapCoordIndex?: 0 | 1;
    _diffuseLightmapLevel?: number;
    _gammaDiffuseLightmap?: boolean;
};

const UV2_MASK_DIFFUSE_LIGHTMAP = 1 << 6;
let enabled = false;

/** Enable color-free baked irradiance lightmaps. Call before registerScene(). */
export async function enablePbrDiffuseLightmap(): Promise<void> {
    if (enabled) {
        return;
    }
    const [{ _registerPbrExt }, { pbrExt }] = await Promise.all([import("./pbr-flags.js"), import("./fragments/diffuse-lightmap-fragment.js")]);
    _registerPbrExt(pbrExt);
    enabled = true;
}

/** Assign a baked irradiance texture that replaces diffuse IBL without affecting specular IBL. */
export function setPbrDiffuseLightmap(material: PbrMaterialProps, texture: Texture2D, options?: PbrDiffuseLightmapOptions): void {
    const target = material as DiffuseLightmapMaterial;
    const coordIndex = options?.coordIndex ?? 1;
    target._diffuseLightmapTexture = texture;
    target._diffuseLightmapCoordIndex = coordIndex;
    target._diffuseLightmapLevel = options?.level;
    target._gammaDiffuseLightmap = options?.gamma;
    if (coordIndex === 1) {
        const uvMaterial = material as { _uv2Mask?: number };
        uvMaterial._uv2Mask = (uvMaterial._uv2Mask ?? 0) | UV2_MASK_DIFFUSE_LIGHTMAP;
    }
}
