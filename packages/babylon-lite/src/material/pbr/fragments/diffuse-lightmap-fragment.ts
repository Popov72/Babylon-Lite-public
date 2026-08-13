import type { ShaderFragment } from "../../../shader/fragment-types.js";
import type { Texture2D } from "../../../texture/texture-2d.js";
import type { PbrMaterialProps } from "../pbr-material.js";
import type { PbrExt } from "../pbr-flags.js";
import { MSH_HAS_UV2 } from "../../mesh-features.js";
import { PBR2_HAS_UV2 } from "../pbr-flag-bits.js";

type DiffuseLightmapMaterial = PbrMaterialProps & {
    _diffuseLightmapTexture?: Texture2D;
    _diffuseLightmapCoordIndex?: 0 | 1;
    _diffuseLightmapLevel?: number;
    _gammaDiffuseLightmap?: boolean;
};

const STAGE_FRAGMENT = 0x2;
const PBR_HAS_DIFFUSE_LIGHTMAP = 1 << 24;
const PBR_DIFFUSE_LIGHTMAP_UV2 = 1 << 14;
const PBR_DIFFUSE_LIGHTMAP_GAMMA = 1 << 18;
const PBR_DIFFUSE_LIGHTMAP_FLIP_V = 1 << 19;

function createDiffuseLightmapFragment(usesUV2: boolean, gamma: boolean, flipV: boolean, hasIbl: boolean): ShaderFragment {
    const baseUv = usesUV2 ? "input.uv2" : "input.uv";
    const uv = flipV ? `vec2<f32>(${baseUv}.x,1.0-${baseUv}.y)` : baseUv;
    const raw = `textureSample(lmTexture,lmSampler,${uv}).rgb`;
    const lm = `${gamma ? `pow(${raw},vec3<f32>(2.2))` : raw}*material.lmLvl`;
    return {
        _id: "a-diffuse-lightmap",
        _dependencies: hasIbl ? ["ibl"] : undefined,
        _uboFields: [{ _name: "lmLvl", _type: "f32" }],
        _bindings: [
            { _name: "lmTexture", _type: { _kind: "texture", _textureType: "texture_2d<f32>" }, _visibility: STAGE_FRAGMENT },
            { _name: "lmSampler", _type: { _kind: "sampler", _samplerType: "sampler" }, _visibility: STAGE_FRAGMENT },
        ],
        _fragmentSlots: {
            MF: `let bakedIrradiance=${lm};`,
            ...(hasIbl
                ? { AI: "color-=finalIrradiance;finalIrradiance=bakedIrradiance*surfaceAlbedo*occlusion;color+=finalIrradiance;" }
                : { NI: "color=bakedIrradiance*surfaceAlbedo*occlusion+directDiffuse+directSpecular+emissive;" }),
        },
    };
}

export const pbrExt: PbrExt = {
    id: "a-diffuse-lightmap",
    phase: "fragment",
    detect(material) {
        const m = material as DiffuseLightmapMaterial;
        if (!m._diffuseLightmapTexture) {
            return { f: 0, f2: 0 };
        }
        let f = PBR_HAS_DIFFUSE_LIGHTMAP;
        let f2 = 0;
        if ((m._diffuseLightmapCoordIndex ?? 1) === 1) {
            f |= PBR_DIFFUSE_LIGHTMAP_UV2;
            f2 |= PBR2_HAS_UV2;
        }
        if (m._gammaDiffuseLightmap) {
            f |= PBR_DIFFUSE_LIGHTMAP_GAMMA;
        }
        if (m._diffuseLightmapTexture.uAng === Math.PI) {
            f |= PBR_DIFFUSE_LIGHTMAP_FLIP_V;
        }
        return { f, f2 };
    },
    frag(ctx) {
        return (ctx._features & PBR_HAS_DIFFUSE_LIGHTMAP) !== 0
            ? createDiffuseLightmapFragment(
                  (ctx._features & PBR_DIFFUSE_LIGHTMAP_UV2) !== 0 && (ctx._meshFeatures & MSH_HAS_UV2) !== 0,
                  (ctx._features & PBR_DIFFUSE_LIGHTMAP_GAMMA) !== 0,
                  (ctx._features & PBR_DIFFUSE_LIGHTMAP_FLIP_V) !== 0,
                  ctx._hasIbl
              )
            : null;
    },
    writeUbo(data, material, offsets) {
        const offset = offsets.get("lmLvl");
        if (offset !== undefined) {
            data[offset / 4] = (material as DiffuseLightmapMaterial)._diffuseLightmapLevel ?? 1;
        }
    },
    bind(ctx, entries, binding) {
        const texture = (ctx._material as DiffuseLightmapMaterial)._diffuseLightmapTexture;
        if (!texture) {
            return binding;
        }
        entries.push({ binding: binding++, resource: texture.view });
        entries.push({ binding: binding++, resource: texture.sampler });
        return binding;
    },
    textures(material, out) {
        const texture = (material as DiffuseLightmapMaterial)._diffuseLightmapTexture;
        if (texture) {
            out.push(texture);
        }
    },
};
