/**
 * Box-projected local image-based lighting.
 *
 * Matches Babylon.js' parallaxCorrectNormal helper: intersect the reflected
 * world-space ray with the probe box, then sample from the probe center toward
 * that intersection. The same probe supplies the material's irradiance SH.
 */

import type { EnvironmentTextures } from "../../../loader-env/load-env.js";
import type { ComposedShader, ShaderFragment } from "../../../shader/fragment-types.js";
import type { PbrMaterialProps } from "../pbr-material.js";
import type { PbrExt } from "../pbr-flags.js";
import { IBL_SCENE_IRRADIANCE } from "./ibl-fragment.js";

const PBR2_HAS_LOCAL_CUBEMAP = 1 << 29;

const LOCAL_CUBEMAP_HELPER = `fn parallaxCorrectNormal(vertexPos:vec3f,origVec:vec3f,cubeSize:vec3f,cubePos:vec3f)->vec3f{
let invOrigVec=vec3f(1.0)/origVec;
let halfSize=cubeSize*0.5;
let intersecAtMaxPlane=(cubePos+halfSize-vertexPos)*invOrigVec;
let intersecAtMinPlane=(cubePos-halfSize-vertexPos)*invOrigVec;
let largestIntersec=max(intersecAtMaxPlane,intersecAtMinPlane);
let distance=min(min(largestIntersec.x,largestIntersec.y),largestIntersec.z);
return vertexPos+origVec*distance-cubePos;
}`;

const LOCAL_SH_FIELDS = [
    "localSphericalL00",
    "localSphericalL1_1",
    "localSphericalL10",
    "localSphericalL11",
    "localSphericalL2_2",
    "localSphericalL2_1",
    "localSphericalL20",
    "localSphericalL21",
    "localSphericalL22",
] as const;

const LOCAL_IRRADIANCE = `let environmentIrradiance = (material.localSphericalL00.rgb
  + material.localSphericalL1_1.rgb * N_env.y + material.localSphericalL10.rgb * N_env.z + material.localSphericalL11.rgb * N_env.x
  + material.localSphericalL2_2.rgb * (N_env.y * N_env.x) + material.localSphericalL2_1.rgb * (N_env.y * N_env.z)
  + material.localSphericalL20.rgb * (3.0 * N_env.z * N_env.z - 1.0) + material.localSphericalL21.rgb * (N_env.z * N_env.x)
  + material.localSphericalL22.rgb * (N_env.x * N_env.x - N_env.y * N_env.y)) * material.environmentIntensity;`;

function getLocalEnvironment(material: unknown): EnvironmentTextures | null | undefined {
    return (material as PbrMaterialProps).localEnvironment;
}

function patchLocalCubemapReflection(composed: ComposedShader): ComposedShader {
    let rewrites = 0;
    let fragmentWGSL = composed._fragmentWGSL.replace(/let\s+R\s*=\s*rotateY\(\s*R_raw\s*,\s*scene\.envRotationY\s*\);/g, () => {
        rewrites++;
        return "let R=rotateY(parallaxCorrectNormal(input.worldPos,R_raw,material.vReflectionSize,material.vReflectionPosition),scene.envRotationY);";
    });
    if (rewrites !== 1) {
        throw new Error(`local cubemap _postCompose: expected to rewrite 1 reflection direction, rewrote ${rewrites}`);
    }
    const irradianceParts = fragmentWGSL.split(IBL_SCENE_IRRADIANCE);
    if (irradianceParts.length !== 2) {
        throw new Error(`local cubemap _postCompose: expected to rewrite 1 irradiance source, rewrote ${irradianceParts.length - 1}`);
    }
    fragmentWGSL = irradianceParts.join(LOCAL_IRRADIANCE);
    let lodRewrites = 0;
    fragmentWGSL = fragmentWGSL.replace(/scene\.vImageInfos\.z/g, () => {
        lodRewrites++;
        return "material.localLodGenerationScale";
    });
    if (lodRewrites < 1) {
        throw new Error("local cubemap _postCompose: expected at least 1 environment LOD scale rewrite");
    }
    if (composed._fragmentKey?.split("|").some((id) => id === "clearcoat" || id.startsWith("clearcoat-"))) {
        let clearcoatRewrites = 0;
        fragmentWGSL = fragmentWGSL.replace(/let\s+ccR_ibl\s*=\s*rotateY\(\s*ccR_raw\s*,\s*scene\.envRotationY\s*\);/g, () => {
            clearcoatRewrites++;
            return "let ccR_ibl=rotateY(parallaxCorrectNormal(input.worldPos,ccR_raw,material.vReflectionSize,material.vReflectionPosition),scene.envRotationY);";
        });
        if (clearcoatRewrites !== 1) {
            throw new Error(`local cubemap _postCompose: expected to rewrite 1 clearcoat reflection direction, rewrote ${clearcoatRewrites}`);
        }
    }
    return { ...composed, _fragmentWGSL: fragmentWGSL };
}

export function createLocalCubemapFragment(): ShaderFragment {
    return {
        _id: "local-cubemap",
        _dependencies: ["ibl"],
        _uboFields: [
            { _name: "vReflectionPosition", _type: "vec3<f32>" },
            { _name: "vReflectionSize", _type: "vec3<f32>" },
            { _name: "localLodGenerationScale", _type: "f32" },
            ...LOCAL_SH_FIELDS.map((_name) => ({ _name, _type: "vec3<f32>" as const })),
        ],
        _helperFunctions: LOCAL_CUBEMAP_HELPER,
        _pc: patchLocalCubemapReflection,
    };
}

export const pbrExt: PbrExt = {
    id: "local-cubemap",
    phase: "fragment",
    detect(material) {
        return { f: 0, f2: getLocalEnvironment(material)?.boundingBoxSize ? PBR2_HAS_LOCAL_CUBEMAP : 0 };
    },
    frag(ctx) {
        return (ctx._features2 & PBR2_HAS_LOCAL_CUBEMAP) !== 0 ? createLocalCubemapFragment() : null;
    },
    writeUbo(data, material, offsets) {
        const positionOffset = offsets.get("vReflectionPosition");
        const sizeOffset = offsets.get("vReflectionSize");
        if (positionOffset === undefined || sizeOffset === undefined) {
            return;
        }
        const environment = getLocalEnvironment(material);
        const position = environment?.boundingBoxPosition;
        const size = environment?.boundingBoxSize;
        const po = positionOffset / 4;
        const so = sizeOffset / 4;
        data[po] = position?.[0] ?? 0;
        data[po + 1] = position?.[1] ?? 0;
        data[po + 2] = position?.[2] ?? 0;
        data[so] = size?.[0] ?? 0;
        data[so + 1] = size?.[1] ?? 0;
        data[so + 2] = size?.[2] ?? 0;
        const lodOffset = offsets.get("localLodGenerationScale");
        if (lodOffset !== undefined) {
            data[lodOffset / 4] = environment?._lodGenerationScale ?? 0.8;
        }
        const sh = environment?._sphericalHarmonics;
        for (let i = 0; i < LOCAL_SH_FIELDS.length; i++) {
            const offset = offsets.get(LOCAL_SH_FIELDS[i]!);
            if (offset === undefined) {
                continue;
            }
            const out = offset / 4;
            const input = i * 4;
            data[out] = sh?.[input] ?? 0;
            data[out + 1] = sh?.[input + 1] ?? 0;
            data[out + 2] = sh?.[input + 2] ?? 0;
        }
    },
};

/** @internal Install and register the extension without module-level side effects. */
export function registerPbrLocalCubemapExt(register: (ext: PbrExt) => void): void {
    register(pbrExt);
}
