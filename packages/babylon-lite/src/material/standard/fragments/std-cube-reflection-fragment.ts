/** Cube reflection fragment — dynamically imported for scenes with cube reflection textures. */
import type { ShaderFragment } from "../../../shader/fragment-types.js";
import type { StdExt } from "../standard-flags.js";
import { CUBE_REFLECTION_GAMMA, HAS_CUBE_REFLECTION } from "../standard-flags.js";

export function createStdCubeReflectionFragment(features: number): ShaderFragment {
    const gamma = features & CUBE_REFLECTION_GAMMA;
    return {
        _id: "std-cube-reflection",
        _bindings: [
            { _name: "cRT", _type: { _kind: "texture", _textureType: "texture_cube<f32>" }, _visibility: 0x2 },
            { _name: "cRS", _type: { _kind: "sampler", _samplerType: "sampler" }, _visibility: 0x2 },
        ],
        _fragmentSlots: {
            AD: `{let v=normalize(input.vp-scene.vEyePosition.xyz);reflectionColor=${gamma ? "pow(" : ""}textureSample(cRT,cRS,reflect(v,normalW)).rgb${gamma ? ",vec3<f32>(1.0/2.2))" : ""}*mat.rLvl;}`,
        },
    };
}

export const stdCubeReflectionExt: StdExt = {
    _id: "std-cube-reflection",
    _phase: "mesh",
    _feature: HAS_CUBE_REFLECTION,
    _frag: createStdCubeReflectionFragment,
    _bind(mat, entries, b) {
        const cube = mat.reflectionCubeTexture!;
        entries.push({ binding: b++, resource: cube._v });
        entries.push({ binding: b++, resource: cube._s });
        return b;
    },
    // Cube textures are tracked separately; no Texture2D[] contribution.
};
