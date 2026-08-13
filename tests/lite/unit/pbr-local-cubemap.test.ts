import { describe, expect, it } from "vitest";

import type { EnvironmentTextures } from "../../../packages/babylon-lite/src/loader-env/load-env";
import type { PbrMaterialProps } from "../../../packages/babylon-lite/src/material/pbr/pbr-material";
import type { MaterialPlugin } from "../../../packages/babylon-lite/src/material/plugin/material-plugin";
import { pbrExt as clearcoatExt } from "../../../packages/babylon-lite/src/material/pbr/fragments/clearcoat-fragment";
import { pbrExt as alphaTestExt } from "../../../packages/babylon-lite/src/material/pbr/fragments/alpha-test-fragment";
import { createIblFragment, pbrExt as iblExt } from "../../../packages/babylon-lite/src/material/pbr/fragments/ibl-fragment";
import { pbrExt as diffuseLightmapExt } from "../../../packages/babylon-lite/src/material/pbr/fragments/diffuse-lightmap-fragment";
import { createLocalCubemapFragment, pbrExt, registerPbrLocalCubemapExt } from "../../../packages/babylon-lite/src/material/pbr/fragments/local-cubemap-fragment";
import { pbrExt as morphExt } from "../../../packages/babylon-lite/src/material/pbr/fragments/morph-fragment";
import { createPbrComposer } from "../../../packages/babylon-lite/src/material/pbr/pbr-compose";
import { _registerPbrExt, PBR_HAS_ENV, PBR_HAS_SPECULAR_AA } from "../../../packages/babylon-lite/src/material/pbr/pbr-flags";
import { _computePbrMaterialFeatures } from "../../../packages/babylon-lite/src/material/pbr/pbr-material";
import { createPbrMeshBindGroup } from "../../../packages/babylon-lite/src/material/pbr/pbr-pipeline";
import { enablePbrLocalCubemap } from "../../../packages/babylon-lite/src/material/pbr/enable-pbr-local-cubemap";
import { registerPbrPlugins } from "../../../packages/babylon-lite/src/material/plugin/pbr-plugin-bridge";
import { createPbrTemplate } from "../../../packages/babylon-lite/src/material/pbr/pbr-template";
import { MSH_HAS_MORPH_TARGETS } from "../../../packages/babylon-lite/src/material/mesh-features";
import { composeShader } from "../../../packages/babylon-lite/src/shader/shader-composer";

function makeEnvironment(overrides: Partial<EnvironmentTextures> = {}): EnvironmentTextures {
    return overrides as EnvironmentTextures;
}

describe("PBR local cubemap projection", () => {
    it("binds a loader-created material's local environment after the feature opt-in", async () => {
        const baseView = {} as GPUTextureView;
        const baseSampler = {} as GPUSampler;
        const ormView = {} as GPUTextureView;
        const ormSampler = {} as GPUSampler;
        const globalCubeView = {} as GPUTextureView;
        const localCubeView = {} as GPUTextureView;
        const globalEnvironment = makeEnvironment({
            _brdfLutView: {} as GPUTextureView,
            _brdfSampler: {} as GPUSampler,
            _specularCubeView: globalCubeView,
            _cubeSampler: {} as GPUSampler,
        });
        const localEnvironment = makeEnvironment({
            _brdfLutView: {} as GPUTextureView,
            _brdfSampler: {} as GPUSampler,
            _specularCubeView: localCubeView,
            _cubeSampler: {} as GPUSampler,
        });
        const material = {
            baseColorTexture: { view: baseView, sampler: baseSampler },
            ormTexture: { view: ormView, sampler: ormSampler },
            localEnvironment,
        } as PbrMaterialProps;
        let descriptor: GPUBindGroupDescriptor | undefined;
        const engine = {
            _device: {
                createBindGroup(value: GPUBindGroupDescriptor): GPUBindGroup {
                    descriptor = value;
                    return {} as GPUBindGroup;
                },
            },
        };

        _registerPbrExt(iblExt);
        await enablePbrLocalCubemap();
        createPbrMeshBindGroup(
            engine as never,
            { _features: 0, _features2: 0, _meshFeatures: 0, _meshBGL: {} as GPUBindGroupLayout, _shadowBGL: null } as never,
            { _fragmentKey: "ibl" } as never,
            {} as GPUBuffer,
            {} as GPUBuffer,
            material,
            globalEnvironment,
            null
        );

        const resources = Array.from(descriptor!.entries, (entry) => entry.resource);
        expect(resources).toContain(localCubeView);
        expect(resources).not.toContain(globalCubeView);
    });

    it("keeps plugin shader variants separate from local-cubemap feature bits", () => {
        const pluginMarker = "if(material.materialAlpha < -1.0){discard;}";
        const plugin: MaterialPlugin = {
            name: "local-cubemap-regression",
            getCustomCode: (shaderType) => (shaderType === "fragment" ? { CUSTOM_FRAGMENT_UPDATE_ALPHA: pluginMarker } : null),
        };
        const material = {
            alphaCutOff: 0.5,
            localEnvironment: makeEnvironment({ boundingBoxSize: [8, 6, 4] }),
            plugins: [plugin],
        } as PbrMaterialProps & { plugins: MaterialPlugin[] };

        _registerPbrExt(alphaTestExt);
        _registerPbrExt(iblExt);
        registerPbrLocalCubemapExt(_registerPbrExt);
        registerPbrPlugins(_registerPbrExt);

        const renderFeatures = _computePbrMaterialFeatures(material);
        expect(renderFeatures.features2).toBe(1 << 29);
        expect(material._pi).toBeGreaterThan(0);

        const composePbr = createPbrComposer({
            _singleLightWGSL: "",
            _getSingleLightBlock: null,
            _multiLightWGSL: "",
            _multiLightLoop: "",
            _toneMappingHelpers: "",
            _toneMappingCall: "",
            _fogHelper: "",
            _fogBlock: "",
            _createPbrTemplateExt: null,
            _anisoExt: null,
            _iblSkyboxCalc: "",
            _flatNormalWgsl: "",
            _gammaTemplate: null,
            _createPbrShadowFragment: null,
            _shadowLights: [],
            _createThinInstanceFragment: null,
        });
        const result = composePbr(renderFeatures.features, renderFeatures.features2, 0, PBR_HAS_ENV, 0, "", "", undefined, "", 0, material._pi);

        expect(result._fragmentWGSL).toContain("parallaxCorrectNormal(input.worldPos,R_raw");
        expect(result._fragmentWGSL).toContain("if(alpha*material.materialAlpha<material.alphaCutOff){discard;}");
        expect(result._fragmentWGSL).toContain(pluginMarker);
    });

    it("activates only when a material-local environment defines a projection box", () => {
        expect(pbrExt.detect?.({})).toEqual({ f: 0, f2: 0 });
        expect(pbrExt.detect?.({ localEnvironment: makeEnvironment({ boundingBoxPosition: [1, 2, 3] }) })).toEqual({ f: 0, f2: 0 });
        expect(pbrExt.detect?.({ localEnvironment: makeEnvironment({ boundingBoxSize: [4, 5, 6] }) })).toEqual({
            f: 0,
            f2: 1 << 29,
        });
    });

    it("injects Babylon.js box-projection math before environment rotation", () => {
        _registerPbrExt(iblExt);
        registerPbrLocalCubemapExt(_registerPbrExt);
        const localCubemapFeature = pbrExt.detect?.({ localEnvironment: makeEnvironment({ boundingBoxSize: [4, 5, 6] }) }).f2 ?? 0;
        const composePbr = createPbrComposer({
            _singleLightWGSL: "",
            _getSingleLightBlock: null,
            _multiLightWGSL: "",
            _multiLightLoop: "",
            _toneMappingHelpers: "",
            _toneMappingCall: "",
            _fogHelper: "",
            _fogBlock: "",
            _createPbrTemplateExt: null,
            _anisoExt: null,
            _iblSkyboxCalc: "",
            _flatNormalWgsl: "",
            _gammaTemplate: null,
            _createPbrShadowFragment: null,
            _shadowLights: [],
            _createThinInstanceFragment: null,
        });

        const result = composePbr(0, localCubemapFeature, 0, PBR_HAS_ENV);

        expect(result._fragmentWGSL).toContain("fn parallaxCorrectNormal");
        expect(result._fragmentWGSL).toContain("let largestIntersec=max(intersecAtMaxPlane,intersecAtMinPlane);");
        expect(result._fragmentWGSL).toContain(
            "let R=rotateY(parallaxCorrectNormal(input.worldPos,R_raw,material.vReflectionSize,material.vReflectionPosition),scene.envRotationY);"
        );
        expect(result._fragmentWGSL).toContain("let environmentIrradiance = (material.localSphericalL00.rgb");
        expect(result._fragmentWGSL).not.toContain("let environmentIrradiance = (scene.vSphericalL00.rgb");
        expect(result._fragmentWGSL).toContain("material.localLodGenerationScale");
        expect(result._fragmentWGSL).not.toContain("scene.vImageInfos.z");
        expect(result._materialUboSpec?._offsets.has("vReflectionPosition")).toBe(true);
        expect(result._materialUboSpec?._offsets.has("vReflectionSize")).toBe(true);
        expect(result._materialUboSpec?._offsets.has("localSphericalL00")).toBe(true);
        expect(result._materialUboSpec?._offsets.has("localSphericalL22")).toBe(true);

        const unbounded = composePbr(0, 0, 0, PBR_HAS_ENV);
        expect(unbounded._fragmentWGSL).not.toContain("parallaxCorrectNormal");
        expect(unbounded._materialUboSpec?._offsets.has("vReflectionPosition")).toBe(false);

        _registerPbrExt(morphExt);
        const morphed = composePbr(0, localCubemapFeature, MSH_HAS_MORPH_TARGETS, PBR_HAS_ENV);
        expect(morphed._fragmentWGSL).toContain("parallaxCorrectNormal(input.worldPos,R_raw");
        expect(morphed._vertexWGSL).toContain("var<storage, read> morphDeltas:");
        expect(morphed._vertexWGSL).toContain("var<storage, read> morph:");

        _registerPbrExt(clearcoatExt);
        const clearcoatFeatures = clearcoatExt.detect?.({ clearCoat: { isEnabled: true } }) ?? { f: 0, f2: 0 };
        const clearcoat = composePbr(clearcoatFeatures.f, localCubemapFeature | clearcoatFeatures.f2, 0, PBR_HAS_ENV);
        expect(clearcoat._fragmentWGSL).toContain("parallaxCorrectNormal(input.worldPos,R_raw");
        expect(clearcoat._fragmentWGSL).toContain("parallaxCorrectNormal(input.worldPos,ccR_raw");

        const clearcoatVariant = composePbr(clearcoatFeatures.f | PBR_HAS_SPECULAR_AA, localCubemapFeature | clearcoatFeatures.f2, 0, PBR_HAS_ENV);
        expect(clearcoatVariant._fragmentKey).toContain("clearcoat-A");
        expect(clearcoatVariant._fragmentWGSL).toContain("parallaxCorrectNormal(input.worldPos,ccR_raw");
    });

    it("writes the probe position, full box size, and local SH to the material UBO", () => {
        const result = composeShader(createPbrTemplate({ _normalMode: "none", _hasIbl: true }), [createIblFragment(false), createLocalCubemapFragment()]);
        const spec = result._materialUboSpec!;
        const data = new Float32Array(spec._totalBytes / 4);
        const material = {
            localEnvironment: makeEnvironment({
                boundingBoxPosition: [1.25, -2.5, 3.75],
                boundingBoxSize: [8, 6, 4],
                _lodGenerationScale: 0.65,
                _sphericalHarmonics: Float32Array.from({ length: 36 }, (_, i) => i + 0.5),
            }),
        } as PbrMaterialProps;

        pbrExt.writeUbo?.(data, material, spec._offsets);

        const positionOffset = spec._offsets.get("vReflectionPosition")! / 4;
        const sizeOffset = spec._offsets.get("vReflectionSize")! / 4;
        const lodOffset = spec._offsets.get("localLodGenerationScale")! / 4;
        expect([...data.slice(positionOffset, positionOffset + 3)]).toEqual([1.25, -2.5, 3.75]);
        expect([...data.slice(sizeOffset, sizeOffset + 3)]).toEqual([8, 6, 4]);
        expect(data[lodOffset]).toBeCloseTo(0.65);
        const l00Offset = spec._offsets.get("localSphericalL00")! / 4;
        const l22Offset = spec._offsets.get("localSphericalL22")! / 4;
        expect([...data.slice(l00Offset, l00Offset + 3)]).toEqual([0.5, 1.5, 2.5]);
        expect([...data.slice(l22Offset, l22Offset + 3)]).toEqual([32.5, 33.5, 34.5]);
    });

    it("replaces diffuse IBL with baked irradiance without modulating local reflections", () => {
        _registerPbrExt(iblExt);
        _registerPbrExt(diffuseLightmapExt);
        registerPbrLocalCubemapExt(_registerPbrExt);
        const material = {
            _diffuseLightmapTexture: { uAng: 0 },
            localEnvironment: makeEnvironment({ boundingBoxSize: [8, 6, 4] }),
        };
        const lightmapFeatures = diffuseLightmapExt.detect?.(material) ?? { f: 0, f2: 0 };
        const localFeatures = pbrExt.detect?.(material) ?? { f: 0, f2: 0 };
        const composePbr = createPbrComposer({
            _singleLightWGSL: "",
            _getSingleLightBlock: null,
            _multiLightWGSL: "",
            _multiLightLoop: "",
            _toneMappingHelpers: "",
            _toneMappingCall: "",
            _fogHelper: "",
            _fogBlock: "",
            _createPbrTemplateExt: null,
            _anisoExt: null,
            _iblSkyboxCalc: "",
            _flatNormalWgsl: "",
            _gammaTemplate: null,
            _createPbrShadowFragment: null,
            _shadowLights: [],
            _createThinInstanceFragment: null,
        });

        const result = composePbr(lightmapFeatures.f, lightmapFeatures.f2 | localFeatures.f2, 0, PBR_HAS_ENV);

        expect(result._fragmentWGSL).toContain("let bakedIrradiance=textureSample(lmTexture,lmSampler,input.uv).rgb*material.lmLvl;");
        expect(result._fragmentWGSL).toContain("finalIrradiance=bakedIrradiance*surfaceAlbedo*occlusion;");
        expect(result._fragmentWGSL).toContain("parallaxCorrectNormal(input.worldPos,R_raw");
        expect(result._fragmentWGSL).not.toContain("color=(color-emissive)");
    });
});
