/**
 * Opt-in box-projected local image-based lighting.
 *
 * This module is reachable only through enablePbrLocalCubemap(). It patches the
 * ordinary scene-IBL shader when one exists and supplies a complete local IBL
 * fragment when the scene has no global environment.
 */

import type { ComposedShader, ShaderFragment, UboField } from "../../../shader/fragment-types.js";
import type { PbrExt, _PbrBindCtx, _PbrFragCtx } from "../pbr-flags.js";
import { _getPbrLocalEnvironment, type PbrLocalEnvironmentState } from "../pbr-local-cubemap-state.js";
import {
    _PBR_LOCAL_ENVIRONMENT_DEBUG_COLOR_FLAG,
    _PBR_LOCAL_ENVIRONMENT_PARALLAX_FLAG,
    _PBR_LOCAL_ENVIRONMENT_SPHERE_FLAG,
    MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES,
    MAX_PBR_LOCAL_ENVIRONMENT_PROBES,
} from "../pbr-local-cubemap-limits.js";

const PBR_HAS_LOCAL_PROBE_SET = 1 << 31;
const PBR_HAS_LOCAL_CUBEMAP = 1 << 24;
const STAGE_FRAGMENT = 0x2;

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

const LOCAL_SH_UBO_FIELDS: readonly UboField[] = LOCAL_SH_FIELDS.map((_name) => ({ _name, _type: "vec3<f32>" }));

const IBL_HELPERS = `fn environmentHorizonOcclusion(V:vec3f,N:vec3f,geoN:vec3f)->f32{
let R=reflect(V,N);
let temp=saturate(1.0+1.1*dot(R,geoN));
return temp*temp;
}
fn getEnergyConservationFactor(F0:vec3f,brdfY:f32)->vec3f{
return 1.0+F0*(1.0/brdfY-1.0);
}
fn rotateY(v:vec3f,angle:f32)->vec3f{
let c=cos(angle);
let s=sin(angle);
return vec3f(v.x*c+v.z*s,v.y,-v.x*s+v.z*c);
}`;

const SINGLE_BOX_CUBEMAP_HELPER = `fn parallaxCorrectNormal(vertexPos:vec3f,origVec:vec3f,cubeSize:vec3f,cubePos:vec3f)->vec3f{
let invOrigVec=vec3f(1.0)/origVec;
let halfSize=cubeSize*0.5;
let intersecAtMaxPlane=(cubePos+halfSize-vertexPos)*invOrigVec;
let intersecAtMinPlane=(cubePos-halfSize-vertexPos)*invOrigVec;
let largestIntersec=max(intersecAtMaxPlane,intersecAtMinPlane);
let distance=min(min(largestIntersec.x,largestIntersec.y),largestIntersec.z);
return vertexPos+origVec*distance-cubePos;
}`;

const SINGLE_SPHERE_CUBEMAP_HELPER = `fn parallaxCorrectNormal(vertexPos:vec3f,origVec:vec3f,sphereSize:vec3f,spherePos:vec3f)->vec3f{
let localPos=vertexPos-spherePos;
let a=dot(origVec,origVec);
let b=dot(localPos,origVec);
let radius=sphereSize.x*0.5;
let c=dot(localPos,localPos)-radius*radius;
let determinant=b*b-a*c;
if(determinant<0.0){return origVec;}
let distance=(-b+sqrt(determinant))/max(a,0.00001);
if(distance<=0.0){return origVec;}
return localPos+origVec*distance;
}`;

const IBL_SCENE_IRRADIANCE = `let environmentIrradiance = (scene.vSphericalL00.rgb
  + scene.vSphericalL1_1.rgb * N_env.y + scene.vSphericalL10.rgb * N_env.z + scene.vSphericalL11.rgb * N_env.x
  + scene.vSphericalL2_2.rgb * (N_env.y * N_env.x) + scene.vSphericalL2_1.rgb * (N_env.y * N_env.z)
  + scene.vSphericalL20.rgb * (3.0 * N_env.z * N_env.z - 1.0) + scene.vSphericalL21.rgb * (N_env.z * N_env.x)
  + scene.vSphericalL22.rgb * (N_env.x * N_env.x - N_env.y * N_env.y)) * material.environmentIntensity;`;

const IBL_LOCAL_IRRADIANCE = `let environmentIrradiance = (material.localSphericalL00.rgb
  + material.localSphericalL1_1.rgb * N_env.y + material.localSphericalL10.rgb * N_env.z + material.localSphericalL11.rgb * N_env.x
  + material.localSphericalL2_2.rgb * (N_env.y * N_env.x) + material.localSphericalL2_1.rgb * (N_env.y * N_env.z)
  + material.localSphericalL20.rgb * (3.0 * N_env.z * N_env.z - 1.0) + material.localSphericalL21.rgb * (N_env.z * N_env.x)
  + material.localSphericalL22.rgb * (N_env.x * N_env.x - N_env.y * N_env.y)) * material.environmentIntensity;`;

function createProbeArrayHelpers(includeIblHelpers: boolean): string {
    return `${includeIblHelpers ? IBL_HELPERS : ""}
struct LocalEnvironmentProbe{
projectionCentreAndLayer:vec4f,
projectionHalfSizeAndLodScale:vec4f,
capturePositionAndLodBias:vec4f,
influenceCentreAndCos:vec4f,
influenceInnerHalfSizeAndSin:vec4f,
influenceOuterHalfSize:vec4f,
}
struct localProbeDataUniforms{
params:vec4u,
probes:array<LocalEnvironmentProbe,${MAX_PBR_LOCAL_ENVIRONMENT_PROBES}>,
}
struct localProbeGridUniforms{
minimumAndInverseCellSize:vec4f,
dimensionsAndStride:vec4u,
indices:array<u32>,
}
fn probeToLocal(v:vec3f,c:f32,s:f32)->vec3f{
return vec3f(c*v.x-s*v.z,v.y,s*v.x+c*v.z);
}
fn probeToWorld(v:vec3f,c:f32,s:f32)->vec3f{
return vec3f(c*v.x+s*v.z,v.y,-s*v.x+c*v.z);
}
fn localProbeVoxelBase(worldPos:vec3f)->u32{
let dimensions=localProbeGrid.dimensionsAndStride.xyz;
let raw=vec3i(floor((worldPos-localProbeGrid.minimumAndInverseCellSize.xyz)*localProbeGrid.minimumAndInverseCellSize.w));
let coordinates=clamp(raw,vec3i(0),vec3i(dimensions)-vec3i(1));
let cellIndex=(u32(coordinates.z)*dimensions.y+u32(coordinates.y))*dimensions.x+u32(coordinates.x);
return cellIndex*localProbeGrid.dimensionsAndStride.w;
}
fn voxelProbeIndex(base:u32,slot:u32)->u32{
return localProbeGrid.indices[base+1u+slot];
}
fn localProbeIsSphere(probe:LocalEnvironmentProbe)->bool{
return (bitcast<u32>(probe.influenceOuterHalfSize.w)&${_PBR_LOCAL_ENVIRONMENT_SPHERE_FLAG}u)!=0u;
}
fn insideProbeVolume(localPosition:vec3f,extent:vec3f,isSphere:bool)->bool{
return select(all(abs(localPosition)<=extent),length(localPosition)<=extent.x,isSphere);
}
fn probeNdf(localPosition:vec3f,innerExtent:vec3f,outerExtent:vec3f,isSphere:bool)->f32{
if(isSphere){
return (length(localPosition)-innerExtent.x)/max(outerExtent.x-innerExtent.x,0.00001);
}
let span=max(outerExtent-innerExtent,vec3f(0.00001));
let axisNdf=(abs(localPosition)-innerExtent)/span;
return max(axisNdf.x,max(axisNdf.y,axisNdf.z));
}
fn probeReflectionDirection(worldPos:vec3f,worldRay:vec3f,probe:LocalEnvironmentProbe)->vec3f{
if((localProbeData.params.w&${_PBR_LOCAL_ENVIRONMENT_PARALLAX_FLAG}u)==0u){return worldRay;}
let c=probe.influenceCentreAndCos.w;
let s=probe.influenceInnerHalfSizeAndSin.w;
let boxCentre=probe.projectionCentreAndLayer.xyz;
let localPos=probeToLocal(worldPos-boxCentre,c,s);
let localRay=probeToLocal(worldRay,c,s);
let halfSize=probe.projectionHalfSizeAndLodScale.xyz;
if(localProbeIsSphere(probe)){
let a=dot(localRay,localRay);
let b=dot(localPos,localRay);
let radius=halfSize.x;
let determinant=b*b-a*(dot(localPos,localPos)-radius*radius);
if(determinant<0.0){return worldRay;}
let distance=(-b+sqrt(determinant))/max(a,0.00001);
if(distance<=0.0){return worldRay;}
let localHit=localPos+localRay*distance;
let localCapture=probeToLocal(probe.capturePositionAndLodBias.xyz-boxCentre,c,s);
return probeToWorld(localHit-localCapture,c,s);
}
let invRay=vec3f(1.0)/localRay;
let maxPlane=(halfSize-localPos)*invRay;
let minPlane=(-halfSize-localPos)*invRay;
let furthest=max(maxPlane,minPlane);
let distance=min(furthest.x,min(furthest.y,furthest.z));
let localHit=localPos+localRay*distance;
let localCapture=probeToLocal(probe.capturePositionAndLodBias.xyz-boxCentre,c,s);
return probeToWorld(localHit-localCapture,c,s);
}
fn localProbeDebugColor(probe:LocalEnvironmentProbe)->vec3f{
let packed=bitcast<u32>(probe.influenceOuterHalfSize.w);
return vec3f(f32(packed&255u),f32((packed>>8u)&255u),f32((packed>>16u)&255u))/255.0;
}
fn sampleOneLocalProbe(probeIndex:u32,worldPos:vec3f,worldRay:vec3f,alphaG:f32,envRotationY:f32)->vec3f{
let probe=localProbeData.probes[probeIndex];
if((localProbeData.params.w&${_PBR_LOCAL_ENVIRONMENT_DEBUG_COLOR_FLAG}u)!=0u){return localProbeDebugColor(probe);}
let direction=rotateY(probeReflectionDirection(worldPos,worldRay,probe),envRotationY);
let dimension=f32(textureDimensions(localProbeTexture).x);
let maxLevel=f32(textureNumLevels(localProbeTexture)-1);
let lod=log2(dimension*alphaG)*probe.projectionHalfSizeAndLodScale.w+probe.capturePositionAndLodBias.w;
return textureSampleLevel(localProbeTexture,localProbeSampler,direction,i32(probe.projectionCentreAndLayer.w),clamp(lod,0.0,maxLevel)).rgb;
}
fn sampleLocalProbeRadiance(worldPos:vec3f,worldRay:vec3f,alphaG:f32,envRotationY:f32,intensity:f32)->vec3f{
let voxelBase=localProbeVoxelBase(worldPos);
let candidateCount=min(localProbeGrid.indices[voxelBase],${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}u);
let outputIntensity=select(intensity,1.0,(localProbeData.params.w&${_PBR_LOCAL_ENVIRONMENT_DEBUG_COLOR_FLAG}u)!=0u);
var ndfs:array<f32,${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}>;
var included:array<bool,${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}>;
var sumNdf=0.0;
var inverseSumNdf=0.0;
var outerCount=0u;
var nearestProbeIndex=voxelProbeIndex(voxelBase,0u);
var nearestNdf=1e30;
for(var slot=0u;slot<${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}u;slot++){
if(slot>=candidateCount){break;}
let probeIndex=voxelProbeIndex(voxelBase,slot);
let probe=localProbeData.probes[probeIndex];
let localPosition=probeToLocal(worldPos-probe.influenceCentreAndCos.xyz,probe.influenceCentreAndCos.w,probe.influenceInnerHalfSizeAndSin.w);
let isSphere=localProbeIsSphere(probe);
if(insideProbeVolume(localPosition,probe.influenceInnerHalfSizeAndSin.xyz,isSphere)){
return sampleOneLocalProbe(probeIndex,worldPos,worldRay,alphaG,envRotationY)*outputIntensity;
}
let rawNdf=probeNdf(localPosition,probe.influenceInnerHalfSizeAndSin.xyz,probe.influenceOuterHalfSize.xyz,isSphere);
if(rawNdf<nearestNdf){
nearestNdf=rawNdf;
nearestProbeIndex=probeIndex;
}
if(insideProbeVolume(localPosition,probe.influenceOuterHalfSize.xyz,isSphere)){
let ndf=clamp(rawNdf,0.0,1.0);
ndfs[slot]=ndf;
included[slot]=true;
sumNdf+=ndf;
inverseSumNdf+=1.0-ndf;
outerCount++;
}
}
if(outerCount==0u){
return sampleOneLocalProbe(nearestProbeIndex,worldPos,worldRay,alphaG,envRotationY)*outputIntensity;
}
if(outerCount==1u){
for(var slot=0u;slot<${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}u;slot++){
if(slot>=candidateCount){break;}
if(included[slot]){
return sampleOneLocalProbe(voxelProbeIndex(voxelBase,slot),worldPos,worldRay,alphaG,envRotationY)*outputIntensity;
}
}
}
var weights:array<f32,${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}>;
var sumWeights=0.0;
let countMinusOne=f32(outerCount-1u);
for(var slot=0u;slot<${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}u;slot++){
if(slot>=candidateCount){break;}
if(included[slot]){
let ndf=ndfs[slot];
let boundaryWeight=(1.0-ndf/max(sumNdf,0.00001))/countMinusOne;
let centreWeight=(1.0-ndf)/max(inverseSumNdf,0.00001);
let weight=max(0.0,boundaryWeight*centreWeight);
weights[slot]=weight;
sumWeights+=weight;
}
}
if(sumWeights<=0.00001){
return sampleOneLocalProbe(nearestProbeIndex,worldPos,worldRay,alphaG,envRotationY)*outputIntensity;
}
var radiance=vec3f(0.0);
for(var slot=0u;slot<${MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES}u;slot++){
if(slot>=candidateCount){break;}
let weight=weights[slot]/sumWeights;
if(weight>0.00001){
radiance+=sampleOneLocalProbe(voxelProbeIndex(voxelBase,slot),worldPos,worldRay,alphaG,envRotationY)*weight;
}
}
return radiance*outputIntensity;
}`;
}

function replaceExactly(source: string, needle: string, replacement: string, label: string): string {
    const parts = source.split(needle);
    if (parts.length !== 2) {
        throw new Error(`${label}: expected to rewrite 1 occurrence, rewrote ${parts.length - 1}`);
    }
    return parts.join(replacement);
}

function patchSingleSceneIbl(composed: ComposedShader): ComposedShader {
    let reflectionRewrites = 0;
    let fragmentWGSL = composed._fragmentWGSL.replace(/let\s+R\s*=\s*rotateY\(\s*R_raw\s*,\s*scene\.envRotationY\s*\);/g, () => {
        reflectionRewrites++;
        return "let R=rotateY(parallaxCorrectNormal(input.worldPos,R_raw,material.vReflectionSize,material.vReflectionPosition),scene.envRotationY);";
    });
    if (reflectionRewrites !== 1) {
        throw new Error(`local cubemap _postCompose: expected to rewrite 1 reflection direction, rewrote ${reflectionRewrites}`);
    }
    fragmentWGSL = replaceExactly(fragmentWGSL, IBL_SCENE_IRRADIANCE, IBL_LOCAL_IRRADIANCE, "local cubemap irradiance");
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

function patchProbeGridStorage(composed: ComposedShader): ComposedShader {
    let binding = -1;
    let rewrites = 0;
    const fragmentWGSL = composed._fragmentWGSL.replace(/@group\(1\)@binding\((\d+)\)\s*var<uniform>\s*localProbeGrid\s*:/g, (_match, value: string) => {
        binding = Number(value);
        rewrites++;
        return `@group(1)@binding(${value}) var<storage, read> localProbeGrid:`;
    });
    if (rewrites !== 1) {
        throw new Error(`local probe array _postCompose: expected to rewrite 1 voxel-grid binding declaration, rewrote ${rewrites}`);
    }
    const entries = (composed._meshBGLDescriptor.entries as GPUBindGroupLayoutEntry[]).map((entry) =>
        entry.binding === binding ? { ...entry, buffer: { type: "read-only-storage" as const } } : entry
    );
    return { ...composed, _fragmentWGSL: fragmentWGSL, _meshBGLDescriptor: { ...composed._meshBGLDescriptor, entries } };
}

function patchProbeCubeArray(composed: ComposedShader): ComposedShader {
    let binding = -1;
    let rewrites = 0;
    const fragmentWGSL = composed._fragmentWGSL.replace(/@group\(1\)@binding\((\d+)\)\s*var\s+localProbeTexture\s*:\s*texture_cube<f32>/g, (_match, value: string) => {
        binding = Number(value);
        rewrites++;
        return `@group(1)@binding(${value}) var localProbeTexture:texture_cube_array<f32>`;
    });
    if (rewrites !== 1) {
        throw new Error(`local probe array _postCompose: expected to rewrite 1 cube-array binding declaration, rewrote ${rewrites}`);
    }
    const entries = (composed._meshBGLDescriptor.entries as GPUBindGroupLayoutEntry[]).map((entry) =>
        entry.binding === binding && entry.texture ? { ...entry, texture: { ...entry.texture, viewDimension: "cube-array" as const } } : entry
    );
    return { ...composed, _fragmentWGSL: fragmentWGSL, _meshBGLDescriptor: { ...composed._meshBGLDescriptor, entries } };
}

function patchProbeSceneIbl(composed: ComposedShader): ComposedShader {
    let fragmentWGSL = composed._fragmentWGSL;
    let baseRewrites = 0;
    fragmentWGSL = fragmentWGSL.replace(
        /var\s+environmentRadiance\s*=\s*textureSampleLevel\(\s*iblTexture\s*,\s*iblSampler\s*,\s*R\s*,\s*clamp\(\s*specLod\s*,\s*0\.0\s*,\s*maxLod\s*\)\s*\)\.rgb\s*\*\s*material\.environmentIntensity\s*;/g,
        () => {
            baseRewrites++;
            return "var environmentRadiance=sampleLocalProbeRadiance(input.worldPos,R_raw,alphaG,scene.envRotationY,material.environmentIntensity);let localProbeDebugOutput=environmentRadiance;";
        }
    );
    if (baseRewrites !== 1) {
        throw new Error(`local probe array _postCompose: expected to rewrite 1 base radiance sample, rewrote ${baseRewrites}`);
    }
    if (composed._fragmentKey?.split("|").some((id) => id === "clearcoat" || id.startsWith("clearcoat-"))) {
        let rewrites = 0;
        fragmentWGSL = fragmentWGSL.replace(
            /let\s+ccEnvRadiance_ibl\s*=\s*textureSampleLevel\(\s*iblTexture\s*,\s*iblSampler\s*,\s*ccR_ibl\s*,\s*clamp\(\s*ccSpecLod_ibl\s*,\s*0\.0\s*,\s*maxLod\s*\)\s*\)\.rgb\s*\*\s*material\.environmentIntensity\s*;/g,
            () => {
                rewrites++;
                return "let ccEnvRadiance_ibl=sampleLocalProbeRadiance(input.worldPos,ccR_raw,ccAlphaG_ibl,scene.envRotationY,material.environmentIntensity);";
            }
        );
        if (rewrites !== 1) {
            throw new Error(`local probe array _postCompose: expected to rewrite 1 clearcoat radiance sample, rewrote ${rewrites}`);
        }
    }
    if (composed._fragmentKey?.split("|").some((id) => id === "sheen" || id.startsWith("sheen-"))) {
        let rewrites = 0;
        fragmentWGSL = fragmentWGSL.replace(
            /let\s+shEnvRadiance\s*=\s*textureSampleLevel\(\s*iblTexture\s*,\s*iblSampler\s*,\s*R\s*,\s*clamp\(\s*shSpecLod\s*,\s*0\.0\s*,\s*maxLod\s*\)\s*\)\.rgb\s*\*\s*material\.environmentIntensity\s*;/g,
            () => {
                rewrites++;
                return "let shEnvRadiance=sampleLocalProbeRadiance(input.worldPos,R_raw,shAlphaG_ibl,scene.envRotationY,material.environmentIntensity);";
            }
        );
        if (rewrites !== 1) {
            throw new Error(`local probe array _postCompose: expected to rewrite 1 sheen radiance sample, rewrote ${rewrites}`);
        }
    }
    return patchProbeCubeArray(patchProbeGridStorage({ ...composed, _fragmentWGSL: fragmentWGSL }));
}

function standaloneIblCode(reflectionCode: string, radianceCode: string, hasNormal: boolean, debugCode = ""): string {
    const ehoLine = hasNormal ? "let eho=environmentHorizonOcclusion(-V,N,N_geom);" : "let eho=1.0;";
    return `${reflectionCode}
let N_env=rotateY(N,scene.envRotationY);
let brdf=textureSample(brdfLUT,brdfSampler_,vec2f(NdotV,roughness));
let environmentBrdf=brdf.rgb;
let specularEnvironmentReflectance=(colorF90-colorF0)*environmentBrdf.x+colorF0*environmentBrdf.y;
let seo=clamp((NdotVUnclamped+occlusion)*(NdotVUnclamped+occlusion)-1.0+occlusion,0.0,1.0);
${ehoLine}
let colorSpecularEnvReflectance=specularEnvironmentReflectance*seo*eho;
let energyConservation=getEnergyConservationFactor(colorF0,max(environmentBrdf.y,0.001));
${IBL_LOCAL_IRRADIANCE}
${radianceCode}
${debugCode}
environmentRadiance=mix(environmentRadiance,environmentIrradiance,alphaG);
var finalIrradiance=environmentIrradiance*surfaceAlbedo*occlusion;
let finalSpecularScaled=directSpecular*energyConservation;
let finalRadianceScaled=environmentRadiance*colorSpecularEnvReflectance*energyConservation;
color=finalIrradiance+finalRadianceScaled+finalSpecularScaled+directDiffuse+emissive;`;
}

function createSingleFragment(ctx: _PbrFragCtx, sphere: boolean): ShaderFragment {
    const uboFields: UboField[] = [
        { _name: "vReflectionPosition", _type: "vec3<f32>" },
        { _name: "vReflectionSize", _type: "vec3<f32>" },
        { _name: "localLodGenerationScale", _type: "f32" },
        ...LOCAL_SH_UBO_FIELDS,
    ];
    const helper = sphere ? SINGLE_SPHERE_CUBEMAP_HELPER : SINGLE_BOX_CUBEMAP_HELPER;
    if (ctx._hasIbl) {
        return {
            _id: "local-cubemap",
            _dependencies: ["ibl"],
            _uboFields: uboFields,
            _helperFunctions: helper,
            _pc: patchSingleSceneIbl,
        };
    }
    return {
        _id: "local-cubemap",
        _uboFields: uboFields,
        _bindings: [
            { _name: "brdfLUT", _type: { _kind: "texture", _textureType: "texture_2d<f32>" }, _visibility: STAGE_FRAGMENT },
            { _name: "brdfSampler_", _type: { _kind: "sampler", _samplerType: "sampler" }, _visibility: STAGE_FRAGMENT },
            { _name: "iblTexture", _type: { _kind: "texture", _textureType: "texture_cube<f32>" }, _visibility: STAGE_FRAGMENT },
            { _name: "iblSampler", _type: { _kind: "sampler", _samplerType: "sampler" }, _visibility: STAGE_FRAGMENT },
        ],
        _helperFunctions: `${IBL_HELPERS}\n${helper}`,
        _fragmentSlots: {
            AI: standaloneIblCode(
                "let R_raw=reflect(-V,N);let R=rotateY(parallaxCorrectNormal(input.worldPos,R_raw,material.vReflectionSize,material.vReflectionPosition),scene.envRotationY);",
                "let maxLod=f32(textureNumLevels(iblTexture)-1);let cubemapDim=f32(textureDimensions(iblTexture).x);var specLod=log2(cubemapDim*alphaG)*material.localLodGenerationScale;var environmentRadiance=textureSampleLevel(iblTexture,iblSampler,R,clamp(specLod,0.0,maxLod)).rgb*material.environmentIntensity;",
                ctx._hasAnyNormal
            ),
            BA: "luminanceOverAlpha+=dot(finalRadianceScaled,vec3f(0.2126,0.7152,0.0722));",
        },
    };
}

function probeBindings(hasSceneIbl: boolean): NonNullable<ShaderFragment["_bindings"]> {
    return [
        ...(!hasSceneIbl
            ? ([
                  { _name: "brdfLUT", _type: { _kind: "texture", _textureType: "texture_2d<f32>" }, _visibility: STAGE_FRAGMENT },
                  { _name: "brdfSampler_", _type: { _kind: "sampler", _samplerType: "sampler" }, _visibility: STAGE_FRAGMENT },
              ] as const)
            : []),
        { _name: "localProbeData", _type: { _kind: "uniform-buffer" }, _visibility: STAGE_FRAGMENT },
        { _name: "localProbeGrid", _type: { _kind: "uniform-buffer" }, _visibility: STAGE_FRAGMENT },
        { _name: "localProbeTexture", _type: { _kind: "texture", _textureType: "texture_cube<f32>" }, _visibility: STAGE_FRAGMENT },
        { _name: "localProbeSampler", _type: { _kind: "sampler", _samplerType: "sampler" }, _visibility: STAGE_FRAGMENT },
    ];
}

function createProbeArrayFragment(ctx: _PbrFragCtx): ShaderFragment {
    if (ctx._hasIbl) {
        return {
            _id: "local-cubemap",
            _dependencies: ["ibl"],
            _bindings: probeBindings(true),
            _helperFunctions: createProbeArrayHelpers(false),
            _fragmentSlots: {
                BC: `if((localProbeData.params.w&${_PBR_LOCAL_ENVIRONMENT_DEBUG_COLOR_FLAG}u)!=0u){color=localProbeDebugOutput;}`,
            },
            _pc: patchProbeSceneIbl,
        };
    }
    return {
        _id: "local-cubemap",
        _uboFields: LOCAL_SH_UBO_FIELDS,
        _bindings: probeBindings(false),
        _helperFunctions: createProbeArrayHelpers(true),
        _fragmentSlots: {
            AI: standaloneIblCode(
                "let R_raw=reflect(-V,N);let R=rotateY(R_raw,scene.envRotationY);",
                "var environmentRadiance=sampleLocalProbeRadiance(input.worldPos,R_raw,alphaG,scene.envRotationY,material.environmentIntensity);",
                ctx._hasAnyNormal,
                "let localProbeDebugOutput=environmentRadiance;"
            ),
            BC: `if((localProbeData.params.w&${_PBR_LOCAL_ENVIRONMENT_DEBUG_COLOR_FLAG}u)!=0u){color=localProbeDebugOutput;}`,
            BA: "luminanceOverAlpha+=dot(finalRadianceScaled,vec3f(0.2126,0.7152,0.0722));",
        },
        _pc: (composed) => patchProbeCubeArray(patchProbeGridStorage(composed)),
    };
}

function localEnvironmentForState(state: PbrLocalEnvironmentState): {
    _brdfLutView: GPUTextureView;
    _brdfSampler: GPUSampler;
    _specularCubeView: GPUTextureView;
    _cubeSampler: GPUSampler;
    _sphericalHarmonics: Float32Array;
    _lodGenerationScale: number;
} {
    return state.kind === "single" ? state.environment : state.set.probes[0]!.environment;
}

function replaceEntryResource(entries: GPUBindGroupEntry[], current: GPUBindingResource, replacement: GPUBindingResource, label: string): void {
    const entry = entries.find((candidate) => candidate.resource === current);
    if (!entry) {
        throw new Error(`[babylon-lite] local cubemap could not replace the scene ${label} binding`);
    }
    entry.resource = replacement;
}

function bindLocalEnvironment(ctx: _PbrBindCtx, entries: GPUBindGroupEntry[], binding: number, state: PbrLocalEnvironmentState): number {
    const environment = localEnvironmentForState(state);
    if (state.kind === "single" && ctx._env) {
        replaceEntryResource(entries, ctx._env._specularCubeView, environment._specularCubeView, "cubemap");
        replaceEntryResource(entries, ctx._env._cubeSampler, environment._cubeSampler, "cubemap sampler");
        return binding;
    }
    if (!ctx._env) {
        entries.push({ binding: binding++, resource: environment._brdfLutView });
        entries.push({ binding: binding++, resource: environment._brdfSampler });
    }
    if (state.kind === "single") {
        entries.push({ binding: binding++, resource: environment._specularCubeView });
        entries.push({ binding: binding++, resource: environment._cubeSampler });
        return binding;
    }
    entries.push({ binding: binding++, resource: { buffer: state.set._uniformBuffer } });
    entries.push({ binding: binding++, resource: { buffer: state.set._gridBuffer } });
    entries.push({ binding: binding++, resource: state.set._textureView });
    entries.push({ binding: binding++, resource: state.set._sampler });
    return binding;
}

function writeLocalSphericalHarmonics(data: Float32Array, offsets: ReadonlyMap<string, number>, state: PbrLocalEnvironmentState): void {
    const sh = localEnvironmentForState(state)._sphericalHarmonics;
    for (let index = 0; index < LOCAL_SH_FIELDS.length; index++) {
        const offset = offsets.get(LOCAL_SH_FIELDS[index]!);
        if (offset === undefined) {
            continue;
        }
        const output = offset / 4;
        const input = index * 4;
        data[output] = sh[input] ?? 0;
        data[output + 1] = sh[input + 1] ?? 0;
        data[output + 2] = sh[input + 2] ?? 0;
    }
}

export const pbrExt: PbrExt = {
    id: "local-cubemap",
    phase: "fragment",
    detect(material) {
        const state = _getPbrLocalEnvironment(material);
        return state?.kind === "probes"
            ? { f: PBR_HAS_LOCAL_PROBE_SET, f2: 0 }
            : { f: state?.kind === "single" ? PBR_HAS_LOCAL_CUBEMAP | (state.shape === "sphere" ? PBR_HAS_LOCAL_PROBE_SET : 0) : 0, f2: 0 };
    },
    frag(ctx) {
        const hasLocalCubemap = (ctx._features & PBR_HAS_LOCAL_CUBEMAP) !== 0;
        const hasProbeSetOrSphere = (ctx._features & PBR_HAS_LOCAL_PROBE_SET) !== 0;
        if (hasProbeSetOrSphere && !hasLocalCubemap) {
            return createProbeArrayFragment(ctx);
        }
        return hasLocalCubemap ? createSingleFragment(ctx, hasProbeSetOrSphere) : null;
    },
    writeUbo(data, material, offsets) {
        const state = _getPbrLocalEnvironment(material);
        if (!state) {
            return;
        }
        writeLocalSphericalHarmonics(data, offsets, state);
        if (state.kind !== "single") {
            return;
        }
        const positionOffset = offsets.get("vReflectionPosition");
        const sizeOffset = offsets.get("vReflectionSize");
        if (positionOffset !== undefined) {
            data.set(state.projectionPosition, positionOffset / 4);
        }
        if (sizeOffset !== undefined) {
            data.set(state.projectionSize, sizeOffset / 4);
        }
        const lodOffset = offsets.get("localLodGenerationScale");
        if (lodOffset !== undefined) {
            data[lodOffset / 4] = state.environment._lodGenerationScale ?? 0.8;
        }
    },
    bind(ctx, entries, binding) {
        const state = _getPbrLocalEnvironment(ctx._material);
        return state ? bindLocalEnvironment(ctx, entries, binding, state) : binding;
    },
};

/** @internal Install and register the extension without module-level side effects. */
export function registerPbrLocalCubemapExt(register: (ext: PbrExt) => void): void {
    register(pbrExt);
}
