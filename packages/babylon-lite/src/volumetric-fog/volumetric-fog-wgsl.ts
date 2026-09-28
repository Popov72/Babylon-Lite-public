import { opticalTransferWgsl } from "../render/volume/optical-transfer-wgsl.js";
import { buildViewVolumeDepthWgsl } from "../render/volume/view-volume-grid-wgsl.js";
import { wgsl, type WgslSource } from "../shader/wgsl.js";

function fogParamsWgsl(): WgslSource {
    return wgsl`struct FogParams {
cameraWorld:mat4x4f,
grid:vec4f,
depth:vec4f,
fog:vec4f,
scatter:vec4f,
ambient:vec4f,
sunDirection:vec4f,
sunColor:vec4f,
screen:vec4f,
};
@group(0) @binding(0) var<uniform> fogParams:FogParams;
fn fogRayView(uv:vec2f)->vec3f{
let ndc=vec2f(uv.x*2.0-1.0,1.0-uv.y*2.0);
return normalize(vec3f(ndc.x/fogParams.depth.z,ndc.y/fogParams.depth.w,1.0));
}`;
}

function fogFroxelRayWgsl(): WgslSource {
    return wgsl`fn fogFroxelViewRay(id:vec2u)->vec3f{
let uv=clamp((vec2f(id.xy)+vec2f(0.5))*fogParams.grid.w/fogParams.screen.xy,vec2f(0.0),vec2f(1.0));
return fogRayView(uv);
}
fn fogWorldRay(viewRay:vec3f)->vec3f{
return normalize((fogParams.cameraWorld*vec4f(viewRay,0.0)).xyz);
}`;
}

function fogIncidentLightingWgsl(): WgslSource {
    return wgsl`fn fogIncidentLighting(worldRay:vec3f)->vec3f{
let g=fogParams.scatter.w;
let cosTheta=dot(fogParams.sunDirection.xyz,-worldRay);
let phaseDenom=max(1.0+g*g-2.0*g*cosTheta,0.0001);
let phase=(1.0-g*g)/(12.566370614359172*phaseDenom*sqrt(phaseDenom));
return fogParams.ambient.xyz+fogParams.sunColor.xyz*fogParams.sunDirection.w*phase;
}`;
}

/** Density provider WGSL defines fogDensityAt(worldPosition) and may use the base-density helpers. */
export function fogInjectWgsl(source: string | null = null): WgslSource {
    const model = source ?? wgsl`fn fogDensityAt(worldPosition:vec3f)->f32{return fogBaseDensity(worldPosition);}`;
    return wgsl`${fogParamsWgsl()}
${buildViewVolumeDepthWgsl({ kind: "power", exponent: 2 })}
${fogFroxelRayWgsl()}
fn fogBaseExtinction()->f32{return fogParams.fog.x;}
fn fogBaseDensity(worldPosition:vec3f)->f32{
return fogBaseExtinction()*exp(clamp(-fogParams.fog.z*(worldPosition.y-fogParams.fog.y),-80.0,80.0));
}
${model}
${fogIncidentLightingWgsl()}
@group(0) @binding(1) var injected:texture_storage_3d<rgba16float,write>;
@compute @workgroup_size(8,8,1) fn main(@builtin(global_invocation_id) id:vec3u){
let grid=vec3u(fogParams.grid.xyz);
if(any(id>=grid)){return;}
let viewRay=fogFroxelViewRay(id.xy);
let worldRay=fogWorldRay(viewRay);
let viewZ=viewVolumeSliceToDepth(f32(id.z)+0.5,fogParams.depth.x,fogParams.depth.y,fogParams.grid.z);
let worldPosition=fogParams.cameraWorld[3].xyz+worldRay*(viewZ/viewRay.z);
let density=clamp(fogDensityAt(worldPosition),0.0,fogParams.fog.w);
if(density==0.0){
textureStore(injected,vec3i(id),vec4f(0.0));
return;
}
let source=density*fogParams.scatter.xyz*fogIncidentLighting(worldRay);
textureStore(injected,vec3i(id),vec4f(source,density));
}`;
}

export function fogIntegrateWgsl(): WgslSource {
    return wgsl`${fogParamsWgsl()}
${buildViewVolumeDepthWgsl({ kind: "power", exponent: 2 })}
${opticalTransferWgsl()}
@group(0) @binding(1) var injected:texture_3d<f32>;
@group(0) @binding(2) var integrated:texture_storage_3d<rgba16float,write>;
@compute @workgroup_size(8,8,1) fn main(@builtin(global_invocation_id) id:vec3u){
let grid=vec3u(fogParams.grid.xyz);
if(any(id.xy>=grid.xy)){return;}
let uv=clamp((vec2f(id.xy)+vec2f(0.5))*fogParams.grid.w/fogParams.screen.xy,vec2f(0.0),vec2f(1.0));
let ray=fogRayView(uv);
var accumulated=OpticalTransfer(vec3f(0.0),vec3f(1.0));
var previousZ=fogParams.depth.x;
for(var z=0u;z<grid.z;z++){
let farZ=viewVolumeSliceToDepth(f32(z+1u),fogParams.depth.x,fogParams.depth.y,fogParams.grid.z);
let distance=(farZ-previousZ)/ray.z;
let cell=textureLoad(injected,vec3i(vec3u(id.xy,z)),0);
let segment=integrateHomogeneousMedium(cell.rgb,vec3f(cell.a),distance);
accumulated=composeOpticalTransfer(accumulated,segment);
textureStore(integrated,vec3i(vec3u(id.xy,z)),vec4f(accumulated.radiance,accumulated.transmittance.x));
previousZ=farZ;
}
}`;
}

export function fogCompositeWgsl(multisampled: boolean): WgslSource {
    const depthType = multisampled ? "texture_depth_multisampled_2d" : "texture_depth_2d";
    const sampleArg = multisampled ? ",@builtin(sample_index) sampleIndex:u32" : "";
    const depthLoad = multisampled ? "textureLoad(sceneDepth,coord,i32(sampleIndex))" : "textureLoad(sceneDepth,coord,0)";
    return wgsl`${fogParamsWgsl()}
${buildViewVolumeDepthWgsl({ kind: "power", exponent: 2 })}
@group(0) @binding(1) var sceneDepth:${depthType};
@group(0) @binding(2) var integrated:texture_3d<f32>;
@group(0) @binding(3) var integratedSampler:sampler;
@vertex fn vs(@builtin(vertex_index) id:u32)->@builtin(position) vec4f{
let p=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0))[id];
return vec4f(p,0.0,1.0);
}
@fragment fn fs(@builtin(position) position:vec4f${sampleArg})->@location(0) vec4f{
let coord=clamp(vec2i(position.xy),vec2i(0),vec2i(fogParams.screen.xy)-vec2i(1));
let volumeUv=(vec2f(coord)+vec2f(0.5))/(fogParams.grid.xy*fogParams.grid.w);
let deviceDepth=${depthLoad};
var viewZ=fogParams.depth.y;
if(deviceDepth>0.0){
viewZ=fogParams.depth.x*fogParams.screen.z/(fogParams.depth.x+deviceDepth*(fogParams.screen.z-fogParams.depth.x));
}
let depthSlices=u32(fogParams.grid.z);
let slice=viewVolumeDepthToSlice(viewZ,fogParams.depth.x,fogParams.depth.y,fogParams.grid.z);
let index=min(u32(floor(slice)),depthSlices-1u);
let t=clamp(slice-f32(index),0.0,1.0);
var front=vec4f(0.0,0.0,0.0,1.0);
if(index>0u){
front=textureSampleLevel(integrated,integratedSampler,vec3f(volumeUv,(f32(index)-0.5)/fogParams.grid.z),0.0);
}
let back=textureSampleLevel(integrated,integratedSampler,vec3f(volumeUv,(f32(index)+0.5)/fogParams.grid.z),0.0);
let fog=mix(front,back,t);
return vec4f(fog.rgb,clamp(fog.a,0.0,1.0));
}`;
}
