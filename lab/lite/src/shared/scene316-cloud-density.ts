import type { VolumetricFogDensitySource } from "babylon-lite";
import { wgsl } from "babylon-lite/shader/wgsl.js";

export interface Scene316CloudState {
    time: number;
}

const cloudDensityWgsl = wgsl`@group(0) @binding(2) var<uniform> cloudTime:vec4f;
fn fogHash3(cell:vec3i,frame:i32)->f32{
let p=vec3u(cell);
var h=(p.x*1597334677u)^(p.y*3812015801u)^(p.z*2798796415u)^(bitcast<u32>(frame)*374761393u);
h=(h^(h>>16u))*2246822519u;
h=(h^(h>>13u))*3266489917u;
return f32(h^(h>>16u))*(1.0/4294967296.0);
}
fn fogNoise3(pos:vec3f,frame:i32)->f32{
let cell=vec3i(floor(pos));
let f=fract(pos);
let s=f*f*(vec3f(3.0)-2.0*f);
let x00=mix(fogHash3(cell,frame),fogHash3(cell+vec3i(1,0,0),frame),s.x);
let x10=mix(fogHash3(cell+vec3i(0,1,0),frame),fogHash3(cell+vec3i(1,1,0),frame),s.x);
let x01=mix(fogHash3(cell+vec3i(0,0,1),frame),fogHash3(cell+vec3i(1,0,1),frame),s.x);
let x11=mix(fogHash3(cell+vec3i(0,1,1),frame),fogHash3(cell+vec3i(1,1,1),frame),s.x);
return mix(mix(x00,x10,s.y),mix(x01,x11,s.y),s.z);
}
fn fogTimeNoise3(pos:vec3f,time:f32)->f32{
let phase=time*0.5;
let frame=i32(floor(phase));
let t=fract(phase);
let weight=t*t*(3.0-2.0*t);
return mix(fogNoise3(pos,frame),fogNoise3(pos,frame+1),weight);
}
fn fogDensityAt(worldPosition:vec3f)->f32{
if(fogBaseExtinction()==0.0||worldPosition.y<0.05-0.2*1.45||worldPosition.y>1.65+1.45){
return 0.0;
}
let ellipse=length((worldPosition.xz-vec2f(0.0,13.0))/vec2f(70.0,12.0));
if(ellipse>=1.0){return 0.0;}
let topPos=vec3f(worldPosition.x,1.65,worldPosition.z)*0.65;
let coarse=fogTimeNoise3(topPos,cloudTime.x);
let fine=fogTimeNoise3(worldPosition*0.65*2.19+vec3f(5.12,9.73,3.84),cloudTime.x);
let top=1.65+1.45*(2.0*coarse-1.0);
let bottom=0.05+0.2*1.45*(2.0*fine-1.0);
let upper=1.0-smoothstep(top-0.75,top,worldPosition.y);
let mask=smoothstep(bottom,bottom+0.75,worldPosition.y)*upper*upper*(1.0-smoothstep(1.0-0.35,1.0,ellipse));
let variation=mix(1.0,2.0*smoothstep(0.35,0.72,0.65*coarse+0.35*fine),1.0);
return fogBaseDensity(worldPosition)*mask*variation;
}`;

export function createScene316CloudDensitySource(state: Scene316CloudState): VolumetricFogDensitySource {
    return {
        shader: cloudDensityWgsl,
        uniformFloatCount: 4,
        validate() {
            if (!Number.isFinite(state.time)) {
                throw new RangeError("Scene 316: cloud time must be finite.");
            }
        },
        writeUniforms(data) {
            data[0] = state.time;
            data[1] = data[2] = data[3] = 0;
        },
    };
}
