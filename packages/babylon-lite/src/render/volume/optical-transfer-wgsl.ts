import { wgsl } from "../../shader/wgsl.js";

export const OPTICAL_TRANSFER_WGSL = wgsl`struct OpticalTransfer{radiance:vec3f,transmittance:vec3f}
fn alphaToOpticalDepth(alpha:f32)->f32{return -log(max(1.0-clamp(alpha,0.0,1.0),1e-6));}
fn transmittanceFromOpticalDepth(opticalDepth:vec3f)->vec3f{return exp(-max(opticalDepth,vec3f(0.0)));}
fn composeOpticalTransfer(front:OpticalTransfer,back:OpticalTransfer)->OpticalTransfer{
return OpticalTransfer(front.radiance+front.transmittance*back.radiance,front.transmittance*back.transmittance);
}
fn integrateHomogeneousMedium(source:vec3f,extinction:vec3f,distance:f32)->OpticalTransfer{
let sigma=max(extinction,vec3f(0.0));
let travel=max(distance,0.0);
let opticalDepth=sigma*travel;
let transmittance=exp(-opticalDepth);
let series=vec3f(1.0)-opticalDepth*0.5+opticalDepth*opticalDepth*(vec3f(1.0/6.0)-opticalDepth/24.0);
let ratio=(vec3f(1.0)-transmittance)/max(opticalDepth,vec3f(0.05));
let integralFactor=select(ratio,series,opticalDepth<vec3f(0.05));
return OpticalTransfer(source*travel*integralFactor,transmittance);
}`;
