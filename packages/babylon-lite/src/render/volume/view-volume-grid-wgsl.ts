import type { ViewVolumeDepthMapping } from "./view-volume-grid.js";
import { wgsl, type WgslSource } from "../../shader/wgsl.js";

export function buildViewVolumeDepthWgsl(depthMapping: ViewVolumeDepthMapping): WgslSource {
    const depthToSlice = depthToSliceExpression(depthMapping);
    const sliceToDepth = sliceToDepthExpression(depthMapping);
    return wgsl`fn viewVolumeDepthToSlice(viewDepth:f32,nearDepth:f32,farDepth:f32,sliceCount:f32)->f32{
let normalizedDepth=clamp((viewDepth-nearDepth)/(farDepth-nearDepth),0.0,1.0);
return ${depthToSlice};
}
fn viewVolumeSliceToDepth(sliceCoordinate:f32,nearDepth:f32,farDepth:f32,sliceCount:f32)->f32{
let normalizedSlice=clamp(sliceCoordinate/sliceCount,0.0,1.0);
return ${sliceToDepth};
}
fn viewVolumeSliceThickness(sliceIndex:u32,nearDepth:f32,farDepth:f32,sliceCount:u32)->f32{
let count=f32(sliceCount);
let z0=viewVolumeSliceToDepth(f32(sliceIndex),nearDepth,farDepth,count);
let z1=viewVolumeSliceToDepth(f32(sliceIndex+1u),nearDepth,farDepth,count);
return max(z1-z0,0.0);
}`;
}

function depthToSliceExpression(mapping: ViewVolumeDepthMapping): string {
    switch (mapping.kind) {
        case "linear":
            return "normalizedDepth*sliceCount";
        case "log":
            return "clamp(log(clamp(viewDepth,nearDepth,farDepth)/nearDepth)/log(farDepth/nearDepth),0.0,1.0)*sliceCount";
        case "power":
            return `pow(normalizedDepth,${wgslFloat(1 / mapping.exponent)})*sliceCount`;
    }
}

function sliceToDepthExpression(mapping: ViewVolumeDepthMapping): string {
    switch (mapping.kind) {
        case "linear":
            return "nearDepth+(farDepth-nearDepth)*normalizedSlice";
        case "log":
            return "nearDepth*pow(farDepth/nearDepth,normalizedSlice)";
        case "power":
            return `nearDepth+(farDepth-nearDepth)*pow(normalizedSlice,${wgslFloat(mapping.exponent)})`;
    }
}

function wgslFloat(value: number): string {
    if (!Number.isFinite(value)) {
        throw new RangeError(`ViewVolumeGrid WGSL: expected a finite value, got ${value}.`);
    }
    const text = String(value);
    return text.includes(".") || text.includes("e") ? text : `${text}.0`;
}
