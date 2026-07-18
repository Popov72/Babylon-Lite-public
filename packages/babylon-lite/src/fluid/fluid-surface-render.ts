// Screen-space fluid surface renderer — a faithful port of Babylon.js's
// FluidRenderer (packages/dev/core/src/Rendering/fluidRenderer) shaders, so the
// visuals match BJS. All demo-local, encoded as one frame-graph task that also
// presents the frame to the swapchain. The scene is rendered to an offscreen
// colour target (`bgRT`) so this pass can sample it for refraction.
//
// References:
//   • Screen-space fluid rendering (depth → bilateral blur → normal reconstruction → refraction):
//     Green 2010, "Screen Space Fluid Rendering for Games" (GDC) —
//     https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf
//     (technique from van der Laan et al. 2009, "Screen Space Fluid Rendering with Curvature Flow").
//   • Narrow-Range Filter (alternate depth smoother, `NARROW_RANGE_WGSL`): Truong & Yuksel 2018,
//     "A Narrow-Range Filter for Screen-Space Fluid Rendering" (i3D) —
//     https://ttnghia.github.io/pdf/NarrowRangeFilter.pdf
//
// Pipeline:
//
//   1. depth     : particle sphere-impostors write the nearest eye-space Z (and
//                  speed) into an RG32F target, with fragDepth = clip.z/clip.w so
//                  the z-buffer keeps the nearest sphere surface; depth-tested
//                  against the shared scene depth so ground/paddle occlude it.
//   2. thickness : impostors again, additively (particleAlpha*sqrt(1-r²)) into an
//                  RGBA16F target — the fluid-column thickness.
//   3. blur      : bilateral (adaptive, depth-weighted) blur of the depth, and a
//                  standard gaussian blur of the thickness — both separable.
//   4. composite : reconstruct view position (inverse projection) + normal,
//                  refract the scene background (refract()), absorb via
//                  Beer-Lambert, Fresnel-mix with an environment reflection, add
//                  specular. Writes the swapchain.
//
// A "blit" mode presents the scene unchanged (sphere-impostor renderer active),
// and a `debug` mode visualises the intermediate textures (like the BJS demo's
// Debug → Feature dropdown).

import { getViewMatrix, getProjectionMatrix } from "../camera/camera.js";
import type { Camera } from "../camera/camera.js";
import type { EngineContext } from "../engine/engine.js";
import type { RenderTarget } from "../engine/render-target.js";
import type { SceneContext } from "../scene/scene-core.js";
import type { Task } from "../frame-graph/task.js";
import { mat4Invert } from "../math/mat4-invert.js";
import type { FluidSim, FluidProfiler } from "./sim-common.js";

// Opt-in GPU timing hook (see FluidProfiler / lab gpu-profiler.ts). Module-scoped:
// null by default so `profiler?.pass(...)` is undefined and timing costs nothing.
let profiler: FluidProfiler | null = null;

/** A cube map (view + sampler) sampled by the fluid surface for env reflections. */
export interface EnvMap {
    view: GPUTextureView;
    sampler: GPUSampler;
}

/** A 1×1 sky-blue cube used until the real environment finishes loading. */
function createPlaceholderCube(device: GPUDevice): GPUTextureView {
    const tex = device.createTexture({
        label: "env-placeholder",
        size: [1, 1, 6],
        format: "rgba8unorm",
        dimension: "2d",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const px = new Uint8Array([110, 150, 210, 255]);
    for (let i = 0; i < 6; i++) {
        device.queue.writeTexture({ texture: tex, origin: [0, 0, i] }, px, { bytesPerRow: 4 }, [1, 1, 1]);
    }
    return tex.createView({ dimension: "cube" });
}

export type FluidDebug = "none" | "depth" | "depthBlur" | "thickness" | "thicknessBlur" | "normals";

export interface FluidSurfaceOptions {
    /** Offscreen scene colour (background) — sampled for refraction / blit. */
    bgRT: RenderTarget;
    /** Final output (the swapchain, typically `engine.scRT`). */
    outRT: RenderTarget;
    /** Scene depth (opaque geometry) — sampled in the composite so opaque objects
     *  like the paddle correctly occlude the fluid surface behind them. */
    depthRT: RenderTarget;
    camera: Camera;
    sim: FluidSim;
}

// ── BJS-equivalent tunables (fluidRenderingTargetRenderer defaults) ──
const DENSITY = 1.0;
const REFRACTION_STRENGTH = 0.1;
const FRESNEL_CLAMP = 1.0;
const SPECULAR_POWER = 250.0;
const MINIMUM_THICKNESS = 0;
const PARTICLE_THICKNESS_ALPHA = 0.05;
const FLUID_COLOR: [number, number, number] = [0.085, 0.6375, 0.765];
const DIR_LIGHT: [number, number, number] = [-2, -1, 1]; // normalized below
const BLUR_DEPTH_FILTER_SIZE = 20;
const BLUR_MAX_FILTER_SIZE = 64;
const BLUR_DEPTH_DEPTH_SCALE = 10;
const BLUR_THICKNESS_FILTER_SIZE = 10;
const PARTICLE_SIZE_SCALE = 3.5; // impostor diameter = particleRadius * this

// ── Depth + thickness particle passes (sphere impostors) ──
const PARTICLE_WGSL = /* wgsl */ `
struct Cam {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    misc: vec4<f32>,   // x = size (diameter), y = sphereRadius, z = speedScale, w = particleAlpha
};
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dbg: array<f32>;
// Scene opaque depth (reverse-Z). Sampled per-fragment so fluid hidden BEHIND opaque
// geometry (ground / paddle / wheel) is discarded from BOTH the depth and thickness
// passes. Without this the additive thickness integral counts occluded particles and the
// surface bleeds through solids.
@group(0) @binding(3) var sceneDepthTex: texture_depth_2d;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) viewPos: vec3<f32>,
    @location(2) @interpolate(flat) speed: f32,
    @location(3) ndc: vec4<f32>,
};

// True when the sphere-surface point (eye-space Z fragEyeZ, positive = away from the
// camera) lies BEHIND the nearest opaque scene surface at this fragment. The clip-space
// ndc varying reconstructs the screen UV (clip.xy/clip.w is screen-linear, so this is
// perspective-correct and resolution-independent — the fluid passes may run at half res
// while the scene depth is full res). Mirrors the composite's occlusion test.
fn occludedByScene(fragEyeZ: f32, ndc: vec4<f32>) -> bool {
    let uv = ndc.xy / ndc.w;
    let screenUV = vec2<f32>(uv.x * 0.5 + 0.5, 0.5 - uv.y * 0.5);
    let dims = vec2<f32>(textureDimensions(sceneDepthTex));
    let coord = vec2<i32>(clamp(screenUV, vec2<f32>(0.0), vec2<f32>(1.0)) * dims);
    let sceneNdc = textureLoad(sceneDepthTex, coord, 0);
    if (sceneNdc <= 0.0) { return false; } // no opaque geometry here (reverse-Z far = 0)
    let sceneEye = cam.proj[3].z / (sceneNdc - cam.proj[2].z);
    return fragEyeZ > sceneEye + 0.02;
}

fn corner(vi: u32) -> vec2<f32> {
    // offset in [0,1]; matches BJS 'offset' attribute (quad corners).
    var c = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
    return c[vi];
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let offset = corner(vi);
    let cornerPos = vec3<f32>((offset - vec2<f32>(0.5)) * cam.misc.x, 0.0);
    let viewPos = (cam.view * vec4<f32>(positions[ii].xyz, 1.0)).xyz;
    var o: VOut;
    o.clip = cam.proj * vec4<f32>(viewPos + cornerPos, 1.0);
    o.uv = offset;
    o.viewPos = viewPos;
    o.speed = dbg[ii];
    o.ndc = o.clip;
    return o;
}

struct DepthOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
};

@fragment fn fsDepth(i: VOut) -> DepthOut {
    let nxy = i.uv * 2.0 - 1.0;
    let r2 = dot(nxy, nxy);
    if (r2 > 1.0) { discard; }
    // LH: front-facing sphere normal points toward camera (negative view z).
    let normal = vec3<f32>(nxy, -sqrt(1.0 - r2));
    let realViewPos = i.viewPos + normal * cam.misc.y;
    // Drop the fragment if this sphere point is behind opaque scene geometry, so the
    // recorded nearest surface is the nearest VISIBLE water (not water behind a solid).
    if (occludedByScene(realViewPos.z, i.ndc)) { discard; }
    let clipPos = cam.proj * vec4<f32>(realViewPos, 1.0);
    var o: DepthOut;
    o.depth = clipPos.z / clipPos.w;          // reverse-Z, tested greater-equal
    o.color = vec4<f32>(realViewPos.z, i.speed, 0.0, 1.0); // eye-space Z + speed
    return o;
}

@fragment fn fsThick(i: VOut) -> @location(0) vec4<f32> {
    let nxy = i.uv * 2.0 - 1.0;
    let r2 = dot(nxy, nxy);
    if (r2 > 1.0) { discard; }
    // Discard particles hidden behind opaque geometry so the additive thickness integral
    // stops at the solid surface (no bleed-through, no over-thick columns from occluded water).
    let normal = vec3<f32>(nxy, -sqrt(1.0 - r2));
    let realViewPos = i.viewPos + normal * cam.misc.y;
    if (occludedByScene(realViewPos.z, i.ndc)) { discard; }
    let thickness = sqrt(1.0 - r2);
    return vec4<f32>(vec3<f32>(cam.misc.w * thickness), 1.0);
}`;

const FULLSCREEN_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}`;

// Bilateral blur of the depth (R) + speed (G), ported from
// fluidRenderingBilateralBlur. Adaptive filter size + depth-weighted. Uses
// textureLoad (integer texel fetch) for exact, sampler-independent reads.
const BILATERAL_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Blur { p: vec4<f32>, q: vec4<f32> }; // p: stepX, stepY, projConst, depthThreshold; q: maxFilterSize, _, _, _
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> b: Blur;

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(floor(pos.xy));
    let dim = vec2<i32>(textureDimensions(src));
    let depth = textureLoad(src, pix, 0).x;
    if (depth >= 1e6 || depth <= 0.0) { return vec4<f32>(depth, depth, 0.0, 1.0); }
    let filterSize = min(i32(b.q.x), i32(ceil(b.p.z / depth)));
    let sigma = f32(filterSize) / 3.0;
    let twoSigma2 = 2.0 * sigma * sigma;
    let sigmaDepth = b.p.w / 3.0;
    let twoSigmaDepth2 = 2.0 * sigmaDepth * sigmaDepth;
    let step = vec2<i32>(i32(b.p.x), i32(b.p.y));
    var sum = 0.0;
    var sumVel = 0.0;
    var wsum = 0.0;
    for (var x = -filterSize; x <= filterSize; x = x + 1) {
        let c = pix + step * x;
        if (any(c < vec2<i32>(0)) || any(c >= dim)) { continue; }
        let s = textureLoad(src, c, 0).rg;
        if (s.r >= 1e6 || s.r <= 0.0) { continue; }
        let w = exp(-f32(x * x) / twoSigma2);
        let rDepth = s.r - depth;
        let wd = exp(-rDepth * rDepth / twoSigmaDepth2);
        sum = sum + s.r * w * wd;
        sumVel = sumVel + s.g * w * wd;
        wsum = wsum + w * wd;
    }
    if (wsum <= 0.0) { return vec4<f32>(depth, 0.0, 0.0, 1.0); }
    return vec4<f32>(sum / wsum, sumVel / wsum, 0.0, 1.0);
}`;

// Narrow-Range Filter (Truong & Yuksel, i3D 2018) — an alternate screen-space
// depth smoother. Filters the eye-space Z (R) + speed (G). One shader serves BOTH
// the two 1D separable passes (q.z = 0, step = p.xy) and the fixed 5×5 2D clean-up
// pass (q.z = 1) that hides the axis-aligned streaks the 1D passes leave near
// discontinuities. Implements the clamp function (Eq 2), far-cutoff + bias
// correction across the centre (Eqs 3, 6), adaptive kernel (Eq 5, same
// projConst/depth machinery as the bilateral) and the sequential dynamic-range
// expansion from the nearest neighbour outward (Eqs 7–9).
const NARROW_RANGE_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Blur { p: vec4<f32>, q: vec4<f32> }; // p: stepX, stepY, projConst, delta; q: maxFilterSize, mu, cleanup2D, _
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> b: Blur;

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(floor(pos.xy));
    let dim = vec2<i32>(textureDimensions(src));
    let c0 = textureLoad(src, pix, 0).rg;
    let zi = c0.x;
    if (zi >= 1e6 || zi <= 0.0) { return vec4<f32>(zi, zi, 0.0, 1.0); }
    let delta = b.p.w;
    let mu = b.q.y;
    // Centre pixel contributes with weight 1 (f(zi, zi) = zi).
    var sum = zi;
    var sumVel = c0.y;
    var wsum = 1.0;
    if (b.q.z < 0.5) {
        // ── 1D separable narrow-range pass (Eqs 2,3,4,6,7,8,9) ──
        let filterSize = min(i32(b.q.x), i32(ceil(b.p.z / zi)));
        let sigma = max(f32(filterSize) / 3.0, 1e-4);
        let twoSigma2 = 2.0 * sigma * sigma;
        let step = vec2<i32>(i32(b.p.x), i32(b.p.y));
        var dLow = delta;   // per-pixel dynamic range (Eq 7), grows outward
        var dHigh = delta;
        // Iterate the pair (+x, -x) from the CLOSEST neighbour outward so the
        // dynamic range expands sequentially along the surface.
        for (var x = 1; x <= filterSize; x = x + 1) {
            let cj = pix + step * x;
            let ck = pix - step * x;
            let jIn = all(cj >= vec2<i32>(0)) && all(cj < dim);
            let kIn = all(ck >= vec2<i32>(0)) && all(ck < dim);
            let sj = select(vec2<f32>(1e6, 0.0), textureLoad(src, cj, 0).rg, jIn);
            let sk = select(vec2<f32>(1e6, 0.0), textureLoad(src, ck, 0).rg, kIn);
            let zj = sj.x;
            let zk = sk.x;
            let jFluid = zj < 1e6 && zj > 0.0;
            let kFluid = zk < 1e6 && zk > 0.0;
            // Far cutoff (Eq 3) + bias correction (Eq 6): reject the WHOLE pair if
            // either the sample OR its mirror is background or beyond zi + dHigh —
            // keeps the kernel symmetric so background surfaces don't bend it.
            let reject = (!jFluid) || (!kFluid) || (zj > zi + dHigh) || (zk > zi + dHigh);
            let w = select(exp(-f32(x * x) / twoSigma2), 0.0, reject);
            // Clamp function (Eq 2): front outliers (zj < zi - dLow) clamp to zi - mu.
            let valj = select(zi - mu, zj, zj >= zi - dLow);
            let valk = select(zi - mu, zk, zk >= zi - dLow);
            sum = sum + (valj + valk) * w;
            sumVel = sumVel + (sj.y + sk.y) * w;
            wsum = wsum + 2.0 * w;
            // Dynamic-range expansion (Eqs 8,9) for in-range neighbours.
            if (jFluid && zj >= zi - dLow && zj <= zi + dHigh) {
                dLow = max(dLow, zi - zj + delta);
                dHigh = max(dHigh, zj - zi + delta);
            }
            if (kFluid && zk >= zi - dLow && zk <= zi + dHigh) {
                dLow = max(dLow, zi - zk + delta);
                dHigh = max(dHigh, zk - zi + delta);
            }
        }
    } else {
        // ── Fixed 5×5 2D clean-up pass (Sec 3.4): same clamp/weight/bias rules on a
        // tiny static-range kernel to erase the 1D passes' axis-aligned streaks. ──
        let sigma = 2.0;
        let twoSigma2 = 2.0 * sigma * sigma;
        for (var dy = -2; dy <= 2; dy = dy + 1) {
            for (var dx = -2; dx <= 2; dx = dx + 1) {
                if (dx == 0 && dy == 0) { continue; }
                let off = vec2<i32>(dx, dy);
                let cj = pix + off;
                let ck = pix - off;
                let jIn = all(cj >= vec2<i32>(0)) && all(cj < dim);
                let kIn = all(ck >= vec2<i32>(0)) && all(ck < dim);
                let sj = select(vec2<f32>(1e6, 0.0), textureLoad(src, cj, 0).rg, jIn);
                let sk = select(vec2<f32>(1e6, 0.0), textureLoad(src, ck, 0).rg, kIn);
                let zj = sj.x;
                let zk = sk.x;
                let jFluid = zj < 1e6 && zj > 0.0;
                let kFluid = zk < 1e6 && zk > 0.0;
                let reject = (!jFluid) || (!kFluid) || (zj > zi + delta) || (zk > zi + delta);
                let d2 = f32(dx * dx + dy * dy);
                let w = select(exp(-d2 / twoSigma2), 0.0, reject);
                let valj = select(zi - mu, zj, zj >= zi - delta);
                sum = sum + valj * w;
                sumVel = sumVel + sj.y * w;
                wsum = wsum + w;
            }
        }
    }
    return vec4<f32>(sum / wsum, sumVel / wsum, 0.0, 1.0);
}`;

// Standard gaussian blur of the thickness, ported from fluidRenderingStandardBlur.
const STANDARD_BLUR_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Blur { p: vec4<f32>, q: vec4<f32> }; // p: stepX, stepY, filterSize, _
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> b: Blur;

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(floor(pos.xy));
    let dim = vec2<i32>(textureDimensions(src));
    let s0 = textureLoad(src, pix, 0);
    if (s0.r == 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let filterSize = i32(b.p.z);
    let sigma = b.p.z / 3.0;
    let twoSigma2 = 2.0 * sigma * sigma;
    let step = vec2<i32>(i32(b.p.x), i32(b.p.y));
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var x = -filterSize; x <= filterSize; x = x + 1) {
        let c = pix + step * x;
        if (any(c < vec2<i32>(0)) || any(c >= dim)) { continue; }
        let s = textureLoad(src, c, 0);
        let w = exp(-f32(x * x) / twoSigma2);
        sum = sum + s * w;
        wsum = wsum + w;
    }
    return vec4<f32>(sum.rgb / wsum, 1.0);
}`;

// Plain copy of the background to the swapchain (sphere-impostor mode).
const BLIT_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
@group(0) @binding(0) var bg: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(bg));
    return textureSampleLevel(bg, samp, pos.xy / dim, 0.0);
}`;

// Composite — faithful port of fluidRenderingRender.fragment (water shading:
// refraction, Beer-Lambert, Fresnel + environment reflection, specular).
// Also implements the debug-texture visualisations.
const COMPOSITE_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
const IOR: f32 = 1.333;
const ETA: f32 = 1.0 / 1.333;
const F0: f32 = 0.02;
// Env reflections sample the LINEAR-HDR specular cube; tonemap it with the SAME
// transform babylon-lite's HDR skybox uses (exposure -> gamma -> clamp ->
// smoothstep contrast, no Reinhard), so the fluid reflection matches the sky it
// reflects. Keep ENV_EXPOSURE / ENV_CONTRAST in sync with the demo's
// scene.imageProcessing values.
const ENV_EXPOSURE: f32 = 1.0;
const ENV_CONTRAST: f32 = 1.1;

struct Comp {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    invProj: mat4x4<f32>,
    camR: vec4<f32>,    // camera world basis (view→world for reflection)
    camU: vec4<f32>,
    camF: vec4<f32>,
    a: vec4<f32>,       // outputTexel.xy (full res, for texCoord), cameraFar, density
    b: vec4<f32>,       // dirLight.xyz, refractionStrength
    c: vec4<f32>,       // fresnelClamp, specularPower, minimumThickness, debugMode
    diffuse: vec4<f32>, // diffuseColor.rgb, _
    extra: vec4<f32>,   // depthTexel.xy (for normal offsets), _, _
};
@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var depthSamp: sampler;
@group(0) @binding(2) var thickTex: texture_2d<f32>;
@group(0) @binding(3) var thickSamp: sampler;
@group(0) @binding(4) var bgTex: texture_2d<f32>;
@group(0) @binding(5) var bgSamp: sampler;
@group(0) @binding(6) var envTex: texture_cube<f32>;
@group(0) @binding(7) var envSamp: sampler;
@group(0) @binding(8) var depthRawTex: texture_2d<f32>;
@group(0) @binding(9) var thickRawTex: texture_2d<f32>;
@group(0) @binding(10) var<uniform> u: Comp;
@group(0) @binding(11) var sceneDepthTex: texture_depth_2d;

fn computeViewPosFromUVDepth(texCoord: vec2<f32>, depth: f32) -> vec3<f32> {
    // Direct perspective unproject (equivalent to invProj * ndc for a standard
    // perspective): eye z = depth, and x/y from the screen position scaled by
    // depth * tan(fov/2). texCoord is y-down, view Y is up → flip y.
    let tanHalfFov = u.camR.w;
    let aspect = u.camU.w;
    let ndcX = texCoord.x * 2.0 - 1.0;
    let ndcY = 1.0 - texCoord.y * 2.0;
    return vec3<f32>(ndcX * depth * tanHalfFov * aspect, ndcY * depth * tanHalfFov, depth);
}
fn getViewPos(texCoord: vec2<f32>) -> vec3<f32> {
    let d = textureSampleLevel(depthTex, depthSamp, texCoord, 0.0).x;
    return computeViewPosFromUVDepth(texCoord, d);
}

// High-contrast eye-depth visualisation for the debug views. A raw depth/cameraFar map is
// nearly useless: the fluid's depth spans only a few percent of cameraFar, so it shows as a flat dark grey where neither the surface shape NOR the effect of the depth
// blur is visible. Iso-depth contour bands (a triangle wave) remove that large DC offset,
// so the fine surface structure appears — raw depth reads as bumpy/wobbly bands, the
// blurred depth as smooth parallel bands, making the "Surface depth blur" effect obvious.
// The band frequency scales with cameraFar so density is roughly scene-independent.
fn depthViz(d: f32, cameraFar: f32) -> vec3<f32> {
    if (d >= 1e6 || d <= 0.0) { return vec3<f32>(1.0); } // background / no water = white
    let tri = abs(fract(d / cameraFar * 512.0) * 2.0 - 1.0);
    return vec3<f32>(tri);
}

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let outputTexel = u.a.xy;       // full-res texel (for texCoord)
    let depthTexel = u.extra.xy;    // depth-texture texel (for normal offsets)
    let texCoord = pos.xy * outputTexel;
    let cameraFar = u.a.z;
    let density = u.a.w;
    let debugMode = u.c.w;

    let depthVel = textureSampleLevel(depthTex, depthSamp, texCoord, 0.0).rg;
    let depth = depthVel.r;
    let thickness = textureSampleLevel(thickTex, thickSamp, texCoord, 0.0).x;
    let backColor = textureSampleLevel(bgTex, bgSamp, texCoord, 0.0);

    // ── Debug visualisations (mirror the BJS Debug→Feature dropdown) ──
    if (debugMode > 0.5) {
        if (debugMode < 1.5) {        // depth (raw)
            let v = textureSampleLevel(depthRawTex, depthSamp, texCoord, 0.0).r;
            return vec4<f32>(depthViz(v, cameraFar), 1.0);
        } else if (debugMode < 2.5) { // depth blurred
            return vec4<f32>(depthViz(depth, cameraFar), 1.0);
        } else if (debugMode < 3.5) { // thickness (raw)
            let t = textureSampleLevel(thickRawTex, thickSamp, texCoord, 0.0).r;
            return vec4<f32>(vec3<f32>(t), 1.0);
        } else if (debugMode < 4.5) { // thickness blurred
            return vec4<f32>(vec3<f32>(thickness), 1.0);
        }
        // else mode 5: normals — fall through after computing the normal below.
    }

    if (depth >= cameraFar || depth <= 0.0 || thickness <= u.c.z) {
        return backColor;
    }

    // Occlusion against opaque scene geometry (e.g. the paddle): convert the scene
    // depth-buffer value (reverse-Z) to eye depth and, if the fluid surface is
    // behind it, show the background (which already contains that geometry).
    let sceneNdc = textureLoad(sceneDepthTex, vec2<i32>(floor(pos.xy)), 0);
    let sceneEyeDepth = u.proj[3].z / (sceneNdc - u.proj[2].z);
    if (sceneNdc > 0.0 && depth > sceneEyeDepth + 0.02) {
        return backColor;
    }

    // View-space position + normal (min-Z one-sided differences, in depth texels).
    let viewPos = computeViewPosFromUVDepth(texCoord, depth);
    // Reconstruct the surface normal on the HALF-RES depth grid, snapped to texel CENTRES.
    // Sampling the half-res depth (NEAREST) with the full-res texCoord makes the ±depthTexel
    // neighbours land EXACTLY on a texel boundary for the centre row whenever the canvas
    // height H ≡ 3 (mod 4): there (floor(H/2)+0.5)/H * ceil(H/2) is an exact integer, so all
    // three stencil taps sit on texel boundaries and their NEAREST snap is float-ambiguous →
    // an inconsistent single-row stencil → a tilted normal on that one row (the "horizontal
    // garbage line", a bright specular streak in the final image; other H land mid-texel and
    // are fine). Snapping texCoord to the texel centre first makes every ±1-texel neighbour
    // fall squarely on an adjacent centre, so the stencil is stable at ANY canvas size.
    let depthDim = vec2<f32>(1.0) / depthTexel;
    let nTC = (floor(texCoord * depthDim) + vec2<f32>(0.5)) * depthTexel;
    let viewPosN = getViewPos(nTC);
    var ddx = getViewPos(nTC + vec2<f32>(depthTexel.x, 0.0)) - viewPosN;
    var ddy = getViewPos(nTC + vec2<f32>(0.0, depthTexel.y)) - viewPosN;
    let ddx2 = viewPosN - getViewPos(nTC + vec2<f32>(-depthTexel.x, 0.0));
    if (abs(ddx.z) > abs(ddx2.z)) { ddx = ddx2; }
    let ddy2 = viewPosN - getViewPos(nTC + vec2<f32>(0.0, -depthTexel.y));
    if (abs(ddy.z) > abs(ddy2.z)) { ddy = ddy2; }
    // Guard against a degenerate cross product (fast/noisy depth under a force
    // can make ddx∥ddy → normalize(0) = NaN → dark specular/fresnel artefacts).
    // Deterministic winding gives a camera-facing normal in our LH view space
    // (matches BJS, which has no conditional flip). A normal.z > 0 orientation
    // test is WRONG at grazing/off-axis angles: the view ray then has large x/y
    // components, so a correctly camera-facing normal can legitimately have z > 0
    // and would be flipped, inverting the whole top surface (purple speckle).
    let cl = cross(ddx, ddy);
    let clLen = length(cl);
    // NaN-SAFE normalize: select() evaluates BOTH operands, so cl/clLen (= 0/0 = NaN
    // when the cross product degenerates, ddx∥ddy) is always computed and can LEAK through
    // select on some drivers (e.g. NVIDIA) → a white NaN normal on the degenerate row (the
    // "horizontal line", white in the final render / dark in the normals view). An if()
    // never evaluates the division unless the cross is well-conditioned.
    var normal = vec3<f32>(0.0, 0.0, -1.0);
    if (clLen > 1e-6) { normal = cl / clLen; }
    // Orient the reconstructed normal toward the camera. cross(ddx,ddy)'s winding can FLIP
    // across the surface at grazing view angles (half-res depth + wave slope), leaving
    // patches whose normal points AWAY from the eye (magenta in the normals view); the flip
    // boundary reads as a hard bright/dark LINE (most visible around mid beta). Use the
    // VIEW-RAY test dot(n, rayDir) > 0 — NOT the naive normal.z > 0 test, which mis-fires
    // when the view ray has large x/y components at grazing/off-axis angles.
    let rayDir = normalize(viewPos); // camera → surface
    if (dot(normal, rayDir) > 0.0) { normal = -normal; }

    if (debugMode > 4.5) { // normals
        return vec4<f32>(normal * 0.5 + 0.5, 1.0);
    }

    let diffuseColor = u.diffuse.rgb;
    let lightDir = normalize((u.view * vec4<f32>(-u.b.xyz, 0.0)).xyz);
    let H = normalize(lightDir - rayDir);
    let specular = pow(max(0.0, dot(H, normal)), u.c.y);

    // Refraction of the scene background. refract() returns 0 on total internal
    // reflection — fall back to the straight-through ray so no dark hole appears.
    var refractionDir = refract(rayDir, normal, ETA);
    if (dot(refractionDir, refractionDir) < 1e-6) { refractionDir = rayDir; }
    let refrUV = texCoord + vec2<f32>(refractionDir.x, -refractionDir.y) * thickness * u.b.w;
    let transmitted = textureSampleLevel(bgTex, bgSamp, clamp(refrUV, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
    let transmittance = exp(-density * thickness * (1.0 - diffuseColor)); // Beer-Lambert
    let refractionColor = transmitted * transmittance;

    // Environment reflection (transform the view-space reflected ray to world).
    let reflViewDir = reflect(rayDir, normal);
    let reflW = reflViewDir.x * u.camR.xyz + reflViewDir.y * u.camU.xyz + reflViewDir.z * u.camF.xyz;
    let reflLin = textureSampleLevel(envTex, envSamp, vec3<f32>(reflW.x, reflW.y, -reflW.z), 0.0).rgb;
    var reflC = reflLin * ENV_EXPOSURE;
    reflC = pow(reflC, vec3<f32>(1.0 / 2.2));
    reflC = clamp(reflC, vec3<f32>(0.0), vec3<f32>(1.0));
    let reflHi = reflC * reflC * (3.0 - 2.0 * reflC); // smoothstep contrast (matches skybox)
    reflC = mix(reflC, reflHi, ENV_CONTRAST - 1.0);
    let reflectionColor = max(reflC, vec3<f32>(0.0));

    let fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - max(dot(normal, -rayDir), 0.0), 5.0), 0.0, u.c.x);
    var finalColor = mix(refractionColor, reflectionColor, fresnel) + specular;

    return vec4<f32>(finalColor, 1.0);
}`;

export function createFluidSurfaceTask(
    engine: EngineContext,
    scene: SceneContext,
    opts: FluidSurfaceOptions
): Task & {
    setSim(s: FluidSim): void;
    setMode(m: "surface" | "blit"): void;
    setEnvMap(e: EnvMap): void;
    setDebug(d: FluidDebug): void;
    setFluidColor(rgb: [number, number, number]): void;
    setAbsorption(v: number): void;
    setHalfRender(on: boolean): void;
    setThicknessDownscale(factor: number): void;
    setSizeScale(s: number): void;
    setRefractionStrength(v: number): void;
    setSpecularPower(v: number): void;
    setDirLight(dir: [number, number, number]): void;
    /** Bilateral depth-blur controls: filter size (surface smoothing amount, drives
     *  the projection const + max-filter clamp) and the depth-threshold scale
     *  (edge preservation). */
    setDepthBlur(filterSize: number, depthScale: number): void;
    /** Select the screen-space depth smoother: the default separable bilateral
     *  filter, or the Narrow-Range Filter (Truong & Yuksel 2018) — a clamped,
     *  bias-corrected filter that preserves depth discontinuities better. */
    setSurfaceFilter(m: "bilateral" | "narrowRange"): void;
    /** Narrow-range params (multipliers of the impostor `size`): `delta` = the
     *  base accepted depth range (δ, edge preservation) and `mu` = the front-clamp
     *  offset (µ). Only affects the narrow-range filter. */
    setNarrowRange(delta: number, mu: number): void;
    /** Standard thickness-blur filter size (how much the thickness is smoothed). */
    setThicknessBlur(filterSize: number): void;
    /** Blurred fluid-surface eye-space Z (RG32F, .r = view-space Z, 1e6 = far/no
     *  water). Consumed by the screen-space foam renderer for surface occlusion
     *  and surface/submerged classification. Null before the first allocation.
     *  The view is reallocated on resize / half-res change — call per frame, don't
     *  cache. */
    surfaceDepthView(): GPUTextureView | null;
    /** Opt-in GPU timing hook: tag every surface pass with timestampWrites, or null
     *  to turn timing off. The profiler machinery lives in the app (lab). */
    setProfiler(p: FluidProfiler | null): void;
} {
    const device = engine._device;
    const { bgRT, outRT, depthRT, camera } = opts;
    let currentSim = opts.sim;
    let mode: "surface" | "blit" = "surface";
    let debug: FluidDebug = "none";
    let fluidColor: [number, number, number] = [...FLUID_COLOR];
    let absorption = DENSITY; // Beer-Lambert absorption coefficient
    let halfRender = false;
    let thicknessDownscale = 2; // thickness textures: size = canvas / this factor (own knob)
    let sizeScale = 1; // user-controlled visual particle-size multiplier
    let refractionStrength = REFRACTION_STRENGTH;
    let specularPower = SPECULAR_POWER;
    let dirLight: [number, number, number] = [...DIR_LIGHT];
    // Blur parameters (runtime-tunable via setDepthBlur / setThicknessBlur).
    let depthFilterSize = BLUR_DEPTH_FILTER_SIZE; // drives the bilateral projConst + max-filter clamp
    let depthScale = BLUR_DEPTH_DEPTH_SCALE; // bilateral depth-threshold scale (edge preservation)
    let thicknessFilterSize = BLUR_THICKNESS_FILTER_SIZE; // standard thickness-blur half-size
    // Narrow-range filter selection + params (multipliers of the impostor `size`).
    let surfaceFilter: "bilateral" | "narrowRange" = "bilateral";
    let nrDelta = 10; // δ / size — base accepted depth range (edge preservation)
    let nrMu = 1; // µ / size — front-clamp offset

    const camData = new Float32Array(36); // view(16) + proj(16) + misc(4)
    const camBuffer = device.createBuffer({ label: "fluid-surf-cam", size: camData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Four separate blur uniform buffers — the four blur passes run in ONE
    // command encoder, but queue.writeBuffer applies on the queue timeline BEFORE
    // the encoder executes, so a single shared buffer would give every pass the
    // last-written value. Distinct buffers keep each pass's params correct.
    const blurDepthXBuf = device.createBuffer({ label: "fluid-surf-blur-dx", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blurDepthYBuf = device.createBuffer({ label: "fluid-surf-blur-dy", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Narrow-range 5×5 clean-up pass runs in the SAME encoder as the two 1D passes,
    // so it needs its own uniform buffer (see the four-blur-buffer note above).
    const blurDepthCleanupBuf = device.createBuffer({ label: "fluid-surf-blur-dc", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blurThickXBuf = device.createBuffer({ label: "fluid-surf-blur-tx", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blurThickYBuf = device.createBuffer({ label: "fluid-surf-blur-ty", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const compBuffer = device.createBuffer({ label: "fluid-surf-comp", size: 16 * 13 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Non-filtering sampler for the RG32F depth (32-float textures aren't filterable).
    const nearestSampler = device.createSampler({ label: "fluid-surf-nearest" });
    const linearSampler = device.createSampler({ label: "fluid-surf-linear", magFilter: "linear", minFilter: "linear" });
    const envFallbackSampler = device.createSampler({ label: "fluid-surf-env", magFilter: "linear", minFilter: "linear", mipmapFilter: "linear" });
    let envView: GPUTextureView = createPlaceholderCube(device);
    let envSampler: GPUSampler = envFallbackSampler;

    let fullW = 0;
    let fullH = 0;
    let depthW = 0;
    let depthH = 0;
    let thickW = 0;
    let thickH = 0;
    let allocHalf = false;
    let allocThickDownscale = 0;
    let depthTex: GPUTexture | null = null;
    let depthTmp: GPUTexture | null = null;
    let depthBlur: GPUTexture | null = null;
    let depthNrTmp: GPUTexture | null = null; // narrow-range: 2nd scratch (Y pass → clean-up)
    let fluidDepthBuf: GPUTexture | null = null; // dedicated depth buffer for the depth pass
    let thickTex: GPUTexture | null = null;
    let thickTmp: GPUTexture | null = null;
    let thickBlur: GPUTexture | null = null;
    let views: Record<string, GPUTextureView> = {};

    function allocTargets(): void {
        const w = engine.canvas.width;
        const h = engine.canvas.height;
        if (w === fullW && h === fullH && allocHalf === halfRender && allocThickDownscale === thicknessDownscale && depthTex) {
            return;
        }
        for (const t of [depthTex, depthTmp, depthBlur, depthNrTmp, fluidDepthBuf, thickTex, thickTmp, thickBlur]) {
            t?.destroy();
        }
        fullW = w;
        fullH = h;
        allocHalf = halfRender;
        allocThickDownscale = thicknessDownscale;
        // Depth: full res, or half when "half rendering" is on.
        depthW = halfRender ? Math.max(1, Math.ceil(w / 2)) : w;
        depthH = halfRender ? Math.max(1, Math.ceil(h / 2)) : h;
        // Thickness renders at its OWN independent downscale (size = canvas /
        // thicknessDownscale). It's low-frequency, so it tolerates far more
        // downscaling than depth — hence a separate factor (default 2 = half res).
        thickW = Math.max(1, Math.ceil(w / thicknessDownscale));
        thickH = Math.max(1, Math.ceil(h / thicknessDownscale));
        const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
        const mkDepth = (label: string): GPUTexture => device.createTexture({ label, size: { width: depthW, height: depthH }, format: "rg32float", usage });
        const mkThick = (label: string): GPUTexture => device.createTexture({ label, size: { width: thickW, height: thickH }, format: "rgba16float", usage });
        depthTex = mkDepth("fluid-surf-depth");
        depthTmp = mkDepth("fluid-surf-depthTmp");
        depthBlur = mkDepth("fluid-surf-depthBlur");
        depthNrTmp = mkDepth("fluid-surf-depthNrTmp");
        fluidDepthBuf = device.createTexture({
            label: "fluid-surf-zbuf",
            size: { width: depthW, height: depthH },
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        thickTex = mkThick("fluid-surf-thick");
        thickTmp = mkThick("fluid-surf-thickTmp");
        thickBlur = mkThick("fluid-surf-thickBlur");
        views = {
            depth: depthTex.createView(),
            depthTmp: depthTmp.createView(),
            depthBlur: depthBlur.createView(),
            depthNrTmp: depthNrTmp.createView(),
            zbuf: fluidDepthBuf.createView(),
            thick: thickTex.createView(),
            thickTmp: thickTmp.createView(),
            thickBlur: thickBlur.createView(),
        };
    }

    let depthPipe: GPURenderPipeline | null = null;
    let thickPipe: GPURenderPipeline | null = null;
    let bilateralPipe: GPURenderPipeline | null = null;
    let narrowRangePipe: GPURenderPipeline | null = null;
    let standardBlurPipe: GPURenderPipeline | null = null;
    let blitPipe: GPURenderPipeline | null = null;
    let compPipe: GPURenderPipeline | null = null;
    let particleBGL: GPUBindGroupLayout | null = null;
    let particleBG: GPUBindGroup | null = null;

    function buildParticleBG(): void {
        if (!particleBGL || !depthRT._depthView) {
            return;
        }
        particleBG = device.createBindGroup({
            label: "fluid-surf-particle",
            layout: particleBGL,
            entries: [
                { binding: 0, resource: { buffer: camBuffer } },
                { binding: 1, resource: { buffer: currentSim.positionBuffer } },
                { binding: 2, resource: { buffer: currentSim.debugBuffer } },
                { binding: 3, resource: depthRT._depthView },
            ],
        });
    }

    function blurPipeline(label: string, code: string): GPURenderPipeline {
        const module = device.createShaderModule({ label, code });
        return device.createRenderPipeline({
            label,
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: { module, entryPoint: "fs", targets: [{ format: code === STANDARD_BLUR_WGSL ? "rgba16float" : "rg32float" }] },
            primitive: { topology: "triangle-list" },
        });
    }

    function build(): void {
        if (depthPipe) {
            return;
        }
        const partMod = device.createShaderModule({ label: "fluid-surf-particle", code: PARTICLE_WGSL });
        // The depth pass owns a dedicated single-sample depth24plus buffer (so the
        // fluid surface no longer depends on / mutates the scene depth, which lets
        // it run at half resolution independently).
        const dFormat: GPUTextureFormat = "depth24plus";
        const samples = 1;
        particleBGL = device.createBindGroupLayout({
            label: "fluid-surf-particle",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
            ],
        });
        const partPL = device.createPipelineLayout({ bindGroupLayouts: [particleBGL] });
        depthPipe = device.createRenderPipeline({
            label: "fluid-surf-depth",
            layout: partPL,
            vertex: { module: partMod, entryPoint: "vs" },
            fragment: { module: partMod, entryPoint: "fsDepth", targets: [{ format: "rg32float" }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: { format: dFormat, depthWriteEnabled: true, depthCompare: "greater-equal" },
            multisample: { count: samples },
        });
        const addBlend = { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } as const;
        thickPipe = device.createRenderPipeline({
            label: "fluid-surf-thick",
            layout: partPL,
            vertex: { module: partMod, entryPoint: "vs" },
            fragment: { module: partMod, entryPoint: "fsThick", targets: [{ format: "rgba16float", blend: addBlend }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
            // No depth test: thickness is a full additive volume integral (every
            // particle along a ray contributes). The composite masks the result by
            // the depth texture, so occluded regions are still culled correctly.
            // (Sharing the scene depth here would let the depth pass — which writes
            // the nearest surface — cull all back-of-volume particles, collapsing
            // thickness to a single layer.)
            multisample: { count: samples },
        });
        bilateralPipe = blurPipeline("fluid-surf-bilateral", BILATERAL_WGSL);
        narrowRangePipe = blurPipeline("fluid-surf-narrow", NARROW_RANGE_WGSL);
        standardBlurPipe = blurPipeline("fluid-surf-standard-blur", STANDARD_BLUR_WGSL);
        const blitMod = device.createShaderModule({ label: "fluid-surf-blit", code: BLIT_WGSL });
        blitPipe = device.createRenderPipeline({
            label: "fluid-surf-blit",
            layout: "auto",
            vertex: { module: blitMod, entryPoint: "vs" },
            fragment: { module: blitMod, entryPoint: "fs", targets: [{ format: engine.format }] },
            primitive: { topology: "triangle-list" },
        });
        const compMod = device.createShaderModule({ label: "fluid-surf-comp", code: COMPOSITE_WGSL });
        compPipe = device.createRenderPipeline({
            label: "fluid-surf-comp",
            layout: "auto",
            vertex: { module: compMod, entryPoint: "vs" },
            fragment: { module: compMod, entryPoint: "fs", targets: [{ format: engine.format }] },
            primitive: { topology: "triangle-list" },
        });
        buildParticleBG();
    }

    function updateUniforms(): void {
        const view = getViewMatrix(camera);
        const proj = getProjectionMatrix(camera, engine.canvas.width / Math.max(1, engine.canvas.height));
        const invProj = mat4Invert(proj) ?? Array.from(proj);
        const radius = currentSim.particleRadius;
        const size = radius * PARTICLE_SIZE_SCALE * (currentSim.surfaceSizeScale ?? 1) * sizeScale;

        // Particle pass uniform.
        for (let k = 0; k < 16; k++) {
            camData[k] = view[k]!;
            camData[16 + k] = proj[k]!;
        }
        camData[32] = size;
        camData[33] = size / 2;
        camData[34] = 0; // unused
        camData[35] = PARTICLE_THICKNESS_ALPHA;
        device.queue.writeBuffer(camBuffer, 0, camData);

        // Composite uniform.
        const wm = camera.worldMatrix;
        const dl = dirLight;
        const dlLen = Math.hypot(dl[0], dl[1], dl[2]);
        const comp = new Float32Array(16 * 13);
        let o = 0;
        for (let k = 0; k < 16; k++) {
            comp[o + k] = view[k]!;
        }
        o += 16;
        for (let k = 0; k < 16; k++) {
            comp[o + k] = proj[k]!;
        }
        o += 16;
        for (let k = 0; k < 16; k++) {
            comp[o + k] = invProj[k]!;
        }
        o += 16;
        comp[o] = wm[0]!;
        comp[o + 1] = wm[1]!;
        comp[o + 2] = wm[2]!;
        comp[o + 3] = Math.tan(camera.fov / 2); // camR.xyz, camR.w = tanHalfFov
        comp[o + 4] = wm[4]!;
        comp[o + 5] = wm[5]!;
        comp[o + 6] = wm[6]!;
        comp[o + 7] = engine.canvas.width / Math.max(1, engine.canvas.height); // camU.xyz, camU.w = aspect
        comp[o + 8] = wm[8]!;
        comp[o + 9] = wm[9]!;
        comp[o + 10] = wm[10]!;
        comp[o + 11] = 0; // camF
        o += 12;
        comp[o] = 1 / fullW;
        comp[o + 1] = 1 / fullH;
        comp[o + 2] = camera.farPlane;
        comp[o + 3] = absorption; // a: output texel, far, density
        comp[o + 4] = dl[0] / dlLen;
        comp[o + 5] = dl[1] / dlLen;
        comp[o + 6] = dl[2] / dlLen;
        comp[o + 7] = refractionStrength; // b
        const debugMode = { none: 0, depth: 1, depthBlur: 2, thickness: 3, thicknessBlur: 4, normals: 5 }[debug];
        comp[o + 8] = FRESNEL_CLAMP;
        comp[o + 9] = specularPower;
        comp[o + 10] = MINIMUM_THICKNESS;
        comp[o + 11] = debugMode; // c
        comp[o + 12] = fluidColor[0];
        comp[o + 13] = fluidColor[1];
        comp[o + 14] = fluidColor[2];
        comp[o + 15] = 0; // diffuse
        o += 16;
        comp[o] = 1 / depthW;
        comp[o + 1] = 1 / depthH;
        comp[o + 2] = 0;
        comp[o + 3] = 0; // extra: depth texel, _, _
        device.queue.writeBuffer(compBuffer, 0, comp);
    }

    // Bilateral (depth) blur uniform: integer step dir, projConst, depthThreshold + maxFilterSize.
    function writeBilateral(buf: GPUBuffer, stepX: number, stepY: number): void {
        const radius = currentSim.particleRadius;
        const size = radius * PARTICLE_SIZE_SCALE * (currentSim.surfaceSizeScale ?? 1) * sizeScale;
        const projConst = (depthFilterSize * size * 0.05 * (depthH / 2)) / Math.tan(camera.fov / 2);
        const depthThreshold = (size / 2) * depthScale;
        // Couple the per-pixel max-filter clamp to the filter-size knob (default 20 → 64,
        // matching the original const ratio) so raising/lowering the slider is visible.
        const maxFilter = Math.min(128, Math.max(4, Math.round(depthFilterSize * (BLUR_MAX_FILTER_SIZE / BLUR_DEPTH_FILTER_SIZE))));
        device.queue.writeBuffer(buf, 0, new Float32Array([stepX, stepY, projConst, depthThreshold, maxFilter, 0, 0, 0]));
    }
    function writeStandard(buf: GPUBuffer, stepX: number, stepY: number): void {
        device.queue.writeBuffer(buf, 0, new Float32Array([stepX, stepY, thicknessFilterSize, 0, 0, 0, 0, 0]));
    }
    // Narrow-range filter uniform. Reuses the bilateral projConst + max-filter clamp
    // (adaptive kernel, Eq 5) and adds the depth-range params δ, µ (multipliers of
    // `size`). `cleanup` selects the fixed 5×5 2D clean-up pass (step is ignored).
    function writeNarrow(buf: GPUBuffer, stepX: number, stepY: number, cleanup: boolean): void {
        const radius = currentSim.particleRadius;
        const size = radius * PARTICLE_SIZE_SCALE * (currentSim.surfaceSizeScale ?? 1) * sizeScale;
        const projConst = (depthFilterSize * size * 0.05 * (depthH / 2)) / Math.tan(camera.fov / 2);
        const maxFilter = Math.min(128, Math.max(4, Math.round(depthFilterSize * (BLUR_MAX_FILTER_SIZE / BLUR_DEPTH_FILTER_SIZE))));
        const delta = nrDelta * size;
        const mu = nrMu * size;
        device.queue.writeBuffer(buf, 0, new Float32Array([stepX, stepY, projConst, delta, maxFilter, mu, cleanup ? 1 : 0, 0]));
    }

    function blurPass(label: string, pipe: GPURenderPipeline, buf: GPUBuffer, srcView: GPUTextureView, dstView: GPUTextureView): void {
        const bg = device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: srcView },
                { binding: 1, resource: { buffer: buf } },
            ],
        });
        const pass = engine._currentEncoder.beginRenderPass({
            label,
            colorAttachments: [{ view: dstView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
            timestampWrites: profiler?.pass("Surface"),
        });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
    }

    return {
        name: "fluid-surface",
        engine,
        scene,
        _passes: [],
        setSim(s: FluidSim): void {
            currentSim = s;
            buildParticleBG();
        },
        setMode(m: "surface" | "blit"): void {
            mode = m;
        },
        setEnvMap(e: EnvMap): void {
            envView = e.view;
            envSampler = e.sampler;
        },
        setDebug(d: FluidDebug): void {
            debug = d;
        },
        setProfiler(p: FluidProfiler | null): void {
            profiler = p;
        },
        setFluidColor(rgb: [number, number, number]): void {
            fluidColor = rgb;
        },
        setAbsorption(v: number): void {
            absorption = v;
        },
        setHalfRender(on: boolean): void {
            halfRender = on;
        },
        setThicknessDownscale(factor: number): void {
            thicknessDownscale = Math.max(1, factor);
        },
        setSizeScale(s: number): void {
            sizeScale = s;
        },
        setRefractionStrength(v: number): void {
            refractionStrength = v;
        },
        setSpecularPower(v: number): void {
            specularPower = v;
        },
        setDirLight(dir: [number, number, number]): void {
            dirLight = dir;
        },
        setDepthBlur(filterSize: number, scale: number): void {
            depthFilterSize = Math.max(0, filterSize);
            depthScale = Math.max(0, scale);
        },
        setSurfaceFilter(m: "bilateral" | "narrowRange"): void {
            surfaceFilter = m;
        },
        setNarrowRange(delta: number, mu: number): void {
            nrDelta = Math.max(0, delta);
            nrMu = Math.max(0, mu);
        },
        setThicknessBlur(filterSize: number): void {
            thicknessFilterSize = Math.max(0, filterSize);
        },
        surfaceDepthView(): GPUTextureView | null {
            // Blurred surface eye-Z (gap-filled by the bilateral blur → cleaner
            // occlusion boundaries than the raw depth). Undefined before the first
            // allocTargets() call, and reallocated on resize / half-res change.
            return views.depthBlur ?? null;
        },
        record(): void {
            build();
        },
        execute(): number {
            const outView = outRT._colorView;
            const bgView = bgRT._colorView;
            if (!depthPipe || !blitPipe || !compPipe || !particleBG || !outView || !bgView) {
                return 0;
            }
            const enc = engine._currentEncoder;

            if (mode === "blit") {
                const bg = device.createBindGroup({
                    layout: blitPipe.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: bgView },
                        { binding: 1, resource: linearSampler },
                    ],
                });
                enc.pushDebugGroup("Fluid blit (spheres)");
                const pass = enc.beginRenderPass({
                    label: "fluid-surf-blit",
                    colorAttachments: [{ view: outView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
                    timestampWrites: profiler?.pass("Surface"),
                });
                pass.setPipeline(blitPipe);
                pass.setBindGroup(0, bg);
                pass.draw(3);
                pass.end();
                enc.popDebugGroup();
                return 1;
            }

            if (!thickPipe || !bilateralPipe || !narrowRangePipe || !standardBlurPipe) {
                return 0;
            }
            allocTargets();
            updateUniforms();
            // Rebuild the particle bind group each frame: it references the SCENE depth
            // view (binding 3), which is recreated on resize — a cached group would stale.
            buildParticleBG();
            // PIX / GPU-capture debug group scoping the whole screen-space surface
            // pipeline (depth, thickness, blur, composite). Balanced before return 1.
            enc.pushDebugGroup("Fluid surface (screen-space)");

            // 1. Depth + speed (cleared to 1e6 so background reads as "far"); the
            // dedicated depth buffer (reverse-Z, cleared to far=0) keeps the
            // nearest sphere surface per pixel.
            {
                const pass = enc.beginRenderPass({
                    label: "fluid-surf-depth",
                    colorAttachments: [{ view: views.depth!, loadOp: "clear", storeOp: "store", clearValue: { r: 1e6, g: 1e6, b: 0, a: 1 } }],
                    depthStencilAttachment: { view: views.zbuf!, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 0 },
                    timestampWrites: profiler?.pass("Surface"),
                });
                pass.setPipeline(depthPipe);
                pass.setBindGroup(0, particleBG);
                pass.draw(6, currentSim.count);
                pass.end();
            }
            // 2. Thickness (additive, no depth test — full volume integral).
            {
                const pass = enc.beginRenderPass({
                    label: "fluid-surf-thick",
                    colorAttachments: [{ view: views.thick!, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
                    timestampWrites: profiler?.pass("Surface"),
                });
                pass.setPipeline(thickPipe);
                pass.setBindGroup(0, particleBG);
                pass.draw(6, currentSim.count);
                pass.end();
            }
            // 3. Depth smoothing (X then Y), and — unless skipped — a standard
            // thickness blur (X then Y). Each pass uses its OWN uniform buffer.
            enc.pushDebugGroup("surface blur (depth + thickness)");
            if (surfaceFilter === "narrowRange") {
                // Narrow-Range Filter: two 1D separable passes (depth → depthTmp →
                // depthNrTmp) followed by the fixed 5×5 2D clean-up (→ depthBlur),
                // which removes the axis-aligned streaks the 1D passes leave.
                writeNarrow(blurDepthXBuf, 1, 0, false);
                writeNarrow(blurDepthYBuf, 0, 1, false);
                writeNarrow(blurDepthCleanupBuf, 0, 0, true);
                blurPass("fluid-surf-nrX", narrowRangePipe, blurDepthXBuf, views.depth!, views.depthTmp!);
                blurPass("fluid-surf-nrY", narrowRangePipe, blurDepthYBuf, views.depthTmp!, views.depthNrTmp!);
                blurPass("fluid-surf-nrCleanup", narrowRangePipe, blurDepthCleanupBuf, views.depthNrTmp!, views.depthBlur!);
            } else {
                // Separable bilateral (adaptive, depth-weighted) blur — X then Y.
                writeBilateral(blurDepthXBuf, 1, 0);
                writeBilateral(blurDepthYBuf, 0, 1);
                blurPass("fluid-surf-depthBlurX", bilateralPipe, blurDepthXBuf, views.depth!, views.depthTmp!);
                blurPass("fluid-surf-depthBlurY", bilateralPipe, blurDepthYBuf, views.depthTmp!, views.depthBlur!);
            }
            // Thickness blur is skippable: at filter size 0 the standard blur is an identity
            // pass, so skip both passes and composite straight from the raw thickness.
            const thickBlurOn = thicknessFilterSize > 0;
            if (thickBlurOn) {
                writeStandard(blurThickXBuf, 1, 0);
                writeStandard(blurThickYBuf, 0, 1);
                blurPass("fluid-surf-thickBlurX", standardBlurPipe, blurThickXBuf, views.thick!, views.thickTmp!);
                blurPass("fluid-surf-thickBlurY", standardBlurPipe, blurThickYBuf, views.thickTmp!, views.thickBlur!);
            }
            enc.popDebugGroup();

            // 4. Composite → swapchain.
            const compBG = device.createBindGroup({
                layout: compPipe.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: views.depthBlur! },
                    { binding: 1, resource: nearestSampler },
                    { binding: 2, resource: thickBlurOn ? views.thickBlur! : views.thick! },
                    { binding: 3, resource: linearSampler },
                    { binding: 4, resource: bgView },
                    { binding: 5, resource: linearSampler },
                    { binding: 6, resource: envView },
                    { binding: 7, resource: envSampler },
                    { binding: 8, resource: views.depth! },
                    { binding: 9, resource: views.thick! },
                    { binding: 10, resource: { buffer: compBuffer } },
                    { binding: 11, resource: depthRT._depthView! },
                ],
            });
            const cpass = enc.beginRenderPass({
                label: "fluid-surf-composite",
                colorAttachments: [{ view: outView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
                timestampWrites: profiler?.pass("Surface"),
            });
            cpass.setPipeline(compPipe);
            cpass.setBindGroup(0, compBG);
            cpass.draw(3);
            cpass.end();
            enc.popDebugGroup();
            return 1;
        },
        dispose(): void {
            camBuffer.destroy();
            blurDepthXBuf.destroy();
            blurDepthYBuf.destroy();
            blurDepthCleanupBuf.destroy();
            blurThickXBuf.destroy();
            blurThickYBuf.destroy();
            compBuffer.destroy();
            for (const t of [depthTex, depthTmp, depthBlur, depthNrTmp, fluidDepthBuf, thickTex, thickTmp, thickBlur]) {
                t?.destroy();
            }
        },
    };
}
