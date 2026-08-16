// Foam (diffuse-particle) renderer — screen-space froth (Akinci 2013).
//
// References:
//   • Diffuse-particle generation/classification (spray/foam/bubbles) that fills the pool this
//     renderer draws: Ihmsen et al. 2012, "Unified spray, foam and air bubbles for particle-based
//     fluids" — https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf
//
// Draws the fluid sim's diffuse pool (spray / foam / bubbles) over the composited
// fluid surface, reading `sim.diffuse.buffer` directly in the shaders (no CPU
// round-trip). Dead slots (p.w <= 0) are collapsed in the vertex shader. Kind is
// packed into v.w by the sim's foam-update pass (0 spray / 1 foam / 2 bubble).
//
// Screen-space path (Akinci 2013 "Screen Space Foam Rendering") — each live
// particle is splatted as a round, radially-weighted coverage disc that
// ADDITIVELY accumulates a per-pixel foam thickness (RGBA16F: R = surface foam,
// G = submerged bubbles, B = spray). Each splat is depth-classified against the
// fluid-surface eye-Z (surface vs submerged) and occluded by opaque scene
// geometry (scene depth). A cheap separable blur smooths it, then a full-screen
// composite blends soft WHITE froth over the water with smoothstep-soft edges.
//
// The pass runs AFTER the fluid surface composite (added to the frame graph after
// surfaceTask), blending over `engine.scRT`.

import { getViewMatrix, getProjectionMatrix } from "../camera/camera.js";
import type { Camera } from "../camera/camera.js";
import type { EngineContext } from "../engine/engine.js";
import type { RenderTarget } from "../engine/render-target.js";
import type { SceneContext } from "../scene/scene-core.js";
import type { Task } from "../frame-graph/task.js";
import type { FluidSim, FluidProfiler } from "./sim-common.js";

// Opt-in GPU timing hook (see FluidProfiler / lab gpu-profiler.ts). Module-scoped:
// null by default so `profiler?.pass(...)` is undefined and timing costs nothing.
let profiler: FluidProfiler | null = null;

/** Foam screen-space debug texture — blits an intermediate render buffer or a
 *  derived value full-screen for inspection (mirrors the surface renderer's
 *  Debug→Feature dropdown). Applies to the screen-space (Akinci) path only; the
 *  buffers are render-side, so it is independent of the generation method. */
export type FoamDebugTexture = "off" | "accum" | "foamR" | "bubbleG" | "sprayB" | "blurred" | "foamAlpha" | "normals";

export interface FoamRenderOptions {
    /** Colour target the foam draws into (typically `engine.scRT`, the swapchain). */
    colorRT: RenderTarget;
    /** Scene depth target (opaque geometry) — sampled/tested for occlusion. */
    depthRT: RenderTarget;
    camera: Camera;
    sim: FluidSim;
    /** Per-frame accessor for the fluid-surface eye-Z (RG32F, .r = view Z, 1e6 =
     *  far). Used by the screen-space path for surface occlusion + submerged
     *  classification. Returns null before the surface renderer has allocated it
     *  (or in sphere/blit mode); the foam then falls back to "no water" (all
     *  spray/foam treated as on-top, no submerged bubbles). */
    getSurfaceDepth?: () => GPUTextureView | null;
}

// ── Screen-space path ──
// Splat: each live particle → a round radially-weighted coverage disc, additively
// accumulated into RGBA16F (R foam / G submerged bubble / B spray), depth-
// classified against the fluid-surface eye-Z and occluded by the scene depth.
function buildSplatWgsl(activeParticles: boolean): string {
    const activeDecl = activeParticles ? "\n@group(0) @binding(5) var<storage, read> activeIndices: array<u32>;" : "";
    const diffuseIndex = activeParticles ? "activeIndices[ii]" : "ii";
    return /* wgsl */ `
// Smallest accumulation-buffer radius, in pixels, a splat is allowed to project to.
const MIN_SPLAT_PX: f32 = 1.5;
struct Splat {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    size: vec4<f32>,   // particleRadius, sizeScale, foamScale, sceneScale (accum->scene res)
    texel: vec4<f32>,  // 1/accumW, 1/accumH, surfBias, sceneBias
    gains: vec4<f32>,  // sprayGain, foamGain, bubbleGain, _
};
@group(0) @binding(0) var<uniform> u: Splat;
struct Diffuse { p: vec4<f32>, v: vec4<f32> };
@group(0) @binding(1) var<storage, read> diffuse: array<Diffuse>;
@group(0) @binding(2) var surfDepth: texture_2d<f32>;
@group(0) @binding(3) var surfSamp: sampler;
@group(0) @binding(4) var sceneDepth: texture_depth_2d;
${activeDecl}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) kind: f32,
    @location(2) @interpolate(flat) gain: f32,
    @location(3) @interpolate(flat) eyeZ: f32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    var o: VOut;
    let d = diffuse[${diffuseIndex}];
    if (d.p.w <= 0.0) {
        o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        o.uv = vec2<f32>(0.0);
        o.kind = 0.0;
        o.gain = 0.0;
        o.eyeZ = 0.0;
        return o;
    }
    let kind = u32(d.v.w + 0.5);
    let baseSize = u.size.x * u.size.y * u.size.z;
    var sz = baseSize * 0.85;   // foam
    var gain = u.gains.y;
    if (kind == 0u) {           // spray: small + sharp
        sz = baseSize * 0.55;
        gain = u.gains.x;
    } else if (kind == 2u) {    // bubble: a touch larger + softer
        sz = baseSize * 1.1;
        gain = u.gains.z;
    }
    let centreEyeZ = (u.view * vec4<f32>(d.p.xyz, 1.0)).z;
    // Minimum screen footprint. A splat projecting to less than a pixel only ever lights the
    // one accumulation texel its centre lands in, and it does so at full strength — so distant
    // mist stops being mist and turns into hard, aliased white dots (very visible once the
    // camera pulls back). Grow such splats to MIN_SPLAT_PX and divide the gain by the area
    // ratio, which leaves the total accumulated energy unchanged while spreading it over a
    // real footprint: the dot becomes the faint soft blob it should have been. Splats already
    // larger than the minimum are untouched (grow == 1).
    let pxRadius = sz * u.proj[1].y / max(centreEyeZ, 1.0e-4) * 0.5 / u.texel.y;
    let grow = max(1.0, MIN_SPLAT_PX / max(pxRadius, 1.0e-4));
    sz = sz * grow;
    gain = gain / (grow * grow);
    let c = corners[vi];
    let world = d.p.xyz + u.right.xyz * (c.x * sz) + u.up.xyz * (c.y * sz);
    let eye = (u.view * vec4<f32>(world, 1.0)).xyz;
    o.clip = u.proj * vec4<f32>(eye, 1.0);
    o.uv = c;
    o.kind = f32(kind);
    o.gain = gain;
    o.eyeZ = centreEyeZ;   // particle-centre eye Z
    return o;
}

@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.uv, i.uv);
    if (r2 > 1.0) { discard; }
    let falloff = 1.0 - r2;
    let w = falloff * falloff * i.gain;   // smooth (squared) radial coverage

    // Occlusion by opaque scene geometry (reverse-Z depth -> eye Z). In the
    // fragment stage the VOut @builtin(position) member holds the framebuffer xy.
    let fragXY = i.clip.xy;
    let sPix = vec2<i32>(floor(fragXY * u.size.w));
    let sceneNdc = textureLoad(sceneDepth, sPix, 0);
    if (sceneNdc > 0.0) {
        let sceneEyeZ = u.proj[3].z / (sceneNdc - u.proj[2].z);
        if (i.eyeZ > sceneEyeZ + u.texel.w) { discard; }
    }

    // Classify against the fluid-surface eye-Z (larger eye Z = farther).
    let surfZ = textureSampleLevel(surfDepth, surfSamp, fragXY * u.texel.xy, 0.0).r;
    let hasWater = surfZ < 1e5;
    let inFront = i.eyeZ <= surfZ + u.texel.z;
    let kind = u32(i.kind + 0.5);
    var outc = vec4<f32>(0.0);
    if (kind == 2u) {
        // Bubble: only when submerged (behind the surface, water in front).
        if (hasWater && i.eyeZ > surfZ - u.texel.z) { outc.g = w; }
    } else if (kind == 0u) {
        // Spray: droplets in the air -> always the surface froth channel.
        if (inFront) { outc.b = w; }
    } else {
        // Foam: on/at the surface froth channel when at or in front of it.
        if (inFront) { outc.r = w; }
    }
    return outc;
}`;
}

const FULLSCREEN_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}`;

// Separable gaussian blur of the RGBA16F accumulation (all channels), for froth
// smoothness. Integer texel fetch at the accumulation resolution.
const FOAM_BLUR_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Blur { p: vec4<f32> };   // stepX, stepY, filterSize, _
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> b: Blur;

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(floor(pos.xy));
    let dim = vec2<i32>(textureDimensions(src));
    let filterSize = i32(b.p.z);
    let sigma = max(b.p.z / 3.0, 0.5);
    let twoSigma2 = 2.0 * sigma * sigma;
    let step = vec2<i32>(i32(b.p.x), i32(b.p.y));
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var x = -filterSize; x <= filterSize; x = x + 1) {
        let c = pix + step * x;
        if (any(c < vec2<i32>(0)) || any(c >= dim)) { continue; }
        let s = textureLoad(src, c, 0);
        let wght = exp(-f32(x * x) / twoSigma2);
        sum = sum + s * wght;
        wsum = wsum + wght;
    }
    return sum / max(wsum, 1e-4);
}`;

// Composite: shaded white froth over the water. Alpha-blended over the swapchain
// (which already holds the composited water), so the water colour shows through
// submerged bubbles for free. A fake 2.5D normal derived from the accumulated
// foam-thickness gradient drives a directional light (Lambert + soft spec + rim)
// and a thickness-based ambient occlusion, giving the froth visible 3D structure
// instead of a flat white grayscale. A `debugTex` mode blits an intermediate
// texture / derived value full-screen for inspection.
const COMPOSITE_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Comp {
    params: vec4<f32>,   // t0, t1, subStrength, debugTex (0=off .. 7=normals)
    texel: vec4<f32>,    // 1/fullW, 1/fullH, smoothOn, debugByKind
    light: vec4<f32>,    // lightIntensity, ambient, aoStrength, normalStrength
    ldir: vec4<f32>,     // lightDir.xyz, specStrength
    sub: vec4<f32>,      // submerged-bubble tint rgb, global opacity
};
@group(0) @binding(0) var accumRaw: texture_2d<f32>;
@group(0) @binding(1) var accumBlur: texture_2d<f32>;
@group(0) @binding(2) var accumSamp: sampler;
@group(0) @binding(3) var<uniform> u: Comp;

fn sampleActive(uv: vec2<f32>) -> vec4<f32> {
    if (u.texel.z > 0.5) { return textureSampleLevel(accumBlur, accumSamp, uv, 0.0); }
    return textureSampleLevel(accumRaw, accumSamp, uv, 0.0);
}
// Surface froth thickness (foam R + spray B) at a UV — the height field the fake
// normal is derived from.
fn surfAt(uv: vec2<f32>) -> f32 {
    let a = sampleActive(uv);
    return a.r + a.b;
}
fn debugComposite(rgb: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(rgb, u.sub.w);
}

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = u.texel.xy;
    let uv = pos.xy * texel;
    let acc = sampleActive(uv);
    let foamT = acc.r;
    let subT = acc.g;
    let sprayT = acc.b;
    let surfaceT = foamT + sprayT;
    let debugTex = u.params.w;

    // Fake normal from the foam-thickness gradient (central differences of the
    // surface froth thickness in screen space -> a 2.5D bump normal). A larger
    // normalStrength tilts the froth more (crisper micro-relief).
    let ns = u.light.w;
    let dTdx = surfAt(uv + vec2<f32>(texel.x, 0.0)) - surfAt(uv - vec2<f32>(texel.x, 0.0));
    let dTdy = surfAt(uv + vec2<f32>(0.0, texel.y)) - surfAt(uv - vec2<f32>(0.0, texel.y));
    let normal = normalize(vec3<f32>(-dTdx * ns, -dTdy * ns, 1.0));

    // ── Debug texture visualisations (full-screen, lifecycle-opacity blended) ──
    if (debugTex > 0.5) {
        if (debugTex < 1.5) {          // 1: accumulation (raw RGB = foam/bubble/spray)
            let a = textureSampleLevel(accumRaw, accumSamp, uv, 0.0);
            return debugComposite(clamp(a.rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
        } else if (debugTex < 2.5) {   // 2: foam channel (R)
            return debugComposite(vec3<f32>(clamp(foamT, 0.0, 1.0)));
        } else if (debugTex < 3.5) {   // 3: bubble channel (G)
            return debugComposite(vec3<f32>(clamp(subT, 0.0, 1.0)));
        } else if (debugTex < 4.5) {   // 4: spray channel (B)
            return debugComposite(vec3<f32>(clamp(sprayT, 0.0, 1.0)));
        } else if (debugTex < 5.5) {   // 5: blurred accumulation (RGB)
            let a = textureSampleLevel(accumBlur, accumSamp, uv, 0.0);
            return debugComposite(clamp(a.rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
        } else if (debugTex < 6.5) {   // 6: foam alpha (post-smoothstep coverage)
            let fa = smoothstep(u.params.x, u.params.y, surfaceT);
            return debugComposite(vec3<f32>(fa));
        }
        // 7: fake normals (screen-space, encoded to 0..1).
        return debugComposite(normal * 0.5 + vec3<f32>(0.5));
    }

    // Legacy "colour by kind" checkbox (screen path): spray red / foam green / bubble blue.
    if (u.texel.w > 0.5) {
        let c = vec3<f32>(clamp(sprayT, 0.0, 1.0), clamp(foamT, 0.0, 1.0), clamp(subT, 0.0, 1.0));
        let a = max(max(sprayT, foamT), subT);
        if (a <= 0.002) { discard; }
        return vec4<f32>(c, min(a, 1.0) * 0.95 * u.sub.w);
    }

    let foamAlpha = smoothstep(u.params.x, u.params.y, surfaceT);

    // ── Directional shading of the froth (Lambert + soft spec + rim + AO) ──
    let L = normalize(u.ldir.xyz);
    let V = vec3<f32>(0.0, 0.0, 1.0);          // screen-space view direction
    let H = normalize(L + V);
    let ndl = clamp(dot(normal, L), 0.0, 1.0);
    let diffuse = ndl * u.light.x;             // lightIntensity
    let ambient = u.light.y;
    let spec = pow(clamp(dot(normal, H), 0.0, 1.0), 40.0) * u.ldir.w;
    let rim = pow(1.0 - clamp(normal.z, 0.0, 1.0), 3.0) * 0.25;
    // Thickness-based ambient occlusion: denser/deeper foam sits slightly darker,
    // giving crevice contrast between froth clumps.
    let ao = 1.0 - u.light.z * clamp(surfaceT * 0.16, 0.0, 1.0);
    let lum = (ambient + diffuse) * ao;
    let foamColor = clamp(vec3<f32>(lum) + vec3<f32>(spec + rim), vec3<f32>(0.0), vec3<f32>(1.25));

    // Submerged bubbles: faint, slightly bluish, but shaded by the same normal so
    // they inherit the thickness variation instead of reading as a flat tint.
    let subShade = ambient + diffuse * 0.5;
    let subColor = u.sub.rgb * clamp(subShade + 0.15, 0.35, 1.1);
    let subAlpha = smoothstep(u.params.x, u.params.y, subT) * u.params.z;

    // Composite the (opaque) foam over the (faint) bubble layer, non-premultiplied.
    let baseA = foamAlpha + subAlpha * (1.0 - foamAlpha);
    if (baseA <= 0.002) { discard; }
    let outRGB = (foamColor * foamAlpha + subColor * subAlpha * (1.0 - foamAlpha)) / max(baseA, 1e-4);
    return vec4<f32>(outRGB, min(baseA, 1.0) * u.sub.w);
}`;

const ALPHA_BLEND = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
} as const;
const ADD_BLEND = {
    color: { srcFactor: "one", dstFactor: "one", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
} as const;

const FOAM_ACCUM_FORMAT: GPUTextureFormat = "rgba16float";
const FOAM_BLUR_DEFAULT = 4; // separable blur half-size (accum smoothing) — runtime tunable

/** Maps a FoamDebugTexture to the composite shader's `debugTex` uniform index. */
const FOAM_DEBUG_INDEX: Record<FoamDebugTexture, number> = {
    off: 0,
    accum: 1,
    foamR: 2,
    bubbleG: 3,
    sprayB: 4,
    blurred: 5,
    foamAlpha: 6,
    normals: 7,
};

export function createFoamRenderTask(
    engine: EngineContext,
    scene: SceneContext,
    opts: FoamRenderOptions
): Task & {
    setSim(s: FluidSim): void;
    setEnabled(on: boolean): void;
    setOpacity(v: number): void;
    setSizeScale(s: number): void;
    setDebugByKind(on: boolean): void;
    setThresholds(t0: number, t1: number): void;
    setSubsurfaceStrength(v: number): void;
    setSubsurfaceColor(rgb: [number, number, number]): void;
    setBlurRadius(n: number): void;
    setDebugTexture(m: FoamDebugTexture): void;
    setLightIntensity(v: number): void;
    setAmbient(v: number): void;
    setAOStrength(v: number): void;
    setNormalStrength(v: number): void;
    /** Opt-in GPU timing hook: tag every foam-render pass with timestampWrites, or
     *  null to turn timing off. The profiler machinery lives in the app (lab). */
    setProfiler(p: FluidProfiler | null): void;
} {
    const device = engine._device;
    const { colorRT, depthRT, camera } = opts;
    const getSurfaceDepth = opts.getSurfaceDepth;
    let currentSim = opts.sim;
    let enabled = true;
    let opacity = 1;
    let sizeScale = 1;
    let debugByKind = false;
    let t0 = 0.25;
    let t1 = 1.6;
    let subStrength = 0.4;
    /** Submerged-bubble tint. Was a hardcoded pale blue; now host-settable so a scene can
     *  match the bubbles to its water colour (murky green, night-time steel, ...). */
    let subColor: [number, number, number] = [0.72, 0.82, 0.95];
    let blurRadius = FOAM_BLUR_DEFAULT;
    let debugTexIndex = 0; // FoamDebugTexture -> composite `debugTex` uniform (0 = off)
    // Screen-space froth shading knobs (fake-normal directional light + AO).
    let lightIntensity = 0.9;
    let ambient = 0.5;
    let aoStrength = 0.5;
    let normalStrength = 6.0;
    const lightDir: [number, number, number] = [-0.4, 0.7, 0.6];
    const specStrength = 0.6;

    // ── Screen-path resources ──
    const splatData = new Float32Array(52); // view(16)+proj(16)+right(4)+up(4)+size(4)+texel(4)+gains(4)
    const splatBuf = device.createBuffer({ label: "fluid-foam-splat", size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blurXBuf = device.createBuffer({ label: "fluid-foam-blur-x", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blurYBuf = device.createBuffer({ label: "fluid-foam-blur-y", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const compBuf = device.createBuffer({ label: "fluid-foam-comp", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const nearestSampler = device.createSampler({ label: "fluid-foam-nearest" });
    const linearSampler = device.createSampler({ label: "fluid-foam-linear", magFilter: "linear", minFilter: "linear" });

    // 1x1 RG32F "no water" fallback (surface eye-Z = far) for sphere/blit mode.
    const noWaterTex = device.createTexture({ label: "fluid-foam-nowater", size: [1, 1], format: "rg32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: noWaterTex }, new Float32Array([1e6, 1e6]), { bytesPerRow: 8 }, [1, 1]);
    const noWaterView = noWaterTex.createView();

    let splatPipe: GPURenderPipeline | null = null;
    let activeSplatPipe: GPURenderPipeline | null = null;
    let blurPipe: GPURenderPipeline | null = null;
    let compPipe: GPURenderPipeline | null = null;

    let accW = 0;
    let accH = 0;
    let accumTex: GPUTexture | null = null;
    let accumTmp: GPUTexture | null = null;
    let accumBlur: GPUTexture | null = null;
    let accumView: GPUTextureView | null = null;
    let accumTmpView: GPUTextureView | null = null;
    let accumBlurView: GPUTextureView | null = null;

    function allocAccum(): void {
        const w = engine.canvas.width;
        const h = engine.canvas.height;
        if (w === accW && h === accH && accumTex) {
            return;
        }
        for (const t of [accumTex, accumTmp, accumBlur]) {
            t?.destroy();
        }
        accW = w;
        accH = h;
        const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
        accumTex = device.createTexture({ label: "fluid-foam-accum", size: { width: w, height: h }, format: FOAM_ACCUM_FORMAT, usage });
        accumTmp = device.createTexture({ label: "fluid-foam-accum-tmp", size: { width: w, height: h }, format: FOAM_ACCUM_FORMAT, usage });
        accumBlur = device.createTexture({ label: "fluid-foam-accum-blur", size: { width: w, height: h }, format: FOAM_ACCUM_FORMAT, usage });
        accumView = accumTex.createView();
        accumTmpView = accumTmp.createView();
        accumBlurView = accumBlur.createView();
    }

    function build(): void {
        if (splatPipe) {
            return;
        }
        const splatMod = device.createShaderModule({ label: "fluid-foam-splat", code: buildSplatWgsl(false) });
        splatPipe = device.createRenderPipeline({
            label: "fluid-foam-splat",
            layout: "auto",
            vertex: { module: splatMod, entryPoint: "vs" },
            fragment: { module: splatMod, entryPoint: "fs", targets: [{ format: FOAM_ACCUM_FORMAT, blend: ADD_BLEND }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
        });
        const activeSplatMod = device.createShaderModule({ label: "fluid-foam-splat-active", code: buildSplatWgsl(true) });
        activeSplatPipe = device.createRenderPipeline({
            label: "fluid-foam-splat-active",
            layout: "auto",
            vertex: { module: activeSplatMod, entryPoint: "vs" },
            fragment: { module: activeSplatMod, entryPoint: "fs", targets: [{ format: FOAM_ACCUM_FORMAT, blend: ADD_BLEND }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
        });
        const blurMod = device.createShaderModule({ label: "fluid-foam-blur", code: FOAM_BLUR_WGSL });
        blurPipe = device.createRenderPipeline({
            label: "fluid-foam-blur",
            layout: "auto",
            vertex: { module: blurMod, entryPoint: "vs" },
            fragment: { module: blurMod, entryPoint: "fs", targets: [{ format: FOAM_ACCUM_FORMAT }] },
            primitive: { topology: "triangle-list" },
        });
        const compMod = device.createShaderModule({ label: "fluid-foam-comp", code: COMPOSITE_WGSL });
        compPipe = device.createRenderPipeline({
            label: "fluid-foam-comp",
            layout: "auto",
            vertex: { module: compMod, entryPoint: "vs" },
            fragment: { module: compMod, entryPoint: "fs", targets: [{ format: engine.format, blend: ALPHA_BLEND }] },
            primitive: { topology: "triangle-list" },
        });
    }

    function updateSplatUniform(): void {
        const aspect = engine.canvas.width / Math.max(1, engine.canvas.height);
        const view = getViewMatrix(camera);
        const proj = getProjectionMatrix(camera, aspect);
        for (let k = 0; k < 16; k++) {
            splatData[k] = view[k]!;
            splatData[16 + k] = proj[k]!;
        }
        const wm = camera.worldMatrix;
        splatData[32] = wm[0]!;
        splatData[33] = wm[1]!;
        splatData[34] = wm[2]!;
        splatData[35] = 0;
        splatData[36] = wm[4]!;
        splatData[37] = wm[5]!;
        splatData[38] = wm[6]!;
        splatData[39] = 0;
        const r = currentSim.particleRadius;
        splatData[40] = r;
        splatData[41] = sizeScale;
        splatData[42] = 2.5; // foamScale (splat radius multiplier — overlap for froth)
        splatData[43] = engine.canvas.width / Math.max(1, accW); // accum→scene res ratio (1 at full res)
        splatData[44] = 1 / Math.max(1, accW);
        splatData[45] = 1 / Math.max(1, accH);
        splatData[46] = r * 4; // surfBias (eye-Z tolerance for "on the surface")
        splatData[47] = 0.02; // sceneBias (eye-Z bias vs opaque geometry)
        splatData[48] = 1.4; // sprayGain
        splatData[49] = 1.0; // foamGain
        splatData[50] = 1.0; // bubbleGain
        splatData[51] = 0;
        device.queue.writeBuffer(splatBuf, 0, splatData);

        device.queue.writeBuffer(blurXBuf, 0, new Float32Array([1, 0, blurRadius, 0]));
        device.queue.writeBuffer(blurYBuf, 0, new Float32Array([0, 1, blurRadius, 0]));
        device.queue.writeBuffer(
            compBuf,
            0,
            new Float32Array([
                // params: t0, t1, subStrength, debugTex
                t0,
                t1,
                subStrength,
                debugTexIndex,
                // texel: 1/fullW, 1/fullH, smoothOn, debugByKind
                1 / Math.max(1, engine.canvas.width),
                1 / Math.max(1, engine.canvas.height),
                // smoothOn: only read the blurred accumulation when a real blur ran
                // (radius > 0). At radius 0 the blur passes are skipped, so read the raw
                // accumulation instead (identical visuals to "no blur", zero extra passes).
                blurRadius > 0 ? 1 : 0,
                debugByKind ? 1 : 0,
                // light: intensity, ambient, aoStrength, normalStrength
                lightIntensity,
                ambient,
                aoStrength,
                normalStrength,
                // ldir: lightDir.xyz, specStrength
                lightDir[0],
                lightDir[1],
                lightDir[2],
                specStrength,
                // sub: submerged-bubble tint rgb, global opacity
                subColor[0],
                subColor[1],
                subColor[2],
                opacity,
            ])
        );
    }

    function blurPass(pipe: GPURenderPipeline, buf: GPUBuffer, src: GPUTextureView, dst: GPUTextureView): void {
        const bg = device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: src },
                { binding: 1, resource: { buffer: buf } },
            ],
        });
        const pass = engine._currentEncoder.beginRenderPass({
            label: "fluid-foam-blur",
            colorAttachments: [{ view: dst, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
            timestampWrites: profiler?.pass("Foam render"),
        });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
    }

    function executeScreen(colorView: GPUTextureView): number {
        const pool = currentSim.diffuse!;
        if (!splatPipe || !activeSplatPipe || !blurPipe || !compPipe) {
            return 0;
        }
        const sceneDepthView = depthRT._depthView;
        if (!sceneDepthView) {
            return 0;
        }
        allocAccum();
        updateSplatUniform();
        const surfView = getSurfaceDepth?.() ?? noWaterView;

        // 1. Splat → accumulation (additive, depth-classified + occluded).
        {
            const activeParticles = !!pool.activeIndices && !!pool.drawIndirect;
            const activePipe = activeParticles ? activeSplatPipe : splatPipe;
            const entries: GPUBindGroupEntry[] = [
                { binding: 0, resource: { buffer: splatBuf } },
                { binding: 1, resource: { buffer: pool.buffer } },
                { binding: 2, resource: surfView },
                { binding: 3, resource: nearestSampler },
                { binding: 4, resource: sceneDepthView },
            ];
            if (activeParticles) {
                entries.push({
                    binding: 5,
                    resource: { buffer: pool.activeIndices!, offset: pool.activeIndicesOffset ?? 0, size: pool.capacity * 4 },
                });
            }
            const bg = device.createBindGroup({
                label: "fluid-foam-splat",
                layout: activePipe.getBindGroupLayout(0),
                entries,
            });
            const pass = engine._currentEncoder.beginRenderPass({
                label: "fluid-foam-splat",
                colorAttachments: [{ view: accumView!, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
                timestampWrites: profiler?.pass("Foam render"),
            });
            pass.setPipeline(activePipe);
            pass.setBindGroup(0, bg);
            if (activeParticles) {
                pass.drawIndirect(pool.drawIndirect!, 0);
            } else {
                pass.draw(6, pool.capacity);
            }
            pass.end();
        }

        // 2. Optional separable blur (X then Y) for froth smoothness. Also forced
        // when the "Blurred accumulation" debug mode needs a fresh blurred texture.
        // Skip the two separable blur passes when there's nothing to blur (radius 0):
        // an identity blur is wasted work, and radius 0 (no blur) is a valid, common look.
        // Still forced for the "Blurred accumulation" debug mode.
        const wantBlur = blurRadius > 0 || debugTexIndex === FOAM_DEBUG_INDEX.blurred;
        if (wantBlur) {
            blurPass(blurPipe, blurXBuf, accumView!, accumTmpView!);
            blurPass(blurPipe, blurYBuf, accumTmpView!, accumBlurView!);
        }

        // 3. Composite shaded froth over the water (alpha-blended). Both the raw and
        // blurred accumulations are bound; the shader picks the active source
        // (blurred when smoothing is on) and can also blit either for debug.
        {
            const bg = device.createBindGroup({
                label: "fluid-foam-comp",
                layout: compPipe.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: accumView! },
                    { binding: 1, resource: accumBlurView! },
                    { binding: 2, resource: linearSampler },
                    { binding: 3, resource: { buffer: compBuf } },
                ],
            });
            const pass = engine._currentEncoder.beginRenderPass({
                label: "fluid-foam-comp",
                colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
                timestampWrites: profiler?.pass("Foam render"),
            });
            pass.setPipeline(compPipe);
            pass.setBindGroup(0, bg);
            pass.draw(3);
            pass.end();
        }
        return 1;
    }

    return {
        name: "fluid-foam",
        engine,
        scene,
        _passes: [],
        /** Switch the rendered simulation backend (the screen-space path binds the
         *  active pool each frame, so no explicit rebind is needed). */
        setSim(s: FluidSim): void {
            currentSim = s;
        },
        /** Enable/disable this renderer. */
        setEnabled(on: boolean): void {
            enabled = on;
        },
        /** Fade all rendered diffuse particles without changing their pool. */
        setOpacity(v: number): void {
            opacity = Math.max(0, Math.min(1, v));
        },
        /** Visual foam size multiplier (screen-space splat radius). */
        setSizeScale(s: number): void {
            sizeScale = s;
        },
        /** Toggle the debug "colour by kind" mode (spray red / foam green / bubble blue). */
        setDebugByKind(on: boolean): void {
            debugByKind = on;
        },
        /** Screen-space froth softness: smoothstep(t0, t1) over accumulated coverage. */
        setThresholds(a: number, b: number): void {
            t0 = a;
            t1 = b;
        },
        /** Screen-space submerged-bubble opacity (0 = hidden, ~0.4 default). */
        setSubsurfaceStrength(v: number): void {
            subStrength = v;
        },
        /** Submerged-bubble tint, linear RGB in 0..1 (default pale blue 0.72/0.82/0.95). */
        setSubsurfaceColor(rgb: [number, number, number]): void {
            subColor = rgb;
        },
        /** Screen-space accumulation blur half-size (0 = off/raw, ~4 default). The single
         *  froth-smoothing control: 0 skips the separable blur passes and composites the
         *  raw accumulation; a positive radius runs them and composites the blurred result. */
        setBlurRadius(n: number): void {
            blurRadius = Math.max(0, Math.round(n));
        },
        /** Select the screen-space debug texture blit ("off" = normal composite). */
        setDebugTexture(m: FoamDebugTexture): void {
            debugTexIndex = FOAM_DEBUG_INDEX[m] ?? 0;
        },
        /** Screen-space froth directional-light intensity (Lambert term). */
        setLightIntensity(v: number): void {
            lightIntensity = v;
        },
        /** Screen-space froth ambient term (baseline brightness). */
        setAmbient(v: number): void {
            ambient = v;
        },
        /** Screen-space froth thickness-based ambient-occlusion strength. */
        setAOStrength(v: number): void {
            aoStrength = v;
        },
        /** Screen-space froth fake-normal strength (thickness-gradient tilt). */
        setNormalStrength(v: number): void {
            normalStrength = v;
        },
        setProfiler(p: FluidProfiler | null): void {
            profiler = p;
        },
        record(): void {
            build();
        },
        execute(): number {
            if (!enabled || opacity <= 0) {
                return 0;
            }
            const pool = currentSim.diffuse;
            const colorView = colorRT._colorView;
            if (!pool || !colorView) {
                return 0;
            }
            engine._currentEncoder.pushDebugGroup("Foam render (screen-space)");
            const r = executeScreen(colorView);
            engine._currentEncoder.popDebugGroup();
            return r;
        },
        dispose(): void {
            splatBuf.destroy();
            blurXBuf.destroy();
            blurYBuf.destroy();
            compBuf.destroy();
            noWaterTex.destroy();
            for (const t of [accumTex, accumTmp, accumBlur]) {
                t?.destroy();
            }
        },
    };
}
