// Screen-space fluid surface renderer for the GPU fluid sim.
//
// Reconstructs a smooth liquid SURFACE in screen space (the technique behind
// Babylon.js's FluidRenderer / van der Laan 2009), all demo-local and encoded as
// one frame-graph task that also PRESENTS the frame to the swapchain. The scene
// is rendered to an offscreen colour target (`bgRT`) so this pass can sample it
// for refraction. Pipeline (surface mode):
//
//   1. depth     : particles as sphere impostors write the nearest eye-space
//                  depth + the surface particle's speed into an RGBA16F target,
//                  depth-tested against the shared scene depth (ground/paddle
//                  occlude the fluid).
//   2. thickness : impostors again, additively, into an RGBA16F target — the
//                  fluid column thickness (drives absorption + refraction).
//   3. blur      : separable bilateral blur of BOTH the depth and the thickness
//                  (depth-weighted), turning the bumpy sphere data into a smooth
//                  surface + smooth thickness.
//   4. composite : reconstruct eye position + normal, then shade like real water
//                  — refract the background (sample `bgRT` offset by the surface
//                  normal), absorb it through the thickness (Beer-Lambert tint),
//                  Fresnel-mix with a reflection colour, add a specular highlight
//                  and whiten by speed (foam). Writes the swapchain.
//
// In "blit" mode (the demo's sphere-impostor renderer is active instead) this
// task just copies `bgRT` (scene + impostors) to the swapchain.

import { getEffectiveAspectRatio, getViewProjectionMatrix } from "babylon-lite";
import type { Camera, EngineContext, RenderTarget, SceneContext, Task } from "babylon-lite";
import type { FluidSim } from "./pbf-sim.js";

export interface FluidSurfaceOptions {
    /** Offscreen scene colour (background) — sampled for refraction / blit. */
    bgRT: RenderTarget;
    /** Final output (the swapchain, typically `engine.scRT`). */
    outRT: RenderTarget;
    /** Depth target shared with the scene render task (depth-test for occlusion). */
    depthRT: RenderTarget;
    camera: Camera;
    sim: FluidSim;
}

// Bilateral-blur half-width in texels (per separable pass).
const BLUR_RADIUS = 14;

// Shared billboard + camera uniform (depth and thickness passes).
const SPHERE_WGSL = /* wgsl */ `
struct Cam {
    vp: mat4x4<f32>,
    right: vec4<f32>,   // camera world right (billboard X)
    up: vec4<f32>,      // camera world up    (billboard Y)
    misc: vec4<f32>,    // x = radius, y = speed norm, z = thickness/particle
};
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dbg: array<f32>;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) viewZc: f32,
    @location(2) @interpolate(flat) speed: f32,
};

fn corner(vi: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    return corners[vi];
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let c = corner(vi);
    let center = positions[ii].xyz;
    let r = cam.misc.x;
    let world = center + cam.right.xyz * (c.x * r) + cam.up.xyz * (c.y * r);
    var o: VOut;
    o.clip = cam.vp * vec4<f32>(world, 1.0);
    o.viewZc = (cam.vp * vec4<f32>(center, 1.0)).w; // LH clip.w = view-space z (eye depth)
    o.uv = c;
    o.speed = clamp(dbg[ii] * cam.misc.y, 0.0, 1.0);
    return o;
}

@fragment fn fsDepth(i: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.uv, i.uv);
    if (r2 > 1.0) { discard; }
    let nz = sqrt(1.0 - r2);
    let eyeDepth = i.viewZc - cam.misc.x * nz;
    return vec4<f32>(eyeDepth, i.speed, 0.0, 1.0);
}

@fragment fn fsThick(i: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.uv, i.uv);
    if (r2 > 1.0) { discard; }
    return vec4<f32>(cam.misc.z * (1.0 - r2), 0.0, 0.0, 1.0);
}`;

const FULLSCREEN_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}`;

// Separable bilateral blur of an RGBA16F target — smooths R (depth or thickness)
// and carries G (speed) along, weighting by the difference in R so silhouettes
// stay sharp. `dir.xy` = texel direction, `dir.z` = 1/(2*rangeSigma^2).
const BLUR_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Blur { dir: vec4<f32> }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> bl: Blur;

const RADIUS: i32 = ${BLUR_RADIUS};
const INV_SS2: f32 = ${(1 / (2 * (BLUR_RADIUS / 2.0) * (BLUR_RADIUS / 2.0))).toFixed(6)};

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(floor(pos.xy));
    let dim = vec2<i32>(textureDimensions(src));
    let center = textureLoad(src, pix, 0);
    let d0 = center.r;
    if (d0 <= 0.0) { return vec4<f32>(0.0); }
    let step = vec2<i32>(i32(bl.dir.x), i32(bl.dir.y));
    var sumD = 0.0;
    var sumS = 0.0;
    var sumW = 0.0;
    for (var i = -RADIUS; i <= RADIUS; i = i + 1) {
        let c = pix + step * i;
        if (any(c < vec2<i32>(0)) || any(c >= dim)) { continue; }
        let t = textureLoad(src, c, 0);
        if (t.r <= 0.0) { continue; }
        let ws = exp(-f32(i * i) * INV_SS2);
        let dd = t.r - d0;
        let wr = exp(-dd * dd * bl.dir.z);
        let w = ws * wr;
        sumD = sumD + t.r * w;
        sumS = sumS + t.g * w;
        sumW = sumW + w;
    }
    if (sumW <= 0.0) { return vec4<f32>(d0, center.g, 0.0, 1.0); }
    return vec4<f32>(sumD / sumW, sumS / sumW, 0.0, 1.0);
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

// Composite: shade the reconstructed surface like water and present it.
const COMPOSITE_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Comp {
    res: vec4<f32>,    // xy = size (px), zw = 1/size
    proj: vec4<f32>,   // tanHalfFov, aspect, refractStrength, foamStrength
    light: vec4<f32>,  // xyz = view light dir, w = specular strength
    absorb: vec4<f32>, // xyz = absorption coeff (per channel), w = fresnel F0
    deep: vec4<f32>,   // xyz = deep fluid colour, w = shininess
};
@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var thickTex: texture_2d<f32>;
@group(0) @binding(2) var bgTex: texture_2d<f32>;
@group(0) @binding(3) var bgSamp: sampler;
@group(0) @binding(4) var<uniform> c: Comp;

fn viewPos(pix: vec2<i32>, d: f32) -> vec3<f32> {
    let ndcX = (f32(pix.x) + 0.5) * c.res.z * 2.0 - 1.0;
    let ndcY = 1.0 - (f32(pix.y) + 0.5) * c.res.w * 2.0;
    return vec3<f32>(ndcX * d * c.proj.y * c.proj.x, ndcY * d * c.proj.x, d);
}

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(floor(pos.xy));
    let dim = vec2<i32>(textureDimensions(depthTex));
    let uv = (vec2<f32>(pix) + 0.5) * c.res.zw;
    let d0 = textureLoad(depthTex, pix, 0).r;
    if (d0 <= 0.0) {
        return vec4<f32>(textureSampleLevel(bgTex, bgSamp, uv, 0.0).rgb, 1.0);
    }

    let p0 = viewPos(pix, d0);
    let xr = clamp(pix + vec2<i32>(1, 0), vec2<i32>(0), dim - vec2<i32>(1));
    let xl = clamp(pix - vec2<i32>(1, 0), vec2<i32>(0), dim - vec2<i32>(1));
    let yd = clamp(pix + vec2<i32>(0, 1), vec2<i32>(0), dim - vec2<i32>(1));
    let yu = clamp(pix - vec2<i32>(0, 1), vec2<i32>(0), dim - vec2<i32>(1));
    let dxr = textureLoad(depthTex, xr, 0).r;
    let dxl = textureLoad(depthTex, xl, 0).r;
    let dyd = textureLoad(depthTex, yd, 0).r;
    let dyu = textureLoad(depthTex, yu, 0).r;
    var ddx = viewPos(xr, dxr) - p0;
    let ddxl = p0 - viewPos(xl, dxl);
    if (dxr <= 0.0 || (dxl > 0.0 && abs(ddxl.z) < abs(ddx.z))) { ddx = ddxl; }
    var ddy = viewPos(yd, dyd) - p0;
    let ddyu = p0 - viewPos(yu, dyu);
    if (dyd <= 0.0 || (dyu > 0.0 && abs(ddyu.z) < abs(ddy.z))) { ddy = ddyu; }
    var n = normalize(cross(ddx, ddy));
    if (n.z > 0.0) { n = -n; }

    let V = normalize(-p0);
    let L = normalize(c.light.xyz);
    let H = normalize(L + V);
    let spec = pow(max(dot(n, H), 0.0), c.deep.w) * c.light.w;
    let fres = c.absorb.w + (1.0 - c.absorb.w) * pow(1.0 - max(dot(n, V), 0.0), 5.0);

    let thick = textureLoad(thickTex, pix, 0).r;

    // Refraction: offset the background lookup by the surface normal, a bit more
    // for a thicker surface. (Screen Y is flipped vs. view Y.)
    let off = vec2<f32>(n.x, -n.y) * c.proj.z * (0.5 + clamp(thick, 0.0, 2.0));
    let bg = textureSampleLevel(bgTex, bgSamp, clamp(uv + off, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;

    // Beer-Lambert absorption: thicker fluid transmits less and tends to the deep
    // colour.
    let trans = exp(-thick * c.absorb.xyz);
    let refr = bg * trans + c.deep.xyz * (1.0 - trans);

    // Reflection: a cheap sky/horizon gradient along the reflected ray.
    let R = reflect(-V, n);
    let refl = mix(vec3<f32>(0.18, 0.26, 0.38), vec3<f32>(0.5, 0.62, 0.8), clamp(R.y * 0.5 + 0.5, 0.0, 1.0));

    var col = mix(refr, refl, fres);
    col = col + vec3<f32>(1.0) * spec;

    // Speed → foam: only genuinely fast-moving surface whitens (settled fluid
    // keeps its translucent look).
    let speed = textureLoad(depthTex, pix, 0).g;
    let foam = smoothstep(0.45, 0.95, speed) * c.proj.w;
    col = mix(col, vec3<f32>(0.92, 0.96, 1.0), foam);
    return vec4<f32>(col, 1.0);
}`;

export function createFluidSurfaceTask(
    engine: EngineContext,
    scene: SceneContext,
    opts: FluidSurfaceOptions,
): Task & { setSim(s: FluidSim): void; setMode(m: "surface" | "blit"): void } {
    const device = engine._device;
    const { bgRT, outRT, depthRT, camera } = opts;
    let currentSim = opts.sim;
    let mode: "surface" | "blit" = "surface";

    const camData = new Float32Array(28);
    const camBuffer = device.createBuffer({ label: "fluid-surf-cam", size: camData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blurBuffer = device.createBuffer({ label: "fluid-surf-blur", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const compBuffer = device.createBuffer({ label: "fluid-surf-comp", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const sampler = device.createSampler({ label: "fluid-surf-samp", magFilter: "linear", minFilter: "linear" });

    let width = 0;
    let height = 0;
    let depthTex: GPUTexture | null = null;
    let thickTex: GPUTexture | null = null;
    let tmpTex: GPUTexture | null = null;
    let depthBlur: GPUTexture | null = null;
    let thickBlur: GPUTexture | null = null;
    let views: Record<string, GPUTextureView> = {};

    function allocTargets(): void {
        const w = engine.canvas.width;
        const h = engine.canvas.height;
        if (w === width && h === height && depthTex) {
            return;
        }
        for (const t of [depthTex, thickTex, tmpTex, depthBlur, thickBlur]) {
            t?.destroy();
        }
        width = w;
        height = h;
        const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
        const mk = (label: string): GPUTexture => device.createTexture({ label, size: { width: w, height: h }, format: "rgba16float", usage });
        depthTex = mk("fluid-surf-depth");
        thickTex = mk("fluid-surf-thick");
        tmpTex = mk("fluid-surf-tmp");
        depthBlur = mk("fluid-surf-depthBlur");
        thickBlur = mk("fluid-surf-thickBlur");
        views = {
            depth: depthTex.createView(),
            thick: thickTex.createView(),
            tmp: tmpTex.createView(),
            depthBlur: depthBlur.createView(),
            thickBlur: thickBlur.createView(),
        };
    }

    let depthPipe: GPURenderPipeline | null = null;
    let thickPipe: GPURenderPipeline | null = null;
    let blurPipe: GPURenderPipeline | null = null;
    let blitPipe: GPURenderPipeline | null = null;
    let compPipe: GPURenderPipeline | null = null;
    let sphereBGL: GPUBindGroupLayout | null = null;
    let sphereBG: GPUBindGroup | null = null;

    function buildSphereBG(): void {
        if (!sphereBGL) {
            return;
        }
        sphereBG = device.createBindGroup({
            label: "fluid-surf-sphere",
            layout: sphereBGL,
            entries: [
                { binding: 0, resource: { buffer: camBuffer } },
                { binding: 1, resource: { buffer: currentSim.positionBuffer } },
                { binding: 2, resource: { buffer: currentSim.debugBuffer } },
            ],
        });
    }

    function build(): void {
        if (depthPipe) {
            return;
        }
        const sphereMod = device.createShaderModule({ label: "fluid-surf-sphere", code: SPHERE_WGSL });
        const dFormat = depthRT._descriptor.dFormat!;
        const samples = depthRT._descriptor.samples;
        sphereBGL = device.createBindGroupLayout({
            label: "fluid-surf-sphere",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            ],
        });
        const spherePL = device.createPipelineLayout({ bindGroupLayouts: [sphereBGL] });
        depthPipe = device.createRenderPipeline({
            label: "fluid-surf-depth",
            layout: spherePL,
            vertex: { module: sphereMod, entryPoint: "vs" },
            fragment: { module: sphereMod, entryPoint: "fsDepth", targets: [{ format: "rgba16float" }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: { format: dFormat, depthWriteEnabled: true, depthCompare: "greater-equal" },
            multisample: { count: samples },
        });
        const addBlend = { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } as const;
        thickPipe = device.createRenderPipeline({
            label: "fluid-surf-thick",
            layout: spherePL,
            vertex: { module: sphereMod, entryPoint: "vs" },
            fragment: { module: sphereMod, entryPoint: "fsThick", targets: [{ format: "rgba16float", blend: addBlend }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: { format: dFormat, depthWriteEnabled: false, depthCompare: "greater-equal" },
            multisample: { count: samples },
        });
        const blurMod = device.createShaderModule({ label: "fluid-surf-blur", code: BLUR_WGSL });
        blurPipe = device.createRenderPipeline({
            label: "fluid-surf-blur",
            layout: "auto",
            vertex: { module: blurMod, entryPoint: "vs" },
            fragment: { module: blurMod, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
            primitive: { topology: "triangle-list" },
        });
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
        buildSphereBG();
    }

    function updateUniforms(): void {
        const aspect = getEffectiveAspectRatio(camera, engine.canvas.width, engine.canvas.height);
        const vp = getViewProjectionMatrix(camera, aspect);
        for (let k = 0; k < 16; k++) {
            camData[k] = vp[k]!;
        }
        const wm = camera.worldMatrix;
        camData[16] = wm[0]!; camData[17] = wm[1]!; camData[18] = wm[2]!; camData[19] = 0;
        camData[20] = wm[4]!; camData[21] = wm[5]!; camData[22] = wm[6]!; camData[23] = 0;
        camData[24] = currentSim.particleRadius * 1.7;
        camData[25] = currentSim.debugNorm;
        camData[26] = 0.1; // thickness per particle (lower = more transparent)
        camData[27] = 0;
        device.queue.writeBuffer(camBuffer, 0, camData);

        const tanHalfFov = Math.tan(camera.fov * 0.5);
        const comp = new Float32Array(20);
        comp[0] = width; comp[1] = height; comp[2] = 1 / width; comp[3] = 1 / height;
        comp[4] = tanHalfFov; comp[5] = aspect; comp[6] = 0.05; comp[7] = 0.6;  // refractStrength, foam
        comp[8] = 0.3; comp[9] = 0.5; comp[10] = -0.8; comp[11] = 0.5;          // view light, spec
        comp[12] = 0.32; comp[13] = 0.16; comp[14] = 0.1; comp[15] = 0.04;      // absorption rgb, fresnel F0
        comp[16] = 0.04; comp[17] = 0.16; comp[18] = 0.28; comp[19] = 120;      // deep colour, shininess
        device.queue.writeBuffer(compBuffer, 0, comp);
    }

    function writeBlur(dirX: number, dirY: number, rangeSigma: number): void {
        device.queue.writeBuffer(blurBuffer, 0, new Float32Array([dirX, dirY, 1 / (2 * rangeSigma * rangeSigma), 0]));
    }

    function blurPass(label: string, srcView: GPUTextureView, dstView: GPUTextureView): void {
        const bg = device.createBindGroup({ layout: blurPipe!.getBindGroupLayout(0), entries: [{ binding: 0, resource: srcView }, { binding: 1, resource: { buffer: blurBuffer } }] });
        const pass = engine._currentEncoder.beginRenderPass({ label, colorAttachments: [{ view: dstView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
        pass.setPipeline(blurPipe!);
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
            buildSphereBG();
        },
        setMode(m: "surface" | "blit"): void {
            mode = m;
        },
        record(): void {
            build();
        },
        execute(): number {
            const outView = outRT._colorView;
            const bgView = bgRT._colorView;
            const sceneDepth = depthRT._depthView;
            if (!depthPipe || !blitPipe || !compPipe || !sphereBG || !outView || !bgView) {
                return 0;
            }
            const enc = engine._currentEncoder;

            // Sphere-impostor mode: just present the scene (which already has the
            // impostors drawn into it) to the swapchain.
            if (mode === "blit") {
                const bg = device.createBindGroup({ layout: blitPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: bgView }, { binding: 1, resource: sampler }] });
                const pass = enc.beginRenderPass({ label: "fluid-surf-blit", colorAttachments: [{ view: outView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
                pass.setPipeline(blitPipe);
                pass.setBindGroup(0, bg);
                pass.draw(3);
                pass.end();
                return 1;
            }

            if (!thickPipe || !blurPipe || !sceneDepth) {
                return 0;
            }
            allocTargets();
            updateUniforms();

            // 1. Depth + speed.
            {
                const pass = enc.beginRenderPass({
                    label: "fluid-surf-depth",
                    colorAttachments: [{ view: views.depth!, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
                    depthStencilAttachment: { view: sceneDepth, depthLoadOp: "load", depthStoreOp: "store" },
                });
                pass.setPipeline(depthPipe);
                pass.setBindGroup(0, sphereBG);
                pass.draw(6, currentSim.count);
                pass.end();
            }
            // 2. Thickness (additive).
            {
                const pass = enc.beginRenderPass({
                    label: "fluid-surf-thick",
                    colorAttachments: [{ view: views.thick!, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
                    depthStencilAttachment: { view: sceneDepth, depthLoadOp: "load", depthStoreOp: "store" },
                });
                pass.setPipeline(thickPipe);
                pass.setBindGroup(0, sphereBG);
                pass.draw(6, currentSim.count);
                pass.end();
            }
            // 3. Bilateral blur of depth (tight range) and thickness (loose range).
            writeBlur(1, 0, 0.5);
            blurPass("fluid-surf-depthBlurH", views.depth!, views.tmp!);
            writeBlur(0, 1, 0.5);
            blurPass("fluid-surf-depthBlurV", views.tmp!, views.depthBlur!);
            writeBlur(1, 0, 6.0);
            blurPass("fluid-surf-thickBlurH", views.thick!, views.tmp!);
            writeBlur(0, 1, 6.0);
            blurPass("fluid-surf-thickBlurV", views.tmp!, views.thickBlur!);

            // 4. Composite (refraction) → swapchain.
            const compBG = device.createBindGroup({
                layout: compPipe.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: views.depthBlur! },
                    { binding: 1, resource: views.thickBlur! },
                    { binding: 2, resource: bgView },
                    { binding: 3, resource: sampler },
                    { binding: 4, resource: { buffer: compBuffer } },
                ],
            });
            const cpass = enc.beginRenderPass({ label: "fluid-surf-composite", colorAttachments: [{ view: outView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
            cpass.setPipeline(compPipe);
            cpass.setBindGroup(0, compBG);
            cpass.draw(3);
            cpass.end();
            return 1;
        },
        dispose(): void {
            camBuffer.destroy();
            blurBuffer.destroy();
            compBuffer.destroy();
            for (const t of [depthTex, thickTex, tmpTex, depthBlur, thickBlur]) {
                t?.destroy();
            }
        },
    };
}
