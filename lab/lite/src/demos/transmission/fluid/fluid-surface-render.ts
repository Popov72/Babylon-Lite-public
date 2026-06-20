// Screen-space fluid surface renderer for the GPU fluid sim.
//
// An alternative to the sphere-impostor renderer (particle-render.ts) that
// reconstructs a smooth liquid SURFACE in screen space instead of drawing
// discrete spheres — the "Screen Space Fluids" technique (van der Laan 2009,
// also used by matsuoka-601's Splash). Pipeline, all demo-local and encoded as
// one frame-graph Task:
//
//   1. depth+speed : draw the particles as sphere impostors, writing the nearest
//                    eye-space depth (and the surface particle's speed) into an
//                    offscreen RGBA16F target (depth-tested against the shared
//                    scene depth, so the ground occludes the fluid).
//   2. thickness   : draw the impostors again, additively, into an R16F target
//                    (no depth write) — an absorption/opacity proxy.
//   3. blur H / V  : separable bilateral blur of the depth+speed target, edge-
//                    preserving (depth-weighted) so the bumpy sphere depth turns
//                    into a smooth surface without bleeding across silhouettes.
//   4. composite   : a fullscreen pass that reconstructs the eye-space position
//                    and normal from the blurred depth, shades the surface
//                    (Fresnel + specular + thickness absorption) and whitens it
//                    by speed (foam), then alpha-blends over the scene colour.
//
// The offscreen textures are allocated here (not via the frame-graph RT system)
// so the task is fully self-contained; they are re-created on canvas resize.

import { getEffectiveAspectRatio, getViewProjectionMatrix } from "babylon-lite";
import type { Camera, EngineContext, RenderTarget, SceneContext, Task } from "babylon-lite";
import type { FluidSim } from "./pbf-sim.js";

export interface FluidSurfaceOptions {
    /** Final colour target to composite the fluid over (typically `engine.scRT`). */
    colorRT: RenderTarget;
    /** Depth target shared with the scene render task (depth-test for occlusion). */
    depthRT: RenderTarget;
    camera: Camera;
    sim: FluidSim;
}

// Bilateral-blur half-width in texels (per separable pass). Larger = smoother
// surface but more cost; the spatial sigma scales with it.
const BLUR_RADIUS = 14;

// Shared billboard + camera uniform (depth and thickness passes). Mirrors the
// impostor renderer's layout plus a thickness-per-particle scale.
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
    @location(1) @interpolate(flat) viewZc: f32, // eye-space z of the sphere centre
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
    // The LH-perspective clip.w equals the view-space z (linear eye depth).
    o.viewZc = (cam.vp * vec4<f32>(center, 1.0)).w;
    o.uv = c;
    o.speed = clamp(dbg[ii] * cam.misc.y, 0.0, 1.0);
    return o;
}

// Depth+speed: store the sphere-surface eye depth in R and the surface speed in
// G. Depth-test/write uses the billboard-plane clip depth (flat per particle),
// so the nearest particle's surface wins.
@fragment fn fsDepth(i: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.uv, i.uv);
    if (r2 > 1.0) { discard; }
    let nz = sqrt(1.0 - r2);
    let eyeDepth = i.viewZc - cam.misc.x * nz; // nearer than the centre by r*nz
    return vec4<f32>(eyeDepth, i.speed, 0.0, 1.0);
}

// Thickness: a soft additive blob per particle (no depth write). Accumulates an
// absorption proxy — thicker fluid is more opaque / more saturated.
@fragment fn fsThick(i: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.uv, i.uv);
    if (r2 > 1.0) { discard; }
    return vec4<f32>(cam.misc.z * (1.0 - r2), 0.0, 0.0, 1.0);
}`;

// Fullscreen triangle (no vertex buffer); fragment uses @builtin(position).
const FULLSCREEN_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}`;

// Separable bilateral blur of the depth (R) + speed (G) target. Depth-weighted so
// the surface stays sharp at silhouettes. `dir` is (1,0) or (0,1) in texels.
const BLUR_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Blur { dir: vec4<f32> }; // xy = texel direction, z = 1/(2*rangeSigma^2)
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

// Composite: reconstruct eye-space position + normal from the blurred depth,
// shade a water surface, whiten by speed (foam), absorb by thickness, and
// alpha-blend over the scene colour already in the swapchain.
const COMPOSITE_WGSL = /* wgsl */ `
${FULLSCREEN_VS}
struct Comp {
    res: vec4<f32>,    // xy = render size (px), zw = 1/size
    proj: vec4<f32>,   // x = tanHalfFov, y = aspect, z/w unused
    light: vec4<f32>,  // xyz = view-space light dir, w = foam strength
    tint: vec4<f32>,   // xyz = deep water colour, w = absorption coeff
};
@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var thickTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> c: Comp;

// Reconstruct view-space position from a pixel + its stored eye depth.
fn viewPos(pix: vec2<i32>, d: f32) -> vec3<f32> {
    let ndcX = (f32(pix.x) + 0.5) * c.res.z * 2.0 - 1.0;
    let ndcY = 1.0 - (f32(pix.y) + 0.5) * c.res.w * 2.0; // flip: +Y up
    let vx = ndcX * d * c.proj.y * c.proj.x; // ndc.x * z * aspect * tanHalfFov
    let vy = ndcY * d * c.proj.x;            // ndc.y * z * tanHalfFov
    return vec3<f32>(vx, vy, d);
}

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(floor(pos.xy));
    let dim = vec2<i32>(textureDimensions(depthTex));
    let d0 = textureLoad(depthTex, pix, 0).r;
    if (d0 <= 0.0) { discard; }

    let p0 = viewPos(pix, d0);

    // Normal from screen-space finite differences, picking the smaller one-sided
    // difference on each axis so silhouettes don't smear the normal.
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
    // The camera looks down +z (LH); orient the normal toward the eye (-z side).
    if (n.z > 0.0) { n = -n; }

    let V = normalize(-p0);              // surface → eye
    let L = normalize(c.light.xyz);
    let H = normalize(L + V);
    let ndl = max(dot(n, L), 0.0);
    let spec = pow(max(dot(n, H), 0.0), 80.0);
    let fres = pow(1.0 - max(dot(n, V), 0.0), 5.0); // Schlick

    // Absorption: thicker fluid trends to the deep tint (Beer-Lambert-ish).
    let thick = textureLoad(thickTex, pix, 0).r;
    let absorb = 1.0 - exp(-c.tint.w * thick);
    var col = mix(vec3<f32>(0.55, 0.78, 0.92), c.tint.xyz, absorb);
    col = col * (0.45 + 0.55 * ndl);
    col = col + vec3<f32>(0.85, 0.92, 1.0) * (fres * 0.5); // reflective rim
    col = col + vec3<f32>(1.0) * (spec * 0.9);             // sharp highlight

    // Speed → foam: fast-moving surface whitens (matches the sphere renderer's
    // speed-tint, here as screen-space foam).
    let speed = textureLoad(depthTex, pix, 0).g;
    let foam = clamp(speed * c.light.w, 0.0, 1.0);
    col = mix(col, vec3<f32>(0.95, 0.97, 1.0), foam);

    // Opacity: thin films and grazing angles are translucent; thick/foamy fluid
    // is near-opaque so the scene shows through the edges only.
    let alpha = clamp(0.35 + 0.5 * absorb + 0.35 * fres + 0.4 * foam, 0.0, 1.0);
    return vec4<f32>(col, alpha);
}`;

export function createFluidSurfaceTask(
    engine: EngineContext,
    scene: SceneContext,
    opts: FluidSurfaceOptions,
): Task & { setSim(s: FluidSim): void; setEnabled(on: boolean): void; enabled: boolean } {
    const device = engine._device;
    const { colorRT, depthRT, camera } = opts;
    let currentSim = opts.sim;
    let enabled = false;

    // ── Camera/billboard uniform (depth + thickness passes) ──────────
    const camData = new Float32Array(28); // mat4(16) + right(4) + up(4) + misc(4)
    const camBuffer = device.createBuffer({ label: "fluid-surf-cam", size: camData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blurBuffer = device.createBuffer({ label: "fluid-surf-blur", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const compBuffer = device.createBuffer({ label: "fluid-surf-comp", size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // ── Offscreen targets (allocated lazily, re-created on resize) ────
    let width = 0;
    let height = 0;
    let depthTex: GPUTexture | null = null; // RGBA16F: R eye-depth, G speed
    let blurTexH: GPUTexture | null = null;
    let blurTexV: GPUTexture | null = null;
    let thickTex: GPUTexture | null = null; // R16F additive thickness
    let depthView: GPUTextureView | null = null;
    let blurViewH: GPUTextureView | null = null;
    let blurViewV: GPUTextureView | null = null;
    let thickView: GPUTextureView | null = null;

    function allocTargets(): void {
        const w = engine.canvas.width;
        const h = engine.canvas.height;
        if (w === width && h === height && depthTex) {
            return;
        }
        for (const t of [depthTex, blurTexH, blurTexV, thickTex]) {
            t?.destroy();
        }
        width = w;
        height = h;
        const colorUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
        depthTex = device.createTexture({ label: "fluid-surf-depth", size: { width: w, height: h }, format: "rgba16float", usage: colorUsage });
        blurTexH = device.createTexture({ label: "fluid-surf-blurH", size: { width: w, height: h }, format: "rgba16float", usage: colorUsage });
        blurTexV = device.createTexture({ label: "fluid-surf-blurV", size: { width: w, height: h }, format: "rgba16float", usage: colorUsage });
        thickTex = device.createTexture({ label: "fluid-surf-thick", size: { width: w, height: h }, format: "r16float", usage: colorUsage });
        depthView = depthTex.createView();
        blurViewH = blurTexH.createView();
        blurViewV = blurTexV.createView();
        thickView = thickTex.createView();
    }

    // ── Pipelines ────────────────────────────────────────────────────
    let depthPipe: GPURenderPipeline | null = null;
    let thickPipe: GPURenderPipeline | null = null;
    let blurPipe: GPURenderPipeline | null = null;
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
        // Shared explicit layout so the single sphere bind group is valid for both
        // the depth and thickness pipelines (auto-layouts would be distinct).
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
        thickPipe = device.createRenderPipeline({
            label: "fluid-surf-thick",
            layout: spherePL,
            vertex: { module: sphereMod, entryPoint: "vs" },
            fragment: {
                module: sphereMod,
                entryPoint: "fsThick",
                targets: [{
                    format: "r16float",
                    blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } },
                }],
            },
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
        const compMod = device.createShaderModule({ label: "fluid-surf-comp", code: COMPOSITE_WGSL });
        compPipe = device.createRenderPipeline({
            label: "fluid-surf-comp",
            layout: "auto",
            vertex: { module: compMod, entryPoint: "vs" },
            fragment: {
                module: compMod,
                entryPoint: "fs",
                targets: [{
                    format: engine.format,
                    blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" } },
                }],
            },
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
        camData[24] = currentSim.particleRadius * 1.7; // slightly fatten to close gaps between particles
        camData[25] = currentSim.debugNorm;
        camData[26] = 0.16; // thickness contribution per particle
        camData[27] = 0;
        device.queue.writeBuffer(camBuffer, 0, camData);

        // Composite uniform.
        const tanHalfFov = Math.tan(camera.fov * 0.5);
        const comp = new Float32Array(16);
        comp[0] = width; comp[1] = height; comp[2] = 1 / width; comp[3] = 1 / height;
        comp[4] = tanHalfFov; comp[5] = aspect; comp[6] = 0; comp[7] = 0;
        // View-space light (roughly over-the-shoulder) and foam strength.
        comp[8] = 0.3; comp[9] = 0.5; comp[10] = -0.8; comp[11] = 1.1;
        comp[12] = 0.06; comp[13] = 0.18; comp[14] = 0.4; comp[15] = 0.9; // deep tint + absorption
        device.queue.writeBuffer(compBuffer, 0, comp);
    }

    function writeBlur(dirX: number, dirY: number): void {
        // rangeSigma in eye-depth units; 1/(2σ²) packed into z.
        const rangeSigma = 0.5;
        device.queue.writeBuffer(blurBuffer, 0, new Float32Array([dirX, dirY, 1 / (2 * rangeSigma * rangeSigma), 0]));
    }

    function fsPass(label: string, pipe: GPURenderPipeline, bg: GPUBindGroup, view: GPUTextureView): void {
        const pass = engine._currentEncoder.beginRenderPass({
            label,
            colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
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
        get enabled() {
            return enabled;
        },
        set enabled(v: boolean) {
            enabled = v;
        },
        setEnabled(on: boolean): void {
            enabled = on;
        },
        setSim(s: FluidSim): void {
            currentSim = s;
            buildSphereBG();
        },
        record(): void {
            build();
        },
        execute(): number {
            if (!enabled) {
                return 0;
            }
            const colorView = colorRT._colorView;
            const sceneDepth = depthRT._depthView;
            if (!depthPipe || !thickPipe || !blurPipe || !compPipe || !sphereBG || !colorView || !sceneDepth) {
                return 0;
            }
            allocTargets();
            updateUniforms();
            const enc = engine._currentEncoder;

            // 1. Depth + speed (depth-tested against the shared scene depth).
            {
                const pass = enc.beginRenderPass({
                    label: "fluid-surf-depth",
                    colorAttachments: [{ view: depthView!, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
                    depthStencilAttachment: { view: sceneDepth, depthLoadOp: "load", depthStoreOp: "store" },
                });
                pass.setPipeline(depthPipe);
                pass.setBindGroup(0, sphereBG);
                pass.draw(6, currentSim.count);
                pass.end();
            }
            // 2. Thickness (additive; depth-test, no write).
            {
                const pass = enc.beginRenderPass({
                    label: "fluid-surf-thick",
                    colorAttachments: [{ view: thickView!, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
                    depthStencilAttachment: { view: sceneDepth, depthLoadOp: "load", depthStoreOp: "store" },
                });
                pass.setPipeline(thickPipe);
                pass.setBindGroup(0, sphereBG);
                pass.draw(6, currentSim.count);
                pass.end();
            }
            // 3. Separable bilateral blur (H: depth → blurH, V: blurH → blurV).
            writeBlur(1, 0);
            const blurBGH = device.createBindGroup({ layout: blurPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: depthView! }, { binding: 1, resource: { buffer: blurBuffer } }] });
            fsPass("fluid-surf-blurH", blurPipe, blurBGH, blurViewH!);
            writeBlur(0, 1);
            const blurBGV = device.createBindGroup({ layout: blurPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: blurViewH! }, { binding: 1, resource: { buffer: blurBuffer } }] });
            fsPass("fluid-surf-blurV", blurPipe, blurBGV, blurViewV!);

            // 4. Composite over the scene colour (alpha-blended).
            const compBG = device.createBindGroup({
                layout: compPipe.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: blurViewV! },
                    { binding: 1, resource: thickView! },
                    { binding: 2, resource: { buffer: compBuffer } },
                ],
            });
            const cpass = enc.beginRenderPass({
                label: "fluid-surf-composite",
                colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
            });
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
            for (const t of [depthTex, blurTexH, blurTexV, thickTex]) {
                t?.destroy();
            }
        },
    };
}
