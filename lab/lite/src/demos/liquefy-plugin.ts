// Liquefy material plugin — clips the solid foe to reveal the single scene water render, with a
// NOISY dissolve boundary and a glowing fire-line gradient (blue/red/yellow) at the separation.
//
// The fluid surface is rendered exactly once by the main fluid-surface task into the swapchain.
// This plugin discards fragments inside a growing world-space front so the already-rendered water
// beneath shows through. The front is perturbed by value noise (so the water/solid boundary is
// irregular, not a perfect sphere), and fragments in a band just OUTSIDE the front are tinted with
// a fire gradient so the dissolving edge glows.

import type { MaterialPlugin } from "babylon-lite";

/** Mutable per-foe liquefaction state, read every UBO write. The host bumps the material's UBO
 *  version each frame while animating so `writeUbo` re-runs with the current `frontR`. */
export interface LiquefyState {
    /** World-space hit point the front grows from. */
    hit: readonly [number, number, number];
    /** Current front radius (world units); 0 = untouched, ≥ mesh diameter = fully clipped. */
    frontR: number;
    /** Fire-gradient band width (world units) just outside the front. */
    edge: number;
    /** Master enable; when false the plugin never discards or tints. */
    enabled: boolean;
    /** World-space amplitude of the boundary noise (default 0.35). */
    noiseAmp?: number;
    /** World-space frequency of the boundary noise (default 1.2). */
    noiseFreq?: number;
}

/** Create a liquefy clip plugin bound to a mutable state getter.
 *  @param host - "pbr" (default) or "std"; selects the world-pos varying + uniform-accessor names.
 *    On Standard materials the plugin is `dynamic` so its self-managed UBO is re-uploaded per frame. */
export function createLiquefyPlugin(getState: () => LiquefyState, host: "pbr" | "std" = "pbr"): MaterialPlugin {
    const wp = host === "pbr" ? "input.worldPos" : "input.vp";
    const u = host === "pbr" ? "material" : "pluginUbo";

    // Helper functions + a per-fragment private carrying the signed distance beyond the (noisy)
    // front from the alpha-test slot to the colour slot. 1e9 = "no fire band" (disabled / far).
    const definitions = `
var<private> lqEdgeDist: f32 = 1.0e9;
fn lqHash13(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
fn lqNoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let n000 = lqHash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = lqHash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = lqHash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = lqHash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = lqHash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = lqHash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = lqHash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = lqHash13(i + vec3<f32>(1.0, 1.0, 1.0));
    let nx00 = mix(n000, n100, w.x);
    let nx10 = mix(n010, n110, w.x);
    let nx01 = mix(n001, n101, w.x);
    let nx11 = mix(n011, n111, w.x);
    return mix(mix(nx00, nx10, w.y), mix(nx01, nx11, w.y), w.z);
}
fn lqFbm(p: vec3<f32>) -> f32 {
    return lqNoise(p) * 0.65 + lqNoise(p * 2.03 + vec3<f32>(11.7, 3.1, 7.9)) * 0.35;
}
fn lqFire(t: f32) -> vec3<f32> {
    let yellow = vec3<f32>(1.0, 0.9, 0.25);
    let red = vec3<f32>(0.95, 0.16, 0.05);
    let blue = vec3<f32>(0.12, 0.32, 0.95);
    let hot = mix(yellow, red, smoothstep(0.0, 0.5, t));
    return mix(hot, blue, smoothstep(0.5, 1.0, t));
}`;

    // Alpha-test slot: perturb the front by noise, discard inside it, record the edge distance.
    const clip = `
if (${u}.lqParams.z > 0.5) {
    let lqN = (lqFbm(${wp} * ${u}.lqParams.w) - 0.5) * 2.0 * ${u}.lqParams.y;
    let lqDist = distance(${wp}, ${u}.lqHitR.xyz) + lqN;
    if (lqDist < ${u}.lqHitR.w) { discard; }
    lqEdgeDist = lqDist - ${u}.lqHitR.w;
}`;

    // Colour slot (after tonemap+gamma): tint the fire band. Colour is vec3 on PBR, vec4 on Standard.
    const tintExpr = host === "pbr" ? "color = mix(color, lqFire(lqT), lqGlow);" : "color = vec4<f32>(mix(color.rgb, lqFire(lqT), lqGlow), color.a);";
    const tint = `
if (lqEdgeDist < ${u}.lqParams.x) {
    let lqT = clamp(lqEdgeDist / max(${u}.lqParams.x, 1.0e-4), 0.0, 1.0);
    let lqGlow = (1.0 - lqT) * (1.0 - lqT);
    ${tintExpr}
}`;

    return {
        name: "liquefy",
        dynamic: true,
        getUniforms: () => ({
            ubo: [
                { name: "lqHitR", type: "vec4<f32>" },
                { name: "lqParams", type: "vec4<f32>" },
            ],
        }),
        getCustomCode: (shaderType) =>
            shaderType === "fragment"
                ? {
                      CUSTOM_FRAGMENT_DEFINITIONS: definitions,
                      CUSTOM_FRAGMENT_UPDATE_ALPHA: clip,
                      CUSTOM_FRAGMENT_BEFORE_FRAGCOLOR: tint,
                  }
                : null,
        writeUbo: (data, offsets) => {
            const s = getState();
            const h = (offsets.get("lqHitR") ?? 0) / 4;
            data[h] = s.hit[0];
            data[h + 1] = s.hit[1];
            data[h + 2] = s.hit[2];
            data[h + 3] = s.frontR;
            const p = (offsets.get("lqParams") ?? 0) / 4;
            data[p] = s.edge; // fire-band width
            data[p + 1] = s.noiseAmp ?? 0.35; // noise amplitude
            data[p + 2] = s.enabled ? 1 : 0;
            data[p + 3] = s.noiseFreq ?? 1.2; // noise frequency
        },
    };
}
