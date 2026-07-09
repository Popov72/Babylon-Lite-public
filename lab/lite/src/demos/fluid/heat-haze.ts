// Heat-haze controller for the Transmission demo.
//
// Heat distortion is animated screen-space refraction. A transmissive PBR
// surface refracts the frame-graph "scene-color grab" through its surface
// normal; perturbing that normal with a scrolling noise normal map — and
// advancing the scroll every frame — produces a believable, time-varying
// shimmer that is:
//   • localized to wherever the surface is drawn (place a quad over the heat
//     source), and
//   • correctly occluded, because the surface is real depth-tested geometry
//     (it depth-tests against the opaque scene but does not write depth), so a
//     foreground object hides the haze behind it per-fragment — no stencil.
//
// The refraction core (packages/.../pbr/fragments/refraction-rtt-fragment.ts)
// samples the grabbed scene-color at a UV derived from refract(-V, N, ior),
// where N is the normal-mapped surface normal — so scrolling the normal map
// warps the background over time.

import { markMaterialUboDirty, onBeforeRender } from "babylon-lite";
import type { PbrMaterialProps, SceneContext } from "babylon-lite";

export interface HeatHazeOptions {
    /** Normal-map scroll speed in UV units per second. Hot air rises, so bias
     *  the vertical (y / V) component well above the horizontal. Default
     *  `{ x: 0.015, y: 0.12 }`. */
    speed?: { x: number; y: number };
}

/**
 * Animate a transmissive PBR material's normal map so its refraction shimmers
 * like rising hot air.
 *
 * Requirements on `material`:
 *   • `transmissive: true` and `subsurface.refraction.intensity > 0`
 *     (so the frame-graph refraction path is active — also enable the scene
 *     transmission copy: `getFrameGraph(scene)._tasks[0]._config.transmission`).
 *   • a tangent-space noise `normalTexture` whose UV transform is COMPILED in,
 *     i.e. created with `cloneTexture2D(noise, { _hasTx: true })`. The
 *     `uOffset` / `vOffset` fields are build-time, but once the UV-transform
 *     path is compiled the engine re-reads them whenever the material UBO is
 *     marked dirty — so per-frame scrolling needs no pipeline rebuild, just a
 *     `markMaterialUboDirty()`.
 *
 * Keep the IOR near 1 (≈1.05–1.2), the thickness small, and the noise smooth
 * and low-frequency for a gentle, believable haze. For a richer plume, drive
 * two stacked quads with opposite scroll directions/speeds.
 */
export function startHeatHaze(scene: SceneContext, material: PbrMaterialProps, options: HeatHazeOptions = {}): void {
    const speedX = options.speed?.x ?? 0.015;
    const speedY = options.speed?.y ?? 0.12;
    const tex = material.normalTexture;
    if (!tex) {
        throw new Error("startHeatHaze: material has no normalTexture to scroll.");
    }
    onBeforeRender(scene, (deltaMs: number) => {
        const dt = deltaMs / 1000;
        tex.uOffset = (tex.uOffset ?? 0) + speedX * dt;
        tex.vOffset = (tex.vOffset ?? 0) + speedY * dt;
        markMaterialUboDirty(material);
    });
}
