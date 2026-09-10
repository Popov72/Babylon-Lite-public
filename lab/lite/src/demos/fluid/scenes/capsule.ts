// Capsule demo — a vertical glass pill floating above the ground. The liquid is
// confined to its interior; LMB (over the tank) / Space punch drain holes the
// fluid escapes through, R refills. The rounded boundary avoids the flat-wall
// lattice artefacts of a box.

import { addToScene, createCsgFromMesh, createCylinder, createMeshFromCsg, createSphere, createStandardMaterial, csgUnion, setMeshVisible } from "babylon-lite";
import type { FluidFlowConfig, Mesh, SceneSdfSpec } from "babylon-lite";
import { disposeMeshGpu } from "babylon-lite";
import type { FluidCtx, FluidDemo } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";
import { pickCapsuleHole } from "../pick.js";

// Capsule tank: a vertical pill (bottom hemisphere centre at y=5, radius 3 →
// bottom at y=2). These also seed the sims' built-in fallback confinement in the
// core's createSims (a legacy default; the injected sceneSdf always overrides).
export const CAP_A: [number, number, number] = [0, 5, 0];
export const CAP_B: [number, number, number] = [0, 11, 0];
export const CAP_R = 3;

const HOLE_RADIUS = 0.4;

// Wall shell half-thickness (world units). The glass is modelled as a SOLID SHELL
// of this half-thickness centred on the visible capsule surface, so the wall is
// thick enough that particles can't tunnel through it in a single sim step.
const WALL_HALF = 0.6;

// Radius of the VISIBLE glass mesh only (collision + hole picking stay on CAP_R).
// Interior fluid particle CENTERS settle near the inner cavity radius
// CAP_R - WALL_HALF = 2.4 and the rendered screen-space surface reaches a little
// beyond that, so the glass built at the full CAP_R = 3 left a visible gap. Pull
// the mesh in to just past the settled surface (small deliberate gap, no clipping
// of the fluid outside the silhouette). The collision SDF, pickCapsuleHole and the
// Space-key hole placement all keep using CAP_R, so drain bores (centred on the
// wall at CAP_R and streaming outward) still cross this smaller glass naturally.
const VISUAL_R = CAP_R - WALL_HALF + 0.15; // ≈ 2.55

export function createCapsuleDemo(ctx: FluidCtx): FluidDemo {
    const { engine } = ctx;

    // Drain-hole radius, tunable via the Demo-parameters UI (LMB / Space use it).
    let holeRadius = HOLE_RADIUS;

    // Per-demo scene SDF: the glass is a SOLID SHELL of half-thickness WALL_HALF
    // centred on the visible capsule surface, so fluid can exist on BOTH sides —
    // the interior cavity AND the exterior/ground. Positive inside the fluid domain.
    const sdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { a: vec4<f32>, b: vec4<f32>, holes: array<vec4<f32>, 8>, };",
        // Unified SDF: the glass is a SOLID SHELL of half-thickness WALL_HALF centred on the
        // visible capsule surface, so fluid can exist on BOTH sides — the interior cavity AND
        // the exterior/ground — separated by a wall thick enough that particles can't tunnel
        // through in one step. Fluid domain (d>0) = above the ground AND outside the wall shell.
        // Drain holes carve the wall (union free space); the hole is centred on the shell
        // centreline so any hole radius opens a through-passage. This unifies interior+ground
        // into one field, so drained fluid is confined by the SAME SDF — no per-particle
        // "escaped" state, no separate floor catch, no suck-back.
        sdf: `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let a = sceneSdfParams.a.xyz;
    let r = sceneSdfParams.a.w;
    let b = sceneSdfParams.b.xyz;
    let groundY = sceneSdfParams.b.w;
    let ba = b - a;
    let hh = clamp(dot(pt - a, ba) / dot(ba, ba), 0.0, 1.0);
    let dCap = length(pt - (a + ba * hh)) - r;   // signed dist to capsule surface (neg inside)
    let wall = abs(dCap) - ${WALL_HALF.toFixed(3)}; // <0 inside the shell [r-WALL_HALF, r+WALL_HALF]
    var d = min(pt.y - groundY, wall);
    // Drain holes: carve a finite cylindrical BORE through the wall, aligned with the
    // wall normal at the hole point, so the opening is a clean circle regardless of wall
    // thickness (a sphere only touches a thick wall's far face tangentially → fluid pools
    // in a spherical pocket and bulges out; a bore streams through cleanly). hole.w = bore
    // radius (opening size); hole.xyz = pick point on the visible surface.
    for (var k = 0u; k < 8u; k = k + 1u) {
        let hole = sceneSdfParams.holes[k];
        if (hole.w > 0.0) {
            let hp = hole.xyz;
            let hk = clamp(dot(hp - a, ba) / dot(ba, ba), 0.0, 1.0);
            let radDir = normalize(hp - (a + ba * hk)); // outward wall normal at the hole
            let lv = pt - hp;
            let axial = dot(lv, radDir);
            let perp = length(lv - axial * radDir);
            // half-length spans the full wall (plus a small margin into the already-free
            // interior/exterior); flat caps overlap free space, so no bulge.
            let bore = min(${(WALL_HALF + 0.25).toFixed(3)} - abs(axial), hole.w - perp);
            d = max(d, bore);
        }
    }
    return d;
}`,
        gridConfine: false,
        buffer: ctx.sceneSdfBuffer,
    };

    // Transparent glass shell built as a single watertight pill: a cylinder body
    // CSG-unioned with a hemispherical cap at each end (so it's a true capsule, not
    // three overlapping meshes with doubled translucency at the seams). CSG bakes
    // each source mesh's world transform, so the parts are positioned before
    // conversion and the result already sits in world space. Alpha < 1 makes the
    // standard material blend and skip depth writes so the liquid stays visible.
    const glass = createStandardMaterial();
    glass.diffuseColor = [0.5, 0.66, 0.82];
    glass.specularColor = [0.8, 0.85, 0.95];
    glass.alpha = 0.15;

    const body = createCylinder(engine, { height: CAP_B[1] - CAP_A[1], diameter: 2 * VISUAL_R, tessellation: 48 });
    body.position.set(0, (CAP_A[1] + CAP_B[1]) / 2, 0);
    const capBottom = createSphere(engine, { diameter: 2 * VISUAL_R, segments: 32 });
    capBottom.position.set(CAP_A[0], CAP_A[1], CAP_A[2]);
    const capTop = createSphere(engine, { diameter: 2 * VISUAL_R, segments: 32 });
    capTop.position.set(CAP_B[0], CAP_B[1], CAP_B[2]);
    const capsuleSolid = csgUnion(csgUnion(createCsgFromMesh(body), createCsgFromMesh(capBottom)), createCsgFromMesh(capTop));
    const capsule = createMeshFromCsg(engine, capsuleSolid, "capsule");
    capsule.material = glass;
    addToScene(ctx.scene, capsule);
    setMeshVisible(capsule, false);
    // The source parts were only scaffolding for the union — free their GPU buffers.
    for (const part of [body, capBottom, capTop]) {
        disposeMeshGpu(part);
    }
    const shell: Mesh[] = [capsule];
    let containerVisible = true; // toggled by the "Show container mesh" UI checkbox
    const flow = (): FluidFlowConfig => ({
        emitters: [
            {
                id: "capsule-fill",
                name: "Initial capsule fill",
                enabled: true,
                behavior: "initial",
                transform: { position: [0, (CAP_A[1] + CAP_B[1]) / 2, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: {
                    type: "capsule",
                    radius: CAP_R - WALL_HALF - 0.2,
                    height: CAP_B[1] - CAP_A[1] + 2 * (CAP_R - WALL_HALF - 0.2),
                },
                sampling: "volume",
                velocity: [0, 0, 0],
                velocitySpace: "world",
                spread: 0,
            },
        ],
        sinks: [],
    });

    return {
        key: "capsule",
        label: "Capsule (drainable)",
        helperText: "Capsule: LMB on tank punches hole · Space random hole",
        envUrl: ENV_STUDIO_URL,
        sdf,
        writeSdfParams(): void {
            // a = (centre, radius), b = (centre, groundY). The hole sub-block
            // (offset 32) is owned by the core's hole ring.
            engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, new Float32Array([CAP_A[0], CAP_A[1], CAP_A[2], CAP_R, CAP_B[0], CAP_B[1], CAP_B[2], 0]));
        },
        flow,
        onEnter(): void {
            for (const m of shell) {
                setMeshVisible(m, containerVisible);
            }
            setMeshVisible(ctx.ground, true);
        },
        onLeave(): void {
            for (const m of shell) {
                setMeshVisible(m, false);
            }
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            for (const m of shell) {
                setMeshVisible(m, v);
            }
        },
        containerMeshes(): Mesh[] {
            // The glass pill is translucent (skips depth writes) so it must be drawn
            // as a post-fluid overlay (see fluid.ts) rather than into the offscreen
            // scene-colour target, or the fluid surface composites OVER it.
            return shell;
        },
        update(): void {
            /* static tank — nothing per-frame */
        },
        demoParams() {
            return [{ key: "holeRadius", label: "Hole radius", type: "number", min: 0.1, max: 1.5, step: 0.05, value: holeRadius }];
        },
        applyParam(key: string, value: number | boolean | string): void {
            if (key === "holeRadius") {
                holeRadius = value as number;
            }
        },
        extraControls() {
            return [];
        },

        claimsPointer(e: PointerEvent): boolean {
            // LMB over the tank wall belongs to the demo (punch a hole), so tell the
            // camera to ignore it instead of rotating. Any other button/miss rotates.
            if (e.button !== 0) {
                return false;
            }
            const rect = ctx.canvas.getBoundingClientRect();
            const hit = pickCapsuleHole(ctx.viewProjection(), e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height, CAP_A, CAP_B, CAP_R);
            return !!hit;
        },
        onPointerDown(e: PointerEvent): void {
            // LMB punches a hole where the cursor hits the tank wall. Re-pick
            // independently (cheap) rather than caching claimsPointer's result.
            if (e.button !== 0) {
                return;
            }
            const rect = ctx.canvas.getBoundingClientRect();
            const hit = pickCapsuleHole(ctx.viewProjection(), e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height, CAP_A, CAP_B, CAP_R);
            if (hit) {
                ctx.addSceneHole(hit, holeRadius);
            }
        },
        onKey(e: KeyboardEvent): void {
            // Space punches a random hole around the tank wall.
            if (e.code === "Space") {
                e.preventDefault();
                const theta = Math.random() * Math.PI * 2;
                const y = CAP_A[1] + Math.random() * 1.5;
                ctx.addSceneHole([CAP_R * Math.cos(theta), y, CAP_R * Math.sin(theta)], holeRadius);
            }
        },
    };
}
