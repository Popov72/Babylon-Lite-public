// Fountain demo — a wide cylindrical basin confining a pool of water; a central
// jet plus a ring of angled jets recirculate it. A pump intake across the floor
// throttles how much settled water is relaunched, keeping most of it pooled.

import { addToScene, createCylinder, createStandardMaterial, setMeshVisible } from "babylon-lite";
import type { Mesh } from "babylon-lite";
import type { EmitterConfig, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import type { DemoParam, FluidCtx, FluidDemo, PairState } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";

const FOUNTAIN_R = 8;
const FOUNTAIN_FLOOR = 0;
const FOUNTAIN_RING_N = 8;
const FOUNTAIN_RING_R = 2.6; // ring-nozzle distance from centre
const FOUNTAIN_CENTRAL_Y = 3.4;
const FOUNTAIN_RING_Y = 2.8;
// Basin-filling block that settles (no clumped floor seed → no start explosion).
const FOUNTAIN_SPAWN_MIN: [number, number, number] = [-5.5, 0.3, -5.5];
const FOUNTAIN_SPAWN_MAX: [number, number, number] = [5.5, 4.5, 5.5];
const NOZZLE_H = 0.6;

// Quaternion rotating +Y onto a (normalised) direction, for orienting nozzles.
function quatFromY(dir: [number, number, number]): [number, number, number, number] {
    let ax = dir[2];
    const ay = 0;
    let az = -dir[0]; // cross(+Y, dir)
    const axisLen = Math.hypot(ax, ay, az);
    if (axisLen < 1e-6) {
        return dir[1] >= 0 ? [0, 0, 0, 1] : [1, 0, 0, 0]; // parallel to ±Y
    }
    ax /= axisLen;
    az /= axisLen;
    const angle = Math.acos(Math.max(-1, Math.min(1, dir[1])));
    const s = Math.sin(angle / 2);
    return [ax * s, ay * s, az * s, Math.cos(angle / 2)];
}

export function createFountainDemo(ctx: FluidCtx): FluidDemo {
    const { engine } = ctx;

    // Wide cylindrical basin confining the water: min(radial wall, floor).
    const sdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { basin: vec4<f32>, };", // basin.x = radius, basin.y = floorY
        sdf: `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    return min(sceneSdfParams.basin.x - length(pt.xz), pt.y - sceneSdfParams.basin.y);
}`,
        buffer: ctx.sceneSdfBuffer,
    };

    // Live-tweakable fountain params (exposed in the Demo-parameters UI).
    const fountainParams = { centralSpeed: 11, ringSpeed: 9, ringOut: 1.0, rate: 0.45, spread: 0.4, centralRadius: 0.15, ringRadius: 0.12 };

    const buildFountainEmitters = (): EmitterConfig["emitters"] => {
        const list: EmitterConfig["emitters"] = [{ pos: [0, FOUNTAIN_CENTRAL_Y, 0], dir: [0, 1, 0], speed: fountainParams.centralSpeed, radius: fountainParams.centralRadius }];
        for (let k = 0; k < FOUNTAIN_RING_N; k++) {
            const th = (k / FOUNTAIN_RING_N) * Math.PI * 2;
            const ox = Math.cos(th);
            const oz = Math.sin(th);
            const dx = ox * fountainParams.ringOut;
            const dz = oz * fountainParams.ringOut;
            const len = Math.hypot(dx, 1.1, dz);
            list.push({ pos: [ox * FOUNTAIN_RING_R, FOUNTAIN_RING_Y, oz * FOUNTAIN_RING_R], dir: [dx / len, 1.1 / len, dz / len], speed: fountainParams.ringSpeed, radius: fountainParams.ringRadius });
        }
        return list;
    };

    const fountainConfig = (): EmitterConfig => ({
        emitters: buildFountainEmitters(),
        // Pump intake: a thin slab across the basin floor. Settled water is pulled
        // up a jet (throttled by `rate` — high enough to feed the jets, low enough
        // that most water stays pooled).
        intakeMin: [-FOUNTAIN_R, FOUNTAIN_FLOOR, -FOUNTAIN_R],
        intakeMax: [FOUNTAIN_R, FOUNTAIN_FLOOR + 0.8, FOUNTAIN_R],
        rate: fountainParams.rate,
        spread: fountainParams.spread,
    });

    // Visible nozzle spouts (dark metal cylinders pointing along each jet).
    // Created at unit diameter (radius 0.5) so the bore can be scaled to the
    // emitter radius (mesh radius == emission radius).
    const nozzleMat = createStandardMaterial();
    nozzleMat.diffuseColor = [0.16, 0.17, 0.2];
    nozzleMat.specularColor = [0.4, 0.4, 0.45];
    const nozzles: Mesh[] = [];
    for (let k = 0; k < FOUNTAIN_RING_N + 1; k++) {
        const m = createCylinder(engine, { height: NOZZLE_H, diameter: 1, tessellation: 14 });
        m.material = nozzleMat;
        addToScene(ctx.scene, m);
        setMeshVisible(m, false);
        nozzles.push(m);
    }
    const updateFountainNozzles = (): void => {
        const es = buildFountainEmitters();
        for (let k = 0; k < nozzles.length; k++) {
            const e = es[k]!;
            const m = nozzles[k]!;
            // Scale the unit-diameter bore to the emitter radius (X/Z only; height fixed).
            m.scaling.set(2 * e.radius, 1, 2 * e.radius);
            // Sink the spout so its tip sits at the emit point.
            m.position.set(e.pos[0] - e.dir[0] * (NOZZLE_H / 2), e.pos[1] - e.dir[1] * (NOZZLE_H / 2), e.pos[2] - e.dir[2] * (NOZZLE_H / 2));
            const q = quatFromY(e.dir);
            m.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
        }
    };
    updateFountainNozzles();
    let containerVisible = true; // toggled by the "Show container / nozzle meshes" UI checkbox

    // Curated first-visit presets (merged over the core defaults on first visit).
    const presets: Record<string, Partial<PairState>> = {
        PBF: {
            schema: { gravity: 17, viscosity: 1, relaxation: 209, scorr: 0, iterations: 1, restDensity: 600, boundaryDensity: 0 },
            demoParams: { centralSpeed: 10.5, ringSpeed: 5.5, ringOut: 0.9, rate: 0.1, spread: 0.5, centralRadius: 0.15, ringRadius: 0.12 },
            color: "#bfe9f3",
            half: true,
            size: 0.6,
            physScale: 0.8,
            count: 150000,
        },
        "MLS-MPM": {
            schema: { gravity: 23, stiffness: 420, viscosity: 0.03, restDensity: 16, damping: 0.998, affineDamping: 0.7, groundDamp: 0.85, groundDampHeight: 1, restitution: 1, substeps: 2 },
            demoParams: { centralSpeed: 18, ringSpeed: 5.5, ringOut: 0.9, rate: 0.05, spread: 0.5, centralRadius: 0.15, ringRadius: 0.12 },
            color: "#bfe9f3",
            half: true,
            size: 0.6,
            physScale: 0.8,
            count: 200000,
        },
    };

    return {
        key: "fountain",
        label: "Fountain (jets)",
        envUrl: ENV_STUDIO_URL,
        sdf,
        writeSdfParams(): void {
            engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, new Float32Array([FOUNTAIN_R, FOUNTAIN_FLOOR, 0, 0]));
        },
        spawn() {
            return { min: FOUNTAIN_SPAWN_MIN, max: FOUNTAIN_SPAWN_MAX };
        },
        emitters() {
            return fountainConfig();
        },
        onEnter(): void {
            updateFountainNozzles();
            for (const m of nozzles) {
                setMeshVisible(m, containerVisible);
            }
            setMeshVisible(ctx.ground, true);
        },
        onLeave(): void {
            for (const m of nozzles) {
                setMeshVisible(m, false);
            }
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            for (const m of nozzles) {
                setMeshVisible(m, v);
            }
        },
        update(): void {
            /* jets refresh on param change, not per-frame */
        },
        demoParams(): DemoParam[] {
            return [
                { key: "centralSpeed", label: "Central jet speed", type: "number", min: 4, max: 18, step: 0.5, value: fountainParams.centralSpeed },
                { key: "ringSpeed", label: "Ring jet speed", type: "number", min: 4, max: 18, step: 0.5, value: fountainParams.ringSpeed },
                { key: "ringOut", label: "Ring outward angle", type: "number", min: 0, max: 2, step: 0.05, value: fountainParams.ringOut },
                { key: "centralRadius", label: "Central nozzle radius", type: "number", min: 0.05, max: 0.6, step: 0.01, value: fountainParams.centralRadius },
                { key: "ringRadius", label: "Ring nozzle radius", type: "number", min: 0.05, max: 0.6, step: 0.01, value: fountainParams.ringRadius },
                { key: "rate", label: "Emit rate", type: "number", min: 0.05, max: 1.5, step: 0.05, value: fountainParams.rate },
                { key: "spread", label: "Jet spread", type: "number", min: 0, max: 2, step: 0.05, value: fountainParams.spread },
            ];
        },
        applyParam(key: string, value: number | boolean | string): void {
            (fountainParams as Record<string, number>)[key] = value as number;
            updateFountainNozzles();
            ctx.refreshEmitters();
        },
        extraControls() {
            return [];
        },
        presets,
    };
}
