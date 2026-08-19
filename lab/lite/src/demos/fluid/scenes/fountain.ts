// Fountain demo — a wide cylindrical basin confining a pool of water; a central
// jet plus a ring of angled jets recirculate it. A pump intake across the floor
// throttles how much settled water is relaunched, keeping most of it pooled.

import { addToScene, createCylinder, createStandardMaterial, setMeshVisible } from "babylon-lite";
import type { FluidEmitter, FluidFlowConfig, Mesh } from "babylon-lite";
import type { SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import type { FluidCtx, FluidDemo } from "../demo.js";
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

    // Defaults used to build the editable, solver-independent flow description.
    const fountainParams = { centralSpeed: 11, ringSpeed: 9, ringOut: 1.0, rate: 0.45, spread: 0.4, centralRadius: 0.15, ringRadius: 0.12 };

    const buildFountainEmitters = (): FluidEmitter[] => {
        const list: FluidEmitter[] = [
            {
                id: "fountain-center",
                name: "Center jet",
                enabled: true,
                behavior: "inflow",
                transform: { position: [0, FOUNTAIN_CENTRAL_Y, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: { type: "sphere", radius: fountainParams.centralRadius },
                sampling: "volume",
                velocity: [0, fountainParams.centralSpeed, 0],
                velocitySpace: "world",
                spread: fountainParams.spread,
                volumeRate: (200 * fountainParams.rate) / (FOUNTAIN_RING_N + 1),
            },
        ];
        for (let k = 0; k < FOUNTAIN_RING_N; k++) {
            const th = (k / FOUNTAIN_RING_N) * Math.PI * 2;
            const ox = Math.cos(th);
            const oz = Math.sin(th);
            const dx = ox * fountainParams.ringOut;
            const dz = oz * fountainParams.ringOut;
            const len = Math.hypot(dx, 1.1, dz);
            list.push({
                id: `fountain-ring-${k + 1}`,
                name: `Ring jet ${k + 1}`,
                enabled: true,
                behavior: "inflow",
                transform: { position: [ox * FOUNTAIN_RING_R, FOUNTAIN_RING_Y, oz * FOUNTAIN_RING_R], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: { type: "sphere", radius: fountainParams.ringRadius },
                sampling: "volume",
                velocity: [(dx / len) * fountainParams.ringSpeed, (1.1 / len) * fountainParams.ringSpeed, (dz / len) * fountainParams.ringSpeed],
                velocitySpace: "world",
                spread: fountainParams.spread,
                volumeRate: (200 * fountainParams.rate) / (FOUNTAIN_RING_N + 1),
            });
        }
        return list;
    };

    const fountainConfig = (): FluidFlowConfig => {
        const jets = buildFountainEmitters();
        return {
            emitters: [
                {
                    id: "fountain-fill",
                    name: "Initial basin fill",
                    enabled: true,
                    behavior: "initial",
                    transform: {
                        position: [
                            (FOUNTAIN_SPAWN_MIN[0] + FOUNTAIN_SPAWN_MAX[0]) / 2,
                            (FOUNTAIN_SPAWN_MIN[1] + FOUNTAIN_SPAWN_MAX[1]) / 2,
                            (FOUNTAIN_SPAWN_MIN[2] + FOUNTAIN_SPAWN_MAX[2]) / 2,
                        ],
                        rotation: [0, 0, 0, 1],
                        scale: [1, 1, 1],
                    },
                    shape: {
                        type: "box",
                        size: [FOUNTAIN_SPAWN_MAX[0] - FOUNTAIN_SPAWN_MIN[0], FOUNTAIN_SPAWN_MAX[1] - FOUNTAIN_SPAWN_MIN[1], FOUNTAIN_SPAWN_MAX[2] - FOUNTAIN_SPAWN_MIN[2]],
                    },
                    sampling: "volume",
                    velocity: [0, 0, 0],
                    velocitySpace: "world",
                    spread: 0,
                },
                ...jets,
            ],
            sinks: [
                {
                    id: "fountain-basin-recycle",
                    name: "Basin recycle",
                    enabled: true,
                    mode: "recycle",
                    transform: { position: [0, FOUNTAIN_FLOOR + 0.4, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                    shape: { type: "box", size: [FOUNTAIN_R * 2, 0.8, FOUNTAIN_R * 2] },
                    targets: jets.map((emitter) => emitter.id),
                    volumeRate: 200 * fountainParams.rate,
                },
            ],
        };
    };

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
    const updateFountainNozzles = (flow: FluidFlowConfig = fountainConfig()): void => {
        const es = flow.emitters.filter((emitter) => emitter.behavior === "inflow");
        for (let k = 0; k < nozzles.length; k++) {
            const e = es[k]!;
            const m = nozzles[k]!;
            if (!e) {
                setMeshVisible(m, false);
                continue;
            }
            const radius = e.shape.type === "sphere" ? e.shape.radius : 0.1;
            const speed = Math.hypot(e.velocity[0], e.velocity[1], e.velocity[2]);
            const dir: [number, number, number] = speed > 1e-6 ? [e.velocity[0] / speed, e.velocity[1] / speed, e.velocity[2] / speed] : [0, 1, 0];
            // Scale the unit-diameter bore to the emitter radius (X/Z only; height fixed).
            m.scaling.set(2 * radius, 1, 2 * radius);
            // Sink the spout so its tip sits at the emit point.
            m.position.set(e.transform.position[0] - dir[0] * (NOZZLE_H / 2), e.transform.position[1] - dir[1] * (NOZZLE_H / 2), e.transform.position[2] - dir[2] * (NOZZLE_H / 2));
            const q = quatFromY(dir);
            m.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
        }
    };
    updateFountainNozzles();
    let containerVisible = true; // toggled by the "Show container / nozzle meshes" UI checkbox

    return {
        key: "fountain",
        label: "Fountain (jets)",
        envUrl: ENV_STUDIO_URL,
        sdf,
        writeSdfParams(): void {
            engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, new Float32Array([FOUNTAIN_R, FOUNTAIN_FLOOR, 0, 0]));
        },
        flow() {
            return fountainConfig();
        },
        onFlowChanged(flow): void {
            updateFountainNozzles(flow);
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
        demoParams() {
            return [];
        },
        applyParam(key: string, value: number | boolean | string): void {
            if (key in fountainParams && typeof value === "number") {
                (fountainParams as unknown as Record<string, number>)[key] = value;
            }
        },
        extraControls() {
            return [];
        },
    };
}
