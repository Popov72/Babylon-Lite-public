// The `L` debug overlay: runtime lights authored for the player's current chunk.
//
// It visualizes the effective records consumed by `lights.ts`, not merely the JSON source:
// transforms come from the exported glTF's LIGHT_* nodes and runtime parameters can be overridden by
// ship_manifest.json. Coverage is deliberately wireframe so it does not obscure the scene:
// point-light ranges are three great-circle rings, and spot lights are a base ring plus radial ribs.

import {
    addToScene,
    createCylinder,
    createSphere,
    createStandardMaterial,
    createTorus,
    setMeshVisible,
    type EngineContext,
    type Material,
    type Mesh,
    type SceneContext,
} from "babylon-lite";
import type { RuntimeLightDebug } from "../lights.js";

export interface LightOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    lights: readonly RuntimeLightDebug[];
    roomAt: () => string;
}

export interface LightOverlay {
    toggle(): void;
    onFrame(): void;
}

interface LightVisual {
    light: RuntimeLightDebug;
    meshes: Mesh[];
}

/** Quaternion rotating +Y onto `d`. */
function quatFromYTo(d: readonly [number, number, number]): [number, number, number, number] {
    const len = Math.hypot(d[0], d[1], d[2]);
    if (len < 1e-9) return [0, 0, 0, 1];
    const x = d[0] / len;
    const y = d[1] / len;
    const z = d[2] / len;
    if (y > 0.999999) return [0, 0, 0, 1];
    if (y < -0.999999) return [1, 0, 0, 0];
    const s = Math.sqrt((1 + y) * 2);
    return [z / s, 0, -x / s, s * 0.5];
}

function setVisible(meshes: readonly Mesh[], visible: boolean): void {
    for (const mesh of meshes) setMeshVisible(mesh, visible);
}

function perpendicularBasis(direction: readonly [number, number, number]): [[number, number, number], [number, number, number]] {
    const helper: [number, number, number] = Math.abs(direction[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const ux = direction[1] * helper[2] - direction[2] * helper[1];
    const uy = direction[2] * helper[0] - direction[0] * helper[2];
    const uz = direction[0] * helper[1] - direction[1] * helper[0];
    const uLen = Math.hypot(ux, uy, uz) || 1;
    const u: [number, number, number] = [ux / uLen, uy / uLen, uz / uLen];
    return [u, [direction[1] * u[2] - direction[2] * u[1], direction[2] * u[0] - direction[0] * u[2], direction[0] * u[1] - direction[1] * u[0]]];
}

export function createLightOverlay({ engine, scene, canvas, lights, roomAt }: LightOverlayOptions): LightOverlay {
    const visuals: LightVisual[] = lights.map((light) => {
        const markerMaterial = createStandardMaterial();
        markerMaterial.disableLighting = true;
        markerMaterial.diffuseColor = light.color;
        markerMaterial.emissiveColor = light.color;
        markerMaterial.backFaceCulling = false;

        const coverageMaterial = createStandardMaterial();
        coverageMaterial.disableLighting = true;
        coverageMaterial.diffuseColor = light.color;
        coverageMaterial.emissiveColor = light.color;
        coverageMaterial.backFaceCulling = false;

        const meshes: Mesh[] = [];
        const add = (mesh: Mesh, material: Material): Mesh => {
            mesh.material = material;
            mesh.pickable = false;
            mesh.visible = false;
            addToScene(scene, mesh);
            meshes.push(mesh);
            return mesh;
        };
        const addEdge = (a: readonly [number, number, number], b: readonly [number, number, number], material: Material, thickness: number): Mesh => {
            const edge = add(createCylinder(engine, { height: 1, diameter: thickness, tessellation: 6 }), material);
            const dx = b[0] - a[0];
            const dy = b[1] - a[1];
            const dz = b[2] - a[2];
            const length = Math.max(Math.hypot(dx, dy, dz), 1e-4);
            edge.position.set((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5);
            edge.scaling.set(1, length, 1);
            const q = quatFromYTo([dx, dy, dz]);
            edge.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
            return edge;
        };

        const marker = add(createSphere(engine, { diameter: 0.18, segments: 8 }), markerMaterial);
        marker.position.set(light.position[0], light.position[1], light.position[2]);

        const directionLength = light.type === "directional" ? 5 : Math.max(light.range, 0.05);
        const lineThickness = Math.min(0.06, Math.max(0.015, directionLength * 0.0015));
        if (light.type === "spot" || light.type === "directional") {
            const end: [number, number, number] = [
                light.position[0] + light.direction[0] * directionLength,
                light.position[1] + light.direction[1] * directionLength,
                light.position[2] + light.direction[2] * directionLength,
            ];
            addEdge(light.position, end, markerMaterial, lineThickness);
            if (light.type === "directional") {
                const headLength = 0.35;
                const head = add(createCylinder(engine, { height: headLength, diameterTop: 0, diameterBottom: 0.22, tessellation: 10 }), markerMaterial);
                head.position.set(end[0] - light.direction[0] * headLength * 0.5, end[1] - light.direction[1] * headLength * 0.5, end[2] - light.direction[2] * headLength * 0.5);
                const q = quatFromYTo(light.direction);
                head.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
            }
        }

        if (light.type === "point") {
            const range = Math.max(light.range, 0.05);
            for (let axis = 0; axis < 3; axis++) {
                const ring = add(createTorus(engine, { diameter: range * 2, thickness: lineThickness, tessellation: 48 }), coverageMaterial);
                ring.position.set(light.position[0], light.position[1], light.position[2]);
                if (axis === 1) ring.rotation.set(Math.PI / 2, 0, 0);
                if (axis === 2) ring.rotation.set(0, 0, Math.PI / 2);
            }
        } else if (light.type === "spot") {
            const range = Math.max(light.range, 0.05);
            const halfAngle = (Math.max(0.1, Math.min(light.angle, 179)) * Math.PI) / 360;
            const baseRadius = Math.max(Math.tan(halfAngle) * range, 0.005);
            const base: [number, number, number] = [
                light.position[0] + light.direction[0] * range,
                light.position[1] + light.direction[1] * range,
                light.position[2] + light.direction[2] * range,
            ];
            const ring = add(createTorus(engine, { diameter: baseRadius * 2, thickness: lineThickness, tessellation: 48 }), coverageMaterial);
            ring.position.set(base[0], base[1], base[2]);
            const q = quatFromYTo(light.direction);
            ring.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
            const [u, v] = perpendicularBasis(light.direction);
            for (let i = 0; i < 12; i++) {
                const a = (i * Math.PI * 2) / 12;
                const ca = Math.cos(a) * baseRadius;
                const sa = Math.sin(a) * baseRadius;
                addEdge(light.position, [base[0] + u[0] * ca + v[0] * sa, base[1] + u[1] * ca + v[1] * sa, base[2] + u[2] * ca + v[2] * sa], coverageMaterial, lineThickness);
            }
        }

        return { light, meshes };
    });

    const panel = document.createElement("div");
    panel.id = "aq-lights";
    panel.style.cssText =
        "position:fixed;left:12px;top:12px;z-index:20;display:none;max-width:72vw;max-height:70vh;overflow:hidden;" +
        "font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;color:#d6eefc;background:rgba(0,0,0,.72);" +
        "padding:8px 11px;border-radius:6px;pointer-events:none;white-space:pre;";
    document.body.appendChild(panel);

    let enabled = false;
    let shownRoom = "";

    const refresh = (): void => {
        const room = roomAt();
        if (!enabled) {
            for (const visual of visuals) setVisible(visual.meshes, false);
            panel.style.display = "none";
            shownRoom = room;
            return;
        }
        const roomLights = visuals.filter((visual) => visual.light.chunk === room);
        for (const visual of visuals) setVisible(visual.meshes, visual.light.chunk === room);
        panel.style.display = "block";
        panel.textContent =
            `RUNTIME LIGHTS (L)   Chunk: ${room}   ${roomLights.length} light(s)\n` +
            "Transforms: ship.glb   Parameters: ship_manifest.json override\n" +
            (roomLights.length
                ? roomLights
                      .map(({ light }) => {
                          const color = light.color.map((v) => v.toFixed(2)).join(",");
                          const pos = light.position.map((v) => v.toFixed(2)).join(",");
                          const dir = light.direction.map((v) => v.toFixed(2)).join(",");
                          const shadow = light.castsShadows ? "yes" : light.shadowRequested ? "NO (requested, unsupported)" : "no";
                          return `${light.id}  ${light.type}${light.clustered ? " clustered" : " scoped"}  rgb(${color})  I=${light.intensity.toFixed(2)}  range=${light.range.toFixed(2)}m  angle=${light.angle.toFixed(1)}°  shadows=${shadow}\n  pos(${pos})  dir(${dir})`;
                      })
                      .join("\n")
                : "No runtime lights authored for this chunk.");
        shownRoom = room;
        canvas.dataset.lightOverlayChunk = room;
    };

    return {
        toggle(): void {
            enabled = !enabled;
            canvas.dataset.lightOverlay = enabled ? "on" : "off";
            refresh();
            if (enabled) {
                // eslint-disable-next-line no-console
                console.log(`[aquanova] runtime-light overlay: ${shownRoom}`);
            }
        },
        onFrame(): void {
            if (enabled && roomAt() !== shownRoom) refresh();
        },
    };
}
