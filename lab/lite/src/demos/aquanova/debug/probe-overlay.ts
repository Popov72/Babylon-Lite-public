// Debug visualization for the fragment-weighted cubemap blend.
//
// Each probe shows its full-influence inner volume, zero-influence outer volume, and capture point.
// Only probes in the camera's current voxel show capture spheres or appear in the panel.

import { addToScene, createCylinder, createSphere, createStandardMaterial, setMeshVisible, type EngineContext, type Material, type Mesh, type SceneContext } from "babylon-lite";
import type { LocalEnvironmentBlendInfo, LocalEnvironmentProbeVolume } from "../local-environments.js";

export interface ProbeOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    probes: readonly LocalEnvironmentProbeVolume[];
    blendInfo: () => LocalEnvironmentBlendInfo;
    blendingEnabled: () => boolean;
    setDebugEnabled: (enabled: boolean) => void;
}

export interface ProbeOverlay {
    toggle(): void;
    onFrame(): void;
}

const EDGE_PAIRS = [
    [0, 1],
    [0, 2],
    [0, 4],
    [1, 3],
    [1, 5],
    [2, 3],
    [2, 6],
    [3, 7],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
] as const;

function quatFromYTo(direction: readonly [number, number, number]): [number, number, number, number] {
    const length = Math.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    if (length < 1e-9) return [0, 0, 0, 1];
    const x = direction[0] / length;
    const y = direction[1] / length;
    const z = direction[2] / length;
    if (y > 0.999999) return [0, 0, 0, 1];
    if (y < -0.999999) return [1, 0, 0, 0];
    const s = Math.sqrt((1 + y) * 2);
    return [z / s, 0, -x / s, s * 0.5];
}

function createUnlitMaterial(color: readonly [number, number, number]): Material {
    const material = createStandardMaterial();
    material.disableLighting = true;
    material.diffuseColor = [...color];
    material.emissiveColor = [...color];
    material.backFaceCulling = false;
    return material;
}

function dim(color: readonly [number, number, number], factor: number): [number, number, number] {
    return [color[0] * factor, color[1] * factor, color[2] * factor];
}

function boxCorners(centre: readonly [number, number, number], halfSize: readonly [number, number, number]): [number, number, number][] {
    const corners: [number, number, number][] = [];
    for (let x = -1; x <= 1; x += 2) {
        for (let y = -1; y <= 1; y += 2) {
            for (let z = -1; z <= 1; z += 2) {
                corners.push([centre[0] + halfSize[0] * x, centre[1] + halfSize[1] * y, centre[2] + halfSize[2] * z]);
            }
        }
    }
    return corners;
}

function addEdge(
    engine: EngineContext,
    scene: SceneContext,
    a: readonly [number, number, number],
    b: readonly [number, number, number],
    material: Material,
    thickness: number
): Mesh {
    const edge = createCylinder(engine, { height: 1, diameter: thickness, tessellation: 6 });
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const dz = b[2] - a[2];
    edge.position.set((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5);
    edge.scaling.set(1, Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), 1e-4), 1);
    const q = quatFromYTo([dx, dy, dz]);
    edge.rotationQuaternion.set(q[0], q[1], q[2], q[3]);
    edge.material = material;
    edge.pickable = false;
    edge.visible = false;
    addToScene(scene, edge);
    return edge;
}

function addBox(
    engine: EngineContext,
    scene: SceneContext,
    centre: readonly [number, number, number],
    halfSize: readonly [number, number, number],
    material: Material,
    thickness: number
): Mesh[] {
    const corners = boxCorners(centre, halfSize);
    return EDGE_PAIRS.map(([a, b]) => addEdge(engine, scene, corners[a]!, corners[b]!, material, thickness));
}

function addSphereWire(engine: EngineContext, scene: SceneContext, centre: readonly [number, number, number], radius: number, material: Material, thickness: number): Mesh[] {
    const meshes: Mesh[] = [];
    const segments = 24;
    for (let plane = 0; plane < 3; plane++) {
        for (let segment = 0; segment < segments; segment++) {
            const point = (angle: number): [number, number, number] => {
                const a = Math.cos(angle) * radius;
                const b = Math.sin(angle) * radius;
                return plane === 0
                    ? [centre[0], centre[1] + a, centre[2] + b]
                    : plane === 1
                      ? [centre[0] + a, centre[1], centre[2] + b]
                      : [centre[0] + a, centre[1] + b, centre[2]];
            };
            meshes.push(addEdge(engine, scene, point((segment / segments) * Math.PI * 2), point(((segment + 1) / segments) * Math.PI * 2), material, thickness));
        }
    }
    return meshes;
}

function setVisible(meshes: readonly Mesh[], visible: boolean): void {
    for (const mesh of meshes) setMeshVisible(mesh, visible);
}

function vec(values: readonly number[]): string {
    return values.map((value) => value.toFixed(2)).join(", ");
}

export function createProbeOverlay({ engine, scene, canvas, probes, blendInfo, blendingEnabled, setDebugEnabled }: ProbeOverlayOptions): ProbeOverlay {
    const boxMeshes: Mesh[] = [];
    const probeMarkers = new Map<string, Mesh>();
    const probesById = new Map(probes.map((probe) => [probe.id, probe]));

    probes.forEach((probe) => {
        const color = probe.debugColor;
        if (probe.shape === "sphere") {
            boxMeshes.push(...addSphereWire(engine, scene, probe.centre, probe.outerRadius, createUnlitMaterial(dim(color, 0.35)), 0.025));
            boxMeshes.push(...addSphereWire(engine, scene, probe.centre, probe.innerRadius, createUnlitMaterial(color), 0.045));
        } else {
            boxMeshes.push(...addBox(engine, scene, probe.centre, probe.outerHalfSize, createUnlitMaterial(dim(color, 0.35)), 0.025));
            boxMeshes.push(...addBox(engine, scene, probe.centre, probe.innerHalfSize, createUnlitMaterial(color), 0.045));
        }

        const marker = createSphere(engine, { diameter: 0.3, segments: 8 });
        marker.position.set(probe.capturePosition[0], probe.capturePosition[1], probe.capturePosition[2]);
        marker.material = createUnlitMaterial(color);
        marker.pickable = false;
        marker.visible = false;
        addToScene(scene, marker);
        probeMarkers.set(probe.id, marker);
    });

    const panel = document.createElement("div");
    panel.id = "aq-probes";
    panel.style.cssText =
        "position:fixed;right:12px;top:12px;z-index:20;display:none;max-width:46vw;max-height:70vh;overflow:hidden;" +
        "font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;color:#e8f7ff;background:rgba(0,0,0,.76);" +
        "padding:8px 11px;border-radius:6px;pointer-events:none;white-space:pre;";
    document.body.appendChild(panel);

    let enabled = false;
    let panelKey = "";

    const refresh = (): void => {
        if (!enabled) return;
        const info = blendInfo();
        const selectedIds = new Set(info.cameraVoxelProbeIds);
        const blending = blendingEnabled();
        for (const probe of probes) {
            const marker = probeMarkers.get(probe.id)!;
            setMeshVisible(marker, blending && selectedIds.has(probe.id));
        }

        const nextKey = `${blending}|${info.cameraVoxelProbeIds.join(",")}`;
        if (nextKey === panelKey) return;
        panelKey = nextKey;
        const selectedProbes = info.cameraVoxelProbeIds.map((id) => probesById.get(id)).filter((probe): probe is LocalEnvironmentProbeVolume => probe !== undefined);
        panel.textContent =
            `CUBEMAP BLEND COLORS (V)\n` +
            `Mode: ${blending ? "per-fragment debug colors" : "disabled — static per-mesh cubemaps"}\n` +
            `Camera voxel probes: ${selectedProbes.length}\n\n` +
            (selectedProbes.length
                ? selectedProbes
                      .map((probe) =>
                          probe.shape === "sphere"
                              ? `${probe.id} [sphere]\n  centre(${vec(probe.centre)})\n  inner radius(${probe.innerRadius.toFixed(2)})\n  outer radius(${probe.outerRadius.toFixed(2)})`
                              : `${probe.id} [box]\n  centre(${vec(probe.centre)})\n  inner(${vec(probe.innerHalfSize.map((value) => value * 2))})\n  outer(${vec(probe.outerHalfSize.map((value) => value * 2))})`
                      )
                      .join("\n\n")
                : "No camera voxel probes while blending is disabled.");
        canvas.dataset.probeOverlaySelected = info.cameraVoxelProbeIds.join(",");
    };

    return {
        toggle(): void {
            enabled = !enabled;
            canvas.dataset.probeOverlay = enabled ? "on" : "off";
            setDebugEnabled(enabled);
            setVisible(boxMeshes, enabled);
            if (!enabled) {
                for (const marker of probeMarkers.values()) {
                    setMeshVisible(marker, false);
                }
            }
            panel.style.display = enabled ? "block" : "none";
            if (enabled) {
                panelKey = "";
                refresh();
            }
        },
        onFrame(): void {
            refresh();
        },
    };
}
