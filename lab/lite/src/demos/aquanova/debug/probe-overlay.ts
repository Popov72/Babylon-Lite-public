// Debug visualization for the point-of-interest cubemap blend.
//
// Each probe shows its full-influence inner box, zero-influence outer box, and capture point.
// The camera marker is the POI consumed by the selector; the panel reports the selected probes,
// their normalized distance fields, and final weights.

import { addToScene, createCylinder, createSphere, createStandardMaterial, setMeshVisible, type EngineContext, type Material, type Mesh, type SceneContext } from "babylon-lite";
import type { LocalEnvironmentBlendInfo, LocalEnvironmentProbeVolume } from "../local-environments.js";

export interface ProbeOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    probes: readonly LocalEnvironmentProbeVolume[];
    blendInfo: () => LocalEnvironmentBlendInfo;
    poi: () => readonly [number, number, number];
}

export interface ProbeOverlay {
    toggle(): void;
    onFrame(): void;
}

const PROBE_COLORS: readonly (readonly [number, number, number])[] = [
    [0.2, 0.85, 1],
    [1, 0.55, 0.15],
    [0.9, 0.3, 0.9],
    [0.35, 1, 0.45],
];

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
    const length = Math.hypot(direction[0], direction[1], direction[2]);
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
    edge.scaling.set(1, Math.max(Math.hypot(dx, dy, dz), 1e-4), 1);
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

function setVisible(meshes: readonly Mesh[], visible: boolean): void {
    for (const mesh of meshes) setMeshVisible(mesh, visible);
}

function vec(values: readonly number[]): string {
    return values.map((value) => value.toFixed(2)).join(", ");
}

export function createProbeOverlay({ engine, scene, canvas, probes, blendInfo, poi }: ProbeOverlayOptions): ProbeOverlay {
    const meshes: Mesh[] = [];
    const weightMarkers = new Map<string, Mesh>();

    probes.forEach((probe, index) => {
        const color = PROBE_COLORS[index % PROBE_COLORS.length]!;
        meshes.push(...addBox(engine, scene, probe.centre, probe.outerHalfSize, createUnlitMaterial(dim(color, 0.35)), 0.025));
        meshes.push(...addBox(engine, scene, probe.centre, probe.innerHalfSize, createUnlitMaterial(color), 0.045));

        const marker = createSphere(engine, { diameter: 0.3, segments: 8 });
        marker.position.set(probe.capturePosition[0], probe.capturePosition[1], probe.capturePosition[2]);
        marker.material = createUnlitMaterial(color);
        marker.pickable = false;
        marker.visible = false;
        addToScene(scene, marker);
        meshes.push(marker);
        weightMarkers.set(probe.id, marker);
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
        const point = poi();
        const info = blendInfo();
        const selected = new Map(info.probes.map((probe) => [probe.id, probe]));
        for (const probe of probes) {
            const weight = selected.get(probe.id)?.weight ?? 0;
            const marker = weightMarkers.get(probe.id)!;
            const scale = 0.65 + weight * 1.8;
            marker.scaling.set(scale, scale, scale);
        }

        const nextKey = `${point.map((value) => value.toFixed(2)).join("|")}|${info.probes.map((probe) => `${probe.id}:${probe.weight.toFixed(4)}:${probe.ndf.toFixed(4)}`).join("|")}`;
        if (nextKey === panelKey) return;
        panelKey = nextKey;
        panel.textContent =
            `CUBEMAP POI BLEND (V)\n` +
            `POI (${vec(point)})\n` +
            `Inner boxes: full influence   Outer boxes: zero-influence boundary\n` +
            `Dominant: ${info.dominantProbeId ?? "none"}\n` +
            (info.probes.length ? info.probes.map((probe) => `${probe.id}  weight=${probe.weight.toFixed(4)}  ndf=${probe.ndf.toFixed(4)}`).join("\n") : "No selected probe.") +
            "\n\n" +
            probes
                .map(
                    (probe) =>
                        `${probe.id}\n  centre(${vec(probe.centre)})  inner(${vec(probe.innerHalfSize.map((value) => value * 2))})  outer(${vec(probe.outerHalfSize.map((value) => value * 2))})`
                )
                .join("\n");
        canvas.dataset.probeOverlayBlend = info.probes.map((probe) => `${probe.id}:${probe.weight.toFixed(4)}:${probe.ndf.toFixed(4)}`).join(",");
    };

    return {
        toggle(): void {
            enabled = !enabled;
            canvas.dataset.probeOverlay = enabled ? "on" : "off";
            setVisible(meshes, enabled);
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
