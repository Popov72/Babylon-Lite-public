import {
    addToScene,
    createLineMaterial,
    createLineSystem,
    createSphere,
    createStandardMaterial,
    setMeshVisible,
    type EngineContext,
    type FluidEmitter,
    type FluidShape,
    type FluidSink,
    type Mesh,
    type SceneContext,
    type Vec3,
} from "babylon-lite";
import type { FluidSimulationState } from "../behaviors/fluid-simulation-runtime.js";

export interface FluidSimulationDebugSnapshot {
    entityName: string;
    state: FluidSimulationState;
    center: readonly [number, number, number];
    size: readonly [number, number, number];
    emitters: readonly FluidEmitter[];
    sinks: readonly FluidSink[];
}

export interface FluidSimulationOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    simulations: () => readonly FluidSimulationDebugSnapshot[];
}

export interface FluidSimulationOverlay {
    toggle(): void;
    isOn(): boolean;
    onFrame(): void;
}

type Segment = readonly [Vec3, Vec3];

interface FlowVisual {
    mesh: Mesh;
    kind: "emitter" | "sink";
    index: number;
}

interface SimulationVisual {
    center: Mesh;
    grid: Mesh;
    flows: FlowVisual[];
}

function point(x: number, y: number, z: number): Vec3 {
    return { x, y, z };
}

function boxSegments(size: readonly [number, number, number]): Segment[] {
    const [hx, hy, hz] = size.map((value) => value * 0.5);
    const corners = [
        point(-hx!, -hy!, -hz!),
        point(hx!, -hy!, -hz!),
        point(hx!, -hy!, hz!),
        point(-hx!, -hy!, hz!),
        point(-hx!, hy!, -hz!),
        point(hx!, hy!, -hz!),
        point(hx!, hy!, hz!),
        point(-hx!, hy!, hz!),
    ];
    return [
        [corners[0]!, corners[1]!],
        [corners[1]!, corners[2]!],
        [corners[2]!, corners[3]!],
        [corners[3]!, corners[0]!],
        [corners[4]!, corners[5]!],
        [corners[5]!, corners[6]!],
        [corners[6]!, corners[7]!],
        [corners[7]!, corners[4]!],
        [corners[0]!, corners[4]!],
        [corners[1]!, corners[5]!],
        [corners[2]!, corners[6]!],
        [corners[3]!, corners[7]!],
    ];
}

function appendPolyline(segments: Segment[], points: readonly Vec3[], closed = true): void {
    for (let index = 1; index < points.length; index++) segments.push([points[index - 1]!, points[index]!]);
    if (closed && points.length > 2) segments.push([points[points.length - 1]!, points[0]!]);
}

function appendRing(segments: Segment[], plane: "xy" | "xz" | "yz", radius: number, center: Vec3 = point(0, 0, 0), steps = 24): void {
    const points: Vec3[] = [];
    for (let index = 0; index < steps; index++) {
        const angle = (index / steps) * Math.PI * 2;
        const a = Math.cos(angle) * radius;
        const b = Math.sin(angle) * radius;
        points.push(
            plane === "xy" ? point(center.x + a, center.y + b, center.z) : plane === "xz" ? point(center.x + a, center.y, center.z + b) : point(center.x, center.y + a, center.z + b)
        );
    }
    appendPolyline(segments, points);
}

export function fluidShapeWireframeLines(shape: FluidShape): Vec3[][] {
    const segments: Segment[] = [];
    if (shape.type === "box") {
        segments.push(...boxSegments(shape.size));
    } else if (shape.type === "sphere") {
        appendRing(segments, "xy", shape.radius);
        appendRing(segments, "xz", shape.radius);
        appendRing(segments, "yz", shape.radius);
    } else if (shape.type === "cylinder") {
        const halfHeight = shape.height * 0.5;
        const radii = shape.innerRadius && shape.innerRadius > 0 ? [shape.radius, shape.innerRadius] : [shape.radius];
        for (const radius of radii) {
            appendRing(segments, "xz", radius, point(0, -halfHeight, 0));
            appendRing(segments, "xz", radius, point(0, halfHeight, 0));
            for (let index = 0; index < 8; index++) {
                const angle = (index / 8) * Math.PI * 2;
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius;
                segments.push([point(x, -halfHeight, z), point(x, halfHeight, z)]);
            }
        }
    } else if (shape.type === "cone") {
        const halfHeight = shape.height * 0.5;
        if (shape.bottomRadius > 0) appendRing(segments, "xz", shape.bottomRadius, point(0, -halfHeight, 0));
        if (shape.topRadius > 0) appendRing(segments, "xz", shape.topRadius, point(0, halfHeight, 0));
        for (let index = 0; index < 8; index++) {
            const angle = (index / 8) * Math.PI * 2;
            const x = Math.cos(angle);
            const z = Math.sin(angle);
            segments.push([point(x * shape.bottomRadius, -halfHeight, z * shape.bottomRadius), point(x * shape.topRadius, halfHeight, z * shape.topRadius)]);
        }
    } else if (shape.type === "capsule") {
        const bodyHalfHeight = Math.max(0, shape.height * 0.5 - shape.radius);
        appendRing(segments, "xz", shape.radius, point(0, -bodyHalfHeight, 0));
        appendRing(segments, "xz", shape.radius, point(0, bodyHalfHeight, 0));
        for (let plane = 0; plane < 2; plane++) {
            const meridian: Vec3[] = [];
            for (let index = 0; index <= 32; index++) {
                const angle = Math.PI - (index / 32) * Math.PI * 2;
                const y = Math.sin(angle) * shape.radius + (angle >= 0 ? bodyHalfHeight : -bodyHalfHeight);
                const radial = Math.cos(angle) * shape.radius;
                meridian.push(plane === 0 ? point(radial, y, 0) : point(0, y, radial));
            }
            appendPolyline(segments, meridian);
        }
    } else {
        const halfThickness = shape.thickness * 0.5;
        for (let index = 0; index < shape.points.length; index++) {
            const current = shape.points[index]!;
            const next = shape.points[(index + 1) % shape.points.length]!;
            const bottom = point(current[0], -halfThickness, current[1]);
            const top = point(current[0], halfThickness, current[1]);
            segments.push([bottom, point(next[0], -halfThickness, next[1])], [top, point(next[0], halfThickness, next[1])], [bottom, top]);
        }
    }
    return segments.map(([a, b]) => [a, b]);
}

export function fluidGridWireframeLines(size: readonly [number, number, number]): Vec3[][] {
    return boxSegments(size).map(([a, b]) => [a, b]);
}

function vec(value: readonly number[]): string {
    return value.map((component) => component.toFixed(3)).join(", ");
}

export function createFluidSimulationOverlay({ engine, scene, canvas, simulations }: FluidSimulationOverlayOptions): FluidSimulationOverlay {
    const initial = simulations();
    const gridMaterial = createLineMaterial({
        name: "behavior-fluid-grid-material",
        color: { r: 0.35, g: 1, b: 0.25, a: 1 },
        depthWrite: false,
        depthCompare: "always",
    });
    const emitterMaterial = createLineMaterial({
        name: "behavior-fluid-emitter-material",
        color: { r: 0.1, g: 0.9, b: 1, a: 1 },
        depthWrite: false,
        depthCompare: "always",
    });
    const sinkMaterial = createLineMaterial({
        name: "behavior-fluid-sink-material",
        color: { r: 1, g: 0.5, b: 0.08, a: 1 },
        depthWrite: false,
        depthCompare: "always",
    });
    const centerMaterial = createStandardMaterial();
    centerMaterial.disableLighting = true;
    centerMaterial.diffuseColor = [1, 1, 1];
    centerMaterial.emissiveColor = [1, 1, 1];

    const addLines = (name: string, lines: Vec3[][], material: ReturnType<typeof createLineMaterial>): Mesh => {
        const mesh = createLineSystem(engine, { name, lines, material });
        mesh.pickable = false;
        mesh.renderOrder = 10_000;
        mesh.visible = false;
        addToScene(scene, mesh);
        return mesh;
    };
    const visuals: SimulationVisual[] = initial.map((simulation) => {
        const center = createSphere(engine, { diameter: 0.25, segments: 8 });
        center.name = `behavior-fluid-center-${simulation.entityName}`;
        center.material = centerMaterial;
        center.pickable = false;
        center.visible = false;
        addToScene(scene, center);
        const grid = addLines(`behavior-fluid-grid-${simulation.entityName}`, fluidGridWireframeLines(simulation.size), gridMaterial);
        const flows: FlowVisual[] = [
            ...simulation.emitters.map((emitter, index) => ({
                mesh: addLines(`behavior-fluid-emitter-${simulation.entityName}-${emitter.id}`, fluidShapeWireframeLines(emitter.shape), emitterMaterial),
                kind: "emitter" as const,
                index,
            })),
            ...simulation.sinks.map((sink, index) => ({
                mesh: addLines(`behavior-fluid-sink-${simulation.entityName}-${sink.id}`, fluidShapeWireframeLines(sink.shape), sinkMaterial),
                kind: "sink" as const,
                index,
            })),
        ];
        return { center, grid, flows };
    });

    const panel = document.createElement("div");
    panel.id = "aq-fluid-sim-debug";
    panel.style.cssText =
        "position:fixed;right:12px;bottom:54px;z-index:20;display:none;max-width:58vw;max-height:52vh;overflow:hidden;" +
        "font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;color:#e8f7ff;background:rgba(0,0,0,.76);" +
        "padding:8px 11px;border-radius:6px;pointer-events:none;white-space:pre;";
    document.body.appendChild(panel);

    let enabled = false;
    const refresh = (): void => {
        if (!enabled) return;
        const snapshots = simulations();
        const lines = ["FLUID SIMULATIONS (G)", "grid: green   center: white   emitters: cyan   sinks: orange"];
        for (let index = 0; index < visuals.length; index++) {
            const visual = visuals[index]!;
            const snapshot = snapshots[index];
            if (!snapshot) {
                setMeshVisible(visual.center, false);
                setMeshVisible(visual.grid, false);
                for (const flow of visual.flows) setMeshVisible(flow.mesh, false);
                continue;
            }
            visual.center.position.set(snapshot.center[0], snapshot.center[1], snapshot.center[2]);
            visual.grid.position.set(snapshot.center[0], snapshot.center[1], snapshot.center[2]);
            setMeshVisible(visual.center, true);
            setMeshVisible(visual.grid, true);
            for (const flow of visual.flows) {
                const object = flow.kind === "emitter" ? snapshot.emitters[flow.index] : snapshot.sinks[flow.index];
                if (!object) {
                    setMeshVisible(flow.mesh, false);
                    continue;
                }
                const { position, rotation, scale } = object.transform;
                flow.mesh.position.set(position[0], position[1], position[2]);
                flow.mesh.rotationQuaternion.set(rotation[0], rotation[1], rotation[2], rotation[3]);
                flow.mesh.scaling.set(scale[0], scale[1], scale[2]);
                setMeshVisible(flow.mesh, true);
            }
            lines.push("", `${snapshot.entityName} [${snapshot.state}]`, `  grid centre (${vec(snapshot.center)})`, `  grid size   (${vec(snapshot.size)})`);
            for (const emitter of snapshot.emitters) lines.push(`  emitter ${emitter.name} [${emitter.enabled ? "on" : "off"}] (${vec(emitter.transform.position)})`);
            for (const sink of snapshot.sinks) lines.push(`  sink    ${sink.name} [${sink.enabled ? "on" : "off"}] (${vec(sink.transform.position)})`);
        }
        panel.textContent = lines.join("\n");
    };

    return {
        isOn: () => enabled,
        toggle(): void {
            enabled = !enabled;
            canvas.dataset.fluidSimulationOverlay = enabled ? "on" : "off";
            panel.style.display = enabled ? "block" : "none";
            if (enabled) {
                refresh();
            } else {
                for (const visual of visuals) {
                    setMeshVisible(visual.center, false);
                    setMeshVisible(visual.grid, false);
                    for (const flow of visual.flows) setMeshVisible(flow.mesh, false);
                }
            }
        },
        onFrame: refresh,
    };
}
