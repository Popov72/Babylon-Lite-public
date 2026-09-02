import { addToScene, createLineSystem, setMeshVisible, updateLineSystem, type EngineContext, type FreeCamera, type Mesh, type SceneContext, type Vec3 } from "babylon-lite";
import type { PortalTraversal } from "../portal-visibility.js";

export interface PortalOverlay {
    toggle(): void;
    onFrame(): void;
}

interface PortalOverlayOptions {
    engine: EngineContext;
    scene: SceneContext;
    canvas: HTMLCanvasElement;
    camera: FreeCamera;
    maxFrusta: number;
    traversals: () => readonly PortalTraversal[];
}

const COLORS = [
    { r: 0.1, g: 0.95, b: 1, a: 1 },
    { r: 1, g: 0.55, b: 0.1, a: 1 },
    { r: 0.55, g: 1, b: 0.2, a: 1 },
    { r: 1, g: 0.2, b: 0.75, a: 1 },
    { r: 0.65, g: 0.45, b: 1, a: 1 },
] as const;

function point([x, y, z]: readonly [number, number, number]): Vec3 {
    return { x, y, z };
}

function emptyLines(): Vec3[][] {
    return Array.from({ length: 12 }, () => [
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
    ]);
}

function frustumLines(camera: FreeCamera, traversal: PortalTraversal): Vec3[][] {
    const cameraPoint: readonly [number, number, number] = [camera.position.x, camera.position.y, camera.position.z];
    const farDistance = Math.min(camera.farPlane, 30);
    const corners = traversal.corners.slice(0, 4);
    if (corners.length < 4) return emptyLines();
    const farCorners = corners.map((corner): readonly [number, number, number] => {
        const dx = corner[0] - cameraPoint[0];
        const dy = corner[1] - cameraPoint[1];
        const dz = corner[2] - cameraPoint[2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
        const length = Math.max(distance + 0.1, farDistance);
        return [cameraPoint[0] + (dx / distance) * length, cameraPoint[1] + (dy / distance) * length, cameraPoint[2] + (dz / distance) * length];
    });
    const lines: Vec3[][] = [];
    for (let i = 0; i < 4; i++) {
        const next = (i + 1) % 4;
        lines.push([point(corners[i]!), point(corners[next]!)]);
        lines.push([point(cameraPoint), point(farCorners[i]!)]);
        lines.push([point(farCorners[i]!), point(farCorners[next]!)]);
    }
    return lines;
}

export function createPortalOverlay(options: PortalOverlayOptions): PortalOverlay {
    const visuals: Mesh[] = [];
    for (let i = 0; i < Math.max(1, options.maxFrusta); i++) {
        const mesh = createLineSystem(options.engine, {
            name: `portalFrustum${i}`,
            lines: emptyLines(),
            color: COLORS[i % COLORS.length],
        });
        mesh.pickable = false;
        mesh.visible = false;
        addToScene(options.scene, mesh);
        visuals.push(mesh);
    }

    let enabled = false;
    const refresh = (): void => {
        const traversals = options.traversals();
        for (let i = 0; i < visuals.length; i++) {
            const mesh = visuals[i]!;
            const traversal = traversals[i];
            if (!enabled || !traversal) {
                setMeshVisible(mesh, false);
                continue;
            }
            updateLineSystem(options.engine, mesh, { lines: frustumLines(options.camera, traversal) });
            setMeshVisible(mesh, true);
        }
        options.canvas.dataset.portalFrusta = enabled ? String(Math.min(traversals.length, visuals.length)) : "off";
    };

    return {
        toggle(): void {
            enabled = !enabled;
            refresh();
        },
        onFrame(): void {
            if (enabled) refresh();
        },
    };
}
