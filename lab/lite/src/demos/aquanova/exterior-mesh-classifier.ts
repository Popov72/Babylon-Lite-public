import {
    addTaskAfter,
    addTaskAtStart,
    addToScene,
    createBox,
    createFreeCamera,
    createRenderTarget,
    createRenderTask,
    createStandardMaterial,
    enableOrthographicCamera,
    removeFromScene,
    setMeshVisible,
    type EngineContext,
    type Mesh,
    type SceneContext,
    type Task,
} from "babylon-lite";
import type { MeshGroupBounds } from "./mesh-bounds.js";
import type { ShipChunk, ShipPortal } from "./manifest.js";
import { buildRuntimePortals, type RuntimePortal } from "./portal-visibility.js";

type Vec3 = readonly [number, number, number];

export interface ExteriorMeshClassifier {
    result: Promise<ReadonlySet<Mesh>>;
}

interface ExteriorMeshClassifierOptions {
    engine: EngineContext;
    scene: SceneContext;
    meshes: readonly Mesh[];
    bounds: MeshGroupBounds;
    chunks: readonly ShipChunk[];
    portals: readonly ShipPortal[];
}

interface ExteriorView {
    direction: Vec3;
    right: Vec3;
    up: Vec3;
    mask: Uint8Array;
}

const SIZE = 384;

function normalize([x, y, z]: Vec3): Vec3 {
    const inv = 1 / Math.sqrt(x * x + y * y + z * z);
    return [x * inv, y * inv, z * inv];
}

function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cameraPosition(centre: Vec3, direction: Vec3, distance: number): Vec3 {
    return [centre[0] - direction[0] * distance, centre[1] - direction[1] * distance, centre[2] - direction[2] * distance];
}

function pointInConvexPolygon(x: number, y: number, polygon: readonly (readonly [number, number])[]): boolean {
    let sign = 0;
    for (let i = 0; i < polygon.length; i++) {
        const a = polygon[i]!;
        const b = polygon[(i + 1) % polygon.length]!;
        const side = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        if (Math.abs(side) < 1e-6) continue;
        const nextSign = side < 0 ? -1 : 1;
        if (sign !== 0 && sign !== nextSign) return false;
        sign = nextSign;
    }
    return true;
}

function separates(portal: RuntimePortal, first: Vec3, second: Vec3): boolean {
    const a = portal.corners[0];
    const b = portal.corners[1];
    const c = portal.corners[2];
    if (!a || !b || !c) return false;
    const normal = cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
    const side = (point: Vec3): number => dot(normal, [point[0] - a[0], point[1] - a[1], point[2] - a[2]]);
    return side(first) * side(second) < 0;
}

export function buildSkyPortalMask(
    view: { direction: Vec3; right: Vec3; up: Vec3 },
    portals: readonly RuntimePortal[],
    centre: Vec3,
    halfExtent: number,
    distance: number
): Uint8Array {
    const mask = new Uint8Array(SIZE * SIZE);
    const eye = cameraPosition(centre, view.direction, distance);
    for (const portal of portals) {
        const skyChunk = portal.chunkA === "__SKYBOX__" ? portal.chunkA : portal.chunkB === "__SKYBOX__" ? portal.chunkB : null;
        if (!skyChunk || !separates(portal, eye, centre)) continue;
        const polygon = portal.corners.map((corner) => {
            const relative: Vec3 = [corner[0] - centre[0], corner[1] - centre[1], corner[2] - centre[2]];
            return [((dot(relative, view.right) / halfExtent + 1) * SIZE) / 2, ((1 - dot(relative, view.up) / halfExtent) * SIZE) / 2] as const;
        });
        const minX = Math.max(0, Math.floor(Math.min(...polygon.map((point) => point[0]))));
        const maxX = Math.min(SIZE - 1, Math.ceil(Math.max(...polygon.map((point) => point[0]))));
        const minY = Math.max(0, Math.floor(Math.min(...polygon.map((point) => point[1]))));
        const maxY = Math.min(SIZE - 1, Math.ceil(Math.max(...polygon.map((point) => point[1]))));
        for (let y = minY; y <= maxY; y++) {
            for (let x = minX; x <= maxX; x++) {
                if (pointInConvexPolygon(x + 0.5, y + 0.5, polygon)) mask[y * SIZE + x] = 1;
            }
        }
    }
    return mask;
}

function buildViews(portals: readonly RuntimePortal[], centre: Vec3, halfExtent: number, distance: number): ExteriorView[] {
    const views: ExteriorView[] = [];
    for (let y = -1; y <= 1; y++) {
        for (let z = -1; z <= 1; z++) {
            for (let x = -1; x <= 1; x++) {
                if ((x === 0 && y === 0 && z === 0) || (x === 0 && z === 0)) continue;
                const direction = normalize([x, y, z]);
                const right = normalize(cross([0, 1, 0], direction));
                const up = cross(direction, right);
                const view = { direction, right, up };
                views.push({ ...view, mask: buildSkyPortalMask(view, portals, centre, halfExtent, distance) });
            }
        }
    }
    return views;
}

function idColor(id: number): [number, number, number] {
    return [(id & 0xff) / 255, ((id >>> 8) & 0xff) / 255, ((id >>> 16) & 0xff) / 255];
}

export function createExteriorMeshClassifier(options: ExteriorMeshClassifierOptions): ExteriorMeshClassifier {
    const { engine, scene, meshes, bounds, chunks, portals } = options;
    const centre = bounds.centre;
    const radius = Math.sqrt(bounds.half[0] * bounds.half[0] + bounds.half[1] * bounds.half[1] + bounds.half[2] * bounds.half[2]);
    const halfExtent = radius * 1.05;
    const distance = radius * 2.5;
    const runtimePortals = buildRuntimePortals(portals, chunks);
    const views = buildViews(runtimePortals, centre, halfExtent, distance);

    const camera = createFreeCamera(
        { x: centre[0] - views[0]!.direction[0] * distance, y: centre[1] - views[0]!.direction[1] * distance, z: centre[2] - views[0]!.direction[2] * distance },
        { x: centre[0], y: centre[1], z: centre[2] }
    );
    camera.nearPlane = Math.max(0.01, distance - radius * 1.2);
    camera.farPlane = distance + radius * 1.2;
    enableOrthographicCamera(camera, { halfHeight: halfExtent });

    const target = createRenderTarget({ lbl: "aq-exterior-id", format: "rgba8unorm", dFormat: "depth24plus", samples: 1, size: { width: SIZE, height: SIZE } });
    const renderTask = createRenderTask(
        {
            name: "aq-exterior-classifier",
            rt: target,
            clr: true,
            clrColor: { r: 0, g: 0, b: 0, a: 0 },
            cam: camera,
            autoMirror: false,
            _skipClusteredLights: true,
        },
        engine,
        scene
    );
    addTaskAtStart(scene, renderTask);

    // Seed the Standard material family in the main scene so per-task overrides can rebuild each
    // PBR ship mesh with the classifier's unlit ID material.
    const seedMaterial = createStandardMaterial();
    seedMaterial.disableLighting = true;
    const seed = createBox(engine, 0.001);
    seed.material = seedMaterial;
    setMeshVisible(seed, false);
    addToScene(scene, seed);

    const meshById: Mesh[] = [];
    for (const mesh of meshes) {
        const id = meshById.push(mesh);
        const material = createStandardMaterial();
        material.disableLighting = true;
        material.diffuseColor = [1, 1, 1];
        material.emissiveColor = idColor(id);
        material.specularColor = [0, 0, 0];
        material.alpha = 1;
        renderTask.addMesh(mesh, { material });
    }

    const bytesPerRow = Math.ceil((SIZE * 4) / 256) * 256;
    const staging = engine._device.createBuffer({
        label: "aq-exterior-readback",
        size: bytesPerRow * SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const exterior = new Set<Mesh>();
    let viewIndex = 0;
    let readbackInFlight = false;
    let resolveResult!: (result: ReadonlySet<Mesh>) => void;
    let rejectResult!: (error: unknown) => void;
    let readbackTask!: Task;
    const result = new Promise<ReadonlySet<Mesh>>((resolve, reject) => {
        resolveResult = resolve;
        rejectResult = reject;
    });

    const setView = (view: ExteriorView): void => {
        const eye = cameraPosition(centre, view.direction, distance);
        camera.position.set(eye[0], eye[1], eye[2]);
        camera.target.set(centre[0], centre[1], centre[2]);
    };

    const finishReadback = async (capturedView: number): Promise<void> => {
        try {
            await Promise.resolve();
            await staging.mapAsync(GPUMapMode.READ);
            const data = new Uint8Array(staging.getMappedRange());
            const mask = views[capturedView]!.mask;
            for (let y = 0; y < SIZE; y++) {
                const row = y * bytesPerRow;
                for (let x = 0; x < SIZE; x++) {
                    if (mask[y * SIZE + x]) continue;
                    const offset = row + x * 4;
                    const id = data[offset]! | (data[offset + 1]! << 8) | (data[offset + 2]! << 16);
                    const mesh = meshById[id - 1];
                    if (mesh) exterior.add(mesh);
                }
            }
            staging.unmap();
            viewIndex++;
            if (viewIndex >= views.length) {
                renderTask.enabled = false;
                staging.destroy();
                const tasks = scene._frameGraph._tasks;
                for (const task of [readbackTask, renderTask]) {
                    const index = tasks.indexOf(task);
                    if (index >= 0) tasks.splice(index, 1);
                    task.dispose();
                }
                removeFromScene(scene, seed);
                resolveResult(exterior);
                return;
            }
            setView(views[viewIndex]!);
            readbackInFlight = false;
        } catch (error) {
            renderTask.enabled = false;
            try {
                staging.unmap();
                staging.destroy();
            } catch {
                /* already unmapped or destroyed */
            }
            rejectResult(error);
        }
    };

    readbackTask = {
        name: "aq-exterior-readback",
        engine,
        scene,
        _passes: [],
        record: (): void => {},
        execute: (): number => {
            if (readbackInFlight || viewIndex >= views.length || !target._colorTexture) return 0;
            readbackInFlight = true;
            engine._currentEncoder.copyTextureToBuffer(
                { texture: target._colorTexture },
                { buffer: staging, bytesPerRow, rowsPerImage: SIZE },
                { width: SIZE, height: SIZE, depthOrArrayLayers: 1 }
            );
            void finishReadback(viewIndex);
            return 0;
        },
        dispose: (): void => {},
    };
    addTaskAfter(scene, readbackTask, renderTask);

    return { result };
}
