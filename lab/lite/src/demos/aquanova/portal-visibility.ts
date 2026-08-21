import { getViewProjectionMatrix, setMeshVisible, type FreeCamera, type Mesh } from "babylon-lite";
import type { ShipChunk, ShipPortal } from "./manifest.js";

type Vec3 = readonly [number, number, number];

export interface Plane {
    x: number;
    y: number;
    z: number;
    d: number;
}

export interface WorldAabb {
    min: Vec3;
    max: Vec3;
}

export interface RuntimePortal {
    id: string;
    chunkA: string;
    chunkB: string;
    door?: string;
    centre: Vec3;
    corners: readonly Vec3[];
    enabled: boolean;
}

export interface PortalTraversal {
    portalId: string;
    fromChunk: string;
    toChunk: string;
    depth: number;
    corners: readonly Vec3[];
}

export interface PortalVisibilityStats {
    currentChunk: string;
    chunks: number;
    exteriorChunks: number;
    meshes: number;
    totalMeshes: number;
}

export interface PortalVisibility {
    update(): void;
    stats(): PortalVisibilityStats;
    viewerChunks(): readonly string[];
    exteriorChunks(): readonly string[];
    traversals(): readonly PortalTraversal[];
    meshOrder(mesh: Mesh): number | undefined;
    chunkIds(mesh: Mesh): string[];
    setPortalEnabled(id: string, enabled: boolean): boolean;
    setDoorEnabled(door: string, enabled: boolean): number;
    portalStates(): Array<{ id: string; door?: string; enabled: boolean }>;
}

export interface PortalGraphTraversalOptions {
    startChunks: readonly string[];
    cameraPosition: Vec3;
    cameraPlanes: readonly Plane[];
    portals: readonly RuntimePortal[];
    chunkExists: (chunk: string) => boolean;
    isLeafChunk?: (chunk: string, depth: number) => boolean;
    onChunk: (chunk: string, planes: readonly Plane[]) => void;
    onSkyPortal?: (portal: RuntimePortal, fromChunk: string, planes: readonly Plane[], depth: number) => void;
}

interface PortalVisibilityOptions {
    canvas: HTMLCanvasElement;
    camera: FreeCamera;
    aspectRatio: () => number;
    roomAt: () => string;
    viewerBounds?: () => WorldAabb;
    chunks: readonly ShipChunk[];
    portals: readonly ShipPortal[];
    meshes: readonly Mesh[];
    /** Manifest-authored membership for meshes that do not move. */
    chunkOfMesh: ReadonlyMap<Mesh, string>;
    isExcluded: (mesh: Mesh) => boolean;
    dynamicMeshes?: ReadonlySet<Mesh>;
    dynamicChunkOfMesh?: (mesh: Mesh) => string | undefined;
    exteriorMeshes?: () => ReadonlySet<Mesh>;
    canRestore?: (mesh: Mesh) => boolean;
}

const PLANE_EPSILON = 1e-5;

function toLite([x, y, z]: Vec3): Vec3 {
    return [-x, y, z];
}

function subtract(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function distanceSquared(a: Vec3, b: Vec3): number {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

function planeDistance(plane: Plane, point: Vec3): number {
    return plane.x * point[0] + plane.y * point[1] + plane.z * point[2] + plane.d;
}

function makePlane(x: number, y: number, z: number, d: number): Plane | null {
    const length = Math.hypot(x, y, z);
    return length > 1e-9 ? { x: x / length, y: y / length, z: z / length, d: d / length } : null;
}

function skyPortalChunk(portal: ShipPortal, chunks: readonly ShipChunk[]): string | undefined {
    if (portal.chunkA !== "__SKYBOX__" && portal.chunkB !== "__SKYBOX__") return undefined;
    let best: ShipChunk | undefined;
    let bestArea = Number.POSITIVE_INFINITY;
    for (const chunk of chunks) {
        const [minX, , minZ] = chunk.aabb.min;
        const [maxX, , maxZ] = chunk.aabb.max;
        if (portal.centre[0] < minX || portal.centre[0] > maxX || portal.centre[2] < minZ || portal.centre[2] > maxZ) continue;
        const area = (maxX - minX) * (maxZ - minZ);
        if (area < bestArea) {
            best = chunk;
            bestArea = area;
        }
    }
    return best?.id;
}

/** Convert the manifest's glTF-space portal records into Lite world space. */
export function buildRuntimePortals(portals: readonly ShipPortal[], chunks: readonly ShipChunk[]): RuntimePortal[] {
    const result: RuntimePortal[] = [];
    for (const portal of portals) {
        if (!portal.id || !portal.chunkA || !portal.chunkB || !portal.corners || portal.corners.length < 3) continue;
        let chunkA = portal.chunkA;
        let chunkB = portal.chunkB;
        const spatialSkyChunk = skyPortalChunk(portal, chunks);
        if (spatialSkyChunk) {
            if (chunkA === "__SKYBOX__") chunkB = spatialSkyChunk;
            else chunkA = spatialSkyChunk;
        }
        const centre = toLite(portal.centre);
        const corners = portal.corners.map(toLite);
        result.push({
            id: portal.id,
            chunkA,
            chunkB,
            ...(portal.door ? { door: portal.door } : {}),
            centre,
            corners,
            enabled: portal.enabled !== false,
        });
    }
    return result;
}

/** Extract Lite/WebGPU reverse-Z frustum planes. Inside is the non-negative half-space. */
export function extractCameraFrustumPlanes(matrix: ArrayLike<number>): Plane[] {
    const values: Array<[number, number, number, number]> = [
        [matrix[3]! + matrix[0]!, matrix[7]! + matrix[4]!, matrix[11]! + matrix[8]!, matrix[15]! + matrix[12]!],
        [matrix[3]! - matrix[0]!, matrix[7]! - matrix[4]!, matrix[11]! - matrix[8]!, matrix[15]! - matrix[12]!],
        [matrix[3]! + matrix[1]!, matrix[7]! + matrix[5]!, matrix[11]! + matrix[9]!, matrix[15]! + matrix[13]!],
        [matrix[3]! - matrix[1]!, matrix[7]! - matrix[5]!, matrix[11]! - matrix[9]!, matrix[15]! - matrix[13]!],
        [matrix[2]!, matrix[6]!, matrix[10]!, matrix[14]!],
        [matrix[3]! - matrix[2]!, matrix[7]! - matrix[6]!, matrix[11]! - matrix[10]!, matrix[15]! - matrix[14]!],
    ];
    return values.map(([x, y, z, d]) => makePlane(x, y, z, d)).filter((plane): plane is Plane => plane !== null);
}

/**
 * Planes added after crossing a portal: one per edge plus the portal's destination-facing plane.
 * The corners define the plane; the camera side defines its orientation. Chunk AABB centres are
 * deliberately irrelevant because connected chunk volumes may overlap or contain one another.
 */
export function createPortalFrustumPlanes(portal: RuntimePortal, cameraPosition: Vec3): Plane[] {
    const planes: Plane[] = [];
    const first = portal.corners[0];
    const second = portal.corners[1];
    const third = portal.corners[2];
    const geometryNormal = first && second && third ? cross(subtract(second, first), subtract(third, first)) : null;
    if (geometryNormal) {
        const cameraDistance = dot(geometryNormal, subtract(cameraPosition, portal.centre));
        if (Math.abs(cameraDistance) > PLANE_EPSILON) {
            const sourceNormal: Vec3 = cameraDistance > 0 ? geometryNormal : [-geometryNormal[0], -geometryNormal[1], -geometryNormal[2]];
            const destination = makePlane(
                -sourceNormal[0],
                -sourceNormal[1],
                -sourceNormal[2],
                sourceNormal[0] * portal.centre[0] + sourceNormal[1] * portal.centre[1] + sourceNormal[2] * portal.centre[2]
            );
            if (destination) planes.push(destination);
        }
    }

    for (let i = 0; i < portal.corners.length; i++) {
        const a = portal.corners[i]!;
        const b = portal.corners[(i + 1) % portal.corners.length]!;
        const normal = cross(subtract(a, cameraPosition), subtract(b, cameraPosition));
        let plane = makePlane(normal[0], normal[1], normal[2], -(normal[0] * cameraPosition[0] + normal[1] * cameraPosition[1] + normal[2] * cameraPosition[2]));
        if (!plane) continue;
        if (planeDistance(plane, portal.centre) < 0) {
            plane = { x: -plane.x, y: -plane.y, z: -plane.z, d: -plane.d };
        }
        planes.push(plane);
    }
    return planes;
}

/** Conservative world-AABB against convex planes test. */
export function aabbIntersectsPlanes(bounds: WorldAabb, planes: readonly Plane[]): boolean {
    const centre: Vec3 = [(bounds.min[0] + bounds.max[0]) * 0.5, (bounds.min[1] + bounds.max[1]) * 0.5, (bounds.min[2] + bounds.max[2]) * 0.5];
    const half: Vec3 = [(bounds.max[0] - bounds.min[0]) * 0.5, (bounds.max[1] - bounds.min[1]) * 0.5, (bounds.max[2] - bounds.min[2]) * 0.5];
    for (const plane of planes) {
        const distance = planeDistance(plane, centre);
        const radius = Math.abs(plane.x) * half[0] + Math.abs(plane.y) * half[1] + Math.abs(plane.z) * half[2];
        if (distance + radius < -PLANE_EPSILON) return false;
    }
    return true;
}

/** Exact convex-polygon/convex-volume intersection by successive plane clipping. */
export function polygonIntersectsPlanes(corners: readonly Vec3[], planes: readonly Plane[]): boolean {
    let polygon = corners.slice();
    for (const plane of planes) {
        if (polygon.length < 3) return false;
        const clipped: Vec3[] = [];
        for (let i = 0; i < polygon.length; i++) {
            const a = polygon[i]!;
            const b = polygon[(i + 1) % polygon.length]!;
            const da = planeDistance(plane, a);
            const db = planeDistance(plane, b);
            const aInside = da >= -PLANE_EPSILON;
            const bInside = db >= -PLANE_EPSILON;
            if (aInside) clipped.push(a);
            if (aInside === bInside) continue;
            const t = da / (da - db);
            clipped.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
        }
        polygon = clipped;
    }
    return polygon.length >= 3;
}

/** Whether the portal polygon crosses a mesh's conservative world AABB. */
export function portalIntersectsAabb(portal: RuntimePortal, bounds: WorldAabb): boolean {
    return polygonIntersectsPlanes(portal.corners, [
        { x: 1, y: 0, z: 0, d: -bounds.min[0] },
        { x: -1, y: 0, z: 0, d: bounds.max[0] },
        { x: 0, y: 1, z: 0, d: -bounds.min[1] },
        { x: 0, y: -1, z: 0, d: bounds.max[1] },
        { x: 0, y: 0, z: 1, d: -bounds.min[2] },
        { x: 0, y: 0, z: -1, d: bounds.max[2] },
    ]);
}

/** Convert a manifest chunk AABB from glTF space into Lite world space. */
export function chunkWorldAabb(chunk: ShipChunk): WorldAabb {
    return {
        min: [-chunk.aabb.max[0], chunk.aabb.min[1], chunk.aabb.min[2]],
        max: [-chunk.aabb.min[0], chunk.aabb.max[1], chunk.aabb.max[2]],
    };
}

/** Recursive chunk traversal. Each root chunk starts with the full camera frustum. */
export function traversePortalGraph(options: PortalGraphTraversalOptions): PortalTraversal[] {
    const byChunk = new Map<string, RuntimePortal[]>();
    for (const portal of options.portals) {
        const a = byChunk.get(portal.chunkA);
        if (a) a.push(portal);
        else byChunk.set(portal.chunkA, [portal]);
        const b = byChunk.get(portal.chunkB);
        if (b) b.push(portal);
        else byChunk.set(portal.chunkB, [portal]);
    }

    const traversals: PortalTraversal[] = [];
    const rootChunks = new Set(options.startChunks);
    const queue: Array<{ chunk: string; planes: readonly Plane[]; depth: number; path: ReadonlySet<string> }> = [...rootChunks].map((chunk) => ({
        chunk,
        planes: options.cameraPlanes,
        depth: 0,
        path: new Set([chunk]),
    }));
    for (let cursor = 0; cursor < queue.length; cursor++) {
        const state = queue[cursor]!;
        options.onChunk(state.chunk, state.planes);
        if (state.depth > options.portals.length) continue;
        const candidates = (byChunk.get(state.chunk) ?? [])
            .filter((portal) => portal.enabled && polygonIntersectsPlanes(portal.corners, state.planes))
            .sort((a, b) => {
                const aSky = (state.chunk === a.chunkA ? a.chunkB : a.chunkA) === "__SKYBOX__";
                const bSky = (state.chunk === b.chunkA ? b.chunkB : b.chunkA) === "__SKYBOX__";
                return Number(bSky) - Number(aSky) || distanceSquared(a.centre, options.cameraPosition) - distanceSquared(b.centre, options.cameraPosition);
            });
        for (const portal of candidates) {
            const toChunk = state.chunk === portal.chunkA ? portal.chunkB : portal.chunkA;
            const nextPlanes = [...state.planes, ...createPortalFrustumPlanes(portal, options.cameraPosition)];
            if (!options.chunkExists(toChunk)) {
                if (toChunk === "__SKYBOX__" && options.onSkyPortal) {
                    traversals.push({ portalId: portal.id, fromChunk: state.chunk, toChunk, depth: state.depth, corners: portal.corners });
                    options.onSkyPortal(portal, state.chunk, nextPlanes, state.depth);
                }
                continue;
            }
            if (rootChunks.has(toChunk) || options.isLeafChunk?.(toChunk, state.depth + 1) || state.path.has(toChunk)) continue;
            traversals.push({ portalId: portal.id, fromChunk: state.chunk, toChunk, depth: state.depth, corners: portal.corners });
            const nextPath = new Set(state.path);
            nextPath.add(toChunk);
            queue.push({ chunk: toChunk, planes: nextPlanes, depth: state.depth + 1, path: nextPath });
        }
    }
    return traversals;
}

function meshWorldAabb(mesh: Mesh): WorldAabb | null {
    const min = mesh.boundMin;
    const max = mesh.boundMax;
    if (!min || !max) return null;
    const centre = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
    const half = [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5, (max[2] - min[2]) * 0.5];
    const world = mesh.worldMatrix;
    const worldCentre: number[] = [];
    const worldHalf: number[] = [];
    for (let row = 0; row < 3; row++) {
        worldCentre[row] = world[12 + row]! + world[row]! * centre[0]! + world[4 + row]! * centre[1]! + world[8 + row]! * centre[2]!;
        worldHalf[row] = Math.abs(world[row]!) * half[0]! + Math.abs(world[4 + row]!) * half[1]! + Math.abs(world[8 + row]!) * half[2]!;
    }
    return {
        min: [worldCentre[0]! - worldHalf[0]!, worldCentre[1]! - worldHalf[1]!, worldCentre[2]! - worldHalf[2]!],
        max: [worldCentre[0]! + worldHalf[0]!, worldCentre[1]! + worldHalf[1]!, worldCentre[2]! + worldHalf[2]!],
    };
}

export function groupMeshesByChunks<T>(items: readonly T[], chunksOf: (item: T) => Iterable<string>): { byChunk: Map<string, T[]>; unassigned: T[] } {
    const byChunk = new Map<string, T[]>();
    const unassigned: T[] = [];
    for (const item of items) {
        const chunks = new Set(chunksOf(item));
        if (chunks.size === 0) {
            unassigned.push(item);
            continue;
        }
        for (const chunk of chunks) {
            const group = byChunk.get(chunk);
            if (group) group.push(item);
            else byChunk.set(chunk, [item]);
        }
    }
    return { byChunk, unassigned };
}

export function createPortalVisibility(options: PortalVisibilityOptions): PortalVisibility {
    const runtimePortals = buildRuntimePortals(options.portals, options.chunks);
    const chunkIds = new Set(options.chunks.map((chunk) => chunk.id));
    const chunkBounds = new Map(options.chunks.map((chunk) => [chunk.id, chunkWorldAabb(chunk)]));
    const meshes = options.meshes.filter((mesh) => !options.isExcluded(mesh));
    const dynamicMeshes = meshes.filter((mesh) => options.dynamicMeshes?.has(mesh));
    const staticMeshes = meshes.filter((mesh) => !options.dynamicMeshes?.has(mesh));
    const chunksForBounds = (bounds: WorldAabb | null, baseChunk: string | undefined): Set<string> => {
        const chunks = new Set<string>();
        if (baseChunk && chunkIds.has(baseChunk)) chunks.add(baseChunk);
        if (!bounds) return chunks;
        for (const portal of runtimePortals) {
            if (!portalIntersectsAabb(portal, bounds)) continue;
            if (chunkIds.has(portal.chunkA)) chunks.add(portal.chunkA);
            if (chunkIds.has(portal.chunkB)) chunks.add(portal.chunkB);
        }
        return chunks;
    };
    const dynamicChunksForMesh = (mesh: Mesh): Set<string> => chunksForBounds(meshWorldAabb(mesh), options.dynamicChunkOfMesh?.(mesh));
    const manifestChunksForMesh = (mesh: Mesh): string[] => {
        const chunk = options.chunkOfMesh.get(mesh);
        return chunk && chunkIds.has(chunk) ? [chunk] : [];
    };
    const { byChunk: meshesByChunk, unassigned: unchunked } = groupMeshesByChunks(staticMeshes, manifestChunksForMesh);

    const portalHidden = new Set<Mesh>();
    let currentStats: PortalVisibilityStats = {
        currentChunk: "—",
        chunks: options.chunks.length,
        exteriorChunks: 0,
        meshes: meshes.filter((mesh) => mesh.visible !== false).length,
        totalMeshes: meshes.length,
    };
    let currentViewerChunks: string[] = [];
    let currentExteriorChunks: string[] = [];
    let currentTraversals: PortalTraversal[] = [];
    let currentMeshOrder = new Map<Mesh, number>();

    const applyVisibility = (visible: ReadonlySet<Mesh>): number => {
        let displayed = 0;
        for (const mesh of meshes) {
            const shouldDisplay = visible.has(mesh);
            if (shouldDisplay) {
                if (portalHidden.has(mesh)) {
                    portalHidden.delete(mesh);
                    if (options.canRestore?.(mesh) !== false) setMeshVisible(mesh, true);
                }
            } else if (mesh.visible !== false) {
                portalHidden.add(mesh);
                setMeshVisible(mesh, false);
            }
            if (mesh.visible !== false) displayed++;
        }
        return displayed;
    };

    return {
        update(): void {
            const startChunk = options.roomAt();
            const startChunks = [...chunksForBounds(options.viewerBounds?.() ?? null, startChunk)];
            currentViewerChunks = startChunks;
            const visible = new Set<Mesh>();
            const displayedChunks = new Set<string>();
            const displayedExteriorChunks = new Map<string, number>();
            const ordered: Mesh[] = [];
            const { byChunk: dynamicMeshesByChunk } = groupMeshesByChunks(dynamicMeshes, dynamicChunksForMesh);
            const addVisible = (mesh: Mesh): void => {
                if (visible.has(mesh)) return;
                visible.add(mesh);
                ordered.push(mesh);
            };
            const addChunkCandidates = (chunk: string, planes: readonly Plane[], cameraPosition: Vec3, allowed?: ReadonlySet<Mesh>): void => {
                const candidates: Array<{ mesh: Mesh; distance: number }> = [];
                for (const mesh of [...(meshesByChunk.get(chunk) ?? []), ...(dynamicMeshesByChunk.get(chunk) ?? [])]) {
                    if (allowed && !allowed.has(mesh)) continue;
                    const bounds = meshWorldAabb(mesh);
                    if (bounds && !aabbIntersectsPlanes(bounds, planes)) continue;
                    const centre: Vec3 = bounds
                        ? [(bounds.min[0] + bounds.max[0]) * 0.5, (bounds.min[1] + bounds.max[1]) * 0.5, (bounds.min[2] + bounds.max[2]) * 0.5]
                        : [mesh.worldMatrix[12]!, mesh.worldMatrix[13]!, mesh.worldMatrix[14]!];
                    candidates.push({ mesh, distance: distanceSquared(centre, cameraPosition) });
                }
                candidates.sort((a, b) => a.distance - b.distance);
                for (const candidate of candidates) addVisible(candidate.mesh);
            };
            if (startChunks.length === 0 || runtimePortals.length === 0) {
                for (const mesh of meshes) addVisible(mesh);
                for (const chunk of chunkIds) displayedChunks.add(chunk);
                currentTraversals = [];
            } else {
                const cameraPosition: Vec3 = [options.camera.position.x, options.camera.position.y, options.camera.position.z];
                const cameraPlanes = extractCameraFrustumPlanes(getViewProjectionMatrix(options.camera, options.aspectRatio()));
                currentTraversals = traversePortalGraph({
                    startChunks,
                    cameraPosition,
                    cameraPlanes,
                    portals: runtimePortals,
                    chunkExists: (chunk) => chunkIds.has(chunk),
                    isLeafChunk: (chunk, depth) => {
                        const exteriorDepth = displayedExteriorChunks.get(chunk);
                        if (exteriorDepth === undefined) return false;
                        if (depth <= exteriorDepth) {
                            displayedExteriorChunks.delete(chunk);
                            return false;
                        }
                        return true;
                    },
                    onChunk: (chunk, planes) => {
                        displayedChunks.add(chunk);
                        addChunkCandidates(chunk, planes, cameraPosition);
                    },
                    onSkyPortal: (_portal, _fromChunk, planes, depth) => {
                        const exterior = options.exteriorMeshes?.();
                        if (!exterior || exterior.size === 0) return;
                        const candidates: Array<{ chunk: string; distance: number }> = [];
                        for (const [chunk, bounds] of chunkBounds) {
                            if (!aabbIntersectsPlanes(bounds, planes)) continue;
                            const centre: Vec3 = [(bounds.min[0] + bounds.max[0]) * 0.5, (bounds.min[1] + bounds.max[1]) * 0.5, (bounds.min[2] + bounds.max[2]) * 0.5];
                            candidates.push({ chunk, distance: distanceSquared(centre, cameraPosition) });
                        }
                        candidates.sort((a, b) => a.distance - b.distance);
                        for (const candidate of candidates) {
                            if (!displayedChunks.has(candidate.chunk)) {
                                const pathDepth = depth + 1;
                                displayedExteriorChunks.set(candidate.chunk, Math.min(displayedExteriorChunks.get(candidate.chunk) ?? Number.POSITIVE_INFINITY, pathDepth));
                            }
                            addChunkCandidates(candidate.chunk, planes, cameraPosition, exterior);
                        }
                    },
                });
                for (const mesh of unchunked) addVisible(mesh);
            }
            currentExteriorChunks = [...displayedExteriorChunks.keys()];
            currentMeshOrder = new Map(ordered.map((mesh, index) => [mesh, index]));
            currentStats = {
                currentChunk: startChunk,
                chunks: displayedChunks.size,
                exteriorChunks: currentExteriorChunks.length,
                meshes: applyVisibility(visible),
                totalMeshes: meshes.length,
            };
            options.canvas.dataset.portalChunk = startChunk;
            options.canvas.dataset.portalViewerChunks = startChunks.join(",");
            options.canvas.dataset.portalChunks = String(currentStats.chunks);
            options.canvas.dataset.portalExteriorChunks = currentExteriorChunks.join(",");
            options.canvas.dataset.portalMeshes = `${currentStats.meshes}/${currentStats.totalMeshes}`;
        },
        stats: () => currentStats,
        viewerChunks: () => currentViewerChunks,
        exteriorChunks: () => currentExteriorChunks,
        traversals: () => currentTraversals,
        meshOrder: (mesh) => currentMeshOrder.get(mesh),
        chunkIds: (mesh) => (options.dynamicMeshes?.has(mesh) ? [...dynamicChunksForMesh(mesh)].sort() : manifestChunksForMesh(mesh)),
        setPortalEnabled(id: string, enabled: boolean): boolean {
            const portal = runtimePortals.find((candidate) => candidate.id === id);
            if (!portal) return false;
            portal.enabled = enabled;
            return true;
        },
        setDoorEnabled(door: string, enabled: boolean): number {
            let changed = 0;
            for (const portal of runtimePortals) {
                if (portal.door !== door || portal.enabled === enabled) continue;
                portal.enabled = enabled;
                changed++;
            }
            return changed;
        },
        portalStates: () => runtimePortals.map(({ id, door, enabled }) => ({ id, ...(door ? { door } : {}), enabled })),
    };
}
