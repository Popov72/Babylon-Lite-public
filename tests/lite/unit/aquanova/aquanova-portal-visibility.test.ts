import { describe, expect, it } from "vitest";
import {
    aabbIntersectsPlanes,
    buildRuntimePortals,
    chunkWorldAabb,
    createPortalFrustumPlanes,
    groupMeshesByChunks,
    polygonIntersectsPlanes,
    portalIntersectsAabb,
    traversePortalGraph,
    type Plane,
    type RuntimePortal,
} from "../../../../lab/lite/src/demos/aquanova/portal-visibility";
import type { ShipChunk, ShipPortal } from "../../../../lab/lite/src/demos/aquanova/manifest";

function portal(id: string, chunkA: string, chunkB: string, z: number, enabled = true): RuntimePortal {
    return {
        id,
        chunkA,
        chunkB,
        centre: [0, 0, z],
        corners: [
            [-1, -1, z],
            [1, -1, z],
            [1, 1, z],
            [-1, 1, z],
        ],
        enabled,
    };
}

describe("Aquanova portal visibility", () => {
    it("converts manifest portal coordinates to Lite space", () => {
        const chunks: ShipChunk[] = [
            { id: "A", aabb: { min: [-4, -1, -1], max: [0, 1, 1] } },
            { id: "B", aabb: { min: [0, -1, -1], max: [4, 1, 1] } },
        ];
        const source: ShipPortal = {
            id: "P",
            chunkA: "A",
            chunkB: "B",
            centre: [-1, 0, 0],
            normal: [-1, 0, 0],
            corners: [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
            ],
        };

        const [runtime] = buildRuntimePortals([source], chunks);

        expect(runtime?.centre).toEqual([1, 0, 0]);
        expect(runtime?.corners[0]).toEqual([1, -1, -1]);
        expect(runtime?.enabled).toBe(true);
    });

    it("does not retain an authored normal whose chunk centres cannot orient an overlapping portal", () => {
        const chunks: ShipChunk[] = [
            { id: "A", aabb: { min: [-5, -1, -1], max: [5, 1, 1] } },
            { id: "B", aabb: { min: [-3, -1, -1], max: [-1, 1, 1] } },
        ];
        const source: ShipPortal = {
            id: "P",
            chunkA: "A",
            chunkB: "B",
            centre: [-3, 0, 0],
            normal: [-1, 0, 0],
            corners: [
                [-3, -1, -1],
                [-3, -1, 1],
                [-3, 1, 1],
                [-3, 1, -1],
            ],
        };

        expect(buildRuntimePortals([source], chunks)[0]).not.toHaveProperty("normal");
    });

    it("repairs stale sky-portal ownership from the portal position", () => {
        const chunks: ShipChunk[] = [
            { id: "CH01", aabb: { min: [-4, -1, -4], max: [4, 4, 4] } },
            { id: "CH03", aabb: { min: [8, -1, -12], max: [16, 4, -4] } },
        ];
        const source: ShipPortal = {
            id: "outside",
            chunkA: "CH01",
            chunkB: "__SKYBOX__",
            centre: [12, 1, -6],
            normal: [1, 0, 0],
            corners: [
                [12, 0, -7],
                [12, 0, -5],
                [12, 2, -5],
                [12, 2, -7],
            ],
        };

        expect(buildRuntimePortals([source], chunks)[0]?.chunkA).toBe("CH03");
    });

    it("culls destination meshes against portal edge and depth planes", () => {
        const p = portal("P", "A", "B", 2);
        const planes = createPortalFrustumPlanes(p, [0, 0, 0]);

        expect(aabbIntersectsPlanes({ min: [-0.25, -0.25, 3.75], max: [0.25, 0.25, 4.25] }, planes)).toBe(true);
        expect(aabbIntersectsPlanes({ min: [2.75, -0.25, 3.75], max: [3.25, 0.25, 4.25] }, planes)).toBe(false);
        expect(aabbIntersectsPlanes({ min: [-0.25, -0.25, 0.75], max: [0.25, 0.25, 1.25] }, planes)).toBe(false);
    });

    it("orients the portal depth plane from the camera", () => {
        const p = portal("P", "A", "B", 2);
        const fromA = createPortalFrustumPlanes(p, [0, 0, 0]);
        const fromB = createPortalFrustumPlanes(p, [0, 0, 4]);

        expect(aabbIntersectsPlanes({ min: [-0.1, -0.1, 3], max: [0.1, 0.1, 3.2] }, fromA)).toBe(true);
        expect(aabbIntersectsPlanes({ min: [-0.1, -0.1, 0.8], max: [0.1, 0.1, 1] }, fromA)).toBe(false);
        expect(aabbIntersectsPlanes({ min: [-0.1, -0.1, 0.8], max: [0.1, 0.1, 1] }, fromB)).toBe(true);
        expect(aabbIntersectsPlanes({ min: [-0.1, -0.1, 3], max: [0.1, 0.1, 3.2] }, fromB)).toBe(false);
    });

    it("recognizes a portal crossing a clipping volume even when no corner is fully inside", () => {
        const planes = [
            { x: 1, y: 0, z: 0, d: 0.25 },
            { x: -1, y: 0, z: 0, d: 0.25 },
            { x: 0, y: 1, z: 0, d: 0.25 },
            { x: 0, y: -1, z: 0, d: 0.25 },
        ];
        const corners = [
            [-1, -1, 2],
            [1, -1, 2],
            [1, 1, 2],
            [-1, 1, 2],
        ] as const;

        expect(polygonIntersectsPlanes(corners, planes)).toBe(true);
    });

    it("recognizes when a mesh AABB straddles a portal", () => {
        const p = portal("P", "A", "B", 2);

        expect(portalIntersectsAabb(p, { min: [-0.5, -0.5, 1.5], max: [0.5, 0.5, 2.5] })).toBe(true);
        expect(portalIntersectsAabb(p, { min: [2, -0.5, 1.5], max: [3, 0.5, 2.5] })).toBe(false);
    });

    it("does not traverse a disabled portal", () => {
        const visited: string[] = [];
        const traversals = traversePortalGraph({
            startChunks: ["A"],
            cameraPosition: [0, 0, 0],
            cameraPlanes: [],
            portals: [portal("AB", "A", "B", 2), portal("BC", "B", "C", 4, false)],
            chunkExists: () => true,
            onChunk: (chunk) => visited.push(chunk),
        });

        expect(visited).toEqual(["A", "B"]);
        expect(traversals.map((entry) => entry.portalId)).toEqual(["AB"]);
    });

    it("traverses a portal between overlapping chunks from the camera's geometric side", () => {
        const chunks: ShipChunk[] = [
            { id: "A", aabb: { min: [-5, -1, -1], max: [5, 1, 1] } },
            { id: "B", aabb: { min: [-3, -1, -1], max: [-1, 1, 1] } },
        ];
        const [overlapping] = buildRuntimePortals(
            [
                {
                    id: "AB",
                    chunkA: "A",
                    chunkB: "B",
                    centre: [-3, 0, 0],
                    normal: [-1, 0, 0],
                    corners: [
                        [-3, -1, -1],
                        [-3, -1, 1],
                        [-3, 1, 1],
                        [-3, 1, -1],
                    ],
                },
            ],
            chunks
        );
        const visited: string[] = [];
        const traversals = traversePortalGraph({
            startChunks: ["A"],
            cameraPosition: [4, 0, 0],
            cameraPlanes: [],
            portals: [overlapping!],
            chunkExists: () => true,
            onChunk: (chunk) => visited.push(chunk),
        });

        expect(visited).toEqual(["A", "B"]);
        expect(traversals.map((entry) => entry.portalId)).toEqual(["AB"]);
    });

    it("does not traverse portals leading to the skybox pseudo-chunk", () => {
        const visited: string[] = [];
        const traversals = traversePortalGraph({
            startChunks: ["A"],
            cameraPosition: [0, 0, 0],
            cameraPlanes: [],
            portals: [portal("outside", "A", "__SKYBOX__", 2)],
            chunkExists: (chunk) => chunk === "A",
            onChunk: (chunk) => visited.push(chunk),
        });

        expect(visited).toEqual(["A"]);
        expect(traversals).toEqual([]);
    });

    it("reports a visible sky portal without traversing into the pseudo-chunk", () => {
        const skyPortals: string[] = [];
        const visited: string[] = [];
        const traversals = traversePortalGraph({
            startChunks: ["A"],
            cameraPosition: [0, 0, 0],
            cameraPlanes: [],
            portals: [portal("outside", "A", "__SKYBOX__", 2)],
            chunkExists: (chunk) => chunk === "A",
            onChunk: (chunk) => visited.push(chunk),
            onSkyPortal: (entry) => skyPortals.push(entry.id),
        });

        expect(skyPortals).toEqual(["outside"]);
        expect(visited).toEqual(["A"]);
        expect(traversals.map((entry) => entry.toChunk)).toEqual(["__SKYBOX__"]);
    });

    it("visits nearer portals before farther portals at the same depth", () => {
        const traversals = traversePortalGraph({
            startChunks: ["A"],
            cameraPosition: [0, 0, 0],
            cameraPlanes: [],
            portals: [portal("far", "A", "C", 5), portal("near", "A", "B", 2)],
            chunkExists: () => true,
            onChunk: () => {},
        });

        expect(traversals.map((entry) => entry.portalId)).toEqual(["near", "far"]);
    });

    it("deduplicates a mesh visible through multiple portal paths", () => {
        const visibleMeshes = new Set<string>();
        const traversals = traversePortalGraph({
            startChunks: ["A"],
            cameraPosition: [0, 0, 0],
            cameraPlanes: [],
            portals: [portal("AB-left", "A", "B", 2), portal("AB-right", "A", "B", 3)],
            chunkExists: () => true,
            onChunk: (chunk) => visibleMeshes.add(`mesh-${chunk}`),
        });

        expect(traversals).toHaveLength(2);
        expect(visibleMeshes).toEqual(new Set(["mesh-A", "mesh-B"]));
    });

    it("fully processes every root chunk touched by the player", () => {
        const cameraPlanes = [{ x: 1, y: 0, z: 0, d: -2 }];
        const visited: Array<{ chunk: string; planes: readonly Plane[] }> = [];
        const traversals = traversePortalGraph({
            startChunks: ["A", "B"],
            cameraPosition: [0, 0, 0],
            cameraPlanes,
            portals: [portal("AB", "A", "B", 2)],
            chunkExists: () => true,
            onChunk: (chunk, planes) => visited.push({ chunk, planes }),
        });

        expect(visited.map((entry) => entry.chunk)).toEqual(["A", "B"]);
        expect(visited.every((entry) => entry.planes === cameraPlanes)).toBe(true);
        expect(traversals).toEqual([]);
    });

    it("lets an equal-depth normal path override exterior classification", () => {
        const leafDepths = new Map<string, number>();
        const visited: string[] = [];
        const traversals = traversePortalGraph({
            startChunks: ["A"],
            cameraPosition: [0, 0, 0],
            cameraPlanes: [],
            portals: [portal("AB-near", "A", "B", 2), portal("outside-far", "A", "__SKYBOX__", 4)],
            chunkExists: (chunk) => chunk !== "__SKYBOX__",
            isLeafChunk: (chunk, depth) => {
                const leafDepth = leafDepths.get(chunk);
                if (leafDepth === undefined || depth <= leafDepth) {
                    leafDepths.delete(chunk);
                    return false;
                }
                return true;
            },
            onChunk: (chunk) => visited.push(chunk),
            onSkyPortal: () => leafDepths.set("B", 1),
        });

        expect(visited).toEqual(["A", "B"]);
        expect(traversals.map((entry) => entry.portalId)).toEqual(["outside-far", "AB-near"]);
    });

    it("blocks a longer normal path to a sky-selected exterior chunk", () => {
        const leafDepths = new Map<string, number>();
        const visited: string[] = [];
        const traversals = traversePortalGraph({
            startChunks: ["A"],
            cameraPosition: [0, 0, 0],
            cameraPlanes: [],
            portals: [portal("AB", "A", "B", 2), portal("BC", "B", "C", 4), portal("outside", "A", "__SKYBOX__", 6)],
            chunkExists: (chunk) => chunk !== "__SKYBOX__",
            isLeafChunk: (chunk, depth) => (leafDepths.get(chunk) ?? Number.POSITIVE_INFINITY) < depth,
            onChunk: (chunk) => visited.push(chunk),
            onSkyPortal: () => leafDepths.set("C", 1),
        });

        expect(visited).toEqual(["A", "B"]);
        expect(traversals.map((entry) => entry.portalId)).toEqual(["outside", "AB"]);
    });

    it("groups a portal-straddling mesh into every intersected chunk", () => {
        const mesh = {};

        const grouped = groupMeshesByChunks([mesh], () => ["CH02", "CH03"]);
        expect(grouped.byChunk.get("CH02")).toEqual([mesh]);
        expect(grouped.byChunk.get("CH03")).toEqual([mesh]);
    });

    it("converts chunk bounds from glTF space to Lite world space", () => {
        expect(chunkWorldAabb({ id: "A", aabb: { min: [-4, 1, 2], max: [6, 3, 8] } })).toEqual({
            min: [-6, 1, 2],
            max: [4, 3, 8],
        });
    });
});
