import { describe, expect, it } from "vitest";
import { fluidPrimitiveWireframeLines, visibleInjectedPrimitives } from "../../../../lab/lite/src/demos/aquanova/debug/collider-overlay";

describe("Aquanova collider overlay", () => {
    it("draws all twelve box edges at the injected half extents", () => {
        const lines = fluidPrimitiveWireframeLines({
            kind: "box",
            a: [10, 20, 30],
            b: [1, 2, 3],
        });

        expect(lines).toHaveLength(12);
        expect(lines.every((line) => line.length === 2)).toBe(true);
        const points = lines.flat();
        expect(Math.max(...points.map((point) => Math.abs(point.x)))).toBe(1);
        expect(Math.max(...points.map((point) => Math.abs(point.y)))).toBe(2);
        expect(Math.max(...points.map((point) => Math.abs(point.z)))).toBe(3);
    });

    it("draws a sphere as three closed great-circle rings", () => {
        const lines = fluidPrimitiveWireframeLines({
            kind: "sphere",
            a: [0, 0, 0],
            radius: 2,
        });

        expect(lines).toHaveLength(3);
        for (const line of lines) {
            expect(line).toHaveLength(25);
            expect(line[0]).toEqual(line.at(-1));
            for (const point of line) {
                expect(Math.hypot(point.x, point.y, point.z)).toBeCloseTo(2);
            }
        }
    });

    it("draws cylinder rings and ribs using the endpoint distance", () => {
        const lines = fluidPrimitiveWireframeLines({
            kind: "cylinder",
            a: [0, -2, 0],
            b: [0, 2, 0],
            radius: 1,
        });

        expect(lines).toHaveLength(10);
        expect(lines[0]![0]!.y).toBe(2);
        expect(lines[1]![0]!.y).toBe(-2);
        expect(lines.slice(2).every((line) => line[0]!.y === -2 && line[1]!.y === 2)).toBe(true);
    });

    it("draws both walls of a hollow cylinder", () => {
        const lines = fluidPrimitiveWireframeLines({
            kind: "hollowCylinder",
            a: [0, -2, 0],
            b: [0, 2, 0],
            innerRadius: 0.75,
            radius: 1,
        });

        expect(lines).toHaveLength(20);
        expect(lines.slice(0, 4).every((line) => line.length === 25 && line[0]!.y === line.at(-1)!.y)).toBe(true);
        expect(Math.hypot(lines[0]![0]!.x, lines[0]![0]!.z)).toBeCloseTo(1);
        expect(Math.hypot(lines[2]![0]!.x, lines[2]![0]!.z)).toBeCloseTo(0.75);
    });

    it("draws capsule rings, body ribs, and hemispherical poles", () => {
        const lines = fluidPrimitiveWireframeLines({
            kind: "capsule",
            a: [0, -2, 0],
            b: [0, 2, 0],
            radius: 1,
        });

        expect(lines).toHaveLength(10);
        expect(lines[0]![0]!.y).toBe(2);
        expect(lines[1]![0]!.y).toBe(-2);
        expect(lines[2]![0]).toEqual({ x: 0, y: 3, z: 0 });
        expect(lines[2]!.at(-1)!.y).toBe(-3);
    });

    it("omits the reserved player slot without removing it from the collision set", () => {
        const wall = { kind: "box", a: [0, 0, 0], b: [1, 1, 1] } as const;
        const player = { kind: "capsule", a: [2, 0, 0], b: [2, 1, 0], radius: 0.8 } as const;
        const inactive = { kind: "sphere", a: [4, 0, 0], radius: 1, active: false } as const;
        const collisionSet = [wall, player, inactive];

        expect(visibleInjectedPrimitives(collisionSet, 1)).toEqual([wall]);
        expect(collisionSet).toEqual([wall, player, inactive]);
    });
});
