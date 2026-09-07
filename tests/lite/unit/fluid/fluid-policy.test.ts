import { describe, expect, it } from "vitest";

import { resolveFluidImpulseForce } from "../../../../packages/babylon-lite/src/fluid/core/fluid-facade";
import { resolveFluidGridCompatibility } from "../../../../packages/babylon-lite/src/fluid/core/allocation-plan";
import {
    fitFluidGridResolution,
    normalizeFluidFlipDiscretization,
    resolveFluidRenderMode,
    transformFluidFlow,
} from "../../../../packages/babylon-lite/src/fluid/core/fluid-policy";
import type { FluidFlowConfig } from "../../../../packages/babylon-lite/src/fluid/core/sim-common";

const flow: FluidFlowConfig = {
    emitters: [
        {
            id: "world",
            name: "World",
            enabled: true,
            behavior: "inflow",
            transform: {
                position: [1, 0, 0],
                rotation: [0, 0, 0, 1],
                scale: [1, 2, 3],
            },
            shape: { type: "sphere", radius: 1 },
            sampling: "volume",
            velocity: [1, 0, 0],
            velocitySpace: "world",
            sourceVelocity: [0, 0, 1],
            spread: 0,
        },
        {
            id: "local",
            name: "Local",
            enabled: true,
            behavior: "inflow",
            transform: {
                position: [0, 1, 0],
                rotation: [0, 0, 0, 1],
                scale: [1, 1, 1],
            },
            shape: { type: "box", size: [1, 1, 1] },
            sampling: "volume",
            velocity: [1, 0, 0],
            velocitySpace: "local",
            spread: 0,
        },
    ],
    sinks: [],
};

describe("shared fluid policy", () => {
    it("transforms flow placement and world velocity without changing local velocity", () => {
        const halfTurnZ: [number, number, number, number] = [0, 0, 1, 0];
        const transformed = transformFluidFlow(flow, {
            translation: [10, 20, 30],
            rotation: halfTurnZ,
            scale: [2, 3, 4],
        });

        expect(transformed.emitters[0]!.transform.position).toEqual([8, 20, 30]);
        expect(transformed.emitters[0]!.transform.scale).toEqual([2, 6, 12]);
        expect(transformed.emitters[0]!.velocity).toEqual([-1, 0, 0]);
        expect(transformed.emitters[0]!.sourceVelocity).toEqual([0, 0, 1]);
        expect(transformed.emitters[1]!.velocity).toEqual([1, 0, 0]);
        expect(flow.emitters[0]!.transform.position).toEqual([1, 0, 0]);
    });

    it("resolves mutually exclusive sphere, surface, polygon, and foam modes", () => {
        expect(
            resolveFluidRenderMode({
                method: "FLIP",
                renderSpheres: false,
                anisotropicSurface: false,
                polygonSurface: true,
                foamEnabled: true,
            })
        ).toEqual({
            particleEnabled: false,
            surfaceMode: "blit",
            polygonEnabled: true,
            foamEnabled: true,
            foamPolygonSurfaceDepth: true,
            diagnosticMode: "polygon",
        });

        expect(
            resolveFluidRenderMode({
                method: "MLS-MPM",
                renderSpheres: true,
                anisotropicSurface: true,
                polygonSurface: true,
                foamEnabled: true,
                surfaceDebugActive: true,
            })
        ).toMatchObject({
            particleEnabled: false,
            surfaceMode: "ellipsoidDebug",
            polygonEnabled: false,
            foamEnabled: false,
            diagnosticMode: "ellipsoids",
        });
    });

    it("fits the highest valid grid resolution", () => {
        expect(fitFluidGridResolution(64.9, 16, (resolution) => resolution <= 48)).toEqual({
            requestedResolution: 64,
            fittedResolution: 48,
        });
        expect(fitFluidGridResolution(32, 16, () => false)).toBeNull();
    });

    it("normalizes imported and interactive FLIP discretization identically", () => {
        expect(normalizeFluidFlipDiscretization(15.6, 80)).toEqual({
            gridResolution: 16,
            markersPerCell: 64,
        });
        expect(normalizeFluidFlipDiscretization(64.6, 3.6)).toEqual({
            gridResolution: 65,
            markersPerCell: 4,
        });
    });

    it("normalizes impulse direction and applies shared force scales", () => {
        expect(
            resolveFluidImpulseForce({
                center: [0, 0, 0],
                radius: 2,
                intensity: 3,
                direction: [0, 0, 0],
                fallbackDirection: [0, 0, -2],
            })
        ).toEqual({
            direction: [0, 0, -1],
            usedFallbackDirection: true,
            directionalMagnitude: 18,
            radialMagnitude: 54,
        });
    });

    it("reports grid device incompatibility from shared allocation policy", () => {
        expect(
            resolveFluidGridCompatibility("PBF", [2049, 4, 4], {
                maxStorageBufferBindingSize: 1_000_000,
                maxBufferSize: 1_000_000,
            })
        ).toMatchObject({
            compatible: false,
            code: "axis-limit",
        });
        expect(
            resolveFluidGridCompatibility("FLIP", [32, 32, 32], {
                maxStorageBufferBindingSize: 1_024,
                maxBufferSize: 1_024,
            })
        ).toMatchObject({
            compatible: false,
            code: "storage-binding-limit",
            availableBytes: 1_024,
        });
        const mlsLimits = {
            maxStorageBufferBindingSize: 2_048 * 1024 * 1024,
            maxBufferSize: 2_048 * 1024 * 1024,
        };
        expect(resolveFluidGridCompatibility("MLS-MPM", [667, 334, 667], mlsLimits)).toMatchObject({
            compatible: false,
            code: "storage-binding-limit",
        });
        expect(
            resolveFluidGridCompatibility("MLS-MPM", [667, 334, 667], mlsLimits, {
                enabled: true,
                maxPages: 84_000,
            })
        ).toMatchObject({
            compatible: true,
            requiredBytes: (84_000 + 1) * 4 * 4 * 4 * 16,
        });
    });
});
