import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";

import {
    applyFluidControls,
    bindFluidControls,
    deriveFluidControlsApplicationPlan,
    disposeFluidControlsBinding,
    normalizeFluidControls,
    projectFluidControlsMemory,
    resolveFluidSimulationConfig,
    syncFluidControls,
} from "../../../../packages/babylon-lite/src/index";
import type { FluidControlValues, FluidControlsHandle, FluidSimulation, FluidSimulationCollection, FluidSimulationOptions } from "../../../../packages/babylon-lite/src/index";

function values(overrides: Partial<FluidControlValues> = {}): FluidControlValues {
    return {
        method: "FLIP",
        material: 0,
        schema: { pressureIterations: 40, polygonSurface: 1 },
        simulationDuration: 0,
        alphaDecay: 2,
        color: "#16a3c3",
        independentRendering: false,
        half: false,
        thicknessDownscale: 2,
        absorption: 1,
        size: 1,
        physScale: 1,
        gridPosition: [0, 0, 0],
        gridSize: [8, 8, 8],
        cellSize: 0.1,
        gridResolution: 80,
        markersPerCell: 8,
        showGridBounds: false,
        showGridBoundsSolid: false,
        count: 20_000,
        renderMode: "surface",
        polygonShader: "physical",
        refraction: 0.1,
        specular: 250,
        reflectionExposure: 1,
        reflectionContrast: 1.1,
        reflectivity: 0.02,
        depthBlur: 20,
        depthBlurThreshold: 10,
        thicknessBlur: 10,
        surfaceFilter: "narrowRange",
        narrowDelta: 10,
        narrowMu: 1,
        anisotropic: false,
        anisoSurfScale: 0.5,
        activeBlocks: false,
        pagedGrid: true,
        pagedGridMaxPages: 10_000,
        fusedBlockDiscovery: false,
        debug: "none",
        showContainer: true,
        foam: {
            enabled: false,
            activeParticles: true,
            generateSpray: true,
            generateFoam: true,
            generateBubbles: true,
            kTa: 40,
            kWc: 40,
            kTurb: 0,
            energySpeedMin: 1,
            energySpeedMax: 6,
            curvatureMin: 0.05,
            curvatureMax: 1.5,
            turbulenceMin: 0.1,
            turbulenceMax: 2.5,
            foamLayerDepth: 0,
            sprayDrag: 0,
            kb: 0.8,
            kd: 0.5,
            tMin: 0.3,
            tMax: 2,
            poolScale: 3,
            size: 1,
            blurRadius: 4,
            lightIntensity: 0.9,
            ambient: 0.5,
            aoStrength: 0.5,
            normalStrength: 6,
            debugTexture: "off",
            softness: 0.25,
            density: 1.6,
            subsurfaceStrength: 0.4,
            subsurfaceColor: "#b8d1f2",
        },
        ...overrides,
    };
}

const limits = {
    maxStorageBufferBindingSize: 256 * 1024 * 1024,
    maxBufferSize: 512 * 1024 * 1024,
    maxTextureDimension2D: 8192,
};

function simulationOptions(foam: boolean): FluidSimulationOptions {
    return {
        method: "FLIP",
        particleCount: 20_000,
        bounds: { min: [-4, -4, -4], max: [4, 4, 4] },
        physicsScale: 1,
        compatibilityProfile: "fluid",
        gridResolution: 80,
        markersPerCell: 8,
        physics: { polygonSurface: 1 },
        pagedGrid: true,
        pagedGridMaxPages: 4_000,
        foam: foam
            ? {
                  activeParticles: true,
                  generateSpray: true,
                  generateFoam: true,
                  generateBubbles: true,
                  kTa: 40,
                  kWc: 40,
                  kb: 0.8,
                  kd: 0.5,
                  tMin: 0.3,
                  tMax: 2,
                  poolScale: 3,
              }
            : null,
    };
}

function fakeSimulation(options: FluidSimulationOptions) {
    const backend = {
        count: options.particleCount,
        activeCount: options.particleCount,
        renderCount: options.particleCount,
        particleRadius: 0.1,
        gpuBytes: 1024,
        setParam: vi.fn(),
        setFoam: vi.fn(),
    };
    const simulation: FluidSimulation = {
        get method() {
            return this._method;
        },
        get options() {
            return this._options;
        },
        get count() {
            return backend.count;
        },
        get activeCount() {
            return backend.activeCount;
        },
        get renderCount() {
            return backend.renderCount;
        },
        get particleRadius() {
            return backend.particleRadius;
        },
        surfaceSizeScale: 1,
        surfaceThicknessScale: 1,
        surfaceRejectSparseMarkers: false,
        initialEmitterParticleCounts: new Map(),
        get gpuBytes() {
            return backend.gpuBytes;
        },
        pressureDiagnostics: undefined,
        polygonTriangleCount: undefined,
        get resolvedConfig() {
            return this._resolvedConfig;
        },
        sceneSdf: null,
        forceField: null,
        profiler: null,
        renderLayers: [],
        disposed: false,
        _engine: {},
        _sim: backend,
        _method: options.method,
        _options: options,
        _resolvedConfig: resolveFluidSimulationConfig(options.compatibilityProfile ?? "fluid", {
            method: options.method,
            physicsScale: options.physicsScale,
            bounds: options.bounds,
            gridResolution: options.gridResolution,
            markersPerCell: options.markersPerCell,
            physics: options.physics,
            semantics: options.semantics,
        }),
        _renderLayers: [],
    };
    return { simulation, backend };
}

function controlsFor(snapshot: FluidControlValues, setParticleUsage: FluidControlsHandle["setParticleUsage"] = () => undefined): FluidControlsHandle {
    return {
        getValues: () => snapshot,
        setParticleUsage,
        setFoamParticleCounts: () => undefined,
        setPressureDiagnostics: () => undefined,
        setPolygonTriangleCount: () => undefined,
        setPagedGrid: () => undefined,
        setPagedGridMaxPages: () => undefined,
        setPagedGridStatus: () => undefined,
    } as unknown as FluidControlsHandle;
}

describe("shared fluid controls binding", () => {
    it("normalizes paging, FLIP discretization and MLS active blocks in one policy", () => {
        const normalized = normalizeFluidControls(
            values({
                method: "MLS-MPM",
                gridResolution: 80.4,
                markersPerCell: 7.6,
                activeBlocks: false,
                pagedGrid: true,
                pagedGridMaxPages: 9_999,
            }),
            { maxParticleCount: 50_000, pageCapacityLimit: () => 1_250 }
        );

        expect(normalized.gridResolution).toBe(80);
        expect(normalized.markersPerCell).toBe(8);
        expect(normalized.pagedGridMaxPages).toBe(1_250);
        expect(normalized.activeBlocks).toBe(true);
        expect(normalized.foam.activeParticles).toBe(true);
    });

    it("intersects backend controls with host capabilities", () => {
        const normalized = normalizeFluidControls(values(), {
            capabilities: { pagedGrid: false, polygonSurface: false, pressureDiagnostics: false },
        });

        expect(normalized.pagedGrid).toBe(false);
        expect(normalized.schema.polygonSurface).toBe(0);
        expect(normalized.schema.pressureDiagnostics).toBe(0);
        expect(normalized.schema.pressureTolerance).toBe(0);
    });

    it("derives and clamps page capacity from shared target options and device limits", () => {
        const initial = values({ pagedGridMaxPages: 1_000_000_000 });
        const { simulation } = fakeSimulation({
            ...simulationOptions(false),
            pagedGridMaxPages: initial.pagedGridMaxPages,
        });
        let hostPageCapacity = initial.pagedGridMaxPages;
        const binding = bindFluidControls({
            controls: controlsFor(initial),
            target: simulation,
            deviceLimits: limits,
            resolveTarget: () => ({ simulation, options: simulation.options }),
            applyHostState: (snapshot) => {
                hostPageCapacity = snapshot.pagedGridMaxPages;
            },
        });

        expect(binding.snapshot.pagedGridMaxPages).toBeLessThan(initial.pagedGridMaxPages);
        expect(hostPageCapacity).toBe(binding.snapshot.pagedGridMaxPages);
    });

    it("derives the same solver and render invalidation from normalized keys", () => {
        const previous = values();
        const next = values({ method: "MLS-MPM", renderMode: "spheres", pagedGridMaxPages: 2_000 });
        const plan = deriveFluidControlsApplicationPlan(previous, next, ["renderMode", "pagedGridMaxPages"]);

        expect(plan).toMatchObject({
            reconfigure: true,
            renderProfile: false,
            renderMode: true,
            foamRender: true,
        });
        expect(plan.changedKeys).toContain("method");
    });

    it.each(["FLIP", "MLS-MPM"] as const)("defers %s page-capacity allocation until restart", (method) => {
        const previous = values({ method, pagedGrid: true, pagedGridMaxPages: 2_000 });
        const next = values({ method, pagedGrid: true, pagedGridMaxPages: 4_000 });

        expect(deriveFluidControlsApplicationPlan(previous, next, ["pagedGridMaxPages"])).toMatchObject({
            reconfigure: false,
            restartRequired: true,
        });
    });

    it("defers switching FLIP paged storage until restart", () => {
        const previous = values({ method: "FLIP", pagedGrid: false });
        const next = values({ method: "FLIP", pagedGrid: true });

        expect(deriveFluidControlsApplicationPlan(previous, next, ["pagedGrid"])).toMatchObject({
            reconfigure: false,
            restartRequired: true,
        });
    });

    it("keeps live physics and foam edits out of structural reconfiguration", () => {
        const previous = values();
        const physics = values({ schema: { ...previous.schema, pressureIterations: 60 } });
        const foam = values({ foam: { ...previous.foam, blurRadius: 8 } });

        expect(deriveFluidControlsApplicationPlan(previous, physics, ["schema"])).toMatchObject({
            reconfigure: false,
            renderProfile: false,
            renderMode: false,
            foamRender: false,
        });
        expect(deriveFluidControlsApplicationPlan(previous, foam, ["foam"]).reconfigure).toBe(false);
    });

    it("updates render mode only for the polygon-surface schema field", () => {
        const previous = values();
        const polygon = values({ schema: { ...previous.schema, polygonSurface: 1 } });

        expect(deriveFluidControlsApplicationPlan(previous, polygon, ["schema"])).toMatchObject({
            reconfigure: false,
            renderMode: true,
            foamRender: true,
        });
    });

    it("keeps FLIP discretization and capacity changes pending until reset", () => {
        const previous = values();
        const next = values({ count: 40_000, gridPosition: [1, 2, 3], gridSize: [10, 11, 12], gridResolution: 120, markersPerCell: 12 });
        const plan = deriveFluidControlsApplicationPlan(previous, next, ["count", "gridPosition", "gridSize", "gridResolution", "markersPerCell"]);

        expect(plan.reconfigure).toBe(false);
        expect(plan.restartRequired).toBe(true);
    });

    it("keeps pending FLIP restart projections visible until the simulation catches up", () => {
        const initial = values();
        const { simulation, backend } = fakeSimulation(simulationOptions(false));
        const usage: number[][] = [];
        const binding = bindFluidControls({
            controls: controlsFor(initial, (...args) => usage.push(args.map((value) => value ?? -1))),
            target: simulation,
            deviceLimits: limits,
            resolveTarget: (_simulation, snapshot) => ({
                simulation,
                options: {
                    ...simulation.options,
                    particleCount: snapshot.count,
                    bounds: {
                        min: [
                            snapshot.gridPosition[0] - snapshot.gridSize[0] * 0.5,
                            snapshot.gridPosition[1] - snapshot.gridSize[1] * 0.5,
                            snapshot.gridPosition[2] - snapshot.gridSize[2] * 0.5,
                        ],
                        max: [
                            snapshot.gridPosition[0] + snapshot.gridSize[0] * 0.5,
                            snapshot.gridPosition[1] + snapshot.gridSize[1] * 0.5,
                            snapshot.gridPosition[2] + snapshot.gridSize[2] * 0.5,
                        ],
                    },
                    gridResolution: snapshot.gridResolution,
                    markersPerCell: snapshot.markersPerCell,
                },
            }),
        });

        const pending = applyFluidControls(binding, values({ count: 40_000 }), ["count"]);
        expect(pending.restartRequired).toBe(true);
        expect(simulation._sim).toBe(backend);
        expect(simulation.count).toBe(20_000);
        expect(usage.at(-1)?.slice(3)).toEqual([40_000, 40_000, binding.memory.steadyBytes]);

        const stillPending = applyFluidControls(binding, values({ count: 40_000, color: "#ffffff" }), ["color"]);
        expect(stillPending.restartRequired).toBe(true);

        simulation._options = { ...simulation.options, particleCount: 40_000 };
        backend.count = 40_000;
        syncFluidControls(binding);
        expect(binding.plan.restartRequired).toBe(false);
        expect(usage.at(-1)?.slice(3)).toEqual([-1, -1, -1]);
    });

    it("reuses the resolved projection while the controls snapshot and target set are unchanged", () => {
        const initial = values();
        const { simulation } = fakeSimulation(simulationOptions(false));
        const setPagedGridMaxPages = vi.fn();
        const resolveTarget = vi.fn((_simulation: FluidSimulation, snapshot: Readonly<FluidControlValues>) => ({
            simulation,
            options: {
                ...simulation.options,
                particleCount: snapshot.count,
                gridResolution: snapshot.gridResolution,
                markersPerCell: snapshot.markersPerCell,
            },
        }));
        const binding = bindFluidControls({
            controls: { ...controlsFor(initial), setPagedGridMaxPages },
            target: simulation,
            deviceLimits: limits,
            resolveTarget,
        });
        const resolveCount = resolveTarget.mock.calls.length;
        const pageCapacitySyncCount = setPagedGridMaxPages.mock.calls.length;
        const memory = binding.memory;

        syncFluidControls(binding);
        syncFluidControls(binding);
        syncFluidControls(binding, { pageDiagnostics: { method: "FLIP", requiredPages: 10, capacity: initial.pagedGridMaxPages } });

        expect(resolveTarget).toHaveBeenCalledTimes(resolveCount);
        expect(setPagedGridMaxPages).toHaveBeenCalledTimes(pageCapacitySyncCount);
        expect(binding.memory).toBe(memory);
    });

    it("applies authored PBF parameters live without rebuilding the simulation", () => {
        const initial = values({
            method: "PBF",
            physScale: 2,
            pagedGrid: false,
            schema: { restDensity: 341, relaxation: 50 },
        });
        const { simulation, backend } = fakeSimulation({
            method: "PBF",
            particleCount: initial.count,
            bounds: { min: [-4, -4, -4], max: [4, 4, 4] },
            physicsScale: 2,
            compatibilityProfile: "fluid",
            physics: { ...initial.schema },
        });
        const binding = bindFluidControls({
            controls: controlsFor(initial),
            target: simulation,
            deviceLimits: limits,
            resolveTarget: () => ({ simulation, options: simulation.options }),
        });
        const next = values({ ...initial, schema: { ...initial.schema, restDensity: 400 } });

        const plan = applyFluidControls(binding, next, ["schema"]);

        expect(plan.reconfigure).toBe(false);
        expect(simulation._sim).toBe(backend);
        expect(backend.setParam).toHaveBeenCalledOnce();
        expect(backend.setParam).toHaveBeenCalledWith("restDensity", 50);
        expect(simulation.options.physics?.restDensity).toBe(400);
    });

    it("rolls back earlier simulations when a collection live update fails", () => {
        const initial = values({ method: "PBF", pagedGrid: false, foam: { ...values().foam, enabled: true } });
        const foam = {
            activeParticles: true,
            generateSpray: true,
            generateFoam: true,
            generateBubbles: true,
            kTa: 40,
            kWc: 40,
            kb: 0.8,
            kd: 0.5,
            tMin: 0.3,
            tMax: 2,
            poolScale: 3,
        };
        const options: FluidSimulationOptions = {
            method: "PBF",
            particleCount: initial.count,
            bounds: { min: [-4, -4, -4], max: [4, 4, 4] },
            physicsScale: 1,
            compatibilityProfile: "fluid",
            physics: { ...initial.schema },
            foam,
        };
        const first = fakeSimulation(options);
        const second = fakeSimulation(options);
        second.backend.setFoam.mockImplementationOnce(() => {
            throw new Error("injected foam failure");
        });
        const simulations = [first.simulation, second.simulation];
        const collection = {
            simulations,
            sources: [],
            renderLayers: [],
            disposed: false,
            _disposed: false,
            _simulations: simulations,
            _sources: [],
            _renderLayers: [],
            _engine: {},
        } as FluidSimulationCollection;
        const binding = bindFluidControls({
            controls: controlsFor(initial),
            target: collection,
            deviceLimits: limits,
            resolveTarget: (simulation) => ({ simulation, options: simulation.options }),
        });
        const next = values({ ...initial, foam: { ...initial.foam, kTa: 50 } });

        expect(() => applyFluidControls(binding, next, ["foam"])).toThrow("injected foam failure");
        expect(first.backend.setFoam).toHaveBeenCalledTimes(2);
        expect(second.backend.setFoam).toHaveBeenCalledTimes(2);
        expect(first.simulation.options.foam?.kTa).toBe(40);
        expect(second.simulation.options.foam?.kTa).toBe(40);
        expect(binding.snapshot.foam.kTa).toBe(40);
    });

    it("retains committed live state when post-application control synchronization throws", () => {
        const initial = values({ method: "PBF", pagedGrid: false, schema: { gravity: 9.8 } });
        const { simulation, backend } = fakeSimulation({
            method: "PBF",
            particleCount: initial.count,
            bounds: { min: [-4, -4, -4], max: [4, 4, 4] },
            physicsScale: 1,
            compatibilityProfile: "fluid",
            physics: { ...initial.schema },
        });
        let failSync = false;
        let hostGravity = initial.schema.gravity!;
        const binding = bindFluidControls({
            controls: controlsFor(initial, () => {
                if (failSync) {
                    throw new Error("injected sync failure");
                }
            }),
            target: simulation,
            deviceLimits: limits,
            resolveTarget: () => ({ simulation, options: simulation.options }),
            captureHostState: () => hostGravity,
            applyHostState: (snapshot) => {
                hostGravity = snapshot.schema.gravity!;
            },
            restoreHostState: (gravity) => {
                hostGravity = gravity;
            },
        });
        const next = values({ ...initial, schema: { gravity: 12 } });
        failSync = true;

        expect(() => applyFluidControls(binding, next, ["schema"])).toThrow("injected sync failure");
        expect(backend.setParam).toHaveBeenCalledWith("gravity", 12);
        expect(hostGravity).toBe(12);
        expect(binding.snapshot.schema.gravity).toBe(12);
        expect(simulation.options.physics?.gravity).toBe(12);
    });

    it("includes foam pool allocations in restart memory projection", () => {
        const withoutFoam = projectFluidControlsMemory([{ options: simulationOptions(false) }], limits);
        const withFoam = projectFluidControlsMemory([{ options: simulationOptions(true) }], limits);

        expect(withFoam.foamCapacity).toBeGreaterThan(0);
        expect(withFoam.steadyBytes).toBeGreaterThan(withoutFoam.steadyBytes);
        expect(withFoam.plans[0]?.resources.some((resource) => resource.name === "flip-foam-pool")).toBe(true);
    });

    it("uses the facade explicit-grid default for public memory projections", () => {
        const base: FluidSimulationOptions = {
            method: "PBF",
            particleCount: 20_000,
            bounds: { min: [-4, -4, -4], max: [4, 4, 4] },
            physicsScale: 1,
            compatibilityProfile: "fluid",
        };

        const implicit = projectFluidControlsMemory([{ options: base }], limits);
        const explicit = projectFluidControlsMemory([{ options: { ...base, explicitGrid: true } }], limits);

        expect(implicit.steadyBytes).toBe(explicit.steadyBytes);
        expect(implicit.plans[0]?.dimensions.gridDim).toEqual(explicit.plans[0]?.dimensions.gridDim);
    });

    it("publishes identical FLIP and MLS-MPM page diagnostics", () => {
        const statuses: Array<{ text: string; error: boolean }> = [];
        const snapshot = values();
        const controls = {
            getValues: () => snapshot,
            setParticleUsage: () => undefined,
            setFoamParticleCounts: () => undefined,
            setPressureDiagnostics: () => undefined,
            setPolygonTriangleCount: () => undefined,
            setPagedGrid: () => undefined,
            setPagedGridMaxPages: () => undefined,
            setPagedGridStatus: (text: string, error = false) => statuses.push({ text, error }),
        } as unknown as FluidControlsHandle;
        const simulations: FluidSimulation[] = [];
        const collection = {
            simulations,
            sources: [],
            renderLayers: [],
            disposed: false,
            _disposed: false,
            _simulations: simulations,
            _sources: [],
            _renderLayers: [],
            _engine: {},
        } as FluidSimulationCollection;
        const binding = bindFluidControls({
            controls,
            target: collection,
            deviceLimits: limits,
            resolveTarget: () => {
                throw new Error("no live simulations");
            },
        });

        syncFluidControls(binding, { pageDiagnostics: { method: "FLIP", requiredPages: 600, capacity: 1_000 } });
        syncFluidControls(binding, { pageDiagnostics: { method: "MLS-MPM", requiredPages: 1_001, capacity: 1_000, overflow: true } });

        expect(statuses).toContainEqual({ text: "600\u00a0/\u00a01,000\u00a0pages", error: false });
        expect(statuses).toContainEqual({ text: "Page capacity exceeded: 1,001 required, 1,000 allocated.", error: true });
    });

    it("restores host and binding state when target preparation cannot begin", () => {
        const initial = values();
        let hostMethod = initial.method;
        const controls = {
            getValues: () => initial,
            setParticleUsage: () => undefined,
            setFoamParticleCounts: () => undefined,
            setPressureDiagnostics: () => undefined,
            setPolygonTriangleCount: () => undefined,
            setPagedGrid: () => undefined,
            setPagedGridMaxPages: () => undefined,
            setPagedGridStatus: () => undefined,
        } as unknown as FluidControlsHandle;
        const simulations: FluidSimulation[] = [];
        const collection = {
            simulations,
            sources: [],
            renderLayers: [],
            disposed: false,
            _disposed: false,
            _simulations: simulations,
            _sources: [],
            _renderLayers: [],
            _engine: {},
        } as FluidSimulationCollection;
        const binding = bindFluidControls({
            controls,
            target: collection,
            deviceLimits: limits,
            captureHostState: () => ({ method: hostMethod }),
            applyHostState: (next) => {
                hostMethod = next.method;
            },
            restoreHostState: (state) => {
                hostMethod = state.method;
            },
            resolveTarget: () => {
                throw new Error("injected preparation failure");
            },
        });
        simulations.push({ disposed: false } as FluidSimulation);

        expect(() => applyFluidControls(binding, values({ method: "MLS-MPM" }), ["method"])).toThrow("injected preparation failure");
        expect(hostMethod).toBe("FLIP");
        expect(binding.snapshot.method).toBe("FLIP");

        disposeFluidControlsBinding(binding);
        expect(binding.disposed).toBe(true);
    });

    it("is the common apply, page-diagnostic and memory authority for both lab hosts", () => {
        const fluid = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid.ts"), "utf8");
        const aquanova = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/aquanova-fluid-sim.ts"), "utf8");

        for (const source of [fluid, aquanova]) {
            expect(source).toContain("bindFluidControls({");
            expect(source).toContain("applyFluidControls(controlsBinding,");
            expect(source).toContain("applyFluidGridSettings");
            expect(source).toContain("syncFluidControls(controlsBinding");
            expect(source).toContain("disposeFluidControlsBinding(controlsBinding)");
            expect(source).not.toContain("pageCapacityLimit:");
        }
        expect(fluid).toContain('method: "FLIP", requiredPages, capacity');
        expect(fluid).toContain('method: "MLS-MPM", requiredPages, capacity');
        expect(aquanova).toContain("method: currentMethod, requiredPages, capacity");
        expect(aquanova).toContain("method: backend, requiredPages, capacity");
        expect(fluid).toContain("controlsBinding?.memory.steadyBytes");
        expect(aquanova).toContain("projectFluidControlsMemory([{ options, activeCount: active }]");
        expect(aquanova).toContain("aquanovaBindingValues(values, changedKeys)");
        expect(aquanova).toContain('changedKeys.includes("count")');
    });
});
