import { beforeEach, describe, expect, it, vi } from "vitest";
import type { FluidSimulationBackend } from "../../../../packages/babylon-lite/src/fluid/core/fluid-facade.js";

const mocks = vi.hoisted(() => {
    const state = {
        backends: [] as ReturnType<typeof backend>[],
        configureFailure: false,
        prepareFailure: false,
        retirements: [] as (() => void)[],
        tasks: [] as ReturnType<typeof renderTask>[],
    };

    function backend(method = "PBF") {
        const value = {
            method,
            count: 16,
            activeCount: 12,
            renderCount: 12,
            particleRadius: 0.1,
            gpuBytes: 1024,
            pressureDiagnostics: {
                maxResidual: 1,
                maxRhs: 2,
                relativeResidual: 0.5,
                maxDivergence: 0.25,
                fluidCellCount: 8,
                pressureIterations: 4,
            },
            diffuse: undefined as { capacity: number; counts?: { total: number; spray: number; foam: number; bubble: number; capacity: number } } | undefined,
            polygonSurface: {
                triangleCount: 42,
                triangleCapacity: 100,
                reconstructionMultiplier: 1,
                gridOrigin: [0, 0, 0] as const,
                gridDimensions: [4, 4, 4] as const,
                gridSpacing: 0.2,
            },
            prepare: vi.fn(() => {
                if (state.prepareFailure) {
                    throw new Error("prepare failed");
                }
            }),
            step: vi.fn(),
            reset: vi.fn(),
            setParam: vi.fn(),
            setSceneSdf: vi.fn(),
            setEmitters: vi.fn(),
            setFlow: vi.fn(() => {
                if (state.configureFailure) {
                    throw new Error("configure failed");
                }
            }),
            updateFlowEmitter: vi.fn(),
            setSpawn: vi.fn(),
            setForceField: vi.fn(),
            setFoam: vi.fn(),
            setProfiler: vi.fn(),
            dispose: vi.fn(),
        };
        state.backends.push(value);
        return value;
    }

    function renderTask() {
        const value = {
            name: "fluid-test-task",
            engine: {},
            scene: {},
            _passes: [],
            executionEnabled: true,
            setSim: vi.fn(),
            setEnabled: vi.fn(),
            setOpacity: vi.fn(),
            setSizeScale: vi.fn(),
            setTint: vi.fn(),
            setVelocityBrighten: vi.fn(),
            setProfiler: vi.fn(),
            setParticleAlpha: vi.fn(),
            setParticleColor: vi.fn(),
            setParticleColorMode: vi.fn(),
            setUseParticleColor: vi.fn(),
            setMode: vi.fn(),
            setDebug: vi.fn(),
            setDirLight: vi.fn(),
            setEnvMap: vi.fn(),
            setEnvRotationY: vi.fn(),
            setSurfaceFiltering: vi.fn(),
            setPolygonSurfaceDepth: vi.fn(),
            setDebugByKind: vi.fn(),
            setThresholds: vi.fn(),
            setSubsurfaceStrength: vi.fn(),
            setSubsurfaceColor: vi.fn(),
            setBlurRadius: vi.fn(),
            setDebugTexture: vi.fn(),
            setLightIntensity: vi.fn(),
            setAmbient: vi.fn(),
            setAOStrength: vi.fn(),
            setNormalStrength: vi.fn(),
            setFluidColor: vi.fn(),
            setAbsorption: vi.fn(),
            setRefractionStrength: vi.fn(),
            setSpecularPower: vi.fn(),
            setEnvReflection: vi.fn(),
            setFresnelF0: vi.fn(),
            setShadingMode: vi.fn(),
            setSims: vi.fn(),
            setWireframe: vi.fn(),
            surfaceDepthView: vi.fn(() => null),
            record: vi.fn(),
            execute: vi.fn(() => 0),
            dispose: vi.fn(),
        };
        state.tasks.push(value);
        return value;
    }

    return {
        state,
        backend,
        renderTask,
        transfer: vi.fn((_encoder?: unknown, _source?: unknown, _target?: unknown) => true),
        addTask: vi.fn(),
        profilerImpl: {
            pass: vi.fn(),
            beginFrame: vi.fn(),
            frameStart: vi.fn(),
            frameStop: vi.fn(),
            resolveInto: vi.fn(),
            results: vi.fn(() => ({ stages: { Simulation: 1 }, total: 1, frameTotal: 2 })),
            dispose: vi.fn(),
        },
        lastAggregateSources: [] as Array<{ sim: unknown }>,
        aggregateUpdate: null as (() => void) | null,
    };
});

vi.mock("../../../../packages/babylon-lite/src/fluid/solvers/pbf-sim.js", () => ({
    createPbfSim: () => mocks.backend("PBF"),
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/solvers/flip-sim.js", () => ({
    createFlipSim: () => mocks.backend("FLIP"),
    transferFlipSimState: mocks.transfer,
    encodeFlipSimStateTransfer: (...args: unknown[]) => {
        mocks.transfer(args[0], args[1], args[2]);
        return () => {};
    },
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.js", () => ({
    createMlsMpmSim: () => mocks.backend("MLS-MPM"),
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/solvers/pbmpm-sim.js", () => ({
    createPbMpmSim: () => mocks.backend("PB-MPM"),
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/core/simulation-config.js", () => ({
    resolveFluidSimulationConfig: (
        _profile: string,
        input: {
            method: string;
            physicsScale: number;
            physics?: Record<string, number>;
            semantics?: { pbfPhysics: "scale-adjusted" | "literal" };
        }
    ) => {
        const physics = { ...(input.physics ?? {}) };
        if (input.method === "PBF" && (input.semantics?.pbfPhysics ?? "scale-adjusted") === "scale-adjusted") {
            if (physics.restDensity !== undefined) {
                physics.restDensity /= input.physicsScale ** 3;
            }
            if (physics.relaxation !== undefined) {
                physics.relaxation /= input.physicsScale ** 2;
            }
        }
        return {
            profile: "fluid",
            method: input.method,
            semantics: { version: 1, profile: "normalized-v1", pbfPhysics: "scale-adjusted" },
            physics,
            particleRadius: 0.1,
            cellSize: 0.2,
            particleVolume: 0.008,
            gridDim: [10, 10, 10],
        };
    },
}));
vi.mock("../../../../packages/babylon-lite/src/engine/gpu-resource-retirement.js", () => ({
    retireGpuResources: (_engine: unknown, dispose: () => void) => mocks.state.retirements.push(dispose),
}));
vi.mock("../../../../packages/babylon-lite/src/frame-graph/frame-graph-actions.js", () => ({
    addTask: mocks.addTask,
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/rendering/particle-render.js", () => ({
    createParticleRenderTask: mocks.renderTask,
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/rendering/fluid-surface-render.js", () => ({
    createFluidSurfaceTask: mocks.renderTask,
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/rendering/polygon-surface-render.js", () => ({
    createFluidPolygonSurfaceTask: mocks.renderTask,
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/rendering/foam-render.js", () => ({
    createFoamRenderTask: mocks.renderTask,
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/rendering/fluid-render-profile.js", () => ({
    applyFluidRenderProfile: vi.fn(),
    fluidRenderHexColor: vi.fn(() => [0.1, 0.2, 0.3]),
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/rendering/fluid-render-compositor.js", () => ({
    createFluidRenderCompositor: () => ({
        ...mocks.renderTask(),
        setLayers: vi.fn(),
    }),
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/core/gpu-profiler.js", () => ({
    createFluidProfiler: () => mocks.profilerImpl,
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/core/fluid-runtime-bindings.js", () => ({
    createSceneSdfRuntimeBinding: () => ({
        spec: { struct: "s", sdf: "f", buffer: {} },
        updateParams: vi.fn(),
        updateSdfGrid: vi.fn(),
        dispose: vi.fn(),
    }),
    createForceFieldRuntimeBinding: () => ({
        spec: { struct: "s", wgsl: "f", buffer: {} },
        updateParams: vi.fn(),
        dispose: vi.fn(),
    }),
}));
vi.mock("../../../../packages/babylon-lite/src/fluid/core/fluid-particle-runtime.js", () => ({
    createFluidParticleChannelRuntime: (_engine: unknown, options: { capacity: number; components: 1 | 4 }) => ({
        buffer: {},
        capacity: options.capacity,
        components: options.components,
        write: vi.fn(),
        fill: vi.fn(),
        read: vi.fn(async () => new Float32Array(options.components)),
        dispose: vi.fn(),
    }),
    createFluidAggregateRuntime: (_engine: unknown, capacity: number, readSources: () => Array<{ sim: unknown }>) => {
        const aggregateTask = mocks.renderTask();
        aggregateTask.execute = vi.fn(() => {
            mocks.lastAggregateSources = readSources();
            mocks.aggregateUpdate?.();
            return 0;
        });
        return {
            sim: {
                count: 0,
                particleRadius: 0.1,
                positionBuffer: {},
                velocityBuffer: {},
                debugBuffer: {},
                debugNorm: 1,
                gpuBytes: 0,
            },
            task: aggregateTask,
            stream: { positionBuffer: {}, alphaBuffer: {}, count: 0, capacity },
            colorBuffer: {},
            usesParticleColor: false,
            overflowed: false,
            setOnUpdate: (callback: () => void) => {
                mocks.aggregateUpdate = callback;
            },
            dispose: vi.fn(),
        };
    },
    createFluidSpatialQueryRuntime: () => ({
        results: [{ key: "receiver", count: 3 }],
        record: vi.fn(),
        dispose: vi.fn(),
    }),
    createFluidWheelTorqueQueryRuntime: () => ({
        torque: 12.5,
        status: "ready",
        poll: vi.fn(),
        record: vi.fn(),
        dispose: vi.fn(),
    }),
    fluidSimParticleStream: (sim: { count: number }) => ({ positionBuffer: {}, count: sim.count, capacity: sim.count }),
    writeFluidSimPositions: vi.fn(),
    readFluidSimPositions: vi.fn(async () => new Float32Array(4)),
}));

const facade = await import("../../../../packages/babylon-lite/src/fluid/core/fluid-facade.js");
const flipModule = await import("../../../../packages/babylon-lite/src/fluid/solvers/flip-sim.js");

function asynchronousBackend(submitStep: (dt: number, beforeSubstep?: (dt: number) => void) => Promise<void>): FluidSimulationBackend {
    return {
        id: "test-async",
        name: "Async test solver",
        method: "FLIP",
        steppingMode: "async",
        renderModes: ["spheres", "surface"],
        supportsFoam: false,
        supportsForces: false,
        supportsContinuousFlow: false,
        physicsParameters: ["gravity"],
        description: "Test-only asynchronous backend.",
        _create: (engine) => ({ ...flipModule.createFlipSim(engine), submitStep }),
    };
}

const engine = {
    _device: {
        createCommandEncoder: vi.fn(() => ({
            finish: vi.fn(() => ({})),
        })),
        queue: {
            submit: vi.fn(),
        },
    },
    _currentEncoder: undefined as object | undefined,
};
const scene = {};
const camera = {};
const target = {};

function options(method: "PBF" | "FLIP" = "PBF", particleCount = 16) {
    return {
        method,
        particleCount,
        bounds: { min: [-1, 0, -1] as const, max: [1, 2, 1] as const },
        physicsScale: 1,
    };
}

beforeEach(() => {
    mocks.state.backends.length = 0;
    mocks.state.tasks.length = 0;
    mocks.state.retirements.length = 0;
    mocks.state.configureFailure = false;
    mocks.state.prepareFailure = false;
    mocks.lastAggregateSources = [];
    mocks.aggregateUpdate = null;
    mocks.transfer.mockReset().mockReturnValue(true);
    mocks.addTask.mockReset();
    engine._device.createCommandEncoder.mockClear();
    engine._device.queue.submit.mockReset();
    engine._currentEncoder = undefined;
});

describe("fluid facade transactional lifecycle", () => {
    it("awaits an explicit asynchronous backend and forwards per-substep scene updates", async () => {
        let finish!: () => void;
        const pending = new Promise<void>((resolve) => {
            finish = resolve;
        });
        const submit = vi.fn(async (dt: number, beforeSubstep?: (dt: number) => void) => {
            beforeSubstep?.(dt / 2);
            await pending;
            beforeSubstep?.(dt / 2);
        });
        const simulation = facade.createFluidSimulation(engine as never, { ...options("FLIP"), backend: asynchronousBackend(submit) });
        const beforeSubstep = vi.fn();
        expect(simulation.steppingMode).toBe("async");
        expect(() => facade.stepFluidSimulation(simulation, 0.02)).toThrow("awaited submitFluidSimulationStep");
        const operation = facade.submitFluidSimulationStep(simulation, 0.02, { beforeSubstep });
        expect(beforeSubstep).toHaveBeenCalledExactlyOnceWith(0.01);
        expect(engine._device.createCommandEncoder).not.toHaveBeenCalled();
        finish();
        await operation;
        expect(beforeSubstep).toHaveBeenCalledTimes(2);
        expect(mocks.state.backends[0]!.step).not.toHaveBeenCalled();
    });

    it("does not route implementation overrides through production FLIP state transfer", () => {
        const backend = asynchronousBackend(vi.fn(async () => {}));
        const simulation = facade.createFluidSimulation(engine as never, { ...options("FLIP"), backend });
        expect(() => facade.prepareFluidReconfigurationUpdate(simulation, { particleCount: 24 }, true)).toThrow("state-preserving");
        expect(mocks.transfer).not.toHaveBeenCalled();
        const replacement = facade.prepareFluidReconfigurationUpdate(simulation, { backend: undefined });
        facade.commitFluidReconfiguration(replacement);
        expect(simulation.steppingMode).toBe("frame");
        expect(simulation.options.backend).toBeUndefined();
    });

    it("rejects unsupported backend features before allocation", () => {
        const backend = asynchronousBackend(vi.fn(async () => {}));
        expect(() => facade.createFluidSimulation(engine as never, { ...options("PBF"), backend })).toThrow("intrinsic method");
        expect(() => facade.createFluidSimulation(engine as never, { ...options("FLIP"), backend, foam: {} })).toThrow("does not support foam");
        expect(mocks.state.backends).toHaveLength(0);
    });

    it("does not invoke an asynchronous step from inside render-frame recording", async () => {
        const submit = vi.fn(async () => {});
        const simulation = facade.createFluidSimulation(engine as never, { ...options("FLIP"), backend: asynchronousBackend(submit) });
        engine._currentEncoder = {};
        await expect(facade.submitFluidSimulationStep(simulation, 0.02)).rejects.toThrow("render frame");
        expect(submit).not.toHaveBeenCalled();
    });

    it("disposes a constructor candidate when configuration throws", () => {
        mocks.state.configureFailure = true;
        expect(() => facade.createFluidSimulation(engine as never, { ...options(), flow: { emitters: [], sinks: [] } })).toThrow("configure failed");
        expect(mocks.state.backends).toHaveLength(1);
        expect(mocks.state.backends[0]!.dispose).toHaveBeenCalledOnce();
    });

    it("disposes a detached candidate when optional resource preparation throws", () => {
        const simulation = facade.createFluidSimulation(engine as never, options("FLIP"));
        mocks.state.prepareFailure = true;

        expect(() => facade.prepareFluidReconfiguration(simulation, options("FLIP", 32), false)).toThrow("prepare failed");
        expect(mocks.state.backends[1]!.dispose).toHaveBeenCalledOnce();
        expect(simulation._sim).toBe(mocks.state.backends[0]);
    });

    it("preserves the old simulation, options, and layers when candidate configuration fails", () => {
        const simulation = facade.createFluidSimulation(engine as never, options());
        const layer = facade.attachFluidSimulationRenderLayer(simulation, {
            scene: scene as never,
            camera: camera as never,
            mode: "spheres",
            depthTarget: target as never,
            colorTarget: target as never,
        });
        const oldBackend = simulation._sim;
        const oldOptions = simulation.options;

        mocks.state.configureFailure = true;
        expect(() =>
            facade.prepareFluidReconfiguration(simulation, {
                ...options("PBF", 32),
                flow: { emitters: [], sinks: [] },
            })
        ).toThrow("configure failed");

        expect(simulation._sim).toBe(oldBackend);
        expect(simulation.options).toBe(oldOptions);
        expect(simulation.renderLayers).toEqual([layer]);
        expect(mocks.state.tasks[0]!.setSim).not.toHaveBeenCalled();
        expect(mocks.state.backends[1]!.dispose).toHaveBeenCalledOnce();
    });

    it("captures FLIP state at commit and preserves live state when transfer throws", () => {
        const simulation = facade.createFluidSimulation(engine as never, options("FLIP"));
        const oldBackend = simulation._sim;
        const oldOptions = simulation.options;
        const prepared = facade.prepareFluidReconfiguration(simulation, options("FLIP", 32), true);
        expect(mocks.transfer).not.toHaveBeenCalled();
        mocks.transfer.mockImplementationOnce(() => {
            throw new Error("transfer failed");
        });

        expect(() => facade.commitFluidReconfiguration(prepared)).toThrow("transfer failed");
        expect(simulation._sim).toBe(oldBackend);
        expect(simulation.options).toBe(oldOptions);
        expect(prepared.status).toBe("prepared");
        expect(engine._device.queue.submit).not.toHaveBeenCalled();
        facade.cancelFluidReconfiguration(prepared);
    });

    it("transfers FLIP state after source work encoded between prepare and commit", () => {
        const simulation = facade.createFluidSimulation(engine as never, options("FLIP"));
        const source = mocks.state.backends[0]!;
        const prepared = facade.prepareFluidReconfiguration(simulation, options("FLIP", 32), true);

        engine._currentEncoder = {};
        facade.stepFluidSimulation(simulation, 1 / 60);
        engine._currentEncoder = undefined;
        facade.commitFluidReconfiguration(prepared);

        expect(source.step).toHaveBeenCalledOnce();
        expect(mocks.transfer).toHaveBeenCalledOnce();
        expect(source.step.mock.invocationCallOrder[0]).toBeLessThan(mocks.transfer.mock.invocationCallOrder[0]!);
        const encoder = engine._device.createCommandEncoder.mock.results[0]!.value;
        expect(mocks.transfer).toHaveBeenCalledWith(encoder, source, mocks.state.backends[1]);
        expect(encoder.finish).toHaveBeenCalledOnce();
        expect(engine._device.queue.submit).toHaveBeenCalledWith([encoder.finish.mock.results[0]!.value]);
    });

    it("rejects a preserve-state commit during frame recording without changing the live backend", () => {
        const simulation = facade.createFluidSimulation(engine as never, options("FLIP"));
        const oldBackend = simulation._sim;
        const prepared = facade.prepareFluidReconfiguration(simulation, options("FLIP", 32), true);
        engine._currentEncoder = {};

        expect(() => facade.commitFluidReconfiguration(prepared)).toThrow("while a frame is being recorded");
        expect(simulation._sim).toBe(oldBackend);
        expect(prepared.status).toBe("prepared");
        expect(mocks.transfer).not.toHaveBeenCalled();
        expect(engine._device.queue.submit).not.toHaveBeenCalled();

        engine._currentEncoder = undefined;
        facade.cancelFluidReconfiguration(prepared);
    });

    it("rejects a preserve-state commit whenever an engine encoder is active", () => {
        const simulation = facade.createFluidSimulation(engine as never, options("FLIP"));
        const oldBackend = simulation._sim;
        const prepared = facade.prepareFluidReconfiguration(simulation, options("FLIP", 32), true);
        engine._currentEncoder = {};

        expect(() => facade.commitFluidReconfiguration(prepared)).toThrow("while a frame is being recorded");
        expect(simulation._sim).toBe(oldBackend);
        expect(prepared.status).toBe("prepared");
        expect(mocks.transfer).not.toHaveBeenCalled();

        engine._currentEncoder = undefined;
        facade.cancelFluidReconfiguration(prepared);
    });

    it("rejects frame-only simulation work without a live frame encoder", () => {
        const simulation = facade.createFluidSimulation(engine as never, options("FLIP"));

        expect(() => facade.stepFluidSimulation(simulation, 1 / 60)).toThrow("during an active engine frame");
        expect(mocks.state.backends[0]!.step).not.toHaveBeenCalled();
    });

    it("keeps authored PBF values and resolves them once for live updates and rebuilds", () => {
        const simulation = facade.createFluidSimulation(engine as never, {
            ...options("PBF"),
            physicsScale: 2,
            physics: { restDensity: 341, relaxation: 50 },
        });
        const backend = mocks.state.backends[0]!;

        expect(simulation.options.physics).toEqual({ restDensity: 341, relaxation: 50 });
        expect(simulation.resolvedConfig.physics).toMatchObject({ restDensity: 42.625, relaxation: 12.5 });

        facade.setFluidSimulationParameter(simulation, "restDensity", 400);
        expect(backend.setParam).toHaveBeenLastCalledWith("restDensity", 50);
        expect(simulation.options.physics?.restDensity).toBe(400);
        expect(simulation.resolvedConfig.physics.restDensity).toBe(50);

        const prepared = facade.prepareFluidReconfigurationUpdate(simulation, { particleCount: 32 });
        expect(prepared.options.physics?.restDensity).toBe(400);
        expect(prepared.resolvedConfig.physics.restDensity).toBe(50);
        facade.cancelFluidReconfiguration(prepared);
    });

    it("commits only after preparation and retires the old backend instead of destroying it immediately", () => {
        const simulation = facade.createFluidSimulation(engine as never, options("FLIP"));
        const oldBackend = mocks.state.backends[0]!;
        const layer = facade.attachFluidSimulationRenderLayer(simulation, {
            scene: scene as never,
            camera: camera as never,
            mode: "surface",
            depthTarget: target as never,
            backgroundTarget: target as never,
            outputTarget: target as never,
        });
        const prepared = facade.prepareFluidReconfiguration(simulation, options("FLIP", 32), true);

        expect(simulation.options.particleCount).toBe(16);
        facade.commitFluidReconfiguration(prepared);

        expect(prepared.status).toBe("committed");
        expect(simulation.options.particleCount).toBe(32);
        expect(mocks.state.tasks[0]!.setSim).toHaveBeenCalledWith(mocks.state.backends[1]);
        expect(simulation.renderLayers).toEqual([layer]);
        expect(oldBackend.dispose).not.toHaveBeenCalled();
        expect(mocks.state.retirements).toHaveLength(1);
        mocks.state.retirements[0]!();
        expect(oldBackend.dispose).toHaveBeenCalledOnce();
    });

    it("atomically commits a collection and reports aggregate transition bytes", () => {
        const first = facade.createFluidSimulation(engine as never, options("FLIP"));
        const second = facade.createFluidSimulation(engine as never, options("FLIP"));
        const firstOld = first._sim;
        const secondOld = second._sim;
        facade.attachFluidSimulationRenderLayer(first, {
            scene: scene as never,
            camera: camera as never,
            mode: "spheres",
            depthTarget: target as never,
            colorTarget: target as never,
        });
        facade.attachFluidSimulationRenderLayer(second, {
            scene: scene as never,
            camera: camera as never,
            mode: "spheres",
            depthTarget: target as never,
            colorTarget: target as never,
        });
        const prepared = facade.prepareFluidCollectionReconfiguration([
            { simulation: first, updates: { particleCount: 24 }, preserveState: true },
            { simulation: second, updates: { particleCount: 32 }, preserveState: true },
        ]);

        expect(prepared.steadyBytes).toBe(2048);
        expect(prepared.transitionPeakBytes).toBe(4096);
        expect(mocks.transfer).not.toHaveBeenCalled();
        facade.commitFluidCollectionReconfiguration(prepared);

        expect(prepared.status).toBe("committed");
        expect(first.options.particleCount).toBe(24);
        expect(second.options.particleCount).toBe(32);
        expect(first._sim).not.toBe(firstOld);
        expect(second._sim).not.toBe(secondOld);
        expect(mocks.transfer).toHaveBeenCalledTimes(2);
        expect(mocks.state.retirements).toHaveLength(2);
    });

    it("rolls every layer back when a later collection retarget fails", () => {
        const first = facade.createFluidSimulation(engine as never, options());
        const second = facade.createFluidSimulation(engine as never, options());
        const firstOld = first._sim;
        const secondOld = second._sim;
        facade.attachFluidSimulationRenderLayer(first, {
            scene: scene as never,
            camera: camera as never,
            mode: "spheres",
            depthTarget: target as never,
            colorTarget: target as never,
        });
        facade.attachFluidSimulationRenderLayer(second, {
            scene: scene as never,
            camera: camera as never,
            mode: "spheres",
            depthTarget: target as never,
            colorTarget: target as never,
        });
        const prepared = facade.prepareFluidCollectionReconfiguration([
            { simulation: first, updates: { particleCount: 24 } },
            { simulation: second, updates: { particleCount: 32 } },
        ]);
        mocks.state.tasks[1]!.setSim.mockImplementationOnce(() => {
            throw new Error("retarget failed");
        });

        expect(() => facade.commitFluidCollectionReconfiguration(prepared)).toThrow("retarget failed");
        expect(first._sim).toBe(firstOld);
        expect(second._sim).toBe(secondOld);
        expect(first.options.particleCount).toBe(16);
        expect(second.options.particleCount).toBe(16);
        expect(mocks.state.tasks[0]!.setSim).toHaveBeenLastCalledWith(firstOld);
        expect(mocks.state.tasks[1]!.setSim).toHaveBeenLastCalledWith(secondOld);
        expect(mocks.state.retirements).toHaveLength(0);
        facade.cancelFluidCollectionReconfiguration(prepared);
    });

    it("cancels prepared candidates through the retirement queue", () => {
        const simulation = facade.createFluidSimulation(engine as never, options());
        const prepared = facade.prepareFluidReconfiguration(simulation, options("PBF", 32));
        const candidate = mocks.state.backends[1]!;

        facade.disposePreparedFluidReconfiguration(prepared);

        expect(prepared.status).toBe("cancelled");
        expect(candidate.dispose).not.toHaveBeenCalled();
        mocks.state.retirements[0]!();
        expect(candidate.dispose).toHaveBeenCalledOnce();
    });

    it("retains candidate bindings until commit or deferred cancellation teardown", () => {
        const simulation = facade.createFluidSimulation(engine as never, options());
        const sdf = facade.createFluidSceneSdf(engine as never, {
            struct: "struct Params { value: vec4<f32>, };",
            sdf: "fn sceneSdf(p: vec3<f32>, dt: f32) -> f32 { return p.y + dt; }",
            params: new Float32Array(4),
        });
        const prepared = facade.prepareFluidReconfiguration(simulation, { ...options("PBF", 32), sceneSdf: sdf });

        expect(() => facade.disposeFluidSceneSdf(sdf)).toThrow("attached");
        facade.cancelFluidReconfiguration(prepared);
        expect(() => facade.disposeFluidSceneSdf(sdf)).toThrow("attached");

        mocks.state.retirements[0]!();
        expect(() => facade.disposeFluidSceneSdf(sdf)).not.toThrow();
    });

    it("transfers retained candidate bindings to the simulation on commit", () => {
        const simulation = facade.createFluidSimulation(engine as never, options());
        const force = facade.createFluidForceField(engine as never, {
            struct: "struct Params { value: vec4<f32>, };",
            wgsl: "fn externalForce(p: vec3<f32>, v: vec3<f32>, dt: f32) -> vec3<f32> { return p + v * dt; }",
            params: new Float32Array(4),
        });
        const prepared = facade.prepareFluidReconfiguration(simulation, { ...options("PBF", 32), forceField: force });

        facade.commitFluidReconfiguration(prepared);

        expect(() => facade.disposeFluidForceField(force)).toThrow("attached");
        facade.disposeFluidSimulation(simulation);
        expect(() => facade.disposeFluidForceField(force)).not.toThrow();
    });

    it("defers detached render-task teardown", () => {
        const simulation = facade.createFluidSimulation(engine as never, options());
        const layer = facade.attachFluidSimulationRenderLayer(simulation, {
            scene: scene as never,
            camera: camera as never,
            mode: "spheres",
            depthTarget: target as never,
            colorTarget: target as never,
        });
        const task = mocks.state.tasks[0]!;

        facade.detachFluidSimulationRenderLayer(simulation, layer);

        expect(layer.disposed).toBe(true);
        expect(task.dispose).not.toHaveBeenCalled();
        mocks.state.retirements[0]!();
        expect(task.dispose).toHaveBeenCalledOnce();
    });
});

describe("fluid facade shared integrations", () => {
    it("attaches and configures a usable polygon layer", () => {
        const simulation = facade.createFluidSimulation(engine as never, options("FLIP"));
        const layer = facade.attachFluidSimulationRenderLayer(simulation, {
            scene: scene as never,
            camera: camera as never,
            mode: "polygon",
            depthTarget: target as never,
            backgroundTarget: target as never,
            outputTarget: target as never,
            profile: { polygonShader: "ocean", absorption: 2 },
        });
        const task = mocks.state.tasks[0]!;

        expect(layer.mode).toBe("polygon");
        expect(task.setEnabled).toHaveBeenCalledWith(true);
        expect(task.setShadingMode).toHaveBeenCalledWith("ocean");
        expect(task.setAbsorption).toHaveBeenCalledWith(2);
        facade.configureFluidSimulationRenderLayer(layer, { enabled: false, opacity: 0.4 });
        expect(task.setEnabled).toHaveBeenLastCalledWith(false);
        expect(task.setOpacity).toHaveBeenCalledWith(0.4);
    });

    it("expresses whiteboard render switching through layer state", () => {
        const simulation = facade.createFluidSimulation(engine as never, options());
        const surface = facade.attachFluidSimulationRenderLayer(simulation, {
            scene: scene as never,
            camera: camera as never,
            mode: "surface",
            depthTarget: target as never,
            backgroundTarget: target as never,
            outputTarget: target as never,
        });
        const spheres = facade.attachFluidSimulationRenderLayer(simulation, {
            scene: scene as never,
            camera: camera as never,
            mode: "spheres",
            depthTarget: target as never,
            colorTarget: target as never,
        });

        facade.configureFluidSimulationRenderLayer(surface, { surfaceMode: "ellipsoidDebug", environmentRotationY: Math.PI });
        facade.configureFluidSimulationRenderLayer(spheres, { particleVelocityBrighten: 0 });

        expect((surface._task as ReturnType<typeof mocks.renderTask>).setMode).toHaveBeenCalledWith("ellipsoidDebug");
        expect((surface._task as ReturnType<typeof mocks.renderTask>).setEnvRotationY).toHaveBeenCalledWith(Math.PI);
        expect((spheres._task as ReturnType<typeof mocks.renderTask>).setVelocityBrighten).toHaveBeenCalledWith(0);
    });

    it("routes scene SDF, force-field, profiler, and pressure diagnostics through opaque state", () => {
        const sdf = facade.createFluidSceneSdf(engine as never, {
            struct: "struct Params { value: vec4<f32>, };",
            sdf: "fn sceneSdf(p: vec3<f32>, dt: f32) -> f32 { return p.y + dt; }",
            params: new Float32Array(4),
        });
        const force = facade.createFluidForceField(engine as never, {
            struct: "struct Params { value: vec4<f32>, };",
            wgsl: "fn externalForce(p: vec3<f32>, v: vec3<f32>, dt: f32) -> vec3<f32> { return p + v * dt; }",
            params: new Float32Array(4),
        });
        const profiler = facade.createFluidSimulationProfiler(engine as never);
        const simulation = facade.createFluidSimulation(engine as never, {
            ...options("FLIP"),
            sceneSdf: sdf,
            forceField: force,
            profiler,
        });
        const backend = mocks.state.backends[0]!;

        expect(backend.setSceneSdf).toHaveBeenCalledWith(expect.objectContaining({ struct: "s" }));
        expect(backend.setForceField).toHaveBeenCalledWith(expect.objectContaining({ wgsl: "f" }));
        expect(backend.setProfiler).toHaveBeenCalledWith(mocks.profilerImpl);
        expect(facade.readFluidSimulationPressureDiagnostics(simulation)).toEqual(expect.objectContaining({ relativeResidual: 0.5, pressureIterations: 4 }));
        expect(simulation.polygonTriangleCount).toBe(42);

        engine._currentEncoder = {};
        facade.beginFluidSimulationProfilerFrame(profiler);
        facade.endFluidSimulationProfilerFrame(profiler);
        engine._currentEncoder = undefined;
        expect(facade.readFluidSimulationProfiler(profiler)).toEqual({
            stages: { Simulation: 1 },
            total: 1,
            frameTotal: 2,
        });
    });

    it("rejects cross-engine simulation dependencies and profiler attachments", () => {
        const otherEngine = { ...engine, _device: { ...engine._device } };
        const sdf = facade.createFluidSceneSdf(otherEngine as never, {
            struct: "struct Params { value: vec4<f32>, };",
            sdf: "fn sceneSdf(p: vec3<f32>, dt: f32) -> f32 { return p.y + dt; }",
            params: new Float32Array(4),
        });
        const force = facade.createFluidForceField(otherEngine as never, {
            struct: "struct Params { value: vec4<f32>, };",
            wgsl: "fn externalForce(p: vec3<f32>, v: vec3<f32>, dt: f32) -> vec3<f32> { return p + v * dt; }",
            params: new Float32Array(4),
        });
        const profiler = facade.createFluidSimulationProfiler(otherEngine as never);

        expect(() => facade.createFluidSimulation(engine as never, { ...options(), sceneSdf: sdf })).toThrow("different engine");
        const simulation = facade.createFluidSimulation(engine as never, options());
        expect(() => facade.setFluidSimulationSceneSdf(simulation, sdf)).toThrow("different engine");
        expect(() => facade.setFluidSimulationForceField(simulation, force)).toThrow("different engine");
        expect(() => facade.setFluidSimulationProfiler(simulation, profiler)).toThrow("different engine");
        expect(() =>
            facade.attachFluidSimulationRenderLayer(simulation, {
                scene: scene as never,
                camera: camera as never,
                mode: "spheres",
                depthTarget: target as never,
                colorTarget: target as never,
                profiler,
            })
        ).toThrow("different engine");
    });

    it("renders a changing collection through one shared surface pass", () => {
        const first = facade.createFluidSimulation(engine as never, options());
        const second = facade.createFluidSimulation(engine as never, options("PBF", 24));
        const collection = facade.createFluidSimulationCollection(engine as never, [first, second]);
        const layer = facade.attachFluidSimulationCollectionRenderLayer(collection, {
            scene: scene as never,
            camera: camera as never,
            mode: "surface",
            depthTarget: target as never,
            backgroundTarget: target as never,
            outputTarget: target as never,
            particleCapacity: 64,
        });

        expect(mocks.addTask).toHaveBeenCalledTimes(2);
        expect(layer.particleStream?.capacity).toBe(64);
        expect(layer.overflowed).toBe(false);
        mocks.state.tasks[0]!.execute?.();
        expect(mocks.lastAggregateSources.map((source) => source.sim)).toEqual([mocks.state.backends[0], mocks.state.backends[1]]);

        facade.setFluidSimulationCollectionSimulations(collection, [second]);
        mocks.state.tasks[0]!.execute?.();
        expect(mocks.lastAggregateSources.map((source) => source.sim)).toEqual([mocks.state.backends[1]]);
    });

    it("reuses one foam task across every diffuse-enabled simulation", () => {
        const first = facade.createFluidSimulation(engine as never, options("FLIP"));
        const second = facade.createFluidSimulation(engine as never, options("FLIP", 24));
        mocks.state.backends[0]!.diffuse = { capacity: 100 };
        mocks.state.backends[1]!.diffuse = { capacity: 200 };
        const collection = facade.createFluidSimulationCollection(engine as never, [first, second]);
        facade.attachFluidSimulationCollectionRenderLayer(collection, {
            scene: scene as never,
            camera: camera as never,
            mode: "foam",
            depthTarget: target as never,
            colorTarget: target as never,
            foam: { polygonSurfaceDepth: false },
        });
        const task = mocks.state.tasks[0]!;

        task.execute?.();

        expect(task.setSim).toHaveBeenNthCalledWith(1, mocks.state.backends[0]);
        expect(task.setSim).toHaveBeenNthCalledWith(2, mocks.state.backends[1]);
        expect(task.setPolygonSurfaceDepth).toHaveBeenLastCalledWith(false);
        expect(mocks.addTask).toHaveBeenCalledTimes(1);
    });

    it("creates polygon and foam layers before a dynamic collection has simulations", () => {
        const collection = facade.createFluidSimulationCollection(engine as never);
        const polygon = facade.attachFluidSimulationCollectionRenderLayer(collection, {
            scene: scene as never,
            camera: camera as never,
            mode: "polygon",
            depthTarget: target as never,
            backgroundTarget: target as never,
            outputTarget: target as never,
        });
        const foam = facade.attachFluidSimulationCollectionRenderLayer(collection, {
            scene: scene as never,
            camera: camera as never,
            mode: "foam",
            depthTarget: target as never,
            colorTarget: target as never,
        });

        expect(() => (polygon._task as ReturnType<typeof mocks.renderTask>).execute()).not.toThrow();
        expect((polygon._task as ReturnType<typeof mocks.renderTask>).setSims).toHaveBeenCalledWith([]);
        expect(() => (foam._task as ReturnType<typeof mocks.renderTask>).execute()).not.toThrow();
    });

    it("defers shared task and aggregate teardown together", () => {
        const simulation = facade.createFluidSimulation(engine as never, options());
        const collection = facade.createFluidSimulationCollection(engine as never, [simulation]);
        const layer = facade.attachFluidSimulationCollectionRenderLayer(collection, {
            scene: scene as never,
            camera: camera as never,
            mode: "surface",
            depthTarget: target as never,
            backgroundTarget: target as never,
            outputTarget: target as never,
        });
        const task = layer._task as ReturnType<typeof mocks.renderTask>;
        const aggregate = layer._aggregate as { dispose: ReturnType<typeof vi.fn> };

        facade.detachFluidSimulationCollectionRenderLayer(collection, layer);

        expect(task.dispose).not.toHaveBeenCalled();
        expect(aggregate.dispose).not.toHaveBeenCalled();
        mocks.state.retirements[0]!();
        expect(task.dispose).toHaveBeenCalledOnce();
        expect(aggregate.dispose).toHaveBeenCalledOnce();
    });

    it("removes and clears a compositor before deferred teardown", () => {
        const compositorScene = { _frameGraph: { _tasks: [] as unknown[] } };
        const compositor = facade.createFluidSimulationRenderCompositor(engine as never, {
            scene: compositorScene as never,
            baseColorTarget: target as never,
        });
        const task = compositor._task as ReturnType<typeof mocks.renderTask> & { setLayers: ReturnType<typeof vi.fn> };
        compositorScene._frameGraph._tasks.push(task);

        facade.disposeFluidSimulationRenderCompositor(compositor);

        expect(task.executionEnabled).toBe(false);
        expect(task.setLayers).toHaveBeenLastCalledWith([], false);
        expect(compositorScene._frameGraph._tasks).toEqual([]);
        expect(task.dispose).not.toHaveBeenCalled();
        mocks.state.retirements[0]!();
        expect(task.dispose).toHaveBeenCalledOnce();
    });

    it("uses opaque channels and local render environments without exposing resources", async () => {
        const channel = facade.createFluidParticleChannel(engine as never, { capacity: 8, components: 4 });
        await expect(facade.readFluidParticleChannel(channel, { particleOffset: 1, particleCount: 1 })).resolves.toEqual(new Float32Array(4));

        const simulation = facade.createFluidSimulation(engine as never, options());
        const environmentSource = { specularCubeView: {}, cubeSampler: {} };
        const environment = facade.createFluidRenderEnvironment(environmentSource);
        facade.attachFluidSimulationRenderLayer(simulation, {
            scene: scene as never,
            camera: camera as never,
            mode: "surface",
            depthTarget: target as never,
            backgroundTarget: target as never,
            outputTarget: target as never,
            environment,
        });

        expect(mocks.state.tasks[0]!.setEnvMap).toHaveBeenCalledWith({
            view: environmentSource.specularCubeView,
            sampler: environmentSource.cubeSampler,
        });
    });

    it("rejects invalid packed channel and spatial-query inputs", () => {
        expect(() => facade.createFluidParticleChannel(engine as never, { capacity: 8, components: 4, initialData: new Float32Array(3) })).toThrow("complete particles");
        expect(() => facade.createFluidParticleSpatialQuery(engine as never, { maximumQueries: 0 })).toThrow("positive number");

        const simulation = facade.createFluidSimulation(engine as never, options());
        const stream = facade.createFluidSimulationParticleStream(simulation);
        const query = facade.createFluidParticleSpatialQuery(engine as never);
        expect(() =>
            facade.sampleFluidParticleSpatialQuery(query, stream, [
                {
                    key: "invalid",
                    bounds: { min: [1, 0, 0], max: [0, 1, 1] },
                },
            ])
        ).toThrow("finite ordered bounds");
    });

    it("retargets one polygon task to the live collection each frame", () => {
        const first = facade.createFluidSimulation(engine as never, options("FLIP"));
        const second = facade.createFluidSimulation(engine as never, options("FLIP", 24));
        const collection = facade.createFluidSimulationCollection(engine as never, [first, second]);
        facade.attachFluidSimulationCollectionRenderLayer(collection, {
            scene: scene as never,
            camera: camera as never,
            mode: "polygon",
            depthTarget: target as never,
            backgroundTarget: target as never,
            outputTarget: target as never,
        });
        const task = mocks.state.tasks[0]!;

        task.execute?.();

        expect(task.setSims).toHaveBeenCalledWith([mocks.state.backends[0], mocks.state.backends[1]]);
    });

    it("aggregates collection diagnostics without exposing diffuse or polygon resources", () => {
        const first = facade.createFluidSimulation(engine as never, options("FLIP"));
        const second = facade.createFluidSimulation(engine as never, options("FLIP", 24));
        mocks.state.backends[0]!.diffuse = {
            capacity: 100,
            counts: { total: 10, spray: 2, foam: 6, bubble: 2, capacity: 100 },
        };
        mocks.state.backends[1]!.diffuse = {
            capacity: 200,
            counts: { total: 20, spray: 4, foam: 12, bubble: 4, capacity: 200 },
        };
        const collection = facade.createFluidSimulationCollection(engine as never, [first, second]);

        expect(facade.getFluidSimulationCollectionDiagnostics(collection)).toMatchObject({
            simulationCount: 2,
            count: 32,
            gpuBytes: 2048,
            diffuse: { total: 30, spray: 6, foam: 18, bubble: 6, capacity: 300 },
        });
    });
});
