import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { pbmpmParamKeysForMaterial } from "../../../../packages/babylon-lite/src/fluid/solvers/pbmpm-sim";
import { captureFluidFlowCarries, restoreFluidFlowCarries } from "../../../../packages/babylon-lite/src/fluid/core/sim-common";
import {
    buildLinearDispatchIndexWgsl,
    createMlsFixedPointCodec,
    MLS_FIXED_POINT_CODEC_FINALIZATION_WGSL,
    MLS_MIN_CELL_PARTICLE_CONTRIBUTORS,
    type MlsFixedPointCodec,
} from "../../../../packages/babylon-lite/src/fluid/solvers/wgsl-shared";

const repoFile = (...parts: string[]) => readFileSync(resolve(process.cwd(), ...parts), "utf8");

describe("Fluid solver particle dispatch helpers", () => {
    it("emits the shared two-dimensional workgroup linearization formula", () => {
        expect(buildLinearDispatchIndexWgsl(64)).toContain("gid.x + gid.y * numWorkgroups.x * 64u");
    });

    describe("fluid flow state transfer", () => {
        it("restores emitter and sink carries by stable ID after reordering", () => {
            const saved = captureFluidFlowCarries(["left", "right"], new Float64Array([1.25, 2.5]));
            const restored = new Float64Array(3);
            restoreFluidFlowCarries(["right", "new", "left"], restored, saved);

            expect([...restored]).toEqual([2.5, 0, 1.25]);
        });

        it("keeps FLIP transfer metadata ID-stable", () => {
            const source = repoFile("packages/babylon-lite/src/fluid/solvers/flip-sim.ts");
            expect(source).toContain("flowEmitterCursorId");
            expect(source).toContain("captureFluidFlowCarries(flowState.emitterIds");
            expect(source).toContain("restoreFluidFlowCarries(flowState.emitterIds");
            expect(source).not.toContain("flowState.emitterCarries.set(metadata.flowEmitterCarries)");
            expect(source).not.toContain("flowState.sinkCarries.set(metadata.flowSinkCarries)");
        });
    });

    it.each(["packages/babylon-lite/src/fluid/solvers/pbf-sim.ts", "packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts", "packages/babylon-lite/src/fluid/solvers/pbmpm-sim.ts"])(
        "removes one-dimensional particle indexing from %s",
        (file) => {
            const source = repoFile(file);
            expect(source).toContain("buildLinearDispatchIndexWgsl(WORKGROUP_SIZE)");
            expect(source).not.toMatch(/\b(?:let|var|const)\s+\w+\s*=\s*gid\.x;/);
            expect(source).not.toContain("gid.y * ${MAX_WORKGROUPS * WORKGROUP_SIZE}u");
        }
    );

    it.each([
        ["packages/babylon-lite/src/fluid/solvers/pbf-sim.ts", "fluid-debug"],
        ["packages/babylon-lite/src/fluid/solvers/flip-sim.ts", "flip-particle-debug"],
        ["packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts", "mpm-debug"],
        ["packages/babylon-lite/src/fluid/solvers/pbmpm-sim.ts", "pbmpm-debug"],
    ])("makes the aggregate-copy debug buffer a COPY_SRC in %s", (file, label) => {
        const aggregate = repoFile("packages/babylon-lite/src/fluid/core/fluid-particle-runtime.ts");
        const source = repoFile(file);
        const labelOffset = source.indexOf(`label: "${label}"`);
        const descriptorEnd = source.indexOf("});", labelOffset);

        expect(aggregate).toContain("encoder.copyBufferToBuffer(source.sim.debugBuffer");
        expect(labelOffset).toBeGreaterThanOrEqual(0);
        expect(descriptorEnd).toBeGreaterThan(labelOffset);
        expect(source.slice(labelOffset, descriptorEnd)).toContain("GPUBufferUsage.COPY_SRC");
    });
});

describe("FLIP timestep budgeting", () => {
    it("uses the shared scheduler with CFL, capillary, and configured substep limits", () => {
        const source = repoFile("packages/babylon-lite/src/fluid/solvers/flip-sim.ts");
        expect(source).toContain("createFluidTimestepScheduler()");
        expect(source).toContain("scheduleFluidTimestep(timestepScheduler, dt, minSubsteps, adaptiveMaxSubDt, maxSubsteps)");
        expect(source).toContain("deferFluidTimestep(timestepScheduler, dt)");
        expect(source).toContain("return getFluidTimestepDiagnostics(timestepScheduler)");
        expect(source).toContain("resetFluidTimestepScheduler(timestepScheduler)");
        expect(source).not.toContain("const hardDtSteps");
    });
});

describe("MLS-MPM fixed-point codec", () => {
    const physicalConfig = {
        dx: 0.25,
        subDt: 1 / 120,
        restDensity: 3,
        stiffness: 350,
        viscosity: 0.3,
    };
    const deviceHighCapacity = Math.floor((128 * 1024 * 1024) / 80);

    function depositMarker(codec: MlsFixedPointCodec, fraction: [number, number, number], velocity = 0) {
        const weights = fraction.map((f) => {
            const d = f - 0.5;
            return [0.5 * (0.5 - d) ** 2, 0.75 - d * d, 0.5 * (0.5 + d) ** 2];
        });
        let mass = 0;
        let momentum = 0;
        const firstMoment: [number, number, number] = [0, 0, 0];
        const momentumFirstMoment: [number, number, number] = [0, 0, 0];
        let support = 0;
        for (let gx = 0; gx < 3; gx++) {
            for (let gy = 0; gy < 3; gy++) {
                for (let gz = 0; gz < 3; gz++) {
                    const weight = weights[0]![gx]! * weights[1]![gy]! * weights[2]![gz]!;
                    const deposited = Math.round(weight * codec.massScale) * codec.inverseMassScale;
                    const depositedMomentum = Math.round(weight * velocity * codec.momentumScale) * codec.inverseMomentumScale;
                    mass += deposited;
                    momentum += depositedMomentum;
                    if (deposited > 0) {
                        support++;
                    }
                    firstMoment[0] += deposited * (gx - 0.5);
                    firstMoment[1] += deposited * (gy - 0.5);
                    firstMoment[2] += deposited * (gz - 0.5);
                    momentumFirstMoment[0] += depositedMomentum * (gx - 0.5);
                    momentumFirstMoment[1] += depositedMomentum * (gy - 0.5);
                    momentumFirstMoment[2] += depositedMomentum * (gz - 0.5);
                }
            }
        }
        return { mass, momentum, firstMoment, momentumFirstMoment, support };
    }

    it.each([
        ["80k", 80_000],
        ["600k", 600_000],
        ["device-supported high", deviceHighCapacity],
    ])("preserves mass, first moments, and stencil support at %s allocation capacity (%d)", (_label, capacity) => {
        const restDensity = Math.max(2, capacity / 8192);
        const codec = createMlsFixedPointCodec({ ...physicalConfig, restDensity });
        const centered = depositMarker(codec, [0.5, 0.5, 0.5], 0.04);
        const offset = depositMarker(codec, [0.37, 0.61, 0.42], 0.04);

        expect(capacity).toBeGreaterThanOrEqual(80_000);
        expect(capacity * 80).toBeLessThanOrEqual(128 * 1024 * 1024);
        expect(centered.mass).toBeCloseTo(1, 3);
        expect(centered.support).toBe(27);
        expect(offset.mass).toBeCloseTo(1, 3);
        expect(Math.abs(offset.firstMoment[0] - offset.mass * 0.37)).toBeLessThan(0.001);
        expect(Math.abs(offset.firstMoment[1] - offset.mass * 0.61)).toBeLessThan(0.001);
        expect(Math.abs(offset.firstMoment[2] - offset.mass * 0.42)).toBeLessThan(0.001);
        expect(offset.support).toBe(27);
        expect(centered.momentum / centered.mass).toBeCloseTo(0.04, 2);
        expect(offset.momentum / offset.mass).toBeCloseTo(0.04, 2);
        expect(Math.abs(offset.momentumFirstMoment[0] - offset.momentum * 0.37)).toBeLessThan(0.002);
        expect(Math.abs(offset.momentumFirstMoment[1] - offset.momentum * 0.61)).toBeLessThan(0.002);
        expect(Math.abs(offset.momentumFirstMoment[2] - offset.momentum * 0.42)).toBeLessThan(0.002);
    });

    it("derives independent overflow-safe mass and momentum scales from local occupancy", () => {
        const codec = createMlsFixedPointCodec(physicalConfig);

        expect(codec.massScale).toBeGreaterThan(codec.momentumScale);
        expect(codec.maxVelocityContribution).toBeGreaterThan(codec.maxResolvedSpeed);
        expect(codec.cellContributorLimit).toBe(MLS_MIN_CELL_PARTICLE_CONTRIBUTORS);
        expect(codec.nodeContributorLimit).toBe(MLS_MIN_CELL_PARTICLE_CONTRIBUTORS * 27);
        expect(codec.massAccumulationBound).toBeLessThan(0x7fffffff * 0.5);
        expect(codec.momentumAccumulationBound).toBeLessThan(0x7fffffff * 0.5);
        expect(codec.densityRatioCap).toBeGreaterThan(codec.densityFloorRatio);
    });

    it("derives the codec from explicit observed occupancy instead of a rest-density heuristic", () => {
        const config = {
            dx: 0.18,
            subDt: 0.0167,
            restDensity: 14.5,
            stiffness: 80,
            viscosity: 0.04,
        };
        const initialCodec = createMlsFixedPointCodec(config);
        const observedCodec = createMlsFixedPointCodec(config, 69);

        expect(initialCodec.cellContributorLimit).toBe(MLS_MIN_CELL_PARTICLE_CONTRIBUTORS);
        expect(observedCodec.cellContributorLimit).toBe(69);
        expect(observedCodec.massAccumulationBound).toBeLessThan(0x7fffffff * 0.5);
        expect(observedCodec.momentumAccumulationBound).toBeLessThan(0x7fffffff * 0.5);
    });

    it("fails only when the observed occupancy and physical bounds have no integer codec", () => {
        const config = {
            dx: 0.5,
            subDt: 1 / 60,
            restDensity: 3,
            stiffness: 9_500_000,
            viscosity: 0,
        };

        expect(createMlsFixedPointCodec(config, 64).momentumScale).toBe(1);
        expect(() => createMlsFixedPointCodec(config, 65)).toThrow("no safe codec");
    });

    it("fails impossible ranges instead of silently overflowing 32-bit atomics", () => {
        expect(() =>
            createMlsFixedPointCodec({
                dx: 1,
                subDt: 1.0e-9,
                restDensity: 1,
                stiffness: 5_000,
                viscosity: 1,
            })
        ).toThrow("fixed-point accumulation");
    });

    it("derives a safe codec from the current GPU occupancy before MLS P2G accumulation", () => {
        const source = repoFile("packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts");
        expect(source).not.toContain("population > p.codecCounts.x");
        expect(source).toContain("atomicMax(&accumulationState[2], population)");
        expect(source).toContain('label: "mpm-params"');
        expect(source).toContain("GPUBufferUsage.STORAGE");
        expect(source).toContain('dispatch(encoder, "mpm-finalize-codec"');
        expect(source).toContain("encoder.clearBuffer(accumulationStateBuffer, 8, 4)");
        expect(source).toContain('label: "mpm-cell-count"');
        expect(source).toContain('dispatch(encoder, "mpm-finalize-dense-p2g"');
        expect(source).toContain("if (accumulationState[0] != 0u) { return; }");
        expect(MLS_FIXED_POINT_CODEC_FINALIZATION_WGSL).toContain("population * MLS_P2G_BASE_CELLS_PER_NODE");
        expect(MLS_FIXED_POINT_CODEC_FINALIZATION_WGSL).toContain("perContributorBudget < 2u");
        expect(source).toContain("Fixed-point accumulation stopped");
    });

    it("latches both rejected status maps and throws them before the next step", () => {
        const source = repoFile("packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts");
        expect(source).toContain('latchAsyncError("[MLS-MPM] Failed to read paged-grid overflow status.", error)');
        expect(source).toContain('latchAsyncError("[MLS-MPM] Failed to read fixed-point accumulation status.", error)');
        expect(source).toMatch(/step\(encoder:[\s\S]*?if \(pendingAsyncError\) \{\s*throw pendingAsyncError;/);
        expect(source).not.toContain("console.error");
    });

    it("stages MLS particle state and gates commit, foam, and copy on GPU overflow status", () => {
        const source = repoFile("packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts");
        expect(source).toContain('label: "mpm-particles-working"');
        expect(source).toContain('dispatch(encoder, "mpm-commit-frame"');
        expect(source).toContain("if (accumulationState[0] != 0u) { return; }");
        expect(source).toContain("if (pageState[2] != 0u) { return; }");
        expect(source.indexOf("encoder.copyBufferToBuffer(particleBuffer")).toBeLessThan(source.indexOf('dispatch(encoder, "mpm-flow-delete"'));
        expect(source.indexOf('dispatch(encoder, "mpm-commit-frame"')).toBeLessThan(source.indexOf('dispatch(encoder, "mpm-foam-emit"'));
        expect(source.indexOf('dispatch(encoder, "mpm-foam-update"')).toBeLessThan(source.indexOf('dispatch(encoder, "mpm-copy"'));
    });

    it("binds status generation and foam metadata to the encoded copy slot", () => {
        const source = repoFile("packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts");
        expect(source).toContain("const generation = pageStatusGenerations[i]!");
        expect(source).toContain("const generation = accumulationStatusGenerations[i]!");
        expect(source).toContain("pageStatusGenerations[pageStagingIndex] = pageStatusGeneration");
        expect(source).toContain("accumulationStatusGenerations[accumulationStagingIndex] = accumulationStatusGeneration");
        expect(source).toContain('if (pageStatusStates[i] === "copied")');
        expect(source).toContain('if (accumulationStatusStates[i] === "copied")');
        expect(source).toContain("publishFoamActiveSide(foamSide)");
        expect(source).not.toContain("const generation = pageStatusGeneration;");
        expect(source).not.toContain("const generation = accumulationStatusGeneration;");
    });

    it("publishes monotonic live page demand for MLS-MPM consumers", () => {
        const source = repoFile("packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts");
        expect(source).toContain("onPagedGridPages?: (requiredPages: number, capacity: number) => void");
        expect(source).toContain("sample > lastPublishedPageStatusSample");
        expect(source).toContain("options.onPagedGridPages?.(requiredPages, pagedGridMaxPages)");
        expect(source).toContain("lastPublishedPageStatusSample = -1");
    });

    it("pauses before encoding when either MLS status ring has no idle slot", () => {
        const source = repoFile("packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts");
        expect(source).toContain('const pageStagingIndex = pagedGrid ? pageStatusStates.indexOf("idle") : -1');
        expect(source).toContain('const accumulationStagingIndex = accumulationStatusStates.indexOf("idle")');
        expect(source).toContain("if ((pagedGrid && pageStagingIndex === -1) || accumulationStagingIndex === -1)");
        expect(source.indexOf("if ((pagedGrid && pageStagingIndex === -1)")).toBeLessThan(source.indexOf("encoder.copyBufferToBuffer(particleBuffer"));
    });

    it("defers blocked time through the shared bounded timestep scheduler", () => {
        const source = repoFile("packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts");
        expect(source).toContain("deferFluidTimestep(timestepScheduler, dt)");
        expect(source).toContain("scheduleFluidTimestep(timestepScheduler, dt, substepsMut, maxSubDt)");
        expect(source).toContain("schedule.substeps");
        expect(source).not.toContain("deferredFrameDt");
    });
});

describe("PB-MPM material controls", () => {
    it.each([0, 1, 2, 3])("keeps restitution available for material %d", (material) => {
        expect(pbmpmParamKeysForMaterial(material)).toContain("restitution");
    });
});
