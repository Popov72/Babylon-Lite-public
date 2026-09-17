import { createEngine, disposeEngine } from "babylon-lite";
import {
    createFlipReferenceSimulation,
    disposeFlipReferenceSimulation,
    readFlipReferenceParticles,
    stepFlipReferenceSimulation,
    updateFlipReferenceSolids,
} from "../../packages/babylon-lite/src/fluid/experimental/flip-reference/solver";
import { compareParticles, measureParticles } from "./metrics";
import { createProductionComparison, readProductionParticles } from "./production";
import { loadComparisonSolids, updateComparisonSolids } from "./solids";
import { createComparisonViewer } from "./viewer";
import { decodeParticleState } from "./types";
import type { ParticleComparison, ParticleMetrics, ParticleState, ReferenceCase } from "./types";

export interface ComparisonFrame {
    frame: number;
    physicalSeconds: number;
    timelineSeconds: number;
    backend: string;
    actual: ParticleMetrics;
    reference: ParticleMetrics | null;
    comparison: ParticleComparison | null;
    pressure: unknown;
}

export interface ComparisonDriver {
    stepTo(index: number): Promise<ComparisonFrame>;
    show(kind: "actual" | "reference"): Promise<void>;
    snapshot(): number[];
    dispose(): Promise<void>;
}

declare global {
    interface Window {
        fluidReference?: ComparisonDriver;
        fluidReferenceError?: string;
    }
}

async function loadState(file: string): Promise<ParticleState> {
    const response = await fetch(file);
    if (!response.ok) {
        throw new Error(`Cannot load particle snapshot ${file}: ${response.status}.`);
    }
    return decodeParticleState(await response.arrayBuffer());
}

async function initialize(): Promise<void> {
    const canvas = document.querySelector("canvas")!;
    const status = document.querySelector("#status")!;
    const response = await fetch("case.json");
    if (!response.ok) {
        throw new Error(`Cannot load comparison case: ${response.status}.`);
    }
    const input = (await response.json()) as ReferenceCase;
    const query = new URLSearchParams(location.search);
    const backend = query.get("backend") ?? "reference";
    if (backend !== "reference" && backend !== "production") {
        throw new Error(`Unknown comparison backend: ${backend}.`);
    }
    const engine = await createEngine(canvas);
    const gpuErrors: string[] = [];
    engine._device.addEventListener("uncapturederror", (event) => {
        gpuErrors.push(event.error.message);
        window.fluidReferenceError = event.error.message;
    });
    const initial = await loadState(input.initialState);
    const solids = await loadComparisonSolids(input);
    updateComparisonSolids(input, solids, 0);
    const referenceSim =
        backend === "reference"
            ? createFlipReferenceSimulation(engine, {
                  referenceNumerics: true,
                  gridOrigin: input.grid.origin,
                  gridDimensions: input.grid.dimensions,
                  cellSize: input.grid.cellSize,
                  domainInset: input.domainInset,
                  solidsIncludeDomain: input.domainInset > 0,
                  removeInsideSolids: true,
                  ...(input.extremeVelocityRemoval && query.get("extreme-removal") !== "0"
                      ? { extremeVelocityRemoval: { frameDtSeconds: 1 / input.simulationFps, ...input.extremeVelocityRemoval } }
                      : {}),
                  initialPositions: initial.positions,
                  initialVelocities: initial.velocities,
                  gravity: input.gravity,
                  picFraction: input.picFraction,
                  pressureTolerance: Number(query.get("tolerance") ?? 1e-5),
                  maxPressureIterations: Number(query.get("iterations") ?? 400),
              })
            : null;
    const production = backend === "production" ? createProductionComparison(engine, input, initial, solids) : null;
    const viewer = query.get("images") === "0" ? null : await createComparisonViewer(engine, input, initial.positions.length / 3);
    const dt = 1 / (input.simulationFps * input.substeps);
    let frameIndex = 0;
    let actual = initial;
    let reference: ParticleState | null = input.referencePattern ? initial : null;
    let pressure: unknown = null;
    window.fluidReference = {
        async stepTo(index): Promise<ComparisonFrame> {
            if (!Number.isSafeInteger(index) || index < frameIndex || index >= input.frames) {
                throw new RangeError(`Requested comparison index ${index} is outside the remaining frame range.`);
            }
            while (frameIndex < index) {
                for (let substep = 1; substep <= input.substeps; substep++) {
                    const step = frameIndex * input.substeps + substep;
                    updateComparisonSolids(input, solids, step);
                    if (referenceSim) {
                        updateFlipReferenceSolids(referenceSim, solids.distances, solids.velocities);
                        const diagnostics = await stepFlipReferenceSimulation(referenceSim, dt);
                        if (!diagnostics.pressureConverged || Math.abs(diagnostics.consumedDtSeconds - dt) > 1e-10) {
                            throw new Error(`Reference substep did not satisfy its pressure/time contract: ${JSON.stringify(diagnostics)}.`);
                        }
                        pressure = diagnostics;
                    } else if (production) {
                        production.update();
                        await production.step();
                        pressure = production.sim.pressureDiagnostics ?? null;
                    }
                    if (gpuErrors.length) {
                        throw new Error(gpuErrors.join("\n"));
                    }
                }
                frameIndex++;
            }
            actual = referenceSim ? await readFlipReferenceParticles(referenceSim) : await readProductionParticles(engine, production!.sim);
            reference = input.referencePattern ? await loadState(input.referencePattern.replace("{frame}", String(input.startFrame + frameIndex))) : null;
            return {
                frame: input.startFrame + frameIndex,
                physicalSeconds: frameIndex / input.simulationFps,
                timelineSeconds: frameIndex / input.timelineFps,
                backend,
                actual: measureParticles(actual, input.grid),
                reference: reference ? measureParticles(reference, input.grid) : null,
                comparison: reference ? compareParticles(actual, reference, input.grid) : null,
                pressure,
            };
        },
        async show(kind): Promise<void> {
            if (!viewer) {
                throw new Error("Image output was disabled for this run.");
            }
            const state = kind === "actual" ? actual : reference;
            if (!state) {
                throw new Error("This case has no Blender particle reference.");
            }
            status.textContent = `${kind === "reference" ? "Blender cache" : backend === "reference" ? "FLIP Reference" : "Production FLIP"} | frame ${input.startFrame + frameIndex} | physics ${(frameIndex / input.simulationFps).toFixed(3)} s | ${state.positions.length / 3} markers`;
            await viewer.show(state, frameIndex * input.substeps);
            if (gpuErrors.length) {
                throw new Error(gpuErrors.join("\n"));
            }
        },
        snapshot(): number[] {
            return [...actual.positions, ...actual.velocities];
        },
        async dispose(): Promise<void> {
            await engine._device.queue.onSubmittedWorkDone();
            if (referenceSim) {
                disposeFlipReferenceSimulation(referenceSim);
            }
            production?.dispose();
            viewer?.dispose();
            disposeEngine(engine);
        },
    };
    status.textContent = "Ready for synchronized comparison.";
}

initialize().catch((error: unknown) => {
    window.fluidReferenceError = error instanceof Error ? (error.stack ?? error.message) : String(error);
    document.querySelector("#status")!.textContent = window.fluidReferenceError;
    console.error(error);
});
