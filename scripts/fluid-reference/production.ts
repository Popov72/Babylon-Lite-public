import type { EngineContext } from "../../packages/babylon-lite/src/engine/engine";
import { createFlipSim } from "../../packages/babylon-lite/src/fluid/solvers/flip-sim";
import { createSceneSdfRuntimeBinding } from "../../packages/babylon-lite/src/fluid/core/fluid-runtime-bindings";
import type { FluidSim } from "../../packages/babylon-lite/src/fluid/core/sim-common";
import type { ParticleState, ReferenceCase } from "./types";
import type { ComparisonSolids } from "./solids";

export function createProductionComparison(engine: EngineContext, input: ReferenceCase, initial: ParticleState, solids: ComparisonSolids) {
    const dt = 1 / (input.simulationFps * input.substeps);
    const sim = createFlipSim(engine, {
        count: initial.positions.length / 3,
        boundsMin: [...input.grid.origin],
        boundsMax: input.grid.origin.map((v, a) => v + input.grid.dimensions[a]! * input.grid.cellSize) as [number, number, number],
        dx: input.grid.cellSize,
        gridDim: [...input.grid.dimensions],
        initialPositions: initial.positions,
        initialVelocities: initial.velocities,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: dt,
        cflNumber: 0,
        gravity: -input.gravity[1],
        flipRatio: 1 - input.picFraction,
        liquidSdf: true,
        ghostFluid: true,
        fractionalSolids: true,
        movingSolidBoundaries: true,
        pressureSolver: "multigrid",
        pressureIterations: 100,
        multigridCycles: 4,
        pressureTolerance: 0.001,
        pressureDiagnostics: true,
        reseedParticles: false,
        particleSheeting: false,
        restitution: 0,
        velocityDamping: 0,
    });
    const packed = new Float32Array(solids.distances.length * 4);
    const binding = createSceneSdfRuntimeBinding(engine, {
        struct: "struct SceneSdfParams { origin: vec4<f32>, dimensions: vec4<f32> };",
        params: Float32Array.from([...input.grid.origin, input.grid.cellSize, ...input.grid.dimensions.map((d) => d + 1), 0]),
        sdfGrid: packed,
        sdf: `
fn referenceField(pt: vec3<f32>) -> vec4<f32> {
    let dims = vec3<i32>(sceneSdfParams.dimensions.xyz);
    let g = (pt - sceneSdfParams.origin.xyz) / sceneSdfParams.origin.w;
    let base = clamp(vec3<i32>(floor(g)), vec3<i32>(0), dims - vec3<i32>(2));
    let f = clamp(g - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));
    var value = vec4<f32>(0.0);
    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                let c = base + vec3<i32>(x,y,z);
                let i = 4u * u32(c.x + dims.x * (c.y + dims.y*c.z));
                let w = select(1.0-f.x,f.x,x==1) * select(1.0-f.y,f.y,y==1) * select(1.0-f.z,f.z,z==1);
                value += w * vec4<f32>(sceneSdfGrid[i],sceneSdfGrid[i+1u],sceneSdfGrid[i+2u],sceneSdfGrid[i+3u]);
            }
        }
    }
    return value;
}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let value = referenceField(pt);
    if (dt == 0.0) { return value.x; }
    let h = 0.5 * sceneSdfParams.origin.w;
    let gradient = vec3<f32>(
        referenceField(pt + vec3<f32>(h,0,0)).x - referenceField(pt - vec3<f32>(h,0,0)).x,
        referenceField(pt + vec3<f32>(0,h,0)).x - referenceField(pt - vec3<f32>(0,h,0)).x,
        referenceField(pt + vec3<f32>(0,0,h)).x - referenceField(pt - vec3<f32>(0,0,h)).x
    ) / (2.0*h);
    return value.x - dt * dot(gradient, value.yzw);
}`,
    });
    const update = (): void => {
        for (let i = 0; i < solids.distances.length; i++) {
            packed[i * 4] = solids.distances[i]!;
            packed[i * 4 + 1] = solids.velocities[i * 3]!;
            packed[i * 4 + 2] = solids.velocities[i * 3 + 1]!;
            packed[i * 4 + 3] = solids.velocities[i * 3 + 2]!;
        }
        binding.updateSdfGrid(packed);
    };
    try {
        update();
        sim.setSceneSdf(binding.spec);
    } catch (error) {
        binding.dispose();
        sim.dispose();
        throw error;
    }
    return {
        sim,
        update,
        async step(): Promise<void> {
            const encoder = engine._device.createCommandEncoder({ label: "production-flip-comparison" });
            sim.step(encoder, dt);
            const timing = sim.timestepDiagnostics;
            if (timing && (timing.droppedSeconds > 1e-9 || timing.deferredSeconds > 1e-9)) {
                throw new Error(`Production comparison did not consume its timestep: ${JSON.stringify(timing)}.`);
            }
            engine._device.queue.submit([encoder.finish()]);
            await engine._device.queue.onSubmittedWorkDone();
        },
        dispose(): void {
            sim.dispose();
            binding.dispose();
        },
    };
}

export async function readProductionParticles(engine: EngineContext, sim: FluidSim): Promise<ParticleState> {
    const bytes = sim.count * 16;
    const buffer = engine._device.createBuffer({ size: bytes * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
        const encoder = engine._device.createCommandEncoder();
        encoder.copyBufferToBuffer(sim.positionBuffer, 0, buffer, 0, bytes);
        encoder.copyBufferToBuffer(sim.velocityBuffer, 0, buffer, bytes, bytes);
        engine._device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const raw = new Float32Array(buffer.getMappedRange());
        const positions = new Float32Array(sim.count * 3);
        const velocities = new Float32Array(sim.count * 3);
        for (let p = 0; p < sim.count; p++) {
            for (let axis = 0; axis < 3; axis++) {
                positions[p * 3 + axis] = raw[p * 4 + axis]!;
                velocities[p * 3 + axis] = raw[sim.count * 4 + p * 4 + axis]!;
            }
        }
        return { positions, velocities };
    } finally {
        buffer.destroy();
    }
}
