import {
    createEngine,
    disposeEngine,
    createDefaultFluidForceField,
    createFluidConfiguredForceField,
    updateFluidConfiguredForceField,
    evaluateFluidForceFields,
    validateFluidForceFields,
    createFluidForceField,
    disposeFluidForceField,
    createFluidSimulation,
    disposeFluidSimulation,
    setFluidSimulationForceField,
    submitFluidSimulationStep,
    readFluidSimulationPositions,
    loadFlipReferenceBackend,
    resetFluidSimulation,
} from "../../../packages/babylon-lite/src/index.js";
import type { FluidForceFieldVector, FluidSimulationOptions } from "../../../packages/babylon-lite/src/index.js";
import type { ForceFieldRuntimeBinding } from "../../../packages/babylon-lite/src/fluid/core/fluid-runtime-bindings.js";

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

export async function runFluidForceFieldProbe(canvas: HTMLCanvasElement): Promise<{ method: FluidSimulationOptions["method"]; authoredDelta: number; baseDelta: number }[]> {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const errors: string[] = [];
    device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
    try {
        const definitions = validateFluidForceFields([
            { ...createDefaultFluidForceField("point", "point", [0, 0, 0]), strength: -3, falloffPower: 2, minDistance: 0.5, maxDistance: 8 },
            { ...createDefaultFluidForceField("guide", "guide", [0, 0, 0]), strength: 2, flowStrength: 4, spinStrength: 6, endCaps: false },
        ]);
        const force = createFluidConfiguredForceField(engine, definitions);
        const spec = (force._binding as ForceFieldRuntimeBinding).spec;
        const points: FluidForceFieldVector[] = [
            [1, 0, 0],
            [0.1, 0, 0],
            [0, 0, 0],
            [2, 3, 0],
            [9, 0, 0],
        ];
        const input = device.createBuffer({ size: points.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const output = device.createBuffer({ size: input.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readback = device.createBuffer({ size: input.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        try {
            device.queue.writeBuffer(input, 0, new Float32Array(points.flatMap((point) => [...point, 0])));
            const module = device.createShaderModule({
                code: `${spec.struct}
@group(0) @binding(0) var<uniform> forceFieldParams: ForceFieldParams;
@group(0) @binding(1) var<storage,read> points:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> result:array<vec4<f32>>;
${spec.wgsl}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    result[id.x]=vec4<f32>(externalForce(points[id.x].xyz,vec3<f32>(0.0),0.02),0.0);
}`,
            });
            const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
            const bindings = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: spec.buffer } },
                    { binding: 1, resource: { buffer: input } },
                    { binding: 2, resource: { buffer: output } },
                ],
            });
            async function deltas(): Promise<Float32Array> {
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindings);
                pass.dispatchWorkgroups(points.length);
                pass.end();
                encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
                device.queue.submit([encoder.finish()]);
                await readback.mapAsync(GPUMapMode.READ);
                const values = new Float32Array(readback.getMappedRange()).slice();
                readback.unmap();
                return values;
            }
            const numeric = await deltas();
            for (let index = 0; index < points.length; index++) {
                const expected = evaluateFluidForceFields(definitions, points[index]!);
                for (let axis = 0; axis < 3; axis++) {
                    assert(Math.abs(numeric[index * 4 + axis]! - expected[axis]! * 0.02) < 1e-5, `Force equation mismatch at ${index}/${axis}.`);
                }
            }
            updateFluidConfiguredForceField(
                force,
                definitions.map((field) => ({ ...field, enabled: false }))
            );
            assert(
                (await deltas()).every((value) => value === 0),
                "Disabled fields still accelerate particles."
            );
        } finally {
            disposeFluidForceField(force);
            input.destroy();
            output.destroy();
            readback.destroy();
        }
        const base = createFluidForceField(engine, {
            struct: "struct ForceFieldParams { acceleration:vec4<f32> }",
            wgsl: "fn externalForce(pos:vec3<f32>,vel:vec3<f32>,dt:f32)->vec3<f32>{return forceFieldParams.acceleration.xyz*dt;}",
            params: new Float32Array([0, 1, 0, 0]),
        });
        const composed = createFluidConfiguredForceField(
            engine,
            [{ ...createDefaultFluidForceField("point", "uniform", [-100, 1, 0]), strength: 2, falloffPower: 0, useMaxDistance: false }],
            base
        );
        const results: { method: FluidSimulationOptions["method"]; authoredDelta: number; baseDelta: number }[] = [];
        try {
            let retained = false;
            try {
                disposeFluidForceField(base);
            } catch {
                retained = true;
            }
            assert(retained, "Composed force did not retain its base.");
            const backend = await loadFlipReferenceBackend();
            let referenceRejected = false;
            try {
                const simulation = createFluidSimulation(engine, {
                    method: "FLIP",
                    backend,
                    particleCount: 64,
                    bounds: { min: [-2, 0, -2], max: [2, 4, 2] },
                    physicsScale: 1,
                    forceField: composed,
                });
                disposeFluidSimulation(simulation);
            } catch (error) {
                referenceRejected = String(error).includes("production");
            }
            assert(referenceRejected, "Reference accepted configured force fields.");
            const seed = new Float32Array(64 * 3);
            for (let index = 0; index < 64; index++) {
                seed.set([-0.15 + (index % 4) * 0.1, 1 + (Math.floor(index / 4) % 4) * 0.1, -0.15 + Math.floor(index / 16) * 0.1], index * 3);
            }
            for (const method of ["PBF", "FLIP", "MLS-MPM", "PB-MPM"] as const) {
                const options = {
                    method,
                    particleCount: 64,
                    bounds: { min: [-2, 0, -2], max: [2, 4, 2] } as const,
                    physicsScale: 1,
                    gridResolution: 24,
                    initialPositions: seed,
                    initialVelocities: new Float32Array(seed.length),
                    physics: { gravity: 0, substeps: 2, minSubsteps: 2, maxSubsteps: 8, maxSubDtMs: 10 },
                };
                const idle = createFluidSimulation(engine, options);
                const plain = createFluidSimulation(engine, { ...options, forceField: base });
                const pushed = createFluidSimulation(engine, { ...options, forceField: composed });
                try {
                    for (let step = 0; step < 8; step++) {
                        await submitFluidSimulationStep(idle, 1 / 120);
                        await submitFluidSimulationStep(plain, 1 / 120);
                        await submitFluidSimulationStep(pushed, 1 / 120);
                    }
                    const zero = await readFluidSimulationPositions(idle);
                    const a = await readFluidSimulationPositions(plain);
                    const b = await readFluidSimulationPositions(pushed);
                    let authoredDelta = 0,
                        baseDelta = 0;
                    for (let index = 0; index < 64; index++) {
                        authoredDelta += (b[index * 4]! - a[index * 4]!) / 64;
                        baseDelta += (b[index * 4 + 1]! - zero[index * 4 + 1]!) / 64;
                    }
                    assert(Number.isFinite(authoredDelta) && authoredDelta > 1e-4, `${method} lost the authored contribution: ${authoredDelta}.`);
                    assert(Number.isFinite(baseDelta) && baseDelta > 1e-4, `${method} lost the base contribution: ${baseDelta}.`);
                    resetFluidSimulation(pushed);
                    const reset = await readFluidSimulationPositions(pushed);
                    for (let index = 0; index < 64; index++) {
                        for (let axis = 0; axis < 3; axis++) {
                            assert(Math.abs(reset[index * 4 + axis]! - seed[index * 3 + axis]!) < 1e-6, `${method} reset changed the seed.`);
                        }
                    }
                    setFluidSimulationForceField(pushed, null);
                    results.push({ method, authoredDelta, baseDelta });
                } finally {
                    disposeFluidSimulation(pushed);
                    disposeFluidSimulation(plain);
                    disposeFluidSimulation(idle);
                }
            }
        } finally {
            disposeFluidForceField(composed);
            disposeFluidForceField(base);
        }
        await device.queue.onSubmittedWorkDone();
        assert(errors.length === 0, errors.join("\n"));
        return results;
    } finally {
        disposeEngine(engine);
    }
}
