export async function runReferenceWhitewaterProbe(input: { code: string; bindings: Readonly<Record<string, readonly number[]>> }) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        throw new Error("WebGPU is required for the Reference whitewater regression.");
    }
    const device = await adapter.requestDevice();
    const particleCount = 137;
    const cellCount = 4096;
    const recipe = `
@group(0) @binding(22) var<storage, read_write> expectedParticles: array<Diffuse>;
@compute @workgroup_size(64)
fn expectedWhitewater(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&expectedParticles)) { return; }
    let index = gid.x / 8u;
    let sample = gid.x % 8u;
    let seed = (index * 2654435761u) ^ 40503u ^ (sample * 2246822519u);
    let radius = 0.05 * sqrt(fRnd(seed));
    let angle = 6.28318530718 * fRnd(seed * 3u + 1u);
    let offset = vec3<f32>(-radius * sin(angle), 0.0, -radius * cos(angle));
    let position = positions[index].xyz + offset + vec3<f32>(0.0, fRnd(seed * 7u + 5u) * 0.02, 0.0);
    expectedParticles[gid.x] = Diffuse(vec4<f32>(position, 2.0), vec4<f32>(vec3<f32>(0.0, 1.0, 0.0) + offset, 1.0));
}`;
    const entries = [
        "initializeFlipReferenceWhitewater",
        "prepareFlipReferenceWhitewater",
        "updateFlipReferenceWhitewater",
        "emitFlipReferenceWhitewater",
        "finishFlipReferenceWhitewater",
        "publishFlipReferenceWhitewater",
        "expectedWhitewater",
    ];
    const results: { capacity: number; emitted: number; rejected: number; cleared: boolean }[] = [];
    try {
        device.pushErrorScope("validation");
        const module = device.createShaderModule({ code: input.code + recipe });
        const pipelines = new Map<string, GPUComputePipeline>();
        for (const entryPoint of entries) {
            pipelines.set(entryPoint, device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } }));
        }
        for (const capacity of [4096, 1024]) {
            const owned: GPUBuffer[] = [];
            const buffers = new Map<number, GPUBuffer>();
            const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
            const make = (binding: number, size: number, usage = storage): GPUBuffer => {
                const buffer = device.createBuffer({ size, usage });
                buffers.set(binding, buffer);
                owned.push(buffer);
                return buffer;
            };
            const stride = Math.ceil(capacity / 64) * 64;
            const stateBytes = (64 + 2 * stride + capacity) * 4;
            make(0, 176, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            make(1, particleCount * 16);
            make(2, particleCount * 16);
            make(4, 32);
            make(7, 16);
            make(11, 128);
            make(12, 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            make(13, cellCount * 32);
            make(14, capacity * 32);
            make(15, stateBytes);
            make(16, 36, storage | GPUBufferUsage.INDIRECT);
            make(17, capacity * 32);
            make(18, stateBytes);
            make(19, 16, storage | GPUBufferUsage.INDIRECT);
            make(20, 12, storage | GPUBufferUsage.INDIRECT);
            make(22, particleCount * 8 * 32);
            try {
                const params = new Uint32Array(44);
                params.set([16, 16, 16], 0);
                new Float32Array(params.buffer)[7] = 1;
                const positions = new Float32Array(particleCount * 4);
                const velocities = new Float32Array(particleCount * 4);
                for (let i = 0; i < particleCount; i++) {
                    positions.set([5 + i * 0.001, 5, 5, 1], i * 4);
                    velocities[i * 4 + 1] = 1;
                }
                const runtime = new Uint32Array(32);
                runtime[0] = particleCount;
                new Float32Array(runtime.buffer)[19] = 0.02;
                const foam = new Float32Array(32);
                foam.set([0.5, 4, 0.05, 1.5, 0.25, 20, 0, 1000, 0.8, 0.5, 0.05, 0, 0.3, 2]);
                foam.set([0, 0, 1, 0, 0, 1, 0, 1], 16);
                new Uint32Array(foam.buffer)[29] = 1;
                const field = new Float32Array(cellCount * 8);
                for (let i = 0; i < cellCount; i++) {
                    field.set([0, -1, 0, 1, 0, 1, 1, 1], i * 8);
                }
                device.queue.writeBuffer(buffers.get(0)!, 0, params);
                device.queue.writeBuffer(buffers.get(1)!, 0, positions);
                device.queue.writeBuffer(buffers.get(2)!, 0, velocities);
                device.queue.writeBuffer(buffers.get(11)!, 0, runtime);
                device.queue.writeBuffer(buffers.get(12)!, 0, foam);
                device.queue.writeBuffer(buffers.get(13)!, 0, field);
                const bindGroups = new Map<string, GPUBindGroup>();
                for (const entry of entries) {
                    const bindings = entry === "expectedWhitewater" ? [1, 22] : input.bindings[entry]!;
                    bindGroups.set(
                        entry,
                        device.createBindGroup({
                            layout: pipelines.get(entry)!.getBindGroupLayout(0),
                            entries: bindings.map((binding) => ({ binding, resource: { buffer: buffers.get(binding)! } })),
                        })
                    );
                }
                function dispatch(pass: GPUComputePassEncoder, entry: string, count?: number, indirectBuffer?: GPUBuffer, offset = 0): void {
                    pass.setPipeline(pipelines.get(entry)!);
                    pass.setBindGroup(0, bindGroups.get(entry)!);
                    if (indirectBuffer) {
                        pass.dispatchWorkgroupsIndirect(indirectBuffer, offset);
                    } else {
                        pass.dispatchWorkgroups(count ?? 1);
                    }
                }
                function frame(encoder: GPUCommandEncoder, update: boolean): void {
                    const pass = encoder.beginComputePass();
                    dispatch(pass, "prepareFlipReferenceWhitewater");
                    if (update) {
                        dispatch(pass, "updateFlipReferenceWhitewater", undefined, buffers.get(16), 12);
                    }
                    dispatch(pass, "emitFlipReferenceWhitewater", undefined, buffers.get(16), 24);
                    dispatch(pass, "finishFlipReferenceWhitewater");
                    dispatch(pass, "publishFlipReferenceWhitewater", undefined, buffers.get(20));
                    pass.end();
                }
                async function read(buffer: GPUBuffer): Promise<ArrayBuffer> {
                    const staging = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
                    const encoder = device.createCommandEncoder();
                    encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
                    device.queue.submit([encoder.finish()]);
                    await staging.mapAsync(GPUMapMode.READ);
                    const result = staging.getMappedRange().slice(0);
                    staging.unmap();
                    staging.destroy();
                    return result;
                }
                const encoder = device.createCommandEncoder();
                const initialize = encoder.beginComputePass();
                dispatch(initialize, "initializeFlipReferenceWhitewater", Math.ceil(capacity / 64));
                initialize.end();
                device.queue.submit([encoder.finish()]);
                device.queue.writeBuffer(buffers.get(15)!, 48, new Uint32Array([1, device.limits.maxComputeWorkgroupsPerDimension]));
                const first = device.createCommandEncoder();
                frame(first, false);
                const expected = first.beginComputePass();
                dispatch(expected, "expectedWhitewater", Math.ceil((particleCount * 8) / 64));
                expected.end();
                device.queue.submit([first.finish()]);
                const working = new Uint32Array(await read(buffers.get(15)!));
                const published = new Float32Array(await read(buffers.get(17)!));
                const expectedData = new Float32Array(await read(buffers.get(22)!));
                const draw = new Uint32Array(await read(buffers.get(19)!));
                const count = Math.min(capacity, particleCount * 8);
                const side = working[3]!;
                const indices = Array.from(working.slice(64 + side * stride, 64 + side * stride + count));
                const free = Array.from(working.slice(64 + 2 * stride, 64 + 2 * stride + working[5]!));
                if (working[1 + side] !== count || draw[1] !== count || count + working[5]! !== capacity || new Set([...indices, ...free]).size !== capacity) {
                    throw new Error("Whitewater allocation lost, duplicated or overwrote a slot.");
                }
                for (let i = 0; i < count; i++) {
                    if (!published.slice(i * 8, i * 8 + 8).every(Number.isFinite) || published[i * 8 + 3] !== 2 || published[i * 8 + 7] !== 1) {
                        throw new Error("A generated particle lost its finite attributes, lifetime or kind.");
                    }
                }
                const records = (data: Float32Array, length: number) =>
                    Array.from({ length }, (_, i) => Array.from(data.slice(i * 8, i * 8 + 8))).sort((a, b) => {
                        for (let j = 0; j < 8; j++) {
                            if (a[j] !== b[j]) {
                                return a[j]! - b[j]!;
                            }
                        }
                        return 0;
                    });
                if (capacity > count) {
                    const actual = records(published, count);
                    const reference = records(expectedData, count);
                    if (actual.some((record, i) => record.some((value, j) => !Number.isFinite(value) || Math.abs(value - reference[i]![j]!) > 1e-6))) {
                        throw new Error("Batched allocation changed the generated particle attributes.");
                    }
                }
                if (working[7] !== particleCount * 8 - count) {
                    throw new Error("Saturation did not report the rejected birth count.");
                }
                new Uint32Array(foam.buffer)[29] = 0;
                device.queue.writeBuffer(buffers.get(12)!, 0, foam);
                const clearing = device.createCommandEncoder();
                frame(clearing, true);
                device.queue.submit([clearing.finish()]);
                const empty = new Float32Array(await read(buffers.get(17)!));
                const emptyDraw = new Uint32Array(await read(buffers.get(19)!));
                if (emptyDraw[1] !== 0 || empty.some((value) => value !== 0)) {
                    throw new Error("Shrinking publication left stale live or tail particles.");
                }
                const noWork = device.createCommandEncoder();
                frame(noWork, true);
                device.queue.submit([noWork.finish()]);
                const idleDispatch = new Uint32Array(await read(buffers.get(20)!));
                if (idleDispatch[0] !== 0) {
                    throw new Error("An already empty pool still dispatched capacity-sized publication.");
                }
                results.push({ capacity, emitted: count, rejected: working[7]!, cleared: true });
            } finally {
                for (const buffer of owned) {
                    buffer.destroy();
                }
            }
        }
        const validation = await device.popErrorScope();
        if (validation) {
            throw new Error(validation.message);
        }
        return results;
    } finally {
        device.destroy();
    }
}
