export interface ReferencePressureProbe {
    code: string;
    bindings: Readonly<Record<string, readonly number[]>>;
    fused: boolean;
    workgroupSize: number;
}

export async function runReferencePressureProbe({ code, bindings, fused, workgroupSize }: ReferencePressureProbe) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        throw new Error("WebGPU is required for the Reference pressure regression.");
    }
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxComputeInvocationsPerWorkgroup: workgroupSize,
            maxComputeWorkgroupSizeX: workgroupSize,
            maxComputeWorkgroupStorageSize: Math.max(16384, workgroupSize * 16 + 48),
        },
    });
    const size = 8192;
    const groups = size / 128;
    const controlOffset = (size + groups) * 4;
    const results: { scenario: string; iterations: number; relativeResidual: number; maxError: number; failure: number }[] = [];
    device.pushErrorScope("validation");
    try {
        const module = device.createShaderModule({ code });
        const entries = [
            "prepareGpuPressure",
            "initializeGpuPressure",
            "initializeParallelGpuPressure",
            "beginParallelPressure",
            "applyParallelPressure",
            "alphaParallelPressure",
            "updateParallelPressure",
            "betaParallelPressure",
            "finishParallelPressureIteration",
            "finishParallelPressure",
            "continueParallelPressure",
            "resumeGpuPressure",
        ];
        const pipelines = new Map<string, GPUComputePipeline>();
        for (const entryPoint of entries) {
            const wide = ["initializeParallelGpuPressure", "resumeGpuPressure", "finishParallelPressureIteration"].includes(entryPoint);
            pipelines.set(
                entryPoint,
                device.createComputePipeline({
                    layout: "auto",
                    compute: { module, entryPoint, ...(wide ? { constants: { pressureWorkgroupSize: workgroupSize } } : {}) },
                })
            );
        }
        for (const scenario of [
            "open",
            "gauges",
            "large-pressure-offset",
            "warm-reject",
            "nonfinite-warm",
            "nonfinite-low-warm",
            "zero-rhs",
            "small-active",
            "short-budget",
            "bad-diagonal",
            "scratch-fallback",
        ]) {
            const active = scenario === "small-active" ? 64 : size;
            const limit = scenario === "short-budget" ? 3 : 400;
            const cells = new Float32Array(size * 16);
            const largeOffset = scenario === "large-pressure-offset";
            const truth = largeOffset ? new Float64Array(size) : new Float32Array(size);
            const initial = new Float32Array((size + groups + 2) * 4);
            const gauge = (index: number) => scenario === "gauges" && (index === 0 || index === 8);
            const neighbor = (index: number, axis: number, sign: number): number => {
                const coordinate = [index % 16, Math.floor(index / 16) % 16, Math.floor(index / 256)];
                const next = coordinate[axis]! + sign;
                if (next < 0 || next >= [16, 16, 32][axis]!) {
                    return -1;
                }
                if (scenario === "gauges" && axis === 0 && Math.floor(coordinate[0]! / 8) !== Math.floor(next / 8)) {
                    return -1;
                }
                const result = index + sign * [1, 16, 256][axis]!;
                return result < active ? result : -1;
            };
            for (let i = 0; i < active; i++) {
                truth[i] = gauge(i)
                    ? 0
                    : (largeOffset ? 32768 : 0) +
                      Math.sin(((i % 16) + 0.5) * 0.17) * Math.cos(((Math.floor(i / 16) % 16) + 0.5) * 0.13) * Math.sin((Math.floor(i / 256) + 0.5) * 0.11);
                cells[i * 16] = -1;
                cells[i * 16 + 3] = 1;
                cells[i * 16 + 7] = gauge(i) ? 1 : 0;
                let diagonal = largeOffset ? (i === 0 ? 0.001 : 0) : scenario === "gauges" ? 0 : 0.025;
                for (let axis = 0; axis < 3; axis++) {
                    for (const sign of [-1, 1]) {
                        const n = neighbor(i, axis, sign);
                        if (n >= 0) {
                            const weight = Math.fround(0.5 + ((Math.min(i, n) * 17 + axis * 13) % 29) / 29);
                            cells[i * 16 + (sign > 0 ? 4 : 8) + axis] = weight;
                            diagonal += weight;
                        } else if (scenario !== "gauges" && !largeOffset) {
                            diagonal += 1;
                        }
                    }
                }
                cells[i * 16 + 1] = diagonal;
            }
            for (let i = 0; i < active; i++) {
                let rhs = cells[i * 16 + 1]! * truth[i]!;
                for (let axis = 0; axis < 3; axis++) {
                    for (const sign of [-1, 1]) {
                        const n = neighbor(i, axis, sign);
                        if (n >= 0) {
                            rhs -= cells[i * 16 + (sign > 0 ? 4 : 8) + axis]! * truth[n]!;
                        }
                    }
                }
                cells[i * 16 + 2] = scenario === "zero-rhs" || gauge(i) ? 0 : rhs;
                initial[i * 4] = largeOffset ? 32768 : scenario === "warm-reject" ? 100000 : scenario === "nonfinite-warm" ? NaN : 0;
            }
            if (scenario === "bad-diagonal") {
                cells[1] = -1;
            }
            const buffers = new Map<number, GPUBuffer>();
            const owned: GPUBuffer[] = [];
            const allocate = (binding: number, bytes: number, usage: GPUBufferUsageFlags) => {
                const buffer = device.createBuffer({ size: bytes, usage });
                buffers.set(binding, buffer);
                owned.push(buffer);
                return buffer;
            };
            const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
            allocate(0, 176, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            allocate(5, cells.byteLength, storage);
            allocate(6, active * 4, storage);
            allocate(8, initial.byteLength, storage);
            const scratch = allocate(9, scenario === "scratch-fallback" ? 16 : active * 16, storage);
            allocate(11, 128, storage);
            allocate(14, 72, storage | GPUBufferUsage.INDIRECT);
            allocate(19, size * 64, storage);
            const low = allocate(21, size * 4, storage);
            const readback = device.createBuffer({ size: initial.byteLength + 128 + scratch.size + low.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            owned.push(readback);
            try {
                const params = new Float32Array(44);
                const integers = new Uint32Array(params.buffer);
                integers.set([16, 16, 32], 0);
                params[7] = 0.1;
                integers[16] = size;
                integers[23] = groups;
                params[24] = 1e-10;
                params[25] = 1e-16;
                params[43] = limit;
                const state = new Uint32Array(32);
                const time = new Float32Array(state.buffer);
                state[0] = size;
                state[2] = 1;
                state[8] = active;
                time[4] = time[20] = 0.01;
                const args = new Uint32Array(18).fill(1);
                args[0] = groups;
                device.queue.writeBuffer(buffers.get(0)!, 0, params);
                device.queue.writeBuffer(buffers.get(5)!, 0, cells);
                device.queue.writeBuffer(
                    buffers.get(6)!,
                    0,
                    Uint32Array.from({ length: active }, (_, i) => i)
                );
                device.queue.writeBuffer(buffers.get(8)!, 0, initial);
                if (scenario === "nonfinite-low-warm") {
                    device.queue.writeBuffer(low, 0, new Float32Array(size).fill(NaN));
                }
                device.queue.writeBuffer(buffers.get(11)!, 0, state);
                device.queue.writeBuffer(buffers.get(14)!, 0, args);
                const bindGroups = new Map<string, GPUBindGroup>();
                for (const entry of entries) {
                    bindGroups.set(
                        entry,
                        device.createBindGroup({
                            layout: pipelines.get(entry)!.getBindGroupLayout(0),
                            entries: bindings[entry]!.map((binding) => ({ binding, resource: { buffer: buffers.get(binding)! } })),
                        })
                    );
                }
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass();
                const dispatch = (entry: string, indirectOffset?: number): void => {
                    pass.setPipeline(pipelines.get(entry)!);
                    pass.setBindGroup(0, bindGroups.get(entry)!);
                    if (indirectOffset === undefined) {
                        pass.dispatchWorkgroups(1);
                    } else {
                        pass.dispatchWorkgroupsIndirect(buffers.get(14)!, indirectOffset);
                    }
                };
                dispatch("prepareGpuPressure", 0);
                dispatch("initializeGpuPressure", 0);
                dispatch("initializeParallelGpuPressure", 48);
                dispatch("beginParallelPressure");
                for (let iteration = 0; iteration < Math.min(128, limit); iteration++) {
                    dispatch("applyParallelPressure", 0);
                    if (fused) {
                        dispatch("finishParallelPressureIteration");
                    } else {
                        dispatch("alphaParallelPressure");
                        dispatch("updateParallelPressure", 0);
                        dispatch("betaParallelPressure");
                    }
                }
                dispatch("finishParallelPressure");
                dispatch("continueParallelPressure", 0);
                dispatch("resumeGpuPressure", 48);
                pass.end();
                encoder.copyBufferToBuffer(buffers.get(8)!, 0, readback, 0, initial.byteLength);
                encoder.copyBufferToBuffer(buffers.get(11)!, 0, readback, initial.byteLength, 128);
                encoder.copyBufferToBuffer(scratch, 0, readback, initial.byteLength + 128, scratch.size);
                encoder.copyBufferToBuffer(low, 0, readback, initial.byteLength + 128 + scratch.size, low.size);
                device.queue.submit([encoder.finish()]);
                await readback.mapAsync(GPUMapMode.READ);
                const mapped = readback.getMappedRange();
                const values = new Float32Array(mapped);
                const status = new Uint32Array(mapped, initial.byteLength, 32);
                const iterations = values[controlOffset + 6]!;
                const failure = values[controlOffset + 7]!;
                const lowParts = new Float32Array(mapped, initial.byteLength + 128 + scratch.size, size);
                if (["open", "gauges", "large-pressure-offset", "warm-reject", "nonfinite-warm", "nonfinite-low-warm"].includes(scenario)) {
                    const directions = new Float32Array(mapped, initial.byteLength + 128, scratch.size / 4);
                    if (!directions.some((value, index) => index % 4 === 0 && value !== 0)) {
                        throw new Error(`${scenario}: the parallel prefix did not execute.`);
                    }
                }
                let maxError = 0;
                if (scenario === "bad-diagonal" || scenario === "short-budget") {
                    if (!(status[1]! & 16) || failure !== (scenario === "bad-diagonal" ? 2 : 4) || (scenario === "short-budget" && iterations !== 3)) {
                        throw new Error(`${scenario}: pressure failure or iteration budget was lost.`);
                    }
                } else {
                    if (
                        status[1] !== 0 ||
                        values[controlOffset + 3] !== 0 ||
                        values[controlOffset + 1]! > Math.max(params[25]!, params[24]! * values[controlOffset + 2]!) * 1.0001
                    ) {
                        throw new Error(
                            `${scenario}: pressure did not satisfy the unchanged true-residual tolerance (iterations ${iterations}, code ${failure}, relative ${Math.sqrt(values[controlOffset + 1]! / Math.max(1e-30, values[controlOffset + 2]!))}).`
                        );
                    }
                    for (let i = 0; i < active; i++) {
                        maxError = Math.max(maxError, Math.abs(values[i * 4]! + lowParts[i]! - (scenario === "zero-rhs" ? 0 : truth[i]!)));
                    }
                    let gradientError = 0;
                    if (largeOffset) {
                        // A weak anchor permits a common offset at this tolerance; projection depends on the gradients.
                        for (let i = 0; i < active; i++) {
                            for (let axis = 0; axis < 3; axis++) {
                                const next = neighbor(i, axis, 1);
                                if (next >= 0) {
                                    const difference = values[i * 4]! - values[next * 4]! + lowParts[i]! - lowParts[next]!;
                                    gradientError = Math.max(gradientError, Math.abs(difference - (truth[i]! - truth[next]!)));
                                }
                            }
                        }
                    }
                    if (!(largeOffset ? gradientError < 0.002 : maxError < 0.002) || (scenario === "zero-rhs" && iterations !== 0)) {
                        throw new Error(`${scenario}: incorrect pressure field (maximum error ${maxError}, gradient error ${gradientError}) or unnecessary zero-RHS iterations.`);
                    }
                    if (largeOffset && !lowParts.some((value) => Math.abs(value) > 1e-5)) {
                        throw new Error("The large-offset solution lost its low pressure components.");
                    }
                }
                results.push({ scenario, iterations, failure, maxError, relativeResidual: Math.sqrt(values[controlOffset + 1]! / Math.max(1e-30, values[controlOffset + 2]!)) });
                readback.unmap();
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
