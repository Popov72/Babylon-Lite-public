export async function runReferencePressureProjectionProbe(code: string): Promise<number> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("WebGPU is required for the pressure projection regression.");
    }
    const device = await adapter.requestDevice();
    const owned: GPUBuffer[] = [];
    const allocate = (size: number, usage: GPUBufferUsageFlags) => {
        const buffer = device.createBuffer({ size, usage });
        owned.push(buffer);
        return buffer;
    };
    try {
        device.pushErrorScope("validation");
        const module = device.createShaderModule({ code });
        const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "project" } });
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const params = allocate(176, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const faces = allocate(11 * 32, storage | GPUBufferUsage.COPY_SRC);
        const cells = allocate(2 * 64, storage);
        const pressure = allocate(2 * 16, storage);
        const low = allocate(2 * 4, storage);
        const readback = allocate(faces.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const dimensions = new Uint32Array(44);
        dimensions.set([2, 1, 1], 0);
        dimensions.set([2, 11, 3, 4], 16);
        const faceData = new Float32Array(11 * 8);
        faceData[10] = 1;
        const cellData = new Float32Array(32);
        cellData[0] = cellData[16] = -1;
        cellData[3] = cellData[19] = 1;
        device.queue.writeBuffer(params, 0, dimensions);
        device.queue.writeBuffer(faces, 0, faceData);
        device.queue.writeBuffer(cells, 0, cellData);
        device.queue.writeBuffer(pressure, 0, new Float32Array([34388, 0, 0, 0, 34388, 0, 0, 0]));
        device.queue.writeBuffer(low, 0, new Float32Array([0.0001, 0.0002]));
        const bindings = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: params } },
                { binding: 4, resource: { buffer: faces } },
                { binding: 5, resource: { buffer: cells } },
                { binding: 8, resource: { buffer: pressure } },
                { binding: 21, resource: { buffer: low } },
            ],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(1);
        pass.end();
        encoder.copyBufferToBuffer(faces, 0, readback, 0, faces.size);
        device.queue.submit([encoder.finish()]);
        const validation = await device.popErrorScope();
        if (validation) {
            throw new Error(validation.message);
        }
        await readback.mapAsync(GPUMapMode.READ);
        const value = new Float32Array(readback.getMappedRange())[8]!;
        readback.unmap();
        if (Math.abs(value + 0.0001) > 1e-9) {
            throw new Error(`Pressure projection lost the low-component gradient: ${value}.`);
        }
        return value;
    } finally {
        for (const buffer of owned) {
            buffer.destroy();
        }
        device.destroy();
    }
}
