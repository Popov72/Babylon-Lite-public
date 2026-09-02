import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const MLS_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts").replace(/\\/g, "/")}`;

test("MLS-MPM preserves sub-0.05 uniform force and gravity across capacity profiles", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createMlsMpmSim } from "${MLS_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

function velocityReadableEngine(device) {
    return {
        _device: new Proxy(device, {
            get(target, prop) {
                if (prop === "createBuffer") {
                    return (descriptor) => target.createBuffer(
                        descriptor.label === "mpm-render-vel"
                            ? { ...descriptor, usage: descriptor.usage | GPUBufferUsage.COPY_SRC }
                            : descriptor
                    );
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        }),
    };
}

async function runCapacity(device, capacity) {
    const positions = new Float32Array(capacity * 3);
    for (let i = 0; i < capacity; i++) {
        positions[i * 3] = 1.25 + (i % 4) * 0.5;
        positions[i * 3 + 1] = 2.25 + (Math.floor(i / 4) % 4) * 0.5;
        positions[i * 3 + 2] = 1.25 + (Math.floor(i / 16) % 4) * 0.5;
    }
    const sim = createMlsMpmSim(velocityReadableEngine(device), {
        count: capacity,
        initialPositions: positions,
        warmupFrames: Math.ceil(capacity / 64),
        boundsMin: [0, 0, 0],
        boundsMax: [6, 6, 6],
        gridDim: [12, 12, 12],
        dx: 0.5,
        particleRadius: 0.05,
        gravity: 1,
        restDensity: 3,
        stiffness: 0,
        viscosity: 0,
        damping: 1,
        affineDamping: 1,
        groundDamp: 1,
        substeps: 4,
        maxSubDt: 1,
    });
    const forceBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(forceBuffer, 0, new Float32Array([1, 0, 0, 0]));
    sim.setForceField({
        struct: "struct ForceFieldParams { acceleration: vec4<f32>, };",
        wgsl: "fn externalForce(position: vec3<f32>, velocity: vec3<f32>, dt: f32) -> vec3<f32> { return forceFieldParams.acceleration.xyz * dt; }",
        buffer: forceBuffer,
    });
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 0.04);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const sampleCount = 128;
    const readback = device.createBuffer({ size: sampleCount * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = device.createCommandEncoder();
    copy.copyBufferToBuffer(sim.velocityBuffer, 0, readback, 0, sampleCount * 16);
    device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const velocities = new Float32Array(readback.getMappedRange());
    let x = 0;
    let y = 0;
    let active = 0;
    for (let i = 0; i < sampleCount; i++) {
        if (Number.isFinite(velocities[i * 4]) && Number.isFinite(velocities[i * 4 + 1])) {
            x += velocities[i * 4];
            y += velocities[i * 4 + 1];
            active++;
        }
    }
    readback.unmap();
    readback.destroy();
    sim.dispose();
    forceBuffer.destroy();
    return { capacity, active, x: x / active, y: y / active };
}

async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const highCapacity = Math.min(800000, Math.floor(device.limits.maxStorageBufferBindingSize / 80));
    const capacities = [...new Set([80000, 600000, highCapacity])];
    const results = [];
    for (const capacity of capacities) {
        results.push(await runCapacity(device, capacity));
    }
    return { highCapacity, results };
}

main()
    .then((result) => {
        canvas.dataset.result = JSON.stringify(result);
    })
    .catch((error) => {
        canvas.dataset.error = error?.message ?? String(error);
    });
</script>`);

    await page.waitForFunction(() => {
        const canvas = document.querySelector("canvas");
        return Boolean(canvas?.dataset.result || canvas?.dataset.error);
    });
    const state = await page.locator("canvas").evaluate((node) => ({ error: node.dataset.error, result: node.dataset.result }));
    expect(state.error).toBeUndefined();
    const result = JSON.parse(state.result!);
    expect(result.highCapacity).toBeGreaterThan(600_000);
    expect(result.results.map((entry: { capacity: number }) => entry.capacity)).toEqual([80_000, 600_000, result.highCapacity]);
    for (const entry of result.results) {
        expect(entry.active).toBe(128);
        expect(entry.x).toBeGreaterThan(0.03);
        expect(entry.x).toBeLessThan(0.05);
        expect(entry.y).toBeLessThan(-0.03);
        expect(entry.y).toBeGreaterThan(-0.05);
    }
});
