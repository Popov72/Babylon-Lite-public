import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const FLIP_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/solvers/flip-sim.ts").replace(/\\/g, "/")}`;

test("FLIP preserves liquid height without corner marker aggregation", async ({ page }) => {
    test.setTimeout(300_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

async function main() {
    const engine = await createEngine(canvas);
    const dx = 40 / 91;
    const wall = 13.5;

    async function run(cflNumber) {
        const sim = createFlipSim(engine, {
            count: 651_086,
            boundsMin: [-20, 0, -20],
            boundsMax: [20, 20, 20],
            groundY: 0,
            dx,
            markersPerCell: 8,
            particleRadius: 0.1125,
            gravity: 43.3,
            flipRatio: 0.95,
            pressureIterations: 40,
            pressureRelaxation: 0.8,
            minSubsteps: 2,
            maxSubsteps: 8,
            cflNumber,
            maxSubDt: 0.0084,
        });
        sim.setFlow({
            emitters: [{
                id: "initial",
                name: "Initial",
                enabled: true,
                behavior: "initial",
                transform: { position: [0, 10, 0], rotation: [0, 0, 0, 1], scale: [6, 2, 6] },
                shape: { type: "box", size: [4, 6, 4] },
                sampling: "volume",
                velocity: [0, 0, 0],
                velocitySpace: "world",
                spread: 0,
            }],
            sinks: [],
        });
        const sdfBuffer = engine._device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        engine._device.queue.writeBuffer(sdfBuffer, 0, new Float32Array([-wall, 0, -wall, 0, wall, 22.5, wall, 0]));
        sim.setSceneSdf({
            struct: "struct SceneSdfParams { lo: vec4<f32>, hi: vec4<f32>, };",
            sdf: \`fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let d1 = pt - sceneSdfParams.lo.xyz;
    let d2 = sceneSdfParams.hi.xyz - pt;
    return min(min(min(d1.x, d1.y), d1.z), min(min(d2.x, d2.y), d2.z));
}\`,
            buffer: sdfBuffer,
        });
        sim.reset();

        async function sample() {
            const bytes = sim.count * 16;
            const readback = engine._device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const encoder = engine._device.createCommandEncoder();
            encoder.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, bytes);
            engine._device.queue.submit([encoder.finish()]);
            await readback.mapAsync(GPUMapMode.READ);
            const positions = new Float32Array(readback.getMappedRange().slice(0));
            const interiorY = [];
            const cornerY = [];
            const occupied = new Set();
            for (let index = 0; index < sim.renderCount; index++) {
                const x = positions[index * 4];
                const y = positions[index * 4 + 1];
                const z = positions[index * 4 + 2];
                if (Math.max(Math.abs(x), Math.abs(z)) < wall - dx * 3) interiorY.push(y);
                if (Math.abs(x) > wall - dx * 1.5 && Math.abs(z) > wall - dx * 1.5) cornerY.push(y);
                occupied.add(\`\${Math.floor((x + 20) / dx)},\${Math.floor(y / dx)},\${Math.floor((z + 20) / dx)}\`);
            }
            interiorY.sort((a, b) => a - b);
            cornerY.sort((a, b) => a - b);
            const percentile = (values, p) => values[Math.min(values.length - 1, Math.floor(values.length * p))] ?? 0;
            const result = {
                interiorTop: percentile(interiorY, 0.99),
                cornerTop: percentile(cornerY, 0.99),
                cornerCount: cornerY.length,
                occupied: occupied.size,
            };
            readback.unmap();
            readback.destroy();
            return result;
        }

        let early;
        for (let frame = 0; frame < 1200; frame++) {
            const encoder = engine._device.createCommandEncoder();
            sim.step(encoder, 1 / 60);
            engine._device.queue.submit([encoder.finish()]);
            if (frame === 119 || frame === 1199) {
                await engine._device.queue.onSubmittedWorkDone();
                if (frame === 119) early = await sample();
            } else if (frame % 60 === 59) {
                await engine._device.queue.onSubmittedWorkDone();
            }
        }
        const late = await sample();
        sim.dispose();
        sdfBuffer.destroy();
        return { early, late };
    }

    canvas.dataset.result = JSON.stringify({ adaptive: await run(2) });
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 280_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        adaptive: {
            early: { interiorTop: number; cornerTop: number; cornerCount: number; occupied: number };
            late: { interiorTop: number; cornerTop: number; cornerCount: number; occupied: number };
        };
    };
    expect(result.adaptive.late.interiorTop).toBeGreaterThan(result.adaptive.early.interiorTop * 0.9);
    expect(result.adaptive.late.occupied).toBeGreaterThan(result.adaptive.early.occupied * 0.95);
    expect(result.adaptive.late.cornerCount).toBeLessThan(result.adaptive.early.cornerCount * 1.25);
    expect(result.adaptive.late.cornerTop).toBeLessThan(result.adaptive.late.interiorTop + 2 * (40 / 91));
});
