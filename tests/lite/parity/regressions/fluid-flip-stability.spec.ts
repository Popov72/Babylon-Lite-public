import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const FLIP_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/flip-sim.ts").replace(/\\/g, "/")}`;

test("FLIP transfers moving SDF boundary velocity to particles", async ({ page }) => {
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
    const sdfBuffer = engine._device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    engine._device.queue.writeBuffer(sdfBuffer, 0, new Float32Array([0.25, 1.5, 1.25, 0, 1, 0, 2, 1]));
    const sim = createFlipSim(engine, {
        count: 1,
        initialPositions: new Float32Array([0.1, 1, 1]),
        boundsMin: [-2, 0, -2],
        boundsMax: [2, 3, 2],
        groundY: 0,
        dx: 0.25,
        particleRadius: 0.05,
        gravity: 0,
        flipRatio: 0,
        pressureIterations: 1,
        minSubsteps: 1,
        maxSubDt: 1 / 60,
    });
    sim.setSceneSdf({
        struct: "struct SceneSdfParams { obstacle: vec4<f32>, motion: vec4<f32>, };",
        sdf: \`fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let cd = cos(sceneSdfParams.motion.z * dt);
    let sd = sin(sceneSdfParams.motion.z * dt);
    let c = sceneSdfParams.motion.x * cd + sceneSdfParams.motion.y * sd;
    let s = sceneSdfParams.motion.y * cd - sceneSdfParams.motion.x * sd;
    let lx = c * pt.x + s * pt.z;
    let lz = -s * pt.x + c * pt.z;
    let q = vec3<f32>(
        abs(lx) - sceneSdfParams.obstacle.x,
        abs(pt.y - sceneSdfParams.obstacle.y) - sceneSdfParams.obstacle.y,
        abs(lz) - sceneSdfParams.obstacle.z);
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}\`,
        buffer: sdfBuffer,
    });
    const encoder = engine._device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    engine._device.queue.submit([encoder.finish()]);
    await engine._device.queue.onSubmittedWorkDone();
    engine._device.queue.writeBuffer(sdfBuffer, 0, new Float32Array([0.25, 1.5, 1.25, 0, 1, 0, 0, 0]));
    const coastEncoder = engine._device.createCommandEncoder();
    sim.step(coastEncoder, 1 / 60);
    engine._device.queue.submit([coastEncoder.finish()]);
    await engine._device.queue.onSubmittedWorkDone();
    const readback = engine._device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readEncoder = engine._device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, 16);
    engine._device.queue.submit([readEncoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    canvas.dataset.result = JSON.stringify(Array.from(new Float32Array(readback.getMappedRange().slice(0))));
    readback.unmap();
    readback.destroy();
    sim.dispose();
    sdfBuffer.destroy();
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./);
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const position = JSON.parse((await canvas.getAttribute("data-result"))!) as number[];
    expect(position[0]).toBeGreaterThan(0.31);
});

test("FLIP runs physical viscosity and surface tension passes", async ({ page }) => {
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
    const device = engine._device;
    device.pushErrorScope("validation");
    const sim = createFlipSim(engine, {
        count: 8,
        initialPositions: new Float32Array([
            -0.1, 0.9, -0.1, 0.1, 0.9, -0.1, -0.1, 1.1, -0.1, 0.1, 1.1, -0.1,
            -0.1, 0.9, 0.1, 0.1, 0.9, 0.1, -0.1, 1.1, 0.1, 0.1, 1.1, 0.1,
        ]),
        boundsMin: [-1, 0, -1],
        boundsMax: [1, 2, 1],
        groundY: 0,
        dx: 0.25,
        particleRadius: 0.05,
        gravity: 0,
        flipRatio: 0.5,
        pressureIterations: 4,
        kinematicViscosity: 0.4,
        viscosityIterations: 4,
        surfaceTension: 0.2,
        minSubsteps: 1,
        maxSubsteps: 4,
        cflNumber: 1,
        maxSubDt: 1 / 120,
    });
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 1 / 60);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    if (validationError) {
        throw validationError;
    }
    const readback = device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readEncoder = device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(sim.velocityBuffer, 0, readback, 0, 8 * 16);
    device.queue.submit([readEncoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const velocity = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    canvas.dataset.result = String(velocity.every(Number.isFinite));
    readback.unmap();
    readback.destroy();
    sim.dispose();
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", "true");
    await expect(canvas).not.toHaveAttribute("data-error", /./);
});

test("FLIP executes multigrid V-cycles and switches pressure solvers live", async ({ page }) => {
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
    const device = engine._device;
    device.pushErrorScope("validation");
    const sim = createFlipSim(engine, {
        count: 8,
        initialPositions: new Float32Array([
            -0.1, 0.9, -0.1, 0.1, 0.9, -0.1, -0.1, 1.1, -0.1, 0.1, 1.1, -0.1,
            -0.1, 0.9, 0.1, 0.1, 0.9, 0.1, -0.1, 1.1, 0.1, 0.1, 1.1, 0.1,
        ]),
        boundsMin: [-1, 0, -1],
        boundsMax: [1, 2, 1],
        groundY: 0,
        dx: 0.25,
        particleRadius: 0.05,
        gravity: 9.8,
        flipRatio: 0.95,
        pressureIterations: 4,
        minSubsteps: 1,
        maxSubDt: 1 / 120,
    });
    const jacobiBytes = sim.gpuBytes;
    sim.setParam("pressureSolver", 1);
    sim.setParam("multigridCycles", 2);
    const multigridBytes = sim.gpuBytes;
    for (let frame = 0; frame < 4; frame++) {
        const encoder = device.createCommandEncoder();
        sim.step(encoder, 1 / 120);
        device.queue.submit([encoder.finish()]);
    }
    sim.setParam("pressureSolver", 0);
    const jacobiEncoder = device.createCommandEncoder();
    sim.step(jacobiEncoder, 1 / 120);
    device.queue.submit([jacobiEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    if (validationError) {
        throw validationError;
    }
    const readback = device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readEncoder = device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(sim.velocityBuffer, 0, readback, 0, 8 * 16);
    device.queue.submit([readEncoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const velocity = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    canvas.dataset.result = JSON.stringify({
        finite: velocity.every(Number.isFinite),
        allocated: multigridBytes > jacobiBytes,
    });
    readback.unmap();
    readback.destroy();
    sim.dispose();
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./);
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    expect(JSON.parse((await canvas.getAttribute("data-result"))!)).toEqual({ finite: true, allocated: true });
});

test("FLIP reports pressure diagnostics and redistributes markers without changing their count", async ({ page }) => {
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
    const device = engine._device;
    device.pushErrorScope("validation");
    const particles = [];
    const cellCenter = (value, origin) => origin + (value + 0.5) * 0.25;
    const addCell = (x, y, z, count) => {
        for (let marker = 0; marker < count; marker++) {
            particles.push(
                cellCenter(x, -1) + ((marker & 1) - 0.5) * 0.02,
                cellCenter(y, 0) + (((marker >> 1) & 1) - 0.5) * 0.02,
                cellCenter(z, -1) + (((marker >> 2) & 1) - 0.5) * 0.02);
        }
    };
    addCell(4, 4, 4, 1);
    addCell(3, 4, 4, 1);
    addCell(4, 3, 4, 1);
    addCell(4, 5, 4, 1);
    addCell(4, 4, 3, 1);
    addCell(4, 4, 5, 1);
    addCell(5, 4, 4, 16);
    const initialPositions = new Float32Array(particles);
    const initialCount = initialPositions.length / 3;
    const sim = createFlipSim(engine, {
        count: 64,
        initialPositions,
        boundsMin: [-1, 0, -1],
        boundsMax: [1, 2, 1],
        groundY: 0,
        dx: 0.25,
        markersPerCell: 8,
        particleRadius: 0.05,
        gravity: 0,
        pressureSolver: "multigrid",
        multigridCycles: 3,
        pressureTolerance: 0.001,
        pressureDiagnostics: true,
        liquidSdf: true,
        ghostFluid: true,
        reseedParticles: true,
        reseedMinParticles: 4,
        reseedTargetParticles: 8,
        reseedMaxParticles: 12,
        reseedInterval: 1,
        minSubsteps: 1,
        maxSubDt: 1 / 120,
    });
    const sdfBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sdfBuffer, 0, new Float32Array([-10, 0, 0, 0]));
    sim.setSceneSdf({
        struct: "struct SceneSdfParams { floor: vec4<f32>, };",
        sdf: "fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return pt.y - sceneSdfParams.floor.x; }",
        buffer: sdfBuffer,
    });
    for (let frame = 0; frame < 8; frame++) {
        const encoder = device.createCommandEncoder();
        sim.step(encoder, 1 / 120);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const allocatedBytes = sim.gpuBytes;
    const diagnostics = sim.pressureDiagnostics;
    const activeCount = sim.activeCount;
    const readback = device.createBuffer({ size: sim.count * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readEncoder = device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, sim.count * 16);
    device.queue.submit([readEncoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const finalPositions = new Float32Array(readback.getMappedRange().slice(0));
    let targetCellCount = 0;
    for (let marker = 0; marker < sim.count; marker++) {
        const offset = marker * 4;
        const x = Math.floor((finalPositions[offset] + 1) / 0.25);
        const y = Math.floor(finalPositions[offset + 1] / 0.25);
        const z = Math.floor((finalPositions[offset + 2] + 1) / 0.25);
        if (x === 4 && y === 4 && z === 4) targetCellCount++;
    }
    readback.unmap();
    readback.destroy();
    sim.setParam("reseedParticles", 0);
    const reseedReleasedBytes = sim.gpuBytes;
    sim.setParam("pressureTolerance", 0);
    sim.setParam("pressureDiagnostics", 0);
    const diagnosticReleasedBytes = sim.gpuBytes;
    const validationError = await device.popErrorScope();
    if (validationError) {
        throw validationError;
    }
    canvas.dataset.result = JSON.stringify({
        activeCount,
        initialCount,
        targetCellCount,
        renderCount: sim.renderCount,
        finiteDiagnostics:
            !!diagnostics &&
            Number.isFinite(diagnostics.relativeResidual) &&
            Number.isFinite(diagnostics.maxDivergence) &&
            diagnostics.fluidCellCount > 0,
        reseedReleased: reseedReleasedBytes < allocatedBytes,
        diagnosticReleased: diagnosticReleasedBytes < reseedReleasedBytes,
    });
    sim.dispose();
    sdfBuffer.destroy();
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./);
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        activeCount: number;
        initialCount: number;
        targetCellCount: number;
        renderCount: number;
        finiteDiagnostics: boolean;
        reseedReleased: boolean;
        diagnosticReleased: boolean;
    };
    expect(result.activeCount).toBe(result.initialCount);
    expect(result.targetCellCount).toBeGreaterThanOrEqual(4);
    expect(result.renderCount).toBe(64);
    expect(result.finiteDiagnostics).toBe(true);
    expect(result.reseedReleased).toBe(true);
    expect(result.diagnosticReleased).toBe(true);
});

test("FLIP runs all opt-in subcell geometry paths", async ({ page }) => {
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
    const device = engine._device;
    device.pushErrorScope("validation");
    const sdfBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sdfBuffer, 0, new Float32Array([0, 0.75, 0, 0.35]));
    const sim = createFlipSim(engine, {
        count: 8,
        initialPositions: new Float32Array([
            -0.1, 0.4, -0.1, 0.1, 0.4, -0.1, -0.1, 0.6, -0.1, 0.1, 0.6, -0.1,
            -0.1, 0.4, 0.1, 0.1, 0.4, 0.1, -0.1, 0.6, 0.1, 0.1, 0.6, 0.1,
        ]),
        boundsMin: [-1, 0, -1],
        boundsMax: [1, 2, 1],
        groundY: 0,
        dx: 0.25,
        particleRadius: 0.05,
        gravity: 1,
        pressureSolver: "multigrid",
        multigridCycles: 1,
        liquidSdf: true,
        ghostFluid: true,
        fractionalSolids: true,
        movingSolidBoundaries: true,
        minSubsteps: 1,
        maxSubsteps: 1,
        maxSubDt: 1 / 120,
    });
    sim.setSceneSdf({
        struct: "struct SceneSdfParams { sphere: vec4<f32>, };",
        sdf: \`fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
            let center = sceneSdfParams.sphere.xyz + vec3<f32>(dt * 0.1, 0.0, 0.0);
            return length(pt - center) - sceneSdfParams.sphere.w;
        }\`,
        buffer: sdfBuffer,
    });
    const enabledBytes = sim.gpuBytes;
    const encoder = device.createCommandEncoder();
    sim.step(encoder, 1 / 120);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    sim.setParam("movingSolidBoundaries", 0);
    sim.setParam("fractionalSolids", 0);
    sim.setParam("ghostFluid", 0);
    sim.setParam("liquidSdf", 0);
    const disabledBytes = sim.gpuBytes;
    const legacyEncoder = device.createCommandEncoder();
    sim.step(legacyEncoder, 1 / 120);
    device.queue.submit([legacyEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    if (validationError) {
        throw validationError;
    }
    const readback = device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readEncoder = device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(sim.velocityBuffer, 0, readback, 0, 8 * 16);
    device.queue.submit([readEncoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const velocity = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    canvas.dataset.result = JSON.stringify({ finite: velocity.every(Number.isFinite), memoryReleased: disabledBytes < enabledBytes });
    readback.unmap();
    readback.destroy();
    sim.dispose();
    sdfBuffer.destroy();
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./);
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    expect(JSON.parse((await canvas.getAttribute("data-result"))!)).toEqual({ finite: true, memoryReleased: true });
});

test("FLIP exposes only the contiguous seeded prefix to renderers", async ({ page }) => {
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
    const sim = createFlipSim(engine, {
        count: 64,
        boundsMin: [-1, 0, -1],
        boundsMax: [1, 2, 1],
        groundY: 0,
        dx: 0.25,
        markersPerCell: 8,
    });
    sim.setFlow({
        emitters: [{
            id: "initial",
            name: "Initial",
            enabled: true,
            behavior: "initial",
            transform: { position: [0, 1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [2, 4, 2] },
            sampling: "volume",
            velocity: [0, 0, 0],
            velocitySpace: "world",
            spread: 0,
        }],
        sinks: [],
    });
    sim.reset();
    canvas.dataset.result = JSON.stringify({ count: sim.count, activeCount: sim.activeCount, renderCount: sim.renderCount });
    sim.dispose();
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./);
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as { count: number; activeCount: number; renderCount: number };
    expect(result.renderCount).toBe(result.activeCount);
    expect(result.renderCount).toBeGreaterThan(0);
    expect(result.renderCount).toBeLessThan(result.count);
});

test("FLIP preserves occupied volume under a rotating paddle", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

function lattice() {
    const positions = [];
    for (let y = 0.0625; y < 2; y += 0.125) {
        for (let z = -1.4375; z <= 1.4375; z += 0.125) {
            for (let x = -1.4375; x <= 1.4375; x += 0.125) {
                positions.push(x, y, z);
            }
        }
    }
    return new Float32Array(positions);
}

async function measure(engine, sim) {
    const bytes = sim.count * 16;
    const readback = engine._device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = engine._device.createCommandEncoder();
    encoder.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, bytes);
    engine._device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const positions = new Float32Array(readback.getMappedRange().slice(0));
    const ys = [];
    const occupied = new Set();
    for (let i = 0; i < sim.count; i++) {
        const x = positions[i * 4];
        const y = positions[i * 4 + 1];
        const z = positions[i * 4 + 2];
        if (!Number.isFinite(y) || y < 0) continue;
        ys.push(y);
        occupied.add(\`\${Math.floor((x + 2) / 0.25)},\${Math.floor(y / 0.25)},\${Math.floor((z + 2) / 0.25)}\`);
    }
    ys.sort((a, b) => a - b);
    const percentile = (p) => ys[Math.min(ys.length - 1, Math.floor(ys.length * p))];
    readback.unmap();
    readback.destroy();
    return { occupied: occupied.size, height: percentile(0.99) - percentile(0.01), top: percentile(0.99) };
}

async function main() {
    const engine = await createEngine(canvas);
    const initialPositions = lattice();
    const sdfBuffer = engine._device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const sim = createFlipSim(engine, {
        count: initialPositions.length / 3,
        initialPositions,
        boundsMin: [-2, 0, -2],
        boundsMax: [2, 4, 2],
        groundY: 0,
        dx: 0.25,
        markersPerCell: 8,
        particleRadius: 0.05,
        gravity: 9.8,
        flipRatio: 0.95,
        pressureIterations: 40,
        pressureRelaxation: 0.8,
        liquidSdf: true,
        ghostFluid: true,
        fractionalSolids: true,
        minSubsteps: 1,
        maxSubDt: 1 / 60,
    });
    sim.setSceneSdf({
        struct: "struct SceneSdfParams { obstacle: vec4<f32>, motion: vec4<f32>, };",
        sdf: \`fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let cd = cos(sceneSdfParams.motion.z * dt);
    let sd = sin(sceneSdfParams.motion.z * dt);
    let c = sceneSdfParams.motion.x * cd + sceneSdfParams.motion.y * sd;
    let s = sceneSdfParams.motion.y * cd - sceneSdfParams.motion.x * sd;
    let lx = c * pt.x + s * pt.z;
    let lz = -s * pt.x + c * pt.z;
    let q = vec3<f32>(
        abs(lx) - sceneSdfParams.obstacle.x,
        abs(pt.y - sceneSdfParams.obstacle.y) - sceneSdfParams.obstacle.y,
        abs(lz) - sceneSdfParams.obstacle.z);
    return select(1.0e30, length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0), sceneSdfParams.motion.w > 0.5);
}\`,
        buffer: sdfBuffer,
    });
    let angle = 0;
    const run = async (frames, paddle) => {
        for (let frame = 0; frame < frames; frame++) {
            if (paddle) angle += 2 * (1 / 60);
            engine._device.queue.writeBuffer(sdfBuffer, 0, new Float32Array([0.15, 1, 1.25, 0, Math.cos(angle), -Math.sin(angle), paddle ? 2 : 0, paddle ? 1 : 0]));
            const encoder = engine._device.createCommandEncoder();
            sim.step(encoder, 1 / 60);
            engine._device.queue.submit([encoder.finish()]);
            if (frame % 30 === 29) await engine._device.queue.onSubmittedWorkDone();
        }
        await engine._device.queue.onSubmittedWorkDone();
    };
    const initial = await measure(engine, sim);
    await run(300, false);
    const settled = await measure(engine, sim);
    await run(600, true);
    const paddled = await measure(engine, sim);
    canvas.dataset.result = JSON.stringify({ initial, settled, paddled });
    sim.dispose();
    sdfBuffer.destroy();
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 150_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        settled: { occupied: number; height: number; top: number };
        paddled: { occupied: number; height: number; top: number };
    };
    expect(result.paddled.occupied).toBeGreaterThanOrEqual(Math.floor(result.settled.occupied * 0.9) - 1);
    expect(result.paddled.height).toBeGreaterThan(result.settled.height * 0.82);
    expect(result.paddled.top).toBeGreaterThan(result.settled.top * 0.82);
});

test("FLIP drains particles from a wall aligned with solid-cell centres", async ({ page }) => {
    test.setTimeout(180_000);
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
    const dx = 0.3125;
    const wall = 4.21875;
    const sim = createFlipSim(engine, {
        count: 47_186,
        boundsMin: [-5, -0.5, -5],
        boundsMax: [5, 5.5, 5],
        groundY: -0.5,
        dx,
        markersPerCell: 8,
        particleRadius: 0.1125,
        gravity: 43.3,
        flipRatio: 0.95,
        pressureIterations: 40,
        pressureRelaxation: 0.8,
        minSubsteps: 2,
        maxSubDt: 0.0084,
    });
    sim.setFlow({
        emitters: [{
            id: "initial",
            name: "Initial",
            enabled: true,
            behavior: "initial",
            transform: { position: [0, 2, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [8, 3.75, 6] },
            sampling: "volume",
            velocity: [0, 0, 0],
            velocitySpace: "world",
            spread: 0,
        }],
        sinks: [],
    });
    const sdfBuffer = engine._device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    engine._device.queue.writeBuffer(sdfBuffer, 0, new Float32Array([-wall, 0, -wall, 0, wall, 5, wall, 0]));
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
    for (let frame = 0; frame < 600; frame++) {
        const encoder = engine._device.createCommandEncoder();
        sim.step(encoder, 1 / 60);
        engine._device.queue.submit([encoder.finish()]);
        if (frame % 30 === 29) await engine._device.queue.onSubmittedWorkDone();
    }
    await engine._device.queue.onSubmittedWorkDone();
    const bytes = sim.count * 16;
    const readback = engine._device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = engine._device.createCommandEncoder();
    encoder.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, bytes);
    engine._device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const positions = new Float32Array(readback.getMappedRange().slice(0));
    const wallY = [];
    const interiorY = [];
    let wallBridgeCount = 0;
    let maxWallCoordinate = 0;
    for (let index = 0; index < sim.renderCount; index++) {
        const x = positions[index * 4];
        const y = positions[index * 4 + 1];
        const z = positions[index * 4 + 2];
        maxWallCoordinate = Math.max(maxWallCoordinate, Math.abs(x), Math.abs(z));
        if (Math.max(Math.abs(x), Math.abs(z)) > wall - dx * 0.75) wallY.push(y);
        if (Math.max(Math.abs(x), Math.abs(z)) < wall - dx * 3) interiorY.push(y);
        if (y < 1.5 && Math.abs(z) < wall - dx * 3 && Math.abs(x) > wall - dx * 2.5 && Math.abs(x) < wall - dx * 0.75) wallBridgeCount++;
    }
    wallY.sort((a, b) => a - b);
    interiorY.sort((a, b) => a - b);
    const p99 = (values) => values[Math.min(values.length - 1, Math.floor(values.length * 0.99))];
    canvas.dataset.result = JSON.stringify({
        wallTop: p99(wallY),
        interiorTop: p99(interiorY),
        wallCount: wallY.length,
        interiorCount: interiorY.length,
        wallBridgeCount,
        maxWallCoordinate,
    });
    readback.unmap();
    readback.destroy();
    sim.dispose();
    sdfBuffer.destroy();
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 150_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        wallTop?: number;
        interiorTop: number;
        wallCount: number;
        interiorCount: number;
        wallBridgeCount: number;
        maxWallCoordinate: number;
    };
    expect(result.interiorCount).toBeGreaterThan(0);
    expect(result.wallBridgeCount).toBeGreaterThan(100);
    expect(result.maxWallCoordinate).toBeLessThanOrEqual(4.21875);
    if (result.wallTop !== undefined) {
        expect(result.wallTop).toBeLessThan(result.interiorTop + 0.125);
    }
});
