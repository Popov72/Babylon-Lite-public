// GPU fluid simulation — Phase 1: particles + gravity + ground bounce.
//
// Demo-local (not babylon-lite core). Builds its own WebGPU compute pipeline
// from `engine._device` and encodes a compute pass into the frame's command
// encoder each frame (see the demo's onBeforeRender → sim.step). This is the
// scaffold the later phases extend: spatial-hash neighbour search (Phase 2),
// PBF density solver (Phase 3), capsule + ground SDF boundaries (Phase 4), and
// the LMB hole (Phase 5).
//
// Particle state lives entirely on the GPU:
//   • positionBuffer — vec4<f32> per particle (xyz + w spare), STORAGE so the
//     compute pass writes it and the vertex shader reads it for rendering.
//   • velocityBuffer — vec4<f32> per particle.
// Nothing is read back to the CPU.

import type { EngineContext } from "babylon-lite";

export interface FluidSimOptions {
    /** Particle count. Default 30000. */
    count?: number;
    /** Render/visual particle radius in world units. Default 0.08. */
    particleRadius?: number;
    /** Axis-aligned spawn box min the particles are seeded into. */
    spawnMin?: [number, number, number];
    /** Axis-aligned spawn box max the particles are seeded into. */
    spawnMax?: [number, number, number];
    /** Gravity acceleration (m/s²). Default 9.8. */
    gravity?: number;
    /** Ground plane height; particles bounce off `y = groundY`. Default 0. */
    groundY?: number;
    /** Restitution of the ground bounce (0 = stick, 1 = perfectly elastic). Default 0.3. */
    restitution?: number;
}

export interface FluidSim {
    readonly count: number;
    readonly particleRadius: number;
    /** vec4<f32>-per-particle position buffer (STORAGE). Read by the renderer. */
    readonly positionBuffer: GPUBuffer;
    /** Encode one simulation step into `encoder`. `dt` is seconds. */
    step(encoder: GPUCommandEncoder, dt: number): void;
    /** Re-seed all particles into the spawn box with zero velocity. */
    reset(): void;
    dispose(): void;
}

const WORKGROUP_SIZE = 64;

const INTEGRATE_WGSL = /* wgsl */ `
struct Params {
    dt: f32,
    gravity: f32,
    groundY: f32,
    restitution: f32,
    count: u32,
    _p0: u32, _p1: u32, _p2: u32,
};
@group(0) @binding(0) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    var p = pos[i].xyz;
    var v = vel[i].xyz;
    v.y -= params.gravity * params.dt;
    p += v * params.dt;
    if (p.y < params.groundY) {
        p.y = params.groundY;
        v.y = -v.y * params.restitution;
        // Tangential friction so particles settle instead of sliding forever.
        v.x *= 0.92;
        v.z *= 0.92;
    }
    pos[i] = vec4<f32>(p, 1.0);
    vel[i] = vec4<f32>(v, 0.0);
}`;

export function createFluidSim(engine: EngineContext, options: FluidSimOptions = {}): FluidSim {
    const device = engine._device;
    const count = options.count ?? 30000;
    const particleRadius = options.particleRadius ?? 0.08;
    const spawnMin = options.spawnMin ?? [-2, 6, -2];
    const spawnMax = options.spawnMax ?? [2, 12, 2];
    const gravity = options.gravity ?? 9.8;
    const groundY = options.groundY ?? 0;
    const restitution = options.restitution ?? 0.3;

    const positionBuffer = device.createBuffer({
        label: "fluid-positions",
        size: count * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const velocityBuffer = device.createBuffer({
        label: "fluid-velocities",
        size: count * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const paramsBuffer = device.createBuffer({
        label: "fluid-params",
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const paramsData = new ArrayBuffer(32);
    const paramsF32 = new Float32Array(paramsData);
    const paramsU32 = new Uint32Array(paramsData);

    function seed(): void {
        const positions = new Float32Array(count * 4);
        const velocities = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
            const o = i * 4;
            positions[o] = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
            positions[o + 1] = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
            positions[o + 2] = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
            positions[o + 3] = 1;
            // velocities default to 0
        }
        device.queue.writeBuffer(positionBuffer, 0, positions);
        device.queue.writeBuffer(velocityBuffer, 0, velocities);
    }
    seed();

    const module = device.createShaderModule({ label: "fluid-integrate", code: INTEGRATE_WGSL });
    const pipeline = device.createComputePipeline({
        label: "fluid-integrate",
        layout: "auto",
        compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
        label: "fluid-integrate",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ],
    });

    const workgroups = Math.ceil(count / WORKGROUP_SIZE);

    return {
        count,
        particleRadius,
        positionBuffer,
        step(encoder: GPUCommandEncoder, dt: number): void {
            paramsF32[0] = dt;
            paramsF32[1] = gravity;
            paramsF32[2] = groundY;
            paramsF32[3] = restitution;
            paramsU32[4] = count;
            device.queue.writeBuffer(paramsBuffer, 0, paramsData);
            const pass = encoder.beginComputePass({ label: "fluid-integrate" });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
        },
        reset(): void {
            seed();
        },
        dispose(): void {
            positionBuffer.destroy();
            velocityBuffer.destroy();
            paramsBuffer.destroy();
        },
    };
}
