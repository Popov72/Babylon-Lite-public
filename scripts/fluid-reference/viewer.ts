import {
    addTask,
    addToScene,
    createArcRotateCamera,
    createHemisphericLight,
    createMeshFromData,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createStandardMaterial,
    registerScene,
    renderFrame,
    updateMeshGeometry,
} from "babylon-lite";
import type { EngineContext, Mesh } from "babylon-lite";
import { computeSmoothNormals } from "../../packages/babylon-lite/src/loader-gltf/gltf-normals";
import { createParticleRenderTask } from "../../packages/babylon-lite/src/fluid/rendering/particle-render";
import type { ParticleState, ReferenceCase } from "./types";

function transformReferencePositions(source: Float32Array, matrix: readonly number[]): Float32Array {
    const positions = new Float32Array(source.length);
    for (let i = 0; i < source.length; i += 3) {
        const x = source[i]!;
        const y = source[i + 1]!;
        const z = source[i + 2]!;
        for (let a = 0; a < 3; a++) {
            positions[i + a] = matrix[a]! * x + matrix[4 + a]! * y + matrix[8 + a]! * z + matrix[12 + a]!;
        }
    }
    return positions;
}

export async function createComparisonViewer(engine: EngineContext, input: ReferenceCase, initialCapacity: number) {
    const scene = createSceneContext(engine, { defaultRenderTask: false });
    scene.clearColor = { r: 0.035, g: 0.045, b: 0.065, a: 1 };
    const c = input.camera;
    const camera = createArcRotateCamera(c.alpha, c.beta, c.radius, { x: c.target[0], y: c.target[1], z: c.target[2] });
    camera.fov = c.fov;
    scene.camera = camera;
    if (input.camera.mirrorX && engine.canvas instanceof HTMLCanvasElement) {
        engine.canvas.style.transform = "scaleX(-1)";
    }
    addToScene(scene, createHemisphericLight([0, 1, 0], 1.2));
    const depth = createRenderTarget({ lbl: "flip-reference-depth", dFormat: "depth24plus", samples: 1, size: engine });
    addTask(scene, createRenderTask({ name: "flip-reference-background", rt: engine.scRT, depth, clr: true }, engine, scene));
    const movingMeshes: Array<{ mesh: Mesh; source: Float32Array; indices: Uint32Array; obstacle: number }> = [];
    const load = async (file: string): Promise<ArrayBuffer> => {
        const response = await fetch(file);
        if (!response.ok) {
            throw new Error(`Cannot load debug mesh ${file}: ${response.status}.`);
        }
        return response.arrayBuffer();
    };
    for (const definition of input.meshes) {
        const source = new Float32Array(await load(definition.positions));
        const indices = new Uint32Array(await load(definition.indices));
        const positions = transformReferencePositions(source, definition.transform);
        const mesh = createMeshFromData(engine, definition.name, positions, computeSmoothNormals(positions, indices, positions.length / 3), indices);
        const material = createStandardMaterial();
        material.diffuseColor = definition.color;
        material.specularColor = [0.08, 0.08, 0.08];
        mesh.material = material;
        addToScene(scene, mesh);
        if (definition.obstacle !== undefined) {
            movingMeshes.push({ mesh, source, indices, obstacle: definition.obstacle });
        }
    }
    let capacity = initialCapacity;
    const makeBuffer = (bytes: number): GPUBuffer => engine._device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const source = {
        count: 0,
        particleRadius: input.grid.cellSize * 0.13,
        positionBuffer: makeBuffer(capacity * 16),
        debugBuffer: makeBuffer(capacity * 4),
        debugNorm: 0.25,
    };
    const particles = createParticleRenderTask(engine, scene, { colorRT: engine.scRT, depthRT: depth, camera, sim: source });
    particles.setTint([0.16, 0.47, 0.87]);
    particles.setVelocityBrighten(0.4);
    addTask(scene, particles);
    await registerScene(scene);
    let displayedStep = -1;
    return {
        async show(state: ParticleState, step: number): Promise<void> {
            const count = state.positions.length / 3;
            if (count > capacity) {
                await engine._device.queue.onSubmittedWorkDone();
                source.positionBuffer.destroy();
                source.debugBuffer.destroy();
                capacity = count;
                source.positionBuffer = makeBuffer(capacity * 16);
                source.debugBuffer = makeBuffer(capacity * 4);
                particles.setSim(source);
            }
            const positions = new Float32Array(count * 4);
            const speeds = new Float32Array(count);
            for (let p = 0; p < count; p++) {
                positions[p * 4] = state.positions[p * 3]!;
                positions[p * 4 + 1] = state.positions[p * 3 + 1]!;
                positions[p * 4 + 2] = state.positions[p * 3 + 2]!;
                positions[p * 4 + 3] = 1;
                speeds[p] = Math.hypot(state.velocities[p * 3]!, state.velocities[p * 3 + 1]!, state.velocities[p * 3 + 2]!);
            }
            source.count = count;
            engine._device.queue.writeBuffer(source.positionBuffer, 0, positions);
            engine._device.queue.writeBuffer(source.debugBuffer, 0, speeds);
            if (displayedStep !== step) {
                for (const entry of movingMeshes) {
                    const transform = input.obstacles[entry.obstacle]!.transforms[step]!;
                    const transformed = transformReferencePositions(entry.source, transform);
                    updateMeshGeometry(engine, entry.mesh, transformed, computeSmoothNormals(transformed, entry.indices, transformed.length / 3), entry.indices);
                }
                displayedStep = step;
            }
            renderFrame(engine, 0);
            await engine._device.queue.onSubmittedWorkDone();
        },
        dispose(): void {
            source.positionBuffer.destroy();
            source.debugBuffer.destroy();
        },
    };
}
