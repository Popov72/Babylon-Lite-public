import {
    addToScene,
    createFluidCompositeSceneSdf,
    createFluidDeformingSceneSdf,
    createLineMaterial,
    createLineSystem,
    createPbrMaterial,
    goToFrame,
    loadGltf,
    mat4Decompose,
    mat4Invert,
    mat4Multiply,
    pauseAnimation,
    playAnimation,
    setMeshVisible,
    updateFluidSceneSdfGridSettings,
    updateFluidSceneSdfTransforms,
    updateFluidDeformingSceneSdf,
    updateFluidDeformingSceneSdfContainer,
    updateFluidSceneSdfContainer,
    updateLineSystem,
} from "babylon-lite";
import type {
    AnimationGroup,
    FluidDeformingSceneSdf,
    FluidFlowConfig,
    FluidLocalSdfData,
    FluidSceneSdf,
    FluidSceneSdfBounds,
    Mat4,
    Mesh,
    SceneNode,
    SceneSdfSpec,
    Vec3,
} from "babylon-lite";

import { demoAssetUrl } from "../../demo-asset-url.js";
import type { FluidCtx, FluidDemo } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";

const WHALE_URL = demoAssetUrl("./jumping-whale/whale.glb", import.meta.url);
const WHALE_SDF_URL = demoAssetUrl("./jumping-whale/whale-sdf.bin", import.meta.url);
// The source breach starts near its authored waterline. Offset it so the bind/start pose is
// fully submerged in the demo's y=0..7 initial ocean, while the animated apex still clears it.
const WHALE_BASE_Y = 6.5;
const DEFORMING_DEBUG_MAX_CROSSINGS = 2_500;

interface ProxyDefinition {
    readonly id: string;
    readonly center: readonly [number, number, number];
    readonly size: readonly [number, number, number];
}

interface SkeletonBindingView {
    readonly inverseBindMatrices: Float32Array;
    readonly boneMatrices: Float32Array;
    readonly boneCount: number;
    readonly runtimeSkeleton?: object;
}

type WhaleAnimationGroup = AnimationGroup & {
    readonly _gltfMixer?: readonly [unknown, unknown, readonly SkeletonBindingView[]];
};

interface WhaleSdfSequence {
    readonly frameCount: number;
    readonly frameRate: number;
    readonly dims: readonly [number, number, number];
    readonly origin: readonly [number, number, number];
    readonly cellSize: number;
    readonly wordsPerFrame: number;
    readonly frames: Uint32Array;
}

// Joint-bind-space bounds derived directly from the raw glTF POSITION / JOINTS_0 /
// WEIGHTS_0 data and the skin's authoritative inverseBindMatrices accessor.
const PROXIES: readonly ProxyDefinition[] = [
    { id: "Spine", center: [0.01127, 1.81265, 0], size: [3.48728, 4.06627, 2.85586] },
    { id: "Fin.L", center: [-1.45, 3.03094, -0.03326], size: [4.7, 4.26719, 0.29268] },
    { id: "Fin.R", center: [-1.45, 3.03094, 0.03326], size: [4.7, 4.2672, 0.29268] },
    { id: "Head", center: [-0.01943, 2.61922, -0.00337], size: [3.61655, 4.42157, 2.81174] },
    { id: "TailBase", center: [-0.00989, 1.60723, 0.43662], size: [2.72261, 3.38554, 3.22675] },
    { id: "TailTip", center: [0, 0.4009, 0], size: [6.3, 3.00181, 0.99566] },
];

function matrixAt(data: Float32Array, index: number): Mat4 {
    const matrix = new Float32Array(16);
    matrix.set(data.subarray(index * 16, index * 16 + 16));
    return matrix as unknown as Mat4;
}

function ellipsoidDistance(x: number, y: number, z: number, center: readonly [number, number, number], radii: readonly [number, number, number]): number {
    const px = x - center[0];
    const py = y - center[1];
    const pz = z - center[2];
    const qx = px / radii[0];
    const qy = py / radii[1];
    const qz = pz / radii[2];
    const k0 = Math.sqrt(qx * qx + qy * qy + qz * qz);
    const k1 = Math.sqrt((px / (radii[0] * radii[0])) ** 2 + (py / (radii[1] * radii[1])) ** 2 + (pz / (radii[2] * radii[2])) ** 2);
    return k1 > 1e-6 ? (k0 * (k0 - 1)) / k1 : -Math.min(...radii);
}

function proxyRadii(proxy: ProxyDefinition): [number, number, number] {
    return proxy.size.map((value) => Math.max(0.08, value * 0.46)) as [number, number, number];
}

function proxyGrid(proxy: ProxyDefinition): FluidLocalSdfData {
    const radii = proxyRadii(proxy);
    const cellSize = Math.max(...proxy.size) / 20;
    const padding = cellSize * 2;
    const origin = proxy.center.map((value, axis) => value - radii[axis]! - padding) as [number, number, number];
    const dims = radii.map((radius) => Math.ceil((radius * 2 + padding * 2) / cellSize) + 1) as [number, number, number];
    const distances = new Float32Array(dims[0] * dims[1] * dims[2]);
    let offset = 0;
    for (let z = 0; z < dims[2]; z++) {
        for (let y = 0; y < dims[1]; y++) {
            for (let x = 0; x < dims[0]; x++) {
                distances[offset++] = ellipsoidDistance(origin[0] + x * cellSize, origin[1] + y * cellSize, origin[2] + z * cellSize, proxy.center, radii);
            }
        }
    }
    return {
        id: proxy.id,
        dims,
        origin,
        cellSize,
        distances,
        enabled: false,
        trilinear: true,
    };
}

function proxyWireLines(proxy: ProxyDefinition): Vec3[][] {
    const radii = proxyRadii(proxy);
    const lines: Vec3[][] = [];
    const segments = 32;
    for (let axis = 0; axis < 3; axis++) {
        const line: Vec3[] = [];
        for (let step = 0; step <= segments; step++) {
            const angle = (step / segments) * Math.PI * 2;
            const point: [number, number, number] = [...proxy.center];
            const a = (axis + 1) % 3;
            const b = (axis + 2) % 3;
            point[a] = point[a]! + Math.cos(angle) * radii[a]!;
            point[b] = point[b]! + Math.sin(angle) * radii[b]!;
            line.push({ x: point[0], y: point[1], z: point[2] });
        }
        lines.push(line);
    }
    return lines;
}

function collectMeshes(node: SceneNode, meshes: Mesh[]): void {
    if ("_gpu" in node) {
        meshes.push(node as Mesh);
    }
    for (const child of node.children) {
        collectMeshes(child, meshes);
    }
}

function float16ToFloat32(bits: number): number {
    const sign = (bits & 0x8000) !== 0 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) {
        return sign * 2 ** -14 * (mantissa / 1024);
    }
    if (exponent === 0x1f) {
        return mantissa === 0 ? sign * Infinity : Number.NaN;
    }
    return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

async function loadWhaleSdfSequence(): Promise<WhaleSdfSequence> {
    const response = await fetch(WHALE_SDF_URL);
    if (!response.ok) {
        throw new Error(`Jumping whale deforming SDF failed to load (${response.status}).`);
    }
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength < 64 || new TextDecoder().decode(new Uint8Array(buffer, 0, 7)) !== "BLWSDF1") {
        throw new Error("Jumping whale deforming SDF has an invalid header.");
    }
    const view = new DataView(buffer);
    const frameCount = view.getUint32(8, true);
    const frameRate = view.getUint32(12, true);
    const dims: [number, number, number] = [view.getUint32(16, true), view.getUint32(20, true), view.getUint32(24, true)];
    const origin: [number, number, number] = [view.getFloat32(28, true), view.getFloat32(32, true), view.getFloat32(36, true)];
    const cellSize = view.getFloat32(40, true);
    const wordsPerFrame = view.getUint32(44, true);
    const expectedWords = Math.ceil((dims[0] * dims[1] * dims[2]) / 2);
    if (
        frameCount < 2 ||
        frameRate < 1 ||
        wordsPerFrame !== expectedWords ||
        !origin.every(Number.isFinite) ||
        !Number.isFinite(cellSize) ||
        cellSize <= 0 ||
        buffer.byteLength !== 64 + frameCount * wordsPerFrame * 4
    ) {
        throw new Error("Jumping whale deforming SDF metadata is invalid.");
    }
    return {
        frameCount,
        frameRate,
        dims,
        origin,
        cellSize,
        wordsPerFrame,
        frames: new Uint32Array(buffer, 64, frameCount * wordsPerFrame),
    };
}

export async function createJumpingWhaleDemo(ctx: FluidCtx): Promise<FluidDemo> {
    const collision: FluidSceneSdf = createFluidCompositeSceneSdf(ctx.engine, {
        localSdfs: PROXIES.map(proxyGrid),
        gridConfine: true,
        movingBoundaries: true,
    });
    const fallbackSdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { unused: vec4<f32>, };",
        sdf: "fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return 1.0e30 + sceneSdfParams.unused.x; }",
        buffer: ctx.sceneSdfBuffer,
        gridConfine: false,
    };
    const status = document.createElement("div");
    status.style.cssText = "color:#9fb4cc;font-size:11px;margin:0 0 6px;";
    status.textContent = "Loading whale and bone collision proxies…";
    const whaleMaterial = createPbrMaterial({
        baseColorFactor: [0.018, 0.028, 0.035, 1],
        metallicFactor: 0,
        roughnessFactor: 0.34,
        environmentIntensity: 0.8,
        directIntensity: 1,
    });
    const proxyMaterial = createLineMaterial({
        name: "jumping-whale-proxy-material",
        color: { r: 0.1, g: 1, b: 0.85, a: 0.9 },
        useVertexAlpha: true,
        depthWrite: false,
        depthCompare: "always",
    });
    const proxyDebugMeshes = PROXIES.map((proxy) => {
        const mesh = createLineSystem(ctx.engine, {
            name: `jumping-whale-proxy-${proxy.id}`,
            lines: proxyWireLines(proxy),
            material: proxyMaterial,
        });
        mesh.pickable = false;
        mesh.renderOrder = 10_000;
        addToScene(ctx.scene, mesh);
        setMeshVisible(mesh, false);
        return mesh;
    });
    const deformingDebugMaterial = createLineMaterial({
        name: "jumping-whale-deforming-sdf-material",
        color: { r: 1, g: 0.15, b: 0.85, a: 0.9 },
        useVertexAlpha: true,
        depthWrite: false,
        depthCompare: "always",
    });

    let active = false;
    let ready = false;
    let showCollisionShapes = false;
    let useDeformingSdf = false;
    let stabilizeInitialState = true;
    let animationSpeed = 0.65;
    let collisionVelocityScale = 1;
    let resetCollisionMotion = true;
    let whaleMeshes: Mesh[] = [];
    let animation: WhaleAnimationGroup | null = null;
    let animationHook: ((deltaMs: number) => void) | undefined;
    let whaleRoot: SceneNode | null = null;
    let whaleRootBaseY = 0;
    let whaleYOffset = -2;
    let bodyMesh: Mesh | null = null;
    let skeletonBinding: SkeletonBindingView | null = null;
    let deformingCollision: FluidDeformingSceneSdf | null = null;
    let deformingSequence: WhaleSdfSequence | null = null;
    let deformingLoad: Promise<void> | null = null;
    let mlsContainer: FluidSceneSdfBounds | null = null;
    let deformingDebugMesh: Mesh | null = null;
    let deformingDebugLines: Vec3[][] | null = null;
    let bindMatrices: Mat4[] = [];
    let previousTime = 0;
    const deformingActive = (): boolean => useDeformingSdf && deformingCollision !== null && deformingSequence !== null;
    const tickAnimation = (deltaMs: number): void => animationHook?.(Math.min(Math.max(deltaMs, 0), 1000 / 30));
    const updateStatus = (): void => {
        status.textContent = useDeformingSdf
            ? deformingSequence
                ? `Collision: exact deforming SDF (${deformingSequence.frameCount} frames at ${deformingSequence.frameRate} fps)`
                : "Loading exact deforming SDF…"
            : "Collision: 6 bone-driven ellipsoid SDF proxies";
        ctx.canvas.dataset.whaleCollisionMode = deformingActive() ? "deforming-sdf" : "bone-proxies";
    };
    const ensureDeformingCollision = (): Promise<void> => {
        if (deformingCollision && deformingSequence) {
            return Promise.resolve();
        }
        if (!deformingLoad) {
            deformingLoad = loadWhaleSdfSequence()
                .then((sequence) => {
                    deformingSequence = sequence;
                    deformingCollision = createFluidDeformingSceneSdf(ctx.engine, {
                        dims: sequence.dims,
                        origin: sequence.origin,
                        cellSize: sequence.cellSize,
                        initialFrame: sequence.frames.subarray(0, sequence.wordsPerFrame),
                        trilinear: true,
                        gridConfine: true,
                        movingBoundaries: true,
                    });
                    updateFluidDeformingSceneSdfContainer(deformingCollision, mlsContainer);
                })
                .catch((error) => {
                    deformingLoad = null;
                    throw error;
                });
        }
        return deformingLoad;
    };
    const activateDeformingCollision = (): void => {
        updateStatus();
        void ensureDeformingCollision()
            .then(() => {
                if (!useDeformingSdf) {
                    return;
                }
                resetCollisionMotion = true;
                updateCollision(0, true);
                if (active) {
                    ctx.rebindSceneSdf();
                }
                updateStatus();
            })
            .catch((error: unknown) => {
                useDeformingSdf = false;
                status.textContent = error instanceof Error ? error.message : String(error);
                console.error("[jumping-whale] failed to load deforming SDF", error);
            });
    };
    const ensureDeformingDebugMesh = (): Mesh => {
        if (deformingDebugMesh) {
            return deformingDebugMesh;
        }
        const lines: Vec3[][] = [];
        for (let crossing = 0; crossing < DEFORMING_DEBUG_MAX_CROSSINGS; crossing++) {
            lines.push(
                [
                    { x: 0, y: 0, z: 0 },
                    { x: 0, y: 0, z: 0 },
                ],
                [
                    { x: 0, y: 0, z: 0 },
                    { x: 0, y: 0, z: 0 },
                ],
                [
                    { x: 0, y: 0, z: 0 },
                    { x: 0, y: 0, z: 0 },
                ]
            );
        }
        deformingDebugLines = lines;
        deformingDebugMesh = createLineSystem(ctx.engine, {
            name: "jumping-whale-deforming-sdf",
            lines,
            material: deformingDebugMaterial,
        });
        deformingDebugMesh.pickable = false;
        deformingDebugMesh.renderOrder = 10_000;
        addToScene(ctx.scene, deformingDebugMesh);
        setMeshVisible(deformingDebugMesh, false);
        return deformingDebugMesh;
    };
    const syncDeformingDebug = (sequence: WhaleSdfSequence, firstFrameIndex: number, secondFrameIndex: number, blend: number, localToWorld: Mat4): void => {
        if (!active || !showCollisionShapes || !deformingActive()) {
            if (deformingDebugMesh) {
                setMeshVisible(deformingDebugMesh, false);
            }
            return;
        }
        const mesh = ensureDeformingDebugMesh();
        const lines = deformingDebugLines!;
        const [dimX, dimY, dimZ] = sequence.dims;
        const stride = Math.max(1, Math.ceil(Math.max(dimX, dimY, dimZ) / 48));
        const half = sequence.cellSize * stride * 0.2;
        const firstWordOffset = firstFrameIndex * sequence.wordsPerFrame;
        const secondWordOffset = secondFrameIndex * sequence.wordsPerFrame;
        const value = (i: number, j: number, k: number): number => {
            const halfIndex = i + dimX * (j + dimY * k);
            const shift = (halfIndex & 1) * 16;
            const first = float16ToFloat32((sequence.frames[firstWordOffset + (halfIndex >>> 1)]! >>> shift) & 0xffff);
            const second = float16ToFloat32((sequence.frames[secondWordOffset + (halfIndex >>> 1)]! >>> shift) & 0xffff);
            return first + (second - first) * blend;
        };
        let crossingCount = 0;
        const appendCrossing = (ax: number, ay: number, az: number, bx: number, by: number, bz: number, va: number, vb: number): boolean => {
            if (va < 0 === vb < 0) {
                return true;
            }
            if (crossingCount >= DEFORMING_DEBUG_MAX_CROSSINGS) {
                return false;
            }
            const t = va / (va - vb);
            const x = ax + (bx - ax) * t;
            const y = ay + (by - ay) * t;
            const z = az + (bz - az) * t;
            const base = crossingCount * 3;
            lines[base]![0]!.x = x - half;
            lines[base]![0]!.y = y;
            lines[base]![0]!.z = z;
            lines[base]![1]!.x = x + half;
            lines[base]![1]!.y = y;
            lines[base]![1]!.z = z;
            lines[base + 1]![0]!.x = x;
            lines[base + 1]![0]!.y = y - half;
            lines[base + 1]![0]!.z = z;
            lines[base + 1]![1]!.x = x;
            lines[base + 1]![1]!.y = y + half;
            lines[base + 1]![1]!.z = z;
            lines[base + 2]![0]!.x = x;
            lines[base + 2]![0]!.y = y;
            lines[base + 2]![0]!.z = z - half;
            lines[base + 2]![1]!.x = x;
            lines[base + 2]![1]!.y = y;
            lines[base + 2]![1]!.z = z + half;
            crossingCount++;
            return true;
        };
        for (let k = 0; k < dimZ; k += stride) {
            const z = sequence.origin[2] + k * sequence.cellSize;
            for (let j = 0; j < dimY; j += stride) {
                const y = sequence.origin[1] + j * sequence.cellSize;
                for (let i = 0; i < dimX; i += stride) {
                    const x = sequence.origin[0] + i * sequence.cellSize;
                    const sample = value(i, j, k);
                    if (
                        (i + stride < dimX && !appendCrossing(x, y, z, x + stride * sequence.cellSize, y, z, sample, value(i + stride, j, k))) ||
                        (j + stride < dimY && !appendCrossing(x, y, z, x, y + stride * sequence.cellSize, z, sample, value(i, j + stride, k))) ||
                        (k + stride < dimZ && !appendCrossing(x, y, z, x, y, z + stride * sequence.cellSize, sample, value(i, j, k + stride)))
                    ) {
                        break;
                    }
                }
            }
        }
        const hidden = sequence.origin;
        for (let crossing = crossingCount; crossing < DEFORMING_DEBUG_MAX_CROSSINGS; crossing++) {
            const base = crossing * 3;
            for (let axis = 0; axis < 3; axis++) {
                for (let end = 0; end < 2; end++) {
                    lines[base + axis]![end]!.x = hidden[0];
                    lines[base + axis]![end]!.y = hidden[1];
                    lines[base + axis]![end]!.z = hidden[2];
                }
            }
        }
        updateLineSystem(ctx.engine, mesh, { lines });
        const transform = mat4Decompose(localToWorld);
        mesh.position.set(transform.translation.x, transform.translation.y, transform.translation.z);
        mesh.rotationQuaternion.set(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w);
        mesh.scaling.set(transform.scale.x, transform.scale.y, transform.scale.z);
        setMeshVisible(mesh, true);
        ctx.canvas.dataset.whaleSdfDebugCrossings = String(crossingCount);
    };

    const removeAnimationHook = (): void => {
        const index = ctx.scene._beforeRender.indexOf(tickAnimation);
        if (index >= 0) {
            ctx.scene._beforeRender.splice(index, 1);
        }
    };
    const restartAnimation = (paused = false): void => {
        if (!animation || !animationHook) {
            return;
        }
        removeAnimationHook();
        // The scene callback is unshifted so skinning is evaluated before Fluid's own
        // before-render callback samples the bone matrices for collision.
        ctx.scene._beforeRender.unshift(tickAnimation);
        goToFrame(animation, 0, ctx.engine);
        previousTime = 0;
        resetCollisionMotion = true;
        if (paused) {
            pauseAnimation(animation);
        } else {
            playAnimation(animation);
        }
        updateCollision(0, true);
    };
    const stopAnimation = (): void => {
        removeAnimationHook();
        if (animation) {
            pauseAnimation(animation);
        }
    };

    const flow = (): FluidFlowConfig => ({
        emitters: [
            {
                id: "whale-pool",
                name: "Initial ocean",
                enabled: true,
                behavior: "initial",
                transform: { position: [0, 3.5, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: { type: "box", size: [36, 7, 24] },
                sampling: "volume",
                velocity: [0, 0, 0],
                velocitySpace: "world",
                spread: 0,
            },
        ],
        sinks: [],
        initialEmittersFillCapacity: true,
    });

    const updateCollision = (elapsedSeconds: number, resetMotion = false): void => {
        if (!bodyMesh || !skeletonBinding) {
            return;
        }
        const transforms = PROXIES.map((proxy, index) => {
            const boneMatrix = matrixAt(skeletonBinding!.boneMatrices, index);
            return {
                id: proxy.id,
                localToWorld: mat4Multiply(mat4Multiply(bodyMesh!.worldMatrix, boneMatrix), bindMatrices[index]!) as Mat4,
            };
        });
        const staticBoundary = collisionVelocityScale <= 0;
        const shouldResetMotion = resetMotion || resetCollisionMotion;
        if (!deformingActive()) {
            const motionElapsed = staticBoundary ? 0 : elapsedSeconds / collisionVelocityScale;
            updateFluidSceneSdfTransforms(collision, transforms, motionElapsed, { resetMotion: shouldResetMotion || staticBoundary });
        }
        if (deformingActive()) {
            const exactCollision = deformingCollision!;
            const sequence = deformingSequence!;
            const sample = animation ? animation.currentTime * sequence.frameRate : 0;
            const unwrappedFrame = Math.floor(sample);
            const firstFrameIndex = ((unwrappedFrame % sequence.frameCount) + sequence.frameCount) % sequence.frameCount;
            const secondFrameIndex = (firstFrameIndex + 1) % sequence.frameCount;
            const firstOffset = firstFrameIndex * sequence.wordsPerFrame;
            const secondOffset = secondFrameIndex * sequence.wordsPerFrame;
            updateFluidDeformingSceneSdf(exactCollision, {
                firstFrameIndex,
                secondFrameIndex,
                firstFrame: sequence.frames.subarray(firstOffset, firstOffset + sequence.wordsPerFrame),
                secondFrame: sequence.frames.subarray(secondOffset, secondOffset + sequence.wordsPerFrame),
                blend: sample - unwrappedFrame,
                frameRate: sequence.frameRate * animationSpeed,
                localToWorld: transforms[0]!.localToWorld,
                elapsedSeconds,
                velocityScale: collisionVelocityScale,
                resetMotion: shouldResetMotion,
            });
            syncDeformingDebug(sequence, firstFrameIndex, secondFrameIndex, sample - unwrappedFrame, transforms[0]!.localToWorld);
        } else if (deformingDebugMesh) {
            setMeshVisible(deformingDebugMesh, false);
        }
        resetCollisionMotion = false;
        for (let index = 0; index < transforms.length; index++) {
            const transform = mat4Decompose(transforms[index]!.localToWorld);
            const mesh = proxyDebugMeshes[index]!;
            mesh.position.set(transform.translation.x, transform.translation.y, transform.translation.z);
            mesh.rotationQuaternion.set(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w);
            mesh.scaling.set(transform.scale.x, transform.scale.y, transform.scale.z);
            setMeshVisible(mesh, active && showCollisionShapes && !deformingActive());
        }
        const head = transforms[3]!.localToWorld;
        ctx.canvas.dataset.whaleHeadPosition = [head[12], head[13], head[14]].join(",");
    };
    const applyWhaleYOffset = (): void => {
        if (!whaleRoot) {
            return;
        }
        whaleRoot.position.y = whaleRootBaseY + WHALE_BASE_Y + whaleYOffset;
        if (ready) {
            updateCollision(0, true);
        }
        ctx.canvas.dataset.whaleYOffset = String(whaleYOffset);
    };

    try {
        const asset = await loadGltf(ctx.engine, WHALE_URL);
        const root = asset.entities[0] as SceneNode;
        whaleRoot = root;
        whaleRootBaseY = root.position.y;
        applyWhaleYOffset();
        collectMeshes(root, whaleMeshes);
        for (const mesh of whaleMeshes) {
            if (mesh.skeleton && !mesh.name.startsWith("Eye") && !mesh.name.startsWith("Blowhole")) {
                mesh.material = whaleMaterial;
            }
            // The source asset includes two unskinned authoring helpers at the origin.
            setMeshVisible(mesh, active && !!mesh.skeleton);
        }
        bodyMesh = whaleMeshes.find((mesh) => mesh.name === "Whale_Body") ?? whaleMeshes.find((mesh) => !!mesh.skeleton) ?? null;
        animation = (asset.animationGroups?.[0] as WhaleAnimationGroup | undefined) ?? null;
        if (!bodyMesh?.skeleton || !animation?._gltfMixer) {
            throw new Error("Jumping whale requires its skinned body and breach animation.");
        }
        skeletonBinding =
            animation._gltfMixer[2].find((binding) => binding.runtimeSkeleton === bodyMesh!.skeleton) ??
            animation._gltfMixer[2].find((binding) => binding.boneCount === PROXIES.length) ??
            null;
        if (!skeletonBinding || skeletonBinding.boneCount !== PROXIES.length) {
            throw new Error(`Jumping whale expected ${PROXIES.length} collision bones.`);
        }
        bindMatrices = PROXIES.map((_proxy, index) => {
            const inverseBind = matrixAt(skeletonBinding!.inverseBindMatrices, index);
            const bind = mat4Invert(inverseBind);
            if (!bind) {
                throw new Error(`Jumping whale bone ${index} has a singular inverse bind matrix.`);
            }
            return bind;
        });
        addToScene(ctx.scene, asset);
        animation.loopAnimation = true;
        animation.speedRatio = animationSpeed;
        animationHook = asset._beforeRenderHook;
        if (animationHook) {
            const index = ctx.scene._beforeRender.indexOf(animationHook);
            if (index >= 0) {
                ctx.scene._beforeRender.splice(index, 1);
            }
        }
        removeAnimationHook();
        if (active) {
            restartAnimation();
        } else {
            pauseAnimation(animation);
        }
        updateCollision(0, true);
        updateFluidSceneSdfGridSettings(
            collision,
            PROXIES.map((proxy) => ({ id: proxy.id, enabled: true }))
        );
        previousTime = animation.currentTime;
        ready = true;
        updateStatus();
    } catch (error) {
        status.textContent = error instanceof Error ? error.message : String(error);
        console.error("[jumping-whale] failed to initialize", error);
    }

    return {
        key: "jumpingWhale",
        label: "Jumping whale",
        envUrl: ENV_STUDIO_URL,
        envKey: "quarry",
        defaultMethod: "FLIP",
        defaultQuality: "middle",
        defaultCamera: { alpha: -Math.PI / 2, beta: 1.2, radius: 52, target: [0, 8, 0] },
        useGridFloor: true,
        sdf: fallbackSdf,
        sceneSdf: () => (deformingActive() ? deformingCollision!.sceneSdf : collision),
        setMlsContainer(bounds: FluidSceneSdfBounds | null): void {
            mlsContainer = bounds;
            updateFluidSceneSdfContainer(collision, bounds);
            if (deformingCollision) {
                updateFluidDeformingSceneSdfContainer(deformingCollision, bounds);
            }
        },
        writeSdfParams(): void {
            ctx.engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, new Float32Array(4));
        },
        flow,
        onEnter(): void {
            active = true;
            for (const mesh of whaleMeshes) {
                setMeshVisible(mesh, !!mesh.skeleton);
            }
            for (const mesh of proxyDebugMeshes) {
                setMeshVisible(mesh, showCollisionShapes && !deformingActive());
            }
            if (deformingDebugMesh) {
                setMeshVisible(deformingDebugMesh, showCollisionShapes && deformingActive());
            }
            updateStatus();
            if (useDeformingSdf && !deformingActive()) {
                activateDeformingCollision();
            }
            restartAnimation();
            setMeshVisible(ctx.ground, false);
        },
        onLeave(): void {
            active = false;
            stopAnimation();
            for (const mesh of whaleMeshes) {
                setMeshVisible(mesh, false);
            }
            for (const mesh of proxyDebugMeshes) {
                setMeshVisible(mesh, false);
            }
            if (deformingDebugMesh) {
                setMeshVisible(deformingDebugMesh, false);
            }
            setMeshVisible(ctx.ground, true);
        },
        onPauseChanged(paused: boolean): void {
            if (!active || !animation) {
                return;
            }
            if (paused) {
                pauseAnimation(animation);
            } else {
                previousTime = animation.currentTime;
                resetCollisionMotion = true;
                playAnimation(animation);
            }
        },
        restartAnimation(paused: boolean): void {
            if (active) {
                restartAnimation(paused);
            }
        },
        initialStabilizationEnabled: () => stabilizeInitialState,
        async prepareInitialStabilization(): Promise<void> {
            if (!useDeformingSdf) {
                return;
            }
            await ensureDeformingCollision();
            resetCollisionMotion = true;
            updateCollision(0, true);
            if (active) {
                ctx.rebindSceneSdf();
            }
            updateStatus();
        },
        onInitialStabilizationProgress(state): void {
            status.textContent = state.running
                ? `Stabilizing initial state… ${state.simulatedSeconds.toFixed(1)} s`
                : state.converged
                  ? `Initial state stabilized in ${state.simulatedSeconds.toFixed(1)} s`
                  : `Initial stabilization stopped after ${state.simulatedSeconds.toFixed(1)} s`;
            if (!state.running) {
                window.setTimeout(updateStatus, 1500);
            }
        },
        update(dt: number): void {
            if (!active || !ready || !animation) {
                return;
            }
            const wrapped = animation.currentTime + 1e-5 < previousTime;
            updateCollision(dt, wrapped);
            previousTime = animation.currentTime;
            ctx.canvas.dataset.whaleAnimationTime = animation.currentTime.toFixed(3);
        },
        isReady: () => ready,
        demoParams: () => [
            { key: "whaleYOffset", label: "Whale Y offset", type: "number", min: -10, max: 10, step: 0.1, value: whaleYOffset },
            { key: "animationSpeed", label: "Animation speed", type: "number", min: 0.05, max: 3, step: 0.05, value: animationSpeed },
            { key: "collisionVelocityScale", label: "Collision velocity", type: "number", min: 0, max: 1000, step: 0.1, value: collisionVelocityScale },
            { key: "stabilizeInitialState", label: "Stabilize initial state", type: "boolean", value: stabilizeInitialState },
            { key: "useDeformingSdf", label: "Exact deforming SDF", type: "boolean", value: useDeformingSdf },
            { key: "showCollisionShapes", label: "Show collision shapes", type: "boolean", value: showCollisionShapes },
        ],
        applyParam(key: string, value: number | boolean | string): void {
            if (key === "whaleYOffset" && typeof value === "number") {
                whaleYOffset = value;
                applyWhaleYOffset();
            } else if (key === "animationSpeed" && typeof value === "number") {
                animationSpeed = value;
                if (animation) {
                    animation.speedRatio = animationSpeed;
                }
            } else if (key === "collisionVelocityScale" && typeof value === "number") {
                collisionVelocityScale = value;
                resetCollisionMotion = true;
            } else if (key === "stabilizeInitialState" && typeof value === "boolean") {
                stabilizeInitialState = value;
            } else if (key === "useDeformingSdf" && typeof value === "boolean") {
                useDeformingSdf = value;
                resetCollisionMotion = true;
                if (useDeformingSdf) {
                    activateDeformingCollision();
                } else {
                    updateCollision(0, true);
                    ctx.rebindSceneSdf();
                    updateStatus();
                }
                for (const mesh of proxyDebugMeshes) {
                    setMeshVisible(mesh, active && showCollisionShapes && !deformingActive());
                }
                if (deformingDebugMesh) {
                    setMeshVisible(deformingDebugMesh, active && showCollisionShapes && deformingActive());
                }
            } else if (key === "showCollisionShapes" && typeof value === "boolean") {
                showCollisionShapes = value;
                for (const mesh of proxyDebugMeshes) {
                    setMeshVisible(mesh, active && showCollisionShapes && !deformingActive());
                }
                if (!deformingActive() && deformingDebugMesh) {
                    setMeshVisible(deformingDebugMesh, false);
                }
            }
        },
        snapshotState: () => ({ showCollisionShapes, useDeformingSdf, stabilizeInitialState }),
        restoreState(state): void {
            if (typeof state.showCollisionShapes === "boolean") {
                showCollisionShapes = state.showCollisionShapes;
            }
            if (typeof state.useDeformingSdf === "boolean") {
                useDeformingSdf = state.useDeformingSdf;
            }
            if (typeof state.stabilizeInitialState === "boolean") {
                stabilizeInitialState = state.stabilizeInitialState;
            }
        },
        commitRestoredParams(): void {
            resetCollisionMotion = true;
            if (useDeformingSdf) {
                activateDeformingCollision();
            } else {
                updateCollision(0, true);
                if (active) {
                    ctx.rebindSceneSdf();
                }
                updateStatus();
            }
            for (const mesh of proxyDebugMeshes) {
                setMeshVisible(mesh, active && showCollisionShapes && !deformingActive());
            }
            if (deformingDebugMesh) {
                setMeshVisible(deformingDebugMesh, active && showCollisionShapes && deformingActive());
            }
        },
        extraControls: () => [status],
    };
}
