import { evaluateSampler } from "../packages/babylon-lite/src/animation/evaluate.js";
import { INTERP_CUBICSPLINE, INTERP_LINEAR, INTERP_STEP } from "../packages/babylon-lite/src/animation/types.js";
import type { AnimationSampler, InterpMode } from "../packages/babylon-lite/src/animation/types.js";
import { composeMat4 as mat4Compose } from "../packages/babylon-lite/src/math/compose-mat4.js";
import { invertMat4 as mat4Invert } from "../packages/babylon-lite/src/math/invert-mat4.js";
import { multiplyMat4 as mat4Multiply } from "../packages/babylon-lite/src/math/multiply-mat4.js";
import type { Mat4 } from "../packages/babylon-lite/src/math/types.js";

const COLLISION_MESHES = new Set(["Whale_Body", "Dorsal_Fin", "Pectoral_Fin_L", "Pectoral_Fin_R", "Tail_Flukes"]);

interface GltfNode {
    readonly name?: string;
    readonly mesh?: number;
    readonly skin?: number;
    readonly children?: readonly number[];
    readonly matrix?: readonly number[];
    readonly translation?: readonly number[];
    readonly rotation?: readonly number[];
    readonly scale?: readonly number[];
}

interface GltfAccessor {
    readonly bufferView: number;
    readonly byteOffset?: number;
    readonly componentType: number;
    readonly count: number;
    readonly type: string;
    readonly normalized?: boolean;
}

interface GltfPrimitive {
    readonly attributes: Record<string, number>;
    readonly indices: number;
    readonly mode?: number;
}

interface GltfJson {
    readonly nodes: readonly GltfNode[];
    readonly meshes: readonly { readonly primitives: readonly GltfPrimitive[] }[];
    readonly skins: readonly { readonly joints: readonly number[]; readonly inverseBindMatrices: number }[];
    readonly animations: readonly {
        readonly samplers: readonly { readonly input: number; readonly output: number; readonly interpolation?: string }[];
        readonly channels: readonly { readonly sampler: number; readonly target: { readonly node?: number; readonly path: string } }[];
    }[];
    readonly accessors: readonly GltfAccessor[];
    readonly bufferViews: readonly { readonly byteOffset?: number; readonly byteStride?: number }[];
}

interface ParsedGlb {
    readonly json: GltfJson;
    readonly binary: Uint8Array;
}

interface SkinVertex {
    readonly x: number;
    readonly y: number;
    readonly z: number;
    readonly joints: readonly number[];
    readonly weights: readonly number[];
}

export interface JumpingWhaleFrameSource {
    readonly frameCount: number;
    readonly frameRate: number;
    readonly duration: number;
    readonly indices: Uint32Array;
    readonly frames: readonly Float32Array[];
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const COMPONENT_BYTES: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };

function parseGlb(bytes: Uint8Array): ParsedGlb {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2) {
        throw new Error("Jumping whale SDF baker requires a glTF 2.0 binary asset.");
    }
    let json: GltfJson | null = null;
    let binary: Uint8Array | null = null;
    for (let offset = 12; offset + 8 <= bytes.byteLength;) {
        const length = view.getUint32(offset, true);
        const type = view.getUint32(offset + 4, true);
        offset += 8;
        if (offset + length > bytes.byteLength) {
            throw new Error("Jumping whale GLB contains a truncated chunk.");
        }
        if (type === 0x4e4f534a) {
            json = JSON.parse(new TextDecoder().decode(bytes.subarray(offset, offset + length)).replace(/[\0 ]+$/, "")) as GltfJson;
        } else if (type === 0x004e4942) {
            binary = bytes.subarray(offset, offset + length);
        }
        offset += length;
    }
    if (!json || !binary) {
        throw new Error("Jumping whale GLB is missing JSON or binary data.");
    }
    return { json, binary };
}

function accessorValues(glb: ParsedGlb, accessorIndex: number): Float32Array {
    const accessor = glb.json.accessors[accessorIndex]!;
    const bufferView = glb.json.bufferViews[accessor.bufferView]!;
    const components = COMPONENTS[accessor.type];
    const componentBytes = COMPONENT_BYTES[accessor.componentType];
    if (!components || !componentBytes) {
        throw new Error(`Unsupported whale accessor format ${accessor.componentType}/${accessor.type}.`);
    }
    const stride = bufferView.byteStride ?? components * componentBytes;
    const base = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const view = new DataView(glb.binary.buffer, glb.binary.byteOffset, glb.binary.byteLength);
    const values = new Float32Array(accessor.count * components);
    const integerScale =
        accessor.normalized === true
            ? accessor.componentType === 5120
                ? 127
                : accessor.componentType === 5121
                  ? 255
                  : accessor.componentType === 5122
                    ? 32767
                    : accessor.componentType === 5123
                      ? 65535
                      : 1
            : 1;
    for (let index = 0; index < accessor.count; index++) {
        for (let component = 0; component < components; component++) {
            const offset = base + index * stride + component * componentBytes;
            let value: number;
            switch (accessor.componentType) {
                case 5120:
                    value = view.getInt8(offset);
                    break;
                case 5121:
                    value = view.getUint8(offset);
                    break;
                case 5122:
                    value = view.getInt16(offset, true);
                    break;
                case 5123:
                    value = view.getUint16(offset, true);
                    break;
                case 5125:
                    value = view.getUint32(offset, true);
                    break;
                case 5126:
                    value = view.getFloat32(offset, true);
                    break;
                default:
                    throw new Error("Unsupported whale accessor component type.");
            }
            values[index * components + component] = value / integerScale;
        }
    }
    return values;
}

function accessorIndices(glb: ParsedGlb, accessorIndex: number): Uint32Array {
    const values = accessorValues(glb, accessorIndex);
    return Uint32Array.from(values);
}

function matrixFrom(values: Float32Array, index: number): Mat4 {
    return values.slice(index * 16, index * 16 + 16) as unknown as Mat4;
}

function transformPoint(matrix: Mat4, x: number, y: number, z: number): [number, number, number] {
    return [
        matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]!,
        matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]!,
        matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]!,
    ];
}

function boundaryLoops(indices: readonly number[]): number[][] {
    const edges = new Map<string, { a: number; b: number; count: number }>();
    const add = (a: number, b: number): void => {
        const key = a < b ? `${a}:${b}` : `${b}:${a}`;
        const edge = edges.get(key);
        if (edge) {
            edge.count++;
        } else {
            edges.set(key, { a, b, count: 1 });
        }
    };
    for (let offset = 0; offset < indices.length; offset += 3) {
        const a = indices[offset]!;
        const b = indices[offset + 1]!;
        const c = indices[offset + 2]!;
        add(a, b);
        add(b, c);
        add(c, a);
    }
    const adjacency = new Map<number, number[]>();
    const unvisited = new Set<string>();
    for (const [key, edge] of edges) {
        if (edge.count !== 1) {
            continue;
        }
        (adjacency.get(edge.a) ?? adjacency.set(edge.a, []).get(edge.a)!).push(edge.b);
        (adjacency.get(edge.b) ?? adjacency.set(edge.b, []).get(edge.b)!).push(edge.a);
        unvisited.add(key);
    }
    const edgeKey = (a: number, b: number): string => (a < b ? `${a}:${b}` : `${b}:${a}`);
    const loops: number[][] = [];
    while (unvisited.size > 0) {
        const first = unvisited.values().next().value as string;
        const [start, next] = first.split(":").map(Number) as [number, number];
        const loop = [start];
        let previous = start;
        let current = next;
        unvisited.delete(first);
        while (current !== start) {
            loop.push(current);
            const candidates = adjacency.get(current) ?? [];
            const candidate = candidates.find((value) => value !== previous && unvisited.has(edgeKey(current, value)));
            if (candidate === undefined) {
                throw new Error("Whale body contains a non-loop boundary.");
            }
            unvisited.delete(edgeKey(current, candidate));
            previous = current;
            current = candidate;
        }
        loops.push(loop);
    }
    return loops;
}

function appendBoundaryCaps(positions: Float32Array, indices: Uint32Array, vertices: SkinVertex[], vertexOffset: number, combinedIndices: number[]): void {
    for (const loop of boundaryLoops(indices)) {
        let x = 0;
        let y = 0;
        let z = 0;
        const jointWeights = new Map<number, number>();
        for (const localIndex of loop) {
            x += positions[localIndex * 3]!;
            y += positions[localIndex * 3 + 1]!;
            z += positions[localIndex * 3 + 2]!;
            const vertex = vertices[vertexOffset + localIndex]!;
            for (let influence = 0; influence < vertex.weights.length; influence++) {
                const joint = vertex.joints[influence]!;
                jointWeights.set(joint, (jointWeights.get(joint) ?? 0) + vertex.weights[influence]!);
            }
        }
        const influences = [...jointWeights].sort((a, b) => b[1] - a[1]).slice(0, 4);
        const totalWeight = influences.reduce((sum, entry) => sum + entry[1], 0);
        const centerIndex = vertices.length;
        vertices.push({
            x: x / loop.length,
            y: y / loop.length,
            z: z / loop.length,
            joints: influences.map((entry) => entry[0]),
            weights: influences.map((entry) => entry[1] / totalWeight),
        });
        for (let index = 0; index < loop.length; index++) {
            combinedIndices.push(vertexOffset + loop[index]!, vertexOffset + loop[(index + 1) % loop.length]!, centerIndex);
        }
    }
}

function interpolationMode(value: string | undefined): InterpMode {
    return value === "STEP" ? INTERP_STEP : value === "CUBICSPLINE" ? INTERP_CUBICSPLINE : INTERP_LINEAR;
}

export function evaluateJumpingWhaleFrames(bytes: Uint8Array, frameRate: number): JumpingWhaleFrameSource {
    const glb = parseGlb(bytes);
    const skin = glb.json.skins[0];
    const animation = glb.json.animations[0];
    if (!skin || !animation) {
        throw new Error("Jumping whale requires one skin and one animation.");
    }
    const spineJoint = skin.joints.findIndex((nodeIndex) => glb.json.nodes[nodeIndex]?.name === "Spine");
    if (spineJoint < 0) {
        throw new Error("Jumping whale skin has no Spine joint.");
    }
    const inverseBindData = accessorValues(glb, skin.inverseBindMatrices);
    const inverseBindMatrices = skin.joints.map((_node, index) => matrixFrom(inverseBindData, index));
    const vertices: SkinVertex[] = [];
    const combinedIndices: number[] = [];
    for (let nodeIndex = 0; nodeIndex < glb.json.nodes.length; nodeIndex++) {
        const node = glb.json.nodes[nodeIndex]!;
        if (!node.name || !COLLISION_MESHES.has(node.name) || node.mesh === undefined || node.skin !== 0) {
            continue;
        }
        for (const primitive of glb.json.meshes[node.mesh]!.primitives) {
            if ((primitive.mode ?? 4) !== 4) {
                throw new Error(`Whale collision mesh "${node.name}" is not triangle topology.`);
            }
            const positions = accessorValues(glb, primitive.attributes.POSITION!);
            const joints0 = accessorValues(glb, primitive.attributes.JOINTS_0!);
            const weights0 = accessorValues(glb, primitive.attributes.WEIGHTS_0!);
            const joints1 = primitive.attributes.JOINTS_1 === undefined ? null : accessorValues(glb, primitive.attributes.JOINTS_1);
            const weights1 = primitive.attributes.WEIGHTS_1 === undefined ? null : accessorValues(glb, primitive.attributes.WEIGHTS_1);
            const localIndices = accessorIndices(glb, primitive.indices);
            const vertexOffset = vertices.length;
            for (let vertex = 0; vertex < positions.length / 3; vertex++) {
                const joints: number[] = [];
                const weights: number[] = [];
                for (let component = 0; component < 4; component++) {
                    joints.push(Math.round(joints0[vertex * 4 + component]!));
                    weights.push(weights0[vertex * 4 + component]!);
                }
                if (joints1 && weights1) {
                    for (let component = 0; component < 4; component++) {
                        joints.push(Math.round(joints1[vertex * 4 + component]!));
                        weights.push(weights1[vertex * 4 + component]!);
                    }
                }
                const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
                if (totalWeight <= 0) {
                    throw new Error(`Whale collision vertex ${vertex} has no skin weight.`);
                }
                for (let index = 0; index < weights.length; index++) {
                    weights[index] = weights[index]! / totalWeight;
                }
                vertices.push({
                    x: positions[vertex * 3]!,
                    y: positions[vertex * 3 + 1]!,
                    z: positions[vertex * 3 + 2]!,
                    joints,
                    weights,
                });
            }
            for (const index of localIndices) {
                combinedIndices.push(vertexOffset + index);
            }
            if (node.name === "Whale_Body") {
                appendBoundaryCaps(positions, localIndices, vertices, vertexOffset, combinedIndices);
            }
        }
    }
    if (vertices.length === 0 || combinedIndices.length === 0) {
        throw new Error("Jumping whale collision meshes were not found.");
    }
    const parent = new Int32Array(glb.json.nodes.length).fill(-1);
    glb.json.nodes.forEach((node, index) => node.children?.forEach((child) => (parent[child] = index)));
    const rest = glb.json.nodes.map((node) => ({
        matrix: node.matrix ? (Float32Array.from(node.matrix) as unknown as Mat4) : null,
        translation: [...(node.translation ?? [0, 0, 0])] as [number, number, number],
        rotation: [...(node.rotation ?? [0, 0, 0, 1])] as [number, number, number, number],
        scale: [...(node.scale ?? [1, 1, 1])] as [number, number, number],
    }));
    const samplers: AnimationSampler[] = animation.samplers.map((sampler) => ({
        input: accessorValues(glb, sampler.input),
        output: accessorValues(glb, sampler.output),
        interpolation: interpolationMode(sampler.interpolation),
    }));
    const duration = samplers.reduce((maximum, sampler) => Math.max(maximum, sampler.input.at(-1) ?? 0), 0);
    const frameCount = Math.max(2, Math.round(duration * frameRate));
    const sample = new Float32Array(4);
    const frames: Float32Array[] = [];
    const localMatrices: Mat4[] = new Array(glb.json.nodes.length);
    const worldMatrices: Mat4[] = new Array(glb.json.nodes.length);
    for (let frame = 0; frame < frameCount; frame++) {
        const time = frame / frameRate;
        const translations = rest.map((node) => [...node.translation] as [number, number, number]);
        const rotations = rest.map((node) => [...node.rotation] as [number, number, number, number]);
        const scales = rest.map((node) => [...node.scale] as [number, number, number]);
        for (const channel of animation.channels) {
            const nodeIndex = channel.target.node;
            if (nodeIndex === undefined || channel.target.path === "weights") {
                continue;
            }
            const stride = channel.target.path === "rotation" ? 4 : 3;
            evaluateSampler(samplers[channel.sampler]!, time, stride, channel.target.path === "rotation", sample, 0);
            if (channel.target.path === "translation") {
                translations[nodeIndex] = [sample[0]!, sample[1]!, sample[2]!];
            } else if (channel.target.path === "rotation") {
                rotations[nodeIndex] = [sample[0]!, sample[1]!, sample[2]!, sample[3]!];
            } else if (channel.target.path === "scale") {
                scales[nodeIndex] = [sample[0]!, sample[1]!, sample[2]!];
            }
        }
        for (let nodeIndex = 0; nodeIndex < rest.length; nodeIndex++) {
            const node = rest[nodeIndex]!;
            const t = translations[nodeIndex]!;
            const r = rotations[nodeIndex]!;
            const s = scales[nodeIndex]!;
            localMatrices[nodeIndex] = node.matrix ?? mat4Compose(t[0], t[1], t[2], r[0], r[1], r[2], r[3], s[0], s[1], s[2]);
        }
        const resolveWorld = (nodeIndex: number): Mat4 => {
            const existing = worldMatrices[nodeIndex];
            if (existing) {
                return existing;
            }
            const parentIndex = parent[nodeIndex]!;
            const world = parentIndex < 0 ? localMatrices[nodeIndex]! : mat4Multiply(resolveWorld(parentIndex), localMatrices[nodeIndex]!);
            worldMatrices[nodeIndex] = world;
            return world;
        };
        worldMatrices.fill(undefined as unknown as Mat4);
        const jointWorld = skin.joints.map(resolveWorld);
        const inverseSpine = mat4Invert(jointWorld[spineJoint]!);
        if (!inverseSpine) {
            throw new Error(`Whale Spine matrix is singular at frame ${frame}.`);
        }
        const positions = new Float32Array(vertices.length * 3);
        for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex++) {
            const vertex = vertices[vertexIndex]!;
            let worldX = 0;
            let worldY = 0;
            let worldZ = 0;
            for (let influence = 0; influence < vertex.weights.length; influence++) {
                const weight = vertex.weights[influence]!;
                if (weight <= 0) {
                    continue;
                }
                const joint = vertex.joints[influence]!;
                const jointPoint = transformPoint(inverseBindMatrices[joint]!, vertex.x, vertex.y, vertex.z);
                const worldPoint = transformPoint(jointWorld[joint]!, jointPoint[0], jointPoint[1], jointPoint[2]);
                worldX += worldPoint[0] * weight;
                worldY += worldPoint[1] * weight;
                worldZ += worldPoint[2] * weight;
            }
            const spinePoint = transformPoint(inverseSpine, worldX, worldY, worldZ);
            positions[vertexIndex * 3] = spinePoint[0];
            positions[vertexIndex * 3 + 1] = spinePoint[1];
            positions[vertexIndex * 3 + 2] = spinePoint[2];
        }
        frames.push(positions);
    }
    return {
        frameCount,
        frameRate,
        duration,
        indices: Uint32Array.from(combinedIndices),
        frames,
    };
}
