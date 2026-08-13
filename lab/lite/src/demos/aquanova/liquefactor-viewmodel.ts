import {
    createTransformNode,
    getProjectionMatrix,
    loadGltf,
    quatFromRotationMatrix,
    setMeshVisible,
    type EngineContext,
    type FreeCamera,
    type Mat4,
    type Mesh,
    type SceneNode,
    type TransformNode,
} from "babylon-lite";
import { LIQUEFACTOR_MODELS, type LiquefactorModel } from "./settings.js";

const MODEL_URLS: Readonly<Record<LiquefactorModel, string>> = {
    "20k": "/aquanova/weapons/liquefactor-20k.glb",
    "80k": "/aquanova/weapons/liquefactor-80k.glb",
    "350k": "/aquanova/weapons/liquefactor-350k.glb",
};

// The root is anchored in normalized screen space, then converted through the live projection.
// This keeps the held weapon in the same place when the canvas aspect ratio changes.
const VIEW_DEPTH = 1.1;
const SCREEN_X = 0.35;
const SCREEN_Y = -0.35;
const AIM_DISTANCE = 2.5;
export const LIQUEFACTOR_MODEL_SCALE = 0.35;
const ADJUSTMENT_POSITION = [-0.009, -0.7394, 0.517] as const;
const ADJUSTMENT_ROTATION_Y = (8.44 * Math.PI) / 180;
const ADJUSTMENT_SCALE = 2.74;
// Coordinates are in the adjustment node's space, after the glTF root's X mirror.
export const LIQUEFACTOR_MUZZLE = { x: 0.115, y: 0.292, z: 0.49 };
const GIZMO_PIVOT = { x: 0, y: 0.215, z: 0 };
export const LIQUEFACTOR_ADJUSTMENT_MUZZLE = {
    x: LIQUEFACTOR_MUZZLE.x - GIZMO_PIVOT.x,
    y: LIQUEFACTOR_MUZZLE.y - GIZMO_PIVOT.y,
    z: LIQUEFACTOR_MUZZLE.z - GIZMO_PIVOT.z,
};
export const LIQUEFACTOR_BARREL_DIRECTION = normalize(0.115, 0, 0.29);

export interface LiquefactorPose {
    position: [number, number, number];
    rotation: [number, number, number, number];
}

export interface LiquefactorViewmodel {
    readonly root: TransformNode;
    readonly adjustment: TransformNode;
    readonly localGuideRoot: TransformNode;
    readonly localGuideOrigin: TransformNode;
    readonly localGuideYaw: TransformNode;
    readonly meshes: readonly Mesh[];
    readonly model: LiquefactorModel;
    select(model: LiquefactorModel): void;
    update(camera: FreeCamera, aspectRatio: number): void;
}

interface ModelEntry {
    root: SceneNode;
    meshes: Mesh[];
}

function setLocalParent(child: SceneNode, parent: SceneNode | FreeCamera): void {
    if (child.parent && "children" in child.parent) {
        const siblings = child.parent.children as SceneNode[];
        const index = siblings.indexOf(child);
        if (index >= 0) siblings.splice(index, 1);
    }
    child.parent = parent;
    if (!parent.children.includes(child)) parent.children.push(child);
}

function collectMeshes(node: SceneNode, out: Mesh[]): void {
    if ("_gpu" in node && "material" in node) out.push(node as Mesh);
    for (const child of node.children) collectMeshes(child as SceneNode, out);
}

function normalize(x: number, y: number, z: number): [number, number, number] {
    const inv = 1 / (Math.hypot(x, y, z) || 1);
    return [x * inv, y * inv, z * inv];
}

function cross(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function rotationForAim(aim: readonly [number, number, number], cameraUp: readonly [number, number, number]): { matrix: Mat4; rotation: [number, number, number, number] } {
    const localForward = LIQUEFACTOR_BARREL_DIRECTION;
    const localRight = normalize(...cross(localForward, [0, 1, 0]));
    const localUp = normalize(...cross(localRight, localForward));
    const worldRight = normalize(...cross(aim, cameraUp));
    const worldUp = normalize(...cross(worldRight, aim));
    const source = [localRight, localForward, localUp] as const;
    const target = [worldRight, aim, worldUp] as const;
    const values: number[] = [];
    for (const column of [0, 1, 2] as const) {
        values.push(
            target[0][0] * source[0][column] + target[1][0] * source[1][column] + target[2][0] * source[2][column],
            target[0][1] * source[0][column] + target[1][1] * source[1][column] + target[2][1] * source[2][column],
            target[0][2] * source[0][column] + target[1][2] * source[1][column] + target[2][2] * source[2][column],
            0
        );
    }
    values.push(0, 0, 0, 1);
    const matrix = new Float32Array(values) as unknown as Mat4;
    const q = quatFromRotationMatrix(matrix);
    return { matrix, rotation: [q.x, q.y, q.z, q.w] };
}

function rotateDirection(rotation: readonly [number, number, number, number], point: { x: number; y: number; z: number }): [number, number, number] {
    const [x, y, z, w] = rotation;
    const tx = 2 * (y * point.z - z * point.y);
    const ty = 2 * (z * point.x - x * point.z);
    const tz = 2 * (x * point.y - y * point.x);
    return [point.x + w * tx + (y * tz - z * ty), point.y + w * ty + (z * tx - x * tz), point.z + w * tz + (x * ty - y * tx)];
}

/**
 * Compute a first-person pose from a camera basis and projection.
 * The authored barrel axis is aimed from the transformed muzzle to the screen-centre ray.
 */
export function computeLiquefactorPose(cameraWorld: ArrayLike<number>, projection: ArrayLike<number>): LiquefactorPose {
    const px = cameraWorld[12]!;
    const py = cameraWorld[13]!;
    const pz = cameraWorld[14]!;
    const rx = cameraWorld[0]!;
    const ry = cameraWorld[1]!;
    const rz = cameraWorld[2]!;
    const ux = cameraWorld[4]!;
    const uy = cameraWorld[5]!;
    const uz = cameraWorld[6]!;
    const fx = cameraWorld[8]!;
    const fy = cameraWorld[9]!;
    const fz = cameraWorld[10]!;
    const viewX = (SCREEN_X * VIEW_DEPTH) / projection[0]!;
    const viewY = (SCREEN_Y * VIEW_DEPTH) / projection[5]!;
    const muzzlePosition: [number, number, number] = [
        px + fx * VIEW_DEPTH + rx * viewX + ux * viewY,
        py + fy * VIEW_DEPTH + ry * viewX + uy * viewY,
        pz + fz * VIEW_DEPTH + rz * viewX + uz * viewY,
    ];

    const target: [number, number, number] = [px + fx * AIM_DISTANCE, py + fy * AIM_DISTANCE, pz + fz * AIM_DISTANCE];
    const cameraUp: [number, number, number] = [ux, uy, uz];
    const aim = normalize(target[0] - muzzlePosition[0], target[1] - muzzlePosition[1], target[2] - muzzlePosition[2]);
    const pose = rotationForAim(aim, cameraUp);
    const muzzleOffset = rotateDirection(pose.rotation, LIQUEFACTOR_ADJUSTMENT_MUZZLE);
    const position: [number, number, number] = [
        muzzlePosition[0] - muzzleOffset[0] * LIQUEFACTOR_MODEL_SCALE,
        muzzlePosition[1] - muzzleOffset[1] * LIQUEFACTOR_MODEL_SCALE,
        muzzlePosition[2] - muzzleOffset[2] * LIQUEFACTOR_MODEL_SCALE,
    ];
    return { position, rotation: pose.rotation };
}

const CAMERA_LOCAL_WORLD = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

export async function createLiquefactorViewmodel(engine: EngineContext, camera: FreeCamera): Promise<LiquefactorViewmodel> {
    const root = createTransformNode("liquefactor-viewmodel");
    root.scaling.set(LIQUEFACTOR_MODEL_SCALE, LIQUEFACTOR_MODEL_SCALE, LIQUEFACTOR_MODEL_SCALE);
    setLocalParent(root, camera);
    const adjustment = createTransformNode("liquefactor-adjustment");
    adjustment.position.set(...ADJUSTMENT_POSITION);
    adjustment.rotation.y = ADJUSTMENT_ROTATION_Y;
    adjustment.scaling.set(ADJUSTMENT_SCALE, ADJUSTMENT_SCALE, ADJUSTMENT_SCALE);
    setLocalParent(adjustment, root);
    const content = createTransformNode("liquefactor-content");
    content.position.set(-GIZMO_PIVOT.x, -GIZMO_PIVOT.y, -GIZMO_PIVOT.z);
    setLocalParent(content, adjustment);

    // This mirrored hierarchy keeps the calibrated aim origin/direction available in the
    // gizmo layer without rendering debug geometry over the weapon.
    const localGuideRoot = createTransformNode("liquefactor-local-guide-root");
    setLocalParent(localGuideRoot, camera);
    const localGuideWeaponSpace = createTransformNode("liquefactor-local-guide-weapon-space");
    setLocalParent(localGuideWeaponSpace, localGuideRoot);
    const localGuideOrigin = createTransformNode("liquefactor-local-guide-origin");
    localGuideOrigin.position.set(0, 0.1707, 0.2889);
    setLocalParent(localGuideOrigin, localGuideWeaponSpace);
    const localGuideYaw = createTransformNode("liquefactor-local-guide-yaw");
    localGuideYaw.rotation.y = (17.5 * Math.PI) / 180;
    setLocalParent(localGuideYaw, localGuideOrigin);

    const entries = new Map<LiquefactorModel, ModelEntry>();
    const allMeshes: Mesh[] = [];

    await Promise.all(
        LIQUEFACTOR_MODELS.map(async (model) => {
            const asset = await loadGltf(engine, MODEL_URLS[model]);
            const modelRoot = asset.entities[0] as SceneNode | undefined;
            if (!modelRoot) throw new Error(`Liquefactor ${model} has no root entity`);
            const meshes: Mesh[] = [];
            collectMeshes(modelRoot, meshes);
            for (const mesh of meshes) {
                mesh.pickable = false;
                allMeshes.push(mesh);
            }
            setLocalParent(modelRoot, content);
            entries.set(model, { root: modelRoot, meshes });
        })
    );

    let selected: LiquefactorModel = "80k";
    const select = (model: LiquefactorModel): void => {
        selected = model;
        for (const [entryModel, entry] of entries) {
            const visible = entryModel === model;
            for (const mesh of entry.meshes) setMeshVisible(mesh, visible);
        }
    };
    select(selected);

    return {
        root,
        adjustment,
        localGuideRoot,
        localGuideOrigin,
        localGuideYaw,
        meshes: allMeshes,
        get model() {
            return selected;
        },
        select,
        update(camera, aspectRatio) {
            // The root is a camera child, so its transform is camera-local and only the
            // projection changes when the viewport aspect ratio changes.
            const pose = computeLiquefactorPose(CAMERA_LOCAL_WORLD, getProjectionMatrix(camera, aspectRatio));
            root.position.set(pose.position[0], pose.position[1], pose.position[2]);
            root.rotationQuaternion.set(pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.rotation[3]);
            localGuideRoot.position.set(root.position.x, root.position.y, root.position.z);
            localGuideRoot.rotationQuaternion.set(root.rotationQuaternion.x, root.rotationQuaternion.y, root.rotationQuaternion.z, root.rotationQuaternion.w);
            localGuideRoot.scaling.set(root.scaling.x, root.scaling.y, root.scaling.z);
            localGuideWeaponSpace.position.set(adjustment.position.x, adjustment.position.y, adjustment.position.z);
            localGuideWeaponSpace.rotationQuaternion.set(
                adjustment.rotationQuaternion.x,
                adjustment.rotationQuaternion.y,
                adjustment.rotationQuaternion.z,
                adjustment.rotationQuaternion.w
            );
            localGuideWeaponSpace.scaling.set(adjustment.scaling.x, adjustment.scaling.y, adjustment.scaling.z);
        },
    };
}
