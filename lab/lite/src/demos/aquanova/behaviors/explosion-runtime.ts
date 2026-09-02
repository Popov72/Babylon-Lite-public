import {
    addToScene,
    createPhysicsAggregate,
    createPbrMaterial,
    disposeMeshGpu,
    getPhysicsBodyAngularVelocity,
    getPhysicsBodyLinearVelocity,
    isPbrMaterial,
    isStandardMaterial,
    markMaterialUboDirty,
    mat4Decompose,
    onBeforeRender,
    onSceneDispose,
    PhysicsShapeType,
    releasePhysicsShape,
    removeFromScene,
    removePhysicsBody,
    setPhysicsBodyAngularVelocity,
    setPhysicsBodyLinearVelocity,
    setPhysicsShapeFilterCollideMask,
    setPhysicsShapeFilterMembershipMask,
} from "babylon-lite";
import type { EngineContext, Mat4, Material, Mesh, PbrMaterialProps, PhysicsAggregate, PhysicsWorld, SceneContext, StandardMaterialProps } from "babylon-lite";
import { breakMesh } from "../../break-mesh.js";
import { meshGroupAabbProvider, meshGroupBounds, type MeshGroupAabb, type MeshGroupBounds } from "../mesh-bounds.js";
import type { ExplosionOptions, ExplosionRuntime } from "./game-context.js";

const REST_LINEAR_SPEED = 0.15;
const REST_ANGULAR_SPEED = 0.2;
const REST_DURATION = 0.75;
const COLLISION_SCALE = 0.8;
const MIN_COLLISION_EXTENT = 0.02;
const DEBRIS_MEMBERSHIP_MASK = 0x40000000;
const DEBRIS_COLLIDE_MASK = 0xbfffffff;
const DEFAULT_STATIC_TARGET_MASS = 10;
const MIN_STANDARD_ALPHA = 1 - 1e-6;

export interface ExplosionTarget {
    readonly entityName: string;
    readonly meshes: readonly Mesh[];
    readonly mass: number;
    readonly linearVelocity: readonly [number, number, number];
}

interface FadeMaterial {
    readonly material: Material;
    readonly baseAlpha: number;
    setAlpha(alpha: number): void;
}

interface DebrisPiece {
    readonly root: Mesh;
    aggregate: PhysicsAggregate | null;
    restSeconds: number;
}

interface DebrisBatch {
    elapsedSeconds: number;
    readonly visibleSeconds: number;
    readonly fadeSeconds: number;
    readonly pieces: readonly DebrisPiece[];
    readonly materials: readonly FadeMaterial[];
}

interface PreparedRoot {
    readonly root: Mesh;
    readonly source: Mesh;
}

interface PreparedDebris {
    readonly roots: PreparedRoot[];
    readonly materials: FadeMaterial[];
}

interface PreparedEntity {
    readonly bounds: () => MeshGroupAabb | null;
    readonly variants: ReadonlyMap<number, PreparedDebris | null>;
}

interface ActivatedRoot {
    readonly root: Mesh;
    readonly worldCentre: readonly [number, number, number];
}

interface ActivatedDebris {
    readonly roots: readonly ActivatedRoot[];
    readonly materials: readonly FadeMaterial[];
}

export interface AquanovaExplosionRuntimeOptions {
    readonly engine: EngineContext;
    readonly scene: SceneContext;
    readonly world: PhysicsWorld;
    readonly targets: () => readonly ExplosionTarget[];
    readonly fragmentCounts: readonly number[];
    readonly retireEntity: (entityName: string, meshes: readonly Mesh[]) => void;
}

function hashString(value: string): number {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index++) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function seededRandom(seed: number): () => number {
    let state = seed || 0x6d2b79f5;
    return () => {
        state += 0x6d2b79f5;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}

function fracturePoints(bounds: MeshGroupBounds, count: number, seed: number): number[][] {
    const random = seededRandom(seed);
    return Array.from({ length: count }, () => [
        bounds.centre[0] + (random() * 2 - 1) * bounds.half[0],
        bounds.centre[1] + (random() * 2 - 1) * bounds.half[1],
        bounds.centre[2] + (random() * 2 - 1) * bounds.half[2],
    ]);
}

function localBounds(mesh: Mesh): MeshGroupBounds | null {
    const min = mesh.boundMin;
    const max = mesh.boundMax;
    if (!min || !max) {
        return null;
    }
    return {
        centre: [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5],
        half: [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5, (max[2] - min[2]) * 0.5],
    };
}

function boundsFromAabb(aabb: MeshGroupAabb | null): MeshGroupBounds | null {
    if (!aabb) {
        return null;
    }
    return {
        centre: [(aabb.min[0] + aabb.max[0]) * 0.5, (aabb.min[1] + aabb.max[1]) * 0.5, (aabb.min[2] + aabb.max[2]) * 0.5],
        half: [(aabb.max[0] - aabb.min[0]) * 0.5, (aabb.max[1] - aabb.min[1]) * 0.5, (aabb.max[2] - aabb.min[2]) * 0.5],
    };
}

function transformPoint(matrix: Mat4, point: readonly [number, number, number]): [number, number, number] {
    return [
        matrix[0]! * point[0] + matrix[4]! * point[1] + matrix[8]! * point[2] + matrix[12]!,
        matrix[1]! * point[0] + matrix[5]! * point[1] + matrix[9]! * point[2] + matrix[13]!,
        matrix[2]! * point[0] + matrix[6]! * point[1] + matrix[10]! * point[2] + matrix[14]!,
    ];
}

function intersectsBlast(bounds: MeshGroupBounds, origin: readonly [number, number, number], radius: number): boolean {
    const dx = Math.max(Math.abs(origin[0] - bounds.centre[0]) - bounds.half[0], 0);
    const dy = Math.max(Math.abs(origin[1] - bounds.centre[1]) - bounds.half[1], 0);
    const dz = Math.max(Math.abs(origin[2] - bounds.centre[2]) - bounds.half[2], 0);
    return dx * dx + dy * dy + dz * dz <= radius * radius;
}

function clonePbrMaterial(source: PbrMaterialProps): FadeMaterial {
    const baseAlpha = source.alpha ?? 1;
    const material: PbrMaterialProps = {
        ...source,
        name: `${source.name ?? "material"}-explosion-debris`,
        baseColorFactor: source.baseColorFactor ? [...source.baseColorFactor] : undefined,
        alpha: baseAlpha,
        alphaBlend: true,
        _renderFeatures: undefined,
        _uboVersion: 0,
    };
    return {
        material,
        baseAlpha,
        setAlpha: (alpha) => {
            material.alpha = alpha;
            markMaterialUboDirty(material);
        },
    };
}

function cloneStandardMaterial(source: StandardMaterialProps): FadeMaterial {
    const baseAlpha = Math.min(source.alpha, MIN_STANDARD_ALPHA);
    const material: StandardMaterialProps = {
        ...source,
        name: `${source.name ?? "material"}-explosion-debris`,
        alpha: baseAlpha,
        _renderFeatures: undefined,
        _uboVersion: 0,
    };
    return {
        material,
        baseAlpha,
        setAlpha: (alpha) => {
            material.alpha = alpha;
            markMaterialUboDirty(material);
        },
    };
}

function cloneFadeMaterial(source: Material): FadeMaterial {
    if (isPbrMaterial(source)) {
        return clonePbrMaterial(source);
    }
    if (isStandardMaterial(source)) {
        return cloneStandardMaterial(source);
    }
    const material = createPbrMaterial();
    material.name = `${source.name ?? "material"}-explosion-debris`;
    material.baseColorFactor = [0.5, 0.5, 0.5, 1];
    material.roughnessFactor = 0.8;
    material.alpha = 1;
    material.alphaBlend = true;
    return {
        material,
        baseAlpha: 1,
        setAlpha: (alpha) => {
            material.alpha = alpha;
            markMaterialUboDirty(material);
        },
    };
}

function createCapMaterial(): FadeMaterial {
    const material = createPbrMaterial();
    material.name = "explosion-debris-interior";
    material.baseColorFactor = [0.13, 0.08, 0.045, 1];
    material.metallicFactor = 0;
    material.roughnessFactor = 0.9;
    material.alpha = 1;
    material.alphaBlend = true;
    return {
        material,
        baseAlpha: 1,
        setAlpha: (alpha) => {
            material.alpha = alpha;
            markMaterialUboDirty(material);
        },
    };
}

function normalizedEjectionDirection(centre: readonly [number, number, number], origin: readonly [number, number, number], random: () => number): [number, number, number] {
    let x = centre[0] - origin[0];
    let y = centre[1] - origin[1] + 0.35;
    let z = centre[2] - origin[2];
    let length = Math.sqrt(x * x + y * y + z * z);
    if (length <= 1e-5) {
        x = random() * 2 - 1;
        y = random() + 0.35;
        z = random() * 2 - 1;
        length = Math.sqrt(x * x + y * y + z * z) || 1;
    }
    return [x / length, y / length, z / length];
}

function setPiecePickable(root: Mesh, pickable: boolean): void {
    root.pickable = pickable;
    for (const child of root.children) {
        setPiecePickable(child as Mesh, pickable);
    }
}

export class AquanovaExplosionRuntime implements ExplosionRuntime {
    private readonly batches: DebrisBatch[] = [];
    private readonly preparedEntities = new Map<string, PreparedEntity>();

    public constructor(private readonly options: AquanovaExplosionRuntimeOptions) {
        this.prepareAllTargets(options.fragmentCounts);
        onBeforeRender(options.scene, (deltaMs) => this.update(deltaMs / 1000));
        onSceneDispose(options.scene, () => this.disposePreparedEntities());
    }

    public explode(entityName: string, ownerMeshes: readonly Mesh[], config: ExplosionOptions): void {
        const ownerBounds = boundsFromAabb(this.preparedEntities.get(entityName)?.bounds() ?? null) ?? meshGroupBounds(ownerMeshes);
        if (!ownerBounds) {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] explode on "${entityName}" has no readable owner geometry`);
            return;
        }
        const origin = ownerBounds.centre;
        const seen = new Set<string>();
        const targetBounds = new Map<string, MeshGroupBounds>();
        const targets = [...this.options.targets()]
            .filter((target) => {
                if (seen.has(target.entityName)) {
                    return false;
                }
                seen.add(target.entityName);
                const bounds = boundsFromAabb(this.preparedEntities.get(target.entityName)?.bounds() ?? null);
                if (!bounds || !intersectsBlast(bounds, origin, config.radius)) {
                    return false;
                }
                targetBounds.set(target.entityName, bounds);
                return true;
            })
            .sort((left, right) => left.entityName.localeCompare(right.entityName));

        let affected = 0;
        for (const target of targets) {
            const prepared = this.takePreparedDebris(target.entityName, config.fragmentCount);
            if (!prepared) {
                // eslint-disable-next-line no-console
                console.warn(`[aquanova] explosion has no precomputed ${config.fragmentCount}-fragment geometry for "${target.entityName}"`);
                continue;
            }
            const activated = this.activatePreparedTransforms(prepared);
            this.options.retireEntity(target.entityName, target.meshes);
            this.activateDebris(target, targetBounds.get(target.entityName)!, activated, origin, config);
            affected++;
        }
        // eslint-disable-next-line no-console
        console.log(`[aquanova] explosion "${entityName}" affected ${affected} eligible entit${affected === 1 ? "y" : "ies"} within ${config.radius} m`);
    }

    private prepareAllTargets(fragmentCounts: readonly number[]): void {
        const counts = [...new Set(fragmentCounts)];
        for (const target of this.options.targets()) {
            const variants = new Map<number, PreparedDebris | null>();
            for (const fragmentCount of counts) {
                variants.set(fragmentCount, this.prepareDebris(target, fragmentCount));
            }
            this.preparedEntities.set(target.entityName, {
                bounds: meshGroupAabbProvider(target.meshes),
                variants,
            });
        }
    }

    private prepareDebris(target: ExplosionTarget, fragmentCount: number): PreparedDebris | null {
        const materialClones = new Map<Material, FadeMaterial>();
        const capMaterial = createCapMaterial();
        const roots: PreparedRoot[] = [];

        for (let meshIndex = 0; meshIndex < target.meshes.length; meshIndex++) {
            const source = target.meshes[meshIndex]!;
            const bounds = localBounds(source);
            if (!bounds || !source.material) {
                continue;
            }
            let shellMaterial = materialClones.get(source.material);
            if (!shellMaterial) {
                shellMaterial = cloneFadeMaterial(source.material);
                materialClones.set(source.material, shellMaterial);
            }
            const points = fracturePoints(bounds, fragmentCount, hashString(`${target.entityName}:${source.name}:${meshIndex}`));
            const pieces = breakMesh(this.options.engine, source, points, capMaterial.material, {
                separation: 0,
                receiveShadows: source.receiveShadows,
                bakeWorldTransform: false,
            });
            for (const piece of pieces) {
                if (piece.material === source.material) {
                    piece.material = shellMaterial.material;
                }
                if (!piece.parent) {
                    roots.push({ root: piece, source });
                }
            }
        }
        if (roots.length === 0) {
            return null;
        }
        return {
            roots,
            materials: [...materialClones.values(), capMaterial],
        };
    }

    private takePreparedDebris(entityName: string, fragmentCount: number): PreparedDebris | null {
        const entity = this.preparedEntities.get(entityName);
        if (!entity) {
            return null;
        }
        const selected = entity.variants.get(fragmentCount) ?? null;
        if (!selected) {
            return null;
        }
        this.preparedEntities.delete(entityName);
        for (const variant of entity.variants.values()) {
            if (variant && variant !== selected) {
                this.disposePreparedDebris(variant);
            }
        }
        return selected;
    }

    private activatePreparedTransforms(prepared: PreparedDebris): ActivatedDebris {
        const roots = prepared.roots.map(({ root, source }): ActivatedRoot => {
            const world = source.worldMatrix;
            const transform = mat4Decompose(world);
            root.position.set(transform.translation.x, transform.translation.y, transform.translation.z);
            root.rotationQuaternion.set(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w);
            root.scaling.set(transform.scale.x, transform.scale.y, transform.scale.z);
            const bounds = localBounds(root);
            return {
                root,
                worldCentre: bounds ? transformPoint(world, bounds.centre) : [transform.translation.x, transform.translation.y, transform.translation.z],
            };
        });
        return { roots, materials: prepared.materials };
    }

    private activateDebris(
        target: ExplosionTarget,
        targetBounds: MeshGroupBounds,
        prepared: ActivatedDebris,
        origin: readonly [number, number, number],
        config: ExplosionOptions
    ): void {
        for (const { root } of prepared.roots) {
            addToScene(this.options.scene, root);
        }
        const dx = targetBounds.centre[0] - origin[0];
        const dy = targetBounds.centre[1] - origin[1];
        const dz = targetBounds.centre[2] - origin[2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const falloff = 1 - 0.65 * Math.min(1, distance / config.radius);
        const random = seededRandom(hashString(`${target.entityName}:ejection`));
        const mass = Math.max(0.1, (target.mass > 0 ? target.mass : DEFAULT_STATIC_TARGET_MASS) / prepared.roots.length);
        const pieces: DebrisPiece[] = [];
        for (const { root, worldCentre } of prepared.roots) {
            const direction = normalizedEjectionDirection(worldCentre, origin, random);
            const speed = config.strength * falloff;
            const min = root.boundMin ?? [-0.5, -0.5, -0.5];
            const max = root.boundMax ?? [0.5, 0.5, 0.5];
            const scaleX = Math.abs(root.scaling.x);
            const scaleY = Math.abs(root.scaling.y);
            const scaleZ = Math.abs(root.scaling.z);
            const signedScaleY = root.scaling.x * root.scaling.y * root.scaling.z < 0 ? -scaleY : scaleY;
            const aggregate = createPhysicsAggregate(this.options.world, root, PhysicsShapeType.BOX, {
                mass,
                friction: 0.65,
                restitution: 0.15,
                center: {
                    x: (min[0] + max[0]) * 0.5 * scaleX,
                    y: (min[1] + max[1]) * 0.5 * signedScaleY,
                    z: (min[2] + max[2]) * 0.5 * scaleZ,
                },
                extents: {
                    x: Math.max(MIN_COLLISION_EXTENT, (max[0] - min[0]) * scaleX * COLLISION_SCALE),
                    y: Math.max(MIN_COLLISION_EXTENT, (max[1] - min[1]) * scaleY * COLLISION_SCALE),
                    z: Math.max(MIN_COLLISION_EXTENT, (max[2] - min[2]) * scaleZ * COLLISION_SCALE),
                },
            });
            setPhysicsShapeFilterMembershipMask(this.options.world, aggregate.shape, DEBRIS_MEMBERSHIP_MASK);
            setPhysicsShapeFilterCollideMask(this.options.world, aggregate.shape, DEBRIS_COLLIDE_MASK);
            setPhysicsBodyLinearVelocity(this.options.world, aggregate.body, {
                x: target.linearVelocity[0] + direction[0] * speed,
                y: target.linearVelocity[1] + direction[1] * speed,
                z: target.linearVelocity[2] + direction[2] * speed,
            });
            setPhysicsBodyAngularVelocity(this.options.world, aggregate.body, {
                x: (random() * 2 - 1) * 5,
                y: (random() * 2 - 1) * 5,
                z: (random() * 2 - 1) * 5,
            });
            pieces.push({
                root,
                aggregate,
                restSeconds: 0,
            });
        }
        this.batches.push({
            elapsedSeconds: 0,
            visibleSeconds: config.debrisLifetime,
            fadeSeconds: config.fadeDuration,
            pieces,
            materials: prepared.materials,
        });
    }

    private update(deltaSeconds: number): void {
        for (let index = this.batches.length - 1; index >= 0; index--) {
            const batch = this.batches[index]!;
            batch.elapsedSeconds += deltaSeconds;
            for (const piece of batch.pieces) {
                this.updatePieceCollision(piece, deltaSeconds);
            }
            if (batch.elapsedSeconds < batch.visibleSeconds) {
                continue;
            }
            const opacity = 1 - (batch.elapsedSeconds - batch.visibleSeconds) / batch.fadeSeconds;
            if (opacity > 0) {
                for (const material of batch.materials) {
                    material.setAlpha(material.baseAlpha * opacity);
                }
                continue;
            }
            for (const piece of batch.pieces) {
                this.removePieceCollision(piece);
                const { root } = piece;
                removeFromScene(this.options.scene, root);
            }
            this.batches.splice(index, 1);
        }
    }

    private updatePieceCollision(piece: DebrisPiece, deltaSeconds: number): void {
        const aggregate = piece.aggregate;
        if (!aggregate) {
            return;
        }
        const linear = getPhysicsBodyLinearVelocity(this.options.world, aggregate.body);
        const angular = getPhysicsBodyAngularVelocity(this.options.world, aggregate.body);
        if (
            linear.x * linear.x + linear.y * linear.y + linear.z * linear.z > REST_LINEAR_SPEED * REST_LINEAR_SPEED ||
            angular.x * angular.x + angular.y * angular.y + angular.z * angular.z > REST_ANGULAR_SPEED * REST_ANGULAR_SPEED
        ) {
            piece.restSeconds = 0;
            return;
        }
        piece.restSeconds += deltaSeconds;
        if (piece.restSeconds >= REST_DURATION) {
            this.removePieceCollision(piece);
            setPiecePickable(piece.root, false);
        }
    }

    private removePieceCollision(piece: DebrisPiece): void {
        const aggregate = piece.aggregate;
        if (!aggregate) {
            return;
        }
        removePhysicsBody(this.options.world, aggregate.body);
        releasePhysicsShape(this.options.world, aggregate.shape);
        piece.aggregate = null;
    }

    private disposePreparedDebris(prepared: PreparedDebris): void {
        const disposeTree = (mesh: Mesh): void => {
            for (const child of mesh.children) {
                disposeTree(child as Mesh);
            }
            disposeMeshGpu(mesh);
        };
        for (const { root } of prepared.roots) {
            disposeTree(root);
        }
    }

    private disposePreparedEntities(): void {
        for (const entity of this.preparedEntities.values()) {
            for (const prepared of entity.variants.values()) {
                if (prepared) {
                    this.disposePreparedDebris(prepared);
                }
            }
        }
        this.preparedEntities.clear();
    }
}
