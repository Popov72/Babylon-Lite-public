import { addToScene, createSphere, createStandardMaterial, setMeshVisible, type EngineContext, type Mesh, type SceneContext } from "babylon-lite";
import type { LiquefactorViewmodel } from "./liquefactor-viewmodel.js";

const PROJECTILE_POOL_SIZE = 16;
const IMPACT_POOL_SIZE = 16;
const IMPACT_DURATION_MS = 220;

export interface PistolImpact {
    readonly mesh: Mesh;
    readonly point: readonly [number, number, number];
    readonly distance: number;
    readonly direction: readonly [number, number, number];
}

interface Projectile {
    readonly mesh: Mesh;
    active: boolean;
    elapsedMs: number;
    durationMs: number;
    start: [number, number, number];
    end: [number, number, number];
    impact: PistolImpact | null;
}

interface ImpactCue {
    readonly mesh: Mesh;
    active: boolean;
    elapsedMs: number;
}

export interface PistolProjectileRuntime {
    fire(mesh: Mesh | null, point: readonly [number, number, number] | null, distance: number | null, range: number, speed: number): void;
    update(deltaMs: number, isCollisionActive?: (mesh: Mesh) => boolean): readonly PistolImpact[];
    clear(): void;
}

function createUnlitMaterial(color: readonly [number, number, number]) {
    const material = createStandardMaterial();
    material.disableLighting = true;
    material.diffuseColor = [...color];
    material.emissiveColor = [...color];
    material.specularColor = [0, 0, 0];
    return material;
}

function hide(mesh: Mesh): void {
    setMeshVisible(mesh, false);
}

function directionFrom(matrix: ArrayLike<number>): [number, number, number] {
    const x = matrix[8] ?? 0;
    const y = matrix[9] ?? 0;
    const z = matrix[10] ?? 1;
    const inverseLength = 1 / (Math.hypot(x, y, z) || 1);
    return [x * inverseLength, y * inverseLength, z * inverseLength];
}

export function createPistolProjectileRuntime(engine: EngineContext, scene: SceneContext, viewmodel: LiquefactorViewmodel): PistolProjectileRuntime {
    const projectileMaterial = createUnlitMaterial([3.5, 2.2, 0.35]);
    const impactMaterial = createUnlitMaterial([4, 1.4, 0.2]);
    const projectiles: Projectile[] = Array.from({ length: PROJECTILE_POOL_SIZE }, (_, index) => {
        const mesh = createSphere(engine, { diameter: 0.045, segments: 6 });
        mesh.name = `pistol-projectile-${index}`;
        mesh.material = projectileMaterial;
        mesh.pickable = false;
        hide(mesh);
        addToScene(scene, mesh);
        return {
            mesh,
            active: false,
            elapsedMs: 0,
            durationMs: 0,
            start: [0, 0, 0],
            end: [0, 0, 0],
            impact: null,
        };
    });
    const impactCues: ImpactCue[] = Array.from({ length: IMPACT_POOL_SIZE }, (_, index) => {
        const mesh = createSphere(engine, { diameter: 0.12, segments: 8 });
        mesh.name = `pistol-impact-${index}`;
        mesh.material = impactMaterial;
        mesh.pickable = false;
        hide(mesh);
        addToScene(scene, mesh);
        return { mesh, active: false, elapsedMs: 0 };
    });
    let nextProjectile = 0;
    let nextImpactCue = 0;

    const showImpactCue = (point: readonly [number, number, number]): void => {
        const cue = impactCues[nextImpactCue]!;
        nextImpactCue = (nextImpactCue + 1) % impactCues.length;
        cue.active = true;
        cue.elapsedMs = 0;
        cue.mesh.position.set(point[0], point[1], point[2]);
        cue.mesh.scaling.set(1, 1, 1);
        setMeshVisible(cue.mesh, true);
    };

    return {
        fire(mesh, point, distance, range, speed) {
            const originMatrix = viewmodel.localGuideOrigin.worldMatrix;
            const aimMatrix = viewmodel.localGuideYaw.worldMatrix;
            const start: [number, number, number] = [originMatrix[12]!, originMatrix[13]!, originMatrix[14]!];
            const direction = directionFrom(aimMatrix);
            const hitsMesh = mesh !== null && point !== null && distance !== null && distance <= range;
            const end: [number, number, number] = hitsMesh
                ? [point[0], point[1], point[2]]
                : [start[0] + direction[0] * range, start[1] + direction[1] * range, start[2] + direction[2] * range];
            const travelDistance = Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
            const projectile = projectiles[nextProjectile]!;
            nextProjectile = (nextProjectile + 1) % projectiles.length;
            projectile.active = true;
            projectile.elapsedMs = 0;
            projectile.durationMs = Math.max(1, (travelDistance / speed) * 1000);
            projectile.start = start;
            projectile.end = end;
            projectile.impact =
                hitsMesh && mesh && point && distance !== null
                    ? {
                          mesh,
                          point: [point[0], point[1], point[2]],
                          distance,
                          direction: normalize(end[0] - start[0], end[1] - start[1], end[2] - start[2]),
                      }
                    : null;
            projectile.mesh.position.set(...start);
            setMeshVisible(projectile.mesh, true);
        },
        update(deltaMs, isCollisionActive) {
            const stepMs = Math.max(0, deltaMs);
            for (const cue of impactCues) {
                if (!cue.active) continue;
                cue.elapsedMs += stepMs;
                const remaining = 1 - cue.elapsedMs / IMPACT_DURATION_MS;
                if (remaining <= 0) {
                    cue.active = false;
                    hide(cue.mesh);
                } else {
                    const scale = 0.35 + remaining * 0.65;
                    cue.mesh.scaling.set(scale, scale, scale);
                }
            }
            const impacts: PistolImpact[] = [];
            for (const projectile of projectiles) {
                if (!projectile.active) continue;
                projectile.elapsedMs += stepMs;
                const progress = Math.min(1, projectile.elapsedMs / projectile.durationMs);
                projectile.mesh.position.set(
                    projectile.start[0] + (projectile.end[0] - projectile.start[0]) * progress,
                    projectile.start[1] + (projectile.end[1] - projectile.start[1]) * progress,
                    projectile.start[2] + (projectile.end[2] - projectile.start[2]) * progress
                );
                if (progress < 1) continue;
                projectile.active = false;
                hide(projectile.mesh);
                const impact = projectile.impact;
                projectile.impact = null;
                if (impact && (!isCollisionActive || isCollisionActive(impact.mesh))) {
                    showImpactCue(impact.point);
                    impacts.push(impact);
                }
            }
            return impacts;
        },
        clear() {
            for (const projectile of projectiles) {
                projectile.active = false;
                projectile.impact = null;
                hide(projectile.mesh);
            }
            for (const cue of impactCues) {
                cue.active = false;
                hide(cue.mesh);
            }
        },
    };
}

function normalize(x: number, y: number, z: number): [number, number, number] {
    const inverseLength = 1 / (Math.hypot(x, y, z) || 1);
    return [x * inverseLength, y * inverseLength, z * inverseLength];
}
