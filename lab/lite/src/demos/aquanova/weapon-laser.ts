import {
    addBillboardSpriteIndex,
    addFacingBillboardSystem,
    billboardBlendAdditive,
    clearBillboardSprites,
    createFacingBillboardSystem,
    createGridSpriteAtlas,
    createTexture2DFromPixels,
} from "babylon-lite";
import type { EngineContext, FacingBillboardSpriteSystem, SceneContext } from "babylon-lite";

const CORE_CAPACITY = 640;
const SHEATH_CAPACITY = 320;
const SPIRAL_CAPACITY = 128;
const SPIRAL_COUNT = 6;
const HALO_CAPACITY = 112;
const RING_CAPACITY = 144;
const SPARK_CAPACITY = 160;
const FLARE_CAPACITY = 14;
const LASER_SPEED = 35;
const MAX_TRAVEL_SECONDS = 1;

export interface WeaponLaserAim {
    readonly origin: readonly [number, number, number];
    readonly direction: readonly [number, number, number];
}

export interface WeaponParticleLaser {
    setTargetDistance(distance: number | null, restart?: boolean): void;
    stop(): void;
    update(deltaMs: number, aim: WeaponLaserAim | null, muzzleWorldMatrix: ArrayLike<number>): boolean;
}

interface BeamFrame {
    readonly startX: number;
    readonly startY: number;
    readonly startZ: number;
    readonly dirX: number;
    readonly dirY: number;
    readonly dirZ: number;
    readonly sideX: number;
    readonly sideY: number;
    readonly sideZ: number;
    readonly upX: number;
    readonly upY: number;
    readonly upZ: number;
    readonly length: number;
}

function createGlowPixels(size: number): Uint8Array {
    const pixels = new Uint8Array(size * size * 4);
    const centre = (size - 1) * 0.5;
    const radius = size * 0.5;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dx = (x - centre) / radius;
            const dy = (y - centre) / radius;
            const d = Math.sqrt(dx * dx + dy * dy);
            const glow = Math.max(0, 1 - d);
            const alpha = Math.pow(glow, 2.2);
            const offset = (y * size + x) * 4;
            pixels[offset] = 255;
            pixels[offset + 1] = 255;
            pixels[offset + 2] = 255;
            pixels[offset + 3] = Math.round(alpha * 255);
        }
    }
    return pixels;
}

function createLayer(scene: SceneContext, atlas: ReturnType<typeof createGridSpriteAtlas>, capacity: number): FacingBillboardSpriteSystem {
    const layer = createFacingBillboardSystem(atlas, { capacity, blendMode: billboardBlendAdditive });
    addFacingBillboardSystem(scene, layer);
    return layer;
}

function fract(value: number): number {
    return value - Math.floor(value);
}

function hash(value: number): number {
    return fract(Math.sin(value * 127.1 + 311.7) * 43758.5453123);
}

function clearLayers(layers: readonly FacingBillboardSpriteSystem[]): void {
    for (const layer of layers) clearBillboardSprites(layer);
}

function resolveBeamFrame(aim: WeaponLaserAim, distance: number, muzzleWorldMatrix: ArrayLike<number>): BeamFrame | null {
    const startX = muzzleWorldMatrix[12]!;
    const startY = muzzleWorldMatrix[13]!;
    const startZ = muzzleWorldMatrix[14]!;
    const targetX = aim.origin[0] + aim.direction[0] * distance;
    const targetY = aim.origin[1] + aim.direction[1] * distance;
    const targetZ = aim.origin[2] + aim.direction[2] * distance;
    const deltaX = targetX - startX;
    const deltaY = targetY - startY;
    const deltaZ = targetZ - startZ;
    const length = Math.hypot(deltaX, deltaY, deltaZ);
    if (length < 1e-6) return null;

    const dirX = deltaX / length;
    const dirY = deltaY / length;
    const dirZ = deltaZ / length;
    const helperX = Math.abs(dirY) > 0.92 ? 1 : 0;
    const helperY = Math.abs(dirY) > 0.92 ? 0 : 1;
    let sideX = helperY * dirZ;
    let sideY = -helperX * dirZ;
    let sideZ = helperX * dirY - helperY * dirX;
    const sideLength = Math.hypot(sideX, sideY, sideZ);
    sideX /= sideLength;
    sideY /= sideLength;
    sideZ /= sideLength;
    const upX = dirY * sideZ - dirZ * sideY;
    const upY = dirZ * sideX - dirX * sideZ;
    const upZ = dirX * sideY - dirY * sideX;

    return { startX, startY, startZ, dirX, dirY, dirZ, sideX, sideY, sideZ, upX, upY, upZ, length };
}

export function createWeaponParticleLaser(engine: EngineContext, scene: SceneContext): WeaponParticleLaser {
    const textureSize = 64;
    const texture = createTexture2DFromPixels(engine, createGlowPixels(textureSize), textureSize, textureSize, {
        minFilter: "linear",
        magFilter: "linear",
    });
    const atlas = createGridSpriteAtlas(texture, { cellWidthPx: textureSize, cellHeightPx: textureSize });
    const core = createLayer(scene, atlas, CORE_CAPACITY);
    const sheath = createLayer(scene, atlas, SHEATH_CAPACITY);
    const spirals = Array.from({ length: SPIRAL_COUNT }, () => createLayer(scene, atlas, SPIRAL_CAPACITY));
    const halo = createLayer(scene, atlas, HALO_CAPACITY);
    const rings = createLayer(scene, atlas, RING_CAPACITY);
    const sparks = createLayer(scene, atlas, SPARK_CAPACITY);
    const flares = createLayer(scene, atlas, FLARE_CAPACITY);
    const layers = [core, sheath, ...spirals, halo, rings, sparks, flares];

    const position: [number, number, number] = [0, 0, 0];
    const size: [number, number] = [0, 0];
    const color: [number, number, number, number] = [1, 1, 1, 1];
    const sprite = { position, sizeWorld: size, color, frame: 0, rotation: 0 };
    let targetDistance: number | null = null;
    let elapsedSeconds = 0;
    let effectSeconds = 0;

    const addParticle = (
        layer: FacingBillboardSpriteSystem,
        x: number,
        y: number,
        z: number,
        width: number,
        height: number,
        red: number,
        green: number,
        blue: number,
        alpha: number,
        rotation = 0
    ): void => {
        position[0] = x;
        position[1] = y;
        position[2] = z;
        size[0] = width;
        size[1] = height;
        color[0] = red;
        color[1] = green;
        color[2] = blue;
        color[3] = alpha;
        sprite.rotation = rotation;
        addBillboardSpriteIndex(layer, sprite);
    };

    const stop = (): void => {
        targetDistance = null;
        elapsedSeconds = 0;
        clearLayers(layers);
    };

    const setTargetDistance = (distance: number | null, restart = false): void => {
        if (distance === null || !Number.isFinite(distance) || distance <= 0) {
            stop();
            return;
        }
        if (targetDistance === null || restart) elapsedSeconds = 0;
        targetDistance = distance;
    };

    const update = (deltaMs: number, aim: WeaponLaserAim | null, muzzleWorldMatrix: ArrayLike<number>): boolean => {
        if (targetDistance === null || !aim) {
            clearLayers(layers);
            return false;
        }
        const beam = resolveBeamFrame(aim, targetDistance, muzzleWorldMatrix);
        if (!beam) {
            clearLayers(layers);
            return false;
        }

        const deltaSeconds = Math.max(0, Math.min(deltaMs * 0.001, 0.1));
        elapsedSeconds += deltaSeconds;
        effectSeconds += deltaSeconds;
        const travelSeconds = Math.min(MAX_TRAVEL_SECONDS, beam.length / LASER_SPEED);
        const reachedLength = travelSeconds > 0 ? beam.length * Math.min(1, elapsedSeconds / travelSeconds) : beam.length;
        clearLayers(layers);

        const sheathCount = Math.min(SHEATH_CAPACITY, Math.max(3, Math.ceil(reachedLength / 0.1)));
        for (let i = 0; i < sheathCount; i++) {
            const along = reachedLength * (i / Math.max(1, sheathCount - 1));
            const pulse = 0.8 + Math.sin(along * 3.2 - effectSeconds * 14) * 0.2;
            addParticle(
                sheath,
                beam.startX + beam.dirX * along,
                beam.startY + beam.dirY * along,
                beam.startZ + beam.dirZ * along,
                0.13 + pulse * 0.025,
                0.13 + pulse * 0.025,
                0.04,
                0.5,
                1,
                0.16 + pulse * 0.08
            );
        }

        const coreCount = Math.min(CORE_CAPACITY, Math.max(3, Math.ceil(reachedLength / 0.052)));
        for (let i = 0; i < coreCount; i++) {
            const along = reachedLength * (i / Math.max(1, coreCount - 1));
            const pulse = 0.84 + Math.sin(along * 5.2 - effectSeconds * 22) * 0.16;
            addParticle(core, beam.startX + beam.dirX * along, beam.startY + beam.dirY * along, beam.startZ + beam.dirZ * along, 0.058, 0.058, 0.76, 0.99, 1, pulse * 0.82);
        }

        const drawSpiral = (layer: FacingBillboardSpriteSystem, strand: number): void => {
            const direction = strand % 2 === 0 ? 1 : -1;
            const phaseOffset = (strand / SPIRAL_COUNT) * Math.PI * 2;
            const count = Math.min(SPIRAL_CAPACITY, Math.max(10, Math.ceil(reachedLength / 0.18)));
            for (let i = 0; i < count; i++) {
                const along = reachedLength * (i / Math.max(1, count - 1));
                const phase = along * (1.5 + strand * 0.08) * direction + effectSeconds * (6.8 + strand * 0.42) * direction + phaseOffset;
                const radiusWave = 0.5 + 0.5 * Math.sin(along * (0.34 + strand * 0.025) - effectSeconds * (2.1 + strand * 0.13) + phaseOffset);
                const radius = 0.12 + strand * 0.008 + radiusWave * 0.19;
                const cos = Math.cos(phase);
                const sin = Math.sin(phase);
                const offsetX = beam.sideX * cos * radius + beam.upX * sin * radius;
                const offsetY = beam.sideY * cos * radius + beam.upY * sin * radius;
                const offsetZ = beam.sideZ * cos * radius + beam.upZ * sin * radius;
                const strandPulse = 0.74 + Math.sin(along * 2.8 - effectSeconds * 13 + phaseOffset) * 0.2;
                addParticle(
                    layer,
                    beam.startX + beam.dirX * along + offsetX,
                    beam.startY + beam.dirY * along + offsetY,
                    beam.startZ + beam.dirZ * along + offsetZ,
                    0.052 + (strand % 3) * 0.008,
                    0.052 + (strand % 3) * 0.008,
                    0.08 + (strand % 3) * 0.07,
                    0.68 + (strand % 2) * 0.16,
                    1,
                    strandPulse
                );
            }
        };
        for (let strand = 0; strand < spirals.length; strand++) {
            drawSpiral(spirals[strand]!, strand);
        }

        const haloCount = Math.min(HALO_CAPACITY, Math.max(8, Math.ceil(reachedLength * 2.25)));
        for (let i = 0; i < haloCount; i++) {
            const along = reachedLength * fract(hash(i + 31) + effectSeconds * (0.035 + hash(i + 7) * 0.025));
            const phase = hash(i + 73) * Math.PI * 2 + effectSeconds * (0.9 + hash(i + 19) * 1.4);
            const radius = 0.1 + hash(i + 113) * 0.3;
            const cos = Math.cos(phase);
            const sin = Math.sin(phase);
            const offsetX = beam.sideX * cos * radius + beam.upX * sin * radius;
            const offsetY = beam.sideY * cos * radius + beam.upY * sin * radius;
            const offsetZ = beam.sideZ * cos * radius + beam.upZ * sin * radius;
            const haloSize = 0.11 + hash(i + 149) * 0.18;
            addParticle(
                halo,
                beam.startX + beam.dirX * along + offsetX,
                beam.startY + beam.dirY * along + offsetY,
                beam.startZ + beam.dirZ * along + offsetZ,
                haloSize,
                haloSize,
                0.04,
                0.56,
                1,
                0.12 + hash(i + 163) * 0.11
            );
        }

        const ringCount = Math.min(10, Math.max(2, Math.ceil(reachedLength / 3.2)));
        const particlesPerRing = Math.floor(RING_CAPACITY / ringCount);
        for (let ring = 0; ring < ringCount; ring++) {
            const along = reachedLength * ((ring + 0.72) / (ringCount + 0.45));
            const ringPhase = effectSeconds * (3.6 + ring * 0.19) * (ring % 2 === 0 ? 1 : -1) + ring * 1.7;
            const ringRadius = 0.26 + Math.sin(effectSeconds * 8 + ring * 2.1) * 0.045;
            for (let i = 0; i < particlesPerRing; i++) {
                const phase = ringPhase + (i / particlesPerRing) * Math.PI * 2;
                const cos = Math.cos(phase);
                const sin = Math.sin(phase);
                const offsetX = beam.sideX * cos * ringRadius + beam.upX * sin * ringRadius;
                const offsetY = beam.sideY * cos * ringRadius + beam.upY * sin * ringRadius;
                const offsetZ = beam.sideZ * cos * ringRadius + beam.upZ * sin * ringRadius;
                const ringPulse = 0.55 + Math.sin(effectSeconds * 15 - ring * 1.4 + i * 0.55) * 0.25;
                addParticle(
                    rings,
                    beam.startX + beam.dirX * along + offsetX,
                    beam.startY + beam.dirY * along + offsetY,
                    beam.startZ + beam.dirZ * along + offsetZ,
                    0.045,
                    0.09,
                    0.16,
                    0.82,
                    1,
                    ringPulse,
                    phase
                );
            }
        }

        const sparkCount = Math.min(SPARK_CAPACITY, Math.max(18, Math.ceil(reachedLength * 3.4)));
        for (let i = 0; i < sparkCount; i++) {
            const speed = 0.16 + hash(i + 211) * 0.24;
            const along = reachedLength * fract(hash(i + 181) + effectSeconds * speed);
            const phase = hash(i + 233) * Math.PI * 2 + effectSeconds * (2.4 + hash(i + 251) * 4.2);
            const radius = 0.12 + hash(i + 271) * 0.46;
            const cos = Math.cos(phase);
            const sin = Math.sin(phase);
            const offsetX = beam.sideX * cos * radius + beam.upX * sin * radius;
            const offsetY = beam.sideY * cos * radius + beam.upY * sin * radius;
            const offsetZ = beam.sideZ * cos * radius + beam.upZ * sin * radius;
            const sparkSize = 0.024 + hash(i + 293) * 0.045;
            addParticle(
                sparks,
                beam.startX + beam.dirX * along + offsetX,
                beam.startY + beam.dirY * along + offsetY,
                beam.startZ + beam.dirZ * along + offsetZ,
                sparkSize,
                sparkSize * (1.4 + hash(i + 307) * 2.5),
                0.38,
                0.9,
                1,
                0.66 + hash(i + 331) * 0.34,
                phase
            );
        }

        const headX = beam.startX + beam.dirX * reachedLength;
        const headY = beam.startY + beam.dirY * reachedLength;
        const headZ = beam.startZ + beam.dirZ * reachedLength;
        const pulse = 1 + Math.sin(effectSeconds * 24) * 0.14;
        addParticle(flares, beam.startX, beam.startY, beam.startZ, 0.28 * pulse, 0.28 * pulse, 0.08, 0.62, 1, 0.28);
        addParticle(flares, beam.startX, beam.startY, beam.startZ, 0.15, 0.15, 0.48, 0.98, 1, 0.9);
        addParticle(flares, headX, headY, headZ, 0.52 * pulse, 0.52 * pulse, 0.06, 0.55, 1, 0.3);
        addParticle(flares, headX, headY, headZ, 0.31 * pulse, 0.31 * pulse, 0.52, 0.98, 1, 0.86);
        addParticle(flares, headX, headY, headZ, 0.14, 0.14, 1, 1, 1, 1);
        for (let i = 0; i < 8; i++) {
            const phase = effectSeconds * (5.5 + (i % 2) * 1.2) * (i % 2 === 0 ? 1 : -1) + (i / 8) * Math.PI * 2;
            const radius = 0.16 + (i % 3) * 0.035;
            const cos = Math.cos(phase);
            const sin = Math.sin(phase);
            addParticle(
                flares,
                headX + beam.sideX * cos * radius + beam.upX * sin * radius,
                headY + beam.sideY * cos * radius + beam.upY * sin * radius,
                headZ + beam.sideZ * cos * radius + beam.upZ * sin * radius,
                0.045,
                0.09,
                0.18,
                0.86,
                1,
                0.84,
                phase
            );
        }
        return travelSeconds <= 0 || elapsedSeconds >= travelSeconds;
    };

    return { setTargetDistance, stop, update };
}
