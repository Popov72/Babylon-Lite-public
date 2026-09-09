import {
    addBillboardSpriteIndex,
    addFacingBillboardSystem,
    billboardBlendAdditive,
    clearBillboardSprites,
    createFacingBillboardSystem,
    createGridSpriteAtlas,
    createTexture2DFromPixels,
    onSceneDispose,
} from "babylon-lite";
import type { EngineContext, FacingBillboardSpriteSystem, Mesh, SceneContext } from "babylon-lite";
import type { SparkOptions, SparkRegistration, SparkRuntime } from "./game-context.js";

const CAPACITY = 1024;
const TEXTURE_SIZE = 32;

interface SparkEmitter {
    readonly id: number;
    readonly mesh: Mesh;
    readonly options: SparkOptions;
    spawnCarry: number;
    displayed: boolean;
}

interface SparkParticle {
    readonly emitterId: number;
    x: number;
    y: number;
    z: number;
    vx: number;
    vy: number;
    vz: number;
    age: number;
    readonly lifetime: number;
    readonly size: number;
    readonly rotation: number;
    readonly rotationSpeed: number;
}

function createSparkPixels(): Uint8Array {
    const pixels = new Uint8Array(TEXTURE_SIZE * TEXTURE_SIZE * 4);
    const centre = (TEXTURE_SIZE - 1) * 0.5;
    const radius = TEXTURE_SIZE * 0.5;
    for (let y = 0; y < TEXTURE_SIZE; y++) {
        for (let x = 0; x < TEXTURE_SIZE; x++) {
            const dx = (x - centre) / radius;
            const dy = (y - centre) / radius;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const glow = Math.max(0, 1 - distance);
            const alpha = Math.pow(glow, 2.5);
            const offset = (y * TEXTURE_SIZE + x) * 4;
            pixels[offset] = 255;
            pixels[offset + 1] = 255;
            pixels[offset + 2] = 255;
            pixels[offset + 3] = Math.round(alpha * 255);
        }
    }
    return pixels;
}

export function sparkEmitterDisplayed(mesh: Mesh, isPortalCulled: (mesh: Mesh) => boolean): boolean {
    return !isPortalCulled(mesh);
}

export class AquanovaSparkRuntime implements SparkRuntime {
    private readonly emitters = new Map<number, SparkEmitter>();
    private particles: SparkParticle[] = [];
    private layer: FacingBillboardSpriteSystem | null = null;
    private nextEmitterId = 1;

    public constructor(
        private readonly engine: EngineContext,
        private readonly scene: SceneContext,
        private readonly isPortalCulled: (mesh: Mesh) => boolean
    ) {
        onSceneDispose(scene, () => {
            this.emitters.clear();
            this.particles.length = 0;
            this.layer = null;
        });
    }

    public register(mesh: Mesh, options: SparkOptions): SparkRegistration {
        this.ensureLayer();
        const id = this.nextEmitterId++;
        const displayed = sparkEmitterDisplayed(mesh, this.isPortalCulled);
        this.emitters.set(id, { id, mesh, options, spawnCarry: displayed ? 1 : 0, displayed });
        let disposed = false;
        return {
            dispose: () => {
                if (disposed) {
                    return;
                }
                disposed = true;
                this.emitters.delete(id);
                this.particles = this.particles.filter((particle) => particle.emitterId !== id);
            },
        };
    }

    private ensureLayer(): FacingBillboardSpriteSystem {
        if (this.layer) {
            return this.layer;
        }
        const texture = createTexture2DFromPixels(this.engine, createSparkPixels(), TEXTURE_SIZE, TEXTURE_SIZE, {
            minFilter: "linear",
            magFilter: "linear",
        });
        const atlas = createGridSpriteAtlas(texture, { cellWidthPx: TEXTURE_SIZE, cellHeightPx: TEXTURE_SIZE });
        this.layer = createFacingBillboardSystem(atlas, { capacity: CAPACITY, blendMode: billboardBlendAdditive });
        addFacingBillboardSystem(this.scene, this.layer);
        return this.layer;
    }

    public update(deltaMs: number): void {
        if (!this.layer) {
            return;
        }
        const deltaSeconds = Math.max(0, Math.min(deltaMs * 0.001, 0.05));
        this.updateEmitterVisibility();
        this.updateParticles(deltaSeconds);
        this.emitParticles(deltaSeconds);
        this.drawParticles();
    }

    private updateEmitterVisibility(): void {
        const hiddenEmitterIds = new Set<number>();
        for (const emitter of this.emitters.values()) {
            const displayed = sparkEmitterDisplayed(emitter.mesh, this.isPortalCulled);
            if (displayed === emitter.displayed) {
                continue;
            }
            emitter.displayed = displayed;
            emitter.spawnCarry = displayed ? 1 : 0;
            if (!displayed) {
                hiddenEmitterIds.add(emitter.id);
            }
        }
        if (hiddenEmitterIds.size > 0) {
            this.particles = this.particles.filter((particle) => !hiddenEmitterIds.has(particle.emitterId));
        }
    }

    private updateParticles(deltaSeconds: number): void {
        for (let index = this.particles.length - 1; index >= 0; index--) {
            const particle = this.particles[index]!;
            const emitter = this.emitters.get(particle.emitterId);
            if (!emitter) {
                this.particles.splice(index, 1);
                continue;
            }
            particle.age += deltaSeconds;
            if (particle.age >= particle.lifetime) {
                this.particles.splice(index, 1);
                continue;
            }
            particle.vy -= emitter.options.gravity * deltaSeconds;
            particle.x += particle.vx * deltaSeconds;
            particle.y += particle.vy * deltaSeconds;
            particle.z += particle.vz * deltaSeconds;
        }
    }

    private emitParticles(deltaSeconds: number): void {
        for (const emitter of this.emitters.values()) {
            if (!emitter.displayed) {
                continue;
            }
            emitter.spawnCarry += emitter.options.rate * deltaSeconds;
            while (emitter.spawnCarry >= 1 && this.particles.length < CAPACITY) {
                emitter.spawnCarry -= 1;
                this.spawnParticle(emitter);
            }
        }
    }

    private spawnParticle(emitter: SparkEmitter): void {
        const matrix = emitter.mesh.worldMatrix;
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.sqrt(Math.random()) * emitter.options.spread;
        const speed = emitter.options.speed * (0.65 + Math.random() * 0.7);
        const horizontalSpeed = speed * (0.35 + Math.random() * 0.45);
        this.particles.push({
            emitterId: emitter.id,
            x: matrix[12]! + Math.cos(angle) * radius,
            y: matrix[13]! + (Math.random() * 2 - 1) * emitter.options.spread,
            z: matrix[14]! + Math.sin(angle) * radius,
            vx: Math.cos(angle) * horizontalSpeed,
            vy: speed * (0.55 + Math.random() * 0.65),
            vz: Math.sin(angle) * horizontalSpeed,
            age: 0,
            lifetime: emitter.options.lifetime * (0.7 + Math.random() * 0.6),
            size: emitter.options.size * (0.75 + Math.random() * 0.5),
            rotation: Math.random() * Math.PI,
            rotationSpeed: (Math.random() * 2 - 1) * 8,
        });
    }

    private drawParticles(): void {
        const layer = this.layer!;
        clearBillboardSprites(layer);
        for (const particle of this.particles) {
            if (!this.emitters.get(particle.emitterId)?.displayed) {
                continue;
            }
            const progress = particle.age / particle.lifetime;
            const fade = Math.pow(1 - progress, 1.5);
            const size = particle.size * (1 - progress * 0.45);
            addBillboardSpriteIndex(layer, {
                position: [particle.x, particle.y, particle.z],
                sizeWorld: [size, size * 2.8],
                color: [1, 0.9 - progress * 0.55, 0.45 - progress * 0.4, fade],
                frame: 0,
                rotation: particle.rotation + particle.rotationSpeed * particle.age,
            });
        }
    }
}
