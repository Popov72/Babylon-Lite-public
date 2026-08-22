import { describe, expect, it, vi } from "vitest";
import { CharacterSupportedState } from "../../../../packages/babylon-lite/src";
import type { Mesh, PhysicsCharacterController } from "../../../../packages/babylon-lite/src";
import {
    PlayerBehavior,
    playerCapsuleSpawnPosition,
    playerSupportedMovementVelocity,
    playerWeaponSwayMultiplier,
    remainingForwardApertureAssist,
    remainingLateralApertureAssist,
    selectClosestClearApertureOffset,
    shouldUseJumpApertureAssist,
    weaponWheelDirection,
} from "../../../../lab/lite/src/demos/aquanova/behaviors/player";
import {
    CROUCH_CAPSULE_HEIGHT,
    CROUCH_CAPSULE_RADIUS,
    MAX_WALKABLE_SLOPE_COSINE,
    PLAYER_CAPSULE_HEIGHT,
    PLAYER_CAPSULE_RADIUS,
} from "../../../../lab/lite/src/demos/aquanova/constants";
import type { AquanovaGameContext } from "../../../../lab/lite/src/demos/aquanova/behaviors/game-context";

describe("Aquanova player", () => {
    it("defaults to the intended dynamic-body push strength", () => {
        const player = new PlayerBehavior("player", [{} as Mesh], {}, {} as AquanovaGameContext);

        expect((player as unknown as { characterStrength: number }).characterStrength).toBe(10_000);
    });

    it("disables collision for its owning marker when started", () => {
        const emit = vi.fn();
        const context = {
            canvas: { dataset: {} },
            character: {
                characterStrength: 0,
                shapeOptions: { capsuleHeight: PLAYER_CAPSULE_HEIGHT, capsuleRadius: PLAYER_CAPSULE_RADIUS },
            },
            events: {
                emit,
                on: vi.fn(() => () => {}),
            },
            capsuleHeight: PLAYER_CAPSULE_HEIGHT,
            capsuleRadius: PLAYER_CAPSULE_RADIUS,
        } as unknown as AquanovaGameContext;
        const player = new PlayerBehavior("playerMarker", [{} as Mesh], {}, context);
        const internals = player as unknown as {
            createCrosshair(): void;
            listen(): void;
        };
        internals.createCrosshair = vi.fn();
        internals.listen = vi.fn();
        vi.stubGlobal("window", {});
        vi.stubGlobal("document", {});

        try {
            player.start();
            expect(emit).toHaveBeenCalledWith("entityEvent", { name: "playerMarker", event: "disableCollision" });
        } finally {
            player.dispose();
            vi.unstubAllGlobals();
        }
    });

    it("keeps slopes below 45 degrees walkable", () => {
        expect(MAX_WALKABLE_SLOPE_COSINE).toBeCloseTo(Math.cos(Math.PI / 4));
        expect(Math.cos((44.999 * Math.PI) / 180)).toBeGreaterThan(MAX_WALKABLE_SLOPE_COSINE);
        const normal = { x: 0.5, y: Math.sqrt(0.75), z: 0 };
        const idle = playerSupportedMovementVelocity(0, 0, normal);
        const idleNormalSpeed = idle.x * normal.x + idle.y * normal.y;
        expect(Math.hypot(idle.x - normal.x * idleNormalSpeed, idle.y - normal.y * idleNormalSpeed)).toBeCloseTo(0);

        const walking = playerSupportedMovementVelocity(4, 0, normal);
        expect(walking.x * normal.x + walking.y * normal.y).toBeCloseTo(-2);
        expect(walking.x).not.toBeCloseTo(0);
    });

    it("uses 0.3 metre standing and crouched capsule radii", () => {
        expect(PLAYER_CAPSULE_HEIGHT).toBe(1.8);
        expect(PLAYER_CAPSULE_RADIUS).toBe(0.3);
        expect(CROUCH_CAPSULE_HEIGHT).toBe(0.7);
        expect(CROUCH_CAPSULE_RADIUS).toBe(0.3);
        expect(PLAYER_CAPSULE_HEIGHT).toBeGreaterThanOrEqual(PLAYER_CAPSULE_RADIUS * 2);
        expect(CROUCH_CAPSULE_HEIGHT).toBeGreaterThanOrEqual(CROUCH_CAPSULE_RADIUS * 2);
    });

    it("spawns the capsule at the centre of its matching 1.8 metre marker", () => {
        expect(playerCapsuleSpawnPosition([-2.3, 0, 4.7], [-1.7, PLAYER_CAPSULE_HEIGHT, 5.3])).toEqual({
            x: -2,
            y: PLAYER_CAPSULE_HEIGHT / 2,
            z: 5,
        });
    });

    it("auto-crouches only during a forward jump when the aperture requires it", () => {
        const assist = { lateralOffset: 0.05 };
        expect(shouldUseJumpApertureAssist(true, 1, false, assist)).toBe(true);
        expect(shouldUseJumpApertureAssist(false, 1, false, assist)).toBe(false);
        expect(shouldUseJumpApertureAssist(true, 0, false, assist)).toBe(false);
        expect(shouldUseJumpApertureAssist(true, 1, true, assist)).toBe(false);
        expect(shouldUseJumpApertureAssist(true, 1, false, null)).toBe(false);
    });

    it("selects the smallest lateral aperture correction that clears", () => {
        expect(selectClosestClearApertureOffset((offset) => offset === 0)).toBe(0);
        expect(selectClosestClearApertureOffset((offset) => offset >= 0.05)).toBe(0.05);
        expect(selectClosestClearApertureOffset((offset) => offset <= -0.1)).toBe(-0.1);
        expect(selectClosestClearApertureOffset(() => false)).toBeNull();
    });

    it("consumes aperture assist only from resolved movement in the requested direction", () => {
        expect(remainingForwardApertureAssist(0.2, 0)).toBe(0.2);
        expect(remainingForwardApertureAssist(0.2, -0.1)).toBe(0.2);
        expect(remainingForwardApertureAssist(0.2, 0.08)).toBeCloseTo(0.12);
        expect(remainingForwardApertureAssist(0.2, 0.3)).toBe(0);

        expect(remainingLateralApertureAssist(0.1, 0)).toBe(0.1);
        expect(remainingLateralApertureAssist(0.1, -0.05)).toBe(0.1);
        expect(remainingLateralApertureAssist(0.1, 0.04)).toBeCloseTo(0.06);
        expect(remainingLateralApertureAssist(-0.1, -0.2)).toBe(0);
    });

    it("keeps a blocked aperture assist active after touching the sill", () => {
        let supported = true;
        const shapeOptions = { capsuleHeight: PLAYER_CAPSULE_HEIGHT, capsuleRadius: PLAYER_CAPSULE_RADIUS };
        const preserveFootCalls: boolean[] = [];
        const position = { x: 0, y: 1, z: 0 };
        const character = {
            shapeOptions,
            checkSupport: () => ({
                supportedState: supported ? CharacterSupportedState.SUPPORTED : CharacterSupportedState.UNSUPPORTED,
                averageSurfaceNormal: { x: 0, y: 1, z: 0 },
            }),
            setShapeOptions: (options: typeof shapeOptions, preserveFootPosition = true) => {
                shapeOptions.capsuleHeight = options.capsuleHeight;
                shapeOptions.capsuleRadius = options.capsuleRadius;
                preserveFootCalls.push(preserveFootPosition);
            },
            moveWithCollisions: () => {},
            getPosition: () => position,
        } as unknown as PhysicsCharacterController;
        const vector = { set: () => {} };
        const context = {
            canvas: { dataset: {} },
            camera: { position: vector, target: vector },
            character,
            capsuleHeight: PLAYER_CAPSULE_HEIGHT,
            capsuleRadius: PLAYER_CAPSULE_RADIUS,
            eyeHeight: 0.62,
            canStand: () => true,
            jumpApertureAssist: () => ({ lateralOffset: 0.05 }),
        } as unknown as AquanovaGameContext;
        const player = new PlayerBehavior("player", [{} as Mesh], {}, context);
        const state = player as unknown as {
            update(deltaSeconds: number): void;
            apertureAdvanceRemaining: number;
            apertureCrouchActive: boolean;
        };

        player.press("KeyW");
        player.press("Space");
        state.update(1 / 60);
        supported = false;
        state.update(0.4);

        expect(preserveFootCalls).toContain(false);
        expect(state.apertureAdvanceRemaining).toBe(0.2);
        expect(state.apertureCrouchActive).toBe(true);

        supported = true;
        state.update(0.1);
        expect(state.apertureAdvanceRemaining).toBe(0.2);
        expect(state.apertureCrouchActive).toBe(true);
    });

    it("uses idle, walking, and running multipliers from movement input", () => {
        expect(playerWeaponSwayMultiplier(new Set())).toBe(1);
        expect(playerWeaponSwayMultiplier(new Set(["ShiftLeft"]))).toBe(1);
        expect(playerWeaponSwayMultiplier(new Set(["KeyW"]))).toBe(2);
        expect(playerWeaponSwayMultiplier(new Set(["ArrowRight", "ShiftRight"]))).toBe(4);
        expect(playerWeaponSwayMultiplier(new Set(["KeyW", "ShiftLeft"]), true)).toBe(1);
    });

    it("normalizes mouse-wheel weapon-cycle directions", () => {
        expect(weaponWheelDirection(-1)).toBe(-1);
        expect(weaponWheelDirection(120)).toBe(1);
        expect(weaponWheelDirection(0)).toBeNull();
        expect(weaponWheelDirection(Number.NaN)).toBeNull();
    });

    it("plays metallic footsteps only for grounded collision-resolved movement", async () => {
        let supportedState = CharacterSupportedState.SUPPORTED;
        let blocked = false;
        const position = { x: 0, y: 1, z: 0 };
        const shapeOptions = { capsuleHeight: PLAYER_CAPSULE_HEIGHT, capsuleRadius: PLAYER_CAPSULE_RADIUS };
        const sound = { label: "player:stepMetallic", source: "/aquanova/sounds/stepMetallic.mp3", sound: {} };
        const load = vi.fn().mockResolvedValue(sound);
        const play = vi.fn();
        const character = {
            shapeOptions,
            checkSupport: () => ({
                supportedState,
                averageSurfaceNormal: { x: 0, y: 1, z: 0 },
            }),
            moveWithCollisions: (movement: { x: number; y: number; z: number }) => {
                if (blocked) return;
                position.x += movement.x;
                position.y += movement.y;
                position.z += movement.z;
            },
            getPosition: () => position,
        } as unknown as PhysicsCharacterController;
        const vector = { set: () => {} };
        const context = {
            canvas: { dataset: {} },
            camera: { position: vector, target: vector },
            character,
            sounds: { load, play },
            capsuleHeight: PLAYER_CAPSULE_HEIGHT,
            capsuleRadius: PLAYER_CAPSULE_RADIUS,
            eyeHeight: 0.62,
            canStand: () => true,
            jumpApertureAssist: () => null,
        } as unknown as AquanovaGameContext;
        const player = new PlayerBehavior("player", [{} as Mesh], {}, context);
        const state = player as unknown as { update(deltaSeconds: number): void };

        await player.init();
        expect(load).toHaveBeenCalledWith("player:stepMetallic", "/aquanova/sounds/stepMetallic.mp3", { preloadCount: 1 });

        player.press("KeyW");
        state.update(1 / 60);
        expect(play).toHaveBeenCalledTimes(1);
        expect(play).toHaveBeenLastCalledWith(sound, { volume: 2.5 });

        player.release("KeyW");
        state.update(1 / 60);
        supportedState = CharacterSupportedState.UNSUPPORTED;
        player.press("KeyW");
        state.update(1 / 60);
        expect(play).toHaveBeenCalledTimes(1);

        supportedState = CharacterSupportedState.SUPPORTED;
        blocked = true;
        state.update(1 / 60);
        expect(play).toHaveBeenCalledTimes(1);

        blocked = false;
        player.press("ShiftLeft");
        state.update(1 / 60);
        expect(play).toHaveBeenCalledTimes(2);

        player.toggleNoclip();
        state.update(1 / 60);
        expect(play).toHaveBeenCalledTimes(2);
    });
});
