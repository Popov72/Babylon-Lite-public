import { describe, expect, it } from "vitest";
import { CharacterSupportedState } from "../../../packages/babylon-lite/src";
import type { Mesh, PhysicsCharacterController } from "../../../packages/babylon-lite/src";
import {
    PlayerBehavior,
    playerSupportedMovementVelocity,
    playerWeaponSwayMultiplier,
    remainingForwardApertureAssist,
    remainingLateralApertureAssist,
    selectClosestClearApertureOffset,
    shouldUseJumpApertureAssist,
    weaponWheelDirection,
} from "../../../lab/lite/src/demos/aquanova/behaviors/player";
import { CROUCH_CAPSULE_HEIGHT, CROUCH_CAPSULE_RADIUS, MAX_WALKABLE_SLOPE_COSINE } from "../../../lab/lite/src/demos/aquanova/constants";
import type { BehaviorContext } from "../../../lab/lite/src/demos/aquanova/behaviors/types";

describe("Aquanova player", () => {
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

    it("uses a valid 0.7 metre crouched capsule", () => {
        expect(CROUCH_CAPSULE_HEIGHT).toBe(0.7);
        expect(CROUCH_CAPSULE_RADIUS).toBe(0.35);
        expect(CROUCH_CAPSULE_HEIGHT).toBeGreaterThanOrEqual(CROUCH_CAPSULE_RADIUS * 2);
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
        const shapeOptions = { capsuleHeight: 1.8, capsuleRadius: 0.4 };
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
            capsuleHeight: 1.8,
            capsuleRadius: 0.4,
            eyeHeight: 0.62,
            canStand: () => true,
            jumpApertureAssist: () => ({ lateralOffset: 0.05 }),
        } as unknown as BehaviorContext;
        const player = new PlayerBehavior({} as Mesh, {}, context);
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
});
