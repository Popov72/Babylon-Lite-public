import { describe, expect, it, vi } from "vitest";
import { attachFluidCameraControls } from "../../../../lab/lite/src/demos/fluid/camera-controls";
import { createArcRotateCamera } from "../../../../packages/babylon-lite/src/camera/arc-rotate";
import type { AttachControlOptions } from "../../../../packages/babylon-lite/src/camera/arc-rotate-controls";
import type { SceneContext } from "../../../../packages/babylon-lite/src/scene/scene";

vi.mock("babylon-lite", () => import("../../../../packages/babylon-lite/src/camera/arc-rotate-controls"));

class FakeCanvas extends EventTarget {
    setPointerCapture(_pointerId: number): void {
        return;
    }
    releasePointerCapture(_pointerId: number): void {
        return;
    }
}

function setup(mirrored: boolean, options?: AttachControlOptions) {
    const canvas = new FakeCanvas();
    const camera = createArcRotateCamera(0, 1, 10, { x: 0, y: 0, z: 0 });
    const hooks: SceneContext["_beforeRender"] = [];
    const scene = { _beforeRender: hooks } as SceneContext;
    const detach = attachFluidCameraControls(camera, canvas as HTMLCanvasElement, scene, () => mirrored, options);
    const fire = (type: string, fields: object = {}): void => {
        canvas.dispatchEvent(Object.assign(new Event(type, { cancelable: true }), { button: 0, pointerId: 1, clientX: 0, clientY: 0 }, fields));
    };
    return {
        canvas,
        camera,
        hooks,
        detach,
        fire,
        setMirrored: (value: boolean): void => {
            mirrored = value;
        },
    };
}

describe("Fluid app camera controls", () => {
    it.each([false, true])("preserves existing inertia and vertical input with mirrored=%s", (mirrored) => {
        const { camera, fire, detach } = setup(mirrored);
        const direction = mirrored ? 1 : -1;
        camera.inertialAlphaOffset = 0.4;
        camera.inertialBetaOffset = 0.3;
        camera.inertialPanningX = 0.8;
        camera.inertialPanningY = 0.7;
        fire("pointerdown");
        fire("pointermove", { clientX: 10, clientY: 20 });
        fire("pointermove", { clientX: 20, clientY: 40 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0.4 + direction * 0.02);
        expect(camera.inertialBetaOffset).toBeCloseTo(0.26);
        expect(camera.inertialPanningX).toBeCloseTo(0.8);
        fire("pointerup");
        fire("pointerdown", { button: 2 });
        fire("pointermove", { clientX: 10, clientY: 20 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0.4 + direction * 0.02);
        expect(camera.inertialPanningX).toBeCloseTo(0.8 + direction * 0.2);
        expect(camera.inertialPanningY).toBeCloseTo(1.1);
        detach();
    });

    it("reads mirroring and sensitivities live without reversing accumulated momentum", () => {
        const { camera, hooks, fire, setMirrored, detach } = setup(true);
        fire("pointerdown");
        fire("pointermove", { clientX: 10 });
        hooks[0]!(16);
        expect(camera.alpha).toBeCloseTo(0.01);
        expect(camera.inertialAlphaOffset).toBeCloseTo(0.009);
        camera.angularSensibility = 2000;
        setMirrored(false);
        fire("pointermove", { clientX: 20 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0.004);
        fire("pointerup");
        setMirrored(true);
        fire("pointermove", { clientX: 30 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0.004);
        detach();
    });

    it("preserves deferred picks and gizmo aborts without resurrecting cancelled inertia", () => {
        let pending = true;
        let dragging = false;
        const { camera, fire, detach } = setup(true, {
            isExternalPickPending: () => pending,
            isExternalDragActive: () => dragging,
        });
        camera.inertialAlphaOffset = 0.3;
        camera.inertialPanningX = 0.4;
        fire("pointerdown");
        fire("pointermove", { clientX: 10 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0.3);
        pending = false;
        fire("pointermove", { clientX: 20 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0.31);
        dragging = true;
        fire("pointermove", { clientX: 30 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0);
        expect(camera.inertialPanningX).toBeCloseTo(0);
        dragging = false;
        fire("pointermove", { clientX: 40 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0);
        detach();
    });

    it("keeps claimed gestures, pointer coordinates, wheel and pinch handling unchanged", () => {
        const claimed = vi.fn<NonNullable<AttachControlOptions["shouldHandlePointerDown"]>>(() => false);
        const { camera, fire, detach } = setup(true, { shouldHandlePointerDown: claimed });
        fire("pointerdown", { button: 2, shiftKey: true, clientX: 123 });
        fire("pointermove", { clientX: 133 });
        expect(claimed.mock.calls[0]![0]).toMatchObject({ clientX: 123, shiftKey: true });
        expect(camera.inertialPanningX).toBeCloseTo(0);
        fire("wheel", { deltaY: 120 });
        expect(camera.inertialRadiusOffset).toBeCloseTo(-0.4);
        const touches = (x: number) => [
            { identifier: 1, clientX: 0, clientY: 0 },
            { identifier: 2, clientX: x, clientY: 0 },
        ];
        fire("touchstart", { changedTouches: touches(200) });
        fire("touchmove", { changedTouches: touches(400) });
        expect(camera.radius).toBeCloseTo(5);
        detach();
    });

    it("detaches the stock controls, both app listeners, and the inertia hook together", () => {
        const { canvas, camera, hooks, fire, detach } = setup(true);
        const remove = vi.spyOn(canvas, "removeEventListener");
        expect(hooks).toHaveLength(1);
        detach();
        expect(remove.mock.calls.filter(([type]) => type === "pointermove")).toHaveLength(3);
        expect(hooks).toHaveLength(0);
        detach();
        fire("pointerdown");
        fire("pointermove", { clientX: 10 });
        expect(camera.inertialAlphaOffset).toBeCloseTo(0);
    });
});
