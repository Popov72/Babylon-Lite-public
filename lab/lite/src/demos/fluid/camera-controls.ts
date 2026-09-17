import { attachControl } from "babylon-lite";
import type { ArcRotateCamera, AttachControlOptions, SceneContext } from "babylon-lite";

export function attachFluidCameraControls(
    camera: ArcRotateCamera,
    canvas: HTMLCanvasElement,
    scene: SceneContext,
    isMirrored: () => boolean,
    options?: AttachControlOptions
): () => void {
    let mirroredMove = false;
    function reverseHorizontalInertia(): void {
        camera.inertialAlphaOffset = -camera.inertialAlphaOffset;
        camera.inertialPanningX = -camera.inertialPanningX;
    }
    function beforePointerMove(): void {
        mirroredMove = isMirrored();
        if (mirroredMove) {
            reverseHorizontalInertia();
        }
    }
    function afterPointerMove(): void {
        if (mirroredMove) {
            mirroredMove = false;
            reverseHorizontalInertia();
        }
    }

    // Paired conversions preserve existing inertia and core aborts, reversing only new input.
    canvas.addEventListener("pointermove", beforePointerMove);
    const detach = attachControl(camera, canvas, scene, options);
    canvas.addEventListener("pointermove", afterPointerMove);
    return () => {
        afterPointerMove();
        canvas.removeEventListener("pointermove", beforePointerMove);
        detach();
        canvas.removeEventListener("pointermove", afterPointerMove);
    };
}
