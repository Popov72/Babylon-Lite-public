// Scene 169 — reserved for proper box-projected local cubemap reflections.

import { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture";
import { PBRMaterial } from "@babylonjs/core/Materials/PBR/pbrMaterial";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine";
import { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { Color3 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { CreateBox } from "@babylonjs/core/Meshes/Builders/boxBuilder";
import { Scene } from "@babylonjs/core/scene";

const GLOBAL_ENV_URL = "https://assets.babylonjs.com/core/environments/environmentSpecular.env";
const LOCAL_ENV_URL = "https://playground.babylonjs.com/textures/environment.env";

(async function () {
    const __initStart = performance.now();
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const engine = new WebGPUEngine(canvas, { antialias: true, adaptToDeviceRatio: true });
    await engine.initAsync();

    const scene = new Scene(engine);
    const camera = new FreeCamera("camera", new Vector3(0, 2.2, -8), scene);
    camera.setTarget(new Vector3(0, 1.2, 0));
    scene.activeCamera = camera;

    const light = new HemisphericLight("light", new Vector3(0, 1, 0), scene);
    light.intensity = 0;

    const environment = CubeTexture.CreateFromPrefilteredData(GLOBAL_ENV_URL, scene);
    const localEnvironment = CubeTexture.CreateFromPrefilteredData(LOCAL_ENV_URL, scene);
    localEnvironment.boundingBoxPosition = new Vector3(1.6, 1, 0);
    localEnvironment.boundingBoxSize = new Vector3(4, 3, 4);
    scene.environmentTexture = environment;

    const standardMaterial = new StandardMaterial("standard", scene);
    standardMaterial.diffuseColor = new Color3(0.18, 0.2, 0.24);
    standardMaterial.specularPower = 96;
    const standardBox = CreateBox("standardBox", { size: 2 }, scene);
    standardBox.position.set(-1.6, 1, 0);
    standardBox.material = standardMaterial;

    const pbrMaterial = new PBRMaterial("pbr", scene);
    pbrMaterial.albedoColor = new Color3(0.72, 0.74, 0.78);
    pbrMaterial.metallic = 0.82;
    pbrMaterial.roughness = 0.18;
    pbrMaterial.reflectionTexture = localEnvironment;
    const pbrBox = CreateBox("pbrBox", { size: 2 }, scene);
    pbrBox.position.set(1.6, 1, 0);
    pbrBox.material = pbrMaterial;

    const eng = engine as unknown as { _drawCalls?: { fetchNewFrame(): void; current: number } };
    scene.onBeforeRenderObservable.add(() => eng._drawCalls?.fetchNewFrame());
    scene.onAfterRenderObservable.add(() => {
        canvas.dataset.drawCalls = String(eng._drawCalls?.current ?? 0);
    });
    await scene.whenReadyAsync();
    engine.runRenderLoop(() => scene.render());
    window.addEventListener("resize", () => engine.resize());
    await new Promise<void>((resolve) => scene.onAfterRenderObservable.addOnce(() => resolve()));
    canvas.dataset.initMs = String(performance.now() - __initStart);
    canvas.dataset.ready = "true";
})().catch(console.error);
