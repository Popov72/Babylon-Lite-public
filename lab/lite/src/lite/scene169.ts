// Scene 169 — Local cubemap reflections
// One Standard and one PBR material use a per-material cubemap while the scene
// keeps the same cubemap as its global fallback environment.

import {
    addToScene,
    createBox,
    createEngine,
    createFreeCamera,
    createHemisphericLight,
    createPbrMaterial,
    createSceneContext,
    createSolidTexture2D,
    createStandardMaterial,
    loadEnvironment,
    registerScene,
    startEngine,
} from "babylon-lite";
import { _enableStandardPrefilteredReflection } from "babylon-lite/material/standard/standard-material.js";

const GLOBAL_ENV_URL = "https://assets.babylonjs.com/core/environments/environmentSpecular.env";
const LOCAL_ENV_URL = "https://playground.babylonjs.com/textures/environment.env";

async function main(): Promise<void> {
    const __initStart = performance.now();
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const engine = await createEngine(canvas);
    const scene = createSceneContext(engine);

    const camera = createFreeCamera({ x: 0, y: 2.2, z: -8 }, { x: 0, y: 1.2, z: 0 });
    scene.camera = camera;

    const light = createHemisphericLight([0, 1, 0]);
    light.intensity = 0;
    addToScene(scene, light);

    const localEnvironment = await loadEnvironment(scene, LOCAL_ENV_URL, {
        skipSkybox: true,
        skipGround: true,
        brdfUrl: "/brdf-lut.png",
    });
    await loadEnvironment(scene, GLOBAL_ENV_URL, {
        skipSkybox: true,
        skipGround: true,
        brdfUrl: "/brdf-lut.png",
    });
    scene.imageProcessing.toneMappingEnabled = false;
    scene.imageProcessing.exposure = 1;
    scene.imageProcessing.contrast = 1;

    const standardMaterial = createStandardMaterial();
    standardMaterial.diffuseColor = [0.18, 0.2, 0.24];
    standardMaterial.specularPower = 96;
    standardMaterial.reflectionLevel = 1;
    standardMaterial.reflectionCubeTexture = localEnvironment;
    _enableStandardPrefilteredReflection(standardMaterial);
    const standardBox = createBox(engine, 2);
    standardBox.position.set(-1.6, 1, 0);
    standardBox.material = standardMaterial;
    addToScene(scene, standardBox);

    const pbrMaterial = createPbrMaterial({
        baseColorTexture: createSolidTexture2D(engine, 0.72, 0.74, 0.78, 1),
        ormTexture: createSolidTexture2D(engine, 1, 0.18, 0.82, 1),
        localEnvironment,
        environmentIntensity: 1,
    });
    const pbrBox = createBox(engine, 2);
    pbrBox.position.set(1.6, 1, 0);
    pbrBox.material = pbrMaterial;
    addToScene(scene, pbrBox);

    await registerScene(scene);
    await startEngine(engine);
    canvas.dataset.drawCalls = String(engine.drawCallCount);
    canvas.dataset.initMs = String(performance.now() - __initStart);
    canvas.dataset.ready = "true";
}

main().catch(console.error);
