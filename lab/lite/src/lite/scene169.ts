// Scene 169 — reserved for proper box-projected local cubemap reflections.
// The Standard box remains as a comparison surface until that feature is added.

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
    enablePbrLocalCubemap,
    loadEnvironment,
    registerScene,
    setPbrLocalEnvironment,
    startEngine,
} from "babylon-lite";

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
    await enablePbrLocalCubemap();
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
    const standardBox = createBox(engine, 2);
    standardBox.position.set(-1.6, 1, 0);
    standardBox.material = standardMaterial;
    addToScene(scene, standardBox);

    const pbrMaterial = createPbrMaterial({
        baseColorTexture: createSolidTexture2D(engine, 0.72, 0.74, 0.78, 1),
        ormTexture: createSolidTexture2D(engine, 1, 0.18, 0.82, 1),
        environmentIntensity: 1,
    });
    setPbrLocalEnvironment(pbrMaterial, localEnvironment, {
        projectionPosition: [1.6, 1, 0],
        projectionSize: [4, 3, 4],
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
