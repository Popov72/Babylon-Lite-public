// Scene 316: an evolving cloud bank around opaque PBR geometry.

import {
    addToScene,
    attachControl,
    createArcRotateCamera,
    createBox,
    createDirectionalLight,
    createEngine,
    createGround,
    createPbrMaterial,
    createSceneContext,
    enableVolumetricFog,
    onBeforeRender,
    onSceneDispose,
    registerScene,
    startEngine,
} from "babylon-lite";

const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
const params = new URLSearchParams(location.search);
const msaaSamples = params.get("msaa") === "1" ? 1 : 4;

async function main(): Promise<void> {
    let seekTime: number | null = null;
    const seekTimeParam = params.get("seekTime");
    if (seekTimeParam !== null) {
        const parsed = Number(seekTimeParam);
        if (seekTimeParam.trim() === "" || !Number.isFinite(parsed) || parsed < 0) {
            throw new RangeError("Scene 316: seekTime must be finite, nonnegative seconds.");
        }
        seekTime = parsed;
    }
    const volumeSize = params.get("volumeSize") ?? "standard";
    let tileSize: number;
    let depthSlices: number;
    switch (volumeSize) {
        case "low":
            tileSize = 16;
            depthSlices = 64;
            break;
        case "standard":
            tileSize = 8;
            depthSlices = 80;
            break;
        case "depth":
            tileSize = 8;
            depthSlices = 160;
            break;
        case "fine":
            tileSize = 4;
            depthSlices = 80;
            break;
        default:
            throw new RangeError(`Scene 316: unknown 3D texture size "${volumeSize}".`);
    }
    const animationSetting = params.get("animate");
    if (animationSetting !== null && animationSetting !== "off") {
        throw new RangeError('Scene 316: animate must be "off" when provided.');
    }
    const scrollSetting = params.get("scroll");
    if (scrollSetting !== null && scrollSetting !== "off") {
        throw new RangeError('Scene 316: scroll must be "off" when provided.');
    }
    let animating = animationSetting !== "off" && scrollSetting !== "off";
    let animationSeconds = 0;
    const engine = await createEngine(canvas, { msaaSamples, maxDevicePixelRatio: 1 });
    const watchGpuErrors = () =>
        engine._device.addEventListener("uncapturederror", (event) => {
            canvas.dataset.error = event.error.message;
        });
    watchGpuErrors();
    engine._device.pushErrorScope("validation");
    if (params.get("recovery") === "1") {
        const { enableDeviceLostSceneRecovery, forceWebGpuDeviceLossForTesting } = await import("../shared/scene316-recovery.js");
        enableDeviceLostSceneRecovery(engine, {
            onRecovered() {
                watchGpuErrors();
                canvas.dataset.recovered = "true";
            },
            onRecoveryFailed(error) {
                canvas.dataset.error = error instanceof Error ? error.message : String(error);
            },
        });
        const button = document.getElementById("deviceLoss") as HTMLButtonElement;
        button.hidden = false;
        button.addEventListener("click", () => {
            canvas.dataset.recovered = "false";
            forceWebGpuDeviceLossForTesting(engine);
        });
    }

    const scene = createSceneContext(engine);
    scene.clearColor = { r: 0.006, g: 0.007, b: 0.008, a: 1 };
    scene.imageProcessing.toneMappingEnabled = true;
    const initialAlpha = -Math.PI / 2;
    const initialBeta = Math.atan2(14, 0.55);
    const initialRadius = Math.hypot(14, 0.55);
    const camera = createArcRotateCamera(initialAlpha, initialBeta, initialRadius, { x: 0, y: 0.7, z: 4 });
    camera.nearPlane = 0.1;
    camera.farPlane = 60;
    scene.camera = camera;
    onSceneDispose(scene, attachControl(camera, canvas, scene, { keyboard: false }));

    const sun = createDirectionalLight([0.3, -1, 0.15], 1.8);
    addToScene(scene, sun);

    const setUnlit = params.get("unlit") === "1" ? (await import("../shared/scene316-unlit.js")).setPbrUnlit : undefined;
    const addMeshes = (): void => {
        const ground = createGround(engine, { width: 50, height: 50, subdivisions: 1 });
        ground.material = createPbrMaterial({ baseColorFactor: [0.035, 0.04, 0.045, 1], metallicFactor: 0, roughnessFactor: 1 });
        addToScene(scene, ground);

        const colors: [number, number, number][] = [
            [0.85, 0.2, 0.13],
            [0.13, 0.65, 0.85],
            [0.85, 0.75, 0.22],
        ];
        for (let i = 0; i < colors.length; i++) {
            const box = createBox(engine, 2.2);
            const material = createPbrMaterial({ baseColorFactor: [...colors[i]!, 1], metallicFactor: 0, roughnessFactor: 0.8 });
            if (i === 0) {
                setUnlit?.(material, [0.3, 1, 0.8]);
            }
            box.material = material;
            box.position.set((i - 1) * 3.2, 1.1, 1 + i * 3);
            addToScene(scene, box);
        }
    };
    const early = params.get("early") === "1";
    if (!early) {
        addMeshes();
    }

    if (params.get("transmission") === "before") {
        const { enableSceneTransmission } = await import("../shared/scene316-transmission.js");
        enableSceneTransmission(scene, engine);
    }
    const baseDensity = params.has("density") ? Number(params.get("density")) : 0.28;
    const cloudState = { time: 0 };
    let fog: ReturnType<typeof enableVolumetricFog> | null = null;
    const shapeButton = document.getElementById("shapeToggle") as HTMLButtonElement;
    const volumeControl = document.getElementById("volumeControl")!;
    const volumeSelect = document.getElementById("volumeSize") as HTMLSelectElement;
    const volumeDimensions = document.getElementById("volumeDims") as HTMLOutputElement;
    if (params.get("baseline") === "1") {
        document.getElementById("fogToggle")!.remove();
        shapeButton.remove();
        volumeControl.remove();
    } else {
        const { createScene316CloudDensitySource } = await import("../shared/scene316-cloud-density.js");
        const activeFog = enableVolumetricFog(scene, engine, {
            density: baseDensity,
            maxDensity: 1.4,
            albedo: [1, 1, 1],
            ambientRadiance: [1.8, 1.8, 1.8],
            sun,
            anisotropy: 0.35,
            maxDistance: 35,
            tileSize,
            depthSlices,
            densitySource: createScene316CloudDensitySource(cloudState),
        });
        fog = activeFog;
        canvas.dataset.fogDensity = baseDensity.toFixed(4);
        document.getElementById("fogToggle")!.addEventListener("click", (event) => {
            activeFog.enabled = !activeFog.enabled;
            (event.currentTarget as HTMLButtonElement).textContent = `Fog: ${activeFog.enabled ? "On" : "Off"}`;
            canvas.dataset.fogEnabled = String(activeFog.enabled);
        });
        const updateShapeButton = (): void => {
            const evolving = animating && seekTime === null;
            shapeButton.textContent = seekTime === null ? `Cloud shape: ${animating ? "On" : "Off"}` : "Animation frozen";
            shapeButton.setAttribute("aria-pressed", String(evolving));
            canvas.dataset.animating = String(evolving);
        };
        updateShapeButton();
        if (seekTime !== null) {
            shapeButton.disabled = true;
            shapeButton.title = "Remove seekTime to animate the cloud shape.";
        } else {
            shapeButton.addEventListener("click", () => {
                animating = !animating;
                updateShapeButton();
                const url = new URL(location.href);
                if (animating) {
                    url.searchParams.delete("animate");
                } else {
                    url.searchParams.set("animate", "off");
                }
                url.searchParams.delete("scroll");
                history.replaceState(null, "", url.href);
            });
        }
        volumeSelect.value = volumeSize;
        volumeSelect.addEventListener("change", () => {
            const url = new URL(location.href);
            url.searchParams.set("volumeSize", volumeSelect.value);
            location.assign(url.href);
        });
    }
    if (early) {
        addMeshes();
    }
    if (params.get("transmission") === "after") {
        const { enableSceneTransmission } = await import("../shared/scene316-transmission.js");
        enableSceneTransmission(scene, engine);
    }
    document.getElementById("cameraReset")!.addEventListener("click", () => {
        camera.inertialAlphaOffset = camera.inertialBetaOffset = camera.inertialRadiusOffset = camera.inertialPanningX = camera.inertialPanningY = 0;
        camera.target.x = 0;
        camera.target.y = 0.7;
        camera.target.z = 4;
        camera.alpha = initialAlpha;
        camera.beta = initialBeta;
        camera.radius = initialRadius;
    });
    let previousWidth = 0;
    let previousHeight = 0;
    onBeforeRender(scene, (deltaMs) => {
        if (fog) {
            if (seekTime === null && animating) {
                animationSeconds += deltaMs / 1000;
            }
            cloudState.time = seekTime ?? animationSeconds;
            canvas.dataset.cloudTime = cloudState.time.toFixed(3);
            if (canvas.width !== previousWidth || canvas.height !== previousHeight) {
                previousWidth = canvas.width;
                previousHeight = canvas.height;
                const size = `${Math.ceil(previousWidth / tileSize)}×${Math.ceil(previousHeight / tileSize)}×${depthSlices}`;
                volumeDimensions.textContent = size;
                canvas.dataset.volumeSize = size;
            }
        }
        canvas.dataset.frames = String(Number(canvas.dataset.frames ?? 0) + 1);
        canvas.dataset.width = String(canvas.width);
        canvas.dataset.height = String(canvas.height);
    });
    await registerScene(scene);
    await startEngine(engine);
    await engine._device.queue.onSubmittedWorkDone();
    const error = await engine._device.popErrorScope();
    if (error) {
        throw error;
    }
    canvas.dataset.ready = "true";
    canvas.dataset.fogEnabled = params.get("baseline") === "1" ? "false" : "true";
    if (seekTime !== null) {
        canvas.dataset.animationFrozen = "true";
    }
}

void main().catch((error: unknown) => {
    canvas.dataset.error = error instanceof Error ? error.message : String(error);
    console.error(error);
});
