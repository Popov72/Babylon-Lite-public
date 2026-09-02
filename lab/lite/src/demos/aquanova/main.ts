// Aquanova demo — first-person sci-fi shooter tech demo built on Lite.
//
// The player traverses a confined spaceship wielding the "Liquefactor": melt flying alien foes
// and ship props into GPU fluid. This slice LOADS THE SHIP: a chunked modular interior authored
// in the Aquanova ship editor from the CC0 Quaternius "Modular SciFi MegaKit" and exported to a
// single glTF (ship.glb) with a companion ship_manifest.json describing chunks, portals, doors
// and gameplay placements (see lab/public/aquanova/ASSET-LICENSES.md and SciFiShip/README.md).
//
// What this slice does:
//   • loads ship.glb and lights it with authored runtime lights + bounded local environment probes;
//   • builds one static trimesh collider from the whole ship so the player collides with every
//     wall, floor and prop — no hand-authored boxes;
//   • spawns a Havok first-person character-controller capsule at the manifest player behavior, the
//     camera riding at eye height (WASD/arrows walk in the horizontal plane, mouse-drag looks).
//
// The ship runs back → front along glTF +X (storage → corridor → junction → cargo bay). Lite's
// glTF loader mirrors handedness on the __root__ (scale x = -1), so glTF (x,y,z) renders at Lite
// (-x, y, z).
//

import HavokPhysics from "@babylonjs/havok";
import {
    addTask,
    addTaskAfter,
    addTaskBefore,
    addToScene,
    applyPhysicsImpulse,
    attachPositionGizmoToNode,
    attachRotationGizmoToNode,
    attachScaleGizmoToNode,
    createDepthResolveTask,
    createEngine,
    createFreeCamera,
    createGpuPicker,
    createHavokWorld,
    createCopyToTextureTask,
    adoptFluidParticleChannel,
    adoptFluidSceneSdf,
    attachFluidSimulationCollectionRenderLayer,
    beginFluidSimulationProfilerFrame,
    configureFluidSimulationRenderCompositor,
    configureFluidSimulationRenderLayer,
    createFluidParticleChannel,
    createFluidImpulseForce,
    createFluidRenderEnvironment,
    createFluidSimulation,
    createFluidSimulationCollection,
    createFluidSimulationCollectionParticleStream,
    createFluidSimulationProfiler,
    createFluidSimulationRenderCompositor,
    createReferenceCountedPool,
    createPhysicsBody,
    createPhysicsCharacterController,
    createPhysicsShape,
    createPositionGizmo,
    createRenderTarget,
    createRenderTask,
    createRotationGizmo,
    createScaleGizmo,
    createSceneContext,
    createSmaaPostProcessTask,
    createTaaPostProcessTask,
    createTransformNode,
    importFluidPresetSession,
    planFluidInitialState,
    createUtilityLayer,
    AcesToneMapping,
    NeutralToneMapping,
    StandardToneMapping,
    setSceneImageProcessing,
    enableMaterialPlugins,
    disposeFluidParticleChannel,
    disposeFluidForceField,
    disposeFluidSimulation,
    disposeFluidSimulationCollection,
    endFluidSimulationProfilerFrame,
    fluidSimulationRenderLayerDepthForSceneIntegration,
    fluidSimulationCellSize,
    fluidSimulationParticleCapacity,
    resolveFluidSimulationConfig,
    resolveFluidReconfigurationPlan,
    resolveFluidImpulseForce,
    aquanovaCombinedParticleCapacity,
    fluidRenderProfileKey,
    getFrameGraph,
    getFluidSimulationDiagnostics,
    readFluidSimulationProfiler,
    readFluidSimulationPositions,
    getPhysicsCharacterControllerBody,
    getPhysicsBodyLinearVelocity,
    getPhysicsTimestepMs,
    getProjectionMatrix,
    getViewMatrix,
    getViewProjectionMatrix,
    isPbrMaterial,
    loadGltf,
    loadSkybox,
    lockPhysicsBodyRotationAxes,
    markMaterialUboDirty,
    mat4Decompose,
    PhysicsMotionType,
    PhysicsShapeType,
    physicsRaycast,
    shapeCast,
    rebuildScenePbrPipelines,
    registerScene,
    registerUtilityLayer,
    releasePhysicsShape,
    removeFromScene,
    removePhysicsBody,
    setMeshVisible,
    setParent,
    setPhysicsBodyAngularVelocity,
    setPhysicsBodyLinearVelocity,
    setPhysicsBodyMass,
    setPhysicsBodyMotionType,
    setPhysicsBodyShape,
    setPhysicsBodyTransform,
    setPhysicsShapeFilterCollideMask,
    setPhysicsTimestepMs,
    setPositionGizmoLocalCoordinates,
    setRotationGizmoLocalCoordinates,
    setScaleGizmoLocalCoordinates,
    startEngine,
    stopAnimation,
    gridFloorY,
    gridTopY,
    setFluidSimulationCollectionProfiler,
    setFluidSimulationCollectionSources,
    setFluidSimulationForceField,
    setFluidSimulationFlow,
    setFluidSimulationProfiler,
    setFluidSimulationSceneSdf,
    stepFluidSimulation,
    stepFluidSimulationForSceneIntegration,
    transformFluidFlow,
    writeFluidParticleChannel,
    writeFluidSimulationPositions,
    FLIP_DEFAULT_MARKERS_PER_CELL,
    fluidMeshSamplingErrorInfo,
    sampleFluidMeshParticles,
} from "babylon-lite";
import type {
    AnimationGroup,
    EnvironmentTextures,
    FluidExportJson,
    FoamDebugTexture,
    FluidFlowConfig,
    FluidForceField,
    FluidMeshSamplingErrorInfo,
    FluidMeshSamplingWarning,
    FluidParticleChannel,
    FluidSimulation,
    FluidSimulationOptions,
    FluidSimulationCollection,
    FluidSimulationProfiler,
    FluidSimulationRenderLayer,
    FluidSimulationRenderSource,
    FluidPresetSession,
    PairState,
    Material,
    Mesh,
    PbrMaterialProps,
    PhysicsBody,
    PhysicsRotationAxis,
    RenderTask,
    RenderTarget,
    SceneNode,
    SceneSdfSpec,
    Task,
    ToneMapping,
} from "babylon-lite";
import type { ParticleFillWorkerResponse } from "../particle-fill-worker.js";
import { productionFluidCapabilityRejection } from "./fluid-capabilities.js";
import { disposeRenderTarget } from "babylon-lite";
import { createLiquefyPlugin, liquefyFrontDistance } from "../liquefy-plugin.js";
import { buildLitParticleColors } from "../particle-lit-colors.js";
import type { LitColorScene } from "../particle-lit-colors.js";
import { DEFAULT_SHIP_IBL_STRENGTH, resolveExposure, resolveToneMapping } from "../ship-manifest.js";
import type { LiquefyState } from "../liquefy-plugin.js";
import {
    CEIL_Y,
    CROUCH_CAPSULE_HEIGHT,
    CROUCH_CAPSULE_RADIUS,
    FLOOR_Y,
    MAX_WALKABLE_SLOPE_COSINE,
    PLAYER_CAPSULE_HEIGHT,
    PLAYER_CAPSULE_RADIUS,
    SHIP_URL,
    SKYBOX_EXT,
    SKYBOX_SIZE,
    SKYBOX_URL,
    type Vec3,
} from "./constants.js";
import { applyLocalEnvironmentProbes, type LocalEnvironmentBlendInfo } from "./local-environments.js";
import { buildRuntimeLights } from "./lights.js";
import { buildManifestColliders, createWorldCollisionShape } from "./colliders.js";
import { collisionShapesForModule, worldShapesForMatrix, type WorldCollisionShape } from "./collision-shapes.js";
import {
    addFluidHole,
    createFluidHoleRing,
    hollowCylinderPrimitiveForMatrix,
    localizePrimitive,
    packPrimitive,
    packPrimitives,
    packSubtractionPrimitive,
    packSubtractionPrimitives,
    primBufferBytes,
    primitiveSdf,
    PRIMITIVES_WGSL,
    PRIM_ACTIVE_OFFSET,
    PRIM_HEADER,
    PRIM_STRIDE,
    setPackedPrimitiveActive,
    shotHolePrimitive,
    type FluidHoleRing,
    type FluidPrimitive,
} from "./collision-field.js";
import { chunkAt, fetchManifest } from "./manifest.js";
import { createInspectOverlay } from "./debug/inspect-overlay.js";
import { createPerfOverlay } from "./debug/perf-overlay.js";
import { createColliderOverlay, visibleInjectedPrimitives } from "./debug/collider-overlay.js";
import { createFluidSimulationOverlay, type FluidSimulationDebugSnapshot } from "./debug/fluid-simulation-overlay.js";
import { createLightOverlay } from "./debug/light-overlay.js";
import { createProbeOverlay } from "./debug/probe-overlay.js";
import { createPortalOverlay } from "./debug/portal-overlay.js";
import { LAB_DEBUG } from "./debug-flag.js";
import { LIQUEFACTOR_MODELS, loadGraphicsSettings, saveGraphicsSettings, type LiquefactorModel } from "./settings.js";
import { meshGroupBounds, type MeshGroupBounds } from "./mesh-bounds.js";
import type { FluidElectricityDomain, FluidElectricityFrameDomain } from "./fluid-runtime.js";
import { createFluidElectricityRenderer } from "./fluid-electricity-renderer.js";
import { createPortalVisibility } from "./portal-visibility.js";
import { registerDoorEntityEventHandlers } from "./door-events.js";
import { registerEntityCollisionEventHandlers } from "./entity-collision-events.js";
import { createIntersectionTriggerRegistry } from "./intersection-triggers.js";
import { createPersistentMeshEntityEventOperations, registerMeshEntityEventHandlers } from "./entity-events.js";
import { forEachFluidCollisionSet } from "./fluid-collision-lifecycle.js";
import { createExteriorMeshClassifier } from "./exterior-mesh-classifier.js";
import { fetchFluidSetting, hexToRgb, type FluidFoamSetting, type FluidRenderSetting, type FluidSimSetting } from "./fluid-setting.js";
import {
    antiGravityCollisionSlideDisplacement,
    AquanovaBehaviorManager,
    AquanovaExplosionRuntime,
    ExplodeBehavior,
    PlayerBehavior,
    SoundManager,
    type FluidSimulationRegistration,
    type FluidSimulationState,
    fluidEmissionCompletionTarget,
    fluidSimulationShutdownLifecycle,
    fluidSimulationShutdownStepDelta,
    type FluidSimShapeRegistration,
    type ExplosionOptions,
    type ExplosionTarget,
    type JumpApertureAssist,
    type LiquefiableBehaviorConfig,
    type MeshBehaviorAvailability,
} from "./behaviors/index.js";
import { playerCapsuleSpawnPosition, selectClosestClearApertureOffset } from "./behaviors/player.js";
import { pauseAnimationsTargetingEntities, resumeAnimations } from "./behaviors/play-animation.js";
import { createAquanovaControlPanel, type AquanovaControlPanel, type WeaponTransformValues } from "./control-panel.js";
import { createAntiGravityGunViewmodel, createLiquefactorViewmodel, createPistolViewmodel, type LiquefactorViewmodel } from "./liquefactor-viewmodel.js";
import { createWeaponParticleLaser, type WeaponLaserAim } from "./weapon-laser.js";
import { createPistolProjectileRuntime } from "./pistol-projectiles.js";
import { createCheatCodeMatcher } from "./cheat-code.js";
import { getMeshPoseGeometry } from "../mesh-pose-geometry.js";

export async function main(): Promise<void> {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    /**
     * SSAA: render the whole frame at `scale`× the display resolution and let the compositor
     * downscale it.
     *
     * Done through CSS rather than by resizing render targets, for two reasons. Every canvas-sized
     * target already derives from `canvas.width` — including the fluid task's screen-space depth and
     * thickness buffers — so scaling the backing store scales the entire pipeline at once with no
     * chance of two passes disagreeing about size. And the engine calls `resizeEngine` every frame,
     * recomputing the backing store from `clientWidth × devicePixelRatio`; setting the size directly
     * would be undone immediately, whereas growing the ELEMENT makes the engine compute the size we
     * want by itself. The transform shrinks it back to the layout box for display.
     *
     * Picking is unaffected: it uses `clientWidth / 2`, which is still the centre of the crosshair.
     */
    const applySsaa = (scale: number): void => {
        const pct = `${scale * 100}%`;
        canvas.style.width = pct;
        canvas.style.height = pct;
        canvas.style.transformOrigin = "0 0";
        canvas.style.transform = scale === 1 ? "" : `scale(${1 / scale})`;
        canvas.dataset.ssaa = String(scale);
    };
    // Fetch the ship manifest up front — it drives the IBL strength, tone mapping, the authored
    // Specular AA default, the reflection-roughness multiplier, colliders, and spawn.
    const manifest = await fetchManifest();

    // Player-tunable graphics (ship's authored defaults ← localStorage ← ?msaa= query override).
    // Resolved before the engine so a future setting that has to be chosen at engine-creation time
    // can be read here too.
    const graphics = loadGraphicsSettings({ specularAA: manifest?.environment?.specularAA });
    applySsaa(graphics.ssaa);
    // msaaSamples: 1 — the ship renders into an offscreen target the fluid surface samples, never
    // straight to the swapchain, so the engine's own MSAA would never apply. The scene pass does its
    // own multisampling instead; see the MSAA block further down.
    const engine = await createEngine(canvas, { msaaSamples: 1, srgb: graphics.srgb });
    const scene = createSceneContext(engine, { defaultRenderTask: false });

    const cam = createFreeCamera({ x: 0, y: 1.5, z: 0 }, { x: -1, y: 1.5, z: 0 });
    cam.nearPlane = 0.1;
    cam.farPlane = 400;
    scene.camera = cam;

    const PRODUCTION_PRESET_DEFAULTS: PairState = {
        schema: {},
        demoParams: {},
        color: "#16a3c3",
        half: false,
        thicknessDownscale: 1,
        absorption: 0.4,
        size: 0.7,
        physScale: 1,
        count: 0,
    };

    // Preload the fluid-simulation settings the manifest lists (parsed once). At liquefy time a mesh
    // uses its pinned override if it has one, else a random pick from this global list.
    //
    // The union of the global list AND every behaviour's own `fluidSim` is loaded. Loading only the
    // global list meant a behaviour naming its own setting (the crates carry
    // `fluidSim: ["liquid-slow"]`) missed the map at liquefy time and silently fell back to
    // DEFAULT_SAMPLE_RADIUS + FLUID_SETTING: the crates sampled at 0.03 instead of the file's 0.08,
    // giving ~19x the particles (6260 vs 330) and default physics instead of the file's.
    const fluidSettingNames = manifest?.fluidSim ?? [];
    const behaviorSettingNames = new Set<string>();
    for (const b of Object.values(manifest?.behaviorPresets ?? {})) {
        if ("fluidSim" in b && Array.isArray(b.fluidSim)) {
            for (const n of b.fluidSim) behaviorSettingNames.add(n);
        }
    }
    for (const e of Object.values(manifest?.entities ?? {})) {
        for (const ref of e?.behaviors ?? []) {
            if ("fluidSim" in ref && Array.isArray(ref.fluidSim)) {
                for (const n of ref.fluidSim) behaviorSettingNames.add(n);
            }
        }
    }
    const fluidSettings = new Map<string, FluidSimSetting>();
    await Promise.all(
        [...new Set([...fluidSettingNames, ...behaviorSettingNames])].map(async (name) => {
            const s = await fetchFluidSetting(name, PRODUCTION_PRESET_DEFAULTS);
            if (s) fluidSettings.set(name, s);
        })
    );

    // ── Load the ship ──────────────────────────────────────────────────────────────────────
    // Authored runtime lamps provide direct lighting. Bounded local probes loaded below provide
    // diffuse SH and specular radiance; there is deliberately no scene-global environment.
    const ship = await loadGltf(engine, SHIP_URL);
    const shipRoot = ship.entities[0] as SceneNode;
    for (const animation of ship.animationGroups ?? []) {
        stopAnimation(animation);
    }
    addToScene(scene, ship);

    // First-person weapon overlay. A utility layer gives it a fresh depth buffer and renders it
    // after the world/fluid/post-process chain, while sharing the gameplay camera. All three detail
    // levels are preloaded so the control-panel selector switches immediately without a runtime
    // pipeline rebuild or a visible gap.
    const weaponLayer = createUtilityLayer(engine, scene, { addDefaultLight: false });
    Object.assign(weaponLayer.scene.imageProcessing, scene.imageProcessing);
    const weaponViewmodel = await createLiquefactorViewmodel(engine, cam);
    const antiGravityGunViewmodel = await createAntiGravityGunViewmodel(engine, cam);
    const pistolViewmodel = await createPistolViewmodel(engine, cam);
    const weaponViewmodels: readonly LiquefactorViewmodel[] = [weaponViewmodel, antiGravityGunViewmodel, pistolViewmodel];
    const weaponMeshes = weaponViewmodels.flatMap((viewmodel) => viewmodel.meshes);
    weaponViewmodel.select(graphics.liquefactorModel);
    antiGravityGunViewmodel.select(graphics.liquefactorModel);
    pistolViewmodel.select(graphics.liquefactorModel);
    for (const viewmodel of weaponViewmodels) {
        viewmodel.setSwayEnabled(graphics.weaponSway);
    }
    canvas.dataset.liquefactorModel = graphics.liquefactorModel;
    addToScene(weaponLayer.scene, weaponViewmodel.root);
    addToScene(weaponLayer.scene, antiGravityGunViewmodel.root);
    addToScene(weaponLayer.scene, pistolViewmodel.root);
    const weaponGizmoLayer = createUtilityLayer(engine, scene);
    addToScene(weaponGizmoLayer.scene, weaponViewmodel.localGuideRoot);
    addToScene(weaponGizmoLayer.scene, antiGravityGunViewmodel.localGuideRoot);
    addToScene(weaponGizmoLayer.scene, pistolViewmodel.localGuideRoot);
    const weaponLaser = createWeaponParticleLaser(engine, weaponGizmoLayer.scene);
    const pistolProjectiles = createPistolProjectileRuntime(engine, scene, pistolViewmodel);
    let weaponAimRay: WeaponLaserAim | null = null;
    let weaponCrosshair: HTMLElement | null = null;
    const updateWeaponCrosshair = (viewmodel: LiquefactorViewmodel): void => {
        weaponCrosshair ??= document.getElementById("aq-crosshair");
        if (!weaponCrosshair) return;
        const lineMatrix = viewmodel.localGuideYaw.worldMatrix;
        const pointX = lineMatrix[8]! * 2 + lineMatrix[12]!;
        const pointY = lineMatrix[9]! * 2 + lineMatrix[13]!;
        const pointZ = lineMatrix[10]! * 2 + lineMatrix[14]!;
        const cameraMatrix = cam.worldMatrix;
        const cameraX = cameraMatrix[12]!;
        const cameraY = cameraMatrix[13]!;
        const cameraZ = cameraMatrix[14]!;
        const rayX = pointX - cameraX;
        const rayY = pointY - cameraY;
        const rayZ = pointZ - cameraZ;
        const rayLength = Math.sqrt(rayX * rayX + rayY * rayY + rayZ * rayZ);
        weaponAimRay =
            rayLength > 1e-8
                ? {
                      origin: [cameraX, cameraY, cameraZ],
                      direction: [rayX / rayLength, rayY / rayLength, rayZ / rayLength],
                  }
                : null;
        const viewProjection = getViewProjectionMatrix(cam, engine.canvas.width / Math.max(1, engine.canvas.height));
        const clipX = viewProjection[0]! * pointX + viewProjection[4]! * pointY + viewProjection[8]! * pointZ + viewProjection[12]!;
        const clipY = viewProjection[1]! * pointX + viewProjection[5]! * pointY + viewProjection[9]! * pointZ + viewProjection[13]!;
        const clipW = viewProjection[3]! * pointX + viewProjection[7]! * pointY + viewProjection[11]! * pointZ + viewProjection[15]!;
        if (clipW <= 0) {
            weaponAimRay = null;
            weaponCrosshair.style.display = "none";
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const ndcX = clipX / clipW;
        const ndcY = clipY / clipW;
        weaponCrosshair.style.display = "block";
        weaponCrosshair.style.left = `${rect.left + (ndcX * 0.5 + 0.5) * rect.width}px`;
        weaponCrosshair.style.top = `${rect.top + (0.5 - ndcY * 0.5) * rect.height}px`;
    };
    let activeWeapon: "liquefactor" | "antiGravityGun" | "pistol" | null = null;
    let refreshWeaponDebugTools = (): void => {};
    let playerBehavior: PlayerBehavior | null = null;
    const setWeaponEnabled = (id: "liquefactor" | "antiGravityGun" | "pistol", viewmodel: LiquefactorViewmodel, enabled: boolean, animated = true): void => {
        if (enabled) {
            activeWeapon = id;
        } else if (activeWeapon === id) {
            activeWeapon = null;
        }
        canvas.dataset.weaponEnabled = String(activeWeapon !== null);
        canvas.dataset.weapon = activeWeapon ?? "hidden";
        if (enabled) {
            viewmodel.select(viewmodel.model);
            viewmodel.setPresented(true, animated);
        } else {
            viewmodel.setPresented(false, animated);
            if (activeWeapon === null) {
                weaponLaser.stop();
                weaponAimRay = null;
                weaponCrosshair ??= document.getElementById("aq-crosshair");
                if (weaponCrosshair) {
                    weaponCrosshair.style.display = "none";
                }
            }
        }
        refreshWeaponDebugTools();
    };
    let grabDynamicWithAntiGravity = (_mesh: Mesh): boolean => false;
    let updateAntiGravityGrab = (_deltaMs: number): boolean => false;
    let releaseAntiGravityGrab = (_throwSpeed: number): void => {};
    let isEntityCollisionActive = (_mesh: Mesh): boolean => true;
    canvas.dataset.antiGravityGrabbed = "none";
    canvas.dataset.antiGravityThrowSpeed = "0";
    canvas.dataset.antiGravityBlocked = "false";
    const weaponLiquefactor = {
        setEnabled: (enabled: boolean, animated = true): void => setWeaponEnabled("liquefactor", weaponViewmodel, enabled, animated),
        setTargetDistance: (distance: number | null, restart = false): void => {
            weaponLaser.setTargetDistance(distance, restart);
        },
        isReady: (): boolean => activeWeapon === "liquefactor" && weaponViewmodel.ready,
        stop: (): void => {
            weaponLaser.stop();
        },
        update: (deltaMs: number): boolean => {
            weaponViewmodel.update(
                cam,
                engine.canvas.width / Math.max(1, engine.canvas.height),
                deltaMs,
                playerBehavior?.weaponSwayMultiplier ?? 1,
                playerBehavior?.isWeaponTriggerHeld ?? false
            );
            if (activeWeapon !== "liquefactor" || !weaponViewmodel.ready) {
                return false;
            }
            updateWeaponCrosshair(weaponViewmodel);
            return weaponLaser.update(deltaMs, weaponAimRay, weaponViewmodel.localGuideOrigin.worldMatrix);
        },
    };
    const weaponAntiGravityGun = {
        setEnabled: (enabled: boolean, animated = true): void => setWeaponEnabled("antiGravityGun", antiGravityGunViewmodel, enabled, animated),
        isReady: (): boolean => activeWeapon === "antiGravityGun" && antiGravityGunViewmodel.ready,
        update: (deltaMs: number): void => {
            antiGravityGunViewmodel.update(
                cam,
                engine.canvas.width / Math.max(1, engine.canvas.height),
                deltaMs,
                playerBehavior?.weaponSwayMultiplier ?? 1,
                playerBehavior?.isWeaponTriggerHeld ?? false
            );
            if (activeWeapon === "antiGravityGun" && antiGravityGunViewmodel.ready) {
                updateWeaponCrosshair(antiGravityGunViewmodel);
            }
        },
        grab: (mesh: Mesh): boolean => grabDynamicWithAntiGravity(mesh),
        updateGrab: (deltaMs: number): boolean => updateAntiGravityGrab(deltaMs),
        releaseGrab: (throwSpeed: number): void => releaseAntiGravityGrab(throwSpeed),
    };
    const weaponPistol = {
        setEnabled: (enabled: boolean, animated = true): void => setWeaponEnabled("pistol", pistolViewmodel, enabled, animated),
        isReady: (): boolean => activeWeapon === "pistol" && pistolViewmodel.ready,
        fire: (mesh: Mesh | null, point: readonly [number, number, number] | null, distance: number | null, range: number, speed: number): void => {
            const activeMesh = mesh && isEntityCollisionActive(mesh) ? mesh : null;
            pistolProjectiles.fire(activeMesh, activeMesh ? point : null, activeMesh ? distance : null, range, speed);
        },
        update: (deltaMs: number) => {
            pistolViewmodel.update(
                cam,
                engine.canvas.width / Math.max(1, engine.canvas.height),
                deltaMs,
                playerBehavior?.weaponSwayMultiplier ?? 1,
                playerBehavior?.isWeaponTriggerHeld ?? false
            );
            if (activeWeapon === "pistol" && pistolViewmodel.ready) {
                updateWeaponCrosshair(pistolViewmodel);
            }
            return pistolProjectiles.update(deltaMs, isEntityCollisionActive);
        },
        clear: (): void => pistolProjectiles.clear(),
    };
    const weaponDebugTools = !LAB_DEBUG
        ? null
        : (() => {
              const positionGizmo = createPositionGizmo(engine, weaponGizmoLayer, { planarEnabled: true });
              const rotationGizmo = createRotationGizmo(engine, weaponGizmoLayer);
              const scaleGizmo = createScaleGizmo(engine, weaponGizmoLayer);
              const localGuidePositionGizmo = createPositionGizmo(engine, weaponGizmoLayer, { planarEnabled: true });
              setPositionGizmoLocalCoordinates(positionGizmo, true);
              setRotationGizmoLocalCoordinates(rotationGizmo, true);
              setScaleGizmoLocalCoordinates(scaleGizmo, true);
              setPositionGizmoLocalCoordinates(localGuidePositionGizmo, true);
              let positionGizmoOn = false;
              let rotationGizmoOn = false;
              let scaleGizmoOn = false;
              let localGuideGizmoOn = false;
              let toolsVisible = false;
              const activeViewmodel = (): LiquefactorViewmodel | null =>
                  activeWeapon === "antiGravityGun"
                      ? antiGravityGunViewmodel
                      : activeWeapon === "pistol"
                        ? pistolViewmodel
                        : activeWeapon === "liquefactor"
                          ? weaponViewmodel
                          : null;
              const positionSubGizmos = [
                  positionGizmo.xGizmo,
                  positionGizmo.yGizmo,
                  positionGizmo.zGizmo,
                  ...(positionGizmo.xPlaneGizmo ? [positionGizmo.xPlaneGizmo] : []),
                  ...(positionGizmo.yPlaneGizmo ? [positionGizmo.yPlaneGizmo] : []),
                  ...(positionGizmo.zPlaneGizmo ? [positionGizmo.zPlaneGizmo] : []),
              ];
              const rotationSubGizmos = [rotationGizmo.xGizmo, rotationGizmo.yGizmo, rotationGizmo.zGizmo];
              const scaleSubGizmos = [scaleGizmo.xGizmo, scaleGizmo.yGizmo, scaleGizmo.zGizmo, scaleGizmo.uniformScaleGizmo];
              const applyTransformGizmo = (): void => {
                  activeViewmodel()?.applyTransformGizmoTarget();
              };
              for (const gizmo of positionSubGizmos) gizmo.onPositionChanged.add(applyTransformGizmo);
              for (const gizmo of rotationSubGizmos) gizmo.onRotationChanged.add(applyTransformGizmo);
              for (const gizmo of scaleSubGizmos) gizmo.onScaleChanged.add(applyTransformGizmo);
              const setGizmoMeshesVisible = (visible: boolean, gizmos: ReadonlyArray<{ _visibleMeshes: Mesh[] }>): void => {
                  for (const gizmo of gizmos) {
                      for (const mesh of gizmo._visibleMeshes) setMeshVisible(mesh, visible);
                  }
              };
              const sync = (): void => {
                  const viewmodel = activeViewmodel();
                  viewmodel?.syncTransformGizmoTarget();
                  const positionVisible = toolsVisible && positionGizmoOn;
                  const rotationVisible = toolsVisible && rotationGizmoOn;
                  const scaleVisible = toolsVisible && scaleGizmoOn;
                  const localGuideVisible = toolsVisible && localGuideGizmoOn;
                  attachPositionGizmoToNode(positionGizmo, positionVisible ? (viewmodel?.transformGizmoTarget ?? null) : null);
                  attachRotationGizmoToNode(rotationGizmo, rotationVisible ? (viewmodel?.transformGizmoTarget ?? null) : null);
                  attachScaleGizmoToNode(scaleGizmo, scaleVisible ? (viewmodel?.transformGizmoTarget ?? null) : null);
                  attachPositionGizmoToNode(localGuidePositionGizmo, localGuideVisible ? (viewmodel?.localGuideOrigin ?? null) : null);
                  setGizmoMeshesVisible(positionVisible, positionSubGizmos);
                  setGizmoMeshesVisible(rotationVisible, rotationSubGizmos);
                  setGizmoMeshesVisible(scaleVisible, scaleSubGizmos);
                  setGizmoMeshesVisible(localGuideVisible, [
                      localGuidePositionGizmo.xGizmo,
                      localGuidePositionGizmo.yGizmo,
                      localGuidePositionGizmo.zGizmo,
                      ...(localGuidePositionGizmo.xPlaneGizmo ? [localGuidePositionGizmo.xPlaneGizmo] : []),
                      ...(localGuidePositionGizmo.yPlaneGizmo ? [localGuidePositionGizmo.yPlaneGizmo] : []),
                      ...(localGuidePositionGizmo.zPlaneGizmo ? [localGuidePositionGizmo.zPlaneGizmo] : []),
                  ]);
                  canvas.dataset.weaponGizmoTarget = toolsVisible && viewmodel ? activeWeapon! : "hidden";
              };
              const values = (): WeaponTransformValues => {
                  const viewmodel = activeViewmodel() ?? weaponViewmodel;
                  const adjustment = viewmodel.adjustment;
                  const localGuideOrigin = viewmodel.localGuideOrigin;
                  const localGuideYaw = viewmodel.localGuideYaw;
                  const degrees = 180 / Math.PI;
                  return {
                      position: [adjustment.position.x, adjustment.position.y, adjustment.position.z],
                      rotationDegrees: [adjustment.rotation.x * degrees, adjustment.rotation.y * degrees, adjustment.rotation.z * degrees],
                      scale: [adjustment.scaling.x, adjustment.scaling.y, adjustment.scaling.z],
                      localGuidePosition: [localGuideOrigin.position.x, localGuideOrigin.position.y, localGuideOrigin.position.z],
                      localGuideRotationDegrees: [localGuideYaw.rotation.x * degrees, localGuideYaw.rotation.y * degrees, localGuideYaw.rotation.z * degrees],
                  };
              };
              const tools = {
                  values,
                  positionGizmoEnabled: (): boolean => positionGizmoOn,
                  setPositionGizmoEnabled: (enabled: boolean): void => {
                      positionGizmoOn = enabled;
                      sync();
                  },
                  rotationGizmoEnabled: (): boolean => rotationGizmoOn,
                  setRotationGizmoEnabled: (enabled: boolean): void => {
                      rotationGizmoOn = enabled;
                      sync();
                  },
                  scaleGizmoEnabled: (): boolean => scaleGizmoOn,
                  setScaleGizmoEnabled: (enabled: boolean): void => {
                      scaleGizmoOn = enabled;
                      sync();
                  },
                  localGuideGizmoEnabled: (): boolean => localGuideGizmoOn,
                  setLocalGuideGizmoEnabled: (enabled: boolean): void => {
                      localGuideGizmoOn = enabled;
                      sync();
                  },
                  localGuideYawDegrees: (): number => ((activeViewmodel() ?? weaponViewmodel).localGuideYaw.rotation.y * 180) / Math.PI,
                  setLocalGuideYawDegrees: (degrees: number): void => {
                      (activeViewmodel() ?? weaponViewmodel).localGuideYaw.rotation.y = (degrees * Math.PI) / 180;
                  },
                  setVisible: (visible: boolean): void => {
                      toolsVisible = visible;
                      sync();
                  },
              };
              refreshWeaponDebugTools = sync;
              sync();
              return tools;
          })();
    canvas.dataset.liquefactorParent = weaponViewmodel.root.parent === cam ? "camera" : "other";

    // Backdrop seen through the ship's openings. Kept separate from the IBL above: the HDRI is what
    // lights the metal, this is only what you see. Non-fatal — the cube faces are gitignored like
    // the ship and the HDR, so a fresh clone without them still runs, just against the clear colour.
    try {
        await loadSkybox(scene, SKYBOX_URL, SKYBOX_EXT, SKYBOX_SIZE);
    } catch (err) {
        console.warn("[aquanova] skybox not loaded (missing cube faces?)", err);
    }

    // Honour the manifest's local-IBL strength per material, leaving emissive fixtures unchanged.
    let environmentIntensity = manifest?.environment?.strength ?? DEFAULT_SHIP_IBL_STRENGTH;
    const seenMats = new Set<object>();
    const applyIblStrength = (node: SceneNode): void => {
        const mat = (node as { material?: { environmentIntensity?: number } }).material;
        if (mat && !seenMats.has(mat)) {
            seenMats.add(mat);
            mat.environmentIntensity = environmentIntensity;
        }
        for (const child of node.children) applyIblStrength(child as SceneNode);
    };
    applyIblStrength(shipRoot);

    // ── Classify ship meshes: collect all, plus global dynamic / liquefiable and per-room static sets ─
    // Entities are keyed by the glTF NODE name (what the Babylon sandbox shows, e.g. "Door_D00_L"),
    // NOT the glTF mesh name Lite puts on `mesh.name`. The loader parents each node's primitives under
    // a TransformNode carrying the node name, so the walk below records that name per mesh and groups
    // one node's primitives together.
    //
    // `_primitiveN` WRAPPERS: the exporter writes the loader's split convention out as real nodes —
    // a mesh-less parent "player" holding a child node "player_primitive0" that owns the geometry
    // (273 of 431 nodes in the current ship). Those
    // wrappers are collapsed into their parent so the author's node name is what the manifest keys,
    // and so a node's primitives regroup into ONE entity instead of N single-primitive ones. The test
    // is exact — the child's name must be the parent's plus `_primitive<digits>` — so a genuinely
    // authored node is never absorbed.
    //
    // `chunkStaticMeshes` (everything NOT dynamic/liquefiable, grouped by the chunk whose AABB contains
    // the mesh centre) feeds the per-room SDF bake; manifest AABBs are glTF-space, so the Lite-space
    // mesh centre is converted back.
    const PRIMITIVE_WRAPPER = /_primitive\d+$/;
    // NODE NAMES ARE THE EDITOR'S, VERBATIM. Two crates placed from the same module both export as
    // `crate4`, and that is wanted: entities are keyed by node name deliberately, so one manifest
    // entry drives every mesh sharing that name.
    const allShipMeshes: Mesh[] = [];
    const nodeNameOfMesh = new Map<Mesh, string>(); // mesh → owning glTF node name
    const ownerOfMesh = new Map<Mesh, SceneNode>(); // mesh → the glTF node that owns it (carries extras)
    const nodePrimitives = new Map<Mesh, Mesh[]>(); // mesh → every primitive of the SAME node (incl. itself)
    const meshesByNodeName = new Map<string, Mesh[]>(); // node name → all its primitives, ship-wide
    const primitivesByOwner = new Map<SceneNode, Mesh[]>();
    // glTF node name of each chunk root ("CHUNK_CH00_Storage") → chunk id, straight from the manifest.
    // A mesh's chunk is its CHUNK_* ancestor, tracked on the way down — the exporter parents every
    // mesh under exactly one chunk root, and `chunks[].meshCount` agrees with that walk exactly.
    const chunkIdByRootNode = new Map<string, string>();
    for (const c of manifest?.chunks ?? []) if (c.node) chunkIdByRootNode.set(c.node, c.id);
    const chunkOfMesh = new Map<Mesh, string>();
    /** Instance id → its placement node, from the exporter's `extras = { id, module, chunk }`. The
     *  node is what carries the placement transform in the scene's own space. */
    const placementNodes = new Map<string, { module: string; node: SceneNode }>();
    const collectMeshes = (node: SceneNode, owner: SceneNode, chunk: string | undefined): void => {
        for (const child of node.children) {
            const c = child as SceneNode;
            const cName = c.name;
            const inChunkNow = chunkIdByRootNode.get(cName) ?? chunk;
            const ex = c.metadata?.gltf?.extras as { id?: string; module?: string } | undefined;
            if (ex?.id && ex.module && !placementNodes.has(ex.id)) placementNodes.set(ex.id, { module: ex.module, node: c });
            if ((c as Mesh).material) {
                const m = c as Mesh;
                const ownerName = owner.name;
                allShipMeshes.push(m);
                nodeNameOfMesh.set(m, ownerName);
                ownerOfMesh.set(m, owner);
                if (inChunkNow) chunkOfMesh.set(m, inChunkNow);
                const group = primitivesByOwner.get(owner);
                if (group) group.push(m);
                else primitivesByOwner.set(owner, [m]);
                const byName = meshesByNodeName.get(ownerName);
                if (byName) byName.push(m);
                else meshesByNodeName.set(ownerName, [m]);
                continue;
            }
            // A wrapper keeps the parent as owner; any other node owns its own subtree.
            const wrapper = PRIMITIVE_WRAPPER.test(cName) && cName.replace(PRIMITIVE_WRAPPER, "") === node.name;
            collectMeshes(c, wrapper ? owner : c, inChunkNow);
        }
    };
    collectMeshes(shipRoot, shipRoot, undefined);
    for (const group of primitivesByOwner.values()) for (const m of group) nodePrimitives.set(m, group);
    const worldBoundsOf = (mesh: Mesh): MeshGroupBounds | null => meshGroupBounds(nodePrimitives.get(mesh) ?? [mesh]);
    const pbrMaterialsOf = (meshes: Iterable<Mesh>): PbrMaterialProps[] => {
        const materials = new Set<PbrMaterialProps>();
        for (const mesh of meshes) {
            if (mesh.material && isPbrMaterial(mesh.material)) materials.add(mesh.material);
        }
        return [...materials];
    };
    const pbrMaterials = (): PbrMaterialProps[] => pbrMaterialsOf([...allShipMeshes, ...weaponMeshes]);
    const fullyMetallicRoughnessOriginal = Symbol("fullyMetallicRoughnessOriginal");
    type RoughnessTaggedMaterial = PbrMaterialProps & {
        [fullyMetallicRoughnessOriginal]?: number | undefined;
    };
    const fullyMetallicRoughnessBackups = new Map<PbrMaterialProps, number | undefined>();
    let forceFullyMetallicRoughnessZero = false;
    const syncFullyMetallicRoughnessOverride = (): void => {
        if (!forceFullyMetallicRoughnessZero) return;
        for (const material of pbrMaterials()) {
            if ((material.metallicFactor ?? 1) !== 1 || fullyMetallicRoughnessBackups.has(material)) continue;
            const tagged = material as RoughnessTaggedMaterial;
            const original = fullyMetallicRoughnessOriginal in tagged ? tagged[fullyMetallicRoughnessOriginal] : material.roughnessFactor;
            fullyMetallicRoughnessBackups.set(material, original);
            tagged[fullyMetallicRoughnessOriginal] = original;
            material.roughnessFactor = 0;
            markMaterialUboDirty(material);
        }
    };
    const setFullyMetallicRoughnessZero = (enabled: boolean): void => {
        if (forceFullyMetallicRoughnessZero === enabled) return;
        forceFullyMetallicRoughnessZero = enabled;
        if (enabled) {
            syncFullyMetallicRoughnessOverride();
            return;
        }
        for (const [material, roughness] of fullyMetallicRoughnessBackups) {
            material.roughnessFactor = roughness;
            delete (material as RoughnessTaggedMaterial)[fullyMetallicRoughnessOriginal];
            markMaterialUboDirty(material);
        }
        fullyMetallicRoughnessBackups.clear();
    };
    const setPbrSpecularAA = (on: boolean): void => {
        for (const material of pbrMaterials()) {
            material.enableSpecularAA = on;
            material._renderFeatures = undefined;
        }
    };
    setPbrSpecularAA(graphics.specularAA);
    canvas.dataset.specularAa = String(graphics.specularAA);
    // Reflection roughness: the editor's Runtime section multiplies every ship material's authored
    // roughness by this and shows the result, so applying the same number here is what makes the two
    // pictures agree. It multiplies `roughnessFactor`, which scales the ORM texture's green channel,
    // rather than replacing it — a flat value would erase the per-texel variation the kit authored.
    //
    // Ship materials only. The weapon viewmodel is not in the editor's preview at all, so a ship-wide
    // dial tuned against corridor panels must not quietly restyle the gun in the player's hands.
    // The .glb keeps the authored values either way; this is the only place the multiplier exists.
    const reflectionRoughness = manifest?.environment?.reflectionRoughness;
    if (typeof reflectionRoughness === "number" && Number.isFinite(reflectionRoughness) && reflectionRoughness !== 1) {
        for (const material of pbrMaterialsOf(allShipMeshes)) {
            material.roughnessFactor = (material.roughnessFactor ?? 1) * reflectionRoughness;
            markMaterialUboDirty(material);
        }
    }
    const soundManager = new SoundManager();
    soundManager.setEnabled(graphics.soundsEnabled);
    soundManager.setVolume(graphics.soundVolume);
    const behaviorManager = new AquanovaBehaviorManager({
        presets: manifest?.behaviorPresets,
        entities: manifest?.entities,
        doors: manifest?.doors,
        meshesByEntityName: meshesByNodeName,
        entityNameOf: (mesh) => nodeNameOfMesh.get(mesh) ?? mesh.name,
    });
    // Portal meshes ("Portal_*") are doorway markers the exporter emits for culling / door pairing —
    // NOT real geometry. They render as a visible pane spanning the doorway (seen from the corridor)
    // AND sit coplanar with the door leaves, so the weapon pick can hit the portal instead of the door
    // behind it (the door then won't liquefy). Hide them, make them non-pickable, and skip them in the
    // mesh classification below so they never enter the SDF bake either.
    const isPortalMesh = (m: Mesh): boolean => m.name.startsWith("Portal_");
    for (const m of allShipMeshes) {
        if (isPortalMesh(m)) {
            setMeshVisible(m, false);
            (m as { pickable?: boolean }).pickable = false;
        }
    }

    // ── Placement markers (player / weapon start) ────────────────────────────────────────
    // An entity carrying the `player` behavior is a PLACEMENT MARKER, not scenery: something
    // spawns at its node and the marker itself is DISABLED — hidden, non-pickable, skipped by every
    // classification pass below, and made non-collidable by PlayerBehavior when behaviors start.
    // Its `direction` parameter is the glTF-space vector the spawned thing initially faces. The
    // library definitions are empty, so a marker is matched by NAME on the entity's reference.
    const markerMeshes = new Set<Mesh>();
    interface Marker {
        entity: string;
        min: Vec3 | null;
        max: Vec3 | null;
        direction?: number[];
    }
    const resolveMarker = (behaviorName: string): Marker | null => {
        const found = behaviorManager.findEntityWithBehavior(behaviorName);
        if (!found) return null;
        const meshes = found.meshes;
        if (!meshes.length) {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] ${behaviorName} entity "${found.entityName}" matches no ship node — ignoring it`);
            return null;
        }
        // World AABB — already Lite space, since it comes from the loaded ship (mirror applied).
        for (const m of meshes) {
            markerMeshes.add(m);
            setMeshVisible(m, false);
            (m as { pickable?: boolean }).pickable = false;
        }
        const bounds = meshGroupBounds(meshes);
        const min: Vec3 | null = bounds ? [bounds.centre[0] - bounds.half[0], bounds.centre[1] - bounds.half[1], bounds.centre[2] - bounds.half[2]] : null;
        const max: Vec3 | null = bounds ? [bounds.centre[0] + bounds.half[0], bounds.centre[1] + bounds.half[1], bounds.centre[2] + bounds.half[2]] : null;
        return { entity: found.entityName, min, max, direction: found.assignment.direction };
    };
    const playerMarker = resolveMarker("player");
    // Disabled marker geometry is treated exactly like a portal: excluded everywhere below.
    const isDisabledMesh = (m: Mesh): boolean => isPortalMesh(m) || markerMeshes.has(m);

    // Disable KHR_materials_transmission on the ship: the fluid surface composites into the swapchain
    // AFTER the scene renders, but transmission retargets the scene task to an offscreen HDR buffer and
    // appends a final tonemap pass that would overwrite (hide) the fluid. Dropping the refraction
    // sub-feature keeps the scene on our sceneColorRT (like the Liquefactor demo) so the water shows;
    // the glass then renders as a plain surface instead of refracting.
    for (const m of allShipMeshes) {
        const mat = m.material;
        if (mat && isPbrMaterial(mat)) {
            const ss = (mat as unknown as { subsurface?: { refraction?: unknown } }).subsurface;
            if (ss?.refraction) ss.refraction = undefined;
        }
    }

    // Which chunk (room) a mesh belongs to. Taken from the exporter's own scene-graph grouping (the
    // CHUNK_* ancestor recorded during collectMeshes), NOT from a centre-in-AABB test: chunk AABBs
    // overlap and a mesh's centre says nothing reliable about which room authored it. The guess put
    // 165 meshes in CH00_Storage, a chunk the manifest says has 48 — so ~117 corridor meshes were being
    // baked into the storage room's SDF, which is what put a phantom wall across the middle of it.
    const roomOfMesh = (m: Mesh): string => chunkOfMesh.get(m) ?? "unknown";

    // Node name → the kit module it was placed from, so collision policy can be decided per module.
    const moduleOfNode = new Map<string, string>();
    for (const inst of manifest?.instances ?? []) {
        const node = inst.node ?? inst.name;
        if (node && !moduleOfNode.has(node)) moduleOfNode.set(node, inst.module);
    }

    // The exporter stamps every PLACEMENT's glTF node with `extras = { id, module, chunk }`, so a mesh
    // reaches its own instance in one hop. That is the only reliable key: node NAMES are not unique
    // (both stacked crates are called `crate4`), and matching by name handed the second crate the
    // first one's shape — its collider ended up a full crate below itself.
    const instanceIdOfMesh = (m: Mesh): string | undefined => {
        const extras = ownerOfMesh.get(m)?.metadata?.gltf?.extras as { id?: string } | undefined;
        return extras?.id;
    };

    // Instance id → its authored collision primitives, resolved into world space from the LOADED
    // node's matrix. Not from `instances[]`: the manifest's own `space` block says `moduleCollision`
    // is glTF space while `instances[]` is editor space, so composing those two mixes spaces — it
    // looks right on anything symmetrical and is wrong on everything turned. The loaded node already
    // carries the glTF→scene mirror, so node.worldMatrix · shape is correct with no conversion.
    const placementById = new Map<string, { node: string; shapes: WorldCollisionShape[] }>();
    for (const [id, pn] of placementNodes) {
        const shapes = collisionShapesForModule(manifest?.moduleCollision, pn.module);
        if (!shapes) continue;
        placementById.set(id, { node: pn.node.name, shapes: worldShapesForMatrix(pn.node.worldMatrix, shapes) });
    }
    if (placementNodes.size > 0 && Object.keys(manifest?.moduleCollision ?? {}).length > 0 && placementById.size === 0) {
        throw new Error("[aquanova] no GLB placement module ids match manifest.moduleCollision");
    }
    const placementIdsByEntityName = new Map<string, string[]>();
    for (const [id, placement] of placementById) {
        const ids = placementIdsByEntityName.get(placement.node);
        if (ids) ids.push(id);
        else placementIdsByEntityName.set(placement.node, [id]);
    }
    canvas.dataset.collisionPlacements = String(placementById.size);
    /** Flattened view for the debug overlay and QA, tagged with the instance that produced each. */
    const manifestPlacements: Array<{ id: string; node: string; shape: WorldCollisionShape }> = [];
    for (const [id, { node, shapes }] of placementById) for (const shape of shapes) manifestPlacements.push({ id, node, shape });

    const chunkStaticMeshes = new Map<string, Mesh[]>();
    behaviorManager.classifyMeshes(allShipMeshes, {
        isDisabled: isDisabledMesh,
        instanceIdOf: instanceIdOfMesh,
    });

    // Pass 2 — per-room STATIC geometry (everything that neither moves nor melts) for the SDF bake.
    for (const c of manifest?.chunks ?? []) {
        const staticList: Mesh[] = [];
        for (const m of allShipMeshes) {
            if (isDisabledMesh(m)) continue;
            if (behaviorManager.dynamicMeshes.has(m) || behaviorManager.dissolvableMeshes.has(m)) continue;
            if (chunkOfMesh.get(m) !== c.id) continue;
            staticList.push(m);
        }
        chunkStaticMeshes.set(c.id, staticList);
    }
    canvas.dataset.dynamicCount = String(behaviorManager.dynamicMeshes.size);
    canvas.dataset.liquefiableCount = String(behaviorManager.liquefiableMeshes.size);

    const LIQUEFY_EDGE = 0.6; // fire-glow band width at the dissolving boundary
    type PluginMat = { plugins?: unknown[]; _uboVersion: number };
    const liquefyStates = new Map<Mesh, LiquefyState>();

    // Ship materials are shared, so every liquefiable mesh needs its own dissolve state. Install it
    // before lighting and local-environment decorators clone the material: local probe assignments
    // are held outside the material object, so cloning after applyLocalEnvironmentProbes would
    // silently discard them.
    for (const m of behaviorManager.dissolvableMeshes) {
        const mat = m.material;
        if (!mat || !isPbrMaterial(mat)) continue;
        const st: LiquefyState = { hit: [0, 0, 0], frontR: 0, edge: LIQUEFY_EDGE, enabled: false };
        const src = mat as unknown as PluginMat;
        m.material = {
            ...(mat as object),
            _renderFeatures: undefined,
            _uboVersion: 0,
            plugins: [...(src.plugins ?? []), createLiquefyPlugin(() => st, "pbr")],
        } as unknown as Material;
        liquefyStates.set(m, st);
    }

    // ── Ship lighting ──────────────────────────────────────────────────────────────────────────
    // Every enabled ship mesh uses the authored runtime lamps. Build them before registerScene:
    // clustered-light state and the regular-light UBO layout are pipeline inputs.
    const runtimeLitMeshes = new Set(allShipMeshes.filter((mesh) => !isDisabledMesh(mesh)));
    canvas.dataset.runtimeLitCount = String(runtimeLitMeshes.size);
    for (const mesh of weaponMeshes) {
        if (mesh.material && isPbrMaterial(mesh.material)) mesh.material.environmentIntensity = environmentIntensity;
    }
    const lights = buildRuntimeLights(scene, shipRoot, runtimeLitMeshes, chunkOfMesh, manifest?.lights);
    // The held weapon renders in a separate utility scene with a fresh depth buffer. Attach the same
    // clustered container there so its world-space materials receive the authored ship lamps too.
    lights.attachClusteredScene(weaponLayer.scene);
    canvas.dataset.clusteredLightCount = String(lights.clusteredPoint + lights.clusteredSpot);
    if (lights.overflow) console.warn(`[aquanova] ${lights.overflow} non-clustered light(s) dropped: the shared lights UBO is full`);
    // Full mode supplies a POI-ranked candidate set whose weights are calculated from each
    // fragment's world position. Disabled mode gives every mesh one immutable intersecting probe.
    const localEnvironmentController = await applyLocalEnvironmentProbes(scene, [...allShipMeshes, ...weaponMeshes], {
        blendingEnabled: graphics.localCubemapBlending,
    });
    if (localEnvironmentController) {
        canvas.dataset.localEnvironmentCount = String(localEnvironmentController.loaded);
        canvas.dataset.localCubemapBlending = String(localEnvironmentController.blendingEnabled());
        if (localEnvironmentController.missing.length) {
            console.warn("[aquanova] local environments not loaded:", localEnvironmentController.missing.join(", "));
        }
        console.log(
            `[aquanova] local environments: ${localEnvironmentController.loaded} probes, ${localEnvironmentController.assigned} PBR meshes, ${(localEnvironmentController.bytes / 1048576).toFixed(2)} MB`
        );
    }

    // Match the view transform the ship was authored against, straight from the manifest: tone
    // mapping (Khronos PBR Neutral keeps the strength-boosted emissive trim from clipping to white
    // and losing its colour) and exposure, both as the editor's Runtime settings recorded them.
    // Set before registerScene so the first PBR build + the deferred skybox snapshot pick it up.
    const tone = resolveToneMapping(manifest?.environment?.toneMapping);
    for (const targetScene of [scene, weaponLayer.scene]) {
        targetScene.imageProcessing.toneMappingEnabled = tone !== null;
        if (tone) targetScene.imageProcessing.toneMapping = tone;
    }

    /** Assigned just below; retained for the QA hook. */
    let cycleTone: () => void = () => {};
    let setToneMappingIndex: (index: number) => Promise<void> = async () => {};

    // Every algorithm Babylon-Lite ships, plus "None" (no curve at all). The manifest still chooses
    // the startup value — this is a look-dev comparison tool, so it deliberately does NOT persist:
    // reloading returns to what `environment.toneMapping` authored.
    //
    // The curve is a compile-time PBR shader feature (it is injected WGSL, not a uniform), so
    // switching recompiles the affected pipelines. setSceneImageProcessing does that for us; it is
    // async, hence the in-flight guard — hammering the key would otherwise stack rebuilds.
    const TONE_MAPPINGS: ReadonlyArray<{ name: string; tm: ToneMapping | null }> = [
        { name: "Neutral", tm: NeutralToneMapping },
        { name: "ACES", tm: AcesToneMapping },
        { name: "Standard", tm: StandardToneMapping },
        { name: "None", tm: null },
    ];
    let toneIndex = Math.max(
        0,
        TONE_MAPPINGS.findIndex((t) => (tone === null ? t.tm === null : t.tm?.id === tone.id))
    );
    let toneBusy = false;
    const showTone = (): void => {
        canvas.dataset.tonemap = TONE_MAPPINGS[toneIndex]!.name;
    };
    showTone();
    setToneMappingIndex = async (index: number): Promise<void> => {
        if (toneBusy) return;
        toneBusy = true;
        const previous = toneIndex;
        toneIndex = ((index % TONE_MAPPINGS.length) + TONE_MAPPINGS.length) % TONE_MAPPINGS.length;
        const next = TONE_MAPPINGS[toneIndex]!;
        showTone();
        try {
            await Promise.all(
                [scene, weaponLayer.scene].map((targetScene) =>
                    setSceneImageProcessing(targetScene, next.tm ? { toneMappingEnabled: true, toneMapping: next.tm } : { toneMappingEnabled: false })
                )
            );
        } catch (err) {
            toneIndex = previous;
            showTone();
            console.warn("[aquanova] tone mapping switch failed", err);
        } finally {
            toneBusy = false;
        }
    };
    cycleTone = (): void => {
        void setToneMappingIndex(toneIndex + 1);
    };
    let exposure = resolveExposure(manifest?.environment?.exposure);
    // The editor keeps Babylon's neutral contrast; loadEnvironment otherwise leaves Lite at 1.2.
    scene.imageProcessing.exposure = exposure;
    scene.imageProcessing.contrast = 1;
    weaponLayer.scene.imageProcessing.exposure = exposure;
    weaponLayer.scene.imageProcessing.contrast = 1;

    // ── Havok physics: clean per-chunk box-shell colliders (see buildShipColliders) ───────
    const hknp = await HavokPhysics({ locateFile: () => "/HavokPhysics.wasm" });
    const world = createHavokWorld(scene, hknp, { x: 0, y: -9.8, z: 0 });
    // FIXED simulation step. With the default variable step the world advances by the real frame
    // delta, so how far the player capsule sinks before the solver stops it depends on how slow the
    // first frames after load happen to be — and it stays wherever it came to rest. Measured on the
    // same build: the capsule settles at y 0.909 at full speed but 1.036 under 3x CPU throttling, i.e.
    // a 13 cm eye-height difference between machines. Babylon.js fixes its step at 1/60 for the same
    // reason. This also keeps the dynamic props' rest poses reproducible.
    setPhysicsTimestepMs(world, 1000 / 60);
    /** Manifest-authored collision primitives that became static bodies (QA/debug). */
    let manifestShapes: WorldCollisionShape[] = [];
    const manifestColliderBodiesByInstanceId = new Map<string, PhysicsBody[]>();
    // The raw ship geometry is one glTF hierarchy of 568 overlapping modules; a trimesh collider
    // built from it pins the character controller on countless coplanar/overlapping triangles.
    // Instead we collide against clean per-chunk box shells from the manifest — smooth for the
    // capsule, and a natural fit since each chunk is already an axis-aligned room.
    if (manifest) {
        // Per-element collision from the manifest: every placed module that has an authored shape
        // becomes a solid static body, so props are collidable without anyone marking them
        // `dynamic` — that flag now means only "Havok may MOVE this", nothing else.
        //
        // Dissolvable placements are skipped: they already get their own body in the dynamic-prop
        // pass below (from the same manifest shape), and that body is removed when the prop melts.
        // A second static body here would outlive the melt as an invisible wall. Matched by INSTANCE
        // ID from the glTF node's extras, never by name — names repeat across placements.
        const manifestColliders = buildManifestColliders(
            world,
            [...placementById]
                .filter(([id]) => !behaviorManager.dynamicInstanceIds.has(id))
                .flatMap(([instanceId, placement]) => placement.shapes.map((shape) => ({ instanceId, shape })))
        );
        manifestShapes = manifestColliders.map(({ shape }) => shape);
        for (const { body, instanceId } of manifestColliders) {
            const bodies = manifestColliderBodiesByInstanceId.get(instanceId);
            if (bodies) bodies.push(body);
            else manifestColliderBodiesByInstanceId.set(instanceId, [body]);
        }
        canvas.dataset.manifestColliders = String(manifestShapes.length);
    } else {
        // Fallback floor so the player at least stands if the manifest failed to load.
        const node = createTransformNode("shipFloor", -30, -0.2, 0);
        const shape = createPhysicsShape(world, { type: PhysicsShapeType.BOX, parameters: { extents: { x: 140, y: 0.4, z: 24 } } });
        setPhysicsBodyShape(world, createPhysicsBody(world, node, PhysicsMotionType.STATIC), shape);
    }

    // ── First-person player at the manifest spawn ─────────────────────────────────────────
    const CAP_H = PLAYER_CAPSULE_HEIGHT;
    const CAP_R = PLAYER_CAPSULE_RADIUS;
    const FLUID_CAP_R = CAP_R * 2;
    const EYE = 0.62; // camera offset above the capsule centre → ~1.5 m eye level
    const pMin = playerMarker?.min;
    const pMax = playerMarker?.max;
    if (!pMin || !pMax) {
        throw new Error("[aquanova] a player behavior must be assigned to an entity with mesh geometry");
    }
    const spawnPosition = playerCapsuleSpawnPosition(pMin, pMax);
    const character = createPhysicsCharacterController(world, spawnPosition, { capsuleHeight: CAP_H, capsuleRadius: CAP_R });
    character.maxSlopeCosine = MAX_WALKABLE_SLOPE_COSINE;
    const capsuleHeight = (): number => character.shapeOptions.capsuleHeight ?? CAP_H;
    const canStand = (): boolean => {
        const currentHeight = capsuleHeight();
        if (currentHeight >= CAP_H - 1e-4) return true;
        const position = character.getPosition();
        const footY = position.y - currentHeight * 0.5;
        const fromY = footY + currentHeight + 1e-3;
        const toY = footY + CAP_H;
        const ring = CAP_R * 0.85;
        const diagonal = ring / Math.SQRT2;
        const offsets: ReadonlyArray<readonly [number, number]> = [
            [0, 0],
            [ring, 0],
            [-ring, 0],
            [0, ring],
            [0, -ring],
            [diagonal, diagonal],
            [diagonal, -diagonal],
            [-diagonal, diagonal],
            [-diagonal, -diagonal],
        ];
        return offsets.every(([dx, dz]) => !physicsRaycast(world, { x: position.x + dx, y: fromY, z: position.z + dz }, { x: position.x + dx, y: toY, z: position.z + dz }).hasHit);
    };
    const jumpApertureAssist = (forwardX: number, forwardZ: number): JumpApertureAssist | null => {
        const length = Math.sqrt(forwardX * forwardX + forwardZ * forwardZ);
        if (length < 1e-6) {
            return null;
        }
        const fx = forwardX / length;
        const fz = forwardZ / length;
        const rightX = fz;
        const rightZ = -fx;
        const position = character.getPosition();
        const startDistance = CAP_R + 0.02;
        const endDistance = startDistance + 0.75;
        const pathClear = (height: number, radius: number, lateralOffset: number, minimumVertical = Number.NEGATIVE_INFINITY): boolean => {
            const axisHalf = Math.max(0, height * 0.5 - radius);
            const sampleRadius = Math.max(0, radius - 0.03);
            const diagonal = sampleRadius / Math.SQRT2;
            const profile: ReadonlyArray<readonly [number, number]> = [
                [0, -axisHalf - sampleRadius],
                [-diagonal, -axisHalf - diagonal],
                [diagonal, -axisHalf - diagonal],
                [-sampleRadius, -axisHalf],
                [0, -axisHalf],
                [sampleRadius, -axisHalf],
                [-sampleRadius, 0],
                [0, 0],
                [sampleRadius, 0],
                [-sampleRadius, axisHalf],
                [0, axisHalf],
                [sampleRadius, axisHalf],
                [-diagonal, axisHalf + diagonal],
                [diagonal, axisHalf + diagonal],
                [0, axisHalf + sampleRadius],
            ];
            for (const [side, vertical] of profile) {
                if (vertical < minimumVertical) continue;
                const correctedSide = side + lateralOffset;
                const offsetX = rightX * correctedSide;
                const offsetZ = rightZ * correctedSide;
                const y = position.y + vertical;
                if (
                    physicsRaycast(
                        world,
                        { x: position.x + offsetX + fx * startDistance, y, z: position.z + offsetZ + fz * startDistance },
                        { x: position.x + offsetX + fx * endDistance, y, z: position.z + offsetZ + fz * endDistance }
                    ).hasHit
                ) {
                    return false;
                }
            }
            return true;
        };
        const crouchedTop = CROUCH_CAPSULE_HEIGHT * 0.5;
        const lateralOffset = selectClosestClearApertureOffset(
            (candidate) => !pathClear(CAP_H, CAP_R, candidate, crouchedTop + 1e-3) && pathClear(CROUCH_CAPSULE_HEIGHT, CROUCH_CAPSULE_RADIUS, candidate)
        );
        return lateralOffset === null ? null : { lateralOffset };
    };

    // ── Dynamic (dissolvable) props ──────────────────────────────────────────────────────
    // A liquefiable prop needs three things: a Havok body so it is solid, a display root it can be
    // posed through while it melts, and its own bounds for mass/inertia and the particle fill.
    //
    // It no longer needs a BAKED SDF. The fluid collides against the ship's authored collision
    // primitives (see buildCollisionSet), so there are no distance grids anywhere in this demo — no
    // bake at load, no per-room grid, no arena to size, and none of the sign problems that came with
    // baking kit-bashed geometry (an open shell has no reliable inside).
    const MIRROR: [number, number, number] = [-1, 1, 1];

    // One body per NODE, not per mesh: a glTF node with several primitives is ONE object to the
    // player, so treating each primitive separately gave it that many independent rigid bodies and
    // the parts drifted apart the moment physics ran.
    const dynGroups: Mesh[][] = [];
    const grouped = new Set<Mesh>();
    for (const m of behaviorManager.dynamicMeshes) {
        if (grouped.has(m)) {
            continue;
        }
        const group = (nodePrimitives.get(m) ?? [m]).filter((p) => behaviorManager.dynamicMeshes.has(p));
        for (const p of group) {
            grouped.add(p);
        }
        dynGroups.push(group.length ? group : [m]);
    }
    const dynBounds = new Map<Mesh, MeshGroupBounds>(); // keyed by the group's FIRST primitive
    for (const group of dynGroups) {
        const b = meshGroupBounds(group);
        if (b) {
            dynBounds.set(group[0]!, b);
        }
    }

    interface DynBody {
        proxy: SceneNode;
        /** Null for a decal: it dissolves and bounds the fluid, but the surface it sits on collides. */
        body: PhysicsBody | null;
        /** Representative primitive (the group's first) — what call sites that want "the" mesh use. */
        mesh: Mesh;
        /** Every primitive below the entity root reparented under `disp`, moving as one rigid body. */
        meshes: Mesh[];
        /** World bounds of meshes at rest. */
        bounds: MeshGroupBounds;
        /** Display-root offset in its own frame, applied when the body is posed. */
        dispOffset: [number, number, number];
        /** Whether Havok may move this body (manifest `dynamic` behavior). Immovable ones are STATIC. */
        movable: boolean;
        /** Authoritative manifest mass in kilograms. Zero for immovable bodies. */
        mass: number;
        /** Principal body axes that cannot rotate. */
        lockedRotationAxes: readonly PhysicsRotationAxis[];
        /** The manifest placement this prop came from, so its debug shape can be dropped on melt. */
        instanceId: string | undefined;
        disp: SceneNode;
        wriggling?: boolean; // true while THIS body is dissolving: its Havok body still collides/supports,
        // but the wriggle owns the display node — so skip the pose sync below.
    }
    const dynBodies: DynBody[] = [];
    const dynBodyByMesh = new Map<Mesh, DynBody>();
    /** Placements whose prop has melted — their colliders and debug shapes are gone. */
    const removedPlacements = new Set<string>();
    const collisionDisabledPlacements = new Set<string>();
    const collisionDisabledDynBodies = new Set<DynBody>();

    // ── The fluid's collision set ────────────────────────────────────────────────────────────────
    // Authored primitives normally feed both Havok and the fluid. A setCollisionShape behavior can
    // replace only its fluid representation with an analytic special shape. `worldAabb` is used
    // only to select what a simulation needs; the solver evaluates the exact primitive.
    const shapeToPrimitive = (s: WorldCollisionShape): FluidPrimitive =>
        s.kind === "box"
            ? { kind: "box", a: s.centre, b: s.halfExtents ?? [0.05, 0.05, 0.05], rotation: s.rotation ?? [0, 0, 0, 1] }
            : s.kind === "sphere"
              ? { kind: "sphere", a: s.centre, radius: s.radius ?? 0.05 }
              : { kind: s.kind, a: s.pointA ?? s.centre, b: s.pointB ?? s.centre, radius: s.radius ?? 0.05 };
    const primAabb = (p: FluidPrimitive): { min: [number, number, number]; max: [number, number, number] } => {
        if (p.kind === "box") {
            // A rotated box's AABB is bounded by the sum of its half extents — coarse but never too
            // small, which is what matters for a selection test.
            const half = p.b ?? [0, 0, 0];
            const r = Math.sqrt(half[0] * half[0] + half[1] * half[1] + half[2] * half[2]);
            return { min: [p.a[0] - r, p.a[1] - r, p.a[2] - r], max: [p.a[0] + r, p.a[1] + r, p.a[2] + r] };
        }
        const r = p.radius ?? 0;
        const b = p.b ?? p.a;
        const lo = (i: number): number => Math.min(p.a[i]!, b[i]!) - r;
        const hi = (i: number): number => Math.max(p.a[i]!, b[i]!) + r;
        return { min: [lo(0), lo(1), lo(2)], max: [hi(0), hi(1), hi(2)] };
    };
    for (const group of dynGroups) {
        const bounds = dynBounds.get(group[0]!);
        if (!bounds) {
            continue;
        }
        const { centre } = bounds;
        // Display root at the prop's rest pose (the glTF root mirror preserved via `scaling`);
        // Reparent the authored entity root under it (world-preserving) so the hierarchy remains
        // intact. Behaviors such as pickEntity rotate this root once, which rotates every primitive
        // together while preserving their relative transforms.
        const r = createTransformNode(`dyn_disp_${dynBodies.length}`);
        r.scaling.set(MIRROR[0], MIRROR[1], MIRROR[2]);
        r.position.set(centre[0] - MIRROR[0] * centre[0], centre[1] - MIRROR[1] * centre[1], centre[2] - MIRROR[2] * centre[2]);
        addToScene(scene, r);
        const entityRoot = ownerOfMesh.get(group[0]!);
        if (!entityRoot || group.some((mesh) => ownerOfMesh.get(mesh) !== entityRoot)) {
            throw new Error(`[aquanova] dynamic entity "${nodeNameOfMesh.get(group[0]!) ?? group[0]!.name}" has no single authored root`);
        }
        setParent(entityRoot, r);
        // Havok proxy. Its shape comes from the manifest — that is the ONLY thing that decides
        // whether a prop is solid. A prop with no authored collision (the "Caution" sticker on the
        // door) simply gets no rigid body: the surface it is applied to already collides. Nothing
        // is inferred from the element's kind or module path.
        // STATIC unless the manifest assigns the `dynamic` behavior — a fixture stays put no matter
        // how hard the player runs into it, while a genuinely loose prop still falls and can be shoved.
        const movable = group.some((m) => behaviorManager.movableMeshes.has(m));
        const mass = movable ? (behaviorManager.getDynamicMass(group[0]!) ?? 10) : 0;
        const lockedRotationAxes = behaviorManager.getLockedRotationAxes(group[0]!);
        const proxy = createTransformNode(`dyn_proxy_${dynBodies.length}`, centre[0], centre[1], centre[2]);
        const instanceId = instanceIdOfMesh(group[0]!);
        const authoredShapes = instanceId ? (placementById.get(instanceId)?.shapes ?? []) : [];
        if (authoredShapes.length > 1) {
            throw new Error(`[aquanova] dynamic entity "${nodeNameOfMesh.get(group[0]!) ?? group[0]!.name}" requires exactly one authored collision shape`);
        }
        const authored = authoredShapes[0];
        // The proxy sits at the prop's bounds centre and the authored shape is offset WITHIN the body,
        // so the display root (posed from the same pose) and the collider stay in agreement.
        let body: PhysicsBody | null = null;
        if (authored) {
            body = createPhysicsBody(world, proxy, movable ? PhysicsMotionType.DYNAMIC : PhysicsMotionType.STATIC, true);
            setPhysicsBodyShape(world, body, createWorldCollisionShape(world, authored, centre));
            if (movable) {
                setPhysicsBodyMass(world, body, mass);
                lockPhysicsBodyRotationAxes(world, body, lockedRotationAxes);
            }
        }
        // The display root's offset in its OWN frame, so a moved body poses it as
        // `position = pose ∘ (−displayScale · centre)` — the transform the floating-body system used
        // to apply, kept identical so a melting prop still lines up with its water.
        const dispOffset: [number, number, number] = [-MIRROR[0] * centre[0], -MIRROR[1] * centre[1], -MIRROR[2] * centre[2]];
        const dyn: DynBody = {
            proxy,
            body,
            mesh: group[0]!,
            meshes: group,
            bounds,
            movable,
            mass,
            lockedRotationAxes,
            instanceId,
            disp: r,
            dispOffset,
        };
        dynBodies.push(dyn);
        for (const m of group) {
            dynBodyByMesh.set(m, dyn);
        }
    }

    // Each physics step, pose a MOVABLE prop's display root from its Havok body, and refresh the
    // primitive the fluid collides against. An immovable prop never moves, so it costs nothing.
    behaviorManager.events.on("physicsStep", () => {
        for (const d of dynBodies) {
            if (!d.movable) continue; // fixture: posed once at load
            if (!d.body) continue; // decal: no rigid body to read a pose from
            if (d.wriggling) continue; // dissolving: the wriggle owns the display
            const p = d.proxy.position;
            const q = d.proxy.rotationQuaternion;
            const off = qRot(q, d.dispOffset);
            d.disp.position.set(p.x + off[0], p.y + off[1], p.z + off[2]);
            d.disp.rotationQuaternion.set(q.x, q.y, q.z, q.w);
        }
        // Movable props also move the fluid's boundary: rewrite just their slots in each running
        // sim's primitive buffer. Only the changed primitives are uploaded, not the whole set.
        for (const a of activeSims) {
            for (const m of a.collision.moving) {
                if (a.collision.prims[m.slot]?.active === false) continue;
                const live = livePrim(m.body);
                if (!live) continue;
                a.collision.prims[m.slot] = live;
                packPrimitive(a.collision.scratch, m.slot, live);
                device.queue.writeBuffer(a.collision.buffer, (PRIM_HEADER + m.slot * PRIM_STRIDE) * 4, a.collision.scratch, PRIM_HEADER + m.slot * PRIM_STRIDE, PRIM_STRIDE);
            }
        }
    });
    canvas.dataset.dynBodies = String(dynBodies.length);

    // Static collision primitives for the fluid: every authored placement EXCEPT the dissolvable
    // ones, which move (and later vanish) and so are tracked per body below.
    const staticPrims: Array<{
        id?: string;
        entityName?: string;
        prim: FluidPrimitive;
        min: [number, number, number];
        max: [number, number, number];
    }> = [];
    for (const [id, pl] of placementById) {
        if (behaviorManager.dynamicInstanceIds.has(id)) continue;
        for (const s of pl.shapes) {
            const prim = shapeToPrimitive(s);
            staticPrims.push({ id, entityName: pl.node, prim, ...primAabb(prim) });
        }
    }
    /** A dissolvable prop's primitive, kept in its proxy's REST frame so a moved body can re-pose it. */
    const dynPrims = new Map<DynBody, { local: FluidPrimitive; rest: [number, number, number] }>();
    for (const d of dynBodies) {
        const s = d.instanceId ? placementById.get(d.instanceId)?.shapes[0] : undefined;
        if (!s) {
            continue;
        }
        const c = d.bounds.centre;
        const p = shapeToPrimitive(s);
        // Express relative to the proxy's rest position; the proxy starts unrotated at `centre`.
        // localizePrimitive knows a box's `b` is half extents and must not be re-based — see its doc.
        dynPrims.set(d, { local: localizePrimitive(p, c), rest: [c[0], c[1], c[2]] });
    }

    const setEntityFluidSimShape = (entityName: string, registration: FluidSimShapeRegistration): void => {
        const shape = registration.shape;
        const handledDynamicBodies = new Set<DynBody>();
        const handledStaticOwners = new Set<SceneNode>();
        const replacedInstanceIds = new Set<string>();
        const replacements: (typeof staticPrims)[number][] = [];
        let targets = 0;

        for (const mesh of registration.meshes) {
            const dynamicBody = dynBodyByMesh.get(mesh);
            if (dynamicBody) {
                if (handledDynamicBodies.has(dynamicBody)) continue;
                handledDynamicBodies.add(dynamicBody);
                const primitive = hollowCylinderPrimitiveForMatrix(mesh.worldMatrix, shape.start, shape.height, shape.innerRadius, shape.outerRadius);
                const centre = dynamicBody.bounds.centre;
                dynPrims.set(dynamicBody, {
                    local: localizePrimitive(primitive, centre),
                    rest: [centre[0], centre[1], centre[2]],
                });
                targets++;
                continue;
            }

            const owner = ownerOfMesh.get(mesh) ?? mesh;
            if (handledStaticOwners.has(owner)) continue;
            handledStaticOwners.add(owner);
            const instanceId = instanceIdOfMesh(mesh);
            if (instanceId && replacedInstanceIds.has(instanceId)) continue;
            if (instanceId) replacedInstanceIds.add(instanceId);
            const primitive = hollowCylinderPrimitiveForMatrix(mesh.worldMatrix, shape.start, shape.height, shape.innerRadius, shape.outerRadius);
            replacements.push({
                id: instanceId,
                entityName,
                prim: primitive,
                ...primAabb(primitive),
            });
            targets++;
        }

        if (targets === 0) {
            throw new Error(`[aquanova] setCollisionShape entity "${entityName}" produced no fluid simulation shapes`);
        }
        for (let i = staticPrims.length - 1; i >= 0; i--) {
            const id = staticPrims[i]!.id;
            if (id && replacedInstanceIds.has(id)) staticPrims.splice(i, 1);
        }
        staticPrims.push(...replacements);
    };

    /** Hamilton product — compose two rotations (`a` applied after `b`). */
    const quatMul = (a: readonly [number, number, number, number], b: readonly [number, number, number, number]): [number, number, number, number] => [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
    /** Rotate `v` by quaternion `q`. */
    const qRot = (q: { x: number; y: number; z: number; w: number }, v: readonly [number, number, number]): [number, number, number] => {
        const tx = 2 * (q.y * v[2] - q.z * v[1]);
        const ty = 2 * (q.z * v[0] - q.x * v[2]);
        const tz = 2 * (q.x * v[1] - q.y * v[0]);
        return [v[0] + q.w * tx + (q.y * tz - q.z * ty), v[1] + q.w * ty + (q.z * tx - q.x * tz), v[2] + q.w * tz + (q.x * ty - q.y * tx)];
    };
    const syncDynDisplay = (d: DynBody): void => {
        const p = d.proxy.position;
        const q = d.proxy.rotationQuaternion;
        const off = qRot(q, d.dispOffset);
        d.disp.position.set(p.x + off[0], p.y + off[1], p.z + off[2]);
        d.disp.rotationQuaternion.set(q.x, q.y, q.z, q.w);
    };
    let antiGravityGrabbedBody: DynBody | null = null;
    grabDynamicWithAntiGravity = (mesh): boolean => {
        const d = dynBodyByMesh.get(mesh);
        if (!isEntityCollisionActive(mesh) || !d?.movable || !d.body || d.wriggling || !dynBodies.includes(d)) {
            return false;
        }
        if (antiGravityGrabbedBody && antiGravityGrabbedBody !== d) {
            releaseAntiGravityGrab(0);
        }
        antiGravityGrabbedBody = d;
        canvas.dataset.antiGravityGrabbed = nodeNameOfMesh.get(d.mesh) ?? d.mesh.name;
        canvas.dataset.antiGravityMass = String(d.mass);
        setPhysicsBodyMotionType(world, d.body, PhysicsMotionType.ANIMATED);
        setPhysicsBodyLinearVelocity(world, d.body, { x: 0, y: 0, z: 0 });
        setPhysicsBodyAngularVelocity(world, d.body, { x: 0, y: 0, z: 0 });
        return true;
    };
    updateAntiGravityGrab = (deltaMs): boolean => {
        const d = antiGravityGrabbedBody;
        if (!d?.body || !dynBodies.includes(d)) {
            antiGravityGrabbedBody = null;
            canvas.dataset.antiGravityGrabbed = "none";
            canvas.dataset.antiGravityMass = "";
            canvas.dataset.antiGravityBlocked = "false";
            return false;
        }
        if (d.wriggling || !isEntityCollisionActive(d.mesh)) {
            releaseAntiGravityGrab(0);
            return false;
        }
        const p = d.proxy.position;
        const playerPosition = character.getPosition();
        const heldDx = p.x - playerPosition.x;
        const heldDy = p.y - playerPosition.y;
        const heldDz = p.z - playerPosition.z;
        const maxHeldDistance = playerBehavior?.maxHeldObjectDistance ?? 8;
        if (heldDx * heldDx + heldDy * heldDy + heldDz * heldDz > maxHeldDistance * maxHeldDistance) {
            releaseAntiGravityGrab(0);
            return false;
        }
        const dx = cam.target.x - cam.position.x;
        const dy = cam.target.y - cam.position.y;
        const dz = cam.target.z - cam.position.z;
        const invLength = 1 / (Math.sqrt(dx * dx + dy * dy + dz * dz) || 1);
        const targetX = cam.position.x + dx * invLength * 2.5;
        const targetY = cam.position.y + dy * invLength * 2.5;
        const targetZ = cam.position.z + dz * invLength * 2.5;
        const blend = 1 - Math.exp((-Math.max(0, deltaMs) * 12) / 1000);
        const displacement: [number, number, number] = [(targetX - p.x) * blend, (targetY - p.y) * blend, (targetZ - p.z) * blend];
        const body = d.body;
        const shape = body._shape;
        const collisionMove = shape
            ? antiGravityCollisionSlideDisplacement(displacement, (offset, step) =>
                  shapeCast(world, {
                      shape,
                      rotation: d.proxy.rotationQuaternion,
                      startPosition: { x: p.x + offset[0], y: p.y + offset[1], z: p.z + offset[2] },
                      endPosition: { x: p.x + offset[0] + step[0], y: p.y + offset[1] + step[1], z: p.z + offset[2] + step[2] },
                      ignoreBody: body,
                  })
              )
            : { movement: displacement, blocked: false };
        canvas.dataset.antiGravityBlocked = String(collisionMove.blocked);
        setPhysicsBodyTransform(
            world,
            body,
            {
                x: p.x + collisionMove.movement[0],
                y: p.y + collisionMove.movement[1],
                z: p.z + collisionMove.movement[2],
            },
            d.proxy.rotationQuaternion
        );
        syncDynDisplay(d);
        return true;
    };
    releaseAntiGravityGrab = (throwSpeed): void => {
        const d = antiGravityGrabbedBody;
        antiGravityGrabbedBody = null;
        canvas.dataset.antiGravityGrabbed = "none";
        canvas.dataset.antiGravityMass = "";
        canvas.dataset.antiGravityThrowSpeed = String(throwSpeed);
        canvas.dataset.antiGravityBlocked = "false";
        if (!d?.body || !dynBodies.includes(d)) {
            return;
        }
        setPhysicsBodyMotionType(world, d.body, PhysicsMotionType.DYNAMIC);
        const dx = cam.target.x - cam.position.x;
        const dy = cam.target.y - cam.position.y;
        const dz = cam.target.z - cam.position.z;
        const invLength = 1 / (Math.sqrt(dx * dx + dy * dy + dz * dz) || 1);
        setPhysicsBodyLinearVelocity(world, d.body, {
            x: dx * invLength * throwSpeed,
            y: dy * invLength * throwSpeed,
            z: dz * invLength * throwSpeed,
        });
        setPhysicsBodyAngularVelocity(world, d.body, { x: 0, y: 0, z: 0 });
    };
    behaviorManager.events.on("hitWithPistol", ({ mesh, point, direction, impulse }) => {
        canvas.dataset.lastPistolHit = nodeNameOfMesh.get(mesh) ?? mesh.name;
        const dynamicBody = dynBodyByMesh.get(mesh);
        if (!dynamicBody?.movable || !dynamicBody.body || dynamicBody.wriggling || !dynBodies.includes(dynamicBody) || impulse === 0) {
            return;
        }
        applyPhysicsImpulse(
            world,
            dynamicBody.body,
            { x: direction[0] * impulse, y: direction[1] * impulse, z: direction[2] * impulse },
            { x: point[0], y: point[1], z: point[2] }
        );
    });
    /** A dissolvable prop's primitive at its CURRENT pose, with the velocity the solver needs. */
    const livePrim = (d: DynBody): FluidPrimitive | null => {
        const e = dynPrims.get(d);
        if (!e) return null;
        const p = d.proxy.position;
        const q = d.proxy.rotationQuaternion;
        const put = (v: readonly [number, number, number]): [number, number, number] => {
            const r = qRot(q, v);
            return [p.x + r[0], p.y + r[1], p.z + r[2]];
        };
        const vel = d.movable && d.body ? getPhysicsBodyLinearVelocity(world, d.body) : { x: 0, y: 0, z: 0 };
        const base = e.local;
        const out: FluidPrimitive = { ...base, a: put(base.a), velocity: [vel.x, vel.y, vel.z] };
        if (base.b) out.b = base.b && base.kind !== "box" ? put(base.b) : base.b; // a box's `b` is half extents, not a point
        if (base.kind === "box") out.rotation = quatMul([q.x, q.y, q.z, q.w], base.rotation ?? [0, 0, 0, 1]);
        return out;
    };

    const livePlayerPrimitive = (): FluidPrimitive => {
        const p = character.getPosition();
        const velocity = character.getVelocity();
        const axisHalf = capsuleHeight() * 0.5 - CAP_R;
        return {
            kind: "capsule",
            a: [p.x, p.y + axisHalf, p.z],
            // Extend only the fluid boundary below the physical capsule so walking through shallow
            // fluid displaces it instead of merely grazing its surface at floor level.
            b: [p.x, p.y - axisHalf - CAP_R * 0.5, p.z],
            radius: FLUID_CAP_R,
            velocity: [velocity.x, velocity.y, velocity.z],
            active: !(playerBehavior?.isNoclip ?? false),
        };
    };

    // Which chunk (room) contains the camera. Camera is Lite-space; convert X back to glTF to test
    // against the manifest AABBs (which are glTF-space).
    const roomAt = (): string => {
        const gx = -cam.position.x;
        const gz = cam.position.z;
        return chunkAt(manifest?.chunks ?? [], gx, gz)?.id ?? "—";
    };
    let poiEnvironmentPosition: Vec3 | undefined;
    let fluidEnvironment: EnvironmentTextures | undefined;
    let applyFluidEnvironment: ((environment: EnvironmentTextures) => void) | undefined;
    const publishLocalCubemapProbes = (blend: LocalEnvironmentBlendInfo | undefined): void => {
        canvas.dataset.localCubemapProbes = blend?.probes.map((probe) => `${probe.id}:${probe.weight.toFixed(4)}`).join(",") ?? "";
    };
    const syncPoiEnvironment = (): void => {
        const current: Vec3 = [cam.position.x, cam.position.y, cam.position.z];
        if (
            poiEnvironmentPosition &&
            (current[0] - poiEnvironmentPosition[0]) ** 2 + (current[1] - poiEnvironmentPosition[1]) ** 2 + (current[2] - poiEnvironmentPosition[2]) ** 2 <= 1e-6
        ) {
            return;
        }
        const previousDominant = localEnvironmentController?.blendInfo().dominantProbeId;
        const blend = localEnvironmentController?.updatePoi(current);
        const probeId = blend?.dominantProbeId;
        if (probeId !== previousDominant) {
            fluidEnvironment = localEnvironmentController?.dominantEnvironment();
            if (fluidEnvironment) applyFluidEnvironment?.(fluidEnvironment);
        }
        poiEnvironmentPosition = current;
        const chunkId = chunkAt(manifest?.chunks ?? [], -current[0], current[2])?.id;
        canvas.dataset.liquefactorChunk = chunkId ?? "—";
        canvas.dataset.liquefactorProbe = probeId ?? "—";
        publishLocalCubemapProbes(blend);
    };
    syncPoiEnvironment();
    const gameplayHiddenMeshes = new Set<Mesh>();
    const behaviorHiddenMeshes = new Set<Mesh>();
    const movableMeshes = new Set(dynBodies.filter((body) => body.movable).flatMap((body) => body.meshes));
    const classifierMeshes = allShipMeshes.filter((mesh) => !isDisabledMesh(mesh));
    const classifierBounds = meshGroupBounds(classifierMeshes);
    const exteriorClassifyStart = performance.now();
    const exteriorClassifier = classifierBounds
        ? createExteriorMeshClassifier({
              engine,
              scene,
              meshes: classifierMeshes,
              bounds: classifierBounds,
              chunks: manifest?.chunks ?? [],
              portals: manifest?.portals ?? [],
          })
        : null;
    let exteriorMeshes: ReadonlySet<Mesh> = new Set();
    let exteriorReady = exteriorClassifier === null;
    const portalVisibility = createPortalVisibility({
        canvas,
        camera: cam,
        aspectRatio: () => engine.canvas.width / Math.max(1, engine.canvas.height),
        roomAt,
        viewerBounds: () => {
            const position = character.getPosition();
            const halfHeight = capsuleHeight() * 0.5;
            return {
                min: [position.x - CAP_R, position.y - halfHeight, position.z - CAP_R],
                max: [position.x + CAP_R, position.y + halfHeight, position.z + CAP_R],
            };
        },
        chunks: manifest?.chunks ?? [],
        portals: manifest?.portals ?? [],
        meshes: allShipMeshes,
        chunkOfMesh,
        isExcluded: isDisabledMesh,
        dynamicMeshes: movableMeshes,
        dynamicChunkOfMesh: (mesh) => {
            const position = dynBodyByMesh.get(mesh)?.proxy.position;
            return position ? chunkAt(manifest?.chunks ?? [], -position.x, position.z)?.id : undefined;
        },
        exteriorMeshes: () => exteriorMeshes,
        canRestore: (mesh) => !gameplayHiddenMeshes.has(mesh) && !behaviorHiddenMeshes.has(mesh),
        onChunkVisibilityChanged: (chunkId, visible) => {
            behaviorManager.events.emit("entityEvent", {
                name: chunkId,
                event: visible ? "visible" : "notVisible",
            });
        },
    });
    registerDoorEntityEventHandlers(behaviorManager.events, manifest?.doors ?? [], (door, enabled) => {
        portalVisibility.setDoorEnabled(door, enabled);
    });
    const entityCollisionStates = new Map<string, boolean>();
    const entityFluidCollisionStates = new Map<string, boolean>();
    isEntityCollisionActive = (mesh) => entityCollisionStates.get(nodeNameOfMesh.get(mesh) ?? mesh.name) !== false;
    let applyEntityCollisionState: ((entityName: string, active: boolean) => void) | null = null;
    let applyEntityFluidCollisionState: ((entityName: string, active: boolean) => void) | null = null;
    let retirePickedEntity: ((entityName: string, meshes: readonly Mesh[]) => void) | null = null;
    let explodeEntity: ((entityName: string, meshes: readonly Mesh[], options: ExplosionOptions) => void) | null = null;
    const pendingExplosions: Array<{ entityName: string; meshes: readonly Mesh[]; options: ExplosionOptions }> = [];
    registerEntityCollisionEventHandlers(behaviorManager.events, (entityName, active, scope) => {
        entityFluidCollisionStates.set(entityName, active);
        if (scope === "fluidSimulation") {
            applyEntityFluidCollisionState?.(entityName, active);
        } else {
            entityCollisionStates.set(entityName, active);
            applyEntityCollisionState?.(entityName, active);
        }
    });
    registerMeshEntityEventHandlers(behaviorManager.events, scene, meshesByNodeName, createPersistentMeshEntityEventOperations(behaviorHiddenMeshes));

    // ── Debug overlays (I inspect, B colliders, L lights, F portal frusta) — see ./debug/ ─────────
    // Gated on LAB_DEBUG so a release bundle folds these to `null` and drops the modules entirely.
    let picker: ReturnType<typeof createGpuPicker> | null = null;
    const getPicker = (): ReturnType<typeof createGpuPicker> => (picker ??= createGpuPicker(scene));
    const inspectOverlay = !LAB_DEBUG
        ? null
        : createInspectOverlay({
              engine,
              scene,
              canvas,
              world,
              cam,
              character,
              getCapsuleHeight: capsuleHeight,
              roomsAt: () => portalVisibility.viewerChunks(),
              getPicker,
              nodeNameOf: (m: Mesh) => nodeNameOfMesh.get(m),
              nodePrimitivesOf: (m: Mesh) => nodePrimitives.get(m),
              chunkIdsOf: (m: Mesh) => [...new Set((nodePrimitives.get(m) ?? [m]).flatMap((primitive) => portalVisibility.chunkIds(primitive)))].sort(),
              isDynamic: (m: Mesh) => (nodePrimitives.get(m) ?? [m]).some((primitive) => movableMeshes.has(primitive)),
              isNoclip: () => playerBehavior?.isNoclip ?? false,
          });
    // Fluid GPU profiling. The sim is stepped directly onto the frame encoder (not as a frame-graph
    // task), so the engine's per-task timer structurally cannot see it; the fluid's own profiler is
    // the only way to attribute that time. Created once if the device supports timestamp queries,
    // but only ATTACHED while the perf panel is open — detached, `profiler?.pass()` returns
    // undefined and no pass requests timestamp writes, so it costs nothing.
    let fluidProfiler: FluidSimulationProfiler | null = null;
    try {
        fluidProfiler = engine._device.features.has("timestamp-query") ? createFluidSimulationProfiler(engine) : null;
    } catch {
        fluidProfiler = null; // never worth failing the demo over a profiler
    }
    let fluidProfilerOn = false;
    let gamePaused = false;
    const perfOverlay = createPerfOverlay({
        engine,
        portalWorkload: () => portalVisibility.stats(),
        exteriorChunks: () => portalVisibility.exteriorChunks(),
        viewpoint: () => ({ position: cam.position, target: cam.target }),
        fluidWorkload: () => {
            const behaviorSims = [...behaviorFluidSims.values()].flatMap(({ activated, sim }) => (activated && sim ? [sim] : []));
            const electricity = behaviorManager.fluidSimulations.electricityStats();
            return {
                simulations: activeSims.length + behaviorSims.length,
                pausedSimulations: [...behaviorFluidSims.values()].filter(({ activated, sim, state }) => activated && sim !== null && state === "paused").length,
                particles: activeSims.reduce((total, active) => total + active.sim.count, 0) + behaviorSims.reduce((total, sim) => total + sim.count, 0),
                electrifiedDomains: electricity.electrified,
                electrifiers: electricity.electrifiers,
                electricityReceivers: electricity.receivers,
                electricityQueryPairs: electricity.queryPairs,
                playerElectrifiedParticles: playerBehavior?.currentElectrifiedParticleCount ?? 0,
            };
        },
        fluidStages: () => (fluidProfilerOn && fluidProfiler ? readFluidSimulationProfiler(fluidProfiler) : null),
        onToggle: (on) => {
            fluidProfilerOn = on && fluidProfiler !== null;
            attachFluidProfiler();
        },
    });
    const colliderOverlay = !LAB_DEBUG
        ? null
        : createColliderOverlay({
              engine,
              scene,
              canvas,
              manifestShapes: manifestPlacements,
              isRemoved: (id) => removedPlacements.has(id) || collisionDisabledPlacements.has(id),
              injectedPrims: () => [
                  ...activeSims.map((a) => ({
                      sim: nodeNameOfMesh.get(a.mesh) ?? a.mesh.name,
                      // The player capsule is always reserved in every set and is too large to be
                      // useful in this visualization. Keep it in the simulation, but do not draw it.
                      prims: visibleInjectedPrimitives(a.collision.prims, a.collision.playerSlot),
                      holes: [] as FluidPrimitive[],
                  })),
                  ...[...behaviorFluidSims.values()].flatMap((entry) =>
                      entry.collision
                          ? [
                                {
                                    sim: entry.registration.entityName,
                                    prims: visibleInjectedPrimitives(entry.collision.prims, entry.collision.playerSlot),
                                    holes: entry.collision.holes?.holes.map((hole) => hole.primitive) ?? [],
                                },
                            ]
                          : []
                  ),
              ],
              dynBodies: () => dynBodies.map((d) => ({ name: nodeNameOfMesh.get(d.mesh) ?? d.mesh.name, position: d.proxy.position, half: d.bounds.half })),
              roomAt,
          });
    const debugUtility = !LAB_DEBUG ? null : createUtilityLayer(engine, scene, { addDefaultLight: false });
    const lightOverlay = !LAB_DEBUG
        ? null
        : createLightOverlay({
              engine,
              scene: debugUtility!.scene,
              canvas,
              lights: lights.lights,
              roomAt,
          });
    const probeOverlay =
        !LAB_DEBUG || !localEnvironmentController
            ? null
            : createProbeOverlay({
                  engine,
                  scene: debugUtility!.scene,
                  canvas,
                  probes: localEnvironmentController.probeVolumes(),
                  blendInfo: () => localEnvironmentController.blendInfo(),
                  blendingEnabled: () => localEnvironmentController.blendingEnabled(),
                  setDebugEnabled: (enabled) => localEnvironmentController.setDebugEnabled(enabled),
              });
    const portalOverlay = !LAB_DEBUG
        ? null
        : createPortalOverlay({
              engine,
              scene: debugUtility!.scene,
              canvas,
              camera: cam,
              maxFrusta: manifest?.portals.length ?? 0,
              traversals: () => portalVisibility.traversals(),
          });
    const inspectOn = (): boolean => inspectOverlay?.isOn() ?? false;

    // Assigned by the Milestone D fluid block below. Liquefiable behaviors call this service after
    // accepting a typed `hitWithWeapon` event addressed to their mesh.
    let liquefyMesh: (mesh: Mesh, hitPoint?: readonly [number, number, number] | null, config?: LiquefiableBehaviorConfig) => void = () => {};
    let requestFusionResume: () => number | null = () => null;
    let resolveFusionResume: (token: number, mesh: Mesh | null) => "resumed" | "start-new" | "await-target" | "continue" = () => "continue";
    let resolveFusionTarget: (mesh: Mesh | null, point: readonly [number, number, number] | null) => Mesh | null = (mesh) => mesh;
    let fusionTargetLost: (mesh: Mesh | null) => boolean = () => false;
    let reverseFusion: () => boolean = () => false;
    // Swap fluid collision between the full ship and a bare ground plane for comparison.
    let toggleGroundOnly: () => void = () => {};
    let toggleMsaa: () => void = () => {};
    let toggleSmaa: () => void = () => {};
    let toggleTaa: () => void = () => {};
    let toggleSpecularAA: () => void = () => {};
    const setLocalCubemapBlending = (on: boolean): void => {
        graphics.localCubemapBlending = on;
        saveGraphicsSettings(graphics);
        localEnvironmentController?.setBlendingEnabled(on);
        canvas.dataset.localCubemapBlending = String(on);
        publishLocalCubemapProbes(localEnvironmentController?.blendInfo());
    };
    const toggleLocalCubemapBlending = (): void => setLocalCubemapBlending(!graphics.localCubemapBlending);
    let setSpecularAAEnabled: (on: boolean) => Promise<void> = async () => {};
    const toggleNearestLight = (): void => {
        const position = character.getPosition();
        const result = lights.toggleNearest([position.x, position.y, position.z]);
        canvas.dataset.nearestLight = result?.id ?? "none";
        canvas.dataset.nearestLightEnabled = result ? String(result.enabled) : "";
    };
    let setSsaa: (scale: number) => void = () => {};
    /** Re-applies the fluid profiler to the surface task and every live sim. Assigned once they exist. */
    let attachFluidProfiler: () => void = () => {};
    let retargetTaa: (task: RenderTask) => void = () => {};
    // Assigned once each subsystem exists. Split from the toggles so the three modes can switch each
    // other off without recursing: a setter never touches another mode, only the toggles do.
    let setMsaaEnabled: (on: boolean) => void = () => {};
    let setSmaaEnabled: (on: boolean) => void = () => {};
    let setTaaEnabled: (on: boolean) => void = () => {};
    let controlPanel: AquanovaControlPanel | null = null;
    const setLiquefactorModel = (model: LiquefactorModel): void => {
        for (const viewmodel of weaponViewmodels) {
            viewmodel.select(model);
        }
        graphics.liquefactorModel = model;
        saveGraphicsSettings(graphics);
        canvas.dataset.liquefactorModel = model;
    };
    const setWeaponSwayEnabled = (enabled: boolean): void => {
        graphics.weaponSway = enabled;
        saveGraphicsSettings(graphics);
        for (const viewmodel of weaponViewmodels) {
            viewmodel.setSwayEnabled(enabled);
        }
        canvas.dataset.weaponSway = String(enabled);
    };
    const setSoundsEnabled = (enabled: boolean): void => {
        graphics.soundsEnabled = enabled;
        saveGraphicsSettings(graphics);
        soundManager.setEnabled(enabled);
        canvas.dataset.soundsEnabled = String(enabled);
    };
    const setSoundVolume = (volume: number): void => {
        graphics.soundVolume = Math.max(0, Math.min(1, volume));
        saveGraphicsSettings(graphics);
        soundManager.setVolume(graphics.soundVolume);
        canvas.dataset.soundVolume = String(graphics.soundVolume);
    };
    canvas.dataset.weaponSway = String(graphics.weaponSway);
    canvas.dataset.soundsEnabled = String(graphics.soundsEnabled);
    canvas.dataset.soundVolume = String(graphics.soundVolume);

    const behaviorCollisionBodiesByEntityName = new Map<string, PhysicsBody[]>();
    const removeBodyAndShape = (body: PhysicsBody): void => {
        const shape = body._shape;
        removePhysicsBody(world, body);
        if (shape) {
            releasePhysicsShape(world, shape);
            body._shape = null;
        }
    };
    const collisionBodiesOfEntity = (entityName: string): PhysicsBody[] => {
        const bodies = new Set(behaviorCollisionBodiesByEntityName.get(entityName) ?? []);
        for (const instanceId of placementIdsByEntityName.get(entityName) ?? []) {
            for (const body of manifestColliderBodiesByInstanceId.get(instanceId) ?? []) bodies.add(body);
        }
        for (const mesh of meshesByNodeName.get(entityName) ?? []) {
            const body = dynBodyByMesh.get(mesh)?.body;
            if (body) bodies.add(body);
        }
        return [...bodies];
    };
    const setEntityCollisionShape = (entityName: string, type: "aabb" | "mesh", fluidSimShape?: FluidSimShapeRegistration): void => {
        const meshes = meshesByNodeName.get(entityName) ?? [];
        if (meshes.length === 0) throw new Error(`[aquanova] setCollisionShape entity "${entityName}" has no meshes`);
        const made: PhysicsBody[] = [];
        const replacedDynamicBodies = new Set<DynBody>();
        const replacedOwners = new Set<SceneNode>();
        for (const mesh of meshes) {
            const dynamicBody = dynBodyByMesh.get(mesh);
            if (dynamicBody) {
                if (replacedDynamicBodies.has(dynamicBody)) continue;
                replacedDynamicBodies.add(dynamicBody);
                if (type === "mesh" && dynamicBody.movable) {
                    throw new Error(`[aquanova] setCollisionShape type "mesh" requires static entity "${entityName}"`);
                }
                const shape =
                    type === "mesh"
                        ? createPhysicsShape(world, { type: PhysicsShapeType.MESH, mesh: dynamicBody.disp, includeChildMeshes: true })
                        : createPhysicsShape(world, {
                              type: PhysicsShapeType.BOX,
                              parameters: {
                                  extents: {
                                      x: dynamicBody.bounds.half[0] * 2,
                                      y: dynamicBody.bounds.half[1] * 2,
                                      z: dynamicBody.bounds.half[2] * 2,
                                  },
                              },
                          });
                if (type === "aabb" && dynamicBody.body) {
                    const previousShape = dynamicBody.body._shape;
                    setPhysicsBodyShape(world, dynamicBody.body, shape);
                    if (previousShape) releasePhysicsShape(world, previousShape);
                } else {
                    const bodyNode = type === "mesh" ? dynamicBody.disp : dynamicBody.proxy;
                    const body = createPhysicsBody(world, bodyNode, dynamicBody.movable ? PhysicsMotionType.DYNAMIC : PhysicsMotionType.STATIC);
                    setPhysicsBodyShape(world, body, shape);
                    if (dynamicBody.body) removeBodyAndShape(dynamicBody.body);
                    dynamicBody.body = body;
                }
                if (dynamicBody.movable && dynamicBody.body) {
                    setPhysicsBodyMass(world, dynamicBody.body, dynamicBody.mass);
                    lockPhysicsBodyRotationAxes(world, dynamicBody.body, dynamicBody.lockedRotationAxes);
                }
                made.push(dynamicBody.body!);
                continue;
            }
            const owner = ownerOfMesh.get(mesh);
            if (!owner || replacedOwners.has(owner)) continue;
            replacedOwners.add(owner);
            const instanceId = instanceIdOfMesh(mesh);
            const bounds = meshGroupBounds(nodePrimitives.get(mesh) ?? [mesh]);
            if (!bounds) continue;
            let shape: ReturnType<typeof createPhysicsShape>;
            let position: readonly [number, number, number];
            let rotation: { x: number; y: number; z: number; w: number } | null = null;
            if (type === "aabb") {
                shape = createPhysicsShape(world, {
                    type: PhysicsShapeType.BOX,
                    parameters: {
                        extents: {
                            x: bounds.half[0] * 2,
                            y: bounds.half[1] * 2,
                            z: bounds.half[2] * 2,
                        },
                    },
                });
                position = bounds.centre;
            } else {
                const worldTransform = mat4Decompose(owner.worldMatrix);
                const localScale = [owner.scaling.x, owner.scaling.y, owner.scaling.z] as const;
                owner.scaling.set(worldTransform.scale.x, worldTransform.scale.y, worldTransform.scale.z);
                try {
                    shape = createPhysicsShape(world, { type: PhysicsShapeType.MESH, mesh: owner, includeChildMeshes: true });
                } finally {
                    owner.scaling.set(...localScale);
                }
                position = [worldTransform.translation.x, worldTransform.translation.y, worldTransform.translation.z];
                rotation = worldTransform.rotation;
            }
            const proxy = createTransformNode(`behavior_${type}_${entityName}_${made.length}`, position[0], position[1], position[2]);
            if (rotation) proxy.rotationQuaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
            const body = createPhysicsBody(world, proxy, PhysicsMotionType.STATIC);
            setPhysicsBodyShape(world, body, shape);
            if (instanceId) {
                for (const body of manifestColliderBodiesByInstanceId.get(instanceId) ?? []) removeBodyAndShape(body);
            }
            made.push(body);
            if (instanceId) manifestColliderBodiesByInstanceId.set(instanceId, [body]);
        }
        if (made.length === 0) throw new Error(`[aquanova] setCollisionShape entity "${entityName}" produced no colliders`);
        behaviorCollisionBodiesByEntityName.set(entityName, made);
        if (fluidSimShape) setEntityFluidSimShape(entityName, fluidSimShape);
    };
    const intersectionTriggers = createIntersectionTriggerRegistry(world, getPhysicsCharacterControllerBody(character), collisionBodiesOfEntity);

    try {
        await behaviorManager.start({
            canvas,
            camera: cam,
            character,
            sounds: soundManager,
            animationGroups: ship.animationGroups ?? [],
            capsuleHeight: CAP_H,
            capsuleRadius: CAP_R,
            eyeHeight: EYE,
            canStand,
            jumpApertureAssist,
            getPicker,
            nodeNameOf: (mesh) => nodeNameOfMesh.get(mesh) ?? mesh.name,
            isCollisionActive: isEntityCollisionActive,
            isLiquefiable: (mesh) => behaviorManager.isLiquefiable(mesh),
            getLiquefiableConfig: (mesh) => behaviorManager.getLiquefiableConfig(mesh),
            isInspecting: inspectOn,
            inspectAt: (x, y) => inspectOverlay?.pickAt(x, y),
            weaponAntiGravityGun,
            weaponLiquefactor,
            weaponPistol,
            playerMaxGrabDistance: () => playerBehavior?.maxGrabDistance ?? 8,
            dynamicMassOf: (mesh) => behaviorManager.getDynamicMass(mesh),
            explosions: {
                explode: (entityName, meshes, options) => {
                    if (!explodeEntity) {
                        pendingExplosions.push({ entityName, meshes, options });
                        return;
                    }
                    explodeEntity(entityName, meshes, options);
                },
            },
            retireEntity: (entityName, meshes) => {
                if (!retirePickedEntity) {
                    throw new Error("[aquanova] entity retirement service is not ready");
                }
                retirePickedEntity(entityName, meshes);
            },
            setCollisionShape: setEntityCollisionShape,
            registerIntersectionTrigger: intersectionTriggers.register,
            requestFusionResume: () => requestFusionResume(),
            resolveFusionResume: (token, mesh) => resolveFusionResume(token, mesh),
            resolveFusionTarget: (mesh, point) => resolveFusionTarget(mesh, point),
            fusionTargetLost: (mesh) => fusionTargetLost(mesh),
            reverseFusion: () => {
                reverseFusion();
            },
            liquefy: (mesh, point, config) => liquefyMesh(mesh, point, config),
        });
    } catch (error) {
        soundManager.dispose();
        throw error;
    }
    playerBehavior = behaviorManager.player;
    if (!playerBehavior) {
        // eslint-disable-next-line no-console
        console.warn("[aquanova] manifest has no player behavior; first-person controls are disabled");
    }
    const dynPrimsSkipped = dynBodies.filter((body) => !dynPrims.has(body));
    if (dynPrimsSkipped.length) {
        // eslint-disable-next-line no-console
        console.warn(
            `[aquanova] ${dynPrimsSkipped.length} dynamic prop(s) have NO fluid collision primitive: ${dynPrimsSkipped
                .map((body) => `${nodeNameOfMesh.get(body.mesh) ?? body.mesh.name}(${body.instanceId ?? "no-id"})`)
                .join(", ")}`
        );
    }
    canvas.dataset.fluidStaticPrims = String(staticPrims.length);
    canvas.dataset.fluidDynPrims = `${dynPrims.size}/${dynBodies.length}`;
    let productionFluidPolygonActive = false;
    let productionFluidPolygonSurfaces = 0;
    let productionFluidPolygonTriangles = 0;

    // Test hook (QA): read the player position + nudge look/movement programmatically.
    (window as unknown as { __aquanova?: unknown }).__aquanova = {
        getPos: (): { x: number; y: number; z: number } => playerBehavior?.getPosition() ?? character.getPosition(),
        setPos: (x: number, y: number, z: number): void => {
            character.setPosition({ x, y, z });
        },
        // Both SNAP (target and current together) rather than easing like the mouse does: a test that
        // aims and then immediately fires must not have the shot land wherever the smoothing had got to.
        setYaw: (y: number): void => {
            playerBehavior?.setYaw(y);
        },
        look: (dx: number, dy: number): void => {
            playerBehavior?.look(dx, dy);
        },
        press: (code: string): void => {
            playerBehavior?.press(code);
        },
        emitEntityEvent: (name: string, event: string): void => {
            behaviorManager.events.emit("entityEvent", { name, event });
        },
        release: (code: string): void => {
            playerBehavior?.release(code);
        },
        isCrouched: (): boolean => playerBehavior?.isCrouched ?? false,
        capsuleHeight,
        toggleNoclip: (): void => playerBehavior?.toggleNoclip(),
        behaviors: (): Array<{ name: string; mesh: string }> => behaviorManager.describeInstances(),
        /** Graphics settings for the control panel and QA. `setMsaa` is idempotent. */
        graphics: (): Record<string, unknown> => ({ ...graphics, msaaActive: msaaOn, smaaActive: smaaOn, taaActive: taaOn }),
        setMsaa: (on: boolean): void => {
            if (on !== msaaOn) toggleMsaa();
        },
        setSmaa: (on: boolean): void => {
            if (on !== smaaOn) toggleSmaa();
        },
        setTaa: (on: boolean): void => {
            if (on !== taaOn) toggleTaa();
        },
        setSpecularAA: (on: boolean): void => {
            if (on !== graphics.specularAA) toggleSpecularAA();
        },
        setLocalCubemapBlending,
        setFullyMetallicRoughnessZero,
        setWeaponLaserDistance: (distance: number | null): void => {
            weaponLaser.setTargetDistance(distance, distance !== null);
        },
        pbrMaterials: (): Array<{ name: string; metallicFactor: number; roughnessFactor: number; specularAA: boolean }> =>
            pbrMaterials().map((material) => ({
                name: material.name ?? "",
                metallicFactor: material.metallicFactor ?? 1,
                roughnessFactor: material.roughnessFactor ?? 1,
                specularAA: material.enableSpecularAA === true,
            })),
        liquefactorModel: (): LiquefactorModel => weaponViewmodel.model,
        setLiquefactorModel,
        /** SSAA render scale (1 = off). Applied immediately; the engine picks up the size next frame. */
        setSsaa: (scale: number): void => {
            graphics.ssaa = scale;
            saveGraphicsSettings(graphics);
            applySsaa(scale);
        },
        /** SMAA edge threshold — lower catches more edges. Live, for tuning and the control panel. */
        setSmaaThreshold: (t: number): void => {
            smaaTask.threshold = t;
            smaaTask.updateUniforms();
        },
        /** SMAA pattern-search length in pixels — longer reconstructs shallower edges. */
        setSmaaSearchSteps: (n: number): void => {
            smaaTask.maxSearchSteps = n;
            smaaTask.updateUniforms();
        },
        /** SMAA blending rule: dominant-axis (canonical) vs all four neighbours. For A/B. */
        setSmaaDominantAxis: (on: boolean): void => {
            smaaTask.dominantAxisBlend = on;
            smaaTask.updateUniforms();
        },
        /** Effective SMAA parameters after clamping — lets QA assert hostile inputs were rejected. */
        smaaParams: (): Record<string, number | boolean> => ({
            threshold: smaaTask.threshold,
            maxSearchSteps: smaaTask.maxSearchSteps,
            diagonalDetection: smaaTask.diagonalDetection,
            minDiagonalRun: smaaTask.minDiagonalRun,
            sourceIsSrgb: smaaTask.sourceIsSrgb,
            dominantAxisBlend: smaaTask.dominantAxisBlend,
        }),
        /** SMAA 45-degree pattern detection on/off. Live, for A/B comparison. */
        setSmaaDiagonal: (on: boolean, minRun?: number): void => {
            smaaTask.diagonalDetection = on;
            if (minRun !== undefined) smaaTask.minDiagonalRun = minRun;
            smaaTask.updateUniforms();
        },
        toggleInspect: (): void => inspectOverlay?.toggle(),
        /** FPS + per-task GPU timing overlay (P). */
        togglePerf: (): void => perfOverlay.toggle(),
        /** Cycle tone mapping. Recompiles PBR pipelines, so it settles a frame or two later. */
        cycleTone: (): void => cycleTone(),
        /** Active tone-mapping algorithm name. */
        toneMapping: (): string => canvas.dataset.tonemap ?? "?",
        toggleColliders: (): void => colliderOverlay?.cycle(),
        toggleLights: (): void => lightOverlay?.toggle(),
        toggleProbeVolumes: (): void => probeOverlay?.toggle(),
        toggleFluidSimulations: (): void => fluidSimulationOverlay?.toggle(),
        toggleNearestLight,
        runtimeLights: () => lights.lights,
        runtimeLitMeshes: (): string[] =>
            allShipMeshes
                .filter((mesh) => !!(mesh.material as { _clusteredLightState?: unknown } | undefined)?._clusteredLightState)
                .map((mesh) => nodeNameOfMesh.get(mesh) ?? mesh.name),
        /** Manifest-authored collision primitives that became static bodies. */
        manifestColliders: (): Array<{ kind: string; c: number[]; r?: number; he?: number[] }> =>
            manifestShapes.map((s) => ({
                kind: s.kind,
                c: [...s.centre],
                ...(s.radius !== undefined ? { r: s.radius } : {}),
                ...(s.halfExtents ? { he: [...s.halfExtents] } : {}),
            })),
        /** Node name → the authored shape used for its rigid body (empty when it has none). */
        shapedNodes: (): Array<{ node: string; kind: string; c: number[] }> => manifestPlacements.map(({ node, shape }) => ({ node, kind: shape.kind, c: [...shape.centre] })),
        /** What each RUNNING simulation was handed: the primitives packed into its shader's buffer. */
        injectedPrims: (): Array<{ sim: string; n: number; kinds: Record<string, number>; moving: number }> =>
            activeSims.map((a) => {
                const kinds: Record<string, number> = {};
                const activePrims = a.collision.prims.filter((primitive) => primitive.active !== false);
                for (const p of activePrims) kinds[p.kind] = (kinds[p.kind] ?? 0) + 1;
                const moving = a.collision.moving.filter(({ slot }) => a.collision.prims[slot]?.active !== false).length;
                return { sim: nodeNameOfMesh.get(a.mesh) ?? a.mesh.name, n: activePrims.length, kinds, moving };
            }),
        /** Instance ids whose prop has melted — their colliders and debug shapes are gone. */
        removedPlacements: (): string[] => [...removedPlacements],
        /** Full primitive data each running sim collides against — the exact values packed for the
         *  shader. Verbose, so kept separate from the `injectedPrims` summary. */
        fluidPrims: (): Array<{ sim: string; prims: FluidPrimitive[] }> => activeSims.map((a) => ({ sim: nodeNameOfMesh.get(a.mesh) ?? a.mesh.name, prims: a.collision.prims })),
        /** Latest delayed count of visible particles inside a world-space AABB. */
        waterInAabb: (min: [number, number, number], max: [number, number, number]): { total: number; inside: number } => ({
            total: Number(canvas.dataset.particleCount ?? 0),
            inside: behaviorManager.fluidSimulations.countParticlesInAabb({ min, max }),
        }),
        roomAt: (): string => roomAt(),
        portalStats: (): { currentChunk: string; chunks: number; exteriorChunks: number; meshes: number; totalMeshes: number } => portalVisibility.stats(),
        portalExteriorChunks: (): readonly string[] => portalVisibility.exteriorChunks(),
        portalStates: (): Array<{ id: string; door?: string; enabled: boolean }> => portalVisibility.portalStates(),
        portalTraversals: (): Array<{ id: string; from: string; to: string; depth: number }> =>
            portalVisibility.traversals().map(({ portalId, fromChunk, toChunk, depth }) => ({ id: portalId, from: fromChunk, to: toChunk, depth })),
        portalOpaqueOrderStats: (): { bindings: number; uniqueMeshes: number; inOrder: boolean } => {
            const ordered = sceneTask._opaqueBindings
                .map((binding) => binding.renderable.mesh)
                .filter((mesh): mesh is Mesh => mesh !== undefined && mesh.visible !== false && portalVisibility.meshOrder(mesh) !== undefined);
            const order = ordered.map((mesh) => portalVisibility.meshOrder(mesh)!);
            return {
                bindings: ordered.length,
                uniqueMeshes: new Set(ordered).size,
                inOrder: order.every((value, index) => index === 0 || order[index - 1]! <= value),
            };
        },
        setPortalEnabled: (id: string, enabled: boolean): boolean => portalVisibility.setPortalEnabled(id, enabled),
        setDoorPortalEnabled: (door: string, enabled: boolean): number => portalVisibility.setDoorEnabled(door, enabled),
        togglePortalFrusta: (): void => portalOverlay?.toggle(),
        camState: (): Record<string, number> => ({
            px: cam.position.x,
            py: cam.position.y,
            pz: cam.position.z,
            tx: cam.target.x,
            ty: cam.target.y,
            tz: cam.target.z,
            fov: (cam as unknown as { fov?: number }).fov ?? -1,
            near: cam.nearPlane,
            far: cam.farPlane,
        }),
        grading: (): Record<string, unknown> => ({
            exposure: scene.imageProcessing.exposure,
            contrast: scene.imageProcessing.contrast,
            toneMapping: scene.imageProcessing.toneMapping,
            toneMappingEnabled: scene.imageProcessing.toneMappingEnabled,
            ibl: environmentIntensity,
        }),
        fire: (): void => {
            void playerBehavior?.fire();
        },
        antiGravityState: (): {
            grabbed: string | null;
            bodies: Array<{ name: string; mass: number; position: [number, number, number]; velocity: [number, number, number] }>;
        } => ({
            grabbed: antiGravityGrabbedBody ? (nodeNameOfMesh.get(antiGravityGrabbedBody.mesh) ?? antiGravityGrabbedBody.mesh.name) : null,
            bodies: dynBodies
                .filter((body) => body.movable && body.body)
                .map((body) => {
                    const velocity = getPhysicsBodyLinearVelocity(world, body.body!);
                    return {
                        name: nodeNameOfMesh.get(body.mesh) ?? body.mesh.name,
                        mass: body.mass,
                        position: [body.proxy.position.x, body.proxy.position.y, body.proxy.position.z],
                        velocity: [velocity.x, velocity.y, velocity.z],
                    };
                }),
        }),
        setFusionPressed: (pressed: boolean, targetName?: string): boolean => {
            if (!pressed) return reverseFusion();
            const token = requestFusionResume();
            if (token === null) return false;
            if (token === 0) return true;
            const target = targetName ? (meshesByNodeName.get(targetName)?.[0] ?? null) : null;
            const result = resolveFusionResume(token, target);
            if (result === "start-new" && target) {
                liquefyMesh(target, null, behaviorManager.getLiquefiableConfig(target));
            }
            return result !== "continue" && result !== "await-target";
        },
        liqTargets: (): Array<{ name: string; c: number[] | null }> =>
            [...behaviorManager.liquefiableMeshes].map((m) => ({
                name: nodeNameOfMesh.get(m) ?? m.name,
                c: worldBoundsOf(m)?.centre ?? null,
            })),
        dynPos: (): Array<{ name: string; pos: number[]; half: number[]; movable: boolean; hasBody: boolean; chunk: string; visible: boolean }> =>
            dynBodies.map((d) => ({
                name: nodeNameOfMesh.get(d.mesh) ?? d.mesh.name,
                pos: [d.proxy.position.x, d.proxy.position.y, d.proxy.position.z],
                half: d.bounds.half,
                movable: d.movable,
                hasBody: d.body !== null,
                chunk: chunkAt(manifest?.chunks ?? [], -d.proxy.position.x, d.proxy.position.z)?.id ?? "—",
                visible: d.mesh.visible !== false,
            })),
        liquefyNamed: (name: string, hit?: [number, number, number]): string | null => {
            // When a hit point is given, pick the nearest match: several props legitimately share a
            // node name (e.g. a stack of identical crates), so name alone is ambiguous.
            let best: Mesh | null = null;
            let bestD = Infinity;
            for (const m of behaviorManager.liquefiableMeshes) {
                if ((nodeNameOfMesh.get(m) ?? m.name) !== name) continue;
                if (!hit) return (liquefyMesh(m, null), name);
                const c = worldBoundsOf(m)?.centre ?? [0, 0, 0];
                const d = (c[0] - hit[0]) ** 2 + (c[1] - hit[1]) ** 2 + (c[2] - hit[2]) ** 2;
                if (d < bestD) [bestD, best] = [d, m];
            }
            if (!best) return null;
            liquefyMesh(best, hit ?? null);
            return name;
        },
        warp: (x: number, y: number, z: number): void => playerBehavior?.warp(x, y, z),
        freeze: (): void => {
            playerBehavior?.freeze();
        },
        meltState: (): Array<{ name: string; phase: string; frontR: number; maxR: number }> =>
            activeSims.flatMap((a) =>
                a.members.map((member) => ({
                    name: nodeNameOfMesh.get(member.mesh) ?? member.mesh.name,
                    phase: a.phase,
                    frontR: member.state ? member.state.frontR : -1,
                    maxR: member.maxR,
                }))
            ),
        /** Read every live sim's particle positions back and report how the water sits relative to the
         *  floor. `below` counting anything but ~0 means the scene SDF has a hole the particles fell
         *  through, since the solver has no floor of its own. (Positions only — `velocityBuffer` is
         *  not COPY_SRC, and a rejected copy would invalidate the whole command buffer.) */
        waterStats: async (): Promise<Array<{ n: number; minY: number; below: number; spread: number }>> => {
            const out: Array<{ n: number; minY: number; below: number; spread: number }> = [];
            for (const a of activeSims) {
                const n = a.sim.count;
                const f = await readFluidSimulationPositions(a.sim);
                let minY = Infinity,
                    below = 0,
                    minX = Infinity,
                    maxX = -Infinity,
                    minZ = Infinity,
                    maxZ = -Infinity;
                for (let i = 0; i < n; i++) {
                    const x = f[i * 4]!,
                        y = f[i * 4 + 1]!,
                        z = f[i * 4 + 2]!;
                    if (y < -1e4) continue; // dormant particle parked off-screen
                    if (y < minY) minY = y;
                    if (y < FLOOR_Y - 1e-3) below++;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (z < minZ) minZ = z;
                    if (z > maxZ) maxZ = z;
                }
                out.push({ n, minY, below, spread: Math.max(maxX - minX, maxZ - minZ) });
            }
            return out;
        },
        /** True when the fluid collides with a bare ground plane instead of the ship. */
        groundOnly: (): boolean => groundOnly,
        setGroundOnly: (on: boolean): void => {
            if (on !== groundOnly) toggleGroundOnly();
        },
        /** What the last liquefaction actually took from its fluidSim file: the physics the solver
         *  was built with and the render/foam blocks pushed through the shared collection layers. */
        fluidSetting: (): Record<string, unknown> => ({
            name: activeSettingName ?? null,
            render: activeRender ?? null,
            foam: activeFoam ?? null,
            impulse: activeImpulse ?? null,
            grid: activeGrid ?? null,
            physics: activePhysics ?? null,
        }),
        fluidSimulationPreparations: (): Array<{ entity: string; setting: string; method: string; dx: number; particleRadius: number }> =>
            [...behaviorFluidSims.values()].flatMap((entry) => {
                const preparation = entry.preparation;
                return preparation
                    ? [
                          {
                              entity: entry.registration.entityName,
                              setting: entry.registration.settingName,
                              method: preparation.method,
                              dx: preparation.dx,
                              particleRadius: preparation.particleRadius,
                          },
                      ]
                    : [];
            }),
        fluidPolygonRendering: (): { active: boolean; surfaces: number; triangles: number } => {
            return {
                active: productionFluidPolygonActive,
                surfaces: productionFluidPolygonSurfaces,
                triangles: productionFluidPolygonTriangles,
            };
        },
        camTo: (px: number, py: number, pz: number, tx: number, ty: number, tz: number): void => {
            cam.position.set(px, py, pz);
            cam.target.set(tx, ty, tz);
        },
    };

    // ── Fluid render path: scene → sceneColorRT, then the fluid surface composites to the swapchain ─
    // (Liquefactor pattern. Single-sample scene target → MSAA is off on this path.) The scene colour
    // target bakes its OWN depth (dFormat) rather than taking an external depth attachment: with
    // transmissive materials + a skybox, the transmission base-task's opaque bundle reads its depth
    // format from the colour RT descriptor, so an externally-supplied depth would desync the bundle
    // from the pipelines and drop every draw. The surface samples this same baked depth for compositing.
    const device = engine._device;
    const sceneColorRT = createRenderTarget({ lbl: "aq-scene-color", format: engine.format, dFormat: "depth24plus", samples: 1, size: engine });
    const sceneTask = createRenderTask({ name: "scene", rt: sceneColorRT, clr: true, clrColor: scene.clearColor }, engine, scene);
    addTask(scene, sceneTask);
    const applyPortalDrawOrder = (task: RenderTask): void => {
        const bindings = task._opaqueBindings;
        if (bindings.length < 2) return;
        const previous = bindings.slice();
        bindings.sort((a, b) => {
            const baseOrder = a.renderable.order - b.renderable.order;
            if (baseOrder !== 0) return baseOrder;
            const aMesh = a.renderable.mesh;
            const bMesh = b.renderable.mesh;
            const aOrder = aMesh ? portalVisibility.meshOrder(aMesh) : undefined;
            const bOrder = bMesh ? portalVisibility.meshOrder(bMesh) : undefined;
            if (aOrder === undefined) return bOrder === undefined ? 0 : 1;
            return bOrder === undefined ? -1 : aOrder - bOrder;
        });
        if (bindings.some((binding, index) => binding !== previous[index])) {
            task._ob.length = 0;
        }
    };

    // The fluid composite's output. It cannot go straight to the swapchain any more: SMAA has to
    // SAMPLE the finished image, and a swapchain texture is renderable but not sampleable. So the
    // composite lands here and one of two presenting tasks copies it to the screen — SMAA when it is
    // on, a plain blit when it is off. Both are always in the graph and gated, so toggling the
    // setting never rebuilds the frame graph.
    const presentRT = createRenderTarget({ lbl: "aq-present", format: engine.format, samples: 1, size: engine });

    // ── Optional 4× MSAA on the scene (ship geometry) pass ───────────────────────────────
    // The ship renders into an OFFSCREEN single-sample target because the fluid surface has to
    // SAMPLE it when compositing, so it gets none of the engine's own MSAA (`createEngine(...,
    // { msaaSamples: 1 })`) — every panel seam, railing and door frame is hard-aliased.
    //
    // Only the SCENE pass is multisampled. The fluid surface is reconstructed in screen space from
    // a depth/thickness buffer rather than rasterised, so MSAA would do nothing for it.
    // Multisampling the one pass that draws hard-edged triangles is where all the benefit is.
    //
    // Wiring: an MSAA colour+depth target is rendered instead of `sceneColorRT` and RESOLVES into
    // it (`rst`), so every downstream consumer (fluid surface, lit-particle colours) keeps reading
    // the same single-sample texture. Depth needs an explicit pass — WebGPU has no hardware depth
    // resolve — so `createDepthResolveTask` copies sample 0 of the MSAA depth into sceneColorRT's
    // own baked depth, which the surface composites against.
    //
    // Task order is [scene-msaa, scene, depth-resolve]: the plain scene task must record LAST of
    // the two so the colour view it bakes into its pass descriptor belongs to the live
    // `sceneColorRT` texture (both tasks build that target — one as `rt`, one as `rst`). Exactly
    // one of the two executes per frame, so the plain task never clears the depth the resolve wrote.
    //
    // Built LAZILY on first enable: the MSAA colour + depth pair is ~4× a normal canvas-sized pair,
    // so a player who leaves it off never pays for it.
    const MSAA_SAMPLES = 4;
    let msaaOn = false;
    let msaaSceneTask: Task | null = null;
    let pendingFrameGraphRebuild = false;
    let frameGraphReady = false;
    const gateExistingTask = (task: Task, active: () => boolean): void => {
        const run = task.execute!.bind(task);
        task.execute = (): number => (active() ? run() : 0);
    };
    gateExistingTask(sceneTask, () => !msaaOn);
    const setMsaa = (on: boolean): void => {
        if (on === msaaOn) return;
        if (on && !msaaSceneTask) {
            const sceneMsaaRT = createRenderTarget({
                lbl: "aq-scene-msaa",
                format: engine.format,
                // Same depth format as sceneColorRT: the resolve copies between them, and the PBR
                // pipelines are keyed on the depth format they were built against.
                dFormat: "depth24plus",
                samples: MSAA_SAMPLES,
                size: engine,
            });
            msaaSceneTask = createRenderTask({ name: "scene-msaa", rt: sceneMsaaRT, rst: sceneColorRT, clr: true, clrColor: scene.clearColor }, engine, scene);
            const depthResolveTask = createDepthResolveTask({ name: "aq-depth-resolve", sourceTexture: sceneMsaaRT, targetTexture: sceneColorRT }, engine, scene);
            gateExistingTask(msaaSceneTask, () => msaaOn);
            gateExistingTask(depthResolveTask, () => msaaOn);
            addTaskBefore(scene, msaaSceneTask, sceneTask);
            addTaskAfter(scene, depthResolveTask, sceneTask);
            // Rebuild so the new tasks record and every canvas-sized target is re-allocated in the
            // right order. This destroys and re-creates textures that other tasks' bind groups and
            // pass descriptors point at, so it must not run from a UI/key event mid-frame — it is
            // deferred to the next frameStart event.
            pendingFrameGraphRebuild = true;
        }
        msaaOn = on;
        // TAA jitters the UBO of the task it points at, so it has to follow the active scene task.
        retargetTaa(on && msaaSceneTask ? (msaaSceneTask as RenderTask) : sceneTask);
    };
    setMsaaEnabled = (on: boolean): void => {
        if (on === msaaOn) return;
        graphics.msaa = on;
        setMsaa(on);
        saveGraphicsSettings(graphics);
        canvas.dataset.msaa = String(on);
    };
    toggleMsaa = (): void => {
        const next = !msaaOn;
        if (next) setTaaEnabled(false); // TAA is an alternative to MSAA, not a companion
        setMsaaEnabled(next);
    };
    // Apply the resolved setting (defaults ← saved ← ?msaa= override). Enabling here rather than at
    // target-creation time keeps a single code path: the first frame does the same deferred graph
    // rebuild a mid-session toggle does.
    setMsaa(graphics.msaa);
    canvas.dataset.msaa = String(graphics.msaa);

    // Where the ship was last drawn, plus that draw's matrices — the source for per-particle LIT
    // colours (see particle-lit-colors.ts). The scene target is written every frame before the fluid
    // composite, so at liquefy time it still holds the solid prop we are about to melt.
    const litScene = (): LitColorScene | null => {
        const rt = sceneColorRT as unknown as { _colorView: GPUTextureView | null; _depthView: GPUTextureView | null };
        // The scene colour/depth textures are allocated lazily on first render. Until then there is
        // no rendered image to promote the albedo against, so skip the lit pass for that shot rather
        // than binding null views (which throws inside createBindGroup).
        if (!rt._colorView || !rt._depthView) return null;
        const aspect = engine.canvas.width / Math.max(1, engine.canvas.height);
        const proj = getProjectionMatrix(cam, aspect);
        return {
            colorView: rt._colorView,
            depthView: rt._depthView,
            view: getViewMatrix(cam),
            viewProj: getViewProjectionMatrix(cam, aspect),
            projZW: proj[14]!,
            projZZ: proj[10]!,
        };
    };

    // Combined render-buffer capacity, resolved from the single shared authority so authoring and
    // production agree exactly (was a 500k-vs-600k mismatch that made a 550k setting authorable but
    // unrunnable here). The intended 600k is preserved when the device's storage buffers allow it.
    const MAX_TOTAL = aquanovaCombinedParticleCapacity(device.limits);
    const allFluidCollection = createFluidSimulationCollection(engine);
    const fastFluidCollection = createFluidSimulationCollection(engine);
    const allFluidStream = createFluidSimulationCollectionParticleStream(allFluidCollection, scene, MAX_TOTAL);
    // Current global surface fallback, kept in step with the most recently applied render setting.
    // Live simulations capture this value in their own immutable flat-colour GPU buffer.
    const waterRgb: [number, number, number] = [0x16 / 255, 0xa3 / 255, 0xc3 / 255];
    const createFlatParticleColorChannel = (label: string, count: number, rgb: readonly [number, number, number]): FluidParticleChannel => {
        const colors = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
            const offset = i * 4;
            colors[offset] = rgb[0];
            colors[offset + 1] = rgb[1];
            colors[offset + 2] = rgb[2];
            colors[offset + 3] = 1;
        }
        return createFluidParticleChannel(engine, { label, capacity: count, components: 4, initialData: colors });
    };
    const createFlatParticleColorBuffer = (label: string, count: number, rgb: readonly [number, number, number]): GPUBuffer => {
        const colors = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) colors.set([rgb[0], rgb[1], rgb[2], 1], i * 4);
        const buffer = device.createBuffer({
            label,
            size: colors.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, colors);
        return buffer;
    };
    const surfaceTask = attachFluidSimulationCollectionRenderLayer(fastFluidCollection, {
        scene,
        camera: cam,
        mode: "surface",
        particleCapacity: MAX_TOTAL,
        depthTarget: sceneColorRT,
        backgroundTarget: sceneColorRT,
        outputTarget: presentRT,
        direction: [-0.4, -0.82, -0.45],
        excludePolygonSurfaces: true,
    });
    const polygonOutputTarget = createRenderTarget({
        lbl: "aq-fluid-polygon-output",
        format: engine.format,
        samples: 1,
        size: engine,
    });
    const polygonTask = attachFluidSimulationCollectionRenderLayer(allFluidCollection, {
        scene,
        camera: cam,
        mode: "polygon",
        depthTarget: sceneColorRT,
        backgroundTarget: sceneColorRT,
        outputTarget: polygonOutputTarget,
        direction: [-0.4, -0.82, -0.45],
    });
    configureFluidSimulationRenderLayer(polygonTask, { enabled: false });
    interface FluidRenderGroup {
        readonly key: string;
        readonly collection: FluidSimulationCollection;
        readonly outputTarget: RenderTarget;
        readonly surfaceLayer: FluidSimulationRenderLayer;
        readonly foamLayer: FluidSimulationRenderLayer;
        readonly foamEnabled: boolean;
        count: number;
        meshColored: boolean;
    }
    // Groups are reference-counted by render-profile key: every simulation resolving to the same
    // profile shares one group, and it is torn down when the last one releases it. Actual disposal
    // (frame-graph task removal + buffer/target destruction) is deferred to the next frame-graph
    // rebuild so it never races the current frame's encoding — see pendingRenderGroupEvictions.
    const pendingRenderGroupEvictions: FluidRenderGroup[] = [];
    const independentRenderGroups = createReferenceCountedPool<string, FluidRenderGroup>((group) => {
        configureFluidSimulationRenderLayer(group.surfaceLayer, { enabled: false });
        pendingRenderGroupEvictions.push(group);
        pendingFrameGraphRebuild = true;
    });
    const releaseIndependentRenderGroup = (group: FluidRenderGroup | null): void => {
        if (group) {
            setFluidSimulationCollectionSources(group.collection, []);
            independentRenderGroups.release(group.key);
        }
    };
    // Monotonic so debug labels stay unique across eviction (a plain group count would repeat).
    let independentRenderGroupSerial = 0;
    const fluidRenderCompositor = createFluidSimulationRenderCompositor(engine, {
        scene,
        baseColorTarget: presentRT,
        baseLayer: surfaceTask,
    });
    const polygonFoamLayer = attachFluidSimulationCollectionRenderLayer(allFluidCollection, {
        scene,
        camera: cam,
        mode: "foam",
        colorTarget: polygonOutputTarget,
        depthTarget: sceneColorRT,
        surfaceLayer: polygonTask,
        polygonSurfacesOnly: true,
        beforeCompositor: fluidRenderCompositor,
    });
    configureFluidSimulationRenderLayer(polygonFoamLayer, { enabled: false, foam: { polygonSurfaceDepth: true } });
    // Scene wiring only — the sun direction the water is lit by, and the current local probe it
    // reflects. Every look
    // parameter (colour, absorption, blur, filter, impostor size…) comes from the active fluidSim
    // file's `render` block instead; see applyRenderSetting.
    applyFluidEnvironment = (environment) => {
        const renderEnvironment = createFluidRenderEnvironment(environment);
        configureFluidSimulationRenderLayer(surfaceTask, { environment: renderEnvironment });
        configureFluidSimulationRenderLayer(polygonTask, { environment: renderEnvironment });
        for (const group of independentRenderGroups.values()) {
            configureFluidSimulationRenderLayer(group.surfaceLayer, { environment: renderEnvironment });
        }
    };
    if (fluidEnvironment) applyFluidEnvironment(fluidEnvironment);
    behaviorManager.fluidSimulations.installParticleCounter({ engine, stream: allFluidStream.stream });
    const fluidElectricityRenderer = createFluidElectricityRenderer(engine, scene, {
        target: presentRT,
        sceneDepth: sceneColorRT,
        camera: cam,
        particleStream: allFluidStream.stream,
        surfaceDepthView: () => fluidSimulationRenderLayerDepthForSceneIntegration(surfaceTask),
    });
    const setImprovedElectricity = (on: boolean): void => {
        graphics.improvedElectricity = on;
        saveGraphicsSettings(graphics);
        fluidElectricityRenderer.setImproved(on);
        canvas.dataset.improvedElectricity = String(on);
    };
    const setAnimateElectricity = (on: boolean): void => {
        graphics.animateElectricity = on;
        saveGraphicsSettings(graphics);
        fluidElectricityRenderer.setAnimationEnabled(on);
        canvas.dataset.animateElectricity = String(on);
    };
    const setElectricityArcDensity = (value: number): void => {
        graphics.electricityArcDensity = Math.max(0.25, Math.min(3, value));
        saveGraphicsSettings(graphics);
        fluidElectricityRenderer.setArcDensity(graphics.electricityArcDensity);
        canvas.dataset.electricityArcDensity = String(graphics.electricityArcDensity);
    };
    let electricityBloomDebug = false;
    const setElectricityBloomDebug = (on: boolean): void => {
        electricityBloomDebug = on;
        if (on && !graphics.improvedElectricity) {
            setImprovedElectricity(true);
        }
        fluidElectricityRenderer.setBloomDebug(on);
        canvas.dataset.electricityBloomDebug = String(on);
    };
    const setElectricityBloomThreshold = (value: number): void => {
        graphics.electricityBloomThreshold = Math.max(0, Math.min(2, value));
        saveGraphicsSettings(graphics);
        fluidElectricityRenderer.setBloom(graphics.electricityBloomThreshold, graphics.electricityBloomStrength, graphics.electricityBloomRadius);
        canvas.dataset.electricityBloomThreshold = String(graphics.electricityBloomThreshold);
    };
    const setElectricityBloomStrength = (value: number): void => {
        graphics.electricityBloomStrength = Math.max(0, Math.min(10, value));
        saveGraphicsSettings(graphics);
        fluidElectricityRenderer.setBloom(graphics.electricityBloomThreshold, graphics.electricityBloomStrength, graphics.electricityBloomRadius);
        canvas.dataset.electricityBloomStrength = String(graphics.electricityBloomStrength);
    };
    const setElectricityBloomRadius = (value: number): void => {
        graphics.electricityBloomRadius = Math.max(1, Math.min(32, value));
        saveGraphicsSettings(graphics);
        fluidElectricityRenderer.setBloom(graphics.electricityBloomThreshold, graphics.electricityBloomStrength, graphics.electricityBloomRadius);
        canvas.dataset.electricityBloomRadius = String(graphics.electricityBloomRadius);
    };
    fluidElectricityRenderer.setImproved(graphics.improvedElectricity);
    fluidElectricityRenderer.setAnimationEnabled(graphics.animateElectricity);
    fluidElectricityRenderer.setArcDensity(graphics.electricityArcDensity);
    fluidElectricityRenderer.setBloom(graphics.electricityBloomThreshold, graphics.electricityBloomStrength, graphics.electricityBloomRadius);
    fluidElectricityRenderer.setBloomDebug(false);
    canvas.dataset.improvedElectricity = String(graphics.improvedElectricity);
    canvas.dataset.animateElectricity = String(graphics.animateElectricity);
    canvas.dataset.electricityArcDensity = String(graphics.electricityArcDensity);
    canvas.dataset.electricityBloomThreshold = String(graphics.electricityBloomThreshold);
    canvas.dataset.electricityBloomStrength = String(graphics.electricityBloomStrength);
    canvas.dataset.electricityBloomRadius = String(graphics.electricityBloomRadius);
    canvas.dataset.electricityBloomDebug = "false";
    addTask(scene, fluidElectricityRenderer);

    // ── Presenting: exactly one pass, chosen by which AA mode is active ─────────────────────────
    // MSAA above only supersamples POLYGON coverage, so it cannot touch the aliasing that dominates
    // this content: the hard straight lines painted into the panel, grate and hazard-stripe
    // textures. SMAA works on the finished image and treats those the same as a silhouette. TAA
    // jitters the camera by a sub-pixel Halton offset each frame and accumulates, so it supersamples
    // EVERYTHING including texture interiors — but core TAA resets to the raw frame whenever the
    // camera moves (disableOnCameraMove), so today it only pays off standing still. Fixing that
    // needs motion-vector reprojection, which core TAA does not have yet.
    //
    // SMAA and TAA are mutually exclusive (see applyExclusivity in settings.ts), so these are
    // PARALLEL alternatives rather than a chain: all three read presentRT and write the swapchain,
    // and gating picks exactly one. A chained design was tried first and cost an extra fullscreen
    // blit on every path, which measurably degraded the image — MSAA scored worse than no AA.
    let smaaOn = false;
    let taaOn = false;
    // TAA must not accumulate while the view is actually changing: with no motion vectors it
    // reprojects a moved camera onto stale history, which is the ghosting you see. Core TAA's own
    // `disableOnCameraMove` cannot do this job here — it keys off the camera's version, and the
    // first-person controller smooths yaw asymptotically (`yaw += (yawTarget - yaw) * k`), so the
    // camera technically changes EVERY frame forever and the flag would fire permanently, leaving
    // TAA resetting every frame (measured: worse than no AA at all). Compare the actual values with
    // an epsilon instead: real walking moves centimetres per frame, the smoothing residual decays to
    // ~1e-9, so 0.1 mm cleanly separates them.
    const TAA_STILL_EPS = 1e-4;
    const TAA_FACTOR = 0.05; // core default: blend weight of the current frame once accumulating
    let taaMoving = true;
    let taaResetPending = false;
    const taaPrevCam = new Float64Array(6);
    const smaaTask = createSmaaPostProcessTask(
        {
            name: "aq-smaa",
            sourceTexture: presentRT,
            targetTexture: engine.scRT,
            // presentRT carries engine.format, which is an sRGB VIEW when the engine was created
            // with srgb: true. Sampling that decodes to linear, so edge detection must be told or its
            // fixed threshold silently means something different (and misses most dark-region edges).
            sourceIsSrgb: graphics.srgb,
        },
        engine,
        scene
    );
    // TAA writes its jitter into the source render task's own scene-uniform block, so it has to
    // point at the task that is actually drawing — which flips when MSAA is toggled (setMsaa
    // re-points it). Otherwise the jitter lands in an unused UBO and TAA accumulates identical
    // frames, doing nothing at all.
    // disableOnCameraMove: false — the character controller re-writes the camera transform every
    // frame (physics runs even when the player is stationary), so the default `true` sees "moved"
    // on every single frame and resets the history forever: TAA then costs three passes and, because
    // the jitter is still applied, hands back a sub-pixel-shifted frame that measures WORSE than no
    // AA at all. With it off, history accumulates and a still view converges; the cost is ghosting
    // while actually moving, which is the trade this demo wants.
    const taaTask = createTaaPostProcessTask(
        { name: "aq-taa", sourceTexture: presentRT, targetTexture: engine.scRT, sourceRenderTask: sceneTask, disableOnCameraMove: false },
        engine,
        scene
    );
    const presentCopyTask = createCopyToTextureTask({ name: "aq-present-copy", sourceTexture: presentRT, targetTexture: engine.scRT }, engine, scene);
    // While moving, TAA is skipped entirely and the plain blit presents the raw frame: no history,
    // no jitter, so no ghosting and no sub-pixel wobble. The scene re-packs a clean (unjittered)
    // matrix on its own whenever the camera changes, so nothing stale is left behind.
    gateExistingTask(smaaTask, () => smaaOn);
    gateExistingTask(taaTask, () => taaOn && !taaMoving);
    gateExistingTask(presentCopyTask, () => !smaaOn && !(taaOn && !taaMoving));
    addTask(scene, smaaTask);
    addTask(scene, taaTask);
    addTask(scene, presentCopyTask);

    // Closes the whole-frame timing envelope and resolves the fluid profiler's queries. Draws
    // nothing, and MUST be the last task added — resolveInto has to be recorded after every timed
    // pass, before the frame's command buffer is submitted.
    if (fluidProfiler) {
        const prof = fluidProfiler;
        addTask(scene, {
            name: "aq-timing-resolve",
            engine,
            scene,
            _passes: [],
            record: (): void => {},
            execute: (): number => {
                if (!fluidProfilerOn) {
                    return 0;
                }
                endFluidSimulationProfilerFrame(prof);
                return 0;
            },
            dispose: (): void => {},
        });
    }
    attachFluidProfiler = (): void => {
        const p = fluidProfilerOn ? fluidProfiler : null;
        configureFluidSimulationRenderLayer(surfaceTask, { profiler: p });
        configureFluidSimulationRenderLayer(polygonTask, { profiler: p });
        configureFluidSimulationRenderLayer(polygonFoamLayer, { profiler: p });
        for (const group of independentRenderGroups.values()) {
            configureFluidSimulationRenderLayer(group.surfaceLayer, { profiler: p });
        }
        setFluidSimulationCollectionProfiler(allFluidCollection, p);
    };
    retargetTaa = (task: RenderTask): void => {
        // `_sourceRenderTask` is internal, but it is the only way to follow the MSAA toggle.
        (taaTask as unknown as { _sourceRenderTask: RenderTask })._sourceRenderTask = task;
    };
    // setMsaa already ran during startup, when retargetTaa was still a no-op, so point TAA at the
    // task that MSAA actually left active.
    retargetTaa(msaaOn && msaaSceneTask ? (msaaSceneTask as RenderTask) : sceneTask);

    smaaOn = graphics.smaa;
    canvas.dataset.smaa = String(smaaOn);
    setSmaaEnabled = (on: boolean): void => {
        if (on === smaaOn) return;
        smaaOn = on;
        graphics.smaa = on;
        saveGraphicsSettings(graphics);
        canvas.dataset.smaa = String(on);
    };
    toggleSmaa = (): void => {
        const next = !smaaOn;
        if (next) setTaaEnabled(false); // exclusive with TAA
        setSmaaEnabled(next);
    };
    taaOn = graphics.taa;
    canvas.dataset.taa = String(taaOn);
    setTaaEnabled = (on: boolean): void => {
        if (on === taaOn) return;
        taaOn = on;
        graphics.taa = on;
        saveGraphicsSettings(graphics);
        canvas.dataset.taa = String(on);
    };
    toggleTaa = (): void => {
        const next = !taaOn;
        if (next) {
            // TAA supersamples by jittering the camera; running it over MSAA/SMAA would accumulate
            // already-filtered frames and the jitter would fight their edge detection.
            setMsaaEnabled(false);
            setSmaaEnabled(false);
        }
        setTaaEnabled(next);
    };
    // SSAA is deliberately NOT exclusive with the others: it is a different axis (how many pixels
    // you render) and composes with any of them, which is exactly what makes it a useful baseline.
    setSsaa = (scale: number): void => {
        if (scale === graphics.ssaa) return;
        graphics.ssaa = scale;
        saveGraphicsSettings(graphics);
        applySsaa(scale);
    };
    let specularAABusy = false;
    setSpecularAAEnabled = async (on: boolean): Promise<void> => {
        if (specularAABusy || on === graphics.specularAA) return;
        specularAABusy = true;
        const previous = graphics.specularAA;
        setPbrSpecularAA(on);
        graphics.specularAA = on;
        saveGraphicsSettings(graphics);
        canvas.dataset.specularAa = String(on);
        try {
            await Promise.all([rebuildScenePbrPipelines(scene), rebuildScenePbrPipelines(weaponLayer.scene)]);
        } catch (err) {
            setPbrSpecularAA(previous);
            graphics.specularAA = previous;
            saveGraphicsSettings(graphics);
            canvas.dataset.specularAa = String(previous);
            console.warn("[aquanova] specular AA switch failed", err);
        } finally {
            specularAABusy = false;
        }
    };
    toggleSpecularAA = (): void => {
        void setSpecularAAEnabled(!graphics.specularAA);
    };

    // The setting that most recently drove the shared surface pass, for the QA hook below.
    let activeRender: FluidRenderSetting | undefined;
    let activeFoam: FluidFoamSetting | undefined;
    let activeFastFoam: PairState["foam"];
    let activeSettingName: string | undefined;
    /** The impulse the last liquefaction actually fired, with `direction` already resolved to a unit
     *  vector (so a (0,0,0) setting shows the shot ray it was replaced with). */
    let activeImpulse: Record<string, unknown> | undefined;
    let activeGrid: Record<string, unknown> | undefined;
    /** The physics record the last liquefaction's solver was actually built with. */
    let activePhysics: Record<string, unknown> | undefined;

    // Push a setting file's `render` block onto the shared fast-path surface pass. Water colour is
    // additionally captured per simulation and aggregated with its particles; the remaining pipeline
    // controls are global, so the most recently enabled fast-path setting wins for them.
    // Previously these were hardcoded (absorption 0.4, size 0.7, refraction 0.06, specular 41), which
    // silently overrode the file — Liquefactor drives the identical surface task from its UI and did
    // honour them, so the same setting file produced visibly different water in the two demos.
    const foamDebugTexture = (value: string): FoamDebugTexture =>
        value === "accum" || value === "foamR" || value === "bubbleG" || value === "sprayB" || value === "blurred" || value === "foamAlpha" || value === "normals" ? value : "off";
    const foamRenderState = (foam: PairState["foam"]) =>
        foam
            ? {
                  softness: foam.softness,
                  density: foam.density,
                  subsurfaceStrength: foam.subsurfaceStrength,
                  subsurfaceColor: foam.subsurfaceColor,
                  sizeScale: foam.size,
                  blurRadius: foam.blurRadius,
                  lightIntensity: foam.lightIntensity,
                  ambient: foam.ambient,
                  aoStrength: foam.aoStrength,
                  normalStrength: foam.normalStrength,
                  debugTexture: foamDebugTexture(foam.debugTexture),
              }
            : {};
    let fastFoamLayer: FluidSimulationRenderLayer | null = null;
    const ensureFastFoamLayer = (): FluidSimulationRenderLayer | null => {
        if (fastFoamLayer || fastFluidCollection.simulations.length === 0) return fastFoamLayer;
        fastFoamLayer = attachFluidSimulationCollectionRenderLayer(fastFluidCollection, {
            scene,
            camera: cam,
            mode: "foam",
            colorTarget: presentRT,
            depthTarget: sceneColorRT,
            surfaceLayer: surfaceTask,
            excludePolygonSurfaces: true,
            beforeCompositor: fluidRenderCompositor,
            profiler: fluidProfilerOn ? fluidProfiler : null,
            foam: foamRenderState(activeFastFoam),
        });
        if (frameGraphReady) pendingFrameGraphRebuild = true;
        return fastFoamLayer;
    };
    const applyRenderSetting = (r: FluidRenderSetting | undefined, foam?: PairState["foam"]): void => {
        activeRender = r;
        activeFastFoam = foam;
        if (fastFoamLayer) {
            configureFluidSimulationRenderLayer(fastFoamLayer, { enabled: foam?.enabled === true, foam: foamRenderState(foam) });
        }
        if (!r) return;
        const rgb = hexToRgb(r.waterColor);
        if (rgb) {
            // Keep the fallback particle fill in step for simulations created from this setting.
            waterRgb[0] = rgb[0];
            waterRgb[1] = rgb[1];
            waterRgb[2] = rgb[2];
        }
        configureFluidSimulationRenderLayer(surfaceTask, { profile: r });
        configureFluidSimulationRenderLayer(polygonTask, { profile: r });
        configureFluidSimulationRenderLayer(polygonFoamLayer, { foam: foamRenderState(foam) });
    };

    const independentRenderGroupFor = (r: FluidRenderSetting | undefined, foam: PairState["foam"], sim: FluidSimulation, meshColored: boolean): FluidRenderGroup | null => {
        if (r?.independentRendering !== true) return null;
        activeRender = r;
        const diagnostics = getFluidSimulationDiagnostics(sim);
        const key = `${fluidRenderProfileKey(r, diagnostics.particleRadius, diagnostics.surfaceSizeScale, meshColored ? "mesh" : "water")}|${JSON.stringify(foam ?? null)}`;
        // Reuse (and add a reference to) the profile's group, or build a new one on first use.
        return independentRenderGroups.acquire(key, () => {
            const serial = independentRenderGroupSerial++;
            const collection = createFluidSimulationCollection(engine);
            const outputTarget = createRenderTarget({
                lbl: `aq-fluid-profile-output:${serial}`,
                format: engine.format,
                samples: 1,
                size: engine,
            });
            setFluidSimulationCollectionSources(collection, [{ simulation: sim }]);
            const surfaceLayer = attachFluidSimulationCollectionRenderLayer(collection, {
                scene,
                camera: cam,
                mode: "surface",
                particleCapacity: MAX_TOTAL,
                depthTarget: sceneColorRT,
                backgroundTarget: sceneColorRT,
                outputTarget,
                direction: [-0.4, -0.82, -0.45],
                profiler: fluidProfilerOn ? fluidProfiler : null,
                profile: r,
                particleColorMode: meshColored ? "mesh" : "water",
                useParticleColor: true,
                excludePolygonSurfaces: true,
                beforeCompositor: fluidRenderCompositor,
            });
            const foamLayer = attachFluidSimulationCollectionRenderLayer(collection, {
                scene,
                camera: cam,
                mode: "foam",
                colorTarget: outputTarget,
                depthTarget: sceneColorRT,
                surfaceLayer,
                excludePolygonSurfaces: true,
                beforeCompositor: fluidRenderCompositor,
                profiler: fluidProfilerOn ? fluidProfiler : null,
                foam: foamRenderState(foam),
            });
            if (fluidEnvironment) {
                configureFluidSimulationRenderLayer(surfaceLayer, { environment: createFluidRenderEnvironment(fluidEnvironment) });
            }
            configureFluidSimulationRenderLayer(surfaceLayer, { enabled: false });
            configureFluidSimulationRenderLayer(foamLayer, { enabled: false });
            const group: FluidRenderGroup = {
                key,
                collection,
                outputTarget,
                surfaceLayer,
                foamLayer,
                foamEnabled: foam?.enabled === true,
                count: 0,
                meshColored,
            };
            if (frameGraphReady) {
                pendingFrameGraphRebuild = true;
            }
            return group;
        });
    };

    // ── Per-particle mesh-colour path (honours a fluidSim setting's useMeshColors) ─────────────────
    // Colour each water particle from the liquefied mesh's base-colour texture: nearest mesh vertex →
    // its UV → textureLoad, gamma-encoded (the sRGB view decodes to linear on load) to match the
    // display-space fluid composite. Ported from the Liquefactor demo's per-particle colour render.
    const COLOR_SAMPLE_WGSL = /* wgsl */ `
struct P { count: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> uvs: array<vec2<f32>>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outCol: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    let dims = vec2<f32>(textureDimensions(tex, 0));
    let w = fract(uvs[i]);                          // repeat-wrap
    let coord = vec2<i32>(clamp(w * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));
    let c = textureLoad(tex, coord, 0);             // sRGB view → linear
    outCol[i] = vec4<f32>(pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}`;
    const colorPipe = device.createComputePipeline({
        label: "aq-color-sample",
        layout: "auto",
        compute: { module: device.createShaderModule({ label: "aq-color-sample", code: COLOR_SAMPLE_WGSL }), entryPoint: "main" },
    });

    // Build a per-particle RGBA colour buffer by texture-sampling each particle's UV. The UVs are
    // computed alongside the particles (in the worker when one is available), so nothing here touches
    // the mesh geometry. Falls back to the base water tint when the mesh lacks UVs or a base-colour
    // texture.
    const buildMeshColorBuffer = (
        count: number,
        particleUvs: Float32Array | null,
        texView: GPUTextureView | undefined,
        fallbackColor: readonly [number, number, number]
    ): GPUBuffer => {
        const buf = createFlatParticleColorBuffer("aq-shot-color", count, fallbackColor);
        if (!particleUvs || particleUvs.length < count * 2 || !texView) return buf;
        const uvBuf = device.createBuffer({ label: "aq-shot-uv", size: count * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(uvBuf, 0, particleUvs, 0, count * 2);
        const cbuf = device.createBuffer({ label: "aq-shot-ccount", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(cbuf, 0, new Uint32Array([count, 0, 0, 0]));
        const enc = device.createCommandEncoder();
        const bg = device.createBindGroup({
            layout: colorPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cbuf } },
                { binding: 1, resource: { buffer: uvBuf } },
                { binding: 2, resource: texView },
                { binding: 3, resource: { buffer: buf } },
            ],
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(colorPipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(count / 64));
        pass.end();
        device.queue.submit([enc.finish()]);
        uvBuf.destroy();
        cbuf.destroy();
        return buf;
    };

    // ── Milestone D: Liquefaction ────────────────────────────────────────────────────────────────
    // Fluid collision is the SHIP'S OWN AUTHORED PRIMITIVES, evaluated analytically — no baked SDF
    // grids anywhere. Each simulation gets the subset of primitives whose bounds meet its domain,
    // packed into a storage buffer; the WGSL is identical for every sim, so they all share one
    // compiled pipeline (the solvers cache on shader source — a per-shot shader would pay a ~450 ms
    // compile every time).
    //
    // A liquefied prop's own primitive is left out of its sim: it has just become the water, so
    // colliding the water against it would trap every particle inside a solid.
    //
    // The ground plane is a BACKSTOP for a domain that reaches past the authored floor slabs. It is a
    // single plane at FLOOR_Y, so it is right for the main deck only — a liquefiable prop on an upper
    // deck would rest on it rather than falling. None are tagged today.
    const sceneSdfUbo = device.createBuffer({ label: "aq-scene-sdf", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sceneSdfUbo, 0, new Float32Array([FLOOR_Y, 0, 0, 0]));
    const SCENE_SDF_STRUCT = "struct SceneSdfParams { ground: vec4<f32>, };";
    const SCENE_SDF_BODY = `${PRIMITIVES_WGSL}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    return min(carvedPrimitivesSdf(pt, dt), pt.y - sceneSdfParams.ground.x);
}`;
    const PISTOL_HOLE_CAPACITY = 8;
    const PISTOL_HOLE_HALF_LENGTH = 0.35 / 3;
    const PISTOL_HOLE_MIN_RADIUS = 0.1;

    /** Buffer + spec for one sim's collision set, with the slots that need refreshing each step. */
    interface CollisionSet {
        buffer: GPUBuffer;
        scratch: Float32Array;
        /** The selected primitives, kept live so the debug overlay can draw exactly what the shader sees. */
        prims: FluidPrimitive[];
        /** Movable props in this set and the slot each occupies — only these are rewritten per step. */
        moving: Array<{ body: DynBody; slot: number }>;
        /** Packed slots owned by a manifest placement, including static primitives. */
        slotsByInstanceId: Map<string, number[]>;
        /** Packed slots owned by a dynamic body, including bodies without an instance id. */
        slotsByBody: Map<DynBody, number[]>;
        /** Packed behavior-specific fluid slots owned directly by an entity. */
        slotsByEntityName: Map<string, number[]>;
        /** Player capsule slot, reserved in every non-ground-only set regardless of its initial domain. */
        playerSlot: number | null;
        /** Preallocated pistol-hole cylinders, absent from ordinary liquefaction simulations. */
        holes: FluidHoleRing | null;
        spec: SceneSdfSpec;
    }
    interface SimulationGridAabb {
        min: [number, number, number];
        max: [number, number, number];
    }
    /** Select, pack and upload the primitives one simulation collides against. */
    const buildCollisionSet = (gridAabb: SimulationGridAabb, exclude: ReadonlySet<string>, holeCapacity = 0): CollisionSet => {
        const hit = (lo: readonly [number, number, number], hi: readonly [number, number, number]): boolean =>
            lo[0] <= gridAabb.max[0] && hi[0] >= gridAabb.min[0] && lo[1] <= gridAabb.max[1] && hi[1] >= gridAabb.min[1] && lo[2] <= gridAabb.max[2] && hi[2] >= gridAabb.min[2];
        const prims: FluidPrimitive[] = [];
        const moving: Array<{ body: DynBody; slot: number }> = [];
        const slotsByInstanceId = new Map<string, number[]>();
        const slotsByBody = new Map<DynBody, number[]>();
        const slotsByEntityName = new Map<string, number[]>();
        let playerSlot: number | null = null;
        const addSlot = <Key>(map: Map<Key, number[]>, key: Key, slot: number): void => {
            const slots = map.get(key);
            if (slots) slots.push(slot);
            else map.set(key, [slot]);
        };
        const addPrimitive = (primitive: FluidPrimitive, instanceId?: string, body?: DynBody, entityName?: string): number => {
            const slot = prims.length;
            prims.push(primitive);
            if (instanceId) addSlot(slotsByInstanceId, instanceId, slot);
            if (body) addSlot(slotsByBody, body, slot);
            if (entityName) addSlot(slotsByEntityName, entityName, slot);
            return slot;
        };
        if (!groundOnly) {
            for (const s of staticPrims) {
                if (
                    (!s.id || (!exclude.has(s.id) && !removedPlacements.has(s.id) && !collisionDisabledPlacements.has(s.id))) &&
                    (!s.entityName ||
                        (!exclude.has(s.entityName) && entityCollisionStates.get(s.entityName) !== false && entityFluidCollisionStates.get(s.entityName) !== false)) &&
                    hit(s.min, s.max)
                ) {
                    addPrimitive(s.prim, s.id, undefined, s.entityName);
                }
            }
            for (const d of dynBodies) {
                const entityName = nodeNameOfMesh.get(d.mesh) ?? d.mesh.name;
                // The melting prop is excluded: it has just BECOME this water, so colliding against it
                // would trap every particle inside a solid.
                if (
                    collisionDisabledDynBodies.has(d) ||
                    entityFluidCollisionStates.get(entityName) === false ||
                    (d.instanceId && (exclude.has(d.instanceId) || removedPlacements.has(d.instanceId) || collisionDisabledPlacements.has(d.instanceId)))
                ) {
                    continue;
                }
                const p = livePrim(d);
                if (!p) continue;
                const bb = primAabb(p);
                if (!hit(bb.min, bb.max)) continue;
                const slot = addPrimitive(p, d.instanceId, d, entityName);
                if (d.movable) moving.push({ body: d, slot });
            }
            playerSlot = addPrimitive(livePlayerPrimitive());
        }
        const holes = holeCapacity > 0 ? createFluidHoleRing(holeCapacity) : null;
        const capacity = Math.max(prims.length + (holes?.holes.length ?? 0), 1);
        const buffer = device.createBuffer({ label: "aq-fluid-prims", size: primBufferBytes(capacity), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const scratch = new Float32Array(PRIM_HEADER + capacity * PRIM_STRIDE);
        packPrimitives(scratch, prims);
        if (holes) packSubtractionPrimitives(scratch, prims.length, holes.holes);
        device.queue.writeBuffer(buffer, 0, scratch);
        return {
            buffer,
            scratch,
            prims,
            moving,
            slotsByInstanceId,
            slotsByBody,
            slotsByEntityName,
            playerSlot,
            holes,
            spec: { struct: SCENE_SDF_STRUCT, sdf: SCENE_SDF_BODY, buffer: sceneSdfUbo, sdfGrid: buffer },
        };
    };
    const addPistolHoleToCollision = (
        collision: CollisionSet,
        point: readonly [number, number, number],
        direction: readonly [number, number, number],
        particleRadius: number,
        targetSolidSlot: number,
        bulletHoleSize: number
    ): void => {
        const holes = collision.holes;
        if (!holes) return;
        const hole = shotHolePrimitive(point, direction, PISTOL_HOLE_HALF_LENGTH, Math.max(PISTOL_HOLE_MIN_RADIUS, particleRadius * 2) * bulletHoleSize);
        const holeSlot = addFluidHole(holes, hole, targetSolidSlot);
        const packedSlot = collision.prims.length + holeSlot;
        packSubtractionPrimitive(collision.scratch, collision.prims.length, holeSlot, holes.holes[holeSlot]!);
        device.queue.writeBuffer(collision.buffer, (PRIM_HEADER + packedSlot * PRIM_STRIDE) * 4, collision.scratch, PRIM_HEADER + packedSlot * PRIM_STRIDE, PRIM_STRIDE);
    };
    const setCollisionSlotsActive = (set: CollisionSet, slots: readonly number[], active: boolean): void => {
        for (const slot of slots) {
            const primitive = set.prims[slot];
            if (!primitive || (primitive.active !== false) === active) continue;
            set.prims[slot] = { ...primitive, active };
            setPackedPrimitiveActive(set.scratch, slot, active);
            const flagOffset = PRIM_HEADER + slot * PRIM_STRIDE + PRIM_ACTIVE_OFFSET;
            device.queue.writeBuffer(set.buffer, flagOffset * 4, new Float32Array([active ? 1 : 0]));
        }
    };
    const refreshPlayerCollision = (set: CollisionSet): void => {
        const slot = set.playerSlot;
        if (slot === null) return;
        const live = livePlayerPrimitive();
        set.prims[slot] = live;
        packPrimitive(set.scratch, slot, live);
        device.queue.writeBuffer(set.buffer, (PRIM_HEADER + slot * PRIM_STRIDE) * 4, set.scratch, PRIM_HEADER + slot * PRIM_STRIDE, PRIM_STRIDE);
    };
    const setPlacementCollisionActive = (instanceId: string, active: boolean): void => {
        if (active) collisionDisabledPlacements.delete(instanceId);
        else collisionDisabledPlacements.add(instanceId);
        const effectiveActive = active && !removedPlacements.has(instanceId);
        forEachFluidCollisionSet(activeSims, behaviorFluidSims.values(), (collision) => {
            setCollisionSlotsActive(collision, collision.slotsByInstanceId.get(instanceId) ?? [], effectiveActive);
        });
    };
    const setDynBodyCollisionActive = (body: DynBody, active: boolean): void => {
        if (body.instanceId) {
            setPlacementCollisionActive(body.instanceId, active);
            return;
        }
        if (active) collisionDisabledDynBodies.delete(body);
        else collisionDisabledDynBodies.add(body);
        forEachFluidCollisionSet(activeSims, behaviorFluidSims.values(), (collision) => {
            setCollisionSlotsActive(collision, collision.slotsByBody.get(body) ?? [], active);
        });
    };
    const retireDynBodyCollision = (body: DynBody): void => {
        const instanceId = body.instanceId;
        if (instanceId) {
            removedPlacements.add(instanceId);
            forEachFluidCollisionSet(activeSims, behaviorFluidSims.values(), (collision) => {
                setCollisionSlotsActive(collision, collision.slotsByInstanceId.get(instanceId) ?? [], false);
            });
            return;
        }
        collisionDisabledDynBodies.add(body);
        forEachFluidCollisionSet(activeSims, behaviorFluidSims.values(), (collision) => {
            setCollisionSlotsActive(collision, collision.slotsByBody.get(body) ?? [], false);
        });
    };
    const setPhysicsBodyCollisionActive = (body: PhysicsBody | null, active: boolean): void => {
        if (body?._shape) {
            setPhysicsShapeFilterCollideMask(world, body._shape, active ? 0xffffffff : 0);
        }
    };
    applyEntityFluidCollisionState = (entityName, active) => {
        forEachFluidCollisionSet(activeSims, behaviorFluidSims.values(), (collision) => {
            setCollisionSlotsActive(collision, collision.slotsByEntityName.get(entityName) ?? [], active);
        });
    };
    applyEntityCollisionState = (entityName, active) => {
        const bodies = new Set<PhysicsBody>(behaviorCollisionBodiesByEntityName.get(entityName) ?? []);
        const instanceIds = placementIdsByEntityName.get(entityName) ?? [];
        for (const instanceId of instanceIds) {
            setPlacementCollisionActive(instanceId, active);
            for (const body of manifestColliderBodiesByInstanceId.get(instanceId) ?? []) bodies.add(body);
        }
        const dynamicBodies = new Set<DynBody>();
        for (const mesh of meshesByNodeName.get(entityName) ?? []) {
            const body = dynBodyByMesh.get(mesh);
            if (body) dynamicBodies.add(body);
        }
        for (const body of dynamicBodies) {
            setDynBodyCollisionActive(body, active);
            if (body.body) bodies.add(body.body);
        }
        forEachFluidCollisionSet(activeSims, behaviorFluidSims.values(), (collision) => {
            setCollisionSlotsActive(collision, collision.slotsByEntityName.get(entityName) ?? [], active);
        });
        for (const body of bodies) setPhysicsBodyCollisionActive(body, active);
    };
    retirePickedEntity = (entityName, meshes) => {
        behaviorManager.retireEntity(entityName, meshes);
        entityCollisionStates.set(entityName, false);
        entityFluidCollisionStates.set(entityName, false);
        applyEntityCollisionState!(entityName, false);

        const instanceIds = placementIdsByEntityName.get(entityName) ?? [];
        const dynamicBodies = new Set<DynBody>();
        const physicsBodies = new Set<PhysicsBody>(behaviorCollisionBodiesByEntityName.get(entityName) ?? []);
        for (const instanceId of instanceIds) {
            removedPlacements.add(instanceId);
            for (const body of manifestColliderBodiesByInstanceId.get(instanceId) ?? []) {
                physicsBodies.add(body);
            }
            manifestColliderBodiesByInstanceId.delete(instanceId);
        }
        behaviorCollisionBodiesByEntityName.delete(entityName);
        for (const mesh of meshes) {
            const dynamicBody = dynBodyByMesh.get(mesh);
            if (dynamicBody) {
                dynamicBodies.add(dynamicBody);
            }
        }
        for (const dynamicBody of dynamicBodies) {
            if (antiGravityGrabbedBody === dynamicBody) {
                releaseAntiGravityGrab(0);
            }
            retireDynBodyCollision(dynamicBody);
            if (dynamicBody.body) {
                physicsBodies.add(dynamicBody.body);
                dynamicBody.body = null;
            }
        }
        for (const body of physicsBodies) {
            removeBodyAndShape(body);
        }

        const retiredMeshes = new Set(meshes);
        const dynamicallyRemovedMeshes = new Set<Mesh>();
        for (const dynamicBody of dynamicBodies) {
            dynPrims.delete(dynamicBody);
            const index = dynBodies.indexOf(dynamicBody);
            if (index >= 0) {
                dynBodies.splice(index, 1);
            }
            for (const mesh of dynamicBody.meshes) {
                retiredMeshes.add(mesh);
                dynamicallyRemovedMeshes.add(mesh);
                dynBodyByMesh.delete(mesh);
            }
            removeFromScene(scene, dynamicBody.disp);
        }
        for (const mesh of retiredMeshes) {
            behaviorHiddenMeshes.add(mesh);
            movableMeshes.delete(mesh);
            if (!dynamicallyRemovedMeshes.has(mesh)) {
                removeFromScene(scene, mesh);
            }
        }
        canvas.dataset.dynBodies = String(dynBodies.length);
    };
    const explosionRuntime = new AquanovaExplosionRuntime({
        engine,
        scene,
        world,
        fragmentCounts: behaviorManager.instances.flatMap((behavior) => (behavior instanceof ExplodeBehavior ? [behavior.fragmentCount] : [])),
        targets: (): ExplosionTarget[] => {
            const targets = new Map<string, ExplosionTarget>();
            for (const mesh of new Set([...behaviorManager.liquefiableMeshes, ...behaviorManager.movableMeshes])) {
                const entityName = nodeNameOfMesh.get(mesh) ?? mesh.name;
                if (targets.has(entityName)) {
                    continue;
                }
                const meshes = meshesByNodeName.get(entityName) ?? nodePrimitives.get(mesh) ?? [mesh];
                const dynamicBody = meshes.map((candidate) => dynBodyByMesh.get(candidate)).find((candidate) => candidate !== undefined);
                const velocity = dynamicBody?.body ? getPhysicsBodyLinearVelocity(world, dynamicBody.body) : { x: 0, y: 0, z: 0 };
                targets.set(entityName, {
                    entityName,
                    meshes,
                    mass: dynamicBody?.mass ?? 10,
                    linearVelocity: [velocity.x, velocity.y, velocity.z],
                });
            }
            return [...targets.values()];
        },
        retireEntity: (entityName, meshes) => retirePickedEntity!(entityName, meshes),
    });
    explodeEntity = (entityName, meshes, options) => explosionRuntime.explode(entityName, meshes, options);
    for (const pending of pendingExplosions.splice(0)) {
        explodeEntity(pending.entityName, pending.meshes, pending.options);
    }
    let groundOnly = false;
    toggleGroundOnly = (): void => {
        groundOnly = !groundOnly;
        // Rebuild every live sim's primitive set: ground-only simply packs ZERO primitives, so the
        // shader (and its compiled pipeline) is unchanged — only the buffer's count header moves.
        for (const a of activeSims) {
            const set = buildCollisionSet(a.gridAabb, a.group.excluded);
            a.collision.buffer.destroy();
            a.collision = set;
            setFluidSimulationSceneSdf(a.sim, adoptFluidSceneSdf(engine, set.spec));
        }
        canvas.dataset.groundOnly = groundOnly ? "on" : "off";
        // eslint-disable-next-line no-console
        console.log(`[aquanova] fluid collision: ${groundOnly ? "GROUND PLANE ONLY (Liquefactor-equivalent)" : "ship collision primitives + ground"}`);
    };

    const LIFETIME = 5.0; // seconds a settled blob lives before it fades out
    const FADE_DUR = 1.2; // seconds the alpha fade-out takes before dispose + mesh removal
    const IMPULSE_DEFAULT_DIR: readonly [number, number, number] = [0, 1, 0]; // straight up, the old behaviour
    const IMPULSE_DEFAULT_INTENSITY = 1;
    const SPREAD_MARGIN = 1.2; // extra half-width (world units) around the mesh footprint for the sim grid
    const SPREAD_SCALE = 3; // widen the X/Z half-extent so a puddle has room to spread instead of piling against the grid wall
    // Fallback MLS-MPM water used when the manifest lists no fluidSim settings (or a name fails to load).
    // `maxSubDtMs` mirrors the solver default (1/120 s): the largest slice one sub-step may integrate,
    // which decides how many sub-steps a frame needs — never how much time it advances.
    const FLUID_SETTING = {
        gravity: 9.8,
        stiffness: 350,
        viscosity: 0.3,
        restDensity: 3,
        damping: 0.995,
        affineDamping: 0.9,
        groundDamp: 0.85,
        groundDampHeight: 1.5,
        restitution: 0.3,
        substeps: 3,
        maxSubDtMs: 1000 / 120,
    };
    const FALLBACK_PRODUCTION_PRESET: PairState = {
        ...PRODUCTION_PRESET_DEFAULTS,
        schema: FLUID_SETTING,
        count: 1,
    };
    type ProductionFluidSimulationState = Pick<
        FluidSimulationOptions,
        "particleCount" | "bounds" | "particleRadius" | "initialPositions" | "groundY" | "gridResolution" | "flow" | "onPagedGridOverflow"
    >;
    const productionFluidSimulationOptions = (method: "PBF" | "FLIP" | "MLS-MPM" | "PB-MPM", preset: PairState, state: ProductionFluidSimulationState): FluidSimulationOptions => ({
        method,
        ...state,
        physicsScale: preset.physScale ?? 1,
        compatibilityProfile: "aquanova",
        samplingType: preset.demoState?.simulationType === "fluid" ? "fluid" : "mesh",
        semantics: preset.simulationSemantics,
        markersPerCell: preset.markersPerCell ?? FLIP_DEFAULT_MARKERS_PER_CELL,
        physics: preset.schema,
        material: method === "PB-MPM" ? (preset.material ?? 0) : undefined,
        activeBlocks: preset.activeBlocks,
        pagedGrid: preset.pagedGrid,
        pagedGridMaxPages: preset.pagedGridMaxPages,
        fusedBlockDiscovery: preset.fusedBlockDiscovery,
        foam: preset.foam?.enabled ? preset.foam : null,
    });
    // Build the solver a liquefied prop erupts into, honouring the chosen setting's method + params. The
    // settings can be MLS-MPM or PB-MPM. Render (colour/absorption/…) is a SHARED surface pass across all
    // active blobs, so it stays global — only the simulation varies per setting.
    const buildFluidSim = (
        setting: FluidSimSetting | undefined,
        o: { count: number; particleRadius: number; initialPositions: Float32Array; boundsMin: [number, number, number]; boundsMax: [number, number, number]; dx: number }
    ): FluidSimulation => {
        const sim = buildFluidSimRaw(setting, o);
        // Sims are created per shot, so a newly-spawned blob has to pick up the current profiler
        // state or its passes would go untimed while the panel is open.
        setFluidSimulationProfiler(sim, fluidProfilerOn ? fluidProfiler : null);
        return sim;
    };
    const buildFluidSimRaw = (
        setting: FluidSimSetting | undefined,
        o: { count: number; particleRadius: number; initialPositions: Float32Array; boundsMin: [number, number, number]; boundsMax: [number, number, number]; dx: number }
    ): FluidSimulation => {
        const method = setting?.method === "PB-MPM" ? "PB-MPM" : "MLS-MPM";
        const preset = setting?.preset ?? FALLBACK_PRODUCTION_PRESET;
        return createFluidSimulation(
            engine,
            productionFluidSimulationOptions(method, preset, {
                particleCount: o.count,
                bounds: { min: o.boundsMin, max: o.boundsMax },
                particleRadius: o.particleRadius,
                initialPositions: o.initialPositions,
                groundY: FLOOR_Y,
                gridResolution: undefined,
                flow: undefined,
                onPagedGridOverflow: undefined,
            })
        );
    };
    /**
     * Compile every fluid pipeline once, at load, on a throwaway one-particle sim.
     *
     * The first liquefaction otherwise stalls for ~480 ms — measured, and measured to be a ONE-time
     * cost: the second shot (and a shot using a different setting) has no frame over 30 ms. It is not
     * JS (the main thread profiles 93% idle through the stall) and not the WebGPU API calls (12 shader
     * modules + 12 compute pipelines return in under a millisecond); the cost lands in the GPU process
     * when those pipelines are first used. `g2pPipeCache` is per-sim so it cannot help across shots —
     * what saves the second shot is the browser's own cache, keyed on the WGSL source. Compiling that
     * same source here, before the demo reports ready, moves the stall into the load screen.
     *
     * Warmed per METHOD, since MLS-MPM and PB-MPM are different shaders, and with the real scene SDF,
     * because the G2P and update-grid variants are keyed on that WGSL too.
     */
    async function warmUpFluidPipelines(): Promise<void> {
        const methods = new Set<string>();
        for (const s of fluidSettings.values()) {
            if (s?.method === "MLS-MPM" || s?.method === "PB-MPM") {
                methods.add(s.method);
            }
        }
        methods.add("MLS-MPM"); // the fallback FLUID_SETTING, used when a mesh names no setting
        const t0 = performance.now();
        for (const method of methods) {
            const setting = [...fluidSettings.values()].find((s) => s?.method === method);
            const sim = buildFluidSim(setting, {
                count: 1,
                particleRadius: 0.05,
                initialPositions: new Float32Array([0, -1e5, 0]),
                boundsMin: [-1, -1, -1],
                boundsMax: [1, 1, 1],
                dx: 0.18,
            });
            const warmSet = buildCollisionSet({ min: [-1, -1, -1], max: [1, 1, 1] }, new Set());
            setFluidSimulationSceneSdf(sim, adoptFluidSceneSdf(engine, warmSet.spec));
            const enc = device.createCommandEncoder({ label: "aq-fluid-warmup" });
            stepFluidSimulationForSceneIntegration(sim, enc, 1 / 60);
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
            disposeFluidSimulation(sim);
            warmSet.buffer.destroy();
        }
        const behaviorPreparations = [
            ...new Map([...behaviorFluidSims.values()].flatMap((entry) => (entry.preparation ? [[entry.registration.settingName, entry.preparation] as const] : []))).values(),
        ];
        if (behaviorPreparations.length) {
            const warmSet = buildCollisionSet({ min: [-1, -1, -1], max: [1, 1, 1] }, new Set());
            try {
                for (const preparation of behaviorPreparations) {
                    const sim = buildBehaviorFluidWarmupSim(preparation);
                    try {
                        setFluidSimulationSceneSdf(sim, adoptFluidSceneSdf(engine, warmSet.spec));
                        const enc = device.createCommandEncoder({ label: "aq-behavior-fluid-warmup" });
                        stepFluidSimulationForSceneIntegration(sim, enc, 1 / 60);
                        device.queue.submit([enc.finish()]);
                        await device.queue.onSubmittedWorkDone();
                    } finally {
                        disposeFluidSimulation(sim);
                    }
                }
            } finally {
                warmSet.buffer.destroy();
            }
        }
        // eslint-disable-next-line no-console
        console.log(
            `[aquanova] fluid pipelines warmed (${[...methods].join(", ")}; ${behaviorPreparations.length} behavior setting${behaviorPreparations.length === 1 ? "" : "s"}) in ${(
                performance.now() - t0
            ).toFixed(0)} ms`
        );
    }

    const WRIGGLE_AMP = 0.016; // per-frame "pain" jitter amplitude (world units)    // Dissolve-front growth (world units / s). Matched to Liquefactor's SHIP-mode front (its gallery
    // speed of 6 divided by 3): ship modules read as architecture rather than a single prop, so a faster
    // sweep pops instead of melting. Liquefactor loads this same ship.glb to audition liquefaction, so
    // the two must agree or what the audition shows is not what the game does.
    const LIQUEFY_SPEED = 1;
    const DEFAULT_SAMPLE_RADIUS = 0.03; // volume-sampling spacing when a fluidSim setting omits demoParams.particleRadius

    const bumpMat = (mat: Material): void => {
        (mat as unknown as PluginMat)._uboVersion++;
    };
    // Front radius that fully engulfs the mesh AABB from the hit point (+ edge band + margin).
    const computeMaxR = (hit: readonly [number, number, number], bMin: readonly [number, number, number], bMax: readonly [number, number, number]): number => {
        let maxD = 0;
        for (const px of [bMin[0], bMax[0]]) {
            for (const py of [bMin[1], bMax[1]]) {
                for (const pz of [bMin[2], bMax[2]]) {
                    const dx = px - hit[0];
                    const dy = py - hit[1];
                    const dz = pz - hit[2];
                    maxD = Math.max(maxD, Math.sqrt(dx * dx + dy * dy + dz * dz));
                }
            }
        }
        return maxD + LIQUEFY_EDGE + 0.5;
    };

    interface DissolveMember {
        mesh: Mesh;
        disp: SceneNode | null; // dynamic-mesh display root (dispose the whole subtree on fade-out)
        dyn: DynBody | null; // dynamic-body handle: its collision is disabled when the fluid phase begins (null for a static prop)
        state: LiquefyState | null; // material clip state (null if the mesh had no PBR material)
        material: Material | null;
        maxR: number;
        wriggleNode: SceneNode; // node jittered during the dissolve (display root, or the mesh itself)
        wriggleBase: [number, number, number];
        particleOffset: number;
        particleCount: number;
        colorBuffer: GPUBuffer | null;
        behaviorAvailability: MeshBehaviorAvailability;
        dissolved: boolean;
    }
    interface ActiveSim {
        sim: FluidSimulation;
        mesh: Mesh; // primary shot mesh, used only to name this grouped simulation
        members: DissolveMember[];
        phase: "dissolving" | "fluid" | "fading";
        fluidElapsed: number;
        electricityElapsed: number;
        fadeElapsed: number;
        impulseRemaining: number;
        impulseForce: FluidForceField;
        /** False when the setting's intensity is 0: the burst would be exactly zero, so the force pass is
         *  never installed (the engine compiles it lazily on first use — see beginFluidPhase). */
        impulseActive: boolean;
        waterBase: Float32Array; // all members' sampled particle positions (xyzw)
        waterScratch: Float32Array;
        waterFrontDistance: Float32Array; // noisy distance used by both the solid clip and water reveal
        colorChannel: FluidParticleChannel;
        alphaChannel: FluidParticleChannel;
        useMeshColors: boolean;
        renderGroup: FluidRenderGroup | null;
        group: ShotGroup;
        /** This sim's collision primitives and the exact solver-grid AABB they were picked for. */
        collision: CollisionSet;
        gridAabb: SimulationGridAabb;
        electricityDomain: FluidElectricityDomain | null;
    }
    const activeSims: ActiveSim[] = [];

    interface BehaviorFluidSim {
        registration: FluidSimulationRegistration;
        preparation: BehaviorFluidPreparation | null;
        state: FluidSimulationState;
        activated: boolean;
        sim: FluidSimulation | null;
        flow: FluidFlowConfig | null;
        collision: CollisionSet | null;
        gridAabb: SimulationGridAabb | null;
        timeScale: number;
        shutdownElapsed: number;
        opacity: number;
        playerCollisionEnabled: boolean;
        overflowWarned: boolean;
        pagedOverflowWarned: boolean;
        pendingDispose: boolean;
        emissionTargetCount: number | null;
        emissionCompleteRaised: boolean;
        electricityDomain: FluidElectricityDomain | null;
        electricityElapsed: number;
        colorChannel: FluidParticleChannel | null;
        renderGroup: FluidRenderGroup | null;
    }

    interface BehaviorFluidPreparation {
        json: FluidExportJson;
        method: "PBF" | "FLIP" | "MLS-MPM" | "PB-MPM";
        session: FluidPresetSession;
        preset: PairState;
        flow: FluidFlowConfig;
        physics: Readonly<Record<string, number>>;
        dx: number;
        particleRadius: number;
        count: number;
        timeScale: number;
    }
    const behaviorFluidSims = new Map<FluidSimulationRegistration, BehaviorFluidSim>();
    behaviorManager.events.on("hitWithPistol", ({ mesh, point, direction, bulletHoleSize }) => {
        if (!mesh.material?.name?.toLowerCase().includes("glass")) return;
        const entityName = nodeNameOfMesh.get(mesh) ?? mesh.name;
        const instanceId = instanceIdOfMesh(mesh);
        for (const entry of behaviorFluidSims.values()) {
            const collision = entry.collision;
            const sim = entry.sim;
            if (!collision || !sim || !entry.activated || entry.state === "shutdown") continue;
            const slots = [...(collision.slotsByEntityName.get(entityName) ?? []), ...(instanceId ? (collision.slotsByInstanceId.get(instanceId) ?? []) : [])].filter(
                (slot, index, all) => all.indexOf(slot) === index && collision.prims[slot]?.active !== false
            );
            let targetSolidSlot: number | null = null;
            let targetDistance = Number.POSITIVE_INFINITY;
            for (const slot of slots) {
                const primitive = collision.prims[slot]!;
                const distance = Math.abs(primitiveSdf(primitive, point));
                if (distance < targetDistance) {
                    targetDistance = distance;
                    targetSolidSlot = slot;
                }
            }
            if (targetSolidSlot === null) continue;
            addPistolHoleToCollision(collision, point, direction, getFluidSimulationDiagnostics(sim).particleRadius, targetSolidSlot, bulletHoleSize);
        }
    });
    for (const [entityName, active] of entityCollisionStates) applyEntityCollisionState(entityName, active);
    for (const [entityName, active] of entityFluidCollisionStates) applyEntityFluidCollisionState(entityName, active);

    const behaviorFluidGrid = (registration: FluidSimulationRegistration, preset: PairState): { center: [number, number, number]; aabb: SimulationGridAabb } => {
        const grid = preset.grid;
        if (!grid || grid.size.some((value) => !Number.isFinite(value) || value <= 0)) {
            throw new Error(`fluidSimulation "${registration.settingName}" requires a finite, positive gridSize.`);
        }
        const world = registration.anchor.worldMatrix;
        const center: [number, number, number] = [world[12]!, world[13]!, world[14]!];
        const half = grid.size.map((value) => value * 0.5) as [number, number, number];
        return {
            center,
            aabb: {
                min: [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
                max: [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
            },
        };
    };

    const behaviorFluidFlowAt = (flow: FluidFlowConfig, center: readonly [number, number, number]): FluidFlowConfig => {
        return transformFluidFlow(flow, { translation: center });
    };

    const behaviorFluidFlow = (preset: PairState, center: readonly [number, number, number]): FluidFlowConfig =>
        behaviorFluidFlowAt(
            {
                emitters: preset.emitters ?? [],
                sinks: preset.sinks ?? [],
                ...(preset.initialEmittersFillCapacity !== undefined ? { initialEmittersFillCapacity: preset.initialEmittersFillCapacity } : {}),
            },
            center
        );

    const prepareBehaviorFluidSim = (registration: FluidSimulationRegistration): BehaviorFluidPreparation => {
        const json = registration.setting as FluidExportJson;
        const method = json.meta?.method;
        if (method !== "PBF" && method !== "FLIP" && method !== "MLS-MPM" && method !== "PB-MPM") {
            throw new Error(`fluidSimulation "${registration.settingName}" has unsupported method "${String(method)}".`);
        }
        const session = importFluidPresetSession(json, PRODUCTION_PRESET_DEFAULTS);
        const preset = session.state;
        // Reject unsupported solver capabilities before allocating a production simulation.
        const capabilityRejection = productionFluidCapabilityRejection(method, preset.schema);
        if (capabilityRejection) {
            throw new Error(`fluidSimulation "${registration.settingName}" ${capabilityRejection}.`);
        }
        const { center, aabb } = behaviorFluidGrid(registration, preset);
        const flow = behaviorFluidFlow(preset, [0, 0, 0]);
        const scale = preset.physScale ?? 1;
        // Single home for radius / cell-size / particle-volume: the "aquanova" compatibility profile.
        // FLIP uses the authored gridResolution and otherwise falls back to the physics-scale cell
        // size (reproduced inside the resolver); non-FLIP honors an authored mesh radius. The
        // per-particle volume feeds capacity, which floors it to 1e-6 regardless.
        const config = resolveFluidSimulationConfig("aquanova", {
            method,
            physicsScale: scale,
            samplingType: preset.demoState?.simulationType === "fluid" ? "fluid" : "mesh",
            semantics: preset.simulationSemantics,
            physics: preset.schema,
            ...(method !== "FLIP" && preset.demoParams?.particleRadius !== undefined ? { particleRadius: preset.demoParams.particleRadius } : {}),
            bounds: { min: aabb.min, max: aabb.max },
            ...(preset.gridResolution && preset.gridResolution > 0 ? { gridResolution: preset.gridResolution } : {}),
            markersPerCell: preset.markersPerCell ?? FLIP_DEFAULT_MARKERS_PER_CELL,
        });
        const dx = config.cellSize;
        const particleRadius = config.particleRadius;
        const count = fluidSimulationParticleCapacity(method, preset.count ?? 0, flow, config.particleVolume);
        if (count > MAX_TOTAL) {
            throw new RangeError(
                `fluidSimulation "${registration.settingName}" needs ${count.toLocaleString()} particles, exceeding Aquanova's ${MAX_TOTAL.toLocaleString()} shared-fluid capacity.`
            );
        }
        const physics = config.physics;
        const initialState = planFluidInitialState({
            particleCapacity: count,
            flow: behaviorFluidFlowAt(flow, center),
            particleVolume: config.particleVolume,
            bounds: aabb,
            deriveInitialCount: method === "FLIP",
        });
        const allocation = resolveFluidReconfigurationPlan({
            allocation: {
                method,
                particleCount: count,
                gridDim: config.gridDim!,
                ...(method === "FLIP"
                    ? {
                          flipWarmup: {
                              initialLiveCount: initialState.activeCount,
                              initialTargetCount: initialState.activeCount,
                          },
                      }
                    : {}),
                pressureSolver: (physics.pressureSolver ?? 0) >= 0.5 ? "multigrid" : "jacobi",
                pagedGrid: preset.pagedGrid,
                pagedGridMaxPages: preset.pagedGridMaxPages,
                activeBlocks: preset.activeBlocks,
                quality: {
                    pagedGrid: preset.pagedGrid,
                    pagedGridMaxPages: preset.pagedGridMaxPages,
                    pressureDiagnostics: (physics.pressureDiagnostics ?? 0) >= 0.5 || (physics.pressureTolerance ?? 0) > 0,
                    liquidSdf: (physics.liquidSdf ?? 0) >= 0.5,
                    fractionalSolids: (physics.fractionalSolids ?? 0) >= 0.5,
                    reseedParticles: (physics.reseedParticles ?? 0) >= 0.5,
                    particleSheeting: (physics.particleSheeting ?? 0) >= 0.5,
                    polygonSurface: (physics.polygonSurface ?? 0) >= 0.5,
                    polygonReconstructionMultiplier: physics.polygonReconstructionMultiplier ?? 1,
                },
                foam: preset.foam
                    ? {
                          enabled: preset.foam.enabled,
                          activeParticles: preset.foam.activeParticles,
                          poolScale: preset.foam.poolScale,
                      }
                    : undefined,
                limits: engine._device.limits,
            },
            particleVolume: config.particleVolume,
            flow: behaviorFluidFlowAt(flow, center),
            bounds: aabb,
            deriveInitialCount: method === "FLIP",
            maximumParticleCount: MAX_TOTAL,
        });
        if (allocation.allocation.errors.length > 0) {
            throw new RangeError(`fluidSimulation "${registration.settingName}" cannot allocate: ${allocation.allocation.errors.join(" ")}`);
        }
        return {
            json,
            method,
            session,
            preset,
            flow,
            physics: preset.schema,
            dx,
            particleRadius,
            count,
            timeScale: Math.min(100, Math.max(0.01, preset.simulationTimeScale ?? 1)),
        };
    };

    const buildBehaviorFluidWarmupSim = (preparation: BehaviorFluidPreparation): FluidSimulation => {
        return createFluidSimulation(
            engine,
            productionFluidSimulationOptions(preparation.method, preparation.preset, {
                particleCount: 1,
                bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
                particleRadius: preparation.method === "FLIP" ? undefined : 0.05,
                groundY: -1,
                gridResolution: preparation.method === "FLIP" ? 4 : undefined,
                flow: structuredClone(preparation.flow),
                onPagedGridOverflow: undefined,
            })
        );
    };

    const buildBehaviorFluidSim = (entry: BehaviorFluidSim): void => {
        if (entry.sim) return;
        const preparation = (entry.preparation ??= prepareBehaviorFluidSim(entry.registration));
        const { method, preset, particleRadius, count } = preparation;
        const { center, aabb } = behaviorFluidGrid(entry.registration, preset);
        const flow = behaviorFluidFlowAt(preparation.flow, center);
        const sim = createFluidSimulation(
            engine,
            productionFluidSimulationOptions(method, preset, {
                particleCount: count,
                bounds: { min: aabb.min, max: aabb.max },
                particleRadius,
                groundY: aabb.min[1],
                gridResolution: preset.gridResolution && preset.gridResolution > 0 ? preset.gridResolution : undefined,
                flow,
                onPagedGridOverflow:
                    method === "FLIP"
                        ? (requiredPages, capacity) => {
                              if (entry.pagedOverflowWarned) return;
                              entry.pagedOverflowWarned = true;
                              console.error(
                                  `[aquanova] fluidSimulation "${entry.registration.entityName}" FLIP page pool overflow: ` +
                                      `${requiredPages.toLocaleString()} pages required, ${capacity.toLocaleString()} allocated.`
                              );
                          }
                        : undefined,
            })
        );
        let collision: CollisionSet | null = null;
        let colorChannel: FluidParticleChannel | null = null;
        try {
            const diagnostics = getFluidSimulationDiagnostics(sim);
            entry.emissionTargetCount = fluidEmissionCompletionTarget(
                diagnostics.count,
                diagnostics.activeCount,
                flow.emitters.some((emitter) => emitter.behavior === "inflow")
            );
            entry.emissionCompleteRaised = false;
            collision = buildCollisionSet(aabb, new Set(), PISTOL_HOLE_CAPACITY);
            const authoredColor = hexToRgb((entry.registration.setting as FluidExportJson).render?.waterColor) ?? waterRgb;
            colorChannel = createFlatParticleColorChannel(`aq-behavior-fluid-color:${entry.registration.entityName}`, count, authoredColor);
            if (collision.playerSlot !== null && !entry.playerCollisionEnabled) {
                setCollisionSlotsActive(collision, [collision.playerSlot], false);
            }
            setFluidSimulationSceneSdf(sim, adoptFluidSceneSdf(engine, collision.spec));
            setFluidSimulationProfiler(sim, fluidProfilerOn ? fluidProfiler : null);
            entry.sim = sim;
            entry.flow = flow;
            entry.collision = collision;
            entry.gridAabb = aabb;
            entry.colorChannel = colorChannel;
            entry.electricityDomain ??= entry.registration.electrifiable
                ? behaviorManager.fluidSimulations.createElectricityDomain(`fluidSimulation:${entry.registration.entityName}`)
                : null;
            entry.timeScale = preparation.timeScale;
            entry.shutdownElapsed = 0;
            entry.electricityElapsed = 0;
            entry.opacity = 1;
        } catch (error) {
            disposeFluidSimulation(sim);
            collision?.buffer.destroy();
            if (colorChannel) disposeFluidParticleChannel(colorChannel);
            throw error;
        }
        // eslint-disable-next-line no-console
        console.log(
            `[aquanova] fluidSimulation "${entry.registration.entityName}" allocated — ${method}, ${count.toLocaleString()} particles, grid ${(aabb.max[0] - aabb.min[0]).toFixed(
                2
            )}×${(aabb.max[1] - aabb.min[1]).toFixed(2)}×${(aabb.max[2] - aabb.min[2]).toFixed(2)} m`
        );
    };

    const activateBehaviorFluidSim = (entry: BehaviorFluidSim): void => {
        const sim = entry.sim;
        if (!sim) {
            throw new Error(`fluidSimulation "${entry.registration.settingName}" was not allocated.`);
        }
        const render = (entry.registration.setting as FluidExportJson).render;
        // Acquire the render group at most once — activate runs every update tick while the sim is
        // running, and re-acquiring would leak references and defeat eviction.
        if (!entry.renderGroup) {
            entry.renderGroup = independentRenderGroupFor(render, entry.preparation?.preset.foam, sim, false);
        }
        if (!entry.renderGroup) {
            applyRenderSetting(render, entry.preparation?.preset.foam);
        }
        activeSettingName = entry.registration.settingName;
        entry.activated = true;
    };

    const disposeBehaviorFluidSim = (entry: BehaviorFluidSim): void => {
        if (entry.electricityDomain) {
            behaviorManager.fluidSimulations.disposeElectricityDomain(entry.electricityDomain);
        }
        const simulation = entry.sim;
        const colorChannel = entry.colorChannel;
        entry.sim = null;
        entry.colorChannel = null;
        syncFluidCollections();
        if (simulation) disposeFluidSimulation(simulation);
        entry.collision?.buffer.destroy();
        if (colorChannel) disposeFluidParticleChannel(colorChannel);
        releaseIndependentRenderGroup(entry.renderGroup);
        entry.flow = null;
        entry.collision = null;
        entry.gridAabb = null;
        entry.renderGroup = null;
        entry.electricityDomain = null;
        behaviorFluidSims.delete(entry.registration);
    };

    const completeBehaviorFluidShutdown = (entry: BehaviorFluidSim): void => {
        disposeBehaviorFluidSim(entry);
        entry.registration.onShutdownComplete?.();
    };

    behaviorManager.fluidSimulations.installBackend({
        register(registration) {
            const entry: BehaviorFluidSim = {
                registration,
                preparation: null,
                state: "registered",
                activated: false,
                sim: null,
                flow: null,
                collision: null,
                gridAabb: null,
                timeScale: 1,
                shutdownElapsed: 0,
                opacity: 1,
                playerCollisionEnabled: true,
                overflowWarned: false,
                pagedOverflowWarned: false,
                pendingDispose: false,
                emissionTargetCount: null,
                emissionCompleteRaised: false,
                electricityDomain: null,
                electricityElapsed: 0,
                colorChannel: null,
                renderGroup: null,
            };
            behaviorFluidSims.set(registration, entry);
            try {
                entry.preparation = prepareBehaviorFluidSim(registration);
            } catch (error) {
                console.error(`[aquanova] fluidSimulation "${registration.entityName}" could not prepare its CPU descriptor:`, error);
            }
        },
        update(registration, state) {
            const entry = behaviorFluidSims.get(registration);
            if (!entry) return;
            const previous = entry.state;
            entry.state = state;
            if (state === "disposed") {
                disposeBehaviorFluidSim(entry);
                return;
            }
            if (state === "running" || state === "shutdown") {
                try {
                    const wasActivated = entry.activated;
                    buildBehaviorFluidSim(entry);
                    activateBehaviorFluidSim(entry);
                    if (!wasActivated) {
                        registration.onStarted?.();
                    }
                } catch (error) {
                    entry.state = "paused";
                    console.error(`[aquanova] fluidSimulation "${registration.entityName}" could not start:`, error);
                }
            }
            if (state === "shutdown" && previous !== "shutdown") {
                entry.shutdownElapsed = 0;
                entry.opacity = 1;
            }
        },
        updateFlowObject(registration, kind, name, enabled) {
            const entry = behaviorFluidSims.get(registration);
            const flow = entry?.flow ?? entry?.preparation?.flow;
            if (!entry || !flow) return;
            const objects = kind === "emitter" ? flow.emitters : flow.sinks;
            const object = objects.find((candidate) => candidate.name === name);
            if (!object) {
                throw new Error(`fluidSimulation "${registration.settingName}" has no ${kind} named "${name}".`);
            }
            object.enabled = enabled;
            if (entry.sim) setFluidSimulationFlow(entry.sim, flow);
        },
        updatePlayerCollision(registration, enabled) {
            const entry = behaviorFluidSims.get(registration);
            if (!entry) return;
            entry.playerCollisionEnabled = enabled;
            const collision = entry.collision;
            if (collision && collision.playerSlot !== null) {
                if (enabled) refreshPlayerCollision(collision);
                else setCollisionSlotsActive(collision, [collision.playerSlot], false);
            }
        },
        unregister(registration) {
            const entry = behaviorFluidSims.get(registration);
            if (entry) disposeBehaviorFluidSim(entry);
        },
    });
    const fluidSimulationOverlay = !LAB_DEBUG
        ? null
        : createFluidSimulationOverlay({
              engine,
              scene: debugUtility!.scene,
              canvas,
              simulations: (): FluidSimulationDebugSnapshot[] =>
                  [...behaviorFluidSims.values()].map((entry) => {
                      const preset = entry.preparation?.preset ?? importFluidPresetSession(entry.registration.setting as FluidExportJson, PRODUCTION_PRESET_DEFAULTS).state;
                      const configuredGrid = behaviorFluidGrid(entry.registration, preset);
                      const aabb = entry.gridAabb ?? configuredGrid.aabb;
                      const center: [number, number, number] = [(aabb.min[0] + aabb.max[0]) * 0.5, (aabb.min[1] + aabb.max[1]) * 0.5, (aabb.min[2] + aabb.max[2]) * 0.5];
                      const flow = entry.flow ?? behaviorFluidFlow(preset, center);
                      return {
                          entityName: entry.registration.entityName,
                          state: entry.state,
                          center,
                          size: [aabb.max[0] - aabb.min[0], aabb.max[1] - aabb.min[1], aabb.max[2] - aabb.min[2]],
                          emitters: flow.emitters,
                          sinks: flow.sinks,
                      };
                  }),
          });

    interface CollectedSample {
        entry: PendingSample;
        sample: SampleResult;
    }

    // One shot = one GROUP (the melted node's primitives + every linked node's, transitively).
    // Every mesh is sampled independently in the worker pool, then all samples are concatenated into
    // ONE solver. Per-mesh state survives only for visual dissolve, color, Havok teardown and cleanup.
    interface ShotGroup {
        sampling: number;
        dispatchComplete: boolean;
        samples: CollectedSample[];
        sim: ActiveSim | null;
        direction: 1 | -1;
        meshes: ReadonlySet<Mesh>;
        resumeGeneration: number;
        resumePending: number | null;
        primaryMesh: Mesh;
        hit: readonly [number, number, number] | null;
        setting: FluidSimSetting | undefined;
        settingName: string | undefined;
        soundCategory: string;
        electrifiable: boolean;
        /** Animations that were playing when sampling began. Kept paused unless the shot is cancelled. */
        pausedAnimations: AnimationGroup[];
        /** Instance ids this shot's water must NOT collide against (its own props). */
        excluded: ReadonlySet<string>;
        /** Unit direction from the player to the crosshair when the shot was fired. Used for a
         *  setting whose `impulse.direction` is (0,0,0) — "push it the way I shot it". */
        shotDir: [number, number, number];
    }

    // ── Sampling worker pool ─────────────────────────────────────────────────────────────
    // The CPU particle fill is 100–400 ms for a typical prop and scales with the group size (a
    // 4-primitive node samples four times, a linked door twice), so running it inline froze the
    // frame on every shot. It now runs in workers: `requestSample` posts the WORLD-space geometry
    // and `collectSample` gathers the result. Once every linked mesh is ready, the main thread builds
    // one shared simulation and its per-mesh colour buffers.
    type SampleResult = { positions: Float32Array; count: number; radius: number; min: [number, number, number]; max: [number, number, number]; uvs: Float32Array | null };
    interface PendingSample {
        mesh: Mesh;
        dyn: DynBody | null;
        wriggleNode: SceneNode; // jittered from the moment the shot lands, before the seed arrives
        wriggleBase: [number, number, number];
        group: ShotGroup;
        behaviorAvailability: MeshBehaviorAvailability;
    }
    type SampleMsg = ParticleFillWorkerResponse;
    interface PoolWorker {
        worker: Worker;
        pending: number;
    }
    const workerPool: PoolWorker[] = [];
    const pendingSamples = new Map<number, PendingSample>();
    // Shots whose seed is still being sampled. The frame loop pain-shakes these so the hit reads as
    // instant feedback instead of the prop sitting still until the worker replies.
    const samplingShots = new Set<PendingSample>();
    let controlledGroup: ShotGroup | null = null;
    let sampleSeq = 0;

    const setSamplingGroupWriggle = (group: ShotGroup, active: boolean): void => {
        const restoredNodes = new Set<SceneNode>();
        for (const entry of samplingShots) {
            if (entry.group !== group) continue;
            // Keep a paused dynamic body visually pinned to the pose whose geometry was sent to the
            // worker. If the trigger is pressed again, the returned particles and display still line
            // up; cancellation re-enables Havok pose sync once the pending samples have settled.
            if (active && entry.dyn) entry.dyn.wriggling = true;
            if (!active && !restoredNodes.has(entry.wriggleNode)) {
                restoredNodes.add(entry.wriggleNode);
                entry.wriggleNode.position.set(entry.wriggleBase[0], entry.wriggleBase[1], entry.wriggleBase[2]);
            }
        }
    };

    requestFusionResume = (): number | null => {
        const group = controlledGroup;
        if (!group) return null;
        if (group.direction > 0) return 0;
        const token = ++group.resumeGeneration;
        group.resumePending = token;
        return token;
    };

    resolveFusionResume = (token: number, mesh: Mesh | null): "resumed" | "start-new" | "await-target" | "continue" => {
        const group = controlledGroup;
        if (!group || group.resumePending !== token) return "continue";
        if (!mesh || (!group.meshes.has(mesh) && !behaviorManager.isLiquefiable(mesh))) return "await-target";
        group.resumePending = null;
        if (group.meshes.has(mesh)) {
            const resumedActiveLiquefaction = group.direction < 0 && group.sim?.phase === "dissolving";
            group.direction = 1;
            setSamplingGroupWriggle(group, true);
            finalizeShotIfReady(group);
            if (resumedActiveLiquefaction) behaviorManager.events.emit("liquefactionStarted", { meshes: [...group.meshes] });
            return "resumed";
        }
        if (behaviorManager.isLiquefiable(mesh)) {
            releaseControlledGroup(group);
            finalizeShotIfReady(group);
            return "start-new";
        }
        finalizeShotIfReady(group);
        return "continue";
    };

    reverseFusion = (): boolean => {
        const group = controlledGroup;
        if (!group) return false;
        if (group.direction > 0) behaviorManager.events.emit("liquefactionReversed", {});
        group.direction = -1;
        group.resumePending = null;
        group.resumeGeneration++;
        setSamplingGroupWriggle(group, false);
        finalizeShotIfReady(group);
        return true;
    };

    const segmentIntersectsBounds = (origin: readonly [number, number, number], target: readonly [number, number, number], bounds: MeshGroupBounds): boolean => {
        const direction = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]] as const;
        let near = 0;
        let far = 1;
        for (let axis = 0; axis < 3; axis++) {
            const min = bounds.centre[axis]! - bounds.half[axis]! - 0.03;
            const max = bounds.centre[axis]! + bounds.half[axis]! + 0.03;
            const axisOrigin = origin[axis]!;
            const axisDirection = direction[axis]!;
            if (Math.abs(axisDirection) < 1e-8) {
                if (axisOrigin < min || axisOrigin > max) return false;
                continue;
            }
            let axisNear = (min - axisOrigin) / axisDirection;
            let axisFar = (max - axisOrigin) / axisDirection;
            if (axisNear > axisFar) [axisNear, axisFar] = [axisFar, axisNear];
            near = Math.max(near, axisNear);
            far = Math.min(far, axisFar);
            if (near > far) return false;
        }
        return far >= 0 && near <= 1;
    };

    resolveFusionTarget = (mesh: Mesh | null, point: readonly [number, number, number] | null): Mesh | null => {
        const group = controlledGroup;
        if (!group || (mesh && group.meshes.has(mesh)) || !point) return mesh;
        const origin = [cam.position.x, cam.position.y, cam.position.z] as const;
        for (const member of group.meshes) {
            const bounds = worldBoundsOf(member);
            if (bounds && segmentIntersectsBounds(origin, point, bounds)) return group.primaryMesh;
        }
        return mesh;
    };

    fusionTargetLost = (mesh: Mesh | null): boolean => {
        const group = controlledGroup;
        return group !== null && group.direction > 0 && (mesh === null || !group.meshes.has(mesh));
    };

    const releaseControlledGroup = (group: ShotGroup): void => {
        if (controlledGroup === group) controlledGroup = null;
    };

    const resumeCancelledAnimations = (group: ShotGroup): void => {
        resumeAnimations(group.pausedAnimations);
        group.pausedAnimations.length = 0;
    };

    const abortSample = (entry: PendingSample): void => {
        samplingShots.delete(entry);
        entry.group.sampling--;
        entry.wriggleNode.position.set(entry.wriggleBase[0], entry.wriggleBase[1], entry.wriggleBase[2]);
        if (entry.dyn && ![...samplingShots].some((pending) => pending.dyn === entry.dyn)) entry.dyn.wriggling = false;
        // Nothing sampled — hand the mesh back so it stays shootable rather than becoming an inert
        // solid that can never be liquefied.
        behaviorManager.restoreMesh(entry.mesh, entry.behaviorAvailability);
        finalizeShotIfReady(entry.group);
    };

    const failSample = (entry: PendingSample, error: FluidMeshSamplingErrorInfo): void => {
        // eslint-disable-next-line no-console
        console.error("[aquanova] mesh particle sampling failed", error);
        abortSample(entry);
    };

    const reportSampleWarnings = (warnings: readonly FluidMeshSamplingWarning[]): void => {
        for (const warning of warnings) {
            // eslint-disable-next-line no-console
            console.warn("[aquanova] mesh particle sampling warning", warning);
        }
    };

    const onSampleMessage = (ev: MessageEvent<SampleMsg>): void => {
        const { id } = ev.data;
        const entry = pendingSamples.get(id);
        pendingSamples.delete(id);
        if (!entry) return;
        if (ev.data.error) {
            failSample(entry, ev.data.error);
            return;
        }
        const { positions, uvs, count, radius, boundsMin, boundsMax, warnings } = ev.data;
        if (count === 0) {
            failSample(entry, { code: "EMPTY_RESULT", message: "Sampling worker returned an empty success result.", requestedStrategy: "auto" });
            return;
        }
        reportSampleWarnings(warnings);
        collectSample(entry, { positions, count, radius, min: boundsMin, max: boundsMax, uvs });
    };
    try {
        if (typeof Worker !== "undefined") {
            const poolSize = Math.min(3, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
            for (let i = 0; i < poolSize; i++) {
                const worker = new Worker(new URL("../particle-fill-worker.ts", import.meta.url), { type: "module" });
                const pw: PoolWorker = { worker, pending: 0 };
                worker.addEventListener("message", (ev: MessageEvent<SampleMsg>) => {
                    pw.pending = Math.max(0, pw.pending - 1);
                    onSampleMessage(ev);
                });
                worker.addEventListener("error", (e) => {
                    // eslint-disable-next-line no-console
                    console.warn("[aquanova] sample worker error", e.message);
                    const idx = workerPool.indexOf(pw);
                    if (idx >= 0) workerPool.splice(idx, 1); // drop the faulty worker; sync fallback once the pool empties
                });
                workerPool.push(pw);
            }
        }
    } catch {
        workerPool.length = 0;
    }
    // Least-loaded worker, so a multi-primitive node's primitives sample in parallel.
    const pickWorker = (): PoolWorker | null => {
        let best: PoolWorker | null = null;
        for (const pw of workerPool) if (!best || pw.pending < best.pending) best = pw;
        return best;
    };
    canvas.dataset.sampleWorkers = String(workerPool.length);

    const requestSample = (mesh: Mesh, group: ShotGroup): void => {
        const g = getMeshPoseGeometry(mesh);
        if (!g) return;
        const worldPos = g.positions;
        // The setting is chosen ONCE per shot by the caller and shared across the whole group; its
        // demoParams.particleRadius drives the volume-sampling spacing (and therefore the particle
        // count), so the same setting file yields the same count here as in the Liquefactor demo.
        const radius = group.setting?.particleRadius ?? DEFAULT_SAMPLE_RADIUS;
        // Retire the mesh as a target IMMEDIATELY: sampling is async now, so leaving it in the sets
        // would let a second shot (or a linked cascade) start a duplicate sim for the same mesh.
        const behaviorAvailability = behaviorManager.retireMesh(mesh);
        // Start the pain shake NOW, on the click, rather than when the seed arrives — the sample takes
        // 100–400 ms in the worker and the prop would otherwise stand still through it. `wriggling`
        // decouples the display node from its Havok pose so the shake doesn't fight physics; the base
        // is captured here and reused when the grouped sim is finalized, which must not read the
        // already-jittered pose.
        const dyn = dynBodyByMesh.get(mesh) ?? null;
        if (dyn) dyn.wriggling = true;
        const wriggleNode = dyn ? dyn.disp : (mesh as unknown as SceneNode);
        const entry: PendingSample = {
            mesh,
            dyn,
            wriggleNode,
            wriggleBase: [wriggleNode.position.x, wriggleNode.position.y, wriggleNode.position.z],
            group,
            behaviorAvailability,
        };
        group.sampling++;
        samplingShots.add(entry);
        const uvs = g.uvs;
        const pw = pickWorker();
        if (pw) {
            const id = ++sampleSeq;
            pendingSamples.set(id, entry);
            pw.pending++;
            const indices = g.indices;
            const transfer: Transferable[] = [worldPos.buffer, indices.buffer];
            if (uvs) transfer.push(uvs.buffer);
            pw.worker.postMessage({ id, positions: worldPos, indices, uvs, texIndices: null, radius, mode: "dense", surfaceOnly: false, ox: 0, oy: 0, oz: 0 }, transfer);
            return;
        }
        // No worker available — sample synchronously on the main thread (blocks).
        try {
            const sample = sampleFluidMeshParticles({ positions: worldPos, indices: g.indices, uvs, radius, mode: "dense" });
            reportSampleWarnings(sample.warnings);
            collectSample(entry, { positions: sample.positions, count: sample.count, radius: sample.radius, min: sample.bounds.min, max: sample.bounds.max, uvs: sample.uvs });
        } catch (error) {
            failSample(entry, fluidMeshSamplingErrorInfo(error));
        }
    };

    function collectSample(entry: PendingSample, sample: SampleResult): void {
        entry.group.samples.push({ entry, sample });
        entry.group.sampling--;
        finalizeShotIfReady(entry.group);
    }

    function finalizeShotIfReady(group: ShotGroup): void {
        if (!group.dispatchComplete || group.sampling > 0 || group.sim) return;
        if (group.samples.length === 0) {
            resumeCancelledAnimations(group);
            releaseControlledGroup(group);
            return;
        }
        if (group.direction < 0 && group.resumePending !== null) return;
        for (const { entry } of group.samples) samplingShots.delete(entry);
        if (group.direction < 0) {
            const restoredNodes = new Set<SceneNode>();
            for (const { entry } of group.samples) {
                if (!restoredNodes.has(entry.wriggleNode)) {
                    restoredNodes.add(entry.wriggleNode);
                    entry.wriggleNode.position.set(entry.wriggleBase[0], entry.wriggleBase[1], entry.wriggleBase[2]);
                }
                if (entry.dyn) entry.dyn.wriggling = false;
                behaviorManager.restoreMesh(entry.mesh, entry.behaviorAvailability);
            }
            group.samples.length = 0;
            resumeCancelledAnimations(group);
            releaseControlledGroup(group);
            return;
        }
        const setting = group.setting;
        const settingName = group.settingName;
        // A member whose geometry could not be sampled was restored as a solid target. Keep its
        // collider in the shared simulation; only successful members become water and exclude their
        // own authored collision placement.
        group.excluded = excludedIds(group.samples.map(({ entry }) => entry.mesh));
        if (setting?.render?.independentRendering === true) {
            activeRender = setting.render;
        } else {
            applyRenderSetting(setting?.render, setting?.preset.foam);
        }
        activeFoam = setting?.foam;
        activeSettingName = settingName;
        const totalCount = group.samples.reduce((total, collected) => total + collected.sample.count, 0);
        const positions = new Float32Array(totalCount * 3);
        const waterBase = new Float32Array(totalCount * 4);
        const min: [number, number, number] = [Infinity, Infinity, Infinity];
        const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        let particleOffset = 0;
        for (const { sample } of group.samples) {
            positions.set(sample.positions, particleOffset * 3);
            for (let axis = 0; axis < 3; axis++) {
                min[axis] = Math.min(min[axis]!, sample.min[axis]!);
                max[axis] = Math.max(max[axis]!, sample.max[axis]!);
            }
            for (let i = 0; i < sample.count; i++) {
                const dst = (particleOffset + i) * 4;
                waterBase[dst] = sample.positions[i * 3]!;
                waterBase[dst + 1] = sample.positions[i * 3 + 1]!;
                waterBase[dst + 2] = sample.positions[i * 3 + 2]!;
                waterBase[dst + 3] = 1;
            }
            particleOffset += sample.count;
        }
        const cx = (min[0] + max[0]) / 2,
            cy = (min[1] + max[1]) / 2,
            cz = (min[2] + max[2]) / 2;
        // Domain: the file's explicit `grid` when it pins an axis, else the automatic size from the
        // prop's own footprint. The wall is a hard boundary, so this is what decides how far a puddle
        // may spread — and the reason a prop tuned in Liquefactor only matches here when it carries
        // its grid across.
        const g = setting?.grid;
        const particleRadius = group.samples[0]!.sample.radius;
        const dx = fluidSimulationCellSize(setting?.method ?? "MLS-MPM", {
            physicsParticleSize: 1,
            samplingType: "mesh",
            particleRadius,
        });
        const autoHalf = (Math.max(max[0] - min[0], max[2] - min[2]) / 2 + SPREAD_MARGIN) * SPREAD_SCALE;
        const halfX = g?.x && g.x > 0 ? g.x / 2 : autoHalf;
        const halfZ = g?.z && g.z > 0 ? g.z / 2 : autoHalf;
        const floorY = gridFloorY(FLOOR_Y, dx);
        const topY = gridTopY(floorY, g?.y, max[1], dx, Math.max(max[1] + 3, CEIL_Y));
        const gridAabb: SimulationGridAabb = {
            min: [cx - halfX, floorY, cz - halfZ],
            max: [cx + halfX, topY, cz + halfZ],
        };
        activeGrid = { x: halfX * 2, y: topY - floorY, z: halfZ * 2, auto: !g?.x && !g?.y && !g?.z };
        // eslint-disable-next-line no-console
        console.log(
            `[aquanova] liquefy — room: ${roomOfMesh(group.primaryMesh)}, node: ${nodeNameOfMesh.get(group.primaryMesh) ?? group.primaryMesh.name}, meshes: ${group.samples.length}, particles: ${totalCount}, fluidSim: ${settingName ?? "(default)"}, ` +
                `grid: ${(halfX * 2).toFixed(2)}×${(topY - floorY).toFixed(2)}×${(halfZ * 2).toFixed(2)} m ` +
                `(${Math.ceil((halfX * 2) / dx)}×${Math.ceil((topY - floorY) / dx)}×${Math.ceil((halfZ * 2) / dx)} cells @ dx ${dx.toFixed(3)})`
        );
        const sim = buildFluidSim(setting, {
            count: totalCount,
            particleRadius,
            initialPositions: positions,
            boundsMin: gridAabb.min,
            boundsMax: gridAabb.max,
            dx,
        });
        // Select authored primitives against the exact AABB passed to the solver, never the sampled prop AABB.
        const collision = buildCollisionSet(gridAabb, group.excluded);
        setFluidSimulationSceneSdf(sim, adoptFluidSceneSdf(engine, collision.spec));
        // eslint-disable-next-line no-console
        console.log(`[aquanova]   collision: ${collision.scratch[0]} primitive(s), ${collision.moving.length} movable`);
        // Render impostor radius, taken from the sample spacing the setting file asked for
        // (`demoParams.particleRadius`) exactly as Liquefactor does. Hardcoding it (it was 0.05) made
        // the splats thinner than the file specifies, which shortens the path light travels through
        // the water and washes out `render.absorption` — the same file looked far more transparent
        // here than in Liquefactor.
        const imp = setting?.impulse;
        const intensity = imp?.intensity ?? IMPULSE_DEFAULT_INTENSITY;
        const rawDir = imp?.direction ?? IMPULSE_DEFAULT_DIR;

        // The dissolve proper. Every member grows from the shared shot origin; the combined sim is
        // frozen during this phase while each particle segment wriggles with its source mesh.
        const hitPoint = group.hit;
        const hit: [number, number, number] = hitPoint ? [hitPoint[0]!, hitPoint[1]!, hitPoint[2]!] : [cx, cy, cz];
        const maxR = computeMaxR(hit, min, max);

        // The blast is centred on the LIQUEFACTION POINT, not the mesh's bounding-box centre: the water
        // should be thrown away from where the shot landed. `maxR` reaches the farthest corner of the
        // whole linked group's union, so one force field covers every particle in the shared sim.
        // A `render`-style explicit `impulse.radius` overrides it, pinning the reach in world units.
        const blastR = imp?.radius && imp.radius > 0 ? imp.radius : Math.max(maxR, particleRadius * 8, 1);
        const impulseOptions = {
            center: hit,
            radius: blastR,
            intensity,
            direction: rawDir,
            fallbackDirection: group.shotDir,
        };
        const resolvedImpulse = resolveFluidImpulseForce(impulseOptions);
        const impulseForce = createFluidImpulseForce(engine, impulseOptions);
        activeImpulse = {
            intensity,
            direction: [...resolvedImpulse.direction],
            radius: blastR,
            fromShotRay: resolvedImpulse.usedFallbackDirection,
        };
        activePhysics = { method: setting?.method ?? "(fallback)", ...(setting?.method === "MLS-MPM" || setting?.method === "PB-MPM" ? setting.physics : FLUID_SETTING) };

        // Per-member visual and color state. The Havok bodies remain until the shared fluid phase
        // begins, so linked props continue supporting one another throughout the dissolve.
        const members: DissolveMember[] = [];
        const waterFrontDistance = new Float32Array(totalCount);
        const wantColor = !!setting?.useMeshColors;
        const flatColor = hexToRgb(setting?.render?.waterColor) ?? waterRgb;
        const ls = wantColor ? litScene() : null;
        particleOffset = 0;
        for (const { entry, sample } of group.samples) {
            const memberMin = sample.min;
            const memberMax = sample.max;
            const state = liquefyStates.get(entry.mesh) ?? null;
            const material = state ? (entry.mesh.material ?? null) : null;
            if (state) {
                state.hit = hit;
                state.frontR = 0;
                state.enabled = true;
                const noiseAmp = state.noiseAmp ?? 0.35;
                const noiseFreq = state.noiseFreq ?? 1.2;
                for (let i = 0; i < sample.count; i++) {
                    const source = i * 3;
                    waterFrontDistance[particleOffset + i] = liquefyFrontDistance(
                        sample.positions[source]!,
                        sample.positions[source + 1]!,
                        sample.positions[source + 2]!,
                        hit,
                        noiseAmp,
                        noiseFreq
                    );
                }
            } else {
                waterFrontDistance.fill(Infinity, particleOffset, particleOffset + sample.count);
            }
            if (material) bumpMat(material);

            const texView = wantColor ? (entry.mesh.material as unknown as { baseColorTexture?: { view?: GPUTextureView } } | undefined)?.baseColorTexture?.view : undefined;
            let colorBuffer = wantColor ? buildMeshColorBuffer(sample.count, sample.uvs, texView, flatColor) : null;
            if (colorBuffer && ls) {
                const lit = buildLitParticleColors(device, sample.count, sample.positions, colorBuffer, ls);
                colorBuffer.destroy();
                colorBuffer = lit;
            }
            members.push({
                mesh: entry.mesh,
                disp: entry.dyn ? entry.dyn.disp : null,
                dyn: entry.dyn,
                state,
                material,
                maxR: computeMaxR(hit, memberMin, memberMax),
                wriggleNode: entry.wriggleNode,
                wriggleBase: entry.wriggleBase,
                particleOffset,
                particleCount: sample.count,
                colorBuffer,
                behaviorAvailability: entry.behaviorAvailability,
                dissolved: false,
            });
            particleOffset += sample.count;
        }

        const renderGroup = independentRenderGroupFor(setting?.render, setting?.preset.foam, sim, wantColor);
        const colorBuffer = createFlatParticleColorBuffer(`aq-liquefaction-color:${group.primaryMesh.name}`, totalCount, flatColor);
        if (wantColor) {
            const encoder = device.createCommandEncoder({ label: "aq-liquefaction-colors" });
            for (const member of members) {
                if (member.colorBuffer) {
                    encoder.copyBufferToBuffer(member.colorBuffer, 0, colorBuffer, member.particleOffset * 16, member.particleCount * 16);
                }
            }
            device.queue.submit([encoder.finish()]);
        }
        for (const member of members) {
            member.colorBuffer?.destroy();
            member.colorBuffer = null;
        }
        const colorChannel = adoptFluidParticleChannel(engine, colorBuffer, {
            capacity: totalCount,
            components: 4,
            ownsBuffer: true,
            label: `aq-liquefaction-color:${group.primaryMesh.name}`,
        });
        const alphaChannel = createFluidParticleChannel(engine, {
            label: `aq-liquefaction-alpha:${group.primaryMesh.name}`,
            capacity: totalCount,
            components: 1,
            initialData: new Float32Array(totalCount),
        });
        const active: ActiveSim = {
            sim,
            mesh: group.primaryMesh,
            members,
            phase: "dissolving",
            fluidElapsed: 0,
            electricityElapsed: 0,
            fadeElapsed: 0,
            impulseRemaining: 0,
            impulseForce,
            impulseActive: intensity > 0,
            waterBase,
            waterScratch: new Float32Array(totalCount * 4),
            waterFrontDistance,
            colorChannel,
            alphaChannel,
            useMeshColors: wantColor,
            renderGroup,
            group,
            collision,
            gridAabb,
            electricityDomain: group.electrifiable
                ? behaviorManager.fluidSimulations.createElectricityDomain(`liquefaction:${nodeNameOfMesh.get(group.primaryMesh) ?? group.primaryMesh.name}`)
                : null,
        };
        group.sim = active;
        activeSims.push(active);
        syncFluidCollections();
        behaviorManager.events.emit("liquefactionStarted", { meshes: [...group.meshes] });
    }
    liquefyMesh = (mesh: Mesh, hitPoint?: readonly [number, number, number] | null, sourceConfig?: LiquefiableBehaviorConfig): void => {
        if (controlledGroup) return;
        // Melting a node melts it WHOLE: every primitive of that node goes at once (a node with 4
        // primitives is one object to the player). A liquefiable behaviour may also name LINKED
        // nodes that have to melt at the same moment (e.g. a door's two leaves); linked names are
        // matched SHIP-WIDE and the cascade is transitive, following each linked node's own links
        // and their primitives in turn.
        //
        // The group is enumerated FIRST, then dispatched, so the whole shot shares one decision:
        //   • one melt ORIGIN — the weapon hit point, or (for a programmatic fire with no pick
        //     point) the shot mesh's centre. Letting each member start at its own centre made a
        //     door's two leaves burn from two separate spots.
        //   • one fluid SETTING, drawn once from the shot mesh's candidates. Picking per mesh meant
        //     a 4-primitive pod could melt into four different fluids.
        // Enumeration is synchronous; only the sampling itself is async, so the members reach the
        // worker pool together and melt in step.
        const group: Mesh[] = [];
        const seen = new Set<Mesh>();
        const queue: Mesh[] = [mesh];
        while (queue.length) {
            const next = queue.shift()!;
            if (seen.has(next)) continue;
            if (next !== mesh && !behaviorManager.isDissolvable(next)) continue; // already melting / not dissolvable
            seen.add(next);
            group.push(next);
            for (const s of nodePrimitives.get(next) ?? []) queue.push(s);
            for (const name of behaviorManager.getLinkedEntityNames(next)) {
                for (const other of meshesByNodeName.get(name) ?? []) queue.push(other);
            }
        }

        const bounds = worldBoundsOf(mesh);
        const origin: readonly [number, number, number] | null = hitPoint ? [hitPoint[0]!, hitPoint[1]!, hitPoint[2]!] : (bounds?.centre ?? null);
        // Candidates come from the SHOT mesh's behaviour (its own `fluidSim` list when it has one,
        // otherwise the manifest's global list); the winner then applies to every member.
        const ownSettings = sourceConfig?.fluidSim ?? behaviorManager.getLiquefiableConfig(mesh)?.fluidSim;
        const candidates = ownSettings?.length ? ownSettings : fluidSettingNames;
        const settingName = candidates.length ? candidates[Math.floor(Math.random() * candidates.length)]! : undefined;
        const setting = settingName ? fluidSettings.get(settingName) : undefined;
        if (settingName && !setting) {
            // Never degrade quietly: falling back changes both the particle count and the physics.
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] fluidSim "${settingName}" was requested but never loaded — using the default water and sample radius`);
        }

        const shot: ShotGroup = {
            sampling: 0,
            dispatchComplete: false,
            samples: [],
            sim: null,
            direction: 1,
            meshes: new Set(group),
            resumeGeneration: 0,
            resumePending: null,
            primaryMesh: mesh,
            hit: origin,
            setting,
            settingName,
            soundCategory: sourceConfig?.sound ?? behaviorManager.getLiquefiableConfig(mesh)?.sound ?? "quickSplash",
            electrifiable: sourceConfig?.electrifiable ?? behaviorManager.getLiquefiableConfig(mesh)?.electrifiable ?? false,
            pausedAnimations: pauseAnimationsTargetingEntities(ship.animationGroups ?? [], new Set(group.map((member) => nodeNameOfMesh.get(member) ?? member.name))),
            excluded: new Set(),
            shotDir: shotDirection(origin),
        };
        controlledGroup = shot;
        for (const m of group) requestSample(m, shot);
        shot.dispatchComplete = true;
        finalizeShotIfReady(shot);
    };

    /** Direction from the player to the crosshair, captured AT FIRE TIME — the shot ray. Used when a
     *  setting's `impulse.direction` is (0,0,0), i.e. "blow it the way I shot it". Taken here rather
     *  than at eruption because the melt takes over a second, during which the player is free to move.
     *  Falls back to straight up when the shot has no hit point (a programmatic fire) or the player is
     *  standing exactly on it. */
    function shotDirection(origin: readonly [number, number, number] | null): [number, number, number] {
        if (!origin) return [...IMPULSE_DEFAULT_DIR] as [number, number, number];
        const dx = origin[0] - cam.position.x;
        const dy = origin[1] - cam.position.y;
        const dz = origin[2] - cam.position.z;
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        return len > 1e-6 ? [dx / len, dy / len, dz / len] : ([...IMPULSE_DEFAULT_DIR] as [number, number, number]);
    }

    // A shot's own props are kept out of its water's collision set: the water is seeded in the
    // prop's own volume, so leaving it in would eject the seeded particles immediately. Keyed by
    // INSTANCE ID, so this covers static placements too.
    function excludedIds(group: readonly Mesh[]): ReadonlySet<string> {
        const out = new Set<string>();
        for (const m of group) {
            const id = instanceIdOfMesh(m); // the prop that is becoming this water
            if (id) out.add(id);
        }
        return out;
    }

    function cancelDissolve(a: ActiveSim): void {
        const restoredNodes = new Set<SceneNode>();
        for (const member of a.members) {
            if (!restoredNodes.has(member.wriggleNode)) {
                restoredNodes.add(member.wriggleNode);
                member.wriggleNode.position.set(member.wriggleBase[0], member.wriggleBase[1], member.wriggleBase[2]);
            }
            if (member.dyn) member.dyn.wriggling = false;
            if (member.state) {
                member.state.frontR = 0;
                member.state.enabled = false;
            }
            if (member.material) bumpMat(member.material);
            member.colorBuffer?.destroy();
            behaviorManager.restoreMesh(member.mesh, member.behaviorAvailability);
        }
        setFluidSimulationForceField(a.sim, null);
        if (a.electricityDomain) {
            behaviorManager.fluidSimulations.disposeElectricityDomain(a.electricityDomain);
        }
        syncFluidCollections();
        disposeFluidSimulation(a.sim);
        disposeFluidForceField(a.impulseForce);
        a.collision.buffer.destroy();
        disposeFluidParticleChannel(a.colorChannel);
        disposeFluidParticleChannel(a.alphaChannel);
        releaseIndependentRenderGroup(a.renderGroup);
        a.renderGroup = null;
        a.group.sim = null;
        resumeCancelledAnimations(a.group);
        releaseControlledGroup(a.group);
    }

    // Transition from the frozen visual dissolve to the running fluid: hide the solid, disable its
    // collision everywhere, remove its Havok body, then mark the simulation ready to step.
    function beginFluidPhase(a: ActiveSim): void {
        const restoredNodes = new Set<SceneNode>();
        const removedBodies = new Set<DynBody>();
        for (const member of a.members) {
            if (!restoredNodes.has(member.wriggleNode)) {
                restoredNodes.add(member.wriggleNode);
                member.wriggleNode.position.set(member.wriggleBase[0], member.wriggleBase[1], member.wriggleBase[2]);
            }
            if (member.state) {
                member.state.enabled = false;
                member.state.frontR = member.maxR;
            }
            if (member.material) bumpMat(member.material);
            gameplayHiddenMeshes.add(member.mesh);
            setMeshVisible(member.mesh, false);
            member.mesh.pickable = false;
            if (!member.dyn || removedBodies.has(member.dyn) || !dynBodies.includes(member.dyn)) continue;
            removedBodies.add(member.dyn);
            if (antiGravityGrabbedBody === member.dyn) {
                releaseAntiGravityGrab(0);
            }
            // Disable collision BEFORE the first fluid step. Any prop resting on this body drops into
            // the erupting water, and every already-running simulation skips the deactivated slots.
            retireDynBodyCollision(member.dyn);
            if (member.dyn.body) removePhysicsBody(world, member.dyn.body);
            for (const m of member.dyn.meshes) {
                dynBodyByMesh.delete(m);
            }
            const di = dynBodies.indexOf(member.dyn);
            if (di >= 0) dynBodies.splice(di, 1);
        }
        // A zero intensity zeroes both the radial and directional terms, so the pass would evaluate to
        // exactly 0 for every particle. Skip it: the engine builds the external-force compute pass
        // lazily on the first non-null injection, so installing one would compile a shader and add a
        // dispatch per substep for 0.35 s to achieve nothing.
        if (a.impulseActive) {
            setFluidSimulationForceField(a.sim, a.impulseForce);
            a.impulseRemaining = 0.35;
        }
        a.phase = "fluid";
        a.fluidElapsed = 0;
        behaviorManager.events.emit("liquefactionCompleted", { meshes: [...a.group.meshes], sound: a.group.soundCategory });
        a.group.pausedAnimations.length = 0;
        releaseControlledGroup(a.group);
    }

    // A shot erupts as ONE event: nothing may start simulating until every member has been sampled
    // AND has finished melting. Without this a door's first leaf would burst while the second was
    // still dissolving.
    function eruptIfReady(group: ShotGroup): void {
        const active = group.sim;
        if (!active || active.phase !== "dissolving" || active.members.some((member) => !member.dissolved)) return;
        // The sim starts STEPPING now (it is frozen while dissolving), so this is the first moment the
        // scene SDF matters — and the last moment every linked body is still resolvable.
        beginFluidPhase(active);
    }

    function syncFluidCollections(): { count: number; electricityDomains: FluidElectricityFrameDomain[] } {
        const allSources: FluidSimulationRenderSource[] = [];
        const fastSources: FluidSimulationRenderSource[] = [];
        const groupSources = new Map<FluidRenderGroup, FluidSimulationRenderSource[]>();
        const electricityDomains: FluidElectricityFrameDomain[] = [];
        let offset = 0;
        let fastMeshColored = false;
        const append = (
            source: FluidSimulationRenderSource,
            renderGroup: FluidRenderGroup | null,
            electricity: {
                domain: FluidElectricityDomain;
                particleRadius: number;
                gridAabb: SimulationGridAabb;
                elapsedSeconds: number;
            } | null,
            meshColored: boolean,
            overflowLabel?: string
        ): void => {
            const count = source.count ?? getFluidSimulationDiagnostics(source.simulation).renderCount;
            if (offset + count > MAX_TOTAL) {
                if (overflowLabel) console.error(`${overflowLabel}: ${(offset + count).toLocaleString()} combined particles exceed ${MAX_TOTAL.toLocaleString()}.`);
                return;
            }
            allSources.push(source);
            if (renderGroup) {
                const sources = groupSources.get(renderGroup) ?? [];
                sources.push(source);
                groupSources.set(renderGroup, sources);
            } else {
                fastSources.push(source);
                fastMeshColored ||= meshColored;
            }
            if (electricity) {
                electricityDomains.push({
                    domain: electricity.domain,
                    offset,
                    count,
                    particleRadius: electricity.particleRadius,
                    gridAabb: electricity.gridAabb,
                    elapsedSeconds: electricity.elapsedSeconds,
                });
            }
            offset += count;
        };
        for (const active of activeSims) {
            const count = active.sim.count;
            const alpha = new Float32Array(count);
            if (active.phase === "dissolving") {
                for (const member of active.members) {
                    if (!member.state) continue;
                    const end = member.particleOffset + member.particleCount;
                    for (let index = member.particleOffset; index < end; index++) {
                        alpha[index] = active.waterFrontDistance[index]! < member.state.frontR ? 1 : 0;
                    }
                }
            } else {
                alpha.fill(active.phase === "fading" ? Math.max(0, 1 - active.fadeElapsed / FADE_DUR) : 1);
            }
            writeFluidParticleChannel(active.alphaChannel, alpha);
            const diagnostics = getFluidSimulationDiagnostics(active.sim);
            append(
                { simulation: active.sim, count, alpha: active.alphaChannel, color: active.colorChannel },
                active.renderGroup,
                active.electricityDomain
                    ? {
                          domain: active.electricityDomain,
                          particleRadius: diagnostics.particleRadius,
                          gridAabb: active.gridAabb,
                          elapsedSeconds: active.electricityElapsed,
                      }
                    : null,
                active.useMeshColors
            );
        }
        for (const entry of behaviorFluidSims.values()) {
            if (!entry.sim || !entry.activated || !entry.colorChannel) continue;
            const diagnostics = getFluidSimulationDiagnostics(entry.sim);
            append(
                { simulation: entry.sim, count: diagnostics.activeCount, opacity: entry.opacity, color: entry.colorChannel },
                entry.renderGroup,
                entry.electricityDomain && entry.gridAabb
                    ? {
                          domain: entry.electricityDomain,
                          particleRadius: diagnostics.particleRadius,
                          gridAabb: entry.gridAabb,
                          elapsedSeconds: entry.electricityElapsed,
                      }
                    : null,
                false,
                entry.overflowWarned ? undefined : `[aquanova] fluidSimulation "${entry.registration.entityName}" cannot be rendered`
            );
            if (offset + diagnostics.activeCount > MAX_TOTAL) entry.overflowWarned = true;
        }
        setFluidSimulationCollectionSources(allFluidCollection, allSources);
        setFluidSimulationCollectionSources(fastFluidCollection, fastSources);
        const polygonDiagnostics = allSources.map((source) => getFluidSimulationDiagnostics(source.simulation).polygon).filter((entry) => entry !== null);
        const polygonActive = polygonDiagnostics.length > 0;
        productionFluidPolygonActive = polygonActive;
        productionFluidPolygonSurfaces = polygonDiagnostics.length;
        productionFluidPolygonTriangles = polygonDiagnostics.reduce((sum, entry) => sum + (entry.triangleCount ?? 0), 0);
        canvas.dataset.fluidPolygonActive = String(polygonActive);
        configureFluidSimulationRenderLayer(polygonTask, { enabled: polygonActive });
        configureFluidSimulationRenderLayer(polygonFoamLayer, { enabled: polygonActive });
        configureFluidSimulationRenderLayer(surfaceTask, {
            // This layer also forwards sceneColorRT into presentRT while idle. Independent and
            // polygon layers composite over that base, and the presentation tasks always read it.
            enabled: true,
            particleColorMode: fastMeshColored ? "mesh" : "water",
            useParticleColor: fastSources.length > 0,
        });
        const activeFastFoamLayer = fastSources.length > 0 ? ensureFastFoamLayer() : fastFoamLayer;
        if (activeFastFoamLayer) {
            configureFluidSimulationRenderLayer(activeFastFoamLayer, {
                enabled: fastSources.length > 0 && activeFastFoam?.enabled === true,
                foam: foamRenderState(activeFastFoam),
            });
        }
        const activeLayers: FluidSimulationRenderLayer[] = polygonActive ? [polygonTask] : [];
        for (const group of independentRenderGroups.values()) {
            const sources = groupSources.get(group) ?? [];
            group.count = sources.reduce((total, source) => total + (source.count ?? 0), 0);
            setFluidSimulationCollectionSources(group.collection, sources);
            configureFluidSimulationRenderLayer(group.surfaceLayer, {
                enabled: sources.length > 0,
                particleColorMode: group.meshColored ? "mesh" : "water",
                useParticleColor: sources.length > 0,
            });
            configureFluidSimulationRenderLayer(group.foamLayer, { enabled: sources.length > 0 && group.foamEnabled });
            if (sources.length > 0) activeLayers.push(group.surfaceLayer);
        }
        configureFluidSimulationRenderCompositor(fluidRenderCompositor, activeLayers, fastSources.length > 0 ? surfaceTask : null);
        return { count: offset, electricityDomains };
    }

    // Per-frame: advance each shot. DISSOLVING — wriggle the solid + water and grow the clip front
    // (the sim does NOT step; the water stays mesh-shaped and is revealed as the solid clips away; the
    // Havok body stays live so props on top keep their support); at full front → beginFluidPhase
    // (hide the solid, disable fluid collision, remove its Havok collider, fire the impulse).
    // FLUID/FADING — step the sim, drive the impulse + lifetime, then fade out and dispose. Encoded
    // BEFORE the shared surface collection reads simulation positions.
    behaviorManager.events.on("frameStart", ({ deltaMs }) => {
        // A newly-inserted task (the MSAA scene pass + its depth resolve) needs the whole graph
        // re-recorded, which re-allocates canvas-sized targets other tasks' bind groups point at.
        // Do it here, at the very top of the frame, before anything is encoded against them.
        if (pendingFrameGraphRebuild) {
            pendingFrameGraphRebuild = false;
            // Evict render groups whose last simulation released them this frame. Their tasks were
            // already disabled at release time; now that no encoding is in flight it is safe to pull
            // them from the graph and destroy their buffers/targets before the rebuild re-records it.
            if (pendingRenderGroupEvictions.length > 0) {
                for (const group of pendingRenderGroupEvictions) {
                    disposeFluidSimulationCollection(group.collection);
                    disposeRenderTarget(group.outputTarget);
                }
                pendingRenderGroupEvictions.length = 0;
            }
            getFrameGraph(scene).build();
        }
        if (controlPanel?.isVisible()) {
            if (weaponDebugTools) controlPanel.updateWeaponTransform(weaponDebugTools.values());
            controlPanel.updateCameraTransform({
                position: [cam.position.x, cam.position.y, cam.position.z],
                target: [cam.target.x, cam.target.y, cam.target.z],
            });
        }
        if (gamePaused) {
            return;
        }
        syncPoiEnvironment();
        syncFullyMetallicRoughnessOverride();
        // Open the fluid profiler's frame BEFORE anything is encoded: beginFrame resets the query
        // cursor, frameStart stamps the whole-frame envelope that "aq-timing-resolve" closes.
        if (fluidProfilerOn && fluidProfiler) {
            beginFluidSimulationProfilerFrame(fluidProfiler);
        }
        // Has the view actually changed since last frame? Position and target together cover both
        // walking and looking. On the first still frame after moving, burn one frame at factor 1 to
        // throw away the history accumulated at the old viewpoint — otherwise it bleeds in as a
        // slowly-fading ghost for about a second.
        const camMoved =
            Math.abs(cam.position.x - taaPrevCam[0]!) > TAA_STILL_EPS ||
            Math.abs(cam.position.y - taaPrevCam[1]!) > TAA_STILL_EPS ||
            Math.abs(cam.position.z - taaPrevCam[2]!) > TAA_STILL_EPS ||
            Math.abs(cam.target.x - taaPrevCam[3]!) > TAA_STILL_EPS ||
            Math.abs(cam.target.y - taaPrevCam[4]!) > TAA_STILL_EPS ||
            Math.abs(cam.target.z - taaPrevCam[5]!) > TAA_STILL_EPS;
        taaPrevCam[0] = cam.position.x;
        taaPrevCam[1] = cam.position.y;
        taaPrevCam[2] = cam.position.z;
        taaPrevCam[3] = cam.target.x;
        taaPrevCam[4] = cam.target.y;
        taaPrevCam[5] = cam.target.z;
        taaMoving = camMoved;
        if (camMoved) {
            taaResetPending = true;
        } else if (taaResetPending) {
            taaResetPending = false;
            taaTask.factor = 1;
        } else {
            taaTask.factor = TAA_FACTOR;
        }
        canvas.dataset.taaAccum = String(taaOn && !taaMoving);
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        const growDt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        for (const entry of [...behaviorFluidSims.values()]) {
            if (entry.pendingDispose) completeBehaviorFluidShutdown(entry);
        }
        // The player can enter a simulation domain long after it was built, so every set reserves a
        // capsule slot and refreshes it from the character controller immediately before fluid steps.
        for (const a of activeSims) {
            refreshPlayerCollision(a.collision);
        }
        for (const entry of behaviorFluidSims.values()) {
            const collision = entry.collision;
            if (!collision) continue;
            for (const moving of collision.moving) {
                if (collision.prims[moving.slot]?.active === false) continue;
                const live = livePrim(moving.body);
                if (!live) continue;
                collision.prims[moving.slot] = live;
                packPrimitive(collision.scratch, moving.slot, live);
                device.queue.writeBuffer(collision.buffer, (PRIM_HEADER + moving.slot * PRIM_STRIDE) * 4, collision.scratch, PRIM_HEADER + moving.slot * PRIM_STRIDE, PRIM_STRIDE);
            }
            const playerSlot = collision.playerSlot;
            if (playerSlot !== null && entry.playerCollisionEnabled) {
                refreshPlayerCollision(collision);
            }
        }
        // Shots still waiting on their seed: shake the solid so the hit reads instantly. There is no
        // water to shake in lock-step yet — that starts with the ActiveSim below.
        for (const s of samplingShots) {
            if (s.group.direction < 0) continue;
            const ox = (Math.random() * 2 - 1) * WRIGGLE_AMP,
                oy = (Math.random() * 2 - 1) * WRIGGLE_AMP,
                oz = (Math.random() * 2 - 1) * WRIGGLE_AMP;
            s.wriggleNode.position.set(s.wriggleBase[0] + ox, s.wriggleBase[1] + oy, s.wriggleBase[2] + oz);
        }
        canvas.dataset.sampling = String(samplingShots.size);
        if (exteriorReady) {
            portalVisibility.update();
            applyPortalDrawOrder(sceneTask);
            if (msaaSceneTask) applyPortalDrawOrder(msaaSceneTask as RenderTask);
        }
        inspectOverlay?.onFrame();
        colliderOverlay?.onFrame(); // dynamic proxies move; the chunk changes as you walk
        fluidSimulationOverlay?.onFrame();
        lightOverlay?.onFrame();
        probeOverlay?.onFrame();
        portalOverlay?.onFrame();
        perfOverlay.onFrame(deltaMs); // FPS + GPU timing readout (P)
        // The room voxels cover EVERY baked room and are world-anchored, so neither walking around nor
        // crossing a chunk boundary changes them. Only the dynamic bodies actually move.
        for (let k = activeSims.length - 1; k >= 0; k--) {
            const a = activeSims[k]!;
            a.electricityElapsed += dt;
            if (a.phase === "dissolving") {
                // Pain-shake each source node and its segment of the combined particle buffer in
                // lock-step. Several render primitives can share one display root, so they must also
                // share one random offset rather than fighting over the node's final pose.
                const wb = a.waterBase,
                    ws = a.waterScratch;
                const offsets = new Map<SceneNode, [number, number, number]>();
                const rate = LIQUEFY_SPEED * growDt * (a.group.direction > 0 ? 1 : -2);
                for (const member of a.members) {
                    let offset = offsets.get(member.wriggleNode);
                    if (!offset) {
                        offset = [(Math.random() * 2 - 1) * WRIGGLE_AMP, (Math.random() * 2 - 1) * WRIGGLE_AMP, (Math.random() * 2 - 1) * WRIGGLE_AMP];
                        offsets.set(member.wriggleNode, offset);
                        member.wriggleNode.position.set(member.wriggleBase[0] + offset[0], member.wriggleBase[1] + offset[1], member.wriggleBase[2] + offset[2]);
                    }
                    const start = member.particleOffset * 4;
                    const end = (member.particleOffset + member.particleCount) * 4;
                    for (let i = start; i < end; i += 4) {
                        ws[i] = wb[i]! + offset[0];
                        ws[i + 1] = wb[i + 1]! + offset[1];
                        ws[i + 2] = wb[i + 2]! + offset[2];
                        ws[i + 3] = 1;
                    }
                    if (member.state) member.state.frontR = Math.max(0, Math.min(member.state.frontR + rate, member.maxR));
                    if (member.material) bumpMat(member.material);
                    member.dissolved = a.group.direction > 0 && (!member.state || member.state.frontR >= member.maxR);
                }
                writeFluidSimulationPositions(a.sim, ws);
                // Radius zero must restore the exact solid even while the held beam could resume.
                // Keeping the noisy clipping plugin enabled at zero leaves small negative-noise pockets dissolved.
                if (a.group.direction < 0 && a.members.every((member) => !member.state || member.state.frontR <= 0)) {
                    behaviorManager.events.emit("liquefactionCancelled", { meshes: [...a.group.meshes] });
                    activeSims.splice(k, 1);
                    cancelDissolve(a);
                    continue;
                }
                if (a.group.direction > 0) eruptIfReady(a.group);
                continue; // do not step the sim while dissolving
            }
            if (a.phase === "fading") {
                a.fadeElapsed += dt;
                if (a.fadeElapsed >= FADE_DUR) {
                    // Dispose BEFORE stepping so we never free a sim's buffers after encoding its step.
                    setFluidSimulationForceField(a.sim, null);
                    if (a.electricityDomain) {
                        behaviorManager.fluidSimulations.disposeElectricityDomain(a.electricityDomain);
                    }
                    activeSims.splice(k, 1);
                    syncFluidCollections();
                    disposeFluidSimulation(a.sim);
                    disposeFluidForceField(a.impulseForce);
                    a.collision.buffer.destroy();
                    disposeFluidParticleChannel(a.colorChannel);
                    disposeFluidParticleChannel(a.alphaChannel);
                    releaseIndependentRenderGroup(a.renderGroup);
                    a.renderGroup = null;
                    const removedRoots = new Set<SceneNode>();
                    for (const member of a.members) {
                        member.colorBuffer?.destroy();
                        const root = member.disp ?? (member.mesh as unknown as SceneNode);
                        if (!removedRoots.has(root)) {
                            removedRoots.add(root);
                            removeFromScene(scene, root);
                        }
                    }
                    // Last of this shot's water gone → put any SDF bodies it excluded back in the union.
                    continue;
                }
            }
            stepFluidSimulation(a.sim, dt);
            if (a.impulseRemaining > 0) {
                a.impulseRemaining = Math.max(0, a.impulseRemaining - dt);
                if (a.impulseRemaining === 0) setFluidSimulationForceField(a.sim, null);
            }
            if (a.phase === "fluid") {
                a.fluidElapsed += dt;
                if (a.fluidElapsed >= LIFETIME) {
                    a.phase = "fading";
                    a.fadeElapsed = 0;
                    a.impulseRemaining = 0;
                    setFluidSimulationForceField(a.sim, null);
                }
            }
        }
        for (const entry of [...behaviorFluidSims.values()]) {
            const sim = entry.sim;
            if (!sim || !entry.activated) continue;
            if (!entry.emissionCompleteRaised && entry.emissionTargetCount !== null && getFluidSimulationDiagnostics(sim).activeCount >= entry.emissionTargetCount) {
                entry.emissionCompleteRaised = true;
                entry.registration.onEmissionComplete?.();
            }
            if (entry.state === "paused" || entry.state === "registered") {
                continue;
            }
            if (entry.state === "shutdown") {
                const stepDt = fluidSimulationShutdownStepDelta(
                    entry.shutdownElapsed,
                    dt * entry.timeScale,
                    entry.registration.shutdownDuration,
                    entry.registration.shutdownAlphaDecay
                );
                if (stepDt > 0) {
                    stepFluidSimulation(sim, stepDt);
                    entry.shutdownElapsed += stepDt;
                    entry.electricityElapsed += stepDt;
                }
                const lifecycle = fluidSimulationShutdownLifecycle(entry.shutdownElapsed, entry.registration.shutdownDuration, entry.registration.shutdownAlphaDecay);
                entry.opacity = lifecycle.opacity;
                if (lifecycle.stopped) {
                    // The final step is already encoded against this sim's buffers. Destroy them
                    // before the next frame records any work, after this frame has been submitted.
                    entry.pendingDispose = true;
                }
                continue;
            }
            const stepDt = dt * entry.timeScale;
            stepFluidSimulation(sim, stepDt);
            entry.electricityElapsed += stepDt;
        }
        const fluidFrame = syncFluidCollections();
        fluidElectricityRenderer.setDomains(fluidFrame.electricityDomains);
        behaviorManager.fluidSimulations.recordParticleCount(fluidFrame.count);
        behaviorManager.fluidSimulations.recordElectricity(fluidFrame.electricityDomains);
        canvas.dataset.particleCount = String(fluidFrame.count);
        canvas.dataset.activeSims = String(activeSims.length + [...behaviorFluidSims.values()].filter(({ activated, sim }) => activated && sim !== null).length);
    });
    behaviorManager.bindSystemEvents(scene, world);

    enableMaterialPlugins(scene);
    await registerScene(scene);
    frameGraphReady = true;
    await registerUtilityLayer(weaponLayer);
    await registerUtilityLayer(weaponGizmoLayer);
    if (debugUtility) await registerUtilityLayer(debugUtility);
    await warmUpFluidPipelines();
    await startEngine(engine);

    const setEnvironmentIntensity = (value: number): void => {
        const materials = new Set<PbrMaterialProps>();
        for (const mesh of [...runtimeLitMeshes, ...weaponMeshes]) {
            if (mesh.material && isPbrMaterial(mesh.material)) materials.add(mesh.material);
        }
        for (const material of materials) {
            material.environmentIntensity = value;
            markMaterialUboDirty(material);
        }
        environmentIntensity = value;
    };
    const debugToggles = [
        {
            label: "Performance overlay",
            get: () => perfOverlay.isOn(),
            set: (on: boolean): void => {
                if (on !== perfOverlay.isOn()) perfOverlay.toggle();
            },
        },
        {
            label: "Free-fly",
            get: () => playerBehavior?.isNoclip ?? false,
            set: (on: boolean): void => {
                if (on !== (playerBehavior?.isNoclip ?? false)) playerBehavior?.toggleNoclip();
            },
        },
        {
            label: "Ground-only fluid collision",
            get: () => groundOnly,
            set: (on: boolean): void => {
                if (on !== groundOnly) toggleGroundOnly();
            },
        },
        ...(inspectOverlay
            ? [
                  {
                      label: "Inspect meshes",
                      get: () => inspectOverlay.isOn(),
                      set: (on: boolean): void => {
                          if (on !== inspectOverlay.isOn()) inspectOverlay.toggle();
                      },
                  },
              ]
            : []),
        ...(fluidSimulationOverlay
            ? [
                  {
                      label: "Fluid simulations (G)",
                      get: () => fluidSimulationOverlay.isOn(),
                      set: (on: boolean): void => {
                          if (on !== fluidSimulationOverlay.isOn()) fluidSimulationOverlay.toggle();
                      },
                  },
              ]
            : []),
        ...(portalOverlay
            ? [
                  {
                      label: "Portal frusta",
                      get: () => canvas.dataset.portalFrusta !== undefined && canvas.dataset.portalFrusta !== "off",
                      set: (on: boolean): void => {
                          const enabled = canvas.dataset.portalFrusta !== undefined && canvas.dataset.portalFrusta !== "off";
                          if (on !== enabled) portalOverlay.toggle();
                      },
                  },
              ]
            : []),
        ...(lightOverlay
            ? [
                  {
                      label: "Runtime lights",
                      get: () => canvas.dataset.lightOverlay === "on",
                      set: (on: boolean): void => {
                          if (on !== (canvas.dataset.lightOverlay === "on")) lightOverlay.toggle();
                      },
                  },
              ]
            : []),
        ...(probeOverlay
            ? [
                  {
                      label: "Cubemap blend colors (V)",
                      get: () => canvas.dataset.probeOverlay === "on",
                      set: (on: boolean): void => {
                          if (on !== (canvas.dataset.probeOverlay === "on")) probeOverlay.toggle();
                      },
                  },
              ]
            : []),
        {
            label: "Metallic roughness = 0",
            get: () => forceFullyMetallicRoughnessZero,
            set: setFullyMetallicRoughnessZero,
        },
    ];
    const debugActions = [
        {
            label: "Toggle nearest light",
            run: toggleNearestLight,
            status: (): string => {
                const id = canvas.dataset.nearestLight;
                if (!id || id === "none") return "none";
                return `${id} ${canvas.dataset.nearestLightEnabled === "true" ? "on" : "off"}`;
            },
        },
        ...(colliderOverlay
            ? [
                  {
                      label: "Cycle colliders",
                      run: (): void => colliderOverlay.cycle(),
                      status: (): string => canvas.dataset.colliderOverlay ?? "off",
                  },
              ]
            : []),
    ];
    controlPanel = createAquanovaControlPanel({
        canvas,
        antiAliasing: {
            msaa: {
                label: "4x MSAA",
                get: () => msaaOn,
                set: (on) => {
                    if (on !== msaaOn) toggleMsaa();
                },
            },
            smaa: {
                label: "SMAA",
                get: () => smaaOn,
                set: (on) => {
                    if (on !== smaaOn) toggleSmaa();
                },
            },
            taa: {
                label: "TAA",
                get: () => taaOn,
                set: (on) => {
                    if (on !== taaOn) toggleTaa();
                },
            },
            specularAA: {
                label: "Specular AA",
                get: () => graphics.specularAA,
                set: setSpecularAAEnabled,
            },
            ssaa: {
                label: "2x SSAA",
                get: () => graphics.ssaa === 2,
                set: (on) => setSsaa(on ? 2 : 1),
            },
        },
        effects: {
            improvedElectricity: {
                label: "Improved electrical effect",
                get: () => graphics.improvedElectricity,
                set: setImprovedElectricity,
            },
            animateElectricity: {
                label: "Animate electrical effect",
                get: () => graphics.animateElectricity,
                set: setAnimateElectricity,
            },
            electricityArcDensity: {
                get: () => graphics.electricityArcDensity,
                set: setElectricityArcDensity,
            },
            electricityBloomDebug: !LAB_DEBUG
                ? undefined
                : {
                      label: "Compare electrical source / bloom",
                      get: () => electricityBloomDebug,
                      set: setElectricityBloomDebug,
                  },
            electricityBloomThreshold: {
                get: () => graphics.electricityBloomThreshold,
                set: setElectricityBloomThreshold,
            },
            electricityBloomStrength: {
                get: () => graphics.electricityBloomStrength,
                set: setElectricityBloomStrength,
            },
            electricityBloomRadius: {
                get: () => graphics.electricityBloomRadius,
                set: setElectricityBloomRadius,
            },
        },
        weapon: {
            model: {
                get: () => Math.max(0, LIQUEFACTOR_MODELS.indexOf(weaponViewmodel.model)),
                set: (index) => {
                    const model = LIQUEFACTOR_MODELS[index] ?? "80k";
                    setLiquefactorModel(model);
                },
                options: LIQUEFACTOR_MODELS,
            },
            sway: {
                label: "Weapon sway",
                get: () => graphics.weaponSway,
                set: setWeaponSwayEnabled,
            },
            debug: weaponDebugTools
                ? {
                      positionGizmo: {
                          label: "Position gizmo",
                          get: weaponDebugTools.positionGizmoEnabled,
                          set: weaponDebugTools.setPositionGizmoEnabled,
                      },
                      rotationGizmo: {
                          label: "Rotation gizmo",
                          get: weaponDebugTools.rotationGizmoEnabled,
                          set: weaponDebugTools.setRotationGizmoEnabled,
                      },
                      scaleGizmo: {
                          label: "Scale gizmo",
                          get: weaponDebugTools.scaleGizmoEnabled,
                          set: weaponDebugTools.setScaleGizmoEnabled,
                      },
                      localGuideGizmo: {
                          label: "Aim origin gizmo",
                          get: weaponDebugTools.localGuideGizmoEnabled,
                          set: weaponDebugTools.setLocalGuideGizmoEnabled,
                      },
                      localGuideYaw: {
                          get: weaponDebugTools.localGuideYawDegrees,
                          set: weaponDebugTools.setLocalGuideYawDegrees,
                      },
                  }
                : undefined,
        },
        audio: {
            sounds: {
                label: "Sounds",
                get: () => graphics.soundsEnabled,
                set: setSoundsEnabled,
            },
            volume: {
                get: () => graphics.soundVolume,
                set: setSoundVolume,
            },
        },
        environment: {
            localCubemapBlending: {
                label: "Local cubemap blending (B)",
                get: () => graphics.localCubemapBlending,
                set: setLocalCubemapBlending,
            },
            envIntensity: {
                get: () => environmentIntensity,
                set: setEnvironmentIntensity,
            },
            exposure: {
                get: () => exposure,
                set: (value) => {
                    exposure = value;
                    scene.imageProcessing.exposure = value;
                    weaponLayer.scene.imageProcessing.exposure = value;
                },
            },
            toneMapping: {
                get: () => toneIndex,
                set: setToneMappingIndex,
                options: TONE_MAPPINGS.map((entry) => entry.name),
            },
        },
        debug: {
            cameraPosition: {
                set: ([x, y, z]) => {
                    if (playerBehavior) playerBehavior.setCameraPosition({ x, y, z });
                    else cam.position.set(x, y, z);
                },
            },
            cameraTarget: {
                set: ([x, y, z]) => {
                    if (playerBehavior) playerBehavior.setCameraTarget({ x, y, z });
                    else cam.target.set(x, y, z);
                },
            },
            toggles: debugToggles,
            actions: debugActions,
        },
        onVisibilityChange: (visible) => {
            weaponDebugTools?.setVisible(visible);
        },
    });
    if (weaponDebugTools) controlPanel.updateWeaponTransform(weaponDebugTools.values());
    controlPanel.updateCameraTransform({
        position: [cam.position.x, cam.position.y, cam.position.z],
        target: [cam.target.x, cam.target.y, cam.target.z],
    });
    const enterCheatCode = !LAB_DEBUG
        ? null
        : createCheatCodeMatcher("idkfa", () => {
              behaviorManager.acquireAllWeapons();
              canvas.dataset.cheatCode = "idkfa";
              controlPanel?.refresh();
          });
    const pauseIndicator = document.createElement("div");
    pauseIndicator.textContent = "PAUSED";
    pauseIndicator.style.cssText =
        "position:fixed;left:50%;top:10%;transform:translateX(-50%);z-index:30;display:none;padding:8px 18px;border:1px solid rgba(150,225,255,.8);border-radius:5px;background:rgba(0,12,20,.78);color:#dff8ff;font:600 18px/1.2 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.16em;pointer-events:none;";
    document.body.appendChild(pauseIndicator);
    const pausedCallbacks = new Map<(deltaMs: number) => void, (deltaMs: number) => void>();
    let pausedPhysicsTimestepMs = 0;
    const setGamePaused = (paused: boolean): void => {
        if (paused === gamePaused) {
            return;
        }
        gamePaused = paused;
        canvas.dataset.gamePaused = String(paused);
        pauseIndicator.style.display = paused ? "block" : "none";
        if (paused) {
            pausedPhysicsTimestepMs = getPhysicsTimestepMs(world);
            setPhysicsTimestepMs(world, 0);
            soundManager.setVolume(0);
            for (let index = 0; index < scene._beforeRender.length; index++) {
                const original = scene._beforeRender[index]!;
                const wrapper = (): void => original(0);
                pausedCallbacks.set(wrapper, original);
                scene._beforeRender[index] = wrapper;
            }
        } else {
            setPhysicsTimestepMs(world, pausedPhysicsTimestepMs);
            soundManager.setVolume(graphics.soundVolume);
            for (let index = 0; index < scene._beforeRender.length; index++) {
                const original = pausedCallbacks.get(scene._beforeRender[index]!);
                if (original) {
                    scene._beforeRender[index] = original;
                }
            }
            pausedCallbacks.clear();
        }
    };
    canvas.dataset.gamePaused = "false";
    window.addEventListener("keydown", (event) => {
        if (event.repeat) return;
        if (!event.altKey && !event.ctrlKey && !event.metaKey) enterCheatCode?.(event.key);
        if (event.code === "Digit1") behaviorManager.events.emit("weaponSlotSelected", { slot: 1 });
        else if (event.code === "Digit2") behaviorManager.events.emit("weaponSlotSelected", { slot: 2 });
        else if (event.code === "Digit3") behaviorManager.events.emit("weaponSlotSelected", { slot: 3 });
        else if (event.code === "KeyH") toggleNearestLight();
        else if (event.code === "KeyP" && event.shiftKey) setGamePaused(!gamePaused);
        else if (event.code === "KeyP") perfOverlay.toggle();
        else if (event.code === "KeyG") fluidSimulationOverlay?.toggle();
        else if (event.code === "KeyB") toggleLocalCubemapBlending();
        else if (event.code === "KeyV") probeOverlay?.toggle();
        else if (event.code === "KeyU") controlPanel?.toggle();
        else return;
        controlPanel?.refresh();
    });

    if (exteriorClassifier) {
        exteriorMeshes = await exteriorClassifier.result;
        exteriorReady = true;
        canvas.dataset.exteriorMeshes = `${exteriorMeshes.size}/${classifierMeshes.length}`;
        canvas.dataset.exteriorClassifyMs = (performance.now() - exteriorClassifyStart).toFixed(1);
        portalVisibility.update();
        applyPortalDrawOrder(sceneTask);
        if (msaaSceneTask) applyPortalDrawOrder(msaaSceneTask as RenderTask);
    }
    canvas.dataset.ready = "true";
}
