// Seeing the game's lighting, without leaving the editor.
//
// The runtime view is the live authored scene, re-lit: the editor's own
// four-light authoring rig is switched off, the authored lamps are rebuilt as
// real Babylon lights over every mesh, and each element's materials are cloned
// onto the environment probe whose box it sits in. That is the whole of the
// game's lighting model, so what is on screen is what the runtime will draw.
//
// It works on the ship you are building rather than on an exported copy of it,
// which is what makes it honest AND editable: every element can still be
// picked, dragged and deleted while the mode is up, and an edit is re-lit on
// the next frame instead of on the next export.
//
// The probe volumes themselves are authored here too - the wireframe box, its
// centre marker, the orange capture point and the optional cubemap preview
// surface - because they are a property of the lighting rather than of the
// geometry. `local-environments.js` is what turns them into .env files.

const {
  ClusteredLightContainer,
  Color3,
  CubeTexture,
  DirectionalLight,
  FreeCamera,
  MeshBuilder,
  PointLight,
  Quaternion,
  ShaderMaterial,
  SpotLight,
  StandardMaterial,
  Texture,
  TransformNode,
  Vector3,
} = BABYLON;

import {
  state, emit, on, hooks, syncLightingMode, applyVisibility, environmentProbeOf, ownerIdOf,
  isProbeExcludedNode, withDeadline,
} from "./editor.js";

const ROOT_NAME = "RUNTIME_PREVIEW";
const CAPTURE_CAMERA = "PROBE_CAPTURE_CAMERA";

/** How long the scene may take to become drawable again after a capture. */
const READY_TIMEOUT_MS = 120000;

let captureCamera = null;

let preview = null;
let editorEnvironmentTexture = null;
let localEnvironmentIndex = null;
let localEnvironmentIndexPromise = null;
let localEnvironmentIndexRequest = 0;
let localEnvironmentBox = null;
let localEnvironmentBoxMaterial = null;
let localEnvironmentBoxCentre = null;
let localEnvironmentBoxCentreMaterial = null;
let localEnvironmentCamera = null;
let localEnvironmentCameraMaterial = null;
let localEnvironmentBoxSurface = null;
let localEnvironmentBoxSurfaceMaterial = null;
let localEnvironmentBoxTexture = null;
let localEnvironmentBoxTextureProbe = null;
let localEnvironmentBoxTexturePromise = null;
let localEnvironmentBoxTexturePromiseProbe = null;
let localEnvironmentBoxTextureRequest = 0;
let localEnvironmentBoxShowRequest = 0;
let localEnvironmentBoxEnvironmentVisible = false;
const authoredRoughness = new WeakMap();

/**
 * Push the environment settings onto the preview's material clones.
 *
 * Env is applied per material rather than through `scene.environmentIntensity`
 * because each clone carries its own room's cubemap: the scene-wide multiplier
 * would be a second, invisible factor on top of a value the runtime applies at
 * the material. Specular AA and the roughness factor sit here for the same
 * reason - they are material state, not scene state. All three are authored
 * ship values: the manifest carries them and the game reads them back, so what
 * this preview shows is what the game will do.
 */
function applyRuntimeReflectionSettings() {
  if (!preview) return;
  for (const mat of preview.materials) {
    if ("environmentIntensity" in mat) mat.environmentIntensity = state.envIntensity;
    mat.enableSpecularAntiAliasing = state.runtimeSpecularAA;
    const base = authoredRoughness.get(mat);
    if (typeof base === "number") {
      // Babylon clamps the final roughness in the shader; keeping the scalar above
      // 1 lets it boost low values sampled from the ORM texture.
      mat.roughness = base * state.runtimeRoughnessFactor;
    }
  }
}

/** The live preview, for tests and the console. Null when it is off. */
export function runtimePreview() {
  return preview;
}

// Every add, delete, move and inspector edit of a lamp already emits this, and
// refreshPreviewLights is a no-op while the preview is down - so this is the
// whole subscription. See refreshPreviewLights for why it is not a rebuild.
on("lights", () => refreshPreviewLights());
on("reflection", () => applyRuntimeReflectionSettings());
on("environment", () => {
  applyRuntimeReflectionSettings();
  updateLocalEnvironmentBoxSurfaceView();
});
on("modes", () => updateLocalEnvironmentBoxSurfaceView());
on("environment-probes", () => {
  applyLocalEnvironmentBoxes();
  syncElementEnvironments();
  if (localEnvironmentBox?.metadata?.probe) {
    void showEnvironmentProbe(localEnvironmentBox.metadata.probe, true, null, localEnvironmentBoxEnvironmentVisible);
  }
});

function waitForTexture(texture) {
  if (texture.isReady()) return Promise.resolve();
  return new Promise((resolve, reject) => {
    let loadObserver = null;
    let errorObserver = null;
    const cleanup = () => {
      if (loadObserver) texture.onLoadObservable.remove(loadObserver);
      if (errorObserver) Texture.OnTextureLoadErrorObservable.remove(errorObserver);
    };
    loadObserver = texture.onLoadObservable.addOnce(() => {
      cleanup();
      resolve();
    });
    errorObserver = Texture.OnTextureLoadErrorObservable.add((failed) => {
      if (failed !== texture) return;
      cleanup();
      reject(new Error(`failed to load ${texture.url}`));
    });
  });
}

async function fetchLocalEnvironmentIndex() {
  if (localEnvironmentIndex) return localEnvironmentIndex;
  if (localEnvironmentIndexPromise) return localEnvironmentIndexPromise;
  const request = ++localEnvironmentIndexRequest;
  const promise = (async () => {
    try {
      const response = await fetch("/api/local-environments", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const index = await response.json();
      if (request !== localEnvironmentIndexRequest) return null;
      localEnvironmentIndex = index;
      return localEnvironmentIndex;
    } catch (err) {
      console.warn("[aquanova] local environments unavailable; using authored bounds", err);
      return null;
    } finally {
      if (request === localEnvironmentIndexRequest) localEnvironmentIndexPromise = null;
    }
  })();
  localEnvironmentIndexPromise = promise;
  return promise;
}

function resetLocalEnvironmentAssets() {
  localEnvironmentIndexRequest++;
  localEnvironmentBoxShowRequest++;
  localEnvironmentIndex = null;
  localEnvironmentIndexPromise = null;
  localEnvironmentBoxTextureRequest++;
  localEnvironmentBoxTexturePromise = null;
  localEnvironmentBoxTexturePromiseProbe = null;
  localEnvironmentBoxTexture?.dispose();
  localEnvironmentBoxTexture = null;
  localEnvironmentBoxTextureProbe = null;
  localEnvironmentBoxSurface?.setEnabled(false);
}

/** Drop the cached probe index and cubemaps after a capture rewrites them. */
export async function refreshEnvironmentProbeAssets() {
  const id = localEnvironmentBox?.metadata?.probe;
  const visible = !!localEnvironmentBox?.isEnabled();
  resetLocalEnvironmentAssets();
  if (id && visible) {
    await showEnvironmentProbe(id, true, null, localEnvironmentBoxEnvironmentVisible);
  }
}

function generatedProbeOf(index, id) {
  return index?.probes?.[id] || index?.chunks?.[id] || null;
}

function editorPoint(point) {
  if (!Array.isArray(point) || point.length !== 3) return null;
  const converted = [-Number(point[0]), Number(point[1]), Number(point[2])];
  return converted.every(Number.isFinite) ? converted : null;
}

/** Authored probe volume plus its generated asset, when one exists. */
export async function localEnvironmentProbeOf(id) {
  const index = await fetchLocalEnvironmentIndex();
  const authored = environmentProbeOf(id);
  const generated = generatedProbeOf(index, id);
  const usableGenerated = generated && (!index?.probes || authored);
  return {
    effective:
      authored ||
      (usableGenerated
        ? {
            id,
            boxPosition: editorPoint(generated.boxPosition),
            boxSize: generated.boxSize?.map(Number),
            capturePosition: editorPoint(generated.position),
            resolution: Number(generated.resolution) || 512,
          }
        : null),
    generated,
  };
}

function applyTextureBox(texture, id, index = localEnvironmentIndex, draft = null) {
  const generated = generatedProbeOf(index, id);
  const box =
    draft ||
    environmentProbeOf(id) ||
    (generated
      ? {
          boxPosition: editorPoint(generated.boxPosition),
          boxSize: generated.boxSize?.map(Number),
        }
      : null);
  if (!box) return;
  texture.boundingBoxPosition = Vector3.FromArray(box.boxPosition);
  texture.boundingBoxSize = Vector3.FromArray(box.boxSize);
}

function applyLocalEnvironmentBoxes() {
  if (!preview) return;
  for (const [id, texture] of preview.localEnvironments) {
    applyTextureBox(texture, id);
  }
}

function ensureLocalEnvironmentBox() {
  if (localEnvironmentBox) {
    return {
      box: localEnvironmentBox,
      centre: localEnvironmentBoxCentre,
      camera: localEnvironmentCamera,
    };
  }
  localEnvironmentBoxMaterial = new StandardMaterial("LOCAL_ENVIRONMENT_BOX_mat", state.scene);
  localEnvironmentBoxMaterial.emissiveColor = new Color3(0.15, 0.9, 1);
  localEnvironmentBoxMaterial.diffuseColor = Color3.Black();
  localEnvironmentBoxMaterial.specularColor = Color3.Black();
  localEnvironmentBoxMaterial.disableLighting = true;
  localEnvironmentBoxMaterial.wireframe = true;
  localEnvironmentBoxMaterial.alpha = 0.85;
  localEnvironmentBox = new TransformNode("LOCAL_ENVIRONMENT_BOX_ROOT", state.scene);
  localEnvironmentBox.metadata = { gizmo: true, localEnvironmentBox: true, probe: null };
  const localEnvironmentBoxMesh = MeshBuilder.CreateBox("LOCAL_ENVIRONMENT_BOX", { size: 1 }, state.scene);
  localEnvironmentBoxMesh.parent = localEnvironmentBox;
  localEnvironmentBoxMesh.material = localEnvironmentBoxMaterial;
  localEnvironmentBoxMesh.isPickable = true;
  localEnvironmentBoxMesh.renderingGroupId = 3;
  localEnvironmentBoxMesh.metadata = {
    gizmo: true,
    localEnvironmentBox: true,
    environmentProbeRoot: localEnvironmentBox,
  };
  localEnvironmentBoxCentreMaterial = new StandardMaterial("LOCAL_ENVIRONMENT_BOX_CENTRE_mat", state.scene);
  localEnvironmentBoxCentreMaterial.emissiveColor = new Color3(0.15, 0.9, 1);
  localEnvironmentBoxCentreMaterial.diffuseColor = Color3.Black();
  localEnvironmentBoxCentreMaterial.specularColor = Color3.Black();
  localEnvironmentBoxCentreMaterial.disableLighting = true;
  localEnvironmentBoxCentre = MeshBuilder.CreateSphere("LOCAL_ENVIRONMENT_BOX_CENTRE", { diameter: 0.3, segments: 12 }, state.scene);
  localEnvironmentBoxCentre.material = localEnvironmentBoxCentreMaterial;
  localEnvironmentBoxCentre.isPickable = false;
  localEnvironmentBoxCentre.renderingGroupId = 3;
  localEnvironmentBoxCentre.metadata = {
    gizmo: true,
    localEnvironmentBoxCentre: true,
    probe: null,
    environmentProbeRoot: localEnvironmentBox,
  };
  localEnvironmentCameraMaterial = new StandardMaterial("LOCAL_ENVIRONMENT_CAMERA_mat", state.scene);
  localEnvironmentCameraMaterial.emissiveColor = new Color3(1, 0.45, 0.1);
  localEnvironmentCameraMaterial.diffuseColor = Color3.Black();
  localEnvironmentCameraMaterial.specularColor = Color3.Black();
  localEnvironmentCameraMaterial.disableLighting = true;
  localEnvironmentCamera = MeshBuilder.CreateSphere("LOCAL_ENVIRONMENT_CAMERA", { diameter: 0.24, segments: 12 }, state.scene);
  localEnvironmentCamera.material = localEnvironmentCameraMaterial;
  localEnvironmentCamera.isPickable = false;
  localEnvironmentCamera.renderingGroupId = 3;
  localEnvironmentCamera.metadata = {
    gizmo: true,
    localEnvironmentCamera: true,
    probe: null,
    environmentProbeRoot: localEnvironmentBox,
  };
  return {
    box: localEnvironmentBox,
    centre: localEnvironmentBoxCentre,
    camera: localEnvironmentCamera,
  };
}

function ensureLocalEnvironmentBoxSurface() {
  if (localEnvironmentBoxSurface) return localEnvironmentBoxSurface;
  localEnvironmentBoxSurfaceMaterial = new ShaderMaterial(
    "LOCAL_ENVIRONMENT_BOX_SURFACE_mat",
    state.scene,
    {
      vertexSource: `
precision highp float;
attribute vec3 position;
uniform mat4 world;
uniform mat4 worldViewProjection;
varying vec3 vWorldPosition;
void main(void) {
  vec4 worldPosition = world * vec4(position, 1.0);
  vWorldPosition = worldPosition.xyz;
  gl_Position = worldViewProjection * vec4(position, 1.0);
}`,
      fragmentSource: `
precision highp float;
uniform samplerCube environmentSampler;
uniform vec3 capturePosition;
uniform mat4 reflectionMatrix;
uniform float decodeRGBD;
uniform float oppositeZ;
uniform float exposureLinear;
uniform float environmentIntensity;
uniform int toneMapping;
varying vec3 vWorldPosition;

vec3 acesToneMapping(vec3 color) {
  mat3 inputTransform = mat3(
    vec3(0.59719, 0.07600, 0.02840),
    vec3(0.35458, 0.90834, 0.13383),
    vec3(0.04823, 0.01566, 0.83777));
  mat3 outputTransform = mat3(
    vec3(1.60475, -0.10208, -0.00327),
    vec3(-0.53108, 1.10813, -0.07276),
    vec3(-0.07367, -0.00605, 1.07602));
  color = inputTransform * color;
  vec3 a = color * (color + 0.0245786) - 0.000090537;
  vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
  return clamp(outputTransform * (a / b), 0.0, 1.0);
}

vec3 neutralToneMapping(vec3 color) {
  float darkest = min(color.r, min(color.g, color.b));
  float offset = darkest < 0.08 ? darkest - 6.25 * darkest * darkest : 0.04;
  color -= offset;
  float peak = max(color.r, max(color.g, color.b));
  const float startCompression = 0.76;
  if (peak < startCompression) return color;
  float distance = 1.0 - startCompression;
  float newPeak = 1.0 - distance * distance / (peak + distance - startCompression);
  color *= newPeak / peak;
  float desaturation = 1.0 - 1.0 / (0.15 * (peak - newPeak) + 1.0);
  return mix(color, vec3(newPeak), desaturation);
}

void main(void) {
  vec3 direction = normalize(vWorldPosition - capturePosition);
  direction = (reflectionMatrix * vec4(direction, 0.0)).xyz;
  direction.z *= oppositeZ;
  vec4 sampleColor = textureCube(environmentSampler, direction);
  vec3 linearColor = sampleColor.rgb;
  if (decodeRGBD > 0.5) {
    linearColor = pow(linearColor, vec3(2.2)) / max(sampleColor.a, 1e-6);
  }
  linearColor *= exposureLinear * environmentIntensity;
  if (toneMapping == 1) {
    linearColor = 1.0 - exp2(-1.590579 * linearColor);
  } else if (toneMapping == 2) {
    linearColor = acesToneMapping(linearColor);
  } else if (toneMapping == 3) {
    linearColor = neutralToneMapping(max(linearColor, vec3(1e-7)));
  }
  vec3 displayColor = clamp(pow(max(linearColor, vec3(0.0)), vec3(1.0 / 2.2)), 0.0, 1.0);
  gl_FragColor = vec4(displayColor, 0.72);
}`,
    },
    {
      attributes: ["position"],
      uniforms: ["world", "worldViewProjection", "capturePosition", "reflectionMatrix", "decodeRGBD", "oppositeZ", "exposureLinear", "environmentIntensity", "toneMapping"],
      samplers: ["environmentSampler"],
      needAlphaBlending: true,
    }
  );
  localEnvironmentBoxSurfaceMaterial.backFaceCulling = false;
  updateLocalEnvironmentBoxSurfaceView();
  localEnvironmentBoxSurface = MeshBuilder.CreateBox("LOCAL_ENVIRONMENT_BOX_SURFACE", { size: 1 }, state.scene);
  localEnvironmentBoxSurface.material = localEnvironmentBoxSurfaceMaterial;
  localEnvironmentBoxSurface.isPickable = false;
  localEnvironmentBoxSurface.renderingGroupId = 2;
  localEnvironmentBoxSurface.metadata = {
    gizmo: true,
    localEnvironmentBoxSurface: true,
    probe: null,
  };
  return localEnvironmentBoxSurface;
}

function updateLocalEnvironmentBoxSurfaceView() {
  if (!localEnvironmentBoxSurfaceMaterial || !state.scene) return;
  const config = state.scene.imageProcessingConfiguration;
  const IPC = BABYLON.ImageProcessingConfiguration;
  let toneMapping = 0;
  if (config.toneMappingEnabled) {
    if (config.toneMappingType === IPC.TONEMAPPING_STANDARD) toneMapping = 1;
    else if (config.toneMappingType === IPC.TONEMAPPING_ACES) toneMapping = 2;
    else if (config.toneMappingType === IPC.TONEMAPPING_KHR_PBR_NEUTRAL) toneMapping = 3;
  }
  localEnvironmentBoxSurfaceMaterial.setFloat("exposureLinear", config.exposure);
  localEnvironmentBoxSurfaceMaterial.setFloat("environmentIntensity", state.envIntensity);
  localEnvironmentBoxSurfaceMaterial.setInt("toneMapping", toneMapping);
}

async function localEnvironmentTextureForBox(id) {
  if (localEnvironmentBoxTexture && localEnvironmentBoxTextureProbe === id) {
    return localEnvironmentBoxTexture;
  }
  if (localEnvironmentBoxTexturePromise && localEnvironmentBoxTexturePromiseProbe === id) {
    return localEnvironmentBoxTexturePromise;
  }
  const index = await fetchLocalEnvironmentIndex();
  const entry = generatedProbeOf(index, id);
  if (!entry?.env) return null;
  const request = ++localEnvironmentBoxTextureRequest;
  localEnvironmentBoxTexturePromiseProbe = id;
  localEnvironmentBoxTexturePromise = (async () => {
    const query = entry.hash ? `?h=${encodeURIComponent(entry.hash)}` : "";
    const texture = CubeTexture.CreateFromPrefilteredData(`/environments/${encodeURIComponent(entry.env)}${query}`, state.scene);
    texture.coordinatesMode = Texture.SKYBOX_MODE;
    try {
      await waitForTexture(texture);
    } catch (err) {
      texture.dispose();
      console.warn(`[aquanova] local environment faces unavailable for ${id}`, err);
      return null;
    }
    if (request !== localEnvironmentBoxTextureRequest) {
      texture.dispose();
      return null;
    }
    localEnvironmentBoxTexture?.dispose();
    localEnvironmentBoxTexture = texture;
    localEnvironmentBoxTextureProbe = id;
    return texture;
  })();
  try {
    return await localEnvironmentBoxTexturePromise;
  } finally {
    if (request === localEnvironmentBoxTextureRequest) {
      localEnvironmentBoxTexturePromise = null;
      localEnvironmentBoxTexturePromiseProbe = null;
    }
  }
}

/** Show, hide, or live-preview one explicit local-environment volume. */
export async function showEnvironmentProbe(id, visible, draft = null, showEnvironment = localEnvironmentBoxEnvironmentVisible) {
  const request = ++localEnvironmentBoxShowRequest;
  localEnvironmentBoxEnvironmentVisible = !!showEnvironment;
  if (!visible || !id) {
    localEnvironmentBox?.setEnabled(false);
    localEnvironmentBoxCentre?.setEnabled(false);
    localEnvironmentCamera?.setEnabled(false);
    localEnvironmentBoxSurface?.setEnabled(false);
    return null;
  }
  const info = await localEnvironmentProbeOf(id);
  if (request !== localEnvironmentBoxShowRequest) return info;
  const box = draft || info.effective;
  if (!box) {
    localEnvironmentBox?.setEnabled(false);
    localEnvironmentBoxCentre?.setEnabled(false);
    localEnvironmentCamera?.setEnabled(false);
    localEnvironmentBoxSurface?.setEnabled(false);
    return info;
  }
  const meshes = ensureLocalEnvironmentBox();
  meshes.box.position.copyFromFloats(...box.boxPosition);
  meshes.box.scaling.copyFromFloats(...box.boxSize);
  meshes.box.metadata.probe = id;
  meshes.box.name = id;
  meshes.box.setEnabled(true);
  meshes.centre.position.copyFromFloats(...box.boxPosition);
  meshes.centre.metadata.probe = id;
  meshes.centre.setEnabled(true);
  meshes.camera.position.copyFromFloats(...box.capturePosition);
  meshes.camera.metadata.probe = id;
  meshes.camera.setEnabled(true);
  if (!localEnvironmentBoxEnvironmentVisible) {
    localEnvironmentBoxSurface?.setEnabled(false);
    return info;
  }
  const surface = ensureLocalEnvironmentBoxSurface();
  surface.setEnabled(false);
  const texture = await localEnvironmentTextureForBox(id);
  if (request !== localEnvironmentBoxShowRequest || !texture || !localEnvironmentBoxEnvironmentVisible || localEnvironmentBox?.metadata?.probe !== id) return info;
  const capturePosition = box.capturePosition || box.boxPosition;
  localEnvironmentBoxSurfaceMaterial.setTexture("environmentSampler", texture);
  localEnvironmentBoxSurfaceMaterial.setVector3("capturePosition", Vector3.FromArray(capturePosition));
  localEnvironmentBoxSurfaceMaterial.setMatrix("reflectionMatrix", texture.getReflectionTextureMatrix());
  localEnvironmentBoxSurfaceMaterial.setFloat("decodeRGBD", texture.isRGBD ? 1 : 0);
  const oppositeZ = state.scene.useRightHandedSystem ? !texture.invertZ : texture.invertZ;
  localEnvironmentBoxSurfaceMaterial.setFloat("oppositeZ", oppositeZ ? -1 : 1);
  updateLocalEnvironmentBoxSurfaceView();
  surface.position.copyFromFloats(...box.boxPosition);
  surface.scaling.copyFromFloats(...box.boxSize);
  surface.metadata.probe = id;
  surface.setEnabled(true);
  return info;
}

async function loadLocalEnvironments() {
  const index = await fetchLocalEnvironmentIndex();
  try {
    if (!index) return new Map();

    const textures = new Map();
    const entries = index?.probes || index?.chunks || {};
    await Promise.all(
      Object.entries(entries).map(async ([id, entry]) => {
        if (!entry?.env || !entry?.boxPosition || !entry?.boxSize) return;
        if (index?.probes && !environmentProbeOf(id)) return;
        const query = entry.hash ? `?h=${encodeURIComponent(entry.hash)}` : "";
        const texture = CubeTexture.CreateFromPrefilteredData(`/environments/${encodeURIComponent(entry.env)}${query}`, state.scene);
        texture.name = `LocalEnvironment_${id}`;
        texture.gammaSpace = false;
        applyTextureBox(texture, id, index);
        try {
          await waitForTexture(texture);
          textures.set(id, texture);
        } catch (err) {
          texture.dispose();
          console.warn(`[aquanova] local environment ${id} failed to load`, err);
        }
      })
    );
    return textures;
  } catch (err) {
    console.warn("[aquanova] local environments unavailable; using the global environment", err);
    return new Map();
  }
}

const PRIMITIVE_WRAPPER = /_primitive\d+$/;
const AXES = ["x", "y", "z"];
/** Two overlap shares closer than this count as equal, so the tie goes to the tighter probe. */
const SHARE_EPSILON = 1e-6;

/** The chunk an authored mesh belongs to, which is what scopes its lamps. */
function chunkOf(node) {
  const id = ownerIdOf(node);
  return id ? state.placements.get(id)?.chunk ?? null : null;
}

/**
 * Put every ship mesh on a material that carries its own room's probe.
 *
 * The clone is not optional and it is not an optimisation. Kit materials are
 * deduplicated across the whole catalogue (see kit.js), so one `MI_Trim_01`
 * serves every room in the ship - and a probe is per room. There is no way to
 * hang two different cubemaps on one material, so the ship gets one copy of
 * each material per probe that uses it.
 *
 * A mesh in no probe box at all gets a clone too, with no reflection texture:
 * that is exactly what the runtime does with it, and giving it the authored
 * material instead would leave it reflecting whatever the editor's global HDRI
 * happens to be - the one difference that would make the preview a lie.
 *
 *
 * WHY THIS CANNOT JUST ASSIGN `mesh.material`
 *
 * Every kit mesh in the ship is an `InstancedMesh` (see kit.js: a module is
 * loaded once as a hidden prototype and every placement is `createInstance`).
 * Babylon's `InstancedMesh.material` is a getter onto the SOURCE mesh, and its
 * setter is a no-op that logs a warning - an instance cannot carry a material
 * of its own, because the whole point of instancing is that one draw call
 * serves them all.
 *
 * So a probe cannot be hung on an instance. It has to be hung on a source
 * mesh, and there has to be one source mesh per probe that the module appears
 * in. That is exactly what this builds: for each (prototype, probe) pair a
 * disabled clone of the prototype - geometry SHARED, Babylon refcounts it, so
 * the clone costs a draw-call bucket and no memory - wearing the probe's
 * material. Each authored instance then gets a preview instance of the right
 * clone, parented to it at identity so it inherits the authored world matrix
 * for free, and the authored instance is made invisible.
 *
 * Invisible, not disabled: `setEnabled(false)` would take the preview instance
 * hanging off it down too, and it would take the element out of the editor's
 * own picking predicate. The authored ship stays enabled, stays pickable and
 * stays exactly where it was - it is only the pixels that come from somewhere
 * else, which is what lets an element be dragged, deleted and undone while the
 * preview is up.
 *
 * A ship mesh that is NOT an instance is dressed the old way, by swapping its
 * material. Nothing in the kit produces one today, but the two paths are one
 * `sourceMesh` check apart and the alternative is a mesh that silently renders
 * with the editor's HDRI.
 *
 * Nothing here mutates an authored material or an authored transform, so
 * leaving the mode is putting the meshes back the way they came in.
 */
function dressMeshes(meshes, localEnvironments) {
  const clones = new Map();
  const prototypes = new Map();
  const materials = new Set();
  const sourceOf = new Map();
  const probeOf = new Map();
  const elementMeshes = new Map();
  const elementGroups = new Map();
  const previewOf = new Map();
  const hidden = new Map();
  const owned = [];

  const materialFor = (source, probe) => {
    const key = `${probe || ""}|${source.uniqueId}`;
    let copy = clones.get(key);
    if (copy) return copy;
    const shared = new Set(source.getActiveTextures());
    copy = source.clone(`${source.name}__runtime__${probe || "none"}`);
    // `Material.clone` deep-copies every texture slot - Babylon's CopySource
    // runs `sourceProperty.clone()` on each one - so the copy owns a private
    // wrapper for each of the ship's base colour, normal and ORM maps. Those
    // wrappers belong to nobody else, so they have to be collected and freed
    // when the mode is left, or every switch leaks one per slot per clone.
    //
    // Diffing against the source is what makes this safe: a texture the clone
    // shares rather than owns - `CopySource` passes render targets straight
    // through - is in both sets and is left alone.
    for (const t of copy.getActiveTextures()) if (!shared.has(t)) owned.push(t);
    if ("reflectionTexture" in copy) {
      copy.reflectionTexture = probe ? localEnvironments.get(probe) ?? null : null;
    }
    // The falloff curve is a MATERIAL setting in Babylon, and the clustered
    // path reads the same one (pbrClusteredLightingFunctions.fx calls the
    // shared computeDistanceLightFalloff), so this one line decides how every
    // preview lamp fades.
    //
    // The default - physical, a plain 1/d² - is the wrong one twice over. It
    // ignores `range`, so the Range field authors a number nothing reads; and
    // the clustered container still SIZES AND CULLS each light proxy by
    // `range`, so a physical clustered lamp is cut off dead in a straight line
    // where its proxy ends. That is the block artefact Babylon's own clustered
    // lighting page warns about, and its remedy is exactly this.
    //
    // Of the two falloffs the page offers, glTF is the one to pick here: it is
    // the curve Babylon-Lite's clustered shader hardcodes (clustered-light-wgsl
    // .ts), so a clustered lamp - which is nearly every lamp on the ship - now
    // fades in the preview exactly as it will in the game. An unclustered lamp
    // is the near miss: Lite's analytic path has no glTF branch, so the game
    // ramps it down linearly to the same `range` this curve windows it at.
    if ("useGLTFLightFalloff" in copy) copy.useGLTFLightFalloff = true;
    // Babylon compiles a fixed number of analytic light slots into a material
    // and silently drops the rest. The clustered container counts as one slot
    // however many lamps it holds, so four is one cluster plus three scoped
    // lamps - which is what a room of this ship actually lights with.
    copy.maxSimultaneousLights = MAX_PREVIEW_LIGHTS;
    // Back-face culling is left exactly as the kit authored it: a one-sided
    // panel mounted the wrong way round should look wrong here too, because
    // that is the bug and this is the view that is meant to show it.
    copy.metadata = { ...(copy.metadata || {}), runtimePreview: true };
    authoredRoughness.set(copy, typeof copy.roughness === "number" ? copy.roughness : 1);
    clones.set(key, copy);
    materials.add(copy);
    return copy;
  };

  /** The stand-in prototype for one kit prototype under one probe. */
  const prototypeFor = (source, probe) => {
    const key = `${probe || ""}|${source.uniqueId}`;
    let proto = prototypes.get(key);
    if (proto) return proto;
    // `doNotCloneChildren`: the prototype's own children are the loader's
    // leftovers, and an instance renders one mesh.
    proto = source.clone(`${source.name}__runtime__${probe || "none"}`, null, true);
    proto.setEnabled(false);
    proto.isPickable = false;
    proto.material = materialFor(source.material, probe);
    proto.metadata = { runtimePreview: true };
    prototypes.set(key, proto);
    return proto;
  };

  /**
   * The mesh that will actually be drawn where `mesh` is.
   *
   * For an instance that is a new instance of a per-probe prototype, parented
   * to the authored one so it needs no per-frame transform sync at all. The
   * placement metadata is carried across because the lamps scope themselves by
   * asking a mesh which chunk it is in (see chunkOf), and the mesh they are
   * asking about is the one being lit.
   */
  const previewFor = (mesh, probe) => {
    const source = mesh.sourceMesh;
    if (!source) {
      // A real mesh: dress it where it stands.
      const authored = sourceOf.get(mesh) || mesh.material;
      sourceOf.set(mesh, authored);
      mesh.material = materialFor(authored, probe);
      return mesh;
    }
    const stand = prototypeFor(source, probe).createInstance(`${mesh.name}__runtime__`);
    stand.parent = mesh;
    stand.position.setAll(0);
    stand.rotationQuaternion = Quaternion.Identity();
    stand.scaling.setAll(1);
    stand.isPickable = false;
    stand.metadata = { runtimePreview: true, placementRoot: mesh.metadata?.placementRoot ?? null };
    if (!hidden.has(mesh)) hidden.set(mesh, mesh.isVisible);
    mesh.isVisible = false;
    previewOf.set(mesh, stand);
    return stand;
  };

  // Probes are resolved once per ELEMENT: every primitive of one placement must
  // land on the same box, so the whole element's unioned bounds decide. A wall
  // cannot have its trim band lit by the corridor and its face by the room.
  for (const mesh of meshes) {
    const element = elementOf(mesh);
    const group = elementMeshes.get(element);
    if (group) group.push(mesh);
    else elementMeshes.set(element, [mesh]);
  }

  const dressed = [];
  for (const [element, group] of elementMeshes) {
    const probe = localEnvironmentProbeForMeshes(group, localEnvironments);
    probeOf.set(element, probe);
    const lot = [];
    for (const mesh of group) {
      if (!mesh.material) continue;
      lot.push(previewFor(mesh, probe));
    }
    elementGroups.set(element, lot);
    dressed.push(...lot);
  }

  return {
    localEnvironments,
    materials,
    materialFor,
    previewFor,
    prototypes,
    previewOf,
    hidden,
    sourceOf,
    probeOf,
    elementMeshes,
    elementGroups,
    owned,
    meshes: dressed,
    inProbe: [...probeOf.values()].filter(Boolean).length,
    outsideProbe: [...probeOf.values()].filter((p) => !p).length,
  };
}

/**
 * Undo everything dressMeshes did to the authored scene, and free what it made.
 *
 * The authored meshes are put back FIRST - visible again, on their own
 * materials - so nothing is left pointing at a disposed material or standing
 * invisible with nothing drawn in its place, even for a frame.
 */
function undressMeshes(dressing) {
  if (!dressing) return;
  for (const stand of dressing.previewOf.values()) if (!stand.isDisposed()) stand.dispose();
  for (const [mesh, wasVisible] of dressing.hidden) if (!mesh.isDisposed()) mesh.isVisible = wasVisible;
  for (const [mesh, source] of dressing.sourceOf) if (!mesh.isDisposed()) mesh.material = source;
  // The prototypes share their geometry with the kit's own, which Babylon
  // refcounts, so this frees a draw-call bucket and leaves the vertex buffers
  // alone. Materials are freed separately below because several prototypes
  // share one.
  for (const proto of dressing.prototypes.values()) proto.dispose(false, false);
  for (const m of dressing.materials) m.dispose(true, false);
  for (const t of dressing.owned) t.dispose();
}

/**
 * The placement a mesh belongs to.
 *
 * Babylon's glTF loader splits a multi-material node into `<name>_primitiveN` meshes under a
 * wrapper TransformNode of the same name, while a single-primitive node IS the mesh. Both shapes
 * collapse to one key here, because a probe is a property of the element: a wall cannot have its
 * trim band lit by the corridor and its face by the room.
 */
function elementOf(mesh) {
  const parent = mesh.parent;
  if (!parent) return mesh;
  const name = String(mesh.name || "");
  return PRIMITIVE_WRAPPER.test(name) && name.replace(PRIMITIVE_WRAPPER, "") === String(parent.name || "") ? parent : mesh;
}

/** World AABB of an element — every primitive unioned. Null when nothing has bounds. */
function elementWorldBounds(meshes) {
  const min = new Vector3(Infinity, Infinity, Infinity);
  const max = new Vector3(-Infinity, -Infinity, -Infinity);
  for (const mesh of meshes) {
    mesh.computeWorldMatrix(true);
    const box = mesh.getBoundingInfo().boundingBox;
    min.minimizeInPlace(box.minimumWorld);
    max.maximizeInPlace(box.maximumWorld);
  }
  return min.x > max.x ? null : { min, max };
}

/**
 * The probe holding the largest share of a world-space box.
 *
 * Membership is INTERSECTION, not containment. A probe box is drawn around the room it captures, so
 * an element flush with a wall — a skirting band, a door frame, a ceiling trim — routinely pokes a
 * centimetre or two through a face, and a containment test would leave it with no environment at
 * all rather than the obvious one.
 *
 * The score is a per-axis FRACTION rather than an overlap volume because ship trim is frequently a
 * zero-thickness sliver whose volume is exactly zero against every probe. A box sitting entirely
 * inside several probes scores 1 against all of them, and that tie goes to the tightest one — so a
 * small room nested inside a corridor's probe still wins its own geometry, exactly as it did when
 * this was a containment test, and a point query is the degenerate case of the same rule.
 */
function localEnvironmentProbeForBounds(min, max, environments = preview?.localEnvironments) {
  let best = null;
  let bestShare = 0;
  let bestVolume = Infinity;
  for (const [id, texture] of environments || []) {
    if (localEnvironmentIndex?.probes && !environmentProbeOf(id)) continue;
    const centre = texture.boundingBoxPosition;
    const size = texture.boundingBoxSize;
    let share = 1;
    for (const axis of AXES) {
      const half = size[axis] * 0.5;
      const lo = Math.max(min[axis], centre[axis] - half);
      const hi = Math.min(max[axis], centre[axis] + half);
      if (hi < lo) {
        share = 0;
        break;
      }
      const extent = max[axis] - min[axis];
      if (extent > 0) share *= (hi - lo) / extent;
    }
    if (share <= 0) continue;
    const volume = size.x * size.y * size.z;
    if (share > bestShare + SHARE_EPSILON || (share > bestShare - SHARE_EPSILON && volume < bestVolume)) {
      best = id;
      bestShare = share;
      bestVolume = volume;
    }
  }
  return best;
}

/** The probe for a whole element, resolved from its unioned world bounds. */
function localEnvironmentProbeForMeshes(meshes, environments = preview?.localEnvironments) {
  const bounds = elementWorldBounds(meshes);
  return bounds ? localEnvironmentProbeForBounds(bounds.min, bounds.max, environments) : null;
}

/**
 * Re-resolve every element's probe, once per frame.
 *
 * Dragging a crate through a doorway has to move it onto the next room's
 * cubemap the moment its bounds cross - which is the whole reason this runs off
 * `onBeforeRender` rather than off an edit event. The comparison is against the
 * probe the element already has, so a still ship costs one bounds union per
 * element and no material churn at all.
 *
 * A changed probe means a changed SOURCE mesh, not just a changed material -
 * see dressMeshes for why an instance cannot carry one - so the element's
 * stand-ins are thrown away and re-made off the right prototype. That is a
 * handful of instances of already-loaded geometry, and it only happens on the
 * frame an element actually crosses a boundary.
 */
function syncElementEnvironments() {
  if (!preview) return;
  let moved = false;
  for (const [element, group] of preview.elementMeshes) {
    const probe = localEnvironmentProbeForMeshes(group);
    if (preview.probeOf.get(element) === probe) continue;
    preview.probeOf.set(element, probe);
    const lot = [];
    for (const mesh of group) {
      if (!mesh.material) continue;
      const stand = preview.previewOf.get(mesh);
      if (stand && !stand.isDisposed()) stand.dispose();
      preview.previewOf.delete(mesh);
      lot.push(preview.previewFor(mesh, probe));
    }
    preview.elementGroups.set(element, lot);
    moved = true;
  }
  if (!moved) return;
  preview.meshes = [...preview.elementGroups.values()].flat();
  preview.inProbe = [...preview.probeOf.values()].filter(Boolean).length;
  preview.outsideProbe = [...preview.probeOf.values()].filter((p) => !p).length;
  applyRuntimeReflectionSettings();
}
// ------------------------------------------------------- the runtime lamps
//
// The lamps the game creates, rebuilt on the live ship. Every mesh is lit by
// them - there is no second, pre-computed half any more - so this is the whole
// of the ship's direct lighting and the only place the editor can show what a
// Range or a cone angle actually does.
//
// The records are read off `state.lights`, the editor's live record, so an
// inspector edit is on screen on the next frame. Nothing here reads a file.
//
// This is `lab/lite/src/demos/aquanova/lights.ts` rebuilt on Babylon.js, and
// the two are kept deliberately parallel - same -Y emission axis, same
// clustered-vs-scoped rule. Where they differ it is because the engines differ,
// and each such place is called out below.

/** How many analytic light slots one preview material is compiled to take. */
const MAX_PREVIEW_LIGHTS = 4;

/** Column-major world matrix -> the world direction of the node's local -Y.
 *  See the EMISSION AXIS note in lights.js for why it is -Y and not -Z. */
function emissionAxis(node) {
  return Vector3.TransformNormal(new Vector3(0, -1, 0), node.getWorldMatrix()).normalize();
}

/**
 * Which authored lamps the preview should build, and what each one lights.
 *
 * `lit` is every dressed mesh in the ship. A clustered lamp takes all of them;
 * a scoped one takes only its own room's, which is what keeps the ship inside
 * Babylon's per-material light budget.
 */
function previewLightSpecs(lit) {
  const byChunk = new Map();
  for (const mesh of lit) {
    const chunk = chunkOf(mesh) ?? "";
    const list = byChunk.get(chunk);
    if (list) list.push(mesh);
    else byChunk.set(chunk, [mesh]);
  }

  const specs = [];
  const stats = { point: 0, spot: 0, directional: 0, off: 0, idle: 0 };
  for (const light of state.lights.values()) {
    const r = light.runtime;
    // "none" is a lamp that creates nothing at runtime - an authored record
    // kept for its position while it is switched off.
    if (!r || r.type === "none") {
      stats.off++;
      continue;
    }

    // Scoping. A clustered lamp is scene-global by agreement - the cluster is
    // cheap enough that a mesh near a doorway picking up the next room's lamps
    // is a feature. A UBO lamp is scarce, so it is held to its own chunk. The
    // chunk comes from the owner placement, so moving an element to another
    // room re-scopes its lamps without a reload.
    //
    // A lamp with nothing to light is not built at all. Babylon-Lite reads an
    // EMPTY `includedOnlyMeshIds` as "lights nothing"; Babylon.js reads an empty
    // `includedOnlyMeshes` as "no filter", i.e. lights EVERYTHING. Skipping is
    // the only spelling that means the same thing in both, and an empty room
    // genuinely has nothing here for its lamps to do.
    const chunk = state.placements.get(light.owner)?.chunk ?? "";
    const scope = r.clustered ? lit : (byChunk.get(chunk) ?? []);
    if (!scope.length) {
      stats.idle++;
      continue;
    }
    specs.push({ light, r, chunk, scope });
    stats[r.type]++;
  }
  return { specs, stats };
}
/** Babylon instances inherit lights from their source mesh. */
function lightScope(meshes) {
  return [...new Set(meshes.map((mesh) => mesh._sourceMesh || mesh))];
}

/**
 * What about a lamp cannot be changed without building a new Babylon light.
 *
 * Everything here either picks the constructor (`type`), picks which of the
 * two lighting paths draws the lamp (`clustered`), or picks the mesh list - and
 * moving a lamp in or out of the container, like changing `includedOnlyMeshes`,
 * dirties every affected material and recompiles shaders. Anything not in this
 * string is a scalar that can be poked in place, which is what makes dragging a
 * Range slider free rather than a recompile per frame.
 *
 * The scope is keyed by chunk and count rather than by mesh identity: a mesh
 * joining or leaving a room changes the count, and the ship's mesh set is
 * otherwise fixed for the life of a preview - so within one preview those two
 * agree with the list itself.
 */
function lightSignature(specs) {
  return specs.map((s) => `${s.light.id}:${s.r.type}:${s.r.clustered ? 1 : 0}` + `:${s.r.clustered ? "*" : s.chunk}:${s.scope.length}`).join("|");
}

function invalidatePreviewLight(light) {
  const clustered = light._clusteredContainer;
  (clustered || light)._markMeshesAsLightDirty?.();
  if (clustered) clustered._lightDataRenderId = -1;
}

/** Everything about a lamp that can be changed without a rebuild. */
function applyPreviewLight(light, spec) {
  const r = spec.r;
  const node = spec.light.node;
  node.computeWorldMatrix(true);
  if (light.position) light.position.copyFrom(node.getAbsolutePosition());
  if (light.direction) light.direction.copyFrom(emissionAxis(node));
  light.diffuse.copyFrom(Color3.FromArray(r.color));
  light.specular.copyFrom(Color3.FromArray(r.color));
  light.intensity = r.intensity;
  light.range = r.range;
  // Babylon's `angle` is the FULL cone, same as the authored field.
  if (r.type === "spot") light.angle = (r.angle * Math.PI) / 180;
  invalidatePreviewLight(light);
}

/**
 * Rebuild the authored runtime lamps over the live ship.
 *
 * @param lit - every dressed mesh. There is no unlit half any more: the lamps
 *   are the ship's only direct lighting, so they reach all of it.
 */
function buildPreviewLights(lit) {
  const scene = state.scene;
  const bounds = chunkBounds();
  const nearest = nearestChunkFor(bounds);
  const { specs, stats } = previewLightSpecs(lit);

  const lights = [];
  const clusteredLights = [];
  for (const spec of specs) {
    const r = spec.r;
    const node = spec.light.node;
    // A lamp node rides its owner placement, which may have been dragged since
    // the last frame, so the cached world matrix cannot be trusted here.
    node.computeWorldMatrix(true);
    const position = node.getAbsolutePosition().clone();
    const direction = emissionAxis(node);
    const angle = (r.angle * Math.PI) / 180;
    let light;
    if (r.type === "spot") {
      // `exponent` is only read by Babylon's STANDARD cone falloff, and the
      // preview materials ask for the glTF one, which shapes its cone from the
      // angle alone (`_lightAngleScale` / `_lightAngleOffset`). So 0 here is
      // not a choice being made - the value is never sampled.
      light = new SpotLight(`${ROOT_NAME}_${spec.light.id}`, position, direction, angle, 0, scene);
    } else if (r.type === "directional") {
      light = new DirectionalLight(`${ROOT_NAME}_${spec.light.id}`, direction, scene);
    } else {
      light = new PointLight(`${ROOT_NAME}_${spec.light.id}`, position, scene);
    }
    if (r.clustered) {
      // Babylon's clustered container supports its default falloff only. The
      // container is scene-global, matching Babylon Lite's runtime path -
      // constraining it to a mesh list breaks instanced meshes, because an
      // instance inherits its source mesh's light list.
      clusteredLights.push(light);
    } else {
      light.includedOnlyMeshes = lightScope(spec.scope);
      light.setEnabled(spec.chunk === nearest);
    }
    // Shadows are deliberately not generated. Babylon-Lite's clustered lights
    // carry no shadow map at all and its point lights have no cube generator,
    // which is why normalizeLight() refuses `castsShadows` for both - so a
    // preview that cast them would be showing something the game cannot.
    applyPreviewLight(light, spec);
    lights.push(light);
  }
  const clusteredContainer = clusteredLights.length ? new ClusteredLightContainer(`${ROOT_NAME}_clustered`, clusteredLights, scene) : null;

  return {
    lights,
    clusteredLights,
    clusteredContainer,
    lightSpecs: specs,
    lightSig: lightSignature(specs),
    lightStats: stats,
    chunkBounds: bounds,
    nearestChunk: nearest,
  };
}

function syncRegularLightVisibility() {
  if (!preview) return;
  const nearest = nearestChunkFor(preview.chunkBounds);
  if (nearest === preview.nearestChunk) return;
  preview.nearestChunk = nearest;
  for (let i = 0; i < preview.lights.length; i++) {
    const spec = preview.lightSpecs[i];
    if (!spec || spec.r.clustered) continue;
    preview.lights[i].setEnabled(spec.chunk === nearest);
  }
}

/**
 * Put an edit to the authored lamps on screen.
 *
 * Hung off the `lights` event, which every add, delete, move and inspector
 * edit already emits. Scalars are poked in place and only a change of shape
 * rebuilds, because a slider drag emits on every tick and disposing a light
 * dirties each material it touched - a recompile per frame, for a number.
 */
function refreshPreviewLights() {
  if (!preview) return;
  const { specs, stats } = previewLightSpecs(preview.meshes);
  preview.lightStats = stats;
  if (lightSignature(specs) !== preview.lightSig) {
    preview.clusteredContainer?.dispose();
    const clustered = new Set(preview.clusteredLights);
    for (const l of preview.lights) if (!clustered.has(l)) l.dispose();
    Object.assign(preview, buildPreviewLights(preview.meshes));
    return;
  }
  // Same lamps in the same order - the signature is built from that list, so
  // an unchanged signature guarantees the two line up index for index.
  preview.lightSpecs = specs;
  for (let i = 0; i < specs.length; i++) applyPreviewLight(preview.lights[i], specs[i]);
  syncRegularLightVisibility();
}

/**
 * Carry the lamps along when the element they ride is dragged.
 *
 * A light is a child of its owner placement, so moving the element moves the
 * lamp - but nothing emits `lights` for that. A lamp left behind while its own
 * fitting slides away would light the room from where the fitting used to be.
 */
function syncPreviewLightNodes() {
  if (!preview) return;
  syncRegularLightVisibility();
  for (let i = 0; i < preview.lights.length; i++) {
    const light = preview.lights[i];
    const node = preview.lightSpecs[i]?.light.node;
    if (!node || node.isDisposed()) continue;
    node.computeWorldMatrix(true);
    if (light.position) light.position.copyFrom(node.getAbsolutePosition());
    if (light.direction) light.direction.copyFrom(emissionAxis(node));
  }
}

// ------------------------------------------------------ which room you are in
//
// A scoped lamp is only enabled in the chunk the camera is standing in, so the
// preview needs a box per chunk. They are measured off the authored placements
// rather than off the chunk record, because a chunk has no volume of its own -
// it is a name that elements carry - and because measuring the live elements is
// what lets a dragged wall move the box with it on the very next frame.

/** World AABB per chunk, unioned over its placements' meshes. */
function chunkBounds() {
  const boxes = new Map();
  for (const placement of state.placements.values()) {
    if (placement.stage || !placement.chunk) continue;
    let box = boxes.get(placement.chunk);
    if (!box) {
      box = { id: placement.chunk, min: { x: Infinity, y: Infinity, z: Infinity }, max: { x: -Infinity, y: -Infinity, z: -Infinity } };
      boxes.set(placement.chunk, box);
    }
    for (const mesh of placement.node.getChildMeshes()) {
      mesh.computeWorldMatrix(true);
      const bounds = mesh.getBoundingInfo().boundingBox;
      box.min.x = Math.min(box.min.x, bounds.minimumWorld.x);
      box.min.y = Math.min(box.min.y, bounds.minimumWorld.y);
      box.min.z = Math.min(box.min.z, bounds.minimumWorld.z);
      box.max.x = Math.max(box.max.x, bounds.maximumWorld.x);
      box.max.y = Math.max(box.max.y, bounds.maximumWorld.y);
      box.max.z = Math.max(box.max.z, bounds.maximumWorld.z);
    }
  }
  // A chunk whose elements are all empty transforms would otherwise carry an
  // inverted box that every distance test reads as infinitely far away.
  return [...boxes.values()].filter((box) => Number.isFinite(box.min.x));
}
/**
 * Where "you" are, for the purpose of scoping lamps.
 *
 * Normally the camera. During a probe capture it is the capture point instead:
 * the six faces are a record of what the player sees standing THERE, so the
 * lamps that reach them have to be the ones the game would have switched on
 * there - not the ones lit for wherever the editor camera happens to be parked.
 *
 * The capture's own camera follows it to the same point. It never renders a
 * frame - the probe drives its six passes itself - but Babylon still picks
 * levels of detail against the active camera, so leaving it behind would let
 * where the user is standing decide which meshes a room's cubemap recorded.
 */
let captureViewpoint = null;

export function setCaptureViewpoint(position) {
  captureViewpoint = position;
  if (captureCamera && position) {
    captureCamera.position.copyFrom(position);
    captureCamera.computeWorldMatrix(true);
  }
  syncRegularLightVisibility();
}

function nearestChunkFor(bounds) {
  if (captureViewpoint) return nearestChunkTo(captureViewpoint, bounds);
  const camera = state.camera || state.scene.activeCamera;
  // `globalPosition` is a cache that a camera only fills in when its view
  // matrix is recomputed, so it still reads as the origin until one frame has
  // rendered - and loading a ship moves the camera to the saved view and then
  // builds this preview with nothing drawn in between. The origin sits inside
  // most chunk boxes at once, so trusting it silently scopes the preview
  // lights to whichever chunk wins the resulting tie. An unparented camera's
  // `position` is already world space, so take it; only a parented one needs
  // the matrix, and then it is worth forcing.
  let position = camera?.position;
  if (camera?.parent) {
    camera.computeWorldMatrix(true);
    position = camera.globalPosition || position;
  }
  if (!position) return state.activeChunk ?? bounds[0]?.id ?? null;
  return nearestChunkTo(position, bounds);
}

/** The chunk box a world point is in, or the nearest one to it. */
function nearestChunkTo(position, bounds) {
  let nearest = null;
  let bestDistance = Infinity;
  let bestVolume = Infinity;
  for (const bound of bounds) {
    const dx = position.x < bound.min.x ? bound.min.x - position.x : position.x > bound.max.x ? position.x - bound.max.x : 0;
    const dy = position.y < bound.min.y ? bound.min.y - position.y : position.y > bound.max.y ? position.y - bound.max.y : 0;
    const dz = position.z < bound.min.z ? bound.min.z - position.z : position.z > bound.max.z ? position.z - bound.max.z : 0;
    const distance = dx * dx + dy * dy + dz * dz;
    // Chunk boxes overlap - a corridor's box sits inside the sprawling box of
    // the room it leaves from - so standing in one usually means standing in
    // several, all at distance zero. Break that tie on the tighter box, the
    // same way a probe volume does: the small room you are in beats the big
    // one that merely encloses it. Without this the winner is whichever chunk
    // happens to come first.
    const volume = (bound.max.x - bound.min.x)
      * (bound.max.y - bound.min.y) * (bound.max.z - bound.min.z);
    if (distance < bestDistance
      || (distance === bestDistance && volume < bestVolume)) {
      bestDistance = distance;
      bestVolume = volume;
      nearest = bound.id;
    }
  }
  return nearest ?? state.activeChunk ?? null;
}

hooks.environmentProbeEntry = (id) => {
  if (!state.environmentProbes.has(id) || localEnvironmentBox?.metadata?.probe !== id) return null;
  return {
    id,
    name: id,
    type: "environment-probe",
    node: localEnvironmentBox,
    canRotate: false,
  };
};

/**
 * Every mesh whose element overlaps a probe box.
 *
 * This is the render list a capture uses, and it is deliberately the ELEMENT's
 * bounds that decide - the same rule that picks which probe a mesh wears - so a
 * wall is either wholly in the room's capture or wholly out of it. Half a wall
 * appearing in a reflection is worse than the wall being missing.
 *
 * A room is captured from inside, so the geometry that matters is the geometry
 * around the camera. Anything beyond the box is another room's business and is
 * left out: it would only be visible through a doorway, and the doorway is what
 * the portal renderer is for.
 *
 * Elements whose behaviours keep them out of probes are dropped here rather
 * than in the capture, and that placement is the point: this one list is both
 * the render list AND what the digest is computed over, so an excluded element
 * neither appears in the cubemap nor marks it stale when it is moved.
 */
export function meshesInProbeBox(boxPosition, boxSize) {
  if (!preview) return [];
  const min = new Vector3(boxPosition[0] - boxSize[0] * 0.5, boxPosition[1] - boxSize[1] * 0.5, boxPosition[2] - boxSize[2] * 0.5);
  const max = new Vector3(boxPosition[0] + boxSize[0] * 0.5, boxPosition[1] + boxSize[1] * 0.5, boxPosition[2] + boxSize[2] * 0.5);
  const out = [];
  for (const group of preview.elementMeshes.values()) {
    if (probeExcluded(group)) continue;
    const bounds = elementWorldBounds(group);
    if (!bounds) continue;
    if (bounds.max.x < min.x || bounds.min.x > max.x) continue;
    if (bounds.max.y < min.y || bounds.min.y > max.y) continue;
    if (bounds.max.z < min.z || bounds.min.z > max.z) continue;
    out.push(...group);
  }
  return out;
}

/**
 * Whether an element's authored behaviours keep it out of every probe.
 *
 * Behaviours hang off an element's NAME, not off its id - naming six crates
 * "crate" is how one entry comes to govern all six - so the chain is mesh to
 * owning placement to that placement's name. An element with no name carries
 * no behaviours and is therefore never excluded.
 *
 * Every mesh in a group belongs to the same placement (see elementOf), so the
 * first one answers for all of them.
 */
function probeExcluded(group) {
  const id = ownerIdOf(group[0], true);
  const name = id ? state.placements.get(id)?.name : "";
  return !!name && isProbeExcludedNode(name);
}

/**
 * The meshes that will actually be drawn for a set of authored ones.
 *
 * The authored list is what everything reasons about - it is stable across
 * captures, it is what the digest describes and it is what the user drags - but
 * under the preview an instance is invisible and a stand-in is drawn in its
 * place (see dressMeshes). A render list built from the authored meshes would
 * photograph an empty room.
 */
export function renderListFor(meshes) {
  return meshes.map((mesh) => preview?.previewOf.get(mesh) || mesh);
}

/**
 * The material a mesh came in with, whatever it is wearing now.
 *
 * Only a ship mesh that is NOT an instance is dressed in place, and only those
 * are in `sourceOf` - but that is exactly the case where asking a mesh for its
 * material would answer with a clone named after a probe, and a probe only
 * exists once a capture has been taken. Anything describing the ship ITSELF -
 * the capture digest above all - has to look through the dressing, or the first
 * capture would change the state the second one is compared against and no
 * probe would ever read as up to date.
 */
export function authoredMaterialOf(mesh) {
  return preview?.sourceOf.get(mesh) || mesh.material || null;
}

/**
 * Run a capture with the ship lit exactly as the game lights it, with its own
 * reflections switched off, and with the editor kept out of the picture.
 *
 * Four things, and every one of them is a way a capture could otherwise come
 * out wrong.
 *
 * The lighting has to be the runtime's, or the .env would record the editor's
 * four-light rig - so the mode is entered if it is not already up.
 *
 * The reflections have to be OFF, or a capture would see the cubemaps left by
 * the previous capture and every generation would differ from the last. Nulling
 * them makes a probe a single-bounce record of direct light and emissive
 * surfaces: well defined, and identical whether the folder was empty or full.
 *
 * **Image processing has to be out of the materials.** A cubemap is a record of
 * radiance, not of a picture: the runtime exposes and tone-maps it at display
 * time, and doing it twice is both wrong and unrecoverable, since tone mapping
 * clamps to 1 and a room lit above that loses its highlights for good. Babylon
 * offers `linearSpace` on the probe for this, and it is not enough on its own -
 * it flips `applyByPostProcess` when the target is BOUND, by which point every
 * material has already compiled with image processing baked into its shader.
 * Measured on the real ship: flipped at bind time the six faces peak at exactly
 * 1.0, flipped before the shaders are asked for they peak at 14.6. So the flag
 * goes up for the whole session, ahead of the readiness wait.
 *
 * The camera has to be the capture's own. The editor's drives level of
 * detail and carries its own layer mask, so which meshes a probe recorded would
 * depend on where the user happened to be parked. A fresh one, moved to each
 * capture point, is the only way two runs from the same ship agree.
 *
 * All of that is scene-wide state, which is why callers hold the busy lock over
 * this: the overlay covers the viewport and makes every panel inert, so the
 * frames the editor keeps drawing meanwhile are seen by nobody. The render loop
 * is deliberately NOT stopped. Babylon compiles a material's shader as part of
 * rendering, and a capture changes the light set and the image-processing
 * define - so a stopped loop means those effects never rebuild and the capture
 * waits for a readiness that can no longer arrive.
 */
export async function withRuntimeCapture(fn) {
  const wasRuntime = !!state.runtime;
  if (!wasRuntime) await setRuntimePreview(true);
  const scene = state.scene;
  const image = scene.imageProcessingConfiguration;
  const suspended = [];
  const wasCamera = scene.activeCamera;
  const wasByPostProcess = image.applyByPostProcess;
  try {
    if (preview) {
      for (const mat of preview.materials) {
        if (!("reflectionTexture" in mat) || !mat.reflectionTexture) continue;
        suspended.push([mat, mat.reflectionTexture]);
        mat.reflectionTexture = null;
      }
    }
    image.applyByPostProcess = true;
    captureCamera = new FreeCamera(CAPTURE_CAMERA, wasCamera?.position?.clone() ?? Vector3.Zero(), scene);
    captureCamera.minZ = wasCamera?.minZ ?? 0.05;
    captureCamera.maxZ = wasCamera?.maxZ ?? 1000;
    scene.activeCamera = captureCamera;
    return await fn(preview);
  } finally {
    scene.activeCamera = wasCamera;
    captureCamera?.dispose();
    captureCamera = null;
    image.applyByPostProcess = wasByPostProcess;
    for (const [mat, texture] of suspended) {
      if (!mat.isFrozen && !texture.isDisposed?.()) mat.reflectionTexture = texture;
    }
    if (!wasRuntime) await setRuntimePreview(false);
    // Putting the flag back changes a shader define on every material in the
    // scene, and a mesh whose effect is recompiling draws nothing at all. Take
    // that cost here, under the lock, rather than as a hole in the first frame
    // the user gets back.
    //
    // Under a deadline, because "ready" is not guaranteed to arrive: a material
    // whose shader fails to link reports not-ready for ever, and this await is
    // the LAST thing a capture does - a hang here reads as the final probe
    // never finishing and leaves the editor locked behind its own overlay.
    await withDeadline(scene.whenReadyAsync(), READY_TIMEOUT_MS, "the scene rebuilding after the capture")
      .catch((err) => { console.warn(`[aquanova] ${err.message}`); });
  }
}

/**
 * The meshes the runtime view dresses: every authored element in the ship.
 *
 * A placement's node carries more than its kit: a lamp's gizmo and its picking
 * stub both ride the element they belong to, and neither is part of the ship
 * the game ships. `placementRoot` is what kit.js stamps on the meshes it
 * instantiates, so it is the only honest test of "this is the model".
 *
 * Stage geometry is left out for the same reason - it is the editor's own floor
 * grid and blockout helpers - and so is the preview's own output, or a second
 * pass would dress the stand-ins it made on the first.
 */
function shipMeshes() {
  const out = [];
  for (const placement of state.placements.values()) {
    if (placement.stage) continue;
    for (const mesh of placement.node.getChildMeshes()) {
      if (mesh.metadata?.placementRoot !== placement.node) continue;
      if (mesh.metadata?.runtimePreview) continue;
      if (mesh.material) out.push(mesh);
    }
  }
  return out;
}

/**
 * Put the ship back the way the editor had it.
 *
 * The order is undressMeshes' business: authored meshes visible and on their
 * own materials before anything the preview made is freed.
 */
function disposePreview() {
  if (!preview) return;
  state.scene.onBeforeRenderObservable.remove(preview.syncer);
  preview.clusteredContainer?.dispose();
  const clustered = new Set(preview.clusteredLights);
  for (const l of preview.lights) if (!clustered.has(l)) l.dispose();
  undressMeshes(preview);
  for (const t of preview.localEnvironments.values()) t.dispose();
  preview = null;
}

/**
 * Re-dress the ship after its element set changes.
 *
 * Adding, deleting or re-kitting an element brings in meshes the preview has
 * never seen, and they would otherwise sit on their authored materials -
 * reflecting the editor's global HDRI instead of their room. The whole dressing
 * is rebuilt rather than patched: the prototypes and clones are keyed by probe
 * and source, so an unchanged ship rebuilds exactly the handful it already had,
 * and a patch would have to reimplement every rule in dressMeshes to decide
 * what to keep.
 */
function redressPreview() {
  if (!preview) return;
  preview.clusteredContainer?.dispose();
  const clustered = new Set(preview.clusteredLights);
  for (const l of preview.lights) if (!clustered.has(l)) l.dispose();
  undressMeshes(preview);
  const dressed = dressMeshes(shipMeshes(), preview.localEnvironments);
  Object.assign(preview, dressed, buildPreviewLights(dressed.meshes));
  applyRuntimeReflectionSettings();
}

// An element added, removed or re-kitted while the runtime view is up has
// meshes that have never been dressed. See redressPreview.
on("placements", () => redressPreview());

/**
 * Switch the game's lighting on over the live ship, or put the editor's back.
 *
 * The probe cubemaps are re-read on every entry rather than cached across
 * switches: they are files that `local-environments.js` rewrites whenever the
 * ship changes, and a preview showing a stale capture of a room that has since
 * been rebuilt is worse than showing none.
 */
export async function setRuntimePreview(on) {
  const wanted = !!on;
  if (wanted === !!state.runtime) return state.runtime;

  if (!wanted) {
    disposePreview();
    state.runtime = false;
    state.scene.environmentTexture = editorEnvironmentTexture;
    editorEnvironmentTexture = null;
    // The rig, and the Env/Exposure pair that goes with it, follow the flag.
    syncLightingMode();
    applyVisibility();
    emit("modes");
    return false;
  }

  resetLocalEnvironmentAssets();
  const localEnvironments = await loadLocalEnvironments();
  const dressed = dressMeshes(shipMeshes(), localEnvironments);
  preview = {
    ...dressed,
    ...buildPreviewLights(dressed.meshes),
    syncer: null,
  };
  applyRuntimeReflectionSettings();
  // Elements are draggable while the mode is up, so a probe assignment and a
  // lamp position are both only true for the frame they were computed in.
  preview.syncer = state.scene.onBeforeRenderObservable.add(() => {
    syncElementEnvironments();
    syncPreviewLightNodes();
  });

  // The editor's rig is four analytic lights the game does not have, and its
  // global HDRI is an environment no room of the ship actually has. Both would
  // sit on top of the authored lighting and hide exactly what is being
  // inspected. The flag moves first: the rig, the slider pair and the status
  // line all read it.
  editorEnvironmentTexture = state.scene.environmentTexture;
  state.scene.environmentTexture = null;
  state.runtime = true;
  syncLightingMode();
  applyVisibility();
  emit("modes");
  if (localEnvironmentBox?.metadata?.probe && localEnvironmentBox.isEnabled()) {
    await showEnvironmentProbe(localEnvironmentBox.metadata.probe, true, null, localEnvironmentBoxEnvironmentVisible);
  }
  return true;
}