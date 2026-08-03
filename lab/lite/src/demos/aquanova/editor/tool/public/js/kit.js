// Loading of MegaKit modules, with materials and textures shared across the
// whole catalogue and geometry re-used through hardware instances.
//
// Every module .gltf in the kit references the same ~20 root-level textures and
// names its materials identically (MI_Trim_01, MI_Trim_02, M_Light, ...), so
// loading 300 modules naively would create 300 copies of the same atlas. The
// registry below keeps the first material seen under a given name and throws
// the duplicates away, textures included.

const { SceneLoader, TransformNode, Vector3, Quaternion, Matrix, Color3, Material } = BABYLON;

import { noteAuthoredEmissive, applyViewportMode } from "./editor.js";

export const materialRegistry = new Map();

let catalogue = null;
let kitMaterials = null;
const protoCache = new Map();
const protoPending = new Map();

export async function loadCatalogue() {
  if (!catalogue) {
    const [cat, mats] = await Promise.all([
      fetch("/api/modules").then((r) => r.json()),
      fetch("/data/kit_materials.json").then((r) => r.json()),
    ]);
    catalogue = cat;
    kitMaterials = mats;
    catalogue.byId = new Map();
    for (const c of catalogue.categories) {
      for (const m of c.modules) catalogue.byId.set(m.id, m);
    }
  }
  return catalogue;
}

export function getCatalogue() { return catalogue; }
export function getModule(id) { return catalogue?.byId.get(id) || null; }
export function getKitMaterials() { return kitMaterials; }

function applyKitValues(mat) {
  // Modules are single-sided, so a wall seen from behind vanishes. That is
  // correct in game but useless while building, where you orbit freely.
  mat.backFaceCulling = false;
  mat.twoSidedLighting = true;

  const em = kitMaterials?.emissive?.[mat.name];
  if (em) {
    const scale = kitMaterials.editorViewport?.emissiveScale ?? 1;
    mat.emissiveColor = new Color3(em.color[0], em.color[1], em.color[2]);
    mat.emissiveIntensity = em.intensity * scale;
  }
  // The kit values *are* the authored state, and they land after the material
  // was constructed - so re-note them, then apply whatever viewport mode is on.
  noteAuthoredEmissive(mat, true);
  applyViewportMode(mat);
}

/**
 * Put back the transparency the .gltf export flattened away.
 *
 * Quaternius' own shaders make some of these materials see-through - the
 * Godot glass is `blend_mix` with `ALPHA = mix(0.05, 0.5, perlin)` - but glTF
 * has no way to say "alpha driven by scrolling noise", so the export wrote
 * them as OPAQUE and the panes came out solid. `kit_materials.json` carries
 * the authored value the same way it already carries emissive.
 *
 * Applied *before* the dedupe key is taken, deliberately: the kit authors
 * `M_Glass` as BLEND in two files and OPAQUE in twelve, and once the override
 * has settled the question both are the same material again and share one
 * copy, rather than being kept apart over a difference that no longer exists.
 */
export function applyKitTransparency(mat) {
  const t = kitMaterials?.transparency?.[mat.name];
  if (!t) return;
  if (typeof t.alpha === "number") {
    mat.alpha = t.alpha;
    mat.transparencyMode = t.alpha < 1
      ? Material.MATERIAL_ALPHABLEND : Material.MATERIAL_OPAQUE;
  }
  if (t.tint && "albedoColor" in mat) mat.albedoColor = new Color3(...t.tint);
  if (typeof t.roughness === "number" && "roughness" in mat) mat.roughness = t.roughness;
  // An index of refraction of 1 is glass that does not catch the light: the
  // dielectric F0 is ((n-1)/(n+1))^2, so n = 1 makes it zero and the pane loses
  // its white sheen, leaving the tint to be read on its own. Guarded because
  // only a PBR material has the property at all.
  if (typeof t.ior === "number" && "indexOfRefraction" in mat) mat.indexOfRefraction = t.ior;
}

/**
 * What makes two materials the same material.
 *
 * Not the name on its own. The kit names materials identically across modules
 * *and* authors some of them two different ways: `M_Glass` is `BLEND` with an
 * alpha of 0 in two files and `OPAQUE` in twelve, and `M_Decal_White` is
 * `MASK` in thirty-one and `OPAQUE` in twenty-six. Keyed by name alone,
 * whichever module loaded first decided how glass looked everywhere - and the
 * palette keeps its own cache, filled in a different order, so a window could
 * be see-through on its tile and solid in the ship at the same time.
 *
 * Transparency is the only thing they differ in, and it is the one thing you
 * cannot share, so it goes in the key. Everything else about a material is the
 * texture set, which is what the sharing is for.
 */
export function materialKey(mat) {
  const mode = mat.transparencyMode === null || mat.transparencyMode === undefined
    ? "opaque" : mat.transparencyMode;
  return `${mat.name}|${mode}|${mat.alpha}`;
}

// Swap every material on `meshes` for the shared instance of the same name.
function dedupeMaterials(meshes) {
  for (const mesh of meshes) {
    const mat = mesh.material;
    if (!mat) continue;
    applyKitTransparency(mat);
    const key = materialKey(mat);
    const shared = materialRegistry.get(key);
    if (shared && shared !== mat) {
      mesh.material = shared;
      mat.dispose(false, true);          // true: drop this copy's textures too
    } else if (!shared) {
      materialRegistry.set(key, mat);
      applyKitValues(mat);
    }
  }
}

/**
 * Load a module once and keep it as a hidden prototype. Returns
 * { meshes: [{ mesh, position, rotationQuaternion, scaling }] } describing the
 * module's parts relative to its own origin.
 */
export async function getProto(moduleId) {
  if (protoCache.has(moduleId)) return protoCache.get(moduleId);
  if (protoPending.has(moduleId)) return protoPending.get(moduleId);

  const mod = getModule(moduleId);
  if (!mod) throw new Error(`unknown module: ${moduleId}`);

  const job = (async () => {
    const dir = mod.url.slice(0, mod.url.lastIndexOf("/") + 1);
    const file = mod.url.slice(mod.url.lastIndexOf("/") + 1);
    const container = await SceneLoader.LoadAssetContainerAsync(dir, file, window.__scene);

    const meshes = container.meshes.filter((m) => m.getTotalVertices() > 0);
    dedupeMaterials(meshes);
    container.addAllToScene();

    const parts = [];
    for (const mesh of meshes) {
      mesh.computeWorldMatrix(true);
      const pos = new Vector3();
      const rot = new Quaternion();
      const scl = new Vector3();
      mesh.getWorldMatrix().decompose(scl, rot, pos);
      mesh.setEnabled(false);
      mesh.isPickable = false;
      parts.push({ mesh, position: pos, rotationQuaternion: rot, scaling: scl });
    }

    // The container's own root/empty nodes are not needed once the parts carry
    // their baked world transforms.
    for (const node of container.transformNodes) node.setEnabled(false);

    const proto = { id: moduleId, parts, container };
    protoCache.set(moduleId, proto);
    protoPending.delete(moduleId);
    return proto;
  })();

  protoPending.set(moduleId, job);
  return job;
}

/** Create a placement node holding instances of every part of `moduleId`. */
export async function instantiate(moduleId, nodeName) {
  const proto = await getProto(moduleId);
  const root = new TransformNode(nodeName, window.__scene);
  root.rotationQuaternion = Quaternion.Identity();

  for (let i = 0; i < proto.parts.length; i++) {
    const part = proto.parts[i];
    const inst = part.mesh.createInstance(`${nodeName}#${i}`);
    inst.parent = root;
    inst.position.copyFrom(part.position);
    inst.rotationQuaternion = part.rotationQuaternion.clone();
    inst.scaling.copyFrom(part.scaling);
    inst.isPickable = true;
    inst.metadata = { placementRoot: root };
  }
  return root;
}

/** Local-space bounding box of a module, in metres. */
export async function moduleBounds(moduleId) {
  const proto = await getProto(moduleId);
  let min = null, max = null;
  for (const part of proto.parts) {
    const bi = part.mesh.getBoundingInfo();
    const m = Matrix.Compose(part.scaling, part.rotationQuaternion, part.position);
    for (const v of bi.boundingBox.vectors) {
      const p = Vector3.TransformCoordinates(v, m);
      min = min ? Vector3.Minimize(min, p) : p.clone();
      max = max ? Vector3.Maximize(max, p) : p.clone();
    }
  }
  return { min: min || Vector3.Zero(), max: max || Vector3.Zero() };
}
