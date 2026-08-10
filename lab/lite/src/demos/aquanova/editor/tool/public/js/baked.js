// Seeing the Blender bake, without leaving the editor.
//
// The lightmaps cannot be shown on the ship the editor is holding, for two
// reasons that are both structural rather than incidental:
//
//   * the editor's meshes have one UV set, the kit's own. The lightmap atlas is
//     a *second* set, and it only exists after Blender has unwrapped it - which
//     happens on the way into `ship_baked.glb` and nowhere else.
//   * kit materials are deduplicated across the whole catalogue (see kit.js), so
//     one `MI_Trim_01` serves every room in the ship. A lightmap is per chunk.
//     There is no way to hang two different maps on one material.
//
// So the preview is the baked ship itself, loaded beside the authored one and
// shown instead of it. That also makes it an honest preview: what is on screen
// is the file the runtime will load, not the editor's idea of it.

const { ClusteredLightContainer, Color3, DirectionalLight, Matrix, PointLight, Quaternion, SceneLoader, SpotLight, Texture, TransformNode, Vector3, VertexBuffer } = BABYLON;

import { state, emit, on, hooks, syncLightingMode, applyVisibility } from "./editor.js";

const ROOT_NAME = "BAKED_PREVIEW";

let preview = null;
const authoredRoughness = new WeakMap();

/** Apply the editor-only reflection experiment to the currently loaded bake. */
function applyBakedReflectionSettings() {
  if (!preview) return;
  for (const mat of preview.materials) {
    mat.environmentIntensity = state.envIntensity;
    mat.enableSpecularAntiAliasing = state.bakedSpecularAA;
    const base = authoredRoughness.get(mat);
    if (typeof base === "number") {
      // Babylon clamps the final roughness in the shader; keeping the scalar above
      // 1 lets it boost low values sampled from the ORM texture.
      mat.roughness = base * state.bakedRoughnessFactor;
    }
  }
}

/** Keep the environment on UV2-less preview materials independent of baked surfaces. */
function applyDynamicEnvironment() {
  if (!preview) return;
  const materials = new Set(preview.dynamic.map((mesh) => mesh.material).filter(Boolean));
  for (const mat of materials) mat.environmentIntensity = state.dynamicEnvIntensity;
}

/** The loaded preview, for tests and the console. Null when it is off. */
export function bakedPreview() { return preview; }

// Every add, delete, move and inspector edit of a lamp already emits this, and
// refreshPreviewLights is a no-op while the preview is down - so this is the
// whole subscription. See refreshPreviewLights for why it is not a rebuild.
on("lights", () => refreshPreviewLights());
on("reflection", () => applyBakedReflectionSettings());
on("environment", () => {
  applyBakedReflectionSettings();
  applyDynamicEnvironment();
});
hooks.setDynamicEnvironment = applyDynamicEnvironment;

/** What the last bake left behind, or a refusal saying there was not one. */
export async function loadBakedIndex() {
  const r = await fetch("/lightmaps/lightmaps.json", { cache: "no-store" });
  if (!r.ok) throw new Error("no bake to show yet — press Bake first");
  const index = await r.json();
  if (!index?.chunks || !Object.keys(index.chunks).length) {
    throw new Error("the bake wrote no lightmaps");
  }
  return index;
}

/** The chunk a baked mesh belongs to, which is what picks its lightmap. */
function chunkOf(node) {
  for (let n = node; n; n = n.parent) {
    const name = String(n.name || "");
    if (name.startsWith("CHUNK_")) return name.slice("CHUNK_".length);
  }
  return null;
}

function lightmapFor(index, chunk, cache) {
  if (cache.has(chunk)) return cache.get(chunk);
  const entry = index.chunks[chunk];
  let tex = null;
  if (entry?.png) {
    // invertY false, matching what the glTF loader does for the ship's own
    // maps: TEXCOORD_1 arrives in glTF's own convention and the atlas was
    // written to match it. Flipping one and not the other puts the light on
    // the ceiling of the room below.
    tex = new Texture(`/lightmaps/${entry.png}`, state.scene, false, false);
    tex.name = `Lightmap_${chunk}`;
    tex.coordinatesIndex = index.uv ?? 1;
    // The map was divided by its own peak so it would fit in 8 bits; `level`
    // is the divisor, put back. See write_png() in bake_lightmaps.py.
    tex.level = Number(entry.level) || 1;
    tex.gammaSpace = index.gamma !== false;
    tex.wrapU = Texture.CLAMP_ADDRESSMODE;
    tex.wrapV = Texture.CLAMP_ADDRESSMODE;
  }
  cache.set(chunk, tex);
  return tex;
}

/**
 * Hang each chunk's map on its own copy of every material it uses.
 *
 * Blender shares a material across chunks exactly as the kit does, so the copy
 * is not optional: without it the last chunk processed would win and every
 * other room would wear its lightmap.
 */
function dressMeshes(meshes, index) {
  const textures = new Map();
  const clones = new Map();
  const owned = [];
  const dynamic = [];
  let lit = 0;
  for (const mesh of meshes) {
    const mat = mesh.material;
    if (!mat) continue;
    // A mesh the bake left out of the atlas has no TEXCOORD_1, and sampling a
    // vertex buffer that is not there reads zero - which under the multiply
    // below paints the mesh black. The props the bake left out are exactly this
    // case: `chunk_meshes` in bake_lightmaps.py drops anything the runtime
    // redraws - the meshes it moves (`dynamic`) and the ones it melts
    // (`liquefiable`, plus everything they are `linked` to) - because a crate
    // that will not stay put must not have its own shadow painted into a
    // static map. So they keep their authored material
    // and are handed to buildPreviewLights instead, which is what the runtime
    // does with them too. Keying off the missing UV2 rather than off the
    // behaviours means this follows whatever the bake decided, for free.
    if (!mesh.isVerticesDataPresent(VertexBuffer.UV2Kind)) {
      dynamic.push(mesh);
      continue;
    }
    if (!("lightmapTexture" in mat)) continue;
    const chunk = chunkOf(mesh);
    const tex = chunk ? lightmapFor(index, chunk, textures) : null;
    if (!tex) continue;
    const key = `${chunk}|${mat.uniqueId}`;
    let copy = clones.get(key);
    if (!copy) {
      const shared = new Set(mat.getActiveTextures());
      copy = mat.clone(`${mat.name}__${chunk}`);
      // `Material.clone` deep-copies every texture slot - Babylon's CopySource
      // runs `sourceProperty.clone()` on each one - so the copy owns a private
      // wrapper for each of the ship's base colour, normal and ORM maps. Those
      // wrappers belong to nobody else: they are not in the glb's
      // AssetContainer, so `container.dispose()` never sees them, and the
      // `dispose(_, forceDisposeTextures = false)` below leaves them alone.
      //
      // They have to be collected and freed or the preview leaks 37 of them
      // per switch, and the leak is not merely wasted memory: a clone shares
      // the glb's InternalTexture through the engine's url cache, so while one
      // survives that entry's reference count never reaches zero. The glTF
      // loader names those entries deterministically ("...ship_baked.glb#image4"),
      // so the next load matches the stale entry and gets the *previous*
      // bake's pixels - and since an authoring change shifts image indices,
      // the re-baked ship comes back wearing the wrong maps.
      //
      // Diffing against the source is what makes this safe: a texture the
      // clone shares rather than owns - `CopySource` passes render targets
      // straight through - is in both sets and is left alone.
      for (const t of copy.getActiveTextures()) if (!shared.has(t)) owned.push(t);
      copy.lightmapTexture = tex;
      // Unlit plus shadowmap is the whole compositing contract, and neither
      // half works without the other.
      //
      // Babylon's PBR shader adds the lightmap by default:
      //     finalColor.rgb += lightmapColor.rgb
      // which never touches the albedo. Baked mode also turns the editor's
      // four lights off and drops the environment, so `finalColor` going into
      // that line is zero and the screen shows the raw irradiance map and
      // nothing else - no red floor bands, no blue wall, and decals blurred to
      // atlas resolution because the base colour texture is not in the picture
      // at all. That is what this pair fixes.
      //
      // Keep the normal PBR path enabled so the scene environment can provide
      // specular reflections on baked metallic surfaces. `useLightmapAsShadowmap`
      // still changes the lightmap from an additive term into a multiplier,
      // preserving the baked diffuse contribution while allowing environment
      // lighting to reach the material.
      copy.unlit = false;
      // Baked materials keep their IBL/reflection path, but analytic runtime
      // lights belong exclusively to meshes omitted from the bake.
      copy.disableLighting = true;
      copy.useLightmapAsShadowmap = true;
      authoredRoughness.set(copy, typeof copy.roughness === "number" ? copy.roughness : 1);
      // Back-face culling is deliberately left as the glb authored it. An
      // earlier version forced it off to make a door visible, which only hid
      // the real fault: the door leaf was a one-sided panel mounted facing the
      // corridor, so the room saw its back and Cycles - correctly - baked that
      // side black. Culling is what makes that kind of mistake visible.
      // Keeps the viewport's own unlit toggle off these: see applyViewportMode.
      copy.metadata = { ...(copy.metadata || {}), bakedPreview: true };
      clones.set(key, copy);
    }
    mesh.material = copy;
    lit++;
  }
  return { textures: [...textures.values()].filter(Boolean),
    materials: [...clones.values()], owned, dynamic, lit, skipped: dynamic.length };
}

// ------------------------------------------------------- the runtime lamps
//
// Baked mode is the only place the editor can show a dynamic prop lit the way
// the game will light it, because the two halves only meet here: the props are
// exactly the meshes the bake left out, and the lamps are exactly the ones the
// bake did NOT replace with an atlas.
//
// The records are read off `state.lights` - the editor's live record - and NOT
// off the `LIGHT_*` extras in the loaded glb, which is what they used to be.
// The glb carries every lamp as the bake saw it, so a preview built from it is
// frozen at bake time: Range, Intensity, colour and cone could all be edited in
// the inspector with nothing happening on screen until the next bake. That is
// the wrong way round. The runtime half of a light is by definition the half a
// bake does not consume - it is applied by the engine, over the atlas, every
// frame - so it is stale the moment it is read from a file, and re-baking to
// see a number that no bake reads is a long way to go for nothing.
//
// The bake half is still the file's to state, because that IS what the file is:
// a record of a render that already happened. `bakedDrift()` is what reports
// when the ship has moved out from under it.
//
// This costs no conversion. The authored lights live in the editor's frame,
// while the glb comes in under the loader's `__root__` and its handedness flip
// - but `syncStandIns` puts every baked node back on its authored element, so
// the props these lamps light are rendered in the editor's frame too. The two
// agree by construction, and the e2e `standOffset` check is what holds them to
// it.
//
// This is `lab/lite/src/demos/aquanova/lights.ts` rebuilt on Babylon.js, and
// the two are kept deliberately parallel - same -Y emission axis, same
// clustered-vs-scoped rule. Where they differ it is because the engines differ,
// and each such place is called out below.

/** How many lamps one dynamic prop's shader is compiled to take. */
const MAX_PREVIEW_LIGHTS = 4;

/** Column-major world matrix -> the world direction of the node's local -Y.
 *  See the EMISSION AXIS note in lights.js for why it is -Y and not -Z. */
function emissionAxis(node) {
  return Vector3.TransformNormal(new Vector3(0, -1, 0), node.getWorldMatrix()).normalize();
}

/**
 * Which authored lamps the preview should build, and what each one lights.
 *
 * Read off `state.lights` - the editor's live record - and NOT off the
 * `LIGHT_*` extras in `ship_baked.glb`. The glb carries the lamps *as the bake
 * saw them*, so a preview built from it is frozen at bake time: turning Range
 * up in the inspector moved the authored record and changed nothing on screen,
 * and the only way to see an edit was to re-bake. Since the runtime half is
 * precisely the half a bake does not consume, that made the one parameter set
 * the preview exists to show the one it could not.
 *
 * The bake half still comes from the file, because that IS what the file is:
 * a record of a render that already happened.
 */
function previewLightSpecs(dynamic) {
  const byChunk = new Map();
  for (const mesh of dynamic) {
    const chunk = chunkOf(mesh) ?? "";
    const list = byChunk.get(chunk);
    if (list) list.push(mesh);
    else byChunk.set(chunk, [mesh]);
  }

  const specs = [];
  const stats = { point: 0, spot: 0, directional: 0, bakeOnly: 0, idle: 0 };
  for (const light of state.lights.values()) {
    const r = light.runtime;
    // "none" is a lamp that only ever bakes - a bounce fill lifting a dark
    // corner. Its whole contribution is already in the atlas.
    if (!r || r.type === "none") { stats.bakeOnly++; continue; }

    // Scoping. A clustered lamp is scene-global by agreement - the cluster is
    // cheap enough that a prop near a doorway picking up the next room's lamps
    // is a feature. A UBO lamp is scarce, so it is held to its own chunk. The
    // chunk comes from the owner placement, so moving an element to another
    // room re-scopes its lamps without a reload.
    //
    // A lamp with nothing to light is not built at all. Babylon-Lite reads an
    // EMPTY `includedOnlyMeshIds` as "lights nothing"; Babylon.js reads an empty
    // `includedOnlyMeshes` as "no filter", i.e. lights EVERYTHING. Skipping is
    // the only spelling that means the same thing in both, and a room with no
    // dynamic props genuinely has nothing here for its lamps to do.
    const chunk = state.placements.get(light.owner)?.chunk ?? "";
    const scope = r.clustered ? dynamic : (byChunk.get(chunk) ?? []);
    if (!scope.length) { stats.idle++; continue; }
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
 * Everything here either picks the constructor (`type`), picks the falloff
 * curve (`clustered`), or picks the mesh list - and `includedOnlyMeshes` and
 * `falloffType` both dirty every affected material, which recompiles shaders.
 * Anything not in this string is a scalar that can be poked in place, which is
 * what makes dragging a Range slider free rather than a recompile per frame.
 *
 * The scope is keyed by chunk and count rather than by mesh identity: the
 * dynamic set is fixed for the life of a preview (it comes from the glb), so
 * within one preview those two agree with the list itself.
 */
function lightSignature(specs) {
  return specs.map((s) => `${s.light.id}:${s.r.type}:${s.r.clustered ? 1 : 0}`
    + `:${s.r.clustered ? "*" : s.chunk}:${s.scope.length}`).join("|");
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
 * Rebuild the authored runtime lamps over the baked ship's dynamic props.
 *
 * @param dynamic - the meshes the bake left out, which is precisely the set
 *   these lamps exist for. Baked material clones disable analytic lighting
 *   while retaining their lightmap and environment response.
 */
function buildPreviewLights(dynamic, container) {
  const scene = state.scene;
  const bounds = chunkBounds(container);
  const nearest = nearestChunkFor(bounds);
  const { specs, stats } = previewLightSpecs(dynamic);

  const lights = [];
  const clusteredLights = [];
  for (const spec of specs) {
    const r = spec.r;
    const node = spec.light.node;
    // The container was parented under a fresh root a moment ago and nothing
    // has rendered since, so the cached world matrices are stale.
    node.computeWorldMatrix(true);
    const position = node.getAbsolutePosition().clone();
    const direction = emissionAxis(node);
    const angle = (r.angle * Math.PI) / 180;
    let light;
    if (r.type === "spot") {
      // `exponent` is only read by the STANDARD falloff - both falloffs used
      // below are cone-shaped by their own maths - so 0 is not a choice.
      light = new SpotLight(`${ROOT_NAME}_${spec.light.id}`, position, direction, angle, 0, scene);
    } else if (r.type === "directional") {
      light = new DirectionalLight(`${ROOT_NAME}_${spec.light.id}`, direction, scene);
    } else {
      light = new PointLight(`${ROOT_NAME}_${spec.light.id}`, position, scene);
    }
    if (r.clustered) {
      // Babylon's clustered container supports its default falloff only. The
      // container is scene-global, matching Babylon Lite's runtime path.
      // Baked material clones opt out through disableLighting; constraining the
      // container itself breaks instanced dynamic meshes because instances
      // inherit their source mesh's light list.
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
  const clusteredContainer = clusteredLights.length
    ? new ClusteredLightContainer(`${ROOT_NAME}_clustered`, clusteredLights, scene)
    : null;

  // Babylon compiles a fixed number of light slots into each material and
  // silently drops the rest, and the default of 4 is well under what a corridor
  // ship puts in one room. Only the props' own materials are touched: the
  // lightmapped clones disable analytic lighting and never enter this loop.
  //
  // Which also makes the Env slider the props' slider in this mode, and only
  // theirs - the ship's own materials cannot see the HDRI at all. That matters
  // more than it sounds: the kit's crates and consoles are metal, and a metal
  // has no diffuse term, so with the sky at 0 an analytic lamp leaves it a rim
  // and nothing else. Measured here at the authored lamp power, one crate goes
  // from mean 2.3 to 37.6 between `strength` 0 and 1 while the walls do not
  // move. Raising it is free; leaving it at 0 is what makes props silhouettes.
  const materials = new Set(dynamic.map((m) => m.material).filter(Boolean));
  for (const mat of materials) {
    if ("maxSimultaneousLights" in mat) mat.maxSimultaneousLights = MAX_PREVIEW_LIGHTS;
  }
  return {
    lights, clusteredLights, clusteredContainer,
    lightSpecs: specs, lightSig: lightSignature(specs), lightStats: stats,
    chunkBounds: bounds, nearestChunk: nearest,
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
  const { specs, stats } = previewLightSpecs(preview.dynamic);
  preview.lightStats = stats;
  if (lightSignature(specs) !== preview.lightSig) {
    preview.clusteredContainer?.dispose();
    const clustered = new Set(preview.clusteredLights);
    for (const l of preview.lights) if (!clustered.has(l)) l.dispose();
    Object.assign(preview, buildPreviewLights(preview.dynamic, preview.container));
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
 * lamp - but nothing emits `lights` for that, and the stand-ins next to it
 * follow every frame. A lamp left behind while its own fitting slides away is
 * the same class of lie the stand-in sync exists to prevent.
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

// -------------------------------------------------- editing through the bake
//
// The preview used to take the ship away: the authored elements were disabled
// wholesale, so nothing could be picked, dragged or deleted while it was up.
// That made the one honest view of the ship the one view you could not work in,
// and the loop became tick Baked, look, untick Baked, guess, edit, tick again.
//
// It does not have to be that way, because the baked ship is not a different
// ship. `bake_lightmaps.py` exports one node per placement, named with the
// element's authoring name (or its id when it has none), under a `CHUNK_<id>`
// parent - and Blender's glTF round trip lands it in *exactly* the same world
// frame, to the last decimal. So each of those nodes can stand in for the
// element it came from: it is drawn instead of the element's own meshes, it
// answers picks on the element's behalf, and it follows the element when it is
// moved.
//
// Two consequences worth stating, because they are what makes this honest
// rather than merely convenient:
//
//   * a stand-in that has been dragged is carrying lighting that was baked
//     somewhere else. It is not re-lit and cannot be. `bakedDrift()` counts
//     exactly that, and the status line says so, so the preview never quietly
//     claims to describe a ship it no longer describes.
//   * an element placed since the bake has no stand-in at all, so it draws its
//     own geometry - unlit, because the rig is off. Looking wrong is the
//     correct answer there: it *is* unlit until the next bake.
//
// Everything here is keyed by placement **id**, never by node reference: undo
// goes through deserialize(), which disposes every placement node and builds
// new ones, and a map of stale nodes would survive one Ctrl+Z and then lie.

/** Blender's collision suffix on a duplicated name: `crate4.001`. */
const DEDUP = /\.\d{3}$/;

/** How far apart two transforms may be before they count as drift. */
const DRIFT_EPSILON = 1e-3;

const tmpMatrix = new Matrix();
const tmpFlipped = new Matrix();

/**
 * The handedness fix-up the glTF loader wrapped this chunk in.
 *
 * glTF is right-handed and Babylon is left-handed, so the loader hangs
 * everything it read under a `__root__` whose world matrix is a *reflection* -
 * `diag(-1, 1, 1)` in practice - and leaves the vertex data in glTF's own
 * frame. The mesh only comes out the right way round because that reflection
 * is still above it.
 *
 * That matters here because the stand-in has to be re-placed every frame, and
 * a re-placement that only reproduces the element's own transform silently
 * drops the reflection: the geometry then renders mirrored about the element's
 * origin, and with its winding inverted. So the conversion is read back off the
 * node the loader put it on, and re-applied. Reading it beats hardcoding
 * `diag(-1, 1, 1)`: a loader that ever converts differently keeps working.
 *
 * `null` is not a failure - a chunk sitting directly under the preview root
 * simply has no conversion above it.
 */
function conversionAbove(chunk, root) {
  let top = chunk;
  while (top.parent && top.parent !== root) top = top.parent;
  if (top === chunk) return null;
  top.computeWorldMatrix(true);
  return top.getWorldMatrix().clone();
}

/** The `CHUNK_*` node a stand-in hangs under, which is its transform parent. */
function chunkNodes(container) {
  return container.transformNodes.filter((n) => String(n.name).startsWith("CHUNK_"));
}

function chunkBounds(container) {
  return chunkNodes(container).map((chunk) => {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    const authoredMeshes = [];
    for (const placement of state.placements.values()) {
      if (placement.stage || placement.chunk !== String(chunk.name).slice("CHUNK_".length)) {
        continue;
      }
      authoredMeshes.push(...placement.node.getChildMeshes());
    }
    const meshes = authoredMeshes.length ? authoredMeshes : chunk.getChildMeshes();
    for (const mesh of meshes) {
      mesh.computeWorldMatrix(true);
      const box = mesh.getBoundingInfo().boundingBox;
      minX = Math.min(minX, box.minimumWorld.x);
      minY = Math.min(minY, box.minimumWorld.y);
      minZ = Math.min(minZ, box.minimumWorld.z);
      maxX = Math.max(maxX, box.maximumWorld.x);
      maxY = Math.max(maxY, box.maximumWorld.y);
      maxZ = Math.max(maxZ, box.maximumWorld.z);
    }
    if (!Number.isFinite(minX)) {
      chunk.computeWorldMatrix(true);
      const position = chunk.getAbsolutePosition();
      minX = maxX = position.x;
      minY = maxY = position.y;
      minZ = maxZ = position.z;
    }
    return {
      id: String(chunk.name).slice("CHUNK_".length),
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ },
    };
  });
}

function nearestChunkFor(bounds) {
  const camera = state.camera || state.scene.activeCamera;
  const position = camera?.globalPosition || camera?.position;
  if (!position) return state.activeChunk ?? bounds[0]?.id ?? null;

  let nearest = null;
  let bestDistance = Infinity;
  for (const bound of bounds) {
    const dx = position.x < bound.min.x ? bound.min.x - position.x
      : position.x > bound.max.x ? position.x - bound.max.x : 0;
    const dy = position.y < bound.min.y ? bound.min.y - position.y
      : position.y > bound.max.y ? position.y - bound.max.y : 0;
    const dz = position.z < bound.min.z ? bound.min.z - position.z
      : position.z > bound.max.z ? position.z - bound.max.z : 0;
    const distance = dx * dx + dy * dy + dz * dz;
    if (distance < bestDistance) {
      bestDistance = distance;
      nearest = bound.id;
    }
  }
  return nearest ?? state.activeChunk ?? null;
}

function sameMatrix(a, b) {
  for (let i = 0; i < 16; i++) if (Math.abs(a.m[i] - b.m[i]) > DRIFT_EPSILON) return false;
  return true;
}

/**
 * Pair each exported node with the placement it was exported from.
 *
 * The key is the element's authoring name, falling back to its id - which is
 * what the exporter writes - with Blender's `.001` dedup suffix stripped,
 * because sharing a name is a deliberate feature here (elements with the same
 * name share one behaviour entry) and Blender cannot keep two objects called
 * `crate4`.
 *
 * That makes the key ambiguous by design, so position breaks the tie: the
 * nearest unclaimed candidate wins. Nearest, not "within a tolerance" - an
 * element moved between the bake and the preview must still find its
 * stand-in, or it would be drawn twice, once as raw geometry where it is now
 * and once as baked geometry where it used to be. How far it had to reach is
 * remembered instead, and reported as drift.
 */
function matchStandIns(container, root) {
  const byKey = new Map();
  for (const e of state.placements.values()) {
    if (e.stage) continue;
    const key = e.name || e.id;
    const list = byKey.get(key);
    if (list) list.push(e); else byKey.set(key, [e]);
  }

  const standIns = new Map();
  const claimed = new Set();
  for (const chunk of chunkNodes(container)) {
    chunk.computeWorldMatrix(true);
    const toLocal = Matrix.Invert(chunk.getWorldMatrix());
    const flip = conversionAbove(chunk, root);
    // `flip` maps an editor-space world matrix to the frame the baked meshes
    // are actually stored in, so its inverse reads the other way: what the bake
    // believed this element's world matrix to be.
    const fromFlip = flip ? Matrix.Invert(flip) : null;
    for (const node of chunk.getChildren(undefined, true)) {
      // the lamps are rebuilt by buildPreviewLights, not stood in for
      if (node.metadata?.gltf?.extras?.kind === "light") continue;
      const key = String(node.name).replace(DEDUP, "");
      const candidates = byKey.get(key);
      if (!candidates?.length) continue;

      node.computeWorldMatrix(true);
      const here = node.getAbsolutePosition();
      let best = null, bestDistance = Infinity;
      for (const e of candidates) {
        if (claimed.has(e.id)) continue;
        e.node.computeWorldMatrix(true);
        const d = Vector3.DistanceSquared(e.node.getAbsolutePosition(), here);
        if (d < bestDistance) { bestDistance = d; best = e; }
      }
      if (!best) continue;

      claimed.add(best.id);
      const meshes = node.getChildMeshes();
      // The pick has to land on the element the mesh stands for. By id and not
      // by node, so that undo - which rebuilds every placement node - does not
      // leave these pointing at disposed objects.
      for (const m of meshes) {
        m.metadata = { ...(m.metadata || {}), standInFor: best.id };
        m.isPickable = true;
      }
      standIns.set(best.id, {
        node, meshes, toLocal, flip,
        // Where the *bake* had this element, recovered from the node the bake
        // wrote rather than from the element itself. Reading it off the element
        // would compare it against itself, so an element dragged between the
        // bake and the preview could never be reported as drift.
        atBake: fromFlip
          ? fromFlip.multiply(node.getWorldMatrix())
          : node.getWorldMatrix().clone(),
        // Zero, not identity: the first frame must sync unconditionally, so
        // that an element moved since the bake is put where it is now rather
        // than left where the file drew it.
        last: Matrix.Zero(),
      });
    }
  }
  return standIns;
}

/**
 * Put every stand-in back on top of its element.
 *
 * Runs before each frame, so a drag is followed live rather than on mouse-up.
 * The element's world matrix is authoritative; the stand-in's parent chunk is
 * fixed, so its local matrix is `flip * world * inverse(parentWorld)`.
 *
 * The `flip` is the whole reason this is not a one-liner. See
 * `conversionAbove`: the baked vertices are still in glTF's right-handed frame
 * and only look right because the loader left a reflection above them. Dropping
 * it - which is what placing the stand-in on the element's own matrix does -
 * mirrors every stand-in about its own origin and inverts its winding. Modules
 * that happen to be symmetric survive that unharmed, which is exactly why the
 * damage reads as "some walls are in the wrong place" rather than as the ship
 * turning inside out.
 *
 * The recompute is **forced**, and that is not belt and braces: while the bake
 * stands in for an element its own meshes are disabled, so the scene never
 * renders them and never refreshes their world matrix. An unforced
 * `computeWorldMatrix()` hands back the cached one, which in this mode is
 * simply the position the element had when it was last drawn - the stand-in
 * would then follow a drag only when something else in the frame happened to
 * force the update. Having forced it, `updateFlag` is useless as a gate (a
 * forced recompute always bumps it), so the work is gated on the matrix
 * actually having changed: 16 float compares against a decompose and two
 * matrix multiplies.
 */
function syncStandIns() {
  if (!preview) return;
  for (const [id, s] of preview.standIns) {
    const e = state.placements.get(id);
    if (!e || e.node.isDisposed()) continue;
    const world = e.node.computeWorldMatrix(true);
    if (sameMatrix(world, s.last)) continue;
    s.last.copyFrom(world);
    if (s.flip) {
      s.flip.multiplyToRef(world, tmpFlipped);
      tmpFlipped.multiplyToRef(s.toLocal, tmpMatrix);
    } else {
      world.multiplyToRef(s.toLocal, tmpMatrix);
    }
    if (!s.node.rotationQuaternion) s.node.rotationQuaternion = new Quaternion();
    tmpMatrix.decompose(s.node.scaling, s.node.rotationQuaternion, s.node.position);
  }
}

/**
 * Draw the stand-in for `entry`, and say whether it took the element's place.
 *
 * Called from applyVisibility with the visibility the element itself worked out
 * - isolation, the layer switch and the Shift+H veil all decided already - so
 * the stand-in is never a second opinion on any of that.
 */
function showStandIn(entry, on) {
  const s = preview?.standIns.get(entry.id);
  if (!s) return false;
  s.node.setEnabled(on);
  return on;
}

/** Take down stand-ins whose element has been deleted, and put back the ones
 *  undo has brought home. Called with an id from removePlacement, which is the
 *  moment a placement actually disappears; called bare from applyVisibility as
 *  a sweep, since a placement the loop never visits is exactly one that is
 *  gone. */
function settleStandIns(id) {
  if (!preview) return;
  if (id !== undefined) {
    // Deliberately not conditioned on state.placements: removePlacement calls
    // this while the entry is still in the map, on its way out.
    preview.standIns.get(id)?.node.setEnabled(false);
    return;
  }
  for (const [key, s] of preview.standIns) {
    if (!state.placements.has(key)) s.node.setEnabled(false);
  }
}

/**
 * What this bake no longer describes.
 *
 * `moved` is an element whose stand-in has been dragged since the bake, and is
 * therefore showing light that was computed somewhere else. `missing` is an
 * element the bake never saw. `gone` is a stand-in whose element has been
 * deleted. Any of the three means the room needs baking again, and the point of
 * counting them is that the preview says so out loud.
 */
export function bakedDrift() {
  const out = { moved: 0, missing: 0, gone: 0 };
  if (!preview) return out;
  for (const e of state.placements.values()) {
    if (e.stage) continue;
    if (!preview.standIns.has(e.id)) out.missing++;
  }
  for (const [id, s] of preview.standIns) {
    const e = state.placements.get(id);
    if (!e) { out.gone++; continue; }
    if (!sameMatrix(e.node.computeWorldMatrix(true), s.atBake)) out.moved++;
  }
  return out;
}

/** The meshes standing in for an element, for the selection and hover outlines
 *  - which have to trace what is on screen, not what is behind it. */
function standInMeshes(id) {
  const s = preview?.standIns.get(id);
  return s && s.node.isEnabled() ? s.meshes : [];
}

hooks.standInMeshes = standInMeshes;
hooks.showStandIn = showStandIn;
hooks.settleStandIns = settleStandIns;

function disposePreview() {
  if (!preview) return;
  state.scene.onBeforeRenderObservable.remove(preview.syncer);
  preview.clusteredContainer?.dispose();
  const clustered = new Set(preview.clusteredLights);
  for (const l of preview.lights) if (!clustered.has(l)) l.dispose();
  for (const m of preview.materials) m.dispose(true, false);
  for (const t of preview.textures) t.dispose();
  for (const t of preview.owned) t.dispose();
  preview.container.removeAllFromScene();
  preview.container.dispose();
  preview.root.dispose();
  preview = null;
}

/**
 * Swap the authored ship for the baked one, or put it back.
 *
 * The container is loaded fresh on every switch rather than kept around: a
 * preview that outlived the bake it came from would show the last render of a
 * room that has since been rebuilt, which is worse than showing nothing.
 */
export async function setBakedPreview(on) {
  const wanted = !!on;
  if (wanted === !!state.baked) return state.baked;

  if (!wanted) {
    disposePreview();
    state.baked = false;
    // The rig, and the Env/Exposure pair that goes with it, follow the flag.
    syncLightingMode();
    applyVisibility();
    emit("modes");
    return false;
  }

  const index = await loadBakedIndex();
  const container = await SceneLoader.LoadAssetContainerAsync(
    "/export/", index.glb || "ship_baked.glb", state.scene);
  const root = new TransformNode(ROOT_NAME, state.scene);
  for (const node of container.rootNodes) node.parent = root;
  container.addAllToScene();
  const dressed = dressMeshes(container.meshes, index);
  preview = {
    root, container, index, ...dressed,
    ...buildPreviewLights(dressed.dynamic, container),
    standIns: matchStandIns(container, root),
    syncer: null,
  };
  applyBakedReflectionSettings();
  applyDynamicEnvironment();
  // Registered after the match, so the first frame already has every stand-in
  // sitting on its element rather than wherever the file left it.
  preview.syncer = state.scene.onBeforeRenderObservable.add(() => {
    syncStandIns();
    syncPreviewLightNodes();
  });

  // The editor's rig is four analytic lights the game does not have, and they
  // would sit on top of the bake and hide exactly what is being inspected. The
  // flag moves first: the rig, the slider pair and the status line all read it.
  state.baked = true;
  syncLightingMode();
  applyVisibility();
  emit("modes");
  return true;
}
