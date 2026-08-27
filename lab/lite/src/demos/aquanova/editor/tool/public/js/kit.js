// Loading of kit modules, with materials and textures shared across the whole
// catalogue and geometry re-used through hardware instances.
//
// Every module .gltf in a kit references the same ~20 root-level textures and
// names its materials identically (MI_Trim_01, MI_Trim_02, M_Light, ...), so
// loading 300 modules naively would create 300 copies of the same atlas. The
// registry below keeps the first material seen under a given name and throws
// the duplicates away, textures included.

const { SceneLoader, TransformNode, Vector3, Quaternion, Matrix, Color3, Material } = BABYLON;

import { noteAuthoredEmissive, applyViewportMode } from "./editor.js";

export const materialRegistry = new Map();

let catalogue = null;
let kitMaterials = null;
let kitLights = null;
let kitReading = {};
const protoCache = new Map();
const protoPending = new Map();

/** Kit URL prefixes, and whether the loader has been taught to use them. */
const KIT_URI_EXTENSION = "AQUANOVA_kit_relative_uri";
let kitBases = [];
let uriResolverInstalled = false;

/**
 * Fetch JSON, or fail with something that names the fault.
 *
 * `fetch(url).then((r) => r.json())` reports an HTTP error as a *parse* error:
 * a 500 whose body reads "ENOENT: no such file or directory, scandir …"
 * surfaces as `SyntaxError: Unexpected token 'E'`, which names neither the URL
 * nor the reason. The server already answers with the reason in plain text, so
 * carry it rather than throw it away. Same rule as `manifest.js` applies to
 * every write it makes.
 */
async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`GET ${url} → ${r.status} ${body.trim().slice(0, 300)}`.trimEnd());
  }
  return r.json();
}

export async function loadCatalogue() {
  if (!catalogue) {
    const [cat, mats, lights, kits] = await Promise.all([
      getJson("/api/modules"),
      getJson("/data/kit_materials.json"),
      getJson("/data/kit_lights.json"),
      getJson("/data/kits.json"),
    ]);
    assertCatalogueShape(cat);
    catalogue = cat;
    kitMaterials = mats;
    kitLights = lights;
    kitReading = kits.kits || {};
    catalogue.byId = new Map();
    for (const c of catalogue.categories) {
      for (const m of c.modules) catalogue.byId.set(m.id, m);
    }
    installTextureRedirect(catalogue.kits);
    installKitUriResolver(catalogue.kits);
  }
  return catalogue;
}

/**
 * Re-read the catalogue in place after external authored data has changed.
 *
 * The server rebuilds it per request and folds the saved compounds in, so
 * saving or deleting one is only visible once the catalogue is fetched again.
 * Fluid simulations and sounds are authored in other tools too, and their
 * behavior-picker options must follow those files without restarting this page.
 * The object is *mutated* rather than replaced because `getCatalogue()` hands
 * it out and the palette holds on to what it was given; swapping the reference
 * would leave the palette rendering the old list forever.
 *
 * The kit textures are deliberately not re-installed: the redirect is a loader
 * rule keyed on kit names that cannot have changed, and re-adding it would
 * stack a second copy on the loader. The URI resolver is re-called, because
 * the one thing that *can* change here is the list of kits - saving the first
 * compound makes one - and it is the list that says which references are
 * allowed. It registers itself once and only refreshes that list.
 */
export async function reloadCatalogue() {
  if (!catalogue) return loadCatalogue();
  const fresh = await getJson("/api/modules");
  assertCatalogueShape(fresh);
  catalogue.categories = fresh.categories;
  catalogue.kits = fresh.kits;
  catalogue.defaultKit = fresh.defaultKit;
  catalogue.fluidSim = fresh.fluidSim;
  catalogue.fluidSimFlow = fresh.fluidSimFlow;
  catalogue.sounds = fresh.sounds;
  catalogue.byId = new Map();
  for (const c of catalogue.categories) {
    for (const m of c.modules) catalogue.byId.set(m.id, m);
  }
  installKitUriResolver(catalogue.kits);
  return catalogue;
}

/**
 * Refuse a catalogue this page cannot read, and say why.
 *
 * The editor server is long-lived - it is started once and left running for
 * days - but it serves `public/` off disk, so editing a file in `public/` is
 * live while editing `server.mjs` is not. The page is then newer than the
 * process answering it, and the catalogue is the contract between the two.
 *
 * That skew has to be loud, because its symptom is not. When `modelDirs` was
 * added to the catalogue, a server started before it kept answering without
 * the field; `installTextureRedirect` found nothing to build a rule from,
 * built none, and every kit quietly lost its texture redirect. What that looks
 * like is a screenful of 404s for `Props/T_Props_Batch1_Normal.png` and
 * untextured modules - which reads as a broken kit, not as a stale process,
 * and sends you looking in entirely the wrong place.
 *
 * So the fields this page needs are checked once, at load, and a catalogue
 * missing any of them stops the editor with the cure in the message rather
 * than letting it run half-wired.
 */
export function assertCatalogueShape(cat) {
  const stale = (name) => new Error(
    `the module catalogue has no "${name}" - the editor server is running older code than this page. `
    + "Restart it (Ctrl+C in its terminal, then `node server.mjs`).",
  );
  if (!Array.isArray(cat?.categories)) throw stale("categories");
  if (!Array.isArray(cat.kits)) throw stale("kits");
  if (!Array.isArray(cat.fluidSim)) throw stale("fluidSim");
  if (!cat.fluidSimFlow || typeof cat.fluidSimFlow !== "object" || Array.isArray(cat.fluidSimFlow)) throw stale("fluidSimFlow");
  if (!Array.isArray(cat.sounds)) throw stale("sounds");
  for (const kit of cat.kits) {
    // Empty is a fine answer for both - a kit whose models sit at its root has
    // no model folders and needs no redirect - so it is the field being absent
    // that says the server is old, not the field being empty.
    if (!Array.isArray(kit.modelDirs)) throw stale(`kits[${JSON.stringify(kit.name)}].modelDirs`);
    if (!Array.isArray(kit.rootTextures)) throw stale(`kits[${JSON.stringify(kit.name)}].rootTextures`);
  }
}

/**
 * Teach the glTF loader where a kit keeps its textures.
 *
 * Every kit serves its modules out of subfolders - Walls/, Platforms/, Props/…
 * - but each one names its textures with a bare filename, `T_Trim_01_ORM.png`,
 * and those textures sit one level up, at the kit root. The loader resolves a
 * bare URI next to the .gltf, so it asks for `Walls/T_Trim_01_ORM.png`, which
 * is not there.
 *
 * Rewriting the URI to `../T_Trim_01_ORM.png` would say it properly, but it
 * would mean editing every file of a pack on the way in, and a re-import from
 * Quaternius would undo the lot. Copying the 27 MB atlas set into each of the
 * six folders is the other way out, and it would put 99 MB on the CDN to say
 * the same thing six times.
 *
 * So the textures are left exactly where Quaternius puts them - which keeps
 * refreshing a kit a straight copy - and the *loader* is told the convention.
 * Only the textures the server actually found at the kit root are redirected,
 * and only from that kit's own model folders, so nothing else can be caught by
 * it: the .bin beside each .gltf keeps resolving normally.
 */
function installTextureRedirect(kits) {
  const rules = [];
  for (const kit of kits) {
    if (!kit.rootTextures?.length || !kit.modelDirs?.length) continue;
    rules.push({
      // The catalogue's URLs are encoded ("Modular%20SciFi%20MegaKit"), and so
      // is the URL the loader hands us, so both sides match as-is. These are
      // the folders on the URL, not the palette's category names: under a
      // format wrapper the two are different words for the same kit.
      dirs: new Set(kit.modelDirs.map((d) => `${kit.base}${encodeURIComponent(d)}/`)),
      base: kit.base,
      textures: new Set(kit.rootTextures.map(encodeURIComponent)),
    });
  }
  if (!rules.length) return;

  const redirect = (url) => {
    const cut = url.lastIndexOf("/") + 1;
    if (cut <= 0) return url;
    const dir = url.slice(0, cut);
    const file = url.slice(cut);
    for (const rule of rules) {
      if (rule.dirs.has(dir) && rule.textures.has(file)) return rule.base + file;
    }
    return url;
  };

  SceneLoader.OnPluginActivatedObservable.add((loader) => {
    if (loader.name !== "gltf") return;
    loader.preprocessUrlAsync = (url) => Promise.resolve(redirect(url));
  });
}

/**
 * Resolve a texture reference that climbs out of the model's own folder.
 *
 * Returns the URL to fetch, in the same shape the rest of the app uses - a
 * path when the assets are served from here, a full URL when they come from
 * the CDN - or null when the reference lands outside every kit.
 *
 * The resolution is the browser's own: `new URL` implements RFC 3986, so a
 * reference behaves exactly as it would in an `<img src>` next to the model,
 * which is what whoever wrote the file was picturing.
 */
export function resolveKitUri(rootUrl, uri, bases = kitBases) {
  let target;
  try {
    target = new URL(uri, new URL(rootUrl || "", document.baseURI));
  } catch {
    return null;
  }
  const url = target.origin === location.origin
    ? target.pathname + target.search
    : target.href;
  // Decoded on both sides: the catalogue encodes a kit name with
  // `encodeURIComponent` and the URL parser encodes a path with its own,
  // slightly shorter, list, so "Modular SciFi MegaKit" is the only spelling
  // the two are certain to agree on.
  const plain = decodeURI(url);
  return bases.some((base) => plain.startsWith(decodeURI(base))) ? url : null;
}

/**
 * Let one kit borrow another kit's textures.
 *
 * A module built for this ship out of parts of the Quaternius packs - a wall
 * of ours wearing their trim - has its textures one kit over, and Blender
 * writes exactly that when it exports: `../../Modular SciFi MegaKit/
 * T_Trim_03_Normal.png`, the path from the .gltf to the image. That is a
 * perfectly good glTF URI. RFC 3986 relative references may climb, and the
 * spec only asks that they be normalised.
 *
 * Babylon refuses it anyway: `_ValidateUri` rejects any URI containing "..",
 * so the file fails to load outright, with `'…' is invalid` and no textures.
 * The guard is there so an asset downloaded from anywhere cannot walk back up
 * the server and read what is above it - which is worth keeping, and costs
 * nothing here as long as a reference stays inside the kits.
 *
 * So the loader is handed an extension that resolves those references itself.
 * Extensions are consulted *before* the check, which is the whole reason this
 * works, and the check is then never reached for the URIs it takes over. What
 * replaces it is narrower: the reference has to land inside a kit the
 * catalogue lists, so it can reach another kit's atlas and nothing else - not
 * the export folder, not the editor's own source, nothing off this origin.
 *
 * Everything without a ".." is left alone, so the loader's usual path, and the
 * bare-filename redirect above it, are untouched.
 */
function installKitUriResolver(kits) {
  // Refreshed on every catalogue read rather than frozen at the first one: a
  // saved compound adds a kit, and a rule that had not heard of it would turn
  // a legitimate reference into a refusal.
  kitBases = kits.map((k) => k.base).filter(Boolean);
  if (uriResolverInstalled) return;

  const gltf2 = BABYLON.GLTF2;
  const register = gltf2.registerGLTFExtension
    // `false`: not a glTF extension a file has to ask for by name in
    // `extensionsUsed`, but one that applies to every file loaded.
    ? (name, factory) => gltf2.registerGLTFExtension(name, false, factory)
    : (name, factory) => gltf2.GLTFLoader.RegisterExtension(name, factory);

  register(KIT_URI_EXTENSION, (loader) => ({
    name: KIT_URI_EXTENSION,
    enabled: true,
    dispose() {},
    // `_loadUriAsync`, not `loadUriAsync`: the loader calls the underscored
    // name (`_applyExtensions(property, "loadUri", …)`), and an extension that
    // spells it the tidier way is simply never consulted - it fails exactly as
    // if it had not been registered at all.
    _loadUriAsync(context, property, uri) {
      // null hands the URI back to the loader, which is what should happen to
      // all but the handful that climb.
      if (!uri.includes("..")) return null;
      const url = resolveKitUri(loader.rootUrl || "", uri);
      if (!url) {
        throw new Error(
          `${context}: '${uri}' points outside the kits, from ${loader.rootUrl || "the page"}`,
        );
      }
      // Through the same preprocess hook as every other URI, so a redirect
      // installed for a kit still applies to what is fetched here.
      return loader.parent.preprocessUrlAsync(url).then(async (final) => {
        const r = await fetch(final);
        if (!r.ok) throw new Error(`${context}: GET ${final} → ${r.status} ${r.statusText}`.trimEnd());
        return new Uint8Array(await r.arrayBuffer());
      });
    },
  }));
  uriResolverInstalled = true;
}

export function getCatalogue() { return catalogue; }
export function getModule(id) { return catalogue?.byId.get(id) || null; }
export function getKitMaterials() { return kitMaterials; }

/**
 * The kit to start on, when nothing has been chosen yet.
 *
 * The catalogue's kit list is alphabetical, because it is read in a combo box
 * and a list nobody can predict the order of has to be read end to end. That
 * makes its first entry an accident of spelling, which is no way to pick the
 * pack a ship is mostly built from - so the server names one, out of the
 * `kits.folders` config. An older server that names none leaves the first kit,
 * which is exactly what this did before.
 */
export function defaultKit() {
  const names = (catalogue?.kits || []).map((k) => k.name);
  return names.includes(catalogue?.defaultKit) ? catalogue.defaultKit : (names[0] || null);
}

/**
 * How many metres one unit of an FBX file is.
 *
 * Babylon's FBX loader reads the scene in the file's own units and leaves them
 * there: it parses `GlobalSettings.UnitScaleFactor` but never applies it, and
 * exposes it nowhere. The Quaternius packs are exported from Blender in
 * centimetres, so a tree arrives 248 units tall and placing one drops a
 * 248-metre tree next to a 3-metre corridor. Everything else in this editor -
 * the grid, the snap steps, the glTF kits - is metres, so the file has to be
 * brought into metres before anything measures it.
 *
 * The factor is the number of centimetres in one unit, so metres = unit / 100.
 * It is read from the file rather than assumed, because "FBX is centimetres"
 * is only the default: a pack exported in metres says 100 here and is already
 * the right size, and scaling that one by 1/100 would be the same bug with the
 * sign flipped.
 *
 * Binary FBX stores it as a property record - the name as a length-prefixed
 * string, then "double", "Number", "", then a 'D' tag and the value - so the
 * name is matched with its length prefix (0x53, 15) rather than as loose text:
 * `OriginalUnitScaleFactor` sits directly after it and ends with the same
 * fifteen characters. ASCII FBX, which Quaternius does not ship but Blender
 * can write, keeps the same fields as plain text.
 */
const FBX_UNIT_NAME = "UnitScaleFactor";
export function fbxMetresPerUnit(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 18));
  let unit = null;

  if (magic === "Kaydara FBX Binary") {
    const token = [0x53, FBX_UNIT_NAME.length, 0, 0, 0,
      ...[...FBX_UNIT_NAME].map((c) => c.charCodeAt(0))];
    for (let i = 0; i + token.length < bytes.length && unit === null; i++) {
      if (bytes[i] !== 0x53 || bytes[i + 1] !== FBX_UNIT_NAME.length) continue;
      if (token.some((b, k) => bytes[i + k] !== b)) continue;
      // The three strings between the name and the value hold no capital D, so
      // the next one is the value's type tag.
      const end = Math.min(bytes.length - 8, i + token.length + 64);
      for (let d = i + token.length; d < end; d++) {
        if (bytes[d] !== 0x44) continue;
        unit = new DataView(bytes.buffer, bytes.byteOffset + d + 1, 8).getFloat64(0, true);
        break;
      }
    }
  } else {
    const text = new TextDecoder("latin1").decode(bytes.subarray(0, 65536));
    const m = text.match(/"UnitScaleFactor"\s*,\s*"double"\s*,\s*"Number"\s*,\s*""\s*,\s*([-\d.eE+]+)/);
    if (m) unit = Number(m[1]);
  }

  // The FBX default, and the only sane answer when the file does not say.
  if (!Number.isFinite(unit) || unit <= 0) unit = 1;
  return unit / 100;
}

/**
 * Re-smooth the normals of a mesh a pack exported flat.
 *
 * Quaternius' .fbx packs carry one normal per face corner and no two of them
 * agree: every vertex of CommonTree_1 is split, up to 159 degrees. That is the
 * file, not the loader - the same numbers come straight out of the FBX's own
 * `LayerElementNormal` - so the smoothing has to be ours.
 *
 * This is Blender's auto-smooth, and it is edge-based for a reason. Averaging
 * every normal that meets at a point would round off the corners of a crate,
 * and clustering by angle to the first normal seen would leave an eight-sided
 * trunk faceted, since its far side is 180 degrees from where the cluster
 * started. Instead an *edge* is smooth when the two faces sharing it are less
 * than `maxAngleDeg` apart, corners are joined across smooth edges, and each
 * group takes the average of its faces. Smoothness chains, so all eight sides
 * of that trunk end up in one group and shade as a cylinder, while the cap
 * stays a cap.
 *
 * Only the normal buffer is rewritten. Vertex count, indices, UVs and skinning
 * are all left exactly as they were, so nothing downstream - instancing, the
 * glb export - can tell the difference beyond the shading.
 */
export function smoothMeshNormals(mesh, maxAngleDeg) {
  const positions = mesh.getVerticesData("position");
  const normals = mesh.getVerticesData("normal");
  const indices = mesh.getIndices();
  if (!positions || !normals || !indices || indices.length < 3) return false;

  const cos = Math.cos((maxAngleDeg * Math.PI) / 180);
  const faces = indices.length / 3;
  const faceNormal = new Float32Array(faces * 3);
  const faceWeight = new Float32Array(faces);

  const at = (i, k) => positions[i * 3 + k];
  for (let f = 0; f < faces; f++) {
    const a = indices[f * 3], b = indices[f * 3 + 1], c = indices[f * 3 + 2];
    const ux = at(b, 0) - at(a, 0), uy = at(b, 1) - at(a, 1), uz = at(b, 2) - at(a, 2);
    const vx = at(c, 0) - at(a, 0), vy = at(c, 1) - at(a, 1), vz = at(c, 2) - at(a, 2);
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) {
      faceNormal[f * 3] = nx / len;
      faceNormal[f * 3 + 1] = ny / len;
      faceNormal[f * 3 + 2] = nz / len;
    }
    // Twice the triangle's area: a sliver then counts for as little as it
    // looks, so a fan of thin triangles cannot drag a normal round with it.
    faceWeight[f] = len;
  }

  // Vertices are welded by position only. A pack that splits a vertex to carry
  // a second UV still means one surface there, and the seam should not show as
  // a shading crease.
  const q = (i) => `${Math.round(at(i, 0) * 1e4)},${Math.round(at(i, 1) * 1e4)},${Math.round(at(i, 2) * 1e4)}`;
  const cornerKey = new Array(positions.length / 3);
  for (let v = 0; v < cornerKey.length; v++) cornerKey[v] = q(v);

  const parent = new Int32Array(indices.length);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const root = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const join = (a, b) => { const ra = root(a), rb = root(b); if (ra !== rb) parent[ra] = rb; };

  // corner slot -> the vertex it points at, and which face it belongs to
  const edges = new Map();
  for (let f = 0; f < faces; f++) {
    for (let e = 0; e < 3; e++) {
      const v0 = indices[f * 3 + e], v1 = indices[f * 3 + ((e + 1) % 3)];
      const k0 = cornerKey[v0], k1 = cornerKey[v1];
      if (k0 === k1) continue;
      const key = k0 < k1 ? `${k0}|${k1}` : `${k1}|${k0}`;
      const other = edges.get(key);
      if (other === undefined) { edges.set(key, { f, e }); continue; }
      const g = other.f;
      const dot = faceNormal[f * 3] * faceNormal[g * 3]
        + faceNormal[f * 3 + 1] * faceNormal[g * 3 + 1]
        + faceNormal[f * 3 + 2] * faceNormal[g * 3 + 2];
      if (dot < cos) continue;                       // a crease: leave it split
      // Join the two corners at each end of the shared edge, matched by
      // position - the two faces need not index the same vertex there.
      for (const key0 of [k0, k1]) {
        let ca = -1, cb = -1;
        for (let i = 0; i < 3; i++) {
          if (cornerKey[indices[f * 3 + i]] === key0) ca = f * 3 + i;
          if (cornerKey[indices[g * 3 + i]] === key0) cb = g * 3 + i;
        }
        if (ca >= 0 && cb >= 0) join(ca, cb);
      }
    }
  }

  const sum = new Map();
  for (let c = 0; c < indices.length; c++) {
    const r = root(c), f = (c / 3) | 0;
    const acc = sum.get(r) || [0, 0, 0];
    acc[0] += faceNormal[f * 3] * faceWeight[f];
    acc[1] += faceNormal[f * 3 + 1] * faceWeight[f];
    acc[2] += faceNormal[f * 3 + 2] * faceWeight[f];
    sum.set(r, acc);
  }

  const out = new Float32Array(normals);   // a vertex no triangle uses keeps what it had
  let changed = false;
  for (let c = 0; c < indices.length; c++) {
    const v = indices[c], acc = sum.get(root(c));
    const len = Math.hypot(acc[0], acc[1], acc[2]);
    const f = (c / 3) | 0;
    const nx = len > 0 ? acc[0] / len : faceNormal[f * 3];
    const ny = len > 0 ? acc[1] / len : faceNormal[f * 3 + 1];
    const nz = len > 0 ? acc[2] / len : faceNormal[f * 3 + 2];
    // A vertex shared by two groups cannot hold two normals; the pack splits
    // every corner, so this only ever writes the same answer twice.
    out[v * 3] = nx; out[v * 3 + 1] = ny; out[v * 3 + 2] = nz;
    if (Math.abs(nx - normals[v * 3]) > 1e-4 || Math.abs(ny - normals[v * 3 + 1]) > 1e-4
      || Math.abs(nz - normals[v * 3 + 2]) > 1e-4) changed = true;
  }
  if (!changed) return false;
  mesh.setVerticesData("normal", out, false);
  return true;
}

/** The reading rules for a kit, from kits.json. */
export function getKitReading(kitName) { return kitReading?.[kitName] || null; }

/**
 * Put back the material values a pack's own export dropped.
 *
 * The RPG pack's `Glass` and its five `Liquid_*` materials are the only ones
 * in it that arrive with no colour at all - 53 of them against 214 that carry
 * one - and they are exactly the see-through ones. Blender's FBX exporter
 * writes Phong properties out of a Principled BSDF and has nothing to write
 * for a transparent shader, so it wrote no property block for them at all and
 * Babylon fell back on its default 0.8 grey. A filled potion was then the same
 * uniform grey as an empty one: the liquid, the glass and the air between them
 * all rendered alike, which reads as "the bottle is empty".
 *
 * Authored per kit rather than by bare material name, because a name is not an
 * identity across kits - `Glass` and `M_Glass` and `MI_Trim_01` all mean
 * different things in different packs.
 *
 * Applied here, on the way out of the loader, so the ship and the thumbnail
 * scene get the same values and - this is the part that matters - the material
 * is already in its final state when either of them takes its dedupe key,
 * which includes the transparency.
 */
function applyKitMaterialOverrides(container, kitName) {
  const rules = getKitReading(kitName)?.materials;
  if (!rules) return;

  const all = new Set();
  for (const mat of container.materials) {
    all.add(mat);
    for (const sub of mat.subMaterials || []) if (sub) all.add(sub);
  }
  for (const mat of all) {
    const rule = rules[mat.name];
    if (!rule || typeof rule !== "object") continue;
    if (rule.tint) {
      const c = new Color3(rule.tint[0], rule.tint[1], rule.tint[2]);
      // A pack can come in as either kind of material - the FBX loader builds
      // Phong, the glTF loader PBR - and the value means the same in both.
      if ("albedoColor" in mat) mat.albedoColor = c;
      if ("diffuseColor" in mat) mat.diffuseColor = c;
    }
    if (typeof rule.alpha === "number") {
      mat.alpha = rule.alpha;
      mat.transparencyMode = rule.alpha < 1
        ? Material.MATERIAL_ALPHABLEND : Material.MATERIAL_OPAQUE;
    }
    if (typeof rule.roughness === "number" && "roughness" in mat) mat.roughness = rule.roughness;
  }
}

/**
 * A module's asset container, loaded in metres and shaded the way the pack is
 * meant to look.
 *
 * Shared by the ship and the thumbnail scene so the two can never disagree
 * about how big a module is, or how it is shaded - which they would the moment
 * one of them grew a correction the other did not have.
 */
export async function loadModuleContainer(mod, scene) {
  const cut = mod.url.lastIndexOf("/") + 1;
  const dir = mod.url.slice(0, cut);
  const file = mod.url.slice(cut);
  let container;

  if (!/\.fbx$/i.test(file)) {
    container = await SceneLoader.LoadAssetContainerAsync(dir, file, scene);
  } else {
    // Fetched here rather than by the loader so the unit can be read off the
    // same bytes: handing the buffer straight on keeps it to one download, and
    // the root url is still given, so a textured pack resolves its maps as
    // usual.
    const buffer = await (await fetch(mod.url)).arrayBuffer();
    const metres = fbxMetresPerUnit(buffer);
    container = await SceneLoader.LoadAssetContainerAsync(dir, new File([buffer], file), scene);
    if (metres !== 1) {
      for (const root of container.rootNodes) root.scaling.scaleInPlace(metres);
    }
  }

  const smooth = getKitReading(mod.kit)?.smoothNormalsBelowDeg;
  if (smooth) {
    for (const mesh of container.meshes) {
      if (mesh.getTotalVertices() > 0) smoothMeshNormals(mesh, smooth);
    }
  }
  applyKitMaterialOverrides(container, mod.kit);
  return container;
}

/**
 * The lights a module comes with, as authored partials.
 *
 * Read once per placement, never re-read: see kit_lights.json's `_note`. The
 * array is per module because one strip is not always one lamp - the corner
 * light is an arc, and an area light is flat.
 */
export function getKitLights(moduleId) {
  const list = kitLights?.modules?.[moduleId];
  return Array.isArray(list) ? list : [];
}

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
 * Not the name on its own. The kits name materials identically across modules
 * *and* give the same name to genuinely different materials:
 *
 *  - the MegaKit authors some of them two ways: `M_Glass` is `BLEND` with an
 *    alpha of 0 in two files and `OPAQUE` in twelve, `M_Decal_White` is `MASK`
 *    in thirty-one and `OPAQUE` in twenty-six;
 *  - every Pirate model calls its material `Atlas` and embeds *its own* 32x32
 *    slice of the palette, so the barrel's atlas is white from row 7 down
 *    while a character reads its skin from row 9;
 *  - `MI_Trim_01`, `MI_Trim_02`, `MI_Trim_03`, `MI_Trim_03_Dark` and `M_Black`
 *    exist in both the MegaKit and the Essentials Kit over different atlases,
 *    and the Essentials Kit alone maps `MI_Trim_02` to two of them.
 *
 * Keyed by name alone, whichever module loaded first decided what everything
 * with that name looked like everywhere - and the palette keeps its own cache,
 * filled in a different order, so a module could be right on its tile and
 * wrong in the ship at the same time. The Pirate characters came out grey
 * except for the prop in their hand, which is the one part of them that reads
 * from a row the barrel's atlas also fills in.
 *
 * So the key is the name, the transparency - the one thing you cannot share -
 * and the textures, which is what the sharing is *for*. Sharing then happens
 * exactly when it is free: the MegaKit's modules all name the same files at
 * the kit root and still collapse to one material, while a kit that embeds a
 * texture per model gets one material per model, which for a 32x32 palette
 * costs nothing.
 */
export function materialKey(mat) {
  const mode = mat.transparencyMode === null || mat.transparencyMode === undefined
    ? "opaque" : mat.transparencyMode;
  return `${mat.name}|${mode}|${mat.alpha}|${textureIdentity(mat)}`;
}

/**
 * The textures a material reads, as something two materials can be compared on.
 *
 * `Texture.url` is the honest answer for both kinds of kit: a file the loader
 * fetched is its resolved URL - already through the kit-root redirect, so two
 * modules naming the same atlas agree - and an image embedded in a .gltf is
 * given `data:<the .gltf's url>#image0`, which names the file it came out of
 * and the index within it. Slot order is fixed by the material class, so the
 * same set of maps always spells the same key.
 */
function textureIdentity(mat) {
  const textures = mat.getActiveTextures?.() ?? [];
  return textures.map((t) => t.url || t.name || "?").join(" ");
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
    const container = await loadModuleContainer(mod, window.__scene);

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

/**
 * The world matrix of a node whose ancestors may be stale.
 *
 * Proto containers are parked disabled once loaded, and a disabled node is not
 * re-evaluated by the render loop, so its cached world matrix - and every one
 * above it - can be left over from load time. Forcing the chain top-down is
 * the only order that is right: `computeWorldMatrix(true)` on a node reads its
 * parent's *cached* matrix, so forcing a child first would compose a fresh
 * local against a stale parent.
 */
function worldMatrixOf(node) {
  const chain = [];
  for (let n = node; n; n = n.parent) chain.push(n);
  for (let i = chain.length - 1; i >= 0; i--) chain[i].computeWorldMatrix(true);
  return node.getWorldMatrix();
}

/**
 * The node an orphan joint hangs from, carrying the basis its meshes were baked
 * into.
 *
 * `getProto` bakes each mesh's **world** matrix into its part, and that world
 * matrix runs through the loader's `__root__` - the node Babylon gives every
 * glTF to turn the file's right-handed data into our left-handed scene. It is a
 * mirror, so the baked scaling holds a -1 and the mesh clones inherit it.
 *
 * A joint cannot be baked the same way: animation channels drive a joint's
 * *local* TRS, so anything written there is overwritten on the first frame. The
 * basis has to arrive from above instead - which is how the asset itself is
 * built, the joints and the meshes sharing an armature under `__root__`.
 *
 * Parenting orphan joints straight to the placement root is what dropped it,
 * and the two halves of one fan then disagreed about which way round the world
 * is. Babylon hides the disagreement: `Skeleton.prepare` copies only a linked
 * node's local TRS into its bone, so an ancestor is invisible to it and the
 * mesh's own world matrix supplies the mirror at draw time. glTF resolves a
 * skin the other way round - the mesh node's transform is ignored and the
 * joint's *global* transform is what counts - so the exported ship lost the
 * mirror, and with it the winding: every skinned normal came out inverted and
 * the fans were lit from the wrong side. Nothing looked broken in the editor,
 * which is exactly why it survived to the runtime.
 *
 * So the fix belongs here rather than in the export: it is not a serialisation
 * quirk to paper over, it is a placement that was only half converted.
 */
function basisFor(sourceParent, root, nodeName, basisNodes, animationNodes) {
  if (!sourceParent) return root;
  const existing = basisNodes.get(sourceParent);
  if (existing) return existing;

  // Named for the job, not for `sourceParent`: the node it stands in for is
  // usually the loader's `__root__`, and a node called `__root__` in a ship
  // .glb invites a loader to treat it as one of its own.
  const suffix = basisNodes.size === 0 ? "Basis" : `Basis${basisNodes.size + 1}`;
  const basis = new TransformNode(`${nodeName}_${suffix}`, window.__scene);
  const pos = new Vector3();
  const rot = new Quaternion();
  const scl = new Vector3();
  worldMatrixOf(sourceParent).decompose(scl, rot, pos);
  basis.position.copyFrom(pos);
  basis.rotationQuaternion = rot;
  basis.scaling.copyFrom(scl);
  basis.parent = root;
  // Rides `_shipAnimationNodes` so the export whitelists and renames it with
  // the joints it carries; left out, the joints would reach the file parented
  // to the placement root and the mirror would be lost all over again.
  basis._shipAnimationSourceName = suffix;
  basisNodes.set(sourceParent, basis);
  animationNodes.push(basis);
  return basis;
}

/**
 * Create a placement of `moduleId`.
 *
 * Static modules use hardware instances. Animated modules need private mesh
 * clones, skeletons, linked transform nodes, and animation groups so every
 * placement can be exported and played independently.
 */
export async function instantiate(moduleId, nodeName) {
  const proto = await getProto(moduleId);
  const root = new TransformNode(nodeName, window.__scene);
  root.rotationQuaternion = Quaternion.Identity();

  const sourceSkeletons = proto.container.skeletons;
  if (sourceSkeletons.length && proto.container.animationGroups.length) {
    const targetMap = new Map();
    const animationNodes = [];
    for (const sourceSkeleton of sourceSkeletons) {
      for (const sourceBone of sourceSkeleton.bones) {
        const sourceTarget = sourceBone.getTransformNode();
        if (!sourceTarget || targetMap.has(sourceTarget)) continue;
        const target = new TransformNode(`${nodeName}_${sourceTarget.name}`, window.__scene);
        target.position.copyFrom(sourceTarget.position);
        target.rotationQuaternion = sourceTarget.rotationQuaternion?.clone() ?? Quaternion.FromEulerAngles(sourceTarget.rotation.x, sourceTarget.rotation.y, sourceTarget.rotation.z);
        target.scaling.copyFrom(sourceTarget.scaling);
        target._shipAnimationSourceName = sourceTarget.name;
        targetMap.set(sourceTarget, target);
        animationNodes.push(target);
      }
    }
    const basisNodes = new Map();
    for (const [sourceTarget, target] of targetMap) {
      target.parent = targetMap.get(sourceTarget.parent)
        ?? basisFor(sourceTarget.parent, root, nodeName, basisNodes, animationNodes);
    }
    const skeletonMap = new Map();
    for (const sourceSkeleton of sourceSkeletons) {
      const skeleton = sourceSkeleton.clone(`${nodeName}_${sourceSkeleton.name}`);
      for (let i = 0; i < sourceSkeleton.bones.length; i++) {
        const sourceTarget = sourceSkeleton.bones[i].getTransformNode();
        if (sourceTarget) skeleton.bones[i].linkTransformNode(targetMap.get(sourceTarget));
      }
      skeletonMap.set(sourceSkeleton, skeleton);
    }

    for (let i = 0; i < proto.parts.length; i++) {
      const part = proto.parts[i];
      const clone = part.mesh.clone(`${nodeName}#${i}`, root, true);
      if (!clone) throw new Error(`could not clone animated module part ${part.mesh.name}`);
      clone.position.copyFrom(part.position);
      clone.rotationQuaternion = part.rotationQuaternion.clone();
      clone.scaling.copyFrom(part.scaling);
      clone.skeleton = skeletonMap.get(part.mesh.skeleton) ?? null;
      clone.isPickable = true;
      clone.metadata = { placementRoot: root };
    }
    const animationGroups = proto.container.animationGroups.map((group) =>
      group.clone(group.name, (target) => {
        const mapped = targetMap.get(target);
        if (!mapped) throw new Error(`animation "${group.name}" targets unsupported node "${target?.name ?? "?"}"`);
        return mapped;
      }, true, true));
    const skeletons = [...skeletonMap.values()];
    root._shipAnimationGroups = animationGroups;
    root._shipSkeletons = skeletons;
    root._shipAnimationNodes = animationNodes;
    root.onDisposeObservable.add(() => {
      for (const group of animationGroups) group.dispose();
      for (const skeleton of skeletons) skeleton.dispose();
    });
    return root;
  }

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
