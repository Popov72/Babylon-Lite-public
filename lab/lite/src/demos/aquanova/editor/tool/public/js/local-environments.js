// Capture each room's environment probe, in the editor, from the live ship.
//
// A probe is a small cubemap of one room, prefiltered for roughness, that the
// runtime hangs on that room's materials so a metal wall reflects its own
// corridor instead of a generic sky. Six faces are rendered at the probe's
// capture point, GGX-prefiltered and serialised to Babylon's .env format.
//
// This deliberately runs in the editor browser rather than in a build step:
// Babylon already owns the GGX prefilter and the .env serializer there, so the
// file that comes out is produced by exactly the code the runtime parses it
// with. It also means a probe can be re-taken the moment a room is rebuilt,
// without leaving the tool.
//
// The six faces are rendered here rather than by Babylon's ReflectionProbe.
// A probe is a cube render target, and a cube render target sets its lighting
// up ONCE for all six faces - which is fine for the analytic lights it was
// written for, and wrong for the clustered lights this ship uses, because a
// clustered light reaches a pixel through a screen-space tile mask built for
// one camera. See captureProbe.
//
// What is NOT captured is as important as what is. The ship's own reflections
// are suspended for the render (see withRuntimeCapture), so a probe records
// direct light and emissive surfaces only. Without that, each generation would
// photograph the previous generation's cubemaps and no two runs would agree.
// The editor's exposure and tone mapping are held off for the same reason: a
// probe is scene-referred radiance, and the runtime exposes it itself.

import { state, whileBusy, setBusyMessage, withDeadline, CONFIG_DEFAULTS } from "./editor.js";
import { meshesInProbeVolume, renderListFor, setCaptureViewpoint, withRuntimeCapture, authoredMaterialOf } from "./runtime.js";

/**
 * How long any one step of a capture may take before it is called a failure.
 *
 * Every await below is settled by Babylon from inside the render loop, and each
 * one can wait for ever on something that will never happen - a shader that
 * fails to link, a read-back on a lost context, a server that stopped
 * answering. None of them throw on their own, and the capture holds the busy
 * lock, so a stall does not cost a probe: it costs the editor, which sits
 * behind its overlay with no way out but a reload.
 *
 * Two minutes is far longer than the slowest step measured on the real ship
 * (a 256 px probe prefilters and serialises in about four seconds), so this can
 * only ever fire on something genuinely wedged.
 */
const CAPTURE_TIMEOUT_MS = 120000;

/**
 * Face size every probe is captured at, in pixels.
 *
 * A ship-wide setting rather than a per-probe one: the runtime holds the
 * captures in a cube texture array, whose slices all share one dimension. It
 * is part of the digest as well as of the render, so raising it is what makes
 * every probe stale at once.
 */
function probeResolution() {
  return Math.max(16, Number(state.config.probeResolution) || CONFIG_DEFAULTS.probeResolution);
}

/**
 * Sample count for the GGX prefilter.
 *
 * Babylon's OFFLINE quality (4096) is meant for a one-off conversion of a sky;
 * a ship has a probe per room and they are re-taken on every edit, so this
 * trades a little roughness noise for a generation that finishes while you are
 * still looking at it.
 */
const FILTER_QUALITY = 1024;

/**
 * The six faces of a cubemap, in the order the .env stores them, each with the
 * camera basis that fills it.
 *
 * A cube face is not simply "the view along that axis": the cubemap convention
 * fixes which world direction lands on which texel, and two of the six are not
 * what a naive look-at produces. `forward` is the face's axis. `up` is the
 * world direction that must land on the FIRST row of the face's data.
 *
 * It follows from the mapping the hardware samples with. For +X, the column
 * axis runs towards -Z and the row axis towards -Y, so the first row - row
 * zero, the top of the image - is +Y, which is exactly the camera's up vector.
 * Work the other five through and the table below falls out. Babylon's own GGX
 * prefilter writes the same six bases down the other way round, as the +column
 * and +row directions (HDRFiltering._prefilterInternal), so a face rendered
 * here and a face read there agree by construction rather than by luck.
 *
 * The poles are the two worth reading twice: looking UP, the top of the image
 * is -Z; looking DOWN, it is +Z. Babylon's ReflectionProbe has neither - it
 * aims the +Y slot DOWNWARDS, because the `_invertYAxis` that would correct it
 * is private, permanently false and has no setter - which is the second reason
 * this file renders the faces itself.
 */
const CUBE_FACES = [
  { forward: [1, 0, 0], up: [0, 1, 0] },
  { forward: [-1, 0, 0], up: [0, 1, 0] },
  { forward: [0, 1, 0], up: [0, 0, -1] },
  { forward: [0, -1, 0], up: [0, 0, 1] },
  { forward: [0, 0, 1], up: [0, 1, 0] },
  { forward: [0, 0, -1], up: [0, 1, 0] },
];

/**
 * A camera whose view is stated outright instead of inferred from a target.
 *
 * Two of the six faces look straight up and straight down, and `setTarget`
 * expresses neither: it recovers pitch through `Math.atan`, which cannot reach
 * a right angle, and it forces roll to zero - so the poles come out twisted by
 * whatever the degenerate case happens to yield. A face basis is not something
 * to be recovered from a point in space; it IS the cubemap convention, so it is
 * handed to the renderer exactly as written in CUBE_FACES.
 */
class FaceCamera extends BABYLON.FreeCamera {
  constructor(name, position, scene) {
    super(name, position, scene, false);
    this._faceView = BABYLON.Matrix.Identity();
    this._faceTarget = new BABYLON.Vector3();
  }

  /** Aim down `forward`, with `up` on the first row of the image. */
  lookAlong(forward, up) {
    this.position.addToRef(forward, this._faceTarget);
    BABYLON.Matrix.LookAtLHToRef(this.position, this._faceTarget, up, this._faceView);
  }

  _getViewMatrix() {
    return this._faceView;
  }

  // The view changes without the position or rotation changing, which is the
  // only thing the inherited cache watches.
  _isSynchronizedViewMatrix() {
    return false;
  }
}

/**
 * Wait until every mesh a face will draw can actually be drawn.
 *
 * Babylon skips a mesh whose effect is still compiling rather than queueing it,
 * so rendering too early does not delay a face - it silently leaves a hole in
 * it, and a hole in a cubemap is a black patch nothing later repairs. Asking a
 * mesh whether it is ready is also what starts its compile, and compiles only
 * advance while the scene draws, which is why the editor's render loop keeps
 * running underneath the busy overlay.
 */
async function waitForRenderList(renderList, timeoutMs = CAPTURE_TIMEOUT_MS) {
  const deadline = performance.now() + timeoutMs;
  for (;;) {
    const waiting = renderList.filter((mesh) => !mesh.isReady(true));
    if (!waiting.length) return;
    if (performance.now() > deadline) {
      throw new Error(`environment probe: ${waiting.length} mesh(es) never became ready to render`);
    }
    // A timer as well as the frame, because the deadline above is only reached
    // by going round the loop: a tab moved to the background stops producing
    // frames altogether, and waiting on requestAnimationFrame alone would sit
    // there for as long as it stayed hidden, holding the busy lock with it.
    await new Promise((resolve) => {
      const timer = setTimeout(resolve, 250);
      requestAnimationFrame(() => { clearTimeout(timer); resolve(); });
    });
  }
}

/** Enough precision that a millimetre of drift re-captures and float noise does not. */
function round(n) {
  return Math.round(Number(n) * 1e4) / 1e4;
}

function environmentTools() {
  const tools = BABYLON.EnvironmentTextureTools;
  if (!tools?.CreateEnvTextureAsync) {
    throw new Error("Babylon environment serializer is not loaded");
  }
  return tools;
}

/**
 * What a probe's capture depends on, as one string.
 *
 * Everything the six renders can see, and nothing else: the probe's own volume
 * and capture point, the world transform and material of every mesh inside the
 * box, and every authored lamp. Two ships that agree on this string cannot
 * produce different cubemaps, which is what makes the stamp on disk a truthful
 * claim that the file is current.
 *
 * The lamps are ALL included rather than only the ones scoped to this room. A
 * clustered lamp is scene-global by agreement, so the room next door genuinely
 * can light this one, and deciding which lamps reach a box is exactly the
 * question the renderer answers - guessing at it here would be a second,
 * disagreeing implementation.
 *
 * Camera position is deliberately absent even though scoped lamps are enabled
 * by proximity: the capture forces its own scoping (see captureProbe), so where
 * the user happens to be standing cannot change the result.
 *
 * So is the dressing. The digest runs inside the Runtime view, where every
 * mesh wears a clone named after the probe it resolved to - and that probe only
 * exists once a capture has been taken. Hashing the clone would make the first
 * capture change the state the second is compared against, and every pass would
 * find every probe stale for ever. The authored material is what is hashed.
 */
function probeDigestSource(probe, meshes) {
  const parts = [
    `probe:${probe.id}`,
    ...(probe.shape === "sphere"
      ? [`shape:sphere`, `sphere:${probe.spherePosition.map(round).join(",")}`, `radius:${round(probe.sphereRadius)}`]
      : [`box:${probe.boxPosition.map(round).join(",")}`, `size:${probe.boxSize.map(round).join(",")}`]),
    `at:${probe.capturePosition.map(round).join(",")}`,
    `angle:${round(probe.angle || 0)}`,
    `res:${probeResolution()}`,
  ];

  const rows = [];
  for (const mesh of meshes) {
    mesh.computeWorldMatrix(true);
    const m = mesh.getWorldMatrix().m;
    rows.push(`${mesh.name}|${authoredMaterialOf(mesh)?.name || ""}|${Array.from(m, round).join(",")}`);
  }
  // Babylon's mesh order follows creation, so an undo that rebuilds the same
  // ship in a different order would otherwise read as a change.
  rows.sort();
  parts.push(`meshes:${rows.length}`, ...rows);

  const lamps = [];
  for (const light of state.lights.values()) {
    const r = light.runtime;
    if (!r || r.type === "none") continue;
    light.node.computeWorldMatrix(true);
    const at = light.node.getAbsolutePosition();
    const m = light.node.getWorldMatrix().m;
    lamps.push(
      `${light.id}|${r.type}|${r.clustered ? 1 : 0}|${r.color.map(round).join(",")}`
        + `|${round(r.intensity)}|${round(r.range)}|${round(r.angle)}`
        + `|${round(at.x)},${round(at.y)},${round(at.z)}`
        + `|${round(m[4])},${round(m[5])},${round(m[6])}`,
    );
  }
  lamps.sort();
  parts.push(`lights:${lamps.length}`, ...lamps);
  return parts.join("\n");
}

/** SHA-256 of the digest source, hex. The editor is same-origin on localhost, so subtle is available. */
async function digest(source) {
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
  return [...new Uint8Array(bytes)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Render one probe and serialise it.
 *
 * **The six faces are rendered one at a time, into a 2D target.** The obvious
 * thing - a cube render target, or the ReflectionProbe that wraps one - cannot
 * light this ship. The ship's lamps are clustered, and a clustered light does
 * not reach a pixel directly: it reaches it through a tile mask, a small buffer
 * that says which lights touch which screen tile at which depth. That mask is
 * built once per render, for one camera. A cube target calls its setup once and
 * then loops six faces inside it, so five of the six would be lit through a
 * mask built for the wrong direction - and in practice all six came out black
 * apart from the emissive surfaces, which need no light at all. One camera, one
 * render, one mask: six separate renders is not an optimisation, it is the only
 * shape that can be correct.
 *
 * `enableClusteredLights` is what asks for the mask to be rebuilt against THIS
 * target's camera, and `incrementRenderId` is what stops that rebuild from
 * being skipped - the light data carries the render id it was last built for
 * and returns early on a match, and a 2D render does not advance the id by
 * itself.
 *
 * **The faces are read back and re-uploaded rather than rendered into a cube.**
 * That is the price of six renders, and it buys the orientation as well: the
 * cubemap convention is applied here, in CUBE_FACES, instead of being inherited
 * from a probe that gets the poles wrong.
 *
 * **Half float, not bytes.** A room lit by a bright lamp carries values well
 * above 1, and an 8-bit target would clip them to white - which the .env
 * serializer then refuses outright, because a prefiltered environment that
 * cannot hold a highlight is not worth writing. `gammaSpace = false` marks the
 * result as linear; keeping image processing out of it is withRuntimeCapture's
 * job, and has to happen before the shaders compile.
 *
 * **Cleared to black.** Anywhere a face sees no geometry, the ship has a hole
 * in it, and what is on the other side of a hole in a spaceship is space. The
 * scene's own clear colour is the editor's backdrop - a grey-blue chosen to
 * make the grid readable - and letting that into a room's environment would
 * tint every reflection in the ship with the colour of the tool that built it.
 */
async function captureProbe(probe, meshes) {
  const scene = state.scene;
  const engine = scene.getEngine();
  const size = probeResolution();
  const at = new BABYLON.Vector3(probe.capturePosition[0], probe.capturePosition[1], probe.capturePosition[2]);
  const type = engine.getCaps().textureHalfFloatRender
    ? BABYLON.Constants.TEXTURETYPE_HALF_FLOAT
    : BABYLON.Constants.TEXTURETYPE_FLOAT;

  const camera = new FaceCamera(`CAPTURE_FACE_${probe.id}`, at.clone(), scene);
  camera.minZ = 0.05;
  camera.maxZ = 1000;
  camera.fov = Math.PI / 2;
  // Left to itself the projection would take its aspect ratio from the canvas,
  // because the target is not bound yet when the tile mask is tiled - a 16:9
  // frustum against a square face, and a mask that lines up with neither.
  camera.freezeProjectionMatrix(
    BABYLON.Matrix.PerspectiveFovLH(camera.fov, 1, camera.minZ, camera.maxZ, engine.isNDCHalfZRange),
  );

  const target = new BABYLON.RenderTargetTexture(`CAPTURE_${probe.id}`, size, scene, {
    type,
    generateMipMaps: false,
    samplingMode: BABYLON.Constants.TEXTURE_BILINEAR_SAMPLINGMODE,
    enableClusteredLights: true,
  });
  target.gammaSpace = false;
  target.clearColor = new BABYLON.Color4(0, 0, 0, 1);
  target.activeCamera = camera;
  target.renderList = renderListFor(meshes);

  let cube = null;
  try {
    // A scoped lamp is only switched on in the chunk the player stands in, so
    // the capture has to stand at the capture point before the lamps are read.
    setCaptureViewpoint(at);

    const faces = [];
    for (const face of CUBE_FACES) {
      camera.lookAlong(
        new BABYLON.Vector3(face.forward[0], face.forward[1], face.forward[2]),
        new BABYLON.Vector3(face.up[0], face.up[1], face.up[2]),
      );
      await waitForRenderList(target.renderList);
      scene.incrementRenderId();
      target.render();
      // Read the half floats as they are: converting to 32-bit only to hand
      // them straight back to a half-float cube would round the values twice.
      // The rows come back bottom first - a frame buffer is read from its
      // origin, and that origin is the bottom left - so the upload below is
      // told to invert them, which is what puts the camera's up on the first
      // row of the face, where the convention wants it.
      faces.push(await withDeadline(target.readPixels(0, 0, null, true, true), CAPTURE_TIMEOUT_MS,
        `${probe.id}: reading back a cubemap face`));
    }

    cube = new BABYLON.RawCubeTexture(
      scene, faces, size, BABYLON.Constants.TEXTUREFORMAT_RGBA, type,
      true, true, BABYLON.Constants.TEXTURE_TRILINEAR_SAMPLINGMODE,
    );
    cube.gammaSpace = false;
    await withDeadline(
      new BABYLON.HDRFiltering(engine, { quality: FILTER_QUALITY }).prefilter(cube),
      CAPTURE_TIMEOUT_MS, `${probe.id}: prefiltering the cubemap`,
    );
    // The runtime reads diffuse irradiance from the spherical harmonics the
    // serializer computes, so the irradiance texture would be a second copy of
    // the same information at several times the size.
    return await withDeadline(environmentTools().CreateEnvTextureAsync(cube, {
      imageType: "image/png",
      disableIrradianceTexture: true,
    }), CAPTURE_TIMEOUT_MS, `${probe.id}: serialising the .env`);
  } finally {
    setCaptureViewpoint(null);
    cube?.dispose();
    target.dispose();
    camera.dispose();
  }
}

/**
 * A fetch that cannot hang.
 *
 * `AbortController` rather than a raced timer: this is the one await in the
 * capture that can genuinely be cancelled, and letting an abandoned upload keep
 * pushing a megabyte at a server nobody is listening to is not a thing to leave
 * running behind an error message.
 */
async function fetchWithDeadline(url, init, what) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CAPTURE_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (controller.signal.aborted) throw new Error(`${what} did not answer within ${CAPTURE_TIMEOUT_MS / 1000}s`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function upload(id, hash, bytes) {
  const response = await fetchWithDeadline(
    `/api/local-environment/${encodeURIComponent(id)}?hash=${encodeURIComponent(hash)}`,
    { method: "PUT", body: bytes },
    `${id}: writing the .env`,
  );
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `server said ${response.status}`);
  return result.bytes;
}

/**
 * Bring every authored probe's .env up to date with the ship as it stands.
 *
 * **Runs under the busy lock.** A capture takes the scene over - the runtime
 * lighting mode, the materials' reflections, the active camera, the image
 * processing flag and the viewport itself - so for its duration the editor is
 * not in a state any edit could be applied to. The overlay says which room is
 * being taken; `whileBusy` is what makes the panels inert and swallows the
 * pointer, so nothing can be touched behind it.
 *
 * The whole authored list is declared first, before anything is rendered. That
 * single call is what prunes a deleted probe's file from the folder and what
 * decides which probes still owe a capture - the server compares the digest
 * being declared against the stamp beside each file, so a probe whose room has
 * not been touched is skipped and a probe that has never been taken is not.
 *
 * `force` declares every probe pending whatever its stamp says. The digest
 * covers what the editor can see, so a change it cannot - a kit asset replaced
 * on disk, a capture bug fixed - leaves stale files that nothing else would
 * ever retake.
 *
 * `only` takes exactly one probe by id and leaves the rest of the folder as it
 * stands. It is an explicit instruction rather than a question about staleness,
 * so that probe is captured whether or not its stamp still matches - which is
 * what `force` does, for one probe. The declaration is still the whole authored
 * list, because that is what prunes deleted probes and records the digests, but
 * it is deliberately sent unforced: a global force deletes every stamp, and a
 * single-probe capture must not leave the other rooms looking stale for a
 * render nobody asked for.
 */
export async function generateLocalEnvironments(onProgress = () => {}, { force = false, only = null } = {}) {
  return await whileBusy("checking environment probes…", async () => await withRuntimeCapture(async () => {
    const probes = [];
    for (const probe of state.environmentProbes.values()) {
      const meshes = meshesInProbeVolume(probe);
      probes.push({
        probe,
        meshes,
        declaration: {
          id: probe.id,
          position: probe.capturePosition,
          shape: probe.shape === "sphere" ? "sphere" : undefined,
          ...(probe.shape === "sphere"
            ? {
                spherePosition: probe.spherePosition,
                sphereRadius: probe.sphereRadius,
              }
            : {
                boxPosition: probe.boxPosition,
                boxSize: probe.boxSize,
                angle: probe.angle || 0,
              }),
          resolution: probeResolution(),
          hash: await digest(probeDigestSource(probe, meshes)),
        },
      });
    }
    const byId = new Map(probes.map((p) => [p.probe.id, p]));
    if (only && !byId.has(only)) throw new Error(`${only} is not an authored probe`);

    const response = await fetchWithDeadline("/api/local-environments", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ probes: probes.map((p) => p.declaration), force: only ? false : force }),
    }, "the probe list");
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || `local environments: HTTP ${response.status}`);

    // One named probe, or whatever the server says still owes a capture.
    const queue = only ? [only] : (status.pending || []).map((entry) => entry.id);
    let bytes = 0;
    for (let i = 0; i < queue.length; i++) {
      const id = queue[i];
      const target = byId.get(id);
      // The server only ever reports back what was just declared, so a miss
      // here is a bug rather than a stale row - say so instead of skipping.
      if (!target) throw new Error(`${id} is pending but was not declared`);
      setBusyMessage(`capturing environment probe ${id} (${i + 1}/${queue.length})…`);
      onProgress(i, queue.length, id);
      bytes += await upload(id, target.declaration.hash, await captureProbe(target.probe, target.meshes));
    }
    onProgress(queue.length, queue.length, null);
    return { converted: queue.length, bytes };
  }));
}
