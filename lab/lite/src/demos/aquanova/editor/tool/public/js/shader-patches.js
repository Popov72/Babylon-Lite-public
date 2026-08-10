/**
 * Patches the classic Babylon.js shader store used by the editor.
 *
 * ANGLE can produce tiny negative values in the Khronos PBR Neutral input on
 * some NVIDIA/WebGL combinations. Clamping the exposure-scaled value before
 * the tone mapper avoids the artifact without changing the tone-mapper curve.
 */
export function patchKhronosPbrNeutralShader() {
  const store = BABYLON.ShaderStore?.IncludesShadersStore;
  const source = store?.imageProcessingFunctions;
  if (typeof source !== "string") {
    console.warn("Khronos PBR Neutral shader patch skipped: Babylon shader include is unavailable");
    return false;
  }

  const clamp = "result.rgb = max(result.rgb, vec3(1e-7));";
  if (source.includes(clamp)) return true;

  const marker = /(#if TONEMAPPING\s*==\s*3\s*\r?\n)(\s*result\.rgb\s*=\s*PBRNeutralToneMapping\(result\.rgb\);)/;
  const patched = source.replace(marker, `$1\t${clamp}\n$2`);
  if (patched === source) {
    console.warn("Khronos PBR Neutral shader patch skipped: tone-mapper call was not found");
    return false;
  }

  store.imageProcessingFunctions = patched;
  return true;
}
