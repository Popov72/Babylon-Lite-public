export function demoAssetUrl(path: string, moduleUrl: string): string {
    // Dev convenience: with LAB_DEMO_SRC=1 the demo page loads its TypeScript SOURCE instead of
    // the prebuilt bundle (so edits hot-reload). The source module then sits at an arbitrary
    // depth under /lite/src/demos/ — fluid.ts is one level down, fluid/scenes/waterfall.ts is
    // three — and resolving the asset RELATIVE to it would inherit that nesting. In dev the
    // assets are served from lab/public at the server root, which is exactly where the leading
    // "./" of every asset path points once anchored there, so resolve against the root instead.
    if (moduleUrl.includes("/lite/src/")) {
        return new URL(path.replace(/^\.\//, "/"), moduleUrl).href;
    }
    const url = new URL(path, moduleUrl);
    url.pathname = url.pathname.replace("/lite/bundle/demos/", "/bundle/demos/");
    return url.href;
}

/**
 * Point the glTF decoders (Draco + meshopt) at the demo-local decoder files,
 * resolved relative to the calling demo module's URL. This keeps deployed demos
 * working under ANY base path (e.g. /lite-demos/) instead of fetching the
 * decoders from the site root. Awaiting also retains `setMeshoptBaseUrl`, which
 * stops the bundler from constant-folding the meshopt loader's `script.src` into
 * a root-relative `/meshopt_decoder.js` literal.
 */
export async function configureDemoDecoderBases(moduleUrl: string): Promise<void> {
    const base = demoAssetUrl("./", moduleUrl);
    const [{ setDracoBaseUrl }, { setMeshoptBaseUrl }] = await Promise.all([
        import("babylon-lite/loader-gltf/draco-decode.js"),
        import("babylon-lite/loader-gltf/meshopt-decode.js"),
    ]);
    setDracoBaseUrl(base);
    setMeshoptBaseUrl(base);
}
