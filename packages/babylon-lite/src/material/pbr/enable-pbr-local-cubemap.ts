/**
 * Published opt-in for PBR box-projected local cubemap reflections.
 *
 * Usage:
 * ```ts
 * await enablePbrLocalCubemap();
 * environment.boundingBoxPosition = [0, 1, 0];
 * environment.boundingBoxSize = [6, 2, 4];
 * material.localEnvironment = environment;
 * ```
 */

import { _registerPbrExt } from "./pbr-flags.js";
import { _installPbrLocalEnvironmentResolver } from "./pbr-pipeline.js";

let _enabled: Promise<void> | null = null;

/**
 * Enable box projection for bounded PBR local environments. Idempotent; await
 * this before registerScene so the extension is available during composition.
 */
export function enablePbrLocalCubemap(): Promise<void> {
    return (_enabled ??= import("./fragments/local-cubemap-fragment.js").then((mod) => {
        _installPbrLocalEnvironmentResolver((material) => material.localEnvironment);
        mod.registerPbrLocalCubemapExt(_registerPbrExt);
    }));
}
