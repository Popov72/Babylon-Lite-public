import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";

import { measureAndPersistDemoManifestEntry, measureDemoManifestEntry } from "../../../scripts/bundle-demos-core.js";

describe("demo bundle measurements", () => {
    it("requires readiness for every configured lite demo measurement", async () => {
        const demos = JSON.parse(readFileSync(resolve(process.cwd(), "demos-config.json"), "utf-8")) as Array<{ slug: string }>;
        const measure = vi.fn(
            async (_browser: unknown, _port: number, _scene: string, _htmlFile: string, _bundlePath: string, requireReady?: boolean) => ({
                rawKB: requireReady ? 1 : 0,
                gzipKB: 0.5,
            })
        );

        for (const demo of demos) {
            await measureDemoManifestEntry({}, 4173, demo.slug, measure);
        }

        expect(demos.length).toBeGreaterThan(0);
        expect(measure.mock.calls.every((call) => call[5] === true)).toBe(true);
    });

    it("retains the previous manifest measurement when a remeasurement fails", async () => {
        const previous = { rawKB: 737.9, gzipKB: 245.1 };
        const manifest = { aquanova: previous };
        const persist = vi.fn();
        const failure = new Error("measurement failed");

        await expect(measureAndPersistDemoManifestEntry(manifest, "aquanova", async () => Promise.reject(failure), persist)).rejects.toBe(failure);

        expect(manifest.aquanova).toEqual(previous);
        expect(persist).not.toHaveBeenCalled();
    });
});
