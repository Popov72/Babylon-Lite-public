// Focused regression for independently animated placements in the exported GLB.

import { createRequire } from "node:module";
import { toolUrl } from "./target.mjs";

const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
    if (m.type() === "error" && !/Failed to load resource/i.test(m.text())) {
        errors.push(m.text());
    }
});

try {
    await page.goto(toolUrl(), { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => document.querySelectorAll("#palette-list .item").length > 0, null, { timeout: 60000 });

    const out = await page.evaluate(async () => {
        const ed = await import("/js/editor.js");
        const mf = await import("/js/manifest.js");
        const moduleId = "Modular SciFi MegaKit/Props/Prop_Fan_Small";
        ed.clearAll();
        ed.select([]);

        const first = await ed.placeAt(moduleId, new BABYLON.Vector3(0, 0, 0), { silent: true });
        const second = await ed.placeAt(moduleId, new BABYLON.Vector3(4, 0, 0), { silent: true });
        ed.renamePlacement(first.id, "FanOne");
        ed.renamePlacement(second.id, "FanTwo");

        const placements = [first, second].map((placement) => ({
            groups: placement.node._shipAnimationGroups?.map((group) => group.name) ?? [],
            skeletons: placement.node._shipSkeletons?.length ?? 0,
        }));
        const sceneCountsBefore = {
            skeletons: ed.state.scene.skeletons.length,
            groups: ed.state.scene.animationGroups.length,
        };

        const realFetch = window.fetch;
        let body = null;
        window.fetch = (url, opts) => {
            if (String(url).includes("/api/export")) {
                body = opts.body;
                return Promise.resolve(new Response('{"ok":true,"bytes":0}', { status: 200, headers: { "Content-Type": "application/json" } }));
            }
            return realFetch(url, opts);
        };
        try {
            await mf.exportGlb();
        } finally {
            window.fetch = realFetch;
        }

        const buffer = await body.arrayBuffer();
        const view = new DataView(buffer);
        const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, view.getUint32(12, true))));

        // The BIN chunk, walked properly rather than assumed: the inverse-bind
        // matrices live there, and without them a skin cannot be resolved.
        let bin = null;
        for (let off = 12; off + 8 <= buffer.byteLength; ) {
            const len = view.getUint32(off, true);
            if (view.getUint32(off + 4, true) === 0x004e4942) {
                bin = new DataView(buffer, off + 8, len);
                break;
            }
            off += 8 + len + ((4 - (len % 4)) % 4);
        }

        const nodeName = (index) => json.nodes?.[index]?.name;
        const animations = (json.animations ?? []).map((animation) => ({
            name: animation.name,
            targets: animation.channels.map((channel) => nodeName(channel.target.node)),
            missingTargets: animation.channels.filter((channel) => channel.target.node === undefined).length,
        }));
        const skins = (json.skins ?? []).map((skin) => skin.joints.map(nodeName));

        // How a skin actually resolves, which is the one thing the node graph
        // alone will not tell you.
        //
        // glTF ignores the skinned mesh node's own transform and poses the mesh
        // by the joint's *global* transform, so mesh and joints have to agree on
        // which way round the world is. When they do not, every rest matrix comes
        // out a mirror - determinant -1 - and the geometry is reflected with its
        // normals inverted, which reads as a model lit from the wrong side.
        // Babylon renders the same scene from the bone hierarchy and never
        // notices, so this is the only place the disagreement is visible.
        const { Matrix, Vector3, Quaternion } = BABYLON;
        const parentOf = new Map();
        (json.nodes ?? []).forEach((n, i) => (n.children ?? []).forEach((c) => parentOf.set(c, i)));
        const localOf = (n) =>
            n.matrix
                ? Matrix.FromArray(n.matrix)
                : Matrix.Compose(
                      Vector3.FromArray(n.scale ?? [1, 1, 1]),
                      Quaternion.FromArray(n.rotation ?? [0, 0, 0, 1]),
                      Vector3.FromArray(n.translation ?? [0, 0, 0])
                  );
        const globalOf = (i) => {
            let m = localOf(json.nodes[i]);
            for (let p = parentOf.get(i); p !== undefined; p = parentOf.get(p)) {
                m = m.multiply(localOf(json.nodes[p]));
            }
            return m;
        };
        const ibmOf = (skin, k) => {
            if (skin.inverseBindMatrices === undefined) return Matrix.Identity();
            const acc = json.accessors[skin.inverseBindMatrices];
            const bv = json.bufferViews[acc.bufferView];
            const base = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0) + k * 64;
            const values = [];
            for (let j = 0; j < 16; j++) values.push(bin.getFloat32(base + j * 4, true));
            return Matrix.FromArray(values);
        };

        const rest = [];
        const IDENTITY = [...Matrix.Identity().m];
        (json.skins ?? []).forEach((skin, si) => {
            const meshNodes = (json.nodes ?? []).map((n, i) => [n, i]).filter(([n]) => n.skin === si);
            for (const [, mi] of meshNodes) {
                const meshGlobal = globalOf(mi);
                const meshDet = meshGlobal.determinant();
                skin.joints.forEach((ji, k) => {
                    const jointGlobal = globalOf(ji);
                    const m = ibmOf(skin, k).multiply(jointGlobal).multiply(Matrix.Invert(meshGlobal));
                    rest.push({
                        mesh: nodeName(mi),
                        joint: nodeName(ji),
                        meshDet,
                        jointDet: jointGlobal.determinant(),
                        det: m.determinant(),
                        maxOffIdentity: Math.max(...[...m.m].map((v, idx) => Math.abs(v - IDENTITY[idx]))),
                    });
                });
            }
        });

        // The ancestor that carries the module's basis down to its joints.
        const jointAncestors = (json.skins ?? []).map((skin) => {
            const names = new Set();
            for (const ji of skin.joints) {
                for (let p = parentOf.get(ji); p !== undefined; p = parentOf.get(p)) names.add(nodeName(p));
            }
            return [...names];
        });

        const sceneCountsAfter = {
            skeletons: ed.state.scene.skeletons.length,
            groups: ed.state.scene.animationGroups.length,
        };

        ed.clearAll();
        ed.select([]);
        return { placements, animations, skins, rest, jointAncestors, sceneCountsBefore, sceneCountsAfter };
    });

    const failures = [];
    const check = (condition, message) => {
        if (!condition) {
            failures.push(message);
        }
    };
    check(
        out.placements.every((placement) => placement.skeletons === 1 && placement.groups.join() === "Fan_Idle"),
        `placements do not own independent animation state: ${JSON.stringify(out.placements)}`
    );
    check(
        out.animations.length === 2 && out.animations.every((animation) => animation.name === "Fan_Idle" && animation.missingTargets === 0),
        `exported animations are incomplete: ${JSON.stringify(out.animations)}`
    );
    check(
        out.animations.some(
            (animation) =>
                animation.targets.includes("FanOne_Root") && animation.targets.includes("FanOne_Propeller") && animation.targets.every((name) => name?.startsWith("FanOne_"))
        ),
        `FanOne targets are not isolated: ${JSON.stringify(out.animations)}`
    );
    check(
        out.animations.some(
            (animation) =>
                animation.targets.includes("FanTwo_Root") && animation.targets.includes("FanTwo_Propeller") && animation.targets.every((name) => name?.startsWith("FanTwo_"))
        ),
        `FanTwo targets are not isolated: ${JSON.stringify(out.animations)}`
    );
    check(
        out.skins.length === 2 && ["FanOne_", "FanTwo_"].every((prefix) => out.skins.some((joints) => joints.length > 0 && joints.every((name) => name?.startsWith(prefix)))),
        `exported skins are not placement-local: ${JSON.stringify(out.skins)}`
    );
    check(JSON.stringify(out.sceneCountsBefore) === JSON.stringify(out.sceneCountsAfter), `export changed the editor scene arrays: ${JSON.stringify(out)}`);

    // The bug this guards: `getProto` bakes the loader's right-to-left-handed
    // conversion into every mesh part, so a cloned mesh is mirrored. Cloned
    // joints were parented straight to the placement root and never got it, so
    // the two halves of a fan disagreed. glTF poses a skin from the joint's
    // global transform and ignores the mesh node's own, which turned that
    // disagreement into a rest pose with determinant -1: geometry reflected and
    // every skinned normal inverted, so the fans were lit from the wrong side in
    // BLite while looking perfectly fine in the editor.
    check(out.rest.length > 0, "no skin/joint pairs were resolved from the exported GLB");
    // Every joint of every primitive fails together, so the whole list says
    // nothing the first few do not.
    const sample = (predicate) => JSON.stringify(out.rest.filter(predicate).slice(0, 3));
    check(out.rest.every((r) => r.det > 0), `exported rest pose is mirrored - skinned normals will be inverted: ${sample((r) => r.det <= 0)}`);
    check(
        out.rest.every((r) => Math.abs(r.det - 1) < 1e-3),
        `exported rest pose is not unit scale: ${sample((r) => Math.abs(r.det - 1) >= 1e-3)}`
    );
    check(
        out.rest.every((r) => r.maxOffIdentity < 1e-3),
        `exported rest pose is not the authored pose: ${sample((r) => r.maxOffIdentity >= 1e-3)}`
    );
    check(
        out.rest.every((r) => Math.sign(r.meshDet) === Math.sign(r.jointDet)),
        `skinned mesh and its joints are in different coordinate bases: ${sample((r) => Math.sign(r.meshDet) !== Math.sign(r.jointDet))}`
    );
    check(
        out.jointAncestors.length === 2 && out.jointAncestors.every((names) => names.some((n) => /_Basis$/.test(n ?? ""))),
        `joints do not hang from a basis node carrying the module conversion: ${JSON.stringify(out.jointAncestors)}`
    );
    check(errors.length === 0, `browser errors: ${errors.join(" | ")}`);

    if (failures.length) {
        throw new Error(failures.join("\n"));
    }
    console.log("animated export: 2 independent skins and clips with complete node targets");
    console.log(`animated export: ${out.rest.length} rest matrices are the authored pose, unmirrored (det ~ +1)`);
} finally {
    await browser.close();
}
