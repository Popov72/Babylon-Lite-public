// Box demo — a closed glass tank sitting on the ground (its floor replaces the
// ground plane). The X/Z footprint is resizable live; a rotating paddle can stir
// the fluid. Camera + interaction are core-owned: LMB orbits, RMB slides,
// Shift+RMB pushes the fluid, and the wheel zooms.

import { addToScene, createBox, createStandardMaterial, setMeshVisible } from "babylon-lite";
import type { Mesh } from "babylon-lite";
import type { SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import type { FluidCtx, FluidDemo } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";

// Box footprint (X/Z resizable via the size slider; height fixed so the box
// stays inside the sim grid bounds and needs no realloc).
const BOX_HALF_BASE = 4.5; // half-footprint at scale 1
// The MLS grid domain is fixed at ±20 (fluid.ts BOUNDS). Keep the box walls a margin
// inside it: past the grid edge, fluid pressed against the SDF wall stacks at the grid's
// 2-cell domain border. The box-size slider max (4.2×) is set so 4.5×max ≈ 18.9 < this.
const BOX_HALF_LIMIT = 19;
const BOX_MIN: [number, number, number] = [-BOX_HALF_BASE, 0, -BOX_HALF_BASE];
const BOX_MAX: [number, number, number] = [BOX_HALF_BASE, 22.5, BOX_HALF_BASE];

// Seed column sized to fall inside the (closed) box height.
const BOX_SPAWN_MIN: [number, number, number] = [-2, 1.5, -2];
const BOX_SPAWN_MAX: [number, number, number] = [2, 7.5, 2];

// Rotating paddle obstacle: a thin, tall vertical slab spinning about the tank's
// vertical axis. OBS_HALF_WIDTH leaves a gap to the walls so fluid flows around
// the blade ends. The tank is centred on the XZ origin.
const OBS_HALF_WIDTH_BASE = 3.0;
const OBS_HALF_THICK = 0.25;
const OBS_CENTER: [number, number] = [(BOX_MIN[0] + BOX_MAX[0]) / 2, (BOX_MIN[2] + BOX_MAX[2]) / 2];

export function createBoxDemo(ctx: FluidCtx): FluidDemo {
    const { engine } = ctx;

    // Box + rotating paddle. The paddle is a subtracted vertical slab whose angle
    // advances by omega·dt inside the SDF, so the moving-boundary velocity comes
    // from -∂sceneSdf/∂t (the sim's resolve handles it). obs0 = (pivotX, pivotZ,
    // halfWidth, halfThickness); obs1 = (cosA, sinA, omega, active).
    const sdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { lo: vec4<f32>, hi: vec4<f32>, obs0: vec4<f32>, obs1: vec4<f32>, };",
        sdf: `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let d1 = pt - sceneSdfParams.lo.xyz;
    let d2 = sceneSdfParams.hi.xyz - pt;
    var d = min(min(min(d1.x, d1.y), d1.z), min(min(d2.x, d2.y), d2.z));
    if (sceneSdfParams.obs1.w > 0.5) {
        let cosA = sceneSdfParams.obs1.x;
        let sinA = sceneSdfParams.obs1.y;
        let cd = cos(sceneSdfParams.obs1.z * dt);
        let sd = sin(sceneSdfParams.obs1.z * dt);
        let c = cosA * cd + sinA * sd;
        let s = sinA * cd - cosA * sd;
        let rx = pt.x - sceneSdfParams.obs0.x;
        let rz = pt.z - sceneSdfParams.obs0.y;
        let lx = c * rx + s * rz;
        let lz = -s * rx + c * rz;
        // Paddle collision spans floor..mid-height, matching the half-height visible mesh
        // (a finite 3D box, not an infinite vertical slab). Bounds derive from the box
        // lo.y / hi.y already in the params, so no extra obstacle fields are needed.
        let pTop = (sceneSdfParams.lo.y + sceneSdfParams.hi.y) * 0.5;
        let pYc = (sceneSdfParams.lo.y + pTop) * 0.5;
        let pYh = (pTop - sceneSdfParams.lo.y) * 0.5;
        let q = vec3<f32>(abs(lx) - sceneSdfParams.obs0.w, abs(pt.y - pYc) - pYh, abs(lz) - sceneSdfParams.obs0.z);
        let outside = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        d = min(d, outside);
    }
    return d;
}`,
        buffer: ctx.sceneSdfBuffer,
    };

    // Glass tank (transparent, blends, skips depth writes so the liquid stays
    // visible through it).
    const glass = createStandardMaterial();
    glass.diffuseColor = [0.5, 0.66, 0.82];
    glass.specularColor = [0.8, 0.85, 0.95];
    glass.alpha = 0.15;
    const boxMesh = createBox(engine, 1);
    boxMesh.material = glass;
    boxMesh.scaling.set(BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], BOX_MAX[2] - BOX_MIN[2]);
    boxMesh.position.set((BOX_MIN[0] + BOX_MAX[0]) / 2, (BOX_MIN[1] + BOX_MAX[1]) / 2, (BOX_MIN[2] + BOX_MAX[2]) / 2);
    addToScene(ctx.scene, boxMesh);
    setMeshVisible(boxMesh, false);
    let containerVisible = true; // toggled by the "Show container mesh" UI checkbox

    const paddleMat = createStandardMaterial();
    paddleMat.diffuseColor = [0.86, 0.5, 0.2];
    paddleMat.specularColor = [0.35, 0.35, 0.35];
    const paddleMesh = createBox(engine, 1);
    paddleMesh.material = paddleMat;
    let obsHalfWidth = OBS_HALF_WIDTH_BASE; // box footprint half-reach (scales with the box)
    // The paddle MESH is a half-height blade anchored on the floor (spans BOX_MIN[1]
    // up to the box mid-height). The collision SDF is an infinite-height vertical
    // slab (no Y extent — see writeScenePaddle/sceneSdf), so shortening the mesh
    // leaves the fluid physics unchanged (the liquid pools at the bottom anyway).
    paddleMesh.scaling.set(2 * OBS_HALF_THICK, 0.5 * (BOX_MAX[1] - BOX_MIN[1]), 2 * obsHalfWidth);
    paddleMesh.position.set(OBS_CENTER[0], BOX_MIN[1] + 0.25 * (BOX_MAX[1] - BOX_MIN[1]), OBS_CENTER[1]);
    addToScene(ctx.scene, paddleMesh);
    setMeshVisible(paddleMesh, false);

    // Current box-size multiplier (mirrors the box-size slider); tracked so the pair
    // state can snapshot/restore it without reading the DOM.
    let boxScale = 1;

    // Paddle state: `obstacleAngle` accumulates from `obstacleSpeed` (rad/s).
    let obstacleOn = false;
    let obstacleSpeed = 1.2;
    let obstacleAngle = 0;

    const updatePaddleGeometry = (): void => {
        paddleMesh.scaling.set(2 * OBS_HALF_THICK, 0.5 * (BOX_MAX[1] - BOX_MIN[1]), 2 * obsHalfWidth);
        paddleMesh.position.set(OBS_CENTER[0], BOX_MIN[1] + 0.25 * (BOX_MAX[1] - BOX_MIN[1]), OBS_CENTER[1]);
    };
    const updatePaddleVisibility = (): void => {
        setMeshVisible(paddleMesh, obstacleOn);
        updatePaddleGeometry();
    };

    // Refresh the rotating-paddle sub-block (obs0/obs1) of the box SDF.
    const writeScenePaddle = (): void => {
        engine._device.queue.writeBuffer(
            ctx.sceneSdfBuffer,
            32,
            new Float32Array([
                OBS_CENTER[0],
                OBS_CENTER[1],
                obsHalfWidth,
                OBS_HALF_THICK, // obs0
                Math.cos(obstacleAngle),
                -Math.sin(obstacleAngle),
                obstacleOn ? obstacleSpeed : 0,
                obstacleOn ? 1 : 0, // obs1
            ])
        );
    };
    const writeSdfParams = (): void => {
        engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, new Float32Array([BOX_MIN[0], BOX_MIN[1], BOX_MIN[2], 0, BOX_MAX[0], BOX_MAX[1], BOX_MAX[2], 0]));
        writeScenePaddle();
    };

    // Resize the box footprint live (X/Z only; height fixed). Mutates BOX_MIN/MAX
    // in place so every later scene-SDF write picks up the current size. No reset:
    // the solver's per-step clamp lets the fluid flow into the new shape and the
    // grid bounds are unchanged (no realloc).
    const applyBoxScale = (scale: number): void => {
        boxScale = scale;
        const half = Math.min(BOX_HALF_BASE * scale, BOX_HALF_LIMIT); // clamp inside the grid
        BOX_MIN[0] = -half;
        BOX_MIN[2] = -half;
        BOX_MAX[0] = half;
        BOX_MAX[2] = half;
        boxMesh.scaling.set(BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], BOX_MAX[2] - BOX_MIN[2]);
        obsHalfWidth = OBS_HALF_WIDTH_BASE * scale;
        updatePaddleGeometry();
        writeSdfParams();
    };

    // ── Extra panel controls (box size slider + paddle toggle/speed) ──────
    const boxSizeRow = document.createElement("div");
    boxSizeRow.style.cssText = "margin:2px 0 8px;";
    const boxSizeHead = document.createElement("div");
    boxSizeHead.style.cssText = "display:flex;justify-content:space-between;";
    const boxSizeLab = document.createElement("span");
    boxSizeLab.textContent = "Box size (box only)";
    const boxSizeVal = document.createElement("span");
    boxSizeVal.style.cssText = "color:#9fb4cc;";
    boxSizeVal.textContent = "1.0×";
    boxSizeHead.append(boxSizeLab, boxSizeVal);
    const boxSizeInput = document.createElement("input");
    boxSizeInput.type = "range";
    boxSizeInput.min = "0.5";
    boxSizeInput.max = "4.2";
    boxSizeInput.step = "0.1";
    boxSizeInput.value = "1";
    boxSizeInput.style.cssText = "width:100%;";
    boxSizeInput.oninput = () => {
        const s = parseFloat(boxSizeInput.value);
        boxSizeVal.textContent = `${s.toFixed(1)}×`;
        applyBoxScale(s);
    };
    boxSizeRow.append(boxSizeHead, boxSizeInput);

    const obstacleTitle = document.createElement("div");
    obstacleTitle.textContent = "Obstacle";
    obstacleTitle.style.cssText = "font-weight:600;margin:8px 0 6px;";
    const obstacleRow = document.createElement("label");
    obstacleRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:6px;cursor:pointer;";
    const obstacleChk = document.createElement("input");
    obstacleChk.type = "checkbox";
    const obstacleChkText = document.createElement("span");
    obstacleChkText.textContent = "Rotating paddle";
    obstacleRow.append(obstacleChk, obstacleChkText);
    const speedRow = document.createElement("div");
    speedRow.style.cssText = "margin:2px 0 4px;";
    const speedHead = document.createElement("div");
    speedHead.style.cssText = "display:flex;justify-content:space-between;";
    const speedLab = document.createElement("span");
    speedLab.textContent = "Paddle speed (rad/s)";
    const speedVal = document.createElement("span");
    speedVal.style.cssText = "color:#9fb4cc;";
    speedVal.textContent = String(obstacleSpeed);
    speedHead.append(speedLab, speedVal);
    const speedInput = document.createElement("input");
    speedInput.type = "range";
    speedInput.min = "0";
    speedInput.max = "15";
    speedInput.step = "0.1";
    speedInput.value = String(obstacleSpeed);
    speedInput.style.cssText = "width:100%;";
    speedInput.oninput = () => {
        obstacleSpeed = parseFloat(speedInput.value);
        speedVal.textContent = speedInput.value;
    };
    obstacleChk.onchange = () => {
        obstacleOn = obstacleChk.checked;
        updatePaddleVisibility();
    };
    speedRow.append(speedHead, speedInput);

    return {
        key: "box",
        label: "Box (closed)",
        envUrl: ENV_STUDIO_URL,
        sdf,
        writeSdfParams,
        spawn() {
            return { min: BOX_SPAWN_MIN, max: BOX_SPAWN_MAX };
        },
        emitters() {
            return null;
        },
        onEnter(): void {
            setMeshVisible(boxMesh, containerVisible);
            updatePaddleVisibility();
            setMeshVisible(ctx.ground, false); // the box floor replaces the ground
        },
        onLeave(): void {
            setMeshVisible(boxMesh, false);
            setMeshVisible(paddleMesh, false);
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            setMeshVisible(boxMesh, v);
        },
        containerMeshes(): Mesh[] {
            // Only the translucent glass tank is drawn as a post-fluid overlay (see
            // fluid.ts). The paddle is OPAQUE — it writes depth and must stay in the
            // scene-colour pass so the fluid surface is correctly occluded by it.
            return [boxMesh];
        },
        update(dt: number): void {
            if (obstacleOn) {
                obstacleAngle += obstacleSpeed * dt;
                paddleMesh.rotation.y = obstacleAngle;
            }
            writeScenePaddle();
        },
        demoParams() {
            return [];
        },
        applyParam(): void {
            /* box uses extraControls, not demo params */
        },
        extraControls() {
            return [boxSizeRow, obstacleTitle, obstacleRow, speedRow];
        },
        snapshotState(): Record<string, number | boolean> {
            return { boxSize: boxScale, paddleOn: obstacleOn, paddleSpeed: obstacleSpeed };
        },
        restoreState(state: Record<string, number | boolean>): void {
            if (typeof state.boxSize === "number") {
                applyBoxScale(state.boxSize);
                boxSizeInput.value = String(state.boxSize);
                boxSizeVal.textContent = `${state.boxSize.toFixed(1)}×`;
            }
            if (typeof state.paddleOn === "boolean") {
                obstacleOn = state.paddleOn;
                obstacleChk.checked = obstacleOn;
            }
            if (typeof state.paddleSpeed === "number") {
                obstacleSpeed = state.paddleSpeed;
                speedInput.value = String(obstacleSpeed);
                speedVal.textContent = String(obstacleSpeed);
            }
            updatePaddleVisibility();
        },
        presets: {
            // First-visit look for the box tank driven by MLS-MPM: a large, dense pool
            // of blue-tinted water with screen-space foam. Mirrors an exported config
            // (render + foam become per-pair state, restored on switch).
            "MLS-MPM": {
                schema: {
                    gravity: 40.6,
                    stiffness: 80,
                    viscosity: 0.03,
                    restDensity: 1,
                    damping: 0.998,
                    affineDamping: 0.7,
                    groundDamp: 0.84,
                    groundDampHeight: 4.7,
                    restitution: 1,
                    substeps: 2,
                },
                physScale: 0.6,
                count: 500000,
                camera: { alpha: -1.584095013664279, beta: 1.3202397371130101, radius: 41.26464332529501 },
                demoState: { boxSize: 3, paddleOn: true, paddleSpeed: 2.1 },
                showContainer: false,
                renderMode: "surface",
                color: "#16a3c3",
                half: true,
                thicknessDownscale: 6,
                absorption: 0.4,
                size: 0.7,
                refraction: 0.06,
                specular: 41,
                depthBlur: 40,
                depthBlurThreshold: 41,
                thicknessBlur: 16,
                surfaceFilter: "narrowRange",
                narrowDelta: 10,
                narrowMu: 1,
                foam: {
                    enabled: true,
                    kTa: 45,
                    kWc: 40,
                    kb: 0.15,
                    kd: 0.25,
                    tMin: 0.3,
                    tMax: 2,
                    poolScale: 3,
                    blurRadius: 0,
                    lightIntensity: 0.2,
                    ambient: 1,
                    aoStrength: 0.12,
                    normalStrength: 2.5,
                    debugTexture: "off",
                    softness: 0,
                    density: 50,
                    subsurfaceStrength: 0.05,
                },
            },
            // First-visit look for the box tank driven by PBF (SPH): a smaller, denser
            // tank with strong absorption + foam. Mirrors an exported config.
            "PBF": {
                schema: {
                    gravity: 35,
                    viscosity: 0.09,
                    relaxation: 44,
                    scorr: 0.025,
                    iterations: 2,
                    restDensity: 340,
                    boundaryDensity: 0,
                },
                physScale: 1,
                count: 80000,
                camera: { alpha: -3.1572752231998966, beta: 1.2743359109088446, radius: 29.574576603965223 },
                demoState: { boxSize: 2.1, paddleOn: true, paddleSpeed: 2.1 },
                showContainer: false,
                renderMode: "surface",
                color: "#16a3c3",
                half: true,
                thicknessDownscale: 5,
                absorption: 0.9,
                size: 1,
                refraction: 0.05,
                specular: 171,
                depthBlur: 48,
                depthBlurThreshold: 16,
                thicknessBlur: 3,
                foam: {
                    enabled: true,
                    kTa: 40,
                    kWc: 40,
                    kb: 0.15,
                    kd: 0.25,
                    tMin: 0.3,
                    tMax: 2,
                    poolScale: 3,
                    blurRadius: 0,
                    lightIntensity: 0.2,
                    ambient: 1,
                    aoStrength: 0.12,
                    normalStrength: 2.5,
                    debugTexture: "off",
                    softness: 0,
                    density: 50,
                    subsurfaceStrength: 0.05,
                },
            },
        },
    };
}
