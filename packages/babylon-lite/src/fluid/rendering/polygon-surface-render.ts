import { getProjectionMatrix, getViewMatrix } from "../../camera/camera.js";
import type { Camera } from "../../camera/camera.js";
import type { EngineContext } from "../../engine/engine.js";
import type { RenderTarget } from "../../engine/render-target.js";
import { buildRenderTarget } from "../../engine/render-target.js";
import { invertMat4 as mat4Invert } from "../../math/invert-mat4.js";
import { packMat4IntoF32 } from "../../math/pack-mat4-into-f32.js";
import type { SceneContext } from "../../scene/scene-core.js";
import type { Task } from "../../frame-graph/task.js";
import { createDepthPyramid } from "../../frame-graph/depth-pyramid.js";
import type { DepthPyramid } from "../../frame-graph/depth-pyramid.js";
import type { Texture2D } from "../../texture/texture-2d.js";
import type { EnvMap } from "./fluid-surface-render.js";
import type { FluidPolygonSurface, FluidProfiler, FluidSim } from "../core/sim-common.js";

/** @internal */
export interface FluidPolygonSurfaceOptions {
    bgRT: RenderTarget;
    outRT: RenderTarget;
    depthRT: RenderTarget;
    camera: Camera;
    sim?: FluidSim;
}

export type FluidPolygonShading = "physical" | "ocean";

/** @internal */
export interface FluidPolygonSurfaceTask extends Task {
    setSim(sim: FluidSim): void;
    /** Render several simulations into one shared front-surface depth target and scene Hi-Z pyramid. */
    setSims(sims: readonly FluidSim[]): void;
    setEnabled(enabled: boolean): void;
    setOpacity(opacity: number): void;
    setFluidColor(color: [number, number, number]): void;
    setAbsorption(absorption: number): void;
    setRefractionStrength(strength: number): void;
    setSpecularPower(power: number): void;
    setDirLight(direction: [number, number, number]): void;
    setEnvMap(environment: EnvMap): void;
    setEnvRotationY(radians: number): void;
    setEnvReflection(exposure: number, contrast: number): void;
    setFresnelF0(value: number): void;
    setShadingMode(mode: FluidPolygonShading): void;
    /** Overlay the GPU-generated Surface Nets edges for reconstruction debugging. */
    setWireframe(enabled: boolean): void;
    setProfiler(profiler: FluidProfiler | null): void;
    surfaceDepthView(): GPUTextureView | null;
}

const POLYGON_SURFACE_WGSL = /* wgsl */ `
struct SurfaceUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverseView: mat4x4<f32>,
    colorAbsorption: vec4<f32>,
    lighting: vec4<f32>,
    render: vec4<f32>,
    environment: vec4<f32>,
};
struct GridUniforms {
    originDx: vec4<f32>,
    dimensionsMaxDistance: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: SurfaceUniforms;
@group(0) @binding(1) var background: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var environment: texture_cube<f32>;
@group(0) @binding(4) var environmentSampler: sampler;
@group(0) @binding(5) var sceneDepth: texture_depth_2d;
@group(0) @binding(6) var hiZDepth: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> liquidSdf: array<f32>;
@group(0) @binding(8) var<uniform> grid: GridUniforms;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) worldPosition: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) eyeDepth: f32,
};

@vertex fn vs(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>
) -> VertexOut {
    let viewPosition = u.view * vec4<f32>(position.xyz, 1.0);
    var out: VertexOut;
    out.clip = u.projection * viewPosition;
    out.worldPosition = position.xyz;
    out.worldNormal = normal.xyz;
    out.eyeDepth = viewPosition.z;
    return out;
}

@vertex fn wireVs(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>
) -> VertexOut {
    let viewPosition = u.view * vec4<f32>(position.xyz, 1.0);
    var out: VertexOut;
    out.clip = u.projection * viewPosition;
    // Reverse-Z: a small positive NDC offset keeps coplanar front edges in
    // front of the filled surface despite independent line rasterization.
    out.clip.z = min(out.clip.z + 1.0e-4 * out.clip.w, out.clip.w);
    out.worldPosition = position.xyz;
    out.worldNormal = normal.xyz;
    out.eyeDepth = viewPosition.z;
    return out;
}

struct SceneHit {
    uv: vec2<f32>,
    distance: f32,
    valid: f32,
};

struct LiquidHit {
    uv: vec2<f32>,
    exitViewPosition: vec3<f32>,
    exitWorldNormal: vec3<f32>,
    waterDistance: f32,
    kind: f32,
};

fn gridDimensions() -> vec3<i32> {
    return vec3<i32>(grid.dimensionsMaxDistance.xyz);
}

fn sdfIndex(c: vec3<i32>) -> u32 {
    let dimensions = gridDimensions();
    return u32(c.x + dimensions.x * (c.y + dimensions.y * c.z));
}

fn sampleLiquidSdf(worldPosition: vec3<f32>) -> f32 {
    let dimensions = gridDimensions();
    let gridPosition = (worldPosition - grid.originDx.xyz) / grid.originDx.w - vec3<f32>(0.5);
    let maximum = vec3<f32>(dimensions - vec3<i32>(1));
    if (any(gridPosition < vec3<f32>(0.0)) || any(gridPosition > maximum)) {
        return grid.originDx.w;
    }
    let base = clamp(vec3<i32>(floor(gridPosition)), vec3<i32>(0), dimensions - vec3<i32>(2));
    let fraction = clamp(gridPosition - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));
    let c000 = liquidSdf[sdfIndex(base)];
    let c100 = liquidSdf[sdfIndex(base + vec3<i32>(1, 0, 0))];
    let c010 = liquidSdf[sdfIndex(base + vec3<i32>(0, 1, 0))];
    let c110 = liquidSdf[sdfIndex(base + vec3<i32>(1, 1, 0))];
    let c001 = liquidSdf[sdfIndex(base + vec3<i32>(0, 0, 1))];
    let c101 = liquidSdf[sdfIndex(base + vec3<i32>(1, 0, 1))];
    let c011 = liquidSdf[sdfIndex(base + vec3<i32>(0, 1, 1))];
    let c111 = liquidSdf[sdfIndex(base + vec3<i32>(1, 1, 1))];
    let z0 = mix(mix(c000, c100, fraction.x), mix(c010, c110, fraction.x), fraction.y);
    let z1 = mix(mix(c001, c101, fraction.x), mix(c011, c111, fraction.x), fraction.y);
    return mix(z0, z1, fraction.z);
}

fn sampleLiquidSdfNearest(worldPosition: vec3<f32>) -> f32 {
    let dimensions = gridDimensions();
    let cell = vec3<i32>(floor((worldPosition - grid.originDx.xyz) / grid.originDx.w));
    if (any(cell < vec3<i32>(0)) || any(cell >= dimensions)) {
        return grid.originDx.w;
    }
    return liquidSdf[sdfIndex(cell)];
}

fn liquidSdfNormal(worldPosition: vec3<f32>) -> vec3<f32> {
    let h = max(0.5 * grid.originDx.w, 1.0e-4);
    let gradient = vec3<f32>(
        sampleLiquidSdf(worldPosition + vec3<f32>(h, 0.0, 0.0)) - sampleLiquidSdf(worldPosition - vec3<f32>(h, 0.0, 0.0)),
        sampleLiquidSdf(worldPosition + vec3<f32>(0.0, h, 0.0)) - sampleLiquidSdf(worldPosition - vec3<f32>(0.0, h, 0.0)),
        sampleLiquidSdf(worldPosition + vec3<f32>(0.0, 0.0, h)) - sampleLiquidSdf(worldPosition - vec3<f32>(0.0, 0.0, h)));
    return select(vec3<f32>(0.0, 1.0, 0.0), normalize(gradient), dot(gradient, gradient) > 1.0e-8);
}

fn viewToUv(viewPosition: vec3<f32>) -> vec3<f32> {
    let clip = u.projection * vec4<f32>(viewPosition, 1.0);
    if (clip.w <= 1.0e-5) {
        return vec3<f32>(-1.0, -1.0, 0.0);
    }
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

fn insideScreen(uv: vec2<f32>) -> bool {
    return all(uv >= vec2<f32>(0.001)) && all(uv <= vec2<f32>(0.999));
}

fn sceneEyeDepthAt(uv: vec2<f32>) -> f32 {
    let dimensions = vec2<i32>(textureDimensions(sceneDepth));
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), dimensions - vec2<i32>(1));
    let ndc = textureLoad(sceneDepth, pixel, 0);
    if (ndc <= 0.0) {
        return 1.0e20;
    }
    return u.projection[3].z / (ndc - u.projection[2].z);
}

fn sceneHitConfidence(
    previousViewPosition: vec3<f32>,
    previousUv: vec2<f32>,
    viewPosition: vec3<f32>,
    uv: vec2<f32>,
    maximumOvershoot: f32
) -> f32 {
    let previousSceneEyeDepth = sceneEyeDepthAt(previousUv);
    let sceneEyeDepth = sceneEyeDepthAt(uv);
    if (previousSceneEyeDepth >= 1.0e19 || sceneEyeDepth >= 1.0e19) {
        return 0.0;
    }
    let tolerance = max(0.08 * grid.originDx.w, 0.005);
    let previousDepthDelta = previousViewPosition.z - previousSceneEyeDepth;
    let depthDelta = viewPosition.z - sceneEyeDepth;
    if (previousDepthDelta > tolerance || depthDelta < -tolerance || depthDelta > maximumOvershoot) {
        return 0.0;
    }
    let minimumSceneDepth = min(previousSceneEyeDepth, sceneEyeDepth);
    let maximumSpan = max(3.0 * grid.originDx.w, 0.08 * minimumSceneDepth);
    return 1.0 - smoothstep(0.35 * maximumSpan, maximumSpan, abs(sceneEyeDepth - previousSceneEyeDepth));
}

fn refineSceneHit(originView: vec3<f32>, directionView: vec3<f32>, nearDistance: f32, farDistance: f32, confidence: f32) -> SceneHit {
    var low = nearDistance;
    var high = farDistance;
    for (var i = 0; i < 5; i = i + 1) {
        let middle = 0.5 * (low + high);
        let viewPosition = originView + directionView * middle;
        let projected = viewToUv(viewPosition);
        let sceneEyeDepth = sceneEyeDepthAt(projected.xy);
        if (insideScreen(projected.xy) && sceneEyeDepth < 1.0e19 && viewPosition.z >= sceneEyeDepth) {
            high = middle;
        } else {
            low = middle;
        }
    }
    let viewPosition = originView + directionView * high;
    let projected = viewToUv(viewPosition);
    let sceneEyeDepth = sceneEyeDepthAt(projected.xy);
    let valid = insideScreen(projected.xy) && sceneEyeDepth < 1.0e19;
    return SceneHit(projected.xy, high, select(0.0, confidence, valid));
}

fn sampleHiZLinear(uv: vec2<f32>, mip: i32) -> f32 {
    let dimensions = vec2<i32>(textureDimensions(hiZDepth, mip));
    let position = uv * vec2<f32>(dimensions) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(position));
    let fraction = fract(position);
    let maximum = dimensions - vec2<i32>(1);
    let p00 = clamp(base, vec2<i32>(0), maximum);
    let p10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), maximum);
    let p01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), maximum);
    let p11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), maximum);
    let z0 = mix(textureLoad(hiZDepth, p00, mip).r, textureLoad(hiZDepth, p10, mip).r, fraction.x);
    let z1 = mix(textureLoad(hiZDepth, p01, mip).r, textureLoad(hiZDepth, p11, mip).r, fraction.x);
    return mix(z0, z1, fraction.y);
}

fn traceOpaqueScene(originView: vec3<f32>, directionView: vec3<f32>, maxDistance: f32, rejectLiquid: bool) -> SceneHit {
    var distance = max(0.15 * grid.originDx.w, 0.005);
    var previousDistance = 0.0;
    var stepLength = max(0.3 * grid.originDx.w, 0.01);
    let maximumStep = max(1.5 * grid.originDx.w, 0.08);
    for (var i = 0; i < 48; i = i + 1) {
        if (distance >= maxDistance) {
            break;
        }
        let viewPosition = originView + directionView * distance;
        let projected = viewToUv(viewPosition);
        if (!insideScreen(projected.xy)) {
            break;
        }
        if (rejectLiquid) {
            let worldPosition = (u.inverseView * vec4<f32>(viewPosition, 1.0)).xyz;
            if (sampleLiquidSdfNearest(worldPosition) < -0.1 * grid.originDx.w) {
                return SceneHit(projected.xy, distance, 0.0);
            }
        }
        let previousViewPosition = originView + directionView * previousDistance;
        let previousProjected = viewToUv(previousViewPosition);
        let maximumOvershoot = max(2.0 * (distance - previousDistance), 0.5 * grid.originDx.w);
        let hitConfidence = sceneHitConfidence(
            previousViewPosition,
            previousProjected.xy,
            viewPosition,
            projected.xy,
            maximumOvershoot);
        if (hitConfidence > 0.0) {
            return refineSceneHit(originView, directionView, previousDistance, distance, hitConfidence);
        }

        // Reverse-Z Hi-Z stores the nearest opaque depth in each tile. Empty or
        // comfortably distant tiles allow a larger step without changing hits.
        // Linear mip sampling prevents the traversal rate from exposing the
        // pyramid's tile boundaries in high-reflectivity water.
        let mip = min(3, i32(max(u.render.w, 0.0)));
        let coarseNdc = sampleHiZLinear(projected.xy, mip);
        var acceleration = 1.35;
        if (coarseNdc <= 0.0) {
            acceleration = 1.5;
        } else {
            let coarseEyeDepth = u.projection[3].z / (coarseNdc - u.projection[2].z);
            let clearance = coarseEyeDepth - viewPosition.z;
            acceleration = mix(1.0, 1.35, smoothstep(2.0 * stepLength, 6.0 * stepLength, clearance));
        }
        previousDistance = distance;
        distance = distance + stepLength * acceleration;
        stepLength = min(stepLength * 1.12, maximumStep);
    }
    return SceneHit(vec2<f32>(0.0), distance, 0.0);
}

fn traceLiquid(originView: vec3<f32>, directionView: vec3<f32>) -> LiquidHit {
    let maximumDistance = grid.dimensionsMaxDistance.w;
    let entryUv = viewToUv(originView).xy;
    let entryWorld = (u.inverseView * vec4<f32>(originView, 1.0)).xyz;
    var distance = max(0.12 * grid.originDx.w, 0.003);
    var previousDistance = 0.0;
    var enteredLiquid = sampleLiquidSdf(entryWorld) <= 0.02 * grid.originDx.w;
    for (var i = 0; i < 48; i = i + 1) {
        if (distance >= maximumDistance) {
            break;
        }
        let viewPosition = originView + directionView * distance;
        let projected = viewToUv(viewPosition);
        if (!insideScreen(projected.xy)) {
            break;
        }
        let worldPosition = (u.inverseView * vec4<f32>(viewPosition, 1.0)).xyz;
        let phi = sampleLiquidSdf(worldPosition);
        enteredLiquid = enteredLiquid || phi <= 0.02 * grid.originDx.w;
        let previousViewPosition = originView + directionView * previousDistance;
        let previousProjected = viewToUv(previousViewPosition);
        let maximumOvershoot = max(2.0 * (distance - previousDistance), 0.5 * grid.originDx.w);
        let hitConfidence = sceneHitConfidence(
            previousViewPosition,
            previousProjected.xy,
            viewPosition,
            projected.xy,
            maximumOvershoot);
        if (enteredLiquid && hitConfidence > 0.5) {
            let hit = refineSceneHit(originView, directionView, previousDistance, distance, hitConfidence);
            if (hit.valid > 0.5) {
                return LiquidHit(hit.uv, viewPosition, vec3<f32>(0.0), hit.distance, 1.0);
            }
        }

        if (enteredLiquid && phi > 0.02 * grid.originDx.w && distance > 0.25 * grid.originDx.w) {
            var low = previousDistance;
            var high = distance;
            for (var j = 0; j < 5; j = j + 1) {
                let middle = 0.5 * (low + high);
                let middleView = originView + directionView * middle;
                let middleWorld = (u.inverseView * vec4<f32>(middleView, 1.0)).xyz;
                if (sampleLiquidSdf(middleWorld) > 0.0) {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            let exitView = originView + directionView * high;
            let exitWorld = (u.inverseView * vec4<f32>(exitView, 1.0)).xyz;
            return LiquidHit(viewToUv(exitView).xy, exitView, liquidSdfNormal(exitWorld), high, 2.0);
        }
        if (!enteredLiquid && distance > 0.75 * grid.originDx.w) {
            return LiquidHit(entryUv, originView, vec3<f32>(0.0), 0.0, 3.0);
        }

        previousDistance = distance;
        let stepLength = select(
            0.15 * grid.originDx.w,
            clamp(abs(phi) * 0.7, 0.2 * grid.originDx.w, 1.75 * grid.originDx.w),
            enteredLiquid);
        distance = distance + stepLength;
    }
    // An unresolved path must not sample an arbitrary far-away screen location.
    // Keep the original projection and retain only its conservative thickness.
    return LiquidHit(entryUv, originView + directionView * distance, vec3<f32>(0.0), distance, 0.0);
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) eyeDepth: vec4<f32>,
};

@fragment fn fs(input: VertexOut) -> FragmentOut {
    let dimensions = vec2<f32>(textureDimensions(background));
    let uv = input.clip.xy / dimensions;
    let pixel = vec2<i32>(floor(input.clip.xy));
    let sceneNdc = textureLoad(sceneDepth, pixel, 0);
    if (sceneNdc > 0.0) {
        let sceneEyeDepth = u.projection[3].z / (sceneNdc - u.projection[2].z);
        if (input.eyeDepth > sceneEyeDepth + 0.02) {
            discard;
        }
    }
    let cameraPosition = u.inverseView[3].xyz;
    let viewDirection = normalize(input.worldPosition - cameraPosition);
    let geometricNormal = cross(dpdx(input.worldPosition), dpdy(input.worldPosition));
    var normalCandidate = input.worldNormal;
    if (dot(normalCandidate, normalCandidate) <= 1.0e-8) {
        normalCandidate = geometricNormal;
    }
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    let normalLength = length(normalCandidate);
    if (normalLength > 1.0e-6) {
        normal = normalCandidate / normalLength;
    }
    if (dot(normal, viewDirection) > 0.0) {
        normal = -normal;
    }
    // Surface Nets shares SDF-gradient normals across adjacent triangles. Use
    // that continuous field for transport so refraction does not jump at every
    // triangle edge; the geometric normal remains only a degenerate fallback.
    let transportNormal = normal;
    let facing = clamp(dot(-viewDirection, normal), 0.0, 1.0);
    if (u.render.z > 0.5) {
        // Adapted from Babylon.js Playground YX6IB8#758. The ocean material is
        // opaque PBR water: it does not project the background through the mesh.
        let viewToCamera = -viewDirection;
        let viewDistance = length(input.worldPosition - cameraPosition);
        let referenceRoughness = 0.311;
        let distanceGloss = mix(
            1.0 - referenceRoughness,
            0.91,
            1.0 / (1.0 + viewDistance * 0.0044));
        let normalDx = dpdx(normal);
        let normalDy = dpdy(normal);
        let normalVariance = min(0.5 * (dot(normalDx, normalDx) + dot(normalDy, normalDy)), 0.5);
        let roughness = clamp(
            sqrt((1.0 - distanceGloss) * (1.0 - distanceGloss) + normalVariance),
            0.12,
            1.0);

        var reflectionDirection = reflect(viewDirection, normal);
        let environmentRotation = u.environment.z;
        if (environmentRotation != 0.0) {
            let cosine = cos(environmentRotation);
            let sine = sin(environmentRotation);
            reflectionDirection = vec3<f32>(
                reflectionDirection.x * cosine + reflectionDirection.z * sine,
                reflectionDirection.y,
                -reflectionDirection.x * sine + reflectionDirection.z * cosine);
        }
        let reflectionLinear = max(
            textureSampleLevel(
                environment,
                environmentSampler,
                vec3<f32>(reflectionDirection.x, reflectionDirection.y, -reflectionDirection.z),
                roughness * 5.0).rgb,
            vec3<f32>(0.0));
        var reflection = pow(
            reflectionLinear * max(u.environment.x, 0.0),
            vec3<f32>(1.0 / 2.2));
        reflection = clamp(reflection, vec3<f32>(0.0), vec3<f32>(1.0));
        let reflectionHighlight = reflection * reflection * (vec3<f32>(3.0) - 2.0 * reflection);
        reflection = mix(reflection, reflectionHighlight, max(u.environment.y, 0.0) - 1.0);

        let lightDirection = normalize(-u.lighting.xyz);
        let subsurfaceHalf = normalize(-normal + lightDirection);
        let subsurfaceView = pow(clamp(dot(viewToCamera, -subsurfaceHalf), 0.0, 1.0), 5.0) * 30.0 * 0.15;
        let splashCrest = 0.06 * smoothstep(0.1, 0.9, normal.y);
        // Ocean mode is opaque, so it needs a body term where Fresnel reflection is
        // weak. Keep that term neutral and let the authored water color define its hue.
        let bodyColor = u.colorAbsorption.rgb * 0.12;
        let subsurfaceColor = vec3<f32>(0.1541919, 0.8857628, 0.990566);
        var waterColor = clamp(
            bodyColor + subsurfaceColor * subsurfaceView * splashCrest,
            vec3<f32>(0.0),
            vec3<f32>(1.0));
        let oceanOpticalPath = mix(1.25, 4.0, 1.0 - facing);
        let oceanExtinction = max(
            vec3<f32>(1.0) - u.colorAbsorption.rgb,
            vec3<f32>(0.05));
        let oceanTransmittance = exp(
            -max(u.colorAbsorption.w, 0.0) *
            0.18 * oceanOpticalPath * oceanExtinction);
        waterColor = waterColor * oceanTransmittance;
        let displayWaterColor = pow(waterColor, vec3<f32>(1.0 / 2.2));
        let backgroundColor = textureSampleLevel(background, linearSampler, uv, 0.0).rgb;
        let transmittedWater =
            backgroundColor * oceanTransmittance +
            displayWaterColor * (vec3<f32>(1.0) - oceanTransmittance);

        let halfDirection = normalize(lightDirection + viewToCamera);
        let nDotL = max(dot(normal, lightDirection), 0.0);
        let nDotV = max(dot(normal, viewToCamera), 1.0e-4);
        let nDotH = max(dot(normal, halfDirection), 0.0);
        let vDotH = max(dot(viewToCamera, halfDirection), 0.0);
        let alphaRoughness = roughness * roughness;
        let alpha2 = alphaRoughness * alphaRoughness;
        let denominator = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
        let distribution = alpha2 / max(3.14159265 * denominator * denominator, 1.0e-5);
        let geometryK = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        let geometryV = nDotV / (nDotV * (1.0 - geometryK) + geometryK);
        let geometryL = nDotL / (nDotL * (1.0 - geometryK) + geometryK);
        let f0 = clamp(u.environment.w, 0.0, 1.0);
        let directFresnel = f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0);
        let specular = min(
            distribution * geometryV * geometryL * directFresnel /
            max(4.0 * nDotV * max(nDotL, 1.0e-4), 1.0e-4),
            1.25) * nDotL;
        let fresnel = clamp(f0 + (1.0 - f0) * pow(1.0 - facing, 5.0), 0.0, 1.0);
        let color = clamp(
            mix(transmittedWater, reflection, fresnel) + vec3<f32>(specular * 0.08),
            vec3<f32>(0.0),
            vec3<f32>(1.0));
        let alpha = clamp(u.render.y, 0.0, 1.0);
        var out: FragmentOut;
        out.color = vec4<f32>(mix(backgroundColor, color, alpha), 1.0);
        out.eyeDepth = vec4<f32>(input.eyeDepth, 0.0, 0.0, 1.0);
        return out;
    }
    let transportFacing = clamp(dot(-viewDirection, transportNormal), 0.0, 1.0);
    var physicalRefraction = refract(viewDirection, transportNormal, 1.0 / 1.333);
    if (dot(physicalRefraction, physicalRefraction) < 1.0e-6) {
        physicalRefraction = viewDirection;
    }
    let refractionDirection = normalize(mix(viewDirection, physicalRefraction, clamp(u.render.x, 0.0, 1.0)));
    let viewPosition = (u.view * vec4<f32>(input.worldPosition, 1.0)).xyz;
    let refractionView = normalize((u.view * vec4<f32>(refractionDirection, 0.0)).xyz);
    // The averaged Surface Nets vertex is not guaranteed to lie exactly on the
    // trilinear zero set. Advance a bounded distance along the refracted ray so
    // tracing starts inside the represented liquid instead of falling back to
    // unrelated opaque-scene samples when the raster surface is slightly out.
    let entryPhi = max(sampleLiquidSdf(input.worldPosition), 0.0);
    let entryFacing = max(-dot(refractionDirection, transportNormal), 0.25);
    let entryBias = clamp(
        (entryPhi + 0.08 * grid.originDx.w) / entryFacing,
        0.08 * grid.originDx.w,
        0.6 * grid.originDx.w);
    let traceWorldPosition = input.worldPosition + refractionDirection * entryBias;
    let traceViewPosition = (u.view * vec4<f32>(traceWorldPosition, 1.0)).xyz;
    let tracedLiquidHit = traceLiquid(traceViewPosition, refractionView);
    let liquidHit = LiquidHit(
        tracedLiquidHit.uv,
        tracedLiquidHit.exitViewPosition,
        tracedLiquidHit.exitWorldNormal,
        tracedLiquidHit.waterDistance + entryBias,
        tracedLiquidHit.kind);
    var refractedUv = liquidHit.uv;
    let refractionAmount = clamp(u.render.x, 0.0, 1.0);
    if (liquidHit.kind == 2.0 && refractionAmount > 1.0e-4) {
        // The opaque scene colour is a single completed projection, not geometry
        // that can be followed after the liquid exit. A second bend based on the
        // noisy exit SDF normal redirects adjacent pixels to unrelated scene
        // regions. Project the continuous entry ray over a bounded water segment.
        let boundedDistance = min(liquidHit.waterDistance, 4.0 * grid.originDx.w);
        let boundedExit = viewToUv(viewPosition + refractionView * boundedDistance);
        if (insideScreen(boundedExit.xy)) {
            refractedUv = boundedExit.xy;
        }
    }
    if (!insideScreen(refractedUv)) {
        refractedUv = uv;
    }
    let backgroundColor = textureSampleLevel(background, linearSampler, refractedUv, 0.0).rgb;
    // A reconstructed sheet represents at least one grid-scale liquid layer.
    // Project that unresolved layer along the view ray so coarse, curved
    // splashes do not alternate between transparent and deep cells.
    let minimumSheetThickness = clamp(
        0.75 * grid.originDx.w / max(transportFacing, 0.25),
        0.75 * grid.originDx.w,
        3.0 * grid.originDx.w);
    let thickness = max(liquidHit.waterDistance, minimumSheetThickness);
    let transmittance = exp(
        -max(u.colorAbsorption.w, 0.0) * thickness *
        max(vec3<f32>(1.0) - u.colorAbsorption.rgb, vec3<f32>(0.0)));
    let scatterAmount = vec3<f32>(1.0) - exp(
        -max(u.colorAbsorption.w, 0.0) * thickness *
        (vec3<f32>(0.12) + 0.18 * u.colorAbsorption.rgb));
    let transmitted = backgroundColor * transmittance +
        u.colorAbsorption.rgb * scatterAmount * 0.28;
    var reflectionDirection = reflect(viewDirection, normal);
    let reflectionView = normalize((u.view * vec4<f32>(reflectionDirection, 0.0)).xyz);
    let reflectionHit = traceOpaqueScene(
        viewPosition + reflectionView * max(0.12 * grid.originDx.w, 0.003),
        reflectionView,
        grid.dimensionsMaxDistance.w,
        true);
    let environmentRotation = u.environment.z;
    if (environmentRotation != 0.0) {
        let cosine = cos(environmentRotation);
        let sine = sin(environmentRotation);
        reflectionDirection = vec3<f32>(
            reflectionDirection.x * cosine + reflectionDirection.z * sine,
            reflectionDirection.y,
            -reflectionDirection.x * sine + reflectionDirection.z * cosine);
    }
    let baseRoughness = clamp(sqrt(2.0 / (max(u.lighting.w, 1.0) + 2.0)), 0.16, 1.0);
    let normalDx = dpdx(normal);
    let normalDy = dpdy(normal);
    let normalVariance = min(0.5 * (dot(normalDx, normalDx) + dot(normalDy, normalDy)), 0.5);
    let roughness = clamp(sqrt(baseRoughness * baseRoughness + normalVariance), 0.16, 1.0);
    let reflectionLinear = max(
        textureSampleLevel(
            environment,
            environmentSampler,
            vec3<f32>(reflectionDirection.x, reflectionDirection.y, -reflectionDirection.z),
            roughness * 5.0).rgb,
        vec3<f32>(0.0));
    let screenReflection = textureSample(background, linearSampler, reflectionHit.uv).rgb;
    let edgeDistance = min(min(reflectionHit.uv.x, 1.0 - reflectionHit.uv.x), min(reflectionHit.uv.y, 1.0 - reflectionHit.uv.y));
    let reflectionConfidence =
        reflectionHit.valid *
        smoothstep(0.0, 0.08, edgeDistance) *
        mix(0.55, 0.3, smoothstep(0.01, 0.15, normalVariance));
    let environmentReflection = pow(reflectionLinear * max(u.environment.x, 0.0), vec3<f32>(1.0 / 2.2));
    var reflection = mix(environmentReflection, screenReflection, reflectionConfidence);
    reflection = clamp(reflection, vec3<f32>(0.0), vec3<f32>(1.0));
    let reflectionHighlight = reflection * reflection * (vec3<f32>(3.0) - 2.0 * reflection);
    reflection = mix(reflection, reflectionHighlight, max(u.environment.y, 0.0) - 1.0);
    let f0 = clamp(u.environment.w, 0.0, 1.0);
    // Roughness-modified Schlick for IBL: unresolved grid-scale normal variance
    // must not become a binary white mirror at grazing angles. Smooth sheets
    // retain strong Fresnel while subpixel folds converge toward their F0.
    let grazingReflectance = max(1.0 - roughness, f0);
    let fresnel = clamp(f0 + (grazingReflectance - f0) * pow(1.0 - facing, 5.0), 0.0, 1.0);

    let lightDirection = normalize(-u.lighting.xyz);
    let viewToCamera = -viewDirection;
    let halfDirection = normalize(lightDirection + viewToCamera);
    let nDotL = max(dot(normal, lightDirection), 0.0);
    let nDotV = max(dot(normal, viewToCamera), 1.0e-4);
    let nDotH = max(dot(normal, halfDirection), 0.0);
    let vDotH = max(dot(viewToCamera, halfDirection), 0.0);
    let alphaRoughness = roughness * roughness;
    let alpha2 = alphaRoughness * alphaRoughness;
    let denominator = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
    let distribution = alpha2 / max(3.14159265 * denominator * denominator, 1.0e-5);
    let geometryK = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let geometryV = nDotV / (nDotV * (1.0 - geometryK) + geometryK);
    let geometryL = nDotL / (nDotL * (1.0 - geometryK) + geometryK);
    let directFresnel = f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0);
    let specular = min(
        distribution * geometryV * geometryL * directFresnel /
        max(4.0 * nDotV * max(nDotL, 1.0e-4), 1.0e-4),
        1.25) * nDotL;
    let color = clamp(
        mix(transmitted, reflection, fresnel) + vec3<f32>(specular * 0.08),
        vec3<f32>(0.0),
        vec3<f32>(1.0));
    let alpha = clamp(u.render.y, 0.0, 1.0);
    var out: FragmentOut;
    out.color = vec4<f32>(mix(backgroundColor, color, alpha), 1.0);
    out.eyeDepth = vec4<f32>(input.eyeDepth, 0.0, 0.0, 1.0);
    return out;
}

@fragment fn wireFs(input: VertexOut) -> FragmentOut {
    var out: FragmentOut;
    out.color = vec4<f32>(0.05, 0.95, 1.0, 1.0);
    out.eyeDepth = vec4<f32>(input.eyeDepth, 0.0, 0.0, 1.0);
    return out;
}
`;

function createPlaceholderCube(device: GPUDevice): GPUTextureView {
    const texture = device.createTexture({
        label: "fluid-polygon-env-placeholder",
        size: [1, 1, 6],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const pixel = new Uint8Array([110, 150, 210, 255]);
    for (let face = 0; face < 6; face++) {
        device.queue.writeTexture({ texture, origin: [0, 0, face] }, pixel, { bytesPerRow: 4 }, [1, 1, 1]);
    }
    return texture.createView({ dimension: "cube" });
}

/** @internal */
export function createFluidPolygonSurfaceTask(engine: EngineContext, scene: SceneContext, options: FluidPolygonSurfaceOptions): FluidPolygonSurfaceTask {
    const device = engine._device;
    const { bgRT, outRT, depthRT, camera } = options;
    let sims: readonly FluidSim[] = options.sim ? [options.sim] : [];
    let enabled = false;
    let opacity = 1;
    let color: [number, number, number] = [0.085, 0.6375, 0.765];
    let absorption = 1;
    let refractionStrength = 0.1;
    let specularPower = 250;
    let lightDirection: [number, number, number] = [-2, -1, 1];
    let environmentRotationY = 0;
    let environmentExposure = 1;
    let environmentContrast = 1.1;
    let fresnelF0 = 0.02;
    let shadingMode: FluidPolygonShading = "physical";
    let wireframe = false;
    let envView = createPlaceholderCube(device);
    let envSampler = device.createSampler({
        label: "fluid-polygon-env-sampler",
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
    });
    const linearSampler = device.createSampler({ label: "fluid-polygon-bg-sampler", magFilter: "linear", minFilter: "linear" });
    const uniformData = new Float32Array(64);
    const uniformBuffer = device.createBuffer({
        label: "fluid-polygon-uniforms",
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const shader = device.createShaderModule({ label: "fluid-polygon-surface", code: POLYGON_SURFACE_WGSL });
    const pipeline = device.createRenderPipeline({
        label: "fluid-polygon-surface",
        layout: "auto",
        vertex: {
            module: shader,
            entryPoint: "vs",
            buffers: [
                {
                    arrayStride: 32,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x4" },
                        { shaderLocation: 1, offset: 16, format: "float32x4" },
                    ],
                },
            ],
        },
        fragment: {
            module: shader,
            entryPoint: "fs",
            targets: [{ format: engine.format }, { format: "rg32float" }],
        },
        // Surface-net quads follow the simulation's left-handed outward winding.
        primitive: { topology: "triangle-list", frontFace: "cw", cullMode: "back" },
        depthStencil: {
            format: depthRT._descriptor.dFormat ?? "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "greater-equal",
        },
    });
    const wireframePipeline = device.createRenderPipeline({
        label: "fluid-polygon-wireframe",
        layout: "auto",
        vertex: {
            module: shader,
            entryPoint: "wireVs",
            buffers: [
                {
                    arrayStride: 32,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x4" },
                        { shaderLocation: 1, offset: 16, format: "float32x4" },
                    ],
                },
            ],
        },
        fragment: {
            module: shader,
            entryPoint: "wireFs",
            targets: [{ format: engine.format }, { format: "rg32float", writeMask: 0 }],
        },
        primitive: { topology: "line-list" },
        depthStencil: {
            format: depthRT._descriptor.dFormat ?? "depth24plus",
            depthWriteEnabled: false,
            depthCompare: "greater-equal",
        },
    });
    const wireframeBindGroup = device.createBindGroup({
        layout: wireframePipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });
    let depthPyramid: DepthPyramid | null = null;
    let profiler: FluidProfiler | null = null;
    let surfaceDepthTexture: GPUTexture | null = null;
    let surfaceDepthView: GPUTextureView | null = null;
    let surfaceDepthWidth = 0;
    let surfaceDepthHeight = 0;
    let frontZTexture: GPUTexture | null = null;
    let frontZView: GPUTextureView | null = null;
    let depthSourceView: GPUTextureView | null = null;
    let depthSource: Texture2D | null = null;
    interface SurfaceBinding {
        gridBuffer: GPUBuffer;
        bindGroup: GPUBindGroup | null;
        background: GPUTextureView | null;
        environment: GPUTextureView | null;
        sceneDepth: GPUTextureView | null;
        hiZDepth: GPUTextureView | null;
    }
    const surfaceBindings = new Map<FluidPolygonSurface, SurfaceBinding>();

    function ensureSurfaceDepth(): GPUTextureView {
        const width = engine.canvas.width;
        const height = engine.canvas.height;
        if (!surfaceDepthTexture || width !== surfaceDepthWidth || height !== surfaceDepthHeight) {
            surfaceDepthTexture?.destroy();
            frontZTexture?.destroy();
            surfaceDepthTexture = device.createTexture({
                label: "fluid-polygon-surface-depth",
                size: [width, height],
                format: "rg32float",
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });
            surfaceDepthView = surfaceDepthTexture.createView();
            frontZTexture = device.createTexture({
                label: "fluid-polygon-front-z",
                size: [width, height],
                format: depthRT._descriptor.dFormat ?? "depth24plus",
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
            frontZView = frontZTexture.createView();
            if (depthPyramid) {
                depthPyramid.resize(width, height);
            } else {
                depthPyramid = createDepthPyramid(engine, { width, height, reduce: "max" });
            }
            surfaceDepthWidth = width;
            surfaceDepthHeight = height;
            for (const binding of surfaceBindings.values()) {
                binding.bindGroup = null;
            }
        }
        return surfaceDepthView!;
    }

    function updateUniforms(): void {
        const view = getViewMatrix(camera);
        const projection = getProjectionMatrix(camera, engine.canvas.width / Math.max(1, engine.canvas.height));
        const inverseView = mat4Invert(view) ?? view;
        packMat4IntoF32(uniformData, view, 0);
        packMat4IntoF32(uniformData, projection, 16);
        packMat4IntoF32(uniformData, inverseView, 32);
        uniformData.set([color[0], color[1], color[2], absorption], 48);
        uniformData.set([lightDirection[0], lightDirection[1], lightDirection[2], specularPower], 52);
        uniformData.set([refractionStrength, opacity, shadingMode === "ocean" ? 1 : 0, (depthPyramid?.mipCount ?? 1) - 1], 56);
        uniformData.set([environmentExposure, environmentContrast, environmentRotationY, fresnelF0], 60);
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    }

    function getSurfaceBinding(surface: FluidPolygonSurface): SurfaceBinding {
        let binding = surfaceBindings.get(surface);
        if (!binding) {
            const gridData = new Float32Array(8);
            gridData.set([surface.gridOrigin[0], surface.gridOrigin[1], surface.gridOrigin[2], surface.gridSpacing], 0);
            const gridWidth = surface.gridDimensions[0] * surface.gridSpacing;
            const gridHeight = surface.gridDimensions[1] * surface.gridSpacing;
            const gridDepth = surface.gridDimensions[2] * surface.gridSpacing;
            const maxDistance = Math.sqrt(gridWidth * gridWidth + gridHeight * gridHeight + gridDepth * gridDepth) + surface.gridSpacing;
            gridData.set([surface.gridDimensions[0], surface.gridDimensions[1], surface.gridDimensions[2], maxDistance], 4);
            const gridBuffer = device.createBuffer({
                label: "fluid-polygon-grid-uniforms",
                size: gridData.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(gridBuffer, 0, gridData);
            binding = {
                gridBuffer,
                bindGroup: null,
                background: null,
                environment: null,
                sceneDepth: null,
                hiZDepth: null,
            };
            surfaceBindings.set(surface, binding);
        }
        return binding;
    }

    function pruneSurfaceBindings(activeSurfaces: ReadonlySet<FluidPolygonSurface>): void {
        for (const [surface, binding] of surfaceBindings) {
            if (!activeSurfaces.has(surface)) {
                binding.gridBuffer.destroy();
                surfaceBindings.delete(surface);
            }
        }
    }

    function getBindGroup(surface: FluidPolygonSurface, background: GPUTextureView, sceneDepth: GPUTextureView): GPUBindGroup {
        const binding = getSurfaceBinding(surface);
        const hiZDepth = depthPyramid!.texture.view;
        if (!binding.bindGroup || binding.background !== background || binding.environment !== envView || binding.sceneDepth !== sceneDepth || binding.hiZDepth !== hiZDepth) {
            binding.bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: background },
                    { binding: 2, resource: linearSampler },
                    { binding: 3, resource: envView },
                    { binding: 4, resource: envSampler },
                    { binding: 5, resource: sceneDepth },
                    { binding: 6, resource: hiZDepth },
                    { binding: 7, resource: { buffer: surface.liquidSdfBuffer } },
                    { binding: 8, resource: { buffer: binding.gridBuffer } },
                ],
            });
            binding.background = background;
            binding.environment = envView;
            binding.sceneDepth = sceneDepth;
            binding.hiZDepth = hiZDepth;
        }
        return binding.bindGroup;
    }

    return {
        name: "fluid-polygon-surface",
        engine,
        scene,
        _passes: [],
        setSim(next: FluidSim): void {
            sims = [next];
        },
        setSims(next: readonly FluidSim[]): void {
            sims = next;
        },
        setEnabled(next: boolean): void {
            enabled = next;
        },
        setOpacity(next: number): void {
            opacity = Math.max(0, Math.min(1, next));
        },
        setFluidColor(next: [number, number, number]): void {
            color = [...next];
        },
        setAbsorption(next: number): void {
            absorption = Math.max(0, next);
        },
        setRefractionStrength(next: number): void {
            refractionStrength = Math.max(0, next);
        },
        setSpecularPower(next: number): void {
            specularPower = Math.max(1, next);
        },
        setDirLight(next: [number, number, number]): void {
            lightDirection = [...next];
        },
        setEnvMap(environment: EnvMap): void {
            envView = environment.view;
            envSampler = environment.sampler;
            for (const binding of surfaceBindings.values()) {
                binding.bindGroup = null;
            }
        },
        setEnvRotationY(radians: number): void {
            environmentRotationY = radians;
        },
        setEnvReflection(exposure: number, contrast: number): void {
            environmentExposure = Math.max(0, exposure);
            environmentContrast = Math.max(0, contrast);
        },
        setFresnelF0(value: number): void {
            fresnelF0 = Math.max(0, Math.min(1, value));
        },
        setShadingMode(mode: FluidPolygonShading): void {
            shadingMode = mode;
        },
        setWireframe(next: boolean): void {
            wireframe = next;
        },
        setProfiler(next: FluidProfiler | null): void {
            profiler = next;
        },
        surfaceDepthView(): GPUTextureView | null {
            return enabled ? surfaceDepthView : null;
        },
        record(): void {
            buildRenderTarget(outRT, engine);
        },
        execute(): number {
            const output = outRT._colorView;
            const background = bgRT._colorView;
            const sceneDepth = depthRT._depthView;
            let surfaceCount = 0;
            const activeSurfaces = new Set<FluidPolygonSurface>();
            for (const currentSim of sims) {
                const surface = currentSim.polygonSurface;
                if (!surface) {
                    continue;
                }
                surfaceCount++;
                activeSurfaces.add(surface);
            }
            pruneSurfaceBindings(activeSurfaces);
            if (!enabled || opacity <= 0 || surfaceCount === 0 || !output || !background || !sceneDepth) {
                return 0;
            }
            ensureSurfaceDepth();
            if (depthSourceView !== sceneDepth || !depthSource) {
                depthSourceView = sceneDepth;
                depthSource = { view: sceneDepth } as Texture2D;
            }
            depthPyramid!.build(depthSource, engine._currentEncoder);
            updateUniforms();
            const pass = engine._currentEncoder.beginRenderPass({
                label: "fluid-polygon-surface",
                colorAttachments: [
                    { view: output, loadOp: "load", storeOp: "store" },
                    { view: surfaceDepthView!, loadOp: "clear", storeOp: "store", clearValue: { r: 1e6, g: 0, b: 0, a: 1 } },
                ],
                depthStencilAttachment: {
                    view: frontZView!,
                    depthClearValue: 0,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
                timestampWrites: profiler?.pass("Surface"),
            });
            pass.setPipeline(pipeline);
            for (const currentSim of sims) {
                const surface = currentSim.polygonSurface;
                if (!surface) {
                    continue;
                }
                pass.setBindGroup(0, getBindGroup(surface, background, sceneDepth));
                pass.setVertexBuffer(0, surface.vertexBuffer);
                pass.setIndexBuffer(surface.indexBuffer, surface.indexFormat);
                pass.drawIndexedIndirect(surface.drawIndirect, 0);
            }
            if (wireframe) {
                pass.setPipeline(wireframePipeline);
                pass.setBindGroup(0, wireframeBindGroup);
                for (const currentSim of sims) {
                    const surface = currentSim.polygonSurface;
                    if (!surface?.wireframeIndexBuffer || !surface.wireframeDrawIndirect) {
                        continue;
                    }
                    pass.setVertexBuffer(0, surface.vertexBuffer);
                    pass.setIndexBuffer(surface.wireframeIndexBuffer, surface.indexFormat);
                    pass.drawIndexedIndirect(surface.wireframeDrawIndirect, 0);
                }
            }
            pass.end();
            return 1;
        },
        dispose(): void {
            uniformBuffer.destroy();
            surfaceDepthTexture?.destroy();
            frontZTexture?.destroy();
            depthPyramid?.dispose();
            for (const binding of surfaceBindings.values()) {
                binding.gridBuffer.destroy();
            }
            surfaceBindings.clear();
        },
    };
}
