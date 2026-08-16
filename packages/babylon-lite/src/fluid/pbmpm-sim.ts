// Position-Based MPM (PB-MPM) fluid backend, liquid-only phase.
//
// This is a 3D, world-unit port of EA SEED's PB-MPM solver (Lewin, SIGGRAPH 2024)
// shaped to the same FluidSim contract as the PBF and MLS-MPM demo backends.

import type { EngineContext } from "../engine/engine.js";
import type { DiffusePool, EmitterConfig, FluidFlowConfig, FluidProfiler, FluidSim, FluidSimBaseOptions, FoamConfig, ForceFieldSpec, SceneSdfSpec } from "./sim-common.js";
import {
    FOAM_BYTES,
    FOAM_ACTIVE_FINISH_WGSL,
    FOAM_ACTIVE_PREPARE_WGSL,
    FOAM_COMMON_WGSL,
    SPAWN_ACCEPT_TRIES,
    FLUID_FLOW_RUNTIME_WGSL,
    FLUID_FLOW_STRUCT_WGSL,
    createFluidFlowState,
    createFluidInitialParticles,
    disposeFluidFlowState,
    legacyEmitterConfigToFluidFlow,
    prepareFluidFlowFrame,
    resetFluidFlowState,
    foamActiveListOffset,
    foamActiveStateBytes,
    SCENE_NORMAL_WGSL,
    SCENE_SDF_GRID_WGSL,
    setFluidFlowConfig,
} from "./sim-common.js";
import { SVD3_WGSL } from "./svd3.js";

const WORKGROUP_SIZE = 64;
const MAX_WORKGROUPS = 65535;
const FIXED_POINT = 1e7;
const PARTICLES_PER_CELL = 4;

const PARAMS_F32 = 8 * 4;
const PARAMS_BYTES = PARAMS_F32 * 4;
const COUNTS_OFFSET_F32 = 24;
const MISC_OFFSET_F32 = 28;
const PARTICLE_STRIDE = 144;
const MATERIAL_LIQUID = 0;
const MATERIAL_VISCO = 3;

const COMMON_WGSL = /* wgsl */ `
const FIXED_POINT: f32 = ${FIXED_POINT};
const FIXED_POINT_INV: f32 = ${1 / FIXED_POINT};
const MAX_ENCODE: f32 = 200.0;
const DISPLACEMENT_DAMPING: f32 = 0.99;
const PARTICLES_PER_CELL: f32 = ${PARTICLES_PER_CELL}.0;

struct Params {
    origin: vec4<f32>,    // xyz = boundsMin, w = dx
    dim: vec4<f32>,       // xyz = grid dimensions, w = groundY
    boundsMin: vec4<f32>,
    boundsMax: vec4<f32>,
    sim: vec4<f32>,       // subDt, gravity, liquidRelaxation, liquidViscosity
    solid: vec4<f32>,     // elasticityRatio, elasticRelaxation, frictionAngleDeg, plasticity
    counts: vec4<u32>,    // particle count, _, _, _
    misc: vec4<f32>,      // frameDt, _, _, _
};

fn enc(x: f32) -> i32 { return i32(clamp(x, -MAX_ENCODE, MAX_ENCODE) * FIXED_POINT); }
fn dec(x: i32) -> f32 { return f32(x) * FIXED_POINT_INV; }

fn cellOf(worldPos: vec3<f32>, p: Params) -> vec3<i32> {
    return vec3<i32>(floor((worldPos - p.origin.xyz) / p.origin.w));
}
fn index1D(c: vec3<i32>, p: Params) -> i32 {
    return (c.x * i32(p.dim.y) + c.y) * i32(p.dim.z) + c.z;
}
fn inGrid(c: vec3<i32>, p: Params) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(p.dim.xyz));
}
fn trace3(m: mat3x3<f32>) -> f32 {
    return m[0].x + m[1].y + m[2].z;
}
fn ident3() -> mat3x3<f32> {
    return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
}
fn diag3(s: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(vec3<f32>(s.x, 0.0, 0.0), vec3<f32>(0.0, s.y, 0.0), vec3<f32>(0.0, 0.0, s.z));
}
fn matScale(s: f32, m: mat3x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(s * m[0], s * m[1], s * m[2]);
}
fn matAdd(a: mat3x3<f32>, b: mat3x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
fn matSub(a: mat3x3<f32>, b: mat3x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
fn matNeg(a: mat3x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(-a[0], -a[1], -a[2]);
}
fn signedUnit(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}
fn det3(m: mat3x3<f32>) -> f32 {
    return dot(m[0], cross(m[1], m[2]));
}
fn inverseMat3(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let r0 = cross(b, c);
    let r1 = cross(c, a);
    let r2 = cross(a, b);
    let d = dot(a, r0);
    let invD = 1.0 / (signedUnit(d) * max(abs(d), 1.0e-6));
    return mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x) * invD,
        vec3<f32>(r0.y, r1.y, r2.y) * invD,
        vec3<f32>(r0.z, r1.z, r2.z) * invD);
}
fn clampLen(v: vec3<f32>, maxLen: f32) -> vec3<f32> {
    let l = length(v);
    if (l > maxLen && l > 1.0e-8) {
        return v * (maxLen / l);
    }
    return v;
}
`;

const PARTICLE_STRUCT = /* wgsl */ `
struct Particle {
    position: vec3<f32>,
    displacement: vec3<f32>,
    F: mat3x3<f32>,
    D: mat3x3<f32>,
    liquidDensity: f32,
    mass: f32,
    material: f32,
    logJp: f32,
};
`;

const WEIGHTS_WGSL = /* wgsl */ `
fn weightsOf(worldPos: vec3<f32>, p: Params) -> array<vec3<f32>, 3> {
    let fc = (worldPos - p.origin.xyz) / p.origin.w;
    let base = floor(fc);
    let d = fc - (base + 0.5);
    var w: array<vec3<f32>, 3>;
    w[0] = 0.5 * (0.5 - d) * (0.5 - d);
    w[1] = 0.75 - d * d;
    w[2] = 0.5 * (0.5 + d) * (0.5 + d);
    return w;
}
`;

const CLEAR_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> cells: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> volumes: array<atomic<i32>>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&cells)) { return; }
    cells[i] = vec4<u32>(0u);
    atomicStore(&volumes[i], 0);
}`;

const CONSTRAINT_WGSL = /* wgsl */ `
${SVD3_WGSL}
${COMMON_WGSL}
${PARTICLE_STRUCT}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x) { return; }
    var part = particles[i];
    let I = ident3();
    if (part.material < 0.5) {
        let deviatoric = matNeg(matAdd(part.D, transpose(part.D)));
        part.D = matAdd(part.D, matScale(p.sim.w * 0.5, deviatoric));
        let alpha = 0.5 * (1.0 / max(part.liquidDensity, 0.1) - trace3(part.D) - 1.0);
        part.D = matAdd(part.D, matScale(p.sim.z * alpha, I));
    } else {
        let F = matAdd(I, part.D) * part.F;
        let r = svd3(F);
        let df = det3(F);
        let elasticityRatio = p.solid.x;
        let elasticRelaxation = p.solid.y;
        let invF0 = inverseMat3(part.F);
        var tgt = I;
        if (part.material < 1.5 || part.material > 2.5) {
            let cdf = clamp(abs(df), 0.1, 1000.0);
            let scl = signedUnit(df) * pow(cdf, 1.0 / 3.0);
            let Q = matScale(1.0 / scl, F);
            let rotation = r.U * transpose(r.V);
            tgt = matAdd(matScale(elasticityRatio, rotation), matScale(1.0 - elasticityRatio, Q));
        } else {
            var S = r.S;
            if (part.logJp == 0.0) {
                S = clamp(S, vec3<f32>(1.0), vec3<f32>(1000.0));
            }
            let cdf = clamp(abs(df), 0.1, 1.0);
            let scl = signedUnit(df) * pow(cdf, 1.0 / 3.0);
            let Q = matScale(1.0 / scl, F);
            let stretch = r.U * diag3(S) * transpose(r.V);
            tgt = matAdd(matScale(elasticityRatio, stretch), matScale(1.0 - elasticityRatio, Q));
        }
        let targetD = matSub(tgt * invF0, I);
        let diff = matSub(targetD, part.D);
        part.D = matAdd(part.D, matScale(elasticRelaxation, diff));
        if (part.material >= 1.5 && part.material < 2.5) {
            let dev = matNeg(matAdd(part.D, transpose(part.D)));
            part.D = matAdd(part.D, matScale(p.sim.w * 0.5, dev));
        }
    }
    particles[i] = part;
}`;

const P2G_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
struct Cell { mx: atomic<i32>, my: atomic<i32>, mz: atomic<i32>, mass: atomic<i32>, };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read_write> volumes: array<atomic<i32>>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let pi = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (pi >= p.counts.x) { return; }
    let part = particles[pi];
    let pos = part.position;
    let base = cellOf(pos, p) - vec3<i32>(1);
    let w = weightsOf(pos, p);
    let dx = p.origin.w;
    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let node = base + vec3<i32>(gx, gy, gz);
        if (!inGrid(node, p)) { continue; }
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let wm = weight * part.mass;
        let nodeCenter = p.origin.xyz + (vec3<f32>(node) + 0.5) * dx;
        let offset = nodeCenter - pos;
        let momentum = wm * (part.displacement + part.D * offset);
        let idx = index1D(node, p);
        atomicAdd(&cells[idx].mx, enc(momentum.x));
        atomicAdd(&cells[idx].my, enc(momentum.y));
        atomicAdd(&cells[idx].mz, enc(momentum.z));
        atomicAdd(&cells[idx].mass, enc(wm));
        if (part.material < 0.5) {
            let restVolume = dx * dx * dx / PARTICLES_PER_CELL;
            atomicAdd(&volumes[idx], enc(weight * restVolume));
        }
    }}}
}`;

function buildGridUpdateWgsl(scene: SceneSdfSpec | null): string {
    const gridInject = scene?.sdfGrid ? `\n@group(0) @binding(3) var<storage, read> sceneSdfGrid: array<f32>;\n${SCENE_SDF_GRID_WGSL}` : "";
    const decls = scene ? `${scene.struct}\n@group(0) @binding(2) var<uniform> sceneSdfParams: SceneSdfParams;${gridInject}\n${scene.sdf}\n${SCENE_NORMAL_WGSL}` : "";
    const sceneResolve = scene
        ? `
    if (sceneSdf(nodePos + disp, 0.0) < 0.0) {
        let n = sceneNormal(nodePos + disp, 0.0);
        let intoSolid = dot(disp, -n);
        if (intoSolid > 0.0) {
            disp = disp - intoSolid * (-n);
        }
    }`
        : "";
    return /* wgsl */ `
${COMMON_WGSL}
${decls}
struct Cell { mx: i32, my: i32, mz: i32, mass: i32, };
@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&cells)) { return; }
    let c = cells[i];
    var disp = vec3<f32>(0.0);
    let mass = dec(c.mass);
    if (mass >= 1.0e-5) {
        disp = vec3<f32>(dec(c.mx), dec(c.my), dec(c.mz)) / mass;
    }
    let dimZ = i32(p.dim.z);
    let dimYZ = i32(p.dim.y) * dimZ;
    let x = i32(i) / dimYZ;
    let y = (i32(i) / dimZ) % i32(p.dim.y);
    let z = i32(i) % dimZ;
    let nodePos = p.origin.xyz + (vec3<f32>(f32(x), f32(y), f32(z)) + 0.5) * p.origin.w;
    disp = clampLen(disp, p.origin.w * 0.9);
${sceneResolve}
    let clamped = clamp(nodePos + disp, p.boundsMin.xyz + vec3<f32>(1.0e-4), p.boundsMax.xyz - vec3<f32>(1.0e-4));
    disp = clamped - nodePos;
    cells[i] = Cell(enc(disp.x), enc(disp.y), enc(disp.z), c.mass);
}`;
}

const G2P_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
struct Cell { mx: i32, my: i32, mz: i32, mass: i32, };
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> volumes: array<i32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let pi = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (pi >= p.counts.x) { return; }
    var part = particles[pi];
    let pos = part.position;
    let base = cellOf(pos, p) - vec3<i32>(1);
    let w = weightsOf(pos, p);
    let dx = p.origin.w;
    var B = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var d = vec3<f32>(0.0);
    var volume = 0.0;
    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let node = base + vec3<i32>(gx, gy, gz);
        if (!inGrid(node, p)) { continue; }
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let idx = index1D(node, p);
        let cv = cells[idx];
        let gridDisp = vec3<f32>(dec(cv.mx), dec(cv.my), dec(cv.mz));
        let wd = weight * gridDisp;
        let nodeCenter = p.origin.xyz + (vec3<f32>(node) + 0.5) * dx;
        let offset = nodeCenter - pos;
        B += mat3x3<f32>(wd * offset.x, wd * offset.y, wd * offset.z);
        d += wd;
        if (part.material < 0.5) {
            volume += weight * dec(volumes[idx]);
        }
    }}}
    if (part.material < 0.5) {
        volume = volume / max(dx * dx * dx, 1.0e-6);
        let density = 1.0 / max(volume, 1.0e-6);
        if (density < 1.0) {
            part.liquidDensity = mix(part.liquidDensity, density, 0.1);
        }
    }
    let Dinv = 4.0 / (dx * dx);
    part.D = B * Dinv;
    part.displacement = clampLen(d, dx * 0.9);
    particles[pi] = part;
}`;

function buildIntegrateWgsl(scene: SceneSdfSpec | null): string {
    const gridInject = scene?.sdfGrid ? `\n@group(0) @binding(3) var<storage, read> sceneSdfGrid: array<f32>;\n${SCENE_SDF_GRID_WGSL}` : "";
    const decls = scene ? `${scene.struct}\n@group(0) @binding(2) var<uniform> sceneSdfParams: SceneSdfParams;${gridInject}\n${scene.sdf}\n${SCENE_NORMAL_WGSL}` : "";
    const sceneResolve = scene
        ? `
    let sd = sceneSdf(np, 0.0);
    if (sd < 0.0) {
        let n = sceneNormal(np, 0.0);
        np = np - sd * n;
    }`
        : "";
    return /* wgsl */ `
${SVD3_WGSL}
${COMMON_WGSL}
${PARTICLE_STRUCT}
${decls}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x) { return; }
    var part = particles[i];
    let oldPos = part.position;
    if (part.material < 0.5) {
        part.liquidDensity = max(part.liquidDensity * (trace3(part.D) + 1.0), 0.1);
    } else {
        let I = ident3();
        part.F = matAdd(I, part.D) * part.F;
        var r = svd3(part.F);
        var S = clamp(r.S, vec3<f32>(0.2), vec3<f32>(10000.0));
        if (part.material >= 1.5 && part.material < 2.5) {
            let sinPhi = sin(p.solid.z / 180.0 * 3.14159265);
            let dpAlpha = sqrt(2.0 / 3.0) * 2.0 * sinPhi / (3.0 - sinPhi);
            let beta = 0.5;
            let eDiag = log(max(abs(S), vec3<f32>(1.0e-6)));
            let tr = eDiag.x + eDiag.y + eDiag.z + part.logJp;
            let eHat = eDiag - vec3<f32>(tr / 3.0);
            let frob = length(eHat);
            if (tr >= 0.0) {
                S = vec3<f32>(1.0);
                part.logJp = beta * tr;
            } else {
                part.logJp = 0.0;
                let deltaGamma = frob + (p.solid.x + 1.0) * tr * dpAlpha;
                if (deltaGamma > 0.0) {
                    let h = eDiag - (deltaGamma / max(frob, 1.0e-6)) * (eDiag - vec3<f32>(tr / 3.0));
                    S = exp(h);
                }
            }
        } else if (part.material > 2.5) {
            let yieldLimit = exp(1.0 - p.solid.w);
            let J = S.x * S.y * S.z;
            S = clamp(S, vec3<f32>(1.0 / yieldLimit), vec3<f32>(yieldLimit));
            let newJ = max(S.x * S.y * S.z, 1.0e-6);
            S = S * pow(abs(J) / newJ, 1.0 / 3.0);
        }
        part.F = r.U * diag3(S) * transpose(r.V);
    }
    var np = oldPos + part.displacement;
${sceneResolve}
    np = clamp(np, p.boundsMin.xyz + vec3<f32>(1.0e-4), p.boundsMax.xyz - vec3<f32>(1.0e-4));
    np.y = max(np.y, p.dim.w + 1.0e-4);
    var nextDisp = clampLen(np - oldPos, p.origin.w * 0.9) * DISPLACEMENT_DAMPING;
    nextDisp.y = nextDisp.y - p.sim.y * p.sim.x * p.sim.x;
    part.position = np;
    part.displacement = clampLen(nextDisp, p.origin.w * 0.9);
    particles[i] = part;
}`;
}

// Optional interactive external-force pass (setForceField, e.g. the demo's Shift+RMB push). PB-MPM
// carries motion as a per-substep DISPLACEMENT, so the caller-supplied velocity delta (already dt-scaled)
// is added as displacement += delta·subDt. Dispatched once at the START of each substep so it flows
// through that substep's P2G→grid→G2P transfer (mirrors how MLS-MPM injects the force before P2G).
function buildForceWgsl(force: ForceFieldSpec): string {
    return /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> p: Params;
${force.struct}
@group(0) @binding(2) var<uniform> forceFieldParams: ForceFieldParams;
${force.wgsl}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.counts.x) { return; }
    let subDt = max(p.sim.x, 1.0e-6);
    var part = particles[i];
    let vel = part.displacement / subDt;
    part.displacement = part.displacement + externalForce(part.position, vel, subDt) * subDt;
    particles[i] = part;
}`;
}

const COPY_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> renderPos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> renderVel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dbg: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x) { return; }
    let part = particles[i];
    let vel = part.displacement / max(p.sim.x, 1.0e-6);
    renderPos[i] = vec4<f32>(part.position, 1.0);
    renderVel[i] = vec4<f32>(vel, 0.0);
    dbg[i] = length(vel);
}`;

function buildFoamEmitWgsl(activeParticles: boolean): string {
    const headDecl = activeParticles
        ? "@group(0) @binding(6) var<storage, read_write> activeState: array<atomic<u32>>;"
        : "@group(0) @binding(6) var<storage, read_write> head: array<atomic<u32>>;";
    const activeDecl = activeParticles
        ? `
fn activeStride(cap: u32) -> u32 { return ((cap + 63u) / 64u) * 64u; }
fn activeListBase(side: u32, cap: u32) -> u32 { return 64u + side * activeStride(cap); }
fn activeFlagBase(cap: u32) -> u32 { return 64u + 2u * activeStride(cap); }`
        : "";
    const activate = activeParticles
        ? `
        if (atomicExchange(&activeState[activeFlagBase(cap) + idx], 1u) == 0u) {
            let side = atomicLoad(&activeState[3]);
            let dst = atomicAdd(&activeState[1u + side], 1u);
            atomicStore(&activeState[activeListBase(side, cap) + dst], idx);
        }`
        : "";
    return /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
${FOAM_COMMON_WGSL}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> volumes: array<i32>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<uniform> foam: Foam;
@group(0) @binding(5) var<storage, read_write> diffuse: array<Diffuse>;
${headDecl}
${activeDecl}

const K_STRAIN: f32 = 1.6;
const WC_SCALE: f32 = 1.2;
const VOL_SURF_LO: f32 = 0.15;
const VOL_SURF_HI: f32 = 0.85;

fn frob(m: mat3x3<f32>) -> f32 { return sqrt(dot(m[0], m[0]) + dot(m[1], m[1]) + dot(m[2], m[2])); }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.counts.x) { return; }
    let part = particles[i];
    if (part.material >= 0.5) { return; }
    let subDt = max(p.sim.x, 1.0e-6);
    let pos = part.position;
    let vi = part.displacement / subDt;
    let speed = length(vi);
    if (speed < 1e-4) { return; }
    let vhat = vi / speed;
    let C = matScale(1.0 / subDt, part.D);
    let dx = p.origin.w;
    let cellVolume = max(dx * dx * dx, 1.0e-6);
    let Dinv = 4.0 / (dx * dx);
    let frameDt = max(p.misc.x, subDt);

    let base = cellOf(pos, p);
    let w = weightsOf(pos, p);
    var volumeRatio = 0.0;
    var gradVolume = vec3<f32>(0.0);
    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
        if (!inGrid(node, p)) { continue; }
        let nodeVolume = dec(volumes[index1D(node, p)]) / cellVolume;
        let cellDist = (p.origin.xyz + (vec3<f32>(node) + 0.5) * dx) - pos;
        volumeRatio += nodeVolume * weight;
        gradVolume += (nodeVolume * weight) * cellDist;
    }}}
    gradVolume *= Dinv;

    let strain = 0.5 * (C + transpose(C));
    let ita = phi(K_STRAIN * frob(strain), foam.tauTaMin, foam.tauTaMax);
    let ek = 0.5 * speed * speed;
    let ik = phi(ek, foam.tauKMin, foam.tauKMax);
    var n = vec3<f32>(0.0, 1.0, 0.0);
    let gl = length(gradVolume);
    if (gl > 1e-6) { n = -gradVolume / gl; }
    let surfaceness = 1.0 - smoothstep(VOL_SURF_LO, VOL_SURF_HI, volumeRatio);
    let vn = dot(vi, n);
    let dvn = select(0.0, 1.0, (vn / speed) >= 0.6);
    let crest = max(0.0, vn);
    let iwc = phi(WC_SCALE * surfaceness * dvn * crest, foam.tauWcMin, foam.tauWcMax);

    let ndf = ik * (foam.kTa * ita + foam.kWc * iwc) * frameDt;
    // Stochastic rounding: nd must have EXPECTED value ndf (Ihmsen 2012). Rounding to nearest
    // instead turned the generation rates into a step function — frameDt is clamped to 1/60, so
    // every particle shares it, and ndf crossed 0.5 for the whole population at the same rate
    // value (kTa = 30 at 60fps). The control jumped from "no foam at all" to "one per particle
    // per frame" in a single slider step. Carrying the fraction as a spawn PROBABILITY spreads
    // that crossing across the population, so the rate responds continuously.
    let ndWhole = floor(ndf);
    var nd = i32(ndWhole) + select(0, 1, fRnd((i * 2246822519u) ^ (foam.frameSeed * 22695477u) ^ 0x9e3779b9u) < (ndf - ndWhole));
    if (nd <= 0) { return; }
    nd = min(nd, 8);

    let potential = clamp(ita + iwc, 0.0, 1.0);
    let life = mix(foam.tMin, foam.tMax, potential);
    var e1 = cross(vhat, vec3<f32>(0.0, 1.0, 0.0));
    if (!(dot(e1, e1) > 1e-6)) { e1 = cross(vhat, vec3<f32>(1.0, 0.0, 0.0)); }
    e1 = normalize(e1);
    let e2 = cross(vhat, e1);
    let cap = arrayLength(&diffuse);
    let dtv = length(frameDt * vi);
    for (var k = 0; k < nd; k = k + 1) {
        let seed = (i * 2654435761u) ^ (foam.frameSeed * 40503u) ^ (u32(k) * 2246822519u);
        let xr = fRnd(seed);
        let xt = fRnd(seed * 3u + 1u);
        let xh = fRnd(seed * 7u + 5u);
        let rr = foam.rv * sqrt(xr);
        let th = 6.28318530718 * xt;
        let off = e1 * (rr * cos(th)) + e2 * (rr * sin(th));
        let xd = pos + off + vhat * (xh * dtv);
        let vd = off + vi;
        let idx = atomicAdd(&${activeParticles ? "activeState" : "head"}[0], 1u) % cap;
        diffuse[idx].p = vec4<f32>(xd, life);
        diffuse[idx].v = vec4<f32>(vd, 0.0);
${activate}
    }
}`;
}

function buildFoamUpdateWgsl(activeParticles: boolean): string {
    const activeDecl = activeParticles
        ? `
@group(0) @binding(5) var<storage, read_write> activeState: array<atomic<u32>>;
fn activeStride(cap: u32) -> u32 { return ((cap + 63u) / 64u) * 64u; }
fn activeListBase(side: u32, cap: u32) -> u32 { return 64u + side * activeStride(cap); }
fn activeFlagBase(cap: u32) -> u32 { return 64u + 2u * activeStride(cap); }`
        : "";
    const slotLookup = activeParticles
        ? `
    let cap = arrayLength(&diffuse);
    let side = atomicLoad(&activeState[3]);
    let slot = gid.x + gid.y * ${MAX_WORKGROUPS * WORKGROUP_SIZE}u;
    if (slot >= atomicLoad(&activeState[1u + side])) { return; }
    let i = atomicLoad(&activeState[activeListBase(side, cap) + slot]);`
        : `
    let i = gid.x + gid.y * ${MAX_WORKGROUPS * WORKGROUP_SIZE}u;
    if (i >= arrayLength(&diffuse)) { return; }`;
    const deactivate = activeParticles ? "atomicStore(&activeState[activeFlagBase(cap) + i], 0u);" : "";
    const keepActive = activeParticles
        ? `
    let nextSide = 1u - side;
    let dst = atomicAdd(&activeState[1u + nextSide], 1u);
    atomicStore(&activeState[activeListBase(nextSide, cap) + dst], i);`
        : "";
    return /* wgsl */ `
${COMMON_WGSL}
${WEIGHTS_WGSL}
${FOAM_COMMON_WGSL}
struct Cell { mx: i32, my: i32, mz: i32, mass: i32, };
@group(0) @binding(0) var<storage, read> cells: array<Cell>;
@group(0) @binding(1) var<storage, read> volumes: array<i32>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<uniform> foam: Foam;
@group(0) @binding(4) var<storage, read_write> diffuse: array<Diffuse>;
${activeDecl}

const VOL_SPRAY: f32 = 0.35;
const VOL_BUBBLE: f32 = 0.9;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Pool slots can exceed one dispatch dimension, in which case dispatch() spills into y
    // with an x extent of exactly MAX_WORKGROUPS groups — fold that back into a flat index.
    // gid.y is 0 whenever the dispatch fits in x, so this is a no-op for small pools.
${slotLookup}
    let p0 = diffuse[i].p;
    if (p0.w <= 0.0) { ${deactivate} return; }
    let pp = p0.xyz;
    if (any(pp < p.boundsMin.xyz) || any(pp > p.boundsMax.xyz)) {
        diffuse[i].p = vec4<f32>(pp, 0.0);
        ${deactivate}
        return;
    }
    var v = diffuse[i].v.xyz;

    let subDt = max(p.sim.x, 1.0e-6);
    let dx = p.origin.w;
    let cellVolume = max(dx * dx * dx, 1.0e-6);
    let base = cellOf(pp, p);
    let w = weightsOf(pp, p);
    var vf = vec3<f32>(0.0);
    var volumeRatio = 0.0;
    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
        if (!inGrid(node, p)) { continue; }
        let idx = index1D(node, p);
        vf += vec3<f32>(dec(cells[idx].mx), dec(cells[idx].my), dec(cells[idx].mz)) * (weight / subDt);
        volumeRatio += (dec(volumes[idx]) / cellVolume) * weight;
    }}}

    let g = p.sim.y;
    let dt = max(p.misc.x, subDt);
    var kind = 1u;
    if (volumeRatio < VOL_SPRAY) { kind = 0u; } else if (volumeRatio > VOL_BUBBLE) { kind = 2u; }

    var np = pp;
    var life = p0.w;
    if (kind == 0u) {
        v.y -= g * dt;
        np = pp + dt * v;
    } else if (kind == 2u) {
        v.y += dt * foam.kb * g;
        v += foam.kd * (vf - v);
        np = pp + dt * v;
    } else {
        v = vf;
        np = pp + dt * vf;
        life = p0.w - dt;
    }
    if (life <= 0.0) {
        diffuse[i].p = vec4<f32>(np, 0.0);
        ${deactivate}
        return;
    }
    diffuse[i].p = vec4<f32>(np, life);
    diffuse[i].v = vec4<f32>(v, f32(kind));
${keepActive}
}`;
}

// Generic flow recycling; PB-MPM maps launch velocity to per-substep displacement.
const FLOW_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${FLUID_FLOW_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> flow: FluidFlowData;
@group(0) @binding(2) var<storage, read_write> flowCounters: array<atomic<u32>>;
${FLUID_FLOW_RUNTIME_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= flow.header.z) { return; }
    let launch = fluidTryRelaunch(particles[i].position, i);
    if (launch.launched != 0u) {
        particles[i].position = launch.position;
        particles[i].displacement = launch.velocity * flow.frame.y;
        particles[i].D = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
        particles[i].F = ident3();
        particles[i].liquidDensity = 1.0;
    }
}`;

/**
 * The physics-slider keys a given PB-MPM material actually uses. The solver branches on the material
 * (see the constraint/integrate passes), so the other sliders have no effect — hosts can hide them.
 * `gravity`, `iterations`, `substeps`, `restitution` are common to every material.
 */
export function pbmpmParamKeysForMaterial(material: number): string[] {
    const common = ["gravity", "iterations", "substeps", "restitution"];
    if (material < 0.5) {
        return [...common, "liquidRelaxation", "liquidViscosity"]; // liquid
    }
    if (material >= 1.5 && material < 2.5) {
        return [...common, "elasticityRatio", "elasticRelaxation", "frictionAngle", "liquidViscosity"]; // sand
    }
    if (material > 2.5) {
        return [...common, "elasticityRatio", "elasticRelaxation", "plasticity"]; // viscoelastic
    }
    return [...common, "elasticityRatio", "elasticRelaxation"]; // elastic
}

export interface PbMpmOptions extends FluidSimBaseOptions {
    /** Simulation box min corner. Default [-20, 0, -20]. */
    boundsMin?: [number, number, number];
    /** Simulation box max corner. Default [20, 15, 20]. */
    boundsMax?: [number, number, number];
    /** Exact grid cell count along X/Y/Z. Derived from bounds when omitted. */
    gridDim?: [number, number, number];
    /** Ground plane height. Default boundsMin.y. */
    groundY?: number;
    /** Grid cell size in world units. Default 0.25. */
    dx?: number;
    /** Sub-steps per frame. Default 3. */
    substeps?: number;
    /** Safety cap on a single sub-step dt. Default 1/120. */
    maxSubDt?: number;
    /** PB-MPM projection iterations per sub-step. Default 5. */
    iterations?: number;
    /** Liquid volume relaxation. Default 1.5. */
    liquidRelaxation?: number;
    /** Liquid viscosity projection strength. Default 0.01. */
    liquidViscosity?: number;
    /** Solid target blend: 1 = pure rotation/stretch SVD target, 0 = volume-preserving target. Default 0.3. */
    elasticityRatio?: number;
    /** Solid constraint step size. Default 0.3. */
    elasticRelaxation?: number;
    /** Sand Drucker-Prager friction angle in degrees. Default 35. */
    frictionAngle?: number;
    /** Viscoelastic yield/plasticity control. Default 0.8. */
    plasticity?: number;
    /** Particle material: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. Default 0. */
    material?: number;
    /** Collision restitution placeholder for interface parity. Default 0. */
    restitution?: number;
    /** Explicit world-space xyz seed positions. */
    initialPositions?: Float32Array;
}

export function createPbMpmSim(engine: EngineContext, options: PbMpmOptions = {}): FluidSim {
    const device = engine._device;
    const count = options.count ?? 60000;
    const particleRadius = options.particleRadius ?? 0.09;
    const spawnMin: [number, number, number] = options.spawnMin ? [...options.spawnMin] : [-2, 4, -2];
    const spawnMax: [number, number, number] = options.spawnMax ? [...options.spawnMax] : [2, 12, 2];
    let spawnAccept: ((x: number, y: number, z: number) => boolean) | null = null;
    const boundsMin = options.boundsMin ?? [-20, 0, -20];
    const boundsMax = options.boundsMax ?? [20, 15, 20];
    const groundY = options.groundY ?? boundsMin[1];
    const dx = options.dx ?? 0.25;
    // Mutable so the host can trade stability against cost at runtime (setParam "maxSubDtMs"):
    // it is the cap that decides whether `step()` has to run MORE sub-steps than requested.
    let maxSubDt = options.maxSubDt ?? 1 / 120;
    let gravity = options.gravity ?? 9.8;
    let substepsMut = Math.max(1, Math.round(options.substeps ?? 3));
    let iterationsMut = Math.max(1, Math.round(options.iterations ?? 5));
    let liquidRelaxation = options.liquidRelaxation ?? 1.5;
    let liquidViscosity = options.liquidViscosity ?? 0.01;
    let elasticityRatio = options.elasticityRatio ?? 0.3;
    let elasticRelaxation = options.elasticRelaxation ?? 0.3;
    let frictionAngle = options.frictionAngle ?? 35;
    let plasticity = options.plasticity ?? 0.8;
    let currentMaterial = Math.min(MATERIAL_VISCO, Math.max(MATERIAL_LIQUID, Math.round(options.material ?? MATERIAL_LIQUID)));
    const initialPositions = options.initialPositions ?? null;
    let simProfiler: FluidProfiler | null = null;

    const gridDim: [number, number, number] = options.gridDim
        ? (options.gridDim.map((value) => Math.max(4, Math.round(value))) as [number, number, number])
        : [
              Math.max(4, Math.ceil((boundsMax[0] - boundsMin[0]) / dx)),
              Math.max(4, Math.ceil((boundsMax[1] - boundsMin[1]) / dx)),
              Math.max(4, Math.ceil((boundsMax[2] - boundsMin[2]) / dx)),
          ];
    const numCells = gridDim[0] * gridDim[1] * gridDim[2];
    // Start-of-sim WARM-UP ramp (mirrors MLS-MPM): only `liveCount` particles are simulated
    // each frame, growing by `warmupStep`. Dormant particles are skipped by every particle pass
    // — the passes all guard on Params.counts.x, which carries liveCount, and they are not even
    // dispatched — and are parked off-screen by seed() so nothing renders until they activate.
    // Lets a demo seed its pool into a volume far smaller than the pool itself (the waterfall
    // fills its summit springs) without an instant density spike. 0 = release everything.
    let warmupFrames = Math.max(0, Math.floor(options.warmupFrames ?? 0));
    let warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(count / warmupFrames)) : count;
    let liveCount = count;
    let initialTargetCount = count;
    let particleGroups = Math.ceil(count / WORKGROUP_SIZE);
    const cellGroups = Math.ceil(numCells / WORKGROUP_SIZE);

    /** Re-derive the dispatch size + the shader-visible particle count from `liveCount`. */
    function applyLiveCount(): void {
        particleGroups = Math.max(1, Math.ceil(liveCount / WORKGROUP_SIZE));
        pu[COUNTS_OFFSET_F32] = liveCount;
    }

    const particleBuffer = device.createBuffer({ label: "pbmpm-particles", size: count * PARTICLE_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const cellBuffer = device.createBuffer({ label: "pbmpm-cells", size: numCells * 16, usage: GPUBufferUsage.STORAGE });
    const volumeBuffer = device.createBuffer({ label: "pbmpm-grid-volume", size: numCells * 4, usage: GPUBufferUsage.STORAGE });
    const positionBuffer = device.createBuffer({ label: "pbmpm-render-pos", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const velocityBuffer = device.createBuffer({ label: "pbmpm-render-vel", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const debugBuffer = device.createBuffer({ label: "pbmpm-debug", size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const paramsBuffer = device.createBuffer({ label: "pbmpm-params", size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const flowState = createFluidFlowState(device, count, particleRadius);
    let flowSeedsInitialParticles = true;

    const paramsData = new ArrayBuffer(PARAMS_BYTES);
    const pf = new Float32Array(paramsData);
    const pu = new Uint32Array(paramsData);
    pf[0] = boundsMin[0];
    pf[1] = boundsMin[1];
    pf[2] = boundsMin[2];
    pf[3] = dx;
    pf[4] = gridDim[0];
    pf[5] = gridDim[1];
    pf[6] = gridDim[2];
    pf[7] = groundY;
    pf[8] = boundsMin[0];
    pf[9] = boundsMin[1];
    pf[10] = boundsMin[2];
    pf[12] = boundsMax[0];
    pf[13] = boundsMax[1];
    pf[14] = boundsMax[2];
    pu[COUNTS_OFFSET_F32] = count;

    function writeDynamicParams(subDt: number, frameDt: number): void {
        pf[16] = subDt;
        pf[17] = gravity;
        pf[18] = liquidRelaxation;
        pf[19] = liquidViscosity;
        pf[20] = elasticityRatio;
        pf[21] = elasticRelaxation;
        pf[22] = frictionAngle;
        pf[23] = plasticity;
        pf[MISC_OFFSET_F32] = frameDt;
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    }

    function seed(): void {
        resetFluidFlowState(flowState);
        const flowParticles = initialPositions || !flowSeedsInitialParticles ? null : createFluidInitialParticles(count, flowState.config, flowState.particleVolume);
        initialTargetCount = initialPositions ? count : (flowParticles?.activeCount ?? count);
        warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(initialTargetCount / warmupFrames)) : initialTargetCount;
        // Reset the warm-up ramp: start with just the first initial batch live (or the
        // whole initial prefix when disabled). Inflow capacity remains dormant.
        liveCount = warmupFrames > 0 ? Math.min(initialTargetCount, warmupStep) : initialTargetCount;
        applyLiveCount();
        const buf = new ArrayBuffer(count * PARTICLE_STRIDE);
        const f = new Float32Array(buf);
        const rp = new Float32Array(count * 4);
        const flowPositions = flowParticles?.positions;
        const flowVelocities = flowParticles?.velocities;
        const renderVelocities = new Float32Array(count * 4);
        const nominalFrameDt = 1 / 60;
        const seedSubDt = nominalFrameDt / Math.max(substepsMut, Math.ceil(nominalFrameDt / maxSubDt));
        // Last position that passed `spawnAccept`, reused when a particle exhausts its retries.
        let lastOkX = 0;
        let lastOkY = 0;
        let lastOkZ = 0;
        let haveLastOk = false;
        for (let i = 0; i < count; i++) {
            const o = (i * PARTICLE_STRIDE) / 4;
            let x: number;
            let y: number;
            let z: number;
            if (initialPositions) {
                x = initialPositions[i * 3]!;
                y = initialPositions[i * 3 + 1]!;
                z = initialPositions[i * 3 + 2]!;
            } else if (flowPositions && i < initialTargetCount) {
                x = flowPositions[i * 3]!;
                y = flowPositions[i * 3 + 1]!;
                z = flowPositions[i * 3 + 2]!;
            } else if (flowParticles) {
                x = 0;
                y = 0;
                z = 0;
            } else {
                x = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
                y = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
                z = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
                if (spawnAccept) {
                    // Reject-sample so particles fit a non-box container shape. On exhaustion
                    // reuse the last ACCEPTED point rather than keeping a rejected one, which
                    // would place particles outside the container (see mls-mpm-sim).
                    let ok = spawnAccept(x, y, z);
                    for (let tries = 0; !ok && tries < SPAWN_ACCEPT_TRIES; tries++) {
                        x = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
                        y = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
                        z = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
                        ok = spawnAccept(x, y, z);
                    }
                    if (ok) {
                        lastOkX = x;
                        lastOkY = y;
                        lastOkZ = z;
                        haveLastOk = true;
                    } else if (haveLastOk) {
                        x = lastOkX;
                        y = lastOkY;
                        z = lastOkZ;
                    }
                }
            }
            f[o] = x;
            f[o + 1] = y;
            f[o + 2] = z;
            if (flowVelocities && i < initialTargetCount) {
                f[o + 4] = flowVelocities[i * 3]! * seedSubDt;
                f[o + 5] = flowVelocities[i * 3 + 1]! * seedSubDt;
                f[o + 6] = flowVelocities[i * 3 + 2]! * seedSubDt;
                renderVelocities[i * 4] = flowVelocities[i * 3]!;
                renderVelocities[i * 4 + 1] = flowVelocities[i * 3 + 1]!;
                renderVelocities[i * 4 + 2] = flowVelocities[i * 3 + 2]!;
            }
            f[o + 8] = 1;
            f[o + 13] = 1;
            f[o + 18] = 1;
            f[o + 32] = 1;
            f[o + 33] = 1;
            f[o + 34] = currentMaterial;
            // The particle keeps its real spawn position; only the RENDER position is parked
            // off-screen while dormant, so activating it needs no teleport.
            const live = i < liveCount;
            rp[i * 4] = live ? x : 0;
            rp[i * 4 + 1] = live ? y : -1.0e5;
            rp[i * 4 + 2] = live ? z : 0;
            rp[i * 4 + 3] = 1;
        }
        device.queue.writeBuffer(particleBuffer, 0, buf);
        device.queue.writeBuffer(positionBuffer, 0, rp);
        device.queue.writeBuffer(velocityBuffer, 0, renderVelocities);
        device.queue.writeBuffer(debugBuffer, 0, new Float32Array(count));
    }

    function activateInflowParticles(start: number, particles: NonNullable<ReturnType<typeof prepareFluidFlowFrame>["particles"]>, subDt: number): void {
        const data = new ArrayBuffer(particles.activeCount * PARTICLE_STRIDE);
        const values = new Float32Array(data);
        for (let i = 0; i < particles.activeCount; i++) {
            const source = i * 3;
            const target = (i * PARTICLE_STRIDE) / 4;
            values[target] = particles.positions[source]!;
            values[target + 1] = particles.positions[source + 1]!;
            values[target + 2] = particles.positions[source + 2]!;
            values[target + 4] = particles.velocities[source]! * subDt;
            values[target + 5] = particles.velocities[source + 1]! * subDt;
            values[target + 6] = particles.velocities[source + 2]! * subDt;
            values[target + 8] = 1;
            values[target + 13] = 1;
            values[target + 18] = 1;
            values[target + 32] = 1;
            values[target + 33] = 1;
            values[target + 34] = currentMaterial;
        }
        device.queue.writeBuffer(particleBuffer, start * PARTICLE_STRIDE, data);
    }

    function pipeline(label: string, code: string): GPUComputePipeline {
        return device.createComputePipeline({ label, layout: "auto", compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" } });
    }

    const clearPipe = pipeline("pbmpm-clear", CLEAR_WGSL);
    const constraintPipe = pipeline("pbmpm-constraint", CONSTRAINT_WGSL);
    const p2gPipe = pipeline("pbmpm-p2g", P2G_WGSL);
    const g2pPipe = pipeline("pbmpm-g2p", G2P_WGSL);
    const copyPipe = pipeline("pbmpm-copy", COPY_WGSL);
    let gridUpdatePipe = pipeline("pbmpm-grid-update", buildGridUpdateWgsl(null));
    let integratePipe = pipeline("pbmpm-integrate", buildIntegrateWgsl(null));

    const clearBG = device.createBindGroup({
        layout: clearPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellBuffer } },
            { binding: 1, resource: { buffer: volumeBuffer } },
        ],
    });
    const constraintBG = device.createBindGroup({
        layout: constraintPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: paramsBuffer } },
        ],
    });
    const p2gBG = device.createBindGroup({
        layout: p2gPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: cellBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
            { binding: 3, resource: { buffer: volumeBuffer } },
        ],
    });
    const g2pBG = device.createBindGroup({
        layout: g2pPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: cellBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
            { binding: 3, resource: { buffer: volumeBuffer } },
        ],
    });
    const copyBG = device.createBindGroup({
        layout: copyPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: positionBuffer } },
            { binding: 2, resource: { buffer: velocityBuffer } },
            { binding: 3, resource: { buffer: debugBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ],
    });
    let gridUpdateBG = buildGridUpdateBG(gridUpdatePipe, null);
    let integrateBG = buildIntegrateBG(integratePipe, null);

    const flowPipe = pipeline("pbmpm-flow", FLOW_WGSL);
    const flowBG = device.createBindGroup({
        layout: flowPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: flowState.uniformBuffer } },
            { binding: 2, resource: { buffer: flowState.counterBuffer } },
        ],
    });

    // Interactive external force (setForceField). Pipeline + bind group built lazily on first use and
    // rebuilt only when the injected WGSL source or the caller's buffer changes; nothing runs while idle.
    let forceSpec: ForceFieldSpec | null = null;
    let forcePipe: GPUComputePipeline | null = null;
    let forceBuiltSrc = "";
    let forceBoundSpec: ForceFieldSpec | null = null;
    let forceBG: GPUBindGroup | null = null;

    function buildForceBG(pipe: GPUComputePipeline, spec: ForceFieldSpec): GPUBindGroup {
        return device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: paramsBuffer } },
                { binding: 2, resource: { buffer: spec.buffer } },
            ],
        });
    }

    function buildGridUpdateBG(pipe: GPUComputePipeline, scene: SceneSdfSpec | null): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: cellBuffer } },
            { binding: 1, resource: { buffer: paramsBuffer } },
        ];
        if (scene) {
            entries.push({ binding: 2, resource: { buffer: scene.buffer } });
            if (scene.sdfGrid) {
                entries.push({ binding: 3, resource: { buffer: scene.sdfGrid } });
            }
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    function buildIntegrateBG(pipe: GPUComputePipeline, scene: SceneSdfSpec | null): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: paramsBuffer } },
        ];
        if (scene) {
            entries.push({ binding: 2, resource: { buffer: scene.buffer } });
            if (scene.sdfGrid) {
                entries.push({ binding: 3, resource: { buffer: scene.sdfGrid } });
            }
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }

    // The pool is sized purely from poolScale × count — no arbitrary ceiling. The only
    // bounds left are the device's: the pool is ONE storage buffer bound to the compute and
    // render passes (so it cannot exceed maxStorageBufferBindingSize), and the update pass
    // dispatches over it in a 2D-spilled grid (MAX_WORKGROUPS² groups, effectively boundless).
    const FOAM_CAP_LIMIT = Math.min(Math.floor(device.limits.maxStorageBufferBindingSize / 32), MAX_WORKGROUPS * MAX_WORKGROUPS * WORKGROUP_SIZE);
    const foamData = new ArrayBuffer(FOAM_BYTES);
    const foamF32 = new Float32Array(foamData);
    const foamU32 = new Uint32Array(foamData);
    let foamEnabled = false;
    let foamSeed = 0;
    let foamCapacity = 0;
    let foamPoolGroups = 0;
    let foamActiveParticles = false;
    let foamActiveSide = 0;
    let diffuseBuffer: GPUBuffer | null = null;
    let diffuseHeadBuffer: GPUBuffer | null = null;
    let foamActiveStateBuffer: GPUBuffer | null = null;
    let foamActiveDispatchBuffer: GPUBuffer | null = null;
    let foamDrawIndirectBuffer: GPUBuffer | null = null;
    let foamParamsBuffer: GPUBuffer | null = null;
    let foamEmitPipe: GPUComputePipeline | null = null;
    let foamDenseEmitPipe: GPUComputePipeline | null = null;
    let foamActiveEmitPipe: GPUComputePipeline | null = null;
    let foamUpdatePipe: GPUComputePipeline | null = null;
    let foamDenseUpdatePipe: GPUComputePipeline | null = null;
    let foamActiveUpdatePipe: GPUComputePipeline | null = null;
    let foamActivePreparePipe: GPUComputePipeline | null = null;
    let foamActiveFinishPipe: GPUComputePipeline | null = null;
    let foamEmitBG: GPUBindGroup | null = null;
    let foamUpdateBG: GPUBindGroup | null = null;
    let foamActivePrepareBG: GPUBindGroup | null = null;
    let foamActiveFinishBG: GPUBindGroup | null = null;
    let diffusePool: DiffusePool | undefined;

    function buildFoamBindGroups(): void {
        foamEmitBG = device.createBindGroup({
            layout: foamEmitPipe!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 2, resource: { buffer: volumeBuffer } },
                { binding: 3, resource: { buffer: paramsBuffer } },
                { binding: 4, resource: { buffer: foamParamsBuffer! } },
                { binding: 5, resource: { buffer: diffuseBuffer! } },
                { binding: 6, resource: { buffer: foamActiveParticles ? foamActiveStateBuffer! : diffuseHeadBuffer! } },
            ],
        });
        const updateEntries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: cellBuffer } },
            { binding: 1, resource: { buffer: volumeBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
            { binding: 3, resource: { buffer: foamParamsBuffer! } },
            { binding: 4, resource: { buffer: diffuseBuffer! } },
        ];
        if (foamActiveParticles) {
            updateEntries.push({ binding: 5, resource: { buffer: foamActiveStateBuffer! } });
        }
        foamUpdateBG = device.createBindGroup({
            layout: foamUpdatePipe!.getBindGroupLayout(0),
            entries: updateEntries,
        });
        foamActivePrepareBG = foamActiveParticles
            ? device.createBindGroup({
                  layout: foamActivePreparePipe!.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: foamActiveStateBuffer! } },
                      { binding: 1, resource: { buffer: foamActiveDispatchBuffer! } },
                  ],
              })
            : null;
        foamActiveFinishBG = foamActiveParticles
            ? device.createBindGroup({
                  layout: foamActiveFinishPipe!.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: foamActiveStateBuffer! } },
                      { binding: 1, resource: { buffer: foamDrawIndirectBuffer! } },
                  ],
              })
            : null;
    }

    function ensureFoam(cfg: FoamConfig): void {
        if (!foamEmitPipe) {
            foamDenseEmitPipe = pipeline("pbmpm-foam-emit", buildFoamEmitWgsl(false));
            foamDenseUpdatePipe = pipeline("pbmpm-foam-update", buildFoamUpdateWgsl(false));
            foamParamsBuffer = device.createBuffer({ label: "pbmpm-foam-params", size: FOAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            diffuseHeadBuffer = device.createBuffer({ label: "pbmpm-foam-head", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        }
        const nextActiveParticles = cfg.activeParticles ?? false;
        const activeModeChanged = nextActiveParticles !== foamActiveParticles;
        if (nextActiveParticles && !foamActiveUpdatePipe) {
            foamActiveEmitPipe = pipeline("pbmpm-foam-emit-active", buildFoamEmitWgsl(true));
            foamActiveUpdatePipe = pipeline("pbmpm-foam-update-active", buildFoamUpdateWgsl(true));
            foamActivePreparePipe = pipeline("pbmpm-foam-active-prepare", FOAM_ACTIVE_PREPARE_WGSL);
            foamActiveFinishPipe = pipeline("pbmpm-foam-active-finish", FOAM_ACTIVE_FINISH_WGSL);
        }
        foamActiveParticles = nextActiveParticles;
        foamEmitPipe = foamActiveParticles ? foamActiveEmitPipe! : foamDenseEmitPipe!;
        foamUpdatePipe = foamActiveParticles ? foamActiveUpdatePipe! : foamDenseUpdatePipe!;
        let cap = Math.round(count * (cfg.poolScale ?? 3));
        cap = Math.max(1024, Math.min(cap, cfg.poolCapMax ?? Infinity, FOAM_CAP_LIMIT));
        if (foamActiveParticles) {
            cap = Math.min(cap, Math.max(1024, Math.floor((device.limits.maxStorageBufferBindingSize - 256) / 12)));
            while (foamActiveStateBytes(cap) > device.limits.maxStorageBufferBindingSize) {
                cap--;
            }
        }
        const resizePool = cap !== foamCapacity || !diffuseBuffer;
        const rebuildActiveResources = foamActiveParticles && (resizePool || !foamActiveStateBuffer || !foamActiveDispatchBuffer || !foamDrawIndirectBuffer);
        if (resizePool) {
            diffuseBuffer?.destroy();
            diffuseBuffer = device.createBuffer({ label: "pbmpm-foam-pool", size: cap * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            foamCapacity = cap;
            foamPoolGroups = Math.ceil(cap / WORKGROUP_SIZE);
            const enc = device.createCommandEncoder({ label: "pbmpm-foam-clear" });
            enc.clearBuffer(diffuseBuffer);
            enc.clearBuffer(diffuseHeadBuffer!);
            device.queue.submit([enc.finish()]);
        }
        if (rebuildActiveResources) {
            foamActiveStateBuffer?.destroy();
            foamActiveDispatchBuffer?.destroy();
            foamDrawIndirectBuffer?.destroy();
            foamActiveStateBuffer = device.createBuffer({
                label: "pbmpm-foam-active-state",
                size: foamActiveStateBytes(cap),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            foamActiveDispatchBuffer = device.createBuffer({
                label: "pbmpm-foam-active-dispatch",
                size: 12,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
            });
            foamDrawIndirectBuffer = device.createBuffer({
                label: "pbmpm-foam-draw-indirect",
                size: 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
            });
            foamActiveSide = 0;
        } else if (!foamActiveParticles && foamActiveStateBuffer) {
            foamActiveStateBuffer.destroy();
            foamActiveDispatchBuffer!.destroy();
            foamDrawIndirectBuffer!.destroy();
            foamActiveStateBuffer = null;
            foamActiveDispatchBuffer = null;
            foamDrawIndirectBuffer = null;
        }
        if (foamActiveParticles && (rebuildActiveResources || activeModeChanged)) {
            const enc = device.createCommandEncoder({ label: "pbmpm-foam-active-reset" });
            enc.clearBuffer(diffuseBuffer!);
            enc.clearBuffer(diffuseHeadBuffer!);
            enc.clearBuffer(foamActiveStateBuffer!);
            device.queue.submit([enc.finish()]);
            device.queue.writeBuffer(foamActiveStateBuffer!, 16, new Uint32Array([cap]));
            foamActiveSide = 0;
        } else if (!foamActiveParticles && activeModeChanged) {
            const enc = device.createCommandEncoder({ label: "pbmpm-foam-dense-head-reset" });
            enc.clearBuffer(diffuseHeadBuffer!);
            device.queue.submit([enc.finish()]);
        }
        diffusePool = {
            buffer: diffuseBuffer!,
            headBuffer: foamActiveParticles ? foamActiveStateBuffer! : diffuseHeadBuffer!,
            capacity: cap,
            ...(foamActiveParticles
                ? {
                      activeIndices: foamActiveStateBuffer!,
                      activeIndicesOffset: foamActiveListOffset(cap, foamActiveSide),
                      drawIndirect: foamDrawIndirectBuffer!,
                  }
                : {}),
        };
        buildFoamBindGroups();
        foamF32[0] = 1;
        foamF32[1] = 12;
        foamF32[2] = 0.5;
        foamF32[3] = 5;
        foamF32[4] = 1;
        foamF32[5] = 25;
        foamF32[6] = cfg.kTa ?? 40;
        foamF32[7] = cfg.kWc ?? 40;
        foamF32[8] = cfg.kb ?? 0.8;
        foamF32[9] = cfg.kd ?? 0.5;
        foamF32[10] = cfg.rv ?? particleRadius;
        foamF32[12] = cfg.tMin ?? 0.3;
        foamF32[13] = cfg.tMax ?? 2.0;
        device.queue.writeBuffer(foamParamsBuffer!, 0, foamData);
    }

    function clearFoamPool(label: string): void {
        if (!diffuseBuffer || !diffuseHeadBuffer) {
            return;
        }
        const enc = device.createCommandEncoder({ label });
        enc.clearBuffer(diffuseBuffer);
        enc.clearBuffer(diffuseHeadBuffer);
        if (foamActiveStateBuffer) {
            enc.clearBuffer(foamActiveStateBuffer);
            enc.clearBuffer(foamDrawIndirectBuffer!);
        }
        device.queue.submit([enc.finish()]);
        if (foamActiveStateBuffer) {
            device.queue.writeBuffer(foamActiveStateBuffer, 16, new Uint32Array([foamCapacity]));
            foamActiveSide = 0;
            if (diffusePool) {
                diffusePool = { ...diffusePool, activeIndicesOffset: foamActiveListOffset(foamCapacity, foamActiveSide) };
            }
        }
    }

    function dispatch(encoder: GPUCommandEncoder, label: string, pipe: GPUComputePipeline, bg: GPUBindGroup, groups: number): void {
        const pass = encoder.beginComputePass({ label, timestampWrites: simProfiler?.pass(label.includes("foam") ? "Foam gen" : "Simulation") });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        if (groups > MAX_WORKGROUPS) {
            pass.dispatchWorkgroups(MAX_WORKGROUPS, Math.ceil(groups / MAX_WORKGROUPS), 1);
        } else {
            pass.dispatchWorkgroups(groups);
        }
        pass.end();
    }

    function dispatchIndirect(encoder: GPUCommandEncoder, label: string, pipe: GPUComputePipeline, bg: GPUBindGroup, args: GPUBuffer): void {
        const pass = encoder.beginComputePass({ label, timestampWrites: simProfiler?.pass(label.includes("foam") ? "Foam gen" : "Simulation") });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroupsIndirect(args, 0);
        pass.end();
    }

    seed();
    writeDynamicParams(1 / 120, 1 / 60);

    return {
        count,
        get activeCount(): number {
            return liveCount;
        },
        particleRadius,
        surfaceSizeScale: 1.5,
        positionBuffer,
        velocityBuffer,
        debugBuffer,
        debugNorm: 1 / 6,
        get gpuBytes(): number {
            let b =
                particleBuffer.size +
                cellBuffer.size +
                volumeBuffer.size +
                positionBuffer.size +
                velocityBuffer.size +
                debugBuffer.size +
                paramsBuffer.size +
                flowState.uniformBuffer.size +
                flowState.counterBuffer.size;
            if (diffuseBuffer) {
                b += diffuseBuffer.size;
            }
            if (diffuseHeadBuffer) {
                b += diffuseHeadBuffer.size;
            }
            if (foamActiveStateBuffer) {
                b += foamActiveStateBuffer.size + foamActiveDispatchBuffer!.size + foamDrawIndirectBuffer!.size;
            }
            if (foamParamsBuffer) {
                b += foamParamsBuffer.size;
            }
            return b;
        },
        step(encoder: GPUCommandEncoder, dt: number): void {
            const frameDt = dt > 0 ? dt : 1 / 60;
            // `maxSubDt` is honoured by ADDING sub-steps, never by shortening the frame: clamping
            // the sub-step dt would drop simulated time, making playback speed track the frame rate
            // rather than the clock. The frame always advances exactly `frameDt`. (The epsilon
            // absorbs float error so an exact multiple stays on the lower sub-step count.)
            const stepCount = Math.max(substepsMut, Math.ceil(frameDt / maxSubDt - 1e-9));
            const subDt = frameDt / stepCount;
            if (liveCount < initialTargetCount) {
                // Release the next slice BEFORE writing params, so this frame simulates it.
                liveCount = Math.min(initialTargetCount, liveCount + warmupStep);
                applyLiveCount();
            }
            const flowFrame = prepareFluidFlowFrame(flowState, frameDt, liveCount, liveCount >= initialTargetCount ? count - liveCount : 0, subDt);
            if (flowFrame.particles) {
                activateInflowParticles(liveCount, flowFrame.particles, subDt);
                liveCount += flowFrame.particles.activeCount;
                applyLiveCount();
            }
            writeDynamicParams(subDt, frameDt);
            encoder.pushDebugGroup("PB-MPM sim step");
            if (flowFrame.recycleActive) {
                dispatch(encoder, "pbmpm-flow", flowPipe, flowBG, particleGroups);
            }
            for (let s = 0; s < stepCount; s++) {
                if (forceSpec && forcePipe && forceBG) {
                    dispatch(encoder, "pbmpm-force", forcePipe, forceBG, particleGroups);
                }
                for (let iter = 0; iter < iterationsMut; iter++) {
                    dispatch(encoder, "pbmpm-constraint", constraintPipe, constraintBG, particleGroups);
                    dispatch(encoder, "pbmpm-clear", clearPipe, clearBG, cellGroups);
                    dispatch(encoder, "pbmpm-p2g", p2gPipe, p2gBG, particleGroups);
                    dispatch(encoder, "pbmpm-grid-update", gridUpdatePipe, gridUpdateBG, cellGroups);
                    dispatch(encoder, "pbmpm-g2p", g2pPipe, g2pBG, particleGroups);
                }
                dispatch(encoder, "pbmpm-integrate", integratePipe, integrateBG, particleGroups);
            }
            if (foamEnabled && foamUpdateBG) {
                foamU32[14] = foamSeed++;
                device.queue.writeBuffer(foamParamsBuffer!, 0, foamData);
                encoder.pushDebugGroup("foam");
                dispatch(encoder, "pbmpm-foam-emit", foamEmitPipe!, foamEmitBG!, particleGroups);
                if (foamActiveParticles) {
                    dispatch(encoder, "pbmpm-foam-active-prepare", foamActivePreparePipe!, foamActivePrepareBG!, 1);
                    dispatchIndirect(encoder, "pbmpm-foam-update", foamUpdatePipe!, foamUpdateBG, foamActiveDispatchBuffer!);
                    dispatch(encoder, "pbmpm-foam-active-finish", foamActiveFinishPipe!, foamActiveFinishBG!, 1);
                    foamActiveSide = 1 - foamActiveSide;
                    diffusePool = {
                        buffer: diffuseBuffer!,
                        headBuffer: foamActiveStateBuffer!,
                        capacity: foamCapacity,
                        activeIndices: foamActiveStateBuffer!,
                        activeIndicesOffset: foamActiveListOffset(foamCapacity, foamActiveSide),
                        drawIndirect: foamDrawIndirectBuffer!,
                    };
                } else {
                    dispatch(encoder, "pbmpm-foam-update", foamUpdatePipe!, foamUpdateBG, foamPoolGroups);
                }
                encoder.popDebugGroup();
            }
            dispatch(encoder, "pbmpm-copy", copyPipe, copyBG, particleGroups);
            encoder.popDebugGroup();
        },
        reset(): void {
            seed();
            clearFoamPool("pbmpm-foam-reset");
        },
        setParam(key: string, value: number): void {
            switch (key) {
                case "gravity":
                    gravity = value;
                    break;
                case "substeps":
                    substepsMut = Math.max(1, Math.round(value));
                    break;
                case "maxSubDtMs":
                    // Largest dt a single sub-step may integrate, in MILLISECONDS. Raising it lets a
                    // frame be covered by fewer sub-steps (cheaper, less stable); lowering it forces
                    // more (costlier, more stable). It never changes how much time a frame advances.
                    maxSubDt = Math.max(1e-4, value / 1000);
                    break;
                case "iterations":
                    iterationsMut = Math.max(1, Math.round(value));
                    break;
                case "liquidRelaxation":
                    liquidRelaxation = value;
                    break;
                case "liquidViscosity":
                    liquidViscosity = value;
                    break;
                case "elasticityRatio":
                    elasticityRatio = value;
                    break;
                case "elasticRelaxation":
                    elasticRelaxation = value;
                    break;
                case "frictionAngle":
                    frictionAngle = value;
                    break;
                case "plasticity":
                    plasticity = value;
                    break;
            }
        },
        setMaterial(material: number): void {
            currentMaterial = Math.min(MATERIAL_VISCO, Math.max(MATERIAL_LIQUID, Math.round(material)));
            seed();
        },
        setSceneSdf(spec: SceneSdfSpec | null): void {
            gridUpdatePipe = pipeline("pbmpm-grid-update", buildGridUpdateWgsl(spec));
            integratePipe = pipeline("pbmpm-integrate", buildIntegrateWgsl(spec));
            gridUpdateBG = buildGridUpdateBG(gridUpdatePipe, spec);
            integrateBG = buildIntegrateBG(integratePipe, spec);
        },
        setEmitters(config: EmitterConfig | null): void {
            flowSeedsInitialParticles = false;
            setFluidFlowConfig(flowState, legacyEmitterConfigToFluidFlow(config, count));
        },
        setFlow(config: FluidFlowConfig | null): void {
            flowSeedsInitialParticles = true;
            setFluidFlowConfig(flowState, config);
        },
        setSpawn(min: [number, number, number], max: [number, number, number], accept?: ((x: number, y: number, z: number) => boolean) | null): void {
            spawnMin[0] = min[0];
            spawnMin[1] = min[1];
            spawnMin[2] = min[2];
            spawnMax[0] = max[0];
            spawnMax[1] = max[1];
            spawnMax[2] = max[2];
            spawnAccept = accept ?? null;
        },
        setWarmup(frames: number): void {
            // Number of frames over which reset()/seed() gradually releases initial
            // particles (0 = all at once). Takes effect on the next seed()/reset().
            warmupFrames = Math.max(0, Math.floor(frames));
            warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(count / warmupFrames)) : count;
        },
        setForceField(spec: ForceFieldSpec | null): void {
            if (!spec) {
                forceSpec = null;
                return;
            }
            const src = buildForceWgsl(spec);
            if (!forcePipe || src !== forceBuiltSrc) {
                forcePipe = pipeline("pbmpm-force", src);
                forceBuiltSrc = src;
                forceBoundSpec = null;
            }
            if (forceBoundSpec !== spec || !forceBG) {
                forceBG = buildForceBG(forcePipe, spec);
                forceBoundSpec = spec;
            }
            forceSpec = spec;
        },
        setFoam(cfg: FoamConfig | null): void {
            if (!cfg) {
                foamEnabled = false;
                clearFoamPool("pbmpm-foam-off-clear");
                return;
            }
            ensureFoam(cfg);
            foamEnabled = true;
        },
        setProfiler(p: FluidProfiler | null): void {
            simProfiler = p;
        },
        get diffuse(): DiffusePool | undefined {
            return diffusePool;
        },
        dispose(): void {
            particleBuffer.destroy();
            cellBuffer.destroy();
            volumeBuffer.destroy();
            positionBuffer.destroy();
            velocityBuffer.destroy();
            debugBuffer.destroy();
            paramsBuffer.destroy();
            disposeFluidFlowState(flowState);
            diffuseBuffer?.destroy();
            diffuseHeadBuffer?.destroy();
            foamActiveStateBuffer?.destroy();
            foamActiveDispatchBuffer?.destroy();
            foamDrawIndirectBuffer?.destroy();
            foamParamsBuffer?.destroy();
        },
    };
}
