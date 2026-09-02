const P2G_BASE_CELLS_PER_NODE = 27;
const FIXED_POINT_ACCUMULATION_BUDGET = 1_073_741_760;
const MAX_EXACT_F32_INTEGER = 16_777_216;
const MAX_CELL_PARTICLE_CONTRIBUTORS = Math.floor(FIXED_POINT_ACCUMULATION_BUDGET / P2G_BASE_CELLS_PER_NODE);
const MAX_DISPLACEMENT_PER_SUBSTEP = 0.9;
const MAX_STENCIL_OFFSET = 1.5 * Math.sqrt(3);
const MAX_CELL_OFFSET_COMPONENT = 1.5;
const DENSITY_RATIO_CAP = 32;
const DENSITY_FLOOR_RATIO = 0.25;

// Default used only until the per-substep GPU histogram supplies the actual
// maximum P2G base-cell population.
export const MLS_MIN_CELL_PARTICLE_CONTRIBUTORS = 64;

export function buildLinearDispatchIndexWgsl(workgroupSize: number, fnName = "linearDispatchIndex"): string {
    return /* wgsl */ `
fn ${fnName}(gid: vec3<u32>, numWorkgroups: vec3<u32>) -> u32 {
    return gid.x + gid.y * numWorkgroups.x * ${workgroupSize}u;
}`;
}

export interface MlsFixedPointCodecConfig {
    dx: number;
    subDt: number;
    restDensity: number;
    stiffness: number;
    viscosity: number;
}

export interface MlsFixedPointCodec {
    massScale: number;
    inverseMassScale: number;
    momentumScale: number;
    inverseMomentumScale: number;
    maxResolvedSpeed: number;
    maxAffineFrobenius: number;
    maxVelocityContribution: number;
    densityRatioCap: number;
    densityFloorRatio: number;
    cellContributorLimit: number;
    nodeContributorLimit: number;
    massAccumulationBound: number;
    momentumAccumulationBound: number;
}

/**
 * Derives independent fixed-point codecs that keep the MLS grid's 32-bit atomic
 * accumulators within range without depending on dormant particle capacity.
 * The bound covers:
 * - the 3^3 base cells whose quadratic stencils can reach one node;
 * - the explicit maximum number of particles in each P2G base cell;
 * - particle velocity plus its APIC affine term; and
 * - pressure / viscous stress contributions under the permitted density clamp.
 */
export function createMlsFixedPointCodec(config: MlsFixedPointCodecConfig, cellContributorLimit = MLS_MIN_CELL_PARTICLE_CONTRIBUTORS): MlsFixedPointCodec {
    const dx = Math.max(config.dx, 1.0e-6);
    const subDt = Math.max(config.subDt, 1.0e-6);
    const restDensity = Math.max(config.restDensity, 1.0e-3);
    const stiffness = Math.max(config.stiffness, 0);
    const viscosity = Math.max(config.viscosity, 0);
    if (!Number.isFinite(cellContributorLimit) || cellContributorLimit < 1) {
        throw new Error("[MLS-MPM] Fixed-point codec requires a positive finite P2G base-cell contributor count.");
    }
    cellContributorLimit = Math.ceil(cellContributorLimit);
    if (cellContributorLimit > MAX_CELL_PARTICLE_CONTRIBUTORS) {
        throw new Error(
            `[MLS-MPM] 32-bit fixed-point accumulation has no safe codec for ${cellContributorLimit} particles per P2G base cell under the current timestep and material bounds.`
        );
    }

    // The solver already clamps per-substep advection to 0.9 * dx; keep the grid
    // velocity codec aligned with that reachable motion.
    const maxResolvedSpeed = (MAX_DISPLACEMENT_PER_SUBSTEP * dx) / subDt;
    const maxAffineFrobenius = maxResolvedSpeed / Math.max(MAX_STENCIL_OFFSET * dx, 1.0e-6);
    const pressureScale = ((4 / (dx * dx)) * subDt * (MAX_CELL_OFFSET_COMPONENT * dx)) / restDensity;
    const maxPressureVelocity = pressureScale * stiffness * ((DENSITY_RATIO_CAP - 1) / DENSITY_RATIO_CAP);
    const maxViscousVelocity = (pressureScale * (viscosity * 2 * maxAffineFrobenius)) / DENSITY_FLOOR_RATIO;
    const maxVelocityContribution = 2 * maxResolvedSpeed + maxPressureVelocity + maxViscousVelocity;
    const nodeContributorLimit = cellContributorLimit * P2G_BASE_CELLS_PER_NODE;
    const atomicBudgetPerContributor = Math.floor(FIXED_POINT_ACCUMULATION_BUDGET / nodeContributorLimit);
    if (atomicBudgetPerContributor < 2) {
        throw new Error(
            `[MLS-MPM] 32-bit fixed-point accumulation has no safe codec for ${cellContributorLimit} particles per P2G base cell under the current timestep and material bounds.`
        );
    }
    // Reserve one encoded integer per contributor for round() and f32 evaluation,
    // then cap scales to the largest integer represented exactly by f32.
    const massScale = Math.min(MAX_EXACT_F32_INTEGER, atomicBudgetPerContributor - 1);
    const momentumScale = Math.min(MAX_EXACT_F32_INTEGER, Math.floor((atomicBudgetPerContributor - 1) / Math.max(maxVelocityContribution, 1.0e-6)));
    if (!(massScale >= 1) || !(momentumScale >= 1)) {
        throw new Error(
            `[MLS-MPM] 32-bit fixed-point accumulation has no safe codec for ${cellContributorLimit} particles per P2G base cell under the current timestep and material bounds.`
        );
    }
    const massAccumulationBound = nodeContributorLimit * (massScale + 0.5);
    const momentumAccumulationBound = nodeContributorLimit * (maxVelocityContribution * momentumScale + 0.5);
    return {
        massScale,
        inverseMassScale: 1 / massScale,
        momentumScale,
        inverseMomentumScale: 1 / momentumScale,
        maxResolvedSpeed,
        maxAffineFrobenius,
        maxVelocityContribution,
        densityRatioCap: DENSITY_RATIO_CAP,
        densityFloorRatio: DENSITY_FLOOR_RATIO,
        cellContributorLimit,
        nodeContributorLimit,
        massAccumulationBound,
        momentumAccumulationBound,
    };
}

export const MLS_FIXED_POINT_CODEC_FINALIZATION_WGSL = /* wgsl */ `
struct MlsFixedPointCodecResult {
    codec: vec4<f32>,
    valid: u32,
}

const MLS_FIXED_POINT_ACCUMULATION_BUDGET: u32 = ${FIXED_POINT_ACCUMULATION_BUDGET}u;
const MLS_P2G_BASE_CELLS_PER_NODE: u32 = ${P2G_BASE_CELLS_PER_NODE}u;
const MLS_MAX_EXACT_F32_INTEGER: u32 = ${MAX_EXACT_F32_INTEGER}u;
const MLS_MAX_CELL_PARTICLE_CONTRIBUTORS: u32 = ${MAX_CELL_PARTICLE_CONTRIBUTORS}u;

fn deriveMlsFixedPointCodec(cellPopulation: u32, maxVelocityContribution: f32) -> MlsFixedPointCodecResult {
    let population = max(cellPopulation, 1u);
    if (population > MLS_MAX_CELL_PARTICLE_CONTRIBUTORS) {
        return MlsFixedPointCodecResult(vec4<f32>(0.0), 0u);
    }
    let nodeContributors = population * MLS_P2G_BASE_CELLS_PER_NODE;
    let perContributorBudget = MLS_FIXED_POINT_ACCUMULATION_BUDGET / nodeContributors;
    if (perContributorBudget < 2u) {
        return MlsFixedPointCodecResult(vec4<f32>(0.0), 0u);
    }
    let availableBudget = perContributorBudget - 1u;
    let massScale = f32(min(MLS_MAX_EXACT_F32_INTEGER, availableBudget));
    let momentumScale = min(
        f32(MLS_MAX_EXACT_F32_INTEGER),
        floor(f32(availableBudget) / max(maxVelocityContribution, 1.0e-6))
    );
    if (!(momentumScale >= 1.0)) {
        return MlsFixedPointCodecResult(vec4<f32>(0.0), 0u);
    }
    return MlsFixedPointCodecResult(vec4<f32>(massScale, 1.0 / massScale, momentumScale, 1.0 / momentumScale), 1u);
}`;

export const MLS_FIXED_POINT_CODEC_WGSL = /* wgsl */ `
fn decodeMass(x: i32, p: Params) -> f32 {
    return f32(x) * p.codec.y;
}

fn decodeMomentum(x: i32, p: Params) -> f32 {
    return f32(x) * p.codec.w;
}

fn encodeMass(x: f32, p: Params) -> i32 {
    let finite = select(0.0, x, x == x);
    return i32(round(clamp(finite, 0.0, 1.0) * p.codec.x));
}

fn encodeMomentum(x: f32, p: Params) -> i32 {
    let finite = select(0.0, x, x == x);
    return i32(round(clamp(finite, -p.codecLimits.z, p.codecLimits.z) * p.codec.z));
}

fn clampCodecVelocity(v: vec3<f32>, p: Params) -> vec3<f32> {
    let maxLen = p.codecLimits.x;
    let len = length(v);
    if (len > maxLen && len > 1.0e-8) {
        return v * (maxLen / len);
    }
    return v;
}

fn clampCodecAffine(m: mat3x3<f32>, p: Params) -> mat3x3<f32> {
    let frob = sqrt(dot(m[0], m[0]) + dot(m[1], m[1]) + dot(m[2], m[2]));
    let maxFrob = p.codecLimits.y;
    if (frob > maxFrob && frob > 1.0e-8) {
        let s = maxFrob / frob;
        return mat3x3<f32>(m[0] * s, m[1] * s, m[2] * s);
    }
    return m;
}

fn clampPressureDensity(rawDensity: f32, restDensity: f32, p: Params) -> f32 {
    let rest = max(restDensity, 1.0e-3);
    return min(rawDensity, rest * p.codecLimits.w);
}

fn clampStressDensity(rawDensity: f32, restDensity: f32, p: Params) -> f32 {
    let rest = max(restDensity, 1.0e-3);
    return clamp(rawDensity, rest * p.codecExtra.x, rest * p.codecLimits.w);
}`;
