/** Density relaxation must not become arbitrarily fast when the first frame/substep is short. */
export const FLIP_DENSITY_CORRECTION_WGSL = /* wgsl */ `
fn flipDensityExpansion(compression: f32, subDt: f32) -> f32 {
    return min(compression * 0.1, 0.5) / max(subDt, 1.0 / 120.0);
}
`;
