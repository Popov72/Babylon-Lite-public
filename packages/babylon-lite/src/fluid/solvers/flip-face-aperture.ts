/** Preserve fully solid/open topology; only intersected faces use the smoothed corner estimate. */
export const FLIP_FACE_APERTURE_WGSL = /* wgsl */ `
fn flipFaceAperture(phi: vec4<f32>, dx: f32) -> f32 {
    if (all(phi <= vec4<f32>(0.0))) { return 0.0; }
    if (all(phi >= vec4<f32>(0.0))) { return 1.0; }
    let corners = clamp(vec4<f32>(0.5) + phi / dx, vec4<f32>(0.0), vec4<f32>(1.0));
    return 0.25 * (corners.x + corners.y + corners.z + corners.w);
}
`;
