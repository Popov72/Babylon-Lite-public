// Vertical placement of a liquefaction sim's grid, shared by Aquanova and Liquefactor.
//
// Both demos centre the domain on the prop in X/Z and anchor it to the ground in Y: the water falls
// and pools on the floor, so the floor has to be inside the box — anchoring to the prop instead would
// let a puddle drain out of the world. These two helpers exist because the Y placement has two
// non-obvious constraints that both demos must satisfy identically, or the same setting file produces
// a different simulation in each.

/** Extra headroom above the prop, in grid cells. The solver's G2P clamps positions into
 *  `[origin + 2*dx, origin + (dim-3)*dx]`, so 3 cells at the top are unusable; 4 leaves one spare. */
const TOP_CLEARANCE_CELLS = 4;
/** Same idea at the bottom (`origin + 2*dx`), plus half a cell of slack. */
const FLOOR_CLEARANCE_CELLS = 2.5;
/** Floor drop when the grid is fine enough that the cell margin alone is small. */
const MIN_FLOOR_DROP = 0.5;

/**
 * World Y of the domain floor for a ground plane at `groundY`.
 *
 * The domain floor sits *below* the ground rather than on it, because G2P clamps every particle to
 * `origin.y + 2*dx`. Put the domain floor exactly on the ground and that clamp lands 2 cells above
 * it, so at coarse resolutions the water settles visibly floating: at Liquefactor's default radius
 * 0.16 (dx 0.384) a fixed -0.5 floor clamps particles to y = +0.27, a quarter of a metre in the air.
 * Scaling the drop with dx keeps the ground reachable at any resolution.
 */
export function gridFloorY(groundY: number, dx: number): number {
    return groundY - Math.max(MIN_FLOOR_DROP, dx * FLOOR_CLEARANCE_CELLS);
}

/**
 * World Y of the domain top.
 *
 * `height` (the setting file's `grid.y`, in world units above {@link gridFloorY}) is a MINIMUM, not
 * an exact size: a box that stopped below the prop would leave the seeded particles outside the
 * domain, where G2P's clamp flattens them onto the ceiling plane on the first step — a silent pancake
 * rather than an error. Props are not all on the floor (a stacked crate, a door sign), so this is
 * reachable with an otherwise reasonable height. When `height` is 0/undefined the caller's automatic
 * size is used instead.
 */
export function gridTopY(floorY: number, height: number | undefined, propTopY: number, dx: number, autoTopY: number): number {
    const clearTop = propTopY + dx * TOP_CLEARANCE_CELLS;
    return height && height > 0 ? Math.max(floorY + height, clearTop) : Math.max(autoTopY, clearTop);
}
