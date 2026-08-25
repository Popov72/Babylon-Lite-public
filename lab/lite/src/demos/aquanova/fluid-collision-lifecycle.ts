interface RequiredCollisionOwner<Collision> {
    readonly collision: Collision;
}

interface OptionalCollisionOwner<Collision> {
    readonly collision: Collision | null;
}

export function forEachFluidCollisionSet<Collision>(
    liquefactionSimulations: Iterable<RequiredCollisionOwner<Collision>>,
    behaviorSimulations: Iterable<OptionalCollisionOwner<Collision>>,
    visit: (collision: Collision) => void
): void {
    for (const simulation of liquefactionSimulations) {
        visit(simulation.collision);
    }
    for (const simulation of behaviorSimulations) {
        if (simulation.collision) {
            visit(simulation.collision);
        }
    }
}
