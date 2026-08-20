export interface SystemEventMap {
    frameStart: { deltaMs: number };
    physicsStep: { deltaSeconds: number };
    frameEnd: { deltaMs: number };
}
