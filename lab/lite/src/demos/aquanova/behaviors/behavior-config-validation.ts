export function assertBehaviorConfigKeys(config: object, path: string, supportedKeys: readonly string[]): void {
    for (const key of Object.keys(config)) {
        if (key !== "name" && !supportedKeys.includes(key)) {
            throw new Error(`[aquanova] ${path}.${key} is not supported`);
        }
    }
}
