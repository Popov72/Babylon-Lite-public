export function createCheatCodeMatcher(code: string, activate: () => void): (key: string) => void {
    const normalized = code.toLowerCase();
    if (!normalized || [...normalized].some((character) => character.length !== 1)) {
        throw new Error("[aquanova] cheat code must contain at least one character");
    }
    let entered = "";
    return (key: string): void => {
        if (key.length !== 1) {
            return;
        }
        entered = (entered + key.toLowerCase()).slice(-normalized.length);
        if (entered === normalized) {
            entered = "";
            activate();
        }
    };
}
