import type { BehaviorEventSubscription } from "./types.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";

export function validateEventSubscriptions(behaviorName: string, subscriptions: readonly BehaviorEventSubscription[] | undefined): void {
    if (subscriptions === undefined) {
        return;
    }
    if (subscriptions.length === 0) {
        throw new Error(`[aquanova] ${behaviorName}.events must contain at least one event`);
    }
    for (const subscription of subscriptions) {
        assertBehaviorConfigKeys(subscription, `${behaviorName}.events[]`, ["name", "source"]);
        if (!subscription.name) {
            throw new Error(`[aquanova] ${behaviorName}.events[].name must be a non-empty event name`);
        }
        if (typeof subscription.source === "string" && !subscription.source) {
            throw new Error(`[aquanova] ${behaviorName}.events[].source must be a non-empty entity or door name`);
        }
        if (Array.isArray(subscription.source) && subscription.source.length === 0) {
            throw new Error(`[aquanova] ${behaviorName}.events[].source must contain at least one entity or door name`);
        }
        if (Array.isArray(subscription.source) && subscription.source.some((source) => !source)) {
            throw new Error(`[aquanova] ${behaviorName}.events[].source entries must be non-empty entity or door names`);
        }
    }
}

export function eventSubscriptionMatches(subscription: BehaviorEventSubscription, source: string, event: string): boolean {
    if (subscription.name !== event) {
        return false;
    }
    return typeof subscription.source === "string" ? subscription.source === source : subscription.source.includes(source);
}
