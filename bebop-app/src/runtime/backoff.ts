/// Auto-reconnect backoff for the runtime WebSocket after an unintended
/// close (server restart, Wi-Fi flap, liveness kick): the first retry goes
/// out after 250 ms and each consecutive failure doubles the wait up to 5 s.
/// A successful open resets it, so a brief outage recovers in well under a
/// second while a robot that is genuinely down costs one attempt every 5 s.
export const RECONNECT_BASE_MS = 250;
export const RECONNECT_MAX_MS = 5_000;

/// Delay before reconnect attempt `attempt` (0-based).
export function reconnectDelayMs(attempt: number): number {
  return Math.min(RECONNECT_BASE_MS * 2 ** Math.max(0, attempt), RECONNECT_MAX_MS);
}
