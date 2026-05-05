// Per-endpoint singleton cache for `RuntimeTransport` instances.
//
// Why this exists: React 18+ StrictMode runs every effect twice in dev,
// which means a naive `useEffect(() => { const t = new RuntimeTransport();
// t.connect(); return () => t.disconnect(); })` opens two WebSockets per
// mount cycle. Even with a perfectly robust connect/disconnect lifecycle,
// the *server* sees a brief flurry of connect/close events and (worse)
// any in-flight requests on the first transport will never resolve once
// it's torn down.
//
// By keying the transport on `${ip}:${port}` and storing it at module
// scope, the StrictMode remount, the back-button → re-entry to motors,
// and any other "connect to the same robot twice" scenario all reuse a
// single live WebSocket. The transport itself is internally idempotent
// (`connect()` is a no-op when `isConnected()`), and the cache only
// disposes when the operator explicitly switches endpoints.

import { RuntimeTransport } from "./wsTransport";

const cache = new Map<string, RuntimeTransport>();

function key(ip: string, port: number): string {
  return `${ip}:${port}`;
}

/** Return the cached transport for this endpoint, creating one on miss. */
export function getOrCreateRuntimeTransport(
  ip: string,
  port: number,
): RuntimeTransport {
  const k = key(ip, port);
  let t = cache.get(k);
  if (!t) {
    t = new RuntimeTransport();
    cache.set(k, t);
  }
  return t;
}

/** Tear down and forget the transport for an endpoint. Safe to call when
 *  no entry exists. */
export function disposeRuntimeTransport(ip: string, port: number): void {
  const k = key(ip, port);
  const t = cache.get(k);
  if (!t) return;
  cache.delete(k);
  try {
    t.disconnect();
  } catch {
    /* ignore */
  }
}
