import assert from "node:assert/strict";
import test from "node:test";

import { endpoints, normalizeHost, PORTS } from "../src/robot/endpoints.ts";
import {
  DEGRADED_AFTER_MS,
  RESET_AFTER_MS,
  TelemetryStore,
} from "../src/robot/telemetryStore.ts";

// ---------------------------------------------------------------- endpoints

test("endpoint registry builds every service URL from one host", () => {
  const ep = endpoints("10.0.0.5");
  assert.equal(ep.runtimeWs, "ws://10.0.0.5:9090/ws");
  assert.equal(ep.setupWs, "ws://10.0.0.5:9091/ws");
  assert.equal(ep.whep("color_near"), "http://10.0.0.5:9092/whep?stream=color_near");
  assert.equal(ep.voiceWs(), "ws://10.0.0.5:9093/voice");
  assert.equal(ep.voiceWs("session=a"), "ws://10.0.0.5:9093/voice?session=a");
  assert.equal(ep.voiceHttp("/config"), "http://10.0.0.5:9093/config");
  assert.equal(ep.models("/models"), "http://10.0.0.5:9094/models");
  assert.equal(ep.places("/whereami"), "http://10.0.0.5:9095/whereami");
  assert.equal(ep.slam("/structure"), "http://10.0.0.5:9096/structure");
  assert.equal(ep.captures(), "http://10.0.0.5:9090/captures");
  assert.equal(ep.captures("/dl/a.mcap"), "http://10.0.0.5:9090/captures/dl/a.mcap");
  assert.equal(ep.healthz("setup"), "http://10.0.0.5:9091/healthz");
  assert.deepEqual(PORTS, {
    runtime: 9090,
    setup: 9091,
    video: 9092,
    voice: 9093,
    models: 9094,
    places: 9095,
    slam: 9096,
  });
});

test("WHEP session Location resolves relative and absolute", () => {
  const ep = endpoints("bebop.local");
  assert.equal(ep.whepResource("/whep/abc"), "http://bebop.local:9092/whep/abc");
  assert.equal(ep.whepResource("abc"), "http://bebop.local:9092/abc");
  assert.equal(ep.whepResource("http://x:1/whep/z"), "http://x:1/whep/z");
});

test("host input is normalized (scheme, port, path, case)", () => {
  assert.equal(normalizeHost(" http://Bebop.Local:9090/ws "), "bebop.local");
  assert.equal(normalizeHost("192.168.42.1:9091"), "192.168.42.1");
  assert.equal(normalizeHost("10.0.0.2"), "10.0.0.2");
});

// ---------------------------------------------------------- telemetry store

function harness() {
  let now = 0;
  const intervals = [];
  const calls = [];
  const listeners = { telemetry: new Set(), conn: new Set(), estop: new Set() };
  const source = {
    connect: async () => calls.push(["connect"]),
    disconnect: () => calls.push(["disconnect"]),
    forceReconnect: () => calls.push(["forceReconnect"]),
    getSnapshot: async () => ({ tag: "snap" }),
    subscribeTelemetry: async (hz) => calls.push(["sub", hz]),
    unsubscribeTelemetry: async () => calls.push(["unsub"]),
    onTelemetry: (cb) => (listeners.telemetry.add(cb), () => listeners.telemetry.delete(cb)),
    onEStopLatched: (cb) => (listeners.estop.add(cb), () => listeners.estop.delete(cb)),
    onConnectionStateChange: (cb) => (listeners.conn.add(cb), () => listeners.conn.delete(cb)),
    isConnected: () => true,
  };
  const store = new TelemetryStore(source, "h", 9090, {
    now: () => now,
    schedule: (fn) => (fn(), 1),
    cancelSchedule: () => {},
    setInterval: (fn) => (intervals.push(fn), intervals.length),
    clearInterval: () => {},
  });
  return {
    store,
    calls,
    advance: (ms) => {
      now += ms;
      for (const fn of intervals) fn();
    },
    connected: () => listeners.conn.forEach((cb) => cb("connected")),
    frame: (s) => listeners.telemetry.forEach((cb) => cb(s)),
  };
}

test("telemetry rate is the max of live consumers and unsubscribes at zero", () => {
  const h = harness();
  h.store.start();
  h.connected();
  const a = h.store.acquire(5);
  const b = h.store.acquire(30);
  assert.equal(h.store.activeRateHz(), 30);
  b();
  assert.equal(h.store.activeRateHz(), 5);
  a();
  assert.equal(h.store.activeRateHz(), null);
  const subs = h.calls.filter((c) => c[0] === "sub" || c[0] === "unsub");
  assert.deepEqual(subs, [["sub", 5], ["sub", 30], ["sub", 5], ["unsub"]]);
  // Releasing twice is harmless.
  a();
  assert.equal(h.store.activeRateHz(), null);
});

test("silence marks the link degraded, then forces a reconnect", () => {
  const h = harness();
  h.store.start();
  h.connected();
  h.store.acquire(10);
  h.frame({ n: 1 });
  assert.equal(h.store.getLink().state, "connected");
  h.advance(DEGRADED_AFTER_MS + 10);
  assert.equal(h.store.getLink().state, "degraded");
  h.frame({ n: 2 });
  assert.equal(h.store.getLink().state, "connected");
  h.advance(RESET_AFTER_MS + 10);
  assert.ok(h.calls.some((c) => c[0] === "forceReconnect"));
});

test("no consumers means no liveness alarms", () => {
  const h = harness();
  h.store.start();
  h.connected();
  h.advance(RESET_AFTER_MS * 3);
  assert.equal(h.store.getLink().state, "connected");
  assert.ok(!h.calls.some((c) => c[0] === "forceReconnect"));
});

test("frames reach subscribers; stop disconnects", () => {
  const h = harness();
  h.store.start();
  h.connected();
  let seen = null;
  h.store.subscribe(() => (seen = h.store.getSnapshot()));
  h.frame({ n: 7 });
  assert.deepEqual(seen, { n: 7 });
  h.store.stop();
  assert.ok(h.calls.some((c) => c[0] === "disconnect"));
  assert.equal(h.store.getLink().state, "disconnected");
});
