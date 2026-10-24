import assert from "node:assert/strict";
import test from "node:test";

import { ESTOP_OPTIMISTIC_MS, EStopStore } from "../src/robot/estopStore.ts";

function harness({ fail = null } = {}) {
  let now = 1_000;
  const timers = [];
  let frame = null;
  const calls = { estop: 0, reset: 0 };
  const transport = {
    emergencyStop: async () => {
      calls.estop += 1;
      if (fail) throw new Error(fail);
    },
    resetEStop: async () => {
      calls.reset += 1;
      frame = { estopLatched: false, estopReason: "" };
    },
  };
  const telemetry = {
    getSnapshot: () => frame,
    refresh: async () => store.observe(frame),
  };
  const store = new EStopStore(transport, telemetry, {
    now: () => now,
    setTimeout: (fn, ms) => timers.push({ at: now + ms, fn }),
  });
  return {
    store,
    calls,
    setFrame(f) {
      frame = f;
      store.observe(f);
    },
    advance(ms) {
      now += ms;
      for (const t of timers.splice(0)) {
        if (t.at <= now) t.fn();
        else timers.push(t);
      }
    },
    setFail(v) {
      fail = v;
    },
  };
}

test("press latches optimistically and stays latched through stale frames", async () => {
  const h = harness();
  const p = h.store.trigger("operator");
  assert.equal(h.store.getSnapshot().latched, true);
  assert.equal(h.store.getSnapshot().busy, "estop");
  await p;
  h.advance(500);
  h.setFrame({ estopLatched: false, estopReason: "" });
  assert.equal(h.store.getSnapshot().latched, true);
  assert.equal(h.store.getSnapshot().confirmed, false);
  assert.equal(h.store.getSnapshot().error, null);
});

test("unconfirmed latch clears after the optimistic window", async () => {
  const h = harness();
  await h.store.trigger();
  h.setFrame({ estopLatched: false, estopReason: "" });
  h.advance(ESTOP_OPTIMISTIC_MS + 100);
  assert.equal(h.store.getSnapshot().latched, false);
});

test("telemetry confirmation keeps it latched past the window", async () => {
  const h = harness();
  await h.store.trigger();
  h.setFrame({ estopLatched: true, estopReason: "operator" });
  h.advance(ESTOP_OPTIMISTIC_MS + 100);
  h.setFrame({ estopLatched: true, estopReason: "operator" });
  const s = h.store.getSnapshot();
  assert.equal(s.latched, true);
  assert.equal(s.confirmed, true);
  assert.equal(s.reason, "operator");
});

test("failed request unlatches at once and keeps the error until a press succeeds", async () => {
  const h = harness({ fail: "WebSocket not connected" });
  await h.store.trigger();
  let s = h.store.getSnapshot();
  assert.equal(s.latched, false);
  assert.equal(s.busy, null);
  assert.equal(s.error, "WebSocket not connected");
  assert.equal(s.errorKind, "estop");
  // Telemetry frames don't clear the error.
  h.setFrame({ estopLatched: false, estopReason: "" });
  h.advance(ESTOP_OPTIMISTIC_MS + 100);
  assert.equal(h.store.getSnapshot().error, "WebSocket not connected");
  // A retry that fails keeps it; one that succeeds clears it.
  await h.store.trigger();
  assert.equal(h.store.getSnapshot().errorKind, "estop");
  h.setFail(null);
  await h.store.trigger();
  s = h.store.getSnapshot();
  assert.equal(s.error, null);
  assert.equal(s.latched, true);
  assert.equal(h.calls.estop, 3);
});

test("reset clears the latch from a fresh snapshot", async () => {
  const h = harness();
  await h.store.trigger();
  h.setFrame({ estopLatched: true, estopReason: "operator" });
  await h.store.reset();
  const s = h.store.getSnapshot();
  assert.equal(h.calls.reset, 1);
  assert.equal(s.latched, false);
  assert.equal(s.confirmed, false);
  assert.equal(s.busy, null);
});
