import assert from "node:assert/strict";
import test from "node:test";
import {
  TWIST_ERROR_INTERVAL_MS,
  TWIST_KEEPALIVE_MS,
  TWIST_MAX_BUFFERED_BYTES,
  TWIST_MIN_INTERVAL_MS,
  createReplyRegistry,
  createTwistStream,
} from "../src/runtime/twistStream.ts";

// The firmware zeroes a non-zero twist that isn't refreshed within this
// (`server.operator_timeout_ms`, bebop_wheeled.yaml).
const OPERATOR_TIMEOUT_MS = 500;

// A stream wired to a fake socket, clock and timer queue. `frames` records
// every written SetVelocityCommand; nothing is ever acknowledged unless a
// test calls a frame's `onReject` by hand.
function rig(options) {
  const env = {
    t: 0,
    buffered: 0,
    open: true,
    throwOnSend: false,
    frames: [],
    errors: [],
    timers: new Map(),
    nextTimer: 1,
  };
  const stream = createTwistStream(
    {
      send(vx, wz, onReject) {
        if (env.throwOnSend) throw new Error("socket exploded");
        if (!env.open) return false;
        env.frames.push({ t: env.t, vx, wz, onReject });
        return true;
      },
      bufferedAmount: () => env.buffered,
      now: () => env.t,
      setTimer(fn, ms) {
        const id = env.nextTimer++;
        env.timers.set(id, { at: env.t + ms, fn });
        // The stream keeps a single wake-up timer; more means one leaked.
        assert.ok(env.timers.size <= 1, `${env.timers.size} live timers`);
        return id;
      },
      clearTimer(id) {
        env.timers.delete(id);
      },
      onError: (message) => env.errors.push({ t: env.t, message }),
    },
    options,
  );
  // Move the fake clock to absolute time `time`, firing due timers in order.
  // A timer that re-arms itself without advancing the clock would spin this
  // synchronous loop forever (no test timeout can interrupt it), so cap it.
  const at = (time) => {
    for (let fired = 0; ; fired++) {
      let next = null;
      for (const [id, timer] of env.timers) {
        if (timer.at <= time && (next === null || timer.at < next.timer.at)) {
          next = { id, timer };
        }
      }
      if (next === null) break;
      if (fired >= 10_000) {
        throw new Error(`at(${time}): timers keep re-arming at t=${env.t}`);
      }
      env.timers.delete(next.id);
      env.t = Math.max(env.t, next.timer.at);
      next.timer.fn();
    }
    env.t = Math.max(env.t, time);
  };
  const times = () => env.frames.map((f) => f.t);
  const values = () => env.frames.map((f) => [f.vx, f.wz]);
  return { env, stream, at, times, values };
}

// Deterministic PRNG so the property-style test is reproducible.
function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

test("a held twist goes out at once and is re-sent every keepalive", () => {
  const { stream, at, times, values } = rig();
  stream.set(0.3, 0);
  assert.deepEqual(times(), [0]);
  at(450);
  assert.deepEqual(times(), [0, 100, 200, 300, 400]);
  assert.ok(values().every(([vx, wz]) => vx === 0.3 && wz === 0));
  assert.equal(stream.isDriving(), true);
});

test("acks never gate the stream: continuous steering keeps frames within the keepalive", () => {
  // Regression for navd_session_20261023_212953: the old sender waited for
  // each Ack, so one slow reply silenced the stream past the robot's
  // operator watchdog. Here nothing is ever acknowledged.
  const { env, stream, at } = rig();
  const rand = lcg(20261023);
  // Every frame must carry the newest twist requested before it was
  // written: frames fired by timers carry the previous request, a frame
  // written inside set() carries that call's twist.
  let current = [0.2, 0.1];
  let seen = 0;
  const expectNewFrames = (where) => {
    for (const f of env.frames.slice(seen)) {
      assert.deepEqual([f.vx, f.wz], current, `${where}, frame at ${f.t}`);
    }
    seen = env.frames.length;
  };
  stream.set(...current);
  expectNewFrames("first set");
  let t = 0;
  while (t < 5000) {
    t += 1 + Math.floor(rand() * 60); // 1-60 ms between inputs, bursty
    at(t);
    expectNewFrames(`timers up to ${t}`);
    // Mostly new values, sometimes a repeat of the last one (key repeat).
    const vx = rand() < 0.3 ? current[0] : 0.05 + Math.round(rand() * 100) / 100;
    const wz = rand() < 0.3 ? current[1] : Math.round((rand() - 0.5) * 200) / 100;
    current = [vx, wz];
    stream.set(vx, wz);
    expectNewFrames(`set at ${t}`);
  }
  at(t + 1000);
  expectNewFrames("hold after the last input");
  const frames = env.frames;
  assert.ok(frames.length > 100, `${frames.length} frames`);
  for (let i = 1; i < frames.length; i++) {
    const gap = frames[i].t - frames[i - 1].t;
    assert.ok(gap <= TWIST_KEEPALIVE_MS, `gap ${gap} ms at ${frames[i].t}`);
    assert.ok(gap < OPERATOR_TIMEOUT_MS);
    assert.ok(gap >= TWIST_MIN_INTERVAL_MS, `burst ${gap} ms at ${frames[i].t}`);
  }
});

test("changes inside the throttle window collapse into one frame with the newest twist", () => {
  const { stream, at, times, values } = rig();
  stream.set(0.1, 0);
  at(5);
  stream.set(0.2, 0);
  at(12);
  stream.set(0.3, 0);
  at(20);
  stream.set(0.4, 0.5);
  at(TWIST_MIN_INTERVAL_MS - 1);
  assert.deepEqual(times(), [0]);
  at(TWIST_MIN_INTERVAL_MS);
  assert.deepEqual(times(), [0, TWIST_MIN_INTERVAL_MS]);
  assert.deepEqual(values(), [
    [0.1, 0],
    [0.4, 0.5],
  ]);
  // Then the keepalive takes over from the last frame.
  at(TWIST_MIN_INTERVAL_MS + TWIST_KEEPALIVE_MS);
  assert.deepEqual(times(), [0, 33, 133]);
});

test("re-requesting the same twist (key auto-repeat) adds no frames", () => {
  const { stream, at, times } = rig();
  for (let t = 0; t <= 300; t += 10) {
    at(t);
    stream.set(0.5, -0.25);
  }
  assert.deepEqual(times(), [0, 100, 200, 300]);
});

test("a stop goes out immediately, even inside the throttle window over a backed-up socket", () => {
  const { env, stream, at, values, times } = rig();
  stream.set(0.3, 0.2);
  env.buffered = 1_000_000;
  at(5);
  stream.stop();
  assert.deepEqual(times(), [0, 5]);
  assert.deepEqual(values()[1], [0, 0]);
  assert.equal(stream.isDriving(), false);
  // The keepalive is over: no timers left, nothing more is written.
  assert.equal(env.timers.size, 0);
  at(2000);
  assert.equal(env.frames.length, 2);
});

test("set(0, 0) is a stop, including negative zero", () => {
  const { env, stream, at, values } = rig();
  stream.set(0.3, 0);
  at(10);
  stream.set(0, 0);
  assert.deepEqual(values(), [
    [0.3, 0],
    [0, 0],
  ]);
  assert.equal(stream.isDriving(), false);
  assert.equal(env.timers.size, 0);
  at(100);
  stream.set(0.3, 0);
  at(200);
  stream.set(-0, -0);
  assert.deepEqual(values().at(-1), [0, 0]);
  assert.equal(stream.isDriving(), false);
});

test("a stop carries no rejection callback, so it stays silent", () => {
  const { env, stream } = rig();
  stream.set(0.3, 0);
  stream.stop();
  assert.equal(typeof env.frames[0].onReject, "function");
  assert.equal(env.frames[1].onReject, undefined);
});

test("a backed-up socket skips twists instead of queueing them; the newest goes out once drained", () => {
  const { env, stream, at, times, values } = rig();
  stream.set(0.1, 0);
  env.buffered = TWIST_MAX_BUFFERED_BYTES + 1;
  at(40);
  stream.set(0.2, 0);
  at(60);
  stream.set(0.3, 0.1);
  at(400);
  // Neither the changes nor the keepalive were written while congested.
  assert.deepEqual(times(), [0]);
  env.buffered = 0;
  at(400 + TWIST_MIN_INTERVAL_MS);
  assert.equal(env.frames.length, 2);
  assert.deepEqual(values()[1], [0.3, 0.1]);
  assert.ok(env.frames[1].t > 400 && env.frames[1].t <= 400 + TWIST_MIN_INTERVAL_MS);
  assert.ok(!values().some(([vx]) => vx === 0.2), "a stale twist was sent");
  // And the keepalive resumes from there.
  at(env.frames[1].t + TWIST_KEEPALIVE_MS);
  assert.equal(env.frames.length, 3);
});

test("a backed-up socket also skips the keepalive of a steady twist, polling at the throttle rate", () => {
  const { env, stream, at, times, values } = rig();
  stream.set(0.3, 0); // the first frame goes out before the backlog
  env.buffered = TWIST_MAX_BUFFERED_BYTES + 1;
  const timersBefore = env.nextTimer;
  at(1000);
  // An unchanged twist is skipped like a changed one: nothing queues behind the backlog.
  assert.deepEqual(times(), [0]);
  // The congestion re-check runs at most once per throttle interval, not as a busy loop.
  const armed = env.nextTimer - timersBefore;
  assert.ok(armed <= Math.ceil(1000 / TWIST_MIN_INTERVAL_MS) + 1, `${armed} re-checks in 1 s`);
  env.buffered = 0;
  at(1000 + TWIST_MIN_INTERVAL_MS);
  assert.equal(env.frames.length, 2);
  assert.ok(env.frames[1].t > 1000 && env.frames[1].t <= 1000 + TWIST_MIN_INTERVAL_MS);
  assert.deepEqual(values()[1], [0.3, 0]);
});

test("a turn-only change, and the same twist again after a stop, go out at the throttle", () => {
  // Keyboard drive: holding W then pressing A changes only wz; tapping W
  // twice sends a stop and then the very same twist. Neither may wait for
  // the next keepalive.
  const { stream, at, times, values } = rig();
  stream.set(0.3, 0);
  at(150); // keepalive at 100
  stream.set(0.3, 0.5); // 50 ms after the last frame: due now
  assert.deepEqual(times(), [0, 100, 150]);
  assert.deepEqual(values().at(-1), [0.3, 0.5]);
  at(200);
  stream.stop();
  at(240);
  stream.set(0.3, 0.5); // 40 ms after the stop: due now
  assert.deepEqual(times(), [0, 100, 150, 200, 240]);
  assert.deepEqual(values().at(-1), [0.3, 0.5]);
});

test("a buffer exactly at the threshold still sends", () => {
  const { env, stream, times } = rig();
  env.buffered = TWIST_MAX_BUFFERED_BYTES;
  stream.set(0.3, 0);
  assert.deepEqual(times(), [0]);
});

test("rejections of driving twists reach onError at most once per interval", () => {
  const { env, stream, at } = rig();
  stream.set(0.3, 0);
  const reject = env.frames[0].onReject;
  const msg = "another operator (connection 7) is driving";
  reject(msg);
  reject(msg);
  at(TWIST_ERROR_INTERVAL_MS - 1);
  env.frames.at(-1).onReject(msg);
  assert.deepEqual(env.errors, [{ t: 0, message: msg }]);
  at(TWIST_ERROR_INTERVAL_MS);
  env.frames.at(-1).onReject(msg);
  assert.equal(env.errors.length, 2);
  assert.equal(env.errors[1].t, TWIST_ERROR_INTERVAL_MS);
  // Rejections never pause the stream.
  assert.ok(env.frames.length >= 10);
});

test("a closed socket keeps the cycle open and the held twist resumes on reconnect", () => {
  const { env, stream, at, values } = rig();
  env.open = false;
  stream.set(0.3, 0.2);
  at(1000);
  assert.equal(env.frames.length, 0);
  assert.equal(stream.isDriving(), true);
  env.open = true;
  at(1000 + TWIST_KEEPALIVE_MS);
  assert.equal(env.frames.length, 1);
  assert.deepEqual(values()[0], [0.3, 0.2]);
  // A stop while closed must not throw.
  env.open = false;
  stream.stop();
  assert.equal(stream.isDriving(), false);
});

test("a throwing send doesn't kill the keepalive", () => {
  const { env, stream, at } = rig();
  env.throwOnSend = true;
  stream.set(0.3, 0);
  assert.equal(env.frames.length, 0);
  env.throwOnSend = false;
  at(TWIST_KEEPALIVE_MS);
  assert.equal(env.frames.length, 1);
});

test("dispose cancels the keepalive and ignores twists, but a late stop still goes out", () => {
  const { env, stream, at, values } = rig();
  stream.set(0.3, 0);
  const reject = env.frames[0].onReject;
  stream.dispose();
  assert.equal(env.timers.size, 0);
  assert.equal(stream.isDriving(), false);
  stream.set(0.4, 0);
  at(1000);
  assert.equal(env.frames.length, 1);
  // An unmounting joystick's stop arrives after the screen's cleanup.
  stream.stop();
  assert.deepEqual(values().at(-1), [0, 0]);
  // A rejection that lands after dispose doesn't reach an unmounted UI.
  reject("late");
  assert.equal(env.errors.length, 0);
});

test("non-finite components never reach the wire", () => {
  const { stream, values } = rig();
  stream.set(Number.NaN, 0.5);
  assert.deepEqual(values(), [[0, 0.5]]);
  stream.set(Number.POSITIVE_INFINITY, Number.NaN);
  assert.deepEqual(values().at(-1), [0, 0]);
  assert.equal(stream.isDriving(), false);
});

test("stop/go chatter still spaces non-zero frames by the throttle interval", () => {
  const { env, stream, at } = rig();
  for (let t = 0; t <= 100; t += 5) {
    at(t);
    if ((t / 5) % 2 === 0) stream.set(0.3, 0);
    else stream.stop();
  }
  at(300);
  const stops = env.frames.filter((f) => f.vx === 0 && f.wz === 0);
  assert.equal(stops.length, 10, "every stop is written");
  let prev = -Infinity;
  for (const f of env.frames) {
    if (f.vx !== 0) {
      assert.ok(f.t - prev >= TWIST_MIN_INTERVAL_MS, `non-zero frame at ${f.t}`);
    }
    prev = f.t;
  }
});

test("options override the defaults", () => {
  const { stream, at, times } = rig({ keepaliveMs: 50, minIntervalMs: 10 });
  stream.set(0.3, 0);
  at(4);
  stream.set(0.4, 0);
  at(120);
  assert.deepEqual(times(), [0, 10, 60, 110]);
});

test("the reply registry is bounded, evicts the oldest, and forgets on take and clear", () => {
  const reg = createReplyRegistry(3);
  for (let id = 1; id <= 5; id++) reg.add(id, `cb${id}`);
  assert.equal(reg.size, 3);
  assert.equal(reg.take(1), undefined);
  assert.equal(reg.take(2), undefined);
  assert.equal(reg.take(3), "cb3");
  assert.equal(reg.take(3), undefined, "a reply is delivered once");
  assert.equal(reg.size, 2);
  // Re-adding an id refreshes it to newest.
  reg.add(4, "cb4b");
  reg.add(6, "cb6");
  reg.add(7, "cb7");
  assert.equal(reg.take(5), undefined);
  assert.equal(reg.take(4), "cb4b");
  reg.clear();
  assert.equal(reg.size, 0);
  assert.equal(reg.take(6), undefined);
});
