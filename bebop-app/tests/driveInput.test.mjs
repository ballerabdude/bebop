import assert from "node:assert/strict";
import test from "node:test";
import {
  DEADMAN_THRESHOLD,
  createDriveArbiter,
  createKeyboardDrive,
  gamepadTwist,
  ignoreDriveKeydown,
  isDriveKey,
  isEditableTarget,
  isModifierKey,
  keyboardTwist,
  mergeDriveSources,
} from "../src/input/driveArbiter.ts";
import { createTwistStream } from "../src/runtime/twistStream.ts";

const LIMITS = { maxLinear: 0.5, maxAngular: 1.0 };

// Arbiter wired to a recording sink.
function rig() {
  const calls = [];
  const states = [];
  const arb = createDriveArbiter(
    {
      set: (vx, wz) => calls.push(["set", vx, wz]),
      stop: () => calls.push(["stop"]),
      halt: () => calls.push(["halt"]),
    },
    (s) => states.push(s),
  );
  const last = () => calls.at(-1);
  return { arb, calls, states, last };
}

// Arbiter feeding a real twist stream over a fake socket.
function streamRig() {
  const frames = [];
  const env = { open: true };
  // Each clock read is 100 ms later, so no frame is held back by the
  // stream's ~33 ms change throttle (no timers fire in this rig).
  let clock = 0;
  const stream = createTwistStream({
    send(vx, wz) {
      if (!env.open) return false;
      frames.push([vx, wz]);
      return true;
    },
    bufferedAmount: () => 0,
    now: () => (clock += 100),
    setTimer: () => 1,
    clearTimer() {},
  });
  const arb = createDriveArbiter({
    set: (vx, wz) => stream.set(vx, wz),
    stop: () => stream.stop(),
    halt: () => stream.requireNeutral(),
  });
  const kb = createKeyboardDrive(arb, () => LIMITS);
  // useDriveInput's halt: forget held keys, then halt the arbiter.
  const halt = () => {
    kb.clear();
    arb.halt();
  };
  const forceStop = () => {
    halt();
    stream.forceStop();
  };
  return { env, stream, arb, frames, kb, halt, forceStop };
}

// ------------------------------------------------------------ pure helpers

test("merge: first non-zero source by priority; neutral only when all are", () => {
  assert.deepEqual(mergeDriveSources({}), { twist: { vx: 0, wz: 0 }, source: null });
  assert.deepEqual(
    mergeDriveSources({ keyboard: { vx: 0.5, wz: 0 }, gamepad: null }),
    { twist: { vx: 0.5, wz: 0 }, source: "keyboard" },
  );
  assert.equal(
    mergeDriveSources({ keyboard: { vx: 0.5, wz: 0 }, gamepad: { vx: 0.1, wz: 0 } }).source,
    "gamepad",
  );
  assert.equal(
    mergeDriveSources({
      keyboard: { vx: 0.5, wz: 0 },
      gamepad: { vx: 0.1, wz: 0 },
      touch: { vx: 0, wz: 0.2 },
    }).source,
    "touch",
  );
  // A zero top-priority source doesn't mask a deflected lower one.
  assert.equal(
    mergeDriveSources({ touch: { vx: 0, wz: 0 }, keyboard: { vx: 0.5, wz: 0 } }).source,
    "keyboard",
  );
});

test("keyboard: WASD / arrows compose, opposites cancel", () => {
  assert.ok(isDriveKey("W") && isDriveKey("ArrowLeft") && !isDriveKey("q"));
  assert.deepEqual(keyboardTwist(new Set(["w"]), LIMITS), { vx: 0.5, wz: 0 });
  assert.deepEqual(keyboardTwist(new Set(["arrowup", "a"]), LIMITS), { vx: 0.5, wz: 1 });
  assert.deepEqual(keyboardTwist(new Set(["w", "s", "d"]), LIMITS), { vx: 0, wz: -1 });
});

test("keyboard: ignored inside text fields and with modifiers held", () => {
  assert.equal(isEditableTarget({ tagName: "INPUT", type: "text" }), true);
  assert.equal(isEditableTarget({ tagName: "INPUT" }), true);
  assert.equal(isEditableTarget({ tagName: "INPUT", type: "range" }), false);
  assert.equal(isEditableTarget({ tagName: "INPUT", type: "checkbox" }), false);
  assert.equal(isEditableTarget({ tagName: "TEXTAREA" }), true);
  assert.equal(isEditableTarget({ tagName: "SELECT" }), true);
  assert.equal(isEditableTarget({ tagName: "DIV", isContentEditable: true }), true);
  assert.equal(isEditableTarget({ tagName: "BUTTON" }), false);
  assert.equal(isEditableTarget(null), false);
  assert.equal(ignoreDriveKeydown({ key: "w", target: { tagName: "BODY" } }), false);
  assert.equal(ignoreDriveKeydown({ key: "w", target: { tagName: "TEXTAREA" } }), true);
  assert.equal(ignoreDriveKeydown({ key: "w", ctrlKey: true }), true);
  assert.equal(ignoreDriveKeydown({ key: "w", metaKey: true }), true);
  assert.equal(ignoreDriveKeydown({ key: "w", altKey: true }), true);
  assert.ok(isModifierKey("Meta") && isModifierKey("Control") && isModifierKey("Alt"));
  assert.ok(!isModifierKey("Shift"));
});

test("gamepad: nothing below the RT deadman, scaled by trigger pressure above it", () => {
  const sticks = { lx: 0, ly: 1, rx: 0.5, deadman: DEADMAN_THRESHOLD - 0.01 };
  assert.equal(DEADMAN_THRESHOLD, 0.4);
  assert.equal(gamepadTwist(sticks, "split", LIMITS), null);
  const light = gamepadTwist({ ...sticks, deadman: 0.4 }, "split", LIMITS);
  const full = gamepadTwist({ ...sticks, deadman: 1 }, "split", LIMITS);
  assert.ok(Math.abs(light.vx - 0.5 * 0.64) < 1e-9);
  assert.ok(Math.abs(full.vx - 0.5) < 1e-9);
  assert.ok(Math.abs(full.wz - -0.5) < 1e-9, "right stick right = turn right");
  // Arcade: the left stick turns.
  assert.deepEqual(gamepadTwist({ lx: 1, ly: 0, rx: 0, deadman: 1 }, "arcade", LIMITS), {
    vx: 0,
    wz: -1,
  });
  // Deadman held, sticks centred: neutral (and never -0).
  const centred = gamepadTwist({ lx: 0, ly: 0, rx: 0, deadman: 1 }, "split", LIMITS);
  assert.ok(Object.is(centred.vx, 0) && Object.is(centred.wz, 0));
  assert.equal(gamepadTwist({ ...sticks, deadman: Number.NaN }, "split", LIMITS), null);
});

// ----------------------------------------------------------------- arbiter

test("WASD keeps driving while a pad is connected but idle", () => {
  const { arb, last } = rig();
  arb.update("gamepad", null); // pad polled, deadman released
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  assert.deepEqual(last(), ["set", 0.5, 0]);
  assert.equal(arb.state().source, "keyboard");
  assert.equal(arb.state().engaged, true);
});

test("a higher-priority deflection takes over; releasing it falls back, neutral stops", () => {
  const { arb, calls } = rig();
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  arb.update("touch", { vx: 0.2, wz: 0.1 });
  arb.update("touch", null);
  arb.update("keyboard", null);
  assert.deepEqual(calls, [
    ["set", 0.5, 0],
    ["set", 0.2, 0.1],
    ["set", 0.5, 0],
    ["stop"],
  ]);
  assert.equal(arb.state().engaged, false);
});

test("identical deflections aren't re-forwarded; every release is", () => {
  const { arb, calls } = rig();
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  arb.update("keyboard", { vx: 0.5, wz: 0 }); // key auto-repeat
  arb.update("keyboard", null);
  arb.update("keyboard", null); // a stray keyup: forwarded (clears latches)
  assert.deepEqual(calls, [["set", 0.5, 0], ["stop"], ["stop"]]);
});

test("disabled: only stops go out; a held input must be released after re-enable", () => {
  const { arb, calls, last } = rig();
  arb.update("gamepad", { vx: 0.3, wz: 0 });
  arb.setEnabled(false);
  assert.deepEqual(last(), ["halt"], "pad still held: a halt, not a release");
  arb.update("gamepad", { vx: 0.35, wz: 0 });
  arb.setEnabled(true);
  assert.equal(arb.state().engaged, false, "held across the gate: blocked");
  assert.deepEqual(arb.state().blocked, ["gamepad"]);
  arb.update("gamepad", { vx: 0.4, wz: 0 });
  assert.equal(calls.filter((c) => c[0] === "set").length, 1);
  arb.update("gamepad", null); // released
  arb.update("gamepad", { vx: 0.4, wz: 0 });
  assert.deepEqual(last(), ["set", 0.4, 0]);
});

test("a key pressed while disabled (E-STOP latched) doesn't drive on reset", () => {
  const { arb, calls } = rig();
  arb.setEnabled(false);
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  arb.setEnabled(true);
  assert.equal(calls.filter((c) => c[0] === "set").length, 0);
  arb.update("keyboard", null);
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  assert.deepEqual(calls.at(-1), ["set", 0.5, 0]);
});

test("halt (blur / hidden): keyboard cleared, held pad and touch need a release", () => {
  const { arb, calls, last } = rig();
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  arb.update("gamepad", { vx: 0.2, wz: 0 });
  arb.halt();
  assert.deepEqual(last(), ["halt"]);
  assert.deepEqual(arb.state().blocked, ["gamepad"]);
  // The pad is still held and keeps ticking: ignored.
  arb.update("gamepad", { vx: 0.25, wz: 0 });
  assert.deepEqual(last(), ["halt"]);
  // A fresh key press is a new operator action and drives.
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  assert.deepEqual(last(), ["set", 0.5, 0]);
  arb.update("keyboard", null);
  arb.update("gamepad", null);
  arb.update("gamepad", { vx: 0.2, wz: 0 });
  assert.deepEqual(last(), ["set", 0.2, 0]);
  assert.ok(calls.length > 0);
});

test("halt with nothing held blocks nothing: the next press drives", () => {
  const { arb, last } = rig();
  arb.halt();
  assert.deepEqual(arb.state().blocked, []);
  arb.update("touch", { vx: 0.1, wz: 0 });
  assert.deepEqual(last(), ["set", 0.1, 0]);
});

test("a dropped pad (disconnect / stalled poll) must report neutral before driving", () => {
  const { arb, last } = rig();
  arb.update("gamepad", { vx: 0.3, wz: 0 });
  arb.drop("gamepad");
  assert.deepEqual(last(), ["halt"]);
  arb.update("gamepad", { vx: 0.3, wz: 0 }); // reconnects still deflected
  assert.deepEqual(last(), ["halt"]);
  arb.update("gamepad", null);
  arb.update("gamepad", { vx: 0.3, wz: 0 });
  assert.deepEqual(last(), ["set", 0.3, 0]);
});

test("non-finite source components count as zero", () => {
  const { arb, last } = rig();
  arb.update("touch", { vx: Number.NaN, wz: Number.POSITIVE_INFINITY });
  assert.deepEqual(last(), ["stop"]);
  arb.update("touch", { vx: Number.NaN, wz: 0.2 });
  assert.deepEqual(last(), ["set", 0, 0.2]);
});

// ------------------------------------------------- arbiter + twist stream

test("re-engage after a link drop: the held joystick doesn't resume, a release re-arms", () => {
  const { env, stream, arb, frames } = streamRig();
  arb.update("touch", { vx: 0.2, wz: 0 });
  assert.equal(frames.length, 1);
  env.open = false;
  stream.linkLost();
  env.open = true;
  // Finger still down, still moving.
  arb.update("touch", { vx: 0.25, wz: 0 });
  arb.update("touch", { vx: 0.3, wz: 0 });
  assert.equal(frames.length, 1);
  assert.equal(stream.needsNeutral(), true);
  // Lift the finger: neutral. No stop owed on the new connection.
  arb.update("touch", null);
  assert.equal(stream.needsNeutral(), false);
  assert.equal(frames.length, 1);
  arb.update("touch", { vx: 0.2, wz: 0 });
  assert.deepEqual(frames.at(-1), [0.2, 0]);
});

test("re-engage after operator-stale: a second input held keeps the latch", () => {
  const { stream, arb, frames } = streamRig();
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  arb.update("gamepad", { vx: 0.2, wz: 0 });
  stream.requireNeutral(); // firmware reported operatorStale
  const n = frames.length;
  arb.update("gamepad", null); // pad released but W still held
  assert.equal(stream.needsNeutral(), true, "W is still down: not neutral");
  assert.equal(frames.length, n);
  arb.update("keyboard", null);
  assert.equal(stream.needsNeutral(), false);
});

test("a spectator's inputs releasing never write a stop", () => {
  // Disabled (spectating): presses and releases only produce un-owed stops.
  const { arb, frames } = streamRig();
  arb.setEnabled(false);
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  arb.update("keyboard", null);
  arb.halt();
  assert.equal(frames.length, 0);
});

test("halt of a keyboard-only drive is a release: the arbiter reports stop", () => {
  const { arb, last } = rig();
  arb.update("keyboard", { vx: 0.5, wz: 0 });
  arb.halt();
  assert.deepEqual(last(), ["stop"], "keyboard cleared, nothing left held");
});

// ------------------------------------------------------- keyboard tracker

test("keyboard tracker: modifiers, text fields and released auto-repeat", () => {
  const { arb, last } = rig();
  const kb = createKeyboardDrive(arb, () => LIMITS);
  assert.equal(kb.keydown({ key: "w", target: { tagName: "TEXTAREA" } }), false);
  assert.equal(last(), undefined);
  assert.equal(kb.keydown({ key: "w" }), true);
  assert.deepEqual(last(), ["set", 0.5, 0]);
  kb.keydown({ key: "Control" }); // modifier releases every drive key
  assert.deepEqual(last(), ["stop"]);
  assert.equal(kb.keydown({ key: "w", repeat: true }), true);
  assert.deepEqual(last(), ["stop"], "repeat of a released key is ignored");
  kb.keydown({ key: "w" }); // fresh press
  assert.deepEqual(last(), ["set", 0.5, 0]);
  kb.focusIn({ tagName: "INPUT", type: "text" });
  assert.deepEqual(last(), ["stop"]);
});

test("held W + halt: a later auto-repeat never reaches the wire", () => {
  // Reviewer case A: halt() used to drop the arbiter's keyboard input but
  // not the hook's held set, so the next repeat re-reported W.
  const { stream, frames, kb, halt } = streamRig();
  kb.keydown({ key: "w" });
  assert.deepEqual(frames.at(-1), [0.5, 0]);
  halt();
  assert.deepEqual(frames.at(-1), [0, 0]);
  const n = frames.length;
  kb.keydown({ key: "w", repeat: true });
  kb.keydown({ key: "w", repeat: true });
  assert.equal(frames.length, n);
  assert.equal(stream.isDriving(), false);
  // Release, then a fresh press drives.
  kb.keyup("w");
  kb.keydown({ key: "w" });
  assert.deepEqual(frames.at(-1), [0.5, 0]);
});

test("held W through link-stale disable + halt: no resume on recovery", () => {
  // Reviewer case B: useDrive disables (link not ok) and halts; the link
  // recovers while W is still physically held and auto-repeating.
  const { arb, frames, kb, halt } = streamRig();
  kb.keydown({ key: "w" });
  arb.setEnabled(false);
  halt();
  arb.setEnabled(true);
  const n = frames.length;
  kb.keydown({ key: "w", repeat: true });
  assert.equal(frames.length, n);
});

test("arbiter halt doesn't clear the stream's latch while an input is held", () => {
  // Reviewer case D: pad held, link drops (stream latched), then the
  // arbiter halts. The pad is blocked; the latch must survive.
  const { env, stream, arb, frames } = streamRig();
  arb.update("gamepad", { vx: 0.2, wz: 0 });
  env.open = false;
  stream.linkLost();
  env.open = true;
  arb.halt();
  assert.equal(stream.needsNeutral(), true);
  // Same on blur: the stream latches first, the arbiter's halt keeps it.
  arb.update("gamepad", null);
  assert.equal(stream.needsNeutral(), false);
  arb.update("gamepad", { vx: 0.2, wz: 0 });
  stream.requireNeutral();
  arb.halt();
  assert.equal(stream.needsNeutral(), true);
  const n = frames.length;
  arb.update("gamepad", { vx: 0.3, wz: 0 });
  assert.equal(frames.length, n);
  arb.update("gamepad", null);
  assert.equal(stream.needsNeutral(), false);
});

test("forceStop with a finger down: moving it doesn't restart motion", () => {
  // Reviewer case C.
  const { stream, arb, frames, forceStop } = streamRig();
  arb.update("touch", { vx: 0.3, wz: 0 });
  forceStop();
  assert.deepEqual(frames.at(-1), [0, 0]);
  const n = frames.length;
  arb.update("touch", { vx: 0.31, wz: 0 });
  assert.equal(frames.length, n);
  assert.equal(stream.needsNeutral(), true);
  arb.update("touch", null);
  arb.update("touch", { vx: 0.31, wz: 0 });
  assert.deepEqual(frames.at(-1), [0.31, 0]);
});
