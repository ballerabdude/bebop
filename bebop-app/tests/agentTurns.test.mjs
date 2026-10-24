import assert from "node:assert/strict";
import test from "node:test";

import {
  activeMotion,
  activeTurn,
  agentReducer,
  assistantText,
  INITIAL_AGENT_STATE,
  motionFraction,
  runningTurn,
  summarizeArgs,
} from "../src/voice/agentTurns.ts";

let clock = 1000;
const ev = (event) => ({ type: "event", event, at: (clock += 10) });
const run = (actions, state = INITIAL_AGENT_STATE) => actions.reduce(agentReducer, state);
const send = (id, text = "hi", speak = false) => ({
  type: "send",
  id,
  source: "text",
  text,
  speak,
  at: (clock += 10),
});

test("a text turn streams deltas into one text item and ends done", () => {
  const s = run([
    send("t1", "Where are you?"),
    ev({ type: "turn_start", turn_id: "t1", source: "text", speak: false }),
    ev({ type: "transcript", turn_id: "t1", text: "Where are you?", final: true, source: "text" }),
    ev({ type: "assistant_delta", turn_id: "t1", text: "In the " }),
    ev({ type: "text", delta: "In the " }), // legacy mirror is ignored
    ev({ type: "assistant_delta", turn_id: "t1", text: "office." }),
    ev({ type: "done" }),
    ev({ type: "turn_end", turn_id: "t1", status: "done" }),
  ]);
  assert.equal(s.turns.length, 1);
  const t = s.turns[0];
  assert.equal(t.status, "done");
  assert.equal(t.userText, "Where are you?");
  assert.equal(t.items.length, 1);
  assert.equal(assistantText(t), "In the office.");
  assert.equal(activeTurn(s), null);
});

test("pending until turn_start, running after", () => {
  let s = run([send("t1")]);
  assert.equal(s.turns[0].status, "pending");
  assert.equal(activeTurn(s)?.id, "t1");
  assert.equal(runningTurn(s), null);
  s = run([ev({ type: "turn_start", turn_id: "t1", source: "text", speak: true })], s);
  assert.equal(s.turns[0].status, "running");
  assert.equal(s.turns[0].speak, true);
  assert.equal(runningTurn(s)?.id, "t1");
});

test("tool_call → motion_progress → tool_result pair by call_id; text splits around tools", () => {
  let s = run([
    send("t1", "Turn left"),
    ev({ type: "turn_start", turn_id: "t1", source: "text" }),
    ev({ type: "assistant_delta", turn_id: "t1", text: "Turning. " }),
    ev({ type: "tool_call", turn_id: "t1", call_id: "c1", name: "move_relative", args: { turn_deg: 90 } }),
    ev({ type: "motion_progress", turn_id: "t1", call_id: "c1", done_m: 0, total_m: 0, done_deg: 45, total_deg: 90 }),
  ]);
  let tool = s.turns[0].items[1];
  assert.equal(tool.kind, "tool");
  assert.equal(tool.ok, undefined);
  assert.deepEqual(tool.progress, { doneM: 0, totalM: 0, doneDeg: 45, totalDeg: 90 });
  assert.equal(activeMotion(s)?.callId, "c1");
  assert.equal(motionFraction(tool.progress), 0.5);

  s = run(
    [
      ev({ type: "motion_progress", turn_id: "t1", call_id: "c1", done_m: 0, total_m: 0, done_deg: 90, total_deg: 90 }),
      ev({ type: "tool_result", turn_id: "t1", call_id: "c1", ok: true, text: "turned 90 deg", ms: 2100 }),
      ev({ type: "assistant_delta", turn_id: "t1", text: "Done." }),
      ev({ type: "turn_end", turn_id: "t1", status: "done" }),
    ],
    s,
  );
  const t = s.turns[0];
  assert.deepEqual(
    t.items.map((i) => i.kind),
    ["text", "tool", "text"],
  );
  tool = t.items[1];
  assert.equal(tool.ok, true);
  assert.equal(tool.result, "turned 90 deg");
  assert.equal(tool.ms, 2100);
  assert.equal(tool.progress.doneDeg, 90);
  assert.equal(activeMotion(s), null);
  assert.equal(assistantText(t), "Turning. Done.");
});

test("two tools in one turn keep separate results", () => {
  const s = run([
    send("t1"),
    ev({ type: "turn_start", turn_id: "t1", source: "text" }),
    ev({ type: "tool_call", turn_id: "t1", call_id: "a", name: "where_am_i", args: {} }),
    ev({ type: "tool_call", turn_id: "t1", call_id: "b", name: "list_places", args: {} }),
    ev({ type: "tool_result", turn_id: "t1", call_id: "b", ok: true, text: "B", ms: 5 }),
    ev({ type: "tool_result", turn_id: "t1", call_id: "a", ok: false, text: "A failed", ms: 7 }),
    ev({ type: "turn_end", turn_id: "t1", status: "done" }),
  ]);
  const [a, b] = s.turns[0].items;
  assert.equal(a.name, "where_am_i");
  assert.equal(a.ok, false);
  assert.equal(a.result, "A failed");
  assert.equal(b.ok, true);
  assert.equal(b.result, "B");
});

test("cancel: flag set locally, running tools closed, turn ends cancelled", () => {
  let s = run([
    send("t1"),
    ev({ type: "turn_start", turn_id: "t1", source: "text" }),
    ev({ type: "tool_call", turn_id: "t1", call_id: "c1", name: "move_relative", args: { forward_m: 0.5 } }),
    ev({ type: "motion_progress", turn_id: "t1", call_id: "c1", done_m: 0.1, total_m: 0.5, done_deg: 0, total_deg: 0 }),
    { type: "cancel_requested", at: (clock += 10) },
  ]);
  assert.equal(s.turns[0].cancelRequested, true);
  // The server reports the cancelled tool, then ends the turn.
  s = run(
    [
      ev({ type: "tool_result", turn_id: "t1", call_id: "c1", ok: false, text: "cancelled", ms: 400 }),
      ev({ type: "done" }),
      ev({ type: "turn_end", turn_id: "t1", status: "cancelled" }),
    ],
    s,
  );
  const t = s.turns[0];
  assert.equal(t.status, "cancelled");
  assert.equal(t.cancelRequested, false);
  assert.equal(t.items[0].ok, false);
  assert.equal(t.items[0].result, "cancelled");
  assert.equal(activeTurn(s), null);
});

test("turn_end cancelled closes a tool that never got its result", () => {
  const s = run([
    send("t1"),
    ev({ type: "turn_start", turn_id: "t1", source: "text" }),
    ev({ type: "tool_call", turn_id: "t1", call_id: "c1", name: "robot_status", args: {} }),
    ev({ type: "turn_end", turn_id: "t1", status: "cancelled" }),
  ]);
  assert.equal(s.turns[0].items[0].ok, false);
  assert.equal(s.turns[0].items[0].result, "cancelled");
});

test("turn_rejected marks the oldest unstarted turn, leaves the running one alone", () => {
  const s = run([
    send("t1"),
    ev({ type: "turn_start", turn_id: "t1", source: "text" }),
    send("t2", "second"),
    ev({ type: "turn_rejected", reason: "busy", active_turn_id: "t1", source: "text" }),
  ]);
  assert.equal(s.turns[0].status, "running");
  assert.equal(s.turns[1].status, "rejected");
  assert.match(s.turns[1].error, /previous request/);
  assert.equal(activeTurn(s)?.id, "t1");
});

test("turn_end error carries the message; the legacy error before it is ignored", () => {
  const s = run([
    send("t1"),
    ev({ type: "turn_start", turn_id: "t1", source: "text" }),
    ev({ type: "error", message: "brain unreachable" }),
    ev({ type: "turn_end", turn_id: "t1", status: "error", error: "brain unreachable" }),
  ]);
  assert.equal(s.turns[0].status, "error");
  assert.equal(s.turns[0].error, "brain unreachable");
  assert.equal(s.sessionError, null);
});

test("a bare error with nothing running fails the unstarted turn", () => {
  const s = run([send("t1"), ev({ type: "error", message: "voice service not ready (building)" })]);
  assert.equal(s.turns[0].status, "error");
  assert.equal(s.sessionError, "voice service not ready (building)");
});

test("a bare error with no turn is a session error", () => {
  const s = run([ev({ type: "error", message: "bad JSON frame" })]);
  assert.equal(s.turns.length, 0);
  assert.equal(s.sessionError, "bad JSON frame");
});

test("voice turn: recording → pending → transcript fills the user text", () => {
  let s = run([{ type: "send", id: "v1", source: "voice", speak: true, at: 1 }]);
  assert.equal(s.turns[0].status, "recording");
  s = run([{ type: "recorded", id: "v1" }], s);
  assert.equal(s.turns[0].status, "pending");
  s = run(
    [
      ev({ type: "turn_start", turn_id: "v1", source: "voice", speak: true }),
      ev({ type: "transcript", turn_id: "v1", text: "hello robot", final: true, source: "voice" }),
      ev({ type: "audio_start", turn_id: "v1", sample_rate: 24000, format: "pcm16" }),
    ],
    s,
  );
  assert.equal(s.turns[0].userText, "hello robot");
  assert.equal(s.turns[0].speaking, true);
  s = run(
    [ev({ type: "audio_end", turn_id: "v1" }), ev({ type: "turn_end", turn_id: "v1", status: "done" })],
    s,
  );
  assert.equal(s.turns[0].speaking, false);
  assert.equal(s.turns[0].status, "done");
});

test("empty voice utterance (bare done, nothing running) drops the pending bubble", () => {
  const s = run([
    { type: "send", id: "v1", source: "voice", speak: true, at: 1 },
    { type: "recorded", id: "v1" },
    ev({ type: "done" }),
  ]);
  assert.equal(s.turns.length, 0);
});

test("legacy done of a winding-down turn doesn't drop the next pending voice turn", () => {
  const s = run([
    send("t1"),
    ev({ type: "turn_start", turn_id: "t1", source: "text" }),
    { type: "send", id: "v2", source: "voice", speak: true, at: 2 },
    { type: "recorded", id: "v2" },
    ev({ type: "done" }),
    ev({ type: "turn_end", turn_id: "t1", status: "cancelled" }),
  ]);
  assert.equal(s.turns.length, 2);
  assert.equal(s.turns[1].status, "pending");
});

test("disconnect interrupts pending and running turns", () => {
  const s = run([
    send("t1"),
    ev({ type: "turn_start", turn_id: "t1", source: "text" }),
    ev({ type: "tool_call", turn_id: "t1", call_id: "c1", name: "move_relative", args: {} }),
    { type: "disconnected", at: 5000 },
  ]);
  assert.equal(s.turns[0].status, "interrupted");
  assert.equal(s.turns[0].items[0].ok, false);
  assert.equal(activeTurn(s), null);
});

test("events for a turn the client didn't name create it", () => {
  const s = run([
    ev({ type: "turn_start", turn_id: "srv-1", source: "voice", speak: true }),
    ev({ type: "assistant_delta", turn_id: "srv-1", text: "Hi" }),
    ev({ type: "turn_end", turn_id: "srv-1", status: "done" }),
  ]);
  assert.equal(s.turns.length, 1);
  assert.equal(s.turns[0].source, "voice");
  assert.equal(assistantText(s.turns[0]), "Hi");
});

test("events for unknown turn types or missing ids are ignored", () => {
  const s0 = run([send("t1")]);
  const s1 = run(
    [ev({ type: "assistant_delta", text: "orphan" }), ev({ type: "whatever", turn_id: "t1" })],
    s0,
  );
  assert.deepEqual(s1.turns, s0.turns);
});

test("abandon removes a voice turn silently or marks it failed with a reason", () => {
  let s = run([{ type: "send", id: "v1", source: "voice", speak: true, at: 1 }]);
  assert.equal(run([{ type: "abandon", id: "v1", at: 2 }], s).turns.length, 0);
  s = run([{ type: "abandon", id: "v1", error: "No microphone found", at: 2 }], s);
  assert.equal(s.turns[0].status, "error");
  assert.equal(s.turns[0].error, "No microphone found");
});

test("load marks restored in-flight turns interrupted; clear empties", () => {
  const restored = [
    { id: "a", source: "text", userText: "x", speak: false, status: "done", items: [], createdAt: 1 },
    { id: "b", source: "text", userText: "y", speak: false, status: "running", items: [], createdAt: 2 },
  ];
  const s = run([{ type: "load", turns: restored }]);
  assert.equal(s.turns[0].status, "done");
  assert.equal(s.turns[1].status, "interrupted");
  // `b` is no longer active after load, so nothing is left to drop.
  assert.deepEqual(run([{ type: "clear" }], s), INITIAL_AGENT_STATE);
});

test("summarizeArgs and motionFraction", () => {
  assert.equal(summarizeArgs({}), "");
  assert.equal(summarizeArgs({ forward_m: 0.5, turn_deg: -90 }), "forward_m 0.5 · turn_deg -90");
  assert.equal(summarizeArgs({ name: "kitchen" }), 'name "kitchen"');
  assert.ok(summarizeArgs({ text: "x".repeat(200) }, 20).length <= 20);
  assert.equal(motionFraction({ doneM: 0.25, totalM: 0.5, doneDeg: 0, totalDeg: 0 }), 0.5);
  assert.equal(motionFraction({ doneM: 0.5, totalM: 0.5, doneDeg: 0, totalDeg: 90 }), 0.5);
  assert.equal(motionFraction({ doneM: 2, totalM: 1, doneDeg: 0, totalDeg: 0 }), 1);
  assert.equal(motionFraction({ doneM: 0, totalM: 0, doneDeg: 0, totalDeg: 0 }), 0);
});

test("new conversation mid-turn: the old turn's late events don't come back", () => {
  let s = run([
    send("c-1", "Where are you?"),
    ev({ type: "turn_start", turn_id: "c-1", source: "text" }),
    ev({ type: "tool_call", turn_id: "c-1", call_id: "a", name: "where_am_i", args: {} }),
    { type: "clear", awaitReset: true },
  ]);
  assert.deepEqual(s.turns, []);
  s = run(
    [
      ev({ type: "tool_result", turn_id: "c-1", call_id: "a", ok: false, text: "cancelled" }),
      ev({ type: "assistant_delta", turn_id: "c-1", text: "late" }),
      ev({ type: "error", message: "late legacy error" }),
      ev({ type: "done" }),
      ev({ type: "turn_end", turn_id: "c-1", status: "cancelled" }),
      ev({ type: "reset" }),
    ],
    s,
  );
  assert.deepEqual(s.turns, []);
  assert.equal(s.sessionError, null);
  assert.equal(s.resetting, false);
  // The fresh conversation works normally afterwards.
  s = run([send("c-2"), ev({ type: "turn_start", turn_id: "c-2" }), ev({ type: "turn_end", turn_id: "c-2", status: "done" })], s);
  assert.equal(s.turns.length, 1);
  assert.equal(s.turns[0].status, "done");
});

test("new conversation before turn_start: a late turn_start is ignored too", () => {
  const s = run([
    send("c-1"),
    { type: "clear", awaitReset: true },
    ev({ type: "turn_start", turn_id: "c-1", source: "text" }),
    ev({ type: "turn_end", turn_id: "c-1", status: "cancelled" }),
  ]);
  assert.deepEqual(s.turns, []);
});

const voiceSend = (id) => ({ type: "send", id, source: "voice", speak: false, at: (clock += 10) });

test("disconnect while recording interrupts the voice turn (no stuck pending)", () => {
  const s = run([
    voiceSend("v-1"),
    { type: "disconnected", at: (clock += 10) },
    { type: "recorded", id: "v-1" },
  ]);
  assert.equal(s.turns[0].status, "interrupted");
  assert.ok(s.turns[0].error);
  assert.equal(activeTurn(s), null);
});

test("Stop resolves a turn the server never started", () => {
  const s = run([
    voiceSend("v-1"),
    { type: "recorded", id: "v-1" },
    { type: "cancel_requested", at: (clock += 10) },
  ]);
  assert.equal(s.turns[0].status, "cancelled");
  assert.equal(activeTurn(s), null);
  // A text turn cancelled before turn_start, which the server did start
  // after all: turn_start revives it and turn_end{cancelled} closes it.
  const t = run([
    send("c-1"),
    { type: "cancel_requested", at: (clock += 10) },
    ev({ type: "turn_start", turn_id: "c-1" }),
    ev({ type: "turn_end", turn_id: "c-1", status: "cancelled" }),
  ]);
  assert.equal(t.turns[0].status, "cancelled");
  assert.equal(activeTurn(t), null);
});

test("Stop on a running turn only flags it until turn_end", () => {
  const s = run([
    send("c-1"),
    ev({ type: "turn_start", turn_id: "c-1" }),
    { type: "cancel_requested", at: (clock += 10) },
  ]);
  assert.equal(s.turns[0].status, "running");
  assert.equal(s.turns[0].cancelRequested, true);
});
