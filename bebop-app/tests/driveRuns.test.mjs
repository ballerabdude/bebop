// Run prompts and stop causes of the drive-note client: resume, end
// unfinished, the stops / last_stop / resumes fields of /notes, and the pure
// helpers behind the "Run still open" prompt and the stop labels.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  OUTCOMES,
  STOP_CAUSES,
  UNFINISHED,
  agoText,
  armStartsDrive,
  createDriveNotes,
  openRun,
  outcomeLabel,
  parseNotes,
  parseRun,
  parseStop,
  resumePrompt,
  runOutcome,
  stopCauseLabel,
  stopText,
} from "../src/robot/driveNotes.ts";

function mockFetch(...answers) {
  const calls = [];
  const fn = async (url, init = {}) => {
    calls.push({ url, method: init.method, body: init.body });
    const [status, body] = answers.shift() ?? [200, {}];
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  };
  return { fn, calls };
}

const MIN = 60e9;
const RUN_ID = "r20261025_104233_9f3a";

test("resumeRun posts {resume: true, note} and parses run and intent", async () => {
  const { fn, calls } = mockFetch([
    200,
    {
      ok: true,
      intent: { task: "go_to_place", target: "kitchen", run_id: RUN_ID, run_started_ns: 5, resumes: 1 },
      run: { run_id: RUN_ID, task: "go_to_place", target: "kitchen", started_ns: 5, ended_ns: 0, resumes: 1, stops: 2 },
    },
  ]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.resumeRun("  moved the chair ");
  assert.equal(calls[0].url, "http://h:9092/intent");
  assert.equal(calls[0].method, "POST");
  assert.deepEqual(JSON.parse(calls[0].body), { resume: true, note: "moved the chair" });
  assert.ok(r.ok);
  assert.equal(r.value.run.runId, RUN_ID);
  assert.equal(r.value.run.resumes, 1);
  assert.equal(r.value.run.stops, 2);
  assert.equal(r.value.intent.resumes, 1);
});

test("resumeRun without a note sends an empty note; 409 maps to no-run", async () => {
  const { fn, calls } = mockFetch([409, { error: "no run is open", intent: null }]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.resumeRun();
  assert.deepEqual(JSON.parse(calls[0].body), { resume: true, note: "" });
  assert.equal(r.ok, false);
  assert.equal(r.kind, "no-run");
  assert.equal(r.error, "no run is open");
});

test("endRun accepts unfinished with a note", async () => {
  const { fn, calls } = mockFetch([
    200,
    {
      ok: true,
      run: { run_id: RUN_ID, started_ns: 5, ended_ns: 9, outcome: "unfinished" },
      event_recorded: false,
      intent: { text: "", from: "", to: "", stamp_ns: 9, version: 4 },
    },
  ]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.endRun(UNFINISHED, "battery swap");
  assert.deepEqual(JSON.parse(calls[0].body), { end: "unfinished", note: "battery swap" });
  assert.ok(r.ok);
  assert.equal(r.value.run.outcome, "unfinished");
});

test("parseNotes reads stops, last_stop and resumes on the run and the intent", () => {
  const n = parseNotes({
    intent: {
      task: "look_for",
      target: "red mug",
      run_id: RUN_ID,
      run_started_ns: 5,
      stops: 3,
      last_stop: { t_ns: 70, cause: "estop_auto", detail: "wheel fault" },
      resumes: 1,
    },
    recent: [
      {
        run_id: RUN_ID,
        task: "look_for",
        target: "red mug",
        started_ns: 5,
        ended_ns: 0,
        stops: 3,
        last_stop: { t_ns: 70, cause: "estop_auto", detail: "wheel fault" },
        resumes: 1,
      },
      // The saved-state shape: lists, no last_stop (the tail stands in).
      {
        run_id: "r20261025_100000_0001",
        started_ns: 1,
        ended_ns: 2,
        outcome: "arrived",
        stops: [
          { t_ns: 1, cause: "rolled", detail: "" },
          { t_ns: 2, cause: "disarmed", detail: "" },
        ],
        resumes: [{ t_ns: 1, note: "" }],
      },
    ],
  });
  assert.deepEqual(n.intent.lastStop, { tNs: 70, cause: "estop_auto", detail: "wheel fault" });
  assert.equal(n.intent.stops, 3);
  assert.equal(n.intent.resumes, 1);
  assert.deepEqual(
    n.recent.map((r) => [r.stops, r.lastStop?.cause ?? null, r.resumes]),
    [
      [3, "estop_auto", 1],
      [2, "disarmed", 1],
    ],
  );
  // last_stop null, junk or causeless reads as no stop.
  assert.equal(parseRun({ run_id: "x", last_stop: null }).lastStop, null);
  assert.equal(parseStop({ t_ns: 3 }), null);
  assert.equal(parseStop("disarmed"), null);
});

test("runOutcome reads an old empty close as unfinished; open runs have none", () => {
  assert.equal(runOutcome({ endedNs: 0, outcome: "" }), "");
  assert.equal(runOutcome({ endedNs: 5, outcome: "" }), UNFINISHED);
  assert.equal(runOutcome({ endedNs: 5, outcome: "arrived" }), "arrived");
  assert.equal(outcomeLabel(runOutcome({ endedNs: 5, outcome: "" })), "Unfinished");
  // One display map: the label comes from OUTCOMES.
  assert.equal(outcomeLabel(UNFINISHED), OUTCOMES.find((o) => o.outcome === UNFINISHED).label);
});

test("openRun: the intent's unended recent row, an intent-only fallback, or null", () => {
  const intent = { task: "explore", run_id: RUN_ID, run_started_ns: 5, stops: 1, last_stop: { t_ns: 9, cause: "idle" } };
  const row = { run_id: RUN_ID, task: "explore", started_ns: 5, ended_ns: 0 };
  // An older server sends stops only on the intent: carried over.
  const a = openRun(parseNotes({ intent, recent: [row] }));
  assert.equal(a.runId, RUN_ID);
  assert.equal(a.lastStop.cause, "idle");
  assert.equal(a.stops, 1);
  // No recent list (an older server): built from the intent.
  const b = openRun(parseNotes({ intent }));
  assert.equal(b.runId, RUN_ID);
  assert.equal(b.startedNs, 5);
  assert.equal(b.endedNs, 0);
  // Ended row, empty intent, intent without a run, no snapshot: no run open.
  assert.equal(openRun(parseNotes({ intent, recent: [{ ...row, ended_ns: 7 }] })), null);
  assert.equal(openRun(parseNotes({ intent: { text: "", stamp_ns: 1 } })), null);
  assert.equal(openRun(parseNotes({ intent: { text: "keep" } })), null);
  assert.equal(openRun(null), null);
});

test("stop cause labels cover STOP_CAUSES and match session_notes", () => {
  assert.deepEqual(
    STOP_CAUSES.map((c) => c.cause),
    ["disarmed", "wheel_disarmed", "idle", "estop_operator", "estop_auto", "link_lost", "rolled", "recorder_stop"],
  );
  assert.equal(stopCauseLabel("estop_auto"), "E-STOP (auto)");
  assert.equal(stopCauseLabel("estop_operator"), "E-STOP (operator)");
  assert.equal(stopCauseLabel("disarmed"), "disarmed");
  // A cause from a newer server still reads.
  assert.equal(stopCauseLabel("brownout_reset"), "brownout reset");
  // When the server's list is in the tree, the app's must equal it.
  const src = readFileSync(new URL("../../bebop-vision/bebop_vision/session_notes.py", import.meta.url), "utf8");
  const m = src.match(/^STOP_CAUSES[^=]*=\s*\(([\s\S]*?)\)/m);
  if (m) {
    assert.deepEqual(
      [...m[1].matchAll(/"([a-z_]+)"/g)].map((x) => x[1]),
      STOP_CAUSES.map((c) => c.cause),
    );
  }
});

test("agoText and stopText", () => {
  const now = 1000 * MIN;
  assert.equal(agoText(0, now), "");
  assert.equal(agoText(now - 10e9, now), "just now");
  assert.equal(agoText(now - 12 * MIN, now), "12 min ago");
  assert.equal(agoText(now - 125 * MIN, now), "2 h ago");
  assert.equal(agoText(now - 72 * 60 * MIN, now), "3 d ago");
  assert.equal(agoText(now + MIN, now), "just now");
  assert.equal(
    stopText({ tNs: now - 12 * MIN, cause: "estop_auto", detail: "wheel fault" }, now),
    "Stopped: E-STOP (auto): wheel fault, 12 min ago",
  );
  assert.equal(stopText({ tNs: 0, cause: "disarmed", detail: "" }, now), "Stopped: disarmed");
  assert.equal(
    stopText({ tNs: now - 2 * MIN, cause: "wheel_disarmed", detail: "left" }, now),
    "Stopped: wheel disarmed: left, 2 min ago",
  );
  assert.equal(stopText(null, now), "");
});

test("resumePrompt: title, detail lines and the last stop", () => {
  const now = 1000 * MIN;
  const run = parseRun({
    run_id: RUN_ID,
    task: "go_to_place",
    target: "master bedroom",
    text: "the long way round",
    from: "office",
    to: "master bedroom",
    started_ns: now - 25 * MIN,
    last_stop: { t_ns: now - 12 * MIN, cause: "estop_auto", detail: "wheel fault" },
  });
  const p = resumePrompt(run, now, () => "10:42");
  assert.equal(p.title, "Run still open: Go to place master bedroom");
  assert.deepEqual(p.lines, ["the long way round", "From office to master bedroom", "Started 10:42, 25 min ago"]);
  assert.equal(p.stop, "Stopped: E-STOP (auto): wheel fault, 12 min ago");

  // Free-text run, only a destination, no stop, default clock.
  const q = resumePrompt(parseRun({ run_id: RUN_ID, text: "to the door", to: "door", started_ns: now - MIN }), now);
  assert.equal(q.title, "Run still open: to the door");
  assert.deepEqual(q.lines, ["To door", "Started 1 min ago"]);
  assert.equal(q.stop, "");

  // Nothing but an id: the tag stands in.
  assert.equal(resumePrompt(parseRun({ run_id: RUN_ID }), now).title, "Run still open: #9f3a");
});

test("armStartsDrive: only the arm that leaves every wheel armed starts a drive", () => {
  const w = (l, r) => [
    { name: "left", armed: l },
    { name: "right", armed: r },
  ];
  // Arm all from none or from one armed completes the set.
  assert.equal(armStartsDrive(w(false, false), null), true);
  assert.equal(armStartsDrive(w(true, false), null), true);
  // Already all armed: nothing starts.
  assert.equal(armStartsDrive(w(true, true), null), false);
  assert.equal(armStartsDrive(w(true, true), "left"), false);
  // Wheel by wheel: the first toggle does not, the last one does.
  assert.equal(armStartsDrive(w(false, false), "left"), false);
  assert.equal(armStartsDrive(w(true, false), "right"), true);
  assert.equal(armStartsDrive(w(true, false), "left"), false);
  // No wheels reported.
  assert.equal(armStartsDrive([], null), false);
});
