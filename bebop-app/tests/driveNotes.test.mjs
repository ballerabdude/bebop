import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  DRIVE_EVENTS,
  INTENT_PRESETS,
  OUTCOMES,
  TASKS,
  createDriveNotes,
  describeNotesError,
  intentIsEmpty,
  intentLabel,
  notesBase,
  parseIntent,
  parseNotes,
  parseRun,
  runTag,
} from "../src/robot/driveNotes.ts";

// Mocked fetch: records calls and answers from a queue of [status, body].
function mockFetch(...answers) {
  const calls = [];
  const fn = async (url, init = {}) => {
    calls.push({ url, method: init.method, headers: init.headers, body: init.body });
    const next = answers.shift();
    if (next instanceof Error) throw next;
    const [status, body] = next ?? [200, {}];
    return new Response(typeof body === "string" ? body : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  };
  return { fn, calls };
}

test("notesBase targets the videoserver port", () => {
  assert.equal(notesBase("bebop.local"), "http://bebop.local:9092");
});

test("getNotes parses the snapshot", async () => {
  const { fn, calls } = mockFetch([
    200,
    {
      recording: true,
      intent: { text: "keep", from: "lab", to: "kitchen", stamp_ns: 5, version: 3 },
      events_pending: 2,
      places_pending: 1,
      events_dropped: 0,
      places_dropped: 0,
    },
  ]);
  const c = createDriveNotes("http://h:9092/", { fetch: fn });
  const r = await c.getNotes();
  assert.equal(calls[0].url, "http://h:9092/notes");
  assert.equal(calls[0].method, "GET");
  assert.equal(calls[0].body, undefined);
  assert.ok(r.ok);
  assert.equal(r.value.recording, true);
  assert.deepEqual(r.value.intent, {
    text: "keep",
    from: "lab",
    to: "kitchen",
    task: "",
    target: "",
    runId: "",
    runStartedNs: 0,
    stampNs: 5,
    version: 3,
    stops: 0,
    lastStop: null,
    resumes: 0,
  });
  assert.deepEqual(r.value.recent, []);
  assert.equal(r.value.eventsPending, 2);
});

test("parseNotes tolerates nulls and junk", () => {
  const n = parseNotes({ recording: "yes", intent: null });
  assert.equal(n.recording, false);
  assert.equal(n.intent, null);
  assert.equal(n.eventsPending, 0);
  assert.equal(intentIsEmpty(n.intent), true);
  assert.deepEqual(n.recent, []);
  const blank = parseIntent({ stamp_ns: 1, version: 2 });
  assert.equal(intentIsEmpty(blank), true);
  assert.equal(intentIsEmpty({ ...blank, text: "left" }), false);
  assert.equal(intentIsEmpty({ ...blank, task: "explore" }), false);
  assert.deepEqual(parseNotes({ recent: "nope" }).recent, []);
  assert.deepEqual(parseNotes({ recent: [null, 3, { text: "no id" }] }).recent, []);
});

test("setIntent posts every field with mode new by default", async () => {
  const { fn, calls } = mockFetch([
    200,
    {
      ok: true,
      intent: {
        text: "left",
        from: "",
        to: "door",
        task: "maneuver",
        target: "left",
        run_id: "r20261025_104233_9f3a",
        run_started_ns: 77,
        stamp_ns: 9,
        version: 4,
      },
    },
  ]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.setIntent({ text: "  left ", to: "door", task: "maneuver", target: "left" });
  assert.equal(calls[0].url, "http://h:9092/intent");
  assert.equal(calls[0].method, "POST");
  assert.equal(calls[0].headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[0].body), {
    mode: "new",
    text: "left",
    from: "",
    to: "door",
    task: "maneuver",
    target: "left",
  });
  assert.ok(r.ok);
  assert.equal(r.value.text, "left");
  assert.equal(r.value.runId, "r20261025_104233_9f3a");
  assert.equal(r.value.runStartedNs, 77);
  assert.equal(r.value.version, 4);
});

test("setIntent repeat sends all fields; continue sends only the given ones", async () => {
  const { fn, calls } = mockFetch([200, { ok: true, intent: null }], [200, { ok: true, intent: null }]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  await c.setIntent({ task: "explore" }, "repeat");
  assert.deepEqual(JSON.parse(calls[0].body), {
    mode: "repeat",
    text: "",
    from: "",
    to: "",
    task: "explore",
    target: "",
  });
  const r = await c.setIntent({ target: " kitchen " }, "continue");
  assert.deepEqual(JSON.parse(calls[1].body), { mode: "continue", target: "kitchen" });
  assert.ok(r.ok);
  assert.equal(r.value, null);
});

test("setIntent continue 409 maps to no-run, not not-recording", async () => {
  const { fn } = mockFetch([409, { error: "no current run to continue" }]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.setIntent({ text: "x" }, "continue");
  assert.equal(r.ok, false);
  assert.equal(r.kind, "no-run");
  assert.equal(r.status, 409);
  assert.match(describeNotesError(r), /No run to continue/);
});

test("setIntent rejects an unknown task or mode locally", async () => {
  const { fn, calls } = mockFetch();
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const a = await c.setIntent({ task: "dance" });
  assert.equal(a.ok, false);
  assert.equal(a.kind, "invalid");
  const b = await c.setIntent({ text: "x" }, "later");
  assert.equal(b.kind, "invalid");
  assert.equal(calls.length, 0);
});

test("repeatRun posts {repeat: id}", async () => {
  const { fn, calls } = mockFetch([
    200,
    { ok: true, intent: { task: "go_to_place", target: "kitchen", run_id: "r20261025_110000_0001", version: 7 } },
  ]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.repeatRun("r20261025_104233_9f3a");
  assert.deepEqual(JSON.parse(calls[0].body), { repeat: "r20261025_104233_9f3a" });
  assert.ok(r.ok);
  assert.equal(r.value.runId, "r20261025_110000_0001");
  const empty = await c.repeatRun("");
  assert.equal(empty.kind, "invalid");
  assert.equal(calls.length, 1);
});

test("endRun posts {end, note} and parses run, event_recorded, intent", async () => {
  const { fn, calls } = mockFetch(
    [
      200,
      {
        ok: true,
        run: {
          run_id: "r20261025_104233_9f3a",
          text: "",
          from: "lab",
          to: "kitchen",
          task: "go_to_place",
          target: "kitchen",
          started_ns: 10,
          ended_ns: 20,
          outcome: "arrived",
          interruptions: 1,
        },
        event_recorded: false,
        intent: null,
      },
    ],
    [
      200,
      {
        ok: true,
        run: { run_id: "r20261025_104233_9f3b", task: "explore", run_started_ns: 30, stamp_ns: 31, version: 2 },
        event_recorded: true,
        intent: { task: "explore", run_id: "r20261025_104233_9f3b", run_started_ns: 30, version: 2 },
      },
    ],
  );
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const a = await c.endRun("arrived", " made it ");
  assert.deepEqual(JSON.parse(calls[0].body), { end: "arrived", note: "made it" });
  assert.ok(a.ok);
  assert.equal(a.value.eventRecorded, false);
  assert.equal(a.value.intent, null);
  assert.deepEqual(a.value.run, {
    runId: "r20261025_104233_9f3a",
    text: "",
    from: "lab",
    to: "kitchen",
    task: "go_to_place",
    target: "kitchen",
    startedNs: 10,
    endedNs: 20,
    outcome: "arrived",
    interruptions: 1,
    stops: 0,
    lastStop: null,
    resumes: 0,
  });
  // interrupted: the run stays open; an intent-shaped run dict is accepted.
  const b = await c.endRun("interrupted");
  assert.deepEqual(JSON.parse(calls[1].body), { end: "interrupted", note: "" });
  assert.ok(b.ok);
  assert.equal(b.value.eventRecorded, true);
  assert.equal(b.value.run.startedNs, 30);
  assert.equal(b.value.run.endedNs, 0);
  assert.equal(b.value.intent.runId, "r20261025_104233_9f3b");
});

test("endRun rejects an unknown outcome locally", async () => {
  const { fn, calls } = mockFetch();
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.endRun("won");
  assert.equal(r.ok, false);
  assert.equal(r.kind, "invalid");
  assert.equal(calls.length, 0);
});

test("parseNotes reads the recent runs, newest first as sent", () => {
  const n = parseNotes({
    recording: false,
    intent: { task: "look_for", target: "red mug", run_id: "r20261025_120000_abcd", run_started_ns: 5 },
    recent: [
      { run_id: "r20261025_120000_abcd", task: "look_for", target: "red mug", started_ns: 5, ended_ns: 0 },
      { run_id: "r20261025_110000_00ff", text: "to the door", started_ns: 1, ended_ns: 3, outcome: "gave_up" },
    ],
  });
  assert.equal(n.intent.runId, "r20261025_120000_abcd");
  assert.deepEqual(
    n.recent.map((r) => [r.runId, r.endedNs, r.outcome, r.interruptions]),
    [
      ["r20261025_120000_abcd", 0, "", 0],
      ["r20261025_110000_00ff", 3, "gave_up", 0],
    ],
  );
  assert.equal(parseRun({ text: "no id" }), null);
});

test("runTag and intentLabel", () => {
  assert.equal(runTag("r20261025_104233_9f3a"), "#9f3a");
  assert.equal(runTag(""), "");
  assert.equal(runTag("odd"), "#odd");
  assert.equal(intentLabel({ text: "", to: "", task: "go_to_place", target: "kitchen" }), "Go to place kitchen");
  assert.equal(intentLabel({ text: "", to: "", task: "go_through_door", target: "left" }), "Through door Left");
  assert.equal(intentLabel({ text: "left", to: "", task: "maneuver", target: "turn-around" }), "Maneuver Turn around");
  assert.equal(intentLabel({ text: "to the lab", to: "lab", task: "", target: "" }), "to the lab → lab");
  assert.equal(intentLabel({ text: "", to: "kitchen", task: "go_to_place", target: "kitchen" }), "Go to place kitchen");
});

test("task and outcome vocabularies match the contract list", () => {
  assert.deepEqual(
    TASKS.map((t) => t.task),
    ["hold", "go_to_place", "go_through_door", "follow_step", "look_for", "explore", "return", "maneuver"],
  );
  const maneuver = TASKS.find((t) => t.task === "maneuver");
  assert.deepEqual(
    maneuver.options.map((o) => o.value),
    INTENT_PRESETS.map((p) => p.text),
  );
  assert.deepEqual(
    TASKS.find((t) => t.task === "go_through_door").options.map((o) => o.value),
    ["left", "right", "ahead"],
  );
  assert.deepEqual(
    OUTCOMES.map((o) => [o.outcome, o.ends]),
    [
      ["arrived", true],
      ["found", true],
      ["gave_up", true],
      ["unfinished", true],
      ["interrupted", false],
    ],
  );
});

test("clearIntent sends {clear: true}, never an empty body", async () => {
  const { fn, calls } = mockFetch(
    [200, { ok: true, intent: { text: "", from: "", to: "", stamp_ns: 1, version: 5 } }],
    [200, { ok: true, intent: null }],
  );
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.clearIntent();
  assert.deepEqual(JSON.parse(calls[0].body), { clear: true });
  assert.ok(r.ok);
  assert.equal(intentIsEmpty(r.value), true);
  const n = await c.clearIntent();
  assert.ok(n.ok);
  assert.equal(n.value, null);
});

test("addEvent success returns the stamped event", async () => {
  const { fn, calls } = mockFetch([200, { ok: true, event: { kind: "obstacle", note: "", stamp_ns: 42 } }]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.addEvent("Obstacle");
  assert.equal(calls[0].url, "http://h:9092/event");
  assert.deepEqual(JSON.parse(calls[0].body), { kind: "obstacle", note: "" });
  assert.ok(r.ok);
  assert.deepEqual(r.value, { kind: "obstacle", note: "", stampNs: 42 });
});

test("addEvent 409 maps to not-recording with the server message", async () => {
  const { fn } = mockFetch([409, { error: "no drive is being recorded", recording: false }]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.addEvent("recovery");
  assert.equal(r.ok, false);
  assert.equal(r.kind, "not-recording");
  assert.equal(r.status, 409);
  assert.equal(r.error, "no drive is being recorded");
  assert.match(describeNotesError(r), /Not recording/);
});

test("addEvent rejects a bad kind locally without a request", async () => {
  const { fn, calls } = mockFetch();
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.addEvent("9 bad kind");
  assert.equal(r.ok, false);
  assert.equal(r.kind, "invalid");
  assert.equal(calls.length, 0);
});

test("400, 503, other statuses and non-JSON bodies", async () => {
  const { fn } = mockFetch(
    [400, { error: "JSON object body required" }],
    [503, { error: "no session notes" }],
    [500, "<html>boom</html>"],
  );
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const a = await c.addEvent("note");
  assert.equal(a.kind, "invalid");
  assert.equal(a.error, "JSON object body required");
  const b = await c.getNotes();
  assert.equal(b.kind, "unavailable");
  const d = await c.getNotes();
  assert.equal(d.kind, "network");
  assert.equal(d.status, 500);
  assert.equal(d.error, "HTTP 500");
});

test("network errors and timeouts become network results", async () => {
  const { fn } = mockFetch(new TypeError("Failed to fetch"));
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.getNotes();
  assert.equal(r.ok, false);
  assert.equal(r.kind, "network");
  assert.equal(r.error, "Failed to fetch");

  const hang = (_url, init) =>
    new Promise((_res, rej) => {
      init.signal.addEventListener("abort", () => {
        const e = new Error("aborted");
        e.name = "AbortError";
        rej(e);
      });
    });
  const slow = createDriveNotes("http://h:9092", { fetch: hang, timeoutMs: 20 });
  const t = await slow.getNotes();
  assert.equal(t.kind, "network");
  assert.equal(t.error, "timed out");
});

test("event presets are valid kinds with unique hotkeys 1-4", () => {
  assert.deepEqual(
    DRIVE_EVENTS.map((e) => e.hotkey),
    ["1", "2", "3", "4"],
  );
  for (const e of DRIVE_EVENTS) assert.match(e.kind, /^[a-z][a-z0-9_-]{0,31}$/);
});

// Reads TASKS out of bebop-vision's session_notes.py (the server's copy of
// the vocabulary) so the app's TaskSpec cannot drift from it unnoticed.
// Server value: a tuple is a closed set ("()" no target), None is free text.
test("TASKS target kinds match session_notes.TASKS", () => {
  const src = readFileSync(new URL("../../bebop-vision/bebop_vision/session_notes.py", import.meta.url), "utf8");
  const block = src.match(/^TASKS:[^\n]*\{\n([\s\S]*?)^\}/m);
  assert.ok(block, "TASKS dict not found in session_notes.py");
  const server = new Map();
  for (const m of block[1].matchAll(/^\s*"([a-z_]+)":\s*(None|\(([^)]*)\))/gm)) {
    server.set(m[1], m[2] === "None" ? null : [...(m[3] ?? "").matchAll(/"([^"]+)"/g)].map((x) => x[1]));
  }
  assert.deepEqual(TASKS.map((t) => t.task), [...server.keys()]);
  // The server accepts a free-text target for these; the contract gives
  // them none, so the app never sends one. Compatible, and kept explicit.
  const appNoneServerFree = new Set(["explore", "return"]);
  for (const t of TASKS) {
    const s = server.get(t.task);
    if (t.target === "choice") assert.deepEqual(t.options.map((o) => o.value), s, t.task);
    else if (t.target === "none") {
      if (appNoneServerFree.has(t.task)) assert.equal(s, null, t.task);
      else assert.deepEqual(s, [], t.task);
    } else assert.equal(s, null, t.task);
  }
});
