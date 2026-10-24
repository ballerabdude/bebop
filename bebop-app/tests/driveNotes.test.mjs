import assert from "node:assert/strict";
import test from "node:test";
import {
  DRIVE_EVENTS,
  createDriveNotes,
  describeNotesError,
  intentIsEmpty,
  notesBase,
  parseNotes,
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
  assert.deepEqual(r.value.intent, { text: "keep", from: "lab", to: "kitchen", stampNs: 5, version: 3 });
  assert.equal(r.value.eventsPending, 2);
});

test("parseNotes tolerates nulls and junk", () => {
  const n = parseNotes({ recording: "yes", intent: null });
  assert.equal(n.recording, false);
  assert.equal(n.intent, null);
  assert.equal(n.eventsPending, 0);
  assert.equal(intentIsEmpty(n.intent), true);
  assert.equal(intentIsEmpty({ text: "", from: "", to: "", stampNs: 1, version: 2 }), true);
  assert.equal(intentIsEmpty({ text: "left", from: "", to: "", stampNs: 1, version: 2 }), false);
});

test("setIntent posts a JSON body with text/from/to", async () => {
  const { fn, calls } = mockFetch([
    200,
    { ok: true, intent: { text: "left", from: "", to: "door", stamp_ns: 9, version: 4 } },
  ]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.setIntent("  left ", "", "door");
  assert.equal(calls[0].url, "http://h:9092/intent");
  assert.equal(calls[0].method, "POST");
  assert.equal(calls[0].headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[0].body), { text: "left", from: "", to: "door" });
  assert.ok(r.ok);
  assert.equal(r.value.text, "left");
  assert.equal(r.value.version, 4);
});

test("clearIntent sends {clear: true}, never an empty body", async () => {
  const { fn, calls } = mockFetch([200, { ok: true, intent: { text: "", from: "", to: "", stamp_ns: 1, version: 5 } }]);
  const c = createDriveNotes("http://h:9092", { fetch: fn });
  const r = await c.clearIntent();
  assert.deepEqual(JSON.parse(calls[0].body), { clear: true });
  assert.ok(r.ok);
  assert.equal(intentIsEmpty(r.value), true);
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
