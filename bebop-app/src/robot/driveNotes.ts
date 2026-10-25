// Drive-note client for the bebop-vision videoserver (:9092).
//
// The recorder writes operator notes into the active drive MCAP
// (bebop_vision/session_notes.py, routes in videoserver.py):
//
//   POST /intent {text, from, to, task, target, mode}
//                                  sticky operator intent, accepted any time.
//                                  mode "new" and "repeat" start a run (fresh
//                                  run id); "continue" keeps the current run
//                                  id and updates only the fields it is given
//                                  (409 when no run is open). All fields
//                                  empty clears the intent.
//                {"clear": true}   clears it (the run closes as unfinished)
//                {"repeat": id}    starts a new run copying a recent one
//                {"end": outcome, "note": ...}
//                                  arrived / found / gave_up end the run;
//                                  unfinished closes it without an ending
//                                  (the operator declined to continue it);
//                                  interrupted only marks it. Accepted whether
//                                  or not a segment records; the marker event
//                                  lands only while one does (event_recorded).
//                {"resume": true, "note": ...}
//                                  the operator chose to continue the open
//                                  run (409 when none is open); the "resume"
//                                  event is held for the next segment when
//                                  nothing records.
//   POST /event  {kind, note}      marker in the current segment; 409 when
//                                  no segment is recording (wheels not armed
//                                  in DialIn, E-STOP closed it, ...)
//   GET  /notes                    {recording, intent, recent, events_pending, ...}
//
// The open run also carries why its recordings stopped (stops, last_stop:
// disarmed, E-STOP, link lost, ...) and how often it was resumed, so the
// app can ask "continue or close?" before the next drive instead of
// guessing what a break was.
//
// A run is one attempt at a task. Separate attempts with identical text get
// separate run ids ("r" + local YYYYMMDD_HHMMSS + "_" + 4 hex), so the
// segments of one attempt can be grouped later and two attempts told apart.
//
// 400 = bad body / kind / task / outcome / mode; 503 = the vision process
// runs without session notes. Kinds must match [a-z][a-z0-9_-]{0,31}; text
// is capped at 500 chars server-side.
//
// Pure TS with an injectable `fetch`, so it runs under `node --test`.

import { httpUrl } from "../runtime/urls.ts";
import { PORTS } from "./endpoints.ts";

export interface DriveIntent {
  text: string;
  from: string;
  to: string;
  /// One of TASKS, or "" (free text only).
  task: string;
  target: string;
  /// "" when the intent carries no run (cleared, or an older server).
  runId: string;
  runStartedNs: number;
  stampNs: number;
  version: number;
  /// Recording stops of the open run (0 on an older server).
  stops: number;
  lastStop: RunStop | null;
  /// Times the operator chose to continue the open run.
  resumes: number;
}

/// Why a recording segment of a run ended (session_notes.STOP_CAUSES).
export interface RunStop {
  tNs: number;
  /// One of STOP_CAUSES, or an unknown cause from a newer server.
  cause: string;
  /// The firmware's E-STOP reason, or the disarmed wheel names.
  detail: string;
}

/// One entry of GET /notes `recent` (newest first, the open run first).
export interface RecentRun {
  runId: string;
  text: string;
  from: string;
  to: string;
  task: string;
  target: string;
  startedNs: number;
  /// 0 while the run is open.
  endedNs: number;
  /// One of OUTCOMES' terminal kinds; "" while open. An older server closed
  /// a cleared or replaced run with "": read it with `runOutcome`.
  outcome: string;
  interruptions: number;
  stops: number;
  lastStop: RunStop | null;
  resumes: number;
}

export interface NotesSnapshot {
  /// A recorder segment is open: events are accepted.
  recording: boolean;
  /// Current sticky intent; null before any was set, or when cleared.
  intent: DriveIntent | null;
  /// Recent runs, newest first (empty on an older server).
  recent: RecentRun[];
  eventsPending: number;
  placesPending: number;
  eventsDropped: number;
  placesDropped: number;
}

export interface DriveEvent {
  kind: string;
  note: string;
  stampNs: number;
}

/// Result of POST /intent {"resume": true}.
export interface ResumeResult {
  /// The still-open run (recent entry) with the resume counted.
  run: RecentRun | null;
  intent: DriveIntent | null;
}

/// Result of POST /intent {"end": ...}.
export interface EndRunResult {
  /// The ended run (recent entry) or, for interrupted, the still-open run.
  run: RecentRun | null;
  /// A marker event went into the current segment (false when not recording).
  eventRecorded: boolean;
  /// The intent after the call (null once a terminal outcome cleared it).
  intent: DriveIntent | null;
}

export type NotesErrorKind =
  /// 409 on /event: no segment recording; the marker was not saved.
  | "not-recording"
  /// 409 on /intent: mode "continue" without an open run.
  | "no-run"
  /// 400: the server rejected the body (kind / task / outcome / JSON).
  | "invalid"
  /// 503: the vision process has no session notes.
  | "unavailable"
  /// Network failure, timeout, CORS, or another HTTP status.
  | "network";

export type NotesResult<T> =
  | { ok: true; value: T }
  | { ok: false; kind: NotesErrorKind; status: number; error: string };

/// Event kinds the operator UI offers (the server accepts any kind matching
/// KIND_RE; these are the ones the recorder docs name).
export const DRIVE_EVENTS = [
  { kind: "obstacle", label: "Obstacle", hotkey: "1", hint: "Something in the way" },
  { kind: "recovery", label: "Recovery", hotkey: "2", hint: "Backing out / correcting" },
  { kind: "exclude", label: "Exclude", hotkey: "3", hint: "Drop this stretch from training" },
  { kind: "note", label: "Note", hotkey: "4", hint: "Generic marker" },
] as const;

export type DriveEventKind = (typeof DRIVE_EVENTS)[number]["kind"];

/// Intent presets: the path model's intent one-hot (docs/path-model.md:
/// keep / left / right / stop / turn-around). They are the targets of the
/// `maneuver` task. Free text is allowed too.
export const INTENT_PRESETS = [
  { text: "keep", label: "Keep" },
  { text: "left", label: "Left" },
  { text: "right", label: "Right" },
  { text: "stop", label: "Stop" },
  { text: "turn-around", label: "Turn around" },
] as const;

/// What a task's target is: nothing, a place name, one of a fixed set, or
/// free text.
export type TargetKind = "none" | "place" | "choice" | "text";

export interface TaskSpec {
  task: string;
  label: string;
  target: TargetKind;
  /// Allowed targets for a "choice" task.
  options?: readonly { value: string; label: string }[];
  /// Placeholder for a "place" / "text" target.
  placeholder?: string;
}

/// Task vocabulary, mirrored from session_notes.TASKS. The server accepts ""
/// or one of these and answers 400 for anything else.
export const TASKS: readonly TaskSpec[] = [
  { task: "hold", label: "Hold", target: "none" },
  { task: "go_to_place", label: "Go to place", target: "place", placeholder: "Place name" },
  {
    task: "go_through_door",
    label: "Through door",
    target: "choice",
    options: [
      { value: "left", label: "Left" },
      { value: "right", label: "Right" },
      { value: "ahead", label: "Ahead" },
    ],
  },
  { task: "follow_step", label: "Follow step", target: "text", placeholder: "Step, e.g. past the desk" },
  { task: "look_for", label: "Look for", target: "text", placeholder: "Object, e.g. red mug" },
  { task: "explore", label: "Explore", target: "none" },
  { task: "return", label: "Return", target: "none" },
  {
    task: "maneuver",
    label: "Maneuver",
    target: "choice",
    options: INTENT_PRESETS.map((p) => ({ value: p.text, label: p.label })),
  },
];

/// Outcome of a run closed without an ending (the operator declined to
/// continue it, cleared it, or started another run while it was open).
export const UNFINISHED = "unfinished";

/// Run outcomes, mirrored from session_notes.OUTCOMES. `ends: false` marks
/// the run without ending it. The labels are the one display map for
/// outcomes (`outcomeLabel`), so a term can be renamed here alone.
export const OUTCOMES = [
  { outcome: "arrived", label: "Arrived", ends: true },
  { outcome: "found", label: "Found", ends: true },
  { outcome: "gave_up", label: "Gave up", ends: true },
  { outcome: UNFINISHED, label: "Unfinished", ends: true },
  { outcome: "interrupted", label: "Interrupted", ends: false },
] as const;

/// Why a recording segment ended, mirrored from session_notes.STOP_CAUSES,
/// with the label the app shows ("Stopped: E-STOP (auto): ...").
export const STOP_CAUSES = [
  { cause: "disarmed", label: "disarmed" },
  { cause: "wheel_disarmed", label: "wheel disarmed" },
  { cause: "idle", label: "idle mode" },
  { cause: "estop_operator", label: "E-STOP (operator)" },
  { cause: "estop_auto", label: "E-STOP (auto)" },
  { cause: "link_lost", label: "link lost" },
  { cause: "rolled", label: "segment rolled over" },
  { cause: "recorder_stop", label: "recorder stopped" },
] as const;

export type RunOutcome = (typeof OUTCOMES)[number]["outcome"];

/// new: fresh run; repeat: fresh run, same meaning (said explicitly);
/// continue: keep the current run id, update only the given fields.
export const INTENT_MODES = ["new", "repeat", "continue"] as const;
export type IntentMode = (typeof INTENT_MODES)[number];

/// Fields of a POST /intent. Omitted fields are sent as "" for new / repeat
/// and left out (kept server-side) for continue.
export interface IntentFields {
  text?: string;
  from?: string;
  to?: string;
  task?: string;
  target?: string;
}

export const KIND_RE = /^[a-z][a-z0-9_-]{0,31}$/;
export const MAX_TEXT = 500;
export const DEFAULT_TIMEOUT_MS = 4000;

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface DriveNotesClient {
  readonly base: string;
  getNotes(): Promise<NotesResult<NotesSnapshot>>;
  setIntent(fields: IntentFields, mode?: IntentMode): Promise<NotesResult<DriveIntent | null>>;
  clearIntent(): Promise<NotesResult<DriveIntent | null>>;
  repeatRun(runId: string): Promise<NotesResult<DriveIntent | null>>;
  endRun(outcome: string, note?: string): Promise<NotesResult<EndRunResult>>;
  resumeRun(note?: string): Promise<NotesResult<ResumeResult>>;
  addEvent(kind: string, note?: string): Promise<NotesResult<DriveEvent>>;
}

export interface DriveNotesOptions {
  fetch?: FetchLike;
  timeoutMs?: number;
}

/// Base URL of the videoserver for `host` (scheme follows the page).
export function notesBase(host: string): string {
  return httpUrl(host, PORTS.video);
}

function num(v: unknown): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : 0;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function clip(v: string | undefined): string {
  return (v ?? "").trim().slice(0, MAX_TEXT);
}

export function taskSpec(task: string): TaskSpec | undefined {
  return TASKS.find((t) => t.task === task);
}

export function isKnownTask(task: string): boolean {
  return task === "" || taskSpec(task) !== undefined;
}

/// How a run ended: its outcome, UNFINISHED for a closed run without one
/// (what an older server wrote for a clear), "" while it is open.
export function runOutcome(r: Pick<RecentRun, "endedNs" | "outcome">): string {
  if (!r.endedNs) return "";
  return r.outcome || UNFINISHED;
}

export function stopCauseLabel(cause: string): string {
  return STOP_CAUSES.find((c) => c.cause === cause)?.label ?? cause.replace(/_/g, " ");
}

/// "just now", "12 min ago", "3 h ago", "2 d ago"; "" without a stamp.
export function agoText(ns: number, nowNs: number): string {
  if (!ns) return "";
  const min = Math.floor(Math.max(0, nowNs - ns) / 60e9);
  if (min < 1) return "just now";
  if (min < 60) return `${min} min ago`;
  const h = Math.floor(min / 60);
  if (h < 48) return `${h} h ago`;
  return `${Math.floor(h / 24)} d ago`;
}

/// "Stopped: E-STOP (auto): wheel fault, 12 min ago"; "" without a stop.
export function stopText(stop: RunStop | null, nowNs: number): string {
  if (!stop) return "";
  const what = [stopCauseLabel(stop.cause), stop.detail].filter(Boolean).join(": ");
  const ago = agoText(stop.tNs, nowNs);
  return `Stopped: ${what}${ago ? `, ${ago}` : ""}`;
}

export function outcomeLabel(outcome: string): string {
  if (!outcome) return "";
  return OUTCOMES.find((o) => o.outcome === outcome)?.label ?? outcome;
}

/// A last_stop dict; null for anything without a cause.
export function parseStop(raw: unknown): RunStop | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const cause = str(r.cause);
  if (!cause) return null;
  return { tNs: num(r.t_ns), cause, detail: str(r.detail) };
}

/// stops / resumes as /notes sends them (a count), or as the saved state
/// keeps them (a list), so either shape reads as a count.
function count(v: unknown): number {
  return Array.isArray(v) ? v.length : num(v);
}

/// The run's stop fields: last_stop, else the tail of a stops list.
function stopFields(r: Record<string, unknown>): Pick<RecentRun, "stops" | "lastStop" | "resumes"> {
  const list = Array.isArray(r.stops) ? r.stops : [];
  return {
    stops: count(r.stops),
    lastStop: parseStop(r.last_stop) ?? parseStop(list[list.length - 1]),
    resumes: count(r.resumes),
  };
}

export function parseIntent(raw: unknown): DriveIntent | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  return {
    text: str(r.text),
    from: str(r.from),
    to: str(r.to),
    task: str(r.task),
    target: str(r.target),
    runId: str(r.run_id),
    runStartedNs: num(r.run_started_ns),
    stampNs: num(r.stamp_ns),
    version: num(r.version),
    ...stopFields(r),
  };
}

/// A recent-runs entry. Also accepts an intent-shaped dict (run_started_ns
/// instead of started_ns), which is what /intent {"end": "interrupted"} may
/// return for the still-open run.
export function parseRun(raw: unknown): RecentRun | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const runId = str(r.run_id);
  if (!runId) return null;
  return {
    runId,
    text: str(r.text),
    from: str(r.from),
    to: str(r.to),
    task: str(r.task),
    target: str(r.target),
    startedNs: num(r.started_ns) || num(r.run_started_ns),
    endedNs: num(r.ended_ns),
    outcome: str(r.outcome),
    interruptions: num(r.interruptions),
    ...stopFields(r),
  };
}

export function parseRecent(raw: unknown): RecentRun[] {
  if (!Array.isArray(raw)) return [];
  return raw.map(parseRun).filter((r): r is RecentRun => r !== null);
}

/// True when the intent carries nothing (never set, or cleared).
export function intentIsEmpty(i: DriveIntent | null): boolean {
  return !i || (!i.text && !i.from && !i.to && !i.task && !i.target);
}

/// Short tag for a run id: its 4-hex suffix ("r20261025_104233_9f3a" ->
/// "#9f3a"); "" without a run.
export function runTag(runId: string): string {
  if (!runId) return "";
  const m = /_([0-9a-f]{4})$/.exec(runId);
  return `#${m ? m[1] : runId.slice(-4)}`;
}

/// "<task> <target>" of an intent or run ("Go to place kitchen",
/// "Through door Left"); "" for a free-text-only intent.
export function taskLabel(i: Pick<DriveIntent, "task" | "target">): string {
  const spec = taskSpec(i.task);
  const target = spec?.options?.find((o) => o.value === i.target)?.label ?? i.target;
  return spec
    ? [spec.label, target].filter(Boolean).join(" ")
    : [i.task, i.target].filter(Boolean).join(" ");
}

/// One-line operator label of an intent or run: "Go to place kitchen",
/// "Maneuver left", or the free text; "-> to" is appended when the target
/// does not already name it.
export function intentLabel(i: Pick<DriveIntent, "text" | "to" | "task" | "target">): string {
  const main = taskLabel(i) || i.text;
  const to = i.to && i.to !== i.target ? `→ ${i.to}` : "";
  return [main, to].filter(Boolean).join(" ") || "Intent";
}

function errorKind(status: number, conflict: NotesErrorKind): NotesErrorKind {
  if (status === 409) return conflict;
  if (status === 400) return "invalid";
  if (status === 503) return "unavailable";
  return "network";
}

function localInvalid(error: string): { ok: false; kind: "invalid"; status: 0; error: string } {
  return { ok: false, kind: "invalid", status: 0, error };
}

export function createDriveNotes(
  base: string,
  { fetch: fetchImpl, timeoutMs = DEFAULT_TIMEOUT_MS }: DriveNotesOptions = {},
): DriveNotesClient {
  const doFetch: FetchLike = fetchImpl ?? ((input, init) => globalThis.fetch(input, init));
  const root = base.replace(/\/+$/, "");

  async function call<T>(
    method: "GET" | "POST",
    path: string,
    body: unknown,
    parse: (json: Record<string, unknown>) => T,
  ): Promise<NotesResult<T>> {
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
    try {
      const init: RequestInit = { method, signal: ctrl?.signal };
      if (body !== undefined) {
        // A JSON body with Content-Length; the server refuses chunked or
        // empty bodies (an empty /intent would otherwise clear it).
        init.headers = { "Content-Type": "application/json" };
        init.body = JSON.stringify(body);
      }
      const res = await doFetch(`${root}${path}`, init);
      let json: Record<string, unknown> = {};
      try {
        const parsed: unknown = await res.json();
        if (parsed && typeof parsed === "object") json = parsed as Record<string, unknown>;
      } catch {
        /* non-JSON body (e.g. send_error HTML) */
      }
      if (!res.ok) {
        return {
          ok: false,
          // 409 means "no run to continue" on /intent, "not recording" on /event.
          kind: errorKind(res.status, path === "/intent" ? "no-run" : "not-recording"),
          status: res.status,
          error: str(json.error) || `HTTP ${res.status}`,
        };
      }
      return { ok: true, value: parse(json) };
    } catch (e) {
      const aborted = e instanceof Error && e.name === "AbortError";
      return {
        ok: false,
        kind: "network",
        status: 0,
        error: aborted ? "timed out" : e instanceof Error ? e.message : String(e),
      };
    } finally {
      if (timer !== null) clearTimeout(timer);
    }
  }

  const intentResult = (json: Record<string, unknown>): DriveIntent | null => parseIntent(json.intent);

  return {
    base: root,
    getNotes: () => call("GET", "/notes", undefined, parseNotes),
    setIntent: async (fields, mode = "new") => {
      if (!INTENT_MODES.includes(mode)) return localInvalid(`unknown mode ${mode}`);
      const task = clip(fields.task);
      if (!isKnownTask(task)) return localInvalid(`unknown task ${task}`);
      const body: Record<string, string> = { mode };
      for (const key of ["text", "from", "to", "task", "target"] as const) {
        // continue: only the given fields change; new / repeat: all are sent.
        if (mode === "continue" && fields[key] === undefined) continue;
        body[key] = clip(fields[key]);
      }
      return call("POST", "/intent", body, intentResult);
    },
    clearIntent: () => call("POST", "/intent", { clear: true }, intentResult),
    repeatRun: async (runId) => {
      if (!runId) return localInvalid("no run to repeat");
      return call("POST", "/intent", { repeat: runId }, intentResult);
    },
    endRun: async (outcome, note = "") => {
      if (!OUTCOMES.some((o) => o.outcome === outcome)) return localInvalid(`unknown outcome ${outcome}`);
      return call("POST", "/intent", { end: outcome, note: clip(note) }, (json) => ({
        run: parseRun(json.run),
        eventRecorded: json.event_recorded === true,
        intent: parseIntent(json.intent),
      }));
    },
    resumeRun: (note = "") =>
      call("POST", "/intent", { resume: true, note: clip(note) }, (json) => ({
        run: parseRun(json.run),
        intent: parseIntent(json.intent),
      })),
    addEvent: async (kind, note = "") => {
      const k = kind.trim().toLowerCase();
      if (!KIND_RE.test(k)) return localInvalid("event kind must match [a-z][a-z0-9_-]{0,31}");
      return call("POST", "/event", { kind: k, note: clip(note) }, (json) => {
        const ev = (json.event ?? {}) as Record<string, unknown>;
        return { kind: str(ev.kind) || k, note: str(ev.note), stampNs: num(ev.stamp_ns) };
      });
    },
  };
}

export function parseNotes(raw: unknown): NotesSnapshot {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  return {
    recording: r.recording === true,
    intent: parseIntent(r.intent),
    recent: parseRecent(r.recent),
    eventsPending: num(r.events_pending),
    placesPending: num(r.places_pending),
    eventsDropped: num(r.events_dropped),
    placesDropped: num(r.places_dropped),
  };
}

/// The open run of a /notes snapshot: the recent entry of the intent's run
/// while it has not ended, else (an older server without `recent`) a row
/// built from the intent itself; null when no run is open.
export function openRun(n: NotesSnapshot | null): RecentRun | null {
  const i = n?.intent ?? null;
  if (!i || intentIsEmpty(i) || !i.runId) return null;
  const row = n!.recent.find((r) => r.runId === i.runId);
  if (row) {
    if (row.endedNs) return null;
    return row.lastStop || row.stops || row.resumes ? row : { ...row, ...pickStops(i) };
  }
  return {
    runId: i.runId,
    text: i.text,
    from: i.from,
    to: i.to,
    task: i.task,
    target: i.target,
    startedNs: i.runStartedNs,
    endedNs: 0,
    outcome: "",
    interruptions: 0,
    ...pickStops(i),
  };
}

function pickStops(i: DriveIntent): Pick<RecentRun, "stops" | "lastStop" | "resumes"> {
  return { stops: i.stops, lastStop: i.lastStop, resumes: i.resumes };
}

/// Text of the "continue or close?" prompt shown before arming while a run
/// is open: title "Run still open: <task> <target>", detail lines (free
/// text, from / to, started) and the last stop ("" when none).
export interface ResumePrompt {
  title: string;
  lines: string[];
  stop: string;
}

export function resumePrompt(
  run: RecentRun,
  nowNs: number,
  clock: (ns: number) => string = () => "",
): ResumePrompt {
  const head = taskLabel(run) || run.text || runTag(run.runId);
  const lines: string[] = [];
  if (run.text && run.text !== head) lines.push(run.text);
  if (run.from || run.to) {
    lines.push(run.from ? `From ${run.from}${run.to ? ` to ${run.to}` : ""}` : `To ${run.to}`);
  }
  if (run.startedNs) {
    const at = clock(run.startedNs);
    lines.push(`Started ${[at, agoText(run.startedNs, nowNs)].filter(Boolean).join(", ")}`);
  }
  return { title: `Run still open: ${head}`, lines, stop: stopText(run.lastStop, nowNs) };
}

/// Short operator-facing message for a failed note call.
export function describeNotesError(r: { kind: NotesErrorKind; error: string }): string {
  switch (r.kind) {
    case "not-recording":
      return "Not recording: arm the wheels in drive mode first. Marker not saved.";
    case "no-run":
      return "No run to continue: start a new one.";
    case "unavailable":
      return "Drive notes unavailable (vision runs without session notes).";
    case "invalid":
      return `Rejected: ${r.error}`;
    default:
      return `Vision unreachable: ${r.error}`;
  }
}

/// Whether an arm request starts a drive: it arms `wheel` (null = every
/// wheel) and leaves every wheel armed when they were not all armed before.
/// The recorder opens a segment only once every wheel is armed, so this is
/// the moment the open-run question must be asked, whether the operator
/// used Start driving, Arm all, the pad or one wheel toggle after another.
export function armStartsDrive(
  wheels: readonly { name: string; armed: boolean }[],
  wheel: string | null,
): boolean {
  if (wheels.length === 0 || wheels.every((w) => w.armed)) return false;
  return wheels.every((w) => w.armed || wheel === null || w.name === wheel);
}
