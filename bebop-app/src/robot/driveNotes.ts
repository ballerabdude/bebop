// Drive-note client for the bebop-vision videoserver (:9092).
//
// The recorder writes operator notes into the active drive MCAP
// (bebop_vision/session_notes.py, routes in videoserver.py):
//
//   POST /intent {text, from, to}  sticky operator intent, accepted any time;
//                                  {"clear": true} clears it
//   POST /event  {kind, note}      marker in the current segment; 409 when
//                                  no segment is recording (wheels not armed
//                                  in DialIn, E-STOP closed it, ...)
//   GET  /notes                    {recording, intent, events_pending, ...}
//
// 400 = bad body / kind; 503 = the vision process runs without session notes.
// Kinds must match [a-z][a-z0-9_-]{0,31}; text is capped at 500 chars
// server-side.
//
// Pure TS with an injectable `fetch`, so it runs under `node --test`.

import { httpUrl } from "../runtime/urls.ts";
import { PORTS } from "./endpoints.ts";

export interface DriveIntent {
  text: string;
  from: string;
  to: string;
  stampNs: number;
  version: number;
}

export interface NotesSnapshot {
  /// A recorder segment is open: events are accepted.
  recording: boolean;
  /// Current sticky intent; null before any was set, or when cleared.
  intent: DriveIntent | null;
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

export type NotesErrorKind =
  /// 409: no segment recording; the marker was not saved.
  | "not-recording"
  /// 400: the server rejected the body (kind / JSON).
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
/// keep / left / right / stop / turn-around). Free text is allowed too.
export const INTENT_PRESETS = [
  { text: "keep", label: "Keep" },
  { text: "left", label: "Left" },
  { text: "right", label: "Right" },
  { text: "stop", label: "Stop" },
  { text: "turn-around", label: "Turn around" },
] as const;

export const KIND_RE = /^[a-z][a-z0-9_-]{0,31}$/;
export const MAX_TEXT = 500;
export const DEFAULT_TIMEOUT_MS = 4000;

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface DriveNotesClient {
  readonly base: string;
  getNotes(): Promise<NotesResult<NotesSnapshot>>;
  setIntent(text: string, from?: string, to?: string): Promise<NotesResult<DriveIntent>>;
  clearIntent(): Promise<NotesResult<DriveIntent>>;
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

export function parseIntent(raw: unknown): DriveIntent | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const intent: DriveIntent = {
    text: str(r.text),
    from: str(r.from),
    to: str(r.to),
    stampNs: num(r.stamp_ns),
    version: num(r.version),
  };
  return intent;
}

/// True when the intent carries nothing (never set, or cleared).
export function intentIsEmpty(i: DriveIntent | null): boolean {
  return !i || (!i.text && !i.from && !i.to);
}

export function parseNotes(raw: unknown): NotesSnapshot {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  return {
    recording: r.recording === true,
    intent: parseIntent(r.intent),
    eventsPending: num(r.events_pending),
    placesPending: num(r.places_pending),
    eventsDropped: num(r.events_dropped),
    placesDropped: num(r.places_dropped),
  };
}

function errorKind(status: number): NotesErrorKind {
  if (status === 409) return "not-recording";
  if (status === 400) return "invalid";
  if (status === 503) return "unavailable";
  return "network";
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
          kind: errorKind(res.status),
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

  const intentResult = (json: Record<string, unknown>): DriveIntent =>
    parseIntent(json.intent) ?? { text: "", from: "", to: "", stampNs: 0, version: 0 };

  return {
    base: root,
    getNotes: () => call("GET", "/notes", undefined, parseNotes),
    setIntent: (text, from = "", to = "") =>
      call(
        "POST",
        "/intent",
        {
          text: text.trim().slice(0, MAX_TEXT),
          from: from.trim().slice(0, MAX_TEXT),
          to: to.trim().slice(0, MAX_TEXT),
        },
        intentResult,
      ),
    clearIntent: () => call("POST", "/intent", { clear: true }, intentResult),
    addEvent: async (kind, note = "") => {
      const k = kind.trim().toLowerCase();
      if (!KIND_RE.test(k)) {
        return {
          ok: false,
          kind: "invalid",
          status: 0,
          error: "event kind must match [a-z][a-z0-9_-]{0,31}",
        };
      }
      return call("POST", "/event", { kind: k, note: note.trim().slice(0, MAX_TEXT) }, (json) => {
        const ev = (json.event ?? {}) as Record<string, unknown>;
        return { kind: str(ev.kind) || k, note: str(ev.note), stampNs: num(ev.stamp_ns) };
      });
    },
  };
}

/// Short operator-facing message for a failed note call.
export function describeNotesError(r: { kind: NotesErrorKind; error: string }): string {
  switch (r.kind) {
    case "not-recording":
      return "Not recording: arm the wheels in drive mode first. Marker not saved.";
    case "unavailable":
      return "Drive notes unavailable (vision runs without session notes).";
    case "invalid":
      return `Rejected: ${r.error}`;
    default:
      return `Vision unreachable: ${r.error}`;
  }
}
