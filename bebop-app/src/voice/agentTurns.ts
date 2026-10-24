// Pure state for the agent conversation: typed events from the voice/agent
// server (:9093 /voice, protocol in docs/voice.md) folded into a list of
// turns. No React, no sockets, so it runs under `node --test`.
//
// A turn is created locally the moment the operator sends something (the
// client picks the `turn_id`), so the bubble shows up before the server
// answers. Server events then advance it:
//
//   turn_start → transcript → (assistant_delta | tool_call → motion_progress*
//   → tool_result)* → turn_end
//
// `turn_rejected` carries no turn id (only the running one), so it is
// matched to the oldest turn that was sent but never started.

export type TurnSource = "text" | "voice";

export type TurnStatus =
  /** Voice turn: the mic is still open. */
  | "recording"
  /** Sent, waiting for `turn_start`. */
  | "pending"
  | "running"
  | "done"
  | "cancelled"
  | "error"
  /** The server refused it because another turn was still running. */
  | "rejected"
  /** The socket dropped while it ran (the server cancels it). */
  | "interrupted";

export interface MotionProgress {
  doneM: number;
  totalM: number;
  doneDeg: number;
  totalDeg: number;
}

export interface TextItem {
  kind: "text";
  text: string;
}

export interface ToolItem {
  kind: "tool";
  callId: string;
  name: string;
  args: Record<string, unknown>;
  /** Undefined while the tool runs. */
  ok?: boolean;
  result?: string;
  ms?: number;
  progress?: MotionProgress;
  startedAt: number;
}

export type TurnItem = TextItem | ToolItem;

export interface Turn {
  id: string;
  source: TurnSource;
  /** What the operator typed, or the ASR transcript of what they said. */
  userText: string;
  speak: boolean;
  status: TurnStatus;
  error?: string;
  items: TurnItem[];
  createdAt: number;
  endedAt?: number;
  /** Set once the operator asked to cancel (until `turn_end` lands). */
  cancelRequested?: boolean;
  /** `audio_start` seen and no `audio_end` yet. */
  speaking?: boolean;
}

export interface AgentState {
  turns: Turn[];
  /** Last error that did not belong to a turn (bad frame, service not ready). */
  sessionError: string | null;
  /**
   * Turns still in flight when the operator started a new conversation. The
   * server still finishes them (cancelled), but their events must not bring
   * them back into the fresh transcript.
   */
  dropped: string[];
  /** A `reset` is on its way: ignore bare `error`/`done` until its echo. */
  resetting: boolean;
}

export const INITIAL_AGENT_STATE: AgentState = {
  turns: [],
  sessionError: null,
  dropped: [],
  resetting: false,
};

const MAX_DROPPED = 32;

/** Turn events as the server sends them (loosely typed: JSON off the wire). */
export interface AgentEvent {
  type?: string;
  turn_id?: string;
  [key: string]: unknown;
}

export type AgentAction =
  | { type: "send"; id: string; source: TurnSource; text?: string; speak: boolean; at: number }
  /** Voice turn: capture ended, the utterance is on its way. */
  | { type: "recorded"; id: string }
  /** Voice turn failed before the server saw it (mic error, socket down). */
  | { type: "abandon"; id: string; error?: string; at: number }
  | { type: "event"; event: AgentEvent; at: number }
  | { type: "cancel_requested"; at: number }
  | { type: "disconnected"; at: number }
  /** New conversation. `awaitReset`: a `reset` frame went to the server. */
  | { type: "clear"; awaitReset?: boolean }
  | { type: "load"; turns: Turn[] };

const ACTIVE: ReadonlySet<TurnStatus> = new Set(["recording", "pending", "running"]);

export function isActive(t: Turn): boolean {
  return ACTIVE.has(t.status);
}

/** The turn the server is running for us, if any. */
export function runningTurn(state: AgentState): Turn | null {
  for (let i = state.turns.length - 1; i >= 0; i -= 1) {
    if (state.turns[i].status === "running") return state.turns[i];
  }
  return null;
}

/** Anything in flight (recording, sent, or running). */
export function activeTurn(state: AgentState): Turn | null {
  for (let i = state.turns.length - 1; i >= 0; i -= 1) {
    if (isActive(state.turns[i])) return state.turns[i];
  }
  return null;
}

/** The motion currently in progress (a tool with progress and no result). */
export function activeMotion(state: AgentState): ToolItem | null {
  const t = runningTurn(state);
  if (!t) return null;
  for (let i = t.items.length - 1; i >= 0; i -= 1) {
    const it = t.items[i];
    if (it.kind === "tool" && it.ok === undefined && it.progress) return it;
  }
  return null;
}

/** Concatenated assistant text of a turn. */
export function assistantText(t: Turn): string {
  return t.items
    .filter((i): i is TextItem => i.kind === "text")
    .map((i) => i.text)
    .join("");
}

function num(v: unknown): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : 0;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

function patchTurn(state: AgentState, id: string, fn: (t: Turn) => Turn): AgentState {
  const i = state.turns.findIndex((t) => t.id === id);
  if (i < 0) return state;
  const next = fn(state.turns[i]);
  if (next === state.turns[i]) return state;
  const turns = state.turns.slice();
  turns[i] = next;
  return { ...state, turns };
}

function patchTool(t: Turn, callId: string, fn: (it: ToolItem) => ToolItem): Turn {
  const i = t.items.findIndex((it) => it.kind === "tool" && it.callId === callId);
  if (i < 0) return t;
  const items = t.items.slice();
  items[i] = fn(items[i] as ToolItem);
  return { ...t, items };
}

/** Close every tool still running in a turn (cancel / error / disconnect). */
function closeTools(t: Turn, text: string, at: number): Turn {
  if (!t.items.some((it) => it.kind === "tool" && it.ok === undefined)) return t;
  return {
    ...t,
    items: t.items.map((it) =>
      it.kind === "tool" && it.ok === undefined
        ? { ...it, ok: false, result: it.result ?? text, ms: it.ms ?? at - it.startedAt }
        : it,
    ),
  };
}

function oldestUnstarted(state: AgentState): Turn | null {
  return state.turns.find((t) => t.status === "pending") ?? null;
}

/** Make sure a server-side turn exists locally (one the client didn't name). */
function ensureTurn(state: AgentState, ev: AgentEvent, at: number): AgentState {
  const id = str(ev.turn_id);
  if (!id || state.turns.some((t) => t.id === id)) return state;
  const source: TurnSource = ev.source === "voice" ? "voice" : "text";
  return {
    ...state,
    turns: [
      ...state.turns,
      {
        id,
        source,
        userText: "",
        speak: ev.speak !== false,
        status: "running",
        items: [],
        createdAt: at,
      },
    ],
  };
}

function reduceEvent(state: AgentState, ev: AgentEvent, at: number): AgentState {
  const id = str(ev.turn_id);
  if (id && state.dropped.includes(id)) return state;
  if (state.resetting && (ev.type === "error" || ev.type === "done")) return state;
  switch (ev.type) {
    case "turn_start": {
      if (!id) return state;
      const s = ensureTurn(state, ev, at);
      return patchTurn(s, id, (t) => ({
        ...t,
        status: "running",
        source: ev.source === "voice" ? "voice" : ev.source === "text" ? "text" : t.source,
        speak: typeof ev.speak === "boolean" ? ev.speak : t.speak,
      }));
    }
    case "transcript": {
      if (!id) return state;
      const s = ensureTurn(state, ev, at);
      return patchTurn(s, id, (t) => ({ ...t, userText: str(ev.text) }));
    }
    case "assistant_delta": {
      const text = str(ev.text);
      if (!id || !text) return state;
      const s = ensureTurn(state, ev, at);
      return patchTurn(s, id, (t) => {
        const last = t.items[t.items.length - 1];
        const items = t.items.slice();
        if (last && last.kind === "text") {
          items[items.length - 1] = { kind: "text", text: last.text + text };
        } else {
          items.push({ kind: "text", text });
        }
        return { ...t, items };
      });
    }
    case "tool_call": {
      if (!id) return state;
      const s = ensureTurn(state, ev, at);
      const callId = str(ev.call_id) || `${id}-tool-${at}`;
      const args =
        ev.args && typeof ev.args === "object" && !Array.isArray(ev.args)
          ? (ev.args as Record<string, unknown>)
          : {};
      return patchTurn(s, id, (t) =>
        t.items.some((it) => it.kind === "tool" && it.callId === callId)
          ? t
          : {
              ...t,
              items: [
                ...t.items,
                { kind: "tool", callId, name: str(ev.name) || "tool", args, startedAt: at },
              ],
            },
      );
    }
    case "motion_progress": {
      if (!id) return state;
      const callId = str(ev.call_id);
      return patchTurn(state, id, (t) => {
        // Pair with the named call, or the newest tool still running.
        let target = callId;
        if (!t.items.some((it) => it.kind === "tool" && it.callId === target)) {
          const open = [...t.items]
            .reverse()
            .find((it): it is ToolItem => it.kind === "tool" && it.ok === undefined);
          if (!open) return t;
          target = open.callId;
        }
        return patchTool(t, target, (it) => ({
          ...it,
          progress: {
            doneM: num(ev.done_m),
            totalM: num(ev.total_m),
            doneDeg: num(ev.done_deg),
            totalDeg: num(ev.total_deg),
          },
        }));
      });
    }
    case "tool_result": {
      if (!id) return state;
      const callId = str(ev.call_id);
      return patchTurn(state, id, (t) => {
        if (!t.items.some((it) => it.kind === "tool" && it.callId === callId)) {
          // A result without its call (shouldn't happen): show it anyway.
          return {
            ...t,
            items: [
              ...t.items,
              {
                kind: "tool",
                callId,
                name: "tool",
                args: {},
                ok: ev.ok !== false,
                result: str(ev.text),
                ms: num(ev.ms),
                startedAt: at,
              },
            ],
          };
        }
        return patchTool(t, callId, (it) => ({
          ...it,
          ok: ev.ok !== false,
          result: str(ev.text),
          ms: ev.ms === undefined ? at - it.startedAt : num(ev.ms),
        }));
      });
    }
    case "audio_start":
      return id ? patchTurn(state, id, (t) => ({ ...t, speaking: true })) : state;
    case "audio_end":
      return id ? patchTurn(state, id, (t) => (t.speaking ? { ...t, speaking: false } : t)) : state;
    case "turn_end": {
      if (!id) return state;
      const s = ensureTurn(state, ev, at);
      const status: TurnStatus =
        ev.status === "cancelled" ? "cancelled" : ev.status === "error" ? "error" : "done";
      return patchTurn(s, id, (t) => {
        const closed = status === "done" ? t : closeTools(t, status, at);
        return {
          ...closed,
          status,
          error: status === "error" ? str(ev.error) || "The turn failed." : undefined,
          endedAt: at,
          speaking: false,
          cancelRequested: false,
        };
      });
    }
    case "turn_rejected": {
      const victim = oldestUnstarted(state);
      if (!victim) return state;
      return patchTurn(state, victim.id, (t) => ({
        ...t,
        status: "rejected",
        error: "The robot was still working on the previous request.",
        endedAt: at,
      }));
    }
    case "done": {
      // A bare `done` answers an empty voice utterance, which never becomes
      // a turn. Older services also sent one right before every turn_end;
      // that copy is ignored while a turn runs.
      if (runningTurn(state)) return state;
      const empty = state.turns.find(
        (t) => t.status === "pending" && t.source === "voice" && !t.userText,
      );
      if (!empty) return state;
      return { ...state, turns: state.turns.filter((t) => t !== empty) };
    }
    case "error": {
      // Typed turns end with turn_end{error}; a bare `error` answers a frame
      // that never became a turn. Older services also sent one right before
      // turn_end{error}; that copy is ignored while a turn runs.
      if (runningTurn(state)) return state;
      const message = str(ev.message) || "agent error";
      const victim = oldestUnstarted(state);
      if (victim) {
        return {
          ...patchTurn(state, victim.id, (t) => ({
            ...t,
            status: "error",
            error: message,
            endedAt: at,
          })),
          sessionError: message,
        };
      }
      return { ...state, sessionError: message };
    }
    case "reset":
      return state.resetting ? { ...state, resetting: false } : state;
    default:
      return state;
  }
}

export function agentReducer(state: AgentState, action: AgentAction): AgentState {
  switch (action.type) {
    case "send":
      return {
        ...state,
        sessionError: null,
        turns: [
          ...state.turns,
          {
            id: action.id,
            source: action.source,
            userText: action.text ?? "",
            speak: action.speak,
            status: action.source === "voice" ? "recording" : "pending",
            items: [],
            createdAt: action.at,
          },
        ],
      };
    case "recorded":
      return patchTurn(state, action.id, (t) =>
        t.status === "recording" ? { ...t, status: "pending" } : t,
      );
    case "abandon": {
      const t = state.turns.find((x) => x.id === action.id);
      if (!t || !isActive(t)) return state;
      if (!action.error) return { ...state, turns: state.turns.filter((x) => x !== t) };
      return patchTurn(state, action.id, (x) => ({
        ...x,
        status: "error",
        error: action.error,
        endedAt: action.at,
      }));
    }
    case "event":
      return reduceEvent(state, action.event, action.at);
    case "cancel_requested": {
      // A turn the server never started (no `turn_start` yet) resolves here:
      // the server's cancel may be a no-op for it (e.g. an `utterance_end`
      // that reached a new socket), and nothing else would ever end it. If
      // the server did start it after all, its `turn_start` revives it and
      // the `turn_end{cancelled}` that follows closes it again.
      let s = state;
      for (const t of state.turns) {
        if (t.status === "pending" || t.status === "recording") {
          s = patchTurn(s, t.id, (x) => ({
            ...x,
            status: "cancelled",
            endedAt: action.at,
            cancelRequested: false,
          }));
        }
      }
      const t = activeTurn(s);
      return t ? patchTurn(s, t.id, (x) => ({ ...x, cancelRequested: true })) : s;
    }
    case "disconnected": {
      if (!state.turns.some(isActive)) return state;
      return {
        ...state,
        turns: state.turns.map((t) =>
          t.status === "recording"
            ? {
                ...t,
                status: "interrupted",
                error: "Connection to the robot dropped while listening.",
                endedAt: action.at,
              }
            : t.status === "pending" || t.status === "running"
            ? {
                ...closeTools(t, "connection lost", action.at),
                status: "interrupted",
                error: "Connection to the robot dropped; the robot stopped this request.",
                endedAt: action.at,
                speaking: false,
              }
            : t,
        ),
      };
    }
    case "clear": {
      const inFlight = state.turns.filter(isActive).map((t) => t.id);
      return {
        ...INITIAL_AGENT_STATE,
        dropped: [...state.dropped, ...inFlight].slice(-MAX_DROPPED),
        resetting: !!action.awaitReset,
      };
    }
    case "load":
      // Restored from storage: anything that was in flight is gone now.
      return {
        ...state,
        sessionError: null,
        turns: action.turns.map((t) =>
          isActive(t)
            ? { ...closeTools(t, "interrupted", t.createdAt), status: "interrupted", speaking: false }
            : t,
        ),
      };
    default:
      return state;
  }
}

/** One-line summary of tool args for a card header: `forward_m 0.5 · turn_deg 90`. */
export function summarizeArgs(args: Record<string, unknown>, max = 80): string {
  const parts = Object.entries(args).map(([k, v]) => {
    const val =
      typeof v === "string"
        ? `"${v}"`
        : typeof v === "number" || typeof v === "boolean"
          ? String(v)
          : JSON.stringify(v);
    return `${k} ${val}`;
  });
  const s = parts.join(" · ");
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

/** Fraction done of a motion (the larger of distance and angle progress). */
export function motionFraction(p: MotionProgress): number {
  const fm = p.totalM ? Math.abs(p.doneM) / Math.abs(p.totalM) : 0;
  const fd = p.totalDeg ? Math.abs(p.doneDeg) / Math.abs(p.totalDeg) : 0;
  const f = p.totalM && p.totalDeg ? (fm + fd) / 2 : Math.max(fm, fd);
  return Math.max(0, Math.min(1, f));
}
