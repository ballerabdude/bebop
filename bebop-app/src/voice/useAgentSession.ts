// One persistent WebSocket to the agent server (`ws://<robot>:9093/voice`)
// and the conversation built from its typed events (docs/voice.md › Agent
// protocol). Reconnects with backoff after an unintended close; the
// `session` query keeps the robot's conversation history across reconnects
// and reloads. Mic capture and playback are `useVoiceIO`'s job; binary
// frames are handed to `onAudio`.

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { endpoints } from "../robot/endpoints";
import { reconnectDelayMs } from "../runtime/backoff";
import {
  activeMotion,
  activeTurn,
  agentReducer,
  INITIAL_AGENT_STATE,
  runningTurn,
} from "./agentTurns";
import type { AgentEvent, AgentState, ToolItem, Turn } from "./agentTurns";

export type AgentLink = "off" | "connecting" | "open" | "reconnecting";

export interface AgentSessionOptions {
  /// Connect only while true (e.g. the service reports ready).
  enabled: boolean;
  /// TTS speaker override; empty uses the service's configured voice.
  voice?: string;
  /// Binary PCM16 frames (24 kHz mono) of the reply speech.
  onAudio?: (bytes: ArrayBuffer) => void;
  /// Called when a turn is cancelled locally (stop queued speech).
  onCancel?: () => void;
}

export interface AgentSession {
  state: AgentState;
  link: AgentLink;
  /// The turn that is recording, waiting or running, if any.
  active: Turn | null;
  /// The turn the server is running.
  running: Turn | null;
  /// A `move_relative` in progress.
  motion: ToolItem | null;
  sendText: (text: string, opts: { speak: boolean }) => boolean;
  /// Voice turn: announce the utterance; returns its id or null when offline.
  beginUtterance: (sampleRate: number, opts: { speak: boolean }) => string | null;
  sendAudio: (pcm: Int16Array) => void;
  /// Mic released: the server starts the turn.
  endUtterance: (id: string) => void;
  /// Capture failed before anything useful was sent.
  abandonUtterance: (id: string, error?: string) => void;
  cancel: () => void;
  newConversation: () => void;
}

const SESSION_PREFIX = "bebop.agent.session.";
const TURNS_PREFIX = "bebop.agent.turns.";
const MAX_STORED_TURNS = 40;

function makeId(prefix: string): string {
  let rand: string;
  try {
    rand = crypto.randomUUID().replace(/-/g, "").slice(0, 12);
  } catch {
    rand = Math.random().toString(36).slice(2, 14);
  }
  return `${prefix}${Date.now().toString(36)}-${rand}`;
}

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* private mode / quota: the conversation just isn't remembered */
  }
}

function loadTurns(host: string, sessionId: string): Turn[] {
  const raw = readStorage(TURNS_PREFIX + host);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as { session?: string; turns?: Turn[] };
    return parsed.session === sessionId && Array.isArray(parsed.turns) ? parsed.turns : [];
  } catch {
    return [];
  }
}

export function useAgentSession(host: string, opts: AgentSessionOptions): AgentSession {
  const { enabled, voice } = opts;
  const [sessionId, setSessionId] = useState(
    () => readStorage(SESSION_PREFIX + host) || makeId("s-"),
  );
  const [state, dispatch] = useReducer(agentReducer, INITIAL_AGENT_STATE, (init) =>
    agentReducer(init, { type: "load", turns: loadTurns(host, sessionId) }),
  );
  const [link, setLink] = useState<AgentLink>("off");
  const wsRef = useRef<WebSocket | null>(null);
  const handlers = useRef(opts);
  handlers.current = opts;

  useEffect(() => writeStorage(SESSION_PREFIX + host, sessionId), [host, sessionId]);

  // Remember the conversation (finished turns are enough to rebuild the
  // transcript after a reload; the robot keeps its own history by session).
  useEffect(() => {
    const t = window.setTimeout(() => {
      writeStorage(
        TURNS_PREFIX + host,
        JSON.stringify({ session: sessionId, turns: state.turns.slice(-MAX_STORED_TURNS) }),
      );
    }, 400);
    return () => window.clearTimeout(t);
  }, [host, sessionId, state.turns]);

  // Never drop the socket under a running turn: the server cancels every
  // turn a closing socket owns, so a health-poll blip or a telemetry
  // reconnect upstream would otherwise kill a task (or a move) mid-way. A
  // real service stop closes the socket from the other end anyway.
  const busy = activeTurn(state) !== null;
  const live = enabled || busy;

  useEffect(() => {
    if (!live || !host) {
      setLink("off");
      return;
    }
    let disposed = false;
    let attempt = 0;
    let timer: number | null = null;
    let ws: WebSocket | null = null;

    const connect = () => {
      if (disposed) return;
      setLink(attempt === 0 ? "connecting" : "reconnecting");
      const qs = new URLSearchParams({ session: sessionId });
      if (voice) qs.set("voice", voice);
      const sock = new WebSocket(endpoints(host).voiceWs(qs.toString()));
      sock.binaryType = "arraybuffer";
      ws = sock;
      sock.onopen = () => {
        if (disposed) return;
        attempt = 0;
        wsRef.current = sock;
        setLink("open");
      };
      sock.onmessage = (ev: MessageEvent) => {
        if (ev.data instanceof ArrayBuffer) {
          handlers.current.onAudio?.(ev.data);
          return;
        }
        if (typeof ev.data !== "string") return;
        let msg: AgentEvent;
        try {
          msg = JSON.parse(ev.data) as AgentEvent;
        } catch {
          return;
        }
        if (msg && typeof msg === "object") {
          dispatch({ type: "event", event: msg, at: Date.now() });
        }
      };
      sock.onclose = () => {
        if (wsRef.current === sock) wsRef.current = null;
        if (disposed) return;
        dispatch({ type: "disconnected", at: Date.now() });
        setLink("reconnecting");
        timer = window.setTimeout(connect, reconnectDelayMs(attempt));
        attempt += 1;
      };
    };
    connect();
    return () => {
      disposed = true;
      if (timer !== null) window.clearTimeout(timer);
      if (ws) {
        ws.onclose = null;
        ws.onmessage = null;
        try {
          ws.close();
        } catch {
          /* ignore */
        }
      }
      wsRef.current = null;
      dispatch({ type: "disconnected", at: Date.now() });
    };
  }, [host, live, sessionId, voice]);

  const send = useCallback((msg: unknown): boolean => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(msg));
    return true;
  }, []);

  const sendText = useCallback(
    (text: string, { speak }: { speak: boolean }) => {
      const body = text.trim();
      if (!body) return false;
      const id = makeId("c-");
      dispatch({ type: "send", id, source: "text", text: body, speak, at: Date.now() });
      if (!send({ type: "user_text", text: body, speak, turn_id: id })) {
        dispatch({
          type: "abandon",
          id,
          error: "Not connected to the robot's agent. Try again in a moment.",
          at: Date.now(),
        });
        return false;
      }
      return true;
    },
    [send],
  );

  const beginUtterance = useCallback(
    (sampleRate: number, { speak }: { speak: boolean }) => {
      const id = makeId("v-");
      if (
        !send({ type: "utterance_start", sample_rate: sampleRate, format: "pcm16", speak, turn_id: id })
      ) {
        return null;
      }
      dispatch({ type: "send", id, source: "voice", speak, at: Date.now() });
      return id;
    },
    [send],
  );

  const sendAudio = useCallback((pcm: Int16Array) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm);
  }, []);

  const endUtterance = useCallback(
    (id: string) => {
      if (send({ type: "utterance_end" })) {
        dispatch({ type: "recorded", id });
      } else {
        dispatch({ type: "abandon", id, error: "Connection dropped while listening.", at: Date.now() });
      }
    },
    [send],
  );

  const abandonUtterance = useCallback(
    (id: string, error?: string) => {
      // Close the open utterance without audio: the server answers `done`.
      send({ type: "utterance_end" });
      dispatch({ type: "abandon", id, error, at: Date.now() });
    },
    [send],
  );

  const cancel = useCallback(() => {
    dispatch({ type: "cancel_requested", at: Date.now() });
    send({ type: "cancel" });
    handlers.current.onCancel?.();
  }, [send]);

  const newConversation = useCallback(() => {
    handlers.current.onCancel?.();
    // `reset` cancels any running turn and clears the robot's history for
    // this session. Offline, start a fresh session id instead.
    const sent = send({ type: "reset" });
    dispatch({ type: "clear", awaitReset: sent });
    if (!sent) setSessionId(makeId("s-"));
  }, [send]);

  return {
    state,
    link,
    active: activeTurn(state),
    running: runningTurn(state),
    motion: activeMotion(state),
    sendText,
    beginUtterance,
    sendAudio,
    endUtterance,
    abandonUtterance,
    cancel,
    newConversation,
  };
}
