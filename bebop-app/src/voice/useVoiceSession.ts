import { useCallback, useEffect, useRef, useState } from "react";

import { httpUrl, wsUrl } from "../runtime/urls";

/// Turn-based speech-to-speech client for `bebop-voice.service` (:9093).
///
/// Push-to-talk: hold `startTalking()` to capture 16 kHz mono PCM16 from the
/// microphone and stream it as binary frames; release (`stopTalking()`) to end
/// the utterance. The service replies with text deltas plus 24 kHz mono PCM16
/// speech, which we schedule on a playback `AudioContext`.
///
/// The wire protocol is documented in `bebop-vision/bebop_vision/voice_server.py`
/// and `docs/voice.md`.

export type VoicePhase =
  | "idle"
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking"
  | "error";

export interface VoiceSession {
  phase: VoicePhase;
  /** Live transcript of the user's utterance, as far as we know it. */
  transcript: string;
  /** The robot's reply text (streamed). */
  reply: string;
  error: string | null;
  /** Name of the tool the robot is currently running, if any. */
  lastTool: string | null;
  startTalking: () => Promise<void>;
  stopTalking: () => void;
  /** Clear the server-side conversation context for this session. */
  resetContext: () => void;
}

export interface VoiceSessionOptions {
  /** Speaker id (`aiden`/`chelsie`/`ethan`); empty uses the service default. */
  voice?: string;
  /** Stable id so conversation context survives reconnects. */
  session?: string;
}

const CAPTURE_RATE = 16000;
const PLAYBACK_RATE = 24000;

/// Inline AudioWorklet processor: forwards each 128-frame render quantum as a
/// Float32Array to the main thread. Kept as a string so it can be loaded from a
/// Blob URL without a separate bundled asset.
const CAPTURE_WORKLET = `
class BebopPcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}
registerProcessor("bebop-pcm-capture", BebopPcmCapture);
`;

function f32ToI16(f: Float32Array): Int16Array {
  const out = new Int16Array(f.length);
  for (let i = 0; i < f.length; i += 1) {
    const s = Math.max(-1, Math.min(1, f[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function i16ToF32(bytes: ArrayBuffer): Float32Array {
  const view = new Int16Array(bytes);
  const out = new Float32Array(view.length);
  for (let i = 0; i < view.length; i += 1) out[i] = view[i] / 32768;
  return out;
}

/// `AudioContext` with a preferred rate, falling back to the device default.
/// Mobile Safari/WebViews often ignore the requested `sampleRate` (or throw),
/// so callers must read `ctx.sampleRate` back and report the real rate.
function createAudioContext(preferredRate: number): AudioContext {
  const w = window as unknown as {
    AudioContext?: typeof AudioContext;
    webkitAudioContext?: typeof AudioContext;
  };
  const Ctor = w.AudioContext ?? w.webkitAudioContext;
  if (!Ctor) throw new Error("Web Audio isn't available in this browser.");
  try {
    return new Ctor({ sampleRate: preferredRate });
  } catch {
    return new Ctor();
  }
}

/// Turn a `getUserMedia` rejection into an actionable message.
function micErrorMessage(e: unknown): string {
  const name = (e as { name?: string } | null)?.name ?? "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone permission denied. Allow microphone access for Bebop, then try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No microphone found on this device.";
    case "NotReadableError":
    case "TrackStartError":
      return "The microphone is in use by another app.";
    default:
      return e instanceof Error ? e.message : String(e);
  }
}

export function useVoiceSession(
  host: string,
  port = 9093,
  options: VoiceSessionOptions = {},
): VoiceSession {
  const { voice, session } = options;
  const [phase, setPhase] = useState<VoicePhase>("idle");
  const [transcript, setTranscript] = useState("");
  const [reply, setReply] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [lastTool, setLastTool] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureCtxRef = useRef<AudioContext | null>(null);
  const captureNodeRef = useRef<AudioWorkletNode | ScriptProcessorNode | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const playHeadRef = useRef(0);
  const capturingRef = useRef(false);
  const stopCaptureRef = useRef<() => void>(() => {});

  // Tear down any live capture/WS when the screen unmounts or host changes.
  useEffect(() => {
    return () => {
      try {
        wsRef.current?.close();
      } catch {
        /* ignore */
      }
      stopCaptureRef.current?.();
    };
  }, [host, port]);

  const ensurePlayback = useCallback(() => {
    if (!playbackCtxRef.current) {
      playbackCtxRef.current = createAudioContext(PLAYBACK_RATE);
      playHeadRef.current = 0;
    }
    // iOS suspends the context until a gesture; resume() is a no-op elsewhere.
    void playbackCtxRef.current.resume().catch(() => {});
    return playbackCtxRef.current;
  }, []);

  const playChunk = useCallback(
    (bytes: ArrayBuffer) => {
      const ctx = ensurePlayback();
      const samples = i16ToF32(bytes);
      const buffer = ctx.createBuffer(1, samples.length, PLAYBACK_RATE);
      buffer.copyToChannel(samples, 0);
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      const now = ctx.currentTime;
      const startAt = Math.max(now, playHeadRef.current);
      src.start(startAt);
      playHeadRef.current = startAt + buffer.duration;
    },
    [ensurePlayback],
  );

  const handleMessage = useCallback(
    (ev: MessageEvent) => {
      if (ev.data instanceof ArrayBuffer) {
        playChunk(ev.data);
        return;
      }
      if (typeof ev.data !== "string") return;
      let msg: {
        type?: string;
        delta?: string;
        message?: string;
        name?: string;
        text?: string;
        sample_rate?: number;
      };
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (msg.type) {
        case "transcript":
          // Cascade backend: the ASR transcript of the user's utterance.
          if (msg.text) setTranscript(msg.text);
          break;
        case "text":
          if (msg.delta) setReply((r) => r + msg.delta);
          break;
        case "audio_start":
          setPhase("speaking");
          break;
        case "audio_end":
          break;
        case "tool":
          setLastTool(msg.name ?? "tool");
          break;
        case "done":
          setPhase("idle");
          setLastTool(null);
          break;
        case "error":
          setError(msg.message ?? "voice error");
          setPhase("error");
          break;
        default:
          break;
      }
    },
    [playChunk],
  );

  const openSocket = useCallback((): Promise<WebSocket> => {
    const existing = wsRef.current;
    if (existing && existing.readyState === WebSocket.OPEN) {
      return Promise.resolve(existing);
    }
    return new Promise((resolve, reject) => {
      const qs = new URLSearchParams();
      if (voice) qs.set("voice", voice);
      if (session) qs.set("session", session);
      const suffix = qs.toString() ? `?${qs.toString()}` : "";
      const ws = new WebSocket(wsUrl(host, port, "/voice") + suffix);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => resolve(ws);
      ws.onerror = () => reject(new Error(`voice socket error (${host}:${port})`));
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
      };
      ws.onmessage = handleMessage;
      wsRef.current = ws;
    });
  }, [host, port, handleMessage, voice, session]);

  const stopCapture = useCallback(() => {
    capturingRef.current = false;
    const node = captureNodeRef.current;
    if (node) {
      if ("port" in node) (node as AudioWorkletNode).port.onmessage = null;
      if ("onaudioprocess" in node) (node as ScriptProcessorNode).onaudioprocess = null;
      try {
        node.disconnect();
      } catch {
        /* ignore */
      }
    }
    captureNodeRef.current = null;
    try {
      captureCtxRef.current?.close();
    } catch {
      /* ignore */
    }
    captureCtxRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);
  stopCaptureRef.current = stopCapture;

  const startTalking = useCallback(async () => {
    if (capturingRef.current) return;
    setError(null);
    setReply("");
    setTranscript("");
    setLastTool(null);
    setPhase("connecting");
    // Must run inside the user gesture for iOS to unlock playback.
    ensurePlayback();
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        "Microphone isn't available in this context. Open Bebop over HTTPS (or the native app) so the browser grants mic access.",
      );
      setPhase("error");
      return;
    }
    try {
      const ws = await openSocket();
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      } catch (e) {
        throw new Error(micErrorMessage(e));
      }
      streamRef.current = stream;

      // Prefer 16 kHz so the WAV the service builds is already at the model's
      // rate, but mobile may coerce the context rate — report the real one.
      const ctx = createAudioContext(CAPTURE_RATE);
      captureCtxRef.current = ctx;
      await ctx.resume().catch(() => {});
      const source = ctx.createMediaStreamSource(stream);
      // A muted gain keeps the node in the render graph without echoing the
      // mic back to the speakers.
      const mute = ctx.createGain();
      mute.gain.value = 0;
      mute.connect(ctx.destination);
      const push = (frame: Float32Array) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(f32ToI16(frame));
      };
      if (ctx.audioWorklet) {
        const blob = new Blob([CAPTURE_WORKLET], { type: "application/javascript" });
        const url = URL.createObjectURL(blob);
        try {
          await ctx.audioWorklet.addModule(url);
        } finally {
          URL.revokeObjectURL(url);
        }
        const worklet = new AudioWorkletNode(ctx, "bebop-pcm-capture");
        worklet.port.onmessage = (ev: MessageEvent<Float32Array>) => push(ev.data);
        source.connect(worklet);
        worklet.connect(mute);
        captureNodeRef.current = worklet;
      } else {
        // Pre-AudioWorklet WebViews (older iOS/Android) fall back to the
        // deprecated ScriptProcessorNode.
        const proc = ctx.createScriptProcessor(4096, 1, 1);
        proc.onaudioprocess = (ev) => push(ev.inputBuffer.getChannelData(0));
        source.connect(proc);
        proc.connect(mute);
        captureNodeRef.current = proc;
      }
      capturingRef.current = true;
      ws.send(
        JSON.stringify({
          type: "utterance_start",
          sample_rate: ctx.sampleRate,
          format: "pcm16",
        }),
      );
      setPhase("listening");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
      stopCapture();
    }
  }, [ensurePlayback, openSocket, stopCapture]);

  const stopTalking = useCallback(() => {
    if (!capturingRef.current) return;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "utterance_end" }));
      setPhase("thinking");
    }
    stopCapture();
  }, [stopCapture]);

  const resetContext = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reset" }));
    }
    setReply("");
  }, []);

  return {
    phase,
    transcript,
    reply,
    error,
    lastTool,
    startTalking,
    stopTalking,
    resetContext,
  };
}

/// Status reported by `GET /healthz` on the voice service.
export interface VoiceHealth {
  ok: boolean;
  /// "starting" | "building" | "ready" | "error" | "stub"
  phase: string;
  /// Human-readable progress line (e.g. "building talker engine").
  detail: string;
  model: string;
  /// "nvfp4" | "fp16" — which checkpoint the service is serving.
  precision: string;
  modelDownloaded: boolean;
  /// Seconds since the model server was started (anchors the build timer).
  elapsedS: number;
  /// Wall-clock ms of the last supervisor heartbeat.
  heartbeatMs: number;
  /// Age of the last heartbeat, computed server-side (clock-consistent). A
  /// value that keeps growing means the build supervisor itself is stuck.
  heartbeatAgeS: number | null;
  /// Component currently being built (e.g. "talker"), or "".
  component: string;
  /// Components finished this build (e.g. ["llm", "visual", "audio"]).
  componentsDone: string[];
}

/// Poll the voice service's `/healthz` while the page is open (and the
/// service is systemd-running). Returns `null` until the first successful
/// response — including during the window before the gateway's HTTP server is
/// listening, or when the service is stopped.
export function useVoiceHealth(
  host: string,
  port = 9093,
  enabled = true,
  pollMs = 1500,
): VoiceHealth | null {
  const [health, setHealth] = useState<VoiceHealth | null>(null);
  useEffect(() => {
    if (!enabled || !host) {
      setHealth(null);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      try {
        const res = await fetch(httpUrl(host, port, "/healthz"), {
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const j = (await res.json()) as Record<string, unknown>;
        if (!cancelled) {
          setHealth({
            ok: Boolean(j.ok),
            phase: String(j.phase ?? ""),
            detail: String(j.detail ?? ""),
            model: String(j.model ?? ""),
            precision: String(j.precision ?? ""),
            modelDownloaded: Boolean(j.model_downloaded),
            elapsedS: Number(j.elapsed_s ?? 0),
            heartbeatMs: Number(j.heartbeat_ms ?? 0),
            heartbeatAgeS:
              j.heartbeat_age_s === null || j.heartbeat_age_s === undefined
                ? null
                : Number(j.heartbeat_age_s),
            component: String(j.component ?? ""),
            componentsDone: Array.isArray(j.components_done)
              ? (j.components_done as string[])
              : [],
          });
        }
      } catch {
        if (!cancelled) setHealth(null);
      } finally {
        if (!cancelled) timer = setTimeout(tick, pollMs);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [host, port, enabled, pollMs]);
  return health;
}

/// Operator-tunable voice settings (`GET/POST /config` on the voice service).
export interface VoiceConfigView {
  voice: string;
  system_prompt: string;
  history_turns: number;
  keep_audio_history: boolean;
  tools_enabled: boolean;
  backend: string;
  brain: string;
  openrouter_model: string;
  openrouter_set: boolean;
  backends: string[];
  voices: string[];
  /** Write-only: sent in POST /config, never returned by the server. */
  openrouter_key?: string;
}

export interface VoiceConfigState {
  config: VoiceConfigView | null;
  error: string | null;
  save: (partial: Partial<VoiceConfigView>) => Promise<void>;
  reload: () => void;
}

function configFromJson(j: Record<string, unknown>): VoiceConfigView {
  return {
    voice: String(j.voice ?? ""),
    system_prompt: String(j.system_prompt ?? ""),
    history_turns: Number(j.history_turns ?? 0),
    keep_audio_history: Boolean(j.keep_audio_history),
    tools_enabled: Boolean(j.tools_enabled),
    backend: String(j.backend ?? "omni"),
    brain: String(j.brain ?? "local"),
    openrouter_model: String(j.openrouter_model ?? ""),
    openrouter_set: Boolean(j.openrouter_set),
    backends: Array.isArray(j.backends) ? (j.backends as string[]) : ["omni", "cascade"],
    voices: Array.isArray(j.voices) ? (j.voices as string[]) : [],
  };
}

export function useVoiceConfig(
  host: string,
  port = 9093,
  enabled = true,
): VoiceConfigState {
  const [config, setConfig] = useState<VoiceConfigView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!enabled || !host) {
      setConfig(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch(httpUrl(host, port, "/config"), {
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        if (!cancelled) {
          setConfig(configFromJson((await res.json()) as Record<string, unknown>));
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [host, port, enabled, nonce]);

  const save = useCallback(
    async (partial: Partial<VoiceConfigView>) => {
      const res = await fetch(httpUrl(host, port, "/config"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(partial),
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
        throw new Error(String(body.error ?? `HTTP ${res.status}`));
      }
      setConfig(configFromJson((await res.json()) as Record<string, unknown>));
      setError(null);
    },
    [host, port],
  );

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { config, error, save, reload };
}
