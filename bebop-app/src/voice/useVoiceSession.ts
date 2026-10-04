import { useCallback, useEffect, useRef, useState } from "react";

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
  startTalking: () => Promise<void>;
  stopTalking: () => void;
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

export function useVoiceSession(host: string, port = 9093): VoiceSession {
  const [phase, setPhase] = useState<VoicePhase>("idle");
  const [transcript, setTranscript] = useState("");
  const [reply, setReply] = useState("");
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureCtxRef = useRef<AudioContext | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const playHeadRef = useRef(0);
  const stopCaptureRef = useRef<() => void>(undefined);

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
      playbackCtxRef.current = new AudioContext({ sampleRate: PLAYBACK_RATE });
      playHeadRef.current = 0;
    }
    void playbackCtxRef.current.resume();
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
      let msg: { type?: string; delta?: string; message?: string; sample_rate?: number };
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (msg.type) {
        case "text":
          if (msg.delta) setReply((r) => r + msg.delta);
          break;
        case "audio_start":
          setPhase("speaking");
          break;
        case "audio_end":
          break;
        case "done":
          setPhase("idle");
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
      const ws = new WebSocket(`ws://${host}:${port}/voice`);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => resolve(ws);
      ws.onerror = () => reject(new Error(`voice socket error (${host}:${port})`));
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
      };
      ws.onmessage = handleMessage;
      wsRef.current = ws;
    });
  }, [host, port, handleMessage]);

  const stopCapture = useCallback(() => {
    try {
      workletRef.current?.port.close();
    } catch {
      /* ignore */
    }
    workletRef.current = null;
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
    setError(null);
    setReply("");
    setTranscript("");
    setPhase("connecting");
    ensurePlayback();
    try {
      const ws = await openSocket();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      // Capture at 16 kHz so the WAV the service builds is already at the
      // model's expected rate (Chromium honours the requested rate; the
      // actual rate is reported to the service either way).
      const ctx = new AudioContext({ sampleRate: CAPTURE_RATE });
      captureCtxRef.current = ctx;
      const blob = new Blob([CAPTURE_WORKLET], { type: "application/javascript" });
      const url = URL.createObjectURL(blob);
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);

      const source = ctx.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(ctx, "bebop-pcm-capture");
      workletRef.current = worklet;
      // A muted gain keeps the worklet in the render graph without echoing the
      // mic back to the speakers.
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(worklet);
      worklet.connect(mute);
      mute.connect(ctx.destination);
      worklet.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        ws.send(f32ToI16(ev.data));
      };

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
    if (phase === "listening" || phase === "connecting") {
      setPhase("thinking");
    }
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "utterance_end" }));
    }
    stopCapture();
  }, [phase, stopCapture]);

  return { phase, transcript, reply, error, startTalking, stopTalking };
}

/// Status reported by `GET /healthz` on the voice service.
export interface VoiceHealth {
  ok: boolean;
  /// "starting" | "building" | "ready" | "error" | "stub"
  phase: string;
  /// Human-readable progress line (e.g. "building talker engine").
  detail: string;
  model: string;
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
        const res = await fetch(`http://${host}:${port}/healthz`, {
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
