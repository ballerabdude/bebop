// Microphone capture and speech playback for the agent console. Sockets and
// turns live in `useAgentSession`; this only turns the mic into PCM16
// frames and plays the robot's 24 kHz PCM16 replies.

import { useCallback, useEffect, useRef, useState } from "react";

const CAPTURE_RATE = 16000;
const PLAYBACK_RATE = 24000;

/// Inline AudioWorklet processor: forwards each 128-frame render quantum as a
/// Float32Array to the main thread. Kept as a string so it can be loaded from
/// a Blob URL without a separate bundled asset.
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
export function micErrorMessage(e: unknown): string {
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

/// Why the mic can't be used here, or null when it can. Browsers expose
/// `getUserMedia` only in a secure context (HTTPS, localhost, the app).
export function micUnavailableReason(): string | null {
  if (typeof window === "undefined") return "No browser.";
  if (!window.isSecureContext) return "Voice needs HTTPS or the app; type instead.";
  if (!navigator.mediaDevices?.getUserMedia) return "This browser has no microphone access; type instead.";
  return null;
}

export interface VoiceIO {
  /// null when the mic is usable, otherwise a short explanation.
  micBlocked: string | null;
  capturing: boolean;
  /// Open the mic and stream PCM16 frames to `onFrame`; resolves with the
  /// real capture rate. Throws a readable error.
  startCapture: (onFrame: (pcm: Int16Array) => void) => Promise<number>;
  stopCapture: () => void;
  /// Call inside a user gesture (iOS keeps audio locked until one).
  unlockPlayback: () => void;
  playChunk: (bytes: ArrayBuffer) => void;
  /// Drop everything queued (cancel).
  stopPlayback: () => void;
}

export function useVoiceIO(): VoiceIO {
  const [capturing, setCapturing] = useState(false);
  const [micBlocked] = useState(() => micUnavailableReason());
  const streamRef = useRef<MediaStream | null>(null);
  const captureCtxRef = useRef<AudioContext | null>(null);
  const captureNodeRef = useRef<AudioWorkletNode | ScriptProcessorNode | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const playHeadRef = useRef(0);
  const sourcesRef = useRef(new Set<AudioBufferSourceNode>());
  const capturingRef = useRef(false);

  const ensurePlayback = useCallback(() => {
    if (!playbackCtxRef.current) {
      playbackCtxRef.current = createAudioContext(PLAYBACK_RATE);
      playHeadRef.current = 0;
    }
    void playbackCtxRef.current.resume().catch(() => {});
    return playbackCtxRef.current;
  }, []);

  const unlockPlayback = useCallback(() => {
    try {
      ensurePlayback();
    } catch {
      /* no Web Audio: replies are text only */
    }
  }, [ensurePlayback]);

  const playChunk = useCallback(
    (bytes: ArrayBuffer) => {
      let ctx: AudioContext;
      try {
        ctx = ensurePlayback();
      } catch {
        return;
      }
      const samples = i16ToF32(bytes);
      if (!samples.length) return;
      const buffer = ctx.createBuffer(1, samples.length, PLAYBACK_RATE);
      buffer.copyToChannel(samples, 0);
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      const startAt = Math.max(ctx.currentTime, playHeadRef.current);
      src.start(startAt);
      playHeadRef.current = startAt + buffer.duration;
      sourcesRef.current.add(src);
      src.onended = () => sourcesRef.current.delete(src);
    },
    [ensurePlayback],
  );

  const stopPlayback = useCallback(() => {
    for (const src of sourcesRef.current) {
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    sourcesRef.current.clear();
    playHeadRef.current = 0;
  }, []);

  const stopCapture = useCallback(() => {
    capturingRef.current = false;
    setCapturing(false);
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
    void captureCtxRef.current?.close().catch(() => {});
    captureCtxRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const startCapture = useCallback(
    async (onFrame: (pcm: Int16Array) => void): Promise<number> => {
      if (capturingRef.current) throw new Error("Already listening.");
      const blocked = micUnavailableReason();
      if (blocked) throw new Error(blocked);
      capturingRef.current = true;
      setCapturing(true);
      try {
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
        // Prefer 16 kHz (the ASR rate); mobile may coerce it, so report the
        // real one.
        const ctx = createAudioContext(CAPTURE_RATE);
        captureCtxRef.current = ctx;
        await ctx.resume().catch(() => {});
        const source = ctx.createMediaStreamSource(stream);
        // A muted gain keeps the node rendering without echoing the mic.
        const mute = ctx.createGain();
        mute.gain.value = 0;
        mute.connect(ctx.destination);
        const push = (frame: Float32Array) => {
          if (capturingRef.current) onFrame(f32ToI16(frame));
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
          // Pre-AudioWorklet WebViews fall back to ScriptProcessorNode.
          const proc = ctx.createScriptProcessor(4096, 1, 1);
          proc.onaudioprocess = (ev) => push(ev.inputBuffer.getChannelData(0));
          source.connect(proc);
          proc.connect(mute);
          captureNodeRef.current = proc;
        }
        if (!capturingRef.current) {
          // Released while the permission prompt was up.
          stopCapture();
          throw new Error("released");
        }
        return ctx.sampleRate;
      } catch (e) {
        stopCapture();
        throw e;
      }
    },
    [stopCapture],
  );

  useEffect(
    () => () => {
      stopCapture();
      stopPlayback();
      void playbackCtxRef.current?.close().catch(() => {});
      playbackCtxRef.current = null;
    },
    [stopCapture, stopPlayback],
  );

  return { micBlocked, capturing, startCapture, stopCapture, unlockPlayback, playChunk, stopPlayback };
}
