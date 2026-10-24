// HTTP side of the voice/agent service (:9093): `/healthz` (engine status,
// polled while a screen shows it) and `/config` (operator settings).

import { useCallback, useEffect, useState } from "react";

import { endpoints } from "../robot/endpoints";

/// Status reported by `GET /healthz` on the voice service.
export interface VoiceHealth {
  ok: boolean;
  /// "starting" | "building" | "ready" | "error" | "stub"
  phase: string;
  /// Human-readable progress line (e.g. "cascade ready").
  detail: string;
  /// Model id the service reports (for the cascade just "cascade").
  model: string;
  /// "nvfp4" | "fp16" | "mixed"
  precision: string;
  modelDownloaded: boolean;
  /// Seconds since the model server was started (anchors the build timer).
  elapsedS: number;
  /// Age of the last supervisor heartbeat (server-side). A value that keeps
  /// growing while building means the build supervisor itself is stuck.
  heartbeatAgeS: number | null;
  /// Component currently being built (e.g. "talker"), or "".
  component: string;
  /// Components finished this build.
  componentsDone: string[];
  /// Engine actually running: "cascade" | "omni" ("" on old services).
  backend: string;
  /// Whether this engine can call tools at all (cascade yes, omni no).
  toolsSupported: boolean;
  /// Live context + non-motion tools switched on.
  toolsEnabled: boolean;
  /// `move_relative` switched on.
  motionEnabled: boolean;
  /// A turn is running for some session on the server.
  turnActive: boolean;
}

function healthFromJson(j: Record<string, unknown>): VoiceHealth {
  return {
    ok: Boolean(j.ok),
    phase: String(j.phase ?? ""),
    detail: String(j.detail ?? ""),
    model: String(j.model ?? ""),
    precision: String(j.precision ?? ""),
    modelDownloaded: Boolean(j.model_downloaded),
    elapsedS: Number(j.elapsed_s ?? 0),
    heartbeatAgeS:
      j.heartbeat_age_s === null || j.heartbeat_age_s === undefined
        ? null
        : Number(j.heartbeat_age_s),
    component: String(j.component ?? ""),
    componentsDone: Array.isArray(j.components_done) ? (j.components_done as string[]) : [],
    backend: String(j.backend ?? ""),
    toolsSupported: Boolean(j.tools_supported),
    toolsEnabled: Boolean(j.tools_enabled),
    motionEnabled: Boolean(j.motion_enabled),
    turnActive: Boolean(j.turn_active),
  };
}

/// Human name of the engine. The firmware's `VoiceState.model` is the
/// catalog id it was installed with (e.g. "qwen3-omni-30b") and goes stale
/// once the service runs the cascade, so this reads `/healthz` instead.
export function engineLabel(backend: string): string {
  if (backend === "cascade") return "Cascade";
  if (backend === "omni") return "Omni";
  return backend || "Unknown engine";
}

/// The cascade's stages (docs/voice.md › Cascade backend).
export const CASCADE_STAGES = [
  { stage: "ASR", model: "Qwen3-ASR 1.7B" },
  { stage: "Brain", model: "Qwen3.8 27B (NVFP4)" },
  { stage: "TTS", model: "Qwen3-TTS 1.7B" },
] as const;

/// Consecutive failed polls before the last good answer is dropped (~6 s at
/// the default rate): one lost request on Wi-Fi must not read as "down".
const HEALTH_MISSES = 3;

/// Poll `/healthz` while `enabled`. `null` until the first answer, and once
/// the service has been unreachable for HEALTH_MISSES polls in a row
/// (stopped, still binding its port); a single failed poll keeps the last
/// good value.
export function useVoiceHealth(host: string, enabled = true, pollMs = 2000): VoiceHealth | null {
  const [health, setHealth] = useState<VoiceHealth | null>(null);
  useEffect(() => {
    if (!enabled || !host) {
      setHealth(null);
      return;
    }
    let cancelled = false;
    let misses = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      try {
        const res = await fetch(endpoints(host).voiceHttp("/healthz"), {
          cache: "no-store",
          signal: AbortSignal.timeout(4000),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const j = (await res.json()) as Record<string, unknown>;
        misses = 0;
        if (!cancelled) setHealth(healthFromJson(j));
      } catch {
        misses += 1;
        if (!cancelled && misses >= HEALTH_MISSES) setHealth(null);
      } finally {
        if (!cancelled) timer = setTimeout(tick, document.hidden ? pollMs * 3 : pollMs);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [host, enabled, pollMs]);
  return health;
}

/// Operator-tunable voice settings (`GET/POST /config` on the voice service).
export interface VoiceConfigView {
  voice: string;
  system_prompt: string;
  history_turns: number;
  keep_audio_history: boolean;
  tools_enabled: boolean;
  motion_enabled: boolean;
  tools_supported: boolean;
  backend: string;
  backends: string[];
  voices: string[];
  /** When set, the bebop-models supervisor owns the omni backend. */
  omni_supervisor_url: string;
  omni_backend_id: string;
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
    motion_enabled: Boolean(j.motion_enabled),
    tools_supported: j.tools_supported === undefined ? true : Boolean(j.tools_supported),
    backend: String(j.backend ?? "cascade"),
    backends: Array.isArray(j.backends) ? (j.backends as string[]) : ["cascade", "omni"],
    voices: Array.isArray(j.voices) ? (j.voices as string[]) : [],
    omni_supervisor_url: String(j.omni_supervisor_url ?? ""),
    omni_backend_id: String(j.omni_backend_id ?? "omni-edgellm"),
  };
}

export function useVoiceConfig(host: string, enabled = true): VoiceConfigState {
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
        const res = await fetch(endpoints(host).voiceHttp("/config"), {
          cache: "no-store",
          signal: AbortSignal.timeout(5000),
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
  }, [host, enabled, nonce]);

  const save = useCallback(
    async (partial: Partial<VoiceConfigView>) => {
      const res = await fetch(endpoints(host).voiceHttp("/config"), {
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
    [host],
  );

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { config, error, save, reload };
}
