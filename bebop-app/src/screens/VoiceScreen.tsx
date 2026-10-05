import { useEffect, useRef, useState } from "react";

import {
  getOrCreateRuntimeTransport,
  type RuntimeTransport,
  type VoiceView,
} from "../runtime";
import { Banner, Button, Card, Field } from "../components/ui";
import {
  useDuplexSession,
  useVoiceConfig,
  useVoiceHealth,
  useVoiceSession,
  type VoiceConfigView,
} from "../voice/useVoiceSession";

const EMPTY_VOICE: VoiceView = {
  present: false,
  running: false,
  state: "",
  detail: "",
  service: "",
  model: "",
};

const SESSION_KEY = "bebop.voice.session";
const INPUT_CLASS =
  "w-full bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-2.5 text-text outline-none focus:border-accent";

function formatElapsed(s: number): string {
  if (!s || s < 0) return "";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function makeSessionId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
}

function loadSessionId(): string {
  try {
    const v = localStorage.getItem(SESSION_KEY);
    if (v) return v;
  } catch {
    /* ignore */
  }
  return makeSessionId();
}

/// Voice (speech-to-speech) control + status. Start/stop the on-robot
/// `bebop-voice.service` (Qwen3-Omni via TensorRT Edge-LLM) over the runtime
/// WebSocket; audio is served by that unit on :9093. Voice, persona, and
/// context are edited here and persisted on the robot (`/config`).
export function VoiceScreen({
  robotIp,
  runtimePort = 9090,
  onBack,
}: {
  robotIp: string;
  runtimePort?: number;
  onBack: () => void;
}) {
  const transportRef = useRef<RuntimeTransport | null>(null);
  const [voice, setVoice] = useState<VoiceView>(EMPTY_VOICE);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>(() => loadSessionId());
  const [personaDraft, setPersonaDraft] = useState("");
  const [orkeyDraft, setOrkeyDraft] = useState("");
  const [savingCfg, setSavingCfg] = useState<string | null>(null);
  const [cfgError, setCfgError] = useState<string | null>(null);

  const cfg = useVoiceConfig(robotIp, 9093, voice.running);
  const health = useVoiceHealth(robotIp, 9093, voice.running);
  const session = useVoiceSession(robotIp, 9093, { session: sessionId });
  const duplex = useDuplexSession(robotIp, 9093);

  useEffect(() => {
    try {
      localStorage.setItem(SESSION_KEY, sessionId);
    } catch {
      /* ignore */
    }
  }, [sessionId]);

  const systemPrompt = cfg.config?.system_prompt;
  useEffect(() => {
    if (systemPrompt !== undefined) setPersonaDraft(systemPrompt);
  }, [systemPrompt]);

  // Telemetry: service lifecycle state from the runtime WS.
  useEffect(() => {
    if (!robotIp) return;
    const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
    transportRef.current = transport;
    let cancelled = false;
    const offTelemetry = transport.onTelemetry((s) => {
      if (!cancelled) setVoice(s.voice);
    });
    void transport
      .connect(robotIp, runtimePort)
      .then(() => transport.subscribeTelemetry(5))
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      offTelemetry();
    };
  }, [robotIp, runtimePort]);

  const transport = transportRef.current;

  const ready = health?.ok ?? false;
  const building = health?.phase === "starting" || health?.phase === "building";
  const failed = health?.phase === "error";
  const statusText = !voice.running
    ? voice.state || "Stopped"
    : health == null
      ? "Starting…"
      : health.phase === "ready"
        ? "Ready"
        : health.phase === "stub"
          ? "Ready (stub)"
          : building
            ? "Building engines…"
            : failed
              ? "Error"
              : health.phase;
  const statusDetail =
    building && health
      ? `${health.detail}${health.elapsedS ? ` · ${formatElapsed(health.elapsedS)}` : ""}`
      : health?.detail || voice.detail;
  const stale = Boolean(
    building && health?.heartbeatAgeS != null && health.heartbeatAgeS > 10,
  );
  const dotClass = ready
    ? "bg-success"
    : failed || stale
      ? "bg-danger"
      : building
        ? "bg-accent animate-pulse"
        : "bg-text-dim/40";
  const talkActive = session.phase === "listening";

  async function toggle() {
    if (!transport) return;
    setError(null);
    setBusy(true);
    try {
      await transport.setVoiceEnabled(!voice.running);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function saveCfg(key: string, partial: Partial<VoiceConfigView>) {
    setCfgError(null);
    setSavingCfg(key);
    try {
      await cfg.save(partial);
    } catch (e) {
      setCfgError(e instanceof Error ? e.message : String(e));
    } finally {
      setSavingCfg(null);
    }
  }

  return (
    <div className="flex flex-col flex-1 gap-4 pb-[max(1rem,env(safe-area-inset-bottom))]">
      <h2 className="text-2xl font-bold mt-2">Voice</h2>
      <p className="text-text-dim leading-relaxed">
        Talk to the robot. Speech-to-speech runs on-device with Qwen3-Omni;
        download the voice model from the Models page first.
      </p>

      {error ? <Banner tone="error">{error}</Banner> : null}

      {!voice.present ? (
        <Banner tone="info">
          Voice support isn&apos;t installed on this robot yet. Deploy the
          current firmware and the <code>bebop-voice.service</code> unit (see{" "}
          <code>scripts/install-jetson.sh</code>), then reconnect.
        </Banner>
      ) : (
        <Card>
          <div className="py-2 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="text-xs text-text-dim uppercase tracking-wider">
                Speech-to-speech
              </div>
              <div className="flex items-center gap-2">
                <div className={`w-2.5 h-2.5 rounded-full ${dotClass}`} />
                <span className="text-[12px] text-text-dim">{statusText}</span>
              </div>
            </div>
            {voice.model ? (
              <div className="text-[13px] text-text-dim font-mono">
                {voice.model}
                {health?.precision ? ` · ${health.precision}` : ""}
              </div>
            ) : null}
            {statusDetail ? (
              <div
                className={`text-[12px] leading-snug ${
                  failed ? "text-danger" : "text-text-dim"
                }`}
              >
                {statusDetail}
              </div>
            ) : null}
            {stale ? (
              <Banner tone="error">
                No heartbeat for {Math.round(health?.heartbeatAgeS ?? 0)}s — the
                build may be stuck. Check{" "}
                <code>journalctl -u bebop-voice</code>.
              </Banner>
            ) : building && health && health.componentsDone.length > 0 ? (
              <div className="text-[11px] text-text-dim leading-snug">
                built: {health.componentsDone.join(", ")}
                {health.component ? ` · now: ${health.component}` : ""}
              </div>
            ) : null}
            <Button
              variant={voice.running ? "ghost" : "secondary"}
              loading={busy}
              onClick={toggle}
            >
              {voice.running ? "Stop voice service" : "Start voice service"}
            </Button>
            {!voice.running ? (
              <p className="text-[12px] text-text-dim leading-relaxed">
                The first start downloads nothing but builds the model&apos;s
                TensorRT engines — this can take several minutes.
              </p>
            ) : null}
          </div>
        </Card>
      )}

      <Card>
        <div className="py-2 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="text-xs text-text-dim uppercase tracking-wider">
              Duplex (live)
            </div>
            <span className="flex items-center gap-2 text-[12px] text-text-dim">
              {duplex.active ? (
                <span
                  className={`w-2 h-2 rounded-full ${
                    duplex.listening ? "bg-accent animate-pulse" : "bg-success"
                  }`}
                />
              ) : null}
              {duplex.active ? (duplex.listening ? "Listening…" : "Speaking") : "Off"}
            </span>
          </div>
          <p className="text-[13px] text-text-dim">
            Continuous, interruptible voice (MiniCPM-o). No tap-to-talk — just talk;
            it listens and speaks at once and uses the robot's cameras.
          </p>
          <div className="flex items-center gap-2">
            {duplex.active ? (
              <Button onClick={() => duplex.stop()}>Stop duplex</Button>
            ) : (
              <Button onClick={() => void duplex.start()} disabled={!voice.running}>
                Start duplex
              </Button>
            )}
          </div>
          {duplex.error ? <Banner tone="error">{duplex.error}</Banner> : null}
          {duplex.reply ? (
            <div className="text-[13px] text-text leading-snug whitespace-pre-wrap">
              {duplex.reply}
            </div>
          ) : null}
        </div>
      </Card>

      <Card>
        <div className="py-2 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="text-xs text-text-dim uppercase tracking-wider">
              Talk
            </div>
            <span className="flex items-center gap-2 text-[12px] text-text-dim">
              {talkActive ? (
                <span className="w-2 h-2 rounded-full bg-danger animate-pulse" />
              ) : null}
              {session.phase === "listening"
                ? "Listening…"
                : session.phase === "thinking"
                  ? session.lastTool
                    ? `Checking (${session.lastTool})…`
                    : "Thinking…"
                  : session.phase === "speaking"
                    ? "Speaking…"
                    : session.phase === "connecting"
                      ? "Connecting…"
                      : session.phase === "error"
                        ? "Error"
                        : ""}
            </span>
          </div>
          {session.error ? <Banner tone="error">{session.error}</Banner> : null}
          <Button
            variant={talkActive ? "primary" : "secondary"}
            disabled={
              !ready ||
              session.phase === "thinking" ||
              session.phase === "speaking"
            }
            onClick={() =>
              talkActive ? session.stopTalking() : void session.startTalking()
            }
            onContextMenu={(e) => e.preventDefault()}
            className="w-full py-5 text-base select-none touch-none"
          >
            {talkActive ? "Tap to send" : "Tap to talk"}
          </Button>
          {session.reply ? (
            <p className="text-[14px] leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto">
              {session.reply}
            </p>
          ) : (
            <p className="text-[13px] text-text-dim leading-relaxed">
              {ready
                ? "Tap the button, speak, then tap again to send. The robot replies with speech."
                : building
                  ? "Waiting for the engines to finish building…"
                  : failed
                    ? "The voice service reported an error — see the status above."
                    : "Start the voice service above, then tap to talk."}
            </p>
          )}
          <p className="text-[11px] text-text-dim leading-snug">
            Microphone access needs a secure connection — use the app over
            HTTPS or the native build on a phone.
          </p>
        </div>
      </Card>

      {voice.present && voice.running ? (
        <Card>
          <div className="py-2 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="text-xs text-text-dim uppercase tracking-wider">
                Voice &amp; personality
              </div>
              <button
                type="button"
                className="text-[12px] text-text-dim hover:text-text disabled:opacity-50"
                disabled={cfg.config == null}
                onClick={() => {
                  setSessionId(makeSessionId());
                  session.resetContext();
                }}
              >
                New conversation
              </button>
            </div>
            {cfgError ? <Banner tone="error">{cfgError}</Banner> : null}
            {!cfg.config ? (
              <p className="text-[13px] text-text-dim">Loading settings…</p>
            ) : (
              <>
                <Field
                  label="Voice"
                  hint="Speaker baked into the checkpoint (from the model)."
                >
                  <select
                    className={INPUT_CLASS}
                    value={cfg.config.voice}
                    disabled={
                      savingCfg === "voice" || cfg.config.voices.length === 0
                    }
                    onChange={(e) =>
                      void saveCfg("voice", { voice: e.currentTarget.value })
                    }
                  >
                    {(cfg.config.voices.length
                      ? cfg.config.voices
                      : [cfg.config.voice]
                    ).map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field
                  label="Memory"
                  hint="How many past turns the robot carries into the next reply."
                >
                  <select
                    className={INPUT_CLASS}
                    value={cfg.config.history_turns}
                    disabled={savingCfg === "turns"}
                    onChange={(e) =>
                      void saveCfg("turns", {
                        history_turns: Number(e.currentTarget.value),
                      })
                    }
                  >
                    <option value={0}>Off</option>
                    <option value={2}>2 turns</option>
                    <option value={4}>4 turns</option>
                    <option value={8}>8 turns</option>
                    <option value={12}>12 turns</option>
                  </select>
                </Field>
                <label className="flex items-center gap-2 text-[13px] text-text-dim">
                  <input
                    type="checkbox"
                    checked={cfg.config.keep_audio_history}
                    disabled={savingCfg === "audio"}
                    onChange={(e) =>
                      void saveCfg("audio", {
                        keep_audio_history: e.currentTarget.checked,
                      })
                    }
                  />
                  Remember my voice (keeps your audio in context; slower)
                </label>
                <label className="flex items-center gap-2 text-[13px] text-text-dim">
                  <input
                    type="checkbox"
                    checked={cfg.config.tools_enabled}
                    disabled={savingCfg === "tools"}
                    onChange={(e) =>
                      void saveCfg("tools", {
                        tools_enabled: e.currentTarget.checked,
                      })
                    }
                  />
                  Live context (attach the camera views and robot status to
                  every turn)
                </label>
                <Field
                  label="Engine"
                  hint="omni = one end-to-end model; cascade = ASR → Qwen3.8 brain → TTS. Toggle Voice off/on to apply (only the selected backend loads)."
                >
                  <select
                    className={INPUT_CLASS}
                    value={cfg.config.backend}
                    disabled={savingCfg === "backend"}
                    onChange={(e) =>
                      void saveCfg("backend", { backend: e.currentTarget.value })
                    }
                  >
                    {(cfg.config.backends.length
                      ? cfg.config.backends
                      : ["omni", "cascade"]
                    ).map((b) => (
                      <option key={b} value={b}>
                        {b === "cascade"
                          ? "Cascade (best model per stage)"
                          : "Omni (end-to-end)"}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field
                  label="Brain"
                  hint="Local = on-device VLM; OpenRouter = a cloud model you pick (needs an API key)."
                >
                  <select
                    className={INPUT_CLASS}
                    value={cfg.config.brain}
                    disabled={savingCfg === "brain"}
                    onChange={(e) =>
                      void saveCfg("brain", { brain: e.currentTarget.value })
                    }
                  >
                    <option value="local">Local (on-device)</option>
                    <option value="openrouter">OpenRouter (cloud)</option>
                  </select>
                </Field>
                {cfg.config.brain === "openrouter" && (
                  <>
                    <Field
                      label="OpenRouter model"
                      hint="Any OpenRouter slug, e.g. openai/gpt-5.6, anthropic/claude-opus-4.6, google/gemini-3-pro."
                    >
                      <input
                        className={INPUT_CLASS}
                        defaultValue={cfg.config.openrouter_model}
                        placeholder="openrouter/auto"
                        onBlur={(e) =>
                          void saveCfg("ormodel", {
                            openrouter_model: e.currentTarget.value,
                          })
                        }
                      />
                    </Field>
                    <Field
                      label="OpenRouter API key"
                      hint={
                        cfg.config.openrouter_set
                          ? "A key is stored on the robot (write-only)."
                          : "Not set. Paste your sk-or-… key."
                      }
                    >
                      <div className="flex gap-2">
                        <input
                          type="password"
                          className={INPUT_CLASS}
                          value={orkeyDraft}
                          placeholder={
                            cfg.config.openrouter_set ? "•••••••• (set)" : "sk-or-…"
                          }
                          onChange={(e) => setOrkeyDraft(e.currentTarget.value)}
                        />
                        <button
                          className="shrink-0 rounded border border-border px-3 text-[13px]"
                          disabled={!orkeyDraft || savingCfg === "orkey"}
                          onClick={() => {
                            void saveCfg("orkey", { openrouter_key: orkeyDraft });
                            setOrkeyDraft("");
                          }}
                        >
                          Save
                        </button>
                        {cfg.config.openrouter_set && (
                          <button
                            className="shrink-0 rounded border border-border px-3 text-[13px]"
                            disabled={savingCfg === "orkey"}
                            onClick={() =>
                              void saveCfg("orkey", { openrouter_key: "" })
                            }
                          >
                            Clear
                          </button>
                        )}
                      </div>
                    </Field>
                  </>
                )}
                <Field
                  label="Personality"
                  hint="System prompt. Applies on the next turn."
                >
                  <textarea
                    className={`${INPUT_CLASS} min-h-24`}
                    value={personaDraft}
                    spellCheck={false}
                    onChange={(e) => setPersonaDraft(e.currentTarget.value)}
                    onBlur={() => {
                      const next = personaDraft.trim();
                      if (
                        cfg.config &&
                        next &&
                        next !== cfg.config.system_prompt
                      ) {
                        void saveCfg("persona", { system_prompt: next });
                      }
                    }}
                  />
                </Field>
                <p className="text-[11px] text-text-dim leading-snug">
                  Conversation context is kept per session and cleared by{" "}
                  <strong>New conversation</strong>.
                </p>
              </>
            )}
          </div>
        </Card>
      ) : null}

      <div className="mt-auto pt-4">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
      </div>
    </div>
  );
}
