// Settings › Voice: engine, voice, memory, context and persona of the
// on-robot voice/agent service (`GET/POST /config` on :9093). Applies to
// every operator of this robot.

import { useEffect, useState } from "react";

import {
  Banner,
  Card,
  CardHeader,
  Field,
  INPUT_CLASS,
  TEXTAREA_CLASS,
} from "../../components/ui";
import { useRobot, useTelemetry } from "../../robot/session";
import { useVoiceConfig } from "../../voice/useVoiceSession";
import type { VoiceConfigView } from "../../voice/useVoiceSession";

export function VoiceSettings() {
  const { host } = useRobot();
  const voice = useTelemetry((s) => ({ present: !!s?.voice.present, running: !!s?.voice.running }), 2);
  const cfg = useVoiceConfig(host, voice.running);
  const [personaDraft, setPersonaDraft] = useState("");
  const [supervisorDraft, setSupervisorDraft] = useState("");
  const [savingCfg, setSavingCfg] = useState<string | null>(null);
  const [cfgError, setCfgError] = useState<string | null>(null);

  const systemPrompt = cfg.config?.system_prompt;
  useEffect(() => {
    if (systemPrompt !== undefined) setPersonaDraft(systemPrompt);
  }, [systemPrompt]);
  const supervisorUrl = cfg.config?.omni_supervisor_url;
  useEffect(() => {
    if (supervisorUrl !== undefined) setSupervisorDraft(supervisorUrl);
  }, [supervisorUrl]);

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
    <Card className="flex flex-col gap-3">
      <CardHeader
        eyebrow="Voice & agent"
        title="Engine, voice and personality"
        description="Stored on the robot. Settings load while the voice service is running."
      />
      {!voice.present ? (
        <Banner tone="info">
          The voice service isn&apos;t installed on this robot.
        </Banner>
      ) : !voice.running ? (
        <Banner tone="info">
          Start the voice service (Autonomy or Diagnostics › Services) to edit
          these settings.
        </Banner>
      ) : null}
      {cfgError ? <Banner tone="error">{cfgError}</Banner> : null}
      {voice.running && !cfg.config ? (
        <p className="text-body-sm text-text-dim">Loading settings…</p>
      ) : null}
      {cfg.config ? (
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
        <label className="flex min-h-touch items-center gap-2.5 text-body-sm text-text-dim">
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
        <label className="flex min-h-touch items-center gap-2.5 text-body-sm text-text-dim">
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
          hint="Cascade (ASR → brain → TTS) is the default and the only engine with tools (places, robot status, moves). Omni is one end-to-end model, chat only. Restart the voice service to apply."
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
        {cfg.config.backend === "omni" && (
          <Field
            label="Omni source (supervisor)"
            hint="Optional. If set, the bebop-models supervisor owns the omni backend (GPU budget + lifecycle). Leave empty to supervise Edge-LLM locally. Applies on the next Voice toggle."
          >
            <input
              className={INPUT_CLASS}
              placeholder="http://127.0.0.1:9094"
              spellCheck={false}
              value={supervisorDraft}
              onChange={(e) =>
                setSupervisorDraft(e.currentTarget.value)
              }
              onBlur={() => {
                const next = supervisorDraft.trim();
                if (next !== (cfg.config?.omni_supervisor_url ?? "")) {
                  void saveCfg("supervisor", {
                    omni_supervisor_url: next,
                  });
                }
              }}
            />
          </Field>
        )}
        <Field
          label="Personality"
          hint="System prompt. Applies on the next turn."
        >
          <textarea
            className={TEXTAREA_CLASS}
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
        </>
      ) : null}
    </Card>
  );
}
