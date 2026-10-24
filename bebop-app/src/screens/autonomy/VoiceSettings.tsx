// Settings › Voice: engine, voice, memory, permissions and persona of the
// on-robot voice/agent service (`GET/POST /config` on :9093). Applies to
// every operator of this robot. Mounted by the Settings route.

import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import {
  Banner,
  Card,
  CardHeader,
  Field,
  INPUT_CLASS,
  Switch,
  TEXTAREA_CLASS,
} from "../../components/ui";
import { useTelemetry } from "../../robot/session";
import { useVoiceConfig } from "../../voice/voiceService";
import type { VoiceConfigView } from "../../voice/voiceService";

function ToggleRow({
  title,
  hint,
  checked,
  disabled,
  onChange,
}: {
  title: string;
  hint: ReactNode;
  checked: boolean;
  disabled?: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-3 border-t border-border pt-3">
      <div className="min-w-0">
        <div className="text-sm font-semibold text-text">{title}</div>
        <p className="mt-0.5 text-label leading-relaxed text-text-dim">{hint}</p>
      </div>
      <Switch label={title} checked={checked} disabled={disabled} onChange={onChange} />
    </div>
  );
}

export function VoiceSettingsPanel({ host }: { host: string }) {
  const voice = useTelemetry(
    (s) => ({ loaded: !!s, present: !!s?.voice.present, running: !!s?.voice.running }),
    2,
  );
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

  const c = cfg.config;
  const toolsOff = c ? !c.tools_supported || c.backend === "omni" : false;

  return (
    <Card className="flex flex-col gap-3">
      <CardHeader
        eyebrow="Voice & agent"
        title="Engine, voice and permissions"
        description="Stored on the robot and shared by everyone who connects to it."
      />
      {voice.loaded && !voice.present ? (
        <Banner tone="info">The voice service isn&apos;t installed on this robot.</Banner>
      ) : voice.loaded && !voice.running ? (
        <Banner tone="info">
          Start the voice service (Autonomy or Diagnostics › Services) to edit these settings.
        </Banner>
      ) : null}
      {cfgError ? <Banner tone="error">{cfgError}</Banner> : null}
      {cfg.error && !c ? <Banner tone="error">Couldn&apos;t load settings: {cfg.error}</Banner> : null}
      {voice.running && !c && !cfg.error ? (
        <p className="text-body-sm text-text-dim">Loading settings…</p>
      ) : null}
      {c ? (
        <>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field
              label="Engine"
              hint="Cascade (speech recognition, brain, speech) is the default and the only engine with tools. Omni is one end-to-end model, chat only. Restart the voice service to apply."
            >
              <select
                className={INPUT_CLASS}
                value={c.backend}
                disabled={savingCfg === "backend"}
                onChange={(e) => void saveCfg("backend", { backend: e.currentTarget.value })}
              >
                {(c.backends.length ? c.backends : ["cascade", "omni"]).map((b) => (
                  <option key={b} value={b}>
                    {b === "cascade" ? "Cascade (tools)" : b === "omni" ? "Omni (chat only)" : b}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Voice" hint="The speaker the robot replies with.">
              <select
                className={INPUT_CLASS}
                value={c.voice}
                disabled={savingCfg === "voice" || c.voices.length === 0}
                onChange={(e) => void saveCfg("voice", { voice: e.currentTarget.value })}
              >
                {(c.voices.length ? c.voices : [c.voice]).map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Memory" hint="How many past turns the robot carries into the next reply.">
              <select
                className={INPUT_CLASS}
                value={c.history_turns}
                disabled={savingCfg === "turns"}
                onChange={(e) =>
                  void saveCfg("turns", { history_turns: Number(e.currentTarget.value) })
                }
              >
                {[0, 2, 4, 8, 12, 16].map((n) => (
                  <option key={n} value={n}>
                    {n === 0 ? "Off" : `${n} turns`}
                  </option>
                ))}
                {[0, 2, 4, 8, 12, 16].includes(c.history_turns) ? null : (
                  <option value={c.history_turns}>{c.history_turns} turns</option>
                )}
              </select>
            </Field>
          </div>

          <ToggleRow
            title="Live context and tools"
            hint={
              toolsOff
                ? "The omni engine has no tools; switch to Cascade to use them."
                : "Attach the camera views and robot status to every turn, and let the robot read and edit place memory."
            }
            checked={c.tools_enabled}
            disabled={savingCfg === "tools"}
            onChange={(v) => void saveCfg("tools", { tools_enabled: v })}
          />
          <ToggleRow
            title="Allow moves"
            hint="Let the robot make small bounded moves when asked (at most 2 m or 180° per step). E-STOP and Stop task always stop it."
            checked={c.motion_enabled}
            disabled={savingCfg === "motion" || !c.tools_enabled}
            onChange={(v) => void saveCfg("motion", { motion_enabled: v })}
          />
          {c.backend === "omni" ? (
            <ToggleRow
              title="Remember my voice"
              hint="Keep your recorded audio in the omni engine's context (slower)."
              checked={c.keep_audio_history}
              disabled={savingCfg === "audio"}
              onChange={(v) => void saveCfg("audio", { keep_audio_history: v })}
            />
          ) : null}
          {c.backend === "omni" ? (
            <Field
              label="Omni source (supervisor)"
              hint="Optional. When set, the bebop-models supervisor owns the omni engine (GPU budget and lifecycle). Leave empty to run it locally. Applies on the next service start."
            >
              <input
                className={INPUT_CLASS}
                placeholder="http://127.0.0.1:9094"
                spellCheck={false}
                value={supervisorDraft}
                onChange={(e) => setSupervisorDraft(e.currentTarget.value)}
                onBlur={() => {
                  const next = supervisorDraft.trim();
                  if (next !== (cfg.config?.omni_supervisor_url ?? "")) {
                    void saveCfg("supervisor", { omni_supervisor_url: next });
                  }
                }}
              />
            </Field>
          ) : null}
          <div className="border-t border-border pt-3">
            <Field label="Personality" hint="System prompt. Saved when you leave the field; applies on the next turn.">
              <textarea
                className={TEXTAREA_CLASS}
                value={personaDraft}
                spellCheck={false}
                onChange={(e) => setPersonaDraft(e.currentTarget.value)}
                onBlur={() => {
                  const next = personaDraft.trim();
                  if (cfg.config && next && next !== cfg.config.system_prompt) {
                    void saveCfg("persona", { system_prompt: next });
                  }
                }}
              />
            </Field>
          </div>
        </>
      ) : null}
    </Card>
  );
}
