import { useEffect, useRef, useState } from "react";

import {
  getOrCreateRuntimeTransport,
  type RuntimeTransport,
  type VoiceView,
} from "../runtime";
import { Banner, Button, Card } from "../components/ui";
import { useVoiceHealth, useVoiceSession } from "../voice/useVoiceSession";

const EMPTY_VOICE: VoiceView = {
  present: false,
  running: false,
  state: "",
  detail: "",
  service: "",
  model: "",
};

function formatElapsed(s: number): string {
  if (!s || s < 0) return "";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

/// Voice (speech-to-speech) control + status. Start/stop the on-robot
/// `bebop-voice.service` (Qwen3-Omni via TensorRT Edge-LLM) over the runtime
/// WebSocket; the audio itself is served by that unit on :9093.
///
/// The push-to-talk audio bridge lands with the voice service; this screen
/// owns service lifecycle and status so the operator can provision the model
/// (Models page), start the service, and see when it is ready.
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
  const session = useVoiceSession(robotIp, 9093);
  const health = useVoiceHealth(robotIp, 9093, voice.running);

  // Human-readable service status. Systemd `running` is necessary but not
  // sufficient: the model server may still be building engines or have failed.
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
  const statusDetail = building && health
    ? `${health.detail}${health.elapsedS ? ` · ${formatElapsed(health.elapsedS)}` : ""}`
    : health?.detail || voice.detail;
  const stale = Boolean(
    building && health?.heartbeatAgeS != null && health.heartbeatAgeS > 10,
  );
  const dotClass = ready
    ? "bg-success"
    : failed
      ? "bg-danger"
      : stale
        ? "bg-danger"
        : building
          ? "bg-accent animate-pulse"
          : "bg-text-dim/40";

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

  return (
    <div className="flex flex-col flex-1 gap-4">
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
            ) : building ? (
              <p className="text-[12px] text-text-dim leading-relaxed">
                First start only. The engines are cached, so later starts are
                quick.
              </p>
            ) : null}
          </div>
        </Card>
      )}

      <Card>
        <div className="py-2 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="text-xs text-text-dim uppercase tracking-wider">
              Talk
            </div>
            <span className="text-[12px] text-text-dim">
              {session.phase === "listening"
                ? "Listening…"
                : session.phase === "thinking"
                  ? "Thinking…"
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
            variant={session.phase === "listening" ? "primary" : "secondary"}
            disabled={!ready || session.phase === "thinking" || session.phase === "speaking"}
            onPointerDown={(e) => {
              e.preventDefault();
              if (!ready) return;
              void session.startTalking();
            }}
            onPointerUp={() => session.stopTalking()}
            onPointerLeave={() => session.stopTalking()}
            onPointerCancel={() => session.stopTalking()}
            onContextMenu={(e) => e.preventDefault()}
            style={{ touchAction: "none" }}
          >
            {session.phase === "listening" ? "Release to send" : "Hold to talk"}
          </Button>
          {session.reply ? (
            <p className="text-[14px] leading-relaxed whitespace-pre-wrap">
              {session.reply}
            </p>
          ) : (
            <p className="text-[13px] text-text-dim leading-relaxed">
              {ready
                ? "Hold the button and speak. The robot replies with speech."
                : building
                  ? "Waiting for the engines to finish building…"
                  : failed
                    ? "The voice service reported an error — see the status above."
                    : "Start the voice service above, then hold to talk."}
            </p>
          )}
        </div>
      </Card>

      <div className="mt-auto pt-4">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
      </div>
    </div>
  );
}
