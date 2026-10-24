// Autonomy: talk to the robot's agent (`bebop-voice.service`, :9093).
//
// Service lifecycle (start/stop) goes over the runtime WS; audio and the
// conversation go over the service's own socket. Engine, voice and persona
// live in Settings › Voice. The global E-STOP in the shell stops any motion
// a tool starts. The text console, transcript and tool cards come with the
// Autonomy redesign; this keeps push-to-talk working inside the shell.

import { useEffect, useState } from "react";
import { Link } from "react-router";

import { Banner, Button, Card, CardHeader, cn, StatusDot } from "../components/ui";
import type { Tone } from "../components/ui";
import { robotPath } from "../app/paths";
import { useRobot, useRuntimeAction, useTelemetry } from "../robot/session";
import { useVoiceConfig, useVoiceHealth, useVoiceSession } from "../voice/useVoiceSession";

const SESSION_KEY = "bebop.voice.session";

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

export function VoiceScreen() {
  const { host } = useRobot();
  const voice = useTelemetry(
    (s) => ({
      present: !!s?.voice.present,
      running: !!s?.voice.running,
      state: s?.voice.state ?? "",
      detail: s?.voice.detail ?? "",
      model: s?.voice.model ?? "",
      loaded: !!s,
    }),
    5,
  );
  const { busy, error, run } = useRuntimeAction();
  const [sessionId, setSessionId] = useState<string>(() => loadSessionId());

  const health = useVoiceHealth(host, voice.running);
  const cfg = useVoiceConfig(host, voice.running);
  const session = useVoiceSession(host, { session: sessionId });

  useEffect(() => {
    try {
      localStorage.setItem(SESSION_KEY, sessionId);
    } catch {
      /* ignore */
    }
  }, [sessionId]);

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
  const stale = Boolean(building && health?.heartbeatAgeS != null && health.heartbeatAgeS > 10);
  const tone: Tone = ready ? "ok" : failed || stale ? "err" : building ? "info" : "dim";
  const talkActive = session.phase === "listening";
  const engine = cfg.config?.backend ?? "";

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-4 px-4 pt-4 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold">Autonomy</h1>
          <p className="text-body-sm text-text-dim">
            Talk to the robot. It can look around, remember places and make
            small moves; E-STOP stops it any time.
          </p>
        </div>
        <Link
          to={robotPath(host, "settings")}
          className="inline-flex h-control-sm items-center rounded-card border border-border px-3 text-body-sm font-semibold text-text-dim hover:bg-bg-elev-2 hover:text-text pointer-coarse:h-touch"
        >
          Voice settings
        </Link>
      </div>

      {error ? <Banner tone="error">{error}</Banner> : null}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        {!voice.loaded ? (
          <Card>
            <p className="text-sm text-text-dim">Waiting for telemetry…</p>
          </Card>
        ) : !voice.present ? (
          <Banner tone="info">
            Voice isn&apos;t installed on this robot. Deploy the{" "}
            <code>bebop-voice.service</code> unit (see{" "}
            <code>scripts/install-jetson.sh</code>), then reconnect.
          </Banner>
        ) : (
          <Card className="flex flex-col gap-3">
            <CardHeader
              eyebrow="Agent service"
              title={
                <span className="inline-flex items-center gap-2">
                  <StatusDot tone={tone} pulse={building} label={statusText} />
                  {statusText}
                </span>
              }
            />
            {voice.model ? (
              <div className="font-mono text-body-sm text-text-dim">
                {voice.model}
                {health?.precision ? ` · ${health.precision}` : ""}
                {engine ? ` · ${engine}` : ""}
              </div>
            ) : null}
            {statusDetail ? (
              <div className={cn("text-label leading-snug", failed ? "text-danger-fg" : "text-text-dim")}>
                {statusDetail}
              </div>
            ) : null}
            {engine === "omni" ? (
              <Banner tone="warning">
                The omni engine has no tools: the robot can talk but not check
                places or move. Switch to cascade in Voice settings.
              </Banner>
            ) : null}
            {stale ? (
              <Banner tone="error">
                No heartbeat for {Math.round(health?.heartbeatAgeS ?? 0)}s; the
                build may be stuck. Check <code>journalctl -u bebop-voice</code>.
              </Banner>
            ) : building && health && health.componentsDone.length > 0 ? (
              <div className="text-caption leading-snug text-text-dim">
                built: {health.componentsDone.join(", ")}
                {health.component ? ` · now: ${health.component}` : ""}
              </div>
            ) : null}
            <Button
              variant={voice.running ? "secondary" : "success"}
              loading={busy === "voice"}
              onClick={() => {
                const next = !voice.running;
                void run("voice", (t) => t.setVoiceEnabled(next));
              }}
            >
              {voice.running ? "Stop voice service" : "Start voice service"}
            </Button>
            {!voice.running ? (
              <p className="text-label leading-relaxed text-text-dim">
                The first start builds the model&apos;s TensorRT engines and can
                take several minutes.
              </p>
            ) : null}
          </Card>
        )}

        <Card className="flex flex-col gap-3">
          <CardHeader
            eyebrow="Talk"
            actions={
              <>
                <span className="flex items-center gap-2 text-label text-text-dim">
                  {talkActive ? (
                    <span className="h-2 w-2 animate-pulse rounded-full bg-danger" />
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
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!ready}
                  onClick={() => {
                    setSessionId(makeSessionId());
                    session.resetContext();
                  }}
                >
                  New conversation
                </Button>
              </>
            }
          />
          {session.error ? <Banner tone="error">{session.error}</Banner> : null}
          <Button
            variant={talkActive ? "primary" : "secondary"}
            size="lg"
            disabled={!ready || session.phase === "thinking" || session.phase === "speaking"}
            onClick={() => (talkActive ? session.stopTalking() : void session.startTalking())}
            onContextMenu={(e) => e.preventDefault()}
            className="h-20 w-full touch-none select-none text-base"
          >
            {talkActive ? "Tap to send" : "Tap to talk"}
          </Button>
          {session.motion || session.phase === "thinking" || session.phase === "speaking" ? (
            <Button
              variant={session.motion ? "danger" : "secondary"}
              size="lg"
              onClick={() => session.cancel()}
              className="w-full"
            >
              {session.motion
                ? `Cancel move (${session.motion.totalM ? `${session.motion.doneM.toFixed(2)} / ${session.motion.totalM.toFixed(2)} m` : `${Math.round(session.motion.doneDeg)} / ${Math.round(session.motion.totalDeg)}°`})`
                : "Cancel"}
            </Button>
          ) : null}
          {session.busy ? (
            <Banner tone="info">The robot is still on the previous request. Cancel it first.</Banner>
          ) : null}
          {session.reply ? (
            <p className="max-h-72 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed">
              {session.reply}
            </p>
          ) : (
            <p className="text-body-sm leading-relaxed text-text-dim">
              {ready
                ? "Tap, speak, then tap again to send. The robot replies with speech."
                : building
                  ? "Waiting for the engines to finish building…"
                  : failed
                    ? "The voice service reported an error; see its status."
                    : "Start the voice service, then tap to talk."}
            </p>
          )}
          <p className="text-caption leading-snug text-text-dim">
            Microphone access needs a secure page (HTTPS) or the native app.
          </p>
        </Card>
      </div>
    </div>
  );
}
