// Autonomy console (`#/r/:host/autonomy`): talk to or type at the robot's
// agent (`bebop-voice.service`, :9093) and watch what it does.
//
//   desktop / tablet landscape (lg+): two columns — camera, robot status and
//     a big Stop task on the left; transcript and composer on the right.
//   phone / tablet portrait: transcript full width under a collapsible
//     camera header; the composer is pinned to the bottom, above the tabs.
//
// Service lifecycle (start/stop) goes over the runtime WS; the conversation
// goes over the agent's own socket. Engine, voice and permissions live in
// Settings › Voice. The global E-STOP stops any motion a tool starts; Stop
// task cancels the running turn (which also stops a voice-driven move).

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { Link } from "react-router";

import { Banner, Button, Card, cn, Spinner, StatusPill } from "../../components/ui";
import type { Tone } from "../../components/ui";
import { SettingsIcon } from "../../app/icons";
import { robotPath } from "../../app/paths";
import { useRobot, useRuntimeAction, useTelemetry } from "../../robot/session";
import { motionFraction } from "../../voice/agentTurns";
import type { ToolItem } from "../../voice/agentTurns";
import { useAgentSession } from "../../voice/useAgentSession";
import { useVoiceIO } from "../../voice/useVoiceIO";
import { CASCADE_STAGES, engineLabel, useVoiceHealth } from "../../voice/voiceService";
import type { VoiceHealth } from "../../voice/voiceService";
import { Composer } from "./Composer";
import { CameraIcon, ChevronIcon, MoveIcon, StopIcon } from "./icons";
import { CameraTile, StatusGrid, useRobotStatus } from "./RobotStatus";
import { Transcript } from "./Transcript";

const SPEAK_KEY = "bebop.agent.speak";
/// Two-column layout (camera + status on the left). Matches Tailwind `lg`.
const WIDE_QUERY = "(min-width: 1024px)";
/// Roomy enough to open the camera header by default: a tablet, not a phone
/// (portrait or held sideways).
const ROOMY = () =>
  typeof window !== "undefined" && window.innerWidth >= 640 && window.innerHeight >= 500;
// Per form factor: a phone wants the camera folded, a tablet open.
const cameraKey = () => `bebop.agent.camera.${ROOMY() ? "wide" : "narrow"}`;

function useMedia(query: string): boolean {
  const mql = useMemo(() => window.matchMedia(query), [query]);
  return useSyncExternalStore(
    (cb) => {
      mql.addEventListener("change", cb);
      return () => mql.removeEventListener("change", cb);
    },
    () => mql.matches,
  );
}

const SUGGESTIONS: { text: string; motion?: boolean }[] = [
  { text: "Where are you?" },
  { text: "What places do you know?" },
  { text: "Turn left 90 degrees", motion: true },
  { text: "Move forward half a meter", motion: true },
];

function readBool(key: string, fallback: boolean): boolean {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : v === "1";
  } catch {
    return fallback;
  }
}

function writeBool(key: string, v: boolean): void {
  try {
    localStorage.setItem(key, v ? "1" : "0");
  } catch {
    /* ignore */
  }
}

function formatElapsed(s: number): string {
  if (!s || s < 0) return "";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

// ---------------------------------------------------------------- engine

interface EngineInfo {
  tone: Tone;
  label: string;
  detail: string;
  ready: boolean;
  building: boolean;
}

function engineInfo(running: boolean, present: boolean, state: string, h: VoiceHealth | null): EngineInfo {
  if (!present) return { tone: "dim", label: "Not installed", detail: "", ready: false, building: false };
  if (!running) return { tone: "dim", label: "Off", detail: state && state !== "inactive" ? state : "", ready: false, building: false };
  if (!h) return { tone: "warn", label: "Starting", detail: "Waiting for the agent service…", ready: false, building: true };
  const building = h.phase === "starting" || h.phase === "building";
  if (h.ok && (h.phase === "ready" || h.phase === "stub")) {
    return {
      tone: "ok",
      label: h.phase === "stub" ? "Ready (stub)" : "Ready",
      detail: h.detail,
      ready: true,
      building: false,
    };
  }
  if (building) {
    return {
      tone: "info",
      label: "Building engines",
      detail: `${h.detail}${h.elapsedS ? ` · ${formatElapsed(h.elapsedS)}` : ""}`,
      ready: false,
      building: true,
    };
  }
  return { tone: "err", label: h.phase === "error" ? "Error" : h.phase || "Unavailable", detail: h.detail, ready: false, building: false };
}

function EngineHeader({
  host,
  health,
  info,
  linkLabel,
}: {
  host: string;
  health: VoiceHealth | null;
  info: EngineInfo;
  linkLabel: string | null;
}) {
  const backend = health?.backend || health?.model || "";
  const cascade = backend === "cascade";
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-lg font-semibold">Autonomy</h1>
          <StatusPill tone={info.tone} dot pulse={info.building} title={info.detail || undefined}>
            {backend && info.ready ? `${engineLabel(backend)} · ${info.label}` : info.label}
          </StatusPill>
          {health && info.ready ? (
            <>
              <StatusPill
                className="max-sm:hidden"
                tone={health.toolsSupported && health.toolsEnabled ? "dim" : "warn"}
                title={
                  !health.toolsSupported
                    ? "This engine has no tools: chat only."
                    : health.toolsEnabled
                      ? "Places, robot status and live camera context are on."
                      : "Tools are off in Settings › Voice."
                }
              >
                {health.toolsSupported ? (health.toolsEnabled ? "Tools on" : "Tools off") : "Chat only"}
              </StatusPill>
              {health.toolsSupported ? (
                <StatusPill
                  tone={health.motionEnabled && health.toolsEnabled ? "info" : "dim"}
                  title={
                    health.motionEnabled
                      ? "The robot may make small bounded moves when asked."
                      : "Moves are off in Settings › Voice."
                  }
                >
                  {health.motionEnabled && health.toolsEnabled ? "Moves on" : "Moves off"}
                </StatusPill>
              ) : null}
            </>
          ) : null}
          {linkLabel ? (
            <StatusPill tone="warn" dot pulse>
              {linkLabel}
            </StatusPill>
          ) : null}
        </div>
        {cascade && info.ready ? (
          <p className="mt-1 hidden text-caption text-text-dim lg:block">
            {CASCADE_STAGES.map((s) => `${s.stage} ${s.model}`).join("  ·  ")}
          </p>
        ) : null}
      </div>
      <Link
        to={robotPath(host, "settings/voice")}
        aria-label="Voice settings"
        title="Voice settings"
        className="inline-flex h-control-sm shrink-0 items-center gap-2 rounded-card border border-border px-3 text-label font-semibold text-text-dim hover:bg-bg-elev-2 hover:text-text pointer-coarse:h-touch max-sm:w-touch max-sm:justify-center max-sm:px-0"
      >
        <SettingsIcon />
        <span className="max-sm:sr-only">Voice settings</span>
      </Link>
    </div>
  );
}

// ---------------------------------------------------------------- stop

function motionText(m: ToolItem | null): string {
  const p = m?.progress;
  if (!p) return "";
  const parts: string[] = [];
  if (p.totalM) parts.push(`${Math.abs(p.doneM).toFixed(2)} of ${Math.abs(p.totalM).toFixed(2)} m`);
  if (p.totalDeg) parts.push(`${Math.round(Math.abs(p.doneDeg))} of ${Math.round(Math.abs(p.totalDeg))}°`);
  return parts.join(", ");
}

/// Big Stop task button. Solid red while the robot is moving, outlined
/// while it is only thinking or talking. Distinct from the global E-STOP:
/// this cancels the agent's task, E-STOP latches the motors.
function StopTask({
  active,
  motion,
  stopping,
  onStop,
  className,
}: {
  active: boolean;
  motion: ToolItem | null;
  stopping: boolean;
  onStop: () => void;
  className?: string;
}) {
  if (!active) return null;
  const moving = !!motion;
  return (
    <button
      type="button"
      onClick={onStop}
      className={cn(
        "relative flex min-h-14 w-full cursor-pointer items-center gap-3 overflow-hidden rounded-card border px-4 py-2 text-left transition-colors active:scale-[0.99]",
        moving
          ? "border-danger bg-danger text-white hover:bg-danger-press"
          : "border-danger/50 bg-danger/10 text-danger-fg hover:bg-danger/20",
        className,
      )}
    >
      {moving && motion?.progress ? (
        <span className="absolute inset-x-0 bottom-0 h-1 bg-black/20" aria-hidden>
          <span
            className="block h-full bg-white/80 transition-[width] duration-200"
            style={{ width: `${motionFraction(motion.progress) * 100}%` }}
          />
        </span>
      ) : null}
      <span className="relative inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-current/15">
        {stopping ? <Spinner /> : moving ? <MoveIcon size={16} /> : <StopIcon size={14} />}
      </span>
      <span className="relative min-w-0 flex-1">
        <span className="block text-base font-bold leading-tight">{stopping ? "Stopping…" : "Stop task"}</span>
        <span className={cn("block truncate text-label", moving ? "text-white/85" : "text-danger-fg/80")}>
          {moving ? `Moving · ${motionText(motion)}` : "Cancels what the robot is doing (Esc)"}
        </span>
      </span>
    </button>
  );
}

// ---------------------------------------------------------------- empty

function EmptyState({
  onPick,
  disabled,
  motionBlocked,
}: {
  onPick: (text: string) => void;
  disabled: boolean;
  motionBlocked: string | null;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-4 py-5 text-center sm:min-h-56 sm:gap-4 sm:px-5 sm:py-8 [@media(max-height:500px)]:min-h-0 [@media(max-height:500px)]:py-3">
      <div
        className="hidden h-12 w-12 items-center justify-center sm:flex [@media(max-height:500px)]:hidden rounded-full border border-accent/40 bg-accent/10 text-lg font-bold text-accent"
        aria-hidden
      >
        B
      </div>
      <div className="[@media(max-height:500px)]:hidden">
        <h2 className="text-base font-semibold">Ask Bebop</h2>
        <p className="mx-auto mt-1 max-w-sm text-body-sm leading-relaxed text-text-dim">
          It sees through its cameras, knows the places it remembers, reports its own
          status and can make small moves. Type or hold the mic.
        </p>
      </div>
      <div className="flex max-w-md flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((s) => {
          const blocked = disabled || (s.motion && !!motionBlocked);
          return (
            <button
              key={s.text}
              type="button"
              disabled={blocked}
              title={s.motion && motionBlocked ? motionBlocked : s.motion ? "The robot will move" : undefined}
              onClick={() => onPick(s.text)}
              className="inline-flex min-h-touch cursor-pointer items-center gap-1.5 rounded-full border border-border bg-bg-elev-2 px-3.5 text-body-sm font-semibold text-text transition-colors hover:border-accent/60 hover:bg-accent/10 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {s.motion ? (
                <span className="text-warning" aria-label="moves the robot">
                  <MoveIcon size={14} />
                </span>
              ) : null}
              {s.text}
            </button>
          );
        })}
      </div>
      {motionBlocked ? <p className="max-w-sm text-caption text-text-dim [@media(max-height:500px)]:hidden">{motionBlocked}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- service off

function ServiceCard({
  info,
  present,
  running,
  health,
  busy,
  onToggle,
}: {
  info: EngineInfo;
  present: boolean;
  running: boolean;
  health: VoiceHealth | null;
  busy: boolean;
  onToggle: () => void;
}) {
  if (!present) {
    return (
      <div className="flex h-full items-center justify-center p-5">
        <Banner tone="info" title="The agent isn't installed">
          Deploy <code>bebop-voice.service</code> (see <code>scripts/install-jetson.sh --setup-voice</code>), then reconnect.
        </Banner>
      </div>
    );
  }
  const stale = info.building && health?.heartbeatAgeS != null && health.heartbeatAgeS > 30 && health.phase === "building";
  return (
    <div className="flex h-full min-h-56 flex-col items-center justify-center gap-3 px-5 py-8 text-center">
      {info.building ? <Spinner large /> : null}
      <div>
        <h2 className="text-base font-semibold">
          {running ? (info.building ? "The agent is starting" : `Agent: ${info.label}`) : "The agent is off"}
        </h2>
        <p className="mx-auto mt-1 max-w-sm text-body-sm leading-relaxed text-text-dim">
          {running
            ? info.detail || "Waiting for the service to answer."
            : "Start the voice and agent service to talk to the robot. The first start builds its engines and can take several minutes."}
        </p>
        {health && health.componentsDone.length > 0 && info.building ? (
          <p className="mt-1 text-caption text-text-dim">
            built: {health.componentsDone.join(", ")}
            {health.component ? ` · now: ${health.component}` : ""}
          </p>
        ) : null}
      </div>
      {stale ? (
        <Banner tone="error">
          No heartbeat for {Math.round(health?.heartbeatAgeS ?? 0)} s; the build may be stuck. Check{" "}
          <code>journalctl -u bebop-voice</code>.
        </Banner>
      ) : null}
      <Button variant={running ? "secondary" : "success"} loading={busy} onClick={onToggle}>
        {running ? "Stop agent service" : "Start agent service"}
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------- screen

export function AutonomyScreen() {
  const { host } = useRobot();
  const svc = useTelemetry(
    (s) => ({
      loaded: !!s,
      present: !!s?.voice.present,
      running: !!s?.voice.running,
      state: s?.voice.state ?? "",
    }),
    2,
  );
  const action = useRuntimeAction();
  // Before the first telemetry snapshot (or while the runtime WS
  // reconnects) trust /healthz alone rather than reading "off".
  const health = useVoiceHealth(host, svc.running || !svc.loaded);
  const running = svc.loaded ? svc.running : !!health;
  const info = engineInfo(running, svc.present || !svc.loaded, svc.state, health);
  // The agent socket follows the service, not each health poll: once the
  // engine has been ready it stays up until telemetry says the service
  // stopped (useAgentSession also holds it open while a turn runs).
  const [everReady, setEverReady] = useState(false);
  useEffect(() => {
    if (info.ready) setEverReady(true);
    else if (svc.loaded && !svc.running) setEverReady(false);
  }, [info.ready, svc.loaded, svc.running]);
  const wide = useMedia(WIDE_QUERY);
  const status = useRobotStatus();

  const io = useVoiceIO();
  const [speak, setSpeak] = useState(() => readBool(SPEAK_KEY, true));
  const speakRef = useRef(speak);
  speakRef.current = speak;
  const agent = useAgentSession(host, {
    enabled: info.ready || (everReady && running),
    onAudio: (bytes) => {
      if (speakRef.current) io.playChunk(bytes);
    },
    onCancel: io.stopPlayback,
  });

  const [draft, setDraft] = useState("");
  const [cameraOpen, setCameraOpen] = useState(() => readBool(cameraKey(), ROOMY()));
  useEffect(() => writeBool(SPEAK_KEY, speak), [speak]);
  useEffect(() => writeBool(cameraKey(), cameraOpen), [cameraOpen]);

  // ---- voice turns: capture → utterance_start + PCM frames → utterance_end
  const uttRef = useRef<string | null>(null);
  const preRoll = useRef<Int16Array[]>([]);
  const onMicDown = useCallback(() => {
    io.unlockPlayback();
    uttRef.current = null;
    preRoll.current = [];
    void io
      .startCapture((pcm) => {
        if (uttRef.current) agent.sendAudio(pcm);
        else preRoll.current.push(pcm);
      })
      .then((rate) => {
        const id = agent.beginUtterance(rate, { speak: speakRef.current });
        if (!id) {
          io.stopCapture();
          return;
        }
        uttRef.current = id;
        for (const f of preRoll.current) agent.sendAudio(f);
        preRoll.current = [];
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        if (msg !== "released") action.setError(msg);
      });
  }, [io, agent, action]);

  const onMicUp = useCallback(() => {
    const id = uttRef.current;
    uttRef.current = null;
    io.stopCapture();
    if (id) agent.endUtterance(id);
  }, [io, agent]);

  const stop = useCallback(() => {
    if (uttRef.current) {
      const id = uttRef.current;
      uttRef.current = null;
      io.stopCapture();
      agent.abandonUtterance(id);
    }
    agent.cancel();
  }, [io, agent]);

  const send = useCallback(
    (text: string) => {
      io.unlockPlayback();
      return agent.sendText(text, { speak: speakRef.current });
    },
    [io, agent],
  );

  // The socket dropped mid-utterance: the reducer marks the turn
  // interrupted; stop the mic so nothing keeps listening into the void.
  useEffect(() => {
    if (agent.link !== "open" && uttRef.current) {
      uttRef.current = null;
      io.stopCapture();
    }
  }, [agent.link, io]);

  // Esc stops a running task from anywhere on the page.
  const activeTurn = agent.active;
  useEffect(() => {
    if (!activeTurn) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeTurn, stop]);

  const busy = !!activeTurn && activeTurn.status !== "recording";
  const online = agent.link === "open";
  const linkLabel =
    info.ready && agent.link !== "open"
      ? agent.link === "reconnecting"
        ? "reconnecting"
        : "connecting"
      : null;
  const motionBlocked = !health?.toolsSupported
    ? "This engine can't move the robot."
    : !health.motionEnabled || !health.toolsEnabled
      ? "Moves are off in Settings › Voice."
      : status.s.estop
        ? "E-STOP is latched, so the robot will refuse to move."
        : null;
  const stopping = !!activeTurn?.cancelRequested;
  const showTranscript = info.ready || everReady || agent.state.turns.length > 0;

  const transcript = showTranscript ? (
    <Transcript
      turns={agent.state.turns}
      className="min-h-16 flex-1 [@media(min-height:600px)]:min-h-40"
      empty={
        <EmptyState
          onPick={(t) => send(t)}
          disabled={!online || busy}
          motionBlocked={motionBlocked}
        />
      }
    />
  ) : (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <ServiceCard
        info={info}
        present={svc.present || !svc.loaded}
        running={svc.running}
        health={health}
        busy={action.busy === "voice"}
        onToggle={() => {
          const next = !svc.running;
          void action.run("voice", (t) => t.setVoiceEnabled(next));
        }}
      />
    </div>
  );

  const composer = (
    <Composer
      online={online}
      busy={busy}
      listening={io.capturing}
      micBlocked={io.micBlocked}
      speak={speak}
      onSpeakChange={setSpeak}
      onSend={send}
      onStop={stop}
      onMicDown={onMicDown}
      onMicUp={onMicUp}
      onNewConversation={() => {
        agent.newConversation();
        setDraft("");
      }}
      canReset={agent.state.turns.length > 0}
      draft={draft}
      onDraftChange={setDraft}
      placeholder={
        !info.ready && !everReady
          ? "The agent isn't ready yet"
          : online
            ? busy
              ? "Working… (Esc to stop)"
              : "Ask the robot…"
            : "Connecting to the agent…"
      }
    />
  );

  return (
    <div className="mx-auto flex min-h-0 w-full max-w-7xl flex-1 flex-col gap-3 pt-3 sm:px-6 sm:pt-4 lg:gap-4 [@media(max-height:500px)]:gap-2 [@media(max-height:500px)]:pt-2">
      <div className="px-4 sm:px-0">
        <EngineHeader host={host} health={health} info={info} linkLabel={linkLabel} />
      </div>
      {action.error || agent.state.sessionError ? (
        <div className="px-4 sm:px-0">
          <Banner tone="error">{action.error || agent.state.sessionError}</Banner>
        </div>
      ) : null}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[minmax(300px,380px)_minmax(0,1fr)]">
        {/* Left column (lg+): camera, status, stop. */}
        {/* Mounted only when wide: a hidden <video> would still hold a WHEP
            viewer (the video server allows 4 per stream). */}
        {wide ? (
        <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto">
          <Card className="flex flex-col gap-3 p-3">
            <CameraTile running={status.s.visionRunning} maxHeight="34dvh" />
            <StatusGrid status={status} className="px-1 pb-1" />
          </Card>
          {status.s.estop ? (
            <Banner tone="warning" title="E-STOP latched">
              The robot will refuse to move until someone checks it and resets the E-STOP.
            </Banner>
          ) : null}
          <StopTask active={busy} motion={agent.motion} stopping={stopping} onStop={stop} />
          {!busy ? (
            <div className="rounded-card border border-dashed border-border px-4 py-3 text-label leading-relaxed text-text-dim">
              No task running. Stop task appears here while the robot works; the E-STOP button
              always stops the motors.
            </div>
          ) : null}
        </aside>
        ) : null}

        {/* Conversation. Flat on phones, a card from sm up. */}
        <section className="flex min-h-0 flex-col overflow-hidden border-border bg-bg-elev/40 sm:rounded-card sm:border lg:bg-bg-elev">
          {/* Collapsible camera header below lg. */}
          {wide ? null : (
          <div className="border-y border-border bg-bg-elev sm:border-t-0 [@media(max-height:500px)]:hidden">
            <button
              type="button"
              onClick={() => setCameraOpen((v) => !v)}
              aria-expanded={cameraOpen}
              className="flex min-h-touch w-full cursor-pointer items-center gap-3 px-4 py-2 text-left"
            >
              <span className="text-text-dim">
                <CameraIcon size={18} />
              </span>
              <span className="flex min-w-0 flex-1 items-center gap-2 text-body-sm">
                <span className="truncate font-semibold text-text">{status.place.text}</span>
                <span className="shrink-0 text-text-dim">· {status.battery.text}</span>
                {status.s.estop ? (
                  <StatusPill tone="err" className="shrink-0">
                    E-STOP
                  </StatusPill>
                ) : status.s.driver !== "none" ? (
                  <span className="truncate text-text-dim">· {status.driving.text} driving</span>
                ) : null}
              </span>
              <span className="shrink-0 text-label font-semibold text-text-dim">
                {cameraOpen ? "Hide" : "Camera"}
              </span>
              <ChevronIcon open={cameraOpen} className="shrink-0 text-text-dim" />
            </button>
            {cameraOpen ? (
              <div className="grid gap-3 px-4 pb-3 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] md:items-center">
                <CameraTile running={status.s.visionRunning} maxHeight="24dvh" />
                <StatusGrid status={status} className="hidden md:grid" />
              </div>
            ) : null}
          </div>
          )}

          {transcript}

          <div className="flex flex-col gap-2 border-t border-border bg-bg-elev px-3 pb-2 pt-3 sm:px-4">
            <StopTask
              active={busy}
              motion={agent.motion}
              stopping={stopping}
              onStop={stop}
              className={wide ? "hidden" : undefined}
            />
            {composer}
          </div>
        </section>
      </div>
    </div>
  );
}
