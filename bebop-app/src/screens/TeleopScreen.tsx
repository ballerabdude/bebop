// Drive page: live video + driving in one place.
//
// Runs inside the robot shell, which owns the connection, the status bar
// and the global E-STOP; this page composes the video tiles
// (`VideoFeed`, WHEP), the drive inputs (`useDrive`: on-screen joystick,
// WASD / arrows, gamepad behind the RT deadman) and a quick-start that
// switches to the drive mode and arms the wheels in one tap.
//
//   * Phones open in the immersive layout (feed fills the screen, pad
//     floats over it); tablets and desktops get the page layout with a
//     Fullscreen button. Esc leaves fullscreen.
//   * Fullscreen is a class flip, not a second tree: the page chrome stays
//     mounted (hidden) so the gamepad card and the stream survive. Only the
//     joystick moves between the page and the overlay, and it is never
//     mounted twice.
//   * Video playback can be turned off (persisted); the robot keeps
//     recording and the drive link stays live.
//
// Stops: every lifecycle stop (blur, hidden tab, link loss, unmount) lives
// in the twist stream + arbiter (`useTwistStream`, `useDriveInput`).

import { useCallback, useEffect, useState } from "react";

import { ControlProfilePicker } from "../components/ControlProfilePicker";
import { DriveJoystick } from "../components/DriveJoystick";
import { GamepadDrive } from "../components/GamepadDrive";
import { VideoFeed } from "../components/VideoFeed";
import type { VideoStreamState } from "../components/VideoFeed";
import {
  Banner,
  Button,
  Card,
  CardHeader,
  cn,
  StatusPill,
  Switch,
} from "../components/ui";
import { useGamepad } from "../input";
import { VIDEO_STREAMS } from "../robot/endpoints";
import { useDrive } from "../robot/useDrive";
import {
  useConnection,
  useEStop,
  useRobot,
  useRuntimeAction,
  useTelemetry,
} from "../robot/session";
import type { RuntimeSnapshot, WheelView } from "../runtime";

/// localStorage key for the video-playback toggle.
const VIDEO_ENABLED_KEY = "bebop.teleop.videoEnabled";

const fmt = (v: number): string => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2));

const select = (s: RuntimeSnapshot | null) => s;

function readVideoEnabled(): boolean {
  try {
    return window.localStorage.getItem(VIDEO_ENABLED_KEY) !== "0";
  } catch {
    return true;
  }
}

export function TeleopScreen() {
  const { endpoints } = useRobot();
  const link = useConnection();
  const snapshot = useTelemetry(select, 30);
  const estop = useEStop();
  const { busy, error, setError, run } = useRuntimeAction();

  const [reconnectKey, setReconnectKey] = useState(0);
  const [videoStreams, setVideoStreams] = useState<string[]>(["color_near"]);
  const [primaryId, setPrimaryId] = useState("color_near");
  const primaryStream = videoStreams.includes(primaryId) ? primaryId : videoStreams[0];
  const secondaryStreams = videoStreams.filter((s) => s !== primaryStream);
  const toggleStream = (id: string) => {
    if (videoStreams.includes(id)) {
      if (videoStreams.length <= 1) return; // never close the last tile
      const next = videoStreams.filter((s) => s !== id);
      setVideoStreams(next);
      if (primaryStream === id) setPrimaryId(next[0]);
    } else {
      setVideoStreams([...videoStreams, id]);
    }
  };
  const [streamState, setStreamState] = useState<VideoStreamState>("loading");
  const [videoEnabled, setVideoEnabledState] = useState(readVideoEnabled);
  const [fullscreen, setFullscreen] = useState(() =>
    typeof window !== "undefined"
      ? videoEnabled && window.matchMedia("(max-width: 767px)").matches
      : false,
  );
  const setVideoEnabled = useCallback((enabled: boolean) => {
    setVideoEnabledState(enabled);
    try {
      window.localStorage.setItem(VIDEO_ENABLED_KEY, enabled ? "1" : "0");
    } catch {
      /* in-memory toggle still works */
    }
    if (!enabled) setFullscreen(false);
  }, []);
  const { connected: padConnected } = useGamepad();

  // -------------------------------------------------------------- derived
  const motors = snapshot?.motors ?? [];
  const wheels = snapshot?.wheels ?? [];
  const drive = snapshot?.drive;
  const mode = snapshot?.mode ?? "UNSPECIFIED";
  const estopLatched = estop.latched;
  const vision = snapshot?.vision;
  const armedWheelCount = wheels.filter((w) => w.armed).length;
  const wheeled = !!drive?.present && motors.length === 0;
  const driveMode = mode === "DIAL_IN" || mode === "RUN_POLICY";
  const otherDriving = !!drive?.hasActiveOperator && !drive?.youAreActiveOperator;
  // A spectator can't drive while someone else holds the seat; the
  // firmware would reject the twists anyway.
  const canDrive =
    wheeled && !estopLatched && driveMode && armedWheelCount > 0 && !otherDriving;

  const input = useDrive({ canDrive, onError: setError });

  const driveBlockedReason = !wheeled
    ? null
    : estopLatched
      ? "E-STOP latched. Reset it (top right) to drive."
      : !input.linkOk
        ? "Runtime link is down or degraded. Driving resumes when telemetry flows again."
        : !driveMode
          ? "Switch to the drive mode to drive."
          : armedWheelCount === 0
            ? "Enable the wheels to drive."
            : otherDriving
              ? `${drive?.activeOperatorName ? `"${drive.activeOperatorName}"` : "Another operator"} is driving. You can watch; driving unlocks once their input goes quiet for a couple of seconds.`
              : null;
  const quickStartable =
    wheeled && !estopLatched && input.linkOk && (!driveMode || armedWheelCount === 0);

  const startDriving = useCallback(() => {
    void run("start-driving", async (t) => {
      if (!driveMode) await t.setMode("DIAL_IN");
      if (armedWheelCount === 0) await t.setAllWheelsEnabled(true);
    });
  }, [run, driveMode, armedWheelCount]);

  const setAllWheels = useCallback(
    (enabled: boolean) => void run(`all-wheels:${enabled}`, (t) => t.setAllWheelsEnabled(enabled)),
    [run],
  );

  // Esc exits fullscreen.
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  // -------------------------------------------------------------- pieces
  const statusPills = (
    <>
      {wheeled ? (
        <StatusPill tone={armedWheelCount === 0 ? "warn" : "ok"}>
          {armedWheelCount}/{wheels.length} wheels
        </StatusPill>
      ) : null}
      {wheeled && drive?.operatorStale ? (
        <StatusPill tone="warn">link stale, halted</StatusPill>
      ) : null}
      {input.needsNeutral && canDrive ? (
        <StatusPill tone="info">release input to drive</StatusPill>
      ) : null}
      {link.state !== "connected" ? (
        <StatusPill tone="warn" dot pulse>
          {link.state}
        </StatusPill>
      ) : null}
    </>
  );

  const drivePad = (
    <DriveJoystick
      onTwist={input.touch.onTwist}
      onStop={input.touch.onStop}
      disabled={!canDrive}
    />
  );

  const streamPicker = (
    <div className="absolute right-2 top-2 z-10 flex max-w-[80%] flex-wrap justify-end gap-1">
      {VIDEO_STREAMS.map((o) => (
        <button
          key={o.id}
          type="button"
          aria-pressed={videoStreams.includes(o.id)}
          onClick={() => toggleStream(o.id)}
          className={cn(
            "h-8 rounded-input px-2.5 text-caption font-semibold backdrop-blur-sm transition-colors pointer-coarse:h-touch",
            videoStreams.includes(o.id)
              ? "bg-white/85 text-black"
              : "bg-black/55 text-white/85 hover:bg-black/70",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );

  const feedProps = (id: string) => ({
    whepUrl: endpoints.whep(id),
    whepResource: endpoints.whepResource,
    reconnectKey,
  });

  const streamThumb = (id: string, sizeClasses: string) => {
    const opt = VIDEO_STREAMS.find((o) => o.id === id)!;
    return (
      <div
        key={id}
        className={cn(
          "relative aspect-video shrink-0 overflow-hidden rounded-card border border-border bg-black hover:border-white/40",
          sizeClasses,
        )}
      >
        <VideoFeed {...feedProps(id)} className="h-full w-full">
          <button
            type="button"
            onClick={() => setPrimaryId(id)}
            className="absolute inset-0 z-10 cursor-pointer"
            aria-label={`Show ${opt.label} in the main view`}
          />
          <button
            type="button"
            onClick={() => toggleStream(id)}
            className="absolute right-1 top-1 z-20 flex h-8 w-8 items-center justify-center rounded-input bg-black/60 pointer-coarse:h-touch pointer-coarse:w-touch text-caption text-white/85 hover:bg-black/80"
            aria-label={`Close ${opt.label}`}
          >
            ✕
          </button>
          <span className="pointer-events-none absolute bottom-1 left-1 z-10 rounded bg-black/60 px-1.5 py-0.5 text-micro font-semibold text-white/85">
            {opt.label}
          </span>
        </VideoFeed>
      </div>
    );
  };

  // -------------------------------------------------------------- render
  return (
    <div
      className={
        fullscreen
          ? "fixed inset-0 z-[35] flex flex-col bg-black pt-safe"
          : "mx-auto flex w-full max-w-6xl flex-col gap-3 px-0 pt-0 sm:px-6 sm:pt-4"
      }
    >
      {fullscreen ? (
        <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-white/10 px-3 py-2 md:pr-[240px]">
          <Button variant="secondary" size="sm" onClick={() => setFullscreen(false)}>
            Exit fullscreen
          </Button>
          <div className="flex min-w-0 flex-wrap items-center gap-1.5">{statusPills}</div>
          <div className="flex-1" aria-hidden />
          {quickStartable ? (
            <Button
              variant="success"
              size="sm"
              onClick={startDriving}
              loading={busy === "start-driving"}
              disabled={!!busy}
            >
              Start driving
            </Button>
          ) : null}
        </div>
      ) : null}

      {/* Page chrome: hidden (not unmounted) in fullscreen. */}
      <div className={fullscreen ? "hidden" : "contents"}>
        <div className="flex flex-wrap items-center gap-2 px-4 pt-3 sm:px-0 sm:pt-0">
          <h1 className="mr-1 text-lg font-semibold">Drive</h1>
          <div className="flex flex-wrap items-center gap-1.5">{statusPills}</div>
          <div className="flex-1" aria-hidden />
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="secondary" size="sm" onClick={() => setVideoEnabled(!videoEnabled)}>
              {videoEnabled ? "Video off" : "Video on"}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setFullscreen(true)}
              disabled={!videoEnabled}
            >
              Fullscreen
            </Button>
          </div>
        </div>
        {error ? (
          <div className="px-4 sm:px-0">
            <Banner tone="error">{error}</Banner>
          </div>
        ) : null}
      </div>

      {/* Live feed. */}
      <div className={fullscreen && videoEnabled ? "relative min-h-0 flex-1" : "contents"}>
        {!videoEnabled ? (
          <div className="mx-4 flex flex-col items-center gap-3 rounded-card border border-border bg-bg-elev px-4 py-6 text-center sm:mx-0">
            <span className="text-sm text-text-dim">
              Video playback is off. The robot still records, and the drive
              controls below are live.
            </span>
            <Button variant="secondary" size="sm" onClick={() => setVideoEnabled(true)}>
              Turn on video
            </Button>
          </div>
        ) : fullscreen ? (
          <div className="flex h-full min-h-0 flex-col-reverse gap-1 overflow-hidden sm:flex-row">
            <div className="relative min-h-0 min-w-0 flex-1">
              <VideoFeed
                {...feedProps(primaryStream)}
                onStreamState={setStreamState}
                className="h-full w-full"
              >
                {streamPicker}
              </VideoFeed>
              {wheeled && !padConnected ? (
                <div className="absolute bottom-[max(1rem,env(safe-area-inset-bottom))] left-3 z-10 rounded-card border border-white/10 bg-bg-elev/75 p-2 backdrop-blur-md">
                  {drivePad}
                </div>
              ) : null}
              {streamState === "error" ? (
                <div className="absolute inset-x-0 top-14 z-10 flex justify-center">
                  <Button variant="secondary" size="sm" onClick={() => setReconnectKey((n) => n + 1)}>
                    Reconnect video
                  </Button>
                </div>
              ) : null}
            </div>
            {secondaryStreams.length > 0 ? (
              <div className="flex shrink-0 gap-1 overflow-x-auto p-0.5 sm:w-36 sm:flex-col sm:overflow-x-hidden sm:overflow-y-auto lg:w-44">
                {secondaryStreams.map((id) => streamThumb(id, "w-28 sm:w-auto"))}
              </div>
            ) : null}
          </div>
        ) : (
          <>
            <VideoFeed
              {...feedProps(primaryStream)}
              onStreamState={setStreamState}
              className="w-full sm:rounded-card sm:border sm:border-border"
              maxHeight="72dvh"
            >
              {streamPicker}
            </VideoFeed>
            {secondaryStreams.length > 0 ? (
              <div className="grid grid-cols-2 gap-2 px-4 sm:grid-cols-3 sm:px-0 lg:grid-cols-4">
                {secondaryStreams.map((id) => streamThumb(id, ""))}
              </div>
            ) : null}
          </>
        )}
      </div>

      {/* Content below the feed: hidden (never unmounted) in fullscreen. */}
      <div className={fullscreen ? "hidden" : "contents"}>
        <div className="flex flex-col gap-3 px-4 sm:px-0">
          {videoEnabled && streamState === "error" ? (
            <Banner
              tone="warning"
              action={
                <Button variant="secondary" size="sm" onClick={() => setReconnectKey((n) => n + 1)}>
                  Reconnect video
                </Button>
              }
            >
              Camera stream failed: bebop-vision may be stopped or
              unreachable. Driving still works.
            </Banner>
          ) : null}

          <div className="grid grid-cols-1 items-start gap-3 lg:grid-cols-2">
            {wheeled ? (
              <Card className="space-y-4">
                <CardHeader
                  eyebrow="Drive"
                  title={`${wheels.length} wheel${wheels.length === 1 ? "" : "s"} · ${armedWheelCount} armed`}
                  actions={
                    <>
                      <ControlProfilePicker />
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setAllWheels(false)}
                        disabled={!!busy || armedWheelCount === 0}
                      >
                        Disable wheels
                      </Button>
                    </>
                  }
                />

                {driveBlockedReason ? (
                  <Banner
                    tone={estopLatched ? "error" : "info"}
                    action={
                      quickStartable ? (
                        <Button
                          variant="success"
                          size="sm"
                          onClick={startDriving}
                          loading={busy === "start-driving"}
                          disabled={!!busy}
                        >
                          Start driving
                        </Button>
                      ) : null
                    }
                  >
                    {driveBlockedReason}
                  </Banner>
                ) : (
                  <Banner tone="success" title="Ready to drive">
                    Joystick, WASD / arrows, or a paired gamepad (hold the right
                    trigger as deadman). Releasing stops the robot.
                  </Banner>
                )}

                <div className="grid grid-cols-1 items-center justify-items-center gap-4 sm:grid-cols-[auto_1fr]">
                  {!fullscreen && !padConnected ? drivePad : null}
                  {padConnected ? (
                    <div className="flex max-w-56 flex-col items-center gap-1 px-2 text-center">
                      <span className="text-caption font-semibold uppercase tracking-wider text-text-dim">
                        Controller driving
                      </span>
                      <span className="text-caption leading-snug text-text-dim">
                        Hold RT and use the sticks. The on-screen pad is hidden
                        while a controller is connected.
                      </span>
                    </div>
                  ) : null}

                  <div className="w-full space-y-3">
                    <div className="rounded-card border border-border bg-bg-elev-2/40 px-3 py-2.5">
                      <div className="mb-1.5 text-caption uppercase tracking-wider text-text-dim">
                        Odometry
                      </div>
                      <div className="font-mono text-body-sm leading-relaxed text-text">
                        <div>
                          x {fmt(drive?.odomX ?? 0)} m · y {fmt(drive?.odomY ?? 0)} m · θ{" "}
                          {fmt(((drive?.odomTheta ?? 0) * 180) / Math.PI)}°
                        </div>
                        <div className="mt-0.5 text-caption text-text-dim">
                          cmd vx {fmt(drive?.cmdLinearX ?? 0)} m/s · ω{" "}
                          {fmt(drive?.cmdAngularZ ?? 0)} rad/s
                        </div>
                      </div>
                    </div>
                    <div className="space-y-1 rounded-card border border-border bg-bg-elev-2/40 px-1.5 py-1">
                      {wheels.map((w) => (
                        <WheelChip
                          key={w.name}
                          wheel={w}
                          busy={busy === `wheel:${w.name}`}
                          estopLatched={estopLatched}
                          canArm={driveMode}
                          onToggle={(enabled) =>
                            void run(`wheel:${w.name}`, (t) => t.setWheelEnabled(w.name, enabled))
                          }
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </Card>
            ) : (
              <Card>
                <CardHeader
                  eyebrow="Drive"
                  description={
                    snapshot
                      ? "This robot reports no wheeled drive, so there is nothing to drive from the app."
                      : "Waiting for the robot's first telemetry frame…"
                  }
                />
              </Card>
            )}

            {vision?.present ? (
              <Card className="space-y-3">
                <CardHeader
                  eyebrow="Vision"
                  title={
                    vision.mode
                      ? `${vision.mode[0].toUpperCase()}${vision.mode.slice(1)} service`
                      : "Vision service"
                  }
                  description={
                    vision.running
                      ? "Running. Cameras, recording and place memory are live."
                      : "Stopped. Video and place memory need it."
                  }
                  actions={
                    <StatusPill
                      tone={vision.running ? "ok" : vision.state === "failed" ? "err" : "dim"}
                    >
                      {vision.running ? "running" : vision.state || "stopped"}
                    </StatusPill>
                  }
                />
                {vision.detail && !vision.running ? (
                  <div className="break-words font-mono text-caption text-text-dim">
                    {vision.detail}
                  </div>
                ) : null}
                <Button
                  variant={vision.running ? "secondary" : "success"}
                  size="sm"
                  onClick={() =>
                    void run(`vision:${!vision.running}`, (t) =>
                      t.setVisionEnabled(!vision.running),
                    )
                  }
                  loading={busy === `vision:${!vision.running}`}
                  disabled={!!busy}
                >
                  {vision.running ? "Stop vision" : "Start vision"}
                </Button>
              </Card>
            ) : null}
          </div>

          {wheeled ? (
            <GamepadDrive
              wheels={wheels}
              mode={mode}
              estopLatched={estopLatched}
              onEStop={() => void estop.trigger("gamepad")}
              onResetEStop={() => void estop.reset()}
              onSetAllWheels={setAllWheels}
            />
          ) : null}

          <p className="pb-2 text-center text-caption text-text-dim">
            {padConnected
              ? "Controller: hold RT, left stick drives"
              : "WASD / arrows drive · Space = E-STOP"}
          </p>
        </div>
      </div>
    </div>
  );
}

/// One wheel's arm switch + live velocity.
function WheelChip({
  wheel,
  busy,
  estopLatched,
  canArm,
  onToggle,
}: {
  wheel: WheelView;
  busy: boolean;
  estopLatched: boolean;
  canArm: boolean;
  onToggle: (enabled: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <Switch
        checked={wheel.armed}
        label={`Arm ${wheel.name}`}
        onChange={onToggle}
        disabled={busy || estopLatched || (!wheel.armed && !canArm)}
      />
      <span className="truncate text-label font-medium text-text">{wheel.name}</span>
      {wheel.errorCode !== 0 ? <StatusPill tone="err">error</StatusPill> : null}
      <span className="ml-auto pr-2 font-mono text-caption text-text-dim">
        {fmt(wheel.velocity)} rad/s
      </span>
    </div>
  );
}
