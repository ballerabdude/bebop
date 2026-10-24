// Drive page (`#/r/:host/drive`): video first, everything else around it.
//
// Two layouts over the same tree (switching never remounts the video):
//   * stacked  (portrait phones / tablets): toolbar, the video band, then
//     the HUD, markers and the joystick below it.
//   * overlay  (landscape, desktop): the video fills the page and the
//     toolbar, HUD, markers and joystick float over it.
// The controls drawer (wheels, mode, vision, odometry, gamepad) slides over
// on phones / tablets and docks as a side panel on desktop.
//
// The shell owns the status bar, the tab bar / rail and the global E-STOP
// (a dock above the phone tab bar, top-right from md up or on landscape
// phones); the page lays out between them.
//
// Driving: `useDrive` (twist stream + input arbiter: joystick, WASD /
// arrows, gamepad behind the RT deadman). Every lifecycle stop lives there.
// Drive notes: `useDriveNotes` -> videoserver /notes, /intent, /event.

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { ReactNode } from "react";

import { DriveJoystick } from "../../components/DriveJoystick";
import type { VideoStats } from "../../components/VideoFeed";
import { Button, cn, StatusPill, TONE_DOT } from "../../components/ui";
import type { Tone } from "../../components/ui";
import {
  CONTROL_PROFILES,
  setActiveControlProfile,
  useControlProfile,
  useGamepad,
} from "../../input";
import type { DriveSource } from "../../input";
import { VIDEO_STREAMS } from "../../robot/endpoints";
import {
  useEStop,
  useRobot,
  useRuntimeAction,
  useTelemetry,
} from "../../robot/session";
import { useDrive } from "../../robot/useDrive";
import type { DriveView, RuntimeSnapshot } from "../../runtime";
import { ControlsDrawer } from "./ControlsDrawer";
import {
  EventButtons,
  IntentChip,
  IntentPanel,
  MarkButton,
  RecChip,
  useMarkerHotkeys,
} from "./DriveNotes";
import { createOdomVelocity, driverStatus, signed } from "./hudModel";
import type { DriverStatus, MeasuredTwist } from "./hudModel";
import { GaugeIcon, GridIcon, PipIcon, SlidersIcon } from "./icons";
import { LinkChip, LinkDetails, useTelemetryHealth } from "./LinkChip";
import { PagePanel } from "./PagePanel";
import { ToastView, useToast } from "./Toasts";
import { useDriveNotes } from "./useDriveNotes";
import { VideoStage } from "./VideoStage";

const VIDEO_ENABLED_KEY = "bebop.teleop.videoEnabled";

const MODE_LABEL: Record<string, string> = {
  UNSPECIFIED: "Unknown",
  IDLE: "Idle",
  DIAL_IN: "Drive mode",
};

const SOURCE_LABEL: Record<DriveSource, string> = {
  touch: "Joystick",
  keyboard: "Keyboard",
  gamepad: "Gamepad",
};

function readVideoEnabled(): boolean {
  try {
    return window.localStorage.getItem(VIDEO_ENABLED_KEY) !== "0";
  } catch {
    return true;
  }
}

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

const selectAll = (s: RuntimeSnapshot | null) => s;

/// DriveState.active_operator_autonomous. Firmware older than the PREEMPT
/// change always reports false, so false falls back to the seat holder's
/// name (hudModel's `driverStatus`).
function seatOf(d: DriveView | undefined) {
  if (!d) return null;
  return {
    hasActiveOperator: d.hasActiveOperator,
    youAreActiveOperator: d.youAreActiveOperator,
    activeOperatorName: d.activeOperatorName,
    activeOperatorAutonomous: d.activeOperatorAutonomous ? true : undefined,
  };
}

export function DriveScreen() {
  const { host, endpoints } = useRobot();
  const snapshot = useTelemetry(selectAll, 30);
  const estop = useEStop();
  const { busy, error, setError, run } = useRuntimeAction();
  const { connected: padConnected } = useGamepad();
  const notes = useDriveNotes(host);
  const toast = useToast();

  // ------------------------------------------------------------ layout
  const md = useMedia("(min-width: 768px)");
  const wide = useMedia("(min-width: 1280px)");
  const stacked = useMedia("(orientation: portrait)");
  const short = useMedia("(max-height: 520px)");
  // Small landscape phones (<768 wide): no room for a marker column, so the
  // markers and intent move into a toolbar panel.
  const phoneLand = !md && !stacked;

  // ------------------------------------------------------------ video
  const [main, setMain] = useState("color_near");
  const [pip, setPip] = useState<string | null>(null);
  const [gridPref, setGrid] = useState(false);
  // Tablet / desktop only: four WHEP sessions in a ~300 px band on a
  // landscape phone just burn viewer slots.
  const gridAllowed = md && !short;
  const grid = gridPref && gridAllowed;
  useEffect(() => {
    if (!gridAllowed) setGrid(false);
  }, [gridAllowed]);
  const [reconnectKey, setReconnectKey] = useState(0);
  const reconnectVideo = useCallback(() => setReconnectKey((k) => k + 1), []);
  const closePip = useCallback(() => setPip(null), []);
  const [videoStats, setVideoStats] = useState<VideoStats | null>(null);
  const [videoEnabled, setVideoEnabledState] = useState(readVideoEnabled);
  const setVideoEnabled = useCallback((on: boolean) => {
    setVideoEnabledState(on);
    try {
      window.localStorage.setItem(VIDEO_ENABLED_KEY, on ? "1" : "0");
    } catch {
      /* in-memory toggle still works */
    }
  }, []);
  const enableVideo = useCallback(() => setVideoEnabled(true), [setVideoEnabled]);

  const selectMain = useCallback(
    (id: string) => {
      if (id === pip) setPip(main); // picking the PiP stream swaps the two
      setMain(id);
      setGrid(false);
    },
    [main, pip],
  );
  const swap = useCallback(() => {
    if (!pip) return;
    setPip(main);
    setMain(pip);
  }, [main, pip]);
  // PiP defaults to the same camera's other view (color <-> depth).
  const togglePip = () =>
    setPip((p) => {
      if (p) return null;
      const pair = main.startsWith("color_")
        ? main.replace("color_", "depth_")
        : main.replace("depth_", "color_");
      const ids = VIDEO_STREAMS.map((s) => s.id);
      return ids.includes(pair) && pair !== main ? pair : (ids.find((id) => id !== main) ?? null);
    });

  // ------------------------------------------------------------ drawer / panels
  const [drawerOpen, setDrawerOpen] = useState(false);
  const closeDrawer = useCallback(() => setDrawerOpen(false), []);
  const [panel, setPanel] = useState<"intent" | "link" | "mark" | null>(null);
  const closePanel = useCallback(() => setPanel(null), []);
  const togglePanel = (p: "intent" | "link" | "mark") => setPanel((cur) => (cur === p ? null : p));
  const health = useTelemetryHealth();
  const linkChip = (
    <LinkChip
      health={health}
      video={videoStats}
      videoOn={videoEnabled}
      open={panel === "link"}
      onToggle={() => togglePanel("link")}
      compact={phoneLand}
    />
  );
  const intentChip = (
    <IntentChip notes={notes} open={panel === "intent"} onToggle={() => togglePanel("intent")} />
  );

  // ------------------------------------------------------------ drive
  const wheels = snapshot?.wheels ?? [];
  const drive = snapshot?.drive;
  const mode = snapshot?.mode ?? "UNSPECIFIED";
  const estopLatched = estop.latched;
  const armed = wheels.filter((w) => w.armed).length;
  const wheeled = !!drive?.present && (snapshot?.motors.length ?? 0) === 0;
  const driveMode = mode === "DIAL_IN";
  const driver: DriverStatus = driverStatus(seatOf(drive));
  // A human holding the seat makes this client a spectator; the robot's
  // own agent can be overridden by moving the joystick.
  const canDrive =
    wheeled && !estopLatched && driveMode && armed > 0 && driver.kind !== "human";
  const input = useDrive({ canDrive, onError: setError });

  const quickStartable =
    wheeled &&
    !estopLatched &&
    input.linkOk &&
    driver.kind !== "human" &&
    (!driveMode || armed === 0);
  const startDriving = useCallback(() => {
    void run("start-driving", async (t) => {
      if (!driveMode) await t.setMode("DIAL_IN");
      if (armed === 0) await t.setAllWheelsEnabled(true);
    });
  }, [run, driveMode, armed]);

  // Measured twist from the odometry pose stream.
  const odomEst = useRef(createOdomVelocity());
  const [measured, setMeasured] = useState<MeasuredTwist>({ vx: 0, wz: 0 });
  useEffect(() => {
    if (!drive?.present || !snapshot) return;
    const m = odomEst.current.push({
      tMs: snapshot.hostUnixMs,
      x: drive.odomX,
      y: drive.odomY,
      theta: drive.odomTheta,
    });
    setMeasured((prev) =>
      Math.abs(prev.vx - m.vx) < 0.005 && Math.abs(prev.wz - m.wz) < 0.005 ? prev : m,
    );
  }, [snapshot, drive]);

  // Surface runtime command errors as toasts too (the drawer shows them).
  useEffect(() => {
    if (error) toast.show("err", error, 4000);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [error]);

  // ------------------------------------------------------------ status line
  const status: { tone: Tone; text: string; action?: "start" } | null = !snapshot
    ? { tone: "dim", text: "Waiting for the robot's telemetry…" }
    : !wheeled
      ? { tone: "dim", text: "This robot reports no wheeled drive." }
      : estopLatched
        ? { tone: "err", text: "E-STOP latched — reset it to drive" }
        : !input.linkOk
          ? { tone: "warn", text: "Runtime link down — driving paused" }
          : driver.kind === "autonomous"
            ? { tone: "info", text: driver.hint! }
            : driver.kind === "human"
              ? { tone: "warn", text: driver.hint! }
              : !driveMode
                ? { tone: "info", text: "Idle — switch to drive mode", action: "start" }
                : armed === 0
                  ? { tone: "info", text: "Wheels disarmed", action: "start" }
                  : input.needsNeutral
                    ? { tone: "info", text: "Release the input to drive" }
                    : drive?.operatorStale
                      ? { tone: "warn", text: "Link stale — robot halted" }
                      : null;

  const notify = toast.show;
  useMarkerHotkeys(notes, notify);

  // ------------------------------------------------------------ pieces
  const streamSwitch = (
    <div
      role="radiogroup"
      aria-label="Camera stream"
      className="inline-flex min-w-0 rounded-card border border-white/10 bg-black/60 p-0.5 backdrop-blur-md"
    >
      {VIDEO_STREAMS.map((s) => {
        const active = !grid && s.id === main;
        return (
          <button
            key={s.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => selectMain(s.id)}
            disabled={!videoEnabled}
            className={cn(
              "h-10 min-w-11 whitespace-nowrap rounded-[calc(var(--radius-card)-3px)] px-2.5 pointer-coarse:h-touch text-label font-semibold transition-colors disabled:opacity-50 sm:px-3",
              active ? "bg-white text-black" : "text-white/75 hover:text-white",
              s.id === pip && !grid && "text-info-fg",
            )}
          >
            {s.label}
          </button>
        );
      })}
    </div>
  );

  const iconBtn = (
    label: string,
    onClick: () => void,
    icon: ReactNode,
    pressed?: boolean,
    extra?: string,
  ) => (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      disabled={!videoEnabled && pressed !== undefined}
      className={cn(
        "flex h-touch min-w-touch shrink-0 items-center justify-center gap-2 rounded-card border px-2.5 text-label font-semibold backdrop-blur-md transition-colors disabled:opacity-50",
        pressed
          ? "border-accent/60 bg-accent/25 text-info-fg"
          : "border-white/10 bg-black/60 text-white/85 hover:bg-black/75",
        extra,
      )}
    >
      {icon}
    </button>
  );

  const toolbar = (
    <div className="flex min-w-0 items-center gap-2">
      {streamSwitch}
      {iconBtn(pip ? "Hide picture-in-picture" : "Show picture-in-picture", togglePip, <PipIcon />, !!pip && !grid)}
      {gridAllowed
        ? iconBtn(grid ? "Single view" : "Grid: all four streams", () => setGrid((g) => !g), <GridIcon />, grid)
        : null}
      <div className="flex-1" />
      {phoneLand ? (
        <>
          {linkChip}
          <MarkButton notes={notes} open={panel === "mark"} onToggle={() => togglePanel("mark")} />
        </>
      ) : !stacked ? (
        <>
          {linkChip}
          <RecChip notes={notes} />
        </>
      ) : null}
      {wide && drawerOpen
        ? null
        : iconBtn(
            "Open controls",
            () => setDrawerOpen(true),
            <>
              <SlidersIcon />
              <span className="hidden lg:inline">Controls</span>
            </>,
            undefined,
          )}
    </div>
  );

  const statusLine = status ? (
    <div
      role="status"
      className={cn(
        "flex min-h-touch items-center gap-2.5 rounded-card border bg-bg-elev/92 py-1.5 pl-3 pr-1.5 text-body-sm font-semibold shadow-lg backdrop-blur-md",
        status.tone === "err" && "border-danger/60 text-danger-fg",
        status.tone === "warn" && "border-warning/55 text-warning-fg",
        status.tone === "info" && "border-info/50 text-info-fg",
        status.tone === "dim" && "border-border text-text-dim",
      )}
    >
      <span className={cn("h-2 w-2 shrink-0 rounded-full", TONE_DOT[status.tone])} aria-hidden />
      <span className="min-w-0 flex-1 leading-snug">{status.text}</span>
      {status.action === "start" && quickStartable ? (
        <Button
          variant="success"
          size="sm"
          onClick={startDriving}
          loading={busy === "start-driving"}
          disabled={!!busy}
        >
          Start driving
        </Button>
      ) : (
        <span className="w-1.5" />
      )}
    </div>
  ) : null;

  const battery = snapshot?.power;
  const socKnown = !!battery?.present && battery.statusReceived && battery.stateOfChargePct >= 0;
  const soc = Math.round(battery?.stateOfChargePct ?? -1);

  // Short landscape screens show only the velocities (battery, mode and the
  // seat are in the shell's status bar and the status line).
  const hudCompact = short && !stacked;
  const hud = (
    <div className="flex min-w-0 flex-wrap items-stretch gap-x-4 gap-y-2 rounded-card border border-white/10 bg-black/60 px-3 py-2 backdrop-blur-md">
      <table className="font-mono text-label tabular-nums text-white" aria-label="Velocities">
        <thead>
          <tr className="text-micro uppercase tracking-wider text-white/55">
            <th className="pr-3 text-left font-semibold" />
            <th className="pr-3 text-right font-semibold">vx m/s</th>
            <th className="text-right font-semibold">ω rad/s</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="pr-3 font-sans text-caption text-white/60">cmd</td>
            <td className="pr-3 text-right">{signed(drive?.cmdLinearX ?? 0)}</td>
            <td className="text-right">{signed(drive?.cmdAngularZ ?? 0)}</td>
          </tr>
          <tr>
            <td className="pr-3 font-sans text-caption text-white/60" title="Measured from wheel odometry">
              odom
            </td>
            <td className="pr-3 text-right">{signed(measured.vx)}</td>
            <td className="text-right">{signed(measured.wz)}</td>
          </tr>
        </tbody>
      </table>
      <div className={cn("min-w-0 flex-1 flex-wrap content-center items-center gap-1.5", hudCompact ? "hidden" : "flex")}>
        {battery?.present ? (
          <StatusPill
            tone={!socKnown ? "dim" : soc < 25 ? "err" : soc < 60 ? "warn" : "ok"}
            title="Battery"
          >
            {socKnown
              ? `${soc}%`
              : battery.statusReceived
                ? `${battery.batteryVoltageV.toFixed(1)} V`
                : "batt —"}
          </StatusPill>
        ) : null}
        <StatusPill tone={mode === "IDLE" || mode === "UNSPECIFIED" ? "dim" : "info"} title="Firmware mode">
          {MODE_LABEL[mode] ?? mode}
        </StatusPill>
        {wheeled ? (
          <StatusPill tone={armed === 0 ? "warn" : "ok"} title="Wheels armed">
            {armed}/{wheels.length} armed
          </StatusPill>
        ) : null}
        <StatusPill
          tone={
            driver.kind === "you" ? "ok" : driver.kind === "none" ? "dim" : driver.kind === "autonomous" ? "info" : "warn"
          }
          title="Who holds the drive seat"
        >
          {driver.kind === "none" ? "seat free" : driver.label}
        </StatusPill>
      </div>
    </div>
  );

  const profile = useControlProfile();
  const cycleProfile = () => {
    const i = CONTROL_PROFILES.findIndex((p) => p.id === profile.id);
    setActiveControlProfile(CONTROL_PROFILES[(i + 1) % CONTROL_PROFILES.length].id);
  };
  const sourceText = input.engaged && input.source
    ? `Driving · ${SOURCE_LABEL[input.source]}`
    : canDrive && input.linkOk
      ? padConnected
        ? "Ready · pad: hold RT"
        : "Ready"
      : "Locked";
  const driveChips = (
    <>
      <span
        role="status"
        className={cn(
          "inline-flex h-touch items-center gap-1.5 whitespace-nowrap rounded-full border px-3 text-caption font-bold uppercase tracking-wider backdrop-blur-md",
          input.engaged
            ? "border-success/60 bg-success/25 text-success-fg"
            : canDrive && input.linkOk
              ? "border-white/15 bg-black/60 text-white/85"
              : "border-white/10 bg-black/50 text-white/55",
        )}
      >
        <span
          className={cn(
            "h-2 w-2 rounded-full",
            input.engaged ? "animate-pulse bg-success" : canDrive && input.linkOk ? "bg-white/70" : "bg-white/30",
          )}
          aria-hidden
        />
        {sourceText}
      </span>
      <button
        type="button"
        onClick={cycleProfile}
        title={`Speed profile: ${profile.label} (${profile.maxLinear} m/s, ${profile.maxAngular} rad/s). Tap to change.`}
        aria-label={`Speed profile ${profile.label}. Change`}
        className={cn(
          "inline-flex h-touch items-center gap-1.5 whitespace-nowrap rounded-full border px-3 text-caption font-bold uppercase tracking-wider backdrop-blur-md",
          profile.id === "sport"
            ? "border-warning/50 bg-warning/20 text-warning-fg"
            : profile.id === "gentle"
              ? "border-success/40 bg-success/15 text-success-fg"
              : "border-white/15 bg-black/60 text-white/85",
        )}
      >
        <GaugeIcon />
        {profile.label}
      </button>
    </>
  );

  const joySize = stacked ? (md ? 210 : 168) : phoneLand ? 120 : short ? 160 : 188;
  const joystick = wheeled ? (
    <div className="rounded-full bg-black/35 backdrop-blur-sm">
      <DriveJoystick
        onTwist={input.touch.onTwist}
        onStop={input.touch.onStop}
        disabled={!canDrive}
        size={joySize}
        showHint={false}
      />
    </div>
  ) : null;

  // ------------------------------------------------------------ render
  // Short landscape screens have no room on the right (markers) or the left
  // (joystick), so the PiP takes the HUD's slot at the bottom centre.
  // Small landscape phones put it top-right, under the toolbar.
  const pipInHudSlot = !stacked && short && !phoneLand;
  const pipPlacement = stacked
    ? "right-2 top-2 w-[34%] min-w-32 max-w-72"
    : phoneLand
      ? "right-3 top-16 w-40"
      : pipInHudSlot
        ? "bottom-3 left-[calc(50%-90px)] w-[180px]"
        : "right-3 top-[4.5rem] w-[24%] min-w-40 max-w-80";

  const drawer = (
    <ControlsDrawer
      open={drawerOpen}
      docked={wide}
      onClose={closeDrawer}
      snapshot={snapshot}
      busy={busy}
      error={error}
      run={run}
      estopLatched={estopLatched}
      onEStop={() => void estop.trigger("gamepad")}
      onResetEStop={() => void estop.reset()}
      quickStartable={quickStartable}
      onStartDriving={startDriving}
      videoEnabled={videoEnabled}
      onVideoEnabled={setVideoEnabled}
      onReconnectVideo={reconnectVideo}
      spectatorOf={driver.kind === "human" ? driver.label : null}
    />
  );

  const toastEl = (cls: string) => (
    <ToastView toast={toast.toast} onDismiss={toast.dismiss} className={cn("absolute inset-x-0 z-[39] px-4", cls)} />
  );

  return (
    <div className="flex min-h-0 flex-1 bg-black">
      <div className={cn("relative flex min-h-0 min-w-0 flex-1 flex-col", stacked ? "bg-bg" : "bg-black")}>
        {stacked ? <div className="shrink-0 px-3 pt-2.5 pb-2">{toolbar}</div> : null}

        <div
          className={
            stacked
              ? "relative aspect-video max-h-[46dvh] w-full shrink-0"
              : cn("absolute inset-0", grid && "pt-[4.25rem]")
          }
        >
          <VideoStage
            endpoints={endpoints}
            main={main}
            pip={grid ? null : pip}
            grid={grid}
            videoEnabled={videoEnabled}
            reconnectKey={reconnectKey}
            onSelectMain={selectMain}
            onSwap={swap}
            onClosePip={closePip}
            onEnableVideo={enableVideo}
            onStats={setVideoStats}
            pipClassName={pipPlacement}
            className="h-full w-full"
          />
          {stacked ? toastEl("bottom-3") : null}
        </div>

        {stacked ? (
          // ---------------------------------------------------- stacked
          <div
            className={cn(
              "flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto px-3 pt-2.5",
              "pb-4",
            )}
          >
            <div className="flex flex-wrap items-center gap-2">
              {linkChip}
              <RecChip notes={notes} />
              {intentChip}
            </div>
            {statusLine}
            {hud}
            <EventButtons notes={notes} notify={notify} />
            <div className="flex-1" />
            {wheeled ? (
              <div className="flex items-start gap-3">
                {joystick}
                <div className="flex flex-col items-start gap-2">
                  {driveChips}
                </div>
              </div>
            ) : null}
          </div>
        ) : (
          // ---------------------------------------------------- overlay
          <>
            <div className="pointer-events-none absolute inset-x-0 top-0 z-20 bg-gradient-to-b from-black/70 to-transparent px-3 pt-3 pb-6">
              <div className="pointer-events-auto">{toolbar}</div>
            </div>
            {statusLine ? (
              <div
                className={cn(
                  "pointer-events-none absolute inset-x-0 z-10 flex px-3",
                  short ? "top-16" : "top-[4.5rem]",
                  phoneLand ? "justify-start" : "justify-center",
                  phoneLand && pip && "pr-[11.5rem]",
                )}
              >
                <div className={cn("pointer-events-auto max-w-[min(36rem,100%)]", pip && !grid && !pipInHudSlot && !phoneLand && "mr-[26%]")}>
                  {statusLine}
                </div>
              </div>
            ) : null}
            {toastEl(short ? "top-28" : "top-32")}
            <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex items-end gap-3 bg-gradient-to-t from-black/60 to-transparent px-3 pt-10 pb-[max(12px,env(safe-area-inset-bottom))]">
              {wheeled ? (
                <div
                  className={cn(
                    "pointer-events-auto flex shrink-0 gap-2",
                    short || phoneLand ? "flex-row items-end" : "flex-col items-start",
                  )}
                >
                  {phoneLand ? (
                    // Too narrow for a marker column: the HUD rides beside
                    // the joystick and the markers live in the Mark panel.
                    <>
                      {joystick}
                      <div className="flex flex-col items-start gap-2">{driveChips}</div>
                      {hud}
                    </>
                  ) : short ? (
                    <>
                      {joystick}
                      <div className="flex flex-col items-start gap-2">{driveChips}</div>
                    </>
                  ) : (
                    <>
                      <div className="flex gap-1.5">{driveChips}</div>
                      {joystick}
                    </>
                  )}
                </div>
              ) : null}
              {phoneLand ? null : (
                <>
                  <div className="pointer-events-auto flex min-w-0 flex-1 justify-center">
                    {(pipInHudSlot && pip && !grid) || !wheeled ? null : (
                      <div className="min-w-0 max-w-xl">{hud}</div>
                    )}
                  </div>
                  <div className="pointer-events-auto flex shrink-0 flex-col items-end gap-1.5">
                    {intentChip}
                    <EventButtons
                      notes={notes}
                      notify={notify}
                      layout={wide && !short ? "row" : short ? "grid" : "column"}
                      className={wide && !short ? "w-[27rem]" : short ? "w-52" : "w-36"}
                    />
                  </div>
                </>
              )}
            </div>
          </>
        )}

        {panel === "mark" ? (
          <PagePanel title="Drive markers" onClose={closePanel}>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <RecChip notes={notes} />
              <IntentChip notes={notes} open={false} onToggle={() => setPanel("intent")} />
            </div>
            <EventButtons notes={notes} notify={notify} layout="grid" />
          </PagePanel>
        ) : panel === "intent" ? (
          <PagePanel title="Drive intent" onClose={closePanel}>
            <IntentPanel notes={notes} notify={notify} onDone={closePanel} />
          </PagePanel>
        ) : panel === "link" ? (
          <PagePanel title="Link" onClose={closePanel}>
            <LinkDetails health={health} video={videoStats} videoOn={videoEnabled} />
          </PagePanel>
        ) : null}
        {wide ? null : drawer}
      </div>
      {wide ? drawer : null}
    </div>
  );
}
