// Teleoperation screen: live video + driving in one place.
//
// The motor bench owns bring-up (arming, calibration, dial-in) and the
// video screen owns camera gazing, but neither is a driving
// experience — until now teleoping meant bouncing between two screens
// with no eyes on the robot. This screen composes the shared pieces
// (`VideoFeed`, `DriveJoystick`, `PtzJoystick` + `useCameraPtz`,
// `GamepadDrive`) into a single operator surface:
//
//   * the MJPEG feed front and center, with the optional navigable-path
//     overlay and a one-tap "Labels" toggle,
//   * a sticky HUD (connection, mode, wheels armed, battery, camera
//     pose) with E-STOP always in reach,
//   * a "Start driving" quick-start that switches the runtime to
//     Dial-in mode and arms every wheel in one tap,
//   * every input path: on-screen joystick, WASD / arrows, a paired
//     Bluetooth gamepad (deadman-gated), and I / J / K / L to aim the
//     camera while WASD drives,
//   * a fullscreen mode that fills the screen with the feed and floats
//     the drive / camera pads over it — Esc or the exit button returns.
//
// Layout note: fullscreen is a class-name flip, not a second tree.
// The page chrome (toolbar, banners, drive/camera cards, gamepad
// bridge) stays mounted and is merely `hidden` while the fullscreen
// chrome renders — so the gamepad drive cycle and the video stream
// survive the toggle untouched. Only the on-screen pads move between
// the page and the overlay (a pad must never exist twice: two
// instances would double-bind the same keyboard chord), and a toggle
// mid-gesture intentionally halts — the unmounting pad enqueues a stop
// and the operator re-engages on the new one.
//
// Safety model (the firmware holds the last commanded twist until told
// otherwise, so *this screen* owns stopping): every gesture end —
// joystick release, last drive key, deadman release, E-STOP latch, pad
// disconnect, fullscreen toggle, tab hidden, or leaving the screen —
// funnels a zero twist through the coalesced sender exactly once per
// cycle. See `DriveJoystick` / `GamepadDrive` for their own edge cases.

import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { ControlProfilePicker } from "../components/ControlProfilePicker";
import { DriveJoystick } from "../components/DriveJoystick";
import { GamepadDrive } from "../components/GamepadDrive";
import { PtzJoystick, PTZ_KEYS_IJKL } from "../components/PtzJoystick";
import { useCameraPtz } from "../components/useCameraPtz";
import { VideoFeed } from "../components/VideoFeed";
import { Banner, Button } from "../components/ui";
import { useGamepad } from "../input";
import { getOrCreateRuntimeTransport } from "../runtime";
import type { RuntimeTransport } from "../runtime";
import type {
  CameraView,
  NavView,
  RuntimeConnectionState,
  RuntimeMode,
  RuntimeSnapshot,
  WheelView,
} from "../runtime";

interface TeleopScreenProps {
  /** IP address of the robot (LAN address of the bebop-linux runtime). */
  robotIp: string;
  /** Optional override for the runtime port. Defaults to 9090. */
  runtimePort?: number;
  onBack: () => void;
  /** Label for the back button, e.g. "Back to motor bench". */
  backLabel?: string;
  /** Optional link to controller pairing — offered when the robot has
   *  no wheeled drive (body-velocity teleop pairs a pad to the robot's
   *  own BlueZ instead) and as a convenience link in the footer. */
  onOpenControllers?: () => void;
}

const MODE_LABEL: Record<RuntimeMode, string> = {
  UNSPECIFIED: "Unknown",
  IDLE: "Idle",
  DIAL_IN: "Dial-in",
  RUN_POLICY: "Policy",
};

/// How often the drive keepalive re-sends the current twist while a
/// drive cycle is open. Must stay comfortably below the firmware's
/// `operator_timeout_ms` (default 500 ms — see
/// `firmware/bebop-linux/src/config.rs`) so a healthy session never
/// trips the robot-side link-loss watchdog; 10 Hz leaves ~4 missed
/// packets of slack.
const TWIST_KEEPALIVE_MS = 100;

const fmt = (v: number): string =>
  v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2);

export function TeleopScreen({
  robotIp,
  runtimePort = 9090,
  onBack,
  backLabel = "Back",
  onOpenControllers,
}: TeleopScreenProps) {
  // Video stream + runtime link are independent transports (HTTP vs
  // WS), so the screen renders immediately: the feed shows its own
  // loading / error placeholders while the WS connects in the
  // background.
  const [reconnectKey, setReconnectKey] = useState(0);
  const [streamState, setStreamState] = useState<
    "loading" | "live" | "error"
  >("loading");
  const [connState, setConnState] = useState<RuntimeConnectionState>(
    "disconnected",
  );
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Nav overlay: off by default so the raw video is what you get on
  // entry; one tap adds the navigable-path wash.
  const [showNav, setShowNav] = useState(false);
  // Phones open in the immersive layout by default — a small screen
  // is all video real estate, with the pads floating over it and the
  // HUD condensed into the top bar. Desktops / tablets keep the page
  // layout (cards + big video) with the explicit Fullscreen button.
  // Evaluated once on mount: rotating later shouldn't yank the
  // operator between layouts mid-session, and Exit keeps whatever the
  // user chose.
  const [fullscreen, setFullscreen] = useState(() =>
    typeof window !== "undefined"
      ? window.matchMedia("(max-width: 639px)").matches
      : false,
  );
  // A connected gamepad replaces the on-screen drive pads (and adds
  // right-stick camera aim below) — one way to drive at a time.
  const { connected: padConnected } = useGamepad();

  const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
  const transportRef = useRef<RuntimeTransport | null>(null);

  // -------------------------------------------------------------- lifecycle
  //
  // Same cached-transport pattern as the motor bench: the socket is
  // owned by the module-level cache (one per ip:port), listeners are
  // per-mount, and reconnects self-heal via the connection-state
  // listener re-fetching a snapshot + re-subscribing telemetry.
  useEffect(() => {
    let cancelled = false;
    const t = getOrCreateRuntimeTransport(robotIp, runtimePort);
    transportRef.current = t;
    const offCallbacks: Array<() => void> = [];

    // Coalesce telemetry-driven re-renders to one per animation frame
    // (the firmware pumps ~30 Hz).
    let pendingSnapshot: RuntimeSnapshot | null = null;
    let rafId: number | null = null;
    const flush = () => {
      rafId = null;
      if (cancelled) return;
      const s = pendingSnapshot;
      pendingSnapshot = null;
      if (s) setSnapshot(s);
    };
    const scheduleFlush = (s: RuntimeSnapshot) => {
      pendingSnapshot = s;
      if (rafId === null) rafId = requestAnimationFrame(flush);
    };

    offCallbacks.push(
      t.onTelemetry(scheduleFlush),
      t.onEStopLatched(() => {
        if (!cancelled) {
          void t.getSnapshot().then((s) => {
            if (!cancelled) setSnapshot(s);
          });
        }
      }),
      t.onConnectionStateChange((state) => {
        if (cancelled) return;
        setConnState(state);
        if (state !== "connected") return;
        void (async () => {
          try {
            const s = await t.getSnapshot();
            if (cancelled) return;
            setSnapshot(s);
            setError(null);
            await t.subscribeTelemetry(30);
          } catch {
            /* the next reconnect will retry */
          }
        })();
      }),
    );

    void t.connect(robotIp, runtimePort).catch(() => {
      /* auto-retry every ~5 s; the conn pill + banner surface it */
    });

    return () => {
      cancelled = true;
      if (rafId !== null) cancelAnimationFrame(rafId);
      rafId = null;
      pendingSnapshot = null;
      for (const off of offCallbacks) off();
      void t.unsubscribeTelemetry().catch(() => {
        /* swallow: socket may already be closed */
      });
      transportRef.current = null;
    };
  }, [robotIp, runtimePort]);

  // -------------------------------------------------------------- actions
  const refreshAfter = useCallback(
    async (label: string, fn: () => Promise<unknown>) => {
      const t = transportRef.current;
      if (!t) return;
      setBusy(label);
      setError(null);
      try {
        await fn();
        const s = await t.getSnapshot();
        setSnapshot(s);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [],
  );

  // Throttled twist sender (single in-flight + latest pending) — the
  // joystick / keyboard / gamepad paths all funnel through this, so
  // the firmware sees a continuous stream without queueing one WS
  // request per pointer move. Stop sends are silent: halting into a
  // not-drivable state (E-STOP just latched, mode switched) would
  // otherwise surface a confusing rejection banner on top of the
  // banner that explains why.
  const twistInFlightRef = useRef(false);
  const twistPendingRef = useRef<{ vx: number; wz: number } | null>(null);
  const sentAnythingRef = useRef(false);

  // Drive-cycle bookkeeping for the keepalive below: `drivingRef` is
  // true while a gesture holds a non-zero twist, `lastTwistRef` holds
  // the latest one. Closed by every stop path (release, deadman drop,
  // visibility change, gesture teardown).
  const drivingRef = useRef(false);
  const lastTwistRef = useRef<{ vx: number; wz: number } | null>(null);

  const sendTwist = useCallback(
    async (vx: number, wz: number, silent = false) => {
      const t = transportRef.current;
      if (!t) return;
      sentAnythingRef.current = true;
      const payload = { vx, wz };
      lastTwistRef.current = payload;
      if (vx !== 0 || wz !== 0) drivingRef.current = true;
      if (twistInFlightRef.current) {
        twistPendingRef.current = payload;
        return;
      }
      twistInFlightRef.current = true;
      try {
        await t.setVelocityCommand(vx, wz);
      } catch (e) {
        if (!silent) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        twistInFlightRef.current = false;
        const next = twistPendingRef.current;
        twistPendingRef.current = null;
        if (next) {
          void Promise.resolve().then(() => sendTwist(next.vx, next.wz, silent));
        }
      }
    },
    [],
  );

  // One-shot "stop": enqueue a (0,0) twist that supersedes anything
  // mid-flight, so every gesture end halts the robot.
  const stopDrive = useCallback(() => {
    drivingRef.current = false;
    lastTwistRef.current = null;
    twistPendingRef.current = { vx: 0, wz: 0 };
    void sendTwist(0, 0, true);
  }, [sendTwist]);

  // Operator-link keepalive: the firmware zeroes any twist that isn't
  // refreshed within its `operator_timeout_ms` (500 ms default) — the
  // robot-side backstop for a dropped link. Every drive input here
  // emits only on *change*, so a steady stick would look exactly like a
  // dead operator to that watchdog. While a drive cycle is open,
  // re-send the current twist at ~10 Hz; the coalescer above keeps this
  // to one in-flight request + one pending. Repeats are silent (a
  // flaky link already shows in the connection pill, and the robot
  // halting is precisely the designed behavior we don't need to nag
  // about).
  useEffect(() => {
    if (connState !== "connected") return;
    const iv = window.setInterval(() => {
      if (!drivingRef.current) return;
      const last = lastTwistRef.current;
      if (last) void sendTwist(last.vx, last.wz, true);
    }, TWIST_KEEPALIVE_MS);
    return () => window.clearInterval(iv);
  }, [connState, sendTwist]);

  // Leaving the screen (or the whole tab) mid-drive must halt the
  // chassis — the firmware would otherwise hold the last twist. The
  // per-gesture stops above cover normal releases; this is the
  // belt-and-braces for "the component went away".
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) stopDrive();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () =>
      document.removeEventListener("visibilitychange", onVisibility);
  }, [stopDrive]);
  useEffect(
    () => () => {
      if (!sentAnythingRef.current) return;
      // Re-fetch from the endpoint cache (the lifecycle cleanup above
      // nulled transportRef); if the socket's already gone the catch
      // swallows it.
      const t = getOrCreateRuntimeTransport(robotIp, runtimePort);
      void t.setVelocityCommand(0, 0).catch(() => {
        /* socket may already be closed */
      });
    },
    [robotIp, runtimePort],
  );

  const eStop = useCallback(
    () => refreshAfter("estop", () => transportRef.current!.emergencyStop("operator")),
    [refreshAfter],
  );

  const resetEStop = useCallback(
    () => refreshAfter("reset", () => transportRef.current!.resetEStop()),
    [refreshAfter],
  );

  const setAllWheels = useCallback(
    (enabled: boolean) =>
      refreshAfter(`all-wheels:${enabled}`, () =>
        transportRef.current!.setAllWheelsEnabled(enabled),
      ),
    [refreshAfter],
  );

  const toggleWheel = useCallback(
    (wheel: string, enabled: boolean) =>
      refreshAfter(`wheel:${wheel}`, () =>
        transportRef.current!.setWheelEnabled(wheel, enabled),
      ),
    [refreshAfter],
  );

  // -------------------------------------------------------------- derived
  const motors = snapshot?.motors ?? [];
  const wheels = snapshot?.wheels ?? [];
  const drive = snapshot?.drive;
  const power = snapshot?.power;
  const camera: CameraView | null = snapshot?.camera ?? null;
  const nav: NavView | null = snapshot?.nav ?? null;
  const mode = snapshot?.mode ?? "UNSPECIFIED";
  const estopLatched = snapshot?.estopLatched ?? false;
  const estopReason = snapshot?.estopReason ?? "";
  const armedWheelCount = wheels.filter((w) => w.armed).length;
  // True when driving a differential-drive chassis (no legged joints).
  const wheeled = !!drive?.present && motors.length === 0;
  const canDrive =
    wheeled &&
    !estopLatched &&
    (mode === "DIAL_IN" || mode === "RUN_POLICY") &&
    armedWheelCount > 0;

  // What's between the operator and driving, in priority order. E-STOP
  // needs its own action (never auto-reset); everything else collapses
  // into the one-tap quick-start below.
  const driveBlockedReason = !wheeled
    ? null
    : estopLatched
      ? "E-STOP latched — reset it, then drive."
      : mode !== "DIAL_IN" && mode !== "RUN_POLICY"
        ? "Switch to Dial-in mode to drive."
        : armedWheelCount === 0
          ? "Enable the wheels to drive."
          : drive?.hasActiveOperator && !drive?.youAreActiveOperator
            ? "Another operator is driving — drive commands from this device are rejected until theirs go quiet for a couple of seconds."
            : null;

  const startDriving = useCallback(() => {
    if (estopLatched) return;
    void refreshAfter("start-driving", async () => {
      const t = transportRef.current!;
      if (mode !== "DIAL_IN" && mode !== "RUN_POLICY") {
        await t.setMode("DIAL_IN");
      }
      if (armedWheelCount === 0) {
        await t.setAllWheelsEnabled(true);
      }
    });
  }, [refreshAfter, mode, armedWheelCount, estopLatched]);

  const ptzReady =
    connState === "connected" && camera !== null && camera.present;
  const ptz = useCameraPtz(transport, camera, ptzReady);

  // Esc exits fullscreen.
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  // -------------------------------------------------------------- HUD bits
  const connTone =
    connState === "connected" ? "ok" : connState === "connecting" ? "warn" : "dim";
  const connLabel =
    connState === "connected"
      ? "live"
      : connState === "connecting"
        ? "reconnecting…"
        : "offline";
  const socKnown =
    power?.present && power.statusReceived && power.stateOfChargePct >= 0;
  const batteryTone = !socKnown
    ? "dim"
    : power!.stateOfChargePct < 25
      ? "err"
      : power!.stateOfChargePct < 60
        ? "warn"
        : "ok";
  const batteryLabel = power?.present
    ? socKnown
      ? `${power.stateOfChargePct.toFixed(0)}% · ${power.batteryVoltageV.toFixed(1)} V`
      : power.statusReceived
        ? `${power.batteryVoltageV.toFixed(1)} V`
        : "battery —"
    : null;

  // The status strip rendered in both the page toolbar and the
  // fullscreen top bar.
  const statusPills = (
    <>
      <Pill tone={connTone} title="Runtime WebSocket link">
        {connLabel}
      </Pill>
      <Pill tone="dim" title="Firmware mode">
        {MODE_LABEL[mode]}
      </Pill>
      {wheeled ? (
        <Pill tone={armedWheelCount === 0 ? "warn" : "ok"} title="Wheels armed">
          {armedWheelCount}/{wheels.length} wheels
        </Pill>
      ) : null}
      {wheeled && drive?.operatorStale ? (
        <Pill
          tone="warn"
          title="No fresh drive commands reached the robot within its operator timeout; it zeroed the twist and is holding still. Motion resumes when a fresh command arrives (this pill clears)."
        >
          link stale — motion halted
        </Pill>
      ) : null}
      {wheeled && drive?.hasActiveOperator && !drive?.youAreActiveOperator ? (
        <Pill
          tone="warn"
          title="Another device holds the drive assignment (input-based arbitration). Drive commands from this device are rejected until the other operator stops for a couple of seconds; stops and E-STOP always work."
        >
          another operator driving
        </Pill>
      ) : null}
      {batteryLabel ? (
        <Pill tone={batteryTone} title="Battery state of charge">
          {batteryLabel}
        </Pill>
      ) : null}
      {camera?.present ? (
        <Pill tone="dim" title="Camera pan / tilt pose">
          {camera.panDeg.toFixed(0)}° / {camera.tiltDeg.toFixed(0)}°
          {camera.moving ? " ⤳" : ""}
        </Pill>
      ) : null}
      {estopLatched ? (
        <Pill tone="err" title={estopReason || "E-STOP latched"}>
          E-STOP
        </Pill>
      ) : null}
    </>
  );

  const labelsToggle = (small = false) => (
    <Button
      variant={showNav ? "secondary" : "ghost"}
      disabled={nav !== null && !nav.present}
      className={`text-xs ${small ? "px-2.5 py-1.5 h-8" : "px-3 py-1.5 h-8"}`}
      onClick={() => setShowNav((on) => !on)}
    >
      Labels{nav?.present && showNav ? ` · ${nav.maskHz.toFixed(0)} Hz` : ""}
    </Button>
  );

  const estopButton = (
    <Button
      onClick={eStop}
      disabled={busy === "estop" || estopLatched}
      className="bg-danger! text-white! hover:bg-[#e94a50]! py-2.5! text-sm! shrink-0 min-w-[88px]"
    >
      E-STOP
    </Button>
  );

  // One live instance of each gesture pad at a time — see the layout
  // note on this component. `drivePad`/`ptzPad` are element
  // descriptions; exactly one mount site renders each.
  const drivePad = (
    <DriveJoystick
      onTwist={(vx, wz) => void sendTwist(vx, wz)}
      onStop={stopDrive}
      disabled={!canDrive}
    />
  );
  const ptzPad = (
    <PtzJoystick
      onRate={ptz.onRate}
      onStop={ptz.onStop}
      disabled={!ptzReady}
      keys={PTZ_KEYS_IJKL}
      hint="I / J / K / L to aim"
    />
  );

  // -------------------------------------------------------------- render
  return (
    <div
      className={
        fullscreen
          ? "fixed inset-0 z-40 bg-black flex flex-col"
          : "flex flex-col gap-3"
      }
    >
      {/* Fullscreen top bar (page chrome below hides while it shows). */}
      {fullscreen ? (
        <div className="flex flex-wrap items-center gap-2 px-3 py-2 border-b border-white/10 shrink-0">
          <Button
            variant="secondary"
            className="py-2! text-sm!"
            onClick={() => setFullscreen(false)}
          >
            Exit fullscreen
          </Button>
          <span className="text-[10px] uppercase tracking-wider text-text-dim hidden sm:inline">
            Esc
          </span>
          <div className="flex items-center gap-1.5 flex-wrap">
            {statusPills}
          </div>
          <div className="flex-1" aria-hidden />
          {wheeled && driveBlockedReason && !estopLatched ? (
            <Button
              onClick={startDriving}
              loading={busy === "start-driving"}
              disabled={!!busy}
              className="bg-success! text-white! hover:brightness-110! py-2! text-sm!"
            >
              Start driving
            </Button>
          ) : null}
          {estopLatched ? (
            <Button
              variant="secondary"
              onClick={resetEStop}
              loading={busy === "reset"}
              className="py-2! text-sm!"
            >
              Reset E-STOP
            </Button>
          ) : null}
          {labelsToggle(true)}
          {estopButton}
        </div>
      ) : null}

      {/* Page chrome: toolbar + banners. Hidden (not unmounted) in
          fullscreen so listeners and the video's neighbours survive. */}
      <div className={fullscreen ? "hidden" : "contents"}>
        {/* Sticky HUD: status pills on top, actions + E-STOP below
            (E-STOP is thumb-anchored on phones, one row away at worst). */}
        <div className="sticky top-0 z-10 -mx-4 px-4 sm:-mx-6 sm:px-6 pt-1 pb-3 bg-bg/85 backdrop-blur-md">
          <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3 py-2.5 flex flex-col gap-3 lg:flex-row lg:items-center lg:gap-5">
            <div className="flex items-center gap-1.5 flex-wrap min-w-0">
              <span className="text-sm font-semibold text-text mr-1">
                Teleop
              </span>
              <span className="text-[11px] text-text-dim font-mono mr-2">
                {robotIp}:{runtimePort}
              </span>
              {statusPills}
            </div>

            <div className="hidden lg:block flex-1" aria-hidden />

            <div className="flex items-center gap-2 flex-wrap">
              {labelsToggle()}
              <ControlProfilePicker />
              <Button
                variant="secondary"
                className="py-2! text-sm!"
                onClick={() => setFullscreen(true)}
                title="Fill the screen with the live feed and float the drive controls over it"
              >
                Fullscreen
              </Button>
              {estopButton}
            </div>
          </div>
        </div>

        {/* E-STOP banner */}
        {estopLatched ? (
          <Banner tone="error">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="font-semibold mb-0.5">E-STOP latched</div>
                <div className="text-xs leading-relaxed">
                  {estopReason || "(no reason recorded)"}
                </div>
              </div>
              <Button
                variant="secondary"
                onClick={resetEStop}
                loading={busy === "reset"}
              >
                Reset
              </Button>
            </div>
          </Banner>
        ) : null}

        {/* WS reconnect banners */}
        {connState !== "connected" && snapshot ? (
          <Banner tone="info">
            <div className="text-sm">
              Reconnecting to the robot runtime… telemetry and controls resume
              automatically; the video stream may keep playing.
            </div>
          </Banner>
        ) : null}
        {!snapshot && connState !== "connected" ? (
          <Banner tone="info">
            <div className="text-sm">
              Connecting to the runtime at{" "}
              <code>
                {robotIp}:{runtimePort}
              </code>
              …
            </div>
          </Banner>
        ) : null}

        {error ? <Banner tone="error">{error}</Banner> : null}
      </div>

      {/* Live feed. `contents` in page mode keeps the feed a direct
          child of the screen's flex column; in fullscreen the wrapper
          becomes the flex-1 video area with the floating pads. On
          phones the feed breaks out of the page padding (-mx-4) and
          drops the card chrome so the video is edge-to-edge — the
          aspect follows the negotiated stream, not a hard-coded box. */}
      <div className={fullscreen ? "relative flex-1 min-h-0" : "contents"}>
        <VideoFeed
          baseUrl={`http://${robotIp}:${runtimePort}`}
          transport={transport}
          showNav={showNav}
          nav={nav}
          reconnectKey={reconnectKey}
          onStreamState={setStreamState}
          className={
            fullscreen
              ? "w-full h-full"
              : "w-full -mx-4 sm:mx-0 sm:rounded-[var(--radius-card)] sm:border sm:border-border"
          }
          maxHeight={fullscreen ? undefined : "72dvh"}
        />
        {fullscreen && wheeled && !padConnected ? (
          <div className="absolute bottom-4 left-4 z-10 rounded-[var(--radius-card)] border border-white/10 bg-bg-elev/75 backdrop-blur-md p-2 max-sm:scale-90 max-sm:origin-bottom-left">
            {drivePad}
          </div>
        ) : null}
        {fullscreen && camera !== null && camera.present ? (
          <div className="absolute bottom-4 right-4 z-10 rounded-[var(--radius-card)] border border-white/10 bg-bg-elev/75 backdrop-blur-md p-2 max-sm:scale-90 max-sm:origin-bottom-right">
            {ptzPad}
          </div>
        ) : null}
        {fullscreen && streamState === "error" ? (
          <div className="absolute inset-x-0 bottom-4 z-10 flex justify-center">
            <Button
              variant="secondary"
              onClick={() => setReconnectKey((n) => n + 1)}
              className="py-2! text-sm!"
            >
              Reconnect video
            </Button>
          </div>
        ) : null}
      </div>

      {/* Page content below the feed — hidden, never unmounted, in
          fullscreen (the gamepad bridge must keep its poll loop). */}
      <div className={fullscreen ? "hidden" : "contents"}>
        {showNav && nav !== null && !nav.present ? (
          <Banner tone="error">
            The robot has no navigable-path runner: add a <code>nav:</code>{" "}
            block to the firmware YAML and ship <code>navseg.onnx</code> next
            to it.
          </Banner>
        ) : null}

        {streamState === "error" ? (
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <span className="text-xs text-text-dim">
              Camera stream failed — bebop-linux may be down or the robot has
              no <code>video:</code> config. Driving controls still work.
            </span>
            <Button
              variant="secondary"
              onClick={() => setReconnectKey((n) => n + 1)}
              className="py-2! text-sm!"
            >
              Reconnect video
            </Button>
          </div>
        ) : null}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-start">
          {wheeled ? (
            <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3.5 py-3 space-y-4">
              {/* Drive card */}
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="text-[11px] uppercase tracking-wider text-text-dim">
                    Drive
                  </div>
                  <div className="text-[13px] text-text font-semibold mt-0.5">
                    {wheels.length} wheel{wheels.length === 1 ? "" : "s"} ·{" "}
                    {armedWheelCount} armed
                  </div>
                </div>
                <div className="flex gap-2 shrink-0 items-center">
                  <ControlProfilePicker />
                  <Button
                    variant="secondary"
                    onClick={() => setAllWheels(false)}
                    disabled={!!busy || armedWheelCount === 0}
                    className="py-2! text-sm!"
                  >
                    Disable wheels
                  </Button>
                </div>
              </div>

              {driveBlockedReason ? (
                <div className="rounded-[var(--radius-card)] border border-accent/30 bg-accent/5 px-3 py-2.5 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="text-[12px] text-text-dim leading-relaxed">
                    {driveBlockedReason}
                  </div>
                  {estopLatched ? (
                    <Button
                      variant="secondary"
                      onClick={resetEStop}
                      loading={busy === "reset"}
                      className="py-2! text-sm! shrink-0"
                    >
                      Reset E-STOP
                    </Button>
                  ) : (
                    <Button
                      onClick={startDriving}
                      loading={busy === "start-driving"}
                      disabled={!!busy}
                      className="bg-success! text-white! hover:brightness-110! py-2! text-sm! shrink-0"
                    >
                      Start driving
                    </Button>
                  )}
                </div>
              ) : (
                <div className="rounded-[var(--radius-card)] border border-success/40 bg-success/10 px-3 py-2 text-[12px] text-text-dim leading-relaxed">
                  <span className="text-success font-semibold">
                    Ready to drive
                  </span>{" "}
                  — joystick, WASD / arrows, or a paired gamepad (hold the
                  right trigger as deadman). Releasing anything stops the
                  robot.
                </div>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-4 items-center justify-items-center">
                {/* On-screen drive pad. Hidden while a controller is
                    connected (the sticks drive — see the bridge card
                    below) and while the fullscreen overlay owns it —
                    never two live instances of the same pad, or its
                    WASD / arrow keyboard binding would double-fire. */}
                {!fullscreen && !padConnected ? drivePad : null}
                {padConnected ? (
                  <div className="flex flex-col items-center gap-1 text-center max-w-56 px-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-text-dim">
                      Controller driving
                    </span>
                    <span className="text-[11px] text-text-dim leading-snug">
                      Left stick drives;{" "}
                      {camera?.present
                        ? "right stick aims the camera. "
                        : "right stick turns in split layout. "}
                      On-screen drive pad hidden while the controller is
                      connected.
                    </span>
                  </div>
                ) : null}

                <div className="w-full space-y-3">
                  {/* Odometry + commanded twist */}
                  <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev-2/40 px-3 py-2.5">
                    <div className="text-[11px] uppercase tracking-wider text-text-dim mb-1.5">
                      Odometry
                    </div>
                    <div className="font-mono text-[13px] text-text leading-relaxed">
                      <div>
                        x {fmt(drive?.odomX ?? 0)} m · y {fmt(drive?.odomY ?? 0)}{" "}
                        m · θ {fmt(((drive?.odomTheta ?? 0) * 180) / Math.PI)}°
                      </div>
                      <div className="text-[11px] text-text-dim mt-0.5">
                        cmd vx {fmt(drive?.cmdLinearX ?? 0)} m/s · ω{" "}
                        {fmt(drive?.cmdAngularZ ?? 0)} rad/s
                      </div>
                    </div>
                  </div>

                  {/* Wheel arm chips */}
                  <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev-2/40 px-3 py-2.5 space-y-2">
                    {wheels.map((w) => (
                      <WheelChip
                        key={w.name}
                        wheel={w}
                        busy={busy === `wheel:${w.name}`}
                        estopLatched={estopLatched}
                        canArm={mode === "DIAL_IN" || mode === "RUN_POLICY"}
                        onToggle={(enabled) => toggleWheel(w.name, enabled)}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // No wheeled drive: legged robot or no `drive:` config.
            <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3.5 py-3 space-y-2">
              <div className="text-[11px] uppercase tracking-wider text-text-dim">
                Drive
              </div>
              <div className="text-[13px] text-text-dim leading-relaxed">
                This robot has no wheeled <code>drive:</code> config, so
                there&rsquo;s nothing to drive from the app. Aim the camera
                with the pad on the right, or use the motor bench for
                per-joint dial-in.
                {onOpenControllers ? (
                  <>
                    {" "}
                    For body-velocity teleop, pair a controller to the{" "}
                    <span className="text-text">robot&rsquo;s</span> BlueZ
                    stack.
                  </>
                ) : null}
              </div>
              {onOpenControllers ? (
                <Button
                  variant="secondary"
                  onClick={onOpenControllers}
                  className="py-2! text-sm! self-start"
                >
                  Bluetooth controller
                </Button>
              ) : null}
            </div>
          )}

          {/* Camera card */}
          <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3.5 py-3 space-y-3">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="text-[11px] uppercase tracking-wider text-text-dim">
                  Camera
                </div>
                <div className="text-[13px] text-text font-semibold mt-0.5 flex items-center gap-2">
                  <span>
                    pan {camera ? `${camera.panDeg.toFixed(1)}°` : "—"} · tilt{" "}
                    {camera ? `${camera.tiltDeg.toFixed(1)}°` : "—"}
                  </span>
                  {camera?.moving ? (
                    <span
                      className="inline-block w-2 h-2 rounded-full bg-accent animate-pulse"
                      title="Gimbal slewing"
                    />
                  ) : null}
                </div>
              </div>
              <Button
                variant="ghost"
                disabled={!ptzReady}
                onClick={ptz.center}
                className="py-1.5! px-2! text-[12px]!"
              >
                Center
              </Button>
            </div>

            <div className="flex justify-center">
              {!fullscreen ? ptzPad : null}
            </div>

            {camera !== null && !camera.present ? (
              <span className="text-xs text-text-dim">
                PTZ unavailable — this robot has no <code>video:</code>{" "}
                config, so the firmware reports no camera.
              </span>
            ) : null}
            {camera !== null && camera.present && connState !== "connected" ? (
              <span className="text-xs text-text-dim">
                Reconnecting to the runtime… controls disabled.
              </span>
            ) : null}
            {ptz.error ? <Banner tone="error">{ptz.error}</Banner> : null}
          </div>
        </div>

        {/* Gamepad bridge — mounted for the screen's whole lifetime
            (hidden in fullscreen) so a drive cycle survives the toggle.
            The PTZ bridge is handed over only when the firmware
            actually reports a camera; without one the right stick
            keeps its split-layout turning role. */}
        {wheeled ? (
          <GamepadDrive
            wheels={wheels}
            mode={mode}
            estopLatched={estopLatched}
            onTwist={(vx, wz) => void sendTwist(vx, wz)}
            onStop={stopDrive}
            onEStop={eStop}
            onResetEStop={resetEStop}
            onSetAllWheels={setAllWheels}
            ptz={camera !== null && camera.present ? ptz : null}
          />
        ) : null}

        <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 mt-2">
          <span className="text-[11px] text-text-dim">
            {padConnected
              ? `Controller: left stick drives${
                  camera?.present ? " · right stick aims the camera" : ""
                }`
              : "WASD / arrows drive · I / J / K / L aim the camera"}
          </span>
          {onOpenControllers ? (
            <Button variant="ghost" onClick={onOpenControllers}>
              Bluetooth controller
            </Button>
          ) : null}
          <Button variant="ghost" onClick={onBack}>
            {backLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}

/// Compact status pill used by the HUD / fullscreen bars.
function Pill({
  tone = "dim",
  title,
  children,
}: {
  tone?: "ok" | "warn" | "err" | "dim";
  title?: string;
  children: ReactNode;
}) {
  const toneClasses = {
    ok: "bg-success/15 text-success border-success/40",
    warn: "bg-yellow-500/15 text-yellow-700 dark:text-yellow-300 border-yellow-500/40",
    err: "bg-danger/12 text-[#ffb5b8] border-danger/40",
    dim: "bg-bg-elev-2 text-text-dim border-border",
  }[tone];
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider whitespace-nowrap ${toneClasses}`}
    >
      {children}
    </span>
  );
}

/// One wheel's arm switch + live velocity, teleop-sized.
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
    <div className="flex items-center gap-2.5">
      <button
        type="button"
        role="switch"
        aria-checked={wheel.armed}
        aria-label={`Arm ${wheel.name}`}
        onClick={() => onToggle(!wheel.armed)}
        disabled={busy || estopLatched || (!wheel.armed && !canArm)}
        className={`relative shrink-0 w-9 h-5 rounded-full border transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer ${
          wheel.armed ? "bg-success border-success" : "bg-bg-elev border-border"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-3.5 h-3.5 rounded-full bg-white shadow transition-transform ${
            wheel.armed ? "translate-x-4" : "translate-x-0"
          }`}
          aria-hidden
        />
      </button>
      <span className="text-[12px] text-text font-medium truncate">
        {wheel.name}
      </span>
      {wheel.errorCode !== 0 ? (
        <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-danger/40 bg-danger/10 text-danger">
          error
        </span>
      ) : null}
      <span className="ml-auto text-[11px] text-text-dim font-mono">
        {fmt(wheel.velocity)} rad/s
      </span>
    </div>
  );
}
