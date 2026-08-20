// Bluetooth-gamepad → differential-drive teleop bridge.
//
// The wheeled-chassis counterpart of `GamepadDriver` (which handles
// per-joint dial-in on legged robots). Streams body twists through
// the runtime WS `SetVelocityCommand` path — the same one the
// on-screen `DriveJoystick` and the WASD keyboard drive use — so no
// firmware or agent change is involved.
//
// Bindings mirror the dial-in bridge and the robot-side teleop state
// machine (`controller/teleop.rs`) so muscle memory transfers between
// flows:
//
//   * sticks drive the chassis — layout selectable below,
//   * RT is a deadman; release to halt,
//   * trigger pressure scales the request so a light pull creeps,
//   * B latches E-STOP, A clears it,
//   * L3 arms / disarms every wheel.
//
// The firmware holds the last commanded twist until told otherwise,
// so this component owns *stopping*: deadman release, E-STOP latch,
// pad disconnect, unmount, tab hidden, and a watchdog on tick
// freshness all enqueue a (0, 0) twist exactly once per drive cycle.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { getActiveControlProfile, subscribeGamepad, useGamepad } from "../input";
import type { GamepadSnapshot, LogicalSnapshot } from "../input";
import type { RuntimeMode, WheelView } from "../runtime";
import { ControllerIcon, Hint, prettifyGamepadId } from "./GamepadDriver";
import { ControlProfilePicker } from "./ControlProfilePicker";

// Drive soft limits come from the active control profile
// (`../input/profile.ts`) — switched at runtime from the picker on
// this card. The firmware clamps to each wheel's `vel_max`
// regardless, so the profile only bounds what the operator can
// request with a full stick deflection. Shared with the on-screen
// `DriveJoystick` / WASD keyboard drive in `MotorBenchScreen`.

/// Stick layout. `split` (default): left stick ↕ drives, right stick ↔
/// turns. `arcade`: the left stick does both, matching the on-screen
/// joystick's convention.
type DriveLayout = "split" | "arcade";

const LAYOUT_STORAGE_KEY = "bebop.driveLayout";

/// Deadman engages at this trigger pressure. Matches the dial-in
/// bridge's threshold so the feel is identical between flows.
const DEADMAN_THRESHOLD = 0.4;

/// If no gamepad tick arrives for this long while driving, halt.
/// Covers the RAF poll loop stalling (tab backgrounded, WebView
/// paused) and a pad that silently drops off mid-drive. Sits just
/// above the 200 ms watchdog in the robot-side `controller/teleop.rs`
/// to avoid racing the ~60 Hz poll loop's normal jitter.
const WATCHDOG_MS = 250;

/// Minimum twist change worth a WS write. Filters stick sensor noise
/// around centre while the deadman is held; the parent's in-flight
/// coalescing bounds the remaining send rate to one outstanding
/// request + one pending.
const TWIST_EPS = 1e-3;

/// UI-only state (deadman pill, twist readout) refreshes at ~10 Hz
/// instead of RAF rate.
const UI_INTERVAL_MS = 100;

interface GamepadDriveProps {
  wheels: WheelView[];
  mode: RuntimeMode;
  estopLatched: boolean;
  /// Stream a body twist (vx m/s, wz rad/s, +wz = left turn). Throttling
  /// + in-flight coalescing is the parent's responsibility — same
  /// pattern as the on-screen joystick.
  onTwist: (vx: number, wz: number) => void;
  /// One-shot stop; supersedes anything in flight.
  onStop: () => void;
  onEStop: () => void;
  onResetEStop: () => void;
  /// Arm / disarm every wheel (L3 toggles between the two).
  onSetAllWheels: (enabled: boolean) => void;
}

/// Renders a compact status card and streams gamepad-driven twists to
/// the parent. Returns null whenever no pad is connected so the bench
/// layout doesn't reserve space for it.
export function GamepadDrive({
  wheels,
  mode,
  estopLatched,
  onTwist,
  onStop,
  onEStop,
  onResetEStop,
  onSetAllWheels,
}: GamepadDriveProps) {
  const { connected, id, standard } = useGamepad();

  const [layout, setLayout] = useState<DriveLayout>(loadLayout);
  // Live deadman state surfaced for the UI. Refreshed at ~10 Hz so we
  // don't re-render every animation frame.
  const [deadmanHeld, setDeadmanHeld] = useState(false);
  // Throttled mirror of the outgoing twist for the readout row.
  const [twistView, setTwistView] = useState<{ vx: number; wz: number }>({
    vx: 0,
    wz: 0,
  });
  // Most recent logical snapshot, kept for the chord-label hints in
  // the UI so they re-skin to whichever mapping the active pad uses
  // (LB/RB vs L1/R1 etc.).
  const [logicalView, setLogicalView] = useState<LogicalSnapshot | null>(null);

  // Refs that the per-frame subscriber reads. Using refs (rather than
  // closing over state) keeps the subscription alive for the lifetime
  // of the component instead of being torn down on every state change
  // — that would reset the hook's edge-detection state and cause every
  // E-STOP press to fire twice. Same pattern as `GamepadDriver`.
  const wheelsRef = useRef(wheels);
  wheelsRef.current = wheels;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const estopRef = useRef(estopLatched);
  estopRef.current = estopLatched;
  const layoutRef = useRef(layout);
  layoutRef.current = layout;
  const onTwistRef = useRef(onTwist);
  onTwistRef.current = onTwist;
  const onStopRef = useRef(onStop);
  onStopRef.current = onStop;
  const onEStopRef = useRef(onEStop);
  onEStopRef.current = onEStop;
  const onResetEStopRef = useRef(onResetEStop);
  onResetEStopRef.current = onResetEStop;
  const onSetAllWheelsRef = useRef(onSetAllWheels);
  onSetAllWheelsRef.current = onSetAllWheels;

  // Drive-session state, all subscriber-owned:
  //
  //   * `drivingRef` — a drive cycle is open (deadman held + gated on);
  //     flipping false→true on the first dispatched twist and back on
  //     the first stop, so the parent's `onStop` fires exactly once
  //     per release rather than every tick.
  //   * `lastSentRef` — last dispatched twist; invariant: non-null iff
  //     a drive cycle is open. Filters sub-epsilon re-sends.
  //   * `lastTickAtRef` — timestamp of the most recent gamepad tick,
  //     feeds the watchdog below.
  const drivingRef = useRef(false);
  const lastSentRef = useRef<{ vx: number; wz: number } | null>(null);
  const lastTickAtRef = useRef(0);
  const lastUiAtRef = useRef(0);

  // Close the current drive cycle, if any. Idempotent per cycle: the
  // first call after release enqueues the (0, 0) twist through the
  // parent's stop path (which supersedes anything mid-flight), and
  // later no-op calls are swallowed.
  const stopDriving = useCallback(() => {
    if (!drivingRef.current) return;
    drivingRef.current = false;
    lastSentRef.current = null;
    onStopRef.current();
  }, []);

  // Per-frame subscription. On teardown — pad disconnect or unmount —
  // halt the chassis if a drive cycle was open.
  useEffect(() => {
    if (!connected) return;
    const unsub = subscribeGamepad((snap) => onTick(snap));
    return () => {
      unsub();
      stopDriving();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  // Watchdog: if we're mid-drive and the RAF ticks dry up, stop. The
  // firmware holds the last commanded twist until changed, so this is
  // the backstop that guarantees the chassis halts even when the poll
  // loop dies without a disconnect event (backgrounded WebView, pad
  // battery pulling mid-drive). `stopDriving` self-guards, so calling
  // it while idle is a no-op.
  useEffect(() => {
    if (!connected) return;
    const iv = window.setInterval(() => {
      if (performance.now() - lastTickAtRef.current > WATCHDOG_MS) {
        stopDriving();
      }
    }, 100);
    return () => window.clearInterval(iv);
  }, [connected, stopDriving]);

  // Backgrounding the tab freezes the RAF loop (and browsers throttle
  // timers, which would delay the watchdog); halt immediately instead.
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) stopDriving();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () =>
      document.removeEventListener("visibilitychange", onVisibility);
  }, [stopDriving]);

  function onTick(snap: GamepadSnapshot) {
    const now = performance.now();
    lastTickAtRef.current = now;
    const lg = snap.logical;

    // ----- e-stop / clear (rising-edge on East / South) ---------------
    // Same bindings as the dial-in bridge so muscle memory transfers.
    if (lg.estopEdge && !estopRef.current) {
      onEStopRef.current();
    }
    if (lg.resetEStopEdge && estopRef.current) {
      onResetEStopRef.current();
    }

    // ----- arm / disarm all wheels (rising-edge on L3) ----------------
    // Mirrors the card's "Enable wheels" / "Disable wheels" buttons:
    // arms everything unless all are already armed. Gated on mode +
    // E-STOP the same way the buttons are; a press at the wrong time
    // just does nothing rather than surfacing a firmware rejection.
    const wheels = wheelsRef.current;
    const armedCount = wheels.filter((w) => w.armed).length;
    if (
      lg.armToggleEdge &&
      wheels.length > 0 &&
      !estopRef.current &&
      (modeRef.current === "DIAL_IN" || modeRef.current === "RUN_POLICY")
    ) {
      onSetAllWheelsRef.current(armedCount !== wheels.length);
    }

    // ----- twist streaming ---------------------------------------------
    // Drive only while the deadman is held and the same gates as the
    // on-screen joystick pass: no E-STOP, a drive-capable mode, and at
    // least one armed wheel. Trigger pressure scales the request —
    // just-clearing the deadman threshold creeps (~64% of the soft
    // limit), a full pull reaches it — mirroring the dial-in bridge's
    // "trigger as gain" feel.
    const trigger = lg.deadman;
    const deadman = trigger >= DEADMAN_THRESHOLD;
    const canDrive =
      deadman &&
      !estopRef.current &&
      (modeRef.current === "DIAL_IN" || modeRef.current === "RUN_POLICY") &&
      armedCount > 0;

    // Stick convention matches the on-screen pad: up = forward,
    // right = turn right (+wz = left turn, hence the sign flip).
    // Limits come from the active profile; read per tick so a picker
    // change applies to the very next frame.
    const profile = getActiveControlProfile();
    const gain = 0.4 + 0.6 * Math.min(1, Math.max(0, trigger));
    const turnStick = layoutRef.current === "arcade" ? snap.lx : snap.rx;
    const vx = canDrive ? snap.ly * profile.maxLinear * gain : 0;
    const wz = canDrive ? -turnStick * profile.maxAngular * gain : 0;

    if (canDrive) {
      const last = lastSentRef.current;
      if (
        last === null ||
        Math.abs(last.vx - vx) > TWIST_EPS ||
        Math.abs(last.wz - wz) > TWIST_EPS
      ) {
        lastSentRef.current = { vx, wz };
        drivingRef.current = true;
        onTwistRef.current(vx, wz);
      }
    } else {
      stopDriving();
    }

    // ----- throttled UI updates ----------------------------------------
    if (now - lastUiAtRef.current > UI_INTERVAL_MS) {
      lastUiAtRef.current = now;
      setDeadmanHeld((prev) => (prev !== deadman ? deadman : prev));
      setTwistView((prev) =>
        Math.abs(prev.vx - vx) > 0.02 || Math.abs(prev.wz - wz) > 0.02
          ? { vx, wz }
          : prev,
      );
      setLogicalView((prev) =>
        prev === null || prev.mappingName !== lg.mappingName ? lg : prev,
      );
    }
  }

  const chooseLayout = useCallback((next: DriveLayout) => {
    setLayout(next);
    try {
      window.localStorage.setItem(LAYOUT_STORAGE_KEY, next);
    } catch {
      /* localStorage may be disabled — preference just won't persist */
    }
    // Force the next tick to re-send under the new mapping even if the
    // composed twist happens to be numerically unchanged (e.g. both
    // turn sticks idling at 0 while driving straight).
    lastSentRef.current = null;
  }, []);

  const friendlyName = useMemo(() => prettifyGamepadId(id), [id]);

  if (!connected) return null;

  // Keep the card visible regardless of mode so the operator notices a
  // controller is attached even from Idle. The "what the buttons do"
  // hints clarify when nothing will actually move.
  const armedWheelCount = wheels.filter((w) => w.armed).length;
  const driveBlockReason = estopLatched
    ? "E-STOP latched"
    : mode !== "DIAL_IN" && mode !== "RUN_POLICY"
      ? "switch to Dial-in (or Policy) mode"
      : armedWheelCount === 0
        ? "enable the wheels"
        : null;

  return (
    <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3.5 py-2.5 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-accent/15 text-accent shrink-0"
            aria-hidden
          >
            <ControllerIcon />
          </span>
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-wider text-text-dim font-semibold">
              Bluetooth controller · drive
            </div>
            <div className="text-sm font-semibold truncate" title={id}>
              {friendlyName}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {logicalView ? (
            <span
              className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded font-semibold border ${
                standard
                  ? "bg-bg-elev-2 text-text-dim border-border"
                  : "bg-yellow-500/15 text-yellow-700 dark:text-yellow-300 border-yellow-500/40"
              }`}
              title={
                standard
                  ? "Browser reports the W3C standard gamepad layout; button indices match the hints below verbatim."
                  : "Browser reports a non-standard layout. The driver routes inputs through a D-input fallback (8BitDo / Android-style) — try the chord labels below; if anything still feels off, switch the pad to X-input mode (8BitDo: hold START + Y for 3 s)."
              }
            >
              layout: {logicalView.mappingName}
            </span>
          ) : null}
          <span
            className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded font-semibold border ${
              deadmanHeld
                ? "bg-success/15 text-success border-success/40"
                : "bg-bg-elev-2 text-text-dim border-border"
            }`}
            title="Right trigger held = deadman engaged; sticks can drive. Trigger pressure scales the requested speed."
          >
            <span className="opacity-70 mr-0.5" aria-hidden>
              {deadmanHeld ? "●" : "○"}
            </span>
            {deadmanHeld ? "deadman held" : "deadman released"}
          </span>
        </div>
      </div>

      {/* Layout toggle + control profile + live twist readout. The
          readout shows what the pad is *requesting* (local), so it
          stays responsive even when telemetry lags; the DriveCard
          shows what the firmware acked. */}
      <div className="flex items-center gap-3 flex-wrap text-[12px]">
        <div
          className="flex items-center gap-1 p-0.5 rounded-lg border border-border bg-bg-elev-2"
          title="Split: left stick ↕ = forward/back, right stick ↔ = turn. Arcade: the left stick does both."
        >
          <LayoutOption
            active={layout === "split"}
            onClick={() => chooseLayout("split")}
          >
            Split
          </LayoutOption>
          <LayoutOption
            active={layout === "arcade"}
            onClick={() => chooseLayout("arcade")}
          >
            Arcade
          </LayoutOption>
        </div>
        <ControlProfilePicker />
        <span className="text-text-dim">
          cmd{" "}
          <span className="font-mono text-text">{fmt(twistView.vx)}</span> m/s
          · ω <span className="font-mono text-text">{fmt(twistView.wz)}</span>{" "}
          rad/s
        </span>
        {driveBlockReason ? (
          <span
            className="text-[11px] text-yellow-700 dark:text-yellow-300"
            title="Reason gamepad input will be ignored even with the deadman held"
          >
            blocked: {driveBlockReason}
          </span>
        ) : null}
      </div>

      {/* Compact button hints. Chord labels come from the active
          mapping so the text reads "RT" on a standard pad and "R2" on
          a D-input one — matches what's printed on the user's
          controller. */}
      <div className="text-[11px] text-text-dim flex flex-wrap gap-x-3 gap-y-1">
        <Hint
          chord={layout === "arcade" ? "L-stick" : "L-stick ↕ + R-stick ↔"}
          label="drive"
        />
        <Hint
          chord={`${logicalView?.chords.deadman ?? "RT"} (hold)`}
          label="deadman · speed"
        />
        <Hint
          chord={logicalView?.chords.armToggle ?? "L3"}
          label="arm/disarm wheels"
        />
        <Hint chord={logicalView?.chords.estop ?? "B / Circle"} label="E-STOP" />
        <Hint
          chord={logicalView?.chords.resetEStop ?? "A / Cross"}
          label="reset E-STOP"
        />
      </div>
    </div>
  );
}

function LayoutOption({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`px-2 py-0.5 rounded-md text-[11px] font-semibold uppercase tracking-wider cursor-pointer ${
        active
          ? "bg-accent/15 text-accent"
          : "bg-transparent text-text-dim hover:text-text"
      }`}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function loadLayout(): DriveLayout {
  try {
    return window.localStorage.getItem(LAYOUT_STORAGE_KEY) === "arcade"
      ? "arcade"
      : "split";
  } catch {
    return "split";
  }
}

function fmt(v: number): string {
  return v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2);
}
