// Gamepad status card and button actions for wheeled teleop.
//
// Driving itself lives in the drive-input arbiter
// (`src/input/useDriveInput.ts`): it polls the pad, applies the RT
// deadman (`DEADMAN_THRESHOLD`, 0.4) and merges the sticks with the
// on-screen joystick and WASD into the one twist stream, along with the
// stop rules (deadman release, pad drop, blur / hidden tab, re-engage
// after a halt). This card shows the pad, the deadman, what the sticks
// request and why driving is blocked, picks the stick layout and profile,
// and handles the pad's buttons:
//
//   * sticks drive the chassis (layout selectable below),
//   * RT is a deadman; release to halt; trigger pressure scales speed,
//   * B latches E-STOP, A clears it,
//   * L3 arms / disarms every wheel.

import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import {
  DEADMAN_THRESHOLD,
  getActiveControlProfile,
  gamepadTwist,
  getGamepadLayout,
  setGamepadLayout,
  subscribeGamepad,
  useGamepad,
  useGamepadLayout,
} from "../input";
import type { GamepadSnapshot, LogicalSnapshot } from "../input";
import type { RuntimeMode, WheelView } from "../runtime";
import { ControllerIcon, Hint, prettifyGamepadId } from "./gamepadBits";
import { ControlProfilePicker } from "./ControlProfilePicker";

/// UI-only state (deadman pill, twist readout) refreshes at ~10 Hz
/// instead of RAF rate.
const UI_INTERVAL_MS = 100;

interface GamepadDriveProps {
  wheels: WheelView[];
  mode: RuntimeMode;
  estopLatched: boolean;
  /// @deprecated Ignored: the pad drives through `useDriveInput`.
  onTwist?: (vx: number, wz: number) => void;
  /// @deprecated Ignored: the pad drives through `useDriveInput`.
  onStop?: () => void;
  onEStop: () => void;
  onResetEStop: () => void;
  /// Arm / disarm every wheel (L3 toggles between the two).
  onSetAllWheels: (enabled: boolean) => void;
}

/// Renders a compact status card and handles the pad's buttons. Returns
/// null whenever no pad is connected so the layout doesn't reserve space.
export function GamepadDrive({
  wheels,
  mode,
  estopLatched,
  onEStop,
  onResetEStop,
  onSetAllWheels,
}: GamepadDriveProps) {
  const { connected, id, standard } = useGamepad();
  const layout = useGamepadLayout();

  const [deadmanHeld, setDeadmanHeld] = useState(false);
  // Throttled mirror of what the sticks request (local; the drive card
  // shows what the firmware reports).
  const [twistView, setTwistView] = useState<{ vx: number; wz: number }>({
    vx: 0,
    wz: 0,
  });
  // Most recent logical snapshot, for the chord-label hints (LB/RB vs
  // L1/R1 etc.).
  const [logicalView, setLogicalView] = useState<LogicalSnapshot | null>(null);

  // Refs read by the per-frame subscriber, so the subscription lives for
  // the component's lifetime (re-subscribing would reset edge detection
  // and fire an E-STOP press twice).
  const wheelsRef = useRef(wheels);
  wheelsRef.current = wheels;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const estopRef = useRef(estopLatched);
  estopRef.current = estopLatched;
  const onEStopRef = useRef(onEStop);
  onEStopRef.current = onEStop;
  const onResetEStopRef = useRef(onResetEStop);
  onResetEStopRef.current = onResetEStop;
  const onSetAllWheelsRef = useRef(onSetAllWheels);
  onSetAllWheelsRef.current = onSetAllWheels;
  const lastUiAtRef = useRef(0);

  useEffect(() => {
    if (!connected) return;
    return subscribeGamepad((snap) => onTick(snap));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  function onTick(snap: GamepadSnapshot) {
    const now = performance.now();
    const lg = snap.logical;

    // ----- e-stop / clear (rising edge on East / South) ---------------
    if (lg.estopEdge && !estopRef.current) {
      onEStopRef.current();
    }
    if (lg.resetEStopEdge && estopRef.current) {
      onResetEStopRef.current();
    }

    // ----- arm / disarm all wheels (rising edge on L3) ----------------
    // Mirrors the "Enable wheels" / "Disable wheels" buttons, gated on
    // mode + E-STOP the same way; a press at the wrong time does nothing.
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

    // ----- throttled UI updates ----------------------------------------
    if (now - lastUiAtRef.current > UI_INTERVAL_MS) {
      lastUiAtRef.current = now;
      const deadman = lg.deadman >= DEADMAN_THRESHOLD;
      const t = gamepadTwist(
        { lx: snap.lx, ly: snap.ly, rx: snap.rx, deadman: lg.deadman },
        getGamepadLayout(),
        getActiveControlProfile(),
      ) ?? { vx: 0, wz: 0 };
      setDeadmanHeld((prev) => (prev !== deadman ? deadman : prev));
      setTwistView((prev) =>
        Math.abs(prev.vx - t.vx) > 0.02 || Math.abs(prev.wz - t.wz) > 0.02 ? t : prev,
      );
      setLogicalView((prev) =>
        prev === null || prev.mappingName !== lg.mappingName ? lg : prev,
      );
    }
  }

  const chooseLayout = setGamepadLayout;

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
  const effectiveLayout = layout;

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
              Gamepad · drive
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
                  : "bg-yellow-500/15 text-yellow-300 border-yellow-500/40"
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
        {(
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
        )}
        <ControlProfilePicker />
        <span className="text-text-dim">
          cmd{" "}
          <span className="font-mono text-text">{fmt(twistView.vx)}</span> m/s
          · ω <span className="font-mono text-text">{fmt(twistView.wz)}</span>{" "}
          rad/s
        </span>
        {driveBlockReason ? (
          <span
            className="text-[11px] text-yellow-300"
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
          chord={effectiveLayout === "arcade" ? "L-stick" : "L-stick ↕ + R-stick ↔"}
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
      className={`px-2.5 py-1 min-h-8 rounded-md text-[11px] font-semibold uppercase tracking-wider cursor-pointer ${
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

function fmt(v: number): string {
  return v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2);
}
