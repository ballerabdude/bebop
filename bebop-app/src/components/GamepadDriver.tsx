// Bluetooth-gamepad → motor-bench bridge.
//
// Lives inside the motor bench so it can:
//
//   * cycle the operator's "active joint" with the shoulder buttons,
//   * use the left stick (Y axis) as a rate source that integrates into
//     the active joint's target position,
//   * gate motion behind the right trigger as a deadman, mirroring the
//     robot-side teleop flow in `controller/teleop.rs`,
//   * latch / clear the runtime E-STOP from the East / South face
//     buttons, again paralleling the agent-side controller bindings so
//     muscle memory transfers between the two flows.
//
// All of this is *client-side*: the gamepad is paired to the phone /
// laptop running bebop-app, not to the robot. Input arrives through
// the Web Gamepad API (`useGamepad`) and is dispatched through the
// existing runtime-WS calls the operator uses for click-and-drag dial-in.
// No firmware or agent change is required.
//
// We don't try to drive *body velocity* (xvel/yvel/angvel) from here —
// that path is owned by the on-robot agent (`controller::teleop::tick`)
// and shipping it from the app would need a new runtime WS message in
// `bebop-linux`. Once that exists, this component can grow a "drive"
// mode that streams velocity at the same cadence as it streams target
// position today.

import { useEffect, useMemo, useRef, useState } from "react";

import { subscribeGamepad, useGamepad } from "../input";
import type { GamepadSnapshot, LogicalSnapshot } from "../input";
import type { MotorView, RuntimeMode } from "../runtime";

/// Maximum target rate at full stick deflection + full trigger
/// pressure (rad/s). The hook polls at RAF (~60 Hz on most platforms)
/// and integrates delta-time, so the actual on-wire rate matches this
/// regardless of polling jitter.
///
/// Coupled with the firmware's `slew.max_pos_step_per_tick` in
/// `firmware/bebop-linux/config/bebop_v2.yaml`. The firmware enforces
/// a hard ceiling of `max_pos_step_per_tick × 100 Hz`; values here
/// above that ceiling get silently clamped per tick. The default
/// firmware cap is `0.005` rad/tick → `0.5` rad/s; this constant sits
/// above that intentionally so the rate is responsive on a robot
/// running a relaxed slew cap, and harmlessly clamps on a stock one.
/// See the README's "Tuning the dial-in rate" section.
const FULL_RATE_RAD_S = 2.0;

interface GamepadDriverProps {
  motors: MotorView[];
  mode: RuntimeMode;
  estopLatched: boolean;
  /// Send a new commanded target for `jointName`. Throttling +
  /// in-flight coalescing is the parent's responsibility — same
  /// pattern as the click-drag dial slider.
  onSetTarget: (jointName: string, value: number) => void;
  onEStop: () => void;
  onResetEStop: () => void;
  /// Optional callback fired when the user presses ThumbL (L3) so the
  /// parent can toggle the selected motor's armed state without the
  /// user having to leave the controller. The motor bench wires this
  /// in but a smaller embedder may leave it undefined.
  onToggleArm?: (jointName: string, enabled: boolean) => void;
}

/// Renders a compact status card and forwards gamepad input to the
/// runtime over the parent's callbacks. Returns null whenever no pad
/// is connected so the bench layout doesn't reserve space for it.
export function GamepadDriver({
  motors,
  mode,
  estopLatched,
  onSetTarget,
  onEStop,
  onResetEStop,
  onToggleArm,
}: GamepadDriverProps) {
  const { connected, id, standard } = useGamepad();

  // Selected joint index. Persists across snapshots; clamped to the
  // current motor list length below so a motor disappearing doesn't
  // leave us pointing past the end of the array.
  const [selectedIdx, setSelectedIdx] = useState(0);
  // Live deadman state surfaced for the UI. Mirrors the logical
  // deadman pressure in the most recent snapshot — kept as React
  // state so we re-render when the user starts/stops holding the
  // trigger.
  const [armed, setArmed] = useState(false);
  // Live stick deflection on Y for the visual indicator. We don't
  // re-render at RAF rate; instead we throttle the visual to ~10 Hz
  // (handled below).
  const [stickY, setStickY] = useState(0);
  // Most recent logical snapshot, kept for the chord-label hints in
  // the UI. Refreshed at ~10 Hz alongside `armed` / `stickY` so the
  // hints can re-skin to whichever mapping the active pad uses
  // (LB/RB vs L1/R1 etc.).
  const [logicalView, setLogicalView] = useState<LogicalSnapshot | null>(null);

  // Refs that the per-frame subscriber reads. Using refs (rather than
  // closing over state) means the subscriber can stay alive for the
  // lifetime of the component without being torn down on every state
  // change — that would otherwise reset the edge-detection state and
  // cause every E-STOP press to fire twice.
  const motorsRef = useRef(motors);
  motorsRef.current = motors;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const estopRef = useRef(estopLatched);
  estopRef.current = estopLatched;
  const selectedRef = useRef(selectedIdx);
  selectedRef.current = selectedIdx;
  const onSetTargetRef = useRef(onSetTarget);
  onSetTargetRef.current = onSetTarget;
  const onEStopRef = useRef(onEStop);
  onEStopRef.current = onEStop;
  const onResetEStopRef = useRef(onResetEStop);
  onResetEStopRef.current = onResetEStop;
  const onToggleArmRef = useRef(onToggleArm);
  onToggleArmRef.current = onToggleArm;

  // Local target accumulator. Initialised from `motor.target` on the
  // first tick the user starts driving a joint, then integrated by the
  // stick. We can't read `motor.target` straight from the snapshot
  // every tick because telemetry-driven updates would fight the user's
  // input (jitter + tracking lag would show up as a stuck / vibrating
  // thumb). Resetting to telemetry happens whenever the user releases
  // the deadman or switches joints.
  const targetRef = useRef<number | null>(null);
  // Tracks which joint the targetRef belongs to so we can detect
  // "user cycled to a different joint" and reseed.
  const targetJointRef = useRef<string | null>(null);
  // For dt integration — `performance.now()` of the last tick that
  // observed the deadman held.
  const lastDriveAtRef = useRef<number | null>(null);

  // Throttle UI-only state updates (deadman pill + stick indicator)
  // to ~10 Hz so we don't trigger a React render every animation
  // frame. The actual command dispatch still happens at RAF rate.
  const lastUiAtRef = useRef(0);

  useEffect(() => {
    if (!connected) return;
    return subscribeGamepad((snap) => onTick(snap));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  function onTick(snap: GamepadSnapshot) {
    const list = motorsRef.current;
    const now = performance.now();
    const sel = clampSelected(selectedRef.current, list.length);
    const lg = snap.logical;

    // ----- joint cycling (rising-edge on shoulder buttons) ------------
    //
    // Reads the *logical* `prevJoint` / `nextJoint` intents rather
    // than raw button indices, so the bindings are correct whether
    // the pad reports the W3C standard layout (LB=4, RB=5) or the
    // 8BitDo / generic D-input layout (L1=6, R1=7). The mapping
    // module owns that translation.
    if (lg.prevJointEdge) {
      const next = list.length === 0 ? 0 : (sel - 1 + list.length) % list.length;
      selectedRef.current = next;
      setSelectedIdx(next);
      // Reset the local target accumulator so the new joint starts
      // from its current telemetry value rather than carrying the
      // previous joint's number across.
      targetRef.current = null;
      targetJointRef.current = null;
    } else if (lg.nextJointEdge) {
      const next = list.length === 0 ? 0 : (sel + 1) % list.length;
      selectedRef.current = next;
      setSelectedIdx(next);
      targetRef.current = null;
      targetJointRef.current = null;
    }

    // ----- e-stop / clear (rising-edge on East / South) ---------------
    if (lg.estopEdge && !estopRef.current) {
      onEStopRef.current();
    }
    if (lg.resetEStopEdge && estopRef.current) {
      onResetEStopRef.current();
    }

    // ----- arm/disarm toggle (rising-edge on left thumbstick click) ---
    const motor = list[selectedRef.current];
    if (lg.armToggleEdge && motor && onToggleArmRef.current) {
      onToggleArmRef.current(motor.jointName, !motor.armed);
    }

    // ----- target nudging --------------------------------------------
    const trigger = lg.deadman;
    const stick = snap.ly;
    const deadmanHeld = trigger >= 0.4;
    const driveAllowed =
      deadmanHeld &&
      modeRef.current === "DIAL_IN" &&
      !estopRef.current &&
      motor !== undefined &&
      motor.armed;

    if (driveAllowed && motor) {
      // Reseed the integrator the first tick of a new drive cycle, or
      // when the active joint changes underneath us. We seed from the
      // motor's current commanded target (if any), falling back to
      // the live measured position so the first nudge starts from
      // where the joint actually is rather than from 0.
      if (
        targetJointRef.current !== motor.jointName ||
        targetRef.current === null
      ) {
        targetRef.current = clampNumber(
          Number.isFinite(motor.target) && motor.armed ? motor.target : motor.position,
          motor.posMin,
          motor.posMax,
        );
        targetJointRef.current = motor.jointName;
        lastDriveAtRef.current = now;
      }

      const last = lastDriveAtRef.current ?? now;
      // Cap dt so a long pause (tab backgrounded, breakpoint hit)
      // doesn't translate into a giant jump on the next tick. 50 ms
      // ≈ 3 RAF frames at 60 Hz.
      const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
      lastDriveAtRef.current = now;

      // Stick magnitude maps linearly to a per-second rad rate. The
      // trigger doubles as a "go faster" gain so the user can hold
      // it lightly to creep and pull it deeper for full-rate moves.
      // Scale: half-pulled trigger is creep speed (FULL_RATE * 0.4),
      // full pull is FULL_RATE.
      const triggerGain = 0.4 + 0.6 * Math.min(1, Math.max(0, trigger));
      const delta = stick * FULL_RATE_RAD_S * triggerGain * dt;
      const next = clampNumber(
        targetRef.current + delta,
        motor.posMin,
        motor.posMax,
      );

      // Only dispatch when the value actually changes by ≥1 quantum;
      // otherwise we're flooding the WS with no-op writes whenever
      // the stick is at rest but the deadman is held.
      if (Math.abs(next - targetRef.current) > 1e-6) {
        targetRef.current = next;
        onSetTargetRef.current(motor.jointName, next);
      }
    } else {
      // Deadman released or drive otherwise blocked — drop the
      // integrator so the next press reseeds from telemetry instead
      // of resuming where we left off (which would surprise the
      // operator after a long idle).
      targetRef.current = null;
      targetJointRef.current = null;
      lastDriveAtRef.current = null;
    }

    // ----- throttled UI updates --------------------------------------
    if (now - lastUiAtRef.current > 100) {
      lastUiAtRef.current = now;
      const newArmed = deadmanHeld && !estopRef.current;
      setArmed((prev) => (prev !== newArmed ? newArmed : prev));
      // Snap to 0 visually when stick is in deadzone (already 0 from
      // the hook, but keep the comparison cheap).
      setStickY((prev) => (Math.abs(prev - stick) > 0.02 ? stick : prev));
      // Surface the active mapping's chord text so the hint row can
      // render the labels printed on the user's pad. We only update
      // when the mapping name actually changes — the chord lookup is
      // a stable reference inside a given mapping.
      setLogicalView((prev) =>
        prev === null || prev.mappingName !== lg.mappingName ? lg : prev,
      );
    }
  }

  const motor = motors[clampSelected(selectedIdx, motors.length)];
  const friendlyName = useMemo(() => prettifyGamepadId(id), [id]);

  if (!connected) return null;

  // Keep the card visible regardless of mode so the operator notices a
  // controller is attached even from Idle. The "what the buttons do"
  // hints clarify when nothing will actually move.
  const driveBlockReason = !motor
    ? "no joint selected"
    : mode !== "DIAL_IN"
      ? "switch to Dial-in mode"
      : estopLatched
        ? "E-STOP latched"
        : !motor.armed
          ? "arm the selected joint"
          : null;

  return (
    <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3.5 py-2.5 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-accent/15 text-accent shrink-0" aria-hidden>
            <ControllerIcon />
          </span>
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-wider text-text-dim font-semibold">
              Bluetooth controller
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
              armed
                ? "bg-success/15 text-success border-success/40"
                : "bg-bg-elev-2 text-text-dim border-border"
            }`}
            title="Right trigger held = deadman engaged; the active joint can move."
          >
            <span className="opacity-70 mr-0.5" aria-hidden>{armed ? "●" : "○"}</span>
            {armed ? "deadman held" : "deadman released"}
          </span>
        </div>
      </div>

      {/* Active joint + stick indicator */}
      <div className="flex items-center gap-3 flex-wrap text-[12px]">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-text-dim">Active joint</span>
          {motor ? (
            <>
              <span className="font-mono text-text truncate">{motor.jointName}</span>
              <span
                className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded font-semibold border ${
                  motor.armed
                    ? "bg-success/15 text-success border-success/40"
                    : "bg-bg-elev-2 text-text-dim border-border"
                }`}
                title={motor.armed ? "Joint is armed; gamepad input will move it." : "Joint is disarmed; press L3 / left-stick click to arm."}
              >
                {motor.armed ? "armed" : "disarmed"}
              </span>
            </>
          ) : (
            <span className="text-text-dim italic">no motors reported</span>
          )}
        </div>
        <div className="hidden md:block w-px h-4 bg-border" aria-hidden />
        <div className="flex items-center gap-1.5">
          <span className="text-text-dim">Stick</span>
          <StickIndicator value={stickY} />
        </div>
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
          mapping so the text reads "LB / RB" on a standard pad and
          "L1 / R1" on a D-input one — matches what's printed on the
          user's controller. */}
      <div className="text-[11px] text-text-dim flex flex-wrap gap-x-3 gap-y-1">
        <Hint
          chord={
            logicalView
              ? `${logicalView.chords.prevJoint} / ${logicalView.chords.nextJoint}`
              : "LB / RB"
          }
          label="prev / next joint"
        />
        <Hint chord="L-stick ↕" label="nudge target" />
        <Hint
          chord={`${logicalView?.chords.deadman ?? "RT"} (hold)`}
          label="deadman"
        />
        <Hint
          chord={logicalView?.chords.armToggle ?? "L3"}
          label="arm/disarm joint"
        />
        <Hint
          chord={logicalView?.chords.estop ?? "B / Circle"}
          label="E-STOP"
        />
        <Hint
          chord={logicalView?.chords.resetEStop ?? "A / Cross"}
          label="reset E-STOP"
        />
      </div>
    </div>
  );
}

function Hint({ chord, label }: { chord: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap">
      <kbd className="text-[10px] font-mono px-1 py-0.5 rounded border border-border bg-bg-elev-2 text-text">
        {chord}
      </kbd>
      <span>{label}</span>
    </span>
  );
}

function StickIndicator({ value }: { value: number }) {
  // Vertical bar centred at 0; positive value = upward fill, negative
  // = downward fill. Mirrors how the user perceives the left stick.
  const pct = Math.max(-1, Math.min(1, value));
  const half = Math.abs(pct) * 50; // % of bar height filled from centre
  return (
    <div
      className="relative w-2.5 h-5 rounded-full bg-bg-elev-2 overflow-hidden"
      aria-label={`Left stick Y deflection ${(value * 100).toFixed(0)}%`}
      title={`stick.y = ${value.toFixed(2)}`}
    >
      <div
        className={`absolute left-0 right-0 ${
          pct === 0 ? "" : pct > 0 ? "bottom-1/2 bg-accent" : "top-1/2 bg-accent"
        }`}
        style={{ height: `${half}%` }}
      />
      <div
        className="absolute left-0 right-0 top-1/2 h-px bg-text-dim/50 -translate-y-1/2"
        aria-hidden
      />
    </div>
  );
}

function ControllerIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M6 12h4M8 10v4" />
      <circle cx="15" cy="11" r="0.8" fill="currentColor" />
      <circle cx="17" cy="13" r="0.8" fill="currentColor" />
      <path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59L2 13.5A2.5 2.5 0 0 0 4.5 16h.5a2 2 0 0 0 1.789-1.106L7.62 13.5h8.76l.831 1.394A2 2 0 0 0 19 16h.5a2.5 2.5 0 0 0 2.5-2.5l-.702-4.91A4 4 0 0 0 17.32 5Z" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function clampSelected(idx: number, len: number): number {
  if (len === 0) return 0;
  if (idx < 0) return 0;
  if (idx >= len) return len - 1;
  return idx;
}

function clampNumber(v: number, lo: number, hi: number): number {
  if (Number.isNaN(v)) return lo;
  return v < lo ? lo : v > hi ? hi : v;
}

/// Best-effort: reduce the verbose `Gamepad.id` ("8BitDo Pro 2 (STANDARD
/// GAMEPAD Vendor: 2dc8 Product: 6003)") down to just the human name.
/// Browsers vary in how they format this; we strip any trailing
/// parenthesised section, which catches Chromium and Safari.
function prettifyGamepadId(id: string): string {
  if (!id) return "Gamepad";
  const open = id.indexOf("(");
  return (open > 0 ? id.slice(0, open) : id).trim();
}
