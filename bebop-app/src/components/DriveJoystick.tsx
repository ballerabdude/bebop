// On-screen differential-drive joystick, shared by the motor bench and
// the teleop screen. Drag the knob from the centre: up/down maps to
// forward/reverse, left/right to turn. Release snaps back and stops.
//
// Touch only: WASD / arrows and the gamepad are handled by the drive-input
// arbiter (`src/input/useDriveInput.ts`), which this reports into through
// `onTwist` / `onStop` (pass `drive.touch.onTwist` / `drive.touch.onStop`).
//
// Touch handling: the pad captures the pointer that started the drag (a
// second finger is ignored), `touch-action: none` keeps the page from
// scrolling or zooming while dragging, and every way a drag can end
// without a pointerup still reports a stop:
//   * pointer released / cancelled / capture lost  -> onStop
//   * window blur mid-drag                         -> onStop
//   * `disabled` flips on mid-drag (E-STOP, mode)   -> onStop
//   * unmount mid-drag                             -> onStop
// A small centre dead zone reads as neutral, so sliding back to the
// middle counts as "released" for the arbiter's re-engage rule.

import { useCallback, useEffect, useRef, useState } from "react";

import { getActiveControlProfile } from "../input";

/// Smallest pad that stays usable with a thumb.
export const JOYSTICK_MIN_SIZE = 160;
/// Default pad diameter, px.
export const JOYSTICK_DEFAULT_SIZE = 200;
/// Knob diameter floor: a comfortable touch target.
const KNOB_MIN = 48;
/// Normalised radius treated as neutral.
const CENTER_DEADZONE = 0.06;

export function DriveJoystick({
  onTwist,
  onStop,
  disabled,
  size = JOYSTICK_DEFAULT_SIZE,
  showHint = true,
}: {
  onTwist: (vx: number, wz: number) => void;
  onStop: () => void;
  disabled: boolean;
  /// Pad diameter in px; clamped to at least `JOYSTICK_MIN_SIZE`.
  size?: number;
  /// Show the "drag to drive" caption under the pad.
  showHint?: boolean;
}) {
  const diameter = Math.max(JOYSTICK_MIN_SIZE, Math.round(size));
  const knob = Math.max(KNOB_MIN, Math.round(diameter * 0.28));
  const travel = (diameter - knob) / 2;

  const padRef = useRef<HTMLDivElement | null>(null);
  /// Pointer id of the active drag, or null.
  const pointerRef = useRef<number | null>(null);
  const [pos, setPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  /// The drag is resting in the centre dead zone and neutral was already
  /// reported: further moves inside it report nothing (each neutral report
  /// can put a (0, 0) frame on the wire while this client holds the seat).
  const inDeadzoneRef = useRef(false);

  const onTwistRef = useRef(onTwist);
  onTwistRef.current = onTwist;
  const onStopRef = useRef(onStop);
  onStopRef.current = onStop;

  const release = useCallback(() => {
    if (pointerRef.current === null) return;
    const pad = padRef.current;
    const id = pointerRef.current;
    pointerRef.current = null;
    inDeadzoneRef.current = false;
    try {
      if (pad?.hasPointerCapture(id)) pad.releasePointerCapture(id);
    } catch {
      /* pointer already gone */
    }
    setPos({ x: 0, y: 0 });
    onStopRef.current();
  }, []);

  const handleMove = useCallback(
    (clientX: number, clientY: number) => {
      const pad = padRef.current;
      if (!pad) return;
      const rect = pad.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const radius = Math.max(1, (rect.width - knob) / 2);
      let nx = (clientX - cx) / radius;
      let ny = (clientY - cy) / radius;
      const mag = Math.hypot(nx, ny);
      if (mag > 1) {
        nx /= mag;
        ny /= mag;
      }
      setPos({ x: nx, y: ny });
      if (mag < CENTER_DEADZONE) {
        if (!inDeadzoneRef.current) {
          inDeadzoneRef.current = true;
          onTwistRef.current(0, 0);
        }
        return;
      }
      inDeadzoneRef.current = false;
      // Up = forward, right = turn right (+wz = left turn).
      const profile = getActiveControlProfile();
      onTwistRef.current(-ny * profile.maxLinear + 0, -nx * profile.maxAngular + 0);
    },
    [knob],
  );

  // Disabled mid-drag: end the gesture.
  useEffect(() => {
    if (disabled) release();
  }, [disabled, release]);

  // Focus leaving the window can swallow pointerup; unmount never sees it.
  useEffect(() => {
    const onBlur = () => release();
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("blur", onBlur);
      release();
    };
  }, [release]);

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        ref={padRef}
        role="application"
        aria-label="Drive joystick"
        aria-disabled={disabled}
        className={`relative shrink-0 rounded-full border border-border bg-bg-elev-2/60 select-none touch-none overscroll-contain [-webkit-touch-callout:none] [-webkit-user-select:none] ${
          disabled ? "opacity-40 cursor-not-allowed" : "cursor-grab active:cursor-grabbing"
        }`}
        style={{ width: diameter, height: diameter }}
        onPointerDown={(e) => {
          if (disabled || pointerRef.current !== null) return;
          if (e.pointerType === "mouse" && e.button !== 0) return;
          e.preventDefault();
          pointerRef.current = e.pointerId;
          try {
            e.currentTarget.setPointerCapture(e.pointerId);
          } catch {
            /* capture is best-effort */
          }
          handleMove(e.clientX, e.clientY);
        }}
        onPointerMove={(e) => {
          if (e.pointerId !== pointerRef.current) return;
          e.preventDefault();
          handleMove(e.clientX, e.clientY);
        }}
        onPointerUp={(e) => {
          if (e.pointerId === pointerRef.current) release();
        }}
        onPointerCancel={(e) => {
          if (e.pointerId === pointerRef.current) release();
        }}
        onLostPointerCapture={(e) => {
          if (e.pointerId === pointerRef.current) release();
        }}
        onContextMenu={(e) => e.preventDefault()}
      >
        {/* Cardinal markers */}
        <span className="pointer-events-none absolute left-1/2 top-2 -translate-x-1/2 text-[10px] text-text-dim">
          fwd
        </span>
        <span className="pointer-events-none absolute left-1/2 bottom-2 -translate-x-1/2 text-[10px] text-text-dim">
          rev
        </span>
        <span className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-[10px] text-text-dim">
          L
        </span>
        <span className="pointer-events-none absolute top-1/2 right-2 -translate-y-1/2 text-[10px] text-text-dim">
          R
        </span>
        {/* Knob */}
        <div
          className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow-md"
          style={{
            width: knob,
            height: knob,
            left: `calc(50% + ${(pos.x * travel).toFixed(1)}px)`,
            top: `calc(50% + ${(pos.y * travel).toFixed(1)}px)`,
          }}
          aria-hidden
        />
      </div>
      {showHint ? (
        <div className="text-[11px] text-text-dim">
          Drag to drive
          <span className="hidden pointer-fine:inline"> · WASD / arrows</span>
        </div>
      ) : null}
    </div>
  );
}
