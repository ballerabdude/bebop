// Virtual differential-drive joystick, shared by the motor bench and
// the teleop screen. Drag the knob from the centre: up/down maps to
// forward/reverse, left/right to turn. Release snaps back and stops the
// robot. WASD / arrow keys do the same for keyboard operators, and
// releasing all drive keys also stops.
//
// Safety contract with the parent (which owns the coalesced twist
// sender): every way a gesture can end without further pointer/key
// events still enqueues a stop —
//   * pointer released / cancelled            → onStop
//   * last drive key released                  → onStop
//   * keyboard drive active at unmount        → onStop (effect cleanup)
//   * pointer drag active at unmount           → onStop (effect cleanup)
//   * `disabled` flips on mid-gesture (E-STOP,
//     mode change, screen switch)             → onStop for the active
//    gesture, and keyboard bindings are dropped entirely.
// The parent's `onStop` must supersede anything mid-flight.

import { useCallback, useEffect, useRef, useState } from "react";

import { getActiveControlProfile } from "../input";

export function DriveJoystick({
  onTwist,
  onStop,
  disabled,
}: {
  onTwist: (vx: number, wz: number) => void;
  onStop: () => void;
  disabled: boolean;
}) {
  const padRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);
  const [knob, setKnob] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Latest callbacks in refs so the keyboard/unmount cleanups below can
  // stay bound for the component's whole lifetime instead of being
  // torn down on every parent render.
  const onTwistRef = useRef(onTwist);
  onTwistRef.current = onTwist;
  const onStopRef = useRef(onStop);
  onStopRef.current = onStop;

  const apply = useCallback((nx: number, ny: number) => {
    // Joystick convention: up = forward, right = turn right (+wz = left turn).
    const profile = getActiveControlProfile();
    const vx = -ny * profile.maxLinear;
    const wz = -nx * profile.maxAngular;
    setKnob({ x: nx, y: ny });
    onTwistRef.current(vx, wz);
  }, []);

  const release = useCallback(() => {
    draggingRef.current = false;
    setKnob({ x: 0, y: 0 });
    onStopRef.current();
  }, []);

  const handleMove = useCallback(
    (clientX: number, clientY: number) => {
      const pad = padRef.current;
      if (!pad) return;
      const rect = pad.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const radius = rect.width / 2;
      let nx = (clientX - cx) / radius;
      let ny = (clientY - cy) / radius;
      const mag = Math.hypot(nx, ny);
      if (mag > 1) {
        nx /= mag;
        ny /= mag;
      }
      apply(nx, ny);
    },
    [apply],
  );

  // A pointer drag that ends in unmount (screen switched mid-gesture)
  // never sees pointerup — halt from the unmount cleanup instead.
  useEffect(
    () => () => {
      if (draggingRef.current) onStopRef.current();
    },
    [],
  );

  // Keyboard drive: WASD + arrow keys. Held keys compose a twist;
  // releasing the last drive key stops. Bound only while enabled so a
  // not-drivable state (E-STOP, wrong mode) can't leak held-key twists
  // into the transport. Limits come from the active control profile,
  // read per event so a picker change applies immediately.
  useEffect(() => {
    if (disabled) return;
    const keys = new Set<string>();
    const forward = ["w", "arrowup"];
    const back = ["s", "arrowdown"];
    const left = ["a", "arrowleft"];
    const right = ["d", "arrowright"];

    const compute = () => {
      const { maxLinear, maxAngular } = getActiveControlProfile();
      let vx = 0;
      let wz = 0;
      if (forward.some((k) => keys.has(k))) vx += maxLinear;
      if (back.some((k) => keys.has(k))) vx -= maxLinear;
      if (left.some((k) => keys.has(k))) wz += maxAngular;
      if (right.some((k) => keys.has(k))) wz -= maxAngular;
      return { vx, wz };
    };

    const isDriveKey = (k: string) =>
      forward.includes(k) ||
      back.includes(k) ||
      left.includes(k) ||
      right.includes(k);

    const onKeyDown = (e: globalThis.KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (!isDriveKey(k)) return;
      e.preventDefault();
      keys.add(k);
      const { vx, wz } = compute();
      onTwistRef.current(vx, wz);
    };
    const onKeyUp = (e: globalThis.KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (!isDriveKey(k)) return;
      keys.delete(k);
      if (keys.size === 0) {
        onStopRef.current();
      } else {
        const { vx, wz } = compute();
        onTwistRef.current(vx, wz);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      // Keys held at teardown (component unmount or `disabled` flipping
      // on mid-drive) will never see their keyup — stop from here.
      if (keys.size > 0) {
        keys.clear();
        onStopRef.current();
      }
    };
  }, [disabled]);

  const knobPx = padRef.current ? padRef.current.offsetWidth / 2 : 88;

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        ref={padRef}
        className={`relative w-44 h-44 rounded-full border border-border bg-bg-elev-2/60 select-none touch-none ${
          disabled ? "opacity-40 cursor-not-allowed" : "cursor-grab active:cursor-grabbing"
        }`}
        onPointerDown={(e) => {
          if (disabled) return;
          draggingRef.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          handleMove(e.clientX, e.clientY);
        }}
        onPointerMove={(e) => {
          if (!draggingRef.current) return;
          handleMove(e.clientX, e.clientY);
        }}
        onPointerUp={release}
        onPointerCancel={release}
      >
        {/* Cardinal markers */}
        <span className="absolute left-1/2 top-2 -translate-x-1/2 text-[10px] text-text-dim">
          fwd
        </span>
        <span className="absolute left-1/2 bottom-2 -translate-x-1/2 text-[10px] text-text-dim">
          rev
        </span>
        <span className="absolute top-1/2 left-2 -translate-y-1/2 text-[10px] text-text-dim">
          L
        </span>
        <span className="absolute top-1/2 right-2 -translate-y-1/2 text-[10px] text-text-dim">
          R
        </span>
        {/* Knob */}
        <div
          className="absolute w-10 h-10 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow-md"
          style={{
            left: `calc(50% + ${(knob.x * knobPx).toFixed(1)}px)`,
            top: `calc(50% + ${(knob.y * knobPx).toFixed(1)}px)`,
          }}
          aria-hidden
        />
      </div>
      <div className="text-[11px] text-text-dim">
        Drag to drive · WASD / arrows
      </div>
    </div>
  );
}
