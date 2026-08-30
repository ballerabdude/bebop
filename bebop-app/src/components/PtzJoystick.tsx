// Virtual PTZ joystick, shared by the video screen and the teleop
// screen. Drag the knob from the centre: right/left ramps pan, up/down
// ramps tilt, at a rate proportional to deflection (full throw = the
// `maxRates` prop). Release snaps the knob back and simply stops
// sending — the gimbal is position-controlled and holds its last
// commanded pose.
//
// The `keys` prop selects the keyboard chord that composes the same
// rates: WASD + arrows on the video screen, I/J/K/L on the teleop
// screen (where WASD + arrows drive the chassis). Releasing the last
// key holds the pose. Bindings are dropped while `disabled` — unlike
// drive commands (firmware mode-gated), PTZ poses go through on any
// connection, so a dead-transport screen must not leak held-key rates
// into it.

import { useCallback, useEffect, useRef, useState } from "react";

export interface PtzKeyset {
  up: string[];
  down: string[];
  left: string[];
  right: string[];
}

/// Video-screen chord: WASD plus arrow keys.
export const PTZ_KEYS_WASD: PtzKeyset = {
  up: ["w", "arrowup"],
  down: ["s", "arrowdown"],
  left: ["a", "arrowleft"],
  right: ["d", "arrowright"],
};

/// Teleop-screen chord: I/J/K/L. WASD + arrows drive the chassis there,
/// so the camera gets its own four keys (same home-row position the
/// right stick occupies on a gamepad).
export const PTZ_KEYS_IJKL: PtzKeyset = {
  up: ["i"],
  down: ["k"],
  left: ["j"],
  right: ["l"],
};

/// Full-throw pan/tilt rates in deg/s. Defaults mirror the OBSBOT
/// Tiny 2 slew (~65°/s pan at default UVC speeds) — see `useCameraPtz`.
export const PTZ_DEFAULT_RATES = { pan: 60, tilt: 40 };

export function PtzJoystick({
  onRate,
  onStop,
  disabled,
  keys = PTZ_KEYS_WASD,
  maxRates = PTZ_DEFAULT_RATES,
  hint = "Drag to aim · WASD / arrows",
}: {
  onRate: (panRate: number, tiltRate: number) => void;
  onStop: () => void;
  disabled: boolean;
  keys?: PtzKeyset;
  maxRates?: { pan: number; tilt: number };
  hint?: string;
}) {
  const padRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);
  const [knob, setKnob] = useState({ x: 0, y: 0 });

  // Latest callbacks in refs so the keyboard cleanup below stays bound
  // for the component's whole lifetime instead of being torn down on
  // every parent render.
  const onRateRef = useRef(onRate);
  onRateRef.current = onRate;
  const onStopRef = useRef(onStop);
  onStopRef.current = onStop;

  const apply = useCallback((nx: number, ny: number) => {
    // Joystick convention: right = pan right (+pan), up = tilt up
    // (+tilt). The pad's ny grows downward, hence the negation.
    setKnob({ x: nx, y: ny });
    onRateRef.current(nx * maxRates.pan, -ny * maxRates.tilt);
  }, [maxRates]);

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

  // Keyboard PTZ: the active keyset composes a rate; releasing the last
  // key holds the pose. Bound only while enabled.
  useEffect(() => {
    if (disabled) return;
    const held = new Set<string>();

    const compute = () => {
      let panRate = 0;
      let tiltRate = 0;
      if (keys.left.some((k) => held.has(k))) panRate -= maxRates.pan;
      if (keys.right.some((k) => held.has(k))) panRate += maxRates.pan;
      if (keys.up.some((k) => held.has(k))) tiltRate += maxRates.tilt;
      if (keys.down.some((k) => held.has(k))) tiltRate -= maxRates.tilt;
      return { panRate, tiltRate };
    };

    const isPtzKey = (k: string) =>
      keys.up.includes(k) ||
      keys.down.includes(k) ||
      keys.left.includes(k) ||
      keys.right.includes(k);

    const onKeyDown = (e: globalThis.KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (!isPtzKey(k)) return;
      e.preventDefault();
      held.add(k);
      const { panRate, tiltRate } = compute();
      onRateRef.current(panRate, tiltRate);
    };
    const onKeyUp = (e: globalThis.KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (!isPtzKey(k)) return;
      held.delete(k);
      if (held.size === 0) {
        onStopRef.current();
      } else {
        const { panRate, tiltRate } = compute();
        onRateRef.current(panRate, tiltRate);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      // Keys held at teardown (component unmount or `disabled` flipping
      // on mid-jog) never see their keyup — hold the pose from here.
      if (held.size > 0) {
        held.clear();
        onStopRef.current();
      }
    };
  }, [disabled, keys, maxRates]);

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
          tilt+
        </span>
        <span className="absolute left-1/2 bottom-2 -translate-x-1/2 text-[10px] text-text-dim">
          tilt−
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
      <div className="text-[11px] text-text-dim">{hint}</div>
    </div>
  );
}
