// React hook over the Web Gamepad API.
//
// The Gamepad API is poll-based by design (there's a `gamepadconnected`
// event but no per-input events), so we drive a `requestAnimationFrame`
// loop while at least one consumer is mounted. Each tick we:
//
//   1. Walk `navigator.getGamepads()` and pick the first connected pad
//      with the "standard" mapping. The 8BitDo Mobile Controller, Xbox
//      Series, DualSense, Switch Pro, etc. all report standard when
//      paired in a generic HID gamepad mode (X-input on 8BitDo,
//      "Bluetooth" on DualSense). Non-standard pads are still surfaced,
//      but the caller is warned via `snapshot.standard === false`.
//   2. Apply a radial deadzone to the sticks and an axial deadzone to
//      the triggers. Defaults are conservative; callers can override.
//   3. Compute rising-edge bits for buttons so consumers can wire
//      "tap to do X" actions without tracking previous frames.
//   4. Forward the snapshot to all subscribed listeners. We use a
//      pub/sub layout (rather than `setState`-per-tick) because at
//      60 Hz a `setState` per frame triggers a render storm; consumers
//      that need React renders can sample inside a `useState` /
//      `useEffect` themselves.
//
// The hook also exposes a `connected` boolean that *does* trigger
// renders, so a UI showing "controller connected" stays reactive.
//
// Browser support, briefly:
//
//   * Chromium (Tauri WebView2 / WkWebView, Edge, Chrome) — full
//     support since forever.
//   * Safari iOS 14.5+ — supports MFi / Bluetooth gamepads via the
//     standard mapping.
//   * Safari macOS — needs the user to interact with the page once
//     before `navigator.getGamepads()` returns non-null entries; the
//     `gamepadconnected` event still fires though, which is enough to
//     bootstrap our polling loop.

import { useEffect, useRef, useState } from "react";

import {
  pickMapping,
  readAnalogSource,
  readBoolSource,
  type LogicalMapping,
} from "./mapping";
import type { ButtonIndex, GamepadSnapshot, LogicalSnapshot } from "./types";

/// Stick/trigger deadzones. These are intentionally small — the motor
/// bench layers an additional STEP_RAD-scaled rate on top, and the user
/// expects a stick at rest to do absolutely nothing.
const DEFAULT_STICK_DEADZONE = 0.12;
const DEFAULT_TRIGGER_DEADZONE = 0.05;

type Listener = (snap: GamepadSnapshot) => void;

interface PollerState {
  raf: number | null;
  prevPressed: boolean[];
  /// Per-mapping rising-edge memory keyed on the intent name. Lives
  /// outside the per-frame `prevPressed` array because logical intents
  /// can be sourced from analog triggers (where "pressed" is a > 0.5
  /// crossing, not a discrete button bit), and we want the same
  /// rising-edge semantics for both.
  prevLogical: Record<string, boolean>;
  listeners: Set<Listener>;
  connectedListeners: Set<(connected: boolean, snap: GamepadSnapshot | null) => void>;
  /// Last connected pad's index. We pin to it across frames so a
  /// transient `getGamepads()` ordering change (some platforms reorder
  /// when a second pad connects) doesn't swap controllers under the
  /// caller's feet.
  pinnedIndex: number | null;
  lastSnap: GamepadSnapshot | null;
}

// Module-level singleton: one RAF loop, many subscribers. Keeps cost
// flat regardless of how many components subscribe.
const state: PollerState = {
  raf: null,
  prevPressed: [],
  prevLogical: {},
  listeners: new Set(),
  connectedListeners: new Set(),
  pinnedIndex: null,
  lastSnap: null,
};

function applyStickDeadzone(x: number, y: number, dz: number): [number, number] {
  const mag = Math.hypot(x, y);
  if (mag <= dz) return [0, 0];
  // Scale post-deadzone vector back to full range so the user gets a
  // smooth ramp out of the dead zone instead of a discontinuity.
  const scale = (mag - dz) / (1 - dz) / mag;
  return [x * scale, y * scale];
}

function applyAxialDeadzone(v: number, dz: number): number {
  if (Math.abs(v) <= dz) return 0;
  const sign = Math.sign(v);
  return sign * ((Math.abs(v) - dz) / (1 - dz));
}

function pickGamepad(): Gamepad | null {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  // Prefer the pinned index if it's still connected.
  if (state.pinnedIndex !== null) {
    const p = pads[state.pinnedIndex];
    if (p && p.connected) return p;
    state.pinnedIndex = null;
  }
  for (const p of pads) {
    if (p && p.connected) {
      state.pinnedIndex = p.index;
      return p;
    }
  }
  return null;
}

function buildSnapshot(pad: Gamepad): GamepadSnapshot {
  // Standard layout maps:
  //   axes[0] = left stick X    (-1 left, +1 right)
  //   axes[1] = left stick Y    (-1 up,   +1 down)  ← we invert
  //   axes[2] = right stick X
  //   axes[3] = right stick Y
  //   buttons[6].value = left trigger  (0 rest, 1 full)
  //   buttons[7].value = right trigger
  const [lx, ly] = applyStickDeadzone(
    pad.axes[0] ?? 0,
    -(pad.axes[1] ?? 0),
    DEFAULT_STICK_DEADZONE,
  );
  const [rx, ry] = applyStickDeadzone(
    pad.axes[2] ?? 0,
    -(pad.axes[3] ?? 0),
    DEFAULT_STICK_DEADZONE,
  );
  const lt = applyAxialDeadzone(pad.buttons[6]?.value ?? 0, DEFAULT_TRIGGER_DEADZONE);
  const rt = applyAxialDeadzone(pad.buttons[7]?.value ?? 0, DEFAULT_TRIGGER_DEADZONE);

  const buttons = pad.buttons.map((b) => b.pressed);
  const buttonValues = pad.buttons.map((b) => b.value);
  const axes = Array.from(pad.axes);
  const prev = state.prevPressed;
  const pressedEdges = buttons.map((p, i) => p && !(prev[i] ?? false));
  state.prevPressed = buttons;

  const mapping = pickMapping(pad);
  const logical = buildLogicalSnapshot(mapping, buttons, buttonValues, axes);

  return {
    index: pad.index,
    id: pad.id,
    standard: pad.mapping === "standard",
    lx,
    ly,
    rx,
    ry,
    lt,
    rt,
    buttons,
    pressedEdges,
    logical,
  };
}

/// Project the raw arrays through the active mapping into named
/// intents, then diff against `prevLogical` for rising-edge bits.
function buildLogicalSnapshot(
  mapping: LogicalMapping,
  buttons: readonly boolean[],
  buttonValues: readonly number[],
  axes: readonly number[],
): LogicalSnapshot {
  const prevJoint = readBoolSource(mapping.prevJoint, buttons, buttonValues, axes);
  const nextJoint = readBoolSource(mapping.nextJoint, buttons, buttonValues, axes);
  const estop = readBoolSource(mapping.estop, buttons, buttonValues, axes);
  const resetEStop = readBoolSource(mapping.resetEStop, buttons, buttonValues, axes);
  const armToggle = readBoolSource(mapping.armToggle, buttons, buttonValues, axes);
  const deadman = readAnalogSource(mapping.deadman, buttons, buttonValues, axes);

  // Edge detection lives here (rather than in the consumer) so that a
  // mapping change between frames — e.g. a controller reconnecting
  // with a different `Gamepad.mapping` — doesn't bleed previous state
  // into the new layout. Reset is handled by `start()` / disconnect.
  const prev = state.prevLogical;
  const snapshot: LogicalSnapshot = {
    mappingName: mapping.name,
    chords: mapping.chords,
    prevJoint,
    nextJoint,
    estop,
    resetEStop,
    armToggle,
    prevJointEdge: prevJoint && !prev.prevJoint,
    nextJointEdge: nextJoint && !prev.nextJoint,
    estopEdge: estop && !prev.estop,
    resetEStopEdge: resetEStop && !prev.resetEStop,
    armToggleEdge: armToggle && !prev.armToggle,
    deadman,
  };
  state.prevLogical = {
    prevJoint,
    nextJoint,
    estop,
    resetEStop,
    armToggle,
  };
  return snapshot;
}

function tick() {
  const pad = pickGamepad();
  const wasConnected = state.lastSnap !== null;
  if (pad) {
    const snap = buildSnapshot(pad);
    state.lastSnap = snap;
    if (!wasConnected) {
      for (const cb of state.connectedListeners) cb(true, snap);
    }
    for (const cb of state.listeners) cb(snap);
  } else if (wasConnected) {
    state.lastSnap = null;
    state.prevPressed = [];
    state.prevLogical = {};
    for (const cb of state.connectedListeners) cb(false, null);
  }
  state.raf = requestAnimationFrame(tick);
}

function start() {
  if (state.raf !== null) return;
  state.raf = requestAnimationFrame(tick);
}

function stop() {
  if (state.raf === null) return;
  cancelAnimationFrame(state.raf);
  state.raf = null;
  state.prevPressed = [];
  state.prevLogical = {};
}

/// Subscribe to gamepad ticks. Returns the unsubscribe fn.
///
/// `onTick` fires on every animation frame while a pad is connected
/// (typically 60 Hz). Don't call `setState` directly inside it — that
/// will re-render at full RAF rate. Buffer the latest snapshot in a
/// ref or schedule less-frequent React updates instead.
export function subscribeGamepad(onTick: Listener): () => void {
  state.listeners.add(onTick);
  if (state.lastSnap !== null) onTick(state.lastSnap);
  start();
  return () => {
    state.listeners.delete(onTick);
    if (state.listeners.size === 0 && state.connectedListeners.size === 0) {
      stop();
    }
  };
}

/// React hook returning a stable `connected` boolean and (when present)
/// the device id. Triggers a render only on connect/disconnect, never
/// on per-frame updates — those are exposed via `subscribeGamepad`.
export function useGamepad(): { connected: boolean; id: string; standard: boolean } {
  const [snap, setSnap] = useState<GamepadSnapshot | null>(state.lastSnap);

  useEffect(() => {
    const onChange = (_connected: boolean, s: GamepadSnapshot | null) => {
      setSnap(s);
    };
    state.connectedListeners.add(onChange);
    start();
    // Pick up an already-connected pad on mount.
    if (state.lastSnap) setSnap(state.lastSnap);
    return () => {
      state.connectedListeners.delete(onChange);
      if (state.listeners.size === 0 && state.connectedListeners.size === 0) {
        stop();
      }
    };
  }, []);

  // The `gamepadconnected` event isn't strictly required (the RAF loop
  // would notice the pad on the next tick anyway), but listening to it
  // wakes up our polling loop the moment a pad pairs even if no
  // consumer was active before. Without this, a user opening the motor
  // bench *before* connecting their controller would see "no controller"
  // until React re-renders for unrelated reasons.
  useEffect(() => {
    const onConn = () => start();
    window.addEventListener("gamepadconnected", onConn);
    return () => window.removeEventListener("gamepadconnected", onConn);
  }, []);

  return {
    connected: snap !== null,
    id: snap?.id ?? "",
    standard: snap?.standard ?? false,
  };
}

/// Convenience: subscribe to ticks but funnel through a stable callback
/// stored in a ref so consumers can change behaviour mid-mount without
/// re-subscribing (which would reset edge-detection state). The ref
/// pattern is what `useEffect`-with-changing-callbacks usually wants.
export function useGamepadCallback(cb: Listener): void {
  const ref = useRef(cb);
  ref.current = cb;
  useEffect(() => subscribeGamepad((s) => ref.current(s)), []);
}

// Re-export for callers that prefer named symbols.
export type { ButtonIndex, GamepadSnapshot };
