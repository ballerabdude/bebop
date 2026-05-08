// Logical → physical button mapping for the Web Gamepad API.
//
// The W3C "standard" mapping (https://w3c.github.io/gamepad/#dfn-standard-gamepad)
// places buttons at well-known indices: LB=4, RB=5, LT=6, RT=7, L3=10,
// R3=11, etc. Most modern pads in their default mode honour it (Xbox
// Wireless, DualSense over BT, Switch Pro through MFi/Steam, 8BitDo
// pads in **X-input** mode).
//
// The 8BitDo Mobile Controller and most 8BitDo pads in their default
// **D-input / Android** mode report a different HID layout that Chromium
// surfaces with `Gamepad.mapping === ""` (empty string for "no mapping
// info"). In that layout:
//
//   buttons[0]  = A
//   buttons[1]  = B
//   buttons[2]  = (varies; sometimes nothing, sometimes Home)
//   buttons[3]  = X
//   buttons[4]  = Y
//   buttons[6]  = L1
//   buttons[7]  = R1
//   buttons[8]  = L2 (analog .value)
//   buttons[9]  = R2 (analog .value)
//   buttons[10] = Select
//   buttons[11] = Start
//   buttons[12] = L3
//   buttons[13] = R3
//
// (Compare the user's bug report: "RB acts like the trigger" → R1 is
// at 7, my old code read RT from buttons[7]; "Y cycles joints" → Y is
// at 4, my old code treated buttons[4] as LB.)
//
// We support both. `pickMapping` looks at `Gamepad.mapping` and, as a
// fallback, sniffs `Gamepad.id` for known device strings so we can
// route a controller correctly even when the browser misclassifies it.

import { BTN } from "./types";

/// Per-input description of the physical source for one logical intent.
/// `axis` is reserved for the (rare) case where a controller exposes a
/// trigger only as a unipolar axis; we surface it here so the abstraction
/// can grow into that without churning consumers. For now both built-in
/// mappings use `analogButton` for the deadman.
export type Source =
  | { kind: "buttonPress"; index: number }
  | { kind: "analogButton"; index: number }
  | { kind: "axis"; index: number; sign?: 1 | -1 };

export interface LogicalMapping {
  /// Short identifier surfaced for diagnostics ("standard", "dinput", …).
  name: string;
  /// User-friendly chord names rendered in the UI hints. Keyed by intent.
  chords: Readonly<Record<LogicalIntent, string>>;
  prevJoint: Source;
  nextJoint: Source;
  /// Continuous deadman pressure source. Output is normalised to [0, 1].
  deadman: Source;
  estop: Source;
  resetEStop: Source;
  armToggle: Source;
}

export type LogicalIntent =
  | "prevJoint"
  | "nextJoint"
  | "deadman"
  | "estop"
  | "resetEStop"
  | "armToggle";

// ---------------------------------------------------------------------------
// Built-in mappings
// ---------------------------------------------------------------------------

/// W3C "standard" layout. What every pad reports when the browser claims
/// `Gamepad.mapping === "standard"`.
export const STANDARD_MAPPING: LogicalMapping = {
  name: "standard",
  chords: {
    prevJoint: "LB",
    nextJoint: "RB",
    deadman: "RT",
    estop: "B / Circle",
    resetEStop: "A / Cross",
    armToggle: "L3",
  },
  prevJoint: { kind: "buttonPress", index: BTN.ShoulderL }, // 4
  nextJoint: { kind: "buttonPress", index: BTN.ShoulderR }, // 5
  deadman: { kind: "analogButton", index: BTN.TriggerR }, // 7
  estop: { kind: "buttonPress", index: BTN.East }, // 1
  resetEStop: { kind: "buttonPress", index: BTN.South }, // 0
  armToggle: { kind: "buttonPress", index: BTN.ThumbL }, // 10
};

/// 8BitDo / generic D-input layout. The right shoulder analog trigger
/// surfaces at buttons[9].value (R2/ZR), the bumpers at 6/7, and L3 at
/// 12. A and B share the standard-mapping indices (0/1) so e-stop and
/// reset don't move.
export const DINPUT_MAPPING: LogicalMapping = {
  name: "dinput",
  chords: {
    prevJoint: "L1",
    nextJoint: "R1",
    deadman: "R2 / ZR",
    estop: "B / Circle",
    resetEStop: "A / Cross",
    armToggle: "L3",
  },
  prevJoint: { kind: "buttonPress", index: 6 },
  nextJoint: { kind: "buttonPress", index: 7 },
  deadman: { kind: "analogButton", index: 9 },
  estop: { kind: "buttonPress", index: 1 },
  resetEStop: { kind: "buttonPress", index: 0 },
  armToggle: { kind: "buttonPress", index: 12 },
};

// ---------------------------------------------------------------------------
// Mapping selection
// ---------------------------------------------------------------------------

/// Pick the right mapping for a connected pad. Heuristic order:
///
///   1. If the browser reports `mapping === "standard"`, trust it. Modern
///      Chromium does the right thing for Xbox Wireless, DualSense over
///      Bluetooth, and 8BitDo controllers in X-input mode.
///   2. Otherwise, treat the pad as D-input (the dominant non-standard
///      layout). This catches 8BitDo Pro 2 / Zero 2 / Mobile in their
///      Android default mode, plus most generic HID pads.
///
/// We deliberately don't try to fingerprint by `Gamepad.id` further —
/// the IDs are vendor-specific and add maintenance for marginal gain.
/// If a specific pad needs a bespoke layout, add a third entry here
/// and slot it into this switch.
export function pickMapping(pad: Gamepad): LogicalMapping {
  if (pad.mapping === "standard") return STANDARD_MAPPING;
  return DINPUT_MAPPING;
}

// ---------------------------------------------------------------------------
// Source evaluation
// ---------------------------------------------------------------------------

/// Read a single `Source` against the raw browser arrays. Buttons fall
/// back to `false` / `0` when the index is out of range; axes default
/// to `0`. This means a mapping pointing at a missing input just stays
/// quiet rather than throwing — useful for partial controllers (e.g.
/// digital-only pads where the deadman source has no analog readback).
export function readBoolSource(
  src: Source,
  buttons: readonly boolean[],
  buttonValues: readonly number[],
  axes: readonly number[],
): boolean {
  switch (src.kind) {
    case "buttonPress":
      return buttons[src.index] ?? false;
    case "analogButton":
      return (buttonValues[src.index] ?? 0) > 0.5;
    case "axis": {
      const v = axes[src.index] ?? 0;
      const signed = v * (src.sign ?? 1);
      return signed > 0.5;
    }
  }
}

/// Read a `Source` as an analog 0..1 value. Boolean sources collapse to
/// `1` when pressed, `0` otherwise — useful for fallback when the
/// abstract intent is "deadman pressure" but the physical pad only has
/// digital triggers.
export function readAnalogSource(
  src: Source,
  buttons: readonly boolean[],
  buttonValues: readonly number[],
  axes: readonly number[],
): number {
  switch (src.kind) {
    case "buttonPress":
      return buttons[src.index] ? 1 : 0;
    case "analogButton":
      return clamp01(buttonValues[src.index] ?? 0);
    case "axis": {
      const v = (axes[src.index] ?? 0) * (src.sign ?? 1);
      return clamp01(v);
    }
  }
}

function clamp01(v: number): number {
  if (Number.isNaN(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}
