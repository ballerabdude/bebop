// Platform-agnostic gamepad model. Two layers live here:
//
//   1. **Raw model** (`buttons`, `pressedEdges`, `lx/ly/rx/ry/lt/rt`): a
//      per-frame snapshot of the pad's hardware state, indexed against
//      the W3C "standard" gamepad layout. The 8BitDo Mobile Controller
//      in **X-input** mode honours that layout exactly; pads in
//      D-input / Switch / older HID modes don't, so the indices on
//      `buttons[]` are not safe to read directly for non-standard pads.
//
//   2. **Logical model** (`logical`): named intents the rest of the app
//      consumes — `prevJoint`, `nextJoint`, `deadman`, etc. The
//      mapping in `input/mapping.ts` translates raw indices into these
//      intents based on `Gamepad.mapping`, so `GamepadDriver` and
//      friends never have to know whether the user's pad is in
//      X-input or D-input mode.

/// Standard-mapping button indices, named for the SDL / W3C convention.
/// Numeric values match the Web Gamepad API's `Gamepad.buttons` array
/// in "standard" mapping mode.
export const BTN = {
  /** PS5 Cross / Xbox A / 8BitDo "B" (south face) */
  South: 0,
  /** PS5 Circle / Xbox B / 8BitDo "A" (east face) */
  East: 1,
  /** PS5 Square / Xbox X / 8BitDo "Y" (west face) */
  West: 2,
  /** PS5 Triangle / Xbox Y / 8BitDo "X" (north face) */
  North: 3,
  ShoulderL: 4,
  ShoulderR: 5,
  TriggerL: 6,
  TriggerR: 7,
  Select: 8,
  Start: 9,
  ThumbL: 10,
  ThumbR: 11,
  DPadUp: 12,
  DPadDown: 13,
  DPadLeft: 14,
  DPadRight: 15,
  Mode: 16,
} as const;

export type ButtonIndex = (typeof BTN)[keyof typeof BTN];

/// Snapshot of a single connected gamepad. Sticks/triggers are
/// post-deadzone, normalised to [-1, 1] for sticks and [0, 1] for
/// triggers.
///
/// `buttons[i]` is `true` iff the corresponding `Gamepad.buttons[i]`
/// is currently pressed; `pressedEdges[i]` is `true` only on the tick
/// the button transitioned from released → pressed (rising edge).
/// Edge detection lives in the hook so consumers don't have to track
/// previous frames themselves — important for "press to e-stop" type
/// bindings where holding the button shouldn't re-fire.
/// Resolved logical-intent state for the active mapping. Same fields
/// regardless of whether the underlying pad is in X-input or D-input
/// mode; the mapping in `input/mapping.ts` decides which physical
/// input each intent reads from.
export interface LogicalSnapshot {
  /// Diagnostic name of the active mapping ("standard", "dinput", …).
  /// Surfaced in the UI so the operator can see which layout is in
  /// use without enabling debug logging.
  mappingName: string;
  /// User-facing chord names for each intent (e.g. "LB" for
  /// `prevJoint` under standard, "L1" under D-input). The driver
  /// renders these into the on-screen hint row so the labels match
  /// what's printed on the user's pad regardless of layout.
  chords: Readonly<Record<
    "prevJoint" | "nextJoint" | "deadman" | "estop" | "resetEStop" | "armToggle",
    string
  >>;
  /// Held state. True for as long as the user is holding the input
  /// down — useful for the deadman, where we care about every tick.
  prevJoint: boolean;
  nextJoint: boolean;
  estop: boolean;
  resetEStop: boolean;
  armToggle: boolean;
  /// Rising-edge bits. True only on the tick the input transitioned
  /// from released → pressed; used for "tap to do X" actions like
  /// e-stop and joint cycling so a hold doesn't re-fire every frame.
  prevJointEdge: boolean;
  nextJointEdge: boolean;
  estopEdge: boolean;
  resetEStopEdge: boolean;
  armToggleEdge: boolean;
  /// Continuous deadman pressure, 0..1. For analog triggers this is
  /// the trigger position; for digital-only deadman sources it
  /// collapses to {0, 1}.
  deadman: number;
}

export interface GamepadSnapshot {
  /** Index returned by `navigator.getGamepads()`. Stable for the life of the connection. */
  index: number;
  /** Browser-reported device id (e.g. "8BitDo Pro 2 (Vendor: 2dc8 Product: 6003)"). */
  id: string;
  /** Whether `Gamepad.mapping` is "standard". When false, the indices in `BTN` may not correspond to the user's expectation, but `logical` is always reliable. */
  standard: boolean;
  /** Left stick X, [-1, 1]. +X = right. */
  lx: number;
  /** Left stick Y, [-1, 1]. +Y = up. We invert the browser's "down is positive" convention. */
  ly: number;
  rx: number;
  ry: number;
  /** Left analog trigger, [0, 1]. Sourced from buttons[6].value (standard layout). */
  lt: number;
  /** Right analog trigger, [0, 1]. Sourced from buttons[7].value (standard layout). May be 0 for non-standard pads — read `logical.deadman` instead when you want the right-trigger pressure. */
  rt: number;
  buttons: readonly boolean[];
  pressedEdges: readonly boolean[];
  /// Resolved intents. Always populated; consumers should prefer this
  /// over `buttons[]` / `pressedEdges[]` for cross-layout correctness.
  logical: LogicalSnapshot;
}
