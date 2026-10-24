// Drive-input arbiter: merges the keyboard (WASD / arrows), the on-screen
// touch joystick and the gamepad into the one twist the drive stream
// (`src/runtime/twistStream.ts`) sends.
//
// Rules:
//   * every source reports its own twist, or null when idle;
//   * the merged twist is the first NON-ZERO source in `DRIVE_PRIORITY`
//     (touch > gamepad > keyboard). Priority only settles conflicts between
//     sources that are deflected at the same time, so WASD keeps driving
//     while a pad sits connected but idle (deadman released = null);
//   * the merged twist is (0, 0) only when every usable source is at
//     neutral. That is what the stream's re-engage latch waits for after a
//     link drop: "release all inputs", not "the top source went quiet";
//   * a source deflected across a halt (blur, hidden tab), a drop (pad
//     disconnected or its poll stalled) or a disable is "blocked": it
//     counts as idle until it reports neutral, so a held input never
//     resumes motion by itself;
//   * while disabled (E-STOP, wrong mode, no armed wheels, spectating)
//     nothing but a stop is forwarded, and anything deflected meanwhile is
//     blocked too (holding W through an E-STOP reset doesn't lurch).
//
// The module is pure (no React or DOM): `useDriveInput.ts` binds it, and
// tests/driveInput.test.mjs drives it directly.

export type DriveSource = "touch" | "gamepad" | "keyboard";

/// Highest priority first.
export const DRIVE_PRIORITY: readonly DriveSource[] = ["touch", "gamepad", "keyboard"];

/// Gamepad right-trigger pressure that engages the deadman. Below it the
/// pad's sticks never drive.
export const DEADMAN_THRESHOLD = 0.4;

export interface DriveTwist {
  vx: number;
  wz: number;
}

export interface DriveLimits {
  maxLinear: number;
  maxAngular: number;
}

const isZero = (t: DriveTwist | null): boolean =>
  t === null || (t.vx === 0 && t.wz === 0);

/// Pick the twist to drive with. `source` is null when every input is at
/// neutral.
export function mergeDriveSources(
  inputs: Partial<Record<DriveSource, DriveTwist | null>>,
): { twist: DriveTwist; source: DriveSource | null } {
  for (const source of DRIVE_PRIORITY) {
    const t = inputs[source] ?? null;
    if (t !== null && !isZero(t)) return { twist: { vx: t.vx, wz: t.wz }, source };
  }
  return { twist: { vx: 0, wz: 0 }, source: null };
}

// ---------------------------------------------------------------- keyboard

const FORWARD = ["w", "arrowup"];
const BACK = ["s", "arrowdown"];
const LEFT = ["a", "arrowleft"];
const RIGHT = ["d", "arrowright"];

/// Normalised key name (`KeyboardEvent.key` lower-cased) of a drive key.
export function isDriveKey(key: string): boolean {
  const k = key.toLowerCase();
  return FORWARD.includes(k) || BACK.includes(k) || LEFT.includes(k) || RIGHT.includes(k);
}

/// Modifier keys whose press releases every held drive key: with Cmd /
/// Ctrl / Alt down the OS often swallows the drive keys' keyup (macOS Cmd
/// in particular, Cmd+Tab, Alt+Tab), which would leave them held here.
export function isModifierKey(key: string): boolean {
  return key === "Meta" || key === "Control" || key === "Alt" || key === "OS";
}

/// Compose held keys into a twist. Opposing keys cancel.
export function keyboardTwist(keys: ReadonlySet<string>, limits: DriveLimits): DriveTwist {
  let vx = 0;
  let wz = 0;
  if (FORWARD.some((k) => keys.has(k))) vx += limits.maxLinear;
  if (BACK.some((k) => keys.has(k))) vx -= limits.maxLinear;
  if (LEFT.some((k) => keys.has(k))) wz += limits.maxAngular;
  if (RIGHT.some((k) => keys.has(k))) wz -= limits.maxAngular;
  return { vx, wz };
}

/// Minimal shape of an event target this needs (DOM `Element` or a test
/// stub).
export interface TargetLike {
  tagName?: string;
  isContentEditable?: boolean;
  type?: string;
}

const NON_TEXT_INPUT_TYPES = new Set([
  "button",
  "checkbox",
  "radio",
  "range",
  "submit",
  "reset",
  "color",
  "file",
  "image",
]);

/// True when keystrokes belong to a text field, not to driving: <input>
/// (text-like types), <textarea>, <select> or anything contenteditable.
export function isEditableTarget(target: unknown): boolean {
  if (target === null || typeof target !== "object") return false;
  const t = target as TargetLike;
  if (t.isContentEditable) return true;
  const tag = (t.tagName ?? "").toUpperCase();
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag === "INPUT") return !NON_TEXT_INPUT_TYPES.has((t.type ?? "text").toLowerCase());
  return false;
}

export interface KeyEventLike {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  target?: unknown;
}

/// A keydown that must not start or change keyboard drive: a modifier is
/// held (browser / OS shortcuts) or focus is in a text field.
export function ignoreDriveKeydown(e: KeyEventLike): boolean {
  return !!(e.ctrlKey || e.metaKey || e.altKey) || isEditableTarget(e.target);
}

/// Held-key tracker for keyboard drive. Pure: `useDriveInput` feeds it
/// window key events; tests drive it directly.
///
/// The held set is the only thing that turns an auto-repeat keydown into a
/// deflection: a repeat of a key that isn't in the set (released by a
/// halt, a modifier, focus moving into a text field) is ignored. Every
/// halt path must therefore `clear()` the set together with the arbiter's
/// `halt()`, or the next auto-repeat of a still-held key would drive again
/// with no fresh press.
export interface KeyboardDrive {
  /// Returns true when the event was a drive key this handled (the caller
  /// should `preventDefault`).
  keydown(e: KeyEventLike & { repeat?: boolean }): boolean;
  keyup(key: string): void;
  /// Focus moved to `target`: release everything if it is a text field.
  focusIn(target: unknown): void;
  /// Release every held key and report the release (neutral).
  releaseAll(): void;
  /// Forget the held keys without reporting (pair with `arbiter.halt()`).
  clear(): void;
}

export function createKeyboardDrive(
  arbiter: Pick<DriveArbiter, "update">,
  limits: () => DriveLimits,
): KeyboardDrive {
  const keys = new Set<string>();
  const report = () =>
    arbiter.update("keyboard", keys.size === 0 ? null : keyboardTwist(keys, limits()));
  const releaseAll = () => {
    if (keys.size === 0) return;
    keys.clear();
    report();
  };
  return {
    keydown(e) {
      if (isModifierKey(e.key)) {
        releaseAll();
        return false;
      }
      if (!isDriveKey(e.key) || ignoreDriveKeydown(e)) return false;
      const k = e.key.toLowerCase();
      // Auto-repeat of a key already released is not a fresh press.
      if (e.repeat && !keys.has(k)) return true;
      keys.add(k);
      report();
      return true;
    },
    keyup(key) {
      if (!isDriveKey(key)) return;
      keys.delete(key.toLowerCase());
      // Always report: a release is what clears a neutral requirement.
      report();
    },
    focusIn(target) {
      if (isEditableTarget(target)) releaseAll();
    },
    releaseAll,
    clear() {
      keys.clear();
    },
  };
}

// ----------------------------------------------------------------- gamepad

export type GamepadDriveLayout = "split" | "arcade";

export interface GamepadSticks {
  /// Left stick, [-1, 1], +x right, +y up.
  lx: number;
  ly: number;
  /// Right stick x, [-1, 1].
  rx: number;
  /// Deadman (right trigger) pressure, [0, 1].
  deadman: number;
}

/// Twist requested by the pad, or null when the deadman is released.
/// Trigger pressure scales the request: just past the threshold creeps
/// (~64 % of the limit), a full pull reaches it. `split`: left stick up /
/// down drives, right stick left / right turns; `arcade`: the left stick
/// does both. Up = forward, right = turn right (+wz = left turn).
export function gamepadTwist(
  sticks: GamepadSticks,
  layout: GamepadDriveLayout,
  limits: DriveLimits,
): DriveTwist | null {
  const trigger = Math.min(1, Math.max(0, Number.isFinite(sticks.deadman) ? sticks.deadman : 0));
  if (trigger < DEADMAN_THRESHOLD) return null;
  const gain = 0.4 + 0.6 * trigger;
  const turn = layout === "arcade" ? sticks.lx : sticks.rx;
  // `+ 0` folds -0 into 0 so a centred stick reads as neutral.
  return {
    vx: sticks.ly * limits.maxLinear * gain + 0,
    wz: -turn * limits.maxAngular * gain + 0,
  };
}

// ------------------------------------------------------------------ arbiter

export interface DriveArbiterSink {
  /// Forward the merged twist (non-zero) to the drive stream.
  set(vx: number, wz: number): void;
  /// Every input is at neutral (a release): stop (the stream decides
  /// whether a stop is owed) and clear the stream's re-engage latch.
  stop(): void;
  /// The arbiter halted while an input is still physically held (halt,
  /// drop or disable left a source blocked): stop if owed but do NOT clear
  /// the stream's re-engage latch; set it if the stream was driving.
  halt(): void;
}

export interface DriveArbiterState {
  twist: DriveTwist;
  source: DriveSource | null;
  /// Some usable input is deflected and the arbiter is enabled.
  engaged: boolean;
  /// Sources that must report neutral before they may drive again.
  blocked: DriveSource[];
}

export interface DriveArbiter {
  /// Report one source's twist; null (or zero) = that source is idle.
  update(source: DriveSource, twist: DriveTwist | null): void;
  /// The source went away without a release (pad disconnected, its poll
  /// stalled). Forget its input; if it was deflected it must report
  /// neutral before it drives again.
  drop(source: DriveSource): void;
  /// Halt (blur, hidden tab, page hide): every deflected source must
  /// return to neutral before it drives again. The keyboard is cleared
  /// outright: its held keys' keyup went to another window. Callers must
  /// also `clear()` the `KeyboardDrive` held set, so an auto-repeat of a
  /// still-held key is not taken as a fresh press.
  halt(): void;
  /// Gate driving. Disabled, nothing but a stop is forwarded, and any
  /// source deflected while disabled (or held across the switch) must
  /// return to neutral after re-enabling: holding W through an E-STOP
  /// reset doesn't lurch the robot.
  setEnabled(enabled: boolean): void;
  state(): DriveArbiterState;
}

/// Turns per-source reports into one twist for the stream. A source is
/// "blocked" after a halt, a drop, or a deflection while disabled; a
/// blocked source counts as idle until it reports neutral.
export function createDriveArbiter(
  sink: DriveArbiterSink,
  onChange?: (state: DriveArbiterState) => void,
): DriveArbiter {
  const inputs: Partial<Record<DriveSource, DriveTwist | null>> = {};
  const blocked = new Set<DriveSource>();
  let enabled = true;
  let last: DriveArbiterState = {
    twist: { vx: 0, wz: 0 },
    source: null,
    engaged: false,
    blocked: [],
  };

  function block(source: DriveSource): void {
    if (!isZero(inputs[source] ?? null)) blocked.add(source);
  }

  function forward(force: boolean): void {
    const usable: Partial<Record<DriveSource, DriveTwist | null>> = {};
    if (enabled) {
      for (const s of DRIVE_PRIORITY) usable[s] = blocked.has(s) ? null : inputs[s] ?? null;
    }
    const merged = mergeDriveSources(usable);
    const engaged = merged.source !== null;
    const prev = last;
    last = {
      twist: merged.twist,
      source: merged.source,
      engaged,
      blocked: DRIVE_PRIORITY.filter((s) => blocked.has(s)),
    };
    const sameTwist =
      prev.twist.vx === merged.twist.vx &&
      prev.twist.wz === merged.twist.wz &&
      prev.source === merged.source;
    const sameBlocked = prev.blocked.join() === last.blocked.join();
    // Identical non-zero twists are not re-forwarded (the stream's
    // keepalive refreshes them). A neutral result is forwarded on every
    // explicit report (`force`): a release is what clears the stream's
    // re-engage latch, and a stop that isn't owed writes nothing.
    // A neutral the arbiter made itself (halt, drop, disable) is a release
    // only if nothing is left blocked, i.e. no input is still held;
    // otherwise it is a halt and the stream's latch must survive it.
    if (!sameTwist || (force && !engaged)) {
      if (engaged) sink.set(merged.twist.vx, merged.twist.wz);
      else if (force || blocked.size === 0) sink.stop();
      else sink.halt();
    }
    if (!sameTwist || !sameBlocked) onChange?.(last);
  }

  return {
    update(source, twist) {
      // A NaN / infinite component counts as 0, never as a deflection.
      const finite = (v: number) => (Number.isFinite(v) ? v + 0 : 0);
      const t = twist === null ? null : { vx: finite(twist.vx), wz: finite(twist.wz) };
      inputs[source] = t;
      if (isZero(t)) blocked.delete(source);
      else if (!enabled) blocked.add(source);
      forward(isZero(t));
    },
    drop(source) {
      block(source);
      inputs[source] = null;
      forward(false);
    },
    halt() {
      inputs.keyboard = null;
      blocked.delete("keyboard");
      for (const s of DRIVE_PRIORITY) block(s);
      forward(false);
    },
    setEnabled(next) {
      if (enabled === next) return;
      enabled = next;
      for (const s of DRIVE_PRIORITY) block(s);
      forward(false);
    },
    state: () => last,
  };
}
