// React binding for the drive-input arbiter (`./driveArbiter.ts`).
//
// One hook per drive screen owns every way an operator can request
// motion and turns them into the single twist the screen's drive stream
// (`useTwistStream`) sends:
//   * keyboard: WASD / arrows on `window`; ignored while focus is in a text
//     field or while Ctrl / Meta / Alt is held (pressing one releases every
//     drive key); auto-repeat never re-engages a key the arbiter released;
//   * touch: the on-screen `DriveJoystick` reports through `touch`;
//   * gamepad: polled here (not in `GamepadDrive`, which is now only the
//     status card and button actions). Sticks drive only while the deadman
//     (RT) is at or past `DEADMAN_THRESHOLD`; a 250 ms tick watchdog and a
//     disconnect drop the pad's input.
// Priority between simultaneous deflections: touch > gamepad > keyboard.
// After a blur / hidden tab / page hide, a disable (E-STOP, wrong mode,
// unarmed wheels) or a dropped pad, a held input must return to neutral
// before it drives again.
//
// Usage (screen):
//   const stream = useTwistStream(ip, port, setError, { operatorStale });
//   const drive = useDriveInput(stream, { enabled: canDrive });
//   <DriveJoystick onTwist={drive.touch.onTwist} onStop={drive.touch.onStop}
//                  disabled={!canDrive} />
//   HUD: drive.twist, drive.source, drive.engaged, drive.needsNeutral.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { TwistStreamControls } from "../runtime/useTwistStream";
import { createDriveArbiter, createKeyboardDrive, gamepadTwist } from "./driveArbiter";
import type {
  DriveArbiter,
  DriveArbiterState,
  DriveSource,
  DriveTwist,
  KeyboardDrive,
} from "./driveArbiter";
import { getGamepadLayout } from "./gamepadLayout";
import { getActiveControlProfile } from "./profile";
import { subscribeGamepad, useGamepad } from "./useGamepad";

/// If no gamepad tick arrives for this long while the pad is deflected,
/// drop its input. Covers the RAF poll loop stalling (tab backgrounded,
/// WebView paused) and a pad that silently drops off mid-drive. Well under
/// the firmware's 500 ms operator watchdog, and above the ~60 Hz poll's
/// normal jitter.
const GAMEPAD_WATCHDOG_MS = 250;

/// HUD state (`twist`) refreshes at most this often; source / engaged /
/// neutral changes render immediately.
const HUD_INTERVAL_MS = 100;

export interface DriveChannel {
  onTwist: (vx: number, wz: number) => void;
  onStop: () => void;
}

export interface DriveInput {
  /// Merged twist being requested (HUD; throttled to ~10 Hz).
  twist: DriveTwist;
  /// Which input is driving, or null at neutral.
  source: DriveSource | null;
  /// Some input is deflected and driving is enabled.
  engaged: boolean;
  /// A held input must be released before the robot moves again (after a
  /// blur, link drop, operator-stale halt, disable, or a dropped pad).
  needsNeutral: boolean;
  /// Sink for the on-screen joystick.
  touch: DriveChannel;
  /// Halt now; held inputs must return to neutral before driving again.
  halt: () => void;
  /// Always send (0, 0), whether or not this client was driving.
  forceStop: () => void;
}

export interface DriveInputOptions {
  /// False while the robot can't be driven (E-STOP, wrong mode, no armed
  /// wheels, spectating). Nothing but a stop is forwarded.
  enabled: boolean;
  /// Bind WASD / arrows (default true).
  keyboard?: boolean;
  /// Poll the gamepad (default true).
  gamepad?: boolean;
}

export function useDriveInput(
  stream: TwistStreamControls,
  { enabled, keyboard = true, gamepad = true }: DriveInputOptions,
): DriveInput {
  const streamRef = useRef(stream);
  streamRef.current = stream;

  const [hud, setHud] = useState<DriveArbiterState>({
    twist: { vx: 0, wz: 0 },
    source: null,
    engaged: false,
    blocked: [],
  });
  const hudTimer = useRef<number | null>(null);
  const hudPending = useRef<DriveArbiterState | null>(null);
  const hudAt = useRef(0);

  const arbiterRef = useRef<DriveArbiter | null>(null);
  if (arbiterRef.current === null) {
    arbiterRef.current = createDriveArbiter(
      {
        set: (vx, wz) => streamRef.current.sendTwist(vx, wz),
        stop: () => streamRef.current.stopDrive(),
        halt: () => streamRef.current.requireNeutral(),
      },
      (state) => {
        // Throttle twist-only updates; anything else renders at once.
        const now = performance.now();
        const prev = hudPending.current ?? null;
        hudPending.current = state;
        const structural =
          prev === null ||
          prev.source !== state.source ||
          prev.engaged !== state.engaged ||
          prev.blocked.join() !== state.blocked.join();
        if (structural || now - hudAt.current >= HUD_INTERVAL_MS) {
          hudAt.current = now;
          if (hudTimer.current !== null) {
            window.clearTimeout(hudTimer.current);
            hudTimer.current = null;
          }
          setHud(state);
          return;
        }
        if (hudTimer.current === null) {
          hudTimer.current = window.setTimeout(() => {
            hudTimer.current = null;
            hudAt.current = performance.now();
            if (hudPending.current) setHud(hudPending.current);
          }, HUD_INTERVAL_MS - (now - hudAt.current));
        }
      },
    );
  }
  const arbiter = arbiterRef.current;

  // Held keys live outside the keyboard effect so every halt path can
  // clear them: otherwise an auto-repeat of a key still physically held
  // after a halt would re-report it and drive with no fresh press.
  const keyboardRef = useRef<KeyboardDrive | null>(null);
  if (keyboardRef.current === null) {
    keyboardRef.current = createKeyboardDrive(arbiter, getActiveControlProfile);
  }
  const keyDrive = keyboardRef.current;
  const haltAll = useCallback(() => {
    keyDrive.clear();
    arbiter.halt();
  }, [arbiter, keyDrive]);

  useEffect(
    () => () => {
      if (hudTimer.current !== null) window.clearTimeout(hudTimer.current);
    },
    [],
  );

  // ----------------------------------------------------------- enable gate
  useEffect(() => {
    arbiter.setEnabled(enabled);
  }, [arbiter, enabled]);

  // ------------------------------------------------------------- lifecycle
  // The stream itself also stops on these (useTwistStream); this releases
  // the inputs: the keyboard set is cleared, held touch / pad inputs must
  // return to neutral.
  useEffect(() => {
    const halt = haltAll;
    const onVisibility = () => {
      if (document.visibilityState === "hidden") halt();
    };
    window.addEventListener("blur", halt);
    window.addEventListener("pagehide", halt);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("blur", halt);
      window.removeEventListener("pagehide", halt);
      document.removeEventListener("visibilitychange", onVisibility);
      // Unmount: release everything (the stream stops on its own unmount).
      haltAll();
    };
  }, [haltAll]);

  // -------------------------------------------------------------- keyboard
  useEffect(() => {
    if (!keyboard) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (keyDrive.keydown(e)) e.preventDefault();
    };
    const onKeyUp = (e: KeyboardEvent) => keyDrive.keyup(e.key);
    // Focus moving into a text field mid-drive: its keyups still reach
    // us, but stop now rather than wait for them.
    const onFocusIn = (e: FocusEvent) => keyDrive.focusIn(e.target);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("focusin", onFocusIn);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("focusin", onFocusIn);
      keyDrive.releaseAll();
    };
  }, [keyDrive, keyboard]);

  // --------------------------------------------------------------- gamepad
  const { connected: padConnected } = useGamepad();
  useEffect(() => {
    if (!gamepad || !padConnected) return;
    let lastTickAt = performance.now();
    let deflected = false;
    const unsubscribe = subscribeGamepad((snap) => {
      lastTickAt = performance.now();
      const t = gamepadTwist(
        { lx: snap.lx, ly: snap.ly, rx: snap.rx, deadman: snap.logical.deadman },
        getGamepadLayout(),
        getActiveControlProfile(),
      );
      const now = t !== null && (t.vx !== 0 || t.wz !== 0);
      // Report deflections, the first neutral tick after one, and neutral
      // while the pad is blocked (that report is what unblocks it); skip
      // repeated neutral ticks so an idle pad costs nothing.
      if (now || deflected || arbiter.state().blocked.includes("gamepad")) {
        arbiter.update("gamepad", now ? t : null);
      }
      deflected = now;
    });
    const watchdog = window.setInterval(() => {
      if (deflected && performance.now() - lastTickAt > GAMEPAD_WATCHDOG_MS) {
        deflected = false;
        arbiter.drop("gamepad");
      }
    }, 100);
    return () => {
      unsubscribe();
      window.clearInterval(watchdog);
      arbiter.drop("gamepad");
    };
  }, [arbiter, gamepad, padConnected]);

  // ------------------------------------------------------------------ API
  const touch = useMemo<DriveChannel>(
    () => ({
      onTwist: (vx, wz) => arbiter.update("touch", { vx, wz }),
      onStop: () => arbiter.update("touch", null),
    }),
    [arbiter],
  );
  const halt = haltAll;
  // Halt the inputs first (held ones are blocked and latch the stream),
  // then write the unconditional stop. A held finger or stick must be
  // released before the robot moves again, E-STOP telemetry or not.
  const forceStop = useCallback(() => {
    haltAll();
    streamRef.current.forceStop();
  }, [haltAll]);

  return {
    twist: hud.twist,
    source: hud.source,
    engaged: hud.engaged,
    needsNeutral: stream.needsNeutral || hud.blocked.length > 0,
    touch,
    halt,
    forceStop,
  };
}
