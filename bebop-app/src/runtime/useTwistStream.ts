// React binding for the drive twist stream (`./twistStream.ts`). Each drive
// screen (teleop, motor bench) gets one stream that writes to the cached
// runtime transport for its endpoint. Drive inputs reach it through the
// `useDriveInput` arbiter (`src/input/useDriveInput.ts`), which merges the
// on-screen joystick, WASD / arrows and the gamepad into one twist, so
// throttling, the keepalive, the congestion skip and stop handling live in
// one place.
//
// The hook owns every lifecycle stop, so no screen has to remember them:
//   * unmount / robot switch          -> stop (if this stream drove)
//   * window blur, tab hidden,
//     pagehide                         -> stop + require neutral
//   * runtime link not connected       -> require neutral (nothing to send)
//   * `operatorStale` rising edge      -> stop + require neutral
// "Require neutral" means a still-held input (a key whose keyup went to
// another window, a joystick finger, a pad's deadman) cannot drive again
// until the operator releases it. Stops are only written when this client
// drove since its last stop (or holds the seat), so a spectator never
// halts another operator; `forceStop` always writes.

import { useCallback, useEffect, useRef, useState } from "react";

import { getOrCreateRuntimeTransport } from "./cache";
import { createTwistStream } from "./twistStream";
import type { TwistStream } from "./twistStream";

export interface TwistStreamControls {
  /// Stream a body twist (vx m/s, wz rad/s, +wz = left turn). (0, 0) stops.
  sendTwist: (vx: number, wz: number) => void;
  /// Close the drive cycle and send (0, 0) immediately if this client
  /// drove since its last stop. Also counts as "input back at neutral".
  stopDrive: () => void;
  /// Always send (0, 0), whether or not this client was driving. A held
  /// input must then return to neutral before driving again.
  forceStop: () => void;
  /// Halt without counting as a release: stop if owed and, if an input was
  /// driving, ignore non-zero twists until it returns to neutral.
  requireNeutral: () => void;
  /// True while a held input must return to neutral before driving again
  /// (after a blur, a hidden tab, a link drop or an operator-stale halt).
  needsNeutral: boolean;
}

export interface TwistStreamOptions {
  /// The firmware's `drive.operatorStale` telemetry bit. On its rising
  /// edge the stream stops and requires the input to return to neutral.
  operatorStale?: boolean;
  /// True while this client holds the drive seat (stops then always go
  /// out). Leave undefined until the firmware reports seat ownership.
  holdsSeat?: boolean;
}

/// `onError` receives firmware rejections of drive twists (e.g. another
/// operator holds the drive seat), at most once per second. Stops are
/// silent.
export function useTwistStream(
  robotIp: string,
  runtimePort: number,
  onError: (message: string) => void,
  options: TwistStreamOptions = {},
): TwistStreamControls {
  const streamRef = useRef<TwistStream | null>(null);
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  const holdsSeatRef = useRef(options.holdsSeat ?? false);
  holdsSeatRef.current = options.holdsSeat ?? false;
  const [needsNeutral, setNeedsNeutral] = useState(false);

  useEffect(() => {
    // The stream holds the transport itself, not the screen's
    // `transportRef`. On unmount React runs the parent's effect cleanups
    // before the children's, so the screen's ref is already null when an
    // unmounting joystick or gamepad bridge sends its stop. Through the
    // stream that stop still goes out.
    const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
    const stream = createTwistStream({
      send: (vx, wz, onReject) => transport.sendVelocityCommand(vx, wz, onReject),
      bufferedAmount: () => transport.bufferedAmount(),
      now: () => performance.now(),
      setTimer: (fn, ms) => window.setTimeout(fn, ms),
      clearTimer: (handle) => window.clearTimeout(handle as number),
      onError: (message) => onErrorRef.current(message),
      holdsSeat: () => holdsSeatRef.current,
      onNeutralRequiredChange: setNeedsNeutral,
    });
    streamRef.current = stream;
    setNeedsNeutral(false);

    // Focus leaving the window delivers held keys' keyup elsewhere, a
    // hidden tab throttles the keepalive timers and freezes the gamepad
    // poll, and a page being unloaded or frozen (mobile app switch,
    // bfcache) sends nothing further: halt on all of them.
    const halt = () => stream.requireNeutral();
    const onVisibility = () => {
      if (document.visibilityState === "hidden") halt();
    };
    window.addEventListener("blur", halt);
    window.addEventListener("pagehide", halt);
    document.addEventListener("visibilitychange", onVisibility);
    // A dropped link already zeroed the robot (seat released, operator
    // watchdog); make sure a held input can't drive on reconnect.
    const unsubscribeLink = transport.onConnectionStateChange((state) => {
      if (state !== "connected") stream.linkLost();
    });

    return () => {
      window.removeEventListener("blur", halt);
      window.removeEventListener("pagehide", halt);
      document.removeEventListener("visibilitychange", onVisibility);
      unsubscribeLink();
      // Leaving the screen (or switching robots) mid-drive must halt the
      // chassis, which would otherwise hold the last twist until the
      // firmware's operator watchdog fires. `stop` only writes if this
      // stream drove.
      stream.stop();
      // The disposed stream stays in the ref: an unmounting child's late
      // stop must still reach the robot (dispose keeps stops working).
      stream.dispose();
    };
  }, [robotIp, runtimePort]);

  // The firmware's operator watchdog zeroed the twist: the operator must
  // release before driving again rather than the next keepalive silently
  // resuming motion.
  const operatorStale = options.operatorStale ?? false;
  useEffect(() => {
    if (operatorStale) streamRef.current?.requireNeutral();
  }, [operatorStale]);

  const sendTwist = useCallback((vx: number, wz: number) => {
    streamRef.current?.set(vx, wz);
  }, []);
  const stopDrive = useCallback(() => {
    streamRef.current?.stop();
  }, []);
  const forceStop = useCallback(() => {
    streamRef.current?.forceStop();
  }, []);
  const requireNeutral = useCallback(() => {
    streamRef.current?.requireNeutral();
  }, []);
  return { sendTwist, stopDrive, forceStop, requireNeutral, needsNeutral };
}
