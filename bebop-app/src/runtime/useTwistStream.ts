// React binding for the drive twist stream (`./twistStream.ts`). Each drive
// screen (teleop, motor bench) gets one stream that writes to the cached
// runtime transport for its endpoint. Every drive input on the screen
// (on-screen joystick, WASD / arrows, gamepad bridge) calls the returned
// `sendTwist` / `stopDrive`, so throttling, the keepalive, the congestion
// skip and stop handling live in one place.

import { useCallback, useEffect, useRef } from "react";

import { getOrCreateRuntimeTransport } from "./cache";
import { createTwistStream } from "./twistStream";
import type { TwistStream } from "./twistStream";

export interface TwistStreamControls {
  /// Stream a body twist (vx m/s, wz rad/s, +wz = left turn). (0, 0) stops.
  sendTwist: (vx: number, wz: number) => void;
  /// Close the drive cycle and send (0, 0) immediately.
  stopDrive: () => void;
}

/// `onError` receives firmware rejections of drive twists (e.g. another
/// operator holds the drive seat), at most once per second. Stops are
/// silent.
export function useTwistStream(
  robotIp: string,
  runtimePort: number,
  onError: (message: string) => void,
): TwistStreamControls {
  const streamRef = useRef<TwistStream | null>(null);
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  useEffect(() => {
    // The stream holds the transport itself, not the screen's
    // `transportRef`. On unmount React runs the parent's effect cleanups
    // before the children's, so the screen's ref is already null when an
    // unmounting joystick or gamepad bridge sends its stop. Through the
    // stream that stop still goes out.
    const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
    let sentAnything = false;
    const stream = createTwistStream({
      send: (vx, wz, onReject) => {
        sentAnything = true;
        return transport.sendVelocityCommand(vx, wz, onReject);
      },
      bufferedAmount: () => transport.bufferedAmount(),
      now: () => performance.now(),
      setTimer: (fn, ms) => window.setTimeout(fn, ms),
      clearTimer: (handle) => window.clearTimeout(handle as number),
      onError: (message) => onErrorRef.current(message),
    });
    streamRef.current = stream;
    return () => {
      // Leaving the screen (or switching robots) mid-drive must halt the
      // chassis, which would otherwise hold the last twist until the
      // firmware's operator watchdog fires.
      if (sentAnything) stream.stop();
      stream.dispose();
    };
  }, [robotIp, runtimePort]);

  const sendTwist = useCallback((vx: number, wz: number) => {
    streamRef.current?.set(vx, wz);
  }, []);
  const stopDrive = useCallback(() => {
    streamRef.current?.stop();
  }, []);
  return { sendTwist, stopDrive };
}
