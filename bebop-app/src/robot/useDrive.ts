// Drive wiring shared by every screen that can move the robot (Drive page,
// Diagnostics › Motors): the twist stream (TWIST workstream's
// `useTwistStream`) plus the input arbiter (`useDriveInput`), bound to this
// session's host and link state.
//
// A degraded runtime link (no telemetry for >1 s) disables driving and
// halts held input, so motion never continues on a link the app can't see.

import { useEffect } from "react";

import { useDriveInput } from "../input";
import type { DriveInput } from "../input";
import { useTwistStream } from "../runtime/useTwistStream";
import { PORTS } from "./endpoints";
import { useConnection, useRobot, useTelemetry } from "./session";

export function useDrive({
  canDrive,
  onError,
}: {
  /// Robot-side readiness (wheeled, armed, right mode, no E-STOP).
  canDrive: boolean;
  onError: (message: string) => void;
}): DriveInput & { linkOk: boolean } {
  const { host } = useRobot();
  const link = useConnection();
  const seat = useTelemetry(
    (s) => ({
      operatorStale: !!s?.drive.operatorStale,
      you: !!s?.drive.youAreActiveOperator,
    }),
    10,
  );
  const stream = useTwistStream(host, PORTS.runtime, onError, {
    operatorStale: seat.operatorStale,
    holdsSeat: seat.you,
  });
  const linkOk = link.state === "connected";
  const drive = useDriveInput(stream, { enabled: canDrive && linkOk });
  const { halt } = drive;
  useEffect(() => {
    if (!linkOk) halt();
  }, [linkOk, halt]);
  return { ...drive, linkOk };
}
