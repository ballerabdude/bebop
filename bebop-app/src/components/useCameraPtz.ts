// Camera PTZ rate controller, shared by the video screen and the teleop
// screen. `setCameraPose` only takes absolute pan/tilt targets, so
// gestures (joystick deflection, held keys) are integrated client-side:
// while input is active, a fixed tick ramps the commanded target at a
// rate proportional to the gesture. Releasing just stops sending — the
// gimbal holds the last commanded pose (position-controlled, no "stop"
// to send, unlike the drive).
//
// A new gesture must move *relative* to where the gimbal is heading:
// while it is still settling toward the previous command, telemetry's
// actual pose lags behind, so seeding from it would snap the gimbal
// back and undo part of the last move. We therefore seed from the last
// commanded target while the firmware reports `moving`, and from the
// actual pose once settled (so external clients / horizon
// compensation drift are picked up between moves).

import { useCallback, useEffect, useRef, useState } from "react";

import type { RuntimeTransport } from "../runtime";
import type { CameraView } from "../runtime";

/// OBSBOT Tiny 2 gimbal limits (UVC-reported; see the recon table in
/// `firmware/bebop-linux/src/video.rs`). The firmware clamps too — this
/// is just so relative jogging doesn't accumulate past the stops.
const PAN_LIMIT_DEG = 130;
const TILT_LIMIT_DEG = 90;

/// Integration tick for the PTZ rate loop (10 Hz): fast enough that a
/// held deflection ramps the target smoothly, slow enough that every
/// tick's absolute command gets its own WS round-trip.
const PTZ_TICK_MS = 100;

const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, v));

export interface CameraPtzApi {
  /// Begin / continue a gesture at the given rates (deg/s; +pan right,
  /// +tilt up). Called repeatedly while the gesture is held.
  onRate: (panRate: number, tiltRate: number) => void;
  /// End the active gesture; the gimbal holds its last commanded pose.
  onStop: () => void;
  /// Command an absolute return-to-center.
  center: () => void;
  /// Last PTZ send error (ack timeout, transport loss). Cleared by the
  /// next gesture.
  error: string | null;
}

export function useCameraPtz(
  transport: RuntimeTransport,
  /// Latest camera view; feeds the relative-jog seeding (actual pose
  /// vs. last commanded target).
  camera: CameraView | null,
  /// False until the runtime connection is live AND the firmware
  /// reports a camera (`camera.present`). The integration loop only
  /// runs while true.
  ptzReady: boolean,
): CameraPtzApi {
  const [error, setError] = useState<string | null>(null);

  // Freshest camera view for the seeding math — a ref so rapid button
  // taps don't race React state (transport acks are faster than a
  // telemetry frame).
  const cameraRef = useRef<CameraView | null>(null);
  cameraRef.current = camera;

  const rateRef = useRef({ pan: 0, tilt: 0 });
  const targetRef = useRef<{ pan: number; tilt: number } | null>(null);
  /// Last pose we commanded — survives release (unlike `targetRef`)
  /// so the next gesture can continue from it.
  const lastCommandedRef = useRef<{ pan: number; tilt: number } | null>(null);

  // Coalesced pose sender — single-in-flight / latest-pending, so the
  // tick's command stream can't queue one WS request per step on a
  // slow link.
  const transportRef = useRef(transport);
  transportRef.current = transport;
  const inFlightRef = useRef(false);
  const pendingRef = useRef<{ pan: number; tilt: number } | null>(null);

  const sendPose = useCallback(async (pan: number, tilt: number) => {
    const t = transportRef.current;
    if (inFlightRef.current) {
      pendingRef.current = { pan, tilt };
      return;
    }
    inFlightRef.current = true;
    lastCommandedRef.current = { pan, tilt };
    try {
      await t.setCameraPose(pan, tilt);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      inFlightRef.current = false;
      const next = pendingRef.current;
      pendingRef.current = null;
      if (next) {
        void Promise.resolve().then(() => sendPose(next.pan, next.tilt));
      }
    }
  }, []);

  const onRate = useCallback((panRate: number, tiltRate: number) => {
    setError(null);
    if (targetRef.current === null) {
      // Relative continuation: while the gimbal is still slewing toward
      // the previous command (`moving`), telemetry's actual pose trails
      // the target, so seed from what we commanded — a re-press then
      // carries on from there instead of snapping the gimbal back.
      // Once settled, seed from the actual pose so external moves and
      // horizon-compensation drift are picked up between gestures.
      const cur = cameraRef.current;
      const seed =
        cur?.moving && lastCommandedRef.current
          ? lastCommandedRef.current
          : {
              pan: cur ? cur.panDeg : 0,
              tilt: cur ? cur.tiltDeg : 0,
            };
      targetRef.current = {
        pan: clamp(seed.pan, -PAN_LIMIT_DEG, PAN_LIMIT_DEG),
        tilt: clamp(seed.tilt, -TILT_LIMIT_DEG, TILT_LIMIT_DEG),
      };
    }
    // Pan sign flip, exactly once, at the only place pan commands are
    // generated. The OBSBOT Tiny 2's UVC `pan_absolute` control is
    // inverted relative to the app's "pan + = camera right"
    // convention — increasing UVC pan turns the camera LEFT (verified
    // on hardware: dragging the pad left commanded decreasing pan and
    // the camera swung right). Every producer (on-screen pad, I/J/K/L
    // keys, the gamepad's right stick) speaks app-space rates where
    // input-left = camera-left, so negate here. The internal target,
    // `lastCommandedRef`, and the telemetry read-back seed then all
    // live in the camera's own UVC space. Tilt has no such quirk.
    rateRef.current = { pan: -panRate, tilt: tiltRate };
  }, []);

  const onStop = useCallback(() => {
    rateRef.current = { pan: 0, tilt: 0 };
    targetRef.current = null;
  }, []);

  const center = useCallback(() => {
    setError(null);
    void sendPose(0, 0);
  }, [sendPose]);

  // Rate integration loop. Runs only while the PTZ is usable, and each
  // tick is a no-op unless a gesture is active (target seeded, non-zero
  // rate) — an idle 10 Hz timer is free.
  useEffect(() => {
    if (!ptzReady) return;
    const id = window.setInterval(() => {
      const rate = rateRef.current;
      const target = targetRef.current;
      if (!target || (rate.pan === 0 && rate.tilt === 0)) return;
      target.pan = clamp(
        target.pan + (rate.pan * PTZ_TICK_MS) / 1000,
        -PAN_LIMIT_DEG,
        PAN_LIMIT_DEG,
      );
      target.tilt = clamp(
        target.tilt + (rate.tilt * PTZ_TICK_MS) / 1000,
        -TILT_LIMIT_DEG,
        TILT_LIMIT_DEG,
      );
      void sendPose(target.pan, target.tilt);
    }, PTZ_TICK_MS);
    return () => window.clearInterval(id);
  }, [ptzReady, sendPose]);

  return { onRate, onStop, center, error };
}
