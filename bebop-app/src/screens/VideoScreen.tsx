import { useEffect, useState } from "react";

import { PtzJoystick } from "../components/PtzJoystick";
import { useCameraPtz } from "../components/useCameraPtz";
import { VideoFeed } from "../components/VideoFeed";
import { Banner, Button, Card } from "../components/ui";
import { getOrCreateRuntimeTransport } from "../runtime";
import type {
  CameraView,
  NavView,
  RuntimeConnectionState,
} from "../runtime";

interface VideoScreenProps {
  /** IP address of the robot (LAN address of the bebop-linux runtime). */
  robotIp: string;
  /** Optional override for the runtime port. Defaults to 9090. */
  runtimePort?: number;
  onBack: () => void;
  /** Label for the back button, e.g. "Back to motor bench". */
  backLabel?: string;
}

/** Live camera view served by the firmware's `GET /video` MJPEG endpoint,
 * with operator PTZ controls.
 *
 * `multipart/x-mixed-replace` renders natively in an `<img>` tag, so the
 * stream needs no JavaScript decode loop — the browser paints each JPEG
 * part as it arrives. The firmware owns the camera exclusively (see
 * `firmware/bebop-linux/src/video.rs`); this screen is just another
 * subscriber alongside bebop-vision. If the camera is missing (no
 * `video:` in the robot YAML) the endpoint answers 503 and we land in the
 * error state with a retry button. The stream itself, its state
 * placeholders, and the nav-mask overlay live in `VideoFeed` (shared
 * with the teleop screen).
 *
 * PTZ rides the same runtime WS the motor bench uses: `SetCameraPose`
 * is not mode-gated (moving the camera can't hurt anyone), so the
 * controls are live in any mode. The gimbal is position-only (absolute
 * UVC pan/tilt targets), so the joystick integrates stick deflection
 * into absolute targets at a fixed tick, clamped against the known
 * gimbal limits — see `useCameraPtz`. Gestures compose relatively:
 * while the gimbal is still slewing, a new press continues from the
 * last commanded target (telemetry's actual pose trails a mid-flight
 * move); once settled it re-syncs to the actual pose (the gimbal
 * occasionally applies horizon compensation, so we never track
 * locally). A 30° move settles in ~0.5-0.7 s; the "moving" dot in the
 * pose readout comes straight from the firmware's settle detector.
 */
export function VideoScreen({
  robotIp,
  runtimePort = 9090,
  onBack,
  backLabel = "Back",
}: VideoScreenProps) {
  // Cache-busting nonce: bumping it remounts the <img> inside
  // VideoFeed, which tears the multipart stream down and reconnects —
  // the retry path.
  const [reconnectKey, setReconnectKey] = useState(0);
  const [streamState, setStreamState] = useState<
    "loading" | "live" | "error"
  >("loading");
  const [conn, setConn] = useState<RuntimeConnectionState>("disconnected");
  const [camera, setCamera] = useState<CameraView | null>(null);
  // Nav overlay: off by default so video playback is untouched until
  // the operator opts in. `nav` mirrors the telemetry summary (present /
  // provider / rate) for the toggle's state hint.
  const [showNav, setShowNav] = useState(false);
  const [nav, setNav] = useState<NavView | null>(null);

  const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
  const url = `http://${robotIp}:${runtimePort}/video`;

  useEffect(() => {
    const offs = [
      transport.onConnectionStateChange(setConn),
      transport.onTelemetry((snap) => {
        setCamera(snap.camera);
        setNav(snap.nav);
      }),
    ];
    void transport
      .connect(robotIp, runtimePort)
      .then(() => transport.getSnapshot())
      .then((snap) => {
        setCamera(snap.camera);
      })
      .catch(() => {
        /* conn state listener surfaces the failure */
      });
    return () => offs.forEach((off) => off());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [robotIp, runtimePort]);

  const ptzReady = conn === "connected" && camera !== null && camera.present;
  const ptz = useCameraPtz(transport, camera, ptzReady);

  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-4xl">
      <div className="flex items-center justify-between w-full">
        <h2 className="text-text font-semibold text-base">
          Live video
          <span className="ml-2 text-text-dim font-normal text-sm">
            {robotIp}:{runtimePort}
          </span>
        </h2>
        <Button
          variant={showNav ? "secondary" : "ghost"}
          disabled={nav !== null && !nav.present}
          className="text-xs px-3 py-1.5 h-8"
          onClick={() => setShowNav((on) => !on)}
        >
          Labels{nav?.present && showNav ? ` · ${nav.maskHz.toFixed(0)} Hz` : ""}
        </Button>
      </div>

      <VideoFeed
        baseUrl={`http://${robotIp}:${runtimePort}`}
        transport={transport}
        showNav={showNav}
        nav={nav}
        reconnectKey={reconnectKey}
        onStreamState={setStreamState}
        className="w-full -mx-4 sm:mx-0 sm:w-[720px] sm:max-w-full sm:rounded-[var(--radius-card)]"
        maxHeight="78dvh"
      />

      {showNav && nav !== null && !nav.present ? (
        <Banner tone="error">
          The robot has no navigable-path runner: add a{" "}
          <code>nav:</code> block to the firmware YAML and ship{" "}
          <code>navseg.onnx</code> next to it.
        </Banner>
      ) : null}

      {showNav && nav !== null && nav.present && streamState === "live" ? (
        <span className="text-xs text-text-dim">
          Overlay enabled — green = navigable, amber = caution. Masks lag
          the video by up to a frame; if they stop arriving, check the nav
          runner on the robot.
        </span>
      ) : null}

      {streamState === "error" ? (
        <Banner tone="error">
          Camera stream failed on <code>{url}</code>. Check that bebop-linux
          is running with a <code>video:</code> config block and that the
          camera is plugged in.
        </Banner>
      ) : null}

      <Card>
        <div className="flex flex-col items-center gap-3 px-2 py-2 w-full">
          <div className="flex flex-wrap items-center justify-center gap-8">
            <PtzJoystick
              onRate={ptz.onRate}
              onStop={ptz.onStop}
              disabled={!ptzReady}
            />

            <div className="flex flex-col items-center gap-2">
              <div className="flex items-center gap-2 text-sm text-text-dim">
                <span>
                  pan {camera ? `${camera.panDeg.toFixed(1)}°` : "—"}
                  {" · "}
                  tilt {camera ? `${camera.tiltDeg.toFixed(1)}°` : "—"}
                </span>
                {camera?.moving ? (
                  <span className="inline-block w-2 h-2 rounded-full bg-accent animate-pulse" />
                ) : null}
              </div>
              <Button
                variant="ghost"
                disabled={!ptzReady}
                onClick={ptz.center}
              >
                Center
              </Button>
            </div>
          </div>

          {camera !== null && !camera.present ? (
            <span className="text-xs text-text-dim">
              PTZ unavailable — this robot has no{" "}
              <code>video:</code> config, so the firmware reports no
              camera.
            </span>
          ) : null}
          {camera !== null && camera.present && conn !== "connected" ? (
            <span className="text-xs text-text-dim">
              Reconnecting to the runtime… controls disabled.
            </span>
          ) : null}
          {ptz.error ? <Banner tone="error">{ptz.error}</Banner> : null}
        </div>
      </Card>

      <div className="flex items-center gap-4">
        {streamState !== "loading" ? (
          <Button
            variant="secondary"
            onClick={() => setReconnectKey((n) => n + 1)}
          >
            Reconnect
          </Button>
        ) : null}
        <Button variant="ghost" onClick={onBack}>
          {backLabel}
        </Button>
      </div>
    </div>
  );
}
