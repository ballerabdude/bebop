import { useCallback, useEffect, useRef, useState } from "react";

import { Banner, Button, Card, Spinner } from "../components/ui";
import { getOrCreateRuntimeTransport } from "../runtime";
import type {
  CameraView,
  NavMaskView,
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

/// OBSBOT Tiny 2 gimbal limits (UVC-reported; see the recon table in
/// `firmware/bebop-linux/src/video.rs`). The firmware clamps too — this
/// is just so relative jogging doesn't accumulate past the stops.
const PAN_LIMIT_DEG = 130;
const TILT_LIMIT_DEG = 90;

/// Joystick rate limits (deg/s). The gimbal only accepts absolute
/// targets, so the joystick's deflection is integrated client-side;
/// these cap how fast the commanded target ramps. The OBSBOT Tiny 2
/// slews at ~65°/s pan at default UVC speeds, so 60/40 keeps the target
/// just ahead of the hardware without whipping it around.
const MAX_PAN_RATE_DEG_S = 60;
const MAX_TILT_RATE_DEG_S = 40;

/// Integration tick for the PTZ rate loop (10 Hz): fast enough that a
/// held deflection ramps the target smoothly, slow enough that every
/// tick's absolute command gets its own WS round-trip.
const PTZ_TICK_MS = 100;

const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, v));

/// Nav-overlay paint colors, [r, g, b, a] per label. Navigable is a
/// translucent green wash, caution a warmer amber; blocked cells stay
/// transparent so the underlying video remains fully visible — the point
/// of the overlay is "where can the robot go", not a full segmentation.
const NAV_COLOR_NAVIGABLE: [number, number, number, number] = [46, 204, 113, 90];
const NAV_COLOR_CAUTION: [number, number, number, number] = [241, 196, 15, 115];

/// Scratch canvas the mask grid is painted into at native resolution
/// (label → RGBA, one pixel per cell) before being scaled up onto the
/// overlay canvas with smoothing disabled — crisp cell boundaries and
/// a per-frame cost of one 160×90 putImageData + one drawImage.
let navMaskScratch: HTMLCanvasElement | null = null;
function paintNavMask(mask: NavMaskView): HTMLCanvasElement {
  if (!navMaskScratch) {
    navMaskScratch = document.createElement("canvas");
  }
  if (navMaskScratch.width !== mask.width) navMaskScratch.width = mask.width;
  if (navMaskScratch.height !== mask.height) navMaskScratch.height = mask.height;
  const ctx = navMaskScratch.getContext("2d");
  if (!ctx) return navMaskScratch;
  const image = ctx.createImageData(mask.width, mask.height);
  const { grid } = mask;
  for (let i = 0; i < grid.length && i < image.data.length / 4; i++) {
    const label = grid[i];
    const [r, g, b, a] =
      label === 1 ? NAV_COLOR_NAVIGABLE : label === 2 ? NAV_COLOR_CAUTION : [0, 0, 0, 0];
    const o = i * 4;
    image.data[o] = r;
    image.data[o + 1] = g;
    image.data[o + 2] = b;
    image.data[o + 3] = a;
  }
  ctx.putImageData(image, 0, 0);
  return navMaskScratch;
}

/// Composite the latest mask onto the overlay canvas, fitting the video's
/// *displayed* rectangle (`<img>` is `object-contain`, so the video may
/// be letterboxed inside the card — the mask must land on the video
/// pixels, not the pillarbox bars). The canvas is sized in device pixels
/// for crisp rendering; a resize is picked up on the next frame.
function drawNavOverlay(
  canvas: HTMLCanvasElement | null,
  img: HTMLImageElement | null,
  mask: NavMaskView | null,
): void {
  if (!canvas || !mask) return;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  if (cssW === 0 || cssH === 0) return;
  const natW = img?.naturalWidth || 1280;
  const natH = img?.naturalHeight || 720;
  const scale = Math.min(cssW / natW, cssH / natH);
  const vidW = natW * scale;
  const vidH = natH * scale;
  const offX = (cssW - vidW) / 2;
  const offY = (cssH - vidH) / 2;
  const dpr = window.devicePixelRatio || 1;
  const pxW = Math.max(1, Math.round(vidW * dpr));
  const pxH = Math.max(1, Math.round(vidH * dpr));
  if (canvas.width !== pxW || canvas.height !== pxH) {
    canvas.width = pxW;
    canvas.height = pxH;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, pxW, pxH);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    paintNavMask(mask),
    Math.round(offX * dpr),
    Math.round(offY * dpr),
    pxW,
    pxH,
  );
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
 * error state with a retry button.
 *
 * PTZ rides the same runtime WS the motor bench uses: `SetCameraPose`
 * is not mode-gated (moving the camera can't hurt anyone), so the
 * controls are live in any mode. The gimbal is position-only (absolute
 * UVC pan/tilt targets), so the joystick integrates stick deflection
 * into absolute targets at a fixed tick, clamped against the known
 * gimbal limits. Gestures compose relatively: while the gimbal is
 * still slewing, a new press continues from the last commanded target
 * (telemetry's actual pose trails a mid-flight move); once settled it
 * re-syncs to the actual pose (the gimbal occasionally applies
 * horizon compensation, so we never track locally). A 30° move
 * settles in ~0.5-0.7 s; the "moving" dot in the pose readout comes
 * straight from the firmware's settle detector.
 */
export function VideoScreen({
  robotIp,
  runtimePort = 9090,
  onBack,
  backLabel = "Back",
}: VideoScreenProps) {
  // Cache-busting nonce: bumping it re-assigns `src`, which tears the
  // multipart stream down and reconnects — the retry path.
  const [nonce, setNonce] = useState(0);
  // "loading" until the first frame paints, "live" while streaming,
  // "error" when the endpoint is unreachable or answers 503.
  const [state, setState] = useState<"loading" | "live" | "error">("loading");
  const [conn, setConn] = useState<RuntimeConnectionState>("disconnected");
  const [camera, setCamera] = useState<CameraView | null>(null);
  const [ptzError, setPtzError] = useState<string | null>(null);
  // Nav overlay: off by default so video playback is untouched until
  // the operator opts in. `nav` mirrors the telemetry summary (present /
  // provider / rate) for the toggle's state hint.
  const [showNav, setShowNav] = useState(false);
  const [nav, setNav] = useState<NavView | null>(null);
  const [navMaskFresh, setNavMaskFresh] = useState(false);

  const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
  // Latest camera view for relative jog math — a ref so rapid button
  // taps don't race React state (the transport acks are faster than a
  // telemetry frame, so the ref is the freshest pose we've seen).
  const cameraRef = useRef<CameraView | null>(null);
  cameraRef.current = camera;
  const url = `http://${robotIp}:${runtimePort}/video`;
  // Cache-busted variant actually assigned to the <img>: bumping the
  // nonce remounts the element (see `key` below), tearing the multipart
  // stream down and reconnecting — the retry path.
  const streamSrc = nonce ? `${url}?r=${nonce}` : url;

  // WebKit (Tauri's engine on Linux) keeps MJPEG `<img>` downloads
  // running even after the element leaves the DOM — the classic
  // "closed the screen but the connection never stops" leak. Removing
  // `src` forces the engine to abort the request.
  //
  // Two React details make this cleanup subtle:
  //   1. On unmount React nulls `ref.current` before the `useEffect`
  //      cleanup runs, so the ref is useless there — capture the
  //      element in the closure instead.
  //   2. StrictMode (dev only) runs setup → cleanup → setup without
  //      re-rendering, so the cleanup would abort the stream it just
  //      started; setup restores the attribute to restart it.
  const imgRef = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const img = imgRef.current;
    if (img && img.getAttribute("src") !== streamSrc) {
      img.src = streamSrc;
    }
    return () => {
      img?.removeAttribute("src");
    };
  }, [streamSrc]);

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

  // --- Nav overlay -----------------------------------------------------
  // Masks are stored in a ref and painted by a rAF loop; the overlay
  // runs at the mask rate (not the video rate) and re-fits the video's
  // displayed rect every frame so PTZ moves / window resizes can't
  // skew it. "Freshness" (mask arrived within the last ~1.5 s) is the
  // only React state the loop feeds back, so 10 Hz masks don't
  // re-render the whole screen.
  const navCanvasRef = useRef<HTMLCanvasElement>(null);
  const latestMaskRef = useRef<{ mask: NavMaskView; arrivedAt: number } | null>(
    null,
  );
  useEffect(() => {
    if (!showNav) return;
    let disposed = false;
    void transport.subscribeNav(10).catch(() => {
      /* nav absent / connection loss — surfaced by the hint below */
    });
    const offMask = transport.onNavMask((mask) => {
      latestMaskRef.current = { mask, arrivedAt: Date.now() };
    });
    let rafId = 0;
    let lastFreshState = false;
    const draw = () => {
      if (disposed) return;
      const latest = latestMaskRef.current;
      if (latest) {
        drawNavOverlay(navCanvasRef.current, imgRef.current, latest.mask);
      }
      const fresh = !!latest && Date.now() - latest.arrivedAt < 1_500;
      if (fresh !== lastFreshState) {
        lastFreshState = fresh;
        setNavMaskFresh(fresh);
      }
      rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);
    return () => {
      disposed = true;
      cancelAnimationFrame(rafId);
      offMask();
      void transport.unsubscribeNav().catch(() => {
        /* best effort — reconnect resume logic re-arms anyway */
      });
      latestMaskRef.current = null;
      const canvas = navCanvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      setNavMaskFresh(false);
    };
  }, [showNav, transport]);

  // PTZ rate control. `setCameraPose` only takes absolute pan/tilt
  // targets, so the joystick is integrated client-side: while input is
  // active, a fixed tick ramps the commanded target at a rate
  // proportional to stick deflection. Releasing just stops sending —
  // the gimbal holds the last commanded pose (position-controlled, no
  // "stop" to send, unlike the drive). A new gesture must move
  // *relative* to where the gimbal is heading: while it is still
  // settling toward the previous command, telemetry's actual pose
  // lags behind, so seeding from it would snap the gimbal back and
  // undo part of the last move. We therefore seed from the last
  // commanded target while the firmware reports `moving`, and from the
  // actual pose once settled (so external clients / horizon
  // compensation drift are picked up between moves).
  const rateRef = useRef({ pan: 0, tilt: 0 });
  const targetRef = useRef<{ pan: number; tilt: number } | null>(null);
  /// Last pose we commanded — survives release (unlike `targetRef`)
  /// so the next gesture can continue from it.
  const lastCommandedRef = useRef<{ pan: number; tilt: number } | null>(null);

  // Coalesced pose sender — same single-in-flight / latest-pending
  // pattern as the motor bench's `sendTwist`, so the tick's command
  // stream can't queue one WS request per step on a slow link.
  const transportRef = useRef(transport);
  transportRef.current = transport;
  const poseInFlightRef = useRef(false);
  const posePendingRef = useRef<{ pan: number; tilt: number } | null>(null);

  const sendPose = useCallback(async (pan: number, tilt: number) => {
    const t = transportRef.current;
    if (poseInFlightRef.current) {
      posePendingRef.current = { pan, tilt };
      return;
    }
    poseInFlightRef.current = true;
    lastCommandedRef.current = { pan, tilt };
    try {
      await t.setCameraPose(pan, tilt);
    } catch (e) {
      setPtzError(e instanceof Error ? e.message : String(e));
    } finally {
      poseInFlightRef.current = false;
      const next = posePendingRef.current;
      posePendingRef.current = null;
      if (next) {
        void Promise.resolve().then(() => sendPose(next.pan, next.tilt));
      }
    }
  }, []);

  const onPtzRate = useCallback((panRate: number, tiltRate: number) => {
    setPtzError(null);
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
    rateRef.current = { pan: panRate, tilt: tiltRate };
  }, []);

  const onPtzStop = useCallback(() => {
    rateRef.current = { pan: 0, tilt: 0 };
    targetRef.current = null;
  }, []);

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

  const centerCamera = useCallback(() => {
    setPtzError(null);
    void sendPose(0, 0);
  }, [sendPose]);

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

      <Card>
        <div className="relative aspect-video w-[720px] max-w-full bg-black overflow-hidden rounded-[var(--radius-card)]">
          {state === "loading" ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
              <Spinner />
              <span className="text-text-dim text-sm">
                Waiting for the first frame…
              </span>
            </div>
          ) : null}
          {state === "error" ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-8 text-center">
              <span className="text-text-dim text-sm">
                No video stream available. Either the firmware is not
                reachable, or the robot has no{" "}
                <code>video:</code> section in its YAML (the endpoint
                answers 503 then).
              </span>
            </div>
          ) : null}
          {/* The img stays mounted in every state: during "error" it has
              no visible frames anyway, and keeping one element lets the
              retry reassignment reuse the same DOM node. `key` forces a
              fresh element on retry so a failed request can't serve a
              cached state. */}
          <img
            key={nonce}
            ref={imgRef}
            src={streamSrc}
            alt="Robot live camera feed"
            className="w-full h-full object-contain"
            onLoad={() => setState("live")}
            onError={() => setState("error")}
          />
          {/* Nav overlay canvas: stacked above the video, pointer-events
              none so it never eats PTZ/keyboard input. Sits inside the
              letterbox-fitted video rect (see drawNavOverlay), covering
              the full container but only painting video pixels. */}
          {showNav ? (
            <canvas
              ref={navCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
          ) : null}
          {showNav && !navMaskFresh ? (
            <div className="absolute top-2 left-2 pointer-events-none rounded-[var(--radius-card)] bg-bg-elev/80 px-2 py-1 text-xs text-text-dim">
              {nav?.present === false
                ? "nav unavailable"
                : "waiting for masks…"}
            </div>
          ) : null}
        </div>
      </Card>

      {showNav && nav !== null && !nav.present ? (
        <Banner tone="error">
          The robot has no navigable-path runner: add a{" "}
          <code>nav:</code> block to the firmware YAML and ship{" "}
          <code>navseg.onnx</code> next to it.
        </Banner>
      ) : null}

      {showNav && nav !== null && nav.present && !navMaskFresh ? (
        <span className="text-xs text-text-dim">
          Overlay enabled but no masks arriving — the nav model may still
          be loading, or the camera is down.
        </span>
      ) : null}

      {state === "error" ? (
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
              onRate={onPtzRate}
              onStop={onPtzStop}
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
                onClick={centerCamera}
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
          {ptzError ? <Banner tone="error">{ptzError}</Banner> : null}
        </div>
      </Card>

      <div className="flex items-center gap-4">
        {state !== "loading" ? (
          <Button variant="secondary" onClick={() => setNonce((n) => n + 1)}>
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

/// Virtual PTZ joystick, modelled on the motor bench's `DriveJoystick`.
/// Drag the knob from the centre: right/left ramps pan, up/down ramps
/// tilt, at a rate proportional to deflection (full throw = the MAX_*
/// rates above). Release snaps the knob back and simply stops sending
/// — the gimbal is position-controlled and holds its last commanded
/// pose. WASD / arrow keys compose the same rates for keyboard
/// operators; releasing the last key holds the pose.
function PtzJoystick({
  onRate,
  onStop,
  disabled,
}: {
  onRate: (panRate: number, tiltRate: number) => void;
  onStop: () => void;
  disabled: boolean;
}) {
  const padRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);
  const [knob, setKnob] = useState({ x: 0, y: 0 });

  const apply = useCallback(
    (nx: number, ny: number) => {
      // Joystick convention: right = pan right (+pan), up = tilt up
      // (+tilt). The pad's ny grows downward, hence the negation.
      setKnob({ x: nx, y: ny });
      onRate(nx * MAX_PAN_RATE_DEG_S, -ny * MAX_TILT_RATE_DEG_S);
    },
    [onRate],
  );

  const release = useCallback(() => {
    draggingRef.current = false;
    setKnob({ x: 0, y: 0 });
    onStop();
  }, [onStop]);

  const handleMove = useCallback(
    (clientX: number, clientY: number) => {
      const pad = padRef.current;
      if (!pad) return;
      const rect = pad.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const radius = rect.width / 2;
      let nx = (clientX - cx) / radius;
      let ny = (clientY - cy) / radius;
      const mag = Math.hypot(nx, ny);
      if (mag > 1) {
        nx /= mag;
        ny /= mag;
      }
      apply(nx, ny);
    },
    [apply],
  );

  // Keyboard PTZ: WASD + arrows compose a rate; releasing the last key
  // holds the pose. Bound only while enabled — unlike drive commands
  // (firmware mode-gated), PTZ poses go through on any connection, so a
  // dead-transport screen must not leak held-key rates into it.
  useEffect(() => {
    if (disabled) return;
    const keys = new Set<string>();
    const up = ["w", "arrowup"];
    const down = ["s", "arrowdown"];
    const left = ["a", "arrowleft"];
    const right = ["d", "arrowright"];

    const compute = () => {
      let pan = 0;
      let tilt = 0;
      if (left.some((k) => keys.has(k))) pan -= MAX_PAN_RATE_DEG_S;
      if (right.some((k) => keys.has(k))) pan += MAX_PAN_RATE_DEG_S;
      if (up.some((k) => keys.has(k))) tilt += MAX_TILT_RATE_DEG_S;
      if (down.some((k) => keys.has(k))) tilt -= MAX_TILT_RATE_DEG_S;
      return { pan, tilt };
    };

    const isPtzKey = (k: string) =>
      up.includes(k) || down.includes(k) || left.includes(k) || right.includes(k);

    const onKeyDown = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (!isPtzKey(k)) return;
      e.preventDefault();
      keys.add(k);
      const { pan, tilt } = compute();
      onRate(pan, tilt);
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (!isPtzKey(k)) return;
      keys.delete(k);
      if (keys.size === 0) {
        onStop();
      } else {
        const { pan, tilt } = compute();
        onRate(pan, tilt);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [onRate, onStop, disabled]);

  const knobPx = padRef.current ? padRef.current.offsetWidth / 2 : 88;

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        ref={padRef}
        className={`relative w-44 h-44 rounded-full border border-border bg-bg-elev-2/60 select-none touch-none ${
          disabled ? "opacity-40 cursor-not-allowed" : "cursor-grab active:cursor-grabbing"
        }`}
        onPointerDown={(e) => {
          if (disabled) return;
          draggingRef.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          handleMove(e.clientX, e.clientY);
        }}
        onPointerMove={(e) => {
          if (!draggingRef.current) return;
          handleMove(e.clientX, e.clientY);
        }}
        onPointerUp={release}
        onPointerCancel={release}
      >
        {/* Cardinal markers */}
        <span className="absolute left-1/2 top-2 -translate-x-1/2 text-[10px] text-text-dim">
          tilt+
        </span>
        <span className="absolute left-1/2 bottom-2 -translate-x-1/2 text-[10px] text-text-dim">
          tilt−
        </span>
        <span className="absolute top-1/2 left-2 -translate-y-1/2 text-[10px] text-text-dim">
          L
        </span>
        <span className="absolute top-1/2 right-2 -translate-y-1/2 text-[10px] text-text-dim">
          R
        </span>
        {/* Knob */}
        <div
          className="absolute w-10 h-10 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow-md"
          style={{
            left: `calc(50% + ${(knob.x * knobPx).toFixed(1)}px)`,
            top: `calc(50% + ${(knob.y * knobPx).toFixed(1)}px)`,
          }}
          aria-hidden
        />
      </div>
      <div className="text-[11px] text-text-dim">
        Drag to aim · WASD / arrows
      </div>
    </div>
  );
}