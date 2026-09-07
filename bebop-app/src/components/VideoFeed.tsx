// Live MJPEG stream tile shared by the video screen and the teleop
// screen. Renders a bebop-vision `:9092/video` multipart stream in an
// `<img>` (the browser paints each JPEG part as it arrives — no
// JavaScript decode loop) with:
//
//   * loading / error placeholder states,
//   * automatic reconnection: an errored or stalled stream is
//     re-requested with exponential backoff (see `autoReconnect`), so
//     the feed comes back on its own after a robot reboot or a
//     bebop-vision restart,
//   * a parent-driven reconnect: bump `reconnectKey` to tear the
//     multipart stream down and re-request it right away.
//
// The stream is served by the bebop-vision process, which owns the
// Orbbec cameras exclusively; this component is just another HTTP
// subscriber. (The legacy nav-mask overlay was removed with the
// OBSBOT pipeline, plan §9 Stage 3 — the BEV is now its own `bev`
// stream, the planner's fused occupancy grid rendered server-side.)

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Button, Spinner } from "./ui";
export type VideoStreamState = "loading" | "live" | "error";

interface VideoFeedProps {
  /// Runtime base URL (`http://<ip>:<port>`); the stream itself is
  /// `<baseUrl>/video`.
  baseUrl: string;
  /// Override for the video stream URL. The operator stream is served by
  /// the bebop-vision process on its own port (9092), separate from the
  /// firmware runtime port — pass `http://<ip>:9092/video` here. When
  /// omitted, falls back to `<baseUrl>/video` (legacy firmware stream).
  videoUrl?: string;
  /// Stream selector understood by the bebop-vision server: color_near |
  /// color_far | depth_near | depth_far | bev. Appended as ?stream= to
  /// the URL.
  stream?: string;
  /// Bump to tear the multipart stream down and reconnect.
  reconnectKey: number;
  /// Self-healing reconnects. When the request errors (robot down, 503)
  /// or goes silent (no frame event for `stallTimeoutMs` — covers
  /// connections that die mid-stream without an error event and loads
  /// that hang on an unreachable host), the stream is re-requested
  /// automatically with exponential backoff. Default true.
  autoReconnect?: boolean;
  /// First reconnect delay (ms); doubles per consecutive failure.
  /// Default 1000.
  reconnectBaseDelayMs?: number;
  /// Backoff ceiling (ms). Default 15000.
  reconnectMaxDelayMs?: number;
  /// Silence threshold (ms): no frame event for this long counts as a
  /// stalled stream and triggers a reconnect. Default 12000 (streams
  /// run at 15 fps; anything past ~10s of nothing is dead). 0 disables
  /// the stall watchdog (error-triggered reconnects still apply).
  stallTimeoutMs?: number;
  /// Stream lifecycle reports for the parent (retry buttons, etc.).
  onStreamState?: (state: VideoStreamState) => void;
  /// Classes for the outer container — sizing (width / h-full) and
  /// decoration (rounded / border / negative margins for mobile
  /// full-bleed). The element is the black letterbox surface; its
  /// *aspect* is owned by the feed (see below), so don't pass
  /// `aspect-*` utilities.
  className?: string;
  /// Cap the container height (CSS length, e.g. "72dvh") so a wide
  /// stream on a landscape phone / short window doesn't overflow the
  /// viewport. The video object-contains inside; the overlay fits the
  /// real video rect either way. Do NOT pass this when the parent
  /// controls the height (fullscreen-style layouts).
  maxHeight?: string;
  /// Optional extra chrome layered on top of the video (badges, PTZ
  /// hints). Rendered above the overlay canvas, below nothing —
  /// pointer events pass through unless the node opts in.
  children?: ReactNode;
}

export function VideoFeed({
  baseUrl,
  videoUrl,
  stream,
  reconnectKey,
  autoReconnect = true,
  reconnectBaseDelayMs = 1000,
  reconnectMaxDelayMs = 15000,
  stallTimeoutMs = 12000,
  onStreamState,
  className = "",
  maxHeight,
  children,
}: VideoFeedProps) {
  // "loading" until the first frame paints, "live" while streaming,
  // "error" when the endpoint is unreachable or answers 503.
  const [state, setState] = useState<VideoStreamState>("loading");
  // Natural size of the decoded stream, measured off the first frame.
  // The container adopts this aspect so the video is never letterboxed
  // inside a hard-coded box: the YAML *requests* 1280x720 but the UVC
  // driver negotiates the nearest mode the camera offers, which may
  // not be 16:9 — and a mismatched box is exactly how the nav overlay
  // ends up painted on the pillarbox bars instead of the video. Until
  // the first frame decodes we assume 16:9 (the loading placeholder is
  // in there anyway) and re-measure on every reconnect.
  const [frameSize, setFrameSize] = useState<{ w: number; h: number } | null>(
    null,
  );

  // ------------------------------------------------------ auto reconnect
  // The manual path is `reconnectKey` (parent-driven); the automatic
  // path is an internal nonce bumped by the backoff timer. Both feed
  // one composite key that (a) remounts the <img>, tearing the
  // multipart stream down, and (b) cache-busts the URL.
  const [autoRetry, setAutoRetry] = useState(0);
  // Consecutive failed attempts — drives the exponential backoff. Only
  // a painted frame resets it, so a down robot backs off to the cap
  // instead of hammering the server every second.
  const attemptsRef = useRef(0);
  // Last sign of life (request start or a decoded frame). The stall
  // watchdog compares this against `stallTimeoutMs`.
  const lastFrameAtRef = useRef(Date.now());
  // Pending backoff timer + its ETA (for the countdown in the error
  // placeholder). Ref mirrors the state so the watchdog interval can
  // read it without re-subscribing.
  const retryTimerRef = useRef<number | null>(null);
  const retryAtRef = useRef<number | null>(null);
  const [retryAt, setRetryAt] = useState<number | null>(null);
  // Ticks once per second while a retry is pending, so the countdown
  // renders. (The watchdog interval owns the ticking.)
  const [tick, setTick] = useState(() => Date.now());

  const base = videoUrl ?? `${baseUrl}/video`;
  const url = stream ? `${base}?stream=${stream}` : base;
  const streamKey = `${reconnectKey}.${autoRetry}`;
  const streamSrc = reconnectKey || autoRetry ? `${url}&r=${streamKey}` : url;

  const report = (s: VideoStreamState) => {
    setState(s);
    onStreamState?.(s);
  };

  const clearPendingRetry = () => {
    if (retryTimerRef.current !== null) {
      window.clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    retryAtRef.current = null;
    setRetryAt(null);
  };

  // Error / stall → schedule the next attempt. Backoff doubles per
  // consecutive failure up to the cap; the timer bumps `autoRetry`,
  // which remounts the <img> and starts a fresh loading window.
  const scheduleRetry = () => {
    if (retryTimerRef.current !== null) return;
    const delay = Math.min(
      reconnectBaseDelayMs * 2 ** attemptsRef.current,
      reconnectMaxDelayMs,
    );
    attemptsRef.current += 1;
    const at = Date.now() + delay;
    retryAtRef.current = at;
    setRetryAt(at);
    setTick(Date.now());
    retryTimerRef.current = window.setTimeout(() => {
      retryTimerRef.current = null;
      retryAtRef.current = null;
      setRetryAt(null);
      lastFrameAtRef.current = Date.now();
      setAutoRetry((n) => n + 1);
    }, delay);
  };

  // Immediate retry (the "Retry now" button): drop the pending backoff
  // and remount right away.
  const retryNow = () => {
    clearPendingRetry();
    lastFrameAtRef.current = Date.now();
    setAutoRetry((n) => n + 1);
  };

  const secondsLeft =
    retryAt !== null ? Math.max(0, Math.ceil((retryAt - tick) / 1000)) : null;

  // A reconnect request resets to "loading" (the remounted <img> fires
  // onLoad / onError to move on from there) and drops the measured
  // frame size — the new stream may negotiate a different mode. A
  // *manual* (parent) bump also resets the backoff ladder: the operator
  // asked for a fresh start. Cancels any pending auto-retry so a late
  // timer can't double-remount after the parent already reconnected.
  const prevParentKeyRef = useRef(reconnectKey);
  useEffect(() => {
    if (prevParentKeyRef.current !== reconnectKey) {
      prevParentKeyRef.current = reconnectKey;
      attemptsRef.current = 0;
    }
    clearPendingRetry();
    lastFrameAtRef.current = Date.now();
    setState("loading");
    setFrameSize(null);
    onStreamState?.("loading");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamKey]);

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
    // New request in flight: open a fresh stall window.
    lastFrameAtRef.current = Date.now();
    const img = imgRef.current;
    if (img && img.getAttribute("src") !== streamSrc) {
      img.src = streamSrc;
    }
    return () => {
      img?.removeAttribute("src");
    };
  }, [streamSrc]);

  // Kill a still-pending retry on unmount (the remount path needs it
  // alive — it *is* the reconnect mechanism).
  useEffect(() => {
    return () => {
      if (retryTimerRef.current !== null) {
        window.clearTimeout(retryTimerRef.current);
      }
    };
  }, []);

  // Stall watchdog: once per second, reconnect if nothing has arrived
  // for `stallTimeoutMs`. This is what catches the ugly failure modes
  // onError misses — a multipart connection dying mid-stream (the img
  // just goes silent, no error event) and a TCP connect hanging on an
  // unreachable host. Skipped while the tab is hidden (background
  // throttling stalls decodes too); returning to the tab re-arms the
  // window instead of punishing the stream for the pause. Also ticks
  // the reconnect countdown.
  useEffect(() => {
    if (!autoReconnect || stallTimeoutMs <= 0) return;
    const onVisibility = () => {
      if (!document.hidden) lastFrameAtRef.current = Date.now();
    };
    document.addEventListener("visibilitychange", onVisibility);
    const interval = window.setInterval(() => {
      if (retryAtRef.current !== null) setTick(Date.now());
      if (document.hidden) return;
      if (
        retryTimerRef.current === null &&
        Date.now() - lastFrameAtRef.current > stallTimeoutMs
      ) {
        report("error");
        scheduleRetry();
      }
    }, 1000);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.clearInterval(interval);
    };
    // report / scheduleRetry close over stable setters and config
    // props; re-subscribing the interval each render would be worse.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoReconnect, stallTimeoutMs]);

  return (
    <div
      className={`relative overflow-hidden bg-black ${className}`}
      style={{
        // Container aspect follows the decoded frame (16:9 until the
        // first frame measures otherwise). An explicit height from the
        // parent (h-full fullscreen layouts) overrides this; a
        // maxHeight cap leaves the box wider than the video, and the
        // overlay's rect math handles the resulting letterbox.
        aspectRatio: frameSize
          ? `${frameSize.w} / ${frameSize.h}`
          : "16 / 9",
        ...(maxHeight ? { maxHeight } : {}),
      }}
    >
      {state === "loading" ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <Spinner />
          <span className="text-text-dim text-sm">
            Waiting for the first frame…
          </span>
        </div>
      ) : null}
      {state === "error" ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center">
          {autoReconnect ? (
            <>
              <Spinner />
              <span className="text-text-dim text-sm">
                Stream lost — reconnecting
                {secondsLeft !== null ? ` in ${secondsLeft}s` : ""}…
              </span>
              <Button
                variant="ghost"
                className="text-xs h-8"
                onClick={retryNow}
              >
                Retry now
              </Button>
            </>
          ) : (
            <span className="text-text-dim text-sm">
              No video stream available. Either the firmware is not
              reachable, or the robot has no{" "}
              <code>video:</code> section in its YAML (the endpoint
              answers 503 then).
            </span>
          )}
        </div>
      ) : null}
      {/* The img stays mounted in every state: during "error" it has
          no visible frames anyway, and keeping one element lets the
          retry reassignment reuse the same DOM node. `key` forces a
          fresh element on retry so a failed request can't serve a
          cached state. */}
      <img
        key={streamKey}
        ref={imgRef}
        src={streamSrc}
        alt="Robot live camera feed"
        className="w-full h-full object-contain"
        onLoad={(e) => {
          // A frame arrived — the connection is alive. Re-arm the
          // stall watchdog, reset the backoff ladder, and cancel any
          // pending reconnect (a late frame beat the retry timer).
          lastFrameAtRef.current = Date.now();
          attemptsRef.current = 0;
          if (retryTimerRef.current !== null) {
            clearPendingRetry();
          }
          // Measure the negotiated stream mode off the first decoded
          // frame so the container can adopt its exact aspect (see
          // frameSize). naturalWidth/Height are stable for MJPEG (the
          // decoder reuses one frame size), so the memo check keeps
          // re-loads from re-rendering the layout.
          const el = e.currentTarget;
          if (el.naturalWidth > 0 && el.naturalHeight > 0) {
            setFrameSize((prev) =>
              prev &&
              prev.w === el.naturalWidth &&
              prev.h === el.naturalHeight
                ? prev
                : { w: el.naturalWidth, h: el.naturalHeight },
            );
          }
          report("live");
        }}
        onError={() => {
          report("error");
          if (autoReconnect) scheduleRetry();
        }}
      />
      {children}
    </div>
  );
}
