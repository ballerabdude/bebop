// Live hardware-encoded video tile: plays the bebop-vision NVENC stream
// (`/video?stream=color_near&codec=h265`) in a plain `<video>`. The
// fragmented-MP4 H.265 bitstream is ~8x smaller on the wire than MJPEG,
// but only decodes where the webview supports HEVC (Safari, Android
// WebView, Chromium builds with HEVC). When it can't, the element fires
// MEDIA_ERR_SRC_NOT_SUPPORTED and we call `onUnsupported` so the parent
// can fall back to the MJPEG `<img>` path.
//
// This mirrors the reconnection behaviour of the MJPEG `VideoFeed`: an
// errored or stalled stream (no `timeupdate`/`progress` for
// `stallTimeoutMs`) is re-requested with exponential backoff.

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Button, Spinner } from "./ui";
import type { VideoStreamState } from "./VideoFeed";

export function encodedVideoSupported(): boolean {
  return (
    typeof window !== "undefined" && typeof HTMLVideoElement !== "undefined"
  );
}

export function isMobileClient(): boolean {
  if (typeof navigator === "undefined") return false;
  return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
}

interface EncodedVideoFeedProps {
  baseUrl: string;
  videoUrl?: string;
  stream?: string;
  codec: string;
  reconnectKey: number;
  autoReconnect?: boolean;
  reconnectBaseDelayMs?: number;
  reconnectMaxDelayMs?: number;
  stallTimeoutMs?: number;
  onStreamState?: (state: VideoStreamState) => void;
  /// Called when the webview cannot decode this codec; the parent should
  /// switch this tile to the MJPEG path.
  onUnsupported?: () => void;
  className?: string;
  maxHeight?: string;
  children?: ReactNode;
}

export function EncodedVideoFeed({
  baseUrl,
  videoUrl,
  stream,
  codec,
  reconnectKey,
  autoReconnect = true,
  reconnectBaseDelayMs = 1000,
  reconnectMaxDelayMs = 15000,
  stallTimeoutMs = 4000,
  onStreamState,
  onUnsupported,
  className = "",
  maxHeight,
  children,
}: EncodedVideoFeedProps) {
  const [state, setState] = useState<VideoStreamState>("loading");
  const [frameSize, setFrameSize] = useState<{ w: number; h: number } | null>(
    null,
  );
  const [autoRetry, setAutoRetry] = useState(0);
  const attemptsRef = useRef(0);
  const lastFrameAtRef = useRef(Date.now());
  const retryTimerRef = useRef<number | null>(null);
  const retryAtRef = useRef<number | null>(null);
  const [retryAt, setRetryAt] = useState<number | null>(null);
  const [tick, setTick] = useState(() => Date.now());
  const videoRef = useRef<HTMLVideoElement>(null);
  // Last observed playback position — the liveness signal. Relying on
  // `timeupdate`/`progress` events is unreliable in some webviews for
  // live streams; sampling `currentTime` works everywhere.
  const lastTimeRef = useRef(0);

  const base = videoUrl ?? `${baseUrl}/video`;
  const streamKey = `${reconnectKey}.${autoRetry}`;
  const params = new URLSearchParams();
  if (stream) params.set("stream", stream);
  params.set("codec", codec);
  params.set("r", streamKey);
  // Phones get a smaller, lower-bitrate H.265 stream: far less bandwidth
  // than 1280x800 (still a fraction of MJPEG) and much less to decode,
  // which is what keeps multiple tiles stable on mobile.
  if (isMobileClient()) {
    params.set("width", "960");
    params.set("height", "600");
    params.set("bitrate", "1200000");
  }  const src = `${base}?${params.toString()}`;

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

  const retryNow = () => {
    clearPendingRetry();
    lastFrameAtRef.current = Date.now();
    setAutoRetry((n) => n + 1);
  };

  const secondsLeft =
    retryAt !== null ? Math.max(0, Math.ceil((retryAt - tick) / 1000)) : null;

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

  useEffect(() => {
    lastFrameAtRef.current = Date.now();
    const video = videoRef.current;
    if (video && video.getAttribute("src") !== src) {
      video.src = src;
      video.load();
    }
    return () => {
      const v = video;
      if (v) {
        v.removeAttribute("src");
        v.load();
      }
    };
  }, [src]);

  useEffect(() => () => {
    if (retryTimerRef.current !== null) {
      window.clearTimeout(retryTimerRef.current);
    }
  }, []);

  useEffect(() => {
    if (!autoReconnect || stallTimeoutMs <= 0) return;
    const onVisibility = () => {
      if (!document.hidden) lastFrameAtRef.current = Date.now();
    };
    document.addEventListener("visibilitychange", onVisibility);
    const interval = window.setInterval(() => {
      if (retryAtRef.current !== null) setTick(Date.now());
      if (document.hidden) return;
      // Liveness from the actual playback position. `timeupdate`/`progress`
      // are not fired reliably by every webview for live streams, so the
      // old event-based watchdog would tear down a perfectly healthy
      // stream after `stallTimeoutMs` ("dies after a few seconds").
      const v = videoRef.current;
      if (v && v.currentTime !== lastTimeRef.current) {
        lastTimeRef.current = v.currentTime;
        lastFrameAtRef.current = Date.now();
        attemptsRef.current = 0;
      }
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoReconnect, stallTimeoutMs]);

  // NOTE: no live-edge "catch-up" seek here. Seeking a live progressive
  // MP4 on iOS/Android (and some desktop webviews) tears the media
  // pipeline down — the exact "stream dies after a few seconds, reload
  // fixes it" failure. We accept the browser's buffering instead; the
  // server already emits 40 ms fragments so latency stays modest.

  return (
    <div
      className={`relative overflow-hidden bg-black ${className}`}
      style={{
        aspectRatio: frameSize ? `${frameSize.w} / ${frameSize.h}` : "16 / 9",
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
              No video stream available.
            </span>
          )}
        </div>
      ) : null}
      <video
        key={streamKey}
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-contain"
        onLoadedData={(e) => {
          lastFrameAtRef.current = Date.now();
          attemptsRef.current = 0;
          if (retryTimerRef.current !== null) clearPendingRetry();
          const el = e.currentTarget;
          if (el.videoWidth > 0 && el.videoHeight > 0) {
            setFrameSize((prev) =>
              prev && prev.w === el.videoWidth && prev.h === el.videoHeight
                ? prev
                : { w: el.videoWidth, h: el.videoHeight },
            );
          }
          report("live");
        }}
        onTimeUpdate={() => {
          lastFrameAtRef.current = Date.now();
        }}
        onProgress={() => {
          lastFrameAtRef.current = Date.now();
        }}
        onError={(e) => {
          const code = e.currentTarget.error?.code;
          if (
            typeof MediaError !== "undefined" &&
            code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
          ) {
            // Webview can't decode this codec at all — no H.265 possible.
            onUnsupported?.();
            return;
          }
          report("error");
          if (autoReconnect) scheduleRetry();
        }}
      />
      {children}
    </div>
  );
}
