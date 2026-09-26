// Live operator video tile, WebRTC-only.
//
// Every stream (color and depth) is served by the bebop-vision process as
// a WHEP endpoint on `:9092/whep`. We create a recvonly peer connection,
// POST the offer, apply the answer, and play the incoming track. Media
// rides SRTP/UDP, so latency is ~150 ms and packet loss doesn't
// head-of-line-block the way the old TCP MJPEG / fragmented-MP4 paths
// did. There is no MSE/progressive/MJPEG fallback any more — if the
// peer connection cannot be established the tile shows an error and
// retries with exponential backoff.

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Button, Spinner } from "./ui";

export type VideoStreamState = "loading" | "live" | "error";

interface VideoFeedProps {
  /// Runtime base URL (`http://<ip>:<port>`); used only when `videoUrl`
  /// is omitted, defaulting to `<baseUrl>/whep`.
  baseUrl: string;
  /// WebRTC (WHEP) endpoint base, e.g. `http://<ip>:9092/whep`. The
  /// stream selector and cache-buster are appended as query params.
  videoUrl?: string;
  /// Stream selector understood by the server: color_near | color_far |
  /// depth_near | depth_far.
  stream?: string;
  /// Bump to tear the peer connection down and reconnect.
  reconnectKey: number;
  /// Self-healing reconnects. When negotiation fails or the connection
  /// stalls (no `currentTime` advance for `stallTimeoutMs`), the stream
  /// is re-requested automatically with exponential backoff. Default true.
  autoReconnect?: boolean;
  /// First reconnect delay (ms); doubles per consecutive failure.
  reconnectBaseDelayMs?: number;
  /// Backoff ceiling (ms). Default 15000.
  reconnectMaxDelayMs?: number;
  /// Silence threshold (ms): no playback advance for this long counts as
  /// a stalled stream and triggers a reconnect. Default 6000. 0 disables
  /// the watchdog (negotiation-failure reconnects still apply).
  stallTimeoutMs?: number;
  /// Stream lifecycle reports for the parent (retry buttons, etc.).
  onStreamState?: (state: VideoStreamState) => void;
  /// Classes for the outer container — sizing (width / h-full) and
  /// decoration. The element is the black letterbox surface; its aspect
  /// is owned by the feed, so don't pass `aspect-*` utilities.
  className?: string;
  /// Cap the container height (CSS length, e.g. "72dvh") so a wide stream
  /// on a landscape phone / short window doesn't overflow the viewport.
  maxHeight?: string;
  /// Optional extra chrome layered on top of the video (badges, pickers).
  children?: ReactNode;
}

/// Non-trickle WHEP: gather ICE candidates before POSTing the offer.
function waitForIceGathering(
  pc: RTCPeerConnection,
  timeoutMs = 3000,
): Promise<void> {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve();
      return;
    }
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      pc.removeEventListener("icegatheringstatechange", check);
      resolve();
    };
    const check = () => {
      if (pc.iceGatheringState === "complete") done();
    };
    pc.addEventListener("icegatheringstatechange", check);
    window.setTimeout(done, timeoutMs);
  });
}

export function VideoFeed({
  baseUrl,
  videoUrl,
  stream,
  reconnectKey,
  autoReconnect = true,
  reconnectBaseDelayMs = 1000,
  reconnectMaxDelayMs = 15000,
  stallTimeoutMs = 6000,
  onStreamState,
  className = "",
  maxHeight,
  children,
}: VideoFeedProps) {
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
  // `timeupdate` events is unreliable in some webviews for live streams;
  // sampling `currentTime` works everywhere.
  const lastTimeRef = useRef(0);

  const whepBase = videoUrl ?? `${baseUrl}/whep`;
  const streamKey = `${reconnectKey}.${autoRetry}`;
  const whepUrl =
    `${whepBase}?stream=${encodeURIComponent(stream ?? "color_near")}` +
    `&r=${streamKey}`;

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

  // WebRTC peer connection.
  useEffect(() => {
    lastFrameAtRef.current = Date.now();
    const video = videoRef.current;
    if (!video) return;
    if (typeof RTCPeerConnection === "undefined") {
      report("error");
      return;
    }

    const pc = new RTCPeerConnection({ iceServers: [] });
    let disposed = false;
    const fail = () => {
      if (disposed) return;
      report("error");
      if (autoReconnect) scheduleRetry();
    };

    pc.addTransceiver("video", { direction: "recvonly" });
    pc.ontrack = (ev) => {
      video.srcObject = ev.streams[0] ?? new MediaStream([ev.track]);
      void video.play().catch(() => {
        /* muted autoplay; rejection is not fatal */
      });
    };
    pc.onconnectionstatechange = () => {
      if (disposed) return;
      if (pc.connectionState === "failed") fail();
    };

    void (async () => {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGathering(pc);
        const res = await fetch(whepUrl, {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: pc.localDescription?.sdp ?? offer.sdp ?? "",
        });
        if (!res.ok) throw new Error(`WHEP HTTP ${res.status}`);
        const answer = await res.text();
        if (disposed) return;
        await pc.setRemoteDescription({ type: "answer", sdp: answer });
      } catch {
        fail();
      }
    })();

    return () => {
      disposed = true;
      pc.ontrack = null;
      pc.onconnectionstatechange = null;
      try {
        pc.close();
      } catch {
        /* already closed */
      }
      try {
        video.srcObject = null;
      } catch {
        /* ignore */
      }
    };
    // report/scheduleRetry close over refs and stable setters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [whepUrl]);

  useEffect(() => () => {
    if (retryTimerRef.current !== null) {
      window.clearTimeout(retryTimerRef.current);
    }
  }, []);

  // Stall watchdog: once per second, reconnect if playback hasn't advanced
  // for `stallTimeoutMs`. Catches a peer connection that dies without a
  // clean state transition. Skipped while the tab is hidden.
  useEffect(() => {
    if (!autoReconnect || stallTimeoutMs <= 0) return;
    const onVisibility = () => {
      if (!document.hidden) lastFrameAtRef.current = Date.now();
    };
    document.addEventListener("visibilitychange", onVisibility);
    const interval = window.setInterval(() => {
      if (retryAtRef.current !== null) setTick(Date.now());
      if (document.hidden) return;
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
          void el.play().catch(() => {
            /* muted autoplay; rejection is not fatal */
          });
          report("live");
        }}
        onTimeUpdate={() => {
          lastFrameAtRef.current = Date.now();
        }}
      />
      {children}
    </div>
  );
}
