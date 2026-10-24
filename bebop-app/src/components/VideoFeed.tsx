// Live operator video tile, WebRTC-only.
//
// Every stream (color and depth) is served by the bebop-vision process as
// a WHEP endpoint (`endpoints(host).whep(stream)`, :9092). We create a
// recvonly peer connection, POST the offer, apply the answer, and play the
// incoming track. When the server returns a `Location` for the session, the
// tile DELETEs it on teardown so the encoder is released right away instead
// of after the server's idle timeout. Media
// rides SRTP/UDP, so latency is ~150 ms and packet loss doesn't
// head-of-line-block the way the old TCP MJPEG / fragmented-MP4 paths
// did. There is no MSE/progressive/MJPEG fallback any more — if the
// peer connection cannot be established the tile shows an error and
// retries with exponential backoff.
//
// `fill` makes the tile fill its parent (the Drive page's full-bleed view)
// instead of sizing itself to the stream's aspect; `onStats` reports
// receive-side WebRTC stats (fps, bitrate, loss, jitter, latency estimate)
// once a second from `RTCPeerConnection.getStats()`.

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Button, cn, Spinner } from "./ui";
import { sampleFromReport, summarizeVideoStats } from "./videoStats";
import type { VideoStats, VideoStatsSample } from "./videoStats";

export type { VideoStats } from "./videoStats";

export type VideoStreamState = "loading" | "live" | "error";

interface VideoFeedProps {
  /// WHEP offer URL for one stream (`endpoints(host).whep(stream)`); a
  /// cache-buster is appended per attempt.
  whepUrl: string;
  /// Resolve the session `Location` header to an absolute URL for the
  /// teardown DELETE (`endpoints(host).whepResource`). Without it, or
  /// when the server sends no Location, teardown just closes the peer.
  whepResource?: (location: string) => string;
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
  /// Fill the parent box (letterboxed per `fit`) instead of sizing the
  /// tile to the stream's aspect ratio.
  fill?: boolean;
  /// How the frame sits in a `fill` tile. Default "contain".
  fit?: "contain" | "cover";
  /// Smaller loading / error captions (thumbnails, grid cells).
  compact?: boolean;
  /// Receive-side stats once a second while connected (null on teardown).
  onStats?: (stats: VideoStats | null) => void;
  /// Extra line under a server-reported error (e.g. "Pick another stream").
  errorHint?: string;
  /// Optional extra chrome layered on top of the video (badges, pickers).
  children?: ReactNode;
}

/// A failure the server explained (shown in the tile instead of "Stream lost").
class WhepError extends Error {}

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
  whepUrl: offerUrl,
  whepResource,
  reconnectKey,
  autoReconnect = true,
  reconnectBaseDelayMs = 1000,
  reconnectMaxDelayMs = 15000,
  stallTimeoutMs = 6000,
  onStreamState,
  className = "",
  maxHeight,
  fill = false,
  fit = "contain",
  compact = false,
  onStats,
  errorHint,
  children,
}: VideoFeedProps) {
  const [state, setState] = useState<VideoStreamState>("loading");
  /// Why the last attempt failed, when the server said (e.g. viewer limit).
  const [errorText, setErrorText] = useState<string | null>(null);
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

  const streamKey = `${reconnectKey}.${autoRetry}`;
  const whepUrl = `${offerUrl}${offerUrl.includes("?") ? "&" : "?"}r=${streamKey}`;
  const resourceRef = useRef(whepResource);
  resourceRef.current = whepResource;
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const onStatsRef = useRef(onStats);
  onStatsRef.current = onStats;
  const wantStats = !!onStats;

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
    setErrorText(null);
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
    pcRef.current = pc;
    let disposed = false;
    // WHEP session resource (from the POST's Location), DELETEd on teardown.
    let sessionUrl: string | null = null;
    const fail = (why?: string) => {
      if (disposed) return;
      setErrorText(why ?? null);
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
        if (res.status === 503) {
          // Viewer limit (JSON body, Retry-After) or encoder unavailable.
          let detail = "";
          let count = "";
          try {
            const j = (await res.json()) as {
              error?: string;
              message?: string;
              viewers?: number;
              max_viewers?: number;
            };
            detail = j.error ?? j.message ?? "";
            if (typeof j.viewers === "number" && typeof j.max_viewers === "number") {
              count = ` (${j.viewers}/${j.max_viewers} viewers)`;
            }
          } catch {
            /* not JSON */
          }
          throw new WhepError(
            detail === "viewer limit"
              ? `Stream busy${count}`
              : `Stream unavailable${detail ? `: ${detail}` : ""}`,
          );
        }
        if (!res.ok) throw new Error(`WHEP HTTP ${res.status}`);
        const location = res.headers.get("Location");
        const resolve = resourceRef.current;
        if (location && resolve) sessionUrl = resolve(location);
        const answer = await res.text();
        if (disposed) {
          if (sessionUrl) void fetch(sessionUrl, { method: "DELETE" }).catch(() => {});
          return;
        }
        await pc.setRemoteDescription({ type: "answer", sdp: answer });
      } catch (e) {
        fail(e instanceof WhepError ? e.message : undefined);
      }
    })();

    return () => {
      disposed = true;
      if (pcRef.current === pc) pcRef.current = null;
      if (sessionUrl) {
        // keepalive lets the DELETE survive a page unload.
        void fetch(sessionUrl, { method: "DELETE", keepalive: true }).catch(() => {});
        sessionUrl = null;
      }
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

  // Receive stats, once a second, only while someone listens.
  useEffect(() => {
    if (!wantStats) return;
    let prev: VideoStatsSample | null = null;
    let prevPc: RTCPeerConnection | null = null;
    let cancelled = false;
    const poll = async () => {
      const pc = pcRef.current;
      if (!pc || pc.connectionState === "closed") {
        prev = null;
        onStatsRef.current?.(null);
        return;
      }
      if (pc !== prevPc) {
        prev = null;
        prevPc = pc;
      }
      try {
        const report = await pc.getStats();
        if (cancelled) return;
        const sample = sampleFromReport(
          report as unknown as Parameters<typeof sampleFromReport>[0],
          performance.now(),
        );
        if (!sample) return;
        onStatsRef.current?.(summarizeVideoStats(prev, sample));
        prev = sample;
      } catch {
        /* peer closed mid-call */
      }
    };
    const interval = window.setInterval(() => void poll(), 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
      onStatsRef.current?.(null);
    };
  }, [wantStats]);

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
      className={cn("relative overflow-hidden bg-black", className)}
      style={
        fill
          ? undefined
          : {
              aspectRatio: frameSize ? `${frameSize.w} / ${frameSize.h}` : "16 / 9",
              ...(maxHeight ? { maxHeight } : {}),
            }
      }
    >
      {state === "loading" ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <Spinner />
          <span className={cn("text-text-dim", compact ? "text-caption" : "text-sm")}>
            {compact ? "Connecting…" : "Waiting for the first frame…"}
          </span>
        </div>
      ) : null}
      {state === "error" ? (
        <div
          className={cn(
            "absolute inset-0 flex flex-col items-center justify-center text-center",
            compact ? "gap-1 px-2" : "gap-3 px-8",
          )}
        >
          {autoReconnect ? (
            <>
              <Spinner />
              <span className={cn("text-text-dim", compact ? "text-caption" : "text-sm")}>
                {errorText
                  ? compact
                    ? errorText
                    : `${errorText} — retrying`
                  : compact
                    ? "Reconnecting"
                    : "Stream lost — reconnecting"}
                {secondsLeft !== null ? ` in ${secondsLeft}s` : ""}…
              </span>
              {errorText && !compact && errorHint ? (
                <span className="max-w-xs text-caption text-text-dim/80">{errorHint}</span>
              ) : null}
              {compact ? null : (
                <Button variant="ghost" size="sm" onClick={retryNow}>
                  Retry now
                </Button>
              )}
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
        className={cn("h-full w-full", fit === "cover" ? "object-cover" : "object-contain")}
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
          setErrorText(null);
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
