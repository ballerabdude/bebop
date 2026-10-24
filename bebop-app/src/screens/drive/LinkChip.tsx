// Link health at a glance: runtime telemetry (WS) rate / age plus the main
// video stream's WebRTC stats. Tap for the details.

import { useEffect, useRef, useState } from "react";

import type { VideoStats } from "../../components/VideoFeed";
import { cn, TONE_DOT } from "../../components/ui";
import type { Tone } from "../../components/ui";
import { useConnection, useTelemetry } from "../../robot/session";

export interface TelemetryHealth {
  state: string;
  /// ms since the last frame arrived (null before the first).
  ageMs: number | null;
  /// Frames per second over the last ~2 s.
  hz: number | null;
}

/// Telemetry arrival rate + age, measured on this client.
export function useTelemetryHealth(): TelemetryHealth {
  const link = useConnection();
  const stamp = useTelemetry((s) => s?.hostUnixMs ?? 0, 30);
  const arrivals = useRef<number[]>([]);
  const [now, setNow] = useState(() => performance.now());

  useEffect(() => {
    if (!stamp) return;
    const t = performance.now();
    const a = arrivals.current;
    a.push(t);
    while (a.length > 0 && t - a[0] > 2000) a.shift();
  }, [stamp]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  const a = arrivals.current;
  const last = a.length ? a[a.length - 1] : null;
  const recent = a.filter((t) => now - t <= 2000);
  const span = recent.length > 1 ? recent[recent.length - 1] - recent[0] : 0;
  return {
    state: link.state,
    ageMs: last === null ? null : Math.max(0, now - last),
    hz: span > 0 ? ((recent.length - 1) * 1000) / span : null,
  };
}

function fmtMs(ms: number | null): string {
  if (ms === null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

export function linkTone(h: TelemetryHealth, video: VideoStats | null, videoOn: boolean): Tone {
  if (h.state === "disconnected") return "err";
  if (h.state !== "connected" || (h.ageMs ?? 0) > 1000) return "warn";
  if (videoOn) {
    if (!video || (video.fps ?? 0) < 5) return "warn";
    if ((video.loss ?? 0) > 0.05 || (video.latencyMs ?? 0) > 400) return "warn";
  }
  return "ok";
}

function linkTexts(h: TelemetryHealth, video: VideoStats | null, videoOn: boolean) {
  const wsText =
    h.state === "connected"
      ? (h.ageMs ?? 0) > 1000
        ? `stale ${fmtMs(h.ageMs)}`
        : `${h.hz ? Math.round(h.hz) : "—"} Hz`
      : h.state === "connecting"
        ? "connecting"
        : h.state === "degraded"
          ? "degraded"
          : "offline";
  const videoText = !videoOn
    ? "video off"
    : video && video.fps !== null
      ? `${Math.round(video.fps)} fps${video.latencyMs !== null ? ` · ~${Math.round(video.latencyMs)} ms` : ""}`
      : "no video";
  return { wsText, videoText };
}

export function LinkChip({
  health,
  video,
  videoOn,
  open,
  onToggle,
  className,
  compact = false,
}: {
  health: TelemetryHealth;
  video: VideoStats | null;
  videoOn: boolean;
  open: boolean;
  onToggle: () => void;
  className?: string;
  /// Runtime rate only (the video figures stay in the details).
  compact?: boolean;
}) {
  const tone = linkTone(health, video, videoOn);
  const { wsText, videoText } = linkTexts(health, video, videoOn);
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      aria-label={`Link: runtime ${wsText}, ${videoText}. Show details`}
      className={cn(
        "flex h-touch shrink-0 items-center gap-2 rounded-full border border-white/15 bg-black/55 px-3 text-caption font-semibold text-white/90 backdrop-blur-md hover:bg-black/70",
        className,
      )}
    >
      <span className={cn("h-2 w-2 shrink-0 rounded-full", TONE_DOT[tone], tone !== "ok" && "animate-pulse")} aria-hidden />
      <span className="whitespace-nowrap tabular-nums">{wsText}</span>
      <span className={cn("hidden whitespace-nowrap tabular-nums text-white/70", !compact && "sm:inline")}>· {videoText}</span>
    </button>
  );
}

export function LinkDetails({
  health: h,
  video,
  videoOn,
}: {
  health: TelemetryHealth;
  video: VideoStats | null;
  videoOn: boolean;
}) {
  const rows: [string, string][] = [
    ["Runtime link", h.state],
    ["Telemetry", `${h.hz ? h.hz.toFixed(0) : "—"} Hz · last ${fmtMs(h.ageMs)} ago`],
  ];
  if (videoOn) {
    rows.push(
      ["Video", video?.fps != null ? `${video.fps.toFixed(0)} fps · ${video.width}×${video.height}` : "—"],
      ["Bitrate", video?.kbps != null ? `${(video.kbps / 1000).toFixed(2)} Mbit/s` : "—"],
      ["Packet loss", video?.loss != null ? `${(video.loss * 100).toFixed(1)} %` : "—"],
      ["Jitter", video?.jitterMs != null ? `${video.jitterMs.toFixed(1)} ms` : "—"],
      ["Round trip", video?.rttMs != null ? `${Math.round(video.rttMs)} ms` : "—"],
      ["Jitter buffer", video?.jitterBufferMs != null ? `${Math.round(video.jitterBufferMs)} ms` : "—"],
      ["Latency (est.)", video?.latencyMs != null ? `~${Math.round(video.latencyMs)} ms` : "—"],
    );
  }
  return (
    <>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-body-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-text-dim">{k}</dt>
            <dd className="text-right font-mono tabular-nums text-text">{v}</dd>
          </div>
        ))}
      </dl>
      {videoOn ? (
        <p className="mt-2.5 text-caption leading-snug text-text-dim">
          Latency is this side only (half the round trip, jitter buffer and decode); camera
          capture and encode on the robot add to it.
        </p>
      ) : null}
    </>
  );
}
