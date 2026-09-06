import { useEffect, useState } from "react";

import { VideoFeed } from "../components/VideoFeed";
import { Banner, Button } from "../components/ui";
import { getOrCreateRuntimeTransport } from "../runtime";
import type { RuntimeConnectionState } from "../runtime";

interface VideoScreenProps {
  /** IP address of the robot (LAN address of the bebop-linux runtime). */
  robotIp: string;
  /** Optional override for the runtime port. Defaults to 9090. */
  runtimePort?: number;
  onBack: () => void;
  /** Label for the back button, e.g. "Back to motor bench". */
  backLabel?: string;
}

/** Live multi-stream viewer served by the bebop-vision process
 * (`:9092/video`, selectable color/depth near+far via the stream
 * picker on each tile).
 *
 * `multipart/x-mixed-replace` renders natively in an `<img>` tag, so the
 * stream needs no JavaScript decode loop — the browser paints each JPEG
 * part as it arrives. The streams come from the robot's Orbbec rig (the
 * same process that runs autonomy and recording, which owns the cameras);
 * tiles are independent MJPEG connections and can be toggled
 * concurrently. Stream state placeholders live in `VideoFeed` (shared
 * with the teleop screen).
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
  const [reconnectKey, bumpReconnectKey] = useState(0);
  // Retry affordance for the stream-error banner.
  const retry = () => {
    setStreamState("loading");
    bumpReconnectKey((k) => k + 1);
  };
  const [streamState, setStreamState] = useState<
    "loading" | "live" | "error"
  >("loading");
  const [conn, setConn] = useState<RuntimeConnectionState>("disconnected");
  // Concurrent tiles, same model as the teleop screen. All four views
  // open by default here — this screen exists to inspect the rig.
  const [videoStreams, setVideoStreams] = useState<string[]>([
    "color_near",
    "depth_near",
    "color_far",
    "depth_far",
  ]);
  const toggleStream = (id: string) =>
    setVideoStreams((cur) =>
      cur.includes(id)
        ? cur.length > 1
          ? cur.filter((s) => s !== id)
          : cur
        : [...cur, id],
    );

  const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
  const url = `http://${robotIp}:9092/video`;

  useEffect(() => {
    const offs = [transport.onConnectionStateChange(setConn)];
    void transport.connect(robotIp, runtimePort).catch(() => {
      /* conn state listener surfaces the failure */
    });
    return () => offs.forEach((off) => off());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [robotIp, runtimePort]);

  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-4xl">
      <div className="flex items-center justify-between w-full">
        <h2 className="text-text font-semibold text-base">
          Live video
          <span className="ml-2 text-text-dim font-normal text-sm">
            {robotIp}:9092
          </span>
        </h2>
        <Button variant="ghost" onClick={onBack} className="text-xs h-8">
          {backLabel}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 w-full">
        {videoStreams.map((id) => (
          <VideoFeed
            key={id}
            baseUrl={`http://${robotIp}:${runtimePort}`}
            videoUrl={url}
            stream={id}
            reconnectKey={reconnectKey}
            onStreamState={setStreamState}
            className="w-full -mx-4 sm:mx-0 sm:rounded-[var(--radius-card)] sm:border sm:border-border"
            maxHeight="72dvh"
          >
            <div className="absolute right-2 top-2 z-10 flex gap-1">
              {VIDEO_STREAM_OPTIONS.map((o) => (
                <button
                  key={o.id}
                  type="button"
                  onClick={() => toggleStream(o.id)}
                  className={`rounded px-2 py-0.5 text-[11px] font-medium backdrop-blur-sm transition-colors ${
                    videoStreams.includes(o.id)
                      ? "bg-white/85 text-black"
                      : "bg-black/50 text-white/80 hover:bg-black/70"
                  }`}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </VideoFeed>
        ))}
      </div>

      {streamState === "error" ? (
        <Banner tone="error">
          Camera stream failed on <code>{url}</code>. The bebop-vision
          process (autonomy/recording) serves the streams — start
          <code> main.py --record-navd ...</code> or
          <code> --goal-drive</code> on the robot.{" "}
          <button type="button" onClick={retry} className="underline">
            Retry
          </button>
        </Banner>
      ) : null}

      {conn !== "connected" ? (
        <span className="text-xs text-text-dim">
          Runtime WS {conn} — the streams are independent of it and keep
          playing.
        </span>
      ) : null}
    </div>
  );
}

const VIDEO_STREAM_OPTIONS: { id: string; label: string }[] = [
  { id: "color_near", label: "Color" },
  { id: "depth_near", label: "Depth" },
  { id: "color_far", label: "Far" },
  { id: "depth_far", label: "Far depth" },
];
