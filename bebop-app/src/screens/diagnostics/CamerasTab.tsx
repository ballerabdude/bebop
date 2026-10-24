// Diagnostics › Cameras: inspect the camera rig, every stream at once.
// Each tile is its own WHEP session; streams are opt-in to keep the radio
// free for control traffic (depth tiles cost as much bandwidth as color).

import { useState } from "react";

import { VideoFeed } from "../../components/VideoFeed";
import { Banner, Button, cn } from "../../components/ui";
import { VIDEO_STREAMS } from "../../robot/endpoints";
import { useRobot } from "../../robot/session";

export function CamerasTab() {
  const { endpoints } = useRobot();
  const [reconnectKey, setReconnectKey] = useState(0);
  const [failed, setFailed] = useState<Record<string, boolean>>({});
  const [open, setOpen] = useState<string[]>(["color_near", "depth_near"]);
  const toggle = (id: string) =>
    setOpen((cur) =>
      cur.includes(id) ? (cur.length > 1 ? cur.filter((s) => s !== id) : cur) : [...cur, id],
    );
  const anyFailed = open.some((id) => failed[id]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        {VIDEO_STREAMS.map((o) => (
          <button
            key={o.id}
            type="button"
            aria-pressed={open.includes(o.id)}
            onClick={() => toggle(o.id)}
            className={cn(
              "h-control-sm rounded-full border px-3.5 text-sm font-semibold transition-colors pointer-coarse:h-touch",
              open.includes(o.id)
                ? "border-accent/50 bg-accent/15 text-accent"
                : "border-border text-text-dim hover:text-text",
            )}
          >
            {o.label}
          </button>
        ))}
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto"
          onClick={() => setReconnectKey((n) => n + 1)}
        >
          Reconnect all
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {open.map((id) => (
          <div key={id} className="relative">
            <VideoFeed
              whepUrl={endpoints.whep(id)}
              whepResource={endpoints.whepResource}
              reconnectKey={reconnectKey}
              onStreamState={(st) => setFailed((f) => ({ ...f, [id]: st === "error" }))}
              className="w-full rounded-card border border-border"
              maxHeight="72dvh"
            >
              <span className="pointer-events-none absolute left-2 top-2 z-10 rounded bg-black/60 px-1.5 py-0.5 text-caption font-semibold text-white/85">
                {VIDEO_STREAMS.find((o) => o.id === id)?.label ?? id}
              </span>
            </VideoFeed>
          </div>
        ))}
      </div>

      {anyFailed ? (
        <Banner tone="warning">
          A camera stream failed. The bebop-vision service serves every stream;
          start it from Diagnostics › Services or the Drive page.
        </Banner>
      ) : null}
    </div>
  );
}
