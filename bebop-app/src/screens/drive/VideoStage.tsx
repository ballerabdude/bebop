// The Drive page's video surface: one main stream, an optional
// picture-in-picture stream, or a 2x2 grid of every stream.
//
// Every tile is a `VideoFeed` keyed by its stream id inside ONE parent, and
// its role (main / pip / grid cell) is only a class. Swapping main and PiP,
// or entering and leaving the grid, therefore moves elements around without
// remounting them: no WHEP renegotiation, no black flash.

import { memo } from "react";

import { VideoFeed } from "../../components/VideoFeed";
import type { VideoStats } from "../../components/VideoFeed";
import { Button, cn } from "../../components/ui";
import { VIDEO_STREAMS } from "../../robot/endpoints";
import type { RobotEndpoints } from "../../robot/endpoints";
import { CloseIcon, SwapIcon } from "./icons";

export function streamLabel(id: string): string {
  return VIDEO_STREAMS.find((s) => s.id === id)?.label ?? id;
}

/// Memoized: the Drive page re-renders at telemetry rate, the tiles need not.
/// Pass stable callbacks.
export const VideoStage = memo(function VideoStage({
  endpoints,
  main,
  pip,
  grid,
  videoEnabled,
  reconnectKey,
  onSelectMain,
  onSwap,
  onClosePip,
  onEnableVideo,
  onStats,
  pipClassName,
  className,
}: {
  endpoints: RobotEndpoints;
  main: string;
  pip: string | null;
  grid: boolean;
  videoEnabled: boolean;
  reconnectKey: number;
  onSelectMain: (id: string) => void;
  onSwap: () => void;
  onClosePip: () => void;
  onEnableVideo: () => void;
  onStats: (s: VideoStats | null) => void;
  /// Placement of the PiP tile (it differs between the stacked and the
  /// overlay layouts).
  pipClassName: string;
  className?: string;
}) {
  const mounted = grid
    ? VIDEO_STREAMS.map((s) => s.id)
    : VIDEO_STREAMS.map((s) => s.id).filter((id) => id === main || id === pip);

  if (!videoEnabled) {
    return (
      <div className={cn("relative flex items-center justify-center bg-bg", className)}>
        <div className="flex max-w-xs flex-col items-center gap-3 px-6 text-center">
          <span className="text-body-sm text-text-dim">
            Video playback is off. The robot still records and the drive controls stay live.
          </span>
          <Button variant="secondary" size="sm" onClick={onEnableVideo}>
            Turn on video
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "relative overflow-hidden bg-black",
        grid && "grid grid-cols-2 grid-rows-2 gap-0.5",
        className,
      )}
    >
      {mounted.map((id) => {
        const role = grid ? "cell" : id === main ? "main" : "pip";
        return (
          <div
            key={id}
            className={cn(
              role === "main" && "absolute inset-0",
              role === "cell" && "relative min-h-0 min-w-0",
              role === "pip" &&
                cn(
                  "absolute z-[5] aspect-video overflow-hidden rounded-card border border-white/25 shadow-xl",
                  pipClassName,
                ),
              role === "cell" && id === main && "ring-2 ring-inset ring-accent",
            )}
          >
            <VideoFeed
              whepUrl={endpoints.whep(id)}
              whepResource={endpoints.whepResource}
              reconnectKey={reconnectKey}
              fill
              compact={role !== "main"}
              onStats={id === main ? onStats : undefined}
              errorHint={role === "main" ? "Pick another stream above, or wait for a slot." : undefined}
              className="h-full w-full"
            >
              {role === "pip" ? (
                <>
                  <button
                    type="button"
                    onClick={onSwap}
                    className="absolute inset-0 z-10 cursor-pointer"
                    aria-label={`Swap ${streamLabel(id)} into the main view`}
                  />
                  <span className="pointer-events-none absolute bottom-1 left-1 z-10 inline-flex items-center gap-1 rounded bg-black/65 px-1.5 py-0.5 text-micro font-semibold text-white/90">
                    <SwapIcon />
                    {streamLabel(id)}
                  </span>
                  <button
                    type="button"
                    onClick={onClosePip}
                    className="absolute right-0 top-0 z-20 flex h-touch w-touch items-start justify-end p-1.5"
                    aria-label="Close picture-in-picture"
                  >
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-black/70 text-white/90">
                      <CloseIcon size={14} />
                    </span>
                  </button>
                </>
              ) : role === "cell" ? (
                <>
                  <button
                    type="button"
                    onClick={() => onSelectMain(id)}
                    className="absolute inset-0 z-10 cursor-pointer"
                    aria-label={`Show ${streamLabel(id)} full size`}
                  />
                  <span
                    className={cn(
                      "pointer-events-none absolute top-2 left-2 z-10 rounded px-2 py-0.5 text-caption font-semibold",
                      id === main ? "bg-accent text-white" : "bg-black/65 text-white/90",
                    )}
                  >
                    {streamLabel(id)}
                  </span>
                </>
              ) : null}
            </VideoFeed>
          </div>
        );
      })}
    </div>
  );
});
