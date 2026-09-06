// Live MJPEG stream tile shared by the video screen and the teleop
// screen. Renders a bebop-vision `:9092/video` multipart stream in an
// `<img>` (the browser paints each JPEG part as it arrives — no
// JavaScript decode loop) with:
//
//   * loading / error placeholder states,
//   * a parent-driven reconnect: bump `reconnectKey` to tear the
//     multipart stream down and re-request it.
//
// The stream is served by the bebop-vision process, which owns the
// Orbbec cameras exclusively; this component is just another HTTP
// subscriber. (The legacy nav-mask overlay was removed with the
// OBSBOT pipeline, plan §9 Stage 3 — the BEV lives in Foxglove.)

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Spinner } from "./ui";
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
  /// color_far | depth_near | depth_far. Appended as ?stream= to the URL.
  stream?: string;
  /// Bump to tear the multipart stream down and reconnect.
  reconnectKey: number;
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

  const base = videoUrl ?? `${baseUrl}/video`;
  const url = stream ? `${base}?stream=${stream}` : base;
  // Cache-busted variant actually assigned to the <img>: bumping the
  // key remounts the element, tearing the multipart stream down and
  // reconnecting — the retry path.
  const streamSrc = reconnectKey
    ? `${url}&r=${reconnectKey}`
    : url;

  const report = (s: VideoStreamState) => {
    setState(s);
    onStreamState?.(s);
  };

  // A reconnect request resets to "loading" (the remounted <img> fires
  // onLoad / onError to move on from there) and drops the measured
  // frame size — the new stream may negotiate a different mode.
  useEffect(() => {
    setState("loading");
    setFrameSize(null);
    onStreamState?.("loading");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reconnectKey]);

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
        key={reconnectKey}
        ref={imgRef}
        src={streamSrc}
        alt="Robot live camera feed"
        className="w-full h-full object-contain"
        onLoad={(e) => {
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
        onError={() => report("error")}
      />
      {children}
    </div>
  );
}
