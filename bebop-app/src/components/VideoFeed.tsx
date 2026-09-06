// Live MJPEG camera view shared by the video screen and the teleop
// screen. Renders the firmware's `GET /video` multipart stream in an
// `<img>` (the browser paints each JPEG part as it arrives — no
// JavaScript decode loop) with:
//
//   * loading / error placeholder states (503 = no `video:` config),
//   * the optional navigable-path mask overlay (subscribe-gated,
//     painted by a rAF loop at the mask rate, fitted to the video's
//     *displayed* rectangle so PTZ moves / letterboxing can't skew it),
//   * a parent-driven reconnect: bump `reconnectKey` to tear the
//     multipart stream down and re-request it.
//
// The firmware owns the camera exclusively (see
// `firmware/bebop-linux/src/video.rs`); this component is just another
// subscriber alongside bebop-vision.

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Spinner } from "./ui";
import type { RuntimeTransport } from "../runtime";
import type { NavMaskView, NavView } from "../runtime";

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
///
/// The video rect is computed from the img's natural size — the only
/// source of truth for what the camera is actually serving. If the img
/// hasn't decoded a frame yet, fall back to the mask's own grid (it
/// covers the full camera frame, so its aspect is the frame's aspect);
/// if neither is known, skip painting entirely rather than guessing a
/// 16:9 box and smearing the overlay across the letterbox bars.
function drawNavOverlay(
  canvas: HTMLCanvasElement | null,
  img: HTMLImageElement | null,
  mask: NavMaskView,
): void {
  if (!canvas) return;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  if (cssW === 0 || cssH === 0) return;
  const natW = img?.naturalWidth || mask.width;
  const natH = img?.naturalHeight || mask.height;
  if (!natW || !natH) return;
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
  /// Runtime WS transport — used to subscribe/unsubscribe the nav-mask
  /// stream while `showNav` is on. Shares the endpoint cache with the
  /// rest of the screen.
  transport: RuntimeTransport;
  /// Paint the navigable-path overlay (subscribes to ~10 Hz masks).
  showNav: boolean;
  /// Nav runner summary from the screen's telemetry subscription —
  /// drives the "nav unavailable / waiting for masks" hints. `null`
  /// until the first telemetry frame.
  nav: NavView | null;
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
  transport,
  showNav,
  nav,
  reconnectKey,
  onStreamState,
  className = "",
  maxHeight,
  children,
}: VideoFeedProps) {
  // "loading" until the first frame paints, "live" while streaming,
  // "error" when the endpoint is unreachable or answers 503.
  const [state, setState] = useState<VideoStreamState>("loading");
  const [navMaskFresh, setNavMaskFresh] = useState(false);
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

  const url = videoUrl ?? `${baseUrl}/video`;
  // Cache-busted variant actually assigned to the <img>: bumping the
  // key remounts the element, tearing the multipart stream down and
  // reconnecting — the retry path.
  const streamSrc = reconnectKey ? `${url}?r=${reconnectKey}` : url;

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
      {children}
    </div>
  );
}
