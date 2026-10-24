// Top-down canvas for one place's stored local map: free / obstacle grid
// layers, fitted walls and the anchor, with drag-to-pan and wheel/pinch zoom.
// Drawing is capped at MAX_PLAN_PRIMITIVES rectangles/segments/labels.
//
// Layer naming is deliberate: white lines are fitted walls (real bounding
// structure), grey cells are obstacles/furniture, green cells observed free
// space. Coordinates are metres in the place's own anchor-relative frame.

import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import {
  decodePlaceLayer,
  fitPlanView,
  gridRuns,
  hasPlaceMapContent,
  panPlanView,
  placeMapBounds,
  scaleBarMeters,
  screenHeading,
  worldToScreen,
  zoomPlanView,
} from "./mapView";
import type { GridMask, PlaceMap, PlanView } from "./mapView";

export const MAX_PLAN_PRIMITIVES = 30000;
export const CANVAS_BG = "#0b0d10";

// Canvas palette from the design tokens (App.css `--color-canvas-*`).
export function canvasToken(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--color-canvas-${name}`).trim();
  return v || fallback;
}

export function useCanvasSize(minW: number, minH: number, init: { w: number; h: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState(init);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setSize({ w: Math.max(minW, r.width), h: Math.max(minH, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [minW, minH]);
  return [ref, size] as const;
}

/// Size the backing store for the device pixel ratio and return a cleared,
/// scaled 2D context.
export function prepareCanvas(canvas: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.max(1, Math.round(w * dpr));
  const bh = Math.max(1, Math.round(h * dpr));
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = CANVAS_BG;
  ctx.fillRect(0, 0, w, h);
  return ctx;
}

function draw(ctx: CanvasRenderingContext2D, map: PlaceMap, plan: PlanView, alpha: number, layers: { free: GridMask | null; occupied: GridMask | null }) {
  const { width: w, height: h } = plan;
  const budgetLimit = MAX_PLAN_PRIMITIVES - 8; // the scale bar draws after
  let primitives = 0;
  ctx.save();
  ctx.globalAlpha = alpha;

  const drawLayer = (mask: GridMask | null, fill: string) => {
    if (!mask || !(mask.resolution_m > 0)) return;
    const budget = budgetLimit - primitives;
    if (budget <= 0) return;
    ctx.fillStyle = fill;
    for (const run of gridRuns(mask, budget)) {
      const x0 = mask.origin[0] + run.x0 * mask.resolution_m;
      const x1 = mask.origin[0] + run.x1 * mask.resolution_m;
      const y0 = mask.origin[1] + run.iy * mask.resolution_m;
      const y1 = y0 + mask.resolution_m;
      const [sx, syTop] = worldToScreen(x0, y1, plan);
      const [sx1, syBottom] = worldToScreen(x1, y0, plan);
      const wpx = sx1 - sx;
      const hpx = syBottom - syTop;
      if (!(wpx > 0) || !(hpx > 0)) continue;
      ctx.fillRect(sx, syTop, wpx, hpx);
      primitives += 1;
    }
  };
  drawLayer(layers.free, canvasToken("free", "rgba(53,194,143,0.16)"));
  drawLayer(layers.occupied, canvasToken("obstacle", "rgba(216,222,233,0.30)"));

  if (map.walls.length > 0 && primitives < budgetLimit) {
    ctx.strokeStyle = canvasToken("wall", "#d8dee9");
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const wall of map.walls) {
      if (wall.length < 4) continue;
      const [ax, ay] = worldToScreen(wall[0], wall[1], plan);
      const [bx, by] = worldToScreen(wall[2], wall[3], plan);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      primitives += 1;
      if (primitives >= budgetLimit) break;
    }
    ctx.stroke();
  }

  const anchor = map.anchor;
  if (anchor && primitives + 3 <= budgetLimit) {
    const [sx, sy] = worldToScreen(anchor.x, anchor.y, plan);
    const [hx, hy] = screenHeading(anchor.yaw_deg);
    const px = -hy; // screen-space left normal (screen Y points down)
    const py = hx;
    ctx.fillStyle = canvasToken("anchor", "#ffd166");
    ctx.beginPath();
    ctx.moveTo(sx + hx * 15, sy + hy * 15);
    ctx.lineTo(sx - hx * 6 + px * 6, sy - hy * 6 + py * 6);
    ctx.lineTo(sx - hx * 6 - px * 6, sy - hy * 6 - py * 6);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText("anchor", sx + 10, sy - 8);
  }
  ctx.restore();

  const barMeters = scaleBarMeters(plan.scale, Math.max(40, w / 3));
  const barPx = barMeters * plan.scale;
  const barY = h - 20;
  ctx.strokeStyle = "rgba(242,244,247,0.75)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(16, barY);
  ctx.lineTo(16 + barPx, barY);
  ctx.moveTo(16, barY - 4);
  ctx.lineTo(16, barY + 4);
  ctx.moveTo(16 + barPx, barY - 4);
  ctx.lineTo(16 + barPx, barY + 4);
  ctx.stroke();
  ctx.fillStyle = "rgba(242,244,247,0.75)";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillText(`${barMeters} m · place frame`, 16, barY - 8);
}

export function LocalMapCanvas({
  map,
  placeId,
  dimmed,
  fitSignal,
  message,
  frozen,
  legend,
}: {
  map: PlaceMap | null;
  placeId: string | null;
  /// Stale record: draw at reduced opacity.
  dimmed: boolean;
  /// Bump to re-fit the view (the Fit button).
  fitSignal: number;
  message: string | null;
  frozen: boolean;
  legend: string;
}) {
  const [wrapRef, size] = useCanvasSize(280, 280, { w: 800, h: 440 });
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [view, setView] = useState<PlanView>({ centerX: 0, centerY: 0, scale: 40, width: 800, height: 440 });
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const drag = useRef<{ x: number; y: number } | null>(null);
  const pinch = useRef<number | null>(null);
  const userMoved = useRef(false);
  const fitPlace = useRef<string | null>(null);

  const layers = useMemo(
    () => ({ free: decodePlaceLayer(map, "free"), occupied: decodePlaceLayer(map, "occupied") }),
    [map],
  );
  const hasContent = hasPlaceMapContent(map);
  const plan = useMemo<PlanView>(() => ({ ...view, width: size.w, height: size.h }), [view, size]);

  // Auto-fit on a new place and keep following the record until the
  // operator pans/zooms away.
  useEffect(() => {
    if (!map || !hasContent) return;
    const newPlace = fitPlace.current !== placeId;
    if (!newPlace && userMoved.current) return;
    if (newPlace) {
      fitPlace.current = placeId;
      userMoved.current = false;
    }
    setView(fitPlanView(placeMapBounds(map), size.w, size.h));
  }, [map, hasContent, placeId, size]);

  useEffect(() => {
    if (fitSignal === 0 || !map) return;
    userMoved.current = false;
    setView(fitPlanView(placeMapBounds(map), size.w, size.h));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitSignal]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      userMoved.current = true;
      setView((v) => zoomPlanView(v, Math.exp(-e.deltaY * 0.0015), e.clientX - rect.left, e.clientY - rect.top));
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = prepareCanvas(canvas, size.w, size.h);
    if (!ctx || !map || !hasContent) return;
    draw(ctx, map, plan, dimmed ? 0.35 : 1, layers);
  }, [size, map, hasContent, layers, plan, dimmed]);

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 1) drag.current = { x: e.clientX, y: e.clientY };
    else {
      drag.current = null;
      const pts = [...pointers.current.values()];
      pinch.current = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    }
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const pts = [...pointers.current.values()];
    if (pts.length >= 2) {
      const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      if (pinch.current && d > 0) {
        const rect = e.currentTarget.getBoundingClientRect();
        const ratio = d / pinch.current;
        userMoved.current = true;
        setView((v) =>
          zoomPlanView(v, ratio, (pts[0].x + pts[1].x) / 2 - rect.left, (pts[0].y + pts[1].y) / 2 - rect.top),
        );
      }
      pinch.current = d;
      return;
    }
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    if (dx === 0 && dy === 0) return;
    userMoved.current = true;
    setView((v) => panPlanView(v, dx, dy));
  };
  const endPointer = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    pointers.current.delete(e.pointerId);
    const pts = [...pointers.current.values()];
    if (pts.length === 0) {
      drag.current = null;
      pinch.current = null;
    } else {
      pinch.current = pts.length >= 2 ? Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) : null;
    }
  };

  const [head, ...rest] = (message ?? "").split(".");
  return (
    <div
      ref={wrapRef}
      className="relative h-[min(60dvh,440px)] overflow-hidden rounded-card border border-border sm:h-[520px]"
      style={{ background: CANVAS_BG }}
    >
      <canvas
        ref={canvasRef}
        className="h-full w-full cursor-grab touch-none active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endPointer}
        onPointerCancel={endPointer}
        aria-label="Local top-down map for the selected place"
      />
      {message ? (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-1 px-6 text-center text-sm text-text-dim">
          <span className="font-semibold text-text">{head}</span>
          <span>{rest.join(".").trim()}</span>
        </div>
      ) : null}
      {frozen ? (
        <div className="absolute right-2 top-2 rounded-input border border-border bg-bg/85 px-2 py-1 text-caption text-text-dim">
          frozen
        </div>
      ) : null}
      <div className="absolute left-2 top-2 max-w-[75%] rounded-input border border-border bg-bg/85 px-2 py-1 text-caption text-text-dim">
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5">
          <span className="inline-flex items-center gap-1">
            <i className="inline-block h-2.5 w-2.5 rounded-sm bg-success/30" /> free
          </span>
          <span className="inline-flex items-center gap-1">
            <i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#d8dee9]/30" /> obstacle
          </span>
          <span className="inline-flex items-center gap-1">
            <i className="inline-block h-0.5 w-3 bg-[#d8dee9]" /> wall
          </span>
          <span className="inline-flex items-center gap-1">
            <i className="inline-block h-2 w-2 rounded-full bg-[#ffd166]" /> anchor
          </span>
        </div>
        <div className="hidden sm:block">{legend}</div>
      </div>
    </div>
  );
}
