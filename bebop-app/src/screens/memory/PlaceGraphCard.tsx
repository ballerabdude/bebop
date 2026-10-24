// The place graph: places as nodes, recorded traversals as directed edges.
// Schematic layout, not geographic. Drawing is capped at MAX_PRIMITIVES
// segments/arcs/labels.

import { useEffect, useMemo, useRef, useState } from "react";

import { Banner, Card, CardHeader } from "../../components/ui";
import { graphEdgeLabel, layoutPlaceGraph } from "./mapView";
import type { PlaceGraphState } from "./mapPolling";
import { plural } from "./places";

const MAX_PRIMITIVES = 30000;
const CANVAS_BG = "#0b0d10";

// Canvas palette from the design tokens (App.css `--color-canvas-*`).
function canvasToken(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--color-canvas-${name}`).trim();
  return v || fallback;
}

function useCanvasSize(minW: number, minH: number, init: { w: number; h: number }) {
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
function prepareCanvas(canvas: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D | null {
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

export function PlaceGraphCard({ state }: { state: PlaceGraphState }) {
  const { graph, error } = state;
  const [wrapRef, size] = useCanvasSize(260, 240, { w: 800, h: 360 });
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const layout = useMemo(() => (graph ? layoutPlaceGraph(graph, size.w, size.h) : null), [graph, size]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = prepareCanvas(canvas, size.w, size.h);
    if (!ctx || !layout) return;
    // The layout caps nodes/edges (256/2048), well under MAX_PRIMITIVES.
    let primitives = 0;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = canvasToken("edge", "rgba(79,140,255,0.72)");
    for (const edge of layout.edges) {
      if (primitives >= MAX_PRIMITIVES) break;
      ctx.beginPath();
      ctx.moveTo(edge.x0, edge.y0);
      ctx.lineTo(edge.x1, edge.y1);
      ctx.stroke();
      const a = Math.atan2(edge.y1 - edge.y0, edge.x1 - edge.x0);
      const tx = edge.x1 - Math.cos(a) * 16;
      const ty = edge.y1 - Math.sin(a) * 16;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - Math.cos(a - 0.45) * 7, ty - Math.sin(a - 0.45) * 7);
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - Math.cos(a + 0.45) * 7, ty - Math.sin(a + 0.45) * 7);
      ctx.stroke();
      primitives += 2;
    }
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "rgba(216,222,233,0.72)";
    for (const edge of layout.edges) {
      if (primitives >= MAX_PRIMITIVES) break;
      ctx.fillText(graphEdgeLabel(edge.edge), edge.labelX + 4, edge.labelY);
      primitives += 1;
    }
    const node = canvasToken("node", "#4f8cff");
    for (const n of layout.nodes) {
      if (primitives >= MAX_PRIMITIVES) break;
      const r = 7 + Math.min(6, n.degree);
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = node;
      ctx.fill();
      ctx.strokeStyle = "rgba(242,244,247,0.7)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(242,244,247,0.92)";
      ctx.textAlign = "center";
      ctx.fillText(n.name, n.x, n.y + r + 14);
      ctx.textAlign = "start";
      primitives += 2;
    }
  }, [layout, size]);

  const message =
    error && !graph
      ? `Can't read the place graph. ${error}`
      : !graph
        ? "Loading the place graph…"
        : graph.nodes.length === 0
          ? "No places in the graph yet."
          : graph.edges.length === 0
            ? "No recorded routes yet. Drive between places to record traversals."
            : null;
  const nameOf = (id: string) => graph?.nodes.find((n) => n.id === id)?.name ?? id;

  return (
    <Card className="flex flex-col gap-3">
      <CardHeader
        eyebrow="Place graph"
        title={graph ? `${plural(graph.nodes.length, "place")} · ${plural(graph.edges.length, "recorded route")}` : "Routes between places"}
        description="Edges are recorded routes, not measured distances; labels show traversals and successes. The layout is schematic."
      />
      {error && graph ? <Banner tone="info">Graph refresh failed; showing the last graph. {error}</Banner> : null}
      <div
        ref={wrapRef}
        className="relative h-[300px] overflow-hidden rounded-card border border-border sm:h-[360px]"
        style={{ background: CANVAS_BG }}
      >
        <canvas ref={canvasRef} className="h-full w-full" aria-label="Place graph of recorded routes" />
        {message ? (
          graph && graph.nodes.length > 0 ? (
            <div className="pointer-events-none absolute inset-x-2 top-2 rounded-input border border-border bg-bg/85 px-2 py-1 text-center text-caption text-text-dim">
              {message}
            </div>
          ) : (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-text-dim">
              {message}
            </div>
          )
        ) : null}
        {layout && (layout.hiddenNodes > 0 || layout.hiddenEdges > 0) ? (
          <div className="absolute bottom-2 right-2 rounded-input border border-border bg-bg/85 px-2 py-1 text-caption text-text-dim">
            showing {layout.nodes.length}/{layout.nodes.length + layout.hiddenNodes} places ·{" "}
            {layout.edges.length}/{layout.edges.length + layout.hiddenEdges} routes
          </div>
        ) : null}
      </div>
      {graph && graph.edges.length > 0 ? (
        <ul className="m-0 flex list-none flex-col gap-1 p-0 text-label text-text-dim">
          {graph.edges.slice(0, 12).map((e) => (
            <li key={`${e.from}->${e.to}`} className="truncate">
              <span className="text-text">{nameOf(e.from)}</span> → <span className="text-text">{nameOf(e.to)}</span>{" "}
              <span className="font-mono">· {graphEdgeLabel(e)}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </Card>
  );
}
