// The place graph: places as nodes, recorded traversals as directed edges.
// Schematic layout, not geographic.

import { useEffect, useMemo, useRef } from "react";

import { Banner, Card, CardHeader } from "../../components/ui";
import { canvasToken, CANVAS_BG, MAX_PLAN_PRIMITIVES, prepareCanvas, useCanvasSize } from "./LocalMapCanvas";
import { graphEdgeLabel, layoutPlaceGraph } from "./mapView";
import type { PlaceGraphState } from "./mapPolling";
import { plural } from "./places";

export function PlaceGraphCard({ state, paused }: { state: PlaceGraphState; paused: boolean }) {
  const { graph, missing, error } = state;
  const [wrapRef, size] = useCanvasSize(260, 240, { w: 800, h: 360 });
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const layout = useMemo(() => (graph ? layoutPlaceGraph(graph, size.w, size.h) : null), [graph, size]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = prepareCanvas(canvas, size.w, size.h);
    if (!ctx || !layout) return;
    // The layout caps nodes/edges (256/2048), well under the page-wide bound.
    let primitives = 0;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = canvasToken("edge", "rgba(79,140,255,0.72)");
    for (const edge of layout.edges) {
      if (primitives >= MAX_PLAN_PRIMITIVES) break;
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
      if (primitives >= MAX_PLAN_PRIMITIVES) break;
      ctx.fillText(graphEdgeLabel(edge.edge), edge.labelX + 4, edge.labelY);
      primitives += 1;
    }
    const node = canvasToken("node", "#4f8cff");
    for (const n of layout.nodes) {
      if (primitives >= MAX_PLAN_PRIMITIVES) break;
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

  const message = paused
    ? "Feed paused. The last place graph is frozen."
    : missing
      ? "This places service has no /places/graph yet; no routes are drawn."
      : error && !graph
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
