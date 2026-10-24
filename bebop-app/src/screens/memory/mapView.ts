/// Pure state/layout helpers for Memory › Place graph. No DOM, no fake data.

export function formatAge(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 1) return `${seconds.toFixed(2)} s`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const whole = Math.floor(seconds);
  const d = Math.floor(whole / 86_400);
  const h = Math.floor((whole % 86_400) / 3_600);
  const m = Math.floor((whole % 3_600) / 60);
  if (whole < 3_600) return `${m}m ${whole % 60}s`;
  if (whole < 86_400) return `${h}h ${m}m`;
  return `${d}d ${h}h`;
}

// ---------------------------------------------------------------------------
// Place graph (GET /places/graph) on the places service (:9095)
//
// Nothing here synthesizes routes: a payload that does not parse is rejected,
// and the card says so.
// ---------------------------------------------------------------------------

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export interface PlaceGraphNode {
  id: string;
  name: string;
}

export interface PlaceGraphEdge {
  from: string;
  to: string;
  traversals: number;
  successes: number;
  median_duration_s: number | null;
  last_success_at: number | null;
}

export interface PlaceGraph {
  nodes: PlaceGraphNode[];
  edges: PlaceGraphEdge[];
}

export function isPlaceGraph(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return Array.isArray(v.nodes) && Array.isArray(v.edges);
}

/// Defensive normalize: keep only nodes with a non-empty id, dedupe ids, and
/// drop edges whose endpoints are not in the node list. Counts floor at 0.
export function normalizePlaceGraph(value: unknown): PlaceGraph {
  const v =
    value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const nodes: PlaceGraphNode[] = [];
  const seen = new Set<string>();
  for (const raw of Array.isArray(v.nodes) ? v.nodes : []) {
    if (!raw || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    if (typeof r.id !== "string" || r.id.length === 0 || seen.has(r.id)) continue;
    seen.add(r.id);
    nodes.push({
      id: r.id,
      name:
        typeof r.name === "string" && r.name.length > 0 ? r.name : r.id,
    });
  }
  const edges: PlaceGraphEdge[] = [];
  for (const raw of Array.isArray(v.edges) ? v.edges : []) {
    if (!raw || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    if (typeof r.from !== "string" || typeof r.to !== "string") continue;
    if (!seen.has(r.from) || !seen.has(r.to)) continue;
    const traversals = Math.max(
      0,
      Math.floor(finiteOrNull(r.traversals) ?? 0),
    );
    const successes = Math.min(
      traversals,
      Math.max(0, Math.floor(finiteOrNull(r.successes) ?? 0)),
    );
    edges.push({
      from: r.from,
      to: r.to,
      traversals,
      successes,
      median_duration_s: finiteOrNull(r.median_duration_s),
      last_success_at: finiteOrNull(r.last_success_at),
    });
  }
  return { nodes, edges };
}

export interface GraphNodeLayout {
  id: string;
  name: string;
  x: number;
  y: number;
  degree: number;
}

export interface GraphEdgeLayout {
  from: string;
  to: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  labelX: number;
  labelY: number;
  edge: PlaceGraphEdge;
}

export interface GraphLayout {
  nodes: GraphNodeLayout[];
  edges: GraphEdgeLayout[];
  /** Nodes/edges omitted by the render caps. */
  hiddenNodes: number;
  hiddenEdges: number;
}

/// Deterministic circular layout — readable, not geographic. Nodes are ordered
/// by the payload's node order; edges are straight lines between them. The
/// caller renders the counts, so this is honest about being a route graph
/// rather than a measured map.
export function layoutPlaceGraph(
  graph: PlaceGraph,
  width: number,
  height: number,
  padding = 48,
  maxNodes = 256,
  maxEdges = 2048,
): GraphLayout {
  const w = Number.isFinite(width) && width > 0 ? width : 0;
  const h = Number.isFinite(height) && height > 0 ? height : 0;
  const nodeCap = Math.max(0, Math.floor(maxNodes));
  const edgeCap = Math.max(0, Math.floor(maxEdges));
  const nodes = graph.nodes.slice(0, nodeCap);
  const included = new Set(nodes.map((n) => n.id));
  const allEdges = graph.edges.filter(
    (e) => included.has(e.from) && included.has(e.to) && e.from !== e.to,
  );
  const edges = allEdges.slice(0, edgeCap);

  const degree = new Map<string, number>();
  for (const e of allEdges) {
    degree.set(e.from, (degree.get(e.from) ?? 0) + 1);
    degree.set(e.to, (degree.get(e.to) ?? 0) + 1);
  }

  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.max(
    0,
    Math.min(w, h) / 2 - Math.max(0, padding),
  );
  const positions = new Map<string, { x: number; y: number }>();
  const laidOut = nodes.map((node, i) => {
    let x = cx;
    let y = cy;
    if (nodes.length === 2) {
      x = cx + (i === 0 ? -radius : radius);
    } else if (nodes.length > 1) {
      const a = -Math.PI / 2 + (2 * Math.PI * i) / nodes.length;
      x = cx + Math.cos(a) * radius;
      y = cy + Math.sin(a) * radius;
    }
    positions.set(node.id, { x, y });
    return {
      id: node.id,
      name: node.name,
      x,
      y,
      degree: degree.get(node.id) ?? 0,
    };
  });

  const offsets = new Map<string, number>();
  const edgeLayouts: GraphEdgeLayout[] = [];
  for (const edge of edges) {
    const a = positions.get(edge.from);
    const b = positions.get(edge.to);
    if (!a || !b) continue;
    const key = [edge.from, edge.to].sort().join("\u0000");
    const n = offsets.get(key) ?? 0;
    offsets.set(key, n + 1);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    // Fan reciprocal/multi edges apart so their labels don't stack.
    const shift = (n % 2 === 0 ? 1 : -1) * Math.ceil(n / 2) * 14;
    const nx = (-dy / len) * shift;
    const ny = (dx / len) * shift;
    edgeLayouts.push({
      from: edge.from,
      to: edge.to,
      x0: a.x + nx,
      y0: a.y + ny,
      x1: b.x + nx,
      y1: b.y + ny,
      labelX: (a.x + b.x) / 2 + nx,
      labelY: (a.y + b.y) / 2 + ny - 4,
      edge,
    });
  }

  return {
    nodes: laidOut,
    edges: edgeLayouts,
    hiddenNodes: Math.max(0, graph.nodes.length - nodes.length),
    // Self-loop edges are not drawable routes; everything else that is not in
    // `edges` (node cap, edge cap) counts as hidden for the caption.
    hiddenEdges: Math.max(
      0,
      graph.edges.filter((e) => e.from !== e.to).length - edges.length,
    ),
  };
}

/// Compact edge label: recorded traversal count and how many succeeded.
export function graphEdgeLabel(edge: PlaceGraphEdge): string {
  const duration =
    edge.median_duration_s != null && Number.isFinite(edge.median_duration_s)
      ? ` · ${edge.median_duration_s.toFixed(1)} s`
      : "";
  return `${edge.traversals}× · ${edge.successes} ok${duration}`;
}
