import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import { Banner, Button, Card } from "../components/ui";
import { httpUrl } from "../runtime/urls";
import {
  decodePlaceLayer,
  fitPlanView,
  formatAge,
  formatPayloadBytes,
  graphEdgeLabel,
  gridRuns,
  hasPlaceMapContent,
  isPlaceGraph,
  isPlaceMapPayload,
  isStructureSnapshot,
  layoutPlaceGraph,
  normalizePlaceGraph,
  normalizePlaceMap,
  normalizeStructure,
  panPlanView,
  placeCaptureGate,
  placeMapAgeSeconds,
  placeMapBounds,
  placeMapGravityLabel,
  placeMapHasGravity,
  placeMapLegend,
  placeMapMissingKind,
  scaleBarMeters,
  screenHeading,
  worldToScreen,
  zoomPlanView,
  type GridMask,
  type PlaceGraph,
  type PlaceGraphEdge,
  type PlaceMap,
  type PlaceMapMissingKind,
  type PlanView,
} from "./mapView";

/// Per-place map page. The single global map is retired: this screen shows the
/// selected place's own stored local top-down map plus the place graph of
/// recorded routes between places. Both come from the Places service (:9095).
///
/// Observe-only: this page never starts SLAM and never invents walls, cells, or
/// routes. A missing/404 endpoint or an empty record is an explicit blank state.
/// Local-map coordinates are metres in the place's own anchor-relative frame
/// (z = floor plane). Drawing is capped at MAX_PLAN_PRIMITIVES rectangles /
/// segments / labels.
///
/// Layer naming is deliberate: the white lines are fitted walls (real bounding
/// structure), while the grey cells are obstacles/furniture and the green cells
/// are observed free space.

const PLACES_PORT = 9095;
const STRUCTURE_PORT = 9096;
const MIN_POLL_MS = 2000;
const LARGE_POLL_MS = 5000;
const HUGE_POLL_MS = 10000;
const GRAPH_POLL_MS = 15000;
const PLACES_POLL_MS = 15000;
const WHEREAMI_POLL_MS = 3000;
const REQUEST_TIMEOUT_MS = 4000;
const POST_TIMEOUT_MS = 15000;
const LARGE_BYTES = 1_500_000;
const HUGE_BYTES = 2_800_000;
const MAX_PLAN_PRIMITIVES = 30000;

const FREE_FILL = "rgba(53,194,143,0.16)";
const OBSTACLE_FILL = "rgba(216,222,233,0.30)";
const WALL_STROKE = "#d8dee9";
const ANCHOR_FILL = "#ffd166";
const EDGE_STROKE = "rgba(79,140,255,0.72)";
const NODE_FILL = "#4f8cff";

interface MapScreenProps {
  robotIp: string;
  onBack: () => void;
  backLabel?: string;
}

interface PlaceListEntry {
  id: string;
  name: string;
  last_seen_at?: number | null;
}

interface WhereAmI {
  matched?: boolean;
  place?: { id: string; name: string } | null;
  score?: number;
  margin?: number;
  min_score?: number;
  reason?: string;
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.error === "string") return body.error;
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}`;
}

function fmtNum(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function placePath(placeId: string, suffix: string): string {
  return `/places/${encodeURIComponent(placeId)}${suffix}`;
}

export function MapScreen({ robotIp, onBack, backLabel = "Back" }: MapScreenProps) {
  const [places, setPlaces] = useState<PlaceListEntry[]>([]);
  const [placesLoading, setPlacesLoading] = useState(true);
  const [placesError, setPlacesError] = useState<string | null>(null);
  const [where, setWhere] = useState<WhereAmI | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [map, setMap] = useState<PlaceMap | null>(null);
  const [mapMissing, setMapMissing] = useState<PlaceMapMissingKind | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapReceivedAt, setMapReceivedAt] = useState(0);
  const [mapStoredAge, setMapStoredAge] = useState<number | null>(null);
  const [mapBytes, setMapBytes] = useState(0);
  const [mapPollMs, setMapPollMs] = useState(MIN_POLL_MS);
  const [capturing, setCapturing] = useState(false);
  const [captureNote, setCaptureNote] = useState<string | null>(null);

  const [graph, setGraph] = useState<PlaceGraph | null>(null);
  const [graphMissing, setGraphMissing] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [graphReceivedAt, setGraphReceivedAt] = useState(0);

  const [paused, setPaused] = useState(false);
  const [retryKey, setRetryKey] = useState(0);
  const [now, setNow] = useState(() => performance.now());

  const [size, setSize] = useState({ w: 800, h: 440 });
  const [graphSize, setGraphSize] = useState({ w: 800, h: 360 });
  const [view, setView] = useState<PlanView>({
    centerX: 0,
    centerY: 0,
    scale: 40,
    width: 800,
    height: 440,
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const graphCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const graphWrapRef = useRef<HTMLDivElement | null>(null);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const drag = useRef<{ x: number; y: number } | null>(null);
  const pinch = useRef<number | null>(null);
  const pollMsRef = useRef(MIN_POLL_MS);
  const userMoved = useRef(false);
  const fitPlaceRef = useRef<string | null>(null);
  const manualPick = useRef(false);
  const liveRef = useRef(true);
  liveRef.current = !paused;

  const selectedPlace = useMemo(
    () => places.find((p) => p.id === selectedId) ?? null,
    [places, selectedId],
  );

  useEffect(() => {
    const id = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setSize({ w: Math.max(300, r.width), h: Math.max(300, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = graphWrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r)
        setGraphSize({
          w: Math.max(300, r.width),
          h: Math.max(240, r.height),
        });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setView((v) =>
      v.width === size.w && v.height === size.h
        ? v
        : { ...v, width: size.w, height: size.h },
    );
  }, [size]);

  // --- places list + recognition -----------------------------------------

  const loadPlaces = useCallback(async () => {
    if (!robotIp) return;
    setPlacesLoading(true);
    try {
      const res = await fetch(httpUrl(robotIp, PLACES_PORT, "/places"), {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(await readError(res));
      const body = (await res.json()) as { places?: PlaceListEntry[] };
      setPlaces(Array.isArray(body.places) ? body.places : []);
      setPlacesError(null);
    } catch (e) {
      setPlacesError(
        `${(e as Error).message}. Places runs in bebop-vision on :${PLACES_PORT}.`,
      );
    } finally {
      setPlacesLoading(false);
    }
  }, [robotIp]);

  useEffect(() => {
    void loadPlaces();
  }, [loadPlaces, retryKey]);

  useEffect(() => {
    if (paused || !robotIp) return;
    const id = window.setInterval(() => {
      if (liveRef.current) void loadPlaces();
    }, PLACES_POLL_MS);
    return () => window.clearInterval(id);
  }, [loadPlaces, paused, robotIp]);

  const pollWhere = useCallback(async () => {
    if (!robotIp) return;
    try {
      const res = await fetch(httpUrl(robotIp, PLACES_PORT, "/whereami"), {
        cache: "no-store",
      });
      if (!res.ok) return;
      setWhere((await res.json()) as WhereAmI);
    } catch {
      /* transient; the places list surfaces hard errors */
    }
  }, [robotIp]);

  useEffect(() => {
    if (paused || !robotIp) return;
    void pollWhere();
    const id = window.setInterval(() => {
      if (liveRef.current) void pollWhere();
    }, WHEREAMI_POLL_MS);
    return () => window.clearInterval(id);
  }, [pollWhere, paused, robotIp]);

  // Drop a selection that disappeared. A manual pick wins over recognition
  // until that place is gone; otherwise default to the recognized place from
  // GET /whereami when it is taught (whereami can arrive after the list), and
  // fall back to the first taught place.
  useEffect(() => {
    if (selectedId && !places.some((p) => p.id === selectedId)) {
      manualPick.current = false;
      setSelectedId(null);
    }
  }, [places, selectedId]);

  useEffect(() => {
    if (places.length === 0) return;
    if (manualPick.current && selectedId) return;
    const recognized =
      where?.matched && where.place ? where.place.id : null;
    const next =
      recognized && places.some((p) => p.id === recognized)
        ? recognized
        : places[0].id;
    if (next !== selectedId) setSelectedId(next);
  }, [places, selectedId, where]);

  // A place switch starts from a blank slate: never show the previous place's
  // map while the new request is in flight.
  useEffect(() => {
    setMap(null);
    setMapMissing(null);
    setMapError(null);
    setMapBytes(0);
    setMapReceivedAt(0);
    setMapStoredAge(null);
    setMapPollMs(MIN_POLL_MS);
    pollMsRef.current = MIN_POLL_MS;
    setMapLoading(!!selectedId);
    setCaptureNote(null);
    userMoved.current = false;
  }, [selectedId]);

  // --- selected place local map ------------------------------------------

  useEffect(() => {
    if (paused || !robotIp || !selectedId) return;
    let cancelled = false;
    let timer = 0;
    let ctrl: AbortController | null = null;

    const poll = async () => {
      ctrl = new AbortController();
      const timeout = window.setTimeout(() => ctrl?.abort(), REQUEST_TIMEOUT_MS);
      try {
        const res = await fetch(
          httpUrl(robotIp, PLACES_PORT, placePath(selectedId, "/map")),
          { cache: "no-store", signal: ctrl.signal },
        );
        if (res.status === 404) {
          // Distinguish "no map recorded for this place" from "this service
          // build predates the endpoint" using the sibling graph endpoint.
          let graphStatus: number | null = null;
          try {
            const g = await fetch(
              httpUrl(robotIp, PLACES_PORT, "/places/graph"),
              { cache: "no-store" },
            );
            graphStatus = g.status;
          } catch {
            graphStatus = null;
          }
          if (!cancelled) {
            setMap(null);
            setMapError(null);
            setMapBytes(0);
            setMapStoredAge(null);
            setMapMissing(placeMapMissingKind(404, graphStatus));
            setMapReceivedAt(performance.now());
            setMapLoading(false);
          }
          return;
        }
        if (!res.ok) {
          throw new Error(
            `places service returned ${res.status} ${res.statusText}`,
          );
        }
        const text = await res.text();
        const contentLength = Number(res.headers.get("content-length"));
        const bytes =
          Number.isFinite(contentLength) && contentLength > 0
            ? contentLength
            : text.length;
        const nextPoll =
          bytes >= HUGE_BYTES
            ? HUGE_POLL_MS
            : bytes >= LARGE_BYTES
              ? LARGE_POLL_MS
              : MIN_POLL_MS;
        pollMsRef.current = nextPoll;
        const body: unknown = JSON.parse(text);
        if (!isPlaceMapPayload(body)) {
          throw new Error("places service returned an unexpected place-map payload");
        }
        const next = normalizePlaceMap(body);
        if (!cancelled) {
          setMapPollMs(nextPoll);
          setMapBytes(bytes);
          const storedAge = placeMapAgeSeconds(next.updated_at, Date.now());
          startTransition(() => {
            setMap(next);
            setMapStoredAge(storedAge);
            setMapReceivedAt(performance.now());
            setMapError(null);
            setMapMissing(null);
            setMapLoading(false);
          });
        }
      } catch (e) {
        if (cancelled) return;
        setMapLoading(false);
        const msg = e instanceof Error ? e.message : String(e);
        setMapError(
          (e as { name?: string }).name === "AbortError"
            ? "Place-map request timed out (4 s)."
            : msg,
        );
      } finally {
        window.clearTimeout(timeout);
        if (!cancelled) timer = window.setTimeout(poll, pollMsRef.current);
      }
    };

    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      ctrl?.abort();
    };
  }, [robotIp, selectedId, paused, retryKey]);

  // --- place graph -------------------------------------------------------

  useEffect(() => {
    if (paused || !robotIp) return;
    let cancelled = false;
    let timer = 0;
    let ctrl: AbortController | null = null;

    const poll = async () => {
      ctrl = new AbortController();
      const timeout = window.setTimeout(() => ctrl?.abort(), REQUEST_TIMEOUT_MS);
      try {
        const res = await fetch(httpUrl(robotIp, PLACES_PORT, "/places/graph"), {
          cache: "no-store",
          signal: ctrl.signal,
        });
        if (res.status === 404) {
          if (!cancelled) {
            setGraph(null);
            setGraphMissing(true);
            setGraphError(null);
            setGraphReceivedAt(performance.now());
          }
          return;
        }
        if (!res.ok) {
          throw new Error(
            `places service returned ${res.status} ${res.statusText}`,
          );
        }
        const body: unknown = await res.json();
        if (!isPlaceGraph(body)) {
          throw new Error("places service returned an unexpected graph payload");
        }
        if (!cancelled) {
          setGraph(normalizePlaceGraph(body));
          setGraphMissing(false);
          setGraphError(null);
          setGraphReceivedAt(performance.now());
        }
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        setGraphError(
          (e as { name?: string }).name === "AbortError"
            ? "Place-graph request timed out (4 s)."
            : msg,
        );
      } finally {
        window.clearTimeout(timeout);
        if (!cancelled) timer = window.setTimeout(poll, GRAPH_POLL_MS);
      }
    };

    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      ctrl?.abort();
    };
  }, [robotIp, paused, retryKey]);

  // --- local map geometry + fit ------------------------------------------

  const layers = useMemo<{
    free: GridMask | null;
    occupied: GridMask | null;
  }>(
    () => ({
      free: decodePlaceLayer(map, "free"),
      occupied: decodePlaceLayer(map, "occupied"),
    }),
    [map],
  );

  const layerCounts = useMemo(() => {
    const count = (mask: GridMask | null): number | null => {
      if (!mask) return null;
      let n = 0;
      for (let i = 0; i < mask.cells.length; i += 1) {
        if (mask.cells[i] === mask.value) n += 1;
      }
      return n;
    };
    return { free: count(layers.free), occupied: count(layers.occupied) };
  }, [layers]);

  const mapHasContent = hasPlaceMapContent(map);
  const plan = useMemo<PlanView>(
    () => ({ ...view, width: size.w, height: size.h }),
    [view, size],
  );

  useEffect(() => {
    if (!map || !mapHasContent) return;
    const newPlace = fitPlaceRef.current !== selectedId;
    // Auto-fit when the place changes, and keep following the record until the
    // operator pans/zooms away. Manual Fit always applies.
    if (!newPlace && userMoved.current) return;
    if (newPlace) {
      fitPlaceRef.current = selectedId;
      userMoved.current = false;
    }
    setView(fitPlanView(placeMapBounds(map), size.w, size.h));
  }, [map, mapHasContent, selectedId, size]);

  const fit = () => {
    if (!map) return;
    userMoved.current = false;
    setView(fitPlanView(placeMapBounds(map), size.w, size.h));
  };

  const clientAge = mapReceivedAt
    ? Math.max(0, (now - mapReceivedAt) / 1000)
    : Infinity;
  const staleAfterS = Math.max(5, (mapPollMs / 1000) * 3);
  const mapStale = !!map && !paused && clientAge > staleAfterS;
  const planDimmed = mapStale;
  const planAlpha = planDimmed ? 0.35 : 1;
  const storedAge =
    mapStoredAge == null
      ? null
      : mapStoredAge + (Number.isFinite(clientAge) ? clientAge : 0);

  // --- local map canvas ---------------------------------------------------

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheelNative = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const anchorX = e.clientX - rect.left;
      const anchorY = e.clientY - rect.top;
      const factor = Math.exp(-e.deltaY * 0.0015);
      userMoved.current = true;
      setView((v) => zoomPlanView(v, factor, anchorX, anchorY));
    };
    canvas.addEventListener("wheel", onWheelNative, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheelNative);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = size.w;
    const h = size.h;
    if (
      canvas.width !== Math.max(1, Math.round(w * dpr)) ||
      canvas.height !== Math.max(1, Math.round(h * dpr))
    ) {
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0b0d10";
    ctx.fillRect(0, 0, w, h);

    if (!map || !mapHasContent) return;

    // Reserve a handful of primitives for the scale bar drawn after the plan,
    // so the whole canvas stays within MAX_PLAN_PRIMITIVES.
    const budgetLimit = MAX_PLAN_PRIMITIVES - 8;
    let primitives = 0;
    ctx.save();
    ctx.globalAlpha = planAlpha;

    // One rectangle per merged row run, capped so a huge grid can never turn
    // into an unbounded draw call.
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

    drawLayer(layers.free, FREE_FILL);
    drawLayer(layers.occupied, OBSTACLE_FILL);

    // Walls are real bounding structure only and may legitimately be absent
    // or few; obstacles still show through the grey grid layer.
    if (map.walls.length > 0 && primitives < budgetLimit) {
      ctx.strokeStyle = WALL_STROKE;
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
      // Screen-space left normal (screen Y points down).
      const px = -hy;
      const py = hx;
      ctx.fillStyle = ANCHOR_FILL;
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
      primitives += 3;
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
  }, [size, map, mapHasContent, layers, plan, planAlpha]);

  // --- graph canvas -------------------------------------------------------

  const graphLayout = useMemo(
    () => (graph ? layoutPlaceGraph(graph, graphSize.w, graphSize.h) : null),
    [graph, graphSize],
  );

  useEffect(() => {
    const canvas = graphCanvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = graphSize.w;
    const h = graphSize.h;
    if (
      canvas.width !== Math.max(1, Math.round(w * dpr)) ||
      canvas.height !== Math.max(1, Math.round(h * dpr))
    ) {
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0b0d10";
    ctx.fillRect(0, 0, w, h);
    if (!graphLayout) return;

    // The layout caps nodes/edges (256/2048), so drawing stays well under the
    // page-wide 30000-primitive bound; the counter makes that explicit.
    let primitives = 0;
    const budget = MAX_PLAN_PRIMITIVES;

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = EDGE_STROKE;
    for (const edge of graphLayout.edges) {
      if (primitives >= budget) break;
      ctx.beginPath();
      ctx.moveTo(edge.x0, edge.y0);
      ctx.lineTo(edge.x1, edge.y1);
      ctx.stroke();
      // Direction arrow near the target node.
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
    for (const edge of graphLayout.edges) {
      if (primitives >= budget) break;
      ctx.fillText(graphEdgeLabel(edge.edge), edge.labelX + 4, edge.labelY);
      primitives += 1;
    }

    for (const node of graphLayout.nodes) {
      if (primitives >= budget) break;
      const radius = 7 + Math.min(6, node.degree);
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = NODE_FILL;
      ctx.fill();
      ctx.strokeStyle = "rgba(242,244,247,0.7)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(242,244,247,0.92)";
      ctx.textAlign = "center";
      ctx.fillText(node.name, node.x, node.y + radius + 14);
      ctx.textAlign = "start";
      primitives += 2;
    }
  }, [graphLayout, graphSize]);

  // --- interaction --------------------------------------------------------

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
        const ax = (pts[0].x + pts[1].x) / 2 - rect.left;
        const ay = (pts[0].y + pts[1].y) / 2 - rect.top;
        const ratio = d / pinch.current;
        userMoved.current = true;
        setView((v) => zoomPlanView(v, ratio, ax, ay));
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
    if (pointers.current.size === 0) {
      drag.current = null;
      pinch.current = null;
    } else {
      const pts = [...pointers.current.values()];
      pinch.current =
        pts.length >= 2
          ? Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y)
          : null;
    }
  };

  async function capturePlace() {
    if (!selectedId || capturing) return;
    const placeName = selectedPlace?.name ?? selectedId;
    setCapturing(true);
    setCaptureNote(null);
    try {
      // 1. Re-confirm recognition at capture time. Storing a map from another
      // room would persist a wrong anchor and corrupt the place graph, so this
      // check is repeated here (the button is also disabled).
      let latestWhere: WhereAmI | null = null;
      {
        const ctrl = new AbortController();
        const timeout = window.setTimeout(
          () => ctrl.abort(),
          REQUEST_TIMEOUT_MS,
        );
        try {
          const res = await fetch(httpUrl(robotIp, PLACES_PORT, "/whereami"), {
            cache: "no-store",
            signal: ctrl.signal,
          });
          if (!res.ok) {
            throw new Error(
              `whereami returned ${res.status} ${res.statusText}`,
            );
          }
          latestWhere = (await res.json()) as WhereAmI;
        } catch (e) {
          const msg =
            (e as { name?: string }).name === "AbortError"
              ? "whereami timed out (4 s)"
              : e instanceof Error
                ? e.message
                : String(e);
          setCaptureNote(
            `Could not confirm the robot's current place (${msg}). Nothing was captured.`,
          );
          return;
        } finally {
          window.clearTimeout(timeout);
        }
      }
      setWhere(latestWhere);
      const gate = placeCaptureGate(selectedId, placeName, latestWhere);
      if (!gate.allowed) {
        setCaptureNote(`${gate.reason} Nothing was captured.`);
        return;
      }

      // 2. Read the live structure snapshot afresh — never reuse a cached
      // payload. The ingest endpoint expects exactly this document.
      let structureText = "";
      {
        const ctrl = new AbortController();
        const timeout = window.setTimeout(
          () => ctrl.abort(),
          REQUEST_TIMEOUT_MS,
        );
        try {
          const res = await fetch(
            httpUrl(robotIp, STRUCTURE_PORT, "/structure"),
            { cache: "no-store", signal: ctrl.signal },
          );
          if (res.status === 404 || res.status === 503) {
            setCaptureNote(
              "The live structure feed is unavailable (live SLAM off or the vision service is restarting). Nothing was captured.",
            );
            return;
          }
          if (!res.ok) {
            throw new Error(
              `structure feed returned ${res.status} ${res.statusText}`,
            );
          }
          structureText = await res.text();
        } catch (e) {
          const msg =
            (e as { name?: string }).name === "AbortError"
              ? "structure request timed out (4 s)"
              : e instanceof Error
                ? e.message
                : String(e);
          setCaptureNote(
            `Can't read the live structure feed (${msg}). Live SLAM may be off or the vision service restarting. Nothing was captured.`,
          );
          return;
        } finally {
          window.clearTimeout(timeout);
        }
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(structureText);
      } catch {
        setCaptureNote(
          "The live structure feed returned invalid JSON. Nothing was captured.",
        );
        return;
      }
      if (!isStructureSnapshot(parsed)) {
        setCaptureNote(
          "The live structure feed returned an unexpected payload. Nothing was captured.",
        );
        return;
      }
      const snapshot = normalizeStructure(parsed);

      // 3. A submap with no gravity estimate cannot be anchored. Refuse
      // rather than storing an unusable map.
      if (snapshot.gravity.valid !== true) {
        setCaptureNote(
          "Gravity is not estimated yet at this place, so a submap can't be anchored. Wait for a valid floor/gravity estimate, then capture again. Nothing was captured.",
        );
        return;
      }

      // 4. POST the live snapshot verbatim.
      const ctrl = new AbortController();
      const timeout = window.setTimeout(() => ctrl.abort(), POST_TIMEOUT_MS);
      try {
        const res = await fetch(
          httpUrl(robotIp, PLACES_PORT, placePath(selectedId, "/map")),
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: structureText,
            signal: ctrl.signal,
          },
        );
        if (res.status === 404) {
          setMapMissing("endpoint-missing");
          setCaptureNote(
            "This places-service build has no map-ingest endpoint yet; nothing was captured.",
          );
          return;
        }
        if (!res.ok) throw new Error(await readError(res));
        let revision: number | null = null;
        try {
          const body: unknown = await res.json();
          const raw =
            body && typeof body === "object" && "map" in body
              ? (body as { map?: unknown }).map
              : body;
          if (raw && typeof raw === "object" && "revision" in raw) {
            const r = Number((raw as { revision?: unknown }).revision);
            if (Number.isFinite(r)) revision = r;
          }
        } catch {
          /* the capture succeeded; the revision is just a nicety */
        }
        setMapMissing(null);
        setMapLoading(true);
        setCaptureNote(
          `Captured ${placeName}${revision != null ? ` · revision ${revision}` : ""}. Refreshing the stored map…`,
        );
        setRetryKey((k) => k + 1);
      } catch (e) {
        const msg =
          (e as { name?: string }).name === "AbortError"
            ? "map ingest timed out (15 s)"
            : e instanceof Error
              ? e.message
              : String(e);
        setCaptureNote(`Capture failed: ${msg}. Nothing was stored.`);
      } finally {
        window.clearTimeout(timeout);
      }
    } finally {
      setCapturing(false);
    }
  }

  // --- status copy --------------------------------------------------------

  const badge = (() => {
    const danger = "bg-danger/15 text-[#ffb5b8] border-danger/40";
    const dim = "bg-bg-elev-2 text-text-dim border-border";
    const accent = "bg-accent/15 text-accent border-accent/40";
    const ok = "bg-success/15 text-success border-success/40";
    const stale = "bg-danger/12 text-[#ffb5b8] border-danger/35";
    if (paused)
      return { label: "PAUSED · MAP FROZEN", className: dim };
    if (placesError) return { label: "PLACES UNREACHABLE", className: danger };
    if (placesLoading && places.length === 0)
      return { label: "LOADING PLACES", className: accent };
    if (places.length === 0) return { label: "NO PLACES YET", className: dim };
    if (!selectedId) return { label: "NO PLACE SELECTED", className: dim };
    if (mapMissing === "endpoint-missing")
      return { label: "PLACE-MAP ENDPOINT MISSING", className: danger };
    if (mapMissing === "no-map")
      return { label: "NO MAP CAPTURED", className: dim };
    if (mapError) return { label: "MAP UNREACHABLE", className: danger };
    if (mapLoading && !map) return { label: "LOADING LOCAL MAP", className: accent };
    if (!map) return { label: "NO LOCAL MAP", className: dim };
    if (!mapHasContent) return { label: "EMPTY MAP RECORD", className: dim };
    if (mapStale) return { label: "STALE · LAST MAP", className: stale };
    return { label: "LOCAL MAP · STORED", className: ok };
  })();

  const selectedName = selectedPlace?.name ?? "this place";
  const captureGate = placeCaptureGate(
    selectedId,
    selectedName,
    where,
    paused,
  );
  const captureBlockReason = captureGate.reason;
  const mapMessage = (() => {
    if (paused)
      return "Feed paused. The last place map is frozen and no freshness is claimed until you resume.";
    if (places.length === 0)
      return "No places yet. Teach a place in Places first; this page never invents a map.";
    if (!selectedId)
      return "Select a place to view its local map.";
    if (mapMissing === "endpoint-missing")
      return "This places service does not expose /places/{id}/map yet. Nothing is drawn or invented.";
    if (mapMissing === "no-map")
      return `No local map captured for ${selectedName} yet. Drive this place and use Capture this place.`;
    if (mapError && !map)
      return `Can't read the local map for ${selectedName}. ${mapError}`;
    if (mapLoading && !map) return `Loading the local map for ${selectedName}…`;
    if (!map) return `No local map for ${selectedName}.`;
    if (!mapHasContent)
      return "The stored record has no walls or occupancy grid, so there is nothing to draw.";
    return null;
  })();

  const graphMessage = (() => {
    if (paused) return "Feed paused. The last place graph is frozen.";
    if (graphMissing)
      return "This places service does not expose /places/graph yet; no routes are drawn.";
    if (graphError && !graph) return `Can't read the place graph. ${graphError}`;
    if (!graph) return "Loading the place graph…";
    if (graph.nodes.length === 0) return "No places in the graph yet.";
    if (graph.edges.length === 0)
      return "No recorded routes yet. Drive between places to record traversals.";
    return null;
  })();

  const legend = placeMapLegend(selectedName, map);

  return (
    <div className="flex flex-col flex-1 gap-4">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mt-2">
        <div>
          <h2 className="text-2xl font-bold">Map</h2>
          <p className="text-text-dim text-sm mt-1 max-w-2xl leading-relaxed">
            Local map for one place · recorded routes between places · estimated
            gravity · not loop-corrected.
          </p>
        </div>
        <span
          className={`self-start rounded-full border px-2.5 py-1 text-[11px] font-semibold tracking-wide ${badge.className}`}
        >
          {badge.label}
        </span>
      </div>

      {paused ? (
        <Banner tone="info">
          Polling paused. The last local map and graph are frozen, and no
          freshness is claimed until you resume.
        </Banner>
      ) : null}
      {placesError ? (
        <Banner tone="error">
          Can&rsquo;t read the places list from{" "}
          <code>
            {robotIp}:{PLACES_PORT}/places
          </code>
          . {placesError}
          <div className="mt-2">
            <Button variant="secondary" onClick={() => setRetryKey((k) => k + 1)}>
              Retry now
            </Button>
          </div>
        </Banner>
      ) : null}

      {/* Place picker */}
      <Card>
        <div className="py-2 flex flex-col gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs text-text-dim uppercase tracking-wider">
              Place
            </div>
            <div className="flex items-center gap-2">
              {where?.matched && where.place ? (
                <span className="text-[11px] text-text-dim">
                  recognized: <span className="text-text">{where.place.name}</span>
                </span>
              ) : (
                <span className="text-[11px] text-text-dim">
                  recognized: {where ? "unknown" : "—"}
                </span>
              )}
              <Button
                variant="ghost"
                onClick={() => void loadPlaces()}
                loading={placesLoading}
                disabled={paused}
              >
                Refresh
              </Button>
            </div>
          </div>
          {places.length === 0 ? (
            <p className="text-text-dim text-sm">
              No places yet. Teach one in Places, then return here.
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {places.map((p) => {
                const active = p.id === selectedId;
                return (
                  <button
                    key={p.id}
                    onClick={() => {
                      manualPick.current = true;
                      setSelectedId(p.id);
                    }}
                    className={`rounded-[var(--radius-card)] border px-3 py-2 text-sm transition-colors ${
                      active
                        ? "border-accent bg-bg-elev-2 text-text"
                        : "border-border bg-bg-elev text-text-dim hover:bg-bg-elev-2"
                    }`}
                  >
                    {p.name}
                    {where?.matched && where.place?.id === p.id ? (
                      <span className="ml-2 text-[10px] text-success">here</span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </Card>

      {/* Local map controls */}
      <Card>
        <div className="flex flex-wrap items-center gap-2 py-2">
          <Button variant="secondary" onClick={fit} disabled={!map || !mapHasContent}>
            Fit
          </Button>
          <Button
            variant={paused ? "primary" : "secondary"}
            onClick={() => {
              const next = !paused;
              setPaused(next);
              if (next) return;
              setMapError(null);
              setPlacesError(null);
              setRetryKey((k) => k + 1);
            }}
          >
            {paused ? "Resume" : "Pause"}
          </Button>
          <Button
            variant="secondary"
            onClick={() => void capturePlace()}
            loading={capturing}
            disabled={!!captureBlockReason}
            title={captureBlockReason ?? "Capture the live structure snapshot for this place"}
          >
            Capture this place
          </Button>
          <span className="text-[11px] text-text-dim">
            drag to pan · wheel/pinch to zoom
          </span>
        </div>
        {captureBlockReason ? (
          <div className="text-[11px] text-text-dim pb-2">
            {captureBlockReason}
          </div>
        ) : null}
        {captureNote ? (
          <div className="text-[11px] text-text-dim pb-2">{captureNote}</div>
        ) : null}
      </Card>

      {mapError && map ? (
        <Banner tone="info">
          Map refresh failed; showing the last stored map for {selectedName}.{" "}
          {mapError}
        </Banner>
      ) : null}

      <div
        ref={wrapRef}
        className="relative h-[440px] sm:h-[540px] rounded-[var(--radius-card)] border border-border overflow-hidden bg-[#0b0d10]"
      >
        <canvas
          ref={canvasRef}
          className="w-full h-full touch-none cursor-grab active:cursor-grabbing"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endPointer}
          onPointerCancel={endPointer}
          aria-label="Local top-down map for the selected place"
        />
        {mapMessage ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-center text-text-dim text-sm pointer-events-none px-6">
            <span className="font-semibold text-text">
              {mapMessage.split(".")[0]}
            </span>
            <span>{mapMessage.split(".").slice(1).join(".").trim()}</span>
          </div>
        ) : null}
        {paused ? (
          <div className="absolute bottom-2 right-2 rounded-md bg-bg/85 border border-border px-2 py-1 text-[11px] text-text-dim">
            frozen
          </div>
        ) : null}
        <div className="absolute bottom-2 left-2 rounded-md bg-bg/85 border border-border px-2 py-1 text-[10px] text-text-dim max-w-[76%]">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1">
              <i className="inline-block w-2.5 h-2.5 rounded-sm bg-success/30" />
              free
            </span>
            <span className="inline-flex items-center gap-1">
              <i className="inline-block w-2.5 h-2.5 rounded-sm bg-[#d8dee9]/30" />
              obstacle
            </span>
            <span className="inline-flex items-center gap-1">
              <i className="inline-block w-3 h-0.5 bg-[#d8dee9]" />
              wall
            </span>
            <span className="inline-flex items-center gap-1">
              <i className="inline-block w-2 h-2 rounded-full bg-[#ffd166]" />
              anchor
            </span>
          </div>
          <div>{legend}</div>
        </div>
      </div>

      {/* Local map stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <div className="py-2 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <Stat label="Place" value={selectedName} />
            <Stat label="Revision" value={map ? String(map.revision) : "—"} />
            <Stat
              label="Walls"
              value={map ? map.walls.length.toLocaleString() : "—"}
            />
            <Stat
              label="Obstacle cells"
              value={layerCounts.occupied?.toLocaleString() ?? "—"}
            />
            <Stat
              label="Free cells"
              value={layerCounts.free?.toLocaleString() ?? "—"}
            />
            <Stat
              label="Grid"
              value={
                map && map.width > 0 && map.height > 0
                  ? `${map.width}×${map.height} @ ${fmtNum(map.resolution_m, 2)} m`
                  : "—"
              }
            />
            <Stat
              label="Anchor"
              value={
                map?.anchor
                  ? `${fmtNum(map.anchor.x, 2)}, ${fmtNum(map.anchor.y, 2)} · ${fmtNum(map.anchor.yaw_deg, 0)}°`
                  : "—"
              }
            />
            <Stat
              label="Camera height"
              value={
                map?.camera_height_m != null
                  ? `${fmtNum(map.camera_height_m, 2)} m`
                  : "—"
              }
            />
          </div>
        </Card>
        <Card>
          <div className="py-2 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <Stat label="Gravity" value={placeMapGravityLabel(map)} />
            <Stat
              label="Tilt"
              value={
                map && placeMapHasGravity(map)
                  ? `r ${fmtNum(map.gravity.roll_deg)}° / p ${fmtNum(map.gravity.pitch_deg)}°`
                  : "—"
              }
            />
            <Stat
              label="Stored map age"
              value={formatAge(storedAge)}
            />
            <Stat
              label="Fetched"
              value={mapReceivedAt ? formatAge(clientAge) : "—"}
            />
            <Stat
              label="Cache state"
              value={
                paused
                  ? "frozen"
                  : mapStale
                    ? "stale"
                    : map
                      ? "fresh"
                      : "—"
              }
              valueClass={mapStale ? "text-danger" : "text-text"}
            />
            <Stat
              label="Poll / payload"
              value={`${(mapPollMs / 1000).toFixed(mapPollMs % 1000 ? 1 : 0)} s${
                mapBytes > 0
                  ? ` · ${formatPayloadBytes(mapBytes)}`
                  : ""
              }`}
            />
          </div>
        </Card>
      </div>

      {/* Place graph */}
      <Card>
        <div className="py-2 flex flex-col gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs text-text-dim uppercase tracking-wider">
              Place graph
            </div>
            <span className="text-[11px] text-text-dim">
              {graph
                ? `${graph.nodes.length} place${graph.nodes.length === 1 ? "" : "s"} · ${graph.edges.length} recorded route${graph.edges.length === 1 ? "" : "s"}`
                : "—"}
            </span>
          </div>
          <p className="text-text-dim text-xs leading-relaxed">
            Edges are <span className="text-text">recorded routes</span>, not
            measured distances. Labels show traversals and successes; the layout
            is schematic, not geographic.
          </p>
          {graphError && graph ? (
            <Banner tone="info">
              Graph refresh failed; showing the last recorded graph. {graphError}
            </Banner>
          ) : null}
          <div
            ref={graphWrapRef}
            className="relative h-[360px] rounded-[var(--radius-card)] border border-border overflow-hidden bg-[#0b0d10]"
          >
            <canvas
              ref={graphCanvasRef}
              className="w-full h-full"
              aria-label="Place graph of recorded routes"
            />
            {graphMessage ? (
              <div className="absolute inset-0 flex items-center justify-center text-center text-text-dim text-sm pointer-events-none px-6">
                {graphMessage}
              </div>
            ) : null}
            {graphLayout &&
            (graphLayout.hiddenNodes > 0 || graphLayout.hiddenEdges > 0) ? (
              <div className="absolute bottom-2 right-2 rounded-md bg-bg/85 border border-border px-2 py-1 text-[10px] text-text-dim">
                showing {graphLayout.nodes.length}/
                {graphLayout.nodes.length + graphLayout.hiddenNodes} places ·{" "}
                {graphLayout.edges.length}/
                {graphLayout.edges.length + graphLayout.hiddenEdges} routes
              </div>
            ) : null}
          </div>
          {graph && graph.edges.length > 0 ? (
            <ul className="flex flex-col gap-1 text-[11px] font-mono text-text-dim">
              {graph.edges.slice(0, 12).map((e: PlaceGraphEdge) => (
                <li key={`${e.from}->${e.to}`}>
                  <span className="text-text">
                    {graph.nodes.find((n) => n.id === e.from)?.name ?? e.from}
                  </span>{" "}
                  →{" "}
                  <span className="text-text">
                    {graph.nodes.find((n) => n.id === e.to)?.name ?? e.to}
                  </span>{" "}
                  · {graphEdgeLabel(e)}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </Card>

      <div className="flex items-center justify-between gap-3 mt-auto pt-2">
        <span className="text-[11px] text-text-dim">
          Read-only ·{" "}
          {placesLoading && places.length === 0
            ? "connecting…"
            : paused
              ? "paused"
              : `places ${(PLACES_POLL_MS / 1000).toFixed(0)} s · map ${(
                  mapPollMs / 1000
                ).toFixed(mapPollMs % 1000 ? 1 : 0)} s · graph ${(
                  GRAPH_POLL_MS / 1000
                ).toFixed(0)} s${
                  graphReceivedAt
                    ? ` · graph ${formatAge((now - graphReceivedAt) / 1000)} old`
                    : ""
                }`}
        </span>
        <Button variant="ghost" onClick={onBack}>
          {backLabel}
        </Button>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  valueClass = "text-text",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] uppercase tracking-wider text-text-dim">
        {label}
      </div>
      <div
        className={`font-mono text-[13px] truncate ${valueClass}`}
        title={value}
      >
        {value}
      </div>
    </div>
  );
}
