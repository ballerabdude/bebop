// Memory › Map & graph: the selected place's own stored local top-down map
// plus the place graph of recorded routes, both from the places service
// (:9095). Observe-only: it never starts SLAM and never invents walls, cells
// or routes; a missing endpoint or an empty record is an explicit blank
// state. "Capture this place" stores the live SLAM snapshot (:9096, only
// with --live-slam) after re-checking recognition.

import { useEffect, useMemo, useRef, useState } from "react";

import { Banner, Button, Card, CardHeader, cn, Stat, StatusPill } from "../../components/ui";
import type { Tone } from "../../components/ui";
import { PORTS } from "../../robot/endpoints";
import { PLACES_POLL_MS } from "../../robot/placesStore";
import { usePlaces, useRobot } from "../../robot/session";
import { capturePlaceMap } from "./capturePlaceMap";
import { LocalMapCanvas } from "./LocalMapCanvas";
import { GRAPH_POLL_MS, usePlaceGraph, usePlaceMap } from "./mapPolling";
import {
  formatAge,
  formatPayloadBytes,
  hasPlaceMapContent,
  placeCaptureGate,
  placeMapGravityLabel,
  placeMapHasGravity,
  placeMapLegend,
} from "./mapView";
import type { GridMask } from "./mapView";
import { PlaceGraphCard } from "./PlaceGraphCard";
import type { WhereAmI } from "./places";
import { decodePlaceLayer } from "./mapView";

interface PlaceEntry {
  id: string;
  name: string;
}

function fmt(v: number | null | undefined, digits = 1): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

function countCells(mask: GridMask | null): number | null {
  if (!mask) return null;
  let n = 0;
  for (let i = 0; i < mask.cells.length; i += 1) if (mask.cells[i] === mask.value) n += 1;
  return n;
}

const secs = (ms: number) => `${(ms / 1000).toFixed(ms % 1000 ? 1 : 0)} s`;

export function MapView() {
  const { endpoints } = useRobot();
  const api = endpoints.places;
  const [paused, setPaused] = useState(false);
  // Shared poller; while paused this view keeps the last values (frozen)
  // and stops counting as a consumer.
  const shared = usePlaces(!paused);
  const frozen = useRef(shared);
  if (!paused) frozen.current = shared;
  const places = frozen.current.places as unknown as PlaceEntry[];
  const where = frozen.current.whereami as WhereAmI | null;
  const placesError = frozen.current.error;
  const placesLoading = frozen.current.reachable === null;

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const manualPick = useRef(false);
  const [retryKey, setRetryKey] = useState(0);
  const [fitSignal, setFitSignal] = useState(0);
  const [capturing, setCapturing] = useState(false);
  const [captureNote, setCaptureNote] = useState<string | null>(null);
  const [now, setNow] = useState(() => performance.now());

  const [m, patchMap] = usePlaceMap(api, selectedId, paused, retryKey);
  const graph = usePlaceGraph(api, paused, retryKey);
  const map = m.map;

  useEffect(() => {
    const id = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  // A manual pick wins until that place disappears; otherwise follow the
  // recognized place, falling back to the first taught place.
  useEffect(() => {
    if (selectedId && !places.some((p) => p.id === selectedId)) {
      manualPick.current = false;
      setSelectedId(null);
      return;
    }
    if (places.length === 0 || (manualPick.current && selectedId)) return;
    const recognized = where?.matched && where.place ? where.place.id : null;
    const next = recognized && places.some((p) => p.id === recognized) ? recognized : places[0].id;
    if (next !== selectedId) setSelectedId(next);
  }, [places, selectedId, where]);

  useEffect(() => setCaptureNote(null), [selectedId]);

  const counts = useMemo(
    () => ({
      free: countCells(decodePlaceLayer(map, "free")),
      occupied: countCells(decodePlaceLayer(map, "occupied")),
    }),
    [map],
  );

  const hasContent = hasPlaceMapContent(map);
  const clientAge = m.receivedAt ? Math.max(0, (now - m.receivedAt) / 1000) : Infinity;
  const stale = !!map && !paused && clientAge > Math.max(5, (m.pollMs / 1000) * 3);
  const storedAge = m.storedAge == null ? null : m.storedAge + (Number.isFinite(clientAge) ? clientAge : 0);
  const selectedName = places.find((p) => p.id === selectedId)?.name ?? "this place";
  const gate = placeCaptureGate(selectedId, selectedName, where, paused);

  const badge: { tone: Tone; label: string } = paused
    ? { tone: "dim", label: "paused · frozen" }
    : placesError
      ? { tone: "err", label: "places unreachable" }
      : placesLoading
        ? { tone: "info", label: "loading" }
        : places.length === 0
          ? { tone: "dim", label: "no places yet" }
          : m.missing === "endpoint-missing"
            ? { tone: "err", label: "map endpoint missing" }
            : m.missing === "no-map"
              ? { tone: "dim", label: "no map captured" }
              : m.error && !map
                ? { tone: "err", label: "map unreachable" }
                : !map
                  ? { tone: "info", label: "loading map" }
                  : !hasContent
                    ? { tone: "dim", label: "empty record" }
                    : stale
                      ? { tone: "warn", label: "stale · last map" }
                      : { tone: "ok", label: "stored map" };

  const message = paused
    ? "Feed paused. The last place map is frozen and no freshness is claimed until you resume."
    : places.length === 0
      ? "No places yet. Teach a place in Places first; this page never invents a map."
      : !selectedId
        ? "Select a place. Its local map shows here."
        : m.missing === "endpoint-missing"
          ? "No map endpoint. This places service has no /places/{id}/map yet; nothing is drawn."
          : m.missing === "no-map"
            ? `No local map for ${selectedName} yet. Drive this place with live SLAM on and use Capture this place.`
            : m.error && !map
              ? `Can't read the local map. ${m.error}`
              : !map
                ? `Loading the local map for ${selectedName}…`
                : !hasContent
                  ? "Nothing to draw. The stored record has no walls or occupancy grid."
                  : null;

  async function capture() {
    if (!selectedId || capturing) return;
    setCapturing(true);
    setCaptureNote(null);
    try {
      const r = await capturePlaceMap({ api, slam: endpoints.slam, placeId: selectedId, placeName: selectedName });
      setCaptureNote(r.note);
      if (r.endpointMissing) patchMap({ missing: "endpoint-missing" });
      if (r.ok) {
        patchMap({ missing: null, loading: true });
        setRetryKey((k) => k + 1);
      }
      void shared.refresh();
    } finally {
      setCapturing(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {placesError ? (
        <Banner
          tone="error"
          title={`Places service unreachable (:${PORTS.places})`}
          action={
            <Button variant="secondary" size="sm" onClick={() => void shared.refresh()}>
              Retry
            </Button>
          }
        >
          {placesError}
        </Banner>
      ) : null}

      <Card className="flex flex-col gap-3">
        <CardHeader
          eyebrow="Local map"
          title="Per-place map · estimated gravity · not loop-corrected"
          actions={<StatusPill tone={badge.tone} dot>{badge.label}</StatusPill>}
        />
        {places.length > 0 ? (
          <div className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1 [scrollbar-width:none] sm:mx-0 sm:flex-wrap sm:px-0">
            {places.map((p) => {
              const active = p.id === selectedId;
              const here = where?.matched && where.place?.id === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => {
                    manualPick.current = true;
                    setSelectedId(p.id);
                  }}
                  className={cn(
                    "inline-flex min-h-touch shrink-0 cursor-pointer items-center gap-2 rounded-full border px-3.5 text-sm font-semibold transition-colors",
                    active
                      ? "border-accent bg-accent/15 text-text"
                      : "border-border text-text-dim hover:bg-bg-elev-2 hover:text-text",
                  )}
                >
                  {p.name}
                  {here ? <span className="h-2 w-2 rounded-full bg-success" aria-label="robot is here" /> : null}
                </button>
              );
            })}
          </div>
        ) : null}
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="secondary" size="sm" onClick={() => setFitSignal((n) => n + 1)} disabled={!hasContent}>
            Fit
          </Button>
          <Button
            variant={paused ? "primary" : "secondary"}
            size="sm"
            onClick={() => {
              setPaused((p) => !p);
              if (paused) setRetryKey((k) => k + 1);
            }}
          >
            {paused ? "Resume" : "Pause"}
          </Button>
          <Button size="sm" variant="secondary" onClick={() => void capture()} loading={capturing} disabled={!!gate.reason}>
            Capture this place
          </Button>
          <span className="hidden text-label text-text-dim sm:inline">drag to pan · wheel or pinch to zoom</span>
        </div>
        {gate.reason || captureNote ? (
          <p className="text-label leading-snug text-text-dim">{captureNote ?? gate.reason}</p>
        ) : null}
      </Card>

      {m.error && map ? (
        <Banner tone="info">Map refresh failed; showing the last stored map for {selectedName}. {m.error}</Banner>
      ) : null}

      <LocalMapCanvas
        map={map}
        placeId={selectedId}
        dimmed={stale}
        fitSignal={fitSignal}
        message={message}
        frozen={paused}
        legend={placeMapLegend(selectedName, map)}
      />

      <Card>
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
          <Stat label="Revision" value={map ? String(map.revision) : "—"} />
          <Stat label="Walls" value={map ? map.walls.length.toLocaleString() : "—"} />
          <Stat label="Obstacle cells" value={counts.occupied?.toLocaleString() ?? "—"} />
          <Stat label="Free cells" value={counts.free?.toLocaleString() ?? "—"} />
          <Stat
            label="Grid"
            value={map && map.width > 0 ? `${map.width}×${map.height} @ ${fmt(map.resolution_m, 2)} m` : "—"}
          />
          <Stat
            label="Anchor"
            value={map?.anchor ? `${fmt(map.anchor.x, 2)}, ${fmt(map.anchor.y, 2)} · ${fmt(map.anchor.yaw_deg, 0)}°` : "—"}
          />
          <Stat label="Camera height" value={map?.camera_height_m != null ? `${fmt(map.camera_height_m, 2)} m` : "—"} />
          <Stat label="Gravity" value={placeMapGravityLabel(map)} />
          <Stat
            label="Tilt"
            value={map && placeMapHasGravity(map) ? `r ${fmt(map.gravity.roll_deg)}° / p ${fmt(map.gravity.pitch_deg)}°` : "—"}
          />
          <Stat label="Stored map age" value={formatAge(storedAge)} />
          <Stat
            label="Fetched"
            value={m.receivedAt ? formatAge(clientAge) : "—"}
            hint={paused ? "frozen" : stale ? <span className="text-danger-fg">stale</span> : map ? "fresh" : undefined}
          />
          <Stat label="Poll / payload" value={`${secs(m.pollMs)}${m.bytes > 0 ? ` · ${formatPayloadBytes(m.bytes)}` : ""}`} />
        </div>
      </Card>

      <PlaceGraphCard state={graph} paused={paused} />

      <p className="text-caption text-text-dim">
        Read-only ·{" "}
        {paused
          ? "paused"
          : `places ${secs(PLACES_POLL_MS)} · map ${secs(m.pollMs)} · graph ${secs(GRAPH_POLL_MS)}${
              graph.receivedAt ? ` · graph ${formatAge((now - graph.receivedAt) / 1000)} old` : ""
            }`}
      </p>
    </div>
  );
}
