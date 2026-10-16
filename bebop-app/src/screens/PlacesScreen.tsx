import { useCallback, useEffect, useRef, useState } from "react";

import { Banner, Button, Card, Field, Spinner } from "../components/ui";
import { httpUrl } from "../runtime/urls";

/// Places: review and teach the robot's semantic place memory. Talks to the
/// Places service that runs inside bebop-vision on :9095 (started with the
/// vision card). A place is a named room/location with keyframe images,
/// DINOv3 descriptors for recognition, and operator/vision facts.

const PLACES_PORT = 9095;
const WHEREAMI_POLL_MS = 3000;

interface Fact {
  text: string;
  source: string;
  created_at: number;
}
interface Keyframe {
  path: string;
  role: string;
  stamp_us: number;
}
interface Tag {
  label: string;
  confidence: number;
  source: string;
}
interface Place {
  id: string;
  name: string;
  created_at: number;
  updated_at: number;
  last_seen_at: number | null;
  odom: number[] | null;
  facts: Fact[];
  tags: Tag[];
  keyframes: Keyframe[];
}
interface Match {
  id: string;
  name: string;
  score: number;
}
interface WhereAmI {
  matched: boolean;
  place?: Place | null;
  score?: number;
  margin?: number;
  min_score?: number;
  reason?: string;
  top?: Match[];
}

function baseName(path: string): string {
  return path.split("/").pop() ?? path;
}

function keyframeUrl(robotIp: string, placeId: string, path: string): string {
  return httpUrl(
    robotIp,
    PLACES_PORT,
    `/places/${encodeURIComponent(placeId)}/keyframes/${encodeURIComponent(baseName(path))}`,
  );
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

export function PlacesScreen({
  robotIp,
  onBack,
}: {
  robotIp: string;
  runtimePort?: number;
  onBack: () => void;
}) {
  const [places, setPlaces] = useState<Place[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [where, setWhere] = useState<WhereAmI | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const [newName, setNewName] = useState("");
  const [factText, setFactText] = useState("");
  const [live, setLive] = useState(true);

  const liveRef = useRef(live);
  liveRef.current = live;

  const load = useCallback(async () => {
    if (!robotIp) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(httpUrl(robotIp, PLACES_PORT, "/places"), {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(await readError(res));
      const body = await res.json();
      setPlaces(body.places ?? []);
    } catch (e) {
      setError(
        `${(e as Error).message}. Is the vision card on? (Places runs in ` +
          `bebop-vision on :${PLACES_PORT}.)`,
      );
    } finally {
      setLoading(false);
    }
  }, [robotIp]);

  const pollWhere = useCallback(async () => {
    if (!robotIp) return;
    try {
      const res = await fetch(httpUrl(robotIp, PLACES_PORT, "/whereami"), {
        cache: "no-store",
      });
      if (!res.ok) return;
      setWhere(await res.json());
    } catch {
      /* transient; the load() path surfaces hard errors */
    }
  }, [robotIp]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    void pollWhere();
    const id = window.setInterval(() => {
      if (liveRef.current) void pollWhere();
    }, WHEREAMI_POLL_MS);
    return () => window.clearInterval(id);
  }, [pollWhere]);

  async function teach() {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const res = await fetch(httpUrl(robotIp, PLACES_PORT, "/places"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const body = await res.json();
      setNewName("");
      setNote(
        `Taught "${body.place.name}" from ${body.views} view(s) → ` +
          `${body.descriptors} descriptor(s).`,
      );
      await load();
      setSelected(body.place.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function addFact() {
    if (!selected || !factText.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(
        httpUrl(robotIp, PLACES_PORT, `/places/${encodeURIComponent(selected)}/facts`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: factText.trim() }),
        },
      );
      if (!res.ok) throw new Error(await readError(res));
      setFactText("");
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function capture(id: string, seconds: number) {
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const res = await fetch(
        httpUrl(robotIp, PLACES_PORT, `/places/${encodeURIComponent(id)}/views`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ seconds, interval_s: 1.5 }),
        },
      );
      if (!res.ok) throw new Error(await readError(res));
      const body = await res.json();
      if (body.keyframes === 0) {
        setNote("No new frame (camera unchanged) — move the robot and retry.");
      } else {
        setNote(
          `Captured ${body.keyframes} new view(s) → ${body.views_total} total.`,
        );
      }
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function removePlace(id: string, name: string) {
    if (!window.confirm(`Forget "${name}"? This cannot be undone.`)) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(
        httpUrl(robotIp, PLACES_PORT, `/places/${encodeURIComponent(id)}`),
        { method: "DELETE" },
      );
      if (!res.ok) throw new Error(await readError(res));
      if (selected === id) setSelected(null);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const detail = places.find((p) => p.id === selected) ?? null;

  return (
    <div className="flex flex-col flex-1 gap-4">
      <div className="flex items-baseline justify-between gap-3 mt-2">
        <h2 className="text-2xl font-bold">Places</h2>
        <Button variant="ghost" onClick={() => void load()} loading={loading}>
          Refresh
        </Button>
      </div>
      <p className="text-text-dim leading-relaxed">
        Named rooms the robot remembers, recognised by appearance (DINOv3) and
        described by you and the on-device model. Localisation is among taught
        places — an untaught room reads as unknown.
      </p>

      {error ? <Banner tone="error">{error}</Banner> : null}
      {note ? <Banner tone="success">{note}</Banner> : null}

      {/* Live recognition */}
      <Card>
        <div className="py-2 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="text-xs text-text-dim uppercase tracking-wider">
              Where am I
            </div>
            <label className="flex items-center gap-2 text-xs text-text-dim">
              <input
                type="checkbox"
                checked={live}
                onChange={(e) => setLive(e.target.checked)}
              />
              live
            </label>
          </div>
          {where?.matched && where.place ? (
            <div className="text-lg font-semibold">{where.place.name}</div>
          ) : (
            <div className="text-lg font-semibold text-text-dim">
              {where?.reason ? `unknown — ${where.reason}` : "unknown"}
            </div>
          )}
          {where?.score !== undefined ? (
            <div className="text-text-dim text-[13px] font-mono">
              score {where.score.toFixed(3)} · margin{" "}
              {(where.margin ?? 0).toFixed(3)} · min{" "}
              {(where.min_score ?? 0).toFixed(2)}
            </div>
          ) : null}
          {where?.top && where.top.length > 1 ? (
            <div className="text-text-dim text-xs font-mono">
              top: {where.top.map((t) => `${t.name} ${t.score.toFixed(2)}`).join(" · ")}
            </div>
          ) : null}
        </div>
      </Card>

      {/* Teach the current view */}
      <Card>
        <div className="py-2 flex flex-col gap-3">
          <div className="text-xs text-text-dim uppercase tracking-wider">
            Teach current view
          </div>
          <Field label="Room name" hint="Aim the camera at the room, then name it.">
            <input
              className="bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-2.5 text-text outline-none focus:border-accent"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="kitchen"
              onKeyDown={(e) => {
                if (e.key === "Enter") void teach();
              }}
            />
          </Field>
          <Button onClick={() => void teach()} loading={busy} disabled={!newName.trim()}>
            Teach this view
          </Button>
          <p className="text-text-dim text-xs">
            Already taught this room? Select it below and use{" "}
            <span className="text-text">Add view</span> to capture more angles.
          </p>
        </div>
      </Card>

      {/* List */}
      <div className="flex flex-col gap-2">
        <div className="text-xs text-text-dim uppercase tracking-wider">
          {places.length} place{places.length === 1 ? "" : "s"}
        </div>
        {loading && places.length === 0 ? (
          <div className="flex justify-center py-6">
            <Spinner large />
          </div>
        ) : null}
        {!loading && places.length === 0 && !error ? (
          <p className="text-text-dim text-sm">
            No places yet. Drive somewhere, aim the camera, and teach it above.
          </p>
        ) : null}
        {places.map((p) => {
          const kf = p.keyframes[0];
          const active = p.id === selected;
          return (
            <button
              key={p.id}
              onClick={() => setSelected(active ? null : p.id)}
              className={`flex items-center gap-3 text-left rounded-[var(--radius-card)] border px-3 py-2.5 transition-colors ${
                active
                  ? "border-accent bg-bg-elev-2"
                  : "border-border bg-bg-elev hover:bg-bg-elev-2"
              }`}
            >
              {kf ? (
                <img
                  src={keyframeUrl(robotIp, p.id, kf.path)}
                  alt=""
                  className="w-16 h-12 object-cover rounded-md border border-border shrink-0"
                />
              ) : (
                <div className="w-16 h-12 rounded-md border border-border bg-bg-elev-2 shrink-0" />
              )}
              <div className="min-w-0 flex-1">
                <div className="font-semibold truncate">{p.name}</div>
                <div className="text-text-dim text-xs truncate">
                  {p.keyframes.length} view{p.keyframes.length === 1 ? "" : "s"} ·{" "}
                  {p.facts.length} fact{p.facts.length === 1 ? "" : "s"}
                  {p.tags.length ? ` · ${p.tags.map((t) => t.label).join(", ")}` : ""}
                  {p.last_seen_at
                    ? ` · seen ${new Date(p.last_seen_at * 1000).toLocaleString()}`
                    : ""}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Detail */}
      {detail ? (
        <Card>
          <div className="py-2 flex flex-col gap-3">
            <div className="flex items-center justify-between gap-2">
              <div className="font-semibold text-lg">{detail.name}</div>
              <Button
                variant="ghost"
                onClick={() => void removePlace(detail.id, detail.name)}
              >
                Forget
              </Button>
            </div>

            {detail.keyframes.length ? (
              <div className="flex gap-2 flex-wrap">
                {detail.keyframes.map((kf) => (
                  <div key={kf.path} className="flex flex-col gap-1">
                    <img
                      src={keyframeUrl(robotIp, detail.id, kf.path)}
                      alt={kf.role}
                      className="w-40 h-28 object-cover rounded-md border border-border"
                    />
                    <span className="text-text-dim text-[11px]">{kf.role}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-text-dim text-sm">No keyframes.</p>
            )}

            <div className="flex items-center gap-2 flex-wrap">
              <Button
                variant="secondary"
                onClick={() => void capture(detail.id, 0)}
                loading={busy}
              >
                Add current view
              </Button>
              <Button
                variant="secondary"
                onClick={() => void capture(detail.id, 10)}
                loading={busy}
              >
                Capture 10 s (pan around)
              </Button>
              <span className="text-text-dim text-xs">
                {detail.keyframes.length} view
                {detail.keyframes.length === 1 ? "" : "s"}
              </span>
            </div>

            <div>
              <div className="text-xs text-text-dim uppercase tracking-wider mb-1">
                Facts
              </div>
              {detail.facts.length ? (
                <ul className="flex flex-col gap-1">
                  {detail.facts.map((f, i) => (
                    <li key={i} className="text-sm">
                      <span className="text-text">{f.text}</span>{" "}
                      <span className="text-text-dim text-xs">({f.source})</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-text-dim text-sm">No facts yet.</p>
              )}
            </div>

            <div className="flex gap-2">
              <input
                className="flex-1 bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-2.5 text-text outline-none focus:border-accent"
                value={factText}
                onChange={(e) => setFactText(e.target.value)}
                placeholder="add a fact (e.g. has a red mug)"
                onKeyDown={(e) => {
                  if (e.key === "Enter") void addFact();
                }}
              />
              <Button
                variant="secondary"
                onClick={() => void addFact()}
                loading={busy}
                disabled={!factText.trim()}
              >
                Add
              </Button>
            </div>
          </div>
        </Card>
      ) : null}

      <div className="mt-auto pt-4">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
      </div>
    </div>
  );
}
