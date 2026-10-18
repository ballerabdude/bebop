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
  display_name?: string;
  status?: string;
  source?: string;
  confirmed_at?: number | null;
  seen_count?: number;
  name_source?: string | null;
  suggested_name?: string | null;
  suggested_facts?: string[];
  suggested_at?: number | null;
  best_match?: BestMatch | null;
  best_match_id?: string | null;
  best_match_score?: number | null;
  created_at: number;
  updated_at: number;
  last_seen_at: number | null;
  odom: number[] | null;
  facts: Fact[];
  tags: Tag[];
  keyframes: Keyframe[];
}
interface BestMatch {
  id: string;
  name: string;
  score: number | null;
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
  const [pending, setPending] = useState<Place[]>([]);
  const [pendingSelected, setPendingSelected] = useState<string[]>([]);
  const [pendingNames, setPendingNames] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [where, setWhere] = useState<WhereAmI | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pendingBusy, setPendingBusy] = useState(false);
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
      const [confirmedRes, pendingRes] = await Promise.all([
        fetch(httpUrl(robotIp, PLACES_PORT, "/places"), {
          cache: "no-store",
        }),
        fetch(httpUrl(robotIp, PLACES_PORT, "/places/pending"), {
          cache: "no-store",
        }),
      ]);
      if (!confirmedRes.ok) throw new Error(await readError(confirmedRes));
      const body = await confirmedRes.json();
      const confirmed: Place[] = body.places ?? [];
      setPlaces(confirmed);
      if (pendingRes.ok) {
        const pendingBody = await pendingRes.json();
        const waiting: Place[] = pendingBody.places ?? [];
        setPending(waiting);
        setPendingSelected((previous) =>
          previous.filter((id) => waiting.some((place) => place.id === id)),
        );
        setPendingNames((previous) =>
          Object.fromEntries(
            Object.entries(previous).filter(([id]) =>
              waiting.some((place) => place.id === id),
            ),
          ),
        );
      } else {
        // Older places services do not expose the pending queue; keep the
        // confirmed list usable rather than failing the whole screen.
        setPending([]);
      }
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

  const pendingNameFor = (place: Place) =>
    (pendingNames[place.id] ?? place.suggested_name ?? "").trim();

  const selectedPendingPlaces = pending.filter((p) =>
    pendingSelected.includes(p.id),
  );
  const allSelectedNamed =
    selectedPendingPlaces.length > 0 &&
    selectedPendingPlaces.every((p) => pendingNameFor(p).length > 0);

  function togglePending(id: string) {
    setPendingSelected((previous) =>
      previous.includes(id)
        ? previous.filter((value) => value !== id)
        : [...previous, id],
    );
  }

  async function suggestOne(id: string): Promise<boolean> {
    const res = await fetch(
      httpUrl(
        robotIp,
        PLACES_PORT,
        `/places/pending/${encodeURIComponent(id)}/suggest`,
      ),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
    );
    const body = await res.json();
    if (!res.ok || (body && body.error)) {
      throw new Error(body?.error ?? `${res.status} ${res.statusText}`);
    }
    setPendingNames((previous) => ({
      ...previous,
      [id]: body.suggested_name ?? "",
    }));
    setPending((previous) =>
      previous.map((place) =>
        place.id === id
          ? {
              ...place,
              display_name: body.suggested_name ?? place.display_name,
              suggested_name: body.suggested_name ?? null,
              suggested_facts: body.facts ?? [],
            }
          : place,
      ),
    );
    if (body.conflict) {
      setNote(
        `Suggested "${body.suggested_name}" conflicts with ` +
          `"${body.conflict.name}"; edit the name before importing.`,
      );
    }
    return true;
  }

  async function suggestSelected() {
    const targets = selectedPendingPlaces.filter((p) => !pendingNameFor(p));
    if (targets.length === 0) {
      setNote("Selected places already have names. Edit one or import as-is.");
      return;
    }
    setPendingBusy(true);
    setError(null);
    setNote(null);
    let suggested = 0;
    let failed = 0;
    try {
      for (const place of targets) {
        try {
          await suggestOne(place.id);
          suggested += 1;
        } catch {
          failed += 1;
        }
      }
      setNote(
        `${suggested} name suggestion(s) ready. Review/edit, then ` +
          `Import selected.`,
      );
      if (failed) {
        setError(
          `${failed} suggestion(s) failed. Type a name or retry; the ` +
            `pending places are unchanged.`,
        );
      }
    } finally {
      setPendingBusy(false);
    }
  }

  async function importSelected() {
    const targets = selectedPendingPlaces;
    if (targets.length === 0) return;
    const missing = targets.filter((p) => !pendingNameFor(p));
    if (missing.length > 0) {
      setNote("Name every selected place, or use Suggest names first.");
      return;
    }
    const names: Record<string, string> = {};
    for (const place of targets) names[place.id] = pendingNameFor(place);
    setPendingBusy(true);
    setError(null);
    setNote(null);
    try {
      const res = await fetch(
        httpUrl(robotIp, PLACES_PORT, "/places/pending/import"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ids: targets.map((place) => place.id),
            names,
          }),
        },
      );
      const body = await res.json();
      if (!res.ok) throw new Error(await readError(res));
      const failedIds = new Set<string>(
        (body.results ?? [])
          .filter((item: { status?: string }) => item.status === "failed")
          .map((item: { id: string }) => item.id),
      );
      setNote(
        `Imported ${body.imported ?? 0} place(s); merged ` +
          `${body.merged ?? 0}; ${body.failed ?? 0} failed.`,
      );
      if (failedIds.size > 0) {
        setError(
          `${failedIds.size} place(s) are still pending. Fix the name or ` +
            `conflict, then retry.`,
        );
      }
      setPendingSelected((previous) =>
        previous.filter((id) => failedIds.has(id)),
      );
      setPendingNames((previous) =>
        Object.fromEntries(
          Object.entries(previous).filter(([id]) => failedIds.has(id)),
        ),
      );
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPendingBusy(false);
    }
  }

  async function mergePending(place: Place, target: BestMatch) {
    if (
      !window.confirm(
        `Merge this pending place into "${target.name}"? ` +
          `The pending row is removed and its evidence is added to ` +
          `"${target.name}".`,
      )
    ) {
      return;
    }
    setPendingBusy(true);
    setError(null);
    setNote(null);
    try {
      const res = await fetch(
        httpUrl(robotIp, PLACES_PORT, "/places/pending/import"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ids: [place.id],
            merge_into: { [place.id]: target.id },
          }),
        },
      );
      const body = await res.json();
      if (!res.ok) throw new Error(await readError(res));
      const result = (body.results ?? [])[0];
      if (!result || result.status !== "merged") {
        throw new Error(
          result?.detail ?? result?.error ?? "merge was not applied",
        );
      }
      setNote(`Merged into "${target.name}".`);
      setPendingSelected((previous) =>
        previous.filter((id) => id !== place.id),
      );
      setPendingNames((previous) => {
        const next = { ...previous };
        delete next[place.id];
        return next;
      });
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPendingBusy(false);
    }
  }

  async function discardPending(place: Place) {
    if (
      !window.confirm(
        `Discard this pending place? Its keyframes and descriptors are ` +
          `deleted. This cannot be undone.`,
      )
    ) {
      return;
    }
    setPendingBusy(true);
    setError(null);
    setNote(null);
    try {
      const res = await fetch(
        httpUrl(robotIp, PLACES_PORT, `/places/${encodeURIComponent(place.id)}`),
        { method: "DELETE" },
      );
      if (!res.ok) throw new Error(await readError(res));
      setNote("Pending place discarded.");
      setPendingSelected((previous) =>
        previous.filter((id) => id !== place.id),
      );
      setPendingNames((previous) => {
        const next = { ...previous };
        delete next[place.id];
        return next;
      });
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPendingBusy(false);
    }
  }

  async function discardSelected() {
    if (selectedPendingPlaces.length === 0) return;
    if (
      !window.confirm(
        `Discard ${selectedPendingPlaces.length} pending place(s)? ` +
          `Their images and descriptors are deleted.`,
      )
    ) {
      return;
    }
    setPendingBusy(true);
    setError(null);
    setNote(null);
    try {
      await Promise.all(
        selectedPendingPlaces.map(async (place) => {
          const res = await fetch(
            httpUrl(
              robotIp,
              PLACES_PORT,
              `/places/${encodeURIComponent(place.id)}`,
            ),
            { method: "DELETE" },
          );
          if (!res.ok) throw new Error(await readError(res));
        }),
      );
      setPendingSelected([]);
      setPendingNames({});
      setNote("Selected pending places discarded.");
      await load();
    } catch (e) {
      setError((e as Error).message);
      await load();
    } finally {
      setPendingBusy(false);
    }
  }

  const detail = places.find((p) => p.id === selected) ?? null;

  return (
    <div className="flex flex-col flex-1 gap-4">
      <div className="flex items-baseline justify-between gap-3 mt-2">
        <h2 className="text-2xl font-bold">
          Places
          {pending.length > 0 ? (
            <span className="ml-2 text-sm font-normal text-text-dim">
              {places.length} confirmed · {pending.length} pending
            </span>
          ) : null}
        </h2>
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
          {!where?.matched && pending.length > 0 ? (
            <div className="text-text-dim text-xs">
              {pending.length} pending place
              {pending.length === 1 ? "" : "s"} waiting for import; pending
              places are not recognized yet.
            </div>
          ) : null}
        </div>
      </Card>

      {/* Pending review queue (captured without an LLM) */}
      {pending.length > 0 ? (
        <Card>
          <div className="py-2 flex flex-col gap-3">
            <div className="flex items-baseline justify-between gap-2">
              <div>
                <div className="text-xs text-text-dim uppercase tracking-wider">
                  Pending places
                </div>
                <p className="text-text-dim text-xs mt-1">
                  Captured automatically by the scout. They are not recognized
                  until you import them.
                </p>
              </div>
              <div className="text-sm font-semibold">{pending.length}</div>
            </div>

            <div className="flex flex-col gap-2">
              {pending.map((p) => {
                const kf = p.keyframes[0];
                const checked = pendingSelected.includes(p.id);
                const score = p.best_match?.score ?? null;
                return (
                  <div
                    key={p.id}
                    className={`flex gap-3 rounded-[var(--radius-card)] border px-3 py-2.5 ${
                      checked
                        ? "border-accent bg-bg-elev-2"
                        : "border-border bg-bg-elev"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => togglePending(p.id)}
                      className="mt-1 shrink-0"
                    />
                    {kf ? (
                      <img
                        src={keyframeUrl(robotIp, p.id, kf.path)}
                        alt=""
                        className="w-16 h-12 object-cover rounded-md border border-border shrink-0"
                      />
                    ) : (
                      <div className="w-16 h-12 rounded-md border border-border bg-bg-elev-2 shrink-0" />
                    )}
                    <div className="min-w-0 flex-1 flex flex-col gap-1.5">
                      <div className="flex items-baseline justify-between gap-2">
                        <div className="font-semibold truncate">
                          {p.display_name ?? "Unnamed place"}
                        </div>
                        <div className="text-text-dim text-xs shrink-0">
                          {p.seen_count ?? 1} sighting
                          {(p.seen_count ?? 1) === 1 ? "" : "s"}
                        </div>
                      </div>
                      <div className="text-text-dim text-xs">
                        first seen{" "}
                        {new Date(p.created_at * 1000).toLocaleString()}
                        {p.last_seen_at
                          ? ` · last ${new Date(
                              p.last_seen_at * 1000,
                            ).toLocaleTimeString()}`
                          : ""}
                        {` · ${p.keyframes.length} view${
                          p.keyframes.length === 1 ? "" : "s"
                        }`}
                        {p.best_match
                          ? ` · closest ${p.best_match.name} ${
                              score !== null ? score.toFixed(2) : ""
                            }`
                          : ""}
                      </div>
                      {p.suggested_facts && p.suggested_facts.length > 0 ? (
                        <div className="text-text-dim text-xs">
                          model facts: {p.suggested_facts.join("; ")}
                        </div>
                      ) : null}
                      <input
                        className="bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-2 text-text text-sm outline-none focus:border-accent"
                        value={pendingNames[p.id] ?? p.suggested_name ?? ""}
                        onChange={(e) =>
                          setPendingNames((previous) => ({
                            ...previous,
                            [p.id]: e.target.value,
                          }))
                        }
                        placeholder="Name this place, or use Suggest names"
                      />
                      {p.best_match && (score ?? 0) >= 0.8 ? (
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            onClick={() => void mergePending(p, p.best_match!)}
                          >
                            Merge into {p.best_match.name}
                          </Button>
                          <span className="text-text-dim text-xs">
                            secondary action; no model call
                          </span>
                        </div>
                      ) : null}
                    </div>
                    <Button
                      variant="ghost"
                      onClick={() => void discardPending(p)}
                    >
                      Discard
                    </Button>
                  </div>
                );
              })}
            </div>

            <div className="flex gap-2 flex-wrap">
              <Button
                variant="secondary"
                onClick={() => void suggestSelected()}
                loading={pendingBusy}
                disabled={
                  pendingBusy ||
                  selectedPendingPlaces.filter((p) => !pendingNameFor(p))
                    .length === 0
                }
              >
                Suggest names
              </Button>
              <Button
                onClick={() => void importSelected()}
                loading={pendingBusy}
                disabled={pendingBusy || !allSelectedNamed}
              >
                Import selected
              </Button>
              <Button
                variant="ghost"
                onClick={() => void discardSelected()}
                disabled={pendingBusy || selectedPendingPlaces.length === 0}
              >
                Discard selected
              </Button>
            </div>
            <p className="text-text-dim text-xs">
              Suggest names runs the model once per selected unnamed place.
              Import is deterministic; nothing is confirmed until you press it.
              Naming a place yourself skips the model entirely.
            </p>
          </div>
        </Card>
      ) : null}

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
                <div className="font-semibold truncate">
                  {p.display_name ?? p.name}
                </div>
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
