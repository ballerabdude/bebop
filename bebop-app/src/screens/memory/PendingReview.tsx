// Pending-place review: scout captures (no LLM) waiting to be named and
// imported. Phone-first cards (large thumbnail, name field, Import/Discard),
// plus batch select with a sticky action bar for clearing many at once.

import { useEffect, useMemo, useState } from "react";

import { Button, CardHeader, cn, INPUT_CLASS, StatusPill } from "../../components/ui";
import type { ConfirmOptions } from "../../components/ui";
import {
  deletePlace,
  importPending,
  keyframeUrl,
  MERGE_SCORE,
  formatMatchScore,
  plural,
  shortTime,
  suggestName,
} from "./places";
import type { Place, PlacesUrl } from "./places";

const PAGE = 12;

export type Notify = (tone: "success" | "error" | "info", text: string) => void;

interface Props {
  api: PlacesUrl;
  pending: Place[];
  refresh: () => Promise<void>;
  confirm: (o: ConfirmOptions) => Promise<boolean>;
  notify: Notify;
}

export function PendingReview({ api, pending, refresh, confirm, notify }: Props) {
  const [selected, setSelected] = useState<string[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [shown, setShown] = useState(PAGE);

  // Drop selections/names of places that left the queue.
  useEffect(() => {
    const ids = new Set(pending.map((p) => p.id));
    setSelected((s) => (s.every((id) => ids.has(id)) ? s : s.filter((id) => ids.has(id))));
    setNames((n) =>
      Object.keys(n).every((id) => ids.has(id))
        ? n
        : Object.fromEntries(Object.entries(n).filter(([id]) => ids.has(id))),
    );
  }, [pending]);

  const nameFor = (p: Place) => (names[p.id] ?? p.suggested_name ?? "").trim();
  const visible = pending.slice(0, shown);
  const chosen = useMemo(() => pending.filter((p) => selected.includes(p.id)), [pending, selected]);
  const unnamed = chosen.filter((p) => !nameFor(p));
  const allVisibleSelected = visible.length > 0 && visible.every((p) => selected.includes(p.id));

  const toggle = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  async function guarded(key: string, fn: () => Promise<void>) {
    setBusy(key);
    try {
      await fn();
    } catch (e) {
      notify("error", e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      await refresh();
    }
  }

  const importIds = (targets: Place[]) =>
    guarded(targets.length === 1 ? targets[0].id : "batch", async () => {
      const missing = targets.filter((p) => !nameFor(p));
      if (missing.length) throw new Error("Name every place first, or use Suggest names.");
      const res = await importPending(
        api,
        targets.map((p) => p.id),
        { names: Object.fromEntries(targets.map((p) => [p.id, nameFor(p)])) },
      );
      const failed = (res.results ?? []).filter((r) => r.status === "failed");
      setSelected((s) => s.filter((id) => failed.some((f) => f.id === id)));
      if (failed.length) {
        notify(
          "error",
          `${plural(failed.length, "place")} still pending: ${failed[0].detail ?? failed[0].error ?? "name conflict"}. Fix the name and retry.`,
        );
      } else {
        notify(
          "success",
          `Imported ${res.imported ?? 0}${res.merged ? `, merged ${res.merged}` : ""}. Recognition now includes ${targets.length === 1 ? `"${nameFor(targets[0])}"` : "them"}.`,
        );
      }
    });

  const discard = async (targets: Place[]) => {
    const ok = await confirm({
      title: targets.length === 1 ? "Discard this pending place?" : `Discard ${targets.length} pending places?`,
      body: "Their images and descriptors are deleted. This can't be undone.",
      confirmLabel: "Discard",
      tone: "danger",
    });
    if (!ok) return;
    await guarded(targets.length === 1 ? targets[0].id : "batch", async () => {
      await Promise.all(targets.map((p) => deletePlace(api, p.id)));
      setSelected((s) => s.filter((id) => !targets.some((p) => p.id === id)));
      notify("info", `Discarded ${plural(targets.length, "pending place")}.`);
    });
  };

  const merge = async (p: Place) => {
    const target = p.best_match!;
    const ok = await confirm({
      title: `Merge into "${target.name}"?`,
      body: `The pending capture is removed and its views are added to "${target.name}". No model call.`,
      confirmLabel: "Merge",
    });
    if (!ok) return;
    await guarded(p.id, async () => {
      const res = await importPending(api, [p.id], { mergeInto: { [p.id]: target.id } });
      const r = res.results?.[0];
      if (!r || r.status !== "merged") throw new Error(r?.detail ?? r?.error ?? "merge was not applied");
      notify("success", `Merged into "${target.name}".`);
    });
  };

  const suggest = () =>
    guarded("batch", async () => {
      let ok = 0;
      let failed = 0;
      for (const p of unnamed) {
        try {
          const r = await suggestName(api, p.id);
          setNames((n) => ({ ...n, [p.id]: r.suggested_name ?? "" }));
          if (r.conflict) notify("info", `"${r.suggested_name}" is already "${r.conflict.name}"; edit it before importing.`);
          ok += 1;
        } catch {
          failed += 1;
        }
      }
      notify(failed ? "error" : "success", `${plural(ok, "name")} suggested${failed ? `, ${failed} failed` : ""}. Review, then Import.`);
    });

  if (pending.length === 0) return null;

  return (
    <section className="flex flex-col gap-3" aria-label="Pending places">
      <CardHeader
        eyebrow={`Review · ${pending.length} pending`}
        title="New places the scout captured"
        description="Name a capture and import it to make it recognizable, merge it into the place it matches, or discard it. Nothing is confirmed until you import."
        actions={
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              setSelected((s) =>
                allVisibleSelected
                  ? s.filter((id) => !visible.some((p) => p.id === id))
                  : [...new Set([...s, ...visible.map((p) => p.id)])],
              )
            }
          >
            {allVisibleSelected ? "Clear selection" : `Select ${visible.length === pending.length ? "all" : "shown"}`}
          </Button>
        }
      />

      <ul className="m-0 grid list-none grid-cols-1 gap-3 p-0 sm:grid-cols-2 xl:grid-cols-3">
        {visible.map((p) => (
          <PendingCard
            key={p.id}
            api={api}
            place={p}
            checked={selected.includes(p.id)}
            onToggle={() => toggle(p.id)}
            name={names[p.id] ?? p.suggested_name ?? ""}
            onName={(v) => setNames((n) => ({ ...n, [p.id]: v }))}
            busy={busy === p.id || busy === "batch"}
            onImport={() => void importIds([p])}
            onDiscard={() => void discard([p])}
            onMerge={() => void merge(p)}
          />
        ))}
      </ul>
      {shown < pending.length ? (
        <Button variant="secondary" onClick={() => setShown((n) => n + PAGE)} className="self-center">
          Show {Math.min(PAGE, pending.length - shown)} more of {pending.length - shown}
        </Button>
      ) : null}

      {chosen.length > 0 ? (
        <div className="sticky bottom-2 z-10 flex flex-wrap items-center gap-2 rounded-card border border-accent/50 bg-bg-elev/95 p-2 shadow-xl backdrop-blur-md">
          <span className="px-1 text-sm font-semibold">{chosen.length} selected</span>
          <div className="ml-auto flex flex-wrap gap-2">
            {unnamed.length > 0 ? (
              <Button variant="secondary" size="sm" loading={busy === "batch"} disabled={busy !== null} onClick={() => void suggest()}>
                Suggest {unnamed.length === 1 ? "name" : `${unnamed.length} names`}
              </Button>
            ) : null}
            <Button size="sm" disabled={busy !== null || unnamed.length > 0} onClick={() => void importIds(chosen)}>
              Import {chosen.length}
            </Button>
            <Button variant="ghost" size="sm" className="text-danger-fg" disabled={busy !== null} onClick={() => void discard(chosen)}>
              Discard
            </Button>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function PendingCard({
  api,
  place: p,
  checked,
  onToggle,
  name,
  onName,
  busy,
  onImport,
  onDiscard,
  onMerge,
}: {
  api: PlacesUrl;
  place: Place;
  checked: boolean;
  onToggle: () => void;
  name: string;
  onName: (v: string) => void;
  busy: boolean;
  onImport: () => void;
  onDiscard: () => void;
  onMerge: () => void;
}) {
  const [view, setView] = useState(0);
  const kfs = p.keyframes;
  const kf = kfs.length ? kfs[view % kfs.length] : null;
  const score = p.best_match?.score ?? null;
  const canMerge = !!p.best_match && (score ?? 0) >= MERGE_SCORE;
  return (
    <li
      className={cn(
        "flex min-w-0 flex-col overflow-hidden rounded-card border bg-bg-elev",
        checked ? "border-accent ring-1 ring-accent/50" : "border-border",
      )}
    >
      <div className="relative aspect-video bg-bg-elev-2 sm:aspect-[4/3]">
        {kf ? (
          <button
            type="button"
            className="block h-full w-full cursor-pointer"
            onClick={() => setView((v) => v + 1)}
            aria-label={kfs.length > 1 ? "Show the next view" : "Capture view"}
          >
            <img
              src={keyframeUrl(api, p.id, kf.path)}
              alt=""
              loading="lazy"
              className="h-full w-full object-cover"
            />
          </button>
        ) : (
          <div className="flex h-full items-center justify-center text-label text-text-dim">no image</div>
        )}
        <label className="absolute left-0 top-0 flex h-12 cursor-pointer items-center gap-1.5 pl-2 pr-3">
          <span className="flex items-center gap-1.5 rounded-full bg-black/65 py-1 pr-2.5 pl-1.5 text-caption font-semibold text-white">
            <input type="checkbox" checked={checked} onChange={onToggle} />
            {checked ? "Selected" : "Select"}
          </span>
        </label>
        {kfs.length > 1 ? (
          <span className="pointer-events-none absolute bottom-2 right-2 rounded-full bg-black/65 px-2 py-0.5 text-caption font-semibold text-white">
            view {(view % kfs.length) + 1}/{kfs.length}
          </span>
        ) : null}
      </div>
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="text-label leading-snug text-text-dim">
          Seen {p.seen_count ?? 1}× · first {shortTime(p.created_at)}
          {p.last_seen_at ? ` · last ${shortTime(p.last_seen_at)}` : ""}
        </div>
        {p.best_match ? (
          <div className="flex flex-wrap items-center gap-2 text-label">
            <span className="text-text-dim">Looks like</span>
            <span className="font-semibold">{p.best_match.name}</span>
            {score !== null ? (
              <StatusPill
                tone={canMerge ? "info" : "dim"}
                title={canMerge ? "Similar enough to merge" : `Below the ${MERGE_SCORE.toFixed(2)} merge threshold`}
              >
                {formatMatchScore(score)}
              </StatusPill>
            ) : null}
            {score !== null && !canMerge ? (
              <span className="text-text-dim">below the {MERGE_SCORE.toFixed(2)} merge threshold</span>
            ) : null}
          </div>
        ) : null}
        {p.suggested_facts?.length ? (
          <div className="line-clamp-2 text-label text-text-dim">Model notes: {p.suggested_facts.join("; ")}</div>
        ) : null}
        <input
          className={INPUT_CLASS}
          value={name}
          onChange={(e) => onName(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && name.trim()) onImport();
          }}
          placeholder="Name this place"
          aria-label="Place name"
          enterKeyHint="done"
        />
        <div className="mt-auto grid grid-cols-2 gap-2">
          <Button size="sm" onClick={onImport} disabled={busy || !name.trim()}>
            Import
          </Button>
          <Button variant="ghost" size="sm" className="border border-border text-danger-fg" onClick={onDiscard} disabled={busy}>
            Discard
          </Button>
          {canMerge ? (
            <Button variant="secondary" size="sm" className="col-span-2" onClick={onMerge} disabled={busy}>
              Merge into {p.best_match!.name}
            </Button>
          ) : null}
        </div>
      </div>
    </li>
  );
}
