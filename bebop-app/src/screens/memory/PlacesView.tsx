// Memory › Places: recognition, the pending-place review queue, teaching and
// the confirmed place list. Everything reads the session's shared places
// poller (`usePlaces`: /places, /places/pending, /whereami); writes go
// straight to the places service and then refresh that poller.

import { useCallback, useEffect, useRef, useState } from "react";

import { Banner, Button, PageState, useConfirm } from "../../components/ui";
import { PORTS } from "../../robot/endpoints";
import { usePlaces, useRobot } from "../../robot/session";
import { PendingReview } from "./PendingReview";
import type { Notify } from "./PendingReview";
import { PlaceDetail, PlaceTiles, TeachCard } from "./KnownPlaces";
import { addFact, captureViews, deletePlace, placeTitle, plural, teachPlace } from "./places";
import type { Place, WhereAmI } from "./places";
import { WhereAmICard } from "./WhereAmICard";

export function PlacesView() {
  const { endpoints } = useRobot();
  const api = endpoints.places;
  const [live, setLive] = useState(true);
  const shared = usePlaces(true, { pending: true });
  const places = shared.places as unknown as Place[];
  const pending = shared.pending as unknown as Place[];
  // "Live" off freezes the recognition readout (the lists keep updating).
  const frozenWhere = useRef<WhereAmI | null>(null);
  if (live) frozenWhere.current = shared.whereami as WhereAmI | null;
  const where = frozenWhere.current;
  const { refresh } = shared;
  const { confirm, dialog } = useConfirm();

  const [notice, setNotice] = useState<{ tone: "success" | "error" | "info"; text: string } | null>(null);
  const notify: Notify = useCallback((tone, text) => setNotice({ tone, text }), []);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const detailRef = useRef<HTMLDivElement | null>(null);

  const detail = places.find((p) => p.id === selected) ?? null;
  useEffect(() => {
    if (selected && !places.some((p) => p.id === selected) && places.length) setSelected(null);
  }, [places, selected]);

  async function act(fn: () => Promise<string | void>): Promise<boolean> {
    setBusy(true);
    try {
      const msg = await fn();
      if (msg) notify("success", msg);
      return true;
    } catch (e) {
      notify("error", e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(false);
      await refresh();
    }
  }

  const select = (id: string | null) => {
    setSelected(id);
    if (id) {
      // Phones stack the detail under the list; bring it into view.
      requestAnimationFrame(() =>
        detailRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
      );
    }
  };

  const firstLoad = shared.updatedAt === null && shared.reachable !== false;
  if (firstLoad) return <PageState kind="loading" title="Loading place memory…" />;

  return (
    <div className="flex flex-col gap-5">
      {dialog}
      {shared.reachable === false ? (
        <Banner
          tone="error"
          title={`Places service unreachable (:${PORTS.places})`}
          action={
            <Button variant="secondary" size="sm" onClick={() => void refresh()}>
              Retry
            </Button>
          }
        >
          {shared.error}. Place memory runs inside bebop-vision; start vision from
          Drive or Diagnostics › Services.
        </Banner>
      ) : null}
      {notice ? (
        <Banner
          tone={notice.tone}
          action={
            <Button variant="ghost" size="sm" onClick={() => setNotice(null)}>
              Dismiss
            </Button>
          }
        >
          {notice.text}
        </Banner>
      ) : null}

      <WhereAmICard where={where} live={live} onLive={setLive} pendingCount={pending.length} />

      <PendingReview api={api} pending={pending} refresh={refresh} confirm={confirm} notify={notify} />

      <section className="flex flex-col gap-3" aria-label="Known places">
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="text-base font-semibold">
            Known places{" "}
            <span className="text-sm font-normal text-text-dim">{places.length}</span>
          </h2>
          <Button variant="ghost" size="sm" onClick={() => void refresh()} loading={shared.loading}>
            Refresh
          </Button>
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] lg:items-start">
          <div className="flex min-w-0 flex-col gap-3">
            {places.length ? (
              <PlaceTiles
                api={api}
                places={places}
                selected={selected}
                hereId={where?.matched ? (where.place?.id ?? null) : null}
                onSelect={select}
              />
            ) : (
              <p className="text-sm text-text-dim">
                No places yet. Drive somewhere, then teach the view below.
              </p>
            )}
            <TeachCard
              busy={busy}
              onTeach={(name) =>
                act(async () => {
                  const r = await teachPlace(api, name);
                  setSelected(r.place.id);
                  return `Taught "${r.place.name}" from ${plural(r.views, "view")}.`;
                })
              }
            />
          </div>
          <div ref={detailRef} className="min-w-0 scroll-mt-4 lg:sticky lg:top-4">
            {detail ? (
              <PlaceDetail
                key={detail.id}
                api={api}
                place={detail}
                busy={busy}
                onClose={() => setSelected(null)}
                onCapture={(seconds) =>
                  void act(async () => {
                    const r = await captureViews(api, detail.id, seconds);
                    return r.keyframes === 0
                      ? "No new view (the camera image hasn't changed). Move the robot and retry."
                      : `Captured ${plural(r.keyframes, "new view")}; ${r.views_total} in total.`;
                  })
                }
                onAddFact={(text) => act(() => addFact(api, detail.id, text).then(() => undefined))}
                onForget={async () => {
                  const ok = await confirm({
                    title: `Forget "${placeTitle(detail)}"?`,
                    body: "Its views, descriptors and facts are deleted. This can't be undone.",
                    confirmLabel: "Forget",
                    tone: "danger",
                  });
                  if (!ok) return;
                  await act(async () => {
                    await deletePlace(api, detail.id);
                    setSelected(null);
                    return `Forgot "${placeTitle(detail)}".`;
                  });
                }}
              />
            ) : places.length ? (
              <p className="hidden rounded-card border border-dashed border-border px-4 py-8 text-center text-sm text-text-dim lg:block">
                Select a place to see its views and facts.
              </p>
            ) : null}
          </div>
        </div>
      </section>
    </div>
  );
}
