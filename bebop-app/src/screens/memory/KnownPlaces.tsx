// Confirmed places: teach the current view, the tile list, and the detail
// panel (views, facts, forget) for the selected place.

import { useState } from "react";

import { Button, Card, CardHeader, cn, Field, INPUT_CLASS, StatusPill } from "../../components/ui";
import { keyframeUrl, placeTitle, plural, shortTime } from "./places";
import type { Place, PlacesUrl } from "./places";

export function TeachCard({
  busy,
  onTeach,
}: {
  busy: boolean;
  onTeach: (name: string) => Promise<boolean>;
}) {
  const [name, setName] = useState("");
  const submit = async () => {
    if (name.trim() && (await onTeach(name.trim()))) setName("");
  };
  return (
    <Card className="flex flex-col gap-3">
      <CardHeader
        eyebrow="Teach"
        title="Name the room the robot sees now"
        description="Uses the current camera views. Already taught? Open the place and add views instead."
      />
      <form
        className="flex flex-col gap-2 sm:flex-row sm:items-end"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <div className="min-w-0 flex-1">
          <Field label="Room name">
            <input
              className={INPUT_CLASS}
              value={name}
              onChange={(e) => setName(e.currentTarget.value)}
              placeholder="kitchen"
              enterKeyHint="done"
            />
          </Field>
        </div>
        <Button type="submit" loading={busy} disabled={!name.trim()}>
          Teach this view
        </Button>
      </form>
    </Card>
  );
}

export function PlaceTiles({
  api,
  places,
  selected,
  hereId,
  onSelect,
}: {
  api: PlacesUrl;
  places: Place[];
  selected: string | null;
  hereId: string | null;
  onSelect: (id: string | null) => void;
}) {
  return (
    <ul className="m-0 grid list-none grid-cols-2 gap-2 p-0 sm:grid-cols-3">
      {places.map((p) => {
        const kf = p.keyframes[0];
        const active = p.id === selected;
        return (
          <li key={p.id} className="min-w-0">
            <button
              type="button"
              onClick={() => onSelect(active ? null : p.id)}
              aria-pressed={active}
              className={cn(
                "flex w-full cursor-pointer flex-col overflow-hidden rounded-card border text-left transition-colors",
                active ? "border-accent ring-1 ring-accent/50" : "border-border hover:border-text-dim/50",
              )}
            >
              <div className="relative aspect-video w-full bg-bg-elev-2">
                {kf ? (
                  <img
                    src={keyframeUrl(api, p.id, kf.path)}
                    alt=""
                    loading="lazy"
                    className="h-full w-full object-cover"
                  />
                ) : null}
                {p.id === hereId ? (
                  <StatusPill tone="ok" dot className="absolute left-1.5 top-1.5 bg-bg/85">
                    here
                  </StatusPill>
                ) : null}
              </div>
              <div className="min-w-0 bg-bg-elev px-2.5 py-2">
                <div className="truncate text-sm font-semibold">{placeTitle(p)}</div>
                <div className="truncate text-caption text-text-dim">
                  {plural(p.keyframes.length, "view")} · {plural(p.facts.length, "fact")}
                </div>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

const FACTS_SHOWN = 8;

export function PlaceDetail({
  api,
  place,
  busy,
  onCapture,
  onAddFact,
  onForget,
  onClose,
}: {
  api: PlacesUrl;
  place: Place;
  busy: boolean;
  onCapture: (seconds: number) => void;
  onAddFact: (text: string) => Promise<boolean>;
  onForget: () => void;
  onClose: () => void;
}) {
  const [fact, setFact] = useState("");
  const [allFacts, setAllFacts] = useState(false);
  // Newest facts first; the long tail stays collapsed (some places hold
  // thousands of vision facts).
  const facts = [...place.facts].reverse();
  const shown = allFacts ? facts.slice(0, 200) : facts.slice(0, FACTS_SHOWN);
  return (
    <Card className="flex flex-col gap-3">
      <CardHeader
        eyebrow="Place"
        title={<span className="text-base">{placeTitle(place)}</span>}
        description={`${plural(place.keyframes.length, "view")} · last seen ${shortTime(place.last_seen_at)}${
          place.tags.length ? ` · ${place.tags.map((t) => t.label).join(", ")}` : ""
        }`}
        actions={
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close place details">
            Close
          </Button>
        }
      />
      {place.keyframes.length ? (
        <div className="-mx-4 flex snap-x gap-2 overflow-x-auto px-4 pb-1">
          {place.keyframes.map((kf) => (
            <figure key={kf.path} className="m-0 shrink-0 snap-start">
              <img
                src={keyframeUrl(api, place.id, kf.path)}
                alt={`${kf.role} view`}
                loading="lazy"
                className="h-28 w-40 rounded-input border border-border object-cover"
              />
              <figcaption className="mt-0.5 text-caption text-text-dim">{kf.role}</figcaption>
            </figure>
          ))}
        </div>
      ) : (
        <p className="text-sm text-text-dim">No views stored.</p>
      )}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <Button variant="secondary" loading={busy} onClick={() => onCapture(0)}>
          Add current view
        </Button>
        <Button variant="secondary" loading={busy} onClick={() => onCapture(10)}>
          Capture 10 s while turning
        </Button>
      </div>

      <div className="flex flex-col gap-1.5">
        <div className="text-caption font-semibold uppercase tracking-wider text-text-dim">
          Facts ({place.facts.length})
        </div>
        {shown.length ? (
          <ul className="m-0 flex list-none flex-col gap-1 p-0">
            {shown.map((f, i) => (
              <li key={`${f.created_at}-${i}`} className="text-body-sm">
                {f.text} <span className="text-caption text-text-dim">· {f.source}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-text-dim">No facts yet.</p>
        )}
        {facts.length > FACTS_SHOWN ? (
          <Button variant="ghost" size="sm" className="self-start" onClick={() => setAllFacts((v) => !v)}>
            {allFacts ? "Show fewer" : `Show ${Math.min(200, facts.length) - FACTS_SHOWN} more`}
          </Button>
        ) : null}
        <form
          className="flex gap-2"
          onSubmit={async (e) => {
            e.preventDefault();
            if (fact.trim() && (await onAddFact(fact.trim()))) setFact("");
          }}
        >
          <input
            className={INPUT_CLASS}
            value={fact}
            onChange={(e) => setFact(e.currentTarget.value)}
            placeholder="Add a fact (has a red mug)"
            aria-label="New fact"
          />
          <Button type="submit" variant="secondary" disabled={!fact.trim() || busy}>
            Add
          </Button>
        </form>
      </div>
      <div className="border-t border-border pt-3">
        <Button variant="ghost" size="sm" className="border border-danger/40 text-danger-fg" onClick={onForget}>
          Forget this place
        </Button>
      </div>
    </Card>
  );
}
