import { Card, StatusPill, Switch } from "../../components/ui";
import { plural } from "./places";
import type { WhereAmI } from "./places";

/// Live recognition (`/whereami` from the shared poller).
export function WhereAmICard({
  where,
  live,
  onLive,
  pendingCount,
}: {
  where: WhereAmI | null;
  live: boolean;
  onLive: (next: boolean) => void;
  pendingCount: number;
}) {
  const matched = !!(where?.matched && where.place);
  const others = (where?.top ?? []).filter((t) => t.id !== where?.place?.id).slice(0, 3);
  return (
    <Card className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <div className="text-caption font-semibold uppercase tracking-wider text-text-dim">
          Where am I
        </div>
        <label className="flex items-center gap-1 text-label text-text-dim">
          Live
          <Switch checked={live} onChange={onLive} label="Live recognition" />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className={matched ? "text-xl font-semibold" : "text-xl font-semibold text-text-dim"}>
          {matched ? where!.place!.name : where ? "Unknown place" : "—"}
        </span>
        {where?.score !== undefined ? (
          <StatusPill tone={matched ? "ok" : "dim"}>score {where.score.toFixed(2)}</StatusPill>
        ) : null}
        {!live ? <StatusPill tone="dim">paused</StatusPill> : null}
      </div>
      {!matched && where?.reason ? (
        <div className="text-body-sm text-text-dim">{where.reason}</div>
      ) : null}
      {others.length > 0 ? (
        <div className="text-label text-text-dim">
          Also close:{" "}
          {others.map((t, i) => (
            <span key={t.id}>
              {i ? " · " : ""}
              <span className="text-text">{t.name}</span>{" "}
              <span className="font-mono">{t.score.toFixed(2)}</span>
            </span>
          ))}
          {where?.margin !== undefined ? (
            <span className="font-mono">
              {" "}
              · margin {where.margin.toFixed(2)} (min {(where.min_score ?? 0).toFixed(2)})
            </span>
          ) : null}
        </div>
      ) : null}
      {pendingCount > 0 ? (
        <div className="text-label text-text-dim">
          {plural(pendingCount, "pending place")} below aren&apos;t recognized until imported.
        </div>
      ) : null}
    </Card>
  );
}
