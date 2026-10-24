import { useMemo, useState } from "react";

import { useRuntime, useTelemetry } from "../robot/session";
import type { ModelEntryView, ModelView } from "../runtime";
import { Banner, Button, Card, Field } from "./ui";

const EMPTY_MODEL: ModelView = {
  tokenSet: false,
  present: false,
  diskFreeBytes: 0,
  detail: "",
  models: [],
  selection: [],
};

/// Human-readable byte size (e.g. "3.3 GB").
function fmtBytes(n: number): string {
  if (!n || n <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function titleCase(s: string): string {
  if (!s) return "Other";
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/// Model provisioning + per-purpose selection. Shows the robot's catalog
/// grouped by purpose, lets the operator store a Hugging Face token, start
/// downloads, and choose which model serves each purpose. Downloads run on
/// the robot (firmware → systemd) and report progress over telemetry.
///
/// Renders nothing when `model.present === false` unless `showWhenUnavailable`
/// is set (the dedicated Models screen uses that to explain the gap).
export function AIModelCard({
  showWhenUnavailable = false,
}: {
  showWhenUnavailable?: boolean;
}) {
  const transport = useRuntime();
  // The catalog changes slowly; 2 Hz is plenty for download progress.
  const model = useTelemetry((s) => s?.model ?? EMPTY_MODEL, 2);
  const [tokenDraft, setTokenDraft] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);


  const groups = useMemo(() => {
    const map = new Map<string, ModelEntryView[]>();
    for (const entry of model.models) {
      const key = entry.purpose || "other";
      const list = map.get(key);
      if (list) list.push(entry);
      else map.set(key, [entry]);
    }
    return [...map.entries()];
  }, [model.models]);

  if (!model.present) {
    return showWhenUnavailable ? (
      <Card>
        <div className="py-2 flex flex-col gap-2">
          <div className="text-xs text-text-dim uppercase tracking-wider">
            AI models
          </div>
          <Banner tone="info">
            Model provisioning isn&apos;t installed on this robot yet. Deploy the
            current firmware and the <code>bebop-model-download@.service</code> unit
            (see <code>scripts/install-jetson.sh</code>), then reconnect.
          </Banner>
        </div>
      </Card>
    ) : null;
  }

  const downloading = model.models.some((m) => m.state === "downloading");
  const selectionFor = (purpose: string) =>
    model.selection.find((s) => s.purpose === purpose)?.modelId ?? "";

  async function run(key: string, fn: () => Promise<void>) {
    setError(null);
    setBusy(key);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {error ? <Banner tone="error">{error}</Banner> : null}
      {model.detail ? <Banner tone="error">{model.detail}</Banner> : null}

      {/* Hugging Face token */}
      <Card>
        <div className="py-2 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="text-xs text-text-dim uppercase tracking-wider">
              Hugging Face token
            </div>
            {model.diskFreeBytes > 0 ? (
              <div className="text-label text-text-dim">
                {fmtBytes(model.diskFreeBytes)} free
              </div>
            ) : null}
          </div>
          {model.tokenSet ? (
            <div className="flex items-center justify-between gap-3">
              <div className="text-body-sm text-text-dim">Token saved.</div>
              <Button
                variant="ghost"
                loading={busy === "clear-token"}
                onClick={() => run("clear-token", () => transport.clearHfToken())}
              >
                Clear
              </Button>
            </div>
          ) : (
            <>
              <Field
                label="Access token"
                hint="Required for gated models. The token's account must have accepted each model's license."
              >
                <input
                  type="password"
                  autoComplete="off"
                  spellCheck={false}
                  value={tokenDraft}
                  onChange={(e) => setTokenDraft(e.currentTarget.value)}
                  placeholder="hf_…"
                  className="w-full bg-bg-elev-2 border border-border rounded-card px-3 py-2.5 text-text outline-none focus:border-accent"
                />
              </Field>
              <Button
                variant="secondary"
                loading={busy === "save-token"}
                disabled={tokenDraft.trim().length < 8}
                onClick={() =>
                  run("save-token", async () => {
                    await transport.setHfToken(tokenDraft.trim());
                    setTokenDraft("");
                  })
                }
              >
                Save token
              </Button>
            </>
          )}
        </div>
      </Card>

      {groups.length === 0 ? (
        <Card>
          <p className="py-2 text-sm text-text-dim">
            No models in the catalog on this robot.
          </p>
        </Card>
      ) : (
        groups.map(([purpose, entries]) => {
          const selected = selectionFor(purpose);
          return (
            <Card key={purpose}>
              <div className="py-2 flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-text-dim uppercase tracking-wider">
                    {titleCase(purpose)}
                  </div>
                  {selected ? (
                    <button
                      type="button"
                      className="inline-flex h-control-sm items-center rounded-input px-2 text-label text-text-dim hover:text-text pointer-coarse:h-touch"
                      onClick={() =>
                        run(`purpose:${purpose}`, () =>
                          transport.setModelPurpose(purpose, ""),
                        )
                      }
                    >
                      Clear
                    </button>
                  ) : null}
                </div>
                <div className="flex flex-col divide-y divide-border">
                  {entries.map((entry) => (
                    <ModelRow
                      key={entry.id}
                      entry={entry}
                      selected={selected === entry.id}
                      tokenSet={model.tokenSet}
                      busy={busy === `download:${entry.id}`}
                      anyDownloading={downloading}
                      onSelect={() =>
                        run(`purpose:${purpose}`, () =>
                          transport.setModelPurpose(purpose, entry.id),
                        )
                      }
                      onDownload={() =>
                        run(`download:${entry.id}`, () =>
                          transport.downloadModel(entry.id),
                        )
                      }
                    />
                  ))}
                </div>
              </div>
            </Card>
          );
        })
      )}
    </div>
  );
}

function StatusPill({ entry }: { entry: ModelEntryView }) {
  if (entry.ready) {
    return <span className="text-label font-semibold text-success">Ready</span>;
  }
  if (entry.kind !== "hf") {
    return <span className="text-label text-text-dim">Not present</span>;
  }
  switch (entry.state) {
    case "downloading":
      return <span className="text-label font-semibold text-accent">Downloading…</span>;
    case "unauthorized":
      return <span className="text-label font-semibold text-danger">Token rejected</span>;
    case "failed":
      return <span className="text-label font-semibold text-danger">Failed</span>;
    default:
      return <span className="text-label text-text-dim">Not downloaded</span>;
  }
}

function ModelRow({
  entry,
  selected,
  tokenSet,
  busy,
  anyDownloading,
  onSelect,
  onDownload,
}: {
  entry: ModelEntryView;
  selected: boolean;
  tokenSet: boolean;
  busy: boolean;
  anyDownloading: boolean;
  onSelect: () => void;
  onDownload: () => void;
}) {
  const pct =
    entry.bytesTotal > 0
      ? Math.min(100, Math.round((entry.bytesDownloaded / entry.bytesTotal) * 100))
      : 0;
  const downloadable = entry.kind === "hf" && !entry.ready;
  const blockedByToken = entry.gated && !tokenSet;
  const isDownloading = entry.state === "downloading";

  return (
    <div className="py-3 flex gap-3">
      <button
        type="button"
        role="radio"
        aria-checked={selected}
        aria-label={`Use ${entry.name || entry.id}`}
        onClick={onSelect}
        className="group -m-2 flex h-control-sm w-10 shrink-0 items-start justify-center pt-3 pointer-coarse:h-touch"
      >
        <span
          aria-hidden
          className={`h-4 w-4 rounded-full border-2 transition-colors ${
            selected ? "border-accent bg-accent" : "border-border group-hover:border-accent"
          }`}
        />
      </button>
      <div className="min-w-0 flex-1 flex flex-col gap-1.5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="font-semibold leading-tight">
              {entry.name || entry.id}
            </div>
            <div className="text-label text-text-dim leading-snug">
              {entry.description || entry.repo || entry.id}
            </div>
          </div>
          <div className="shrink-0 pt-0.5">
            <StatusPill entry={entry} />
          </div>
        </div>

        {isDownloading ? (
          <div className="flex flex-col gap-1">
            <div className="h-1.5 w-full rounded-full bg-bg-elev-2 overflow-hidden">
              <div
                className={`h-full bg-accent transition-[width] duration-300 ${
                  pct === 0 ? "animate-pulse" : ""
                }`}
                style={{ width: pct === 0 ? "20%" : `${pct}%` }}
              />
            </div>
            <div className="text-caption text-text-dim">
              {fmtBytes(entry.bytesDownloaded)}
              {entry.bytesTotal > 0 ? ` / ${fmtBytes(entry.bytesTotal)}` : ""}
            </div>
          </div>
        ) : null}

        {!isDownloading &&
        entry.detail &&
        (entry.state === "failed" || entry.state === "unauthorized") ? (
          <div className="text-caption text-danger leading-snug">{entry.detail}</div>
        ) : null}

        {downloadable ? (
          <div className="pt-0.5">
            <Button
              variant="secondary"
              loading={busy}
              disabled={blockedByToken || (anyDownloading && !isDownloading)}
              onClick={onDownload}
            >
              {blockedByToken ? "Add token to download" : "Download"}
            </Button>
          </div>
        ) : entry.kind !== "hf" && !entry.ready ? (
          <div className="text-caption text-text-dim leading-snug">
            Train on the robot or copy from a workstation.
          </div>
        ) : null}
      </div>
    </div>
  );
}
