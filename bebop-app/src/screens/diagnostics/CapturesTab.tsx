// Diagnostics › Captures: every MCAP on the robot, with raw and Rerun
// downloads served by the firmware's HTTP server (:9090/captures).

import { useCallback, useEffect, useState } from "react";

import { Banner, Button, Card, CardHeader } from "../../components/ui";
import { useRobot } from "../../robot/session";

const LINK_BTN =
  "inline-flex h-control-sm items-center justify-center rounded-input border border-border bg-bg-elev-2 px-3 text-label font-semibold text-text no-underline transition-colors hover:border-text-dim/40 hover:bg-bg-elev pointer-coarse:h-touch";

export function CapturesTab() {
  return <CaptureDownloads />;
}

// --------------------------------------------------------------------------
// Capture downloads
// --------------------------------------------------------------------------

/// One row from the firmware's `GET /captures` endpoint. Field names
/// mirror `CaptureEntry` in `firmware/bebop-linux/src/server/ws.rs`.
interface CaptureFile {
  name: string;
  sizeBytes: number;
  modifiedMs: number;
  /// True for the segment a firmware writer currently has open. The app
  /// downloads it through `/captures/finalize-dl/<name>`, which rolls it
  /// over first so the saved file has an MCAP footer.
  active?: boolean;
}

interface CapturesResponse {
  files: CaptureFile[];
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

function formatModified(ms: number): string {
  if (!ms) return "—";
  const d = new Date(ms);
  return d.toLocaleString();
}

/// Lists every `.mcap` capture on the robot (drive sessions and the
/// always-on `system_*` health log) and lets the operator download any
/// of them. The health log rolls daily and is hidden by default so the
/// drive sessions stay findable. Polls the JSON endpoint on mount + on a
/// manual refresh; downloads are plain links to the same HTTP server
/// (`/captures/dl/<name>`; the live segment goes through `finalize-dl`),
/// so the browser streams them straight to disk (see `downloadUrl`).
function CaptureDownloads() {
  const { endpoints } = useRobot();
  const [files, setFiles] = useState<CaptureFile[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /// Page size for the segments table. The list is newest-first and a
  /// robot can accumulate dozens of segments over a long session, so we
  /// cap the visible rows to keep the card from growing unbounded and
  /// pushing the rest of the dashboard off-screen.
  const PAGE_SIZE = 8;
  const [page, setPage] = useState(0);
  /// The firmware's always-on health log (`system_*`, one file per day)
  /// is not drive data; hide it unless asked so the list is the drives.
  const [showSystem, setShowSystem] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(endpoints.captures(), { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const body = (await res.json()) as CapturesResponse;
      setFiles(body.files ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setFiles([]);
    } finally {
      setLoading(false);
    }
  }, [endpoints]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  /// Clamp the page back into range whenever the file list shrinks
  /// (e.g. after a refresh that returns fewer segments, or after the
  /// operator deleted captures out of band). Without this the user
  /// could be stranded on a page that has no rows.
  const isSystemLog = (name: string) => name.startsWith("system_");
  const visibleFiles = files
    ? showSystem
      ? files
      : files.filter((f) => !isSystemLog(f.name))
    : [];
  const hiddenCount = files ? files.length - visibleFiles.length : 0;
  const pageCount = Math.max(1, Math.ceil(visibleFiles.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  if (safePage !== page) setPage(safePage);
  const pageStart = safePage * PAGE_SIZE;
  const pageFiles = visibleFiles.slice(pageStart, pageStart + PAGE_SIZE);

  /// Build the direct download URL for a segment. We deliberately do
  /// NOT fetch + create a Blob for the download:
  ///
  ///  - A blob URL on an HTTP page triggers Chrome's "loaded over an
  ///    insecure connection" console warning and is blocked outright
  ///    in some configurations.
  ///  - `fetch()` participates in the browser's HTTP cache, so a
  ///    second click on the same file returns 304 Not Modified with
  ///    no body and the saved file ends up empty.
  ///  - Loading a multi-hundred-MiB MCAP fully into memory before
  ///    handing it to the download saver is wasteful when the
  ///    browser can stream it straight to disk.
  ///
  /// A plain `<a href={url} download={name}>` sidesteps all three:
  /// the browser issues its own GET, follows redirects, supports
  /// range requests for resume, and writes the bytes directly to
  /// the destination file.
  /// Raw download URL. A real, copyable link — paste it straight into
  /// Foxglove's "remote file" (or any MCAP tool).
  const downloadUrl = useCallback(
    (name: string) => endpoints.captures(`/dl/${encodeURIComponent(name)}`),
    [endpoints],
  );

  /// Finalizing download: same file, but the server rolls the segment over
  /// first (if it's the live one) so it has a footer. Copyable too.
  const finalizeDlUrl = useCallback(
    (name: string) => endpoints.captures(`/finalize-dl/${encodeURIComponent(name)}`),
    [endpoints],
  );

  /// On-robot MCAP -> Rerun .rrd conversion. The URL ends in `.rrd` so
  /// Rerun's loader picks the Rerun format; the firmware converts on demand
  /// and streams the bytes. Paste it into Rerun's "open URL", or click to
  /// download it.
  const rerunUrl = useCallback(
    (name: string) =>
      endpoints.captures(`/rrd/${encodeURIComponent(name.replace(/\.mcap$/, ".rrd"))}`),
    [endpoints],
  );

  return (
    <Card className="space-y-3">
      <CardHeader
        eyebrow="Capture downloads"
        title="MCAP segments on robot"
        description={
          <>
            Every capture on the robot, newest first. <b>Download</b> saves
            the raw <code className="text-text">.mcap</code> (the
            currently-writing segment is finalized first, so the file is
            complete). <b>Rerun</b> converts it on the robot to a{" "}
            <code className="text-text">.rrd</code> that opens with the navd
            dashboard.
          </>
        }
        actions={
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void refresh()}
            loading={loading}
            disabled={loading}
          >
            Refresh
          </Button>
        }
      />

      {error ? (
        <Banner tone="error">Capture list error: {error}</Banner>
      ) : null}

      {files !== null ? (
        <label className="flex items-center gap-2 text-label text-text-dim">
          <input
            type="checkbox"
            checked={showSystem}
            onChange={(e) => {
              setShowSystem(e.target.checked);
              setPage(0);
            }}
          />
          Show system logs
          {hiddenCount > 0 ? (
            <span className="tabular-nums">({hiddenCount} hidden)</span>
          ) : null}
        </label>
      ) : null}

      {files === null ? (
        <div className="text-label text-text-dim italic">Loading…</div>
      ) : files.length === 0 ? (
        <div className="text-label text-text-dim italic">
          No captures on robot yet. They appear as soon as the wheels are
          armed in drive mode.
        </div>
      ) : visibleFiles.length === 0 ? (
        <div className="text-label text-text-dim italic">
          No drive sessions yet; only system logs are on the robot (tick
          &ldquo;Show system logs&rdquo; to list them).
        </div>
      ) : (
        <div className="space-y-2">
          <ul className="m-0 flex list-none flex-col p-0">
            {pageFiles.map((f) => (
              <li
                key={f.name}
                className="flex flex-col gap-2 border-b border-border/40 py-2.5 last:border-b-0 sm:flex-row sm:items-center sm:gap-3"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-mono text-label text-text" title={f.name}>
                      {f.name}
                    </span>
                    {f.active ? (
                      <span className="shrink-0 rounded-full border border-success/40 bg-success/15 px-1.5 text-micro font-semibold uppercase text-success">
                        live
                      </span>
                    ) : null}
                  </div>
                  <div className="text-caption tabular-nums text-text-dim">
                    {formatBytes(f.sizeBytes)} · {formatModified(f.modifiedMs)}
                  </div>
                </div>
                {/* Plain copyable links: copy the URL into Foxglove (raw MCAP)
                    or Rerun (.rrd). The live segment uses the finalizing URL
                    so the saved file has a footer. */}
                <div className="grid shrink-0 grid-cols-2 gap-2 sm:flex">
                  <a
                    href={f.active ? finalizeDlUrl(f.name) : downloadUrl(f.name)}
                    download={f.name}
                    className={LINK_BTN}
                  >
                    {f.active ? "Finalize + download" : "Download"}
                  </a>
                  <a
                    href={rerunUrl(f.name)}
                    download={f.name.replace(/\.mcap$/, ".rrd")}
                    className={LINK_BTN}
                  >
                    Rerun .rrd
                  </a>
                </div>
              </li>
            ))}
          </ul>

          {/* Pagination row: only shown when there's more than one
              page. Prev/Next are disabled at the ends; the indicator
              uses 1-based numbering for human readability. */}
          {pageCount > 1 ? (
            <div className="flex items-center justify-between gap-2 pt-1">
              <span className="text-caption text-text-dim tabular-nums">
                Showing {pageStart + 1}–{Math.min(pageStart + PAGE_SIZE, visibleFiles.length)} of {visibleFiles.length}
              </span>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={safePage === 0}
                  className={`${LINK_BTN} cursor-pointer disabled:cursor-not-allowed disabled:opacity-40`}
                >
                  Prev
                </button>
                <span className="text-caption text-text-dim tabular-nums px-1">
                  {safePage + 1} / {pageCount}
                </span>
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                  disabled={safePage >= pageCount - 1}
                  className={`${LINK_BTN} cursor-pointer disabled:cursor-not-allowed disabled:opacity-40`}
                >
                  Next
                </button>
              </div>
            </div>
          ) : null}
        </div>
      )}
    </Card>
  );
}
