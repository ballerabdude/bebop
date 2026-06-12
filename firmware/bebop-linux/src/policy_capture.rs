//! Off-the-hot-loop MCAP writer for operator-toggled policy captures.
//!
//! The 100 Hz control loop produces one
//! [`bebop_proto::capture::v1::PolicyCaptureSample`] per tick while a
//! capture is open. We deliberately do NOT touch the filesystem on the
//! tick thread — a single `write` syscall (and any periodic `flush`) is
//! easily a few hundred microseconds on the Jetson's eMMC and the tail
//! latencies are unbounded. Even with `BufWriter` and chunked MCAP
//! compression, allocator + compression + occasional flush hiccups can
//! eat enough of the 10 ms tick budget to make the control loop late.
//!
//! Architecture:
//!
//! ```text
//!   ┌───────────────────────┐  try_send(SAMPLE)   ┌────────────────────┐
//!   │ PolicyRunner (100 Hz) │ ──────────────────▶ │ writer thread      │
//!   │ tick thread           │                     │ owns mcap::Writer  │
//!   │ - builds sample       │                     │ - serializes proto │
//!   │ - request_open/close  │ ◀── reads status ── │ - chunked Zstd     │
//!   └───────────────────────┘  PolicyIoShared     │ - publishes status │
//!                                                 └────────────────────┘
//! ```
//!
//! Back-pressure: the channel is bounded. If the writer falls behind
//! (slow disk, a stall on the IO scheduler), [`CaptureHandle::send_sample`]
//! drops the sample and increments [`PolicyIoSnapshot::capture_dropped`]
//! so the operator sees the data loss in the UI. We deliberately drop
//! rather than block so a fault on the SD card can never wedge the
//! control loop.
//!
//! File layout: each `request_open` → `request_close` cycle yields ONE
//! or more MCAP files. The writer transparently rotates whenever the
//! active file passes [`MAX_FILE_BYTES`] so a single multi-hour session
//! never produces an unbounded file (long files are painful to scrub in
//! Foxglove and to copy off the robot). After each rotation we also
//! prune oldest files in the capture dir until the total on-disk size
//! is under [`DISK_BUDGET_BYTES`], so the robot can stay on for hours
//! at a time without filling the eMMC.
//!
//! Files are named `policy_capture_<YYYYmmdd_HHMMSS>.mcap`, with:
//!
//! - schema name  = `bebop.capture.v1.PolicyCaptureSample`
//! - schema enc.  = `protobuf` + serialized `FileDescriptorSet`
//! - channel topic = `/policy_capture`
//! - message enc. = `protobuf`
//! - per-message `log_time` = `publish_time` = wall-clock ns.

use std::fs;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use bebop_proto::capture::v1::PolicyCaptureSample;
use bebop_proto::Message;
use tracing::{debug, error, info, warn};

use crate::policy_io::PolicyIoShared;

/// MCAP topic the writer publishes samples on. Foxglove uses this as
/// the display label.
const CAPTURE_TOPIC: &str = "/policy_capture";

/// Schema name (matches the proto fully-qualified type name).
const CAPTURE_SCHEMA_NAME: &str = "bebop.capture.v1.PolicyCaptureSample";

/// MCAP well-known encoding strings.
const PROTOBUF_ENCODING: &str = "protobuf";

/// Sample channel capacity. At 100 Hz this gives the writer ~20 s of
/// slack to drain a backlog before we start dropping samples. Each
/// sample is on the order of a few hundred bytes serialized, so the
/// max channel memory is ~1 MB — negligible on the Jetson.
const SAMPLE_CHANNEL_CAPACITY: usize = 2000;

/// Periodic flush cadence in the writer thread's idle path. MCAP
/// chunks are written to disk every time the buffer fills (mcap-rs
/// handles that internally) OR when we call `flush` explicitly; the
/// timeout-driven flush below caps the worst-case data loss on a
/// power yank at ~1 s of samples regardless of message rate.
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Rotate the active capture file when its on-disk size reaches this
/// many bytes. 256 MiB keeps each file small enough that Foxglove can
/// load it in a few seconds and that scp / the web download path
/// completes quickly, while still being large enough that a single
/// rotation isn't a per-minute event on a continuous 100 Hz capture.
/// (At ~700 B/sample compressed, this is ~1 h of recording per file.)
const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// Total on-disk size budget for the capture directory. After each
/// rotation we delete the oldest `policy_capture_*.mcap` files (never
/// the currently open one) until the directory falls below this. 4 GiB
/// is comfortable headroom on a 64 GiB eMMC and covers many hours of
/// continuous capture before the oldest segments start being pruned.
const DISK_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// How often to check the active file's size for rotation. The Sample
/// branch fires at ~100 Hz; checking every Nth sample keeps `stat`
/// overhead negligible while still catching the size threshold within
/// ~1 s of crossing it. (Worst-case file overshoot at our sample
/// rate is therefore ~1 s × bytes/s ≪ MAX_FILE_BYTES.)
const ROTATE_CHECK_EVERY: u64 = 100;

/// Minimum interval between identical "open failed" `error!` lines.
/// Even with the runner's 5 s open-retry back-off, an operator
/// toggling modes (or any other path that issues Open commands) can
/// produce duplicate failures. Logging once per 30 s while the
/// underlying issue persists is enough to keep the operator aware
/// without flooding journald. Errors with a NEW message bypass the
/// throttle and log immediately.
const OPEN_ERROR_LOG_BACKOFF: Duration = Duration::from_secs(30);

/// Commands sent from the tick thread to the writer thread. Open /
/// close are explicit instead of being inferred from sample stream
/// state so the writer thread can ack each transition by publishing
/// to [`PolicyIoShared`].
enum CaptureCommand {
    /// Open a new capture session. The writer chooses the filename
    /// (`policy_capture_<timestamp>.mcap`) and publishes the resolved
    /// path to [`PolicyIoShared::set_capture_state`]. The writer may
    /// then rotate the file underneath the runner whenever the active
    /// segment exceeds [`MAX_FILE_BYTES`] — the runner does NOT need
    /// to know about rotation; from its perspective the capture stays
    /// "active" continuously.
    Open,
    /// One control-tick sample. Serialized inside the writer thread
    /// to keep tick CPU cost minimal.
    Sample(Box<PolicyCaptureSample>),
    /// Finish + close the current file. No-op if no file is open.
    Close,
    /// Drain + close and exit the writer thread. Used on shutdown.
    Shutdown,
}

/// Operator-facing handle on the capture writer thread. Cheap to clone
/// (it just clones two `Arc`s under the hood).
#[derive(Clone)]
pub struct CaptureHandle {
    tx: SyncSender<CaptureCommand>,
    /// Set by [`spawn_capture_thread`] when the thread exits cleanly
    /// (normal Shutdown). The handle treats `closed = true` as
    /// "nothing more to send"; subsequent `send_sample` calls become
    /// no-ops instead of panicking on a disconnected channel.
    closed: Arc<AtomicBool>,
    /// Cumulative count of samples the tick thread tried to send but
    /// the channel was full for. Surfaced through
    /// [`PolicyIoSnapshot::capture_dropped`] so the operator UI can
    /// flag data loss.
    dropped_total: Arc<AtomicU64>,
}

impl CaptureHandle {
    /// Try to enqueue a sample. Drops it (and bumps the dropped
    /// counter) if the writer is backlogged — never blocks the tick
    /// thread, even momentarily. Returns `true` if the sample was
    /// accepted.
    pub fn send_sample(&self, sample: PolicyCaptureSample) -> bool {
        if self.closed.load(Ordering::Relaxed) {
            return false;
        }
        // Boxing keeps the channel slot pointer-sized; the sample is
        // already heap-allocated for the repeated f32 vecs anyway.
        match self.tx.try_send(CaptureCommand::Sample(Box::new(sample))) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                self.dropped_total.fetch_add(1, Ordering::Relaxed);
                false
            }
            Err(TrySendError::Disconnected(_)) => {
                self.closed.store(true, Ordering::Relaxed);
                false
            }
        }
    }

    /// Request a fresh capture session. The writer opens the first
    /// file on its own thread and updates `PolicyIoShared` once the
    /// path is known. A repeat `request_open` while a session is
    /// already open is a no-op in the writer thread (the writer
    /// auto-rotates on size; the runner doesn't have to nudge it).
    pub fn request_open(&self) {
        let _ = self.send_command(CaptureCommand::Open);
    }

    /// Request the writer flush + close the active capture file (if
    /// any). Safe to call from the tick thread.
    pub fn request_close(&self) {
        let _ = self.send_command(CaptureCommand::Close);
    }

    /// Cumulative samples that were dropped because the channel was
    /// full. Monotonic — the operator UI compares against a remembered
    /// value to detect bursts during a session.
    pub fn dropped_total(&self) -> u64 {
        self.dropped_total.load(Ordering::Relaxed)
    }

    fn send_command(&self, cmd: CaptureCommand) -> bool {
        if self.closed.load(Ordering::Relaxed) {
            return false;
        }
        match self.tx.try_send(cmd) {
            Ok(()) => true,
            // Open/Close are rare control messages; if the channel is
            // full of samples, fall back to a small blocking send (≤
            // one writer iteration, which is a single `write` syscall)
            // rather than dropping the control message and confusing
            // operator state.
            Err(TrySendError::Full(cmd)) => match self.tx.send(cmd) {
                Ok(()) => true,
                Err(_) => {
                    self.closed.store(true, Ordering::Relaxed);
                    false
                }
            },
            Err(TrySendError::Disconnected(_)) => {
                self.closed.store(true, Ordering::Relaxed);
                false
            }
        }
    }

    /// Tell the writer to drain + exit. Intended for process shutdown.
    /// Idempotent.
    pub fn shutdown(&self) {
        if self.closed.swap(true, Ordering::Relaxed) {
            return;
        }
        let _ = self.tx.send(CaptureCommand::Shutdown);
    }
}

/// Spawn the capture writer thread. The returned handle is held by
/// [`crate::policy_runner::PolicyRunner`] for sample submission and
/// open/close requests; the join handle is returned so `main.rs` can
/// wait for the thread to drain on shutdown.
pub fn spawn_capture_thread(
    capture_dir: PathBuf,
    policy_io: PolicyIoShared,
) -> (CaptureHandle, JoinHandle<()>) {
    let (tx, rx) = mpsc::sync_channel::<CaptureCommand>(SAMPLE_CHANNEL_CAPACITY);
    let closed = Arc::new(AtomicBool::new(false));
    let dropped_total = Arc::new(AtomicU64::new(0));

    let handle = CaptureHandle {
        tx,
        closed: closed.clone(),
        dropped_total: dropped_total.clone(),
    };

    let join = std::thread::Builder::new()
        .name("policy-capture".to_string())
        .spawn(move || {
            run_capture_thread(rx, capture_dir, policy_io, dropped_total, closed);
        })
        .expect("spawn policy-capture thread");

    (handle, join)
}

/// Per-open file state. Lives entirely on the writer thread; the only
/// shared mutability is the status fields published into
/// [`PolicyIoShared`].
struct OpenCapture {
    path: PathBuf,
    writer: mcap::Writer<BufWriter<fs::File>>,
    channel_id: u16,
    rows: u64,
    started_at: Instant,
    last_flush: Instant,
    /// Cached encode buffer reused across samples to avoid per-tick
    /// `Vec<u8>` allocations.
    encode_buf: Vec<u8>,
}

impl OpenCapture {
    fn write_sample(&mut self, sample: &PolicyCaptureSample) -> Result<()> {
        self.encode_buf.clear();
        sample
            .encode(&mut self.encode_buf)
            .context("encode PolicyCaptureSample")?;
        let log_time = sample.wall_time_ns;
        self.writer
            .write_to_known_channel(
                &mcap::records::MessageHeader {
                    channel_id: self.channel_id,
                    sequence: self.rows as u32,
                    log_time,
                    publish_time: log_time,
                },
                &self.encode_buf,
            )
            .context("mcap: write_to_known_channel")?;
        self.rows += 1;
        Ok(())
    }

    fn flush_if_due(&mut self, now: Instant) -> Result<()> {
        if now.duration_since(self.last_flush) >= FLUSH_INTERVAL {
            self.writer.flush().context("mcap: periodic flush")?;
            self.last_flush = now;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.writer.finish().context("mcap: finish")?;
        Ok(())
    }
}

fn run_capture_thread(
    rx: Receiver<CaptureCommand>,
    capture_dir: PathBuf,
    policy_io: PolicyIoShared,
    dropped_total: Arc<AtomicU64>,
    closed: Arc<AtomicBool>,
) {
    debug!(dir = %capture_dir.display(), "capture: writer thread started");
    let mut open: Option<OpenCapture> = None;
    // Throttle the "open failed" log: when the underlying problem is
    // persistent (read-only fs, missing perms) and the runner keeps
    // re-requesting (e.g. operator re-toggles modes), we'd otherwise
    // produce one identical error line per attempt and drown the
    // journal. Log on the first failure, and again only when the error
    // text changes OR `OPEN_ERROR_LOG_BACKOFF` has elapsed.
    let mut last_open_error: Option<(Instant, String)> = None;

    loop {
        // `recv_timeout` lets us run a periodic flush even when the
        // tick thread is quiescent. The timeout matches the flush
        // cadence so worst case is two intervals between flushes.
        match rx.recv_timeout(FLUSH_INTERVAL) {
            Ok(CaptureCommand::Open) => {
                if open.is_some() {
                    debug!(
                        "capture: ignoring Open while a session is already open \
                         (the writer auto-rotates on size; no manual nudge needed)"
                    );
                    continue;
                }
                match open_capture(&capture_dir, "") {
                    Ok(c) => {
                        info!(path = %c.path.display(), "capture: opened (MCAP)");
                        publish_status(&policy_io, Some(&c), &dropped_total);
                        // First file of a session: opportunistically
                        // prune older segments left behind by prior
                        // runs so the disk budget is enforced even
                        // before the first rotation happens.
                        prune_capture_dir(&capture_dir, DISK_BUDGET_BYTES, &c.path);
                        open = Some(c);
                        // Clear the open-error throttle so the next
                        // failure (if any) logs immediately rather
                        // than being silently swallowed.
                        last_open_error = None;
                    }
                    Err(e) => {
                        let msg = format!("{e:#}");
                        let now = Instant::now();
                        let should_log = match last_open_error.as_ref() {
                            Some((t, prev)) => {
                                prev != &msg
                                    || now.duration_since(*t)
                                        >= OPEN_ERROR_LOG_BACKOFF
                            }
                            None => true,
                        };
                        if should_log {
                            error!(
                                dir = %capture_dir.display(),
                                error = %msg,
                                "capture: open failed (retrying every ~5 s in the runner)"
                            );
                            last_open_error = Some((now, msg));
                        }
                        publish_status(&policy_io, None, &dropped_total);
                    }
                }
            }
            Ok(CaptureCommand::Sample(sample)) => {
                if let Some(c) = open.as_mut() {
                    if let Err(e) = c.write_sample(&sample) {
                        error!(
                            path = %c.path.display(),
                            error = %format!("{e:#}"),
                            "capture: write failed; closing file"
                        );
                        // Try to flush + finish what we have so the
                        // operator gets a readable partial file.
                        let bad = open.take().expect("just matched");
                        let path = bad.path.clone();
                        if let Err(e) = bad.finish() {
                            warn!(path = %path.display(), error = %e, "capture: partial finish failed");
                        }
                        publish_status(&policy_io, None, &dropped_total);
                        continue;
                    }
                    // Periodic rotation check. The mcap writer chunks
                    // its output and flushes lazily, so we explicitly
                    // flush here before stat'ing to make the size
                    // observation accurate. Frequency is bounded by
                    // `ROTATE_CHECK_EVERY` so the syscall overhead
                    // stays in the noise even at 100 Hz.
                    let rows_now = c.rows;
                    if rows_now % ROTATE_CHECK_EVERY == 0 {
                        if let Err(e) = c.writer.flush() {
                            warn!(
                                path = %c.path.display(),
                                error = %e,
                                "capture: pre-rotate flush failed"
                            );
                        } else {
                            c.last_flush = Instant::now();
                        }
                        let size = fs::metadata(&c.path).map(|m| m.len()).unwrap_or(0);
                        if size >= MAX_FILE_BYTES {
                            let prev_path = c.path.clone();
                            let prev_rows = c.rows;
                            // Finish the current segment.
                            let prev = open.take().expect("just matched");
                            if let Err(e) = prev.finish() {
                                warn!(
                                    path = %prev_path.display(),
                                    error = %e,
                                    "capture: finish-on-rotate failed"
                                );
                            } else {
                                info!(
                                    path = %prev_path.display(),
                                    rows = prev_rows,
                                    size_bytes = size,
                                    "capture: rotated segment (size cap)"
                                );
                            }
                            // Open the next segment immediately so the
                            // tick thread keeps streaming without a gap.
                            match open_capture(&capture_dir, "") {
                                Ok(next) => {
                                    info!(
                                        path = %next.path.display(),
                                        "capture: opened next segment (MCAP)"
                                    );
                                    publish_status(&policy_io, Some(&next), &dropped_total);
                                    prune_capture_dir(
                                        &capture_dir,
                                        DISK_BUDGET_BYTES,
                                        &next.path,
                                    );
                                    open = Some(next);
                                }
                                Err(e) => {
                                    error!(
                                        dir = %capture_dir.display(),
                                        error = %format!("{e:#}"),
                                        "capture: open-next-segment failed; capture paused"
                                    );
                                    publish_status(&policy_io, None, &dropped_total);
                                }
                            }
                        }
                    }
                }
                // Else: a stale sample arrived after Close. Quietly
                // drop — easier than draining the channel inside Close,
                // and the tick thread stops sending almost immediately.
            }
            Ok(CaptureCommand::Close) => {
                if let Some(c) = open.take() {
                    let path = c.path.clone();
                    let rows = c.rows;
                    if let Err(e) = c.finish() {
                        warn!(path = %path.display(), error = %e, "capture: finish failed");
                    } else {
                        info!(
                            path = %path.display(),
                            rows,
                            "capture: closed (MCAP)"
                        );
                    }
                    publish_status(&policy_io, None, &dropped_total);
                }
            }
            Ok(CaptureCommand::Shutdown) => {
                if let Some(c) = open.take() {
                    let path = c.path.clone();
                    let rows = c.rows;
                    if let Err(e) = c.finish() {
                        warn!(path = %path.display(), error = %e, "capture: finish on shutdown failed");
                    } else {
                        info!(path = %path.display(), rows, "capture: closed on shutdown");
                    }
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(c) = open.as_mut() {
                    if let Err(e) = c.flush_if_due(Instant::now()) {
                        warn!(
                            path = %c.path.display(),
                            error = %e,
                            "capture: periodic flush failed"
                        );
                    }
                    // Refresh published status so `dropped` increments
                    // surface even during quiet periods.
                    publish_status(&policy_io, Some(c), &dropped_total);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // All senders dropped (shouldn't happen under normal
                // operation — main.rs holds the handle for the lifetime
                // of the process). Drain + exit.
                if let Some(c) = open.take() {
                    let path = c.path.clone();
                    let _ = c.finish();
                    warn!(path = %path.display(), "capture: senders dropped; closed file");
                }
                break;
            }
        }
    }

    closed.store(true, Ordering::Relaxed);
    publish_status(&policy_io, None, &dropped_total);
    debug!("capture: writer thread exiting");
}

fn open_capture(dir: &Path, label: &str) -> Result<OpenCapture> {
    fs::create_dir_all(dir).with_context(|| format!("create capture dir {}", dir.display()))?;
    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let sanitized = sanitize_label(label);
    let filename = if sanitized.is_empty() {
        format!("policy_capture_{stamp}.mcap")
    } else {
        format!("policy_capture_{sanitized}_{stamp}.mcap")
    };
    let path = dir.join(filename);
    let file = fs::File::create(&path)
        .with_context(|| format!("create capture file {}", path.display()))?;
    let buf = BufWriter::with_capacity(64 * 1024, file);
    let mut writer = build_writer(buf).context("mcap: build writer")?;

    let schema_id = writer
        .add_schema(
            CAPTURE_SCHEMA_NAME,
            PROTOBUF_ENCODING,
            bebop_proto::FILE_DESCRIPTOR_SET,
        )
        .context("mcap: add_schema")?;
    let channel_id = writer
        .add_channel(
            schema_id,
            CAPTURE_TOPIC,
            PROTOBUF_ENCODING,
            &std::collections::BTreeMap::new(),
        )
        .context("mcap: add_channel")?;

    Ok(OpenCapture {
        path,
        writer,
        channel_id,
        rows: 0,
        started_at: Instant::now(),
        last_flush: Instant::now(),
        encode_buf: Vec::with_capacity(1024),
    })
}

fn build_writer<W: Write + Seek>(w: W) -> mcap::McapResult<mcap::Writer<W>> {
    // Zstd compresses our smooth telemetry well (mostly slowly-varying
    // floats) and is fast enough on the Jetson at our data rate to
    // never block the writer thread meaningfully. Defaults otherwise.
    mcap::Writer::with_options(
        w,
        mcap::WriteOptions::new()
            .compression(Some(mcap::Compression::Zstd))
            .profile("policy_capture")
            .library(format!("bebop-linux mcap {}", mcap::VERSION)),
    )
}

fn publish_status(
    policy_io: &PolicyIoShared,
    open: Option<&OpenCapture>,
    dropped_total: &AtomicU64,
) {
    if let Ok(mut g) = policy_io.lock() {
        let dropped = dropped_total.load(Ordering::Relaxed);
        match open {
            Some(c) => {
                let path = c.path.to_string_lossy();
                g.set_capture_state(true, &path, c.rows, dropped);
            }
            None => g.set_capture_state(false, "", 0, dropped),
        }
    }
}

/// Enforce a total on-disk budget for the capture directory. Walks
/// the dir for `policy_capture_*.mcap` files, sums their sizes, and
/// deletes the oldest (by mtime) until the total falls below `budget`.
/// `keep_open` is never deleted regardless of age — that's the file
/// the writer thread is currently appending to.
///
/// Best-effort: any IO error here is logged at `warn!` but does NOT
/// halt the writer thread. The capture loop is the only consumer of
/// the disk budget, so a partial prune just means we'll try again on
/// the next rotation.
pub(crate) fn prune_capture_dir(dir: &Path, budget: u64, keep_open: &Path) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            warn!(
                dir = %dir.display(),
                error = %e,
                "capture: prune skipped (read_dir failed)"
            );
            return;
        }
    };

    /// One `policy_capture_*.mcap` segment on disk, captured for the
    /// oldest-first pruning sweep.
    struct Segment {
        path: PathBuf,
        size: u64,
        mtime: SystemTime,
    }

    let mut segments: Vec<Segment> = Vec::new();
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("policy_capture_") && n.ends_with(".mcap"))
        {
            continue;
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.is_file() {
            continue;
        }
        let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
        let size = meta.len();
        total += size;
        segments.push(Segment { path, size, mtime });
    }

    if total <= budget {
        return;
    }

    segments.sort_by_key(|s| s.mtime);

    let mut deleted_count: u32 = 0;
    let mut deleted_bytes: u64 = 0;
    for seg in segments {
        if total <= budget {
            break;
        }
        if seg.path == keep_open {
            continue;
        }
        match fs::remove_file(&seg.path) {
            Ok(()) => {
                total = total.saturating_sub(seg.size);
                deleted_bytes += seg.size;
                deleted_count += 1;
            }
            Err(e) => {
                warn!(
                    path = %seg.path.display(),
                    error = %e,
                    "capture: prune could not delete segment"
                );
            }
        }
    }

    if deleted_count > 0 {
        info!(
            dir = %dir.display(),
            deleted_count,
            deleted_bytes,
            remaining_bytes = total,
            budget_bytes = budget,
            "capture: pruned old segments to keep dir under disk budget"
        );
    }
}

/// Strip a label down to filesystem-safe characters. Accepts ASCII
/// alphanumerics plus `_` / `-`; everything else (including spaces and
/// `.`) becomes `_`. Truncated to 32 chars so the timestamp prefix
/// stays the dominant part of the filename. Production callers pass an
/// empty label (the writer auto-rotates rather than tagging files);
/// kept around so the round-trip test can exercise a labeled file.
fn sanitize_label(label: &str) -> String {
    let mut out = String::with_capacity(label.len().min(32));
    for c in label.chars().take(32) {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

/// Build a sample timestamp pair (wall_time_ns, sim_time_s) from an
/// `Instant` taken on the tick thread and the capture's start time.
/// Public so the runner can populate samples without re-deriving the
/// formula. `started_at` is `None` until the writer thread reports
/// a capture has actually opened — in that case we still produce a
/// non-zero `wall_time_ns` (the MCAP `log_time` field requires it)
/// and a `sim_time_s` of 0.
pub fn timestamps(now: Instant, started_at: Option<Instant>) -> (u64, f64) {
    let wall_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let sim_s = started_at
        .map(|t0| now.saturating_duration_since(t0).as_secs_f64())
        .unwrap_or(0.0);
    (wall_ns, sim_s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn sanitize_label_strips_unsafe_chars() {
        assert_eq!(sanitize_label("noise_floor"), "noise_floor");
        assert_eq!(sanitize_label("policy-v17"), "policy-v17");
        assert_eq!(sanitize_label("foo bar"), "foo_bar");
        assert_eq!(sanitize_label("/etc/passwd"), "etc_passwd");
        assert_eq!(sanitize_label(""), "");
        assert_eq!(sanitize_label("   "), "");
        assert!(sanitize_label(&"a".repeat(100)).len() <= 32);
    }

    /// MCAP round-trip: write a single sample through `OpenCapture`,
    /// finish, then read it back via `mcap::MessageStream` and confirm
    /// the schema name + topic + payload survive. This is the
    /// canonical guard against schema-emission drift (e.g. someone
    /// stops including the FileDescriptorSet in build.rs).
    #[test]
    fn mcap_write_then_read_round_trips_a_sample() {
        let dir = std::env::temp_dir().join(format!(
            "bebop_mcap_round_trip_{}",
            std::process::id()
        ));
        let _ = fs::create_dir_all(&dir);
        let mut cap = open_capture(&dir, "round_trip").expect("open capture");
        let path = cap.path.clone();

        let mut sample = PolicyCaptureSample {
            tick: 7,
            wall_time_ns: 1_700_000_000_000_000_000,
            sim_time_s: 0.42,
            mode: "RUN_POLICY".into(),
            dry_run: true,
            imu_live: true,
            quat_x: 0.0,
            quat_y: 0.0,
            quat_z: 0.0,
            quat_w: 1.0,
            ang_vel_x: 0.01,
            ang_vel_y: -0.02,
            ang_vel_z: 0.03,
            joint_pos_rad: vec![0.1; 8],
            joint_vel_rad_s: vec![0.0; 8],
            joint_armed: vec![true; 8],
            observation: (0..49).map(|i| i as f32 * 0.01).collect(),
            raw_action: (0..24).map(|i| i as f32 * 0.05).collect(),
            position_targets_rad: vec![0.0; 8],
            kp: vec![20.0; 8],
            kd: vec![1.0; 8],
        };
        cap.write_sample(&sample).expect("write sample");
        // Bump rows by hand here would be a test smell — the real
        // writer increments inside write_sample.
        sample.tick = 8;
        cap.write_sample(&sample).expect("write sample 2");
        cap.finish().expect("finish capture");

        // Read back.
        let bytes = fs::read(&path).expect("read capture file");
        let mut by_topic: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();
        let mut schema_names: Vec<String> = Vec::new();
        for msg in mcap::MessageStream::new(&bytes).expect("open stream") {
            let msg = msg.expect("decode message");
            let topic = msg.channel.topic.clone();
            if let Some(s) = msg.channel.schema.as_ref() {
                if !schema_names.contains(&s.name) {
                    schema_names.push(s.name.clone());
                }
            }
            by_topic.entry(topic).or_default().push(msg.data.to_vec());
        }
        assert_eq!(by_topic.len(), 1, "exactly one capture topic");
        assert_eq!(
            by_topic.keys().next().unwrap(),
            CAPTURE_TOPIC,
            "topic name drift"
        );
        assert_eq!(
            schema_names,
            vec![CAPTURE_SCHEMA_NAME.to_string()],
            "schema name drift"
        );
        let messages = by_topic.get(CAPTURE_TOPIC).unwrap();
        assert_eq!(messages.len(), 2, "wrote two samples, should read two back");
        let decoded =
            PolicyCaptureSample::decode(messages[0].as_slice()).expect("decode sample 0");
        assert_eq!(decoded.tick, 7);
        assert!((decoded.sim_time_s - 0.42).abs() < 1e-9);
        assert_eq!(decoded.mode, "RUN_POLICY");
        assert!(decoded.dry_run && decoded.imu_live);
        assert_eq!(decoded.observation.len(), 49);
        assert_eq!(decoded.raw_action.len(), 24);

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }
}
