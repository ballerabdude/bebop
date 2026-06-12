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
//! File layout: one MCAP file per `request_open` → `request_close`
//! cycle, named
//! `policy_capture_<label?>_<YYYYmmdd_HHMMSS>.mcap`, with:
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

/// Commands sent from the tick thread to the writer thread. Open /
/// close are explicit instead of being inferred from sample stream
/// state so the writer thread can ack each transition by publishing
/// to [`PolicyIoShared`].
enum CaptureCommand {
    /// Open a new capture file. The writer chooses the filename
    /// (sanitized label + timestamp) and publishes the resolved path
    /// to [`PolicyIoShared::set_capture_state`].
    Open { label: String },
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

    /// Request a fresh capture file. The writer opens it on its own
    /// thread and updates `PolicyIoShared` once the path is known. A
    /// repeat `request_open` while a file is already open is a no-op
    /// in the writer thread (we don't auto-rotate mid-capture).
    pub fn request_open(&self, label: &str) {
        let _ = self.send_command(CaptureCommand::Open {
            label: label.to_string(),
        });
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

    loop {
        // `recv_timeout` lets us run a periodic flush even when the
        // tick thread is quiescent. The timeout matches the flush
        // cadence so worst case is two intervals between flushes.
        match rx.recv_timeout(FLUSH_INTERVAL) {
            Ok(CaptureCommand::Open { label }) => {
                if open.is_some() {
                    debug!(
                        "capture: ignoring Open while a file is already open \
                         (toggle Close first to rotate)"
                    );
                    continue;
                }
                match open_capture(&capture_dir, &label) {
                    Ok(c) => {
                        info!(path = %c.path.display(), "capture: opened (MCAP)");
                        publish_status(&policy_io, Some(&c), &dropped_total);
                        open = Some(c);
                    }
                    Err(e) => {
                        error!(
                            dir = %capture_dir.display(),
                            error = %format!("{e:#}"),
                            "capture: open failed"
                        );
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

/// Strip a label down to filesystem-safe characters. Accepts ASCII
/// alphanumerics plus `_` / `-`; everything else (including spaces and
/// `.`) becomes `_`. Truncated to 32 chars so the timestamp prefix
/// stays the dominant part of the filename.
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
