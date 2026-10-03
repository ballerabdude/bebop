//! Always-on system telemetry MCAP logger.
//!
//! A dedicated 1 Hz OS thread that records the power-board snapshot
//! (battery / motor voltage, branch currents, board temperature, fault
//! word, SOC) together with host utilization (CPU, GPU, RAM, thermals,
//! load, capture-disk fill) to a rolling MCAP in the capture dir. It is
//! independent of any drive/policy session and runs from boot until the
//! process exits, so idle periods, thermal soak, and power spin-up are
//! captured too — the context a drive-session-only log would miss.
//!
//! Protobuf-encoded (`bebop.system.*`, schema data = the embedded
//! `FileDescriptorSet`), so Rerun's generic protobuf MCAP decoder maps
//! each message to a `<full_name>:message` component and the navd
//! dashboard's `SeriesLines` can plot its fields — the same mechanism the
//! navd recorder's `bebop.navd.*` telemetry uses. Three channels:
//!
//! - `/power`   → `bebop.system.Power`
//! - `/host`    → `bebop.system.Host`
//! - `/session` → `bebop.system.SessionMarker` (drive-state start/stop; a
//!   `navd_session_*.mcap` is correlated with the system timeline by
//!   timestamp)
//!
//! All writers live in this thread; there is no high-rate producer, so no
//! mpsc is needed. Files rotate hourly / by size and are pruned
//! oldest-first under a disk budget, mirroring `policy_capture`.

use std::collections::BTreeMap;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use prost::Message;
use tracing::{debug, error, info};

use crate::mode::Mode;
use crate::powerboard::describe_faults;
use crate::safety::Supervisor;

/// prost-generated `bebop.system.*` messages from `protos/system.proto`.
mod proto {
    include!(concat!(env!("OUT_DIR"), "/bebop.system.rs"));
}

/// Serialized `FileDescriptorSet` for the system messages, embedded as the
/// MCAP schema data so Rerun/Foxglove can build the descriptors.
const SYSTEM_FDS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/system_fds.bin"));

const PROTOBUF_ENCODING: &str = "protobuf";
const PROFILE: &str = "bebop_system";

/// Sampling period: the requested always-on 1 Hz.
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);
/// Flush cadence. `flush()` finalizes the in-progress chunk, so a sudden
/// power loss can only drop samples buffered since the last flush.
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);
/// `fsync` cadence: bounds how much flushed data can still be sitting in
/// the OS page cache when power is cut. `sync_all` on eMMC/NVMe is not
/// free, so we don't do it every second.
const SYNC_INTERVAL: Duration = Duration::from_secs(5);
/// Roll to a new file after this long even if the size budget is nowhere
/// near — keeps individual files small and lets pruning work.
const ROTATE_INTERVAL: Duration = Duration::from_secs(3600);
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Oldest system logs are pruned above this total. At ~1 Hz the log is a
/// few tens of MB/day, so this holds months.
const DISK_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const OPEN_ERROR_LOG_BACKOFF: Duration = Duration::from_secs(30);

// --- Channel descriptors ---------------------------------------------------

struct ChanInfo {
    topic: &'static str,
    schema_name: &'static str,
}

const CHANNELS: &[ChanInfo] = &[
    ChanInfo {
        topic: "/power",
        schema_name: "bebop.system.Power",
    },
    ChanInfo {
        topic: "/host",
        schema_name: "bebop.system.Host",
    },
    ChanInfo {
        topic: "/session",
        schema_name: "bebop.system.SessionMarker",
    },
];

// --- Host sampler ----------------------------------------------------------

/// Dependency-free host metrics reader (`/proc` + `/sys`). Discovery of
/// thermal zones and GPU load paths happens once; per-sample reads are
/// cheap. Every read is best-effort — a missing path just leaves the
/// field at its default rather than failing the tick.
struct HostStatsSampler {
    /// `(total, idle)` CPU jiffies from the previous sample, for the delta.
    prev_cpu: Option<(u64, u64)>,
    disk_path: PathBuf,
    /// Candidate GPU load files, in priority order (per-mille or percent).
    gpu_paths: Vec<PathBuf>,
    /// `(thermal_zone_dir, zone_name)`.
    thermal: Vec<(PathBuf, String)>,
}

impl HostStatsSampler {
    fn new(disk_path: &Path) -> Self {
        Self {
            prev_cpu: read_cpu_ticks(),
            disk_path: disk_path.to_path_buf(),
            gpu_paths: discover_gpu(),
            thermal: discover_thermal(),
        }
    }

    fn sample(&mut self, wall_ns: u64) -> proto::Host {
        let mut h = proto::Host {
            stamp_ns: wall_ns as i64,
            ..Default::default()
        };

        if let Some((total, idle)) = read_cpu_ticks() {
            if let Some((pt, pi)) = self.prev_cpu {
                let dt = total.saturating_sub(pt);
                let di = idle.saturating_sub(pi).min(dt);
                if dt > 0 {
                    h.cpu_pct = (dt - di) as f32 / dt as f32 * 100.0;
                }
            }
            self.prev_cpu = Some((total, idle));
        }

        if let Some((total, available)) = read_meminfo() {
            h.ram_total_bytes = total;
            h.ram_used_bytes = total.saturating_sub(available);
            if total > 0 {
                h.ram_used_pct = h.ram_used_bytes as f32 / total as f32 * 100.0;
            }
        }

        if let Some(pct) = read_gpu(&self.gpu_paths) {
            h.gpu_pct = pct;
            h.gpu_present = true;
        }

        if let Some(pct) = disk_used_pct(&self.disk_path) {
            h.disk_used_pct = pct;
        }

        let (l1, l5, l15) = read_loadavg();
        h.load1 = l1;
        h.load5 = l5;
        h.load15 = l15;

        for (dir, name) in &self.thermal {
            if let Some(temp_c) = read_thermal(dir) {
                h.thermal.push(proto::ThermalZone {
                    name: name.clone(),
                    temp_c,
                });
            }
        }

        h
    }
}

/// `(total, idle)` CPU jiffies from the aggregate `cpu` line of
/// `/proc/stat`. `iowait` counts as idle.
fn read_cpu_ticks() -> Option<(u64, u64)> {
    let s = fs::read_to_string("/proc/stat").ok()?;
    let line = s.lines().next()?;
    let mut it = line.split_whitespace();
    if it.next()? != "cpu" {
        return None;
    }
    let vals: Vec<u64> = it.filter_map(|v| v.parse().ok()).collect();
    if vals.len() < 4 {
        return None;
    }
    let idle = vals.get(3).copied().unwrap_or(0) + vals.get(4).copied().unwrap_or(0);
    let total: u64 = vals.iter().take(8).sum();
    Some((total, idle))
}

/// `(total_bytes, available_bytes)` from `/proc/meminfo`.
fn read_meminfo() -> Option<(u64, u64)> {
    let s = fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = 0u64;
    let mut available = 0u64;
    let mut free = 0u64;
    for line in s.lines() {
        let mut it = line.split_whitespace();
        let key = it.next().unwrap_or("");
        let val: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        match key {
            "MemTotal:" => total = val.saturating_mul(1024),
            "MemAvailable:" => available = val.saturating_mul(1024),
            "MemFree:" => free = val.saturating_mul(1024),
            _ => {}
        }
    }
    if total == 0 {
        return None;
    }
    let avail = if available > 0 { available } else { free };
    Some((total, avail))
}

fn discover_thermal() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/thermal") else {
        return out;
    };
    let mut zones: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("thermal_zone"))
        })
        .collect();
    zones.sort();
    for z in zones {
        let name = fs::read_to_string(z.join("type"))
            .map(|s| s.trim().to_string())
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                z.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
        out.push((z, name));
    }
    out
}

fn read_thermal(dir: &Path) -> Option<f32> {
    let s = fs::read_to_string(dir.join("temp")).ok()?;
    let milli: i64 = s.trim().parse().ok()?;
    Some(milli as f32 / 1000.0)
}

fn discover_gpu() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/devfreq") {
        for e in entries.flatten() {
            let base = e.path();
            let name = e.file_name().to_string_lossy().to_lowercase();
            if name.contains("gpu") {
                for cand in ["device/load", "load"] {
                    let p = base.join(cand);
                    if p.exists() {
                        out.push(p);
                    }
                }
            }
        }
    }
    for p in [
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/gpu.0/load",
    ] {
        let p = PathBuf::from(p);
        if p.exists() {
            out.push(p);
        }
    }
    out
}

/// GPU load as a percentage. Jetson devfreq `load` is per-mille
/// (0..1000); some sysfs nodes report percent directly, so a value above
/// 100 is treated as per-mille.
fn read_gpu(paths: &[PathBuf]) -> Option<f32> {
    for p in paths {
        if let Ok(s) = fs::read_to_string(p) {
            if let Ok(v) = s.trim().parse::<f32>() {
                let pct = if v > 100.0 { v / 10.0 } else { v };
                return Some(pct.clamp(0.0, 100.0));
            }
        }
    }
    None
}

fn read_loadavg() -> (f32, f32, f32) {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| {
            let mut it = s.split_whitespace();
            let a = it.next()?.parse().ok()?;
            let b = it.next()?.parse().ok()?;
            let c = it.next()?.parse().ok()?;
            Some((a, b, c))
        })
        .unwrap_or((0.0, 0.0, 0.0))
}

/// Used / total capacity of the filesystem containing `path`, in percent.
fn disk_used_pct(path: &Path) -> Option<f32> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
    if rc != 0 {
        return None;
    }
    let total = st.f_blocks * st.f_frsize;
    let free = st.f_bavail * st.f_frsize;
    if total == 0 {
        return None;
    }
    Some(((total - free) as f64 / total as f64 * 100.0) as f32)
}

// --- Power sampling --------------------------------------------------------

/// Snapshot the power-board cache into a [`proto::Power`]. Mirrors the
/// staleness / SOC logic in `server::telemetry::build_power_stats` but
/// keeps the raw branch currents and fault word for the log.
fn sample_power(sup: &Arc<Supervisor>, wall_ns: u64) -> proto::Power {
    let mut p = proto::Power {
        stamp_ns: wall_ns as i64,
        ..Default::default()
    };
    let Some(cfg) = sup.cfg().power.as_ref() else {
        return p;
    };
    let Some(snapshot) = sup.power_snapshot() else {
        return p;
    };

    let now = Instant::now();
    let staleness_ms = cfg.poll_interval_ms.saturating_mul(3).max(2_000);

    p.present = true;
    p.status_received = snapshot.status.is_some();
    p.status_stale = snapshot.is_stale(now, staleness_ms);
    p.can_interface = cfg.can_interface.clone();
    p.firmware_version = snapshot.version.clone().unwrap_or_default();
    p.battery_cells = cfg.battery_cells;
    p.pack_full_voltage_v = cfg.pack_full_voltage();
    p.pack_empty_voltage_v = cfg.pack_empty_voltage();

    if let Some(s) = snapshot.status.as_ref() {
        p.battery_voltage_v = s.battery_voltage_v;
        p.motor_voltage_v = s.motor_voltage_v;
        p.board_temperature_c = s.board_temperature_c;
        p.fault_bits = s.fault_bits;
        p.fault_description = describe_faults(s.fault_bits);
        p.state_of_charge_pct = cfg.estimate_soc_pct(s.battery_voltage_v).unwrap_or(-1.0);
    } else {
        p.state_of_charge_pct = -1.0;
    }

    let currents = snapshot.currents.unwrap_or_default();
    p.current_al_a = currents.al_current_a;
    p.current_ar_a = currents.ar_current_a;
    p.current_ll_a = currents.ll_current_a;
    p.current_lr_a = currents.lr_current_a;
    p.total_motor_current_a = currents.al_current_a
        + currents.ar_current_a
        + currents.ll_current_a
        + currents.lr_current_a;

    p
}

// --- Drive-state predicate -------------------------------------------------

/// Whether the chassis is in a drivable, armed, non-estop state. Mirrors
/// the recorder's gate: DIAL_IN / RUN_POLICY, no E-STOP, every wheel
/// armed (an empty wheel set counts as armed — legged build).
fn is_drive_active(mode: Mode, estop: bool, wheels_armed: &[bool]) -> bool {
    matches!(mode, Mode::DialIn | Mode::RunPolicy) && !estop && wheels_armed.iter().all(|&a| a)
}

fn drive_active(sup: &Arc<Supervisor>) -> bool {
    let wheels: Vec<bool> = sup.snapshot_wheels().iter().map(|w| w.armed).collect();
    is_drive_active(sup.mode(), sup.estop_active(), &wheels)
}

// --- Open file (writer-thread state) ---------------------------------------

struct OpenSystem {
    path: PathBuf,
    writer: mcap::Writer<BufWriter<fs::File>>,
    /// Second handle to the same file, for `sync_all` (the writer owns the
    /// BufWriter, so we can't reach the File through it).
    sync_file: fs::File,
    channels: [u16; 3],
    rows: u64,
    last_flush: Instant,
    last_sync: Instant,
    opened_at: Instant,
}

impl OpenSystem {
    fn post(&mut self, ch: usize, log_time: u64, payload: &[u8]) -> Result<()> {
        self.writer
            .write_to_known_channel(
                &mcap::records::MessageHeader {
                    channel_id: self.channels[ch],
                    sequence: self.rows as u32,
                    log_time,
                    publish_time: log_time,
                },
                payload,
            )
            .context("mcap: write_to_known_channel")?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("mcap: flush")?;
        self.last_flush = Instant::now();
        // Periodically push the flushed bytes out of the OS page cache too,
        // so a sudden power cut can't lose more than SYNC_INTERVAL.
        if self.last_flush.duration_since(self.last_sync) >= SYNC_INTERVAL {
            self.sync_file.sync_all().context("mcap: fsync")?;
            self.last_sync = Instant::now();
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.writer.finish().context("mcap: finish")?;
        Ok(())
    }
}

fn unique_path(dir: &Path, stem: &str) -> PathBuf {
    let mut path = dir.join(format!("{stem}.mcap"));
    let mut n = 1;
    while path.exists() {
        path = dir.join(format!("{stem}_{n}.mcap"));
        n += 1;
    }
    path
}

fn open_system(dir: &Path) -> Result<OpenSystem> {
    fs::create_dir_all(dir).with_context(|| format!("create dir {}", dir.display()))?;
    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let path = unique_path(dir, &format!("system_{stamp}"));
    let file =
        fs::File::create(&path).with_context(|| format!("create file {}", path.display()))?;
    let sync_file = file
        .try_clone()
        .with_context(|| format!("clone handle for {}", path.display()))?;
    let buf = BufWriter::with_capacity(64 * 1024, file);
    let mut writer = mcap::Writer::with_options(
        buf,
        mcap::WriteOptions::new()
            .compression(Some(mcap::Compression::Zstd))
            .profile(PROFILE)
            .library(format!("bebop-linux system {}", mcap::VERSION)),
    )
    .context("mcap: create system writer")?;

    let mut channels = [0u16; 3];
    for (i, ch) in CHANNELS.iter().enumerate() {
        let schema_id = writer
            .add_schema(ch.schema_name, PROTOBUF_ENCODING, SYSTEM_FDS)
            .context("mcap: add_schema")?;
        let metadata = BTreeMap::new();
        let cid = writer
            .add_channel(schema_id, ch.topic, PROTOBUF_ENCODING, &metadata)
            .context("mcap: add_channel")?;
        channels[i] = cid;
        info!(topic = %ch.topic, schema = %ch.schema_name, "system capture: registered channel");
    }

    Ok(OpenSystem {
        path,
        writer,
        sync_file,
        channels,
        rows: 0,
        last_flush: Instant::now(),
        last_sync: Instant::now(),
        opened_at: Instant::now(),
    })
}

fn prune(dir: &Path, budget: u64, keep: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    struct Seg {
        path: PathBuf,
        size: u64,
        mtime: SystemTime,
    }
    let mut segs: Vec<Seg> = Vec::new();
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("system_") && n.ends_with(".mcap"))
        {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        total += meta.len();
        segs.push(Seg {
            path,
            size: meta.len(),
            mtime: meta.modified().unwrap_or(UNIX_EPOCH),
        });
    }
    if total <= budget {
        return;
    }
    segs.sort_by_key(|s| s.mtime);
    for seg in segs {
        if total <= budget {
            break;
        }
        if seg.path == keep {
            continue;
        }
        if fs::remove_file(&seg.path).is_ok() {
            info!(path = %seg.path.display(), "system capture: pruned (disk budget)");
            total = total.saturating_sub(seg.size);
        }
    }
}

// --- Writer thread ---------------------------------------------------------

/// Spawn the always-on system logger. Returns the join handle so the
/// process can wait for it on shutdown.
pub fn spawn_system_capture(
    sup: Arc<Supervisor>,
    capture_dir: PathBuf,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("system-capture".to_string())
        .spawn(move || run_system_capture(sup, capture_dir, shutdown))
        .expect("spawn system-capture thread")
}

fn run_system_capture(sup: Arc<Supervisor>, dir: PathBuf, shutdown: Arc<AtomicBool>) {
    debug!(dir = %dir.display(), "system capture: thread started");
    let mut sampler = HostStatsSampler::new(&dir);
    let mut open = match open_system(&dir) {
        Ok(c) => {
            info!(path = %c.path.display(), "system capture: opened");
            prune(&dir, DISK_BUDGET_BYTES, &c.path);
            Some(c)
        }
        Err(e) => {
            error!(error = %format!("{e:#}"), "system capture: initial open failed");
            None
        }
    };
    let mut last_open_error: Option<Instant> = None;
    let mut prev_active = false;
    let mut session = String::new();

    while !shutdown.load(Ordering::SeqCst) {
        let tick = Instant::now();
        let wall_ns = now_ns();

        if open.is_none() {
            let now = Instant::now();
            let should_try = last_open_error
                .map(|t| now.duration_since(t) >= OPEN_ERROR_LOG_BACKOFF)
                .unwrap_or(true);
            if should_try {
                match open_system(&dir) {
                    Ok(c) => {
                        info!(path = %c.path.display(), "system capture: opened");
                        prune(&dir, DISK_BUDGET_BYTES, &c.path);
                        open = Some(c);
                        last_open_error = None;
                    }
                    Err(e) => {
                        error!(error = %format!("{e:#}"), "system capture: open failed");
                        last_open_error = Some(now);
                    }
                }
            }
        }

        if open.is_some() {
            let power = sample_power(&sup, wall_ns).encode_to_vec();
            let host = sampler.sample(wall_ns).encode_to_vec();
            let mode = sup.mode();
            let active = drive_active(&sup);

            let write_res = {
                let c = open.as_mut().expect("open is Some");
                let mut res = c.post(0, wall_ns, &power);
                if res.is_ok() {
                    res = c.post(1, wall_ns, &host);
                }
                if res.is_ok() && active != prev_active {
                    if active {
                        session = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
                    }
                    let event = if active { "start" } else { "stop" };
                    let marker = proto::SessionMarker {
                        stamp_ns: wall_ns as i64,
                        event: event.to_string(),
                        mode: format!("{mode:?}"),
                        session: session.clone(),
                    }
                    .encode_to_vec();
                    res = c.post(2, wall_ns, &marker);
                }
                res
            };

            match write_res {
                Ok(()) => {
                    prev_active = active;
                    let c = open.as_mut().expect("open is Some");
                    c.rows += 1;
                    if tick.duration_since(c.last_flush) >= FLUSH_INTERVAL {
                        let _ = c.flush();
                    }
                    let too_big = fs::metadata(&c.path)
                        .map(|m| m.len() >= MAX_FILE_BYTES)
                        .unwrap_or(false);
                    if too_big || tick.duration_since(c.opened_at) >= ROTATE_INTERVAL {
                        let prev = open.take().expect("open is Some");
                        let path = prev.path.clone();
                        let rows = prev.rows;
                        let _ = prev.finish();
                        info!(path = %path.display(), rows, "system capture: rotated");
                        match open_system(&dir) {
                            Ok(next) => {
                                info!(path = %next.path.display(), "system capture: opened next");
                                prune(&dir, DISK_BUDGET_BYTES, &next.path);
                                open = Some(next);
                            }
                            Err(e) => {
                                error!(error = %format!("{e:#}"), "system capture: open next failed");
                            }
                        }
                    }
                }
                Err(e) => {
                    error!(error = %format!("{e:#}"), "system capture: write failed");
                    if let Some(bad) = open.take() {
                        let _ = bad.finish();
                    }
                }
            }
        }

        let elapsed = tick.elapsed();
        if elapsed < SAMPLE_PERIOD {
            std::thread::sleep(SAMPLE_PERIOD - elapsed);
        }
    }

    if let Some(c) = open.take() {
        let _ = c.finish();
    }
    debug!("system capture: thread exiting");
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

// --- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bebop_system_mcap_{tag}_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn system_mcap_round_trip() {
        let dir = tmp_dir("roundtrip");
        let mut cap = open_system(&dir).expect("open system capture");
        let path = cap.path.clone();
        let wall = 1_700_000_000_000_000_000u64;

        let power = proto::Power {
            present: true,
            status_received: true,
            battery_voltage_v: 25.2,
            state_of_charge_pct: 80.0,
            ..Default::default()
        };
        let host = proto::Host {
            cpu_pct: 42.0,
            gpu_pct: 10.0,
            gpu_present: true,
            thermal: vec![proto::ThermalZone {
                name: "cpu-thermal".into(),
                temp_c: 45.5,
            }],
            ..Default::default()
        };
        let marker = proto::SessionMarker {
            event: "start".into(),
            mode: "DialIn".into(),
            session: "20260101_000000".into(),
            ..Default::default()
        };
        cap.post(0, wall, &power.encode_to_vec()).unwrap();
        cap.post(1, wall, &host.encode_to_vec()).unwrap();
        cap.post(2, wall, &marker.encode_to_vec()).unwrap();
        cap.finish().unwrap();

        let bytes = fs::read(&path).expect("read capture file");
        let mut topics: Vec<String> = Vec::new();
        let mut schema_names: Vec<String> = Vec::new();
        let mut fds_embedded = false;
        let mut decoded_host: Option<proto::Host> = None;
        let mut decoded_power: Option<proto::Power> = None;
        for msg in mcap::MessageStream::new(&bytes).expect("open stream") {
            let msg = msg.expect("decode message");
            topics.push(msg.channel.topic.clone());
            assert_eq!(msg.channel.message_encoding, PROTOBUF_ENCODING);
            if let Some(s) = msg.channel.schema.as_ref() {
                if !schema_names.contains(&s.name) {
                    schema_names.push(s.name.clone());
                }
                // The schema data must be the embedded FileDescriptorSet,
                // not empty — Rerun builds its descriptors from it.
                fds_embedded |= String::from_utf8_lossy(&s.data).contains("bebop.system.Power");
            }
            match msg.channel.topic.as_str() {
                "/power" => decoded_power = Some(proto::Power::decode(msg.data.as_ref()).unwrap()),
                "/host" => decoded_host = Some(proto::Host::decode(msg.data.as_ref()).unwrap()),
                _ => {}
            }
        }

        assert_eq!(topics.len(), 3, "one message per channel");
        assert!(schema_names.contains(&"bebop.system.Power".to_string()));
        assert!(schema_names.contains(&"bebop.system.Host".to_string()));
        assert!(schema_names.contains(&"bebop.system.SessionMarker".to_string()));
        assert!(fds_embedded, "schema data carries the FileDescriptorSet");

        let power = decoded_power.expect("power payload decodes");
        assert!(power.present);
        assert!((power.battery_voltage_v - 25.2).abs() < 1e-6);
        let host = decoded_host.expect("host payload decodes");
        assert!((host.cpu_pct - 42.0).abs() < 1e-6);
        assert_eq!(host.thermal.len(), 1);
        assert_eq!(host.thermal[0].name, "cpu-thermal");

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn host_sampler_smoke() {
        let mut s = HostStatsSampler::new(Path::new("/tmp"));
        let h = s.sample(0);
        assert!((0.0..=100.0).contains(&h.cpu_pct));
        assert!(h.ram_total_bytes > 0);
        assert!(h.ram_used_pct <= 100.0);
    }

    #[test]
    fn drive_active_predicate() {
        assert!(is_drive_active(Mode::DialIn, false, &[true, true]));
        assert!(is_drive_active(Mode::RunPolicy, false, &[true]));
        assert!(!is_drive_active(Mode::DialIn, false, &[true, false]));
        assert!(!is_drive_active(Mode::Idle, false, &[true, true]));
        assert!(!is_drive_active(Mode::DialIn, true, &[true, true]));
        // empty wheel set counts as armed (legged build), matching the
        // Python recorder's `all({}.values())`.
        assert!(is_drive_active(Mode::DialIn, false, &[]));
    }

    /// A sudden power cut leaves the file without a footer (no `finish()`).
    /// `flush()` finalizes the in-progress chunk, so everything up to the
    /// last flush is a sequence of complete chunks and must still be
    /// recoverable by a streaming reader.
    #[test]
    fn unterminated_capture_is_streamable() {
        let dir = tmp_dir("unterminated");
        let mut cap = open_system(&dir).expect("open system capture");
        let path = cap.path.clone();
        let wall = 1_700_000_000_000_000_000u64;
        for i in 0..3u64 {
            let p = proto::Power {
                stamp_ns: (wall + i) as i64,
                present: true,
                ..Default::default()
            };
            cap.post(0, wall + i, &p.encode_to_vec()).unwrap();
        }
        cap.flush().expect("flush completes the chunk");
        // Skip Drop -> no finish(), exactly like losing power mid-run.
        std::mem::forget(cap);

        let bytes = fs::read(&path).expect("read capture file");
        let n = mcap::MessageStream::new(&bytes)
            .expect("stream unterminated file")
            .filter_map(Result::ok)
            .count();
        assert_eq!(n, 3, "flushed messages survive without a footer");

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }
}
