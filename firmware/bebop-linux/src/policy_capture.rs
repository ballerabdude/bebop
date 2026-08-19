//! Off-the-hot-loop MCAP writer for operator-toggled policy captures.
//!
//! ROS2-compatible: uses the `ros2` profile, `ros2msg` schema encoding,
//! and `cdr` message encoding. Foxglove Studio's ROS2 panel can plot
//! every channel out of the box. Data is split across 5 standard channels:
//!
//! - `/joint_states`   → sensor_msgs/JointState
//! - `/imu`            → sensor_msgs/Imu
//! - `/policy/status`  → bebop_msgs/PolicyStatus
//! - `/policy/observation` → bebop_msgs/Float32Stamped
//! - `/policy/action`  → bebop_msgs/PolicyAction

use std::collections::BTreeMap;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use tracing::{debug, error, info};

use crate::logging::cdr::CdrEncoder;
use crate::logging::ros2_msgs;
use crate::observation::JOINT_NAMES;
use crate::policy_io::PolicyIoShared;

// --- Tick sample ---

/// All data for one 100 Hz tick, ready for CDR serialization.
#[derive(Clone)]
pub struct TickSample {
    pub tick: u64,
    pub wall_time_ns: u64,
    pub sim_time_s: f64,
    pub mode: String,
    pub dry_run: bool,
    pub imu_live: bool,
    pub quaternion: [f32; 4],
    pub angular_velocity: [f32; 3],
    pub joint_pos_rad: Vec<f32>,
    pub joint_vel_rad_s: Vec<f32>,
    /// Motor torque feedback (Nm), URDF joint convention (direction-
    /// corrected at the RX boundary, same as position / velocity). Written
    /// to `/joint_states.effort` so captures can be checked against the
    /// static gravity load and the per-joint `tau_max` envelopes.
    pub joint_torque_nm: Vec<f32>,
    pub joint_armed: Vec<bool>,
    pub observation: Vec<f32>,
    pub raw_action: Vec<f32>,
    pub position_targets_rad: Vec<f32>,
    pub kp: Vec<f32>,
    pub kd: Vec<f32>,
}

// --- Channel descriptors ---

struct ChanInfo {
    topic: &'static str,
    schema_name: &'static str,
    schema_data: &'static str,
}

const CHANNELS: &[ChanInfo] = &[
    ChanInfo { topic: "/joint_states",   schema_name: ros2_msgs::JOINT_STATE_SCHEMA_NAME,   schema_data: ros2_msgs::JOINT_STATE_SCHEMA },
    ChanInfo { topic: "/imu",            schema_name: ros2_msgs::IMU_SCHEMA_NAME,            schema_data: ros2_msgs::IMU_SCHEMA },
    ChanInfo { topic: "/policy/status",  schema_name: ros2_msgs::POLICY_STATUS_SCHEMA_NAME,  schema_data: ros2_msgs::POLICY_STATUS_SCHEMA },
    ChanInfo { topic: "/policy/observation", schema_name: ros2_msgs::OBSERVATION_SCHEMA_NAME, schema_data: ros2_msgs::OBSERVATION_SCHEMA },
    ChanInfo { topic: "/policy/action",  schema_name: ros2_msgs::POLICY_ACTION_SCHEMA_NAME,  schema_data: ros2_msgs::POLICY_ACTION_SCHEMA },
];

const CDR_ENCODING: &str = "cdr";
const ROS2MSG_ENCODING: &str = "ros2msg";
const ROS2_PROFILE: &str = "ros2";

const SAMPLE_CHANNEL_CAPACITY: usize = 2000;
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);
const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;
const DISK_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const ROTATE_CHECK_EVERY: u64 = 100;
const OPEN_ERROR_LOG_BACKOFF: Duration = Duration::from_secs(30);

// --- Commands ---

enum CaptureCommand {
    Open,
    Sample(Box<TickSample>),
    Close,
    Shutdown,
}

// --- CaptureHandle ---

#[derive(Clone)]
pub struct CaptureHandle {
    tx: SyncSender<CaptureCommand>,
    closed: Arc<AtomicBool>,
    dropped_total: Arc<AtomicU64>,
}

impl CaptureHandle {
    pub fn send_sample(&self, sample: TickSample) -> bool {
        if self.closed.load(Ordering::Relaxed) {
            return false;
        }
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

    pub fn request_open(&self) {
        let _ = self.send_command(CaptureCommand::Open);
    }

    pub fn request_close(&self) {
        let _ = self.send_command(CaptureCommand::Close);
    }

    pub fn dropped_total(&self) -> u64 {
        self.dropped_total.load(Ordering::Relaxed)
    }

    fn send_command(&self, cmd: CaptureCommand) -> bool {
        if self.closed.load(Ordering::Relaxed) {
            return false;
        }
        match self.tx.try_send(cmd) {
            Ok(()) => true,
            Err(TrySendError::Full(cmd)) => match self.tx.send(cmd) {
                Ok(()) => true,
                Err(_) => { self.closed.store(true, Ordering::Relaxed); false }
            },
            Err(TrySendError::Disconnected(_)) => {
                self.closed.store(true, Ordering::Relaxed);
                false
            }
        }
    }

    pub fn shutdown(&self) {
        if self.closed.swap(true, Ordering::Relaxed) {
            return;
        }
        let _ = self.tx.send(CaptureCommand::Shutdown);
    }
}

pub fn spawn_capture_thread(
    capture_dir: PathBuf,
    policy_io: PolicyIoShared,
) -> (CaptureHandle, JoinHandle<()>) {
    let (tx, rx) = mpsc::sync_channel::<CaptureCommand>(SAMPLE_CHANNEL_CAPACITY);
    let closed = Arc::new(AtomicBool::new(false));
    let dropped_total = Arc::new(AtomicU64::new(0));
    let handle = CaptureHandle { tx, closed: closed.clone(), dropped_total: dropped_total.clone() };
    let join = std::thread::Builder::new()
        .name("policy-capture".to_string())
        .spawn(move || run_writer(rx, capture_dir, policy_io, dropped_total, closed))
        .expect("spawn policy-capture thread");
    (handle, join)
}

// --- OpenCapture (writer-thread state) ---

struct OpenCapture {
    path: PathBuf,
    writer: mcap::Writer<BufWriter<fs::File>>,
    channel_ids: [u16; 5],
    rows: u64,
    last_flush: Instant,
}

fn ns_to_sec_nsec(ns: u64) -> (u32, u32) {
    ((ns / 1_000_000_000) as u32, (ns % 1_000_000_000) as u32)
}

impl OpenCapture {
    fn write_header(enc: &mut CdrEncoder, frame_id: &str, wall_ns: u64) {
        let (sec, nsec) = ns_to_sec_nsec(wall_ns);
        enc.write_header();
        enc.write_u32(sec);
        enc.write_u32(nsec);
        enc.write_string(frame_id);
    }

    fn post_message(&mut self, ch_idx: usize, log_time: u64, enc: &CdrEncoder) -> Result<()> {
        self.writer.write_to_known_channel(
            &mcap::records::MessageHeader {
                channel_id: self.channel_ids[ch_idx],
                sequence: self.rows as u32,
                log_time,
                publish_time: log_time,
            },
            enc.as_bytes(),
        ).context("mcap: write_to_known_channel")?;
        Ok(())
    }

    fn write_joint_state(&mut self, data: &TickSample) -> Result<()> {
        let mut enc = CdrEncoder::with_capacity(512);
        // Tag with the URDF root link so the frame the joint_states reference
        // matches the frame the URDF panel populates (base_link). Using an
        // unrelated frame like "world" leaves a disconnected frame and makes
        // the 3D panel's fixed/display frame unresolvable.
        Self::write_header(&mut enc, "base_link", data.wall_time_ns);
        // Use the canonical joint names so Foxglove's URDF panel can map
        // each JointState entry to the matching URDF joint. These MUST match
        // the joint names in bebopv2.urdf (and observation::JOINT_NAMES).
        enc.write_u32(JOINT_NAMES.len() as u32);
        for n in &JOINT_NAMES { enc.write_string(n); }
        enc.write_u32(data.joint_pos_rad.len() as u32);
        for &v in &data.joint_pos_rad { enc.write_f64(v as f64); }
        enc.write_u32(data.joint_vel_rad_s.len() as u32);
        for &v in &data.joint_vel_rad_s { enc.write_f64(v as f64); }
        enc.write_u32(data.joint_torque_nm.len() as u32);
        for &v in &data.joint_torque_nm { enc.write_f64(v as f64); }
        self.post_message(0, data.wall_time_ns, &enc)
    }

    fn write_imu(&mut self, data: &TickSample) -> Result<()> {
        let mut enc = CdrEncoder::with_capacity(256);
        Self::write_header(&mut enc, "imu_link", data.wall_time_ns);
        let (qx, qy, qz, qw) = (data.quaternion[0] as f64, data.quaternion[1] as f64, data.quaternion[2] as f64, data.quaternion[3] as f64);
        enc.write_f64(qx); enc.write_f64(qy); enc.write_f64(qz); enc.write_f64(qw);
        for _ in 0..9 { enc.write_f64(0.0); }
        enc.write_f64(data.angular_velocity[0] as f64);
        enc.write_f64(data.angular_velocity[1] as f64);
        enc.write_f64(data.angular_velocity[2] as f64);
        for _ in 0..9 { enc.write_f64(0.0); }
        for _ in 0..3 { enc.write_f64(0.0); }
        for _ in 0..9 { enc.write_f64(0.0); }
        self.post_message(1, data.wall_time_ns, &enc)
    }

    fn write_policy_status(&mut self, data: &TickSample) -> Result<()> {
        let mut enc = CdrEncoder::with_capacity(128);
        Self::write_header(&mut enc, "", data.wall_time_ns);
        enc.write_string(&data.mode);
        enc.write_bool(data.dry_run);
        enc.write_bool(data.imu_live);
        enc.write_f64(data.sim_time_s);
        self.post_message(2, data.wall_time_ns, &enc)
    }

    fn write_observation(&mut self, data: &TickSample) -> Result<()> {
        let mut enc = CdrEncoder::with_capacity(256);
        Self::write_header(&mut enc, "", data.wall_time_ns);
        enc.write_u32(data.observation.len() as u32);
        for &v in &data.observation { enc.write_u32(v.to_bits()); }
        self.post_message(3, data.wall_time_ns, &enc)
    }

    fn write_action(&mut self, data: &TickSample) -> Result<()> {
        if data.raw_action.is_empty() { return Ok(()); }
        let mut enc = CdrEncoder::with_capacity(192);
        Self::write_header(&mut enc, "", data.wall_time_ns);
        for arr in &[&data.raw_action, &data.position_targets_rad, &data.kp, &data.kd] {
            enc.write_u32(arr.len() as u32);
            for &v in *arr { enc.write_u32(v.to_bits()); }
        }
        self.post_message(4, data.wall_time_ns, &enc)
    }

    fn flush_if_due(&mut self) -> Result<()> {
        let now = Instant::now();
        if now.duration_since(self.last_flush) >= FLUSH_INTERVAL {
            self.writer.flush().context("mcap: flush")?;
            self.last_flush = now;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.writer.finish().context("mcap: finish")?;
        Ok(())
    }
}

// --- Writer thread ---

fn open_capture(dir: &Path) -> Result<OpenCapture> {
    fs::create_dir_all(dir).with_context(|| format!("create dir {}", dir.display()))?;
    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("policy_capture_{stamp}.mcap");
    let path = dir.join(filename);
    let file = fs::File::create(&path).with_context(|| format!("create file {}", path.display()))?;
    let buf = BufWriter::with_capacity(64 * 1024, file);
    let mut writer = mcap::Writer::with_options(
        buf,
        mcap::WriteOptions::new()
            .compression(Some(mcap::Compression::Zstd))
            .profile(ROS2_PROFILE)
            .library(format!("bebop-linux mcap {}", mcap::VERSION)),
    ).context("mcap: create writer")?;

    let mut channel_ids = [0u16; 5];
    for (i, ch) in CHANNELS.iter().enumerate() {
        let schema_id = writer
            .add_schema(ch.schema_name, ROS2MSG_ENCODING, ch.schema_data.as_bytes())
            .context("mcap: add_schema")?;
        let mut metadata = BTreeMap::new();
        metadata.insert("offered_qos_profiles".to_string(), "reliability: reliable\ndurability: volatile\n".to_string());
        let cid = writer.add_channel(schema_id, ch.topic, CDR_ENCODING, &metadata).context("mcap: add_channel")?;
        channel_ids[i] = cid;
        info!(topic = %ch.topic, schema = %ch.schema_name, channel_id = cid, "capture: registered channel");
    }
    Ok(OpenCapture { path, writer, channel_ids, rows: 0, last_flush: Instant::now() })
}

fn publish_status(policy_io: &PolicyIoShared, open: Option<&OpenCapture>, dropped_total: &AtomicU64) {
    if let Ok(mut g) = policy_io.lock() {
        let dropped = dropped_total.load(Ordering::Relaxed);
        match open {
            Some(c) => g.set_capture_state(true, &c.path.to_string_lossy(), c.rows, dropped),
            None => g.set_capture_state(false, "", 0, dropped),
        }
    }
}

fn prune(dir: &Path, budget: u64, keep: &Path) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    struct Seg { path: PathBuf, size: u64, mtime: SystemTime }
    let mut segs: Vec<Seg> = Vec::new();
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("policy_capture_") && n.ends_with(".mcap")) { continue; }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() { continue; }
        total += meta.len();
        segs.push(Seg { path, size: meta.len(), mtime: meta.modified().unwrap_or(UNIX_EPOCH) });
    }
    if total <= budget { return }
    segs.sort_by_key(|s| s.mtime);
    for seg in segs {
        if total <= budget { break }
        if seg.path == keep { continue }
        if fs::remove_file(&seg.path).is_ok() { total = total.saturating_sub(seg.size); }
    }
}

fn run_writer(
    rx: Receiver<CaptureCommand>,
    capture_dir: PathBuf,
    policy_io: PolicyIoShared,
    dropped_total: Arc<AtomicU64>,
    closed: Arc<AtomicBool>,
) {
    debug!(dir = %capture_dir.display(), "capture: writer started (ros2 mcap)");
    let mut open: Option<OpenCapture> = None;
    let mut last_open_error: Option<(Instant, String)> = None;

    loop {
        match rx.recv_timeout(FLUSH_INTERVAL) {
            Ok(CaptureCommand::Open) => {
                if open.is_some() { continue }
                match open_capture(&capture_dir) {
                    Ok(c) => {
                        info!(path = %c.path.display(), "capture: opened (ros2 MCAP)");
                        publish_status(&policy_io, Some(&c), &dropped_total);
                        prune(&capture_dir, DISK_BUDGET_BYTES, &c.path);
                        open = Some(c);
                        last_open_error = None;
                    }
                    Err(e) => {
                        let msg = format!("{e:#}");
                        let now = Instant::now();
                        let should_log = match last_open_error.as_ref() {
                            Some((t, prev)) => prev != &msg || now.duration_since(*t) >= OPEN_ERROR_LOG_BACKOFF,
                            None => true,
                        };
                        if should_log {
                            error!(dir = %capture_dir.display(), error = %msg, "capture: open failed");
                            last_open_error = Some((now, msg));
                        }
                        publish_status(&policy_io, None, &dropped_total);
                    }
                }
            }
            Ok(CaptureCommand::Sample(data)) => {
                let Some(c) = open.as_mut() else { continue };
                macro_rules! try_write {
                    ($e:expr) => {
                        if let Err(err) = $e {
                            error!(path = %c.path.display(), error = %err, "capture: write failed");
                            let bad = open.take().unwrap(); let _ = bad.finish();
                            publish_status(&policy_io, None, &dropped_total);
                            continue;
                        }
                    };
                }
                try_write!(c.write_joint_state(&data));
                try_write!(c.write_imu(&data));
                try_write!(c.write_policy_status(&data));
                try_write!(c.write_observation(&data));
                let _ = c.write_action(&data);
                c.rows += 1;

                if c.rows % ROTATE_CHECK_EVERY == 0 {
                    let _ = c.writer.flush();
                    c.last_flush = Instant::now();
                    if let Ok(meta) = fs::metadata(&c.path) {
                        if meta.len() >= MAX_FILE_BYTES {
                            let prev = open.take().unwrap();
                            let prev_path = prev.path.clone();
                            let prev_rows = prev.rows;
                            let _ = prev.finish();
                            info!(path = %prev_path.display(), rows = prev_rows, "capture: rotated segment");
                            match open_capture(&capture_dir) {
                                Ok(next) => {
                                    info!(path = %next.path.display(), "capture: opened next segment");
                                    publish_status(&policy_io, Some(&next), &dropped_total);
                                    prune(&capture_dir, DISK_BUDGET_BYTES, &next.path);
                                    open = Some(next);
                                }
                                Err(e) => {
                                    error!(error = %e, "capture: open next segment failed");
                                    publish_status(&policy_io, None, &dropped_total);
                                }
                            }
                        }
                    }
                }
            }
            Ok(CaptureCommand::Close) => {
                if let Some(c) = open.take() {
                    let path = c.path.clone();
                    let rows = c.rows;
                    let _ = c.finish();
                    info!(path = %path.display(), rows, "capture: closed");
                    publish_status(&policy_io, None, &dropped_total);
                }
            }
            Ok(CaptureCommand::Shutdown) => {
                if let Some(c) = open.take() { let _ = c.finish(); }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(c) = open.as_mut() { let _ = c.flush_if_due(); publish_status(&policy_io, Some(c), &dropped_total); }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if let Some(c) = open.take() { let _ = c.finish(); }
                break;
            }
        }
    }
    closed.store(true, Ordering::Relaxed);
    publish_status(&policy_io, None, &dropped_total);
    debug!("capture: writer exiting");
}

// --- Timestamps helper ---

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

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    /// Open a capture, write one tick, finish, and verify the MCAP
    /// stream contains the expected channels and schema names.
    #[test]
    fn ros2_mcap_round_trip() {
        let dir = std::env::temp_dir().join(format!("bebop_ros2_mcap_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let mut cap = open_capture(&dir).expect("open ros2 capture");
        let path = cap.path.clone();

        let data = TickSample {
            tick: 1,
            wall_time_ns: 1_700_000_000_000_000_000,
            sim_time_s: 0.0,
            mode: "RUN_POLICY".into(),
            dry_run: false,
            imu_live: true,
            quaternion: [0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            joint_pos_rad: vec![0.1; 8],
            joint_vel_rad_s: vec![0.0; 8],
            joint_torque_nm: vec![0.0; 8],
            joint_armed: vec![true; 8],
            observation: vec![0.0; 49],
            raw_action: vec![0.0; 24],
            position_targets_rad: vec![0.0; 8],
            kp: vec![20.0; 8],
            kd: vec![1.0; 8],
        };

        cap.write_joint_state(&data).unwrap();
        cap.write_imu(&data).unwrap();
        cap.write_policy_status(&data).unwrap();
        cap.write_observation(&data).unwrap();
        cap.write_action(&data).unwrap();
        cap.finish().unwrap();

        let bytes = fs::read(&path).expect("read capture file");
        let mut topics: Vec<String> = Vec::new();
        let mut schema_names: Vec<String> = Vec::new();
        for msg in mcap::MessageStream::new(&bytes).expect("open stream") {
            let msg = msg.expect("decode message");
            topics.push(msg.channel.topic.clone());
            if let Some(s) = msg.channel.schema.as_ref() {
                if !schema_names.contains(&s.name) { schema_names.push(s.name.clone()); }
            }
        }

        assert_eq!(topics.len(), 5, "should have 5 messages (one per channel)");
        assert!(schema_names.contains(&"sensor_msgs/msg/JointState".to_string()));
        assert!(schema_names.contains(&"sensor_msgs/msg/Imu".to_string()));
        assert!(schema_names.contains(&"bebop_msgs/msg/PolicyStatus".to_string()));
        assert!(schema_names.contains(&"bebop_msgs/msg/Float32Stamped".to_string()));
        assert!(schema_names.contains(&"bebop_msgs/msg/PolicyAction".to_string()));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }
}