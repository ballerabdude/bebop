//! MJPEG camera hub: the single owner of the robot's camera.
//!
//! A dedicated OS thread grabs MJPEG-compressed frames straight off the
//! V4L2 device (the camera compresses in hardware, so the host only
//! shuttles bytes) and publishes them on a `tokio::sync::broadcast`
//! channel. The HTTP layer serves `GET /video` as a
//! `multipart/x-mixed-replace` stream from that channel — browsers
//! render it directly in an `<img>` tag, and OpenCV/FFmpeg clients
//! (bebop-vision) read the same URL as a capture source.
//!
//! Consumers subscribe independently; a slow client only lags itself
//! (broadcast drops to the newest frame on `Lagged`). The capture thread
//! is best-effort: if the camera unplugs, we retry with backoff and the
//! endpoint keeps serving the last-known state (no frames) rather than
//! taking anything else down. Video is explicitly *not* safety-critical —
//! the control loop never touches this module.
//!
//! # OBSBOT Tiny 2 PTZ — recon notes (2026-08-30, measured on the robot)
//!
//! The gimbal rides *standard UVC pan/tilt controls* on the Camera
//! Terminal of `/dev/video0` — the same device node this module already
//! owns. There is no HID interface and no vendor protocol needed for
//! PTZ (a vendor extension unit exists but only for extras like the
//! built-in AI tracking, which does **not** fight our commands: the
//! gimbal holds commanded positions indefinitely with zero drift).
//!
//! Control map (UVC units are 1/3600 of a degree, i.e. 3600 units = 1°):
//!
//! | Control             | V4L2 id     | Range                  | Step  |
//! |---------------------|-------------|------------------------|-------|
//! | `pan_absolute`      | 0x009a0908  | ±468000 = **±130°**    | 1°    |
//! | `tilt_absolute`     | 0x009a0909  | ±324000 = **±90°**     | 1°    |
//! | `pan_speed`         | 0x009a0920  | -1..160 (default 20)   | 1     |
//! | `tilt_speed`        | 0x009a0921  | -1..120 (default 20)   | 1     |
//! | `zoom_absolute`     | 0x009a090d  | 0..100                 | 1     |
//!
//! Measured behavior (default speeds, 30° pan):
//!   * motion time ~0.45 s (~65°/s), frames stable ~0.5-0.7 s after a
//!     local ioctl (1.06 s end-to-end via ssh + v4l2-ctl, which adds
//!     connection + sudo overhead)
//!   * read-back is exact: commanded 108000 reads back 108000
//!   * quirk: tilt readback shifted 5°→0° during a pan move (suspected
//!     horizon compensation) — always read the actual pose, never assume
//!
//! Operator diagnostics on the robot (after `install-jetson.sh` lays
//! down v4l-utils):
//!
//! ```text
//! v4l2-ctl -d /dev/video0 --list-ctrls
//! v4l2-ctl -d /dev/video0 --set-ctrl pan_absolute=108000   # +30°
//! ```

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::broadcast;
use tracing::{error, info, warn};

use crate::config::VideoConfig;
use crate::ptz::Ptz;

/// One camera frame as captured (already JPEG-compressed bytes).
pub struct Frame {
    pub jpeg: Arc<Vec<u8>>,
    /// Host capture timestamp in microseconds since the Unix epoch.
    pub ts_us: u64,
    /// Monotonic sequence number since hub start.
    pub seq: u64,
    /// Gimbal pose at grab time (degrees, actual read-back — see
    /// [`crate::ptz`]). Served as `X-Pan-Deg` / `X-Tilt-Deg` per-part
    /// headers on `GET /video` so consumers can align frames with
    /// headings without a separate telemetry join.
    pub pan_deg: f32,
    pub tilt_deg: f32,
}

/// Shared handle to the capture thread's broadcast channel.
#[derive(Clone)]
pub struct VideoHub {
    tx: broadcast::Sender<Arc<Frame>>,
    /// Gimbal controls on the same device (second fd, controls only).
    /// Shared with the WS handler (set), the telemetry pump (state) and
    /// the capture thread (per-frame pose stamping).
    pub ptz: Arc<Ptz>,
}

impl VideoHub {
    /// Spawn the capture thread for `cfg`. Errors opening the device are
    /// logged and retried from inside the thread — `spawn` itself always
    /// succeeds so a missing camera never blocks robot startup.
    pub fn spawn(cfg: VideoConfig) -> VideoHub {
        let (tx, _) = broadcast::channel(cfg.queue_depth());
        let ptz = Arc::new(Ptz::open(&cfg.device));
        let hub = VideoHub { tx: tx.clone(), ptz: ptz.clone() };
        std::thread::Builder::new()
            .name("video-capture".into())
            .spawn(move || capture_loop(cfg, tx, ptz))
            .expect("spawn video-capture thread");
        hub
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<Frame>> {
        self.tx.subscribe()
    }

    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

fn capture_loop(cfg: VideoConfig, tx: broadcast::Sender<Arc<Frame>>, ptz: Arc<Ptz>) {
    let mut seq: u64 = 0;
    let mut backoff = Duration::from_secs(1);
    loop {
        match run_camera(&cfg, &mut seq, &tx, &ptz) {
            Ok(()) => unreachable!("camera loop never returns Ok"),
            Err(e) => error!(error = %e, "camera capture failed; retrying"),
        }
        if tx.receiver_count() > 0 {
            warn!("camera dropped while clients were streaming");
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(Duration::from_secs(10));
        if tx.receiver_count() == 0 && backoff >= Duration::from_secs(10) {
            backoff = Duration::from_secs(2);
        }
    }
}

fn run_camera(
    cfg: &VideoConfig,
    seq: &mut u64,
    tx: &broadcast::Sender<Arc<Frame>>,
    ptz: &Ptz,
) -> anyhow::Result<()> {
    use rscam::Camera;

    let mut camera = Camera::new(&cfg.device)?;
    let interval = (1u32, cfg.fps.max(1));
    let config = rscam::Config {
        interval,
        resolution: (cfg.width, cfg.height),
        format: b"MJPG",
        ..Default::default()
    };
    camera.start(&config)?;
    info!(
        device = %cfg.device,
        width = cfg.width,
        height = cfg.height,
        fps = cfg.fps,
        "camera streaming (MJPEG passthrough)"
    );

    let frame_interval = Duration::from_secs_f64(1.0 / cfg.fps.max(1) as f64);
    let mut last_emit = std::time::Instant::now();
    loop {
        let frame = camera.capture()?;
        let jpeg = Arc::new(frame.to_vec());
        let (pan_deg, tilt_deg) = ptz.pose();
        *seq += 1;
        let frame = Arc::new(Frame {
            jpeg,
            ts_us: now_us(),
            seq: *seq,
            pan_deg,
            tilt_deg,
        });
        // send() failing just means no subscribers right now.
        let _ = tx.send(frame);
        // Pace roughly to the configured rate so a 60 fps camera doesn't
        // flood a 15 fps consumer path (clients also drop on lag).
        let elapsed = last_emit.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
        last_emit = std::time::Instant::now();
    }
}
