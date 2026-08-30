//! Camera gimbal (PTZ) control via standard UVC pan/tilt controls.
//!
//! The OBSBOT Tiny 2's gimbal rides the Camera Terminal's UVC
//! `pan_absolute` / `tilt_absolute` controls on the same `/dev/video*`
//! node the video hub streams from — there is no separate HID device
//! (see the recon table in [`crate::video`]). V4L2 allows a second fd on
//! the same node for control access, so this module opens its own handle
//! and never touches streaming. All ioctls are quick (µs); a
//! [`std::sync::Mutex`] is enough to serialize callers between the WS
//! handler (set), the telemetry pump (state) and the capture thread
//! (per-frame pose stamping).
//!
//! Units: UVC control values are 1/3600 of a degree (3600 = 1°); this
//! module speaks degrees at its API and converts at the edges.
//!
//! Robustness model: [`Ptz`] is created once at boot and is always
//! usable — if the camera is missing it returns `Err` / last-known
//! values, and every call retries opening the device so a replugged
//! camera recovers on the next command without a firmware restart.

use std::sync::Mutex;

use tracing::{info, warn};

/// V4L2 control ids on the UVC Camera Terminal (recon: `v4l2-ctl
/// --list-ctrls` on the robot's OBSBOT Tiny 2).
const CID_PAN_ABSOLUTE: u32 = 0x009a_0908;
const CID_TILT_ABSOLUTE: u32 = 0x009a_0909;

/// `struct v4l2_control` is 8 bytes → `_IOWR('V', 27, 8)`.
const VIDIOC_G_CTRL: libc::c_ulong = 0xc008_561b;
/// `_IOWR('V', 28, 8)`.
const VIDIOC_S_CTRL: libc::c_ulong = 0xc008_561c;
/// `struct v4l2_queryctrl` is 68 bytes → `_IOWR('V', 36, 68)`. Note the
/// ioctl *number*: nr 36 is QUERYCTRL; nr 56 is the different
/// `VIDIOC_QUERY_EXT_CTRL` (152-byte struct) — passing nr 56 with a
/// 68-byte size is a size mismatch the kernel answers with ENOTTY.
const VIDIOC_QUERYCTRL: libc::c_ulong = 0xc044_5624;

/// Degrees the actual pose may differ from the commanded target before
/// the gimbal counts as settled (~1° ≈ the control step).
const SETTLE_TOL_DEG: f32 = 1.0;

/// Minimum spacing between (re)open attempts while the camera is
/// unreachable. Without this the per-frame `pose()` stamping (30 Hz),
/// telemetry (30 Hz) and any WS command each retry `open()` — a broken
/// camera turns into an open/ioctl storm (and previously leaked a fd
/// per attempt: "too many open files").
const OPEN_RETRY_PERIOD: std::time::Duration = std::time::Duration::from_secs(2);

#[repr(C)]
struct V4l2Control {
    id: u32,
    value: i32,
}

/// Field order per `struct v4l2_queryctrl` in `linux/videodev2.h` —
/// `name[32]` sits between `type` and `minimum` (a subtle layout the
/// ENOTTY + garbage-range bug came from).
#[repr(C)]
struct V4l2QueryCtrl {
    id: u32,
    ctrl_type: u32,
    name: [u8; 32],
    minimum: i32,
    maximum: i32,
    step: u32,
    default_value: i32,
    flags: u32,
    reserved: [u32; 2],
}

#[derive(Clone, Copy)]
struct Axis {
    min: i32,
    max: i32,
}

impl Axis {
    /// Degrees → clamped UVC units (3600 per degree).
    fn units(&self, deg: f32) -> i32 {
        ((deg * 3600.0).round() as i32).clamp(self.min, self.max)
    }
}

struct PtzInner {
    fd: i32,
    pan: Axis,
    tilt: Axis,
    /// Last commanded target, degrees.
    target_pan: f32,
    target_tilt: f32,
    /// Last successfully read actual pose, degrees.
    last_pan: f32,
    last_tilt: f32,
}

/// The control fd is closed exactly once per open, on any exit path:
/// explicit failure in [`Ptz::try_open`], or replacement/error-drop in
/// [`Ptz::with_inner`] clearing the `Option`. Without this the per-frame
/// `pose()` stamping + telemetry polling leaked a fd per failed attempt
/// and the process hit "too many open files" within minutes.
impl Drop for PtzInner {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// Thread-safe camera gimbal handle. Cheap to clone via [`Arc`].
pub struct Ptz {
    device: String,
    inner: Mutex<Option<PtzInner>>,
    /// Earliest a failed open may be retried (see `OPEN_RETRY_PERIOD`).
    /// Stored outside `inner` so it survives the handle being dropped on
    /// error. Guards the open/ioctl storm when the camera is absent.
    next_open: std::sync::Mutex<std::time::Instant>,
}

impl Ptz {
    /// Open the gimbal controls on `device`. Never fails: a missing
    /// camera is logged and retried (rate-limited) on subsequent calls,
    /// so robot startup never blocks on the camera (same philosophy as
    /// the video hub).
    pub fn open(device: &str) -> Ptz {
        let ptz = Ptz {
            device: device.to_string(),
            inner: Mutex::new(None),
            next_open: std::sync::Mutex::new(std::time::Instant::now()),
        };
        match ptz.try_open() {
            Ok(inner) => {
                info!(device = %ptz.device, "camera gimbal ready (UVC pan/tilt)");
                *ptz.inner.lock().expect("ptz mutex poisoned") = Some(inner);
            }
            Err(e) => {
                warn!(device = %ptz.device, error = %e, "camera gimbal not available yet; will retry on use")
            }
        }
        ptz
    }

    fn try_open(&self) -> anyhow::Result<PtzInner> {
        let path = std::ffi::CString::new(self.device.clone())?;
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if fd < 0 {
            anyhow::bail!("open {}: {}", self.device, std::io::Error::last_os_error());
        }
        let result = (|| -> anyhow::Result<PtzInner> {
            let pan = query_axis(fd, CID_PAN_ABSOLUTE)?;
            let tilt = query_axis(fd, CID_TILT_ABSOLUTE)?;
            let (pan_units, tilt_units) = (
                read_ctrl(fd, CID_PAN_ABSOLUTE)?,
                read_ctrl(fd, CID_TILT_ABSOLUTE)?,
            );
            Ok(PtzInner {
                fd,
                pan,
                tilt,
                target_pan: units_to_deg(pan_units),
                target_tilt: units_to_deg(tilt_units),
                last_pan: units_to_deg(pan_units),
                last_tilt: units_to_deg(tilt_units),
            })
        })();
        // `PtzInner::drop` closes the fd when the Ok value drops, but on
        // the error paths nothing owns it yet — close here or it leaks.
        if result.is_err() {
            unsafe {
                libc::close(fd);
            }
        }
        result
    }

    /// Run `f` with the inner state, transparently (re)opening the
    /// device if the previous handle died (camera unplugged/replugged).
    /// An io error from `f` drops the handle so the *next* call reopens
    /// — replug recovery without a firmware restart. Open attempts are
    /// rate-limited (`OPEN_RETRY_PERIOD`): pose stamping + telemetry
    /// call this at ~60 Hz combined, and a missing camera must not turn
    /// that into an open/ioctl storm.
    fn with_inner<T>(
        &self,
        f: impl FnOnce(&mut PtzInner) -> std::io::Result<T>,
    ) -> std::io::Result<T> {
        let mut guard = self.inner.lock().expect("ptz mutex poisoned");
        if guard.is_none() {
            {
                let mut next = self.next_open.lock().expect("ptz mutex poisoned");
                if std::time::Instant::now() < *next {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotConnected,
                        "camera control unavailable (backing off)",
                    ));
                }
                *next = std::time::Instant::now() + OPEN_RETRY_PERIOD;
            }
            match self.try_open() {
                Ok(inner) => {
                    info!(device = %self.device, "camera gimbal reconnected");
                    *guard = Some(inner);
                }
                Err(e) => {
                    return Err(std::io::Error::other(e));
                }
            }
        }
        let inner = guard.as_mut().expect("inner just ensured");
        match f(inner) {
            Ok(v) => Ok(v),
            Err(e) => {
                warn!(error = %e, "camera gimbal control failed; dropping handle");
                *guard = None;
                Err(e)
            }
        }
    }

    /// Command an absolute pose in degrees. Returns the clamped pose
    /// actually sent. Errors mean the camera is unreachable.
    pub fn set_pose(&self, pan_deg: f32, tilt_deg: f32) -> anyhow::Result<(f32, f32)> {
        let (pan_units, tilt_units) = self
            .with_inner(|inner| {
                let pan_units = inner.pan.units(pan_deg);
                let tilt_units = inner.tilt.units(tilt_deg);
                write_ctrl(inner.fd, CID_PAN_ABSOLUTE, pan_units)?;
                write_ctrl(inner.fd, CID_TILT_ABSOLUTE, tilt_units)?;
                inner.target_pan = units_to_deg(pan_units);
                inner.target_tilt = units_to_deg(tilt_units);
                Ok((pan_units, tilt_units))
            })
            .map_err(|e| anyhow::anyhow!("set_ctrl: {e}"))?;
        let pose = (units_to_deg(pan_units), units_to_deg(tilt_units));
        info!(pan = pose.0, tilt = pose.1, "camera pose commanded");
        Ok(pose)
    }

    /// Actual pose in degrees (read back from the camera); falls back to
    /// the last known pose if the camera is momentarily unreadable.
    /// Before any successful read this is (0, 0) — the gimbal's
    /// power-on center.
    pub fn pose(&self) -> (f32, f32) {
        let _ = self.with_inner(|inner| {
            let pan = units_to_deg(read_ctrl(inner.fd, CID_PAN_ABSOLUTE)?);
            let tilt = units_to_deg(read_ctrl(inner.fd, CID_TILT_ABSOLUTE)?);
            inner.last_pan = pan;
            inner.last_tilt = tilt;
            Ok(())
        });
        let guard = self.inner.lock().expect("ptz mutex poisoned");
        match guard.as_ref() {
            Some(inner) => (inner.last_pan, inner.last_tilt),
            None => (0.0, 0.0),
        }
    }

    /// (pan, tilt, moving) for telemetry. `moving` compares the actual
    /// pose against the last commanded target with a ~1° tolerance, so
    /// consumers get a settle signal without a dedicated motion irq.
    pub fn state(&self) -> (f32, f32, bool) {
        let (pan, tilt) = self.pose();
        let moving = {
            let guard = self.inner.lock().expect("ptz mutex poisoned");
            match guard.as_ref() {
                Some(inner) => {
                    (inner.target_pan - pan).abs() > SETTLE_TOL_DEG
                        || (inner.target_tilt - tilt).abs() > SETTLE_TOL_DEG
                }
                None => false,
            }
        };
        (pan, tilt, moving)
    }
}

fn units_to_deg(units: i32) -> f32 {
    units as f32 / 3600.0
}

fn read_ctrl(fd: i32, id: u32) -> std::io::Result<i32> {
    let mut ctrl = V4l2Control { id, value: 0 };
    let rc = unsafe { libc::ioctl(fd, VIDIOC_G_CTRL, &mut ctrl as *mut V4l2Control) };
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(ctrl.value)
}

fn write_ctrl(fd: i32, id: u32, value: i32) -> std::io::Result<()> {
    let mut ctrl = V4l2Control { id, value };
    let rc = unsafe { libc::ioctl(fd, VIDIOC_S_CTRL, &mut ctrl as *mut V4l2Control) };
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn query_axis(fd: i32, id: u32) -> anyhow::Result<Axis> {
    let mut qc = V4l2QueryCtrl {
        id,
        ctrl_type: 0,
        minimum: 0,
        maximum: 0,
        step: 0,
        default_value: 0,
        flags: 0,
        name: [0; 32],
        reserved: [0; 2],
    };
    let rc = unsafe { libc::ioctl(fd, VIDIOC_QUERYCTRL, &mut qc as *mut V4l2QueryCtrl) };
    if rc < 0 {
        anyhow::bail!("QUERYCTRL({id:#x}): {}", std::io::Error::last_os_error());
    }
    if qc.maximum <= qc.minimum {
        anyhow::bail!("control {id:#x} reports empty range");
    }
    Ok(Axis {
        min: qc.minimum,
        max: qc.maximum,
    })
}
