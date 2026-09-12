//! bebop-vision service control.
//!
//! The Python vision/recorder stack (`bebop-vision/main.py`) runs as its
//! own systemd unit rather than as a child of this process: `bebop-linux`
//! runs sandboxed (`ProtectHome=true`, `PrivateTmp=true`, no camera device
//! policy), so a child would inherit a sandbox that can't reach the venv,
//! the Orbbec `/dev` nodes, or the shared `/tmp/navd_recorder.lock`. A
//! separate `bebop-vision.service` avoids all of that and gives the service
//! its own restart policy.
//!
//! This module owns the bridge: the WS handler calls [`VisionShared::request`]
//! with the operator's intent, a background thread runs `systemctl` and
//! re-polls the unit, and [`VisionShared::snapshot`] feeds `VisionState`
//! into telemetry. Requests are asynchronous by design — a `systemctl`
//! start can take a second, and the reader task must not block on it.

use anyhow::{Context, Result};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// systemd unit that runs the Python bebop-vision recorder.
pub const VISION_SERVICE: &str = "bebop-vision.service";

/// Human-readable run mode baked into the unit. Surfaced in telemetry so
/// the UI can label the control without hard-coding the ExecStart line.
pub const VISION_MODE: &str = "recorder";

/// How often to re-poll systemd when idle. Keeps telemetry fresh without
/// hammering PID1 with `systemctl show`.
const POLL_PERIOD: Duration = Duration::from_millis(1500);

/// Last observed state of the `bebop-vision.service` unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionSnapshot {
    /// Unit is installed (`LoadState=loaded`).
    pub present: bool,
    /// Unit is `active` (the Python process is up).
    pub running: bool,
    /// Raw systemd `ActiveState`.
    pub state: String,
    /// systemd `SubState`, with `Result` appended when it isn't "success".
    pub detail: String,
    /// Unit name (constant, echoed for the UI).
    pub service: String,
    /// Run mode label (constant).
    pub mode: String,
    /// Most recent `systemctl` failure, or empty. Cleared once the unit
    /// reports running again.
    pub last_error: String,
}

impl Default for VisionSnapshot {
    fn default() -> Self {
        Self {
            present: false,
            running: false,
            state: String::new(),
            detail: String::new(),
            service: VISION_SERVICE.to_string(),
            mode: VISION_MODE.to_string(),
            last_error: String::new(),
        }
    }
}

/// Shared view of the vision service, cloned into the WS server and the
/// telemetry builders.
#[derive(Clone)]
pub struct VisionShared {
    state: Arc<Mutex<VisionSnapshot>>,
    tx: Sender<bool>,
}

impl VisionShared {
    /// Most recent snapshot. Cheap: a mutex-guarded clone of small strings.
    pub fn snapshot(&self) -> VisionSnapshot {
        self.state.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Queue a start (`true`) / stop (`false`). Non-blocking; the worker
    /// thread runs `systemctl` and refreshes the snapshot.
    pub fn request(&self, enabled: bool) {
        let _ = self.tx.send(enabled);
    }
}

/// Spawn the background systemd poller / command worker.
///
/// Does an immediate probe so the first telemetry frames carry a real
/// value on a service that's already running, then loops: handle any
/// queued request, then re-poll. Exits when `shutdown` flips.
pub fn spawn_vision_supervisor(shutdown: Arc<AtomicBool>) -> VisionShared {
    let state = Arc::new(Mutex::new(VisionSnapshot::default()));
    let (tx, rx) = mpsc::channel::<bool>();
    let worker_state = state.clone();
    thread::Builder::new()
        .name("vision-service".to_string())
        .spawn(move || {
            refresh_into(&worker_state);
            loop {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                match rx.recv_timeout(POLL_PERIOD) {
                    Ok(enabled) => {
                        let verb = if enabled { "start" } else { "stop" };
                        match run_systemctl(verb) {
                            Ok(()) => clear_error(&worker_state),
                            Err(e) => store_error(&worker_state, format!("{e:#}")),
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
                refresh_into(&worker_state);
            }
        })
        .expect("spawn vision-service thread");
    VisionShared { state, tx }
}

fn with_snapshot<F: FnOnce(&mut VisionSnapshot)>(state: &Mutex<VisionSnapshot>, f: F) {
    if let Ok(mut g) = state.lock() {
        f(&mut g);
    }
}

fn store_error(state: &Mutex<VisionSnapshot>, message: String) {
    with_snapshot(state, |s| s.last_error = message);
}

fn clear_error(state: &Mutex<VisionSnapshot>) {
    with_snapshot(state, |s| s.last_error.clear());
}

/// Run `systemctl <verb> bebop-vision.service`, surfacing stderr on failure.
fn run_systemctl(verb: &str) -> Result<()> {
    let out = Command::new("systemctl")
        .arg(verb)
        .arg(VISION_SERVICE)
        .output()
        .with_context(|| format!("spawn `systemctl {verb} {VISION_SERVICE}`"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        let detail = if stderr.is_empty() {
            format!("exit status {}", out.status)
        } else {
            stderr
        };
        anyhow::bail!("systemctl {verb} {VISION_SERVICE} failed: {detail}");
    }
    Ok(())
}

/// `systemctl show` the unit and fold the result into `state`, preserving
/// `last_error` unless the unit is now running.
fn refresh_into(state: &Mutex<VisionSnapshot>) {
    match query_state() {
        Ok(snap) => with_snapshot(state, |s| {
            s.present = snap.present;
            s.running = snap.running;
            s.state = snap.state;
            s.detail = snap.detail;
            if s.running {
                s.last_error.clear();
            }
        }),
        Err(e) => with_snapshot(state, |s| {
            s.present = false;
            s.running = false;
            s.state.clear();
            s.detail.clear();
            s.last_error = format!("{e:#}");
        }),
    }
}

fn query_state() -> Result<VisionSnapshot> {
    let out = Command::new("systemctl")
        .args([
            "show",
            VISION_SERVICE,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "Result",
        ])
        .output()
        .context("spawn `systemctl show`")?;
    if !out.status.success() {
        anyhow::bail!("systemctl show {VISION_SERVICE} failed");
    }
    Ok(parse_show(&String::from_utf8_lossy(&out.stdout)))
}

/// Parse the `key=value` lines from `systemctl show` into a snapshot. Split
/// out so it can be unit-tested without a running systemd.
fn parse_show(text: &str) -> VisionSnapshot {
    let mut load = "";
    let mut active = "";
    let mut sub = "";
    let mut result = "";
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "LoadState" => load = value,
            "ActiveState" => active = value,
            "SubState" => sub = value,
            "Result" => result = value,
            _ => {}
        }
    }
    let present = load == "loaded";
    let running = active == "active";
    let detail = if result.is_empty() || result == "success" {
        sub.to_string()
    } else {
        format!("{sub} ({result})")
    };
    VisionSnapshot {
        present,
        running,
        state: active.to_string(),
        detail,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_show_reads_active_unit() {
        let snap =
            parse_show("LoadState=loaded\nActiveState=active\nSubState=running\nResult=success\n");
        assert!(snap.present);
        assert!(snap.running);
        assert_eq!(snap.state, "active");
        assert_eq!(snap.detail, "running");
        assert_eq!(snap.service, VISION_SERVICE);
        assert_eq!(snap.mode, VISION_MODE);
    }

    #[test]
    fn parse_show_flags_missing_unit() {
        let snap = parse_show("LoadState=not-found\nActiveState=inactive\nSubState=dead\n");
        assert!(!snap.present);
        assert!(!snap.running);
        assert_eq!(snap.state, "inactive");
        assert_eq!(snap.detail, "dead");
    }

    #[test]
    fn parse_show_surfaces_failure_result() {
        let snap =
            parse_show("LoadState=loaded\nActiveState=failed\nSubState=failed\nResult=exit-code\n");
        assert!(snap.present);
        assert!(!snap.running);
        assert_eq!(snap.detail, "failed (exit-code)");
    }
}
