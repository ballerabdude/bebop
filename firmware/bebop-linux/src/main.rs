//! Bebop V2 Linux runtime.
//!
//! Multi-mode server: starts in [`mode::Mode::Idle`], accepts mode
//! transitions / motor commands / telemetry subscriptions over a
//! protobuf-over-WebSocket API exposed by [`server`].
//!
//! Every motor TX flows through [`safety::Supervisor`], which clamps to
//! per-joint hard limits, runs a feedback watchdog, and latches an E-STOP
//! on any breach. The supervisor's `Drop` impl disables every motor on
//! every bus before the process exits.

use anyhow::{Context, Result};
use bebop_linux::config::{ImuSource, RobotConfig};
use bebop_linux::imu;
use bebop_linux::imu_serial;
use bebop_linux::mode::Mode;
use bebop_linux::policy_capture;
use bebop_linux::policy_control;
use bebop_linux::policy_io;
use bebop_linux::policy_runner::PolicyRunner;
use bebop_linux::realtime;
use bebop_linux::safety::power_monitor::spawn_power_monitor;
use bebop_linux::safety::supervisor::spawn_rx_threads;
use bebop_linux::safety::{BusPool, Supervisor};
use bebop_linux::server;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Debug, Clone)]
struct Args {
    config: PathBuf,
    /// Path to the trained policy ONNX. If `None`, defaults to
    /// `<config_dir>/policy.onnx` (sibling of the joint YAML).
    policy: Option<PathBuf>,
    /// Directory to write operator-toggled observation/action MCAP
    /// captures into. `~/` is expanded; defaults to `~/bebop-captures`.
    /// See `crate::policy_capture` for the file naming scheme.
    capture_dir: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            config: PathBuf::from("config/bebop_v2.yaml"),
            policy: None,
            capture_dir: None,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let cli: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < cli.len() {
        match cli[i].as_str() {
            "--config" | "-c" => {
                if i + 1 < cli.len() {
                    args.config = PathBuf::from(&cli[i + 1]);
                    i += 1;
                }
            }
            "--policy" | "-p" => {
                if i + 1 < cli.len() {
                    args.policy = Some(PathBuf::from(&cli[i + 1]));
                    i += 1;
                }
            }
            "--capture-dir" => {
                if i + 1 < cli.len() {
                    args.capture_dir = Some(PathBuf::from(&cli[i + 1]));
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!(
                    "bebop-linux v2 runtime\n\
                     \n\
                     USAGE:\n    bebop-linux [OPTIONS]\n\
                     \n\
                     OPTIONS:\n  \
                       -c, --config <PATH>      Joint config YAML \
                                                 [default: config/bebop_v2.yaml]\n  \
                       -p, --policy <PATH>      Trained policy ONNX \
                                                 [default: <config_dir>/policy.onnx]\n  \
                           --capture-dir <DIR>  Where to write operator-toggled MCAP captures \
                                                 [default: ~/bebop-captures]\n  \
                       -h, --help               Print help\n"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    args
}

/// Resolve the policy ONNX path. The CLI override wins; otherwise we pick
/// `<config_dir>/policy.onnx` so the operator can ship the policy as a
/// drop-in next to `bebop_v2.yaml`.
fn resolve_policy_path(args: &Args) -> PathBuf {
    if let Some(p) = args.policy.as_ref() {
        return p.clone();
    }
    let cfg_dir = args
        .config
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    cfg_dir.join("policy.onnx")
}

/// Resolve the operator capture directory. The CLI override wins (with
/// `~` expanded); otherwise we default to `~/bebop-captures` so the
/// files land in the operator's home dir without root permissions.
/// Falls back to `./bebop-captures` if `$HOME` is unset (CI / containers).
fn resolve_capture_dir(args: &Args) -> PathBuf {
    let raw = args
        .capture_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("~/bebop-captures"));
    expand_tilde(&raw)
}

fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if s == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    path.to_path_buf()
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    info!("╔════════════════════════════════════════╗");
    info!("║    Bebop V2 Linux Runtime              ║");
    info!("║    SocketCAN + WS API + ONNX           ║");
    info!("╚════════════════════════════════════════╝");

    let args = parse_args();
    info!(config = %args.config.display(), "loading config");
    let cfg = Arc::new(
        RobotConfig::from_yaml(&args.config)
            .with_context(|| format!("load config from {}", args.config.display()))?,
    );
    info!(
        joints = cfg.num_joints(),
        buses = cfg.can_interfaces.len(),
        bind = %cfg.server.bind_addr,
        "config loaded"
    );

    // Open every CAN bus. Pre-flight check refuses ERROR-PASSIVE / BUS-OFF.
    let bus_pool = Arc::new(BusPool::open(&cfg.can_interfaces).context("open CAN buses")?);

    // Build the supervisor. Stays in scope for the lifetime of `main` so its
    // `Drop` disables every motor before we leave.
    let supervisor = Arc::new(Supervisor::new(cfg.clone(), bus_pool.clone()));

    // Spawn one OS thread per CAN bus to drain feedback frames.
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    // Shared latest IMU reading (always present; the I²C reader fills it
    // when an `imu:` block exists in the YAML, otherwise stays at default
    // and the telemetry builder reports `present = false` so the
    // operator UI hides the orientation card).
    let imu_shared = imu::new_shared();
    let imu_present = cfg.imu.is_some();
    let policy_io_shared = policy_io::new_shared();
    // Operator-toggled policy control (dry-run + MCAP capture). Owned
    // once here; cloned into the WS server (writer) and the policy
    // runner (reader / file-state owner).
    let policy_control_shared = policy_control::new_shared();
    // Pick the IMU backend: read the BNO directly over the Jetson's SPI
    // bus, or consume pre-fused frames from the Teensy `imu_bridge` over
    // USB serial. Both fill `imu_shared` with the same body-frame snapshot.
    let imu_handle = cfg.imu.as_ref().and_then(|imu_cfg| match imu_cfg.source {
        ImuSource::Spi => {
            imu::spawn_imu_thread(imu_cfg.clone(), shutdown_flag.clone(), imu_shared.clone())
        }
        ImuSource::Serial => imu_serial::spawn_imu_serial_thread(
            imu_cfg.clone(),
            shutdown_flag.clone(),
            imu_shared.clone(),
        ),
    });

    let rx_handles = spawn_rx_threads(supervisor.clone(), bus_pool.clone(), shutdown_flag.clone());

    // Spawn the power-board poller (no-op if `power:` is omitted from
    // the YAML). The handle is collected so we can join it on shutdown.
    let power_handle = supervisor
        .power_monitor()
        .and_then(|monitor| spawn_power_monitor(monitor, bus_pool.clone(), shutdown_flag.clone()));

    // Try to load the trained policy. Soft-fail: if the file is missing
    // or doesn't match the expected I/O contract, log a warning and
    // continue — DialIn / Idle still work without a policy on disk.
    //
    // The runner takes a clone of `imu_shared` so it can read the
    // latest body-frame quaternion + calibrated gyro on every tick.
    // The IMU thread is the sole writer; the runner and the telemetry
    // builder both clone independent reader handles.
    let policy_path = resolve_policy_path(&args);
    let capture_dir = resolve_capture_dir(&args);
    info!(dir = %capture_dir.display(), "capture: output directory resolved");

    // Spawn the off-the-hot-loop MCAP writer thread once at startup.
    // It lives for the whole process; the runner submits samples via
    // its `CaptureHandle` clone and the writer publishes capture status
    // (`capture_active` / `capture_path` / `capture_rows` /
    // `capture_dropped`) directly back into `policy_io_shared`.
    let (capture_handle, capture_join) =
        policy_capture::spawn_capture_thread(capture_dir.clone(), policy_io_shared.clone());

    let policy_io_for_runner = policy_io_shared.clone();
    let policy_control_for_runner = policy_control_shared.clone();
    let policy_runner: Arc<Mutex<Option<PolicyRunner>>> =
        match PolicyRunner::new(
            supervisor.clone(),
            imu_shared.clone(),
            policy_io_for_runner,
            policy_control_for_runner,
            capture_handle.clone(),
            &policy_path,
        ) {
            Ok(pr) => {
                if let Ok(mut g) = policy_io_shared.lock() {
                    g.set_present(true);
                }
                info!(model = %policy_path.display(), "policy loaded; RunPolicy mode is available");
                Arc::new(Mutex::new(Some(pr)))
            }
            Err(e) => {
                if let Ok(mut g) = policy_io_shared.lock() {
                    g.set_present(false);
                }
                warn!(
                    model = %policy_path.display(),
                    error = %e,
                    "policy not loaded; RunPolicy mode will be a no-op"
                );
                Arc::new(Mutex::new(None))
            }
        };

    // Periodic supervisor tick: hold-cycle TX in DialIn mode, RunPolicy
    // inference + TX in RunPolicy mode, watchdog every cycle.
    let sup_tick = supervisor.clone();
    let pr_tick = policy_runner.clone();
    let shutdown_tick = shutdown_flag.clone();

    /// Control-loop period: 100 Hz. Also the per-cycle work budget — if
    /// the body takes longer than this the loop can't sustain 100 Hz.
    const TICK_PERIOD: Duration = Duration::from_millis(10);
    /// A wake-to-wake interval longer than this counts as a "late" tick
    /// (1.5× the period) — the loop slipped behind schedule.
    const TICK_LATE_THRESHOLD: Duration = Duration::from_millis(15);
    /// How often to emit the loop-health summary.
    const HEALTH_LOG_INTERVAL: Duration = Duration::from_secs(5);
    /// `SCHED_FIFO` priority for the control loop. Mid-range (1..=99) so
    /// it preempts all timeshare work and ordinary kernel threads, while
    /// leaving headroom under the highest IRQ-handler bands.
    const CONTROL_LOOP_RT_PRIO: i32 = 80;

    // The 100 Hz control loop runs on its own OS thread — NOT a tokio
    // task — so it never competes with the async WS/telemetry workers for
    // a runtime worker and isn't at the mercy of tokio's timer wheel. We
    // raise it to SCHED_FIFO and drive it from an absolute CLOCK_MONOTONIC
    // deadline (`realtime::Deadline`) so wakes are timely and drift-free.
    let tick_handle = std::thread::Builder::new()
        .name("control-loop".to_string())
        .spawn(move || {
            match realtime::set_current_thread_fifo(CONTROL_LOOP_RT_PRIO) {
                Ok(()) => info!(
                    target: "bebop_linux::loop",
                    prio = CONTROL_LOOP_RT_PRIO,
                    "control loop: SCHED_FIFO real-time scheduling enabled"
                ),
                Err(e) => warn!(
                    target: "bebop_linux::loop",
                    error = %e,
                    "control loop: could not set SCHED_FIFO (needs CAP_SYS_NICE / root); \
                     running at default SCHED_OTHER — expect timing jitter under load"
                ),
            }

            let mut deadline = realtime::Deadline::start(TICK_PERIOD);

            // --- 100 Hz loop-health instrumentation -----------------------
            // We can't tell from the outside whether the control loop is
            // actually hitting 100 Hz, so measure it here: per cycle we
            // track the body's work time and the wake-to-wake interval,
            // then summarize once per HEALTH_LOG_INTERVAL. A clean loop
            // logs at `info`; any missed deadline (work over budget, or a
            // wake that slipped late) escalates to `warn` for journalctl.
            let mut last_wake: Option<Instant> = None;
            let mut window_start = Instant::now();
            let mut cycles: u64 = 0;
            let mut overruns: u64 = 0; // body work exceeded the tick budget
            let mut late_wakes: u64 = 0; // wake-to-wake interval slipped late
            let mut sum_work = Duration::ZERO;
            let mut max_work = Duration::ZERO;
            let mut max_interval = Duration::ZERO;

            loop {
                deadline.advance();
                deadline.sleep_until();
                if shutdown_tick.load(Ordering::SeqCst) {
                    break;
                }
                let wake = Instant::now();
                if let Some(prev) = last_wake {
                    let interval = wake.duration_since(prev);
                    if interval > max_interval {
                        max_interval = interval;
                    }
                    if interval > TICK_LATE_THRESHOLD {
                        late_wakes += 1;
                    }
                }
                last_wake = Some(wake);

                sup_tick.run_watchdog();
                sup_tick.tick_telemetry_probe();
                // Wheeled chassis: apply the operator twist + integrate
                // odometry every tick (no-op on the legged humanoid, which
                // has no `drive:` config).
                sup_tick.tick_drive();
                if sup_tick.mode() == Mode::DialIn {
                    sup_tick.tick_dial_in_hold();
                }
                // Always pump the policy runner: it has internal mode +
                // estop gates, runs inference only in RunPolicy, and on
                // mode exit it resets the controller + publishes
                // `active=false` to the shared `PolicyIoSnapshot`. Skipping
                // the call in non-RunPolicy modes meant the snapshot stayed
                // at `active=true` with the last observation/action
                // forever, so the operator app's history sparklines kept
                // appending duplicate "live" samples and the real history
                // was overwritten in the ring buffer.
                if let Ok(mut g) = pr_tick.lock() {
                    if let Some(pr) = g.as_mut() {
                        pr.tick();
                    }
                }

                let work = wake.elapsed();
                cycles += 1;
                sum_work += work;
                if work > max_work {
                    max_work = work;
                }
                if work > TICK_PERIOD {
                    overruns += 1;
                }

                let window = window_start.elapsed();
                if window >= HEALTH_LOG_INTERVAL {
                    let hz = cycles as f64 / window.as_secs_f64();
                    let mean_work_us = if cycles > 0 {
                        (sum_work.as_micros() as u64) / cycles
                    } else {
                        0
                    };
                    if overruns > 0 || late_wakes > 0 {
                        warn!(
                            target: "bebop_linux::loop",
                            hz = format_args!("{hz:.1}"),
                            cycles,
                            overruns,
                            late_wakes,
                            mean_work_us,
                            max_work_us = max_work.as_micros() as u64,
                            max_interval_us = max_interval.as_micros() as u64,
                            "100 Hz control loop missing its deadline (target 100 Hz / 10 ms)"
                        );
                    } else {
                        info!(
                            target: "bebop_linux::loop",
                            hz = format_args!("{hz:.1}"),
                            cycles,
                            mean_work_us,
                            max_work_us = max_work.as_micros() as u64,
                            max_interval_us = max_interval.as_micros() as u64,
                            "100 Hz control loop healthy"
                        );
                    }
                    window_start = Instant::now();
                    cycles = 0;
                    overruns = 0;
                    late_wakes = 0;
                    sum_work = Duration::ZERO;
                    max_work = Duration::ZERO;
                    max_interval = Duration::ZERO;
                }
            }
        })
        .expect("spawn control-loop thread");

    // Run the WS server in its own task so we can also wait for ctrl-c.
    let server_sup = supervisor.clone();
    let server_imu = imu_shared.clone();
    let server_policy_io = policy_io_shared.clone();
    let server_policy_control = policy_control_shared.clone();
    let server_capture_dir = capture_dir.clone();
    let bind_addr = cfg.server.bind_addr.clone();
    let server_handle = tokio::spawn(async move {
        if let Err(e) = server::run_server(
            server_sup,
            server_imu,
            imu_present,
            server_policy_io,
            server_policy_control,
            server_capture_dir,
            &bind_addr,
        )
        .await
        {
            error!(error = %e, "server task exited with error");
        }
    });

    info!("ready: mode = Idle");

    // Wait for shutdown signal. SIGINT first; ignore SIGTERM beyond logging
    // because the Drop impl on `supervisor` will fire either way.
    tokio::select! {
        res = signal::ctrl_c() => {
            if let Err(e) = res {
                warn!(error = %e, "ctrl-c handler error");
            }
            info!("ctrl-c received; shutting down");
        }
        _ = server_handle => {
            warn!("server task ended; shutting down");
        }
    }

    // Cooperative shutdown: stop the tick + RX threads. The control loop
    // is now a dedicated OS thread that polls `shutdown_flag` once per
    // tick, so it exits within ~one period (10 ms) of the flag flipping —
    // join it rather than aborting a task.
    shutdown_flag.store(true, Ordering::SeqCst);
    let _ = tick_handle.join();
    for h in rx_handles {
        let _ = h.join();
    }
    if let Some(h) = power_handle {
        let _ = h.join();
    }
    if let Some(h) = imu_handle {
        let _ = h.join();
    }

    // Drain + finalize any open MCAP capture before we drop the
    // supervisor. Doing this before the supervisor's `Drop` keeps the
    // capture file from being closed *while* the disable-on-shutdown
    // command frames are in flight — the last samples in the file are
    // therefore the last meaningful policy ticks, not a noisy tail of
    // "motors disabling".
    capture_handle.shutdown();
    let _ = capture_join.join();

    // `supervisor` Arc ends here; its inner Drop sends Disable to every motor.
    info!("bye");
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,bebop_linux=debug"));
    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .init();
}
