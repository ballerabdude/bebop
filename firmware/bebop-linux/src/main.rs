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
use bebop_linux::config::RobotConfig;
use bebop_linux::imu;
use bebop_linux::mode::Mode;
use bebop_linux::policy_runner::PolicyRunner;
use bebop_linux::safety::power_monitor::spawn_power_monitor;
use bebop_linux::safety::supervisor::spawn_rx_threads;
use bebop_linux::safety::{BusPool, Supervisor};
use bebop_linux::server;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Debug, Clone)]
struct Args {
    config: PathBuf,
    /// Path to the trained policy ONNX. If `None`, defaults to
    /// `<config_dir>/policy.onnx` (sibling of the joint YAML).
    policy: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            config: PathBuf::from("config/bebop_v2.yaml"),
            policy: None,
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
            "--help" | "-h" => {
                println!(
                    "bebop-linux v2 runtime\n\
                     \n\
                     USAGE:\n    bebop-linux [OPTIONS]\n\
                     \n\
                     OPTIONS:\n  \
                       -c, --config <PATH>   Joint config YAML \
                                              [default: config/bebop_v2.yaml]\n  \
                       -p, --policy <PATH>   Trained policy ONNX \
                                              [default: <config_dir>/policy.onnx]\n  \
                       -h, --help            Print help\n"
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
    let imu_handle = cfg.imu.as_ref().and_then(|imu_cfg| {
        imu::spawn_imu_thread(imu_cfg.clone(), shutdown_flag.clone(), imu_shared.clone())
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
    let policy_path = resolve_policy_path(&args);
    let policy_runner: Arc<Mutex<Option<PolicyRunner>>> =
        match PolicyRunner::new(supervisor.clone(), &policy_path) {
            Ok(pr) => {
                info!(model = %policy_path.display(), "policy loaded; RunPolicy mode is available");
                Arc::new(Mutex::new(Some(pr)))
            }
            Err(e) => {
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
    let tick_handle = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(10));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            if shutdown_tick.load(Ordering::SeqCst) {
                break;
            }
            sup_tick.run_watchdog();
            match sup_tick.mode() {
                Mode::DialIn => sup_tick.tick_dial_in_hold(),
                Mode::RunPolicy => {
                    if let Ok(mut g) = pr_tick.lock() {
                        if let Some(pr) = g.as_mut() {
                            pr.tick();
                        }
                    }
                }
                Mode::Idle => {}
            }
        }
    });

    // Run the WS server in its own task so we can also wait for ctrl-c.
    let server_sup = supervisor.clone();
    let server_imu = imu_shared.clone();
    let bind_addr = cfg.server.bind_addr.clone();
    let server_handle = tokio::spawn(async move {
        if let Err(e) = server::run_server(server_sup, server_imu, imu_present, &bind_addr).await {
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

    // Cooperative shutdown: stop the tick + RX threads.
    shutdown_flag.store(true, Ordering::SeqCst);
    tick_handle.abort();
    for h in rx_handles {
        let _ = h.join();
    }
    if let Some(h) = power_handle {
        let _ = h.join();
    }
    if let Some(h) = imu_handle {
        let _ = h.join();
    }

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
