//! Bebop Robot Linux Firmware
#![allow(dead_code)]
//!
//! This is the main control loop for the Bebop robot running on Linux.
//! It communicates with Robstride and ODrive motors via SocketCAN and
//! runs neural network policy inference using ONNX Runtime.
//!
//! Features:
//! - 50 Hz policy control loop (configurable)
//! - Robstride position control (legs)
//! - ODrive velocity control (wheels)
//! - UDP command input for teleoperation
//! - Safety checks (IMU-based upright detection)

mod can_interface;
mod config;
mod observation;
mod odrive;
mod policy;
mod robstride;
mod udp_command;

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use can_interface::CanInterface;
use config::{timing, JointCommand, MotorType, RobotConfig};
use observation::{scale_actions, ImuState, ObservationBuilder};
use odrive::ODriveMotorBus;
use policy::PolicyController;
use robstride::RobstrideMotorBus;
use udp_command::UdpCommandListener;

/// CLI arguments
#[derive(Debug, Clone)]
struct Args {
    /// Path to ONNX model file
    model_path: PathBuf,
    /// CAN interface name
    can_interface: String,
    /// UDP command port
    udp_port: u16,
    /// Control loop rate (Hz)
    control_rate: u64,
    /// Enable policy (false = passthrough mode)
    enable_policy: bool,
    /// Simulation mode (no real hardware)
    simulation: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("model.onnx"),
            can_interface: "can0".to_string(),
            udp_port: 10000,
            control_rate: timing::POLICY_RATE_HZ,
            enable_policy: true,
            simulation: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();

    let cli_args: Vec<String> = std::env::args().collect();
    let mut i = 1;

    while i < cli_args.len() {
        match cli_args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < cli_args.len() {
                    args.model_path = PathBuf::from(&cli_args[i + 1]);
                    i += 1;
                }
            }
            "--can" | "-c" => {
                if i + 1 < cli_args.len() {
                    args.can_interface = cli_args[i + 1].clone();
                    i += 1;
                }
            }
            "--port" | "-p" => {
                if i + 1 < cli_args.len() {
                    args.udp_port = cli_args[i + 1].parse().unwrap_or(10000);
                    i += 1;
                }
            }
            "--rate" | "-r" => {
                if i + 1 < cli_args.len() {
                    args.control_rate = cli_args[i + 1].parse().unwrap_or(50);
                    i += 1;
                }
            }
            "--no-policy" => {
                args.enable_policy = false;
            }
            "--sim" => {
                args.simulation = true;
            }
            "--help" | "-h" => {
                println!(
                    "Bebop Linux Firmware

USAGE:
    bebop-linux [OPTIONS]

OPTIONS:
    -m, --model <PATH>    Path to ONNX model file [default: model.onnx]
    -c, --can <IFACE>     CAN interface name [default: can0]
    -p, --port <PORT>     UDP command port [default: 10000]
    -r, --rate <HZ>       Control loop rate [default: 50]
    --no-policy           Disable policy, passthrough mode
    --sim                 Simulation mode (no hardware)
    -h, --help            Print help
"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    args
}

/// Control mode
#[derive(Debug, Clone, Copy, PartialEq)]
enum ControlMode {
    Idle,
    Passthrough,
    Policy,
}

/// Main robot controller
struct RobotController {
    config: RobotConfig,
    can: Option<CanInterface>,
    robstride_motors: RobstrideMotorBus,
    odrive_motors: ODriveMotorBus,
    policy: Option<PolicyController>,
    obs_builder: ObservationBuilder,
    udp_listener: UdpCommandListener,
    mode: ControlMode,
    step_count: u64,
    last_status_time: Instant,
}

impl RobotController {
    /// Create a new robot controller
    fn new(args: &Args) -> Result<Self> {
        let config = RobotConfig::bebop_wheeled();

        // Open CAN interface (unless in simulation mode)
        let can = if args.simulation {
            info!("Simulation mode - no CAN interface");
            None
        } else {
            Some(
                CanInterface::open(&args.can_interface)
                    .with_context(|| format!("Failed to open CAN interface: {}", args.can_interface))?,
            )
        };

        // Initialize Robstride motors
        let mut robstride_motors = RobstrideMotorBus::new();
        for joint in config.robstride_joints() {
            if let MotorType::Robstride(model) = joint.motor_type {
                robstride_motors.add_motor(joint.can_id, model);
                info!("Added Robstride motor {} ({})", joint.name, joint.can_id);
            }
        }

        // Initialize ODrive motors
        let mut odrive_motors = ODriveMotorBus::new();
        for joint in config.odrive_joints() {
            odrive_motors.add_motor(joint.can_id);
            info!("Added ODrive motor {} (node {})", joint.name, joint.can_id);
        }

        // Load policy (if enabled)
        let policy = if args.enable_policy {
            if args.model_path.exists() {
                Some(PolicyController::new(&args.model_path)?)
            } else {
                warn!("Model file not found: {}", args.model_path.display());
                warn!("Running in passthrough mode");
                None
            }
        } else {
            None
        };

        // Initialize observation builder
        let obs_builder = ObservationBuilder::new(config.num_joints());

        // Initialize UDP listener
        let udp_listener = UdpCommandListener::new(args.udp_port);

        Ok(Self {
            config,
            can,
            robstride_motors,
            odrive_motors,
            policy,
            obs_builder,
            udp_listener,
            mode: ControlMode::Idle,
            step_count: 0,
            last_status_time: Instant::now(),
        })
    }

    /// Initialize hardware
    fn init(&mut self) -> Result<()> {
        // Start UDP listener
        self.udp_listener.start()?;

        if let Some(can) = &self.can {
            // Enable Robstride motors
            info!("Enabling Robstride motors...");
            self.robstride_motors.enable_all(can)?;
            std::thread::sleep(Duration::from_millis(100));

            // Enable active reporting
            for motor in &self.robstride_motors.motors {
                motor.enable_active_reporting(can, 20)?;
            }
            std::thread::sleep(Duration::from_millis(100));

            // Enable ODrive motors
            info!("Enabling ODrive motors...");
            self.odrive_motors.enable_all(can)?;
            std::thread::sleep(Duration::from_millis(100));
        }

        self.mode = if self.policy.is_some() {
            ControlMode::Policy
        } else {
            ControlMode::Passthrough
        };

        info!("Robot initialized in {:?} mode", self.mode);
        Ok(())
    }

    /// Shutdown hardware
    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down...");

        if let Some(can) = &self.can {
            // Stop wheels first
            for motor in &self.odrive_motors.motors {
                let _ = motor.set_velocity(can, 0.0, 0.0);
            }

            std::thread::sleep(Duration::from_millis(50));

            // Disable motors
            let _ = self.odrive_motors.disable_all(can);
            let _ = self.robstride_motors.disable_all(can);
        }

        info!("Shutdown complete");
        Ok(())
    }

    /// Process received CAN frames
    fn process_can_frames(&mut self) {
        if let Some(can) = &self.can {
            for frame in can.drain() {
                self.robstride_motors.process_frame(&frame);
                self.odrive_motors.process_frame(&frame);
            }
        }
    }

    /// Update IMU state (placeholder - implement based on your IMU interface)
    fn update_imu(&mut self) {
        // TODO: Implement IMU reading
        // For now, assume upright orientation
        self.obs_builder.update_imu(ImuState {
            quaternion: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            linear_acceleration: [0.0, 0.0, 0.0],
        });
    }

    /// Run one control loop iteration
    fn step(&mut self, dt: f32) -> Result<()> {
        self.step_count += 1;

        // Process CAN feedback
        self.process_can_frames();

        // Update IMU
        self.update_imu();

        // Get velocity command from UDP
        let cmd_vel = self.udp_listener.get_command();
        self.obs_builder.update_cmd_vel(cmd_vel);

        // Update joint states in observation builder
        let joint_states: Vec<_> = self
            .robstride_motors
            .motors
            .iter()
            .map(|m| m.state.clone())
            .chain(self.odrive_motors.motors.iter().map(|m| m.state.clone()))
            .collect();
        self.obs_builder.update_joints(&joint_states);

        // Update velocity estimate
        self.obs_builder.update_velocity_estimate(dt);

        // Safety check
        let is_upright = self.obs_builder.is_upright();

        // Run policy if enabled and upright
        if self.mode == ControlMode::Policy && self.policy.is_some() {
            // Build observation
            let obs = self.obs_builder.build();

            // Run inference
            let policy = self.policy.as_mut().unwrap();
            let actions = policy.step(&obs)?;

            // Update last action for next observation
            self.obs_builder.update_last_action(&actions);

            // Scale actions
            let (leg_cmds, wheel_cmds) = scale_actions(&actions);

            // Send commands
            if let Some(can) = &self.can {
                if is_upright {
                    // Send leg commands (position control)
                    for (i, motor) in self.robstride_motors.motors.iter().enumerate() {
                        if i < leg_cmds.len() {
                            let joint = self.config.get_joint_by_index(i).unwrap();
                            let cmd = JointCommand {
                                position: leg_cmds[i] + joint.default_position,
                                velocity: 0.0,
                                torque: 0.0,
                                kp: joint.kp,
                                kd: joint.kd,
                            };
                            motor.send_command(can, &cmd)?;
                        }
                    }

                    // Send wheel commands (velocity control)
                    for (i, motor) in self.odrive_motors.motors.iter().enumerate() {
                        if i < wheel_cmds.len() {
                            motor.set_velocity(can, wheel_cmds[i], 0.0)?;
                        }
                    }
                } else {
                    // Robot fallen - stop wheels
                    for motor in &self.odrive_motors.motors {
                        motor.set_velocity(can, 0.0, 0.0)?;
                    }
                }
            }

            // Periodic logging
            if self.step_count % 50 == 0 {
                debug!(
                    "POLICY[{}]: legs=[{:.2},{:.2},{:.2},{:.2}] wheels=[{:.1},{:.1}] upright={}",
                    self.step_count,
                    leg_cmds.get(0).unwrap_or(&0.0),
                    leg_cmds.get(1).unwrap_or(&0.0),
                    leg_cmds.get(2).unwrap_or(&0.0),
                    leg_cmds.get(3).unwrap_or(&0.0),
                    wheel_cmds.get(0).unwrap_or(&0.0),
                    wheel_cmds.get(1).unwrap_or(&0.0),
                    is_upright
                );
            }
        }

        // Periodic status report
        if self.last_status_time.elapsed() > Duration::from_secs(2) {
            self.print_status();
            self.last_status_time = Instant::now();
        }

        Ok(())
    }

    /// Print status information
    fn print_status(&self) {
        info!("=== STATUS (step {}) ===", self.step_count);
        info!("Mode: {:?}", self.mode);
        info!("UDP: {}", if self.udp_listener.is_connected() { "connected" } else { "disconnected" });

        for motor in &self.robstride_motors.motors {
            let status = if motor.is_alive(500) { "OK" } else { "STALE" };
            info!(
                "  RS{}: {} pos={:.2} vel={:.2} temp={:.1}°C",
                motor.can_id, status, motor.state.position, motor.state.velocity, motor.state.temperature
            );
        }

        for motor in &self.odrive_motors.motors {
            let status = if motor.is_alive(500) { "OK" } else { "STALE" };
            info!(
                "  OD{}: {} pos={:.2} vel={:.2} state={} err=0x{:X}",
                motor.node_id, status, motor.state.position, motor.state.velocity,
                motor.axis_state, motor.axis_error
            );
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,bebop_linux=debug")),
        )
        .init();

    info!("╔════════════════════════════════════════╗");
    info!("║    Bebop Linux Firmware v0.1.0         ║");
    info!("║    SocketCAN + ONNX Runtime            ║");
    info!("╚════════════════════════════════════════╝");

    // Parse arguments
    let args = parse_args();
    info!("Model: {}", args.model_path.display());
    info!("CAN: {}", args.can_interface);
    info!("UDP port: {}", args.udp_port);
    info!("Control rate: {} Hz", args.control_rate);

    // Create controller
    let mut controller = RobotController::new(&args)?;

    // Initialize hardware
    controller.init()?;

    // Setup shutdown signal
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
        info!("Received Ctrl+C, shutting down...");
        r.store(false, Ordering::SeqCst);
    });

    // Main control loop
    let loop_interval = Duration::from_millis(1000 / args.control_rate);
    let mut interval = interval(loop_interval);
    let mut last_time = Instant::now();

    info!("Starting control loop at {} Hz...", args.control_rate);

    while running.load(Ordering::SeqCst) {
        interval.tick().await;

        let now = Instant::now();
        let dt = (now - last_time).as_secs_f32();
        last_time = now;

        if let Err(e) = controller.step(dt) {
            error!("Control loop error: {}", e);
        }
    }

    // Shutdown
    controller.shutdown()?;

    Ok(())
}
