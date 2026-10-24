//! Bebop robot runtime — library face.
//!
//! `bebop-linux` runs the wheeled Bebop (two ODrive S1 BotWheels and a
//! Robstride PowerBoard over SocketCAN) and keeps the Robstride MIT-mode
//! joint path for the arms that come next. It operates in one of two
//! [`mode::Mode`]s:
//!
//! - [`mode::Mode::Idle`]   — boot mode: nothing can be armed; telemetry
//!   streams, and WS subscribers trigger passive feedback on disarmed joints.
//! - [`mode::Mode::DialIn`] — the armed mode: armed wheels track the
//!   operator twist, armed joints get the slew-limited hold cycle, and the
//!   feedback watchdog runs.
//!
//! The operator app drives mode transitions over a protobuf-over-WebSocket
//! API ([`server`]). All motor TX traffic flows through [`safety::Supervisor`],
//! which clamps to per-joint hard limits, runs a feedback watchdog, and
//! latches an E-STOP that disables every motor on every bus.

pub mod can_interface;
pub mod capture_control;
pub mod config;
pub mod drive;
pub mod imu;
pub mod imu_serial;
pub mod mode;
pub mod model;
pub mod odrive;
pub mod powerboard;
pub mod realtime;
pub mod robstride;
pub mod safety;
pub mod server;
pub mod system_capture;
pub mod vision;
pub mod voice;
