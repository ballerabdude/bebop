//! Bebop V2 robot runtime — library face.
//!
//! `bebop-linux` is a single binary that operates the robot in one of three
//! [`mode::Mode`]s:
//!
//! - [`mode::Mode::Idle`]      — motors disabled, telemetry streams.
//! - [`mode::Mode::DialIn`]    — per-motor enable/disable, slew-limited
//!   hold commands, watchdog. Used during bench bring-up.
//! - [`mode::Mode::RunPolicy`] — ONNX policy drives the joints (legacy
//!   behavior of this binary).
//!
//! The Tauri app drives mode transitions over a protobuf-over-WebSocket
//! API ([`server`]). All motor TX traffic flows through [`safety::Supervisor`],
//! which clamps to per-joint hard limits, runs a feedback watchdog, and
//! latches an E-STOP that disables every motor on the bus.

#![allow(dead_code)]

pub mod can_interface;
pub mod config;
pub mod imu;
pub mod mode;
pub mod observation;
pub mod policy;
pub mod policy_io;
pub mod policy_runner;
pub mod powerboard;
pub mod robstride;
pub mod safety;
pub mod server;
pub mod udp_command;
