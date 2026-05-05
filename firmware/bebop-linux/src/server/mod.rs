//! Protobuf-over-WebSocket runtime API.
//!
//! Listens on `cfg.server.bind_addr`. Each accepted WS client gets:
//!
//! - A bidirectional binary protobuf stream
//!   ([`bebop_proto::runtime::v1::ClientRuntimeMessage`] in,
//!   [`bebop_proto::runtime::v1::ServerRuntimeMessage`] out).
//! - A telemetry subscription (off by default; enabled via
//!   `SubscribeTelemetry`).
//! - Async forwarding of supervisor events (mode changes, E-STOP latches).
//!
//! All commands eventually call methods on [`crate::safety::Supervisor`].

pub mod handlers;
pub mod telemetry;
pub mod ws;

pub use ws::run_server;
