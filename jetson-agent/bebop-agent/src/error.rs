//! Crate-wide error types.
//!
//! Subsystems return `Result<T, AgentError>` for errors that need to be
//! surfaced over the BLE protocol. Internal/plumbing errors use `anyhow`.

use thiserror::Error;

// Canonical alias for subsystem-level results. Not yet used at any call site
// (subsystems currently surface `anyhow::Result`); kept so the protocol layer
// can adopt it without a churny rename when the BLE dispatcher learns to map
// `AgentError` -> proto error codes.
#[allow(dead_code)]
pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, Error)]
pub enum AgentError {
    // Scaffolding: returned by stubbed-out request handlers once the BLE
    // dispatcher starts validating method coverage.
    #[allow(dead_code)]
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    // Scaffolding: emitted by the (TODO) auth handshake before the pairing
    // code check lands.
    #[allow(dead_code)]
    #[error("unauthorized")]
    Unauthorized,

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("wifi error: {0}")]
    Wifi(String),

    #[error("container error: {0}")]
    Container(String),

    // Scaffolding: surfaced once the OTA updater reports failures back over
    // BLE instead of just logging them.
    #[allow(dead_code)]
    #[error("ota error: {0}")]
    Ota(String),

    // Scaffolding: surfaced once BLE-layer faults (adapter loss, pairing
    // rejection) get mapped into the protocol response stream.
    #[allow(dead_code)]
    #[error("ble error: {0}")]
    Ble(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
