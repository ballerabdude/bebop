//! Crate-wide error types.
//!
//! Subsystems return `Result<T, AgentError>` for errors that need to be
//! surfaced over the BLE protocol. Internal/plumbing errors use `anyhow`.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    #[error("unauthorized")]
    Unauthorized,

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("wifi error: {0}")]
    Wifi(String),

    #[error("container error: {0}")]
    Container(String),

    #[error("ota error: {0}")]
    Ota(String),

    #[error("ble error: {0}")]
    Ble(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
