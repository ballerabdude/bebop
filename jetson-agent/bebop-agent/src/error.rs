//! Crate-wide error types.
//!
//! Subsystems return `Result<T, AgentError>` for errors that need to be
//! surfaced over the setup protocol. Internal/plumbing errors use `anyhow`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("config error: {0}")]
    Config(String),

    #[error("wifi error: {0}")]
    Wifi(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
