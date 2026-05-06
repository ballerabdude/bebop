//! BLE subsystem: advertises the Bebop GATT service and dispatches
//! incoming requests from the mobile app to the rest of the agent.
//!
//! Transport sketch:
//!   * One custom primary service (UUID below).
//!   * One "request" characteristic (Write / WriteWithoutResponse) — the phone
//!     writes a [`bebop_proto::v1::ClientRequest`] (protobuf-encoded, possibly
//!     chunked — see `framing.rs`).
//!   * One "response" characteristic (Notify) — the agent pushes
//!     [`bebop_proto::v1::AgentResponse`] frames back.
//!   * One "status" characteristic (Read / Notify) — periodic status blob so
//!     the app UI stays live without polling.
//!
//! Authentication is layered on top of the protocol: the first frame the
//! phone sends after connect must be an auth handshake (TODO) using a
//! pre-shared pairing code. Until then only `GetDeviceInfoRequest` is
//! allowed.

// Exposed `pub(crate)` so the network control surface (`crate::ws`) can
// reuse the exact same request/response handler the BLE GATT server uses.
pub(crate) mod dispatcher;
mod framing;
mod uuids;

#[cfg(target_os = "linux")]
mod server;

#[cfg(not(target_os = "linux"))]
mod server_stub;

// Re-export UUIDs so the (Linux-only) server module can reach them via
// `super::SERVICE_UUID`. The `allow` keeps macOS/CI builds quiet when the
// server module is cfg'd out.
#[allow(unused_imports)]
pub use uuids::*;

use crate::state::AppState;

/// Entrypoint: start advertising and run the GATT server forever.
pub async fn run(state: AppState) -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    {
        server::serve(state).await
    }
    #[cfg(not(target_os = "linux"))]
    {
        server_stub::serve(state).await
    }
}
