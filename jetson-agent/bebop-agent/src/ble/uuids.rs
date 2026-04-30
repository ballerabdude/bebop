//! Stable UUIDs for the Bebop BLE service and its characteristics.
//!
//! These values MUST match what the mobile app expects — treat them as
//! part of the public ABI.

use uuid::{uuid, Uuid};

/// Primary Bebop service.
pub const SERVICE_UUID: Uuid = uuid!("b3b0b000-0b3b-4f9b-9b3b-b3b0b3b0b3b0");

/// Write: mobile app -> agent (ClientRequest frames).
pub const CHAR_REQUEST_UUID: Uuid = uuid!("b3b0b001-0b3b-4f9b-9b3b-b3b0b3b0b3b0");

/// Notify: agent -> mobile app (AgentResponse frames).
pub const CHAR_RESPONSE_UUID: Uuid = uuid!("b3b0b002-0b3b-4f9b-9b3b-b3b0b3b0b3b0");

/// Read + Notify: periodic summary status blob.
pub const CHAR_STATUS_UUID: Uuid = uuid!("b3b0b003-0b3b-4f9b-9b3b-b3b0b3b0b3b0");
