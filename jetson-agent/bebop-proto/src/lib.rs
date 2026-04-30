//! Shared wire-format for communication between `bebop-agent` (Rust, on Jetson)
//! and the Bebop mobile application.
//!
//! Generated with [`prost`] from `proto/bebop.proto`.

pub mod v1 {
    include!(concat!(env!("OUT_DIR"), "/bebop.v1.rs"));
}

pub use prost::Message;
