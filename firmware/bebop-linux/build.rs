//! Build script: compile `protos/system.proto` into prost Rust types and
//! emit the serialized `FileDescriptorSet` the MCAP schema registry needs.
//!
//! `bebop-proto` (a path dependency) already generates the wire-format
//! types from its own protos via prost-build, so `protoc` is already a
//! firmware build requirement — this only adds the system-telemetry
//! schema and the descriptor set that Rerun's generic protobuf MCAP
//! decoder consumes.

use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let fds_path = out_dir.join("system_fds.bin");

    let mut config = prost_build::Config::new();
    config.file_descriptor_set_path(&fds_path);
    config
        .compile_protos(&["protos/system.proto"], &["protos"])
        .expect("compile system.proto");

    println!("cargo:rerun-if-changed=protos/system.proto");
}
