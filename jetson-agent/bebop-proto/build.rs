//! Build script: compile every `.proto` under `proto/` into Rust types.
//!
//! Globbing keeps `bebop-proto` "drop a new .proto in and it works" without
//! needing to update this file. Three protos live here:
//!
//!  - `bebop.proto`         — BLE control surface
//!  - `bebop_runtime.proto` — WS runtime API
//!  - `bebop_capture.proto` — schema for MCAP capture files
//!
//! We also emit a combined [`FileDescriptorSet`] to `OUT_DIR/descriptor.bin`
//! so the MCAP writer in `bebop-linux` can embed it as the schema record
//! at the head of each capture file (MCAP's well-known `protobuf` schema
//! encoding wants the FDS bytes for the root message and all of its
//! dependencies).

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let proto_dir = "proto";
    let mut proto_files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(proto_dir).expect("read proto dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|s| s.to_str()) == Some("proto") {
            println!("cargo:rerun-if-changed={}", path.display());
            proto_files.push(path);
        }
    }
    proto_files.sort();
    if proto_files.is_empty() {
        panic!("no .proto files found in {proto_dir}/");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let descriptor_path = out_dir.join("descriptor.bin");

    let mut config = prost_build::Config::new();
    config.file_descriptor_set_path(&descriptor_path);
    config
        .compile_protos(&proto_files, &[proto_dir])
        .expect("failed to compile bebop protos");
}
