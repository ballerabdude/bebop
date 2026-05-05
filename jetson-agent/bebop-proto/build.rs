//! Build script: compile every `.proto` under `proto/` into Rust types.
//!
//! Globbing keeps `bebop-proto` "drop a new .proto in and it works" without
//! needing to update this file. Both `bebop.proto` (BLE control surface) and
//! `bebop_runtime.proto` (WS runtime API) live in the same directory.

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

    prost_build::compile_protos(&proto_files, &[proto_dir])
        .expect("failed to compile bebop protos");
}
