fn main() {
    let proto_files = ["proto/bebop.proto"];
    for f in &proto_files {
        println!("cargo:rerun-if-changed={}", f);
    }
    prost_build::compile_protos(&proto_files, &["proto"]).expect("failed to compile bebop protos");
}
