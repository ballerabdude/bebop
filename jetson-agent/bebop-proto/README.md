# `bebop-proto`

Shared protobuf types for Bebop. Compiled into Rust via `prost-build` and
reused by the agent and (eventually) any Rust-based tooling on the mobile
side.

Mobile app codegen (TypeScript / Dart / Swift) should read directly from
`proto/bebop.proto` — keep this file as the single source of truth.

## Regenerate

```sh
cargo build -p bebop-proto
```

The generated Rust lives in `target/.../build/bebop-proto-*/out/bebop.v1.rs`
and is re-exported from `src/lib.rs` as `bebop_proto::v1`.
