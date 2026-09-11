# jetson-agent/

Everything that runs **on the Jetson** in support of provisioning.

| Path            | What it is                                                                  |
|-----------------|-----------------------------------------------------------------------------|
| `bebop-agent/`  | The Rust provisioning daemon (systemd service). Owns Wi-Fi setup + SoftAP.  |
| `bebop-proto/`  | Shared protobuf schema for the setup wire protocol (WS on `:9091`).         |
| `deploy/`       | Systemd unit, install/uninstall scripts, example `agent.toml`.              |
| `Cargo.toml`    | Workspace root (`bebop-agent` + `bebop-proto`).                             |
| `rust-toolchain.toml` | Pinned Rust toolchain for the workspace.                              |

## Quick reference

All recipes live in the repo-root `justfile`:

```sh
just check          # cargo check --workspace --all-targets
just test           # cargo test --workspace
just lint           # cargo clippy --workspace --all-targets -- -D warnings
just build-jetson   # cargo build --release -p bebop-agent (run on arm64 Linux)
just deploy HOST    # scp + rsync + install on a robot
```

If you'd rather call cargo directly, do it from inside this folder:

```sh
cd jetson-agent
cargo build --workspace
```

## Why this is its own folder

The agent + its deploy scripts ship together to the Jetson. Keeping them
under one roof means:

- one Rust workspace (`Cargo.toml`, single `Cargo.lock`, single `target/`)
- the install script's `WORKSPACE_ROOT` walk lands on `jetson-agent/`,
  where `target/release/bebop-agent` lives
- adding more on-device pieces later doesn't pollute the repo root

The customer-facing mobile app is intentionally outside this folder — it
runs off-device and has its own build toolchain. See `../bebop-app/`.

## Provisioning flow

The agent serves a protobuf-over-WebSocket setup server on `:9091`. When the
robot has no usable Wi-Fi it also hosts a `Bebop-XXXX` SoftAP so a phone can
join and reach that server at `192.168.42.1:9091`. See
[`../docs/setup-protocol.md`](../docs/setup-protocol.md).
