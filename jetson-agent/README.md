# jetson-agent/

Everything that runs **on the Jetson** in support of provisioning.

| Path            | What it is                                                                  |
|-----------------|-----------------------------------------------------------------------------|
| `bebop-agent/`  | The Rust provisioning daemon (systemd service): Wi-Fi setup, the Hosted Network, the mode button, the setup server. See [`bebop-agent/README.md`](bebop-agent/README.md). |
| `bebop-proto/`  | Protobuf wire types for both robot APIs: the setup protocol (`bebop.proto`, WS on `:9091`) and the `bebop-linux` runtime API (`bebop_runtime.proto`, WS on `:9090`). See [`bebop-proto/README.md`](bebop-proto/README.md). |
| `deploy/`       | Systemd unit, install/uninstall scripts, example `agent.toml`. See [`deploy/README.md`](deploy/README.md). |
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

The operator app (`../bebop-app/`, a web app) is intentionally outside this
folder: it runs off-device and has its own toolchain. The robot runtime
(`../firmware/bebop-linux/`) is outside it too and builds against
`bebop-proto` as a path dependency.

## Provisioning flow

The agent serves a protobuf-over-WebSocket setup server on `:9091`. In Hosted
Network mode (`ap`, the default) it hosts the
`Bebop-<first 6 chars of /etc/machine-id>` hotspot so a phone can join and
reach that server at `192.168.42.1:9091`; in `client` mode it joins a
saved network. Only the physical mode button switches between the two (by
default on the Teensy, read from `/dev/bebop-control`); there is no automatic
fallback. See [`../docs/setup-protocol.md`](../docs/setup-protocol.md).
