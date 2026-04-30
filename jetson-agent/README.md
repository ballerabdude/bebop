# jetson-agent/

Everything that runs **on the Jetson**.

| Path            | What it is                                                                  |
|-----------------|-----------------------------------------------------------------------------|
| `bebop-agent/`  | The Rust daemon (systemd service). Owns BLE / Wi-Fi / Docker / OTA.         |
| `bebop-proto/`  | Shared protobuf schema for the BLE wire protocol.                           |
| `robot-app/`    | Dockerfile + entrypoint for the robot application container the agent runs. |
| `deploy/`       | Systemd unit, install/uninstall scripts, example `agent.toml`.              |
| `Cargo.toml`    | Workspace root (`bebop-agent` + `bebop-proto`).                             |
| `Cross.toml`    | `cross` config for the `aarch64-unknown-linux-gnu` build.                   |
| `rust-toolchain.toml` | Pinned Rust toolchain for the workspace.                              |

## Quick reference

All recipes live in the repo-root `justfile`:

```sh
just check          # cargo check --workspace --all-targets
just test           # cargo test --workspace
just lint           # cargo clippy --workspace --all-targets -- -D warnings
just build-jetson   # cross build --release --target aarch64-unknown-linux-gnu -p bebop-agent
just deploy HOST    # scp + rsync + install on a robot
just build-app      # docker buildx build for the robot-app container
```

If you'd rather call cargo directly, do it from inside this folder:

```sh
cd jetson-agent
cargo build --workspace
```

## Why this is its own folder

The agent + its deploy scripts + the robot-app container all ship together
to the Jetson. Keeping them under one roof means:

- one Rust workspace (`Cargo.toml`, single `Cargo.lock`, single `target/`)
- the install script's `WORKSPACE_ROOT` walk lands on `jetson-agent/`,
  where `target/aarch64-unknown-linux-gnu/release/bebop-agent` lives
- adding more on-device pieces later (e.g. extra containers, udev rules,
  log-shipping configs) doesn't pollute the repo root

The customer-facing mobile app is intentionally outside this folder — it
runs off-device and has its own build toolchain. See `../bebop-app/`.
