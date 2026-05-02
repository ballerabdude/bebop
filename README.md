# Bebop

Bebop is an NVIDIA Jetson–based robot platform. This monorepo contains
everything needed to design, simulate, build, provision, and update a fleet
of Bebop robots — plus the customer-facing companion app.

## High-Level Architecture

```
 ┌────────────────────┐   BLE (GATT)    ┌──────────────────────────┐
 │  Mobile App        │ ───────────────▶│  bebop-agent (Rust)      │
 │ (iOS / Android)    │ ◀─── status ────│  running on Jetson       │
 └────────────────────┘                  │                          │
                                         │  ┌────────────────────┐  │
                                         │  │ BLE GATT server    │  │
                                         │  │ Wi-Fi provisioner  │  │
                                         │  │ Container manager  │  │
                                         │  │ OTA updater        │  │
                                         │  └────────────────────┘  │
                                         └──────────────┬───────────┘
                                                        │ Docker API
                                                        ▼
                                            ┌──────────────────────┐
                                            │ Robot App container  │
                                            │ (nvidia runtime)     │
                                            └──────────────────────┘
```

## Repository Layout

| Path                | Purpose                                                                                                                   |
|---------------------|---------------------------------------------------------------------------------------------------------------------------|
| `jetson-flash/`     | One-time host-side provisioning: downloads NVIDIA L4T and flashes a fresh Jetson Orin Nano over USB. Runs once per device. |
| `jetson-agent/`     | Rust workspace + on-device deployables. `bebop-agent` daemon, `bebop-proto` (BLE wire format), `deploy/` (systemd + install scripts), and `robot-app/` (the container the agent supervises). |
| `bebop-app/`        | Customer-facing companion app (Tauri 2 + React + TypeScript). Desktop / iOS / Android.                                    |
| `firmware/`         | Embedded C++ firmware (PlatformIO): `bebop-linux/` and `bebop-locomotion/`. C/C++ tooling pinned via `firmware/.clangd`.  |
| `sim/`              | Isaac Sim / Isaac Lab containers, the `bebop_training` Python RL extension, and `usd/` scene assets. Runs off-robot on a workstation with an NVIDIA GPU. |
| `ros2/`             | ROS 2 Jazzy workspace used on the dev workstation: `src/bebop_pilot`, `src/bebopv2_description` (URDF), and the dev container under `docker/`. |
| `docs/`             | Cross-cutting docs: architecture, BLE protocol, OTA flow, onboarding, hardware reference.                                 |
| `docker-compose.yml`| Orchestrates the dev-workstation containers (`ros2_docker`, `isaac_sim`, `isaac_lab`) under the `sim` / `lab` profiles.   |
| `justfile`          | Convenience recipes (`just check`, `just build-jetson`, `just sim-up`, `just lab-up`, ...).                               |
| `.github/`          | CI workflows.                                                                                                             |

## Getting Started

Read [`docs/onboarding.md`](docs/onboarding.md) for dev setup, your first
build, and how to deploy to a real Jetson. Then `docs/architecture.md`
shows how the on-robot pieces connect.

```sh
just                # list available recipes
just check          # workspace-wide cargo check (jetson-agent)
just build-jetson   # native arm64 build of the agent (run on arm64 Linux)
just app-dev        # run the mobile app in Tauri dev mode
just sim-up         # bring up Isaac Sim + ROS 2 dev container
just lab-up         # bring up Isaac Lab + ROS 2 dev container
```

## Naming

The robot product is **Bebop**. The on-device daemon is **bebop-agent**.
The companion app is also called **Bebop** (binary: `bebop-app`).
