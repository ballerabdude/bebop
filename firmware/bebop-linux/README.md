# Bebop Linux Firmware

Linux-based firmware for the Bebop robot using SocketCAN and ONNX Runtime for neural network inference.

## Features

- **50 Hz control loop** (configurable up to 200+ Hz)
- **SocketCAN** for direct CAN bus communication
- **ONNX Runtime** for neural network policy inference
- **Robstride motor support** (RS01-RS04) with MIT-style impedance control
- **ODrive motor support** (S1) with velocity control
- **UDP command interface** for teleoperation
- **Safety checks** with IMU-based fall detection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Control Loop (50 Hz)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   IMU    │→ │  Observe │→ │  Policy  │→ │ Commands │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│        ↑                          ↑              │          │
│        │                          │              ↓          │
│  ┌──────────┐              ┌──────────┐   ┌──────────┐    │
│  │   UDP    │              │   ONNX   │   │ SocketCAN│    │
│  │ Commands │              │  Model   │   │   Bus    │    │
│  └──────────┘              └──────────┘   └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ↓                             ↓                             ↓
            ┌──────────────┐             ┌──────────────┐             ┌──────────────┐
            │  Robstride   │             │  Robstride   │             │   ODrive     │
            │  Hip Motor   │             │  Knee Motor  │             │   Wheel      │
            └──────────────┘             └──────────────┘             └──────────────┘
```

## Requirements

### System Requirements

- Linux (tested on Ubuntu 22.04+, Raspberry Pi OS, Jetson)
- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- CAN bus interface (USB-CAN adapter, built-in CAN, etc.)
- ONNX Runtime 1.16+ (dynamically loaded)

### Hardware Setup

1. **CAN Interface**: Configure your CAN interface at 1 Mbps:

```bash
# Bring up can0 at 1 Mbps
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# Verify
ip link show can0
```

2. **ONNX Runtime**: Install ONNX Runtime:

```bash
# Download from https://github.com/microsoft/onnxruntime/releases
# Or install via package manager if available

# For aarch64 (Jetson, Raspberry Pi):
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz
tar xzf onnxruntime-linux-aarch64-1.16.3.tgz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/onnxruntime-linux-aarch64-1.16.3/lib

# For x86_64:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
```

## Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test
```

## Releasing & deploying to a Jetson

`bebop-linux` ships as a single tarball that bundles the aarch64 binary,
the joint config (`bebop_v2.yaml`), the ONNX policy (`policy.onnx` +
`policy.onnx.data`), and the systemd unit. The robot's runtime, config
and policy weights are always rolled forward together so a "good build"
on the Jetson is exactly the contents of one tarball.

### Cutting a release

```bash
# from a clean main checkout
git tag firmware/v0.2.0
git push origin firmware/v0.2.0
```

The `firmware-jetson` job in `.github/workflows/ci.yml` builds the
binary on `ubuntu-22.04-arm`, packs `bebop-linux-aarch64.tar.gz`
(`bin/`, `config/`, `systemd/`, `VERSION`) with a sibling `.sha256`,
and publishes both to a GitHub Release named after the tag.

### Installing on the Jetson

The `scripts/install-jetson.sh` installer fetches the latest matching
Release by default. From any reasonably fresh checkout on the robot:

```bash
sudo ./scripts/install-jetson.sh --linux-only
```

That one command installs the latest tagged firmware bundle, writes the
config and policy into `/etc/bebop/`, installs the systemd unit, and
enables + restarts `bebop-linux.service`.

For pinned or pre-release installs:

```bash
sudo ./scripts/install-jetson.sh --linux-only --release firmware/v0.2.0
sudo ./scripts/install-jetson.sh --linux-only --run-id N
```

For installing straight from a local checkout (no GitHub round-trip —
useful for on-device dev iteration):

```bash
# build first, then install the pre-built binary + working-tree configs
cargo build --release            # in firmware/bebop-linux/
sudo ./scripts/install-jetson.sh --linux-only --local

# or do both in one shot (cargo runs as your invoking user)
sudo ./scripts/install-jetson.sh --linux-only --local --build
```

It lays things down as:

```
/usr/local/bin/bebop-linux
/etc/bebop/bebop_v2.yaml          # overwritten; previous saved as .yaml.bak
/etc/bebop/policy.onnx            # graph
/etc/bebop/policy.onnx.data       # external-data weights (must travel with policy.onnx)
/etc/systemd/system/bebop-linux.service
```

The unit invokes `bebop-linux --config /etc/bebop/bebop_v2.yaml`, which
resolves `--policy` to `<config-dir>/policy.onnx` by default — so the
policy files just need to sit next to the YAML.

### Rolling back

Either re-run the installer pointing at the previous tag
(`--release firmware/v0.1.9`) or restore the saved YAML on its own:

```bash
sudo mv /etc/bebop/bebop_v2.yaml.bak /etc/bebop/bebop_v2.yaml
sudo systemctl restart bebop-linux
```

## Running

### Basic Usage

```bash
# Run with default settings
./target/release/bebop-linux --model model.onnx

# Specify CAN interface
./target/release/bebop-linux --model model.onnx --can can0

# Run at 100 Hz
./target/release/bebop-linux --model model.onnx --rate 100

# Simulation mode (no hardware)
./target/release/bebop-linux --sim

# Passthrough mode (no policy)
./target/release/bebop-linux --no-policy
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `model.onnx` | Path to ONNX policy file |
| `--can` | `-c` | `can0` | CAN interface name |
| `--port` | `-p` | `10000` | UDP command port |
| `--rate` | `-r` | `50` | Control loop rate (Hz) |
| `--no-policy` | | | Disable policy, passthrough mode |
| `--sim` | | | Simulation mode (no hardware) |
| `--help` | `-h` | | Print help |

### Environment Variables

```bash
# Set log level
export RUST_LOG=debug  # or info, warn, error

# Specify CAN interfaces
export CAN_INTERFACES=can0,can1,can2,can3
```

## UDP Command Protocol

Send JSON commands to UDP port 10000:

```json
// Velocity command
{"xvel": 0.5, "yvel": 0.0, "angvel": 0.1}

// Reset command
{"type": "reset"}
```

### Python Example

```python
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send velocity command
cmd = {"xvel": 0.5, "yvel": 0.0, "angvel": 0.0}
sock.sendto(json.dumps(cmd).encode(), ("localhost", 10000))
```

## Converting Your Model to ONNX

If your policy is trained in PyTorch:

```python
import torch
import torch.onnx

# Load your trained policy
model = YourPolicy()
model.load_state_dict(torch.load("policy.pt"))
model.eval()

# Create dummy input matching observation size
dummy_input = torch.randn(1, 30)  # [batch, obs_dim]

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=14,
)
```

## Configuration

### Observation Format (must match training!)

| Index | Description | Scaling |
|-------|-------------|---------|
| 0-2 | Base linear velocity (x, y, z) | × SCALE_LIN_VEL |
| 3-5 | Base angular velocity (x, y, z) | × SCALE_ANG_VEL |
| 6-8 | Projected gravity | (no scaling) |
| 9-11 | Command velocity (vx, vy, wz) | × SCALE_LIN_VEL/ANG_VEL |
| 12-17 | Joint positions (relative to default) | × SCALE_DOF_POS |
| 18-23 | Joint velocities | × SCALE_DOF_VEL |
| 24-29 | Last action | (no scaling) |

### Motor Configuration

Edit `src/config.rs` to match your hardware:

```rust
JointConfig {
    name: "left_hip_pitch".to_string(),
    index: 0,
    can_id: 31,
    can_bus: "can0".to_string(),
    motor_type: MotorType::Robstride(RobstrideModel::RS04),
    default_position: 0.0,
    position_min: -0.8,
    position_max: 0.8,
    kp: 50.0,
    kd: 2.0,
},
```

## Comparison to Teensy Firmware

| Feature | Teensy | Linux |
|---------|--------|-------|
| Control Rate | 200 Hz | 50-500 Hz |
| CAN Interface | FlexCAN_T4 | SocketCAN |
| Policy Inference | Embedded C++ | ONNX Runtime |
| ROS Integration | micro-ROS | Native ROS 2 (optional) |
| Development | PlatformIO | Cargo |
| Debug | Serial | Full Linux tooling |
| Multi-bus | 3 buses max | Unlimited (per hardware) |

## Troubleshooting

### CAN Bus Issues

```bash
# Check CAN interface status
ip -details link show can0

# Monitor CAN traffic
candump can0

# Check for bus errors
dmesg | grep can0
```

### Permission Issues

```bash
# Add user to dialout group (for USB-CAN adapters)
sudo usermod -a -G dialout $USER

# Or run with sudo (not recommended for production)
sudo ./target/release/bebop-linux
```

### ONNX Runtime Issues

```bash
# Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/onnxruntime/lib

# Check if library is found
ldd target/release/bebop-linux | grep onnx
```
