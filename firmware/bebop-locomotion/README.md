# Bebop Locomotion Firmware

Teensy 4.1 firmware for the Bebop robot (PlatformIO).

The robot runs the **`teensy_bridge`** env. It reads the BNO085 over SPI and
streams IMU frames to the Jetson over USB serial (`/dev/bebop-imu`, read by
`firmware/bebop-linux/src/imu_serial.rs` when the robot YAML has an `imu:`
block), and forwards the network-mode button on a control channel
(`/dev/bebop-control`, read by `bebop-agent`). The Jetson does not talk to the
BNO085 itself.

```
BNO085 ──SPI0──▶ Teensy 4.1 (teensy_bridge) ══ USB_TRIPLE_SERIAL, 16c0:048c ══▶ Jetson
mode button ───▶   pin 2      if00 /dev/bebop-imu        52-byte IMU frames, 200 Hz → bebop-linux
                              if02 /dev/bebop-imu-debug  diagnostics
                              if04 /dev/bebop-control    button events → bebop-agent
```

The `sim` and `ros` envs are legacy: the micro-ROS controller of the earlier
wheel-legged robot (hip / knee Robstrides plus ODrive wheels) with policy
inference on the Teensy. Everything below about motor control, ROS 2 topics and
the policy applies to them only.

## Architecture (legacy `sim` / `ros`)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Teensy 4.1                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Policy    │───▶│   Motor     │───▶│  CAN Bus            │  │
│  │  Inference  │    │  Commands   │    │  (ODrive/Robstride) │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         ▲                                        │              │
│         │           ┌─────────────┐              │              │
│         └───────────│ Observation │◀─────────────┘              │
│                     │   Builder   │◀──── IMU (BNO085)           │
│                     └─────────────┘                             │
│                            │                                    │
│                     ┌──────▼──────┐                             │
│                     │  micro-ROS  │◀────── USB Serial           │
│                     └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## Build Modes

| Env | Description | Entry Point | USB |
|-----|-------------|-------------|-----|
| `teensy_bridge` | **Production.** BNO085 over SPI → 52-byte IMU frames at 200 Hz, plus mode-button control frames | `main_teensy_bridge.cpp` | `USB_TRIPLE_SERIAL` |
| `imu_spi_test` | Standalone BNO085 check over SPI (the `teensy_bridge` wiring) | `main_imu_spi_test.cpp` | serial |
| `imu_test` | Standalone BNO085 check over I2C | `main_imu_test.cpp` | serial |
| `sim` | Legacy **Isaac Sim HIL** - Policy on Teensy, sensor data from simulation via micro-ROS | `main_sim.cpp` | `USB_DUAL_SERIAL` |
| `ros` | Legacy **Hardware Mode** - Policy on Teensy, sensor data from real IMU/motors via CAN | `main.cpp` | `USB_DUAL_SERIAL` |

## Quick Start

```bash
cd firmware/bebop-locomotion

# Production firmware (what the robot runs)
pio run -e teensy_bridge --target upload
cat /dev/bebop-imu-debug          # its diagnostics (after install-jetson.sh --setup-imu)

# Standalone IMU check
pio run -e imu_spi_test --target upload

# Legacy envs
pio run -e sim
pio run -e ros
```

`platformio.ini` still sets `default_envs = sim`, so a bare `pio run` or
`pio run --target upload`, and `just fw-build` / `just fw-flash`, which run
exactly that, build and flash the legacy `sim` env. Flashing it replaces the
`USB_TRIPLE_SERIAL` firmware: the `/dev/bebop-*` devices disappear and
`bebop-agent`'s mode button goes dead. Always pass `-e teensy_bridge`.
`sudo ./scripts/install-jetson.sh --setup-imu` (once, on the robot, from the
repo root) installs the udev rule behind the `/dev/bebop-*` symlinks.

## Motor Control

Legacy `sim` / `ros` envs only.

### Joint Mapping

| Index | Name | Motor Type | Control Mode |
|-------|------|------------|--------------|
| 0 | `left_hip_pitch` | Robstride | Position |
| 1 | `right_hip_pitch` | Robstride | Position |
| 2 | `left_knee_pitch` | Robstride | Position |
| 3 | `right_knee_pitch` | Robstride | Position |
| 4 | `left_wheel` | ODrive | Velocity |
| 5 | `right_wheel` | ODrive | Velocity |

### Serial Debug Commands

Connect to the debug serial port (`SerialUSB1`, 115200 baud):

```bash
# On Linux
screen /dev/ttyACM1 115200
# or
pio device monitor --port /dev/ttyACM1
```

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `pos` / `status` | Show all joint positions (rad & deg) |
| `zero <joint>` | Set current position as zero (Robstride joints 0-3 only) |
| `enable <joint\|all>` | Enable motor(s) |
| `disable <joint\|all>` | Disable motor(s) |

**Example:**
```
> pos
  Joint Status:
    [0] left_hip_pitch: OK pos=0.123 rad (7.0 deg)
    [1] right_hip_pitch: OK pos=-0.045 rad (-2.6 deg)
    ...

> zero 0
  Setting zero for joint 0 (left_hip_pitch)...
    1. Disabling motor...
    2. Setting mechanical zero...
    3. Saving to flash...
    4. Re-enabling motor...
  Done!

> enable all
  Enabled all motors
```

### ROS2 Motor Commands

Commands are sent via `/joint_commands` topic (`sensor_msgs/JointState`):

**Position/Velocity Control:**
- `position[]` - Target positions for leg joints (rad)
- `velocity[]` - Target velocities for wheel joints (rad/s)

**Special Commands (via `effort[]` field):**

| Value | Command | Description |
|-------|---------|-------------|
| `-999.0` | Clear Errors | Clear motor error flags |
| `-998.0` | Enable All | Enable all motors |
| `-997.0` | Disable All | Disable all motors |
| `-996.0` | Reset & Enable | Clear errors and enable |
| `-995.0` | Policy Mode | Switch to neural network control |
| `-994.0` | Manual Mode | Switch to passthrough/manual control |

**Example (Python):**
```python
from sensor_msgs.msg import JointState

# Send motor enable command
cmd = JointState()
cmd.effort = [-998.0]  # MOTOR_CMD_ENABLE_ALL
publisher.publish(cmd)

# Send position/velocity command
cmd = JointState()
cmd.name = ['left_hip_pitch', 'right_hip_pitch', 'left_knee_pitch', 
            'right_knee_pitch', 'left_wheel', 'right_wheel']
cmd.position = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]  # Leg positions
cmd.velocity = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0]  # Wheel velocities
publisher.publish(cmd)
```

## ROS2 Topics

Legacy `sim` / `ros` envs only.

### Published Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | 100 Hz | Joint positions and velocities |
| `/imu/data` | `sensor_msgs/Imu` | 100 Hz | IMU orientation, angular velocity, acceleration |
| `/motor_temps` | `std_msgs/Float32MultiArray` | 10 Hz | Motor temperatures (see format below) |
| `/motor_status` | `std_msgs/Int32MultiArray` | 10 Hz | Motor error codes and states |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_commands` | `sensor_msgs/JointState` | Motor position/velocity commands |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands for policy |

### Diagnostic Data Format

**`/motor_temps`** - `Float32MultiArray` with `NUM_JOINTS * 2` floats:
```
[motor0_temp, motor0_board_temp, motor1_temp, motor1_board_temp, ...]
```

**`/motor_status`** - `Int32MultiArray` with `NUM_JOINTS * 4` ints:
```
[motor0_error, motor0_state, motor0_enabled, motor0_extra, ...]
```
- `extra`: Bus voltage × 100 for ODrive, fault bits for Robstride

## Project Structure

```
bebop-locomotion/
├── include/
│   ├── BebopPolicy.h          # Auto-generated neural network weights (legacy)
│   ├── BNO085_IMU.h           # BNO085 driver (SPI)
│   ├── ControlProtocol.h      # teensy_bridge control frames (14 bytes)
│   ├── GenericMotor.h         # Motor interface abstraction
│   ├── ImuSerialProtocol.h    # teensy_bridge IMU frames (52 bytes)
│   ├── MicroROS.h             # micro-ROS communication (sim mode)
│   ├── ODriveMotor.h          # ODrive CAN protocol
│   ├── RobotConfig.h          # Joint/motor configuration (legacy)
│   ├── RobstrideMotor.h       # Robstride CAN protocol
│   ├── RosPublishers.h        # ROS topic publishers (ros mode)
│   └── SerialCommands.h       # Debug serial interface
├── src/
│   ├── main_teensy_bridge.cpp # teensy_bridge entry point (production)
│   ├── main_imu_spi_test.cpp  # imu_spi_test entry point
│   ├── main_imu_test.cpp      # imu_test entry point (I2C)
│   ├── main.cpp               # ros entry point (legacy)
│   ├── main_sim.cpp           # sim entry point (legacy)
│   ├── MicroROS.cpp           # micro-ROS implementation
│   ├── RosPublishers.cpp      # ROS publisher implementation
│   └── SerialCommands.cpp     # Serial command parser
└── platformio.ini             # Build configuration
```

## Hardware Connections

### Teensy 4.1 Pinout

BNO085 (Adafruit breakout) on hardware SPI0: the wiring of
`include/BNO085_IMU.h` (used by `teensy_bridge` and the legacy `ros` env) and of
`imu_spi_test`:

| BNO085 | Teensy pin |
|--------|------------|
| SCL (SCK) | 13 (also the onboard LED) |
| DI (MOSI) | 11 |
| SDA (MISO) | 12 |
| CS | 10 |
| INT | 37 (required for SPI) |
| RST | 36 |
| P0, P1 | 3V3 (SPI mode) |

`imu_test` uses I2C instead: SCL 19, SDA 18 (INT 37, RST 36 optional).

| Function | Pin |
|----------|-----|
| Network-mode button (`teensy_bridge`) | 2, switch to GND (internal pull-up) |
| CAN1 TX / RX (legacy `ros`) | 22 / 23 |
| Status LED (legacy `sim` / `ros`) | 13, shared with SPI SCK |

### USB Ports

`teensy_bridge` (`USB_TRIPLE_SERIAL`, enumerates as `16c0:048c`):
- **if00** (`Serial`) → `/dev/bebop-imu` - binary IMU frames
- **if02** (`SerialUSB1`) → `/dev/bebop-imu-debug` - diagnostics
- **if04** (`SerialUSB2`) → `/dev/bebop-control` - control frames (button events)

The `/dev/bebop-*` names are udev symlinks from
`sudo ./scripts/install-jetson.sh --setup-imu`.

Legacy `sim` / `ros` (`USB_DUAL_SERIAL`):
- **USB0** (`Serial`) - micro-ROS agent communication
- **USB1** (`SerialUSB1`) - Debug output and serial commands

## Updating the Policy

Legacy `sim` / `ros` envs only. The weights are compiled in from
`include/BebopPolicy.h` (its header: generated 2026-01-29, 30 inputs -> 6
outputs). Nothing in the tree regenerates it:
`sim/bebop_training/export_bebop_model.py` stopped writing the header on
2026-05-10 (`090be8b`) and exports the v2 humanoid's policy to ONNX instead.

## IDE Setup (VSCode + clangd)

```bash
# Generate compilation database (follows default_envs = sim unless -e is given)
cd firmware/bebop-locomotion
pio run -e teensy_bridge -t compiledb
```

`firmware/.clangd` points clangd at the generated `compile_commands.json`.

Ensure your VSCode settings include:
```json
{
    "clangd.arguments": [
        "--query-driver=**/*arm-none-eabi*",
        "--header-insertion=never"
    ]
}
```

Then restart clangd: `Ctrl+Shift+P` → `clangd: Restart language server`

## Troubleshooting

### Build fails with missing headers
```bash
pio run -e <env> -t clean && pio run -e <env>
```

### Upload fails
1. Press Teensy button for bootloader mode
2. Check connection: `ls /dev/ttyACM*`
3. Install udev rules (see below)

### Install udev rules (Linux)
```bash
curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core/develop/platformio/assets/system/99-platformio-udev.rules | sudo tee /etc/udev/rules.d/99-platformio-udev.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## License

Part of the Bebop Robot project.
