# Bebop Locomotion Firmware

Teensy 4.1 locomotion controller for the Bebop robot. Runs neural network policy inference on-board, interfaces with CAN motor controllers (ODrive, Robstride), and communicates with ROS2 via micro-ROS.

## Architecture

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

| Mode | Description | Entry Point |
|------|-------------|-------------|
| `sim` | **Isaac Sim HIL** - Policy on Teensy, sensor data from simulation via micro-ROS | `main_sim.cpp` |
| `ros` | **Hardware Mode** - Policy on Teensy, sensor data from real IMU/motors via CAN | `main.cpp` |

## Quick Start

```bash
cd firmware/bebop-locomotion

# Build (default: sim mode)
pio run

# Build for hardware
pio run -e ros

# Upload to Teensy
pio run --target upload

# Monitor debug output (SerialUSB1)
pio device monitor
```

## Motor Control

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
│   ├── BebopPolicy.h      # Auto-generated neural network weights
│   ├── BNO085_IMU.h       # IMU driver
│   ├── GenericMotor.h     # Motor interface abstraction
│   ├── MicroROS.h         # micro-ROS communication (sim mode)
│   ├── ODriveMotor.h      # ODrive CAN protocol
│   ├── RobotConfig.h      # Joint/motor configuration
│   ├── RobstrideMotor.h   # Robstride CAN protocol
│   ├── RosPublishers.h    # ROS topic publishers (ros mode)
│   └── SerialCommands.h   # Debug serial interface
├── src/
│   ├── main.cpp           # Hardware mode entry point
│   ├── main_sim.cpp       # Simulation mode entry point
│   ├── MicroROS.cpp       # micro-ROS implementation
│   ├── RosPublishers.cpp  # ROS publisher implementation
│   └── SerialCommands.cpp # Serial command parser
└── platformio.ini         # Build configuration
```

## Hardware Connections

### Teensy 4.1 Pinout

| Function | Pin |
|----------|-----|
| CAN1 TX | 22 |
| CAN1 RX | 23 |
| IMU SCL | 19 |
| IMU SDA | 18 |
| IMU INT | 36 |
| IMU RST | 35 |
| Status LED | 13 |

### USB Ports

With `USB_DUAL_SERIAL` enabled:
- **USB0** (`Serial`) - micro-ROS agent communication
- **USB1** (`SerialUSB1`) - Debug output and serial commands

## Updating the Policy

Neural network weights are stored in `include/BebopPolicy.h`. To update:

```bash
# 1. Export from Isaac Lab
./isaaclab.sh -p bebop_training/export_bebop_model.py \
    --checkpoint training_logs/.../model_XXXX.pt \
    --headless

# 2. Copy to firmware
cp training_logs/.../BebopPolicy.h firmware/bebop-locomotion/include/

# 3. Rebuild and upload
cd firmware/bebop-locomotion
pio run --target upload
```

## IDE Setup (VSCode + clangd)

```bash
# Generate compilation database
cd firmware/bebop-locomotion
pio run -t compiledb
```

The repo includes `.clangd` at the root pointing to the firmware's `compile_commands.json`.

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
pio run -t clean && pio run
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
