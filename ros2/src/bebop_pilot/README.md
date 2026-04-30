# bebop_pilot

The full **on-Jetson deployment stack** for the Bebop V2 humanoid. This is the
single ROS 2 package you need to drive the real robot from a PS5 DualSense
gamepad with a trained Isaac Lab policy.

> Replaces the deprecated **`bebop_control`** package, which was built around
> the wheel-legged V1 robot (6 DoF, mixed position+velocity).

## What's in the box

| Node              | Topic / role                                          | When it runs                        |
|-------------------|-------------------------------------------------------|-------------------------------------|
| `pilot_node`      | DualSense → `/cmd_vel`, `/policy_enable`              | Always                              |
| `policy_runner`   | ONNX policy → `/joint_commands` (8 joints)            | Always                              |
| `motor_bus`       | SocketCAN ↔ 8x Robstride; `/joint_commands` ↔ `/joint_states` | Always (skip with `launch_motors:=false` for dry runs) |
| `bno085_imu`      | BNO085 I2C → `/imu/data`                              | Optional (skip if IMU comes from elsewhere) |
| `rate_monitor`    | Diagnostic: rates, jitter, latency                    | Manual, for debugging               |

This package owns the **entire on-Jetson stack** for V2 — there is no
microcontroller / Teensy step in the loop. `motor_bus.py` talks directly
to the 8 Robstride motors over SocketCAN and is the only thing that
publishes `/joint_states`.

## Topic graph

```
                 PS5 DualSense (USB / BT)
                          │  evdev
                          ▼
                   ┌─────────────┐
                   │ pilot_node  │
                   └──────┬──────┘
                /cmd_vel  │  /policy_enable
                          ▼
   /imu/data ──┐   ┌──────────────┐
   /joint_states ─►│ policy_runner│──► /joint_commands
                   └──────────────┘            │
                                               ▼
                                       ┌──────────────┐    can0 1Mbps
                                       │  motor_bus   │ ◄──► 8x Robstride
                                       └──────────────┘    (RS04/RS03/RS02)
                                          ▲     │
                                          └─ /joint_states (back to policy_runner)
```

## Default DualSense mapping

| Input                 | Action                                                  |
|-----------------------|---------------------------------------------------------|
| Left stick Y (up)     | Walk forward (`+linear.x`)                              |
| Left stick X (right)  | Strafe right (`-linear.y`)                              |
| Right stick X (right) | Yaw clockwise (`-angular.z`)                            |
| **L1 (hold)**         | Deadman switch — required to send any non-zero command. |
| Triangle              | Re-enable policy (`/policy_enable = true`).             |
| Circle                | Emergency stop (zeros command, disables policy).        |
| PS button             | Quit the pilot node.                                    |

Override in `config/ps5_dualsense.yaml`. Keep `max_lin_vel_*` / `max_ang_vel_z`
≤ the velocity command ranges used in
`bebop_training/envs/bebop_v2_base_cfg.py` to stay in-distribution.

## Install

### 1. System packages on the Jetson

```bash
sudo apt update
sudo apt install python3-evdev python3-can
```

### 2. Python packages (pip)

```bash
pip install adafruit-circuitpython-bno08x   # BNO085 IMU driver
pip install onnxruntime-gpu                 # or onnxruntime (CPU only)
```

If your JetPack is recent enough, install the matching `onnxruntime-gpu` wheel
from NVIDIA's Jetson Zoo so you get TensorRT support automatically — the
runner will pick it up at startup.

### 3. Bring up the CAN bus

The Jetson Orin Nano Super dev kit exposes native CAN on the 40-pin header
(`CAN0_DIN` = pin 29, `CAN0_DOUT` = pin 32) but it is not enabled by
default. One-time setup:

```bash
# Enable CAN pin functions (interactive; pick CAN0_DIN + CAN0_DOUT, save, reboot).
sudo /opt/nvidia/jetson-io/jetson-io.py
```

After reboot, every boot:

```bash
sudo modprobe can can_raw mttcan
sudo ip link set can0 up type can bitrate 1000000
ip -details link show can0    # confirm state UP, bitrate 1000000
```

To make this persistent, drop a `systemd-networkd` config:

```bash
sudo tee /etc/systemd/network/80-can.network >/dev/null <<'EOF'
[Match]
Name=can0
[CAN]
BitRate=1000000
[Link]
RequiredForOnline=no
EOF
sudo systemctl enable --now systemd-networkd
```

If you prefer a USB-CAN dongle (CANable / candleLight / Innomaker /
PEAK-USB), it shows up as `can0` once you `ip link set <iface> up`. The
`motor_bus.yaml` `can_interface` parameter is just the Linux netdev name.

Quick sanity test:

```bash
sudo apt install can-utils
candump can0           # should be quiet at idle
cansend can0 123#DEADBEEF    # should round-trip on a loopback bus
```

### 4. Pair the DualSense

USB-C tether is the simplest path:

```bash
ls /dev/input/by-id/ | grep -i dualsense
```

Or via Bluetooth:

```bash
sudo bluetoothctl
> power on
> agent on
> default-agent
> scan on
# Hold PS + Create until the lightbar pulses, then:
> pair  <MAC>
> trust <MAC>
> connect <MAC>
```

Verify:

```bash
python3 -m evdev.evtest
# pick the "Sony Interactive Entertainment DualSense Wireless Controller" entry
```

### 5. udev rule for non-root access

```bash
sudo usermod -aG input $USER
sudo tee /etc/udev/rules.d/99-dualsense.rules >/dev/null <<'EOF'
SUBSYSTEM=="input", ATTRS{idVendor}=="054c", ATTRS{idProduct}=="0ce6", MODE="0660", GROUP="input"
KERNEL=="hidraw*",  ATTRS{idVendor}=="054c", ATTRS{idProduct}=="0ce6", MODE="0660", GROUP="input"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Log out and back in for the group change to take effect.

## Build

```bash
cd ros2_ws
colcon build --packages-select bebop_pilot --symlink-install
source install/setup.bash
```

## Configure

The two YAMLs you'll edit:

### `config/policy_runner.yaml` — policy + safety

| Key                | What it does                                                              |
|--------------------|---------------------------------------------------------------------------|
| `model_path`       | Path to the trained `policy.onnx`. Pass via launch arg in normal usage.   |
| `control_rate`     | Hz. Match `decimation × sim.dt` from training (V2 base cfg → 100 Hz).     |
| `action_scale`     | Must match `ActionsCfg.joints_pos.scale` (V2 base cfg → 0.8).             |
| `joint_default_pos`| 8-vec. Encoder reading at the training "home" pose. Zeros if encoders match training mechanical zero. |
| `joint_min_pos` / `joint_max_pos` | Hard safety clamps applied to commanded position.          |
| `lin_vel_source`   | `zeros` (stub), `odom` (`nav_msgs/Odometry`), `twist` (`TwistStamped`).   |
| `max_lin_vel` / `max_ang_vel` | Clamp on incoming `/cmd_vel`.                                  |

### `config/ps5_dualsense.yaml` — gamepad behavior

| Key             | What it does                                                |
|-----------------|-------------------------------------------------------------|
| `device_name_substring` | Auto-discovery filter; default `DualSense`.         |
| `deadzone` / `expo`     | Stick shaping; tune for fine center control.        |
| `require_deadman`       | If true, L1 must be held to publish non-zero cmd.   |
| `watchdog_timeout`      | Seconds; falls back to zero command on disconnect.  |

### `config/motor_bus.yaml` — Robstride CAN map

> **Terminology:** `can_interface` is the **Linux netdev name** (e.g. `can0`).
> `motor_ids` are the **Robstride node IDs** you assigned each motor in the
> Robstride debugger (e.g. `31` for the left hip abduction). The full 29-bit
> CAN frame ID is computed per frame from `cmd_type + data + motor_id` — you
> never configure that directly.

V2 motor ID mapping (already filled in):

| Joint                          | Motor ID | Model |
|--------------------------------|---------:|-------|
| `hip_abduction_left_joint`     |       31 | RS04  |
| `hip_abduction_right_joint`    |       41 | RS04  |
| `femur_left_joint`             |       32 | RS03  |
| `femur_right_joint`            |       42 | RS03  |
| `shin_left_joint`              |       34 | RS04  |
| `shin_right_joint`             |       44 | RS04  |
| `foot_left_joint`              |       35 | RS02  |
| `foot_right_joint`             |       45 | RS02  |

Top-level params:

| Key                    | What it does                                                          |
|------------------------|-----------------------------------------------------------------------|
| `can_interface`        | Linux netdev name; `can0` for native or USB dongle.                   |
| `joint_names`          | 8 joint names in the order they should appear in `/joint_states`. **Must** match `bebop_v2_base_cfg.py:JOINT_NAMES_ALL`. |
| `command_rate`         | TX rate to motors (Hz). 100 Hz matches V2 training (`decimation=2 * sim.dt=0.005`). |
| `publish_rate`         | `/joint_states` rate (Hz).                                            |
| `feedback_interval_ms` | Asks each motor to stream feedback every N ms. 5 → 200 Hz.            |
| `watchdog_timeout`     | If `/joint_commands` is older than this, kp drops to 0 (kd-only damping). |
| `enable_on_start`      | If false, motors stay disabled until `True` is published on `/motor_enable`. |
| `set_zero_on_start`    | DANGEROUS: writes current pose as flash mechanical zero. Leave false unless calibrating. |

Per-joint params, looked up under `joints.<joint_name>.*`:

```yaml
joints:
  hip_abduction_left_joint:
    motor_id: 31     # 8-bit Robstride node ID
    model: RS04      # one of RS01 / RS02 / RS03 / RS04
    kp: 200.0        # MIT-mode position gain
    kd: 8.0          # MIT-mode velocity gain
  ...
```

## Run

### Full deployment stack (one command):

```bash
ros2 launch bebop_pilot bringup.launch.py \
    model_path:=/home/jetson/models/policy.onnx
```

### Drive only (assume IMU is already running elsewhere):

```bash
ros2 launch bebop_pilot drive.launch.py \
    model_path:=/home/jetson/models/policy.onnx
```

### Pilot only (no policy — useful for `ros2 topic echo /cmd_vel`):

```bash
ros2 launch bebop_pilot pilot.launch.py
```

### IMU only:

```bash
ros2 run bebop_pilot bno085_imu.py
```

### Motor bus only (no policy — for direct joint testing):

```bash
ros2 run bebop_pilot motor_bus.py --ros-args \
    --params-file install/bebop_pilot/share/bebop_pilot/config/motor_bus.yaml

# Enable motors:
ros2 topic pub --once /motor_enable std_msgs/Bool '{data: true}'

# Send a single zero pose (PD will hold position):
ros2 topic pub --once /joint_commands sensor_msgs/JointState \
    '{name: ["hip_abduction_left_joint","hip_abduction_right_joint","femur_left_joint","femur_right_joint","shin_left_joint","shin_right_joint","foot_left_joint","foot_right_joint"], position: [0,0,0,0,0,0,0,0]}'

# Watch live feedback:
ros2 topic echo /joint_states
```

### Debug timing:

```bash
ros2 run bebop_pilot rate_monitor.py
```

Prints rates + jitter for `/joint_states`, `/joint_commands`, `/imu/data`
every second. Useful for spotting Teensy or transport hiccups.

### Override a single param:

```bash
ros2 run bebop_pilot policy_runner.py --ros-args \
    --params-file install/bebop_pilot/share/bebop_pilot/config/policy_runner.yaml \
    -p model_path:=/path/to/policy.onnx \
    -p control_rate:=100.0
```

## Safety notes

- **Deadman is on by default.** Releasing L1 publishes a zero `Twist` and
  drops `/policy_enable` to `false` on the next tick. The policy runner
  treats this as a stop and stops emitting joint commands.
- The pilot has a 250 ms watchdog on gamepad events; if the controller
  disconnects mid-walk, it falls back to a zero command automatically.
- The policy runner clamps every commanded joint position to
  `[joint_min_pos, joint_max_pos]` before publishing — set these to your
  actual mechanical limits (URDF limits minus ~10% for margin).
- The motor bus has its own watchdog: if `/joint_commands` goes stale
  (default 250 ms), it sends MIT-mode frames with `kp = 0`, leaving only
  `kd` for damping. Combined with the policy runner clamps, this gives
  you defense in depth.
- **Motors are disabled at boot by default** (`enable_on_start: false`).
  You must publish `True` on `/motor_enable` before the bus will energize.
  Recommended bring-up order: launch motor_bus → verify `/joint_states` →
  publish `/motor_enable: true` → launch policy_runner.
- On node shutdown (`Ctrl-C` or kill), `motor_bus` sends `Disable` to
  every motor before closing the CAN socket.
- `lin_vel_source: zeros` keeps the policy mildly out of distribution. Start
  with low commanded speeds (≤ 0.3 m/s) and consider retraining a "blind"
  policy that doesn't observe `base_lin_vel` once you're confident in the
  rest of the stack.

## File layout

```
bebop_pilot/
├── package.xml
├── CMakeLists.txt
├── README.md
├── bebop_pilot/
│   ├── __init__.py
│   └── robstride.py         pure-Python Robstride CAN protocol library
├── scripts/
│   ├── pilot_node.py        DualSense → /cmd_vel, /policy_enable
│   ├── policy_runner.py     ONNX policy → /joint_commands
│   ├── motor_bus.py         CAN bridge: /joint_commands ↔ 8x Robstride ↔ /joint_states
│   ├── bno085_imu.py        BNO085 I2C → /imu/data
│   └── rate_monitor.py      timing diagnostic
├── launch/
│   ├── bringup.launch.py    IMU + motor_bus + pilot + policy
│   ├── drive.launch.py      pilot + policy (assumes IMU + motor_bus running)
│   └── pilot.launch.py      pilot only
└── config/
    ├── ps5_dualsense.yaml   gamepad axis mapping + safety
    ├── policy_runner.yaml   joint defaults, limits, lin_vel source
    └── motor_bus.yaml       Robstride CAN IDs, models, MIT-mode gains
```

## Migration note

If you're coming from the old `bebop_control` package, see
`ros2_ws/src/bebop_control/DEPRECATED.md` for the mapping of old → new
node and topic names. Notably:

- 6-DoF wheel+leg interface → 8-DoF position-only interface.
- `policy_runner.py` (V1) and `policy_runner_v2.py` (transitional) →
  `bebop_pilot/policy_runner.py`.
- `bebop_v2_runtime.yaml` → `bebop_pilot/config/policy_runner.yaml`.
