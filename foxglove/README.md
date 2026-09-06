# Foxglove tooling

Scripts and layout definitions for reviewing Bebop **policy capture** MCAP
recordings in [Foxglove](https://foxglove.dev/). Captures are now written as
**ROS2 CDR-encoded MCAP** (profile=`ros2`, schema=`ros2msg`, message=`cdr`)
by `firmware/bebop-linux/src/policy_capture.rs`.

## Quick start

1. Open an MCAP file in Foxglove Desktop.
2. Import a layout: **Layouts -> Import from file...** and choose one of the
   generated JSON files.
3. For the robot layout, set **Settings -> Desktop -> ROS_PACKAGE_PATH** to:
   ```
   /Users/ahagi/Documents/projects/bebop/ros2/src
   ```

Regenerate layouts after editing the Python sources:

```bash
python3 foxglove/make_foxglove_layout.py --layout all --force
```

## Channels

The ROS2 MCAP file contains 5 channels:

| Channel | ROS2 Type |
| --- | --- |
| `/joint_states` | `sensor_msgs/msg/JointState` |
| `/imu` | `sensor_msgs/msg/Imu` |
| `/policy/status` | `bebop_msgs/msg/PolicyStatus` |
| `/policy/observation` | `bebop_msgs/msg/Float32Stamped` |
| `/policy/action` | `bebop_msgs/msg/PolicyAction` |

## Layouts

### Robot (`bebop_robot_layout.json`)

- **3D panel** loads the URDF and animates from `/joint_states`
- **Plot panels** read from `/joint_states.position[]` / `.velocity[]`,
  `/imu.orientation` / `.angular_velocity`, with X axis `/policy/status.sim_time_s`

### Policy debug (`bebop_policy_layout.json`)

Same animated URDF **3D panel** as the robot layout, plus two plot columns
for debugging policy behavior tick by tick:

- **Inputs** (`/policy/observation.data[]`): base angular velocity,
  projected gravity, joint pos/vel (relative, scaled), and velocity
  commands — the exact 49-element vector fed to the network.
- **Outputs** (`/policy/action.*`): decoded position targets (rad), `kp`,
  and `kd`.

See `firmware/bebop-linux/src/observation.rs::build` for the observation
index layout.

### Noise (`bebop_noise_layout.json`)

2x2 mosaic for static-capture noise review.

### navd sessions (`bebop_navd_layout.json`)

Review layout for navd recorder-v2 MCAP sessions
(`bebop-vision/bebop_vision/recorder_mcap.py`, JSON-encoded channels):

- **Image panels**: `/color_near` (`foxglove.CompressedImage`),
  `/depth_near_preview` (16uc1 turbo depth preview),
  `/bev_map` (top-down 60x60 teacher occupancy map with goal arrow)
- **Plots**: teleop twist `/cmd_vel.vx|.wz` (the imitation label),
  odometry, goal vs heading, ground-plane-fit health

Record a session on the robot, copy it off, open it here:

```bash
# robot
python main.py --record-navd /tmp/navd_sessions --seconds 600
# workstation
scp bebop@bebop.local:/tmp/navd_sessions/*.mcap datasets/sessions/
```

## MCAP noise analysis

```bash
pip install mcap
python3 foxglove/mcap_noise.py ~/bebop-captures/policy_capture_*.mcap
```
