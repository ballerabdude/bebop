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
(`bebop-vision/bebop_vision/recorder_mcap.py`). Image/video channels are
Protobuf-encoded Foxglove messages; the custom twist/odom/calib channels
are JSON:

- **Image panels**: `/color_near` + `/color_far` (`foxglove.CompressedVideo`
  H.265, hardware NVENC; falls back to `foxglove.CompressedImage` JPEG),
  `/depth_near` + `/depth_far` (16-bit PNG, turbo colormap)
- **Plots**: teleop twist `/cmd_vel.vx|.wz` (the imitation label), odometry
  `/odom.x|.y|.theta`

The protobuf encoding also lets the native [Rerun](https://rerun.io) viewer
open a session directly (`rerun <session>.mcap`): Rerun maps the color
channels to a `VideoStream` and the depth channels to images, decoding
H.265 via the system FFmpeg with no GPU. (The Rerun web viewer still needs
browser HEVC; Foxglove's WebCodecs has no software HEVC path.)

For the bebop_navd Rerun dashboard (camera grid + telemetry plots), open the
session through the launcher instead; it takes a local path or a capture URL
(cached under `~/.cache/bebop/captures`):

```bash
python rerun_dashboards/open.py \
    http://bebop.local:9090/captures/dl/navd_session_<stamp>.mcap
```

Plain `rerun <session>.mcap rerun_dashboards/bebop_navd.rbl` does *not*
apply the dashboard: the MCAP loader names the app after the file, and a
blueprint only applies to recordings with its own app id (`bebop_navd`).

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
