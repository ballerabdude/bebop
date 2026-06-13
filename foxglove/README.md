# Foxglove tooling

Scripts and layout definitions for reviewing Bebop **policy capture** MCAP
recordings in [Foxglove](https://foxglove.dev/). Captures are written by
`firmware/bebop-linux` as protobuf-encoded samples on topic `/policy_capture`
(see `jetson-agent/bebop-proto/proto/bebop_capture.proto`).

## Quick start

1. Open an MCAP file in Foxglove Desktop (e.g. from `~/Downloads/`).
2. Import a layout: **Layouts → Import from file…** and choose one of the
   generated JSON files below.
3. For the robot layout, set **Settings → Desktop → ROS_PACKAGE_PATH** to:
   ```
   /Users/ahagi/Documents/projects/bebop/ros2/src
   ```
   This lets Foxglove resolve `package://bebopv2_description/...` mesh paths in
   the URDF.

Regenerate layouts after editing the Python sources:

```bash
python3 foxglove/make_foxglove_layout.py --layout all --force
```

## Files

| File | Purpose |
| --- | --- |
| `layout_common.py` | Shared helpers: `/policy_capture` plot paths, panel config, and the top-level Foxglove JSON envelope. |
| `layout_robot.py` | Builds the **robot review** layout: 3D URDF panel, notes panel, and plots for joint position/velocity, IMU quaternion, and gyro. |
| `layout_noise.py` | Builds the **noise review** layout: original 2×2 plot mosaic for static-capture noise analysis. |
| `make_foxglove_layout.py` | CLI entry point that writes one or all layout JSON files. Skips existing outputs unless `--force` is passed. |
| `bebop_robot_layout.json` | Generated layout for URDF + position/IMU plots. Import this into Foxglove. |
| `bebop_noise_layout.json` | Generated layout for noise-floor plotting. Import this into Foxglove. |
| `mcap_noise.py` | Offline CLI that prints mean / std / peak-to-peak / RMS stats per signal in a capture file. |

## Layouts

### Robot (`bebop_robot_layout.json`)

- **3D panel** loads the local URDF at
  `ros2/src/bebopv2_description/urdf/bebopv2.urdf`.
- **Plot panels** read scalar and array fields from `/policy_capture`:
  - `joint_pos_rad[]`, `joint_vel_rad_s[]`
  - `quat_x/y/z/w`, `ang_vel_x/y/z`
  - X axis: `sim_time_s`
- Joint labels match URDF / firmware order (`hip_flexion_left_joint`, …).

The 3D panel is configured for `/joint_states` animation. Current captures
store positions in `/policy_capture.joint_pos_rad[]` instead, so the plots show
motion even if the URDF stays in its default pose. Publish or derive
`/joint_states` if you need the 3D model to move during playback.

Edit paths at the top of `layout_robot.py` (`MCAP_PATH`, `URDF_PATH`) if your
files live elsewhere.

### Noise (`bebop_noise_layout.json`)

Four synced plots for reviewing a **stationary** capture (robot hanging, motors
armed, not moving):

- IMU angular velocity (gyro noise)
- Joint velocity (typically noisiest channel)
- IMU quaternion (orientation drift)
- Joint position (encoder noise)

Uses short joint labels (`hip_flex_L`, …) in the legend.

## Generating layouts

Combined CLI:

```bash
python3 foxglove/make_foxglove_layout.py --layout robot
python3 foxglove/make_foxglove_layout.py --layout noise
python3 foxglove/make_foxglove_layout.py --layout all
python3 foxglove/make_foxglove_layout.py --layout noise --out /tmp/custom.json
python3 foxglove/make_foxglove_layout.py --layout robot --force
```

Or run a layout module directly:

```bash
python3 foxglove/layout_robot.py
python3 foxglove/layout_noise.py --force
```

By default, existing JSON files are **not** overwritten. Pass `--force` to
replace them.

### Adding a new layout

1. Add `layout_<name>.py` with `NAME`, `OUTPUT`, and `build()` returning a
   Foxglove layout dict (use `layout_common` helpers).
2. Register it in `make_foxglove_layout.py` under `LAYOUTS`.
3. Generate with `--layout <name>`.

## MCAP noise analysis

`mcap_noise.py` summarizes sensor noise in a capture without Foxglove:

```bash
pip install mcap mcap-protobuf-support protobuf
python3 foxglove/mcap_noise.py ~/Downloads/policy_capture_20260612_032423.mcap
```

Prints per-field mean, standard deviation, peak-to-peak, and RMS-about-mean for
IMU and joint signals on `/policy_capture`.

## Related repo paths

- URDF: `ros2/src/bebopv2_description/urdf/bebopv2.urdf`
- Capture schema: `jetson-agent/bebop-proto/proto/bebop_capture.proto`
- Firmware writer: `firmware/bebop-linux/src/policy_capture.rs`
- Joint order: `firmware/bebop-linux/src/observation.rs` (`JOINT_NAMES`)
