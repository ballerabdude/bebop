"""Rerun blueprint (dashboard) for navd sessions.

Layout: a 2x2 image/video grid (color and depth, near/far) on the left, and
telemetry time-series (`/cmd_vel`, `/odom`, camera IMU) on the right.

The telemetry messages are Protobuf structs (`bebop.navd.Twist`/`Odom`/
`ImuAccel`/`ImuGyro`), so their fields are plotted directly by mapping each
`SeriesLines` scalar input to a struct field via a jq-style selector — no
export step. If a session predates the protobuf telemetry (JSON channels,
or IMU capture), those series stay empty.

The `power` / `host` views read the firmware's always-on system log
(`system_*.mcap`, Protobuf `bebop.system.*`). Load one or more of those
files alongside the session (pass them as extra paths to `open.py`) to
overlay power draw and host utilization on the drive. Without a system log
those two views stay empty, like the IMU views on old sessions.
"""

import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint.datatypes import (ComponentSourceKind,
                                       VisualizerComponentMapping)

NAME = "bebop_navd"

# (entity path, panel title) for the image grid.
VIEWS = (
    ("/color_near", "Color (near)"),
    ("/color_far", "Color (far)"),
    ("/depth_near", "Depth (near)"),
    ("/depth_far", "Depth (far)"),
)

# origin -> (title, [(source_component, selector, label, rgb)])
TELEMETRY = (
    ("/cmd_vel", "cmd_vel (imitation label)", (
        ("bebop.navd.Twist:message", ".vx", "vx (m/s)", (0, 90, 255)),
        ("bebop.navd.Twist:message", ".wz", "wz (rad/s)", (255, 140, 0)),
    )),
    ("/odom", "odometry", (
        ("bebop.navd.Odom:message", ".x", "x (m)", (0, 90, 255)),
        ("bebop.navd.Odom:message", ".y", "y (m)", (0, 170, 0)),
        ("bebop.navd.Odom:message", ".theta", "theta (rad)", (255, 140, 0)),
    )),
)

# Camera IMU (the recorder's VIO input), one view per topic — the axis field
# names differ between accel (ax..) and gyro (gx..). near/far are paired side
# by side in build().
_X = (230, 60, 60)
_Y = (60, 190, 90)
_Z = (80, 120, 255)
_ACCEL = (("bebop.navd.ImuAccel:message", ".ax", "ax (m/s^2)", _X),
          ("bebop.navd.ImuAccel:message", ".ay", "ay (m/s^2)", _Y),
          ("bebop.navd.ImuAccel:message", ".az", "az (m/s^2)", _Z))
_GYRO = (("bebop.navd.ImuGyro:message", ".gx", "gx (rad/s)", _X),
         ("bebop.navd.ImuGyro:message", ".gy", "gy (rad/s)", _Y),
         ("bebop.navd.ImuGyro:message", ".gz", "gz (rad/s)", _Z))
IMU = (
    ("/imu_accel_near", "IMU accel (near)", _ACCEL),
    ("/imu_gyro_near", "IMU gyro (near)", _GYRO),
    ("/imu_accel_far", "IMU accel (far)", _ACCEL),
    ("/imu_gyro_far", "IMU gyro (far)", _GYRO),
)

# Always-on system log (`system_*.mcap`): power board + host utilization.
# These series only populate when a system log is loaded alongside the
# session (see the module docstring).
_PWR = (
    ("bebop.system.Power:message", ".battery_voltage_v", "battery (V)", (0, 170, 0)),
    ("bebop.system.Power:message", ".motor_voltage_v", "motor (V)", (0, 90, 255)),
    ("bebop.system.Power:message", ".board_temperature_c", "board temp (C)", (255, 140, 0)),
    ("bebop.system.Power:message", ".total_motor_current_a", "motor current (A)", (230, 60, 60)),
    ("bebop.system.Power:message", ".state_of_charge_pct", "SOC (%)", (150, 90, 255)),
)
_HOST = (
    ("bebop.system.Host:message", ".cpu_pct", "cpu (%)", (0, 90, 255)),
    ("bebop.system.Host:message", ".gpu_pct", "gpu (%)", (0, 170, 0)),
    ("bebop.system.Host:message", ".ram_used_pct", "ram (%)", (255, 140, 0)),
    ("bebop.system.Host:message", ".load1", "load1", (230, 60, 60)),
    ("bebop.system.Host:message", ".disk_used_pct", "disk (%)", (150, 90, 255)),
)
SYSTEM = (
    ("/power", "power (system log)", _PWR),
    ("/host", "host (system log)", _HOST),
)


def _series(source_component, selector, label, color):
    mapping = VisualizerComponentMapping(
        target="Scalars:scalars",
        source_kind=ComponentSourceKind.SourceComponent,
        source_component=source_component,
        selector=selector,
    )
    return rr.SeriesLines(names=label, colors=list(color)).visualizer(
        mappings=[mapping])


def _telemetry_view(origin, title, specs):
    visualizers = [_series(*spec) for spec in specs]
    return rrb.TimeSeriesView(origin=origin, name=title,
                              overrides={origin: visualizers})


def build():
    imu_rows = [rrb.Horizontal(_telemetry_view(IMU[i][0], IMU[i][1], IMU[i][2]),
                               _telemetry_view(IMU[i + 1][0], IMU[i + 1][1],
                                               IMU[i + 1][2]))
                for i in range(0, len(IMU), 2)]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Grid(
                *[rrb.Spatial2DView(origin=origin, name=name)
                  for origin, name in VIEWS],
                grid_columns=2,
            ),
            rrb.Vertical(
                *[_telemetry_view(*spec) for spec in TELEMETRY],
                *[_telemetry_view(*spec) for spec in SYSTEM],
                *imu_rows,
            ),
            column_shares=[3.0, 1.0],
        ),
        rrb.TimePanel(state="expanded"),
    )
