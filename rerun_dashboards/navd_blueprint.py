"""Rerun blueprint (dashboard) for navd sessions.

Layout: a 2x2 image/video grid (color and depth, near/far) on the left with
the drive notes log under it (`/intent`, `/event`, `/place`), and three tabs
of time-series on the right:

- chassis: `/cmd_vel`, per-wheel velocity against target, mode, and the
  E-STOP / watchdog-stale / operator-seat steps;
- odometry & IMU: `/odom` and the four camera IMU streams;
- power & host: power from the drive's own telemetry, and the firmware
  system log's power / host views.

The wheel, chassis and drive-power panels read `/runtime`, the firmware
telemetry the recorder stores verbatim since 2026-10-23
(`bebop.runtime.v1.ServerRuntimeMessage`): the selectors walk
`.payload.telemetry.<field>`, enums decode as `{name, value}` (so `.mode.value`
plots 1 idle / 2 dial-in / 3 run-policy) and bools plot as 0/1 steps. Wheel
index 0 is left and 1 is right, the firmware's config order. Older sessions
have no `/runtime`, `/intent`, `/event` or `/place`, so those panels stay
empty.

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

# Firmware telemetry stored verbatim in the drive file (`/runtime`).
RUNTIME = "bebop.runtime.v1.ServerRuntimeMessage:message"
_TEL = ".payload.telemetry"
_MEASURED = (230, 60, 60)
_TARGET = (0, 170, 0)
WHEELS = (
    ("/runtime", "left wheel (rad/s)", (
        (RUNTIME, f"{_TEL}.wheels[0].velocity_rad_s", "measured", _MEASURED),
        (RUNTIME, f"{_TEL}.wheels[0].target_velocity_rad_s", "target", _TARGET),
    )),
    ("/runtime", "right wheel (rad/s)", (
        (RUNTIME, f"{_TEL}.wheels[1].velocity_rad_s", "measured", _MEASURED),
        (RUNTIME, f"{_TEL}.wheels[1].target_velocity_rad_s", "target", _TARGET),
    )),
)
MODE = ("/runtime", "mode (1 idle, 2 dial-in, 3 policy)", (
    (RUNTIME, f"{_TEL}.mode.value", "mode", (150, 90, 255)),))
# One small step plot per flag (0/1 lines on a shared plot hide each other);
# single series, so the title says it all and the legend stays off.
FLAGS = (
    ("/runtime", "E-STOP", (
        (RUNTIME, f"{_TEL}.estop_latched", "estop", (255, 140, 0)),)),
    ("/runtime", "watchdog stale", (
        (RUNTIME, f"{_TEL}.drive.operator_stale", "stale", (230, 60, 60)),)),
    ("/runtime", "seat held", (
        (RUNTIME, f"{_TEL}.drive.has_active_operator", "held", (0, 90, 255)),)),
)
POWER_RT = ("/runtime", "power (drive telemetry)", (
    (RUNTIME, f"{_TEL}.power.battery_voltage_v", "battery (V)", (0, 170, 0)),
    (RUNTIME, f"{_TEL}.power.total_motor_current_a", "motor current (A)", (230, 60, 60)),
    (RUNTIME, f"{_TEL}.power.state_of_charge_pct", "SOC (%)", (150, 90, 255)),
))

# Drive notes as one log: TextLog fields mapped from the note messages.
# Events also map their kind (obstacle / recovery / exclude / note) to the
# level column; intent and place rows are told apart by the entity column.
NOTES = (
    ("/intent", "bebop.navd.Intent:message", (("TextLog:text", ".text"),)),
    ("/event", "bebop.navd.Event:message", (("TextLog:text", ".note"),
                                            ("TextLog:level", ".kind"))),
    ("/place", "bebop.navd.PlaceSeen:message", (("TextLog:text", ".name"),)),
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


def _telemetry_view(origin, title, specs, legend=True):
    visualizers = [_series(*spec) for spec in specs]
    extra = {} if legend else {"plot_legend": rrb.PlotLegend(visible=False)}
    return rrb.TimeSeriesView(origin=origin, name=title,
                              overrides={origin: visualizers}, **extra)


def _notes_view():
    overrides = {}
    for origin, source, fields in NOTES:
        mappings = [VisualizerComponentMapping(
            target=target, source_kind=ComponentSourceKind.SourceComponent,
            source_component=source, selector=selector)
            for target, selector in fields]
        overrides[origin] = [rr.TextLog.from_fields().visualizer(mappings=mappings)]
    return rrb.TextLogView(origin="/", contents=[o for o, _, _ in NOTES],
                           name="notes (intent, events, places)",
                           overrides=overrides)


# Right-column tabs, in order (Tabs selects by index).
TABS = ("chassis", "odometry & IMU", "power & host")


def build(active_tab="chassis"):
    """The dashboard; `active_tab` (one of TABS) is the right column's open tab."""
    imu_rows = [rrb.Horizontal(_telemetry_view(IMU[i][0], IMU[i][1], IMU[i][2]),
                               _telemetry_view(IMU[i + 1][0], IMU[i + 1][1],
                                               IMU[i + 1][2]))
                for i in range(0, len(IMU), 2)]
    cmd_vel, odom = TELEMETRY
    chassis = rrb.Vertical(
        _telemetry_view(*cmd_vel),
        *[_telemetry_view(*spec) for spec in WHEELS],
        _telemetry_view(*MODE, legend=False),
        rrb.Horizontal(*[_telemetry_view(*spec, legend=False) for spec in FLAGS]),
        row_shares=[1.0, 1.0, 1.0, 0.7, 0.8],
        name="chassis",
    )
    motion = rrb.Vertical(_telemetry_view(*odom), *imu_rows, name="odometry & IMU")
    power = rrb.Vertical(
        _telemetry_view(*POWER_RT),
        *[_telemetry_view(*spec) for spec in SYSTEM],
        name="power & host",
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Grid(
                    *[rrb.Spatial2DView(origin=origin, name=name)
                      for origin, name in VIEWS],
                    grid_columns=2,
                ),
                _notes_view(),
                row_shares=[3.0, 1.0],
            ),
            rrb.Tabs(chassis, motion, power, active_tab=TABS.index(active_tab)),
            # Camera tiles are height-limited (1280x800 in a 2x2 grid), so the
            # left half loses nothing and the plots get room for their titles.
            column_shares=[1.0, 1.0],
        ),
        # The layout is shipped as active on every load (mcap_to_rrd.py,
        # open.py), so pin the viewer's side panels shut: the dashboard gets the
        # width, and the panels open with one click when needed.
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )
