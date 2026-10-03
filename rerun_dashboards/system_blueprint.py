"""Rerun blueprint (dashboard) for the firmware's always-on system log.

System logs (`system_*.mcap`) carry power-board telemetry
(`bebop.system.Power`) and host utilization (`bebop.system.Host`) at 1 Hz —
no video, odometry, or IMU. This is a focused dashboard for those two topics
so a system log opens without the navd dashboard's empty panels.

`mcap_to_rrd.py` picks this blueprint for `system_*` captures and the navd
one for everything else. Loaded under app id `bebop_system`, so its saved
blueprint is separate from the navd dashboard.
"""

import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint.datatypes import (ComponentSourceKind,
                                       VisualizerComponentMapping)

NAME = "bebop_system"

_PWR = "bebop.system.Power:message"
_HOST = "bebop.system.Host:message"

# (source_component, selector, label, rgb)
POWER = (
    (_PWR, ".battery_voltage_v", "battery (V)", (0, 170, 0)),
    (_PWR, ".motor_voltage_v", "motor (V)", (0, 90, 255)),
    (_PWR, ".state_of_charge_pct", "SOC (%)", (150, 90, 255)),
    (_PWR, ".board_temperature_c", "board temp (C)", (255, 140, 0)),
    (_PWR, ".current_al_a", "AL (A)", (230, 60, 60)),
    (_PWR, ".current_ar_a", "AR (A)", (230, 120, 60)),
    (_PWR, ".current_ll_a", "LL (A)", (60, 190, 90)),
    (_PWR, ".current_lr_a", "LR (A)", (80, 120, 255)),
    (_PWR, ".total_motor_current_a", "total (A)", (200, 200, 200)),
)

HOST = (
    (_HOST, ".cpu_pct", "cpu (%)", (0, 90, 255)),
    (_HOST, ".gpu_pct", "gpu (%)", (0, 170, 0)),
    (_HOST, ".ram_used_pct", "ram (%)", (255, 140, 0)),
    (_HOST, ".load1", "load1", (230, 60, 60)),
    (_HOST, ".load5", "load5", (230, 120, 60)),
    (_HOST, ".load15", "load15", (200, 200, 60)),
    (_HOST, ".disk_used_pct", "disk (%)", (150, 90, 255)),
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


def _view(origin, title, specs):
    return rrb.TimeSeriesView(origin=origin, name=title,
                              overrides={origin: [_series(*s) for s in specs]})


def build():
    return rrb.Blueprint(
        rrb.Horizontal(
            _view("/power", "power board", POWER),
            _view("/host", "host", HOST),
            column_shares=[1.0, 1.0],
        ),
        rrb.TimePanel(state="expanded"),
    )
