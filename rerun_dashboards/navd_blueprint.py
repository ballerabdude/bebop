"""Rerun blueprint (dashboard) for navd sessions.

Layout: a 2x2 image/video grid (color and depth, near/far) on the left, and
telemetry time-series (`/cmd_vel`, `/odom`) on the right.

The telemetry messages are Protobuf structs (`bebop.navd.Twist`/`Odom`), so
their fields are plotted directly by mapping each `SeriesLines` scalar input
to a struct field via a jq-style selector — no export step. If a session
predates the protobuf telemetry (JSON channels), these series stay empty.
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
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Grid(
                *[rrb.Spatial2DView(origin=origin, name=name)
                  for origin, name in VIEWS],
                grid_columns=2,
            ),
            rrb.Vertical(
                *[_telemetry_view(*spec) for spec in TELEMETRY],
            ),
            column_shares=[3.0, 1.0],
        ),
        rrb.TimePanel(state="expanded"),
    )
