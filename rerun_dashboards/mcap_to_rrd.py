"""Convert an MCAP capture into a Rerun .rrd that opens with the navd dashboard.

Rerun picks the application id from the loading stream, not from the MCAP,
so a direct `.mcap` open never matches the saved `bebop_navd` blueprint.
This loads the MCAP under that app id with the dashboard and writes a
self-contained `.rrd` (data + default blueprint).

    python mcap_to_rrd.py <in.mcap> <out.rrd>

Runs under the robot's dedicated rerun venv (`scripts/setup-rerun.sh`); the
firmware invokes it from `GET /captures/rerun/<name>` and caches the result
next to the source file. The venv's rerun-sdk version must match the
viewer's, or the .rrd won't load.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rerun as rr  # noqa: E402

import navd_blueprint  # noqa: E402
import system_blueprint  # noqa: E402


def _blueprint_for(in_path: str):
    """(app id, blueprint) — the system dashboard for `system_*` logs, the
    navd dashboard otherwise."""
    name = os.path.basename(in_path)
    if name.startswith("system_"):
        return system_blueprint.NAME, system_blueprint.build()
    return navd_blueprint.NAME, navd_blueprint.build()


def convert(in_path: str, out_path: str) -> None:
    app_id, blueprint = _blueprint_for(in_path)
    rec = rr.RecordingStream(app_id)
    rec.save(out_path, default_blueprint=blueprint)
    rec.log_file_from_path(in_path)
    rec.flush()
    rec.disconnect()


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit("usage: mcap_to_rrd.py <in.mcap> <out.rrd>")
    convert(argv[0], argv[1])


if __name__ == "__main__":
    main()
