"""Convert an MCAP capture into a Rerun .rrd that opens with the dashboard.

Rerun picks the application id from the loading stream, not from the MCAP,
so a direct `.mcap` open never matches the saved blueprint. This loads the
MCAP under the right app id with the matching dashboard and writes a
self-contained `.rrd` (data + default blueprint). `system_*` captures get
the `bebop_system` dashboard (power + host); everything else gets the navd
dashboard.

A footer-less MCAP (a live segment, or one killed before the SIGTERM handler
ran) makes Rerun's loader silently produce an empty recording, so we repair
it first: re-stream the complete chunks into a terminated temp MCAP.

    python mcap_to_rrd.py <in.mcap> <out.rrd>

Runs under the robot's dedicated rerun venv (`scripts/setup-rerun.sh`); the
firmware invokes it from `GET /captures/rrd/<stem>.rrd`. The venv's
rerun-sdk version must match the viewer's, or the .rrd won't load.
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
    import mcap_prep  # same directory: footer repair + /runtime split
    app_id, blueprint = _blueprint_for(in_path)
    load_path, repaired = mcap_prep.prepare(in_path, out_path + ".src.mcap")
    try:
        rec = rr.RecordingStream(app_id)
        rec.save(out_path)
        # The viewer keeps a per-app-id blueprint and prefers it over a file's
        # default, so a dashboard change would never show for anyone who has
        # opened a session before: ship it as the active layout. make_default
        # also makes it the default, so this is the only copy in the file
        # (open.py does the same).
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.log_file_from_path(load_path)
        rec.flush()
        rec.disconnect()
    finally:
        if repaired is not None and os.path.exists(repaired):
            os.remove(repaired)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit("usage: mcap_to_rrd.py <in.mcap> <out.rrd>")
    convert(argv[0], argv[1])


if __name__ == "__main__":
    main()
