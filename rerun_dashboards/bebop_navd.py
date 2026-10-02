"""Generate the Rerun blueprint artifact `bebop_navd.rbl`.

Normally you don't need this: `open.py` loads a session (path or URL) and
sends the dashboard straight from `navd_blueprint.build()`:

    python rerun_dashboards/open.py <session.mcap | URL>

This `.rbl` is only for the plain `rerun` CLI. It is saved under
application id `bebop_navd`, and the viewer only applies a blueprint to
recordings with the *same* app id. Opening a `.mcap` directly names the app
after the file (e.g. `navd_session_20261001_223516.mcap`), so
`rerun x.mcap bebop_navd.rbl` loads the data but leaves the dashboard
unapplied. Convert with a matching app id first, then open the `.rrd`:

    rerun mcap convert --application-id bebop_navd <session>.mcap -o <session>.rrd
    rerun <session>.rrd rerun_dashboards/bebop_navd.rbl

Important: use `Blueprint.save(...)`, which writes a *Blueprint* store. The
older `rr.init(); rr.send_blueprint(); rr.save()` path writes a *Recording*
store that also contains the blueprint, so the viewer would focus that empty
recording (blank panels) or ignore the blueprint when the mcap loads last.

Regenerate after editing the blueprint:

    python rerun_dashboards/bebop_navd.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import navd_blueprint  # noqa: E402

OUTPUT = "bebop_navd.rbl"


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT)
    navd_blueprint.build().save(navd_blueprint.NAME, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
