"""Generate the Rerun blueprint artifact `bebop_navd.rbl`.

The dashboard itself is defined in `navd_blueprint.build()`; this writes it
to a Blueprint-kind `.rbl` that is applied to a session directly:

    rerun <session>.mcap rerun_dashboards/bebop_navd.rbl

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
