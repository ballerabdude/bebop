#!/usr/bin/env python3
"""Print the driver's per-tick decision trace.

    .venv/bin/python tools/nav_trace.py                  # last 300 ticks
    .venv/bin/python tools/nav_trace.py --follow         # live, new rows only
    .venv/bin/python tools/nav_trace.py --host bebop.local --n 900 --all

By default consecutive rows with the same decision are collapsed into one
line with a count, which is what you want for "why did it dither": every
line is a change of mind.  --all prints every tick.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

COLS = ("state", "mode", "chosen_deg", "goal_bearing_deg", "dist_m",
        "clearance_m", "corridor_m", "min_obstacle_m", "reason", "escape",
        "escapes", "commit_deg", "side", "blocked_s", "vx", "wz",
        "heading_deg", "goal")


def fetch(host: str, port: int, n: int) -> list:
    with urllib.request.urlopen(f"http://{host}:{port}/trace?n={n}", timeout=5) as r:
        return json.loads(r.read())["rows"]


def key(row: dict):
    ch = row.get("chosen_deg")
    return (row.get("state"), row.get("mode"), row.get("reason"),
            (row.get("escape") or "").split(" (")[0],
            None if ch is None else (ch > 0) - (ch < 0),
            row.get("commit_deg"), row.get("side"), row.get("goal"))


def fmt(row: dict, count: int = 1, t0: float | None = None) -> str:
    t = row["t"] - (t0 or row["t"])
    parts = [f"{t:7.2f}s", f"x{count:<3d}" if count > 1 else "    "]
    for c in COLS:
        v = row.get(c)
        if v is None:
            continue
        parts.append(f"{c}={v}")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9097)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--all", action="store_true", help="every tick")
    ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()
    seen_t = None
    t0 = None
    last_key, last_row, count = None, None, 0
    while True:
        rows = fetch(args.host, args.port, args.n)
        if seen_t is not None:
            rows = [r for r in rows if r["t"] > seen_t]
        for r in rows:
            seen_t = r["t"]
            if t0 is None:
                t0 = r["t"]
            if args.all:
                print(fmt(r, 1, t0))
                continue
            k = key(r)
            if k == last_key:
                count += 1
                continue
            if last_row is not None:
                print(fmt(last_row, count, t0))
            last_key, last_row, count = k, r, 1
        if not args.follow:
            if last_row is not None and not args.all:
                print(fmt(last_row, count, t0))
            return
        time.sleep(0.5)


if __name__ == "__main__":
    main()
