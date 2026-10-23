"""Mirror drive sessions and the firmware health log from the robot over HTTP.

The robot's capture dir (/var/lib/bebop-captures) is root-only, so rsync as
the ``bebop`` user cannot read it.  The firmware serves the same files:
``GET /captures`` lists them and ``GET /captures/dl/<name>`` streams one
(with Range support).  This needs no robot-side change.

    python tools/pull_captures.py [--robot http://bebop.local:9090] \\
        [--dst datasets/sessions]

- ``POST /captures/finalize`` first (``--no-finalize`` to skip), so the live
  health log rolls over and the copy taken has an MCAP footer.
- Skips files the firmware flags ``active`` and files modified less than
  ``--min-age-s`` ago, measured on the robot's own clock (the HTTP Date
  header): a drive segment still being recorded is never flagged active,
  because the firmware does not know about it.
- Downloads into ``<name>.part`` and renames when the size matches, resuming
  a partial ``.part`` with a Range request.  A file already present at the
  remote size is skipped.
"""

from __future__ import annotations

import argparse
import email.utils
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PREFIXES = ("navd_session_", "system_")
CHUNK = 1 << 20


def _robot_now_ms(headers) -> float:
    """The robot's clock from the Date header, else this machine's."""
    try:
        return email.utils.parsedate_to_datetime(headers["Date"]).timestamp() * 1000.0
    except (KeyError, TypeError, ValueError):
        return time.time() * 1000.0


def list_remote(robot, timeout=10.0):
    """(files, robot_now_ms) from GET /captures."""
    with urllib.request.urlopen(f"{robot}/captures", timeout=timeout) as resp:
        files = json.load(resp)["files"]
        return files, _robot_now_ms(resp.headers)


def select(files, now_ms, *, prefixes=PREFIXES, min_age_s=30.0):
    """Finished captures worth pulling: matching prefix, not active, old enough."""
    out = []
    for f in files:
        name = f.get("name", "")
        if not name.startswith(prefixes) or "/" in name or f.get("active"):
            continue
        if now_ms - float(f.get("modifiedMs", 0)) < min_age_s * 1000.0:
            continue
        out.append(f)
    return out


def fetch(robot, name, size, dst: Path, timeout=30.0) -> str:
    """Download ``name`` into ``dst``; returns 'have', 'pulled' or 'resumed'."""
    final = dst / name
    if final.exists() and final.stat().st_size == size:
        return "have"
    part = dst / (name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    req = urllib.request.Request(f"{robot}/captures/dl/{name}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resumed = bool(have) and resp.status == 206
        with open(part, "ab" if resumed else "wb") as out:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                out.write(block)
    got = part.stat().st_size
    if got != size:
        raise OSError(f"{name}: got {got} bytes, expected {size}")
    os.replace(part, final)
    return "resumed" if resumed else "pulled"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--robot", default="http://bebop.local:9090")
    ap.add_argument("--dst", default="datasets/sessions")
    ap.add_argument("--min-age-s", type=float, default=30.0)
    ap.add_argument("--no-finalize", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be pulled; no finalize, no download")
    args = ap.parse_args(argv)
    robot = args.robot.rstrip("/")
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not (args.no_finalize or args.dry_run):
        try:
            req = urllib.request.Request(f"{robot}/captures/finalize", method="POST")
            urllib.request.urlopen(req, timeout=10.0).close()
        except (urllib.error.URLError, OSError) as exc:
            print(f"[pull] finalize failed ({exc}); continuing")
    try:
        files, now_ms = list_remote(robot)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        print(f"[pull] cannot list {robot}/captures: {exc}")
        return 1
    chosen = select(files, now_ms, min_age_s=args.min_age_s)
    if args.dry_run:
        todo = [f for f in chosen
                if not ((dst / f["name"]).exists()
                        and (dst / f["name"]).stat().st_size == int(f.get("sizeBytes", 0)))]
        for f in todo:
            print(f"[pull] would pull {f['name']} ({int(f.get('sizeBytes', 0)) / 1e6:.1f} MB)")
        total = sum(int(f.get("sizeBytes", 0)) for f in todo) / 1e9
        print(f"[pull] dry run: {len(todo)} of {len(chosen)} eligible ({len(files)} listed), "
              f"{total:.1f} GB to pull -> {dst}")
        return 0
    counts = {"have": 0, "pulled": 0, "resumed": 0, "failed": 0}
    t0 = time.monotonic()
    for f in chosen:
        name, size = f["name"], int(f.get("sizeBytes", 0))
        try:
            state = fetch(robot, name, size, dst)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[pull] {name}: {exc}")
            counts["failed"] += 1
            continue
        counts[state] += 1
        if state != "have":
            print(f"[pull] {state:7} {name} ({size / 1e6:.1f} MB)")
    print(f"[pull] {len(chosen)} of {len(files)} listed: {counts['pulled']} pulled, "
          f"{counts['resumed']} resumed, {counts['have']} already here, "
          f"{counts['failed']} failed ({time.monotonic() - t0:.1f} s) -> {dst}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
