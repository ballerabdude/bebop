"""Probe: what the recorder's latest-slot reads see (recording-path skew).

The hardware-sync bench (orbbec_sync_test.py) proves capture-instant lock;
this quantifies the SECOND source of stopwatch-test error: the recorder and
videoserver read each camera's independent latest-wins slot at 10 Hz, so the
two stored frames can come from different capture instants. Reports the
near/far slot-arrival delta the way the recorder sees it.

  .venv/bin/python tools/orbbec_slot_probe.py [seconds]
"""

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bebop_vision.orbbec import OrbbecRig  # noqa: E402

RATE_HZ = 10.0
HALF_PERIOD_MS = 1000.0 / 15 / 2  # pairing gate: half the 15 fps period


def stats(vals):
    s = sorted(vals)
    return (f"med={statistics.median(s):.1f} p95={s[int(0.95 * len(s))]:.1f} "
            f"max={s[-1]:.1f}")


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 40
    rig = OrbbecRig(color=True)  # matches main.py --record-navd
    try:
        if not rig.wait_for_pair(timeout=15):
            raise SystemExit("no fresh camera pair within 15 s")
        deltas, ages, unpaired = [], {r: [] for r in rig.cameras}, 0
        n = 0
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            t0 = time.monotonic()
            frames = rig.read_all()
            if all(f is not None for f in frames.values()):
                d = abs(frames["near"].recv_ts - frames["far"].recv_ts) * 1e3
                deltas.append(d)
                if d > HALF_PERIOD_MS:
                    unpaired += 1
                for role, f in frames.items():
                    ages[role].append(f.age_s(t0) * 1e3)
                n += 1
            time.sleep(max(0.0, 1.0 / RATE_HZ - (time.monotonic() - t0)))
        print(f"ticks={n} ({secs:.0f}s @ {RATE_HZ:.0f} Hz, production rig "
              f"class + color)")
        print(f"near/far slot arrival delta ms: {stats(deltas)}")
        for role in sorted(ages):
            print(f"  {role} slot age ms: {stats(ages[role])}")
        print(f"ticks with delta > half period ({HALF_PERIOD_MS:.1f} ms): "
              f"{unpaired}/{n} = {100.0 * unpaired / max(1, n):.1f}% "
              f"(stored as 'same instant' by the recorder today)")
    finally:
        rig.stop()


if __name__ == "__main__":
    main()
