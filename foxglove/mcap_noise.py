#!/usr/bin/env python3
"""Noise-floor characterization for a bebop policy_capture MCAP.

Intended for a "robot hanging, motors armed, not moving" capture: every
signal should be ~constant, so std / peak-to-peak / RMS-about-mean
quantify the sensor + actuator-feedback noise floor.

Usage:
    python3 mcap_noise.py <file.mcap>

Requires:
    pip install mcap mcap-protobuf-support protobuf
"""
import sys
import math
import statistics as st
from collections import defaultdict

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

# Joint slot order (see firmware/bebop-linux/src/observation.rs::JOINT_NAMES).
JOINTS = [
    "hip_flex_L", "hip_flex_R", "hip_abd_L", "hip_abd_R",
    "knee_L", "knee_R", "foot_L", "foot_R",
]


def stats(vals):
    n = len(vals)
    mean = st.fmean(vals)
    sd = st.pstdev(vals, mean) if n > 1 else 0.0
    p2p = max(vals) - min(vals)
    rms = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n else 0.0
    return mean, sd, p2p, rms


def main(path):
    cols = defaultdict(list)
    n = 0
    with open(path, "rb") as f:
        rdr = make_reader(f, decoder_factories=[DecoderFactory()])
        for _s, _c, _m, o in rdr.iter_decoded_messages():
            n += 1
            for fld in ["quat_x", "quat_y", "quat_z", "quat_w",
                        "ang_vel_x", "ang_vel_y", "ang_vel_z"]:
                cols[fld].append(getattr(o, fld))
            for i in range(len(o.joint_pos_rad)):
                lbl = JOINTS[i] if i < len(JOINTS) else str(i)
                cols[f"joint_pos_rad[{i}] {lbl}"].append(o.joint_pos_rad[i])
                cols[f"joint_vel_rad_s[{i}] {lbl}"].append(o.joint_vel_rad_s[i])

    print(f"file: {path}")
    print(f"samples: {n}\n")
    hdr = f"{'signal':30s} {'mean':>11s} {'std':>11s} {'pk-pk':>11s} {'rms':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for fld, vals in cols.items():
        mean, sd, p2p, rms = stats(vals)
        print(f"{fld:30s} {mean:+11.6f} {sd:11.6f} {p2p:11.6f} {rms:11.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 mcap_noise.py <file.mcap>")
    main(sys.argv[1])
