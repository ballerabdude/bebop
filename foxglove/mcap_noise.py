#!/usr/bin/env python3
"""Noise-floor characterization for a Bebop ROS2 MCAP capture.

Reads CDR-encoded channels (/joint_states, /imu) from a ROS2 MCAP file.

Usage:
    pip install mcap
    python3 mcap_noise.py ~/bebop-captures/policy_capture_*.mcap
"""
import sys, struct
from collections import defaultdict

from mcap.reader import make_reader

JOINTS = [
    "hip_flex_L", "hip_flex_R", "hip_abd_L", "hip_abd_R",
    "knee_L", "knee_R", "foot_L", "foot_R",
]

def stats(vals):
    n = len(vals)
    mean = sum(vals) / n if n else 0
    sd = (sum((v - mean)**2 for v in vals) / n) ** 0.5 if n > 1 else 0
    p2p = max(vals) - min(vals)
    rms = (sum(v**2 for v in vals) / n) ** 0.5 if n else 0
    return mean, sd, p2p, rms

def align4(off):
    return (off + 3) & ~3

def dec_string(data, off):
    slen = struct.unpack_from("<I", data, off)[0]; off += 4
    s = data[off:off+slen-1]
    return s, off + slen

def dec_f64_seq(data, off):
    n = struct.unpack_from("<I", data, off)[0]; off += 4
    vals = [struct.unpack_from("<d", data, off + i*8)[0] for i in range(n)]
    return vals, off + n * 8

def read_joint_state(data):
    off = 4        # CDR LE header
    off += 8       # header stamp (sec + nsec)
    _, off = dec_string(data, off)  # frame_id
    off = align4(off)
    n_names = struct.unpack_from("<I", data, off)[0]; off += 4
    names = []
    for _ in range(n_names):
        s, off = dec_string(data, off)
        names.append(s.decode())
        off = align4(off)
    positions, off = dec_f64_seq(data, off)
    velocities, _ = dec_f64_seq(data, off)
    return names, positions, velocities

def read_imu(data):
    off = 4 + 8    # CDR header + stamp
    _, off = dec_string(data, off); off = align4(off)
    quat = [struct.unpack_from("<d", data, off + i*8)[0] for i in range(4)]
    off += 4*8 + 9*8  # quat + covariance
    ang = [struct.unpack_from("<d", data, off + i*8)[0] for i in range(3)]
    return quat, ang

def main(path):
    cols = defaultdict(list)
    n = 0
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            if channel.topic == "/joint_states":
                names, positions, velocities = read_joint_state(message.data)
                for i, (nm, p, v) in enumerate(zip(names, positions, velocities)):
                    label = JOINTS[i] if i < len(JOINTS) else nm
                    cols[f"pos[{i}] {label}"].append(p)
                    cols[f"vel[{i}] {label}"].append(v)
                n += 1
            elif channel.topic == "/imu":
                quat, ang = read_imu(message.data)
                for a, c in zip("xyzw", quat): cols[f"quat_{a}"].append(c)
                for a, c in zip("xyz", ang): cols[f"ang_vel_{a}"].append(c)

    if n == 0:
        sys.exit("no /joint_states messages found")

    print(f"file: {path}")
    print(f"joint_state samples: {n}\n")
    hdr = f"{'signal':30s} {'mean':>11s} {'std':>11s} {'pk-pk':>11s} {'rms':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for fld in sorted(cols.keys()):
        mean, sd, p2p, rms = stats(cols[fld])
        print(f"{fld:30s} {mean:+11.6f} {sd:11.6f} {p2p:11.6f} {rms:11.6f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 mcap_noise.py <file.mcap>")
    main(sys.argv[1])
