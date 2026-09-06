"""Extract all mirrored navd sessions and produce a data audit."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "datasets" / "sessions"
OUT = ROOT / "datasets" / "navd-v0"


def audit(out_dir):
    rows = [json.loads(l) for l in open(out_dir / "manifest.jsonl")]
    n = len(rows)
    color = sum(r["has_color"] for r in rows)
    color_far = sum(r["has_color_far"] for r in rows)
    plane = [r["plane_ok"] for r in rows]
    plane_near = sum(p.get("near", False) for p in plane)
    plane_far = sum(p.get("far", False) for p in plane)
    # depth validity on a sample
    valid = {"near": [], "far": []}
    mid = {"near": [], "far": []}
    import cv2
    for r in rows[:: max(1, n // 60)]:
        d = np.load(out_dir / "depth" / f"{r['stamp_ns']:020d}.npz")
        for role in ("near", "far"):
            a = d[role] if role in d.files else None
            if a is None or a.size == 0:
                continue
            v = a[a > 0]
            valid[role].append(v.size / a.size)
            if v.size:
                mid[role].append(float(np.median(v.astype(float))) / 1000.0)
    return {
        "ticks": n,
        "has_color%": round(100 * color / n),
        "has_color_far%": round(100 * color_far / n),
        "plane_near%": round(100 * plane_near / n),
        "plane_far%": round(100 * plane_far / n),
        "valid_near%": round(100 * float(np.mean(valid["near"]))) if valid["near"] else -1,
        "valid_far%": round(100 * float(np.mean(valid["far"]))) if valid["far"] else -1,
        "med_near_m": round(float(np.median(mid["near"])), 2) if mid["near"] else -1,
        "med_far_m": round(float(np.median(mid["far"])), 2) if mid["far"] else -1,
    }


def main():
    files = sorted(SESSIONS.glob("navd_session_*.mcap"))
    print(f"{len(files)} session files")
    for f in files:
        name = f.stem
        out = OUT / name
        if (out / "manifest.jsonl").exists():
            print(f"{name}: already extracted")
        else:
            print(f"{name}: extracting ...", flush=True)
            r = subprocess.run(
                [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "tools" / "mcap_extract.py"),
                 str(f), str(out)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAILED: {r.stderr[-300:]}")
                continue
        stats = audit(out)
        print(f"  {json.dumps(stats)}")


if __name__ == "__main__":
    main()
