#!/usr/bin/env python3
"""Ad-hoc CPU benchmark: libx264 software encode of both Orbbec RGB streams.

Orin Nano has no NVENC (Jetson L4T DeveloperGuide "Software Encode in Orin
Nano"), so operator/mission video on this robot has to go through libx264.
This tool answers "what does that cost?" for the real rig:

  1. open both cameras in RAW RGB (1280x800@15, matching the rig) and dump
     N frames per camera to a raw cache (camera + depth filters stopped
     before any measurement, so they never pollute the encode numbers);
  2. replay the cache through libx264 at each requested preset, sampling
     /proc/stat (system) and /proc/self/stat (process) for CPU load;
  3. print fps, process-cores, and per-core system usage per preset.

x264 options default to the NVIDIA "hardware-like GOP" tuning from the app
note (IDR/I = 30, refs = 1, no B-frames, AQ off) so the numbers line up
with the doc's tuned table.

Run on the Jetson from the repo root, no camera holder running:
  sudo .venv/bin/python tools/h264_sw_encode_bench.py --frames 45
"""

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import av  # noqa: E402
from bebop_vision.orbbec import (  # noqa: E402
    DEFAULT_RIG_YAML, OrbbecCamera, load_rig_config, parse_self_mask_entries)

DEFAULT_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
                   "medium", "slow"]


def _ticks_per_s():
    return os.sysconf("SC_CLK_TCK")


def _proc_cpu_ticks():
    with open("/proc/self/stat") as f:
        parts = f.read().split()
    return int(parts[13]) + int(parts[14])


def _stat_snapshot():
    """(total_ticks, idle_ticks, ncpu) aggregated across all CPUs."""
    total = idle = ncpu = 0
    with open("/proc/stat") as f:
        for line in f:
            parts = line.split()
            if not parts or not parts[0].startswith("cpu"):
                continue
            if parts[0] == "cpu":
                vals = [int(v) for v in parts[1:]]
                total += sum(vals)
                idle += vals[3] + (vals[4] if len(vals) > 4 else 0)
            else:
                ncpu += 1
    return total, idle, ncpu


class CpuSampler:
    """Background /proc/stat + process-CPU sampler."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        t0, i0, _ = _stat_snapshot()
        p0 = _proc_cpu_ticks()
        w0 = time.monotonic()
        while not self._stop.wait(self.interval):
            t1, i1, _ = _stat_snapshot()
            p1 = _proc_cpu_ticks()
            w1 = time.monotonic()
            dt, di = t1 - t0, i1 - i0
            self.samples.append({
                "sys_pct": 100.0 * (dt - di) / dt if dt else 0.0,
                "proc_cores": ((p1 - p0) / _ticks_per_s()) / (w1 - w0)
                if w1 > w0 else 0.0,
            })
            t0, i0, p0, w0 = t1, i1, p1, w1

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {"sys_pct": 0.0, "proc_cores": 0.0, "sys_peak_pct": 0.0,
                    "proc_peak_cores": 0.0}
        return {
            "sys_pct": float(np.mean([s["sys_pct"] for s in self.samples])),
            "sys_peak_pct": float(np.max([s["sys_pct"] for s in self.samples])),
            "proc_cores": float(np.mean([s["proc_cores"] for s in self.samples])),
            "proc_peak_cores": float(np.max([s["proc_cores"]
                                             for s in self.samples])),
        }


def capture(roles, frames, w, h, fps, out_dir, warmup_s=1.5):
    """Open both cameras in RGB, dump `frames` raw frames each, stop them."""
    cfg = load_rig_config(DEFAULT_RIG_YAML)
    cams = cfg["robots"]["default"]["cameras"]
    by_role = {c["role"]: (serial, c) for serial, c in cams.items()}
    rig = {}
    try:
        for role in roles:
            serial, c = by_role[role]
            rects, polys = parse_self_mask_entries(c.get("self_mask_pixels"))
            rig[role] = OrbbecCamera(
                serial=serial, role=role,
                depth_profile=(848, 480, fps),
                color_profile=(w, h, fps),
                color_format="rgb",
                mask_rects=rects, mask_polys=polys)
        print(f"[capture] cameras open; warming up {warmup_s:.1f}s ...")
        time.sleep(warmup_s)
        caches = {role: out_dir / f"{role}_{w}x{h}.raw" for role in roles}
        counts = {role: 0 for role in roles}
        last = {role: None for role in roles}
        fhs = {role: open(p, "wb") for role, p in caches.items()}
        t0 = time.monotonic()
        try:
            while any(counts[r] < frames for r in roles):
                for role in roles:
                    if counts[role] >= frames:
                        continue
                    fr = rig[role].read()
                    if fr is None or fr.color is None or fr is last[role]:
                        continue
                    last[role] = fr
                    if fr.color.shape[:2] != (h, w):
                        raise RuntimeError(
                            f"{role}: got color {fr.color.shape[:2]}, want {(h, w)}")
                    fhs[role].write(fr.color.tobytes())
                    counts[role] += 1
                time.sleep(1.0 / (fps * 2))
        finally:
            for fh in fhs.values():
                fh.close()
        dt = time.monotonic() - t0
        print(f"[capture] {counts} frames in {dt:.1f}s "
              f"({sum(counts.values()) / dt:.1f} fps aggregate)")
    finally:
        for cam in rig.values():
            cam.stop()
    return caches


def encode_cache(path, w, h, fps, preset, bitrate, hw_gop, sink):
    """Encode one raw RGB cache through libx264; return (n, wall_s)."""
    stream = sink.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    opts = {"preset": preset}
    if hw_gop:
        opts["x264-params"] = "keyint=30:min-keyint=30:refs=1:bframes=0:aq-mode=0"
    if bitrate:
        opts["b"] = str(bitrate)
        opts["maxrate"] = str(bitrate)
        opts["bufsize"] = str(bitrate * 2)
    else:
        opts["crf"] = "23"
    stream.options = opts

    frame_bytes = w * h * 3
    n = 0
    t0 = time.monotonic()
    with open(path, "rb") as f:
        while True:
            buf = f.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            vf = av.VideoFrame.from_ndarray(
                arr, format="rgb24").reformat(format="yuv420p")
            for pkt in stream.encode(vf):
                sink.mux(pkt)
            n += 1
    for pkt in stream.encode():
        sink.mux(pkt)
    wall = time.monotonic() - t0
    return n, wall


def run_encode(path, w, h, fps, preset, bitrate, hw_gop, sink_fmt):
    fd, tmp = tempfile.mkstemp(suffix=".h264", dir="/tmp")
    os.close(fd)
    sink = av.open(tmp, mode="w", format=sink_fmt)
    sampler = CpuSampler()
    sampler.start()
    try:
        n, wall = encode_cache(path, w, h, fps, preset, bitrate, hw_gop, sink)
    finally:
        stats = sampler.stop()
        sink.close()
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return n, wall, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=45,
                    help="RGB frames to capture per camera (default 45 = 3s@15)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    ap.add_argument("--bitrate", type=int, default=0,
                    help="target bitrate bps (ABR); 0 = CRF 23")
    ap.add_argument("--no-hw-gop", action="store_true",
                    help="do not force the HW-like GOP x264 params")
    ap.add_argument("--roles", nargs="+", default=["near", "far"])
    ap.add_argument("--cache-dir", default="/var/tmp/h264_sw_bench")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing raw cache (skip camera capture)")
    args = ap.parse_args()

    out_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = {r: out_dir / f"{r}_{args.width}x{args.height}.raw"
              for r in args.roles}

    if args.reuse and all(p.exists() for p in caches.values()):
        print(f"[capture] reusing cache in {out_dir}")
    else:
        caches = capture(args.roles, args.frames, args.width, args.height,
                         args.fps, out_dir)

    _, _, ncpu = _stat_snapshot()
    frame_mb = args.width * args.height * 3 / 1e6
    print(f"\n[bench] {ncpu} cores; {frame_mb:.1f} MB/frame; "
          f"real-time budget = {1000.0 / args.fps:.1f} ms/frame/camera")
    print(f"[bench] {'preset':>9} {'role':>4} {'fps':>7} {'ms/fr':>7} "
          f"{'pcores':>7} {'sys%':>6} {'syspk%':>7}")

    for preset in args.presets:
        for role in args.roles:
            n, wall, st = run_encode(
                caches[role], args.width, args.height, args.fps, preset,
                args.bitrate, not args.no_hw_gop, "h264")
            fps = n / wall if wall else 0.0
            ms = 1000.0 * wall / n if n else 0.0
            print(f"[bench] {preset:>9} {role:>4} {fps:7.1f} {ms:7.1f} "
                  f"{st['proc_cores']:7.2f} {st['sys_pct']:6.1f} "
                  f"{st['sys_peak_pct']:7.1f}")

    print("\n[bench] pcores = cores burned by this process (encode+swscale). "
          "Two cameras at real time\n"
          "        need pcores <= 2.00 (i.e. pcores <= 1.00 per camera) to "
          "leave the robot's\n"
          "        BEV/fusion work on the other cores.")


if __name__ == "__main__":
    main()