"""Bench: verify cross-camera hardware sync on the Orbbec pair (navd rig).

Why: a stopwatch-in-view test through /snapshot or MCAP cannot see sync —
those paths return each camera's latest slot independently (no pairing),
plus per-camera auto-exposure differences.

Method — two domains, one verdict:
  * device stamp domain: each camera stamps frames with its own crystal
    clock (~ppm-scale relative skew). A locked pair STILL sweeps here at
    the clock-skew rate (~1 ms/s), so a sweep is NOT evidence of drift.
  * host monotonic domain: both cameras measured on the same clock. If
    capture instants are phase-locked to the primary's vsync, the
    host-domain phase offset is FLAT (constant per-camera USB/SDK latency);
    if the pair free-runs, it sweeps the 66.7 ms frame period.
  * strict SECONDARY check: in that mode the far camera captures ONLY on
    trigger (ObTypes.h) — frames flowing at all proves the trigger is
    physically delivered.

Stages (depth-only 848x480@15, interleaved serial capture — §2.8):
  A. read back persisted multi-device sync config of both devices
  B. stream both interleaved ~40 s with the production config and report
     both phase analyses
  C. flip far to strict SECONDARY for 15 s (trigger-delivery proof)

Run on the Jetson with no other camera user:
  .venv/bin/python tools/orbbec_sync_test.py
"""

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pyorbbecsdk as ob  # noqa: E402

from bebop_vision.orbbec import load_rig_config  # noqa: E402

DEPTH_W, DEPTH_H, FPS = 848, 480, 15
PERIOD_US = 1_000_000 // FPS


def sync_cfg_apply(dev, role):
    cfg = dev.get_multi_device_sync_config()
    cfg.mode = (ob.OBMultiDeviceSyncMode.PRIMARY if role == "near"
                else ob.OBMultiDeviceSyncMode.SECONDARY_SYNCED)
    cfg.depth_delay_us = 0
    cfg.color_delay_us = 0
    cfg.trigger_to_image_delay_us = 0
    cfg.trigger_out_delay_us = 0
    cfg.trigger_out_enable = role == "near"
    cfg.frames_per_trigger = 1
    dev.set_multi_device_sync_config(cfg)


def sync_cfg_str(dev):
    c = dev.get_multi_device_sync_config()
    mode = str(c.mode).split(".")[-1]
    return (f"mode={mode} depth_dly={c.depth_delay_us} color_dly={c.color_delay_us} "
            f"trig2img={c.trigger_to_image_delay_us} out_en={c.trigger_out_enable} "
            f"out_dly={c.trigger_out_delay_us} fpt={c.frames_per_trigger}")


def _exposure_us(frame):
    try:
        t = ob.OBFrameMetadataType.OB_FRAME_METADATA_TYPE_EXPOSURE
        if frame.has_metadata(t):
            return int(frame.get_metadata(t))
    except Exception:
        pass
    return None


def stream_interleaved(devs, seconds):
    """Round-robin both pipelines in one thread; {role: [(stamp_us, host_t, expo)]}."""
    pipes = {}
    for role, dev in devs.items():
        config = ob.Config()
        config.enable_video_stream(ob.OBStreamType.DEPTH_STREAM, DEPTH_W,
                                   DEPTH_H, FPS, ob.OBFormat.Y16)
        pipe = ob.Pipeline(dev)
        pipe.start(config)
        pipes[role] = pipe
    out = {role: [] for role in pipes}
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        for role, pipe in pipes.items():
            fs = pipe.wait_for_frames(20)
            if fs is None:
                continue
            depth = fs.get_depth_frame()
            if depth is None:
                continue
            out[role].append((int(depth.get_timestamp_us()), time.monotonic(),
                              _exposure_us(depth)))
    for pipe in pipes.values():
        pipe.stop()
    for role, rows in out.items():
        if len(rows) >= 2:
            span = rows[-1][0] - rows[0][0]
            fps = (len(rows) - 1) * 1e6 / span if span > 0 else 0.0
            dts = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
            expo = [e for _, _, e in rows if e is not None]
            ex = (f" expo_us med={statistics.median(expo):.0f}"
                  if expo else "")
            print(f"  [{role}] frames={len(rows)} fps={fps:.2f} "
                  f"dt_ms med={statistics.median(dts)/1e3:.2f} "
                  f"min={min(dts)/1e3:.2f} max={max(dts)/1e3:.2f}{ex}")
        else:
            print(f"  [{role}] frames={len(rows)} (too few to profile)")
    return out


def _slope(times, offs):
    n = len(offs)
    mt, mo = sum(times) / n, sum(offs) / n
    den = sum((t - mt) ** 2 for t in times)
    return (sum((t - mt) * (o - mo) for t, o in zip(times, offs)) / den
            * 10.0) if den else 0.0  # ms per 10 s


def _phase_report(near_rows, far_rows, idx, label):
    """Offset of each near frame to the nearest far frame, mod period."""
    if len(near_rows) < 10 or len(far_rows) < 10:
        print(f"  [phase/{label}] not enough frames")
        return
    far_vals = [r[idx] for r in far_rows]
    offs, times = [], []
    for r in near_rows:
        v, t = r[idx], r[1]
        j = min(range(len(far_vals)), key=lambda k: abs(far_vals[k] - v))
        raw = (v - far_vals[j]) % PERIOD_US
        if raw > PERIOD_US // 2:
            raw -= PERIOD_US
        offs.append(raw / 1e3)
        times.append(t)
    srt = sorted(offs)
    p5, p50, p95 = (srt[int(0.05 * len(srt))], srt[len(srt) // 2],
                    srt[int(0.95 * len(srt))])
    slope = _slope(times, offs)
    print(f"  [phase/{label}] offset ms: p5={p5:+.2f} p50={p50:+.2f} "
          f"p95={p95:+.2f} spread={p95 - p5:.2f} slope={slope:+.2f} ms/10s")
    return p95 - p5, slope


def main():
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cfg = load_rig_config()
    cams = cfg["robots"]["default"]["cameras"]
    ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
    ctx = ob.Context()

    devs, serials = {}, {}
    for serial, c in cams.items():
        role = c["role"]
        dev = ctx.query_devices().get_device_by_serial_number(serial)
        if dev is None:
            raise SystemExit(f"{role} camera {serial} not found (close "
                             f"OrbbecViewer / other users)")
        devs[role] = dev
        serials[role] = serial

    print("=== A. persisted sync config (before any changes) ===")
    for role, dev in devs.items():
        print(f"  {role} ({serials[role]}): {sync_cfg_str(dev)}")

    print(f"=== B. production config, interleaved stream ({secs}s) ===")
    for role, dev in devs.items():
        sync_cfg_apply(dev, role)
    for role, dev in devs.items():  # verify it stuck
        print(f"  {role}: {sync_cfg_str(dev)}")
    rows = stream_interleaved(devs, secs)
    stamp = _phase_report(rows["near"], rows["far"], 0, "device-stamp")
    host = _phase_report(rows["near"], rows["far"], 1, "host-clock")
    print("  ----")
    print("  device-stamp sweep is EXPECTED (independent crystal clocks, "
          "~ppm relative skew)")
    if host is not None:
        spread, slope = host
        if spread < 8.0 and abs(slope) < 1.5:
            print("  HOST-CLOCK PHASE FLAT => capture instants are "
                  "phase-locked (hardware sync OK)")
        else:
            print("  HOST-CLOCK PHASE SWEEPS => pair is free-running "
                  "(sync signal not steering the far camera)")

    print("=== C. trigger-delivery test: far in strict SECONDARY "
          "(captures ONLY on trigger), 15 s ===")
    cfgf = devs["far"].get_multi_device_sync_config()
    cfgf.mode = ob.OBMultiDeviceSyncMode.SECONDARY
    cfgf.trigger_out_enable = False
    devs["far"].set_multi_device_sync_config(cfgf)
    print(f"  far: {sync_cfg_str(devs['far'])}")
    far2 = stream_interleaved({"far": devs["far"]}, 15)["far"]
    print(f"  VERDICT: {'TRIGGER DELIVERED — sync hub path works' if len(far2) > 20 else 'NO TRIGGER — far camera gets no sync signal (wiring/hub/mode)'}")

    print("=== restore production config ===")
    for role, dev in devs.items():
        sync_cfg_apply(dev, role)
        print(f"  {role}: {sync_cfg_str(dev)}")


if __name__ == "__main__":
    main()
