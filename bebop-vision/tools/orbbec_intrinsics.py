"""Dump depth intrinsics for connected Orbbec devices to JSON.

Writes config/orbbec_intrinsics_<serial>.json per device — the file BEV
requires (plan Section 6.2). Run once per robot after mounting; re-run if
the firmware update changes calibration.

    python tools/orbbec_intrinsics.py            # all connected devices
    python tools/orbbec_intrinsics.py CPBLC53000PE
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bebop_vision.orbbec import (_negotiate_depth_profile, _sdk, CONFIG_DIR,
                                 intrinsics_path)


def dump(serial=None, config_dir=CONFIG_DIR):
    ob = _sdk()
    ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
    ctx = ob.Context()
    devices = ctx.query_devices()
    n = devices.get_count()
    if n == 0:
        raise SystemExit("no Orbbec devices found (close OrbbecViewer, check udev rules)")
    out = []
    for i in range(n):
        dev = devices.get_device_by_index(i)
        info = dev.get_device_info()
        sn = info.get_serial_number()
        if serial and sn != serial:
            continue
        depth_sensor = None
        sensors = dev.get_sensor_list()
        for j in range(sensors.get_count()):
            s = sensors.get_sensor_by_index(j)
            if s.get_type() == ob.OBSensorType.DEPTH_SENSOR:
                depth_sensor = s
        w, h, fps = _negotiate_depth_profile(depth_sensor, ob, (848, 480, 30))
        config = ob.Config()
        config.enable_video_stream(ob.OBStreamType.DEPTH_STREAM, w, h, fps,
                                   ob.OBFormat.Y16)
        pipeline = ob.Pipeline(dev)
        pipeline.start(config)
        param = pipeline.get_camera_param()
        pipeline.stop()
        intr, dist = param.depth_intrinsic, param.depth_distortion
        data = {
            "serial": sn,
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx), "fy": float(intr.fy),
            "cx": float(intr.cx), "cy": float(intr.cy),
            "distortion": {"model": int(dist.model),
                           "k1": float(dist.k1), "k2": float(dist.k2),
                           "p1": float(dist.p1), "p2": float(dist.p2),
                           "k3": float(dist.k3), "k4": float(dist.k4),
                           "k5": float(dist.k5), "k6": float(dist.k6)},
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = intrinsics_path(sn, config_dir)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"{sn}: {data['width']}x{data['height']} fx={data['fx']:.2f} "
              f"fy={data['fy']:.2f} cx={data['cx']:.2f} cy={data['cy']:.2f} -> {path}")
        out.append(data)
    if not out:
        raise SystemExit(f"device {serial} not found")
    return out


if __name__ == "__main__":
    dump(sys.argv[1] if len(sys.argv) > 1 else None)
