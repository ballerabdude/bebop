import json, sys
from pathlib import Path
sys.path.insert(0, "/home/bebop/bebop/bebop-vision")
import numpy as np
from bebop_vision.orbbec import CONFIG_DIR, _sdk

def main():
    ob = _sdk()
    ctx = ob.Context()
    devs = ctx.query_devices()
    for i in range(devs.get_count()):
        dev = devs.get_device_by_index(i)
        serial = dev.get_device_info().get_serial_number()
        pipe = ob.Pipeline(dev)
        param = pipe.get_camera_param()
        ci = param.rgb_intrinsic
        rgbd = param.rgb_distortion
        data = {
            "color_width": int(ci.width),
            "color_height": int(ci.height),
            "color_fx": float(ci.fx), "color_fy": float(ci.fy),
            "color_cx": float(ci.cx), "color_cy": float(ci.cy),
            "color_rgb_distortion": {
                "model": int(rgbd.model),
                "k1": float(rgbd.k1), "k2": float(rgbd.k2),
                "p1": float(rgbd.p1), "p2": float(rgbd.p2),
                "k3": float(rgbd.k3), "k4": float(rgbd.k4),
                "k5": float(rgbd.k5), "k6": float(rgbd.k6)},
        }
        ext = param.transform
        rot = np.asarray(ext.rot, dtype=np.float64).reshape(-1)
        tr = np.asarray(ext.transform, dtype=np.float64).reshape(-1)
        print(f"[{serial}] rot({rot.size}): {rot[:4]}... trans({tr.size}): {tr}")
        if rot.size == 9 and np.any(rot):
            data["color_to_depth_transform"] = {
                "rotation": rot.tolist(),
                "translation": tr.tolist() if tr.size >= 3 else [0.0, 0.0, 0.0],
            }
        print(f"[{serial}] color intr: fx={data['color_fx']:.1f} fy={data['color_fy']:.1f} "
              f"cx={data['color_cx']:.1f} cy={data['color_cy']:.1f} ({data['color_width']}x{data['color_height']})")
        path = Path(CONFIG_DIR) / f"orbbec_intrinsics_{serial}.json"
        cur = json.load(open(path))
        cur.update(data)
        json.dump(cur, open(path, "w"), indent=2)
        print(f"[{serial}] -> {path}")
        pipe.stop() if hasattr(pipe, "stop") else None

main()
