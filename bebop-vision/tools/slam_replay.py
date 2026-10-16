#!/usr/bin/env python3
"""Offline SLAM replay — navd MCAP session -> PyCuVSLAM trajectory.

Phase 0 of the SLAM plan (docs/slam.md): read a recorded navd session
(`recorder_mcap.py` output), feed the chosen camera's RGB-D stream and its
on-camera IMU through NVIDIA cuVSLAM, and write the estimated trajectory next
to the wheel odometry so the wheel-slip question can be answered offline,
with no robot and no camera ownership conflict.

This tool does NOT open any camera and never talks to the robot.

Inputs (all already in the MCAP):
  /color_<role>      foxglove.CompressedVideo  H.265 access units
  /depth_<role>      foxglove.CompressedImage  16-bit PNG (mm)
  /imu_accel_<role>  bebop.navd.ImuAccel       device_stamp_us (true clock)
  /imu_gyro_<role>   bebop.navd.ImuGyro
  /odom              bebop.navd.Odom           wheel odometry (compare only)
The per-serial calibration (color intrinsics + distortion + color->depth
transform) is NOT in the MCAP /calib block yet, so it is read from
config/orbbec_intrinsics_<serial>.json.

Run from a python 3.12 env with PyCuVSLAM + av + opencv + mcap installed
(e.g. /tmp/opencode/cuvslam-venv):

  <venv>/bin/python tools/slam_replay.py \
      /path/navd_session_XXXX.mcap --camera far --mode rgbd [--slam]

cuVSLAM RGB-D occasionally relocalizes with a physically impossible
single-frame teleport (metres in ~33 ms) — confirmed as tracker glitches,
not motion, by the wheel odom. `--max-step-m` (default 0.25 m) re-anchors
the trajectory to the last good pose when it detects one, instead of
writing a fake jump into the CSV/Rerun trajectory. 0 disables the guard.

Outputs go to <mcap stem>.slam/:
  trajectory_cuvslam.csv   t_ns,x,y,z,qx,qy,qz,qw  (rig = camera optical frame)
  trajectory_wheel.csv     t_ns,x,y,theta
  metrics.json             path lengths, rotation totals, odom comparison
  trajectory_plot.png      top-down overlay (cuVSLAM vs wheel odom)
  align_XXXX.png           optional: depth->color alignment sanity overlays
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bebop_vision.proto.foxglove import (  # noqa: E402
    CompressedImage_pb2 as ci_pb,
    CompressedVideo_pb2 as cv_pb,
)
from bebop_vision.proto.navd import telemetry_pb2 as navd_pb  # noqa: E402

try:
    import av
except ImportError:  # pragma: no cover
    av = None

try:
    import cuvslam as vslam
except ImportError:  # pragma: no cover
    vslam = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import rerun as rr
except ImportError:  # pragma: no cover
    rr = None

from mcap.reader import make_reader  # noqa: E402

# RealSense defaults shipped with the cuVSLAM examples; the Orbbec IMU has no
# published noise model yet, so reuse these as a reasonable prior.
IMU_GYRO_NOISE_DENSITY = 6.0673370376614875e-03
IMU_GYRO_RANDOM_WALK = 3.6211951458325785e-05
IMU_ACCEL_NOISE_DENSITY = 3.3621979208052800e-02
IMU_ACCEL_RANDOM_WALK = 9.8256589971851467e-04
IMU_RATE_HZ = 200.0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def load_rig_roles(config_dir: Path) -> dict[str, str]:
    """role -> serial, from config/orbbec_rig.yaml."""
    path = config_dir / "orbbec_rig.yaml"
    if yaml is None or not path.exists():
        return {"near": "CPBLC53000PE", "far": "CPBLC53000ED"}
    data = yaml.safe_load(path.read_text())
    cams = data["robots"]["default"]["cameras"]
    return {v["role"]: k for k, v in cams.items()}


def load_intrinsics(config_dir: Path, serial: str) -> dict:
    path = config_dir / f"orbbec_intrinsics_{serial}.json"
    return json.loads(path.read_text())


def _poly_params(cal: dict) -> np.ndarray:
    d = cal["color_rgb_distortion"]
    return np.array([d["k1"], d["k2"], d["p1"], d["p2"],
                     d["k3"], d["k4"], d["k5"], d["k6"]], dtype=np.float64)


class DepthAligner:
    """Offline depth -> color registration using the factory calibration.

    Mirrors the Orbbec `AlignFilter(align_to_stream=COLOR_STREAM)` the live
    pipeline uses: depth is unprojected with the depth intrinsics, moved into
    the color frame via the inverse of `color_to_depth_transform`, then
    projected with the color intrinsics + rational distortion. The camera is
    rigid, so the normalized rays are precomputed once.
    """

    def __init__(self, cal: dict, use_distortion: bool = True):
        self.fx = cal["fx"]
        self.fy = cal["fy"]
        self.cx = cal["cx"]
        self.cy = cal["cy"]
        self.w = cal["width"]
        self.h = cal["height"]
        self.cw = cal["color_width"]
        self.ch = cal["color_height"]
        self.cfx = cal["color_fx"]
        self.cfy = cal["color_fy"]
        self.ccx = cal["color_cx"]
        self.ccy = cal["color_cy"]
        self.dist = _poly_params(cal) if use_distortion else np.zeros(8)

        R = np.array(cal["color_to_depth_transform"]["rotation"],
                     dtype=np.float64).reshape(3, 3)
        t = np.array(cal["color_to_depth_transform"]["translation"],
                     dtype=np.float64) / 1000.0
        # `color_to_depth` maps color-frame points into the depth frame; we
        # need the inverse (depth -> color).
        self.R_dc = R.T
        self.t_dc = -R.T @ t

        ys, xs = np.mgrid[0:self.h, 0:self.w]
        self.xn = ((xs - self.cx) / self.fx).astype(np.float32)
        self.yn = ((ys - self.cy) / self.fy).astype(np.float32)

    def align(self, depth_mm: np.ndarray) -> np.ndarray:
        """uint16 depth (mm, depth frame) -> float32 meters (color frame)."""
        z = depth_mm.astype(np.float32) / 1000.0
        valid = z > 0.05
        if not valid.any():
            return np.zeros((self.ch, self.cw), np.float32)
        zv = z[valid]
        xv = self.xn[valid] * zv
        yv = self.yn[valid] * zv
        R, t = self.R_dc, self.t_dc
        xc = R[0, 0] * xv + R[0, 1] * yv + R[0, 2] * zv + t[0]
        yc = R[1, 0] * xv + R[1, 1] * yv + R[1, 2] * zv + t[1]
        zc = R[2, 0] * xv + R[2, 1] * yv + R[2, 2] * zv + t[2]
        front = zc > 0.05
        xc, yc, zc = xc[front], yc[front], zc[front]
        if zc.size == 0:
            return np.zeros((self.ch, self.cw), np.float32)
        xn = xc / zc
        yn = yc / zc
        k1, k2, p1, p2, k3, k4, k5, k6 = self.dist
        r2 = xn * xn + yn * yn
        num = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        den = 1.0 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        radial = num / den
        tx = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        ty = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        u = self.cfx * (xn * radial + tx) + self.ccx
        v = self.cfy * (yn * radial + ty) + self.ccy
        ok = (u >= 0) & (u < self.cw - 1) & (v >= 0) & (v < self.ch - 1)
        u, v, zc = u[ok], v[ok], zc[ok]
        out = np.full((self.ch, self.cw), np.inf, np.float32)
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        np.minimum.at(out, (vi, ui), zc)
        out[~np.isfinite(out)] = 0.0
        return out


def alignment_overlay(color_rgb: np.ndarray, aligned_m: np.ndarray) -> np.ndarray:
    """Side-by-side RGB | colorized aligned depth | blend, for eyeballing."""
    d = np.clip(aligned_m / 4.0, 0.0, 1.0)
    vis = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (aligned_m > 0)[..., None]
    blend = (color_rgb * mask + vis[:, :, ::-1] * (~mask) * 1.0).astype(np.uint8)
    return np.concatenate([color_rgb, vis[:, :, ::-1], blend], axis=1)


# ---------------------------------------------------------------------------
# dense map (accumulated point cloud)
# ---------------------------------------------------------------------------

def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product of two xyzw quaternions."""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_conj(q) -> np.ndarray:
    return np.array([-float(q[0]), -float(q[1]), -float(q[2]),
                     float(q[3])], dtype=np.float64)


class PoseOut:
    """Simple pose holder matching cuVSLAM's `.translation`/`.rotation`."""
    __slots__ = ("translation", "rotation")

    def __init__(self, translation: np.ndarray, rotation: np.ndarray):
        self.translation = translation
        self.rotation = rotation


class DenseMap:
    """Global voxel-fused colored point cloud from aligned depth + pose.

    Every Nth frame the aligned depth is deprojected with the colour
    intrinsics, transformed into the world frame by `world_from_rig`, and
    fused into a persistent voxel hash: each cell keeps the running mean
    position and colour of every observation that landed in it. Emitting the
    *fused* set (rather than every frame's raw points) is what turns a
    drifting raw accumulation into a world map.
    """

    def __init__(self, cal, voxel=0.05, stride=8, max_range=5.0,
                 max_points=3_000_000):
        self.voxel = voxel
        self.stride = stride
        self.max_range = max_range
        self.max_points = max_points
        self.total = 0
        # packed voxel key -> [sx, sy, sz, sr, sg, sb, n]
        self._acc: dict[int, np.ndarray] = {}
        self.cw, self.ch = cal["color_width"], cal["color_height"]
        ys, xs = np.mgrid[0:self.ch, 0:self.cw]
        self.xn = ((xs - cal["color_cx"]) / cal["color_fx"]).astype(np.float32)
        self.yn = ((ys - cal["color_cy"]) / cal["color_fy"]).astype(np.float32)
        self.sl = (slice(None, None, stride), slice(None, None, stride))

    def add(self, depth_m: np.ndarray, color_rgb: np.ndarray, pose):
        z = depth_m[self.sl].astype(np.float32)
        xn = self.xn[self.sl]
        yn = self.yn[self.sl]
        rgb = color_rgb[self.sl]
        valid = (z > 0.4) & (z < self.max_range)
        if not valid.any():
            return
        xn, yn, z, rgb = xn[valid], yn[valid], z[valid], rgb[valid]
        pc = np.stack([xn * z, yn * z, z], axis=1)
        R = quat_to_matrix(pose.rotation)
        t = np.asarray(pose.translation, np.float32)
        pw = pc @ R.T + t
        keys = np.floor(pw / self.voxel).astype(np.int64)
        packed = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) \
            ^ (keys[:, 2] * 83492791)
        uniq, inv = np.unique(packed, return_inverse=True)
        ones = np.ones((len(pw), 1))
        obs = np.hstack([pw.astype(np.float64), rgb.astype(np.float64), ones])
        sums = np.zeros((len(uniq), 7), np.float64)
        np.add.at(sums, inv, obs)
        for k, a in zip(uniq.tolist(), sums):
            cur = self._acc.get(k)
            if cur is None:
                if self.total >= self.max_points:
                    continue
                self._acc[k] = a
                self.total += 1
            else:
                cur += a

    def fused(self):
        """Merged point cloud: per-voxel mean position and colour."""
        if not self._acc:
            return None, None
        a = np.array(list(self._acc.values()), np.float64)
        n = a[:, 6:7]
        pts = (a[:, 0:3] / n).astype(np.float32)
        col = np.clip(a[:, 3:6] / n, 0, 255).astype(np.uint8)
        return pts, col


def write_ply(path: Path, pts: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian coloured PLY (x y z + r g b)."""
    n = len(pts)
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)


# ---------------------------------------------------------------------------
# MCAP decoding
# ---------------------------------------------------------------------------

class H265Decoder:
    """Streaming Annex-B H.265 decoder (PyAV), one AU at a time."""

    def __init__(self):
        self._codec = av.CodecContext.create("hevc", "r")

    def push(self, au: bytes):
        try:
            for frame in self._codec.decode(av.packet.Packet(au)):
                yield frame.to_ndarray(format="rgb24")
        except Exception:
            return


def decode_depth(blob: bytes) -> np.ndarray | None:
    arr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
    return arr


def prescan_imu_clock(mcap_path: Path, role: str) -> tuple[float, int, float]:
    """Estimate the host<->device clock offset from IMU samples.

    Image messages only carry host `log_time`; IMU carries the true
    `device_stamp_us` that is shared with the depth/colour streams. The IMU
    host stamp is assigned at drain time, so fit a single constant offset
    `b = median(stamp_ns - device_stamp_us*1000)` and use it to place the IMU
    on the host timeline alongside the images.
    """
    offs = []
    n = 0
    first_host = last_host = 0
    with open(mcap_path, "rb") as f:
        for sch, ch, msg in make_reader(f).iter_messages(
                topics=[f"/imu_accel_{role}"]):
            m = navd_pb.ImuAccel()
            m.ParseFromString(msg.data)
            offs.append(m.stamp_ns - m.device_stamp_us * 1000)
            if not first_host:
                first_host = m.stamp_ns
            last_host = m.stamp_ns
            n += 1
    if not offs:
        return 0.0, 0, 0.0
    b = float(np.median(np.asarray(offs, np.int64)))
    return b, n, (last_host - first_host) / 1e9


def load_imu(mcap_path: Path, role: str, clock_offset: float):
    """Merge accel+gyro by device_stamp_us; return host_ns-sorted samples."""
    accel: dict[int, tuple] = {}
    gyro: dict[int, tuple] = {}
    with open(mcap_path, "rb") as f:
        for sch, ch, msg in make_reader(f).iter_messages():
            if ch.topic == f"/imu_accel_{role}":
                m = navd_pb.ImuAccel()
                m.ParseFromString(msg.data)
                accel[int(m.device_stamp_us)] = (m.ax, m.ay, m.az)
            elif ch.topic == f"/imu_gyro_{role}":
                m = navd_pb.ImuGyro()
                m.ParseFromString(msg.data)
                gyro[int(m.device_stamp_us)] = (m.gx, m.gy, m.gz)
    rows = []
    for dev_us in accel.keys() & gyro.keys():
        host_ns = int(dev_us * 1000 + clock_offset)
        rows.append((host_ns, accel[dev_us], gyro[dev_us]))
    rows.sort(key=lambda r: r[0])
    return rows


# ---------------------------------------------------------------------------
# trajectory math
# ---------------------------------------------------------------------------

def quat_angle(q: np.ndarray) -> float:
    """Rotation angle (rad) from a xyzw quaternion."""
    w = abs(float(q[3]))
    return 2.0 * math.acos(min(1.0, max(-1.0, w)))


def quat_rel_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def path_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def umeyama_2d(src: np.ndarray, dst: np.ndarray):
    """Similarity fit src->dst (2D). Returns (R, t, scale)."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    s0 = src - mu_s
    d0 = dst - mu_d
    cov = d0.T @ s0 / len(src)
    u, sv, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    R = u @ np.diag(d) @ vt
    var_s = (s0 ** 2).sum() / len(src)
    scale = float((sv * d).sum() / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * (R @ mu_s)
    return R, t, scale


# ---------------------------------------------------------------------------
# cuVSLAM tracker
# ---------------------------------------------------------------------------

def build_tracker(cal, args):
    cam = vslam.Camera()
    if args.distortion == "poly":
        cam.distortion = vslam.Distortion(
            vslam.Distortion.Model.Polynomial, _poly_params(cal).tolist())
    else:
        cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
    cam.focal = (cal["color_fx"], cal["color_fy"])
    cam.principal = (cal["color_cx"], cal["color_cy"])
    cam.size = (cal["color_width"], cal["color_height"])

    rig = vslam.Rig()
    rig.cameras = [cam]
    rig.imus = []

    mode = args.mode
    if mode in ("rgbd", "rgbd-imu"):
        rgbd = vslam.Tracker.OdometryRGBDSettings()
        # cuVSLAM requires depth as 2D uint16 (mm); the aligner returns
        # float meters, converted to mm at the call site.
        rgbd.depth_scale_factor = 1000.0
        rgbd.depth_camera_id = 0
        rgbd.enable_depth_stereo_tracking = False

    if mode == "rgbd":
        odom = vslam.Tracker.OdometryConfig(
            odometry_mode=vslam.Tracker.OdometryMode.RGBD,
            rgbd_settings=rgbd, use_gpu=args.gpu,
            async_sba=True, enable_final_landmarks_export=False,
            enable_landmarks_export=(args.vis != "none"),
            enable_observations_export=(args.vis != "none"))
    else:
        if vslam.ImuCalibration is None:
            raise SystemExit("IMU mode requires cuVSLAM ImuCalibration")
        imu = vslam.ImuCalibration()
        imu.rig_from_imu = vslam.Pose(rotation=(0.0, 0.0, 0.0, 1.0),
                                      translation=(0.0, 0.0, 0.0))
        imu.frequency = IMU_RATE_HZ
        imu.gyroscope_noise_density = IMU_GYRO_NOISE_DENSITY
        imu.gyroscope_random_walk = IMU_GYRO_RANDOM_WALK
        imu.accelerometer_noise_density = IMU_ACCEL_NOISE_DENSITY
        imu.accelerometer_random_walk = IMU_ACCEL_RANDOM_WALK
        rig.imus = [imu]
        ms = vslam.Tracker.OdometryMultisensorSettings()
        ms.depth_scale_factor = 1000.0
        ms.depth_camera_ids = [0] if mode == "rgbd-imu" else []
        ms.enable_depth_stereo_tracking = False
        odom = vslam.Tracker.OdometryConfig(
            odometry_mode=vslam.Tracker.OdometryMode.Multisensor,
            multisensor_settings=ms, use_gpu=args.gpu,
            async_sba=True, enable_final_landmarks_export=False,
            enable_landmarks_export=(args.vis != "none"),
            enable_observations_export=(args.vis != "none"))

    slam = None
    if args.slam:
        slam = vslam.Tracker.SlamConfig()
        slam.use_gpu = args.gpu
        slam.sync_mode = False
        slam.map_cache_path = str(args.map_dir) if args.map_dir else ""
        slam.planar_constraints = False

    return vslam.Tracker(rig, odom, slam)


# ---------------------------------------------------------------------------
# Rerun visualization
# ---------------------------------------------------------------------------

class RerunSink:
    """Logs the replay to Rerun: 3D trajectory, map landmarks, camera feed."""

    def __init__(self, mode: str, out_path: Path, every: int):
        self.enabled = (rr is not None) and mode != "none"
        self.every = max(1, every)
        self.prev = None
        if not self.enabled:
            return
        rr.init("bebop_slam_replay", spawn=(mode == "spawn"))
        if mode == "rrd":
            rr.save(str(out_path))
        # cuVSLAM poses live in the OpenCV convention (x right, y down, z fwd).
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        try:
            import rerun.blueprint as rrb
            rr.send_blueprint(rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(column_shares=[1, 2], contents=[
                    rrb.Vertical(contents=[
                        rrb.Spatial2DView(origin="world/camera/image"),
                        rrb.Spatial2DView(origin="world/camera/depth"),
                    ]),
                    rrb.Spatial3DView(origin="world"),
                ])), make_active=True)
        except Exception:
            pass

    @staticmethod
    def _colors(ids: np.ndarray) -> np.ndarray:
        ids = ids.astype(np.int64)
        return np.stack([(ids * 37) % 256, (ids * 91) % 256,
                         (ids * 53) % 256], axis=1).astype(np.uint8)

    def log(self, i, color, aligned_mm, pose, landmarks, observations):
        if not self.enabled:
            return
        rr.set_time("frame", sequence=int(i))
        t = np.asarray(pose.translation, np.float32)
        q = np.asarray(pose.rotation, np.float32)
        rr.log("world/rig", rr.Transform3D(
            translation=t, quaternion=rr.Quaternion(xyzw=q)))
        if self.prev is not None:
            rr.log("world/trajectory", rr.LineStrips3D(
                [np.stack([self.prev, t])], colors=[[255, 80, 80]], radii=0.008))
        self.prev = t

        if i % self.every == 0:
            rr.log("world/camera/image", rr.Image(color).compress())
            if aligned_mm is not None:
                small = cv2.resize(aligned_mm, None, fx=0.5, fy=0.5,
                                   interpolation=cv2.INTER_NEAREST)
                rr.log("world/camera/depth",
                       rr.DepthImage(small, meter=1000.0))
        if landmarks:
            pts = np.array([l.coords for l in landmarks], np.float32)
            ids = np.array([l.id for l in landmarks])
            rr.log("world/landmarks", rr.Points3D(
                positions=pts, colors=self._colors(ids), radii=0.006))
        if observations:
            uv = np.array([[o.u, o.v] for o in observations], np.float32)
            oid = np.array([o.id for o in observations])
            rr.log("world/camera/observations", rr.Points2D(
                positions=uv, colors=self._colors(oid), radii=3.0))

    def log_map(self, pts, colors, voxel):
        if not self.enabled or pts is None or len(pts) == 0:
            return
        # Static log: re-logging replaces, so the viewer shows the current
        # *fused* map rather than every frame's raw points accumulated.
        rr.log("world/map", rr.Points3D(
            positions=pts, colors=colors,
            radii=max(0.005, voxel * 0.4)), static=True)

    def close(self):
        if self.enabled:
            try:
                rr.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# main replay
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mcap", type=Path)
    ap.add_argument("--camera", choices=["far", "near"], default="far")
    ap.add_argument("--mode", choices=["rgbd", "rgbd-imu", "rgb-imu"],
                    default="rgbd")
    ap.add_argument("--distortion", choices=["poly", "pinhole"], default="poly")
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="multiply the aligned depth (metres) by this before "
                         "tracking. The Orbbec stream here is ~2 mm units, so "
                         "RGB-D *odometry* needs 2.0 to restore metric scale. "
                         "For VIO (rgbd-imu) leave 1.0 (the IMU sets scale).")
    ap.add_argument("--map-depth-scale", type=float, default=None,
                    help="depth multiplier used only for the fused map "
                         "(default: same as --depth-scale). Set separately so "
                         "the map can be metric while VIO tracks raw depth.")
    ap.add_argument("--slam", action="store_true",
                    help="enable cuVSLAM SLAM (loop closure + map db)")
    ap.add_argument("--map-dir", type=Path, default=None,
                    help="LMDB map path for --slam (default: <out>/map)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config-dir", type=Path,
                    default=REPO_ROOT / "config")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float, default=0.0)
    ap.add_argument("--max-step-m", type=float, default=0.25,
                    help="cuVSLAM RGB-D occasionally relocalizes with an "
                         "impossible single-frame teleport (metres in ~33 ms). "
                         "When a tracked step exceeds this, re-anchor the "
                         "trajectory to the last good pose instead of letting "
                         "the jump corrupt it. 0 disables the guard.")
    ap.add_argument("--no-gpu", dest="gpu", action="store_false")
    ap.add_argument("--dump-align", type=int, default=0,
                    help="write N depth->color alignment overlays")
    ap.add_argument("--vis", choices=["none", "rrd", "spawn"], default="rrd",
                    help="Rerun visualization: write <out>/slam.rrd (default), "
                         "open a live viewer (spawn), or none")
    ap.add_argument("--rrd", type=Path, default=None,
                    help="path for the .rrd output (default <out>/slam.rrd)")
    ap.add_argument("--vis-every", type=int, default=3,
                    help="log camera image/depth every Nth frame")
    ap.add_argument("--no-map", dest="map", action="store_false",
                    help="disable the accumulated dense point-cloud map")
    ap.add_argument("--map-every", type=int, default=10,
                    help="add a dense cloud slice every Nth frame")
    ap.add_argument("--map-stride", type=int, default=8,
                    help="depth pixel stride for the dense map")
    ap.add_argument("--map-voxel", type=float, default=0.05,
                    help="voxel size (m) for map dedup")
    ap.add_argument("--map-max-points", type=int, default=3_000_000)
    ap.add_argument("--map-emit-every", type=int, default=40,
                    help="re-log the fused map every Nth frame (0 = only at end)")
    ap.add_argument("--map-ply", type=Path, default=None,
                    help="also write the fused map as a binary PLY point cloud")
    ap.set_defaults(gpu=True, map=True)
    args = ap.parse_args()

    if vslam is None:
        print("PyCuVSLAM not installed in this interpreter. Use the cuVSLAM "
              "python 3.12 env (see module docstring).", file=sys.stderr)
        return 2
    if av is None:
        print("PyAV (av) is required to decode H.265 color.", file=sys.stderr)
        return 2
    if not args.mcap.exists():
        print(f"no such MCAP: {args.mcap}", file=sys.stderr)
        return 2

    out_dir = args.out or args.mcap.with_suffix("").with_suffix(".slam")
    out_dir = out_dir.with_name(args.mcap.stem + ".slam") \
        if args.out is None else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir = args.map_dir or (out_dir / "map")

    roles = load_rig_roles(args.config_dir)
    serial = roles[args.camera]
    cal = load_intrinsics(args.config_dir, serial)
    print(f"[slam] camera={args.camera} serial={serial} "
          f"color={cal['color_width']}x{cal['color_height']} "
          f"depth={cal['width']}x{cal['height']} mode={args.mode} "
          f"gpu={args.gpu} slam={args.slam}")

    aligner = DepthAligner(cal, use_distortion=(args.distortion == "poly"))
    clock_off, n_imu, dur_imu = prescan_imu_clock(args.mcap, args.camera)
    print(f"[slam] IMU: {n_imu} samples, {dur_imu:.1f}s, "
          f"host-device offset {clock_off/1e6:.1f} ms")
    imu_rows = load_imu(args.mcap, args.camera, clock_off)
    up_dev = None
    if imu_rows:
        acc = np.array([r[1] for r in imu_rows], float)
        up_dev = np.median(acc, axis=0)
        n = np.linalg.norm(up_dev)
        if n > 0:
            up_dev = up_dev / n
            print(f"[slam] device-frame up (med accel): "
                  f"[{up_dev[0]:.3f} {up_dev[1]:.3f} {up_dev[2]:.3f}]")
    use_imu = args.mode in ("rgbd-imu", "rgb-imu")
    print(f"[slam] IMU rows paired: {len(imu_rows)} (use={use_imu})")

    tracker = build_tracker(cal, args)
    rrd_path = args.rrd or (out_dir / "slam.rrd")
    sink = RerunSink(args.vis, rrd_path, args.vis_every)
    if args.vis != "none":
        if rr is None:
            print("[slam] rerun-sdk not installed; --vis disabled")
        else:
            loc = str(rrd_path) if args.vis == "rrd" else "live viewer"
            print(f"[slam] rerun visualization: {args.vis} ({loc})")

    mapper = None
    if args.map and args.mode in ("rgbd", "rgbd-imu") \
            and (args.vis != "none" or args.map_ply is not None):
        mapper = DenseMap(cal, voxel=args.map_voxel, stride=args.map_stride,
                          max_points=args.map_max_points)
        print(f"[slam] dense map: every {args.map_every} frames, stride "
              f"{args.map_stride}, voxel {args.map_voxel} m, "
              f"cap {args.map_max_points} points")

    topic_depth = f"/depth_{args.camera}"
    topic_color = f"/color_{args.camera}"
    decoders: dict[str, H265Decoder] = {}
    latest_color: dict[str, tuple[int, np.ndarray]] = {}
    pending_depth: dict[str, tuple[int, np.ndarray]] = {}
    processed: set[int] = set()

    poses = []          # (host_ns, xyz, quat)
    wheel = []          # (host_ns, x, y, theta)
    imu_i = 0
    imu_reg = 0
    frames = 0
    tracked = 0
    t0 = None
    t_start = time.time()
    dumped = 0
    jumps = 0
    # Left-multiplied SE(3) correction re-anchoring the tracker to the last
    # good pose after an implausible teleport (see --max-step-m).
    offset_q = np.array([0.0, 0.0, 0.0, 1.0])
    offset_t = np.zeros(3)
    offset_R = np.eye(3)

    def process_pair(role: str, ts_ns: int):
        nonlocal imu_i, imu_reg, frames, tracked, dumped, t0, jumps
        nonlocal offset_q, offset_t, offset_R
        if ts_ns in processed:
            return
        # Require color and depth from the *same* tick. Never reuse a stale
        # frame: the far camera's colour stream drops out for seconds at a
        # time, and pairing a stale colour with fresh depth destroys tracking.
        c = latest_color.get(role)
        d = pending_depth.get(role)
        if c is None or d is None or c[0] != ts_ns or d[0] != ts_ns:
            return
        processed.add(ts_ns)
        color, depth = c[1], d[1]
        del latest_color[role]
        del pending_depth[role]
        if t0 is None:
            t0 = ts_ns
        rel = (ts_ns - t0) / 1e9
        if rel < args.start_s:
            return
        if args.end_s and rel > args.end_s:
            return
        if args.max_frames and frames >= args.max_frames:
            return

        # Register every IMU sample up to this frame (same host timeline).
        if use_imu:
            while imu_i < len(imu_rows) and imu_rows[imu_i][0] <= ts_ns:
                host, acc, gyr = imu_rows[imu_i]
                imu_i += 1
                m = vslam.ImuMeasurement()
                m.timestamp_ns = int(host)
                m.linear_accelerations = (float(acc[0]), float(acc[1]), float(acc[2]))
                m.angular_velocities = (float(gyr[0]), float(gyr[1]), float(gyr[2]))
                try:
                    tracker.register_imu_measurement(0, m)
                    imu_reg += 1
                except Exception:
                    pass

        aligned = aligner.align(depth) if args.mode in ("rgbd", "rgbd-imu") else None
        if aligned is not None and args.depth_scale != 1.0:
            aligned = aligned * args.depth_scale
        if args.dump_align and dumped < args.dump_align:
            cv2.imwrite(str(out_dir / f"align_{dumped:04d}.png"),
                        alignment_overlay(color, aligned)[:, :, ::-1])
            dumped += 1

        images = [color]
        # cuVSLAM RGB-D wants uint16 (mm); aligner emits float meters.
        depths = None
        aligned_mm = None
        if aligned is not None:
            aligned_mm = np.clip(np.rint(aligned * 1000.0), 0,
                                 65535).astype(np.uint16)
            depths = [aligned_mm]
        frames += 1
        try:
            pe, slam_pose = tracker.track(int(ts_ns), images=images,
                                          depths=depths)
        except Exception as e:
            msg = str(e)
            if "unsupported toolchain" in msg or "cunls" in msg.lower():
                raise SystemExit(
                    "[slam] cuVSLAM Multisensor/IMU mode uses cuNLS, which "
                    "requires CUDA >= 13.2. Update the NVIDIA driver or run "
                    "this replay on the Jetson Thor (JetPack 7.2 / CUDA 13.2).")
            if frames <= 5:
                print(f"[slam] track error: {e}")
            return
        pose = None
        if args.slam and slam_pose is not None:
            pose = slam_pose
        elif pe is not None and pe.world_from_rig is not None:
            pose = pe.world_from_rig.pose
        if pose is None:
            return
        raw_t = np.asarray(pose.translation, np.float64)
        raw_q = np.asarray(pose.rotation, np.float64)
        if poses and args.max_step_m > 0:
            step_m = float(np.linalg.norm(offset_R @ raw_t + offset_t
                                          - poses[-1][1]))
            if step_m > args.max_step_m:
                # cuVSLAM RGB-D relocalization teleport (physically impossible
                # for this chassis): re-anchor so the pose continues smoothly
                # from the last good one instead of writing a fake jump.
                last_t, last_q = poses[-1][1], poses[-1][2]
                offset_q = quat_mul(last_q, quat_conj(raw_q))
                offset_R = quat_to_matrix(offset_q).astype(np.float64)
                offset_t = last_t - offset_R @ raw_t
                jumps += 1
        adj_t = offset_R @ raw_t + offset_t
        adj_q = quat_mul(offset_q, raw_q)
        adj_q = adj_q / np.linalg.norm(adj_q)
        pose = PoseOut(adj_t, adj_q)
        poses.append((int(ts_ns), adj_t, adj_q))
        tracked += 1
        if args.vis != "none" and sink.enabled:
            try:
                lms = tracker.get_last_landmarks()
            except Exception:
                lms = None
            try:
                obs = tracker.get_last_observations(0)
            except Exception:
                obs = None
            sink.log(tracked, color, aligned_mm, pose, lms, obs)
        if mapper is not None and aligned is not None \
                and frames % max(1, args.map_every) == 0:
            mapper.add(aligned * (args.map_depth_scale or 1.0), color, pose)
            if sink.enabled and args.map_emit_every \
                    and tracked % args.map_emit_every == 0:
                sink.log_map(*mapper.fused(), mapper.voxel)
        if tracked % 200 == 0:
            print(f"[slam] {tracked} tracked / {frames} frames, "
                  f"{rel:.1f}s, {(time.time()-t_start):.1f}s wall")

    # Single ordered pass over the MCAP.
    with open(args.mcap, "rb") as f:
        for sch, ch, msg in make_reader(f).iter_messages():
            if ch.topic == topic_depth:
                m = ci_pb.CompressedImage()
                m.ParseFromString(msg.data)
                d = decode_depth(m.data)
                if d is not None:
                    pending_depth[args.camera] = (msg.log_time, d)
                    if args.camera in latest_color:
                        process_pair(args.camera, msg.log_time)
            elif ch.topic == topic_color:
                m = cv_pb.CompressedVideo()
                m.ParseFromString(msg.data)
                dec = decoders.setdefault(args.camera, H265Decoder())
                frame = None
                for rgb in dec.push(bytes(m.data)):
                    frame = rgb
                if frame is not None:
                    latest_color[args.camera] = (msg.log_time, frame)
                    if args.camera in pending_depth:
                        process_pair(args.camera, msg.log_time)
            elif ch.topic == "/odom":
                m = navd_pb.Odom()
                m.ParseFromString(msg.data)
                wheel.append((msg.log_time, m.x, m.y, m.theta))

    wall = time.time() - t_start
    print(f"[slam] done: {tracked}/{frames} frames tracked, "
          f"{imu_reg} IMU samples registered, {wall:.1f}s wall")

    if mapper is not None:
        pts, cols = mapper.fused()
        if sink.enabled:
            sink.log_map(pts, cols, mapper.voxel)
        if args.map_ply is not None and pts is not None:
            write_ply(args.map_ply, pts, cols)
            print(f"[slam] wrote {args.map_ply} "
                  f"({len(pts)} fused voxels)")

    # ---- outputs ----------------------------------------------------------
    cuv = np.array([[t, *xyz, *q] for t, xyz, q in poses]) if poses else \
        np.zeros((0, 8))
    np.savetxt(out_dir / "trajectory_cuvslam.csv", cuv, delimiter=",",
               header="t_ns,x,y,z,qx,qy,qz,qw", comments="")
    np.savetxt(out_dir / "trajectory_wheel.csv",
               np.array(wheel) if wheel else np.zeros((0, 4)),
               delimiter=",", header="t_ns,x,y,theta", comments="")

    metrics = summarize(poses, wheel, up=up_dev)
    metrics.update({
        "mcap": str(args.mcap),
        "camera": args.camera,
        "serial": serial,
        "mode": args.mode,
        "slam": args.slam,
        "gpu": args.gpu,
        "frames": frames,
        "tracked": tracked,
        "imu_registered": imu_reg,
        "jumps_corrected": jumps,
        "max_step_m": args.max_step_m,
        "wall_s": round(wall, 1),
        "distortion": args.distortion,
        "map_points": (mapper.total if mapper is not None else 0),
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_trajectories(out_dir / "trajectory_plot.png", poses, wheel, up=up_dev)
    sink.close()
    print(f"[slam] wrote {out_dir}/")
    if args.vis == "rrd" and rr is not None:
        print(f"[slam] open the 3D SLAM view with:  rerun {rrd_path}")
    print(json.dumps(metrics, indent=2))
    return 0


def summarize(poses, wheel, up=None) -> dict:
    m = {}
    w = np.array(wheel, dtype=float) if wheel else np.zeros((0, 4))
    if w.size:
        w = w[np.argsort(w[:, 0])]
    if poses:
        t = np.array([p[0] for p in poses], np.int64)
        xyz = np.array([p[1] for p in poses])
        quat = np.array([p[2] for p in poses])
        m["cuvslam_duration_s"] = round((t[-1] - t[0]) / 1e9, 2)
        m["cuvslam_path_m"] = round(path_length(xyz), 3)
        rot = sum(quat_rel_angle(quat[i], quat[i + 1])
                  for i in range(len(quat) - 1))
        m["cuvslam_total_rotation_rad"] = round(rot, 3)
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if step.size:
            m["cuvslam_max_step_m"] = round(float(step.max()), 4)
            m["cuvslam_steps_gt_0_15m"] = int((step > 0.15).sum())
        m["cuvslam_final_xyz"] = [round(v, 3) for v in xyz[-1].tolist()]
        if up is not None and np.linalg.norm(up) > 0:
            u = np.asarray(up, float)
            u = u / np.linalg.norm(u)
            vert = xyz @ u
            ph = xyz - np.outer(vert, u)
            m["cuvslam_horizontal_path_m"] = round(path_length(ph), 3)
            # decimate to 2 Hz to suppress per-frame jitter inflation
            keep = [0]
            for i in range(1, len(t)):
                if (t[i] - t[keep[-1]]) >= 5e8:
                    keep.append(i)
            if len(keep) > 1:
                m["cuvslam_horizontal_path_2hz_m"] = round(
                    path_length(ph[keep]), 3)
    if len(w):
        m["wheel_path_m_full"] = round(path_length(w[:, 1:3]), 3)
        m["wheel_total_rotation_rad_full"] = round(
            float(np.abs(np.diff(w[:, 3])).sum()), 3)
    if poses and len(w):
        t0, t1 = int(t[0]), int(t[-1])
        wm = w[(w[:, 0] >= t0) & (w[:, 0] <= t1)]
        if len(wm) > 1:
            m["wheel_path_m_window"] = round(path_length(wm[:, 1:3]), 3)
            m["wheel_total_rotation_rad_window"] = round(
                float(np.abs(np.diff(wm[:, 3])).sum()), 3)
        src = w if len(wm) < 2 else wm
        idx = np.searchsorted(src[:, 0], t).clip(1, len(src) - 1)
        pick = np.where(np.abs(src[idx, 0] - t) < np.abs(src[idx - 1, 0] - t),
                        idx, idx - 1)
        wxy = src[pick][:, 1:3]
        # Horizontal cuVSLAM points in the gravity-normal plane (camera IMU
        # defines vertical), so the 2D fit to wheel odom is frame-agnostic.
        if up is not None and np.linalg.norm(up) > 0:
            u = np.asarray(up, float)
            u = u / np.linalg.norm(u)
            a = np.cross(u, [1.0, 0.0, 0.0])
            if np.linalg.norm(a) < 1e-3:
                a = np.cross(u, [0.0, 1.0, 0.0])
            a = a / np.linalg.norm(a)
            b = np.cross(u, a)
            vert = xyz @ u
            ph = xyz - np.outer(vert, u)
            cv = np.stack([ph @ a, ph @ b], 1)
        else:
            cv = np.array([[p[1][2], -p[1][0]] for p in poses])
        n = min(len(cv), len(wxy))
        if n >= 2 and np.ptp(cv[:n]) > 1e-6:
            R, tr, scale = umeyama_2d(cv[:n], wxy[:n])
            fit = (scale * (R @ cv[:n].T).T + tr)
            rmse = float(np.sqrt(((fit - wxy[:n]) ** 2).sum(1).mean()))
            m["odom_fit_scale"] = round(scale, 3)
            m["odom_fit_rmse_m"] = round(rmse, 3)
            m["odom_fit_note"] = ("horizontal-plane similarity fit cuVSLAM->"
                                  "wheel; scale != 1 means wheel distance and "
                                  "vision distance disagree (slip/scale)")
    return m


def plot_trajectories(path: Path, poses, wheel, up=None) -> None:
    if not poses and not wheel:
        return
    scale = 120.0
    size = 900
    img = np.full((size, size, 3), 255, np.uint8)

    def to_px(pts):
        pts = np.asarray(pts, float)
        if len(pts) == 0:
            return pts
        c = pts.mean(0)
        return (pts - c) * scale + size / 2

    if poses:
        xyz = np.array([p[1] for p in poses])
        if up is not None and np.linalg.norm(up) > 0:
            u = np.asarray(up, float)
            u = u / np.linalg.norm(u)
            a = np.cross(u, [1.0, 0.0, 0.0])
            if np.linalg.norm(a) < 1e-3:
                a = np.cross(u, [0.0, 1.0, 0.0])
            a = a / np.linalg.norm(a)
            b = np.cross(u, a)
            cv = np.stack([xyz @ a, xyz @ b], 1)
        else:
            cv = np.array([[p[1][2], -p[1][0]] for p in poses])
        p = to_px(cv)
        cv2.polylines(img, [p.astype(np.int32)], False, (0, 0, 255), 2)
        cv2.circle(img, tuple(p[0].astype(int)), 5, (0, 200, 0), -1)
    if wheel:
        wh = np.array([[r[1], r[2]] for r in wheel])
        p = to_px(wh)
        cv2.polylines(img, [p.astype(np.int32)], False, (200, 0, 0), 2)
        cv2.circle(img, tuple(p[0].astype(int)), 5, (0, 200, 0), -1)
    cv2.putText(img, "cuVSLAM (red) vs wheel odom (blue), gravity-leveled",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    raise SystemExit(main())
