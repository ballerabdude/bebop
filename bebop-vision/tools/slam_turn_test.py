#!/usr/bin/env python3
"""Manual visual-yaw turn diagnostic for the live SLAM POC.

Reads the live pose from the recorder's status endpoint (`:9096`), turns the
robot with zero linear velocity and a small capped angular speed, and stops on
stale/lost/segment change.  Wheel odometry is never read or used: the angle
comes only from the camera pose, converted to a body yaw with the known mount
orientation.

This is a bench diagnostic, not an autonomous behavior.  It refuses to move
unless `--execute` is passed explicitly:

    # read-only: print the current pose / yaw and exit
    python tools/slam_turn_test.py --camera far

    # authorized motion test: 45 deg left, back to start, 45 deg right
    python tools/slam_turn_test.py --camera far --execute

The default leg list is ``+45,0,-45`` (cumulative body-yaw targets in degrees,
relative to the first valid pose).  Each leg has a hard timeout; on timeout,
tracking loss, stale pose, a jump, wrong-direction motion, no progress, or a
segment change the script commands zero twist and aborts.  The finally block
always zeroes the twist, disarms the wheels, and restores the mode that was
active before the run, waiting for telemetry confirmation before disconnecting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]  # bebop-vision/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bebop_vision.robot import RobotClient  # noqa: E402
from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb  # noqa: E402

# Camera optical (x right, y down, z forward) -> body (x forward, y left,
# z up), for a zero-pitch/zero-yaw mount.
R_BODY_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def quat_to_matrix(q) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n <= 0:
        raise ValueError("bad quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def yaw_from_matrix(R: np.ndarray) -> float:
    """Body yaw (rad, + left) from a body-frame rotation matrix."""
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def parse_targets(text: str) -> list[float]:
    """Cumulative target angles in degrees (e.g. '45,0,-45')."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not math.isfinite(value) or abs(value) > 180.0:
            raise ValueError(f"target {value!r} outside [-180, 180]")
        out.append(value)
    if not out:
        raise ValueError("no targets")
    return out


def load_mount(config_dir: Path, role: str):
    """Return (pitch_deg, yaw_deg) for a camera role, or None."""
    path = Path(config_dir) / "orbbec_rig.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    cams = data["robots"]["default"]["cameras"]
    for cam in cams.values():
        if cam.get("role") == role:
            return float(cam.get("pitch_deg", 0.0)), float(cam.get("yaw_deg", 0.0))
    return None


def body_from_optical_mount(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rigid rotation from camera optical to body, including mount angles."""
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cz, sz = math.cos(y), math.sin(y)
    cy, sy = math.cos(-p), math.sin(-p)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return Rz @ Ry @ R_BODY_FROM_OPTICAL


class StatusClient:
    def __init__(self, base_url: str, timeout: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/status",
                                    timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class PoseSample:
    __slots__ = ("t_ns", "translation", "rotation", "frame_id", "segment",
                 "state", "age_s")

    def __init__(self, pose: dict, frame_id: str, segment, state, age_s):
        self.t_ns = int(pose.get("t_ns", 0))
        self.translation = np.asarray(pose["translation"], dtype=np.float64)
        self.rotation = np.asarray(pose["rotation"], dtype=np.float64)
        self.frame_id = str(frame_id or "")
        self.segment = segment
        self.state = state
        self.age_s = age_s


def parse_pose(status: dict, max_age_s: float,
               expected_role: str | None = None) -> PoseSample | None:
    """Return a valid current pose, or None when stale/lost/invalid."""
    state = str(status.get("tracking_state", status.get("state", ""))).lower()
    if state != "tracking" or status.get("pose_valid") is False:
        return None
    if expected_role is not None and status.get("role") != expected_role:
        return None
    pose = status.get("pose")
    if not isinstance(pose, dict):
        return None
    age = pose.get("age_s")
    if age is None:
        age = status.get("pose_age_s")
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(age) or age < 0 or age > max_age_s:
        return None
    try:
        tr = [float(v) for v in pose["translation"]]
        rot = [float(v) for v in pose["rotation"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(tr) != 3 or len(rot) != 4:
        return None
    if not all(math.isfinite(v) for v in tr + rot):
        return None
    if math.sqrt(sum(v * v for v in rot)) <= 1e-6:
        return None
    frame_id = pose.get("frame_id") or status.get("frame_id") or ""
    if "optical" not in frame_id.lower():
        return None
    try:
        t_ns = int(pose.get("t_ns"))
    except (TypeError, ValueError):
        return None
    if t_ns <= 0:
        return None
    segment = pose.get("segment")
    if segment is None:
        return None
    try:
        segment = int(segment)
    except (TypeError, ValueError):
        return None
    return PoseSample(pose, frame_id, segment, state, age)


def camera_yaw(sample: PoseSample, mount, role: str) -> float:
    """Absolute body yaw (rad, + left) from an optical/body pose."""
    frame = sample.frame_id.lower()
    R = quat_to_matrix(sample.rotation)
    if "body" in frame:
        return yaw_from_matrix(R)
    if "optical" not in frame:
        raise ValueError(f"unknown pose frame_id {sample.frame_id!r}")
    if mount is None:
        raise ValueError(f"no mount calibration for role {role!r}")
    R_bc = body_from_optical_mount(*mount)
    return yaw_from_matrix(R_bc @ R @ R_bc.T)


def relative_yaw(sample: PoseSample, initial: PoseSample, mount,
                 role: str) -> float:
    """Body yaw of `sample` relative to `initial`, in radians."""
    frame = sample.frame_id.lower()
    R0 = quat_to_matrix(initial.rotation)
    R1 = quat_to_matrix(sample.rotation)
    R_rel = R0.T @ R1
    if "body" in frame:
        return yaw_from_matrix(R_rel)
    if "optical" not in frame:
        raise ValueError(f"unknown pose frame_id {sample.frame_id!r}")
    if mount is None:
        raise ValueError(f"no mount calibration for role {role!r}")
    R_bc = body_from_optical_mount(*mount)
    return yaw_from_matrix(R_bc @ R_rel @ R_bc.T)


def wait_valid(client: StatusClient, max_age_s: float, timeout_s: float,
               role: str, segment=None) -> PoseSample:
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout_s:
        try:
            st = client.status()
            last = st
            sample = parse_pose(st, max_age_s, role)
            if sample is not None:
                if segment is not None and sample.segment != segment:
                    raise RuntimeError(
                        f"SLAM segment changed {segment!r} -> {sample.segment!r}")
                return sample
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(0.05)
    raise RuntimeError(f"no valid live pose within {timeout_s:.1f}s "
                       f"(last status: {last!r})")


def _fresh_telemetry(robot, max_age_s: float = 0.3) -> bool:
    st = robot.state
    return (bool(st.connected) and st.last_telemetry_ts > 0
            and time.monotonic() - st.last_telemetry_ts <= max_age_s)


def _all_wheels_armed(robot) -> bool:
    armed = robot.state.wheel_armed or {}
    return bool(armed) and all(armed.values())


def _robot_ok(robot, expected_mode) -> str | None:
    st = robot.state
    if not st.connected:
        return "robot disconnected"
    if not _fresh_telemetry(robot):
        return "robot telemetry stale"
    if st.estop_latched:
        return "estop latched"
    if st.mode != expected_mode:
        return f"mode changed to {pb.Mode.Name(st.mode)}"
    if not _all_wheels_armed(robot):
        return "wheels not armed"
    return None


def _wait_until(predicate, timeout_s: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _wait_telemetry(robot, predicate, after_ts: float,
                    timeout_s: float) -> bool:
    """Wait for a telemetry frame received *after* the command that also
    satisfies `predicate`.  The stale state from before the command cannot
    count as confirmation."""
    return _wait_until(
        lambda: _fresh_telemetry(robot)
        and robot.state.last_telemetry_ts > after_ts
        and predicate(), timeout_s)


def cleanup(robot, initial_mode, timeout_s: float = 2.0) -> list[str]:
    """Zero, disarm, restore mode; wait for telemetry before returning.

    Returns human-readable failures (empty = confirmed clean).  The caller
    must surface a non-empty result; it must not claim a successful stop.
    """
    failures = []
    try:
        t_send = time.monotonic()
        robot.send_twist(0.0, 0.0)
        if not _wait_telemetry(
                robot, lambda: max(abs(v) for v in robot.state.cmd) < 0.01,
                t_send, timeout_s):
            failures.append("zero twist not confirmed by fresh telemetry")
    except Exception as exc:
        failures.append(f"zero twist failed: {exc}")
    try:
        t_send = time.monotonic()
        robot.arm_wheels(False)
        if not _wait_telemetry(
                robot, lambda: bool(robot.state.wheel_armed)
                and not any(robot.state.wheel_armed.values()),
                t_send, timeout_s):
            failures.append("wheel disarm not confirmed by fresh telemetry")
    except Exception as exc:
        failures.append(f"wheel disarm failed: {exc}")
    try:
        t_send = time.monotonic()
        robot.set_mode(initial_mode)
        if not _wait_telemetry(robot, lambda: robot.state.mode == initial_mode,
                               t_send, timeout_s):
            failures.append("mode restore not confirmed by fresh telemetry")
    except Exception as exc:
        failures.append(f"mode restore failed: {exc}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Manual visual-yaw turn test (live SLAM :9096 pose only).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status-url", default="http://127.0.0.1:9096")
    ap.add_argument("--robot-url", default="ws://bebop.local:9090/ws")
    ap.add_argument("--camera", choices=["far", "near"], default="far")
    ap.add_argument("--config-dir", type=Path, default=REPO_ROOT / "config")
    ap.add_argument("--legs", default="45,0,-45",
                    help="cumulative body-yaw targets in degrees relative to "
                         "the first valid pose (default: 45,0,-45)")
    ap.add_argument("--angular-speed", type=float, default=0.15,
                    help="capped |wz| in rad/s (default: 0.15, max 0.25)")
    ap.add_argument("--timeout", type=float, default=8.0,
                    help="hard timeout per leg in seconds (default: 8)")
    ap.add_argument("--settle", type=float, default=0.4,
                    help="seconds to settle after zeroing each leg")
    ap.add_argument("--pose-max-age", type=float, default=0.35,
                    help="reject poses older than this (default: 0.35 s)")
    ap.add_argument("--max-yaw-jump-deg", type=float, default=15.0,
                    help="abort on a larger single-tick visual yaw jump")
    ap.add_argument("--max-translation-jump-m", type=float, default=0.30,
                    help="abort on a larger visual translation jump (zero "
                         "linear motion is commanded)")
    ap.add_argument("--progress-timeout", type=float, default=2.5,
                    help="abort if the yaw stops making progress for this long")
    ap.add_argument("--execute", action="store_true",
                    help="actually arm and move; without it this is read-only")
    args = ap.parse_args()

    speed = min(abs(args.angular_speed), 0.25)
    if speed <= 0:
        ap.error("--angular-speed must be > 0")
    try:
        targets = [math.radians(v) for v in parse_targets(args.legs)]
    except ValueError as exc:
        ap.error(str(exc))

    client = StatusClient(args.status_url)
    try:
        status = client.status()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"[turn-test] cannot read {args.status_url}/status: {exc}",
              file=sys.stderr)
        return 2
    print(f"[turn-test] status: {json.dumps(status, sort_keys=True)}")

    sample = parse_pose(status, args.pose_max_age, args.camera)
    if sample is None:
        print("[turn-test] no valid current pose; is the tracker tracking?",
              file=sys.stderr)
        return 2
    mount = load_mount(args.config_dir, args.camera)
    try:
        yaw0 = camera_yaw(sample, mount, args.camera)
    except ValueError as exc:
        print(f"[turn-test] {exc}", file=sys.stderr)
        return 2
    print(f"[turn-test] frame={sample.frame_id} segment={sample.segment} "
          f"mount={mount} yaw={math.degrees(yaw0):+.1f} deg")

    if not args.execute:
        print(f"[turn-test] read-only: current body yaw "
              f"{math.degrees(yaw0):+.1f} deg; pass --execute to move")
        return 0

    robot = RobotClient(args.robot_url)
    initial_mode = pb.MODE_IDLE
    rc = 0
    try:
        robot.start()
        if not robot.await_connection(5.0):
            print(f"[turn-test] cannot reach robot runtime at {args.robot_url}",
                  file=sys.stderr)
            return 2
        initial_mode = robot.state.mode
        if robot.state.estop_latched:
            print(f"[turn-test] estop is latched ({robot.state.estop_reason!r}); "
                  "refusing to move and not resetting it", file=sys.stderr)
            return 2

        robot.set_mode(pb.MODE_DIAL_IN)
        if not _wait_until(lambda: robot.state.mode == pb.MODE_DIAL_IN, 2.0):
            raise RuntimeError("robot did not enter DIAL_IN")
        robot.arm_wheels(True)
        if not _wait_until(lambda: _all_wheels_armed(robot), 2.0):
            raise RuntimeError("wheels did not arm")

        start = wait_valid(client, args.pose_max_age, 3.0, args.camera)
        segment = start.segment
        print(f"[turn-test] zero pose segment={segment} "
              f"frame={start.frame_id}")

        last_yaw = 0.0
        for i, target in enumerate(targets, 1):
            print(f"[turn-test] leg {i}/{len(targets)}: target "
                  f"{math.degrees(target):+.1f} deg")
            t0 = time.monotonic()
            progress_t = t0
            best = abs(wrap_pi(target - last_yaw))
            away = 0
            while True:
                if time.monotonic() - t0 > args.timeout:
                    raise RuntimeError(
                        f"leg {i} timeout at {math.degrees(last_yaw):+.1f} deg")
                bad = _robot_ok(robot, pb.MODE_DIAL_IN)
                if bad:
                    raise RuntimeError(f"leg {i}: {bad}")
                st = client.status()
                cur = parse_pose(st, args.pose_max_age, args.camera)
                if cur is None:
                    raise RuntimeError(
                        f"leg {i}: live pose became invalid "
                        f"(state={st.get('tracking_state')!r}, "
                        f"age={(st.get('pose') or {}).get('age_s')!r})")
                if segment is not None and cur.segment != segment:
                    raise RuntimeError(
                        f"leg {i}: SLAM segment changed {segment!r} -> "
                        f"{cur.segment!r}")
                yaw = relative_yaw(cur, start, mount, args.camera)
                delta = wrap_pi(yaw - last_yaw)
                if abs(delta) > math.radians(args.max_yaw_jump_deg):
                    raise RuntimeError(
                        f"leg {i}: visual yaw jumped {math.degrees(delta):+.1f} "
                        "deg in one tick")
                if np.linalg.norm(cur.translation - start.translation) > \
                        args.max_translation_jump_m:
                    raise RuntimeError(
                        f"leg {i}: visual translation jumped "
                        f"{np.linalg.norm(cur.translation - start.translation):.2f} m")
                err = wrap_pi(target - yaw)
                if abs(err) < math.radians(2.0):
                    last_yaw = yaw
                    break
                if delta * math.copysign(1.0, err) < 0 and abs(delta) > \
                        math.radians(2.0):
                    away += 1
                    if away >= 3:
                        raise RuntimeError(
                            f"leg {i}: moving away from target "
                            f"(yaw {math.degrees(yaw):+.1f}, "
                            f"target {math.degrees(target):+.1f})")
                else:
                    away = 0
                if abs(err) < best - math.radians(0.5):
                    best = abs(err)
                    progress_t = time.monotonic()
                elif time.monotonic() - progress_t > args.progress_timeout:
                    raise RuntimeError(
                        f"leg {i}: no yaw progress for "
                        f"{args.progress_timeout:.1f}s at "
                        f"{math.degrees(yaw):+.1f} deg")
                wz = math.copysign(min(speed, max(0.03, abs(err) * 2.0)), err)
                robot.send_twist(0.0, wz)
                last_yaw = yaw
                time.sleep(0.05)
            robot.send_twist(0.0, 0.0)
            time.sleep(args.settle)
            print(f"[turn-test] leg {i} done: yaw "
                  f"{math.degrees(last_yaw):+.1f} deg")
    except KeyboardInterrupt:
        print("\n[turn-test] interrupted", file=sys.stderr)
        rc = 130
    except Exception as exc:
        print(f"\n[turn-test] abort: {exc}", file=sys.stderr)
        rc = 1
    finally:
        failures = cleanup(robot, initial_mode)
        robot.stop()
        if failures:
            print("[turn-test] CLEANUP FAILED: " + "; ".join(failures),
                  file=sys.stderr)
            rc = rc or 1
        else:
            print("[turn-test] stopped: zero twist, wheels disarmed, "
                  "mode restored (telemetry-confirmed)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
