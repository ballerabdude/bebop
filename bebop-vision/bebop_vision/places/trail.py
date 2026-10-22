"""Teach mode for routes: record a trail while the robot is driven.

A route is "how you get from A to B": an ordered list of nodes, each with a
pose in the route frame (origin and heading at the start of the drive), the
distance along the trail, keyframe JPEGs and a DINOv3 descriptor of the
forward view.  Poses come from the nav pose estimate (camera-gyro heading +
wheel-odometry distance; see `nav/pose.py`), never from raw wheel-odometry
heading, which over-reads turns on carpet.

    teacher.start(from_place)   anchor the route frame, take node 0
    ... operator drives (teleop or nav goals) ...
    teacher.stop(to_place)      take the last node, persist, record the edge

Node policy: a new node every `step_m` of travel, or when the heading has
changed by `turn_deg` since the last node, with at least `min_interval_s`
between nodes; the first and last poses are always nodes.  Teaching never
moves the robot.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time

import numpy as np

from .store import PlaceStore


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclasses.dataclass
class TrailConfig:
    hz: float = 5.0
    step_m: float = 0.6
    turn_deg: float = 25.0
    min_interval_s: float = 0.4
    max_nodes: int = 400
    max_duration_s: float = 1800.0
    keyframe_roles: tuple = ("far", "near")
    descriptor_role: str = "far"
    require_gyro: bool = True


class TrailTeacher:
    """Owns one in-progress teach session at a time."""

    def __init__(self, store: PlaceStore, capture, encoder, pose_fn,
                 config: TrailConfig | None = None, *, recognize=None,
                 log=print):
        self.store = store
        self.capture = capture
        self.encoder = encoder
        self.pose_fn = pose_fn            # () -> Pose-like or None
        self.recognize = recognize        # () -> place dict or None
        self.config = config or TrailConfig()
        self._log = log
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._from_id: str | None = None
        self._label: str = ""
        self._origin = None               # (x, y, theta) of the route frame
        self._nodes: list[dict] = []
        self._t0 = 0.0
        self._last_pose = None            # local (x, y, theta) at last node
        self._last_node_ts = 0.0
        self._length = 0.0
        self._prev_local = None
        self._heading_sources: set = set()
        self._error: str | None = None
        self._last_result: dict | None = None

    # -- lifecycle ----------------------------------------------------------

    def start_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop,
                                            name="trail-teach", daemon=True)
            self._thread.start()
        return self

    def stop_thread(self, timeout: float = 3.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self):
        period = 1.0 / max(1.0, self.config.hz)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
                self._log(f"[trail] tick error: {self._error}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    # -- pose helpers -------------------------------------------------------

    def _pose(self):
        p = self.pose_fn() if self.pose_fn is not None else None
        if p is None:
            return None
        if hasattr(p, "as_tuple"):
            src = getattr(p, "heading_source", "unknown")
            return tuple(p.as_tuple()), src
        return (float(p[0]), float(p[1]), float(p[2])), "unknown"

    def _to_local(self, pose):
        ox, oy, oth = self._origin
        dx, dy = pose[0] - ox, pose[1] - oy
        c, s = math.cos(-oth), math.sin(-oth)
        return (c * dx - s * dy, s * dx + c * dy, _wrap(pose[2] - oth))

    # -- session API ----------------------------------------------------------

    def start(self, from_place: str | None = None, *, label: str = "",
              now: float | None = None) -> dict:
        c = self.config
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._active:
                raise RuntimeError("a teach session is already active")
            got = self._pose()
            if got is None:
                raise RuntimeError("no pose available (is --nav running?)")
            pose, src = got
            if c.require_gyro and src != "gyro":
                raise RuntimeError(
                    f"pose heading source is {src!r}, not the camera gyro; "
                    "wait for gyro calibration (robot still for ~1 s)")
            place = None
            if from_place:
                place = self.store.get_place(from_place)
                if place is None:
                    raise KeyError(from_place)
            elif self.recognize is not None:
                rec = self.recognize() or {}
                if rec.get("place"):
                    place = self.store.get_place(rec["place"]["id"])
            if place is None:
                raise ValueError(
                    "cannot tell where the route starts: pass from_place "
                    "(recognition did not clear its threshold)")
            self._active = True
            self._from_id = place.id
            self._label = label or ""
            self._origin = pose
            self._nodes = []
            self._t0 = now
            self._last_pose = None
            self._last_node_ts = 0.0
            self._length = 0.0
            self._prev_local = (0.0, 0.0, 0.0)
            self._heading_sources = {src}
            self._error = None
            self._take_node((0.0, 0.0, 0.0), force=True, now=now)
            self._log(f"[trail] teaching from {place.name!r}")
            return self.status()

    def stop(self, to_place: str | None = None, *, facts=None,
             now: float | None = None) -> dict:
        """Finish the session.  `to_place` is an existing place id/name,
        or a new name (a place is created from the current view), or None
        to use recognition."""
        with self._lock:
            if not self._active:
                raise RuntimeError("no teach session is active")
            now = time.monotonic() if now is None else now
            self._take_node(None, force=True, now=now)
            dst = None
            created_place = False
            if to_place:
                dst = self.store.get_place(to_place)
                if dst is None:
                    dst = self._create_place(to_place, facts)
                    created_place = True
            elif self.recognize is not None:
                rec = self.recognize() or {}
                if rec.get("place"):
                    dst = self.store.get_place(rec["place"]["id"])
            if dst is None:
                raise ValueError(
                    "cannot tell where the route ends: pass to_place "
                    "(an existing place, or a new name to create one)")
            nodes = list(self._nodes)
            duration = now - self._t0
            route = self.store.create_route(
                self._from_id, dst.id, nodes, length_m=self._length,
                duration_s=duration, label=self._label or None,
                heading_source="+".join(sorted(self._heading_sources)))
            self._active = False
            self._nodes = []
            result = {
                "route": route.to_dict(root=self.store.root, nodes=False),
                "nodes": route.node_count,
                "created_place": created_place,
                "to_place": dst.to_dict(root=self.store.root),
            }
            self._last_result = result
            self._log(f"[trail] stored {route.id}: {route.from_name!r} -> "
                      f"{route.to_name!r}, {route.node_count} nodes, "
                      f"{route.length_m:.2f} m, {route.duration_s:.0f} s")
            return result

    def cancel(self) -> dict:
        with self._lock:
            was = self._active
            self._active = False
            self._nodes = []
            if was:
                self._log("[trail] teach cancelled")
            return {"cancelled": was}

    def status(self) -> dict:
        with self._lock:
            got = self._pose()
            pose_d = None
            if got is not None:
                (x, y, th), src = got
                pose_d = {"x": round(x, 3), "y": round(y, 3),
                          "theta_deg": round(math.degrees(th), 1),
                          "heading_source": src}
            out = {
                "active": self._active,
                "from_place": self._from_id,
                "label": self._label if self._active else None,
                "nodes": len(self._nodes),
                "length_m": round(self._length, 3),
                "elapsed_s": (round(time.monotonic() - self._t0, 1)
                              if self._active else None),
                "pose": pose_d,
                "error": self._error,
                "last_result": self._last_result,
                "config": {"step_m": self.config.step_m,
                           "turn_deg": self.config.turn_deg,
                           "max_nodes": self.config.max_nodes},
            }
            if self._active and self._nodes:
                last = self._nodes[-1]
                out["last_node"] = {"seq": len(self._nodes) - 1,
                                    "x": round(last["x"], 3),
                                    "y": round(last["y"], 3),
                                    "dist_m": round(last["dist_m"], 3)}
            return out

    # -- per tick -------------------------------------------------------------

    def tick(self, now: float | None = None) -> bool:
        """Sample the pose; take a node when the policy says so.  Returns
        True when a node was taken."""
        c = self.config
        with self._lock:
            if not self._active:
                return False
            got = self._pose()
            if got is None:
                self._error = "pose unavailable"
                return False
            pose, src = got
            self._heading_sources.add(src)
            local = self._to_local(pose)
            # Accumulate travelled length from consecutive samples.
            px, py, _ = self._prev_local
            self._length += math.hypot(local[0] - px, local[1] - py)
            self._prev_local = local
            now = time.monotonic() if now is None else now
            elapsed = now - self._t0
            if (len(self._nodes) >= c.max_nodes
                    or elapsed > c.max_duration_s):
                self._error = "node/duration limit reached; stop the session"
                return False
            if self._last_pose is None:
                return self._take_node(local, force=True, now=now)
            lx, ly, lth = self._last_pose
            moved = math.hypot(local[0] - lx, local[1] - ly)
            turned = abs(math.degrees(_wrap(local[2] - lth)))
            if now - self._last_node_ts < c.min_interval_s:
                return False
            if moved >= c.step_m or turned >= c.turn_deg:
                return self._take_node(local, now=now)
            return False

    def _take_node(self, local, *, force: bool = False, now=None) -> bool:
        c = self.config
        now = time.monotonic() if now is None else now
        if local is None:
            got = self._pose()
            if got is None:
                return False
            local = self._to_local(got[0])
        views = []
        try:
            views = list(self.capture.grab()) if self.capture else []
        except Exception as exc:  # noqa: BLE001
            self._error = f"capture failed: {exc}"
        keyframes = {v.role: v.jpeg for v in views
                     if v.role in c.keyframe_roles and v.jpeg}
        vec = None
        vec_role = None
        src = next((v for v in views if v.role == c.descriptor_role), None)
        if src is None and views:
            src = views[0]
        if src is not None and self.encoder is not None:
            try:
                vec = np.asarray(self.encoder.encode_jpeg(src.jpeg),
                                 dtype=np.float32)
                vec_role = src.role
            except Exception as exc:  # noqa: BLE001
                self._error = f"encode failed: {exc}"
        if not keyframes and not force:
            # No fresh view: do not spend a node on it.
            return False
        self._nodes.append({
            "x": float(local[0]), "y": float(local[1]),
            "theta": float(local[2]),
            "dist_m": float(self._length),
            "t_s": float(now - self._t0),
            "keyframes": keyframes, "vec": vec, "vec_role": vec_role,
        })
        self._last_pose = tuple(local)
        self._last_node_ts = now
        return True

    def _create_place(self, name: str, facts=None):
        place = self.store.create_place(name)
        views = []
        try:
            views = list(self.capture.grab()) if self.capture else []
        except Exception:  # noqa: BLE001
            pass
        for v in views:
            self.store.add_keyframe(place.id, v.jpeg, role=v.role,
                                    stamp_us=v.stamp_us)
            if self.encoder is not None:
                try:
                    self.store.add_descriptor(
                        place.id, self.encoder.encode_jpeg(v.jpeg),
                        source=f"dinov3:{v.role}")
                except Exception as exc:  # noqa: BLE001
                    self._log(f"[trail] encode failed for {v.role}: {exc}")
        for text in facts or []:
            self.store.add_fact(place.id, str(text), source="operator")
        return self.store.get_place(place.id)
