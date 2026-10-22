"""Route following: drive a taught trail node by node.

Teach-and-repeat on top of the waypoint driver:

* the route frame is anchored to the live pose at start (the robot is
  assumed to stand at node 0, at the recognised/stated start place);
* for the current target node, dead reckoning gives bearing + distance in
  the gyro/odom pose frame;
* the live far view is matched against the node's keyframe (DINOv3 patch
  grids, `visual.py`); when the match is confident its horizontal shift
  replaces the dead-reckoned bearing and slowly corrects the frame yaw;
* the result is streamed to `NavDrive.set_goal` (bearing, short lookahead)
  at a few Hz, so obstacle avoidance and all the safety rules stay with the
  driver;
* a node is passed when within `node_tol_m` or once its segment is
  overshot; arrival at the last node asks place recognition to confirm and
  records the traversal on the place graph (success or failure).

Nothing here talks to wheels directly.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time

import numpy as np

from .bearing import wrap_pi
from .visual import MatchConfig, MatchResult, match_patches


@dataclasses.dataclass
class FollowConfig:
    hz: float = 3.0
    node_tol_m: float = 0.35
    arrive_tol_m: float = 0.40
    lookahead_m: float = 0.8
    min_goal_m: float = 0.25
    yaw_correction_alpha: float = 0.3
    max_blocked_aborts: int = 2
    patch_image_size: int = 224
    visual: bool = True
    confirm_arrival: bool = True
    max_nodes_lost: int = 4         # consecutive nodes without a match


class RouteFollower:
    def __init__(self, drive, store, encoder, capture, *, recognize=None,
                 config: FollowConfig | None = None,
                 match_config: MatchConfig | None = None, log=print):
        self.drive = drive
        self.store = store
        self.encoder = encoder
        self.capture = capture
        self.recognize = recognize
        self.config = config or FollowConfig()
        self.match_config = match_config or MatchConfig()
        self._log = log
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._route = None
        self._nodes: list = []
        self._idx = 0
        self._origin = None          # pose-frame (x, y, yaw) of route node 0
        self._yaw_corr = 0.0
        self._state = "idle"
        self._detail: dict = {}
        self._t0 = 0.0
        self._blocked_aborts = 0
        self._nodes_lost = 0
        self._last_match: MatchResult | None = None
        self._patch_cache: dict = {}
        self._history: list = []
        self._last_nav_state = None

    # -- thread -------------------------------------------------------------

    def start_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop,
                                            name="route-follow", daemon=True)
            self._thread.start()
        return self

    def stop_thread(self, timeout: float = 3.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self):
        period = 1.0 / max(0.5, self.config.hz)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[follow] tick error: {type(exc).__name__}: {exc}")
                self._fail(f"error: {exc}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    # -- API ------------------------------------------------------------------

    def follow(self, route_id: str | None = None, *, to_place: str | None = None,
               from_place: str | None = None) -> dict:
        with self._lock:
            if self._state == "following":
                raise RuntimeError("already following a route")
            route = None
            if route_id:
                route = self.store.get_route(route_id)
                if route is None:
                    raise KeyError(route_id)
            else:
                if not to_place:
                    raise ValueError("pass route_id or to_place")
                start = from_place
                if not start and self.recognize is not None:
                    rec = self.recognize() or {}
                    if rec.get("place"):
                        start = rec["place"]["id"]
                if not start:
                    raise ValueError("cannot tell where we are: pass "
                                     "from_place or route_id")
                routes = self.store.list_routes(from_place=start,
                                                to_place=to_place)
                if not routes:
                    raise KeyError(f"no taught route from {start!r} to "
                                   f"{to_place!r}")
                route = self.store.get_route(routes[0].id)
            if len(route.nodes) < 2:
                raise ValueError("route has fewer than two nodes")
            pose = self.drive.current_pose()
            if pose is None:
                raise RuntimeError("no pose yet")
            if getattr(pose, "heading_source", "gyro") != "gyro":
                raise RuntimeError("pose heading is not gyro-sourced yet")
            x, y, th = pose.as_tuple()
            self._route = route
            self._nodes = list(route.nodes)
            self._origin = (x, y, th)
            self._yaw_corr = 0.0
            self._idx = 1
            self._state = "following"
            self._detail = {}
            self._t0 = time.monotonic()
            self._blocked_aborts = 0
            self._nodes_lost = 0
            self._last_match = None
            self._patch_cache = {}
            self._last_nav_state = None
            self._log(f"[follow] {route.id}: {route.from_name!r} -> "
                      f"{route.to_name!r}, {len(self._nodes)} nodes")
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            if self._state == "following":
                self._finish("stopped", success=False)
            return self.status()

    def status(self) -> dict:
        with self._lock:
            r = self._route
            return {
                "state": self._state,
                "route": (None if r is None else {
                    "id": r.id, "from": r.from_name, "to": r.to_name,
                    "nodes": len(self._nodes)}),
                "target_node": (self._idx if self._state == "following"
                                else None),
                "elapsed_s": (round(time.monotonic() - self._t0, 1)
                              if self._state == "following" else None),
                "yaw_correction_deg": round(math.degrees(self._yaw_corr), 1),
                "match": (None if self._last_match is None
                          else self._last_match.to_dict()),
                "detail": dict(self._detail),
                "history": list(self._history[-10:]),
            }

    # -- geometry -------------------------------------------------------------

    def _node_in_pose_frame(self, node):
        ox, oy, oth = self._origin
        th = oth + self._yaw_corr
        c, s = math.cos(th), math.sin(th)
        return (ox + c * node.x - s * node.y, oy + s * node.x + c * node.y)

    # -- tick -------------------------------------------------------------------

    def tick(self, now: float | None = None) -> None:
        c = self.config
        with self._lock:
            if self._state != "following":
                return
            now = time.monotonic() if now is None else now
            pose = self.drive.current_pose()
            if pose is None:
                self._detail = {"hold": "no pose"}
                return
            px, py, pth = pose.as_tuple()
            nav = self.drive.status()
            nav_state = nav.get("state")
            robot = nav.get("robot") or {}
            if robot.get("estop_latched"):
                self._finish("failed_estop", success=False)
                return
            # Count driver aborts; the driver already stopped the wheels.
            if (nav_state in ("aborted_blocked",) and
                    self._last_nav_state != nav_state):
                self._blocked_aborts += 1
                if self._blocked_aborts >= c.max_blocked_aborts:
                    self._finish("failed_blocked", success=False)
                    return
            elif nav_state in ("aborted_estop", "aborted_link",
                               "aborted_arm"):
                self._finish(f"failed_{nav_state[8:]}", success=False)
                return
            self._last_nav_state = nav_state

            # --- advance over passed nodes -------------------------------
            while True:
                node = self._nodes[self._idx]
                tx, ty = self._node_in_pose_frame(node)
                dist = math.hypot(tx - px, ty - py)
                last = self._idx >= len(self._nodes) - 1
                if last:
                    break
                prev = self._nodes[self._idx - 1]
                ax, ay = self._node_in_pose_frame(prev)
                seg = (tx - ax, ty - ay)
                seg_len = math.hypot(*seg)
                along = (((px - ax) * seg[0] + (py - ay) * seg[1]) / seg_len
                         if seg_len > 1e-6 else 0.0)
                if dist <= c.node_tol_m or along >= seg_len:
                    self._idx += 1
                    continue
                break
            node = self._nodes[self._idx]
            tx, ty = self._node_in_pose_frame(node)
            dist = math.hypot(tx - px, ty - py)
            bearing_dr = wrap_pi(math.atan2(ty - py, tx - px) - pth)
            last = self._idx >= len(self._nodes) - 1
            if last and dist <= c.arrive_tol_m:
                self._arrive()
                return

            # --- visual correction ---------------------------------------
            bearing = bearing_dr
            source = "dead_reckoning"
            match = None
            if c.visual and self.encoder is not None:
                match = self._match_node(node)
                self._last_match = match
                if match is not None and match.confident:
                    self._nodes_lost = 0
                    source = "visual"
                    bearing = match.bearing_rad
                    err = wrap_pi(match.bearing_rad - bearing_dr)
                    self._yaw_corr = wrap_pi(
                        self._yaw_corr + c.yaw_correction_alpha * err)
            goal_d = max(c.min_goal_m, min(c.lookahead_m, dist))
            self._detail = {
                "node": self._idx, "dist_m": round(dist, 3),
                "bearing_dr_deg": round(math.degrees(bearing_dr), 1),
                "bearing_deg": round(math.degrees(bearing), 1),
                "source": source, "goal_m": round(goal_d, 2),
                "nav_state": nav_state,
            }
            # The driver owns motion: refresh its goal unless it is arming
            # or blocked.  A fresh goal resets the driver's blocked timer,
            # so re-sending while blocked would keep it blocked forever
            # instead of letting it abort (seen live 2026-10-21).
            if nav_state in ("arming", "blocked"):
                self._detail["hold"] = nav_state
                return
            try:
                self.drive.set_goal(bearing_deg=math.degrees(bearing),
                                    distance_m=goal_d,
                                    label=f"route:{self._idx}")
            except RuntimeError as exc:
                self._finish(f"failed: {exc}", success=False)

    def _match_node(self, node) -> MatchResult | None:
        role = node.vec_role or "far"
        path = node.keyframes.get(role) or node.keyframes.get("far")
        if not path:
            return None
        key = (node.seq, role)
        ref = self._patch_cache.get(key)
        if ref is None:
            try:
                with open(path, "rb") as f:
                    _cls, ref = self.encoder.encode_patches_jpeg(
                        f.read(), self.config.patch_image_size)
            except Exception as exc:  # noqa: BLE001
                self._detail["match_error"] = str(exc)
                return None
            self._patch_cache[key] = ref
        views = []
        try:
            views = list(self.capture.grab()) if self.capture else []
        except Exception:  # noqa: BLE001
            pass
        view = next((v for v in views if v.role == role), None)
        if view is None:
            return None
        try:
            _cls, live = self.encoder.encode_patches_jpeg(
                view.jpeg, self.config.patch_image_size)
        except Exception as exc:  # noqa: BLE001
            self._detail["match_error"] = str(exc)
            return None
        return match_patches(live, ref, self.match_config)

    # -- endings ----------------------------------------------------------------

    def _arrive(self):
        c = self.config
        confirmed = None
        if c.confirm_arrival and self.recognize is not None:
            try:
                rec = self.recognize() or {}
            except Exception:  # noqa: BLE001
                rec = {}
            place = rec.get("place")
            confirmed = bool(place and place.get("id") == self._route.to_place)
            self._detail["recognized"] = (place or {}).get("name")
        state = ("arrived" if confirmed in (True, None)
                 else "arrived_unconfirmed")
        self._finish(state, success=True)

    def _fail(self, reason: str):
        with self._lock:
            if self._state == "following":
                self._finish(f"failed: {reason}", success=False)

    def _finish(self, state: str, *, success: bool):
        try:
            self.drive.cancel()
        except Exception:  # noqa: BLE001
            pass
        r = self._route
        duration = time.monotonic() - self._t0
        entry = {"state": state, "route": r.id if r else None,
                 "from": r.from_name if r else None,
                 "to": r.to_name if r else None,
                 "duration_s": round(duration, 1),
                 "node_reached": self._idx, "nodes": len(self._nodes),
                 "ts": time.time()}
        if r is not None and state != "stopped":
            try:
                self.store.record_traversal(r.from_place, r.to_place,
                                            success=success,
                                            duration_s=duration if success
                                            else None)
            except Exception as exc:  # noqa: BLE001
                entry["record_error"] = str(exc)
        self._history.append(entry)
        del self._history[:-50]
        self._state = state
        self._log(f"[follow] {entry}")
