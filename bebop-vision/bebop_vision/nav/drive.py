"""NavDrive: the loop that turns depth + a waypoint into wheel twists.

Runs inside the recorder process (the cameras are exclusive), reading the
already-open rig slots.  While a goal is active it arms the chassis the
same way the voice `move_relative` tool does — DIAL_IN, enable wheels,
drive, then stop + disarm + restore the previous mode on every exit path
— and streams twists at `hz` (the firmware watchdog zeroes a twist not
refreshed within 500 ms).  It never touches the E-STOP: a latched E-STOP
aborts the goal and the loop waits for a person.

The small HTTP surface (default :9097).  The one command surface is
``POST /navigate {"command": "go to the kitchen"}`` (voice and text both
land there; see nav/navigator.py); the rest are the building blocks it
drives:

    POST /navigate {command}   resolve the words and keep driving waypoints
    GET  /navigate             mission state + a spoken-style summary
    POST /navigate/stop

    GET  /healthz              liveness + state
    GET  /status               controller state, last command, goal, robot
    GET  /clearance            latest clearance map (sectors, corridor)
    GET  /pose                 gyro+odom pose estimate (the goal frame)
    POST /goal {bearing_deg, distance_m [, label]}   relative waypoint
    POST /goal {x, y [, label]}                      odom-frame waypoint
    POST /goal {reverse_m [, label]}                 straight reverse (blind)
    POST /goal {turn_deg [, label]}                  rotate in place (+ = left)
    POST /stop                 cancel the goal (stops, disarms)
    POST /route/follow {route_id | to [, from]}      follow a taught route
    POST /route/stop           stop following (and the goal)
    GET  /route                follower state
    POST /trip {to [, from]}   plan over the place graph and drive every leg
    POST /trip/stop            stop the trip (and the follower, and the goal)
    GET  /trip                 trip state (plan, leg, follower)
    POST /object/locate {what} ask the VLM where it is (bearing, range); no motion
    POST /object/seek {what}   drive toward it, one VLM look per leg
    POST /object/stop          stop seeking (and the goal)
    GET  /object               seeker state
"""
from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
from .bearing import (TERMINAL_STATES, BearingController, Command,
                      ControllerConfig, Goal)
from .depth_cone import ClearanceMap, DepthCone
from .pose import Pose, PoseEstimator

DEFAULT_PORT = 9097
ARM_WAIT_S = 4.0


class NavDrive:
    def __init__(self, rig, robot, cone: DepthCone,
                 controller: BearingController | None = None, *,
                 hz: float = 15.0, max_frame_age_s: float = 0.3,
                 max_telemetry_age_s: float = 0.5,
                 roles=None, log=print, pose: PoseEstimator | None = None):
        self.rig = rig
        self.robot = robot
        self.cone = cone
        self.controller = controller or BearingController()
        self.hz = max(2.0, float(hz))
        self.max_frame_age_s = float(max_frame_age_s)
        self.max_telemetry_age_s = float(max_telemetry_age_s)
        self.roles = tuple(roles) if roles else tuple(cone.geometries)
        self._log = log
        if pose is None:
            taps = {}
            for role in self.roles:
                cam = rig.cameras.get(role)
                add = getattr(cam, "add_imu_tap", None)
                tap = add() if add is not None else None
                if tap is not None:
                    taps[role] = tap
            pose = PoseEstimator(list(cone.geometries.values()), taps)
        self.pose = pose
        self._pose: Pose | None = None
        # Optional route follower / trip runner (set by main.py when
        # places are on).
        self.follower = None
        self.trip = None
        self.seeker = None
        self.navigator = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cmap: ClearanceMap | None = None
        self._cmap_error: str | None = None
        self._frames: dict = {}
        self._armed_by_us = False
        self._mode_before = None
        self._mode_changed = False
        self._arming = False
        self._arm_error: str | None = None
        self._goal_set_ts: float | None = None
        self._goal_started_odom = None
        self._goal_current: Goal | None = None
        self._ticks = 0
        self._tick_ms = 0.0
        self._last_cmd: Command = Command(0.0, 0.0, "idle")
        self._last_detail: dict = {}
        self._history: list[dict] = []
        self._cancel = False

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name="nav-drive",
                                        daemon=True)
        self._thread.start()
        self._log(f"[nav] drive loop {self.hz:.0f} Hz on roles {self.roles}")
        return self

    def stop(self, timeout: float = 3.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Belt and braces: never leave the chassis armed by us.
        self._finish("stopped")

    # -- goal API (thread-safe) --------------------------------------------

    def set_goal(self, *, bearing_deg=None, distance_m=None, x=None, y=None,
                 reverse_m=None, turn_deg=None, label: str = "") -> dict:
        st = self.robot.state
        if not st.connected:
            raise RuntimeError("robot runtime not connected")
        if st.estop_latched:
            raise RuntimeError(
                f"E-STOP latched ({st.estop_reason or 'no reason'}); "
                "a person must clear it")
        with self._lock:
            est = self._pose
        odom = est.as_tuple() if est is not None else tuple(st.odom)
        if turn_deg is not None:
            if not (1.0 <= abs(float(turn_deg)) <= 180.0):
                raise ValueError("turn_deg must be between 1 and 180 in magnitude")
            goal = Goal.turn(odom, math.radians(float(turn_deg)), label)
        elif reverse_m is not None:
            if float(reverse_m) <= 0:
                raise ValueError("reverse_m must be > 0")
            goal = Goal.reverse(odom, float(reverse_m), label)
        elif x is not None and y is not None:
            goal = Goal(float(x), float(y), label)
        elif bearing_deg is not None and distance_m is not None:
            if float(distance_m) <= 0:
                raise ValueError("distance_m must be > 0")
            goal = Goal.from_bearing(odom, math.radians(float(bearing_deg)),
                                     float(distance_m), label)
        else:
            raise ValueError(
                "goal needs bearing_deg+distance_m, x+y, reverse_m or turn_deg")
        with self._lock:
            self.controller.set_goal(goal, odom)
            self._goal_current = goal
            self._goal_set_ts = time.monotonic()
            self._goal_started_odom = odom
            self._last_detail = {}
            self._cancel = False
            self._arm_error = None
        self._log(f"[nav] goal {goal.label or ''} -> pose ({goal.x:.2f}, "
                  f"{goal.y:.2f}) from ({odom[0]:.2f}, {odom[1]:.2f}, "
                  f"{math.degrees(odom[2]):.0f} deg, "
                  f"{est.heading_source if est else 'odom'})")
        return self._goal_dict(goal)

    def cancel(self) -> dict:
        with self._lock:
            self._cancel = True
        return self.status()

    # -- status -------------------------------------------------------------

    def status(self) -> dict:
        st = self.robot.state
        with self._lock:
            goal = self.controller.goal
            cmd = self._last_cmd
            cm = self._cmap
            return {
                "ok": True,
                "service": "nav",
                "state": cmd.state,
                "active": goal is not None,
                "goal": self._goal_dict(goal) if goal else None,
                "command": {"vx": cmd.vx, "wz": cmd.wz, "detail": cmd.detail},
                "armed_by_nav": self._armed_by_us,
                "arming": self._arming,
                "arm_error": self._arm_error,
                "pose": (None if self._pose is None
                         else self._pose.to_dict()),
                "clearance_age_s": (None if cm is None
                                    else round(cm.age_s(), 3)),
                "clearance_error": self._cmap_error,
                "hz": self.hz,
                "ticks": self._ticks,
                "tick_ms": round(self._tick_ms, 2),
                "robot": {
                    "connected": st.connected,
                    "mode": pb.Mode.Name(st.mode) if st.mode else None,
                    "estop_latched": st.estop_latched,
                    "estop_reason": st.estop_reason,
                    "wheel_armed": dict(st.wheel_armed),
                    "odom": [round(float(v), 4) for v in st.odom],
                    "telemetry_age_s": round(
                        time.monotonic() - st.last_telemetry_ts, 3)
                        if st.last_telemetry_ts else None,
                },
                "history": list(self._history[-10:]),
            }

    def current_pose(self) -> Pose | None:
        with self._lock:
            return self._pose

    def clearance(self) -> dict | None:
        with self._lock:
            cm = self._cmap
        return None if cm is None else cm.to_dict()

    def depth_png(self, role: str) -> bytes | None:
        """Latest raw uint16 depth frame for `role` as PNG (debugging:
        feed it to tools/nav_cone_preview.py)."""
        with self._lock:
            fr = self._frames.get(role)
        if fr is None or getattr(fr, "depth", None) is None:
            return None
        import cv2
        ok, buf = cv2.imencode(".png", fr.depth)
        return buf.tobytes() if ok else None

    @staticmethod
    def _goal_dict(goal: Goal | None):
        if goal is None:
            return None
        out = {"x": round(goal.x, 3), "y": round(goal.y, 3),
               "label": goal.label}
        if goal.reverse_m > 0:
            out["reverse_m"] = round(goal.reverse_m, 3)
        if goal.rotate_to is not None:
            out["rotate_to_deg"] = round(math.degrees(goal.rotate_to), 1)
        return out

    # -- loop -----------------------------------------------------------------

    def _loop(self):
        period = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._tick(t0)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self._log(f"[nav] tick error: {type(exc).__name__}: {exc}")
                try:
                    self.robot.send_twist(0.0, 0.0)
                except Exception:  # noqa: BLE001
                    pass
            self._tick_ms = (time.monotonic() - t0) * 1000.0
            self._ticks += 1
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _read_frames(self, now: float) -> dict:
        frames = {}
        for role in self.roles:
            cam = self.rig.cameras.get(role)
            fr = cam.read() if cam is not None else None
            if fr is not None and (
                    now - float(getattr(fr, "recv_ts", 0.0) or 0.0)
                    > self.max_frame_age_s):
                fr = None
            frames[role] = fr
        return frames

    def _tick(self, now: float):
        frames = self._read_frames(now)
        try:
            cm = self.cone.evaluate(frames)
            err = None if cm is not None else "no fresh depth frame"
        except Exception as exc:  # noqa: BLE001
            cm, err = None, f"{type(exc).__name__}: {exc}"
        st = self.robot.state
        est = self.pose.update(tuple(st.odom), tuple(st.cmd), now)
        with self._lock:
            if cm is not None:
                self._cmap = cm
            self._cmap_error = err
            self._frames = frames
            self._pose = est
            goal = self.controller.goal
            cancel = self._cancel
        if goal is None:
            return
        if cancel:
            self._finish("stopped")
            return
        if st.estop_latched:
            self._log(f"[nav] abort: E-STOP latched ({st.estop_reason!r})")
            self._finish("aborted_estop")
            return
        if not st.connected or (
                now - st.last_telemetry_ts > self.max_telemetry_age_s):
            self._log("[nav] abort: robot link lost / telemetry stale")
            self._finish("aborted_link")
            return
        if not self._armed_by_us:
            if not self._arm():
                self._finish("aborted_arm")
                return
        with self._lock:
            cmd = self.controller.step(est.as_tuple(), self._cmap, now)
            self._last_cmd = cmd
            if cmd.detail:
                self._last_detail = dict(cmd.detail, state=cmd.state)
        if cmd.state in TERMINAL_STATES or cmd.state.startswith("aborted"):
            self._log(f"[nav] {cmd.state}: {cmd.detail}")
            self._finish(cmd.state)
            return
        self.robot.send_twist(cmd.vx, cmd.wz)

    # -- arming / release ------------------------------------------------------

    def _arm(self) -> bool:
        """DIAL_IN + enable wheels, waiting on telemetry.  Mirrors the voice
        tool's sequence; the firmware is the real backstop."""
        robot = self.robot
        st = robot.state
        self._arming = True
        try:
            self._mode_before = st.mode
            self._mode_changed = False
            if st.mode != pb.MODE_DIAL_IN:
                robot.set_mode(pb.MODE_DIAL_IN)
                self._mode_changed = True
                deadline = time.monotonic() + ARM_WAIT_S
                while time.monotonic() < deadline and st.mode != pb.MODE_DIAL_IN:
                    if st.estop_latched or self._stop.is_set():
                        break
                    time.sleep(0.05)
                if st.mode != pb.MODE_DIAL_IN:
                    self._arm_error = "could not switch to DIAL_IN"
                    self._log(f"[nav] {self._arm_error}")
                    return False
            robot.arm_wheels(True)
            deadline = time.monotonic() + ARM_WAIT_S
            while time.monotonic() < deadline:
                if st.estop_latched or self._stop.is_set():
                    break
                if st.wheel_armed and all(st.wheel_armed.values()):
                    self._armed_by_us = True
                    self._log("[nav] wheels armed")
                    return True
                time.sleep(0.05)
            if st.estop_latched:
                self._arm_error = (f"E-STOP latched while arming: "
                                   f"{st.estop_reason or 'no reason'}")
            else:
                self._arm_error = "wheels did not arm"
            self._log(f"[nav] {self._arm_error}")
            return False
        finally:
            self._arming = False

    def _finish(self, state: str):
        """Stop, disarm, restore the mode, record the outcome."""
        robot = self.robot
        try:
            robot.send_twist(0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            goal = self._goal_current
            self._goal_current = None
            self.controller.clear(state)
            self._last_cmd = self.controller.last
            started = self._goal_started_odom
            t_set = self._goal_set_ts
            self._goal_set_ts = None
            self._goal_started_odom = None
            self._cancel = False
            armed = self._armed_by_us
            self._armed_by_us = False
            mode_changed = self._mode_changed
            mode_before = self._mode_before
            self._mode_changed = False
        # Record the outcome before the disarm settle so a status poll that
        # sees active=False also sees the history entry.
        if t_set is not None:
            with self._lock:
                est = self._pose
            od = est.as_tuple() if est is not None else tuple(robot.state.odom)
            entry = {
                "state": state,
                "goal": self._goal_dict(goal),
                "duration_s": round(time.monotonic() - t_set, 2),
                "travelled_m": (None if started is None else round(
                    math.hypot(od[0] - started[0], od[1] - started[1]), 3)),
                "end_pose": [round(float(v), 3) for v in od],
                "heading_source": (est.heading_source if est else "odom"),
                "last": dict(self._last_detail),
                "ts": time.time(),
            }
            with self._lock:
                self._history.append(entry)
                del self._history[:-50]
            self._log(f"[nav] done: {entry}")
        if armed or mode_changed:
            time.sleep(0.1)
            try:
                if armed:
                    robot.arm_wheels(False)
                if mode_changed and mode_before is not None:
                    robot.set_mode(mode_before)
            except Exception:  # noqa: BLE001
                pass


# -- HTTP ---------------------------------------------------------------------

def _json_bytes(value) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


class NavServer:
    def __init__(self, drive: NavDrive, host: str = "0.0.0.0",
                 port: int = DEFAULT_PORT):
        self.drive = drive
        self.host = host
        self.port = int(port)
        self.bound_port: int | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        if self._httpd is not None:
            return self
        drive = self.drive

        class Handler(_NavHandler):
            nav = drive

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.bound_port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="nav-http", daemon=True)
        self._thread.start()
        print(f"[nav] http on :{self.bound_port} "
              f"(/healthz /status /clearance; POST /goal /stop)", flush=True)
        return self

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            self._thread = None


class _NavHandler(BaseHTTPRequestHandler):
    nav: NavDrive

    def log_message(self, format, *args):
        return

    def _send(self, code: int, value) -> None:
        body = _json_bytes(value)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            s = self.nav.status()
            self._send(200, {"ok": True, "service": "nav",
                             "state": s["state"], "active": s["active"],
                             "clearance_age_s": s["clearance_age_s"]})
        elif path == "/status":
            self._send(200, self.nav.status())
        elif path == "/navigate":
            n = self.nav.navigator
            if n is None:
                self._send(503, {"ok": False, "error": "navigator not configured"})
            else:
                self._send(200, n.status())
        elif path == "/trip":
            t = self.nav.trip
            if t is None:
                self._send(503, {"ok": False, "error": "trips not configured"})
            else:
                self._send(200, t.status())
        elif path == "/object":
            o = self.nav.seeker
            if o is None:
                self._send(503, {"ok": False, "error": "object goals not configured"})
            else:
                self._send(200, o.status())
        elif path == "/route":
            f = self.nav.follower
            if f is None:
                self._send(503, {"ok": False, "error": "route following "
                                 "not configured (needs places + --nav)"})
            else:
                self._send(200, f.status())
        elif path == "/pose":
            s = self.nav.status()
            self._send(200 if s["pose"] else 503,
                       s["pose"] or {"ok": False, "error": "no pose yet"})
        elif path == "/clearance":
            cm = self.nav.clearance()
            if cm is None:
                self._send(503, {"ok": False, "error": "no clearance yet"})
            else:
                self._send(200, cm)
        elif path == "/depth":
            from urllib.parse import parse_qs
            q = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            role = (q.get("role") or ["near"])[0]
            png = self.nav.depth_png(role)
            if png is None:
                self._send(503, {"ok": False, "error": f"no {role} depth"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(png)
        elif path == "/":
            self._send(200, {"service": "nav", "endpoints": [
                "GET /healthz", "GET /status", "GET /clearance",
                "GET /depth?role=near|far", "GET /route", "POST /goal",
                "POST /stop", "POST /route/follow", "POST /route/stop",
                "GET /trip", "POST /trip", "POST /trip/stop",
                "GET /object", "POST /object/locate", "POST /object/seek",
                "POST /object/stop", "GET /navigate", "POST /navigate",
                "POST /navigate/stop"]})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except ValueError as exc:
            self._send(400, {"ok": False, "error": f"bad json: {exc}"})
            return
        if path == "/goal":
            try:
                goal = self.nav.set_goal(
                    bearing_deg=body.get("bearing_deg"),
                    distance_m=body.get("distance_m"),
                    x=body.get("x"), y=body.get("y"),
                    reverse_m=body.get("reverse_m"),
                    turn_deg=body.get("turn_deg"),
                    label=str(body.get("label") or ""))
            except (ValueError, TypeError) as exc:
                self._send(400, {"ok": False, "error": str(exc)})
                return
            except RuntimeError as exc:
                self._send(409, {"ok": False, "error": str(exc)})
                return
            self._send(202, {"ok": True, "goal": goal})
        elif path == "/stop":
            n = self.nav.navigator
            if n is not None:
                n.stop()
            o = self.nav.seeker
            if o is not None:
                o.stop()
            t = self.nav.trip
            if t is not None:
                t.stop()
            f = self.nav.follower
            if f is not None:
                f.stop()
            self._send(200, {"ok": True, "status": self.nav.cancel()})
        elif path == "/trip":
            t = self.nav.trip
            if t is None:
                self._send(503, {"ok": False, "error": "trips not configured"})
                return
            to = body.get("to") or body.get("to_place")
            if not to:
                self._send(400, {"ok": False, "error": "to is required"})
                return
            try:
                st = t.start(str(to), from_place=body.get("from")
                             or body.get("from_place"))
            except KeyError as exc:
                self._send(404, {"ok": False,
                                 "error": f"no such place: {exc.args[0]}"})
                return
            except LookupError as exc:
                self._send(404, {"ok": False, "error": str(exc)})
                return
            except ValueError as exc:
                self._send(400, {"ok": False, "error": str(exc)})
                return
            except RuntimeError as exc:
                self._send(409, {"ok": False, "error": str(exc)})
                return
            self._send(202, {"ok": True, "status": st})
        elif path == "/navigate":
            n = self.nav.navigator
            if n is None:
                self._send(503, {"ok": False, "error": "navigator not configured"})
                return
            text = str(body.get("command") or body.get("text") or "").strip()
            if not text:
                self._send(400, {"ok": False, "error": "command is required"})
                return
            try:
                st = n.command(text)
            except (KeyError, LookupError) as exc:
                self._send(404, {"ok": False, "error": str(exc)})
                return
            except ValueError as exc:
                self._send(400, {"ok": False, "error": str(exc)})
                return
            except RuntimeError as exc:
                self._send(409, {"ok": False, "error": str(exc)})
                return
            self._send(202, {"ok": True, "status": st})
        elif path == "/navigate/stop":
            n = self.nav.navigator
            if n is None:
                self._send(503, {"ok": False, "error": "navigator not configured"})
            else:
                self._send(200, {"ok": True, "status": n.stop()})
        elif path in ("/object/locate", "/object/seek"):
            o = self.nav.seeker
            if o is None:
                self._send(503, {"ok": False, "error": "object goals not configured"})
                return
            what = str(body.get("what") or "").strip()
            if not what:
                self._send(400, {"ok": False, "error": "what is required"})
                return
            try:
                if path == "/object/locate":
                    self._send(200, {"ok": True, "location": o.locate(what)})
                else:
                    self._send(202, {"ok": True, "status": o.seek(what)})
            except RuntimeError as exc:
                self._send(409, {"ok": False, "error": str(exc)})
            except ValueError as exc:
                self._send(400, {"ok": False, "error": str(exc)})
        elif path == "/object/stop":
            o = self.nav.seeker
            if o is None:
                self._send(503, {"ok": False, "error": "object goals not configured"})
            else:
                self._send(200, {"ok": True, "status": o.stop()})
        elif path == "/trip/stop":
            t = self.nav.trip
            if t is None:
                self._send(503, {"ok": False, "error": "trips not configured"})
            else:
                self._send(200, {"ok": True, "status": t.stop()})
        elif path == "/route/follow":
            f = self.nav.follower
            if f is None:
                self._send(503, {"ok": False, "error": "route following "
                                 "not configured"})
                return
            try:
                st = f.follow(body.get("route_id"),
                              to_place=body.get("to") or body.get("to_place"),
                              from_place=body.get("from")
                              or body.get("from_place"))
            except KeyError as exc:
                self._send(404, {"ok": False, "error": str(exc.args[0])})
                return
            except ValueError as exc:
                self._send(400, {"ok": False, "error": str(exc)})
                return
            except RuntimeError as exc:
                self._send(409, {"ok": False, "error": str(exc)})
                return
            self._send(202, {"ok": True, "status": st})
        elif path == "/route/stop":
            f = self.nav.follower
            if f is None:
                self._send(503, {"ok": False, "error": "route following "
                                 "not configured"})
            else:
                self._send(200, {"ok": True, "status": f.stop()})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_PUT(self):
        self._send(405, {"ok": False, "error": "method not allowed"})

    do_DELETE = do_PATCH = do_PUT
