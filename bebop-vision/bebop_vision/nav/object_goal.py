"""Object goals: "go to the dresser" without a taught route.

The VLM is the decision layer, called once per leg, never per tick:

    1. grab the far view, ask the VLM for a bounding box of the object
       (strict JSON), convert the box centre to a bearing through the far
       color intrinsics;
    2. read the far depth at that direction for the range (median over a
       window; depth frame and color frame have different intrinsics, so
       the direction is re-projected rather than the pixel copied);
    3. hand the driver one bounded goal (bearing, min(step, range - standoff));
    4. when the driver finishes that goal, look again; stop when the range
       is within `standoff_m` + tolerance, or after too many legs / misses.

Everything about motion safety stays with `NavDrive`.  `locate()` alone
answers "where is it / are we there yet" without moving.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
import threading
import time

import numpy as np

from ..router.vlm import image_data_url


@dataclasses.dataclass
class ObjectGoalConfig:
    standoff_m: float = 0.9           # stop this far from the object
    arrive_tol_m: float = 0.25
    step_m: float = 1.0               # longest single leg
    min_step_m: float = 0.3
    max_legs: int = 12
    max_misses: int = 2               # consecutive "not found" before failing
    # Search: on a miss, turn to the next scan heading (relative to where
    # the search began) and look again.  The far camera sees ~46 deg to
    # each side, so 45 deg steps tile the surroundings.  A hint in the
    # words ("on the left") orders the scan to that side first.
    scan: bool = True
    scan_headings_deg: tuple = (45, -45, 90, -90, 135, -135, 180)
    # Once the object has been seen, a single miss (a turn step put it just
    # outside the frame, or the model blinked) re-looks before scanning.
    relook_after_seen: int = 1
    max_blocked: int = 2
    # Detours: when a leg toward the object is blocked, drive up to 1 m
    # into the most open sector near the object's bearing and look again;
    # when nothing is open, back up a little first.  The stop-cone only
    # sees +-60 deg, so this is local exploration, not planning.
    max_detours: int = 3
    # A leg that has not finished in this long is cancelled and treated
    # as blocked (the driver can dither for a minute on a hard leg).
    leg_timeout_s: float = 25.0
    scan_timeout_s: float = 12.0
    detour_min_clear_m: float = 1.0
    detour_reverse_m: float = 0.4
    image_width: int = 640            # downscale for the VLM
    depth_window_px: int = 24
    depth_min_valid: int = 20
    default_range_m: float = 1.5      # no depth: assume this far and step
    min_confidence: float = 0.4
    hz: float = 2.0


LOCATE_PROMPT = (
    "You are the forward camera of a small indoor robot. Find {what} in this "
    "image. Reply with ONLY a JSON object: {{\"found\": true or false, "
    "\"box\": [x0, y0, x1, y1] or null, \"confidence\": 0 to 1, "
    "\"note\": \"at most 10 words\"}} where the box coordinates are "
    "decimal fractions of the image width and height between 0 and 1 (never "
    "pixels), x to the right, y down, and the box covers only {what}. If it "
    "is not visible, found is false and box is null."
)


_HINT_RE = re.compile(
    r"\b(?:on|to|at)\s+(?:the|my|your|our)?\s*(?:left|right)(?:\s+side)?\b"
    r"|\b(?:behind|in front of)\s+(?:you|me|us|the robot)\b"
    r"|\b(?:over|back)\s+there\b", re.I)


def strip_hints(what: str) -> str:
    """Remove positional hints ("on the left") from an object description.

    The hint orders the scan; it must not reach the model, which otherwise
    answers "not visible on the left" once the robot has turned to face it
    (seen live 2026-10-21)."""
    out = _HINT_RE.sub("", what)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.")
    return out or what


def parse_locate_reply(text: str, image_size=None, original_size=None) -> dict:
    """Pull the JSON object out of a model reply (tolerates prose).

    Box coordinates are fractions; a value above 1 is taken as pixels of
    the image that was sent (`image_size` = (w, h)) when it fits, else of
    the original frame (`original_size`), per axis.  The 27B has answered
    in all three conventions, sometimes mixed in one box.
    """
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:120]!r}")
    d = json.loads(m.group(0))
    found = bool(d.get("found")) and d.get("box") is not None
    box = None
    if found:
        b = [float(v) for v in d["box"]]
        if len(b) != 4:
            raise ValueError("box needs four numbers")
        if max(b) > 1.5:
            # Pixel coordinates (sometimes only on one axis): normalise
            # each value by the sent image size when it exceeds 1.
            if not image_size:
                raise ValueError(f"box looks like pixels: {b}")
            w, h = image_size
            ow, oh = original_size or image_size

            def norm_pair(lo, hi, sent, orig):
                if lo <= 1.5 and hi <= 1.5:
                    return lo, hi
                # Pick the divisor that puts both ends of the axis in range,
                # preferring the original frame size (what the 27B tends to
                # answer in), then the sent size.
                for div in (orig, sent):
                    if lo / div <= 1.02 and hi / div <= 1.02:
                        return lo / div, hi / div
                raise ValueError(f"box out of range: {b}")
            x0, x1 = norm_pair(b[0], b[2], w, ow)
            y0, y1 = norm_pair(b[1], b[3], h, oh)
            b = [x0, y0, x1, y1]
        x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in b)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"degenerate box: {d.get('box')}")
        box = [x0, y0, x1, y1]
    return {"found": found and box is not None, "box": box,
            "confidence": float(d.get("confidence") or 0.0),
            "note": str(d.get("note") or "")[:80]}


class ObjectSeeker:
    def __init__(self, drive, vlm, capture, geometry, color_intrinsics: dict,
                 *, config: ObjectGoalConfig | None = None, log=print):
        """`geometry` is the far camera's CameraGeometry (depth intrinsics +
        extrinsics); `color_intrinsics` has color_fx/fy/cx/cy/width/height."""
        self.drive = drive
        self.vlm = vlm
        self.capture = capture
        self.geom = geometry
        self.ci = color_intrinsics
        self.config = config or ObjectGoalConfig()
        self._log = log
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._what = ""
        self._legs = 0
        self._misses = 0
        self._blocked = 0
        self._last: dict = {}
        self._waiting_goal = False
        self._t0 = 0.0
        self._history: list = []
        self._scan_plan: list = []
        self._scan_turned = 0.0           # cumulative turn since the search began
        self._scanning = False
        self._ask = ""                    # description sent to the model
        self._seen = False
        self._detours = 0
        self._detouring = False
        self._last_bearing = 0.0
        self._retry_scan = None
        self._goal_t0 = 0.0

    # -- thread ---------------------------------------------------------------

    def start_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="object-goal",
                                            daemon=True)
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
                self._log(f"[object] tick error: {type(exc).__name__}: {exc}")
                with self._lock:
                    if self._state == "seeking":
                        self._finish(f"failed: {exc}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    # -- perception -------------------------------------------------------------

    def locate(self, what: str, frames=None) -> dict:
        """Ask the VLM where `what` is; add bearing and range.  Never moves."""
        import cv2

        c = self.config
        views = list(self.capture.grab()) if self.capture else []
        view = next((v for v in views if v.role == "far"), None)
        if view is None:
            raise RuntimeError("no fresh far color frame")
        img = cv2.imdecode(np.frombuffer(view.jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if w > c.image_width:
            img_s = cv2.resize(img, (c.image_width, int(c.image_width * h / w)))
        else:
            img_s = img
        ok, buf = cv2.imencode(".jpg", img_s, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url(buf.tobytes())}},
            {"type": "text", "text": LOCATE_PROMPT.format(what=what)}]}]
        t0 = time.monotonic()
        reply = self.vlm.complete(msgs, max_tokens=120, temperature=0.0)
        sizes = ((img_s.shape[1], img_s.shape[0]), (w, h))
        try:
            out = parse_locate_reply(reply, *sizes)
        except ValueError as exc:
            # One retry with the format spelled out; the 27B occasionally
            # mixes pixel rows into a fractional box.
            msgs[0]["content"][1]["text"] += (
                " Your previous reply was invalid (" + str(exc)[:60] +
                "). All four box numbers must be decimals between 0 and 1.")
            reply = self.vlm.complete(msgs, max_tokens=120, temperature=0.0)
            out = parse_locate_reply(reply, *sizes)
        out["reply"] = reply.strip()[:160]
        out["vlm_ms"] = round((time.monotonic() - t0) * 1000)
        out["what"] = what
        if not out["found"]:
            return out
        x0, y0, x1, y1 = out["box"]
        cx_px = (x0 + x1) / 2 * w
        cy_px = (y0 + y1) / 2 * h
        # Bearing (+ left) and elevation (+ up) of the box centre.
        bearing = -math.atan2(cx_px - self.ci["color_cx"], self.ci["color_fx"])
        elev = -math.atan2(cy_px - self.ci["color_cy"], self.ci["color_fy"])
        out["bearing_deg"] = round(math.degrees(bearing), 1)
        out["elevation_deg"] = round(math.degrees(elev), 1)
        out["range_m"] = self._range_at(bearing, elev, frames)
        return out

    def _range_at(self, bearing: float, elev: float, frames=None):
        """Median far-camera depth (m) around the given direction, or None."""
        c = self.config
        cam = None
        if frames is not None:
            fr = frames.get("far")
        else:
            rig = getattr(self.capture, "rig", None)
            cams = getattr(rig, "cameras", {}) or {}
            cam = cams.get("far")
            fr = cam.read() if cam is not None else None
        if fr is None or getattr(fr, "depth", None) is None:
            return None
        g = self.geom
        u = g.cx + g.fx * math.tan(-bearing)
        v = g.cy + g.fy * math.tan(-elev)
        if not (0 <= u < g.width and 0 <= v < g.height):
            return None
        k = c.depth_window_px
        u0, u1 = max(0, int(u) - k), min(g.width, int(u) + k)
        v0, v1 = max(0, int(v) - k), min(g.height, int(v) + k)
        win = fr.depth[v0:v1, u0:u1].astype(np.float32)
        scale = float(getattr(fr, "depth_scale_mm", None) or 1.0) / 1000.0
        vals = win[win > 0] * scale
        vals = vals[(vals > 0.2) & (vals < 8.0)]
        if vals.size < c.depth_min_valid:
            return None
        return round(float(np.median(vals)), 3)

    # -- API ------------------------------------------------------------------------

    def seek(self, what: str) -> dict:
        with self._lock:
            if self._state == "seeking":
                raise RuntimeError("already seeking an object")
            what = str(what).strip()
            if not what:
                raise ValueError("what is required")
            self._what = what
            self._ask = strip_hints(what)
            self._seen = False
            self._legs = 0
            self._misses = 0
            self._blocked = 0
            self._last = {}
            self._waiting_goal = False
            self._scan_plan = self._plan_scan(what)
            self._scan_turned = 0.0
            self._scanning = False
            self._detours = 0
            self._detouring = False
            self._last_bearing = 0.0
            self._retry_scan = None
            self._state = "seeking"
            self._t0 = time.monotonic()
            self._log(f"[object] seeking {what!r}")
            return self.status()

    def _plan_scan(self, what: str) -> list:
        """Scan headings (deg, relative to the search start), hinted side
        first."""
        if not self.config.scan:
            return []
        hs = list(self.config.scan_headings_deg)
        w = what.lower()
        if "left" in w:
            hs.sort(key=lambda h: (h < 0, abs(h)))
        elif "right" in w:
            hs.sort(key=lambda h: (h > 0, abs(h)))
        elif "behind" in w or "back" in w:
            hs.sort(key=lambda h: -abs(h))
        return hs

    def stop(self) -> dict:
        with self._lock:
            if self._state == "seeking":
                try:
                    self.drive.cancel()
                except Exception:  # noqa: BLE001
                    pass
                self._finish("stopped")
            return self.status()

    def status(self) -> dict:
        with self._lock:
            return {"state": self._state, "what": self._what,
                    "asked": self._ask, "seen": self._seen,
                    "legs": self._legs, "misses": self._misses,
                    "scanning": self._scanning,
                    "detours": self._detours, "detouring": self._detouring,
                    "scan_turned_deg": self._scan_turned,
                    "scans_left": len(self._scan_plan),
                    "last": dict(self._last),
                    "elapsed_s": (round(time.monotonic() - self._t0, 1)
                                  if self._state == "seeking" else None),
                    "history": list(self._history[-10:])}

    # -- tick -----------------------------------------------------------------------

    def tick(self) -> None:
        c = self.config
        with self._lock:
            if self._state != "seeking":
                return
            nav = self.drive.status()
            nav_state = nav.get("state")
            if (nav.get("robot") or {}).get("estop_latched"):
                self._finish("failed_estop")
                return
            if self._waiting_goal:
                if nav.get("active") or nav.get("arming"):
                    limit = (c.scan_timeout_s if self._scanning
                             else c.leg_timeout_s)
                    if time.monotonic() - self._goal_t0 > limit:
                        self._log(f"[object] leg timed out after {limit:.0f} s: "
                                  "cancelling, treating as blocked")
                        try:
                            self.drive.cancel()
                        except Exception:  # noqa: BLE001
                            pass
                        nav_state = "aborted_blocked"
                    else:
                        return
                self._waiting_goal = False
                was_scan = self._scanning
                was_detour = self._detouring
                self._detouring = False
                if nav_state == "aborted_blocked":
                    self._blocked += 1
                    if self._try_unblock(was_scan, was_detour):
                        return
                    self._finish("failed_blocked")
                    return
                elif nav_state and nav_state.startswith("aborted"):
                    self._finish(f"failed_{nav_state[8:]}")
                    return
                elif nav_state == "reached" and not was_scan and not was_detour:
                    self._blocked = 0
            if self._legs >= c.max_legs:
                self._finish("failed_max_legs")
                return
            what = self._ask or self._what
        # Look (one VLM call per leg) WITHOUT the lock: the call takes
        # seconds and status()/stop() must stay responsive meanwhile.
        try:
            loc = self.locate(what)
        except Exception as exc:  # noqa: BLE001
            loc = {"found": False, "error": str(exc), "what": what}
        with self._lock:
            if self._state != "seeking":
                return                      # stopped while we were looking
            self._last = loc
            if not loc.get("found") or loc.get("confidence", 0) < c.min_confidence:
                self._misses += 1
                self._log(f"[object] not found ({self._misses}): "
                          f"{loc.get('note') or loc.get('error')}")
                if self._seen and self._misses <= c.relook_after_seen:
                    return                   # look again before turning away
                if self._scan_plan:
                    # Turn to the next scan heading and look again.
                    target = self._scan_plan.pop(0)
                    delta = target - self._scan_turned
                    delta = (delta + 180.0) % 360.0 - 180.0
                    if abs(delta) >= 1.0:
                        try:
                            self.drive.set_goal(turn_deg=delta,
                                                label=f"scan:{self._what}")
                        except RuntimeError as exc:
                            self._finish(f"failed: {exc}")
                            return
                        self._retry_scan = (target, self._scan_turned)
                        self._scan_turned = target
                        self._scanning = True
                        self._waiting_goal = True
                        self._goal_t0 = time.monotonic()
                        self._log(f"[object] scanning: turn {delta:+.0f} deg")
                    return
                if self._misses >= c.max_misses:
                    self._finish("failed_not_found")
                return
            self._misses = 0
            self._scanning = False
            self._seen = True
            rng = loc.get("range_m")
            bearing = float(loc["bearing_deg"])
            self._last_bearing = bearing
            if rng is not None and rng <= c.standoff_m + c.arrive_tol_m:
                self._finish("arrived")
                return
            remaining = (rng - c.standoff_m) if rng is not None \
                else c.default_range_m - c.standoff_m
            step = max(c.min_step_m, min(c.step_m, remaining))
            try:
                self.drive.set_goal(bearing_deg=bearing, distance_m=step,
                                    label=f"object:{self._what}")
            except RuntimeError as exc:
                self._finish(f"failed: {exc}")
                return
            self._legs += 1
            self._waiting_goal = True
            self._goal_t0 = time.monotonic()
            self._log(f"[object] leg {self._legs}: bearing {bearing:+.1f} deg, "
                      f"range {rng}, step {step:.2f} m")

    def _try_unblock(self, was_scan: bool, was_detour: bool) -> bool:
        """After a blocked goal: detour into the most open sector, or back
        up a little.  Returns True when a recovery goal was issued."""
        c = self.config
        if self._detours >= c.max_detours:
            return False
        self._detours += 1
        cm = None
        try:
            cm = self.drive.clearance()
        except Exception:  # noqa: BLE001
            cm = None
        best = None
        if cm and not was_scan:
            bearings = cm.get("bearings_deg") or []
            clears = cm.get("clearance_m") or []
            for b, cl in zip(bearings, clears):
                if cl is None or cl < c.detour_min_clear_m:
                    continue
                # Prefer open sectors near the object's last bearing.
                score = cl - 0.01 * abs(b - self._last_bearing)
                if best is None or score > best[0]:
                    best = (score, b, cl)
        try:
            if best is not None:
                _, b, cl = best
                dist = max(c.min_step_m, min(c.step_m, cl - 0.7))
                self.drive.set_goal(bearing_deg=b, distance_m=dist,
                                    label=f"detour:{self._what}")
                self._log(f"[object] detour {self._detours}: bearing {b:+.0f} "
                          f"deg, {dist:.2f} m")
            else:
                self.drive.set_goal(reverse_m=c.detour_reverse_m,
                                    label=f"unblock:{self._what}")
                self._log(f"[object] unblock {self._detours}: reverse "
                          f"{c.detour_reverse_m} m")
                if was_scan and self._retry_scan is not None:
                    target, before = self._retry_scan
                    self._scan_plan.insert(0, target)
                    self._scan_turned = before      # the turn did not happen
        except RuntimeError:
            return False
        self._detouring = True
        self._waiting_goal = True
        self._goal_t0 = time.monotonic()
        return True

    def _finish(self, state: str):
        entry = {"state": state, "what": self._what, "legs": self._legs,
                 "duration_s": round(time.monotonic() - self._t0, 1),
                 "last": dict(self._last), "ts": time.time()}
        self._history.append(entry)
        del self._history[:-50]
        self._state = state
        self._log(f"[object] {entry}")
