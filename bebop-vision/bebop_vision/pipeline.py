"""Camera -> navigable-path model -> annotated output, with a planner sink."""

import threading
import time

import cv2

from . import config
from .camera import Camera
from .navseg import NavSegmenter
from .visualize import draw_hud, draw_nav


class PipelineStats:
    def __init__(self):
        self.fps = 0.0
        self.infer_fps = 0.0
        self._t = time.perf_counter()
        self._frames = 0

    def tick(self):
        self._frames += 1
        now = time.perf_counter()
        if now - self._t >= 1.0:
            self.fps = self._frames / (now - self._t)
            self._frames = 0
            self._t = now


class NavPipeline:
    """Runs the navigable-path model on a dedicated thread; display at camera rate.

    frame_sink(frame, results, stats) fires on every camera frame with the
    freshest nav result — {"nav": (label_map, stats) or None}. The planner
    attaches here.
    """

    def __init__(self, source, nav_model, display, record_path, duration=None, show_hud=True):
        self.source = source
        self.nav_model = nav_model
        self.display = display
        self.record_path = record_path
        self.duration = duration
        self.show_hud = show_hud
        self._stop = False
        self._frame_count = 0
        self._infer_count = 0
        self._lock = threading.Lock()
        self._latest = None
        self._error_logged = False
        self.stats = PipelineStats()

    def stop(self):
        self._stop = True

    def _worker(self, camera, stats):
        frames = 0
        t = time.perf_counter()
        last_ts = 0.0
        while not self._stop:
            frame, ts = camera.read()
            if frame is None or ts == last_ts:
                time.sleep(0.005)
                continue
            last_ts = ts
            try:
                result = self.nav.segment(frame)
            except Exception:
                if not self._error_logged:
                    import traceback

                    traceback.print_exc()
                    self._error_logged = True
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = result
            self._infer_count += 1
            frames += 1
            now = time.perf_counter()
            if now - t >= 1.0:
                stats.infer_fps = frames / (now - t)
                frames = 0
                t = now

    def run(self, frame_sink=None):
        camera = Camera(self.source)
        camera.start()
        print(
            f"[pipeline] camera opened: {camera.actual_width}x{camera.actual_height} "
            f"@ {camera.actual_fps:.0f} fps"
        )
        self.nav = NavSegmenter(self.nav_model, config.DEVICE)

        writer = None
        if self.record_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.record_path, fourcc, camera.actual_fps or 30.0,
                (camera.actual_width, camera.actual_height),
            )
            print(f"[pipeline] recording to {self.record_path}")

        timer = None
        if self.duration:
            timer = threading.Timer(self.duration, self.stop)
            timer.daemon = True
            timer.start()

        worker = threading.Thread(target=self._worker, args=(camera, self.stats), daemon=True)
        worker.start()

        last_ts = 0.0
        try:
            while not self._stop:
                frame, ts = camera.read()
                if frame is None or ts == last_ts:
                    time.sleep(0.002)
                    continue
                last_ts = ts
                self.stats.tick()
                with self._lock:
                    result = self._latest

                annotated = frame.copy()
                if result is not None:
                    draw_nav(annotated, result[0])
                if self.show_hud:
                    draw_hud(annotated, self.stats.fps, 0, "nav", infer_fps=self.stats.infer_fps)

                if frame_sink is not None:
                    frame_sink(frame, {"nav": result}, self.stats)

                if writer is not None:
                    writer.write(annotated)
                if self.display:
                    cv2.imshow("bebop-vision", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                if self._frame_count % 60 == 0:
                    summary = ""
                    if result is not None and result[1]:
                        summary = ", ".join(f"{k} {v:.0%}" for k, v in result[1].items())
                    print(
                        f"[pipeline] frame {self._frame_count} | display {self.stats.fps:5.1f} fps | "
                        f"infer {self.stats.infer_fps:5.1f} Hz | {summary}"
                    )
                self._frame_count += 1
        except KeyboardInterrupt:
            print("\n[pipeline] interrupted")
        finally:
            camera.stop()
            worker.join(timeout=3.0)
            if writer is not None:
                writer.release()
            if self.display:
                cv2.destroyAllWindows()
            print(
                f"[pipeline] stopped: {self._frame_count} display frames, "
                f"{self._infer_count} inference passes"
            )
        return self._frame_count