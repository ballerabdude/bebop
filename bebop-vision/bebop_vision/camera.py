"""Threaded video stream consumer for the bebop robot.

The firmware owns the camera exclusively and republishes it as an MJPEG
stream on `GET /video` (see `firmware/bebop-linux/src/video.rs`); the
operator app plays the same stream, and bebop-vision is a third
subscriber. This module is the consumer side of that split: a threaded
reader over a source URL (the robot's MJPEG endpoint, an RTSP(S)
camera, or a video file) that always serves the freshest frame,
reconnects with backoff when a network source drops, and never touches
a capture device directly — two processes cannot stream the same V4L2
device anyway, which is exactly why the firmware owns it.
"""

import os
import threading
import time

import cv2

RECONNECT_PREFIXES = ("http://", "https://", "rtsp://", "rtsps://")


class Camera:
    def __init__(self, source):
        if not isinstance(source, str) or not source.strip():
            raise TypeError(
                "source must be a stream URL or file path — the firmware owns "
                "capture devices; point at its MJPEG endpoint, e.g. "
                "http://bebop.local:9090/video"
            )
        if source.startswith(("rtsp://", "rtsps://")) \
                and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.source = source
        self.reconnects = source.startswith(RECONNECT_PREFIXES)
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._running = False
        self._reader = None
        self._consecutive_failures = 0
        self._needs_reopen = False
        self.read_fps = 0.0
        self._open()
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

    def start(self):
        if self._running:
            return self
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        return self

    def _read_loop(self):
        frames = 0
        t = time.monotonic()
        while self._running:
            if self._needs_reopen:
                self._needs_reopen = False
                try:
                    self.cap.release()
                    self._open()
                    print("[camera] stream reconnected")
                except RuntimeError as exc:
                    print(f"[camera] reconnect failed: {exc}")
                    time.sleep(1.0)
                    self._needs_reopen = True
                continue
            ok, frame = self.cap.read()
            if not ok:
                self._consecutive_failures += 1
                # Network sources can drop mid-flight (firmware restart,
                # WiFi blip): reopen after ~1.5 s of failed reads. A file
                # that ran to EOF just stays ended — the pipeline idles on
                # "no new frame" rather than looping the file.
                if self.reconnects and self._consecutive_failures >= 30:
                    print("[camera] stream stalled, reconnecting ...")
                    self._consecutive_failures = 0
                    self._needs_reopen = True
                else:
                    time.sleep(0.05)
                continue
            self._consecutive_failures = 0
            with self._lock:
                self._frame = frame
                self._frame_ts = time.monotonic()
            frames += 1
            now = time.monotonic()
            if now - t >= 1.0:
                self.read_fps = frames / (now - t)
                frames = 0
                t = now

    def read(self):
        """Return (frame, timestamp) of the most recent captured frame."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._frame_ts

    def stop(self):
        self._running = False
        self._needs_reopen = False
        if self._reader is not None:
            self._reader.join(timeout=3.0)
        self.cap.release()