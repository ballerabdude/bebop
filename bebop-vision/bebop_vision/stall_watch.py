"""Opt-in stall detector: every thread's stack, captured *during* a stall.

A daemon heartbeat thread arms ``faulthandler.dump_traceback_later`` and
sleeps a few milliseconds. If it cannot get the GIL back within the
threshold, faulthandler's own C watchdog thread (which needs no GIL) writes
the stacks of all Python threads to the log while the stall is still in
progress, so the dump shows what was running, not what ran afterwards.
When the heartbeat wakes it cancels the pending dump and, if it woke late,
logs the stall length with wall-clock stamps plus the current thread-id to
name map (faulthandler prints ids only). Garbage collections longer than
``gc_ms`` are logged from ``gc.callbacks``.

Log format (one file, appended):

    Timeout (0:00:00.150000)!            <- faulthandler, during the stall
    Thread 0x0000ffff... (most recent call first):
      File ".../places/encoder.py", line 161 in encode_bgr
      ...
    [stall] 223 ms, ended 2026-10-23T21:30:00.812 (1792805400812345678 ns)
    [threads] 0x0000ffff...=place-scout 0x0000ffff...=navd-recorder ...
    [gc] gen 2 took 131 ms (collected 12034), 2026-10-23T21:30:04.100

Enable with ``main.py --stall-watch <file> [--stall-ms 150]``. Costs one
short-lived C thread per heartbeat (50 per second at the default 20 ms).
"""
from __future__ import annotations

import faulthandler
import gc
import threading
import time


def _stamp(ns: int | None = None) -> str:
    ns = time.time_ns() if ns is None else ns
    return (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ns / 1e9))
            + f".{(ns // 1_000_000) % 1000:03d}")


class StallWatch:
    def __init__(self, path, threshold_ms: float = 150.0,
                 interval_ms: float = 20.0, gc_ms: float = 50.0):
        self.path = str(path)
        self.threshold_s = float(threshold_ms) / 1000.0
        self.interval_s = float(interval_ms) / 1000.0
        self.gc_ms = float(gc_ms)
        self.stalls = 0
        self.max_stall_ms = 0.0
        self._file = None
        self._thread = None
        self._stop = threading.Event()
        self._gc_t0 = None

    def _write(self, line: str) -> None:
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()

    def _thread_map(self) -> str:
        return " ".join(f"0x{t.ident:016x}={t.name}"
                        for t in threading.enumerate() if t.ident)

    def _gc_callback(self, phase, info):
        if phase == "start":
            self._gc_t0 = time.perf_counter()
            return
        if self._gc_t0 is None:
            return
        ms = (time.perf_counter() - self._gc_t0) * 1000.0
        self._gc_t0 = None
        if ms >= self.gc_ms:
            self._write(f"[gc] gen {info.get('generation')} took {ms:.0f} ms "
                        f"(collected {info.get('collected')}), {_stamp()}")

    def start(self) -> "StallWatch":
        if self._thread is not None:
            return self
        self._file = open(self.path, "a", buffering=1)
        self._write(f"[start] threshold {self.threshold_s * 1000:.0f} ms, "
                    f"heartbeat {self.interval_s * 1000:.0f} ms, {_stamp()}")
        self._write(f"[threads] {self._thread_map()}")
        gc.callbacks.append(self._gc_callback)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="stall-watch",
                                        daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            faulthandler.dump_traceback_later(self.threshold_s, repeat=False,
                                              file=self._file, exit=False)
            t0 = time.perf_counter()
            time.sleep(self.interval_s)
            late = time.perf_counter() - t0 - self.interval_s
            faulthandler.cancel_dump_traceback_later()
            if late >= self.threshold_s:
                ms = late * 1000.0
                self.stalls += 1
                self.max_stall_ms = max(self.max_stall_ms, ms)
                now = time.time_ns()
                self._write(f"[stall] {ms:.0f} ms, ended {_stamp(now)} ({now} ns)")
                self._write(f"[threads] {self._thread_map()}")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        faulthandler.cancel_dump_traceback_later()
        try:
            gc.callbacks.remove(self._gc_callback)
        except ValueError:
            pass
        if self._file is not None:
            self._write(f"[stop] {self.stalls} stalls, max {self.max_stall_ms:.0f} ms, {_stamp()}")
            self._file.close()
            self._file = None
