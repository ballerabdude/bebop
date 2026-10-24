"""StallWatch catches a GIL hog and dumps the hog's stack during the stall."""

import gc
import sys
import time

from bebop_vision.stall_watch import StallWatch


def _hog(seconds):
    t_end = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < t_end:      # pure Python: keeps the GIL
        x += 1
    return x


def test_detects_a_gil_hog_and_dumps_its_stack(tmp_path):
    log = tmp_path / "stall.log"
    w = StallWatch(log, threshold_ms=100, interval_ms=10, gc_ms=1e9).start()
    old = sys.getswitchinterval()
    try:
        time.sleep(0.05)
        sys.setswitchinterval(1.0)          # the hog will not yield the GIL
        _hog(0.35)
    finally:
        sys.setswitchinterval(old)
    time.sleep(0.1)
    w.stop()
    text = log.read_text()
    assert w.stalls >= 1 and w.max_stall_ms >= 100
    assert "Timeout (" in text                # faulthandler fired mid-stall
    assert "in _hog" in text                  # ... and caught the hog
    assert "[stall]" in text and "[threads]" in text and "stall-watch" in text


def test_logs_long_garbage_collections(tmp_path):
    log = tmp_path / "gc.log"
    w = StallWatch(log, threshold_ms=10_000, interval_ms=50, gc_ms=0).start()
    gc.collect()
    w.stop()
    assert "[gc] gen 2 took" in log.read_text()
