"""Keep full garbage collections short in the long-running vision process.

CPython's full (generation 2) collection walks every tracked object, and the
interpreter lock is held for the whole walk, so every thread stops. In the
vision process the long-lived startup objects dominate: measured on the Thor
2026-10-23, numpy/cv2/torch/transformers alone are ~227 k tracked objects
(full collection 55 ms) and the places DINOv3 ViT-L model brings it to
~505 k (154 ms). A full collection then stalls the recorder for 150-180 ms
(seen as frame and gyro gaps), usually triggered by the place scout's
allocations.

``freeze_heap`` collects once and moves every object that exists now into
CPython's permanent generation (``gc.freeze``), which later collections
skip: after freezing, a full collection scans only what was allocated since
(measured < 1 ms). Call it when startup objects are in place, and again after
any large lazy load (the places encoder warms up on a background thread).
Frozen objects are never reclaimed, which is right for objects that live as
long as the process.
"""
from __future__ import annotations

import gc
import time


def freeze_heap(label: str = "") -> int:
    """Collect, then freeze everything alive; returns the frozen count."""
    t0 = time.perf_counter()
    gc.collect()
    gc.freeze()
    n = gc.get_freeze_count()
    print(f"[gc] froze {n:,} objects{f' after {label}' if label else ''} "
          f"({(time.perf_counter() - t0) * 1000:.0f} ms)", flush=True)
    return n
