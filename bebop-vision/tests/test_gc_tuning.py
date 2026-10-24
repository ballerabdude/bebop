"""freeze_heap moves live objects out of the collector's way."""

import gc

from bebop_vision.gc_tuning import freeze_heap


def test_freeze_heap_freezes_and_can_be_undone():
    keep = [[i] for i in range(20_000)]          # some tracked, long-lived objects
    try:
        n = freeze_heap("test")
        assert n >= len(keep) and gc.get_freeze_count() == n
    finally:
        gc.unfreeze()                             # do not leak into other tests
    assert gc.get_freeze_count() == 0
