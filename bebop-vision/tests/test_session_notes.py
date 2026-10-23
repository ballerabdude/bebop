"""SessionNotes: intent slot, recording gate, bounded queues, text safety."""

import threading

import pytest

from bebop_vision.session_notes import MAX_EVENTS, NotRecording, SessionNotes


def test_intent_is_sticky_versioned_and_allowed_while_idle():
    clock = iter(range(100, 200))
    notes = SessionNotes(clock_ns=lambda: next(clock))
    assert notes.intent() is None and not notes.recording
    a = notes.set_intent("to the kitchen", "office", "kitchen")     # no segment open
    assert (a.text, a.from_place, a.to_place, a.version) == ("to the kitchen", "office", "kitchen", 1)
    assert notes.intent() is a
    b = notes.clear_intent()
    assert b.empty and b.version == 2
    assert notes.snapshot()["intent"]["text"] == ""


def test_events_need_an_open_segment():
    notes = SessionNotes()
    with pytest.raises(NotRecording):
        notes.add_event("exclude", "before arming")
    seg = object()
    notes.begin_segment(seg)
    assert notes.snapshot()["recording"] is True
    notes.add_event("obstacle", "sock")
    notes.end_segment(object())              # a stale token changes nothing
    assert notes.recording
    notes.end_segment(seg)
    with pytest.raises(NotRecording):
        notes.add_event("exclude", "after the estop")
    assert [e["kind"] for e in notes.drain_events()] == ["obstacle"]


def test_places_are_ignored_while_idle():
    notes = SessionNotes()
    assert notes.add_place("p1", "Kitchen", 0.9) is None
    notes.begin_segment("seg")
    assert notes.add_place("p1", "Kitchen", 0.9)["name"] == "Kitchen"
    assert len(notes.drain_places()) == 1


def test_events_validate_kind_and_bound():
    notes = SessionNotes()
    notes.begin_segment("seg")
    with pytest.raises(ValueError):
        notes.add_event("not a token!")
    ev = notes.add_event("  Obstacle ", " sock at 1 m ")
    assert ev["kind"] == "obstacle" and ev["note"] == "sock at 1 m"
    for i in range(MAX_EVENTS + 5):
        notes.add_event("note", str(i))
    assert notes.events_dropped == 6
    drained = notes.drain_events()
    assert len(drained) == MAX_EVENTS and notes.drain_events() == []


def test_lone_surrogates_are_made_encodable():
    from bebop_vision import navd_proto
    notes = SessionNotes()
    notes.begin_segment("seg")
    ev = notes.add_event("note", "bad \ud800 text")
    it = notes.set_intent("to \udfff the kitchen", "of\ud800fice", "")
    # protobuf must be able to serialize what the recorder will build
    navd_proto.Event(kind=ev["kind"], note=ev["note"]).SerializeToString()
    navd_proto.Intent(text=it.text, from_place=it.from_place).SerializeToString()
    assert "\ud800" not in ev["note"] and "\udfff" not in it.text


def test_places_drain_from_threads():
    notes = SessionNotes()
    notes.begin_segment("seg")

    def worker(k):
        for i in range(50):
            notes.add_place(f"p{k}", f"Room {k}", 0.9, 0.1)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(notes.drain_places()) == 200 and notes.snapshot()["places_pending"] == 0
