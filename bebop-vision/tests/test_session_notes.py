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


# -- drive runs -----------------------------------------------------------

_RUN_ID = r"^r\d{8}_\d{6}_[0-9a-f]{4}$"


def _clock(start=1_792_000_000_000_000_000, step=1_000_000_000):
    t = iter(range(start, start + 1000 * step, step))
    return lambda: next(t)


def test_runs_new_repeat_and_continue():
    import re
    from bebop_vision.session_notes import NoRun
    notes = SessionNotes(clock_ns=_clock())
    with pytest.raises(NoRun):
        notes.set_intent("x", mode="continue")
    with pytest.raises(ValueError):                 # NoRun is a ValueError too
        notes.set_intent(None, task="explore", mode="continue")
    a = notes.set_intent("to the kitchen", "office", "kitchen", "go_to_place", "Kitchen")
    assert re.match(_RUN_ID, a.run_id) and a.run_started_ns == a.stamp_ns
    assert (a.task, a.target) == ("go_to_place", "Kitchen")
    # continue keeps the run and changes only what was given
    b = notes.set_intent(None, to_place="pantry", target="Pantry", mode="continue")
    assert (b.run_id, b.run_started_ns) == (a.run_id, a.run_started_ns)
    assert (b.text, b.from_place, b.to_place, b.target) == (
        "to the kitchen", "office", "pantry", "Pantry")
    assert b.version == a.version + 1
    # a task change without a target drops the old task's target; the same
    # task again keeps it
    e = notes.set_intent(None, task="look_for", mode="continue")
    assert (e.run_id, e.task, e.target) == (a.run_id, "look_for", "")
    e = notes.set_intent(None, target="keys", mode="continue")
    e = notes.set_intent(None, task="look_for", mode="continue")
    assert e.target == "keys"
    e = notes.set_intent(None, task="go_through_door", mode="continue")
    assert (e.task, e.target) == ("go_through_door", "")
    b = notes.set_intent(None, task="go_to_place", target="Pantry", mode="continue")
    # the same text again is a different run (new and repeat both start one)
    c = notes.set_intent("to the kitchen", mode="repeat")
    d = notes.set_intent("to the kitchen")
    assert len({a.run_id, c.run_id, d.run_id}) == 3
    rows = notes.recent()
    assert [r["run_id"] for r in rows] == [d.run_id, c.run_id, a.run_id]
    assert rows[0]["ended_ns"] == 0 and rows[1]["ended_ns"] > 0
    # a run replaced while open closes as unfinished
    assert rows[2]["to"] == "pantry" and rows[2]["outcome"] == "unfinished"
    assert rows[1]["outcome"] == "unfinished" and rows[0]["outcome"] == ""
    d2 = notes.intent().to_dict()
    assert set(d2) == {"text", "from", "to", "task", "target", "run_id",
                       "run_started_ns", "stamp_ns", "version"}


def test_task_target_and_mode_validation():
    notes = SessionNotes()
    for bad in ({"task": "dance"}, {"task": "go_through_door", "target": "up"},
                {"task": "maneuver", "target": "spin"}, {"mode": "again"}):
        with pytest.raises(ValueError):
            notes.set_intent("x", **bad)
    assert notes.intent() is None
    it = notes.set_intent("", task="Maneuver", target="Turn-Around")
    assert (it.task, it.target, it.empty) == ("maneuver", "turn-around", False)
    assert notes.set_intent("", task="hold", target="anything").target == ""
    assert notes.set_intent("x", task="go_through_door", target="ahead").target == "ahead"
    empty = notes.set_intent("", "", "", "", "")
    assert empty.empty and empty.run_id == "" and empty.run_started_ns == 0


def test_end_run_outcomes_and_events():
    notes = SessionNotes(clock_ns=_clock())
    a = notes.set_intent("find the sock", task="look_for", target="sock")
    # interrupted only marks the run; not recording, so no event
    out = notes.end_run("interrupted", "estop")
    assert out["event_recorded"] is False and out["run"]["interruptions"] == 1
    assert notes.intent().run_id == a.run_id
    notes.begin_segment("seg")
    out = notes.end_run("found", "under the couch")
    assert out["event_recorded"] is True
    assert out["run"]["outcome"] == "found" and out["run"]["ended_ns"] > 0
    assert out["run"]["interruptions"] == 1
    assert notes.intent().empty and notes.intent().run_id == ""
    assert notes.drain_events()[0]["kind"] == "found"
    with pytest.raises(ValueError):
        notes.end_run("teleported")
    # with no run open (a double tap) nothing is queued, and run is None
    out = notes.end_run("gave_up")
    assert out == {"run": None, "event_recorded": False}
    assert notes.drain_events() == []
    notes.end_segment("seg")
    assert notes.end_run("arrived") == {"run": None, "event_recorded": False}
    assert notes.recent()[0]["outcome"] == "found"


def test_clear_and_repeat_run_and_cap():
    from bebop_vision.session_notes import MAX_RECENT
    notes = SessionNotes(clock_ns=_clock())
    a = notes.set_intent("loop", "hall", "office", "explore", "")
    notes.clear_intent()
    assert notes.recent()[0]["run_id"] == a.run_id
    assert notes.recent()[0]["outcome"] == "unfinished"     # a clear
    b = notes.repeat_run(a.run_id)
    assert b.run_id != a.run_id
    assert (b.text, b.from_place, b.to_place, b.task) == ("loop", "hall", "office", "explore")
    with pytest.raises(ValueError):
        notes.repeat_run("r_nope")
    for i in range(MAX_RECENT + 3):
        notes.set_intent(f"run {i}")
    rows = notes.recent()
    assert len(rows) == MAX_RECENT and rows[0]["text"] == f"run {MAX_RECENT + 2}"
    assert notes.snapshot()["recent"] == rows


def test_state_persists_across_restarts(tmp_path, capsys):
    path = tmp_path / "intent.json"
    notes = SessionNotes(clock_ns=_clock(), state_path=path)
    old = notes.set_intent("to the door", task="go_through_door", target="left")
    notes.end_run("arrived")
    cur = notes.set_intent("to the kitchen", "office", "kitchen", "go_to_place", "Kitchen")
    notes.end_run("interrupted")
    assert not (tmp_path / "intent.json.tmp").exists()

    again = SessionNotes(clock_ns=_clock(), state_path=path)
    it = again.intent()
    assert (it.run_id, it.run_started_ns, it.text, it.task, it.target) == (
        cur.run_id, cur.run_started_ns, "to the kitchen", "go_to_place", "Kitchen")
    assert it.version == 1                      # fresh: the recorder writes it
    rows = again.recent()
    assert [r["run_id"] for r in rows] == [cur.run_id, old.run_id]
    assert rows[0]["interruptions"] == 1 and rows[1]["outcome"] == "arrived"
    # a continue after the restart keeps the restored run
    assert again.set_intent(None, to_place="pantry", mode="continue").run_id == cur.run_id

    # an ended run leaves no open intent behind
    again.end_run("gave_up")
    third = SessionNotes(state_path=path)
    assert third.intent() is None and third.recent()[0]["outcome"] == "gave_up"


def test_missing_or_corrupt_state_is_ignored(tmp_path, capsys):
    assert SessionNotes(state_path=tmp_path / "none.json").intent() is None
    for body in ("{oops", "[1, 2]", '{"intent": {"text": "x", "task": "dance", "run_id": "r1"}}',
                 '{"recent": [{"text": "no id"}]}'):
        p = tmp_path / "bad.json"
        p.write_text(body)
        notes = SessionNotes(state_path=p)
        assert notes.intent() is None and notes.recent() == []
    assert "ignoring" in capsys.readouterr().out
    # a state file that cannot be written only warns
    notes = SessionNotes(state_path=tmp_path / "no-dir" / "intent.json")
    notes.set_intent("still works")
    assert notes.intent().text == "still works"
    assert "could not save" in capsys.readouterr().out


# -- breaks: unfinished, resume, stops ------------------------------------


def test_end_unfinished_closes_the_run_with_its_note():
    from bebop_vision.session_notes import ENDING_OUTCOMES, OUTCOMES, UNFINISHED
    assert UNFINISHED == "unfinished" and UNFINISHED in ENDING_OUTCOMES
    assert UNFINISHED in OUTCOMES
    notes = SessionNotes(clock_ns=_clock())
    a = notes.set_intent("to the kitchen", task="go_to_place", target="Kitchen")
    out = notes.end_run("unfinished", "battery ran out")
    assert out["event_recorded"] is False
    assert out["run"]["outcome"] == "unfinished" and out["run"]["ended_ns"] > 0
    assert notes.intent().run_id == "" and notes.recent()[0]["run_id"] == a.run_id
    # while recording the decline is also an event
    notes.set_intent("again")
    notes.begin_segment("seg")
    assert notes.end_run("unfinished", "x")["event_recorded"] is True
    assert [(e["kind"], e["note"]) for e in notes.drain_events()] == [("unfinished", "x")]


def test_resume_needs_an_open_run_and_is_logged():
    from bebop_vision.session_notes import MAX_RESUMES, NoRun
    notes = SessionNotes(clock_ns=_clock())
    with pytest.raises(NoRun):
        notes.resume_run("nothing open")
    a = notes.set_intent("loop", task="explore")
    notes.begin_segment("seg")
    out = notes.resume_run(" after the charger ")
    assert out["event_recorded"] is True and out["held"] is False
    assert out["run"]["run_id"] == a.run_id and out["run"]["resumes"] == 1
    ev = notes.drain_events()
    assert [(e["kind"], e["note"]) for e in ev] == [("resume", "after the charger")]
    for i in range(MAX_RESUMES + 3):
        notes.resume_run(str(i))
    assert notes.recent()[0]["resumes"] == MAX_RESUMES
    assert notes.intent_dict()["resumes"] == MAX_RESUMES
    notes.end_run("arrived")
    with pytest.raises(NoRun):
        notes.resume_run()


def test_resume_while_idle_is_the_first_event_of_the_next_segment():
    notes = SessionNotes(clock_ns=_clock())
    notes.set_intent("loop", task="explore")
    out = notes.resume_run("was charging")
    assert (out["event_recorded"], out["held"], out["run"]["resumes"]) == (False, True, 1)
    snap = notes.snapshot()
    assert snap["events_pending"] == 0 and snap["events_held"] == 1
    notes.begin_segment("seg")
    notes.add_event("obstacle", "sock")
    ev = notes.drain_events()
    assert [e["kind"] for e in ev] == ["resume", "obstacle"]
    assert ev[0]["note"] == "was charging"
    assert "run_id" not in ev[0]
    # restamped at the segment open, so the recorder's pre-segment filter
    # keeps it; never after the events that follow it
    assert ev[0]["stamp_ns"] <= ev[1]["stamp_ns"]
    assert notes.snapshot()["events_held"] == 0
    notes.end_segment("seg")


def test_a_held_resume_is_dropped_when_its_run_closes():
    notes = SessionNotes(clock_ns=_clock())
    notes.set_intent("loop", task="explore")
    notes.resume_run("going on")
    notes.end_run("unfinished")                 # changed their mind
    notes.set_intent("other")
    notes.begin_segment("seg")
    assert notes.drain_events() == []


def test_note_stop_logs_the_run_and_the_closing_segment():
    from bebop_vision.session_notes import MAX_STOPS, STOP_CAUSES
    assert set(STOP_CAUSES) == {"disarmed", "wheel_disarmed", "idle",
                                "estop_operator", "estop_auto", "link_lost",
                                "rolled", "recorder_stop"}
    notes = SessionNotes(clock_ns=_clock())
    with pytest.raises(ValueError):
        notes.note_stop("bored")
    # no run, no segment: nothing to log, nothing queued
    out = notes.note_stop("disarmed")
    assert out["run"] is None and out["event_recorded"] is False
    a = notes.set_intent("to the door", task="go_through_door", target="left")
    assert notes.intent_dict()["last_stop"] is None
    assert notes.intent_dict()["stops"] == 0
    notes.begin_segment("seg")
    out = notes.note_stop("estop_auto", "left: motor fault 0x01 (overcurrent)")
    assert out["event_recorded"] is True and out["run"]["stops"] == 1
    notes.end_segment("seg")
    ev = notes.drain_events()
    assert [(e["kind"], e["note"]) for e in ev] == [
        ("stop", "estop_auto: left: motor fault 0x01 (overcurrent)")]
    notes.note_stop("disarmed")                 # outside a segment: run only
    assert notes.drain_events() == []
    it = notes.intent_dict()
    assert it["run_id"] == a.run_id and it["stops"] == 2
    assert it["last_stop"]["cause"] == "disarmed" and it["last_stop"]["detail"] == ""
    assert it["last_stop"]["t_ns"] > 0
    row = notes.snapshot()["recent"][0]
    assert (row["stops"], row["resumes"], row["last_stop"]["cause"]) == (2, 0, "disarmed")
    for _ in range(MAX_STOPS + 2):
        notes.note_stop("rolled")
    assert notes.recent()[0]["stops"] == MAX_STOPS
    # the closed run keeps its stops; the empty intent has none
    notes.end_run("arrived")
    assert notes.recent()[0]["stops"] == MAX_STOPS
    assert notes.recent()[0]["last_stop"]["cause"] == "rolled"
    it = notes.intent_dict()
    assert (it["stops"], it["resumes"], it["last_stop"]) == (0, 0, None)


def test_breaks_persist_and_old_state_reads_as_unfinished(tmp_path):
    import json
    path = tmp_path / "intent.json"
    notes = SessionNotes(clock_ns=_clock(), state_path=path)
    a = notes.set_intent("to the kitchen", task="go_to_place", target="Kitchen")
    notes.note_stop("wheel_disarmed", "right")
    notes.resume_run("right wheel re-enabled")
    again = SessionNotes(clock_ns=_clock(), state_path=path)
    it = again.intent_dict()
    assert (it["run_id"], it["stops"], it["resumes"]) == (a.run_id, 1, 1)
    assert it["last_stop"]["cause"] == "wheel_disarmed"
    assert it["last_stop"]["detail"] == "right"
    saved = json.loads(path.read_text())
    assert saved["run"]["resumes"][0]["note"] == "right wheel re-enabled"

    # a state file from before stops / resumes / unfinished existed
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"format": 1, "intent": None, "run": None, "recent": [
        {"run_id": "r20261024_100000_abcd", "text": "loop", "from": "", "to": "",
         "task": "explore", "target": "", "started_ns": 1, "ended_ns": 2,
         "outcome": "", "interruptions": 0}]}))
    row = SessionNotes(state_path=old).recent()[0]
    assert row["outcome"] == "unfinished"
    assert (row["stops"], row["resumes"], row["last_stop"]) == (0, 0, None)


def test_end_note_is_kept_on_the_run_row_while_idle(tmp_path):
    """The app's end flows (start prompt "close it", the stop sheet) run with
    no segment open; the operator's note must still reach the run row and
    the state file."""
    path = tmp_path / "intent.json"
    notes = SessionNotes(clock_ns=_clock(), state_path=path)
    a = notes.set_intent("to the couch", task="go_to_place", target="Couch")
    out = notes.end_run("arrived", " parked by the couch ")
    assert out["event_recorded"] is False
    assert out["run"]["end_note"] == "parked by the couch"
    assert notes.recent()[0]["end_note"] == "parked by the couch"
    b = notes.set_intent("loop", task="explore")
    notes.end_run("unfinished", "battery died in the hall")
    # an interruption keeps the run open and does not set end_note
    c = notes.set_intent("again", task="explore")
    notes.end_run("interrupted", "kid in the way")
    assert notes.recent()[0]["end_note"] == ""
    # a run replaced by a new one closes with no note
    notes.set_intent("new one")
    again = SessionNotes(state_path=path)
    rows = {r["run_id"]: r for r in again.recent()}
    assert rows[a.run_id]["end_note"] == "parked by the couch"
    assert rows[b.run_id]["end_note"] == "battery died in the hall"
    assert rows[c.run_id]["end_note"] == "" and rows[c.run_id]["outcome"] == "unfinished"


def test_held_resume_survives_a_restart_once(tmp_path):
    path = tmp_path / "intent.json"
    notes = SessionNotes(clock_ns=_clock(), state_path=path)
    a = notes.set_intent("loop", task="explore")
    notes.resume_run("was charging")
    # vision restarts before the drive starts
    again = SessionNotes(clock_ns=_clock(), state_path=path)
    assert again.intent().run_id == a.run_id
    assert again.snapshot()["events_held"] == 1
    again.begin_segment("seg")
    ev = again.drain_events()
    assert [(e["kind"], e["note"]) for e in ev] == [("resume", "was charging")]
    assert "run_id" not in ev[0]
    # delivered: another restart does not deliver it twice
    third = SessionNotes(clock_ns=_clock(), state_path=path)
    assert third.snapshot()["events_held"] == 0


def test_held_events_of_another_run_are_not_restored(tmp_path):
    import json
    path = tmp_path / "intent.json"
    notes = SessionNotes(clock_ns=_clock(), state_path=path)
    notes.set_intent("loop", task="explore")
    notes.resume_run("x")
    state = json.loads(path.read_text())
    state["held"][0]["run_id"] = "r_other"
    state["held"].append("garbage")
    path.write_text(json.dumps(state))
    assert SessionNotes(state_path=path).snapshot()["events_held"] == 0


def test_closed_runs_are_appended_to_the_run_log(tmp_path):
    # The recent list keeps 8 runs; the log keeps every one, with the outcome
    # picked after Stop driving (no segment recording, so no event).
    path = tmp_path / "intent.json"
    notes = SessionNotes(clock_ns=_clock(), state_path=path)
    assert notes.run_log() == []
    ids = []
    for i in range(10):
        ids.append(notes.set_intent(f"pass {i}", task="go_to_place", target="kitchen").run_id)
        notes.note_stop("disarmed")
        notes.end_run("arrived" if i % 2 == 0 else "gave_up", note=f"n{i}")
    log = notes.run_log()
    assert [r["run_id"] for r in log] == ids
    assert [r["outcome"] for r in log] == ["arrived", "gave_up"] * 5
    assert log[3]["end_note"] == "n3" and log[3]["stops"][0]["cause"] == "disarmed"
    assert len(notes.recent()) == 8                 # the recent list is capped
    # a later instance reads the same file; since filters by end time
    again = SessionNotes(state_path=path)
    assert len(again.run_log()) == 10
    assert [r["run_id"] for r in again.run_log(since_ns=log[7]["ended_ns"])] == ids[8:]
    # a line cut by a power loss is skipped, not fatal
    with open(tmp_path / "runs.jsonl", "a") as f:
        f.write('{"run_id": "cut')
    assert len(SessionNotes(state_path=path).run_log()) == 10
    # without persistence there is no log
    mem = SessionNotes()
    mem.set_intent("x", task="explore")
    mem.end_run("arrived")
    assert mem.run_log() == []


def test_run_log_recovers_after_a_torn_last_line(tmp_path):
    path = tmp_path / "intent.json"
    notes = SessionNotes(clock_ns=_clock(), state_path=path)
    notes.set_intent("a", task="explore")
    notes.end_run("arrived")
    with open(tmp_path / "runs.jsonl", "a") as f:
        f.write('{"run_id": "rtorn", "ended')        # power cut mid-append
    notes.set_intent("b", task="explore")
    notes.end_run("found")
    assert [r["outcome"] for r in notes.run_log()] == ["arrived", "found"]
