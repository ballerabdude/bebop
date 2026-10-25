"""Drive runs in the lab: the intent series' run fields, Catalog.runs()
and make_run_splits, on synthetic sessions (no robot)."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import synth
from bebop_vision import navd_proto
from bebop_vision.lab.catalog import (RUN_EVENT_SLACK_NS, RUNS_SCHEMA, UNFINISHED, Catalog,
                                      run_breaks, run_log_outcomes, run_spans)
from bebop_vision.lab.dataset import make_run_splits
from bebop_vision.lab.session import Session
from bebop_vision.lab.types import SERIES_DTYPES

S = 1_000_000_000
KW = dict(duration_s=1.0, hz=10.0, codec=None, with_runtime=False)
RUN_A = "r20261025_104233_9f3a"
RUN_B = "r20261025_104501_0b1c"


def _intent_rows(*rows):
    """Structured intent rows from ``(t_ns, run_id, text)`` tuples."""
    dt = SERIES_DTYPES["intent"]
    out = np.zeros(len(rows), dtype=dt)
    for i, (t, rid, text) in enumerate(rows):
        out[i]["t_ns"], out[i]["run_id"], out[i]["text"] = t, rid, text
    return out


def _events(*rows):
    out = np.zeros(len(rows), dtype=SERIES_DTYPES["event"])
    for i, (t, kind) in enumerate(rows):
        out[i]["t_ns"], out[i]["kind"] = t, kind
    return out


# -- Session decoding -----------------------------------------------------------

def test_intent_series_decodes_run_fields(tmp_path, lab_cache):
    started = synth.T0_NS - 5 * S
    p = synth.write_synthetic_session(
        tmp_path / "navd_session_20261025_104233.mcap",
        intent_kw=dict(task="go_to_place", target="kitchen", run_id=RUN_A, run_started_ns=started),
        extra_intents=[(0.5, dict(text="", run_id=""))], **KW)
    it = Session(p, cache_dir=lab_cache / p.stem).series("intent")
    assert it.fields == SERIES_DTYPES["intent"].names
    assert len(it) == 2
    assert it["task"][0] == "go_to_place" and it["target"][0] == "kitchen"
    assert it["run_id"][0] == RUN_A and int(it["run_started_ns"][0]) == started
    assert it["text"][0] == "hall -> kitchen" and it["to_place"][0] == "kitchen"
    assert it["run_id"][1] == "" and int(it["run_started_ns"][1]) == 0   # the clear


def test_intent_without_run_fields_decodes_empty(h265_session, lab_cache):
    """A recording made before drive runs (fields unset) reads "" / 0."""
    it = Session(h265_session, cache_dir=lab_cache / "h265").series("intent")
    assert len(it) == 1
    assert it["task"][0] == "" and it["target"][0] == "" and it["run_id"][0] == ""
    assert int(it["run_started_ns"][0]) == 0


def _old_intent_schema():
    """``bebop.navd.Intent`` as embedded before drive runs: fields 1-4 only."""
    from google.protobuf import descriptor_pb2
    name, enc, data = navd_proto.schema(navd_proto.Intent)
    fds = descriptor_pb2.FileDescriptorSet.FromString(data)
    for f in fds.file:
        for m in f.message_type:
            if f.package + "." + m.name == name:
                keep = [fld for fld in m.field if fld.number <= 4]
                del m.field[:]
                m.field.extend(keep)
    return name, enc, fds.SerializeToString()


def test_intent_from_an_old_embedded_schema(tmp_path, lab_cache):
    """The embedded schema has no task / run_id at all (an old file): the
    absent-safe decode fills "" / 0 instead of raising."""
    from mcap.writer import CompressionType, Writer
    p = tmp_path / "navd_session_20261001_000000.mcap"
    with open(p, "wb") as f:
        wr = Writer(f, compression=CompressionType.NONE)
        wr.start(profile="", library="test")
        ch_odom = wr.register_channel("/odom", "protobuf", wr.register_schema(*navd_proto.schema(navd_proto.Odom)))
        ch_int = wr.register_channel("/intent", "protobuf", wr.register_schema(*_old_intent_schema()))
        old = navd_proto.Intent(text="old drive", from_place="a", to_place="b").SerializeToString()
        wr.add_message(ch_int, synth.T0_NS, old, synth.T0_NS)
        for i in range(5):
            t = synth.T0_NS + i * 100_000_000
            wr.add_message(ch_odom, t, navd_proto.Odom(x=0.1 * i).SerializeToString(), t)
        wr.finish()
    it = Session(p, cache_dir=lab_cache / p.stem).series("intent")
    assert len(it) == 1 and it["text"][0] == "old drive" and it["from_place"][0] == "a"
    assert it["run_id"][0] == "" and it["task"][0] == "" and int(it["run_started_ns"][0]) == 0


# -- run_spans ------------------------------------------------------------------

def test_run_spans_boundaries_continue_and_outcome():
    t = 1_000 * S
    intent = _intent_rows((t, RUN_A, "to kitchen"), (t + 2 * S, RUN_A, "to kitchen, left door"),
                          (t + 5 * S, "", ""), (t + 6 * S, RUN_B, "explore"))
    events = _events((t + 3 * S, "interrupted"), (t + 5 * S, "arrived"), (t + 7 * S, "obstacle"))
    spans = run_spans(intent, events, end_ns=t + 9 * S)
    assert [s["run_id"] for s in spans] == [RUN_A, RUN_B]
    a, b = spans
    assert (a["start_ns"], a["end_ns"], a["last_ns"]) == (t, t + 5 * S, t + 2 * S)
    assert a["text"] == "to kitchen, left door"              # continue updates the fields
    assert a["outcome"] == "arrived" and a["interruptions"] == 1
    assert (b["start_ns"], b["end_ns"]) == (t + 6 * S, t + 9 * S)
    assert b["outcome"] == "" and b["interruptions"] == 0


def test_run_spans_event_on_a_boundary_ends_the_earlier_run():
    t = 1_000 * S
    intent = _intent_rows((t, RUN_A, "a"), (t + 4 * S, RUN_B, "b"))
    spans = run_spans(intent, _events((t + 4 * S, "gave_up")), end_ns=t + 8 * S)
    assert spans[0]["outcome"] == "gave_up" and spans[1]["outcome"] == ""


@pytest.mark.parametrize("dt_ns", [-3_000_000, 0, 2_000_000, 900_000_000])
def test_run_spans_outcome_near_the_clear(dt_ns):
    """The recorder stamps the clear row with the tick's log_ns and the
    outcome event with end_run's own clock, so the event can land a few ms
    before or after the clear. Either side gives the outcome to the run."""
    t = 1_000 * S
    intent = _intent_rows((t, RUN_A, "a"), (t + 5 * S, "", ""))
    ev = _events((t + 5 * S + dt_ns, "arrived"), (t + 5 * S + dt_ns, "interrupted"))
    (a,) = run_spans(intent, ev, end_ns=t + 9 * S)
    assert a["outcome"] == "arrived" and a["outcome_ns"] == t + 5 * S + dt_ns
    assert a["interruptions"] == 1


def test_run_spans_event_long_after_a_clear_is_no_runs():
    t = 1_000 * S
    intent = _intent_rows((t, RUN_A, "a"), (t + 5 * S, "", ""), (t + 8 * S, RUN_B, "b"))
    ev = _events((t + 5 * S + RUN_EVENT_SLACK_NS + 1, "found"), (t + 7 * S, "gave_up"),
                 (t + 8 * S + 1, "arrived"))
    a, b = run_spans(intent, ev, end_ns=t + 9 * S)
    assert a["outcome"] == "" and b["outcome"] == "arrived"


def test_run_spans_ignores_rows_without_run_and_empty_input():
    t = 1_000 * S
    assert run_spans(_intent_rows((t, "", "legacy text")), None, t + S) == []
    assert run_spans(np.empty(0, SERIES_DTYPES["intent"]), None, t) == []


# -- Catalog.runs ---------------------------------------------------------------

@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    """Four segments: run A over the first two (cleared in the second), run B starting in the second and going on in the third, and a
    fourth segment recorded before drive runs (text, no run id)."""
    d = tmp_path_factory.mktemp("runs")
    started = synth.T0_NS - 30 * S
    a = dict(task="go_to_place", target="kitchen", run_id=RUN_A, run_started_ns=started)
    b = dict(text="look for the sock", from_place="", to_place="", task="look_for", target="sock",
             run_id=RUN_B, run_started_ns=synth.T0_NS + 10 * S + 600_000_000)
    synth.write_synthetic_session(d / "navd_session_20261025_100000.mcap", t0_ns=synth.T0_NS,
                                  intent_kw=a, **KW)
    synth.write_synthetic_session(d / "navd_session_20261025_100010.mcap", t0_ns=synth.T0_NS + 10 * S,
                                  intent_kw=dict(a, target="kitchen table"),
                                  extra_intents=[(0.4, dict(text="", run_id="")), (0.6, b)], **KW)
    synth.write_synthetic_session(d / "navd_session_20261025_100020.mcap", t0_ns=synth.T0_NS + 20 * S,
                                  intent_kw=b, **KW)
    synth.write_synthetic_session(d / "navd_session_20261024_090000.mcap", t0_ns=synth.T0_NS - 3600 * S,
                                  **KW)
    return d


def _names(d):
    return sorted(p.stem for p in d.glob("*.mcap"))


def test_runs_skips_unindexed_unless_asked(run_root, lab_cache):
    cat = Catalog(roots=(run_root,), cache_dir=lab_cache, robot="http://127.0.0.1:9")
    df = cat.runs()
    assert df.schema == pl.Schema(RUNS_SCHEMA) and df.height == 0
    assert cat.runs_unindexed == _names(run_root)
    assert not any((lab_cache / n).exists() for n in _names(run_root))   # nothing was built
    df = cat.runs(index=True)
    assert cat.runs_unindexed == []
    assert df["run_id"].to_list() == [RUN_A, RUN_B]
    a, b = df.rows(named=True)
    assert a["sessions"] == ["navd_session_20261025_100000", "navd_session_20261025_100010"]
    assert a["n_sessions"] == 2 and a["task"] == "go_to_place"
    assert a["target"] == "kitchen table"                         # the last recorded value
    assert a["start_ns"] == synth.T0_NS - 30 * S                  # run_started_ns, before recording
    assert a["end_ns"] == synth.T0_NS + 10 * S + 400_000_000      # the clear
    assert b["sessions"] == ["navd_session_20261025_100010", "navd_session_20261025_100020"]
    assert b["task"] == "look_for" and b["target"] == "sock" and b["text"] == "look for the sock"
    assert b["start_ns"] == synth.T0_NS + 10 * S + 600_000_000
    assert b["end_ns"] == Session(run_root / "navd_session_20261025_100020.mcap",
                                  cache_dir=lab_cache / "x").end_ns
    assert b["outcome"] == UNFINISHED and b["interruptions"] == 0   # still open
    assert b["outcome_source"] == "none"
    assert a["outcome"] == UNFINISHED                             # cleared, no event (pre-2026-10-25)
    assert a["outcome_source"] == "intent"
    # a second call reads the indexes it built (cheap path), same answer
    assert cat.runs().equals(df) and cat.runs_unindexed == []


def test_runs_outcome_from_events(tmp_path, lab_cache):
    """The synthetic session's own "obstacle" event is not an outcome; an
    "arrived" event inside the run is."""
    p = synth.write_synthetic_session(tmp_path / "navd_session_20261025_110000.mcap",
                                      intent_kw=dict(run_id=RUN_A, task="explore"),
                                      extra_intents=[(0.7, dict(text="", run_id=""))], **KW)
    cat = Catalog(roots=(tmp_path,), cache_dir=lab_cache)
    row = cat.runs(index=True).row(0, named=True)
    assert row["outcome"] == UNFINISHED and row["end_ns"] == synth.T0_NS + 700_000_000
    # run_spans on the same series with an outcome event added
    sess = Session(p, cache_dir=lab_cache / p.stem)
    ev = np.concatenate([sess.series("event").data, _events((synth.T0_NS + 700_000_000, "found"))])
    spans = run_spans(sess.series("intent").data, ev, sess.end_ns)
    assert spans[0]["outcome"] == "found"


def test_runs_outcome_and_interruptions_across_sessions(tmp_path, lab_cache):
    """Run A over two recorded segments: one "interrupted" in each (summed),
    then "arrived" stamped 2 ms after the clear row (the recorder race), in
    the second. Run B starts later in the second segment and stays open."""
    a = dict(task="go_to_place", target="kitchen", run_id=RUN_A, run_started_ns=synth.T0_NS)
    b = dict(text="explore", from_place="", to_place="", task="explore", target="", run_id=RUN_B)
    synth.write_synthetic_session(tmp_path / "navd_session_20261025_120000.mcap", t0_ns=synth.T0_NS,
                                  intent_kw=a, extra_events=[(0.3, "interrupted", "")], **KW)
    synth.write_synthetic_session(tmp_path / "navd_session_20261025_120010.mcap",
                                  t0_ns=synth.T0_NS + 10 * S, intent_kw=a,
                                  extra_intents=[(0.5, dict(text="", run_id="")), (0.8, b)],
                                  extra_events=[(0.2, "interrupted", "bump"),
                                                (0.502, "arrived", "")], **KW)
    df = Catalog(roots=(tmp_path,), cache_dir=lab_cache).runs(index=True)
    assert df["run_id"].to_list() == [RUN_A, RUN_B]
    ra, rb = df.rows(named=True)
    assert ra["n_sessions"] == 2 and ra["outcome"] == "arrived" and ra["interruptions"] == 2
    assert ra["outcome_source"] == "event"
    assert ra["end_ns"] == synth.T0_NS + 10 * S + 500_000_000
    assert rb["outcome"] == UNFINISHED and rb["interruptions"] == 0 and rb["n_sessions"] == 1


# -- stops, resumes and "unfinished" ---------------------------------------------

def _notes(*rows):
    """Structured event rows from ``(t_ns, kind, note)`` tuples."""
    out = np.zeros(len(rows), dtype=SERIES_DTYPES["event"])
    for i, (t, kind, note) in enumerate(rows):
        out[i]["t_ns"], out[i]["kind"], out[i]["note"] = t, kind, note
    return out


def test_run_spans_stops_resumes_and_unfinished():
    """Stops and resumes go to the run they are in; a resume stamped just
    before a segment's first /intent row goes to the run that row carries;
    a stop detail keeps its own colons; an "unfinished" event ends a run."""
    t = 1_000 * S
    intent = _intent_rows((t, RUN_A, "a"), (t + 5 * S, "", ""), (t + 6 * S, RUN_B, "b"))
    ev = _notes((t - 2_000_000, "resume", "picked it back up"),
                (t + 2 * S, "stop", "estop_auto: watchdog: no command for 0.5 s"),
                (t + 5 * S, "unfinished", "battery"),
                (t + 7 * S, "stop", "disarmed"))
    a, b = run_spans(intent, ev, end_ns=t + 9 * S)
    assert a["outcome"] == UNFINISHED and b["outcome"] == ""
    assert a["resumes"] == [{"t_ns": t - 2_000_000, "kind": "resume", "cause": "",
                             "note": "picked it back up"}]
    assert a["stops"] == [{"t_ns": t + 2 * S, "kind": "stop", "cause": "estop_auto",
                           "note": "watchdog: no command for 0.5 s"}]
    assert [x["cause"] for x in b["stops"]] == ["disarmed"] and b["stops"][0]["note"] == ""
    assert b["resumes"] == []


def test_run_spans_resume_in_a_gap_goes_to_the_next_run():
    t = 1_000 * S
    intent = _intent_rows((t, RUN_A, "a"), (t + 5 * S, "", ""), (t + 8 * S, RUN_B, "b"))
    a, b = run_spans(intent, _notes((t + 7 * S, "resume", "")), end_ns=t + 9 * S)
    assert a["resumes"] == [] and len(b["resumes"]) == 1


def test_run_breaks_from_an_array_and_a_session(tmp_path, lab_cache, h265_session):
    t = 1_000 * S
    ev = _notes((t + 3 * S, "stop", "wheel_disarmed: left"), (t + S, "resume", "go on"),
                (t + 2 * S, "obstacle", "sock"))
    assert run_breaks(ev) == [
        {"t_ns": t + S, "kind": "resume", "cause": "", "note": "go on"},
        {"t_ns": t + 3 * S, "kind": "stop", "cause": "wheel_disarmed", "note": "left"}]
    assert run_breaks(np.empty(0, SERIES_DTYPES["event"])) == []
    # an older recording: its obstacle marker is no break
    assert run_breaks(Session(h265_session, cache_dir=lab_cache / "h265")) == []
    p = synth.write_synthetic_session(
        tmp_path / "navd_session_20261025_130000.mcap", intent_kw=dict(run_id=RUN_A),
        extra_events=[synth.resume_event(0.0, "after lunch"),
                      synth.stop_event(0.9, "estop_operator", "operator pressed E-STOP")], **KW)
    br = run_breaks(Session(p, cache_dir=lab_cache / p.stem))
    assert [(x["kind"], x["cause"], x["note"]) for x in br] == [
        ("resume", "", "after lunch"), ("stop", "estop_operator", "operator pressed E-STOP")]


def test_runs_stops_and_resumes_across_sessions(tmp_path, lab_cache):
    """Run A: E-STOP ends the first segment, the operator resumes it in the
    second (the resume stamped 1 ms before the segment's first /intent),
    then closes it as unfinished from the intent panel while recording (the
    event and the clear row), then disarms: that stop comes after the run
    closed, so it is in no span and no run's (the robot logs it on no run
    either)."""
    a = dict(task="go_to_place", target="kitchen", run_id=RUN_A, run_started_ns=synth.T0_NS)
    synth.write_synthetic_session(
        tmp_path / "navd_session_20261025_140000.mcap", t0_ns=synth.T0_NS, intent_kw=a,
        extra_events=[synth.stop_event(0.9, "estop_auto", "motor fault")], **KW)
    synth.write_synthetic_session(
        tmp_path / "navd_session_20261025_140030.mcap", t0_ns=synth.T0_NS + 30 * S, intent_kw=a,
        extra_intents=[(0.6, dict(text="", run_id=""))],
        extra_events=[synth.resume_event(-0.001, "cable moved"), (0.6, "unfinished", "battery"),
                      synth.stop_event(0.9, "disarmed")], **KW)
    df = Catalog(roots=(tmp_path,), cache_dir=lab_cache).runs(index=True)
    assert df.schema == pl.Schema(RUNS_SCHEMA)
    (row,) = df.rows(named=True)
    # "event": the explicit decline, not the no-ending default
    assert row["outcome"] == UNFINISHED and row["outcome_source"] == "event"
    assert row["n_sessions"] == 2
    assert row["stops"] == 1 and row["stop_causes"] == ["estop_auto"]
    assert row["resumes"] == 1 and row["resume_notes"] == ["cable moved"]


def test_run_spans_stop_in_no_span_is_no_runs():
    """A stop shortly after a clear is not credited to the closed run (the
    1 s slack is for outcome events only); an outcome there still is."""
    t = 1_000 * S
    intent = _intent_rows((t, RUN_A, "a"), (t + 5 * S, "", ""))
    (a,) = run_spans(intent, _notes((t + 5 * S + 400_000_000, "stop", "disarmed"),
                                    (t + 5 * S + 2_000_000, "arrived", "")), end_ns=t + 9 * S)
    assert a["stops"] == [] and a["outcome"] == "arrived" and a["closed"]


def _stop_then_sheet(tmp_path):
    """The normal end since Stop driving (2026-10-25): the disarm stop ends
    run A's last segment with the run open in the file; the operator then
    picks Arrived on the sheet, unrecorded. Run B is still open."""
    a = dict(task="go_to_place", target="kitchen", run_id=RUN_A, run_started_ns=synth.T0_NS)
    b = dict(text="explore", from_place="", to_place="", task="explore", target="", run_id=RUN_B)
    synth.write_synthetic_session(
        tmp_path / "navd_session_20261025_150000.mcap", t0_ns=synth.T0_NS, intent_kw=a,
        extra_events=[synth.stop_event(0.9, "disarmed")], **KW)
    synth.write_synthetic_session(
        tmp_path / "navd_session_20261025_150100.mcap", t0_ns=synth.T0_NS + 60 * S, intent_kw=b,
        extra_events=[synth.stop_event(0.9, "estop_auto", "watchdog")], **KW)


def test_runs_outcome_from_the_robot_run_log(tmp_path, lab_cache):
    _stop_then_sheet(tmp_path)
    cat = Catalog(roots=(tmp_path,), cache_dir=lab_cache)
    ra, rb = cat.runs(index=True).rows(named=True)
    assert (ra["outcome"], ra["outcome_source"]) == (UNFINISHED, "none")   # nothing in the files
    assert ra["stop_causes"] == ["disarmed"]
    # the robot's state file: A closed arrived, B open (``run``, and as an
    # open recent row from /notes: ended_ns 0), an older run closed with ""
    state = {"format": 1, "intent": {"run_id": RUN_B}, "run": {"run_id": RUN_B, "ended_ns": 0},
             "recent": [{"run_id": RUN_B, "ended_ns": 0, "outcome": ""},
                        {"run_id": RUN_A, "ended_ns": synth.T0_NS + 20 * S, "outcome": "arrived"},
                        {"run_id": "r20261024_090000_aaaa", "ended_ns": 5, "outcome": ""}]}
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(state))
    for log in (path, str(path), state, state["recent"]):
        ra, rb = cat.runs(run_log=log).rows(named=True)
        assert (ra["outcome"], ra["outcome_source"]) == ("arrived", "log")
        assert (rb["outcome"], rb["outcome_source"]) == (UNFINISHED, "none")   # may still resume
    assert run_log_outcomes(state)["r20261024_090000_aaaa"] == UNFINISHED
    assert run_log_outcomes(None) == {} and run_log_outcomes([]) == {}


def test_runs_recorded_ending_wins_over_the_run_log(tmp_path, lab_cache):
    synth.write_synthetic_session(
        tmp_path / "navd_session_20261025_160000.mcap", intent_kw=dict(run_id=RUN_A),
        extra_intents=[(0.5, dict(text="", run_id=""))], extra_events=[(0.5, "found", "")], **KW)
    log = [{"run_id": RUN_A, "ended_ns": 1, "outcome": "gave_up"}]
    (row,) = Catalog(roots=(tmp_path,), cache_dir=lab_cache).runs(index=True, run_log=log).rows(
        named=True)
    assert (row["outcome"], row["outcome_source"]) == ("found", "event")


def test_runs_without_breaks_have_empty_columns(run_root, lab_cache):
    df = Catalog(roots=(run_root,), cache_dir=lab_cache).runs(index=True)
    assert df["stops"].to_list() == [0, 0] and df["resumes"].to_list() == [0, 0]
    assert df["stop_causes"].to_list() == [[], []] and df["resume_notes"].to_list() == [[], []]


# -- make_run_splits ------------------------------------------------------------

def test_make_run_splits_keeps_runs_together(run_root, lab_cache):
    cat = Catalog(roots=(run_root,), cache_dir=lab_cache)
    cat.runs(index=True)
    names = _names(run_root)
    # runs A and B share the 100010 segment: one group of three, plus the legacy one
    for seed in range(6):
        sp = make_run_splits(cat, holdout=1, seed=seed)
        assert set(sp["train"]) | set(sp["val"]) == set(names)
        assert not set(sp["train"]) & set(sp["val"])
        assert len(sp["val"]) in (1, 3)
        if len(sp["val"]) == 3:
            assert sp["val_runs"] == [RUN_A, RUN_B]
        else:
            assert sp["val"] == ["navd_session_20261024_090000"] and sp["val_runs"] == []
    assert make_run_splits(cat, 1, seed=3) == make_run_splits(cat, 1, seed=3)
    explicit = make_run_splits(cat, holdout=[RUN_B])
    assert explicit["val"] == names[1:] and explicit["train"] == names[:1]
    with pytest.raises(ValueError):
        make_run_splits(cat, holdout=2)
    with pytest.raises(ValueError):
        make_run_splits(cat, holdout=["nope"])


def test_make_run_splits_from_rows_and_extra_sessions():
    rows = [{"run_id": "r1", "sessions": ["s1", "s2"]}, {"run_id": "r2", "sessions": ["s3"]},
            {"run_id": "r3", "sessions": ["s4", "s5"]}]
    df = pl.DataFrame(rows, schema={"run_id": pl.Utf8, "sessions": pl.List(pl.Utf8)})
    a = make_run_splits(rows, holdout=0.5, seed=0, sessions=["s0"])
    b = make_run_splits(df, holdout=0.5, seed=0, sessions=["s0"])
    assert a == b
    assert sorted(a["train"] + a["val"]) == ["s0", "s1", "s2", "s3", "s4", "s5"]
    for pair in (("s1", "s2"), ("s4", "s5")):
        assert (pair[0] in a["val"]) == (pair[1] in a["val"])
    assert make_run_splits(rows, holdout=0)["val"] == []
    explicit = make_run_splits(rows, holdout=["s2", "r2"])
    assert explicit["val"] == ["s1", "s2", "s3"] and explicit["val_runs"] == ["r1", "r2"]
    with pytest.raises(TypeError):
        make_run_splits(rows, holdout=True)


def test_make_run_splits_warns_about_unindexed_sessions(run_root, lab_cache):
    cat = Catalog(roots=(run_root,), cache_dir=lab_cache)   # fresh cache: nothing indexed
    with pytest.warns(RuntimeWarning, match="no lab index"):
        sp = make_run_splits(cat, holdout=1)
    assert sorted(sp["train"] + sp["val"]) == _names(run_root) and len(sp["val"]) == 1


def test_pull_runs_mirrors_the_robot_log_and_runs_reads_it(tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from bebop_vision.lab.catalog import Catalog, run_log_outcomes
    rows = [{"run_id": "r20261025_100000_aaaa", "ended_ns": 5, "outcome": "arrived", "end_note": "dresser"},
            {"run_id": "r20261025_101000_bbbb", "ended_ns": 9, "outcome": "", "end_note": ""}]

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"runs": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cat = Catalog(roots=(tmp_path,))
        assert cat.run_log_path() is None
        out = cat.pull_runs(url=f"http://127.0.0.1:{srv.server_address[1]}/runs")
    finally:
        srv.shutdown()
    assert out == tmp_path / "runs.jsonl" and cat.run_log_path() == out
    with open(out, "a") as f:
        f.write('{"run_id": "cut')                  # a truncated last line
    assert run_log_outcomes(out) == {"r20261025_100000_aaaa": "arrived",
                                     "r20261025_101000_bbbb": "unfinished"}


def test_pull_runs_merges_and_never_shrinks_the_mirror(tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from bebop_vision.lab.catalog import Catalog, run_log_outcomes
    answers = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(answers.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/runs"
    cat = Catalog(roots=(tmp_path,))
    a = {"run_id": "ra", "ended_ns": 5, "outcome": "arrived"}
    b1 = {"run_id": "rb", "ended_ns": 7, "outcome": "arrived"}   # before a crash
    b2 = {"run_id": "rb", "ended_ns": 9, "outcome": "gave_up"}   # re-closed after it
    try:
        answers.extend([{"runs": [a, b1]}, {"runs": []}, {"runs": [b2]}, [1, 2]])
        cat.pull_runs(url=url)
        cat.pull_runs(url=url)                        # an empty answer keeps history
        assert run_log_outcomes(cat.run_log_path()) == {"ra": "arrived", "rb": "arrived"}
        cat.pull_runs(url=url)                        # the later close wins
        assert run_log_outcomes(cat.run_log_path()) == {"ra": "arrived", "rb": "gave_up"}
        with pytest.raises(ValueError):
            cat.pull_runs(url=url)                    # not {"runs": [...]}
    finally:
        srv.shutdown()
    # order-independent: rows oldest-first or newest-first give the same answer
    assert run_log_outcomes([b2, b1]) == run_log_outcomes([b1, b2]) == {"rb": "gave_up"}
